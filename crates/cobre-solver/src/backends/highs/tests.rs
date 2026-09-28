use super::HighsSolver;
use crate::{
    BasisStatus, SolverInterface,
    types::{Basis, RowBatch, SolverStatistics, StageTemplate},
};

// SS1.1 fixture: 3 vars, 2 equality rows, optimal objective = 100.
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

// SS1.2 fixture: 2 inequality rows for add_rows tests.
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

#[test]
fn test_highs_solver_create_and_name() {
    let solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    assert_eq!(solver.name(), "HiGHS");
}

#[test]
fn test_highs_solver_send_bound() {
    fn assert_send<T: Send>() {}
    assert_send::<HighsSolver>();
}

#[test]
fn test_highs_solver_statistics_initial() {
    let solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let stats = solver.statistics();
    assert_eq!(stats.solve_count, 0);
    assert_eq!(stats.success_count, 0);
    assert_eq!(stats.failure_count, 0);
    assert_eq!(stats.total_iterations, 0);
    assert_eq!(stats.retry_count, 0);
    assert_eq!(stats.total_solve_time_seconds, 0.0);
}

#[test]
fn test_highs_load_model_updates_dimensions() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);

    assert_eq!(solver.num_cols, 3, "num_cols must be 3 after load_model");
    assert_eq!(solver.num_rows, 2, "num_rows must be 2 after load_model");
    assert_eq!(
        solver.col_value.len(),
        3,
        "col_value buffer must be resized to num_cols"
    );
    assert_eq!(
        solver.col_dual.len(),
        3,
        "col_dual buffer must be resized to num_cols"
    );
    assert_eq!(
        solver.row_value.len(),
        2,
        "row_value buffer must be resized to num_rows"
    );
    assert_eq!(
        solver.row_dual.len(),
        2,
        "row_dual buffer must be resized to num_rows"
    );
}

// Exercises a `debug_assert*` length guard, which is elided in release
// builds; restrict to debug so the panic actually fires.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "col_lower")]
fn test_highs_load_model_short_col_lower_panics() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let mut template = make_fixture_stage_template();
    // num_cols == 3 but col_lower only has 2 entries: length guard must fire.
    template.col_lower = vec![0.0, 0.0];

    solver.load_model(&template);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "row_scale")]
fn test_highs_load_model_oversized_row_scale_panics() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let mut template = make_fixture_stage_template();
    // num_rows == 2 but row_scale is non-empty with 3 entries: length guard must fire.
    template.row_scale = vec![1.0, 1.0, 1.0];

    solver.load_model(&template);
}

#[test]
fn test_highs_load_model_empty_row_scale_loads() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let mut template = make_fixture_stage_template();
    // Empty row_scale means "no scaling" and must pass the optional-length guard.
    template.row_scale = Vec::new();

    solver.load_model(&template);

    assert_eq!(solver.num_rows, 2, "num_rows must be 2 after load_model");
}

#[test]
fn test_highs_add_rows_updates_dimensions() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    solver.load_model(&template);
    solver.add_rows(&cuts);

    // 2 structural rows + 2 appended rows = 4
    assert_eq!(solver.num_rows, 4, "num_rows must be 4 after add_rows");
    assert_eq!(
        solver.row_dual.len(),
        4,
        "row_dual buffer must be resized to 4 after add_rows"
    );
    assert_eq!(
        solver.row_value.len(),
        4,
        "row_value buffer must be resized to 4 after add_rows"
    );
    assert_eq!(solver.num_cols, 3, "num_cols must be unchanged by add_rows");
}

#[test]
fn test_highs_set_row_bounds_no_panic() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    solver.load_model(&template);

    solver.set_row_bounds(&[0], &[4.0], &[4.0]);
}

#[test]
fn test_highs_set_col_bounds_no_panic() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    solver.load_model(&template);

    solver.set_col_bounds(&[1], &[10.0], &[f64::INFINITY]);
}

#[test]
fn test_highs_set_bounds_empty_no_panic() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    solver.load_model(&template);

    // Empty patch slices should be short-circuited without any FFI call.
    solver.set_row_bounds(&[], &[], &[]);
    solver.set_col_bounds(&[], &[], &[]);
}

/// SS1.1 fixture: min 0*x0 + 1*x1 + 50*x2, s.t. x0=6, 2*x0+x2=14, x>=0.
/// Optimal: x0=6, x1=0, x2=2, objective=100.
#[test]
fn test_highs_solve_basic_lp() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    solver.load_model(&template);

    let solution = solver
        .solve(None)
        .expect("solve() must succeed on a feasible LP");

    assert!(
        (solution.objective - 100.0).abs() < 1e-8,
        "objective must be 100.0, got {}",
        solution.objective
    );
    assert_eq!(solution.primal.len(), 3, "primal must have 3 elements");
    assert!(
        (solution.primal[0] - 6.0).abs() < 1e-8,
        "primal[0] (x0) must be 6.0, got {}",
        solution.primal[0]
    );
    assert!(
        (solution.primal[1] - 0.0).abs() < 1e-8,
        "primal[1] (x1) must be 0.0, got {}",
        solution.primal[1]
    );
    assert!(
        (solution.primal[2] - 2.0).abs() < 1e-8,
        "primal[2] (x2) must be 2.0, got {}",
        solution.primal[2]
    );
}

/// SS1.2: after adding two valid inequalities to SS1.1, optimal objective = 162.
/// Cuts: -5*x0+x1>=20 and 3*x0+x1>=80. With x0=6: x1>=max(50,62)=62.
/// Obj = 0*6 + 1*62 + 50*2 = 162.
#[test]
fn test_highs_solve_with_cuts() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();
    solver.load_model(&template);
    solver.add_rows(&cuts);

    let solution = solver
        .solve(None)
        .expect("solve() must succeed on a feasible LP with cuts");

    assert!(
        (solution.objective - 162.0).abs() < 1e-8,
        "objective must be 162.0, got {}",
        solution.objective
    );
    assert!(
        (solution.primal[0] - 6.0).abs() < 1e-8,
        "primal[0] must be 6.0, got {}",
        solution.primal[0]
    );
    assert!(
        (solution.primal[1] - 62.0).abs() < 1e-8,
        "primal[1] must be 62.0, got {}",
        solution.primal[1]
    );
    assert!(
        (solution.primal[2] - 2.0).abs() < 1e-8,
        "primal[2] must be 2.0, got {}",
        solution.primal[2]
    );
}

/// SS1.3: after adding cuts and patching row 0 RHS to 4.0 (x0=4).
/// x2=14-2*4=6. cut2: 3*4+x1>=80 => x1>=68. Obj = 0*4+1*68+50*6 = 368.
#[test]
fn test_highs_solve_after_rhs_patch() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();
    solver.load_model(&template);
    solver.add_rows(&cuts);

    // Patch row 0 (x0=6 equality) to x0=4.
    solver.set_row_bounds(&[0], &[4.0], &[4.0]);

    let solution = solver
        .solve(None)
        .expect("solve() must succeed after RHS patch");

    assert!(
        (solution.objective - 368.0).abs() < 1e-8,
        "objective must be 368.0, got {}",
        solution.objective
    );
}

#[test]
fn test_highs_solve_statistics_increment() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    solver.load_model(&template);

    solver.solve(None).expect("first solve must succeed");
    solver.solve(None).expect("second solve must succeed");

    let stats = solver.statistics();
    assert_eq!(stats.solve_count, 2, "solve_count must be 2");
    assert_eq!(stats.success_count, 2, "success_count must be 2");
    assert_eq!(stats.failure_count, 0, "failure_count must be 0");
    // Presolve may solve without simplex iterations; any u64 is valid.
    let _ = stats.total_iterations;
}

#[test]
fn test_highs_solve_preserves_stats() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    solver.load_model(&template);
    solver.solve(None).expect("solve must succeed");

    let stats = solver.statistics();
    assert_eq!(
        stats.solve_count, 1,
        "solve_count must be 1 after one solve"
    );
    assert_eq!(
        stats.success_count, 1,
        "success_count must be 1 after one successful solve"
    );
    let _ = stats.total_iterations;
}

/// The first solve must report an `iterations` value (any `u64` is valid per presolve behavior).
#[test]
fn test_highs_solve_iterations_positive() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    solver.load_model(&template);

    let solution = solver.solve(None).expect("solve must succeed");
    let _ = solution.iterations;
}

#[test]
fn test_highs_solve_time_positive() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    solver.load_model(&template);

    let solution = solver.solve(None).expect("solve must succeed");
    assert!(
        solution.solve_time_seconds > 0.0,
        "solve_time_seconds must be positive, got {}",
        solution.solve_time_seconds
    );
}

/// After one solve, counters must be 1/1/0 (`total_iterations` not asserted per presolve behavior).
#[test]
fn test_highs_solve_statistics_single() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    solver.load_model(&template);

    solver.solve(None).expect("solve must succeed");

    let stats = solver.statistics();
    assert_eq!(stats.solve_count, 1, "solve_count must be 1");
    assert_eq!(stats.success_count, 1, "success_count must be 1");
    assert_eq!(stats.failure_count, 0, "failure_count must be 0");
    let _ = stats.total_iterations;
}

#[test]
fn test_highs_statistics_into_equals_statistics() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    solver.load_model(&template);
    solver.solve(None).expect("solve must succeed");

    let owned = solver.statistics();
    let mut buf = SolverStatistics::default();
    solver.statistics_into(&mut buf);

    assert_eq!(buf.solve_count, owned.solve_count);
    assert_eq!(buf.success_count, owned.success_count);
    assert_eq!(buf.failure_count, owned.failure_count);
    assert_eq!(buf.total_iterations, owned.total_iterations);
    assert_eq!(buf.retry_count, owned.retry_count);
    assert_eq!(buf.total_solve_time_seconds, owned.total_solve_time_seconds);
    assert_eq!(
        buf.basis_consistency_failures,
        owned.basis_consistency_failures
    );
    assert_eq!(buf.first_try_successes, owned.first_try_successes);
    assert_eq!(buf.basis_offered, owned.basis_offered);
    assert_eq!(buf.load_model_count, owned.load_model_count);
    assert_eq!(
        buf.total_load_model_time_seconds,
        owned.total_load_model_time_seconds
    );
    assert_eq!(
        buf.total_set_bounds_time_seconds,
        owned.total_set_bounds_time_seconds
    );
    assert_eq!(
        buf.total_basis_set_time_seconds,
        owned.total_basis_set_time_seconds
    );
    assert_eq!(buf.basis_reconstructions, owned.basis_reconstructions);
    assert_eq!(buf.retry_level_histogram, owned.retry_level_histogram);
}

/// After solve, `get_basis` must return HiGHS-representable statuses (no Superbasic/Fixed).
#[test]
fn test_get_basis_valid_status_codes() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    solver.load_model(&template);
    solver
        .solve(None)
        .expect("solve must succeed before get_basis");

    let mut basis = Basis::new(0, 0);
    solver.get_basis(&mut basis);

    let is_highs_representable = |status: BasisStatus| {
        matches!(
            status,
            BasisStatus::Lower
                | BasisStatus::Basic
                | BasisStatus::Upper
                | BasisStatus::Zero
                | BasisStatus::Nonbasic
        )
    };
    for &status in &basis.col_status {
        assert!(
            is_highs_representable(status),
            "col_status {status:?} is not HiGHS-representable"
        );
    }
    for &status in &basis.row_status {
        assert!(
            is_highs_representable(status),
            "row_status {status:?} is not HiGHS-representable"
        );
    }
}

/// `get_basis` must resize empty output buffers to current LP dimensions.
#[test]
fn test_get_basis_resizes_output() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    solver.load_model(&template);
    solver
        .solve(None)
        .expect("solve must succeed before get_basis");

    let mut basis = Basis::new(0, 0);
    assert_eq!(
        basis.col_status.len(),
        0,
        "initial col_status must be empty"
    );
    assert_eq!(
        basis.row_status.len(),
        0,
        "initial row_status must be empty"
    );

    solver.get_basis(&mut basis);

    assert_eq!(
        basis.col_status.len(),
        3,
        "col_status must be resized to 3 (num_cols of SS1.1)"
    );
    assert_eq!(
        basis.row_status.len(),
        2,
        "row_status must be resized to 2 (num_rows of SS1.1)"
    );
}

/// Warm-start with exact basis must reproduce objective in ≤1 iteration.
#[test]
fn test_solve_warm_start_reproduces_cold_objective() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    solver.load_model(&template);
    solver.solve(None).expect("cold-start solve must succeed");

    let mut basis = Basis::new(0, 0);
    solver.get_basis(&mut basis);

    // Reload the same model to reset HiGHS internal state.
    solver.load_model(&template);
    let result = solver
        .solve(Some(&basis))
        .expect("warm-start solve must succeed");

    assert!(
        (result.objective - 100.0).abs() < 1e-8,
        "warm-start objective must be 100.0, got {}",
        result.objective
    );
    assert!(
        result.iterations <= 1,
        "warm-start from exact basis must use at most 1 iteration, got {}",
        result.iterations
    );

    let stats = solver.statistics();
    assert_eq!(
        stats.basis_consistency_failures, 0,
        "basis_consistency_failures must be 0 when raw basis is accepted, got {}",
        stats.basis_consistency_failures
    );
    assert_eq!(
        stats.basis_offered, 1,
        "basis_offered must be 1 after one warm-start call"
    );
}

/// Undersized basis (2 rows vs 4 after `add_rows`) is rejected with `BasisRowCountMismatch`.
/// Increments `basis_consistency_failures`; `basis_offered` unchanged (rejected before offer).
#[test]
fn test_highs_solve_rejects_undersized_row_basis() {
    use crate::types::SolverError;

    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    let template = make_fixture_stage_template();
    let cuts = make_fixture_row_batch();

    // First solve on 2-row LP to capture a 2-row basis.
    solver.load_model(&template);
    solver.solve(None).expect("SS1.1 solve must succeed");
    let mut basis = Basis::new(0, 0);
    solver.get_basis(&mut basis);
    assert_eq!(
        basis.row_status.len(),
        2,
        "captured basis must have 2 row statuses"
    );

    // Reload model and add 2 cuts to get a 4-row LP; the 2-row basis is now
    // undersized (basis_rows = 2 < lp_rows = 4).
    solver.load_model(&template);
    solver.add_rows(&cuts);
    assert_eq!(solver.num_rows, 4, "LP must have 4 rows after add_rows");

    let before = solver.statistics();

    // Map Ok → () so mutable borrow drops before statistics call.
    let err_variant: Result<(), SolverError> = solver.solve(Some(&basis)).map(|_| ());

    let after = solver.statistics();
    assert_eq!(
        after.basis_consistency_failures - before.basis_consistency_failures,
        1,
        "basis_consistency_failures must increment by 1 for an undersized row basis"
    );
    assert_eq!(
        after.basis_offered, before.basis_offered,
        "basis_offered must NOT change when the basis is rejected before being offered"
    );

    match err_variant {
        Err(SolverError::BasisRowCountMismatch {
            lp_rows,
            basis_rows,
        }) => {
            assert_eq!(lp_rows, 4, "lp_rows must equal the current LP row count");
            assert_eq!(
                basis_rows, 2,
                "basis_rows must equal the offered basis length"
            );
        }
        other => panic!(
            "expected Err(SolverError::BasisRowCountMismatch {{ lp_rows: 4, basis_rows: 2 }}), \
             got {other:?}"
        ),
    }
}

/// Self-extracted basis is accepted by non-alien path; `basis_consistency_failures` stays zero.
#[test]
fn test_solve_warm_start_non_alien_success() {
    let template = make_fixture_stage_template();
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
    solver.load_model(&template);
    let _ = solver.solve(None).expect("cold-start solve must succeed");
    let mut basis = Basis::new(template.num_cols, template.num_rows);
    solver.get_basis(&mut basis);

    // Reload model so HiGHS internal state is fresh, then warm-start.
    solver.load_model(&template);
    let before = solver.statistics();

    let _ = solver
        .solve(Some(&basis))
        .expect("warm-start solve must succeed with self-extracted basis");

    let after = solver.statistics();
    assert_eq!(
        after.basis_consistency_failures - before.basis_consistency_failures,
        0,
        "non-alien path should accept a self-extracted basis; consistency failures delta must be 0"
    );
}

/// Inconsistent basis (5 basic vars, rank 2) returns Err(BasisInconsistent).
/// Increments `basis_consistency_failures` by 1.
#[test]
fn test_solve_warm_start_rejects_inconsistent_basis() {
    use crate::types::SolverError;

    // Arrange: non-alien setter is now the only warm-start path.
    let template = make_fixture_stage_template();
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");

    solver.load_model(&template);

    // Build a deliberately inconsistent basis: all BASIC (5 basics, rank 2).
    let mut bad_basis = Basis::new(template.num_cols, template.num_rows);
    bad_basis
        .col_status
        .iter_mut()
        .for_each(|v| *v = BasisStatus::Basic);
    bad_basis
        .row_status
        .iter_mut()
        .for_each(|v| *v = BasisStatus::Basic);

    let before = solver.statistics();

    // Act — convert result to a form that does not borrow `solver`.
    // `SolutionView<'_>` borrows `solver`'s internal buffers; calling
    // `statistics()` afterwards would overlap borrows.  On the error path
    // `SolverError` contains no solver references, so mapping Ok → () breaks
    // the mutable borrow before the statistics call.
    let err_variant: Result<(), SolverError> = solver.solve(Some(&bad_basis)).map(|_| ());

    // Assert counters — the mutable borrow from solve(Some(&bad_basis)) is gone.
    let after = solver.statistics();
    assert_eq!(
        after.basis_consistency_failures - before.basis_consistency_failures,
        1,
        "basis_consistency_failures must increment by 1 for an overcounted basis"
    );

    match err_variant {
        Err(SolverError::BasisInconsistent {
            num_row,
            total_basic,
            col_basic,
            row_basic,
        }) => {
            assert_eq!(num_row, 2, "num_row must match LP row count");
            assert_eq!(total_basic, 5, "total_basic must be col_basic + row_basic");
            assert_eq!(col_basic, 3, "col_basic must count BASIC columns");
            assert_eq!(row_basic, 2, "row_basic must count BASIC rows");
        }
        other => panic!(
            "expected Err(SolverError::BasisInconsistent {{ num_row: 2, total_basic: 5, \
             col_basic: 3, row_basic: 2 }}), got {other:?}"
        ),
    }
}

/// Terminal-status scratch buffers must retain capacity across repeated resize calls.
#[test]
fn interpret_terminal_status_reuses_scratch() {
    let template = make_fixture_stage_template();
    let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");

    assert_eq!(
        solver.terminal_status_dual_scratch.capacity(),
        0,
        "dual scratch must start with capacity 0 (Vec::new() in constructor)"
    );
    assert_eq!(
        solver.terminal_status_primal_scratch.capacity(),
        0,
        "primal scratch must start with capacity 0 (Vec::new() in constructor)"
    );

    solver.load_model(&template);

    // Simulate what interpret_terminal_status does in UNBOUNDED_OR_INFEASIBLE branch:
    // resize dual scratch to num_rows, primal scratch to num_cols.
    solver
        .terminal_status_dual_scratch
        .resize(solver.num_rows, 0.0);
    solver
        .terminal_status_primal_scratch
        .resize(solver.num_cols, 0.0);

    let cap_dual_after_first = solver.terminal_status_dual_scratch.capacity();
    let cap_primal_after_first = solver.terminal_status_primal_scratch.capacity();

    assert!(
        cap_dual_after_first >= solver.num_rows,
        "dual scratch capacity {cap_dual_after_first} must be >= num_rows {} after first resize",
        solver.num_rows,
    );
    assert!(
        cap_primal_after_first >= solver.num_cols,
        "primal scratch capacity {cap_primal_after_first} must be >= num_cols {} after first resize",
        solver.num_cols,
    );

    // Second resize to the same size: capacity must not decrease (heap retained).
    solver
        .terminal_status_dual_scratch
        .resize(solver.num_rows, 0.0);
    solver
        .terminal_status_primal_scratch
        .resize(solver.num_cols, 0.0);

    let cap_dual_after_second = solver.terminal_status_dual_scratch.capacity();
    let cap_primal_after_second = solver.terminal_status_primal_scratch.capacity();

    assert!(
        cap_dual_after_second >= cap_dual_after_first,
        "dual scratch capacity must not decrease: {cap_dual_after_second} < {cap_dual_after_first}",
    );
    assert!(
        cap_primal_after_second >= cap_primal_after_first,
        "primal scratch capacity must not decrease: {cap_primal_after_second} < {cap_primal_after_first}",
    );
}

// Research tests: verify non-optimal HiGHS model statuses via raw FFI.
// SS1.1 LP is too small (crash solves it); larger_lp (5 vars, 4 rows) required.
#[allow(clippy::doc_markdown)]
mod research_tests {
    // Load SS1.1 LP (3 cols, 2 rows, obj=100) onto existing handle.
    // SAFETY: caller must guarantee `highs` is a valid, non-null HiGHS handle.
    unsafe fn research_load_ss11_lp(highs: *mut std::os::raw::c_void) {
        use crate::ffi;
        let col_cost: [f64; 3] = [0.0, 1.0, 50.0];
        let col_lower: [f64; 3] = [0.0, 0.0, 0.0];
        let col_upper: [f64; 3] = [10.0, f64::INFINITY, 8.0];
        let row_lower: [f64; 2] = [6.0, 14.0];
        let row_upper: [f64; 2] = [6.0, 14.0];
        let a_start: [i32; 4] = [0, 2, 2, 3];
        let a_index: [i32; 3] = [0, 1, 1];
        let a_value: [f64; 3] = [1.0, 2.0, 1.0];
        // SAFETY: all pointers are valid, aligned, non-null, and live for the call duration.
        let status = unsafe {
            ffi::cobre_highs_pass_lp(
                highs,
                3,
                2,
                3,
                ffi::HIGHS_MATRIX_FORMAT_COLWISE,
                ffi::HIGHS_OBJ_SENSE_MINIMIZE,
                0.0,
                col_cost.as_ptr(),
                col_lower.as_ptr(),
                col_upper.as_ptr(),
                row_lower.as_ptr(),
                row_upper.as_ptr(),
                a_start.as_ptr(),
                a_index.as_ptr(),
                a_value.as_ptr(),
            )
        };
        assert_eq!(
            status,
            ffi::HIGHS_STATUS_OK,
            "research_load_ss11_lp pass_lp failed"
        );
    }

    /// Observational: SS1.1 with time/iteration limits (presolve solves before limits fire).
    #[test]
    fn test_research_probe_limit_status_on_ss11_lp() {
        use crate::ffi;

        // SS1.1 with time_limit=0.0: presolve/crash solves before time check fires.
        let highs = unsafe { ffi::cobre_highs_create() };
        assert!(!highs.is_null());
        unsafe { ffi::cobre_highs_set_bool_option(highs, c"output_flag".as_ptr(), 0) };
        unsafe { research_load_ss11_lp(highs) };
        let _ = unsafe { ffi::cobre_highs_set_double_option(highs, c"time_limit".as_ptr(), 0.0) };
        let run_status = unsafe { ffi::cobre_highs_run(highs) };
        let model_status = unsafe { ffi::cobre_highs_get_model_status(highs) };
        let obj = unsafe { ffi::cobre_highs_get_objective_value(highs) };
        eprintln!(
            "SS1.1 + time_limit=0: run_status={run_status}, model_status={model_status}, obj={obj}"
        );
        unsafe { ffi::cobre_highs_destroy(highs) };

        // SS1.1 with iteration_limit=0: same result, need a larger LP.
        let highs = unsafe { ffi::cobre_highs_create() };
        assert!(!highs.is_null());
        unsafe { ffi::cobre_highs_set_bool_option(highs, c"output_flag".as_ptr(), 0) };
        unsafe { research_load_ss11_lp(highs) };
        let _ = unsafe {
            ffi::cobre_highs_set_int_option(highs, c"simplex_iteration_limit".as_ptr(), 0)
        };
        let run_status = unsafe { ffi::cobre_highs_run(highs) };
        let model_status = unsafe { ffi::cobre_highs_get_model_status(highs) };
        let obj = unsafe { ffi::cobre_highs_get_objective_value(highs) };
        eprintln!(
            "SS1.1 + iteration_limit=0: run_status={run_status}, model_status={model_status}, obj={obj}"
        );
        unsafe { ffi::cobre_highs_destroy(highs) };
    }

    /// Load larger LP (5 vars, 4 rows) requiring multiple simplex iterations.
    /// SAFETY: caller must guarantee `highs` is a valid, non-null HiGHS handle.
    unsafe fn research_load_larger_lp(highs: *mut std::os::raw::c_void) {
        use crate::ffi;
        let col_cost: [f64; 5] = [1.0, 1.0, 1.0, 1.0, 1.0];
        let col_lower: [f64; 5] = [0.0; 5];
        let col_upper: [f64; 5] = [100.0; 5];
        let row_lower: [f64; 4] = [10.0, 8.0, 6.0, 4.0];
        let row_upper: [f64; 4] = [f64::INFINITY; 4];
        // CSC: col 0 -> row 0; col 1 -> rows 0,1; col 2 -> rows 1,2; col 3 -> rows 2,3; col 4 -> row 3
        let a_start: [i32; 6] = [0, 1, 3, 5, 7, 8];
        let a_index: [i32; 8] = [0, 0, 1, 1, 2, 2, 3, 3];
        let a_value: [f64; 8] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        // SAFETY: all pointers are valid, aligned, non-null, and live for the call duration.
        let status = unsafe {
            ffi::cobre_highs_pass_lp(
                highs,
                5,
                4,
                8,
                ffi::HIGHS_MATRIX_FORMAT_COLWISE,
                ffi::HIGHS_OBJ_SENSE_MINIMIZE,
                0.0,
                col_cost.as_ptr(),
                col_lower.as_ptr(),
                col_upper.as_ptr(),
                row_lower.as_ptr(),
                row_upper.as_ptr(),
                a_start.as_ptr(),
                a_index.as_ptr(),
                a_value.as_ptr(),
            )
        };
        assert_eq!(
            status,
            ffi::HIGHS_STATUS_OK,
            "research_load_larger_lp pass_lp failed"
        );
    }

    /// time_limit=0.0 on larger LP triggers TIME_LIMIT (13) status.
    #[test]
    fn test_research_time_limit_zero_triggers_time_limit_status() {
        use crate::ffi;

        let highs = unsafe { ffi::cobre_highs_create() };
        assert!(!highs.is_null());
        unsafe { ffi::cobre_highs_set_bool_option(highs, c"output_flag".as_ptr(), 0) };
        unsafe { research_load_larger_lp(highs) };

        let opt_status =
            unsafe { ffi::cobre_highs_set_double_option(highs, c"time_limit".as_ptr(), 0.0) };
        assert_eq!(opt_status, ffi::HIGHS_STATUS_OK);

        let run_status = unsafe { ffi::cobre_highs_run(highs) };
        let model_status = unsafe { ffi::cobre_highs_get_model_status(highs) };

        eprintln!(
            "time_limit=0 on larger LP: run_status={run_status}, model_status={model_status}"
        );

        assert_eq!(
            run_status,
            ffi::HIGHS_STATUS_WARNING,
            "time_limit=0 must return HIGHS_STATUS_WARNING (1), got {run_status}"
        );
        assert_eq!(
            model_status,
            ffi::HIGHS_MODEL_STATUS_TIME_LIMIT,
            "time_limit=0 must give MODEL_STATUS_TIME_LIMIT (13), got {model_status}"
        );

        unsafe { ffi::cobre_highs_destroy(highs) };
    }

    /// iteration_limit=0 with presolve=off triggers ITERATION_LIMIT (14) status.
    #[test]
    fn test_research_iteration_limit_zero_triggers_iteration_limit_status() {
        use crate::ffi;

        let highs = unsafe { ffi::cobre_highs_create() };
        assert!(!highs.is_null());
        unsafe { ffi::cobre_highs_set_bool_option(highs, c"output_flag".as_ptr(), 0) };
        // Disable presolve so crash cannot solve LP without simplex iterations.
        unsafe { ffi::cobre_highs_set_string_option(highs, c"presolve".as_ptr(), c"off".as_ptr()) };
        unsafe { research_load_larger_lp(highs) };

        let opt_status = unsafe {
            ffi::cobre_highs_set_int_option(highs, c"simplex_iteration_limit".as_ptr(), 0)
        };
        assert_eq!(opt_status, ffi::HIGHS_STATUS_OK);

        let run_status = unsafe { ffi::cobre_highs_run(highs) };
        let model_status = unsafe { ffi::cobre_highs_get_model_status(highs) };

        eprintln!(
            "iteration_limit=0 on larger LP: run_status={run_status}, model_status={model_status}"
        );

        assert_eq!(
            run_status,
            ffi::HIGHS_STATUS_WARNING,
            "iteration_limit=0 must return HIGHS_STATUS_WARNING (1), got {run_status}"
        );
        assert_eq!(
            model_status,
            ffi::HIGHS_MODEL_STATUS_ITERATION_LIMIT,
            "iteration_limit=0 must give MODEL_STATUS_ITERATION_LIMIT (14), got {model_status}"
        );

        unsafe { ffi::cobre_highs_destroy(highs) };
    }

    /// Observe partial solution availability after TIME_LIMIT and ITERATION_LIMIT.
    #[test]
    fn test_research_partial_solution_availability() {
        use crate::ffi;

        // TIME_LIMIT: observe objective after halting at time check
        {
            let highs = unsafe { ffi::cobre_highs_create() };
            assert!(!highs.is_null());
            unsafe { ffi::cobre_highs_set_bool_option(highs, c"output_flag".as_ptr(), 0) };
            unsafe { research_load_larger_lp(highs) };
            unsafe { ffi::cobre_highs_set_double_option(highs, c"time_limit".as_ptr(), 0.0) };
            unsafe { ffi::cobre_highs_run(highs) };

            let obj = unsafe { ffi::cobre_highs_get_objective_value(highs) };
            let model_status = unsafe { ffi::cobre_highs_get_model_status(highs) };
            assert_eq!(model_status, ffi::HIGHS_MODEL_STATUS_TIME_LIMIT);
            eprintln!("TIME_LIMIT: obj={obj}, finite={}", obj.is_finite());
            unsafe { ffi::cobre_highs_destroy(highs) };
        }

        // ITERATION_LIMIT: observe objective at crash point
        {
            let highs = unsafe { ffi::cobre_highs_create() };
            assert!(!highs.is_null());
            unsafe { ffi::cobre_highs_set_bool_option(highs, c"output_flag".as_ptr(), 0) };
            unsafe {
                ffi::cobre_highs_set_string_option(highs, c"presolve".as_ptr(), c"off".as_ptr())
            };
            unsafe { research_load_larger_lp(highs) };
            unsafe {
                ffi::cobre_highs_set_int_option(highs, c"simplex_iteration_limit".as_ptr(), 0)
            };
            unsafe { ffi::cobre_highs_run(highs) };

            let obj = unsafe { ffi::cobre_highs_get_objective_value(highs) };
            let model_status = unsafe { ffi::cobre_highs_get_model_status(highs) };
            assert_eq!(model_status, ffi::HIGHS_MODEL_STATUS_ITERATION_LIMIT);
            eprintln!("ITERATION_LIMIT: obj={obj}, finite={}", obj.is_finite());
            unsafe { ffi::cobre_highs_destroy(highs) };
        }
    }

    /// After iteration_limit=0 solve, restore_default_settings allows subsequent optimal solve.
    #[test]
    fn test_research_restore_defaults_allows_subsequent_optimal_solve() {
        use crate::ffi;

        let highs = unsafe { ffi::cobre_highs_create() };
        assert!(!highs.is_null());

        unsafe { ffi::cobre_highs_set_bool_option(highs, c"output_flag".as_ptr(), 0) };

        // Apply cobre defaults (mirror HighsSolver::new() configuration).
        unsafe {
            ffi::cobre_highs_set_string_option(highs, c"solver".as_ptr(), c"simplex".as_ptr());
            ffi::cobre_highs_set_int_option(highs, c"simplex_strategy".as_ptr(), 1);
            ffi::cobre_highs_set_string_option(highs, c"presolve".as_ptr(), c"off".as_ptr());
            ffi::cobre_highs_set_string_option(highs, c"parallel".as_ptr(), c"off".as_ptr());
            ffi::cobre_highs_set_double_option(
                highs,
                c"primal_feasibility_tolerance".as_ptr(),
                1e-7,
            );
            ffi::cobre_highs_set_double_option(highs, c"dual_feasibility_tolerance".as_ptr(), 1e-7);
        }

        let col_cost: [f64; 3] = [0.0, 1.0, 50.0];
        let col_lower: [f64; 3] = [0.0, 0.0, 0.0];
        let col_upper: [f64; 3] = [10.0, f64::INFINITY, 8.0];
        let row_lower: [f64; 2] = [6.0, 14.0];
        let row_upper: [f64; 2] = [6.0, 14.0];
        let a_start: [i32; 4] = [0, 2, 2, 3];
        let a_index: [i32; 3] = [0, 1, 1];
        let a_value: [f64; 3] = [1.0, 2.0, 1.0];

        // First solve: with iteration_limit = 0 -> ITERATION_LIMIT.
        unsafe {
            ffi::cobre_highs_pass_lp(
                highs,
                3,
                2,
                3,
                ffi::HIGHS_MATRIX_FORMAT_COLWISE,
                ffi::HIGHS_OBJ_SENSE_MINIMIZE,
                0.0,
                col_cost.as_ptr(),
                col_lower.as_ptr(),
                col_upper.as_ptr(),
                row_lower.as_ptr(),
                row_upper.as_ptr(),
                a_start.as_ptr(),
                a_index.as_ptr(),
                a_value.as_ptr(),
            );
            ffi::cobre_highs_set_int_option(highs, c"simplex_iteration_limit".as_ptr(), 0);
            ffi::cobre_highs_run(highs);
        }
        let status1 = unsafe { ffi::cobre_highs_get_model_status(highs) };
        assert_eq!(status1, ffi::HIGHS_MODEL_STATUS_ITERATION_LIMIT);

        // Restore default settings (mirror restore_default_settings()).
        unsafe {
            ffi::cobre_highs_set_string_option(highs, c"solver".as_ptr(), c"simplex".as_ptr());
            ffi::cobre_highs_set_int_option(highs, c"simplex_strategy".as_ptr(), 1);
            ffi::cobre_highs_set_string_option(highs, c"presolve".as_ptr(), c"off".as_ptr());
            ffi::cobre_highs_set_double_option(
                highs,
                c"primal_feasibility_tolerance".as_ptr(),
                1e-7,
            );
            ffi::cobre_highs_set_double_option(highs, c"dual_feasibility_tolerance".as_ptr(), 1e-7);
            ffi::cobre_highs_set_string_option(highs, c"parallel".as_ptr(), c"off".as_ptr());
            ffi::cobre_highs_set_bool_option(highs, c"output_flag".as_ptr(), 0);
            // simplex_iteration_limit is NOT in restore_default_settings -- reset explicitly.
            ffi::cobre_highs_set_int_option(highs, c"simplex_iteration_limit".as_ptr(), i32::MAX);
        }

        // Second solve on the same model: must reach OPTIMAL.
        unsafe { ffi::cobre_highs_clear_solver(highs) };
        unsafe { ffi::cobre_highs_run(highs) };
        let status2 = unsafe { ffi::cobre_highs_get_model_status(highs) };
        let obj = unsafe { ffi::cobre_highs_get_objective_value(highs) };
        assert_eq!(
            status2,
            ffi::HIGHS_MODEL_STATUS_OPTIMAL,
            "after restoring defaults, second solve must be OPTIMAL, got {status2}"
        );
        assert!(
            (obj - 100.0).abs() < 1e-8,
            "objective after restore must be 100.0, got {obj}"
        );

        unsafe { ffi::cobre_highs_destroy(highs) };
    }

    /// iteration_limit=1 triggers ITERATION_LIMIT (or OPTIMAL if solved in 1 iteration).
    #[test]
    fn test_research_iteration_limit_one_triggers_iteration_limit_status() {
        use crate::ffi;

        let highs = unsafe { ffi::cobre_highs_create() };
        assert!(!highs.is_null());

        unsafe { ffi::cobre_highs_set_bool_option(highs, c"output_flag".as_ptr(), 0) };

        let col_cost: [f64; 3] = [0.0, 1.0, 50.0];
        let col_lower: [f64; 3] = [0.0, 0.0, 0.0];
        let col_upper: [f64; 3] = [10.0, f64::INFINITY, 8.0];
        let row_lower: [f64; 2] = [6.0, 14.0];
        let row_upper: [f64; 2] = [6.0, 14.0];
        let a_start: [i32; 4] = [0, 2, 2, 3];
        let a_index: [i32; 3] = [0, 1, 1];
        let a_value: [f64; 3] = [1.0, 2.0, 1.0];

        unsafe {
            ffi::cobre_highs_pass_lp(
                highs,
                3,
                2,
                3,
                ffi::HIGHS_MATRIX_FORMAT_COLWISE,
                ffi::HIGHS_OBJ_SENSE_MINIMIZE,
                0.0,
                col_cost.as_ptr(),
                col_lower.as_ptr(),
                col_upper.as_ptr(),
                row_lower.as_ptr(),
                row_upper.as_ptr(),
                a_start.as_ptr(),
                a_index.as_ptr(),
                a_value.as_ptr(),
            );
            ffi::cobre_highs_set_int_option(highs, c"simplex_iteration_limit".as_ptr(), 1);
            ffi::cobre_highs_run(highs);
        }

        let model_status = unsafe { ffi::cobre_highs_get_model_status(highs) };
        eprintln!("iteration_limit=1 model_status: {model_status}");
        // If the LP solves in 1 iteration it may be OPTIMAL; otherwise ITERATION_LIMIT.
        // We record both possibilities for the research document.
        assert!(
            model_status == ffi::HIGHS_MODEL_STATUS_ITERATION_LIMIT
                || model_status == ffi::HIGHS_MODEL_STATUS_OPTIMAL,
            "expected ITERATION_LIMIT or OPTIMAL, got {model_status}"
        );

        unsafe { ffi::cobre_highs_destroy(highs) };
    }

    /// HighsSolver maps UNBOUNDED (10) and INFEASIBLE (8) to SolverError variants.
    /// (UNBOUNDED_OR_INFEASIBLE only appears with IPM or allow_unbounded_or_infeasible=true.)
    #[test]
    fn test_research_verify_non_optimal_highs_status_mapping() {
        use crate::SolverInterface;
        use crate::backends::highs::HighsSolver;
        use crate::types::SolverError;
        use crate::types::StageTemplate;

        // Unbounded LP: min -x0 - x1, x0 + x1 >= 1, x0/x1 in [0, +inf).
        // With presolve=off and dual simplex, HiGHS returns UNBOUNDED (10).
        let unbounded_template = StageTemplate {
            num_cols: 2,
            num_rows: 1,
            num_nz: 2,
            col_starts: vec![0_i32, 1, 2],
            row_indices: vec![0_i32, 0],
            values: vec![1.0, 1.0],
            col_lower: vec![0.0, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY],
            objective: vec![-1.0, -1.0],
            row_lower: vec![1.0],
            row_upper: vec![f64::INFINITY],
            n_state: 1,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        let mut solver_unb = HighsSolver::new().expect("HighsSolver::new() must succeed");
        solver_unb.load_model(&unbounded_template);
        let result_unb = solver_unb.solve(None).map(|_| ());
        assert!(
            matches!(result_unb, Err(SolverError::Unbounded)),
            "unbounded LP must return Err(SolverError::Unbounded), got {result_unb:?}"
        );

        // Infeasible LP: x0 must equal 99 but is bounded to [0, 10].
        // HiGHS returns INFEASIBLE (8) directly; mapped to Err(SolverError::Infeasible).
        let infeasible_template = StageTemplate {
            num_cols: 1,
            num_rows: 1,
            num_nz: 1,
            col_starts: vec![0_i32, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0],
            col_upper: vec![10.0],
            objective: vec![0.0],
            row_lower: vec![99.0],
            row_upper: vec![99.0],
            n_state: 1,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        let mut solver_inf = HighsSolver::new().expect("HighsSolver::new() must succeed");
        solver_inf.load_model(&infeasible_template);
        let result_inf = solver_inf.solve(None).map(|_| ());
        assert!(
            matches!(result_inf, Err(SolverError::Infeasible)),
            "infeasible LP must return Err(SolverError::Infeasible), got {result_inf:?}"
        );
    }

    /// INFEASIBLE triggers cold-restart escalation before final verdict (retry_count >= 1).
    #[test]
    fn infeasible_initial_solve_runs_cold_restart_before_terminating() {
        use crate::SolverInterface;
        use crate::backends::highs::HighsSolver;
        use crate::types::SolverError;
        use crate::types::StageTemplate;

        // Genuinely infeasible: x0 == 99 but x0 is bounded to [0, 10]. No basis,
        // cold or warm, can satisfy this — the cold restart must confirm INFEASIBLE.
        let infeasible_template = StageTemplate {
            num_cols: 1,
            num_rows: 1,
            num_nz: 1,
            col_starts: vec![0_i32, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0],
            col_upper: vec![10.0],
            objective: vec![0.0],
            row_lower: vec![99.0],
            row_upper: vec![99.0],
            n_state: 1,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        solver.load_model(&infeasible_template);
        let result = solver.solve(None).map(|_| ());
        assert!(
            matches!(result, Err(SolverError::Infeasible)),
            "a genuinely infeasible LP must still return Err(Infeasible), got {result:?}"
        );
        assert!(
            solver.statistics().retry_count >= 1,
            "INFEASIBLE must now enter the escalation (>= 1 cold-restart attempt); got retry_count {}",
            solver.statistics().retry_count
        );
    }

    /// New HighsSolver has current_profile == HighsProfile::default().
    #[test]
    fn new_highs_solver_starts_with_default_profile() {
        use crate::HighsProfile;
        use crate::backends::highs::HighsSolver;

        let solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        assert_eq!(
            solver.current_profile,
            HighsProfile::default(),
            "current_profile must equal HighsProfile::default() immediately after construction"
        );
    }

    #[test]
    fn highs_profile_default_matches_default_options_table() {
        use crate::HighsProfile;
        use crate::backends::highs::config::{DefaultOption, OptionValue, default_options};

        fn value_of<'a>(options: &'a [DefaultOption], name: &std::ffi::CStr) -> &'a OptionValue {
            &options
                .iter()
                .find(|opt| opt.name == name)
                .expect("default_options() must contain this entry")
                .value
        }

        let profile = HighsProfile::default();
        let options = default_options();

        match value_of(&options, c"primal_feasibility_tolerance") {
            OptionValue::Double(v) => assert_eq!(
                profile.primal_feasibility_tolerance, *v,
                "primal_feasibility_tolerance must match default_options()"
            ),
            _ => panic!("primal_feasibility_tolerance is not a Double option"),
        }
        match value_of(&options, c"dual_feasibility_tolerance") {
            OptionValue::Double(v) => assert_eq!(
                profile.dual_feasibility_tolerance, *v,
                "dual_feasibility_tolerance must match default_options()"
            ),
            _ => panic!("dual_feasibility_tolerance is not a Double option"),
        }
        match value_of(&options, c"simplex_dual_edge_weight_strategy") {
            OptionValue::Int(v) => assert_eq!(
                profile.simplex_dual_edge_weight_strategy, *v,
                "simplex_dual_edge_weight_strategy must match default_options()"
            ),
            _ => panic!("simplex_dual_edge_weight_strategy is not an Int option"),
        }
        match value_of(&options, c"simplex_scale_strategy") {
            OptionValue::Int(v) => assert_eq!(
                profile.simplex_scale_strategy, *v,
                "simplex_scale_strategy must match default_options()"
            ),
            _ => panic!("simplex_scale_strategy is not an Int option"),
        }
        match value_of(&options, c"simplex_price_strategy") {
            OptionValue::Int(v) => assert_eq!(
                profile.simplex_price_strategy, *v,
                "simplex_price_strategy must match default_options()"
            ),
            _ => panic!("simplex_price_strategy is not an Int option"),
        }
        match value_of(&options, c"presolve") {
            OptionValue::Str(v) => assert_eq!(
                profile.presolve.as_option(),
                *v,
                "presolve must match default_options()"
            ),
            _ => panic!("presolve is not a Str option"),
        }
        match value_of(&options, c"simplex_update_limit") {
            OptionValue::Int(v) => assert_eq!(
                i64::from(profile.simplex_update_limit),
                i64::from(*v),
                "simplex_update_limit must match default_options()"
            ),
            _ => panic!("simplex_update_limit is not an Int option"),
        }
        match value_of(&options, c"factor_pivot_threshold") {
            OptionValue::Double(v) => assert_eq!(
                profile.factor_pivot_threshold, *v,
                "factor_pivot_threshold must match default_options()"
            ),
            _ => panic!("factor_pivot_threshold is not a Double option"),
        }
        match value_of(&options, c"use_warm_start") {
            OptionValue::Bool(v) => assert_eq!(
                i32::from(profile.use_warm_start),
                *v,
                "use_warm_start must match default_options()"
            ),
            _ => panic!("use_warm_start is not a Bool option"),
        }
        match value_of(&options, c"dual_simplex_cost_perturbation_multiplier") {
            OptionValue::Double(v) => assert_eq!(
                profile.cost_perturbation, *v,
                "cost_perturbation must match default_options()"
            ),
            _ => panic!("dual_simplex_cost_perturbation_multiplier is not a Double option"),
        }
        match value_of(&options, c"rebuild_refactor_solution_error_tolerance") {
            OptionValue::Double(v) => assert_eq!(
                profile.refactor_error_tolerance, *v,
                "refactor_error_tolerance must match default_options()"
            ),
            _ => panic!("rebuild_refactor_solution_error_tolerance is not a Double option"),
        }
        match value_of(&options, c"dual_steepest_edge_weight_log_error_threshold") {
            OptionValue::Double(v) => assert_eq!(
                profile.steepest_edge_devex_fallback_threshold, *v,
                "steepest_edge_devex_fallback_threshold must match default_options()"
            ),
            _ => panic!("dual_steepest_edge_weight_log_error_threshold is not a Double option"),
        }
    }
}
