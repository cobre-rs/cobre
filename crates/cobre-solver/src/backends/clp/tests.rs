use super::retry::LADDER_RUNGS;
use crate::backends::clp::{ClpAlgorithm, ClpProfile, ClpSolver, clp_version};
use crate::profile::DEFAULT_PROFILE_HEURISTIC_SENTINEL;
use crate::types::{Basis, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate};
use crate::{BasisStatus, ProfiledSolver, SolverInterface};

fn assert_profile_bounds<P: Copy + PartialEq + Default + Send>() {}

fn assert_send<T: Send>() {}

// Solver Interface Testing SS1.1 fixture (mirrors the HiGHS test fixture):
//   minimize  0*x0 + 1*x1 + 50*x2
//   subject to  row0:  x0           = 6
//               row1: 2*x0     + x2 = 14
//   x0 in [0, 10], x1 in [0, +inf), x2 in [0, 8]
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

// Solver Interface Testing SS1.2 fixture (mirrors the conformance fixture):
// two appended `>=` rows over the SS1.1 columns:
//   row2: -5*x0 + x1 >= 20   (row_lower 20, row_upper +inf)
//   row3:  3*x0 + x1 >= 80   (row_lower 80, row_upper +inf)
//
// CSR (row-major):
//   row_starts  = [0, 2, 4]
//   col_indices = [0, 1, 0, 1]
//   values      = [-5.0, 1.0, 3.0, 1.0]
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
fn test_clp_profile_default_values() {
    let p = ClpProfile::default();
    assert_eq!(p.perturbation, 102);
    assert_eq!(p.scaling, 0);
    assert_eq!(p.primal_feasibility_tolerance, 1e-9);
    assert_eq!(p.dual_feasibility_tolerance, 1e-9);
    assert_eq!(
        p.simplex_iteration_limit,
        DEFAULT_PROFILE_HEURISTIC_SENTINEL
    );
    assert_eq!(p.algorithm, ClpAlgorithm::Dual);
    // mode 3 is CLP's steepest-edge ctor default; 0 is the "leave CLP's internal
    // default" factorization-cadence sentinel.
    assert_eq!(p.dual_pricing_mode, 3);
    assert_eq!(p.factorization_frequency, 0);
    assert_profile_bounds::<ClpProfile>();
    let copied = p;
    assert_eq!(copied, p);
}

#[test]
fn test_clp_solver_create_and_name() {
    let solver = ClpSolver::new().expect("CLP solver creation failed");
    assert_eq!(solver.name(), "CLP");
}

#[test]
fn test_clp_solver_name_version_format() {
    let solver = ClpSolver::new().expect("CLP solver creation failed");
    let nv = solver.solver_name_version();
    assert!(nv.starts_with("CLP "), "got {nv}");
    assert_eq!(nv.matches('.').count(), 2, "got {nv}");
}

#[test]
fn test_clp_profile_satisfies_associated_type_bounds() {
    assert_profile_bounds::<ClpProfile>();
}

#[test]
fn test_clp_solver_send_bound() {
    assert_send::<ClpSolver>();
    // Also exercise the free version function for coverage.
    let _ = clp_version();
}

#[test]
fn test_clp_load_model_updates_dimensions() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    let template = make_fixture_stage_template();
    solver.load_model(&template);

    assert_eq!(solver.num_cols, 3);
    assert_eq!(solver.num_rows, 2);
    assert!(solver.has_model);

    assert_eq!(solver.col_value.len(), 3);
    assert_eq!(solver.col_dual.len(), 3);
    assert_eq!(solver.row_dual.len(), 2);
}

#[test]
fn test_clp_load_model_accepts_infinite_bounds() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    let mut template = make_fixture_stage_template();
    template.col_upper[1] = f64::INFINITY;

    // The C wrapper owns the DBL_MAX translation, so the Rust layer never reads
    // or mutates the infinity.
    solver.load_model(&template);

    assert_eq!(solver.num_cols, 3);
    assert_eq!(solver.num_rows, 2);
    assert!(solver.has_model);
}

#[test]
fn test_clp_load_model_reload_zero_row() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.load_model(&make_fixture_stage_template());

    let zero_row = StageTemplate {
        num_cols: 1,
        num_rows: 0,
        num_nz: 0,
        col_starts: vec![0_i32, 0],
        row_indices: Vec::new(),
        values: Vec::new(),
        col_lower: vec![0.0],
        col_upper: vec![1.0],
        objective: vec![1.0],
        row_lower: Vec::new(),
        row_upper: Vec::new(),
        n_state: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };
    solver.load_model(&zero_row);

    assert_eq!(solver.num_rows, 0);
    assert_eq!(solver.row_dual.len(), 0);
    assert_eq!(solver.num_cols, 1);
}

#[test]
fn test_clp_load_model_count_accumulates() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    assert_eq!(solver.stats.load_model_count, 1);

    solver.load_model(&template);
    assert_eq!(solver.stats.load_model_count, 2);
}

#[test]
fn test_clp_add_rows_updates_dimensions() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.load_model(&make_fixture_stage_template());
    let batch = make_fixture_row_batch();
    solver.add_rows(&batch);

    assert_eq!(solver.num_rows, 4);
    assert_eq!(solver.row_dual.len(), 4);
    assert_eq!(solver.num_cols, 3);

    assert_eq!(solver.num_nz, 3 + 4);
    assert_eq!(solver.col_starts.len(), 4);
    assert_eq!(solver.col_starts[3], 7);

    // Hand-checked merged CSC. SS1.1 columns:
    //   col0: (row0, 1.0), (row1, 2.0)
    //   col1: (none)
    //   col2: (row1, 1.0)
    // SS1.2 batch appends rows 2 and 3:
    //   col0 gains (row2, -5.0), (row3, 3.0)
    //   col1 gains (row2,  1.0), (row3, 1.0)
    //   col2 gains nothing
    // Merged column-major layout (existing entries first, then appended):
    //   col0: [ (0,1.0), (1,2.0), (2,-5.0), (3,3.0) ]  -> starts at 0
    //   col1: [ (2,1.0), (3,1.0) ]                      -> starts at 4
    //   col2: [ (1,1.0) ]                               -> starts at 6
    assert_eq!(solver.col_starts, vec![0, 4, 6, 7]);
    assert_eq!(solver.row_indices, vec![0, 1, 2, 3, 2, 3, 1]);
    assert_eq!(solver.values, vec![1.0, 2.0, -5.0, 3.0, 1.0, 1.0, 1.0]);

    assert_eq!(solver.row_lower, vec![6.0, 14.0, 20.0, 80.0]);
    assert!(solver.row_upper[2].is_infinite());
    assert!(solver.row_upper[3].is_infinite());
}

#[test]
fn test_clp_set_row_bounds_patches_retained() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.load_model(&make_fixture_stage_template());

    solver.set_row_bounds(&[0], &[4.0], &[4.0]);
    assert_eq!(solver.row_lower[0], 4.0);
    assert_eq!(solver.row_upper[0], 4.0);
    assert_eq!(solver.row_lower[1], 14.0);
}

#[test]
fn test_clp_set_col_bounds_patches_retained() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.load_model(&make_fixture_stage_template());

    solver.set_col_bounds(&[1], &[10.0], &[f64::INFINITY]);
    assert_eq!(solver.col_lower[1], 10.0);
    assert!(solver.col_upper[1].is_infinite());
    assert_eq!(solver.col_lower[0], 0.0);
}

#[test]
fn test_clp_set_bounds_empty_no_reload() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.load_model(&make_fixture_stage_template());

    let load_count_before = solver.stats.load_model_count;

    // Empty index slices are a no-op: no FFI mutation, dimensions and bound
    // vectors unchanged.
    solver.set_row_bounds(&[], &[], &[]);
    solver.set_col_bounds(&[], &[], &[]);

    assert_eq!(solver.num_rows, 2);
    assert_eq!(solver.num_cols, 3);
    assert_eq!(solver.stats.load_model_count, load_count_before);
    assert_eq!(solver.row_lower, vec![6.0, 14.0]);
    assert_eq!(solver.col_lower, vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_clp_add_rows_then_solve_objective() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.load_model(&make_fixture_stage_template());
    solver.add_rows(&make_fixture_row_batch());

    assert_eq!(solver.num_cols, 3);
    assert_eq!(solver.num_rows, 4);
    assert_eq!(solver.num_nz, 7);
    assert_eq!(solver.row_dual.len(), 4);
}

#[test]
fn test_clp_solve_basic_lp() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.load_model(&make_fixture_stage_template());

    let view = solver
        .solve(None)
        .expect("SS1.1 LP should solve to optimal");

    // Optimal x = (6, 0, 2), obj = 100.
    assert!(
        (view.objective - 100.0).abs() < 1e-8,
        "objective {} not within 1e-8 of 100.0",
        view.objective
    );
    assert!(
        (view.primal[0] - 6.0).abs() < 1e-8,
        "primal[0] {} not within 1e-8 of 6.0",
        view.primal[0]
    );
    assert!(
        (view.primal[2] - 2.0).abs() < 1e-8,
        "primal[2] {} not within 1e-8 of 2.0",
        view.primal[2]
    );

    assert_eq!(view.primal.len(), 3);
    assert_eq!(view.reduced_costs.len(), 3);
    assert_eq!(view.dual.len(), 2);
}

#[test]
fn test_clp_solve_basic_lp_primal_algorithm() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.apply_profile(&ClpProfile {
        algorithm: ClpAlgorithm::Primal,
        ..ClpProfile::default()
    });
    solver.load_model(&make_fixture_stage_template());

    let view = solver
        .solve(None)
        .expect("SS1.1 LP should solve to optimal via the primal simplex");

    // Optimal x = (6, 0, 2), obj = 100.
    assert!(
        (view.objective - 100.0).abs() < 1e-8,
        "objective {} not within 1e-8 of 100.0",
        view.objective
    );
    assert!(
        (view.primal[0] - 6.0).abs() < 1e-8,
        "primal[0] {} not within 1e-8 of 6.0",
        view.primal[0]
    );
    assert!(
        (view.primal[1] - 0.0).abs() < 1e-8,
        "primal[1] {} not within 1e-8 of 0.0",
        view.primal[1]
    );
    assert!(
        (view.primal[2] - 2.0).abs() < 1e-8,
        "primal[2] {} not within 1e-8 of 2.0",
        view.primal[2]
    );
}

#[test]
fn test_clp_solve_infeasible() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.load_model(&make_fixture_stage_template());

    // Pin x0 = 100, violating the row0 equality x0 = 6.
    solver.set_col_bounds(&[0], &[100.0], &[100.0]);
    let result = solver.solve(None);

    assert!(
        matches!(result, Err(SolverError::Infeasible)),
        "expected Err(Infeasible), got {result:?}"
    );

    // Stats reconcile: the failed solve is counted once as a failure (not
    // double-counted), every rung is charged to `retry_count`, and neither
    // `success_count` nor `first_try_successes` moves.
    assert_eq!(solver.stats.solve_count, 1);
    assert_eq!(solver.stats.failure_count, 1);
    assert_eq!(solver.stats.success_count, 0);
    assert_eq!(solver.stats.first_try_successes, 0);
    assert_eq!(solver.stats.retry_count, LADDER_RUNGS as u64);
}

#[test]
fn test_clp_escalation_restores_floor_after_exhaustion() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.load_model(&make_fixture_stage_template());

    let floor_perturbation = solver.current_profile.perturbation;
    let floor_scaling = solver.current_profile.scaling;

    solver.set_col_bounds(&[0], &[100.0], &[100.0]);
    let infeasible = solver.solve(None);
    assert!(
        matches!(infeasible, Err(SolverError::Infeasible)),
        "expected Err(Infeasible), got {infeasible:?}"
    );

    // The ladder restored the floor, so the NEXT solve starts clean.
    assert_eq!(solver.current_profile.perturbation, floor_perturbation);
    assert_eq!(solver.current_profile.scaling, floor_scaling);

    // Relax back to feasibility: with the floor restored this is a clean
    // first-try win — the ladder does NOT fire again (retry_count unchanged).
    solver.set_col_bounds(&[0], &[0.0], &[f64::INFINITY]);
    let view = solver
        .solve(None)
        .expect("feasible re-solve after floor restore should be optimal");
    assert!(
        (view.objective - 100.0).abs() < 1e-8,
        "objective {} not within 1e-8 of 100.0",
        view.objective
    );

    // The ladder ran only once (during the exhausted first solve), so
    // retry_count stays at exactly one ladder's worth of rungs.
    assert_eq!(solver.stats.solve_count, 2);
    assert_eq!(solver.stats.success_count, 1);
    assert_eq!(solver.stats.first_try_successes, 1);
    assert_eq!(solver.stats.failure_count, 1);
    assert_eq!(solver.stats.retry_count, LADDER_RUNGS as u64);
}

/// Every `ClpProfile` field survives escalation finalization — the cached
/// `current_profile` is restored in full, not narrowed to perturbation/scaling
/// only (unlike `test_clp_escalation_restores_floor_after_exhaustion`).
#[test]
fn test_clp_full_profile_survives_escalation_finalization() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.load_model(&make_fixture_stage_template());

    let profile = ClpProfile {
        perturbation: 100,
        scaling: 1,
        primal_feasibility_tolerance: 1e-7,
        dual_feasibility_tolerance: 1e-7,
        simplex_iteration_limit: 50_000,
        algorithm: ClpAlgorithm::Primal,
        dual_pricing_mode: 1,
        factorization_frequency: 200,
    };
    assert_ne!(
        profile,
        ClpProfile::default(),
        "fixture profile must differ from every ClpProfile::default() field"
    );
    solver.apply_profile(&profile);

    solver.set_col_bounds(&[0], &[100.0], &[100.0]);
    let infeasible = solver.solve(None);
    assert!(
        matches!(infeasible, Err(SolverError::Infeasible)),
        "expected Err(Infeasible), got {infeasible:?}"
    );

    assert_eq!(
        solver.current_profile, profile,
        "every ClpProfile field must survive the escalation finalization"
    );
}

#[test]
fn test_clp_solve_unbounded() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");

    // Single column, objective -1, lower 0, upper +inf, no rows.
    let unbounded = StageTemplate {
        num_cols: 1,
        num_rows: 0,
        num_nz: 0,
        col_starts: vec![0_i32, 0],
        row_indices: Vec::new(),
        values: Vec::new(),
        col_lower: vec![0.0],
        col_upper: vec![f64::INFINITY],
        objective: vec![-1.0],
        row_lower: Vec::new(),
        row_upper: Vec::new(),
        n_state: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };
    solver.load_model(&unbounded);
    let result = solver.solve(None);

    assert!(
        matches!(result, Err(SolverError::Unbounded)),
        "expected Err(Unbounded), got {result:?}"
    );

    // DUAL_INFEASIBLE (unbounded) stays terminal: the escalation ladder is
    // NOT run for it (a genuine unbounded LP is not retry-recoverable), so
    // no retries are charged and the solve is counted once as a failure.
    assert_eq!(solver.stats.failure_count, 1);
    assert_eq!(solver.stats.retry_count, 0);
    assert_eq!(solver.stats.success_count, 0);
}

#[test]
fn test_clp_solve_twice_stats() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.load_model(&make_fixture_stage_template());

    let _ = solver.solve(None).expect("first solve should be optimal");
    let _ = solver.solve(None).expect("second solve should be optimal");

    assert_eq!(solver.stats.solve_count, 2);
    assert_eq!(solver.stats.success_count, 2);
    // Both solves are first-try wins, the escalation ladder never runs, so no
    // retries are charged.
    assert_eq!(solver.stats.first_try_successes, 2);
    assert_eq!(solver.stats.retry_count, 0);
    assert_eq!(solver.stats.failure_count, 0);
}

#[test]
fn test_clp_statistics_into_equals_statistics() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.load_model(&make_fixture_stage_template());
    let _ = solver.solve(None).expect("solve should be optimal");

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

#[test]
fn test_clp_get_basis_dimensions_and_codes() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.load_model(&make_fixture_stage_template());
    let _ = solver
        .solve(None)
        .expect("SS1.1 LP should solve to optimal");

    // get_basis resizes a 0/0 Basis to the LP dimensions and fills it with
    // canonical statuses mapped from the raw CLP codes via `from_clp_code`.
    let mut out = Basis::new(0, 0);
    solver.get_basis(&mut out);

    assert_eq!(out.col_status.len(), 3);
    assert_eq!(out.row_status.len(), 2);
    for &s in out.col_status.iter().chain(out.row_status.iter()) {
        assert!(
            !matches!(s, BasisStatus::Nonbasic),
            "status {s:?} is not CLP-representable (from_clp_code never yields Nonbasic)"
        );
    }
}

#[test]
fn test_clp_warm_start_roundtrip_objective() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.load_model(&make_fixture_stage_template());
    let _ = solver.solve(None).expect("first solve should be optimal");

    let mut captured = Basis::new(0, 0);
    solver.get_basis(&mut captured);

    let view = solver
        .solve(Some(&captured))
        .expect("warm-start solve should be optimal");
    assert!(
        (view.objective - 100.0).abs() < 1e-8,
        "warm-start objective {} not within 1e-8 of 100.0",
        view.objective
    );

    assert_eq!(solver.stats.basis_offered, 1);
    assert!(
        solver.stats.total_basis_set_time_seconds >= 0.0,
        "total_basis_set_time_seconds {} should be non-negative",
        solver.stats.total_basis_set_time_seconds
    );
}

/// An undersized offered basis (2 rows vs 4 after `add_rows`) is rejected with
/// `Err(SolverError::BasisRowCountMismatch)`; the rejection increments
/// `basis_consistency_failures` and leaves `basis_offered` untouched.
#[test]
fn test_clp_solve_rejects_undersized_row_basis() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.load_model(&make_fixture_stage_template());
    let _ = solver.solve(None).expect("first solve should be optimal");

    let mut captured = Basis::new(0, 0);
    solver.get_basis(&mut captured);

    solver.load_model(&make_fixture_stage_template());
    solver.add_rows(&make_fixture_row_batch());

    let offered_before = solver.stats.basis_offered;
    let failures_before = solver.stats.basis_consistency_failures;

    let err_variant = solver.solve(Some(&captured)).map(|_| ());

    assert_eq!(
        solver.stats.basis_consistency_failures - failures_before,
        1,
        "basis_consistency_failures must increment by 1 for an undersized row basis"
    );
    assert_eq!(
        solver.stats.basis_offered, offered_before,
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

#[test]
#[should_panic(expected = "loaded model")]
fn test_clp_get_basis_without_model_panics() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.get_basis(&mut Basis::new(0, 0));
}

#[test]
fn test_clp_basis_roundtrip_identity() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.load_model(&make_fixture_stage_template());
    let _ = solver
        .solve(None)
        .expect("SS1.1 LP should solve to optimal");

    // Two get_basis calls into fresh Basis values yield identical vectors — the
    // round-trip stores exactly the CLP-reported codes with no drift.
    let mut first = Basis::new(0, 0);
    solver.get_basis(&mut first);
    let mut second = Basis::new(0, 0);
    solver.get_basis(&mut second);

    assert_eq!(first.col_status, second.col_status);
    assert_eq!(first.row_status, second.row_status);
}

#[test]
fn test_clp_apply_default_profile_then_solve() {
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.apply_profile(&ClpProfile::default());
    assert_eq!(solver.current_profile, ClpProfile::default());

    solver.load_model(&make_fixture_stage_template());
    let view = solver
        .solve(None)
        .expect("SS1.1 LP should solve to optimal after apply_profile");
    assert!(
        (view.objective - 100.0).abs() < 1e-8,
        "objective {} not within 1e-8 of 100.0",
        view.objective
    );
}

#[test]
fn test_clp_apply_tuned_pricing_profile_then_solve() {
    // A tuned profile (full DSE = 1, refactor cadence = 200) drives both knobs
    // through the shim and must still reach the SS1.1 optimum (obj 100.0): the
    // knobs change the iteration path, not the optimum.
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.apply_profile(&ClpProfile {
        dual_pricing_mode: 1,
        factorization_frequency: 200,
        ..ClpProfile::default()
    });
    assert_eq!(solver.current_profile.dual_pricing_mode, 1);
    assert_eq!(solver.current_profile.factorization_frequency, 200);

    solver.load_model(&make_fixture_stage_template());
    let view = solver
        .solve(None)
        .expect("SS1.1 LP should solve to optimal with a tuned DSE/refactor profile");
    assert!(
        (view.objective - 100.0).abs() < 1e-8,
        "objective {} not within 1e-8 of 100.0",
        view.objective
    );
}

#[test]
fn test_clp_tolerance_setters_cache_profile() {
    // apply_profile caches the tolerance values in current_profile so
    // ProfiledSolver delta-tracking observes the change.
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");

    solver.apply_profile(&ClpProfile {
        primal_feasibility_tolerance: 1e-7,
        dual_feasibility_tolerance: 1e-7,
        ..Default::default()
    });
    assert_eq!(solver.current_profile.primal_feasibility_tolerance, 1e-7);
    assert_eq!(solver.current_profile.dual_feasibility_tolerance, 1e-7);
}

#[test]
fn test_clp_profiled_solver_default_noop() {
    // set_profile(&default) is a delta-tracking no-op — the wrapper starts at
    // default, so no apply_profile is issued and current_profile() is unchanged.
    let solver = ClpSolver::new().expect("CLP solver creation failed");
    let mut profiled = ProfiledSolver::new(solver);

    assert_eq!(*profiled.current_profile(), ClpProfile::default());
    profiled.set_profile(&ClpProfile::default());
    assert_eq!(*profiled.current_profile(), ClpProfile::default());
}

#[test]
fn test_clp_usable_as_generic_solver_bound() {
    fn run_solve<S: SolverInterface>(s: &mut S) -> Result<SolutionView<'_>, SolverError> {
        s.solve(None)
    }

    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.load_model(&make_fixture_stage_template());
    let view =
        run_solve::<ClpSolver>(&mut solver).expect("generic run_solve over ClpSolver should be Ok");
    assert!(
        (view.objective - 100.0).abs() < 1e-8,
        "objective {} not within 1e-8 of 100.0",
        view.objective
    );
}

#[test]
fn test_clp_resolve_simplex_cap_sentinel_and_explicit() {
    // On a 3-col LP, the sentinel (0) resolves to max(100_000, 3*50) = 100_000;
    // an explicit 500 resolves to 500 verbatim.
    let mut solver = ClpSolver::new().expect("CLP solver creation failed");
    solver.load_model(&make_fixture_stage_template());

    solver.apply_profile(&ClpProfile {
        simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        ..Default::default()
    });
    assert_eq!(solver.resolve_simplex_cap(), 100_000);

    solver.apply_profile(&ClpProfile {
        simplex_iteration_limit: 500,
        ..Default::default()
    });
    assert_eq!(solver.resolve_simplex_cap(), 500);
}
