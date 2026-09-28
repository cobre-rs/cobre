//! Integration tests for profile × retry-level tolerance composition.
//!
//! Verifies that primal and dual feasibility tolerances at retry levels 3, 7,
//! 10, and 11 satisfy `applied = max(level_default, profile_value)`. Also
//! verifies the iteration-cap composition at retry level 0, and that an
//! unchanged profile is NOT re-applied on subsequent solve calls (the
//! delta-only dispatch contract; retry finalization owns re-application).
//!
//! Requires the `test-support` feature:
//!   cargo nextest run -p cobre-solver --features test-support
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

#[cfg(feature = "test-support")]
mod tests {
    use std::cell::Cell;

    use cobre_solver::types::{Basis, RowBatch, SolutionView, SolverError, SolverStatistics};
    use cobre_solver::{
        HighsProfile, HighsSolver, PresolveKind, ProfiledSolver, SolverInterface, StageTemplate,
    };

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Minimal 2-column, no-constraint LP fixture. `num_cols = 2` feeds the
    /// `set_iteration_limits` heuristic the iteration-cap tests check.
    fn make_minimal_template() -> StageTemplate {
        StageTemplate {
            num_cols: 2,
            num_rows: 0,
            num_nz: 0,
            col_starts: vec![0_i32, 0, 0],
            row_indices: vec![],
            values: vec![],
            col_lower: vec![0.0, 0.0],
            col_upper: vec![10.0, 10.0],
            objective: vec![1.0, 1.0],
            row_lower: vec![],
            row_upper: vec![],
            n_state: 0,
            col_scale: vec![],
            row_scale: vec![],
        }
    }

    fn make_solver() -> HighsSolver {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        solver.load_model(&make_minimal_template());
        solver
    }

    fn read_tolerances(solver: &HighsSolver) -> (f64, f64) {
        let primal = solver
            .get_double_option(c"primal_feasibility_tolerance")
            .expect("primal_feasibility_tolerance must be readable");
        let dual = solver
            .get_double_option(c"dual_feasibility_tolerance")
            .expect("dual_feasibility_tolerance must be readable");
        (primal, dual)
    }

    // ── Iteration-cap composition ─────────────────────────────────────────────

    /// Sentinel branch: when the profile simplex cap is the sentinel
    /// value (0), `apply_retry_level_options(0)` must apply the historical
    /// heuristic `max(100_000, num_cols * 50)`.
    ///
    /// With 2 columns the heuristic yields `max(100_000, 100) = 100_000`.
    #[test]
    fn sentinel_uses_heuristic() {
        let mut solver = make_solver();
        solver.apply_profile(&HighsProfile {
            simplex_iteration_limit: 0,
            ..Default::default()
        });

        solver.apply_retry_level_options_for_test(0);

        let cap = solver
            .get_int_option(c"simplex_iteration_limit")
            .expect("simplex_iteration_limit must be readable");

        assert_eq!(
            cap, 100_000,
            "sentinel should produce heuristic cap 100_000; got {cap}"
        );
    }

    /// Literal-value branch: when the profile simplex cap is non-zero,
    /// `apply_retry_level_options(0)` must apply that exact value verbatim.
    #[test]
    fn nonzero_simplex_cap_used_verbatim() {
        let mut solver = make_solver();
        solver.apply_profile(&HighsProfile {
            simplex_iteration_limit: 42_000,
            ..Default::default()
        });

        solver.apply_retry_level_options_for_test(0);

        let cap = solver
            .get_int_option(c"simplex_iteration_limit")
            .expect("simplex_iteration_limit must be readable");

        assert_eq!(
            cap, 42_000,
            "non-zero profile cap must be applied verbatim; got {cap}"
        );
    }

    /// IPM cap branch: `ipm_iteration_limit` from the profile is applied
    /// verbatim regardless of which retry level is active (`set_iteration_limits`
    /// always sets it).
    #[test]
    fn ipm_profile_value_applied() {
        let mut solver = make_solver();
        solver.apply_profile(&HighsProfile {
            ipm_iteration_limit: 500,
            ..Default::default()
        });

        solver.apply_retry_level_options_for_test(0);

        let cap = solver
            .get_int_option(c"ipm_iteration_limit")
            .expect("ipm_iteration_limit must be readable");

        assert_eq!(
            cap, 500,
            "profile IPM cap must be applied verbatim; got {cap}"
        );
    }

    // ── Default-attempt tolerance setter wires to FFI ────────────────────────

    /// Verify that `apply_profile` writes the primal tolerance to the
    /// underlying `HiGHS` option immediately (not only at retry time).
    #[test]
    fn primal_setter_propagates_to_ffi() {
        let mut solver = make_solver();
        solver.apply_profile(&HighsProfile {
            primal_feasibility_tolerance: 3e-8,
            ..Default::default()
        });

        let tol = solver
            .get_double_option(c"primal_feasibility_tolerance")
            .expect("primal_feasibility_tolerance must be readable");

        assert!(
            (tol - 3e-8).abs() < 1e-20,
            "apply_profile must write primal tolerance to FFI; expected 3e-8, got {tol}"
        );
    }

    // ── Loose profile (profile_value > level_default) ────────────────────────
    //
    // Profile primal = dual = 1e-5.
    // Levels 3 and 7  have level_default = 1e-8  →  max(1e-8, 1e-5) = 1e-5.
    // Levels 10 and 11 have level_default = 1e-7  →  max(1e-7, 1e-5) = 1e-5.

    fn make_loose_profile_solver() -> HighsSolver {
        let mut solver = make_solver();
        solver.apply_profile(&HighsProfile {
            primal_feasibility_tolerance: 1e-5,
            dual_feasibility_tolerance: 1e-5,
            ..Default::default()
        });
        solver
    }

    #[test]
    fn loose_profile_level3_applies_profile_value() {
        let mut solver = make_loose_profile_solver();
        solver.apply_retry_level_options_for_test(3);

        let (primal, dual) = read_tolerances(&solver);

        assert!(
            (primal - 1e-5).abs() < 1e-20,
            "level 3 loose: primal must be max(1e-8, 1e-5) = 1e-5; got {primal}"
        );
        assert!(
            (dual - 1e-5).abs() < 1e-20,
            "level 3 loose: dual must be max(1e-8, 1e-5) = 1e-5; got {dual}"
        );
    }

    #[test]
    fn loose_profile_level7_applies_profile_value() {
        let mut solver = make_loose_profile_solver();
        solver.apply_extended_retry_options_for_test(7);

        let (primal, dual) = read_tolerances(&solver);

        assert!(
            (primal - 1e-5).abs() < 1e-20,
            "level 7 loose: primal must be max(1e-8, 1e-5) = 1e-5; got {primal}"
        );
        assert!(
            (dual - 1e-5).abs() < 1e-20,
            "level 7 loose: dual must be max(1e-8, 1e-5) = 1e-5; got {dual}"
        );
    }

    #[test]
    fn loose_profile_level10_applies_profile_value() {
        let mut solver = make_loose_profile_solver();
        solver.apply_extended_retry_options_for_test(10);

        let (primal, dual) = read_tolerances(&solver);

        assert!(
            (primal - 1e-5).abs() < 1e-20,
            "level 10 loose: primal must be max(1e-7, 1e-5) = 1e-5; got {primal}"
        );
        assert!(
            (dual - 1e-5).abs() < 1e-20,
            "level 10 loose: dual must be max(1e-7, 1e-5) = 1e-5; got {dual}"
        );
    }

    #[test]
    fn loose_profile_level11_applies_profile_value() {
        let mut solver = make_loose_profile_solver();
        solver.apply_extended_retry_options_for_test(11);

        let (primal, dual) = read_tolerances(&solver);

        assert!(
            (primal - 1e-5).abs() < 1e-20,
            "level 11 loose: primal must be max(1e-7, 1e-5) = 1e-5; got {primal}"
        );
        assert!(
            (dual - 1e-5).abs() < 1e-20,
            "level 11 loose: dual must be max(1e-7, 1e-5) = 1e-5; got {dual}"
        );
    }

    // ── Strict profile (profile_value < level_default) ───────────────────────
    //
    // Profile primal = dual = 1e-12.
    // Levels 3 and 7  have level_default = 1e-8  →  max(1e-8, 1e-12) = 1e-8.
    // Levels 10 and 11 have level_default = 1e-7  →  max(1e-7, 1e-12) = 1e-7.

    fn make_strict_profile_solver() -> HighsSolver {
        let mut solver = make_solver();
        solver.apply_profile(&HighsProfile {
            primal_feasibility_tolerance: 1e-12,
            dual_feasibility_tolerance: 1e-12,
            ..Default::default()
        });
        solver
    }

    #[test]
    fn strict_profile_level3_applies_level_default() {
        let mut solver = make_strict_profile_solver();
        solver.apply_retry_level_options_for_test(3);

        let (primal, dual) = read_tolerances(&solver);

        assert!(
            (primal - 1e-8).abs() < 1e-20,
            "level 3 strict: primal must be max(1e-8, 1e-12) = 1e-8; got {primal}"
        );
        assert!(
            (dual - 1e-8).abs() < 1e-20,
            "level 3 strict: dual must be max(1e-8, 1e-12) = 1e-8; got {dual}"
        );
    }

    #[test]
    fn strict_profile_level7_applies_level_default() {
        let mut solver = make_strict_profile_solver();
        solver.apply_extended_retry_options_for_test(7);

        let (primal, dual) = read_tolerances(&solver);

        assert!(
            (primal - 1e-8).abs() < 1e-20,
            "level 7 strict: primal must be max(1e-8, 1e-12) = 1e-8; got {primal}"
        );
        assert!(
            (dual - 1e-8).abs() < 1e-20,
            "level 7 strict: dual must be max(1e-8, 1e-12) = 1e-8; got {dual}"
        );
    }

    #[test]
    fn strict_profile_level10_applies_level_default() {
        let mut solver = make_strict_profile_solver();
        solver.apply_extended_retry_options_for_test(10);

        let (primal, dual) = read_tolerances(&solver);

        assert!(
            (primal - 1e-7).abs() < 1e-20,
            "level 10 strict: primal must be max(1e-7, 1e-12) = 1e-7; got {primal}"
        );
        assert!(
            (dual - 1e-7).abs() < 1e-20,
            "level 10 strict: dual must be max(1e-7, 1e-12) = 1e-7; got {dual}"
        );
    }

    #[test]
    fn strict_profile_level11_applies_level_default() {
        let mut solver = make_strict_profile_solver();
        solver.apply_extended_retry_options_for_test(11);

        let (primal, dual) = read_tolerances(&solver);

        assert!(
            (primal - 1e-7).abs() < 1e-20,
            "level 11 strict: primal must be max(1e-7, 1e-12) = 1e-7; got {primal}"
        );
        assert!(
            (dual - 1e-7).abs() < 1e-20,
            "level 11 strict: dual must be max(1e-7, 1e-12) = 1e-7; got {dual}"
        );
    }

    // ── Profile-aware restore after retry escalation ─────────────────────────
    //
    // After `restore_default_settings()`, the tolerances reset to 1e-6 (the
    // hardcoded table values). `reapply_profile()` must then overwrite
    // them with the profile's values so HiGHS state and `current_profile` agree.

    /// Primal tolerance survives restore+profile sequence.
    ///
    /// Sets the profile primal tolerance to a non-default value (3e-8), then
    /// calls the combined `restore_defaults_then_reapply_profile_for_test` helper
    /// (which mirrors the finalization path in `retry_escalation`). The FFI
    /// read-back must return 3e-8 (the profile value), not 1e-6 (the default
    /// table value), proving that `reapply_profile` wins.
    #[test]
    fn profile_primal_tolerance_restored_after_retry_finalization() {
        let mut solver = make_solver();
        solver.apply_profile(&HighsProfile {
            primal_feasibility_tolerance: 3e-8,
            ..Default::default()
        });

        solver.restore_defaults_then_reapply_profile_for_test();

        let tol = solver
            .get_double_option(c"primal_feasibility_tolerance")
            .expect("primal_feasibility_tolerance must be readable");

        assert!(
            (tol - 3e-8).abs() < 1e-20,
            "primal tolerance must be 3e-8 (profile value) after restore+profile, got {tol}"
        );
    }

    /// Dual tolerance survives restore+profile sequence.
    ///
    /// Sets the profile dual tolerance to 5e-9 (below the 1e-6 default),
    /// simulates the retry finalization path, and verifies the profile value wins.
    #[test]
    fn profile_dual_tolerance_restored_after_retry_finalization() {
        let mut solver = make_solver();
        solver.apply_profile(&HighsProfile {
            dual_feasibility_tolerance: 5e-9,
            ..Default::default()
        });

        solver.restore_defaults_then_reapply_profile_for_test();

        let tol = solver
            .get_double_option(c"dual_feasibility_tolerance")
            .expect("dual_feasibility_tolerance must be readable");

        assert!(
            (tol - 5e-9).abs() < 1e-20,
            "dual tolerance must be 5e-9 (profile value) after restore+profile, got {tol}"
        );
    }

    // ── Full-field profile-seam readback ─────────────────────────────────────
    //
    // Power statement: catches a `HighsProfile` field that `reapply_profile`
    // forgets to re-install after `restore_default_settings` — the seam the two
    // tolerance-readback tests above only sample. It does NOT catch a
    // HiGHS-internal option the profile never carries, and the `presolve`/
    // `use_warm_start` fields are exercised through `apply_profile` but not
    // read back here — `HighsSolver`'s test-support surface has no
    // string/bool option getter (only the fields readable via
    // `get_double_option`/`get_int_option` are asserted below).

    /// Every int/double-valued field `reapply_profile` restores survives
    /// `restore_default_settings` followed by `reapply_profile`, with every
    /// profile field (including `presolve`/`use_warm_start`) installed at a
    /// non-default value simultaneously.
    #[test]
    fn full_profile_survives_retry_finalization_seam() {
        let mut solver = make_solver();
        let profile = HighsProfile {
            primal_feasibility_tolerance: 1e-7,
            dual_feasibility_tolerance: 1e-7,
            simplex_iteration_limit: 50_000,
            ipm_iteration_limit: 5_000,
            simplex_dual_edge_weight_strategy: 0, // Dantzig
            simplex_scale_strategy: 2,            // Curtis-Reid
            simplex_price_strategy: 2,            // RowHyperSparse
            presolve: PresolveKind::Off,
            simplex_update_limit: 1000,
            cost_perturbation: 1.0,
            refactor_error_tolerance: 1e-5,
            factor_pivot_threshold: 0.2,
            use_warm_start: false,
            steepest_edge_devex_fallback_threshold: 20.0,
        };
        assert_ne!(
            profile,
            HighsProfile::default(),
            "fixture profile must differ from every HighsProfile::default() field"
        );
        solver.apply_profile(&profile);

        solver.restore_defaults_then_reapply_profile_for_test();

        let primal = solver
            .get_double_option(c"primal_feasibility_tolerance")
            .expect("primal_feasibility_tolerance must be readable");
        let dual = solver
            .get_double_option(c"dual_feasibility_tolerance")
            .expect("dual_feasibility_tolerance must be readable");
        let dual_edge_weight = solver
            .get_int_option(c"simplex_dual_edge_weight_strategy")
            .expect("simplex_dual_edge_weight_strategy must be readable");
        let scale = solver
            .get_int_option(c"simplex_scale_strategy")
            .expect("simplex_scale_strategy must be readable");
        let price = solver
            .get_int_option(c"simplex_price_strategy")
            .expect("simplex_price_strategy must be readable");
        let simplex_update_limit = solver
            .get_int_option(c"simplex_update_limit")
            .expect("simplex_update_limit must be readable");
        let cost_perturbation = solver
            .get_double_option(c"dual_simplex_cost_perturbation_multiplier")
            .expect("dual_simplex_cost_perturbation_multiplier must be readable");
        let refactor_error_tolerance = solver
            .get_double_option(c"rebuild_refactor_solution_error_tolerance")
            .expect("rebuild_refactor_solution_error_tolerance must be readable");
        let factor_pivot_threshold = solver
            .get_double_option(c"factor_pivot_threshold")
            .expect("factor_pivot_threshold must be readable");
        let steepest_edge_devex_fallback_threshold = solver
            .get_double_option(c"dual_steepest_edge_weight_log_error_threshold")
            .expect("dual_steepest_edge_weight_log_error_threshold must be readable");

        assert!(
            (primal - profile.primal_feasibility_tolerance).abs() < 1e-20,
            "primal tolerance must survive the finalization seam; expected {}, got {primal}",
            profile.primal_feasibility_tolerance
        );
        assert!(
            (dual - profile.dual_feasibility_tolerance).abs() < 1e-20,
            "dual tolerance must survive the finalization seam; expected {}, got {dual}",
            profile.dual_feasibility_tolerance
        );
        assert_eq!(
            dual_edge_weight, profile.simplex_dual_edge_weight_strategy,
            "simplex_dual_edge_weight_strategy must survive the finalization seam"
        );
        assert_eq!(
            scale, profile.simplex_scale_strategy,
            "simplex_scale_strategy must survive the finalization seam"
        );
        assert_eq!(
            price, profile.simplex_price_strategy,
            "simplex_price_strategy must survive the finalization seam"
        );
        #[allow(clippy::cast_possible_wrap)]
        let expected_simplex_update_limit = profile.simplex_update_limit as i32;
        assert_eq!(
            simplex_update_limit, expected_simplex_update_limit,
            "simplex_update_limit must survive the finalization seam"
        );
        assert!(
            (cost_perturbation - profile.cost_perturbation).abs() < 1e-20,
            "cost_perturbation must survive the finalization seam; expected {}, got {cost_perturbation}",
            profile.cost_perturbation
        );
        assert!(
            (refactor_error_tolerance - profile.refactor_error_tolerance).abs() < 1e-20,
            "refactor_error_tolerance must survive the finalization seam; expected {}, got {refactor_error_tolerance}",
            profile.refactor_error_tolerance
        );
        assert!(
            (factor_pivot_threshold - profile.factor_pivot_threshold).abs() < 1e-20,
            "factor_pivot_threshold must survive the finalization seam; expected {}, got {factor_pivot_threshold}",
            profile.factor_pivot_threshold
        );
        assert!(
            (steepest_edge_devex_fallback_threshold
                - profile.steepest_edge_devex_fallback_threshold)
                .abs()
                < 1e-20,
            "steepest_edge_devex_fallback_threshold must survive the finalization seam; expected {}, got {steepest_edge_devex_fallback_threshold}",
            profile.steepest_edge_devex_fallback_threshold
        );
    }

    // ── IPM iteration_limit sentinel for "unbounded" ─────────────────────────
    //
    // When `ipm_iteration_limit == 0` (DEFAULT_PROFILE_IPM_UNBOUNDED_SENTINEL),
    // `set_iteration_limits` must pass `i32::MAX` to HiGHS — not 0, which HiGHS
    // would interpret as "no iterations allowed".

    /// `ipm_iteration_limit = 0` maps to `i32::MAX` at the FFI layer.
    ///
    /// Sets the profile IPM cap to 0 (the "unbounded" sentinel), triggers
    /// `set_iteration_limits` via `apply_retry_level_options(0)`, and reads
    /// back the `HiGHS` option. The value must be `i32::MAX`, not 0.
    #[test]
    fn ipm_sentinel_zero_maps_to_i32_max() {
        let mut solver = make_solver();
        solver.apply_profile(&HighsProfile {
            ipm_iteration_limit: 0,
            ..Default::default()
        });

        solver.apply_retry_level_options_for_test(0);

        let cap = solver
            .get_int_option(c"ipm_iteration_limit")
            .expect("ipm_iteration_limit must be readable");

        assert_eq!(
            cap,
            i32::MAX,
            "ipm_iteration_limit sentinel 0 must produce i32::MAX; got {cap}"
        );
    }

    // ── Delta-only profile dispatch: solve() does not re-apply ───────────────
    //
    // ProfiledSolver installs the active profile at phase boundaries (delta-gated
    // by PartialEq) and on the retry path, but solve() does NOT re-apply it: on
    // the hot path the inner solver's options already equal the active profile, so
    // re-applying every solve would issue redundant FFI option setters.

    /// Counts calls to `apply_profile`.
    struct SetterCountMock {
        apply_profile_call_count: Cell<usize>,
    }

    impl SetterCountMock {
        fn new() -> Self {
            Self {
                apply_profile_call_count: Cell::new(0),
            }
        }
    }

    impl SolverInterface for SetterCountMock {
        type Profile = HighsProfile;

        fn apply_profile(&mut self, _profile: &HighsProfile) {
            self.apply_profile_call_count
                .set(self.apply_profile_call_count.get() + 1);
        }

        fn load_model(&mut self, _t: &StageTemplate) {}
        fn add_rows(&mut self, _r: &RowBatch) {}
        fn set_row_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
        fn set_col_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
        fn solve(&mut self, _b: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
            Err(SolverError::InternalError {
                message: "mock".to_string(),
                error_code: None,
            })
        }
        fn get_basis(&mut self, _o: &mut Basis) {}
        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }
        fn statistics_into(&self, out: &mut SolverStatistics) {
            out.copy_from(&SolverStatistics::default());
        }
        fn name(&self) -> &'static str {
            "SetterCountMock"
        }
        fn solver_name_version(&self) -> String {
            "SetterCountMock 0.0.0".to_string()
        }
    }

    /// Delta-only dispatch: once a profile is installed via `set_profile`,
    /// subsequent `solve()` calls with no intervening profile change dispatch
    /// no further `apply_profile` calls to the inner solver, and re-installing
    /// the identical profile is a no-op.
    #[test]
    fn solve_does_not_reapply_unchanged_profile() {
        let mock = SetterCountMock::new();
        let mut solver = ProfiledSolver::new(mock);

        let non_default = HighsProfile {
            primal_feasibility_tolerance: 1e-7,
            dual_feasibility_tolerance: 1e-7,
            simplex_iteration_limit: 50_000,
            ipm_iteration_limit: 5_000,
            simplex_dual_edge_weight_strategy: 0,
            simplex_scale_strategy: 0,
            simplex_price_strategy: 2,
            presolve: PresolveKind::Off,
            simplex_update_limit: 1000,
            cost_perturbation: 1.0,
            refactor_error_tolerance: 1e-5,
            factor_pivot_threshold: 0.2,
            use_warm_start: false,
            steepest_edge_devex_fallback_threshold: 20.0,
        };
        solver.set_profile(&non_default);

        // Reset the counter `set_profile` just bumped, to measure only the
        // solve-triggered calls below.
        solver.inner_mut().apply_profile_call_count.set(0);

        for _ in 0..3 {
            let _ = solver.solve(None);
        }

        let solve_count = solver.inner().apply_profile_call_count.get();
        assert_eq!(
            solve_count, 0,
            "solve() must not re-apply an unchanged profile; expected 0, got {solve_count}"
        );

        solver.set_profile(&non_default);
        let after_reset = solver.inner().apply_profile_call_count.get();
        assert_eq!(
            after_reset, 0,
            "set_profile with an unchanged profile must be a no-op; expected 0, got {after_reset}"
        );
    }
}
