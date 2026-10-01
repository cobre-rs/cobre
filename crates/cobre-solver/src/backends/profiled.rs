//! Generic `ProfiledSolver<S>` wrapper with per-phase LP-solver configuration.
//!
//! [`ProfiledSolver`] wraps any [`SolverInterface`] implementor and skips FFI
//! option-setter calls when the new profile matches the current one (delta-only
//! dispatch via `PartialEq`).
//!
//! # Usage
//!
//! ```rust
//! # #[cfg(feature = "highs")] {
//! use cobre_solver::{HighsProfile, ProfiledSolver, SolverInterface, HighsSolver};
//!
//! let inner = HighsSolver::new().expect("HiGHS init");
//! let mut solver = ProfiledSolver::new(inner);
//! solver.set_profile(&HighsProfile::default());
//! assert_eq!(solver.current_profile(), &HighsProfile::default());
//! # }
//! ```

use crate::{
    SolverInterface,
    types::{Basis, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate},
};

/// Wraps any [`SolverInterface`] implementor with per-phase profile
/// configuration, forwarding every trait method to the inner solver.
///
/// Monomorphized over `S` (not `Box<dyn>`) to keep forwarding zero-cost on the
/// hot path.
pub struct ProfiledSolver<S: SolverInterface> {
    inner: S,
    current_profile: S::Profile,
}

impl<S: SolverInterface> ProfiledSolver<S> {
    /// Wrap an existing solver with the default profile.
    ///
    /// Issues no FFI calls: the inner solver is assumed already consistent with
    /// `S::Profile::default()`.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            current_profile: S::Profile::default(),
        }
    }

    /// Apply a new profile to the inner solver, skipping the FFI calls when it
    /// equals `current_profile`.
    ///
    /// Call once per phase boundary, NOT inside the hot solve loop — per-solve
    /// calls defeat the delta-only dispatch.
    pub fn set_profile(&mut self, profile: &S::Profile) {
        if *profile == self.current_profile {
            return;
        }
        self.apply_profile(profile);
    }

    /// The currently applied profile.
    pub fn current_profile(&self) -> &S::Profile {
        &self.current_profile
    }

    /// Shared reference to the wrapped inner solver, for test/adapter inspection.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Exclusive reference to the wrapped inner solver, for test/adapter mutation.
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }
}

impl<S: SolverInterface> SolverInterface for ProfiledSolver<S> {
    type Profile = S::Profile;

    fn apply_profile(&mut self, profile: &S::Profile) {
        self.inner.apply_profile(profile);
        self.current_profile = *profile;
    }

    fn load_model(&mut self, template: &StageTemplate) {
        self.inner.load_model(template);
    }

    fn add_rows(&mut self, rows: &RowBatch) {
        self.inner.add_rows(rows);
    }

    fn set_row_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
        self.inner.set_row_bounds(indices, lower, upper);
    }

    fn set_col_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
        self.inner.set_col_bounds(indices, lower, upper);
    }

    fn solve(&mut self, basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
        // Forward directly without re-applying the profile: the inner options
        // already equal `current_profile` on every solve (set at phase
        // boundaries; the retry path restores them after a default reset).
        self.inner.solve(basis)
    }

    fn get_basis(&mut self, out: &mut Basis) {
        self.inner.get_basis(out);
    }

    fn statistics(&self) -> SolverStatistics {
        self.inner.statistics()
    }

    fn statistics_into(&self, out: &mut SolverStatistics) {
        self.inner.statistics_into(out);
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn solver_name_version(&self) -> String {
        self.inner.solver_name_version()
    }

    fn record_reconstruction_stats(&mut self) {
        self.inner.record_reconstruction_stats();
    }

    fn reset_solver_state(&mut self) {
        self.inner.reset_solver_state();
    }
}

// Gated on `highs`: the delta-tracking assertions need a profile with fields,
// and exercising the backend-agnostic logic on one backend is sufficient.
#[cfg(all(test, feature = "highs"))]
mod tests {
    use std::cell::RefCell;

    use super::ProfiledSolver;
    use crate::{
        HighsProfile, PresolveKind, SolverInterface,
        types::{Basis, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate},
    };

    // ── RecordingMockSolver ───────────────────────────────────────────────────

    #[derive(Debug, Clone, PartialEq)]
    enum RecordedCall {
        LoadModel,
        AddRows,
        SetRowBounds,
        SetColBounds,
        Solve,
        ApplyProfile(HighsProfile),
    }

    /// Records every [`SolverInterface`] invocation into a call log; `solve`
    /// always returns `Err` because the mock is not a real LP solver.
    struct RecordingMockSolver {
        calls: RefCell<Vec<RecordedCall>>,
    }

    impl RecordingMockSolver {
        fn new() -> Self {
            Self {
                calls: RefCell::new(Vec::new()),
            }
        }

        pub(crate) fn recorded_calls(&self) -> Vec<RecordedCall> {
            self.calls.borrow().clone()
        }
    }

    impl SolverInterface for RecordingMockSolver {
        type Profile = HighsProfile;

        fn apply_profile(&mut self, profile: &HighsProfile) {
            self.calls
                .borrow_mut()
                .push(RecordedCall::ApplyProfile(*profile));
        }

        fn load_model(&mut self, _template: &StageTemplate) {
            self.calls.borrow_mut().push(RecordedCall::LoadModel);
        }

        fn add_rows(&mut self, _rows: &RowBatch) {
            self.calls.borrow_mut().push(RecordedCall::AddRows);
        }

        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {
            self.calls.borrow_mut().push(RecordedCall::SetRowBounds);
        }

        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {
            self.calls.borrow_mut().push(RecordedCall::SetColBounds);
        }

        fn solve(&mut self, _basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
            self.calls.borrow_mut().push(RecordedCall::Solve);
            Err(SolverError::InternalError {
                message: "mock".to_string(),
                error_code: None,
            })
        }

        fn get_basis(&mut self, _out: &mut Basis) {}

        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }

        fn statistics_into(&self, out: &mut SolverStatistics) {
            out.copy_from(&SolverStatistics::default());
        }

        fn name(&self) -> &'static str {
            "RecordingMock"
        }

        fn solver_name_version(&self) -> String {
            "RecordingMockSolver 0.0.0".to_string()
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn filter_profile_calls(calls: &[RecordedCall]) -> Vec<&RecordedCall> {
        calls
            .iter()
            .filter(|c| matches!(c, RecordedCall::ApplyProfile(_)))
            .collect()
    }

    fn make_test_template() -> StageTemplate {
        StageTemplate {
            num_cols: 1,
            num_rows: 0,
            num_nz: 0,
            col_starts: vec![0_i32, 0],
            row_indices: vec![],
            values: vec![],
            col_lower: vec![0.0],
            col_upper: vec![1.0],
            objective: vec![0.0],
            row_lower: vec![],
            row_upper: vec![],
            n_state: 0,
            col_scale: vec![],
            row_scale: vec![],
        }
    }

    fn make_test_row_batch() -> RowBatch {
        RowBatch {
            num_rows: 0,
            row_starts: vec![0_i32],
            col_indices: vec![],
            values: vec![],
            row_lower: vec![],
            row_upper: vec![],
        }
    }

    #[test]
    fn new_issues_no_ffi_calls() {
        let mock = RecordingMockSolver::new();
        let solver = ProfiledSolver::new(mock);
        let calls = solver.inner.recorded_calls();
        assert!(
            calls.is_empty(),
            "expected zero calls after ProfiledSolver::new, got: {calls:?}"
        );
    }

    #[test]
    fn set_profile_noop_when_unchanged() {
        let mock = RecordingMockSolver::new();
        let mut solver = ProfiledSolver::new(mock);

        solver.set_profile(&HighsProfile::default());

        let calls = solver.inner.recorded_calls();
        let profile_calls = filter_profile_calls(&calls);
        assert!(
            profile_calls.is_empty(),
            "expected zero apply_profile calls when profile unchanged, got: {profile_calls:?}"
        );
    }

    /// Each per-field change dispatches a single `apply_profile` carrying the
    /// whole profile — there is no per-field dispatch; the noop guard is a
    /// `PartialEq` on the whole profile.
    #[test]
    fn set_profile_dispatches_apply_profile_when_changed() {
        let cases = [
            (
                "primal-only",
                HighsProfile {
                    primal_feasibility_tolerance: 1e-7,
                    ..HighsProfile::default()
                },
            ),
            (
                "dual-only",
                HighsProfile {
                    dual_feasibility_tolerance: 1e-7,
                    ..HighsProfile::default()
                },
            ),
            (
                "simplex-cap-only",
                HighsProfile {
                    simplex_iteration_limit: 50_000,
                    ..HighsProfile::default()
                },
            ),
            (
                "ipm-cap-only",
                HighsProfile {
                    ipm_iteration_limit: 5_000,
                    ..HighsProfile::default()
                },
            ),
            (
                "dual-edge-weight-only",
                HighsProfile {
                    simplex_dual_edge_weight_strategy: 0, // Dantzig
                    ..HighsProfile::default()
                },
            ),
            (
                "price-strategy-only",
                HighsProfile {
                    simplex_price_strategy: 2, // RowHyperSparse
                    ..HighsProfile::default()
                },
            ),
        ];
        for (label, p) in cases {
            let mock = RecordingMockSolver::new();
            let mut solver = ProfiledSolver::new(mock);
            solver.set_profile(&p);
            let calls = solver.inner.recorded_calls();
            let profile_calls = filter_profile_calls(&calls);
            assert_eq!(
                profile_calls,
                vec![&RecordedCall::ApplyProfile(p)],
                "expected one ApplyProfile(p) for {label} change"
            );
        }
    }

    #[test]
    fn set_profile_full_change_dispatches_single_apply_profile() {
        let mock = RecordingMockSolver::new();
        let mut solver = ProfiledSolver::new(mock);

        let p = HighsProfile {
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
        solver.set_profile(&p);

        let calls = solver.inner.recorded_calls();
        let profile_calls: Vec<_> = filter_profile_calls(&calls).into_iter().cloned().collect();

        assert_eq!(
            profile_calls,
            vec![RecordedCall::ApplyProfile(p)],
            "expected exactly one ApplyProfile call with the complete profile"
        );
    }

    /// `solve` forwards directly without re-applying the profile (installed at
    /// phase boundaries via `set_profile`, not per solve).
    #[test]
    fn solver_interface_methods_forward_to_inner() {
        let mock = RecordingMockSolver::new();
        let mut solver = ProfiledSolver::new(mock);

        let template = make_test_template();
        let rows = make_test_row_batch();

        solver.load_model(&template);
        solver.add_rows(&rows);
        solver.set_row_bounds(&[], &[], &[]);
        solver.set_col_bounds(&[], &[], &[]);
        let _ = solver.solve(None);

        let calls = solver.inner.recorded_calls();
        assert!(
            calls.contains(&RecordedCall::LoadModel),
            "expected LoadModel in call log, got: {calls:?}"
        );
        assert!(
            calls.contains(&RecordedCall::AddRows),
            "expected AddRows in call log, got: {calls:?}"
        );
        assert!(
            calls.contains(&RecordedCall::SetRowBounds),
            "expected SetRowBounds in call log, got: {calls:?}"
        );
        assert!(
            calls.contains(&RecordedCall::SetColBounds),
            "expected SetColBounds in call log, got: {calls:?}"
        );
        assert!(
            calls.contains(&RecordedCall::Solve),
            "expected Solve in call log, got: {calls:?}"
        );
        let profile_calls = filter_profile_calls(&calls);
        assert!(
            profile_calls.is_empty(),
            "solve() must not trigger an ApplyProfile call, got: {calls:?}"
        );
    }
}
