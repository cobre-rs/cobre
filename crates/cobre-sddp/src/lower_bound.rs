//! Lower bound evaluation for iterative LP-based optimization algorithms.
//!
//! [`evaluate_lower_bound`] computes the risk-adjusted lower bound by solving
//! the stage-0 LP for every opening in the scenario tree and aggregating the
//! per-opening objectives through the stage-0 risk measure. The result is
//! broadcast from rank 0 to all other ranks so that every rank holds the same
//! global lower bound value.
//!
//! ## Algorithm
//!
//! 1. Rank 0 iterates over all `n_openings` openings at stage 0.
//! 2. For each opening the LP is rebuilt: `load_model` → `add_rows(cut_batch)`
//!    → `fill_forward_patches` → `set_row_bounds` → `solve`.
//! 3. The per-opening objectives are aggregated by the risk measure using
//!    uniform probabilities `1 / n_openings`.
//! 4. The scalar lower bound is broadcast from rank 0 to all ranks.
//!
//! ## Correctness notes
//!
//! - The cut batch is built **once** before the opening loop because it does
//!   not change per opening.
//! - `fill_forward_patches` is used (not `fill_state_patches`) because each
//!   opening carries different noise values that must be patched into the LP.
//! - The function must be called **after** the backward pass and cut sync so
//!   that the FCF holds the latest cuts when the LPs are solved.
//! - Stage 0 should never be infeasible (the penalty system guarantees recourse),
//!   so `SddpError::Infeasible` from this function indicates a modelling error.

use cobre_comm::Communicator;
use cobre_solver::{SolverError, SolverInterface};
use cobre_stochastic::OpeningTree;

use crate::{
    FutureCostFunction, PatchBuffer, RiskMeasure, SddpError, StageIndexer,
    forward::build_cut_row_batch,
};
use cobre_solver::StageTemplate;

/// Evaluate the global lower bound for the current FCF approximation.
///
/// On rank 0 the function iterates over all stage-0 openings, solves the LP
/// for each, and applies the risk measure to produce a risk-adjusted scalar
/// lower bound. The scalar is then broadcast to all ranks.
///
/// ## Arguments
///
/// - `solver` — Mutable LP solver instance. Only rank 0 calls `solve`; other
///   ranks skip the opening loop entirely.
/// - `template` — Structural LP for stage 0, shared read-only.
/// - `fcf` — Future Cost Function with all accumulated cuts.
/// - `initial_state` — Known initial state vector `x_0` (length `indexer.n_state`).
/// - `base_row` — AR-dynamics base row index for stage 0, used by
///   [`PatchBuffer::fill_forward_patches`].
/// - `indexer` — LP layout map for stage 0.
/// - `patch_buf` — Reusable patch buffer (must be sized for this stage).
/// - `opening_tree` — Scenario tree providing per-opening noise vectors.
/// - `risk_measure` — Risk measure for stage 0 (determines probability weighting).
/// - `comm` — Communicator for rank/size queries and broadcast.
///
/// ## Returns
///
/// `Ok(lb)` where `lb` is the risk-adjusted lower bound (finite when all
/// stage-0 LPs have finite objectives). Returns `Err` on LP infeasibility,
/// other solver failure, or communication error.
///
/// ## Errors
///
/// - [`SddpError::Infeasible`] — Stage-0 LP has no feasible solution for an
///   opening. This indicates a modelling error; stage 0 should always be
///   feasible due to the penalty/recourse structure.
/// - [`SddpError::Solver`] — LP solve failed for a reason other than
///   infeasibility (numerical difficulty, time limit, etc.).
/// - [`SddpError::Communication`] — The broadcast to non-root ranks failed.
///
/// ## Panics
///
/// Panics if `opening_tree.n_openings(0) == 0` on rank 0. Stage 0 must have
/// at least one opening; this is a caller contract violation.
///
/// ## Example
///
/// ```rust,ignore
/// // Illustrative usage (not compiled in doctests: requires full solver setup).
/// let lb = evaluate_lower_bound(
///     &mut solver, &template, &fcf, &initial_state,
///     base_row, &indexer, &mut patch_buf,
///     &opening_tree, &risk_measure, &comm,
/// )?;
/// ```
#[allow(clippy::too_many_arguments)]
pub fn evaluate_lower_bound<S: SolverInterface, C: Communicator>(
    solver: &mut S,
    template: &StageTemplate,
    fcf: &FutureCostFunction,
    initial_state: &[f64],
    base_row: usize,
    indexer: &StageIndexer,
    patch_buf: &mut PatchBuffer,
    opening_tree: &OpeningTree,
    risk_measure: &RiskMeasure,
    comm: &C,
) -> Result<f64, SddpError> {
    let mut lb = 0.0_f64;

    if comm.rank() == 0 {
        let n_openings = opening_tree.n_openings(0);
        assert!(
            n_openings > 0,
            "evaluate_lower_bound: stage 0 must have at least one opening"
        );

        // Build cut rows once — the cut batch does not change per opening.
        let cut_batch = build_cut_row_batch(fcf, 0, indexer);

        let mut objectives = Vec::with_capacity(n_openings);

        for opening_idx in 0..n_openings {
            solver.load_model(template);

            if cut_batch.num_rows > 0 {
                solver.add_rows(&cut_batch);
            }

            let noise = opening_tree.opening(0, opening_idx);
            patch_buf.fill_forward_patches(indexer, initial_state, noise, base_row);

            let n_patches = patch_buf.forward_patch_count();
            solver.set_row_bounds(
                &patch_buf.indices[..n_patches],
                &patch_buf.lower[..n_patches],
                &patch_buf.upper[..n_patches],
            );

            let view = solver.solve().map_err(|e| match e {
                SolverError::Infeasible => SddpError::Infeasible {
                    stage: 0,
                    iteration: 0,
                    scenario: opening_idx,
                },
                other => SddpError::Solver(other),
            })?;

            objectives.push(view.objective);
        }

        #[allow(clippy::cast_precision_loss)]
        let uniform_prob = 1.0_f64 / n_openings as f64;
        let probs = vec![uniform_prob; n_openings];
        lb = risk_measure.evaluate_risk(&objectives, &probs);
    }

    comm.broadcast(std::slice::from_mut(&mut lb), 0)
        .map_err(SddpError::from)?;
    Ok(lb)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]
mod tests {
    use super::evaluate_lower_bound;
    use crate::{FutureCostFunction, PatchBuffer, RiskMeasure, SddpError, StageIndexer};
    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
    use cobre_solver::{
        Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };
    use cobre_stochastic::OpeningTree;

    /// Minimal stage template for N=1 hydro, L=0 PAR order.
    ///
    /// Column layout: [storage (0), `storage_in` (1), theta (2)]
    /// Row layout: [`storage_fixing` (0)]
    fn minimal_template() -> StageTemplate {
        StageTemplate {
            num_cols: 3,
            num_rows: 1,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 1, 1], // col 1 (storage_in) has NZ at row 0
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
            objective: vec![0.0, 0.0, 1.0], // minimise theta
            row_lower: vec![0.0],
            row_upper: vec![0.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
        }
    }

    /// Build an `OpeningTree` with `n_openings` openings at stage 0.
    ///
    /// Uses `generate_opening_tree` with a single-entity identity-correlation
    /// model. Because `MockSolver` ignores the noise values returned by
    /// `opening_tree.opening(...)`, the tree only needs to have the right shape
    /// (correct stage count and branching factor at stage 0).
    ///
    /// `dim = 1` throughout (single hydro, `StageIndexer::new(1, 0)`).
    fn simple_opening_tree(n_openings: usize) -> OpeningTree {
        use chrono::NaiveDate;
        use cobre_core::{
            EntityId,
            scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
            temporal::{
                Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
                StageStateConfig,
            },
        };
        use cobre_stochastic::correlation::resolve::DecomposedCorrelation;
        use std::collections::BTreeMap;

        // Single study stage with the requested branching factor.
        let stage = Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: n_openings,
                noise_method: NoiseMethod::Saa,
            },
        };

        let entity_id = EntityId(1);
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g1".to_string(),
                    entities: vec![CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: entity_id,
                    }],
                    matrix: vec![vec![1.0]],
                }],
            },
        );
        let corr_model = CorrelationModel {
            method: "cholesky".to_string(),
            profiles,
            schedule: vec![],
        };
        let mut decomposed = DecomposedCorrelation::build(&corr_model).unwrap();
        let entity_order = vec![entity_id];

        cobre_stochastic::tree::generate::generate_opening_tree(
            42,
            &[stage],
            1, // dim = 1 hydro
            &mut decomposed,
            &entity_order,
        )
    }

    // ── Mock communicator ────────────────────────────────────────────────────

    /// Single-rank stub communicator. broadcast is a no-op (identity operation).
    struct LocalComm;

    impl Communicator for LocalComm {
        fn allgatherv<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            unreachable!("LocalComm allgatherv not used in lower_bound tests")
        }

        fn allreduce<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            unreachable!("LocalComm allreduce not used in lower_bound tests")
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            // Single rank: no-op. The value is already in buf from the rank-0 computation.
            Ok(())
        }

        fn barrier(&self) -> Result<(), CommError> {
            Ok(())
        }

        fn rank(&self) -> usize {
            0
        }

        fn size(&self) -> usize {
            1
        }
    }

    /// Communicator that fails on `broadcast` with `CommError::CollectiveFailed`.
    struct FailingBcastComm;

    impl Communicator for FailingBcastComm {
        fn allgatherv<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            unreachable!()
        }

        fn allreduce<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            unreachable!()
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            Err(CommError::CollectiveFailed {
                operation: "broadcast",
                mpi_error_code: -1,
                message: "test-induced broadcast failure".to_string(),
            })
        }

        fn barrier(&self) -> Result<(), CommError> {
            Ok(())
        }

        fn rank(&self) -> usize {
            0
        }

        fn size(&self) -> usize {
            1
        }
    }

    // ── Mock solver ──────────────────────────────────────────────────────────

    /// Mock solver that returns configurable objective values in sequence.
    ///
    /// Each call to `solve()` returns the next value from `objectives`. If
    /// `infeasible_on_call` is set and the call index matches, returns
    /// `SolverError::Infeasible` instead.
    struct MockSolver {
        objectives: Vec<f64>,
        call_count: usize,
        infeasible_on_call: Option<usize>,
    }

    impl MockSolver {
        fn with_objectives(objectives: Vec<f64>) -> Self {
            Self {
                objectives,
                call_count: 0,
                infeasible_on_call: None,
            }
        }

        fn infeasible_on_first() -> Self {
            Self {
                objectives: vec![0.0],
                call_count: 0,
                infeasible_on_call: Some(0),
            }
        }
    }

    impl SolverInterface for MockSolver {
        fn load_model(&mut self, _template: &StageTemplate) {}
        fn add_rows(&mut self, _cuts: &RowBatch) {}
        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn solve(&mut self) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            let call = self.call_count;
            self.call_count += 1;
            if self.infeasible_on_call == Some(call) {
                return Err(SolverError::Infeasible);
            }
            let obj = self.objectives[call % self.objectives.len()];
            // Return a minimal view; evaluate_lower_bound only reads `view.objective`.
            // Use static empty slices for primal/dual/reduced_costs.
            Ok(cobre_solver::SolutionView {
                objective: obj,
                primal: &[],
                dual: &[],
                reduced_costs: &[],
                iterations: 0,
                solve_time_seconds: 0.0,
            })
        }

        fn reset(&mut self) {
            self.call_count = 0;
        }

        fn get_basis(&mut self, _out: &mut Basis) {}

        fn solve_with_basis(
            &mut self,
            _basis: &Basis,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            self.solve()
        }

        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }

        fn name(&self) -> &'static str {
            "Mock"
        }
    }

    // ── Shared test setup ────────────────────────────────────────────────────

    fn make_fcf(n_stages: usize, n_state: usize) -> FutureCostFunction {
        // max_cuts=100, n_transfer=0
        FutureCostFunction::new(n_stages, n_state, 2, 100, 0)
    }

    // ── Unit tests ───────────────────────────────────────────────────────────

    /// AC1: 1 opening, Expectation — LB equals the single LP objective.
    #[test]
    fn one_opening_expectation_lb_equals_single_objective() {
        let indexer = StageIndexer::new(1, 0); // n_state=1, hydro_count=1
        let template = minimal_template();
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order);
        let opening_tree = simple_opening_tree(1);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let mut solver = MockSolver::with_objectives(vec![100.0]);

        let lb = evaluate_lower_bound(
            &mut solver,
            &template,
            &fcf,
            &initial_state,
            1, // base_row
            &indexer,
            &mut patch_buf,
            &opening_tree,
            &rm,
            &comm,
        )
        .unwrap();

        assert!(
            (lb - 100.0).abs() < 1e-10,
            "single opening expectation LB must equal objective 100.0, got {lb}"
        );
    }

    /// AC2: 3 openings, Expectation — LB equals mean of objectives.
    #[test]
    fn three_openings_expectation_lb_equals_mean() {
        let indexer = StageIndexer::new(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order);
        let opening_tree = simple_opening_tree(3);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        // Solver returns 60, 80, 100 for the three openings.
        let mut solver = MockSolver::with_objectives(vec![60.0, 80.0, 100.0]);

        let lb = evaluate_lower_bound(
            &mut solver,
            &template,
            &fcf,
            &initial_state,
            1,
            &indexer,
            &mut patch_buf,
            &opening_tree,
            &rm,
            &comm,
        )
        .unwrap();

        // E[60, 80, 100] with uniform probs = (60+80+100)/3 = 80.0
        assert!(
            (lb - 80.0).abs() < 1e-10,
            "three openings expectation LB must equal 80.0, got {lb}"
        );
    }

    /// AC3: 2 openings, CVaR(alpha=0.5, lambda=1.0) — pure `CVaR` selects worst.
    #[test]
    fn two_openings_pure_cvar_alpha_half_lb_equals_worst() {
        let indexer = StageIndexer::new(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order);
        let opening_tree = simple_opening_tree(2);
        // CVaR(alpha=0.5, lambda=1.0): pure CVaR; upper bound per scenario =
        // p / alpha = 0.5 / 0.5 = 1.0. With 2 equal-probability scenarios the
        // greedy allocation places all mass on the worst scenario.
        let rm = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 1.0,
        };
        let comm = LocalComm;

        // Solver returns 50, 150.
        let mut solver = MockSolver::with_objectives(vec![50.0, 150.0]);

        let lb = evaluate_lower_bound(
            &mut solver,
            &template,
            &fcf,
            &initial_state,
            1,
            &indexer,
            &mut patch_buf,
            &opening_tree,
            &rm,
            &comm,
        )
        .unwrap();

        // CVaR(alpha=0.5, lambda=1.0) with 2 uniform-probability openings
        // concentrates all weight on the worst (150.0).
        assert!(
            (lb - 150.0).abs() < 1e-10,
            "pure CVaR(0.5, 1.0) with 2 openings must equal 150.0, got {lb}"
        );
    }

    /// AC4 (extra): 2 openings, CVaR(alpha=1.0, lambda=1.0) = Expectation.
    #[test]
    fn two_openings_cvar_alpha_one_equals_expectation() {
        let indexer = StageIndexer::new(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order);
        let opening_tree = simple_opening_tree(2);
        let rm = RiskMeasure::CVaR {
            alpha: 1.0,
            lambda: 1.0,
        };
        let comm = LocalComm;

        let mut solver = MockSolver::with_objectives(vec![50.0, 150.0]);

        let lb = evaluate_lower_bound(
            &mut solver,
            &template,
            &fcf,
            &initial_state,
            1,
            &indexer,
            &mut patch_buf,
            &opening_tree,
            &rm,
            &comm,
        )
        .unwrap();

        // CVaR(alpha=1) = Expectation = (50+150)/2 = 100.0
        assert!(
            (lb - 100.0).abs() < 1e-10,
            "CVaR(alpha=1, lambda=1) must equal expectation 100.0, got {lb}"
        );
    }

    /// AC5: solver returns Infeasible for the first opening — must propagate as `SddpError::Infeasible`.
    #[test]
    fn infeasible_solve_maps_to_sddp_infeasible() {
        let indexer = StageIndexer::new(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order);
        let opening_tree = simple_opening_tree(1);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let mut solver = MockSolver::infeasible_on_first();

        let result = evaluate_lower_bound(
            &mut solver,
            &template,
            &fcf,
            &initial_state,
            1,
            &indexer,
            &mut patch_buf,
            &opening_tree,
            &rm,
            &comm,
        );

        assert!(
            matches!(result, Err(SddpError::Infeasible { stage: 0, .. })),
            "infeasible solver must produce SddpError::Infeasible at stage 0, got {result:?}"
        );
    }

    /// AC6: broadcast failure maps to `SddpError::Communication`.
    #[test]
    fn broadcast_failure_maps_to_communication_error() {
        let indexer = StageIndexer::new(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order);
        let opening_tree = simple_opening_tree(1);
        let rm = RiskMeasure::Expectation;
        let comm = FailingBcastComm;

        let mut solver = MockSolver::with_objectives(vec![100.0]);

        let result = evaluate_lower_bound(
            &mut solver,
            &template,
            &fcf,
            &initial_state,
            1,
            &indexer,
            &mut patch_buf,
            &opening_tree,
            &rm,
            &comm,
        );

        assert!(
            matches!(result, Err(SddpError::Communication(_))),
            "broadcast failure must produce SddpError::Communication, got {result:?}"
        );
    }

    // ── Integration tests ────────────────────────────────────────────────────

    /// Integration: full round-trip with `LocalComm` and 2 openings.
    ///
    /// Verifies that the function correctly integrates with `build_cut_row_batch`
    /// (`cut_batch` with 0 cuts still produces the right result), `fill_forward_patches`,
    /// and `RiskMeasure::Expectation`.
    #[test]
    fn integration_two_openings_local_backend_expectation() {
        let indexer = StageIndexer::new(1, 0);
        let template = minimal_template();
        // Start with 0 cuts (empty FCF).
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![50.0_f64]; // non-zero initial state
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order);
        let opening_tree = simple_opening_tree(2);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        let mut solver = MockSolver::with_objectives(vec![200.0, 300.0]);

        let lb = evaluate_lower_bound(
            &mut solver,
            &template,
            &fcf,
            &initial_state,
            1,
            &indexer,
            &mut patch_buf,
            &opening_tree,
            &rm,
            &comm,
        )
        .unwrap();

        // E[200, 300] = 250.0
        assert!(
            (lb - 250.0).abs() < 1e-10,
            "integration round-trip must produce 250.0, got {lb}"
        );
    }

    /// Integration: monotonicity — adding cuts can only increase the LB.
    ///
    /// This test calls `evaluate_lower_bound` twice: first with 0 cuts, then
    /// with objectives set higher (simulating tighter cuts). The second LB
    /// must be >= the first.
    #[test]
    fn integration_monotonicity_more_cuts_yields_higher_or_equal_lb() {
        let indexer = StageIndexer::new(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![0.0_f64];
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order);
        let opening_tree = simple_opening_tree(2);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        // First call: solver returns [50, 100] → LB = 75.
        let mut solver1 = MockSolver::with_objectives(vec![50.0, 100.0]);
        let lb1 = evaluate_lower_bound(
            &mut solver1,
            &template,
            &fcf,
            &initial_state,
            1,
            &indexer,
            &mut patch_buf,
            &opening_tree,
            &rm,
            &comm,
        )
        .unwrap();

        // Second call: solver returns [80, 120] → LB = 100 (tighter cuts raise obj).
        let mut solver2 = MockSolver::with_objectives(vec![80.0, 120.0]);
        let lb2 = evaluate_lower_bound(
            &mut solver2,
            &template,
            &fcf,
            &initial_state,
            1,
            &indexer,
            &mut patch_buf,
            &opening_tree,
            &rm,
            &comm,
        )
        .unwrap();

        assert!(
            lb2 >= lb1,
            "second LB ({lb2}) must be >= first LB ({lb1}) when cuts are tighter"
        );
    }
}
