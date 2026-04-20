//! Unified LP-solve entry point shared by the three hot-path drivers.
//!
//! [`run_stage_solve`] encapsulates basis reconstruction, invariant enforcement,
//! and the solver call so that `forward.rs`, `backward.rs`, and
//! `simulation/pipeline.rs` can delegate to a single implementation instead of
//! each maintaining their own copy. The body is filled in ticket 002; this
//! module freezes the public API shape that tickets 003–005 depend on.

use cobre_solver::{SolutionView, SolverInterface, StageTemplate};

use crate::{
    SddpError, StageIndexer,
    basis_reconstruct::{
        PaddingContext, ReconstructionStats, ReconstructionTarget, enforce_basic_count_invariant,
        reconstruct_basis,
    },
    context::StageContext,
    cut::pool::CutPool,
    workspace::{CapturedBasis, SolverWorkspace},
};

use cobre_solver::SolverError;

// ---------------------------------------------------------------------------
// Phase
// ---------------------------------------------------------------------------

/// Which driver called `run_stage_solve`. Gates post-solve capture only;
/// `load_model` / `set_bounds` / `reconstruct_basis` / solve sequence is
/// identical across phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Forward pass: scenario simulation with policy evaluation.
    Forward,
    /// Backward pass: Benders cut generation.
    Backward,
    /// Simulation pipeline: post-training scenario evaluation.
    Simulation,
}

// ---------------------------------------------------------------------------
// StageInputs
// ---------------------------------------------------------------------------

/// Read-only inputs for one LP solve at stage `t`, scenario `m`.
///
/// All fields are borrows or `Copy` primitives — no owned allocation. The
/// struct has no `Default` implementation; callers must supply every field.
/// This prevents the "new config flag wired three places, missed the fourth"
/// bug class: the compiler rejects any construction that omits a field.
///
/// Constructed per-call inside each driver's inner loop; never stored across
/// solves.
pub struct StageInputs<'a> {
    /// Per-stage LP layout and noise scaling parameters.
    pub stage_context: &'a StageContext<'a>,
    /// LP index layout for stage `t`.
    pub indexer: &'a StageIndexer,
    /// Active cut pool for stage `t`.
    pub pool: &'a CutPool,
    /// Current state vector; length equals `indexer.n_state`.
    pub current_state: &'a [f64],
    /// Stored basis from the previous iteration, if any.
    pub stored_basis: Option<&'a CapturedBasis>,
    /// Baked stage template for this stage. Always populated — the caller
    /// guarantees that baked templates have been constructed before the solve.
    pub baked_template: &'a StageTemplate,
    /// Stage index `t`.
    pub stage_index: usize,
    /// Scenario index `m`.
    pub scenario_index: usize,
    /// True when the current stage is the terminal stage of the horizon.
    /// Forward-only gate; harmless for other phases.
    pub horizon_is_terminal: bool,
    /// True when the terminal stage has boundary cuts loaded.
    /// Forward-only gate; harmless for other phases.
    pub terminal_has_boundary_cuts: bool,
    /// Training iteration number (1-based), if the caller is the forward pass.
    ///
    /// `None` on the backward and simulation phases, where iteration context
    /// is either absent or available through other channels. Present on the
    /// forward phase to populate `SddpError::Infeasible { iteration }` without
    /// the caller having to re-wrap the error.
    pub iteration: Option<u64>,
}

// ---------------------------------------------------------------------------
// StageOutcome
// ---------------------------------------------------------------------------

/// What each phase captures from one LP solve.
///
/// Variant layout covers what forward, backward, and simulation need today
/// without requiring owned allocations inside `run_stage_solve`. The three
/// variants carry identical data for now; they exist so that later epics can
/// specialize per-phase fields (e.g., add `captured_basis` to `Forward` only)
/// without a breaking re-shape.
#[derive(Debug)]
pub enum StageOutcome<'solver> {
    /// Outcome for the forward pass.
    Forward {
        /// LP solution view borrowing the solver's internal buffers.
        view: SolutionView<'solver>,
        /// Basis reconstruction counters (preserved, `new_tight`, `new_slack`,
        /// demotions are tracked in the workspace stats).
        recon_stats: ReconstructionStats,
    },
    /// Outcome for the backward pass.
    Backward {
        /// LP solution view borrowing the solver's internal buffers.
        view: SolutionView<'solver>,
        /// Basis reconstruction counters.
        recon_stats: ReconstructionStats,
    },
    /// Outcome for the simulation pipeline.
    Simulation {
        /// LP solution view borrowing the solver's internal buffers.
        view: SolutionView<'solver>,
        /// Basis reconstruction counters.
        recon_stats: ReconstructionStats,
    },
}

// ---------------------------------------------------------------------------
// run_stage_solve
// ---------------------------------------------------------------------------

/// Execute one LP solve at stage `t` for scenario `m` on the given phase.
///
/// Load-and-bounds setup is the caller's responsibility (via `StageInputs`);
/// `run_stage_solve` owns basis reconstruction, invariant enforcement, and
/// the solve call.
///
/// Returns on success an outcome referencing the solver's current solution
/// view. The returned borrow is alive until the next mutation of `ws.solver`.
///
/// # Errors
///
/// `SolverError::Infeasible` bubbles as `SddpError::Infeasible { ... }` with
/// stage/scenario context filled in. Other solver errors propagate as
/// `SddpError::Solver`.
pub fn run_stage_solve<'ws, S: SolverInterface>(
    ws: &'ws mut SolverWorkspace<S>,
    phase: Phase,
    inputs: &StageInputs<'_>,
) -> Result<StageOutcome<'ws>, SddpError> {
    // Grow slot-lookup scratch if the pool has allocated new slots since the last
    // call. `pool.populated_count` is monotonically non-decreasing, so after the
    // first few iterations this check is a no-op.
    if ws.scratch.recon_slot_lookup.len() < inputs.pool.populated_count {
        ws.scratch
            .recon_slot_lookup
            .resize(inputs.pool.populated_count, None);
    }

    // Select the basis path and solve.
    let (view, recon_stats) = if let Some(captured) = inputs.stored_basis {
        let theta_value = inputs
            .pool
            .evaluate_at_state(&inputs.current_state[..inputs.indexer.n_state]);
        let padding = PaddingContext {
            state: &inputs.current_state[..inputs.indexer.n_state],
            theta: theta_value,
            tolerance: 1e-7,
        };

        // All solves now use the baked path: cuts are structural rows in the
        // baked template; the delta-cut iterator is always empty (AD-3).
        let baked = inputs.baked_template;
        let recon_stats = reconstruct_basis(
            captured,
            ReconstructionTarget {
                base_row_count: baked.num_rows,
                num_cols: inputs.stage_context.templates[inputs.stage_index].num_cols,
            },
            std::iter::empty(),
            padding,
            &mut ws.scratch_basis,
            &mut ws.scratch.recon_slot_lookup,
        );
        // Baked path: no delta cut rows, so num_row == base_row_count.
        // enforce_basic_count_invariant is a no-op when the basis already
        // balances, but applying it uniformly closes the simulation-drift
        // bug class (source doc AD-2).
        let num_row_for_invariant = baked.num_rows;
        let base_row_for_invariant = baked.num_rows;

        // Enforce basic-count invariant across all phases (source doc AD-2):
        // no-op on the baked path in the common case.
        enforce_basic_count_invariant(
            &mut ws.scratch_basis,
            num_row_for_invariant,
            base_row_for_invariant,
        );

        ws.solver.record_reconstruction_stats();

        let view = ws.solver.solve(Some(&ws.scratch_basis)).map_err(|e| {
            map_solver_error(
                e,
                inputs.stage_index,
                inputs.scenario_index,
                inputs.iteration,
            )
        })?;
        (view, recon_stats)
    } else {
        // Cold path: no stored basis; solver starts from scratch.
        let view = ws.solver.solve(None).map_err(|e| {
            map_solver_error(
                e,
                inputs.stage_index,
                inputs.scenario_index,
                inputs.iteration,
            )
        })?;
        (view, ReconstructionStats::default())
    };

    // Wrap the solution in the phase-appropriate outcome variant.
    let outcome = match phase {
        Phase::Forward => StageOutcome::Forward { view, recon_stats },
        Phase::Backward => StageOutcome::Backward { view, recon_stats },
        Phase::Simulation => StageOutcome::Simulation { view, recon_stats },
    };
    Ok(outcome)
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Map a [`SolverError`] to a [`SddpError`] with stage/scenario context.
///
/// `Infeasible` is wrapped with the full context fields so the caller can
/// report which stage/scenario triggered the infeasibility.  All other
/// variants are wrapped by the `From<SolverError>` impl (i.e. `SddpError::Solver`).
fn map_solver_error(
    e: SolverError,
    stage: usize,
    scenario: usize,
    iteration: Option<u64>,
) -> SddpError {
    match e {
        SolverError::Infeasible => SddpError::Infeasible {
            stage,
            iteration: iteration.unwrap_or(0),
            scenario,
        },
        other => SddpError::Solver(other),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use cobre_solver::{HighsSolver, SolverError, SolverInterface, StageTemplate};

    use super::{Phase, StageInputs, run_stage_solve};
    use crate::{
        SddpError,
        basis_reconstruct::{
            HIGHS_BASIS_STATUS_BASIC as B, HIGHS_BASIS_STATUS_LOWER as L, ReconstructionStats,
        },
        context::StageContext,
        cut::pool::CutPool,
        indexer::StageIndexer,
        lp_builder::PatchBuffer,
        workspace::{CapturedBasis, SolverWorkspace, WorkspaceSizing},
    };

    // -----------------------------------------------------------------------
    // Shared fixtures
    // -----------------------------------------------------------------------

    /// Minimal LP: 3 columns, 2 rows (same fixture used in cobre-solver tests).
    ///
    ///   min  0*x0 + 1*x1 + 50*x2
    ///   s.t. x0            = 6   (state-fixing)
    ///        2*x0 + x2     = 14  (power balance)
    ///   x0 in [0, 10], x1 in [0, +inf), x2 in [0, 8]
    fn make_template() -> StageTemplate {
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

    /// Infeasible LP: 1 column with x0 >= 5 AND x0 <= 2.
    fn make_infeasible_template() -> StageTemplate {
        StageTemplate {
            num_cols: 1,
            num_rows: 0,
            num_nz: 0,
            col_starts: vec![0_i32, 0],
            row_indices: vec![],
            values: vec![],
            col_lower: vec![5.0],
            col_upper: vec![2.0], // infeasible: lower > upper
            objective: vec![1.0],
            row_lower: vec![],
            row_upper: vec![],
            n_state: 0,
            n_transfer: 0,
            n_dual_relevant: 0,
            n_hydro: 0,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Build a fresh `SolverWorkspace<HighsSolver>` with an LP loaded.
    fn make_workspace(template: &StageTemplate) -> SolverWorkspace<HighsSolver> {
        let mut solver = HighsSolver::new().expect("HighsSolver::new()");
        solver.load_model(template);
        SolverWorkspace::new(
            0,
            0,
            solver,
            PatchBuffer::new(0, 0, 0, 0),
            0,
            WorkspaceSizing::default(),
        )
    }

    /// Build a minimal `StageContext` wrapping a single template.
    fn make_context(templates: &[StageTemplate]) -> StageContext<'_> {
        StageContext {
            templates,
            base_rows: &[],
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        }
    }

    /// Build an empty `CutPool` (no active cuts, `populated_count = 0`).
    fn make_empty_pool() -> CutPool {
        CutPool::new(16, 1, 1, 0)
    }

    /// Build a minimal `StageIndexer` with `n_state = 1` (matches the fixture LP).
    fn make_indexer() -> StageIndexer {
        StageIndexer::new(1, 0)
    }

    /// Build a baked `StageTemplate` with the given total row count.
    // -----------------------------------------------------------------------
    // Test 1: cold start — `stored_basis: None` returns default stats
    // -----------------------------------------------------------------------

    #[test]
    fn run_stage_solve_cold_start_returns_outcome() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let ctx = make_context(templates);
        let pool = make_empty_pool();
        let indexer = make_indexer();
        let mut ws = make_workspace(&template);

        let inputs = StageInputs {
            stage_context: &ctx,
            indexer: &indexer,
            pool: &pool,
            current_state: &[0.0],
            stored_basis: None,
            baked_template: &template,
            stage_index: 0,
            scenario_index: 0,
            horizon_is_terminal: false,
            terminal_has_boundary_cuts: false,
            iteration: Some(1),
        };

        let result = run_stage_solve(&mut ws, Phase::Forward, &inputs);
        let outcome = result.expect("cold start should succeed");

        match outcome {
            crate::stage_solve::StageOutcome::Forward { recon_stats, .. } => {
                assert_eq!(recon_stats, ReconstructionStats::default());
            }
            _ => panic!("expected Forward variant"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: warm start on the baked path — successful solve returns outcome
    // -----------------------------------------------------------------------

    /// Verifies that a warm-start with a valid `CapturedBasis` on the baked
    /// path completes successfully.  On the baked path all cut rows are
    /// structural, so `reconstruct_basis` uses an empty iterator and
    /// `enforce_basic_count_invariant` is a no-op (no excess BASIC rows).
    ///
    /// Basis accounting: `CapturedBasis` is built with `base_row_count=2` and
    /// `cut_row_slots=[]` (no stored cut rows).  `reconstruct_basis` with
    /// `target.base_row_count=2` copies the 2 stored row statuses.
    /// With 2 BASIC col statuses + 0 BASIC row statuses, `total_basic=2 == num_row=2`.
    #[test]
    fn run_stage_solve_warm_start_baked_path_succeeds() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let ctx = make_context(templates);
        let pool = make_empty_pool();
        let indexer = make_indexer();
        let mut ws = make_workspace(&template);
        ws.scratch.recon_slot_lookup = vec![None; 16];

        // Build a CapturedBasis with base_row_count=2, 0 cut rows.
        // col_status: first 2 cols BASIC (needed to match num_row=2),
        // row_status: 2 LOWER entries (rows are at bound in optimal solution).
        // total_basic = col_basic(2) + row_basic(0) = 2 == num_row(2). Valid.
        let mut captured = CapturedBasis::new(
            template.num_cols,
            template.num_rows,
            template.num_rows,
            0,
            1,
        );
        captured.basis.col_status.clear();
        captured.basis.col_status.push(B); // x0 BASIC
        captured.basis.col_status.push(B); // x1 BASIC
        captured.basis.col_status.push(L); // x2 LOWER
        captured.basis.row_status.clear();
        captured.basis.row_status.push(L); // row 0 at bound
        captured.basis.row_status.push(L); // row 1 at bound
        captured.state_at_capture.push(6.0); // captured at state = [6.0]

        let inputs = StageInputs {
            stage_context: &ctx,
            indexer: &indexer,
            pool: &pool,
            current_state: &[6.0],
            stored_basis: Some(&captured),
            baked_template: &template,
            stage_index: 0,
            scenario_index: 0,
            horizon_is_terminal: false,
            terminal_has_boundary_cuts: false,
            iteration: None,
        };

        let result = run_stage_solve(&mut ws, Phase::Simulation, &inputs);
        assert!(
            result.is_ok(),
            "warm start on baked path should succeed: {result:?}"
        );
        // No cut rows → no demotions. Row statuses are all LOWER (non-basic).
        let lower_count = ws
            .scratch_basis
            .row_status
            .iter()
            .filter(|&&s| s == L)
            .count();
        assert_eq!(
            lower_count, 2,
            "both rows should be LOWER after reconstruction"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: infeasible LP propagates as SddpError::Infeasible
    // -----------------------------------------------------------------------

    #[test]
    fn run_stage_solve_propagates_infeasible() {
        let template = make_infeasible_template();
        let templates = std::slice::from_ref(&template);
        let ctx = make_context(templates);
        let pool = CutPool::new(16, 0, 1, 0);
        // StageIndexer with n_state=0 (no hydros).
        let indexer = StageIndexer::new(0, 0);
        let mut ws = make_workspace(&template);

        let inputs = StageInputs {
            stage_context: &ctx,
            indexer: &indexer,
            pool: &pool,
            current_state: &[],
            stored_basis: None,
            baked_template: &template,
            stage_index: 0,
            scenario_index: 7,
            horizon_is_terminal: false,
            terminal_has_boundary_cuts: false,
            iteration: Some(42),
        };

        let result = run_stage_solve(&mut ws, Phase::Forward, &inputs);
        match result {
            Err(SddpError::Infeasible {
                stage,
                scenario,
                iteration,
            }) => {
                assert_eq!(stage, 0, "stage must match inputs.stage_index");
                assert_eq!(scenario, 7, "scenario must match inputs.scenario_index");
                assert_eq!(iteration, 42, "iteration must match inputs.iteration");
            }
            other => panic!("expected SddpError::Infeasible, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 6: BasisInconsistent propagates as SddpError::Solver
    // -----------------------------------------------------------------------

    /// A basis with zero basic variables (all LOWER) against `num_row = 2`
    /// causes `cobre_highs_set_basis_non_alien` to fail `isBasisConsistent`
    /// and return `HIGHS_STATUS_ERROR`.  The solver converts this to
    /// `SolverError::BasisInconsistent`, which `map_solver_error` must route
    /// to `SddpError::Solver(...)` — not `SddpError::Infeasible`.
    #[test]
    fn basis_inconsistent_propagates_as_sddp_solver_error() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let ctx = make_context(templates);
        let pool = make_empty_pool();
        let indexer = make_indexer();
        let mut ws = make_workspace(&template);
        ws.scratch.recon_slot_lookup = vec![None; 16];

        // Build a CapturedBasis where all col_status and row_status are LOWER
        // (= 0, the zero-fill default from Basis::new).  After reconstruct_basis
        // copies these values into scratch_basis and enforce_basic_count_invariant
        // runs (it only demotes excess basics, never promotes), the delivered
        // basis has col_basic = 0, row_basic = 0, total_basic = 0 against
        // num_row = 2.  cobre_highs_set_basis_non_alien rejects this because
        // isBasisConsistent requires total_basic == num_row.
        let all_lower = CapturedBasis::new(
            template.num_cols,
            template.num_rows,
            template.num_rows,
            0,
            1,
        );
        // state_at_capture is empty (capacity only), which satisfies the
        // reconstruct_basis debug_assert (stored.state_at_capture.is_empty()).

        let inputs = StageInputs {
            stage_context: &ctx,
            indexer: &indexer,
            pool: &pool,
            current_state: &[0.0],
            stored_basis: Some(&all_lower),
            baked_template: &template,
            stage_index: 0,
            scenario_index: 3,
            horizon_is_terminal: false,
            terminal_has_boundary_cuts: false,
            iteration: Some(5),
        };

        let result = run_stage_solve(&mut ws, Phase::Forward, &inputs);
        match result {
            Err(SddpError::Solver(SolverError::BasisInconsistent { .. })) => {
                // Correct: BasisInconsistent routes to SddpError::Solver, not
                // SddpError::Infeasible.
            }
            Err(SddpError::Infeasible { .. }) => {
                panic!("BasisInconsistent must not map to SddpError::Infeasible")
            }
            other => panic!(
                "expected Err(SddpError::Solver(SolverError::BasisInconsistent {{ .. }})), \
                 got {other:?}"
            ),
        }
    }
}
