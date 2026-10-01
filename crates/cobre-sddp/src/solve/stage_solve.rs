//! LP-solve entry points shared by the hot-path drivers (`forward.rs`,
//! `backward.rs`, `simulation/pipeline.rs`): [`run_stage_solve`] (the default,
//! slot-identity warm-start via `reconstruct_basis`) and
//! [`run_stage_solve_terminal_static`] (terminal-forward-only, a 1:1 basis
//! apply that skips slot-identity reconciliation because the terminal
//! template's shape never changes after priming).

use cobre_core::temporal::StageLagTransition;
use cobre_solver::{SolutionView, SolverError, SolverInterface};

use crate::{
    basis_reconstruct::{ReconstructionTarget, enforce_basic_count_invariant, reconstruct_basis},
    context::StageContext,
    cut::pool::CutPool,
    error::SddpError,
    lp::builder::StateBox,
    lp::indexer::StateSpace,
    noise::{DownstreamAccumState, LagAccumState, accumulate_and_shift_lag_state},
    setup::{NodeId, StageIdx},
    workspace::{CapturedBasis, SolverWorkspace},
};

/// Read-only inputs for one LP solve at stage `t`, scenario `m`.
///
/// Intentionally has no `Default`: callers must supply every field, so the
/// compiler rejects any construction that omits one — preventing the "new flag
/// wired three places, missed the fourth" bug class.
pub struct StageInputs<'a> {
    /// Per-stage LP layout and noise scaling parameters.
    pub stage_context: &'a StageContext<'a>,
    /// Active cut pool for stage `t`.
    pub pool: &'a CutPool,
    /// Stored basis from the previous iteration, if any.
    pub stored_basis: Option<&'a CapturedBasis>,
    /// Stage index `t`.
    pub stage_index: StageIdx,
    /// Scenario index `m`.
    pub scenario_index: usize,
    /// Training iteration (1-based) on the forward pass; `None` on backward and
    /// simulation. Forward supplies it so `SddpError::Infeasible { iteration }`
    /// is populated without the caller re-wrapping the error.
    pub iteration: Option<u64>,
    /// Declared id of the node being solved (`NodeGraph::node_ids[node]`); a
    /// `stored_basis` whose `node_id` differs is treated as cold.
    pub node_id: NodeId,
}

/// Execute one LP solve at stage `t` for scenario `m`, owning basis
/// reconstruction, invariant enforcement, and the solve call.
///
/// Load-and-bounds setup is the caller's responsibility (via `StageInputs`).
/// Returns the solver's solution view on success.
///
/// # Errors
///
/// `SolverError::Infeasible` bubbles as `SddpError::Infeasible { ... }` with
/// stage/scenario context filled in. Other solver errors propagate as
/// `SddpError::Solver`.
pub fn run_stage_solve<'ws, S: SolverInterface>(
    ws: &'ws mut SolverWorkspace<S>,
    inputs: &StageInputs<'_>,
) -> Result<SolutionView<'ws>, SddpError> {
    // `pool.populated()` only grows, so this resize is a no-op after the
    // first few iterations.
    if ws.scratch.recon_slot_lookup.len() < inputs.pool.populated() {
        ws.scratch
            .recon_slot_lookup
            .resize(inputs.pool.populated(), None);
    }

    let stored_basis = filtered_stored_basis(inputs);

    let solved = if let Some(captured) = stored_basis {
        let template = inputs.stage_context.template(inputs.stage_index);
        // `base_row_count` is the non-frozen template row count so cut rows are
        // matched by slot identity, not positional copy from the stored basis.
        let target = ReconstructionTarget::from_template(template);

        let _ = reconstruct_basis(
            captured,
            target,
            inputs.pool.active_cuts(),
            &mut ws.scratch_basis,
            &mut ws.scratch.recon_slot_lookup,
        );
        // `num_row_for_invariant` is the reconstructed length — `target.base_row_count`
        // plus one row per active cut, which is this LP's row count on every frozen
        // path (forward and simulation load `frozen[t]`; backward loads `frozen[t]`
        // then adds the delta cuts the pool has meanwhile gained). `frozen.num_rows`
        // is the wrong-but-plausible alternative: it omits those delta cuts.
        //
        // `base_row_for_invariant = 0` is safe: the loop demotes only currently-
        // BASIC rows, and equality rows are never BASIC by LP duality.
        let num_row_for_invariant = ws.scratch_basis.row_status.len();
        let base_row_for_invariant = 0;

        enforce_basic_count_invariant(
            &mut ws.scratch_basis,
            num_row_for_invariant,
            base_row_for_invariant,
        )?;

        ws.solver.record_reconstruction_stats();

        ws.solver.solve(Some(&ws.scratch_basis))
    } else {
        ws.solver.solve(None)
    };

    finish_solve(solved, inputs)
}

/// Terminal-forward-only 1:1 basis apply, bypassing [`reconstruct_basis`]'s
/// slot-identity reconciliation.
///
/// The terminal leaf's active-cut set is fixed once primed (a leaf never
/// gains or loses a cut), so the terminal template's shape never changes
/// across iterations: a node-matching stored basis maps onto the current LP
/// by plain position. A stored basis that fails the node-tag filter, or whose
/// shape does not match the current template (should be impossible given
/// that invariance), is treated as cold — never applied — rather than
/// risking `solve`'s own hard assertion on a column-count mismatch.
///
/// # Errors
///
/// Same as [`run_stage_solve`]: `SolverError::Infeasible` bubbles as
/// `SddpError::Infeasible { .. }`; other solver errors propagate as
/// `SddpError::Solver`.
pub fn run_stage_solve_terminal_static<'ws, S: SolverInterface>(
    ws: &'ws mut SolverWorkspace<S>,
    inputs: &StageInputs<'_>,
) -> Result<SolutionView<'ws>, SddpError> {
    let stored_basis = filtered_stored_basis(inputs)
        .filter(|captured| terminal_basis_shape_matches(captured, inputs));

    let solved = if let Some(captured) = stored_basis {
        ws.scratch_basis.col_status.clear();
        ws.scratch_basis
            .col_status
            .extend_from_slice(&captured.basis.col_status);
        ws.scratch_basis.row_status.clear();
        ws.scratch_basis
            .row_status
            .extend_from_slice(&captured.basis.row_status);

        ws.solver.solve(Some(&ws.scratch_basis))
    } else {
        ws.solver.solve(None)
    };

    finish_solve(solved, inputs)
}

/// A stored basis whose `node_id` differs from the node being solved is
/// treated as cold: a resampled path may visit a different node than the
/// one that captured it, and CLP accepts a shape-mismatched basis silently,
/// so the check must live here, never delegated to a solver-specific path.
fn filtered_stored_basis<'a>(inputs: &StageInputs<'a>) -> Option<&'a CapturedBasis> {
    inputs
        .stored_basis
        .filter(|captured| captured.node_id == inputs.node_id)
}

/// Whether `captured`'s column/row status vectors already match the terminal
/// template's current shape (base rows plus active cuts) — the only shape a
/// genuine same-node terminal capture can have.
fn terminal_basis_shape_matches(captured: &CapturedBasis, inputs: &StageInputs<'_>) -> bool {
    let template = inputs.stage_context.template(inputs.stage_index);
    captured.basis.col_status.len() == template.num_cols
        && captured.basis.row_status.len() == template.num_rows + inputs.pool.active_count()
}

/// Map a raw solve `Result` to the `SddpError` the caller returns, filling in
/// stage/scenario/iteration context on the `Infeasible` path.
fn finish_solve<'ws>(
    solved: Result<SolutionView<'ws>, SolverError>,
    inputs: &StageInputs<'_>,
) -> Result<SolutionView<'ws>, SddpError> {
    solved.map_err(|e| {
        map_solver_error(
            e,
            inputs.stage_index.0,
            inputs.scenario_index,
            inputs.iteration,
        )
    })
}

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

/// Unscale a primal vector into `out`: `x_original[i] = col_scale[i] *
/// x_scaled[i]` (or the raw values when `col_scale` is empty). `out` retains its
/// pre-warmed capacity (`clear` + `extend`).
pub(crate) fn fill_unscaled(out: &mut Vec<f64>, scaled: &[f64], col_scale: &[f64]) {
    debug_assert!(
        col_scale.is_empty() || col_scale.len() == scaled.len(),
        "col_scale length {} != primal length {}",
        col_scale.len(),
        scaled.len()
    );
    out.clear();
    if col_scale.is_empty() {
        out.extend_from_slice(scaled);
    } else {
        out.extend(scaled.iter().zip(col_scale.iter()).map(|(&xp, &d)| d * xp));
    }
}

/// Unscale a dual vector into `out`: `dual_original[i] = row_scale[i] *
/// dual_scaled[i]` for structural rows; rows beyond the template length (cut
/// rows) have implicit `row_scale = 1.0`. Empty `row_scale` means no scaling.
pub(crate) fn fill_unscaled_dual(out: &mut Vec<f64>, scaled: &[f64], row_scale: &[f64]) {
    // Unlike `fill_unscaled`, `scaled` may be *longer* than `row_scale`: cut
    // rows extend past the template-row count and take an implicit scale of
    // 1.0. The invariant is therefore `<=`, not `==`.
    debug_assert!(
        row_scale.len() <= scaled.len(),
        "row_scale length {} exceeds dual length {}",
        row_scale.len(),
        scaled.len()
    );
    out.clear();
    if row_scale.is_empty() {
        out.extend_from_slice(scaled);
    } else {
        out.extend(scaled.iter().enumerate().map(|(i, &d)| {
            let scale = row_scale.get(i).copied().unwrap_or(1.0);
            d * scale
        }));
    }
}

/// Canonicalize the outgoing state vector: copy the LP primal's state
/// columns, run the lag accumulation/shift, verify the bucket/commitment-hold
/// copy-gap invariant, then clamp every dimension onto its admissible box.
/// The clamp is the identity for an in-box value — a solve whose state stays
/// in box is byte-for-byte unaffected by it. Every consumer of the outgoing
/// state (the trajectory, the backward pass's `x_hat`, the cut intercept, the
/// simulation output) reads this same canonical vector; there is no second
/// canonicalization point at LP pin time.
// Rationale (too_many_arguments): every argument is a caller-owned buffer or
// borrow threaded straight through to `accumulate_and_shift_lag_state` (which
// already takes the same shape) plus the box this seam adds; a bundle would
// just wrap them for one call site.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::float_cmp)]
pub(crate) fn assemble_outgoing_state(
    current_state: &mut Vec<f64>,
    unscaled_primal: &[f64],
    lag_matrix_buf: &[f64],
    layout: &StateSpace,
    state_box: &StateBox,
    stage_lag: StageLagTransition,
    lag: &mut LagAccumState<'_>,
    ds: &mut DownstreamAccumState<'_>,
) {
    current_state.clear();
    current_state.extend_from_slice(&unscaled_primal[..layout.n_state]);
    accumulate_and_shift_lag_state(
        current_state,
        lag_matrix_buf,
        unscaled_primal,
        layout,
        &stage_lag,
        lag,
        ds,
    );
    debug_assert_bucket_copy_gap_intact(current_state.as_slice(), unscaled_primal, layout);

    for (j, value) in current_state.iter_mut().take(layout.n_state).enumerate() {
        let clamped = value.clamp(state_box.lower[j], state_box.upper[j]);
        if clamped == *value {
            continue;
        }
        *value = clamped;
        debug_assert!(
            layout.storage.contains(&j)
                || layout.transit_buckets_out.contains(&j)
                || layout.commit_out.contains(&j),
            "a clamped state dimension must be storage, transit-bucket, or \
             commitment-hold — inflow lags are unbounded and never clamp"
        );
    }
}

/// Assert the bucket and commitment-hold state rode the state-assembly plain
/// copy untouched.
///
/// The forward/simulation assembly plain-copies `unscaled_primal[..n_state]`
/// into the advanced state, then overwrites only `inflow_lags` in place;
/// `transit_buckets_out` and `commit_out` sit in the shift-gap that overwrite
/// never reaches, because both outgoing columns equal their state-vector
/// index (the `storage` identity convention). Call after the lag overwrite,
/// before the clamp — the clamp legitimately moves bucket/commitment-hold
/// values off the primal, which would trip this assert if it ran after.
// Rationale: this checks a verbatim copy invariant (`extend_from_slice`), not
// a numerical result, so exact equality is the correct comparison — a
// tolerance would mask the one bug this guards against (an overwrite landing
// on the bucket/commitment-hold range).
#[allow(clippy::float_cmp)]
fn debug_assert_bucket_copy_gap_intact(
    assembled_state: &[f64],
    unscaled_primal: &[f64],
    layout: &StateSpace,
) {
    debug_assert!(
        layout
            .transit_buckets_out
            .clone()
            .chain(layout.commit_out.clone())
            .all(|j| assembled_state[j] == unscaled_primal[j]),
        "bucket/commitment-hold state must equal the LP primal's identity \
         columns: the lag overwrite must never touch the bucket/commitment-hold \
         shift-gap"
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use cobre_core::temporal::StageLagTransition;
    use cobre_solver::BasisStatus::{Basic as B, Lower as L};
    use cobre_solver::{ActiveSolver, SolverError, SolverInterface, StageTemplate};

    use super::{
        StageInputs, assemble_outgoing_state, run_stage_solve, run_stage_solve_terminal_static,
    };
    use crate::{
        SddpError,
        cut::pool::CutPool,
        lp::builder::{PatchBuffer, StageGeometry, StateBox},
        lp::indexer::HydroSys,
        noise::{DownstreamAccumState, LagAccumState, accumulate_and_shift_lag_state},
        setup::{NodeId, StageIdx},
        test_support::{
            StageContextFixture, TrainingContextFixture, equipment_free_geometry, state_layout,
            state_layout_with_transit_buckets,
        },
        workspace::{CapturedBasis, SolverWorkspace, WorkspaceSizing},
    };

    // -----------------------------------------------------------------------
    // Shared fixtures
    // -----------------------------------------------------------------------

    /// Minimal LP: 3 columns, 2 rows (same fixture used in cobre-solver tests).
    ///
    ///   min  0*x0 + 1*x1 + 50*x2
    ///   s.t. x0            = 6   (pins x0)
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
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Build a fresh `SolverWorkspace<ActiveSolver>` with an LP loaded.
    fn make_workspace(template: &StageTemplate) -> SolverWorkspace<ActiveSolver> {
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new()");
        solver.load_model(template);
        let training_fixture = TrainingContextFixture::new(state_layout(0, 0));
        let training_ctx = training_fixture.training_ctx();
        let stage_ctx_fixture = StageContextFixture::new(&[], &[], &[]);
        SolverWorkspace::new(
            0,
            0,
            solver,
            PatchBuffer::new(training_ctx.state, &[], &[]),
            &training_ctx,
            &stage_ctx_fixture.ctx(),
            WorkspaceSizing::default(),
        )
    }

    /// Build a minimal `StageContext` fixture wrapping a single template.
    fn make_context<'a>(
        templates: &'a [StageTemplate],
        geometry_per_stage: &'a [StageGeometry],
    ) -> StageContextFixture<'a> {
        StageContextFixture::new(templates, &[], geometry_per_stage)
    }

    /// Build an empty `CutPool` (no active cuts, `populated_count = 0`).
    fn make_empty_pool() -> CutPool {
        CutPool::new(16, 1, 1, 0)
    }

    // -----------------------------------------------------------------------
    // Test 1: cold start — `stored_basis: None` solves from scratch
    // -----------------------------------------------------------------------

    #[test]
    fn run_stage_solve_cold_start_returns_view() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = make_empty_pool();
        let mut ws = make_workspace(&template);

        let inputs = StageInputs {
            stage_context: &ctx,
            pool: &pool,
            stored_basis: None,
            stage_index: StageIdx(0),
            scenario_index: 0,
            iteration: Some(1),
            node_id: NodeId(0),
        };

        let result = run_stage_solve(&mut ws, &inputs);
        let view = result.expect("cold start should succeed");
        // Optimal objective for the fixture LP: x1 = 14 - 2*6 = 2, cost = 2.
        assert!(view.objective.is_finite());
    }

    // -----------------------------------------------------------------------
    // Test 2: warm start on the frozen path — successful solve returns outcome
    // -----------------------------------------------------------------------

    /// Verifies that a warm-start with a valid `CapturedBasis` on the frozen
    /// path completes successfully.  On the frozen path all cut rows are
    /// structural, so `reconstruct_basis` uses an empty iterator and
    /// `enforce_basic_count_invariant` is a no-op (no excess BASIC rows).
    ///
    /// Basis accounting: `CapturedBasis` is built with `base_row_count=2` and
    /// `cut_row_slots=[]` (no stored cut rows).  `reconstruct_basis` with
    /// `target.base_row_count=2` copies the 2 stored row statuses.
    /// With 2 BASIC col statuses + 0 BASIC row statuses, `total_basic=2 == num_row=2`.
    #[test]
    fn run_stage_solve_warm_start_frozen_path_succeeds() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = make_empty_pool();
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
            NodeId(5), // node_id — matches inputs.node_id below (the match/warm case)
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
            pool: &pool,
            stored_basis: Some(&captured),
            stage_index: StageIdx(0),
            scenario_index: 0,
            iteration: None,
            node_id: NodeId(5),
        };

        let result = run_stage_solve(&mut ws, &inputs);
        assert!(
            result.is_ok(),
            "warm start on frozen path should succeed: {result:?}"
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
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = CutPool::new(16, 0, 1, 0);
        let mut ws = make_workspace(&template);

        let inputs = StageInputs {
            stage_context: &ctx,
            pool: &pool,
            stored_basis: None,
            stage_index: StageIdx(0),
            scenario_index: 7,
            iteration: Some(42),
            node_id: NodeId(0),
        };

        let result = run_stage_solve(&mut ws, &inputs);
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
    // Test 6: a basic-count deficit is rejected before the solver sees it
    // -----------------------------------------------------------------------

    /// A stored basis with zero basic variables (all LOWER) against `num_row = 2`
    /// is a deficit, which only a shape mismatch can produce. `run_stage_solve`
    /// must reject it with `SddpError::BasisShapeMismatch` naming the counters —
    /// never pass it to the solver, whose own `isBasisConsistent` rejection
    /// (`SolverError::BasisInconsistent` under `HiGHS`; accepted outright under CLP)
    /// cannot say which LP the basis came from.
    #[test]
    fn basis_deficit_rejected_before_solver_sees_it() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = make_empty_pool();
        let mut ws = make_workspace(&template);
        ws.scratch.recon_slot_lookup = vec![None; 16];

        // All col_status and row_status LOWER (the zero-fill default from
        // Basis::new) → col_basic = 0, row_basic = 0, total_basic = 0 against
        // num_row = 2.
        let all_lower = CapturedBasis::new(
            template.num_cols,
            template.num_rows,
            template.num_rows,
            0,
            1,
            NodeId(9), // node_id — matches inputs.node_id below (the match/warm case)
        );

        let inputs = StageInputs {
            stage_context: &ctx,
            pool: &pool,
            stored_basis: Some(&all_lower),
            stage_index: StageIdx(0),
            scenario_index: 3,
            iteration: Some(5),
            node_id: NodeId(9),
        };

        match run_stage_solve(&mut ws, &inputs) {
            Err(SddpError::BasisShapeMismatch {
                num_row,
                total_basic,
                col_basic,
                row_basic,
            }) => {
                assert_eq!(num_row, template.num_rows);
                assert_eq!(total_basic, 0);
                assert_eq!(col_basic, 0);
                assert_eq!(row_basic, 0);
            }
            other => panic!("expected Err(SddpError::BasisShapeMismatch {{ .. }}), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Node-tag mismatch: cross-node reuse is treated as cold
    // -----------------------------------------------------------------------

    /// The same deficit-shaped `CapturedBasis` `basis_deficit_rejected_before_solver_sees_it`
    /// uses to prove the warm path surfaces `BasisShapeMismatch` — but tagged with a
    /// node id that does NOT match `inputs.node_id`. The mismatch must drop to the
    /// cold path before `reconstruct_basis`/`enforce_basic_count_invariant` ever see
    /// the deficit-shaped basis, so the solve succeeds instead of erroring.
    #[test]
    fn run_stage_solve_cross_node_stored_basis_is_treated_as_cold() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = make_empty_pool();
        let mut ws = make_workspace(&template);
        ws.scratch.recon_slot_lookup = vec![None; 16];

        // Deficit shape identical to basis_deficit_rejected_before_solver_sees_it —
        // would surface as BasisShapeMismatch if the warm path were taken.
        let stored_at_node_a = CapturedBasis::new(
            template.num_cols,
            template.num_rows,
            template.num_rows,
            0,
            1,
            NodeId(1), // node A
        );

        let inputs = StageInputs {
            stage_context: &ctx,
            pool: &pool,
            stored_basis: Some(&stored_at_node_a),
            stage_index: StageIdx(0),
            scenario_index: 3,
            iteration: Some(5),
            node_id: NodeId(2), // node B — mismatches the stored basis's node A
        };

        let result = run_stage_solve(&mut ws, &inputs);
        assert!(
            result.is_ok(),
            "a node-id mismatch must drop to the cold path, never attempting the \
             deficit-shaped stored basis: {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Terminal-static entry: 1:1 basis apply, no reconstruct_basis
    // -----------------------------------------------------------------------

    /// The terminal-static entry applies a node-matching stored basis
    /// verbatim — `ws.scratch_basis` ends up byte-identical to the stored
    /// `col_status`/`row_status` — and never calls `reconstruct_basis`:
    /// `basis_reconstructions` (incremented only by `record_reconstruction_stats`,
    /// which `run_stage_solve`'s reconstruct branch calls and
    /// `run_stage_solve_terminal_static` does not) stays at zero.
    #[test]
    fn run_stage_solve_terminal_static_applies_basis_1to1_without_reconstruct_basis() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = make_empty_pool();
        let mut ws = make_workspace(&template);

        let mut captured = CapturedBasis::new(
            template.num_cols,
            template.num_rows,
            template.num_rows,
            0,
            1,
            NodeId(5),
        );
        captured.basis.col_status.clear();
        captured.basis.col_status.push(B);
        captured.basis.col_status.push(B);
        captured.basis.col_status.push(L);
        captured.basis.row_status.clear();
        captured.basis.row_status.push(L);
        captured.basis.row_status.push(L);
        captured.state_at_capture.push(6.0);

        let inputs = StageInputs {
            stage_context: &ctx,
            pool: &pool,
            stored_basis: Some(&captured),
            stage_index: StageIdx(0),
            scenario_index: 0,
            iteration: None,
            node_id: NodeId(5),
        };

        // `basis_reconstructions` is HiGHS-only telemetry (CLP's
        // `SolverInterface` never overrides `record_reconstruction_stats`), so
        // it is meaningful only under the `highs` backend; the backend-agnostic
        // proof that no reconstruction happened is the byte-for-byte copy
        // assertion below, which runs under both backends.
        #[cfg(feature = "highs")]
        assert_eq!(
            ws.solver.statistics().basis_reconstructions,
            0,
            "sanity: a fresh workspace starts with no reconstructions"
        );

        let result = run_stage_solve_terminal_static(&mut ws, &inputs);
        assert!(
            result.is_ok(),
            "terminal-static warm start should succeed: {result:?}"
        );
        assert_eq!(
            ws.scratch_basis.col_status, captured.basis.col_status,
            "terminal-static must copy col_status verbatim, never reconstruct it"
        );
        assert_eq!(
            ws.scratch_basis.row_status, captured.basis.row_status,
            "terminal-static must copy row_status verbatim, never reconstruct it"
        );
        #[cfg(feature = "highs")]
        assert_eq!(
            ws.solver.statistics().basis_reconstructions,
            0,
            "run_stage_solve_terminal_static must never invoke reconstruct_basis"
        );
    }

    /// Interior (non-terminal) warm starts still go through `reconstruct_basis`:
    /// `basis_reconstructions` increments by exactly one, unlike the
    /// terminal-static entry above. HiGHS-only: CLP's `SolverInterface` never
    /// overrides `record_reconstruction_stats`, so the counter would stay `0`
    /// there regardless of whether `reconstruct_basis` ran.
    #[cfg(feature = "highs")]
    #[test]
    fn run_stage_solve_interior_warm_start_invokes_reconstruct_basis() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = make_empty_pool();
        let mut ws = make_workspace(&template);
        ws.scratch.recon_slot_lookup = vec![None; 16];

        let mut captured = CapturedBasis::new(
            template.num_cols,
            template.num_rows,
            template.num_rows,
            0,
            1,
            NodeId(5),
        );
        captured.basis.col_status.clear();
        captured.basis.col_status.push(B);
        captured.basis.col_status.push(B);
        captured.basis.col_status.push(L);
        captured.basis.row_status.clear();
        captured.basis.row_status.push(L);
        captured.basis.row_status.push(L);
        captured.state_at_capture.push(6.0);

        let inputs = StageInputs {
            stage_context: &ctx,
            pool: &pool,
            stored_basis: Some(&captured),
            stage_index: StageIdx(0),
            scenario_index: 0,
            iteration: None,
            node_id: NodeId(5),
        };

        let result = run_stage_solve(&mut ws, &inputs);
        assert!(
            result.is_ok(),
            "interior warm start should succeed: {result:?}"
        );
        assert_eq!(
            ws.solver.statistics().basis_reconstructions,
            1,
            "run_stage_solve's slot-identity path must invoke reconstruct_basis"
        );
    }

    /// The terminal-static entry preserves the node-tag filter: a stored basis
    /// tagged at a different node must drop to cold before
    /// `terminal_basis_shape_matches` ever inspects the shape — mirrors
    /// `run_stage_solve_cross_node_stored_basis_is_treated_as_cold` above.
    #[test]
    fn run_stage_solve_terminal_static_cross_node_stored_basis_is_treated_as_cold() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = make_empty_pool();
        let mut ws = make_workspace(&template);

        // Deficit shape (all LOWER) — would be nonsense to apply even if the
        // node tag matched; the point here is that the node mismatch alone
        // must drop to cold before this shape is ever inspected.
        let stored_at_node_a = CapturedBasis::new(
            template.num_cols,
            template.num_rows,
            template.num_rows,
            0,
            1,
            NodeId(1),
        );

        let inputs = StageInputs {
            stage_context: &ctx,
            pool: &pool,
            stored_basis: Some(&stored_at_node_a),
            stage_index: StageIdx(0),
            scenario_index: 3,
            iteration: Some(5),
            node_id: NodeId(2), // node B — mismatches the stored basis's node A
        };

        let result = run_stage_solve_terminal_static(&mut ws, &inputs);
        assert!(
            result.is_ok(),
            "a node-id mismatch must drop to the cold path on the terminal-static \
             entry too: {result:?}"
        );
    }

    /// A shape mismatch on the terminal-static path — genuinely impossible
    /// given the terminal template's structural invariance, but defended
    /// against rather than risking `solve`'s own hard assertion on a
    /// column-count mismatch — must also drop to cold, never applying the
    /// wrongly-shaped basis.
    #[test]
    fn run_stage_solve_terminal_static_shape_mismatch_is_treated_as_cold() {
        let template = make_template();
        let templates = std::slice::from_ref(&template);
        let geometry = equipment_free_geometry(&[0]);
        let fixture = make_context(templates, &geometry);
        let ctx = fixture.ctx();
        let pool = make_empty_pool();
        let mut ws = make_workspace(&template);

        // col_status.len() == template.num_cols + 1: node id matches, so only
        // the shape check can stop this from being applied.
        let wrong_shape = CapturedBasis::new(
            template.num_cols + 1,
            template.num_rows,
            template.num_rows,
            0,
            1,
            NodeId(5),
        );

        let inputs = StageInputs {
            stage_context: &ctx,
            pool: &pool,
            stored_basis: Some(&wrong_shape),
            stage_index: StageIdx(0),
            scenario_index: 0,
            iteration: None,
            node_id: NodeId(5),
        };

        let result = run_stage_solve_terminal_static(&mut ws, &inputs);
        assert!(
            result.is_ok(),
            "a col/row shape mismatch must drop to the cold path, never applying \
             the wrongly-shaped basis: {result:?}"
        );
    }

    /// `SolverError::BasisInconsistent` must route to `SddpError::Solver`, not
    /// `SddpError::Infeasible`: it is a basis defect, not a proof of
    /// infeasibility, and conflating them would abort a recoverable run.
    #[test]
    fn basis_inconsistent_maps_to_sddp_solver_error() {
        let mapped = super::map_solver_error(
            SolverError::BasisInconsistent {
                num_row: 4385,
                total_basic: 4383,
                col_basic: 4000,
                row_basic: 383,
            },
            0,
            3,
            Some(5),
        );
        assert!(
            matches!(
                mapped,
                SddpError::Solver(SolverError::BasisInconsistent { .. })
            ),
            "expected SddpError::Solver(BasisInconsistent), got {mapped:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: bucket copy-gap assertion helper
    // -----------------------------------------------------------------------

    #[test]
    fn debug_assert_bucket_copy_gap_intact_passes_when_bucket_matches_primal() {
        let layout = state_layout_with_transit_buckets(
            0,
            0,
            vec![(HydroSys::new(0), 0), (HydroSys::new(0), 1)],
            vec![],
        );
        let primal = vec![7.0, 11.0];
        let assembled = primal.clone();
        super::debug_assert_bucket_copy_gap_intact(&assembled, &primal, &layout);
    }

    #[test]
    #[should_panic(expected = "bucket/commitment-hold state must equal the LP primal's identity")]
    fn debug_assert_bucket_copy_gap_intact_panics_when_bucket_diverges() {
        let layout = state_layout_with_transit_buckets(
            0,
            0,
            vec![(HydroSys::new(0), 0), (HydroSys::new(0), 1)],
            vec![],
        );
        let primal = vec![7.0, 11.0];
        let mut assembled = primal.clone();
        assembled[1] = 999.0; // simulate an accidental overwrite of the bucket block
        super::debug_assert_bucket_copy_gap_intact(&assembled, &primal, &layout);
    }

    #[test]
    fn debug_assert_bucket_copy_gap_intact_passes_when_commitment_hold_matches_primal() {
        let layout = state_layout_with_transit_buckets(0, 0, vec![], vec![1, 1]);
        let primal = vec![3.0, 5.0];
        let assembled = primal.clone();
        super::debug_assert_bucket_copy_gap_intact(&assembled, &primal, &layout);
    }

    #[test]
    #[should_panic(expected = "bucket/commitment-hold state must equal the LP primal's identity")]
    fn debug_assert_bucket_copy_gap_intact_panics_when_commitment_hold_diverges() {
        let layout = state_layout_with_transit_buckets(0, 0, vec![], vec![1, 1]);
        let primal = vec![3.0, 5.0];
        let mut assembled = primal.clone();
        assembled[0] = 999.0; // simulate an accidental overwrite of the commitment-hold slot
        super::debug_assert_bucket_copy_gap_intact(&assembled, &primal, &layout);
    }

    // -----------------------------------------------------------------------
    // Test 8: assemble_outgoing_state — the canonicalization seam
    // -----------------------------------------------------------------------

    /// `N=1, L=1, B=1, A=1, K_max=1` layout: `n_state = 4`
    /// (storage, lag0, `bucket_out`, `commit_out`), mirroring the
    /// `transit_bucket_copy_gap` fixture shape.
    fn seam_layout() -> crate::lp::indexer::StateSpace {
        state_layout_with_transit_buckets(1, 1, vec![(HydroSys::new(0), 0)], vec![1])
    }

    fn identity_stage_lag() -> StageLagTransition {
        StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: true,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: false,
        }
    }

    /// `unscaled_primal` for `seam_layout`'s 9-column LP: `[storage, lag0,
    /// bucket_out, commit_out, z_inflow, storage_in, bucket_in, commit_in,
    /// theta]`.
    fn seam_primal(storage: f64, bucket: f64, commit: f64) -> Vec<f64> {
        vec![storage, 0.0, bucket, commit, 5.0, 0.0, 0.0, 0.0, 0.0]
    }

    #[test]
    fn assemble_outgoing_state_in_box_is_byte_identical() {
        let layout = seam_layout();
        let unscaled_primal = seam_primal(10.0, 30.0, 20.0);
        let lag_matrix_buf = vec![2.0_f64];
        let stage_lag = identity_stage_lag();
        let state_box = StateBox {
            lower: vec![0.0, f64::NEG_INFINITY, 0.0, 0.0],
            upper: vec![100.0, f64::INFINITY, 100.0, 50.0],
        };

        // Reference: the pre-seam copy + lag accumulation, with no clamp step.
        let mut expected = Vec::new();
        expected.clear();
        expected.extend_from_slice(&unscaled_primal[..layout.n_state]);
        let mut exp_lag_accumulator = vec![0.0_f64; 1];
        let mut exp_lag_weight_accum = vec![0.0_f64; 1];
        let mut exp_ds_accum: Vec<f64> = vec![];
        let mut exp_ds_weight = 0.0_f64;
        let mut exp_ds_completed: Vec<f64> = vec![];
        let mut exp_ds_n_completed = 0_usize;
        accumulate_and_shift_lag_state(
            &mut expected,
            &lag_matrix_buf,
            &unscaled_primal,
            &layout,
            &stage_lag,
            &mut LagAccumState {
                accumulator: &mut exp_lag_accumulator,
                weight_accum: &mut exp_lag_weight_accum,
            },
            &mut DownstreamAccumState {
                accumulator: &mut exp_ds_accum,
                weight_accum: &mut exp_ds_weight,
                completed_lags: &mut exp_ds_completed,
                n_completed: &mut exp_ds_n_completed,
                par_order: 0,
            },
        );

        let mut current_state = Vec::new();
        let mut lag_accumulator = vec![0.0_f64; 1];
        let mut lag_weight_accum = vec![0.0_f64; 1];
        let mut ds_accum: Vec<f64> = vec![];
        let mut ds_weight = 0.0_f64;
        let mut ds_completed: Vec<f64> = vec![];
        let mut ds_n_completed = 0_usize;
        assemble_outgoing_state(
            &mut current_state,
            &unscaled_primal,
            &lag_matrix_buf,
            &layout,
            &state_box,
            stage_lag,
            &mut LagAccumState {
                accumulator: &mut lag_accumulator,
                weight_accum: &mut lag_weight_accum,
            },
            &mut DownstreamAccumState {
                accumulator: &mut ds_accum,
                weight_accum: &mut ds_weight,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n_completed,
                par_order: 0,
            },
        );

        assert_eq!(
            current_state, expected,
            "an in-box clamp must be byte-identical to the pre-clamp assembled state"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn assemble_outgoing_state_clamps_out_of_box() {
        let layout = seam_layout();
        let commit_j = layout.commit_out.start;
        let mut unscaled_primal = seam_primal(10.0, 30.0, 20.0);
        unscaled_primal[commit_j] = -0.5; // 0.5 below the box lower of 0.0
        let lag_matrix_buf = vec![2.0_f64];
        let stage_lag = identity_stage_lag();
        let state_box = StateBox {
            lower: vec![0.0, f64::NEG_INFINITY, 0.0, 0.0],
            upper: vec![100.0, f64::INFINITY, 100.0, 50.0],
        };

        let mut current_state = Vec::new();
        let mut lag_accumulator = vec![0.0_f64; 1];
        let mut lag_weight_accum = vec![0.0_f64; 1];
        let mut ds_accum: Vec<f64> = vec![];
        let mut ds_weight = 0.0_f64;
        let mut ds_completed: Vec<f64> = vec![];
        let mut ds_n_completed = 0_usize;
        assemble_outgoing_state(
            &mut current_state,
            &unscaled_primal,
            &lag_matrix_buf,
            &layout,
            &state_box,
            stage_lag,
            &mut LagAccumState {
                accumulator: &mut lag_accumulator,
                weight_accum: &mut lag_weight_accum,
            },
            &mut DownstreamAccumState {
                accumulator: &mut ds_accum,
                weight_accum: &mut ds_weight,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n_completed,
                par_order: 0,
            },
        );

        assert_eq!(current_state[commit_j], 0.0, "must equal the clamped bound");
    }
}
