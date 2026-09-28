//! Replicated per-node solve for the node-native enumerated backward
//! (`BackwardPassState::run_enumerated_backward`): ranks split a cut-generating
//! node's flattened successor outcome set, each solves its own opening slice,
//! and the outcomes are allgathered so every rank runs the identical flat
//! aggregation and appends the identical cut — the batched cut `allgatherv`
//! (`cut_sync`'s per-level sync) is never invoked for this path, because every
//! rank's `FutureCostFunction` already carries the bit-identical cut locally.
//!
//! [`run_backward_node_replicated`] is the per-node orchestrator: it partitions
//! the node's whole flattened outcome set — every successor CHILD's every
//! opening, canonical `(child, ω)` order — across ranks via
//! [`fn@crate::solve::partition`], and for this rank's slice, loops over every
//! child that intersects it, loading each intersecting child's own LP once
//! ([`solve_replicated_outcome_slice`]) — never pricing a fan against a single
//! child's LP (sddp.md "The branching backward integrates every successor
//! exhaustively"). At world = 1 the slice is the whole outcome set, reducing to
//! one LP load per child, exactly as the by-scenario backward's per-trial-point
//! loop does minus the trial-point axis.

use cobre_comm::Communicator;
use cobre_solver::SolverInterface;

use crate::{
    SddpError,
    context::{StageContext, TrainingContext},
    cut_sync::OutcomeExchangeScratch,
    forward::EnumeratedForwardScratch,
    risk_measure::{BackwardOutcome, RiskMeasure},
    setup::node_graph::OpeningSource,
    solve::partition,
    stage_solve::{StageInputs, run_stage_solve},
    workspace::SolverWorkspace,
};

use super::{
    SuccessorChild, SuccessorOutcomes, SuccessorSpec,
    duals_extraction::extract_state_duals_only,
    lp_setup::{fill_external_opening_noise, patch_opening_bounds},
};

/// Per-rank scratch for [`run_backward_node_replicated`]: the local
/// opening-slice buffer and the allgather machinery that reassembles the full
/// joint outcome set. Reused across nodes and iterations; starts empty (zero
/// footprint under `Traversal::Sampled`, which never touches it).
#[derive(Default)]
pub(crate) struct ReplicatedScratch {
    /// This rank's local `(objective, subgradient)` pairs for its partition of
    /// the current node's flattened outcome set, before the allgather.
    local_slice: Vec<f64>,
    /// Per-rank outcome counts for the allgather (`partition`'s per-rank
    /// share of the node's flattened outcome count).
    outcome_counts: Vec<usize>,
    /// Reused allgather scratch (`OutcomeExchangeScratch::allgather_outcomes`).
    exchange: OutcomeExchangeScratch,
}

/// Per-outcome payload width in [`solve_replicated_outcome_slice`]'s output:
/// one objective followed by the `n_state` subgradient entries.
#[must_use]
pub(crate) fn outcome_stride(n_state: usize) -> usize {
    1 + n_state
}

/// Solve successor child `child`'s local opening range `[local_start,
/// local_end)` (indices into the child's own Ω, canonical order — the same
/// index space `child.outcome_range`'s offset addresses into the node's joint
/// arena) at the shared incoming state `x_hat`, appending each opening's
/// `objective ‖ subgradient` (`1 + cut_state.n_slots()` doubles) to `out`. The
/// child's LP loads once; an `External` child (`openings.len == 1`) reads its
/// declared column via [`fill_external_opening_noise`] instead of the
/// generated opening tree.
///
/// # Errors
///
/// Propagates `SddpError::Infeasible`/`SddpError::Solver` from the slice's
/// solves and `SddpError::Validation` from an `External` child's noise assembly.
// RATIONALE: the per-slice solve inputs (workspace, stage/training contexts,
// successor+child specs, trial state, slice bounds, output buffer) are threaded
// positionally across all three backward variants; bundling them into a struct
// here alone would desync that shared shape without cutting the real coupling.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_replicated_outcome_slice<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    succ: &SuccessorSpec<'_>,
    child: &SuccessorChild<'_>,
    x_hat: &[f64],
    local_start: usize,
    local_end: usize,
    scenario_base: usize,
    iteration: u64,
    out: &mut Vec<f64>,
) -> Result<(), SddpError> {
    let s = succ.successor;
    let n_state = succ.cut_state.n_slots();
    let tree_view = training_ctx.stochastic.tree_view();

    ws.solver.reset_solver_state();
    super::lp_setup::load_backward_lp(ws, child);

    let mut state_duals = std::mem::take(&mut ws.backward_accum.state_duals_buf);

    if child.openings.source == OpeningSource::External {
        debug_assert_eq!(
            (local_start, local_end),
            (0, 1),
            "an External child has exactly one opening"
        );
        let mut buf = std::mem::take(&mut ws.backward_accum.external_noise_buf);
        fill_external_opening_noise(
            training_ctx,
            s,
            child.openings.offset,
            child.successor_node_id,
            &mut buf,
        )?;
        patch_opening_bounds(ws, ctx, training_ctx, &buf, x_hat, s);
        let inputs = StageInputs {
            stage_context: ctx,
            pool: child.successor_pool,
            stored_basis: None,
            stage_index: s,
            scenario_index: scenario_base,
            iteration: Some(iteration),
            node_id: child.successor_node_id,
        };
        let view = run_stage_solve(ws, &inputs)?;
        let objective = extract_state_duals_only(
            &view,
            succ.cut_state,
            &ctx.template(s).col_scale,
            &mut state_duals,
        );
        out.push(objective);
        out.extend_from_slice(&state_duals[..n_state]);
        ws.backward_accum.external_noise_buf = buf;
    } else {
        for local_omega in local_start..local_end {
            let raw_noise = tree_view.opening(s.0, local_omega);
            patch_opening_bounds(ws, ctx, training_ctx, raw_noise, x_hat, s);
            let inputs = StageInputs {
                stage_context: ctx,
                pool: child.successor_pool,
                stored_basis: None,
                stage_index: s,
                scenario_index: scenario_base + local_omega,
                iteration: Some(iteration),
                node_id: child.successor_node_id,
            };
            let view = run_stage_solve(ws, &inputs)?;
            let objective = extract_state_duals_only(
                &view,
                succ.cut_state,
                &ctx.template(s).col_scale,
                &mut state_duals,
            );
            out.push(objective);
            out.extend_from_slice(&state_duals[..n_state]);
        }
    }
    ws.backward_accum.state_duals_buf = state_duals;
    Ok(())
}

/// Solve one cut-generating node's backward cut via the replicated per-node
/// solve: partitions the node's whole flattened successor outcome set across
/// ranks, and for this rank's intersecting children either reads a child's
/// captured terminal-leaf slice from `enumerated_state` (an External terminal
/// leaf, `NodeGraph::is_external_terminal_leaf`, whose forward-solved LP is
/// byte-identical to the one the backward would solve) or
/// solves its local slice ([`solve_replicated_outcome_slice`]), then
/// allgathers every rank's `(objective, subgradient)` slice into the full
/// joint outcome set in canonical order, and runs the identical
/// [`RiskMeasure::aggregate_cut_into`] on every rank — every rank computes the
/// bit-identical `(intercept, coefficients)`, returning the intercept; the
/// aggregated coefficients are left in `ws.backward_accum.agg_coefficients`.
/// The caller appends the cut (`fcf.add_cut`, `forward_pass_index ≡ 0`,
/// sddp.md "Cut pool is append-only") — `fcf` is not threaded through here
/// because its `pools` are already borrowed via `outcomes`, so an internal
/// `add_cut` call would need a second, conflicting mutable borrow of the same
/// value.
///
/// A Generated terminal leaf or an interior successor is never eligible
/// (sddp.md "The branching backward integrates every successor
/// exhaustively") — every non-eligible child still loads and solves its own
/// LP, exactly as before fusion. A predicate-eligible child whose captured
/// slice is unexpectedly absent falls back to solving it directly instead of
/// unwrapping — correctness-preserving, logged as an error since the forward
/// capture should have populated it.
///
/// # Errors
///
/// Propagates a child slice solve's `SddpError`, and
/// `SddpError::Validation`/`SddpError::Communication` from the outcome
/// `allgatherv`.
// RATIONALE: one per-node replicated solve threading disjoint scratch (the
// reused ReplicatedScratch, the solver workspace) and per-node scalars
// (x_hat, pool_id, iteration); splitting further would pass the whole bundle
// for no gain or re-introduce the argument count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_backward_node_replicated<S: SolverInterface + Send, C: Communicator>(
    scratch: &mut ReplicatedScratch,
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    comm: &C,
    risk_measure: &RiskMeasure,
    probabilities: &[f64],
    succ: &SuccessorSpec<'_>,
    outcomes: &SuccessorOutcomes<'_>,
    x_hat: &[f64],
    pool_id: usize,
    iteration: u64,
    enumerated_state: &EnumeratedForwardScratch,
) -> Result<f64, SddpError> {
    let n_state = succ.cut_state.n_slots();
    let n_openings = outcomes.total_outcomes();
    debug_assert_eq!(
        n_openings,
        probabilities.len(),
        "the reified successor outcome set must have as many outcomes as flattened weights"
    );
    let n_ranks = comm.size();
    let my_rank = comm.rank();
    let num_stages = training_ctx.horizon.num_stages();

    let (o_start, o_end) = partition(n_openings, n_ranks, my_rank);

    scratch.local_slice.clear();
    for ci in 0..outcomes.n_children() {
        let child = outcomes.child(ci);
        let lo = o_start.max(child.outcome_range.start);
        let hi = o_end.min(child.outcome_range.end);
        if lo >= hi {
            continue;
        }
        let is_eligible_terminal_leaf = training_ctx
            .node_graph
            .is_external_terminal_leaf(child.successor_node, num_stages);
        let fused = is_eligible_terminal_leaf
            .then(|| enumerated_state.fused_terminal_slice(child.successor_node))
            .flatten();
        if let Some((objective, duals)) = fused {
            debug_assert_eq!(
                (
                    lo - child.outcome_range.start,
                    hi - child.outcome_range.start
                ),
                (0, 1),
                "an External terminal leaf has exactly one opening"
            );
            debug_assert_eq!(
                duals.len(),
                n_state,
                "the fused slice's dual length must equal the cut-state dimension"
            );
            scratch.local_slice.push(objective);
            scratch.local_slice.extend_from_slice(duals);
            continue;
        }
        if is_eligible_terminal_leaf {
            tracing::error!(
                node = ?child.successor_node,
                "fused terminal slice missing for a predicate-eligible External terminal \
                 leaf; falling back to solving this child's LP directly \
                 (correctness-preserving — the forward capture should have populated it)"
            );
        }
        solve_replicated_outcome_slice(
            ws,
            ctx,
            training_ctx,
            succ,
            &child,
            x_hat,
            lo - child.outcome_range.start,
            hi - child.outcome_range.start,
            pool_id,
            iteration,
            &mut scratch.local_slice,
        )?;
    }

    scratch.outcome_counts.clear();
    for r in 0..n_ranks {
        let (s0, e0) = partition(n_openings, n_ranks, r);
        scratch.outcome_counts.push(e0 - s0);
    }
    let full = scratch.exchange.allgather_outcomes(
        &scratch.local_slice,
        n_state,
        &scratch.outcome_counts,
        comm,
    )?;

    while ws.backward_accum.outcomes.len() < n_openings {
        ws.backward_accum.outcomes.push(BackwardOutcome {
            intercept: 0.0,
            coefficients: vec![0.0_f64; n_state],
            objective_value: 0.0,
        });
    }
    let stride = outcome_stride(n_state);
    for (o, out) in ws.backward_accum.outcomes[..n_openings]
        .iter_mut()
        .enumerate()
    {
        let base = o * stride;
        let objective = full[base];
        let coefficients = &full[base + 1..base + 1 + n_state];
        out.coefficients.resize(n_state, 0.0);
        out.coefficients.copy_from_slice(coefficients);
        out.objective_value = objective;
        out.intercept = objective - succ.cut_state.dot_trial_state(coefficients, x_hat);
    }

    ws.backward_accum.agg_coefficients.resize(n_state, 0.0);
    let mut intercept = 0.0_f64;
    risk_measure.aggregate_cut_into(
        &ws.backward_accum.outcomes[..n_openings],
        probabilities,
        &mut intercept,
        &mut ws.backward_accum.agg_coefficients,
        &mut ws.backward_accum.risk_scratch,
    );
    Ok(intercept)
}
