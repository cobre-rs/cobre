//! Opening-block backward stage scheduler: the work unit is an
//! opening-block of one trial point (`B_s` consecutive `solve_order` positions),
//! not a whole trial point. Units are claimed dynamically by workers from a
//! shared atomic counter (work-stealing dispatch) in block-major order — every
//! `m` of one block is claimed before the next block, in the `block_order`
//! permutation ([`hardest_first_block_order`]: hardest-first by the previous iteration's
//! mean pivot, or [`identity_block_order`] when hardest-first is disabled).
//! Each unit's
//! chain is self-contained (head anchored on the forward capture at `(m, s)`,
//! warm-continue inside the block), so results are independent of the worker
//! count and of the claim/block order, by construction. Outcomes are
//! accumulated worker-locally and scattered into a per-`(m, ω)` arena after the
//! parallel region; cut aggregation runs canonically over ω per trial point, in
//! ascending m (sddp.md "By-node scheduler is warm-start-only").

use std::cmp;
use std::num::NonZeroUsize;
use std::time::Instant;

use cobre_solver::SolverInterface;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{
    SddpError,
    claim_scatter::{ClaimCursor, canonical_scatter},
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    risk_measure::{BackwardOutcome, RiskMeasure},
    setup::node_graph::{NodeId, NodePos, OpeningSource},
    solver_stats::SolverStatsDelta,
    stage_solve::{StageInputs, run_stage_solve},
    state_exchange::ExchangeBuffers,
    workspace::{BasisStore, ByNodeScratch, SolverWorkspace},
};

use super::{
    SuccessorOutcomes, SuccessorSpec,
    duals_extraction::extract_duals_from_view,
    lp_setup::{fill_external_opening_noise, load_backward_lp, patch_opening_bounds},
    outcome_aggregation::{accumulate_opening_outcome, fold_slot_increments_into_sync},
};

/// One solved (trial point, opening) outcome produced by an opening-block unit.
pub(crate) struct OpeningOutcome {
    /// Position of the trial point within this node's routed set (the arena row),
    /// NOT the rank-local trial-point index — the two coincide only on a
    /// single-node level, where routing is the identity `0..local_work`.
    pub(crate) trial_pos: usize,
    pub(crate) omega: usize,
    pub(crate) outcome: BackwardOutcome,
}

/// Resolves the per-stage opening-block size `B_s`. Never a function of worker
/// or rank count — by-node determinism (sddp.md) requires the block boundaries
/// to be fixed independent of parallelism.
pub(crate) fn resolve_block_size(
    n_openings: usize,
    node_block_size: Option<NonZeroUsize>,
) -> usize {
    let requested = node_block_size.map_or(n_openings.div_ceil(2), NonZeroUsize::get);
    requested.min(n_openings)
}

/// Resolve the number of opening-blocks `n_blocks` a stage's `n_openings`
/// split into under `block_size` — the single owner of this formula, shared
/// by [`process_stage_backward_by_node`] (the claim-loop unit count) and its
/// caller (the block-pivot merge's per-stage block count).
pub(crate) fn by_node_block_count(n_openings: usize, block_size: usize) -> usize {
    n_openings.div_ceil(block_size.max(1))
}

/// Fill `out[..n_blocks]` with the identity permutation `0..n_blocks` — the
/// canonical claim order (hardest-first disabled) and [`hardest_first_block_order`]'s pre-sort
/// seed. `out` is pre-allocated (`ByNodeScratch::block_order`, reserved
/// capacity `n_blocks_max`); cleared and refilled in place, never reallocated
/// on the hot path.
pub(crate) fn identity_block_order(n_blocks: usize, out: &mut Vec<u32>) {
    debug_assert!(
        n_blocks <= out.capacity(),
        "n_blocks ({n_blocks}) exceeds the block_order scratch's reserved capacity ({}) — \
         resizing must never reallocate on the opening-block hot path",
        out.capacity()
    );
    out.clear();
    #[allow(clippy::cast_possible_truncation)]
    out.extend((0..n_blocks).map(|b| b as u32));
}

/// Compute the hardest-block-first (longest-processing-time, LPT) claim
/// order for one stage's `n_blocks`
/// opening-blocks from the previous iteration's per-block mean
/// `(simplex_iterations sum, count)` pivot row: block indices sorted by
/// descending mean, ties — including an all-zero row (iteration 1, or any
/// block with no prior data) — broken by ascending index, a total order.
/// Compares means by cross-multiplying `sum`/`count` in `u128` rather than
/// dividing `f64`, avoiding both the `count == 0` divide-by-zero and a
/// float-compare edge case.
pub(crate) fn hardest_first_block_order(
    prev_row: &[(u64, u64)],
    n_blocks: usize,
    out: &mut Vec<u32>,
) {
    debug_assert!(prev_row.len() >= n_blocks);
    identity_block_order(n_blocks, out);
    out.sort_by(|&a, &b| {
        let (sum_a, count_a) = prev_row[a as usize];
        let (sum_b, count_b) = prev_row[b as usize];
        match (count_a == 0, count_b == 0) {
            (true, true) => a.cmp(&b),
            (true, false) => cmp::Ordering::Greater,
            (false, true) => cmp::Ordering::Less,
            (false, false) => {
                let cross_a = u128::from(sum_a) * u128::from(count_b);
                let cross_b = u128::from(sum_b) * u128::from(count_a);
                cross_b.cmp(&cross_a).then_with(|| a.cmp(&b))
            }
        }
    });
}

/// Solve one backward stage opening-block-style; returns, per worker, either the error
/// that aborted its claim loop or `(worker_index, outcome_count)` — the count
/// of entries this worker recorded into its own
/// `ws.backward_accum.opening_outcomes_buf[..outcome_count]`, which
/// [`by_node_finish`] resolves back through `workspaces[worker_index]`.
// Rationale: mirrors `process_by_scenario_backward`'s disjoint-borrow argument list;
// too_many_lines is the claim loop's single sequential region (LP load, warm-chain
// solve, dual extraction, pivot accumulation, outcome recording) — splitting it
// would scatter state the next step in the same loop iteration reads.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn process_stage_backward_by_node<S: SolverInterface + Send>(
    workspaces: &mut [SolverWorkspace<S>],
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    trial_points: &[usize],
    exchange: &ExchangeBuffers,
    fwd_offset: usize,
    iteration: u64,
    succ: &SuccessorSpec<'_>,
    outcomes: &SuccessorOutcomes<'_>,
    basis_store: &BasisStore,
    block_size: usize,
    block_order: &[u32],
) -> Vec<Result<(usize, usize), SddpError>> {
    let n_openings = succ.probabilities.len();
    let cut_n_state = succ.cut_state.n_slots();
    // Sum over every child's populated pool: each child's binding increments land
    // in ITS OWN region (`metadata_offset + slot`), so a fan's sibling pools never
    // collide — the same total the by-scenario path sizes.
    let pop = outcomes.total_metadata_len();
    let n_blocks = by_node_block_count(n_openings, block_size);
    debug_assert_eq!(
        block_order.len(),
        n_blocks,
        "block_order must hold exactly n_blocks entries"
    );
    debug_assert_eq!(
        outcomes.total_outcomes(),
        n_openings,
        "the reified successor outcome set's total outcomes must equal the flattened weight length"
    );
    debug_assert!(
        (0..outcomes.n_children()).all(|ci| {
            let c = outcomes.child(ci);
            c.metadata_offset + c.populated_count <= pop
        }),
        "each child's binding-metadata region [metadata_offset, +populated_count) must fit the \
         flattened total {pop}"
    );
    let n_trial = trial_points.len();
    let cursor = ClaimCursor::new(n_trial * n_blocks);
    let tree_view = training_ctx.stochastic.tree_view();
    let s = succ.successor;
    let solve_order = tree_view.solve_order_data(s.0);

    workspaces
        .par_iter_mut()
        .enumerate()
        .map(|(w, ws)| {
            while ws.backward_accum.outcomes.len() < n_openings {
                ws.backward_accum.outcomes.push(BackwardOutcome {
                    intercept: 0.0,
                    coefficients: vec![0.0_f64; cut_n_state],
                    objective_value: 0.0,
                });
            }
            for outcome in &mut ws.backward_accum.outcomes[..n_openings] {
                outcome.coefficients.resize(cut_n_state, 0.0_f64);
            }
            if ws.backward_accum.slot_increments.len() < pop {
                ws.backward_accum.slot_increments.resize(pop, 0u64);
            }
            if ws.backward_accum.metadata_sync_contribution.len() < pop {
                ws.backward_accum
                    .metadata_sync_contribution
                    .resize(pop, 0u64);
            }
            ws.backward_accum.metadata_sync_contribution[..pop].fill(0);
            ws.backward_accum.slot_increments[..pop].fill(0);
            ws.backward_accum
                .per_opening_stats
                .resize_with(n_openings, SolverStatsDelta::default);
            for slot in &mut ws.backward_accum.per_opening_stats[..n_openings] {
                *slot = SolverStatsDelta::default();
            }
            if ws.backward_accum.block_pivot_sum.len() < n_blocks {
                ws.backward_accum.block_pivot_sum.resize(n_blocks, 0u64);
                ws.backward_accum.block_pivot_count.resize(n_blocks, 0u64);
            }
            ws.backward_accum.block_pivot_sum[..n_blocks].fill(0);
            ws.backward_accum.block_pivot_count[..n_blocks].fill(0);

            let worker_stage_wall_start = Instant::now();
            let mut count = 0usize;

            while let Some(u) = cursor.claim() {
                let pos = u % n_trial;
                let b = block_order[u / n_trial] as usize;
                let m = trial_points[pos];
                let x_hat = exchange.state_at(succ.my_rank, m);
                let scenario = fwd_offset + m;

                let block_lo = b * block_size;
                let block_hi = (block_lo + block_size).min(n_openings);

                // Canonical child order (sddp.md "The branching backward integrates
                // every successor exhaustively"): each run loads THAT child's LP, never
                // child 0's.
                for ci in 0..outcomes.n_children() {
                    let child = outcomes.child(ci);
                    let run_lo = child.outcome_range.start.max(block_lo);
                    let run_hi = child.outcome_range.end.min(block_hi);
                    if run_lo >= run_hi {
                        continue;
                    }

                    ws.solver.reset_solver_state();
                    load_backward_lp(ws, &child);

                    // take/fill/restore reuses `external_noise_buf`: no hot-path allocation.
                    let mut external_noise: Option<Vec<f64>> = None;
                    if child.openings.source == OpeningSource::External {
                        let mut buf = std::mem::take(&mut ws.backward_accum.external_noise_buf);
                        fill_external_opening_noise(
                            training_ctx,
                            s,
                            child.openings.offset,
                            child.successor_node_id,
                            &mut buf,
                        )?;
                        external_noise = Some(buf);
                    }

                    for (run_local_idx, pp) in (run_lo..run_hi).enumerate() {
                        let sp = pp - child.outcome_range.start;
                        let (omega, raw_noise): (usize, &[f64]) = if let Some(buf) = &external_noise
                        {
                            debug_assert_eq!(
                                child.openings.len, 1,
                                "an External child has exactly one opening"
                            );
                            (child.outcome_range.start, buf.as_slice())
                        } else {
                            // sddp.md "Backward opening order is warm-start-only".
                            let local_omega = solve_order[sp] as usize;
                            (
                                child.outcome_range.start + local_omega,
                                tree_view.opening(s.0, local_omega),
                            )
                        };
                        patch_opening_bounds(ws, ctx, training_ctx, raw_noise, x_hat, s);

                        let mut state_duals =
                            std::mem::take(&mut ws.backward_accum.state_duals_buf);
                        let mut cut_duals = std::mem::take(&mut ws.backward_accum.cut_duals_buf);
                        let mut stats_before =
                            std::mem::take(&mut ws.backward_accum.stats_before_buf);
                        ws.solver.statistics_into(&mut stats_before);

                        // Only each child run's first-solved outcome warm-starts from
                        // the captured `(m, child node)` basis; later outcomes of the
                        // SAME child warm-continue its retained factorization. A child
                        // boundary starts a new run: a different LP is loaded, so
                        // continuing the prior factorization is invalid, not merely
                        // suboptimal.
                        let stored = if run_local_idx == 0 {
                            basis_store.get(m, child.successor_node)
                        } else {
                            None
                        };
                        let inputs = StageInputs {
                            stage_context: ctx,
                            pool: child.successor_pool,
                            stored_basis: stored,
                            stage_index: s,
                            scenario_index: scenario,
                            iteration: Some(iteration),
                            node_id: child.successor_node_id,
                        };
                        let view = run_stage_solve(ws, &inputs)?;
                        let objective = extract_duals_from_view(
                            &view,
                            succ.cut_state,
                            &ctx.template(s).col_scale,
                            &child,
                            &mut state_duals,
                            &mut cut_duals,
                        );
                        let _ = view;
                        ws.backward_accum.state_duals_buf = state_duals;
                        ws.backward_accum.cut_duals_buf = cut_duals;

                        let mut stats_after =
                            std::mem::take(&mut ws.backward_accum.stats_after_buf);
                        ws.solver.statistics_into(&mut stats_after);
                        // Counters are monotonically increasing (mirrors
                        // `SolverStatsDelta::from_snapshots`); the running sum below
                        // saturates against overflow across many iterations.
                        let opening_simplex_iters =
                            stats_after.total_iterations - stats_before.total_iterations;
                        ws.backward_accum.block_pivot_sum[b] = ws.backward_accum.block_pivot_sum[b]
                            .saturating_add(opening_simplex_iters);
                        ws.backward_accum.block_pivot_count[b] =
                            ws.backward_accum.block_pivot_count[b].saturating_add(1);
                        accumulate_opening_outcome(
                            ws,
                            &child,
                            succ.cut_state,
                            omega,
                            objective,
                            x_hat,
                            &stats_before,
                            &stats_after,
                        );
                        ws.backward_accum.stats_before_buf = stats_before;
                        ws.backward_accum.stats_after_buf = stats_after;

                        // A reused slot's `coefficients` may carry a DIFFERENT prior
                        // stage's `cut_n_state` (a successor disabling a state group
                        // shrinks it, a later stage regrows it), so it is resized to
                        // THIS stage's `cut_n_state` before every write — otherwise
                        // `copy_from_slice` panics on a length mismatch.
                        if ws.backward_accum.opening_outcomes_buf.len() <= count {
                            ws.backward_accum.opening_outcomes_buf.push(OpeningOutcome {
                                trial_pos: 0,
                                omega: 0,
                                outcome: BackwardOutcome {
                                    intercept: 0.0,
                                    coefficients: vec![0.0_f64; cut_n_state],
                                    objective_value: 0.0,
                                },
                            });
                        }
                        let recorded = &mut ws.backward_accum.opening_outcomes_buf[count];
                        recorded.outcome.coefficients.resize(cut_n_state, 0.0_f64);
                        recorded.trial_pos = pos;
                        recorded.omega = omega;
                        recorded
                            .outcome
                            .coefficients
                            .copy_from_slice(&ws.backward_accum.outcomes[omega].coefficients);
                        recorded.outcome.intercept = ws.backward_accum.outcomes[omega].intercept;
                        recorded.outcome.objective_value =
                            ws.backward_accum.outcomes[omega].objective_value;
                        count += 1;
                    }

                    if let Some(buf) = external_noise {
                        ws.backward_accum.external_noise_buf = buf;
                    }
                }
            }

            fold_slot_increments_into_sync(
                &ws.backward_accum.slot_increments,
                &mut ws.backward_accum.metadata_sync_contribution,
                pop,
            );

            ws.worker_timing_buf.backward_wall_ms +=
                worker_stage_wall_start.elapsed().as_secs_f64() * 1_000.0;

            Ok((w, count))
        })
        .collect()
}

/// Scatter worker outcomes into the pre-allocated per-`(routed position, ω)`
/// arena, aggregate each routed trial point's cut canonically over ω, and insert
/// into the FCF in ascending routed position — which is ascending trial-point
/// index (sddp.md "By-node scheduler is warm-start-only" — never claim order).
///
/// `scratch` is `BackwardPassState::by_node_scratch`, sized once by `set_scheduler`;
/// the scatter overwrites `arena[0..trial_points.len() * n_openings]` in full before
/// the aggregation loop reads it, so no clear pass is required between stages. Each
/// touched slot's `coefficients` (and `coeffs_buf`) resize to THIS pool's
/// `cut_n_state`, which may differ from a prior pool's — always within the
/// capacity `ByNodeScratch::sized` reserved at the run's global `n_state`, so
/// this never reallocates on the hot path.
// Rationale: mirrors the staged-cut merge's disjoint-borrow argument list.
#[allow(clippy::too_many_arguments)]
pub(crate) fn by_node_finish<S: SolverInterface>(
    worker_out: Vec<Result<(usize, usize), SddpError>>,
    workspaces: &[SolverWorkspace<S>],
    trial_points: &[usize],
    n_openings: usize,
    cut_n_state: usize,
    probabilities: &[f64],
    risk_measure: &RiskMeasure,
    fcf: &mut FutureCostFunction,
    node_id: NodeId,
    pool: usize,
    iteration: u64,
    node_visit_offset: usize,
    scratch: &mut ByNodeScratch,
) -> Result<usize, SddpError> {
    let n_trial = trial_points.len();
    let arena_len = n_trial * n_openings;
    debug_assert!(
        scratch.arena.len() >= arena_len,
        "ByNodeScratch arena must already cover trial_points.len() * n_openings ({arena_len}); got \
         arena len = {}",
        scratch.arena.len(),
    );
    debug_assert!(
        cut_n_state <= scratch.coeffs_buf.capacity(),
        "cut_n_state ({cut_n_state}) exceeds ByNodeScratch::coeffs_buf's reserved capacity \
         ({}) — resizing must never reallocate on the opening-block hot path",
        scratch.coeffs_buf.capacity()
    );
    scratch.coeffs_buf.resize(cut_n_state, 0.0_f64);
    let counts: Vec<usize> = worker_out
        .into_iter()
        .map(|res| res.map(|(_, c)| c))
        .collect::<Result<_, SddpError>>()?;
    for (w, i) in canonical_scatter(&counts) {
        let recorded = &workspaces[w].backward_accum.opening_outcomes_buf[i];
        let slot = &mut scratch.arena[recorded.trial_pos * n_openings + recorded.omega];
        debug_assert!(
            cut_n_state <= slot.coefficients.capacity(),
            "cut_n_state ({cut_n_state}) exceeds this arena slot's reserved capacity ({}) — \
             resizing must never reallocate on the opening-block hot path",
            slot.coefficients.capacity()
        );
        slot.coefficients.resize(cut_n_state, 0.0_f64);
        slot.coefficients
            .copy_from_slice(&recorded.outcome.coefficients);
        slot.intercept = recorded.outcome.intercept;
        slot.objective_value = recorded.outcome.objective_value;
    }
    let mut cuts_added = 0usize;
    // Ascending routed position == ascending trial-point index (`trial_points`
    // is built ascending), so the cut set is claim/worker-count independent
    // (sddp.md "By-node scheduler is warm-start-only").
    for pos in 0..n_trial {
        let mut intercept = 0.0_f64;
        risk_measure.aggregate_cut_into(
            &scratch.arena[pos * n_openings..(pos + 1) * n_openings],
            probabilities,
            &mut intercept,
            &mut scratch.coeffs_buf,
            &mut scratch.risk_scratch,
        );
        // The FCF slot addresses the pool's per-iteration block at stride
        // `visit_bound[pool]` (this node's routed visit count), so the tag is the
        // NODE-RELATIVE position `node_visit_offset + pos`: `pos` is the compacted
        // index (the arena is keyed by routed position) and `node_visit_offset` is
        // this node's visits from strictly-lower ranks, keeping the slot globally
        // unique once a multi-rank fan makes the by-node path multi-branching. On a
        // single-node level `node_visit_offset == fwd_offset` and on a single rank it
        // is `0`, both reducing to the pre-change value. `fwd_offset + pos` (the global
        // forward-pass offset) is the wrong-but-compiling alternative — it overshoots
        // the smaller `visit_bound` stride across ranks on a fan; mirrors the
        // by-scenario path.
        let node_relative_index = node_visit_offset + pos;
        debug_assert!(
            u32::try_from(node_relative_index).is_ok(),
            "node-relative forward-pass index overflows u32"
        );
        #[allow(clippy::cast_possible_truncation)]
        fcf.add_cut(
            node_id,
            pool,
            iteration,
            node_relative_index as u32,
            intercept,
            &scratch.coeffs_buf,
        );
        cuts_added += 1;
    }
    debug_assert_eq!(
        cuts_added, n_trial,
        "by_node_finish must add exactly one cut per routed trial point"
    );
    Ok(cuts_added)
}

/// Merge each worker's per-block `(simplex_iterations sum, count)` — filled
/// during [`process_stage_backward_by_node`]'s claim loop, aggregated over every
/// trial point `m` a worker claimed for a block by construction — into
/// [`ByNodeScratch::block_pivots`] at `node_key`'s row. Keyed by the GENERATING
/// node position, not the successor stage: sibling fan nodes at one level own
/// distinct rows, so each computes its hardest-first order from its own history
/// (on a chain, one node per stage, the row is byte-identical to a stage key).
/// Telemetry-only: touches no cut-generation state, so it is cut-neutral by
/// construction (disjoint from `by_node_finish`'s per-`(m, ω)` arena and FCF
/// insertion).
///
/// `per_worker` yields each worker's `(sums, counts)` slices, both indexed
/// `[0, n_blocks)`; `n_blocks <= scratch.n_blocks_max` is the caller's
/// contract (checked in debug builds).
pub(crate) fn merge_block_pivots<'a>(
    per_worker: impl Iterator<Item = (&'a [u64], &'a [u64])>,
    n_blocks: usize,
    node_key: NodePos,
    scratch: &mut ByNodeScratch,
) {
    debug_assert!(
        n_blocks <= scratch.n_blocks_max,
        "n_blocks ({n_blocks}) must not exceed scratch.n_blocks_max ({})",
        scratch.n_blocks_max
    );
    let row = scratch.block_pivot_row(node_key, n_blocks);
    debug_assert!(
        row.end <= scratch.block_pivots.len(),
        "node_key ({node_key}) row must fit scratch.block_pivots (len {})",
        scratch.block_pivots.len()
    );
    for (sums, counts) in per_worker {
        debug_assert!(sums.len() >= n_blocks && counts.len() >= n_blocks);
        for (b, idx) in row.clone().enumerate() {
            let (sum, count) = scratch.block_pivots[idx];
            scratch.block_pivots[idx] =
                (sum.saturating_add(sums[b]), count.saturating_add(counts[b]));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{
        by_node_block_count, hardest_first_block_order, identity_block_order, merge_block_pivots,
        resolve_block_size,
    };
    use crate::setup::NodePos;
    use crate::workspace::ByNodeScratch;

    #[test]
    fn resolve_block_size_defaults_to_half_openings_rounded_up() {
        assert_eq!(resolve_block_size(7, None), 4);
        assert_eq!(resolve_block_size(8, None), 4);
        assert_eq!(resolve_block_size(1, None), 1);
    }

    #[test]
    fn resolve_block_size_clamps_configured_size_to_n_openings() {
        assert_eq!(resolve_block_size(7, NonZeroUsize::new(3)), 3);
        assert_eq!(resolve_block_size(7, NonZeroUsize::new(100)), 7);
    }

    #[test]
    fn by_node_block_count_matches_div_ceil() {
        assert_eq!(by_node_block_count(7, 4), 2);
        assert_eq!(by_node_block_count(8, 4), 2);
        assert_eq!(by_node_block_count(1, 1), 1);
    }

    /// Two workers' per-block `simplex_iterations` sums/counts merged into the
    /// same `(stage, block)` bucket yield the expected combined sum, count,
    /// and mean — the aggregate-over-`m` contract (sddp.md "Hardest-first claim
    /// order is result-neutral") at its smallest scale.
    #[test]
    fn merge_block_pivots_sums_two_workers_into_one_bucket() {
        let n_blocks = 2_usize;
        let bwd_max_openings = 4_usize;
        let num_stages = 3_usize;
        let mut scratch = ByNodeScratch::sized(1, bwd_max_openings, 1, num_stages);
        let node_key = NodePos(1);

        let worker0_sum = [10_u64, 0];
        let worker0_count = [2_u64, 0];
        let worker1_sum = [5_u64, 0];
        let worker1_count = [1_u64, 0];
        let per_worker = [
            (worker0_sum.as_slice(), worker0_count.as_slice()),
            (worker1_sum.as_slice(), worker1_count.as_slice()),
        ];

        merge_block_pivots(per_worker.into_iter(), n_blocks, node_key, &mut scratch);

        let idx = node_key.0 * bwd_max_openings;
        assert_eq!(
            scratch.block_pivots[idx],
            (15, 3),
            "block 0 must hold the summed (sum, count) across both workers"
        );
        #[allow(clippy::float_cmp, clippy::cast_precision_loss)]
        {
            assert_eq!(
                scratch.block_pivots[idx].0 as f64 / scratch.block_pivots[idx].1 as f64,
                5.0
            );
        }
        assert_eq!(
            scratch.block_pivots[idx + 1],
            (0, 0),
            "block 1 (untouched by either worker) must stay zeroed"
        );
    }

    /// Two sibling nodes at one level (distinct `node_key`s) merge into distinct
    /// rows — the CA5 re-key: their per-block pivot histories never mix, so each
    /// node's hardest-first order is computed from its own claims. On a chain (one
    /// node per stage) the row a node writes is byte-identical to a stage key.
    #[test]
    fn merge_block_pivots_keys_distinct_rows_per_node() {
        let n_blocks = 1_usize;
        let bwd_max_openings = 4_usize;
        // Row count follows the node axis: a fan level carries more nodes than
        // stages, so the grid is sized to the (larger) node count.
        let num_nodes = 5_usize;
        let mut scratch = ByNodeScratch::sized(1, bwd_max_openings, 1, num_nodes);

        let three_sum = [7_u64];
        let three_count = [1_u64];
        let four_sum = [40_u64];
        let four_count = [2_u64];

        // Sibling node 3 and sibling node 4 at the same level.
        merge_block_pivots(
            [(three_sum.as_slice(), three_count.as_slice())].into_iter(),
            n_blocks,
            NodePos(3),
            &mut scratch,
        );
        merge_block_pivots(
            [(four_sum.as_slice(), four_count.as_slice())].into_iter(),
            n_blocks,
            NodePos(4),
            &mut scratch,
        );

        assert_eq!(
            scratch.block_pivots[3 * bwd_max_openings],
            (7, 1),
            "sibling node 3's row must hold only its own claims"
        );
        assert_eq!(
            scratch.block_pivots[4 * bwd_max_openings],
            (40, 2),
            "sibling node 4's row must hold only its own claims — never mixed with node 3's"
        );
    }

    #[test]
    fn identity_block_order_fills_ascending_prefix() {
        let mut out = vec![9_u32; 5];
        identity_block_order(3, &mut out);
        assert_eq!(out, vec![0, 1, 2]);
    }

    #[test]
    fn hardest_first_block_order_sorts_descending_mean_ties_by_index() {
        // Means: block 0 = 5.0, block 1 = 10.0, block 2 = 4.0, block 3 = 5.0 (ties block 0).
        let prev_row = [(10_u64, 2_u64), (30, 3), (8, 2), (10, 2)];
        let mut out = vec![0_u32; 4];
        hardest_first_block_order(&prev_row, 4, &mut out);
        assert_eq!(
            out,
            vec![1, 0, 3, 2],
            "blocks sort by descending mean; the (block 0, block 3) tie at mean 5.0 breaks by \
             ascending index"
        );
    }

    #[test]
    fn hardest_first_block_order_falls_back_to_identity_when_no_prior_data() {
        let prev_row = [(0_u64, 0_u64); 3];
        let mut out = vec![0_u32; 3];
        hardest_first_block_order(&prev_row, 3, &mut out);
        assert_eq!(
            out,
            vec![0, 1, 2],
            "an all-zero previous-iteration row (iteration 1) must yield the identity order"
        );
    }

    #[test]
    fn hardest_first_block_order_sorts_zero_count_block_as_least_hard() {
        // Block 1 has no prior data (undefined mean); blocks 0 and 2 have real means
        // 5.0 and 10.0. Block 1 must sort last, never panic on the count == 0 divide.
        let prev_row = [(10_u64, 2_u64), (0, 0), (30, 3)];
        let mut out = vec![0_u32; 3];
        hardest_first_block_order(&prev_row, 3, &mut out);
        assert_eq!(
            out,
            vec![2, 0, 1],
            "a zero-count block sorts least-hard, after every block with real prior data"
        );
    }
}
