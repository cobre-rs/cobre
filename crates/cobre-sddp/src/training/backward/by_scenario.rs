//! Per-trial-point opening-solve dispatch and the deterministic by-scenario kernel.
//!
//! [`StageOpeningSolver`] is the closed two-variant per-opening solve strategy
//! (frozen all-cuts vs lazy resident-set DCS), and `process_by_scenario_backward`
//! drives it: it solves a trial point's openings in a run-constant, rank-invariant
//! `solve_order` permutation but writes and aggregates outcomes by canonical ω, so
//! the generated cut is bit-identical regardless of solve order.

use cobre_solver::SolverInterface;

use crate::{
    SddpError,
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    dcs::{DcsParams, DcsSolveContext, build_initial_resident_set, lazy_solve_preloaded},
    risk_measure::RiskMeasure,
    setup::node_graph::{NodeId, OpeningSource, StageIdx},
    stage_solve::{StageInputs, run_stage_solve},
    state_exchange::ExchangeBuffers,
    workspace::{BasisStoreSliceMut, SolverWorkspace},
};

use super::{
    StagedCut, SuccessorChild, SuccessorOutcomes, SuccessorSpec,
    duals_extraction::{extract_duals_from_view, extract_state_duals_only},
    lp_setup::{
        fill_external_opening_noise, load_backward_lp, patch_opening_bounds, resolve_backward_basis,
    },
    outcome_aggregation::{
        accumulate_dcs_binding_counts, accumulate_opening_outcome, fold_slot_increments_into_sync,
        save_basis_at_omega_zero, write_opening_outcome,
    },
};

/// Per-trial-point opening-solve strategy for the backward pass.
///
/// A closed two-variant sum type dispatched by `match` — never a trait object
/// (CLAUDE.md closed-variant-set rule). The two variants are the two per-opening
/// solve paths:
///
/// - [`StageOpeningSolver::Frozen`]: the frozen all-cuts LP. Cross-iteration warm
///   basis on the first-solved opening, full state+cut dual extraction,
///   frozen-order slot bumps, and first-solved basis capture.
/// - [`StageOpeningSolver::Lazy`]: the resident-set LP (DCS). A cut-free core is
///   loaded once per trial point and reused across the openings; the first-solved
///   opening solves fresh, the rest warm-carry; state-dual-only extraction;
///   row-map-correct slot bumps; no basis capture.
pub(crate) enum StageOpeningSolver {
    /// Frozen all-cuts LP path (no DCS).
    Frozen,
    /// Lazy resident-set LP path (Dynamic Cut Selection). Carries the active
    /// [`DcsParams`] so the per-opening call needs no separate `Option` test.
    Lazy(DcsParams),
}

impl StageOpeningSolver {
    /// `Some(params)` → [`StageOpeningSolver::Lazy`], `None` → [`StageOpeningSolver::Frozen`].
    pub(crate) fn from_dcs_params(dcs_params: Option<DcsParams>) -> Self {
        match dcs_params {
            Some(params) => StageOpeningSolver::Lazy(params),
            None => StageOpeningSolver::Frozen,
        }
    }

    /// Per-CHILD LP load, issued once after `reset_solver_state()` and before that
    /// child's opening solves. Each child loads ITS OWN pool's LP, so a fan never
    /// reuses child 0's LP (the child-0 collapse). One child ⟹ one load per trial
    /// point ⟹ chain byte-parity.
    pub(crate) fn prepare<S: SolverInterface + Send>(
        &self,
        ws: &mut SolverWorkspace<S>,
        ctx: &StageContext<'_>,
        succ: &SuccessorSpec<'_>,
        child: &SuccessorChild<'_>,
        iteration: u64,
    ) {
        match self {
            StageOpeningSolver::Frozen => {
                load_backward_lp(ws, child);
            }
            StageOpeningSolver::Lazy(params) => {
                ws.solver.load_model(ctx.template(succ.successor));
                build_initial_resident_set(
                    child.successor_pool,
                    iteration,
                    params.k2,
                    &mut ws.backward_accum.dcs_initial_resident,
                );
            }
        }
    }

    /// Solve one backward opening and accumulate its outcome, dispatching to the
    /// active variant's path.
    ///
    /// `is_first` is `true` for the trial point's **first-solved** opening — ω=0
    /// under the identity order, else the first entry of the solve order. The frozen
    /// path loads and captures the per-(m, s) stored basis only on it; the lazy
    /// path passes `continue_carry == !is_first`. Decoupling basis identity from the
    /// literal ω=0 is what lets the openings be solved in any order while the
    /// per-(m, s) basis store stays consistent with the actual first solve.
    // Rationale: the args are disjoint borrows (ws, ctx, training_ctx, succ,
    // basis_slice) and per-opening scalars (raw_noise, x_hat, s, scenario,
    // iteration, m, omega, is_first); no natural grouping reduces caller-side
    // borrows.
    #[allow(clippy::too_many_arguments)]
    fn solve_opening<S: SolverInterface + Send>(
        &self,
        ws: &mut SolverWorkspace<S>,
        ctx: &StageContext<'_>,
        training_ctx: &TrainingContext<'_>,
        succ: &SuccessorSpec<'_>,
        child: &SuccessorChild<'_>,
        basis_slice: &mut BasisStoreSliceMut<'_>,
        raw_noise: &[f64],
        x_hat: &[f64],
        s: StageIdx,
        scenario: usize,
        iteration: u64,
        m: usize,
        omega: usize,
        is_first: bool,
    ) -> Result<(), SddpError> {
        match self {
            StageOpeningSolver::Frozen => Self::solve_frozen(
                ws,
                ctx,
                training_ctx,
                succ,
                child,
                basis_slice,
                raw_noise,
                x_hat,
                s,
                scenario,
                iteration,
                m,
                omega,
                is_first,
            ),
            StageOpeningSolver::Lazy(params) => Self::solve_lazy(
                ws,
                ctx,
                training_ctx,
                succ,
                child,
                *params,
                raw_noise,
                x_hat,
                s,
                scenario,
                iteration,
                omega,
                !is_first,
            ),
        }
    }

    /// Frozen all-cuts per-opening solve; captures the first-solved opening's basis.
    // Rationale: see [`StageOpeningSolver::solve_opening`] — disjoint-borrow args
    // with no natural grouping.
    #[allow(clippy::too_many_arguments)]
    fn solve_frozen<S: SolverInterface + Send>(
        ws: &mut SolverWorkspace<S>,
        ctx: &StageContext<'_>,
        training_ctx: &TrainingContext<'_>,
        succ: &SuccessorSpec<'_>,
        child: &SuccessorChild<'_>,
        basis_slice: &mut BasisStoreSliceMut<'_>,
        raw_noise: &[f64],
        x_hat: &[f64],
        s: StageIdx,
        scenario: usize,
        iteration: u64,
        m: usize,
        omega: usize,
        is_first: bool,
    ) -> Result<(), SddpError> {
        patch_opening_bounds(ws, ctx, training_ctx, raw_noise, x_hat, s);

        // Moved out before the solve to avoid a borrow conflict with `view`'s
        // lifetime; pre-warmed capacity is reused across openings.
        let mut state_duals = std::mem::take(&mut ws.backward_accum.state_duals_buf);
        let mut cut_duals = std::mem::take(&mut ws.backward_accum.cut_duals_buf);

        let mut stats_before_omega = std::mem::take(&mut ws.backward_accum.stats_before_buf);
        ws.solver.statistics_into(&mut stats_before_omega);

        // First-solved opening of THIS child warms from and captures its own
        // `(m, child node)` basis; the successor pool and node id are the child's.
        let stored_basis = if is_first {
            resolve_backward_basis(basis_slice, m, child.successor_node)
        } else {
            None
        };
        let inputs = StageInputs {
            stage_context: ctx,
            pool: child.successor_pool,
            stored_basis,
            stage_index: s,
            scenario_index: scenario,
            iteration: Some(iteration),
            node_id: child.successor_node_id,
        };

        let view = run_stage_solve(ws, &inputs)?;

        // Statistics must be captured after `view` is dropped (the `let _ = view`
        // below).
        let objective = extract_duals_from_view(
            &view,
            succ.cut_state,
            &ctx.template(s).col_scale,
            child,
            &mut state_duals,
            &mut cut_duals,
        );
        let _ = view;

        ws.backward_accum.state_duals_buf = state_duals;
        ws.backward_accum.cut_duals_buf = cut_duals;

        let mut stats_after_omega = std::mem::take(&mut ws.backward_accum.stats_after_buf);
        ws.solver.statistics_into(&mut stats_after_omega);

        accumulate_opening_outcome(
            ws,
            child,
            succ.cut_state,
            omega,
            objective,
            x_hat,
            &stats_before_omega,
            &stats_after_omega,
        );

        ws.backward_accum.stats_before_buf = stats_before_omega;
        ws.backward_accum.stats_after_buf = stats_after_omega;

        if is_first {
            save_basis_at_omega_zero(ws, child, basis_slice, m, x_hat);
        }

        Ok(())
    }

    /// Lazy resident-set per-opening solve under Dynamic Cut Selection.
    ///
    /// The cut-free core and metadata seed are loaded ONCE per trial point by
    /// [`StageOpeningSolver::prepare`]; this routine never reloads or re-seeds, only
    /// patches the opening's bounds and runs [`lazy_solve_preloaded`]:
    ///
    /// - `continue_carry == false` (first-solved opening): the lazy loop resets the
    ///   carried row map, appends the seed, and solves with `stored_basis = None`
    ///   (no warm per-opening basis, but reusing the loaded cut-free core).
    /// - `continue_carry == true` (subsequent openings): warm-carry the prior
    ///   opening's LP, basis, and monotonically grown resident cut set, adding only
    ///   the cuts this opening additionally violates (the paper's §3.4 base
    ///   recovery, extended across the openings).
    ///
    /// The gradient and intercept are read from the final all-satisfied LP via
    /// [`extract_state_duals_only`] in both cases. Binding-count metadata is
    /// maintained slot-correct under the lazy layout via
    /// [`accumulate_dcs_binding_counts`], feeding the per-stage
    /// `metadata_sync_contribution` allreduce without altering the gradient.
    ///
    /// First-solved basis capture is intentionally NOT performed: a captured basis
    /// would describe the frozen layout, not the DCS resident subset.
    // Rationale: the args are disjoint borrows (ws, ctx, training_ctx, succ) and
    // per-opening scalars (params, raw_noise, x_hat, s, scenario, iteration,
    // omega); no natural grouping reduces caller-side borrows.
    #[allow(clippy::too_many_arguments)]
    fn solve_lazy<S: SolverInterface + Send>(
        ws: &mut SolverWorkspace<S>,
        ctx: &StageContext<'_>,
        training_ctx: &TrainingContext<'_>,
        succ: &SuccessorSpec<'_>,
        child: &SuccessorChild<'_>,
        params: DcsParams,
        raw_noise: &[f64],
        x_hat: &[f64],
        s: StageIdx,
        scenario: usize,
        iteration: u64,
        omega: usize,
        continue_carry: bool,
    ) -> Result<(), SddpError> {
        let state = training_ctx.state;
        // The DCS LP must start from the cut-free base template (`ctx.templates[s]`),
        // NOT the child's frozen template: the frozen template already carries the
        // active cut rows, and loading it would make the lazy loop's fresh CutRowMap
        // treat those slots as non-resident and append them again (duplicate rows,
        // broken laziness).
        let core = ctx.template(s);
        let col_scale = &ctx.template(s).col_scale;

        patch_opening_bounds(ws, ctx, training_ctx, raw_noise, x_hat, s);

        let mut stats_before_omega = std::mem::take(&mut ws.backward_accum.stats_before_buf);
        ws.solver.statistics_into(&mut stats_before_omega);

        // Moved out so `extract_state_duals_only` can fill it while `view` holds an
        // immutable borrow of the sibling `dcs_solve` field; restored at the end.
        let mut state_duals = std::mem::take(&mut ws.backward_accum.state_duals_buf);

        let dcs_ctx = DcsSolveContext {
            stage_index: s,
            scenario_index: scenario,
            iteration: Some(iteration),
            continue_carry,
            node_id: child.successor_node_id,
        };
        // The DCS LP renders the CHILD's pool's cuts into stage `s` (== successor);
        // its projection is the child's pool's, resolved by the child's node POSITION
        // (`child.pool_id`) — never `nodes[stage]`, which mis-indexes the node array
        // with a stage on a branching graph. `succ.cut_state` (the generating node's
        // projection) is used only for the incoming extraction below.
        let successor_cut_layout = &training_ctx.cut_state_layouts[child.pool_id];
        lazy_solve_preloaded(
            &mut ws.solver,
            core,
            child.successor_pool,
            state,
            successor_cut_layout,
            col_scale,
            None,
            &ws.backward_accum.dcs_initial_resident,
            &params,
            &mut ws.backward_accum.dcs_solve,
            dcs_ctx,
        )?;
        let view = ws.backward_accum.dcs_solve.result_view();

        let objective =
            extract_state_duals_only(&view, succ.cut_state, col_scale, &mut state_duals);

        // `view` and `dcs_solve.row_map` both borrow `dcs_solve` immutably (so they
        // coexist); `slot_increments` is a distinct field borrowed mutably. The
        // child's own pool region (`metadata_offset..`) keeps a fan's sibling pools
        // from colliding on a shared slot index.
        accumulate_dcs_binding_counts(
            view.dual,
            &ws.backward_accum.dcs_solve.row_map,
            child.successor_pool,
            child.cut_activity_tolerance,
            &mut ws.backward_accum.slot_increments[child.metadata_offset..],
        );
        let _ = view;

        ws.backward_accum.state_duals_buf = state_duals;

        let mut stats_after_omega = std::mem::take(&mut ws.backward_accum.stats_after_buf);
        ws.solver.statistics_into(&mut stats_after_omega);

        write_opening_outcome(
            ws,
            succ.cut_state,
            omega,
            objective,
            x_hat,
            &stats_before_omega,
            &stats_after_omega,
        );

        ws.backward_accum.stats_before_buf = stats_before_omega;
        ws.backward_accum.stats_after_buf = stats_after_omega;

        Ok(())
    }
}

/// Process one trial point `m` in the backward pass, iterating over the node's
/// successor outcome set child-by-child.
///
/// Each child loads ITS OWN LP (frozen template, delta cut batch, pool, basis key,
/// External column) and solves its own openings, so pricing every child against
/// child 0's LP — the child-0 collapse, which silently misprices a non-interchangeable
/// fan — is unrepresentable. A chain is the one-element case: one child, one LP load
/// per trial point, byte-identical. The single joint `aggregate_cut_into` over the
/// flattened outcome arena is unchanged (joint risk applied once, not nested).
///
/// On the frozen path the post-solve basis is written into `basis_slice` only at each
/// child's first-solved opening; later writes are forbidden (retained-LU corruption).
/// The DCS arm skips capture (see [`StageOpeningSolver::solve_lazy`]).
// RATIONALE: 13 args required — each is a disjoint borrow (ws, ctx, training_ctx, exchange,
// succ, outcomes, basis_slice, opening_solver) or a plain scalar (fwd_offset, iteration, m,
// arena_offset) or a risk slice. Merging into a struct would add indirection without reducing
// the caller's borrow count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_by_scenario_backward<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    exchange: &ExchangeBuffers,
    fwd_offset: usize,
    node_visit_offset: usize,
    iteration: u64,
    risk_measures: &[RiskMeasure],
    succ: &SuccessorSpec<'_>,
    outcomes: &SuccessorOutcomes<'_>,
    basis_slice: &mut BasisStoreSliceMut<'_>,
    opening_solver: &StageOpeningSolver,
    m: usize,
    compacted: usize,
    arena_offset: usize,
) -> Result<StagedCut, SddpError> {
    let tree_view = training_ctx.stochastic.tree_view();
    let x_hat = exchange.state_at(succ.my_rank, m);
    let scenario = fwd_offset + m;
    let s = succ.successor;

    debug_assert_eq!(
        ws.backward_accum.per_opening_stats.len(),
        succ.probabilities.len(),
        "per_opening_stats must be initialised to n_openings before each stage's trial-point loop"
    );
    debug_assert_eq!(
        outcomes.total_outcomes(),
        succ.probabilities.len(),
        "the reified outcome set's total outcomes must equal the flattened weight vector length"
    );

    // Openings within a child are SOLVED in `solve_order(s)` (a run-constant,
    // rank-invariant permutation) but WRITTEN and AGGREGATED by CANONICAL ω (the
    // child's `outcome_range` offset plus the local ω), so the generated cut is
    // bit-identical regardless of solve order: the reorder changes only the
    // warm-start chain, never which cuts are produced.
    let solve_order = tree_view.solve_order_data(s.0);

    for ci in 0..outcomes.n_children() {
        let child = outcomes.child(ci);
        // Fresh cold head per child: each child loads a different LP, so its
        // warm-start chain across its own openings starts clean (CLP determinism).
        // One child ⟹ once per trial point ⟹ chain byte-parity.
        ws.solver.reset_solver_state();
        opening_solver.prepare(ws, ctx, succ, &child, iteration);

        // An External child reads its declared column `eta_slice(s, offset)`
        // (assembled once per child — invariant across that child's ω); a Generated
        // child keeps `tree_view.opening` byte-for-byte. The assembly is per-child,
        // never once-from-child-0.
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

        if let Some(buf) = &external_noise {
            // External child: a single realization (`len == 1`), no within-child order.
            debug_assert_eq!(child.openings.len, 1, "an External child has one opening");
            let omega = child.outcome_range.start;
            opening_solver.solve_opening(
                ws,
                ctx,
                training_ctx,
                succ,
                &child,
                basis_slice,
                buf,
                x_hat,
                s,
                scenario,
                iteration,
                m,
                omega,
                true,
            )?;
        } else {
            debug_assert_eq!(
                solve_order.len(),
                child.openings.len,
                "a Generated child's opening count must equal stage {s}'s own opening count"
            );
            for (solve_pos, &local_omega_u32) in solve_order.iter().enumerate() {
                let local_omega = local_omega_u32 as usize;
                let raw_noise = tree_view.opening(s.0, local_omega);
                let omega = child.outcome_range.start + local_omega;
                // First-solved opening of THIS child: owns the per-(m, child node)
                // basis load/capture and the fresh (non-warm-carry) solve.
                let is_first = solve_pos == 0;
                opening_solver.solve_opening(
                    ws,
                    ctx,
                    training_ctx,
                    succ,
                    &child,
                    basis_slice,
                    raw_noise,
                    x_hat,
                    s,
                    scenario,
                    iteration,
                    m,
                    omega,
                    is_first,
                )?;
            }
        }

        if let Some(buf) = external_noise {
            ws.backward_accum.external_noise_buf = buf;
        }
    }

    // Aggregate into `agg_coefficients`, then copy into this trial point's arena
    // slot so the bytes outlive the parallel closure without a per-cut allocation.
    let n_openings = succ.probabilities.len();
    let mut agg_intercept = 0.0_f64;
    risk_measures[succ.t.0].aggregate_cut_into(
        &ws.backward_accum.outcomes[..n_openings],
        succ.probabilities,
        &mut agg_intercept,
        &mut ws.backward_accum.agg_coefficients,
        &mut ws.backward_accum.risk_scratch,
    );
    let n_state = ws.backward_accum.agg_coefficients.len();
    let coefficients_range = arena_offset..arena_offset + n_state;
    debug_assert!(
        coefficients_range.end <= ws.backward_accum.agg_arena.len(),
        "agg_arena must be sized to cover this trial point's slot before the solve"
    );
    ws.backward_accum.agg_arena[coefficients_range.clone()]
        .copy_from_slice(&ws.backward_accum.agg_coefficients[..n_state]);
    // The cut's `forward_pass_index` addresses `CutPool`'s per-iteration slot
    // block at stride `visit_bound[pool]` (the node's own routed visit count),
    // NOT `forward_passes`, so it must be the node-relative position within THIS
    // node's GLOBAL routed subset: `node_visit_offset + compacted`, where
    // `node_visit_offset` is the visits to this node from strictly-lower ranks.
    // On a single rank the offset is `0`, leaving `compacted`; on a single-node
    // level it is `fwd_offset` and `compacted == m`, reducing to the global
    // `scenario` (byte-neutral, cross-rank-disjoint on a chain). The GLOBAL
    // `scenario` (`fwd_offset + m`) is the wrong-but-compiling alternative: it
    // indexes `[0, forward_passes)`, overshooting the smaller `visit_bound` stride
    // on a fan; the bare `fwd_offset + compacted` is the OTHER wrong alternative —
    // `fwd_offset` is the global forward-pass offset, not the node's own lower-rank
    // visit prefix, so across ranks it overshoots the stride and collides.
    let node_relative_index = node_visit_offset + compacted;
    debug_assert!(
        u32::try_from(node_relative_index).is_ok(),
        "node-relative forward-pass index overflows u32"
    );
    #[allow(clippy::cast_possible_truncation)]
    let forward_pass_index = node_relative_index as u32;
    fold_slot_increments_into_sync(
        &ws.backward_accum.slot_increments,
        &mut ws.backward_accum.metadata_sync_contribution,
        outcomes.total_metadata_len(),
    );
    Ok(StagedCut {
        trial_state_idx: m,
        intercept: agg_intercept,
        coefficients_range,
        forward_pass_index,
    })
}

/// Merge each worker's staged cuts into `staged_cuts_buf`, order them by trial
/// state, and commit one cut per trial point to the FCF. `trial_state_idx` is
/// globally unique across workers, so the committed order — and thus the cut
/// set — is independent of worker count and completion order. `staged_cuts_buf`
/// is the caller-owned scratch (`BackwardPassState::staged_cuts_buf`), cleared
/// and reused, never reallocated on the hot path.
///
/// # Errors
/// Returns the first failing worker's [`SddpError`]; on error no cut is committed.
// Rationale: disjoint borrows (worker_staged, workspaces, fcf, the staged-cut
// scratch) plus the node/pool/iteration commit scalars; a struct would add
// indirection without reducing the caller's borrow count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn by_scenario_finish<S: SolverInterface>(
    worker_staged: Vec<Result<(usize, Vec<StagedCut>), SddpError>>,
    workspaces: &[SolverWorkspace<S>],
    trial_points: &[usize],
    cut_n_state: usize,
    fcf: &mut FutureCostFunction,
    node_id: NodeId,
    pool: usize,
    iteration: u64,
    staged_cuts_buf: &mut Vec<(usize, StagedCut)>,
) -> Result<usize, SddpError> {
    staged_cuts_buf.clear();
    for worker_result in worker_staged {
        let (w, cuts) = worker_result?;
        staged_cuts_buf.extend(cuts.into_iter().map(|cut| (w, cut)));
    }
    // `trial_state_idx` is the SOLE sort key: globally unique across workers
    // (disjoint contiguous partitions), so the merge order is identical regardless
    // of worker index.
    staged_cuts_buf.sort_by_key(|(_, cut)| cut.trial_state_idx);
    debug_assert_eq!(staged_cuts_buf.len(), trial_points.len());
    for (w, cut) in &*staged_cuts_buf {
        let range = cut.coefficients_range.clone();
        let arena = &workspaces[*w].backward_accum.agg_arena;
        debug_assert!(
            range.len() == cut_n_state && range.end <= arena.len(),
            "coefficients_range must span exactly the pool's cut n_state and lie within the worker arena"
        );
        fcf.add_cut(
            node_id,
            pool,
            iteration,
            cut.forward_pass_index,
            cut.intercept,
            &arena[range],
        );
    }
    Ok(staged_cuts_buf.len())
}
