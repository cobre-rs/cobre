//! Enumerated all-paths forward execution engine.
//!
//! Selected by `selection = enumerated`; the sampled M-trajectory driver
//! (`forward_pass_state::run_forward_worker`) owns `selection = sampled` and is
//! untouched. The engine visits every distinct node once per iteration
//! (`enumerated_node_visit_counts`), pins each visit at its parent visit's
//! outgoing state, and reconstructs each root→leaf path's cost from its visited
//! nodes' stage costs — the exact bound `Σ_ℓ P(ℓ)·C(ℓ)` the caller reduces with
//! the compensated `Σ w·c` primitive.
//!
//! Determinism mirrors the by-node backward scheduler's own
//! "By-node scheduler is warm-start-only" contract: a stage's
//! visits are claimed from a shared atomic counter in any order, written into a
//! per-node arena keyed by canonical node position, and aggregated in canonical
//! order — so the exact bound, the lower bound, and the generated cut set are
//! bit-identical across worker, thread, and rank counts.

use std::sync::mpsc::Sender;
use std::time::Instant;

use cobre_core::{TrainingEvent, WorkerTimingPhase};
use cobre_solver::{SolutionView, SolverInterface, StageTemplate};
use cobre_stochastic::{ClassSampleRequest, ForwardNoiseTables, ForwardSampler, SampleRequest};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{
    claim_scatter::{ClaimCursor, canonical_scatter},
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    dcs::{DcsParams, DcsSolveContext, build_initial_resident_set, lazy_solve_preloaded},
    error::SddpError,
    indexer::CutStateProjection,
    noise::{AccumSnapshot, DownstreamAccumState, LagAccumState},
    setup::node_graph::{EnumeratedPlan, NodeGraph, NodeId, NodePos, StageIdx, TypedVec},
    solver_stats::SolverStatsDelta,
    stage_solve::{
        StageInputs, assemble_outgoing_state, fill_unscaled, run_stage_solve,
        run_stage_solve_terminal_static,
    },
    training::{
        backward::extract_state_duals_only,
        stage_solve_prep::{InflowNoise, StageSolvePrep, StageSolvePrepParams, StateSource},
    },
    trajectory::TrajectoryRecord,
    workspace::{BasisStore, CapturedBasis, SolverWorkspace},
};

use super::write_capture_metadata;

/// A node's terminal-leaf subgradient, captured on the forward for the
/// backward to consume instead of re-solving (`NodeGraph::is_external_terminal_leaf`).
/// `captured` is `false` for every non-eligible node; `objective`/`state_duals`
/// are meaningful only when it is `true`.
#[derive(Clone, Default)]
struct FusedTerminalSlice {
    captured: bool,
    objective: f64,
    state_duals: Vec<f64>,
}

/// One node's forward visit outcome, keyed by canonical node position in the
/// per-node arena and produced worker-locally in the claim loop.
#[derive(Clone, Default)]
struct NodeVisit {
    /// Canonical node position this visit solved.
    node: NodePos,
    /// Raw (pre-cumulative-discount) stage cost, the same quantity the sampled
    /// driver accumulates via `cum_d * stage_cost`.
    stage_cost: f64,
    node_id: NodeId,
    /// Outgoing LP state (length `n_state`) — a child visit's incoming state and
    /// the value scattered into every visiting path's record `state` field.
    out_state: Vec<f64>,
    accum: AccumSnapshot,
    /// Post-solve captured basis (frozen path only; `None` on the DCS path).
    basis: Option<CapturedBasis>,
    /// Populated only for an eligible External terminal leaf; see [`FusedTerminalSlice`].
    fused: FusedTerminalSlice,
}

/// Reused per-iteration mutable scratch for the enumerated forward engine —
/// the per-node visit arena and per-worker capture buffers. Meaningful only
/// while [`crate::setup::node_graph::Traversal`] is `Enumerated`; grows to
/// size against the graph-invariant [`EnumeratedPlan`] on the first call and
/// is reused in place thereafter, so no allocation occurs after warm-up.
/// Always present on `ForwardPassState` (never `Option`): dispatch is driven
/// solely by the resolved `Traversal`, read fresh each call, so there is no
/// separate flag this scratch's population could drift out of sync with.
/// `pub` (not `pub(crate)`) solely so `BackwardPassInputs::enumerated_state`
/// (a public field) stays nameable from the crate's own integration tests;
/// every field stays module-private, read only through [`Self::out_state`].
#[derive(Default)]
pub struct EnumeratedForwardScratch {
    /// Per-node visit arena, indexed by canonical node position.
    arena: TypedVec<NodePos, NodeVisit>,
    /// Per-node `arena` population flag, reset each run.
    solved: TypedVec<NodePos, bool>,
    /// Nodes this rank must solve this run (on its assigned paths), reset each run.
    on_my_paths: TypedVec<NodePos, bool>,
    /// Representative local path index for each node — the smallest path in
    /// this rank's range visiting it, minus `fwd_offset`. One value, three
    /// consumption roles at its read sites: the noise-sampling seed
    /// (`local_m`/`global_scenario`), the basis-store row
    /// (`BasisStore::get`/`get_mut`), and the solve/stats bookkeeping index
    /// (`solve_forward_node`'s `scenario_index`).
    m_rep: TypedVec<NodePos, usize>,
    /// Reused per-stage claim unit list (canonical node order).
    stage_units: Vec<NodePos>,
    /// Per-worker capture buffers (worker-local; scattered into `arena` after
    /// each stage's parallel region).
    worker_captures: Vec<Vec<NodeVisit>>,
    /// Per-worker, per-stage solver-stats accumulator, reused across iterations.
    worker_stage_stats: Vec<Vec<SolverStatsDelta>>,
}

impl EnumeratedForwardScratch {
    /// Grow `arena`/`solved`/`on_my_paths`/`m_rep`/`stage_units` to `plan`'s
    /// node count (each `arena` slot's `out_state` to `n_state`); a no-op once
    /// sized, since neither dimension changes across a training run's
    /// iterations. `worker_captures`/`worker_stage_stats` grow separately,
    /// against the worker count, inside [`run_enumerated_forward`].
    fn ensure_sized(&mut self, plan: &EnumeratedPlan, n_state: usize) {
        let n_nodes = plan.parent.len();
        if self.arena.len() == n_nodes {
            return;
        }
        self.arena = (0..n_nodes)
            .map(|_| NodeVisit {
                node: NodePos(usize::MAX),
                stage_cost: 0.0,
                node_id: NodeId(0),
                out_state: vec![0.0; n_state],
                accum: AccumSnapshot::default(),
                basis: None,
                fused: FusedTerminalSlice {
                    captured: false,
                    objective: 0.0,
                    state_duals: Vec::with_capacity(n_state),
                },
            })
            .collect();
        self.solved = vec![false; n_nodes].into();
        self.on_my_paths = vec![false; n_nodes].into();
        self.m_rep = vec![0; n_nodes].into();
        self.stage_units = Vec::with_capacity(n_nodes);
    }

    /// Node `node`'s outgoing state from the most recent [`run_enumerated_forward`]
    /// call — persisted across the fwd→bwd boundary via
    /// `ForwardPassState::enumerated_state` so the enumerated backward can read a
    /// cut-generating node's own trial state directly, never through
    /// `records`/`exchange.state_at`.
    pub(crate) fn out_state(&self, node: NodePos) -> &[f64] {
        &self.arena[node].out_state
    }

    /// Node `node`'s captured terminal-leaf subgradient from the most recent
    /// [`run_enumerated_forward`] call — `Some((objective, state_duals))` for
    /// an eligible External terminal leaf, `None` otherwise. Mirrors
    /// [`Self::out_state`]'s node-indexed read.
    pub(crate) fn fused_terminal_slice(&self, node: NodePos) -> Option<(f64, &[f64])> {
        let fused = &self.arena[node].fused;
        fused
            .captured
            .then_some((fused.objective, fused.state_duals.as_slice()))
    }

    /// Test-only: directly seed node `node`'s `out_state`, growing the arena to
    /// `n_nodes` positions first if needed. Lets a `BackwardPassState`-level unit
    /// test hand-set the persisted arena without running a full forward solve.
    #[cfg(test)]
    pub(crate) fn set_out_state_for_test(&mut self, node: NodePos, n_nodes: usize, state: &[f64]) {
        if self.arena.len() < n_nodes {
            self.arena = (0..n_nodes).map(|_| NodeVisit::default()).collect();
        }
        self.arena[node].out_state.clear();
        self.arena[node].out_state.extend_from_slice(state);
    }
}

/// Read-only per-run captures the enumerated claim loop shares across workers.
pub(crate) struct EnumeratedParams<'a> {
    pub iteration: u64,
    pub fwd_offset: usize,
    pub local_forward_passes: usize,
    pub total_forward_passes: usize,
    pub ctx: &'a StageContext<'a>,
    pub frozen: &'a [StageTemplate],
    pub fcf: &'a FutureCostFunction,
    pub training_ctx: &'a TrainingContext<'a>,
    pub sampler: &'a ForwardSampler<'a>,
    pub noise_tables: &'a ForwardNoiseTables,
    pub dcs: Option<DcsParams>,
    pub event_sender: Option<&'a Sender<TrainingEvent>>,
}

/// Aggregate return of the enumerated forward driver.
pub(crate) struct EnumeratedForwardResult {
    pub scenario_costs: Vec<f64>,
    pub lp_solves: u64,
    pub stage_stats: Vec<SolverStatsDelta>,
}

/// For an eligible node (`cut_state.is_some()`), extract this node's own
/// incoming-state subgradient via [`extract_state_duals_only`] — reused
/// verbatim, never re-derived (`rc / col_scale`, divided; sddp.md "Benders cut
/// sign & subgradient extraction") — before `view` is dropped; otherwise mark
/// `fused_out` uncaptured.
fn capture_fused_terminal_slice(
    view: &SolutionView<'_>,
    col_scale: &[f64],
    cut_state: Option<&CutStateProjection>,
    fused_out: &mut FusedTerminalSlice,
) {
    match cut_state {
        Some(cut_state) => {
            fused_out.objective =
                extract_state_duals_only(view, cut_state, col_scale, &mut fused_out.state_duals);
            fused_out.captured = true;
        }
        None => fused_out.captured = false,
    }
}

/// Solve one node's LP at the incoming state already installed on
/// `ws.current_state`, warm-started from `stored_basis` (matched by node id in
/// `run_stage_solve`, or — at the terminal stage — the 1:1 apply in
/// `run_stage_solve_terminal_static`); leaves `ws.current_state` holding the
/// outgoing state and returns the raw stage cost plus the captured basis
/// (frozen path only) and fused terminal-leaf subgradient.
///
/// Mirrors `forward_pass_state::run_forward_stage`'s solve/record/advance
/// sequence but keys the basis by node — `stored_basis` is read immutably
/// (shared across the claim loop), and the frozen-path capture is written into
/// `basis_out` in place (reusing its buffer via `get_or_insert_with`, so no
/// per-solve allocation), for the caller's sequential post-region scatter. So a
/// dynamically-claimed node never races the session basis store. The DCS path
/// captures no basis (leaving `basis_out` untouched), mirroring
/// `run_forward_stage`. `fused_out` is populated independently of the basis
/// capture, gated on `NodeGraph::is_external_terminal_leaf` — see
/// [`capture_fused_terminal_slice`].
///
/// `scenario_index` (the caller's `EnumeratedForwardScratch::m_rep[node]`) is
/// this call's solve/stats-bookkeeping role only — `DcsSolveContext`'s and
/// `StageInputs`' own `scenario_index` field; the SAME underlying value's
/// basis-store and noise-sampling roles are the caller's concern
/// (`BasisStore::get`/`get_mut`, `global_scenario`).
///
/// # Errors
///
/// Propagates `SddpError::Infeasible`/`SddpError::Solver` from the LP solve.
// RATIONALE: one (node, prefix) LP solve threading disjoint partial borrows of the
// solver workspace, cut state, and basis store; the pin-noise-solve-capture sequence is
// one correctness-critical unit — extracting it would pass the whole workspace for no gain
// or re-introduce the argument count.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn solve_forward_node<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    params: &EnumeratedParams<'_>,
    node: NodePos,
    parent: Option<NodePos>,
    t: StageIdx,
    scenario_index: usize,
    raw_noise: &[f64],
    stored_basis: Option<&CapturedBasis>,
    basis_out: &mut Option<CapturedBasis>,
    fused_out: &mut FusedTerminalSlice,
) -> Result<f64, SddpError> {
    let training_ctx = params.training_ctx;
    let ctx = params.ctx;
    let node_graph = training_ctx.node_graph;
    let pool_id = node_graph.nodes[node].pool_id;
    let node_id = node_graph.node_ids[node];
    let state = training_ctx.state;
    let horizon = training_ctx.horizon;
    let pool = &params.fcf.pools[pool_id];
    let is_terminal = horizon.is_terminal(t.next().0);
    // Fuse the leaf's forward slice only with its CUT-GENERATING PARENT's
    // cut-state projection, never the leaf's own pool — the parent-pool
    // fusion-projection contract (sddp.md). Disabled under DCS: the forward's
    // lazily cut-reduced LP need not match the full frozen template the backward
    // loads (mirrors the `params.dcs.is_none()` basis-capture gate below). A
    // parentless leaf captures nothing → direct backward solve.
    let fusion_cut_state = (params.dcs.is_none()
        && node_graph.is_external_terminal_leaf(node, horizon.num_stages()))
    .then_some(parent)
    .flatten()
    .map(|p| &training_ctx.cut_state_layouts[node_graph.nodes[p].pool_id]);

    // Reset + reload per solve so the landed vertex cannot depend on which nodes
    // a worker solved before it (determinism across worker counts); on the DCS
    // path `run_stage_solve`/`lazy_solve_preloaded` loads the cut-free base
    // instead, so the frozen load is skipped — mirrors the sampled driver.
    ws.solver.reset_solver_state();
    if params.dcs.is_none() {
        ws.solver.load_model(&params.frozen[pool_id]);
    } else {
        ws.solver.load_model(ctx.template(t));
    }

    let prep_params = StageSolvePrepParams {
        state_source: StateSource(&ws.current_state),
        inflow_noise: InflowNoise::Transform,
        raw_noise,
    };
    StageSolvePrep::run(
        &mut ws.solver,
        &mut ws.patch_buf,
        &mut ws.scratch,
        ctx,
        training_ctx,
        t,
        &prep_params,
    );
    if is_terminal && !pool.has_warm_start_cuts() {
        ws.solver.set_col_bounds(&[state.theta], &[0.0], &[0.0]);
    }

    let col_scale = &ctx.template(t).col_scale;
    let mut unscaled_primal: Vec<f64> = std::mem::take(&mut ws.scratch.unscaled_primal);

    let view_objective: f64 = if let Some(dcs_params) = params.dcs {
        build_initial_resident_set(
            pool,
            params.iteration,
            dcs_params.k2,
            &mut ws.backward_accum.dcs_initial_resident,
        );
        let dcs_ctx = DcsSolveContext {
            stage_index: t,
            scenario_index,
            iteration: Some(params.iteration),
            continue_carry: false,
            node_id,
        };
        lazy_solve_preloaded(
            &mut ws.solver,
            ctx.template(t),
            pool,
            state,
            &training_ctx.cut_state_layouts[pool_id],
            col_scale,
            None,
            &ws.backward_accum.dcs_initial_resident,
            &dcs_params,
            &mut ws.backward_accum.dcs_solve,
            dcs_ctx,
        )?;
        let view = ws.backward_accum.dcs_solve.result_view();
        fill_unscaled(&mut unscaled_primal, view.primal, col_scale);
        capture_fused_terminal_slice(&view, col_scale, fusion_cut_state, fused_out);
        view.objective
    } else {
        let inputs = StageInputs {
            stage_context: ctx,
            pool,
            stored_basis,
            stage_index: t,
            scenario_index,
            iteration: Some(params.iteration),
            node_id,
        };
        let view = if is_terminal {
            run_stage_solve_terminal_static(ws, &inputs)?
        } else {
            run_stage_solve(ws, &inputs)?
        };
        fill_unscaled(&mut unscaled_primal, view.primal, col_scale);
        capture_fused_terminal_slice(&view, col_scale, fusion_cut_state, fused_out);
        view.objective
    };

    let d_t = ctx.discount_factor(t);
    // Terminal boundary θ prices the post-horizon value-to-go: KEEP it in the cost
    // (subtracting it, the interior form, drops it from the UB only — understating
    // it below the LB). sddp.md "Terminal boundary FCF in the reported total cost".
    let stage_cost = if is_terminal && pool.has_warm_start_cuts() {
        view_objective * ctx.cost_scale_factor
    } else {
        (view_objective - d_t * unscaled_primal[state.theta]) * ctx.cost_scale_factor
    };

    ws.scratch.lag_matrix_buf.clear();
    ws.scratch
        .lag_matrix_buf
        .extend_from_slice(&ws.current_state[state.inflow_lags.clone()]);

    let stage_lag = ctx.stage_lag(t);
    let downstream_par_order = training_ctx.study_dims.downstream_par_order;
    assemble_outgoing_state(
        &mut ws.current_state,
        &unscaled_primal,
        &ws.scratch.lag_matrix_buf,
        state,
        ctx.state_box(t),
        stage_lag,
        &mut LagAccumState {
            accumulator: &mut ws.scratch.lag_accumulator,
            weight_accum: &mut ws.scratch.lag_weight_accum,
        },
        &mut DownstreamAccumState {
            accumulator: &mut ws.scratch.downstream_accumulator,
            weight_accum: &mut ws.scratch.downstream_weight_accum,
            completed_lags: &mut ws.scratch.downstream_completed_lags,
            n_completed: &mut ws.scratch.downstream_n_completed,
            par_order: downstream_par_order,
        },
    );
    ws.scratch.unscaled_primal = unscaled_primal;

    if params.dcs.is_none() {
        let basis_row_capacity = params.frozen[pool_id].num_rows;
        let cut_row_count = basis_row_capacity.saturating_sub(ctx.template(t).num_rows);
        // Reuse `basis_out`'s buffer if present (get_basis + write_capture_metadata
        // fully overwrite every field), so no per-solve allocation.
        let cap = basis_out.get_or_insert_with(|| {
            CapturedBasis::new(
                ctx.template(t).num_cols,
                basis_row_capacity,
                ctx.template(t).num_rows,
                cut_row_count,
                state.n_state,
                node_id,
            )
        });
        ws.solver.get_basis(&mut cap.basis);
        write_capture_metadata(
            cap,
            pool,
            ctx.template(t).num_rows,
            cut_row_count,
            &ws.current_state[..state.n_state],
            node_id,
        );
    }

    Ok(stage_cost)
}

/// Execute the enumerated all-paths forward for one iteration on this rank.
///
/// Solves every node on this rank's assigned root→leaf paths exactly once
/// (trunk nodes shared by all paths are replicated across ranks; a rank owns
/// the complete subtrees under its path range, so no cross-rank forward state
/// exchange is needed), scatters each node's outcome into the dense
/// `records[m * num_stages + t]` layout the backward pass reads, and returns the
/// per-path costs in canonical path order for the exact `Σ P(ℓ)·C(ℓ)` bound.
///
/// # Errors
///
/// Propagates `SddpError::Infeasible`/`SddpError::Solver` from any node solve.
// RATIONALE: the stage-synced outer loop, per-stage (node, prefix) claim distribution, and
// canonical-order arena scatter are one reproducibility-critical sequence; splitting it
// would fragment the claim-order-independence contract for no clarity gain.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub(crate) fn run_enumerated_forward<S>(
    plan: &EnumeratedPlan,
    scratch: &mut EnumeratedForwardScratch,
    workspaces: &mut [SolverWorkspace<S>],
    basis_store: &mut BasisStore,
    records: &mut [TrajectoryRecord],
    params: &EnumeratedParams<'_>,
) -> Result<EnumeratedForwardResult, SddpError>
where
    S: SolverInterface + Send,
{
    let node_graph = params.training_ctx.node_graph;
    let num_stages = params.training_ctx.horizon.num_stages();
    let n_state = params.training_ctx.state.n_state;
    let n_workers = workspaces.len().max(1);
    let path_range = params.fwd_offset..params.fwd_offset + params.local_forward_passes;

    scratch.ensure_sized(plan, n_state);

    // Reset the per-run arena membership. `on_my_paths`/`m_rep` derive from this
    // rank's path range; `solved` tracks arena population for the child lookup.
    for flag in &mut scratch.on_my_paths {
        *flag = false;
    }
    for flag in &mut scratch.solved {
        *flag = false;
    }
    // Mark every node on this rank's paths and record each node's representative
    // (smallest) local path index for its warm-start basis slot. Paths are
    // canonical, so the smallest visiting path in range is the first walk to
    // reach the node.
    let mut node_seq: Vec<NodePos> = Vec::with_capacity(num_stages);
    for m in path_range.clone() {
        node_seq.clear();
        plan.walk_path(m, num_stages, &mut node_seq);
        let local_m = m - params.fwd_offset;
        for &node in &node_seq {
            if !scratch.on_my_paths[node] {
                scratch.on_my_paths[node] = true;
                scratch.m_rep[node] = local_m;
            }
        }
    }

    if scratch.worker_captures.len() != n_workers {
        scratch.worker_captures = (0..n_workers).map(|_| Vec::new()).collect();
        scratch.worker_stage_stats = (0..n_workers)
            .map(|_| {
                (0..num_stages)
                    .map(|_| SolverStatsDelta::default())
                    .collect()
            })
            .collect();
    }
    for inner in &mut scratch.worker_stage_stats {
        for d in inner.iter_mut() {
            d.reset_in_place();
        }
    }

    let mut lp_solves = 0u64;
    for t in (0..num_stages).map(StageIdx) {
        // Units: the nodes at stage `t` on this rank's paths, canonical order.
        scratch.stage_units.clear();
        for node in node_graph.stage_frontier(t) {
            if scratch.on_my_paths[node] {
                scratch.stage_units.push(node);
            }
        }
        if scratch.stage_units.is_empty() {
            continue;
        }

        let cursor = ClaimCursor::new(scratch.stage_units.len());
        let capture_basis = params.dcs.is_none();
        // Immutable borrows for the parallel region: the previous stage's arena
        // (children read parents), the plan, and the session basis store (warm
        // start read only — captures are written into per-worker slots for a
        // sequential post-region scatter).
        let arena = &scratch.arena;
        let solved = &scratch.solved;
        let m_rep = &scratch.m_rep;
        let stage_units = &scratch.stage_units;
        let parent = &plan.parent;
        let store: &BasisStore = basis_store;

        let worker_counts: Vec<Result<usize, SddpError>> = workspaces
            .par_iter_mut()
            .zip(scratch.worker_captures.par_iter_mut())
            .zip(scratch.worker_stage_stats.par_iter_mut())
            .map(|((ws, captures), stage_stats)| {
                let worker_start = Instant::now();
                let result = enumerated_stage_worker(
                    ws,
                    captures,
                    stage_stats,
                    &cursor,
                    t,
                    params,
                    stage_units,
                    parent,
                    arena,
                    solved,
                    m_rep,
                    store,
                );
                ws.worker_timing_buf.forward_wall_ms +=
                    worker_start.elapsed().as_secs_f64() * 1_000.0;
                result
            })
            .collect();

        // Propagate the first worker error by value (`SddpError` is not `Clone`),
        // mirroring `by_node_finish`'s `res?` handling.
        let counts = worker_counts
            .into_iter()
            .collect::<Result<Vec<usize>, SddpError>>()?;
        lp_solves += counts.iter().map(|&c| c as u64).sum::<u64>();

        // Sequential scatter: copy each worker's captures into the per-node arena
        // and swap each captured basis into the session store, keyed by canonical
        // node — claim/worker order cannot reach the arena, so the outcome is
        // worker-count invariant. Copying (not moving) keeps every worker-slot
        // buffer alive for reuse next stage.
        for (w, i) in canonical_scatter(&counts) {
            let node = scratch.worker_captures[w][i].node;
            // The basis-store role of `m_rep`: the row `BasisStore::get_mut`
            // swaps this node's freshly captured basis into.
            let basis_scenario = scratch.m_rep[node];
            {
                let src = &scratch.worker_captures[w][i];
                let dst = &mut scratch.arena[node];
                dst.node = node;
                dst.stage_cost = src.stage_cost;
                dst.node_id = src.node_id;
                dst.out_state.clear();
                dst.out_state.extend_from_slice(&src.out_state);
                src.accum.copy_into(&mut dst.accum);
                dst.fused.captured = src.fused.captured;
                dst.fused.objective = src.fused.objective;
                dst.fused.state_duals.clear();
                dst.fused
                    .state_duals
                    .extend_from_slice(&src.fused.state_duals);
            }
            scratch.solved[node] = true;
            if capture_basis {
                std::mem::swap(
                    basis_store.get_mut(basis_scenario, node),
                    &mut scratch.worker_captures[w][i].basis,
                );
            }
        }
    }

    let mut stage_stats: Vec<SolverStatsDelta> = (0..num_stages)
        .map(|_| SolverStatsDelta::default())
        .collect();
    for inner in &scratch.worker_stage_stats {
        for (dst, src) in stage_stats.iter_mut().zip(inner.iter()) {
            SolverStatsDelta::accumulate_into(dst, src);
        }
    }

    // Scatter the arena into the dense record layout and reconstruct per-path
    // costs in canonical path order (ascending trajectory id within this rank).
    let mut scenario_costs = Vec::with_capacity(params.local_forward_passes);
    let mut seq: Vec<NodePos> = Vec::with_capacity(num_stages);
    for m in path_range.clone() {
        let local_m = m - params.fwd_offset;
        seq.clear();
        plan.walk_path(m, num_stages, &mut seq);
        debug_assert_eq!(seq.len(), num_stages, "path must span every stage");
        let mut cost = 0.0_f64;
        for (t, &node) in seq.iter().enumerate() {
            debug_assert!(scratch.solved[node], "node on an owned path was not solved");
            let visit = &scratch.arena[node];
            let cum_d = params
                .ctx
                .cumulative_discount_factors
                .get(t)
                .copied()
                .unwrap_or(1.0);
            cost += cum_d * visit.stage_cost;
            let rec = &mut records[local_m * num_stages + t];
            rec.primal.clear();
            rec.dual.clear();
            rec.stage_cost = visit.stage_cost;
            rec.node_id = visit.node_id;
            rec.state.clear();
            rec.state.extend_from_slice(&visit.out_state);
            debug_assert_eq!(
                rec.state.len(),
                n_state,
                "record state must be n_state long"
            );
        }
        scenario_costs.push(cost);
    }

    // Single-rank pins the dedup factor: solving every node once totals
    // `Σ forward_solve_counts`. Multi-rank replicates trunk nodes per rank, so
    // the equality holds only when this rank owns every path.
    #[cfg(debug_assertions)]
    if params.local_forward_passes == params.total_forward_passes {
        let expected = expected_single_rank_solves(node_graph)?;
        debug_assert_eq!(
            lp_solves, expected,
            "single-rank enumerated solve total must equal Σ forward_solve_counts"
        );
    }

    if let Some(sender) = params.event_sender {
        for ws in workspaces.iter() {
            let _ = sender.send(TrainingEvent::WorkerTiming {
                rank: ws.rank,
                worker_id: ws.worker_id,
                iteration: params.iteration,
                phase: WorkerTimingPhase::Forward,
                timings: ws.worker_timing_buf,
            });
        }
    }

    Ok(EnumeratedForwardResult {
        scenario_costs,
        lp_solves,
        stage_stats,
    })
}

/// One worker's claim loop over a stage's node units. Claims units from the
/// shared atomic counter in any order; each solved node's outcome is written into
/// a reused worker-local capture slot for the caller's sequential, canonical-order
/// scatter. Returns the count of slots this worker filled (`captures[..count]`).
/// The capture slots — and every buffer they hold (`out_state`, the accumulator
/// snapshot, the captured basis) — persist across stages and iterations, so no
/// allocation occurs after warm-up.
// RATIONALE: the per-worker claim loop threads disjoint partial borrows of one
// &mut SolverWorkspace plus the shared claim counter and per-node arena; extracting would
// pass the whole workspace for no gain or re-introduce the argument count.
#[allow(clippy::too_many_arguments)]
fn enumerated_stage_worker<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    captures: &mut Vec<NodeVisit>,
    stage_stats: &mut [SolverStatsDelta],
    cursor: &ClaimCursor,
    t: StageIdx,
    params: &EnumeratedParams<'_>,
    stage_units: &[NodePos],
    parent: &TypedVec<NodePos, Option<NodePos>>,
    arena: &TypedVec<NodePos, NodeVisit>,
    solved: &TypedVec<NodePos, bool>,
    m_rep: &TypedVec<NodePos, usize>,
    store: &BasisStore,
) -> Result<usize, SddpError> {
    let node_graph = params.training_ctx.node_graph;
    let state_space = params.training_ctx.state;
    let n_state = state_space.n_state;

    let noise_dim = params.training_ctx.stochastic.dim();
    let mut raw_noise_buf = std::mem::take(&mut ws.scratch.raw_noise_buf);
    raw_noise_buf.resize(noise_dim, 0.0_f64);
    let mut corr_scratch = std::mem::take(&mut ws.scratch.corr_scratch);
    corr_scratch.resize(2 * noise_dim, 0.0_f64);

    #[allow(clippy::cast_possible_truncation)]
    let total_scenarios_u32 = params.total_forward_passes as u32;

    let mut count = 0usize;
    while let Some(u) = cursor.claim() {
        let node = stage_units[u];
        let local_m = m_rep[node];
        let global_scenario = params.fwd_offset + local_m;
        let parent_node = parent[node];

        // Install the incoming state: the parent visit's outgoing state (already
        // scattered in the previous stage's sequential pass), or the initial
        // state at a root.
        ws.current_state.clear();
        if let Some(p) = parent_node {
            debug_assert!(solved[p], "parent visit must be solved before its child");
            ws.current_state.extend_from_slice(&arena[p].out_state);
            arena[p].accum.restore_into(&mut ws.scratch);
        } else {
            ws.current_state
                .extend_from_slice(params.training_ctx.initial_state);
            seed_root_accumulators(ws, params);
        }

        #[allow(clippy::cast_possible_truncation)]
        let (i32_it, s32, t32) = (params.iteration as u32, global_scenario as u32, t.0 as u32);
        let (node_opening_offset, node_opening_len) = node_graph.node_opening_range(node);
        let pinned_scenario = node_graph.node_pinned_scenario(node);

        if parent_node.is_none() {
            let class_req = ClassSampleRequest {
                iteration: i32_it,
                scenario: s32,
                stage: t32,
                stage_idx: t.0,
                total_scenarios: total_scenarios_u32,
                noise_group_id: 0,
                node_opening_offset,
                node_opening_len,
                pinned_scenario,
            };
            params.sampler.apply_initial_state(
                &class_req,
                &mut ws.current_state,
                state_space.inflow_lags.start,
            );
        }

        let noise = params.sampler.sample(SampleRequest {
            iteration: i32_it,
            scenario: s32,
            stage: t32,
            stage_idx: t.0,
            noise_buf: &mut raw_noise_buf,
            corr_scratch: &mut corr_scratch,
            total_scenarios: total_scenarios_u32,
            noise_group_id: params.ctx.noise_group_id_at(t),
            node_opening_offset,
            node_opening_len,
            pinned_scenario,
            tables: params.noise_tables,
        })?;

        // Reuse the capture slot at `count`, growing only until the widest stage's
        // frontier is covered; every buffer inside is then reused across stages
        // and iterations.
        if captures.len() <= count {
            captures.push(NodeVisit::default());
        }
        let stored = store.get(local_m, node);
        let stats_before = ws.solver.statistics();
        // `noise`/`stored` borrow the worker-local `raw_noise_buf` and the shared
        // read-only basis store, never `ws`, so both coexist with the `&mut ws`
        // solve — no per-solve copy needed (mirrors the sampled driver).
        let raw_noise = noise.as_slice();
        let visit = &mut captures[count];
        let stage_cost = solve_forward_node(
            ws,
            params,
            node,
            parent_node,
            t,
            local_m,
            raw_noise,
            stored,
            &mut visit.basis,
            &mut visit.fused,
        )?;
        let delta = SolverStatsDelta::from_snapshots(&stats_before, &ws.solver.statistics());
        SolverStatsDelta::accumulate_into(&mut stage_stats[t.0], &delta);

        visit.node = node;
        visit.stage_cost = stage_cost;
        visit.node_id = node_graph.node_ids[node];
        visit.out_state.clear();
        visit
            .out_state
            .extend_from_slice(&ws.current_state[..n_state]);
        visit.accum.capture_from(&ws.scratch);
        count += 1;
    }

    ws.scratch.raw_noise_buf = raw_noise_buf;
    ws.scratch.corr_scratch = corr_scratch;
    Ok(count)
}

/// Seed the trajectory-carried scratch accumulators at a root visit, matching
/// the sampled driver's stage-0 seed (`DerivedInflowSeeds` copy or zero-fill).
fn seed_root_accumulators<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    params: &EnumeratedParams<'_>,
) {
    if params.training_ctx.lag_accum_seed.is_empty() {
        ws.scratch.lag_accumulator.fill(0.0);
        ws.scratch.lag_weight_accum.fill(0.0);
    } else {
        ws.scratch.lag_accumulator[..params.training_ctx.lag_accum_seed.len()]
            .copy_from_slice(params.training_ctx.lag_accum_seed);
        ws.scratch.lag_weight_accum[..params.training_ctx.lag_weight_seed.len()]
            .copy_from_slice(params.training_ctx.lag_weight_seed);
    }
    ws.scratch.downstream_accumulator.fill(0.0);
    ws.scratch.downstream_weight_accum = 0.0;
    ws.scratch.downstream_completed_lags.fill(0.0);
    ws.scratch.downstream_n_completed = 0;
}

/// The single-rank per-iteration solve total: `Σ forward_solve_counts`.
/// Exposed so the caller can `debug_assert` the enumerated engine's realized
/// single-rank solve count against it (the dedup scale invariant).
///
/// # Errors
///
/// Propagates the `u64` path-product overflow guard.
// Consumed only by the single-rank solve-count debug-assert above; legitimately
// unused in a release non-test build (debug-assertions off), so allow it there.
#[cfg_attr(not(debug_assertions), allow(dead_code))]
pub(crate) fn expected_single_rank_solves(node_graph: &NodeGraph) -> Result<u64, SddpError> {
    Ok(node_graph.forward_solve_counts()?.into_iter().sum())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

    use super::*;
    use cobre_comm::LocalBackend;
    use cobre_core::temporal::StageStateConfig;
    use cobre_solver::ActiveSolver;

    use crate::{
        indexer::StateSpace,
        lead_time::AnticipatedResolution,
        setup::{StudySetup, node_graph::Traversal},
        test_support,
        training::forward::build_sampler_from_ctx,
        workspace::WorkspacePool,
    };

    /// An eligible node's captured `(objective, duals)` equals a direct
    /// [`extract_state_duals_only`] call on the same `view`/`cut_state`/`col_scale` —
    /// [`capture_fused_terminal_slice`]'s "reuse verbatim" contract, proved by
    /// construction rather than by re-deriving the `rc`/`col_scale` math a second time.
    #[test]
    fn capture_fused_terminal_slice_matches_extract_state_duals_only() {
        let state = StateSpace::new(
            2,
            1,
            Vec::new(),
            vec![],
            AnticipatedResolution::default(),
            &[1, 1],
        );
        let cut_state = CutStateProjection::new(
            &state,
            StageStateConfig {
                storage: true,
                inflow_lags: true,
            },
        );
        // A lag slot's incoming column (`inflow_lags` block) lands past
        // `state.n_state`, so this must exceed the largest incoming column
        // `cut_state` resolves, not just `[0, n_state)`.
        let n_cols = 64;
        let reduced_costs: Vec<f64> = (0..n_cols).map(|c| c as f64 + 1.0).collect();
        let col_scale: Vec<f64> = vec![2.0; n_cols];
        let view = SolutionView {
            objective: 42.0,
            primal: &[],
            dual: &[],
            reduced_costs: &reduced_costs,
            iterations: 0,
            solve_time_seconds: 0.0,
        };

        let mut expected_duals = Vec::new();
        let expected_objective =
            extract_state_duals_only(&view, &cut_state, &col_scale, &mut expected_duals);

        let mut fused_out = FusedTerminalSlice::default();
        capture_fused_terminal_slice(&view, &col_scale, Some(&cut_state), &mut fused_out);

        assert!(fused_out.captured);
        assert_eq!(fused_out.objective, expected_objective);
        assert_eq!(fused_out.state_duals, expected_duals);
    }

    /// A `None` `cut_state` (the non-eligible-node signal) resets an
    /// already-populated `fused_out` to uncaptured, not merely leaves it as
    /// constructed — catching a "only ever sets true" regression.
    #[test]
    fn capture_fused_terminal_slice_none_when_ineligible() {
        let view = SolutionView {
            objective: 7.0,
            primal: &[],
            dual: &[],
            reduced_costs: &[],
            iterations: 0,
            solve_time_seconds: 0.0,
        };
        let mut fused_out = FusedTerminalSlice {
            captured: true,
            objective: 99.0,
            state_duals: vec![1.0, 2.0],
        };

        capture_fused_terminal_slice(&view, &[], None, &mut fused_out);

        assert!(!fused_out.captured);
    }

    /// Fresh, unpopulated scaffolding for one [`run_enumerated_forward`] call
    /// over `setup`'s whole graph: a single-worker pool, an empty
    /// [`BasisStore`], and zeroed [`TrajectoryRecord`]s.
    fn fresh_rig(
        setup: &StudySetup,
    ) -> (
        WorkspacePool<ActiveSolver>,
        BasisStore,
        Vec<TrajectoryRecord>,
    ) {
        let node_graph = &setup.inputs.node_graph;
        let total_forward_passes =
            usize::try_from(test_support::node_scenario_count(node_graph).expect("scenario count"))
                .expect("fits usize");
        let comm = LocalBackend;
        let pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("workspace pool");
        let basis_store = BasisStore::new(total_forward_passes, node_graph.nodes.len());
        let records: Vec<TrajectoryRecord> = (0..total_forward_passes * setup.num_stages())
            .map(|_| TrajectoryRecord {
                primal: Vec::new(),
                dual: Vec::new(),
                stage_cost: 0.0,
                node_id: NodeId(0),
                state: Vec::new(),
            })
            .collect();
        (pool, basis_store, records)
    }

    /// Run one `enumerated` forward iteration over `setup`'s whole graph on a
    /// single worker, mirroring `ForwardPassState::run_enumerated`'s param
    /// assembly. `scratch`/`workspaces`/`basis_store`/`records` persist across
    /// repeated calls, so a second call at `iteration + 1` exercises
    /// cross-iteration basis warm-start.
    fn run_iteration(
        setup: &StudySetup,
        scratch: &mut EnumeratedForwardScratch,
        workspaces: &mut [SolverWorkspace<ActiveSolver>],
        basis_store: &mut BasisStore,
        records: &mut [TrajectoryRecord],
        iteration: u64,
        event_sender: Option<&Sender<TrainingEvent>>,
    ) -> EnumeratedForwardResult {
        let node_graph = &setup.inputs.node_graph;
        let stage_ctx = setup.stage_ctx();
        let training_ctx = setup.training_ctx();
        let sampler = build_sampler_from_ctx(&training_ctx).expect("forward sampler");
        let frozen: Vec<StageTemplate> = (0..node_graph.n_pools)
            .map(|p| stage_ctx.templates[node_graph.pool_stage[p].0].clone())
            .collect();
        let traversal = Traversal::resolve(node_graph, true, 0);
        let Traversal::Enumerated(plan) = &traversal else {
            unreachable!("resolve(is_enumerated=true, ..) always yields Enumerated");
        };
        let total_forward_passes =
            usize::try_from(test_support::node_scenario_count(node_graph).expect("scenario count"))
                .expect("fits usize");
        let dcs = training_ctx.dcs.filter(|p| p.is_active(iteration));
        let mut noise_tables = ForwardNoiseTables::default();
        sampler
            .rebuild_noise_tables(
                u32::try_from(iteration).expect("fits u32"),
                u32::try_from(total_forward_passes).expect("fits u32"),
                stage_ctx.noise_group_ids,
                &mut noise_tables,
            )
            .expect("test fixture never exceeds the Sobol dimension cap");

        let params = EnumeratedParams {
            iteration,
            fwd_offset: 0,
            local_forward_passes: total_forward_passes,
            total_forward_passes,
            ctx: &stage_ctx,
            frozen: &frozen,
            fcf: &setup.fcf,
            training_ctx: &training_ctx,
            sampler: &sampler,
            noise_tables: &noise_tables,
            dcs,
            event_sender,
        };

        run_enumerated_forward(plan, scratch, workspaces, basis_store, records, &params)
            .expect("run_enumerated_forward must succeed on a well-formed fixture")
    }

    /// An External, single-opening terminal leaf is captured; the interior
    /// External root is not; the eligible leaves' basis is still captured into
    /// [`BasisStore`] and a second iteration (warm-started from it) succeeds.
    #[test]
    fn enumerated_forward_captures_fused_slice_only_for_eligible_external_terminal_leaves() {
        let setup = test_support::external_distinct_fan_setup(2, 1);
        let node_graph = &setup.inputs.node_graph;
        let num_stages = setup.num_stages();

        let eligible: Vec<NodePos> = (0..node_graph.nodes.len())
            .map(NodePos)
            .filter(|&p| node_graph.is_external_terminal_leaf(p, num_stages))
            .collect();
        let ineligible: Vec<NodePos> = (0..node_graph.nodes.len())
            .map(NodePos)
            .filter(|&p| !node_graph.is_external_terminal_leaf(p, num_stages))
            .collect();
        assert_eq!(
            eligible.len(),
            2,
            "both leaves of a 2-leaf external-distinct fan are eligible"
        );
        assert_eq!(
            ineligible.len(),
            1,
            "only the interior External root is ineligible"
        );

        let (mut pool, mut basis_store, mut records) = fresh_rig(&setup);
        let mut scratch = EnumeratedForwardScratch::default();

        run_iteration(
            &setup,
            &mut scratch,
            &mut pool.workspaces,
            &mut basis_store,
            &mut records,
            1,
            None,
        );

        for &node in &ineligible {
            assert!(
                scratch.fused_terminal_slice(node).is_none(),
                "non-eligible node {node:?} must not be captured"
            );
        }
        for &node in &eligible {
            let (objective, duals) = scratch
                .fused_terminal_slice(node)
                .unwrap_or_else(|| panic!("eligible leaf {node:?} must be captured"));
            assert!(objective.is_finite());
            let pool_id = node_graph.nodes[node].pool_id;
            assert_eq!(
                duals.len(),
                setup.inputs.cut_state_layouts[pool_id].n_slots(),
                "captured duals must span the leaf's own cut-state projection"
            );
            let local_m = scratch.m_rep[node];
            assert!(
                basis_store.get(local_m, node).is_some(),
                "eligible leaf {node:?}'s basis must still be captured into BasisStore"
            );
        }

        run_iteration(
            &setup,
            &mut scratch,
            &mut pool.workspaces,
            &mut basis_store,
            &mut records,
            2,
            None,
        );

        for &node in &eligible {
            assert!(
                scratch.fused_terminal_slice(node).is_some(),
                "eligible leaf {node:?} must stay captured across a warm-started iteration"
            );
            let local_m = scratch.m_rep[node];
            assert!(
                basis_store.get(local_m, node).is_some(),
                "eligible leaf {node:?}'s basis must still be present after iteration 2"
            );
        }
    }

    /// An eligible leaf's captured dual length must span its CUT-GENERATING
    /// PARENT's cut-state projection — never the leaf's own terminal (no-successor,
    /// always-full) pool. [`test_support::external_distinct_fan_setup_heterogeneous_cut_state`]
    /// declares an active inflow-lag slot so the two genuinely diverge
    /// (`build_cut_state_layouts`'s no-successor rule keeps every leaf's own pool
    /// full-dimension regardless of its declared `state_config`); on the ORIGINAL
    /// `external_distinct_fan_setup` fixture (no lag slot) the two pools coincide,
    /// which is exactly the coverage gap that let the wrong-projection capture ship.
    #[test]
    fn enumerated_forward_fused_slice_projects_with_parent_pool_not_leaf_pool() {
        let setup = test_support::external_distinct_fan_setup_heterogeneous_cut_state(2, 1);
        let node_graph = &setup.inputs.node_graph;
        let num_stages = setup.num_stages();

        let eligible: Vec<NodePos> = (0..node_graph.nodes.len())
            .map(NodePos)
            .filter(|&p| node_graph.is_external_terminal_leaf(p, num_stages))
            .collect();
        assert_eq!(
            eligible.len(),
            2,
            "both leaves of a 2-leaf external-distinct fan are eligible"
        );

        for &node in &eligible {
            let leaf_pool = node_graph.nodes[node].pool_id;
            let parent = node_graph
                .node_parent(node)
                .unwrap_or_else(|| panic!("eligible leaf {node:?} must have a parent"));
            let parent_pool = node_graph.nodes[parent].pool_id;
            assert_ne!(
                setup.inputs.cut_state_layouts[leaf_pool].n_slots(),
                setup.inputs.cut_state_layouts[parent_pool].n_slots(),
                "fixture power check: leaf pool {leaf_pool} and parent pool {parent_pool} must \
                 project DIFFERENT dimensions, or this test cannot distinguish the fix from the bug"
            );
        }

        let (mut pool, mut basis_store, mut records) = fresh_rig(&setup);
        let mut scratch = EnumeratedForwardScratch::default();
        run_iteration(
            &setup,
            &mut scratch,
            &mut pool.workspaces,
            &mut basis_store,
            &mut records,
            1,
            None,
        );

        for &node in &eligible {
            let (_, duals) = scratch
                .fused_terminal_slice(node)
                .unwrap_or_else(|| panic!("eligible leaf {node:?} must be captured"));
            let leaf_pool = node_graph.nodes[node].pool_id;
            let parent = node_graph.node_parent(node).expect("checked above");
            let parent_pool = node_graph.nodes[parent].pool_id;
            assert_eq!(
                duals.len(),
                setup.inputs.cut_state_layouts[parent_pool].n_slots(),
                "captured duals must span the CUT-GENERATING PARENT's cut-state projection \
                 (the backward's own `SuccessorSpec::cut_state`), not the leaf's own pool"
            );
            assert_ne!(
                duals.len(),
                setup.inputs.cut_state_layouts[leaf_pool].n_slots(),
                "power check: the leaf's own pool dimension must differ from the parent's, or \
                 this assertion cannot distinguish the fix from the wrong-projection bug"
            );
        }
    }

    /// A Generated terminal leaf is never captured, even though its basis is —
    /// fusion eligibility and basis capture are independent mechanisms.
    #[test]
    fn enumerated_forward_generated_terminal_leaf_stays_uncaptured() {
        let setup = test_support::terminal_generated_fan_setup(2, 1);
        let node_graph = &setup.inputs.node_graph;
        let num_stages = setup.num_stages();

        assert!(
            (0..node_graph.nodes.len())
                .map(NodePos)
                .all(|p| !node_graph.is_external_terminal_leaf(p, num_stages)),
            "a terminal-Generated fan has no eligible node"
        );

        let (mut pool, mut basis_store, mut records) = fresh_rig(&setup);
        let mut scratch = EnumeratedForwardScratch::default();

        run_iteration(
            &setup,
            &mut scratch,
            &mut pool.workspaces,
            &mut basis_store,
            &mut records,
            1,
            None,
        );

        for pos in (0..node_graph.nodes.len()).map(NodePos) {
            assert!(
                scratch.fused_terminal_slice(pos).is_none(),
                "node {pos:?} in a terminal-Generated fan must not be captured"
            );
            let local_m = scratch.m_rep[pos];
            assert!(
                basis_store.get(local_m, pos).is_some(),
                "node {pos:?}'s basis must still be captured regardless of fusion eligibility"
            );
        }
    }

    /// `run_enumerated_forward` emits one `WorkerTiming { phase: Forward }`
    /// event per workspace, each carrying a non-zero `forward_wall_ms` no
    /// larger than the call's own wall-clock elapsed — every worker's
    /// accumulated per-stage busy time is a subset of the call's total wall.
    #[test]
    fn enumerated_forward_emits_worker_timing_per_workspace() {
        let setup = test_support::terminal_generated_fan_setup(2, 1);
        let (mut pool, mut basis_store, mut records) = fresh_rig(&setup);
        let mut scratch = EnumeratedForwardScratch::default();
        let (tx, rx) = std::sync::mpsc::channel::<TrainingEvent>();

        let call_start = Instant::now();
        run_iteration(
            &setup,
            &mut scratch,
            &mut pool.workspaces,
            &mut basis_store,
            &mut records,
            1,
            Some(&tx),
        );
        let call_elapsed_ms = call_start.elapsed().as_secs_f64() * 1_000.0;
        drop(tx);

        let n_workers = pool.workspaces.len();
        let forward_walls: Vec<f64> = rx
            .try_iter()
            .filter_map(|e| match e {
                TrainingEvent::WorkerTiming {
                    phase: WorkerTimingPhase::Forward,
                    timings,
                    ..
                } => Some(timings.forward_wall_ms),
                _ => None,
            })
            .collect();

        assert_eq!(
            forward_walls.len(),
            n_workers,
            "one Forward WorkerTiming event must be emitted per workspace"
        );
        assert!(
            forward_walls.iter().all(|&ms| ms > 0.0),
            "every worker must report a non-zero forward_wall_ms on a genuine (non-fused) \
             real solve, got {forward_walls:?}"
        );
        assert!(
            forward_walls.iter().all(|&ms| ms <= call_elapsed_ms + 1.0),
            "a worker's own accumulated wall time cannot exceed run_enumerated_forward's own \
             call wall ({call_elapsed_ms} ms), got {forward_walls:?}"
        );
    }
}
