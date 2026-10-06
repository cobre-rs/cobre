//! Enumerated (census) simulation execution engine — a node-native fork of the
//! sampled simulation, mirroring
//! `training::forward::enumerated::run_enumerated_forward` step-for-step:
//! own-set marking, a stage-synchronous node-parallel solve sweep (each
//! distinct `on_my_paths` node solved exactly once), then a sequential
//! per-path re-expansion that streams every owned leaf's result.
//!
//! Distribution model: replicate. A rank owns the complete subtrees under its
//! assigned leaf-path range ([`assign_scenarios`]); a node shared by several of
//! the rank's own paths is solved once and copied into every owning leaf's
//! output row — never re-solved per path. A trunk node shared across ranks is
//! replicated: each owning rank solves it independently and, by the
//! determinacy lemma (single predecessor + one opening per node ⇒ a node's
//! solved outcome is a pure function of the node), lands on the identical
//! vertex, so no cross-rank state exchange or interior-node read ever occurs.

use std::ops::Range;

use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use cobre_comm::Communicator;
use cobre_solver::{SolverInterface, StageTemplate};
use cobre_stochastic::{ForwardNoiseTables, ForwardSampler, SampleRequest};

use crate::{
    claim_scatter::{ClaimCursor, canonical_scatter},
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    noise::AccumSnapshot,
    setup::node_graph::{EnumeratedPlan, NodeId, NodePos, StageIdx, TypedVec},
    simulation::{
        error::SimulationError,
        extraction::assign_scenarios,
        pipeline::{
            SIMULATION_ITERATION, SimLookups, SimScenarioLoadSpec, SimStageIds,
            SimulationOutputSpec, WorkerCosts, WorkerStats, dispatch_scenario_result,
            reset_scenario_state, solve_simulation_stage,
        },
        state::SimulationInputs,
        types::SimulationStageResult,
    },
    solver_stats::SolverStatsDelta,
    workspace::SolverWorkspace,
};

/// One node's census visit outcome — the simulation analog of the forward's
/// `NodeVisit`. Carries no captured basis (the census warm-starts read-only
/// from `node_bases[n]`) and, unlike the forward, the full per-entity
/// extraction the sampled simulation also produces. `out_state`/`accum` feed
/// children during the sweep; `result`/`immediate_cost`/`stats` feed
/// re-expansion.
struct SimNodeVisit {
    node: NodePos,
    node_id: NodeId,
    immediate_cost: f64,
    out_state: Vec<f64>,
    accum: AccumSnapshot,
    stats: SolverStatsDelta,
    result: SimulationStageResult,
}

/// Reused per-run mutable scratch: the per-node visit arena, own-set marking,
/// and per-worker capture buffers — the analog of
/// `training::forward::enumerated::EnumeratedForwardScratch`. The simulation
/// pass runs once per call (no cross-iteration reuse the training forward's
/// own scratch needs), so this is sized once at the top of
/// [`run_enumerated_simulation`] and dropped at the end of that call.
struct EnumeratedSimScratch {
    /// Per-node visit arena, indexed by canonical node position; `None` until
    /// the sweep solves that node.
    arena: TypedVec<NodePos, Option<SimNodeVisit>>,
    /// Nodes this rank must solve (on its assigned paths).
    on_my_paths: TypedVec<NodePos, bool>,
    /// Representative global scenario id for each node — the smallest owned
    /// leaf visiting it — read as the noise-sampling seed and the
    /// solve/stats-bookkeeping index. Unlike the forward's `m_rep`, this plays
    /// no basis-store role: the census warm-start basis is keyed by node
    /// (`node_bases[n]`), never by scenario.
    m_rep: TypedVec<NodePos, u32>,
    /// Reused per-stage claim-unit list (canonical node order).
    stage_units: Vec<NodePos>,
    /// Per-worker capture slots, `None` until the claim loop fills them.
    /// `Option`-wrapped so the sequential scatter *moves* each capture into the
    /// arena via `Option::take`, rather than copying its buffers and reusing the
    /// slot the way the training forward engine's arena does: that sweep runs
    /// across many training iterations and amortizes a per-iteration copy, but
    /// this one runs once per call, so a move allocates each arena buffer exactly
    /// once with zero copies — copy-and-reuse would only add work here.
    /// (`SimNodeVisit` also has no `Default` — it carries a
    /// [`SimulationStageResult`], which has none.)
    worker_captures: Vec<Vec<Option<SimNodeVisit>>>,
}

impl EnumeratedSimScratch {
    fn new(plan: &EnumeratedPlan, n_workers: usize) -> Self {
        let n_nodes = plan.parent.len();
        Self {
            arena: (0..n_nodes).map(|_| None).collect(),
            on_my_paths: vec![false; n_nodes].into(),
            m_rep: vec![0u32; n_nodes].into(),
            stage_units: Vec::with_capacity(n_nodes),
            worker_captures: (0..n_workers.max(1)).map(|_| Vec::new()).collect(),
        }
    }
}

/// Read-only per-run captures the claim loop shares across workers — the
/// simulation analog of `training::forward::enumerated::EnumeratedParams`.
struct EnumeratedSimParams<'p> {
    ctx: &'p StageContext<'p>,
    fcf: &'p FutureCostFunction,
    training_ctx: &'p TrainingContext<'p>,
    output: &'p SimulationOutputSpec<'p>,
    load_spec: &'p SimScenarioLoadSpec<'p>,
    sampler: &'p ForwardSampler<'p>,
    noise_tables: &'p ForwardNoiseTables,
    lookups: &'p SimLookups,
    total_scenarios: u32,
}

/// Mark every node on this rank's assigned leaf paths `on_my_paths`, and
/// record each node's representative (smallest) owned global scenario id —
/// the single own-set-marking pass both the sweep and re-expansion read.
fn mark_own_paths(
    plan: &EnumeratedPlan,
    scenario_range: Range<u32>,
    num_stages: usize,
    scratch: &mut EnumeratedSimScratch,
) {
    let mut node_seq: Vec<NodePos> = Vec::with_capacity(num_stages);
    for p in scenario_range {
        node_seq.clear();
        plan.walk_path(p as usize, num_stages, &mut node_seq);
        for &node in &node_seq {
            if !scratch.on_my_paths[node] {
                scratch.on_my_paths[node] = true;
                scratch.m_rep[node] = p;
            }
        }
    }
}

/// One worker's claim loop over a stage's node units — the simulation analog
/// of `training::forward::enumerated::enumerated_stage_worker`. Returns the
/// count of slots filled (`captures[..count]`), consumed by the caller's
/// canonical-order [`canonical_scatter`].
///
/// # Errors
///
/// Propagates `SimulationError::LpInfeasible`/`SolverError` from the LP solve,
/// and `SimulationError::InvalidConfiguration` if a claimed node's parent has
/// no populated arena entry (a marking/sweep defect).
// RATIONALE: the per-worker claim loop threads disjoint partial borrows of one
// &mut SolverWorkspace plus the shared claim counter, per-node arena, and read-only
// run-scoped params; extracting would pass the whole workspace for no gain or
// re-introduce the argument count.
#[allow(clippy::too_many_arguments)]
fn enumerated_sim_stage_worker<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    captures: &mut Vec<Option<SimNodeVisit>>,
    cursor: &ClaimCursor,
    t: StageIdx,
    params: &EnumeratedSimParams<'_>,
    stage_units: &[NodePos],
    parent: &TypedVec<NodePos, Option<NodePos>>,
    arena: &TypedVec<NodePos, Option<SimNodeVisit>>,
    m_rep: &TypedVec<NodePos, u32>,
) -> Result<usize, SimulationError> {
    let training_ctx = params.training_ctx;
    let node_graph = training_ctx.node_graph;
    let state = training_ctx.state;
    let n_state = state.n_state;

    let mut raw_noise_buf = std::mem::take(&mut ws.scratch.raw_noise_buf);
    raw_noise_buf.resize(training_ctx.stochastic.dim(), 0.0_f64);
    let mut corr_scratch = std::mem::take(&mut ws.scratch.corr_scratch);
    corr_scratch.resize(2 * training_ctx.stochastic.dim(), 0.0_f64);

    let mut count = 0usize;
    while let Some(u) = cursor.claim() {
        let node = stage_units[u];
        let global_scenario = m_rep[node];

        if let Some(p) = parent[node] {
            let parent_visit = arena[p].as_ref().ok_or_else(|| {
                SimulationError::InvalidConfiguration(format!(
                    "enumerated census sweep: parent node {p} of {node} has no populated \
                     arena entry (a marking/sweep defect)"
                ))
            })?;
            ws.solver.reset_solver_state();
            ws.current_state.clear();
            ws.current_state.extend_from_slice(&parent_visit.out_state);
            parent_visit.accum.restore_into(&mut ws.scratch);
        } else {
            reset_scenario_state(
                ws,
                params.sampler,
                global_scenario,
                params.total_scenarios,
                state.inflow_lags.start,
                training_ctx,
                node,
            );
        }

        let (node_opening_offset, node_opening_len) = node_graph.node_opening_range(node);
        let pinned_scenario = node_graph.node_pinned_scenario(node);
        #[allow(clippy::cast_possible_truncation)]
        let t32 = t.0 as u32;
        let noise = params.sampler.sample(SampleRequest {
            iteration: SIMULATION_ITERATION,
            scenario: global_scenario,
            stage: t32,
            stage_idx: t.0,
            noise_buf: &mut raw_noise_buf,
            corr_scratch: &mut corr_scratch,
            total_scenarios: params.total_scenarios,
            noise_group_id: params.ctx.noise_group_id_at(t),
            node_opening_offset,
            node_opening_len,
            pinned_scenario,
            tables: params.noise_tables,
        })?;

        let pool_id = node_graph.nodes[node].pool_id;
        let node_id = node_graph.node_ids[node];
        let output_stage_id = params
            .ctx
            .study_stage_id(t)
            .unwrap_or_else(|| i32::try_from(t.0).unwrap_or(i32::MAX));
        #[allow(clippy::cast_sign_loss)]
        let stage_id_u32 = output_stage_id as u32;

        let load_spec = params.load_spec.stage(node, pool_id);
        let stats_before = ws.solver.statistics();
        let raw_noise = noise.as_slice();
        let (immediate_cost, result) = solve_simulation_stage(
            ws,
            params.ctx,
            params.fcf,
            training_ctx,
            &load_spec,
            params.output,
            &SimStageIds {
                t,
                stage_id_u32,
                scenario_id: global_scenario,
                node,
                node_id,
            },
            params.lookups,
            raw_noise,
        )?;
        let visit_stats = SolverStatsDelta::from_snapshots(&stats_before, &ws.solver.statistics());

        if captures.len() <= count {
            captures.push(None);
        }
        let mut accum = AccumSnapshot::default();
        accum.capture_from(&ws.scratch);
        captures[count] = Some(SimNodeVisit {
            node,
            node_id,
            immediate_cost,
            out_state: ws.current_state[..n_state].to_vec(),
            accum,
            stats: visit_stats,
            result,
        });
        count += 1;
    }

    ws.scratch.raw_noise_buf = raw_noise_buf;
    ws.scratch.corr_scratch = corr_scratch;
    Ok(count)
}

/// Execute the stage-synchronous, node-parallel solve sweep: for each stage in
/// turn, claim this rank's `on_my_paths` nodes at that stage from a shared
/// [`ClaimCursor`], solve each exactly once across the worker pool, and
/// scatter every capture into `scratch.arena` in the canonical claim-order-
/// independent order [`canonical_scatter`] defines. Returns the total LP-solve
/// count (the dedup scale invariant a caller may check against the distinct
/// `on_my_paths` node count).
///
/// # Errors
///
/// Propagates the first worker's `SimulationError` from any node solve.
fn run_sweep<S: SolverInterface + Send>(
    plan: &EnumeratedPlan,
    scratch: &mut EnumeratedSimScratch,
    workspaces: &mut [SolverWorkspace<S>],
    num_stages: usize,
    params: &EnumeratedSimParams<'_>,
) -> Result<u64, SimulationError> {
    let node_graph = params.training_ctx.node_graph;
    let mut lp_solves = 0u64;

    for t in (0..num_stages).map(StageIdx) {
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
        let arena = &scratch.arena;
        let m_rep = &scratch.m_rep;
        let stage_units = &scratch.stage_units;
        let parent = &plan.parent;

        let worker_counts: Vec<Result<usize, SimulationError>> = workspaces
            .par_iter_mut()
            .zip(scratch.worker_captures.par_iter_mut())
            .map(|(ws, captures)| {
                enumerated_sim_stage_worker(
                    ws,
                    captures,
                    &cursor,
                    t,
                    params,
                    stage_units,
                    parent,
                    arena,
                    m_rep,
                )
            })
            .collect();

        let counts = worker_counts
            .into_iter()
            .collect::<Result<Vec<usize>, SimulationError>>()?;
        lp_solves += counts.iter().map(|&c| c as u64).sum::<u64>();

        for (w, i) in canonical_scatter(&counts) {
            let Some(visit) = scratch.worker_captures[w][i].take() else {
                debug_assert!(
                    false,
                    "canonical_scatter index must be populated by the claim loop"
                );
                continue;
            };
            let node = visit.node;
            scratch.arena[node] = Some(visit);
        }
    }

    Ok(lp_solves)
}

/// Re-expand the per-node arena into per-path output rows: each owned leaf's
/// canonical node sequence reuses `arena[node].result` directly — exact by
/// the determinacy lemma (a node's solved outcome is path-independent) — and
/// streams the assembled scenario result.
///
/// # Errors
///
/// Returns `SimulationError::InvalidConfiguration` when an owned path's node
/// has no populated arena entry (a marking/sweep defect), never a silent
/// zero-filled default. Propagates channel-send failure from
/// [`dispatch_scenario_result`].
fn re_expand(
    plan: &EnumeratedPlan,
    scratch: &EnumeratedSimScratch,
    scenario_range: Range<u32>,
    num_stages: usize,
    ctx: &StageContext<'_>,
    output: &SimulationOutputSpec<'_>,
) -> Result<(WorkerCosts, WorkerStats), SimulationError> {
    let local_count = scenario_range.len();
    let mut costs: WorkerCosts = Vec::with_capacity(local_count);
    let mut stats: WorkerStats = Vec::with_capacity(local_count);
    let mut seq: Vec<NodePos> = Vec::with_capacity(num_stages);

    for p in scenario_range {
        seq.clear();
        plan.walk_path(p as usize, num_stages, &mut seq);

        let mut total_cost = 0.0_f64;
        let mut leaf_stats = SolverStatsDelta::default();
        let mut stage_results: Vec<SimulationStageResult> = Vec::with_capacity(num_stages);
        for (t, &node) in seq.iter().enumerate() {
            let visit = scratch.arena[node].as_ref().ok_or_else(|| {
                SimulationError::InvalidConfiguration(format!(
                    "enumerated census: node {node} on path {p}, stage {t} has no populated \
                     arena entry at re-expansion (a marking/sweep defect)"
                ))
            })?;
            debug_assert_eq!(
                visit.node_id, visit.result.node_id,
                "a captured visit's own node_id must match its extracted result's node_id"
            );
            let cum_d = ctx.cumulative_discount_factor(StageIdx(t));
            total_cost += cum_d * visit.immediate_cost;
            SolverStatsDelta::accumulate_into(&mut leaf_stats, &visit.stats);
            stage_results.push(visit.result.clone());
        }
        stats.push((p, -1_i32, leaf_stats));
        costs.push(dispatch_scenario_result(
            output,
            p,
            total_cost,
            stage_results,
        )?);
    }

    Ok((costs, stats))
}

/// Execute the enumerated all-paths census simulation on this rank. Every
/// ancestor of an owned leaf is owned and solved locally from this rank's own
/// arena — no cross-rank state exchange occurs at any world size.
///
/// # Errors
///
/// Propagates `SimulationError::LpInfeasible`/`SolverError` from any node
/// solve, and `SimulationError::InvalidConfiguration` if a marking/sweep
/// defect leaves an owned path's node unsolved at re-expansion.
pub(crate) fn run_enumerated_simulation<S, C: Communicator>(
    plan: &EnumeratedPlan,
    inputs: &mut SimulationInputs<'_, S, C>,
    frozen_templates: &[StageTemplate],
    sampler: &ForwardSampler<'_>,
    noise_tables: &ForwardNoiseTables,
) -> Result<(WorkerCosts, WorkerStats), SimulationError>
where
    S: SolverInterface + Send,
{
    let training_ctx = inputs.training_ctx;
    let num_stages = training_ctx.horizon.num_stages();
    let rank = inputs.comm.rank();
    let world_size = inputs.comm.size();
    let total_scenarios = inputs.config.n_scenarios;

    let scenario_range = assign_scenarios(total_scenarios, rank, world_size);
    let n_workers = inputs.workspaces.len().max(1);
    let mut scratch = EnumeratedSimScratch::new(plan, n_workers);

    mark_own_paths(plan, scenario_range.clone(), num_stages, &mut scratch);

    let lookups = SimLookups::build(
        training_ctx.study_dims,
        inputs.ctx.geometry_per_stage,
        inputs.output.hydro_cell_index,
        inputs.output.entity_counts.hydro_ids.len(),
    );
    let load_spec = SimScenarioLoadSpec {
        frozen_templates,
        node_bases: inputs.node_bases,
    };
    let sim_params = EnumeratedSimParams {
        ctx: inputs.ctx,
        fcf: inputs.fcf,
        training_ctx,
        output: &inputs.output,
        load_spec: &load_spec,
        sampler,
        noise_tables,
        lookups: &lookups,
        total_scenarios,
    };

    // `lp_solves` is read only by the debug-assert block below; in a release
    // build (debug-assertions off) it is intentionally unused.
    #[cfg_attr(not(debug_assertions), allow(unused_variables))]
    let lp_solves = run_sweep(
        plan,
        &mut scratch,
        inputs.workspaces,
        num_stages,
        &sim_params,
    )?;

    #[cfg(debug_assertions)]
    {
        let distinct = scratch.on_my_paths.iter().filter(|&&flag| flag).count() as u64;
        debug_assert_eq!(
            lp_solves, distinct,
            "the sweep must solve every on_my_paths node exactly once"
        );
        let populated = scratch.arena.iter().filter(|v| v.is_some()).count() as u64;
        debug_assert_eq!(
            populated, distinct,
            "the arena must hold exactly one populated entry per on_my_paths node"
        );
    }

    // Sanctioned drop, not dead code: out_state/accum are unread after the
    // sweep; the arena's `result`s stay resident through re-expansion.
    for visit in scratch.arena.iter_mut().flatten() {
        visit.out_state = Vec::new();
        visit.accum = AccumSnapshot::default();
    }

    re_expand(
        plan,
        &scratch,
        scenario_range,
        num_stages,
        inputs.ctx,
        &inputs.output,
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use cobre_comm::LocalBackend;
    use cobre_solver::ActiveSolver;

    use crate::{setup::node_graph::Traversal, test_support};

    /// A hand-built small tree (the DECOMP K-fan: one root shared by every
    /// path, `k` branch nodes, `k` leaves) drives the sweep+scatter mechanics
    /// directly (bypassing [`run_enumerated_simulation`]'s frozen-template
    /// resolution, which needs a real per-iteration re-freeze this fixture's
    /// cut-free FCF makes trivial): every distinct `on_my_paths` node is
    /// solved exactly once (claim count == distinct node count) and the arena
    /// is fully populated before re-expansion.
    #[test]
    fn sweep_solves_each_distinct_node_once_and_populates_arena() {
        let k = 3usize;
        let fixture = test_support::k_fan_setup_enumerated(k, 1);
        let setup = &fixture.setup;
        let node_graph = &setup.inputs.node_graph;

        #[allow(clippy::cast_possible_truncation)]
        let traversal = Traversal::resolve(node_graph, true, k as u32);
        let Traversal::Enumerated(plan) = &traversal else {
            unreachable!("resolve(is_enumerated=true, ..) always yields Enumerated");
        };

        let num_stages = setup.num_stages();
        let stage_ctx = setup.stage_ctx();
        let training_ctx = setup.simulation_ctx();

        let comm = LocalBackend;
        let n_workers = 2;
        let mut pool = setup
            .create_workspace_pool(&comm, n_workers, ActiveSolver::new)
            .expect("workspace pool");

        // A fresh, untrained FCF carries no cuts, so the per-pool frozen
        // overlay is exactly the pool's home-stage base template.
        let frozen_templates: Vec<StageTemplate> = (0..node_graph.n_pools)
            .map(|p| stage_ctx.templates[node_graph.pool_stage[p].0].clone())
            .collect();

        let (result_tx, _result_rx) = std::sync::mpsc::sync_channel(64);
        let output = SimulationOutputSpec {
            result_tx: &result_tx,
            block_hours_per_stage: &setup
                .inputs
                .stage_data
                .stage_templates
                .block_hours_per_stage,
            entity_counts: &setup.inputs.stage_data.entity_counts,
            generic_constraint_row_entries: &setup
                .inputs
                .stage_data
                .stage_templates
                .generic_constraint_row_entries,
            hydro_cell_index: &setup.inputs.stage_data.hydro_cell_index,
            pumping_consumption_mw_per_m3s: &setup.inputs.stage_data.pumping_consumption_mw_per_m3s,
            contract_prices_per_stage: &setup.inputs.stage_data.contract_prices_per_stage,
            contract_slots: &setup.inputs.stage_data.contract_slots,
            diversion_upstream: &setup.inputs.stage_data.stage_templates.diversion_upstream,
            hydro_productivities_per_stage: &setup
                .inputs
                .stage_data
                .stage_templates
                .hydro_productivities_per_stage,
            energy_conversion: &setup.energy_conversion,
            hydro_min_storage_hm3: &setup.hydro_min_storage_hm3,
            event_sender: None,
            extended_delivery_anchors: &setup.extended_delivery_anchors,
            transit_seed_arcs: &setup.transit_seed_arcs,
            past_defluences: &setup.past_defluences,
            study_stage_dates: &setup.study_stage_dates,
        };

        let sampler = crate::simulation::state::build_sim_sampler(
            &training_ctx,
            setup.simulation_config().forward_seed,
        )
        .expect("forward sampler");
        #[allow(clippy::cast_possible_truncation)]
        let total_scenarios_u32 = k as u32;
        let mut noise_tables = ForwardNoiseTables::default();
        sampler
            .rebuild_noise_tables(
                SIMULATION_ITERATION,
                total_scenarios_u32,
                stage_ctx.noise_group_ids,
                &mut noise_tables,
            )
            .expect("test fixture never exceeds the Sobol dimension cap");

        let lookups = SimLookups::build(
            training_ctx.study_dims,
            stage_ctx.geometry_per_stage,
            output.hydro_cell_index,
            output.entity_counts.hydro_ids.len(),
        );
        let load_spec = SimScenarioLoadSpec {
            frozen_templates: &frozen_templates,
            node_bases: &[],
        };
        let params = EnumeratedSimParams {
            ctx: &stage_ctx,
            fcf: &setup.fcf,
            training_ctx: &training_ctx,
            output: &output,
            load_spec: &load_spec,
            sampler: &sampler,
            noise_tables: &noise_tables,
            lookups: &lookups,
            total_scenarios: total_scenarios_u32,
        };

        let mut scratch = EnumeratedSimScratch::new(plan, n_workers);
        #[allow(clippy::cast_possible_truncation)]
        let scenario_range = 0..k as u32;
        mark_own_paths(plan, scenario_range.clone(), num_stages, &mut scratch);

        let distinct_before_sweep = scratch.on_my_paths.iter().filter(|&&f| f).count();
        // A root shared by every path, k branches, k leaves: k+2 pool-distinct
        // nodes among 2k+1 total, and the root is the one node every path shares.
        assert_eq!(
            distinct_before_sweep,
            2 * k + 1,
            "own-set marking must mark the root, every branch, and every leaf"
        );

        let lp_solves = run_sweep(
            plan,
            &mut scratch,
            &mut pool.workspaces,
            num_stages,
            &params,
        )
        .expect("sweep must solve every claimed node");

        assert_eq!(
            lp_solves as usize, distinct_before_sweep,
            "claim count must equal the distinct on_my_paths node count (dedup, not k*num_stages)"
        );
        let populated = scratch.arena.iter().filter(|v| v.is_some()).count();
        assert_eq!(
            populated, distinct_before_sweep,
            "the arena must be fully populated before re-expansion"
        );
    }
}
