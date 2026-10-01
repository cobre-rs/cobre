//! Backward pass state management and entry point.
//!
//! [`BackwardPassState`] owns pre-allocated scratch buffers reused each iteration.
//! [`BackwardPassInputs`] bundles per-call borrowed inputs (no allocation on hot path).

use std::ops::Range;
use std::sync::mpsc::Sender;
use std::time::Instant;

use cobre_comm::{Communicator, ReduceOp};
use cobre_core::{TrainingEvent, WorkerPhaseTimings, WorkerTimingPhase};
use cobre_io::config::BackwardScheduler;
use cobre_solver::ActiveProfile;
use cobre_solver::{RowBatch, SolverInterface, SolverStatistics, StageTemplate};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::risk_measure::BackwardOutcome;
#[cfg(test)]
use crate::setup::node_graph::NodeId;
use crate::setup::node_graph::{EnumeratedPlan, NodeGraph, NodePos, StageIdx, Traversal};
use crate::{
    backward::{
        BackwardResult, ReplicatedScratch, StageOpeningSolver, StageWorkerOpeningDelta, StagedCut,
        SuccessorEntry, SuccessorOutcomes, SuccessorSpec, by_node_block_count, by_node_finish,
        by_scenario_finish, hardest_first_block_order, identity_block_order, merge_block_pivots,
        process_by_scenario_backward, process_stage_backward_by_node, reify_successor_outcomes,
        resolve_block_size, run_backward_node_replicated,
    },
    config::CutManagementConfig,
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    cut_sync::CutSyncBuffers,
    error::SddpError,
    forward::EnumeratedForwardScratch,
    horizon_mode::HorizonMode,
    lp::indexer::StateSpace,
    rank_reconcile::reconcile_result,
    risk_measure::RiskMeasure,
    solver_phase::Phase,
    solver_stats::{
        SolverStatsDelta, StageWorkerStatsBuffer, WORKER_STATS_ENTRY_STRIDE,
        pack_worker_opening_stats, unpack_worker_opening_stats,
    },
    state_exchange::ExchangeBuffers,
    training_session::{rank_distribution::RankDistribution, runtime::RuntimeHandles},
    trajectory::TrajectoryRecord,
    visited_states::VisitedStatesArchive,
    workspace::{BasisStore, BasisStoreSliceMut, ByNodeScratch, SolverWorkspace, WorkspacePool},
};

/// Per-iteration argument bundle for [`BackwardPassState::run`].
///
/// Groups all borrowed inputs that vary between calls: exchange buffers,
/// trajectory records, risk measures, cut-sync state, and the event sender.
/// Owned scratch buffers live on [`BackwardPassState`] and are not repeated
/// here.
pub struct BackwardPassInputs<'a, S: SolverInterface + Send, C: Communicator> {
    /// Solver workspaces (one per rayon worker thread).
    pub workspaces: &'a mut [SolverWorkspace<S>],
    /// Basis warm-start store (one slot per `(scenario, stage)` pair).
    pub basis_store: &'a mut BasisStore,
    /// Stage-level LP context (templates, row counts, noise scales).
    pub ctx: &'a StageContext<'a>,
    /// Frozen LP templates including pre-appended prior-iteration cuts.
    pub frozen: &'a [StageTemplate],
    /// Future-cost function — receives new cuts after each stage.
    pub fcf: &'a mut FutureCostFunction,
    /// Per-stage delta cut row batches (reused scratch, resized per stage).
    pub cut_batches: &'a mut [RowBatch],
    /// Study-level training context (horizon, indexer, stochastic model).
    pub training_ctx: &'a TrainingContext<'a>,
    /// MPI communicator.
    pub comm: &'a C,

    /// Exchange buffers for gathering trial-point states via `allgatherv`.
    pub exchange: &'a mut ExchangeBuffers,

    /// Forward-pass trajectory records used to populate `exchange` per stage.
    ///
    /// Length is `max_local_fwd * num_stages` — rank-uniform, NOT this rank's
    /// `local_work`; a rank drawing zero trial points still passes full-length
    /// padding, which `real_total_scenarios` discards after the gather. Re-slicing
    /// to `local_work * num_stages` (as `ForwardPassInputs` correctly does) empties
    /// it on exactly those ranks and desynchronises the per-stage `allgatherv`.
    pub records: &'a [TrajectoryRecord],

    /// Pre-allocated cut synchronisation buffers for per-stage `allgatherv`.
    pub cut_sync_bufs: &'a mut CutSyncBuffers,

    /// Optional visited-states archive for dominated cut selection.
    pub visited_archive: Option<&'a mut VisitedStatesArchive>,

    /// Optional event channel for emitting [`TrainingEvent::WorkerTiming`] events.
    pub event_sender: Option<&'a Sender<TrainingEvent>>,

    /// Per-stage risk measures (length = `num_stages`).
    pub risk_measures: &'a [RiskMeasure],

    /// Minimum dual multiplier for a cut to count as binding.
    pub cut_activity_tolerance: f64,

    /// Current training iteration index (1-based), used for cut metadata.
    pub iteration: u64,

    /// Number of trial points assigned to this rank for the backward pass.
    pub local_work: usize,

    /// Global offset for this rank's trial points (`rank * fwd_per_rank`).
    pub fwd_offset: usize,

    /// The resolved forward-traversal axis (mirrors
    /// `ForwardPassState::traversal()`): [`BackwardPassState::run`] dispatches
    /// on this directly, `Traversal::Enumerated` routing to the node-native
    /// fork and `Traversal::Sampled` to the by-scenario/by-node level driver.
    pub traversal: &'a Traversal,

    /// The forward pass's persisted per-node outgoing-state arena
    /// (`ForwardPassState::enumerated_state`), read by the enumerated
    /// backward fork via `EnumeratedPlan`'s node-native level sweep. Unused
    /// under `Traversal::Sampled`.
    pub enumerated_state: &'a EnumeratedForwardScratch,
}

impl<'a, S: SolverInterface + Send, C: Communicator> BackwardPassInputs<'a, S, C> {
    /// Construct inputs from the fields of a `TrainingSession`, minus `bwd_state`.
    // Rationale: the arguments are disjoint mutable borrows into `TrainingSession` fields that
    // Rust NLL cannot split from a single `&mut self`; bundling them into a context struct would
    // just move the arity to the struct literal without resolving the borrow-splitting requirement.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_session_fields(
        fwd_pool: &'a mut WorkspacePool<S>,
        basis_store: &'a mut BasisStore,
        ctx: &'a StageContext<'a>,
        frozen_templates: &'a [StageTemplate],
        cut_batches: &'a mut [RowBatch],
        records: &'a [TrajectoryRecord],
        fcf: &'a mut FutureCostFunction,
        exchange: &'a mut ExchangeBuffers,
        cut_sync_bufs: &'a mut CutSyncBuffers,
        visited_archive: &'a mut Option<VisitedStatesArchive>,
        training_ctx: &'a TrainingContext<'a>,
        comm: &'a C,
        cut_mgmt: &'a CutManagementConfig,
        ranks: &RankDistribution,
        runtime: &'a RuntimeHandles,
        iteration: u64,
        traversal: &'a Traversal,
        enumerated_state: &'a EnumeratedForwardScratch,
    ) -> Self {
        Self {
            workspaces: &mut fwd_pool.workspaces,
            basis_store,
            ctx,
            frozen: frozen_templates,
            fcf,
            cut_batches,
            training_ctx,
            comm,
            exchange,
            records,
            cut_sync_bufs,
            visited_archive: visited_archive.as_mut(),
            event_sender: runtime.event_sender(),
            risk_measures: &cut_mgmt.risk_measures,
            cut_activity_tolerance: cut_mgmt.cut_activity_tolerance,
            iteration,
            local_work: ranks.my_actual_fwd,
            fwd_offset: ranks.my_fwd_offset,
            traversal,
            enumerated_state,
        }
    }
}

/// Owned scratch buffers for the backward pass, allocated once and reused.
///
/// `BackwardPassState` is constructed once by `TrainingSession::new` and stored
/// as a field on `TrainingSession`. The buffers are pre-sized from the study
/// dimensions and reused across every iteration via `clear()` / `resize()` /
/// `fill()`. No allocation occurs on the hot path.
///
/// Per-iteration inputs (exchange buffers, records, risk measures, etc.) are
/// passed via [`BackwardPassInputs`] at each `run()` call.
pub struct BackwardPassState {
    /// Uniform opening probabilities.
    pub(crate) probabilities_buf: Vec<f64>,

    /// Per-child successor metadata for the current node, canonical order. Reused
    /// across nodes (cleared + refilled), so the reified successor set costs no
    /// per-node allocation.
    pub(crate) successor_meta_buf: Vec<SuccessorEntry>,

    /// Every child's active-cut slot indices, concatenated in child order; each
    /// [`SuccessorEntry::active_slots`] range indexes into this.
    pub(crate) successor_active_slots_buf: Vec<usize>,

    /// Per-slot binding increment aggregation, source of the `allreduce(Sum)` that
    /// synchronises cut binding metadata across MPI ranks.
    pub(crate) metadata_sync_buf: Vec<u64>,

    /// Receive buffer for the `allreduce(Sum)` aggregating per-slot binding
    /// increments across MPI ranks.
    pub(crate) global_increments_buf: Vec<u64>,

    /// Packs real (non-padded) gathered state vectors when archiving visited
    /// states for dominated cut selection.
    pub(crate) real_states_buf: Vec<f64>,

    /// Per-(worker, opening) gather buffer for backward-pass solver stats.
    ///
    /// Shape: `n_workers_local × max_openings`. Reset at the start of each stage.
    pub(crate) stage_worker_stats_buf: StageWorkerStatsBuffer,

    /// MPI send buffer for the per-`(worker, opening)` stats `allgatherv`.
    ///
    /// Length: `n_workers_local * bwd_max_openings * WORKER_STATS_ENTRY_STRIDE`.
    pub(crate) bwd_stats_send_buf: Vec<f64>,

    /// MPI receive buffer for the per-`(rank, worker, opening)` stats `allgatherv`.
    ///
    /// Length: `n_ranks * n_workers_local * bwd_max_openings * WORKER_STATS_ENTRY_STRIDE`.
    pub(crate) bwd_stats_recv_buf: Vec<f64>,

    /// Per-rank element counts for the `allgatherv` of backward stats.
    pub(crate) bwd_stats_counts: Vec<usize>,

    /// Displacement array for the `allgatherv` of backward stats.
    pub(crate) bwd_stats_displs: Vec<usize>,

    /// Unpack destination buffer for the per-`(rank, worker, opening)` stats.
    ///
    /// Length: `n_ranks * n_workers_local * bwd_max_openings` `SolverStatsDelta` entries.
    pub(crate) bwd_stats_unpack_buf: Vec<SolverStatsDelta>,

    // ── Per-iteration scratch (reused across stages within one `run()` call) ──
    /// Staging buffer for cuts produced by one stage's parallel trial-point
    /// loop, each paired with the index `w` of the worker that produced it.
    ///
    /// The worker index resolves the cut's `coefficients_range` against
    /// `workspaces[w].backward_accum.agg_arena` at merge time. Cleared at the
    /// start of each stage and grown monotonically.
    pub(crate) staged_cuts_buf: Vec<(usize, StagedCut)>,

    /// Per-worker solver statistics snapshot taken **before** the stage's parallel
    /// region.
    pub(crate) worker_stats_before: Vec<SolverStatistics>,

    /// Per-worker solver statistics snapshot taken **after** the stage's parallel
    /// region.
    pub(crate) worker_stats_after: Vec<SolverStatistics>,

    /// Per-worker solver-statistics delta for this stage (after − before).
    pub(crate) worker_deltas: Vec<SolverStatsDelta>,

    /// Per-worker total work time (solve + load + set-bounds + basis-set) for
    /// load-imbalance decomposition.
    pub(crate) worker_totals: Vec<f64>,

    /// Cross-rank error-reconciliation scratch, reused each stage by the
    /// pre-`sync_level_records` reconcile so that reconciliation never allocates.
    pub(crate) reconcile_scratch: [i32; 1],

    /// Resolved backward-phase solver profile applied at [`Self::run`] entry.
    /// Defaults to `Phase::Backward.profile()`; override with
    /// [`Self::set_profile`] before the first `run()` call.
    profile: ActiveProfile,

    /// Backward-pass work-unit scheduler, carrying the opening-block size
    /// when the `ByNode` method is selected. Defaults to
    /// [`BackwardScheduler::ByScenario`] (byte-neutral with the pre-scheduler
    /// path); override with [`Self::set_scheduler`] before the first `run()`
    /// call.
    scheduler: BackwardScheduler,

    /// Whether the by-node scheduler claims hardest-`(stage,
    /// block)`-first using [`Self::by_node_scratch`]'s
    /// `block_pivots_prev` row, or the canonical ascending block order.
    /// Defaults to `true`; override with [`Self::set_hardest_first_claim_order`]
    /// before the first `run()` call.
    hardest_first_claim_order: bool,

    /// Maximum local forward-pass count across the run; with
    /// `bwd_max_openings` and `n_state`, sizes [`Self::by_node_scratch`] when
    /// [`Self::set_scheduler`] resolves `ByNode`.
    max_local_fwd: usize,

    /// Maximum opening count across all stages; see [`Self::max_local_fwd`].
    bwd_max_openings: usize,

    /// State dimension; see [`Self::max_local_fwd`].
    n_state: usize,

    /// Number of stages in the study; with `bwd_max_openings`, sizes
    /// [`Self::by_node_scratch`]'s per-`(stage, block-index)` pivot accumulator
    /// when [`Self::set_scheduler`] resolves `ByNode`.
    num_stages: usize,

    /// Pre-allocated by-node scheduler scratch (per-`(m, ω)` outcome
    /// arena + aggregation buffers). Empty until [`Self::set_scheduler`] sizes
    /// it for `BackwardScheduler::ByNode` (sddp.md "By-node
    /// scheduler is warm-start-only").
    by_node_scratch: ByNodeScratch,

    /// Per-level node-compute results, reused across levels (mem-swapped out in
    /// `run_one_backward_level` so the level loop allocates no per-level buffer).
    level_nodes_scratch: Vec<NodeCompute>,

    /// Per-level trial-point-distributed pool list for the batched cut exchange,
    /// reused across levels alongside `level_nodes_scratch`.
    level_pools_scratch: Vec<usize>,

    /// Per-level routing of this rank's own trial points to the level's
    /// cut-generating nodes, CSR values: `routed_trials_scratch[
    /// routed_offsets_scratch[i]..routed_offsets_scratch[i + 1]]` are the
    /// ascending trial-point indices whose stage visit resolves to `level[i]`'s
    /// pool. Reused across levels alongside `level_nodes_scratch`.
    routed_trials_scratch: Vec<usize>,

    /// CSR offsets into `routed_trials_scratch`: one entry per level node plus a
    /// trailing bound (`offsets[i]..offsets[i + 1]` bounds level node `i`).
    routed_offsets_scratch: Vec<usize>,

    /// Per-level list of every cut-generating node's successor pool regions (one
    /// entry per child), reused across levels. Each [`NodeCompute::pool_regions`]
    /// range indexes into it so the level's metadata sync targets each child's OWN
    /// pool at its own binding-metadata slot region.
    level_pool_regions_scratch: Vec<PoolRegion>,

    /// This rank's per-node base slot offset for the current level: the count of
    /// visits to `level[i]` from strictly-lower ranks, so a cut's node-relative
    /// slot stays globally unique across ranks at the pool's own `visit_bound`
    /// stride. `fwd_offset` on a single-node level (every trial visits it), `0`
    /// on a single rank; the allgathered per-node prefix otherwise. Reused across
    /// levels alongside the buffers below (never a hot-path `Vec::new`).
    node_visit_offsets: Vec<usize>,

    /// This rank's per-node local visit counts, the send buffer for the per-node
    /// count `allgatherv` (`u64`; `usize` is not an `MpiDatatype`). Reused.
    bwd_visit_send: Vec<u64>,

    /// Gathered per-`(rank, node)` visit counts for the current level, rank-major.
    /// Reused.
    bwd_visit_recv: Vec<u64>,

    /// Per-rank `counts`/`displs` for the per-node count `allgatherv` (uniform
    /// `n_nodes` per rank). Reused.
    bwd_visit_counts: Vec<usize>,
    bwd_visit_displs: Vec<usize>,

    /// Scratch for the enumerated backward's replicated per-node solve
    /// (`run_backward_node_replicated`). Empty (zero footprint) under
    /// `Traversal::Sampled`, which never touches it.
    replicated_scratch: ReplicatedScratch,

    /// Per-successor-stage accumulated solver-stats delta for the enumerated
    /// backward, indexed by stage. Reset via `reset_in_place` at the start of
    /// each [`Self::run_enumerated_backward`] call, mirroring the enumerated
    /// forward's `worker_stage_stats` reuse; never touched under
    /// `Traversal::Sampled`.
    enumerated_stage_stats: Vec<SolverStatsDelta>,
}

/// One successor child's cut-pool binding-metadata region: which pool, the base
/// offset of its slots in the per-worker `metadata_sync_contribution` buffer, and
/// its populated slot count.
#[derive(Clone, Copy)]
struct PoolRegion {
    pool_id: usize,
    region_offset: usize,
    populated_count: usize,
}

impl BackwardPassState {
    /// Allocate all scratch buffers sized for the given study dimensions.
    ///
    /// # Parameters
    ///
    /// - `n_workers_local`: number of rayon worker threads on this rank.
    /// - `n_ranks`: total MPI rank count.
    /// - `bwd_max_openings`: maximum opening count across all stages.
    /// - `real_states_capacity`: capacity hint for `real_states_buf`
    ///   (`real_total_scenarios * n_state`).
    /// - `max_local_fwd`: maximum local forward-pass count across the run.
    /// - `state`: state-vector layout owner (`n_state` dimension).
    /// - `horizon`: horizon owner (`num_stages` dimension).
    ///
    /// `max_local_fwd`, `state.n_state`, and `horizon.num_stages()`, together with
    /// `bwd_max_openings`, size `Self::by_node_scratch` once `set_scheduler`
    /// calls `Self::resize_by_node_scratch`; `by_node_scratch` starts empty.
    #[must_use]
    pub fn new(
        n_workers_local: usize,
        n_ranks: usize,
        bwd_max_openings: usize,
        real_states_capacity: usize,
        max_local_fwd: usize,
        state: &StateSpace,
        horizon: &HorizonMode,
    ) -> Self {
        let n_state = state.n_state;
        let num_stages = horizon.num_stages();
        let send_stride = n_workers_local * bwd_max_openings * WORKER_STATS_ENTRY_STRIDE;
        Self {
            probabilities_buf: Vec::new(),
            successor_meta_buf: Vec::new(),
            successor_active_slots_buf: Vec::new(),
            metadata_sync_buf: Vec::new(),
            global_increments_buf: Vec::new(),
            real_states_buf: Vec::with_capacity(real_states_capacity),
            stage_worker_stats_buf: StageWorkerStatsBuffer::new(n_workers_local, bwd_max_openings),
            bwd_stats_send_buf: vec![0.0; send_stride],
            bwd_stats_recv_buf: vec![0.0; n_ranks * send_stride],
            bwd_stats_counts: vec![send_stride; n_ranks],
            bwd_stats_displs: (0..n_ranks).map(|r| r * send_stride).collect(),
            bwd_stats_unpack_buf: vec![
                SolverStatsDelta::default();
                n_ranks * n_workers_local * bwd_max_openings
            ],
            staged_cuts_buf: Vec::new(),
            worker_stats_before: Vec::with_capacity(n_workers_local),
            worker_stats_after: Vec::with_capacity(n_workers_local),
            worker_deltas: Vec::with_capacity(n_workers_local),
            worker_totals: Vec::with_capacity(n_workers_local),
            reconcile_scratch: [0_i32; 1],
            profile: Phase::Backward.profile(),
            scheduler: BackwardScheduler::default(),
            hardest_first_claim_order: true,
            max_local_fwd,
            bwd_max_openings,
            n_state,
            num_stages,
            by_node_scratch: ByNodeScratch::default(),
            level_nodes_scratch: Vec::new(),
            level_pools_scratch: Vec::new(),
            routed_trials_scratch: Vec::new(),
            routed_offsets_scratch: Vec::new(),
            node_visit_offsets: Vec::new(),
            bwd_visit_send: Vec::new(),
            bwd_visit_recv: Vec::new(),
            bwd_visit_counts: Vec::new(),
            bwd_visit_displs: Vec::new(),
            level_pool_regions_scratch: Vec::new(),
            replicated_scratch: ReplicatedScratch::default(),
            enumerated_stage_stats: (0..num_stages)
                .map(|_| SolverStatsDelta::default())
                .collect(),
        }
    }

    /// Overrides the backward-phase solver profile applied at [`Self::run`]
    /// entry (default: `Phase::Backward.profile()`). Call before `run()`.
    pub fn set_profile(&mut self, profile: ActiveProfile) {
        self.profile = profile;
    }

    /// Overrides the backward-pass scheduler applied at [`Self::run`] entry
    /// (default: [`BackwardScheduler::ByScenario`]). Call before `run()`.
    pub fn set_scheduler(&mut self, scheduler: BackwardScheduler) {
        self.scheduler = scheduler;
        self.resize_by_node_scratch();
    }

    /// Single owner of `Self::by_node_scratch`'s size decision: sized to the
    /// full `max_local_fwd * bwd_max_openings` shape whenever `Self::scheduler`
    /// is literally `ByNode`, empty otherwise. Never on the hot path.
    fn resize_by_node_scratch(&mut self) {
        let may_dispatch_by_node = matches!(self.scheduler, BackwardScheduler::ByNode { .. });
        self.by_node_scratch = if may_dispatch_by_node {
            ByNodeScratch::sized(
                self.max_local_fwd,
                self.bwd_max_openings,
                self.n_state,
                self.num_stages,
            )
        } else {
            ByNodeScratch::default()
        };
    }

    /// Overrides whether the by-node scheduler claims work
    /// hardest-`(stage, block)`-first using the previous iteration's
    /// mean pivots (default `true`). `false` forces the canonical ascending
    /// block order — the byte-neutrality gate's off leg. Call before `run()`.
    pub fn set_hardest_first_claim_order(&mut self, enabled: bool) {
        self.hardest_first_claim_order = enabled;
    }

    /// Execute the backward pass for one training iteration on this rank.
    ///
    /// Dispatches on `inputs.traversal`, mirroring the forward fork
    /// (`ForwardPassState::run`'s `Traversal::Enumerated` match arm):
    /// `Traversal::Enumerated` routes to the node-native
    /// [`Self::run_enumerated_backward`], `Traversal::Sampled` to
    /// [`Self::run_sampled_backward`] (the pre-existing per-level driver,
    /// byte-frozen).
    ///
    /// # Errors
    ///
    /// Returns `Err(SddpError::Infeasible { .. })` when a stage LP has no
    /// feasible solution during the backward sweep. Returns
    /// `Err(SddpError::Solver(_))` for all other terminal LP solver failures.
    pub fn run<S, C: Communicator>(
        &mut self,
        inputs: &mut BackwardPassInputs<'_, S, C>,
    ) -> Result<BackwardResult, SddpError>
    where
        S: SolverInterface<Profile = ActiveProfile> + Send,
    {
        match inputs.traversal {
            Traversal::Enumerated(plan) => self.run_enumerated_backward(inputs, plan),
            Traversal::Sampled { .. } => self.run_sampled_backward(inputs),
        }
    }

    /// The sampled-traversal backward driver: the per-level reverse-topological
    /// sweep (`run_one_backward_level`/`compute_one_backward_node`), byte-frozen
    /// — [`Self::run`]'s `Traversal::Sampled` arm.
    ///
    /// # Errors
    ///
    /// Returns `Err(SddpError::Infeasible { .. })` when a stage LP has no
    /// feasible solution during the backward sweep. Returns
    /// `Err(SddpError::Solver(_))` for all other terminal LP solver failures.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if any of the following debug preconditions are violated:
    ///
    /// - `inputs.ctx.templates.len() != num_stages`
    /// - `inputs.risk_measures.len() != num_stages`
    /// - `inputs.frozen.len() != n_pools`
    fn run_sampled_backward<S, C: Communicator>(
        &mut self,
        inputs: &mut BackwardPassInputs<'_, S, C>,
    ) -> Result<BackwardResult, SddpError>
    where
        S: SolverInterface<Profile = ActiveProfile> + Send,
    {
        let training_ctx = inputs.training_ctx;
        let num_stages = training_ctx.horizon.num_stages();

        debug_assert_eq!(inputs.ctx.templates.len(), num_stages);
        debug_assert_eq!(inputs.risk_measures.len(), num_stages);
        debug_assert_eq!(
            inputs.frozen.len(),
            training_ctx.node_graph.n_pools,
            "frozen.len() must equal n_pools"
        );

        let start = Instant::now();
        let solves_before: u64 = total_solve_count(inputs.workspaces);

        // `set_profile` is delta-tracked: it issues solver-option calls only for
        // fields that differ from the solver's current state.
        let backward_profile = self.profile;
        for ws in inputs.workspaces.iter_mut() {
            ws.solver.set_profile(&backward_profile);
            debug_assert!(
                ws.solver.current_profile() == &backward_profile,
                "solver profile must equal the profile passed to set_profile"
            );
            ws.worker_timing_buf = WorkerPhaseTimings::default();
        }
        // The opening-block scheduler's per-(generating node, block-index) pivot
        // accumulator is keyed by node position (CA5), so its row count follows the
        // node axis. `set_scheduler` pre-sizes a stage-count floor before the graph is
        // known; grow it to the node count once here (a branching graph has more nodes
        // than stages). Chain: `num_nodes == num_stages`, so this never grows and stays
        // byte-identical; `ByScenario` keeps `n_blocks_max == 0`, a no-op.
        if self.by_node_scratch.n_blocks_max > 0 {
            let needed = training_ctx.node_graph.nodes.len() * self.by_node_scratch.n_blocks_max;
            if self.by_node_scratch.block_pivots.len() < needed {
                self.by_node_scratch.block_pivots.resize(needed, (0, 0));
                self.by_node_scratch
                    .block_pivots_prev
                    .resize(needed, (0, 0));
            }
        }
        // The pivot accumulator is iteration-local: swapped here (once per `run()`
        // call), never per stage. The hardest-first order reads the swapped-in
        // `block_pivots_prev` (last iteration's fully-merged means); reading
        // `block_pivots` during the sweep instead is the wrong-but-compiling
        // alternative — it is reset-then-partially-filled, yielding zeros or a
        // half-filled row.
        std::mem::swap(
            &mut self.by_node_scratch.block_pivots,
            &mut self.by_node_scratch.block_pivots_prev,
        );
        self.by_node_scratch.block_pivots.fill((0, 0));

        #[allow(clippy::cast_precision_loss)]
        let params = StageDerivedParams {
            my_rank: inputs.comm.rank(),
            n_workers_local: inputs.workspaces.len(),
            n_ranks: inputs.comm.size(),
            bwd_max_openings: self.bwd_stats_send_buf.len()
                / inputs.workspaces.len().max(1)
                / WORKER_STATS_ENTRY_STRIDE,
            n_workers: inputs.workspaces.len() as f64,
        };

        // Verify all ranks agree on n_workers_local. A mismatch silently
        // corrupts the per-worker stats allgatherv buffer; surface it as a
        // typed error before the exchange.
        let local_workers = u64::try_from(inputs.workspaces.len())
            .map_err(|_| SddpError::Validation("workspaces.len() exceeds u64::MAX".into()))?;
        let send = [local_workers];
        let mut min_recv = [0_u64; 1];
        let mut max_recv = [0_u64; 1];
        inputs
            .comm
            .allreduce(&send, &mut min_recv, ReduceOp::Min)
            .map_err(SddpError::Communication)?;
        inputs
            .comm
            .allreduce(&send, &mut max_recv, ReduceOp::Max)
            .map_err(SddpError::Communication)?;
        if min_recv[0] != max_recv[0] {
            return Err(SddpError::Validation(format!(
                "non-uniform n_workers_local across MPI ranks: \
                 local={local_workers}, min={}, max={}; all ranks must \
                 run with the same --threads value",
                min_recv[0], max_recv[0],
            )));
        }

        let mut cuts_generated: usize = 0;
        let mut stage_stats: Vec<(usize, Vec<StageWorkerOpeningDelta>)> =
            Vec::with_capacity(num_stages.saturating_sub(1));
        let mut state_exchange_ms: u64 = 0;
        let mut cut_batch_build_ms: u64 = 0;
        let mut setup_ms: u64 = 0;
        let mut imbalance_ms: u64 = 0;
        let mut scheduling_ms: u64 = 0;
        let mut cut_sync_ms: u64 = 0;

        // Reverse-topological cut-sharing sweep: descending-stage levels, nodes
        // within a level processed independently (nested per-node risk makes
        // siblings barrier-free). Absent `nodes[]` every level is one node
        // (== stage), reducing to the reversed stage loop byte-for-byte.
        let levels = training_ctx.node_graph.backward_cut_levels();
        for level in &levels {
            let out = run_one_backward_level(self, inputs, level, &params)?;
            cuts_generated += out.cuts_generated;
            state_exchange_ms += out.state_exchange_ms;
            cut_batch_build_ms += out.cut_batch_build_ms;
            setup_ms += out.setup_ms;
            imbalance_ms += out.imbalance_ms;
            scheduling_ms += out.scheduling_ms;
            cut_sync_ms += out.cut_sync_ms;
            stage_stats.extend(out.stage_entries);
        }

        if let Some(sender) = inputs.event_sender {
            for ws in inputs.workspaces.iter() {
                let _ = sender.send(TrainingEvent::WorkerTiming {
                    rank: ws.rank,
                    worker_id: ws.worker_id,
                    iteration: inputs.iteration,
                    phase: WorkerTimingPhase::Backward,
                    timings: ws.worker_timing_buf,
                });
            }
        }

        let elapsed_ms = ms_elapsed(start);
        let solves_after: u64 = total_solve_count(inputs.workspaces);

        Ok(BackwardResult {
            cuts_generated,
            elapsed_ms,
            lp_solves: solves_after - solves_before,
            stage_stats,
            state_exchange_time_ms: state_exchange_ms,
            cut_batch_build_time_ms: cut_batch_build_ms,
            setup_time_ms: setup_ms,
            load_imbalance_ms: imbalance_ms,
            scheduling_overhead_ms: scheduling_ms,
            cut_sync_time_ms: cut_sync_ms,
        })
    }

    /// The enumerated-traversal backward driver: walks `backward_cut_levels`
    /// (already node-native, non-leaf reverse-topological) and, for each
    /// cut-generating node, reads its own persisted outgoing state
    /// (`inputs.enumerated_state`, populated by the enumerated forward — the
    /// value every trajectory sharing this node saw, by construction), builds
    /// its successor outcome set once (`assemble_successor_outcome_weights` +
    /// `SuccessorOutcomes`, unchanged from the sampled path), and solves it via
    /// the replicated per-node solve ([`run_backward_node_replicated`]) —
    /// exactly ONE cut per non-leaf node, never one per trajectory. Neither
    /// the per-level state exchange nor the batched cut exchange is ever
    /// invoked: every rank already holds every cut-generating node's state
    /// locally at world = 1 (the forward replicates trunk nodes on every
    /// rank), and the replicated solve appends the bit-identical cut on every
    /// rank directly.
    ///
    /// # Errors
    ///
    /// Returns `Err(SddpError::Validation(_))` when a cut-generating node's
    /// successor does not record that node as its `EnumeratedPlan` parent (a
    /// malformed plan — the node-native cut cannot resolve which persisted
    /// state is its own). Returns `Err(SddpError::Infeasible { .. })`/
    /// `Err(SddpError::Solver(_))` from a successor LP solve.
    // RATIONALE: the per-node level sweep — the parent-consistency validation,
    // the successor-outcome-set assembly (mirroring compute_one_backward_node's
    // sampled-path build, which the sampled path must stay byte-frozen and
    // therefore cannot share), and the replicated solve dispatch — is one
    // reproducibility-critical sequence; splitting it would fragment the
    // node-native level loop for no clarity gain.
    #[allow(clippy::too_many_lines)]
    fn run_enumerated_backward<S, C: Communicator>(
        &mut self,
        inputs: &mut BackwardPassInputs<'_, S, C>,
        plan: &EnumeratedPlan,
    ) -> Result<BackwardResult, SddpError>
    where
        S: SolverInterface<Profile = ActiveProfile> + Send,
    {
        let start = Instant::now();
        let solves_before: u64 = total_solve_count(inputs.workspaces);

        let backward_profile = self.profile;
        for ws in inputs.workspaces.iter_mut() {
            ws.solver.set_profile(&backward_profile);
        }
        let ws0 = inputs.workspaces.first_mut().ok_or_else(|| {
            SddpError::Validation("enumerated backward: no solver workspace available".into())
        })?;
        ws0.worker_timing_buf = WorkerPhaseTimings::default();

        let training_ctx = inputs.training_ctx;
        let node_graph = training_ctx.node_graph;
        let my_rank = inputs.comm.rank();

        #[cfg(debug_assertions)]
        debug_assert_node_predecessor_states_distinct(node_graph, inputs.enumerated_state);

        for delta in &mut self.enumerated_stage_stats {
            delta.reset_in_place();
        }

        let levels = node_graph.backward_cut_levels();
        let mut cuts_generated: usize = 0;
        for level in &levels {
            for &node_pos in level {
                let node_id = node_graph.node_ids[node_pos];
                let pool_id = node_graph.nodes[node_pos].pool_id;
                let node_stage = node_graph.nodes[node_pos].stage;
                let successor_stage = node_stage.next();

                let x_hat = inputs.enumerated_state.out_state(node_pos);

                // A node on the enumerated path has ONE distinct incoming state,
                // so archive that single vector (never `total_fwd` copies as the
                // sampled level driver does).
                if let Some(archive) = inputs.visited_archive.as_mut() {
                    archive.archive_one_state(node_pos, x_hat);
                }

                assemble_successor_outcome_weights(
                    node_graph,
                    node_pos,
                    &mut self.probabilities_buf,
                );
                let n_openings = self.probabilities_buf.len();

                let template_num_rows = inputs.ctx.template(successor_stage).num_rows;
                for succ_edge in &node_graph.successors[node_pos] {
                    let child_node = succ_edge.child;
                    let parent = plan.parent[child_node];
                    if parent != Some(node_pos) {
                        return Err(SddpError::Validation(format!(
                            "enumerated backward: node {}'s successor {} does not record \
                             node {} as its EnumeratedPlan parent (found {parent:?}); the \
                             node-native cut cannot resolve which persisted state is its own",
                            node_id, node_graph.node_ids[child_node], node_id,
                        )));
                    }
                }
                reify_successor_outcomes(
                    &mut self.successor_meta_buf,
                    &mut self.successor_active_slots_buf,
                    inputs.ctx,
                    training_ctx,
                    inputs.fcf,
                    inputs.cut_batches,
                    inputs.frozen,
                    node_pos,
                    inputs.iteration,
                );

                let outcomes = SuccessorOutcomes::new(
                    &self.successor_meta_buf,
                    &self.successor_active_slots_buf,
                    inputs.frozen,
                    &*inputs.cut_batches,
                    &inputs.fcf.pools,
                    template_num_rows,
                    inputs.cut_activity_tolerance,
                );
                debug_assert_eq!(
                    outcomes.total_outcomes(),
                    n_openings,
                    "the reified successor outcome set must have as many outcomes as \
                     flattened weights"
                );

                let succ_spec = SuccessorSpec {
                    t: node_stage,
                    successor: successor_stage,
                    my_rank,
                    probabilities: &self.probabilities_buf,
                    cut_state: &training_ctx.cut_state_layouts[pool_id],
                };

                let n_state = succ_spec.cut_state.n_slots();
                let stats_before = ws0.solver.statistics();
                let intercept = run_backward_node_replicated(
                    &mut self.replicated_scratch,
                    ws0,
                    inputs.ctx,
                    training_ctx,
                    inputs.comm,
                    &inputs.risk_measures[node_stage.0],
                    &self.probabilities_buf,
                    &succ_spec,
                    &outcomes,
                    x_hat,
                    pool_id,
                    inputs.iteration,
                    inputs.enumerated_state,
                )?;
                let stage_delta =
                    SolverStatsDelta::from_snapshots(&stats_before, &ws0.solver.statistics());
                SolverStatsDelta::accumulate_into(
                    &mut self.enumerated_stage_stats[successor_stage.0],
                    &stage_delta,
                );
                inputs.fcf.add_cut(
                    node_id,
                    pool_id,
                    inputs.iteration,
                    0,
                    intercept,
                    &ws0.backward_accum.agg_coefficients[..n_state],
                );
                cuts_generated += 1;
            }
        }

        let worker_id = ws0.worker_id;

        let elapsed = start.elapsed();
        #[allow(clippy::cast_possible_truncation)]
        let elapsed_ms = elapsed.as_millis() as u64;
        ws0.worker_timing_buf.backward_wall_ms = elapsed.as_secs_f64() * 1_000.0;
        if let Some(sender) = inputs.event_sender {
            let _ = sender.send(TrainingEvent::WorkerTiming {
                rank: ws0.rank,
                worker_id,
                iteration: inputs.iteration,
                phase: WorkerTimingPhase::Backward,
                timings: ws0.worker_timing_buf,
            });
        }
        let solves_after: u64 = total_solve_count(inputs.workspaces);

        let rank_i32 = i32::try_from(my_rank).map_err(|_| {
            SddpError::Validation(format!(
                "enumerated backward: MPI rank {my_rank} overflows i32 (max {})",
                i32::MAX
            ))
        })?;
        let stage_stats: Vec<(usize, Vec<StageWorkerOpeningDelta>)> = self
            .enumerated_stage_stats
            .iter()
            .enumerate()
            .filter(|(_, delta)| delta.lp_solves > 0)
            .map(|(stage_idx, delta)| {
                (
                    stage_idx,
                    vec![(rank_i32, worker_id, 0_usize, delta.clone())],
                )
            })
            .collect();

        Ok(BackwardResult {
            cuts_generated,
            elapsed_ms,
            lp_solves: solves_after - solves_before,
            stage_stats,
            state_exchange_time_ms: 0,
            cut_batch_build_time_ms: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
            cut_sync_time_ms: 0,
        })
    }

    /// Populate [`Self::node_visit_offsets`] with each level node's base slot
    /// offset for this rank: the count of visits to `level[i]` from strictly-lower
    /// ranks. A cut's node-relative slot index (`offset + compacted`) is then
    /// globally unique across ranks within the pool's own `visit_bound` stride, so
    /// no two ranks' cuts collide and none overshoots into the next iteration's
    /// slot block.
    ///
    /// Degenerate by construction, not by shape dispatch: a single-node level's
    /// one node is visited by every trial, so lower ranks contribute exactly their
    /// full trial counts — `fwd_offset` — and no exchange is needed (the chain hot
    /// path stays collective-free and byte-identical). A multi-node level
    /// allgathers each rank's per-node local visit count (from `routed_offsets`)
    /// and prefix-sums the strictly-lower ranks; on a single rank that prefix is
    /// `0`, leaving the slot derivation `compacted` — the pre-change single-rank
    /// branching layout, byte-for-byte.
    ///
    /// # Errors
    ///
    /// Returns `Err(SddpError::Communication(_))` if the `allgatherv` fails, or
    /// `Err(SddpError::Validation(_))` if a gathered count exceeds `usize`.
    fn compute_node_visit_offsets<C: Communicator>(
        &mut self,
        routed_offsets: &[usize],
        n_nodes: usize,
        fwd_offset: usize,
        comm: &C,
    ) -> Result<(), SddpError> {
        self.node_visit_offsets.clear();
        if n_nodes == 1 {
            self.node_visit_offsets.push(fwd_offset);
            return Ok(());
        }

        let my_rank = comm.rank();
        let num_ranks = comm.size();

        self.bwd_visit_send.clear();
        for i in 0..n_nodes {
            self.bwd_visit_send
                .push((routed_offsets[i + 1] - routed_offsets[i]) as u64);
        }
        let total = num_ranks * n_nodes;
        if self.bwd_visit_recv.len() < total {
            self.bwd_visit_recv.resize(total, 0);
        }
        self.bwd_visit_counts.clear();
        self.bwd_visit_counts
            .extend((0..num_ranks).map(|_| n_nodes));
        self.bwd_visit_displs.clear();
        self.bwd_visit_displs
            .extend((0..num_ranks).map(|r| r * n_nodes));

        comm.allgatherv(
            &self.bwd_visit_send,
            &mut self.bwd_visit_recv[..total],
            &self.bwd_visit_counts,
            &self.bwd_visit_displs,
        )?;

        for node_i in 0..n_nodes {
            let mut offset = 0usize;
            for r in 0..my_rank {
                offset +=
                    usize::try_from(self.bwd_visit_recv[r * n_nodes + node_i]).map_err(|_| {
                        SddpError::Validation(format!(
                            "node-visit count exchange: rank {r}'s count for level node {node_i} \
                         exceeds usize"
                        ))
                    })?;
            }
            self.node_visit_offsets.push(offset);
        }
        Ok(())
    }

    /// Synchronise per-slot cut binding metadata across MPI ranks for one successor
    /// pool via one `allreduce(Sum)` over that pool's `region_offset` slice of
    /// `metadata_sync_contribution`.
    ///
    /// The reduction is bounded to `populated_count`, not the full pool capacity:
    /// slots in `[populated_count, capacity)` are structurally zero on every rank.
    /// `populated_count` is rank-invariant (cuts are added identically on every
    /// rank), so all ranks reduce over the same length.
    fn sync_stage_metadata<C: Communicator>(
        &mut self,
        pool: usize,
        region_offset: usize,
        populated_count: usize,
        iteration: u64,
        workspaces: &[SolverWorkspace<impl SolverInterface>],
        fcf: &mut FutureCostFunction,
        comm: &C,
    ) -> Result<(), SddpError> {
        if populated_count == 0 {
            return Ok(());
        }
        self.metadata_sync_buf.clear();
        self.metadata_sync_buf.resize(populated_count, 0u64);
        for ws in workspaces {
            for (slot, &inc) in ws.backward_accum.metadata_sync_contribution
                [region_offset..region_offset + populated_count]
                .iter()
                .enumerate()
            {
                self.metadata_sync_buf[slot] += inc;
            }
        }
        self.global_increments_buf.clear();
        self.global_increments_buf.resize(populated_count, 0u64);
        comm.allreduce(
            &self.metadata_sync_buf,
            &mut self.global_increments_buf,
            ReduceOp::Sum,
        )
        .map_err(SddpError::from)?;
        for (slot, &inc) in self.global_increments_buf.iter().enumerate() {
            if inc > 0 {
                fcf.pools[pool].record_binding(slot, inc, iteration);
            }
        }
        Ok(())
    }

    /// Decompose one stage's parallel overhead into setup, load-imbalance, and
    /// scheduling components, returning `(setup_ms, imbalance_ms, scheduling_ms)`.
    ///
    /// Deltas are taken against the before-snapshot in `self.worker_stats_before`.
    fn collect_stage_timing_stats<S: SolverInterface + Send>(
        &mut self,
        parallel_wall_ms: u64,
        n_workers: f64,
        workspaces: &mut [SolverWorkspace<S>],
    ) -> (u64, u64, u64) {
        self.worker_stats_after.clear();
        self.worker_stats_after
            .extend(workspaces.iter().map(|w| w.solver.statistics()));
        self.worker_deltas.clear();
        self.worker_deltas.extend(
            self.worker_stats_before
                .iter()
                .zip(&self.worker_stats_after)
                .map(|(before, after)| SolverStatsDelta::from_snapshots(before, after)),
        );
        let stage_setup_ms: f64 = self.worker_deltas.iter().map(non_solve_setup_ms).sum();
        for (ws, delta) in workspaces.iter_mut().zip(&self.worker_deltas) {
            ws.worker_timing_buf.bwd_setup_ms += non_solve_setup_ms(delta);
        }
        self.worker_totals.clear();
        self.worker_totals.extend(
            self.worker_deltas
                .iter()
                .map(|d| d.solve_time_ms + non_solve_setup_ms(d)),
        );
        let max_worker_ms = self.worker_totals.iter().copied().fold(0.0_f64, f64::max);
        let avg_worker_ms = if self.worker_totals.is_empty() {
            0.0_f64
        } else {
            self.worker_totals.iter().sum::<f64>() / n_workers
        };
        let stage_imbalance_ms = (max_worker_ms - avg_worker_ms).max(0.0);
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let stage_scheduling_ms = (parallel_wall_ms as f64 - max_worker_ms).max(0.0);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            (
                stage_setup_ms as u64,
                stage_imbalance_ms as u64,
                stage_scheduling_ms as u64,
            )
        }
    }

    /// Pack local per-worker solver statistics, `allgatherv` across all MPI ranks,
    /// unpack the result, and return per-`(rank, worker_id, opening)` delta entries.
    ///
    /// Returns one `StageWorkerOpeningDelta` per `(rank, worker, opening)` triple
    /// where `delta.lp_solves > 0` or `omega == 0` (the ω=0 sentinel always
    /// appears so that downstream stage-stats consumers can detect "stage visited").
    fn gather_stage_solver_stats<C: Communicator>(
        &mut self,
        n_openings: usize,
        n_ranks: usize,
        n_workers_local: usize,
        bwd_max_openings: usize,
        workspaces: &[SolverWorkspace<impl SolverInterface>],
        comm: &C,
    ) -> Result<Vec<StageWorkerOpeningDelta>, SddpError> {
        self.stage_worker_stats_buf.reset();
        for ws in workspaces {
            debug_assert_eq!(
                ws.backward_accum.per_opening_stats.len(),
                n_openings,
                "per_opening_stats length must equal n_openings on every worker"
            );
            #[allow(clippy::cast_sign_loss)]
            let wid = ws.worker_id as usize;
            for omega in 0..n_openings {
                self.stage_worker_stats_buf.set(
                    wid,
                    omega,
                    ws.backward_accum.per_opening_stats[omega].clone(),
                );
            }
        }
        pack_worker_opening_stats(
            &mut self.bwd_stats_send_buf,
            self.stage_worker_stats_buf.as_slice(),
            n_workers_local,
            bwd_max_openings,
        );
        comm.allgatherv(
            &self.bwd_stats_send_buf,
            &mut self.bwd_stats_recv_buf,
            &self.bwd_stats_counts,
            &self.bwd_stats_displs,
        )
        .map_err(SddpError::Communication)?;
        debug_assert_eq!(
            self.bwd_stats_recv_buf.len(),
            n_ranks * n_workers_local * bwd_max_openings * WORKER_STATS_ENTRY_STRIDE,
            "recv buffer length must equal n_ranks * n_workers_local * bwd_max_openings * STRIDE"
        );
        unpack_worker_opening_stats(
            &self.bwd_stats_recv_buf,
            &mut self.bwd_stats_unpack_buf,
            n_ranks * n_workers_local,
            bwd_max_openings,
        );
        let mut entries: Vec<StageWorkerOpeningDelta> =
            Vec::with_capacity(n_ranks * n_workers_local * n_openings);
        for r in 0..n_ranks {
            let rank_i32 = i32::try_from(r).map_err(|_| {
                SddpError::Validation(format!(
                    "MPI rank count {r} overflows i32 (max {})",
                    i32::MAX
                ))
            })?;
            for w in 0..n_workers_local {
                let wid_i32 = i32::try_from(w).map_err(|_| {
                    SddpError::Validation(format!(
                        "worker count {w} overflows i32 (max {})",
                        i32::MAX
                    ))
                })?;
                for omega in 0..n_openings {
                    let flat = (r * n_workers_local + w) * bwd_max_openings + omega;
                    let delta = &self.bwd_stats_unpack_buf[flat];
                    if delta.lp_solves > 0 || omega == 0 {
                        entries.push((rank_i32, wid_i32, omega, delta.clone()));
                    }
                }
            }
        }
        Ok(entries)
    }
}

/// Test/tooling accessors for `BackwardPassState::by_node_scratch` — never called
/// from production hot-path code.
#[cfg(any(test, feature = "test-support"))]
impl BackwardPassState {
    /// `.capacity()` of the opening-block scratch arena — `0` unless `ByNode`
    /// could be dispatched, else the `max_local_fwd * bwd_max_openings` shape
    /// `ByNodeScratch::sized` allocated by [`Self::resize_by_node_scratch`].
    #[must_use]
    pub fn by_node_scratch_arena_capacity(&self) -> usize {
        self.by_node_scratch.arena.capacity()
    }

    /// Read-only view of the opening-block scratch arena, for sizing assertions
    /// (`.len()`, each entry's `coefficients.len()`).
    #[must_use]
    pub fn by_node_scratch_arena(&self) -> &[BackwardOutcome] {
        &self.by_node_scratch.arena
    }

    /// Per-`(generating node, block-index)` mean `simplex_iterations` pivot from
    /// the by-node scheduler's rank-local accumulator (CA5). Outer index is the
    /// generating node's canonical position (on a chain, equal to the backward
    /// pass's successor stage `t + 1`); inner index is the block index. `None`
    /// where no opening was solved this iteration (count == 0); empty under
    /// `BackwardScheduler::ByScenario`.
    #[must_use]
    pub fn block_pivot_means(&self) -> Vec<Vec<Option<f64>>> {
        let stride = self.by_node_scratch.n_blocks_max;
        if stride == 0 {
            return Vec::new();
        }
        self.by_node_scratch
            .block_pivots
            .chunks(stride)
            .map(|stage_row| {
                stage_row
                    .iter()
                    .map(|&(sum, count)| {
                        #[allow(clippy::cast_precision_loss)]
                        (count > 0).then(|| sum as f64 / count as f64)
                    })
                    .collect()
            })
            .collect()
    }
}

/// Per-worker non-solve overhead for one stage: LP load + bound patch + basis-set
/// time. Shared by `collect_stage_timing_stats`'s setup-time sum, its
/// `bwd_setup_ms` accumulation, and (plus `solve_time_ms`) its per-worker totals.
fn non_solve_setup_ms(d: &SolverStatsDelta) -> f64 {
    d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms
}

/// Iteration-constant values derived once from `BackwardPassInputs` at the start of `run`.
///
/// Passed to `compute_one_backward_node` to avoid recomputing them on every node
/// and to keep the argument count of that helper within budget.
struct StageDerivedParams {
    /// This rank's MPI rank index.
    my_rank: usize,
    /// Number of rayon workers on this rank.
    n_workers_local: usize,
    /// Total MPI rank count.
    n_ranks: usize,
    /// Maximum opening count across all stages (stride for stats buffers).
    bwd_max_openings: usize,
    /// `n_workers_local as f64`, pre-cast for load-imbalance arithmetic.
    n_workers: f64,
}

/// Aggregate output of one cut-sharing level, accumulated in [`BackwardPassState::run`].
struct LevelOutput {
    /// Global cuts added across every node in the level (rank-count invariant).
    cuts_generated: usize,
    /// Per-node `(successor stage, per-(rank, worker, opening) deltas)`.
    stage_entries: Vec<(usize, Vec<StageWorkerOpeningDelta>)>,
    /// Time in the level's one `exchange.exchange()`, in ms.
    state_exchange_ms: u64,
    /// `build_delta_cut_row_batch_into` time summed over the level's nodes, in ms.
    cut_batch_build_ms: u64,
    /// Aggregate non-solve setup time summed over the level's nodes, in ms.
    setup_ms: u64,
    /// Load-imbalance component summed over the level's nodes, in ms.
    imbalance_ms: u64,
    /// Scheduling overhead summed over the level's nodes, in ms.
    scheduling_ms: u64,
    /// Time in the level's one batched cut-sync `allgatherv`, in ms.
    cut_sync_ms: u64,
}

/// Per-node result of [`compute_one_backward_node`], consumed by the level's
/// batched cut exchange and per-node metadata/timing/stats epilogue.
struct NodeCompute {
    /// Successor stage index (`node.stage + 1`), the `stage_stats` key.
    successor_stage: usize,
    /// The node's own cut pool (where its cut was inserted).
    pool_id: usize,
    /// Range into `level_pool_regions_scratch` of this node's successor pool
    /// regions — one per child, so the level's metadata sync targets each child's
    /// OWN pool.
    pool_regions: Range<usize>,
    /// Opening count of the node's successor outcome set (stats stride).
    n_openings: usize,
    /// `build_delta_cut_row_batch_into` time for this node, in ms.
    cut_batch_build_ms: u64,
    /// Aggregate non-solve setup time for this node, in ms.
    setup_ms: u64,
    /// Load-imbalance component for this node, in ms.
    imbalance_ms: u64,
    /// Scheduling overhead for this node, in ms.
    scheduling_ms: u64,
}

/// Milliseconds elapsed since `start`; every duration measured here is well
/// under `u64::MAX` ms.
#[allow(clippy::cast_possible_truncation)]
fn ms_elapsed(start: Instant) -> u64 {
    start.elapsed().as_millis() as u64
}

/// Total LP solve count across every worker's solver, for a before/after delta.
fn total_solve_count<S: SolverInterface>(workspaces: &[SolverWorkspace<S>]) -> u64 {
    workspaces
        .iter()
        .map(|ws| ws.solver.statistics().solve_count)
        .sum()
}

/// Resolve the effective backward thread scheduler for the SAMPLED path
/// (`compute_one_backward_node`'s only caller): an active Dynamic Cut
/// Selection iteration always forces the by-scenario path (its cut-free lazy
/// core is incompatible with the by-node frozen-LP load); otherwise the
/// configured scheduler is unchanged.
fn resolve_backward_scheduler(
    dcs_active: bool,
    configured: BackwardScheduler,
) -> BackwardScheduler {
    if dcs_active {
        BackwardScheduler::ByScenario {}
    } else {
        configured
    }
}

/// Flatten node `node_pos`'s successor outcome set
/// `O(n) = {(m, ψ): m ∈ n⁺, ψ ∈ Ω_m}` into `out`, canonical order (ascending
/// child node id — `node_graph.successors[node_pos]`'s own invariant — then
/// within-child ω): `CVaR`'s tail weighting is index-order-sensitive (sddp.md
/// "Backward opening order is warm-start-only"). Delegates to the shared
/// [`crate::setup::node_graph::assemble_outcome_weights`] primitive — the
/// single owner of this fill, shared with
/// `lower_bound::assemble_root_outcome_weights`.
fn assemble_successor_outcome_weights(
    node_graph: &NodeGraph,
    node_pos: NodePos,
    out: &mut Vec<f64>,
) {
    crate::setup::node_graph::assemble_outcome_weights(
        node_graph,
        &node_graph.successors[node_pos],
        out,
    );
}

/// Future recombination/DAG insurance (debug-only): the enumerated backward
/// keys each node's own incoming state purely structurally — a node reads the
/// persisted `out_state` of its single predecessor by `NodePos`. This asserts the
/// only way that keying could become ambiguous never arises: for every node,
/// the `out_states` of its predecessors (the nodes with an edge into it) are
/// pairwise `to_bits`-distinct. A recombination node (in-degree >= 2) whose two
/// predecessors carry identical states could not tell which predecessor's state
/// it inherited. The check is PER-NODE-PREDECESSORS, never pairwise across
/// distinct cut-generating nodes: two distinct nodes legitimately share a state
/// (a deterministic trunk's parent and child, two fan branches settling on the
/// same binding bound) — ordinary LP degeneracy, not a keying hazard, and a
/// backend-dependent one at that. The current graph is in-degree 1
/// (`build_parent_map` rejects a second predecessor), so every node has at most
/// one predecessor and this is vacuous; it only bites a future >= 2-predecessor
/// recombination.
#[cfg(debug_assertions)]
fn debug_assert_node_predecessor_states_distinct(
    node_graph: &NodeGraph,
    enumerated_state: &EnumeratedForwardScratch,
) {
    let mut predecessors: Vec<Vec<NodePos>> = vec![Vec::new(); node_graph.nodes.len()];
    for (pos, succs) in node_graph.successors.iter_indexed() {
        for succ in succs {
            predecessors[succ.child.0].push(pos);
        }
    }
    assert_predecessor_states_distinct(&predecessors, enumerated_state);
}

/// The per-node-predecessors core of
/// [`debug_assert_node_predecessor_states_distinct`], split out so a unit test
/// can drive it with a hand-built predecessor adjacency and `out_state` arena
/// (`StudySetup::new` hard-rejects recombination under enumerated, so no
/// >= 2-predecessor node is constructible through a real study).
#[cfg(debug_assertions)]
fn assert_predecessor_states_distinct(
    predecessors: &[Vec<NodePos>],
    enumerated_state: &EnumeratedForwardScratch,
) {
    for preds in predecessors {
        for (i, &p) in preds.iter().enumerate() {
            let p_state = enumerated_state.out_state(p);
            for &q in &preds[i + 1..] {
                let q_state = enumerated_state.out_state(q);
                let identical = p_state.len() == q_state.len()
                    && p_state
                        .iter()
                        .zip(q_state)
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                debug_assert!(
                    !identical,
                    "enumerated backward: predecessors {p} and {q} of one node carry \
                     bit-identical persisted states; a recombination node's incoming state \
                     is ambiguous under structural (NodePos) keying"
                );
            }
        }
    }
}

/// Group this rank's own local trial points (`0..local_work`) by the
/// cut-generating node each visited at `level_stage`, into CSR form:
/// `routed[offsets[i]..offsets[i + 1]]` are the ascending trial-point indices
/// whose visit resolves to `level[i]`'s pool. A cut-generating node always owns
/// its own pool, so grouping by visited node *is* per-pool routing — each cut
/// anchors at its own node's sampled incoming states (Σ over a level's pools of
/// their trial counts = `local_work`).
///
/// A single-node level carries exactly one node at its stage (a mid-horizon
/// leaf beside it is rejected at setup), so every trial point that reached the
/// stage visited that node: route all with no per-record lookup. This is the
/// chain and terminal-fan case, and reproduces the pre-routing `0..local_work`
/// slice byte-for-byte. Only a multi-node level reads `TrajectoryRecord::node_id`
/// to split siblings that own distinct pools.
fn build_trial_routing(
    node_graph: &NodeGraph,
    level: &[NodePos],
    level_stage: StageIdx,
    records: &[TrajectoryRecord],
    num_stages: usize,
    local_work: usize,
    routed: &mut Vec<usize>,
    offsets: &mut Vec<usize>,
) {
    routed.clear();
    offsets.clear();
    offsets.push(0);
    if level.len() == 1 {
        routed.extend(0..local_work);
        offsets.push(local_work);
        return;
    }
    for &node_pos in level {
        let node_id = node_graph.node_ids[node_pos];
        for m in 0..local_work {
            if records[m * num_stages + level_stage.0].node_id == node_id {
                routed.push(m);
            }
        }
        offsets.push(routed.len());
    }
    debug_assert_eq!(
        routed.len(),
        local_work,
        "every local trial point must route to exactly one cut-generating node in the level \
         (a mid-horizon leaf would be rejected at setup); routed {} of {local_work}",
        routed.len()
    );
}

/// Guard the visit-bound overflow once per `(pool, iteration)`, at the routing
/// site and ahead of any per-node compute: [`CutPool::visit_stride`] is a
/// statistical floor under sampled branching
/// ([`node_graph::NodeGraph::pool_cut_stride`](crate::setup::node_graph::NodeGraph::pool_cut_stride)),
/// never a guarantee, so a realized routed count above it would address the
/// next iteration's slot block (`cut/pool.rs`'s append-only slot formula).
/// Rejects rather than clamps: the run is seed-deterministic, so the
/// rejection reproduces.
///
/// A single-node level's total is `total_fwd` — every local trial routes to
/// that sole node on every rank, so the level-wide real forward-pass count
/// already equals it, with no extra collective. A multi-node level's per-node
/// total is summed from `bwd_visit_recv`'s `(rank, node)` counts, gathered by
/// the immediately-preceding [`BackwardPassState::compute_node_visit_offsets`]
/// call for the SAME purpose — reading it here adds no collective either. A
/// chain or `derived_count == 1` enumerated pool's reach probability is
/// exactly `1`, so `pool_cut_stride`'s variance term is exactly `0` and its
/// stride equals `total_fwd` bit-for-bit: the comparison below can only ever
/// see equality on that shape, never an overflow — no shape predicate needed.
///
/// # Errors
///
/// Returns `Err(SddpError::Validation(_))` naming the pool, the iteration,
/// the routed count, and the stride when a level node's total routed count
/// exceeds its pool's [`CutPool::visit_stride`].
///
/// [`CutPool::visit_stride`]: crate::cut::pool::CutPool::visit_stride
fn check_visit_bound(
    node_graph: &NodeGraph,
    level: &[NodePos],
    iteration: u64,
    fcf: &FutureCostFunction,
    total_fwd: usize,
    bwd_visit_recv: &[u64],
    num_ranks: usize,
) -> Result<(), SddpError> {
    let n_nodes = level.len();
    for (level_idx, &node_pos) in level.iter().enumerate() {
        let pool_id = node_graph.nodes[node_pos].pool_id;
        let routed: u64 = if n_nodes == 1 {
            total_fwd as u64
        } else {
            (0..num_ranks)
                .map(|r| bwd_visit_recv[r * n_nodes + level_idx])
                .sum()
        };
        let stride = u64::from(fcf.pools[pool_id].visit_stride);
        if routed > stride {
            return Err(SddpError::Validation(format!(
                "visit-bound overflow: pool {pool_id} routed {routed} trial points at \
                 iteration {iteration}, exceeding its cut-receipt stride of {stride}. The \
                 stride is a statistical floor under sampled branching (pool_cut_stride), not \
                 a guarantee; a cut at forward_pass_index >= stride would address into the \
                 next iteration's slot block, corrupting slot identity. This iteration's \
                 realized branching exceeded the configured margin — the rejection is \
                 reproducible (seed-deterministic); raising forward_passes changes the margin \
                 for a future run, but this run's cuts cannot be admitted silently."
            )));
        }
    }
    Ok(())
}

/// Execute one reverse-topological cut-sharing level: one state exchange for the
/// level, each node's backward cut computed independently over ONLY the trial
/// points routed to its pool, ONE batched cut exchange over the level's
/// trial-point-distributed pools, then per-node metadata/timing/stats. Absent
/// `nodes[]` a level is one node (== stage) and every trial routes to it, so
/// this is byte-for-byte the reversed stage loop (one state exchange, one cut
/// exchange, one metadata/stats gather).
fn run_one_backward_level<S: SolverInterface + Send, C: Communicator>(
    state: &mut BackwardPassState,
    inputs: &mut BackwardPassInputs<'_, S, C>,
    level: &[NodePos],
    params: &StageDerivedParams,
) -> Result<LevelOutput, SddpError> {
    let training_ctx = inputs.training_ctx;
    let num_stages = training_ctx.horizon.num_stages();
    // Every node in a level shares one stage; the stage-keyed state records
    // cover them all in ONE `allgatherv` — one exchange per level, never per node
    // (sddp.md "Per-level exchange in the backward pass").
    let level_stage = training_ctx.node_graph.nodes[level[0]].stage;

    let exch_start = Instant::now();
    inputs
        .exchange
        .exchange(inputs.records, level_stage, num_stages, inputs.comm)?;
    let state_exchange_ms = ms_elapsed(exch_start);

    // Every rank's real (non-padded) forward-pass count for this level, already
    // computed by `exchange()` above with no further collective — the total
    // [`check_visit_bound`] compares a single-node level's routed count
    // against (every trial routes to that level's sole node).
    let total_fwd = inputs.exchange.real_total_scenarios();

    if let Some(ref mut archive) = inputs.visited_archive {
        inputs
            .exchange
            .pack_real_states_into(&mut state.real_states_buf);
        // Archive by NODE position, not `level_stage` used as a node index:
        // sibling nodes at one stage own distinct pools and cut regions, so each
        // reads back its own archive (`states_for_node`). The exchange gathers the
        // level's states without per-node split, so every node in a multi-node
        // level receives the level's states — a conservative over-inclusion that
        // never drops a binding cut. On a chain the level is one node, giving the
        // former single stage-keyed bucket byte-for-byte.
        for &node_pos in level {
            archive.archive_gathered_states(node_pos, &state.real_states_buf, total_fwd);
        }
    }

    // Route this rank's trial points to the level's pools once (a mem-swapped
    // local so the borrow does not conflict with `&mut state` below); a
    // cut-generating node then anchors its cut ONLY at the states its own pool
    // was visited with.
    let mut routed_trials = std::mem::take(&mut state.routed_trials_scratch);
    let mut routed_offsets = std::mem::take(&mut state.routed_offsets_scratch);
    build_trial_routing(
        training_ctx.node_graph,
        level,
        level_stage,
        inputs.records,
        num_stages,
        inputs.local_work,
        &mut routed_trials,
        &mut routed_offsets,
    );

    // Base slot offset per level node so a cut's node-relative slot index stays
    // globally unique across ranks at the pool's own `visit_bound` stride.
    state.compute_node_visit_offsets(
        &routed_offsets,
        level.len(),
        inputs.fwd_offset,
        inputs.comm,
    )?;

    // Once per (pool, iteration), ahead of any per-node compute: reject a
    // realized routed count above the pool's cut-receipt stride rather than
    // let it silently address into the next iteration's slot block.
    check_visit_bound(
        training_ctx.node_graph,
        level,
        inputs.iteration,
        inputs.fcf,
        total_fwd,
        &state.bwd_visit_recv,
        inputs.comm.size(),
    )?;

    let node_visit_offsets = std::mem::take(&mut state.node_visit_offsets);

    // Reuse the per-level buffers across levels (mem-swapped out so the loop
    // allocates nothing per level; the empty placeholder left on `state` frees
    // the mutable-borrow conflict with `compute_one_backward_node`).
    let mut nodes_out = std::mem::take(&mut state.level_nodes_scratch);
    nodes_out.clear();
    // Each node appends its successor pool regions here; ranges recorded in
    // `NodeCompute::pool_regions` drive the per-distinct-pool metadata sync below.
    state.level_pool_regions_scratch.clear();
    let mut cut_batch_build_ms = 0u64;
    for (level_idx, &node_pos) in level.iter().enumerate() {
        let trial_points = &routed_trials[routed_offsets[level_idx]..routed_offsets[level_idx + 1]];
        let nc = compute_one_backward_node(
            state,
            inputs,
            node_pos,
            trial_points,
            node_visit_offsets[level_idx],
            params,
        )?;
        cut_batch_build_ms += nc.cut_batch_build_ms;
        nodes_out.push(nc);
    }

    // ONE batched cut exchange over the level's pools (not one collective per
    // node): all the level's nodes' cuts go out in a single `allgatherv`.
    let sync_start = Instant::now();
    let mut level_pools = std::mem::take(&mut state.level_pools_scratch);
    level_pools.clear();
    level_pools.extend(nodes_out.iter().map(|nc| nc.pool_id));
    // The sampled path aggregates every pool by trial-point distribution, so no
    // pool is replicated: the batched exchange runs over all of the level's pools.
    let (n_local_total, remote_total) = inputs.cut_sync_bufs.sync_level_records(
        &level_pools,
        &[],
        inputs.fcf,
        inputs.iteration,
        inputs.comm,
    )?;
    // Global cuts added across the level: every rank's local share plus every
    // peer's; the local count scales with rank count while the global total is
    // rank-count invariant.
    let cuts_generated = n_local_total + remote_total;
    let cut_sync_ms = ms_elapsed(sync_start);

    let mut stage_entries = Vec::with_capacity(nodes_out.len());
    let mut setup_ms = 0u64;
    let mut imbalance_ms = 0u64;
    let mut scheduling_ms = 0u64;
    // Mem-swapped local so the per-region borrow does not conflict with
    // `state.sync_stage_metadata`'s `&mut self`.
    let pool_regions = std::mem::take(&mut state.level_pool_regions_scratch);
    for nc in &nodes_out {
        // Sync EACH successor child's own pool at its own metadata slot region, so a
        // child's binding activity lands in its own pool (an empty pool early-returns).
        for pr in &pool_regions[nc.pool_regions.clone()] {
            state.sync_stage_metadata(
                pr.pool_id,
                pr.region_offset,
                pr.populated_count,
                inputs.iteration,
                inputs.workspaces,
                inputs.fcf,
                inputs.comm,
            )?;
        }
        setup_ms += nc.setup_ms;
        imbalance_ms += nc.imbalance_ms;
        scheduling_ms += nc.scheduling_ms;
        let entries = state.gather_stage_solver_stats(
            nc.n_openings,
            params.n_ranks,
            params.n_workers_local,
            params.bwd_max_openings,
            inputs.workspaces,
            inputs.comm,
        )?;
        stage_entries.push((nc.successor_stage, entries));
    }

    state.level_nodes_scratch = nodes_out;
    state.level_pools_scratch = level_pools;
    state.level_pool_regions_scratch = pool_regions;
    state.routed_trials_scratch = routed_trials;
    state.routed_offsets_scratch = routed_offsets;
    state.node_visit_offsets = node_visit_offsets;

    Ok(LevelOutput {
        cuts_generated,
        stage_entries,
        state_exchange_ms,
        cut_batch_build_ms,
        setup_ms,
        imbalance_ms,
        scheduling_ms,
        cut_sync_ms,
    })
}

/// Compute one node's backward cut over `trial_points` — the trial-point
/// indices routed to this node's pool — building the successor cut batch,
/// solving (the by-node or by-scenario path), inserting the local cut(s),
/// reconciling, and computing this node's timing. The level driver owns the
/// state exchange, the routing, the batched cut exchange, and the
/// metadata/stats gather. Sampled-only — the enumerated backward's per-node
/// solve is `BackwardPassState::run_enumerated_backward`.
// RATIONALE: sequences the per-node solve phases whose intermediate state each
// next phase reads; splitting further would thread every disjoint sub-field as
// &mut without gaining clarity.
#[allow(clippy::too_many_lines)]
fn compute_one_backward_node<S: SolverInterface + Send, C: Communicator>(
    state: &mut BackwardPassState,
    inputs: &mut BackwardPassInputs<'_, S, C>,
    node_pos: NodePos,
    trial_points: &[usize],
    node_visit_offset: usize,
    params: &StageDerivedParams,
) -> Result<NodeCompute, SddpError> {
    let training_ctx = inputs.training_ctx;
    let ctx = inputs.ctx;
    let node_graph = training_ctx.node_graph;
    let node_stage = node_graph.nodes[node_pos].stage;
    let successor_stage = node_stage.next();
    let pool_id = node_graph.nodes[node_pos].pool_id;
    // Declared id of the GENERATING node — carried as cut provenance so a
    // shared-leaf pool's cuts stay distinguishable by node. Canonical
    // (declaration-order-invariant); on a chain it equals the stage.
    let node_id = node_graph.node_ids[node_pos];
    let cut_state_projection = &training_ctx.cut_state_layouts[pool_id];

    state.worker_stats_before.clear();
    state
        .worker_stats_before
        .extend(inputs.workspaces.iter().map(|w| w.solver.statistics()));

    // Flattened weights over the node's whole successor outcome set (canonical
    // order); the reified `SuccessorOutcomes` built below carries one entry per
    // child aligned to this vector's contiguous per-child slices.
    assemble_successor_outcome_weights(node_graph, node_pos, &mut state.probabilities_buf);
    let n_openings = state.probabilities_buf.len();

    let batch_start = Instant::now();
    let template_num_rows = ctx.template(successor_stage).num_rows;
    reify_successor_outcomes(
        &mut state.successor_meta_buf,
        &mut state.successor_active_slots_buf,
        ctx,
        training_ctx,
        inputs.fcf,
        inputs.cut_batches,
        inputs.frozen,
        node_pos,
        inputs.iteration,
    );
    // Successor binding-metadata regions; kept in this caller because the enumerated
    // fork syncs metadata through run_backward_node_replicated instead.
    let pool_regions_start = state.level_pool_regions_scratch.len();
    for entry in &state.successor_meta_buf {
        state.level_pool_regions_scratch.push(PoolRegion {
            pool_id: entry.pool_id,
            region_offset: entry.metadata_offset,
            populated_count: entry.populated_count,
        });
    }
    let pool_regions = pool_regions_start..state.level_pool_regions_scratch.len();
    let cut_batch_build_ms = ms_elapsed(batch_start);

    let outcomes = SuccessorOutcomes::new(
        &state.successor_meta_buf,
        &state.successor_active_slots_buf,
        inputs.frozen,
        &*inputs.cut_batches,
        &inputs.fcf.pools,
        template_num_rows,
        inputs.cut_activity_tolerance,
    );
    debug_assert_eq!(
        outcomes.total_outcomes(),
        n_openings,
        "the reified successor outcome set must have as many outcomes as flattened weights"
    );

    let succ_spec = SuccessorSpec {
        t: node_stage,
        successor: successor_stage,
        my_rank: params.my_rank,
        probabilities: &state.probabilities_buf,
        cut_state: cut_state_projection,
    };

    // `resolve_backward_scheduler` owns the DCS fallback: its cut-free lazy
    // core is incompatible with the by-node frozen-LP load (sddp.md "By-node
    // scheduler is warm-start-only").
    let dcs_active = training_ctx
        .dcs
        .filter(|p| p.is_active(inputs.iteration))
        .is_some();
    let use_by_node = matches!(
        resolve_backward_scheduler(dcs_active, state.scheduler),
        BackwardScheduler::ByNode { .. }
    );

    let process_start = Instant::now();
    let (local_solve, parallel_wall_ms): (Result<usize, SddpError>, u64) = if use_by_node {
        let configured_block_size = match state.scheduler {
            BackwardScheduler::ByNode { block_size } => block_size,
            BackwardScheduler::ByScenario {} => None,
        };
        let block_size = resolve_block_size(n_openings, configured_block_size);
        let n_blocks = by_node_block_count(n_openings, block_size);
        if state.hardest_first_claim_order {
            // Hardest-first order keyed by the GENERATING node (CA5), so sibling fan
            // nodes at one level each read their own history — never a shared
            // successor-stage row. Chain: one node per stage, byte-identical.
            let row = state.by_node_scratch.block_pivot_row(node_pos, n_blocks);
            hardest_first_block_order(
                &state.by_node_scratch.block_pivots_prev[row],
                n_blocks,
                &mut state.by_node_scratch.block_order,
            );
        } else {
            identity_block_order(n_blocks, &mut state.by_node_scratch.block_order);
        }
        // The by-node scheduler claims over the whole reified successor outcome set:
        // each block partitions the flattened outcomes and every contiguous same-child
        // run inside it prices that child against its OWN LP. A single-child node has
        // one run per block, reproducing the chain claim loop byte-for-byte.
        let worker_out = process_stage_backward_by_node(
            inputs.workspaces,
            ctx,
            training_ctx,
            trial_points,
            inputs.exchange,
            inputs.fwd_offset,
            inputs.iteration,
            &succ_spec,
            &outcomes,
            &*inputs.basis_store,
            block_size,
            &state.by_node_scratch.block_order[..n_blocks],
        );
        let elapsed_ms = ms_elapsed(process_start);
        // Telemetry-only merge (sddp.md "by-node scheduler is
        // warm-start-only" — disjoint from the per-`(m, ω)` arena scatter and
        // ascending-m aggregation `by_node_finish` performs below); keyed by the
        // generating node so multi-node siblings never share a hardest-first row.
        merge_block_pivots(
            inputs.workspaces.iter().map(|ws| {
                (
                    ws.backward_accum.block_pivot_sum.as_slice(),
                    ws.backward_accum.block_pivot_count.as_slice(),
                )
            }),
            n_blocks,
            node_pos,
            &mut state.by_node_scratch,
        );
        let result = by_node_finish(
            worker_out,
            &*inputs.workspaces,
            trial_points,
            n_openings,
            cut_state_projection.n_slots(),
            &state.probabilities_buf,
            &inputs.risk_measures[node_stage.0],
            inputs.fcf,
            node_id,
            pool_id,
            inputs.iteration,
            node_visit_offset,
            &mut state.by_node_scratch,
        );
        (result, elapsed_ms)
    } else {
        let basis_slices = inputs
            .basis_store
            .split_workers_mut(params.n_workers_local.max(1));
        let worker_staged = process_stage_backward(
            inputs.workspaces,
            ctx,
            training_ctx,
            trial_points,
            inputs.exchange,
            inputs.fwd_offset,
            node_visit_offset,
            inputs.iteration,
            inputs.risk_measures,
            &succ_spec,
            &outcomes,
            basis_slices,
        );
        let elapsed_ms = ms_elapsed(process_start);

        let result = by_scenario_finish(
            worker_staged,
            &*inputs.workspaces,
            trial_points,
            cut_state_projection.n_slots(),
            inputs.fcf,
            node_id,
            pool_id,
            inputs.iteration,
            &mut state.staged_cuts_buf,
        );
        (result, elapsed_ms)
    };

    // Reconcile the divergent backward solve outcome BEFORE the level's sync
    // collectives (mirrors the forward-phase precedent): ranks solve disjoint
    // trial points, so a failure on a strict subset makes every rank return Err
    // here rather than let a healthy rank block in the batched cut sync /
    // metadata sync / stats gather while a peer skipped them.
    reconcile_result(local_solve, inputs.comm, &mut state.reconcile_scratch)?;

    let (setup_ms, imbalance_ms, scheduling_ms) =
        state.collect_stage_timing_stats(parallel_wall_ms, params.n_workers, inputs.workspaces);

    Ok(NodeCompute {
        successor_stage: successor_stage.0,
        pool_id,
        pool_regions,
        n_openings,
        cut_batch_build_ms,
        setup_ms,
        imbalance_ms,
        scheduling_ms,
    })
}

/// Evaluate this node's routed `trial_points` for a single backward stage,
/// returning staged cuts (one per trial point).
// RATIONALE: 10 args are individually-borrowed slices passed through the rayon closure
// boundary. Bundling them into a struct would require either cloning or an `Arc`, both of
// which conflict with the zero-allocation HPC constraint for backward-pass hot code.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_stage_backward<S: SolverInterface + Send>(
    workspaces: &mut [SolverWorkspace<S>],
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    trial_points: &[usize],
    exchange: &ExchangeBuffers,
    fwd_offset: usize,
    node_visit_offset: usize,
    iteration: u64,
    risk_measures: &[RiskMeasure],
    succ: &SuccessorSpec<'_>,
    outcomes: &SuccessorOutcomes<'_>,
    basis_slices: Vec<BasisStoreSliceMut<'_>>,
) -> Vec<Result<(usize, Vec<StagedCut>), SddpError>> {
    let n_openings = succ.probabilities.len();
    // Per-stage cut dimension (pool `t`'s `CutStateProjection`). Buffers reused
    // across stages are resized to EXACTLY this each stage, never grown to a
    // per-worker max: a reduced stage after a full one must shrink, or
    // `write_opening_outcome`'s `copy_from_slice` reads stale full-length data.
    let cut_n_state = succ.cut_state.n_slots();
    // Sized to the max populated pool across the node's children so a child with
    // more cuts than the first cannot index the binding-metadata buffer out of bounds.
    let pop = outcomes.total_metadata_len();

    // Opening-solve strategy, chosen once per stage. The decision depends only on
    // `iteration`, so it is constant across workers and trial points.
    let opening_solver = StageOpeningSolver::from_dcs_params(
        training_ctx
            .dcs
            .filter(|params| params.is_active(iteration)),
    );

    workspaces
        .par_iter_mut()
        .zip(basis_slices.into_par_iter())
        .enumerate()
        .map(|(w, (ws, mut basis_slice))| {
            // Pre-allocate per-stage buffers. This touches only `ws.backward_accum`,
            // never `ws.solver`: each child's LP load is issued inside
            // `process_by_scenario_backward`, not here.
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
            ws.backward_accum
                .agg_coefficients
                .resize(cut_n_state, 0.0_f64);
            if ws.backward_accum.metadata_sync_contribution.len() < pop {
                ws.backward_accum
                    .metadata_sync_contribution
                    .resize(pop, 0u64);
            }
            ws.backward_accum.metadata_sync_contribution[..pop].fill(0);
            ws.backward_accum
                .per_opening_stats
                .resize_with(n_openings, SolverStatsDelta::default);
            for slot in &mut ws.backward_accum.per_opening_stats[..n_openings] {
                *slot = SolverStatsDelta::default();
            }

            // Worker `w` processes exactly the routed trial points whose GLOBAL
            // scenario index falls in its own basis-slice window, so
            // `basis_slice.get(m, node)` is in-bounds by construction. A node's
            // routed subset is scattered across the global scenario axis, so an
            // even split of `trial_points.len()` would misalign with the
            // contiguous per-worker basis window and index out of the slice. On a
            // single-node level the window tiling reproduces the pre-routing
            // `partition(local_work, n_workers, w)` assignment exactly.
            let (w_start, w_end) = basis_slice.scenario_window();
            let owns = move |m: usize| w_start <= m && m < w_end;
            let count_w = trial_points.iter().filter(|&&m| owns(m)).count();
            // Grow-only arena (one `cut_n_state` slot per owned trial point);
            // content is overwritten per trial point before read, so no zero-fill.
            // Stride is `cut_n_state` — must match the `coefficients_range` length
            // `process_by_scenario_backward` writes.
            let arena_len = count_w * cut_n_state;
            if ws.backward_accum.agg_arena.len() < arena_len {
                ws.backward_accum.agg_arena.resize(arena_len, 0.0_f64);
            }
            ws.backward_accum.staged_cuts_buf.clear();
            let worker_stage_wall_start = Instant::now();
            // Snapshot the cumulative lazy-scoring accumulator; the delta below
            // attributes this stage's scoring to the backward phase (the accumulator
            // is never reset, so a snapshot-delta is the only correct attribution).
            let scoring_seconds_before = ws.backward_accum.dcs_solve.scoring_time_seconds;

            // `compacted` is the trial point's index within the node's FULL routed
            // subset (the cut slot's per-pool position); `local_i` is its row in
            // this worker's own arena. They diverge only when a peer worker owns
            // earlier routed points — `local_i` skips those, `compacted` counts them.
            let mut local_i = 0usize;
            for (compacted, &m) in trial_points.iter().enumerate() {
                if !owns(m) {
                    continue;
                }
                // Reset once per trial point across all its children's binding
                // increments; each child's own cold-head reset + LP load lives inside
                // the per-child loop in `process_by_scenario_backward` (a child loads a
                // different LP, so its warm-start chain must start clean for CLP
                // determinism).
                ws.backward_accum.slot_increments[..pop].fill(0);
                // Call before the push to avoid a simultaneous mutable borrow of
                // `staged_cuts_buf` (push receiver) and `ws` (function argument).
                let arena_offset = local_i * cut_n_state;
                let cut = process_by_scenario_backward(
                    ws,
                    ctx,
                    training_ctx,
                    exchange,
                    fwd_offset,
                    node_visit_offset,
                    iteration,
                    risk_measures,
                    succ,
                    outcomes,
                    &mut basis_slice,
                    &opening_solver,
                    m,
                    compacted,
                    arena_offset,
                )?;
                ws.backward_accum.staged_cuts_buf.push(cut);
                local_i += 1;
            }

            ws.worker_timing_buf.backward_wall_ms +=
                worker_stage_wall_start.elapsed().as_secs_f64() * 1_000.0;
            // The forward and backward folds write the same physical `scoring_ms`
            // field on disjoint phase emissions, so the per-iteration total is
            // recovered by summing across both phases' events.
            ws.worker_timing_buf.scoring_ms += (ws.backward_accum.dcs_solve.scoring_time_seconds
                - scoring_seconds_before)
                * 1_000.0;

            // `drain(..)` leaves capacity intact for the next stage; `w` rides
            // alongside so the merge resolves each cut's `coefficients_range`
            // against that worker's arena.
            Ok((w, ws.backward_accum.staged_cuts_buf.drain(..).collect()))
        })
        .collect()
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
mod tests {
    use super::*;
    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
    use cobre_core::scenario::{InflowModel, SamplingScheme};
    use cobre_solver::{
        Basis, LpSolution, ProfiledSolver, RowBatch, SolverError, SolverInterface,
        SolverStatistics, StageTemplate,
    };

    use crate::{
        context::TrainingContext,
        cut::FutureCostFunction,
        cut_sync::CutSyncBuffers,
        horizon_mode::HorizonMode,
        inflow_method::InflowNonNegativityMethod,
        lp::indexer::StateSpace,
        risk_measure::{BackwardOutcome, RiskMeasure},
        setup::node_graph::{NodeOpenings, NodeRuntime, NodeSuccessor, OpeningSource},
        solver_stats::WORKER_STATS_ENTRY_STRIDE,
        state_exchange::ExchangeBuffers,
        test_support::{
            StageContextFixture, all_enabled_cut_state_layouts, equipment_free_geometry,
            hydro_free_stochastic_context, permissive_state_boxes, state_layout, study_dims,
            trial_state_records,
        },
        trajectory::TrajectoryRecord,
        workspace::{
            BackwardAccumulators, BasisStore, CapturedBasis, ScratchBuffers, SolverWorkspace,
        },
    };

    // ── test stubs ──────────────────────────────────────────────────────────

    struct StubComm;

    impl Communicator for StubComm {
        fn allgatherv<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            recv[..send.len()].copy_from_slice(send);
            Ok(())
        }
        fn allreduce<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            recv[..send.len()].copy_from_slice(send);
            Ok(())
        }
        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
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
        fn abort(&self, code: i32) -> ! {
            std::process::exit(code)
        }
    }

    /// `size() == 2` sibling of [`StubComm`]: `allgatherv` writes only this
    /// rank's own slot (`recv[displs[0]..displs[0] + send.len()]`), the faithful
    /// single-process view of rank 0's contribution to a 2-rank collective.
    struct Rank0Of2;

    impl Communicator for Rank0Of2 {
        fn allgatherv<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _counts: &[usize],
            displs: &[usize],
        ) -> Result<(), CommError> {
            let start = displs[0];
            recv[start..start + send.len()].copy_from_slice(send);
            Ok(())
        }
        fn allreduce<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            recv[..send.len()].copy_from_slice(send);
            Ok(())
        }
        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            Ok(())
        }
        fn barrier(&self) -> Result<(), CommError> {
            Ok(())
        }
        fn rank(&self) -> usize {
            0
        }
        fn size(&self) -> usize {
            2
        }
        fn abort(&self, code: i32) -> ! {
            std::process::exit(code)
        }
    }

    struct MockSolver {
        solution: LpSolution,
        call_count: usize,
        current_num_rows: usize,
        buf_primal: Vec<f64>,
        buf_dual: Vec<f64>,
        buf_reduced_costs: Vec<f64>,
    }

    impl MockSolver {
        fn always_ok(solution: LpSolution) -> Self {
            let base_rows = solution.dual.len();
            let buf_primal = solution.primal.clone();
            let buf_dual = solution.dual.clone();
            let buf_reduced_costs = solution.reduced_costs.clone();
            Self {
                solution,
                call_count: 0,
                current_num_rows: base_rows,
                buf_primal,
                buf_dual,
                buf_reduced_costs,
            }
        }
    }

    impl SolverInterface for MockSolver {
        type Profile = cobre_solver::ActiveProfile;

        fn apply_profile(&mut self, _profile: &cobre_solver::ActiveProfile) {}

        fn name(&self) -> &'static str {
            "mock"
        }
        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
        fn load_model(&mut self, template: &StageTemplate) {
            self.current_num_rows = template.num_rows;
            self.buf_primal = self.solution.primal.clone();
            self.buf_dual = self.solution.dual.clone();
            self.buf_reduced_costs = self.solution.reduced_costs.clone();
            self.buf_dual.resize(self.current_num_rows, 0.0);
        }
        fn add_rows(&mut self, cuts: &RowBatch) {
            self.current_num_rows += cuts.num_rows;
            self.buf_dual.resize(self.current_num_rows, 0.0);
        }
        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
        fn solve(
            &mut self,
            _basis: Option<&Basis>,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            self.call_count += 1;
            Ok(cobre_solver::SolutionView {
                objective: self.solution.objective,
                primal: &self.buf_primal,
                dual: &self.buf_dual,
                reduced_costs: &self.buf_reduced_costs,
                iterations: 0,
                solve_time_seconds: 0.0,
            })
        }
        fn get_basis(&mut self, out: &mut Basis) {
            *out = Basis::new(0, 0);
        }
        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }
        fn statistics_into(&self, out: &mut SolverStatistics) {
            out.copy_from(&SolverStatistics::default());
        }
    }

    // row 0: z-inflow definition (z_inflow[0] = rhs), the only row; col 1 is the free z column.
    fn minimal_template_1_0() -> StageTemplate {
        StageTemplate {
            num_cols: 3,
            num_rows: 1,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, f64::NEG_INFINITY, 0.0],
            col_upper: vec![f64::INFINITY; 3],
            objective: vec![0.0, 0.0, 1.0],
            row_lower: vec![0.0],
            row_upper: vec![0.0],
            n_state: 1,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// A hydro-free, state-free counterpart of [`minimal_template_1_0`] — a
    /// single theta column, no rows.
    fn zero_hydro_template() -> StageTemplate {
        StageTemplate {
            num_cols: 1,
            num_rows: 0,
            num_nz: 0,
            col_starts: vec![0_i32, 0],
            row_indices: vec![],
            values: vec![],
            col_lower: vec![0.0],
            col_upper: vec![f64::INFINITY],
            objective: vec![1.0],
            row_lower: vec![],
            row_upper: vec![],
            n_state: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    fn solution_1_0(objective: f64, dual_storage: f64) -> LpSolution {
        LpSolution {
            objective,
            primal: vec![0.0, 0.0, 0.0],
            dual: vec![dual_storage],
            reduced_costs: vec![0.0; 3],
            iterations: 0,
            solve_time_seconds: 0.0,
        }
    }

    fn single_workspace(
        solver: MockSolver,
        n_state: usize,
        hydro_count: usize,
    ) -> Vec<SolverWorkspace<MockSolver>> {
        use crate::lp::builder::PatchBuffer;
        vec![SolverWorkspace {
            rank: 0,
            worker_id: 0,
            solver: ProfiledSolver::new(solver),
            patch_buf: PatchBuffer::new(
                &crate::test_support::state_layout(hydro_count, 0),
                &[],
                &[],
            ),
            current_state: Vec::with_capacity(n_state),
            scratch: ScratchBuffers {
                inflow_m3s_buf: Vec::new(),
                lag_matrix_buf: Vec::new(),
                par_inflow_buf: Vec::new(),
                eta_floor_buf: Vec::new(),
                zero_targets_buf: Vec::new(),
                ncs_col_upper_buf: Vec::new(),
                ncs_col_lower_buf: Vec::new(),
                ncs_col_indices_buf: Vec::new(),
                ncs_col_lower_active_buf: Vec::new(),
                ncs_col_upper_active_buf: Vec::new(),
                last_ncs_col_start: usize::MAX,
                ncs_col_upper_extract_buf: Vec::new(),
                load_rhs_buf: Vec::new(),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::new(),
                effective_eta_buf: Vec::new(),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
                lag_accumulator: Vec::new(),
                lag_weight_accum: Vec::new(),
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                recon_slot_lookup: Vec::new(),
                trajectory_costs_buf: Vec::new(),
                raw_noise_buf: Vec::new(),
                corr_scratch: Vec::new(),
                current_node_buf: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: WorkerPhaseTimings::default(),
        }]
    }

    fn empty_basis_store(num_scenarios: usize, num_nodes: usize) -> BasisStore {
        BasisStore::new(num_scenarios, num_nodes)
    }

    fn empty_cut_batches(n_stages: usize) -> Vec<RowBatch> {
        (0..n_stages)
            .map(|_| RowBatch {
                num_rows: 0,
                row_starts: Vec::new(),
                col_indices: Vec::new(),
                values: Vec::new(),
                row_lower: Vec::new(),
                row_upper: Vec::new(),
            })
            .collect()
    }

    fn make_stochastic_context(
        n_stages: usize,
        branching_factor: usize,
    ) -> cobre_stochastic::StochasticContext {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::{
            Bus, DeficitSegment, EntityId, SystemBuilder,
            scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
            temporal::{
                Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
                StageStateConfig,
            },
        };
        use cobre_stochastic::context::{
            ClassSchemes, OpeningTreeInputs, build_stochastic_context,
        };
        use std::collections::BTreeMap;

        let bus = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(1),
            name: "H1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
        };
        hydro.declare_mirror_unit_group(EntityId(0));

        let make_stage = |idx: usize| Stage {
            index: idx,
            id: idx as i32,
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
                branching_factor,
                noise_method: NoiseMethod::Saa,
            },
        };

        let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();
        let inflow_models: Vec<_> = (0..n_stages)
            .map(|idx| InflowModel {
                hydro_id: EntityId(1),
                stage_id: idx as i32,
                mean_m3s: 100.0,
                std_m3s: 30.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g1".to_string(),
                    entities: vec![CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(1),
                    }],
                    matrix: vec![vec![1.0]],
                }],
            },
        );
        let correlation = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(correlation)
            .build()
            .unwrap();

        build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap()
    }

    // ── tests ───────────────────────────────────────────────────────────────

    #[test]
    fn backward_pass_state_new_sizes_buffers_correctly() {
        let n_workers_local = 2_usize;
        let n_ranks = 3_usize;
        let bwd_max_openings = 5_usize;
        let real_states_capacity = 10_usize;

        let state = BackwardPassState::new(
            n_workers_local,
            n_ranks,
            bwd_max_openings,
            real_states_capacity,
            7,
            &state_layout(4, 0),
            &HorizonMode::Finite { num_stages: 3 },
        );

        let send_stride = n_workers_local * bwd_max_openings * WORKER_STATS_ENTRY_STRIDE;

        // Empty/zero-sized on construction (grown lazily):
        assert!(state.probabilities_buf.is_empty());
        assert!(state.successor_active_slots_buf.is_empty());
        assert!(state.metadata_sync_buf.is_empty());
        assert!(state.global_increments_buf.is_empty());

        // Pre-sized:
        assert_eq!(state.bwd_stats_send_buf.len(), send_stride);
        assert_eq!(state.bwd_stats_recv_buf.len(), n_ranks * send_stride);
        assert_eq!(state.bwd_stats_counts.len(), n_ranks);
        assert!(state.bwd_stats_counts.iter().all(|&c| c == send_stride));
        assert_eq!(state.bwd_stats_displs.len(), n_ranks);
        assert_eq!(state.bwd_stats_displs[0], 0);
        assert_eq!(state.bwd_stats_displs[1], send_stride);
        assert_eq!(state.bwd_stats_displs[2], 2 * send_stride);
        assert_eq!(
            state.bwd_stats_unpack_buf.len(),
            n_ranks * n_workers_local * bwd_max_openings
        );
    }

    /// Verify that `BackwardPassState::run` on a minimal 2-stage, 1-hydro,
    /// 1-opening study produces a non-empty `BackwardResult` with the expected
    /// cut count and preserves result parity with the equivalent
    /// `run_backward_pass` shim call.
    ///
    /// Setup mirrors `two_stage_system_two_trial_states_generates_two_cuts_at_stage_0`
    /// in `backward.rs`, which documents the expected arithmetic:
    ///
    /// - `MockSolver` returns `objective=100.0`, `dual[0]=-5.0`
    /// - Two trial points with states `[10.0]` and `[20.0]`
    /// - Expected: 2 cuts at stage 0, 0 cuts at stage 1
    #[test]
    fn backward_pass_state_run_preserves_one_stage_scenario_result() {
        let n_stages = 2_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let state = state_layout(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        // Production carries a separate frozen-template buffer alongside
        // `ctx.templates`; mirror that here so `frozen` does not alias the
        // `&templates` borrow held by `ctx`.
        let frozen_templates = templates.clone();
        let n_state = state.n_state;
        let forward_passes = 2_u32;

        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let trial_states = vec![vec![10.0], vec![20.0]];
        let records = trial_state_records(&trial_states, n_stages);
        let mut exchange = ExchangeBuffers::new(&state, trial_states.len(), 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let comm = StubComm;
        let mut workspaces = single_workspace(MockSolver::always_ok(solution), n_state, 1);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let mut cut_batches = empty_cut_batches(n_stages);
        let state_boxes = permissive_state_boxes(n_state, n_stages);
        let geometry = equipment_free_geometry(&vec![0; templates.len()]);
        let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
        let ctx = fixture.ctx();
        let study_dims = study_dims();
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };

        let bwd_max_openings = n_openings;
        // Capture local_count before mutably borrowing exchange inside the struct literal.
        let local_count = exchange.local_count();
        let mut state = BackwardPassState::new(
            1,
            1,
            bwd_max_openings,
            n_state,
            local_count,
            &state,
            &horizon,
        );

        let mut inputs = BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &ctx,
            frozen: &frozen_templates,
            fcf: &mut fcf,
            cut_batches: &mut cut_batches,
            training_ctx: &training_ctx,
            comm: &comm,
            exchange: &mut exchange,
            records: &records,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            event_sender: None,
            risk_measures: &risk_measures,
            cut_activity_tolerance: 0.0,
            iteration: 1,
            local_work: local_count,
            fwd_offset: 0,

            traversal: &Traversal::default(),
            enumerated_state: &EnumeratedForwardScratch::default(),
        };

        let result = state
            .run(&mut inputs)
            .expect("backward pass must not error");

        // 2 trial points × 1 stage with a successor → 2 cuts at stage 0.
        assert_eq!(
            result.cuts_generated, 2,
            "expected 2 cuts from 2 trial points"
        );
        assert_eq!(
            fcf.active_cuts(0).count(),
            2,
            "stage 0 must hold exactly 2 active cuts"
        );
        assert_eq!(
            fcf.active_cuts(1).count(),
            0,
            "stage 1 (last stage) must have no cuts"
        );
        // BackwardResult must be non-empty (stage_stats populated for stage 0).
        assert!(
            !result.stage_stats.is_empty(),
            "stage_stats must be non-empty after a successful backward pass"
        );
    }

    /// `records`, with `leaf_node_ids[m]` written into trial point `m`'s
    /// LAST-stage record (the leaf visit); every earlier stage keeps the
    /// placeholder `0` — mirroring [`trial_state_records`] except for that one
    /// per-trial override.
    fn trial_state_records_with_leaf_ids(
        states: &[Vec<f64>],
        n_stages: usize,
        leaf_node_ids: &[NodeId],
    ) -> Vec<TrajectoryRecord> {
        states
            .iter()
            .zip(leaf_node_ids)
            .flat_map(|(state, &leaf_node_id)| {
                (0..n_stages).map(move |t| TrajectoryRecord {
                    primal: Vec::new(),
                    dual: Vec::new(),
                    stage_cost: 0.0,
                    node_id: if t + 1 == n_stages {
                        leaf_node_id
                    } else {
                        NodeId(0)
                    },
                    state: state.clone(),
                })
            })
            .collect()
    }

    /// Runs `BackwardPassState::run` once over a declared K-fan and returns
    /// `(cuts_generated, root_pool_active_cuts, leaf_pool_active_cuts)`.
    #[allow(clippy::too_many_arguments)]
    fn run_backward_over_k_fan(
        node_graph: &NodeGraph,
        stochastic: &cobre_stochastic::StochasticContext,
        state: &StateSpace,
        templates: &[StageTemplate],
        n_stages: usize,
        records: &[TrajectoryRecord],
    ) -> (usize, usize, usize) {
        let frozen_templates = templates.to_vec();
        let n_state = state.n_state;
        let trial_count = records.len() / n_stages;
        let forward_passes = u32::try_from(trial_count).expect("trial_count fits u32");
        let bwd_max_openings = node_graph.max_successor_outcome_count().max(1);

        let mut fcf = FutureCostFunction::new(
            node_graph.n_pools,
            n_state,
            forward_passes,
            10,
            &vec![0; node_graph.n_pools],
        );
        let mut exchange = ExchangeBuffers::new(state, trial_count, 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let comm = StubComm;
        let mut workspaces = single_workspace(MockSolver::always_ok(solution), n_state, 1);
        // The basis node axis is the canonical NODE count (production sizes it to
        // `node_graph.nodes.len()`), not `n_stages`: a fan's per-child basis keys on
        // sibling node positions that exceed the stage count.
        let mut basis_store = empty_basis_store(exchange.local_count(), node_graph.nodes.len());
        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let mut cut_batches = empty_cut_batches(n_stages);
        let state_boxes = permissive_state_boxes(n_state, n_stages);
        let geometry = equipment_free_geometry(&vec![0; templates.len()]);
        let fixture = StageContextFixture::new(templates, &state_boxes, &geometry);
        let ctx = fixture.ctx();
        let study_dims = study_dims();
        let training_ctx = TrainingContext {
            node_graph,
            horizon: &horizon,
            state,
            cut_state_layouts: &all_enabled_cut_state_layouts(state, n_stages),
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };

        let local_count = exchange.local_count();
        let mut state_machine = BackwardPassState::new(
            1,
            1,
            bwd_max_openings,
            n_state,
            local_count,
            state,
            &horizon,
        );

        let mut inputs = BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &ctx,
            frozen: &frozen_templates,
            fcf: &mut fcf,
            cut_batches: &mut cut_batches,
            training_ctx: &training_ctx,
            comm: &comm,
            exchange: &mut exchange,
            records,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            event_sender: None,
            risk_measures: &risk_measures,
            cut_activity_tolerance: 0.0,
            iteration: 1,
            local_work: local_count,
            fwd_offset: 0,

            traversal: &Traversal::default(),
            enumerated_state: &EnumeratedForwardScratch::default(),
        };

        let result = state_machine
            .run(&mut inputs)
            .expect("backward pass over a declared K-fan must not error");

        // Per-child isolation: the root's backward warm-starts and captures a basis
        // at EACH successor child's OWN node position, not just child 0's. Pricing
        // every child against child 0's basis key would leave the siblings' slots
        // empty — the reified per-child solve populates them all.
        {
            let slice = basis_store.split_workers_mut(1).remove(0);
            for succ in &node_graph.successors[NodePos(0)] {
                assert!(
                    slice.get(0, succ.child).is_some(),
                    "trial point 0's child at node {} must capture its own (m, child node) basis",
                    succ.child
                );
            }
        }

        let root_pool = node_graph.nodes[NodePos(0)].pool_id;
        let leaf_pool = node_graph.nodes[node_graph.successors[NodePos(0)][0].child].pool_id;
        (
            result.cuts_generated,
            fcf.active_cuts(root_pool).count(),
            fcf.active_cuts(leaf_pool).count(),
        )
    }

    /// Trial states exchanged to the backward sweep route to the K-fan's
    /// shared leaf pool the same way whether each trial's stage-1 record
    /// carries the SPECIFIC leaf it visited or one stage-uniform id — the
    /// exchange and cut generation are keyed by trial-point position and the
    /// declared graph's own (leaf-sharing) pool structure, never by the
    /// per-record `node_id`. Two runs differing ONLY in that field must
    /// produce byte-identical cut counts.
    #[test]
    fn backward_pass_state_run_over_k_fan_is_invariant_to_per_trial_leaf_node_id() {
        use cobre_core::HorizonGraph;
        use cobre_core::temporal::{Node, PolicyGraphType, Transition};
        use cobre_io::StageIdResolver;

        use crate::setup::node_graph::build_node_graph;

        fn node(id: i32, stage_id: i32) -> Node {
            Node {
                id,
                stage_id,
                scenario_id: None,
                label: None,
            }
        }
        fn transition(source_id: i32, target_id: i32, probability: f64) -> Transition {
            Transition {
                source_id,
                target_id,
                probability,
                annual_discount_rate_override: None,
            }
        }

        let n_stages = 2_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let study_stage_ids = [0_i32, 1_i32];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let graph = HorizonGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            nodes: vec![node(0, 0), node(1, 1), node(2, 1), node(3, 1)],
            transitions: vec![
                transition(0, 1, 1.0 / 3.0),
                transition(0, 2, 1.0 / 3.0),
                transition(0, 3, 1.0 / 3.0),
            ],
            stage_discount_rate_overrides: std::collections::BTreeMap::new(),
            season_map: None,
        };
        let node_graph = build_node_graph(&graph, n_stages, &resolver, &stochastic)
            .expect("declared K-fan graph must build");
        let leaf_node_ids: Vec<NodeId> = node_graph.successors[NodePos(0)]
            .iter()
            .map(|s| node_graph.node_ids[s.child])
            .collect();
        assert_eq!(leaf_node_ids.len(), 3, "the K-fan must declare 3 leaves");

        let state = state_layout(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let trial_states = vec![vec![10.0], vec![20.0], vec![30.0]];

        let hetero_records =
            trial_state_records_with_leaf_ids(&trial_states, n_stages, &leaf_node_ids);
        let homogeneous_leaf_ids = vec![leaf_node_ids[0]; trial_states.len()];
        let homo_records =
            trial_state_records_with_leaf_ids(&trial_states, n_stages, &homogeneous_leaf_ids);

        let hetero = run_backward_over_k_fan(
            &node_graph,
            &stochastic,
            &state,
            &templates,
            n_stages,
            &hetero_records,
        );
        let homo = run_backward_over_k_fan(
            &node_graph,
            &stochastic,
            &state,
            &templates,
            n_stages,
            &homo_records,
        );

        assert_eq!(
            hetero, homo,
            "(cuts_generated, root_pool_active_cuts, leaf_pool_active_cuts) must be identical \
             whether stage-1 records carry each trial's own visited leaf or one stage-uniform \
             leaf id"
        );
        assert_eq!(
            hetero.0,
            trial_states.len(),
            "3 trial points must each produce one cut at the root pool"
        );
    }

    /// U1 regression: the successor pool must be resolved by the child's NODE
    /// POSITION, never by conflating it with the successor's STAGE index. With
    /// `NodeGraph::nodes`/`successors` typed `TypedVec<NodePos, _>`, indexing
    /// with a `StageIdx` is now a COMPILE ERROR — `ng.nodes[successor_stage]`
    /// does not type-check — so the historical runtime misindex this test used
    /// to construct by hand is unrepresentable without an explicit, visible
    /// re-wrap. What this test still pins: on a genuine fan, the child's own
    /// `NodePos` and a `NodePos` reconstructed from the successor stage's raw
    /// value resolve to DIFFERENT, non-interchangeable pools — proving the two
    /// index spaces are not numeric synonyms on this fixture, so the type-level
    /// guarantee is not vacuously true here. The reified `SuccessorEntry.pool_id`
    /// carries `nodes[child].pool_id`; the DCS arm's end-to-end regression guard
    /// is `dcs_arm_generated_fan_value_matches_oracle`.
    #[test]
    fn successor_pool_resolves_by_node_position_not_stage_index() {
        use cobre_core::HorizonGraph;
        use cobre_core::temporal::{Node, PolicyGraphType, Transition};
        use cobre_io::StageIdResolver;

        use crate::setup::node_graph::build_node_graph;

        fn node(id: i32, stage_id: i32) -> Node {
            Node {
                id,
                stage_id,
                scenario_id: None,
                label: None,
            }
        }
        fn transition(source_id: i32, target_id: i32, probability: f64) -> Transition {
            Transition {
                source_id,
                target_id,
                probability,
                annual_discount_rate_override: None,
            }
        }

        let stochastic = make_stochastic_context(3, 1);
        let study_stage_ids = [0_i32, 1, 2];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        // Root (0) → two interior siblings (1, 2 at stage 1), each with two leaves
        // (3, 4 and 5, 6 at stage 2). Interior siblings own distinct pools; leaves
        // share one.
        let graph = HorizonGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            nodes: vec![
                node(0, 0),
                node(1, 1),
                node(2, 1),
                node(3, 2),
                node(4, 2),
                node(5, 2),
                node(6, 2),
            ],
            transitions: vec![
                transition(0, 1, 0.5),
                transition(0, 2, 0.5),
                transition(1, 3, 0.5),
                transition(1, 4, 0.5),
                transition(2, 5, 0.5),
                transition(2, 6, 0.5),
            ],
            stage_discount_rate_overrides: std::collections::BTreeMap::new(),
            season_map: None,
        };
        let ng =
            build_node_graph(&graph, 3, &resolver, &stochastic).expect("3-stage fan must build");

        // Interior node 1 (position 1, stage 1); its successor is a leaf.
        let interior_pos = NodePos(1);
        let child = ng.successors[interior_pos][0].child;
        let successor_stage = ng.nodes[interior_pos].stage.next();
        // The wrong-but-once-compiling alternative, now reachable only through
        // an explicit re-wrap: `ng.nodes[successor_stage]` itself is a compile
        // error (a `StageIdx` cannot index a `NodePos`-typed array), so a
        // maintainer must deliberately reconstruct a `NodePos` from the raw
        // stage value to even express the historical bug.
        let stage_misindexed_as_node_pos = NodePos(successor_stage.0);

        let pool_by_node_position = ng.nodes[child].pool_id;
        let pool_by_stage_misindex = ng.nodes[stage_misindexed_as_node_pos].pool_id;

        assert_ne!(
            pool_by_node_position, pool_by_stage_misindex,
            "the fan must genuinely exercise U1: a NodePos reconstructed from the successor \
             STAGE ({successor_stage}) must resolve a DIFFERENT pool ({pool_by_stage_misindex}) \
             than the child's own node position ({child} → pool {pool_by_node_position})"
        );
        // The child is a shared-leaf-pool leaf; the misindexed node is a cut-generating
        // interior sibling — a genuinely wrong pool with its own cut projection.
        assert!(
            ng.successors[child].is_empty(),
            "the true successor is a terminal leaf"
        );
        assert!(
            !ng.successors[stage_misindexed_as_node_pos].is_empty(),
            "the stage-misindexed node is a cut-generating interior sibling"
        );
    }

    /// `records` with `stage1_ids[m]` written into trial point `m`'s STAGE-1
    /// record — the interior-node visit a two-interior-node level routes on.
    /// Stage 0 carries the root id `0`; every other stage a placeholder `0`
    /// (routing never reads a single-node level's records).
    fn trial_state_records_with_stage1_ids(
        states: &[Vec<f64>],
        n_stages: usize,
        stage1_ids: &[NodeId],
    ) -> Vec<TrajectoryRecord> {
        states
            .iter()
            .zip(stage1_ids)
            .flat_map(|(state, &stage1_id)| {
                (0..n_stages).map(move |t| TrajectoryRecord {
                    primal: Vec::new(),
                    dual: Vec::new(),
                    stage_cost: 0.0,
                    node_id: if t == 1 { stage1_id } else { NodeId(0) },
                    state: state.clone(),
                })
            })
            .collect()
    }

    /// `n_workers` `MockSolver` workspaces, worker ids `0..n_workers`, for the
    /// multi-worker routing regression (`split_workers_mut` splits the basis store
    /// across exactly these).
    fn workspaces_n(count: usize, n_state: usize) -> Vec<SolverWorkspace<MockSolver>> {
        (0..count)
            .map(|i| {
                let mut ws =
                    single_workspace(MockSolver::always_ok(solution_1_0(100.0, -5.0)), n_state, 1);
                let mut w = ws.remove(0);
                w.worker_id = i as i32;
                w
            })
            .collect()
    }

    /// Runs `BackwardPassState::run` once over a declared 3-stage binary tree
    /// (root → 2 interior nodes → 4 leaves) under `scheduler` with `n_workers`
    /// worker threads, returning `(cuts_generated, per_pool_active_cuts)` with the
    /// second entry indexed by pool id.
    #[allow(clippy::too_many_arguments)]
    fn run_backward_over_binary_tree(
        node_graph: &NodeGraph,
        stochastic: &cobre_stochastic::StochasticContext,
        state: &StateSpace,
        templates: &[StageTemplate],
        n_stages: usize,
        records: &[TrajectoryRecord],
        scheduler: BackwardScheduler,
        n_workers: usize,
    ) -> (usize, Vec<usize>) {
        // Frozen overlay is per POOL: pool `p`'s base is `templates[pool_stage[p]]`
        // (all templates identical here, so this just spans every pool, including
        // the leaf pool at position >= n_stages).
        let frozen_templates: Vec<StageTemplate> = (0..node_graph.n_pools)
            .map(|p| templates[node_graph.pool_stage[p].0].clone())
            .collect();
        let n_state = state.n_state;
        let trial_count = records.len() / n_stages;
        let forward_passes = u32::try_from(trial_count).expect("trial_count fits u32");
        let bwd_max_openings = node_graph
            .successors
            .iter()
            .map(|succs| {
                succs
                    .iter()
                    .map(|s| node_graph.nodes[s.child].openings.len)
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0)
            .max(1);

        let mut fcf = FutureCostFunction::new(
            node_graph.n_pools,
            n_state,
            forward_passes,
            10,
            &vec![0; node_graph.n_pools],
        );
        let mut exchange = ExchangeBuffers::new(state, trial_count, 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let comm = StubComm;
        let mut workspaces = workspaces_n(n_workers, n_state);
        // This fixture's binary tree has more nodes than stages, so
        // `node_graph.nodes.len()` — not `n_stages` — sizes the store.
        let mut basis_store = empty_basis_store(exchange.local_count(), node_graph.nodes.len());
        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        // Cut-batch scratch is pool-indexed (backward writes `cut_batches[successor_pool_id]`).
        let mut cut_batches = empty_cut_batches(node_graph.n_pools);
        let state_boxes = permissive_state_boxes(n_state, n_stages);
        let geometry = equipment_free_geometry(&vec![0; templates.len()]);
        let fixture = StageContextFixture::new(templates, &state_boxes, &geometry);
        let ctx = fixture.ctx();
        let study_dims = study_dims();
        let training_ctx = TrainingContext {
            node_graph,
            horizon: &horizon,
            state,
            // `cut_state_layouts` is keyed by POOL id (a cut-generating node
            // owns its own pool), sized by `n_pools` — this fixture's binary
            // tree makes `n_pools` diverge from `n_stages`.
            cut_state_layouts: &all_enabled_cut_state_layouts(state, node_graph.n_pools),
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };

        let local_count = exchange.local_count();
        let mut state_machine = BackwardPassState::new(
            n_workers,
            1,
            bwd_max_openings,
            n_state,
            local_count,
            state,
            &horizon,
        );
        state_machine.set_scheduler(scheduler);

        let mut inputs = BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &ctx,
            frozen: &frozen_templates,
            fcf: &mut fcf,
            cut_batches: &mut cut_batches,
            training_ctx: &training_ctx,
            comm: &comm,
            exchange: &mut exchange,
            records,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            event_sender: None,
            risk_measures: &risk_measures,
            cut_activity_tolerance: 0.0,
            iteration: 1,
            local_work: local_count,
            fwd_offset: 0,

            traversal: &Traversal::default(),
            enumerated_state: &EnumeratedForwardScratch::default(),
        };

        let result = state_machine
            .run(&mut inputs)
            .expect("backward pass over a declared binary tree must not error");

        let per_pool: Vec<usize> = (0..node_graph.n_pools)
            .map(|p| fcf.active_cuts(p).count())
            .collect();
        (result.cuts_generated, per_pool)
    }

    // ── enumerated backward: node-native fork ──────────────────────────────

    /// A hand-built small trunk+fan graph: root (stage 0, id 0) → ONE trunk
    /// node (stage 1, id 1) → THREE leaves (stage 2, ids 2..4, one shared
    /// pool). Two non-leaf (cut-generating) nodes: root and the trunk node.
    fn trunk_fan_graph() -> (NodeGraph, cobre_stochastic::StochasticContext) {
        use crate::setup::node_graph::build_node_graph;
        use cobre_core::HorizonGraph;
        use cobre_core::temporal::{Node, PolicyGraphType, Transition};
        use cobre_io::StageIdResolver;

        fn node(id: i32, stage_id: i32) -> Node {
            Node {
                id,
                stage_id,
                scenario_id: None,
                label: None,
            }
        }
        fn transition(source_id: i32, target_id: i32, probability: f64) -> Transition {
            Transition {
                source_id,
                target_id,
                probability,
                annual_discount_rate_override: None,
            }
        }

        let n_stages = 3_usize;
        let stochastic = make_stochastic_context(n_stages, 3);
        let study_stage_ids = [0_i32, 1, 2];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let graph = HorizonGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            nodes: vec![node(0, 0), node(1, 1), node(2, 2), node(3, 2), node(4, 2)],
            transitions: vec![
                transition(0, 1, 1.0),
                transition(1, 2, 1.0 / 3.0),
                transition(1, 3, 1.0 / 3.0),
                transition(1, 4, 1.0 / 3.0),
            ],
            stage_discount_rate_overrides: std::collections::BTreeMap::new(),
            season_map: None,
        };
        let node_graph = build_node_graph(&graph, n_stages, &resolver, &stochastic)
            .expect("declared trunk+fan graph must build");
        (node_graph, stochastic)
    }

    /// A hand-built trunk+fan graph mirroring [`trunk_fan_graph`], except the
    /// trunk's two terminal children are ONE External leaf (id 2, scenario
    /// column 0 — `is_external_terminal_leaf` eligible) and ONE Generated leaf
    /// (id 3 — never eligible), so a single node's backward exercises both the
    /// fusion-eligible and the exhaustive-solve branch side by side.
    ///
    /// Hydro-free: `run_enumerated_backward_over_graph`'s harness declares no
    /// `external_inflow_library`, and the External leaf's declared column must
    /// resolve against an empty inflow noise class to avoid it.
    fn mixed_terminal_fan_graph() -> (NodeGraph, cobre_stochastic::StochasticContext) {
        use crate::setup::node_graph::build_node_graph;
        use cobre_core::HorizonGraph;
        use cobre_core::temporal::{Node, PolicyGraphType, Transition};
        use cobre_io::StageIdResolver;

        fn node(id: i32, stage_id: i32, scenario_id: Option<i32>) -> Node {
            Node {
                id,
                stage_id,
                scenario_id,
                label: None,
            }
        }
        fn transition(source_id: i32, target_id: i32, probability: f64) -> Transition {
            Transition {
                source_id,
                target_id,
                probability,
                annual_discount_rate_override: None,
            }
        }

        let n_stages = 3_usize;
        let stochastic = hydro_free_stochastic_context(n_stages, 3);
        let study_stage_ids = [0_i32, 1, 2];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let graph = HorizonGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            nodes: vec![
                node(0, 0, None),
                node(1, 1, None),
                node(2, 2, Some(0)),
                node(3, 2, None),
            ],
            transitions: vec![
                transition(0, 1, 1.0),
                transition(1, 2, 0.5),
                transition(1, 3, 0.5),
            ],
            stage_discount_rate_overrides: std::collections::BTreeMap::new(),
            season_map: None,
        };
        let node_graph = build_node_graph(&graph, n_stages, &resolver, &stochastic)
            .expect("declared mixed terminal fan graph must build");
        (node_graph, stochastic)
    }

    /// Run the enumerated backward over `node_graph` with the given resolved
    /// `traversal`/`enumerated_state`, returning the raw result (so both the
    /// happy-path and the validation-error tests can share this setup) and
    /// the per-pool active-cut counts (computed regardless of `Ok`/`Err`,
    /// since `fcf` retains whatever cuts were appended before any error).
    fn run_enumerated_backward_over_graph<C: Communicator>(
        node_graph: &NodeGraph,
        stochastic: &cobre_stochastic::StochasticContext,
        traversal: &Traversal,
        enumerated_state: &EnumeratedForwardScratch,
        comm: &C,
    ) -> (Result<BackwardResult, SddpError>, Vec<usize>) {
        let n_stages = 3_usize;
        let n_hydros = stochastic.n_hydros();
        let state = state_layout(n_hydros, 0);
        let template = if n_hydros == 0 {
            zero_hydro_template()
        } else {
            minimal_template_1_0()
        };
        let templates = vec![template; n_stages];
        let frozen_templates: Vec<StageTemplate> = (0..node_graph.n_pools)
            .map(|p| templates[node_graph.pool_stage[p].0].clone())
            .collect();
        let n_state = state.n_state;

        let mut fcf = FutureCostFunction::new(
            node_graph.n_pools,
            n_state,
            1,
            10,
            &vec![0; node_graph.n_pools],
        );
        let mut exchange = ExchangeBuffers::new(&state, 1, 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let mut workspaces = single_workspace(MockSolver::always_ok(solution), n_state, n_hydros);
        let mut basis_store = empty_basis_store(1, node_graph.nodes.len());
        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, comm.size(), comm.size());
        let mut cut_batches = empty_cut_batches(node_graph.n_pools);
        let state_boxes = permissive_state_boxes(n_state, n_stages);
        let geometry = equipment_free_geometry(&vec![0; templates.len()]);
        let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
        let ctx = fixture.ctx();
        let study_dims = study_dims();
        let training_ctx = TrainingContext {
            node_graph,
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, node_graph.n_pools),
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };

        let bwd_max_openings = node_graph
            .successors
            .iter()
            .map(|succs| {
                succs
                    .iter()
                    .map(|s| node_graph.nodes[s.child].openings.len)
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0)
            .max(1);
        let mut state_machine =
            BackwardPassState::new(1, 1, bwd_max_openings, n_state, 1, &state, &horizon);

        let mut inputs = BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &ctx,
            frozen: &frozen_templates,
            fcf: &mut fcf,
            cut_batches: &mut cut_batches,
            training_ctx: &training_ctx,
            comm,
            exchange: &mut exchange,
            records: &[],
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            event_sender: None,
            risk_measures: &risk_measures,
            cut_activity_tolerance: 0.0,
            iteration: 1,
            local_work: 0,
            fwd_offset: 0,
            traversal,
            enumerated_state,
        };

        let result = state_machine.run(&mut inputs);
        let per_pool: Vec<usize> = (0..node_graph.n_pools)
            .map(|p| fcf.active_cuts(p).count())
            .collect();
        (result, per_pool)
    }

    /// Over a hand-built small trunk+fan graph, `run_enumerated_backward`
    /// (dispatched via `BackwardPassState::run` under `Traversal::Enumerated`)
    /// appends exactly ONE cut per non-leaf node — never one per trajectory.
    #[test]
    fn enumerated_backward_appends_one_cut_per_nonleaf_node() {
        let (node_graph, stochastic) = trunk_fan_graph();
        let root = NodePos(0);
        let trunk = NodePos(1);
        let root_pool = node_graph.nodes[root].pool_id;
        let trunk_pool = node_graph.nodes[trunk].pool_id;
        assert_ne!(root_pool, trunk_pool, "root and trunk own distinct pools");

        let traversal = Traversal::resolve(&node_graph, true, 1);
        let mut enumerated_state = EnumeratedForwardScratch::default();
        enumerated_state.set_out_state_for_test(root, node_graph.nodes.len(), &[10.0]);
        enumerated_state.set_out_state_for_test(trunk, node_graph.nodes.len(), &[20.0]);

        let (result, per_pool) = run_enumerated_backward_over_graph(
            &node_graph,
            &stochastic,
            &traversal,
            &enumerated_state,
            &StubComm,
        );
        let result = result.expect("enumerated backward over a trunk+fan graph must not error");

        assert_eq!(
            result.cuts_generated, 2,
            "exactly one cut per non-leaf node (root, trunk) — never one per trajectory"
        );
        assert_eq!(
            per_pool[root_pool], 1,
            "root's pool receives exactly one cut"
        );
        assert_eq!(
            per_pool[trunk_pool], 1,
            "trunk's pool receives exactly one cut"
        );
    }

    /// World >= 2 cut-exchange hygiene: the enumerated backward appends the
    /// replicated cut on every rank via `run_backward_node_replicated`, so a
    /// 2-rank run appends exactly the world=1 count — one cut per non-leaf pool,
    /// never multiplied by rank count. Guards against a naive future cut exchange
    /// that would re-insert the replicated cut.
    #[test]
    fn enumerated_backward_two_ranks_matches_world_one_cut_count() {
        let (node_graph, stochastic) = trunk_fan_graph();
        let root = NodePos(0);
        let trunk = NodePos(1);
        let root_pool = node_graph.nodes[root].pool_id;
        let trunk_pool = node_graph.nodes[trunk].pool_id;

        let traversal = Traversal::resolve(&node_graph, true, 1);
        let mut enumerated_state = EnumeratedForwardScratch::default();
        enumerated_state.set_out_state_for_test(root, node_graph.nodes.len(), &[10.0]);
        enumerated_state.set_out_state_for_test(trunk, node_graph.nodes.len(), &[20.0]);

        let (world1, per_pool_1) = run_enumerated_backward_over_graph(
            &node_graph,
            &stochastic,
            &traversal,
            &enumerated_state,
            &StubComm,
        );
        let (world2, per_pool_2) = run_enumerated_backward_over_graph(
            &node_graph,
            &stochastic,
            &traversal,
            &enumerated_state,
            &Rank0Of2,
        );
        let world1 = world1.expect("world=1 enumerated backward must not error");
        let world2 = world2.expect("world=2 enumerated backward must not error");

        assert_eq!(
            world2.cuts_generated, world1.cuts_generated,
            "2-rank cuts_generated equals the world=1 count — no rank-multiplication"
        );
        assert_eq!(
            per_pool_2[root_pool], per_pool_1[root_pool],
            "root pool: one cut on 2 ranks, exactly as on 1"
        );
        assert_eq!(
            per_pool_2[trunk_pool], per_pool_1[trunk_pool],
            "trunk pool: one cut on 2 ranks, exactly as on 1"
        );
        assert_eq!(per_pool_2[trunk_pool], 1, "and that count is exactly one");
    }

    /// A predicate-eligible External terminal leaf with NO captured fused
    /// slice (`EnumeratedForwardScratch::default()` — the forward capture
    /// never ran) falls back to solving the child directly instead of
    /// unwrapping. The Generated sibling is unaffected: it always solves. The
    /// run must complete with `Ok`, never panic.
    #[test]
    fn enumerated_backward_falls_back_to_solving_when_fused_slice_is_absent() {
        let (node_graph, stochastic) = mixed_terminal_fan_graph();
        let root = NodePos(0);
        let trunk = NodePos(1);
        let external_leaf = NodePos(2);
        let generated_leaf = NodePos(3);
        let num_stages = 3;

        assert!(
            node_graph.is_external_terminal_leaf(external_leaf, num_stages),
            "node 2 must be a fusion-eligible External terminal leaf"
        );
        assert!(
            !node_graph.is_external_terminal_leaf(generated_leaf, num_stages),
            "node 3 (Generated) must never be fusion-eligible"
        );

        let traversal = Traversal::resolve(&node_graph, true, 1);
        let mut enumerated_state = EnumeratedForwardScratch::default();
        enumerated_state.set_out_state_for_test(root, node_graph.nodes.len(), &[]);
        enumerated_state.set_out_state_for_test(trunk, node_graph.nodes.len(), &[]);
        assert!(
            enumerated_state
                .fused_terminal_slice(external_leaf)
                .is_none(),
            "power: the eligible leaf's fused slice must genuinely be absent"
        );

        let (result, per_pool) = run_enumerated_backward_over_graph(
            &node_graph,
            &stochastic,
            &traversal,
            &enumerated_state,
            &StubComm,
        );
        let result = result.expect("a missing fused slice must fall back to solving, never error");

        let trunk_pool = node_graph.nodes[trunk].pool_id;
        assert_eq!(
            per_pool[trunk_pool], 1,
            "the trunk's cut is still generated despite the fallback solve"
        );
        assert!(
            result.cuts_generated >= 2,
            "both cut-generating nodes (root, trunk) still append a cut"
        );
    }

    /// A successor whose `EnumeratedPlan::parent` does not record the
    /// cut-generating node as its parent is a named `SddpError::Validation`,
    /// never a silent default or a panic.
    #[test]
    fn enumerated_backward_missing_parent_is_validation_error() {
        use crate::setup::node_graph::{EnumeratedForwardPaths, EnumeratedPlan};

        let (node_graph, stochastic) = trunk_fan_graph();
        let root = NodePos(0);
        let trunk = NodePos(1);

        // Deliberately corrupt: the trunk node's parent should be `Some(root)`;
        // recording `None` instead makes root's own successor check fail.
        // `paths` is irrelevant here — `run_enumerated_backward` never reads it.
        let bad_plan = EnumeratedPlan::from_parts(
            EnumeratedForwardPaths {
                leaf: Vec::new(),
                weight: Vec::new(),
            },
            vec![None, None, Some(trunk), Some(trunk), Some(trunk)].into(),
        );
        let traversal = Traversal::Enumerated(bad_plan);

        let mut enumerated_state = EnumeratedForwardScratch::default();
        enumerated_state.set_out_state_for_test(root, node_graph.nodes.len(), &[10.0]);
        enumerated_state.set_out_state_for_test(trunk, node_graph.nodes.len(), &[20.0]);

        let (result, _) = run_enumerated_backward_over_graph(
            &node_graph,
            &stochastic,
            &traversal,
            &enumerated_state,
            &StubComm,
        );
        match result {
            Err(SddpError::Validation(msg)) => {
                assert!(
                    msg.contains("EnumeratedPlan parent"),
                    "message names the malformed-plan condition: {msg}"
                );
            }
            other => panic!("expected SddpError::Validation, got {other:?}"),
        }
    }

    /// OQ-3 future-recombination insurance, driven on the assert HELPER directly
    /// (a real enumerated study cannot construct a >= 2-predecessor node —
    /// `StudySetup::new` rejects recombination). A node with TWO predecessors
    /// whose persisted `out_states` are `to_bits`-identical makes its incoming
    /// state ambiguous under structural keying: the helper fires.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "bit-identical persisted states")]
    fn predecessor_states_distinct_fires_on_two_same_state_predecessors() {
        // Node 2's predecessors are nodes 0 and 1, both carrying [7.0].
        let predecessors = vec![Vec::new(), Vec::new(), vec![NodePos(0), NodePos(1)]];
        let mut arena = EnumeratedForwardScratch::default();
        arena.set_out_state_for_test(NodePos(0), 3, &[7.0]);
        arena.set_out_state_for_test(NodePos(1), 3, &[7.0]);
        assert_predecessor_states_distinct(&predecessors, &arena);
    }

    /// The false positive the old wrong-axis assert produced under CLP: two
    /// DISTINCT nodes (a deterministic trunk's parent and child) settle on the
    /// SAME persisted state. Each is a single predecessor of a different node,
    /// so no node has >= 2 predecessors — the helper is vacuous and does NOT fire.
    #[cfg(debug_assertions)]
    #[test]
    fn predecessor_states_distinct_ignores_parent_child_same_state() {
        // Node 1's only predecessor is node 0; both carry the same state.
        let predecessors = vec![Vec::new(), vec![NodePos(0)]];
        let mut arena = EnumeratedForwardScratch::default();
        arena.set_out_state_for_test(NodePos(0), 2, &[10.0]);
        arena.set_out_state_for_test(NodePos(1), 2, &[10.0]);
        assert_predecessor_states_distinct(&predecessors, &arena);
    }

    /// A legitimate recombination node whose two predecessors carry DISTINCT
    /// states does NOT fire — the helper flags ambiguous (identical) predecessor
    /// states, never recombination itself.
    #[cfg(debug_assertions)]
    #[test]
    fn predecessor_states_distinct_allows_distinct_predecessor_states() {
        let predecessors = vec![Vec::new(), Vec::new(), vec![NodePos(0), NodePos(1)]];
        let mut arena = EnumeratedForwardScratch::default();
        arena.set_out_state_for_test(NodePos(0), 3, &[7.0]);
        arena.set_out_state_for_test(NodePos(1), 3, &[8.0]);
        assert_predecessor_states_distinct(&predecessors, &arena);
    }

    /// Cut-generation routing at a two-INTERIOR-node level: each cut-generating
    /// node anchors its cut ONLY at the trial states routed to its own pool. Two
    /// forwards that partition trajectories differently across the siblings must
    /// produce DIFFERENT per-pool cut sets — a trajectory switching sibling moves
    /// its cut from one pool to the other. This is the mirror of
    /// `backward_pass_state_run_over_k_fan_is_invariant_to_per_trial_leaf_node_id`
    /// (where a single cut-generating node makes routing a no-op); under per-stage
    /// aggregation both partitions would wrongly give each pool `F` cuts. Both
    /// backward schedulers must route identically.
    #[test]
    fn backward_pass_state_routes_trial_states_to_the_visited_nodes_pool() {
        use cobre_core::HorizonGraph;
        use cobre_core::temporal::{Node, PolicyGraphType, Transition};
        use cobre_io::StageIdResolver;

        use crate::setup::node_graph::build_node_graph;

        fn node(id: i32, stage_id: i32) -> Node {
            Node {
                id,
                stage_id,
                scenario_id: None,
                label: None,
            }
        }
        fn transition(source_id: i32, target_id: i32, probability: f64) -> Transition {
            Transition {
                source_id,
                target_id,
                probability,
                annual_discount_rate_override: None,
            }
        }

        let n_stages = 3_usize;
        let branching = 2_usize;
        let stochastic = make_stochastic_context(n_stages, branching);
        let study_stage_ids = [0_i32, 1, 2];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        // Root (stage 0) → two INTERIOR nodes (stage 1, distinct pools) → four
        // leaves (stage 2, one shared pool). backward_cut_levels == [[1, 2], [0]].
        let graph = HorizonGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            nodes: vec![
                node(0, 0),
                node(1, 1),
                node(2, 1),
                node(3, 2),
                node(4, 2),
                node(5, 2),
                node(6, 2),
            ],
            transitions: vec![
                transition(0, 1, 0.5),
                transition(0, 2, 0.5),
                transition(1, 3, 0.5),
                transition(1, 4, 0.5),
                transition(2, 5, 0.5),
                transition(2, 6, 0.5),
            ],
            stage_discount_rate_overrides: std::collections::BTreeMap::new(),
            season_map: None,
        };
        let node_graph = build_node_graph(&graph, n_stages, &resolver, &stochastic)
            .expect("declared binary-tree graph must build");

        let root_pool = node_graph.nodes[NodePos(0)].pool_id;
        let pool1 = node_graph.nodes[NodePos(1)].pool_id;
        let pool2 = node_graph.nodes[NodePos(2)].pool_id;
        assert_ne!(
            pool1, pool2,
            "two interior siblings at one level must own distinct pools"
        );

        let state = state_layout(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let trial_states = vec![vec![10.0], vec![20.0], vec![30.0], vec![40.0]];
        let f = trial_states.len();

        // Config A: 3 trajectories visit node 1, 1 visits node 2 (at stage 1).
        let ids_a = [NodeId(1), NodeId(1), NodeId(1), NodeId(2)];
        // Config B: the mirror partition — 1 visits node 1, 3 visit node 2.
        let ids_b = [NodeId(1), NodeId(2), NodeId(2), NodeId(2)];

        // The by-scenario path exercises routing on a multi-successor,
        // multi-pool level. The by-node scheduler is not run here: it indexes a
        // single successor's `solve_order` and does not yet handle a node's
        // flattened multi-successor opening set (orthogonal to routing; its
        // routing threading is covered on chains by the by-node determinism
        // gates in `tests/mpi_wire.rs`).
        let scheduler = BackwardScheduler::ByScenario {};
        let records_a = trial_state_records_with_stage1_ids(&trial_states, n_stages, &ids_a);
        let records_b = trial_state_records_with_stage1_ids(&trial_states, n_stages, &ids_b);

        // Both worker counts. `n_workers == 2` is the regression: a node's routed
        // subset is scattered across the global scenario axis, so a worker must be
        // handed only the routed points inside its own contiguous basis window —
        // splitting `trial_points.len()` evenly instead indexes a routed `m` out of
        // another worker's basis slice (a deterministic OOB panic). Routing is
        // worker-count invariant, so the per-pool counts match across both legs.
        for n_workers in [1_usize, 2] {
            let (gen_a, pools_a) = run_backward_over_binary_tree(
                &node_graph,
                &stochastic,
                &state,
                &templates,
                n_stages,
                &records_a,
                scheduler,
                n_workers,
            );
            let (_gen_b, pools_b) = run_backward_over_binary_tree(
                &node_graph,
                &stochastic,
                &state,
                &templates,
                n_stages,
                &records_b,
                scheduler,
                n_workers,
            );

            assert_eq!(
                pools_a[root_pool], f,
                "the root (single-node level) anchors a cut at every trajectory's stage-0 state"
            );
            assert_eq!(
                pools_a[pool1], 3,
                "node 1's pool: the 3 trajectories that visited it"
            );
            assert_eq!(
                pools_a[pool2], 1,
                "node 2's pool: the 1 trajectory that visited it"
            );
            assert_eq!(
                pools_a[pool1] + pools_a[pool2],
                f,
                "Σ over the level's pools of their trial counts must equal F"
            );
            assert_eq!(
                gen_a,
                2 * f,
                "F cuts at the two-interior-node level plus F at the root"
            );

            assert_eq!(pools_b[root_pool], f);
            assert_eq!(pools_b[pool1], 1);
            assert_eq!(pools_b[pool2], 3);

            assert_ne!(
                pools_a[pool1], pools_b[pool1],
                "routing is load-bearing: repartitioning trajectories across the siblings must \
                 move cuts between the sibling pools (per-stage aggregation would give each F)"
            );
        }
    }

    /// A profile installed via `set_profile` before `run()` is the one
    /// `ProfiledSolver::current_profile()` reports afterwards — the resolved
    /// profile reaches the solver, not just the stored default.
    #[test]
    fn backward_pass_state_set_profile_reaches_current_profile_after_run() {
        let n_stages = 2_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let state_layout_fixture = state_layout(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let frozen_templates = templates.clone();
        let n_state = state_layout_fixture.n_state;
        let forward_passes = 2_u32;

        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let trial_states = vec![vec![10.0], vec![20.0]];
        let records = trial_state_records(&trial_states, n_stages);
        let mut exchange = ExchangeBuffers::new(&state_layout_fixture, trial_states.len(), 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let comm = StubComm;
        let mut workspaces = single_workspace(MockSolver::always_ok(solution), n_state, 1);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let mut cut_batches = empty_cut_batches(n_stages);
        let state_boxes = permissive_state_boxes(n_state, n_stages);
        let geometry = equipment_free_geometry(&vec![0; templates.len()]);
        let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
        let ctx = fixture.ctx();
        let study_dims = study_dims();
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state_layout_fixture,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state_layout_fixture, n_stages),
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };

        let bwd_max_openings = n_openings;
        let local_count = exchange.local_count();
        let mut bwd_state = BackwardPassState::new(
            1,
            1,
            bwd_max_openings,
            n_state,
            local_count,
            &state_layout_fixture,
            &horizon,
        );
        let resolved =
            Phase::Backward.resolve_profile(Some(&cobre_io::config::PhaseSolverProfileConfig {
                dual_edge_weight: Some(cobre_io::config::DualEdgeWeight::SteepestEdge),
                scale: Some(cobre_io::config::ScaleStrategy::SolverScaling),
                price: Some(cobre_io::config::PriceStrategy::Row),
                primal_feasibility_tolerance: Some(1e-7),
                dual_feasibility_tolerance: None,
                presolve: None,
                simplex_update_limit: None,
                cost_perturbation: None,
                refactor_error_tolerance: None,
                factor_pivot_threshold: None,
                use_warm_start: None,
                steepest_edge_devex_fallback_threshold: None,
            }));
        bwd_state.set_profile(resolved);

        let mut inputs = BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &ctx,
            frozen: &frozen_templates,
            fcf: &mut fcf,
            cut_batches: &mut cut_batches,
            training_ctx: &training_ctx,
            comm: &comm,
            exchange: &mut exchange,
            records: &records,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            event_sender: None,
            risk_measures: &risk_measures,
            cut_activity_tolerance: 0.0,
            iteration: 1,
            local_work: local_count,
            fwd_offset: 0,

            traversal: &Traversal::default(),
            enumerated_state: &EnumeratedForwardScratch::default(),
        };

        let _ = bwd_state
            .run(&mut inputs)
            .expect("backward pass must not error");

        assert_eq!(
            inputs.workspaces[0].solver.current_profile(),
            &resolved,
            "the profile installed via set_profile must be the one stored on \
             current_profile after run()"
        );
    }

    /// Verify that `state_duals_buf` on the per-worker `BackwardAccumulators`
    /// is correctly sized after the backward pass completes.
    ///
    /// After `BackwardPassState::run` returns, the single worker's
    /// `backward_accum.state_duals_buf` must hold exactly `state.n_state`
    /// entries — the last opening's unscaled duals that were written during
    /// the final trial-point/opening iteration.
    ///
    /// This guards against buffer re-use bugs where the fill loop writes the
    /// wrong number of entries across consecutive openings.
    #[test]
    fn backward_pass_state_duals_buf_len_equals_n_state_after_run() {
        let n_stages = 2_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let state = state_layout(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let frozen_templates = templates.clone();
        let n_state = state.n_state;
        let forward_passes = 2_u32;

        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let trial_states = vec![vec![10.0], vec![20.0]];
        let records = trial_state_records(&trial_states, n_stages);
        let mut exchange = ExchangeBuffers::new(&state, trial_states.len(), 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let comm = StubComm;
        let mut workspaces = single_workspace(MockSolver::always_ok(solution), n_state, 1);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let mut cut_batches = empty_cut_batches(n_stages);
        let state_boxes = permissive_state_boxes(n_state, n_stages);
        let geometry = equipment_free_geometry(&vec![0; templates.len()]);
        let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
        let ctx = fixture.ctx();
        let study_dims = study_dims();
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };

        let bwd_max_openings = n_openings;
        let local_count = exchange.local_count();
        let mut state = BackwardPassState::new(
            1,
            1,
            bwd_max_openings,
            n_state,
            local_count,
            &state,
            &horizon,
        );

        let mut inputs = BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &ctx,
            frozen: &frozen_templates,
            fcf: &mut fcf,
            cut_batches: &mut cut_batches,
            training_ctx: &training_ctx,
            comm: &comm,
            exchange: &mut exchange,
            records: &records,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            event_sender: None,
            risk_measures: &risk_measures,
            cut_activity_tolerance: 0.0,
            iteration: 1,
            local_work: local_count,
            fwd_offset: 0,

            traversal: &Traversal::default(),
            enumerated_state: &EnumeratedForwardScratch::default(),
        };

        let _ = state
            .run(&mut inputs)
            .expect("backward pass must not error");

        assert_eq!(
            inputs.workspaces[0].backward_accum.state_duals_buf.len(),
            n_state,
            "state_duals_buf must have length n_state after backward pass"
        );
    }

    /// AC2: a cut generated at iteration `g` that binds at a later
    /// iteration `i > g` under DCS must have its `last_active_iter` advanced to
    /// `i` by the (unchanged) per-stage metadata sync, fed by the DCS-maintained
    /// binding-count contribution.
    ///
    /// This drives the real [`BackwardPassState::sync_stage_metadata`] with a
    /// per-worker `metadata_sync_contribution` shaped exactly as the DCS backward
    /// path produces it (see `backward::tests::backward_dcs_binding_counts_match_frozen`,
    /// which proves the DCS path bumps exactly the binding slot): slot 1 bound at
    /// iteration `i`, slot 0 did not. Before the sync `last_active_iter == g` for
    /// both slots; after, the binding slot 1 advances to `i` (even though its
    /// `iteration_generated` is the older `g`), while the non-binding slot 0
    /// stays frozen at `g` (metadata staleness, unrelated to the frozen-template
    /// LP mode). This is the §3.1 clause-1 prerequisite the seed reads.
    #[test]
    fn dcs_binding_contribution_advances_last_active_iter() {
        use crate::cut_selection::CutMetadata;

        let g = 1_u64; // generation iteration
        let i = 5_u64; // binding (current) iteration, i > g
        let successor = 1_usize;
        let n_state = 1_usize;
        let n_stages = 2_usize;
        let n_openings = 1_usize;

        // FCF with two cuts at the successor stage, both generated at `g` and
        // last active at `g` (stale). Slot 1 is the one that will bind at `i`.
        let mut fcf = FutureCostFunction::new(n_stages, n_state, 8, 10, &vec![0; n_stages]);
        fcf.add_cut(NodeId(0), successor, g, 0, 1.0, &[0.0]);
        fcf.add_cut(NodeId(0), successor, g, 1, 0.0, &[2.0]);
        let meta = |generated: u64, last: u64| CutMetadata {
            iteration_generated: generated,
            forward_pass_index: 0,
            node: NodeId(0),
            active_count: 0,
            last_active_iter: last,
        };
        fcf.pools[successor].set_metadata_for_test(0, meta(g, g));
        fcf.pools[successor].set_metadata_for_test(1, meta(g, g));
        let pop = fcf.pools[successor].populated();

        // Pre-state: both slots frozen at generation iteration `g`.
        assert_eq!(fcf.pools[successor].metadata(0).last_active_iter, g);
        assert_eq!(fcf.pools[successor].metadata(1).last_active_iter, g);

        // One worker whose DCS binding-count contribution bumps only slot 1 (the
        // resident binding cut), matching what the DCS path emits at iteration i.
        let mut workspaces =
            single_workspace(MockSolver::always_ok(solution_1_0(0.0, 0.0)), n_state, 1);
        let contrib = &mut workspaces[0].backward_accum.metadata_sync_contribution;
        contrib.clear();
        contrib.resize(pop, 0);
        contrib[1] = 1;

        let comm = StubComm;
        let mut state = BackwardPassState::new(
            1,
            1,
            n_openings,
            n_state,
            1,
            &state_layout(n_state, 0),
            &HorizonMode::Finite {
                num_stages: n_stages,
            },
        );

        state
            .sync_stage_metadata(successor, 0, pop, i, &workspaces, &mut fcf, &comm)
            .expect("metadata sync must succeed");

        // Slot 1 bound at `i`: last_active_iter advances from g to i; active_count
        // accrues the increment. Slot 0 did not bind: it stays frozen at g.
        assert_eq!(
            fcf.pools[successor].metadata(1).last_active_iter,
            i,
            "binding slot 1 must advance to iteration {i}"
        );
        assert_eq!(fcf.pools[successor].metadata(1).active_count, 1);
        assert_eq!(
            fcf.pools[successor].metadata(0).last_active_iter,
            g,
            "non-binding slot 0 must stay frozen at its generation iteration {g}"
        );
        assert_eq!(fcf.pools[successor].metadata(0).active_count, 0);
    }

    // ── successor outcome set assembly ──────────────────────────────────────

    /// A single node fanning into two children (ascending id `1`, `2`) with
    /// distinct within-node opening counts and distinct out-edge
    /// probabilities: child `1` carries 2 openings at `P(0→1) = 0.25`, child
    /// `2` carries 2 openings at `P(0→2) = 0.75`.
    fn k_fan_node_graph() -> NodeGraph {
        NodeGraph {
            node_ids: vec![NodeId(0), NodeId(1), NodeId(2)].into(),
            nodes: vec![
                NodeRuntime {
                    stage: StageIdx(0),
                    pool_id: 0,
                    openings: NodeOpenings {
                        source: OpeningSource::Generated,
                        offset: 0,
                        len: 1,
                        q: 1.0,
                    },
                },
                NodeRuntime {
                    stage: StageIdx(1),
                    pool_id: 1,
                    openings: NodeOpenings {
                        source: OpeningSource::Generated,
                        offset: 0,
                        len: 2,
                        q: 0.5,
                    },
                },
                NodeRuntime {
                    stage: StageIdx(1),
                    pool_id: 2,
                    openings: NodeOpenings {
                        source: OpeningSource::Generated,
                        offset: 0,
                        len: 2,
                        q: 0.5,
                    },
                },
            ]
            .into(),
            successors: vec![
                vec![
                    NodeSuccessor {
                        child: NodePos(1),
                        probability: 0.25,
                    },
                    NodeSuccessor {
                        child: NodePos(2),
                        probability: 0.75,
                    },
                ],
                Vec::new(),
                Vec::new(),
            ]
            .into(),
            n_pools: 3,
            pool_stage: vec![StageIdx(0), StageIdx(1), StageIdx(1)],
        }
    }

    #[test]
    fn assemble_successor_outcome_weights_k_fan_canonical_order_and_product_weights() {
        let node_graph = k_fan_node_graph();
        let mut buf = Vec::new();
        assemble_successor_outcome_weights(&node_graph, NodePos(0), &mut buf);
        // Ascending child id (1 then 2), then within-child ω: P(0→1)·q_{1,ψ} =
        // 0.25·0.5 = 0.125 (twice), P(0→2)·q_{2,ψ} = 0.75·0.5 = 0.375 (twice).
        assert_eq!(buf, vec![0.125, 0.125, 0.375, 0.375]);
    }

    #[test]
    fn assemble_successor_outcome_weights_chain_degenerate_reduces_to_pinned_uniform_bit_pattern() {
        let n = 7_usize;
        let q = 1.0_f64 / (n as f64);
        let node_graph = NodeGraph {
            node_ids: vec![NodeId(0), NodeId(1)].into(),
            nodes: vec![
                NodeRuntime {
                    stage: StageIdx(0),
                    pool_id: 0,
                    openings: NodeOpenings {
                        source: OpeningSource::Generated,
                        offset: 0,
                        len: 1,
                        q: 1.0,
                    },
                },
                NodeRuntime {
                    stage: StageIdx(1),
                    pool_id: 1,
                    openings: NodeOpenings {
                        source: OpeningSource::Generated,
                        offset: 0,
                        len: n,
                        q,
                    },
                },
            ]
            .into(),
            successors: vec![
                vec![NodeSuccessor {
                    child: NodePos(1),
                    probability: 1.0,
                }],
                Vec::new(),
            ]
            .into(),
            n_pools: 2,
            pool_stage: vec![StageIdx(0), StageIdx(1)],
        };
        let mut buf = Vec::new();
        assemble_successor_outcome_weights(&node_graph, NodePos(0), &mut buf);
        assert_eq!(buf.len(), n);
        let expected_bits = q.to_bits();
        for &w in &buf {
            assert_eq!(
                w.to_bits(),
                expected_bits,
                "chain weight (P=1.0 times q) must be the exact pinned \
                 1.0/(n as f64) bit pattern, not a re-derived value"
            );
        }
    }

    /// AC: analytical product-weighted cut regression. `aggregate_cut`'s
    /// result over the assembled weights must equal the closed-form flat sum
    /// `Σ_(m,ψ) P(n→m)·q_{m,ψ}·outcome`, with the outcome vector ordered by
    /// ascending child node id then ω.
    #[test]
    fn analytical_product_weighted_cut_regression_k_fan() {
        let node_graph = k_fan_node_graph();
        let mut probabilities = Vec::new();
        assemble_successor_outcome_weights(&node_graph, NodePos(0), &mut probabilities);

        // One outcome per (child, ψ) in the same canonical order as `probabilities`:
        // child 1's two openings, then child 2's two.
        let outcomes = vec![
            BackwardOutcome {
                intercept: 10.0,
                coefficients: vec![1.0, 0.0],
                objective_value: 10.0,
            },
            BackwardOutcome {
                intercept: 20.0,
                coefficients: vec![2.0, 0.0],
                objective_value: 20.0,
            },
            BackwardOutcome {
                intercept: 30.0,
                coefficients: vec![0.0, 3.0],
                objective_value: 30.0,
            },
            BackwardOutcome {
                intercept: 40.0,
                coefficients: vec![0.0, 4.0],
                objective_value: 40.0,
            },
        ];

        let (intercept, coefficients) =
            RiskMeasure::Expectation.aggregate_cut(&outcomes, &probabilities);

        let expected_intercept: f64 = probabilities
            .iter()
            .zip(&outcomes)
            .map(|(p, o)| p * o.intercept)
            .sum();
        let expected_c0: f64 = probabilities
            .iter()
            .zip(&outcomes)
            .map(|(p, o)| p * o.coefficients[0])
            .sum();
        let expected_c1: f64 = probabilities
            .iter()
            .zip(&outcomes)
            .map(|(p, o)| p * o.coefficients[1])
            .sum();

        assert!((intercept - expected_intercept).abs() < 1e-12);
        assert!((coefficients[0] - expected_c0).abs() < 1e-12);
        assert!((coefficients[1] - expected_c1).abs() < 1e-12);
    }

    /// AC: `CVaR` tie-break order. Two tied-objective outcomes with distinct
    /// product weights: `CVaR`'s greedy allocation processes tied entries in
    /// their array (= canonical child-id-then-ω) index order — `child 1`
    /// (index 0) is allocated its full upper bound first, `child 2` (index 1)
    /// receives only the remainder. A `scenario_id`- or declaration-ordered
    /// successor set (child 2 processed first) would allocate the opposite
    /// way and produce a different intercept.
    #[test]
    fn cvar_aggregation_tie_break_follows_canonical_child_order() {
        let node_graph = NodeGraph {
            node_ids: vec![NodeId(0), NodeId(1), NodeId(2)].into(),
            nodes: vec![
                NodeRuntime {
                    stage: StageIdx(0),
                    pool_id: 0,
                    openings: NodeOpenings {
                        source: OpeningSource::Generated,
                        offset: 0,
                        len: 1,
                        q: 1.0,
                    },
                },
                NodeRuntime {
                    stage: StageIdx(1),
                    pool_id: 1,
                    openings: NodeOpenings {
                        source: OpeningSource::Generated,
                        offset: 0,
                        len: 1,
                        q: 1.0,
                    },
                },
                NodeRuntime {
                    stage: StageIdx(1),
                    pool_id: 2,
                    openings: NodeOpenings {
                        source: OpeningSource::Generated,
                        offset: 0,
                        len: 1,
                        q: 1.0,
                    },
                },
            ]
            .into(),
            successors: vec![
                vec![
                    NodeSuccessor {
                        child: NodePos(1),
                        probability: 0.3,
                    },
                    NodeSuccessor {
                        child: NodePos(2),
                        probability: 0.7,
                    },
                ],
                Vec::new(),
                Vec::new(),
            ]
            .into(),
            n_pools: 3,
            pool_stage: vec![StageIdx(0), StageIdx(1), StageIdx(1)],
        };
        let mut probabilities = Vec::new();
        assemble_successor_outcome_weights(&node_graph, NodePos(0), &mut probabilities);
        assert_eq!(probabilities, vec![0.3, 0.7]);

        // Tied objective_value: the sort key carries no information, so the
        // greedy allocation's processing order is decided purely by array
        // (= canonical) index.
        let outcomes = vec![
            BackwardOutcome {
                intercept: 10.0,
                coefficients: vec![],
                objective_value: 100.0,
            },
            BackwardOutcome {
                intercept: 20.0,
                coefficients: vec![],
                objective_value: 100.0,
            },
        ];
        let risk = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 1.0,
        };
        let (intercept, _) = risk.aggregate_cut(&outcomes, &probabilities);

        // mu[0] = min(0.3/0.5, 1.0) = 0.6, remaining = 0.4;
        // mu[1] = min(0.7/0.5, 0.4) = 0.4.
        // intercept = 0.6*10.0 + 0.4*20.0 = 14.0 — NOT 20.0, which is what a
        // child-2-processed-first (declaration/scenario_id) order would give.
        assert!(
            (intercept - 14.0).abs() < 1e-12,
            "expected canonical-order CVaR tie-break intercept 14.0, got {intercept}"
        );
    }

    // ── sampled scheduler resolution ────────────────────────────────────

    #[test]
    fn resolve_backward_scheduler_dcs_forces_by_scenario_else_keeps_configured() {
        use cobre_io::config::BackwardScheduler;

        // No active DCS: the configured scheduler passes through unchanged.
        assert!(matches!(
            resolve_backward_scheduler(false, BackwardScheduler::ByScenario {}),
            BackwardScheduler::ByScenario {}
        ));
        assert!(matches!(
            resolve_backward_scheduler(false, BackwardScheduler::ByNode { block_size: None }),
            BackwardScheduler::ByNode { .. }
        ));
        // An active DCS iteration always forces the by-scenario path — its
        // cut-free lazy core is incompatible with the by-node frozen-LP load.
        assert!(matches!(
            resolve_backward_scheduler(true, BackwardScheduler::ByNode { block_size: None }),
            BackwardScheduler::ByScenario {}
        ));
        assert!(matches!(
            resolve_backward_scheduler(true, BackwardScheduler::ByScenario {}),
            BackwardScheduler::ByScenario {}
        ));
    }

    #[test]
    fn by_node_scratch_sizing_follows_configured_scheduler_only() {
        use cobre_io::config::BackwardScheduler;

        let state = state_layout(3, 0);
        let horizon = HorizonMode::Finite { num_stages: 5 };

        // No scheduler set ⇒ empty (the pre-existing ByScenario/sampled default).
        let baseline = BackwardPassState::new(1, 1, 4, 0, 2, &state, &horizon);
        assert_eq!(baseline.by_node_scratch_arena_capacity(), 0);

        // set_scheduler(ByNode) sizes it.
        let mut by_node_only = BackwardPassState::new(1, 1, 4, 0, 2, &state, &horizon);
        by_node_only.set_scheduler(BackwardScheduler::ByNode { block_size: None });
        assert!(by_node_only.by_node_scratch_arena_capacity() > 0);

        // set_scheduler(ByScenario) keeps it empty.
        let mut by_scenario_only = BackwardPassState::new(1, 1, 4, 0, 2, &state, &horizon);
        by_scenario_only.set_scheduler(BackwardScheduler::ByScenario {});
        assert_eq!(by_scenario_only.by_node_scratch_arena_capacity(), 0);
    }

    // ── per-node basis isolation + chain byte-address ──────────────────

    #[test]
    fn per_node_basis_isolation_and_chain_byte_address() {
        // K-fan: node 0's two distinct successor nodes (positions 1 and 2) share
        // one `BasisStoreSliceMut` sized by node count.
        let ng = k_fan_node_graph();
        let node_a = ng.successors[NodePos(0)][0].child;
        let node_b = ng.successors[NodePos(0)][1].child;
        assert_ne!(node_a, node_b);

        let mut store = BasisStore::new(1, ng.nodes.len());
        {
            let mut slices = store.split_workers_mut(1);
            let slice = &mut slices[0];
            *slice.get_mut(0, node_a) =
                Some(CapturedBasis::new(2, 1, 0, 0, 0, ng.node_ids[node_a]));
            assert!(
                slice.get(0, node_a).is_some(),
                "basis saved at successor node A resolves at node A"
            );
            assert!(
                slice.get(0, node_b).is_none(),
                "successor node B resolves nothing — per-node isolation by successor node"
            );
        }

        // Synthesized chain: successor node position == successor stage, so the
        // backward basis key `(m, successor_node)` is the pre-change
        // `(m, successor_stage)` flat slot `[m*num_nodes + node]` byte-for-byte
        // (num_nodes == num_stages on the chain).
        let n_stages = 4usize;
        let stochastic = make_stochastic_context(n_stages, 3);
        let chain = crate::test_support::chain_node_graph(&stochastic);
        assert_eq!(chain.nodes.len(), n_stages);
        for t in 0..n_stages - 1 {
            assert_eq!(chain.nodes[NodePos(t)].stage, StageIdx(t));
            assert_eq!(
                chain.successors[NodePos(t)][0].child,
                NodePos(t + 1),
                "chain successor node position must equal the successor stage"
            );
        }
        let chain_store = BasisStore::new(2, chain.nodes.len());
        assert_eq!(
            chain_store.num_nodes(),
            n_stages,
            "chain node-axis stride equals num_stages, so (m, successor_node) == (m, successor_stage)"
        );
    }

    // ── no graph-shape dispatch on the backward path ──────────────

    #[test]
    fn backward_engine_has_no_graph_shape_dispatch() {
        // Chain parity is degeneracy, not dispatch: no backward engine path
        // branches on a graph-shape predicate to reach preserved legacy code.
        let sources: [(&str, &str); 6] = [
            (
                "backward_pass_state",
                include_str!("backward_pass_state.rs"),
            ),
            ("backward/mod", include_str!("backward/mod.rs")),
            (
                "backward/by_scenario",
                include_str!("backward/by_scenario.rs"),
            ),
            ("backward/by_node", include_str!("backward/by_node.rs")),
            ("backward/lp_setup", include_str!("backward/lp_setup.rs")),
            (
                "backward/replicated",
                include_str!("backward/replicated.rs"),
            ),
        ];
        // Assemble the forbidden tokens from fragments so this test's own source
        // (folded into the first `include_str!`) does not self-match.
        let us = "_";
        let forbidden = [
            format!("is{us}chain"),
            format!("nodes.is{us}empty()"),
            format!("graph.is{us}none()"),
        ];
        for (name, src) in sources {
            for pat in &forbidden {
                assert!(
                    !src.contains(pat.as_str()),
                    "backward engine file `{name}` contains shape-dispatch predicate `{pat}`"
                );
            }
        }
    }

    // ── fusion: real-solver AC1/AC2 solve-count checks ─────────────────────

    /// Run one real (`ActiveSolver`) enumerated forward+backward round on
    /// `setup`'s whole graph, returning the raw [`BackwardResult`] — in
    /// particular its real `lp_solves` count (`ws.solver.statistics()
    /// .solve_count` deltas before/after the backward sweep), the metric
    /// fusion must reduce to zero on an all-External terminal fan and leave
    /// unchanged on a Generated one.
    ///
    /// `stage_ctx`/`training_ctx` are built as direct field literals (mirroring
    /// `StudySetup::train_inner`), not via the `setup.stage_ctx()`/
    /// `setup.training_ctx()` accessors: those take `&self` opaquely, so the
    /// compiler would treat their returned borrows as covering the whole
    /// struct and reject the later `&mut setup.fcf` this harness also needs.
    #[allow(clippy::too_many_lines)]
    fn run_real_enumerated_round(
        setup: &mut crate::StudySetup,
        event_sender: Option<&Sender<TrainingEvent>>,
    ) -> BackwardResult {
        use cobre_solver::ActiveSolver;
        use cobre_stochastic::ForwardNoiseTables;

        let comm = StubComm;
        let num_stages = setup.inputs.stage_data.stages.len();
        let node_graph = &setup.inputs.node_graph;
        let total_forward_passes = usize::try_from(
            crate::test_support::node_scenario_count(node_graph)
                .expect("scenario count must not overflow"),
        )
        .expect("fits usize");

        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("workspace pool must build");
        let mut basis_store = BasisStore::new(total_forward_passes, node_graph.nodes.len());
        let mut records: Vec<TrajectoryRecord> = (0..total_forward_passes * num_stages)
            .map(|_| TrajectoryRecord {
                primal: Vec::new(),
                dual: Vec::new(),
                stage_cost: 0.0,
                node_id: NodeId(0),
                state: Vec::new(),
            })
            .collect();

        let fixture = StageContextFixture::from_stage_templates(
            &setup.inputs.stage_data.stage_templates,
            setup.inputs.stage_data.stage_templates.state_boxes(),
        )
        .ncs_stochastic_dense_col(&setup.inputs.ncs.stochastic_dense_col)
        .ncs_stochastic_windows(&setup.inputs.ncs.stochastic_windows)
        .anticipated_windows(
            setup
                .inputs
                .stage_data
                .study_dims
                .anticipated_plants
                .windows(),
        )
        .study_stage_ids(&setup.inputs.study_stage_ids)
        .ncs_max_gen(&setup.inputs.ncs.max_gen)
        .ncs_allow_curtailment(&setup.inputs.ncs.allow_curtailment)
        .discount_factors(setup.inputs.stage_data.time_value.discount_factors())
        .cumulative_discount_factors(
            setup
                .inputs
                .stage_data
                .time_value
                .cumulative_discount_factors(),
        )
        .stage_lag_transitions(&setup.inputs.stage_data.stage_lag_transitions)
        .noise_group_ids(&setup.inputs.stage_data.noise_group_ids);
        let stage_ctx = fixture.ctx();
        let tr = &setup.inputs.scenario_libraries.training;
        let training_ctx = TrainingContext {
            horizon: &setup.inputs.horizon,
            state: &setup.inputs.stage_data.state,
            cut_state_layouts: &setup.inputs.cut_state_layouts,
            study_dims: &setup.inputs.stage_data.study_dims,
            inflow_method: &setup.inputs.stage_data.study_dims.inflow_method,
            stochastic: &setup.inputs.stochastic,
            initial_state: &setup.inputs.initial.state,
            inflow_scheme: tr.inflow_scheme,
            load_scheme: tr.load_scheme,
            ncs_scheme: tr.ncs_scheme,
            stages: &setup.inputs.stage_data.stages,
            historical_library: tr.historical.as_ref(),
            external_inflow_library: tr.external_inflow.as_ref(),
            external_load_library: tr.external_load.as_ref(),
            external_ncs_library: tr.external_ncs.as_ref(),
            lag_accum_seed: &setup.inputs.initial.inflow_seeds.accum,
            lag_weight_seed: &setup.inputs.initial.inflow_seeds.weight,
            dcs: None,
            node_graph: &setup.inputs.node_graph,
        };

        let sampler =
            crate::forward::build_sampler_from_ctx(&training_ctx).expect("forward sampler");
        let mut noise_tables = ForwardNoiseTables::default();
        sampler
            .rebuild_noise_tables(
                1,
                u32::try_from(total_forward_passes).expect("fits u32"),
                stage_ctx.noise_group_ids,
                &mut noise_tables,
            )
            .expect("test fixture never exceeds the Sobol dimension cap");
        let frozen: Vec<StageTemplate> = (0..node_graph.n_pools)
            .map(|p| stage_ctx.templates[node_graph.pool_stage[p].0].clone())
            .collect();
        let traversal = Traversal::resolve(node_graph, true, 0);
        let Traversal::Enumerated(plan) = &traversal else {
            unreachable!("resolve(is_enumerated=true, ..) always yields Enumerated");
        };

        let fwd_params = crate::forward::EnumeratedParams {
            iteration: 1,
            fwd_offset: 0,
            local_forward_passes: total_forward_passes,
            total_forward_passes,
            ctx: &stage_ctx,
            frozen: &frozen,
            fcf: &setup.fcf,
            training_ctx: &training_ctx,
            sampler: &sampler,
            noise_tables: &noise_tables,
            dcs: None,
            event_sender,
        };

        let mut scratch = EnumeratedForwardScratch::default();
        crate::forward::run_enumerated_forward(
            plan,
            &mut scratch,
            &mut pool.workspaces,
            &mut basis_store,
            &mut records,
            &fwd_params,
        )
        .expect("real forward round must not error");

        let n_state = training_ctx.state.n_state;
        let mut cut_batches = empty_cut_batches(node_graph.n_pools);
        let mut exchange = ExchangeBuffers::new(training_ctx.state, total_forward_passes.max(1), 1);
        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, 1);
        let risk_measures = vec![RiskMeasure::Expectation; num_stages];
        let bwd_max_openings = node_graph
            .successors
            .iter()
            .map(|succs| {
                succs
                    .iter()
                    .map(|s| node_graph.nodes[s.child].openings.len)
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0)
            .max(1);
        let mut state_machine = BackwardPassState::new(
            1,
            1,
            bwd_max_openings,
            n_state,
            total_forward_passes,
            training_ctx.state,
            training_ctx.horizon,
        );

        let mut inputs = BackwardPassInputs {
            workspaces: &mut pool.workspaces,
            basis_store: &mut basis_store,
            ctx: &stage_ctx,
            frozen: &frozen,
            fcf: &mut setup.fcf,
            cut_batches: &mut cut_batches,
            training_ctx: &training_ctx,
            comm: &comm,
            exchange: &mut exchange,
            records: &records,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            event_sender,
            risk_measures: &risk_measures,
            cut_activity_tolerance: 0.0,
            iteration: 1,
            local_work: 0,
            fwd_offset: 0,
            traversal: &traversal,
            enumerated_state: &scratch,
        };

        state_machine
            .run(&mut inputs)
            .expect("real backward round must not error")
    }

    /// AC1: on an all-External terminal fan (every child of the sole
    /// cut-generating node is a fusion-eligible External terminal leaf), the
    /// real backward performs ZERO LP solves — every child's outcome is
    /// filled from the forward-captured fused slice instead of being solved.
    #[test]
    fn enumerated_backward_fuses_all_external_terminal_fan_to_zero_solves() {
        let mut setup = crate::test_support::external_distinct_fan_setup(2, 1);
        let result = run_real_enumerated_round(&mut setup, None);
        assert_eq!(
            result.lp_solves, 0,
            "an all-External terminal fan must be fully fused: zero backward LP solves"
        );
        assert_eq!(
            result.cuts_generated, 1,
            "the sole cut-generating node (root) still appends exactly one cut"
        );
    }

    /// AC2: on a terminal-Generated fan (no child is fusion-eligible), the
    /// real backward still solves every opening of every child — exhaustive
    /// integration is unaffected by fusion.
    #[test]
    fn enumerated_backward_still_solves_every_opening_for_generated_terminal_fan() {
        let mut setup = crate::test_support::terminal_generated_fan_setup(2, 1);
        let result = run_real_enumerated_round(&mut setup, None);
        assert!(
            result.lp_solves > 0,
            "power: a terminal-Generated fan has no fusion-eligible child, so the backward \
             must genuinely solve"
        );
        assert_eq!(
            result.cuts_generated, 1,
            "the sole cut-generating node (root) still appends exactly one cut"
        );
    }

    /// `run_enumerated_backward`'s `stage_stats` is populated on a genuine
    /// (non-fused) real solve and its summed `lp_solves` reconciles exactly
    /// with `BackwardResult::lp_solves`; every entry's `rank`/`worker_id` are
    /// real, not the `-1` sentinel.
    #[test]
    fn enumerated_backward_stage_stats_reconciles_with_lp_solves() {
        let mut setup = crate::test_support::terminal_generated_fan_setup(2, 1);
        let result = run_real_enumerated_round(&mut setup, None);

        assert!(
            !result.stage_stats.is_empty(),
            "stage_stats must be populated on the enumerated backward path"
        );
        let mut summed_lp_solves = 0_u64;
        for (_, entries) in &result.stage_stats {
            for &(rank, worker_id, _, ref delta) in entries {
                assert_ne!(rank, -1, "backward stage_stats rank must be real, not -1");
                assert_ne!(
                    worker_id, -1,
                    "backward stage_stats worker_id must be real, not -1"
                );
                summed_lp_solves += delta.lp_solves;
            }
        }
        assert_eq!(
            summed_lp_solves, result.lp_solves,
            "the per-solve stage_stats sum must reconcile with the aggregate lp_solves"
        );
    }

    /// The enumerated backward drives a single workspace (never a rayon
    /// pool), so it emits exactly one `WorkerTiming { phase: Backward }`
    /// event, and its `backward_wall_ms` is consistent with
    /// `BackwardResult::elapsed_ms` — both derive from the same
    /// `Instant::elapsed()` call: `elapsed_ms` floors to whole milliseconds,
    /// `backward_wall_ms` keeps the fractional part.
    #[test]
    fn enumerated_backward_emits_worker_timing_consistent_with_elapsed_ms() {
        let mut setup = crate::test_support::terminal_generated_fan_setup(2, 1);
        let (tx, rx) = std::sync::mpsc::channel::<TrainingEvent>();
        let result = run_real_enumerated_round(&mut setup, Some(&tx));
        drop(tx);

        let backward_walls: Vec<f64> = rx
            .try_iter()
            .filter_map(|e| match e {
                TrainingEvent::WorkerTiming {
                    phase: WorkerTimingPhase::Backward,
                    timings,
                    ..
                } => Some(timings.backward_wall_ms),
                _ => None,
            })
            .collect();

        assert_eq!(
            backward_walls.len(),
            1,
            "the enumerated backward drives a single workspace, so it must emit exactly one \
             Backward WorkerTiming event, got {backward_walls:?}"
        );
        let backward_wall_ms = backward_walls[0];
        assert!(
            backward_wall_ms > 0.0,
            "backward_wall_ms must be non-zero on a genuine (non-fused) real solve"
        );
        #[allow(clippy::cast_precision_loss)]
        let elapsed_ms_f64 = result.elapsed_ms as f64;
        assert!(
            (elapsed_ms_f64 - 0.001..elapsed_ms_f64 + 1.0).contains(&backward_wall_ms),
            "backward_wall_ms ({backward_wall_ms}) must be consistent with the same \
             Instant::elapsed() call that produced elapsed_ms ({}, floored to whole ms)",
            result.elapsed_ms
        );
    }
}
