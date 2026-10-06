//! Training session state container and iteration loop.
//!
//! [`TrainingSession`] owns all scratch buffers for a single [`crate::training::train`] call.
//! No hot-path allocations; forward and backward passes encapsulated in their own state structs.

use cobre_comm::CommError::CollectiveFailed;
use cobre_solver::SolverError;
use cobre_solver::freeze_rows_into_template;

use crate::lp::indexer::CutSlot;

use crate::cut_selection::CutSelectionStrategy::Dominated;
use crate::cut_selection::CutSelectionStrategy::Dynamic;
use crate::cut_selection::CutSelectionStrategy::Level1;
use crate::cut_selection::CutSelectionStrategy::Lml1;
use crate::visited_states::VisitedStatesArchive;
use crate::workspace::CapturedBasis;

use std::ops::RangeInclusive;
pub(crate) mod iteration_scratch;
pub(crate) mod rank_distribution;
pub(crate) mod results;
pub(crate) mod runtime;
use self::iteration_scratch::IterationScratch;
use self::rank_distribution::RankDistribution;
use self::results::TrainingResults;
use self::runtime::RuntimeHandles;

use std::sync::mpsc::Sender;
use std::time::Instant;

use cobre_comm::{Communicator, ReduceOp, per_rank_counts};
use cobre_core::{StageRowSelectionRecord, TrainingEvent};
use cobre_solver::SolverInterface;

use crate::{
    SddpError, SolverProfiles, TrainingConfig,
    backward::BackwardResult,
    backward_pass_state::{BackwardPassInputs, BackwardPassState},
    context::{StageContext, TrainingContext},
    convergence::convergence::ConvergenceMonitor,
    cut::fcf::FutureCostFunction,
    cut::row::build_cut_row_batch_into,
    cut_selection::CutActivityUpdates,
    cut_sync::CutSyncBuffers,
    forward::{ForwardBound, ForwardResult, SyncResult, sync_forward},
    forward_pass_state::{ForwardPassInputs, ForwardPassState},
    lower_bound::LbEvalScratchBundle,
    lower_bound::evaluate_lower_bound,
    policy::orchestration::CheckpointState,
    rank_reconcile::{StopInputs, agree_stop_inputs, reconcile_error_flag, reconcile_result},
    risk_measure::{RiskMeasure, uniform_effective_measure},
    setup::NodeGraph,
    setup::node_graph::{NodePos, StageIdx, Traversal, enumerated_requires_state_exchange},
    solver_stats::{
        SOLVER_STATS_DELTA_SCALAR_FIELDS, SolverStatsDelta, SolverStatsLogEntry,
        aggregate_solver_statistics, pack_delta_scalars, unpack_delta_scalars,
    },
    state_exchange::ExchangeBuffers,
    training::training::rank_local_basis_cache,
    training::{TrainingOutcome, TrainingResult, broadcast_basis_cache},
    workspace::{BasisStore, NoisePreallocation, WorkspacePool, WorkspaceSizing},
};

// ---------------------------------------------------------------------------
// emit helper (mirrors training.rs emit)
// ---------------------------------------------------------------------------

#[inline]
fn emit(sender: Option<&Sender<TrainingEvent>>, event: TrainingEvent) {
    if let Some(s) = sender {
        let _ = s.send(event);
    }
}

// ---------------------------------------------------------------------------
// IterationOutcome
// ---------------------------------------------------------------------------

/// Result of a single training iteration.
///
/// Returned by [`TrainingSession::run_iteration`] to let the outer orchestrator
/// in [`crate::training::train`] decide whether to continue, stop, or handle
/// an error.
#[derive(Debug)]
pub(crate) enum IterationOutcome {
    /// The iteration completed normally; the loop should continue.
    Continue,
    /// A configured stop was met or the iteration budget ran out; the loop
    /// should break.
    Converged,
    /// A shutdown request ended training with neither; the loop should break.
    Shutdown,
}

// ---------------------------------------------------------------------------
// TrainingSession
// ---------------------------------------------------------------------------

/// Owns all per-training-run scratch state for one call to `train`, borrowing
/// `solver`, `fcf`, `stage_ctx`, `training_ctx`, and `comm` for the lifetime
/// `'a` of the run.
pub(crate) struct TrainingSession<'a, S: SolverInterface + Send, C: Communicator> {
    // ── Borrowed inputs (live for 'a) ─────────────────────────────────────
    solver: &'a mut S,
    fcf: &'a mut FutureCostFunction,
    stage_ctx: &'a StageContext<'a>,
    training_ctx: &'a TrainingContext<'a>,
    comm: &'a C,

    // ── Training configuration for this run ───────────────────────────────
    config: TrainingConfig,

    // ── Runtime handles (per-invocation hooks; set once in new) ───────────
    runtime: RuntimeHandles,

    // ── Rank math (constant for the run) ──────────────────────────────────
    ranks: RankDistribution,

    // ── Per-run scratch buffers (owned; reused across iterations) ─────────
    fwd_pool: WorkspacePool<S>,
    basis_store: BasisStore,
    exchange_bufs: ExchangeBuffers,
    cut_sync_bufs: CutSyncBuffers,
    visited_archive: Option<VisitedStatesArchive>,
    /// Reusable projected trial-state buffer for cut selection; grown on first
    /// use, reused across iterations (no per-cut/per-trial-point allocation).
    cut_selection_state_scratch: Vec<f64>,
    /// Interior cut-generating nodes (canonical order), the loop-invariant filter
    /// cut management sweeps each cycle; the graph topology is fixed for the run,
    /// so it is resolved once here rather than re-derived per cut-selection cycle.
    interior_cut_nodes: Vec<NodePos>,
    scratch: IterationScratch,
    convergence_monitor: ConvergenceMonitor,

    // ── Forward-pass owned scratch ────────────────────────────────────────
    fwd_state: ForwardPassState,

    // ── Backward-pass owned scratch ───────────────────────────────────────
    bwd_state: BackwardPassState,

    // ── Result accumulators (updated each iteration; finalized in finalize()) ─
    results: TrainingResults,
}

impl<'a, S, C: Communicator> TrainingSession<'a, S, C>
where
    S: SolverInterface<Profile = cobre_solver::ActiveProfile> + Send,
{
    /// Allocate all per-training-run scratch and emit the `TrainingStarted` event.
    ///
    /// # Errors
    ///
    /// Returns `SddpError::Solver(e)` if the workspace pool cannot be constructed.
    // Rationale: allocates and wires every per-run scratch buffer in one place;
    // splitting it would thread each field through a helper for no clarity gain.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn new(
        solver: &'a mut S,
        mut config: TrainingConfig,
        fcf: &'a mut FutureCostFunction,
        stage_ctx: &'a StageContext<'a>,
        training_ctx: &'a TrainingContext<'a>,
        comm: &'a C,
        solver_factory: impl Fn() -> Result<S, SolverError>,
        solver_profiles: SolverProfiles,
    ) -> Result<Self, SddpError> {
        let horizon = training_ctx.horizon;
        let state = training_ctx.state;
        let num_stages = horizon.num_stages();
        let total_forward_passes = config.loop_config.forward_passes as usize;
        let ranks = RankDistribution::new(comm, total_forward_passes);

        // Map the first training iteration to slot `warm_start_count` so
        // training cuts pack densely with no reserved leading block.
        fcf.set_iteration_base(config.loop_config.start_iteration + 1);

        // Per-slot backward buffers (`slot_increments`, `metadata_sync_contribution`,
        // the recon lookup) are indexed by cut-pool slot, so they must cover the
        // LARGEST pool a worker's sweep may touch — sizing from `pools[0]` alone
        // truncates a pool whose heterogeneous visit bound exceeds pool 0's. On a
        // chain every pool has identical capacity, so this equals `pools[0]`.
        let max_pool_capacity = fcf.pools.iter().map(|p| p.capacity).max().unwrap_or(0);

        let n_threads = config.loop_config.n_fwd_threads.max(1);
        // The wrong-but-compiling alternative is the per-stage term alone: on a
        // declared multi-successor node (a fan-out), `n_openings` is the
        // FLATTENED successor-outcome count (`assemble_outcome_weights`), which
        // can exceed every stage's own opening-tree size and overflow
        // `StageWorkerStatsBuffer`; the per-node term below covers it. On a chain
        // a node's successor IS the next stage, so it is already covered here.
        let max_openings = (0..num_stages)
            .map(|t| training_ctx.stochastic.opening_tree().n_openings(t))
            .max()
            .unwrap_or(0)
            .max(training_ctx.node_graph.max_successor_outcome_count());
        let mut fwd_pool = WorkspacePool::try_new(
            ranks.fwd_rank,
            n_threads,
            training_ctx,
            stage_ctx,
            WorkspaceSizing {
                max_openings,
                initial_pool_capacity: max_pool_capacity,
                max_local_fwd: ranks.max_local_fwd,
                noise: NoisePreallocation::StochasticDim,
            },
            solver_factory,
        )
        .map_err(SddpError::Solver)?;
        // Pre-size scratch_basis to the largest template: `reconstruct_basis`
        // runs on every stored-basis apply, so this keeps the hot path
        // allocation-free.
        let max_cols = stage_ctx
            .templates
            .iter()
            .map(|t| t.num_cols)
            .max()
            .unwrap_or(0);
        let max_rows = stage_ctx
            .templates
            .iter()
            .map(|t| t.num_rows)
            .max()
            .unwrap_or(0);
        fwd_pool.resize_scratch_bases(max_cols, max_rows);

        // DcsSolveScratch/DcsScoringScratch are shared per worker, not per pool,
        // so they must cover the LARGEST pool a worker's sweep may touch (same
        // `max_pool_capacity` the per-slot backward buffers size from).
        fwd_pool.reserve_dcs_scratch(state.n_state, max_pool_capacity);

        // Sized for max local forward passes so scenario indices stay stable
        // across iterations; the second axis is the node count (== num_stages on
        // the chain) so the backward warm-start keys by successor node position.
        let basis_store = BasisStore::new(ranks.max_local_fwd, training_ctx.node_graph.nodes.len());

        let actual_per_rank = per_rank_counts(total_forward_passes, ranks.num_ranks);
        let exchange_bufs = ExchangeBuffers::with_actual_counts(
            state,
            ranks.max_local_fwd,
            ranks.num_ranks,
            &actual_per_rank,
        );
        let cut_sync_bufs = CutSyncBuffers::with_distribution(
            state.n_state,
            ranks.max_local_fwd,
            ranks.num_ranks,
            total_forward_passes,
        );

        // Needed by every cut-selection strategy (the value-evaluation kernel
        // scores cuts at these trial points) and by `export_states`.
        let needs_archive =
            config.cut_management.cut_selection.is_some() || config.events.export_states;
        let visited_archive = if needs_archive {
            Some(VisitedStatesArchive::new(
                training_ctx.node_graph.nodes.len(),
                state.n_state,
                config.loop_config.max_iterations,
                total_forward_passes,
            ))
        } else {
            None
        };

        // `.take()` the non-Copy handles so `config` stays valid and can be
        // stored by value; `export_states` is Copy and read directly.
        let event_sender = config.events.event_sender.take();
        let shutdown_flag = config.events.shutdown_flag.take();
        let periodic_checkpoint = config.events.periodic_checkpoint.take();
        let export_states = config.events.export_states;

        let mut convergence_monitor = ConvergenceMonitor::with_iteration_budget(
            config.loop_config.stopping_rules.clone(),
            config.loop_config.max_iterations,
        );
        convergence_monitor.resume_at(
            config.loop_config.start_iteration,
            &config.loop_config.resume_lower_bound_history,
        );

        // Emit before the locals move into RuntimeHandles, while `event_sender`
        // is still bound here.
        #[allow(clippy::cast_possible_truncation)]
        emit(
            event_sender.as_ref(),
            TrainingEvent::TrainingStarted {
                case_name: String::new(),
                stages: num_stages as u32,
                hydros: state.hydro_count as u32,
                thermals: 0,
                ranks: ranks.num_ranks as u32,
                threads_per_rank: n_threads as u32,
                timestamp: String::new(),
            },
        );

        let runtime = RuntimeHandles::new(
            event_sender,
            shutdown_flag,
            export_states,
            periodic_checkpoint,
        );

        let results = TrainingResults::new(config.loop_config.start_iteration);

        // The LB LP is the ROOT's stage-0 LP, so its append-only cut-row map is
        // sized from the root pool's capacity — resolved by STAGE, never by array
        // position: nodes[0] is the smallest-id node, not necessarily the root, so
        // fcf.pools[0] could under-size the map for a root pool larger than pool 0.
        // On a chain the root IS nodes[0] (pool 0), so this equals pools[0].
        let lb_root_node = training_ctx
            .node_graph
            .frontier_node(StageIdx(0))
            .ok_or_else(|| {
                SddpError::Validation("training session: stage 0 carries no alive node".to_string())
            })?;
        let lb_root_pool = training_ctx.node_graph.nodes[lb_root_node].pool_id;
        let scratch = IterationScratch::new(
            ranks.max_local_fwd,
            &training_ctx.node_graph.pool_stage,
            fcf.pools[lb_root_pool].capacity,
            stage_ctx.template(StageIdx(0)).num_rows,
            training_ctx,
            stage_ctx,
        );

        let n_workers_local = fwd_pool.workspaces.len();
        let mut fwd_state = ForwardPassState::new(n_workers_local, num_stages, ranks.max_local_fwd);
        fwd_state.set_profile(solver_profiles.forward);
        // Resolved once here, at training start — the study's node graph has
        // existed since `StudySetup` construction, and `resolve_enumerated_training_count`
        // already ran the enumerated admissibility guards there (a `StudySetup`
        // that failed them never reaches a `TrainingSession::new` call), so this
        // resolution is representational, not a second admissibility check.
        fwd_state.set_traversal(Traversal::resolve(
            training_ctx.node_graph,
            config.loop_config.training_enumerated,
            config.loop_config.forward_passes,
        ));

        // Enumerated forward leaves a node's persisted outgoing state
        // zero-filled on any rank whose assigned paths never visit it
        // (`EnumeratedForwardScratch::ensure_sized`), while the replicated
        // backward (`run_backward_node_replicated`) partitions every non-leaf
        // node's openings across ALL ranks regardless — sound only absent
        // interior branching. Hard-reject before any solve rather than let a
        // world >= 2 interior-branching run cut against a zeroed state.
        if comm.size() > 1
            && matches!(fwd_state.traversal(), Traversal::Enumerated(_))
            && let Some(node_id) = enumerated_requires_state_exchange(training_ctx.node_graph)
        {
            return Err(SddpError::Validation(format!(
                "enumerated training at world >= 2 requires a graph with no interior \
                 branching (a deterministic trunk + terminal fan): node {node_id} is a \
                 non-leaf node not shared by every root→leaf path — the enumerated \
                 forward's per-rank state exchange is elided for it, so the replicated \
                 backward would cut against a zeroed incoming state on any rank that never \
                 visits it, pending the interior-node state-exchange fix"
            )));
        }

        let real_states_capacity = exchange_bufs.real_total_scenarios() * state.n_state;
        let mut bwd_state = BackwardPassState::new(
            n_workers_local,
            ranks.num_ranks,
            max_openings,
            real_states_capacity,
            ranks.max_local_fwd,
            state,
            horizon,
        );
        bwd_state.set_profile(solver_profiles.backward);
        bwd_state.set_scheduler(solver_profiles.backward_scheduler);
        bwd_state.set_hardest_first_claim_order(solver_profiles.hardest_first_claim_order);

        Ok(Self {
            solver,
            fcf,
            stage_ctx,
            training_ctx,
            comm,
            config,
            runtime,
            ranks,
            fwd_pool,
            basis_store,
            exchange_bufs,
            cut_sync_bufs,
            visited_archive,
            cut_selection_state_scratch: Vec::new(),
            interior_cut_nodes: {
                let ng = training_ctx.node_graph;
                ng.nodes
                    .iter_indexed()
                    .filter(|&(pos, n)| n.stage >= StageIdx(1) && !ng.successors[pos].is_empty())
                    .map(|(pos, _)| pos)
                    .collect()
            },
            scratch,
            convergence_monitor,
            fwd_state,
            bwd_state,
            results,
        })
    }

    /// Returns the range of iteration indices this session should run.
    pub(crate) fn iteration_range(&self) -> RangeInclusive<u64> {
        (self.config.loop_config.start_iteration + 1)..=self.config.loop_config.max_iterations
    }

    /// Sum the cumulative rows-in-LP accumulators (`(sum, count)`) across this
    /// rank's forward-pass workspaces. Zero for non-lazy methods — only the lazy
    /// solve path touches these accumulators.
    fn rows_in_lp_local_totals(&self) -> (u64, u64) {
        self.fwd_pool.workspaces.iter().fold((0, 0), |(s, c), w| {
            let d = &w.backward_accum.dcs_solve;
            (s + d.rows_in_lp_sum, c + d.rows_in_lp_count)
        })
    }

    /// Largest resident-set size observed across this rank's forward-pass
    /// workspaces (zero when the lazy path never ran).
    fn rows_in_lp_local_max(&self) -> u64 {
        self.fwd_pool
            .workspaces
            .iter()
            .map(|w| w.backward_accum.dcs_solve.rows_in_lp_max)
            .max()
            .unwrap_or(0)
    }

    /// Execute one training iteration.
    ///
    /// On `Err(e)` the caller must call `finalize_with_error(e)`.
    ///
    /// # Errors
    ///
    /// Propagates `SddpError` from forward pass, sync, backward pass, or lower
    /// bound evaluation failures.
    pub(crate) fn run_iteration(&mut self, iteration: u64) -> Result<IterationOutcome, SddpError> {
        let iter_start = Instant::now();

        // Snapshot before this iteration's solves so the post-backward delta
        // isolates this iteration's contribution.
        let (rows_in_lp_sum_before, rows_in_lp_count_before) = self.rows_in_lp_local_totals();

        let (forward_result, sync_result, fwd_solve_time_ms) = self.run_forward_phase(iteration)?;
        let (backward_result, bwd_solve_time_ms) = self.run_backward_phase(iteration)?;

        // Reduced across ranks so the reported figures are work-distribution
        // invariant (Sum for the per-iteration delta, Max for the running peak).
        // Forward + backward are the only lazy solves; LB is all-cuts.
        let (rows_in_lp_sum, rows_in_lp_count, rows_in_lp_max) = {
            let (sum_after, count_after) = self.rows_in_lp_local_totals();
            let sum_delta = sum_after - rows_in_lp_sum_before;
            let count_delta = count_after - rows_in_lp_count_before;
            #[allow(clippy::cast_precision_loss)]
            let local_sum = [sum_delta as f64, count_delta as f64];
            let mut global_sum = [0.0_f64; 2];
            self.comm
                .allreduce(&local_sum, &mut global_sum, ReduceOp::Sum)
                .map_err(SddpError::Communication)?;
            #[allow(clippy::cast_precision_loss)]
            let local_max = [self.rows_in_lp_local_max() as f64];
            let mut global_max = [0.0_f64; 1];
            self.comm
                .allreduce(&local_max, &mut global_max, ReduceOp::Max)
                .map_err(SddpError::Communication)?;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            (
                global_sum[0].round() as u64,
                global_sum[1].round() as u64,
                global_max[0].round() as u64,
            )
        };

        self.run_cut_management(iteration)?;

        grow_pools_for_next_iteration(
            self.fcf,
            u64::from(self.config.loop_config.forward_passes),
            self.training_ctx.node_graph,
            self.training_ctx.horizon.num_stages(),
        );
        // Growth-only: a pool `grow_pools_for_next_iteration` just grew may now
        // exceed what the DCS scratch covers; re-reserve before the next
        // sweep touches it (never inside the sweep itself).
        let max_pool_capacity = self.fcf.pools.iter().map(|p| p.capacity).max().unwrap_or(0);
        self.fwd_pool
            .reserve_dcs_scratch(self.training_ctx.state.n_state, max_pool_capacity);

        let (lb, lb_lp_solves, lb_wall_ms, lb_solve_time_ms) = self.run_lower_bound(iteration)?;

        let local = StopInputs {
            shutdown: self.runtime.shutdown_requested(),
            wall_time_seconds: self.results.start_time.elapsed().as_secs_f64(),
        };
        let agreed = agree_stop_inputs(local, self.comm).map_err(SddpError::Communication)?;
        if let Some(source) = agreed.shutdown {
            self.convergence_monitor.set_shutdown(source);
        }
        let decision = self
            .convergence_monitor
            .update(lb, &sync_result, agreed.wall_time_seconds);

        self.results.final_lb = self.convergence_monitor.lower_bound();
        self.results.final_ub = self.convergence_monitor.upper_bound();
        self.results.final_ub_std = self.convergence_monitor.upper_bound_std();
        self.results.final_gap = self.convergence_monitor.gap();

        emit(
            self.runtime.event_sender(),
            TrainingEvent::ConvergenceUpdate {
                iteration,
                lower_bound: self.results.final_lb,
                upper_bound: self.results.final_ub,
                upper_bound_std: self.results.final_ub_std,
                gap: self.results.final_gap,
            },
        );

        #[allow(clippy::cast_possible_truncation)]
        let wall_time_ms = self.results.start_time.elapsed().as_millis() as u64;
        #[allow(clippy::cast_possible_truncation)]
        let iteration_time_ms = iter_start.elapsed().as_millis() as u64;

        // Sum across ranks: forward/backward solves partition across ranks and the
        // lower bound runs only on rank 0, so the per-rank sum scales with rank
        // count while the global total is rank-count invariant.
        let lp_solves = {
            #[allow(clippy::cast_precision_loss)]
            let local =
                [(forward_result.lp_solves + backward_result.lp_solves + lb_lp_solves) as f64];
            let mut global = [0.0_f64; 1];
            self.comm
                .allreduce(&local, &mut global, ReduceOp::Sum)
                .map_err(SddpError::Communication)?;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                global[0].round() as u64
            }
        };

        emit(
            self.runtime.event_sender(),
            TrainingEvent::IterationSummary {
                iteration,
                lower_bound: self.results.final_lb,
                upper_bound: self.results.final_ub,
                gap: self.results.final_gap,
                wall_time_ms,
                iteration_time_ms,
                forward_ms: forward_result.elapsed_ms,
                backward_ms: backward_result.elapsed_ms,
                lp_solves,
                solve_time_ms: fwd_solve_time_ms + bwd_solve_time_ms + lb_solve_time_ms,
                lower_bound_eval_ms: lb_wall_ms,
                fwd_setup_time_ms: forward_result.setup_time_ms,
                fwd_load_imbalance_ms: forward_result.load_imbalance_ms,
                fwd_scheduling_overhead_ms: forward_result.scheduling_overhead_ms,
                rows_in_lp_sum,
                rows_in_lp_count,
                rows_in_lp_max,
            },
        );

        self.results.completed_iterations = iteration;

        if let Some(reason) = decision.termination_reason() {
            self.results.stop_decision = decision;
            self.results.termination_reason = reason.to_string();

            if decision.ended_by_shutdown() {
                return Ok(IterationOutcome::Shutdown);
            }
            return Ok(IterationOutcome::Converged);
        }

        self.write_periodic_checkpoint(iteration)?;

        Ok(IterationOutcome::Continue)
    }

    /// Rank 0 writes the periodic checkpoint when the schedule fires at
    /// `iteration`; every rank then agrees on the write's outcome, so a failed
    /// write ends training on every rank at this iteration.
    ///
    /// # Errors
    ///
    /// [`SddpError::CheckpointWrite`] on rank 0 when the write fails, the peer
    /// failure from [`reconcile_error_flag`] on every other rank.
    fn write_periodic_checkpoint(&mut self, iteration: u64) -> Result<(), SddpError> {
        let Some(periodic) = self.runtime.periodic_checkpoint() else {
            return Ok(());
        };
        if !periodic.fires_at(iteration) {
            return Ok(());
        }

        let start = Instant::now();
        let local = if self.comm.rank() == 0 {
            let basis_cache = rank_local_basis_cache(&self.basis_store);
            periodic
                .write(
                    self.fcf,
                    self.training_ctx.node_graph,
                    CheckpointState {
                        iterations: iteration,
                        final_lb: self.results.final_lb,
                        final_ub: self.results.final_ub,
                        basis_cache: &basis_cache,
                        visited_archive: self.visited_archive.as_ref(),
                        lower_bound_history: self.convergence_monitor.lower_bound_history(),
                    },
                )
                .map_err(|source| SddpError::CheckpointWrite { iteration, source })
        } else {
            Ok(())
        };
        let elapsed_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
        reconcile_error_flag(local, self.comm, &mut self.fwd_state.reconcile_scratch)?;

        if self.comm.rank() == 0 {
            emit(
                self.runtime.event_sender(),
                TrainingEvent::CheckpointComplete {
                    iteration,
                    checkpoint_path: periodic.policy_dir().display().to_string(),
                    elapsed_ms,
                },
            );
        }
        Ok(())
    }

    /// Assemble and return the successful `TrainingOutcome`.
    ///
    /// # Errors
    ///
    /// Returns `SddpError::Communication` if `broadcast_basis_cache` fails.
    pub(crate) fn finalize(mut self) -> Result<TrainingOutcome, SddpError> {
        // Reconcile finalize arrival before the basis-cache broadcast so
        // broadcast_basis_cache is entered by all ranks or none: a peer that failed
        // makes this clean rank take the same no-broadcast exit in lockstep, rather
        // than block alone in the broadcast.
        if let Err(peer_err) =
            reconcile_error_flag(Ok(()), self.comm, &mut self.fwd_state.reconcile_scratch)
        {
            return Ok(self.finalize_without_broadcast(peer_err, "error"));
        }

        #[allow(clippy::cast_possible_truncation)]
        let total_time_ms = (self.results.start_time.elapsed().as_millis() as u64).max(1);

        let frozen_templates = self.scratch.frozen_templates;
        let visited_archive = self.visited_archive;
        let TrainingResults {
            final_lb,
            final_ub,
            final_ub_std,
            final_gap,
            completed_iterations,
            termination_reason,
            stop_decision,
            solver_stats_log,
            ..
        } = self.results;

        #[allow(clippy::cast_possible_truncation)]
        emit(
            self.runtime.event_sender(),
            TrainingEvent::TrainingFinished {
                reason: termination_reason.clone(),
                iterations: completed_iterations,
                final_lb,
                final_ub,
                total_time_ms,
                total_rows: self.fcf.total_active_cuts() as u64,
            },
        );

        let basis_cache = broadcast_basis_cache(&self.basis_store, self.comm)?;

        let mut result = TrainingResult::new(
            final_lb,
            final_ub,
            final_ub_std,
            final_gap,
            completed_iterations,
            termination_reason,
            total_time_ms,
            basis_cache,
            solver_stats_log,
            visited_archive,
            Some(frozen_templates),
        );
        result.stop_decision = stop_decision;
        result.lower_bound_history =
            committed_lower_bound_history(&self.convergence_monitor, completed_iterations).to_vec();

        Ok(TrainingOutcome {
            result,
            error: None,
        })
    }

    /// Emit `TrainingFinished` with `reason = "error"` and return a partial
    /// `TrainingOutcome` carrying the original error, taking the coordinated
    /// no-broadcast exit so no rank blocks in `broadcast_basis_cache`.
    pub(crate) fn finalize_with_error(mut self, err: SddpError) -> TrainingOutcome {
        // Announce this rank's failure to peers still in the clean `finalize` so they
        // skip their basis-cache broadcast in lockstep; reconcile over a local Err
        // hands this rank's own error back (single-rank: the same error unchanged).
        let err = reconcile_error_flag(Err(err), self.comm, &mut self.fwd_state.reconcile_scratch)
            .err()
            .unwrap_or_else(|| {
                SddpError::Communication(CollectiveFailed {
                    operation: "reconcile_error_flag",
                    mpi_error_code: 0,
                    message: "reconcile over a local failure unexpectedly reported agreement"
                        .to_string(),
                })
            });
        self.finalize_without_broadcast(err, "error")
    }

    /// Build the error-exit `TrainingOutcome` WITHOUT entering
    /// `broadcast_basis_cache`. Used once the ranks did not all agree to broadcast
    /// (a rank-local or peer failure), so the basis cache is empty — no collective
    /// is safe on this path. The caller must have already reconciled the failure
    /// across ranks so every rank takes this exit in lockstep.
    fn finalize_without_broadcast(self, err: SddpError, reason: &str) -> TrainingOutcome {
        let frozen_templates = self.scratch.frozen_templates;
        let visited_archive = self.visited_archive;
        let TrainingResults {
            final_lb,
            final_ub,
            final_ub_std,
            final_gap,
            completed_iterations,
            solver_stats_log,
            start_time,
            ..
        } = self.results;

        #[allow(clippy::cast_possible_truncation)]
        let total_time_ms = (start_time.elapsed().as_millis() as u64).max(1);

        #[allow(clippy::cast_possible_truncation)]
        emit(
            self.runtime.event_sender(),
            TrainingEvent::TrainingFinished {
                reason: reason.to_string(),
                iterations: completed_iterations,
                final_lb,
                final_ub,
                total_time_ms,
                total_rows: self.fcf.total_active_cuts() as u64,
            },
        );

        let mut result = TrainingResult::new(
            final_lb,
            final_ub,
            final_ub_std,
            final_gap,
            completed_iterations,
            reason.to_string(),
            total_time_ms,
            Vec::new(),
            solver_stats_log,
            visited_archive,
            Some(frozen_templates),
        );
        result.lower_bound_history =
            committed_lower_bound_history(&self.convergence_monitor, completed_iterations).to_vec();

        TrainingOutcome {
            result,
            error: Some(err),
        }
    }

    // ── Private phase helpers ──────────────────────────────────────────────

    /// Run the forward pass and forward synchronisation for one iteration.
    fn run_forward_phase(
        &mut self,
        iteration: u64,
    ) -> Result<(ForwardResult, SyncResult, f64), SddpError> {
        let fwd_stats_before = aggregate_solver_statistics(
            self.fwd_pool
                .workspaces
                .iter()
                .map(|w| w.solver.statistics()),
        );

        // Borrow fwd_state alone so the remaining fields can be passed without a
        // whole-struct borrow conflict.
        let fwd = &mut self.fwd_state;
        let mut inputs = ForwardPassInputs::from_session_fields(
            &mut self.fwd_pool,
            &mut self.basis_store,
            self.stage_ctx,
            &mut self.scratch,
            self.fcf,
            self.training_ctx,
            &self.ranks,
            &self.runtime,
            iteration,
        );
        // Reconcile the rank-local forward result before the forward phase's first
        // collective (the stage-stats allreduce and sync_forward's allgatherv): a
        // solve failure on any rank makes every rank return Err here and break
        // toward a coordinated finalize, so no rank enters those collectives while a
        // peer has skipped them.
        let forward_local = fwd.run(&mut inputs);
        let forward_result =
            reconcile_result(forward_local, self.comm, &mut fwd.reconcile_scratch)?;

        let fwd_solve_time_ms = {
            let fwd_stats_after = aggregate_solver_statistics(
                self.fwd_pool
                    .workspaces
                    .iter()
                    .map(|w| w.solver.statistics()),
            );
            SolverStatsDelta::from_snapshots(&fwd_stats_before, &fwd_stats_after).solve_time_ms
        };

        // Aggregate per-stage forward stats across ranks: `write_training_outputs`
        // is rank-0-gated, so without this only rank 0's workers would reach
        // `iterations.parquet` and `lp_solves` would understate the global count.
        // `retry_level_histogram` is not packed here, so it is absent from the
        // aggregated forward rows (backward keeps it via per-worker rows).
        let num_stages = forward_result.stage_stats.len();
        self.scratch.fwd_stats_unpacked.clear();
        if num_stages != 0 {
            let n_packed = num_stages * SOLVER_STATS_DELTA_SCALAR_FIELDS;
            self.scratch.fwd_stats_pack_local.clear();
            self.scratch.fwd_stats_pack_local.resize(n_packed, 0.0);
            for (i, delta) in forward_result.stage_stats.iter().enumerate() {
                let packed = pack_delta_scalars(delta);
                self.scratch.fwd_stats_pack_local[i * SOLVER_STATS_DELTA_SCALAR_FIELDS..]
                    [..SOLVER_STATS_DELTA_SCALAR_FIELDS]
                    .copy_from_slice(&packed);
            }
            self.scratch.fwd_stats_pack_global.clear();
            self.scratch.fwd_stats_pack_global.resize(n_packed, 0.0);
            self.comm
                .allreduce(
                    &self.scratch.fwd_stats_pack_local,
                    &mut self.scratch.fwd_stats_pack_global,
                    ReduceOp::Sum,
                )
                .map_err(SddpError::Communication)?;
            let mut unpacked = std::mem::take(&mut self.scratch.fwd_stats_unpacked);
            for chunk in self
                .scratch
                .fwd_stats_pack_global
                .chunks_exact(SOLVER_STATS_DELTA_SCALAR_FIELDS)
            {
                #[allow(clippy::expect_used)]
                let arr: [f64; SOLVER_STATS_DELTA_SCALAR_FIELDS] = chunk.try_into().expect(
                    "chunks_exact yields slices of exactly SOLVER_STATS_DELTA_SCALAR_FIELDS",
                );
                unpacked.push(unpack_delta_scalars(&arr));
            }
            self.scratch.fwd_stats_unpacked = unpacked;
        }

        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        for (stage_idx, delta) in self.scratch.fwd_stats_unpacked.iter().enumerate() {
            let mut entry = SolverStatsDelta::default();
            delta.clone_into_reuse(&mut entry);
            self.results
                .solver_stats_log
                .push(SolverStatsLogEntry::from_raw(
                    iteration,
                    "forward",
                    self.stage_ctx.study_stage_ids.get(stage_idx).copied(),
                    -1,
                    self.ranks.fwd_rank,
                    -1,
                    entry,
                ));
        }

        let local_n = forward_result.scenario_costs.len();
        let local_cost_sum: f64 = forward_result.scenario_costs.iter().sum();
        emit(
            self.runtime.event_sender(),
            TrainingEvent::ForwardPassComplete {
                iteration,
                scenarios: self.config.loop_config.forward_passes,
                #[allow(clippy::cast_precision_loss)]
                ub_mean: if local_n > 0 {
                    local_cost_sum / local_n as f64
                } else {
                    0.0
                },
                ub_std: 0.0,
                elapsed_ms: forward_result.elapsed_ms,
            },
        );

        let sync_result = self.sync_forward_bound(&forward_result)?;

        emit(
            self.runtime.event_sender(),
            TrainingEvent::ForwardSyncComplete {
                iteration,
                global_ub_mean: sync_result.global_ub_mean,
                global_ub_std: sync_result.global_ub_std,
                sync_time_ms: sync_result.sync_time_ms,
            },
        );

        Ok((forward_result, sync_result, fwd_solve_time_ms))
    }

    /// Aggregate this rank's forward-pass upper bound across ranks, selecting the
    /// estimator from the traversal and effective risk measure: `Statistical` for
    /// a sampled forward, the risk-neutral `Exact` `Σ w·c` or the nested-risk
    /// recursion for an enumerated one. `sync_forward` owns all three estimators;
    /// weights/costs are read off the `Traversal` resolved once at training start.
    ///
    /// # Errors
    ///
    /// Propagates the `allgatherv` failure from [`sync_forward`].
    fn sync_forward_bound(
        &mut self,
        forward_result: &ForwardResult,
    ) -> Result<SyncResult, SddpError> {
        let global_n = self.ranks.num_total_forward_passes;
        if let Traversal::Enumerated(plan) = self.fwd_state.traversal() {
            debug_assert_eq!(
                plan.paths.weight.len(),
                global_n,
                "enumerated path count must equal the resolved forward-pass count"
            );
            let ub_measure = uniform_effective_measure(&self.config.cut_management.risk_measures)
                .unwrap_or(RiskMeasure::Expectation);
            if ub_measure == RiskMeasure::Expectation {
                self.scratch.ub_path_weights.clear();
                self.scratch
                    .ub_path_weights
                    .extend_from_slice(&plan.paths.weight);
                sync_forward(
                    forward_result,
                    self.comm,
                    global_n,
                    ForwardBound::Exact {
                        path_weights: &self.scratch.ub_path_weights,
                    },
                )
            } else {
                // Uniform effective CVaR: gather per-path per-stage costs and apply
                // the nested risk recursion (the end-of-horizon `Σ w·c` cannot
                // represent a nested measure — it can fall below the nested LB).
                let num_stages = self.training_ctx.horizon.num_stages();
                let local_n = forward_result.scenario_costs.len();
                self.scratch.ub_stage_costs.clear();
                for i in 0..local_n * num_stages {
                    self.scratch
                        .ub_stage_costs
                        .push(self.scratch.records[i].stage_cost);
                }
                sync_forward(
                    forward_result,
                    self.comm,
                    global_n,
                    ForwardBound::NestedRisk {
                        path_stage_costs: &self.scratch.ub_stage_costs,
                        topology: &plan.nested_ub_topology,
                        cumulative_discounts: self.stage_ctx.cumulative_discount_factors,
                        risk_measure: ub_measure,
                        num_stages,
                        scratch: &mut self.scratch.nested_ub,
                    },
                )
            }
        } else {
            sync_forward(
                forward_result,
                self.comm,
                global_n,
                ForwardBound::Statistical,
            )
        }
    }

    /// Run the backward pass for one iteration.
    // Rationale: the `i32::try_from(*omega).expect(...)` cannot fire — opening
    // indices derive from `branching_factor: u16` and stay well below `i32::MAX`.
    #[allow(clippy::expect_used)]
    fn run_backward_phase(&mut self, iteration: u64) -> Result<(BackwardResult, f64), SddpError> {
        // Borrow bwd_state and each disjoint `self.scratch` sub-field separately
        // so they can be co-borrowed mutably without a whole-struct conflict.
        let bwd = &mut self.bwd_state;
        let mut inputs = BackwardPassInputs::from_session_fields(
            &mut self.fwd_pool,
            &mut self.basis_store,
            self.stage_ctx,
            &self.scratch.frozen_templates,
            &mut self.scratch.cut_batches,
            &self.scratch.records,
            self.fcf,
            &mut self.exchange_bufs,
            &mut self.cut_sync_bufs,
            &mut self.visited_archive,
            self.training_ctx,
            self.comm,
            &self.config.cut_management,
            &self.ranks,
            &self.runtime,
            iteration,
            self.fwd_state.traversal(),
            self.fwd_state.enumerated_state(),
        );
        let backward_result = bwd.run(&mut inputs)?;

        let bwd_solve_time_ms = {
            let agg = SolverStatsDelta::aggregate(
                backward_result
                    .stage_stats
                    .iter()
                    .flat_map(|(_, entries)| entries.iter().map(|(_, _, _, d)| d)),
            );
            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            for (stage_idx, entries) in &backward_result.stage_stats {
                for (rank, worker_id, omega, delta) in entries {
                    let mut entry = SolverStatsDelta::default();
                    delta.clone_into_reuse(&mut entry);
                    self.results
                        .solver_stats_log
                        .push(SolverStatsLogEntry::from_raw(
                            iteration,
                            "backward",
                            self.stage_ctx.study_stage_ids.get(*stage_idx).copied(),
                            i32::try_from(*omega)
                                .expect("opening index is bounded well below i32::MAX"),
                            *rank,
                            *worker_id,
                            entry,
                        ));
                }
            }
            agg.solve_time_ms
        };

        #[allow(clippy::cast_possible_truncation)]
        emit(
            self.runtime.event_sender(),
            TrainingEvent::BackwardPassComplete {
                iteration,
                rows_generated: backward_result.cuts_generated as u32,
                stages_processed: self.training_ctx.horizon.num_stages().saturating_sub(1) as u32,
                elapsed_ms: backward_result.elapsed_ms,
                state_exchange_time_ms: backward_result.state_exchange_time_ms,
                row_batch_build_time_ms: backward_result.cut_batch_build_time_ms,
                setup_time_ms: backward_result.setup_time_ms,
                load_imbalance_ms: backward_result.load_imbalance_ms,
                scheduling_overhead_ms: backward_result.scheduling_overhead_ms,
            },
        );

        #[allow(clippy::cast_possible_truncation)]
        emit(
            self.runtime.event_sender(),
            TrainingEvent::PolicySyncComplete {
                iteration,
                rows_distributed: backward_result.cuts_generated as u32,
                rows_active: self.fcf.total_active_cuts() as u32,
                rows_removed: 0,
                sync_time_ms: backward_result.cut_sync_time_ms,
            },
        );

        Ok((backward_result, bwd_solve_time_ms))
    }

    /// Apply cut selection, budget enforcement, bitmap shift, and template freeze.
    ///
    /// All operations are O(active cuts) and perform no heap allocation when the
    /// cut pools have not grown since the previous iteration.
    ///
    /// # Errors
    ///
    /// Returns [`SddpError::Validation`] if stage 0 carries no alive node.
    // Rationale: the phases mutate `&mut self` and each reads state the prior
    // phase wrote, so splitting into helpers would pass every field individually.
    #[allow(clippy::too_many_lines)]
    fn run_cut_management(&mut self, iteration: u64) -> Result<(), SddpError> {
        // `sel_state` is `Some` only when strategy-based selection ran;
        // `record_by_pool` (built in the same block) maps a pool id to its
        // `per_stage` record index for the budget back-annotation below.
        let mut sel_state: Option<(Vec<StageRowSelectionRecord>, u32, u64, u32)> = None;
        let mut record_by_pool: Option<Vec<Option<usize>>> = None;

        if let Some(strategy) = self.config.cut_management.cut_selection.as_ref()
            && strategy.should_run(iteration)
        {
            let sel_start = Instant::now();
            let num_sel_stages = self.training_ctx.horizon.num_stages().saturating_sub(1);
            let mut rows_deactivated = 0u32;
            let mut per_stage = Vec::with_capacity(num_sel_stages);

            let node_graph = self.training_ctx.node_graph;
            // Root pool resolved by STAGE (the sole stage-0 node), never by array
            // position: nodes[0] is the smallest-id node, not necessarily the
            // root on a branching graph (the same resolution the LB path uses).
            let root_node = node_graph.frontier_node(StageIdx(0)).ok_or_else(|| {
                SddpError::Validation("training session: stage 0 carries no alive node".to_string())
            })?;
            let root_pool = node_graph.nodes[root_node].pool_id;

            // Selection covers interior cut-generating nodes only. The root's
            // cuts are never a backward-pass successor (activity never updated);
            // terminal leaves receive no cuts. The root (the sole stage-0 alive
            // node) is recorded here; the loop below scores each interior
            // cut-generating node against ITS OWN visited region.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                let root_cut_pool = &self.fcf.pools[root_pool];
                let root_active = root_cut_pool.active_count() as u32;
                per_stage.push(StageRowSelectionRecord {
                    // Domain stage_id of the root (study-stage position 0).
                    stage: self.stage_ctx.study_stage_ids.first().copied().unwrap_or(0) as u32,
                    rows_populated: root_cut_pool.populated() as u32,
                    rows_active_before: root_active,
                    rows_deactivated: 0,
                    rows_reactivated: 0,
                    rows_active_after: root_active,
                    selection_time_ms: 0.0,
                    budget_evicted: None,
                    active_after_budget: None,
                    rows_in_lp: root_cut_pool.cuts_in_lp() as u32,
                });
            }

            let archive_ref = self.visited_archive.as_ref();
            // Interior cut-generating nodes in canonical order. A cut-generating
            // (non-leaf) node always owns its own pool, so scoring by node is
            // per-pool; the visited states are read by NODE position (siblings at
            // one stage own distinct visited regions). On a chain node position
            // equals stage, reproducing the former 1..T-2 stage loop.
            let interior_nodes = &self.interior_cut_nodes;
            let pools = &self.fcf.pools;
            let cut_state_layouts = self.training_ctx.cut_state_layouts;
            let n_global = archive_ref.map_or(0, VisitedStatesArchive::packing_stride);
            // Sequential (not `into_par_iter`): the m-block kernel already
            // saturates the cores via its inner parallelism.
            let mut scratch = std::mem::take(&mut self.cut_selection_state_scratch);
            let mut deactivations: Vec<(usize, usize, CutActivityUpdates, f64)> =
                Vec::with_capacity(interior_nodes.len());
            #[allow(clippy::cast_possible_truncation)]
            for &node_pos in interior_nodes {
                let node = &node_graph.nodes[node_pos];
                let pool_id = node.pool_id;
                let pool = &pools[pool_id];
                let proj = &cut_state_layouts[pool_id];
                let n_slots = proj.n_slots();
                let n_trials = archive_ref.map_or(0, |a| a.count(node_pos));
                let global_states =
                    archive_ref.map_or(&[] as &[f64], |a| a.states_for_node(node_pos));
                // Project the StateDim-packed archive into the pool's projected
                // slot space through its own CutStateProjection (identity when
                // n_slots == n_global). A positional prefix is wrong — a reduced
                // pool's slots are not the first n_slots StateDims.
                scratch.clear();
                scratch.reserve(n_trials * n_slots);
                for m in 0..n_trials {
                    let base = m * n_global;
                    for s in 0..n_slots {
                        scratch.push(
                            global_states[base + proj.global_state_index(CutSlot::new(s)).get()],
                        );
                    }
                }
                let start = Instant::now();
                let deact = strategy.select_for_stage(
                    pool,
                    &scratch,
                    n_trials,
                    iteration,
                    node.stage.0 as u32,
                );
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                deactivations.push((node.stage.0, pool_id, deact, elapsed_ms));
            }
            self.cut_selection_state_scratch = scratch;

            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            for (stage, pool_id, deact, stage_sel_time_ms) in deactivations {
                let pool = &self.fcf.pools[pool_id];
                let populated = pool.populated() as u32;
                let active_before = pool.active_count() as u32;
                let n_deact = deact.updates.len() as u32;
                let n_reactivated = deact.reactivations.len() as u32;
                rows_deactivated += n_deact;

                self.fcf.pools[pool_id].apply_updates(&deact);

                let active_after = self.fcf.pools[pool_id].active_count() as u32;
                let rows_in_lp = self.fcf.pools[pool_id].cuts_in_lp() as u32;
                per_stage.push(StageRowSelectionRecord {
                    // Domain stage_id by position from the ordered study ids;
                    // falls back to the positional index for a short stub context.
                    stage: self
                        .stage_ctx
                        .study_stage_ids
                        .get(stage)
                        .copied()
                        .unwrap_or_else(|| i32::try_from(stage).unwrap_or(i32::MAX))
                        as u32,
                    rows_populated: populated,
                    rows_active_before: active_before,
                    rows_deactivated: n_deact,
                    rows_reactivated: n_reactivated,
                    rows_active_after: active_after,
                    selection_time_ms: stage_sel_time_ms,
                    budget_evicted: None,
                    active_after_budget: None,
                    rows_in_lp,
                });
            }

            // Trim AFTER the application loop so the immutable borrow held by
            // `deactivations` is released, and AFTER selection so the kernel sees
            // the full ~2x-peak archive before shrinking to the steady-state
            // bound documented on [`VisitedStatesArchive::trim_to_window`].
            if let Some(ref mut archive) = self.visited_archive {
                let check_freq = match strategy {
                    Level1 {
                        check_frequency, ..
                    }
                    | Lml1 {
                        check_frequency, ..
                    }
                    | Dominated {
                        check_frequency, ..
                    } => *check_frequency,
                    Dynamic { .. } => {
                        unreachable!(
                            "DCS never runs as a periodic pool pass; should_run is always false"
                        )
                    }
                };
                archive.trim_to_window(check_freq);
            }

            #[allow(clippy::cast_possible_truncation)]
            let selection_time_ms = sel_start.elapsed().as_millis() as u64;
            #[allow(clippy::cast_possible_truncation)]
            let stages_processed_sel = num_sel_stages as u32;

            record_by_pool = Some(selection_record_index_by_pool(
                node_graph,
                root_pool,
                interior_nodes,
            ));

            sel_state = Some((
                per_stage,
                rows_deactivated,
                selection_time_ms,
                stages_processed_sel,
            ));
        }

        // Budget is a hard cap: enforced every iteration when set, never gated
        // by `check_frequency` like selection.
        if let Some(b) = self.config.cut_management.budget {
            let budget_start = Instant::now();
            let mut total_evicted = 0u32;
            // Budget is a per-POOL cap: enforce over every pool, never
            // `nodes[stage].pool_id` (a stage-as-node-position read that misses
            // sibling pools on a branching graph). On a chain `pool_id == stage`.
            for pool_id in 0..self.fcf.pools.len() {
                #[allow(clippy::cast_possible_truncation)]
                let result = self.fcf.pools[pool_id].enforce_budget(
                    b,
                    iteration,
                    self.config.loop_config.forward_passes,
                );
                total_evicted += result.evicted_count;
                // Annotate the pool's OWN selection record via `record_by_pool`,
                // never `per_stage[pool_id]`: the record index equals the pool id
                // only on a chain (see `selection_record_index_by_pool`).
                if let Some((ref mut per_stage, _, _, _)) = sel_state
                    && let Some(rec_idx) = record_by_pool
                        .as_ref()
                        .and_then(|m| m.get(pool_id).copied().flatten())
                    && let Some(rec) = per_stage.get_mut(rec_idx)
                {
                    rec.budget_evicted = Some(result.evicted_count);
                    rec.active_after_budget = Some(result.active_after);
                }
            }
            #[allow(clippy::cast_possible_truncation)]
            let enforcement_time_ms = budget_start.elapsed().as_millis() as u64;
            emit(
                self.runtime.event_sender(),
                #[allow(clippy::cast_possible_truncation)]
                TrainingEvent::PolicyBudgetEnforcementComplete {
                    iteration,
                    rows_evicted: total_evicted,
                    stages_processed: self.training_ctx.horizon.num_stages() as u32,
                    enforcement_time_ms,
                },
            );
        }

        // Emit after the budget loop has annotated every per-stage record.
        if let Some((per_stage, rows_deactivated, selection_time_ms, stages_processed)) = sel_state
        {
            emit(
                self.runtime.event_sender(),
                TrainingEvent::PolicySelectionComplete {
                    iteration,
                    rows_deactivated,
                    stages_processed,
                    selection_time_ms,
                    allgatherv_time_ms: 0,
                    per_stage,
                },
            );
        }

        let freeze_start = Instant::now();
        let total_rows_frozen = self.freeze_active_cuts_into_templates(true);
        #[allow(clippy::cast_possible_truncation)]
        let freeze_time_ms = freeze_start.elapsed().as_millis() as u64;
        emit(
            self.runtime.event_sender(),
            #[allow(clippy::cast_possible_truncation)]
            TrainingEvent::PolicyTemplateFreezeComplete {
                iteration,
                stages_processed: self.training_ctx.horizon.num_stages() as u32,
                total_rows_frozen,
                freeze_time_ms,
            },
        );
        Ok(())
    }

    /// Rebuild every pool's frozen template from the current active cut set,
    /// returning the total number of cut rows frozen.
    ///
    /// Each `frozen_templates[p]` becomes pool `p`'s base stage template
    /// (`templates[pool_stage[p]]`) plus one structural row per active cut in
    /// pool `p` (`active_cuts()` order). Indexing the overlay by POOL, not stage,
    /// is the whole correction: a branching stage holds several nodes with
    /// distinct pools, so freezing one pool per stage would bake one node's cuts
    /// into a sibling's (or the terminal leaf's) LP. With no active cuts (a fresh
    /// start) every batch is empty and the freeze is a structural copy of the
    /// base template — identical to the pre-freeze done in `IterationScratch::new`.
    ///
    /// `skip_static_terminal` skips any pool whose base stage is the terminal
    /// stage (`pool_stage[p] == StageIdx(num_stages - 1)`): a terminal leaf never
    /// adds or removes a cut, so its template is baked once by
    /// [`Self::prime_frozen_templates`] (`false`) and left untouched by every
    /// per-iteration refreeze (`true`). Skipping is deliberately confined to this
    /// flag, never a `pool_stage`-derived early return baked into the loop bounds,
    /// so the priming call still bakes the terminal pool.
    ///
    /// Deliberately left unoptimized: the refreeze is quadratic in the active-cut
    /// count only in the no-cut-selection default, which production never runs at
    /// scale. An append-only fast path would fire only there, so the full refreeze
    /// is kept.
    fn freeze_active_cuts_into_templates(&mut self, skip_static_terminal: bool) -> u64 {
        let mut total_rows_frozen: u64 = 0;
        let state = self.training_ctx.state;
        let node_graph = self.training_ctx.node_graph;
        let num_stages = self.training_ctx.horizon.num_stages();
        let terminal_stage = (num_stages > 0).then(|| StageIdx(num_stages - 1));
        for p in 0..node_graph.n_pools {
            let t = node_graph.pool_stage[p];
            let is_terminal = terminal_stage == Some(t);
            if skip_static_terminal && is_terminal {
                continue;
            }
            build_cut_row_batch_into(
                &mut self.scratch.freeze_row_batches[p],
                self.fcf,
                p,
                state,
                &self.training_ctx.cut_state_layouts[p],
                &self.stage_ctx.template(t).col_scale,
            );
            #[allow(clippy::cast_possible_truncation)]
            {
                total_rows_frozen += self.scratch.freeze_row_batches[p].num_rows as u64;
            }
            freeze_rows_into_template(
                self.stage_ctx.template(t),
                &self.scratch.freeze_row_batches[p],
                &mut self.scratch.frozen_templates[p],
                &mut self.scratch.freeze_scratch,
            );
        }
        total_rows_frozen
    }

    /// Seed the per-scenario basis store from a checkpoint's stored bases
    /// before the first training iteration runs.
    ///
    /// `cache` carries one [`CapturedBasis`]
    /// per canonical node (built by
    /// [`build_basis_cache_from_checkpoint`](crate::build_basis_cache_from_checkpoint)).
    /// The checkpoint holds a single basis per node; the forward pass keeps one
    /// per `(worker, node)`, so each node's basis is replicated across every
    /// worker for iteration 1's warm-start. `reconstruct_basis` reconciles the
    /// stored cut rows against the current active set by slot identity, so a
    /// seeded basis stays correct even if cut selection diverges. Seeds every
    /// node (`num_nodes`), never truncated at `num_stages`, so a branching
    /// graph's leaves warm-start too. No-op for a fresh start (no cache).
    pub(crate) fn seed_basis_store(&mut self, cache: &[Option<CapturedBasis>]) {
        let max_local_fwd = self.ranks.max_local_fwd;
        let num_nodes = self.basis_store.num_nodes();
        for (t, slot) in cache.iter().enumerate().take(num_nodes) {
            let Some(captured) = slot else { continue };
            for scenario in 0..max_local_fwd {
                *self.basis_store.get_mut(scenario, NodePos(t)) = Some(captured.clone());
            }
        }
    }

    /// Freeze the warm-start / resume pre-loaded cuts into the stage templates
    /// before the first training iteration runs.
    ///
    /// `IterationScratch::new` pre-freezes with an empty cut batch, and iteration
    /// 1's passes read `scratch.frozen_templates` before `run_cut_management`
    /// refreezes; without this, the first post-resume iteration would solve a
    /// cut-less, myopic policy. No-op for a fresh start (no active cuts).
    pub(crate) fn prime_frozen_templates(&mut self) {
        if self.fcf.total_active_cuts() > 0 {
            let _ = self.freeze_active_cuts_into_templates(false);
        }
    }

    /// Evaluate the lower bound and push the solver stats entry.
    fn run_lower_bound(&mut self, iteration: u64) -> Result<(f64, u64, u64, f64), SddpError> {
        let lb_wall_start = Instant::now();
        let lb_stats_before = self.solver.statistics();

        let mut lb_bundle = LbEvalScratchBundle::from_scratch_fields(
            &mut self.scratch.patch_buf,
            &mut self.scratch.lb_cut_batch,
            Some(&mut self.scratch.lb_cut_row_map),
            &mut self.scratch.lb_noise_scratch,
            &mut self.scratch.lb_scratch,
        );
        let lb = evaluate_lower_bound(
            self.solver,
            self.fcf,
            self.stage_ctx,
            self.training_ctx,
            &self.config.cut_management.risk_measures[0],
            &mut lb_bundle,
            self.comm,
        )?;

        let lb_stats_after = self.solver.statistics();
        let lb_lp_solves = lb_stats_after.solve_count - lb_stats_before.solve_count;
        let lb_delta = SolverStatsDelta::from_snapshots(&lb_stats_before, &lb_stats_after);
        let lb_solve_time_ms = lb_delta.solve_time_ms;
        self.results
            .solver_stats_log
            .push(SolverStatsLogEntry::from_raw(
                iteration,
                "lower_bound",
                None,
                -1,
                self.ranks.fwd_rank,
                -1,
                lb_delta,
            ));
        #[allow(clippy::cast_possible_truncation)]
        let lb_wall_ms = lb_wall_start.elapsed().as_millis() as u64;

        Ok((lb, lb_lp_solves, lb_wall_ms, lb_solve_time_ms))
    }
}

/// The monitor's lower-bound series without the trailing entries of iterations
/// the run did not commit: a failed collective after the stop decision leaves
/// the monitor one iteration ahead of `completed_iterations`.
fn committed_lower_bound_history(
    monitor: &ConvergenceMonitor,
    completed_iterations: u64,
) -> &[f64] {
    let history = monitor.lower_bound_history();
    let uncommitted =
        usize::try_from(monitor.iteration_count() - completed_iterations).unwrap_or(usize::MAX);
    &history[..history.len().saturating_sub(uncommitted)]
}

/// Pool → per-iteration selection-record index, in the order
/// [`TrainingSession::run_cut_management`] emits `per_stage`: the root pool's
/// record first (index 0), then each interior cut-generating node's pool in
/// canonical order (`interior_nodes`). The shared leaf pool (no selection
/// record) stays `None`. Budget enforcement iterates every pool but must
/// annotate each pool's record through THIS map, never by pool id used as a
/// `per_stage` index — on a branching graph whose root is not the smallest-id
/// node the two differ, so eviction stats would land on a sibling's record. On
/// a chain `pool_id` equals the record index, so the map is the identity.
fn selection_record_index_by_pool(
    node_graph: &NodeGraph,
    root_pool: usize,
    interior_nodes: &[NodePos],
) -> Vec<Option<usize>> {
    let mut by_pool = vec![None; node_graph.n_pools];
    by_pool[root_pool] = Some(0);
    for (record_idx, &pos) in interior_nodes.iter().enumerate() {
        by_pool[node_graph.nodes[pos].pool_id] = Some(1 + record_idx);
    }
    by_pool
}

/// Between-iteration capacity growth (reserved seam): doubles a pool's
/// capacity via [`crate::cut::CutPool::grow`] (`Vec::resize`, position-stable
/// — the append-only/slot-identity contract survives) when
/// `realized_visits` exceeds its remaining free slots. Runs once per
/// iteration from [`TrainingSession::run_iteration`], never inside the
/// forward/backward sweep. A chain pool's construction-time floor already
/// equals exactly `forward_passes` per iteration
/// ([`crate::setup::node_graph::NodeGraph::pool_cut_stride`]), so feeding it the
/// same value here never triggers growth; a declared graph's realized
/// per-pool visit source is supplied once a general-graph traversal is
/// wired.
///
/// Skips the terminal-stage pool (`node_graph.pool_stage[p] ==
/// StageIdx(num_stages - 1)`): a leaf's populated count never advances, so an
/// unconditional grow would re-introduce the growable slack a fixed
/// boundary-injected pool (`CutPool::new_with_warm_start` at
/// `max_iterations = 0`) was built without.
fn grow_pools_for_next_iteration(
    fcf: &mut FutureCostFunction,
    realized_visits: u64,
    node_graph: &NodeGraph,
    num_stages: usize,
) {
    let terminal_stage = (num_stages > 0).then(|| StageIdx(num_stages - 1));
    for (p, pool) in fcf.pools.iter_mut().enumerate() {
        if terminal_stage == Some(node_graph.pool_stage[p]) {
            continue;
        }
        let populated = pool.populated();
        // Cast cannot truncate: SDDP runs only on 64-bit targets.
        #[allow(clippy::cast_possible_truncation)]
        let remaining = (pool.capacity - populated) as u64;
        if realized_visits <= remaining {
            continue;
        }
        let mut new_capacity = pool.capacity.max(1);
        loop {
            #[allow(clippy::cast_possible_truncation)]
            let free = (new_capacity - populated) as u64;
            if free >= realized_visits {
                break;
            }
            new_capacity *= 2;
        }
        pool.grow(new_capacity);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::needless_range_loop
)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::mpsc;

    use chrono::NaiveDate;
    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
    use cobre_core::{
        Bus, EntityId, SystemBuilder, TrainingEvent, WorkerTimingPhase,
        scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
            SamplingScheme,
        },
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };
    use cobre_io::OwnedPolicyCutRecord;
    use cobre_solver::{
        Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };
    use cobre_stochastic::{
        ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context,
    };

    use super::{
        IterationOutcome, TrainingSession, grow_pools_for_next_iteration,
        selection_record_index_by_pool,
    };
    use crate::{
        CutPool, SolverProfiles, StoppingMode, StoppingRule, StoppingRuleSet, TrainingConfig,
        config::{CutManagementConfig, EventConfig, LoopConfig},
        context::TrainingContext,
        cut::fcf::FutureCostFunction,
        error::SddpError,
        horizon_mode::HorizonMode,
        inflow_method::InflowNonNegativityMethod,
        lp::builder::{StageGeometry, StateBox},
        lp::indexer::{CutStateProjection, StateSpace, StudyDimensions},
        risk_measure::RiskMeasure,
        setup::node_graph::StageIdx,
        setup::{
            NodeGraph, NodeId, NodeOpenings, NodePos, NodeRuntime, NodeSuccessor, OpeningSource,
        },
        solver_stats::WORKER_STATS_ENTRY_STRIDE,
        test_support::{
            self, StageContextFixture, equipment_free_geometry, permissive_state_boxes,
        },
    };

    // ── Shared helpers (mirrors training.rs test helpers) ──────────────────

    fn minimal_template(_n_state: usize) -> StageTemplate {
        StageTemplate {
            num_cols: 4,
            num_rows: 2,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, f64::NEG_INFINITY, 0.0, 0.0],
            col_upper: vec![f64::INFINITY; 4],
            objective: vec![0.0, 0.0, 0.0, 1.0],
            row_lower: vec![0.0, 0.0],
            row_upper: vec![0.0, 0.0],
            n_state: 1,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    struct MockSolver {
        objectives: Vec<f64>,
        call_count: usize,
    }

    impl MockSolver {
        fn with_fixed(objective: f64) -> Self {
            Self {
                objectives: vec![objective],
                call_count: 0,
            }
        }
    }

    impl SolverInterface for MockSolver {
        type Profile = cobre_solver::ActiveProfile;

        fn apply_profile(&mut self, _profile: &cobre_solver::ActiveProfile) {}

        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
        fn load_model(&mut self, _t: &StageTemplate) {}
        fn add_rows(&mut self, _r: &RowBatch) {}
        fn set_row_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
        fn set_col_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}

        fn solve(
            &mut self,
            _basis: Option<&Basis>,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            let call = self.call_count;
            self.call_count += 1;
            let obj = self.objectives[call % self.objectives.len()];
            Ok(cobre_solver::SolutionView {
                objective: obj,
                primal: &[0.0, 0.0, 0.0, 0.0],
                dual: &[0.0, 0.0],
                reduced_costs: &[0.0, 0.0, 0.0, 0.0],
                iterations: 0,
                solve_time_seconds: 0.0,
            })
        }

        fn get_basis(&mut self, out: &mut Basis) {
            crate::test_support::fill_consistent_basis(out);
        }

        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }

        fn statistics_into(&self, out: &mut SolverStatistics) {
            out.copy_from(&SolverStatistics::default());
        }

        fn name(&self) -> &'static str {
            "Mock"
        }
    }

    struct StubComm;

    impl Communicator for StubComm {
        fn allgatherv<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            recv[..send.len()].clone_from_slice(send);
            Ok(())
        }

        fn allreduce<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            recv.clone_from_slice(send);
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

        fn abort(&self, error_code: i32) -> ! {
            std::process::exit(error_code)
        }
    }

    #[allow(clippy::cast_possible_wrap)]
    fn make_stochastic_context(n_stages: usize, n_openings: usize) -> StochasticContext {
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};

        let bus = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![cobre_core::DeficitSegment {
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
                branching_factor: n_openings,
                noise_method: NoiseMethod::Saa,
            },
        };

        let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();

        let inflow_models: Vec<_> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(1),
                stage_id: i as i32,
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

    fn make_stages(n_stages: usize) -> Vec<Stage> {
        (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
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
                    branching_factor: 1,
                    noise_method: NoiseMethod::Saa,
                },
            })
            .collect()
    }

    fn make_fcf(
        n_stages: usize,
        n_state: usize,
        forward_passes: u32,
        max_iter: u64,
    ) -> FutureCostFunction {
        FutureCostFunction::new(
            n_stages,
            n_state,
            forward_passes,
            max_iter,
            &vec![0; n_stages],
        )
    }

    fn iteration_limit_rules(limit: u64) -> StoppingRuleSet {
        StoppingRuleSet {
            rules: vec![StoppingRule::IterationLimit { limit }],
            mode: StoppingMode::Any,
        }
    }

    fn make_config(
        forward_passes: u32,
        max_iterations: u64,
        limit: u64,
        n_stages: usize,
    ) -> TrainingConfig {
        TrainingConfig {
            loop_config: LoopConfig {
                forward_passes,
                training_enumerated: false,
                max_iterations,
                start_iteration: 0,
                resume_lower_bound_history: Vec::new(),
                n_fwd_threads: 1,
                stopping_rules: iteration_limit_rules(limit),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                risk_measures: vec![RiskMeasure::Expectation; n_stages],
            },
            events: EventConfig {
                event_sender: None,
                periodic_checkpoint: None,
                shutdown_flag: None,
                export_states: false,
            },
        }
    }

    fn make_stage_ctx<'a>(
        templates: &'a [StageTemplate],
        geometry_per_stage: &'a [StageGeometry],
        state_boxes: &'a [StateBox],
    ) -> StageContextFixture<'a> {
        StageContextFixture::new(templates, state_boxes, geometry_per_stage)
    }

    fn make_training_ctx<'a>(
        horizon: &'a HorizonMode,
        study_dims: &'a StudyDimensions,
        state: &'a StateSpace,
        cut_state_layouts: &'a [CutStateProjection],
        stochastic: &'a StochasticContext,
        initial_state: &'a [f64],
        stages: &'a [Stage],
        node_graph: &'a NodeGraph,
    ) -> TrainingContext<'a> {
        TrainingContext {
            horizon,
            state,
            cut_state_layouts,
            study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic,
            initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
            node_graph,
        }
    }

    // ── Test: training_session_new_preallocates_all_buffers ────────────────

    /// Verify that `TrainingSession::new` pre-allocates all scratch buffers to
    /// their expected sizes before the first iteration.
    #[test]
    fn training_session_new_preallocates_all_buffers() {
        let n_stages = 2;
        let state = test_support::state_layout(1, 0);
        let templates = vec![minimal_template(state.n_state); n_stages];
        let initial_state = vec![0.0_f64; state.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, state.n_state, 1, 10);
        let config = make_config(1, 10, 1, n_stages);
        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;
        let block_counts = vec![1usize; n_stages];
        let state_boxes = permissive_state_boxes(templates[0].n_state, n_stages);
        let geometry = equipment_free_geometry(&block_counts);
        let fixture = make_stage_ctx(&templates, &geometry, &state_boxes);
        let stage_ctx = fixture.ctx();
        let study_dims = test_support::study_dims();
        let cut_state_layouts = test_support::all_enabled_cut_state_layouts(&state, n_stages);
        let node_graph_fixture = test_support::chain_node_graph(&stochastic);
        let training_ctx = make_training_ctx(
            &horizon,
            &study_dims,
            &state,
            &cut_state_layouts,
            &stochastic,
            &initial_state,
            &stages,
            &node_graph_fixture,
        );

        let session = TrainingSession::new(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &training_ctx,
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
            SolverProfiles::default(),
        )
        .unwrap();

        // forward_passes=1, num_ranks=1 → max_local_fwd=1
        let max_local_fwd = 1usize;
        assert_eq!(
            session.scratch.records.len(),
            max_local_fwd * n_stages,
            "records must be pre-sized to max_local_fwd * num_stages"
        );
        assert_eq!(
            session.scratch.cut_batches.len(),
            n_stages,
            "cut_batches must have one RowBatch per stage"
        );
        assert_eq!(
            session.scratch.frozen_templates.len(),
            n_stages,
            "frozen_templates must have one per stage"
        );
        // send_stride = n_workers_local * max_openings * WORKER_STATS_ENTRY_STRIDE
        // n_fwd_threads=1 → n_workers_local=1; max_openings=1 for this fixture
        let expected_send_stride = WORKER_STATS_ENTRY_STRIDE;
        assert_eq!(
            session.bwd_state.bwd_stats_send_buf.len(),
            expected_send_stride,
            "bwd_stats_send_buf must equal send_stride"
        );
    }

    // ── Test: per-iteration freeze skips the terminal pool ────────────────

    /// The terminal pool's frozen template is baked once by
    /// `prime_frozen_templates` and left byte-identical by a subsequent
    /// per-iteration `freeze_active_cuts_into_templates(true)` call, while a
    /// non-terminal pool is re-baked with a newly added cut.
    #[test]
    fn per_iteration_freeze_skips_terminal_pool_but_rebakes_others() {
        let n_stages = 2;
        let state = test_support::state_layout(1, 0);
        let templates = vec![minimal_template(state.n_state); n_stages];
        let initial_state = vec![0.0_f64; state.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, state.n_state, 1, 10);
        fcf.pools[0].add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
        fcf.pools[1].add_cut(NodeId(0), 0, 0, 2.0, &[2.0]);

        let config = make_config(1, 10, 1, n_stages);
        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;
        let block_counts = vec![1usize; n_stages];
        let state_boxes = permissive_state_boxes(templates[0].n_state, n_stages);
        let geometry = equipment_free_geometry(&block_counts);
        let fixture = make_stage_ctx(&templates, &geometry, &state_boxes);
        let stage_ctx = fixture.ctx();
        let study_dims = test_support::study_dims();
        let cut_state_layouts = test_support::all_enabled_cut_state_layouts(&state, n_stages);
        let node_graph_fixture = test_support::chain_node_graph(&stochastic);
        let training_ctx = make_training_ctx(
            &horizon,
            &study_dims,
            &state,
            &cut_state_layouts,
            &stochastic,
            &initial_state,
            &stages,
            &node_graph_fixture,
        );

        let mut session = TrainingSession::new(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &training_ctx,
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
            SolverProfiles::default(),
        )
        .unwrap();

        session.prime_frozen_templates();

        let terminal_pool = n_stages - 1;
        let terminal_after_priming = session.scratch.frozen_templates[terminal_pool].clone();
        let non_terminal_after_priming = session.scratch.frozen_templates[0].clone();

        // Add a second cut to the non-terminal pool only; the terminal pool's
        // active-cut set is untouched.
        session.fcf.pools[0].add_cut(NodeId(0), 2, 0, 3.0, &[3.0]);

        let _ = session.freeze_active_cuts_into_templates(true);

        let terminal_after_iteration = &session.scratch.frozen_templates[terminal_pool];
        assert_eq!(
            terminal_after_iteration.num_rows, terminal_after_priming.num_rows,
            "terminal pool's row count must be unchanged by the per-iteration refreeze"
        );
        assert_eq!(
            terminal_after_iteration.row_indices, terminal_after_priming.row_indices,
            "terminal template rows must stay byte-identical across the per-iteration freeze"
        );
        assert_eq!(
            terminal_after_iteration.values, terminal_after_priming.values,
            "terminal template coefficients must stay byte-identical across the per-iteration freeze"
        );
        assert_eq!(
            terminal_after_iteration.row_lower, terminal_after_priming.row_lower,
            "terminal template row bounds must stay byte-identical across the per-iteration freeze"
        );
        assert_eq!(
            terminal_after_iteration.row_upper, terminal_after_priming.row_upper,
            "terminal template row bounds must stay byte-identical across the per-iteration freeze"
        );

        let non_terminal_after_iteration = &session.scratch.frozen_templates[0];
        assert_ne!(
            non_terminal_after_iteration.num_rows, non_terminal_after_priming.num_rows,
            "non-terminal pool must be re-baked to include the newly added cut"
        );
    }

    // ── Test: training_session_finalize_emits_training_finished ───────────

    /// Verify that `finalize()` emits exactly one `TrainingFinished` event.
    #[test]
    fn training_session_finalize_emits_training_finished() {
        let n_stages = 2;
        let state = test_support::state_layout(1, 0);
        let templates = vec![minimal_template(state.n_state); n_stages];
        let initial_state = vec![0.0_f64; state.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, state.n_state, 1, 10);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let mut config = make_config(1, 10, 1, n_stages);
        config.events.event_sender = Some(tx);

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;
        let block_counts = vec![1usize; n_stages];
        let state_boxes = permissive_state_boxes(templates[0].n_state, n_stages);
        let geometry = equipment_free_geometry(&block_counts);
        let fixture = make_stage_ctx(&templates, &geometry, &state_boxes);
        let stage_ctx = fixture.ctx();
        let study_dims = test_support::study_dims();
        let cut_state_layouts = test_support::all_enabled_cut_state_layouts(&state, n_stages);
        let node_graph_fixture = test_support::chain_node_graph(&stochastic);
        let training_ctx = make_training_ctx(
            &horizon,
            &study_dims,
            &state,
            &cut_state_layouts,
            &stochastic,
            &initial_state,
            &stages,
            &node_graph_fixture,
        );

        let session = TrainingSession::new(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &training_ctx,
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
            SolverProfiles::default(),
        )
        .unwrap();

        // finalize without running any iterations
        let outcome = session.finalize().unwrap();

        assert!(outcome.error.is_none(), "no error expected");
        assert_eq!(outcome.result.iterations, 0);

        let events: Vec<TrainingEvent> = rx.try_iter().collect();
        let last = events.last().unwrap();
        assert!(
            matches!(last, TrainingEvent::TrainingFinished { iterations: 0, .. }),
            "last event must be TrainingFinished with iterations=0, got: {last:?}"
        );
    }

    // ── Test: training_session_finalize_with_error_emits_training_finished_with_error_reason ──

    /// Verify that `finalize_with_error` emits `TrainingFinished` with `reason = "error"`.
    #[test]
    fn training_session_finalize_with_error_emits_training_finished_with_error_reason() {
        let n_stages = 2;
        let state = test_support::state_layout(1, 0);
        let templates = vec![minimal_template(state.n_state); n_stages];
        let initial_state = vec![0.0_f64; state.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, state.n_state, 1, 10);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let mut config = make_config(1, 10, 1, n_stages);
        config.events.event_sender = Some(tx);

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;
        let block_counts = vec![1usize; n_stages];
        let state_boxes = permissive_state_boxes(templates[0].n_state, n_stages);
        let geometry = equipment_free_geometry(&block_counts);
        let fixture = make_stage_ctx(&templates, &geometry, &state_boxes);
        let stage_ctx = fixture.ctx();
        let study_dims = test_support::study_dims();
        let cut_state_layouts = test_support::all_enabled_cut_state_layouts(&state, n_stages);
        let node_graph_fixture = test_support::chain_node_graph(&stochastic);
        let training_ctx = make_training_ctx(
            &horizon,
            &study_dims,
            &state,
            &cut_state_layouts,
            &stochastic,
            &initial_state,
            &stages,
            &node_graph_fixture,
        );

        let session = TrainingSession::new(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &training_ctx,
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
            SolverProfiles::default(),
        )
        .unwrap();

        let outcome = session.finalize_with_error(SddpError::Validation("test error".to_string()));

        assert!(outcome.error.is_some(), "expected error in outcome");
        assert_eq!(outcome.result.reason, "error");

        let events: Vec<TrainingEvent> = rx.try_iter().collect();
        // Events: TrainingStarted + TrainingFinished(reason="error")
        let last = events.last().unwrap();
        assert!(
            matches!(last, TrainingEvent::TrainingFinished { .. }),
            "last event must be TrainingFinished"
        );
        if let TrainingEvent::TrainingFinished { reason, .. } = last {
            assert_eq!(reason, "error");
        }
    }

    // ── Test: training_session_run_iteration_returns_continue_when_not_converged ──

    /// Verify that `run_iteration` returns `Continue` when stopping rules have not triggered.
    #[test]
    fn training_session_run_iteration_returns_continue_when_not_converged() {
        let n_stages = 2;
        let state = test_support::state_layout(1, 0);
        let templates = vec![minimal_template(state.n_state); n_stages];
        let initial_state = vec![0.0_f64; state.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, state.n_state, 1, 10);
        // iteration_limit=5, so iteration 1 should return Continue
        let config = make_config(1, 10, 5, n_stages);

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;
        let block_counts = vec![1usize; n_stages];
        let state_boxes = permissive_state_boxes(templates[0].n_state, n_stages);
        let geometry = equipment_free_geometry(&block_counts);
        let fixture = make_stage_ctx(&templates, &geometry, &state_boxes);
        let stage_ctx = fixture.ctx();
        let study_dims = test_support::study_dims();
        let cut_state_layouts = test_support::all_enabled_cut_state_layouts(&state, n_stages);
        let node_graph_fixture = test_support::chain_node_graph(&stochastic);
        let training_ctx = make_training_ctx(
            &horizon,
            &study_dims,
            &state,
            &cut_state_layouts,
            &stochastic,
            &initial_state,
            &stages,
            &node_graph_fixture,
        );

        let mut session = TrainingSession::new(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &training_ctx,
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
            SolverProfiles::default(),
        )
        .unwrap();

        let result = session.run_iteration(1).unwrap();
        assert!(
            matches!(result, IterationOutcome::Continue),
            "expected Continue when limit is 5, got: {result:?}"
        );
    }

    // ── Test: training_session_run_iteration_returns_converged_when_gap_closes ──

    /// Verify that `run_iteration` eventually returns `Converged` when a stopping
    /// rule triggers.
    #[test]
    fn training_session_run_iteration_returns_converged_when_gap_closes() {
        let n_stages = 2;
        let state = test_support::state_layout(1, 0);
        let templates = vec![minimal_template(state.n_state); n_stages];
        let initial_state = vec![0.0_f64; state.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, state.n_state, 1, 10);
        // iteration_limit=1 so the first iteration triggers convergence
        let config = make_config(1, 10, 1, n_stages);

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;
        let block_counts = vec![1usize; n_stages];
        let state_boxes = permissive_state_boxes(templates[0].n_state, n_stages);
        let geometry = equipment_free_geometry(&block_counts);
        let fixture = make_stage_ctx(&templates, &geometry, &state_boxes);
        let stage_ctx = fixture.ctx();
        let study_dims = test_support::study_dims();
        let cut_state_layouts = test_support::all_enabled_cut_state_layouts(&state, n_stages);
        let node_graph_fixture = test_support::chain_node_graph(&stochastic);
        let training_ctx = make_training_ctx(
            &horizon,
            &study_dims,
            &state,
            &cut_state_layouts,
            &stochastic,
            &initial_state,
            &stages,
            &node_graph_fixture,
        );

        let mut session = TrainingSession::new(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &training_ctx,
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
            SolverProfiles::default(),
        )
        .unwrap();

        let mut last_outcome = IterationOutcome::Continue;
        for iter in session.iteration_range() {
            last_outcome = session.run_iteration(iter).unwrap();
            if !matches!(last_outcome, IterationOutcome::Continue) {
                break;
            }
        }
        assert!(
            matches!(last_outcome, IterationOutcome::Converged),
            "expected Converged after iteration limit triggers, got: {last_outcome:?}"
        );
    }

    // ── Test: training_session_run_iteration_emits_correct_event_sequence ───

    /// Verify that one call to `run_iteration(1)` followed by `finalize()` emits
    /// exactly 11 events in the correct order:
    ///
    /// 1  × `TrainingStarted`   (emitted by `new`)
    /// 9  × per-iteration events (emitted by `run_iteration`):
    ///        `WorkerTiming(Forward)`, `ForwardPassComplete`, `ForwardSyncComplete`,
    ///        `WorkerTiming(Backward)`, `BackwardPassComplete`, `PolicySyncComplete`,
    ///        `PolicyTemplateFreezeComplete`, `ConvergenceUpdate`, `IterationSummary`
    /// 1  × `TrainingFinished`  (emitted by `finalize`)
    ///
    /// Total = 11 events.
    #[test]
    fn training_session_run_iteration_emits_correct_event_sequence() {
        let n_stages = 2;
        let state = test_support::state_layout(1, 0);
        let templates = vec![minimal_template(state.n_state); n_stages];
        let initial_state = vec![0.0_f64; state.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, state.n_state, 1, 10);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let mut config = make_config(1, 10, 10, n_stages);
        config.events.event_sender = Some(tx);

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;
        let block_counts = vec![1usize; n_stages];
        let state_boxes = permissive_state_boxes(templates[0].n_state, n_stages);
        let geometry = equipment_free_geometry(&block_counts);
        let fixture = make_stage_ctx(&templates, &geometry, &state_boxes);
        let stage_ctx = fixture.ctx();
        let study_dims = test_support::study_dims();
        let cut_state_layouts = test_support::all_enabled_cut_state_layouts(&state, n_stages);
        let node_graph_fixture = test_support::chain_node_graph(&stochastic);
        let training_ctx = make_training_ctx(
            &horizon,
            &study_dims,
            &state,
            &cut_state_layouts,
            &stochastic,
            &initial_state,
            &stages,
            &node_graph_fixture,
        );

        let mut session = TrainingSession::new(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &training_ctx,
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
            SolverProfiles::default(),
        )
        .unwrap();

        // Run exactly one iteration (does not trigger the stopping rule since limit=10).
        let outcome = session.run_iteration(1).unwrap();
        assert!(
            matches!(outcome, IterationOutcome::Continue),
            "expected Continue for iteration 1 with limit=10, got: {outcome:?}"
        );

        // Finalize emits TrainingFinished.
        session.finalize().unwrap();

        let events: Vec<TrainingEvent> = rx.try_iter().collect();

        // ── Count assertion ────────────────────────────────────────────────
        // 1 (TrainingStarted) + 9 (per-iteration) + 1 (TrainingFinished) = 11
        assert_eq!(
            events.len(),
            11,
            "expected 11 events for 1 iteration, got {} ({events:?})",
            events.len()
        );

        // ── Order assertion ────────────────────────────────────────────────
        assert!(
            matches!(events[0], TrainingEvent::TrainingStarted { .. }),
            "events[0] must be TrainingStarted, got: {:?}",
            events[0]
        );

        // Per-iteration block (events[1..=9])
        assert!(
            matches!(
                events[1],
                TrainingEvent::WorkerTiming {
                    phase: WorkerTimingPhase::Forward,
                    ..
                }
            ),
            "events[1] must be WorkerTiming(Forward), got: {:?}",
            events[1]
        );
        assert!(
            matches!(events[2], TrainingEvent::ForwardPassComplete { .. }),
            "events[2] must be ForwardPassComplete, got: {:?}",
            events[2]
        );
        assert!(
            matches!(events[3], TrainingEvent::ForwardSyncComplete { .. }),
            "events[3] must be ForwardSyncComplete, got: {:?}",
            events[3]
        );
        assert!(
            matches!(
                events[4],
                TrainingEvent::WorkerTiming {
                    phase: WorkerTimingPhase::Backward,
                    ..
                }
            ),
            "events[4] must be WorkerTiming(Backward), got: {:?}",
            events[4]
        );
        assert!(
            matches!(events[5], TrainingEvent::BackwardPassComplete { .. }),
            "events[5] must be BackwardPassComplete, got: {:?}",
            events[5]
        );
        assert!(
            matches!(events[6], TrainingEvent::PolicySyncComplete { .. }),
            "events[6] must be PolicySyncComplete, got: {:?}",
            events[6]
        );
        assert!(
            matches!(
                events[7],
                TrainingEvent::PolicyTemplateFreezeComplete { .. }
            ),
            "events[7] must be PolicyTemplateFreezeComplete, got: {:?}",
            events[7]
        );
        assert!(
            matches!(events[8], TrainingEvent::ConvergenceUpdate { .. }),
            "events[8] must be ConvergenceUpdate, got: {:?}",
            events[8]
        );
        assert!(
            matches!(events[9], TrainingEvent::IterationSummary { .. }),
            "events[9] must be IterationSummary, got: {:?}",
            events[9]
        );

        assert!(
            matches!(events[10], TrainingEvent::TrainingFinished { .. }),
            "events[10] must be TrainingFinished, got: {:?}",
            events[10]
        );
    }

    // ── grow_pools_for_next_iteration ───────────────────────────────────────

    #[test]
    fn grow_pools_for_next_iteration_doubles_when_floor_exceeded_and_preserves_slots() {
        // A deliberately undersized "sampled-general" floor: one pool, one
        // populated slot, one remaining slot.
        let mut fcf = FutureCostFunction::new_per_pool(&[1], 1, 1, 2, &[0], &[1]);
        assert_eq!(fcf.pools[0].capacity, 2);
        fcf.pools[0].add_cut(NodeId(0), 0, 0, 7.0, &[3.0]);
        let prior_capacity = fcf.pools[0].capacity;
        let stochastic = make_stochastic_context(2, 1);
        let node_graph = test_support::chain_node_graph(&stochastic);

        grow_pools_for_next_iteration(&mut fcf, 3, &node_graph, 2);

        assert!(
            fcf.pools[0].capacity > prior_capacity,
            "capacity must grow when realized visits exceed the remaining slots"
        );
        assert!(
            fcf.pools[0].is_active(0),
            "slot 0 stays active after growth"
        );
        assert_eq!(fcf.pools[0].intercept(0), 7.0);
        assert_eq!(fcf.pools[0].coefficient_row(0), &[3.0]);
    }

    #[test]
    fn grow_pools_for_next_iteration_is_noop_when_floor_suffices() {
        let mut fcf = FutureCostFunction::new_per_pool(&[1], 1, 1, 10, &[0], &[1]);
        fcf.pools[0].add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
        let prior_capacity = fcf.pools[0].capacity;
        let stochastic = make_stochastic_context(2, 1);
        let node_graph = test_support::chain_node_graph(&stochastic);

        grow_pools_for_next_iteration(&mut fcf, 1, &node_graph, 2);

        assert_eq!(
            fcf.pools[0].capacity, prior_capacity,
            "no growth (no allocation) when the floor already suffices"
        );
    }

    #[test]
    fn grow_pools_for_next_iteration_chain_never_triggers() {
        // A chain-shaped FCF (visit_bound == forward_passes for every pool,
        // pool_cut_stride): remaining capacity after any
        // iteration is always >= forward_passes, so this reserved seam is
        // inert on the only live path.
        let forward_passes = 4u32;
        let max_iterations = 3u64;
        let mut fcf = FutureCostFunction::new(2, 1, forward_passes, max_iterations, &[0, 0]);
        for pool in &mut fcf.pools {
            for fp in 0..forward_passes {
                pool.add_cut(NodeId(0), 0, fp, 1.0, &[1.0]);
            }
        }
        let capacities_before: Vec<usize> = fcf.pools.iter().map(|p| p.capacity).collect();
        let stochastic = make_stochastic_context(2, 1);
        let node_graph = test_support::chain_node_graph(&stochastic);

        grow_pools_for_next_iteration(&mut fcf, u64::from(forward_passes), &node_graph, 2);

        let capacities_after: Vec<usize> = fcf.pools.iter().map(|p| p.capacity).collect();
        assert_eq!(capacities_before, capacities_after);
    }

    /// The fixed terminal pool [`inject_boundary_cuts`](crate::inject_boundary_cuts)
    /// builds (`CutPool::new_with_warm_start` at `max_iterations = 0`, so
    /// `capacity == warm_start_count == populated`, `remaining == 0`) must never
    /// grow: the unguarded loop would double it the moment any realized visit
    /// count is positive, silently re-introducing the growable slack removed at
    /// injection.
    #[test]
    fn grow_pools_for_next_iteration_skips_a_fixed_capacity_terminal_pool() {
        let n_stages = 2;
        let forward_passes = 4u32;
        let mut fcf = make_fcf(n_stages, 1, forward_passes, 10);
        let terminal_idx = n_stages - 1;
        let records = vec![OwnedPolicyCutRecord {
            cut_id: 0,
            slot_index: 0,
            coefficients: vec![1.0],
            intercept: 5.0,
            is_active: true,
            iteration: 0,
            forward_pass_index: 0,
        }];
        fcf.pools[terminal_idx] = CutPool::new_with_warm_start(1, forward_passes, 0, &records);
        let capacity_before = fcf.pools[terminal_idx].capacity;
        assert_eq!(
            capacity_before,
            records.len(),
            "fixed terminal pool starts with no growable slack"
        );

        let stochastic = make_stochastic_context(n_stages, 1);
        let node_graph = test_support::chain_node_graph(&stochastic);
        grow_pools_for_next_iteration(&mut fcf, u64::from(forward_passes), &node_graph, n_stages);

        assert_eq!(
            fcf.pools[terminal_idx].capacity, capacity_before,
            "the terminal pool must never grow, even though its remaining capacity is 0"
        );
    }

    fn gen_openings() -> NodeOpenings {
        NodeOpenings {
            source: OpeningSource::Generated,
            offset: 0,
            len: 1,
            q: 1.0,
        }
    }

    /// On a branching graph whose canonical (ascending-id) order does NOT
    /// place the root first, the cut-budget back-annotation must reach each
    /// pool's OWN selection record through `selection_record_index_by_pool`, not
    /// by using the pool id as a `per_stage` index.
    ///
    /// Ids 1,2 (leaves, positions 0,1), 3,7 (fan nodes, positions 2,3), 10
    /// (root, position 4). Pools: fan A→0, fan B→1, root→2, shared leaf→3. The
    /// `per_stage` record order is [root, fan A, fan B], so the correct map sends
    /// pool 2→record 0, pool 0→record 1, pool 1→record 2, leaf pool 3→None —
    /// deliberately NOT the identity `pool_id→pool_id` the pre-fix read assumed.
    #[test]
    fn selection_record_index_by_pool_maps_branching_root_not_by_pool_id() {
        let leaf = |pool_id| NodeRuntime {
            stage: StageIdx(2),
            pool_id,
            openings: gen_openings(),
        };
        let fan = |pool_id| NodeRuntime {
            stage: StageIdx(1),
            pool_id,
            openings: gen_openings(),
        };
        let root = NodeRuntime {
            stage: StageIdx(0),
            pool_id: 2,
            openings: gen_openings(),
        };
        let succ = |child: NodePos| NodeSuccessor {
            child,
            probability: 1.0,
        };
        let node_graph = NodeGraph {
            node_ids: vec![NodeId(1), NodeId(2), NodeId(3), NodeId(7), NodeId(10)].into(),
            nodes: vec![leaf(3), leaf(3), fan(0), fan(1), root].into(),
            successors: vec![
                Vec::new(),
                Vec::new(),
                vec![succ(NodePos(0))],
                vec![succ(NodePos(1))],
                vec![succ(NodePos(2)), succ(NodePos(3))],
            ]
            .into(),
            n_pools: 4,
            pool_stage: vec![StageIdx(1), StageIdx(1), StageIdx(0), StageIdx(2)],
        };

        // root_pool + interior_nodes computed exactly as run_cut_management does.
        let root_pool = node_graph.nodes[node_graph.frontier_node(StageIdx(0)).unwrap()].pool_id;
        let interior_nodes: Vec<NodePos> = node_graph
            .nodes
            .iter_indexed()
            .filter(|&(pos, n)| n.stage >= StageIdx(1) && !node_graph.successors[pos].is_empty())
            .map(|(pos, _)| pos)
            .collect();
        assert_eq!(root_pool, 2, "the root owns pool 2, not pool 0");
        assert_eq!(
            interior_nodes,
            vec![NodePos(2), NodePos(3)],
            "fan nodes at canonical positions 2, 3"
        );

        let map = selection_record_index_by_pool(&node_graph, root_pool, &interior_nodes);
        assert_eq!(
            map,
            vec![Some(1), Some(2), Some(0), None],
            "pool 0→record 1, pool 1→record 2, pool 2 (root)→record 0, leaf pool 3→None"
        );

        // Power: the fix genuinely differs from the pre-fix identity map.
        let identity: Vec<Option<usize>> = (0..node_graph.n_pools).map(Some).collect();
        assert_ne!(
            map, identity,
            "on a root-not-smallest-id graph the record map must NOT be the identity — \
             using pool id as a per_stage index misattributes eviction stats"
        );
    }

    // ── TrainingSession::new: enumerated world >= 2 interior-branching guard ─

    /// `size() == 2` sibling of [`StubComm`] — the same faithful rank-0
    /// partial-write stub `backward_pass_state.rs`'s test module uses for its
    /// 2-rank enumerated backward regression. `TrainingSession::new` never
    /// issues a collective itself (only `RankDistribution::new` reads
    /// `size()`/`rank()`), so the collective bodies below are never exercised
    /// by the tests that use this stub — only `size()` is load-bearing.
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
            recv[start..start + send.len()].clone_from_slice(send);
            Ok(())
        }

        fn allreduce<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            recv.clone_from_slice(send);
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

        fn abort(&self, error_code: i32) -> ! {
            std::process::exit(error_code)
        }
    }

    /// Root branches into two non-leaf nodes at stage 1, each fanning into
    /// its own leaf at stage 2 — node 1 (and node 2) sit on only one of the
    /// two root→leaf paths, never on both.
    fn interior_branching_node_graph() -> NodeGraph {
        let leaf = |pool_id| NodeRuntime {
            stage: StageIdx(2),
            pool_id,
            openings: gen_openings(),
        };
        let branch = |pool_id| NodeRuntime {
            stage: StageIdx(1),
            pool_id,
            openings: gen_openings(),
        };
        let root = NodeRuntime {
            stage: StageIdx(0),
            pool_id: 0,
            openings: gen_openings(),
        };
        NodeGraph {
            node_ids: vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3), NodeId(4)].into(),
            nodes: vec![root, branch(1), branch(2), leaf(3), leaf(3)].into(),
            successors: vec![
                vec![
                    NodeSuccessor {
                        child: NodePos(1),
                        probability: 0.5,
                    },
                    NodeSuccessor {
                        child: NodePos(2),
                        probability: 0.5,
                    },
                ],
                vec![NodeSuccessor {
                    child: NodePos(3),
                    probability: 1.0,
                }],
                vec![NodeSuccessor {
                    child: NodePos(4),
                    probability: 1.0,
                }],
                Vec::new(),
                Vec::new(),
            ]
            .into(),
            n_pools: 4,
            pool_stage: vec![StageIdx(0), StageIdx(1), StageIdx(1), StageIdx(2)],
        }
    }

    /// Root → trunk → three leaves: every non-leaf node (root, trunk) lies on
    /// every root→leaf path — the deterministic-trunk + terminal-fan shape.
    fn trunk_terminal_fan_node_graph() -> NodeGraph {
        let leaf = || NodeRuntime {
            stage: StageIdx(2),
            pool_id: 2,
            openings: gen_openings(),
        };
        let trunk = NodeRuntime {
            stage: StageIdx(1),
            pool_id: 1,
            openings: gen_openings(),
        };
        let root = NodeRuntime {
            stage: StageIdx(0),
            pool_id: 0,
            openings: gen_openings(),
        };
        let succ = |child: NodePos| NodeSuccessor {
            child,
            probability: 1.0 / 3.0,
        };
        NodeGraph {
            node_ids: vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3), NodeId(4)].into(),
            nodes: vec![root, trunk, leaf(), leaf(), leaf()].into(),
            successors: vec![
                vec![NodeSuccessor {
                    child: NodePos(1),
                    probability: 1.0,
                }],
                vec![succ(NodePos(2)), succ(NodePos(3)), succ(NodePos(4))],
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ]
            .into(),
            n_pools: 3,
            pool_stage: vec![StageIdx(0), StageIdx(1), StageIdx(2)],
        }
    }

    /// World >= 2 enumerated training over an interior-branching graph is a
    /// named `SddpError::Validation`, never a panic and never a silent
    /// rank-shape-dependent wrong cut.
    #[test]
    fn training_session_new_rejects_interior_branching_enumerated_at_world_ge_2() {
        let n_stages = 3;
        let node_graph = interior_branching_node_graph();
        let state = test_support::state_layout(1, 0);
        let templates = vec![minimal_template(state.n_state); n_stages];
        let initial_state = vec![0.0_f64; state.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = FutureCostFunction::new(
            node_graph.n_pools,
            state.n_state,
            1,
            10,
            &vec![0; node_graph.n_pools],
        );
        let mut config = make_config(1, 10, 1, n_stages);
        config.loop_config.training_enumerated = true;
        let mut solver = MockSolver::with_fixed(100.0);
        let comm = Rank0Of2;
        let block_counts = vec![1usize; n_stages];
        let state_boxes = permissive_state_boxes(templates[0].n_state, n_stages);
        let geometry = equipment_free_geometry(&block_counts);
        let fixture = make_stage_ctx(&templates, &geometry, &state_boxes);
        let stage_ctx = fixture.ctx();
        let study_dims = test_support::study_dims();
        let cut_state_layouts =
            test_support::all_enabled_cut_state_layouts(&state, node_graph.n_pools);
        let training_ctx = make_training_ctx(
            &horizon,
            &study_dims,
            &state,
            &cut_state_layouts,
            &stochastic,
            &initial_state,
            &stages,
            &node_graph,
        );

        // `TrainingSession` carries no `Debug` impl, so `.err().expect(..)`
        // (never `.expect_err(..)`, which requires `Debug` on the `Ok` side).
        let err = TrainingSession::new(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &training_ctx,
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
            SolverProfiles::default(),
        )
        .err()
        .expect(
            "interior branching under enumerated at world >= 2 must be rejected, not \
             silently wrong",
        );

        match err {
            SddpError::Validation(msg) => {
                assert!(
                    msg.contains("interior branching"),
                    "message must name the interior-branching condition: {msg}"
                );
                assert!(
                    msg.contains(&format!("node {}", NodeId(1))),
                    "message must name the offending node: {msg}"
                );
            }
            other => panic!("expected SddpError::Validation, got {other:?}"),
        }
    }

    /// The same interior-branching graph at world = 1 is always sound (a
    /// single rank holds every node's state), so `TrainingSession::new`
    /// succeeds.
    #[test]
    fn training_session_new_accepts_interior_branching_enumerated_at_world_1() {
        let n_stages = 3;
        let node_graph = interior_branching_node_graph();
        let state = test_support::state_layout(1, 0);
        let templates = vec![minimal_template(state.n_state); n_stages];
        let initial_state = vec![0.0_f64; state.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = FutureCostFunction::new(
            node_graph.n_pools,
            state.n_state,
            1,
            10,
            &vec![0; node_graph.n_pools],
        );
        let mut config = make_config(1, 10, 1, n_stages);
        config.loop_config.training_enumerated = true;
        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;
        let block_counts = vec![1usize; n_stages];
        let state_boxes = permissive_state_boxes(templates[0].n_state, n_stages);
        let geometry = equipment_free_geometry(&block_counts);
        let fixture = make_stage_ctx(&templates, &geometry, &state_boxes);
        let stage_ctx = fixture.ctx();
        let study_dims = test_support::study_dims();
        let cut_state_layouts =
            test_support::all_enabled_cut_state_layouts(&state, node_graph.n_pools);
        let training_ctx = make_training_ctx(
            &horizon,
            &study_dims,
            &state,
            &cut_state_layouts,
            &stochastic,
            &initial_state,
            &stages,
            &node_graph,
        );

        let _session = TrainingSession::new(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &training_ctx,
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
            SolverProfiles::default(),
        )
        .expect("world = 1 is always sound, even with interior branching");
    }

    /// The deterministic-trunk + terminal-fan shape at world >= 2 does NOT
    /// trip the guard — the DECOMP shape stays live.
    #[test]
    fn training_session_new_accepts_trunk_terminal_fan_enumerated_at_world_ge_2() {
        let n_stages = 3;
        let node_graph = trunk_terminal_fan_node_graph();
        let state = test_support::state_layout(1, 0);
        let templates = vec![minimal_template(state.n_state); n_stages];
        let initial_state = vec![0.0_f64; state.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = FutureCostFunction::new(
            node_graph.n_pools,
            state.n_state,
            1,
            10,
            &vec![0; node_graph.n_pools],
        );
        let mut config = make_config(1, 10, 1, n_stages);
        config.loop_config.training_enumerated = true;
        let mut solver = MockSolver::with_fixed(100.0);
        let comm = Rank0Of2;
        let block_counts = vec![1usize; n_stages];
        let state_boxes = permissive_state_boxes(templates[0].n_state, n_stages);
        let geometry = equipment_free_geometry(&block_counts);
        let fixture = make_stage_ctx(&templates, &geometry, &state_boxes);
        let stage_ctx = fixture.ctx();
        let study_dims = test_support::study_dims();
        let cut_state_layouts =
            test_support::all_enabled_cut_state_layouts(&state, node_graph.n_pools);
        let training_ctx = make_training_ctx(
            &horizon,
            &study_dims,
            &state,
            &cut_state_layouts,
            &stochastic,
            &initial_state,
            &stages,
            &node_graph,
        );

        let _session = TrainingSession::new(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &training_ctx,
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
            SolverProfiles::default(),
        )
        .expect("a deterministic trunk + terminal fan stays sound at world >= 2");
    }
}
