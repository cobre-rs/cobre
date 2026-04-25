//! Backward pass state management and entry point.
//!
//! [`BackwardPassState`] owns pre-allocated scratch buffers reused each iteration.
//! [`BackwardPassInputs`] bundles per-call borrowed inputs (no allocation on hot path).

use std::sync::mpsc::Sender;
use std::time::Instant;

use cobre_comm::{Communicator, ReduceOp};
use cobre_core::{TrainingEvent, WorkerPhaseTimings, WorkerTimingPhase};
use cobre_solver::{RowBatch, SolverInterface, SolverStatistics, StageTemplate};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{
    backward::{
        BackwardResult, StageWorkerOpeningDelta, StagedCut, SuccessorSpec, load_backward_lp,
        process_trial_point_backward,
    },
    config::CutManagementConfig,
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    cut_sync::CutSyncBuffers,
    error::SddpError,
    forward::{build_delta_cut_row_batch_into, partition},
    risk_measure::RiskMeasure,
    solver_stats::{
        SolverStatsDelta, StageWorkerStatsBuffer, WORKER_STATS_ENTRY_STRIDE,
        pack_worker_opening_stats, unpack_worker_opening_stats,
    },
    state_exchange::ExchangeBuffers,
    training_session::{
        iteration_scratch::IterationScratch, rank_distribution::RankDistribution,
        runtime::RuntimeHandles,
    },
    trajectory::TrajectoryRecord,
    visited_states::VisitedStatesArchive,
    workspace::{BasisStore, BasisStoreSliceMut, SolverWorkspace, WorkspacePool},
};

/// Per-iteration argument bundle for [`BackwardPassState::run`].
///
/// Groups all borrowed inputs that vary between calls: exchange buffers,
/// trajectory records, risk measures, cut-sync state, and the event sender.
/// Owned scratch buffers live on [`BackwardPassState`] and are not repeated here.
pub struct BackwardPassInputs<'a, S: SolverInterface + Send, C: Communicator> {
    /// Solver workspaces (one per rayon worker thread).
    pub workspaces: &'a mut [SolverWorkspace<S>],
    /// Basis warm-start store (one slot per `(scenario, stage)` pair).
    pub basis_store: &'a mut BasisStore,
    /// Stage-level LP context (templates, row counts, noise scales).
    pub ctx: &'a StageContext<'a>,
    /// Baked LP templates including pre-appended prior-iteration cuts.
    pub baked: &'a [StageTemplate],
    /// Future-cost function — receives new cuts after each stage.
    pub fcf: &'a mut FutureCostFunction,
    /// Per-stage delta cut row batches (reused scratch, resized per stage).
    pub cut_batches: &'a mut [RowBatch],
    /// Study-level training context (horizon, indexer, stochastic model).
    pub training_ctx: &'a TrainingContext<'a>,
    /// MPI communicator.
    pub comm: &'a C,

    /// Exchange buffers for gathering trial-point states via `allgatherv`.
    ///
    /// When non-empty records are provided, the exchange is populated per
    /// stage. When `records` is empty the caller has pre-populated the
    /// exchange buffers (test path).
    pub exchange: &'a mut ExchangeBuffers,

    /// Forward-pass trajectory records used to populate `exchange` per stage.
    ///
    /// Length must be `local_work * num_stages` when non-empty; pass `&[]` in
    /// tests that pre-populate `exchange` directly.
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

    /// Activity-window size for the basis-reconstruction classifier (1..=31).
    pub basis_activity_window: u32,

    /// Current training iteration index (1-based), used for cut metadata.
    pub iteration: u64,

    /// Number of trial points assigned to this rank for the backward pass.
    pub local_work: usize,

    /// Global offset for this rank's trial points (`rank * fwd_per_rank`).
    pub fwd_offset: usize,
}

impl<'a, S: SolverInterface + Send, C: Communicator> BackwardPassInputs<'a, S, C> {
    /// Construct inputs from the fields of a `TrainingSession`, minus `bwd_state`.
    ///
    /// The caller is responsible for taking `&mut session.bwd_state` separately
    /// (which the borrow checker treats as a disjoint field borrow), then passing
    /// the remaining session fields here.  This collapses the 20-field struct
    /// literal into a single compact call:
    ///
    /// ```text
    /// let bwd = &mut self.bwd_state;
    /// let mut inputs = BackwardPassInputs::from_session_fields(
    ///     &mut self.fwd_pool, &mut self.basis_store, self.stage_ctx,
    ///     &mut self.scratch, self.fcf, &mut self.exchange_bufs,
    ///     &mut self.cut_sync_bufs, &mut self.visited_archive,
    ///     self.training_ctx, self.comm,
    ///     &self.config.cut_management, &self.ranks, &self.runtime, iteration,
    /// );
    /// bwd.run(&mut inputs)?;
    /// ```
    // RATIONALE: 14 args are disjoint borrows of `TrainingSession` fields required because
    // Rust NLL cannot split a single `&mut TrainingSession` borrow when `bwd_state` is also
    // borrowed mutably. Grouping would either reintroduce the aliasing problem or add an
    // extra indirection level that hides the borrow structure.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_session_fields(
        fwd_pool: &'a mut WorkspacePool<S>,
        basis_store: &'a mut BasisStore,
        ctx: &'a StageContext<'a>,
        scratch: &'a mut IterationScratch,
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
    ) -> Self {
        Self {
            workspaces: &mut fwd_pool.workspaces,
            basis_store,
            ctx,
            baked: &scratch.baked_templates,
            fcf,
            cut_batches: &mut scratch.cut_batches,
            training_ctx,
            comm,
            exchange,
            records: &scratch.records,
            cut_sync_bufs,
            visited_archive: visited_archive.as_mut(),
            event_sender: runtime.event_sender(),
            risk_measures: &cut_mgmt.risk_measures,
            cut_activity_tolerance: cut_mgmt.cut_activity_tolerance,
            basis_activity_window: cut_mgmt.basis_activity_window,
            iteration,
            local_work: ranks.my_actual_fwd,
            fwd_offset: ranks.my_fwd_offset,
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
    /// Pre-allocated buffer for uniform opening probabilities. Reused per stage
    /// via `clear()` + `resize()` to avoid per-stage allocation.
    pub(crate) probabilities_buf: Vec<f64>,

    /// Pre-allocated buffer for successor active cut slot indices. Reused per
    /// stage via `clear()` + `extend()` to avoid per-stage allocation.
    pub(crate) successor_active_slots_buf: Vec<usize>,

    /// Pre-allocated buffer for per-slot binding increment aggregation.
    ///
    /// Reused across stages to avoid per-stage allocation. Used for the
    /// `allreduce(Sum)` that synchronises cut binding metadata across MPI ranks.
    pub(crate) metadata_sync_buf: Vec<u64>,

    /// Pre-allocated receive buffer for the `allreduce(Sum)` that aggregates
    /// per-slot binding increments across MPI ranks.
    pub(crate) global_increments_buf: Vec<u64>,

    /// Pre-allocated send buffer for the per-iteration `allreduce(BitwiseOr)`
    /// that aggregates sliding-window binding-activity bitmaps across MPI ranks.
    pub(crate) metadata_sync_window_buf: Vec<u32>,

    /// Pre-allocated receive buffer for the per-iteration `allreduce(BitwiseOr)`.
    pub(crate) global_window_increments_buf: Vec<u32>,

    /// Pre-allocated buffer for packing real (non-padded) gathered state
    /// vectors when archiving visited states for dominated cut selection.
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
    /// Staging buffer for cuts produced by one stage's parallel trial-point loop.
    ///
    /// Cleared at the start of each stage and grown monotonically.
    pub(crate) staged_cuts_buf: Vec<StagedCut>,

    /// Per-worker solver statistics snapshot taken **before** the stage's parallel
    /// region. Cleared and repopulated each stage.
    pub(crate) worker_stats_before: Vec<SolverStatistics>,

    /// Per-worker solver statistics snapshot taken **after** the stage's parallel
    /// region. Cleared and repopulated each stage.
    pub(crate) worker_stats_after: Vec<SolverStatistics>,

    /// Per-worker solver-statistics delta for this stage (after − before).
    /// Cleared and repopulated each stage.
    pub(crate) worker_deltas: Vec<SolverStatsDelta>,

    /// Per-worker total work time (solve + load + set-bounds + basis-set) for
    /// load-imbalance decomposition. Cleared and repopulated each stage.
    pub(crate) worker_totals: Vec<f64>,
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
    pub fn new(
        n_workers_local: usize,
        n_ranks: usize,
        bwd_max_openings: usize,
        real_states_capacity: usize,
    ) -> Self {
        let send_stride = n_workers_local * bwd_max_openings * WORKER_STATS_ENTRY_STRIDE;
        Self {
            probabilities_buf: Vec::new(),
            successor_active_slots_buf: Vec::new(),
            metadata_sync_buf: Vec::new(),
            global_increments_buf: Vec::new(),
            metadata_sync_window_buf: Vec::new(),
            global_window_increments_buf: Vec::new(),
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
        }
    }

    /// Execute the backward pass for one training iteration on this rank.
    ///
    /// Sweeps stages from `num_stages - 2` down to `0`. For each stage,
    /// the trial-point loop is parallelised with static scenario partitioning.
    /// Each worker generates cuts into a thread-local buffer, sorted by
    /// trial-point index before FCF insertion.
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
    /// - `inputs.ctx.base_rows.len() != num_stages`
    /// - `inputs.risk_measures.len() != num_stages`
    /// - `inputs.baked.len() != num_stages`
    pub fn run<S: SolverInterface + Send, C: Communicator>(
        &mut self,
        inputs: &mut BackwardPassInputs<'_, S, C>,
    ) -> Result<BackwardResult, SddpError> {
        let training_ctx = inputs.training_ctx;
        let num_stages = training_ctx.horizon.num_stages();

        debug_assert_eq!(inputs.ctx.templates.len(), num_stages);
        debug_assert_eq!(inputs.ctx.base_rows.len(), num_stages);
        debug_assert_eq!(inputs.risk_measures.len(), num_stages);
        debug_assert_eq!(
            inputs.baked.len(),
            num_stages,
            "baked.len() must equal num_stages"
        );

        let start = Instant::now();
        let solves_before: u64 = inputs
            .workspaces
            .iter()
            .map(|ws| ws.solver.statistics().solve_count)
            .sum();

        // Reset per-worker timing and iteration-scoped window buffers.
        for ws in inputs.workspaces.iter_mut() {
            ws.worker_timing_buf = WorkerPhaseTimings::default();
            ws.backward_accum.metadata_sync_window_contribution.fill(0);
        }
        self.metadata_sync_window_buf.fill(0);
        self.global_window_increments_buf.fill(0);

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
        let mut stage_stats: Vec<(usize, Vec<StageWorkerOpeningDelta>)> = Vec::new();
        let mut state_exchange_ms: u64 = 0;
        let mut cut_batch_build_ms: u64 = 0;
        let mut setup_ms: u64 = 0;
        let mut imbalance_ms: u64 = 0;
        let mut scheduling_ms: u64 = 0;
        let mut cut_sync_ms: u64 = 0;

        for t in (0..num_stages.saturating_sub(1)).rev() {
            let out = run_one_backward_stage(self, inputs, t, &params)?;
            cuts_generated += out.cuts_generated;
            state_exchange_ms += out.state_exchange_ms;
            cut_batch_build_ms += out.cut_batch_build_ms;
            setup_ms += out.setup_ms;
            imbalance_ms += out.imbalance_ms;
            scheduling_ms += out.scheduling_ms;
            cut_sync_ms += out.cut_sync_ms;
            stage_stats.push((t + 1, out.stage_entries));
        }

        // Clear iteration-scoped window buffers after all stages.
        for ws in inputs.workspaces.iter_mut() {
            ws.backward_accum.metadata_sync_window_contribution.fill(0);
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

        #[allow(clippy::cast_possible_truncation)]
        let elapsed_ms = start.elapsed().as_millis() as u64;
        let solves_after: u64 = inputs
            .workspaces
            .iter()
            .map(|ws| ws.solver.statistics().solve_count)
            .sum();

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

    /// Synchronise per-slot cut binding metadata across MPI ranks for one stage.
    ///
    /// Performs two `allreduce` operations:
    /// 1. `Sum` over `metadata_sync_contribution` to accumulate `active_count` and
    ///    `last_active_iter` across all ranks.
    /// 2. `BitwiseOr` over `metadata_sync_window_contribution` to merge the
    ///    iteration-level sliding-window activity bit.
    ///
    /// Clears the window accumulation buffers after the `allreduce` so they are
    /// ready for the next stage.  Called once per backward stage after cut
    /// insertion.
    fn sync_stage_metadata<C: Communicator>(
        &mut self,
        successor: usize,
        iteration: u64,
        workspaces: &[SolverWorkspace<impl SolverInterface>],
        fcf: &mut FutureCostFunction,
        comm: &C,
    ) -> Result<(), SddpError> {
        let pool_size = fcf.pools[successor].metadata.len();
        if pool_size == 0 {
            return Ok(());
        }
        // Sum per-worker binding increment contributions into the send buffer.
        self.metadata_sync_buf.clear();
        self.metadata_sync_buf.resize(pool_size, 0u64);
        for ws in workspaces {
            for (slot, &inc) in ws
                .backward_accum
                .metadata_sync_contribution
                .iter()
                .enumerate()
                .take(pool_size)
            {
                self.metadata_sync_buf[slot] += inc;
            }
        }
        self.global_increments_buf.clear();
        self.global_increments_buf.resize(pool_size, 0u64);
        comm.allreduce(
            &self.metadata_sync_buf,
            &mut self.global_increments_buf,
            ReduceOp::Sum,
        )
        .map_err(SddpError::from)?;
        for (slot, &inc) in self.global_increments_buf.iter().enumerate() {
            if inc > 0 {
                fcf.pools[successor].metadata[slot].active_count += inc;
                fcf.pools[successor].metadata[slot].last_active_iter = iteration;
            }
        }
        // BitwiseOr per-worker sliding-window bits into the global window buffer.
        self.metadata_sync_window_buf.resize(pool_size, 0u32);
        for ws in workspaces {
            for (slot, &bit) in ws
                .backward_accum
                .metadata_sync_window_contribution
                .iter()
                .enumerate()
                .take(pool_size)
            {
                self.metadata_sync_window_buf[slot] |= bit;
            }
        }
        self.global_window_increments_buf.resize(pool_size, 0u32);
        comm.allreduce(
            &self.metadata_sync_window_buf,
            &mut self.global_window_increments_buf,
            ReduceOp::BitwiseOr,
        )
        .map_err(SddpError::from)?;
        for (slot, &bits) in self.global_window_increments_buf.iter().enumerate() {
            if bits & 1 != 0 {
                fcf.pools[successor].metadata[slot].active_window |= 1u32;
            }
        }
        self.metadata_sync_window_buf.fill(0);
        self.global_window_increments_buf.fill(0);
        Ok(())
    }

    /// Collect per-worker timing statistics for one stage and decompose
    /// parallel overhead into setup, load-imbalance, and scheduling components.
    ///
    /// Snapshots solver statistics after the parallel region, computes per-worker
    /// deltas against the before-snapshot already stored in `self.worker_stats_before`,
    /// updates `worker_timing_buf.bwd_setup_ms` on each workspace, and returns
    /// `(setup_ms_delta, imbalance_ms_delta, scheduling_ms_delta)`.
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
        let stage_setup_ms: f64 = self
            .worker_deltas
            .iter()
            .map(|d| d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms)
            .sum();
        for (ws, delta) in workspaces.iter_mut().zip(&self.worker_deltas) {
            ws.worker_timing_buf.bwd_setup_ms +=
                delta.load_model_time_ms + delta.set_bounds_time_ms + delta.basis_set_time_ms;
        }
        self.worker_totals.clear();
        self.worker_totals
            .extend(self.worker_deltas.iter().map(|d| {
                d.solve_time_ms + d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms
            }));
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
        // Copy per-worker per-opening stats into the local StageWorkerStatsBuffer.
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
        let mut entries: Vec<StageWorkerOpeningDelta> = Vec::new();
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
                    let delta = self.bwd_stats_unpack_buf[flat].clone();
                    if delta.lp_solves > 0 || omega == 0 {
                        entries.push((rank_i32, wid_i32, omega, delta));
                    }
                }
            }
        }
        Ok(entries)
    }
}

/// Iteration-constant values derived once from `BackwardPassInputs` at the start of `run`.
///
/// Passed to `run_one_backward_stage` to avoid recomputing them on every loop iteration
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

/// Per-stage output produced by `run_one_backward_stage`.
struct StageOutput {
    /// Number of cuts generated at this stage (always equals `local_work`).
    cuts_generated: usize,
    /// Per-`(rank, worker_id, opening)` solver delta entries for this stage.
    stage_entries: Vec<StageWorkerOpeningDelta>,
    /// Time spent in `exchange.exchange()` for this stage, in milliseconds.
    state_exchange_ms: u64,
    /// Time spent in `build_delta_cut_row_batch_into`, in milliseconds.
    cut_batch_build_ms: u64,
    /// Aggregate non-solve setup time (`load_model` + `set_bounds` + `basis_set`) in ms.
    setup_ms: u64,
    /// Load-imbalance component: `max_worker_total_ms − avg_worker_total_ms`, in ms.
    imbalance_ms: u64,
    /// Scheduling overhead: `parallel_wall_ms − max_worker_total_ms`, in ms.
    scheduling_ms: u64,
    /// Time spent in per-stage cut-sync `allgatherv`, in milliseconds.
    cut_sync_ms: u64,
}

/// Execute one backward stage (index `t`, solving at successor `t + 1`).
///
/// Performs state exchange, builds the successor LP batch, runs the parallel
/// trial-point loop, inserts cuts, syncs cuts across ranks, syncs cut metadata,
/// and collects per-worker solver statistics.
///
/// Returns a [`StageOutput`] accumulating all timing components and the cut
/// entries for this stage.
fn run_one_backward_stage<S: SolverInterface + Send, C: Communicator>(
    state: &mut BackwardPassState,
    inputs: &mut BackwardPassInputs<'_, S, C>,
    t: usize,
    params: &StageDerivedParams,
) -> Result<StageOutput, SddpError> {
    let training_ctx = inputs.training_ctx;
    let ctx = inputs.ctx;
    let indexer = training_ctx.indexer;
    let num_stages = training_ctx.horizon.num_stages();
    let successor = t + 1;

    // State exchange: gather trial-point states for stage `t` via allgatherv.
    let mut state_exchange_ms: u64 = 0;
    if !inputs.records.is_empty() {
        let exch_start = Instant::now();
        inputs
            .exchange
            .exchange(inputs.records, t, num_stages, inputs.comm)?;
        #[allow(clippy::cast_possible_truncation)]
        {
            state_exchange_ms = exch_start.elapsed().as_millis() as u64;
        }
    }

    // Archive visited states for dominated cut selection (if active).
    if let Some(ref mut archive) = inputs.visited_archive {
        let total_fwd = inputs.exchange.real_total_scenarios();
        inputs
            .exchange
            .pack_real_states_into(&mut state.real_states_buf);
        archive.archive_gathered_states(t, &state.real_states_buf, total_fwd);
    }

    // Build uniform opening probabilities and delta cut batch for the successor.
    state.worker_stats_before.clear();
    state
        .worker_stats_before
        .extend(inputs.workspaces.iter().map(|w| w.solver.statistics()));

    let n_openings = training_ctx.stochastic.tree_view().n_openings(successor);
    state.probabilities_buf.clear();
    #[allow(clippy::cast_precision_loss)]
    state
        .probabilities_buf
        .resize(n_openings, 1.0_f64 / n_openings as f64);

    let batch_start = Instant::now();
    let template_num_rows = ctx.templates[successor].num_rows;
    build_delta_cut_row_batch_into(
        &mut inputs.cut_batches[successor],
        inputs.fcf,
        successor,
        indexer,
        &ctx.templates[successor].col_scale,
        inputs.iteration,
    );
    let baked_tmpl = &inputs.baked[successor];
    let num_cuts_at_successor =
        (baked_tmpl.num_rows - template_num_rows) + inputs.cut_batches[successor].num_rows;
    #[allow(clippy::cast_possible_truncation)]
    let cut_batch_build_ms = batch_start.elapsed().as_millis() as u64;

    state.successor_active_slots_buf.clear();
    state
        .successor_active_slots_buf
        .extend(inputs.fcf.active_cuts(successor).map(|(slot, _, _)| slot));

    let succ_spec = SuccessorSpec {
        t,
        successor,
        my_rank: params.my_rank,
        probabilities: &state.probabilities_buf,
        cut_batch: &inputs.cut_batches[successor],
        num_cuts_at_successor,
        template_num_rows,
        baked_template: baked_tmpl,
        successor_active_slots: &state.successor_active_slots_buf,
        cut_activity_tolerance: inputs.cut_activity_tolerance,
        basis_activity_window: inputs.basis_activity_window,
        successor_populated_count: inputs.fcf.pools[successor].populated_count,
        successor_pool: &inputs.fcf.pools[successor],
    };

    // Parallel trial-point solves.
    let basis_slices = inputs
        .basis_store
        .split_workers_mut(params.n_workers_local.max(1));
    let process_start = Instant::now();
    let worker_staged = process_stage_backward(
        inputs.workspaces,
        ctx,
        training_ctx,
        inputs.local_work,
        inputs.exchange,
        inputs.fwd_offset,
        inputs.iteration,
        inputs.risk_measures,
        &succ_spec,
        basis_slices,
    );
    #[allow(clippy::cast_possible_truncation)]
    let parallel_wall_ms = process_start.elapsed().as_millis() as u64;

    // Collect cuts and insert into FCF in deterministic order.
    state.staged_cuts_buf.clear();
    for worker_result in worker_staged {
        state.staged_cuts_buf.extend(worker_result?);
    }
    state.staged_cuts_buf.sort_by_key(|cut| cut.trial_point_idx);
    debug_assert_eq!(state.staged_cuts_buf.len(), inputs.local_work);
    let cuts_generated = state.staged_cuts_buf.len();
    for cut in &state.staged_cuts_buf {
        inputs.fcf.add_cut(
            t,
            inputs.iteration,
            cut.forward_pass_index,
            cut.intercept,
            &cut.coefficients,
        );
    }

    // Per-stage cut sync across MPI ranks.
    let sync_start = Instant::now();
    let n_local = inputs
        .cut_sync_bufs
        .pack_local_cuts(inputs.fcf, t, inputs.iteration);
    inputs
        .cut_sync_bufs
        .sync_packed_cuts(t, n_local, inputs.fcf, inputs.comm)?;
    #[allow(clippy::cast_possible_truncation)]
    let cut_sync_ms = sync_start.elapsed().as_millis() as u64;

    state.sync_stage_metadata(
        successor,
        inputs.iteration,
        inputs.workspaces,
        inputs.fcf,
        inputs.comm,
    )?;

    let (setup_ms, imbalance_ms, scheduling_ms) =
        state.collect_stage_timing_stats(parallel_wall_ms, params.n_workers, inputs.workspaces);

    let stage_entries = state.gather_stage_solver_stats(
        n_openings,
        params.n_ranks,
        params.n_workers_local,
        params.bwd_max_openings,
        inputs.workspaces,
        inputs.comm,
    )?;

    Ok(StageOutput {
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

/// Evaluate all trial points for a single backward stage, returning staged cuts.
///
/// This is the `BackwardPassState`-aware replacement for the free function of the
/// same name in `backward.rs`. It receives the per-call buffers that are needed
/// inside the rayon parallel region as individual parameters (rather than a
/// `BackwardPassState` borrow) to avoid whole-struct borrow conflicts across the
/// parallel closure boundary.
// RATIONALE: 10 args are individually-borrowed slices passed through the rayon closure
// boundary. Bundling them into a struct would require either cloning or an `Arc`, both of
// which conflict with the zero-allocation HPC constraint for backward-pass hot code.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_stage_backward<S: SolverInterface + Send>(
    workspaces: &mut [SolverWorkspace<S>],
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    local_work: usize,
    exchange: &ExchangeBuffers,
    fwd_offset: usize,
    iteration: u64,
    risk_measures: &[RiskMeasure],
    succ: &SuccessorSpec<'_>,
    basis_slices: Vec<BasisStoreSliceMut<'_>>,
) -> Vec<Result<Vec<StagedCut>, SddpError>> {
    let n_openings = succ.probabilities.len();
    let n_state = training_ctx.indexer.n_state;
    let pop = succ.successor_populated_count;
    let n_workers = workspaces.len().max(1);

    workspaces
        .par_iter_mut()
        .zip(basis_slices.into_par_iter())
        .enumerate()
        .map(|(w, (ws, mut basis_slice))| {
            // Load template and pre-allocate per-stage buffers.
            load_backward_lp(ws, succ);
            while ws.backward_accum.outcomes.len() < n_openings {
                ws.backward_accum
                    .outcomes
                    .push(crate::risk_measure::BackwardOutcome {
                        intercept: 0.0,
                        coefficients: vec![0.0_f64; n_state],
                        objective_value: 0.0,
                    });
            }
            if ws.backward_accum.slot_increments.len() < pop {
                ws.backward_accum.slot_increments.resize(pop, 0u64);
            }
            if ws.backward_accum.agg_coefficients.len() < n_state {
                ws.backward_accum.agg_coefficients.resize(n_state, 0.0_f64);
            }
            if ws.backward_accum.metadata_sync_contribution.len() < pop {
                ws.backward_accum
                    .metadata_sync_contribution
                    .resize(pop, 0u64);
            }
            ws.backward_accum.metadata_sync_contribution[..pop].fill(0);
            // Per-stage clear: slot indices in the contribution buffer are
            // per-pool. Slot N in pool[s] is a different cut than slot N in
            // pool[s+1], but the buffer is shared across all stages within an
            // iteration. Without this clear, bits set while processing stage
            // s+1 (binding observations on pool[s+1] cuts) leak into stage
            // s's sync (incorrectly setting active_window bit 0 on pool[s]
            // cuts at the same slot index).
            if ws.backward_accum.metadata_sync_window_contribution.len() < pop {
                ws.backward_accum
                    .metadata_sync_window_contribution
                    .resize(pop, 0u32);
            }
            ws.backward_accum.metadata_sync_window_contribution[..pop].fill(0);
            ws.backward_accum
                .per_opening_stats
                .resize_with(n_openings, SolverStatsDelta::default);
            for slot in &mut ws.backward_accum.per_opening_stats[..n_openings] {
                *slot = SolverStatsDelta::default();
            }

            // Static partition: assign scenarios to worker, matching basis_slice view.
            let (start_m, end_m) = partition(local_work, n_workers, w);
            // Reuse the per-worker staged-cuts buffer; `clear()` preserves the
            // allocation from prior stages so no heap activity occurs after the
            // first stage of the first iteration.
            ws.backward_accum.staged_cuts_buf.clear();
            let worker_stage_wall_start = Instant::now();

            for m in start_m..end_m {
                // Reload LP per trial point to reset HiGHS's internal simplex
                // basis, factorization, and RNG position.
                load_backward_lp(ws, succ);
                ws.backward_accum.slot_increments[..pop].fill(0);
                // Call process_trial_point_backward before the push to avoid a
                // simultaneous mutable borrow of `ws.backward_accum.staged_cuts_buf`
                // (for the push receiver) and `ws` (for the function argument).
                let cut = process_trial_point_backward(
                    ws,
                    ctx,
                    training_ctx,
                    exchange,
                    fwd_offset,
                    iteration,
                    risk_measures,
                    succ,
                    &mut basis_slice,
                    m,
                )?;
                ws.backward_accum.staged_cuts_buf.push(cut);
            }

            // Accumulate per-worker elapsed into the iteration-level timing buffer.
            ws.worker_timing_buf.backward_wall_ms +=
                worker_stage_wall_start.elapsed().as_secs_f64() * 1_000.0;

            // Drain the buffer into an owned Vec to cross the rayon closure
            // boundary.  `drain(..)` leaves `staged_cuts_buf` empty with its
            // capacity intact so the next stage reuses the same allocation.
            Ok(ws.backward_accum.staged_cuts_buf.drain(..).collect())
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
    use cobre_core::scenario::SamplingScheme;
    use cobre_solver::{
        Basis, LpSolution, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };

    use crate::{
        context::{StageContext, TrainingContext},
        cut::FutureCostFunction,
        cut_sync::CutSyncBuffers,
        horizon_mode::HorizonMode,
        indexer::StageIndexer,
        inflow_method::InflowNonNegativityMethod,
        risk_measure::RiskMeasure,
        solver_stats::WORKER_STATS_ENTRY_STRIDE,
        state_exchange::ExchangeBuffers,
        trajectory::TrajectoryRecord,
        workspace::{BackwardAccumulators, BasisStore, SolverWorkspace},
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
    }

    fn minimal_template_1_0() -> StageTemplate {
        StageTemplate {
            num_cols: 3,
            num_rows: 1,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![f64::INFINITY; 3],
            objective: vec![0.0, 0.0, 1.0],
            row_lower: vec![0.0],
            row_upper: vec![0.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
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

    fn single_workspace(solver: MockSolver, n_state: usize) -> Vec<SolverWorkspace<MockSolver>> {
        use crate::lp_builder::PatchBuffer;
        vec![SolverWorkspace {
            rank: 0,
            worker_id: 0,
            solver,
            patch_buf: PatchBuffer::new(1, 0, 0, 0),
            current_state: Vec::with_capacity(n_state),
            scratch: crate::workspace::ScratchBuffers {
                noise_buf: Vec::new(),
                inflow_m3s_buf: Vec::new(),
                lag_matrix_buf: Vec::new(),
                par_inflow_buf: Vec::new(),
                eta_floor_buf: Vec::new(),
                zero_targets_buf: Vec::new(),
                ncs_col_upper_buf: Vec::new(),
                ncs_col_lower_buf: Vec::new(),
                ncs_col_indices_buf: Vec::new(),
                load_rhs_buf: Vec::new(),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::new(),
                effective_eta_buf: Vec::new(),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
                lag_accumulator: Vec::new(),
                lag_weight_accum: 0.0,
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                current_state_scratch: Vec::new(),
                recon_slot_lookup: Vec::new(),
                promotion_scratch: crate::basis_reconstruct::PromotionScratch::default(),
                trajectory_costs_buf: Vec::new(),
                raw_noise_buf: Vec::new(),
                perm_scratch: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: WorkerPhaseTimings::default(),
        }]
    }

    fn empty_basis_store(num_scenarios: usize, num_stages: usize) -> BasisStore {
        BasisStore::new(num_scenarios, num_stages)
    }

    fn exchange_with_states(n_state: usize, states: Vec<Vec<f64>>) -> ExchangeBuffers {
        use cobre_comm::LocalBackend;
        let local_count = states.len();
        let mut bufs = ExchangeBuffers::new(n_state, local_count, 1);
        let records: Vec<TrajectoryRecord> = states
            .into_iter()
            .map(|state| TrajectoryRecord {
                primal: vec![],
                dual: vec![],
                stage_cost: 0.0,
                state,
            })
            .collect();
        let comm = LocalBackend;
        bufs.exchange(&records, 0, 1, &comm).unwrap();
        bufs
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
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let hydro = Hydro {
            id: EntityId(1),
            name: "H1".to_string(),
            bus_id: EntityId(0),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 1.0,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
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
                fpha_turbined_cost: 0.0,
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
            .map(|idx| cobre_core::scenario::InflowModel {
                hydro_id: EntityId(1),
                stage_id: idx as i32,
                mean_m3s: 100.0,
                std_m3s: 30.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
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
        // n_workers_local=2, n_ranks=3, bwd_max_openings=5
        let n_workers_local = 2_usize;
        let n_ranks = 3_usize;
        let bwd_max_openings = 5_usize;
        let real_states_capacity = 10_usize;

        let state = BackwardPassState::new(
            n_workers_local,
            n_ranks,
            bwd_max_openings,
            real_states_capacity,
        );

        let send_stride = n_workers_local * bwd_max_openings * WORKER_STATS_ENTRY_STRIDE;

        // Empty/zero-sized on construction (grown lazily):
        assert!(state.probabilities_buf.is_empty());
        assert!(state.successor_active_slots_buf.is_empty());
        assert!(state.metadata_sync_buf.is_empty());
        assert!(state.global_increments_buf.is_empty());
        assert!(state.metadata_sync_window_buf.is_empty());
        assert!(state.global_window_increments_buf.is_empty());

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
    /// Setup mirrors `two_stage_system_two_trial_points_generates_two_cuts_at_stage_0`
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
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];
        let n_state = indexer.n_state;
        let forward_passes = 2_u32;

        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0], vec![20.0]]);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let comm = StubComm;
        let mut workspaces = single_workspace(MockSolver::always_ok(solution), n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let mut cut_batches = empty_cut_batches(n_stages);
        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
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
        };
        let training_ctx = TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
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
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
        };

        // Allocate BackwardPassState sized for this minimal study.
        let bwd_max_openings = n_openings;
        let mut state = BackwardPassState::new(1, 1, bwd_max_openings, n_state);
        // Capture local_count before mutably borrowing exchange inside the struct literal.
        let local_count = exchange.local_count();

        let mut inputs = BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &ctx,
            baked: &templates,
            fcf: &mut fcf,
            cut_batches: &mut cut_batches,
            training_ctx: &training_ctx,
            comm: &comm,
            exchange: &mut exchange,
            records: &[],
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            event_sender: None,
            risk_measures: &risk_measures,
            cut_activity_tolerance: 0.0,
            basis_activity_window: crate::basis_reconstruct::DEFAULT_BASIS_ACTIVITY_WINDOW,
            iteration: 1,
            local_work: local_count,
            fwd_offset: 0,
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

    /// Verify that `state_duals_buf` on the per-worker `BackwardAccumulators`
    /// is correctly sized after the backward pass completes.
    ///
    /// After `BackwardPassState::run` returns, the single worker's
    /// `backward_accum.state_duals_buf` must hold exactly `indexer.n_state`
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
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];
        let n_state = indexer.n_state;
        let forward_passes = 2_u32;

        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0], vec![20.0]]);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let comm = StubComm;
        let mut workspaces = single_workspace(MockSolver::always_ok(solution), n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let mut cut_batches = empty_cut_batches(n_stages);
        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
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
        };
        let training_ctx = TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
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
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
        };

        let bwd_max_openings = n_openings;
        let mut state = BackwardPassState::new(1, 1, bwd_max_openings, n_state);
        let local_count = exchange.local_count();

        let mut inputs = BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &ctx,
            baked: &templates,
            fcf: &mut fcf,
            cut_batches: &mut cut_batches,
            training_ctx: &training_ctx,
            comm: &comm,
            exchange: &mut exchange,
            records: &[],
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            event_sender: None,
            risk_measures: &risk_measures,
            cut_activity_tolerance: 0.0,
            basis_activity_window: crate::basis_reconstruct::DEFAULT_BASIS_ACTIVITY_WINDOW,
            iteration: 1,
            local_work: local_count,
            fwd_offset: 0,
        };

        let _ = state
            .run(&mut inputs)
            .expect("backward pass must not error");

        // After the backward pass, the sole worker's `state_duals_buf` must
        // hold exactly `n_state` entries — the duals from the last opening
        // processed during the last trial-point/stage iteration.
        assert_eq!(
            inputs.workspaces[0].backward_accum.state_duals_buf.len(),
            n_state,
            "state_duals_buf must have length n_state after backward pass"
        );
    }
}
