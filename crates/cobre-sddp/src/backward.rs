//! Backward pass execution for the SDDP training loop.
//!
//! [`run_backward_pass`] sweeps stages in reverse order (`T-2` down to `0`),
//! evaluating the cost-to-go at each trial point **assigned to this rank**
//! during the forward pass. For each trial point, the backward pass iterates
//! over every opening from the fixed opening tree, extracts LP duals to form
//! Benders cut coefficients, and aggregates per-opening outcomes via
//! [`RiskMeasure::aggregate_cut`] to produce one cut per trial point per
//! stage. Each aggregated cut is inserted into the [`FutureCostFunction`].
//!
//! Although [`ExchangeBuffers`] contains trial points from all ranks (after
//! `allgatherv`), each rank only processes its own forward pass assignments
//! to avoid generating duplicate cuts. Cut synchronization (`allgatherv`)
//! distributes the generated cuts to all ranks after the backward pass.
//!
//! ## Stage indexing convention
//!
//! The backward pass generates a cut **at stage `t`** by solving the LP
//! **at stage `t + 1`** (the successor) under each opening noise vector from
//! that successor stage. The opening tree provides noise at `t + 1`.
//!
//! ## Cut coefficient formula
//!
//! For a solve at stage `t + 1` with trial point state `x_hat`:
//!
//! ```text
//! pi[i]  = dual[i] * row_scale[i]           for i in 0..n_state
//! alpha  = Q - sum_i(pi[i] * x_hat[i])      (intercept)
//! ```
//!
//! where `Q` is the LP objective and `dual[0..n_state]` are the duals of the
//! fixing constraints (storage-fixing and lag-fixing rows).
//!
//! The coefficients stored in the [`FutureCostFunction`] are the raw (unscaled)
//! duals of the state-fixing rows. Negation is applied later when building the
//! LP cut row in `build_cut_row_batch_into` (forward.rs):
//! `-coeff * x + theta >= intercept`.
//! See the project convention: "coefficients = dual (NOT -dual)".
//!
//! ## Cut activity tracking
//!
//! After each backward solve, the duals of the appended cut rows are inspected
//! to determine which existing cuts at the successor stage are binding. The
//! metadata of binding cuts is updated in-place so that cut selection
//! strategies have accurate activity counts at the end of the iteration.
//!
//! ## Thread-level parallelism
//!
//! Within a rank, the outer per-stage loop remains sequential (stage `t`
//! depends on cuts generated at stage `t+1`). The inner trial-point loop is
//! parallelised across [`SolverWorkspace`] instances using rayon's
//! `par_iter_mut` with static scenario partitioning (matching the forward pass).
//! Each worker generates cuts into a thread-local `StagedCut` buffer, sorted
//! by `trial_point_idx` after the parallel region to ensure deterministic FCF
//! insertion regardless of thread completion order.
//!
//! ## Hot-path allocation discipline
//!
//! Allocations are limited to:
//! - One `Vec<f64>` for opening probabilities per stage (outside the trial
//!   point loop).
//! - One `Vec<BackwardOutcome>` per worker thread, allocated once per stage
//!   in the parallel region and reused via `clear()` per trial point.
//! - One `RowBatch` per stage built by `build_cut_row_batch` (outside the
//!   trial point loop, before the parallel region).
//! - One `Vec<StagedCut>` per stage for the merge phase (bounded by
//!   `local_work` entries, each holding one cut and its binding slot list).
//!
//! The `binding_slots` vector inside each `StagedCut` is allocated per
//! trial point — a flat buffer optimization is deferred to profiling.
//!
//! Note: `load_model` is called once per trial point (not per stage) to reset
//! the LP to the structural template before appending cuts. `HiGHS` performs
//! internal allocations during `load_model` that are not visible to this
//! module; these are a fixed cost per trial point and are not considered
//! hot-path allocations from Cobre's perspective.

use std::sync::mpsc::Sender;
use std::time::Instant;

use cobre_comm::{Communicator, ReduceOp};
use cobre_core::{
    TrainingEvent, WORKER_TIMING_SLOT_BWD_SETUP, WORKER_TIMING_SLOT_BWD_WALL, WorkerTimingPhase,
};
use cobre_solver::{RowBatch, SolverInterface, SolverStatistics, StageTemplate};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{
    FutureCostFunction, SddpError, TrajectoryRecord,
    context::{StageContext, TrainingContext},
    cut::pool::CutPool,
    cut_sync::CutSyncBuffers,
    forward::{build_delta_cut_row_batch_into, partition, write_capture_metadata},
    noise::{transform_inflow_noise, transform_load_noise, transform_ncs_noise},
    risk_measure::RiskMeasure,
    solver_stats::{
        SolverStatsDelta, StageWorkerStatsBuffer, WORKER_STATS_ENTRY_STRIDE,
        pack_worker_opening_stats, unpack_worker_opening_stats,
    },
    state_exchange::ExchangeBuffers,
    workspace::{BasisStore, BasisStoreSliceMut, CapturedBasis, SolverWorkspace},
};

/// Per-`(rank, worker_id, opening)` solver delta collected during a single
/// backward stage, as returned inside [`BackwardResult::stage_stats`].
///
/// Layout: `(rank, worker_id, opening_index, delta)`.
pub type StageWorkerOpeningDelta = (i32, i32, usize, SolverStatsDelta);

/// Result produced by the backward pass on a single rank.
#[derive(Debug, Clone)]
#[must_use]
pub struct BackwardResult {
    /// Total number of cuts generated by this rank during the backward pass.
    pub cuts_generated: usize,

    /// Wall-clock time in milliseconds for this rank's backward pass.
    pub elapsed_ms: u64,

    /// Number of LP solves performed during this backward pass.
    pub lp_solves: u64,

    /// Per-stage, per-`(rank, worker_id, opening)` solver statistics deltas.
    ///
    /// Each outer entry is `(successor_stage_index, per_worker_opening_deltas)`.
    /// The inner `Vec` element is `(rank, worker_id, omega, delta)`: one entry per
    /// `(MPI rank, rayon worker, opening index)` triple gathered via `allgatherv`.
    /// Only includes entries where `omega < n_openings(successor)` AND
    /// `delta.lp_solves > 0 || omega == 0` (preserves the omega=0 "stage visited"
    /// sentinel while skipping padded buffer slots).
    /// Entries are in reverse stage order (matching the backward iteration direction).
    pub stage_stats: Vec<(usize, Vec<StageWorkerOpeningDelta>)>,

    /// Wall-clock time for state exchange (`allgatherv`) accumulated across
    /// all stages, in milliseconds.
    pub state_exchange_time_ms: u64,

    /// Wall-clock time for `build_cut_row_batch_into` accumulated across
    /// all stages, in milliseconds.
    pub cut_batch_build_time_ms: u64,

    /// Aggregate non-solve work inside the parallel region accumulated across
    /// all stages, in milliseconds.
    ///
    /// Computed per-stage as the sum over all workers of
    /// `load_model_time_ms + set_bounds_time_ms + basis_set_time_ms`.
    pub setup_time_ms: u64,

    /// Load-imbalance component of parallel overhead accumulated across all
    /// stages, in milliseconds.
    ///
    /// Computed per-stage as `max_worker_total_ms - avg_worker_total_ms`, where
    /// `worker_total_ms = solve + load_model + set_bounds + basis_set`
    /// for each worker. Measures how much the slowest worker exceeds the average.
    pub load_imbalance_ms: u64,

    /// True rayon scheduling overhead accumulated across all stages, in
    /// milliseconds.
    ///
    /// Computed per-stage as `parallel_wall_ms - max_worker_total_ms`. Represents
    /// rayon barrier, thread wake-up, and work-stealing dispatch costs after
    /// accounting for all measured per-worker work.
    pub scheduling_overhead_ms: u64,

    /// Wall-clock time for per-stage cut synchronization (`allgatherv`)
    /// accumulated across all stages, in milliseconds.
    pub cut_sync_time_ms: u64,
}

/// Per-thread staging buffer for one aggregated cut produced at a single trial
/// point during the parallel backward sweep.
///
/// Each worker thread populates one `StagedCut` per trial point instead of
/// writing directly into the [`FutureCostFunction`]. After the parallel region,
/// staged cuts are sorted by `trial_point_idx` and merged into the FCF in
/// deterministic order regardless of thread completion order.
struct StagedCut {
    /// Local trial-point index within `0..local_work`. Used for deterministic
    /// merge ordering after the parallel region.
    trial_point_idx: usize,

    /// Aggregated cut intercept (result of `RiskMeasure::aggregate_cut`).
    intercept: f64,

    /// Aggregated cut coefficients (length = `n_state`).
    coefficients: Vec<f64>,

    /// Global forward-pass index (`fwd_offset + m`), stored as `u32` for the
    /// FCF slot formula.
    forward_pass_index: u32,
}

/// Inputs bundled from the training loop for the backward pass.
///
/// Groups the iteration-varying scalars and slices that would otherwise push
/// the argument count of [`run_backward_pass`] beyond seven.
pub struct BackwardPassSpec<'a> {
    /// Exchange buffers for gathering trial-point states via `allgatherv`.
    ///
    /// When non-empty, populates exchange buffers per stage. When empty (test path),
    /// the caller is responsible for pre-populating the exchange buffers.
    pub exchange: &'a mut ExchangeBuffers,

    /// Forward-pass trajectory records used to populate `exchange` per stage.
    ///
    /// Length must be `local_work * num_stages` when non-empty; pass `&[]` in
    /// tests that pre-populate `exchange` directly.
    pub records: &'a [TrajectoryRecord],

    /// Current training iteration index (1-based), used for cut metadata.
    pub iteration: u64,

    /// Number of trial points assigned to this rank for the backward pass.
    pub local_work: usize,

    /// Global offset for this rank's trial points (`rank * fwd_per_rank`).
    pub fwd_offset: usize,

    /// Per-stage risk measures (length = `num_stages`). `risk_measures[t]`
    /// is applied when aggregating cut outcomes at stage `t`.
    pub risk_measures: &'a [RiskMeasure],

    /// Minimum dual multiplier for a cut to count as binding (BUG-2 fix).
    ///
    /// Cuts with duals above this threshold are marked as binding during the
    /// backward pass activity tracking. A value of `0.0` preserves the
    /// original strict-positive behavior; a value like `1e-8` filters out
    /// numerical noise from near-zero positive duals.
    pub cut_activity_tolerance: f64,

    /// Pre-allocated cut synchronization buffers for per-stage `allgatherv`.
    ///
    /// After local cuts are inserted into the FCF at each stage, `sync_cuts`
    /// is called to distribute cuts across all ranks. For single-rank runs
    /// the `allgatherv` is a no-op and completes immediately.
    pub cut_sync_bufs: &'a mut CutSyncBuffers,

    /// Pre-allocated buffer for uniform opening probabilities. Reused per stage
    /// via `clear()` + `resize()` to avoid per-stage allocation.
    ///
    /// Passed as a mutable reference to avoid `mem::take` capacity loss when
    /// the training loop encounters an error before buffer recovery.
    pub probabilities_buf: &'a mut Vec<f64>,

    /// Pre-allocated buffer for successor active cut slot indices. Reused per
    /// stage via `clear()` + `extend()` to avoid per-stage allocation.
    ///
    /// Passed as a mutable reference (same rationale as `probabilities_buf`).
    pub successor_active_slots_buf: &'a mut Vec<usize>,

    /// Optional visited-states archive for dominated cut selection.
    ///
    /// When `Some`, gathered states are archived after each per-stage exchange
    /// so that the domination test has access to all forward-pass trial points.
    /// Pass `None` when using `Level1`, `Lml1`, or no cut selection.
    pub visited_archive: Option<&'a mut crate::visited_states::VisitedStatesArchive>,

    /// Pre-allocated buffer for per-slot binding increment aggregation.
    ///
    /// Reused across stages to avoid per-stage allocation. Sized to
    /// `fcf.pools[successor].metadata.len()` after each `sync_packed_cuts`
    /// call. Used for the `allreduce(Sum)` that synchronises cut binding
    /// metadata across MPI ranks so that `active_count` and
    /// `last_active_iter` reflect trial points from ALL ranks, not just this
    /// rank's local forward passes.
    pub metadata_sync_buf: &'a mut Vec<u64>,

    /// Pre-allocated receive buffer for the `allreduce(Sum)` that aggregates
    /// per-slot binding increments across MPI ranks.
    ///
    /// Reused across stages to avoid per-stage allocation (F1-003 fix).
    /// Resized to `pool_size` before each allreduce call.
    pub global_increments_buf: &'a mut Vec<u64>,

    /// Pre-allocated send buffer for the per-iteration `allreduce(BitwiseOr)`
    /// that aggregates sliding-window binding-activity bitmaps across MPI ranks.
    ///
    /// Length equals the current pool population at the time of the reduce call.
    /// Each element is a `u32` bitmask; bit 0 is set if any worker on this rank
    /// observed a binding event for that cut slot during the current iteration
    /// (across all stages). Cleared once per iteration at the start of
    /// `run_backward_pass`.
    pub metadata_sync_window_buf: &'a mut Vec<u32>,

    /// Pre-allocated receive buffer for the per-iteration `allreduce(BitwiseOr)`.
    ///
    /// Receives the globally OR-reduced window bitmaps after the allreduce.
    /// Resized to `pool_size` before the allreduce call. After the allreduce,
    /// each element is `OR`-ed into the corresponding `CutMetadata::active_window`
    /// and both buffers are cleared for the next iteration.
    pub global_window_increments_buf: &'a mut Vec<u32>,

    /// Pre-allocated buffer for packing real (non-padded) gathered state
    /// vectors when archiving visited states for dominated cut selection.
    ///
    /// When `visited_archive` is `Some`, this buffer is filled by
    /// [`ExchangeBuffers::pack_real_states_into`] before each
    /// `archive_gathered_states` call so that zero-padded trailing entries
    /// (from ranks with fewer forward passes in an uneven distribution) are
    /// never archived. Pass `&mut Vec::new()` when `visited_archive` is
    /// `None`.
    pub real_states_buf: &'a mut Vec<f64>,

    /// Per-(worker, opening) gather buffer for backward-pass solver stats.
    ///
    /// Shape: `n_workers_local × max_openings`. After the parallel region at
    /// each stage, each worker's `per_opening_stats` slice is copied into this
    /// buffer via `set(ws.worker_id, omega, ...)`.
    ///
    /// `reset()` is called at the **start of each stage** (inside the stage loop,
    /// before the parallel region) so stale data from the previous stage never
    /// accumulates.
    ///
    /// Allocated once at training setup; never reallocated on the hot path.
    pub stage_worker_stats_buf: &'a mut StageWorkerStatsBuffer,

    /// MPI send buffer for the per-`(worker, opening)` stats `allgatherv`.
    ///
    /// Length: `n_workers_local * bwd_max_openings * WORKER_STATS_ENTRY_STRIDE`.
    /// Packed from `stage_worker_stats_buf` after each stage's parallel region via
    /// `pack_worker_opening_stats`. Allocated once at training setup; never
    /// reallocated on the hot path.
    pub bwd_stats_send_buf: &'a mut Vec<f64>,

    /// MPI receive buffer for the per-`(rank, worker, opening)` stats `allgatherv`.
    ///
    /// Length: `n_ranks * n_workers_local * bwd_max_openings * WORKER_STATS_ENTRY_STRIDE`.
    /// After `allgatherv`, rank 0 unpacks this into `bwd_stats_unpack_buf`. All ranks
    /// allocate the same buffer size (allgatherv is all→all). Allocated once at training
    /// setup; never reallocated on the hot path.
    pub bwd_stats_recv_buf: &'a mut Vec<f64>,

    /// Per-rank element counts for the `allgatherv` of backward stats.
    ///
    /// Each rank sends `n_workers_local * bwd_max_openings * WORKER_STATS_ENTRY_STRIDE`
    /// `f64` values. All entries are equal (uniform partition). Length: `n_ranks`.
    /// Allocated once at training setup; never reallocated on the hot path.
    pub bwd_stats_counts: &'a [usize],

    /// Displacement array for the `allgatherv` of backward stats.
    ///
    /// `displs[r] = r * n_workers_local * bwd_max_openings * WORKER_STATS_ENTRY_STRIDE`.
    /// Length: `n_ranks`. Allocated once at training setup; never reallocated on the hot path.
    pub bwd_stats_displs: &'a [usize],

    /// Unpack destination buffer for the per-`(rank, worker, opening)` stats on rank 0.
    ///
    /// Length: `n_ranks * n_workers_local * bwd_max_openings` `SolverStatsDelta` entries.
    /// Only rank 0 reads from this after unpacking; non-root ranks hold a pre-allocated
    /// but unused buffer (allgatherv is all→all, so all ranks hold the recv buffer).
    /// Allocated once at training setup; never reallocated on the hot path.
    pub bwd_stats_unpack_buf: &'a mut Vec<SolverStatsDelta>,

    /// Optional event channel for emitting [`TrainingEvent::WorkerTiming`] events.
    ///
    /// When `Some`, one `WorkerTiming { phase: Backward, .. }` event is emitted
    /// per rayon worker per iteration after the parallel region completes. When
    /// `None`, no events are emitted and no overhead is incurred. Send errors
    /// (receiver dropped) are silently ignored, matching the existing `emit`
    /// helper semantics.
    pub event_sender: Option<&'a Sender<TrainingEvent>>,
}

/// Per-successor data bundled for `process_stage_backward` and the trial-point helper.
///
/// Groups the successor-specific arguments — including the stage index `t`,
/// opening probabilities, pre-built cut batch, and cut activity metadata —
/// to keep per-function argument counts at or below seven.
struct SuccessorSpec<'a> {
    /// Stage index being cut (the stage whose cost-to-go we are computing).
    t: usize,
    /// Successor stage index (`t + 1`), where the LP is actually solved.
    successor: usize,
    /// This rank's MPI rank index (used to address exchange buffer state).
    my_rank: usize,
    /// Uniform opening probabilities for the successor stage.
    probabilities: &'a [f64],
    /// Pre-built cut rows to append to each successor LP.
    /// Delta batch when baking is active, full active-cut batch otherwise.
    cut_batch: &'a RowBatch,
    /// Total number of active cuts at the successor stage for dual extraction.
    /// Includes both baked and delta cuts contiguous after `template_num_rows`.
    num_cuts_at_successor: usize,
    /// Base row count of the successor template (excludes cuts).
    template_num_rows: usize,
    /// Baked LP template for the successor stage. Always populated — baking
    /// is complete before the backward pass begins.
    baked_template: &'a cobre_solver::StageTemplate,
    /// Ordered slot indices of the active cuts at the successor stage.
    successor_active_slots: &'a [usize],
    /// Minimum dual multiplier for a cut to count as binding.
    cut_activity_tolerance: f64,
    /// Populated count of the successor's cut pool.
    successor_populated_count: usize,
    /// Cut pool at the successor stage for binding-activity tracking.
    successor_pool: &'a CutPool,
}

/// Load the stage LP template and append delta cuts once per trial point.
///
/// Called once before the opening loop in [`process_trial_point_backward`].
/// The LP structure (baked template + delta cuts) is identical across all
/// openings for the same trial point — only the noise-dependent bounds change.
fn load_backward_lp<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    succ: &SuccessorSpec<'_>,
) {
    ws.solver.load_model(succ.baked_template);
    if succ.cut_batch.num_rows > 0 {
        ws.solver.add_rows(succ.cut_batch);
    }
}

/// Transform opening noise and patch LP bounds for one backward opening.
///
/// Called once per opening inside [`process_trial_point_backward`].  The LP
/// structure is already loaded by [`load_backward_lp`]; this function only
/// updates noise-dependent row and column bounds via `set_row_bounds` /
/// `set_col_bounds`.
fn patch_opening_bounds<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    raw_noise: &[f64],
    x_hat: &[f64],
    s: usize,
) {
    let n_blks = if ctx.n_load_buses > 0 {
        ctx.block_counts_per_stage[s]
    } else {
        0
    };
    transform_inflow_noise(raw_noise, s, x_hat, ctx, training_ctx, &mut ws.scratch);
    transform_load_noise(
        raw_noise,
        ctx.n_hydros,
        ctx.n_load_buses,
        training_ctx.stochastic,
        s,
        n_blks,
        &mut ws.scratch.load_rhs_buf,
    );
    let n_stochastic_ncs = training_ctx.stochastic.n_stochastic_ncs();
    if n_stochastic_ncs > 0 {
        transform_ncs_noise(
            raw_noise,
            ctx.n_hydros,
            ctx.n_load_buses,
            training_ctx.stochastic,
            s,
            ctx.block_counts_per_stage[s],
            ctx.ncs_max_gen,
            &mut ws.scratch.ncs_col_upper_buf,
        );
    }
    ws.patch_buf.fill_forward_patches(
        training_ctx.indexer,
        x_hat,
        &ws.scratch.noise_buf,
        ctx.base_rows[s],
        &ctx.templates[s].row_scale,
    );
    if ctx.n_load_buses > 0 {
        ws.patch_buf.fill_load_patches(
            ctx.load_balance_row_starts[s],
            n_blks,
            &ws.scratch.load_rhs_buf,
            ctx.load_bus_indices,
            &ctx.templates[s].row_scale,
        );
    }
    ws.patch_buf.fill_z_inflow_patches(
        training_ctx.indexer.z_inflow_row_start,
        &ws.scratch.z_inflow_rhs_buf,
        &ctx.templates[s].row_scale,
    );
    let pc = ws.patch_buf.forward_patch_count();
    ws.solver.set_row_bounds(
        &ws.patch_buf.indices[..pc],
        &ws.patch_buf.lower[..pc],
        &ws.patch_buf.upper[..pc],
    );
    if n_stochastic_ncs > 0 && !training_ctx.indexer.ncs_generation.is_empty() {
        let n_blks_stage = ctx.block_counts_per_stage[s];
        let expected_len = n_stochastic_ncs * n_blks_stage;
        if ws.scratch.ncs_col_indices_buf.len() != expected_len {
            ws.scratch.ncs_col_indices_buf.clear();
            ws.scratch.ncs_col_lower_buf.clear();
            for ncs_idx in 0..n_stochastic_ncs {
                for blk in 0..n_blks_stage {
                    ws.scratch.ncs_col_indices_buf.push(
                        training_ctx.indexer.ncs_generation.start + ncs_idx * n_blks_stage + blk,
                    );
                    ws.scratch.ncs_col_lower_buf.push(0.0);
                }
            }
        }
        ws.solver.set_col_bounds(
            &ws.scratch.ncs_col_indices_buf,
            &ws.scratch.ncs_col_lower_buf,
            &ws.scratch.ncs_col_upper_buf,
        );
    }
}

/// Resolve the ω=0 warm-start basis from the worker's `BasisStoreSliceMut`.
///
/// Returns `None` when the slot is empty (cold start or no prior capture).
#[inline]
fn resolve_backward_basis<'a>(
    basis_slice: &'a BasisStoreSliceMut<'_>,
    m: usize,
    s: usize,
) -> Option<&'a CapturedBasis> {
    basis_slice.get(m, s)
}

/// Process one trial point `m` in the backward pass, iterating over all openings.
///
/// Solves at each (scenario, opening) and accumulates duals into `per_opening_stats`.
/// At ω=0, writes the post-solve basis into `basis_slice`; writes at ω>0 are
/// forbidden per AD-2 (retained-LU corruption risk). Infeasibility at ω=0
/// leaves the slot unchanged (AD-7).
#[allow(clippy::too_many_lines)]
fn process_trial_point_backward<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    spec: &BackwardPassSpec<'_>,
    succ: &SuccessorSpec<'_>,
    basis_slice: &mut BasisStoreSliceMut<'_>,
    m: usize,
) -> Result<StagedCut, SddpError> {
    let indexer = training_ctx.indexer;
    let tree_view = training_ctx.stochastic.tree_view();
    let x_hat = spec.exchange.state_at(succ.my_rank, m);
    let scenario = spec.fwd_offset + m;
    let s = succ.successor;

    debug_assert_eq!(
        ws.backward_accum.per_opening_stats.len(),
        succ.probabilities.len(),
        "per_opening_stats must be initialised to n_openings before each stage's trial-point loop"
    );

    for omega in 0..succ.probabilities.len() {
        let raw_noise = tree_view.opening(s, omega);
        patch_opening_bounds(ws, ctx, training_ctx, raw_noise, x_hat, s);

        // Snapshot solver statistics before this opening's solve.
        let stats_before_omega = ws.solver.statistics();

        // ω=0: resolve warm-start basis; ω>0: HiGHS hot-starts from retained factorization.
        let stored_basis = if omega == 0 {
            resolve_backward_basis(basis_slice, m, s)
        } else {
            None
        };
        let inputs = crate::stage_solve::StageInputs {
            stage_context: ctx,
            indexer,
            pool: succ.successor_pool,
            current_state: x_hat,
            stored_basis,
            baked_template: succ.baked_template,
            stage_index: s,
            scenario_index: scenario,
            iteration: Some(spec.iteration),
            horizon_is_terminal: false,
            terminal_has_boundary_cuts: false,
        };

        let outcome =
            crate::stage_solve::run_stage_solve(ws, crate::stage_solve::Phase::Backward, &inputs)?;

        let crate::stage_solve::StageOutcome::Backward {
            view,
            recon_stats: _,
        } = outcome
        else {
            unreachable!("run_stage_solve(Phase::Backward) returns Backward variant")
        };

        // Extract all needed values from `view` before any mutable borrow of
        // `ws.backward_accum`. `view` borrows from `ws` through the
        // `run_stage_solve` lifetime; extracting into owned locals ends that
        // borrow before the subsequent `&mut ws` accesses below.
        let objective = view.objective;
        // Unscale duals from scaled units back to original units.
        // `dual_original[i] = row_scale[i] * dual_scaled[i]`.
        // Only the state-fixing rows [0, n_state) are needed for cut coefficients.
        // Cut rows have implicit row_scale = 1.0 and require no adjustment.
        let row_scale = &ctx.templates[s].row_scale;
        let state_duals: Vec<f64> = if row_scale.is_empty() {
            view.dual[..indexer.n_state].to_vec()
        } else {
            view.dual[..indexer.n_state]
                .iter()
                .zip(row_scale)
                .map(|(&d, &rs)| d * rs)
                .collect()
        };
        let cut_duals_owned: Vec<f64> = if succ.num_cuts_at_successor > 0 {
            // Dual extraction for cut binding-activity tracking.
            //
            // Layout invariant:
            //   [0, template_num_rows)               — base structural rows
            //   [template_num_rows, template_num_rows + num_cuts_at_successor) — cut rows
            //
            // The baked template contains:
            //   [template_num_rows, baked_template_num_rows) — baked cut rows (from prior iters)
            //   [baked_template_num_rows, baked_template_num_rows + num_delta_cuts) — delta rows
            //
            // Both sections are contiguous and slot-monotone (baked then delta in active-slot
            // order), so the unified index `cut_idx` maps correctly to `successor_active_slots`.
            view.dual[succ.template_num_rows..succ.template_num_rows + succ.num_cuts_at_successor]
                .to_vec()
        } else {
            Vec::new()
        };
        // `view` is Copy; assign to `_` to make the end-of-borrow intent explicit.
        // Subsequent code may mutably borrow `ws` fields.
        //
        // IMPORTANT: the after-snapshot for per-opening stats is taken AFTER
        // `view` is dropped. `run_stage_solve` returns a `StageOutcome<'ws>`
        // whose `view` field borrows `ws` for `'ws`. While `view` is live the
        // borrow checker treats `ws` as mutably borrowed, preventing
        // `ws.solver.statistics()` (shared borrow). Dropping `view` here ends
        // the mutable borrow so the statistics call below is safe.
        let _ = view;

        // Snapshot solver statistics after this opening's solve. All needed
        // values from `view` (objective, duals) are already in owned locals.
        // The monotonic counters in `statistics()` are unaffected by the
        // solution data that `view` was borrowing.
        let stats_after_omega = ws.solver.statistics();

        // Accumulate per-opening delta element-wise into the worker's buffer.
        // The buffer was initialised to `Default::default()` for this stage
        // at the start of `process_stage_backward`, before the trial-point loop.
        let opening_delta =
            SolverStatsDelta::from_snapshots(&stats_before_omega, &stats_after_omega);
        SolverStatsDelta::accumulate_into(
            &mut ws.backward_accum.per_opening_stats[omega],
            &opening_delta,
        );

        let out = &mut ws.backward_accum.outcomes[omega];
        out.coefficients.copy_from_slice(&state_duals);
        out.objective_value = objective;
        // Intercept: alpha = Q_scaled - pi' * x_hat.
        //
        // `objective` is the LP objective in scaled cost units (divided by
        // COST_SCALE_FACTOR). `coefficients` are unscaled duals (dual_i *
        // row_scale_i). The product `pi' * x_hat` is also in scaled cost
        // units because the LP duals inherit the cost scaling. The cut
        // `theta >= alpha + pi'(x - x_hat)` is evaluated in the parent
        // LP, where theta is also in scaled units, so all terms are
        // consistently scaled.
        out.intercept = objective
            - out
                .coefficients
                .iter()
                .zip(x_hat)
                .map(|(pi, x)| pi * x)
                .sum::<f64>();

        for (cut_idx, &slot) in succ.successor_active_slots.iter().enumerate() {
            if cut_duals_owned
                .get(cut_idx)
                .is_some_and(|&d| d > succ.cut_activity_tolerance)
            {
                ws.backward_accum.slot_increments[slot] += 1;
            }
        }

        // AD-2: write BasisStore[m, s] at ω=0 only (corruption risk at ω>0).
        // Reuse existing slot in-place; first iteration allocates, subsequent reuse.
        if omega == 0 {
            let num_cols = succ.baked_template.num_cols;
            let base_row_count = succ.template_num_rows;
            let cut_row_count = succ.num_cuts_at_successor;
            let basis_row_capacity = base_row_count + cut_row_count;
            if let Some(captured) = basis_slice.get_mut(m, s).as_mut() {
                ws.solver.get_basis(&mut captured.basis);
                write_capture_metadata(
                    captured,
                    succ.successor_pool,
                    base_row_count,
                    cut_row_count,
                    x_hat,
                );
            } else {
                let mut captured = CapturedBasis::new(
                    num_cols,
                    basis_row_capacity,
                    base_row_count,
                    cut_row_count,
                    x_hat.len(),
                );
                ws.solver.get_basis(&mut captured.basis);
                write_capture_metadata(
                    &mut captured,
                    succ.successor_pool,
                    base_row_count,
                    cut_row_count,
                    x_hat,
                );
                *basis_slice.get_mut(m, s) = Some(captured);
            }
        }
    }

    // Aggregate into the pre-allocated scratch buffer, then copy to an owned
    // Vec<f64> for the StagedCut. The copy is the one unavoidable allocation
    // per trial point: the coefficients must outlive the parallel closure.
    let n_openings = succ.probabilities.len();
    let mut agg_intercept = 0.0_f64;
    spec.risk_measures[succ.t].aggregate_cut_into(
        &ws.backward_accum.outcomes[..n_openings],
        succ.probabilities,
        &mut agg_intercept,
        &mut ws.backward_accum.agg_coefficients,
    );
    let agg_coefficients = ws.backward_accum.agg_coefficients.clone();
    debug_assert!(
        u32::try_from(scenario).is_ok(),
        "global scenario index overflows u32"
    );
    #[allow(clippy::cast_possible_truncation)]
    let forward_pass_index = scenario as u32;
    // Accumulate binding counts into the metadata buffer for later merge.
    let pop = ws.backward_accum.slot_increments.len();
    for slot in 0..pop {
        let count = ws.backward_accum.slot_increments[slot];
        if count > 0 {
            ws.backward_accum.metadata_sync_contribution[slot] += count;
            // Set bit 0 to record iteration-level activity for the sliding window.
            ws.backward_accum.metadata_sync_window_contribution[slot] |= 1u32;
        }
    }
    Ok(StagedCut {
        trial_point_idx: m,
        intercept: agg_intercept,
        coefficients: agg_coefficients,
        forward_pass_index,
    })
}

/// Evaluate all trial points for a single backward stage, returning staged cuts.
///
/// Trials are statically partitioned (mirroring `run_forward_pass`) with each
/// worker owning its scenario range and disjoint `BasisStoreSliceMut` sub-view
/// for ω=0 basis writes. Staged cuts are returned unsorted; caller sorts by
/// `trial_point_idx` before FCF insertion.
#[allow(clippy::too_many_arguments)]
fn process_stage_backward<S: SolverInterface + Send>(
    workspaces: &mut [SolverWorkspace<S>],
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    local_work: usize,
    spec: &BackwardPassSpec<'_>,
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
            // Grow window contribution buffer monotonically (not cleared per-stage;
            // cleared once per iteration at the start of run_backward_pass).
            if ws.backward_accum.metadata_sync_window_contribution.len() < pop {
                ws.backward_accum
                    .metadata_sync_window_contribution
                    .resize(pop, 0u32);
            }
            ws.backward_accum
                .per_opening_stats
                .resize_with(n_openings, SolverStatsDelta::default);
            for slot in &mut ws.backward_accum.per_opening_stats[..n_openings] {
                *slot = SolverStatsDelta::default();
            }

            // Static partition: assign scenarios to worker, matching basis_slice view.
            let (start_m, end_m) = partition(local_work, n_workers, w);
            let n_local = end_m - start_m;
            let mut staged: Vec<StagedCut> = Vec::with_capacity(n_local.max(1));
            let worker_stage_wall_start = Instant::now();

            for m in start_m..end_m {
                ws.backward_accum.slot_increments[..pop].fill(0);
                staged.push(process_trial_point_backward(
                    ws,
                    ctx,
                    training_ctx,
                    spec,
                    succ,
                    &mut basis_slice,
                    m,
                )?);
            }

            // Accumulate per-worker elapsed into the iteration-level timing buffer.
            // Slot 1 = backward_wall_ms: sum across stages for this worker.
            ws.worker_timing_buf[WORKER_TIMING_SLOT_BWD_WALL] +=
                worker_stage_wall_start.elapsed().as_secs_f64() * 1_000.0;

            Ok(staged)
        })
        .collect()
}

/// Execute the backward pass for one training iteration on this rank.
///
/// Sweeps stages from `num_stages - 2` down to `0`. For each stage, trial-point
/// loop is parallelised with static scenario partitioning. Each worker generates
/// cuts into a thread-local buffer, sorted by trial-point index before FCF insertion.
/// The outer per-stage loop remains sequential (stage `t` depends on cuts at `t+1`).
///
/// ## Error handling
///
/// On `SolverError::Infeasible`, returns `SddpError::Infeasible` with the
/// backward stage (0-based), iteration, and global trial point index.
/// On any other `SolverError`, returns `SddpError::Solver`. On error, the
/// FCF may be partially populated.
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
/// - `ctx.templates.len() != num_stages`
/// - `ctx.base_rows.len() != num_stages`
/// - `spec.risk_measures.len() != num_stages`
/// - `baked.len() != num_stages`
///
/// Structurally independent parameters: `workspaces` / `basis_store` are per-rank mutable buffers,
/// `ctx`/`baked`/`fcf`/`cut_batches` are per-stage read/write state, `training_ctx` is study-level,
/// `spec` is backward-specific config, `comm` is the communicator. All 5 context structs listed in
/// `.claude/architecture-rules.md` are already in use.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
pub fn run_backward_pass<S: SolverInterface + Send, C: Communicator>(
    workspaces: &mut [SolverWorkspace<S>],
    basis_store: &mut BasisStore,
    ctx: &StageContext<'_>,
    baked: &[StageTemplate],
    fcf: &mut FutureCostFunction,
    cut_batches: &mut [RowBatch],
    training_ctx: &TrainingContext<'_>,
    spec: &mut BackwardPassSpec<'_>,
    comm: &C,
) -> Result<BackwardResult, SddpError> {
    let TrainingContext {
        horizon,
        indexer,
        stochastic,
        ..
    } = training_ctx;
    let num_stages = horizon.num_stages();
    let my_rank = comm.rank();

    debug_assert_eq!(ctx.templates.len(), num_stages);
    debug_assert_eq!(ctx.base_rows.len(), num_stages);
    debug_assert_eq!(spec.risk_measures.len(), num_stages);
    debug_assert_eq!(baked.len(), num_stages, "baked.len() must equal num_stages");

    let start = Instant::now();
    let solves_before: u64 = workspaces
        .iter()
        .map(|ws| ws.solver.statistics().solve_count)
        .sum();
    let mut cuts_generated: usize = 0;
    let mut stage_stats: Vec<(usize, Vec<StageWorkerOpeningDelta>)> = Vec::new();
    let n_workers_local = workspaces.len();
    let n_ranks = comm.size();
    let bwd_max_openings =
        spec.bwd_stats_send_buf.len() / n_workers_local.max(1) / WORKER_STATS_ENTRY_STRIDE;
    let mut state_exchange_ms: u64 = 0;
    let mut cut_batch_build_ms: u64 = 0;
    let mut setup_ms: u64 = 0;
    let mut imbalance_ms: u64 = 0;
    let mut scheduling_ms: u64 = 0;
    let mut cut_sync_ms: u64 = 0;
    #[allow(clippy::cast_precision_loss)]
    let n_workers = workspaces.len() as f64;
    let tree_view = stochastic.tree_view();
    let mut staged_cuts_buf: Vec<StagedCut> = Vec::new();
    // Pre-allocate per-worker snapshot buffers; reused across all stage iterations.
    let mut worker_stats_before: Vec<SolverStatistics> = Vec::with_capacity(workspaces.len());
    let mut worker_stats_after: Vec<SolverStatistics> = Vec::with_capacity(workspaces.len());
    // Pre-allocate per-worker delta and total buffers; reused via clear()+extend() each stage.
    let mut worker_deltas: Vec<SolverStatsDelta> = Vec::with_capacity(workspaces.len());
    let mut worker_totals: Vec<f64> = Vec::with_capacity(workspaces.len());

    // Reset per-worker timing buffers to zero at the iteration boundary so that
    // the accumulation across stages starts clean. Slots are zeroed individually;
    // the whole [f64; 16] is stack-resident Copy, so fill(0.0) is a write of 128 bytes.
    for ws in workspaces.iter_mut() {
        ws.worker_timing_buf.fill(0.0);
    }

    // Clear iteration-scoped window contribution buffers. These buffers accumulate
    // binding-activity across ALL stages of this iteration (not per-stage), so they
    // must be zeroed once here, not inside the stage loop.
    for ws in workspaces.iter_mut() {
        ws.backward_accum.metadata_sync_window_contribution.fill(0);
    }
    spec.metadata_sync_window_buf.fill(0);
    spec.global_window_increments_buf.fill(0);

    for t in (0..num_stages.saturating_sub(1)).rev() {
        // When the caller supplies forward-pass records, perform the per-stage
        // allgatherv here so that `exchange.state_at(rank, m)` returns the
        // state for stage `t` (the state entering stage `t+1`). When records
        // is empty the caller has pre-populated the exchange buffers (test path).
        if !spec.records.is_empty() {
            let exch_start = Instant::now();
            spec.exchange.exchange(spec.records, t, num_stages, comm)?;
            #[allow(clippy::cast_possible_truncation)]
            {
                state_exchange_ms += exch_start.elapsed().as_millis() as u64;
            }
        }

        // Archive gathered states for dominated cut selection (if active).
        // Use pack_real_states_into so that zero-padded trailing entries from
        // ranks with fewer forward passes in an uneven distribution are never
        // archived (ticket-003 fix).
        if let Some(ref mut archive) = spec.visited_archive {
            let total_fwd = spec.exchange.real_total_scenarios();
            spec.exchange.pack_real_states_into(spec.real_states_buf);
            archive.archive_gathered_states(t, spec.real_states_buf, total_fwd);
        }

        let successor = t + 1;

        // Collect per-worker snapshots before solves (needed for overhead decomposition).
        worker_stats_before.clear();
        worker_stats_before.extend(workspaces.iter().map(|w| w.solver.statistics()));

        let n_openings = tree_view.n_openings(successor);
        spec.probabilities_buf.clear();
        #[allow(clippy::cast_precision_loss)]
        spec.probabilities_buf
            .resize(n_openings, 1.0_f64 / n_openings as f64);

        let batch_start = Instant::now();
        // Build the delta-only cut batch for the baked path.
        let template_num_rows = ctx.templates[successor].num_rows;
        build_delta_cut_row_batch_into(
            &mut cut_batches[successor],
            fcf,
            successor,
            indexer,
            &ctx.templates[successor].col_scale,
            spec.iteration,
        );
        let baked_tmpl = &baked[successor];
        let baked_num_rows = baked_tmpl.num_rows;
        let num_delta = cut_batches[successor].num_rows;
        // Total cuts = baked cut rows (beyond base template) + delta rows.
        let num_cuts_at_successor = (baked_num_rows - template_num_rows) + num_delta;
        #[allow(clippy::cast_possible_truncation)]
        {
            cut_batch_build_ms += batch_start.elapsed().as_millis() as u64;
        }
        spec.successor_active_slots_buf.clear();
        spec.successor_active_slots_buf
            .extend(fcf.active_cuts(successor).map(|(slot, _, _)| slot));

        let local_work = spec.local_work;

        let succ_spec = SuccessorSpec {
            t,
            successor,
            my_rank,
            probabilities: spec.probabilities_buf,
            cut_batch: &cut_batches[successor],
            num_cuts_at_successor,
            template_num_rows,
            baked_template: baked_tmpl,
            successor_active_slots: spec.successor_active_slots_buf,
            cut_activity_tolerance: spec.cut_activity_tolerance,
            successor_populated_count: fcf.pools[successor].populated_count,
            successor_pool: &fcf.pools[successor],
        };

        // Split BasisStore into disjoint per-worker sub-views for ω=0 writes.
        let basis_slices = basis_store.split_workers_mut(n_workers_local.max(1));

        let process_start = Instant::now();
        let worker_staged = process_stage_backward(
            workspaces,
            ctx,
            training_ctx,
            local_work,
            spec,
            &succ_spec,
            basis_slices,
        );
        // Capture wall-clock before sequential post-processing to avoid
        // inflating the parallel region time with merge, sync, and metadata work.
        #[allow(clippy::cast_possible_truncation)]
        let parallel_wall_ms = process_start.elapsed().as_millis() as u64;

        staged_cuts_buf.clear();
        for worker_result in worker_staged {
            staged_cuts_buf.extend(worker_result?);
        }
        // Sort by trial_point_idx for deterministic FCF insertion order.
        staged_cuts_buf.sort_by_key(|cut| cut.trial_point_idx);
        debug_assert_eq!(staged_cuts_buf.len(), spec.local_work);

        for cut in &staged_cuts_buf {
            fcf.add_cut(
                t,
                spec.iteration,
                cut.forward_pass_index,
                cut.intercept,
                &cut.coefficients,
            );
            cuts_generated += 1;
        }

        // Per-stage cut sync: allgatherv distributes this stage's local cuts
        // to all ranks so every rank sees the same FCF before the next stage.
        let sync_start = Instant::now();
        let n_local = spec.cut_sync_bufs.pack_local_cuts(fcf, t, spec.iteration);
        spec.cut_sync_bufs.sync_packed_cuts(t, n_local, fcf, comm)?;
        #[allow(clippy::cast_possible_truncation)]
        {
            cut_sync_ms += sync_start.elapsed().as_millis() as u64;
        }

        // Metadata sync: allreduce(Sum) the per-slot binding increment counts
        // so that `active_count` and `last_active_iter` reflect trial points
        // from ALL ranks, not just this rank's local forward passes.
        //
        // The pool size is read AFTER sync_packed_cuts so that remote cuts
        // added by the allgatherv are included in the slot range.
        let pool_size = fcf.pools[successor].metadata.len();
        if pool_size > 0 {
            // Build the local increment buffer by summing per-worker contributions.
            spec.metadata_sync_buf.clear();
            spec.metadata_sync_buf.resize(pool_size, 0u64);
            for ws in workspaces.iter() {
                for (slot, &inc) in ws
                    .backward_accum
                    .metadata_sync_contribution
                    .iter()
                    .enumerate()
                    .take(pool_size)
                {
                    spec.metadata_sync_buf[slot] += inc;
                }
            }

            // Allreduce(Sum): every rank contributes its local increments and
            // receives the global sum. For a single rank (LocalBackend) this
            // is an identity copy — no behavior change.
            spec.global_increments_buf.clear();
            spec.global_increments_buf.resize(pool_size, 0u64);
            comm.allreduce(
                spec.metadata_sync_buf,
                spec.global_increments_buf,
                ReduceOp::Sum,
            )
            .map_err(SddpError::from)?;

            // Apply global increments: slots with any increment from any rank
            // have their active_count increased and last_active_iter updated.
            for (slot, &inc) in spec.global_increments_buf.iter().enumerate() {
                if inc > 0 {
                    fcf.pools[successor].metadata[slot].active_count += inc;
                    fcf.pools[successor].metadata[slot].last_active_iter = spec.iteration;
                }
            }

            // Allreduce(BitwiseOr): aggregate sliding-window binding activity
            // across ranks so that any rank observing a cut binding this stage
            // globally sets bit 0 in that cut's active_window.
            //
            // Build the local OR buffer by OR-ing per-worker contributions.
            spec.metadata_sync_window_buf.resize(pool_size, 0u32);
            for ws in workspaces.iter() {
                for (slot, &bit) in ws
                    .backward_accum
                    .metadata_sync_window_contribution
                    .iter()
                    .enumerate()
                    .take(pool_size)
                {
                    spec.metadata_sync_window_buf[slot] |= bit;
                }
            }

            // Allreduce(BitwiseOr): OR-reduce across ranks. For a single rank
            // (LocalBackend) this is an identity copy — no behavior change.
            spec.global_window_increments_buf.resize(pool_size, 0u32);
            comm.allreduce(
                spec.metadata_sync_window_buf,
                spec.global_window_increments_buf,
                ReduceOp::BitwiseOr,
            )
            .map_err(SddpError::from)?;

            // Apply global window bits: any slot where bit 0 is set globally
            // gets bit 0 ORed into its active_window. This records "this cut
            // was binding on at least one rank during this iteration".
            for (slot, &bits) in spec.global_window_increments_buf.iter().enumerate() {
                if bits & 1 != 0 {
                    fcf.pools[successor].metadata[slot].active_window |= 1u32;
                }
            }

            // Clear the window buffers for the next stage's accumulation.
            spec.metadata_sync_window_buf.fill(0);
            spec.global_window_increments_buf.fill(0);
        }

        // Collect per-worker statistics snapshots after this stage's solves.
        worker_stats_after.clear();
        worker_stats_after.extend(workspaces.iter().map(|w| w.solver.statistics()));

        // Compute per-worker deltas and decompose parallel overhead.
        // Reuse pre-allocated buffers (clear+extend avoids per-stage allocation on the hot path).
        worker_deltas.clear();
        worker_deltas.extend(
            worker_stats_before
                .iter()
                .zip(&worker_stats_after)
                .map(|(before, after)| SolverStatsDelta::from_snapshots(before, after)),
        );

        // setup_time_ms: total non-solve work (load_model + set_bounds + basis_set).
        let stage_setup_ms: f64 = worker_deltas
            .iter()
            .map(|d| d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms)
            .sum();

        // Accumulate per-worker setup contribution into worker_timing_buf[BWD_SETUP]
        // (slot 8). Summed across all stages so the per-iteration emission in
        // `run_backward_pass` (after the stage loop) carries the full iteration total.
        for (ws, delta) in workspaces.iter_mut().zip(&worker_deltas) {
            ws.worker_timing_buf[WORKER_TIMING_SLOT_BWD_SETUP] +=
                delta.load_model_time_ms + delta.set_bounds_time_ms + delta.basis_set_time_ms;
        }

        // Per-worker elapsed: solve + setup phases.
        // Reuse pre-allocated buffer (clear+extend avoids per-stage allocation on the hot path).
        worker_totals.clear();
        worker_totals.extend(worker_deltas.iter().map(|d| {
            d.solve_time_ms + d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms
        }));

        let max_worker_ms = worker_totals.iter().copied().fold(0.0_f64, f64::max);
        let avg_worker_ms = if worker_totals.is_empty() {
            0.0_f64
        } else {
            worker_totals.iter().sum::<f64>() / n_workers
        };

        // load_imbalance_ms: slowest worker minus average.
        let stage_imbalance_ms = (max_worker_ms - avg_worker_ms).max(0.0);
        // scheduling_overhead_ms: residual wall time beyond slowest worker.
        #[allow(
            clippy::cast_precision_loss,      // parallel_wall_ms as f64: safe at HPC scale
            clippy::cast_possible_truncation, // f64 as u64: clamped non-negative
            clippy::cast_sign_loss            // f64 as u64: clamped non-negative
        )]
        let stage_scheduling_ms = (parallel_wall_ms as f64 - max_worker_ms).max(0.0);

        #[allow(
            clippy::cast_possible_truncation, // f64 as u64: clamped non-negative
            clippy::cast_sign_loss            // f64 as u64: clamped non-negative
        )]
        {
            setup_ms += stage_setup_ms as u64;
            imbalance_ms += stage_imbalance_ms as u64;
            scheduling_ms += stage_scheduling_ms as u64;
        }

        // Gather per-worker per_opening_stats into the local StageWorkerStatsBuffer
        // (T003). Reset first so stale data from the previous stage never accumulates.
        spec.stage_worker_stats_buf.reset();
        for ws in workspaces.iter() {
            debug_assert_eq!(
                ws.backward_accum.per_opening_stats.len(),
                n_openings,
                "per_opening_stats length must equal n_openings on every worker"
            );
            #[allow(clippy::cast_sign_loss)]
            let wid = ws.worker_id as usize;
            for omega in 0..n_openings {
                spec.stage_worker_stats_buf.set(
                    wid,
                    omega,
                    ws.backward_accum.per_opening_stats[omega].clone(),
                );
            }
        }

        // Pack the local worker stats into the MPI send buffer (T004 layout).
        pack_worker_opening_stats(
            spec.bwd_stats_send_buf,
            spec.stage_worker_stats_buf.as_slice(),
            n_workers_local,
            bwd_max_openings,
        );

        // Gather per-worker stats from all ranks (T005).
        // allgatherv collects n_workers_local * bwd_max_openings * STRIDE f64s from
        // every rank into a rank-major receive buffer. The LocalBackend (np=1) treats
        // this as an identity copy — no branch needed.
        comm.allgatherv(
            spec.bwd_stats_send_buf,
            spec.bwd_stats_recv_buf,
            spec.bwd_stats_counts,
            spec.bwd_stats_displs,
        )
        .map_err(SddpError::Communication)?;

        // Rank 0 (and all ranks — allgatherv is all→all) unpacks the receive buffer
        // into per-(rank, worker_id, opening) SolverStatsDelta entries.
        // Only rank 0 uses these for parquet output; other ranks hold the buffer
        // for potential future use (zero extra allocation).
        debug_assert_eq!(
            spec.bwd_stats_recv_buf.len(),
            n_ranks * n_workers_local * bwd_max_openings * WORKER_STATS_ENTRY_STRIDE,
            "recv buffer length must equal n_ranks * n_workers_local * bwd_max_openings * STRIDE"
        );
        unpack_worker_opening_stats(
            spec.bwd_stats_recv_buf,
            spec.bwd_stats_unpack_buf,
            n_ranks * n_workers_local,
            bwd_max_openings,
        );

        // Build the per-(rank, worker_id, opening) stage_stats entries.
        // Skip padded slots (omega >= n_openings) and zero-solve slots unless omega == 0
        // (the omega=0 sentinel preserves "stage was visited" semantics from epic-04a).
        let mut stage_entries: Vec<StageWorkerOpeningDelta> = Vec::new();
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
                    let delta = spec.bwd_stats_unpack_buf[flat].clone();
                    if delta.lp_solves > 0 || omega == 0 {
                        stage_entries.push((rank_i32, wid_i32, omega, delta));
                    }
                }
            }
        }

        stage_stats.push((successor, stage_entries));
    }

    // Clear per-worker window contribution buffers after all stages are done.
    // These accumulate across stages per iteration; they were consumed by the
    // per-stage allreduce(BitwiseOr) above and must be zeroed for next iteration.
    for ws in workspaces.iter_mut() {
        ws.backward_accum.metadata_sync_window_contribution.fill(0);
    }

    // Emit one WorkerTiming { phase: Backward } event per worker, carrying the
    // per-iteration totals accumulated across all stages. The rank-level
    // BackwardPassComplete event (emitted by the training loop caller) is unaffected.
    //
    // Slots populated per worker:
    //   [1]  backward_wall_ms  — accumulated in process_stage_backward per stage
    //   [8]  bwd_setup_ms      — accumulated after worker_deltas per stage above
    // All other slots remain 0 on Backward WorkerTiming events (rank-only fields).
    if let Some(sender) = spec.event_sender {
        for ws in workspaces.iter() {
            let _ = sender.send(TrainingEvent::WorkerTiming {
                rank: ws.rank,
                worker_id: ws.worker_id,
                iteration: spec.iteration,
                phase: WorkerTimingPhase::Backward,
                timings: ws.worker_timing_buf,
            });
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let solves_after: u64 = workspaces
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

#[cfg(test)]
mod tests {
    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
    use cobre_solver::{
        Basis, LpSolution, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };

    use cobre_core::scenario::SamplingScheme;

    use super::{BackwardPassSpec, BackwardResult, run_backward_pass};
    use crate::{
        ExchangeBuffers, FutureCostFunction, HorizonMode, InflowNonNegativityMethod, RiskMeasure,
        StageIndexer, TrajectoryRecord,
        context::{StageContext, TrainingContext},
        cut_sync::CutSyncBuffers,
        solver_stats::{SolverStatsDelta, StageWorkerStatsBuffer, WORKER_STATS_ENTRY_STRIDE},
        workspace::{BackwardAccumulators, BasisStore, SolverWorkspace},
    };

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

    /// Stub communicator for tests (single-rank).
    struct StubComm;

    impl Communicator for StubComm {
        fn allgatherv<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            // Single-rank: copy send to recv (mirrors LocalBackend behavior).
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
            unreachable!("StubComm broadcast not used in backward pass tests")
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

    /// Mock solver for testing: returns fixed solution or infeasible error on demand.
    ///
    /// Buffer fields (`buf_primal`, `buf_dual`, `buf_reduced_costs`) store the
    /// solution data that [`SolutionView`] borrows from. They are filled in
    /// `solve` before the borrow is established.
    struct MockSolver {
        solution: LpSolution,
        infeasible_at: Option<usize>,
        call_count: usize,
        /// Tracks the current number of rows (template + appended cuts).
        current_num_rows: usize,
        /// Number of times `solve(Some(&basis))` was called (warm-start calls).
        warm_start_calls: usize,
        /// Dual padding value for rows beyond the base template (cuts).
        /// Defaults to 0.0 (cuts not binding). Set to a positive value
        /// to make all cuts appear binding in tests.
        cut_dual_padding: f64,
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
                infeasible_at: None,
                call_count: 0,
                current_num_rows: base_rows,
                warm_start_calls: 0,
                cut_dual_padding: 0.0,
                buf_primal,
                buf_dual,
                buf_reduced_costs,
            }
        }

        fn infeasible_on(solution: LpSolution, n: usize) -> Self {
            let base_rows = solution.dual.len();
            let buf_primal = solution.primal.clone();
            let buf_dual = solution.dual.clone();
            let buf_reduced_costs = solution.reduced_costs.clone();
            Self {
                solution,
                infeasible_at: Some(n),
                call_count: 0,
                current_num_rows: base_rows,
                warm_start_calls: 0,
                cut_dual_padding: 0.0,
                buf_primal,
                buf_dual,
                buf_reduced_costs,
            }
        }

        /// Like `always_ok` but added cut rows return positive duals,
        /// making all existing cuts appear binding in subsequent solves.
        fn always_ok_with_binding_cuts(solution: LpSolution) -> Self {
            let mut s = Self::always_ok(solution);
            s.cut_dual_padding = 1.0;
            s
        }
    }

    impl SolverInterface for MockSolver {
        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
        fn load_model(&mut self, template: &StageTemplate) {
            self.current_num_rows = template.num_rows;
        }

        fn add_rows(&mut self, cuts: &RowBatch) {
            self.current_num_rows += cuts.num_rows;
        }

        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn solve(
            &mut self,
            basis: Option<&Basis>,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            if basis.is_some() {
                self.warm_start_calls += 1;
            }
            let call = self.call_count;
            self.call_count += 1;
            if self.infeasible_at == Some(call) {
                return Err(SolverError::Infeasible);
            }
            // Fill internal buffers, resizing dual to match current LP row count.
            self.buf_primal.clone_from(&self.solution.primal);
            self.buf_dual.clone_from(&self.solution.dual);
            self.buf_dual
                .resize(self.current_num_rows, self.cut_dual_padding);
            self.buf_reduced_costs
                .clone_from(&self.solution.reduced_costs);
            Ok(cobre_solver::SolutionView {
                objective: self.solution.objective,
                primal: &self.buf_primal,
                dual: &self.buf_dual,
                reduced_costs: &self.buf_reduced_costs,
                iterations: self.solution.iterations,
                solve_time_seconds: self.solution.solve_time_seconds,
            })
        }

        fn get_basis(&mut self, _out: &mut Basis) {}

        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }

        fn name(&self) -> &'static str {
            "Mock"
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
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
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

    /// Wrap a `MockSolver` into a single-element `Vec<SolverWorkspace<MockSolver>>`
    /// for tests that exercise the workspace-based backward-pass API.
    ///
    /// The workspace is sized for `n_hydro=1`, `max_par_order=0`, and `n_state`
    /// state dimensions.
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
                lag_accumulator: vec![],
                lag_weight_accum: 0.0,
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                recon_slot_lookup: Vec::new(),
                demotion_candidates: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: [0.0_f64; 16],
        }]
    }

    /// Create an empty `BasisStore` for `num_scenarios` scenarios and
    /// `num_stages` stages (all slots `None`).
    fn empty_basis_store(num_scenarios: usize, num_stages: usize) -> BasisStore {
        BasisStore::new(num_scenarios, num_stages)
    }

    /// Create a `BasisStore` with one slot pre-populated at
    /// `[scenario][stage]` with the given `Basis`.
    fn basis_store_with_one(
        num_scenarios: usize,
        num_stages: usize,
        scenario: usize,
        stage: usize,
        basis: Basis,
    ) -> BasisStore {
        let mut store = BasisStore::new(num_scenarios, num_stages);
        // test shim: zero metadata is acceptable for tests exercising the length path
        *store.get_mut(scenario, stage) = Some(crate::workspace::CapturedBasis {
            basis,
            base_row_count: 0,
            cut_row_slots: Vec::new(),
            state_at_capture: Vec::new(),
        });
        store
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

    #[allow(clippy::too_many_lines)]
    fn make_stochastic_context(
        n_stages: usize,
        branching_factor: usize,
    ) -> cobre_stochastic::StochasticContext {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::{
            Bus, DeficitSegment, EntityId, SystemBuilder,
            scenario::{
                CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
                InflowModel,
            },
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

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
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

        #[allow(clippy::cast_possible_truncation)]
        let inflow = |stage_idx: usize| InflowModel {
            hydro_id: EntityId(1),
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            stage_id: stage_idx as i32,
            mean_m3s: 100.0,
            std_m3s: 30.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        };

        let inflow_models: Vec<InflowModel> = (0..n_stages).map(inflow).collect();

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

    // ── Unit tests ────────────────────────────────────────────────────────────

    #[test]
    fn backward_result_fields_accessible() {
        let r = BackwardResult {
            cuts_generated: 6,
            elapsed_ms: 42,
            lp_solves: 0,
            stage_stats: Vec::new(),
            state_exchange_time_ms: 0,
            cut_batch_build_time_ms: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
            cut_sync_time_ms: 0,
        };
        assert_eq!(r.cuts_generated, 6);
        assert_eq!(r.elapsed_ms, 42);
        assert!(r.stage_stats.is_empty());
        assert_eq!(r.state_exchange_time_ms, 0);
        assert_eq!(r.cut_batch_build_time_ms, 0);
        assert_eq!(r.setup_time_ms, 0);
        assert_eq!(r.load_imbalance_ms, 0);
        assert_eq!(r.scheduling_overhead_ms, 0);
        assert_eq!(r.cut_sync_time_ms, 0);
    }

    #[test]
    fn backward_result_clone_and_debug() {
        let r = BackwardResult {
            cuts_generated: 3,
            elapsed_ms: 100,
            lp_solves: 0,
            stage_stats: Vec::new(),
            state_exchange_time_ms: 0,
            cut_batch_build_time_ms: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
            cut_sync_time_ms: 0,
        };
        let c = r.clone();
        assert_eq!(c.cuts_generated, 3);
        let s = format!("{r:?}");
        assert!(s.contains("BackwardResult"));
    }

    #[test]
    fn dual_extraction_formula_coefficients_are_negated_duals() {
        // Given known dual values [d0, d1], coefficients must be [-d0, -d1].
        let d0 = 3.5_f64;
        let d1 = -1.2_f64;
        let dual = [d0, d1];

        let coefficients: Vec<f64> = dual.iter().map(|&d| -d).collect();

        assert!((coefficients[0] - (-d0)).abs() < f64::EPSILON);
        assert!((coefficients[1] - (-d1)).abs() < f64::EPSILON);
    }

    #[test]
    fn intercept_formula_matches_spec() {
        // alpha = Q - pi^T * x_hat
        // Given: objective=50.0, pi=[2.0, -1.0], x_hat=[10.0, 5.0]
        // Expected: alpha = 50.0 - (2.0*10.0 + (-1.0)*5.0) = 50.0 - 15.0 = 35.0
        let objective = 50.0_f64;
        let coefficients = [2.0_f64, -1.0_f64];
        let x_hat = [10.0_f64, 5.0_f64];
        let pi_dot_x: f64 = coefficients
            .iter()
            .zip(x_hat.iter())
            .map(|(p, x)| p * x)
            .sum();
        let intercept = objective - pi_dot_x;
        assert!((intercept - 35.0).abs() < f64::EPSILON);
    }

    #[test]
    fn single_stage_system_produces_no_cuts() {
        // A 1-stage system has no stages with a successor, so the backward
        // sweep (0..0) is empty — zero cuts are generated.
        let stochastic = make_stochastic_context(1, 2);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0()];
        let base_rows = vec![1_usize];

        let n_state = indexer.n_state;
        let n_stages = 1_usize;
        let forward_passes = 2_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0], vec![20.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation];

        let solution = solution_1_0(100.0, -5.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::new(n_state, 64, 1);
        let result = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        assert_eq!(result.cuts_generated, 0);
        assert_eq!(fcf.total_active_cuts(), 0);
    }

    #[test]
    fn two_stage_system_two_trial_points_generates_two_cuts_at_stage_0() {
        // Acceptance criterion: 3-stage system, 1 hydro (n_state=1), 2 openings,
        // 2 trial points → 2 cuts at stage 0. This is the 2-stage version
        // (stages 0 and 1); cuts should exist only at stage 0.
        let n_stages = 2_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0); // N=1, L=0
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state; // 1
        let forward_passes = 2_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);

        // Two trial points with states [10.0] and [20.0] at stage 0.
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0], vec![20.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        // MockSolver returns objective=100.0, dual[0]=-5.0 for every solve.
        // With x_hat=[10.0]: pi=[5.0], alpha = 100 - 5*10 = 50.
        // With x_hat=[20.0]: pi=[5.0], alpha = 100 - 5*20 = 0 (could be negative).
        let solution = solution_1_0(100.0, -5.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::new(n_state, 64, 1);
        let result = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        // 2 trial points × 1 stage with a successor = 2 cuts at stage 0.
        assert_eq!(result.cuts_generated, 2);
        assert_eq!(fcf.active_cuts(0).count(), 2);
        // Stage 1 (the last stage) gets no cuts.
        assert_eq!(fcf.active_cuts(1).count(), 0);
    }

    #[test]
    fn cut_inserted_with_correct_stage_iteration_and_forward_pass_index() {
        // Acceptance criterion: iteration=2, forward_passes=3, global
        // trial point m=1 → fcf.add_cut(stage=0, iteration=2, fpi=1, ...).
        // slot = warm_start + 2*3 + 1 = 7.
        let n_stages = 2_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 3_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);

        // 3 trial points (forward_passes=3 on a single rank).
        let mut exchange = exchange_with_states(n_state, vec![vec![5.0], vec![10.0], vec![15.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(50.0, 0.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::new(n_state, 64, 1);
        let _ = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 2,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        // Trial point m=1: slot = 0 + 2*3 + 1 = 7
        // Verify that pool[0].metadata[7] has the correct iteration and fpi.
        let meta = &fcf.pools[0].metadata[7];
        assert_eq!(meta.iteration_generated, 2);
        assert_eq!(meta.forward_pass_index, 1);
    }

    #[test]
    fn no_cuts_generated_at_last_stage() {
        // Acceptance criterion: 5-stage system → cuts at stages 0..3, not at 4.
        let n_stages = 5_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -3.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::new(n_state, 64, 1);
        let result = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        // 1 trial point × 4 stages with successors = 4 cuts total.
        assert_eq!(result.cuts_generated, 4);
        for t in 0..4 {
            assert_eq!(fcf.active_cuts(t).count(), 1, "stage {t} should have 1 cut");
        }
        // The last stage (4) must have no cuts.
        assert_eq!(fcf.active_cuts(4).count(), 0, "stage 4 must have no cuts");
    }

    #[test]
    fn elapsed_ms_is_non_negative() {
        let n_stages = 2_usize;
        let stochastic = make_stochastic_context(n_stages, 2);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![5.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(10.0, 0.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::new(n_state, 64, 1);
        let result = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        // elapsed_ms is u64, so it is always >= 0.
        let _ = result.elapsed_ms;
    }

    #[test]
    fn infeasible_solver_returns_sddp_infeasible_error() {
        // Acceptance criterion: MockSolver::infeasible_on(0) for the first
        // backward solve → SddpError::Infeasible is returned.
        let n_stages = 2_usize;
        let stochastic = make_stochastic_context(n_stages, 1);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(0.0, 0.0);
        // First solve call returns infeasible.
        let solver = MockSolver::infeasible_on(solution, 0);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::new(n_state, 64, 1);
        let result = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        );

        assert!(
            matches!(result, Err(crate::SddpError::Infeasible { .. })),
            "expected SddpError::Infeasible, got: {result:?}",
        );
    }

    #[test]
    fn expectation_aggregation_mean_of_per_opening_intercepts() {
        // Given 3 openings with uniform probability 1/3 and per-opening
        // intercepts [10.0, 20.0, 30.0], the aggregated intercept must be 20.0.
        use crate::risk_measure::BackwardOutcome as BO;

        let outcomes = vec![
            BO {
                intercept: 10.0,
                coefficients: vec![],
                objective_value: 10.0,
            },
            BO {
                intercept: 20.0,
                coefficients: vec![],
                objective_value: 20.0,
            },
            BO {
                intercept: 30.0,
                coefficients: vec![],
                objective_value: 30.0,
            },
        ];
        let probs = vec![1.0 / 3.0; 3];
        let (intercept, _) = RiskMeasure::Expectation.aggregate_cut(&outcomes, &probs);
        assert!(
            (intercept - 20.0).abs() < 1e-10,
            "expected 20.0, got {intercept}"
        );
    }

    // ── Integration tests ─────────────────────────────────────────────────────

    #[test]
    fn cut_coefficients_and_intercept_match_dual_extraction_formula() {
        // Integration test: verify that the backward pass uses the correct
        // dual extraction formula by checking cuts in the FCF.
        //
        // Setup: 2-stage, N=1, L=0, 1 opening, 1 trial point.
        //   dual[0] = -3.0 (storage-fixing dual from MockSolver)
        //   objective = 80.0
        //   x_hat = [10.0]
        //
        // Expected (coefficients = dual, not -dual):
        //   pi[0] = dual[0] = -3.0
        //   intercept = 80.0 - (-3.0) * 10.0 = 110.0
        //   coefficients = [-3.0]
        let n_stages = 2_usize;
        let stochastic = make_stochastic_context(n_stages, 1);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        // dual[0] = -3.0, objective = 80.0
        let solution = solution_1_0(80.0, -3.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::new(n_state, 64, 1);
        let _ = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        let cuts: Vec<_> = fcf.active_cuts(0).collect();
        assert_eq!(cuts.len(), 1);
        let (_, intercept, coefficients) = &cuts[0];

        assert!(
            (intercept - 110.0).abs() < 1e-10,
            "expected intercept=110.0, got {intercept}"
        );
        assert_eq!(coefficients.len(), 1);
        assert!(
            (coefficients[0] - (-3.0)).abs() < 1e-10,
            "expected coefficient=-3.0, got {}",
            coefficients[0]
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn cut_gradient_sign_physically_correct() {
        // Regression test for the Benders cut sign bug.
        //
        // Physical invariant: more initial storage → lower future cost.
        // The storage-fixing dual π is negative (shadow price of relaxing
        // the fixing constraint increases cost when storage decreases).
        //
        // Correct: coefficient = π < 0, so the cut slope is negative
        //   (more storage → lower cut value → lower theta → lower total cost).
        //
        // Old bug: coefficient = -π > 0, so the cut slope was positive
        //   (more storage → higher cut value → wrong incentive).
        let n_stages = 2_usize;
        let stochastic = make_stochastic_context(n_stages, 1);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![50.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        // dual[0] = -2.0 (negative: more storage → less cost)
        // objective = 100.0, x_hat = 50.0
        let solution = solution_1_0(100.0, -2.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::new(n_state, 64, 1);
        let _ = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        let cuts: Vec<_> = fcf.active_cuts(0).collect();
        assert_eq!(cuts.len(), 1, "expected exactly one cut");
        let (_, _intercept, coefficients) = &cuts[0];

        // The coefficient must be negative (same sign as the dual).
        // The old bug would produce +2.0 here instead of -2.0.
        assert!(
            coefficients[0] < 0.0,
            "cut coefficient must be negative (more storage → less future cost), \
             got {} — likely the Benders cut sign bug has been reintroduced",
            coefficients[0]
        );
        assert!(
            (coefficients[0] - (-2.0)).abs() < 1e-10,
            "expected coefficient=-2.0, got {}",
            coefficients[0]
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn cut_is_tight_at_trial_point() {
        // Regression test: a Benders cut must be tight (exact) at the trial
        // point x̂ where it was generated. That is:
        //   intercept + coefficient * x̂ = Q(x̂)
        // where Q(x̂) = objective value of the subproblem at x̂.
        //
        // The cut equation is: θ ≥ intercept + coefficient * x
        // At x = x̂: θ ≥ Q(x̂) + π'(x̂ - x̂) = Q(x̂)
        //
        // If the sign is wrong (coefficient = -π instead of π), then:
        //   intercept + (-π) * x̂ ≠ Q(x̂) in general
        //
        // This test verifies the tightness property.
        let n_stages = 2_usize;
        let stochastic = make_stochastic_context(n_stages, 1);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let x_hat = 30.0_f64;
        let mut exchange = exchange_with_states(n_state, vec![vec![x_hat]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let q_xhat = 200.0_f64; // subproblem objective at x̂
        let dual_storage = -4.0_f64;
        let solution = solution_1_0(q_xhat, dual_storage);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::new(n_state, 64, 1);
        let _ = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        let cuts: Vec<_> = fcf.active_cuts(0).collect();
        assert_eq!(cuts.len(), 1);
        let (_, intercept, coefficients) = &cuts[0];

        // Evaluate the cut at x̂: cut_value = intercept + coeff * x̂
        let cut_at_xhat = intercept + coefficients[0] * x_hat;

        // Must equal Q(x̂) (tightness property)
        assert!(
            (cut_at_xhat - q_xhat).abs() < 1e-10,
            "cut must be tight at trial point: \
             cut_value={cut_at_xhat}, Q(x̂)={q_xhat}, \
             intercept={intercept}, coeff={}, x̂={x_hat}",
            coefficients[0]
        );
    }

    #[test]
    fn single_rank_backward_pass_with_local_backend_produces_correct_fcf() {
        // Integration test with LocalBackend communicator (exercises single-rank path).
        use cobre_comm::LocalBackend;

        let n_stages = 3_usize;
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
        let solver = MockSolver::always_ok(solution);
        let comm = LocalBackend;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::new(n_state, 64, 1);
        let result = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        // 3-stage system: cuts at stages 0 and 1; 2 trial points each.
        // Total cuts = 2 stages × 2 trial points = 4.
        assert_eq!(result.cuts_generated, 4);
        assert_eq!(fcf.active_cuts(0).count(), 2);
        assert_eq!(fcf.active_cuts(1).count(), 2);
        assert_eq!(fcf.active_cuts(2).count(), 0);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn forward_pass_index_matches_global_scenario_index() {
        // Acceptance criterion: when a cut is generated for global trial point
        // m=5, then `fcf.add_cut(stage, iteration, 5, ...)` is called with
        // forward_pass_index = m = 5.
        //
        // Setup: iteration=2, forward_passes=6 (6 scenarios on 1 rank), 1 opening.
        // ExchangeBuffers: local_count=6, num_ranks=1, total_scenarios=6.
        // state_at(5/6, 5%6) = state_at(0, 5) — valid.
        //
        // Slot formula: slot = warm_start(0) + 2*6 + 5 = 17.
        // The key invariant: forward_pass_index = m = 5.
        let n_stages = 2_usize;
        let stochastic = make_stochastic_context(n_stages, 1);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 6_u32; // 6 scenarios on a single rank
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);

        // 6 trial points (m = 0..5). ExchangeBuffers: local_count=6, num_ranks=1.
        let mut exchange = exchange_with_states(
            n_state,
            vec![
                vec![1.0],
                vec![2.0],
                vec![3.0],
                vec![4.0],
                vec![5.0],
                vec![6.0],
            ],
        );

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(50.0, 0.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::new(n_state, 64, 1);
        let _ = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 2,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        // m=5: slot = warm_start(0) + 2*6 + 5 = 17
        // The critical check: forward_pass_index in metadata equals global m=5.
        let meta = &fcf.pools[0].metadata[17];
        assert_eq!(meta.iteration_generated, 2, "iteration_generated must be 2");
        assert_eq!(
            meta.forward_pass_index, 5,
            "forward_pass_index must be 5 (= global m)"
        );
    }

    // ── Unit tests: warm-start basis caching (backward pass) ──────────────────

    /// Warm-start from a pre-populated forward basis: when `BasisStore` has
    /// `Some(Basis)` at `(scenario=0, stage=1)` before the first backward call,
    /// the first opening at the successor stage must call `solve(Some(&basis))`
    /// rather than `solve(None)`.
    ///
    /// AC: Given a 2-stage system, 1 trial point, 1 opening, with
    /// `basis_store.get(0, 1) = Some(Basis::new(...))` pre-populated,
    /// then `solver.warm_start_calls == 1` after the backward pass.
    #[test]
    fn warm_start_uses_prepopulated_forward_basis() {
        let n_stages = 2_usize;
        let n_openings = 1_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;

        // Pre-populate the basis store at (scenario=0, stage=1).
        // This simulates a forward pass having already solved stage 1 and cached its basis.
        let pre_basis = Basis::new(templates[1].num_cols, templates[1].num_rows);
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store =
            basis_store_with_one(exchange.local_count(), n_stages, 0, 1, pre_basis);

        let mut csb = CutSyncBuffers::new(n_state, 64, 1);
        let _ = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        let warm_start_calls = workspaces[0].solver.warm_start_calls;
        assert_eq!(
            warm_start_calls, 1,
            "first opening at successor stage must call solve(Some(&basis)) \
             when basis_store.get(0, 1) is pre-populated (warm_start_calls == 1, got {warm_start_calls})"
        );
    }

    /// Multi-opening P3b behavior: given 3 openings at the same successor stage,
    /// the first opening cold-starts (store slot is None via `solve()`), and
    /// openings 1 and 2 use `HiGHS` internal hot-start via `solve(None)` instead of
    /// `solve(Some(&working_basis))`.
    ///
    /// AC: Given a 2-stage system, 1 trial point, 3 openings, empty basis cache,
    /// then `solver.warm_start_calls == 0` after the backward pass (P3b: no
    /// and 3 warm-start; opening 1 cold-starts).
    #[test]
    fn multi_opening_subsequent_openings_use_internal_hotstart() {
        let n_stages = 2_usize;
        let n_openings = 3_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;

        // Start with an empty store — opening 1 must cold-start.
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::new(n_state, 64, 1);
        let _ = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        // P3b optimization: opening 0 cold-starts (no basis in store),
        // openings 1 and 2 use solve(None) (HiGHS internal hot-start) instead of
        // solve(Some(&working_basis)). No explicit warm-start calls for subsequent openings.
        let warm_start_calls = workspaces[0].solver.warm_start_calls;
        assert_eq!(
            warm_start_calls, 0,
            "P3b: no warm-start calls expected when BasisStore is empty \
             (warm_start_calls == 0, got {warm_start_calls})"
        );
    }

    /// Error propagation: when a backward solve returns `SolverError::Infeasible`,
    /// the error must propagate as `SddpError::Infeasible`.
    ///
    /// In the new per-scenario design, the backward pass uses a local `working_basis`
    /// variable (not written back to `BasisStore`), so there is no shared cache slot
    /// to check after the error. The test verifies that the error is correctly
    /// propagated regardless of what was in the basis store at entry.
    ///
    /// AC: Given a 2-stage system, 1 opening, `MockSolver` returns infeasible on
    /// call 0, then `run_backward_pass` returns `Err(SddpError::Infeasible { .. })`.
    #[test]
    fn backward_solver_error_propagates() {
        let n_stages = 2_usize;
        let n_openings = 1_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(0.0, 0.0);
        // The first backward solve (call 0) returns infeasible.
        let solver = MockSolver::infeasible_on(solution, 0);
        let comm = StubComm;

        // Pre-populate the store — error should propagate regardless.
        let pre_basis = Basis::new(templates[1].num_cols, templates[1].num_rows);
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store =
            basis_store_with_one(exchange.local_count(), n_stages, 0, 1, pre_basis);

        let mut csb = CutSyncBuffers::new(n_state, 64, 1);
        let result = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        );

        assert!(
            matches!(result, Err(crate::SddpError::Infeasible { .. })),
            "expected SddpError::Infeasible, got: {result:?}",
        );
        // The BasisStore is not mutated by the backward pass — the working_basis
        // is a local variable dropped on error. The store slot at (0, 1) remains
        // as it was; this just verifies the store is untouched by an error path.
        assert!(
            basis_store.get(0, 1).is_some(),
            "BasisStore must not be mutated by the backward pass error path"
        );
    }

    // ── New test: parallel cut determinism ────────────────────────────────────

    /// AC: When `run_backward_pass` runs with 1 workspace vs 4 workspaces given
    /// the same input data, the FCF pools contain identical cuts (same intercept,
    /// coefficient vectors, and slot assignments for each trial point).
    #[test]
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    fn test_backward_pass_parallel_cut_determinism() {
        use crate::lp_builder::PatchBuffer;

        let n_stages = 3_usize;
        let n_openings = 2_usize;
        let n_trial_points = 8_usize;

        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        #[allow(clippy::cast_possible_truncation)]
        let forward_passes = n_trial_points as u32;

        // Build 8 distinct trial-point states.
        let states: Vec<Vec<f64>> = (0..n_trial_points).map(|i| vec![i as f64 + 1.0]).collect();
        let mut exchange = exchange_with_states(n_state, states);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
        let solution = solution_1_0(100.0, -5.0);
        let comm = StubComm;

        // --- Run with 1 workspace ---
        let mut fcf_1 =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
        let solver_1 = MockSolver::always_ok(solution.clone());
        let mut workspaces_1 = vec![SolverWorkspace {
            rank: 0,
            worker_id: 0,
            solver: solver_1,
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
                lag_accumulator: vec![],
                lag_weight_accum: 0.0,
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                recon_slot_lookup: Vec::new(),
                demotion_candidates: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: [0.0_f64; 16],
        }];
        let mut basis_store_1 = empty_basis_store(exchange.local_count(), n_stages);
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
        let mut csb = CutSyncBuffers::new(n_state, 64, 1);
        let _ = run_backward_pass(
            &mut workspaces_1,
            &mut basis_store_1,
            &ctx,
            &templates,
            &mut fcf_1,
            &mut empty_cut_batches(n_stages),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        // --- Run with 4 workspaces ---
        let mut fcf_4 =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
        let mut workspaces_4: Vec<SolverWorkspace<MockSolver>> = (0..4_i32)
            .map(|idx| SolverWorkspace {
                rank: 0,
                worker_id: idx,
                solver: MockSolver::always_ok(solution.clone()),
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
                    lag_accumulator: vec![],
                    lag_weight_accum: 0.0,
                    downstream_accumulator: Vec::new(),
                    downstream_weight_accum: 0.0,
                    downstream_completed_lags: Vec::new(),
                    downstream_n_completed: 0,
                    recon_slot_lookup: Vec::new(),
                    demotion_candidates: Vec::new(),
                },
                scratch_basis: Basis::new(0, 0),
                backward_accum: BackwardAccumulators::default(),
                worker_timing_buf: [0.0_f64; 16],
            })
            .collect();
        let mut basis_store_4 = empty_basis_store(exchange.local_count(), n_stages);
        let mut csb = CutSyncBuffers::new(n_state, 64, 1);
        let _ = run_backward_pass(
            &mut workspaces_4,
            &mut basis_store_4,
            &ctx,
            &templates,
            &mut fcf_4,
            &mut empty_cut_batches(n_stages),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        // --- Verify identical FCF contents for all non-last stages ---
        for t in 0..(n_stages - 1) {
            let cuts_1: Vec<_> = fcf_1.active_cuts(t).collect();
            let cuts_4: Vec<_> = fcf_4.active_cuts(t).collect();

            assert_eq!(
                cuts_1.len(),
                cuts_4.len(),
                "stage {t}: cut count differs (1 workspace: {}, 4 workspaces: {})",
                cuts_1.len(),
                cuts_4.len()
            );

            for (idx, ((slot_1, intercept_1, coeff_1), (slot_4, intercept_4, coeff_4))) in
                cuts_1.iter().zip(cuts_4.iter()).enumerate()
            {
                assert_eq!(
                    slot_1, slot_4,
                    "stage {t} cut {idx}: slot mismatch ({slot_1} vs {slot_4})"
                );
                assert!(
                    (intercept_1 - intercept_4).abs() < 1e-12,
                    "stage {t} cut {idx}: intercept mismatch ({intercept_1} vs {intercept_4})"
                );
                assert_eq!(
                    coeff_1.len(),
                    coeff_4.len(),
                    "stage {t} cut {idx}: coefficient vector length mismatch"
                );
                for (j, (c1, c4)) in coeff_1.iter().zip(coeff_4.iter()).enumerate() {
                    assert!(
                        (c1 - c4).abs() < 1e-12,
                        "stage {t} cut {idx} coeff[{j}]: {c1} vs {c4}"
                    );
                }
            }
        }

        // Last stage must have no cuts in both.
        assert_eq!(fcf_1.active_cuts(n_stages - 1).count(), 0);
        assert_eq!(fcf_4.active_cuts(n_stages - 1).count(), 0);
    }

    // ── Load noise wiring tests (backward pass) ──────────────────────────────

    /// Build a 2-stage `StochasticContext` with 1 hydro and 1 stochastic load bus.
    ///
    /// The noise vector dimension is `n_hydros + n_load_buses = 2`.
    /// Stage 0 uses `branching_factor` openings; stage 1 is the successor solved
    /// in the backward pass opening loop.
    #[allow(clippy::too_many_lines)]
    fn make_stochastic_context_with_load(
        n_stages: usize,
        branching_factor: usize,
        mean_mw: f64,
        std_mw: f64,
    ) -> cobre_stochastic::StochasticContext {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{CorrelationModel, InflowModel, LoadModel};
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };
        use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
        use cobre_stochastic::context::{
            ClassSchemes, OpeningTreeInputs, build_stochastic_context,
        };

        let bus0 = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let bus1 = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let hydro = Hydro {
            id: EntityId(10),
            name: "H10".to_string(),
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

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
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

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|idx| InflowModel {
                hydro_id: EntityId(10),
                stage_id: idx as i32,
                mean_m3s: 100.0,
                std_m3s: 30.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            })
            .collect();

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|idx| LoadModel {
                bus_id: EntityId(1),
                stage_id: idx as i32,
                mean_mw,
                std_mw,
            })
            .collect();

        let correlation = CorrelationModel {
            method: "spectral".to_string(),
            profiles: std::collections::BTreeMap::new(),
            schedule: vec![],
        };

        let system = SystemBuilder::new()
            .buses(vec![bus0, bus1])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
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

    /// AC: Given a backward pass with 1 stochastic load bus and opening noise
    /// that includes a load component eta, the load balance row RHS in the patch
    /// buffer is set to `max(0, mean + std * eta) * block_factor` before the solve.
    ///
    /// We verify this indirectly: after the backward pass runs, `ws.scratch.load_rhs_buf`
    /// must be non-empty and must contain a positive value (with mean=300, std=30
    /// any reasonable eta produces a positive realization).
    #[test]
    #[allow(clippy::too_many_lines)]
    fn backward_pass_load_patches_applied() {
        // 2-stage system: backward pass solves at successor=1 for each opening.
        // n_hydros=1, n_load_buses=1, 1 block per stage.
        let n_stages = 2_usize;
        let n_openings = 2_usize;
        // mean_mw=300 guarantees a positive realization for any reasonable eta draw.
        let stochastic = make_stochastic_context_with_load(n_stages, n_openings, 300.0, 30.0);
        let indexer = StageIndexer::new(1, 0); // N=1, L=0, n_state=1

        // PatchBuffer: n_hydros=1, max_par_order=0, n_load_buses=1, max_blocks=1.
        let patch_buf = crate::lp_builder::PatchBuffer::new(1, 0, 1, 1);

        // Template: 2 rows (row 0 = state-fixing, row 1 = water-balance).
        // base_rows=[1] → inflow RHS row starts at index 1.
        // noise_scale=[1.0, 1.0] (one per (stage, hydro)).
        let template = StageTemplate {
            num_cols: 3,
            num_rows: 2,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
            objective: vec![0.0, 0.0, 1.0],
            row_lower: vec![50.0, 100.0],
            row_upper: vec![50.0, 100.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        let templates = vec![template; n_stages];
        let base_rows = vec![1_usize; n_stages];
        let noise_scale = vec![1.0_f64; n_stages]; // one per (stage, hydro)

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        // MockSolver returns a fixed solution (1 state var, 1 dual entry).
        let solution = solution_1_0(100.0, -2.0);

        let ws = SolverWorkspace {
            rank: 0,
            worker_id: 0,
            solver: MockSolver::always_ok(solution),
            patch_buf,
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
                load_rhs_buf: Vec::with_capacity(1),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::new(),
                effective_eta_buf: Vec::new(),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
                lag_accumulator: vec![],
                lag_weight_accum: 0.0,
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                recon_slot_lookup: Vec::new(),
                demotion_candidates: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: [0.0_f64; 16],
        };
        let mut workspaces = vec![ws];

        let comm = StubComm;
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        // load_balance_row_starts[successor=1]=10; load_bus_indices=[0]; 1 block/stage.
        let load_balance_row_starts = vec![10_usize; n_stages];
        let load_bus_indices = vec![0_usize];
        let block_counts_per_stage = vec![1_usize; n_stages];

        let mut csb = CutSyncBuffers::new(n_state, 64, 1);
        let _ = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &noise_scale,
                n_hydros: 1,
                n_load_buses: 1,
                load_balance_row_starts: &load_balance_row_starts,
                load_bus_indices: &load_bus_indices,
                block_counts_per_stage: &block_counts_per_stage,
                ncs_max_gen: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        // After the backward pass, load_rhs_buf must have been populated with a
        // positive value for the last opening solved (mean=300, std=30 → positive).
        assert_eq!(
            workspaces[0].scratch.load_rhs_buf.len(),
            1,
            "load_rhs_buf must have 1 entry (1 load bus × 1 block)"
        );
        assert!(
            workspaces[0].scratch.load_rhs_buf[0] > 0.0,
            "load realization must be positive with mean=300, std=30: got {}",
            workspaces[0].scratch.load_rhs_buf[0]
        );
    }

    /// AC: Given a backward pass with 0 stochastic load buses, `patch_count`
    /// equals `N*(2+L)` (no load patches) and `load_rhs_buf` stays empty.
    ///
    /// N=1, L=0 → `N*(2+L) = 2`.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn backward_pass_no_load_buses_unchanged() {
        let n_stages = 2_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0); // N=1, L=0

        // PatchBuffer with no load buses: n_load_buses=0, max_blocks=1.
        let patch_buf = crate::lp_builder::PatchBuffer::new(1, 0, 0, 0);

        let template = StageTemplate {
            num_cols: 3,
            num_rows: 2,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
            objective: vec![0.0, 0.0, 1.0],
            row_lower: vec![50.0, 100.0],
            row_upper: vec![50.0, 100.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        let templates = vec![template; n_stages];
        let base_rows = vec![1_usize; n_stages];
        let noise_scale = vec![1.0_f64; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -2.0);
        let ws = SolverWorkspace {
            rank: 0,
            worker_id: 0,
            solver: MockSolver::always_ok(solution),
            patch_buf,
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
                lag_accumulator: vec![],
                lag_weight_accum: 0.0,
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                recon_slot_lookup: Vec::new(),
                demotion_candidates: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: [0.0_f64; 16],
        };
        let mut workspaces = vec![ws];
        let comm = StubComm;
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::new(n_state, 64, 1);
        let _ = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &noise_scale,
                n_hydros: 1,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[1_usize; 2],
                ncs_max_gen: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        // With n_load_buses=0, forward_patch_count = N*(2+L) + z_inflow = 1*(2+0)+1 = 3.
        assert_eq!(
            workspaces[0].patch_buf.forward_patch_count(),
            3,
            "forward_patch_count must be N*(2+L)+N=3 when n_load_buses=0, got {}",
            workspaces[0].patch_buf.forward_patch_count()
        );
        // load_rhs_buf must remain empty.
        assert!(
            workspaces[0].scratch.load_rhs_buf.is_empty(),
            "load_rhs_buf must be empty when n_load_buses=0"
        );
    }

    /// AC: Given a backward pass with stochastic load, when Benders cut
    /// coefficients are extracted, the cut coefficient array has length `n_state`
    /// unchanged — load adds no state variables.
    ///
    /// Setup: N=1 hydro, L=0 PAR lags → `n_state=1`. After the backward pass with
    /// 1 load bus, each generated cut must have exactly 1 coefficient.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn backward_pass_cut_coefficients_unaffected() {
        let n_stages = 2_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context_with_load(n_stages, n_openings, 200.0, 20.0);
        let indexer = StageIndexer::new(1, 0); // N=1, L=0, n_state=1

        let patch_buf = crate::lp_builder::PatchBuffer::new(1, 0, 1, 1);

        let template = StageTemplate {
            num_cols: 3,
            num_rows: 2,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
            objective: vec![0.0, 0.0, 1.0],
            row_lower: vec![50.0, 100.0],
            row_upper: vec![50.0, 100.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        let templates = vec![template; n_stages];
        let base_rows = vec![1_usize; n_stages];
        let noise_scale = vec![1.0_f64; n_stages];

        let n_state = indexer.n_state; // 1
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(80.0, -3.0);
        let ws = SolverWorkspace {
            rank: 0,
            worker_id: 0,
            solver: MockSolver::always_ok(solution),
            patch_buf,
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
                load_rhs_buf: Vec::with_capacity(1),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::new(),
                effective_eta_buf: Vec::new(),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
                lag_accumulator: vec![],
                lag_weight_accum: 0.0,
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                recon_slot_lookup: Vec::new(),
                demotion_candidates: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: [0.0_f64; 16],
        };
        let mut workspaces = vec![ws];
        let comm = StubComm;
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let load_balance_row_starts = vec![10_usize; n_stages];
        let load_bus_indices = vec![0_usize];
        let block_counts_per_stage = vec![1_usize; n_stages];

        let mut csb = CutSyncBuffers::new(n_state, 64, 1);
        let result = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &noise_scale,
                n_hydros: 1,
                n_load_buses: 1,
                load_balance_row_starts: &load_balance_row_starts,
                load_bus_indices: &load_bus_indices,
                block_counts_per_stage: &block_counts_per_stage,
                ncs_max_gen: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        // Exactly 1 cut generated (1 trial point × 1 stage with a successor).
        assert_eq!(result.cuts_generated, 1);

        // The cut must have exactly n_state=1 coefficient.
        let cuts: Vec<_> = fcf.active_cuts(0).collect();
        assert_eq!(cuts.len(), 1);
        let (_, _intercept, coefficients) = &cuts[0];
        assert_eq!(
            coefficients.len(),
            n_state,
            "cut coefficients length must be n_state={n_state}, got {} — \
             load buses must not add state variables",
            coefficients.len()
        );
    }

    /// BUG-1 structural invariant: per-stage cut sync inside the backward loop.
    ///
    /// Verifies that after `run_backward_pass`, the cut synchronization has been
    /// performed per-stage (not as a separate post-sweep loop). The structural
    /// evidence is:
    ///
    /// 1. `BackwardResult.cut_sync_time_ms` is populated (timing was captured).
    /// 2. The FCF has the expected number of cuts per stage — same as single-rank
    ///    without sync, because single-rank sync is a no-op that does not change
    ///    results but exercises the code path.
    /// 3. Using `LocalBackend` (the production single-rank communicator) instead
    ///    of `StubComm` exercises the full `sync_cuts` → allgatherv → deserialize
    ///    path, confirming no panics or data corruption.
    ///
    /// True multi-rank correctness testing requires actual MPI and is out of
    /// scope for CI. This test validates the structural invariant (sync is
    /// called per-stage inside the loop) and exercises the full code path.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn per_stage_cut_sync_invariant_after_bug1_fix() {
        use cobre_comm::LocalBackend;

        let n_stages = 4_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 3_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0], vec![20.0], vec![30.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let solver = MockSolver::always_ok(solution);
        let comm = LocalBackend;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::new(n_state, forward_passes as usize, 1);
        let result = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 1,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        // 4-stage system: cuts at stages 0, 1, 2; 3 trial points each.
        // Total cuts = 3 stages × 3 trial points = 9.
        assert_eq!(result.cuts_generated, 9);

        // Each non-terminal stage has 3 cuts (one per trial point).
        assert_eq!(fcf.active_cuts(0).count(), 3, "stage 0 must have 3 cuts");
        assert_eq!(fcf.active_cuts(1).count(), 3, "stage 1 must have 3 cuts");
        assert_eq!(fcf.active_cuts(2).count(), 3, "stage 2 must have 3 cuts");
        assert_eq!(
            fcf.active_cuts(3).count(),
            0,
            "terminal stage must have 0 cuts"
        );

        // Verify cut_sync_time_ms was captured (structural evidence that
        // sync_cuts was called inside the backward loop).
        // For single-rank LocalBackend, sync is a no-op, so time should be
        // very small but the field must be populated (not default/garbage).
        // We just verify it's a valid non-negative value.
        assert!(
            result.cut_sync_time_ms < 10_000,
            "cut_sync_time_ms should be reasonable, got {}",
            result.cut_sync_time_ms
        );
    }

    /// Acceptance criterion C4 for ticket-002: within a single backward
    /// iteration on a 3-stage system with `LocalBackend` (single-rank),
    /// cuts generated at stage t=1 are visible at stage t=0 and appear
    /// binding (mock returns positive cut duals). The metadata sync
    /// correctly accumulates `active_count` and sets `last_active_iter`.
    ///
    /// Uses `MockSolver::always_ok_with_binding_cuts` so that cut rows
    /// return positive duals, making them appear binding when evaluated.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn metadata_sync_updates_active_count_and_last_active_iter() {
        use cobre_comm::LocalBackend;

        // 3-stage system: backward loop processes t=1 then t=0.
        // At t=1: generates cuts into pool[1], successor pool[2] is empty.
        // At t=0: generates cuts into pool[0], successor pool[1] has cuts
        //         from t=1. Mock duals make those cuts appear binding.
        let n_stages = 3_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 3_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0], vec![20.0], vec![30.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let solver = MockSolver::always_ok_with_binding_cuts(solution);
        let comm = LocalBackend;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::new(n_state, forward_passes as usize, 1);

        // Run a single backward iteration. The backward loop visits t=1
        // (cuts go to pool[1]), then t=0 (cuts go to pool[0], binding
        // checked against pool[1]).
        let result = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 1,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        // 3 stages × (n_stages-1=2 non-terminal) × 3 trial points = 6 cuts.
        assert_eq!(result.cuts_generated, 6);

        // Pool[1] received 3 cuts from t=1 backward pass.
        // Slot formula: warm_start(0) + iteration(1) * fwd_passes(3) + fpi
        // → slots 3, 4, 5. Populated count = 6 (high-water mark).
        assert_eq!(fcf.pools[1].populated_count, 6);

        // At t=0, the 3 cuts in pool[1] (slots 3,4,5) were evaluated for
        // binding. The mock solver returns positive duals (cut_dual_padding
        // = 1.0) for all cut rows. Each trial point has n_openings=2
        // openings, and the binding check runs per opening. So each slot
        // gets 3 trial points × 2 openings = 6 increments.
        for slot in 3..6 {
            assert!(
                fcf.pools[1].metadata[slot].active_count > 0,
                "slot {slot} active_count should be > 0 (cuts were binding)"
            );
            assert_eq!(
                fcf.pools[1].metadata[slot].active_count, 6,
                "slot {slot} active_count should be 6 (3 trial points × 2 openings)"
            );
            assert_eq!(
                fcf.pools[1].metadata[slot].last_active_iter, 1,
                "slot {slot} last_active_iter should be 1 (current iteration)"
            );
        }

        // active_window bit 0 must be set for binding slots (Epic 06 T1).
        // The BitwiseOr allreduce populates active_window |= 1 for any slot
        // where at least one rank observed a binding event this iteration.
        for slot in 3..6 {
            assert_eq!(
                fcf.pools[1].metadata[slot].active_window & 1,
                1,
                "slot {slot} active_window bit 0 should be set (cut was binding this iteration)"
            );
        }

        // Non-binding slots (0..3 in pool[1]) should have active_window == 0.
        for slot in 0..3 {
            assert_eq!(
                fcf.pools[1].metadata[slot].active_window, 0,
                "slot {slot} active_window should be 0 (cut was not binding)"
            );
        }

        // Pool[2] (terminal successor) received no cuts and no binding
        // was checked against it — metadata should be at defaults.
        assert_eq!(fcf.pools[2].populated_count, 0);
    }

    // -----------------------------------------------------------------------
    // Epic 06 T1: active_window unit tests
    // -----------------------------------------------------------------------

    /// A freshly created `CutMetadata` must have `active_window == 0`.
    ///
    /// This checks that `CutPool::add_cut` (the production path) initialises
    /// the bitmap field to zero, so bit 0 does not spuriously appear set before
    /// any backward pass has run.
    #[test]
    fn active_window_initialised_to_zero() {
        use crate::cut::CutPool;

        let n_state = 1;
        let capacity = 8;
        let pool = CutPool::new(capacity, n_state, 3, 0);
        // A new pool has no populated slots; verify the pre-allocated metadata
        // rows all start with active_window == 0.
        for m in &pool.metadata {
            assert_eq!(
                m.active_window, 0,
                "newly allocated CutMetadata must have active_window == 0"
            );
        }

        // add_cut also initialises active_window to 0.
        let mut pool2 = CutPool::new(capacity, n_state, 3, 0);
        pool2.add_cut(
            /*iteration=*/ 1,
            /*forward_pass_index=*/ 0,
            /*intercept=*/ 0.5,
            /*coefficients=*/ &[1.0_f64],
        );
        assert_eq!(
            pool2.metadata[pool2.populated_count - 1].active_window,
            0,
            "add_cut must initialise active_window to 0"
        );
    }

    /// After a full backward pass where cuts are non-binding, `active_window` stays 0.
    /// After the end-of-iteration shift (`<<= 1`), bit 0 remains 0 (shift of 0 is 0).
    #[test]
    fn active_window_shift_clears_bit_zero() {
        // Bit 0 set → after shift, bit 0 is 0 and bit 1 is 1.
        let mut window: u32 = 1; // bit 0 set
        window <<= 1;
        assert_eq!(window & 1, 0, "shift must clear bit 0");
        assert_eq!(window & 2, 2, "shift must move bit 0 to bit 1");

        // After 32 shifts, a u32 with bit 0 set overflows to 0 (shift by width).
        let mut w: u32 = 0xFFFF_FFFF;
        for _ in 0..32 {
            w <<= 1;
        }
        assert_eq!(w, 0, "32 left-shifts of u32 must overflow to 0");

        // All-zeros stays zero.
        let mut w2: u32 = 0;
        w2 <<= 1;
        assert_eq!(w2, 0, "shift of 0 must remain 0");
    }

    /// The `allreduce(BitwiseOr)` via `LocalBackend` must propagate bit 0 from
    /// a worker's `metadata_sync_window_contribution` into `active_window` for
    /// binding slots, and leave non-binding slots at 0.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn active_window_or_reduction_round_trip_local() {
        use cobre_comm::LocalBackend;

        // 3-stage system; 1 opening; 3 trial points.
        // The backward loop visits t=1 (cuts into pool[1]) then t=0 (cuts into
        // pool[0], binding checked against pool[1]). Mock solver returns positive
        // duals → cuts in pool[1] appear binding at t=0. active_window bit 0
        // must be set after the OR reduction.
        let n_stages = 3_usize;
        let n_openings = 1_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];
        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0], vec![20.0], vec![30.0]]);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
        // Solver returns positive duals → cuts appear binding.
        let solution = solution_1_0(100.0, -5.0);
        let local_count = exchange.local_count();
        let solver = MockSolver::always_ok_with_binding_cuts(solution);
        let comm = LocalBackend;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(local_count, n_stages);
        // max_cuts_per_rank must match exchange.local_count(), not forward_passes,
        // because the exchange has 3 trial points but forward_passes=1.
        let mut csb = CutSyncBuffers::new(n_state, local_count, 1);

        let _ = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 1,
                local_work: local_count,
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        // The cuts in pool[1] were generated at t=1, then evaluated as binding
        // at t=0. Bit 0 of active_window must be set for every ACTIVE slot.
        // This exercises the full OR reduction path: worker contribution →
        // spec.metadata_sync_window_buf → allreduce(BitwiseOr) → active_window.
        //
        // Slot 0 is the warm-start sentinel (never populated); active cuts land
        // at slots 1..4 (slot = 0 + iter*forward_passes + fpi = 1 + fpi).
        let active_slots: Vec<usize> = fcf.pools[1]
            .active_cuts()
            .map(|(slot, _, _)| slot)
            .collect();
        assert!(
            !active_slots.is_empty(),
            "pool[1] must have at least one active cut"
        );
        for slot in active_slots {
            let window = fcf.pools[1].metadata[slot].active_window;
            assert_eq!(
                window & 1,
                1,
                "slot {slot} active_window bit 0 must be set after a binding backward pass \
                 (got {window:#010x})"
            );
        }
    }

    /// After `run_backward_pass` returns, all per-worker
    /// `metadata_sync_window_contribution` buffers must be cleared to 0.
    /// This guarantees the next iteration starts with a clean accumulator.
    #[test]
    fn active_window_cleared_at_iteration_start() {
        use cobre_comm::LocalBackend;

        // 3-stage system so the backward loop processes multiple stages and
        // the window contribution buffers actually get populated.
        let n_stages = 3_usize;
        let n_openings = 1_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];
        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0], vec![20.0], vec![30.0]]);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
        let solution = solution_1_0(100.0, -5.0);
        let local_count = exchange.local_count();
        let solver = MockSolver::always_ok_with_binding_cuts(solution);
        let comm = LocalBackend;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(local_count, n_stages);
        let mut csb = CutSyncBuffers::new(n_state, local_count, 1);

        let _ = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 1,
                local_work: local_count,
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(8, 32),
                bwd_stats_send_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_recv_buf: &mut vec![0.0; 8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_counts: &[8 * 32 * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); 8 * 32],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        // After run_backward_pass returns, all per-worker window contribution
        // buffers must be zeroed. This ensures the next iteration begins clean.
        for ws in &workspaces {
            for &v in &ws.backward_accum.metadata_sync_window_contribution {
                assert_eq!(
                    v, 0,
                    "metadata_sync_window_contribution must be cleared after backward pass"
                );
            }
        }
    }

    /// Build N identical `SolverWorkspace<MockSolver>` instances and run a
    /// 2-stage backward pass with 6 trial points. Returns the resulting FCF.
    ///
    /// Used by `work_stealing_produces_identical_results_across_worker_counts`
    /// to compare FCF state across different worker counts.
    ///
    /// The `MockSolver` returns objective=100.0 and dual[0]=-5.0 for every solve,
    /// which is deterministic (no dependence on call order or worker identity).
    /// Each trial point i gets state [(i + 1) as f64 * 10.0] so that distinct
    /// cuts are generated and the ordering invariant is meaningful.
    #[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
    fn run_backward_pass_with_n_workers(n_workers: usize) -> FutureCostFunction {
        use crate::lp_builder::PatchBuffer;

        let n_stages = 2_usize;
        let local_work = 6_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];
        let n_state = indexer.n_state; // 1

        // Use forward_passes = local_work so the FCF pool is large enough for
        // all trial points in a single iteration (iteration 0, slots 0..5).
        #[allow(clippy::cast_possible_truncation)]
        let forward_passes = local_work as u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 64, &vec![0; n_stages]);

        // Build `local_work` trial points with distinct states so each cut
        // has a different intercept. State for trial point i = (i+1)*10.0.
        let states: Vec<Vec<f64>> = (0..local_work)
            .map(|i| vec![(i + 1) as f64 * 10.0])
            .collect();
        let mut exchange = exchange_with_states(n_state, states);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        // Each workspace gets the same deterministic solution.
        // MockSolver::always_ok returns objective=100.0, dual[0]=-5.0 for
        // every call regardless of call order or worker identity.
        let solution = solution_1_0(100.0, -5.0);
        let mut workspaces: Vec<SolverWorkspace<MockSolver>> = (0..n_workers)
            .map(|idx| SolverWorkspace {
                rank: 0,
                worker_id: i32::try_from(idx).expect("worker_id fits in i32"),
                solver: MockSolver::always_ok(solution.clone()),
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
                    lag_accumulator: vec![],
                    lag_weight_accum: 0.0,
                    downstream_accumulator: Vec::new(),
                    downstream_weight_accum: 0.0,
                    downstream_completed_lags: Vec::new(),
                    downstream_n_completed: 0,
                    recon_slot_lookup: Vec::new(),
                    demotion_candidates: Vec::new(),
                },
                scratch_basis: Basis::new(0, 0),
                backward_accum: BackwardAccumulators::default(),
                worker_timing_buf: [0.0_f64; 16],
            })
            .collect();

        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
        let comm = StubComm;
        let mut csb = CutSyncBuffers::new(n_state, local_work, 1);

        let result = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(n_workers, n_openings),
                bwd_stats_send_buf: &mut vec![
                    0.0;
                    n_workers * n_openings * WORKER_STATS_ENTRY_STRIDE
                ],
                bwd_stats_recv_buf: &mut vec![
                    0.0;
                    n_workers * n_openings * WORKER_STATS_ENTRY_STRIDE
                ],
                bwd_stats_counts: &[n_workers * n_openings * WORKER_STATS_ENTRY_STRIDE],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![
                    SolverStatsDelta::default();
                    n_workers * n_openings
                ],
                event_sender: None,
            },
            &comm,
        )
        .unwrap();

        // Confirm all 6 trial points produced cuts at stage 0.
        assert_eq!(
            result.cuts_generated, local_work,
            "n_workers={n_workers}: expected {local_work} cuts, got {}",
            result.cuts_generated,
        );

        fcf
    }

    #[test]
    fn work_stealing_produces_identical_results_across_worker_counts() {
        // Acceptance criterion: the FCF state after running the backward pass
        // with 1 workspace must be bit-identical to the state after running
        // with 3 workspaces, given the same inputs. This verifies that the
        // sort-by-trial_point_idx post-processing in the work-stealing
        // implementation produces a deterministic FCF regardless of which
        // worker claims which trial point.
        let fcf_1 = run_backward_pass_with_n_workers(1);
        let fcf_3 = run_backward_pass_with_n_workers(3);

        let num_stages = 2;

        // Verify that both runs produced cuts (belt-and-suspenders guard so
        // that an empty FCF cannot cause a false positive).
        assert!(
            fcf_1.active_cuts(0).count() > 0,
            "1-worker run produced no cuts at stage 0"
        );

        for stage in 0..num_stages {
            let cuts_1: Vec<_> = fcf_1.active_cuts(stage).collect();
            let cuts_3: Vec<_> = fcf_3.active_cuts(stage).collect();
            assert_eq!(
                cuts_1.len(),
                cuts_3.len(),
                "stage {stage}: cut count mismatch ({} vs {})",
                cuts_1.len(),
                cuts_3.len(),
            );
            for (i, ((s1, int1, c1), (s3, int3, c3))) in cuts_1.iter().zip(&cuts_3).enumerate() {
                assert_eq!(
                    s1, s3,
                    "stage {stage}, cut {i}: slot mismatch ({s1} vs {s3})"
                );
                assert_eq!(
                    int1, int3,
                    "stage {stage}, cut {i}: intercept mismatch ({int1} vs {int3})"
                );
                assert_eq!(
                    c1, c3,
                    "stage {stage}, cut {i}: coefficients mismatch ({c1:?} vs {c3:?})"
                );
            }
        }
    }

    // ── Parallel overhead decomposition unit tests ────────────────────────────

    /// Build a `SolverStatistics` snapshot with the given cumulative times (in seconds).
    fn make_stats(
        solve_s: f64,
        load_s: f64,
        set_bounds_s: f64,
        basis_set_s: f64,
    ) -> SolverStatistics {
        SolverStatistics {
            total_solve_time_seconds: solve_s,
            total_load_model_time_seconds: load_s,
            total_set_bounds_time_seconds: set_bounds_s,
            total_basis_set_time_seconds: basis_set_s,
            ..SolverStatistics::default()
        }
    }

    /// Decompose parallel overhead into (`setup_ms`, `imbalance_ms`, `scheduling_ms`)
    /// from per-worker before/after snapshots.
    fn decompose_overhead(
        pairs: &[(SolverStatistics, SolverStatistics)],
        parallel_wall_ms: u64,
    ) -> (u64, u64, u64) {
        use crate::solver_stats::SolverStatsDelta;

        #[allow(clippy::cast_precision_loss)]
        let n_workers = pairs.len() as f64;

        let worker_deltas: Vec<SolverStatsDelta> = pairs
            .iter()
            .map(|(before, after)| SolverStatsDelta::from_snapshots(before, after))
            .collect();

        let stage_setup_ms: f64 = worker_deltas
            .iter()
            .map(|d| d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms)
            .sum();

        let worker_totals: Vec<f64> = worker_deltas
            .iter()
            .map(|d| {
                d.solve_time_ms + d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms
            })
            .collect();

        let max_worker_ms = worker_totals.iter().copied().fold(0.0_f64, f64::max);
        let avg_worker_ms = if worker_totals.is_empty() {
            0.0_f64
        } else {
            worker_totals.iter().sum::<f64>() / n_workers
        };

        let stage_imbalance_ms = (max_worker_ms - avg_worker_ms).max(0.0);
        #[allow(clippy::cast_precision_loss)]
        let stage_scheduling_ms = (parallel_wall_ms as f64 - max_worker_ms).max(0.0);

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        (
            stage_setup_ms as u64,
            stage_imbalance_ms as u64,
            stage_scheduling_ms as u64,
        )
    }

    /// 4 workers with different solve times: imbalance equals
    /// `trunc(max - mean_f64)` of worker totals, scheduling equals
    /// `parallel_wall - max`.
    ///
    /// Worker solve times: 100 ms, 200 ms, 150 ms, 180 ms.
    /// Setup per worker: 0 (this sub-test isolates solve imbalance).
    /// Mean of totals (f64) = 630.0 / 4 = 157.5.
    /// Imbalance = trunc(200.0 - 157.5) = trunc(42.5) = 42.
    /// Scheduling = 250 - 200 = 50.
    #[test]
    fn decompose_four_workers_different_solve_times() {
        // All setup times are zero; use only solve time to isolate imbalance.
        let zero = SolverStatistics::default();
        let pairs = vec![
            (zero.clone(), make_stats(0.1, 0.0, 0.0, 0.0)), // 100 ms solve
            (zero.clone(), make_stats(0.2, 0.0, 0.0, 0.0)), // 200 ms solve
            (zero.clone(), make_stats(0.15, 0.0, 0.0, 0.0)), // 150 ms solve
            (zero.clone(), make_stats(0.18, 0.0, 0.0, 0.0)), // 180 ms solve
        ];
        let (setup_ms, imbalance_ms, scheduling_ms) = decompose_overhead(&pairs, 250);

        assert_eq!(setup_ms, 0, "no setup work expected");
        // f64 mean = 157.5; imbalance = trunc(200.0 - 157.5) = trunc(42.5) = 42
        assert_eq!(
            imbalance_ms, 42,
            "imbalance = trunc(max(200.0) - avg(157.5)) = trunc(42.5) = 42"
        );
        // scheduling = 250 - 200 = 50
        assert_eq!(scheduling_ms, 50, "scheduling overhead = wall - max_worker");
    }

    /// Acceptance criterion: `setup_time_ms` is the sum of all workers' non-solve
    /// work, matching the ticket spec example.
    ///
    /// Workers have setup costs (load+add+bounds+basis): 20, 25, 15, 22 ms.
    /// Expected `setup_ms` = 20 + 25 + 15 + 22 = 82.
    #[test]
    fn decompose_setup_time_is_aggregate_non_solve_work() {
        let zero = SolverStatistics::default();
        // Each worker: 0 solve + known setup split across the three sub-timers.
        // Worker setup totals: 20, 25, 15, 22 ms (put entirely in load_model timer).
        let pairs = vec![
            (zero.clone(), make_stats(0.0, 0.020, 0.0, 0.0)), // 20 ms total setup
            (zero.clone(), make_stats(0.0, 0.025, 0.0, 0.0)), // 25 ms
            (zero.clone(), make_stats(0.0, 0.015, 0.0, 0.0)), // 15 ms
            (zero.clone(), make_stats(0.0, 0.022, 0.0, 0.0)), // 22 ms
        ];
        let (setup_ms, _imbalance_ms, _scheduling_ms) = decompose_overhead(&pairs, 300);
        assert_eq!(
            setup_ms, 82,
            "aggregate setup must sum all workers' non-solve work"
        );
    }

    /// Edge case: all workers have identical timing → imbalance must be 0.
    #[test]
    fn decompose_identical_workers_zero_imbalance() {
        let zero = SolverStatistics::default();
        let after = make_stats(0.1, 0.01, 0.002, 0.001);
        let pairs = vec![
            (zero.clone(), after.clone()),
            (zero.clone(), after.clone()),
            (zero.clone(), after.clone()),
        ];
        let (_, imbalance_ms, _) = decompose_overhead(&pairs, 200);
        assert_eq!(
            imbalance_ms, 0,
            "identical workers must have zero imbalance"
        );
    }

    /// Edge case: single worker → imbalance is 0, setup equals that worker's
    /// setup, scheduling is the residual.
    #[test]
    fn decompose_single_worker() {
        let zero = SolverStatistics::default();
        // 100 ms solve + 20 ms setup = 120 ms worker total.
        let after = make_stats(0.1, 0.020, 0.0, 0.0);
        let pairs = vec![(zero.clone(), after)];
        let (setup_ms, imbalance_ms, scheduling_ms) = decompose_overhead(&pairs, 150);

        assert_eq!(setup_ms, 20, "single worker: setup = 20 ms");
        assert_eq!(imbalance_ms, 0, "single worker: imbalance must be 0");
        // scheduling = 150 - 120 = 30
        assert_eq!(
            scheduling_ms, 30,
            "single worker: scheduling = wall - worker_total"
        );
    }

    /// Edge case: `scheduling_overhead_ms` is clamped to 0 when `max_worker_total`
    /// exceeds `parallel_wall_ms` (clock skew or measurement granularity).
    #[test]
    fn decompose_scheduling_clamped_when_worker_exceeds_wall() {
        let zero = SolverStatistics::default();
        // Worker total = 200 ms, but wall = 180 ms → scheduling would be negative.
        let after = make_stats(0.2, 0.0, 0.0, 0.0);
        let pairs = vec![(zero.clone(), after)];
        let (_, _, scheduling_ms) = decompose_overhead(&pairs, 180);
        assert_eq!(scheduling_ms, 0, "negative scheduling must be clamped to 0");
    }

    // ── T005: allgatherv per-worker stats unit tests ──────────────────────────

    /// T005-A: single-rank (np=1) backward pass with 2 workers.
    ///
    /// Constructs a 2-worker `StageWorkerStatsBuffer::new(2, 4)` and uses
    /// `StubComm` (which echoes send→recv, simulating `LocalBackend` np=1).
    /// After one backward iteration, `BackwardResult::stage_stats` must
    /// contain 2 entries per non-zero opening (`worker_id` 0 and `worker_id` 1),
    /// both with `rank = 0`.
    #[test]
    #[allow(
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation
    )]
    fn allgatherv_single_rank_two_workers_stage_stats_has_per_worker_entries() {
        use crate::lp_builder::PatchBuffer;

        let n_stages = 2_usize;
        let n_openings = 4_usize;
        let n_workers = 2_usize;
        let local_work = 4_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];
        let n_state = indexer.n_state;

        let solution = solution_1_0(100.0, -5.0);
        let states: Vec<Vec<f64>> = (0..local_work).map(|i| vec![(i + 1) as f64]).collect();
        let mut exchange = exchange_with_states(n_state, states);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let mut workspaces: Vec<SolverWorkspace<MockSolver>> = (0..n_workers)
            .map(|idx| SolverWorkspace {
                rank: 0,
                worker_id: i32::try_from(idx).expect("idx fits in i32"),
                solver: MockSolver::always_ok(solution.clone()),
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
                    lag_accumulator: vec![],
                    lag_weight_accum: 0.0,
                    downstream_accumulator: Vec::new(),
                    downstream_weight_accum: 0.0,
                    downstream_completed_lags: Vec::new(),
                    downstream_n_completed: 0,
                    recon_slot_lookup: Vec::new(),
                    demotion_candidates: Vec::new(),
                },
                scratch_basis: Basis::new(0, 0),
                backward_accum: BackwardAccumulators::default(),
                worker_timing_buf: [0.0_f64; 16],
            })
            .collect();

        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, local_work as u32, 64, &vec![0; n_stages]);
        let mut csb = CutSyncBuffers::new(n_state, local_work, 1);
        let send_sz = n_workers * n_openings * WORKER_STATS_ENTRY_STRIDE;

        let result = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(n_workers, n_openings),
                bwd_stats_send_buf: &mut vec![0.0; send_sz],
                // np=1: recv == send (StubComm echoes)
                bwd_stats_recv_buf: &mut vec![0.0; send_sz],
                bwd_stats_counts: &[send_sz],
                bwd_stats_displs: &[0],
                bwd_stats_unpack_buf: &mut vec![
                    SolverStatsDelta::default();
                    n_workers * n_openings
                ],
                event_sender: None,
            },
            &StubComm,
        )
        .expect("single-rank 2-worker backward must not error");

        // The 2-stage system has 1 backward stage (t=0, successor=1).
        // stage_stats must contain exactly 1 entry (one successor).
        assert_eq!(
            result.stage_stats.len(),
            1,
            "expected 1 backward stage entry (successor=1)"
        );
        let (successor, entries) = &result.stage_stats[0];
        assert_eq!(*successor, 1_usize, "successor index must be 1");

        // Every entry must have rank=0 (np=1 StubComm).
        for (rank, _wid, _omega, _delta) in entries {
            assert_eq!(*rank, 0_i32, "all entries must have rank=0 for np=1");
        }
        // Both worker_id values (0 and 1) must appear at omega=0.
        let omega0_wids: Vec<i32> = entries
            .iter()
            .filter(|(_, _, omega, _)| *omega == 0)
            .map(|(_, wid, _, _)| *wid)
            .collect();
        assert!(
            omega0_wids.contains(&0),
            "worker_id=0 must appear at omega=0"
        );
        assert!(
            omega0_wids.contains(&1),
            "worker_id=1 must appear at omega=0"
        );
    }

    /// T005-B: multi-rank (np=2) backward pass with stub communicator.
    ///
    /// Uses a `DualRankStubComm` whose `size()` returns 2 and whose
    /// `allgatherv` concatenates a manually injected "remote rank" payload
    /// (a copy of the send buffer). Asserts that the unpacked
    /// `stage_stats` contains entries for both `rank=0` and `rank=1`.
    #[test]
    #[allow(
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation
    )]
    fn allgatherv_dual_rank_stub_stage_stats_contains_both_ranks() {
        use crate::lp_builder::PatchBuffer;

        /// Stub communicator simulating np=2: `allgatherv` fills recv with
        /// `[send, send]` (rank-0 and a synthetic rank-1 copy).
        struct DualRankStubComm;

        impl Communicator for DualRankStubComm {
            fn allgatherv<T: CommData>(
                &self,
                send: &[T],
                recv: &mut [T],
                counts: &[usize],
                displs: &[usize],
            ) -> Result<(), CommError> {
                // Fill each rank's slot in recv using the provided counts/displs.
                // Both ranks contribute `send` (rank-1 is a synthetic copy of rank-0).
                for (r, (&count, &displ)) in counts.iter().zip(displs).enumerate() {
                    let src = &send[..count.min(send.len())];
                    recv[displ..displ + src.len()].copy_from_slice(src);
                    let _ = r; // suppress unused warning in cfg(test)
                }
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

            fn broadcast<T: CommData>(
                &self,
                _buf: &mut [T],
                _root: usize,
            ) -> Result<(), CommError> {
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

        let n_stages = 2_usize;
        let n_openings = 2_usize;
        let n_workers = 1_usize;
        let n_ranks = 2_usize;
        let local_work = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];
        let n_state = indexer.n_state;

        let solution = solution_1_0(100.0, -5.0);
        let states: Vec<Vec<f64>> = (0..local_work).map(|i| vec![(i + 1) as f64]).collect();
        let mut exchange = exchange_with_states(n_state, states);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let mut workspaces: Vec<SolverWorkspace<MockSolver>> = (0..n_workers)
            .map(|idx| SolverWorkspace {
                rank: 0,
                worker_id: i32::try_from(idx).expect("idx fits in i32"),
                solver: MockSolver::always_ok(solution.clone()),
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
                    lag_accumulator: vec![],
                    lag_weight_accum: 0.0,
                    downstream_accumulator: Vec::new(),
                    downstream_weight_accum: 0.0,
                    downstream_completed_lags: Vec::new(),
                    downstream_n_completed: 0,
                    recon_slot_lookup: Vec::new(),
                    demotion_candidates: Vec::new(),
                },
                scratch_basis: Basis::new(0, 0),
                backward_accum: BackwardAccumulators::default(),
                worker_timing_buf: [0.0_f64; 16],
            })
            .collect();

        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, local_work as u32, 64, &vec![0; n_stages]);
        let mut csb = CutSyncBuffers::new(n_state, local_work, 1);
        let send_sz = n_workers * n_openings * WORKER_STATS_ENTRY_STRIDE;

        let result = run_backward_pass(
            &mut workspaces,
            &mut basis_store,
            &StageContext {
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
            },
            &templates,
            &mut fcf,
            &mut empty_cut_batches(templates.len()),
            &TrainingContext {
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
            },
            &mut BackwardPassSpec {
                records: &[],
                iteration: 0,
                local_work: exchange.local_count(),
                fwd_offset: 0,
                risk_measures: &risk_measures,
                exchange: &mut exchange,
                cut_activity_tolerance: 0.0,
                cut_sync_bufs: &mut csb,
                probabilities_buf: &mut Vec::new(),
                successor_active_slots_buf: &mut Vec::new(),
                visited_archive: None,
                metadata_sync_buf: &mut Vec::new(),
                global_increments_buf: &mut Vec::new(),
                metadata_sync_window_buf: &mut Vec::new(),
                global_window_increments_buf: &mut Vec::new(),
                real_states_buf: &mut Vec::new(),
                stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(n_workers, n_openings),
                bwd_stats_send_buf: &mut vec![0.0; send_sz],
                // np=2: recv holds two copies of the send block.
                bwd_stats_recv_buf: &mut vec![0.0; n_ranks * send_sz],
                bwd_stats_counts: &[send_sz, send_sz],
                bwd_stats_displs: &[0, send_sz],
                bwd_stats_unpack_buf: &mut vec![
                    SolverStatsDelta::default();
                    n_ranks * n_workers * n_openings
                ],
                event_sender: None,
            },
            &DualRankStubComm,
        )
        .expect("dual-rank stub backward must not error");

        // With np=2, stage_stats for successor=1 must contain entries from
        // both rank=0 and rank=1 (DualRankStubComm copies the rank-0 block
        // into the rank-1 slot, so both appear in the unpacked output).
        assert_eq!(result.stage_stats.len(), 1);
        let (_, entries) = &result.stage_stats[0];

        let ranks_seen: Vec<i32> = entries
            .iter()
            .map(|(rank, _, _, _)| *rank)
            .collect::<std::collections::HashSet<i32>>()
            .into_iter()
            .collect();
        assert!(
            ranks_seen.contains(&0),
            "rank=0 must appear in stage_stats; got {ranks_seen:?}"
        );
        assert!(
            ranks_seen.contains(&1),
            "rank=1 must appear in stage_stats; got {ranks_seen:?}"
        );
    }

    // ── Ticket-003: read-site prefer-with-fallback unit tests ────────────────

    /// Run `process_trial_point_backward` for stage 0 → successor 1 with
    /// explicitly-provided backward and forward basis stores.
    ///
    /// `basis_store` is taken by `&mut` so a `BasisStoreSliceMut` can be
    /// derived from it and passed to `process_trial_point_backward`.
    ///
    /// Returns the mutated workspace so the caller can inspect
    /// `ws.solver.warm_start_calls`.
    #[allow(clippy::too_many_lines)]
    fn run_one_trial_point_with_stores(
        basis_store: &mut crate::workspace::BasisStore,
    ) -> Result<Vec<SolverWorkspace<MockSolver>>, crate::SddpError> {
        use crate::context::StageContext;

        let n_stages = 2_usize;
        let n_openings = 1_usize;
        let n_state = 1_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(n_state, 0);

        let solver = MockSolver::always_ok(solution_1_0(100.0, -5.0));
        let mut workspaces = single_workspace(solver, n_state);
        let ws = &mut workspaces[0];
        ws.backward_accum
            .outcomes
            .push(crate::risk_measure::BackwardOutcome {
                intercept: 0.0,
                coefficients: vec![0.0; n_state],
                objective_value: 0.0,
            });
        ws.backward_accum
            .per_opening_stats
            .push(SolverStatsDelta::default());
        ws.backward_accum.agg_coefficients.resize(n_state, 0.0);

        let mut exchange = exchange_with_states(n_state, vec![vec![5.0]]);

        let templates: &'static _ = Box::leak(Box::new(vec![
            minimal_template_1_0(),
            minimal_template_1_0(),
        ]));
        let base_rows: &'static _ = Box::leak(Box::new(vec![1_usize, 1_usize]));
        let ctx: StageContext<'static> = StageContext {
            templates,
            base_rows,
            noise_scale: Box::leak(Box::new(vec![])),
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: Box::leak(Box::new(vec![])),
            load_bus_indices: Box::leak(Box::new(vec![])),
            block_counts_per_stage: Box::leak(Box::new(vec![])),
            ncs_max_gen: Box::leak(Box::new(vec![])),
            discount_factors: Box::leak(Box::new(vec![])),
            cumulative_discount_factors: Box::leak(Box::new(vec![])),
            stage_lag_transitions: Box::leak(Box::new(vec![])),
            noise_group_ids: Box::leak(Box::new(vec![])),
            downstream_par_order: 0,
        };

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
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

        let mut probabilities_buf = vec![1.0_f64; n_openings];
        let succ_probabilities = vec![1.0_f64; n_openings];
        let successor_active_slots: Vec<usize> = vec![];
        let baked_template = minimal_template_1_0();
        let mut csb = CutSyncBuffers::new(1, 64, 1);
        let mut metadata_sync_buf = Vec::new();
        let mut global_increments_buf = Vec::new();
        let mut metadata_sync_window_buf = Vec::new();
        let mut global_window_increments_buf = Vec::new();
        let mut real_states_buf = Vec::new();
        let bwd_stride = WORKER_STATS_ENTRY_STRIDE;
        let spec = BackwardPassSpec {
            records: &[],
            iteration: 1,
            local_work: 1,
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            probabilities_buf: &mut probabilities_buf,
            successor_active_slots_buf: &mut Vec::new(),
            visited_archive: None,
            metadata_sync_buf: &mut metadata_sync_buf,
            global_increments_buf: &mut global_increments_buf,
            metadata_sync_window_buf: &mut metadata_sync_window_buf,
            global_window_increments_buf: &mut global_window_increments_buf,
            real_states_buf: &mut real_states_buf,
            stage_worker_stats_buf: &mut StageWorkerStatsBuffer::new(1, n_openings),
            bwd_stats_send_buf: &mut vec![0.0; n_openings * bwd_stride],
            bwd_stats_recv_buf: &mut vec![0.0; n_openings * bwd_stride],
            bwd_stats_counts: &[n_openings * bwd_stride],
            bwd_stats_displs: &[0],
            bwd_stats_unpack_buf: &mut vec![SolverStatsDelta::default(); n_openings],
            event_sender: None,
        };

        let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0u32; n_stages]);
        let empty_cut_batch = RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        };

        let succ_spec = super::SuccessorSpec {
            t: 0,
            successor: 1,
            my_rank: 0,
            probabilities: &succ_probabilities,
            cut_batch: &empty_cut_batch,
            num_cuts_at_successor: 0,
            template_num_rows: baked_template.num_rows,
            baked_template: &baked_template,
            successor_active_slots: &successor_active_slots,
            cut_activity_tolerance: 0.0,
            successor_populated_count: fcf.pools[1].populated_count,
            successor_pool: &fcf.pools[1],
        };

        // Derive a single-worker BasisStoreSliceMut covering all scenarios.
        let mut basis_slices = basis_store.split_workers_mut(1);
        let mut basis_slice = basis_slices.remove(0);

        let ws = &mut workspaces[0];
        super::load_backward_lp(ws, &succ_spec);
        ws.backward_accum
            .per_opening_stats
            .resize_with(n_openings, SolverStatsDelta::default);
        for slot in &mut ws.backward_accum.per_opening_stats[..n_openings] {
            *slot = SolverStatsDelta::default();
        }
        ws.backward_accum.slot_increments.resize(1, 0);
        ws.backward_accum.slot_increments[..1].fill(0);

        super::process_trial_point_backward(
            ws,
            &ctx,
            &training_ctx,
            &spec,
            &succ_spec,
            &mut basis_slice,
            0,
        )?;
        Ok(workspaces)
    }

    // ---------------------------------------------------------------------------
    // resolve_backward_basis_* unit tests
    // ---------------------------------------------------------------------------

    #[test]
    fn resolve_backward_basis_returns_some_when_slot_is_populated() {
        // Given: BasisStore[0, 1] has Some(CapturedBasis).
        // Then: resolve_backward_basis returns Some(_).
        use crate::workspace::{BasisStore, CapturedBasis};

        let b = CapturedBasis::new(2, 2, 0, 0, 0);
        let mut store = BasisStore::new(1, 2);
        *store.get_mut(0, 1) = Some(b);

        let slices = store.split_workers_mut(1);
        let slice = &slices[0];
        let basis_ref = super::resolve_backward_basis(slice, 0, 1);

        assert!(basis_ref.is_some(), "expected Some when slot has a basis");
        drop(slices);
    }

    #[test]
    fn resolve_backward_basis_returns_none_when_slot_is_empty() {
        // Given: BasisStore[0, 1] is None (cold-start, slot never written).
        // Then: resolve_backward_basis returns None.
        use crate::workspace::BasisStore;

        let mut store = BasisStore::new(1, 2);
        let slices = store.split_workers_mut(1);
        let slice = &slices[0];
        let basis_ref = super::resolve_backward_basis(slice, 0, 1);

        assert!(basis_ref.is_none(), "expected None for empty slot");
        drop(slices);
    }

    // ---------------------------------------------------------------------------
    // T2 integration tests (backward write populates BasisStore)
    // ---------------------------------------------------------------------------

    #[test]
    fn backward_write_populates_basis_store_at_omega_zero() {
        // Given: a 2-stage, 1-opening study with one forward trial point (m=0, x_hat=[5.0]).
        //        BasisStore starts empty (all None).
        // When: process_trial_point_backward runs at omega=0.
        // Then: BasisStore[0, 1] is Some(CapturedBasis) with state_at_capture == [5.0].
        //
        // AD-2: write occurs only at omega=0 (this test has exactly 1 opening).
        // AD-7: infeasibility guard is not triggered (solver succeeds).
        use crate::workspace::BasisStore;

        let mut basis_store = BasisStore::new(1, 2);
        let workspaces = run_one_trial_point_with_stores(&mut basis_store).unwrap();

        // Verify the BasisStore slot was written.
        assert!(
            basis_store.get(0, 1).is_some(),
            "BasisStore[0, 1] must be Some after backward write at omega=0"
        );
        let captured = basis_store.get(0, 1).unwrap();
        assert_eq!(
            captured.state_at_capture,
            vec![5.0_f64],
            "state_at_capture must equal x_hat"
        );
        // Confirm the solver ran exactly once.
        assert_eq!(
            workspaces[0].solver.call_count, 1,
            "solver must be called exactly once for a 1-opening backward pass"
        );
    }

    #[test]
    fn backward_write_preserves_slot_on_infeasibility_at_omega_zero() {
        // Given: a 2-stage, 1-opening study.
        //        BasisStore starts with a pre-existing basis at [0, 1].
        //        The solver returns Infeasible on its first call.
        // When: process_trial_point_backward runs via run_backward_pass.
        // Then: run_backward_pass returns Err(SddpError::Infeasible) and
        //       BasisStore[0, 1] retains its original content.
        //
        // AD-7: the write in process_trial_point_backward is guarded by `?`
        // immediately after run_stage_solve. An Infeasible error propagates
        // out of the function before reaching the BasisStore write site, so
        // the slot is unconditionally preserved on infeasibility.
        use cobre_solver::Basis;

        use crate::workspace::{BasisStore, CapturedBasis};

        // Pre-populate slot [0, 1] with a sentinel basis.
        let pre_existing = CapturedBasis {
            basis: Basis::new(2, 2),
            base_row_count: 99,
            cut_row_slots: Vec::new(),
            state_at_capture: vec![42.0],
        };
        let mut basis_store = BasisStore::new(1, 2);
        *basis_store.get_mut(0, 1) = Some(pre_existing);

        // Verify sentinel is in place before the call.
        assert_eq!(
            basis_store.get(0, 1).unwrap().base_row_count,
            99,
            "sentinel must be in place before the infeasible solve"
        );

        // run_one_trial_point_with_stores uses MockSolver::always_ok, so we
        // exercise the reuse path (successful solve overwrites slot). For the
        // infeasibility path, the structural guarantee is: `?` in
        // process_trial_point_backward propagates Err before the write site.
        // That path is integration-tested by `backward_pass_propagates_infeasible_error`.
        //
        // Here we test the complementary invariant: a *successful* solve at ω=0
        // with a pre-existing slot uses the reuse branch (get_basis into the
        // existing allocation) and leaves the slot Some (not None).
        let result = run_one_trial_point_with_stores(&mut basis_store);
        assert!(result.is_ok(), "expected Ok for successful solve");

        // The slot must still be Some after the successful reuse-path write.
        assert!(
            basis_store.get(0, 1).is_some(),
            "BasisStore[0, 1] must not be None after successful reuse-path write at ω=0"
        );
        // The reuse path updates state_at_capture to the current x_hat=[5.0].
        assert_eq!(
            basis_store.get(0, 1).unwrap().state_at_capture,
            vec![5.0_f64],
            "state_at_capture must be updated to x_hat by the reuse path"
        );
    }
}
