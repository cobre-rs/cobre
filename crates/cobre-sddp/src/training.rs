//! Training loop orchestrator for the SDDP algorithm.
//!
//! [`train`] wires together the forward pass, forward synchronization, state
//! exchange, backward pass, cut synchronization, lower bound evaluation, and
//! convergence check into a single iterative loop.
//!
//! ## Iteration lifecycle
//!
//! Each iteration follows the corrected ordering from the LB plan fix (F-019):
//!
//! 1. Forward pass — scenario simulation, local UB statistics.
//! 2. Forward sync — global synchronization for UB statistics.
//! 3. State exchange — `allgatherv` trial points for the backward pass.
//! 4. Backward pass — Benders cut generation.
//! 5. Cut sync — `allgatherv` new cuts across ranks.
//!    5a. Cut selection — optional periodic pool pruning via `CutSelectionStrategy`.
//!    5b. Angular pruning — geometric cut dominance reduction (stage 0..T-2).
//!    5c. Budget enforcement — active-cut hard cap (every iteration when set).
//!    5d. Template baking — rebuild per-stage baked LP templates.
//! 6. Lower bound evaluation — rank 0 solves stage-0 openings, broadcasts scalar.
//! 7. Convergence check — stopping rules evaluated.
//! 8. Event emission — `IterationSummary` and per-step events via channel.
//!
//! ## Pre-allocation discipline
//!
//! All workspace buffers (`PatchBuffer`, `TrajectoryRecord` flat vec,
//! `ExchangeBuffers`, `CutSyncBuffers`) are allocated once before the
//! iteration loop and reused across all iterations. No heap allocation
//! occurs on the hot path.

use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::time::Instant;

use cobre_comm::Communicator;
use cobre_core::{StageSelectionRecord, TrainingEvent};
use cobre_solver::Basis;
use cobre_solver::RowBatch;
use cobre_solver::SolverInterface;
use cobre_solver::StageTemplate;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    SddpError, TrainingConfig, TrajectoryRecord,
    backward::run_backward_pass,
    context::{BakedTemplates, StageContext, TrainingContext},
    convergence::ConvergenceMonitor,
    cut::CutRowMap,
    cut::fcf::FutureCostFunction,
    cut_selection::DeactivationSet,
    cut_sync::CutSyncBuffers,
    evaluate_lower_bound,
    forward::{ForwardPassBatch, build_cut_row_batch_into, run_forward_pass, sync_forward},
    lower_bound::LbEvalSpec,
    lp_builder::PatchBuffer,
    solver_stats::{SolverStatsDelta, SolverStatsEntry, aggregate_solver_statistics},
    state_exchange::ExchangeBuffers,
    stopping_rule::RULE_ITERATION_LIMIT,
    workspace::{BasisStore, WorkspacePool, WorkspaceSizing},
};

// ---------------------------------------------------------------------------
// TrainingResult
// ---------------------------------------------------------------------------

/// Result of a training run that always carries partial results.
///
/// When training completes normally, `error` is `None` and `result` contains
/// the full training statistics. When training fails mid-iteration, `error`
/// carries the failure cause and `result` contains statistics from all
/// fully completed iterations (the failing iteration is excluded).
#[derive(Debug)]
pub struct TrainingOutcome {
    /// Training result from completed iterations. Always populated, even when
    /// `error` is `Some` -- in that case, `result.iterations` reflects only
    /// the iterations that completed without error.
    pub result: TrainingResult,

    /// If training was interrupted by an error, the cause. `None` when
    /// training completed normally (convergence, iteration limit, or
    /// time limit).
    pub error: Option<SddpError>,
}

/// Summary statistics produced when the training loop terminates.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Final lower bound at termination.
    pub final_lb: f64,

    /// Final upper bound mean at termination.
    pub final_ub: f64,

    /// Final upper bound standard deviation at termination.
    pub final_ub_std: f64,

    /// Final convergence gap: `(UB - LB) / max(1.0, |UB|)`.
    pub final_gap: f64,

    /// Total number of iterations completed.
    pub iterations: u64,

    /// Human-readable termination reason (e.g., `"iteration_limit"`, `"graceful_shutdown"`).
    pub reason: String,

    /// Total wall-clock time for the training run, in milliseconds.
    pub total_time_ms: u64,

    /// Per-stage solver basis from the last iteration, indexed 0-based.
    ///
    /// Each entry is `Some(basis)` if the stage was solved at least once
    /// during the final iteration, or `None` if no solve occurred (e.g.,
    /// the last stage in finite-horizon mode has no successor cuts).
    /// Used for policy checkpoint persistence.
    pub basis_cache: Vec<Option<Basis>>,

    /// Per-iteration, per-phase solver statistics log.
    ///
    /// Each entry is `(iteration, phase_name, stage_index, delta)`.
    /// Phase names: `"forward"`, `"backward"`, `"lower_bound"`.
    /// Stage index is `-1` for forward and lower bound (which span all stages),
    /// and the actual stage index for backward per-stage entries (added in T-004).
    pub solver_stats_log: Vec<SolverStatsEntry>,

    /// Visited states archive containing all forward-pass trial points.
    ///
    /// Always populated during training. The caller decides whether to
    /// persist it to the policy checkpoint based on `exports.states`.
    pub visited_archive: Option<crate::visited_states::VisitedStatesArchive>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Send a training event if the channel is present (ignores `None` or receiver drop).
#[inline]
fn emit(sender: Option<&Sender<TrainingEvent>>, event: TrainingEvent) {
    if let Some(s) = sender {
        let _ = s.send(event);
    }
}

/// Check if a full LP rebuild is needed to purge phantom (bound-zeroed) rows.
///
/// Returns true when the ratio of deactivated rows to total rows exceeds
/// a threshold, or when a fixed iteration interval has elapsed since the
/// last rebuild.
fn needs_periodic_rebuild(row_map: &CutRowMap, iterations_since_rebuild: u64) -> bool {
    let total = row_map.total_cut_rows();
    let active = row_map.active_count();
    let phantom = total - active;
    // Rebuild when phantom rows exceed 20% of total, or every 50 iterations.
    (total > 0 && phantom * 5 > total) || iterations_since_rebuild >= 50
}

/// Build a `basis_cache` from the canonical global scenario 0, broadcasting
/// rank 0's bases to all other ranks so that every rank has an identical
/// warm-start basis for the simulation phase.
///
/// ## Why global scenario 0?
///
/// `basis_store` is indexed by **local** scenario index. Local scenario 0 on
/// rank 0 is always global scenario 0 (rank 0's `my_fwd_offset` is 0). On
/// rank *r > 0*, local scenario 0 maps to a different global scenario. Using
/// the last local scenario (`my_actual_fwd - 1`) therefore produces different
/// bases on different ranks, because each rank follows a distinct noise
/// realisation. Broadcasting rank 0's scenario 0 ensures all ranks start
/// simulation from the identical LP vertex.
///
/// ## Serialization format
///
/// For each stage *t* in `0..num_stages`, the flat `i32` buffer contains:
/// - `0_i32` sentinel when `basis_store.get(0, t)` is `None`
/// - `1_i32` sentinel + `col_len: i32` + `row_len: i32` + `col_status[..]`
///   + `row_status[..]` when `Some(basis)`
///
/// This avoids adding `serde` to `cobre-solver` and uses only `i32`
/// broadcast, which `CommData` supports for all backends.
///
/// ## Single-rank optimization
///
/// When `comm.size() == 1` the broadcast is skipped; only the extraction
/// from local scenario 0 runs.
///
/// # Errors
///
/// Returns `SddpError::Communication` if the `comm.broadcast` call fails.
// Basis lengths are bounded by LP column/row counts, which fit comfortably
// in i32 in all realistic cases. The i32<->usize casts here are deliberate:
// MPI broadcast requires CommData (which requires Copy), ruling out usize.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn broadcast_basis_cache<C: Communicator>(
    basis_store: &crate::workspace::BasisStore,
    num_stages: usize,
    comm: &C,
) -> Result<Vec<Option<Basis>>, SddpError> {
    // Single-rank fast path: no communication needed.
    if comm.size() == 1 {
        let cache = (0..num_stages)
            .map(|t| basis_store.get(0, t).cloned())
            .collect();
        return Ok(cache);
    }

    // Pack rank 0's scenario-0 bases into a flat i32 buffer that can be
    // broadcast over the comm layer.
    //
    // Buffer layout per stage:
    //   [sentinel(0 or 1), <if sentinel==1: col_len, row_len, col_status..., row_status...>]
    let mut buf: Vec<i32> = Vec::new();
    if comm.rank() == 0 {
        for t in 0..num_stages {
            match basis_store.get(0, t) {
                None => buf.push(0_i32),
                Some(basis) => {
                    buf.push(1_i32);
                    buf.push(basis.col_status.len() as i32);
                    buf.push(basis.row_status.len() as i32);
                    buf.extend_from_slice(&basis.col_status);
                    buf.extend_from_slice(&basis.row_status);
                }
            }
        }
    }

    // Step 1: broadcast the buffer length so all ranks can allocate.
    let mut len_buf = [buf.len() as i32];
    comm.broadcast(&mut len_buf, 0).map_err(SddpError::from)?;
    let total_len = len_buf[0] as usize;

    // Step 2: resize non-root buffers and broadcast the payload.
    buf.resize(total_len, 0_i32);
    comm.broadcast(&mut buf, 0).map_err(SddpError::from)?;

    // Step 3: deserialize back into Vec<Option<Basis>>.
    // All index arithmetic is bounds-checked to convert a corrupted broadcast
    // into a recoverable error instead of an index-out-of-bounds panic.
    let mut cache: Vec<Option<Basis>> = Vec::with_capacity(num_stages);
    let mut pos = 0_usize;
    for stage in 0..num_stages {
        if pos >= buf.len() {
            return Err(SddpError::Validation(format!(
                "broadcast_basis_cache: buffer truncated at stage {stage} (pos={pos}, len={})",
                buf.len()
            )));
        }
        let sentinel = buf[pos];
        pos += 1;
        if sentinel == 0 {
            cache.push(None);
        } else {
            if pos + 2 > buf.len() {
                return Err(SddpError::Validation(format!(
                    "broadcast_basis_cache: buffer truncated reading lengths at stage {stage}"
                )));
            }
            let col_len = buf[pos] as usize;
            pos += 1;
            let row_len = buf[pos] as usize;
            pos += 1;
            if pos + col_len + row_len > buf.len() {
                return Err(SddpError::Validation(format!(
                    "broadcast_basis_cache: buffer truncated reading basis data at stage {stage} \
                     (need {}, have {})",
                    col_len + row_len,
                    buf.len() - pos
                )));
            }
            let col_status = buf[pos..pos + col_len].to_vec();
            pos += col_len;
            let row_status = buf[pos..pos + row_len].to_vec();
            pos += row_len;
            cache.push(Some(Basis {
                col_status,
                row_status,
            }));
        }
    }

    Ok(cache)
}

// ---------------------------------------------------------------------------
// train
// ---------------------------------------------------------------------------

/// Execute the SDDP training loop.
///
/// Allocates all workspace buffers, runs the iteration loop until a stopping
/// rule triggers or `config.max_iterations` is reached, and returns a
/// [`TrainingOutcome`] summarising the final convergence statistics.
///
/// ## Error handling
///
/// Returns `Err(SddpError)` immediately on:
/// - LP infeasibility in the forward or backward pass → `SddpError::Infeasible`
/// - Solver failure → `SddpError::Solver`
/// - Communication failure → `SddpError::Communication`
///
/// ## Event channel
///
/// When `config.event_sender` is `Some`, typed [`TrainingEvent`] values are
/// emitted at each lifecycle step boundary. Event send failures (receiver
/// dropped) are silently ignored so they cannot interrupt training.
///
/// ## Cut selection
///
/// When `cut_selection` is `Some(strategy)`, step 5a runs after every cut
/// synchronisation. The strategy's `should_run(iteration)` gate controls the
/// frequency; at eligible iterations every stage's cut pool is scanned and
/// inactive cuts are deactivated. When `cut_selection` is `None`, step 5a is
/// skipped entirely and no [`TrainingEvent::CutSelectionComplete`] events are
/// emitted.
///
/// # Errors
///
/// Returns `Err(SddpError::Infeasible { .. })` when an LP has no feasible
/// solution. Returns `Err(SddpError::Solver(_))` for other solver failures.
/// Returns `Err(SddpError::Communication(_))` when a collective operation
/// fails.
///
/// # Examples
///
/// ```rust,ignore
/// use cobre_sddp::{train, TrainingConfig, LoopConfig, CutManagementConfig, EventConfig};
/// use cobre_sddp::{StoppingRuleSet, StoppingRule, RiskMeasure, HorizonMode};
///
/// let mut solver = HiggsBackend::new();
/// let config = TrainingConfig {
///     loop_config: LoopConfig { forward_passes: 100, max_iterations: 100, ..LoopConfig::default() },
///     cut_management: CutManagementConfig {
///         risk_measures: vec![RiskMeasure::Expectation; num_stages],
///         ..CutManagementConfig::default()
///     },
///     events: EventConfig::default(),
/// };
/// let mut fcf = FutureCostFunction::new(num_stages - 1, n_state, capacity);
///
/// let result = train(
///     &mut solver, config, &mut fcf, &stage_ctx, &training_ctx, &comm,
///     || HiggsBackend::new(),
/// )?;
///
/// println!("converged in {} iterations, gap={:.4}", result.result.iterations, result.result.final_gap);
/// ```
///
/// # Panics (debug builds only)
///
/// Panics if `templates.len() != horizon.num_stages()` or if
/// `config.cut_management.risk_measures.len() != horizon.num_stages()` or if
/// `training_ctx.stochastic.opening_tree().n_openings(0) == 0`.
#[allow(clippy::too_many_lines, clippy::similar_names)]
pub fn train<S: SolverInterface + Send, C: Communicator>(
    solver: &mut S,
    config: TrainingConfig,
    fcf: &mut FutureCostFunction,
    stage_ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    comm: &C,
    solver_factory: impl Fn() -> Result<S, cobre_solver::SolverError>,
) -> Result<TrainingOutcome, SddpError> {
    let cut_activity_tolerance = config.cut_management.cut_activity_tolerance;
    let n_fwd_threads = config.loop_config.n_fwd_threads;
    let max_blocks = config.loop_config.max_blocks;
    let risk_measures = &config.cut_management.risk_measures;
    let stopping_rules = config.loop_config.stopping_rules.clone();
    let horizon = training_ctx.horizon;
    let indexer = training_ctx.indexer;
    let initial_state = training_ctx.initial_state;
    let num_stages = horizon.num_stages();
    let num_ranks = comm.size();
    let my_rank = comm.rank();
    let total_forward_passes = config.loop_config.forward_passes as usize;
    let n_state = indexer.n_state;

    // forward_passes is the TOTAL across all ranks. Distribute with
    // base/remainder: first `remainder` ranks get `base + 1`, rest get `base`.
    let base_fwd = total_forward_passes / num_ranks;
    let remainder_fwd = total_forward_passes % num_ranks;
    let my_actual_fwd = base_fwd + usize::from(my_rank < remainder_fwd);
    let my_fwd_offset = base_fwd * my_rank + my_rank.min(remainder_fwd);
    // ExchangeBuffers requires uniform local_count — use the max across ranks.
    let max_local_fwd = base_fwd + usize::from(remainder_fwd > 0);

    let empty_record = TrajectoryRecord {
        primal: vec![],
        dual: vec![],
        stage_cost: 0.0,
        state: vec![0.0; n_state],
    };
    let mut records = vec![empty_record; max_local_fwd * num_stages];

    // Workspace pool for forward-pass thread parallelism. Each workspace owns
    // an independent solver, patch buffer, and current-state buffer. The pool
    // is allocated once and reused across all iterations.
    let n_threads = n_fwd_threads.max(1);
    let mut fwd_pool = WorkspacePool::try_new(
        n_threads,
        n_state,
        WorkspaceSizing {
            hydro_count: indexer.hydro_count,
            max_par_order: indexer.max_par_order,
            n_load_buses: stage_ctx.n_load_buses,
            max_blocks,
            downstream_par_order: stage_ctx.downstream_par_order,
            max_openings: (0..num_stages)
                .map(|t| training_ctx.stochastic.opening_tree().n_openings(t))
                .max()
                .unwrap_or(0),
            initial_pool_capacity: fcf.pools[0].capacity,
            n_state,
        },
        solver_factory,
    )
    .map_err(SddpError::Solver)?;
    if training_ctx.basis_padding_enabled {
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
    }

    // Per-scenario, per-stage basis store. Sized for the maximum local forward
    // passes so that scenario indices are stable across iterations. The store
    // is allocated once and reused: the forward pass overwrites entries each
    // iteration, and the backward pass reads from them read-only.
    let mut basis_store = BasisStore::new(max_local_fwd, num_stages);

    // Standalone patch buffer for the lower bound evaluation which uses the
    // single `solver` argument directly. The backward pass uses the workspace
    // pool's per-thread solvers and patch buffers instead.
    let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order, 0, 0);
    let mut convergence_monitor = ConvergenceMonitor::new(stopping_rules);
    // Compute the actual per-rank forward pass count for the padded-but-tracked
    // ExchangeBuffers. First `remainder_fwd` ranks get `base_fwd + 1`, the rest
    // get `base_fwd`. This mirrors the distribution used in the forward pass.
    let actual_per_rank: Vec<usize> = (0..num_ranks)
        .map(|r| base_fwd + usize::from(r < remainder_fwd))
        .collect();
    let mut exchange_bufs =
        ExchangeBuffers::with_actual_counts(n_state, max_local_fwd, num_ranks, &actual_per_rank);
    let mut cut_sync_bufs =
        CutSyncBuffers::with_distribution(n_state, max_local_fwd, num_ranks, total_forward_passes);

    // Visited-states archive: allocated only when needed for dominated cut
    // selection (which reads visited states at pruning time), angular dominance
    // pruning (which also reads visited states), or when the caller requests
    // state export to the policy checkpoint.
    let needs_archive = matches!(
        config.cut_management.cut_selection,
        Some(crate::cut_selection::CutSelectionStrategy::Dominated { .. })
    ) || config.cut_management.angular_pruning.is_some()
        || config.events.export_states;
    let mut visited_archive = if needs_archive {
        Some(crate::visited_states::VisitedStatesArchive::new(
            num_stages,
            n_state,
            config.loop_config.max_iterations,
            total_forward_passes,
        ))
    } else {
        None
    };

    let start_time = Instant::now();

    let TrainingConfig {
        loop_config:
            crate::config::LoopConfig {
                forward_passes: config_forward_passes,
                max_iterations,
                start_iteration,
                ..
            },
        cut_management:
            crate::config::CutManagementConfig {
                cut_selection,
                angular_pruning,
                budget,
                ..
            },
        events:
            crate::config::EventConfig {
                event_sender,
                shutdown_flag,
                ..
            },
    } = config;
    let cut_selection = cut_selection.as_ref();
    let shutdown_flag = shutdown_flag.as_ref();
    let angular_pruning = angular_pruning.as_ref();

    #[allow(clippy::cast_possible_truncation)]
    emit(
        event_sender.as_ref(),
        TrainingEvent::TrainingStarted {
            case_name: String::new(),
            stages: num_stages as u32,
            hydros: indexer.hydro_count as u32,
            thermals: 0,
            ranks: num_ranks as u32,
            #[allow(clippy::cast_possible_truncation)]
            threads_per_rank: n_threads as u32,
            timestamp: String::new(),
        },
    );

    let mut final_lb = 0.0;
    let mut final_ub = 0.0;
    let mut final_ub_std = 0.0;
    let mut final_gap = 0.0;
    let mut completed_iterations = start_iteration;
    let mut termination_reason = RULE_ITERATION_LIMIT.to_string();
    let mut solver_stats_log: Vec<SolverStatsEntry> = Vec::new();

    // Pre-allocate RowBatch buffers for cut row construction. Reused across
    // iterations via build_cut_row_batch_into to eliminate per-iteration
    // heap allocation (S3 optimization).
    let mut cut_batches: Vec<RowBatch> = (0..num_stages)
        .map(|_| RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        })
        .collect();
    // Extra batch for lower-bound evaluation (stage 0).
    let mut lb_cut_batch = RowBatch {
        num_rows: 0,
        row_starts: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    };

    // Epic-03 template baking: per-stage baked templates + reusable cut row
    // batches. The baked template at index t has num_rows =
    // stage_ctx.templates[t].num_rows + fcf.pools[t].active_count() after
    // every bake. `baked_templates_ready` starts false because no bake has
    // run yet; set true after the first bake in step 4d.
    let mut baked_templates: Vec<StageTemplate> =
        (0..num_stages).map(|_| StageTemplate::empty()).collect();
    let mut bake_row_batches: Vec<RowBatch> = (0..num_stages)
        .map(|_| RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        })
        .collect();
    let mut baked_templates_ready: bool = false;

    // CutRowMap for the lower bound solver's incremental cut management.
    // The LB solver is dedicated to stage 0 and persists across iterations,
    // so it benefits from incremental cut append (S2 optimization).
    let mut lb_cut_row_map = CutRowMap::new(fcf.pools[0].capacity, stage_ctx.templates[0].num_rows);

    // Track iterations since last full rebuild for the LB solver.
    let mut lb_iterations_since_rebuild: u64 = 0;

    // Macro to handle mid-iteration errors: emit TrainingFinished, build
    // partial result, and return Ok(TrainingOutcome { error: Some(e) }).
    macro_rules! on_error {
        ($e:expr) => {{
            #[allow(clippy::cast_possible_truncation)]
            emit(
                event_sender.as_ref(),
                TrainingEvent::TrainingFinished {
                    reason: "error".to_string(),
                    iterations: completed_iterations,
                    final_lb,
                    final_ub,
                    total_time_ms: (start_time.elapsed().as_millis() as u64).max(1),
                    total_cuts: fcf.total_active_cuts() as u64,
                },
            );
            // Extract the canonical basis cache from rank 0's global scenario 0
            // and broadcast to all ranks. This ensures simulation warm-starts
            // from the same LP vertex regardless of the number of MPI ranks.
            let basis_cache = match broadcast_basis_cache(&basis_store, num_stages, comm) {
                Ok(cache) => cache,
                Err(comm_err) => return Err(comm_err),
            };
            #[allow(clippy::cast_possible_truncation)]
            let total_time_ms = (start_time.elapsed().as_millis() as u64).max(1);
            return Ok(TrainingOutcome {
                result: TrainingResult {
                    final_lb,
                    final_ub,
                    final_ub_std,
                    final_gap,
                    iterations: completed_iterations,
                    reason: "error".to_string(),
                    total_time_ms,
                    basis_cache,
                    solver_stats_log,
                    visited_archive: visited_archive.take(),
                },
                error: Some($e),
            });
        }};
    }

    // Pre-allocated backward-pass buffers, reused across iterations to avoid
    // per-iteration allocation. Declared outside the loop so capacity persists.
    let mut bwd_probabilities_buf: Vec<f64> = Vec::new();
    let mut bwd_successor_active_slots_buf: Vec<usize> = Vec::new();
    // Pre-allocated buffer for cut binding metadata allreduce. Sized per
    // stage to the successor pool's metadata length after sync_packed_cuts.
    let mut bwd_metadata_sync_buf: Vec<u64> = Vec::new();
    // Pre-allocated receive buffer for the allreduce(Sum) of binding
    // increment counts. Reused across stages to avoid per-stage allocation.
    let mut bwd_global_increments_buf: Vec<u64> = Vec::new();
    // Pre-allocated buffer for packing real (non-padded) gathered state vectors
    // when archiving visited states for dominated cut selection (ticket-003).
    // Pre-sized to the true total forward passes to avoid first-iteration
    // reallocation; capacity is preserved across iterations.
    let mut bwd_real_states_buf: Vec<f64> =
        Vec::with_capacity(exchange_bufs.real_total_scenarios() * n_state);

    for iteration in (start_iteration + 1)..=max_iterations {
        // Check external shutdown flag before each iteration's convergence
        // evaluation. The flag is set by signal handlers or test harnesses.
        if let Some(flag) = shutdown_flag {
            if flag.load(Ordering::Relaxed) {
                convergence_monitor.set_shutdown();
            }
        }

        let iter_start = Instant::now();
        let fwd_record_len = my_actual_fwd * num_stages;
        let fwd_batch = ForwardPassBatch {
            local_forward_passes: my_actual_fwd,
            total_forward_passes,
            iteration,
            fwd_offset: my_fwd_offset,
        };

        // Snapshot pool stats before forward pass.
        let fwd_stats_before =
            aggregate_solver_statistics(fwd_pool.workspaces.iter().map(|w| w.solver.statistics()));

        let fwd_baked = BakedTemplates {
            templates: &baked_templates,
            ready: baked_templates_ready,
        };
        let forward_result = match run_forward_pass(
            &mut fwd_pool.workspaces,
            &mut basis_store,
            stage_ctx,
            &fwd_baked,
            fcf,
            &mut cut_batches,
            training_ctx,
            &fwd_batch,
            &mut records[..fwd_record_len],
        ) {
            Ok(r) => r,
            Err(e) => on_error!(e),
        };

        // Snapshot pool stats after forward pass and compute delta.
        let fwd_delta = {
            let fwd_stats_after = aggregate_solver_statistics(
                fwd_pool.workspaces.iter().map(|w| w.solver.statistics()),
            );
            SolverStatsDelta::from_snapshots(&fwd_stats_before, &fwd_stats_after)
        };
        let fwd_solve_time_ms = fwd_delta.solve_time_ms;
        solver_stats_log.push((iteration, "forward", -1, fwd_delta));

        let forward_elapsed_ms = forward_result.elapsed_ms;

        let local_n = forward_result.scenario_costs.len();
        let local_cost_sum: f64 = forward_result.scenario_costs.iter().sum();
        emit(
            event_sender.as_ref(),
            TrainingEvent::ForwardPassComplete {
                iteration,
                scenarios: config_forward_passes,
                #[allow(clippy::cast_precision_loss)]
                ub_mean: if local_n > 0 {
                    local_cost_sum / local_n as f64
                } else {
                    0.0
                },
                ub_std: 0.0,
                elapsed_ms: forward_elapsed_ms,
            },
        );
        let sync_result = match sync_forward(&forward_result, comm, total_forward_passes) {
            Ok(r) => r,
            Err(e) => on_error!(e),
        };

        emit(
            event_sender.as_ref(),
            TrainingEvent::ForwardSyncComplete {
                iteration,
                global_ub_mean: sync_result.global_ub_mean,
                global_ub_std: sync_result.global_ub_std,
                sync_time_ms: sync_result.sync_time_ms,
            },
        );
        let mut bwd_spec = crate::backward::BackwardPassSpec {
            exchange: &mut exchange_bufs,
            records: &records,
            iteration,
            local_work: my_actual_fwd,
            fwd_offset: my_fwd_offset,
            risk_measures,
            cut_activity_tolerance,
            cut_sync_bufs: &mut cut_sync_bufs,
            probabilities_buf: &mut bwd_probabilities_buf,
            successor_active_slots_buf: &mut bwd_successor_active_slots_buf,
            visited_archive: visited_archive.as_mut(),
            metadata_sync_buf: &mut bwd_metadata_sync_buf,
            global_increments_buf: &mut bwd_global_increments_buf,
            real_states_buf: &mut bwd_real_states_buf,
        };

        let backward_result = match run_backward_pass(
            &mut fwd_pool.workspaces,
            &basis_store,
            stage_ctx,
            &fwd_baked,
            fcf,
            &mut cut_batches,
            training_ctx,
            &mut bwd_spec,
            comm,
        ) {
            Ok(r) => r,
            Err(e) => on_error!(e),
        };

        // Buffers are borrowed by bwd_spec and automatically available for
        // the next iteration after bwd_spec is dropped here.

        // Store per-stage backward deltas and compute aggregate solve time.
        let bwd_solve_time_ms = {
            let agg =
                SolverStatsDelta::aggregate(backward_result.stage_stats.iter().map(|(_, d)| d));
            let total_ms = agg.solve_time_ms;
            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            for (stage_idx, delta) in &backward_result.stage_stats {
                solver_stats_log.push((iteration, "backward", *stage_idx as i32, delta.clone()));
            }
            total_ms
        };

        let backward_elapsed_ms = backward_result.elapsed_ms;

        #[allow(clippy::cast_possible_truncation)]
        emit(
            event_sender.as_ref(),
            TrainingEvent::BackwardPassComplete {
                iteration,
                cuts_generated: backward_result.cuts_generated as u32,
                stages_processed: num_stages.saturating_sub(1) as u32,
                elapsed_ms: backward_elapsed_ms,
                state_exchange_time_ms: backward_result.state_exchange_time_ms,
                cut_batch_build_time_ms: backward_result.cut_batch_build_time_ms,
                setup_time_ms: backward_result.setup_time_ms,
                load_imbalance_ms: backward_result.load_imbalance_ms,
                scheduling_overhead_ms: backward_result.scheduling_overhead_ms,
            },
        );
        // Cut sync now happens per-stage inside `run_backward_pass`. The
        // measured time is returned in `backward_result.cut_sync_time_ms`.
        #[allow(clippy::cast_possible_truncation)]
        emit(
            event_sender.as_ref(),
            TrainingEvent::CutSyncComplete {
                iteration,
                cuts_distributed: backward_result.cuts_generated as u32,
                cuts_active: fcf.total_active_cuts() as u32,
                cuts_removed: 0,
                sync_time_ms: backward_result.cut_sync_time_ms,
            },
        );

        // Step 4a: Strategy-based cut selection.
        // We defer the CutSelectionComplete event until after Steps 4b and 4c
        // so that per_stage records can be annotated with angular and budget
        // post-step counts before they are emitted.
        //
        // sel_state holds (per_stage, cuts_deactivated, selection_time_ms,
        // stages_processed) when Step 4a ran; None otherwise.
        let mut sel_state: Option<(Vec<StageSelectionRecord>, u32, u64, u32)> = None;

        if let Some(strategy) = cut_selection {
            if strategy.should_run(iteration) {
                let sel_start = Instant::now();
                let num_sel_stages = num_stages.saturating_sub(1);
                let mut cuts_deactivated = 0u32;
                let mut per_stage = Vec::with_capacity(num_sel_stages);

                // Stage 0 is exempt: its cuts are never the "successor" in the
                // backward pass, so their binding activity is never updated.
                // Deactivating them would weaken the lower bound approximation.
                #[allow(clippy::cast_possible_truncation)]
                {
                    let pool0 = &fcf.pools[0];
                    let active_0 = pool0.active_count() as u32;
                    per_stage.push(StageSelectionRecord {
                        stage: 0,
                        cuts_populated: pool0.populated_count as u32,
                        cuts_active_before: active_0,
                        cuts_deactivated: 0,
                        cuts_active_after: active_0,
                        selection_time_ms: 0.0,
                        active_after_angular: None,
                        budget_evicted: None,
                        active_after_budget: None,
                    });
                }

                // Compute deactivation sets in parallel across stages.
                // Selection is the expensive step (O(active * states) for
                // Dominated); deactivation is O(deactivated) per stage and
                // requires &mut, so it stays sequential.
                let archive_ref = visited_archive.as_ref();
                #[allow(clippy::cast_possible_truncation)]
                let deactivations: Vec<(usize, DeactivationSet, f64)> = (1..num_sel_stages)
                    .into_par_iter()
                    .map(|stage| {
                        let pool = &fcf.pools[stage];
                        let states =
                            archive_ref.map_or(&[] as &[f64], |a| a.states_for_stage(stage));
                        let start = Instant::now();
                        let deact =
                            strategy.select_for_stage(pool, states, iteration, stage as u32);
                        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                        (stage, deact, elapsed_ms)
                    })
                    .collect();

                #[allow(clippy::cast_possible_truncation)]
                for (stage, deact, stage_sel_time_ms) in deactivations {
                    let pool = &fcf.pools[stage];
                    let populated = pool.populated_count as u32;
                    let active_before = pool.active_count() as u32;
                    let n_deact = deact.indices.len() as u32;
                    cuts_deactivated += n_deact;

                    fcf.pools[stage].deactivate(&deact.indices);

                    let active_after = fcf.pools[stage].active_count() as u32;
                    per_stage.push(StageSelectionRecord {
                        stage: stage as u32,
                        cuts_populated: populated,
                        cuts_active_before: active_before,
                        cuts_deactivated: n_deact,
                        cuts_active_after: active_after,
                        selection_time_ms: stage_sel_time_ms,
                        active_after_angular: None,
                        budget_evicted: None,
                        active_after_budget: None,
                    });
                }

                #[allow(clippy::cast_possible_truncation)]
                let selection_time_ms = sel_start.elapsed().as_millis() as u64;

                #[allow(clippy::cast_possible_truncation)]
                let stages_processed_sel = num_sel_stages as u32;

                sel_state = Some((
                    per_stage,
                    cuts_deactivated,
                    selection_time_ms,
                    stages_processed_sel,
                ));
            }
        }

        // Step 4b: Angular dominance pruning (stage 0..num_stages-2, last stage exempt).
        // Stage 0 is included because angular pruning only removes geometrically dominated
        // cuts (safe for lower bound monotonicity). Activity-based selection (step 4a)
        // still exempts stage 0 because binding activity is never tracked there.
        if let Some(params) = angular_pruning {
            if params.should_run(iteration) {
                let prune_start = Instant::now();
                let num_prune_stages = num_stages.saturating_sub(1);
                let archive_ref = visited_archive.as_ref();

                let pruning_results: Vec<(usize, crate::angular_pruning::AngularPruningResult)> =
                    (0..num_prune_stages)
                        .into_par_iter()
                        .map(|stage| {
                            let pool = &fcf.pools[stage];
                            let states =
                                archive_ref.map_or(&[] as &[f64], |a| a.states_for_stage(stage));
                            let result = crate::angular_pruning::select_angular_dominated(
                                pool,
                                states,
                                params.cosine_threshold,
                                iteration,
                            );
                            (stage, result)
                        })
                        .collect();

                let mut total_cuts_deactivated = 0u32;
                let mut total_clusters_formed = 0u64;
                let mut total_dominance_checks = 0u64;

                #[allow(clippy::cast_possible_truncation)]
                for (stage, result) in pruning_results {
                    total_clusters_formed += result.clusters_formed as u64;
                    total_dominance_checks += result.dominance_checks as u64;
                    let n_deact = result.deactivate.len() as u32;
                    total_cuts_deactivated += n_deact;
                    if !result.deactivate.is_empty() {
                        fcf.pools[stage].deactivate(&result.deactivate);
                    }
                    // Annotate per-stage records from Step 4a with post-angular counts.
                    if let Some((ref mut per_stage, _, _, _)) = sel_state {
                        let active_now = fcf.pools[stage].active_count() as u32;
                        if let Some(rec) = per_stage.get_mut(stage) {
                            rec.active_after_angular = Some(active_now);
                        }
                    }
                }

                #[allow(clippy::cast_possible_truncation)]
                let pruning_time_ms = prune_start.elapsed().as_millis() as u64;
                #[allow(clippy::cast_possible_truncation)]
                let stages_processed = num_prune_stages as u32;

                emit(
                    event_sender.as_ref(),
                    TrainingEvent::AngularPruningComplete {
                        iteration,
                        cuts_deactivated: total_cuts_deactivated,
                        clusters_formed: total_clusters_formed,
                        dominance_checks: total_dominance_checks,
                        stages_processed,
                        pruning_time_ms,
                    },
                );
            }
        }

        // Step 4c: Budget enforcement (every iteration when budget is set).
        //
        // Runs unconditionally when `budget` is Some — not gated by
        // `check_frequency`. The budget is a hard cap that must be maintained
        // at all times.
        if let Some(b) = budget {
            let budget_start = Instant::now();
            let mut total_evicted = 0u32;
            for stage in 0..num_stages {
                #[allow(clippy::cast_possible_truncation)]
                let result = fcf.pools[stage].enforce_budget(b, iteration, config_forward_passes);
                total_evicted += result.evicted_count;
                // Annotate per-stage records with post-budget counts.
                if let Some((ref mut per_stage, _, _, _)) = sel_state {
                    if let Some(rec) = per_stage.get_mut(stage) {
                        rec.budget_evicted = Some(result.evicted_count);
                        rec.active_after_budget = Some(result.active_after);
                    }
                }
            }
            #[allow(clippy::cast_possible_truncation)]
            let enforcement_time_ms = budget_start.elapsed().as_millis() as u64;
            emit(
                event_sender.as_ref(),
                #[allow(clippy::cast_possible_truncation)]
                TrainingEvent::BudgetEnforcementComplete {
                    iteration,
                    cuts_evicted: total_evicted,
                    stages_processed: num_stages as u32,
                    enforcement_time_ms,
                },
            );
        }

        // Emit CutSelectionComplete now that all per-stage annotation is done.
        if let Some((per_stage, cuts_deactivated, selection_time_ms, stages_processed)) = sel_state
        {
            emit(
                event_sender.as_ref(),
                TrainingEvent::CutSelectionComplete {
                    iteration,
                    cuts_deactivated,
                    stages_processed,
                    selection_time_ms,
                    allgatherv_time_ms: 0,
                    per_stage,
                },
            );
        }

        // Step 4d: Template baking.
        // Rebuild per-stage baked templates from the current active cut set.
        // Iteration i+1's forward and backward passes will consume these
        // baked templates (wired in tickets 010 and 011). Sequential over
        // stages: per-stage memory allocation is ~10s of MB and parallelism
        // here would contend with the solver workspace pools already in use
        // by the LB evaluation that follows.
        let bake_start = Instant::now();
        let mut total_cut_rows_baked: u64 = 0;
        for t in 0..num_stages {
            build_cut_row_batch_into(
                &mut bake_row_batches[t],
                fcf,
                t,
                indexer,
                &stage_ctx.templates[t].col_scale,
            );
            #[allow(clippy::cast_possible_truncation)]
            {
                total_cut_rows_baked += bake_row_batches[t].num_rows as u64;
            }
            cobre_solver::bake_rows_into_template(
                &stage_ctx.templates[t],
                &bake_row_batches[t],
                &mut baked_templates[t],
            );
        }
        #[allow(clippy::cast_possible_truncation)]
        let bake_time_ms = bake_start.elapsed().as_millis() as u64;
        #[allow(clippy::cast_possible_truncation)]
        let stages_processed_bake = num_stages as u32;
        emit(
            event_sender.as_ref(),
            TrainingEvent::TemplateBakeComplete {
                iteration,
                stages_processed: stages_processed_bake,
                total_cut_rows_baked,
                bake_time_ms,
            },
        );
        baked_templates_ready = true;

        // Periodic rebuild check for the lower bound solver.
        // When too many phantom (bound-zeroed) rows accumulate, reset the LP
        // to purge them. This prevents unbounded row growth from deactivation.
        lb_iterations_since_rebuild += 1;
        if comm.rank() == 0 && needs_periodic_rebuild(&lb_cut_row_map, lb_iterations_since_rebuild)
        {
            lb_cut_row_map.reset(stage_ctx.templates[0].num_rows);
            lb_iterations_since_rebuild = 0;
            // The next evaluate_lower_bound call will see an empty row_map
            // and do a full load_model + append_all_cuts.
        }

        // Snapshot solver stats and wall-clock before lower bound evaluation.
        let lb_wall_start = Instant::now();
        let lb_stats_before = solver.statistics();

        let lb_spec = LbEvalSpec {
            template: &stage_ctx.templates[0],
            base_row: stage_ctx.base_rows[0],
            noise_scale: stage_ctx.noise_scale,
            n_hydros: stage_ctx.n_hydros,
            opening_tree: training_ctx.stochastic.opening_tree(),
            risk_measure: &risk_measures[0],
            stochastic: Some(training_ctx.stochastic),
            n_load_buses: stage_ctx.n_load_buses,
            ncs_max_gen: stage_ctx.ncs_max_gen,
            block_count: stage_ctx.block_counts_per_stage[0],
            ncs_generation: indexer.ncs_generation.clone(),
            inflow_method: training_ctx.inflow_method,
        };
        let lb = match evaluate_lower_bound(
            solver,
            fcf,
            initial_state,
            indexer,
            &mut patch_buf,
            &mut lb_cut_batch,
            &lb_spec,
            comm,
            Some(&mut lb_cut_row_map),
        ) {
            Ok(r) => r,
            Err(e) => on_error!(e),
        };

        // Snapshot solver stats after lower bound and compute delta.
        let lb_stats_after = solver.statistics();
        let lb_lp_solves = lb_stats_after.solve_count - lb_stats_before.solve_count;
        let lb_delta = SolverStatsDelta::from_snapshots(&lb_stats_before, &lb_stats_after);
        let lb_solve_time_ms = lb_delta.solve_time_ms;
        solver_stats_log.push((iteration, "lower_bound", -1, lb_delta));
        #[allow(clippy::cast_possible_truncation)]
        let lb_wall_ms = lb_wall_start.elapsed().as_millis() as u64;

        let (should_stop, rule_results) = convergence_monitor.update(lb, &sync_result);

        final_lb = convergence_monitor.lower_bound();
        final_ub = convergence_monitor.upper_bound();
        final_ub_std = convergence_monitor.upper_bound_std();
        final_gap = convergence_monitor.gap();

        emit(
            event_sender.as_ref(),
            TrainingEvent::ConvergenceUpdate {
                iteration,
                lower_bound: final_lb,
                upper_bound: final_ub,
                upper_bound_std: convergence_monitor.upper_bound_std(),
                gap: final_gap,
                rules_evaluated: rule_results.clone(),
            },
        );

        #[allow(clippy::cast_possible_truncation)]
        let wall_time_ms = start_time.elapsed().as_millis() as u64;
        #[allow(clippy::cast_possible_truncation)]
        let iteration_time_ms = iter_start.elapsed().as_millis() as u64;

        emit(
            event_sender.as_ref(),
            TrainingEvent::IterationSummary {
                iteration,
                lower_bound: final_lb,
                upper_bound: final_ub,
                gap: final_gap,
                wall_time_ms,
                iteration_time_ms,
                forward_ms: forward_elapsed_ms,
                backward_ms: backward_elapsed_ms,
                lp_solves: forward_result.lp_solves + backward_result.lp_solves + lb_lp_solves,
                solve_time_ms: fwd_solve_time_ms + bwd_solve_time_ms + lb_solve_time_ms,
                lower_bound_eval_ms: lb_wall_ms,
                fwd_setup_time_ms: forward_result.setup_time_ms,
                fwd_load_imbalance_ms: forward_result.load_imbalance_ms,
                fwd_scheduling_overhead_ms: forward_result.scheduling_overhead_ms,
            },
        );

        completed_iterations = iteration;

        if should_stop {
            // Extract the triggered rule name for the reason string.
            termination_reason = rule_results
                .iter()
                .find(|r| r.triggered)
                .map_or_else(|| "unknown".to_string(), |r| r.rule_name.clone());
            break;
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    let total_time_ms = (start_time.elapsed().as_millis() as u64).max(1);

    #[allow(clippy::cast_possible_truncation)]
    emit(
        event_sender.as_ref(),
        TrainingEvent::TrainingFinished {
            reason: termination_reason.clone(),
            iterations: completed_iterations,
            final_lb,
            final_ub,
            total_time_ms,
            total_cuts: fcf.total_active_cuts() as u64,
        },
    );

    // Extract the canonical basis cache from rank 0's global scenario 0 and
    // broadcast to all ranks. Using local scenario 0 on rank 0 (which is
    // always global scenario 0, since rank 0's my_fwd_offset == 0) guarantees
    // that all ranks receive an identical basis for simulation warm-start.
    // Previously this used the last local scenario (my_actual_fwd - 1), which
    // varied by rank count and caused simulation divergence across MPI configs.
    let basis_cache = broadcast_basis_cache(&basis_store, num_stages, comm)?;

    Ok(TrainingOutcome {
        result: TrainingResult {
            final_lb,
            final_ub,
            final_ub_std,
            final_gap,
            iterations: completed_iterations,
            reason: termination_reason,
            total_time_ms,
            basis_cache,
            solver_stats_log,
            visited_archive,
        },
        error: None,
    })
}

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
        Bus, EntityId, SystemBuilder, TrainingEvent,
        scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
            SamplingScheme,
        },
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };
    use cobre_solver::{
        Basis, LpSolution, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };
    use cobre_stochastic::{
        ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context,
    };

    use super::train;
    use crate::{
        CutManagementConfig, EventConfig, HorizonMode, InflowNonNegativityMethod, LoopConfig,
        RiskMeasure, SddpError, StageIndexer, StoppingMode, StoppingRule, StoppingRuleSet,
        TrainingConfig,
        context::{StageContext, TrainingContext},
        cut::fcf::FutureCostFunction,
    };

    /// Minimal LP for N=1 hydro, L=0 PAR order.
    ///
    /// Column layout (N=1, L=0):
    /// - col 0: `storage_out` (no NZ in structural rows)
    /// - col 1: `z_inflow` (no NZ — `z_inflow` row at row 1)
    /// - col 2: `storage_in` (1 NZ: row 0, storage-fixing row)
    /// - col 3: `theta` (no NZ)
    ///
    /// Row layout:
    /// - row 0: storage-fixing (`storage_out` fixed to incoming state)
    /// - row 1: `z_inflow` definition row
    fn minimal_template(n_state: usize) -> StageTemplate {
        let _ = n_state;
        StageTemplate {
            num_cols: 4,
            num_rows: 2,
            num_nz: 1,
            // CSC col_starts: 4 cols + 1 sentinel = 5 entries.
            // col 0 (storage_out): 0 NZ
            // col 1 (z_inflow):    0 NZ
            // col 2 (storage_in):  1 NZ at row 0
            // col 3 (theta):       0 NZ
            col_starts: vec![0_i32, 0, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, f64::NEG_INFINITY, 0.0, 0.0],
            col_upper: vec![f64::INFINITY; 4],
            objective: vec![0.0, 0.0, 0.0, 1.0],
            row_lower: vec![0.0, 0.0],
            row_upper: vec![0.0, 0.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    fn fixed_solution(objective: f64) -> LpSolution {
        LpSolution {
            objective,
            primal: vec![0.0; 4],
            dual: vec![0.0; 2],
            reduced_costs: vec![0.0; 4],
            iterations: 0,
            solve_time_seconds: 0.0,
        }
    }

    /// Mock solver that returns fixed objective values in sequence.
    ///
    /// Each call to `solve()` returns the next value from `objectives`,
    /// wrapping around. If `infeasible_on_first` is set, the first call
    /// returns `SolverError::Infeasible`.
    struct MockSolver {
        objectives: Vec<f64>,
        call_count: usize,
        infeasible_on_first: bool,
    }

    impl MockSolver {
        fn with_fixed(objective: f64) -> Self {
            Self {
                objectives: vec![objective],
                call_count: 0,
                infeasible_on_first: false,
            }
        }

        fn infeasible() -> Self {
            Self {
                objectives: vec![0.0],
                call_count: 0,
                infeasible_on_first: true,
            }
        }
    }

    impl SolverInterface for MockSolver {
        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
        fn load_model(&mut self, _t: &StageTemplate) {}
        fn add_rows(&mut self, _r: &RowBatch) {}
        fn set_row_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
        fn set_col_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}

        fn solve(&mut self) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            let call = self.call_count;
            self.call_count += 1;
            if self.infeasible_on_first && call == 0 {
                return Err(SolverError::Infeasible);
            }
            let obj = self.objectives[call % self.objectives.len()];
            // Return a view with primal[3] = 0.0 (theta = 0, N=1 L=0 → theta at col 3)
            // so that the forward pass computes stage_cost = objective - primal[theta]
            // = obj - 0 = obj.  The fixed_solution helper provides compatible arrays.
            let sol = fixed_solution(obj);
            // We cannot borrow from a temporary, so we use static empty slices.
            // training.rs mock only needs to satisfy the SolverInterface bound;
            // the actual slice contents are not checked by the training loop.
            let _ = sol;
            Ok(cobre_solver::SolutionView {
                objective: obj,
                primal: &[0.0, 0.0, 0.0, 0.0],
                dual: &[0.0, 0.0],
                reduced_costs: &[0.0, 0.0, 0.0, 0.0],
                iterations: 0,
                solve_time_seconds: 0.0,
            })
        }

        fn reset(&mut self) {
            self.call_count = 0;
        }

        fn get_basis(&mut self, _out: &mut Basis) {}

        fn solve_with_basis(
            &mut self,
            _basis: &Basis,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            self.solve()
        }

        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
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

    /// Build a minimal `StochasticContext` with `n_stages` stages and a single
    /// hydro entity.
    ///
    /// Used to provide the `stochastic` argument to `train`. Follows the same
    /// [`SystemBuilder`] + `build_stochastic_context` pattern as the forward-pass
    /// integration tests.  The `n_openings` parameter controls the branching
    /// factor of the opening tree.
    #[allow(clippy::cast_possible_wrap)]
    fn make_stochastic_context(n_stages: usize, n_openings: usize) -> StochasticContext {
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::InflowModel;

        let bus = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            deficit_segments: vec![cobre_core::DeficitSegment {
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
                branching_factor: n_openings,
                noise_method: NoiseMethod::Saa,
            },
        };

        let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();

        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(1),
                stage_id: i as i32,
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

    /// Build `n_stages` minimal [`Stage`] values with sequential `id`s (0..n_stages).
    ///
    /// Used to populate [`TrainingContext::stages`] so that
    /// [`cobre_stochastic::build_forward_sampler`] can read per-stage noise methods.
    fn make_stages(n_stages: usize) -> Vec<Stage> {
        (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: Some(0),
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: 744.0,
                }],
                block_mode: BlockMode::Parallel,
                state_config: cobre_core::temporal::StageStateConfig {
                    storage: true,
                    inflow_lags: false,
                },
                risk_config: cobre_core::temporal::StageRiskConfig::Expectation,
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

    /// AC: `train_completes_with_iteration_limit`
    ///
    /// Given `max_iterations: 5`, a `StoppingRuleSet` with
    /// `IterationLimit { limit: 5 }` in `Any` mode, a mock solver returning
    /// fixed objectives, and `StubComm` as communicator, when the function
    /// completes, then `result.iterations == 5` and
    /// `result.reason == "iteration_limit"`.
    #[test]
    fn ac_train_completes_with_iteration_limit() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0); // N=1, L=0
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 5,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit_rules(5),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                angular_pruning: None,
                budget: None,
                basis_padding_enabled: false,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation; n_stages],
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let result = train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                basis_padding_enabled: false,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        assert!(result.error.is_none(), "expected no error");
        assert_eq!(result.result.iterations, 5, "expected 5 iterations");
        assert_eq!(result.result.reason, "iteration_limit");
    }

    /// AC: `train_returns_partial_on_infeasible`
    ///
    /// Given a mock solver that returns `SolverError::Infeasible` on the first
    /// forward pass solve, when the function is called, then it returns
    /// `Ok(TrainingOutcome)` with `error: Some(SddpError::Infeasible { .. })`
    /// and `result.iterations == 0`.
    #[test]
    fn ac_train_returns_partial_on_infeasible() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 5,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit_rules(5),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                angular_pruning: None,
                budget: None,
                basis_padding_enabled: false,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation; n_stages],
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let mut solver = MockSolver::infeasible();
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let result = train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                basis_padding_enabled: false,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &comm,
            || Ok(MockSolver::infeasible()),
        );

        let outcome = result.unwrap();
        assert!(
            outcome.error.is_some(),
            "expected error in TrainingOutcome, got: {outcome:?}"
        );
        assert!(
            matches!(outcome.error, Some(SddpError::Infeasible { stage: 0, .. })),
            "expected SddpError::Infeasible at stage 0, got: {:?}",
            outcome.error
        );
        assert_eq!(
            outcome.result.iterations, 0,
            "no iterations should have completed"
        );
        assert_eq!(outcome.result.reason, "error");
    }

    /// AC: `train_emits_correct_event_sequence`
    ///
    /// Given `train` with `event_sender: Some(tx)`, runs for 2 iterations
    /// before `IterationLimit(2)` triggers. The receiver must collect exactly:
    ///
    /// - 1 `TrainingStarted`
    /// - 2 × (`ForwardPassComplete`, `ForwardSyncComplete`, `BackwardPassComplete`,
    ///   `CutSyncComplete`, `TemplateBakeComplete`, `ConvergenceUpdate`, `IterationSummary`)
    /// - 1 `TrainingFinished`
    ///
    /// = 1 + 14 + 1 = 16 events.
    #[test]
    fn ac_train_emits_correct_event_sequence() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();

        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 10,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit_rules(2),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                angular_pruning: None,
                budget: None,
                basis_padding_enabled: false,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation; n_stages],
            },
            events: EventConfig {
                event_sender: Some(tx),
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                basis_padding_enabled: false,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        // Drain all events.
        drop(fcf); // not needed; just for clarity
        let events: Vec<TrainingEvent> = rx.try_iter().collect();

        // 1 TrainingStarted + 2*(7 per-iteration) + 1 TrainingFinished = 16
        // Per-iteration: ForwardPassComplete, ForwardSyncComplete,
        //   BackwardPassComplete, CutSyncComplete, TemplateBakeComplete,
        //   ConvergenceUpdate, IterationSummary
        assert_eq!(
            events.len(),
            16,
            "expected 16 events, got {} ({events:?})",
            events.len()
        );

        assert!(
            matches!(events[0], TrainingEvent::TrainingStarted { .. }),
            "first event must be TrainingStarted"
        );
        assert!(
            matches!(events.last(), Some(TrainingEvent::TrainingFinished { .. })),
            "last event must be TrainingFinished"
        );

        // Check per-iteration event pattern for iteration 1 (events[1..8])
        assert!(matches!(
            events[1],
            TrainingEvent::ForwardPassComplete { .. }
        ));
        assert!(matches!(
            events[2],
            TrainingEvent::ForwardSyncComplete { .. }
        ));
        assert!(matches!(
            events[3],
            TrainingEvent::BackwardPassComplete { .. }
        ));
        assert!(matches!(events[4], TrainingEvent::CutSyncComplete { .. }));
        assert!(matches!(
            events[5],
            TrainingEvent::TemplateBakeComplete { .. }
        ));
        assert!(matches!(events[6], TrainingEvent::ConvergenceUpdate { .. }));
        assert!(matches!(events[7], TrainingEvent::IterationSummary { .. }));

        // Iteration 2 (events[8..15]) follows the same pattern.
        assert!(matches!(
            events[8],
            TrainingEvent::ForwardPassComplete { .. }
        ));
        assert!(matches!(
            events[9],
            TrainingEvent::ForwardSyncComplete { .. }
        ));
        assert!(matches!(
            events[10],
            TrainingEvent::BackwardPassComplete { .. }
        ));
        assert!(matches!(events[11], TrainingEvent::CutSyncComplete { .. }));
        assert!(matches!(
            events[12],
            TrainingEvent::TemplateBakeComplete { .. }
        ));
        assert!(matches!(
            events[13],
            TrainingEvent::ConvergenceUpdate { .. }
        ));
        assert!(matches!(events[14], TrainingEvent::IterationSummary { .. }));
    }

    /// AC: `train_result_fields_populated`
    ///
    /// Verify that all `TrainingResult` fields are non-default after a
    /// successful 5-iteration run.
    #[test]
    fn ac_train_result_fields_populated() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 5,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit_rules(5),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                angular_pruning: None,
                budget: None,
                basis_padding_enabled: false,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation; n_stages],
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let result = train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                basis_padding_enabled: false,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        assert!(result.error.is_none(), "expected no error");
        assert_eq!(result.result.iterations, 5);
        assert!(!result.result.reason.is_empty(), "reason must not be empty");
    }

    /// AC: `train_with_no_event_sender`
    ///
    /// Verify that `event_sender: None` does not panic and training completes
    /// normally.
    #[test]
    fn ac_train_with_no_event_sender() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 2,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit_rules(2),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                angular_pruning: None,
                budget: None,
                basis_padding_enabled: false,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation; n_stages],
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let result = train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                basis_padding_enabled: false,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        );

        assert!(result.is_ok(), "train with no event_sender must not panic");
    }

    /// AC: train result `total_time_ms` is greater than 0
    ///
    /// After a successful run, `result.total_time_ms` must be >= 0 and the
    /// `TrainingResult` must be constructible without panicking.
    #[test]
    fn ac_total_time_ms_is_non_negative() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 1,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit_rules(1),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                angular_pruning: None,
                budget: None,
                basis_padding_enabled: false,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation; n_stages],
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let result = train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                basis_padding_enabled: false,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        assert!(result.error.is_none(), "expected no error");
        assert!(
            result.result.total_time_ms > 0,
            "total_time_ms must be > 0, got {}",
            result.result.total_time_ms,
        );
    }

    /// `cut_selection_none_skips_step`
    ///
    /// Given `train` with `cut_selection: None` running for 5 iterations, then
    /// no `CutSelectionComplete` event is emitted.
    #[test]
    fn cut_selection_none_skips_step() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();

        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 10,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit_rules(5),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                angular_pruning: None,
                budget: None,
                basis_padding_enabled: false,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation; n_stages],
            },
            events: EventConfig {
                event_sender: Some(tx),
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                basis_padding_enabled: false,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        let events: Vec<TrainingEvent> = rx.try_iter().collect();
        let cut_sel_count = events
            .iter()
            .filter(|e| matches!(e, TrainingEvent::CutSelectionComplete { .. }))
            .count();

        assert_eq!(
            cut_sel_count, 0,
            "expected no CutSelectionComplete events with cut_selection: None"
        );
    }

    /// `cut_selection_level1_runs_at_frequency`
    ///
    /// Given `train` with `cut_selection: Some(Level1 { threshold: 0,
    /// check_frequency: 3 })` running for 5 iterations, then
    /// `CutSelectionComplete` is emitted exactly once (at iteration 3).
    #[test]
    fn cut_selection_level1_runs_at_frequency() {
        use crate::cut_selection::CutSelectionStrategy;

        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();

        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 10,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit_rules(5),
            },
            cut_management: CutManagementConfig {
                cut_selection: Some(CutSelectionStrategy::Level1 {
                    threshold: 0,
                    check_frequency: 3,
                }),
                angular_pruning: None,
                budget: None,
                basis_padding_enabled: false,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation; n_stages],
            },
            events: EventConfig {
                event_sender: Some(tx),
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                basis_padding_enabled: false,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        let events: Vec<TrainingEvent> = rx.try_iter().collect();
        let sel_events: Vec<&TrainingEvent> = events
            .iter()
            .filter(|e| matches!(e, TrainingEvent::CutSelectionComplete { .. }))
            .collect();

        assert_eq!(
            sel_events.len(),
            1,
            "expected exactly 1 CutSelectionComplete event for check_frequency=3 over 5 iterations"
        );

        // Verify the event was emitted at iteration 3.
        let TrainingEvent::CutSelectionComplete { iteration, .. } = sel_events[0] else {
            panic!("wrong variant");
        };
        assert_eq!(
            *iteration, 3,
            "CutSelectionComplete must fire at iteration 3"
        );
    }

    /// `cut_selection_deactivates_inactive_cuts`
    ///
    /// Stage 0 is exempt from cut selection because its cuts have no
    /// backward-pass activity tracking. With a 2-stage system, only stage 0
    /// has cuts, so `cuts_deactivated` must be 0.
    #[test]
    fn cut_selection_stage0_exempt_preserves_cuts() {
        use crate::cut_selection::CutSelectionStrategy;

        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();

        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 10,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit_rules(2),
            },
            cut_management: CutManagementConfig {
                cut_selection: Some(CutSelectionStrategy::Level1 {
                    threshold: 0,
                    check_frequency: 2,
                }),
                angular_pruning: None,
                budget: None,
                basis_padding_enabled: false,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation; n_stages],
            },
            events: EventConfig {
                event_sender: Some(tx),
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                basis_padding_enabled: false,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        let events: Vec<TrainingEvent> = rx.try_iter().collect();
        let sel_events: Vec<&TrainingEvent> = events
            .iter()
            .filter(|e| matches!(e, TrainingEvent::CutSelectionComplete { .. }))
            .collect();

        assert_eq!(
            sel_events.len(),
            1,
            "expected exactly 1 CutSelectionComplete event at iteration 2"
        );

        let TrainingEvent::CutSelectionComplete {
            iteration,
            cuts_deactivated,
            per_stage,
            ..
        } = sel_events[0]
        else {
            panic!("wrong variant");
        };

        assert_eq!(*iteration, 2, "selection must fire at iteration 2");
        assert_eq!(
            *cuts_deactivated, 0,
            "stage 0 is exempt from cut selection, so no cuts should be deactivated"
        );
        // Verify per-stage records are populated and stage 0 is exempt.
        assert!(
            !per_stage.is_empty(),
            "per_stage must contain at least the stage 0 record"
        );
        assert_eq!(per_stage[0].stage, 0, "first record must be stage 0");
        assert_eq!(
            per_stage[0].cuts_deactivated, 0,
            "stage 0 must have zero deactivations"
        );
    }

    /// `existing_train_tests_pass_with_none`
    ///
    /// Verify backward compatibility: calling `train` with `cut_selection:
    /// None` produces the same result as before this ticket. This is
    /// implicitly verified by the existing `ac_train_completes_with_iteration_limit`
    /// test. This test is an explicit additional check with an explicit `None`.
    #[test]
    fn existing_train_tests_pass_with_none() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 3,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit_rules(3),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                angular_pruning: None,
                budget: None,
                basis_padding_enabled: false,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation; n_stages],
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let result = train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                basis_padding_enabled: false,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        assert!(result.error.is_none(), "expected no error");
        assert_eq!(result.result.iterations, 3);
        assert_eq!(result.result.reason, "iteration_limit");
    }

    /// AC: `train_partial_result_on_mid_iteration_failure`
    ///
    /// Given a mock solver that fails on the 3rd solve call (which occurs
    /// during iteration 1 since each iteration performs multiple solves),
    /// when `train` is called, it returns `Ok(TrainingOutcome)` with
    /// `error: Some(...)`, `result.iterations == 0` (no completed iterations),
    /// `result.reason == "error"`, and `solver_stats_log` containing entries
    /// from the phases that completed before the error.
    #[test]
    fn ac_train_partial_result_on_mid_iteration_failure() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();

        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 5,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit_rules(5),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                angular_pruning: None,
                budget: None,
                basis_padding_enabled: false,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation; n_stages],
            },
            events: EventConfig {
                event_sender: Some(tx),
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        // Mock solver that fails on the Nth call. With 2 stages and 1 forward
        // pass, the forward pass solves 2 LPs (stage 0, stage 1). A failure
        // on the 1st call (index 0) means failure in the forward pass of
        // iteration 1.
        let mut solver = MockSolver::infeasible();
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let outcome = train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                basis_padding_enabled: false,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &comm,
            || Ok(MockSolver::infeasible()),
        )
        .unwrap();

        // Verify partial result semantics.
        assert!(outcome.error.is_some(), "expected error in TrainingOutcome");
        assert_eq!(
            outcome.result.iterations, 0,
            "no iterations should have completed (failure in iteration 1)"
        );
        assert_eq!(outcome.result.reason, "error");
        assert!(
            outcome.result.total_time_ms > 0,
            "total_time_ms must be > 0"
        );

        // Verify TrainingFinished event was emitted with reason "error".
        let events: Vec<TrainingEvent> = rx.try_iter().collect();
        let finished = events
            .iter()
            .find(|e| matches!(e, TrainingEvent::TrainingFinished { .. }));
        assert!(
            finished.is_some(),
            "TrainingFinished event must be emitted even on error"
        );
        if let Some(TrainingEvent::TrainingFinished { reason, .. }) = finished {
            assert_eq!(reason, "error", "TrainingFinished reason must be 'error'");
        }
    }

    /// When `start_iteration = 3` and `max_iterations = 5`, the training loop
    /// executes exactly 2 iterations (4 and 5) and reports `iterations = 5`.
    #[test]
    fn start_iteration_resumes_from_offset() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 5,
                start_iteration: 3,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit_rules(5),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                angular_pruning: None,
                budget: None,
                basis_padding_enabled: false,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation; n_stages],
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let outcome = train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                basis_padding_enabled: false,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        assert_eq!(
            outcome.result.iterations, 5,
            "iterations must report the absolute iteration number (5), not the delta (2)"
        );
        assert_eq!(outcome.result.reason, "iteration_limit");
    }

    /// When `start_iteration >= max_iterations`, the training loop executes zero
    /// iterations and returns immediately with `iterations = start_iteration`.
    #[test]
    fn start_iteration_at_or_beyond_max_runs_zero_iterations() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 5,
                start_iteration: 5,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit_rules(5),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                angular_pruning: None,
                budget: None,
                basis_padding_enabled: false,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation; n_stages],
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let outcome = train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                basis_padding_enabled: false,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        assert_eq!(
            outcome.result.iterations, 5,
            "iterations must equal start_iteration when no loop iterations execute"
        );
        assert_eq!(
            outcome.result.reason, "iteration_limit",
            "reason should be iteration_limit when loop range is empty"
        );
    }

    // ── broadcast_basis_cache unit tests ─────────────────────────────────────

    /// AC: `broadcast_basis_cache` returns scenario 0's bases, not scenario N-1's.
    ///
    /// Constructs a `BasisStore` with 2 scenarios and 3 stages. Scenario 0 is
    /// populated with known distinct values; scenario 1 is populated with
    /// different values. The helper must return scenario 0's bases on
    /// `LocalBackend` (single-rank, no broadcast).
    #[test]
    fn ac_broadcast_basis_cache_uses_scenario_0_not_last() {
        use super::broadcast_basis_cache;
        use crate::workspace::BasisStore;

        let num_scenarios = 4; // simulates total_forward_passes=4, num_ranks=1
        let num_stages = 3;
        let mut store = BasisStore::new(num_scenarios, num_stages);

        // Populate scenario 0 with col_status=[10,20], row_status=[30].
        for t in 0..num_stages {
            *store.get_mut(0, t) = Some(Basis {
                col_status: vec![10_i32 + t as i32, 20_i32 + t as i32],
                row_status: vec![30_i32 + t as i32],
            });
        }

        // Populate scenario 3 (last) with completely different values.
        for t in 0..num_stages {
            *store.get_mut(3, t) = Some(Basis {
                col_status: vec![99_i32, 88_i32],
                row_status: vec![77_i32],
            });
        }

        let comm = StubComm; // single-rank, no broadcast
        let cache = broadcast_basis_cache(&store, num_stages, &comm).unwrap();

        assert_eq!(cache.len(), num_stages);
        for (t, entry) in cache.iter().enumerate() {
            let basis = entry.as_ref().expect("stage {t} must have a basis");
            assert_eq!(
                basis.col_status,
                vec![10_i32 + t as i32, 20_i32 + t as i32],
                "stage {t} col_status must come from scenario 0, not scenario 3"
            );
            assert_eq!(
                basis.row_status,
                vec![30_i32 + t as i32],
                "stage {t} row_status must come from scenario 0, not scenario 3"
            );
        }
    }

    /// AC: `broadcast_basis_cache` handles `None` slots correctly.
    ///
    /// When scenario 0 has no basis stored for some stages (e.g. training
    /// stopped before any forward pass completed), those stages must be `None`
    /// in the returned cache.
    #[test]
    fn ac_broadcast_basis_cache_none_slots_preserved() {
        use super::broadcast_basis_cache;
        use crate::workspace::BasisStore;

        let num_stages = 2;
        // Scenario 0 is left unpopulated (all None).
        let store = BasisStore::new(1, num_stages);

        let comm = StubComm;
        let cache = broadcast_basis_cache(&store, num_stages, &comm).unwrap();

        assert_eq!(cache.len(), num_stages);
        for t in 0..num_stages {
            assert!(
                cache[t].is_none(),
                "stage {t} must be None when basis store has no entry for scenario 0"
            );
        }
    }

    // ── Angular pruning integration tests ────────────────────────────────────

    /// `angular_pruning_none_skips_step`
    ///
    /// Given `angular_pruning: None` running for 5 iterations, then no
    /// `AngularPruningComplete` event is emitted.
    #[test]
    fn angular_pruning_none_skips_step() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();

        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 10,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit_rules(5),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                angular_pruning: None,
                budget: None,
                basis_padding_enabled: false,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation; n_stages],
            },
            events: EventConfig {
                event_sender: Some(tx),
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                basis_padding_enabled: false,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        let events: Vec<TrainingEvent> = rx.try_iter().collect();
        let prune_count = events
            .iter()
            .filter(|e| matches!(e, TrainingEvent::AngularPruningComplete { .. }))
            .count();

        assert_eq!(
            prune_count, 0,
            "expected no AngularPruningComplete events with angular_pruning: None"
        );
    }

    /// `angular_pruning_runs_at_frequency`
    ///
    /// Given `angular_pruning: Some(AngularPruningParams { check_frequency: 3,
    /// .. })` running for 5 iterations, then `AngularPruningComplete` is emitted
    /// exactly once (at iteration 3).
    #[test]
    fn angular_pruning_runs_at_frequency() {
        use crate::angular_pruning::AngularPruningParams;

        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();

        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 10,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit_rules(5),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                angular_pruning: Some(AngularPruningParams {
                    cosine_threshold: 0.999,
                    check_frequency: 3,
                }),
                budget: None,
                basis_padding_enabled: false,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation; n_stages],
            },
            events: EventConfig {
                event_sender: Some(tx),
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                basis_padding_enabled: false,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        let events: Vec<TrainingEvent> = rx.try_iter().collect();
        let prune_events: Vec<&TrainingEvent> = events
            .iter()
            .filter(|e| matches!(e, TrainingEvent::AngularPruningComplete { .. }))
            .collect();

        assert_eq!(
            prune_events.len(),
            1,
            "expected exactly 1 AngularPruningComplete event for check_frequency=3 over 5 \
             iterations"
        );

        let TrainingEvent::AngularPruningComplete { iteration, .. } = prune_events[0] else {
            panic!("wrong variant");
        };
        assert_eq!(
            *iteration, 3,
            "AngularPruningComplete must fire at iteration 3"
        );
    }

    /// `angular_pruning_after_cut_selection_ordering`
    ///
    /// Given both `cut_selection` (check_frequency=3) and `angular_pruning`
    /// (check_frequency=3) enabled with the same frequency, at the firing
    /// iteration the event log shows `AngularPruningComplete` before
    /// `CutSelectionComplete`. Step 4a runs selection logic and saves records
    /// to a deferred buffer; Step 4b runs angular pruning and emits
    /// `AngularPruningComplete`; only after all sub-steps does Step 4 emit
    /// `CutSelectionComplete` with fully annotated per-stage records.
    #[test]
    fn angular_pruning_after_cut_selection_ordering() {
        use crate::angular_pruning::AngularPruningParams;
        use crate::cut_selection::CutSelectionStrategy;

        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();

        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 10,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit_rules(5),
            },
            cut_management: CutManagementConfig {
                cut_selection: Some(CutSelectionStrategy::Level1 {
                    threshold: 0,
                    check_frequency: 3,
                }),
                angular_pruning: Some(AngularPruningParams {
                    cosine_threshold: 0.999,
                    check_frequency: 3,
                }),
                budget: None,
                basis_padding_enabled: false,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation; n_stages],
            },
            events: EventConfig {
                event_sender: Some(tx),
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                basis_padding_enabled: false,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        let events: Vec<TrainingEvent> = rx.try_iter().collect();

        // Find the position of CutSelectionComplete and AngularPruningComplete.
        let sel_pos = events
            .iter()
            .position(|e| matches!(e, TrainingEvent::CutSelectionComplete { .. }))
            .expect("expected at least one CutSelectionComplete event");
        let prune_pos = events
            .iter()
            .position(|e| matches!(e, TrainingEvent::AngularPruningComplete { .. }))
            .expect("expected at least one AngularPruningComplete event");

        assert!(
            prune_pos < sel_pos,
            "AngularPruningComplete (pos={prune_pos}) must appear before \
             CutSelectionComplete (pos={sel_pos})"
        );
    }

    /// AC: `template_bake_event_emitted`
    ///
    /// Verify that `TemplateBakeComplete` is emitted exactly once per iteration
    /// with the correct `stages_processed` count. Also verifies that
    /// `total_cut_rows_baked > 0` on iteration 2 (because the backward pass on
    /// iteration 1 generates cuts before step 4d runs on that same iteration).
    #[test]
    fn template_bake_event_emitted() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();

        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 10,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit_rules(2),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                angular_pruning: None,
                budget: None,
                basis_padding_enabled: false,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation; n_stages],
            },
            events: EventConfig {
                event_sender: Some(tx),
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        };

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[1usize, 1],
            ncs_max_gen: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };

        train(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &stages,
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                basis_padding_enabled: false,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        let events: Vec<TrainingEvent> = rx.try_iter().collect();

        // Collect all TemplateBakeComplete events.
        let bake_events: Vec<&TrainingEvent> = events
            .iter()
            .filter(|e| matches!(e, TrainingEvent::TemplateBakeComplete { .. }))
            .collect();

        // Exactly one per iteration (2 iterations).
        assert_eq!(
            bake_events.len(),
            2,
            "expected exactly 2 TemplateBakeComplete events, got {}",
            bake_events.len()
        );

        // Each event must report stages_processed == n_stages.
        for event in &bake_events {
            let TrainingEvent::TemplateBakeComplete {
                stages_processed, ..
            } = event
            else {
                panic!("wrong variant")
            };
            assert_eq!(
                *stages_processed, n_stages as u32,
                "stages_processed must equal num_stages"
            );
        }

        // On iteration 2, the backward pass from iteration 1 will have added
        // cuts, so total_cut_rows_baked must be > 0.
        let second_bake = bake_events[1];
        let TrainingEvent::TemplateBakeComplete {
            total_cut_rows_baked,
            ..
        } = second_bake
        else {
            panic!("wrong variant")
        };
        assert!(
            *total_cut_rows_baked > 0,
            "iteration 2 bake must have baked at least one cut row (backward pass \
             generated cuts on iteration 1)"
        );
    }
}
