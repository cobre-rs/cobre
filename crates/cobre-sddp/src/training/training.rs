//! Training loop orchestrator for the SDDP algorithm.
//!
//! [`train`] wires the forward pass, forward sync, state exchange, backward
//! pass, cut sync, lower-bound evaluation, and convergence check into a single
//! iterative loop. The per-iteration ordering is enforced by
//! `TrainingSession::run_iteration`.
//!
//! All workspace buffers are allocated once before the loop and reused: no heap
//! allocation occurs on the hot path.

use cobre_comm::CommError::InvalidBufferSize;
use cobre_comm::Communicator;
use cobre_solver::ActiveProfile;
use cobre_solver::SolverError;
use cobre_solver::{SolverInterface, StageTemplate};

use crate::visited_states::VisitedStatesArchive;
use crate::workspace::BasisStore;
use crate::{
    SddpError, SolverProfiles, TrainingConfig,
    context::{StageContext, TrainingContext},
    cut::fcf::FutureCostFunction,
    setup::NodePos,
    solver_stats::SolverStatsLogEntry,
    stopping_rule::StopDecision,
    training_session::{IterationOutcome, TrainingSession},
    workspace::CapturedBasis,
};

// ---------------------------------------------------------------------------
// TrainingResult
// ---------------------------------------------------------------------------

/// Result of a training run that always carries partial results.
///
/// On normal completion `error` is `None` and `result` holds the full
/// statistics. On mid-iteration failure `error` carries the cause and `result`
/// holds statistics from the fully completed iterations (the failing one
/// excluded).
#[derive(Debug)]
pub struct TrainingOutcome {
    /// Training result from the completed iterations; always populated.
    pub result: TrainingResult,

    /// If training was interrupted by an error, the cause. `None` when
    /// training completed normally (convergence, iteration limit, or
    /// time limit).
    pub error: Option<SddpError>,
}

/// Summary statistics produced when the training loop terminates.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Final lower bound at termination.
    pub final_lb: f64,

    /// The lower bound of each completed iteration the run or its resumed
    /// checkpoint recorded, oldest first; empty when the result was not
    /// produced by a training loop.
    pub lower_bound_history: Vec<f64>,

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

    /// The stop decision of the iteration where training stopped; the default
    /// (nothing triggered) when training ended otherwise or the result was not
    /// produced by a training loop.
    pub stop_decision: StopDecision,

    /// Total wall-clock time for the training run, in milliseconds.
    pub total_time_ms: u64,

    /// Per-stage captured basis from the last iteration, 0-based. `None` when
    /// the stage was never solved in the final iteration (e.g. the last
    /// finite-horizon stage has no successor cuts).
    pub basis_cache: Vec<Option<CapturedBasis>>,

    /// Per-iteration, per-phase solver statistics log. Stage index is `-1` for
    /// `"lower_bound"` entries (rank-aggregated, no stage axis).
    pub solver_stats_log: Vec<SolverStatsLogEntry>,

    /// Visited-states archive of all forward-pass trial points; the caller
    /// decides whether to persist it based on `exports.states`.
    pub visited_archive: Option<VisitedStatesArchive>,

    /// Final-iteration frozen templates, one per POOL (each is a pool's base
    /// stage template plus that pool's active cuts). Always `Some`: freeze runs
    /// unconditionally before the first iteration and is never reverted.
    pub frozen_templates: Option<Vec<StageTemplate>>,
}

impl TrainingResult {
    /// Construct a `TrainingResult` with explicit field values.
    ///
    /// Prefer this over struct-literal syntax: adding a field then fails to
    /// compile until every call site is updated.
    #[must_use]
    // Rationale: the arguments are independently sourced phase outputs and every
    // call site constructs the full result, so a context struct would not reduce
    // the arity.
    #[allow(clippy::too_many_arguments, clippy::similar_names)]
    pub fn new(
        final_lb: f64,
        final_ub: f64,
        final_ub_std: f64,
        final_gap: f64,
        iterations: u64,
        reason: String,
        total_time_ms: u64,
        basis_cache: Vec<Option<CapturedBasis>>,
        solver_stats_log: Vec<SolverStatsLogEntry>,
        visited_archive: Option<VisitedStatesArchive>,
        frozen_templates: Option<Vec<StageTemplate>>,
    ) -> Self {
        Self {
            final_lb,
            lower_bound_history: Vec::new(),
            final_ub,
            final_ub_std,
            final_gap,
            iterations,
            reason,
            stop_decision: StopDecision::default(),
            total_time_ms,
            basis_cache,
            solver_stats_log,
            visited_archive,
            frozen_templates,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert a buffer length to `i32` for use as an MPI broadcast count.
///
/// # Errors
///
/// Returns `SddpError::Communication(CommError::InvalidBufferSize { .. })`
/// when `len > i32::MAX` (the MPI count limit).
fn checked_broadcast_len(len: usize, operation: &'static str) -> Result<i32, SddpError> {
    i32::try_from(len).map_err(|_| {
        SddpError::Communication(InvalidBufferSize {
            operation,
            expected: i32::MAX as usize,
            actual: len,
        })
    })
}

/// This rank's local scenario-0 basis at every node, metadata included
/// (`cut_row_slots`, `state_at_capture`, `base_row_count`), so a reconstruction
/// from it keeps full slot identity. On rank 0 it is what
/// [`broadcast_basis_cache`] delivers to every rank.
pub(crate) fn rank_local_basis_cache(basis_store: &BasisStore) -> Vec<Option<CapturedBasis>> {
    (0..basis_store.num_nodes())
        .map(|t| basis_store.get(0, NodePos(t)).cloned())
        .collect()
}

/// Build a `basis_cache` from global scenario 0, broadcasting rank 0's bases to
/// all ranks so every rank starts simulation from an identical warm-start vertex.
///
/// Scenario 0 (not the last local scenario) is broadcast because `basis_store`
/// is local-indexed: each rank's last local scenario follows a distinct noise
/// realisation and would yield divergent bases, whereas rank 0's local
/// scenario 0 is always global scenario 0 (`my_fwd_offset == 0`).
///
/// The two-buffer wire layout is owned by
/// [`CapturedBasis::to_broadcast_payload`](crate::workspace::CapturedBasis::to_broadcast_payload)
/// /
/// [`try_from_broadcast_payload`](crate::workspace::CapturedBasis::try_from_broadcast_payload);
/// this function only sequences the four MPI broadcasts (length then payload,
/// for the i32 and f64 buffers). Single-rank runs skip the broadcast and return
/// [`rank_local_basis_cache`].
///
/// Each reconstructed basis's `node_id` now rides the wire payload
/// (`try_from_broadcast_payload` reads it), so no out-of-band fill from a
/// per-stage node array is needed. The single-rank clone path likewise keeps
/// the `node_id` capture set via `write_capture_metadata`. The cache is sized
/// and keyed by `basis_store.num_nodes()` (one slot per canonical node), so a
/// branching graph's leaf nodes are broadcast too, never truncated at
/// `num_stages`.
///
/// # Errors
///
/// Returns `SddpError::Communication(CommError::InvalidBufferSize { .. })` when
/// a buffer length exceeds `i32::MAX`, `SddpError::Communication` when a
/// `comm.broadcast` fails, or `SddpError::Validation` when a length prefix is
/// inconsistent with the received buffer (naming the offending node).
pub(crate) fn broadcast_basis_cache<C: Communicator>(
    basis_store: &BasisStore,
    comm: &C,
) -> Result<Vec<Option<CapturedBasis>>, SddpError> {
    let num_nodes = basis_store.num_nodes();
    if comm.size() == 1 {
        return Ok(rank_local_basis_cache(basis_store));
    }

    // Multi-rank path: pack rank 0's scenario-0 full CapturedBasis into two
    // flat buffers (one i32 for integer/status fields, one f64 for
    // state_at_capture) and broadcast both.
    //
    // Wire format is owned by CapturedBasis::to_broadcast_payload /
    // CapturedBasis::try_from_broadcast_payload. The None case writes a
    // 0_i32 sentinel directly; the Some case delegates to the method.
    let mut buf: Vec<i32> = Vec::new();
    let mut f64_buf: Vec<f64> = Vec::new();
    if comm.rank() == 0 {
        for t in 0..num_nodes {
            match basis_store.get(0, NodePos(t)) {
                None => buf.push(0_i32),
                Some(captured) => captured.to_broadcast_payload(&mut buf, &mut f64_buf),
            }
        }
    }

    let mut len_buf = [checked_broadcast_len(
        buf.len(),
        "broadcast_basis_cache_i32",
    )?];
    comm.broadcast(&mut len_buf, 0).map_err(SddpError::from)?;
    let total_len = usize::try_from(len_buf[0]).map_err(|_| {
        SddpError::Validation(format!(
            "broadcast_basis_cache_i32: received negative length {}",
            len_buf[0]
        ))
    })?;

    buf.resize(total_len, 0_i32);
    comm.broadcast(&mut buf, 0).map_err(SddpError::from)?;

    let mut f64_len_buf = [checked_broadcast_len(
        f64_buf.len(),
        "broadcast_basis_cache_f64",
    )?];
    comm.broadcast(&mut f64_len_buf, 0)
        .map_err(SddpError::from)?;
    let f64_total_len = usize::try_from(f64_len_buf[0]).map_err(|_| {
        SddpError::Validation(format!(
            "broadcast_basis_cache_f64: received negative length {}",
            f64_len_buf[0]
        ))
    })?;

    f64_buf.resize(f64_total_len, 0.0_f64);
    comm.broadcast(&mut f64_buf, 0).map_err(SddpError::from)?;

    let mut cache: Vec<Option<CapturedBasis>> = Vec::with_capacity(num_nodes);
    let mut pos = 0_usize;
    let mut f64_pos = 0_usize;
    for node in 0..num_nodes {
        let captured = CapturedBasis::try_from_broadcast_payload(
            node,
            &buf,
            &mut pos,
            &f64_buf,
            &mut f64_pos,
        )?;
        cache.push(captured);
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
/// ## Event channel
///
/// When `config.event_sender` is `Some`, typed [`cobre_core::TrainingEvent`]
/// values are emitted at each lifecycle boundary. Send failures (receiver
/// dropped) are silently ignored so they cannot interrupt training.
///
/// ## Cut selection
///
/// When `cut_selection` is `Some(strategy)`, its `should_run(iteration)` gate
/// controls how often each stage's pool is scanned for inactive cuts. When
/// `None`, no [`cobre_core::TrainingEvent::PolicySelectionComplete`] events are
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
/// use cobre_sddp::{train, TrainingConfig, LoopConfig, CutManagementConfig, EventConfig, SolverProfiles};
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
///     || HiggsBackend::new(), None, SolverProfiles::default(),
/// )?;
///
/// println!("converged in {} iterations, gap={:.4}", result.result.iterations, result.result.final_gap);
/// ```
///
/// # Panics
///
/// In debug builds, panics if `templates.len() != horizon.num_stages()` or if
/// `config.cut_management.risk_measures.len() != horizon.num_stages()` or if
/// `training_ctx.stochastic.opening_tree().n_openings(0) == 0`.
///
/// Always panics if `comm.rank() > i32::MAX`. MPI world sizes are bounded well
/// below this on all real systems.
pub fn train<S, C: Communicator>(
    solver: &mut S,
    config: TrainingConfig,
    fcf: &mut FutureCostFunction,
    stage_ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    comm: &C,
    solver_factory: impl Fn() -> Result<S, SolverError>,
    warm_start_basis_cache: Option<Vec<Option<CapturedBasis>>>,
    solver_profiles: SolverProfiles,
) -> Result<TrainingOutcome, SddpError>
where
    S: SolverInterface<Profile = ActiveProfile> + Send,
{
    let mut session = TrainingSession::new(
        solver,
        config,
        fcf,
        stage_ctx,
        training_ctx,
        comm,
        solver_factory,
        solver_profiles,
    )?;
    // Must seed the basis store before `prime_frozen_templates` freezes the loaded
    // cuts into the templates.
    if let Some(cache) = warm_start_basis_cache {
        session.seed_basis_store(&cache);
    }
    session.prime_frozen_templates();
    for iteration in session.iteration_range() {
        match session.run_iteration(iteration) {
            Ok(IterationOutcome::Continue) => {}
            Ok(IterationOutcome::Converged | IterationOutcome::Shutdown) => break,
            Err(e) => return Ok(session.finalize_with_error(e)),
        }
    }
    session.finalize()
}

#[cfg(test)]
mod tests;
