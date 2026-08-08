//! Training phase for `cobre run`.

use std::sync::mpsc;

use cobre_comm::{Communicator, ReduceOp};
use cobre_core::TrainingEvent;
use cobre_io::MetadataTrainingSolveStats;
use cobre_io::TrainingOutput;
use cobre_sddp::SddpError;
use cobre_sddp::SolverStatsDelta;
use cobre_sddp::StudySetup;
use cobre_sddp::TrainingResult;
use cobre_sddp::sum_phase_timing_ms;
use cobre_solver::ActiveSolver;

use crate::error::CliError;
use crate::summary::TrainingSummary;

use super::{RunContext, check_stats_overflow};
use crate::progress::run_progress_thread;
use crate::summary::print_training_summary;

/// Output of [`run_training_phase`]: result, training output, and optional error.
pub(super) struct TrainingPhaseResult {
    pub(super) result: TrainingResult,
    pub(super) output: TrainingOutput,
    /// Mid-iteration training error, if any. Partial results are still valid.
    pub(super) error: Option<SddpError>,
}

/// Single owner of the training-run stats the printed [`TrainingSummary`] and
/// the persisted [`cobre_io::MetadataTrainingSolveStats`] both read from, so the
/// two stay identical. `first_try`/`retried`/`failed`/`*_solve_seconds` are
/// cross-rank allreduce sums; the phase-wall/wait/serial `*_ms` fields are a
/// rank-0-local sum over this rank's own convergence records (never allreduced
/// — printing is gated to root, so no other rank's copy is ever observed).
struct GlobalTrainingStats {
    first_try: u64,
    retried: u64,
    failed: u64,
    forward_solve_seconds: f64,
    backward_solve_seconds: f64,
    forward_phase_wall_ms: u64,
    backward_phase_wall_ms: u64,
    forward_wait_ms: u64,
    backward_wait_ms: u64,
    serial_lower_bound_ms: u64,
    serial_cut_selection_ms: u64,
    serial_cut_sync_ms: u64,
    serial_allreduce_ms: u64,
    serial_scheduling_ms: u64,
}

/// Run training and collect results, events, and summary stats.
// Rationale: an inherently sequential orchestration whose steps each depend on
// the prior's result; splitting would thread shared mutable state across
// boundaries for no benefit.
#[allow(clippy::too_many_lines)]
pub(super) fn run_training_phase(
    ctx: &RunContext<impl Communicator>,
    setup: &mut StudySetup,
) -> Result<TrainingPhaseResult, CliError> {
    let solver_factory = ActiveSolver::new;

    let mut solver = ActiveSolver::new().map_err(|e| CliError::Solver {
        message: format!(
            "{} initialisation failed: {e}",
            cobre_solver::active_solver_name()
        ),
    })?;

    let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();

    let quiet_rx: Option<mpsc::Receiver<TrainingEvent>>;
    let progress_handle = if ctx.quiet {
        quiet_rx = Some(event_rx);
        None
    } else {
        quiet_rx = None;
        Some(run_progress_thread(
            event_rx,
            ctx.render_mode,
            setup.loop_params.max_iterations,
            ctx.term_width,
        ))
    };

    let training_outcome = match setup.train(
        &mut solver,
        &ctx.comm,
        ctx.n_threads,
        solver_factory,
        Some(event_tx),
        None,
    ) {
        Ok(outcome) => outcome,
        Err(e) => {
            if let Some(handle) = progress_handle {
                let _ = handle.join();
            }
            return Err(CliError::from(e));
        }
    };
    let training_result = training_outcome.result;

    let events: Vec<TrainingEvent> = match (progress_handle, quiet_rx) {
        (Some(handle), _) => handle.join(),
        (None, Some(rx)) => rx.try_iter().collect(),
        (None, None) => Vec::new(),
    };
    let mut training_output = setup.build_training_output(&training_result, &events);

    let local_lp_solves: u64 = training_output
        .convergence_records
        .iter()
        .map(|r| u64::from(r.lp_solves))
        .sum();
    let mut global_lp_solves = [0u64];
    ctx.comm
        .allreduce(&[local_lp_solves], &mut global_lp_solves, ReduceOp::Sum)
        .map_err(|e| CliError::Internal {
            message: format!("LP solve count allreduce error: {e}"),
        })?;
    let global_lp_solves = global_lp_solves[0];

    ctx.comm.barrier().map_err(|e| CliError::Internal {
        message: format!("post-training barrier error: {e}"),
    })?;

    // Backward entries are allgatherv-replicated across every rank, so filter to
    // this rank's own contribution before allreduce(Sum); summing the replicated
    // entries would multiply backward stats by world_size.
    let my_rank = i32::try_from(ctx.comm.rank()).unwrap_or(i32::MAX);
    let (
        local_first_try,
        local_retried,
        local_failed,
        local_forward_solve_s,
        local_backward_solve_s,
    ) = aggregate_solver_stats(&training_result.solver_stats_log, my_rank);

    let training_guard_delta = SolverStatsDelta {
        lp_successes: local_first_try.saturating_add(local_retried),
        first_try_successes: local_first_try,
        lp_failures: local_failed,
        ..SolverStatsDelta::default()
    };
    check_stats_overflow(&training_guard_delta)?;

    #[allow(clippy::cast_precision_loss)]
    let send_stats = [
        local_first_try as f64,
        local_retried as f64,
        local_failed as f64,
        local_forward_solve_s,
        local_backward_solve_s,
    ];
    let mut recv_stats = [0.0_f64; 5];
    ctx.comm
        .allreduce(&send_stats, &mut recv_stats, ReduceOp::Sum)
        .map_err(|e| CliError::Internal {
            message: format!("training solver stats allreduce error: {e}"),
        })?;
    let phase_timing = sum_phase_timing_ms(&training_output.convergence_records);

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let global_stats = GlobalTrainingStats {
        first_try: recv_stats[0] as u64,
        retried: recv_stats[1] as u64,
        failed: recv_stats[2] as u64,
        forward_solve_seconds: recv_stats[3],
        backward_solve_seconds: recv_stats[4],
        forward_phase_wall_ms: phase_timing.forward_wall_ms,
        backward_phase_wall_ms: phase_timing.backward_wall_ms,
        forward_wait_ms: phase_timing.forward_wait_ms,
        backward_wait_ms: phase_timing.backward_wait_ms,
        serial_lower_bound_ms: phase_timing.lower_bound_ms,
        serial_cut_selection_ms: phase_timing.cut_selection_ms,
        serial_cut_sync_ms: phase_timing.cut_sync_ms,
        serial_allreduce_ms: phase_timing.allreduce_ms,
        serial_scheduling_ms: phase_timing.scheduling_ms,
    };

    // The convergence records are not yet persisted at this point, so the
    // in-memory copy is authoritative; reading from disk would return None.
    let initial_gap_percent = training_output
        .convergence_records
        .first()
        .and_then(|r| r.gap_percent);

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let parallelism = (ctx.n_threads as u32).saturating_mul(ctx.comm.size() as u32);

    #[allow(clippy::cast_precision_loss)]
    let training_summary = TrainingSummary {
        iterations: training_result.iterations,
        converged: training_output.converged,
        converged_at: if training_output.converged {
            Some(training_result.iterations)
        } else {
            None
        },
        reason: training_result.reason.clone(),
        lower_bound: training_result.final_lb,
        upper_bound: training_result.final_ub,
        upper_bound_std: training_result.final_ub_std,
        gap_percent: training_result.final_gap * 100.0,
        total_rows_active: training_output.cut_stats.total_active,
        total_rows_generated: training_output.cut_stats.total_generated,
        rows_in_lp_total: training_output.cut_stats.rows_in_lp_total,
        rows_in_lp_solve_count: training_output.cut_stats.rows_in_lp_solve_count,
        rows_in_lp_max: training_output.cut_stats.rows_in_lp_max,
        num_stages: u32::try_from(setup.num_stages()).unwrap_or(u32::MAX),
        total_lp_solves: global_lp_solves,
        total_time_ms: training_result.total_time_ms,
        total_first_try: Some(global_stats.first_try),
        total_retried: Some(global_stats.retried),
        total_failed: Some(global_stats.failed),
        total_forward_solve_seconds: Some(global_stats.forward_solve_seconds),
        total_backward_solve_seconds: Some(global_stats.backward_solve_seconds),
        parallelism: Some(parallelism),
        initial_gap_percent,
        forward_phase_wall_seconds: Some(global_stats.forward_phase_wall_ms as f64 / 1000.0),
        backward_phase_wall_seconds: Some(global_stats.backward_phase_wall_ms as f64 / 1000.0),
        forward_wait_seconds: Some(global_stats.forward_wait_ms as f64 / 1000.0),
        backward_wait_seconds: Some(global_stats.backward_wait_ms as f64 / 1000.0),
        serial_lower_bound_seconds: Some(global_stats.serial_lower_bound_ms as f64 / 1000.0),
        serial_cut_selection_seconds: Some(global_stats.serial_cut_selection_ms as f64 / 1000.0),
        serial_cut_sync_seconds: Some(global_stats.serial_cut_sync_ms as f64 / 1000.0),
        serial_allreduce_seconds: Some(global_stats.serial_allreduce_ms as f64 / 1000.0),
        serial_scheduling_seconds: Some(global_stats.serial_scheduling_ms as f64 / 1000.0),
    };
    if !ctx.quiet && ctx.is_root {
        print_training_summary(&ctx.stderr, &training_summary);
    }

    // Assign only this field; `final_upper_bound_std` is populated upstream and
    // must stay untouched.
    #[allow(clippy::cast_precision_loss)]
    let persisted_stats = MetadataTrainingSolveStats {
        total_lp_solves: Some(global_lp_solves),
        first_try: Some(global_stats.first_try),
        retried: Some(global_stats.retried),
        failed: Some(global_stats.failed),
        forward_solve_seconds: Some(global_stats.forward_solve_seconds),
        backward_solve_seconds: Some(global_stats.backward_solve_seconds),
        parallelism: Some(parallelism),
        forward_phase_wall_seconds: Some(global_stats.forward_phase_wall_ms as f64 / 1000.0),
        backward_phase_wall_seconds: Some(global_stats.backward_phase_wall_ms as f64 / 1000.0),
        forward_wait_seconds: Some(global_stats.forward_wait_ms as f64 / 1000.0),
        backward_wait_seconds: Some(global_stats.backward_wait_ms as f64 / 1000.0),
        serial_lower_bound_seconds: Some(global_stats.serial_lower_bound_ms as f64 / 1000.0),
        serial_row_selection_seconds: Some(global_stats.serial_cut_selection_ms as f64 / 1000.0),
        serial_row_sync_seconds: Some(global_stats.serial_cut_sync_ms as f64 / 1000.0),
        serial_allreduce_seconds: Some(global_stats.serial_allreduce_ms as f64 / 1000.0),
        serial_scheduling_seconds: Some(global_stats.serial_scheduling_ms as f64 / 1000.0),
    };
    training_output.training_solve_stats = persisted_stats;

    Ok(TrainingPhaseResult {
        result: training_result,
        output: training_output,
        error: training_outcome.error,
    })
}

/// Aggregate this rank's own contribution from the training stats log.
///
/// Backward entries are allgatherv-replicated across ranks, so filtering by
/// originating rank is required for the per-rank totals that `allreduce(Sum)`
/// then sums correctly.
fn aggregate_solver_stats(
    stats_log: &[cobre_sddp::SolverStatsLogEntry],
    my_rank: i32,
) -> (u64, u64, u64, f64, f64) {
    let mut first_try = 0u64;
    let mut retried = 0u64;
    let mut failed = 0u64;
    let mut forward_solve_ms = 0.0_f64;
    let mut backward_solve_ms = 0.0_f64;
    for entry in stats_log {
        if entry.rank != my_rank {
            continue;
        }
        let delta = &entry.delta;
        // lower_bound solve counts are included, but its solve_time_ms is
        // deliberately dropped by the `_ => {}` arm: no output field receives it.
        first_try += delta.first_try_successes;
        retried += delta.lp_successes.saturating_sub(delta.first_try_successes);
        failed += delta.lp_failures;
        match entry.phase {
            "forward" => forward_solve_ms += delta.solve_time_ms,
            "backward" => backward_solve_ms += delta.solve_time_ms,
            _ => {}
        }
    }
    (
        first_try,
        retried,
        failed,
        forward_solve_ms / 1000.0,
        backward_solve_ms / 1000.0,
    )
}
