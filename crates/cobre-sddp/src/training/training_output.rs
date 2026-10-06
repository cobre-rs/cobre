//! Bridge from `TrainingResult` and `TrainingEvent` log to `TrainingOutput`.
//!
//! [`build_training_output`] reads the post-training [`TrainingEvent`] log and
//! reconstructs the per-iteration records [`cobre_io::TrainingOutput`] needs,
//! rather than modifying the hot-path `train()` function. Missing events for an
//! iteration produce zero values for the affected fields.

use std::collections::BTreeMap;

use cobre_core::TrainingEvent;
use cobre_io::{
    IterationRecord, MetadataTrainingSolveStats, RowPoolStatistics, RowSelectionRecord, RunStatus,
    TrainingOutput,
};

use crate::stopping_rule::StopMask;
use crate::{FutureCostFunction, TrainingResult};

/// Per-iteration accumulator filled by [`accumulate_partial_records`] from
/// multiple [`TrainingEvent`] variants, then converted to one [`IterationRecord`].
/// Time fields are milliseconds.
#[derive(Default)]
struct PartialRecord {
    lower_bound: f64,
    upper_bound_mean: f64,
    upper_bound_std: f64,
    gap: f64,
    forward_ms: u64,
    backward_ms: u64,
    iteration_time_ms: u64,
    lp_solves: u64,
    forward_passes: u32,
    cuts_added: u32,
    cuts_removed: u32,
    cuts_active: u32,
    /// Scalar bound-statistic `allreduce` time — not the `allgatherv` scenario exchange.
    forward_sync_ms: u64,
    /// Policy sync allgatherv time.
    cut_sync_ms: u64,
    /// Local row selection time.
    cut_selection_ms: u64,
    cut_selection_allgatherv_ms: u64,
    /// Cumulative LP solve wall-clock time for this iteration.
    solve_time_ms: f64,
    state_exchange_ms: u64,
    cut_batch_build_ms: u64,
    bwd_load_imbalance_ms: u64,
    bwd_scheduling_overhead_ms: u64,
    lower_bound_eval_ms: u64,
    fwd_load_imbalance_ms: u64,
    fwd_scheduling_overhead_ms: u64,
    /// Sum of resident rows-in-LP over this iteration's lazy solves (all ranks). Zero for non-lazy methods.
    rows_in_lp_sum: u64,
    /// Count of this iteration's lazy solves (all ranks) — the per-iteration mean denominator.
    rows_in_lp_count: u64,
    /// Running peak resident rows-in-LP through this iteration (all ranks); `max`-folds to the run-level peak. Zero for non-lazy methods.
    rows_in_lp_max: u64,
}

/// Accumulate per-iteration partial records from the event log, returning
/// `(partials keyed by iteration, peak active cut count)`.
fn accumulate_partial_records(events: &[TrainingEvent]) -> (BTreeMap<u64, PartialRecord>, u64) {
    let mut partials: BTreeMap<u64, PartialRecord> = BTreeMap::new();
    let mut peak_active: u64 = 0;

    for event in events {
        match event {
            TrainingEvent::IterationSummary {
                iteration,
                lower_bound,
                upper_bound,
                gap,
                iteration_time_ms,
                forward_ms,
                backward_ms,
                lp_solves,
                solve_time_ms,
                lower_bound_eval_ms,
                fwd_load_imbalance_ms,
                fwd_scheduling_overhead_ms,
                rows_in_lp_sum,
                rows_in_lp_count,
                rows_in_lp_max,
                ..
            } => {
                let record = partials.entry(*iteration).or_default();
                record.lower_bound = *lower_bound;
                record.upper_bound_mean = *upper_bound;
                record.gap = *gap;
                record.iteration_time_ms = *iteration_time_ms;
                record.forward_ms = *forward_ms;
                record.backward_ms = *backward_ms;
                record.lp_solves = *lp_solves;
                record.solve_time_ms = *solve_time_ms;
                record.lower_bound_eval_ms = *lower_bound_eval_ms;
                record.fwd_load_imbalance_ms = *fwd_load_imbalance_ms;
                record.fwd_scheduling_overhead_ms = *fwd_scheduling_overhead_ms;
                record.rows_in_lp_sum = *rows_in_lp_sum;
                record.rows_in_lp_count = *rows_in_lp_count;
                record.rows_in_lp_max = *rows_in_lp_max;
            }

            TrainingEvent::ForwardSyncComplete {
                iteration,
                global_ub_std,
                sync_time_ms,
                ..
            } => {
                let record = partials.entry(*iteration).or_default();
                record.upper_bound_std = *global_ub_std;
                record.forward_sync_ms = *sync_time_ms;
            }

            TrainingEvent::ForwardPassComplete {
                iteration,
                scenarios,
                ..
            } => {
                let record = partials.entry(*iteration).or_default();
                record.forward_passes = *scenarios;
            }

            TrainingEvent::BackwardPassComplete {
                iteration,
                rows_generated,
                state_exchange_time_ms,
                row_batch_build_time_ms,
                load_imbalance_ms,
                scheduling_overhead_ms,
                ..
            } => {
                let record = partials.entry(*iteration).or_default();
                record.cuts_added = *rows_generated;
                record.state_exchange_ms = *state_exchange_time_ms;
                record.cut_batch_build_ms = *row_batch_build_time_ms;
                record.bwd_load_imbalance_ms = *load_imbalance_ms;
                record.bwd_scheduling_overhead_ms = *scheduling_overhead_ms;
            }

            TrainingEvent::PolicySyncComplete {
                iteration,
                rows_active,
                rows_removed,
                sync_time_ms,
                ..
            } => {
                let record = partials.entry(*iteration).or_default();
                record.cuts_active = *rows_active;
                record.cuts_removed = *rows_removed;
                record.cut_sync_ms = *sync_time_ms;
                peak_active = peak_active.max(u64::from(*rows_active));
            }

            TrainingEvent::PolicySelectionComplete {
                iteration,
                rows_deactivated,
                selection_time_ms,
                allgatherv_time_ms,
                ..
            } => {
                let record = partials.entry(*iteration).or_default();
                record.cut_selection_ms = *selection_time_ms;
                record.cut_selection_allgatherv_ms = *allgatherv_time_ms;
                record.cuts_active = record.cuts_active.saturating_sub(*rows_deactivated);
            }

            _ => {}
        }
    }

    (partials, peak_active)
}

/// Convert a single [`PartialRecord`] into an [`IterationRecord`].
fn partial_to_iteration_record(iter: u64, partial: &PartialRecord) -> IterationRecord {
    let gap_percent = if partial.lower_bound > 0.0 {
        Some(partial.gap * 100.0)
    } else {
        None
    };

    #[allow(clippy::cast_possible_truncation)]
    let iteration_u32 = iter as u32;
    #[allow(clippy::cast_possible_truncation)]
    let lp_solves_u32 = partial.lp_solves as u32;

    // Sum only TOP-LEVEL non-overlapping phases: cut_sync_ms is a sub-component
    // of backward_ms and must NOT be added here (would double-count).
    let attributed_ms = partial
        .forward_ms
        .saturating_add(partial.backward_ms)
        .saturating_add(partial.cut_selection_ms)
        .saturating_add(partial.cut_selection_allgatherv_ms)
        .saturating_add(partial.forward_sync_ms)
        .saturating_add(partial.lower_bound_eval_ms);
    let overhead_ms = partial.iteration_time_ms.saturating_sub(attributed_ms);

    #[allow(clippy::cast_precision_loss)]
    let mean_rows_in_lp = if partial.rows_in_lp_count > 0 {
        partial.rows_in_lp_sum as f64 / partial.rows_in_lp_count as f64
    } else {
        0.0
    };

    IterationRecord {
        iteration: iteration_u32,
        lower_bound: partial.lower_bound,
        upper_bound: partial.upper_bound_mean,
        upper_bound_std: partial.upper_bound_std,
        gap_percent,
        cuts_added: partial.cuts_added,
        cuts_removed: partial.cuts_removed,
        cuts_active: partial.cuts_active,
        time_forward_ms: partial.forward_ms,
        time_backward_ms: partial.backward_ms,
        time_total_ms: partial.iteration_time_ms,
        forward_passes: partial.forward_passes,
        lp_solves: lp_solves_u32,
        time_forward_wall_ms: partial.forward_ms,
        time_backward_wall_ms: partial.backward_ms,
        time_cut_selection_ms: partial.cut_selection_ms,
        time_mpi_allreduce_ms: partial.forward_sync_ms,
        time_cut_sync_ms: partial.cut_sync_ms,
        time_lower_bound_ms: partial.lower_bound_eval_ms,
        time_state_exchange_ms: partial.state_exchange_ms,
        time_cut_batch_build_ms: partial.cut_batch_build_ms,
        time_bwd_load_imbalance_ms: partial.bwd_load_imbalance_ms,
        time_bwd_scheduling_overhead_ms: partial.bwd_scheduling_overhead_ms,
        time_fwd_load_imbalance_ms: partial.fwd_load_imbalance_ms,
        time_fwd_scheduling_overhead_ms: partial.fwd_scheduling_overhead_ms,
        time_overhead_ms: overhead_ms,
        solve_time_ms: partial.solve_time_ms,
        mean_rows_in_lp,
    }
}

/// Convert a [`TrainingResult`] and collected event log into a [`TrainingOutput`],
/// one [`IterationRecord`] per completed iteration.
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::{build_training_output, TrainingResult, FutureCostFunction};
/// use cobre_core::TrainingEvent;
///
/// let result = TrainingResult::new(
///     100.0,
///     110.0,
///     5.0,
///     0.091,
///     1,
///     "iteration_limit".to_string(),
///     500,
///     Vec::new(),
///     Vec::new(),
///     None,
///     None,
/// );
///
/// let events = vec![TrainingEvent::IterationSummary {
///     iteration: 1,
///     lower_bound: 100.0,
///     upper_bound: 110.0,
///     gap: 0.091,
///     wall_time_ms: 500,
///     iteration_time_ms: 500,
///     forward_ms: 200,
///     backward_ms: 250,
///     lp_solves: 60,
///     solve_time_ms: 0.0,
///     lower_bound_eval_ms: 0,
///     fwd_setup_time_ms: 0,
///     fwd_load_imbalance_ms: 0,
///     fwd_scheduling_overhead_ms: 0,
///     rows_in_lp_sum: 0,
///     rows_in_lp_count: 0,
///     rows_in_lp_max: 0,
/// }];
///
/// let fcf = FutureCostFunction::new(2, 1, 4, 1, &[0; 2]);
/// let output = build_training_output(&result, &events, &fcf, false);
///
/// assert_eq!(output.convergence_records.len(), 1);
/// assert!(!output.converged);
/// ```
#[must_use]
pub fn build_training_output(
    result: &TrainingResult,
    events: &[TrainingEvent],
    fcf: &FutureCostFunction,
    upper_bound_exact: bool,
) -> TrainingOutput {
    let (partials, peak_active) = accumulate_partial_records(events);

    // Drop partial-only iterations: a record is emitted only when an
    // IterationSummary event exists for that iteration.
    let summary_iterations: std::collections::BTreeSet<u64> = events
        .iter()
        .filter_map(|e| {
            if let TrainingEvent::IterationSummary { iteration, .. } = e {
                Some(*iteration)
            } else {
                None
            }
        })
        .collect();

    let (rows_in_lp_total, rows_in_lp_solve_count, rows_in_lp_max) =
        partials.values().fold((0u64, 0u64, 0u64), |(s, c, m), p| {
            (
                s + p.rows_in_lp_sum,
                c + p.rows_in_lp_count,
                m.max(p.rows_in_lp_max),
            )
        });

    let convergence_records: Vec<IterationRecord> = partials
        .into_iter()
        .filter(|(iter, _)| summary_iterations.contains(iter))
        .map(|(iter, partial)| partial_to_iteration_record(iter, &partial))
        .collect();

    let cut_stats = RowPoolStatistics {
        total_generated: fcf.total_generated_cuts() as u64,
        total_active: fcf.total_active_cuts() as u64,
        peak_active,
        cuts_active: fcf.total_active_cuts() as u64,
        rows_in_lp_total,
        rows_in_lp_solve_count,
        rows_in_lp_max,
        total_loaded: fcf.total_warm_start_cuts() as u64,
    };

    let mask = result.stop_decision.mask();
    let converged = result.stop_decision.configured_stop()
        && (mask.contains(StopMask::GAP) || mask.contains(StopMask::BOUND_STALLING));
    let status = if result.stop_decision.ended_by_shutdown() {
        RunStatus::Partial
    } else {
        RunStatus::Complete
    };

    // None for non-positive lower bound: the gap percentage is undefined
    // (final_lb == 0) or sign-inverted (final_lb < 0).
    let final_gap_percent = if result.final_lb > 0.0 {
        Some(result.final_gap * 100.0)
    } else {
        None
    };

    #[allow(clippy::cast_possible_truncation)]
    let iterations_completed = result.iterations as u32;

    #[allow(clippy::cast_possible_truncation)]
    let cut_selection_records: Vec<RowSelectionRecord> = events
        .iter()
        .filter_map(|event| {
            if let TrainingEvent::PolicySelectionComplete {
                iteration,
                per_stage,
                ..
            } = event
            {
                Some(per_stage.iter().map(move |rec| RowSelectionRecord {
                    iteration: *iteration as u32,
                    stage: rec.stage,
                    cuts_populated: rec.rows_populated,
                    cuts_active_before: rec.rows_active_before,
                    cuts_deactivated: rec.rows_deactivated,
                    cuts_reactivated: rec.rows_reactivated,
                    cuts_active_after: rec.rows_active_after,
                    selection_time_ms: rec.selection_time_ms,
                    budget_evicted: rec.budget_evicted,
                    active_after_budget: rec.active_after_budget,
                }))
            } else {
                None
            }
        })
        .flatten()
        .collect();

    let worker_timing_records = build_worker_timing_records(events, &convergence_records);

    // The bound regime is set by the forward variant `ForwardPassesResolution`
    // selects: an enumerated forward is exact (no sampling distribution, so std
    // is NULL), a sampled forward is statistical. This site WRITES the kind; it
    // does not decide the bound semantics.
    let (final_upper_bound_kind, final_upper_bound_std) = if upper_bound_exact {
        ("exact".to_string(), None)
    } else {
        ("statistical".to_string(), Some(result.final_ub_std))
    };

    TrainingOutput {
        convergence_records,
        final_lower_bound: result.final_lb,
        final_upper_bound: Some(result.final_ub),
        final_gap_percent,
        final_upper_bound_std,
        final_upper_bound_kind,
        iterations_completed,
        converged,
        termination_reason: result.reason.clone(),
        status,
        total_time_ms: result.total_time_ms,
        cut_stats,
        cut_selection_records,
        worker_timing_records,
        // Populated downstream (CLI/Python) from live solver statistics.
        training_solve_stats: MetadataTrainingSolveStats::default(),
    }
}

/// Run-level sums (milliseconds) of the phase-wall, wait, and serial-bucket
/// timing components carried per iteration in [`IterationRecord`]. Feeds
/// `cobre-cli`'s post-run summary (`TrainingSummary::forward_phase_wall_seconds`
/// and its siblings), which divides each field by 1000 once to seconds.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PhaseTimingTotals {
    /// Σ `forward_wall_ms` — coordinator-measured forward-phase wall.
    pub forward_wall_ms: u64,
    /// Σ `backward_wall_ms` — coordinator-measured backward-phase wall.
    pub backward_wall_ms: u64,
    /// Σ `fwd_load_imbalance_ms` — forward-phase worker wait.
    pub forward_wait_ms: u64,
    /// Σ `bwd_load_imbalance_ms` — backward-phase worker wait.
    pub backward_wait_ms: u64,
    /// Σ `lower_bound_ms`.
    pub lower_bound_ms: u64,
    /// Σ `cut_selection_ms`.
    pub cut_selection_ms: u64,
    /// Σ `cut_sync_ms`.
    pub cut_sync_ms: u64,
    /// Σ `mpi_allreduce_ms`.
    pub allreduce_ms: u64,
    /// Σ (`fwd_scheduling_overhead_ms` + `bwd_scheduling_overhead_ms`).
    pub scheduling_ms: u64,
}

/// Sum the phase-wall/wait/serial timing components across every iteration in
/// `records`. Integer-ms accumulation only — the caller converts to seconds once
/// (declaration-order invariance).
#[must_use]
pub fn sum_phase_timing_ms(records: &[IterationRecord]) -> PhaseTimingTotals {
    records
        .iter()
        .fold(PhaseTimingTotals::default(), |acc, r| PhaseTimingTotals {
            forward_wall_ms: acc.forward_wall_ms.saturating_add(r.time_forward_wall_ms),
            backward_wall_ms: acc.backward_wall_ms.saturating_add(r.time_backward_wall_ms),
            forward_wait_ms: acc
                .forward_wait_ms
                .saturating_add(r.time_fwd_load_imbalance_ms),
            backward_wait_ms: acc
                .backward_wait_ms
                .saturating_add(r.time_bwd_load_imbalance_ms),
            lower_bound_ms: acc.lower_bound_ms.saturating_add(r.time_lower_bound_ms),
            cut_selection_ms: acc.cut_selection_ms.saturating_add(r.time_cut_selection_ms),
            cut_sync_ms: acc.cut_sync_ms.saturating_add(r.time_cut_sync_ms),
            allreduce_ms: acc.allreduce_ms.saturating_add(r.time_mpi_allreduce_ms),
            scheduling_ms: acc
                .scheduling_ms
                .saturating_add(r.time_fwd_scheduling_overhead_ms)
                .saturating_add(r.time_bwd_scheduling_overhead_ms),
        })
}

/// Build the `(iteration, rank, worker_id)` timing rows for
/// `training/timing/iterations.parquet`. Each iteration emits one rank-aggregated
/// row (`worker_id=None`, rank-only columns) plus one per-worker row per
/// `(rank, worker_id)` (parallel-region contributions); the two row kinds use
/// disjoint columns so `SUM(col) GROUP BY iteration` recovers the totals.
fn build_worker_timing_records(
    events: &[TrainingEvent],
    convergence_records: &[IterationRecord],
) -> Vec<cobre_io::WorkerTimingRecord> {
    use cobre_core::{
        WORKER_TIMING_SLOT_BWD_SETUP, WORKER_TIMING_SLOT_BWD_WALL, WORKER_TIMING_SLOT_COUNT,
        WORKER_TIMING_SLOT_FWD_SETUP, WORKER_TIMING_SLOT_FWD_WALL, WORKER_TIMING_SLOT_SCORING,
    };

    let mut per_worker: BTreeMap<(u32, i32, i32), [u64; WORKER_TIMING_SLOT_COUNT]> =
        BTreeMap::new();
    for event in events {
        if let TrainingEvent::WorkerTiming {
            iteration,
            rank,
            worker_id,
            timings,
            ..
        } = event
        {
            #[allow(clippy::cast_possible_truncation)]
            let iter_u32 = *iteration as u32;
            let entry = per_worker
                .entry((iter_u32, *rank, *worker_id))
                .or_insert([0_u64; WORKER_TIMING_SLOT_COUNT]);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                entry[WORKER_TIMING_SLOT_FWD_WALL] = entry[WORKER_TIMING_SLOT_FWD_WALL]
                    .saturating_add(timings.forward_wall_ms.max(0.0).round() as u64);
                entry[WORKER_TIMING_SLOT_BWD_WALL] = entry[WORKER_TIMING_SLOT_BWD_WALL]
                    .saturating_add(timings.backward_wall_ms.max(0.0).round() as u64);
                entry[WORKER_TIMING_SLOT_BWD_SETUP] = entry[WORKER_TIMING_SLOT_BWD_SETUP]
                    .saturating_add(timings.bwd_setup_ms.max(0.0).round() as u64);
                entry[WORKER_TIMING_SLOT_FWD_SETUP] = entry[WORKER_TIMING_SLOT_FWD_SETUP]
                    .saturating_add(timings.fwd_setup_ms.max(0.0).round() as u64);
                entry[WORKER_TIMING_SLOT_SCORING] = entry[WORKER_TIMING_SLOT_SCORING]
                    .saturating_add(timings.scoring_ms.max(0.0).round() as u64);
            }
        }
    }

    let mut out: Vec<cobre_io::WorkerTimingRecord> =
        Vec::with_capacity(convergence_records.len() + per_worker.len());

    for record in convergence_records {
        let mut timings = [0_u64; WORKER_TIMING_SLOT_COUNT];
        timings[2] = record.time_cut_selection_ms;
        timings[3] = record.time_mpi_allreduce_ms;
        timings[4] = record.time_cut_sync_ms;
        timings[5] = record.time_lower_bound_ms;
        timings[6] = record.time_state_exchange_ms;
        timings[7] = record.time_cut_batch_build_ms;
        timings[9] = record.time_bwd_load_imbalance_ms;
        timings[10] = record.time_bwd_scheduling_overhead_ms;
        timings[12] = record.time_fwd_load_imbalance_ms;
        timings[13] = record.time_fwd_scheduling_overhead_ms;
        timings[14] = record.time_overhead_ms;
        out.push(cobre_io::WorkerTimingRecord {
            iteration: record.iteration,
            rank: 0,
            worker_id: None,
            timings,
        });
    }

    for ((iteration, rank, worker_id), timings) in per_worker {
        out.push(cobre_io::WorkerTimingRecord {
            iteration,
            rank,
            worker_id: Some(worker_id),
            timings,
        });
    }

    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::doc_markdown)]
mod tests {
    use cobre_core::TrainingEvent;
    use cobre_io::{IterationRecord, RunStatus};

    use super::{PhaseTimingTotals, build_training_output, sum_phase_timing_ms};
    use crate::config::ShutdownSource;
    use crate::setup::NodeId;
    use crate::stopping_rule::{
        MonitorState, StopDecision, StoppingMode, StoppingRule, StoppingRuleSet,
    };
    use crate::{ConvergenceMonitor, FutureCostFunction, SyncResult, TrainingResult};

    fn make_result(reason: &str, lb: f64, ub: f64, gap: f64, iterations: u64) -> TrainingResult {
        TrainingResult::new(
            lb,
            ub,
            0.0,
            gap,
            iterations,
            reason.to_string(),
            1_000,
            Vec::new(),
            Vec::new(),
            None,
            None,
        )
    }

    fn make_iteration_summary(iter: u64, lb: f64, ub: f64, gap: f64) -> TrainingEvent {
        TrainingEvent::IterationSummary {
            iteration: iter,
            lower_bound: lb,
            upper_bound: ub,
            gap,
            wall_time_ms: iter * 100,
            iteration_time_ms: 100,
            forward_ms: 40,
            backward_ms: 50,
            lp_solves: 60,
            solve_time_ms: 0.0,
            lower_bound_eval_ms: 0,
            fwd_setup_time_ms: 0,
            fwd_load_imbalance_ms: 0,
            fwd_scheduling_overhead_ms: 0,
            rows_in_lp_sum: 0,
            rows_in_lp_count: 0,
            rows_in_lp_max: 0,
        }
    }

    fn make_empty_fcf() -> FutureCostFunction {
        FutureCostFunction::new(2, 1, 4, 10, &[0; 2])
    }

    #[test]
    fn records_count_matches_iteration_summaries() {
        let result = make_result("iteration_limit", 100.0, 110.0, 0.091, 3);
        let events = vec![
            make_iteration_summary(1, 95.0, 112.0, 0.15),
            make_iteration_summary(2, 98.0, 111.0, 0.12),
            make_iteration_summary(3, 100.0, 110.0, 0.091),
        ];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf, false);

        assert_eq!(output.convergence_records.len(), 3);
    }

    #[test]
    fn cut_stats_from_fcf() {
        let result = make_result("iteration_limit", 80.0, 100.0, 0.2, 1);
        let events = vec![make_iteration_summary(1, 80.0, 100.0, 0.2)];

        let mut fcf = FutureCostFunction::new(2, 1, 4, 10, &[0; 2]);

        fcf.add_cut(NodeId(0), 0, 0, 0, 1.0, &[1.0]);
        fcf.add_cut(NodeId(0), 0, 0, 1, 2.0, &[0.5]);
        fcf.add_cut(NodeId(0), 0, 0, 2, 3.0, &[0.25]);
        fcf.add_cut(NodeId(0), 1, 0, 0, 4.0, &[1.0]);
        fcf.add_cut(NodeId(0), 1, 0, 1, 5.0, &[0.5]);

        let output = build_training_output(&result, &events, &fcf, false);

        assert_eq!(
            output.cut_stats.total_generated, 5,
            "total_generated must equal the true number of cuts added (iteration 0 has no gap)"
        );
        assert_eq!(
            output.cut_stats.total_active, 5,
            "total_active must equal active cuts in all pools"
        );
    }

    #[test]
    fn total_generated_excludes_reserved_leading_slots() {
        // With 1-based iterations and forward_passes > 1, the slot block
        // [0, forward_passes) is never written, so populated_count over-counts
        // by forward_passes per cut-receiving pool. total_generated must report
        // the true number of cuts added, not the high-water mark.
        let result = make_result("iteration_limit", 80.0, 100.0, 0.2, 1);
        let events = vec![make_iteration_summary(1, 80.0, 100.0, 0.2)];

        let mut fcf = FutureCostFunction::new(2, 1, 2, 10, &[0; 2]);
        // iteration 1, forward_passes 2 -> slots 2,3 (block [0,2) stays empty).
        fcf.add_cut(NodeId(0), 0, 1, 0, 1.0, &[1.0]);
        fcf.add_cut(NodeId(0), 0, 1, 1, 2.0, &[1.0]);
        // stage 1: one cut at iteration 1 -> slot 2.
        fcf.add_cut(NodeId(0), 1, 1, 0, 3.0, &[1.0]);

        let populated: u64 = fcf.pools.iter().map(|p| p.populated() as u64).sum();
        assert_eq!(
            populated, 7,
            "high-water mark includes the empty leading block"
        );

        let output = build_training_output(&result, &events, &fcf, false);
        assert_eq!(
            output.cut_stats.total_generated, 3,
            "total_generated must count the 3 cuts actually added"
        );
        assert!(
            output.cut_stats.total_generated < populated,
            "total_generated must exclude reserved-but-empty leading slots"
        );
    }

    #[test]
    fn gap_percent_none_when_lb_nonpositive() {
        let result = make_result("iteration_limit", 0.0, 10.0, 1.0, 1);
        let events = vec![make_iteration_summary(1, 0.0, 10.0, 1.0)];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf, false);

        assert!(
            output.final_gap_percent.is_none(),
            "final_gap_percent must be None when final_lb <= 0"
        );
    }

    const KIND_ITERATION_LIMIT: u8 = 1 << 0;
    const KIND_TIME_LIMIT: u8 = 1 << 1;
    const KIND_BOUND_STALLING: u8 = 1 << 2;
    const KIND_GAP: u8 = 1 << 3;

    /// Evaluate one rule per kind in `configured`, in declared order, at a state
    /// where exactly the kinds in `triggered` hold.
    fn stop_case(
        configured: u8,
        triggered: u8,
        mode: StoppingMode,
        shutdown: bool,
    ) -> StopDecision {
        let lb = 100.0;
        let mut rules = Vec::new();
        if configured & KIND_ITERATION_LIMIT != 0 {
            rules.push(StoppingRule::IterationLimit { limit: 5 });
        }
        if configured & KIND_TIME_LIMIT != 0 {
            rules.push(StoppingRule::TimeLimit { seconds: 10.0 });
        }
        if configured & KIND_BOUND_STALLING != 0 {
            rules.push(StoppingRule::BoundStalling {
                tolerance: 1e-3,
                iterations: 1,
            });
        }
        if configured & KIND_GAP != 0 {
            rules.push(StoppingRule::Gap {
                tolerance: Some(1.0),
                relative_tolerance: None,
            });
        }
        let holds = |kind: u8| triggered & kind != 0;
        let state = MonitorState {
            iteration: if holds(KIND_ITERATION_LIMIT) { 5 } else { 1 },
            wall_time_seconds: if holds(KIND_TIME_LIMIT) { 20.0 } else { 0.0 },
            lower_bound: lb,
            upper_bound: if holds(KIND_GAP) {
                lb + 0.5
            } else {
                lb + 100.0
            },
            lower_bound_history: if holds(KIND_BOUND_STALLING) {
                vec![lb]
            } else {
                vec![]
            },
            shutdown_requested: shutdown,
        };
        StoppingRuleSet { rules, mode }.evaluate(&state)
    }

    fn converged_for(stop_decision: StopDecision) -> bool {
        let mut result = make_result("iteration_limit", 100.0, 110.0, 0.1, 1);
        result.stop_decision = stop_decision;
        build_training_output(&result, &[], &make_empty_fcf(), false).converged
    }

    #[test]
    fn converged_matches_the_stop_mask_truth_table() {
        let mut cases = 0;
        for configured in 1..16_u8 {
            for triggered in (0..16_u8).filter(|t| t & !configured == 0) {
                for mode in [StoppingMode::Any, StoppingMode::All] {
                    for shutdown in [false, true] {
                        let conjuncts = configured & !KIND_ITERATION_LIMIT;
                        let stopped_by_rules = match mode {
                            StoppingMode::Any => triggered != 0,
                            StoppingMode::All => conjuncts != 0 && conjuncts & !triggered == 0,
                        };
                        let expected =
                            stopped_by_rules && triggered & (KIND_GAP | KIND_BOUND_STALLING) != 0;
                        assert_eq!(
                            converged_for(stop_case(configured, triggered, mode, shutdown)),
                            expected,
                            "C={configured:04b} T={triggered:04b} {mode:?} shutdown={shutdown}"
                        );
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 320);
    }

    #[test]
    fn converged_is_false_when_only_part_of_an_all_mode_set_triggered() {
        let decision = stop_case(
            KIND_ITERATION_LIMIT | KIND_TIME_LIMIT | KIND_GAP,
            KIND_GAP,
            StoppingMode::All,
            true,
        );
        assert!(!converged_for(decision));
    }

    #[test]
    fn converged_is_false_for_a_result_without_a_stop_decision() {
        let result = make_result("bound_stalling", 100.0, 101.0, 0.01, 5);
        let output = build_training_output(&result, &[], &make_empty_fcf(), false);
        assert!(!output.converged);
    }

    fn status_for(reason: &str, stop_decision: StopDecision) -> RunStatus {
        let mut result = make_result(reason, 100.0, 110.0, 0.1, 1);
        result.stop_decision = stop_decision;
        build_training_output(&result, &[], &make_empty_fcf(), false).status
    }

    fn monitor_decision(max_iterations: u64, shutdown: ShutdownSource) -> StopDecision {
        let rules = StoppingRuleSet {
            rules: vec![StoppingRule::IterationLimit { limit: 100 }],
            mode: StoppingMode::Any,
        };
        let mut monitor = ConvergenceMonitor::with_iteration_budget(rules, max_iterations);
        monitor.set_shutdown(shutdown);
        monitor.update(
            100.0,
            &SyncResult {
                global_ub_mean: 110.0,
                global_ub_std: 1.0,
                ci_95_half_width: 0.5,
                sync_time_ms: 0,
            },
            0.0,
        )
    }

    #[test]
    fn training_status_is_partial_only_when_a_shutdown_alone_ended_training() {
        let cases = [
            (
                "cooperative shutdown, no configured stop",
                status_for(
                    "graceful_shutdown",
                    monitor_decision(100, ShutdownSource::Cooperative),
                ),
                RunStatus::Partial,
            ),
            (
                "signal shutdown, no configured stop",
                status_for(
                    "graceful_shutdown",
                    monitor_decision(100, ShutdownSource::Signal),
                ),
                RunStatus::Partial,
            ),
            (
                "shutdown with only part of an all-mode set triggered",
                status_for(
                    "graceful_shutdown",
                    StoppingRuleSet {
                        rules: vec![
                            StoppingRule::IterationLimit { limit: 100 },
                            StoppingRule::TimeLimit { seconds: 10.0 },
                            StoppingRule::BoundStalling {
                                tolerance: 1e-3,
                                iterations: 1,
                            },
                        ],
                        mode: StoppingMode::All,
                    }
                    .evaluate(&MonitorState {
                        iteration: 1,
                        wall_time_seconds: 0.0,
                        lower_bound: 100.0,
                        upper_bound: 200.0,
                        lower_bound_history: vec![100.0],
                        shutdown_requested: true,
                    }),
                ),
                RunStatus::Partial,
            ),
            (
                "configured stop with a shutdown at the same iteration",
                status_for(
                    "bound_stalling",
                    stop_case(
                        KIND_BOUND_STALLING,
                        KIND_BOUND_STALLING,
                        StoppingMode::Any,
                        true,
                    ),
                ),
                RunStatus::Complete,
            ),
            (
                "exhausted budget with a signal shutdown",
                status_for(
                    "iteration_limit",
                    monitor_decision(1, ShutdownSource::Signal),
                ),
                RunStatus::Complete,
            ),
            (
                "configured stop, no shutdown",
                status_for(
                    "bound_stalling",
                    stop_case(
                        KIND_BOUND_STALLING,
                        KIND_BOUND_STALLING,
                        StoppingMode::Any,
                        false,
                    ),
                ),
                RunStatus::Complete,
            ),
            (
                "default decision ended on an error",
                status_for("error", StopDecision::default()),
                RunStatus::Complete,
            ),
            (
                "default decision with a graceful_shutdown reason",
                status_for("graceful_shutdown", StopDecision::default()),
                RunStatus::Complete,
            ),
        ];
        for (case, actual, expected) in cases {
            assert_eq!(actual, expected, "{case}");
        }
    }

    #[test]
    fn empty_events_produces_zero_records() {
        let result = make_result("iteration_limit", 50.0, 60.0, 0.2, 0);
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &[], &fcf, false);

        assert_eq!(output.convergence_records.len(), 0);
        assert_eq!(output.final_lower_bound, 50.0);
        assert_eq!(output.final_upper_bound, Some(60.0));
        assert_eq!(output.total_time_ms, 1_000);
        assert!(!output.converged);
    }

    #[test]
    fn gap_percent_computed_correctly() {
        let result = make_result("bound_stalling", 100.0, 102.0, 0.02, 3);
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &[], &fcf, false);

        assert_eq!(output.final_gap_percent, Some(2.0));
    }

    #[test]
    fn iteration_gap_percent_none_when_lb_zero_or_negative() {
        let result = make_result("iteration_limit", 0.0, 10.0, 1.0, 1);
        let events = vec![make_iteration_summary(1, 0.0, 10.0, 1.0)];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf, false);

        assert!(output.convergence_records[0].gap_percent.is_none());
    }

    #[test]
    fn upper_bound_std_from_forward_sync_complete() {
        let result = make_result("iteration_limit", 100.0, 110.0, 0.1, 1);
        let events = vec![
            make_iteration_summary(1, 100.0, 110.0, 0.1),
            TrainingEvent::ForwardSyncComplete {
                iteration: 1,
                global_ub_mean: 110.0,
                global_ub_std: 3.5,
                sync_time_ms: 5,
            },
        ];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf, false);

        assert_eq!(output.convergence_records[0].upper_bound_std, 3.5);
    }

    #[test]
    fn forward_passes_from_forward_pass_complete() {
        let result = make_result("iteration_limit", 100.0, 110.0, 0.1, 1);
        let events = vec![
            make_iteration_summary(1, 100.0, 110.0, 0.1),
            TrainingEvent::ForwardPassComplete {
                iteration: 1,
                scenarios: 8,
                ub_mean: 110.0,
                ub_std: 2.0,
                elapsed_ms: 40,
            },
        ];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf, false);

        assert_eq!(output.convergence_records[0].forward_passes, 8);
    }

    #[test]
    fn cut_fields_from_backward_and_sync_events() {
        let result = make_result("iteration_limit", 100.0, 110.0, 0.1, 1);
        let events = vec![
            make_iteration_summary(1, 100.0, 110.0, 0.1),
            TrainingEvent::BackwardPassComplete {
                iteration: 1,
                rows_generated: 12,
                stages_processed: 3,
                elapsed_ms: 80,
                state_exchange_time_ms: 0,
                row_batch_build_time_ms: 0,
                setup_time_ms: 0,
                load_imbalance_ms: 0,
                scheduling_overhead_ms: 0,
            },
            TrainingEvent::PolicySyncComplete {
                iteration: 1,
                rows_distributed: 12,
                rows_active: 24,
                rows_removed: 2,
                sync_time_ms: 4,
            },
        ];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf, false);

        let rec = &output.convergence_records[0];
        assert_eq!(rec.cuts_added, 12);
        assert_eq!(rec.cuts_removed, 2);
        assert_eq!(rec.cuts_active, 24);
    }

    #[test]
    fn peak_active_tracks_maximum_cuts_active() {
        let result = make_result("iteration_limit", 100.0, 110.0, 0.1, 3);
        let events = vec![
            make_iteration_summary(1, 95.0, 112.0, 0.15),
            TrainingEvent::PolicySyncComplete {
                iteration: 1,
                rows_distributed: 10,
                rows_active: 10,
                rows_removed: 0,
                sync_time_ms: 2,
            },
            make_iteration_summary(2, 98.0, 111.0, 0.12),
            TrainingEvent::PolicySyncComplete {
                iteration: 2,
                rows_distributed: 10,
                rows_active: 20,
                rows_removed: 0,
                sync_time_ms: 2,
            },
            make_iteration_summary(3, 100.0, 110.0, 0.1),
            TrainingEvent::PolicySyncComplete {
                iteration: 3,
                rows_distributed: 5,
                rows_active: 18, // peak was 20 in iteration 2
                rows_removed: 7,
                sync_time_ms: 2,
            },
        ];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf, false);

        assert_eq!(output.cut_stats.peak_active, 20);
    }

    #[test]
    fn iterations_completed_from_result() {
        let result = make_result("iteration_limit", 80.0, 100.0, 0.2, 42);
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &[], &fcf, false);

        assert_eq!(output.iterations_completed, 42);
    }

    #[test]
    fn termination_reason_copied_from_result() {
        let result = make_result("time_limit", 70.0, 100.0, 0.3, 20);
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &[], &fcf, false);

        assert_eq!(output.termination_reason, "time_limit");
    }

    #[test]
    fn per_phase_timing_captured_from_sync_and_selection_events() {
        let result = make_result("iteration_limit", 100.0, 110.0, 0.1, 1);
        let events = vec![
            TrainingEvent::IterationSummary {
                iteration: 1,
                lower_bound: 100.0,
                upper_bound: 110.0,
                gap: 0.1,
                wall_time_ms: 120,
                iteration_time_ms: 120,
                forward_ms: 40,
                backward_ms: 50,
                lp_solves: 60,
                solve_time_ms: 0.0,
                lower_bound_eval_ms: 0,
                fwd_setup_time_ms: 0,
                fwd_load_imbalance_ms: 0,
                fwd_scheduling_overhead_ms: 0,
                rows_in_lp_sum: 0,
                rows_in_lp_count: 0,
                rows_in_lp_max: 0,
            },
            TrainingEvent::ForwardSyncComplete {
                iteration: 1,
                global_ub_mean: 110.0,
                global_ub_std: 2.0,
                sync_time_ms: 7,
            },
            TrainingEvent::PolicySyncComplete {
                iteration: 1,
                rows_distributed: 10,
                rows_active: 10,
                rows_removed: 0,
                sync_time_ms: 5,
            },
            TrainingEvent::PolicySelectionComplete {
                iteration: 1,
                rows_deactivated: 3,
                stages_processed: 2,
                selection_time_ms: 8,
                allgatherv_time_ms: 2,
                per_stage: vec![],
            },
        ];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf, false);
        let rec = &output.convergence_records[0];

        assert_eq!(
            rec.time_forward_wall_ms, 40,
            "forward wall must equal forward_ms"
        );
        assert_eq!(
            rec.time_backward_wall_ms, 50,
            "backward wall must equal backward_ms"
        );
        assert_eq!(
            rec.time_mpi_allreduce_ms, 7,
            "allreduce must come from ForwardSyncComplete"
        );
        assert_eq!(
            rec.time_cut_sync_ms, 5,
            "cut_sync must come from PolicySyncComplete"
        );
        assert_eq!(
            rec.time_cut_selection_ms, 8,
            "selection must come from PolicySelectionComplete"
        );
    }

    #[test]
    fn overhead_ms_is_total_minus_attributed_phases() {
        let result = make_result("iteration_limit", 100.0, 110.0, 0.1, 1);
        // Top-level non-overlapping phases:
        //   forward=40, backward=50, allreduce=7, selection=8,
        //   allgatherv=2, lower_bound=3 → attributed=110
        // total=120 → overhead=10; cut_sync(5) is a sub-component of
        // backward(50) and is NOT in the attributed sum.
        let events = vec![
            TrainingEvent::IterationSummary {
                iteration: 1,
                lower_bound: 100.0,
                upper_bound: 110.0,
                gap: 0.1,
                wall_time_ms: 120,
                iteration_time_ms: 120,
                forward_ms: 40,
                backward_ms: 50,
                lp_solves: 60,
                solve_time_ms: 0.0,
                lower_bound_eval_ms: 3,
                fwd_setup_time_ms: 0,
                fwd_load_imbalance_ms: 0,
                fwd_scheduling_overhead_ms: 0,
                rows_in_lp_sum: 0,
                rows_in_lp_count: 0,
                rows_in_lp_max: 0,
            },
            TrainingEvent::ForwardSyncComplete {
                iteration: 1,
                global_ub_mean: 110.0,
                global_ub_std: 2.0,
                sync_time_ms: 7,
            },
            TrainingEvent::PolicySyncComplete {
                iteration: 1,
                rows_distributed: 10,
                rows_active: 10,
                rows_removed: 0,
                sync_time_ms: 5,
            },
            TrainingEvent::PolicySelectionComplete {
                iteration: 1,
                rows_deactivated: 3,
                stages_processed: 2,
                selection_time_ms: 8,
                allgatherv_time_ms: 2,
                per_stage: vec![],
            },
        ];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf, false);
        let rec = &output.convergence_records[0];

        assert_eq!(
            rec.time_overhead_ms, 10,
            "overhead_ms must equal total(120) - attributed(110) = 10"
        );
    }

    #[test]
    fn overhead_ms_saturates_at_zero_when_attributed_exceeds_total() {
        let result = make_result("iteration_limit", 100.0, 110.0, 0.1, 1);
        let events = vec![TrainingEvent::IterationSummary {
            iteration: 1,
            lower_bound: 100.0,
            upper_bound: 110.0,
            gap: 0.1,
            wall_time_ms: 10,
            iteration_time_ms: 10,
            forward_ms: 50,
            backward_ms: 50,
            lp_solves: 5,
            solve_time_ms: 0.0,
            lower_bound_eval_ms: 0,
            fwd_setup_time_ms: 0,
            fwd_load_imbalance_ms: 0,
            fwd_scheduling_overhead_ms: 0,
            rows_in_lp_sum: 0,
            rows_in_lp_count: 0,
            rows_in_lp_max: 0,
        }];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf, false);
        let rec = &output.convergence_records[0];

        assert_eq!(
            rec.time_overhead_ms, 0,
            "overhead_ms must be 0 when attributed phases exceed total (saturating sub)"
        );
    }

    #[test]
    fn cut_selection_records_extracted_from_events() {
        use cobre_core::StageRowSelectionRecord;

        let result = make_result("iteration_limit", 100.0, 110.0, 0.1, 3);
        let events = vec![
            make_iteration_summary(1, 95.0, 112.0, 0.15),
            make_iteration_summary(2, 98.0, 111.0, 0.12),
            make_iteration_summary(3, 100.0, 110.0, 0.1),
            TrainingEvent::PolicySelectionComplete {
                iteration: 2,
                rows_deactivated: 3,
                stages_processed: 2,
                selection_time_ms: 10,
                allgatherv_time_ms: 0,
                per_stage: vec![
                    StageRowSelectionRecord {
                        stage: 0,
                        rows_populated: 5,
                        rows_active_before: 5,
                        rows_deactivated: 0,
                        rows_reactivated: 0,
                        rows_active_after: 5,
                        selection_time_ms: 0.0,
                        budget_evicted: None,
                        active_after_budget: None,
                        rows_in_lp: 5,
                    },
                    StageRowSelectionRecord {
                        stage: 1,
                        rows_populated: 8,
                        rows_active_before: 8,
                        rows_deactivated: 3,
                        rows_reactivated: 0,
                        rows_active_after: 5,
                        selection_time_ms: 2.5,
                        budget_evicted: None,
                        active_after_budget: None,
                        rows_in_lp: 8,
                    },
                ],
            },
        ];
        let fcf = make_empty_fcf();
        let output = build_training_output(&result, &events, &fcf, false);

        assert_eq!(output.cut_selection_records.len(), 2);
        assert_eq!(output.cut_selection_records[0].iteration, 2);
        assert_eq!(output.cut_selection_records[0].stage, 0);
        assert_eq!(output.cut_selection_records[0].cuts_deactivated, 0);
        assert_eq!(output.cut_selection_records[1].stage, 1);
        assert_eq!(output.cut_selection_records[1].cuts_deactivated, 3);
        assert_eq!(output.cut_selection_records[1].cuts_active_after, 5);
    }

    #[test]
    fn no_cut_selection_events_produces_empty_records() {
        let result = make_result("iteration_limit", 100.0, 110.0, 0.1, 1);
        let events = vec![make_iteration_summary(1, 100.0, 110.0, 0.1)];
        let fcf = make_empty_fcf();
        let output = build_training_output(&result, &events, &fcf, false);

        assert!(output.cut_selection_records.is_empty());
    }

    fn zero_iteration_record() -> IterationRecord {
        IterationRecord {
            iteration: 1,
            lower_bound: 0.0,
            upper_bound: 0.0,
            upper_bound_std: 0.0,
            gap_percent: None,
            cuts_added: 0,
            cuts_removed: 0,
            cuts_active: 0,
            time_forward_ms: 0,
            time_backward_ms: 0,
            time_total_ms: 0,
            time_forward_wall_ms: 0,
            time_backward_wall_ms: 0,
            time_cut_selection_ms: 0,
            time_mpi_allreduce_ms: 0,
            time_cut_sync_ms: 0,
            time_lower_bound_ms: 0,
            time_state_exchange_ms: 0,
            time_cut_batch_build_ms: 0,
            time_bwd_load_imbalance_ms: 0,
            time_bwd_scheduling_overhead_ms: 0,
            time_fwd_load_imbalance_ms: 0,
            time_fwd_scheduling_overhead_ms: 0,
            time_overhead_ms: 0,
            forward_passes: 0,
            lp_solves: 0,
            solve_time_ms: 0.0,
            mean_rows_in_lp: 0.0,
        }
    }

    #[test]
    fn sum_phase_timing_ms_sums_all_components_across_iterations() {
        let record_1 = IterationRecord {
            time_forward_wall_ms: 40,
            time_backward_wall_ms: 50,
            time_fwd_load_imbalance_ms: 5,
            time_bwd_load_imbalance_ms: 15,
            time_lower_bound_ms: 3,
            time_cut_selection_ms: 8,
            time_cut_sync_ms: 4,
            time_mpi_allreduce_ms: 7,
            time_fwd_scheduling_overhead_ms: 2,
            time_bwd_scheduling_overhead_ms: 6,
            ..zero_iteration_record()
        };
        let record_2 = IterationRecord {
            time_forward_wall_ms: 60,
            time_backward_wall_ms: 70,
            time_fwd_load_imbalance_ms: 9,
            time_bwd_load_imbalance_ms: 25,
            time_lower_bound_ms: 1,
            time_cut_selection_ms: 2,
            time_cut_sync_ms: 3,
            time_mpi_allreduce_ms: 4,
            time_fwd_scheduling_overhead_ms: 5,
            time_bwd_scheduling_overhead_ms: 6,
            ..zero_iteration_record()
        };

        let totals = sum_phase_timing_ms(&[record_1, record_2]);

        assert_eq!(totals.forward_wall_ms, 100, "Σ forward_wall_ms");
        assert_eq!(
            totals.backward_wall_ms, 120,
            "backward_phase_wall_seconds source: Σ backward_wall_ms"
        );
        assert_eq!(totals.forward_wait_ms, 14, "Σ fwd_load_imbalance_ms");
        assert_eq!(
            totals.backward_wait_ms, 40,
            "backward_wait_seconds source: Σ bwd_load_imbalance_ms"
        );
        assert_eq!(totals.lower_bound_ms, 4, "Σ lower_bound_ms");
        assert_eq!(totals.cut_selection_ms, 10, "Σ cut_selection_ms");
        assert_eq!(totals.cut_sync_ms, 7, "Σ cut_sync_ms");
        assert_eq!(totals.allreduce_ms, 11, "Σ mpi_allreduce_ms");
        assert_eq!(
            totals.scheduling_ms, 19,
            "Σ (fwd_scheduling_overhead_ms + bwd_scheduling_overhead_ms)"
        );
    }

    #[test]
    fn sum_phase_timing_ms_empty_slice_returns_zero() {
        let totals = sum_phase_timing_ms(&[]);
        assert_eq!(totals, PhaseTimingTotals::default());
    }
}
