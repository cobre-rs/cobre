//! Bridge from `TrainingResult` and `TrainingEvent` log to `TrainingOutput`.
//!
//! [`build_training_output`] converts the summary produced by the training loop
//! ([`TrainingResult`]) plus the collected event log into the structured
//! [`TrainingOutput`] type required by the output writers in `cobre-io`.
//!
//! ## Design
//!
//! The training loop already emits [`TrainingEvent`] variants at each lifecycle
//! step boundary. Rather than modifying the hot-path `train()` function, this
//! module reads those events after training completes and reconstructs the
//! per-iteration records required by [`cobre_io::TrainingOutput`].
//!
//! The conversion is a pure function — it cannot fail. Missing events for a
//! given iteration produce zero values for the affected fields.

use std::collections::BTreeMap;

use cobre_core::TrainingEvent;
use cobre_io::{CutStatistics, IterationRecord, TrainingOutput};

use crate::{FutureCostFunction, TrainingResult};

/// Partial iteration record accumulated from multiple [`TrainingEvent`] variants
/// before the final [`IterationRecord`] is assembled.
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
    /// MPI allreduce time from [`TrainingEvent::ForwardSyncComplete`] (ms).
    forward_sync_ms: u64,
    /// MPI broadcast time from [`TrainingEvent::CutSyncComplete`] (ms).
    cut_sync_ms: u64,
    /// Local cut selection time from [`TrainingEvent::CutSelectionComplete`] (ms).
    cut_selection_ms: u64,
    /// Allgatherv time from [`TrainingEvent::CutSelectionComplete`] (ms).
    cut_selection_allgatherv_ms: u64,
}

/// Accumulate per-iteration partial records from the event log.
///
/// Processes each [`TrainingEvent`] variant that carries per-iteration timing
/// or convergence data, building a [`BTreeMap`] keyed by iteration number.
/// Also tracks the peak active cut count observed in the log.
///
/// Returns `(partials, peak_active)`.
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
                cuts_generated,
                ..
            } => {
                let record = partials.entry(*iteration).or_default();
                record.cuts_added = *cuts_generated;
            }

            TrainingEvent::CutSyncComplete {
                iteration,
                cuts_active,
                cuts_removed,
                sync_time_ms,
                ..
            } => {
                let record = partials.entry(*iteration).or_default();
                record.cuts_active = *cuts_active;
                record.cuts_removed = *cuts_removed;
                record.cut_sync_ms = *sync_time_ms;
                peak_active = peak_active.max(u64::from(*cuts_active));
            }

            TrainingEvent::CutSelectionComplete {
                iteration,
                selection_time_ms,
                allgatherv_time_ms,
                ..
            } => {
                let record = partials.entry(*iteration).or_default();
                record.cut_selection_ms = *selection_time_ms;
                record.cut_selection_allgatherv_ms = *allgatherv_time_ms;
            }

            _ => {}
        }
    }

    (partials, peak_active)
}

/// Convert a single [`PartialRecord`] into an [`IterationRecord`].
///
/// Computes derived fields (gap percent, overhead) from the accumulated timing
/// data and casts u64 iteration/solve counts to u32.
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

    // Compute overhead as total minus the sum of all attributed phases.
    // Saturating subtraction guards against floating-point or measurement
    // inconsistencies that could cause the sum to exceed total.
    let attributed_ms = partial
        .forward_ms
        .saturating_add(partial.backward_ms)
        .saturating_add(partial.cut_selection_ms)
        .saturating_add(partial.cut_selection_allgatherv_ms)
        .saturating_add(partial.forward_sync_ms)
        .saturating_add(partial.cut_sync_ms);
    let overhead_ms = partial.iteration_time_ms.saturating_sub(attributed_ms);

    IterationRecord {
        iteration: iteration_u32,
        lower_bound: partial.lower_bound,
        upper_bound_mean: partial.upper_bound_mean,
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
        time_forward_solve_ms: partial.forward_ms,
        time_forward_sample_ms: 0,
        time_backward_solve_ms: partial.backward_ms,
        time_backward_cut_ms: 0,
        time_cut_selection_ms: partial.cut_selection_ms,
        time_mpi_allreduce_ms: partial.forward_sync_ms,
        time_mpi_broadcast_ms: partial.cut_sync_ms,
        time_io_write_ms: 0,
        time_overhead_ms: overhead_ms,
    }
}

/// Convert a [`TrainingResult`] and collected event log into a [`TrainingOutput`].
///
/// The caller passes the full event log received from the training loop's
/// `mpsc::Receiver<TrainingEvent>`. Events from multiple lifecycle steps are
/// correlated by their `iteration` field to produce one [`IterationRecord`] per
/// completed iteration.
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::{build_training_output, TrainingResult, FutureCostFunction};
/// use cobre_core::TrainingEvent;
///
/// let result = TrainingResult {
///     final_lb: 100.0,
///     final_ub: 110.0,
///     final_ub_std: 5.0,
///     final_gap: 0.091,
///     iterations: 1,
///     reason: "iteration_limit".to_string(),
///     total_time_ms: 500,
///     basis_cache: Vec::new(),
/// };
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
/// }];
///
/// let fcf = FutureCostFunction::new(2, 1, 4, 1, 0);
/// let output = build_training_output(&result, &events, &fcf);
///
/// assert_eq!(output.convergence_records.len(), 1);
/// assert!(!output.converged);
/// ```
#[must_use]
pub fn build_training_output(
    result: &TrainingResult,
    events: &[TrainingEvent],
    fcf: &FutureCostFunction,
) -> TrainingOutput {
    let (partials, peak_active) = accumulate_partial_records(events);

    // Only include iterations that have an IterationSummary event.
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

    let convergence_records: Vec<IterationRecord> = partials
        .into_iter()
        .filter(|(iter, _)| summary_iterations.contains(iter))
        .map(|(iter, partial)| partial_to_iteration_record(iter, &partial))
        .collect();

    let cut_stats = CutStatistics {
        total_generated: fcf.pools.iter().map(|p| p.populated_count as u64).sum(),
        total_active: fcf.total_active_cuts() as u64,
        peak_active,
    };

    let converged = result.reason == crate::stopping_rule::RULE_BOUND_STALLING
        || result.reason == crate::stopping_rule::RULE_SIMULATION_BASED;

    let final_gap_percent = if result.final_lb > 0.0 {
        Some(result.final_gap * 100.0)
    } else {
        None
    };

    #[allow(clippy::cast_possible_truncation)]
    let iterations_completed = result.iterations as u32;

    TrainingOutput {
        convergence_records,
        final_lower_bound: result.final_lb,
        final_upper_bound: Some(result.final_ub),
        final_gap_percent,
        iterations_completed,
        converged,
        termination_reason: result.reason.clone(),
        total_time_ms: result.total_time_ms,
        cut_stats,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::doc_markdown)]
mod tests {
    use cobre_core::TrainingEvent;

    use super::build_training_output;
    use crate::{FutureCostFunction, TrainingResult};

    fn make_result(reason: &str, lb: f64, ub: f64, gap: f64, iterations: u64) -> TrainingResult {
        TrainingResult {
            final_lb: lb,
            final_ub: ub,
            final_ub_std: 0.0,
            final_gap: gap,
            iterations,
            reason: reason.to_string(),
            total_time_ms: 1_000,
            basis_cache: Vec::new(),
        }
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
        }
    }

    fn make_empty_fcf() -> FutureCostFunction {
        FutureCostFunction::new(2, 1, 4, 10, 0)
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

        let output = build_training_output(&result, &events, &fcf);

        assert_eq!(output.convergence_records.len(), 3);
    }

    #[test]
    fn converged_true_for_bound_stalling() {
        let result = make_result("bound_stalling", 100.0, 101.0, 0.01, 5);
        let events = vec![make_iteration_summary(1, 100.0, 101.0, 0.01)];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf);

        assert!(output.converged);
    }

    #[test]
    fn converged_true_for_simulation_based() {
        let result = make_result("simulation_based", 100.0, 101.0, 0.01, 5);
        let events = vec![make_iteration_summary(1, 100.0, 101.0, 0.01)];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf);

        assert!(output.converged);
    }

    #[test]
    fn converged_false_for_iteration_limit() {
        let result = make_result("iteration_limit", 90.0, 110.0, 0.2, 100);
        let events = vec![make_iteration_summary(1, 90.0, 110.0, 0.2)];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf);

        assert!(!output.converged);
    }

    #[test]
    fn cut_stats_from_fcf() {
        let result = make_result("iteration_limit", 80.0, 100.0, 0.2, 1);
        let events = vec![make_iteration_summary(1, 80.0, 100.0, 0.2)];

        let mut fcf = FutureCostFunction::new(2, 1, 4, 10, 0);

        // Add 3 cuts to pool[0] and 2 cuts to pool[1].
        fcf.add_cut(0, 0, 0, 1.0, &[1.0]);
        fcf.add_cut(0, 0, 1, 2.0, &[0.5]);
        fcf.add_cut(0, 0, 2, 3.0, &[0.25]);
        fcf.add_cut(1, 0, 0, 4.0, &[1.0]);
        fcf.add_cut(1, 0, 1, 5.0, &[0.5]);

        let output = build_training_output(&result, &events, &fcf);

        assert_eq!(
            output.cut_stats.total_generated, 5,
            "total_generated must equal sum of populated_count across all pools"
        );
        assert_eq!(
            output.cut_stats.total_active, 5,
            "total_active must equal active cuts in all pools"
        );
    }

    #[test]
    fn gap_percent_none_when_lb_nonpositive() {
        let result = make_result("iteration_limit", 0.0, 10.0, 1.0, 1);
        let events = vec![make_iteration_summary(1, 0.0, 10.0, 1.0)];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf);

        assert!(
            output.final_gap_percent.is_none(),
            "final_gap_percent must be None when final_lb <= 0"
        );
    }

    #[test]
    fn converged_false_for_all_other_reasons() {
        let reasons = [
            "iteration_limit",
            "time_limit",
            "graceful_shutdown",
            "unknown",
        ];
        let fcf = make_empty_fcf();
        for reason in reasons {
            let result = make_result(reason, 100.0, 110.0, 0.1, 1);
            let output = build_training_output(&result, &[], &fcf);
            assert!(
                !output.converged,
                "converged must be false for reason = {reason}"
            );
        }
    }

    #[test]
    fn empty_events_produces_zero_records() {
        let result = make_result("iteration_limit", 50.0, 60.0, 0.2, 0);
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &[], &fcf);

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

        let output = build_training_output(&result, &[], &fcf);

        assert_eq!(output.final_gap_percent, Some(2.0));
    }

    #[test]
    fn iteration_gap_percent_none_when_lb_zero_or_negative() {
        let result = make_result("iteration_limit", 0.0, 10.0, 1.0, 1);
        let events = vec![make_iteration_summary(1, 0.0, 10.0, 1.0)];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf);

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

        let output = build_training_output(&result, &events, &fcf);

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

        let output = build_training_output(&result, &events, &fcf);

        assert_eq!(output.convergence_records[0].forward_passes, 8);
    }

    #[test]
    fn cut_fields_from_backward_and_sync_events() {
        let result = make_result("iteration_limit", 100.0, 110.0, 0.1, 1);
        let events = vec![
            make_iteration_summary(1, 100.0, 110.0, 0.1),
            TrainingEvent::BackwardPassComplete {
                iteration: 1,
                cuts_generated: 12,
                stages_processed: 3,
                elapsed_ms: 80,
            },
            TrainingEvent::CutSyncComplete {
                iteration: 1,
                cuts_distributed: 12,
                cuts_active: 24,
                cuts_removed: 2,
                sync_time_ms: 4,
            },
        ];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf);

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
            TrainingEvent::CutSyncComplete {
                iteration: 1,
                cuts_distributed: 10,
                cuts_active: 10,
                cuts_removed: 0,
                sync_time_ms: 2,
            },
            make_iteration_summary(2, 98.0, 111.0, 0.12),
            TrainingEvent::CutSyncComplete {
                iteration: 2,
                cuts_distributed: 10,
                cuts_active: 20,
                cuts_removed: 0,
                sync_time_ms: 2,
            },
            make_iteration_summary(3, 100.0, 110.0, 0.1),
            TrainingEvent::CutSyncComplete {
                iteration: 3,
                cuts_distributed: 5,
                cuts_active: 18, // peak was 20 in iteration 2
                cuts_removed: 7,
                sync_time_ms: 2,
            },
        ];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf);

        assert_eq!(output.cut_stats.peak_active, 20);
    }

    #[test]
    fn iterations_completed_from_result() {
        let result = make_result("iteration_limit", 80.0, 100.0, 0.2, 42);
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &[], &fcf);

        assert_eq!(output.iterations_completed, 42);
    }

    #[test]
    fn termination_reason_copied_from_result() {
        let result = make_result("time_limit", 70.0, 100.0, 0.3, 20);
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &[], &fcf);

        assert_eq!(output.termination_reason, "time_limit");
    }

    #[test]
    fn per_phase_timing_captured_from_sync_and_selection_events() {
        let result = make_result("iteration_limit", 100.0, 110.0, 0.1, 1);
        // IterationSummary: forward=40ms, backward=50ms, total=120ms
        // ForwardSyncComplete: sync_time_ms=7
        // CutSyncComplete: sync_time_ms=5
        // CutSelectionComplete: selection_time_ms=8, allgatherv_time_ms=2
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
            },
            TrainingEvent::ForwardSyncComplete {
                iteration: 1,
                global_ub_mean: 110.0,
                global_ub_std: 2.0,
                sync_time_ms: 7,
            },
            TrainingEvent::CutSyncComplete {
                iteration: 1,
                cuts_distributed: 10,
                cuts_active: 10,
                cuts_removed: 0,
                sync_time_ms: 5,
            },
            TrainingEvent::CutSelectionComplete {
                iteration: 1,
                cuts_deactivated: 3,
                stages_processed: 2,
                selection_time_ms: 8,
                allgatherv_time_ms: 2,
            },
        ];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf);
        let rec = &output.convergence_records[0];

        assert_eq!(
            rec.time_forward_solve_ms, 40,
            "forward solve must equal forward_ms"
        );
        assert_eq!(
            rec.time_backward_solve_ms, 50,
            "backward solve must equal backward_ms"
        );
        assert_eq!(
            rec.time_mpi_allreduce_ms, 7,
            "allreduce must come from ForwardSyncComplete"
        );
        assert_eq!(
            rec.time_mpi_broadcast_ms, 5,
            "broadcast must come from CutSyncComplete"
        );
        assert_eq!(
            rec.time_cut_selection_ms, 8,
            "selection must come from CutSelectionComplete"
        );
        assert_eq!(
            rec.time_forward_sample_ms, 0,
            "sample must be 0 (not measured)"
        );
        assert_eq!(
            rec.time_backward_cut_ms, 0,
            "backward_cut must be 0 (not measured)"
        );
        assert_eq!(rec.time_io_write_ms, 0, "io_write must be 0 (not measured)");
    }

    #[test]
    fn overhead_ms_is_total_minus_attributed_phases() {
        let result = make_result("iteration_limit", 100.0, 110.0, 0.1, 1);
        // forward=40, backward=50, allreduce=7, broadcast=5, selection=8 → attributed=110
        // total=120 → overhead=10
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
            },
            TrainingEvent::ForwardSyncComplete {
                iteration: 1,
                global_ub_mean: 110.0,
                global_ub_std: 2.0,
                sync_time_ms: 7,
            },
            TrainingEvent::CutSyncComplete {
                iteration: 1,
                cuts_distributed: 10,
                cuts_active: 10,
                cuts_removed: 0,
                sync_time_ms: 5,
            },
            TrainingEvent::CutSelectionComplete {
                iteration: 1,
                cuts_deactivated: 3,
                stages_processed: 2,
                selection_time_ms: 8,
                allgatherv_time_ms: 2,
            },
        ];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf);
        let rec = &output.convergence_records[0];

        // attributed = forward(40) + backward(50) + selection(8) + allgatherv(2)
        //            + allreduce(7) + broadcast(5) = 112
        // overhead = total(120) - attributed(112) = 8
        //
        // Note: cut_selection_allgatherv_ms(2) is included in the implementation's
        // attributed sum even though it is not surfaced as a separate IterationRecord
        // field. The expected value here accounts for this.
        assert_eq!(
            rec.time_overhead_ms, 8,
            "overhead_ms must equal total(120) - attributed(112) = 8"
        );
    }

    #[test]
    fn overhead_ms_saturates_at_zero_when_attributed_exceeds_total() {
        // If timing measurements are slightly inconsistent, overhead must not underflow.
        let result = make_result("iteration_limit", 100.0, 110.0, 0.1, 1);
        // Construct events where attributed > total (e.g., total=10, forward=50+backward=50)
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
        }];
        let fcf = make_empty_fcf();

        let output = build_training_output(&result, &events, &fcf);
        let rec = &output.convergence_records[0];

        assert_eq!(
            rec.time_overhead_ms, 0,
            "overhead_ms must be 0 when attributed phases exceed total (saturating sub)"
        );
    }
}
