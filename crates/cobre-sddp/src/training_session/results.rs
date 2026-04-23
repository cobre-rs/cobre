//! Result-accumulator sub-struct for one training run.

use std::time::Instant;

use crate::solver_stats::SolverStatsEntry;
use crate::stopping_rule::RULE_ITERATION_LIMIT;

/// Accumulates per-iteration results for one training run.
///
/// Fields are initialised once in `new`, written on each call to
/// `TrainingSession::run_iteration` (and its phase helpers), and consumed
/// by `TrainingSession::finalize` / `finalize_with_error`.
pub(crate) struct TrainingResults {
    pub final_lb: f64,
    pub final_ub: f64,
    pub final_ub_std: f64,
    pub final_gap: f64,
    pub completed_iterations: u64,
    pub termination_reason: String,
    pub solver_stats_log: Vec<SolverStatsEntry>,
    pub start_time: Instant,
}

impl TrainingResults {
    /// Initialise all result accumulators for a training run starting at
    /// `start_iteration`.
    ///
    /// `start_time` is captured here so that `total_time_ms` in the emitted
    /// `TrainingFinished` event reflects elapsed time from the same instant as
    /// the pre-refactor code.
    pub(crate) fn new(start_iteration: u64) -> Self {
        Self {
            start_time: Instant::now(),
            final_lb: 0.0,
            final_ub: 0.0,
            final_ub_std: 0.0,
            final_gap: 0.0,
            completed_iterations: start_iteration,
            termination_reason: RULE_ITERATION_LIMIT.to_string(),
            solver_stats_log: Vec::new(),
        }
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
    clippy::float_cmp
)]
mod tests {
    use std::time::Duration;

    use super::TrainingResults;
    use crate::stopping_rule::RULE_ITERATION_LIMIT;

    #[test]
    fn training_results_new_initialises_all_fields() {
        let r = TrainingResults::new(0);
        assert_eq!(r.final_lb, 0.0);
        assert_eq!(r.final_ub, 0.0);
        assert_eq!(r.final_ub_std, 0.0);
        assert_eq!(r.final_gap, 0.0);
        assert_eq!(r.completed_iterations, 0);
        assert_eq!(r.termination_reason, RULE_ITERATION_LIMIT);
        assert!(r.solver_stats_log.is_empty());
    }

    #[test]
    fn training_results_start_time_is_recent() {
        let r = TrainingResults::new(0);
        let elapsed = r.start_time.elapsed();
        assert!(elapsed < Duration::from_millis(1));
    }
}
