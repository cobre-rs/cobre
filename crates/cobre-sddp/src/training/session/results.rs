//! Result-accumulator sub-struct for one training run.

use std::time::Instant;

use crate::solver_stats::SolverStatsLogEntry;
use crate::stopping_rule::{RULE_ITERATION_LIMIT, StopDecision};

/// Accumulates per-iteration results for one training run.
pub(crate) struct TrainingResults {
    pub final_lb: f64,
    pub final_ub: f64,
    pub final_ub_std: f64,
    pub final_gap: f64,
    pub completed_iterations: u64,
    pub termination_reason: String,
    pub stop_decision: StopDecision,
    pub solver_stats_log: Vec<SolverStatsLogEntry>,
    pub start_time: Instant,
}

impl TrainingResults {
    /// Initialise all result accumulators for a training run starting at
    /// `start_iteration`.
    pub(crate) fn new(start_iteration: u64) -> Self {
        Self {
            start_time: Instant::now(),
            final_lb: 0.0,
            final_ub: 0.0,
            final_ub_std: 0.0,
            final_gap: 0.0,
            completed_iterations: start_iteration,
            termination_reason: RULE_ITERATION_LIMIT.to_string(),
            stop_decision: StopDecision::default(),
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
    use crate::stopping_rule::{RULE_ITERATION_LIMIT, StopDecision};

    #[test]
    fn training_results_new_initialises_all_fields() {
        let r = TrainingResults::new(0);
        assert_eq!(r.final_lb, 0.0);
        assert_eq!(r.final_ub, 0.0);
        assert_eq!(r.final_ub_std, 0.0);
        assert_eq!(r.final_gap, 0.0);
        assert_eq!(r.completed_iterations, 0);
        assert_eq!(r.termination_reason, RULE_ITERATION_LIMIT);
        assert_eq!(r.stop_decision, StopDecision::default());
        assert!(r.solver_stats_log.is_empty());
    }

    #[test]
    fn training_results_start_time_is_recent() {
        let r = TrainingResults::new(0);
        let elapsed = r.start_time.elapsed();
        assert!(elapsed < Duration::from_millis(1));
    }
}
