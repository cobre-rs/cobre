//! Convergence monitor for the SDDP training loop.
//!
//! [`ConvergenceMonitor`] tracks the lower bound (LB), upper bound (UB), gap,
//! and per-iteration history across training iterations, and evaluates the
//! configured stopping rules to determine when training should terminate.
//!
//! [`ConvergenceMonitor::upper_bound`] returns the raw per-iteration UB with
//! **no** exponential smoothing — a deliberate contract, not an oversight.
//!
//! ## Usage
//!
//! ```rust
//! use cobre_sddp::ConvergenceMonitor;
//! use cobre_sddp::SyncResult;
//! use cobre_sddp::{StoppingMode, StoppingRule, StoppingRuleSet};
//!
//! let rule_set = StoppingRuleSet {
//!     rules: vec![StoppingRule::IterationLimit { limit: 5 }],
//!     mode: StoppingMode::Any,
//! };
//!
//! let mut monitor = ConvergenceMonitor::new(rule_set);
//!
//! let sync = SyncResult {
//!     global_ub_mean: 110.0,
//!     global_ub_std: 5.0,
//!     ci_95_half_width: 2.0,
//!     sync_time_ms: 10,
//! };
//!
//! let decision = monitor.update(100.0, &sync, 0.0);
//! assert!(!decision.should_stop());
//! assert_eq!(monitor.iteration_count(), 1);
//! assert!((monitor.gap() - 10.0 / 100.0).abs() < 1e-10);
//! ```

use crate::{
    config::ShutdownSource,
    forward::SyncResult,
    stopping_rule::{MonitorState, StopDecision, StoppingRuleSet, relative_gap_denominator},
};

/// Tracks bound statistics and evaluates stopping rules across training
/// iterations.
///
/// Constructed once before the training loop begins. On each iteration, the
/// training loop calls [`ConvergenceMonitor::update`] with the latest LB and
/// UB statistics, which returns the termination decision.
#[derive(Debug)]
pub struct ConvergenceMonitor {
    rule_set: StoppingRuleSet,
    lower_bound: f64,
    upper_bound: f64,
    upper_bound_std: f64,
    ci_95_half_width: f64,
    gap: f64,
    lower_bound_history: Vec<f64>,
    iteration_count: u64,
    iteration_budget: u64,
    shutdown: Option<ShutdownSource>,
}

impl ConvergenceMonitor {
    /// Create a new convergence monitor with the given stopping rule set and no
    /// iteration budget.
    ///
    /// The lower-bound history grows on demand; [`Self::with_iteration_budget`]
    /// reserves it up front.
    #[must_use]
    pub fn new(rule_set: StoppingRuleSet) -> Self {
        Self {
            rule_set,
            lower_bound: 0.0,
            upper_bound: 0.0,
            upper_bound_std: 0.0,
            ci_95_half_width: 0.0,
            gap: 0.0,
            lower_bound_history: Vec::new(),
            iteration_count: 0,
            iteration_budget: u64::MAX,
            shutdown: None,
        }
    }

    /// Create a monitor for the run's absolute iteration budget: the decision
    /// at iteration `max_iterations` carries [`StopMask::BUDGET_EXHAUSTED`], and
    /// the lower-bound history is reserved so [`Self::update`] never reallocates
    /// it.
    ///
    /// A resumed run that restores earlier entries stays within the budget,
    /// because restored plus remaining iterations never exceed `max_iterations`.
    ///
    /// [`StopMask::BUDGET_EXHAUSTED`]: crate::StopMask::BUDGET_EXHAUSTED
    #[must_use]
    pub fn with_iteration_budget(rule_set: StoppingRuleSet, max_iterations: u64) -> Self {
        let mut monitor = Self::new(rule_set);
        monitor.iteration_budget = max_iterations;
        monitor
            .lower_bound_history
            .reserve_exact(usize::try_from(max_iterations).unwrap_or(0));
        monitor
    }

    /// Restore a resumed run's iteration counter and lower-bound history, so the
    /// next [`Self::update`] evaluates iteration `completed_iterations + 1`:
    /// `IterationLimit` and the budget fire at absolute iteration numbers and
    /// `BoundStalling` reads the window the earlier run recorded.
    ///
    /// `lower_bound_history` is the earlier run's series, oldest first. Only its
    /// first `completed_iterations` entries are restored; a shorter series is
    /// restored as recorded.
    ///
    /// Call it once, before the first `update`.
    pub fn resume_at(&mut self, completed_iterations: u64, lower_bound_history: &[f64]) {
        let committed = usize::try_from(completed_iterations)
            .unwrap_or(usize::MAX)
            .min(lower_bound_history.len());
        self.iteration_count = completed_iterations;
        self.lower_bound_history.clear();
        self.lower_bound_history
            .extend_from_slice(&lower_bound_history[..committed]);
    }

    /// Update bound statistics and evaluate stopping rules at
    /// `wall_time_seconds`, the elapsed training time the stop decision
    /// evaluates; under MPI the caller passes the value agreed across ranks.
    ///
    /// Returns the [`StopDecision`], with [`StopMask::SIGNAL`] added for a
    /// signal shutdown and [`StopMask::BUDGET_EXHAUSTED`] at the last budgeted
    /// iteration. Gap is normalized by the LOWER bound (`max(1.0, |LB|)`
    /// denominator, shared with the `Gap` stopping rule) to guard against
    /// division by zero.
    ///
    /// [`StopMask::SIGNAL`]: crate::StopMask::SIGNAL
    /// [`StopMask::BUDGET_EXHAUSTED`]: crate::StopMask::BUDGET_EXHAUSTED
    pub fn update(
        &mut self,
        lb: f64,
        sync_result: &SyncResult,
        wall_time_seconds: f64,
    ) -> StopDecision {
        self.lower_bound = lb;
        self.upper_bound = sync_result.global_ub_mean;
        self.upper_bound_std = sync_result.global_ub_std;
        self.ci_95_half_width = sync_result.ci_95_half_width;

        self.gap = (self.upper_bound - lb) / relative_gap_denominator(lb);

        self.iteration_count += 1;
        self.lower_bound_history.push(lb);

        // avoid cloning the growing history vec; restored below
        let history = std::mem::take(&mut self.lower_bound_history);
        let state = MonitorState {
            iteration: self.iteration_count,
            wall_time_seconds,
            lower_bound: self.lower_bound,
            upper_bound: self.upper_bound,
            lower_bound_history: history,
            shutdown_requested: self.shutdown.is_some(),
        };

        let mut decision = self.rule_set.evaluate(&state);
        self.lower_bound_history = state.lower_bound_history;
        if self.shutdown == Some(ShutdownSource::Signal) {
            decision = decision.with_signal_source();
        }
        if self.iteration_count >= self.iteration_budget {
            decision = decision.with_budget_exhausted();
        }
        decision
    }

    /// Record a shutdown request from `source`, keeping the stronger of it and
    /// any earlier request; the next [`ConvergenceMonitor::update`] returns a
    /// decision that stops and carries `StopMask::SHUTDOWN`.
    pub fn set_shutdown(&mut self, source: ShutdownSource) {
        self.shutdown = self.shutdown.max(Some(source));
    }

    /// Current lower bound.
    #[must_use]
    pub fn lower_bound(&self) -> f64 {
        self.lower_bound
    }

    /// Current upper bound mean from the latest forward pass.
    #[must_use]
    pub fn upper_bound(&self) -> f64 {
        self.upper_bound
    }

    /// Current upper bound standard deviation from the latest forward pass.
    #[must_use]
    pub fn upper_bound_std(&self) -> f64 {
        self.upper_bound_std
    }

    /// Current 95% confidence interval half-width (from latest forward pass).
    #[must_use]
    pub fn ci_95_half_width(&self) -> f64 {
        self.ci_95_half_width
    }

    /// Current convergence gap: `(UB - LB) / max(1.0, |LB|)`.
    #[must_use]
    pub fn gap(&self) -> f64 {
        self.gap
    }

    /// Number of completed update calls.
    #[must_use]
    pub fn iteration_count(&self) -> u64 {
        self.iteration_count
    }

    /// Lower bound recorded by each [`Self::update`], oldest first.
    #[must_use]
    pub fn lower_bound_history(&self) -> &[f64] {
        &self.lower_bound_history
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::ConvergenceMonitor;
    use crate::{
        config::ShutdownSource,
        forward::SyncResult,
        stopping_rule::{StopMask, StoppingMode, StoppingRule, StoppingRuleSet},
    };

    fn make_rule_set(rule: StoppingRule) -> StoppingRuleSet {
        StoppingRuleSet {
            rules: vec![rule],
            mode: StoppingMode::Any,
        }
    }

    fn make_sync(ub_mean: f64) -> SyncResult {
        SyncResult {
            global_ub_mean: ub_mean,
            global_ub_std: 5.0,
            ci_95_half_width: 2.0,
            sync_time_ms: 10,
        }
    }

    fn default_sync() -> SyncResult {
        make_sync(110.0)
    }

    #[test]
    fn new_initializes_all_fields_to_default() {
        let monitor =
            ConvergenceMonitor::new(make_rule_set(StoppingRule::IterationLimit { limit: 10 }));
        assert_eq!(monitor.lower_bound(), 0.0);
        assert_eq!(monitor.upper_bound(), 0.0);
        assert_eq!(monitor.upper_bound_std(), 0.0);
        assert_eq!(monitor.ci_95_half_width(), 0.0);
        assert_eq!(monitor.gap(), 0.0);
        assert_eq!(monitor.iteration_count(), 0);
    }

    #[test]
    fn with_iteration_budget_reserves_the_lower_bound_history() {
        let rule_set = make_rule_set(StoppingRule::IterationLimit { limit: 64 });
        let unreserved = ConvergenceMonitor::new(rule_set.clone());
        assert_eq!(unreserved.lower_bound_history.capacity(), 0);
        let reserved = ConvergenceMonitor::with_iteration_budget(rule_set, 64);
        assert!(reserved.lower_bound_history.capacity() >= 64);
        assert!(reserved.lower_bound_history.is_empty());
    }

    #[test]
    fn update_increments_iteration_count() {
        let mut monitor =
            ConvergenceMonitor::new(make_rule_set(StoppingRule::IterationLimit { limit: 100 }));
        monitor.update(100.0, &default_sync(), 0.0);
        assert_eq!(monitor.iteration_count(), 1);
        monitor.update(101.0, &default_sync(), 0.0);
        assert_eq!(monitor.iteration_count(), 2);
    }

    #[test]
    fn update_stores_lb_and_ub_correctly() {
        let mut monitor =
            ConvergenceMonitor::new(make_rule_set(StoppingRule::IterationLimit { limit: 100 }));
        let sync = SyncResult {
            global_ub_mean: 200.0,
            global_ub_std: 10.0,
            ci_95_half_width: 3.0,
            sync_time_ms: 5,
        };
        monitor.update(150.0, &sync, 0.0);
        assert!((monitor.lower_bound() - 150.0).abs() < 1e-10);
        assert!((monitor.upper_bound() - 200.0).abs() < 1e-10);
        assert!((monitor.upper_bound_std() - 10.0).abs() < 1e-10);
        assert!((monitor.ci_95_half_width() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn gap_formula_uses_max_guard() {
        // LB = 0.5 → denominator = max(1.0, |0.5|) = 1.0 (the lower-bound floor)
        // gap = (100.5 - 0.5) / 1.0 = 100.0
        let mut monitor =
            ConvergenceMonitor::new(make_rule_set(StoppingRule::IterationLimit { limit: 100 }));
        let sync = make_sync(100.5);
        monitor.update(0.5, &sync, 0.0);
        let expected = (100.5_f64 - 0.5) / 1.0_f64;
        assert!(
            (monitor.gap() - expected).abs() < 1e-10,
            "gap with LB=0.5 must use max guard of 1.0, got {}",
            monitor.gap()
        );
    }

    #[test]
    fn gap_formula_normal_case() {
        // UB = 110, LB = 100 → gap = (110 - 100) / max(1.0, 100.0) = 10/100
        let mut monitor =
            ConvergenceMonitor::new(make_rule_set(StoppingRule::IterationLimit { limit: 100 }));
        let sync = make_sync(110.0);
        monitor.update(100.0, &sync, 0.0);
        let expected = 10.0_f64 / 100.0_f64;
        assert!(
            (monitor.gap() - expected).abs() < 1e-10,
            "gap must be 10/100, got {}",
            monitor.gap()
        );
    }

    #[test]
    fn lower_bound_history_grows() {
        let mut monitor =
            ConvergenceMonitor::new(make_rule_set(StoppingRule::IterationLimit { limit: 100 }));
        for i in 0..5 {
            monitor.update(f64::from(i) * 10.0, &default_sync(), 0.0);
        }
        assert_eq!(monitor.lower_bound_history.len(), 5);
    }

    #[test]
    fn set_shutdown_reports_graceful_shutdown_under_any_mode() {
        let rule_set = StoppingRuleSet {
            rules: vec![StoppingRule::IterationLimit { limit: 100 }],
            mode: StoppingMode::Any,
        };
        let mut monitor = ConvergenceMonitor::new(rule_set);
        monitor.set_shutdown(ShutdownSource::Cooperative);
        let decision = monitor.update(100.0, &default_sync(), 0.0);
        assert!(decision.should_stop(), "should stop after shutdown signal");
        assert!(decision.mask().contains(StopMask::SHUTDOWN));
        assert!(!decision.mask().contains(StopMask::SIGNAL));
        assert_eq!(decision.termination_reason(), Some("graceful_shutdown"));
        assert!(decision.ended_by_shutdown());
    }

    #[test]
    fn gap_rule_evaluates_exact_gap_through_update() {
        let rule_set = StoppingRuleSet {
            rules: vec![StoppingRule::Gap {
                tolerance: Some(1000.0),
                relative_tolerance: None,
            }],
            mode: StoppingMode::Any,
        };
        let mut monitor = ConvergenceMonitor::new(rule_set);
        // update threads sync_result.global_ub_mean (110) as the upper bound;
        // gap = 110 - 80 = 30 <= 1000 → stop.
        let decision = monitor.update(80.0, &default_sync(), 0.0);
        assert!(
            decision.should_stop(),
            "gap 30 within tolerance 1000 must stop"
        );
        assert_eq!(decision.termination_reason(), Some("gap"));
        assert!(decision.mask().contains(StopMask::GAP));
    }

    #[test]
    fn gap_rule_does_not_stop_when_gap_exceeds_tolerance() {
        let rule_set = StoppingRuleSet {
            rules: vec![StoppingRule::Gap {
                tolerance: Some(10.0),
                relative_tolerance: None,
            }],
            mode: StoppingMode::Any,
        };
        let mut monitor = ConvergenceMonitor::new(rule_set);
        // gap = 110 - 80 = 30 > 10 → no stop.
        let decision = monitor.update(80.0, &default_sync(), 0.0);
        assert!(
            !decision.should_stop(),
            "gap 30 exceeds tolerance 10; must not stop"
        );
        assert!(!decision.mask().contains(StopMask::GAP));
    }

    #[test]
    fn iteration_limit_triggers_at_limit() {
        let mut monitor =
            ConvergenceMonitor::new(make_rule_set(StoppingRule::IterationLimit { limit: 3 }));
        let sync = default_sync();
        let decision1 = monitor.update(100.0, &sync, 0.0);
        let decision2 = monitor.update(100.0, &sync, 0.0);
        let decision3 = monitor.update(100.0, &sync, 0.0);
        assert!(!decision1.should_stop(), "should not stop at iteration 1");
        assert!(!decision2.should_stop(), "should not stop at iteration 2");
        assert!(
            decision3.should_stop(),
            "should stop at iteration 3 (limit reached)"
        );
        assert!(decision3.mask().contains(StopMask::ITERATION_LIMIT));
        assert_eq!(decision3.termination_reason(), Some("iteration_limit"));
    }

    #[test]
    fn time_limit_is_evaluated_from_the_supplied_elapsed_time() {
        let mut monitor =
            ConvergenceMonitor::new(make_rule_set(StoppingRule::TimeLimit { seconds: 10.0 }));
        let before = monitor.update(100.0, &default_sync(), 9.9);
        assert!(!before.should_stop(), "9.9 s is under the 10 s limit");
        assert!(!before.mask().contains(StopMask::TIME_LIMIT));

        let at_limit = monitor.update(100.0, &default_sync(), 10.0);
        assert!(at_limit.should_stop(), "10 s reaches the 10 s limit");
        assert!(at_limit.mask().contains(StopMask::TIME_LIMIT));
        assert_eq!(at_limit.termination_reason(), Some("time_limit"));
    }

    #[test]
    fn bound_stalling_triggers_when_stable() {
        let sync = default_sync();
        // 4 updates: history after each is [90], [90,99], [90,99,99.5], [90,99,99.5,100]
        // After 4th update: lb_window_start = history[4-3] = history[1] = 99.0
        // Δ = (100 - 99) / max(1, 100) = 1/100 = 0.01 → NOT triggered (tolerance is strict <)
        // Use tolerance=0.011 to trigger
        let rule_set = StoppingRuleSet {
            rules: vec![StoppingRule::BoundStalling {
                tolerance: 0.011,
                iterations: 3,
            }],
            mode: StoppingMode::Any,
        };
        let mut monitor2 = ConvergenceMonitor::new(rule_set);
        monitor2.update(90.0, &sync, 0.0);
        monitor2.update(99.0, &sync, 0.0);
        monitor2.update(99.5, &sync, 0.0);
        let decision = monitor2.update(100.0, &sync, 0.0);
        assert!(
            decision.should_stop(),
            "BoundStalling should trigger when improvement is < 0.011"
        );
        // Also verify gap on the last iteration: (110 - 100) / 100 = 10/100
        assert!(
            (monitor2.gap() - 10.0 / 100.0).abs() < 1e-10,
            "gap after 4th update must equal 10/100, got {}",
            monitor2.gap()
        );
    }

    #[test]
    fn ac_iteration_limit_triggers_at_third_call() {
        let rule_set = StoppingRuleSet {
            rules: vec![StoppingRule::IterationLimit { limit: 3 }],
            mode: StoppingMode::Any,
        };
        let mut monitor = ConvergenceMonitor::new(rule_set);
        let sync = SyncResult {
            global_ub_mean: 110.0,
            global_ub_std: 5.0,
            ci_95_half_width: 2.0,
            sync_time_ms: 10,
        };
        monitor.update(100.0, &sync, 0.0);
        monitor.update(100.0, &sync, 0.0);
        let decision = monitor.update(100.0, &sync, 0.0);
        assert!(
            decision.should_stop(),
            "third update must trigger IterationLimit(3)"
        );
        assert!(decision.mask().contains(StopMask::ITERATION_LIMIT));
        assert_eq!(decision.termination_reason(), Some("iteration_limit"));
    }

    #[test]
    fn ac_gap_formula_with_ub_110_lb_100() {
        let mut monitor =
            ConvergenceMonitor::new(make_rule_set(StoppingRule::IterationLimit { limit: 100 }));
        let sync = SyncResult {
            global_ub_mean: 110.0,
            global_ub_std: 5.0,
            ci_95_half_width: 2.0,
            sync_time_ms: 10,
        };
        // 4 updates simulating BoundStalling AC scenario
        monitor.update(90.0, &sync, 0.0);
        monitor.update(99.0, &sync, 0.0);
        monitor.update(99.5, &sync, 0.0);
        monitor.update(100.0, &sync, 0.0);
        let expected = 10.0_f64 / 100.0_f64;
        assert!(
            (monitor.gap() - expected).abs() < 1e-10,
            "gap must equal {expected}, got {}",
            monitor.gap()
        );
    }

    #[test]
    fn set_shutdown_reports_graceful_shutdown_under_all_mode() {
        let rule_set = StoppingRuleSet {
            rules: vec![StoppingRule::IterationLimit { limit: 100 }],
            mode: StoppingMode::All,
        };
        let mut monitor = ConvergenceMonitor::new(rule_set);
        monitor.set_shutdown(ShutdownSource::Cooperative);
        let decision = monitor.update(100.0, &default_sync(), 0.0);
        assert!(decision.should_stop());
        assert!(decision.mask().contains(StopMask::SHUTDOWN));
        assert_eq!(decision.termination_reason(), Some("graceful_shutdown"));
        assert!(decision.ended_by_shutdown());
    }

    #[test]
    fn ac_lb_and_iteration_count_track_correctly() {
        let mut monitor =
            ConvergenceMonitor::new(make_rule_set(StoppingRule::IterationLimit { limit: 100 }));
        monitor.update(50.0, &default_sync(), 0.0);
        monitor.update(60.0, &default_sync(), 0.0);
        assert!(
            (monitor.lower_bound() - 60.0).abs() < 1e-10,
            "lower_bound must return latest LB 60.0, got {}",
            monitor.lower_bound()
        );
        assert_eq!(monitor.iteration_count(), 2);
    }

    #[test]
    fn exhausted_budget_wins_over_a_signal_shutdown() {
        let mut monitor = ConvergenceMonitor::with_iteration_budget(
            make_rule_set(StoppingRule::IterationLimit { limit: 100 }),
            2,
        );
        assert!(!monitor.update(100.0, &default_sync(), 0.0).should_stop());

        monitor.set_shutdown(ShutdownSource::Signal);
        let decision = monitor.update(100.0, &default_sync(), 0.0);
        assert!(!decision.configured_stop());
        assert!(!decision.mask().contains(StopMask::ITERATION_LIMIT));
        assert!(decision.mask().contains(StopMask::BUDGET_EXHAUSTED));
        assert!(decision.mask().contains(StopMask::SHUTDOWN));
        assert!(decision.mask().contains(StopMask::SIGNAL));
        assert_eq!(decision.termination_reason(), Some("iteration_limit"));
        assert!(!decision.ended_by_shutdown());
    }

    #[test]
    fn exhausted_budget_without_a_configured_stop_reports_iteration_limit() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 3 },
                StoppingRule::BoundStalling {
                    tolerance: 1e-12,
                    iterations: 50,
                },
            ],
            mode: StoppingMode::All,
        };
        let mut monitor = ConvergenceMonitor::with_iteration_budget(rule_set, 3);
        for iteration in 1..=2 {
            let decision = monitor.update(100.0, &default_sync(), 0.0);
            assert!(!decision.should_stop(), "iteration {iteration}");
            assert!(!decision.mask().contains(StopMask::BUDGET_EXHAUSTED));
        }

        let decision = monitor.update(100.0, &default_sync(), 0.0);
        assert!(!decision.configured_stop());
        assert!(decision.mask().contains(StopMask::BUDGET_EXHAUSTED));
        assert!(!decision.mask().contains(StopMask::SHUTDOWN));
        assert!(decision.should_stop());
        assert_eq!(decision.termination_reason(), Some("iteration_limit"));
        assert!(!decision.ended_by_shutdown());
    }

    #[test]
    fn monitor_without_a_budget_never_reports_one_exhausted() {
        let mut monitor =
            ConvergenceMonitor::new(make_rule_set(StoppingRule::TimeLimit { seconds: 1e9 }));
        for _ in 0..5 {
            let decision = monitor.update(100.0, &default_sync(), 0.0);
            assert!(!decision.mask().contains(StopMask::BUDGET_EXHAUSTED));
            assert!(!decision.should_stop());
        }
    }

    #[test]
    fn resumed_monitor_fires_the_iteration_limit_at_the_absolute_iteration() {
        let mut monitor = ConvergenceMonitor::with_iteration_budget(
            make_rule_set(StoppingRule::IterationLimit { limit: 7 }),
            7,
        );
        monitor.resume_at(5, &[]);

        let first = monitor.update(100.0, &default_sync(), 0.0);
        assert_eq!(monitor.iteration_count(), 6);
        assert!(!first.should_stop());
        assert_eq!(first.termination_reason(), None);

        let second = monitor.update(100.0, &default_sync(), 0.0);
        assert_eq!(monitor.iteration_count(), 7);
        assert!(second.configured_stop());
        assert!(second.mask().contains(StopMask::ITERATION_LIMIT));
        assert_eq!(second.termination_reason(), Some("iteration_limit"));
    }

    #[test]
    fn resumed_monitor_restores_the_recorded_lower_bound_series() {
        let mut monitor = ConvergenceMonitor::with_iteration_budget(
            make_rule_set(StoppingRule::IterationLimit { limit: 8 }),
            8,
        );
        monitor.resume_at(2, &[10.0, 12.0]);
        assert_eq!(monitor.lower_bound_history(), [10.0, 12.0]);

        monitor.update(13.0, &default_sync(), 0.0);

        assert_eq!(monitor.lower_bound_history(), [10.0, 12.0, 13.0]);
        assert_eq!(monitor.iteration_count(), 3);
    }

    #[test]
    fn resumed_monitor_drops_entries_beyond_the_committed_iterations() {
        let mut monitor = ConvergenceMonitor::with_iteration_budget(
            make_rule_set(StoppingRule::IterationLimit { limit: 8 }),
            8,
        );
        monitor.resume_at(2, &[10.0, 12.0, 14.0]);

        assert_eq!(monitor.lower_bound_history(), [10.0, 12.0]);
        assert_eq!(monitor.iteration_count(), 2);
    }

    #[test]
    fn cooperative_shutdown_after_a_signal_keeps_the_signal_source() {
        let mut monitor =
            ConvergenceMonitor::new(make_rule_set(StoppingRule::IterationLimit { limit: 100 }));
        monitor.set_shutdown(ShutdownSource::Signal);
        monitor.set_shutdown(ShutdownSource::Cooperative);
        let decision = monitor.update(100.0, &default_sync(), 0.0);
        assert!(decision.mask().contains(StopMask::SHUTDOWN));
        assert!(decision.mask().contains(StopMask::SIGNAL));
        assert_eq!(decision.termination_reason(), Some("graceful_shutdown"));
        assert!(decision.ended_by_shutdown());
    }
}
