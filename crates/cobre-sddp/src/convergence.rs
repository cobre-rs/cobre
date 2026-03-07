//! Convergence monitor for the SDDP training loop.
//!
//! [`ConvergenceMonitor`] tracks the lower bound (LB), upper bound (UB), gap,
//! and per-iteration history across training iterations, and evaluates the
//! configured stopping rules to determine when training should terminate.
//!
//! ## Design
//!
//! The monitor is a pure computation component: it receives bound values as
//! inputs and produces termination decisions as outputs. It does not run
//! simulations, emit events, or perform checkpointing — those responsibilities
//! belong to the training loop orchestrator.
//!
//! The LB is received as a separate scalar from [`crate::lower_bound::evaluate_lower_bound`]
//! (evaluated after the backward pass). It is **not** derived from the forward
//! synchronisation step. The UB statistics come from [`crate::forward::SyncResult`]
//! produced by [`crate::forward::sync_forward`].
//!
//! ## Gap formula
//!
//! The convergence gap is computed as:
//!
//! `gap = (UB - LB) / max(1.0, |UB|)`
//!
//! The `max(1.0, |UB|)` guard prevents division by zero when the UB is near
//! zero (F-004 resolution).
//!
//! ## Usage
//!
//! ```rust
//! use cobre_sddp::convergence::ConvergenceMonitor;
//! use cobre_sddp::forward::SyncResult;
//! use cobre_sddp::stopping_rule::{StoppingMode, StoppingRule, StoppingRuleSet};
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
//! let (stop, results) = monitor.update(100.0, &sync);
//! assert!(!stop);
//! assert_eq!(monitor.iteration_count(), 1);
//! assert!((monitor.gap() - 10.0 / 110.0).abs() < 1e-10);
//! ```

use std::time::Instant;

use cobre_core::StoppingRuleResult;

use crate::{
    forward::SyncResult,
    stopping_rule::{MonitorState, StoppingRuleSet},
};

// ---------------------------------------------------------------------------
// ConvergenceMonitor
// ---------------------------------------------------------------------------

/// Tracks bound statistics and evaluates stopping rules across training
/// iterations.
///
/// Constructed once before the training loop begins. On each iteration, the
/// training loop calls [`ConvergenceMonitor::update`] with the latest LB and
/// UB statistics, which returns the termination decision.
///
/// ## Fields (private)
///
/// - `rule_set` — the configured stopping rules and combination mode.
/// - `lower_bound` — latest LB value (0.0 before first update).
/// - `upper_bound` — latest UB mean (0.0 before first update).
/// - `upper_bound_std` — latest UB standard deviation.
/// - `ci_95_half_width` — latest 95% CI half-width.
/// - `gap` — latest convergence gap: `(UB - LB) / max(1.0, |UB|)`.
/// - `lower_bound_history` — all LB values in chronological order.
/// - `iteration_count` — 0-based counter, incremented by each `update` call.
/// - `start_time` — wall-clock origin set at construction.
/// - `shutdown_requested` — set by [`ConvergenceMonitor::set_shutdown`].
/// - `simulation_costs` — set by [`ConvergenceMonitor::set_simulation_costs`].
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
    start_time: Instant,
    shutdown_requested: bool,
    simulation_costs: Option<Vec<f64>>,
}

impl ConvergenceMonitor {
    /// Create a new convergence monitor with the given stopping rule set.
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
            start_time: Instant::now(),
            shutdown_requested: false,
            simulation_costs: None,
        }
    }

    /// Update bound statistics and evaluate stopping rules.
    ///
    /// Incorporates the latest lower bound and forward-pass UB statistics,
    /// increments the iteration counter, appends `lb` to the history, and
    /// evaluates the configured stopping rules via [`StoppingRuleSet::evaluate`].
    ///
    /// Returns `(should_stop, results)` where:
    /// - `should_stop` is the combined termination decision.
    /// - `results` lists the evaluation result for every configured rule.
    ///
    /// This method is infallible. Gap computation uses `max(1.0, |UB|)` in the
    /// denominator to guard against division by zero.
    ///
    /// # Arguments
    ///
    /// - `lb` — lower bound from [`crate::lower_bound::evaluate_lower_bound`],
    ///   evaluated after the backward pass.
    /// - `sync_result` — global UB statistics from
    ///   [`crate::forward::sync_forward`], evaluated after the forward pass.
    pub fn update(&mut self, lb: f64, sync_result: &SyncResult) -> (bool, Vec<StoppingRuleResult>) {
        self.lower_bound = lb;
        self.upper_bound = sync_result.global_ub_mean;
        self.upper_bound_std = sync_result.global_ub_std;
        self.ci_95_half_width = sync_result.ci_95_half_width;

        // gap = (UB - LB) / max(1.0, |UB|)  — F-004 resolution
        let denominator = self.upper_bound.abs().max(1.0_f64);
        self.gap = (self.upper_bound - lb) / denominator;

        self.iteration_count += 1;
        self.lower_bound_history.push(lb);

        let state = MonitorState {
            iteration: self.iteration_count,
            wall_time_seconds: self.start_time.elapsed().as_secs_f64(),
            lower_bound: self.lower_bound,
            lower_bound_history: self.lower_bound_history.clone(),
            shutdown_requested: self.shutdown_requested,
            simulation_costs: self.simulation_costs.clone(),
        };

        self.rule_set.evaluate(&state)
    }

    /// Signal a graceful shutdown request.
    ///
    /// After this call, the next [`ConvergenceMonitor::update`] will return
    /// `(true, results)` with the `GracefulShutdown` rule reporting
    /// `triggered: true`.
    pub fn set_shutdown(&mut self) {
        self.shutdown_requested = true;
    }

    /// Provide simulation costs for the [`crate::stopping_rule::StoppingRule::SimulationBased`] rule.
    ///
    /// The training loop calls this before [`ConvergenceMonitor::update`] on
    /// check iterations where a Monte Carlo simulation has been run. The costs
    /// are forwarded into [`MonitorState::simulation_costs`] during the next
    /// `update` call.
    pub fn set_simulation_costs(&mut self, costs: Vec<f64>) {
        self.simulation_costs = Some(costs);
    }

    /// Current lower bound.
    #[must_use]
    pub fn lower_bound(&self) -> f64 {
        self.lower_bound
    }

    /// Current upper bound mean.
    #[must_use]
    pub fn upper_bound(&self) -> f64 {
        self.upper_bound
    }

    /// Current upper bound standard deviation.
    #[must_use]
    pub fn upper_bound_std(&self) -> f64 {
        self.upper_bound_std
    }

    /// Current 95% confidence interval half-width.
    #[must_use]
    pub fn ci_95_half_width(&self) -> f64 {
        self.ci_95_half_width
    }

    /// Current convergence gap: `(UB - LB) / max(1.0, |UB|)`.
    #[must_use]
    pub fn gap(&self) -> f64 {
        self.gap
    }

    /// Number of completed update calls.
    #[must_use]
    pub fn iteration_count(&self) -> u64 {
        self.iteration_count
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::ConvergenceMonitor;
    use crate::{
        forward::SyncResult,
        stopping_rule::{StoppingMode, StoppingRule, StoppingRuleSet},
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
    fn update_increments_iteration_count() {
        let mut monitor =
            ConvergenceMonitor::new(make_rule_set(StoppingRule::IterationLimit { limit: 100 }));
        monitor.update(100.0, &default_sync());
        assert_eq!(monitor.iteration_count(), 1);
        monitor.update(101.0, &default_sync());
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
        monitor.update(150.0, &sync);
        assert!((monitor.lower_bound() - 150.0).abs() < 1e-10);
        assert!((monitor.upper_bound() - 200.0).abs() < 1e-10);
        assert!((monitor.upper_bound_std() - 10.0).abs() < 1e-10);
        assert!((monitor.ci_95_half_width() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn gap_formula_uses_max_guard() {
        // UB = 0.5 → denominator = max(1.0, 0.5) = 1.0
        // gap = (0.5 - 100.0) / 1.0 = -99.5
        let mut monitor =
            ConvergenceMonitor::new(make_rule_set(StoppingRule::IterationLimit { limit: 100 }));
        let sync = make_sync(0.5);
        monitor.update(100.0, &sync);
        let expected = (0.5_f64 - 100.0) / 1.0_f64;
        assert!(
            (monitor.gap() - expected).abs() < 1e-10,
            "gap with UB=0.5 must use max guard of 1.0, got {}",
            monitor.gap()
        );
    }

    #[test]
    fn gap_formula_normal_case() {
        // UB = 110, LB = 100 → gap = (110 - 100) / max(1.0, 110.0) = 10/110
        let mut monitor =
            ConvergenceMonitor::new(make_rule_set(StoppingRule::IterationLimit { limit: 100 }));
        let sync = make_sync(110.0);
        monitor.update(100.0, &sync);
        let expected = 10.0_f64 / 110.0_f64;
        assert!(
            (monitor.gap() - expected).abs() < 1e-10,
            "gap must be 10/110, got {}",
            monitor.gap()
        );
    }

    #[test]
    fn lower_bound_history_grows() {
        let mut monitor =
            ConvergenceMonitor::new(make_rule_set(StoppingRule::IterationLimit { limit: 100 }));
        for i in 0..5 {
            monitor.update(f64::from(i) * 10.0, &default_sync());
        }
        assert_eq!(monitor.lower_bound_history.len(), 5);
    }

    #[test]
    fn set_shutdown_triggers_graceful_rule() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::GracefulShutdown,
                StoppingRule::IterationLimit { limit: 100 },
            ],
            mode: StoppingMode::Any,
        };
        let mut monitor = ConvergenceMonitor::new(rule_set);
        monitor.set_shutdown();
        let (stop, results) = monitor.update(100.0, &default_sync());
        assert!(stop, "should stop after shutdown signal");
        // GracefulShutdown is results[0]
        assert!(
            results[0].triggered,
            "GracefulShutdown result must be triggered"
        );
        assert_eq!(results[0].rule_name, "graceful_shutdown");
    }

    #[test]
    fn set_simulation_costs_populates_monitor_state() {
        // Use a SimulationBased rule evaluated at period=1.
        // Provide simulation costs and verify the rule reaches evaluation
        // (i.e., costs pass through to MonitorState).
        let rule_set = StoppingRuleSet {
            rules: vec![StoppingRule::SimulationBased {
                period: 1,
                distance_tolerance: 1e6, // always trigger if costs present
                replications: 10,
                bound_stability_window: 1,
            }],
            mode: StoppingMode::Any,
        };
        let mut monitor = ConvergenceMonitor::new(rule_set);
        monitor.set_simulation_costs(vec![100.0, 200.0, 300.0]);
        let (_stop, results) = monitor.update(80.0, &default_sync());
        // The SimulationBased rule must have received the costs (it evaluates at
        // iteration 1 which is divisible by period=1) and reached the distance check.
        assert_eq!(results[0].rule_name, "simulation_based");
        // costs are present → detail must NOT contain "no simulation results available"
        assert!(
            !results[0]
                .detail
                .contains("no simulation results available"),
            "detail should not indicate missing costs: {}",
            results[0].detail
        );
    }

    #[test]
    fn iteration_limit_triggers_at_limit() {
        let mut monitor =
            ConvergenceMonitor::new(make_rule_set(StoppingRule::IterationLimit { limit: 3 }));
        let sync = default_sync();
        let (stop1, _) = monitor.update(100.0, &sync);
        let (stop2, _) = monitor.update(100.0, &sync);
        let (stop3, results) = monitor.update(100.0, &sync);
        assert!(!stop1, "should not stop at iteration 1");
        assert!(!stop2, "should not stop at iteration 2");
        assert!(stop3, "should stop at iteration 3 (limit reached)");
        assert!(results[0].triggered);
        assert_eq!(results[0].rule_name, "iteration_limit");
    }

    #[test]
    fn bound_stalling_triggers_when_stable() {
        let monitor = ConvergenceMonitor::new(make_rule_set(StoppingRule::BoundStalling {
            tolerance: 0.01,
            iterations: 3,
        }));
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
        let (_, _) = monitor2.update(90.0, &sync);
        let (_, _) = monitor2.update(99.0, &sync);
        let (_, _) = monitor2.update(99.5, &sync);
        let (stop, _) = monitor2.update(100.0, &sync);
        assert!(
            stop,
            "BoundStalling should trigger when improvement is < 0.011"
        );
        // Also verify gap on the last iteration: (110 - 100) / 110 = 10/110
        assert!(
            (monitor2.gap() - 10.0 / 110.0).abs() < 1e-10,
            "gap after 4th update must equal 10/110, got {}",
            monitor2.gap()
        );
        let _ = monitor; // suppress unused warning
    }

    /// AC: IterationLimit(3) in Any mode triggers at the third update.
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
        monitor.update(100.0, &sync);
        monitor.update(100.0, &sync);
        let (stop, results) = monitor.update(100.0, &sync);
        assert!(stop, "third update must trigger IterationLimit(3)");
        assert!(results[0].triggered);
        assert_eq!(results[0].rule_name, "iteration_limit");
    }

    /// AC: gap formula uses |UB| denominator; with UB=110, LB=100 → gap=10/110.
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
        monitor.update(90.0, &sync);
        monitor.update(99.0, &sync);
        monitor.update(99.5, &sync);
        monitor.update(100.0, &sync);
        let expected = 10.0_f64 / 110.0_f64;
        assert!(
            (monitor.gap() - expected).abs() < 1e-10,
            "gap must equal {expected}, got {}",
            monitor.gap()
        );
    }

    /// AC: `set_shutdown` causes `GracefulShutdown` to trigger on next update.
    #[test]
    fn ac_set_shutdown_triggers_graceful_shutdown_rule() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::GracefulShutdown,
                StoppingRule::IterationLimit { limit: 100 },
            ],
            mode: StoppingMode::Any,
        };
        let mut monitor = ConvergenceMonitor::new(rule_set);
        monitor.set_shutdown();
        let (stop, results) = monitor.update(100.0, &default_sync());
        assert!(stop);
        // GracefulShutdown is at index 0
        assert!(results[0].triggered);
        assert_eq!(results[0].rule_name, "graceful_shutdown");
    }

    /// AC: `lower_bound` and `iteration_count` track correctly after 2 updates.
    #[test]
    fn ac_lb_and_iteration_count_track_correctly() {
        let mut monitor =
            ConvergenceMonitor::new(make_rule_set(StoppingRule::IterationLimit { limit: 100 }));
        monitor.update(50.0, &default_sync());
        monitor.update(60.0, &default_sync());
        assert!(
            (monitor.lower_bound() - 60.0).abs() < 1e-10,
            "lower_bound must return latest LB 60.0, got {}",
            monitor.lower_bound()
        );
        assert_eq!(monitor.iteration_count(), 2);
    }
}
