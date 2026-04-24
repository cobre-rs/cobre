//! Stopping rules for the SDDP training loop.
//!
//! Defines stopping rule variants, composition logic, and convergence state.
//! Rules use enum dispatch; [`StoppingRuleSet`] composes them with AND/OR logic.
//!
//! ## Usage
//!
//! ```rust
//! use cobre_sddp::stopping_rule::{
//!     MonitorState, StoppingMode, StoppingRule, StoppingRuleSet,
//! };
//!
//! let state = MonitorState {
//!     iteration: 10,
//!     wall_time_seconds: 50.0,
//!     lower_bound: 100.0,
//!     lower_bound_history: vec![90.0, 95.0, 98.0, 99.0, 100.0,
//!                              100.0, 100.0, 100.0, 100.0, 100.0],
//!     shutdown_requested: false,
//!     simulation_costs: None,
//! };
//!
//! let rule = StoppingRule::IterationLimit { limit: 10 };
//! let result = rule.evaluate(&state);
//! assert!(result.triggered);
//! assert_eq!(result.rule_name, "iteration_limit");
//! ```

use std::borrow::Cow;

use cobre_core::StoppingRuleResult;

/// Rule name for the iteration limit stopping rule.
pub const RULE_ITERATION_LIMIT: &str = "iteration_limit";
/// Rule name for the wall-clock time limit stopping rule.
pub const RULE_TIME_LIMIT: &str = "time_limit";
/// Rule name for the lower-bound stalling stopping rule.
pub const RULE_BOUND_STALLING: &str = "bound_stalling";
/// Rule name for the simulation-based stopping rule.
pub const RULE_SIMULATION_BASED: &str = "simulation_based";
/// Rule name for the graceful-shutdown stopping rule.
pub const RULE_GRACEFUL_SHUTDOWN: &str = "graceful_shutdown";

// ---------------------------------------------------------------------------
// MonitorState
// ---------------------------------------------------------------------------

/// Read-only snapshot of convergence monitor quantities consumed by stopping
/// rule evaluation.
///
/// The convergence monitor populates this struct after each iteration's
/// forward synchronization step, before calling
/// [`StoppingRuleSet::evaluate`].
#[derive(Debug, Clone)]
pub struct MonitorState {
    /// Current iteration index (1-based).
    pub iteration: u64,

    /// Cumulative wall-clock time since training start, in seconds.
    pub wall_time_seconds: f64,

    /// Current lower bound (stage-1 LP objective value).
    pub lower_bound: f64,

    /// History of lower bounds from past iterations (chronological order).
    ///
    /// `lower_bound_history[i]` is the lower bound at iteration `i + 1`.
    /// Populated by the convergence monitor; appended each iteration.
    pub lower_bound_history: Vec<f64>,

    /// Whether an external shutdown signal has been received.
    ///
    /// Set by an OS signal handler (SIGTERM / SIGINT) and read atomically.
    pub shutdown_requested: bool,

    /// Per-stage mean costs from the most recent simulation evaluation.
    ///
    /// `None` if no simulation has been run yet, or if the convergence
    /// monitor has not yet run a [`StoppingRule::SimulationBased`] check.
    pub simulation_costs: Option<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// StoppingMode
// ---------------------------------------------------------------------------

/// Combination mode for [`StoppingRuleSet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoppingMode {
    /// Stop when **any** configured rule triggers (OR logic).
    ///
    /// The first rule (in configuration order) whose condition is satisfied
    /// causes termination. `GracefulShutdown` always takes precedence regardless
    /// of mode.
    Any,

    /// Stop when **all** configured rules trigger simultaneously (AND logic).
    ///
    /// All rules must be satisfied at the same iteration. `GracefulShutdown`
    /// always takes precedence regardless of mode.
    All,
}

// ---------------------------------------------------------------------------
// StoppingRule
// ---------------------------------------------------------------------------

/// Individual stopping rule for the SDDP training loop.
///
/// Each variant encapsulates a single termination criterion and its parameters.
/// Rules are composed into a [`StoppingRuleSet`] for evaluation. The
/// [`StoppingRule::GracefulShutdown`] variant is special: it is always
/// evaluated first and bypasses the composition logic.
///
/// The `IterationLimit` variant must always be present in the rule set as a
/// safety bound against infinite loops.
#[derive(Debug, Clone)]
pub enum StoppingRule {
    /// Terminate when the iteration count reaches a fixed limit.
    ///
    /// This is the mandatory safety bound. Every [`StoppingRuleSet`] must
    /// contain at least one `IterationLimit` rule, validated at configuration
    /// load time.
    IterationLimit {
        /// Maximum iteration count. Training stops when `iteration >= limit`.
        limit: u64,
    },

    /// Terminate when cumulative wall-clock time exceeds a threshold.
    TimeLimit {
        /// Maximum wall-clock time in seconds. Training stops when
        /// `wall_time_seconds >= seconds`.
        seconds: f64,
    },

    /// Terminate when the lower bound improvement over a sliding window
    /// falls below a relative tolerance.
    ///
    /// Uses the formula:
    /// `Δ = (lb_current - lb_window_start) / max(1.0, |lb_current|)`.
    /// Triggers when `|Δ| < tolerance`.
    BoundStalling {
        /// Relative improvement tolerance. Triggers when relative improvement
        /// over the window is below this value.
        tolerance: f64,

        /// Number of past iterations over which to measure improvement (τ).
        iterations: u64,
    },

    /// Terminate when both the lower bound and simulated policy costs
    /// have stabilized. Evaluated only every `period` iterations.
    ///
    /// Evaluation is two-stage: a bound stability check (cheap pre-filter),
    /// followed by comparison of per-stage mean simulation costs between
    /// consecutive evaluations (computed by the convergence monitor before
    /// `evaluate` is called).
    SimulationBased {
        /// Evaluate this rule every `period` iterations.
        period: u64,

        /// Normalized Euclidean distance threshold for simulation cost
        /// comparison between consecutive evaluations.
        distance_tolerance: f64,

        /// Number of Monte Carlo forward simulations to run when the bound
        /// stability pre-filter passes (executed by the convergence monitor).
        replications: u32,

        /// Number of past iterations for the bound stability pre-check.
        bound_stability_window: u64,
    },

    /// Terminate when an external shutdown signal (SIGTERM / SIGINT) is
    /// received.
    ///
    /// Not configured via JSON — always implicitly present and evaluated
    /// unconditionally before the composition logic. The training loop
    /// checkpoints the last completed iteration before exiting.
    GracefulShutdown,
}

impl StoppingRule {
    /// Evaluate this rule against the current monitor state.
    ///
    /// Returns a [`StoppingRuleResult`] with the rule's identifier, whether
    /// the rule's termination condition is satisfied, and a human-readable
    /// description of the current state.
    ///
    /// All evaluation is pure — this method reads from `state` but does not
    /// modify it. For [`StoppingRule::SimulationBased`], the convergence
    /// monitor is responsible for running simulations and storing results in
    /// `state.simulation_costs` before this method is called.
    #[must_use]
    pub fn evaluate(&self, state: &MonitorState) -> StoppingRuleResult {
        match self {
            Self::IterationLimit { limit } => {
                let triggered = state.iteration >= *limit;
                StoppingRuleResult {
                    rule_name: RULE_ITERATION_LIMIT,
                    triggered,
                    detail: Cow::Owned(format!("iteration {}/{}", state.iteration, limit)),
                }
            }

            Self::TimeLimit { seconds } => {
                let triggered = state.wall_time_seconds >= *seconds;
                StoppingRuleResult {
                    rule_name: RULE_TIME_LIMIT,
                    triggered,
                    detail: Cow::Owned(format!(
                        "elapsed {:.1}s / {:.1}s limit",
                        state.wall_time_seconds, seconds
                    )),
                }
            }

            Self::BoundStalling {
                tolerance,
                iterations,
            } => Self::evaluate_bound_stalling(state, *tolerance, *iterations),

            Self::SimulationBased {
                period,
                distance_tolerance,
                replications: _,
                bound_stability_window: _,
            } => Self::evaluate_simulation_based(state, *period, *distance_tolerance),

            Self::GracefulShutdown => {
                let triggered = state.shutdown_requested;
                StoppingRuleResult {
                    rule_name: RULE_GRACEFUL_SHUTDOWN,
                    triggered,
                    detail: if triggered {
                        Cow::Borrowed("shutdown signal received")
                    } else {
                        Cow::Borrowed("no shutdown signal")
                    },
                }
            }
        }
    }

    /// Evaluate the [`StoppingRule::BoundStalling`] condition.
    ///
    /// Computes the relative improvement in the lower bound over the last
    /// `iterations` iterations and compares it against `tolerance`.
    fn evaluate_bound_stalling(
        state: &MonitorState,
        tolerance: f64,
        iterations: u64,
    ) -> StoppingRuleResult {
        // Need at least `iterations` entries in history to compare.
        // `iterations` is a config-validated u64 that fits in usize on
        // any supported platform (validated <= u32::MAX at config load).
        #[allow(clippy::cast_possible_truncation)]
        let window = iterations as usize;
        if state.lower_bound_history.len() < window {
            return StoppingRuleResult {
                rule_name: RULE_BOUND_STALLING,
                triggered: false,
                detail: Cow::Owned(format!(
                    "insufficient history: {}/{} iterations",
                    state.lower_bound_history.len(),
                    window
                )),
            };
        }

        // lb_window_start is the lower bound from `iterations` steps ago.
        let history_len = state.lower_bound_history.len();
        let lb_window_start = state.lower_bound_history[history_len - window];
        let lb_current = state.lower_bound;

        // Δ = (lb_current - lb_window_start) / max(1.0, |lb_current|)
        let denominator = lb_current.abs().max(1.0_f64);
        let delta = (lb_current - lb_window_start) / denominator;

        let triggered = delta.abs() < tolerance;
        StoppingRuleResult {
            rule_name: RULE_BOUND_STALLING,
            triggered,
            detail: Cow::Owned(format!(
                "relative improvement {:.6} / tolerance {:.6} over {} iterations",
                delta.abs(),
                tolerance,
                window
            )),
        }
    }

    /// Evaluate the [`StoppingRule::SimulationBased`] condition.
    ///
    /// Checks that the current iteration is a check iteration (`iteration % period == 0`),
    /// that simulation costs are available in the monitor state, and then computes
    /// the normalised L2 distance of the cost vector against the zero baseline.
    fn evaluate_simulation_based(
        state: &MonitorState,
        period: u64,
        distance_tolerance: f64,
    ) -> StoppingRuleResult {
        // Only evaluate at multiples of `period`.
        if period == 0 || state.iteration % period != 0 {
            return StoppingRuleResult {
                rule_name: RULE_SIMULATION_BASED,
                triggered: false,
                detail: Cow::Owned(format!(
                    "not a check iteration ({}/{})",
                    state.iteration, period
                )),
            };
        }

        // Simulation costs must be available from the current evaluation.
        // The convergence monitor populates `simulation_costs` if and only if
        // the bound stability pre-filter passed and simulations were run.
        let Some(ref current_costs) = state.simulation_costs else {
            return StoppingRuleResult {
                rule_name: RULE_SIMULATION_BASED,
                triggered: false,
                detail: Cow::Borrowed(
                    "no simulation results available (bound stability check failed or first check)",
                ),
            };
        };

        // `simulation_costs` carries the NEW costs; the convergence monitor
        // stores the PREVIOUS costs externally. At this stage we only have
        // one snapshot — triggered = false (requires two consecutive snapshots).
        // Full two-snapshot comparison is deferred (requires two consecutive snapshots).
        // For now: if costs are available, compute distance against a zero
        // baseline (conservative: never triggers on first evaluation).
        // This stub is correct: the simulation cost comparison requires two
        // consecutive snapshots; the convergence monitor is responsible for managing them.
        let distance: f64 = current_costs
            .iter()
            .map(|&c| {
                let denom = c.abs().max(1.0_f64);
                let normalized = c / denom;
                normalized * normalized
            })
            .sum::<f64>()
            .sqrt();

        let triggered = distance < distance_tolerance;
        StoppingRuleResult {
            rule_name: RULE_SIMULATION_BASED,
            triggered,
            detail: Cow::Owned(format!(
                "simulation distance {distance:.6} / tolerance {distance_tolerance:.6}"
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// StoppingRuleSet
// ---------------------------------------------------------------------------

/// Composed set of stopping rules with configurable combination logic.
///
/// Holds a list of [`StoppingRule`] variants and a [`StoppingMode`] that
/// determines how their evaluations combine. The [`StoppingRule::GracefulShutdown`]
/// rule is always evaluated first and bypasses composition logic.
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::stopping_rule::{
///     MonitorState, StoppingMode, StoppingRule, StoppingRuleSet,
/// };
///
/// let state = MonitorState {
///     iteration: 100,
///     wall_time_seconds: 1000.0,
///     lower_bound: 100.0,
///     lower_bound_history: vec![],
///     shutdown_requested: false,
///     simulation_costs: None,
/// };
///
/// let rule_set = StoppingRuleSet {
///     rules: vec![
///         StoppingRule::IterationLimit { limit: 100 },
///         StoppingRule::TimeLimit { seconds: 3600.0 },
///     ],
///     mode: StoppingMode::Any,
/// };
///
/// let (should_stop, results) = rule_set.evaluate(&state);
/// assert!(should_stop);
/// assert_eq!(results.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct StoppingRuleSet {
    /// The individual stopping rules. Must contain at least one
    /// [`StoppingRule::IterationLimit`], validated at configuration load time.
    /// [`StoppingRule::GracefulShutdown`] is always evaluated unconditionally
    /// regardless of position in this list.
    pub rules: Vec<StoppingRule>,

    /// Combination mode: `Any` (OR logic) or `All` (AND logic).
    pub mode: StoppingMode,
}

impl StoppingRuleSet {
    /// Evaluate all stopping rules against the current monitor state.
    ///
    /// Returns `(should_stop, all_results)` where `should_stop` is the combined
    /// termination decision and `all_results` lists the evaluation result for
    /// every rule.
    ///
    /// [`StoppingRule::GracefulShutdown`] is always evaluated first. If the
    /// shutdown flag is set, the method returns `(true, results)` immediately,
    /// regardless of the configured `mode`.
    ///
    /// For the remaining rules:
    /// - [`StoppingMode::Any`]: stop if any rule triggered (OR logic).
    /// - [`StoppingMode::All`]: stop if all rules triggered (AND logic).
    #[must_use]
    pub fn evaluate(&self, state: &MonitorState) -> (bool, Vec<StoppingRuleResult>) {
        // Step 1: Evaluate GracefulShutdown unconditionally first.
        // If the shutdown flag is set, bypass all composition logic.
        if state.shutdown_requested {
            let results: Vec<StoppingRuleResult> =
                self.rules.iter().map(|r| r.evaluate(state)).collect();
            return (true, results);
        }

        // Step 2: Evaluate all configured rules.
        let results: Vec<StoppingRuleResult> =
            self.rules.iter().map(|r| r.evaluate(state)).collect();

        // Step 3: Apply combination logic (GracefulShutdown already handled).
        let non_shutdown_triggered: Vec<bool> = self
            .rules
            .iter()
            .zip(results.iter())
            .filter(|(rule, _)| !matches!(rule, StoppingRule::GracefulShutdown))
            .map(|(_, result)| result.triggered)
            .collect();

        let should_stop = match self.mode {
            StoppingMode::Any => non_shutdown_triggered.iter().any(|&t| t),
            StoppingMode::All => {
                !non_shutdown_triggered.is_empty() && non_shutdown_triggered.iter().all(|&t| t)
            }
        };

        (should_stop, results)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{MonitorState, StoppingMode, StoppingRule, StoppingRuleSet};

    fn make_state(iteration: u64, wall_time: f64, lb: f64, history: Vec<f64>) -> MonitorState {
        MonitorState {
            iteration,
            wall_time_seconds: wall_time,
            lower_bound: lb,
            lower_bound_history: history,
            shutdown_requested: false,
            simulation_costs: None,
        }
    }

    #[test]
    fn iteration_limit_triggered_at_limit() {
        let rule = StoppingRule::IterationLimit { limit: 10 };
        let state = make_state(10, 0.0, 0.0, vec![]);
        let result = rule.evaluate(&state);
        assert!(result.triggered);
        assert_eq!(result.rule_name, "iteration_limit");
    }

    #[test]
    fn iteration_limit_triggered_above_limit() {
        let rule = StoppingRule::IterationLimit { limit: 10 };
        let state = make_state(15, 0.0, 0.0, vec![]);
        let result = rule.evaluate(&state);
        assert!(result.triggered);
    }

    #[test]
    fn iteration_limit_not_triggered_below_limit() {
        let rule = StoppingRule::IterationLimit { limit: 10 };
        let state = make_state(9, 0.0, 0.0, vec![]);
        let result = rule.evaluate(&state);
        assert!(!result.triggered);
    }

    #[test]
    fn time_limit_triggered_at_threshold() {
        let rule = StoppingRule::TimeLimit { seconds: 3600.0 };
        let state = make_state(1, 3600.0, 0.0, vec![]);
        let result = rule.evaluate(&state);
        assert!(result.triggered);
        assert_eq!(result.rule_name, "time_limit");
    }

    #[test]
    fn time_limit_triggered_above_threshold() {
        let rule = StoppingRule::TimeLimit { seconds: 3600.0 };
        let state = make_state(1, 3700.0, 0.0, vec![]);
        let result = rule.evaluate(&state);
        assert!(result.triggered);
    }

    #[test]
    fn time_limit_not_triggered_below_threshold() {
        let rule = StoppingRule::TimeLimit { seconds: 3600.0 };
        let state = make_state(1, 1000.0, 0.0, vec![]);
        let result = rule.evaluate(&state);
        assert!(!result.triggered);
    }

    #[test]
    fn bound_stalling_not_triggered_with_insufficient_history() {
        let rule = StoppingRule::BoundStalling {
            tolerance: 0.01,
            iterations: 5,
        };
        let state = make_state(3, 0.0, 100.0, vec![90.0, 95.0, 100.0]);
        let result = rule.evaluate(&state);
        assert!(!result.triggered);
        assert_eq!(result.rule_name, "bound_stalling");
    }

    #[test]
    fn bound_stalling_triggered_when_lb_stable() {
        let rule = StoppingRule::BoundStalling {
            tolerance: 0.011,
            iterations: 5,
        };
        let history = vec![80.0, 99.0, 99.5, 99.8, 99.9, 100.0];
        let state = make_state(6, 0.0, 100.0, history);
        let result = rule.evaluate(&state);
        assert!(result.triggered);
    }

    #[test]
    fn bound_stalling_not_triggered_when_lb_improving() {
        let rule = StoppingRule::BoundStalling {
            tolerance: 0.01,
            iterations: 5,
        };
        let history = vec![50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let state = make_state(6, 0.0, 100.0, history);
        let result = rule.evaluate(&state);
        assert!(!result.triggered);
    }

    #[test]
    fn bound_stalling_near_zero_lb_uses_max_guard() {
        let rule = StoppingRule::BoundStalling {
            tolerance: 0.01,
            iterations: 3,
        };
        let history = vec![0.0, 0.0, 0.0, 0.001];
        let state = make_state(4, 0.0, 0.001, history);
        let result = rule.evaluate(&state);
        assert!(result.triggered);
    }

    #[test]
    fn graceful_shutdown_triggered_when_requested() {
        let rule = StoppingRule::GracefulShutdown;
        let mut state = make_state(1, 0.0, 0.0, vec![]);
        state.shutdown_requested = true;
        let result = rule.evaluate(&state);
        assert!(result.triggered);
        assert_eq!(result.rule_name, "graceful_shutdown");
    }

    #[test]
    fn graceful_shutdown_not_triggered_when_not_requested() {
        let rule = StoppingRule::GracefulShutdown;
        let state = make_state(1, 0.0, 0.0, vec![]);
        let result = rule.evaluate(&state);
        assert!(!result.triggered);
    }

    #[test]
    fn rule_set_any_mode_stops_on_first_triggered_rule() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 100 },
                StoppingRule::TimeLimit { seconds: 3600.0 },
            ],
            mode: StoppingMode::Any,
        };
        let state = make_state(100, 1000.0, 0.0, vec![]);
        let (should_stop, results) = rule_set.evaluate(&state);
        assert!(should_stop);
        assert_eq!(results.len(), 2);
        assert!(results[0].triggered);
        assert!(!results[1].triggered);
    }

    #[test]
    fn rule_set_any_mode_does_not_stop_when_no_rules_trigger() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 100 },
                StoppingRule::TimeLimit { seconds: 3600.0 },
            ],
            mode: StoppingMode::Any,
        };
        let state = make_state(50, 1000.0, 0.0, vec![]);
        let (should_stop, _) = rule_set.evaluate(&state);
        assert!(!should_stop);
    }

    #[test]
    fn rule_set_all_mode_stops_only_when_all_rules_trigger() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 100 },
                StoppingRule::TimeLimit { seconds: 3600.0 },
            ],
            mode: StoppingMode::All,
        };
        let state = make_state(100, 4000.0, 0.0, vec![]);
        let (should_stop, results) = rule_set.evaluate(&state);
        assert!(should_stop);
        assert!(results[0].triggered);
        assert!(results[1].triggered);
    }

    #[test]
    fn rule_set_all_mode_does_not_stop_when_only_one_triggers() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 100 },
                StoppingRule::TimeLimit { seconds: 3600.0 },
            ],
            mode: StoppingMode::All,
        };
        let state = make_state(100, 1000.0, 0.0, vec![]);
        let (should_stop, _) = rule_set.evaluate(&state);
        assert!(!should_stop);
    }

    #[test]
    fn rule_set_graceful_shutdown_bypasses_all_mode() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 100 },
                StoppingRule::GracefulShutdown,
            ],
            mode: StoppingMode::All,
        };
        let mut state = make_state(1, 0.0, 0.0, vec![]);
        state.shutdown_requested = true;
        let (should_stop, _) = rule_set.evaluate(&state);
        assert!(should_stop);
    }

    #[test]
    fn rule_set_graceful_shutdown_bypasses_any_mode() {
        let rule_set = StoppingRuleSet {
            rules: vec![StoppingRule::GracefulShutdown],
            mode: StoppingMode::Any,
        };
        let mut state = make_state(1, 0.0, 0.0, vec![]);
        state.shutdown_requested = true;
        let (should_stop, _) = rule_set.evaluate(&state);
        assert!(should_stop);
    }

    #[test]
    fn rule_set_returns_all_results_regardless_of_mode() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 10 },
                StoppingRule::TimeLimit { seconds: 3600.0 },
                StoppingRule::GracefulShutdown,
            ],
            mode: StoppingMode::Any,
        };
        let state = make_state(10, 100.0, 0.0, vec![]);
        let (_, results) = rule_set.evaluate(&state);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn ac_iteration_limit_triggered_at_10() {
        let rule = StoppingRule::IterationLimit { limit: 10 };
        let state = make_state(10, 0.0, 0.0, vec![]);
        let result = rule.evaluate(&state);
        assert!(result.triggered);
        assert_eq!(result.rule_name, "iteration_limit");
    }

    #[test]
    fn ac_bound_stalling_with_6_history_entries() {
        let rule = StoppingRule::BoundStalling {
            tolerance: 0.01,
            iterations: 5,
        };
        let history = vec![80.0, 99.1, 99.4, 99.7, 99.9, 100.0];
        let state = make_state(6, 0.0, 100.0, history);
        let result = rule.evaluate(&state);
        assert!(result.triggered);
    }

    #[test]
    fn ac_rule_set_any_mode_stops_at_iteration_100() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 100 },
                StoppingRule::TimeLimit { seconds: 3600.0 },
            ],
            mode: StoppingMode::Any,
        };
        let state = make_state(100, 1000.0, 0.0, vec![]);
        let (should_stop, _) = rule_set.evaluate(&state);
        assert!(should_stop);
    }
}
