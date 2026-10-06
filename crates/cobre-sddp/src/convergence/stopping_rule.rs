//! Stopping rules for the SDDP training loop.
//!
//! Defines stopping rule variants, composition logic, and convergence state.
//! Rules use enum dispatch; [`StoppingRuleSet`] composes them with AND/OR logic
//! into a [`StopDecision`].
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
//!     upper_bound: 110.0,
//!     lower_bound_history: vec![90.0, 95.0, 98.0, 99.0, 100.0,
//!                              100.0, 100.0, 100.0, 100.0, 100.0],
//!     shutdown_requested: false,
//! };
//!
//! let rule = StoppingRule::IterationLimit { limit: 10 };
//! assert!(rule.is_triggered(&state));
//! assert_eq!(rule.name(), "iteration_limit");
//! ```

/// Rule name for the iteration limit stopping rule.
pub const RULE_ITERATION_LIMIT: &str = "iteration_limit";
/// Rule name for the wall-clock time limit stopping rule.
pub const RULE_TIME_LIMIT: &str = "time_limit";
/// Rule name for the lower-bound stalling stopping rule.
pub const RULE_BOUND_STALLING: &str = "bound_stalling";
/// Rule name for the exact-upper-bound gap stopping rule.
pub const RULE_GAP: &str = "gap";
/// Rule name for the graceful-shutdown stopping rule.
pub const RULE_GRACEFUL_SHUTDOWN: &str = "graceful_shutdown";

/// Guarded denominator for the RELATIVE gap: `|lower_bound|` floored at `1.0`.
/// Both the reported gap ([`crate::ConvergenceMonitor::gap`]) and the
/// [`StoppingRule::Gap`] relative arm divide by this — normalized by the LOWER
/// bound, never the upper, so the two never disagree on what "relative gap"
/// means. The floor only matters near startup / a zero-cost study; a converged
/// lower bound is far larger than `1.0`.
pub(crate) fn relative_gap_denominator(lower_bound: f64) -> f64 {
    lower_bound.abs().max(1.0_f64)
}

// ---------------------------------------------------------------------------
// MonitorState
// ---------------------------------------------------------------------------

/// Read-only snapshot of convergence-monitor quantities consumed by
/// [`StoppingRuleSet::evaluate`].
#[derive(Debug, Clone)]
pub struct MonitorState {
    /// Current iteration index (1-based).
    pub iteration: u64,

    /// Cumulative wall-clock time since training start, in seconds.
    pub wall_time_seconds: f64,

    /// Current lower bound (stage-1 LP objective value).
    pub lower_bound: f64,

    /// Current upper bound, canonical R$; the exact `Σ w·c` bound under
    /// enumerated forwards that [`StoppingRule::Gap`] compares against.
    pub upper_bound: f64,

    /// Lower bounds of the evaluated iterations, oldest first; the last entry is
    /// the current iteration. A resumed run continues the series its checkpoint
    /// recorded.
    pub lower_bound_history: Vec<f64>,

    /// Whether a shutdown has been requested, from any
    /// [`crate::config::ShutdownSource`].
    pub shutdown_requested: bool,
}

// ---------------------------------------------------------------------------
// StoppingMode
// ---------------------------------------------------------------------------

/// Combination mode for [`StoppingRuleSet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoppingMode {
    /// Stop when **any** configured rule triggers (OR logic). A shutdown
    /// request also stops the run, as the [`StopMask::SHUTDOWN`] bit; a
    /// configured stop or an exhausted budget at the same iteration is the
    /// reported reason.
    Any,

    /// Stop when every rule other than `IterationLimit` triggers at the same
    /// iteration (AND logic), with at least one such rule; the run's iteration
    /// budget, the largest `IterationLimit`, caps the run whatever the other
    /// rules do. A shutdown request also stops the run, as the
    /// [`StopMask::SHUTDOWN`] bit; a configured stop or an exhausted budget at
    /// the same iteration is the reported reason.
    All,
}

// ---------------------------------------------------------------------------
// StoppingRule
// ---------------------------------------------------------------------------

/// Individual stopping rule for the SDDP training loop, composed into a
/// [`StoppingRuleSet`]. Every set must contain at least one `IterationLimit`
/// (the safety bound against infinite loops, validated at config load).
#[derive(Debug, Clone)]
pub enum StoppingRule {
    /// Terminate when the iteration count reaches a fixed limit.
    IterationLimit {
        /// Maximum iteration count. The rule triggers when `iteration >= limit`.
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

    /// Terminate once the clamped canonical-R$ gap `UB_exact − LB` satisfies the
    /// disjunction of the configured tolerance arms. Admissible only under
    /// enumerated forwards + an expectation measure (enforced by the setup
    /// admission gate); at least one tolerance arm is required (enforced at
    /// config mapping).
    Gap {
        /// Absolute gap tolerance, canonical R$.
        tolerance: Option<f64>,

        /// Relative gap tolerance in PERCENT: stops when
        /// `100·gap / max(1, |LB|) ≤ relative_tolerance`. A value of `0.01` means
        /// 0.01%, directly comparable to the reported `gap_percent`.
        relative_tolerance: Option<f64>,
    },
}

impl StoppingRule {
    /// Whether this rule's condition holds at `state` (pure; reads `state` only).
    #[must_use]
    pub fn is_triggered(&self, state: &MonitorState) -> bool {
        match self {
            Self::IterationLimit { limit } => state.iteration >= *limit,
            Self::TimeLimit { seconds } => state.wall_time_seconds >= *seconds,
            Self::BoundStalling {
                tolerance,
                iterations,
            } => Self::bound_stalling_triggered(state, *tolerance, *iterations),
            Self::Gap {
                tolerance,
                relative_tolerance,
            } => Self::gap_triggered(state, *tolerance, *relative_tolerance),
        }
    }

    /// The `RULE_*` name of this rule's kind.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::IterationLimit { .. } => RULE_ITERATION_LIMIT,
            Self::TimeLimit { .. } => RULE_TIME_LIMIT,
            Self::BoundStalling { .. } => RULE_BOUND_STALLING,
            Self::Gap { .. } => RULE_GAP,
        }
    }

    fn stop_bit(&self) -> StopMask {
        match self {
            Self::IterationLimit { .. } => StopMask::ITERATION_LIMIT,
            Self::TimeLimit { .. } => StopMask::TIME_LIMIT,
            Self::BoundStalling { .. } => StopMask::BOUND_STALLING,
            Self::Gap { .. } => StopMask::GAP,
        }
    }

    /// The [`StoppingRule::BoundStalling`] condition.
    fn bound_stalling_triggered(state: &MonitorState, tolerance: f64, iterations: u64) -> bool {
        // `iterations` is config-validated <= u32::MAX, so the cast cannot truncate.
        #[allow(clippy::cast_possible_truncation)]
        let window = iterations as usize;
        let history_len = state.lower_bound_history.len();
        if history_len < window {
            return false;
        }

        let lb_window_start = state.lower_bound_history[history_len - window];
        let lb_current = state.lower_bound;

        let denominator = lb_current.abs().max(1.0_f64);
        let delta = (lb_current - lb_window_start) / denominator;

        delta.abs() < tolerance
    }

    /// The [`StoppingRule::Gap`] condition: the clamped canonical-R$ gap
    /// `UB_exact − LB` against the disjunction of the configured tolerance arms
    /// (`gap ≤ tolerance` OR `100·gap / max(1, |LB|) ≤ relative_tolerance`). The
    /// relative arm normalizes by the LOWER bound
    /// ([`relative_gap_denominator`]) and is expressed in percent, the same
    /// convention the reported `gap_percent` uses. A small negative gap (float
    /// noise at a closed gap) is clamped to `0` before comparing.
    fn gap_triggered(
        state: &MonitorState,
        tolerance: Option<f64>,
        relative_tolerance: Option<f64>,
    ) -> bool {
        let gap = (state.upper_bound - state.lower_bound).max(0.0);
        let absolute_hit = tolerance.is_some_and(|t| gap <= t);
        let relative_hit = relative_tolerance
            .is_some_and(|r| 100.0 * gap / relative_gap_denominator(state.lower_bound) <= r);
        absolute_hit || relative_hit
    }
}

// ---------------------------------------------------------------------------
// StopMask / StopDecision
// ---------------------------------------------------------------------------

/// Bitset of the rule kinds that triggered at one iteration, plus the
/// qualifiers only the [`crate::ConvergenceMonitor`] knows.
///
/// A kind bit means "at least one listed rule of this kind triggered", under
/// either [`StoppingMode`]. [`StopMask::SHUTDOWN`] is set exactly when the
/// evaluated [`MonitorState`] carries a shutdown request, whatever rules the
/// set lists. [`StoppingRuleSet::evaluate`] never sets [`StopMask::SIGNAL`] or
/// [`StopMask::BUDGET_EXHAUSTED`]; [`crate::ConvergenceMonitor::update`] adds
/// them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StopMask(u32);

impl StopMask {
    /// An [`StoppingRule::IterationLimit`] triggered.
    pub const ITERATION_LIMIT: Self = Self(1 << 0);
    /// A [`StoppingRule::TimeLimit`] triggered.
    pub const TIME_LIMIT: Self = Self(1 << 1);
    /// A [`StoppingRule::BoundStalling`] triggered.
    pub const BOUND_STALLING: Self = Self(1 << 2);
    /// A [`StoppingRule::Gap`] triggered.
    pub const GAP: Self = Self(1 << 3);
    /// A shutdown was requested.
    pub const SHUTDOWN: Self = Self(1 << 4);
    /// The shutdown request came from a process signal; only ever set together
    /// with [`StopMask::SHUTDOWN`].
    pub const SIGNAL: Self = Self(1 << 5);
    /// This iteration is the run's last budgeted iteration.
    pub const BUDGET_EXHAUSTED: Self = Self(1 << 6);

    /// Whether every bit of `other` is set in `self`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

/// The outcome of one stop decision: which rule kinds triggered, whether the
/// configured rules call for a stop, and why the run ended.
///
/// `configured_stop` and `first_triggered` come from the per-rule pass, never
/// from the mask: a per-kind bit cannot tell repeated kinds apart under
/// [`StoppingMode::All`] and would pick a different reason under
/// [`StoppingMode::Any`] than the first triggered rule in declared order.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StopDecision {
    mask: StopMask,
    configured_stop: bool,
    first_triggered: Option<&'static str>,
}

impl StopDecision {
    /// The rule kinds that triggered, independent of the mode.
    #[must_use]
    pub fn mask(&self) -> StopMask {
        self.mask
    }

    /// The mode applied to the configured rules. Under [`StoppingMode::All`]
    /// the conjuncts are the rules other than `IterationLimit`, and a set with
    /// no conjunct never has a configured stop.
    #[must_use]
    pub fn configured_stop(&self) -> bool {
        self.configured_stop
    }

    /// Whether training stops: a configured stop, an exhausted iteration
    /// budget, or a shutdown request.
    #[must_use]
    pub fn should_stop(&self) -> bool {
        self.configured_stop
            || self.mask.contains(StopMask::BUDGET_EXHAUSTED)
            || self.mask.contains(StopMask::SHUTDOWN)
    }

    /// Why training stops, or `None` if it does not.
    ///
    /// A configured stop names its first triggered rule in declared order among
    /// the rules the mode decides on (every listed rule under
    /// [`StoppingMode::Any`], every rule but `IterationLimit` under
    /// [`StoppingMode::All`]). Otherwise an exhausted budget is
    /// [`RULE_ITERATION_LIMIT`], and a shutdown request alone is
    /// [`RULE_GRACEFUL_SHUTDOWN`].
    #[must_use]
    pub fn termination_reason(self) -> Option<&'static str> {
        if self.configured_stop {
            self.first_triggered
        } else if self.mask.contains(StopMask::BUDGET_EXHAUSTED) {
            Some(RULE_ITERATION_LIMIT)
        } else if self.mask.contains(StopMask::SHUTDOWN) {
            Some(RULE_GRACEFUL_SHUTDOWN)
        } else {
            None
        }
    }

    /// Whether a shutdown request ended training with neither a configured
    /// stop nor an exhausted budget, so the reason is [`RULE_GRACEFUL_SHUTDOWN`].
    #[must_use]
    pub fn ended_by_shutdown(self) -> bool {
        self.mask.contains(StopMask::SHUTDOWN)
            && !self.configured_stop
            && !self.mask.contains(StopMask::BUDGET_EXHAUSTED)
    }

    pub(crate) fn with_signal_source(mut self) -> Self {
        self.mask.insert(StopMask::SIGNAL);
        self
    }

    pub(crate) fn with_budget_exhausted(mut self) -> Self {
        self.mask.insert(StopMask::BUDGET_EXHAUSTED);
        self
    }
}

// ---------------------------------------------------------------------------
// StoppingRuleSet
// ---------------------------------------------------------------------------

/// Composed set of [`StoppingRule`] variants combined under a [`StoppingMode`].
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::stopping_rule::{
///     MonitorState, StopMask, StoppingMode, StoppingRule, StoppingRuleSet,
/// };
///
/// let state = MonitorState {
///     iteration: 100,
///     wall_time_seconds: 1000.0,
///     lower_bound: 100.0,
///     upper_bound: 110.0,
///     lower_bound_history: vec![],
///     shutdown_requested: false,
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
/// let decision = rule_set.evaluate(&state);
/// assert!(decision.should_stop());
/// assert!(decision.mask().contains(StopMask::ITERATION_LIMIT));
/// assert!(!decision.mask().contains(StopMask::TIME_LIMIT));
/// ```
#[derive(Debug, Clone)]
pub struct StoppingRuleSet {
    /// The individual stopping rules. Must contain at least one
    /// [`StoppingRule::IterationLimit`] (validated at config load). A shutdown
    /// request is the [`StopMask::SHUTDOWN`] bit of every set, reported as the
    /// reason only without a configured stop or an exhausted budget at the same
    /// iteration.
    pub rules: Vec<StoppingRule>,

    /// Combination mode for the rules.
    pub mode: StoppingMode,
}

impl StoppingRuleSet {
    /// Decide whether training stops at `state`, in one allocation-free pass.
    ///
    /// A set shutdown flag stops the run regardless of `mode`. Otherwise
    /// [`StoppingMode::Any`] stops if any rule triggered, and
    /// [`StoppingMode::All`] stops only if every rule other than
    /// `IterationLimit` triggered. The `IterationLimit` entries still set their
    /// mask bit under `All`.
    #[must_use]
    pub fn evaluate(&self, state: &MonitorState) -> StopDecision {
        let all_mode = self.mode == StoppingMode::All;
        let mut mask = StopMask::default();
        if state.shutdown_requested {
            mask.insert(StopMask::SHUTDOWN);
        }
        let mut any_triggered = false;
        let mut all_triggered = true;
        let mut has_configured = false;
        let mut first_triggered = None;

        for rule in &self.rules {
            let triggered = rule.is_triggered(state);
            if triggered {
                mask.insert(rule.stop_bit());
            }
            if all_mode && matches!(rule, StoppingRule::IterationLimit { .. }) {
                continue;
            }
            if triggered && first_triggered.is_none() {
                first_triggered = Some(rule.name());
            }
            has_configured = true;
            any_triggered |= triggered;
            all_triggered &= triggered;
        }

        let configured_stop = match self.mode {
            StoppingMode::Any => any_triggered,
            StoppingMode::All => has_configured && all_triggered,
        };
        StopDecision {
            mask,
            configured_stop,
            first_triggered,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        MonitorState, RULE_GRACEFUL_SHUTDOWN, RULE_ITERATION_LIMIT, RULE_TIME_LIMIT, StopDecision,
        StopMask, StoppingMode, StoppingRule, StoppingRuleSet,
    };

    fn make_state(iteration: u64, wall_time: f64, lb: f64, history: Vec<f64>) -> MonitorState {
        MonitorState {
            iteration,
            wall_time_seconds: wall_time,
            lower_bound: lb,
            upper_bound: 0.0,
            lower_bound_history: history,
            shutdown_requested: false,
        }
    }

    fn gap_state(lb: f64, ub: f64) -> MonitorState {
        MonitorState {
            iteration: 1,
            wall_time_seconds: 0.0,
            lower_bound: lb,
            upper_bound: ub,
            lower_bound_history: vec![],
            shutdown_requested: false,
        }
    }

    fn stalling_set(mode: StoppingMode) -> StoppingRuleSet {
        StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 10 },
                StoppingRule::BoundStalling {
                    tolerance: 1e-3,
                    iterations: 1,
                },
            ],
            mode,
        }
    }

    #[test]
    fn iteration_limit_triggered_at_limit() {
        let rule = StoppingRule::IterationLimit { limit: 10 };
        let state = make_state(10, 0.0, 0.0, vec![]);
        assert!(rule.is_triggered(&state));
        assert_eq!(rule.name(), "iteration_limit");
    }

    #[test]
    fn iteration_limit_triggered_above_limit() {
        let rule = StoppingRule::IterationLimit { limit: 10 };
        let state = make_state(15, 0.0, 0.0, vec![]);
        assert!(rule.is_triggered(&state));
    }

    #[test]
    fn iteration_limit_not_triggered_below_limit() {
        let rule = StoppingRule::IterationLimit { limit: 10 };
        let state = make_state(9, 0.0, 0.0, vec![]);
        assert!(!rule.is_triggered(&state));
    }

    #[test]
    fn time_limit_triggered_at_threshold() {
        let rule = StoppingRule::TimeLimit { seconds: 3600.0 };
        let state = make_state(1, 3600.0, 0.0, vec![]);
        assert!(rule.is_triggered(&state));
        assert_eq!(rule.name(), "time_limit");
    }

    #[test]
    fn time_limit_triggered_above_threshold() {
        let rule = StoppingRule::TimeLimit { seconds: 3600.0 };
        let state = make_state(1, 3700.0, 0.0, vec![]);
        assert!(rule.is_triggered(&state));
    }

    #[test]
    fn time_limit_not_triggered_below_threshold() {
        let rule = StoppingRule::TimeLimit { seconds: 3600.0 };
        let state = make_state(1, 1000.0, 0.0, vec![]);
        assert!(!rule.is_triggered(&state));
    }

    #[test]
    fn bound_stalling_not_triggered_with_insufficient_history() {
        let rule = StoppingRule::BoundStalling {
            tolerance: 0.01,
            iterations: 5,
        };
        let state = make_state(3, 0.0, 100.0, vec![90.0, 95.0, 100.0]);
        assert!(!rule.is_triggered(&state));
        assert_eq!(rule.name(), "bound_stalling");
    }

    #[test]
    fn bound_stalling_triggered_when_lb_stable() {
        let rule = StoppingRule::BoundStalling {
            tolerance: 0.011,
            iterations: 5,
        };
        let history = vec![80.0, 99.0, 99.5, 99.8, 99.9, 100.0];
        let state = make_state(6, 0.0, 100.0, history);
        assert!(rule.is_triggered(&state));
    }

    #[test]
    fn bound_stalling_not_triggered_when_lb_improving() {
        let rule = StoppingRule::BoundStalling {
            tolerance: 0.01,
            iterations: 5,
        };
        let history = vec![50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let state = make_state(6, 0.0, 100.0, history);
        assert!(!rule.is_triggered(&state));
    }

    #[test]
    fn bound_stalling_near_zero_lb_uses_max_guard() {
        let rule = StoppingRule::BoundStalling {
            tolerance: 0.01,
            iterations: 3,
        };
        let history = vec![0.0, 0.0, 0.0, 0.001];
        let state = make_state(4, 0.0, 0.001, history);
        assert!(rule.is_triggered(&state));
    }

    #[test]
    fn gap_absolute_arm_stops_within_tolerance() {
        let rule = StoppingRule::Gap {
            tolerance: Some(10.0),
            relative_tolerance: None,
        };
        // gap = 105 - 100 = 5 <= 10 → stop; 120 - 100 = 20 > 10 → no stop.
        assert!(rule.is_triggered(&gap_state(100.0, 105.0)));
        assert!(!rule.is_triggered(&gap_state(100.0, 120.0)));
        assert_eq!(rule.name(), "gap");
    }

    #[test]
    fn gap_relative_arm_stops_within_relative_tolerance() {
        let rule = StoppingRule::Gap {
            tolerance: None,
            relative_tolerance: Some(10.0),
        };
        // percent gap = 100·5/100 = 5% <= 10% → stop; 100·20/100 = 20% > 10% → no stop.
        assert!(rule.is_triggered(&gap_state(100.0, 105.0)));
        assert!(!rule.is_triggered(&gap_state(100.0, 120.0)));
    }

    #[test]
    fn gap_disjunction_stops_when_only_relative_arm_holds() {
        // abs tol 1.0 NOT met (gap 5 > 1); rel tol 10% met (5% <= 10%) → OR stops.
        let rule = StoppingRule::Gap {
            tolerance: Some(1.0),
            relative_tolerance: Some(10.0),
        };
        assert!(rule.is_triggered(&gap_state(100.0, 105.0)));
    }

    #[test]
    fn gap_disjunction_stops_when_only_absolute_arm_holds() {
        // Large |LB| starves the relative arm (percent gap 100·50/1e6 = 5e-3% >
        // 1e-9%); abs tol 100 met (gap 50 <= 100) → OR stops.
        let rule = StoppingRule::Gap {
            tolerance: Some(100.0),
            relative_tolerance: Some(1e-9),
        };
        assert!(rule.is_triggered(&gap_state(1_000_000.0, 1_000_050.0)));
    }

    #[test]
    fn gap_negative_float_noise_clamps_to_zero_and_converges() {
        // UB a hair below LB (closed-gap float noise): clamped to 0, so a zero
        // tolerance counts as converged.
        let rule = StoppingRule::Gap {
            tolerance: Some(0.0),
            relative_tolerance: None,
        };
        assert!(
            rule.is_triggered(&gap_state(100.0, 100.0 - 1e-9)),
            "a small negative gap must clamp to 0 and count as converged"
        );
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
        let decision = rule_set.evaluate(&state);
        assert!(decision.should_stop());
        assert_eq!(decision.termination_reason(), Some("iteration_limit"));
        assert!(decision.mask().contains(StopMask::ITERATION_LIMIT));
        assert!(!decision.mask().contains(StopMask::TIME_LIMIT));
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
        assert!(!rule_set.evaluate(&state).should_stop());
    }

    #[test]
    fn rule_set_all_mode_stops_when_every_rule_triggers() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 100 },
                StoppingRule::TimeLimit { seconds: 3600.0 },
            ],
            mode: StoppingMode::All,
        };
        let state = make_state(100, 4000.0, 0.0, vec![]);
        let decision = rule_set.evaluate(&state);
        assert!(decision.should_stop());
        assert!(decision.mask().contains(StopMask::ITERATION_LIMIT));
        assert!(decision.mask().contains(StopMask::TIME_LIMIT));
    }

    #[test]
    fn rule_set_all_mode_does_not_stop_when_only_the_iteration_limit_triggers() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 100 },
                StoppingRule::TimeLimit { seconds: 3600.0 },
            ],
            mode: StoppingMode::All,
        };
        let state = make_state(100, 1000.0, 0.0, vec![]);
        assert!(!rule_set.evaluate(&state).should_stop());
    }

    #[test]
    fn shutdown_request_stops_an_all_mode_set_whose_rules_have_not_triggered() {
        let rule_set = StoppingRuleSet {
            rules: vec![StoppingRule::IterationLimit { limit: 100 }],
            mode: StoppingMode::All,
        };
        let state = MonitorState {
            shutdown_requested: true,
            ..make_state(1, 0.0, 0.0, vec![])
        };
        let decision = rule_set.evaluate(&state);
        assert!(decision.should_stop());
        assert!(decision.mask().contains(StopMask::SHUTDOWN));
        assert_eq!(decision.termination_reason(), Some("graceful_shutdown"));
        assert!(decision.ended_by_shutdown());
    }

    #[test]
    fn shutdown_request_stops_an_any_mode_set_whose_rules_have_not_triggered() {
        let rule_set = StoppingRuleSet {
            rules: vec![StoppingRule::IterationLimit { limit: 100 }],
            mode: StoppingMode::Any,
        };
        let state = MonitorState {
            shutdown_requested: true,
            ..make_state(1, 0.0, 0.0, vec![])
        };
        let decision = rule_set.evaluate(&state);
        assert!(decision.should_stop());
        assert!(decision.mask().contains(StopMask::SHUTDOWN));
        assert_eq!(decision.termination_reason(), Some("graceful_shutdown"));
        assert!(decision.ended_by_shutdown());
    }

    #[test]
    fn ac_iteration_limit_triggered_at_10() {
        let rule = StoppingRule::IterationLimit { limit: 10 };
        let state = make_state(10, 0.0, 0.0, vec![]);
        assert!(rule.is_triggered(&state));
        assert_eq!(rule.name(), "iteration_limit");
    }

    #[test]
    fn ac_bound_stalling_with_6_history_entries() {
        let rule = StoppingRule::BoundStalling {
            tolerance: 0.01,
            iterations: 5,
        };
        let history = vec![80.0, 99.1, 99.4, 99.7, 99.9, 100.0];
        let state = make_state(6, 0.0, 100.0, history);
        assert!(rule.is_triggered(&state));
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
        assert!(rule_set.evaluate(&state).should_stop());
    }

    #[test]
    fn any_mode_reason_is_the_first_triggered_rule_in_declared_order() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 100 },
                StoppingRule::TimeLimit { seconds: 10.0 },
                StoppingRule::IterationLimit { limit: 5 },
            ],
            mode: StoppingMode::Any,
        };
        let decision = rule_set.evaluate(&make_state(6, 50.0, 0.0, vec![]));
        assert_eq!(decision.termination_reason(), Some("time_limit"));
    }

    #[test]
    fn all_mode_with_duplicate_rule_kinds_requires_every_rule() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 5 },
                StoppingRule::IterationLimit { limit: 100 },
                StoppingRule::TimeLimit { seconds: 10.0 },
                StoppingRule::TimeLimit { seconds: 100.0 },
            ],
            mode: StoppingMode::All,
        };
        assert!(
            !rule_set
                .evaluate(&make_state(6, 50.0, 0.0, vec![]))
                .configured_stop()
        );
        assert!(
            rule_set
                .evaluate(&make_state(6, 150.0, 0.0, vec![]))
                .configured_stop(),
            "the conjunction holds although IterationLimit{{100}} has not triggered"
        );
    }

    #[test]
    fn all_mode_conjunction_leaves_out_the_iteration_limit() {
        let rule_set = stalling_set(StoppingMode::All);

        let stalled = rule_set.evaluate(&make_state(4, 0.0, 100.0, vec![100.0]));
        assert!(stalled.configured_stop());
        assert!(stalled.should_stop());
        assert_eq!(stalled.termination_reason(), Some("bound_stalling"));
        assert!(stalled.mask().contains(StopMask::BOUND_STALLING));
        assert!(!stalled.mask().contains(StopMask::ITERATION_LIMIT));

        let capped = rule_set.evaluate(&make_state(10, 0.0, 100.0, vec![]));
        assert!(!capped.configured_stop());
        assert!(!capped.should_stop());
        assert_eq!(capped.termination_reason(), None);
        assert!(capped.mask().contains(StopMask::ITERATION_LIMIT));
    }

    #[test]
    fn all_mode_reason_names_a_conjunct_when_the_cap_coincides() {
        let decision =
            stalling_set(StoppingMode::All).evaluate(&make_state(10, 0.0, 100.0, vec![100.0]));
        assert!(decision.configured_stop());
        assert_eq!(decision.termination_reason(), Some("bound_stalling"));
    }

    #[test]
    fn all_mode_with_only_iteration_limit_rules_never_has_a_configured_stop() {
        let sets = [
            vec![StoppingRule::IterationLimit { limit: 5 }],
            vec![
                StoppingRule::IterationLimit { limit: 5 },
                StoppingRule::IterationLimit { limit: 8 },
            ],
        ];
        for rules in sets {
            let rule_set = StoppingRuleSet {
                rules,
                mode: StoppingMode::All,
            };
            for iteration in [5, 8, 100] {
                let mut state = make_state(iteration, 0.0, 0.0, vec![]);
                let decision = rule_set.evaluate(&state);
                assert!(!decision.configured_stop(), "iteration {iteration}");
                assert!(!decision.should_stop(), "iteration {iteration}");
                assert_eq!(decision.termination_reason(), None, "iteration {iteration}");
                assert!(
                    decision.mask().contains(StopMask::ITERATION_LIMIT),
                    "iteration {iteration}"
                );

                state.shutdown_requested = true;
                let decision = rule_set.evaluate(&state);
                assert!(decision.should_stop(), "iteration {iteration}");
                assert!(!decision.configured_stop(), "iteration {iteration}");
                assert_eq!(
                    decision.termination_reason(),
                    Some("graceful_shutdown"),
                    "iteration {iteration}"
                );
            }
        }
    }

    #[test]
    fn stop_decision_sets_the_mask_bit_of_each_triggered_rule_kind() {
        let gap_triggering = MonitorState {
            upper_bound: 101.0,
            ..make_state(10, 50.0, 100.0, vec![100.0])
        };
        let cases = [
            (
                StoppingRule::IterationLimit { limit: 10 },
                StopMask::ITERATION_LIMIT,
            ),
            (
                StoppingRule::TimeLimit { seconds: 50.0 },
                StopMask::TIME_LIMIT,
            ),
            (
                StoppingRule::BoundStalling {
                    tolerance: 1e-3,
                    iterations: 1,
                },
                StopMask::BOUND_STALLING,
            ),
            (
                StoppingRule::Gap {
                    tolerance: Some(5.0),
                    relative_tolerance: None,
                },
                StopMask::GAP,
            ),
        ];
        for mode in [StoppingMode::Any, StoppingMode::All] {
            for (rule, bit) in &cases {
                let rule_set = StoppingRuleSet {
                    rules: vec![rule.clone()],
                    mode,
                };
                assert_eq!(
                    rule_set.evaluate(&gap_triggering).mask(),
                    *bit,
                    "{} under {mode:?}",
                    rule.name()
                );
            }

            let shutdown_only = StoppingRuleSet {
                rules: vec![StoppingRule::IterationLimit { limit: 100 }],
                mode,
            };
            let state = MonitorState {
                shutdown_requested: true,
                ..make_state(1, 0.0, 0.0, vec![])
            };
            assert_eq!(
                shutdown_only.evaluate(&state).mask(),
                StopMask::SHUTDOWN,
                "a shutdown request sets the SHUTDOWN bit and no other"
            );
        }
    }

    #[test]
    fn configured_rule_wins_a_coincident_shutdown() {
        let rule_set = StoppingRuleSet {
            rules: vec![StoppingRule::IterationLimit { limit: 5 }],
            mode: StoppingMode::Any,
        };
        let state = MonitorState {
            shutdown_requested: true,
            ..make_state(5, 0.0, 0.0, vec![])
        };
        let decision = rule_set.evaluate(&state);
        assert!(decision.configured_stop());
        assert!(decision.mask().contains(StopMask::SHUTDOWN));
        assert_eq!(decision.termination_reason(), Some("iteration_limit"));
        assert!(!decision.ended_by_shutdown());
    }

    #[test]
    fn all_mode_partial_set_with_a_shutdown_reports_graceful_shutdown() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 100 },
                StoppingRule::TimeLimit { seconds: 3600.0 },
                StoppingRule::BoundStalling {
                    tolerance: 1e-3,
                    iterations: 1,
                },
            ],
            mode: StoppingMode::All,
        };
        let state = MonitorState {
            shutdown_requested: true,
            ..make_state(4, 0.0, 100.0, vec![100.0])
        };
        let decision = rule_set.evaluate(&state);
        assert!(decision.mask().contains(StopMask::BOUND_STALLING));
        assert!(!decision.mask().contains(StopMask::TIME_LIMIT));
        assert!(!decision.configured_stop());
        assert!(decision.should_stop());
        assert_eq!(decision.termination_reason(), Some("graceful_shutdown"));
        assert!(decision.ended_by_shutdown());
    }

    #[test]
    fn shutdown_without_a_configured_stop_reports_graceful_shutdown() {
        for mode in [StoppingMode::Any, StoppingMode::All] {
            let rule_set = StoppingRuleSet {
                rules: vec![
                    StoppingRule::IterationLimit { limit: 100 },
                    StoppingRule::TimeLimit { seconds: 3600.0 },
                ],
                mode,
            };
            let state = MonitorState {
                shutdown_requested: true,
                ..make_state(1, 0.0, 0.0, vec![])
            };
            let decision = rule_set.evaluate(&state);
            assert!(!decision.configured_stop(), "{mode:?}");
            assert_eq!(
                decision.termination_reason(),
                Some("graceful_shutdown"),
                "{mode:?}"
            );
            assert!(decision.ended_by_shutdown(), "{mode:?}");
            assert!(!decision.mask().contains(StopMask::SIGNAL), "{mode:?}");
        }
    }

    #[test]
    fn should_stop_equals_a_present_termination_reason() {
        for bits in 0..16_u8 {
            let [configured_stop, budget, shutdown, signal] =
                [0, 1, 2, 3].map(|i| bits >> i & 1 == 1);
            let mut mask = StopMask::default();
            for (set, bit) in [
                (budget, StopMask::BUDGET_EXHAUSTED),
                (shutdown, StopMask::SHUTDOWN),
                (signal, StopMask::SIGNAL),
            ] {
                if set {
                    mask.insert(bit);
                }
            }
            let decision = StopDecision {
                mask,
                configured_stop,
                first_triggered: configured_stop.then_some(RULE_TIME_LIMIT),
            };
            let expected = if configured_stop {
                Some(RULE_TIME_LIMIT)
            } else if budget {
                Some(RULE_ITERATION_LIMIT)
            } else if shutdown {
                Some(RULE_GRACEFUL_SHUTDOWN)
            } else {
                None
            };
            assert_eq!(decision.termination_reason(), expected, "{decision:?}");
            assert_eq!(
                decision.should_stop(),
                decision.termination_reason().is_some(),
                "{decision:?}"
            );
            assert_eq!(
                decision.ended_by_shutdown(),
                decision.termination_reason() == Some(RULE_GRACEFUL_SHUTDOWN),
                "{decision:?}"
            );
        }
    }
}

#[cfg(test)]
mod proptests {
    use proptest::prelude::*;
    use proptest::test_runner::RngSeed;

    use super::{
        MonitorState, RULE_GRACEFUL_SHUTDOWN, RULE_ITERATION_LIMIT, StopMask, StoppingMode,
        StoppingRule, StoppingRuleSet,
    };

    fn fixed_config() -> ProptestConfig {
        ProptestConfig {
            cases: 1024,
            rng_seed: RngSeed::Fixed(42),
            ..ProptestConfig::default()
        }
    }

    fn rule() -> impl Strategy<Value = StoppingRule> {
        prop_oneof![
            (1..=8u64).prop_map(|limit| StoppingRule::IterationLimit { limit }),
            (0.0..=20.0_f64).prop_map(|seconds| StoppingRule::TimeLimit { seconds }),
            (prop_oneof![Just(0.0_f64), 0.0..=0.5_f64], 1..=4u64).prop_map(
                |(tolerance, iterations)| StoppingRule::BoundStalling {
                    tolerance,
                    iterations,
                }
            ),
            (
                proptest::option::of(0.0..=20.0_f64),
                proptest::option::of(0.0..=20.0_f64)
            )
                .prop_map(|(tolerance, relative_tolerance)| StoppingRule::Gap {
                    tolerance,
                    relative_tolerance,
                }),
        ]
    }

    fn rule_set() -> impl Strategy<Value = StoppingRuleSet> {
        (
            proptest::collection::vec(rule(), 1..=6),
            prop_oneof![Just(StoppingMode::Any), Just(StoppingMode::All)],
        )
            .prop_map(|(rules, mode)| StoppingRuleSet { rules, mode })
    }

    fn state() -> impl Strategy<Value = MonitorState> {
        (
            1..=10u64,
            0.0..=20.0_f64,
            1.0..=100.0_f64,
            0.0..=120.0_f64,
            proptest::collection::vec(1.0..=100.0_f64, 0..=6),
            any::<bool>(),
        )
            .prop_map(
                |(
                    iteration,
                    wall_time_seconds,
                    lower_bound,
                    upper_bound,
                    lower_bound_history,
                    shutdown_requested,
                )| MonitorState {
                    iteration,
                    wall_time_seconds,
                    lower_bound,
                    upper_bound,
                    lower_bound_history,
                    shutdown_requested,
                },
            )
    }

    /// The stop precedence written from the per-rule `is_triggered` flags: a
    /// configured stop names its first triggered configured rule, else an
    /// exhausted budget is `iteration_limit`, else a shutdown is
    /// `graceful_shutdown`. The configured rules are every listed rule under
    /// `Any`, and every rule but `IterationLimit` under `All`.
    fn precedence_oracle(
        rule_set: &StoppingRuleSet,
        state: &MonitorState,
        budget_exhausted: bool,
    ) -> (bool, Option<&'static str>) {
        let configured: Vec<(&'static str, bool)> = rule_set
            .rules
            .iter()
            .filter(|r| match r {
                StoppingRule::IterationLimit { .. } => rule_set.mode == StoppingMode::Any,
                _ => true,
            })
            .map(|r| (r.name(), r.is_triggered(state)))
            .collect();
        let configured_stop = match rule_set.mode {
            StoppingMode::Any => configured.iter().any(|(_, t)| *t),
            StoppingMode::All => !configured.is_empty() && configured.iter().all(|(_, t)| *t),
        };
        let reason = if configured_stop {
            configured.iter().find(|(_, t)| *t).map(|(n, _)| *n)
        } else if budget_exhausted {
            Some(RULE_ITERATION_LIMIT)
        } else if state.shutdown_requested {
            Some(RULE_GRACEFUL_SHUTDOWN)
        } else {
            None
        };
        (configured_stop, reason)
    }

    proptest! {
        #![proptest_config(fixed_config())]

        #[test]
        fn stop_decision_reason_follows_configured_rule_precedence_for_any_rule_set_and_state(
            rule_set in rule_set(),
            state in state(),
            budget_exhausted in any::<bool>(),
        ) {
            let evaluated = rule_set.evaluate(&state);
            prop_assert!(!evaluated.mask().contains(StopMask::SIGNAL));
            prop_assert!(!evaluated.mask().contains(StopMask::BUDGET_EXHAUSTED));
            prop_assert_eq!(
                evaluated.should_stop(),
                evaluated.configured_stop() || state.shutdown_requested
            );

            let decision = if budget_exhausted {
                evaluated.with_budget_exhausted()
            } else {
                evaluated
            };
            let (configured_stop, reason) = precedence_oracle(&rule_set, &state, budget_exhausted);
            prop_assert_eq!(decision.configured_stop(), configured_stop);
            prop_assert_eq!(decision.termination_reason(), reason);
            prop_assert_eq!(decision.should_stop(), reason.is_some());
            prop_assert_eq!(
                decision.ended_by_shutdown(),
                reason == Some(RULE_GRACEFUL_SHUTDOWN)
            );

            for bit in [
                StopMask::ITERATION_LIMIT,
                StopMask::TIME_LIMIT,
                StopMask::BOUND_STALLING,
                StopMask::GAP,
            ] {
                let any_of_kind = rule_set
                    .rules
                    .iter()
                    .any(|r| r.stop_bit() == bit && r.is_triggered(&state));
                prop_assert_eq!(decision.mask().contains(bit), any_of_kind);
            }
            prop_assert_eq!(
                decision.mask().contains(StopMask::SHUTDOWN),
                state.shutdown_requested
            );
        }
    }
}
