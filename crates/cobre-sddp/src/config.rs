//! Configuration types for the SDDP training loop.
//!
//! [`TrainingConfig`] groups training parameters into [`LoopConfig`],
//! [`CutManagementConfig`], and [`EventConfig`]. It does not implement `Default`
//! — every sub-struct must be supplied explicitly to prevent silent
//! misconfiguration; each sub-struct's `Default` carries test values for
//! `..Default::default()` overrides.
//!
//! # Examples
//!
//! ```rust
//! use cobre_sddp::TrainingConfig;
//! use cobre_sddp::config::{CutManagementConfig, EventConfig, LoopConfig};
//!
//! let config = TrainingConfig {
//!     loop_config: LoopConfig {
//!         forward_passes: 10,
//!         max_iterations: 200,
//!         ..LoopConfig::default()
//!     },
//!     cut_management: CutManagementConfig {
//!         cut_activity_tolerance: 1e-6,
//!         ..CutManagementConfig::default()
//!     },
//!     events: EventConfig {
//!         checkpoint_interval: Some(50),
//!         ..EventConfig::default()
//!     },
//! };
//! assert_eq!(config.loop_config.forward_passes, 10);
//! assert_eq!(config.loop_config.max_iterations, 200);
//! assert_eq!(config.events.checkpoint_interval, Some(50));
//! ```

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Sender;

use cobre_core::TrainingEvent;

use crate::cut_selection::CutSelectionStrategy;
use crate::risk_measure::RiskMeasure;
use crate::stopping_rule::{StoppingMode, StoppingRule, StoppingRuleSet};

/// Pure-data iteration parameters stored on [`crate::setup::StudySetup`].
///
/// Projection of [`LoopConfig`] to the fields stable across training
/// invocations. `n_fwd_threads` is excluded — it is derived per-call from the
/// `--threads` CLI flag and passed to [`crate::setup::StudySetup::train`].
#[derive(Debug)]
pub struct LoopParams {
    /// Random seed for forward-pass stochastic trajectory generation.
    pub seed: u64,
    /// Number of forward-pass trajectories per training iteration.
    pub forward_passes: u32,
    /// `true` when the forward selection is `enumerated`; selects the exact
    /// probability-weighted upper bound instead of the sampled statistical one.
    pub training_enumerated: bool,
    /// Maximum iteration budget (also used for FCF cut-pool pre-sizing).
    pub max_iterations: u64,
    /// Starting iteration offset for resumed training runs.
    pub(crate) start_iteration: u64,
    /// Stopping rules controlling convergence.
    pub(crate) stopping_rules: StoppingRuleSet,
}

/// Controls the iteration loop and convergence.
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::config::LoopConfig;
///
/// let cfg = LoopConfig { forward_passes: 10, max_iterations: 200, ..LoopConfig::default() };
/// assert_eq!(cfg.forward_passes, 10);
/// ```
#[derive(Debug)]
pub struct LoopConfig {
    /// Total forward scenarios per iteration across all ranks. Must be `>= 1`.
    pub forward_passes: u32,

    /// `true` when the forward selection is `enumerated`; [`crate::train`] then
    /// assembles the exact probability-weighted upper bound rather than the
    /// sampled Welford mean + CI.
    pub training_enumerated: bool,

    /// Maximum training iterations before forced termination. Must be `>= 1`.
    /// Also drives cut-pool capacity pre-sizing.
    pub max_iterations: u64,

    /// Starting iteration for resumed runs (checkpoint `completed_iterations`);
    /// the loop runs `start_iteration + 1` through `max_iterations`. Default `0`.
    pub start_iteration: u64,

    /// Number of rayon threads for forward-pass parallelism; `1` is single-threaded.
    pub n_fwd_threads: usize,

    /// Stopping rules evaluated after each iteration's lower-bound update.
    pub stopping_rules: StoppingRuleSet,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            forward_passes: 1,
            training_enumerated: false,
            max_iterations: 1,
            start_iteration: 0,
            n_fwd_threads: 1,
            stopping_rules: StoppingRuleSet {
                rules: vec![StoppingRule::IterationLimit { limit: 1 }],
                mode: StoppingMode::Any,
            },
        }
    }
}

/// Two-stage cut management pipeline configuration.
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::config::CutManagementConfig;
///
/// let cfg = CutManagementConfig { cut_activity_tolerance: 1e-8, ..CutManagementConfig::default() };
/// assert_eq!(cfg.cut_activity_tolerance, 1e-8);
/// ```
#[derive(Debug)]
pub struct CutManagementConfig {
    /// Cut selection strategy for deactivating dominated cuts; `None` keeps all cuts active.
    pub cut_selection: Option<CutSelectionStrategy>,

    /// Hard cap on active cuts per stage (cut-selection stage 2); `None` is uncapped.
    /// Cuts from the current iteration are never evicted.
    pub budget: Option<u32>,

    /// Activity (dual-value) threshold below which a cut is a deactivation candidate.
    pub cut_activity_tolerance: f64,

    /// Per-stage backward-pass risk measures; length must equal `num_stages`.
    pub risk_measures: Vec<RiskMeasure>,
}

impl Default for CutManagementConfig {
    fn default() -> Self {
        Self {
            cut_selection: None,
            budget: None,
            cut_activity_tolerance: 1e-6,
            risk_measures: vec![RiskMeasure::Expectation],
        }
    }
}

/// Event infrastructure for monitoring and checkpointing.
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::config::EventConfig;
///
/// let cfg = EventConfig { checkpoint_interval: Some(10), ..EventConfig::default() };
/// assert_eq!(cfg.checkpoint_interval, Some(10));
/// ```
#[derive(Debug, Default)]
pub struct EventConfig {
    /// Channel sender for training progress events; `None` emits none.
    /// The receiver must be drained on another thread or it blocks the loop.
    pub event_sender: Option<Sender<TrainingEvent>>,

    /// Iterations between checkpoint writes (`iteration % n == 0`); `None` writes none.
    pub checkpoint_interval: Option<u64>,

    /// Shutdown signal checked (`load(Relaxed)`) at each iteration boundary for early exit.
    pub shutdown_flag: Option<Arc<AtomicBool>>,

    /// Allocate the visited-states archive for state export. Also forced on when any
    /// [`CutSelectionStrategy`] is enabled — the value-evaluation kernel scores every
    /// cut at every archived state. Default `false`.
    pub export_states: bool,
}

/// Pure-data event parameters stored on [`crate::setup::StudySetup`].
///
/// Projection of [`EventConfig`] to the fields stable across invocations and
/// safe to persist; the runtime handles (`event_sender`, `shutdown_flag`) and
/// `checkpoint_interval` are excluded.
#[derive(Debug)]
pub(crate) struct EventParams {
    /// See [`EventConfig::export_states`].
    pub(crate) export_states: bool,
}

/// Parameters controlling the SDDP training loop.
///
/// Composes [`LoopConfig`], [`CutManagementConfig`], and [`EventConfig`]. No
/// `Default` — every sub-group must be supplied explicitly.
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::TrainingConfig;
/// use cobre_sddp::config::{CutManagementConfig, EventConfig, LoopConfig};
///
/// let config = TrainingConfig {
///     loop_config: LoopConfig {
///         forward_passes: 10,
///         max_iterations: 100,
///         ..LoopConfig::default()
///     },
///     cut_management: CutManagementConfig::default(),
///     events: EventConfig::default(),
/// };
/// assert_eq!(config.loop_config.forward_passes, 10);
/// assert_eq!(config.loop_config.max_iterations, 100);
/// ```
#[derive(Debug)]
pub struct TrainingConfig {
    /// Controls the iteration loop, forward pass count, and convergence rules.
    pub loop_config: LoopConfig,

    /// Two-stage cut management pipeline configuration.
    pub cut_management: CutManagementConfig,

    /// Event infrastructure for monitoring and checkpointing.
    pub events: EventConfig,
}

#[cfg(test)]
mod tests {
    use super::{CutManagementConfig, EventConfig, LoopConfig, TrainingConfig};
    use cobre_core::TrainingEvent;

    // ── Field access ─────────────────────────────────────────────────────────

    #[test]
    fn field_access_forward_passes_and_max_iterations() {
        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 10,
                max_iterations: 100,
                ..LoopConfig::default()
            },
            cut_management: CutManagementConfig::default(),
            events: EventConfig::default(),
        };
        assert_eq!(config.loop_config.forward_passes, 10);
        assert_eq!(config.loop_config.max_iterations, 100);
    }

    #[test]
    fn checkpoint_interval_none_and_some() {
        let config_none = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 5,
                max_iterations: 50,
                ..LoopConfig::default()
            },
            cut_management: CutManagementConfig::default(),
            events: EventConfig::default(),
        };
        assert!(config_none.events.checkpoint_interval.is_none());

        let config_some = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 5,
                max_iterations: 50,
                ..LoopConfig::default()
            },
            cut_management: CutManagementConfig::default(),
            events: EventConfig {
                checkpoint_interval: Some(10),
                ..EventConfig::default()
            },
        };
        assert_eq!(config_some.events.checkpoint_interval, Some(10));
    }

    // ── Event sender ─────────────────────────────────────────────────────────

    #[test]
    fn event_sender_none() {
        let config = TrainingConfig {
            loop_config: LoopConfig::default(),
            cut_management: CutManagementConfig::default(),
            events: EventConfig::default(),
        };
        assert!(config.events.event_sender.is_none());
    }

    #[test]
    fn event_sender_some_can_send_training_event() {
        let (tx, rx) = std::sync::mpsc::channel::<TrainingEvent>();
        let config = TrainingConfig {
            loop_config: LoopConfig::default(),
            cut_management: CutManagementConfig::default(),
            events: EventConfig {
                event_sender: Some(tx),
                ..EventConfig::default()
            },
        };

        assert!(config.events.event_sender.is_some());

        if let Some(sender) = &config.events.event_sender {
            sender
                .send(TrainingEvent::TrainingFinished {
                    reason: "test".to_string(),
                    iterations: 1,
                    final_lb: 0.0,
                    final_ub: 1.0,
                    total_time_ms: 100,
                    total_rows: 4,
                })
                .unwrap();
        }

        let received = rx.recv().unwrap();
        assert!(matches!(received, TrainingEvent::TrainingFinished { .. }));
    }

    // ── Debug output ─────────────────────────────────────────────────────────

    #[test]
    fn debug_output_non_empty() {
        let config = TrainingConfig {
            loop_config: LoopConfig::default(),
            cut_management: CutManagementConfig::default(),
            events: EventConfig::default(),
        };
        let debug = format!("{config:?}");
        assert!(!debug.is_empty());
        assert!(
            debug.contains("forward_passes"),
            "debug must contain field name: {debug}"
        );
        assert!(
            debug.contains("max_iterations"),
            "debug must contain field name: {debug}"
        );
    }
}
