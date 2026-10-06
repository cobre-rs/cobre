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
//!         export_states: true,
//!         ..EventConfig::default()
//!     },
//! };
//! assert_eq!(config.loop_config.forward_passes, 10);
//! assert_eq!(config.loop_config.max_iterations, 200);
//! assert!(config.events.export_states);
//! ```

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::mpsc::Sender;

use cobre_core::TrainingEvent;
use cobre_io::config::CheckpointSchedule;

use crate::cut_selection::CutSelectionStrategy;
use crate::policy::orchestration::PeriodicCheckpoint;
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
    /// Lower bounds of the resumed run's recorded iterations.
    pub(crate) resume_lower_bound_history: Vec<f64>,
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

    /// Lower bounds of the resumed run's recorded iterations, oldest first,
    /// restored with `start_iteration`. Default empty.
    pub resume_lower_bound_history: Vec<f64>,

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
            resume_lower_bound_history: Vec::new(),
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
/// let cfg = EventConfig { export_states: true, ..EventConfig::default() };
/// assert!(cfg.periodic_checkpoint.is_none());
/// ```
#[derive(Debug, Default)]
pub struct EventConfig {
    /// Channel sender for training progress events; `None` emits none.
    /// The receiver must be drained on another thread or it blocks the loop.
    pub event_sender: Option<Sender<TrainingEvent>>,

    /// The periodic checkpoint the writing rank commits on scheduled non-stop
    /// iterations; `None` writes none. Built by
    /// [`StudySetup::enable_periodic_checkpoints`](crate::setup::StudySetup::enable_periodic_checkpoints).
    pub periodic_checkpoint: Option<PeriodicCheckpoint>,

    /// Shutdown request, read once per iteration just before the stop decision.
    ///
    /// `0` means no request; otherwise the value is the [`ShutdownSource::level`]
    /// of the strongest request. Writers only raise it (`fetch_max`, or
    /// signal-hook's `register_usize` with the signal level) and never lower it,
    /// and a cooperative writer never stores the signal level.
    pub shutdown_flag: Option<Arc<AtomicUsize>>,

    /// Allocate the visited-states archive for state export. Also forced on when any
    /// [`CutSelectionStrategy`] is enabled — the value-evaluation kernel scores every
    /// cut at every archived state. Default `false`.
    pub export_states: bool,
}

/// Where a shutdown request came from, ordered from weakest to strongest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ShutdownSource {
    /// A request from the embedding program, such as a progress callback.
    Cooperative,
    /// A process signal.
    Signal,
}

impl ShutdownSource {
    /// The value this source stores in [`EventConfig::shutdown_flag`].
    #[must_use]
    pub const fn level(self) -> usize {
        match self {
            Self::Cooperative => 1,
            Self::Signal => 2,
        }
    }

    /// The source a stored level stands for; `0` is no request, and a level
    /// above the signal's reads as a signal.
    pub(crate) const fn from_level(level: usize) -> Option<Self> {
        match level {
            0 => None,
            1 => Some(Self::Cooperative),
            _ => Some(Self::Signal),
        }
    }
}

/// Pure-data event parameters stored on [`crate::setup::StudySetup`].
///
/// Projection of [`EventConfig`] to the fields stable across invocations and
/// safe to persist; the runtime handles (`event_sender`, `shutdown_flag`) are
/// excluded, and the periodic checkpoint is kept as its schedule.
#[derive(Debug)]
pub(crate) struct EventParams {
    /// See [`EventConfig::export_states`].
    pub(crate) export_states: bool,
    /// `policy.checkpointing`'s resolved schedule; `None` when off.
    pub(crate) checkpoint_schedule: Option<CheckpointSchedule>,
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
