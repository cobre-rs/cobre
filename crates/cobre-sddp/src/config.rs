//! Configuration types for the SDDP training loop.
//!
//! [`TrainingConfig`] bundles all algorithm parameters that control the
//! training loop behaviour, grouped into three sub-structs:
//!
//! - [`LoopConfig`]: iteration loop control and convergence.
//! - [`CutManagementConfig`]: three-stage cut management pipeline.
//! - [`EventConfig`]: event infrastructure for monitoring and checkpointing.
//!
//! ## Construction
//!
//! `TrainingConfig` does not implement `Default` — every sub-struct group must
//! be explicitly supplied by the caller to prevent silent misconfigurations.
//! Each sub-struct implements `Default` with sensible test values, allowing
//! tests to specify only the fields they care about via `..Default::default()`.
//!
//! ## Event channel
//!
//! The `event_sender` field in [`EventConfig`] carries an
//! `Option<Sender<TrainingEvent>>`. When `None`, the training loop emits no
//! events and incurs no channel overhead. When `Some(sender)`, typed
//! [`cobre_core::TrainingEvent`] values are moved into the channel at each
//! lifecycle step boundary.
//!
//! Because `Sender<T>` is not `Clone` in the general sense (it can be cloned,
//! but ownership transfer is the primary pattern), `TrainingConfig` does not
//! derive `Clone`. Callers that need to pass config to multiple locations
//! should construct separate instances.
//!
//! # Examples
//!
//! ```rust
//! use cobre_sddp::{TrainingConfig, LoopConfig, CutManagementConfig, EventConfig};
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

use cobre_core::TrainingEvent;

use crate::angular_pruning::AngularPruningParams;
use crate::cut_selection::CutSelectionStrategy;
use crate::risk_measure::RiskMeasure;
use crate::stopping_rule::{StoppingMode, StoppingRule, StoppingRuleSet};

/// Controls the iteration loop and convergence.
///
/// Construct via [`Default::default()`] for tests, or explicitly set all fields
/// for production configuration.
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::LoopConfig;
///
/// let cfg = LoopConfig { forward_passes: 10, max_iterations: 200, ..LoopConfig::default() };
/// assert_eq!(cfg.forward_passes, 10);
/// ```
#[derive(Debug)]
pub struct LoopConfig {
    /// Total number of forward scenarios evaluated per iteration across all ranks.
    ///
    /// The work is divided among MPI ranks: each rank evaluates
    /// `forward_passes / num_ranks` scenarios (with remainder distributed to
    /// the first ranks). Must be at least 1.
    pub forward_passes: u32,

    /// Maximum number of training iterations before forced termination.
    ///
    /// Also used for cut pool pre-sizing: the cut pool allocates capacity
    /// for `max_iterations * forward_passes * num_stages` cuts at
    /// initialisation to avoid reallocation during the training loop.
    /// Must be at least 1.
    pub max_iterations: u64,

    /// Starting iteration for resumed training runs.
    ///
    /// When resuming from a checkpoint, this is set to the checkpoint's
    /// `completed_iterations`. The training loop starts at
    /// `start_iteration + 1` and runs up to `max_iterations`.
    /// Default: `0` (fresh training).
    pub start_iteration: u64,

    /// Number of rayon threads for forward pass parallelism.
    ///
    /// Controls how many worker threads execute forward scenarios in parallel.
    /// Set to `1` for single-threaded execution.
    pub n_fwd_threads: usize,

    /// Maximum number of demand blocks across all stages.
    ///
    /// Used for pre-sizing buffers and determining the LP column layout.
    pub max_blocks: usize,

    /// Stopping rules controlling convergence.
    ///
    /// The training loop evaluates these rules after each iteration's lower
    /// bound update. Training terminates when the rule set triggers.
    pub stopping_rules: StoppingRuleSet,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            forward_passes: 1,
            max_iterations: 1,
            start_iteration: 0,
            n_fwd_threads: 1,
            max_blocks: 1,
            stopping_rules: StoppingRuleSet {
                rules: vec![StoppingRule::IterationLimit { limit: 1 }],
                mode: StoppingMode::Any,
            },
        }
    }
}

/// Three-stage cut management pipeline configuration.
///
/// Construct via [`Default::default()`] for tests, or explicitly set all fields
/// for production configuration.
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::CutManagementConfig;
///
/// let cfg = CutManagementConfig { cut_activity_tolerance: 1e-8, ..CutManagementConfig::default() };
/// assert_eq!(cfg.cut_activity_tolerance, 1e-8);
/// ```
#[derive(Debug)]
pub struct CutManagementConfig {
    /// Optional cut selection strategy for deactivating dominated cuts.
    ///
    /// When `Some(strategy)`, the training loop applies cut selection after each
    /// backward pass, deactivating cuts that do not meet the strategy's activity
    /// criteria. When `None`, all generated cuts remain active.
    pub cut_selection: Option<CutSelectionStrategy>,

    /// Optional angular diversity pruning parameters (stage 2 of the cut
    /// selection pipeline).
    ///
    /// When `Some(params)`, the training loop applies angular pruning after the
    /// strategy-based selection pass, using cosine similarity clustering as a
    /// computational accelerator for pointwise dominance verification.
    /// When `None`, angular pruning is disabled.
    pub angular_pruning: Option<AngularPruningParams>,

    /// Maximum number of active cuts per stage (stage 3 of the cut selection
    /// pipeline — hard cap on LP size).
    ///
    /// When `Some(n)`, the training loop enforces a hard cap of `n` active cuts
    /// per stage after strategy selection and angular pruning have completed.
    /// Cuts are evicted in order of staleness (`last_active_iter` ascending),
    /// tie-broken by usage frequency (`active_count` ascending). Cuts generated
    /// in the current iteration are never evicted.
    ///
    /// When `None`, no hard cap is enforced.
    pub budget: Option<u32>,

    /// Enable basis padding for warm-start.
    ///
    /// When `true`, the forward pass applies informed basis status assignment for
    /// new cut rows before warm-starting the LP solver, reducing the number of
    /// simplex pivots required after each cut addition.
    ///
    /// Disabled by default (`false`).
    pub basis_padding_enabled: bool,

    /// Activity tolerance for cut selection deactivation.
    ///
    /// Cuts with activity (dual value) below this threshold across all openings
    /// in a backward pass are candidates for deactivation. Typical value: `1e-6`.
    pub cut_activity_tolerance: f64,

    /// Number of pre-loaded cuts imported from a warm-start policy file.
    ///
    /// When non-zero, the cut pool is pre-populated from a serialised policy
    /// before the first training iteration begins. The warm-start cut count
    /// contributes to the cut pool capacity calculation alongside
    /// `max_iterations`.
    pub warm_start_cuts: u32,

    /// Per-stage risk measures for the backward pass.
    ///
    /// Controls whether the Benders cut aggregation uses expected value,
    /// `CVaR`, or a convex combination thereof. The length must equal
    /// `num_stages`.
    pub risk_measures: Vec<RiskMeasure>,
}

impl Default for CutManagementConfig {
    fn default() -> Self {
        Self {
            cut_selection: None,
            angular_pruning: None,
            budget: None,
            basis_padding_enabled: false,
            cut_activity_tolerance: 1e-6,
            warm_start_cuts: 0,
            risk_measures: vec![RiskMeasure::Expectation],
        }
    }
}

/// Event infrastructure for monitoring and checkpointing.
///
/// Construct via [`Default::default()`] for tests, or explicitly set all fields
/// for production configuration.
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::EventConfig;
///
/// let cfg = EventConfig { checkpoint_interval: Some(10), ..EventConfig::default() };
/// assert_eq!(cfg.checkpoint_interval, Some(10));
/// ```
#[derive(Debug, Default)]
pub struct EventConfig {
    /// Optional channel sender for real-time training progress events.
    ///
    /// When `Some(sender)`, the training loop emits [`TrainingEvent`] values
    /// at each lifecycle step boundary (forward pass, backward pass,
    /// convergence update, etc.). When `None`, no events are emitted and no
    /// channel allocation occurs on the hot path.
    ///
    /// The receiver end must be consumed on a separate thread or task to
    /// prevent the channel from filling and blocking the training loop.
    pub event_sender: Option<std::sync::mpsc::Sender<TrainingEvent>>,

    /// Number of iterations between checkpoint writes.
    ///
    /// When `Some(n)`, the training loop writes a checkpoint after every `n`
    /// completed iterations (i.e., when `iteration % n == 0`). When `None`,
    /// no checkpoints are written during training (a final checkpoint may
    /// still be written at convergence depending on caller configuration).
    pub checkpoint_interval: Option<u64>,

    /// Optional shutdown signal for graceful early termination.
    ///
    /// When `Some(flag)`, the training loop checks `flag.load(Relaxed)` at each
    /// iteration boundary. If the flag is `true`, the loop terminates early with
    /// a "shutdown requested" reason. When `None`, the loop runs until
    /// convergence or iteration limit.
    pub shutdown_flag: Option<Arc<AtomicBool>>,

    /// Whether to allocate the visited-states archive for state export.
    ///
    /// When `true`, the archive is allocated so forward-pass trial points are
    /// recorded for checkpoint persistence. When `false`, the archive is only
    /// allocated if `cut_selection` requires it (i.e., `Dominated` variant).
    /// Default: `false`.
    pub export_states: bool,
}

/// Parameters controlling the SDDP training loop.
///
/// Composes three sub-structs, each covering a distinct concern:
/// [`LoopConfig`] for iteration loop control, [`CutManagementConfig`] for the
/// cut management pipeline, and [`EventConfig`] for monitoring infrastructure.
///
/// `TrainingConfig` does not implement `Default` — every sub-group must be
/// explicitly supplied to prevent silent misconfiguration. Each sub-struct
/// implements `Default` with sensible test values, allowing tests to override
/// only the fields they care about via `..Default::default()`.
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::{TrainingConfig, LoopConfig, CutManagementConfig, EventConfig};
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

    /// Three-stage cut management pipeline configuration.
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

    #[test]
    fn warm_start_cuts_field_accessible() {
        let config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 10,
                ..LoopConfig::default()
            },
            cut_management: CutManagementConfig {
                warm_start_cuts: 500,
                ..CutManagementConfig::default()
            },
            events: EventConfig::default(),
        };
        assert_eq!(config.cut_management.warm_start_cuts, 500);
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
            loop_config: LoopConfig {
                forward_passes: 4,
                max_iterations: 200,
                ..LoopConfig::default()
            },
            cut_management: CutManagementConfig {
                warm_start_cuts: 100,
                cut_activity_tolerance: 1e-6,
                ..CutManagementConfig::default()
            },
            events: EventConfig {
                event_sender: Some(tx),
                checkpoint_interval: Some(50),
                ..EventConfig::default()
            },
        };

        assert!(config.events.event_sender.is_some());

        // Verify the sender in the config can actually send events.
        if let Some(sender) = &config.events.event_sender {
            sender
                .send(TrainingEvent::TrainingFinished {
                    reason: "test".to_string(),
                    iterations: 1,
                    final_lb: 0.0,
                    final_ub: 1.0,
                    total_time_ms: 100,
                    total_cuts: 4,
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
            loop_config: LoopConfig {
                forward_passes: 8,
                max_iterations: 500,
                ..LoopConfig::default()
            },
            cut_management: CutManagementConfig::default(),
            events: EventConfig {
                checkpoint_interval: Some(100),
                ..EventConfig::default()
            },
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
