//! Configuration types for the SDDP training loop.
//!
//! [`TrainingConfig`] bundles all algorithm parameters that control the
//! training loop behaviour: iteration budget, forward scenario count,
//! checkpoint cadence, warm-start cut count, and the optional event channel
//! for real-time progress reporting.
//!
//! ## Construction
//!
//! `TrainingConfig` does not implement `Default` — every field must be
//! explicitly supplied by the caller to prevent silent misconfigurations
//! (e.g., a zero `forward_passes` count or an unintentionally large
//! `max_iterations`).
//!
//! ## Event channel
//!
//! The `event_sender` field carries an `Option<Sender<TrainingEvent>>`.
//! When `None`, the training loop emits no events and incurs no channel
//! overhead. When `Some(sender)`, typed [`cobre_core::TrainingEvent`] values
//! are moved into the channel at each lifecycle step boundary.
//!
//! Because `Sender<T>` is not `Clone` in the general sense (it can be cloned,
//! but ownership transfer is the primary pattern), `TrainingConfig` does not
//! derive `Clone`. Callers that need to pass config to multiple locations
//! should construct separate instances.
//!
//! # Examples
//!
//! ```rust
//! use cobre_sddp::TrainingConfig;
//!
//! let config = TrainingConfig {
//!     forward_passes: 10,
//!     max_iterations: 200,
//!     checkpoint_interval: Some(50),
//!     warm_start_cuts: 0,
//!     event_sender: None,
//! };
//! assert_eq!(config.forward_passes, 10);
//! assert_eq!(config.max_iterations, 200);
//! assert_eq!(config.checkpoint_interval, Some(50));
//! ```

use cobre_core::TrainingEvent;

/// Parameters controlling the SDDP training loop.
///
/// Construct this struct directly — all fields are public and there is no
/// builder or `Default` implementation. Every field must be set explicitly
/// to prevent silent misconfiguration.
///
/// # Examples
///
/// ```rust
/// use cobre_sddp::TrainingConfig;
///
/// let config = TrainingConfig {
///     forward_passes: 10,
///     max_iterations: 100,
///     checkpoint_interval: None,
///     warm_start_cuts: 0,
///     event_sender: None,
/// };
/// assert_eq!(config.forward_passes, 10);
/// assert_eq!(config.max_iterations, 100);
/// ```
#[derive(Debug)]
pub struct TrainingConfig {
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

    /// Number of iterations between checkpoint writes.
    ///
    /// When `Some(n)`, the training loop writes a checkpoint after every `n`
    /// completed iterations (i.e., when `iteration % n == 0`). When `None`,
    /// no checkpoints are written during training (a final checkpoint may
    /// still be written at convergence depending on caller configuration).
    pub checkpoint_interval: Option<u64>,

    /// Number of pre-loaded cuts imported from a warm-start policy file.
    ///
    /// When non-zero, the cut pool is pre-populated from a serialised policy
    /// before the first training iteration begins. The warm-start cut count
    /// contributes to the cut pool capacity calculation alongside
    /// `max_iterations`.
    pub warm_start_cuts: u32,

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
}

#[cfg(test)]
mod tests {
    use super::TrainingConfig;
    use cobre_core::TrainingEvent;

    // ── Field access ─────────────────────────────────────────────────────────

    #[test]
    fn field_access_forward_passes_and_max_iterations() {
        let config = TrainingConfig {
            forward_passes: 10,
            max_iterations: 100,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
        };
        assert_eq!(config.forward_passes, 10);
        assert_eq!(config.max_iterations, 100);
    }

    #[test]
    fn checkpoint_interval_none_and_some() {
        let config_none = TrainingConfig {
            forward_passes: 5,
            max_iterations: 50,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
        };
        assert!(config_none.checkpoint_interval.is_none());

        let config_some = TrainingConfig {
            forward_passes: 5,
            max_iterations: 50,
            checkpoint_interval: Some(10),
            warm_start_cuts: 0,
            event_sender: None,
        };
        assert_eq!(config_some.checkpoint_interval, Some(10));
    }

    #[test]
    fn warm_start_cuts_field_accessible() {
        let config = TrainingConfig {
            forward_passes: 1,
            max_iterations: 10,
            checkpoint_interval: None,
            warm_start_cuts: 500,
            event_sender: None,
        };
        assert_eq!(config.warm_start_cuts, 500);
    }

    // ── Event sender ─────────────────────────────────────────────────────────

    #[test]
    fn event_sender_none() {
        let config = TrainingConfig {
            forward_passes: 1,
            max_iterations: 1,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
        };
        assert!(config.event_sender.is_none());
    }

    #[test]
    fn event_sender_some_can_send_training_event() {
        let (tx, rx) = std::sync::mpsc::channel::<TrainingEvent>();
        let config = TrainingConfig {
            forward_passes: 4,
            max_iterations: 200,
            checkpoint_interval: Some(50),
            warm_start_cuts: 100,
            event_sender: Some(tx),
        };

        assert!(config.event_sender.is_some());

        // Verify the sender in the config can actually send events.
        if let Some(sender) = &config.event_sender {
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
            forward_passes: 8,
            max_iterations: 500,
            checkpoint_interval: Some(100),
            warm_start_cuts: 0,
            event_sender: None,
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
