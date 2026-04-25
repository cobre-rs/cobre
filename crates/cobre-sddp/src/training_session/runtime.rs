//! Per-invocation runtime-handle sub-struct for one training run.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Sender;

use cobre_core::TrainingEvent;

/// Per-invocation runtime handles for a training run.
///
/// These are the per-call hooks that allow an external caller to observe
/// the training's progress (`event_sender`), abort it gracefully
/// (`shutdown_flag`), and opt into state export (`export_states`). They
/// are not configuration values that govern the training algorithm; they
/// are runtime integration points. This mirrors the `EventParams`
/// projection on `StudySetup` (see `crates/cobre-sddp/src/config.rs`),
/// which deliberately excludes runtime handles from the long-lived
/// study configuration.
///
/// All three fields are moved into the struct exactly once at construction
/// time (inside `TrainingSession::new`) and are never mutated thereafter.
pub(crate) struct RuntimeHandles {
    pub event_sender: Option<Sender<TrainingEvent>>,
    pub shutdown_flag: Option<Arc<AtomicBool>>,
    #[allow(dead_code)]
    pub export_states: bool,
}

impl RuntimeHandles {
    /// Construct the handles from the three per-invocation values.
    ///
    /// Trivially stores the inputs; no derivation or heap activity.
    pub(crate) fn new(
        event_sender: Option<Sender<TrainingEvent>>,
        shutdown_flag: Option<Arc<AtomicBool>>,
        export_states: bool,
    ) -> Self {
        Self {
            event_sender,
            shutdown_flag,
            export_states,
        }
    }

    /// Return a borrowed reference to the event sender, if present.
    ///
    /// Called at every event emission site (6+ times per iteration path).
    /// Centralising `Option::as_ref()` here removes repetition without
    /// adding accessor bloat.
    pub(crate) fn event_sender(&self) -> Option<&Sender<TrainingEvent>> {
        self.event_sender.as_ref()
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
    use std::sync::mpsc;

    use cobre_core::TrainingEvent;

    use super::RuntimeHandles;

    #[test]
    fn runtime_handles_new_stores_inputs() {
        let runtime = RuntimeHandles::new(None, None, true);
        assert!(runtime.event_sender.is_none());
        assert!(runtime.shutdown_flag.is_none());
        assert!(runtime.export_states);
    }

    #[test]
    fn runtime_handles_event_sender_returns_borrowed_ref() {
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let runtime = RuntimeHandles::new(Some(tx), None, false);

        assert!(runtime.event_sender().is_some());

        // Send through the accessor's borrowed reference.
        runtime
            .event_sender()
            .unwrap()
            .send(TrainingEvent::TrainingFinished {
                reason: "test".to_string(),
                iterations: 0,
                final_lb: 0.0,
                final_ub: 0.0,
                total_time_ms: 0,
                total_rows: 0,
            })
            .unwrap();

        let received = rx.recv().unwrap();
        assert!(matches!(received, TrainingEvent::TrainingFinished { .. }));
    }
}
