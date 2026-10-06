//! Per-invocation runtime-handle sub-struct for one training run.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Sender;

use cobre_core::TrainingEvent;

use crate::config::ShutdownSource;
use crate::policy::orchestration::PeriodicCheckpoint;

/// Per-invocation runtime integration hooks, kept separate from the long-lived
/// study configuration — mirrors the `EventParams` projection on `StudySetup`.
pub(crate) struct RuntimeHandles {
    pub event_sender: Option<Sender<TrainingEvent>>,
    shutdown_flag: Option<Arc<AtomicUsize>>,
    // Rationale: production reads `config.events.export_states` directly; this
    // field exists for symmetry and is asserted by the constructor unit test.
    #[allow(dead_code)]
    pub export_states: bool,
    periodic_checkpoint: Option<PeriodicCheckpoint>,
}

impl RuntimeHandles {
    /// Construct the handles from the four per-invocation values.
    pub(crate) fn new(
        event_sender: Option<Sender<TrainingEvent>>,
        shutdown_flag: Option<Arc<AtomicUsize>>,
        export_states: bool,
        periodic_checkpoint: Option<PeriodicCheckpoint>,
    ) -> Self {
        Self {
            event_sender,
            shutdown_flag,
            export_states,
            periodic_checkpoint,
        }
    }

    /// Return a borrowed reference to the event sender, if present.
    pub(crate) fn event_sender(&self) -> Option<&Sender<TrainingEvent>> {
        self.event_sender.as_ref()
    }

    pub(crate) fn periodic_checkpoint(&self) -> Option<&PeriodicCheckpoint> {
        self.periodic_checkpoint.as_ref()
    }

    /// The strongest shutdown request made so far, if any. The only read of the
    /// shared flag in training.
    pub(crate) fn shutdown_requested(&self) -> Option<ShutdownSource> {
        self.shutdown_flag
            .as_ref()
            .and_then(|flag| ShutdownSource::from_level(flag.load(Ordering::Relaxed)))
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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;

    use cobre_core::TrainingEvent;

    use super::RuntimeHandles;
    use crate::config::ShutdownSource;

    #[test]
    fn runtime_handles_new_stores_inputs() {
        let runtime = RuntimeHandles::new(None, None, true, None);
        assert!(runtime.event_sender.is_none());
        assert!(runtime.shutdown_flag.is_none());
        assert!(runtime.export_states);
        assert!(runtime.periodic_checkpoint().is_none());
    }

    #[test]
    fn shutdown_requested_maps_the_shared_level() {
        assert_eq!(
            RuntimeHandles::new(None, None, false, None).shutdown_requested(),
            None
        );

        let flag = Arc::new(AtomicUsize::new(0));
        let runtime = RuntimeHandles::new(None, Some(Arc::clone(&flag)), false, None);
        for (level, expected) in [
            (0, None),
            (1, Some(ShutdownSource::Cooperative)),
            (2, Some(ShutdownSource::Signal)),
        ] {
            flag.store(level, Ordering::Relaxed);
            assert_eq!(runtime.shutdown_requested(), expected, "level {level}");
        }

        flag.fetch_max(ShutdownSource::Cooperative.level(), Ordering::Relaxed);
        assert_eq!(
            runtime.shutdown_requested(),
            Some(ShutdownSource::Signal),
            "a cooperative request after a signal keeps the signal source"
        );
    }

    #[test]
    fn runtime_handles_event_sender_returns_borrowed_ref() {
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let runtime = RuntimeHandles::new(Some(tx), None, false, None);

        assert!(runtime.event_sender().is_some());

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
