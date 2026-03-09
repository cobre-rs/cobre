//! Progress bar rendering for training and simulation phases.
//!
//! This module provides [`run_progress_thread`], which spawns a background thread
//! that consumes [`TrainingEvent`] values from an `mpsc::Receiver`, drives
//! [`indicatif::ProgressBar`] instances on stderr, and returns all collected events
//! when the sender disconnects.
//!
//! ## Design
//!
//! Training and simulation phases are sequential: the training bar is active during
//! the training loop, the simulation bar during policy evaluation. Because only one
//! bar is active at a time, [`indicatif::MultiProgress`] is not needed.
//!
//! The collected events are returned via [`ProgressHandle::join`] so the caller can
//! pass them directly to the output pipeline without maintaining a separate consumer.
//!
//! ## Example
//!
//! ```rust,no_run
//! use std::sync::mpsc;
//! use cobre_core::TrainingEvent;
//! use cobre_cli::progress::run_progress_thread;
//!
//! let (tx, rx) = mpsc::channel::<TrainingEvent>();
//! let handle = run_progress_thread(rx, 100);
//! drop(tx); // sender disconnects — thread exits
//! let events = handle.join();
//! assert!(events.is_empty());
//! ```

use std::sync::mpsc;
use std::thread;

use cobre_core::TrainingEvent;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

const TRAINING_TEMPLATE: &str = "Training {bar:40} {pos}/{len} iter  {msg}";
const SIMULATION_TEMPLATE: &str =
    "Simulation {bar:40} {pos}/{len} scenarios  [{elapsed_precise} < {eta_precise}]";

/// Handle to a background progress thread.
pub struct ProgressHandle {
    handle: thread::JoinHandle<Vec<TrainingEvent>>,
}

impl ProgressHandle {
    /// Wait for the progress thread to finish and return all collected events.
    ///
    /// The events are returned in the order they were received. The caller is
    /// expected to pass them to the output pipeline (e.g., `build_training_output`).
    ///
    /// # Panics
    ///
    /// Propagates a panic from the progress thread. If the thread panicked, this
    /// method panics with the message `"progress thread panicked"`.
    #[allow(clippy::expect_used)]
    pub fn join(self) -> Vec<TrainingEvent> {
        // Intentional: a panic in the progress thread is a programming error,
        // not a recoverable condition. Propagating it here matches the contract
        // described in the ticket's error-handling section.
        self.handle.join().expect("progress thread panicked")
    }
}

/// Spawn a background thread that renders progress bars and collects events.
pub fn run_progress_thread(
    receiver: mpsc::Receiver<TrainingEvent>,
    max_iterations: u64,
) -> ProgressHandle {
    let handle = thread::spawn(move || {
        let mut events: Vec<TrainingEvent> = Vec::new();
        let mut training_bar: Option<ProgressBar> = None;
        let mut simulation_bar: Option<ProgressBar> = None;

        loop {
            if let Ok(event) = receiver.recv() {
                events.push(event.clone());
                match event {
                    TrainingEvent::IterationSummary {
                        iteration,
                        lower_bound,
                        upper_bound,
                        gap,
                        ..
                    } => {
                        let bar =
                            training_bar.get_or_insert_with(|| create_training_bar(max_iterations));
                        let gap_pct = gap * 100.0;
                        bar.set_position(iteration);
                        bar.set_message(format!(
                            "LB: {lower_bound:.1}  UB: {upper_bound:.1}  gap: {gap_pct:.1}%"
                        ));
                    }

                    TrainingEvent::TrainingFinished {
                        iterations,
                        final_lb,
                        final_ub,
                        ..
                    } => {
                        if let Some(bar) = training_bar.take() {
                            bar.set_position(iterations);
                            bar.finish_with_message(format!(
                                "LB: {final_lb:.1}  UB: {final_ub:.1}  done"
                            ));
                        }
                    }

                    TrainingEvent::SimulationProgress {
                        scenarios_complete,
                        scenarios_total,
                        ..
                    } => {
                        let bar = simulation_bar.get_or_insert_with(|| {
                            create_simulation_bar(u64::from(scenarios_total))
                        });
                        bar.set_position(u64::from(scenarios_complete));
                    }

                    TrainingEvent::SimulationFinished { scenarios, .. } => {
                        if let Some(bar) = simulation_bar.take() {
                            bar.set_position(u64::from(scenarios));
                            bar.finish_with_message("complete");
                        }
                    }

                    TrainingEvent::ForwardPassComplete { .. }
                    | TrainingEvent::ForwardSyncComplete { .. }
                    | TrainingEvent::BackwardPassComplete { .. }
                    | TrainingEvent::CutSyncComplete { .. }
                    | TrainingEvent::CutSelectionComplete { .. }
                    | TrainingEvent::ConvergenceUpdate { .. }
                    | TrainingEvent::CheckpointComplete { .. }
                    | TrainingEvent::TrainingStarted { .. } => {}
                }
            } else {
                if let Some(bar) = training_bar.take() {
                    bar.abandon();
                }
                if let Some(bar) = simulation_bar.take() {
                    bar.abandon();
                }
                break;
            }
        }

        events
    });

    ProgressHandle { handle }
}

fn create_training_bar(max_iterations: u64) -> ProgressBar {
    let bar = ProgressBar::with_draw_target(Some(max_iterations), ProgressDrawTarget::stderr());
    let style = ProgressStyle::with_template(TRAINING_TEMPLATE)
        .unwrap_or_else(|_| ProgressStyle::default_bar());
    bar.set_style(style);
    bar
}

fn create_simulation_bar(scenarios_total: u64) -> ProgressBar {
    let bar = ProgressBar::with_draw_target(Some(scenarios_total), ProgressDrawTarget::stderr());
    let style = ProgressStyle::with_template(SIMULATION_TEMPLATE)
        .unwrap_or_else(|_| ProgressStyle::default_bar());
    bar.set_style(style);
    bar
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use cobre_core::TrainingEvent;

    use super::run_progress_thread;

    #[allow(clippy::cast_precision_loss)]
    fn make_iteration_summary(iteration: u64) -> TrainingEvent {
        TrainingEvent::IterationSummary {
            iteration,
            lower_bound: 100.0 + iteration as f64,
            upper_bound: 110.0 + iteration as f64,
            gap: 0.09,
            wall_time_ms: iteration * 200,
            iteration_time_ms: 200,
            forward_ms: 80,
            backward_ms: 100,
            lp_solves: 240,
        }
    }

    fn make_training_finished() -> TrainingEvent {
        TrainingEvent::TrainingFinished {
            reason: "iteration_limit".to_string(),
            iterations: 3,
            final_lb: 105.0,
            final_ub: 106.0,
            total_time_ms: 600,
            total_cuts: 144,
        }
    }

    fn make_simulation_progress(complete: u32, total: u32) -> TrainingEvent {
        TrainingEvent::SimulationProgress {
            scenarios_complete: complete,
            scenarios_total: total,
            elapsed_ms: u64::from(complete) * 50,
        }
    }

    fn make_simulation_finished() -> TrainingEvent {
        TrainingEvent::SimulationFinished {
            scenarios: 200,
            output_dir: "/tmp/output".to_string(),
            elapsed_ms: 10_000,
        }
    }

    #[test]
    fn test_progress_handle_training_events_returned() {
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let handle = run_progress_thread(rx, 10);

        tx.send(make_iteration_summary(1)).unwrap();
        tx.send(make_iteration_summary(2)).unwrap();
        tx.send(make_iteration_summary(3)).unwrap();
        tx.send(make_training_finished()).unwrap();
        drop(tx);

        let events = handle.join();
        assert_eq!(events.len(), 4, "expected 4 events, got {}", events.len());
    }

    #[test]
    fn test_progress_handle_simulation_events_returned() {
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let handle = run_progress_thread(rx, 10);

        tx.send(make_simulation_progress(50, 200)).unwrap();
        tx.send(make_simulation_progress(100, 200)).unwrap();
        tx.send(make_simulation_finished()).unwrap();
        drop(tx);

        let events = handle.join();
        assert_eq!(events.len(), 3, "expected 3 events, got {}", events.len());

        assert!(
            matches!(
                events[0],
                TrainingEvent::SimulationProgress {
                    scenarios_complete: 50,
                    ..
                }
            ),
            "first event must be SimulationProgress(50)"
        );
        assert!(
            matches!(
                events[1],
                TrainingEvent::SimulationProgress {
                    scenarios_complete: 100,
                    ..
                }
            ),
            "second event must be SimulationProgress(100)"
        );
        assert!(
            matches!(events[2], TrainingEvent::SimulationFinished { .. }),
            "third event must be SimulationFinished"
        );
    }

    #[test]
    fn test_progress_handle_returns_all_events() {
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let handle = run_progress_thread(rx, 10);

        tx.send(make_iteration_summary(1)).unwrap();
        tx.send(make_iteration_summary(2)).unwrap();
        tx.send(make_training_finished()).unwrap();
        tx.send(make_simulation_progress(50, 200)).unwrap();
        tx.send(make_simulation_finished()).unwrap();
        drop(tx);

        let events = handle.join();
        assert_eq!(events.len(), 5, "expected 5 events, got {}", events.len());

        // Verify ordering by checking variant sequence.
        assert!(matches!(
            events[0],
            TrainingEvent::IterationSummary { iteration: 1, .. }
        ));
        assert!(matches!(
            events[1],
            TrainingEvent::IterationSummary { iteration: 2, .. }
        ));
        assert!(matches!(events[2], TrainingEvent::TrainingFinished { .. }));
        assert!(matches!(
            events[3],
            TrainingEvent::SimulationProgress { .. }
        ));
        assert!(matches!(
            events[4],
            TrainingEvent::SimulationFinished { .. }
        ));
    }

    #[test]
    fn test_empty_channel_returns_empty_vec() {
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let handle = run_progress_thread(rx, 10);
        drop(tx);

        let events = handle.join();
        assert!(
            events.is_empty(),
            "expected empty vec, got {} events",
            events.len()
        );
    }

    #[test]
    fn test_training_only_no_simulation_events() {
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let handle = run_progress_thread(rx, 10);

        for i in 1..=5 {
            tx.send(make_iteration_summary(i)).unwrap();
        }
        tx.send(make_training_finished()).unwrap();
        drop(tx);

        let events = handle.join();
        assert_eq!(
            events.len(),
            6,
            "expected 6 events (5 summaries + 1 finished)"
        );

        for (i, event) in events[..5].iter().enumerate() {
            let expected_iter = (i + 1) as u64;
            assert!(
                matches!(event, TrainingEvent::IterationSummary { iteration, .. } if *iteration == expected_iter),
                "event[{i}] must be IterationSummary({expected_iter})"
            );
        }
        assert!(matches!(events[5], TrainingEvent::TrainingFinished { .. }));
    }

    #[test]
    fn test_non_ui_events_are_collected() {
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let handle = run_progress_thread(rx, 10);

        tx.send(TrainingEvent::ForwardPassComplete {
            iteration: 1,
            scenarios: 10,
            ub_mean: 110.0,
            ub_std: 5.0,
            elapsed_ms: 42,
        })
        .unwrap();
        tx.send(TrainingEvent::BackwardPassComplete {
            iteration: 1,
            cuts_generated: 48,
            stages_processed: 12,
            elapsed_ms: 87,
        })
        .unwrap();
        tx.send(make_iteration_summary(1)).unwrap();
        tx.send(make_training_finished()).unwrap();
        drop(tx);

        let events = handle.join();
        assert_eq!(events.len(), 4, "expected 4 events, got {}", events.len());
        assert!(matches!(
            events[0],
            TrainingEvent::ForwardPassComplete { .. }
        ));
        assert!(matches!(
            events[1],
            TrainingEvent::BackwardPassComplete { .. }
        ));
    }
}
