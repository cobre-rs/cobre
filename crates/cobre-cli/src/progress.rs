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

use cobre_core::{TrainingEvent, WelfordAccumulator};
use console::Term;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle, TermLike};

const TRAINING_TEMPLATE: &str =
    "Training   {bar:40} {pos}/{len} iter  {msg}  [{elapsed_precise} < {eta_precise}]";
const SIMULATION_TEMPLATE: &str =
    "Simulation {bar:40} {pos}/{len} scenarios  {msg}  [{elapsed_precise} < {eta_precise}]";

/// Format a floating-point value as scientific notation (6 significant digits).
fn fmt_sci(v: f64) -> String {
    let raw = format!("{v:.5e}");
    if let Some(pos) = raw.find('e') {
        let mantissa = &raw[..pos];
        let exp_str = &raw[pos + 1..];
        if let Ok(exp) = exp_str.parse::<i32>() {
            return format!("{mantissa}e{exp}");
        }
    }
    raw
}

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

/// Terminal wrapper that overrides the width reported to `indicatif`.
///
/// Under MPI, stderr is a non-TTY pipe, causing `indicatif`'s cursor math to
/// overshoot. This wrapper preserves the real terminal width.
#[derive(Debug)]
struct MpiTerm {
    inner: Term,
    width: u16,
}

impl TermLike for MpiTerm {
    fn width(&self) -> u16 {
        self.width
    }

    fn move_cursor_up(&self, n: usize) -> std::io::Result<()> {
        self.inner.move_cursor_up(n)
    }

    fn move_cursor_down(&self, n: usize) -> std::io::Result<()> {
        self.inner.move_cursor_down(n)
    }

    fn move_cursor_right(&self, n: usize) -> std::io::Result<()> {
        self.inner.move_cursor_right(n)
    }

    fn move_cursor_left(&self, n: usize) -> std::io::Result<()> {
        self.inner.move_cursor_left(n)
    }

    fn write_line(&self, s: &str) -> std::io::Result<()> {
        self.inner.write_line(s)
    }

    fn write_str(&self, s: &str) -> std::io::Result<()> {
        self.inner.write_str(s)
    }

    fn clear_line(&self) -> std::io::Result<()> {
        self.inner.clear_line()
    }

    fn flush(&self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Resolve the terminal width for progress bar rendering.
///
/// Tries: `Term::stderr()` detection, `$COLUMNS` environment variable, then 120.
pub fn resolve_term_width() -> u16 {
    let term = Term::stderr();
    if let Some((_, w)) = term.size_checked() {
        return w;
    }
    if let Ok(val) = std::env::var("COLUMNS") {
        if let Ok(w) = val.parse::<u16>() {
            if w > 0 {
                return w;
            }
        }
    }
    120
}

/// Spawn a background thread that renders progress bars and collects events.
#[allow(clippy::too_many_lines)]
pub fn run_progress_thread(
    receiver: mpsc::Receiver<TrainingEvent>,
    max_iterations: u64,
    term_width: u16,
) -> ProgressHandle {
    let handle = thread::spawn(move || {
        let mut events: Vec<TrainingEvent> = Vec::new();
        let mut training_bar: Option<ProgressBar> = None;
        let mut simulation_bar: Option<ProgressBar> = None;
        let mut sim_acc: Option<WelfordAccumulator> = None;

        loop {
            if let Ok(event) = receiver.recv() {
                events.push(event.clone());
                match event {
                    TrainingEvent::IterationSummary {
                        iteration,
                        lower_bound,
                        upper_bound,
                        gap,
                        lp_solves,
                        solve_time_ms,
                        ..
                    } => {
                        let bar = training_bar
                            .get_or_insert_with(|| create_training_bar(max_iterations, term_width));
                        let gap_pct = gap * 100.0;
                        bar.set_position(iteration);
                        #[allow(clippy::cast_precision_loss)]
                        let avg_lp = if lp_solves > 0 {
                            format!("LP: {:.1}ms", solve_time_ms / lp_solves as f64)
                        } else {
                            "LP: --".to_string()
                        };
                        bar.set_message(format!(
                            "LB: {}  UB: {}  gap: {gap_pct:.1}%  {avg_lp}",
                            fmt_sci(lower_bound),
                            fmt_sci(upper_bound)
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
                                "LB: {}  UB: {}  done",
                                fmt_sci(final_lb),
                                fmt_sci(final_ub)
                            ));
                            // Force a newline after the finished bar so the
                            // summary printed by the main thread starts on
                            // a fresh line.
                            let _ = Term::stderr().write_line("");
                        }
                    }

                    TrainingEvent::SimulationProgress {
                        scenarios_complete,
                        scenarios_total,
                        scenario_cost,
                        ..
                    } => {
                        let bar = simulation_bar.get_or_insert_with(|| {
                            create_simulation_bar(u64::from(scenarios_total), term_width)
                        });
                        bar.set_position(u64::from(scenarios_complete));
                        let acc = sim_acc.get_or_insert_with(WelfordAccumulator::new);
                        acc.update(scenario_cost);
                        let msg = if acc.count() >= 2 {
                            format!(
                                "mean: {}  std: {}  CI95: +/-{}",
                                fmt_sci(acc.mean()),
                                fmt_sci(acc.std_dev()),
                                fmt_sci(acc.ci_95_half_width())
                            )
                        } else {
                            format!("mean: {}", fmt_sci(acc.mean()))
                        };
                        bar.set_message(msg);
                    }

                    TrainingEvent::SimulationFinished { scenarios, .. } => {
                        if let Some(bar) = simulation_bar.take() {
                            bar.set_position(u64::from(scenarios));
                            let final_msg = if let Some(ref acc) = sim_acc {
                                if scenarios >= 2 {
                                    format!(
                                        "mean: {}  std: {}  CI95: +/-{}",
                                        fmt_sci(acc.mean()),
                                        fmt_sci(acc.std_dev()),
                                        fmt_sci(acc.ci_95_half_width())
                                    )
                                } else {
                                    format!("mean: {}", fmt_sci(acc.mean()))
                                }
                            } else {
                                "complete".to_string()
                            };
                            bar.finish_with_message(final_msg);
                            let _ = Term::stderr().write_line("");
                        }
                    }

                    TrainingEvent::TrainingStarted { .. } => {
                        let bar = training_bar
                            .get_or_insert_with(|| create_training_bar(max_iterations, term_width));
                        bar.set_position(0);
                        bar.set_message("starting...");
                    }

                    TrainingEvent::ForwardPassComplete { .. }
                    | TrainingEvent::ForwardSyncComplete { .. }
                    | TrainingEvent::BackwardPassComplete { .. }
                    | TrainingEvent::CutSyncComplete { .. }
                    | TrainingEvent::CutSelectionComplete { .. }
                    | TrainingEvent::ConvergenceUpdate { .. }
                    | TrainingEvent::CheckpointComplete { .. } => {}
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

fn create_training_bar(max_iterations: u64, term_width: u16) -> ProgressBar {
    let target = ProgressDrawTarget::term_like_with_hz(
        Box::new(MpiTerm {
            inner: Term::stderr(),
            width: term_width,
        }),
        8,
    );
    let bar = ProgressBar::with_draw_target(Some(max_iterations), target);
    let style = ProgressStyle::with_template(TRAINING_TEMPLATE)
        .unwrap_or_else(|_| ProgressStyle::default_bar());
    bar.set_style(style);
    bar
}

fn create_simulation_bar(scenarios_total: u64, term_width: u16) -> ProgressBar {
    let target = ProgressDrawTarget::term_like_with_hz(
        Box::new(MpiTerm {
            inner: Term::stderr(),
            width: term_width,
        }),
        8,
    );
    let bar = ProgressBar::with_draw_target(Some(scenarios_total), target);
    let style = ProgressStyle::with_template(SIMULATION_TEMPLATE)
        .unwrap_or_else(|_| ProgressStyle::default_bar());
    bar.set_style(style);
    bar
}

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
            solve_time_ms: 0.0,
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
            scenario_cost: 45_230.4,
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
        let handle = run_progress_thread(rx, 10, 120);

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
        let handle = run_progress_thread(rx, 10, 120);

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
        let handle = run_progress_thread(rx, 10, 120);

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
        let handle = run_progress_thread(rx, 10, 120);
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
        let handle = run_progress_thread(rx, 10, 120);

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
    fn test_simulation_progress_message_with_statistics() {
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let handle = run_progress_thread(rx, 10, 120);

        tx.send(TrainingEvent::SimulationProgress {
            scenarios_complete: 50,
            scenarios_total: 200,
            elapsed_ms: 2_500,
            scenario_cost: 45_230.4,
        })
        .unwrap();
        tx.send(make_simulation_finished()).unwrap();
        drop(tx);

        let events = handle.join();
        assert_eq!(events.len(), 2, "expected 2 events, got {}", events.len());
        assert!(
            matches!(
                events[0],
                TrainingEvent::SimulationProgress {
                    scenarios_complete: 50,
                    scenario_cost,
                    ..
                } if (scenario_cost - 45_230.4).abs() < f64::EPSILON
            ),
            "event must be SimulationProgress with expected scenario_cost"
        );
    }

    #[test]
    fn test_simulation_progress_message_single_scenario() {
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let handle = run_progress_thread(rx, 10, 120);

        tx.send(TrainingEvent::SimulationProgress {
            scenarios_complete: 1,
            scenarios_total: 200,
            elapsed_ms: 50,
            scenario_cost: 45_230.4,
        })
        .unwrap();
        tx.send(make_simulation_finished()).unwrap();
        drop(tx);

        let events = handle.join();
        assert_eq!(events.len(), 2, "expected 2 events, got {}", events.len());
        assert!(
            matches!(
                events[0],
                TrainingEvent::SimulationProgress {
                    scenarios_complete: 1,
                    ..
                }
            ),
            "first event must be SimulationProgress with scenarios_complete == 1"
        );
    }

    #[test]
    fn test_non_ui_events_are_collected() {
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let handle = run_progress_thread(rx, 10, 120);

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

    #[test]
    fn test_simulation_progress_five_events_no_panic() {
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let handle = run_progress_thread(rx, 10, 120);

        let costs = [10_000.0_f64, 20_000.0, 30_000.0, 40_000.0, 50_000.0];
        for (i, &cost) in costs.iter().enumerate() {
            let complete = u32::try_from(i + 1).unwrap();
            tx.send(TrainingEvent::SimulationProgress {
                scenarios_complete: complete,
                scenarios_total: 100,
                elapsed_ms: u64::from(complete) * 50,
                scenario_cost: cost,
            })
            .unwrap();
        }
        tx.send(make_simulation_finished()).unwrap();
        drop(tx);

        let events = handle.join();
        assert_eq!(
            events.len(),
            6,
            "expected 6 events (5 progress + 1 finished), got {}",
            events.len()
        );
        let progress_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, TrainingEvent::SimulationProgress { .. }))
            .collect();
        assert_eq!(
            progress_events.len(),
            5,
            "all 5 SimulationProgress events must be collected"
        );
    }

    #[test]
    fn test_simulation_progress_accumulator_costs_collected_correctly() {
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let handle = run_progress_thread(rx, 10, 120);

        // Known costs: mean = 300.0, values = [100, 200, 300, 400, 500].
        let costs = [100.0_f64, 200.0, 300.0, 400.0, 500.0];
        for (i, &cost) in costs.iter().enumerate() {
            tx.send(TrainingEvent::SimulationProgress {
                scenarios_complete: u32::try_from(i + 1).unwrap(),
                scenarios_total: 5,
                elapsed_ms: (i as u64 + 1) * 50,
                scenario_cost: cost,
            })
            .unwrap();
        }
        tx.send(make_simulation_finished()).unwrap();
        drop(tx);

        let events = handle.join();
        assert_eq!(events.len(), 6, "expected 6 events");

        // Extract and verify each scenario_cost was delivered unmodified.
        let collected_costs: Vec<f64> = events
            .iter()
            .filter_map(|e| {
                if let TrainingEvent::SimulationProgress { scenario_cost, .. } = e {
                    Some(*scenario_cost)
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(collected_costs.len(), 5, "must collect 5 scenario costs");
        for (expected, actual) in costs.iter().zip(collected_costs.iter()) {
            assert!(
                (expected - actual).abs() < f64::EPSILON,
                "scenario_cost mismatch: expected {expected}, got {actual}"
            );
        }

        // Independently verify WelfordAccumulator produces the correct mean
        // for this known sequence (mean of [100, 200, 300, 400, 500] = 300.0).
        let mut acc = cobre_core::WelfordAccumulator::new();
        for &c in &costs {
            acc.update(c);
        }
        assert!(
            (acc.mean() - 300.0).abs() < 1e-9,
            "WelfordAccumulator mean must be 300.0 for [100..500], got {}",
            acc.mean()
        );
        // WelfordAccumulator.std_dev() uses population variance (m2/n).
        // Population std dev of [100, 200, 300, 400, 500] = sqrt(20000) ≈ 141.421.
        let expected_std = (((100.0_f64 - 300.0_f64).powi(2)
            + (200.0_f64 - 300.0_f64).powi(2)
            + (300.0_f64 - 300.0_f64).powi(2)
            + (400.0_f64 - 300.0_f64).powi(2)
            + (500.0_f64 - 300.0_f64).powi(2))
            / 5.0_f64)
            .sqrt();
        assert!(
            (acc.std_dev() - expected_std).abs() < 1e-6,
            "WelfordAccumulator std_dev must be ~{expected_std:.3}, got {}",
            acc.std_dev()
        );
    }

    #[test]
    fn test_simulation_progress_accumulator_three_events_no_panic() {
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let handle = run_progress_thread(rx, 10, 120);

        // Send 3 events with known costs to exercise the >= 2 branch.
        tx.send(TrainingEvent::SimulationProgress {
            scenarios_complete: 1,
            scenarios_total: 200,
            elapsed_ms: 50,
            scenario_cost: 100.0,
        })
        .unwrap();
        tx.send(TrainingEvent::SimulationProgress {
            scenarios_complete: 2,
            scenarios_total: 200,
            elapsed_ms: 100,
            scenario_cost: 200.0,
        })
        .unwrap();
        tx.send(TrainingEvent::SimulationProgress {
            scenarios_complete: 3,
            scenarios_total: 200,
            elapsed_ms: 150,
            scenario_cost: 300.0,
        })
        .unwrap();
        tx.send(make_simulation_finished()).unwrap();
        drop(tx);

        let events = handle.join();
        assert_eq!(
            events.len(),
            4,
            "expected 4 events (3 progress + 1 finished)"
        );

        // Verify all 3 SimulationProgress events were collected.
        let progress_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, TrainingEvent::SimulationProgress { .. }))
            .collect();
        assert_eq!(
            progress_events.len(),
            3,
            "all 3 SimulationProgress events must be collected"
        );
    }
}
