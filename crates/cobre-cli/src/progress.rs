//! Progress reporting for training and simulation phases.
//!
//! [`run_progress_thread`] spawns a background thread that consumes
//! [`TrainingEvent`] values from an `mpsc::Receiver` and reports progress
//! through one of two [`RenderMode`] strategies:
//!
//! - [`RenderMode::Interactive`] — live-redrawn [`indicatif`] progress bars,
//!   appropriate when stderr is a user-attended terminal.
//! - [`RenderMode::Log`] — plain append-only log lines, appropriate for
//!   non-TTY streams (pipes, log files, `mpirun` aggregators, CI). Bars
//!   would otherwise degrade into streams of ANSI cursor escapes and
//!   appear as duplicated lines in captured output.
//!
//! Both strategies collect every received event verbatim and return the full
//! sequence via [`ProgressHandle::join`], so the caller can feed the events
//! into the output pipeline regardless of which strategy was used.
//!
//! ## Example
//!
//! ```rust,no_run
//! use std::sync::mpsc;
//! use cobre_core::TrainingEvent;
//! use cobre_cli::progress::{run_progress_thread, RenderMode};
//!
//! let (tx, rx) = mpsc::channel::<TrainingEvent>();
//! let handle = run_progress_thread(rx, RenderMode::auto(), 100, 120);
//! drop(tx);
//! let events = handle.join();
//! assert!(events.is_empty());
//! ```

use std::sync::mpsc;
use std::thread;

use cobre_core::TrainingEvent;
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

/// Format a millisecond count as `HH:MM:SS`.
///
/// Matches the width of `indicatif`'s `{elapsed_precise}` / `{eta_precise}`
/// tokens so log-mode lines stay visually aligned with bar-mode output for
/// the same run.
fn fmt_hms(millis: u64) -> String {
    let total_secs = millis / 1000;
    let h = total_secs / 3600;
    let m = (total_secs % 3600) / 60;
    let s = total_secs % 60;
    format!("{h:02}:{m:02}:{s:02}")
}

/// Linear-extrapolation ETA: remaining time = elapsed * (total - done) / done.
///
/// Returns `None` when `done == 0` or `done >= total` (nothing left to
/// estimate). Arithmetic is done in `u128` to avoid overflow at large
/// elapsed values.
fn eta_millis(elapsed_ms: u64, done: u64, total: u64) -> Option<u64> {
    if done == 0 || done >= total {
        return None;
    }
    let remaining = u128::from(total - done);
    let elapsed = u128::from(elapsed_ms);
    let eta = elapsed.saturating_mul(remaining) / u128::from(done);
    u64::try_from(eta).ok()
}

/// Render the trailing `[elapsed HH:MM:SS < eta HH:MM:SS]` time cell.
///
/// When `eta` is `None` (first iteration, or all work done) only elapsed is
/// shown. Mirrors the `{elapsed_precise} < {eta_precise}` segment of the
/// interactive progress bars so log-mode output carries the same info.
fn fmt_time_cell(elapsed_ms: u64, eta_ms: Option<u64>) -> String {
    match eta_ms {
        Some(eta) => format!("[{} < {}]", fmt_hms(elapsed_ms), fmt_hms(eta)),
        None => format!("[{}]", fmt_hms(elapsed_ms)),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Rendering strategy for progress events.
///
/// Chosen once per run based on whether stderr is user-attended.
#[derive(Debug, Clone, Copy)]
pub enum RenderMode {
    /// Live-redrawn [`indicatif`] bars. Correct when stderr is a TTY.
    Interactive,
    /// Plain append-only log lines. Correct for pipes, files, or
    /// `mpirun` aggregators, where ANSI cursor escapes cannot be
    /// interpreted live and would appear as duplicated output otherwise.
    Log,
}

impl RenderMode {
    /// Pick the mode from stderr's TTY status.
    ///
    /// [`RenderMode::Interactive`] when stderr is a terminal, otherwise
    /// [`RenderMode::Log`].
    #[must_use]
    pub fn auto() -> Self {
        if Term::stderr().is_term() {
            Self::Interactive
        } else {
            Self::Log
        }
    }
}

/// Handle returned by [`run_progress_thread`].
pub struct ProgressHandle {
    handle: thread::JoinHandle<Vec<TrainingEvent>>,
}

impl ProgressHandle {
    /// Wait for the progress thread to finish and return all collected events.
    ///
    /// Events are returned in receive order. The caller is expected to pass
    /// them to the output pipeline (e.g., `build_training_output`).
    ///
    /// # Panics
    ///
    /// Propagates a panic from the progress thread. If the thread panicked,
    /// this method panics with the message `"progress thread panicked"`.
    #[allow(clippy::expect_used)]
    pub fn join(self) -> Vec<TrainingEvent> {
        self.handle.join().expect("progress thread panicked")
    }
}

/// Spawn a background thread that consumes events and renders progress
/// according to `mode`.
///
/// `max_iterations` is used to size the training bar and to render the
/// `iter/max` ratio in log lines. `term_width` is consulted only in
/// [`RenderMode::Interactive`].
pub fn run_progress_thread(
    receiver: mpsc::Receiver<TrainingEvent>,
    mode: RenderMode,
    max_iterations: u64,
    term_width: u16,
) -> ProgressHandle {
    let handle = thread::spawn(move || {
        let mut renderer = ProgressRenderer::new(mode, max_iterations, term_width);
        let mut events: Vec<TrainingEvent> = Vec::new();
        for event in &receiver {
            renderer.handle(&event);
            events.push(event);
        }
        renderer.finish();
        events
    });
    ProgressHandle { handle }
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

// ---------------------------------------------------------------------------
// Internal dispatcher
// ---------------------------------------------------------------------------

/// Enum-dispatched renderer. Keeps the event loop in [`run_progress_thread`]
/// free of dynamic dispatch while still letting each mode own its state.
enum ProgressRenderer {
    Interactive(BarRenderer),
    Log(LineRenderer),
}

impl ProgressRenderer {
    fn new(mode: RenderMode, max_iterations: u64, term_width: u16) -> Self {
        match mode {
            RenderMode::Interactive => {
                Self::Interactive(BarRenderer::new(max_iterations, term_width))
            }
            RenderMode::Log => Self::Log(LineRenderer::new(max_iterations)),
        }
    }

    fn handle(&mut self, event: &TrainingEvent) {
        match self {
            Self::Interactive(r) => r.handle(event),
            Self::Log(r) => r.handle(event),
        }
    }

    /// Flush any transient state when the event channel closes.
    fn finish(&mut self) {
        match self {
            Self::Interactive(r) => r.finish(),
            // Log mode keeps no transient state.
            Self::Log(_) => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Bar renderer (TTY)
// ---------------------------------------------------------------------------

/// [`TermLike`] wrapper that overrides the width reported to `indicatif`.
///
/// Under `mpirun`, stderr may not carry a width even when the end terminal
/// is a TTY. This wrapper forwards every other operation to the real `Term`.
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

/// Drives [`indicatif`] progress bars on a user-attended terminal.
struct BarRenderer {
    max_iterations: u64,
    term_width: u16,
    training_bar: Option<ProgressBar>,
    simulation_bar: Option<ProgressBar>,
    sim_solve_time_ms: f64,
    sim_lp_count: u64,
}

impl BarRenderer {
    fn new(max_iterations: u64, term_width: u16) -> Self {
        Self {
            max_iterations,
            term_width,
            training_bar: None,
            simulation_bar: None,
            sim_solve_time_ms: 0.0,
            sim_lp_count: 0,
        }
    }

    fn handle(&mut self, event: &TrainingEvent) {
        match *event {
            TrainingEvent::TrainingStarted { .. } => {
                let bar = self.training_bar.get_or_insert_with(|| {
                    create_training_bar(self.max_iterations, self.term_width)
                });
                bar.set_position(0);
                bar.set_message("starting...");
            }
            TrainingEvent::SimulationStarted {
                n_scenarios,
                ranks,
                threads_per_rank,
                ..
            } => {
                let bar = self.simulation_bar.get_or_insert_with(|| {
                    create_simulation_bar(u64::from(n_scenarios), self.term_width)
                });
                bar.set_position(0);
                bar.set_message(format!(
                    "starting... ({ranks} ranks × {threads_per_rank} threads)"
                ));
            }
            TrainingEvent::IterationSummary {
                iteration,
                lower_bound,
                upper_bound,
                gap,
                lp_solves,
                solve_time_ms,
                ..
            } => {
                let bar = self.training_bar.get_or_insert_with(|| {
                    create_training_bar(self.max_iterations, self.term_width)
                });
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
                if let Some(bar) = self.training_bar.take() {
                    bar.set_position(iterations);
                    bar.finish_with_message(format!(
                        "LB: {}  UB: {}  done",
                        fmt_sci(final_lb),
                        fmt_sci(final_ub)
                    ));
                    // Force a newline after the finished bar so the summary
                    // printed by the main thread starts on a fresh line.
                    let _ = Term::stderr().write_line("");
                }
            }
            TrainingEvent::SimulationProgress {
                scenarios_complete,
                scenarios_total,
                solve_time_ms,
                lp_solves,
                ..
            } => {
                let bar = self.simulation_bar.get_or_insert_with(|| {
                    create_simulation_bar(u64::from(scenarios_total), self.term_width)
                });
                bar.set_position(u64::from(scenarios_complete));
                self.sim_solve_time_ms += solve_time_ms;
                self.sim_lp_count += lp_solves;
                #[allow(clippy::cast_precision_loss)]
                let msg = if self.sim_lp_count > 0 {
                    format!(
                        "LP: {:.1}ms avg",
                        self.sim_solve_time_ms / self.sim_lp_count as f64
                    )
                } else {
                    String::new()
                };
                bar.set_message(msg);
            }
            TrainingEvent::SimulationFinished { scenarios, .. } => {
                if let Some(bar) = self.simulation_bar.take() {
                    bar.set_position(u64::from(scenarios));
                    bar.finish_with_message("done");
                    let _ = Term::stderr().write_line("");
                }
            }
            TrainingEvent::ForwardPassComplete { .. }
            | TrainingEvent::ForwardSyncComplete { .. }
            | TrainingEvent::BackwardPassComplete { .. }
            | TrainingEvent::PolicySyncComplete { .. }
            | TrainingEvent::PolicySelectionComplete { .. }
            | TrainingEvent::PolicyBudgetEnforcementComplete { .. }
            | TrainingEvent::PolicyTemplateBakeComplete { .. }
            | TrainingEvent::ConvergenceUpdate { .. }
            | TrainingEvent::CheckpointComplete { .. }
            | TrainingEvent::WorkerTiming { .. } => {}
        }
    }

    fn finish(&mut self) {
        if let Some(bar) = self.training_bar.take() {
            bar.abandon();
        }
        if let Some(bar) = self.simulation_bar.take() {
            bar.abandon();
        }
    }
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

// ---------------------------------------------------------------------------
// Line renderer (non-TTY)
// ---------------------------------------------------------------------------

/// Emits one compact append-only line per training iteration and per
/// simulation-progress event.
///
/// Output format mirrors the message portion of the interactive bars so
/// monitoring workflows (`tail -f`, log scraping) see the same fields.
/// Final training/simulation summaries are printed by the main thread, so
/// `TrainingFinished` / `SimulationFinished` produce no output here.
///
/// Write errors are silently ignored, matching the fire-and-forget style of
/// the banner and the bar renderer's trailing newlines.
struct LineRenderer {
    stderr: Term,
    max_iterations: u64,
    sim_solve_time_ms: f64,
    sim_lp_count: u64,
}

impl LineRenderer {
    fn new(max_iterations: u64) -> Self {
        Self {
            stderr: Term::stderr(),
            max_iterations,
            sim_solve_time_ms: 0.0,
            sim_lp_count: 0,
        }
    }

    fn handle(&mut self, event: &TrainingEvent) {
        match *event {
            TrainingEvent::TrainingStarted { .. } => {
                let _ = self.stderr.write_line(&format!(
                    "Training   starting... (max {} iterations)",
                    self.max_iterations
                ));
            }
            TrainingEvent::SimulationStarted {
                n_scenarios,
                ranks,
                threads_per_rank,
                ..
            } => {
                let _ = self.stderr.write_line(&format!(
                    "Simulation starting... ({n_scenarios} scenarios across {ranks} ranks × {threads_per_rank} threads)"
                ));
            }
            TrainingEvent::IterationSummary {
                iteration,
                lower_bound,
                upper_bound,
                gap,
                wall_time_ms,
                lp_solves,
                solve_time_ms,
                ..
            } => {
                let gap_pct = gap * 100.0;
                #[allow(clippy::cast_precision_loss)]
                let avg_lp = if lp_solves > 0 {
                    format!("LP: {:.1}ms", solve_time_ms / lp_solves as f64)
                } else {
                    "LP: --".to_string()
                };
                let time_cell = fmt_time_cell(
                    wall_time_ms,
                    eta_millis(wall_time_ms, iteration, self.max_iterations),
                );
                let _ = self.stderr.write_line(&format!(
                    "Training   {iteration}/{max} iter  LB: {lb}  UB: {ub}  gap: {gap_pct:.1}%  {avg_lp}  {time_cell}",
                    max = self.max_iterations,
                    lb = fmt_sci(lower_bound),
                    ub = fmt_sci(upper_bound),
                ));
            }
            TrainingEvent::SimulationProgress {
                scenarios_complete,
                scenarios_total,
                elapsed_ms,
                solve_time_ms,
                lp_solves,
                ..
            } => {
                self.sim_solve_time_ms += solve_time_ms;
                self.sim_lp_count += lp_solves;
                #[allow(clippy::cast_precision_loss)]
                let avg_lp = if self.sim_lp_count > 0 {
                    format!(
                        "LP: {:.1}ms avg",
                        self.sim_solve_time_ms / self.sim_lp_count as f64
                    )
                } else {
                    String::new()
                };
                let time_cell = fmt_time_cell(
                    elapsed_ms,
                    eta_millis(
                        elapsed_ms,
                        u64::from(scenarios_complete),
                        u64::from(scenarios_total),
                    ),
                );
                let _ = self.stderr.write_line(&format!(
                    "Simulation {scenarios_complete}/{scenarios_total} scenarios  {avg_lp}  {time_cell}"
                ));
            }
            TrainingEvent::TrainingFinished { .. }
            | TrainingEvent::SimulationFinished { .. }
            | TrainingEvent::ForwardPassComplete { .. }
            | TrainingEvent::ForwardSyncComplete { .. }
            | TrainingEvent::BackwardPassComplete { .. }
            | TrainingEvent::PolicySyncComplete { .. }
            | TrainingEvent::PolicySelectionComplete { .. }
            | TrainingEvent::PolicyBudgetEnforcementComplete { .. }
            | TrainingEvent::PolicyTemplateBakeComplete { .. }
            | TrainingEvent::ConvergenceUpdate { .. }
            | TrainingEvent::CheckpointComplete { .. }
            | TrainingEvent::WorkerTiming { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use cobre_core::TrainingEvent;

    use super::{RenderMode, eta_millis, fmt_hms, fmt_time_cell, run_progress_thread};

    #[test]
    fn fmt_hms_pads_hours_minutes_seconds() {
        assert_eq!(fmt_hms(0), "00:00:00");
        assert_eq!(fmt_hms(999), "00:00:00");
        assert_eq!(fmt_hms(1_000), "00:00:01");
        assert_eq!(fmt_hms(61_500), "00:01:01");
        assert_eq!(fmt_hms(3_600_000), "01:00:00");
        assert_eq!(fmt_hms(90 * 60_000 + 45_000), "01:30:45");
        assert_eq!(fmt_hms(100 * 3_600_000), "100:00:00");
    }

    #[test]
    fn eta_millis_none_when_done_is_zero() {
        assert_eq!(eta_millis(5_000, 0, 10), None);
    }

    #[test]
    fn eta_millis_none_when_done_reaches_total() {
        assert_eq!(eta_millis(5_000, 10, 10), None);
        assert_eq!(eta_millis(5_000, 11, 10), None);
    }

    #[test]
    fn eta_millis_linear_extrapolation() {
        // elapsed = 10s after 2/10 iterations -> 40s remaining (5s/iter × 8).
        assert_eq!(eta_millis(10_000, 2, 10), Some(40_000));
        // elapsed = 30s after 3/6 iterations -> 30s remaining.
        assert_eq!(eta_millis(30_000, 3, 6), Some(30_000));
    }

    #[test]
    fn fmt_time_cell_shows_both_when_eta_present() {
        assert_eq!(
            fmt_time_cell(61_500, Some(120_000)),
            "[00:01:01 < 00:02:00]"
        );
    }

    #[test]
    fn fmt_time_cell_shows_elapsed_only_when_eta_absent() {
        assert_eq!(fmt_time_cell(61_500, None), "[00:01:01]");
    }

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
            lower_bound_eval_ms: 0,
            fwd_setup_time_ms: 0,
            fwd_load_imbalance_ms: 0,
            fwd_scheduling_overhead_ms: 0,
        }
    }

    fn make_training_finished() -> TrainingEvent {
        TrainingEvent::TrainingFinished {
            reason: "iteration_limit".to_string(),
            iterations: 3,
            final_lb: 105.0,
            final_ub: 106.0,
            total_time_ms: 600,
            total_rows: 144,
        }
    }

    fn make_simulation_progress(complete: u32, total: u32) -> TrainingEvent {
        TrainingEvent::SimulationProgress {
            scenarios_complete: complete,
            scenarios_total: total,
            elapsed_ms: u64::from(complete) * 50,
            scenario_cost: 45_230.4,
            solve_time_ms: 0.0,
            lp_solves: 0,
        }
    }

    fn make_simulation_finished() -> TrainingEvent {
        TrainingEvent::SimulationFinished {
            scenarios: 200,
            output_dir: "/tmp/output".to_string(),
            elapsed_ms: 10_000,
        }
    }

    fn make_simulation_started(n_scenarios: u32, ranks: u32) -> TrainingEvent {
        TrainingEvent::SimulationStarted {
            case_name: "test-case".to_string(),
            n_scenarios,
            n_stages: 12,
            ranks,
            threads_per_rank: 4,
            timestamp: "2026-04-24T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn simulation_started_event_is_preserved_and_renders_banner() {
        // Log mode exercises the `SimulationStarted` arm and writes a banner
        // to stderr (not captured here); we rely on the channel round-trip
        // to verify event propagation.
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let handle = run_progress_thread(rx, RenderMode::Log, 10, 120);

        tx.send(make_simulation_started(100, 2)).unwrap();
        tx.send(make_simulation_progress(2, 100)).unwrap();
        tx.send(make_simulation_progress(100, 100)).unwrap();
        drop(tx);

        let events = handle.join();
        assert_eq!(events.len(), 3);
        assert!(matches!(
            events[0],
            TrainingEvent::SimulationStarted {
                n_scenarios: 100,
                ranks: 2,
                ..
            }
        ));
    }

    #[test]
    fn test_progress_handle_training_events_returned() {
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let handle = run_progress_thread(rx, RenderMode::Interactive, 10, 120);

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
        let handle = run_progress_thread(rx, RenderMode::Interactive, 10, 120);

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
        let handle = run_progress_thread(rx, RenderMode::Interactive, 10, 120);

        tx.send(make_iteration_summary(1)).unwrap();
        tx.send(make_iteration_summary(2)).unwrap();
        tx.send(make_training_finished()).unwrap();
        tx.send(make_simulation_progress(50, 200)).unwrap();
        tx.send(make_simulation_finished()).unwrap();
        drop(tx);

        let events = handle.join();
        assert_eq!(events.len(), 5, "expected 5 events, got {}", events.len());

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
        let handle = run_progress_thread(rx, RenderMode::Interactive, 10, 120);
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
        let handle = run_progress_thread(rx, RenderMode::Interactive, 10, 120);

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
        let handle = run_progress_thread(rx, RenderMode::Interactive, 10, 120);

        tx.send(TrainingEvent::SimulationProgress {
            scenarios_complete: 50,
            scenarios_total: 200,
            elapsed_ms: 2_500,
            scenario_cost: 45_230.4,
            solve_time_ms: 0.0,
            lp_solves: 0,
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
        let handle = run_progress_thread(rx, RenderMode::Interactive, 10, 120);

        tx.send(TrainingEvent::SimulationProgress {
            scenarios_complete: 1,
            scenarios_total: 200,
            elapsed_ms: 50,
            scenario_cost: 45_230.4,
            solve_time_ms: 0.0,
            lp_solves: 0,
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
        let handle = run_progress_thread(rx, RenderMode::Interactive, 10, 120);

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
            rows_generated: 48,
            stages_processed: 12,
            elapsed_ms: 87,
            state_exchange_time_ms: 0,
            row_batch_build_time_ms: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
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
        let handle = run_progress_thread(rx, RenderMode::Interactive, 10, 120);

        let costs = [10_000.0_f64, 20_000.0, 30_000.0, 40_000.0, 50_000.0];
        for (i, &cost) in costs.iter().enumerate() {
            let complete = u32::try_from(i + 1).unwrap();
            tx.send(TrainingEvent::SimulationProgress {
                scenarios_complete: complete,
                scenarios_total: 100,
                elapsed_ms: u64::from(complete) * 50,
                scenario_cost: cost,
                solve_time_ms: 0.0,
                lp_solves: 0,
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
        let handle = run_progress_thread(rx, RenderMode::Interactive, 10, 120);

        let costs = [100.0_f64, 200.0, 300.0, 400.0, 500.0];
        for (i, &cost) in costs.iter().enumerate() {
            tx.send(TrainingEvent::SimulationProgress {
                scenarios_complete: u32::try_from(i + 1).unwrap(),
                scenarios_total: 5,
                elapsed_ms: (i as u64 + 1) * 50,
                scenario_cost: cost,
                solve_time_ms: 0.0,
                lp_solves: 0,
            })
            .unwrap();
        }
        tx.send(make_simulation_finished()).unwrap();
        drop(tx);

        let events = handle.join();
        assert_eq!(events.len(), 6, "expected 6 events");

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

        let mut acc = cobre_core::WelfordAccumulator::new();
        for &c in &costs {
            acc.update(c);
        }
        assert!(
            (acc.mean() - 300.0).abs() < 1e-9,
            "WelfordAccumulator mean must be 300.0 for [100..500], got {}",
            acc.mean()
        );
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
        let handle = run_progress_thread(rx, RenderMode::Interactive, 10, 120);

        tx.send(TrainingEvent::SimulationProgress {
            scenarios_complete: 1,
            scenarios_total: 200,
            elapsed_ms: 50,
            scenario_cost: 100.0,
            solve_time_ms: 0.0,
            lp_solves: 0,
        })
        .unwrap();
        tx.send(TrainingEvent::SimulationProgress {
            scenarios_complete: 2,
            scenarios_total: 200,
            elapsed_ms: 100,
            scenario_cost: 200.0,
            solve_time_ms: 0.0,
            lp_solves: 0,
        })
        .unwrap();
        tx.send(TrainingEvent::SimulationProgress {
            scenarios_complete: 3,
            scenarios_total: 200,
            elapsed_ms: 150,
            scenario_cost: 300.0,
            solve_time_ms: 0.0,
            lp_solves: 0,
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

    // -------------------------------------------------------------------
    // Log mode coverage
    // -------------------------------------------------------------------

    /// Log mode must collect the same events as bar mode and not panic when
    /// rendering writes to the (captured-in-tests) stderr.
    #[test]
    fn test_log_mode_training_events_round_trip() {
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let handle = run_progress_thread(rx, RenderMode::Log, 5, 120);

        tx.send(TrainingEvent::TrainingStarted {
            case_name: "test".to_string(),
            stages: 12,
            hydros: 4,
            thermals: 126,
            ranks: 1,
            threads_per_rank: 1,
            timestamp: "2026-04-22T00:00:00Z".to_string(),
        })
        .unwrap();
        for i in 1..=5 {
            tx.send(make_iteration_summary(i)).unwrap();
        }
        tx.send(make_training_finished()).unwrap();
        drop(tx);

        let events = handle.join();
        assert_eq!(
            events.len(),
            7,
            "expected 7 events (started + 5 summaries + finished), got {}",
            events.len()
        );
    }

    #[test]
    fn test_log_mode_simulation_events_round_trip() {
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let handle = run_progress_thread(rx, RenderMode::Log, 0, 120);

        for i in 1..=3 {
            tx.send(make_simulation_progress(i, 3)).unwrap();
        }
        tx.send(make_simulation_finished()).unwrap();
        drop(tx);

        let events = handle.join();
        assert_eq!(events.len(), 4, "expected 4 events, got {}", events.len());
    }

    #[test]
    fn test_log_mode_ignores_non_ui_events_without_panic() {
        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let handle = run_progress_thread(rx, RenderMode::Log, 2, 120);

        tx.send(TrainingEvent::ForwardPassComplete {
            iteration: 1,
            scenarios: 10,
            ub_mean: 110.0,
            ub_std: 5.0,
            elapsed_ms: 42,
        })
        .unwrap();
        tx.send(make_iteration_summary(1)).unwrap();
        drop(tx);

        let events = handle.join();
        assert_eq!(events.len(), 2, "expected 2 events, got {}", events.len());
    }

    #[test]
    fn test_render_mode_auto_returns_a_variant() {
        // `is_term()` depends on the test runner; just assert we resolve to
        // one of the two defined variants (the match is exhaustive, so a
        // future variant would force this test to be updated).
        match RenderMode::auto() {
            RenderMode::Interactive | RenderMode::Log => {}
        }
    }
}
