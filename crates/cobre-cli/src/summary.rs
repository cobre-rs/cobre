//! Post-run summary block for the `cobre run` command.
//!
//! Formats a multi-line summary of training convergence metrics, cut statistics,
//! and optional simulation results after a run completes.
//!
//! ## Design
//!
//! The module separates formatting from printing: [`format_summary_string`] builds
//! a plain-text `String` without any ANSI escapes (enabling unit testing without a
//! real terminal), and [`print_summary`] applies `console::style` for bold headers
//! and dim file paths when writing to stderr.
//!
//! ## Example
//!
//! ```rust,no_run
//! use std::path::PathBuf;
//! use console::Term;
//! use cobre_cli::summary::{RunSummary, TrainingSummary, print_summary};
//!
//! let summary = RunSummary {
//!     training: TrainingSummary {
//!         iterations: 50,
//!         converged: false,
//!         converged_at: None,
//!         reason: "iteration_limit".to_string(),
//!         lower_bound: 45230.41,
//!         upper_bound: 47800.0,
//!         upper_bound_std: 310.5,
//!         gap_percent: 5.7,
//!         total_cuts_active: 480,
//!         total_cuts_generated: 1200,
//!         total_lp_solves: 36000,
//!         total_time_ms: 222_000,
//!     },
//!     simulation: None,
//!     output_dir: PathBuf::from("/results/study-001"),
//! };
//!
//! print_summary(&Term::buffered_stderr(), &summary);
//! ```

use std::path::PathBuf;

use console::Term;

/// Training convergence metrics and timing for display in the post-run summary.
pub struct TrainingSummary {
    /// Total number of iterations completed.
    pub iterations: u64,

    /// Whether training converged within the configured tolerance.
    pub converged: bool,

    /// Iteration at which convergence was detected, if applicable.
    ///
    /// Populated when `converged` is `true` and a convergence iteration is
    /// known. `None` when training terminated for another reason.
    pub converged_at: Option<u64>,

    /// Human-readable termination reason (e.g., `"iteration_limit"`).
    pub reason: String,

    /// Final lower bound on the optimal value ($/stage).
    pub lower_bound: f64,

    /// Final upper bound estimate ($/stage).
    pub upper_bound: f64,

    /// Standard deviation of the upper bound estimate across forward-pass scenarios.
    pub upper_bound_std: f64,

    /// Relative gap between upper and lower bounds as a percentage.
    pub gap_percent: f64,

    /// Number of Benders cuts active in the pool at the end of training.
    pub total_cuts_active: u64,

    /// Total number of Benders cuts generated over the entire training run.
    pub total_cuts_generated: u64,

    /// Total number of LP solves across all stages, iterations, and passes.
    pub total_lp_solves: u64,

    /// Total elapsed wall-clock time for the training run (milliseconds).
    pub total_time_ms: u64,
}

/// Simulation completion statistics for display in the post-run summary.
pub struct SimulationSummary {
    /// Total number of scenarios dispatched for simulation.
    pub n_scenarios: u32,

    /// Number of scenarios that completed without error.
    pub completed: u32,

    /// Number of scenarios that failed during simulation.
    pub failed: u32,
}

/// All data needed to render the post-run summary block.
pub struct RunSummary {
    /// Training convergence and timing data.
    pub training: TrainingSummary,

    /// Simulation completion data, or `None` when simulation was skipped.
    pub simulation: Option<SimulationSummary>,

    /// Root output directory where all artifacts were written.
    pub output_dir: PathBuf,
}

/// Format a millisecond duration as a human-readable string.
///
/// - Under 60 s: `"12.3s"` (one decimal place using tenths of a second).
/// - Under 1 h: `"3m 42s"`.
/// - 1 h or more: `"1h 23m"`.
fn format_duration(ms: u64) -> String {
    let total_secs = ms / 1000;
    if total_secs < 60 {
        let frac = (ms % 1000) / 100;
        format!("{total_secs}.{frac}s")
    } else if total_secs < 3600 {
        let mins = total_secs / 60;
        let secs = total_secs % 60;
        format!("{mins}m {secs}s")
    } else {
        let hours = total_secs / 3600;
        let mins = (total_secs % 3600) / 60;
        format!("{hours}h {mins}m")
    }
}

/// Build the convergence detail fragment for the first summary line.
///
/// When `converged` is `true` and `converged_at` is `Some(iter)`, returns
/// `"converged at iter {iter}"`. Otherwise returns the raw `reason` string.
fn format_convergence_detail(converged: bool, converged_at: Option<u64>, reason: &str) -> String {
    if converged {
        if let Some(iter) = converged_at {
            return format!("converged at iter {iter}");
        }
    }
    reason.to_string()
}

/// Render the complete post-run summary as a plain-text `String`.
///
/// The returned string contains no ANSI escape sequences. Color and styling
/// are applied by [`print_summary`] when writing to the terminal. This design
/// allows unit tests to assert on summary content without requiring a real
/// terminal.
///
/// # Format
///
/// ```text
/// Training complete in {time} ({iterations} iterations, {reason_detail})
///   Lower bound:  {lb} $/stage
///   Upper bound:  {ub} +/- {std} $/stage
///   Gap:          {gap}%
///   Cuts:         {active} active / {generated} generated
///   LP solves:    {total_lp}
///
/// Simulation complete ({scenarios} scenarios)
///   Completed: {completed}  Failed: {failed}
///
/// Output written to {output_dir}/
/// ```
///
/// The simulation section is omitted entirely when `summary.simulation` is `None`.
#[cfg(test)]
pub fn format_summary_string(summary: &RunSummary) -> String {
    let t = &summary.training;
    let duration = format_duration(t.total_time_ms);
    let convergence_detail = format_convergence_detail(t.converged, t.converged_at, &t.reason);

    let mut lines: Vec<String> = Vec::new();

    // Training section header
    lines.push(format!(
        "Training complete in {duration} ({} iterations, {convergence_detail})",
        t.iterations
    ));
    lines.push(format!("  Lower bound:  {:.1} $/stage", t.lower_bound));
    lines.push(format!(
        "  Upper bound:  {:.1} +/- {:.1} $/stage",
        t.upper_bound, t.upper_bound_std
    ));
    lines.push(format!("  Gap:          {:.1}%", t.gap_percent));
    lines.push(format!(
        "  Cuts:         {} active / {} generated",
        t.total_cuts_active, t.total_cuts_generated
    ));
    lines.push(format!("  LP solves:    {}", t.total_lp_solves));

    // Simulation section (optional)
    if let Some(sim) = &summary.simulation {
        lines.push(String::new());
        lines.push(format!(
            "Simulation complete ({} scenarios)",
            sim.n_scenarios
        ));
        lines.push(format!(
            "  Completed: {}  Failed: {}",
            sim.completed, sim.failed
        ));
    }

    // Output path
    lines.push(String::new());
    lines.push(format!(
        "Output written to {}/",
        summary.output_dir.display()
    ));

    lines.join("\n")
}

/// Write the post-run summary block to `stderr`.
///
/// Section headers ("Training complete", "Simulation complete", "Output written")
/// are rendered bold. The output directory path is rendered muted (dim).
/// Numerical values use plain text for copy-paste friendliness. Write errors
/// are silently ignored (fire-and-forget, same pattern as [`crate::banner`]).
///
/// When `summary.simulation` is `None`, the "Simulation complete" section is
/// omitted entirely.
pub fn print_summary(stderr: &Term, summary: &RunSummary) {
    let t = &summary.training;
    let duration = format_duration(t.total_time_ms);
    let convergence_detail = format_convergence_detail(t.converged, t.converged_at, &t.reason);

    // Training section header — bold
    let _ = stderr.write_line(&format!(
        "{} ({} iterations, {convergence_detail})",
        console::style(format!("Training complete in {duration}")).bold(),
        t.iterations
    ));
    let _ = stderr.write_line(&format!("  Lower bound:  {:.1} $/stage", t.lower_bound));
    let _ = stderr.write_line(&format!(
        "  Upper bound:  {:.1} +/- {:.1} $/stage",
        t.upper_bound, t.upper_bound_std
    ));
    let _ = stderr.write_line(&format!("  Gap:          {:.1}%", t.gap_percent));
    let _ = stderr.write_line(&format!(
        "  Cuts:         {} active / {} generated",
        t.total_cuts_active, t.total_cuts_generated
    ));
    let _ = stderr.write_line(&format!("  LP solves:    {}", t.total_lp_solves));

    // Simulation section (optional)
    if let Some(sim) = &summary.simulation {
        let _ = stderr.write_line("");
        let _ = stderr.write_line(&format!(
            "{} ({} scenarios)",
            console::style("Simulation complete").bold(),
            sim.n_scenarios
        ));
        let _ = stderr.write_line(&format!(
            "  Completed: {}  Failed: {}",
            sim.completed, sim.failed
        ));
    }

    // Output path — dim
    let _ = stderr.write_line("");
    let _ = stderr.write_line(&format!(
        "{} {}/",
        console::style("Output written to").bold(),
        console::style(summary.output_dir.display()).dim()
    ));
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use console::Term;

    use super::{
        RunSummary, SimulationSummary, TrainingSummary, format_duration, format_summary_string,
        print_summary,
    };

    fn make_training_summary() -> TrainingSummary {
        TrainingSummary {
            iterations: 50,
            converged: false,
            converged_at: None,
            reason: "iteration_limit".to_string(),
            lower_bound: 100.0,
            upper_bound: 105.0,
            upper_bound_std: 2.5,
            gap_percent: 4.8,
            total_cuts_active: 480,
            total_cuts_generated: 1200,
            total_lp_solves: 36_000,
            total_time_ms: 5_000,
        }
    }

    fn make_run_summary(simulation: Option<SimulationSummary>) -> RunSummary {
        RunSummary {
            training: make_training_summary(),
            simulation,
            output_dir: PathBuf::from("/results/study-001"),
        }
    }

    #[test]
    fn test_format_duration_seconds() {
        assert_eq!(format_duration(12_300), "12.3s");
    }

    #[test]
    fn test_format_duration_minutes() {
        assert_eq!(format_duration(222_000), "3m 42s");
    }

    #[test]
    fn test_format_duration_hours() {
        assert_eq!(format_duration(4_980_000), "1h 23m");
    }

    #[test]
    fn test_format_duration_exactly_zero() {
        assert_eq!(format_duration(0), "0.0s");
    }

    #[test]
    fn test_format_duration_exactly_60s() {
        assert_eq!(format_duration(60_000), "1m 0s");
    }

    #[test]
    fn test_format_duration_exactly_1h() {
        assert_eq!(format_duration(3_600_000), "1h 0m");
    }

    #[test]
    fn test_format_summary_training_only() {
        let summary = make_run_summary(None);
        let s = format_summary_string(&summary);

        assert!(
            s.contains("Training complete"),
            "summary must contain 'Training complete'"
        );
        assert!(
            !s.contains("Simulation"),
            "summary must NOT contain 'Simulation' when simulation is None, got: {s}"
        );
    }

    #[test]
    fn test_format_summary_with_simulation() {
        let sim = SimulationSummary {
            n_scenarios: 200,
            completed: 198,
            failed: 2,
        };
        let summary = make_run_summary(Some(sim));
        let s = format_summary_string(&summary);

        assert!(
            s.contains("Training complete"),
            "summary must contain 'Training complete'"
        );
        assert!(
            s.contains("Simulation complete"),
            "summary must contain 'Simulation complete' when simulation is Some"
        );
    }

    #[test]
    fn test_format_summary_contains_bounds() {
        let summary = RunSummary {
            training: TrainingSummary {
                lower_bound: 100.5,
                ..make_training_summary()
            },
            simulation: None,
            output_dir: PathBuf::from("/tmp/out"),
        };
        let s = format_summary_string(&summary);

        assert!(
            s.contains("100.5"),
            "summary must contain '100.5' for lower_bound = 100.5, got: {s}"
        );
    }

    #[test]
    fn test_format_summary_converged_detail() {
        let summary = RunSummary {
            training: TrainingSummary {
                converged: true,
                converged_at: Some(38),
                reason: "bound_stalling".to_string(),
                ..make_training_summary()
            },
            simulation: None,
            output_dir: PathBuf::from("/tmp/out"),
        };
        let s = format_summary_string(&summary);

        assert!(
            s.contains("converged at iter 38"),
            "summary must contain 'converged at iter 38', got: {s}"
        );
    }

    #[test]
    fn test_format_summary_non_converged_shows_reason() {
        let summary = RunSummary {
            training: TrainingSummary {
                converged: false,
                converged_at: None,
                reason: "iteration_limit".to_string(),
                ..make_training_summary()
            },
            simulation: None,
            output_dir: PathBuf::from("/tmp/out"),
        };
        let s = format_summary_string(&summary);

        assert!(
            s.contains("iteration_limit"),
            "summary must contain the termination reason when not converged, got: {s}"
        );
    }

    #[test]
    fn test_format_summary_time_3m42s() {
        let summary = RunSummary {
            training: TrainingSummary {
                total_time_ms: 222_000,
                ..make_training_summary()
            },
            simulation: None,
            output_dir: PathBuf::from("/tmp/out"),
        };
        let s = format_summary_string(&summary);

        assert!(
            s.contains("3m 42s"),
            "summary must contain '3m 42s' for total_time_ms = 222_000, got: {s}"
        );
    }

    #[test]
    fn test_format_summary_one_decimal_place() {
        let summary = RunSummary {
            training: TrainingSummary {
                lower_bound: 45230.41,
                ..make_training_summary()
            },
            simulation: None,
            output_dir: PathBuf::from("/tmp/out"),
        };
        let s = format_summary_string(&summary);

        assert!(
            s.contains("45230.4"),
            "summary must contain '45230.4' (one decimal place) for lower_bound = 45230.41, got: {s}"
        );
    }

    #[test]
    fn test_format_summary_output_dir() {
        let summary = RunSummary {
            training: make_training_summary(),
            simulation: None,
            output_dir: PathBuf::from("/my/output/dir"),
        };
        let s = format_summary_string(&summary);

        assert!(
            s.contains("/my/output/dir"),
            "summary must contain the output_dir path, got: {s}"
        );
    }

    #[test]
    fn test_format_summary_cut_stats() {
        let summary = RunSummary {
            training: TrainingSummary {
                total_cuts_active: 480,
                total_cuts_generated: 1200,
                ..make_training_summary()
            },
            simulation: None,
            output_dir: PathBuf::from("/tmp/out"),
        };
        let s = format_summary_string(&summary);

        assert!(
            s.contains("480 active / 1200 generated"),
            "summary must contain cut counts, got: {s}"
        );
    }

    #[test]
    fn test_print_summary_does_not_panic() {
        let summary = make_run_summary(None);
        print_summary(&Term::buffered_stderr(), &summary);
    }

    #[test]
    fn test_print_summary_with_simulation_does_not_panic() {
        let sim = SimulationSummary {
            n_scenarios: 100,
            completed: 100,
            failed: 0,
        };
        let summary = make_run_summary(Some(sim));
        print_summary(&Term::buffered_stderr(), &summary);
    }
}
