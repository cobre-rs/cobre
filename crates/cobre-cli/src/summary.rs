//! Post-run summary block for the `cobre run` command.
//!
//! Provides separate printing functions for each phase of the run:
//! - [`print_stochastic_summary`] — stochastic preprocessing statistics
//! - [`print_training_summary`] — training convergence metrics
//! - [`print_simulation_summary`] — simulation completion stats
//! - [`print_output_path`] — output directory location
//!
//! Each function prints its section independently so the caller can display
//! results at the right point in the execution flow.

use console::Term;

/// Source of stochastic data for a given component.
#[derive(Debug)]
pub enum StochasticSource {
    /// Data was estimated from historical records.
    Estimated,
    /// Data was loaded from user-supplied files.
    Loaded,
    /// No data available (component not modeled).
    None,
}

/// Summary of AR order selection across hydro plants.
#[derive(Debug)]
pub struct ArOrderSummary {
    /// Method used for order selection (e.g., `"AIC"`, `"fixed"`).
    pub method: String,
    /// Count of hydros at each AR order. Index = order, value = count.
    ///
    /// For example, `[0, 3, 2]` means 0 hydros at order 0, 3 at order 1,
    /// 2 at order 2.
    pub order_counts: Vec<usize>,
    /// Minimum AR order across all hydros.
    pub min_order: usize,
    /// Maximum AR order across all hydros.
    pub max_order: usize,
    /// Number of hydro plants included in the summary.
    pub n_hydros: usize,
}

impl ArOrderSummary {
    /// Render a compact human-readable string describing the AR order
    /// distribution.
    ///
    /// Three tiers based on the number of hydro plants:
    ///
    /// - **≤10 hydros**: compact distribution, e.g. `"AIC (3x order-1, 2x order-2)"`.
    /// - **11–30 hydros**: range format, e.g. `"AIC (orders 1-4, 15 hydros)"`.
    /// - **31+ hydros**: histogram format, e.g. `"AIC (order 1: 12, order 2: 8, 31 hydros)"`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_cli::summary::ArOrderSummary;
    ///
    /// let s = ArOrderSummary {
    ///     method: "AIC".into(),
    ///     order_counts: vec![0, 3, 2],
    ///     min_order: 1,
    ///     max_order: 2,
    ///     n_hydros: 5,
    /// };
    /// let text = s.display_string();
    /// assert!(text.contains("3x order-1"));
    /// assert!(text.contains("2x order-2"));
    /// ```
    pub fn display_string(&self) -> String {
        if self.n_hydros <= 10 {
            let parts: Vec<String> = self
                .order_counts
                .iter()
                .enumerate()
                .filter(|&(_, count)| *count > 0)
                .map(|(order, count)| format!("{count}x order-{order}"))
                .collect();
            format!("{} ({})", self.method, parts.join(", "))
        } else if self.n_hydros <= 30 {
            format!(
                "{} (orders {}-{}, {} hydros)",
                self.method, self.min_order, self.max_order, self.n_hydros
            )
        } else {
            let parts: Vec<String> = self
                .order_counts
                .iter()
                .enumerate()
                .filter(|&(_, count)| *count > 0)
                .map(|(order, count)| format!("order {order}: {count}"))
                .collect();
            format!(
                "{} ({}, {} hydros)",
                self.method,
                parts.join(", "),
                self.n_hydros
            )
        }
    }
}

/// Summary of the stochastic preprocessing pipeline for display.
#[derive(Debug)]
pub struct StochasticSummary {
    /// Source of inflow seasonal statistics.
    pub inflow_source: StochasticSource,
    /// Number of hydro plants in the system.
    pub n_hydros: usize,
    /// Number of seasons in the PAR model.
    pub n_seasons: usize,
    /// AR order summary (`None` if no hydros or no AR model).
    pub ar_summary: Option<ArOrderSummary>,
    /// Source of correlation data.
    pub correlation_source: StochasticSource,
    /// Dimension of the correlation matrix (e.g., `"5x5"`).
    pub correlation_dim: Option<String>,
    /// Source of the opening tree.
    pub opening_tree_source: StochasticSource,
    /// Number of openings at each stage.
    pub openings_per_stage: Vec<usize>,
    /// Number of stages in the stochastic context.
    pub n_stages: usize,
    /// Number of buses with stochastic load noise.
    pub n_load_buses: usize,
    /// Random seed used for noise generation.
    pub seed: u64,
}

/// Format a [`StochasticSource`] variant as a short label string.
fn source_label(source: &StochasticSource) -> &'static str {
    match source {
        StochasticSource::Estimated => "estimated",
        StochasticSource::Loaded => "loaded",
        StochasticSource::None => "none",
    }
}

/// Format the openings-per-stage information compactly.
///
/// - All stages same count: `"20 openings/stage"`
/// - Varying counts: `"10-20 openings/stage"`
/// - Empty: `"0 openings/stage"`
fn format_openings_per_stage(openings: &[usize]) -> String {
    if openings.is_empty() {
        return "0 openings/stage".to_string();
    }
    let min = openings.iter().copied().min().unwrap_or(0);
    let max = openings.iter().copied().max().unwrap_or(0);
    if min == max {
        format!("{min} openings/stage")
    } else {
        format!("{min}-{max} openings/stage")
    }
}

/// Print the stochastic preprocessing summary to `stderr`.
///
/// Renders a bold header followed by indented lines covering seed, inflow
/// source, AR order distribution, correlation, opening tree, and load noise.
/// Write errors are silently ignored (fire-and-forget).
pub fn print_stochastic_summary(stderr: &Term, summary: &StochasticSummary) {
    let _ = stderr.write_line(&format!(
        "{}",
        console::style("Stochastic preprocessing").bold()
    ));
    let _ = stderr.write_line(&format!(
        "  Seed:          {}",
        console::style(summary.seed).bold()
    ));
    let _ = stderr.write_line(&format!(
        "  Inflow stats:  {} ({} hydros, {} seasons)",
        source_label(&summary.inflow_source),
        summary.n_hydros,
        summary.n_seasons,
    ));
    if let Some(ref ar) = summary.ar_summary {
        let _ = stderr.write_line(&format!(
            "  AR orders:     {}",
            console::style(ar.display_string()).bold()
        ));
    }
    let correlation_detail = match &summary.correlation_dim {
        Some(dim) => format!("{} ({})", source_label(&summary.correlation_source), dim),
        None => source_label(&summary.correlation_source).to_string(),
    };
    let _ = stderr.write_line(&format!("  Correlation:   {correlation_detail}"));
    let openings_detail = format_openings_per_stage(&summary.openings_per_stage);
    let _ = stderr.write_line(&format!(
        "  Opening tree:  {} ({openings_detail}, {} stages)",
        source_label(&summary.opening_tree_source),
        summary.n_stages,
    ));
    let _ = stderr.write_line(&format!(
        "  Load noise:    {} stochastic buses",
        summary.n_load_buses
    ));
}

/// Render the stochastic preprocessing summary as a plain-text `String`.
///
/// The returned string contains no ANSI escape sequences. Color and styling
/// are applied by [`print_stochastic_summary`] when writing to the terminal.
/// This function exists to allow unit tests to assert on summary content
/// without requiring a real terminal.
#[cfg(test)]
pub fn format_stochastic_summary_string(summary: &StochasticSummary) -> String {
    let mut lines: Vec<String> = Vec::new();

    lines.push("Stochastic preprocessing".to_string());
    lines.push(format!("  Seed:          {}", summary.seed));
    lines.push(format!(
        "  Inflow stats:  {} ({} hydros, {} seasons)",
        source_label(&summary.inflow_source),
        summary.n_hydros,
        summary.n_seasons,
    ));
    if let Some(ref ar) = summary.ar_summary {
        lines.push(format!("  AR orders:     {}", ar.display_string()));
    }
    let correlation_detail = match &summary.correlation_dim {
        Some(dim) => format!("{} ({})", source_label(&summary.correlation_source), dim),
        None => source_label(&summary.correlation_source).to_string(),
    };
    lines.push(format!("  Correlation:   {correlation_detail}"));
    let openings_detail = format_openings_per_stage(&summary.openings_per_stage);
    lines.push(format!(
        "  Opening tree:  {} ({openings_detail}, {} stages)",
        source_label(&summary.opening_tree_source),
        summary.n_stages,
    ));
    lines.push(format!(
        "  Load noise:    {} stochastic buses",
        summary.n_load_buses
    ));

    lines.join("\n")
}

/// Format a floating-point value as scientific notation with 6 significant digits.
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

    /// Total number of LP solves across all ranks, stages, iterations, and
    /// passes.  Aggregated via `allreduce(Sum)` so that the reported value is
    /// invariant regardless of the parallel configuration.
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

    /// Total elapsed wall-clock time for the simulation phase (milliseconds).
    pub total_time_ms: u64,
}

/// All data needed to render the complete post-run summary block.
///
/// Used only in tests; the production code calls the individual print functions.
#[cfg(test)]
pub struct RunSummary {
    pub training: TrainingSummary,
    pub simulation: Option<SimulationSummary>,
    pub output_dir: std::path::PathBuf,
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
    lines.push(format!(
        "  Lower bound:  {} $/stage",
        fmt_sci(t.lower_bound)
    ));
    lines.push(format!(
        "  Upper bound:  {} +/- {} $/stage",
        fmt_sci(t.upper_bound),
        fmt_sci(t.upper_bound_std)
    ));
    lines.push(format!("  Gap:          {:.1}%", t.gap_percent));
    lines.push(format!(
        "  Cuts:         {} active / {} generated",
        t.total_cuts_active, t.total_cuts_generated
    ));
    lines.push(format!("  LP solves:    {}", t.total_lp_solves));

    // Simulation section (optional)
    if let Some(sim) = &summary.simulation {
        let sim_duration = format_duration(sim.total_time_ms);
        lines.push(String::new());
        lines.push(format!(
            "Simulation complete in {sim_duration} ({} scenarios)",
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

/// Print the training completion summary to `stderr`.
///
/// Rendered bold header with convergence metrics, bounds, cuts, and LP solves.
/// Write errors are silently ignored (fire-and-forget).
pub fn print_training_summary(stderr: &Term, t: &TrainingSummary) {
    let duration = format_duration(t.total_time_ms);
    let convergence_detail = format_convergence_detail(t.converged, t.converged_at, &t.reason);

    let _ = stderr.write_line(&format!(
        "{} ({} iterations, {convergence_detail})",
        console::style(format!("Training complete in {duration}")).bold(),
        t.iterations
    ));
    let _ = stderr.write_line(&format!(
        "  Lower bound:  {} $/stage",
        fmt_sci(t.lower_bound)
    ));
    let _ = stderr.write_line(&format!(
        "  Upper bound:  {} +/- {} $/stage",
        fmt_sci(t.upper_bound),
        fmt_sci(t.upper_bound_std)
    ));
    let _ = stderr.write_line(&format!("  Gap:          {:.1}%", t.gap_percent));
    let _ = stderr.write_line(&format!(
        "  Cuts:         {} active / {} generated",
        t.total_cuts_active, t.total_cuts_generated
    ));
    let _ = stderr.write_line(&format!("  LP solves:    {}", t.total_lp_solves));
}

/// Print the simulation completion summary to `stderr`.
///
/// Rendered bold header with scenario counts. Write errors are silently ignored.
pub fn print_simulation_summary(stderr: &Term, sim: &SimulationSummary) {
    let duration = format_duration(sim.total_time_ms);
    let _ = stderr.write_line(&format!(
        "{} ({} scenarios)",
        console::style(format!("Simulation complete in {duration}")).bold(),
        sim.n_scenarios
    ));
    let _ = stderr.write_line(&format!(
        "  Completed: {}  Failed: {}",
        sim.completed, sim.failed
    ));
}

/// Print the output directory path and write duration to `stderr`.
///
/// Rendered with bold label and dim path. Write errors are silently ignored.
pub fn print_output_path(stderr: &Term, output_dir: &std::path::Path, write_secs: f64) {
    let _ = stderr.write_line(&format!(
        "{} {}/ {}",
        console::style("Output written to").bold(),
        console::style(output_dir.display()).dim(),
        console::style(format!("({write_secs:.1}s)")).dim()
    ));
}

/// Write the complete post-run summary block to `stderr`.
///
/// Convenience wrapper that calls [`print_training_summary`],
/// [`print_simulation_summary`] (if present), and [`print_output_path`]
/// in sequence.
///
/// Used only in tests; the production code calls the individual print functions.
#[cfg(test)]
pub fn print_summary(stderr: &Term, summary: &RunSummary) {
    print_training_summary(stderr, &summary.training);
    if let Some(sim) = &summary.simulation {
        let _ = stderr.write_line("");
        print_simulation_summary(stderr, sim);
    }
    let _ = stderr.write_line("");
    print_output_path(stderr, &summary.output_dir, 0.0);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use console::Term;

    use super::{
        format_duration, format_summary_string, print_summary, RunSummary, SimulationSummary,
        TrainingSummary,
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
            total_time_ms: 10_000,
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
            s.contains("1.00500e2"),
            "summary must contain '1.00500e2' (scientific notation) for lower_bound = 100.5, got: {s}"
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
    fn test_format_summary_scientific_notation() {
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
            s.contains("4.52304e4"),
            "summary must contain '4.52304e4' (scientific notation) for lower_bound = 45230.41, got: {s}"
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
            total_time_ms: 5_000,
        };
        let summary = make_run_summary(Some(sim));
        print_summary(&Term::buffered_stderr(), &summary);
    }

    // ── StochasticSummary tests ────────────────────────────────────────────

    use super::{
        format_stochastic_summary_string, print_stochastic_summary, ArOrderSummary,
        StochasticSource, StochasticSummary,
    };

    fn make_stochastic_summary() -> StochasticSummary {
        StochasticSummary {
            inflow_source: StochasticSource::Estimated,
            n_hydros: 5,
            n_seasons: 12,
            ar_summary: Some(ArOrderSummary {
                method: "AIC".into(),
                order_counts: vec![0, 3, 2],
                min_order: 1,
                max_order: 2,
                n_hydros: 5,
            }),
            correlation_source: StochasticSource::Estimated,
            correlation_dim: Some("5x5".into()),
            opening_tree_source: StochasticSource::Loaded,
            openings_per_stage: vec![20; 60],
            n_stages: 60,
            n_load_buses: 3,
            seed: 42,
        }
    }

    #[test]
    fn test_ar_order_display_compact_10_or_fewer_hydros() {
        let ar = ArOrderSummary {
            method: "AIC".into(),
            order_counts: vec![0, 3, 2],
            min_order: 1,
            max_order: 2,
            n_hydros: 5,
        };
        let s = ar.display_string();
        assert!(
            s.contains("3x order-1"),
            "compact format must contain '3x order-1', got: {s}"
        );
        assert!(
            s.contains("2x order-2"),
            "compact format must contain '2x order-2', got: {s}"
        );
    }

    #[test]
    fn test_ar_order_display_range_11_to_30_hydros() {
        let ar = ArOrderSummary {
            method: "AIC".into(),
            order_counts: vec![0, 10, 8, 5, 2],
            min_order: 1,
            max_order: 4,
            n_hydros: 25,
        };
        let s = ar.display_string();
        assert!(
            s.contains("orders 1-"),
            "range format must contain 'orders 1-', got: {s}"
        );
        assert!(
            s.contains("25 hydros"),
            "range format must contain '25 hydros', got: {s}"
        );
    }

    #[test]
    fn test_ar_order_display_histogram_31_plus_hydros() {
        let ar = ArOrderSummary {
            method: "AIC".into(),
            order_counts: vec![0, 12, 8, 5, 6],
            min_order: 1,
            max_order: 4,
            n_hydros: 31,
        };
        let s = ar.display_string();
        assert!(
            s.contains("order 1:"),
            "histogram format must contain 'order 1:', got: {s}"
        );
        assert!(
            s.contains("31 hydros"),
            "histogram format must contain '31 hydros', got: {s}"
        );
    }

    #[test]
    fn test_print_stochastic_summary_does_not_panic() {
        let summary = make_stochastic_summary();
        print_stochastic_summary(&Term::buffered_stderr(), &summary);
    }

    #[test]
    fn test_format_stochastic_summary_estimated_sources() {
        let summary = make_stochastic_summary();
        let s = format_stochastic_summary_string(&summary);
        assert!(
            s.contains("estimated"),
            "summary must contain 'estimated' for estimated inflow source, got: {s}"
        );
        assert!(
            s.contains("5x5"),
            "summary must contain correlation dim '5x5', got: {s}"
        );
    }

    #[test]
    fn test_format_stochastic_summary_loaded_source() {
        let summary = StochasticSummary {
            inflow_source: StochasticSource::Loaded,
            ..make_stochastic_summary()
        };
        let s = format_stochastic_summary_string(&summary);
        assert!(
            s.contains("loaded"),
            "summary must contain 'loaded' for loaded inflow source, got: {s}"
        );
    }

    #[test]
    fn test_format_stochastic_summary_none_source() {
        let summary = StochasticSummary {
            inflow_source: StochasticSource::None,
            n_hydros: 0,
            ar_summary: None,
            ..make_stochastic_summary()
        };
        let s = format_stochastic_summary_string(&summary);
        assert!(
            s.contains("none"),
            "summary must contain 'none' for None inflow source, got: {s}"
        );
    }

    // ── StochasticSource variant display tests ────────────────────────────────

    #[test]
    fn test_stochastic_source_estimated_renders_estimated() {
        let summary = StochasticSummary {
            inflow_source: StochasticSource::Estimated,
            ..make_stochastic_summary()
        };
        let s = format_stochastic_summary_string(&summary);
        assert!(
            s.contains("estimated"),
            "Estimated source must render as 'estimated' substring, got: {s}"
        );
    }

    #[test]
    fn test_stochastic_source_loaded_renders_loaded() {
        let summary = StochasticSummary {
            inflow_source: StochasticSource::Loaded,
            ..make_stochastic_summary()
        };
        let s = format_stochastic_summary_string(&summary);
        assert!(
            s.contains("loaded"),
            "Loaded source must render as 'loaded' substring, got: {s}"
        );
    }

    #[test]
    fn test_stochastic_source_none_renders_none() {
        let summary = StochasticSummary {
            inflow_source: StochasticSource::None,
            n_hydros: 0,
            ar_summary: None,
            ..make_stochastic_summary()
        };
        let s = format_stochastic_summary_string(&summary);
        assert!(
            s.contains("none"),
            "None source must render as 'none' substring, got: {s}"
        );
    }

    // ── ArOrderSummary::display_string edge cases ─────────────────────────────

    #[test]
    fn test_ar_order_display_all_same_order() {
        let ar = ArOrderSummary {
            method: "AIC".into(),
            order_counts: vec![0, 5],
            min_order: 1,
            max_order: 1,
            n_hydros: 5,
        };
        let s = ar.display_string();
        assert!(
            s.contains("5x order-1"),
            "all-same-order compact must show '5x order-1', got: {s}"
        );
        // Must not contain a comma (only one entry).
        assert!(
            !s.contains(','),
            "single-order compact format must not contain a comma, got: {s}"
        );
    }

    #[test]
    fn test_ar_order_display_single_hydro() {
        let ar = ArOrderSummary {
            method: "fixed".into(),
            order_counts: vec![0, 0, 0, 1],
            min_order: 3,
            max_order: 3,
            n_hydros: 1,
        };
        let s = ar.display_string();
        assert!(
            s.contains("1x order-3"),
            "single hydro at order 3 must show '1x order-3', got: {s}"
        );
    }

    #[test]
    fn test_ar_order_display_orders_with_gaps() {
        // Order 1: 2 hydros, order 2: 0 hydros, order 3: 3 hydros.
        let ar = ArOrderSummary {
            method: "AIC".into(),
            order_counts: vec![0, 2, 0, 3],
            min_order: 1,
            max_order: 3,
            n_hydros: 5,
        };
        let s = ar.display_string();
        assert!(
            s.contains("2x order-1"),
            "gap case must show '2x order-1', got: {s}"
        );
        assert!(
            s.contains("3x order-3"),
            "gap case must show '3x order-3', got: {s}"
        );
        assert!(
            !s.contains("order-2"),
            "gap case must NOT show 'order-2' (count is 0), got: {s}"
        );
    }

    #[test]
    fn test_ar_order_display_boundary_exactly_10_compact() {
        // Exactly 10 hydros: must use compact format (contains "x order-").
        let ar = ArOrderSummary {
            method: "AIC".into(),
            order_counts: vec![0, 6, 4],
            min_order: 1,
            max_order: 2,
            n_hydros: 10,
        };
        let s = ar.display_string();
        assert!(
            s.contains("x order-"),
            "exactly 10 hydros must use compact format (contains 'x order-'), got: {s}"
        );
        assert!(
            s.contains("6x order-1"),
            "compact must show '6x order-1', got: {s}"
        );
        assert!(
            s.contains("4x order-2"),
            "compact must show '4x order-2', got: {s}"
        );
    }

    #[test]
    fn test_ar_order_display_boundary_exactly_11_range() {
        // Exactly 11 hydros: must use range format (contains "orders" and "-").
        let ar = ArOrderSummary {
            method: "AIC".into(),
            order_counts: vec![0, 7, 4],
            min_order: 1,
            max_order: 2,
            n_hydros: 11,
        };
        let s = ar.display_string();
        assert!(
            s.contains("orders"),
            "exactly 11 hydros must use range format (contains 'orders'), got: {s}"
        );
        assert!(
            s.contains('-'),
            "range format must contain '-' between min and max orders, got: {s}"
        );
        assert!(
            s.contains("11 hydros"),
            "range format must contain '11 hydros', got: {s}"
        );
    }

    #[test]
    fn test_ar_order_display_boundary_exactly_30_range() {
        // Exactly 30 hydros: still range format (upper boundary of range tier).
        let ar = ArOrderSummary {
            method: "AIC".into(),
            order_counts: vec![0, 15, 10, 5],
            min_order: 1,
            max_order: 3,
            n_hydros: 30,
        };
        let s = ar.display_string();
        assert!(
            s.contains("orders"),
            "exactly 30 hydros must use range format (contains 'orders'), got: {s}"
        );
        assert!(
            s.contains("30 hydros"),
            "range format must contain '30 hydros', got: {s}"
        );
        // Must NOT use histogram format.
        assert!(
            !s.contains("order 1:"),
            "exactly 30 hydros must NOT use histogram format, got: {s}"
        );
    }

    #[test]
    fn test_ar_order_display_boundary_exactly_31_histogram() {
        // Exactly 31 hydros: must use histogram format (contains "order N:").
        let ar = ArOrderSummary {
            method: "AIC".into(),
            order_counts: vec![0, 16, 10, 5],
            min_order: 1,
            max_order: 3,
            n_hydros: 31,
        };
        let s = ar.display_string();
        assert!(
            s.contains("order 1:"),
            "exactly 31 hydros must use histogram format (contains 'order 1:'), got: {s}"
        );
        assert!(
            s.contains("31 hydros"),
            "histogram format must contain '31 hydros', got: {s}"
        );
        // Must NOT use range-style "orders N-M" phrase.
        assert!(
            !s.contains("orders "),
            "exactly 31 hydros must NOT use range format, got: {s}"
        );
    }

    // ── format_stochastic_summary_string full-output test ─────────────────────

    #[test]
    fn test_format_stochastic_summary_full_output() {
        let summary = make_stochastic_summary();
        let s = format_stochastic_summary_string(&summary);

        // Header line
        assert!(
            s.contains("Stochastic preprocessing"),
            "output must contain header 'Stochastic preprocessing', got: {s}"
        );
        // Seed line
        assert!(
            s.contains("Seed:"),
            "output must contain 'Seed:' line, got: {s}"
        );
        assert!(
            s.contains("42"),
            "output must contain seed value '42', got: {s}"
        );
        // Inflow stats line
        assert!(
            s.contains("Inflow stats:"),
            "output must contain 'Inflow stats:' line, got: {s}"
        );
        assert!(
            s.contains("5 hydros"),
            "output must contain '5 hydros', got: {s}"
        );
        assert!(
            s.contains("12 seasons"),
            "output must contain '12 seasons', got: {s}"
        );
        // AR orders line
        assert!(
            s.contains("AR orders:"),
            "output must contain 'AR orders:' line, got: {s}"
        );
        // Correlation line
        assert!(
            s.contains("Correlation:"),
            "output must contain 'Correlation:' line, got: {s}"
        );
        assert!(
            s.contains("5x5"),
            "output must contain correlation dim '5x5', got: {s}"
        );
        // Opening tree line
        assert!(
            s.contains("Opening tree:"),
            "output must contain 'Opening tree:' line, got: {s}"
        );
        assert!(
            s.contains("20 openings/stage"),
            "output must contain '20 openings/stage', got: {s}"
        );
        assert!(
            s.contains("60 stages"),
            "output must contain '60 stages', got: {s}"
        );
        // Load noise line
        assert!(
            s.contains("Load noise:"),
            "output must contain 'Load noise:' line, got: {s}"
        );
        assert!(
            s.contains("3 stochastic buses"),
            "output must contain '3 stochastic buses', got: {s}"
        );
    }
}
