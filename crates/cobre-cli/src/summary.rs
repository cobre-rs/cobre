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

// Re-export data types from cobre-sddp so callers can import them from this module.
// ArOrderSummary is not referenced directly in non-test code here (only via StochasticSummary
// fields), so suppress the false-positive unused-import warning on the pub use.
#[allow(unused_imports)]
pub use cobre_sddp::{ArOrderSummary, HydroModelSummary, StochasticSource, StochasticSummary};
use console::Term;

fn source_label(source: &StochasticSource) -> &'static str {
    match source {
        StochasticSource::Estimated => "estimated",
        StochasticSource::Loaded => "loaded",
        StochasticSource::None => "none",
    }
}

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

/// Print the hydro model preprocessing summary to `stderr`.
///
/// Renders a bold header followed by two indented lines:
/// - `Production:` — counts of constant and FPHA hydros with plane totals.
/// - `Evaporation:` — counts of linearized and un-modelled hydro plants.
///
/// The singular/plural form `"hydro"` vs `"hydros"` is applied to the
/// evaporation counts. Write errors are silently ignored (fire-and-forget).
pub fn print_hydro_model_summary(stderr: &Term, summary: &HydroModelSummary) {
    let _ = stderr.write_line(&format!("{}", console::style("Hydro models").bold()));
    let _ = stderr.write_line(&format!(
        "  Production:    {}",
        format_production_line(summary)
    ));
    let _ = stderr.write_line(&format!(
        "  Evaporation:   {}",
        format_evaporation_line(summary)
    ));
}

/// Classify how FPHA planes were obtained for display purposes.
enum FphaSourceLabel {
    /// All FPHA hydros loaded from `fpha_hyperplanes.parquet`.
    AllPrecomputed,
    /// All FPHA hydros computed from reservoir geometry.
    AllComputed,
    /// A mix: `n_precomputed` from Parquet and `n_computed` from geometry.
    Mixed {
        n_precomputed: usize,
        n_computed: usize,
    },
}

/// Determine the [`FphaSourceLabel`] for the FPHA hydros in a summary.
fn fpha_source_label(summary: &HydroModelSummary) -> FphaSourceLabel {
    use cobre_sddp::ProductionModelSource;

    let n_precomputed = summary
        .fpha_details
        .iter()
        .filter(|d| d.source == ProductionModelSource::PrecomputedHyperplanes)
        .count();
    let n_computed = summary
        .fpha_details
        .iter()
        .filter(|d| d.source == ProductionModelSource::ComputedFromGeometry)
        .count();

    match (n_precomputed, n_computed) {
        (_, 0) => FphaSourceLabel::AllPrecomputed,
        (0, _) => FphaSourceLabel::AllComputed,
        _ => FphaSourceLabel::Mixed {
            n_precomputed,
            n_computed,
        },
    }
}

/// Format the production detail line for a [`HydroModelSummary`].
fn format_production_line(summary: &HydroModelSummary) -> String {
    match (summary.n_constant, summary.n_fpha) {
        (0, 0) => "0 hydros".to_string(),
        (n_const, 0) => format!("{n_const} constant"),
        (0, n_fpha) => {
            let source_detail = match fpha_source_label(summary) {
                FphaSourceLabel::AllPrecomputed => {
                    "loaded from fpha_hyperplanes.parquet".to_string()
                }
                FphaSourceLabel::AllComputed => "computed from geometry".to_string(),
                FphaSourceLabel::Mixed {
                    n_precomputed,
                    n_computed,
                } => format!("{n_precomputed} precomputed, {n_computed} computed from geometry"),
            };
            format!(
                "{n_fpha} FPHA ({} planes, {source_detail})",
                summary.total_planes
            )
        }
        (n_const, n_fpha) => {
            let source_detail = match fpha_source_label(summary) {
                FphaSourceLabel::AllPrecomputed => "loaded".to_string(),
                FphaSourceLabel::AllComputed => "computed from geometry".to_string(),
                FphaSourceLabel::Mixed {
                    n_precomputed,
                    n_computed,
                } => format!("{n_precomputed} precomputed, {n_computed} computed from geometry"),
            };
            format!(
                "{n_const} constant, {n_fpha} FPHA ({} planes, {source_detail})",
                summary.total_planes
            )
        }
    }
}

/// Pluralize "hydro" or "hydros" based on count.
fn hydro_plural(count: usize) -> &'static str {
    if count == 1 { "hydro" } else { "hydros" }
}

/// Format the evaporation detail line for a [`HydroModelSummary`].
///
/// When there are no evaporating hydros the line has no reference source detail.
/// When all evaporating hydros share a single source the label is unqualified;
/// when they are split between sources the counts are shown explicitly.
fn format_evaporation_line(summary: &HydroModelSummary) -> String {
    if summary.n_evaporation == 0 {
        return format!(
            "0 hydros linearized, {} {} without",
            summary.n_no_evaporation,
            hydro_plural(summary.n_no_evaporation),
        );
    }
    let ref_detail = match (summary.n_user_supplied_ref, summary.n_default_midpoint_ref) {
        (0, _) => "midpoint v_ref".to_string(),
        (_, 0) => "user v_ref".to_string(),
        (u, m) => format!("{u} user v_ref, {m} midpoint v_ref"),
    };
    format!(
        "{} {} linearized (from geometry, {ref_detail}), {} {} without",
        summary.n_evaporation,
        hydro_plural(summary.n_evaporation),
        summary.n_no_evaporation,
        hydro_plural(summary.n_no_evaporation),
    )
}

/// Render the hydro model preprocessing summary as a plain-text `String`.
///
/// The returned string contains no ANSI escape sequences. Color and styling
/// are applied by [`print_hydro_model_summary`] when writing to the terminal.
/// This function exists to allow unit tests to assert on summary content
/// without requiring a real terminal.
#[cfg(test)]
pub fn format_hydro_model_summary_string(summary: &HydroModelSummary) -> String {
    let mut lines: Vec<String> = Vec::new();
    lines.push("Hydro models".to_string());
    lines.push(format!(
        "  Production:    {}",
        format_production_line(summary)
    ));
    lines.push(format!(
        "  Evaporation:   {}",
        format_evaporation_line(summary)
    ));
    lines.join("\n")
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
        ArOrderSummary, StochasticSource, StochasticSummary, format_stochastic_summary_string,
        print_stochastic_summary,
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

    // Compact format used when n_hydros <= 10.
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

    // Range format used when 11 <= n_hydros <= 30.
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

    // Histogram format used when n_hydros >= 31.
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

    // ── HydroModelSummary tests ────────────────────────────────────────────

    use super::{HydroModelSummary, format_hydro_model_summary_string, print_hydro_model_summary};
    use cobre_core::EntityId;
    use cobre_sddp::FphaHydroDetail;

    fn make_hydro_model_summary_mixed() -> HydroModelSummary {
        HydroModelSummary {
            n_constant: 2,
            n_fpha: 2,
            total_planes: 10,
            fpha_details: vec![
                FphaHydroDetail {
                    hydro_id: EntityId(3),
                    name: "Hydro3".to_string(),
                    source: cobre_sddp::ProductionModelSource::PrecomputedHyperplanes,
                    n_planes: 5,
                },
                FphaHydroDetail {
                    hydro_id: EntityId(4),
                    name: "Hydro4".to_string(),
                    source: cobre_sddp::ProductionModelSource::PrecomputedHyperplanes,
                    n_planes: 5,
                },
            ],
            n_evaporation: 3,
            n_no_evaporation: 1,
            n_user_supplied_ref: 0,
            n_default_midpoint_ref: 3,
            kappa_warnings: Vec::new(),
        }
    }

    fn make_hydro_model_summary_all_constant() -> HydroModelSummary {
        HydroModelSummary {
            n_constant: 4,
            n_fpha: 0,
            total_planes: 0,
            fpha_details: vec![],
            n_evaporation: 0,
            n_no_evaporation: 4,
            n_user_supplied_ref: 0,
            n_default_midpoint_ref: 0,
            kappa_warnings: Vec::new(),
        }
    }

    fn make_hydro_model_summary_all_fpha() -> HydroModelSummary {
        HydroModelSummary {
            n_constant: 0,
            n_fpha: 165,
            total_planes: 825,
            fpha_details: vec![],
            n_evaporation: 162,
            n_no_evaporation: 3,
            n_user_supplied_ref: 0,
            n_default_midpoint_ref: 162,
            kappa_warnings: Vec::new(),
        }
    }

    /// Acceptance criterion: 2 FPHA hydros → output contains "2 FPHA" and "planes" and "loaded".
    #[test]
    fn format_hydro_model_summary_with_fpha_contains_key_terms() {
        let summary = make_hydro_model_summary_mixed();
        let s = format_hydro_model_summary_string(&summary);

        assert!(
            s.contains("2 FPHA"),
            "mixed summary must contain '2 FPHA', got: {s}"
        );
        assert!(
            s.contains("planes"),
            "mixed summary must contain 'planes', got: {s}"
        );
        assert!(
            s.contains("loaded"),
            "mixed summary must contain 'loaded', got: {s}"
        );
    }

    /// Acceptance criterion: 0 FPHA hydros → output contains "constant" and NOT "FPHA".
    #[test]
    fn format_hydro_model_summary_without_fpha_contains_constant_not_fpha() {
        let summary = make_hydro_model_summary_all_constant();
        let s = format_hydro_model_summary_string(&summary);

        assert!(
            s.contains("constant"),
            "all-constant summary must contain 'constant', got: {s}"
        );
        assert!(
            !s.contains("FPHA"),
            "all-constant summary must NOT contain 'FPHA', got: {s}"
        );
    }

    /// Header line is always present.
    #[test]
    fn format_hydro_model_summary_contains_header() {
        let summary = make_hydro_model_summary_mixed();
        let s = format_hydro_model_summary_string(&summary);

        assert!(
            s.contains("Hydro models"),
            "summary must contain 'Hydro models' header, got: {s}"
        );
    }

    /// Mixed summary: production line shows plane count and "loaded".
    #[test]
    fn format_hydro_model_summary_mixed_production_line() {
        let summary = make_hydro_model_summary_mixed();
        let s = format_hydro_model_summary_string(&summary);

        assert!(
            s.contains("10"),
            "mixed summary must contain plane count '10', got: {s}"
        );
        assert!(
            s.contains("2 constant"),
            "mixed summary must contain '2 constant', got: {s}"
        );
    }

    /// All-FPHA large system: production line includes filename.
    #[test]
    fn format_hydro_model_summary_all_fpha_shows_filename() {
        let summary = make_hydro_model_summary_all_fpha();
        let s = format_hydro_model_summary_string(&summary);

        assert!(
            s.contains("165 FPHA"),
            "all-fpha summary must contain '165 FPHA', got: {s}"
        );
        assert!(
            s.contains("825"),
            "all-fpha summary must contain '825' (plane count), got: {s}"
        );
        assert!(
            s.contains("fpha_hyperplanes.parquet"),
            "all-fpha summary must contain the filename, got: {s}"
        );
    }

    /// Singular/plural: "1 hydro" vs "3 hydros" in evaporation line.
    #[test]
    fn format_hydro_model_summary_singular_evaporation() {
        let summary = HydroModelSummary {
            n_constant: 3,
            n_fpha: 0,
            total_planes: 0,
            fpha_details: vec![],
            n_evaporation: 1,
            n_no_evaporation: 2,
            n_user_supplied_ref: 0,
            n_default_midpoint_ref: 1,
            kappa_warnings: Vec::new(),
        };
        let s = format_hydro_model_summary_string(&summary);

        assert!(
            s.contains("1 hydro linearized"),
            "singular evaporation must use 'hydro' not 'hydros', got: {s}"
        );
    }

    #[test]
    fn format_hydro_model_summary_plural_evaporation() {
        let summary = make_hydro_model_summary_mixed();
        let s = format_hydro_model_summary_string(&summary);

        assert!(
            s.contains("3 hydros linearized"),
            "plural evaporation must use 'hydros', got: {s}"
        );
    }

    /// Acceptance criterion: `print_hydro_model_summary` does not panic with buffered stderr.
    #[test]
    fn print_hydro_model_summary_does_not_panic() {
        let summary = make_hydro_model_summary_mixed();
        print_hydro_model_summary(&Term::buffered_stderr(), &summary);
    }

    #[test]
    fn print_hydro_model_summary_all_constant_does_not_panic() {
        let summary = make_hydro_model_summary_all_constant();
        print_hydro_model_summary(&Term::buffered_stderr(), &summary);
    }

    #[test]
    fn print_hydro_model_summary_all_fpha_does_not_panic() {
        let summary = make_hydro_model_summary_all_fpha();
        print_hydro_model_summary(&Term::buffered_stderr(), &summary);
    }

    // ── format_evaporation_line reference-source variant tests ───────────────

    /// AC: all-midpoint — line contains "midpoint `v_ref`" and does NOT contain "user `v_ref`".
    #[test]
    fn test_evaporation_line_all_midpoint() {
        let summary = HydroModelSummary {
            n_constant: 2,
            n_fpha: 0,
            total_planes: 0,
            fpha_details: vec![],
            n_evaporation: 2,
            n_no_evaporation: 0,
            n_user_supplied_ref: 0,
            n_default_midpoint_ref: 2,
            kappa_warnings: Vec::new(),
        };
        let s = format_hydro_model_summary_string(&summary);
        assert!(
            s.contains("midpoint v_ref"),
            "all-midpoint must contain 'midpoint v_ref', got: {s}"
        );
        assert!(
            !s.contains("user v_ref"),
            "all-midpoint must NOT contain 'user v_ref', got: {s}"
        );
    }

    /// AC: all-user-supplied — line contains "user `v_ref`" and does NOT contain "midpoint `v_ref`".
    #[test]
    fn test_evaporation_line_all_user_supplied() {
        let summary = HydroModelSummary {
            n_constant: 3,
            n_fpha: 0,
            total_planes: 0,
            fpha_details: vec![],
            n_evaporation: 3,
            n_no_evaporation: 1,
            n_user_supplied_ref: 3,
            n_default_midpoint_ref: 0,
            kappa_warnings: Vec::new(),
        };
        let s = format_hydro_model_summary_string(&summary);
        assert!(
            s.contains("user v_ref"),
            "all-user-supplied must contain 'user v_ref', got: {s}"
        );
        assert!(
            !s.contains("midpoint v_ref"),
            "all-user-supplied must NOT contain 'midpoint v_ref', got: {s}"
        );
        assert!(
            s.contains("3 hydros linearized"),
            "all-user-supplied must contain '3 hydros linearized', got: {s}"
        );
        assert!(
            s.contains("1 hydro without"),
            "all-user-supplied must contain '1 hydro without', got: {s}"
        );
    }

    /// AC: mixed — line contains "2 user `v_ref`" and "1 midpoint `v_ref`".
    #[test]
    fn test_evaporation_line_mixed() {
        let summary = HydroModelSummary {
            n_constant: 3,
            n_fpha: 0,
            total_planes: 0,
            fpha_details: vec![],
            n_evaporation: 3,
            n_no_evaporation: 1,
            n_user_supplied_ref: 2,
            n_default_midpoint_ref: 1,
            kappa_warnings: Vec::new(),
        };
        let s = format_hydro_model_summary_string(&summary);
        assert!(
            s.contains("2 user v_ref"),
            "mixed must contain '2 user v_ref', got: {s}"
        );
        assert!(
            s.contains("1 midpoint v_ref"),
            "mixed must contain '1 midpoint v_ref', got: {s}"
        );
        assert!(
            s.contains("3 hydros linearized"),
            "mixed must contain '3 hydros linearized', got: {s}"
        );
    }

    /// AC: no evaporation — line does NOT contain "`v_ref`".
    #[test]
    fn test_evaporation_line_no_evaporation() {
        let summary = make_hydro_model_summary_all_constant();
        let s = format_hydro_model_summary_string(&summary);
        assert!(
            !s.contains("v_ref"),
            "zero-evaporation must NOT contain 'v_ref', got: {s}"
        );
        assert!(
            s.contains("0 hydros linearized"),
            "zero-evaporation must contain '0 hydros linearized', got: {s}"
        );
    }
}
