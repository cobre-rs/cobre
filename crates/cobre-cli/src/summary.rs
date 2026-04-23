//! Post-run summary block for the `cobre run` command.
//!
//! Provides separate printing functions for each phase of the run:
//! - [`print_execution_topology`] — execution topology (backend, threads, layout)
//! - [`print_stochastic_summary`] — stochastic preprocessing statistics
//! - [`print_training_summary`] — training convergence metrics
//! - [`print_simulation_summary`] — simulation completion stats
//! - [`print_output_path`] — output directory location
//!
//! Each function prints its section independently so the caller can display
//! results at the right point in the execution flow.

#[allow(unused_imports)]
pub use cobre_sddp::{HydroModelSummary, ModelProvenanceReport, ProvenanceSource};
use console::Term;

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

/// Format a sorted list of rank indices into a compact range string.
///
/// Contiguous sequences use en-dash notation (`0–3`). Non-contiguous ranks are
/// comma-separated. Mixed sequences interleave both styles: `0–2, 7, 9–11`.
/// An empty slice returns an empty string.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(format_rank_list(&[0, 1, 2, 3]), "0–3");
/// assert_eq!(format_rank_list(&[0, 2, 5]),    "0, 2, 5");
/// assert_eq!(format_rank_list(&[0, 1, 2, 7, 9, 10, 11]), "0–2, 7, 9–11");
/// ```
fn format_rank_list(ranks: &[usize]) -> String {
    if ranks.is_empty() {
        return String::new();
    }
    let mut segments: Vec<String> = Vec::new();
    let mut start = ranks[0];
    let mut end = ranks[0];

    for &r in &ranks[1..] {
        if r == end + 1 {
            end = r;
        } else {
            if end > start {
                segments.push(format!("{start}\u{2013}{end}"));
            } else {
                segments.push(format!("{start}"));
            }
            start = r;
            end = r;
        }
    }
    // Flush final segment.
    if end > start {
        segments.push(format!("{start}\u{2013}{end}"));
    } else {
        segments.push(format!("{start}"));
    }
    segments.join(", ")
}

/// Print the execution topology summary to `stderr`.
///
/// Renders a bold header followed by indented detail lines showing the
/// communication backend, threading configuration, and process layout.
/// Called once after the banner, before any phase output.
///
/// For a **local** backend the output is:
///
/// ```text
/// Execution
///   Backend:   local
///   Host:      hostname
///   Threads:   5 rayon threads
/// ```
///
/// For **MPI** on a single node:
///
/// ```text
/// Execution
///   Backend:   MPI (Open MPI v4.1.6, MPI 4.0)
///   Threads:   Funneled, 5 rayon threads per rank
///   Layout:    4 ranks on hostname
/// ```
///
/// For **MPI** across multiple nodes, a per-host breakdown is added, and an
/// optional SLURM line is appended when scheduler metadata is present.
pub fn print_execution_topology(
    stderr: &Term,
    topology: &cobre_comm::ExecutionTopology,
    n_threads: usize,
    solver_name: &str,
    solver_version: Option<&str>,
) {
    use cobre_comm::BackendKind;

    let thread_word = if n_threads == 1 {
        "rayon thread"
    } else {
        "rayon threads"
    };

    let _ = stderr.write_line(&format!("{}", console::style("Execution").bold()));

    // Solver line — always shown regardless of backend.
    let solver_line = match solver_version {
        Some(v) => format!("{solver_name} {v}"),
        None => solver_name.to_string(),
    };
    let _ = stderr.write_line(&format!("  Solver:    {solver_line}"));

    match topology.backend {
        BackendKind::Local => {
            let _ = stderr.write_line("  Backend:   local");
            let _ = stderr.write_line(&format!("  Host:      {}", topology.leader_hostname()));
            let _ = stderr.write_line(&format!("  Threads:   {n_threads} {thread_word}"));
        }
        BackendKind::Mpi => {
            // Backend line with library and standard version.
            let backend_detail = if let Some(ref mpi) = topology.mpi {
                format!("MPI ({}, {})", mpi.library_version, mpi.standard_version)
            } else {
                "MPI".to_string()
            };
            let _ = stderr.write_line(&format!("  Backend:   {backend_detail}"));

            // Thread level + rayon thread count.
            let thread_line = if let Some(ref mpi) = topology.mpi {
                format!("{}, {n_threads} {thread_word} per rank", mpi.thread_level)
            } else {
                format!("{n_threads} {thread_word} per rank")
            };
            let _ = stderr.write_line(&format!("  Threads:   {thread_line}"));

            // Layout: single-node or multi-node.
            let world_size = topology.world_size;
            let rank_word = if world_size == 1 { "rank" } else { "ranks" };
            let num_hosts = topology.num_hosts();
            if num_hosts <= 1 {
                let _ = stderr.write_line(&format!(
                    "  Layout:    {world_size} {rank_word} on {}",
                    topology.leader_hostname()
                ));
            } else {
                let node_word = if num_hosts == 1 { "node" } else { "nodes" };
                let _ = stderr.write_line(&format!(
                    "  Layout:    {world_size} {rank_word} across {num_hosts} {node_word}"
                ));
                for host in &topology.hosts {
                    let count = host.ranks.len();
                    let rank_count_word = if count == 1 { "rank" } else { "ranks" };
                    let range = format_rank_list(&host.ranks);
                    let _ = stderr.write_line(&format!(
                        "    {}: ranks {range}  ({count} {rank_count_word})",
                        host.hostname
                    ));
                }
            }

            // Optional SLURM line.
            if let Some(ref slurm) = topology.slurm {
                let mut slurm_parts: Vec<String> = Vec::new();
                slurm_parts.push(format!("job {}", slurm.job_id));
                if let Some(ref node_list) = slurm.node_list {
                    slurm_parts.push(format!("nodes {node_list}"));
                }
                if let Some(cpus) = slurm.cpus_per_task {
                    slurm_parts.push(format!("{cpus} CPUs/task"));
                }
                let _ = stderr.write_line(&format!("  SLURM:     {}", slurm_parts.join(", ")));
            }
        }
        // Auto (unresolved) backend: print minimal info.
        BackendKind::Auto => {
            let _ = stderr.write_line(&format!("  Backend:   {:?}", topology.backend));
            let _ = stderr.write_line(&format!("  Threads:   {n_threads} {thread_word}"));
        }
    }
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

/// Format the AR detail parenthetical for the provenance summary line.
///
/// Returns `" (method, max order N)"` when AR method is known, or an empty
/// string when AR is `NotApplicable` (no parenthetical shown).
fn provenance_ar_detail(report: &ModelProvenanceReport) -> String {
    match (&report.ar_method, report.ar_max_order) {
        (Some(method), Some(max_order)) => format!(" ({method}, max order {max_order})"),
        _ => String::new(),
    }
}

/// Print the model provenance summary to `stderr`.
///
/// Renders a bold header followed by indented lines covering the estimation
/// path, seasonal stats source, AR coefficients source, correlation source,
/// and opening tree source.
///
/// The AR line includes a parenthetical detail (`(method, max order N)`) when
/// `ar_method` is `Some`; this detail is omitted when AR is `NotApplicable`.
/// Write errors are silently ignored (fire-and-forget).
pub fn print_provenance_summary(stderr: &Term, report: &ModelProvenanceReport) {
    let _ = stderr.write_line(&format!("{}", console::style("Model provenance").bold()));
    let _ = stderr.write_line(&format!("  Estimation path: {}", report.estimation_path));
    let _ = stderr.write_line(&format!(
        "  Seasonal stats:  {}",
        report.seasonal_stats_source
    ));
    let ar_detail = provenance_ar_detail(report);
    let _ = stderr.write_line(&format!(
        "  AR coefficients: {}{}",
        report.ar_coefficients_source, ar_detail
    ));
    let _ = stderr.write_line(&format!("  Correlation:     {}", report.correlation_source));
    let _ = stderr.write_line(&format!(
        "  Opening tree:    {}",
        report.opening_tree_source
    ));
}

/// Render the model provenance summary as a plain-text `String`.
///
/// The returned string contains no ANSI escape sequences. Color and styling
/// are applied by [`print_provenance_summary`] when writing to the terminal.
/// This function exists to allow unit tests to assert on summary content
/// without requiring a real terminal.
#[cfg(test)]
pub fn format_provenance_summary_string(report: &ModelProvenanceReport) -> String {
    let mut lines: Vec<String> = Vec::new();
    lines.push("Model provenance".to_string());
    lines.push(format!("  Estimation path: {}", report.estimation_path));
    lines.push(format!(
        "  Seasonal stats:  {}",
        report.seasonal_stats_source
    ));
    let ar_detail = provenance_ar_detail(report);
    lines.push(format!(
        "  AR coefficients: {}{}",
        report.ar_coefficients_source, ar_detail
    ));
    lines.push(format!("  Correlation:     {}", report.correlation_source));
    lines.push(format!("  Opening tree:    {}", report.opening_tree_source));
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

    /// Number of policy rows active in the pool at the end of training.
    pub total_rows_active: u64,

    /// Total number of policy rows generated over the entire training run.
    pub total_rows_generated: u64,

    /// Total number of LP solves across all ranks, stages, iterations, and
    /// passes.  Aggregated via `allreduce(Sum)` so that the reported value is
    /// invariant regardless of the parallel configuration.
    pub total_lp_solves: u64,

    /// Total elapsed wall-clock time for the training run (milliseconds).
    pub total_time_ms: u64,

    /// Number of solves that returned optimal on the first attempt.
    ///
    /// `None` when solver stats are unavailable (e.g. `cobre summary`
    /// reads metadata.json which does not persist per-solve stats).
    pub total_first_try: Option<u64>,

    /// Number of solves that required retry escalation.
    pub total_retried: Option<u64>,

    /// Number of solves that exhausted all retry levels.
    pub total_failed: Option<u64>,

    /// Total LP solve wall-clock time in seconds.
    pub total_solve_time_seconds: Option<f64>,

    /// Total warm-start solve calls (basis offers) across all solvers in the training phase.
    pub total_basis_offered: Option<u64>,

    /// Number of warm-start calls in which `isBasisConsistent` returned false.
    pub total_basis_consistency_failures: Option<u64>,

    /// Total simplex iterations across all solves.
    pub total_simplex_iterations: Option<u64>,
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

    /// Global mean cost across all scenarios (aggregated across MPI ranks).
    pub mean_cost: Option<f64>,

    /// Global standard deviation of cost across all scenarios.
    pub std_cost: Option<f64>,

    /// Total LP solves across all scenarios.
    pub total_lp_solves: Option<u64>,

    /// Solves that returned optimal on the first attempt.
    pub total_first_try: Option<u64>,

    /// Solves that required retry escalation before succeeding.
    pub total_retried: Option<u64>,

    /// Solves that exhausted all retry levels.
    pub total_failed_solves: Option<u64>,

    /// Cumulative LP solve wall-clock time in seconds.
    pub total_solve_time_seconds: Option<f64>,

    /// Total warm-start solve calls (basis offers) across all solvers in the simulation phase.
    pub total_basis_offered: Option<u64>,

    /// Number of warm-start calls in which `isBasisConsistent` returned false.
    pub total_basis_consistency_failures: Option<u64>,

    /// Total simplex iterations across all solves.
    pub total_simplex_iterations: Option<u64>,
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
///   Policy rows:  {active} active / {generated} generated
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
        "  Policy rows:  {} active / {} generated",
        t.total_rows_active, t.total_rows_generated
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
/// Rendered bold header with convergence metrics, bounds, policy rows, and LP solves.
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
        "  Policy rows:  {} active / {} generated",
        t.total_rows_active, t.total_rows_generated
    ));
    if let (Some(first_try), Some(retried), Some(failed)) =
        (t.total_first_try, t.total_retried, t.total_failed)
    {
        let _ = stderr.write_line(&format!(
            "  LP solves:    {} ({first_try} first-try, {retried} retried, {failed} failed)",
            t.total_lp_solves
        ));
    } else {
        let _ = stderr.write_line(&format!("  LP solves:    {}", t.total_lp_solves));
    }
    if let Some(solve_time) = t.total_solve_time_seconds {
        #[allow(clippy::cast_precision_loss)]
        let avg_ms = if t.total_lp_solves > 0 {
            solve_time * 1000.0 / t.total_lp_solves as f64
        } else {
            0.0
        };
        let _ = stderr.write_line(&format!(
            "  LP time:      {solve_time:.1}s total, {avg_ms:.1}ms avg"
        ));
    }
    if let (Some(offered), Some(failures)) =
        (t.total_basis_offered, t.total_basis_consistency_failures)
    {
        if offered > 0 {
            #[allow(clippy::cast_precision_loss)]
            let hit_pct = (1.0 - failures as f64 / offered as f64) * 100.0;
            let _ = stderr.write_line(&format!(
                "  Basis reuse:  {hit_pct:.1}% hit ({failures} rejected / {offered} offered)"
            ));
        } else if failures > 0 {
            let _ = stderr.write_line(&format!("  Basis consistency failures: {failures}"));
        }
    }
    if let Some(simplex) = t.total_simplex_iterations {
        let _ = stderr.write_line(&format!("  Simplex iter: {simplex}"));
    }
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
    if let (Some(mean), Some(std)) = (sim.mean_cost, sim.std_cost) {
        #[allow(clippy::cast_precision_loss)]
        let ci95 = if sim.n_scenarios >= 2 {
            1.96 * std / (f64::from(sim.n_scenarios)).sqrt()
        } else {
            0.0
        };
        let _ = stderr.write_line(&format!(
            "  Expected cost: {mean:.5e} +/- {ci95:.5e} (std: {std:.5e})"
        ));
    }
    if let (Some(lp_solves), Some(first_try), Some(retried), Some(failed)) = (
        sim.total_lp_solves,
        sim.total_first_try,
        sim.total_retried,
        sim.total_failed_solves,
    ) {
        let _ = stderr.write_line(&format!(
            "  LP solves:    {lp_solves} ({first_try} first-try, {retried} retried, {failed} failed)"
        ));
    } else if let Some(lp_solves) = sim.total_lp_solves {
        let _ = stderr.write_line(&format!("  LP solves:    {lp_solves}"));
    }
    if let (Some(lp_solves), Some(solve_time)) = (sim.total_lp_solves, sim.total_solve_time_seconds)
    {
        #[allow(clippy::cast_precision_loss)]
        let avg_ms = if lp_solves > 0 {
            solve_time * 1000.0 / lp_solves as f64
        } else {
            0.0
        };
        let _ = stderr.write_line(&format!(
            "  LP time:      {solve_time:.1}s total, {avg_ms:.1}ms avg"
        ));
    }
    if let (Some(offered), Some(failures)) = (
        sim.total_basis_offered,
        sim.total_basis_consistency_failures,
    ) {
        if offered > 0 {
            #[allow(clippy::cast_precision_loss)]
            let hit_pct = (1.0 - failures as f64 / offered as f64) * 100.0;
            let _ = stderr.write_line(&format!(
                "  Basis reuse:  {hit_pct:.1}% hit ({failures} rejected / {offered} offered)"
            ));
        } else if failures > 0 {
            let _ = stderr.write_line(&format!("  Basis consistency failures: {failures}"));
        }
    }
    if let Some(simplex) = sim.total_simplex_iterations {
        let _ = stderr.write_line(&format!("  Simplex iter: {simplex}"));
    }
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
            total_rows_active: 480,
            total_rows_generated: 1200,
            total_lp_solves: 36_000,
            total_time_ms: 5_000,
            total_first_try: Some(35_900),
            total_retried: Some(100),
            total_failed: Some(0),
            total_solve_time_seconds: Some(28.8),
            total_basis_offered: Some(34_000),
            total_basis_consistency_failures: Some(200),
            total_simplex_iterations: Some(1_800_000),
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
            mean_cost: None,
            std_cost: None,
            total_lp_solves: None,
            total_first_try: None,
            total_retried: None,
            total_failed_solves: None,
            total_solve_time_seconds: None,
            total_basis_offered: None,
            total_basis_consistency_failures: None,
            total_simplex_iterations: None,
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
    fn test_format_summary_row_stats() {
        let summary = RunSummary {
            training: TrainingSummary {
                total_rows_active: 480,
                total_rows_generated: 1200,
                ..make_training_summary()
            },
            simulation: None,
            output_dir: PathBuf::from("/tmp/out"),
        };
        let s = format_summary_string(&summary);

        assert!(
            s.contains("480 active / 1200 generated"),
            "summary must contain policy row counts, got: {s}"
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
            mean_cost: None,
            std_cost: None,
            total_lp_solves: None,
            total_first_try: None,
            total_retried: None,
            total_failed_solves: None,
            total_solve_time_seconds: None,
            total_basis_offered: None,
            total_basis_consistency_failures: None,
            total_simplex_iterations: None,
        };
        let summary = make_run_summary(Some(sim));
        print_summary(&Term::buffered_stderr(), &summary);
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

    // ── ModelProvenanceReport tests ───────────────────────────────────────────

    use super::{
        ModelProvenanceReport, ProvenanceSource, format_provenance_summary_string,
        print_provenance_summary,
    };

    fn make_provenance_report_full_estimation() -> ModelProvenanceReport {
        ModelProvenanceReport {
            estimation_path: "full_estimation".to_string(),
            seasonal_stats_source: ProvenanceSource::Estimated,
            ar_coefficients_source: ProvenanceSource::Estimated,
            correlation_source: ProvenanceSource::Estimated,
            opening_tree_source: ProvenanceSource::Estimated,
            n_hydros: 3,
            ar_method: Some("PACF".to_string()),
            ar_max_order: Some(6),
            white_noise_fallbacks: vec![],
        }
    }

    fn make_provenance_report_deterministic() -> ModelProvenanceReport {
        ModelProvenanceReport {
            estimation_path: "deterministic".to_string(),
            seasonal_stats_source: ProvenanceSource::NotApplicable,
            ar_coefficients_source: ProvenanceSource::NotApplicable,
            correlation_source: ProvenanceSource::NotApplicable,
            opening_tree_source: ProvenanceSource::NotApplicable,
            n_hydros: 0,
            ar_method: None,
            ar_max_order: None,
            white_noise_fallbacks: vec![],
        }
    }

    #[test]
    fn print_provenance_summary_does_not_panic() {
        let report = make_provenance_report_full_estimation();
        print_provenance_summary(&Term::buffered_stderr(), &report);
    }

    #[test]
    fn print_provenance_summary_deterministic_does_not_panic() {
        let report = make_provenance_report_deterministic();
        print_provenance_summary(&Term::buffered_stderr(), &report);
    }

    #[test]
    fn format_provenance_summary_contains_all_section_keys() {
        let report = make_provenance_report_full_estimation();
        let s = format_provenance_summary_string(&report);
        assert!(
            s.contains("Model provenance"),
            "output must contain header 'Model provenance', got: {s}"
        );
        assert!(
            s.contains("Estimation path:"),
            "output must contain 'Estimation path:' line, got: {s}"
        );
        assert!(
            s.contains("Seasonal stats:"),
            "output must contain 'Seasonal stats:' line, got: {s}"
        );
        assert!(
            s.contains("AR coefficients:"),
            "output must contain 'AR coefficients:' line, got: {s}"
        );
        assert!(
            s.contains("Correlation:"),
            "output must contain 'Correlation:' line, got: {s}"
        );
        assert!(
            s.contains("Opening tree:"),
            "output must contain 'Opening tree:' line, got: {s}"
        );
    }

    #[test]
    fn format_provenance_summary_full_estimation_includes_ar_detail() {
        let report = make_provenance_report_full_estimation();
        let s = format_provenance_summary_string(&report);
        assert!(
            s.contains("full_estimation"),
            "output must contain 'full_estimation' estimation path, got: {s}"
        );
        assert!(
            s.contains("(PACF, max order 6)"),
            "output must include AR method and max order parenthetical, got: {s}"
        );
    }

    #[test]
    fn format_provenance_summary_deterministic_no_ar_detail() {
        let report = make_provenance_report_deterministic();
        let s = format_provenance_summary_string(&report);
        assert!(
            s.contains("deterministic"),
            "output must contain 'deterministic' estimation path, got: {s}"
        );
        assert!(
            s.contains("n/a"),
            "output must contain 'n/a' for NotApplicable sources, got: {s}"
        );
        // No parenthetical when AR is NotApplicable.
        assert!(
            !s.contains("max order"),
            "output must NOT contain 'max order' for deterministic case, got: {s}"
        );
    }

    #[test]
    fn format_provenance_summary_user_file_source() {
        let report = ModelProvenanceReport {
            estimation_path: "user_provided_no_history".to_string(),
            seasonal_stats_source: ProvenanceSource::UserFile,
            ar_coefficients_source: ProvenanceSource::UserFile,
            correlation_source: ProvenanceSource::Estimated,
            opening_tree_source: ProvenanceSource::Estimated,
            n_hydros: 2,
            ar_method: None,
            ar_max_order: None,
            white_noise_fallbacks: vec![],
        };
        let s = format_provenance_summary_string(&report);
        assert!(
            s.contains("user_file"),
            "output must contain 'user_file' for UserFile source, got: {s}"
        );
        assert!(
            !s.contains("max order"),
            "output must NOT contain 'max order' when ar_method is None, got: {s}"
        );
    }

    // ── format_rank_list tests ────────────────────────────────────────────────

    use super::format_rank_list;

    #[test]
    fn test_format_rank_list_empty() {
        assert_eq!(format_rank_list(&[]), "");
    }

    #[test]
    fn test_format_rank_list_single() {
        assert_eq!(format_rank_list(&[5]), "5");
    }

    #[test]
    fn test_format_rank_list_contiguous() {
        assert_eq!(format_rank_list(&[0, 1, 2, 3]), "0\u{2013}3");
    }

    #[test]
    fn test_format_rank_list_non_contiguous() {
        assert_eq!(format_rank_list(&[0, 2, 5]), "0, 2, 5");
    }

    #[test]
    fn test_format_rank_list_mixed() {
        assert_eq!(
            format_rank_list(&[0, 1, 2, 7, 9, 10, 11]),
            "0\u{2013}2, 7, 9\u{2013}11"
        );
    }
}
