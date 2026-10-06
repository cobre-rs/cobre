//! Post-run summary block for the `cobre run` command.
//!
//! One printing function per run phase, each emitting its section independently
//! so the caller can place it at the right point in the execution flow. Every
//! `print_*` writer ignores write errors (fire-and-forget); most also share their
//! exact rendered lines with tests through a private `*_lines`/`*_line` helper, so
//! tests assert on the same code path production prints from.

use chrono::NaiveDate;
use cobre_comm::ExecutionTopology;
use cobre_io::SetupTimings;
use console::Term;

use std::path::Path;

use cobre_sddp::{BoundaryReconciliationReport, HydroModelSummary, ModelProvenanceReport};

fn hydro_model_summary_lines(summary: &HydroModelSummary) -> Vec<String> {
    vec![
        console::style("Hydro models").bold().to_string(),
        format!("  Production:    {}", format_production_line(summary)),
        format!("  Evaporation:   {}", format_evaporation_line(summary)),
    ]
}

/// Print hydro model summary to `stderr`.
pub fn print_hydro_model_summary(stderr: &Term, summary: &HydroModelSummary) {
    for line in hydro_model_summary_lines(summary) {
        let _ = stderr.write_line(&line);
    }
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

    let format_segment = |start: usize, end: usize| {
        if end > start {
            format!("{start}\u{2013}{end}")
        } else {
            format!("{start}")
        }
    };

    for &r in &ranks[1..] {
        if r == end + 1 {
            end = r;
        } else {
            segments.push(format_segment(start, end));
            start = r;
            end = r;
        }
    }
    segments.push(format_segment(start, end));
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
    topology: &ExecutionTopology,
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

    let _ = stderr.write_line(&console::style("Execution").bold().to_string());

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
            let backend_detail = if let Some(ref mpi) = topology.mpi {
                format!("MPI ({}, {})", mpi.library_version, mpi.standard_version)
            } else {
                "MPI".to_string()
            };
            let _ = stderr.write_line(&format!("  Backend:   {backend_detail}"));

            let thread_line = if let Some(ref mpi) = topology.mpi {
                format!("{}, {n_threads} {thread_word} per rank", mpi.thread_level)
            } else {
                format!("{n_threads} {thread_word} per rank")
            };
            let _ = stderr.write_line(&format!("  Threads:   {thread_line}"));

            let world_size = topology.world_size;
            let rank_word = if world_size == 1 { "rank" } else { "ranks" };
            let num_hosts = topology.num_hosts();
            if num_hosts <= 1 {
                let _ = stderr.write_line(&format!(
                    "  Layout:    {world_size} {rank_word} on {}",
                    topology.leader_hostname()
                ));
            } else {
                let _ = stderr.write_line(&format!(
                    "  Layout:    {world_size} {rank_word} across {num_hosts} nodes"
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

            if let Some(ref slurm) = topology.slurm {
                let mut slurm_parts: Vec<String> = vec![format!("job {}", slurm.job_id)];
                if let Some(ref node_list) = slurm.node_list {
                    slurm_parts.push(format!("nodes {node_list}"));
                }
                if let Some(cpus) = slurm.cpus_per_task {
                    slurm_parts.push(format!("{cpus} CPUs/task"));
                }
                let _ = stderr.write_line(&format!("  SLURM:     {}", slurm_parts.join(", ")));
            }
        }
        BackendKind::Auto => {
            let _ = stderr.write_line(&format!("  Backend:   {:?}", topology.backend));
            let _ = stderr.write_line(&format!("  Threads:   {n_threads} {thread_word}"));
        }
    }
}

/// Plane-source qualifiers (precomputed vs. computed from geometry) belong to
/// the model provenance section, not here.
fn format_production_line(summary: &HydroModelSummary) -> String {
    match (summary.n_constant, summary.n_fpha) {
        (0, 0) => "0 hydros".to_string(),
        (n_const, 0) => format!("{n_const} constant"),
        (0, n_fpha) => format!("{n_fpha} FPHA ({} planes)", summary.total_planes),
        (n_const, n_fpha) => format!(
            "{n_fpha} FPHA ({} planes), {n_const} constant",
            summary.total_planes
        ),
    }
}

/// Reference-volume source qualifiers (user-supplied vs. midpoint) belong to
/// the model provenance section.
fn format_evaporation_line(summary: &HydroModelSummary) -> String {
    format!(
        "{} linearized, {} without",
        summary.n_evaporation, summary.n_no_evaporation,
    )
}

fn setup_summary_lines(timings: &SetupTimings) -> Vec<String> {
    vec![
        console::style("Setup").bold().to_string(),
        format!(
            "  Load:            {}",
            format_split_duration(timings.load_seconds)
        ),
        format!(
            "  Stochastic fit:  {}",
            format_split_duration(timings.stochastic_fit_seconds)
        ),
        format!(
            "  Production fit:  {}",
            format_split_duration(timings.production_fit_seconds)
        ),
        format!(
            "  Evaporation fit: {}",
            format_split_duration(timings.evaporation_fit_seconds)
        ),
        format!(
            "  Broadcast:       {}",
            format_split_duration(timings.broadcast_seconds)
        ),
    ]
}

/// Print setup timing summary to `stderr`.
pub fn print_setup_summary(stderr: &Term, timings: &SetupTimings) {
    for line in setup_summary_lines(timings) {
        let _ = stderr.write_line(&line);
    }
}

/// Returns `" (method, max order N)"` when AR method is known, or an empty
/// string when AR is `NotApplicable`.
fn provenance_ar_detail(report: &ModelProvenanceReport) -> String {
    match (&report.inflow.ar_method, report.inflow.ar_max_order) {
        (Some(method), Some(max_order)) => format!(" ({method}, max order {max_order})"),
        _ => String::new(),
    }
}

fn provenance_summary_lines(report: &ModelProvenanceReport) -> Vec<String> {
    let ar_detail = provenance_ar_detail(report);
    vec![
        console::style("Model provenance").bold().to_string(),
        format!("  Estimation path: {}", report.inflow.estimation_path),
        format!("  Seasonal stats:  {}", report.inflow.seasonal_stats_source),
        format!(
            "  AR coefficients: {}{}",
            report.inflow.ar_coefficients_source, ar_detail
        ),
        format!("  Correlation:     {}", report.inflow.correlation_source),
        format!("  Opening tree:    {}", report.inflow.opening_tree_source),
    ]
}

/// Print model provenance summary to `stderr`.
pub fn print_provenance_summary(stderr: &Term, report: &ModelProvenanceReport) {
    for line in provenance_summary_lines(report) {
        let _ = stderr.write_line(&line);
    }
}

/// Wording is independent of [`BoundaryReconciliationReport::summary_line`]
/// (legacy "N source slots dropped" for `cobre validate`); totals come from
/// [`BoundaryReconciliationReport::tally_totals`].
fn format_boundary_reconciliation_row(report: &BoundaryReconciliationReport) -> String {
    if !report.reconciled {
        return "dimension-only load (entity manifest absent)".to_string();
    }
    let (copy, fan_out, default_zero, dropped) = report.tally_totals();
    format!(
        "{copy} copied, {fan_out} fanned out, {default_zero} defaulted to 0.0, {dropped} dropped"
    )
}

fn boundary_summary_lines(
    loaded: usize,
    boundary_date: NaiveDate,
    path: &Path,
    report: &BoundaryReconciliationReport,
) -> Vec<String> {
    vec![
        console::style("Boundary policy").bold().to_string(),
        format!("  Cuts loaded:    {loaded} (priced at {boundary_date})"),
        format!("  Source:         {}", path.display()),
        format!(
            "  Reconciliation: {}",
            format_boundary_reconciliation_row(report)
        ),
    ]
}

/// Print boundary-policy load summary to `stderr`.
pub fn print_boundary_summary(
    stderr: &Term,
    loaded: usize,
    boundary_date: NaiveDate,
    path: &Path,
    report: &BoundaryReconciliationReport,
) {
    for line in boundary_summary_lines(loaded, boundary_date, path, report) {
        let _ = stderr.write_line(&line);
    }
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
///
/// The phase-wall, wait, and serial `*_seconds` timing fields are `None` when
/// per-iteration timing is unavailable.
pub struct TrainingSummary {
    /// Total number of iterations completed.
    pub iterations: u64,

    /// Whether training converged within the configured tolerance.
    pub converged: bool,

    /// Iteration at which convergence was detected; `None` when training terminated for another reason.
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

    /// Rows loaded from a boundary policy; zero when no boundary policy loaded.
    pub total_rows_loaded: u64,

    /// Sum of resident rows-in-LP over every lazy-selection solve (reduced across
    /// ranks). Zero for pool-deactivating methods — the renderer uses zero to gate
    /// the rows-in-LP line; only Dynamic Cut Selection populates it.
    pub rows_in_lp_total: u64,

    /// Number of lazy-selection solves (reduced across ranks).
    pub rows_in_lp_solve_count: u64,

    /// Largest resident rows-in-LP over any single lazy-selection solve (reduced
    /// across ranks).
    pub rows_in_lp_max: u64,

    /// Number of stages in the planning horizon.
    pub num_stages: u32,

    /// Total number of LP solves across all ranks, stages, iterations, and
    /// passes.  Aggregated via `allreduce(Sum)` so that the reported value is
    /// invariant regardless of the parallel configuration.
    pub total_lp_solves: u64,

    /// Total elapsed wall-clock time for the training run (milliseconds).
    pub total_time_ms: u64,

    /// Number of solves that returned optimal on the first attempt.
    pub total_first_try: u64,

    /// Number of solves that required retry escalation.
    pub total_retried: u64,

    /// Number of solves that exhausted all retry levels.
    pub total_failed: u64,

    /// Forward-phase cumulative LP solve wall time in seconds, summed across all (rank, worker) pairs.
    pub total_forward_solve_seconds: f64,

    /// Backward-phase cumulative LP solve wall time in seconds, summed
    /// across all (rank, worker) pairs. See [`Self::total_forward_solve_seconds`].
    pub total_backward_solve_seconds: f64,

    /// Effective parallelism (`n_ranks * n_workers_local`).
    pub parallelism: u32,

    /// Initial optimality gap (iteration 1) in percent; `None` when convergence data is unavailable or the run completed in zero iterations.
    pub initial_gap_percent: Option<f64>,

    /// Coordinator-measured forward-phase wall, from `forward_wall_ms`.
    pub forward_phase_wall_seconds: Option<f64>,

    /// Coordinator-measured backward-phase wall, from `backward_wall_ms`.
    pub backward_phase_wall_seconds: Option<f64>,

    /// Forward-phase worker wait (load imbalance), from `fwd_load_imbalance_ms`.
    pub forward_wait_seconds: Option<f64>,

    /// Backward-phase worker wait (load imbalance), from `bwd_load_imbalance_ms`.
    pub backward_wait_seconds: Option<f64>,

    /// `Serial`-bucket lower-bound evaluation time, from `lower_bound_ms`.
    pub serial_lower_bound_seconds: Option<f64>,

    /// `Serial`-bucket row-selection time, from `cut_selection_ms`.
    pub serial_cut_selection_seconds: Option<f64>,

    /// `Serial`-bucket per-stage row-sync `allgatherv` time, from `cut_sync_ms`.
    pub serial_cut_sync_seconds: Option<f64>,

    /// `Serial`-bucket MPI allreduce (forward bound synchronization) time,
    /// from `mpi_allreduce_ms`.
    pub serial_allreduce_seconds: Option<f64>,

    /// `Serial`-bucket rayon scheduling overhead, summed over both phases
    /// (`fwd_scheduling_overhead_ms + bwd_scheduling_overhead_ms`).
    pub serial_scheduling_seconds: Option<f64>,
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
    pub total_lp_solves: u64,

    /// Solves that returned optimal on the first attempt.
    pub total_first_try: u64,

    /// Solves that required retry escalation before succeeding.
    pub total_retried: u64,

    /// Solves that exhausted all retry levels.
    pub total_failed_solves: u64,

    /// Cumulative LP solve wall time in seconds, summed across all (rank, worker) pairs.
    pub total_solve_time_seconds: f64,

    /// Effective parallelism (`n_ranks * n_workers_local`).
    pub parallelism: u32,
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
    if converged && let Some(iter) = converged_at {
        return format!("converged at iter {iter}");
    }
    reason.to_string()
}

/// Format a duration for the time-split breakdown lines.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn format_split_duration(seconds: f64) -> String {
    if seconds < 1.0 {
        let ms = (seconds * 1000.0).round() as u64;
        format!("{ms}ms")
    } else if seconds < 60.0 {
        format!("{seconds:.1}s")
    } else if seconds < 3600.0 {
        let total = seconds.round() as u64;
        format!("{}m {}s", total / 60, total % 60)
    } else {
        let total = seconds.round() as u64;
        format!("{}h {}m", total / 3600, (total % 3600) / 60)
    }
}

/// Per-worker average wall time = `cumulative_seconds / parallelism`, capped at `cap_seconds`.
fn per_worker_mean_seconds(cumulative_seconds: f64, parallelism: u32, cap_seconds: f64) -> f64 {
    if parallelism == 0 {
        return 0.0;
    }
    (cumulative_seconds / f64::from(parallelism))
        .min(cap_seconds)
        .max(0.0)
}

/// Render a percentage for the time-split breakdown, showing `<1%` instead of
/// rounding a nonzero share down to `0%`.
fn format_pct(pct: f64) -> String {
    if pct > 0.0 && pct < 1.0 {
        "<1%".to_string()
    } else {
        format!("{pct:.0}%")
    }
}

fn pct_of(part: f64, total: f64) -> f64 {
    if total > 0.0 {
        100.0 * part / total
    } else {
        0.0
    }
}

/// The three training Time-split component walls (forward, backward, serial),
/// in seconds. `None` when per-iteration phase-wall timing is unavailable.
#[allow(clippy::cast_precision_loss)]
fn time_split_training_walls(t: &TrainingSummary) -> Option<(f64, f64, f64)> {
    let forward_wall = t.forward_phase_wall_seconds?;
    let backward_wall = t.backward_phase_wall_seconds?;
    let total_s = t.total_time_ms as f64 / 1000.0;
    let serial_wall = (total_s - forward_wall - backward_wall).max(0.0);
    Some((forward_wall, backward_wall, serial_wall))
}

/// The `solve {..} · wait {..}` suffix for a phase Time-split line; empty when per-worker solve data is unavailable.
fn format_phase_solve_wait(
    solve_cumulative: f64,
    wait: Option<f64>,
    parallelism: u32,
    phase_wall: f64,
) -> String {
    let Some(wait) = wait else {
        return String::new();
    };
    if parallelism == 0 {
        return String::new();
    }
    let solve = per_worker_mean_seconds(solve_cumulative, parallelism, phase_wall);
    let wait_pct = if phase_wall > 0.0 {
        100.0 * wait / phase_wall
    } else {
        0.0
    };
    format!(
        "   solve {} \u{b7} wait {} ({} of phase)",
        format_split_duration(solve),
        format_split_duration(wait),
        format_pct(wait_pct)
    )
}

/// Serial Time-split breakdown; `allreduce`/`sync` inserted only when nonzero (MPI runs).
fn format_serial_breakdown(t: &TrainingSummary, serial_wall: f64) -> String {
    let bound = t.serial_lower_bound_seconds.unwrap_or(0.0);
    let selection = t.serial_cut_selection_seconds.unwrap_or(0.0);
    let allreduce = t.serial_allreduce_seconds.unwrap_or(0.0);
    let cut_sync = t.serial_cut_sync_seconds.unwrap_or(0.0);
    let scheduling = t.serial_scheduling_seconds.unwrap_or(0.0);

    let mut parts = vec![
        format!("bound {}", format_split_duration(bound)),
        format!("selection {}", format_split_duration(selection)),
    ];
    let mut accounted = bound + selection;
    if allreduce > 0.0 {
        parts.push(format!("allreduce {}", format_split_duration(allreduce)));
        accounted += allreduce;
    }
    if cut_sync > 0.0 {
        parts.push(format!("sync {}", format_split_duration(cut_sync)));
        accounted += cut_sync;
    }
    // Floor at `scheduling` so known component is never hidden below the residual clamp.
    let other = (serial_wall - accounted).max(scheduling).max(0.0);
    parts.push(format!("other {}", format_split_duration(other)));

    format!("   {}", parts.join(" \u{b7} "))
}

/// Three-line training `Time split` block; empty when phase-wall timing unavailable.
#[allow(clippy::cast_precision_loss)]
fn format_time_split_training(t: &TrainingSummary) -> Vec<String> {
    let Some((forward_wall, backward_wall, serial_wall)) = time_split_training_walls(t) else {
        return Vec::new();
    };
    let total_s = t.total_time_ms as f64 / 1000.0;

    vec![
        format!(
            "  Time split:   Forward  {} ({}){}",
            format_split_duration(forward_wall),
            format_pct(pct_of(forward_wall, total_s)),
            format_phase_solve_wait(
                t.total_forward_solve_seconds,
                t.forward_wait_seconds,
                t.parallelism,
                forward_wall,
            )
        ),
        format!(
            "                Backward {} ({}){}",
            format_split_duration(backward_wall),
            format_pct(pct_of(backward_wall, total_s)),
            format_phase_solve_wait(
                t.total_backward_solve_seconds,
                t.backward_wait_seconds,
                t.parallelism,
                backward_wall,
            )
        ),
        format!(
            "                Serial   {} ({}){}",
            format_split_duration(serial_wall),
            format_pct(pct_of(serial_wall, total_s)),
            format_serial_breakdown(t, serial_wall)
        ),
    ]
}

/// Pool-level `active / generated` counts, plus a rows-in-LP line when lazy selection ran (`rows_in_lp_solve_count > 0`).
/// Lazy runs annotate the pool total per-stage so it's comparable to per-solve rows-in-LP figures.
/// Boundary-loaded cuts split the generated count when `total_rows_loaded > 0`.
fn policy_rows_lines(t: &TrainingSummary) -> Vec<String> {
    let per_stage = if t.rows_in_lp_solve_count > 0 && t.num_stages > 0 {
        #[allow(clippy::cast_precision_loss)]
        let avg = t.total_rows_active as f64 / f64::from(t.num_stages);
        format!("  (mean {avg:.1} active/stage)")
    } else {
        String::new()
    };
    let loaded_annotation = if t.total_rows_loaded > 0 {
        let this_run = t.total_rows_generated.saturating_sub(t.total_rows_loaded);
        format!(
            " ({this_run} this run + {} loaded boundary)",
            t.total_rows_loaded
        )
    } else {
        String::new()
    };
    let mut out = vec![format!(
        "  Policy rows:  {} active / {} generated{loaded_annotation}{per_stage}",
        t.total_rows_active, t.total_rows_generated
    )];
    if t.rows_in_lp_solve_count > 0 {
        #[allow(clippy::cast_precision_loss)]
        let mean = t.rows_in_lp_total as f64 / t.rows_in_lp_solve_count as f64;
        out.push(format!(
            "  Rows in LP/solve:  mean {mean:.1}, max {}  (over {} solves)",
            t.rows_in_lp_max, t.rows_in_lp_solve_count
        ));
    }
    out
}

fn training_summary_lines(t: &TrainingSummary) -> Vec<String> {
    let duration = format_duration(t.total_time_ms);
    let convergence_detail = format_convergence_detail(t.converged, t.converged_at, &t.reason);

    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "{} ({} iterations, {convergence_detail})",
        console::style(format!("Training complete in {duration}")).bold(),
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
    if let Some(initial) = t.initial_gap_percent {
        lines.push(format!(
            "  Gap:          {:.1}% (started at {:.1}%)",
            t.gap_percent, initial
        ));
    } else {
        lines.push(format!("  Gap:          {:.1}%", t.gap_percent));
    }
    lines.extend(policy_rows_lines(t));
    lines.push(format!(
        "  LP solves:    {} ({} first-try, {} retried, {} failed)",
        t.total_lp_solves, t.total_first_try, t.total_retried, t.total_failed
    ));
    if t.iterations > 0 {
        #[allow(clippy::cast_precision_loss)]
        let avg_iter_ms = t.total_time_ms as f64 / t.iterations as f64;
        lines.push(format!("  Avg iter:     {avg_iter_ms:.0}ms"));
    }
    lines.extend(format_time_split_training(t));
    lines
}

/// Print training summary to `stderr`.
pub fn print_training_summary(stderr: &Term, t: &TrainingSummary) {
    for line in training_summary_lines(t) {
        let _ = stderr.write_line(&line);
    }
}

fn simulation_summary_lines(sim: &SimulationSummary) -> Vec<String> {
    let duration = format_duration(sim.total_time_ms);

    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "{} ({} scenarios)",
        console::style(format!("Simulation complete in {duration}")).bold(),
        sim.n_scenarios
    ));
    lines.push(format!(
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
        lines.push(format!(
            "  Expected cost: {mean:.5e} +/- {ci95:.5e} (std: {std:.5e})"
        ));
    }
    lines.push(format!(
        "  LP solves:    {} ({} first-try, {} retried, {} failed)",
        sim.total_lp_solves, sim.total_first_try, sim.total_retried, sim.total_failed_solves
    ));
    if sim.completed > 0 {
        #[allow(clippy::cast_precision_loss)]
        let avg_s = sim.total_time_ms as f64 / 1000.0 / f64::from(sim.completed);
        lines.push(format!("  Avg/scenario: {avg_s:.3}s"));
    }
    #[allow(clippy::cast_precision_loss)]
    let total_s = sim.total_time_ms as f64 / 1000.0;
    let solver = per_worker_mean_seconds(sim.total_solve_time_seconds, sim.parallelism, total_s);
    let other = (total_s - solver).max(0.0);
    lines.push(format!(
        "  Time split:   Solver {} ({:.0}%)",
        format_split_duration(solver),
        pct_of(solver, total_s)
    ));
    lines.push(format!(
        "                Other  {} ({:.0}%)",
        format_split_duration(other),
        pct_of(other, total_s)
    ));
    lines
}

/// Print simulation summary to `stderr`.
pub fn print_simulation_summary(stderr: &Term, sim: &SimulationSummary) {
    for line in simulation_summary_lines(sim) {
        let _ = stderr.write_line(&line);
    }
}

fn output_path_line(output_dir: &Path, write_secs: f64) -> String {
    format!(
        "{} {}/ {}",
        console::style("Output written to").bold(),
        console::style(output_dir.display()).dim(),
        console::style(format!("({write_secs:.1}s)")).dim()
    )
}

/// Print output path and write duration to `stderr`.
pub fn print_output_path(stderr: &Term, output_dir: &Path, write_secs: f64) {
    let _ = stderr.write_line(&output_path_line(output_dir, write_secs));
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use console::Term;

    use super::{
        SimulationSummary, TrainingSummary, format_duration, format_split_duration,
        output_path_line, policy_rows_lines, print_output_path, print_simulation_summary,
        print_training_summary, simulation_summary_lines, time_split_training_walls,
        training_summary_lines,
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
            total_rows_loaded: 0,
            rows_in_lp_total: 1_440,
            rows_in_lp_solve_count: 120,
            rows_in_lp_max: 18,
            num_stages: 40,
            total_lp_solves: 36_000,
            total_time_ms: 5_000,
            total_first_try: 35_900,
            total_retried: 100,
            total_failed: 0,
            total_forward_solve_seconds: 12.0,
            total_backward_solve_seconds: 16.8,
            parallelism: 1,
            initial_gap_percent: Some(28.0),
            forward_phase_wall_seconds: Some(15.0),
            backward_phase_wall_seconds: Some(20.0),
            forward_wait_seconds: Some(3.0),
            backward_wait_seconds: Some(5.0),
            serial_lower_bound_seconds: Some(0.5),
            serial_cut_selection_seconds: Some(0.3),
            serial_cut_sync_seconds: Some(0.2),
            serial_allreduce_seconds: Some(0.1),
            serial_scheduling_seconds: Some(0.4),
        }
    }

    fn make_simulation_summary() -> SimulationSummary {
        SimulationSummary {
            n_scenarios: 100,
            completed: 100,
            failed: 0,
            total_time_ms: 5_000,
            mean_cost: None,
            std_cost: None,
            total_lp_solves: 4000,
            total_first_try: 3950,
            total_retried: 40,
            total_failed_solves: 10,
            total_solve_time_seconds: 4.2,
            parallelism: 2,
        }
    }

    #[test]
    fn policy_rows_lines_adds_rows_in_lp_only_for_lazy_runs() {
        let mut t = make_training_summary();

        // Lazy run (solve_count > 0): the pool line is kept AND a rows-in-LP line
        // is added (mean = 1440/120 = 12.0, max 18, over 120 solves).
        t.rows_in_lp_total = 1_440;
        t.rows_in_lp_solve_count = 120;
        t.rows_in_lp_max = 18;
        let lines = policy_rows_lines(&t);
        assert_eq!(lines.len(), 2, "lazy run shows pool line + rows-in-LP line");
        assert!(lines[0].contains("480 active / 1200 generated"));
        // Pool total annotated per-stage (480 / 40 stages = 12.0) — same basis as
        // the per-solve rows-in-LP figures below.
        assert!(
            lines[0].contains("mean 12.0 active/stage"),
            "pool line must carry the per-stage average for lazy runs: {:?}",
            lines[0]
        );
        assert!(
            lines[1].contains("Rows in LP/solve")
                && lines[1].contains("mean 12.0")
                && lines[1].contains("max 18")
                && lines[1].contains("over 120 solves"),
            "rows-in-LP line: {:?}",
            lines[1]
        );

        t.rows_in_lp_total = 0;
        t.rows_in_lp_solve_count = 0;
        t.rows_in_lp_max = 0;
        let lines = policy_rows_lines(&t);
        assert_eq!(lines.len(), 1, "non-lazy run shows only the pool line");
        assert!(
            !lines.iter().any(|l| l.contains("Rows in LP/solve")),
            "non-lazy run must not show the rows-in-LP line: {lines:?}"
        );
        assert!(
            !lines[0].contains("active/stage"),
            "non-lazy run must not annotate the pool line per-stage: {:?}",
            lines[0]
        );
    }

    #[test]
    fn policy_rows_lines_splits_generated_when_boundary_cuts_are_loaded() {
        let mut t = make_training_summary();
        t.total_rows_active = 10_009;
        t.total_rows_generated = 10_009;
        t.total_rows_loaded = 10_000;
        t.rows_in_lp_total = 0;
        t.rows_in_lp_solve_count = 0;
        t.rows_in_lp_max = 0;

        let lines = policy_rows_lines(&t);
        assert_eq!(lines.len(), 1);
        assert!(
            lines[0]
                .contains("10009 active / 10009 generated (9 this run + 10000 loaded boundary)"),
            "loaded > 0 must split the generated count: {:?}",
            lines[0]
        );
    }

    #[test]
    fn policy_rows_lines_omits_loaded_annotation_when_nothing_was_loaded() {
        let mut t = make_training_summary();
        t.total_rows_loaded = 0;

        let lines = policy_rows_lines(&t);
        assert!(
            !lines[0].contains("loaded boundary"),
            "loaded == 0 must render exactly as today, with no annotation: {:?}",
            lines[0]
        );
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
    fn training_summary_lines_never_mentions_simulation() {
        let s = training_summary_lines(&make_training_summary()).join("\n");

        assert!(
            s.contains("Training complete"),
            "training summary must contain 'Training complete'"
        );
        assert!(
            !s.contains("Simulation"),
            "training summary must never mention 'Simulation', got: {s}"
        );
    }

    #[test]
    fn training_and_simulation_lines_each_contain_their_header() {
        let sim = SimulationSummary {
            n_scenarios: 200,
            completed: 198,
            failed: 2,
            total_time_ms: 10_000,
            total_lp_solves: 7920,
            total_first_try: 7850,
            total_retried: 60,
            total_solve_time_seconds: 8.5,
            parallelism: 4,
            ..make_simulation_summary()
        };
        let training_s = training_summary_lines(&make_training_summary()).join("\n");
        let simulation_s = simulation_summary_lines(&sim).join("\n");

        assert!(
            training_s.contains("Training complete"),
            "training summary must contain 'Training complete'"
        );
        assert!(
            simulation_s.contains("Simulation complete"),
            "simulation summary must contain 'Simulation complete'"
        );
    }

    #[test]
    fn training_summary_lines_contains_bounds() {
        let training = TrainingSummary {
            lower_bound: 100.5,
            ..make_training_summary()
        };
        let s = training_summary_lines(&training).join("\n");

        assert!(
            s.contains("1.00500e2"),
            "summary must contain '1.00500e2' (scientific notation) for lower_bound = 100.5, got: {s}"
        );
    }

    #[test]
    fn training_summary_lines_converged_detail() {
        let training = TrainingSummary {
            converged: true,
            converged_at: Some(38),
            reason: "bound_stalling".to_string(),
            ..make_training_summary()
        };
        let s = training_summary_lines(&training).join("\n");

        assert!(
            s.contains("converged at iter 38"),
            "summary must contain 'converged at iter 38', got: {s}"
        );
    }

    #[test]
    fn training_summary_lines_non_converged_shows_reason() {
        let training = TrainingSummary {
            converged: false,
            converged_at: None,
            reason: "iteration_limit".to_string(),
            ..make_training_summary()
        };
        let s = training_summary_lines(&training).join("\n");

        assert!(
            s.contains("iteration_limit"),
            "summary must contain the termination reason when not converged, got: {s}"
        );
    }

    #[test]
    fn training_summary_lines_time_3m42s() {
        let training = TrainingSummary {
            total_time_ms: 222_000,
            ..make_training_summary()
        };
        let s = training_summary_lines(&training).join("\n");

        assert!(
            s.contains("3m 42s"),
            "summary must contain '3m 42s' for total_time_ms = 222_000, got: {s}"
        );
    }

    #[test]
    fn training_summary_lines_scientific_notation() {
        let training = TrainingSummary {
            lower_bound: 45230.41,
            ..make_training_summary()
        };
        let s = training_summary_lines(&training).join("\n");

        assert!(
            s.contains("4.52304e4"),
            "summary must contain '4.52304e4' (scientific notation) for lower_bound = 45230.41, got: {s}"
        );
    }

    #[test]
    fn output_path_line_contains_output_dir() {
        let s = output_path_line(&PathBuf::from("/my/output/dir"), 0.0);

        assert!(
            s.contains("/my/output/dir"),
            "output path line must contain the output_dir path, got: {s}"
        );
    }

    #[test]
    fn training_summary_lines_includes_row_stats() {
        let training = TrainingSummary {
            total_rows_active: 480,
            total_rows_generated: 1200,
            ..make_training_summary()
        };
        let s = training_summary_lines(&training).join("\n");

        assert!(
            s.contains("480 active / 1200 generated"),
            "summary must contain policy row counts, got: {s}"
        );
    }

    // ── Time split (training) tests ───────────────────────────────────────

    #[test]
    fn format_time_split_training_forward_line_shows_solve_and_wait() {
        let s = training_summary_lines(&make_training_summary()).join("\n");

        let forward_line = s.lines().find(|l| l.contains("Forward"));
        assert!(forward_line.is_some(), "expected a Forward line in: {s}");
        let forward_line = forward_line.expect("checked above");
        assert!(
            forward_line.contains("solve") && forward_line.contains("wait"),
            "Forward line must show the solve/wait decomposition, got: {forward_line:?}"
        );
        assert!(
            s.lines().any(|l| l.contains("Serial")),
            "summary must contain a Serial line, got: {s}"
        );
    }

    #[test]
    fn format_time_split_training_backward_degrades_without_backward_wait() {
        let training = TrainingSummary {
            backward_wait_seconds: None,
            ..make_training_summary()
        };
        let s = training_summary_lines(&training).join("\n");

        let backward_line = s.lines().find(|l| l.contains("Backward"));
        assert!(backward_line.is_some(), "expected a Backward line in: {s}");
        let backward_line = backward_line.expect("checked above");
        assert!(
            !backward_line.contains("solve") && !backward_line.contains("wait"),
            "Backward line must omit the solve/wait decomposition when solve data is absent, got: {backward_line:?}"
        );

        let forward_line = s.lines().find(|l| l.contains("Forward"));
        assert!(forward_line.is_some(), "expected a Forward line in: {s}");
        let forward_line = forward_line.expect("checked above");
        assert!(
            forward_line.contains("solve") && forward_line.contains("wait"),
            "Forward line must still decompose into solve/wait when its own data is present, got: {forward_line:?}"
        );
    }

    #[test]
    fn format_phase_solve_wait_degrades_when_parallelism_zero() {
        let training = TrainingSummary {
            parallelism: 0,
            ..make_training_summary()
        };
        let s = training_summary_lines(&training).join("\n");

        let forward_line = s.lines().find(|l| l.contains("Forward"));
        assert!(forward_line.is_some(), "expected a Forward line in: {s}");
        let forward_line = forward_line.expect("checked above");
        assert!(
            !forward_line.contains("solve"),
            "zero parallelism must never be rendered as 'solve 0', got: {forward_line:?}"
        );
    }

    #[test]
    fn time_split_training_walls_sum_to_total_wall() {
        let training = TrainingSummary {
            total_time_ms: 40_000,
            forward_phase_wall_seconds: Some(25.0),
            backward_phase_wall_seconds: Some(12.0),
            ..make_training_summary()
        };
        let (forward_wall, backward_wall, serial_wall) =
            time_split_training_walls(&training).expect("phase walls are present");
        assert!(
            (forward_wall + backward_wall + serial_wall - 40.0).abs() < 1e-9,
            "the three phase walls must sum to total_time_ms/1000, got {forward_wall} + {backward_wall} + {serial_wall}"
        );

        let s = training_summary_lines(&training).join("\n");
        assert!(s.contains(&format_split_duration(forward_wall)), "got: {s}");
        assert!(
            s.contains(&format_split_duration(backward_wall)),
            "got: {s}"
        );
        assert!(s.contains(&format_split_duration(serial_wall)), "got: {s}");
    }

    #[test]
    fn format_time_split_training_omits_block_when_phase_wall_absent() {
        let training = TrainingSummary {
            forward_phase_wall_seconds: None,
            backward_phase_wall_seconds: None,
            ..make_training_summary()
        };
        assert!(time_split_training_walls(&training).is_none());

        let s = training_summary_lines(&training).join("\n");
        assert!(
            !s.contains("Time split"),
            "the training Time split block must be omitted entirely when phase-wall data is unavailable, got: {s}"
        );
    }

    #[test]
    fn format_serial_breakdown_computes_other_residual() {
        let bound = 1.5;
        let selection = 0.4;
        let scheduling = 0.3;
        let training = TrainingSummary {
            total_time_ms: 20_000,
            forward_phase_wall_seconds: Some(10.0),
            backward_phase_wall_seconds: Some(4.0),
            serial_lower_bound_seconds: Some(bound),
            serial_cut_selection_seconds: Some(selection),
            serial_allreduce_seconds: Some(0.0),
            serial_cut_sync_seconds: Some(0.0),
            serial_scheduling_seconds: Some(scheduling),
            ..make_training_summary()
        };
        let (_, _, serial_wall) =
            time_split_training_walls(&training).expect("phase walls are present");
        let expected_other = (serial_wall - bound - selection).max(scheduling).max(0.0);

        let s = training_summary_lines(&training).join("\n");
        let serial_line = s.lines().find(|l| l.contains("Serial"));
        assert!(serial_line.is_some(), "expected a Serial line in: {s}");
        let serial_line = serial_line.expect("checked above");

        assert!(
            serial_line.contains(&format!("bound {}", format_split_duration(bound))),
            "got: {serial_line}"
        );
        assert!(
            serial_line.contains(&format!("selection {}", format_split_duration(selection))),
            "got: {serial_line}"
        );
        assert!(
            serial_line.contains(&format!("other {}", format_split_duration(expected_other))),
            "got: {serial_line}"
        );
        assert!(
            !serial_line.contains("allreduce") && !serial_line.contains("sync"),
            "zero MPI serial components must collapse into other, got: {serial_line}"
        );
    }

    // ── Progress-line / summary-builder phase-wall invariant ───────────────

    use std::sync::mpsc;

    use cobre_core::TrainingEvent;
    use cobre_io::IterationRecord;
    use cobre_sddp::sum_phase_timing_ms;

    use crate::progress::{RenderMode, run_progress_thread};

    fn make_synthetic_iteration_summary(
        iteration: u64,
        forward_ms: u64,
        backward_ms: u64,
    ) -> TrainingEvent {
        TrainingEvent::IterationSummary {
            iteration,
            lower_bound: 100.0,
            upper_bound: 110.0,
            gap: 0.09,
            wall_time_ms: forward_ms + backward_ms,
            iteration_time_ms: forward_ms + backward_ms,
            forward_ms,
            backward_ms,
            lp_solves: 12,
            solve_time_ms: 0.0,
            lower_bound_eval_ms: 0,
            fwd_setup_time_ms: 0,
            fwd_load_imbalance_ms: 0,
            fwd_scheduling_overhead_ms: 0,
            rows_in_lp_sum: 0,
            rows_in_lp_count: 0,
            rows_in_lp_max: 0,
        }
    }

    fn zero_iteration_record_with_walls(
        iteration: u32,
        forward_wall_ms: u64,
        backward_wall_ms: u64,
    ) -> IterationRecord {
        IterationRecord {
            iteration,
            lower_bound: 0.0,
            upper_bound: 0.0,
            upper_bound_std: 0.0,
            gap_percent: None,
            cuts_added: 0,
            cuts_removed: 0,
            cuts_active: 0,
            time_forward_ms: 0,
            time_backward_ms: 0,
            time_total_ms: 0,
            time_forward_wall_ms: forward_wall_ms,
            time_backward_wall_ms: backward_wall_ms,
            time_cut_selection_ms: 0,
            time_mpi_allreduce_ms: 0,
            time_cut_sync_ms: 0,
            time_lower_bound_ms: 0,
            time_state_exchange_ms: 0,
            time_cut_batch_build_ms: 0,
            time_bwd_load_imbalance_ms: 0,
            time_bwd_scheduling_overhead_ms: 0,
            time_fwd_load_imbalance_ms: 0,
            time_fwd_scheduling_overhead_ms: 0,
            time_overhead_ms: 0,
            forward_passes: 0,
            lp_solves: 0,
            solve_time_ms: 0.0,
            mean_rows_in_lp: 0.0,
        }
    }

    /// Feeds the same synthetic per-iteration `forward_ms`/`backward_ms` sequence
    /// through the real progress-line accumulator (`run_progress_thread`, the
    /// code the per-iteration `fwd: {forward_ms}ms / bwd: {backward_ms}ms` line
    /// renders from) and the real summary-builder accumulation path
    /// (`sum_phase_timing_ms`, what `commands/run/training.rs` calls to
    /// populate `TrainingSummary::forward_phase_wall_seconds`), then asserts
    /// both totals agree.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn forward_backward_phase_wall_seconds_match_summed_progress_line_walls() {
        let iter_1 = (1_u64, 1_250_u64, 3_400_u64);
        let iter_2 = (2_u64, 980_u64, 2_100_u64);
        let iter_3 = (3_u64, 1_430_u64, 3_050_u64);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let handle = run_progress_thread(rx, RenderMode::Log, 3, 120);
        for &(iteration, forward_ms, backward_ms) in &[iter_1, iter_2, iter_3] {
            tx.send(make_synthetic_iteration_summary(
                iteration,
                forward_ms,
                backward_ms,
            ))
            .unwrap();
        }
        drop(tx);

        let (progress_forward_ms, progress_backward_ms) =
            handle.join().iter().fold((0_u64, 0_u64), |(f, b), event| {
                if let TrainingEvent::IterationSummary {
                    forward_ms,
                    backward_ms,
                    ..
                } = event
                {
                    (f + forward_ms, b + backward_ms)
                } else {
                    (f, b)
                }
            });

        let expected_forward_ms = iter_1.1 + iter_2.1 + iter_3.1;
        let expected_backward_ms = iter_1.2 + iter_2.2 + iter_3.2;
        assert_eq!(
            progress_forward_ms, expected_forward_ms,
            "the progress accumulator must not drop or alter forward_ms"
        );
        assert_eq!(
            progress_backward_ms, expected_backward_ms,
            "the progress accumulator must not drop or alter backward_ms"
        );

        let records = vec![
            zero_iteration_record_with_walls(1, iter_1.1, iter_1.2),
            zero_iteration_record_with_walls(2, iter_2.1, iter_2.2),
            zero_iteration_record_with_walls(3, iter_3.1, iter_3.2),
        ];
        let totals = sum_phase_timing_ms(&records);

        let training = TrainingSummary {
            forward_phase_wall_seconds: Some(totals.forward_wall_ms as f64 / 1000.0),
            backward_phase_wall_seconds: Some(totals.backward_wall_ms as f64 / 1000.0),
            ..make_training_summary()
        };

        assert_eq!(
            training.forward_phase_wall_seconds,
            Some(progress_forward_ms as f64 / 1000.0),
            "forward_phase_wall_seconds must equal the summed progress-line forward walls"
        );
        assert_eq!(
            training.backward_phase_wall_seconds,
            Some(progress_backward_ms as f64 / 1000.0),
            "backward_phase_wall_seconds must equal the summed progress-line backward walls"
        );
    }

    #[test]
    fn print_training_summary_does_not_panic() {
        let training = make_training_summary();
        print_training_summary(&Term::buffered_stderr(), &training);
    }

    #[test]
    fn print_simulation_summary_does_not_panic() {
        let sim = make_simulation_summary();
        print_simulation_summary(&Term::buffered_stderr(), &sim);
    }

    #[test]
    fn print_output_path_does_not_panic() {
        print_output_path(&Term::buffered_stderr(), &PathBuf::from("/tmp/out"), 1.2);
    }

    // ── HydroModelSummary tests ────────────────────────────────────────────

    use super::{HydroModelSummary, hydro_model_summary_lines, print_hydro_model_summary};
    use cobre_core::EntityId;
    use cobre_sddp::{FphaHydroDetail, ProductionModelSource};

    fn make_hydro_model_summary_mixed() -> HydroModelSummary {
        HydroModelSummary {
            n_constant: 2,
            n_fpha: 2,
            total_planes: 10,
            fpha_details: vec![
                FphaHydroDetail {
                    hydro_id: EntityId(3),
                    name: "Hydro3".to_string(),
                    source: ProductionModelSource::PrecomputedHyperplanes,
                    n_planes: 5,
                },
                FphaHydroDetail {
                    hydro_id: EntityId(4),
                    name: "Hydro4".to_string(),
                    source: ProductionModelSource::PrecomputedHyperplanes,
                    n_planes: 5,
                },
            ],
            no_turbine_capacity: Vec::new(),
            n_evaporation: 3,
            n_no_evaporation: 1,
            n_user_supplied_ref: 0,
            n_default_midpoint_ref: 3,
        }
    }

    fn make_hydro_model_summary_all_constant() -> HydroModelSummary {
        HydroModelSummary {
            n_constant: 4,
            n_fpha: 0,
            total_planes: 0,
            fpha_details: vec![],
            no_turbine_capacity: Vec::new(),
            n_evaporation: 0,
            n_no_evaporation: 4,
            n_user_supplied_ref: 0,
            n_default_midpoint_ref: 0,
        }
    }

    fn make_hydro_model_summary_all_fpha() -> HydroModelSummary {
        HydroModelSummary {
            n_constant: 0,
            n_fpha: 165,
            total_planes: 825,
            fpha_details: vec![],
            no_turbine_capacity: Vec::new(),
            n_evaporation: 162,
            n_no_evaporation: 3,
            n_user_supplied_ref: 0,
            n_default_midpoint_ref: 162,
        }
    }

    /// Acceptance criterion: 2 FPHA hydros → output contains "2 FPHA" and "planes"
    /// and does NOT contain the relocated source qualifiers.
    #[test]
    fn format_hydro_model_summary_with_fpha_contains_key_terms() {
        let summary = make_hydro_model_summary_mixed();
        let s = hydro_model_summary_lines(&summary).join("\n");

        assert!(
            s.contains("2 FPHA"),
            "mixed summary must contain '2 FPHA', got: {s}"
        );
        assert!(
            s.contains("planes"),
            "mixed summary must contain 'planes', got: {s}"
        );
        assert!(
            !s.contains("loaded"),
            "production source qualifier 'loaded' must be relocated out of the display, got: {s}"
        );
        assert!(
            !s.contains("precomputed"),
            "production source qualifier 'precomputed' must be relocated out of the display, got: {s}"
        );
        assert!(
            !s.contains("computed from geometry"),
            "production source qualifier 'computed from geometry' must be relocated out of the display, got: {s}"
        );
    }

    /// Acceptance criterion: 0 FPHA hydros → output contains "constant" and NOT "FPHA".
    #[test]
    fn format_hydro_model_summary_without_fpha_contains_constant_not_fpha() {
        let summary = make_hydro_model_summary_all_constant();
        let s = hydro_model_summary_lines(&summary).join("\n");

        assert!(
            s.contains("constant"),
            "all-constant summary must contain 'constant', got: {s}"
        );
        assert!(
            !s.contains("FPHA"),
            "all-constant summary must NOT contain 'FPHA', got: {s}"
        );
    }

    /// AC: pure-FPHA (`n_constant = 0`) — output contains "2 FPHA" and "7 planes"
    /// and does NOT contain any relocated production-source qualifier.
    #[test]
    fn format_hydro_model_summary_pure_fpha_counts_only() {
        let summary = HydroModelSummary {
            n_constant: 0,
            n_fpha: 2,
            total_planes: 7,
            fpha_details: vec![
                FphaHydroDetail {
                    hydro_id: EntityId(1),
                    name: "Hydro1".to_string(),
                    source: ProductionModelSource::PrecomputedHyperplanes,
                    n_planes: 4,
                },
                FphaHydroDetail {
                    hydro_id: EntityId(2),
                    name: "Hydro2".to_string(),
                    source: ProductionModelSource::ComputedFromGeometry,
                    n_planes: 3,
                },
            ],
            no_turbine_capacity: Vec::new(),
            n_evaporation: 0,
            n_no_evaporation: 2,
            n_user_supplied_ref: 0,
            n_default_midpoint_ref: 0,
        };
        let s = hydro_model_summary_lines(&summary).join("\n");

        assert!(
            s.contains("2 FPHA"),
            "pure-fpha summary must contain '2 FPHA', got: {s}"
        );
        assert!(
            s.contains("7 planes"),
            "pure-fpha summary must contain '7 planes', got: {s}"
        );
        assert!(
            !s.contains("loaded"),
            "production source qualifier 'loaded' must be absent, got: {s}"
        );
        assert!(
            !s.contains("precomputed"),
            "production source qualifier 'precomputed' must be absent, got: {s}"
        );
        assert!(
            !s.contains("computed from geometry"),
            "production source qualifier 'computed from geometry' must be absent, got: {s}"
        );
    }

    /// Header line is always present.
    #[test]
    fn format_hydro_model_summary_contains_header() {
        let summary = make_hydro_model_summary_mixed();
        let s = hydro_model_summary_lines(&summary).join("\n");

        assert!(
            s.contains("Hydro models"),
            "summary must contain 'Hydro models' header, got: {s}"
        );
    }

    /// Mixed summary: production line shows plane count and "loaded".
    #[test]
    fn format_hydro_model_summary_mixed_production_line() {
        let summary = make_hydro_model_summary_mixed();
        let s = hydro_model_summary_lines(&summary).join("\n");

        assert!(
            s.contains("10"),
            "mixed summary must contain plane count '10', got: {s}"
        );
        assert!(
            s.contains("2 constant"),
            "mixed summary must contain '2 constant', got: {s}"
        );
    }

    /// All-FPHA large system: production line shows counts only, no source filename.
    #[test]
    fn format_hydro_model_summary_all_fpha_counts_only() {
        let summary = make_hydro_model_summary_all_fpha();
        let s = hydro_model_summary_lines(&summary).join("\n");

        assert!(
            s.contains("165 FPHA"),
            "all-fpha summary must contain '165 FPHA', got: {s}"
        );
        assert!(
            s.contains("825"),
            "all-fpha summary must contain '825' (plane count), got: {s}"
        );
        assert!(
            !s.contains("fpha_hyperplanes.parquet"),
            "production source filename must be relocated out of the display, got: {s}"
        );
        assert!(
            !s.contains("loaded"),
            "production source qualifier 'loaded' must be relocated out of the display, got: {s}"
        );
    }

    /// AC (C2): `n_evaporation=1, n_no_evaporation=1, n_user_supplied_ref=1` →
    /// output contains the literal `"1 linearized"` (count-only, no noun) and
    /// none of the relocated reference-source qualifiers.
    #[test]
    fn format_hydro_model_summary_evaporation_count_only_singular() {
        let summary = HydroModelSummary {
            n_constant: 0,
            n_fpha: 0,
            total_planes: 0,
            fpha_details: vec![],
            no_turbine_capacity: Vec::new(),
            n_evaporation: 1,
            n_no_evaporation: 1,
            n_user_supplied_ref: 1,
            n_default_midpoint_ref: 0,
        };
        let s = hydro_model_summary_lines(&summary).join("\n");

        assert!(
            s.contains("1 linearized"),
            "evaporation line must contain the literal '1 linearized', got: {s}"
        );
        assert!(
            !s.contains("v_ref"),
            "evaporation reference qualifier 'v_ref' must be absent, got: {s}"
        );
        assert!(
            !s.contains("user"),
            "evaporation reference qualifier 'user' must be absent, got: {s}"
        );
        assert!(
            !s.contains("midpoint"),
            "evaporation reference qualifier 'midpoint' must be absent, got: {s}"
        );
    }

    #[test]
    fn format_hydro_model_summary_plural_evaporation() {
        let summary = make_hydro_model_summary_mixed();
        let s = hydro_model_summary_lines(&summary).join("\n");

        assert!(
            s.contains("3 linearized"),
            "evaporation line must contain '3 linearized' (count-only), got: {s}"
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

    // ── format_evaporation_line count-only tests ─────────────────────────────

    /// AC: all-midpoint refs — line shows counts only, no `v_ref` qualifier.
    #[test]
    fn test_evaporation_line_all_midpoint() {
        let summary = HydroModelSummary {
            n_constant: 2,
            n_fpha: 0,
            total_planes: 0,
            fpha_details: vec![],
            no_turbine_capacity: Vec::new(),
            n_evaporation: 2,
            n_no_evaporation: 0,
            n_user_supplied_ref: 0,
            n_default_midpoint_ref: 2,
        };
        let s = hydro_model_summary_lines(&summary).join("\n");
        assert!(
            s.contains("2 linearized"),
            "all-midpoint must contain '2 linearized', got: {s}"
        );
        assert!(
            !s.contains("v_ref"),
            "evaporation reference qualifier 'v_ref' must be relocated out of the display, got: {s}"
        );
        assert!(
            !s.contains("midpoint"),
            "evaporation reference qualifier 'midpoint' must be relocated out of the display, got: {s}"
        );
    }

    /// AC: all-user-supplied refs — line shows counts only, no `v_ref` qualifier.
    #[test]
    fn test_evaporation_line_all_user_supplied() {
        let summary = HydroModelSummary {
            n_constant: 3,
            n_fpha: 0,
            total_planes: 0,
            fpha_details: vec![],
            no_turbine_capacity: Vec::new(),
            n_evaporation: 3,
            n_no_evaporation: 1,
            n_user_supplied_ref: 3,
            n_default_midpoint_ref: 0,
        };
        let s = hydro_model_summary_lines(&summary).join("\n");
        assert!(
            !s.contains("v_ref"),
            "evaporation reference qualifier 'v_ref' must be relocated out of the display, got: {s}"
        );
        assert!(
            !s.contains("user"),
            "evaporation reference qualifier 'user' must be relocated out of the display, got: {s}"
        );
        assert!(
            s.contains("3 linearized"),
            "all-user-supplied must contain '3 linearized', got: {s}"
        );
        assert!(
            s.contains("1 without"),
            "all-user-supplied must contain '1 without', got: {s}"
        );
    }

    /// AC: mixed refs — line shows counts only, no `v_ref` qualifiers.
    #[test]
    fn test_evaporation_line_mixed() {
        let summary = HydroModelSummary {
            n_constant: 3,
            n_fpha: 0,
            total_planes: 0,
            fpha_details: vec![],
            no_turbine_capacity: Vec::new(),
            n_evaporation: 3,
            n_no_evaporation: 1,
            n_user_supplied_ref: 2,
            n_default_midpoint_ref: 1,
        };
        let s = hydro_model_summary_lines(&summary).join("\n");
        assert!(
            !s.contains("v_ref"),
            "evaporation reference qualifier 'v_ref' must be relocated out of the display, got: {s}"
        );
        assert!(
            !s.contains("user"),
            "evaporation reference qualifier 'user' must be relocated out of the display, got: {s}"
        );
        assert!(
            !s.contains("midpoint"),
            "evaporation reference qualifier 'midpoint' must be relocated out of the display, got: {s}"
        );
        assert!(
            s.contains("3 linearized"),
            "mixed must contain '3 linearized', got: {s}"
        );
    }

    /// AC: no evaporation — line does NOT contain "`v_ref`".
    #[test]
    fn test_evaporation_line_no_evaporation() {
        let summary = make_hydro_model_summary_all_constant();
        let s = hydro_model_summary_lines(&summary).join("\n");
        assert!(
            !s.contains("v_ref"),
            "zero-evaporation must NOT contain 'v_ref', got: {s}"
        );
        assert!(
            s.contains("0 linearized"),
            "zero-evaporation must contain '0 linearized', got: {s}"
        );
    }

    // ── SetupTimings tests ─────────────────────────────────────────────────

    use super::{SetupTimings, print_setup_summary, setup_summary_lines};

    fn make_setup_timings() -> SetupTimings {
        SetupTimings {
            load_seconds: 1.2,
            stochastic_fit_seconds: 0.5,
            production_fit_seconds: 2.0,
            evaporation_fit_seconds: 0.0,
            broadcast_seconds: 0.1,
        }
    }

    /// AC: the rendered body carries the header and every phase label.
    #[test]
    fn format_setup_summary_contains_phase_labels() {
        let timings = make_setup_timings();
        let s = setup_summary_lines(&timings).join("\n");

        assert!(
            s.contains("Setup"),
            "output must contain 'Setup' header, got: {s}"
        );
        assert!(
            s.contains("Load"),
            "output must contain 'Load' label, got: {s}"
        );
        assert!(
            s.contains("Stochastic fit"),
            "output must contain 'Stochastic fit' label, got: {s}"
        );
        assert!(
            s.contains("Production fit"),
            "output must contain 'Production fit' label, got: {s}"
        );
        assert!(
            s.contains("Evaporation fit"),
            "output must contain 'Evaporation fit' label, got: {s}"
        );
        assert!(
            s.contains("Broadcast"),
            "output must contain 'Broadcast' label, got: {s}"
        );
    }

    #[test]
    fn print_setup_summary_does_not_panic() {
        let timings = make_setup_timings();
        print_setup_summary(&Term::buffered_stderr(), &timings);
    }

    // ── ModelProvenanceReport tests ───────────────────────────────────────────

    use cobre_sddp::{HydroProductionProvenance, InflowProvenance, ProvenanceSource};

    use super::{ModelProvenanceReport, print_provenance_summary, provenance_summary_lines};

    fn make_provenance_report_full_estimation() -> ModelProvenanceReport {
        ModelProvenanceReport {
            inflow: InflowProvenance {
                estimation_path: "full_estimation".to_string(),
                seasonal_stats_source: ProvenanceSource::Estimated,
                ar_coefficients_source: ProvenanceSource::Estimated,
                correlation_source: ProvenanceSource::Estimated,
                opening_tree_source: ProvenanceSource::Estimated,
                n_hydros: 3,
                ar_method: Some("PACF".to_string()),
                ar_max_order: Some(6),
                white_noise_fallbacks: vec![],
                historical_library_seed_digest: None,
            },
            hydro_production: HydroProductionProvenance::default(),
        }
    }

    fn make_provenance_report_deterministic() -> ModelProvenanceReport {
        ModelProvenanceReport {
            inflow: InflowProvenance {
                estimation_path: "deterministic".to_string(),
                seasonal_stats_source: ProvenanceSource::NotApplicable,
                ar_coefficients_source: ProvenanceSource::NotApplicable,
                correlation_source: ProvenanceSource::NotApplicable,
                opening_tree_source: ProvenanceSource::NotApplicable,
                n_hydros: 0,
                ar_method: None,
                ar_max_order: None,
                white_noise_fallbacks: vec![],
                historical_library_seed_digest: None,
            },
            hydro_production: HydroProductionProvenance::default(),
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
        let s = provenance_summary_lines(&report).join("\n");
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
        let s = provenance_summary_lines(&report).join("\n");
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
        let s = provenance_summary_lines(&report).join("\n");
        assert!(
            s.contains("deterministic"),
            "output must contain 'deterministic' estimation path, got: {s}"
        );
        assert!(
            s.contains("n/a"),
            "output must contain 'n/a' for NotApplicable sources, got: {s}"
        );
        assert!(
            !s.contains("max order"),
            "output must NOT contain 'max order' for deterministic case, got: {s}"
        );
    }

    #[test]
    fn format_provenance_summary_user_file_source() {
        let report = ModelProvenanceReport {
            inflow: InflowProvenance {
                estimation_path: "user_provided_no_history".to_string(),
                seasonal_stats_source: ProvenanceSource::UserFile,
                ar_coefficients_source: ProvenanceSource::UserFile,
                correlation_source: ProvenanceSource::Estimated,
                opening_tree_source: ProvenanceSource::Estimated,
                n_hydros: 2,
                ar_method: None,
                ar_max_order: None,
                white_noise_fallbacks: vec![],
                historical_library_seed_digest: None,
            },
            hydro_production: HydroProductionProvenance::default(),
        };
        let s = provenance_summary_lines(&report).join("\n");
        assert!(
            s.contains("user_file"),
            "output must contain 'user_file' for UserFile source, got: {s}"
        );
        assert!(
            !s.contains("max order"),
            "output must NOT contain 'max order' when ar_method is None, got: {s}"
        );
    }

    fn make_provenance_report_with_hydro_production(
        hydro_production: HydroProductionProvenance,
    ) -> ModelProvenanceReport {
        ModelProvenanceReport {
            inflow: InflowProvenance {
                estimation_path: "full_estimation".to_string(),
                seasonal_stats_source: ProvenanceSource::Estimated,
                ar_coefficients_source: ProvenanceSource::Estimated,
                correlation_source: ProvenanceSource::Estimated,
                opening_tree_source: ProvenanceSource::Estimated,
                n_hydros: 3,
                ar_method: Some("PACF".to_string()),
                ar_max_order: Some(6),
                white_noise_fallbacks: vec![],
                historical_library_seed_digest: None,
            },
            hydro_production,
        }
    }

    /// The provenance display shows only the inflow lines; the hydro-production
    /// source counts are persisted in `model_provenance.json` but never rendered
    /// in the human summary, even when non-zero.
    #[test]
    fn format_provenance_summary_omits_hydro_production_section() {
        let report = make_provenance_report_with_hydro_production(HydroProductionProvenance {
            n_fpha_computed_from_geometry: 2,
            n_fpha_precomputed_hyperplanes: 1,
            n_evaporation_ref_user_supplied: 4,
            n_evaporation_ref_default_midpoint: 3,
        });
        let s = provenance_summary_lines(&report).join("\n");
        assert!(s.contains("Model provenance"), "got: {s}");
        assert!(s.contains("Estimation path:"), "got: {s}");
        assert!(
            !s.contains("Hydro production"),
            "hydro-production sub-section must not be rendered, got: {s}"
        );
        assert!(!s.contains("FPHA planes:"), "got: {s}");
        assert!(!s.contains("Evaporation ref:"), "got: {s}");
    }

    #[test]
    fn print_provenance_summary_with_hydro_counts_does_not_panic() {
        let report = make_provenance_report_with_hydro_production(HydroProductionProvenance {
            n_fpha_computed_from_geometry: 5,
            n_fpha_precomputed_hyperplanes: 2,
            n_evaporation_ref_user_supplied: 4,
            n_evaporation_ref_default_midpoint: 1,
        });
        print_provenance_summary(&Term::buffered_stderr(), &report);
    }

    // ── BoundaryReconciliationReport / boundary summary tests ───────────────

    use std::path::Path;

    use chrono::NaiveDate;

    use super::{BoundaryReconciliationReport, boundary_summary_lines, print_boundary_summary};
    use cobre_sddp::{AnticipatedCoverage, FamilyTally};

    fn make_reconciled_boundary_report() -> BoundaryReconciliationReport {
        BoundaryReconciliationReport {
            reconciled: true,
            storage: FamilyTally {
                copy: 2184,
                ..FamilyTally::default()
            },
            inflow_lag: FamilyTally {
                fan_out: 1,
                ..FamilyTally::default()
            },
            transit_bucket: FamilyTally {
                dropped_source: 2,
                ..FamilyTally::default()
            },
            anticipated: FamilyTally {
                default_zero: 1,
                ..FamilyTally::default()
            },
            anticipated_coverage: AnticipatedCoverage::default(),
            other_identity: FamilyTally::default(),
            dropped_source_slots: Vec::new(),
            straddling_slots: Vec::new(),
        }
    }

    #[test]
    fn format_boundary_summary_reconciled_renders_house_style() {
        let report = make_reconciled_boundary_report();
        let boundary_date = NaiveDate::from_ymd_opt(2031, 12, 1).unwrap();
        let s = boundary_summary_lines(10_000, boundary_date, Path::new("/case/boundary"), &report)
            .join("\n");

        assert!(
            s.contains("Boundary policy"),
            "output must contain the 'Boundary policy' header, got: {s}"
        );
        assert!(
            s.contains("Cuts loaded:    10000 (priced at 2031-12-01)"),
            "output must contain the aligned 'Cuts loaded:' row, got: {s}"
        );
        assert!(
            s.contains("Source:         /case/boundary"),
            "output must contain the aligned 'Source:' row, got: {s}"
        );
        assert!(
            s.contains("Reconciliation: 2184 copied, 1 fanned out, 1 defaulted to 0.0, 2 dropped"),
            "output must contain the compact reconciliation tally, got: {s}"
        );
    }

    #[test]
    fn format_boundary_summary_dimension_only_renders_notice() {
        let report = BoundaryReconciliationReport::default();
        let boundary_date = NaiveDate::from_ymd_opt(2030, 2, 1).unwrap();
        let s = boundary_summary_lines(500, boundary_date, Path::new("/case/boundary"), &report)
            .join("\n");

        assert!(
            s.contains("Reconciliation: dimension-only load (entity manifest absent)"),
            "output must render the dimension-only notice, got: {s}"
        );
    }

    #[test]
    fn print_boundary_summary_does_not_panic() {
        let report = make_reconciled_boundary_report();
        let boundary_date = NaiveDate::from_ymd_opt(2031, 12, 1).unwrap();
        print_boundary_summary(
            &Term::buffered_stderr(),
            10_000,
            boundary_date,
            Path::new("/case/boundary"),
            &report,
        );

        let s = boundary_summary_lines(10_000, boundary_date, Path::new("/case/boundary"), &report)
            .join("\n");
        assert!(
            s.contains("2031-12-01"),
            "the rendered line must contain the ISO date: {s}"
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
