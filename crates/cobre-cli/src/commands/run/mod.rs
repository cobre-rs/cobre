//! `cobre run <CASE_DIR>` subcommand: load, train the SDDP policy, optionally
//! simulate, and write outputs.

use cobre_core::System;
use cobre_io::Config;
use cobre_io::DistributionInfo;
use cobre_io::HostLayout;
use cobre_io::OutputContext;
use cobre_io::PolicyMode;
use cobre_io::SetupTimings;
use cobre_io::now_iso8601;
use cobre_sddp::SolverStatsDelta;
use cobre_sddp::StudySetup;
use cobre_sddp::build_deviation_summary;
use cobre_sddp::setup::PostTrainingSimulation;
use cobre_sddp::setup::RunPhasePlan;
use cobre_sddp::setup::signal_stop_requested;
use cobre_solver::active_solver_metadata_id;

use crate::progress::RenderMode;
mod graceful_stop;
mod outputs;
mod policy;
mod setup;
mod simulation;
mod training;

use std::path::PathBuf;

use clap::{Args, ValueEnum};
use console::Term;

use cobre_comm::{BackendKind, Communicator, ExecutionTopology};

use crate::error::CliError;

/// Communication backend selected by `--comm-backend`.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum CommBackendArg {
    /// Auto-detect (default): the MPI backend when launched under an MPI
    /// launcher (`mpiexec`/`mpirun`/`srun`), otherwise the local backend.
    #[default]
    Auto,
    /// Single-process local backend.
    Local,
    /// MPI backend; requires the binary to be built with the `mpi` feature.
    Mpi,
}

impl From<CommBackendArg> for BackendKind {
    fn from(arg: CommBackendArg) -> Self {
        match arg {
            CommBackendArg::Auto => BackendKind::Auto,
            CommBackendArg::Local => BackendKind::Local,
            CommBackendArg::Mpi => BackendKind::Mpi,
        }
    }
}

use graceful_stop::{SignalWindow, agree_post_write, failure_code, into_agreed_result};
use outputs::{WriteTrainingArgs, write_training_outputs};
use policy::{apply_training_policy, load_policy_for_simulation};
use setup::{LoadBroadcastResult, broadcast_and_build_setup, run_pre_training, setup_communicator};
use simulation::{run_simulation_phase, skip_simulation_phase};
use training::run_training_phase;

/// Arguments for the `cobre run` subcommand.
#[derive(Debug, Args)]
#[command(about = "Load a case directory, train an SDDP policy, and run simulation")]
pub struct RunArgs {
    /// Path to the case directory containing the input data files.
    pub case_dir: PathBuf,

    /// Output directory for results (defaults to `<CASE_DIR>/output/`).
    #[arg(long, value_name = "DIR")]
    pub output: Option<PathBuf>,

    /// Suppress the banner and progress bars. Errors still go to stderr.
    #[arg(long)]
    pub quiet: bool,

    /// Worker threads per MPI rank for parallel scenario processing.
    /// Defaults to 1 when omitted.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub threads: Option<u32>,

    /// Communication backend. `auto` (default) selects `mpi` when launched under
    /// an MPI launcher (`mpiexec`/`mpirun`/`srun`) and `local` otherwise; `local`
    /// forces single-process; `mpi` forces the MPI backend (requires the binary
    /// to be built with the `mpi` feature and launched under an MPI launcher).
    #[arg(long, value_enum, default_value_t = CommBackendArg::Auto)]
    pub comm_backend: CommBackendArg,
}

/// Shared context for execute phases (communicator, output, topology, etc.).
pub(super) struct RunContext<C: Communicator> {
    pub(super) comm: C,
    pub(super) is_root: bool,
    pub(super) quiet: bool,
    pub(super) n_threads: usize,
    pub(super) output_dir: PathBuf,
    /// Input root every path resolves against, never the output directory.
    pub(super) case_dir: PathBuf,
    pub(super) term_width: u16,
    pub(super) stderr: Term,
    /// Progress strategy: non-TTY stderr gets append-only lines, not bars.
    pub(super) render_mode: RenderMode,
    pub(super) topology: ExecutionTopology,
    pub(super) solver_version: String,
}

/// How a `cobre run` that returned no error ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum RunOutcome {
    /// The run finished every configured phase.
    Completed,
    /// A signal stopped training, and the training outputs and policy checkpoint were written.
    StoppedOnRequest,
}

impl RunOutcome {
    /// The process exit code for this outcome.
    #[must_use]
    pub const fn exit_code(self) -> i32 {
        match self {
            Self::Completed => 0,
            Self::StoppedOnRequest => 5,
        }
    }
}

/// Execute the `run` subcommand (load, train, optionally simulate, write outputs).
///
/// # Errors
///
/// Returns [`CliError`] when loading, training, simulation, or I/O fails.
pub fn execute(args: &RunArgs) -> Result<RunOutcome, CliError> {
    let ctx = setup_communicator(args)?;
    let result = graceful_stop::install(ctx.comm.size() == 1)
        .and_then(|signals| execute_inner(&ctx, args, signals));
    if let Err(ref e) = result
        && ctx.comm.size() > 1
    {
        // `abort` diverges, so `main` never reaches its own `format_error`:
        // render here or the failing rank's diagnostic is lost to the launcher.
        let _ = ctx.stderr.write_line(&format!("rank {}:", ctx.comm.rank()));
        e.format_error(&ctx.stderr);
        ctx.comm.abort(e.exit_code());
    }
    result
}

fn execute_inner<C: Communicator>(
    ctx: &RunContext<C>,
    args: &RunArgs,
    signals: &SignalWindow,
) -> Result<RunOutcome, CliError> {
    let LoadBroadcastResult {
        system,
        mut setup,
        root_config,
        root_estimation_report,
        root_estimation_path,
        training_enabled,
        policy_mode,
        setup_timings,
    } = broadcast_and_build_setup(ctx, args)?;

    run_pre_training(
        ctx,
        &system,
        &setup,
        root_config.as_ref(),
        root_estimation_report.as_ref(),
        root_estimation_path,
        setup_timings.as_ref(),
    )?;

    let hostname = ctx.topology.leader_hostname().to_string();

    match RunPhasePlan::resolve(training_enabled, setup.simulation_config.n_scenarios > 0) {
        RunPhasePlan::TrainedThenSimulated => train_then_simulate(
            ctx,
            signals,
            &system,
            &mut setup,
            root_config,
            policy_mode,
            setup_timings,
            &hostname,
        ),
        RunPhasePlan::SimulateFromPolicy => {
            let training_result = load_policy_for_simulation(ctx, &system, &mut setup)?;
            run_simulation_phase(ctx, &system, &mut setup, &training_result, &hostname)?;
            Ok(RunOutcome::Completed)
        }
        RunPhasePlan::Nothing => {
            if ctx.is_root && !ctx.quiet {
                let _ = ctx
                    .stderr
                    .write_line("Training disabled, simulation disabled — nothing to do.");
            }
            Ok(RunOutcome::Completed)
        }
    }
}

fn train_then_simulate<C: Communicator>(
    ctx: &RunContext<C>,
    signals: &SignalWindow,
    system: &System,
    setup: &mut StudySetup,
    mut root_config: Option<Config>,
    policy_mode: PolicyMode,
    setup_timings: Option<SetupTimings>,
    hostname: &str,
) -> Result<RunOutcome, CliError> {
    apply_training_policy(ctx, system, setup, root_config.as_ref(), policy_mode)?;
    setup.enable_periodic_checkpoints(system, &ctx.output_dir);
    let training_started_at = now_iso8601();
    signals.open();
    let training = run_training_phase(ctx, setup, signals.shutdown_flag())?;
    let training_completed_at = now_iso8601();

    let decision = &training.result.stop_decision;
    let n_scenarios = setup.simulation_config.n_scenarios;
    let decided_skip = training.error.is_none()
        && PostTrainingSimulation::resolve(n_scenarios > 0, decision, 0)
            == PostTrainingSimulation::SkipAfterSignalStop;

    let mpi_world_size = u32::try_from(ctx.topology.world_size).unwrap_or(u32::MAX);
    let mut local = if ctx.is_root {
        root_config
            .take()
            .ok_or_else(|| CliError::Internal {
                message: "root_config was None on rank 0 — internal invariant violated".to_string(),
            })
            .and_then(|config| {
                let training_ctx = OutputContext {
                    hostname: hostname.to_string(),
                    solver: active_solver_metadata_id().to_string(),
                    solver_version: Some(ctx.solver_version.clone()),
                    started_at: training_started_at,
                    completed_at: training_completed_at,
                    distribution: build_distribution_info(
                        &ctx.topology,
                        ctx.n_threads,
                        mpi_world_size,
                    ),
                    setup: setup_timings,
                    production_fit_deviation: build_deviation_summary(
                        &setup.hydro_models.fpha_fit_deviations,
                    ),
                };
                write_training_outputs(&WriteTrainingArgs {
                    output_dir: &ctx.output_dir,
                    system,
                    config: &config,
                    training_output: &training.output,
                    setup,
                    training_result: &training.result,
                    output_ctx: &training_ctx,
                    quiet: ctx.quiet,
                    stderr: &ctx.stderr,
                })
            })
    } else {
        Ok(())
    };
    let level = signals.sample_level();
    // Every rank-0 write of the stop path precedes the agreement, with no
    // early return: a peer would otherwise wait in the allreduce, or exit
    // while rank 0 still writes.
    if decided_skip && ctx.is_root && local.is_ok() {
        local = skip_simulation_phase(ctx, hostname, n_scenarios);
    }
    let agreed = agree_post_write(&ctx.comm, failure_code(&local), level)?;
    into_agreed_result(local, &agreed)?;

    if let Some(training_error) = training.error {
        if ctx.is_root {
            tracing::error!(
                "training failed after {} iterations: {training_error}",
                training.result.iterations
            );
            if !ctx.quiet {
                let _ = ctx.stderr.write_line(&format!(
                    "Training failed after {} iterations. Partial outputs written to {}.",
                    training.result.iterations,
                    ctx.output_dir.display()
                ));
            }
        }
        return Err(CliError::from(training_error));
    }

    let signal_stop = signal_stop_requested(decision, agreed.level);
    if !signal_stop {
        signals.close()?;
    }

    match PostTrainingSimulation::resolve(n_scenarios > 0, decision, agreed.level) {
        PostTrainingSimulation::Run => {
            run_simulation_phase(ctx, system, setup, &training.result, hostname)?;
        }
        PostTrainingSimulation::SkipAfterSignalStop if !decided_skip => {
            let local = if ctx.is_root {
                skip_simulation_phase(ctx, hostname, n_scenarios)
            } else {
                Ok(())
            };
            let late = agree_post_write(&ctx.comm, failure_code(&local), agreed.level)?;
            into_agreed_result(local, &late)?;
        }
        PostTrainingSimulation::SkipAfterSignalStop | PostTrainingSimulation::NotRequested => {}
    }

    Ok(if signal_stop {
        RunOutcome::StoppedOnRequest
    } else {
        RunOutcome::Completed
    })
}

/// Guard `u64 as f64` cast before MPI `allreduce(Sum)`: reject counters ≥ `2^53` that lose precision.
///
/// # Errors
///
/// Returns [`CliError`] when any guarded counter exceeds `2^53`.
pub(super) fn check_stats_overflow(delta: &SolverStatsDelta) -> Result<(), CliError> {
    const F64_INTEGER_LIMIT: u64 = 1u64 << 53;
    for (label, value) in [
        ("lp_solves", delta.lp_solves),
        ("lp_successes", delta.lp_successes),
        ("first_try_successes", delta.first_try_successes),
        ("lp_failures", delta.lp_failures),
        ("retry_attempts", delta.retry_attempts),
        ("basis_offered", delta.basis_offered),
        (
            "basis_consistency_failures",
            delta.basis_consistency_failures,
        ),
        ("simplex_iterations", delta.simplex_iterations),
        ("load_model_count", delta.load_model_count),
    ] {
        if value > F64_INTEGER_LIMIT {
            return Err(CliError::Internal {
                message: format!(
                    "solver stats counter '{label}' = {value} \
                     exceeds 2^53 (f64 integer-precision limit). MPI \
                     allreduce(Sum) packing would lose precision. \
                     Reduce iteration count or split the run."
                ),
            });
        }
    }
    Ok(())
}

/// Build a [`cobre_io::DistributionInfo`] from the cached execution topology.
pub(super) fn build_distribution_info(
    topology: &ExecutionTopology,
    n_threads: usize,
    ranks_participated: u32,
) -> DistributionInfo {
    DistributionInfo {
        backend: match topology.backend {
            BackendKind::Mpi => "mpi",
            BackendKind::Local => "local",
            BackendKind::Auto => "unknown",
        }
        .to_string(),
        world_size: u32::try_from(topology.world_size).unwrap_or(u32::MAX),
        ranks_participated,
        num_hosts: u32::try_from(topology.num_hosts()).unwrap_or(u32::MAX),
        threads_per_rank: u32::try_from(n_threads).unwrap_or(u32::MAX),
        mpi_library: topology.mpi.as_ref().map(|m| m.library_version.clone()),
        mpi_standard: topology.mpi.as_ref().map(|m| m.standard_version.clone()),
        thread_level: topology.mpi.as_ref().map(|m| m.thread_level.clone()),
        slurm_job_id: topology.slurm.as_ref().map(|s| s.job_id.clone()),
        hosts: host_layouts(topology),
    }
}

pub(super) fn compute_parallelism(n_threads: usize, comm_size: usize) -> u32 {
    u32::try_from(n_threads)
        .unwrap_or(u32::MAX)
        .saturating_mul(u32::try_from(comm_size).unwrap_or(u32::MAX))
}

fn host_layouts(topology: &ExecutionTopology) -> Vec<HostLayout> {
    topology
        .hosts
        .iter()
        .map(|h| HostLayout {
            hostname: h.hostname.clone(),
            ranks: h
                .ranks
                .iter()
                .map(|&r| u32::try_from(r).unwrap_or(u32::MAX))
                .collect(),
        })
        .collect()
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic
)]
mod tests {
    use super::setup::resolve_thread_count;
    use super::{RunOutcome, check_stats_overflow, host_layouts};
    use cobre_comm::{BackendKind, ExecutionTopology, HostInfo};
    use cobre_sddp::{SolverStatsDelta, delta_to_stats_row};

    fn topology_with_hosts(hosts: Vec<HostInfo>) -> ExecutionTopology {
        let world_size = hosts.iter().map(|h| h.ranks.len()).sum();
        ExecutionTopology {
            backend: BackendKind::Local,
            world_size,
            hosts,
            mpi: None,
            slurm: None,
        }
    }

    #[test]
    fn test_host_layouts_two_hosts_preserve_order_and_u32_ranks() {
        let topology = topology_with_hosts(vec![
            HostInfo {
                hostname: "node-a".to_string(),
                ranks: vec![0, 1],
            },
            HostInfo {
                hostname: "node-b".to_string(),
                ranks: vec![2, 3],
            },
        ]);
        let layouts = host_layouts(&topology);
        assert_eq!(layouts.len(), 2);
        assert_eq!(layouts[0].hostname, "node-a");
        assert_eq!(layouts[0].ranks, vec![0u32, 1u32]);
        assert_eq!(layouts[1].hostname, "node-b");
        assert_eq!(layouts[1].ranks, vec![2u32, 3u32]);
    }

    #[test]
    fn test_host_layouts_single_host_ranks_is_zero() {
        let topology = topology_with_hosts(vec![HostInfo {
            hostname: "localhost".to_string(),
            ranks: vec![0],
        }]);
        let layouts = host_layouts(&topology);
        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[0].hostname, "localhost");
        assert_eq!(layouts[0].ranks, vec![0u32]);
    }

    fn make_delta(lp_solves: u64) -> SolverStatsDelta {
        SolverStatsDelta {
            lp_solves,
            ..SolverStatsDelta::default()
        }
    }

    #[test]
    fn run_outcome_maps_to_its_exit_code() {
        assert_eq!(RunOutcome::Completed.exit_code(), 0);
        assert_eq!(RunOutcome::StoppedOnRequest.exit_code(), 5);
    }

    #[test]
    fn test_resolve_thread_count_cli_value() {
        assert_eq!(resolve_thread_count(Some(4)), 4);
    }

    #[test]
    fn test_resolve_thread_count_default() {
        assert_eq!(resolve_thread_count(None), 1);
        assert_eq!(resolve_thread_count(Some(1)), 1);
    }

    #[test]
    fn test_delta_to_stats_row_backward_carries_opening_rank_worker() {
        let delta = make_delta(10);
        let row = delta_to_stats_row(
            Some(1),
            None,
            "backward",
            Some(2),
            Some(0),
            Some(1),
            Some(3),
            &delta,
        );
        assert_eq!(row.opening_index, Some(0));
        assert_eq!(row.rank, Some(1));
        assert_eq!(row.worker_id, Some(3));
        assert_eq!(row.stage_id, Some(2));
        assert_eq!(row.phase, "backward");
        assert_eq!(row.lp_solves, 10);
    }

    #[test]
    fn test_delta_to_stats_row_forward_opening_and_worker_id_are_none() {
        // Forward rows carry the real (domain) stage_id, not the -1 sentinel.
        let delta = make_delta(4);
        let row = delta_to_stats_row(
            Some(1),
            None,
            "forward",
            Some(0),
            None,
            Some(0),
            None,
            &delta,
        );
        assert_eq!(row.opening_index, None);
        assert_eq!(row.rank, Some(0));
        assert_eq!(row.worker_id, None);
        assert_eq!(row.stage_id, Some(0));
        assert_eq!(row.lp_solves, 4);
    }

    #[test]
    fn test_delta_to_stats_row_simulation_rank_and_worker_id_are_none() {
        let delta = make_delta(7);
        let row = delta_to_stats_row(None, Some(42), "simulation", None, None, None, None, &delta);
        assert_eq!(row.opening_index, None);
        assert_eq!(row.rank, None);
        assert_eq!(row.worker_id, None);
        assert_eq!(row.scenario_id, Some(42));
        assert_eq!(row.iteration, None);
    }

    // ── overflow guard tests ──────────────────────────────────────────────────

    fn delta_with_field(field: &str, value: u64) -> SolverStatsDelta {
        let mut d = SolverStatsDelta::default();
        match field {
            "lp_solves" => d.lp_solves = value,
            "lp_successes" => d.lp_successes = value,
            "first_try_successes" => d.first_try_successes = value,
            "lp_failures" => d.lp_failures = value,
            "retry_attempts" => d.retry_attempts = value,
            "basis_offered" => d.basis_offered = value,
            "basis_consistency_failures" => d.basis_consistency_failures = value,
            "simplex_iterations" => d.simplex_iterations = value,
            "load_model_count" => d.load_model_count = value,
            other => panic!("unknown field: {other}"),
        }
        d
    }

    #[test]
    fn test_overflow_guard_rejects_excessive_counter() {
        let over_limit = (1u64 << 53) + 1;

        let fields = [
            "lp_solves",
            "lp_successes",
            "first_try_successes",
            "lp_failures",
            "retry_attempts",
            "basis_offered",
            "basis_consistency_failures",
            "simplex_iterations",
            "load_model_count",
        ];

        for field in fields {
            let delta = delta_with_field(field, over_limit);
            let err = check_stats_overflow(&delta).unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("exceeds 2^53"),
                "field '{field}': message was: {msg}"
            );
            assert!(
                msg.contains(field),
                "field '{field}': label missing in message: {msg}"
            );
        }
    }

    /// `2^53` is the largest integer exactly representable as `f64`, so it must
    /// NOT trip the guard.
    #[test]
    fn test_overflow_guard_allows_exact_limit() {
        let at_limit = 1u64 << 53;
        let delta = SolverStatsDelta {
            lp_solves: at_limit,
            lp_successes: at_limit,
            first_try_successes: at_limit,
            lp_failures: at_limit,
            retry_attempts: at_limit,
            basis_offered: at_limit,
            basis_consistency_failures: at_limit,
            simplex_iterations: at_limit,
            load_model_count: at_limit,
            ..SolverStatsDelta::default()
        };
        check_stats_overflow(&delta)
            .expect("2^53 is representable in f64 and must not trigger the guard");
    }
}
