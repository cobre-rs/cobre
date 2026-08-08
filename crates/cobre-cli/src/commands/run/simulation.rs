//! Simulation phase for `cobre run`.

use std::sync::mpsc;

use console::Term;

use cobre_comm::{Communicator, ReduceOp};
use cobre_core::{System, TrainingEvent};
use cobre_io::MetadataCost;
use cobre_io::MetadataSimulationSolveStats;
use cobre_io::OutputContext;
use cobre_io::ParquetWriterConfig;
use cobre_io::SimulationOutput;
use cobre_io::now_iso8601;
use cobre_io::output::simulation_writer::ScenarioWritePayload;
use cobre_io::output::simulation_writer::SimulationParquetWriter;
use cobre_sddp::SOLVER_STATS_DELTA_SCALAR_FIELDS;
use cobre_sddp::SolverStatsDelta;
use cobre_sddp::StudySetup;
use cobre_sddp::TrainingResult;
use cobre_sddp::aggregate_simulation;
use cobre_sddp::pack_delta_scalars;
use cobre_sddp::pack_scenario_stats;
use cobre_sddp::reconcile_global_ok;
use cobre_sddp::unpack_delta_scalars;
use cobre_sddp::unpack_scenario_stats;
use cobre_solver::ActiveSolver;
use cobre_solver::active_solver_metadata_id;

use crate::error::CliError;
use crate::summary::SimulationSummary;

use super::outputs::{WriteSimulationArgs, write_simulation_outputs};
use super::{RunContext, build_distribution_info, check_stats_overflow};
use crate::progress::run_progress_thread;
use crate::summary::print_simulation_summary;

/// Run the simulation phase: workspace pool, Parquet writing, and output.
pub(super) fn run_simulation_phase(
    ctx: &RunContext<impl Communicator>,
    system: &System,
    setup: &mut StudySetup,
    training_result: &TrainingResult,
    hostname: &str,
) -> Result<(), CliError> {
    let solver_factory = ActiveSolver::new;
    let n_scenarios = setup.simulation_config.n_scenarios;
    let sim_config = setup.simulation_config();

    let mut sim_pool = setup
        .create_workspace_pool(&ctx.comm, ctx.n_threads, solver_factory)
        .map_err(|e| CliError::Solver {
            message: format!(
                "{} initialisation failed for simulation pool: {e}",
                cobre_solver::active_solver_name()
            ),
        })?;

    let (sim_event_tx, sim_event_rx) = mpsc::channel::<TrainingEvent>();
    let sim_progress_handle = if ctx.quiet {
        drop(sim_event_rx);
        None
    } else {
        Some(run_progress_thread(
            sim_event_rx,
            ctx.render_mode,
            u64::from(n_scenarios),
            ctx.term_width,
        ))
    };

    let io_capacity = sim_config.io_channel_capacity;
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity.max(1));

    let parquet_config = ParquetWriterConfig::default();
    let mut sim_writer = SimulationParquetWriter::new(&ctx.output_dir, system, &parquet_config)
        .map_err(CliError::from)?;

    // Drain straight to Parquet rather than collecting into a Vec and gathering
    // on rank 0 via MPI, which overflows i32 on large cases.
    let drain_handle = std::thread::spawn(move || {
        let mut failed: u32 = 0;
        for scenario_result in result_rx {
            let payload = ScenarioWritePayload::from(scenario_result);
            if let Err(e) = sim_writer.write_scenario(payload) {
                tracing::error!("simulation write error: {e}");
                failed += 1;
            }
        }
        (sim_writer, failed)
    });

    let sim_started_at = now_iso8601();
    let sim_start = std::time::Instant::now();

    let sim_result = setup
        .simulate(
            &mut sim_pool.workspaces,
            &ctx.comm,
            &result_tx,
            Some(sim_event_tx),
            training_result.frozen_templates.as_deref(),
            &training_result.basis_cache,
        )
        .map_err(CliError::from);
    if let Some(handle) = sim_progress_handle {
        let _ = handle.join();
    }

    drop(result_tx);

    let drain_join = drain_handle.join();

    // Reconcile the per-rank simulation outcome BEFORE the first post-sim
    // collective (`merge_simulation_metadata`'s allreduce): `simulate()` is
    // collective-free, so a failure on a strict subset of ranks would otherwise
    // strand every healthy rank in that allreduce while the failing ranks skip it.
    let mut reconcile_scratch = [0_i32];
    let local_ok = drain_join.is_ok() && sim_result.is_ok();
    let global_ok =
        reconcile_global_ok(local_ok, &ctx.comm, &mut reconcile_scratch).map_err(|e| {
            CliError::Internal {
                message: format!("simulation outcome reconcile error: {e}"),
            }
        })?;

    let (sim_writer, write_failures) = drain_join.map_err(|_| CliError::Internal {
        message: "simulation drain thread panicked".to_string(),
    })?;
    let sim_run_result = sim_result?;
    if !global_ok {
        return Err(CliError::Internal {
            message: "a peer rank failed simulation; failing on every rank in lockstep".to_string(),
        });
    }

    #[allow(clippy::cast_possible_truncation)]
    let sim_time_ms = sim_start.elapsed().as_millis() as u64;

    let mut local_sim_output = sim_writer.finalize(sim_time_ms);
    local_sim_output.failed = write_failures;

    let mut merged_sim_output = merge_simulation_metadata(&ctx.comm, &local_sim_output)?;

    ctx.comm.barrier().map_err(|e| CliError::Internal {
        message: format!("post-simulation barrier error: {e}"),
    })?;

    let (global_agg, global_scenario_stats) =
        aggregate_simulation_solver_stats(&ctx.comm, &sim_run_result.solver_stats)?;

    // Aggregate across all ranks so the printed mean/std/CI95 reflect every
    // scenario, not just rank 0's.
    let cost_summary =
        aggregate_simulation(&sim_run_result.costs, sim_config, &ctx.comm).map_err(|e| {
            CliError::Internal {
                message: format!("simulation cost aggregation error: {e}"),
            }
        })?;

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let parallelism = (ctx.n_threads as u32).saturating_mul(ctx.comm.size() as u32);

    merged_sim_output.cost = Some(MetadataCost {
        mean_cost: cost_summary.mean_cost,
        std_cost: cost_summary.std_cost,
        cvar: cost_summary.cvar,
        cvar_alpha: cost_summary.cvar_alpha,
    });
    merged_sim_output.solve_stats = MetadataSimulationSolveStats {
        total_lp_solves: Some(global_agg.lp_solves),
        first_try: Some(global_agg.first_try_successes),
        retried: Some(
            global_agg
                .lp_successes
                .saturating_sub(global_agg.first_try_successes),
        ),
        failed: Some(global_agg.lp_failures),
        solve_seconds: Some(global_agg.solve_time_ms / 1000.0),
        parallelism: Some(parallelism),
    };

    if !ctx.quiet && ctx.is_root {
        print_sim_summary(
            &ctx.stderr,
            n_scenarios,
            sim_time_ms,
            &global_agg,
            &cost_summary,
            parallelism,
        );
    }

    if ctx.is_root {
        write_sim_outputs_on_root(
            ctx,
            hostname,
            sim_started_at,
            &merged_sim_output,
            &global_scenario_stats,
        )?;
    }

    Ok(())
}

/// Write simulation output files on rank 0.
fn write_sim_outputs_on_root(
    ctx: &RunContext<impl Communicator>,
    hostname: &str,
    sim_started_at: String,
    merged_sim_output: &SimulationOutput,
    global_scenario_stats: &[(u32, SolverStatsDelta)],
) -> Result<(), CliError> {
    let mpi_world_size = u32::try_from(ctx.topology.world_size).unwrap_or(u32::MAX);
    let sim_ctx = OutputContext {
        hostname: hostname.to_string(),
        solver: active_solver_metadata_id().to_string(),
        solver_version: None,
        started_at: sim_started_at,
        completed_at: now_iso8601(),
        distribution: build_distribution_info(
            &ctx.topology,
            ctx.n_threads,
            mpi_world_size,
            &ctx.rank_affinity,
        ),
        setup: None,
        production_fit_deviation: None,
    };
    write_simulation_outputs(&WriteSimulationArgs {
        output_dir: &ctx.output_dir,
        sim_output: merged_sim_output,
        sim_solver_stats: global_scenario_stats,
        output_ctx: &sim_ctx,
        quiet: ctx.quiet,
        stderr: &ctx.stderr,
    })
}

/// Print the simulation summary from aggregated solver stats and cost statistics.
fn print_sim_summary(
    stderr: &Term,
    n_scenarios: u32,
    sim_time_ms: u64,
    agg: &SolverStatsDelta,
    cost_summary: &cobre_sddp::SimulationSummary,
    parallelism: u32,
) {
    print_simulation_summary(
        stderr,
        &SimulationSummary {
            n_scenarios,
            completed: n_scenarios,
            failed: 0,
            total_time_ms: sim_time_ms,
            mean_cost: Some(cost_summary.mean_cost),
            std_cost: Some(cost_summary.std_cost),
            total_lp_solves: Some(agg.lp_solves),
            total_first_try: Some(agg.first_try_successes),
            total_retried: Some(agg.lp_successes.saturating_sub(agg.first_try_successes)),
            total_failed_solves: Some(agg.lp_failures),
            total_solve_time_seconds: Some(agg.solve_time_ms / 1000.0),
            parallelism: Some(parallelism),
        },
    );
}

/// Merge each rank's local [`SimulationOutput`](cobre_io::SimulationOutput) via
/// MPI collectives.
#[allow(clippy::cast_possible_truncation)]
fn merge_simulation_metadata<C: Communicator>(
    comm: &C,
    local: &SimulationOutput,
) -> Result<SimulationOutput, CliError> {
    let send_counts = [local.n_scenarios, local.completed, local.failed];
    let mut merged_counts = [0u32; 3];
    comm.allreduce(&send_counts, &mut merged_counts, ReduceOp::Sum)
        .map_err(|e| CliError::Internal {
            message: format!("simulation metadata count allreduce error: {e}"),
        })?;

    // Max, not Sum: wall-clock is the slowest rank's time, not the total.
    let send_time = [local.total_time_ms];
    let mut merged_time = [0u64; 1];
    comm.allreduce(&send_time, &mut merged_time, ReduceOp::Max)
        .map_err(|e| CliError::Internal {
            message: format!("simulation metadata time allreduce error: {e}"),
        })?;

    let local_paths_bytes = local.partitions_written.join("\n").into_bytes();

    let send_len = [local_paths_bytes.len() as u64];
    let n_ranks = comm.size();
    let mut all_lens = vec![0u64; n_ranks];
    let len_counts: Vec<usize> = vec![1; n_ranks];
    let len_displs: Vec<usize> = (0..n_ranks).collect();
    comm.allgatherv(&send_len, &mut all_lens, &len_counts, &len_displs)
        .map_err(|e| CliError::Internal {
            message: format!("partition path length exchange error: {e}"),
        })?;

    let recv_counts: Vec<usize> = all_lens.iter().map(|&l| l as usize).collect();
    let recv_displs: Vec<usize> = recv_counts
        .iter()
        .scan(0usize, |acc, &c| {
            let d = *acc;
            *acc += c;
            Some(d)
        })
        .collect();
    let total_bytes: usize = recv_counts.iter().sum();
    let mut all_bytes = vec![0u8; total_bytes];
    comm.allgatherv(
        &local_paths_bytes,
        &mut all_bytes,
        &recv_counts,
        &recv_displs,
    )
    .map_err(|e| CliError::Internal {
        message: format!("partition path gather error: {e}"),
    })?;

    let mut all_partitions: Vec<String> = Vec::new();
    for (i, &count) in recv_counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let start = recv_displs[i];
        let chunk = &all_bytes[start..start + count];
        let text = std::str::from_utf8(chunk).map_err(|e| CliError::Internal {
            message: format!("partition path UTF-8 decode error from rank {i}: {e}"),
        })?;
        all_partitions.extend(text.split('\n').filter(|s| !s.is_empty()).map(String::from));
    }
    all_partitions.sort();

    Ok(SimulationOutput {
        n_scenarios: merged_counts[0],
        completed: merged_counts[1],
        failed: merged_counts[2],
        total_time_ms: merged_time[0],
        partitions_written: all_partitions,
        cost: None,
        solve_stats: MetadataSimulationSolveStats::default(),
    })
}

/// Aggregate simulation solver statistics across all MPI ranks.
///
/// Returns the global [`cobre_sddp::SolverStatsDelta`] (sum over all ranks, for
/// the root summary) and a per-global-scenario `Vec`, sorted by scenario ID for
/// deterministic Parquet output.
#[allow(clippy::cast_possible_truncation)]
fn aggregate_simulation_solver_stats<C: Communicator>(
    comm: &C,
    local_stats: &[(u32, i32, SolverStatsDelta)],
) -> Result<(SolverStatsDelta, Vec<(u32, SolverStatsDelta)>), CliError> {
    let local_agg = SolverStatsDelta::aggregate(local_stats.iter().map(|(_, _, d)| d));
    check_stats_overflow(&local_agg)?;
    let send_scalars = pack_delta_scalars(&local_agg);
    let mut recv_scalars = [0.0_f64; SOLVER_STATS_DELTA_SCALAR_FIELDS];
    comm.allreduce(&send_scalars, &mut recv_scalars, ReduceOp::Sum)
        .map_err(|e| CliError::Internal {
            message: format!("simulation solver stats allreduce error: {e}"),
        })?;
    let global_agg = unpack_delta_scalars(&recv_scalars);

    // Strip the opening field (always -1 here): the MPI wire format omits it.
    let local_stats_stripped: Vec<(u32, SolverStatsDelta)> = local_stats
        .iter()
        .map(|(id, _opening, delta)| (*id, delta.clone()))
        .collect();
    let n_ranks = comm.size();
    let local_buf = pack_scenario_stats(&local_stats_stripped);
    let local_count = local_buf.len();

    let send_len = [local_count as u64];
    let mut all_lens = vec![0u64; n_ranks];
    let len_counts: Vec<usize> = vec![1; n_ranks];
    let len_displs: Vec<usize> = (0..n_ranks).collect();
    comm.allgatherv(&send_len, &mut all_lens, &len_counts, &len_displs)
        .map_err(|e| CliError::Internal {
            message: format!("simulation solver stats length exchange error: {e}"),
        })?;

    let recv_counts: Vec<usize> = all_lens.iter().map(|&l| l as usize).collect();
    let recv_displs: Vec<usize> = recv_counts
        .iter()
        .scan(0usize, |acc, &c| {
            let d = *acc;
            *acc += c;
            Some(d)
        })
        .collect();
    let total_floats: usize = recv_counts.iter().sum();
    let mut all_buf = vec![0.0_f64; total_floats];
    comm.allgatherv(&local_buf, &mut all_buf, &recv_counts, &recv_displs)
        .map_err(|e| CliError::Internal {
            message: format!("simulation solver stats gather error: {e}"),
        })?;

    let mut global_scenario_stats = unpack_scenario_stats(&all_buf);
    global_scenario_stats.sort_by_key(|(id, _)| *id);

    Ok((global_agg, global_scenario_stats))
}
