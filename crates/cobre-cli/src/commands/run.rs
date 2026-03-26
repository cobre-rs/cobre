//! `cobre run <CASE_DIR>` subcommand.
//!
//! Executes the full solve lifecycle:
//!
//! 1. Load the case directory (`cobre_io::load_case`) on rank 0, then broadcast.
//! 2. Parse `config.json` (`cobre_io::parse_config`) on rank 0. Extract a
//!    postcard-safe [`BroadcastConfig`] and broadcast it to all ranks.
//!    The raw `Config` stays on rank 0 for output writing only.
//! 3. Build stage LP templates (`cobre_sddp::build_stage_templates`) on all ranks.
//! 4. Build the stochastic context (`cobre_stochastic::build_stochastic_context`) on all ranks.
//!    If `scenarios/noise_openings.parquet` is present, the user-supplied opening tree is
//!    loaded on rank 0, broadcast to all ranks, and passed to `build_stochastic_context`.
//! 5. Train the SDDP policy (`cobre_sddp::train`).
//! 6. Optionally run simulation (`cobre_sddp::simulate`).
//! 7. Write all outputs (`cobre_io::write_results`).

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use clap::Args;
use console::Term;

use cobre_comm::{Communicator, ReduceOp, create_communicator};
use cobre_core::{System, TrainingEvent};
use cobre_io::output::{
    write_correlation_json, write_fitting_report, write_inflow_ar_coefficients,
    write_inflow_seasonal_stats, write_load_seasonal_stats, write_noise_openings,
};
use cobre_io::scenarios::LoadSeasonalStatsRow;
use cobre_io::write_results;
use cobre_sddp::{
    EstimationReport, PrepareHydroModelsResult, PrepareStochasticResult, SimulationScenarioResult,
    StudySetup, build_hydro_model_summary, build_stochastic_summary,
    estimation_report_to_fitting_report, inflow_models_to_ar_rows, inflow_models_to_stats_rows,
    prepare_hydro_models, prepare_stochastic,
};
use cobre_solver::HighsSolver;
use cobre_stochastic::{
    build_stochastic_context, context::OpeningTree, provenance::ComponentProvenance,
};

use crate::error::CliError;
use crate::summary::{SimulationSummary, TrainingSummary};

use super::broadcast::{
    BroadcastConfig, BroadcastCutSelection, BroadcastOpeningTree, broadcast_value,
    stopping_rules_from_broadcast,
};

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

    /// Number of worker threads for parallel scenario processing within each
    /// MPI rank.  Each thread solves its own LP instances sequentially; multiple
    /// scenarios (forward passes, backward trial points, simulation runs) are
    /// processed in parallel across threads.
    ///
    /// Resolves in this order: (1) this flag, (2) `COBRE_THREADS` env var,
    /// (3) default of 1.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub threads: Option<u32>,
}

fn resolve_thread_count(cli_threads: Option<u32>) -> usize {
    if let Some(n) = cli_threads {
        return n as usize;
    }
    if let Ok(val) = std::env::var("COBRE_THREADS") {
        if let Ok(n) = val.parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    1
}

/// Return type of [`load_case_and_config`]: the values loaded on rank 0.
///
/// The [`PrepareStochasticResult`] bundles the updated system, built stochastic
/// context, and optional estimation report from the pre-setup pipeline.
/// The [`PrepareHydroModelsResult`] bundles the resolved production and evaporation
/// models for all hydro plants.
type LoadedCase = (
    PrepareStochasticResult,
    PrepareHydroModelsResult,
    BroadcastConfig,
    cobre_io::Config,
);

/// Load the case directory and parse the config on rank 0.
///
/// Extracted into a separate function so that errors are captured as `Err`
/// rather than causing an early return from `execute()`. This allows
/// `execute()` to always reach the `broadcast_value` calls, ensuring all
/// MPI ranks participate in the collectives even when rank 0 fails.
fn load_case_and_config(
    args: &RunArgs,
    quiet: bool,
    stderr: &Term,
) -> Result<LoadedCase, CliError> {
    if !args.case_dir.exists() {
        return Err(CliError::Io {
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "case directory does not exist",
            ),
            context: args.case_dir.display().to_string(),
        });
    }
    if !quiet {
        let _ = stderr.write_line(&format!("Loading case: {}", args.case_dir.display()));
    }
    let system = cobre_io::load_case(&args.case_dir)?;
    let config_path = args.case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path)?;
    let bcast = BroadcastConfig::from_config(&config)?;
    let seed = bcast.seed;
    let prepared =
        prepare_stochastic(system, &args.case_dir, &config, seed).map_err(CliError::from)?;
    let hydro_models =
        prepare_hydro_models(&prepared.system, &args.case_dir).map_err(CliError::from)?;
    Ok((prepared, hydro_models, bcast, config))
}

/// Execute the `run` subcommand.
///
/// Runs the full lifecycle: load → build templates → train → simulate → write.
///
/// Under MPI, only rank 0 loads from disk. The loaded `System` is serialized
/// with postcard and broadcast to all other ranks. The raw `Config` is parsed
/// on rank 0 only; a postcard-safe [`BroadcastConfig`] is extracted and
/// broadcast so that non-root ranks receive training and simulation parameters
/// without needing to deserialize the `Config` types that use
/// `#[serde(tag)]` (internally-tagged enums that postcard cannot handle).
///
/// All I/O, UI, and output writing is performed exclusively by rank 0.
/// Non-root ranks always behave as if `--quiet` is set, participating in MPI
/// collectives but producing no terminal output and writing no files.
///
/// # Errors
///
/// Returns [`CliError`] when loading, training, simulation, or I/O fails.
/// The exit code indicates the category of failure.
#[allow(clippy::too_many_lines)]
pub fn execute(args: RunArgs) -> Result<(), CliError> {
    let comm = create_communicator()?;
    let is_root = comm.rank() == 0;
    let quiet = args.quiet || !is_root;

    // Under MPI, mpiexec pipes rank 0's stderr through to the user's terminal
    // without allocating a PTY. Force color and terminal rendering on rank 0
    // so the banner and progress bars display correctly.
    let mpi_active = comm.size() > 1;
    if mpi_active && is_root && !args.quiet {
        console::set_colors_enabled_stderr(true);
    }

    let stderr = Term::stderr();

    if !quiet {
        crate::banner::print_banner(&stderr);
    }

    let n_threads = resolve_thread_count(args.threads);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .unwrap_or_else(|_| {
            tracing::warn!("rayon global thread pool already initialized; ignoring --threads");
        });

    // Rank 0 loads from disk; system and config are broadcast to all ranks.
    let (
        raw_system,
        raw_bcast_config,
        root_config,
        root_stochastic,
        root_estimation_report,
        raw_bcast_tree,
        root_hydro_models,
        load_err,
    ) = if is_root {
        match load_case_and_config(&args, quiet, &stderr) {
            Ok((prepared, hydro_models, bcast, config)) => {
                // Extract the opening tree for broadcast to non-root ranks.
                // If the stochastic context was built from a user-supplied tree,
                // re-serialize it into BroadcastOpeningTree; otherwise broadcast None.
                let bcast_tree = if prepared.stochastic.provenance().opening_tree
                    == ComponentProvenance::UserSupplied
                {
                    let t = prepared.stochastic.opening_tree();
                    Some(BroadcastOpeningTree {
                        data: t.data().to_vec(),
                        openings_per_stage: t.openings_per_stage_slice().to_vec(),
                        dim: t.dim(),
                    })
                } else {
                    None
                };
                let cobre_sddp::PrepareStochasticResult {
                    system,
                    stochastic,
                    estimation_report,
                } = prepared;
                (
                    Some(system),
                    Some(bcast),
                    Some(config),
                    Some(stochastic),
                    Some(estimation_report),
                    Some(bcast_tree),
                    Some(hydro_models),
                    None,
                )
            }
            Err(e) => (None, None, None, None, None, None, None, Some(e)),
        }
    } else {
        (None, None, None, None, None, None, None, None)
    };
    let root_estimation_report: Option<Option<EstimationReport>> = root_estimation_report;

    let system_result = broadcast_value(raw_system, &comm);
    let bcast_config_result = broadcast_value(raw_bcast_config, &comm);
    let root_hydro_models: Option<PrepareHydroModelsResult> = root_hydro_models;

    // Broadcast the optional opening tree. Wrap in Option<BroadcastOpeningTree> so that
    // both the "no user tree" (None) and "user tree present" (Some) cases can be broadcast
    // as a single postcard-serializable value. postcard supports Option natively.
    let tree_result = broadcast_value(raw_bcast_tree, &comm);

    if let Some(e) = load_err {
        return Err(e);
    }
    let system = system_result?;
    let mut bcast_config = bcast_config_result?;

    // Consume args here — no fields are needed after output_dir is resolved.
    let output_dir: PathBuf = args.output.unwrap_or_else(|| args.case_dir.join("output"));

    let seed = bcast_config.seed;

    // Rank 0 uses the stochastic context already built by `prepare_stochastic`.
    // Non-root ranks reconstruct it from the broadcast opening tree.
    let stochastic = if is_root {
        // tree_result was produced by the broadcast collective above; discard it here
        // since rank 0 already has the stochastic context from `prepare_stochastic`.
        drop(tree_result);
        root_stochastic.ok_or_else(|| CliError::Internal {
            message: "stochastic context missing on rank 0 after successful load".to_string(),
        })?
    } else {
        let user_tree: Option<OpeningTree> =
            tree_result?.map(|bt| OpeningTree::from_parts(bt.data, bt.openings_per_stage, bt.dim));
        build_stochastic_context(&system, seed, &[], &[], user_tree).map_err(|e| {
            CliError::Internal {
                message: format!("stochastic context error: {e}"),
            }
        })?
    };

    // Rank 0 uses the hydro models result already built by `prepare_hydro_models`.
    // Non-root ranks reconstruct it independently from the system and case_dir.
    // All ranks have access to the same shared filesystem, so independent loading
    // produces identical results.
    let hydro_models = if is_root {
        root_hydro_models.ok_or_else(|| CliError::Internal {
            message: "hydro models missing on rank 0 after successful load".to_string(),
        })?
    } else {
        prepare_hydro_models(&system, &args.case_dir).map_err(|e| CliError::Internal {
            message: format!("hydro model preprocessing error on non-root rank: {e}"),
        })?
    };

    // Construct StudySetup on all ranks from broadcast parameters.
    // Ownership of stochastic moves into setup; use setup.stochastic() for all
    // subsequent stochastic references.
    let stopping_rule_set = stopping_rules_from_broadcast(&bcast_config);
    let cut_selection = std::mem::replace(
        &mut bcast_config.cut_selection,
        BroadcastCutSelection::Disabled,
    )
    .into_strategy();
    let mut setup = StudySetup::from_broadcast_params(
        &system,
        stochastic,
        bcast_config.seed,
        bcast_config.forward_passes,
        stopping_rule_set,
        bcast_config.n_scenarios,
        usize::try_from(bcast_config.io_channel_capacity).unwrap_or(64),
        bcast_config.policy_path.clone(),
        bcast_config.inflow_method.clone(),
        cut_selection,
        bcast_config.cut_activity_tolerance,
        hydro_models,
    )
    .map_err(CliError::from)?;

    // Print stochastic preprocessing summary on rank 0 before training starts.
    if !quiet && is_root {
        let estimation = root_estimation_report.as_ref().and_then(|r| r.as_ref());
        let stochastic_summary =
            build_stochastic_summary(&system, setup.stochastic(), estimation, seed);
        crate::summary::print_stochastic_summary(&stderr, &stochastic_summary);
        let hydro_summary = build_hydro_model_summary(setup.hydro_models(), &system);
        crate::summary::print_hydro_model_summary(&stderr, &hydro_summary);
    }

    // Export stochastic preprocessing artifacts when requested (rank 0 only).
    if is_root && root_config.as_ref().is_some_and(|c| c.exports.stochastic) {
        let estimation = root_estimation_report.as_ref().and_then(|r| r.as_ref());
        export_stochastic_artifacts(
            &output_dir,
            setup.stochastic(),
            &system,
            estimation,
            quiet,
            &stderr,
        );
    }
    // Write scaling report (rank 0 only, before training starts).
    if is_root {
        let scaling_path = output_dir.join("training/scaling_report.json");
        cobre_io::write_scaling_report(&scaling_path, setup.scaling_report()).map_err(|e| {
            CliError::Internal {
                message: format!("failed to write scaling report: {e}"),
            }
        })?;
    }

    comm.barrier().map_err(|e| CliError::Internal {
        message: format!("post-export barrier error: {e}"),
    })?;

    let solver_factory = HighsSolver::new;

    let mut solver = HighsSolver::new().map_err(|e| CliError::Solver {
        message: format!("HiGHS initialisation failed: {e}"),
    })?;

    let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();

    let term_width = crate::progress::resolve_term_width();
    let quiet_rx: Option<mpsc::Receiver<TrainingEvent>>;
    let progress_handle = if quiet {
        quiet_rx = Some(event_rx);
        None
    } else {
        quiet_rx = None;
        Some(crate::progress::run_progress_thread(
            event_rx,
            setup.max_iterations(),
            term_width,
        ))
    };

    let training_result = match setup.train(
        &mut solver,
        &comm,
        n_threads,
        solver_factory,
        Some(event_tx),
        None,
    ) {
        Ok(result) => result,
        Err(e) => {
            if let Some(handle) = progress_handle {
                let _ = handle.join();
            }
            return Err(CliError::from(e));
        }
    };

    let events: Vec<TrainingEvent> = match (progress_handle, quiet_rx) {
        (Some(handle), _) => handle.join(),
        (None, Some(rx)) => rx.try_iter().collect(),
        (None, None) => Vec::new(),
    };
    let training_output = setup.build_training_output(&training_result, &events);

    let local_lp_solves: u64 = training_output
        .convergence_records
        .iter()
        .map(|r| u64::from(r.lp_solves))
        .sum();
    let mut global_lp_solves = [0u64];
    comm.allreduce(&[local_lp_solves], &mut global_lp_solves, ReduceOp::Sum)
        .map_err(|e| CliError::Internal {
            message: format!("LP solve count allreduce error: {e}"),
        })?;
    let global_lp_solves = global_lp_solves[0];

    comm.barrier().map_err(|e| CliError::Internal {
        message: format!("post-training barrier error: {e}"),
    })?;

    // Aggregate solver stats from the stats log for the summary display.
    let (
        total_first_try,
        total_retried,
        total_failed,
        total_solve_time_s,
        total_basis_offered,
        total_basis_rejections,
        total_simplex_iter,
    ) = {
        let mut first_try = 0u64;
        let mut retried = 0u64;
        let mut failed = 0u64;
        let mut solve_time = 0.0_f64;
        let mut basis_offered = 0u64;
        let mut basis_rejections = 0u64;
        let mut simplex = 0u64;
        for (_, _, _, delta) in &training_result.solver_stats_log {
            first_try += delta.first_try_successes;
            retried += delta.lp_successes.saturating_sub(delta.first_try_successes);
            failed += delta.lp_failures;
            solve_time += delta.solve_time_ms;
            basis_offered += delta.basis_offered;
            basis_rejections += delta.basis_rejections;
            simplex += delta.simplex_iterations;
        }
        (
            first_try,
            retried,
            failed,
            solve_time / 1000.0,
            basis_offered,
            basis_rejections,
            simplex,
        )
    };

    // Print training summary immediately after training completes.
    let training_summary = TrainingSummary {
        iterations: training_result.iterations,
        converged: training_output.converged,
        converged_at: if training_output.converged {
            Some(training_result.iterations)
        } else {
            None
        },
        reason: training_result.reason.clone(),
        lower_bound: training_result.final_lb,
        upper_bound: training_result.final_ub,
        upper_bound_std: training_result.final_ub_std,
        gap_percent: training_result.final_gap * 100.0,
        total_cuts_active: training_output.cut_stats.total_active,
        total_cuts_generated: training_output.cut_stats.total_generated,
        total_lp_solves: global_lp_solves,
        total_time_ms: training_result.total_time_ms,
        total_first_try,
        total_retried,
        total_failed,
        total_solve_time_seconds: total_solve_time_s,
        total_basis_offered,
        total_basis_rejections,
        total_simplex_iterations: total_simplex_iter,
    };
    if !quiet && is_root {
        crate::summary::print_training_summary(&stderr, &training_summary);
    }

    let should_simulate = setup.n_scenarios() > 0;

    if should_simulate {
        let n_scenarios = setup.n_scenarios();
        let sim_config = setup.simulation_config();

        let mut sim_pool = setup
            .create_workspace_pool(n_threads, solver_factory)
            .map_err(|e| CliError::Solver {
                message: format!("HiGHS initialisation failed for simulation pool: {e}"),
            })?;

        let (sim_event_tx, sim_event_rx) = mpsc::channel::<TrainingEvent>();
        let sim_progress_handle = if quiet {
            drop(sim_event_rx);
            None
        } else {
            Some(crate::progress::run_progress_thread(
                sim_event_rx,
                u64::from(n_scenarios),
                term_width,
            ))
        };

        let io_capacity = sim_config.io_channel_capacity;
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity.max(1));

        let drain_handle = std::thread::spawn(move || {
            result_rx
                .into_iter()
                .collect::<Vec<SimulationScenarioResult>>()
        });

        let sim_start = std::time::Instant::now();

        let sim_result = setup
            .simulate(
                &mut sim_pool.workspaces,
                &comm,
                &result_tx,
                Some(sim_event_tx),
                &training_result.basis_cache,
            )
            .map_err(CliError::from);
        if let Some(handle) = sim_progress_handle {
            let _ = handle.join();
        }

        drop(result_tx);

        #[allow(clippy::expect_used)]
        let local_results = drain_handle.join().expect("drain thread panicked");

        let sim_run_result = sim_result?;

        #[allow(clippy::cast_possible_truncation)]
        let sim_time_ms = sim_start.elapsed().as_millis() as u64;

        let all_results = gather_simulation_results(&comm, &local_results)?;

        comm.barrier().map_err(|e| CliError::Internal {
            message: format!("post-simulation barrier error: {e}"),
        })?;

        // Aggregate solver stats from per-scenario deltas for the summary display.
        let sim_solver_agg = cobre_sddp::SolverStatsDelta::aggregate(
            &sim_run_result
                .solver_stats
                .iter()
                .map(|(_, delta)| delta.clone())
                .collect::<Vec<_>>(),
        );

        // Print the simulation summary now — before I/O starts.
        if !quiet && is_root {
            crate::summary::print_simulation_summary(
                &stderr,
                &SimulationSummary {
                    n_scenarios,
                    completed: n_scenarios,
                    failed: 0,
                    total_time_ms: sim_time_ms,
                    total_lp_solves: sim_solver_agg.lp_solves,
                    total_first_try: sim_solver_agg.first_try_successes,
                    total_retried: sim_solver_agg
                        .lp_successes
                        .saturating_sub(sim_solver_agg.first_try_successes),
                    total_failed_solves: sim_solver_agg.lp_failures,
                    total_solve_time_seconds: sim_solver_agg.solve_time_ms / 1000.0,
                    total_basis_offered: sim_solver_agg.basis_offered,
                    total_basis_rejections: sim_solver_agg.basis_rejections,
                    total_simplex_iterations: sim_solver_agg.simplex_iterations,
                },
            );
        }

        if is_root {
            let config = root_config.ok_or_else(|| CliError::Internal {
                message: "root_config was None on rank 0 — internal invariant violated".to_string(),
            })?;

            let parquet_config = cobre_io::ParquetWriterConfig::default();
            let mut sim_writer = cobre_io::output::simulation_writer::SimulationParquetWriter::new(
                &output_dir,
                &system,
                &parquet_config,
            )
            .map_err(CliError::from)?;

            let mut failed: u32 = 0;
            for scenario_result in all_results {
                let payload = cobre_io::output::simulation_writer::ScenarioWritePayload::from(
                    scenario_result,
                );
                if let Err(e) = sim_writer.write_scenario(payload) {
                    tracing::error!("simulation write error: {e}");
                    failed += 1;
                }
            }
            let mut sim_output = sim_writer.finalize(sim_time_ms);
            sim_output.failed = failed;

            write_outputs(
                &output_dir,
                &system,
                &config,
                &training_output,
                Some(&sim_output),
                &setup,
                &training_result,
                Some(&sim_run_result.solver_stats),
                quiet,
                &stderr,
            )?;
        }
    } else if is_root {
        let config = root_config.ok_or_else(|| CliError::Internal {
            message: "root_config was None on rank 0 — internal invariant violated".to_string(),
        })?;

        write_outputs(
            &output_dir,
            &system,
            &config,
            &training_output,
            None,
            &setup,
            &training_result,
            None,
            quiet,
            &stderr,
        )?;
    }

    Ok(())
}

/// Gather simulation results from all MPI ranks into a single `Vec`.
///
/// Extracted from `execute()` so that the allgatherv + deserialization pattern has
/// a clear name and boundary. Runs on all ranks: each rank serializes its local
/// results, exchanges byte-count lengths via `allgatherv`, exchanges the raw bytes
/// via a second `allgatherv`, then deserializes each rank's partition into the
/// returned `Vec`. All ranks return the complete set of results.
fn gather_simulation_results<C: Communicator>(
    comm: &C,
    local_results: &[SimulationScenarioResult],
) -> Result<Vec<SimulationScenarioResult>, CliError> {
    let local_bytes = postcard::to_allocvec(local_results).map_err(|e| CliError::Internal {
        message: format!("simulation result serialization error: {e}"),
    })?;

    let n_ranks = comm.size();
    #[allow(clippy::cast_possible_truncation)]
    let send_len = [local_bytes.len() as u64];
    let mut all_lens = vec![0u64; n_ranks];
    let len_counts: Vec<usize> = vec![1; n_ranks];
    let len_displs: Vec<usize> = (0..n_ranks).collect();
    comm.allgatherv(&send_len, &mut all_lens, &len_counts, &len_displs)
        .map_err(|e| CliError::Internal {
            message: format!("simulation result length exchange error: {e}"),
        })?;

    let recv_counts: Vec<usize> = all_lens
        .iter()
        .map(|&l| {
            usize::try_from(l).map_err(|e| CliError::Internal {
                message: format!("simulation result byte count exceeds usize: {e}"),
            })
        })
        .collect::<Result<_, _>>()?;
    let recv_displs: Vec<usize> = recv_counts
        .iter()
        .scan(0usize, |acc, &c| {
            let displ = *acc;
            *acc += c;
            Some(displ)
        })
        .collect();
    let total_bytes: usize = recv_counts.iter().sum();

    let mut all_bytes = vec![0u8; total_bytes];
    comm.allgatherv(&local_bytes, &mut all_bytes, &recv_counts, &recv_displs)
        .map_err(|e| CliError::Internal {
            message: format!("simulation result gather error: {e}"),
        })?;

    let mut all_results: Vec<SimulationScenarioResult> = Vec::new();
    let mut offset = 0;
    for &count in &recv_counts {
        let partition: Vec<SimulationScenarioResult> =
            postcard::from_bytes(&all_bytes[offset..offset + count]).map_err(|e| {
                CliError::Internal {
                    message: format!("simulation result deserialization error: {e}"),
                }
            })?;
        all_results.extend(partition);
        offset += count;
    }

    Ok(all_results)
}

/// Write training checkpoint and results to the output directory.
///
/// Convert a [`SolverStatsDelta`] into a [`SolverStatsRow`] for Parquet output.
///
/// The `id` parameter is the row identifier: iteration number for training phases,
/// scenario ID for the simulation phase.
#[allow(clippy::cast_possible_truncation)]
fn delta_to_stats_row(
    id: u32,
    phase: &str,
    stage: i32,
    delta: &cobre_sddp::SolverStatsDelta,
) -> cobre_io::SolverStatsRow {
    cobre_io::SolverStatsRow {
        iteration: id,
        phase: phase.to_string(),
        stage,
        lp_solves: delta.lp_solves as u32,
        lp_successes: delta.lp_successes as u32,
        lp_retries: delta.lp_successes.saturating_sub(delta.first_try_successes) as u32,
        lp_failures: delta.lp_failures as u32,
        retry_attempts: delta.retry_attempts as u32,
        basis_offered: delta.basis_offered as u32,
        basis_rejections: delta.basis_rejections as u32,
        simplex_iterations: delta.simplex_iterations,
        solve_time_ms: delta.solve_time_ms,
        load_model_time_ms: delta.load_model_time_ms,
        add_rows_time_ms: delta.add_rows_time_ms,
        set_bounds_time_ms: delta.set_bounds_time_ms,
        basis_set_time_ms: delta.basis_set_time_ms,
    }
}

/// Extracted from `execute()` to give the output-writing step a clear boundary.
/// Handles both the with-simulation path (`sim_output = Some(...)`) and the
/// training-only path (`sim_output = None`). Prints "Writing outputs..." and
/// the output path with timing when `quiet` is false.
#[allow(clippy::too_many_arguments)]
fn write_outputs(
    output_dir: &Path,
    system: &System,
    config: &cobre_io::Config,
    training_output: &cobre_io::TrainingOutput,
    sim_output: Option<&cobre_io::SimulationOutput>,
    setup: &StudySetup,
    training_result: &cobre_sddp::TrainingResult,
    sim_solver_stats: Option<&[(u32, cobre_sddp::SolverStatsDelta)]>,
    quiet: bool,
    stderr: &Term,
) -> Result<(), CliError> {
    if !quiet {
        use std::io::Write;
        let _ = stderr.write_line("Writing outputs...");
        let _ = std::io::stderr().flush();
    }
    let write_start = std::time::Instant::now();

    let policy_dir = output_dir.join(setup.policy_path());
    crate::policy_io::write_checkpoint(
        &policy_dir,
        setup.fcf(),
        training_result,
        &crate::policy_io::CheckpointParams {
            max_iterations: setup.max_iterations(),
            forward_passes: setup.forward_passes(),
            seed: setup.seed(),
        },
    )?;

    write_results(output_dir, training_output, sim_output, system, config)
        .map_err(CliError::from)?;

    // Write training solver stats to training/solver/iterations.parquet.
    if !training_result.solver_stats_log.is_empty() {
        let rows: Vec<cobre_io::SolverStatsRow> = training_result
            .solver_stats_log
            .iter()
            .map(|(iter, phase, stage, delta)| {
                #[allow(clippy::cast_possible_truncation)] // iteration count fits in u32
                delta_to_stats_row(*iter as u32, phase, *stage, delta)
            })
            .collect();
        cobre_io::write_solver_stats(output_dir, &rows).map_err(CliError::from)?;
    }

    // Write per-stage cut selection records to training/cut_selection/iterations.parquet.
    if !training_output.cut_selection_records.is_empty() {
        let parquet_config = cobre_io::ParquetWriterConfig::default();
        cobre_io::write_cut_selection_records(
            output_dir,
            &training_output.cut_selection_records,
            &parquet_config,
        )
        .map_err(CliError::from)?;
    }

    // Write simulation solver stats to simulation/solver/iterations.parquet.
    if let Some(stats) = sim_solver_stats {
        if !stats.is_empty() {
            let rows: Vec<cobre_io::SolverStatsRow> = stats
                .iter()
                .map(|(scenario_id, delta)| {
                    delta_to_stats_row(*scenario_id, "simulation", -1, delta)
                })
                .collect();
            cobre_io::write_simulation_solver_stats(output_dir, &rows).map_err(CliError::from)?;
        }
    }

    if !quiet {
        let write_secs = write_start.elapsed().as_secs_f64();
        crate::summary::print_output_path(stderr, output_dir, write_secs);
    }

    Ok(())
}

/// Write all applicable stochastic preprocessing artifacts to `{output_dir}/stochastic/`.
///
/// Called on rank 0 only when `--export-stochastic` is set (or `exports.stochastic = true`
/// in config). Each writer call is independent: failure is logged as a warning on stderr
/// and does not prevent the remaining files or training from proceeding.
///
/// Files written:
/// - `noise_openings.parquet` — always
/// - `inflow_seasonal_stats.parquet` — always
/// - `inflow_ar_coefficients.parquet` — always
/// - `correlation.json` — always
/// - `load_seasonal_stats.parquet` — only when `system.load_models()` contains any model with `std_mw > 0`
/// - `fitting_report.json` — only when `estimation_report` is `Some`
fn export_stochastic_artifacts(
    output_dir: &Path,
    stochastic: &cobre_stochastic::StochasticContext,
    system: &System,
    estimation_report: Option<&EstimationReport>,
    quiet: bool,
    stderr: &Term,
) {
    use cobre_core::scenario::LoadModel;

    let stochastic_dir = output_dir.join("stochastic");

    if !quiet {
        let _ = stderr.write_line("Exporting stochastic artifacts...");
    }

    if let Err(e) = write_noise_openings(
        &stochastic_dir.join("noise_openings.parquet"),
        stochastic.opening_tree(),
    ) {
        if !quiet {
            let _ = stderr.write_line(&format!(
                "warning: stochastic export failed (noise_openings): {e}"
            ));
        }
    }

    let stats_rows = inflow_models_to_stats_rows(system.inflow_models());
    if let Err(e) = write_inflow_seasonal_stats(
        &stochastic_dir.join("inflow_seasonal_stats.parquet"),
        &stats_rows,
    ) {
        if !quiet {
            let _ = stderr.write_line(&format!(
                "warning: stochastic export failed (inflow_seasonal_stats): {e}"
            ));
        }
    }

    let ar_rows = inflow_models_to_ar_rows(system.inflow_models());
    if let Err(e) = write_inflow_ar_coefficients(
        &stochastic_dir.join("inflow_ar_coefficients.parquet"),
        &ar_rows,
    ) {
        if !quiet {
            let _ = stderr.write_line(&format!(
                "warning: stochastic export failed (inflow_ar_coefficients): {e}"
            ));
        }
    }

    if let Err(e) = write_correlation_json(
        &stochastic_dir.join("correlation.json"),
        system.correlation(),
    ) {
        if !quiet {
            let _ = stderr.write_line(&format!(
                "warning: stochastic export failed (correlation): {e}"
            ));
        }
    }

    let has_stochastic_load = system
        .load_models()
        .iter()
        .any(|m: &LoadModel| m.std_mw > 0.0);
    if has_stochastic_load {
        let load_rows: Vec<LoadSeasonalStatsRow> = system
            .load_models()
            .iter()
            .map(|m| LoadSeasonalStatsRow {
                bus_id: m.bus_id,
                stage_id: m.stage_id,
                mean_mw: m.mean_mw,
                std_mw: m.std_mw,
            })
            .collect();
        if let Err(e) = write_load_seasonal_stats(
            &stochastic_dir.join("load_seasonal_stats.parquet"),
            &load_rows,
        ) {
            if !quiet {
                let _ = stderr.write_line(&format!(
                    "warning: stochastic export failed (load_seasonal_stats): {e}"
                ));
            }
        }
    }

    if let Some(report) = estimation_report {
        let fitting = estimation_report_to_fitting_report(report);
        if let Err(e) = write_fitting_report(&stochastic_dir.join("fitting_report.json"), &fitting)
        {
            if !quiet {
                let _ = stderr.write_line(&format!(
                    "warning: stochastic export failed (fitting_report): {e}"
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_thread_count;

    #[test]
    fn test_resolve_thread_count_cli_value() {
        assert_eq!(resolve_thread_count(Some(4)), 4);
    }

    #[test]
    fn test_resolve_thread_count_default() {
        assert_eq!(resolve_thread_count(Some(1)), 1);
    }
}
