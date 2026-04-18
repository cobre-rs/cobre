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

use cobre_comm::{
    create_communicator, Communicator, ExecutionTopology, ReduceOp, TopologyProvider,
};
use cobre_core::{System, TrainingEvent};
use cobre_io::output::{
    write_correlation_json, write_fitting_report, write_inflow_ar_coefficients,
    write_inflow_seasonal_stats, write_load_seasonal_stats, write_noise_openings,
};
use cobre_io::scenarios::LoadSeasonalStatsRow;
use cobre_sddp::{
    build_hydro_model_summary, estimation_report_to_fitting_report, inflow_models_to_ar_rows,
    inflow_models_to_stats_rows, prepare_hydro_models, prepare_stochastic,
    setup::{build_ncs_factor_entries, load_load_factors_for_stochastic, ConstructionConfig},
    EstimationReport, PrepareHydroModelsResult, PrepareStochasticResult, StudySetup,
};
use cobre_solver::HighsSolver;
use cobre_stochastic::{
    build_stochastic_context, context::OpeningTree, provenance::ComponentProvenance,
    OpeningTreeInputs,
};

use crate::error::CliError;
use crate::summary::{SimulationSummary, TrainingSummary};

use super::broadcast::{
    broadcast_value, stopping_rules_from_broadcast, BroadcastConfig, BroadcastCutSelection,
    BroadcastOpeningTree,
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

/// Values loaded on rank 0 by [`load_case_and_config`].
type LoadedCase = (
    PrepareStochasticResult,
    PrepareHydroModelsResult,
    BroadcastConfig,
    cobre_io::Config,
);

/// Load case and config on rank 0, capturing errors for MPI collective participation.
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
    let prepared = prepare_stochastic(
        system,
        &args.case_dir,
        &config,
        seed,
        &bcast.training_source,
    )
    .map_err(CliError::from)?;
    let hydro_models =
        prepare_hydro_models(&prepared.system, &args.case_dir).map_err(CliError::from)?;
    Ok((prepared, hydro_models, bcast, config))
}

/// Shared context for execute phases (communicator, output, topology, etc.).
struct RunContext<C: Communicator> {
    /// The MPI (or local) communicator.
    comm: C,
    /// Whether this rank is rank 0.
    is_root: bool,
    /// Whether terminal output is suppressed.
    quiet: bool,
    /// Number of rayon worker threads.
    n_threads: usize,
    /// Resolved output directory.
    output_dir: PathBuf,
    /// Terminal width for progress bars.
    term_width: u16,
    /// Terminal handle for stderr output.
    stderr: Term,
    /// Execution topology gathered during communicator setup.
    topology: ExecutionTopology,
    /// Solver version string (e.g. `"1.8.0"`).
    solver_version: String,
}

/// Output of [`broadcast_and_build_setup`]: system, setup, config, and metadata.
struct LoadBroadcastResult {
    system: System,
    setup: StudySetup,
    /// Root-only config for output writing (None on non-root ranks).
    root_config: Option<cobre_io::Config>,
    /// Root-only estimation report for summaries (None on non-root ranks).
    root_estimation_report: Option<EstimationReport>,
    /// Root-only estimation path from stochastic preprocessing (None on non-root ranks).
    root_estimation_path: Option<cobre_sddp::EstimationPath>,
    /// Whether the training phase is enabled (broadcast from rank 0).
    training_enabled: bool,
    /// Policy initialization mode (broadcast from rank 0).
    policy_mode: cobre_io::PolicyMode,
    /// Which `HiGHS` basis-setter to call on each warm-start (broadcast from rank 0).
    warm_start_basis_mode: cobre_solver::highs::WarmStartBasisMode,
}

/// Output of [`run_training_phase`]: result, training output, and optional error.
struct TrainingPhaseResult {
    result: cobre_sddp::TrainingResult,
    output: cobre_io::TrainingOutput,
    /// Mid-iteration training error, if any. Partial results are still valid.
    error: Option<cobre_sddp::SddpError>,
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
pub fn execute(args: &RunArgs) -> Result<(), CliError> {
    let ctx = setup_communicator(args)?;
    let result = execute_inner(&ctx, args);
    if let Err(ref e) = result {
        if ctx.comm.size() > 1 {
            ctx.comm.abort(e.exit_code());
        }
    }
    result
}

fn execute_inner<C: Communicator>(ctx: &RunContext<C>, args: &RunArgs) -> Result<(), CliError> {
    let LoadBroadcastResult {
        system,
        mut setup,
        mut root_config,
        root_estimation_report,
        root_estimation_path,
        training_enabled,
        policy_mode,
        warm_start_basis_mode,
    } = broadcast_and_build_setup(ctx, args)?;

    // Pre-training outputs (estimation artifacts, scaling report) run
    // regardless of training_enabled — they are data preparation outputs.
    run_pre_training(
        ctx,
        &system,
        &setup,
        root_config.as_ref(),
        root_estimation_report.as_ref(),
        root_estimation_path,
    )?;

    // Shared runtime context for metadata output files.
    let hostname = ctx.topology.leader_hostname().to_string();
    let mpi_world_size = u32::try_from(ctx.topology.world_size).unwrap_or(u32::MAX);

    if training_enabled {
        apply_training_policy(ctx, &system, &mut setup, root_config.as_ref(), policy_mode)?;
        let training_started_at = cobre_io::now_iso8601();
        let training = run_training_phase(ctx, &mut setup, warm_start_basis_mode)?;
        let training_completed_at = cobre_io::now_iso8601();

        // Write training outputs immediately (before simulation), so training
        // artifacts are persisted even if simulation fails.
        if ctx.is_root {
            let config = root_config.take().ok_or_else(|| CliError::Internal {
                message: "root_config was None on rank 0 — internal invariant violated".to_string(),
            })?;
            let training_ctx = cobre_io::OutputContext {
                hostname: hostname.clone(),
                solver: "highs".to_string(),
                solver_version: Some(ctx.solver_version.clone()),
                started_at: training_started_at,
                completed_at: training_completed_at,
                distribution: build_distribution_info(&ctx.topology, ctx.n_threads, mpi_world_size),
            };
            write_training_outputs(&WriteTrainingArgs {
                output_dir: &ctx.output_dir,
                system: &system,
                config: &config,
                training_output: &training.output,
                setup: &setup,
                training_result: &training.result,
                output_ctx: &training_ctx,
                quiet: ctx.quiet,
                stderr: &ctx.stderr,
            })?;
            drop(config);
        }

        // If training failed mid-iteration, report the error after writing
        // partial outputs. All ranks return here — simulation is skipped.
        if let Some(ref training_error) = training.error {
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
            return Err(CliError::Internal {
                message: format!("training error: {training_error}"),
            });
        }

        if setup.n_scenarios() > 0 {
            run_simulation_phase(ctx, &system, &mut setup, &training.result, &hostname)?;
        }
    } else if setup.n_scenarios() > 0 {
        // Training disabled but simulation requested: load policy from disk.
        let training_result =
            load_policy_for_simulation(ctx, &system, &mut setup, root_config.as_ref())?;
        run_simulation_phase(ctx, &system, &mut setup, &training_result, &hostname)?;
    } else {
        // Both training and simulation disabled — nothing to do.
        if ctx.is_root && !ctx.quiet {
            let _ = ctx
                .stderr
                .write_line("Training disabled, simulation disabled — nothing to do.");
        }
    }

    Ok(())
}

/// Load a policy checkpoint from disk and optionally validate its compatibility.
///
/// Returns the loaded checkpoint. The `policy_dir` must already exist.
fn load_and_validate_checkpoint(
    policy_dir: &Path,
    system: &System,
    setup: &StudySetup,
    root_config: Option<&cobre_io::Config>,
) -> Result<cobre_io::PolicyCheckpoint, CliError> {
    let checkpoint = cobre_io::output::policy::read_policy_checkpoint(policy_dir).map_err(|e| {
        CliError::Internal {
            message: format!("failed to read policy checkpoint: {e}"),
        }
    })?;

    if let Some(config) = root_config {
        if config.policy.validate_compatibility {
            #[allow(clippy::cast_possible_truncation)]
            let n_stages = system.stages().iter().filter(|s| s.id >= 0).count() as u32;
            let state_dim =
                u32::try_from(setup.fcf().state_dimension).map_err(|e| CliError::Internal {
                    message: format!("state_dimension overflows u32: {e}"),
                })?;
            cobre_sddp::validate_policy_compatibility(&checkpoint.metadata, state_dim, n_stages)
                .map_err(CliError::from)?;
        }
    }

    Ok(checkpoint)
}

/// Apply warm-start or resume policy before training, if requested.
fn apply_training_policy(
    ctx: &RunContext<impl Communicator>,
    system: &System,
    setup: &mut StudySetup,
    root_config: Option<&cobre_io::Config>,
    policy_mode: cobre_io::PolicyMode,
) -> Result<(), CliError> {
    match policy_mode {
        cobre_io::PolicyMode::WarmStart => {
            let policy_dir = ctx.output_dir.join(setup.policy_path());
            if !policy_dir.exists() {
                return Err(CliError::Internal {
                    message: format!(
                        "Policy directory not found: {}. Cannot warm-start \
                         without a prior policy.",
                        policy_dir.display()
                    ),
                });
            }
            if ctx.is_root && !ctx.quiet {
                let _ = ctx
                    .stderr
                    .write_line("Loading prior policy for warm-start training...");
            }
            let checkpoint = load_and_validate_checkpoint(&policy_dir, system, setup, root_config)?;
            // Reserve one extra slot for cuts added in the final iteration.
            let warm_fcf = cobre_sddp::FutureCostFunction::new_with_warm_start(
                &checkpoint.stage_cuts,
                setup.forward_passes(),
                setup.max_iterations().saturating_add(1),
            )
            .map_err(CliError::from)?;
            setup.replace_fcf(warm_fcf);
            if ctx.is_root && !ctx.quiet {
                let warm_count = setup.fcf().pools[0].warm_start_count;
                let _ = ctx.stderr.write_line(&format!(
                    "Warm-start: loaded {warm_count} cuts per stage from prior policy."
                ));
            }
        }
        cobre_io::PolicyMode::Resume => {
            let policy_dir = ctx.output_dir.join(setup.policy_path());
            if !policy_dir.exists() {
                return Err(CliError::Internal {
                    message: format!(
                        "Policy directory not found: {}. Cannot resume \
                         without a prior checkpoint.",
                        policy_dir.display()
                    ),
                });
            }
            if ctx.is_root && !ctx.quiet {
                let _ = ctx
                    .stderr
                    .write_line("Loading prior checkpoint for resume training...");
            }
            let checkpoint = load_and_validate_checkpoint(&policy_dir, system, setup, root_config)?;
            let completed = u64::from(checkpoint.metadata.completed_iterations);
            if completed >= setup.max_iterations() && ctx.is_root && !ctx.quiet {
                let _ = ctx.stderr.write_line(&format!(
                    "WARNING: Checkpoint already completed {completed} iterations \
                     (max_iterations = {}). No additional training will occur.",
                    setup.max_iterations()
                ));
            }
            // Reserve one extra slot for cuts added in the final iteration.
            let warm_fcf = cobre_sddp::FutureCostFunction::new_with_warm_start(
                &checkpoint.stage_cuts,
                setup.forward_passes(),
                setup.max_iterations().saturating_add(1),
            )
            .map_err(CliError::from)?;
            setup.replace_fcf(warm_fcf);
            setup.set_start_iteration(completed);
            if ctx.is_root && !ctx.quiet {
                let warm_count = setup.fcf().pools[0].warm_start_count;
                let _ = ctx.stderr.write_line(&format!(
                    "Resume: loaded {warm_count} cuts per stage, \
                     resuming from iteration {completed}."
                ));
            }
        }
        cobre_io::PolicyMode::Fresh => {}
    }

    // Boundary cuts — orthogonal to policy mode. Runs after the match block so
    // that warm-start and boundary cuts compose correctly: warm-start replaces the
    // entire FCF first, then boundary cuts overwrite only the terminal pool.
    if let Some(bp) = root_config.and_then(|c| c.policy.boundary.as_ref()) {
        let boundary_path = ctx.output_dir.join(&bp.path);
        #[allow(clippy::cast_possible_truncation)]
        let state_dim = setup.fcf().state_dimension as u32;
        let boundary_records =
            cobre_sddp::load_boundary_cuts(&boundary_path, bp.source_stage, state_dim)
                .map_err(CliError::from)?;
        cobre_sddp::inject_boundary_cuts(setup, &boundary_records);
        if ctx.is_root && !ctx.quiet {
            let _ = ctx.stderr.write_line(&format!(
                "Boundary cuts: loaded {} cuts from stage {} of {}",
                boundary_records.len(),
                bp.source_stage,
                boundary_path.display()
            ));
        }
    }

    Ok(())
}

/// Load a policy checkpoint and build a synthetic `TrainingResult` for simulation-only mode.
fn load_policy_for_simulation(
    ctx: &RunContext<impl Communicator>,
    system: &System,
    setup: &mut StudySetup,
    root_config: Option<&cobre_io::Config>,
) -> Result<cobre_sddp::TrainingResult, CliError> {
    if ctx.is_root && !ctx.quiet {
        let _ = ctx
            .stderr
            .write_line("Training disabled. Loading policy for simulation-only mode...");
    }

    let policy_dir = ctx.output_dir.join(setup.policy_path());
    if !policy_dir.exists() {
        return Err(CliError::Internal {
            message: format!(
                "Policy directory not found: {}. Cannot run simulation-only \
                 mode without a trained policy.",
                policy_dir.display()
            ),
        });
    }

    let checkpoint = load_and_validate_checkpoint(&policy_dir, system, setup, root_config)?;

    let loaded_fcf = cobre_sddp::FutureCostFunction::from_deserialized(&checkpoint.stage_cuts)
        .map_err(CliError::from)?;
    setup.replace_fcf(loaded_fcf);

    let basis_cache =
        cobre_sddp::build_basis_cache_from_checkpoint(setup.num_stages(), &checkpoint.stage_bases);

    Ok(cobre_sddp::TrainingResult {
        iterations: checkpoint.metadata.completed_iterations.into(),
        final_lb: checkpoint.metadata.final_lower_bound,
        final_ub: checkpoint
            .metadata
            .best_upper_bound
            .unwrap_or(f64::INFINITY),
        final_ub_std: 0.0,
        final_gap: 0.0,
        total_time_ms: 0,
        reason: "loaded from checkpoint".to_string(),
        solver_stats_log: Vec::new(),
        basis_cache,
        visited_archive: None,
        // Baked templates are not stored in policy checkpoints; the
        // simulation path will use the fallback load_model + add_rows path.
        baked_templates: None,
    })
}

/// Set up the communicator, terminal, rayon pool, and resolve the output directory.
fn setup_communicator(args: &RunArgs) -> Result<RunContext<impl Communicator>, CliError> {
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

    // Gather topology while the concrete backend type is still in scope.
    // `comm.topology()` is non-collective and allocation-free after this point.
    let topology = comm.topology().clone();

    let n_threads = resolve_thread_count(args.threads);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .unwrap_or_else(|_| {
            tracing::warn!("rayon global thread pool already initialized; ignoring --threads");
        });

    let solver_version = cobre_solver::highs_version();

    if !quiet {
        crate::banner::print_banner(&stderr);
        crate::summary::print_execution_topology(
            &stderr,
            &topology,
            n_threads,
            "HiGHS",
            Some(&solver_version),
        );
    }

    let output_dir: PathBuf = args
        .output
        .clone()
        .unwrap_or_else(|| args.case_dir.join("output"));
    let term_width = crate::progress::resolve_term_width();

    Ok(RunContext {
        comm,
        is_root,
        quiet,
        n_threads,
        output_dir,
        term_width,
        stderr,
        topology,
        solver_version,
    })
}

/// Load the case on rank 0, broadcast system/config/tree, and build `StudySetup` on all ranks.
#[allow(clippy::too_many_lines)]
fn broadcast_and_build_setup(
    ctx: &RunContext<impl Communicator>,
    args: &RunArgs,
) -> Result<LoadBroadcastResult, CliError> {
    // Rank 0 loads from disk; system and config are broadcast to all ranks.
    let (
        raw_system,
        raw_bcast_config,
        mut root_config,
        root_stochastic,
        root_estimation_report,
        root_estimation_path,
        raw_bcast_tree,
        root_hydro_models,
        load_err,
    ) = if ctx.is_root {
        match load_case_and_config(args, ctx.quiet, &ctx.stderr) {
            Ok((prepared, hydro_models, bcast, config)) => {
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
                    estimation_path,
                } = prepared;
                (
                    Some(system),
                    Some(bcast),
                    Some(config),
                    Some(stochastic),
                    estimation_report,
                    Some(estimation_path),
                    Some(bcast_tree),
                    Some(hydro_models),
                    None,
                )
            }
            Err(e) => (None, None, None, None, None, None, None, None, Some(e)),
        }
    } else {
        (None, None, None, None, None, None, None, None, None)
    };
    let root_estimation_report: Option<EstimationReport> = root_estimation_report;
    let root_estimation_path: Option<cobre_sddp::EstimationPath> = root_estimation_path;

    let system_result = broadcast_value(raw_system, &ctx.comm);
    let bcast_config_result = broadcast_value(raw_bcast_config, &ctx.comm);
    let root_hydro_models: Option<PrepareHydroModelsResult> = root_hydro_models;

    let tree_result = broadcast_value(raw_bcast_tree, &ctx.comm);

    if let Some(e) = load_err {
        return Err(e);
    }
    let system = system_result?;
    let mut bcast_config = bcast_config_result?;

    let seed = bcast_config.seed;

    // Rank 0 uses the stochastic context already built by `prepare_stochastic`.
    // Non-root ranks reconstruct it from the broadcast system, opening tree,
    // and case directory. All factor entries (load, NCS) and the forward seed
    // must match rank 0's values for MPI reproducibility.
    let stochastic = if ctx.is_root {
        drop(tree_result);
        root_stochastic.ok_or_else(|| CliError::Internal {
            message: "stochastic context missing on rank 0 after successful load".to_string(),
        })?
    } else {
        let user_tree: Option<OpeningTree> =
            tree_result?.map(|bt| OpeningTree::from_parts(bt.data, bt.openings_per_stage, bt.dim));
        let training_src = &bcast_config.training_source;
        let forward_seed = training_src.seed.map(i64::unsigned_abs);

        let load_factor_entries =
            load_load_factors_for_stochastic(&args.case_dir).map_err(|e| CliError::Internal {
                message: format!("load factor error on non-root rank: {e}"),
            })?;
        let load_block_pairs: Vec<Vec<cobre_stochastic::normal::precompute::BlockFactorPair>> =
            load_factor_entries
                .iter()
                .map(|e| {
                    e.block_factors
                        .iter()
                        .map(|bf| (bf.block_id, bf.factor))
                        .collect()
                })
                .collect();
        let load_entity_factors: Vec<cobre_stochastic::normal::precompute::EntityFactorEntry<'_>> =
            load_factor_entries
                .iter()
                .zip(load_block_pairs.iter())
                .map(|(e, pairs)| (e.bus_id, e.stage_id, pairs.as_slice()))
                .collect();

        let ncs_raw = build_ncs_factor_entries(&system);
        let ncs_entity_factors: Vec<cobre_stochastic::normal::precompute::EntityFactorEntry<'_>> =
            ncs_raw
                .iter()
                .map(|(ncs_id, stage_id, pairs)| (*ncs_id, *stage_id, pairs.as_slice()))
                .collect();

        // Build HistoricalScenarioLibrary on non-root ranks when any stage
        // uses HistoricalResiduals (mirrors prepare_stochastic on rank 0).
        let opening_tree_library = {
            use cobre_core::temporal::NoiseMethod;

            let needs_historical_tree = system.stages().iter().any(|s| {
                s.id >= 0 && s.scenario_config.noise_method == NoiseMethod::HistoricalResiduals
            });

            if needs_historical_tree {
                let study_stages: Vec<_> = system
                    .stages()
                    .iter()
                    .filter(|s| s.id >= 0)
                    .cloned()
                    .collect();
                let hydro_ids: Vec<cobre_core::EntityId> =
                    system.hydros().iter().map(|h| h.id).collect();
                let par = cobre_stochastic::PrecomputedPar::build(
                    system.inflow_models(),
                    &study_stages,
                    &hydro_ids,
                )
                .map_err(|e| CliError::Internal {
                    message: format!("PAR build error on non-root rank: {e}"),
                })?;
                let max_order = par.max_order();
                let user_pool = training_src.historical_years.as_ref();
                let window_years = cobre_stochastic::discover_historical_windows(
                    system.inflow_history(),
                    &hydro_ids,
                    &study_stages,
                    max_order,
                    user_pool,
                    system.policy_graph().season_map.as_ref(),
                    1,
                )
                .map_err(|e| CliError::Internal {
                    message: format!("historical window discovery error on non-root rank: {e}"),
                })?;
                let mut lib = cobre_stochastic::HistoricalScenarioLibrary::new(
                    window_years.len(),
                    study_stages.len(),
                    hydro_ids.len(),
                    max_order,
                    window_years.clone(),
                );
                cobre_stochastic::standardize_historical_windows(
                    &mut lib,
                    system.inflow_history(),
                    &hydro_ids,
                    &study_stages,
                    &par,
                    &window_years,
                    system.policy_graph().season_map.as_ref(),
                );
                Some(lib)
            } else {
                None
            }
        };

        build_stochastic_context(
            &system,
            seed,
            forward_seed,
            &load_entity_factors,
            &ncs_entity_factors,
            OpeningTreeInputs {
                user_tree,
                historical_library: opening_tree_library.as_ref(),
                external_scenario_counts: None,
                // noise_group_ids: None for non-root ranks — the opened tree
                // is broadcast from rank 0 when auto-generated, so independent
                // noise per stage is acceptable here. Pattern C wiring for
                // non-root SAA tree generation is deferred to Epic 5.
                noise_group_ids: None,
            },
            cobre_stochastic::ClassSchemes {
                inflow: Some(training_src.inflow_scheme),
                load: Some(training_src.load_scheme),
                ncs: Some(training_src.ncs_scheme),
            },
        )
        .map_err(|e| CliError::Internal {
            message: format!("stochastic context error: {e}"),
        })?
    };

    // Rank 0 uses the hydro models result already built by `prepare_hydro_models`.
    // Non-root ranks reconstruct it independently from the system and case_dir.
    let hydro_models = if ctx.is_root {
        root_hydro_models.ok_or_else(|| CliError::Internal {
            message: "hydro models missing on rank 0 after successful load".to_string(),
        })?
    } else {
        prepare_hydro_models(&system, &args.case_dir).map_err(|e| CliError::Internal {
            message: format!("hydro model preprocessing error on non-root rank: {e}"),
        })?
    };

    let training_enabled = bcast_config.training_enabled;
    let policy_mode = bcast_config.policy_mode;
    let warm_start_basis_mode =
        cobre_solver::highs::WarmStartBasisMode::from(bcast_config.warm_start_basis_mode);
    let setup = build_study_setup(&system, &mut bcast_config, stochastic, hydro_models)?;

    Ok(LoadBroadcastResult {
        system,
        setup,
        root_config: root_config.take(),
        root_estimation_report,
        root_estimation_path,
        training_enabled,
        policy_mode,
        warm_start_basis_mode,
    })
}

/// Construct `StudySetup` on all ranks from broadcast parameters.
fn build_study_setup(
    system: &System,
    bcast_config: &mut BroadcastConfig,
    stochastic: cobre_stochastic::StochasticContext,
    hydro_models: PrepareHydroModelsResult,
) -> Result<StudySetup, CliError> {
    let stopping_rule_set = stopping_rules_from_broadcast(bcast_config);
    let cut_selection = std::mem::replace(
        &mut bcast_config.cut_selection,
        BroadcastCutSelection::Disabled,
    )
    .into_strategy();
    let config = ConstructionConfig {
        seed: bcast_config.seed,
        forward_passes: bcast_config.forward_passes,
        stopping_rule_set,
        n_scenarios: bcast_config.n_scenarios,
        io_channel_capacity: usize::try_from(bcast_config.io_channel_capacity).unwrap_or(64),
        policy_path: bcast_config.policy_path.clone(),
        inflow_method: bcast_config.inflow_method.clone(),
        cut_selection,
        cut_activity_tolerance: bcast_config.cut_activity_tolerance,
        budget: bcast_config.budget,
        export_states: bcast_config.export_states,
    };
    StudySetup::from_broadcast_params(
        system,
        stochastic,
        config,
        hydro_models,
        &bcast_config.training_source,
        &bcast_config.simulation_source,
    )
    .map_err(CliError::from)
}

/// Print summaries, export stochastic artifacts, and write the scaling report.
fn run_pre_training(
    ctx: &RunContext<impl Communicator>,
    system: &System,
    setup: &StudySetup,
    root_config: Option<&cobre_io::Config>,
    root_estimation_report: Option<&EstimationReport>,
    root_estimation_path: Option<cobre_sddp::EstimationPath>,
) -> Result<(), CliError> {
    if !ctx.quiet && ctx.is_root {
        let hydro_summary = build_hydro_model_summary(setup.hydro_models(), system);
        crate::summary::print_hydro_model_summary(&ctx.stderr, &hydro_summary);
    }

    // Build and emit provenance report.
    if ctx.is_root {
        if let Some(path) = root_estimation_path {
            let provenance = cobre_sddp::build_provenance_report(
                path,
                root_estimation_report,
                setup.stochastic().provenance(),
                system.hydros().len(),
            );
            if !ctx.quiet {
                crate::summary::print_provenance_summary(&ctx.stderr, &provenance);
            }
            let provenance_path = ctx.output_dir.join("training/model_provenance.json");
            cobre_io::write_provenance_report(&provenance_path, &provenance).map_err(|e| {
                CliError::Internal {
                    message: format!("failed to write provenance report: {e}"),
                }
            })?;
        }
    }

    if ctx.is_root && root_config.is_some_and(|c| c.exports.stochastic) {
        export_stochastic_artifacts(
            &ctx.output_dir,
            setup.stochastic(),
            system,
            root_estimation_report,
            ctx.quiet,
            &ctx.stderr,
        );
    }

    if ctx.is_root {
        let scaling_path = ctx.output_dir.join("training/scaling_report.json");
        cobre_io::write_scaling_report(&scaling_path, setup.scaling_report()).map_err(|e| {
            CliError::Internal {
                message: format!("failed to write scaling report: {e}"),
            }
        })?;
    }

    ctx.comm.barrier().map_err(|e| CliError::Internal {
        message: format!("post-export barrier error: {e}"),
    })?;

    Ok(())
}

/// Run training and collect results, events, and summary stats.
#[allow(clippy::too_many_lines)]
fn run_training_phase(
    ctx: &RunContext<impl Communicator>,
    setup: &mut StudySetup,
    warm_start_basis_mode: cobre_solver::highs::WarmStartBasisMode,
) -> Result<TrainingPhaseResult, CliError> {
    let solver_factory =
        move || HighsSolver::new().map(|s| s.with_warm_start_mode(warm_start_basis_mode));

    let mut solver = HighsSolver::new()
        .map_err(|e| CliError::Solver {
            message: format!("HiGHS initialisation failed: {e}"),
        })?
        .with_warm_start_mode(warm_start_basis_mode);

    let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();

    let quiet_rx: Option<mpsc::Receiver<TrainingEvent>>;
    let progress_handle = if ctx.quiet {
        quiet_rx = Some(event_rx);
        None
    } else {
        quiet_rx = None;
        Some(crate::progress::run_progress_thread(
            event_rx,
            setup.max_iterations(),
            ctx.term_width,
        ))
    };

    let training_outcome = match setup.train(
        &mut solver,
        &ctx.comm,
        ctx.n_threads,
        solver_factory,
        Some(event_tx),
        None,
    ) {
        Ok(outcome) => outcome,
        Err(e) => {
            if let Some(handle) = progress_handle {
                let _ = handle.join();
            }
            return Err(CliError::from(e));
        }
    };
    let training_result = training_outcome.result;

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
    ctx.comm
        .allreduce(&[local_lp_solves], &mut global_lp_solves, ReduceOp::Sum)
        .map_err(|e| CliError::Internal {
            message: format!("LP solve count allreduce error: {e}"),
        })?;
    let global_lp_solves = global_lp_solves[0];

    ctx.comm.barrier().map_err(|e| CliError::Internal {
        message: format!("post-training barrier error: {e}"),
    })?;

    // Aggregate solver stats from the stats log and allreduce across ranks.
    let (
        local_first_try,
        local_retried,
        local_failed,
        local_solve_time_s,
        local_basis_offered,
        local_basis_rejections,
        local_basis_non_alien_rejections,
        local_simplex_iter,
    ) = aggregate_solver_stats(&training_result.solver_stats_log);

    #[allow(clippy::cast_precision_loss)]
    let send_stats = [
        local_first_try as f64,
        local_retried as f64,
        local_failed as f64,
        local_solve_time_s,
        local_basis_offered as f64,
        local_basis_rejections as f64,
        local_basis_non_alien_rejections as f64,
        local_simplex_iter as f64,
    ];
    let mut recv_stats = [0.0_f64; 8];
    ctx.comm
        .allreduce(&send_stats, &mut recv_stats, ReduceOp::Sum)
        .map_err(|e| CliError::Internal {
            message: format!("training solver stats allreduce error: {e}"),
        })?;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (
        total_first_try,
        total_retried,
        total_failed,
        total_solve_time_s,
        total_basis_offered,
        total_basis_rejections,
        total_basis_non_alien_rejections,
        total_simplex_iter,
    ) = (
        recv_stats[0] as u64,
        recv_stats[1] as u64,
        recv_stats[2] as u64,
        recv_stats[3],
        recv_stats[4] as u64,
        recv_stats[5] as u64,
        recv_stats[6] as u64,
        recv_stats[7] as u64,
    );

    // Print training summary on rank 0.
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
        total_first_try: Some(total_first_try),
        total_retried: Some(total_retried),
        total_failed: Some(total_failed),
        total_solve_time_seconds: Some(total_solve_time_s),
        total_basis_offered: Some(total_basis_offered),
        total_basis_rejections: Some(total_basis_rejections),
        total_basis_non_alien_rejections: Some(total_basis_non_alien_rejections),
        total_simplex_iterations: Some(total_simplex_iter),
    };
    if !ctx.quiet && ctx.is_root {
        crate::summary::print_training_summary(&ctx.stderr, &training_summary);
    }

    Ok(TrainingPhaseResult {
        result: training_result,
        output: training_output,
        error: training_outcome.error,
    })
}

/// Aggregate solver statistics from the training stats log.
fn aggregate_solver_stats(
    stats_log: &[(u64, &'static str, i32, cobre_sddp::SolverStatsDelta)],
) -> (u64, u64, u64, f64, u64, u64, u64, u64) {
    let mut first_try = 0u64;
    let mut retried = 0u64;
    let mut failed = 0u64;
    let mut solve_time = 0.0_f64;
    let mut basis_offered = 0u64;
    let mut basis_rejections = 0u64;
    let mut basis_non_alien_rejections = 0u64;
    let mut simplex = 0u64;
    for (_, _, _, delta) in stats_log {
        first_try += delta.first_try_successes;
        retried += delta.lp_successes.saturating_sub(delta.first_try_successes);
        failed += delta.lp_failures;
        solve_time += delta.solve_time_ms;
        basis_offered += delta.basis_offered;
        basis_rejections += delta.basis_rejections;
        basis_non_alien_rejections += delta.basis_non_alien_rejections;
        simplex += delta.simplex_iterations;
    }
    (
        first_try,
        retried,
        failed,
        solve_time / 1000.0,
        basis_offered,
        basis_rejections,
        basis_non_alien_rejections,
        simplex,
    )
}

/// Run the simulation phase: workspace pool, Parquet writing, and output.
fn run_simulation_phase(
    ctx: &RunContext<impl Communicator>,
    system: &System,
    setup: &mut StudySetup,
    training_result: &cobre_sddp::TrainingResult,
    hostname: &str,
) -> Result<(), CliError> {
    let solver_factory = HighsSolver::new;
    let n_scenarios = setup.n_scenarios();
    let sim_config = setup.simulation_config();

    let mut sim_pool = setup
        .create_workspace_pool(ctx.n_threads, solver_factory)
        .map_err(|e| CliError::Solver {
            message: format!("HiGHS initialisation failed for simulation pool: {e}"),
        })?;

    let (sim_event_tx, sim_event_rx) = mpsc::channel::<TrainingEvent>();
    let sim_progress_handle = if ctx.quiet {
        drop(sim_event_rx);
        None
    } else {
        Some(crate::progress::run_progress_thread(
            sim_event_rx,
            u64::from(n_scenarios),
            ctx.term_width,
        ))
    };

    let io_capacity = sim_config.io_channel_capacity;
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity.max(1));

    let parquet_config = cobre_io::ParquetWriterConfig::default();
    let mut sim_writer = cobre_io::output::simulation_writer::SimulationParquetWriter::new(
        &ctx.output_dir,
        system,
        &parquet_config,
    )
    .map_err(CliError::from)?;

    // The drain thread writes scenarios directly to Parquet instead of
    // collecting into a Vec. This avoids the need to gather all results
    // on rank 0 via MPI (which overflows i32 on large cases).
    let drain_handle = std::thread::spawn(move || {
        let mut failed: u32 = 0;
        for scenario_result in result_rx {
            let payload =
                cobre_io::output::simulation_writer::ScenarioWritePayload::from(scenario_result);
            if let Err(e) = sim_writer.write_scenario(payload) {
                tracing::error!("simulation write error: {e}");
                failed += 1;
            }
        }
        (sim_writer, failed)
    });

    let sim_started_at = cobre_io::now_iso8601();
    let sim_start = std::time::Instant::now();

    let sim_result = setup
        .simulate(
            &mut sim_pool.workspaces,
            &ctx.comm,
            &result_tx,
            Some(sim_event_tx),
            training_result.baked_templates.as_deref(),
            &training_result.basis_cache,
        )
        .map_err(CliError::from);
    if let Some(handle) = sim_progress_handle {
        let _ = handle.join();
    }

    drop(result_tx);

    #[allow(clippy::expect_used)]
    let (sim_writer, write_failures) = drain_handle.join().expect("drain thread panicked");

    let sim_run_result = sim_result?;

    #[allow(clippy::cast_possible_truncation)]
    let sim_time_ms = sim_start.elapsed().as_millis() as u64;

    let mut local_sim_output = sim_writer.finalize(sim_time_ms);
    local_sim_output.failed = write_failures;

    let merged_sim_output = merge_simulation_metadata(&ctx.comm, &local_sim_output)?;

    ctx.comm.barrier().map_err(|e| CliError::Internal {
        message: format!("post-simulation barrier error: {e}"),
    })?;

    // Aggregate simulation solver stats across all MPI ranks.
    let (global_agg, global_scenario_stats) =
        aggregate_simulation_solver_stats(&ctx.comm, &sim_run_result.solver_stats)?;

    // Aggregate simulation cost statistics across all MPI ranks so that
    // the printed mean/std/CI95 reflect ALL scenarios, not just rank 0's.
    let cost_summary =
        cobre_sddp::aggregate_simulation(&sim_run_result.costs, &sim_config, &ctx.comm).map_err(
            |e| CliError::Internal {
                message: format!("simulation cost aggregation error: {e}"),
            },
        )?;

    if !ctx.quiet && ctx.is_root {
        print_sim_summary(
            &ctx.stderr,
            n_scenarios,
            sim_time_ms,
            &global_agg,
            &cost_summary,
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
    merged_sim_output: &cobre_io::SimulationOutput,
    global_scenario_stats: &[(u32, cobre_sddp::SolverStatsDelta)],
) -> Result<(), CliError> {
    let mpi_world_size = u32::try_from(ctx.topology.world_size).unwrap_or(u32::MAX);
    let sim_ctx = cobre_io::OutputContext {
        hostname: hostname.to_string(),
        solver: "highs".to_string(),
        solver_version: None,
        started_at: sim_started_at,
        completed_at: cobre_io::now_iso8601(),
        distribution: build_distribution_info(&ctx.topology, ctx.n_threads, mpi_world_size),
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

/// Build a [`cobre_io::DistributionInfo`] from the cached execution topology.
///
/// `ranks_participated` is the number of MPI ranks that actively contributed
/// to the computation (may differ from `world_size` if some ranks were idle).
fn build_distribution_info(
    topology: &ExecutionTopology,
    n_threads: usize,
    ranks_participated: u32,
) -> cobre_io::DistributionInfo {
    use cobre_comm::BackendKind;
    cobre_io::DistributionInfo {
        backend: match topology.backend {
            BackendKind::Mpi => "mpi",
            BackendKind::Local => "local",
            BackendKind::Auto => "unknown",
        }
        .to_string(),
        world_size: u32::try_from(topology.world_size).unwrap_or(u32::MAX),
        ranks_participated,
        num_nodes: u32::try_from(topology.num_hosts()).unwrap_or(u32::MAX),
        threads_per_rank: u32::try_from(n_threads).unwrap_or(u32::MAX),
        mpi_library: topology.mpi.as_ref().map(|m| m.library_version.clone()),
        mpi_standard: topology.mpi.as_ref().map(|m| m.standard_version.clone()),
        thread_level: topology.mpi.as_ref().map(|m| m.thread_level.clone()),
        slurm_job_id: topology.slurm.as_ref().map(|s| s.job_id.clone()),
    }
}

/// Print the simulation summary from aggregated solver stats and cost statistics.
fn print_sim_summary(
    stderr: &Term,
    n_scenarios: u32,
    sim_time_ms: u64,
    agg: &cobre_sddp::SolverStatsDelta,
    cost_summary: &cobre_sddp::SimulationSummary,
) {
    crate::summary::print_simulation_summary(
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
            total_basis_offered: Some(agg.basis_offered),
            total_basis_rejections: Some(agg.basis_rejections),
            total_basis_non_alien_rejections: Some(agg.basis_non_alien_rejections),
            total_simplex_iterations: Some(agg.simplex_iterations),
        },
    );
}

/// Merge per-rank simulation metadata using lightweight MPI collective operations.
///
/// Each rank contributes its local [`SimulationOutput`](cobre_io::SimulationOutput) from
/// the distributed Parquet writing phase. The merge uses:
///
/// - `allreduce(Sum)` for scenario counts (`n_scenarios`, `completed`, `failed`)
/// - `allreduce(Max)` for `total_time_ms` (wall-clock = slowest rank)
/// - `allgatherv` for partition path strings (newline-delimited UTF-8)
///
/// For single-rank runs (local communicator), this is equivalent to a passthrough.
#[allow(clippy::cast_possible_truncation)]
fn merge_simulation_metadata<C: Communicator>(
    comm: &C,
    local: &cobre_io::SimulationOutput,
) -> Result<cobre_io::SimulationOutput, CliError> {
    // Scalar counts: allreduce(Sum) on [n_scenarios, completed, failed].
    let send_counts = [local.n_scenarios, local.completed, local.failed];
    let mut merged_counts = [0u32; 3];
    comm.allreduce(&send_counts, &mut merged_counts, ReduceOp::Sum)
        .map_err(|e| CliError::Internal {
            message: format!("simulation metadata count allreduce error: {e}"),
        })?;

    // Wall-clock time: allreduce(Max) — slowest rank determines total time.
    let send_time = [local.total_time_ms];
    let mut merged_time = [0u64; 1];
    comm.allreduce(&send_time, &mut merged_time, ReduceOp::Max)
        .map_err(|e| CliError::Internal {
            message: format!("simulation metadata time allreduce error: {e}"),
        })?;

    // Partition paths: encode as newline-delimited UTF-8, exchange via allgatherv.
    let local_paths_bytes = local.partitions_written.join("\n").into_bytes();

    // Exchange per-rank byte counts.
    let send_len = [local_paths_bytes.len() as u64];
    let n_ranks = comm.size();
    let mut all_lens = vec![0u64; n_ranks];
    let len_counts: Vec<usize> = vec![1; n_ranks];
    let len_displs: Vec<usize> = (0..n_ranks).collect();
    comm.allgatherv(&send_len, &mut all_lens, &len_counts, &len_displs)
        .map_err(|e| CliError::Internal {
            message: format!("partition path length exchange error: {e}"),
        })?;

    // Exchange path bytes.
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

    // Parse received bytes back into path strings.
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

    Ok(cobre_io::SimulationOutput {
        n_scenarios: merged_counts[0],
        completed: merged_counts[1],
        failed: merged_counts[2],
        total_time_ms: merged_time[0],
        partitions_written: all_partitions,
    })
}

/// Aggregate simulation solver statistics across all MPI ranks.
///
/// Returns:
/// - The global [`cobre_sddp::SolverStatsDelta`] aggregate (sum of all ranks' local
///   aggregates), used for the simulation summary printed on root.
/// - A [`Vec<(u32, cobre_sddp::SolverStatsDelta)>`] containing one entry per global
///   scenario (gathered from all ranks), sorted by scenario ID, used for Parquet output.
///
/// ## Protocol
///
/// **Summary (F1-002)**: Each rank first computes its local aggregate via
/// [`cobre_sddp::SolverStatsDelta::aggregate`], packs the 15 scalar fields into a
/// `[f64; 15]` buffer, and calls `allreduce(Sum)`. Root reconstructs the global
/// aggregate from the received buffer.
///
/// **Per-scenario gather (F1-003)**: Each rank packs its per-scenario stats into a
/// flat `f64` buffer (16 values per scenario: `scenario_id` + 15 scalar fields).
/// An `allgatherv` collects all scenarios on all ranks. Root sorts the result by
/// scenario ID. All ranks participate but only root uses the gathered data.
///
/// ## Single-rank behaviour
///
/// With `LocalBackend` (single rank), both operations are identity pass-throughs:
/// the allreduce returns the local values unchanged, and the allgatherv returns
/// the local buffer unchanged. The result is identical to calling
/// `SolverStatsDelta::aggregate` directly.
#[allow(clippy::cast_possible_truncation)]
fn aggregate_simulation_solver_stats<C: Communicator>(
    comm: &C,
    local_stats: &[(u32, cobre_sddp::SolverStatsDelta)],
) -> Result<
    (
        cobre_sddp::SolverStatsDelta,
        Vec<(u32, cobre_sddp::SolverStatsDelta)>,
    ),
    CliError,
> {
    // ── Part A: summary allreduce (F1-002) ────────────────────────────────────
    let local_agg = cobre_sddp::SolverStatsDelta::aggregate(local_stats.iter().map(|(_, d)| d));
    let send_scalars = cobre_sddp::pack_delta_scalars(&local_agg);
    let mut recv_scalars = [0.0_f64; cobre_sddp::SOLVER_STATS_DELTA_SCALAR_FIELDS];
    comm.allreduce(&send_scalars, &mut recv_scalars, ReduceOp::Sum)
        .map_err(|e| CliError::Internal {
            message: format!("simulation solver stats allreduce error: {e}"),
        })?;
    let global_agg = cobre_sddp::unpack_delta_scalars(&recv_scalars);

    // ── Part B: per-scenario allgatherv (F1-003) ──────────────────────────────
    let n_ranks = comm.size();
    let local_buf = cobre_sddp::pack_scenario_stats(local_stats);
    let local_count = local_buf.len();

    // Step 1: exchange per-rank buffer lengths.
    let send_len = [local_count as u64];
    let mut all_lens = vec![0u64; n_ranks];
    let len_counts: Vec<usize> = vec![1; n_ranks];
    let len_displs: Vec<usize> = (0..n_ranks).collect();
    comm.allgatherv(&send_len, &mut all_lens, &len_counts, &len_displs)
        .map_err(|e| CliError::Internal {
            message: format!("simulation solver stats length exchange error: {e}"),
        })?;

    // Step 2: gather all per-scenario buffers.
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

    // Step 3: unpack and sort by scenario ID.
    let mut global_scenario_stats = cobre_sddp::unpack_scenario_stats(&all_buf);
    global_scenario_stats.sort_by_key(|(id, _)| *id);

    Ok((global_agg, global_scenario_stats))
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
        basis_non_alien_rejections: delta.basis_non_alien_rejections as u32,
        simplex_iterations: delta.simplex_iterations,
        solve_time_ms: delta.solve_time_ms,
        load_model_time_ms: delta.load_model_time_ms,
        add_rows_time_ms: delta.add_rows_time_ms,
        set_bounds_time_ms: delta.set_bounds_time_ms,
        basis_set_time_ms: delta.basis_set_time_ms,
        basis_preserved: delta.basis_preserved,
        basis_new_tight: delta.basis_new_tight,
        basis_new_slack: delta.basis_new_slack,
        basis_demotions: delta.basis_demotions,
        retry_level_histogram: delta.retry_level_histogram.clone(),
    }
}

/// Arguments for [`write_training_outputs`].
struct WriteTrainingArgs<'a> {
    output_dir: &'a Path,
    system: &'a System,
    config: &'a cobre_io::Config,
    training_output: &'a cobre_io::TrainingOutput,
    setup: &'a StudySetup,
    training_result: &'a cobre_sddp::TrainingResult,
    output_ctx: &'a cobre_io::OutputContext,
    quiet: bool,
    stderr: &'a Term,
}

/// Write training artifacts: policy checkpoint, training results, solver stats,
/// and cut selection records. Called immediately after training completes, before
/// simulation starts.
fn write_training_outputs(args: &WriteTrainingArgs<'_>) -> Result<(), CliError> {
    if !args.quiet {
        use std::io::Write;
        let _ = args.stderr.write_line("Writing training outputs...");
        let _ = std::io::stderr().flush();
    }
    let write_start = std::time::Instant::now();

    let policy_dir = args.output_dir.join(args.setup.policy_path());
    crate::policy_io::write_checkpoint(
        &policy_dir,
        args.setup.fcf(),
        args.training_result,
        &crate::policy_io::CheckpointParams {
            max_iterations: args.setup.max_iterations(),
            forward_passes: args.setup.forward_passes(),
            seed: args.setup.seed(),
            export_states: args.config.exports.states,
        },
    )?;

    cobre_io::write_training_results(
        args.output_dir,
        args.training_output,
        args.system,
        args.config,
        args.output_ctx,
    )
    .map_err(CliError::from)?;

    // Write training solver stats to training/solver/iterations.parquet.
    if !args.training_result.solver_stats_log.is_empty() {
        let rows: Vec<cobre_io::SolverStatsRow> = args
            .training_result
            .solver_stats_log
            .iter()
            .map(|(iter, phase, stage, delta)| {
                #[allow(clippy::cast_possible_truncation)] // iteration count fits in u32
                delta_to_stats_row(*iter as u32, phase, *stage, delta)
            })
            .collect();
        cobre_io::write_solver_stats(args.output_dir, &rows).map_err(CliError::from)?;
    }

    // Write per-stage cut selection records to training/cut_selection/iterations.parquet.
    if !args.training_output.cut_selection_records.is_empty() {
        let parquet_config = cobre_io::ParquetWriterConfig::default();
        cobre_io::write_cut_selection_records(
            args.output_dir,
            &args.training_output.cut_selection_records,
            &parquet_config,
        )
        .map_err(CliError::from)?;
    }

    if !args.quiet {
        let write_secs = write_start.elapsed().as_secs_f64();
        crate::summary::print_output_path(args.stderr, args.output_dir, write_secs);
    }

    Ok(())
}

/// Arguments for [`write_simulation_outputs`].
struct WriteSimulationArgs<'a> {
    output_dir: &'a Path,
    sim_output: &'a cobre_io::SimulationOutput,
    sim_solver_stats: &'a [(u32, cobre_sddp::SolverStatsDelta)],
    output_ctx: &'a cobre_io::OutputContext,
    quiet: bool,
    stderr: &'a Term,
}

/// Write simulation artifacts: simulation results manifest and solver stats.
/// Called after simulation completes.
fn write_simulation_outputs(args: &WriteSimulationArgs<'_>) -> Result<(), CliError> {
    if !args.quiet {
        use std::io::Write;
        let _ = args.stderr.write_line("Writing simulation outputs...");
        let _ = std::io::stderr().flush();
    }
    let write_start = std::time::Instant::now();

    cobre_io::write_simulation_results(args.output_dir, args.sim_output, args.output_ctx)
        .map_err(CliError::from)?;

    // Write simulation solver stats to simulation/solver/iterations.parquet.
    if !args.sim_solver_stats.is_empty() {
        let rows: Vec<cobre_io::SolverStatsRow> = args
            .sim_solver_stats
            .iter()
            .map(|(scenario_id, delta)| delta_to_stats_row(*scenario_id, "simulation", -1, delta))
            .collect();
        cobre_io::write_simulation_solver_stats(args.output_dir, &rows).map_err(CliError::from)?;
    }

    if !args.quiet {
        let write_secs = write_start.elapsed().as_secs_f64();
        crate::summary::print_output_path(args.stderr, args.output_dir, write_secs);
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
