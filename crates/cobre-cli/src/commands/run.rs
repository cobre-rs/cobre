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
//! 5. Train the SDDP policy (`cobre_sddp::train`).
//! 6. Optionally run simulation (`cobre_sddp::simulate`).
//! 7. Write all outputs (`cobre_io::write_results`).

use std::path::PathBuf;
use std::sync::mpsc;

use clap::Args;
use console::Term;

use cobre_comm::{create_communicator, Communicator, ReduceOp};
use cobre_core::TrainingEvent;
use cobre_io::write_results;
use cobre_sddp::estimation::estimate_from_history;
use cobre_sddp::{
    build_stage_templates, build_training_output, simulate, train, EntityCounts,
    FutureCostFunction, HorizonMode, InflowNonNegativityMethod, RiskMeasure, SimulationConfig,
    StageIndexer, StoppingMode, StoppingRule, StoppingRuleSet, TrainingConfig, WorkspacePool,
};
use cobre_solver::HighsSolver;
use cobre_stochastic::build_stochastic_context;

use crate::error::CliError;
use crate::summary::{SimulationSummary, TrainingSummary};

/// Default number of forward-pass trajectories when not specified in config.
const DEFAULT_FORWARD_PASSES: u32 = 1;

/// Default maximum iterations when no stopping rule specifies an iteration limit.
const DEFAULT_MAX_ITERATIONS: u64 = 100;

/// Default random seed for stochastic scenario generation.
const DEFAULT_SEED: u64 = 42;

// ---------------------------------------------------------------------------
// BroadcastConfig — postcard-safe configuration snapshot for MPI broadcast
// ---------------------------------------------------------------------------

/// Postcard-serializable stopping rule (external-tagged for serialization).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum BroadcastStoppingRule {
    IterationLimit { limit: u64 },
    TimeLimit { seconds: f64 },
    BoundStalling { iterations: u64, tolerance: f64 },
}

/// Postcard-serializable stopping mode.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum BroadcastStoppingMode {
    Any,
    All,
}

/// Postcard-safe cut selection strategy.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
enum BroadcastCutSelection {
    Disabled,
    Level1 {
        threshold: u64,
        check_frequency: u64,
    },
    Lml1 {
        memory_window: u64,
        check_frequency: u64,
    },
    Dominated {
        threshold: f64,
        check_frequency: u64,
    },
}

impl BroadcastCutSelection {
    fn from_strategy(strategy: Option<&cobre_sddp::CutSelectionStrategy>) -> Self {
        use cobre_sddp::CutSelectionStrategy;
        match strategy {
            None => Self::Disabled,
            Some(CutSelectionStrategy::Level1 {
                threshold,
                check_frequency,
            }) => Self::Level1 {
                threshold: *threshold,
                check_frequency: *check_frequency,
            },
            Some(CutSelectionStrategy::Lml1 {
                memory_window,
                check_frequency,
            }) => Self::Lml1 {
                memory_window: *memory_window,
                check_frequency: *check_frequency,
            },
            Some(CutSelectionStrategy::Dominated {
                threshold,
                check_frequency,
            }) => Self::Dominated {
                threshold: *threshold,
                check_frequency: *check_frequency,
            },
        }
    }

    fn into_strategy(self) -> Option<cobre_sddp::CutSelectionStrategy> {
        use cobre_sddp::CutSelectionStrategy;
        match self {
            Self::Disabled => None,
            Self::Level1 {
                threshold,
                check_frequency,
            } => Some(CutSelectionStrategy::Level1 {
                threshold,
                check_frequency,
            }),
            Self::Lml1 {
                memory_window,
                check_frequency,
            } => Some(CutSelectionStrategy::Lml1 {
                memory_window,
                check_frequency,
            }),
            Self::Dominated {
                threshold,
                check_frequency,
            } => Some(CutSelectionStrategy::Dominated {
                threshold,
                check_frequency,
            }),
        }
    }
}

/// Configuration snapshot broadcast from rank 0 to all ranks.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct BroadcastConfig {
    seed: u64,
    forward_passes: u32,
    stopping_rules: Vec<BroadcastStoppingRule>,
    stopping_mode: BroadcastStoppingMode,
    should_simulate: bool,
    n_scenarios: u32,
    io_channel_capacity: u32,
    policy_path: String,
    inflow_method: InflowNonNegativityMethod,
    cut_selection: BroadcastCutSelection,
}

impl BroadcastConfig {
    fn from_config(config: &cobre_io::Config, skip_simulation: bool) -> Result<Self, CliError> {
        use cobre_io::config::StoppingRuleConfig;

        if config.training.seed.is_none() {
            eprintln!(
                "warning: no random seed specified in config.json (training.seed); \
                 using default seed 42. Set training.seed for reproducible results."
            );
        }
        let seed = config.training.seed.map_or(DEFAULT_SEED, i64::unsigned_abs);

        let forward_passes = config
            .training
            .forward_passes
            .unwrap_or(DEFAULT_FORWARD_PASSES);

        let rule_configs = match &config.training.stopping_rules {
            Some(rules) if !rules.is_empty() => rules.clone(),
            _ => vec![StoppingRuleConfig::IterationLimit {
                limit: u32::try_from(DEFAULT_MAX_ITERATIONS).unwrap_or(u32::MAX),
            }],
        };

        let stopping_rules = rule_configs
            .into_iter()
            .map(|c| match c {
                StoppingRuleConfig::IterationLimit { limit } => {
                    BroadcastStoppingRule::IterationLimit {
                        limit: u64::from(limit),
                    }
                }
                StoppingRuleConfig::TimeLimit { seconds } => {
                    BroadcastStoppingRule::TimeLimit { seconds }
                }
                StoppingRuleConfig::BoundStalling {
                    iterations,
                    tolerance,
                } => BroadcastStoppingRule::BoundStalling {
                    iterations: u64::from(iterations),
                    tolerance,
                },
                StoppingRuleConfig::Simulation { .. } => {
                    // Simulation stopping rule requires upper-bound evaluation;
                    // not implemented in the MVP — fold into iteration limit.
                    BroadcastStoppingRule::IterationLimit {
                        limit: DEFAULT_MAX_ITERATIONS,
                    }
                }
            })
            .collect();

        let stopping_mode = if config.training.stopping_mode.eq_ignore_ascii_case("all") {
            BroadcastStoppingMode::All
        } else {
            BroadcastStoppingMode::Any
        };

        let should_simulate =
            !skip_simulation && config.simulation.enabled && config.simulation.num_scenarios > 0;

        // Parse and validate the cut selection config on rank 0.
        // Errors here are caught before the broadcast so all ranks get a valid
        // BroadcastConfig (or the run is aborted on rank 0 before broadcast).
        let parsed_cut_selection =
            cobre_sddp::parse_cut_selection_config(&config.training.cut_selection)
                .map_err(|msg| CliError::Validation { report: msg })?;
        let cut_selection = BroadcastCutSelection::from_strategy(parsed_cut_selection.as_ref());

        Ok(Self {
            seed,
            forward_passes,
            stopping_rules,
            stopping_mode,
            should_simulate,
            n_scenarios: config.simulation.num_scenarios,
            io_channel_capacity: config.simulation.io_channel_capacity,
            policy_path: config.policy.path.clone(),
            inflow_method: InflowNonNegativityMethod::from(&config.modeling.inflow_non_negativity),
            cut_selection,
        })
    }
}

fn stopping_rules_from_broadcast(cfg: &BroadcastConfig) -> StoppingRuleSet {
    let rules = cfg
        .stopping_rules
        .iter()
        .map(|r| match r {
            BroadcastStoppingRule::IterationLimit { limit } => {
                StoppingRule::IterationLimit { limit: *limit }
            }
            BroadcastStoppingRule::TimeLimit { seconds } => {
                StoppingRule::TimeLimit { seconds: *seconds }
            }
            BroadcastStoppingRule::BoundStalling {
                iterations,
                tolerance,
            } => StoppingRule::BoundStalling {
                iterations: *iterations,
                tolerance: *tolerance,
            },
        })
        .collect();

    let mode = match cfg.stopping_mode {
        BroadcastStoppingMode::All => StoppingMode::All,
        BroadcastStoppingMode::Any => StoppingMode::Any,
    };

    StoppingRuleSet { rules, mode }
}

/// Arguments for the `cobre run` subcommand.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Args)]
#[command(about = "Load a case directory, train an SDDP policy, and run simulation")]
pub struct RunArgs {
    /// Path to the case directory containing the input data files.
    pub case_dir: PathBuf,

    /// Output directory for results (defaults to `<CASE_DIR>/output/`).
    #[arg(long, value_name = "DIR")]
    pub output: Option<PathBuf>,

    /// Train only; skip the simulation phase.
    #[arg(long)]
    pub skip_simulation: bool,

    /// Suppress the banner and progress bars. Errors still go to stderr.
    #[arg(long)]
    pub quiet: bool,

    /// Suppress the banner but keep progress bars.
    #[arg(long)]
    pub no_banner: bool,

    /// Increase the tracing log level (debug for `cobre_cli`, info for library crates).
    #[arg(long)]
    pub verbose: bool,

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

/// Broadcast a serializable value from rank 0 to all ranks.
///
/// Serializes on rank 0, broadcasts length and bytes. Non-rank-0 deserializes.
/// A length of 0 signals failure on rank 0, allowing all ranks to participate.
///
/// # Errors
///
/// Returns [`CliError::Internal`] on serialization, broadcast, or deserialization failure.
fn broadcast_value<T, C>(value: Option<T>, comm: &C) -> Result<T, CliError>
where
    T: serde::Serialize + serde::de::DeserializeOwned,
    C: cobre_comm::Communicator,
{
    let is_root = comm.rank() == 0;

    let serialized: Vec<u8> = if is_root {
        match value {
            Some(ref v) => postcard::to_allocvec(v).map_err(|e| CliError::Internal {
                message: format!("serialization error: {e}"),
            })?,
            None => Vec::new(),
        }
    } else {
        Vec::new()
    };

    let raw_len = serialized.len();
    #[allow(clippy::cast_possible_truncation)]
    let mut len_buf = [raw_len as u64];
    comm.broadcast(&mut len_buf, 0)
        .map_err(|e| CliError::Internal {
            message: format!("broadcast error (length): {e}"),
        })?;

    let len = usize::try_from(len_buf[0]).map_err(|e| CliError::Internal {
        message: format!("broadcast error (length conversion): {e}"),
    })?;
    if len == 0 {
        return Err(CliError::Internal {
            message: "rank 0 signaled broadcast failure (length 0)".to_string(),
        });
    }

    let mut bytes = if is_root { serialized } else { vec![0u8; len] };
    comm.broadcast(&mut bytes, 0)
        .map_err(|e| CliError::Internal {
            message: format!("broadcast error (data): {e}"),
        })?;

    if is_root {
        value.ok_or_else(|| CliError::Internal {
            message: "broadcast_value: root value disappeared after serialization".to_string(),
        })
    } else {
        postcard::from_bytes(&bytes).map_err(|e| CliError::Internal {
            message: format!("deserialization error: {e}"),
        })
    }
}

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
) -> Result<(cobre_core::System, BroadcastConfig, cobre_io::Config), CliError> {
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
    let system =
        estimate_from_history(system, &args.case_dir, &config).map_err(|e| CliError::Internal {
            message: format!("estimation error: {e}"),
        })?;
    let bcast = BroadcastConfig::from_config(&config, args.skip_simulation)?;
    Ok((system, bcast, config))
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

    // Non-root ranks are always quiet: they produce no terminal output and
    // write no files. This single flag controls all UI/banner/summary paths.
    let quiet = args.quiet || !is_root;

    // Under MPI, mpiexec pipes rank 0's stderr through to the user's terminal
    // without allocating a PTY. Force color and terminal rendering on rank 0
    // so the banner and progress bars display correctly.
    let mpi_active = comm.size() > 1;
    if mpi_active && is_root && !args.quiet {
        console::set_colors_enabled_stderr(true);
    }

    let stderr = Term::stderr();

    if !quiet && !args.no_banner {
        crate::banner::print_banner(&stderr);
    }

    let n_threads = resolve_thread_count(args.threads);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .unwrap_or_else(|_| {
            tracing::warn!("rayon global thread pool already initialized; ignoring --threads");
        });

    // Only rank 0 accesses the filesystem. Config is converted to BroadcastConfig
    // (postcard-safe) and broadcast. Root config stays on rank 0 for output writing.
    let (raw_system, raw_bcast_config, root_config, load_err) = if is_root {
        match load_case_and_config(&args, quiet, &stderr) {
            Ok((system, bcast, config)) => (Some(system), Some(bcast), Some(config), None),
            Err(e) => (None, None, None, Some(e)),
        }
    } else {
        (None, None, None, None)
    };

    let system_result = broadcast_value(raw_system, &comm);
    let bcast_config_result = broadcast_value(raw_bcast_config, &comm);

    if let Some(e) = load_err {
        return Err(e);
    }
    let system = system_result?;
    let mut bcast_config = bcast_config_result?;

    let seed = bcast_config.seed;
    let stochastic =
        build_stochastic_context(&system, seed, &[]).map_err(|e| CliError::Internal {
            message: format!("stochastic context error: {e}"),
        })?;

    let stage_templates = build_stage_templates(
        &system,
        &bcast_config.inflow_method,
        stochastic.par_lp(),
        stochastic.normal_lp(),
    )
    .map_err(|e| CliError::Validation {
        report: e.to_string(),
    })?;
    if stage_templates.templates.is_empty() {
        return Err(CliError::Validation {
            report: "system has no study stages — cannot train".to_string(),
        });
    }
    let stage_templates_ref = &stage_templates.templates;
    let base_rows = &stage_templates.base_rows;
    let noise_scale = &stage_templates.noise_scale;
    let zeta_per_stage = &stage_templates.zeta_per_stage;
    let block_hours_per_stage = &stage_templates.block_hours_per_stage;
    let n_hydros_lp = stage_templates.n_hydros;

    let n_blks_stage0 = system.stages().first().map_or(1, |s| s.blocks.len().max(1));
    let has_inflow_penalty =
        bcast_config.inflow_method.has_slack_columns() && stage_templates_ref[0].n_hydro > 0;
    let indexer = StageIndexer::with_equipment(
        stage_templates_ref[0].n_hydro,
        stage_templates_ref[0].max_par_order,
        system.thermals().len(),
        system.lines().len(),
        system.buses().len(),
        n_blks_stage0,
        has_inflow_penalty,
    );
    let initial_state = build_initial_state(&system, &indexer);

    let forward_passes = bcast_config.forward_passes;
    let stopping_rules = stopping_rules_from_broadcast(&bcast_config);
    let n_stages = stage_templates_ref.len();
    let max_iterations = max_iterations_from_rules(&stopping_rules);

    let fcf_capacity_iterations = max_iterations.saturating_add(1);
    let mut fcf = FutureCostFunction::new(
        n_stages,
        indexer.n_state,
        forward_passes,
        fcf_capacity_iterations,
        0,
    );

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures: Vec<RiskMeasure> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| RiskMeasure::from(s.risk_config))
        .collect();

    let mut solver = HighsSolver::new().map_err(|e| CliError::Solver {
        message: format!("HiGHS initialisation failed: {e}"),
    })?;

    let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();
    let training_config = TrainingConfig {
        forward_passes,
        max_iterations,
        checkpoint_interval: None,
        warm_start_cuts: 0,
        event_sender: Some(event_tx),
    };

    let quiet_rx: Option<mpsc::Receiver<TrainingEvent>>;
    let progress_handle = if quiet {
        quiet_rx = Some(event_rx);
        None
    } else {
        quiet_rx = None;
        Some(crate::progress::run_progress_thread(
            event_rx,
            max_iterations,
        ))
    };

    let n_load_buses = stage_templates.n_load_buses;
    let max_blocks = block_hours_per_stage
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(0);
    let block_counts_per_stage: Vec<usize> = block_hours_per_stage.iter().map(Vec::len).collect();
    let load_balance_row_starts = &stage_templates.load_balance_row_starts;
    let load_bus_indices = &stage_templates.load_bus_indices;

    // Reconstruct the cut selection strategy from the broadcast config so all
    // ranks pass the same strategy to train().
    let cut_selection_strategy = std::mem::replace(
        &mut bcast_config.cut_selection,
        BroadcastCutSelection::Disabled,
    )
    .into_strategy();

    let training_result = match train(
        &mut solver,
        training_config,
        &mut fcf,
        stage_templates_ref,
        base_rows,
        &indexer,
        &initial_state,
        stochastic.opening_tree(),
        &stochastic,
        &horizon,
        &risk_measures,
        stopping_rules,
        cut_selection_strategy.as_ref(),
        None,
        &comm,
        n_threads,
        HighsSolver::new,
        &bcast_config.inflow_method,
        noise_scale,
        n_hydros_lp,
        n_load_buses,
        max_blocks,
        load_balance_row_starts,
        load_bus_indices,
        &block_counts_per_stage,
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
    let training_output = build_training_output(&training_result, &events, &fcf);

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
        upper_bound_std: 0.0,
        gap_percent: training_result.final_gap * 100.0,
        total_cuts_active: training_output.cut_stats.total_active,
        total_cuts_generated: training_output.cut_stats.total_generated,
        total_lp_solves: global_lp_solves,
        total_time_ms: training_result.total_time_ms,
    };
    if !quiet && is_root {
        crate::summary::print_training_summary(&stderr, &training_summary);
    }

    let should_simulate = bcast_config.should_simulate;
    let output_dir: PathBuf = args.output.unwrap_or_else(|| args.case_dir.join("output"));

    if should_simulate {
        let n_scenarios = bcast_config.n_scenarios;
        let io_capacity = bcast_config.io_channel_capacity as usize;

        let sim_config = SimulationConfig {
            n_scenarios,
            io_channel_capacity: io_capacity,
        };

        let entity_counts = build_entity_counts(&system);

        let mut sim_pool = WorkspacePool::try_new(
            n_threads,
            indexer.hydro_count,
            indexer.max_par_order,
            indexer.n_state,
            n_load_buses,
            max_blocks,
            HighsSolver::new,
        )
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
            ))
        };

        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity.max(1));

        let drain_handle = std::thread::spawn(move || {
            result_rx
                .into_iter()
                .collect::<Vec<cobre_sddp::SimulationScenarioResult>>()
        });

        let sim_start = std::time::Instant::now();

        let sim_result = simulate(
            &mut sim_pool.workspaces,
            stage_templates_ref,
            base_rows,
            &fcf,
            &stochastic,
            &sim_config,
            &horizon,
            &initial_state,
            &indexer,
            &entity_counts,
            &comm,
            &result_tx,
            &bcast_config.inflow_method,
            noise_scale,
            n_hydros_lp,
            n_load_buses,
            load_balance_row_starts,
            load_bus_indices,
            &block_counts_per_stage,
            zeta_per_stage,
            block_hours_per_stage,
            Some(sim_event_tx),
        )
        .map_err(CliError::from);
        if let Some(handle) = sim_progress_handle {
            let _ = handle.join();
        }

        drop(result_tx);

        #[allow(clippy::expect_used)]
        let local_results = drain_handle.join().expect("drain thread panicked");

        sim_result?;

        #[allow(clippy::cast_possible_truncation)]
        let sim_time_ms = sim_start.elapsed().as_millis() as u64;

        let local_bytes =
            postcard::to_allocvec(&local_results).map_err(|e| CliError::Internal {
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

        comm.barrier().map_err(|e| CliError::Internal {
            message: format!("post-simulation barrier error: {e}"),
        })?;

        // Print the simulation summary now — before I/O starts.
        if !quiet && is_root {
            crate::summary::print_simulation_summary(
                &stderr,
                &SimulationSummary {
                    n_scenarios,
                    completed: n_scenarios,
                    failed: 0,
                    total_time_ms: sim_time_ms,
                },
            );
        }

        if is_root {
            let config = root_config.ok_or_else(|| CliError::Internal {
                message: "root_config was None on rank 0 — internal invariant violated".to_string(),
            })?;
            if !quiet {
                use std::io::Write;
                let _ = stderr.write_line("Writing outputs...");
                let _ = std::io::stderr().flush();
            }
            let write_start = std::time::Instant::now();

            let mut all_results: Vec<cobre_sddp::SimulationScenarioResult> =
                Vec::with_capacity(n_scenarios as usize);
            let mut offset = 0;
            for &count in &recv_counts {
                let partition: Vec<cobre_sddp::SimulationScenarioResult> =
                    postcard::from_bytes(&all_bytes[offset..offset + count]).map_err(|e| {
                        CliError::Internal {
                            message: format!("simulation result deserialization error: {e}"),
                        }
                    })?;
                all_results.extend(partition);
                offset += count;
            }

            let parquet_config = cobre_io::ParquetWriterConfig::default();
            let mut sim_writer = cobre_io::output::simulation_writer::SimulationParquetWriter::new(
                &output_dir,
                &system,
                &parquet_config,
            )
            .map_err(CliError::from)?;

            let mut failed: u32 = 0;
            for scenario_result in all_results {
                let payload = crate::simulation_io::convert_scenario(scenario_result);
                if let Err(e) = sim_writer.write_scenario(payload) {
                    tracing::error!("simulation write error: {e}");
                    failed += 1;
                }
            }
            let mut sim_output = sim_writer.finalize();
            sim_output.failed = failed;

            let policy_dir = output_dir.join(&bcast_config.policy_path);
            crate::policy_io::write_checkpoint(
                &policy_dir,
                &fcf,
                &training_result,
                &crate::policy_io::CheckpointParams {
                    max_iterations,
                    forward_passes,
                    seed,
                },
            )?;

            write_results(
                &output_dir,
                &training_output,
                Some(&sim_output),
                &system,
                &config,
            )
            .map_err(CliError::from)?;

            if !quiet {
                let write_secs = write_start.elapsed().as_secs_f64();
                crate::summary::print_output_path(&stderr, &output_dir, write_secs);
            }
        }
    } else if is_root {
        let config = root_config.ok_or_else(|| CliError::Internal {
            message: "root_config was None on rank 0 — internal invariant violated".to_string(),
        })?;
        if !quiet {
            use std::io::Write;
            let _ = stderr.write_line("Writing outputs...");
            let _ = std::io::stderr().flush();
        }
        let write_start = std::time::Instant::now();

        let policy_dir = output_dir.join(&bcast_config.policy_path);
        crate::policy_io::write_checkpoint(
            &policy_dir,
            &fcf,
            &training_result,
            &crate::policy_io::CheckpointParams {
                max_iterations,
                forward_passes,
                seed,
            },
        )?;

        write_results(&output_dir, &training_output, None, &system, &config)
            .map_err(CliError::from)?;

        if !quiet {
            let write_secs = write_start.elapsed().as_secs_f64();
            crate::summary::print_output_path(&stderr, &output_dir, write_secs);
        }
    }

    Ok(())
}

/// Return the maximum iteration budget from the stopping rule set.
///
/// Used for FCF pre-sizing. If no iteration limit is present, returns
/// [`DEFAULT_MAX_ITERATIONS`].
fn max_iterations_from_rules(rules: &StoppingRuleSet) -> u64 {
    rules
        .rules
        .iter()
        .filter_map(|r| {
            if let StoppingRule::IterationLimit { limit } = r {
                Some(*limit)
            } else {
                None
            }
        })
        .max()
        .unwrap_or(DEFAULT_MAX_ITERATIONS)
}

/// Build [`EntityCounts`] from the loaded system.
///
/// Entity IDs are extracted from [`cobre_core::EntityId`], which stores
/// an `i32` in its inner field.
fn build_entity_counts(system: &cobre_core::System) -> EntityCounts {
    use cobre_core::entities::hydro::HydroGenerationModel;

    EntityCounts {
        hydro_ids: system.hydros().iter().map(|h| h.id.0).collect(),
        hydro_productivities: system
            .hydros()
            .iter()
            .map(|h| match &h.generation_model {
                HydroGenerationModel::ConstantProductivity {
                    productivity_mw_per_m3s,
                }
                | HydroGenerationModel::LinearizedHead {
                    productivity_mw_per_m3s,
                } => *productivity_mw_per_m3s,
                HydroGenerationModel::Fpha => 0.0,
            })
            .collect(),
        thermal_ids: system.thermals().iter().map(|t| t.id.0).collect(),
        line_ids: system.lines().iter().map(|l| l.id.0).collect(),
        bus_ids: system.buses().iter().map(|b| b.id.0).collect(),
        pumping_station_ids: system.pumping_stations().iter().map(|p| p.id.0).collect(),
        contract_ids: system.contracts().iter().map(|c| c.id.0).collect(),
        non_controllable_ids: system
            .non_controllable_sources()
            .iter()
            .map(|n| n.id.0)
            .collect(),
    }
}

/// Build the initial state vector from the system's initial conditions.
///
/// The state vector layout is `[storage(0..N), lags(N..N*(1+L))]` where N is
/// the number of hydros and L is the maximum PAR order. Storage positions
/// correspond to hydros in canonical ID order. Lag variables are initialised
/// to zero (no historical inflow information at the start of the study).
///
/// Each `HydroStorage` entry in `initial_conditions.storage` is matched to
/// its positional index among the system's hydros (both sorted by `hydro_id`).
fn build_initial_state(system: &cobre_core::System, indexer: &StageIndexer) -> Vec<f64> {
    let mut state = vec![0.0_f64; indexer.n_state];
    let hydros = system.hydros();
    let ic = system.initial_conditions();

    for hs in &ic.storage {
        // Both hydros() and ic.storage are sorted by hydro_id.
        // Find the positional index for this hydro.
        if let Ok(idx) = hydros.binary_search_by_key(&hs.hydro_id.0, |h| h.id.0) {
            state[idx] = hs.value_hm3;
        }
    }

    state
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{broadcast_value, resolve_thread_count};

    /// A minimal serializable struct for testing the broadcast helper.
    #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Simple {
        x: f64,
        label: String,
    }

    /// `broadcast_value` with `LocalBackend` (single rank) round-trips a struct.
    ///
    /// With `LocalBackend`, broadcast is a no-op and the root-path code path is
    /// exercised: the function returns the original `Some(value)` unchanged after
    /// verifying that serialization succeeds (len > 0).
    #[test]
    fn broadcast_value_local_round_trips_simple() {
        let comm = cobre_comm::LocalBackend;
        let original = Simple {
            x: std::f64::consts::PI,
            label: "test".to_string(),
        };
        let result = broadcast_value(Some(original.clone()), &comm).unwrap();
        assert_eq!(result, original);
    }

    /// `broadcast_value` with `LocalBackend` round-trips a `Vec<f64>`.
    ///
    /// Verifies that the helper handles collection types that postcard can serialize.
    #[test]
    fn broadcast_value_local_round_trips_vec() {
        let comm = cobre_comm::LocalBackend;
        let original: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let result = broadcast_value(Some(original.clone()), &comm).unwrap();
        assert_eq!(result, original);
    }

    /// `broadcast_value` with `LocalBackend` round-trips a nested struct matching
    /// the shape of `cobre_io::config::TrainingConfig`.
    ///
    /// Uses a locally defined struct to avoid a test dependency on cobre-io internals.
    #[test]
    fn broadcast_value_local_round_trips_config_like() {
        #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
        struct ConfigLike {
            forward_passes: u32,
            seed: Option<i64>,
        }

        let comm = cobre_comm::LocalBackend;
        let original = ConfigLike {
            forward_passes: 4,
            seed: Some(42),
        };
        let result = broadcast_value(Some(original.clone()), &comm).unwrap();
        assert_eq!(result, original);
    }

    /// `broadcast_value` returns `CliError::Internal` when `None` is passed on root.
    ///
    /// Root rank must always supply `Some(value)`. Passing `None` on the only rank
    /// (`LocalBackend`, rank 0 == root) triggers the internal error path, returning
    /// [`crate::error::CliError::Internal`] rather than panicking.
    #[test]
    fn broadcast_value_returns_err_when_root_passes_none() {
        let comm = cobre_comm::LocalBackend;
        let result: Result<Simple, _> = broadcast_value(None, &comm);
        assert!(result.is_err(), "expected Err when root passes None");
        let err = result.unwrap_err();
        assert!(
            matches!(err, crate::error::CliError::Internal { .. }),
            "expected CliError::Internal, got: {err:?}"
        );
    }

    /// `broadcast_value` with `LocalBackend` round-trips a `u64` value.
    ///
    /// Verifies the broadcast helper serializes and deserializes primitive
    /// integer types correctly. Gated behind the `mpi` feature because this
    /// test exercises the same code path invoked by MPI-enabled runs (the
    /// `LocalBackend` substitutes for the real MPI communicator in tests).
    #[cfg(feature = "mpi")]
    #[test]
    fn broadcast_value_round_trips_u64() {
        let comm = cobre_comm::LocalBackend;
        let value: u64 = 42;
        let result = broadcast_value(Some(value), &comm).unwrap();
        assert_eq!(result, 42u64);
    }

    // ------------------------------------------------------------------
    // resolve_thread_count tests
    //
    // Note: env var mutation (`set_var`/`remove_var`) is unsafe in Rust 2024
    // and is forbidden by the workspace `unsafe_code = "forbid"` lint.
    // These tests therefore exercise only the paths that do not require env
    // var mutation: the CLI argument path and the fixed default value.
    // ------------------------------------------------------------------

    /// CLI `--threads` value is returned directly without consulting env vars.
    #[test]
    fn test_resolve_thread_count_cli_value() {
        assert_eq!(
            resolve_thread_count(Some(4)),
            4,
            "CLI value must be returned as-is"
        );
    }

    /// Single-thread default: passing Some(1) yields 1, matching the hardcoded
    /// fallback value and confirming single-threaded operation is always available.
    #[test]
    fn test_resolve_thread_count_default() {
        assert_eq!(
            resolve_thread_count(Some(1)),
            1,
            "single-thread CLI value must produce 1"
        );
    }
}
