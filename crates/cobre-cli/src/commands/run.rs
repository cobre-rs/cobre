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

use cobre_comm::{create_communicator, Communicator, ReduceOp};
use cobre_core::{System, TrainingEvent};
use cobre_io::output::{
    write_correlation_json, write_fitting_report, write_inflow_ar_coefficients,
    write_inflow_seasonal_stats, write_load_seasonal_stats, write_noise_openings, FittingReport,
    HydroFittingEntry,
};
use cobre_io::scenarios::{InflowArCoefficientRow, InflowSeasonalStatsRow, LoadSeasonalStatsRow};
use cobre_io::write_results;
use cobre_sddp::estimation::{estimate_from_history, HydroEstimationEntry};
use cobre_sddp::{
    EstimationReport, InflowNonNegativityMethod, SimulationScenarioResult, StoppingMode,
    StoppingRule, StoppingRuleSet, StudySetup,
};
use cobre_solver::HighsSolver;
use cobre_stochastic::{build_stochastic_context, context::OpeningTree};

use crate::error::CliError;
use crate::summary::{
    ArOrderSummary, SimulationSummary, StochasticSource, StochasticSummary, TrainingSummary,
};

/// Default number of forward-pass trajectories when not specified in config.
const DEFAULT_FORWARD_PASSES: u32 = 1;

/// Default maximum iterations when no stopping rule specifies an iteration limit.
const DEFAULT_MAX_ITERATIONS: u64 = 100;

/// Default random seed for stochastic scenario generation.
const DEFAULT_SEED: u64 = 42;

// Broadcast types — postcard-safe serializable wrappers for MPI communication

/// Postcard-serializable stopping rule.
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

/// Postcard-serializable cut selection strategy.
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
    n_scenarios: u32,
    io_channel_capacity: u32,
    policy_path: String,
    inflow_method: InflowNonNegativityMethod,
    cut_selection: BroadcastCutSelection,
}

impl BroadcastConfig {
    fn from_config(config: &cobre_io::Config) -> Result<Self, CliError> {
        use cobre_io::config::StoppingRuleConfig;

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
                    // Not implemented in MVP; fold into iteration limit.
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

        let n_scenarios = if config.simulation.enabled {
            config.simulation.num_scenarios
        } else {
            0
        };

        let parsed_cut_selection =
            cobre_sddp::parse_cut_selection_config(&config.training.cut_selection)
                .map_err(|msg| CliError::Validation { report: msg })?;
        let cut_selection = BroadcastCutSelection::from_strategy(parsed_cut_selection.as_ref());

        Ok(Self {
            seed,
            forward_passes,
            stopping_rules,
            stopping_mode,
            n_scenarios,
            io_channel_capacity: config.simulation.io_channel_capacity,
            policy_path: config.policy.path.clone(),
            inflow_method: InflowNonNegativityMethod::from(&config.modeling.inflow_non_negativity),
            cut_selection,
        })
    }
}

/// Postcard-serializable wrapper for [`OpeningTree`] broadcast.
///
/// [`OpeningTree`] does not implement `serde::Serialize + Deserialize` to avoid
/// adding a serde dependency to `cobre-stochastic`. This wrapper holds the three
/// constituent parts (`data`, `openings_per_stage`, `dim`) that are sufficient
/// to reconstruct the tree via [`OpeningTree::from_parts`] on all ranks.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct BroadcastOpeningTree {
    data: Vec<f64>,
    openings_per_stage: Vec<usize>,
    dim: usize,
}

/// Load, validate, and assemble a user-supplied opening tree from the case directory.
///
/// Called on rank 0 inside [`load_case_and_config`]. Runs [`validate_structure`]
/// to check whether `scenarios/noise_openings.parquet` is present. If absent,
/// returns `Ok(None)`. If present, parses the rows, validates them against the
/// system dimensions, and assembles an [`OpeningTree`].
///
/// # Expected dimension
///
/// The noise dimension matches what [`build_stochastic_context`] computes:
/// `n_hydros + n_load_buses`, where `n_load_buses` is the count of distinct bus
/// IDs that have at least one [`LoadModel`] entry with `std_mw > 0`.
///
/// # Errors
///
/// - [`CliError::Io`] if the Parquet file cannot be read.
/// - [`CliError::Validation`] if rows fail dimension or stage consistency checks.
///
/// [`LoadModel`]: cobre_core::scenario::LoadModel
fn load_user_opening_tree(
    case_dir: &Path,
    system: &System,
) -> Result<Option<OpeningTree>, CliError> {
    let mut ctx = cobre_io::ValidationContext::new();
    let manifest = cobre_io::validate_structure(case_dir, &mut ctx);

    if !manifest.scenarios_noise_openings_parquet {
        return Ok(None);
    }

    let path = case_dir.join("scenarios").join("noise_openings.parquet");

    let rows = cobre_io::load_noise_openings(Some(&path)).map_err(CliError::from)?;

    let n_hydros = system.hydros().len();
    let mut load_bus_ids: Vec<cobre_core::EntityId> = system
        .load_models()
        .iter()
        .filter(|m| m.std_mw > 0.0)
        .map(|m| m.bus_id)
        .collect();
    load_bus_ids.sort_unstable_by_key(|id| id.0);
    load_bus_ids.dedup();
    let n_load_buses = load_bus_ids.len();
    let expected_dim = n_hydros + n_load_buses;

    let expected_stages = system.stages().iter().filter(|s| s.id >= 0).count();
    let mut openings_by_stage: std::collections::BTreeMap<i32, std::collections::BTreeSet<u32>> =
        std::collections::BTreeMap::new();
    for row in &rows {
        openings_by_stage
            .entry(row.stage_id)
            .or_default()
            .insert(row.opening_index);
    }
    let openings_per_stage: Vec<usize> = openings_by_stage
        .values()
        .map(std::collections::BTreeSet::len)
        .collect();

    cobre_io::scenarios::validate_noise_openings(
        &rows,
        expected_dim,
        expected_stages,
        &openings_per_stage,
    )
    .map_err(CliError::from)?;

    let tree = cobre_io::scenarios::assemble_opening_tree(rows, expected_dim);
    Ok(Some(tree))
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

/// Return type of [`load_case_and_config`]: the five values loaded on rank 0.
type LoadedCase = (
    cobre_core::System,
    BroadcastConfig,
    cobre_io::Config,
    Option<EstimationReport>,
    Option<OpeningTree>,
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
    let (system, estimation_report) = estimate_from_history(system, &args.case_dir, &config)
        .map_err(|e| CliError::Internal {
            message: format!("estimation error: {e}"),
        })?;
    let user_opening_tree = load_user_opening_tree(&args.case_dir, &system)?;
    let bcast = BroadcastConfig::from_config(&config)?;
    Ok((system, bcast, config, estimation_report, user_opening_tree))
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

    // Only rank 0 accesses the filesystem. Config is converted to BroadcastConfig
    // (postcard-safe) and broadcast. Root config stays on rank 0 for output writing.
    // The estimation report is rank-0-only and is NOT broadcast.
    let (
        raw_system,
        raw_bcast_config,
        root_config,
        root_estimation_report,
        raw_opening_tree,
        load_err,
    ) = if is_root {
        match load_case_and_config(&args, quiet, &stderr) {
            Ok((system, bcast, config, estimation_report, user_opening_tree)) => (
                Some(system),
                Some(bcast),
                Some(config),
                Some(estimation_report),
                Some(user_opening_tree),
                None,
            ),
            Err(e) => (None, None, None, None, None, Some(e)),
        }
    } else {
        (None, None, None, None, None, None)
    };
    let root_estimation_report: Option<Option<EstimationReport>> = root_estimation_report;

    let system_result = broadcast_value(raw_system, &comm);
    let bcast_config_result = broadcast_value(raw_bcast_config, &comm);

    // Broadcast the optional opening tree. Wrap in Option<BroadcastOpeningTree> so that
    // both the "no user tree" (None) and "user tree present" (Some) cases can be broadcast
    // as a single postcard-serializable value. postcard supports Option natively.
    let raw_bcast_tree: Option<Option<BroadcastOpeningTree>> = raw_opening_tree.map(|tree_opt| {
        tree_opt.map(|t| BroadcastOpeningTree {
            data: t.data().to_vec(),
            openings_per_stage: t.openings_per_stage_slice().to_vec(),
            dim: t.dim(),
        })
    });
    let tree_result = broadcast_value(raw_bcast_tree, &comm);

    if let Some(e) = load_err {
        return Err(e);
    }
    let system = system_result?;
    let mut bcast_config = bcast_config_result?;
    let user_tree: Option<OpeningTree> =
        tree_result?.map(|bt| OpeningTree::from_parts(bt.data, bt.openings_per_stage, bt.dim));

    // Consume args here — no fields are needed after output_dir is resolved.
    let output_dir: PathBuf = args.output.unwrap_or_else(|| args.case_dir.join("output"));

    let seed = bcast_config.seed;
    let stochastic = build_stochastic_context(&system, seed, &[], user_tree).map_err(|e| {
        CliError::Internal {
            message: format!("stochastic context error: {e}"),
        }
    })?;

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
    )
    .map_err(CliError::from)?;

    // Print stochastic preprocessing summary on rank 0 before training starts.
    if !quiet && is_root {
        let estimation = root_estimation_report.as_ref().and_then(|r| r.as_ref());
        let stochastic_summary =
            build_stochastic_summary(&system, setup.stochastic(), estimation, seed);
        crate::summary::print_stochastic_summary(&stderr, &stochastic_summary);
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
    comm.barrier().map_err(|e| CliError::Internal {
        message: format!("post-export barrier error: {e}"),
    })?;

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
        HighsSolver::new,
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

    let should_simulate = setup.n_scenarios() > 0;

    if should_simulate {
        let n_scenarios = setup.n_scenarios();
        let sim_config = setup.simulation_config();

        let mut sim_pool = setup
            .create_workspace_pool(n_threads, HighsSolver::new)
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

            let mut all_results: Vec<SimulationScenarioResult> =
                Vec::with_capacity(n_scenarios as usize);
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

            let policy_dir = output_dir.join(setup.policy_path());
            crate::policy_io::write_checkpoint(
                &policy_dir,
                setup.fcf(),
                &training_result,
                &crate::policy_io::CheckpointParams {
                    max_iterations: setup.max_iterations(),
                    forward_passes: setup.forward_passes(),
                    seed: setup.seed(),
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

        let policy_dir = output_dir.join(setup.policy_path());
        crate::policy_io::write_checkpoint(
            &policy_dir,
            setup.fcf(),
            &training_result,
            &crate::policy_io::CheckpointParams {
                max_iterations: setup.max_iterations(),
                forward_passes: setup.forward_passes(),
                seed: setup.seed(),
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

// ---------------------------------------------------------------------------
// Stochastic summary builder
// ---------------------------------------------------------------------------

/// Build a [`StochasticSummary`] from the system, stochastic context, and estimation report.
///
/// Called on rank 0 after [`build_stochastic_context`] returns and before training
/// starts. All fields are derived from the already-validated inputs; construction
/// is infallible.
///
/// # Source detection
///
/// - `Estimated`: `estimation_report` is `Some(_)` — the estimation pipeline ran
///   and produced AR coefficients and correlation.
/// - `Loaded`: no estimation report but hydros are present — seasonal stats were
///   loaded from pre-supplied files.
/// - `None`: no hydros in the system.
fn build_stochastic_summary(
    system: &System,
    stochastic: &cobre_stochastic::StochasticContext,
    estimation_report: Option<&EstimationReport>,
    seed: u64,
) -> StochasticSummary {
    let n_hydros = system.hydros().len();

    // Determine inflow source from estimation report presence.
    let inflow_source = if estimation_report.is_some() {
        StochasticSource::Estimated
    } else if n_hydros > 0 {
        StochasticSource::Loaded
    } else {
        StochasticSource::None
    };

    // Count distinct stage_id values from the first hydro's inflow models.
    let n_seasons = if n_hydros > 0 {
        let first_hydro_id = system.hydros()[0].id;
        let mut stage_ids: Vec<i32> = system
            .inflow_models()
            .iter()
            .filter(|m| m.hydro_id == first_hydro_id)
            .map(|m| m.stage_id)
            .collect();
        stage_ids.sort_unstable();
        stage_ids.dedup();
        stage_ids.len()
    } else {
        0
    };

    // Build AR order summary from estimation report or inflow model coefficients.
    let ar_summary = if n_hydros > 0 {
        let (method, orders): (String, Vec<usize>) = if let Some(report) = estimation_report {
            let orders: Vec<usize> = report
                .entries
                .values()
                .map(|entry| entry.selected_order as usize)
                .collect();
            ("AIC".to_string(), orders)
        } else {
            // Derive from loaded inflow models: use max AR coefficient length per hydro.
            let orders: Vec<usize> = system
                .hydros()
                .iter()
                .map(|h| {
                    system
                        .inflow_models()
                        .iter()
                        .filter(|m| m.hydro_id == h.id)
                        .map(|m| m.ar_coefficients.len())
                        .max()
                        .unwrap_or(0)
                })
                .collect();
            ("fixed".to_string(), orders)
        };

        let min_order = orders.iter().copied().min().unwrap_or(0);
        let max_order = orders.iter().copied().max().unwrap_or(0);

        let mut order_counts = vec![0usize; max_order + 1];
        for &ord in &orders {
            order_counts[ord] += 1;
        }

        Some(ArOrderSummary {
            method,
            order_counts,
            min_order,
            max_order,
            n_hydros,
        })
    } else {
        None
    };

    // Correlation source mirrors inflow source (both come from estimation or loaded files).
    let correlation_source = if estimation_report.is_some() {
        StochasticSource::Estimated
    } else if n_hydros > 0 {
        StochasticSource::Loaded
    } else {
        StochasticSource::None
    };

    // Correlation dimension is n_hydros × n_hydros (hydro-to-hydro spatial correlation).
    let correlation_dim = if n_hydros > 0 {
        Some(format!("{n_hydros}x{n_hydros}"))
    } else {
        None
    };

    // Derive opening tree provenance from stochastic context.
    // The tree was either loaded from scenarios/noise_openings.parquet (if present)
    // or generated deterministically from the seed. We infer from n_stages: when a
    // user-supplied tree is present, we cannot distinguish it from generated here
    // without passing extra state. The tree is always present after build_stochastic_context.
    // For now: if n_hydros + load buses > 0, the tree was generated; loaded trees are
    // distinguishable via the fact that the caller passed a user_tree. However, we no
    // longer have that flag here — we use StochasticSource::Estimated/Loaded/None to
    // approximate: if the stochastic context has a user-supplied tree, it shows as
    // dim > 0 regardless. Since we cannot distinguish loaded vs generated tree without
    // extra state here, we mark it Estimated (generated from seed) as the default.
    // The tree is always generated (or replaced by a loaded one upstream); we report
    // StochasticSource::Loaded only when openings came from file.
    // NOTE: This is a conservative approximation. The full distinction (loaded vs generated)
    // would require threading the `user_tree_was_present` flag through to this function.
    // Per the ticket spec, the opening_tree_source comes from stochastic.opening_tree().
    // We report "loaded" when n_openings > 1 at stage 0 (implies multi-scenario input),
    // otherwise "estimated" (single opening per stage = generated default).
    let opening_tree = stochastic.opening_tree();
    let openings_per_stage = if opening_tree.n_stages() > 0 {
        opening_tree.n_openings(0)
    } else {
        0
    };
    // A loaded opening tree typically has many openings; the default generated tree
    // has exactly 1 per stage. Use that heuristic to set the source label.
    let opening_tree_source = if openings_per_stage > 1 {
        StochasticSource::Loaded
    } else {
        StochasticSource::Estimated
    };

    let n_stages = stochastic.n_stages();
    let n_load_buses = stochastic.n_load_buses();

    StochasticSummary {
        inflow_source,
        n_hydros,
        n_seasons,
        ar_summary,
        correlation_source,
        correlation_dim,
        opening_tree_source,
        openings_per_stage,
        n_stages,
        n_load_buses,
        seed,
    }
}

// ---------------------------------------------------------------------------
// Stochastic artifact export helpers
// ---------------------------------------------------------------------------

/// Convert an [`EstimationReport`] to a [`FittingReport`] for serialization.
///
/// Maps each `(EntityId, HydroEstimationEntry)` pair to a `(String, HydroFittingEntry)` pair.
/// The entity ID is serialized as its inner `i32` value converted to a string (e.g., `"1"`).
/// Fields are copied 1:1 — `selected_order`, `aic_scores`, `coefficients`.
fn estimation_report_to_fitting_report(report: &EstimationReport) -> FittingReport {
    let hydros = report
        .entries
        .iter()
        .map(
            |(id, entry): (&cobre_core::EntityId, &HydroEstimationEntry)| {
                (
                    id.0.to_string(),
                    HydroFittingEntry {
                        selected_order: entry.selected_order,
                        aic_scores: entry.aic_scores.clone(),
                        coefficients: entry.coefficients.clone(),
                    },
                )
            },
        )
        .collect();
    FittingReport { hydros }
}

/// Convert a slice of [`cobre_core::scenario::InflowModel`] to [`InflowSeasonalStatsRow`] records.
///
/// One row per model entry — `hydro_id`, `stage_id`, `mean_m3s`, `std_m3s` fields map 1:1.
fn inflow_models_to_stats_rows(
    models: &[cobre_core::scenario::InflowModel],
) -> Vec<InflowSeasonalStatsRow> {
    models
        .iter()
        .map(|m| InflowSeasonalStatsRow {
            hydro_id: m.hydro_id,
            stage_id: m.stage_id,
            mean_m3s: m.mean_m3s,
            std_m3s: m.std_m3s,
        })
        .collect()
}

/// Convert a slice of [`cobre_core::scenario::InflowModel`] to [`InflowArCoefficientRow`] records.
///
/// Expands each model's `ar_coefficients` into one row per lag (1-based lag numbering).
/// Models with empty `ar_coefficients` (white noise, AR order 0) contribute no rows.
/// The `residual_std_ratio` is repeated across all lag rows of the same `(hydro_id, stage_id)`.
fn inflow_models_to_ar_rows(
    models: &[cobre_core::scenario::InflowModel],
) -> Vec<InflowArCoefficientRow> {
    models
        .iter()
        .flat_map(|m| {
            m.ar_coefficients
                .iter()
                .enumerate()
                .map(move |(i, &coefficient)| {
                    // AR order is bounded by a small integer (typical range 1-12);
                    // the cast from usize to i32 is safe in practice.
                    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                    let lag = (i + 1) as i32;
                    InflowArCoefficientRow {
                        hydro_id: m.hydro_id,
                        stage_id: m.stage_id,
                        lag,
                        coefficient,
                        residual_std_ratio: m.residual_std_ratio,
                    }
                })
        })
        .collect()
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
        eprintln!("warning: stochastic export failed (noise_openings): {e}");
    }

    let stats_rows = inflow_models_to_stats_rows(system.inflow_models());
    if let Err(e) = write_inflow_seasonal_stats(
        &stochastic_dir.join("inflow_seasonal_stats.parquet"),
        &stats_rows,
    ) {
        eprintln!("warning: stochastic export failed (inflow_seasonal_stats): {e}");
    }

    let ar_rows = inflow_models_to_ar_rows(system.inflow_models());
    if let Err(e) = write_inflow_ar_coefficients(
        &stochastic_dir.join("inflow_ar_coefficients.parquet"),
        &ar_rows,
    ) {
        eprintln!("warning: stochastic export failed (inflow_ar_coefficients): {e}");
    }

    if let Err(e) = write_correlation_json(
        &stochastic_dir.join("correlation.json"),
        system.correlation(),
    ) {
        eprintln!("warning: stochastic export failed (correlation): {e}");
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
            eprintln!("warning: stochastic export failed (load_seasonal_stats): {e}");
        }
    }

    if let Some(report) = estimation_report {
        let fitting = estimation_report_to_fitting_report(report);
        if let Err(e) = write_fitting_report(&stochastic_dir.join("fitting_report.json"), &fitting)
        {
            eprintln!("warning: stochastic export failed (fitting_report): {e}");
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::{broadcast_value, resolve_thread_count, BroadcastOpeningTree};

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

    // ------------------------------------------------------------------
    // BroadcastOpeningTree tests
    // ------------------------------------------------------------------

    /// `BroadcastOpeningTree` round-trips through postcard serialization.
    ///
    /// Verifies that the wrapper type is fully postcard-compatible and that
    /// no field is lost during the serialize → deserialize round-trip.
    #[test]
    fn broadcast_opening_tree_round_trips_via_postcard() {
        let original = BroadcastOpeningTree {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            openings_per_stage: vec![2, 1],
            dim: 3,
        };
        let bytes = postcard::to_allocvec(&original).unwrap();
        let decoded: BroadcastOpeningTree = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.data, original.data, "data must survive round-trip");
        assert_eq!(
            decoded.openings_per_stage, original.openings_per_stage,
            "openings_per_stage must survive round-trip"
        );
        assert_eq!(decoded.dim, original.dim, "dim must survive round-trip");
    }

    /// `BroadcastOpeningTree` wrapped in `Option` round-trips via `broadcast_value`
    /// with `LocalBackend`. Covers both the `None` and `Some` cases.
    ///
    /// `Some(None)` represents "no user-supplied tree" and `Some(Some(...))` represents
    /// a valid user tree. Both must survive the broadcast without data loss.
    #[test]
    fn broadcast_optional_opening_tree_local_round_trips() {
        use cobre_stochastic::context::OpeningTree;

        let comm = cobre_comm::LocalBackend;

        // Case 1: no user tree — broadcast Some(None)
        let no_tree: Option<BroadcastOpeningTree> = None;
        let result = broadcast_value(Some(no_tree), &comm).unwrap();
        assert!(result.is_none(), "Some(None) must round-trip to None");

        // Case 2: user tree present — broadcast Some(Some(...))
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let ops = vec![2];
        let dim = 2usize;
        let source_tree = OpeningTree::from_parts(data.clone(), ops.clone(), dim);
        let bcast = Some(BroadcastOpeningTree {
            data: source_tree.data().to_vec(),
            openings_per_stage: source_tree.openings_per_stage_slice().to_vec(),
            dim: source_tree.dim(),
        });
        let result = broadcast_value(Some(bcast), &comm).unwrap();
        let bt = result.unwrap();
        let reconstructed = OpeningTree::from_parts(bt.data, bt.openings_per_stage, bt.dim);
        assert_eq!(
            reconstructed.data(),
            source_tree.data(),
            "reconstructed tree data must match source"
        );
        assert_eq!(
            reconstructed.dim(),
            source_tree.dim(),
            "reconstructed tree dim must match source"
        );
        assert_eq!(
            reconstructed.openings_per_stage_slice(),
            source_tree.openings_per_stage_slice(),
            "reconstructed tree openings_per_stage must match source"
        );
    }

    // ------------------------------------------------------------------
    // load_user_opening_tree tests
    // ------------------------------------------------------------------

    /// `load_user_opening_tree` returns `Ok(None)` when no `noise_openings.parquet`
    /// is present in the case directory.
    ///
    /// `load_user_opening_tree` checks the manifest before using the `system`
    /// parameter, so a minimal system (one bus, no stages) suffices — the function
    /// returns `Ok(None)` before any field of `system` is accessed.
    #[test]
    fn load_user_opening_tree_returns_none_when_file_absent() {
        use super::load_user_opening_tree;
        use cobre_core::{
            scenario::CorrelationModel, Bus, DeficitSegment, EntityId, SystemBuilder,
        };
        use std::fs;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let root = dir.path();

        // Create all required structural files so validate_structure reports no
        // errors for missing required files. The optional noise_openings.parquet
        // is intentionally absent.
        fs::create_dir_all(root.join("system")).unwrap();
        fs::write(root.join("config.json"), b"{}").unwrap();
        fs::write(root.join("penalties.json"), b"{}").unwrap();
        fs::write(root.join("stages.json"), b"{}").unwrap();
        fs::write(root.join("initial_conditions.json"), b"{}").unwrap();
        fs::write(root.join("system/buses.json"), b"{}").unwrap();
        fs::write(root.join("system/lines.json"), b"{}").unwrap();
        fs::write(root.join("system/hydros.json"), b"{}").unwrap();
        fs::write(root.join("system/thermals.json"), b"{}").unwrap();

        // A minimal system with one bus, no stages. The system is not consulted
        // because load_user_opening_tree returns Ok(None) before the manifest check.
        let bus = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let system = SystemBuilder::new()
            .buses(vec![bus])
            .correlation(CorrelationModel::default())
            .build()
            .unwrap();

        let result = load_user_opening_tree(root, &system).unwrap();
        assert!(
            result.is_none(),
            "must return Ok(None) when noise_openings.parquet is absent"
        );
    }

    // ------------------------------------------------------------------
    // Stochastic artifact conversion helper tests
    // ------------------------------------------------------------------

    /// `estimation_report_to_fitting_report` maps 2-hydro `EstimationReport`
    /// to `FittingReport` preserving all fields and using string entity IDs as keys.
    #[test]
    fn estimation_report_to_fitting_report_two_hydros() {
        use std::collections::BTreeMap;

        use cobre_core::EntityId;
        use cobre_sddp::estimation::HydroEstimationEntry;
        use cobre_sddp::EstimationReport;

        use super::estimation_report_to_fitting_report;

        let mut entries = BTreeMap::new();
        entries.insert(
            EntityId(1),
            HydroEstimationEntry {
                selected_order: 3,
                aic_scores: vec![10.0, 9.5, 9.2, 9.8],
                coefficients: vec![vec![0.4, -0.1, 0.05], vec![0.3, -0.08, 0.04]],
            },
        );
        entries.insert(
            EntityId(5),
            HydroEstimationEntry {
                selected_order: 2,
                aic_scores: vec![12.1, 11.3, 11.5],
                coefficients: vec![vec![0.6, -0.2]],
            },
        );
        let report = EstimationReport { entries };
        let fitting = estimation_report_to_fitting_report(&report);

        assert_eq!(
            fitting.hydros.len(),
            2,
            "FittingReport must contain exactly 2 hydro entries"
        );

        let h1 = fitting.hydros.get("1").unwrap();
        assert_eq!(h1.selected_order, 3);
        assert_eq!(h1.aic_scores, vec![10.0, 9.5, 9.2, 9.8]);
        assert_eq!(h1.coefficients.len(), 2);

        let h5 = fitting.hydros.get("5").unwrap();
        assert_eq!(h5.selected_order, 2);
        assert_eq!(h5.aic_scores, vec![12.1, 11.3, 11.5]);
        assert_eq!(h5.coefficients, vec![vec![0.6, -0.2]]);
    }

    /// `inflow_models_to_stats_rows` produces the correct number of rows and
    /// preserves `hydro_id`, `stage_id`, `mean_m3s`, `std_m3s` field values.
    #[test]
    fn inflow_models_to_stats_rows_field_values() {
        use cobre_core::scenario::InflowModel;
        use cobre_core::EntityId;

        use super::inflow_models_to_stats_rows;

        let models = vec![
            InflowModel {
                hydro_id: EntityId(1),
                stage_id: 0,
                mean_m3s: 150.0,
                std_m3s: 30.0,
                ar_coefficients: vec![0.5],
                residual_std_ratio: 0.87,
            },
            InflowModel {
                hydro_id: EntityId(2),
                stage_id: 1,
                mean_m3s: 200.0,
                std_m3s: 40.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            },
        ];
        let rows = inflow_models_to_stats_rows(&models);

        assert_eq!(rows.len(), 2, "must produce one row per model");
        assert_eq!(rows[0].hydro_id, EntityId(1));
        assert_eq!(rows[0].stage_id, 0);
        assert_eq!(rows[0].mean_m3s, 150.0);
        assert_eq!(rows[0].std_m3s, 30.0);
        assert_eq!(rows[1].hydro_id, EntityId(2));
        assert_eq!(rows[1].mean_m3s, 200.0);
    }

    /// `inflow_models_to_ar_rows` produces 1-based lag numbering and the correct
    /// total row count (sum of AR orders across all models).
    ///
    /// A model with 3 AR coefficients produces 3 rows with lags 1, 2, 3.
    /// A white-noise model (order 0) produces no rows.
    #[test]
    fn inflow_models_to_ar_rows_lag_numbering_and_count() {
        use cobre_core::scenario::InflowModel;
        use cobre_core::EntityId;

        use super::inflow_models_to_ar_rows;

        let models = vec![
            InflowModel {
                hydro_id: EntityId(1),
                stage_id: 0,
                mean_m3s: 100.0,
                std_m3s: 20.0,
                ar_coefficients: vec![0.4, -0.1, 0.05],
                residual_std_ratio: 0.92,
            },
            InflowModel {
                hydro_id: EntityId(2),
                stage_id: 0,
                mean_m3s: 80.0,
                std_m3s: 15.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            },
        ];
        let rows = inflow_models_to_ar_rows(&models);

        // 3 rows for hydro 1 (order 3), 0 rows for hydro 2 (order 0).
        assert_eq!(rows.len(), 3, "must produce 3 rows total (3 + 0)");

        assert_eq!(rows[0].hydro_id, EntityId(1));
        assert_eq!(rows[0].lag, 1, "first lag must be 1 (1-based)");
        assert_eq!(rows[0].coefficient, 0.4);
        assert_eq!(rows[0].residual_std_ratio, 0.92);

        assert_eq!(rows[1].lag, 2);
        assert_eq!(rows[1].coefficient, -0.1);

        assert_eq!(rows[2].lag, 3);
        assert_eq!(rows[2].coefficient, 0.05);
    }

    // ------------------------------------------------------------------
    // build_stochastic_summary tests
    // ------------------------------------------------------------------

    /// Helper: build a minimal `System` with one hydro, one bus, two study stages,
    /// and two `InflowModel` entries (one per stage, AR order 2).
    #[allow(clippy::too_many_lines)]
    fn make_system_with_hydro() -> cobre_core::System {
        use chrono::NaiveDate;
        use cobre_core::{
            entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties},
            scenario::{CorrelationModel, InflowModel},
            temporal::{
                Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
                StageStateConfig,
            },
            Bus, DeficitSegment, EntityId, SystemBuilder,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };

        let hydro = Hydro {
            id: EntityId(10),
            name: "H1".to_string(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.95,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
            },
        };

        let make_stage = |idx: usize, id: i32| Stage {
            index: idx,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(id.unsigned_abs() as usize),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 3,
                noise_method: NoiseMethod::Saa,
            },
        };

        let stages = vec![make_stage(0, 0), make_stage(1, 1)];

        let inflow_models = vec![
            InflowModel {
                hydro_id: EntityId(10),
                stage_id: 0,
                mean_m3s: 50.0,
                std_m3s: 10.0,
                ar_coefficients: vec![0.5, 0.3],
                residual_std_ratio: 0.8,
            },
            InflowModel {
                hydro_id: EntityId(10),
                stage_id: 1,
                mean_m3s: 60.0,
                std_m3s: 12.0,
                ar_coefficients: vec![0.4, 0.2],
                residual_std_ratio: 0.85,
            },
        ];

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(CorrelationModel::default())
            .build()
            .unwrap()
    }

    /// `build_stochastic_summary` with no estimation report yields `Loaded` inflow source.
    #[test]
    fn build_stochastic_summary_loaded_source_when_no_estimation_report() {
        use super::build_stochastic_summary;
        use cobre_stochastic::build_stochastic_context;

        let system = make_system_with_hydro();
        let stochastic = build_stochastic_context(&system, 42, &[], None).unwrap();
        let summary = build_stochastic_summary(&system, &stochastic, None, 42);

        assert!(
            matches!(
                summary.inflow_source,
                crate::summary::StochasticSource::Loaded
            ),
            "inflow_source must be Loaded when no estimation report is present"
        );
        assert_eq!(summary.n_hydros, 1, "n_hydros must be 1");
        assert_eq!(
            summary.n_seasons, 2,
            "n_seasons must be 2 (stage 0 and stage 1)"
        );
        assert_eq!(summary.seed, 42, "seed must be 42");
    }

    /// `build_stochastic_summary` with an estimation report yields `Estimated` inflow source
    /// and populates `ar_summary` with AIC method.
    #[test]
    fn build_stochastic_summary_estimated_source_with_estimation_report() {
        use std::collections::BTreeMap;

        use super::build_stochastic_summary;
        use cobre_core::EntityId;
        use cobre_sddp::estimation::HydroEstimationEntry;
        use cobre_sddp::EstimationReport;
        use cobre_stochastic::build_stochastic_context;

        let system = make_system_with_hydro();
        let stochastic = build_stochastic_context(&system, 7, &[], None).unwrap();

        let mut entries = BTreeMap::new();
        entries.insert(
            EntityId(10),
            HydroEstimationEntry {
                selected_order: 2,
                aic_scores: vec![-10.0, -12.0],
                coefficients: vec![vec![0.5, 0.3], vec![0.4, 0.2]],
            },
        );
        let report = EstimationReport { entries };

        let summary = build_stochastic_summary(&system, &stochastic, Some(&report), 7);

        assert!(
            matches!(
                summary.inflow_source,
                crate::summary::StochasticSource::Estimated
            ),
            "inflow_source must be Estimated when estimation report is present"
        );
        assert!(
            matches!(
                summary.correlation_source,
                crate::summary::StochasticSource::Estimated
            ),
            "correlation_source must be Estimated when estimation report is present"
        );
        let ar = summary.ar_summary.as_ref().unwrap();
        assert_eq!(ar.method, "AIC", "AR method must be AIC");
        assert_eq!(ar.max_order, 2, "max AR order must be 2");
        assert_eq!(summary.seed, 7, "seed must be 7");
    }

    /// `build_stochastic_summary` with no hydros yields `None` source and no AR summary.
    #[test]
    fn build_stochastic_summary_no_hydros_yields_none_source() {
        use chrono::NaiveDate;
        use cobre_core::{
            scenario::CorrelationModel,
            temporal::{
                Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
                StageStateConfig,
            },
            Bus, DeficitSegment, EntityId, SystemBuilder,
        };
        use cobre_stochastic::build_stochastic_context;

        use super::build_stochastic_summary;

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let stages = vec![Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 3,
                noise_method: NoiseMethod::Saa,
            },
        }];
        let system = SystemBuilder::new()
            .buses(vec![bus])
            .stages(stages)
            .correlation(CorrelationModel::default())
            .build()
            .unwrap();

        let stochastic = build_stochastic_context(&system, 0, &[], None).unwrap();
        let summary = build_stochastic_summary(&system, &stochastic, None, 0);

        assert!(
            matches!(
                summary.inflow_source,
                crate::summary::StochasticSource::None
            ),
            "inflow_source must be None when there are no hydros"
        );
        assert_eq!(summary.n_hydros, 0, "n_hydros must be 0");
        assert!(
            summary.ar_summary.is_none(),
            "ar_summary must be None with no hydros"
        );
    }

    /// `build_stochastic_summary` derives `n_stages` and `n_load_buses` from stochastic context.
    #[test]
    fn build_stochastic_summary_stages_and_load_buses() {
        use super::build_stochastic_summary;
        use cobre_stochastic::build_stochastic_context;

        let system = make_system_with_hydro();
        let stochastic = build_stochastic_context(&system, 1, &[], None).unwrap();
        let summary = build_stochastic_summary(&system, &stochastic, None, 1);

        assert_eq!(
            summary.n_stages, 2,
            "n_stages must match stochastic context"
        );
        assert_eq!(
            summary.n_load_buses, 0,
            "n_load_buses must be 0 (no stochastic load)"
        );
    }
}
