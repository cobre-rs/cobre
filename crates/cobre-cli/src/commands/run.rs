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

use cobre_comm::{create_communicator, Communicator};
use cobre_core::TrainingEvent;
use cobre_io::write_results;
use cobre_sddp::{
    build_stage_templates, build_training_output, simulate, train, EntityCounts,
    FutureCostFunction, HorizonMode, RiskMeasure, SimulationConfig, StageIndexer, StoppingMode,
    StoppingRule, StoppingRuleSet, TrainingConfig,
};
use cobre_solver::HighsSolver;
use cobre_stochastic::build_stochastic_context;

use crate::error::CliError;
use crate::summary::{RunSummary, SimulationSummary, TrainingSummary};

/// Default number of forward-pass trajectories when not specified in config.
const DEFAULT_FORWARD_PASSES: u32 = 1;

/// Default maximum iterations when no stopping rule specifies an iteration limit.
const DEFAULT_MAX_ITERATIONS: u64 = 100;

/// Default random seed for stochastic scenario generation.
const DEFAULT_SEED: u64 = 42;

// ---------------------------------------------------------------------------
// BroadcastConfig — postcard-safe configuration snapshot for MPI broadcast
// ---------------------------------------------------------------------------

/// A postcard-serializable stopping rule, mirroring [`StoppingRule`].
///
/// [`StoppingRule`] does not implement `serde::Serialize`/`Deserialize` and
/// uses no `#[serde(tag)]` attribute, so it cannot be broadcast directly.
/// [`StoppingRuleConfig`](cobre_io::config::StoppingRuleConfig) uses an
/// internally-tagged enum (`#[serde(tag = "type")]`) that postcard cannot
/// deserialize (postcard refuses `deserialize_any`). This type uses the
/// default external-tag representation which postcard supports fully.
///
/// Only the variants reachable via [`resolve_stopping_rules`] are included.
/// The `Simulation` variant from [`StoppingRuleConfig`] is folded into
/// `IterationLimit` by [`resolve_stopping_rules`], so it never appears here.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum BroadcastStoppingRule {
    IterationLimit { limit: u64 },
    TimeLimit { seconds: f64 },
    BoundStalling { iterations: u64, tolerance: f64 },
}

/// A postcard-serializable stopping mode, mirroring [`StoppingMode`].
///
/// [`StoppingMode`] does not implement serde traits.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum BroadcastStoppingMode {
    Any,
    All,
}

/// Configuration snapshot broadcast from rank 0 to all ranks before training.
///
/// [`cobre_io::Config`] cannot be broadcast via postcard because it contains
/// [`cobre_io::config::StoppingRuleConfig`], an internally-tagged enum
/// (`#[serde(tag = "type")]`) that requires `deserialize_any` — a feature
/// postcard explicitly refuses to implement. This struct holds only the fields
/// that all ranks need for training and simulation, using types that are
/// postcard-serializable.
///
/// Rank 0 resolves the raw [`cobre_io::Config`] into a `BroadcastConfig`
/// before broadcasting. Rank 0 retains the raw `Config` for output-writing
/// paths that require it (e.g., `write_results`).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct BroadcastConfig {
    /// Random seed for stochastic scenario generation.
    seed: u64,
    /// Number of forward-pass trajectories per iteration.
    forward_passes: u32,
    /// Stopping rules in postcard-safe form.
    stopping_rules: Vec<BroadcastStoppingRule>,
    /// Combination mode for stopping rules.
    stopping_mode: BroadcastStoppingMode,
    /// Whether to run post-training simulation on all ranks.
    should_simulate: bool,
    /// Number of simulation scenarios (0 when simulation is disabled).
    n_scenarios: u32,
    /// Bounded channel capacity for simulation result collection.
    io_channel_capacity: u32,
    /// Relative path for the policy checkpoint directory (within `output_dir`).
    /// Used by rank 0 for writing the FCF checkpoint.
    policy_path: String,
}

impl BroadcastConfig {
    /// Build a `BroadcastConfig` from a loaded [`cobre_io::Config`] and the
    /// `--skip-simulation` flag.
    ///
    /// Resolves all values that are needed by every rank during training and
    /// simulation. The raw `Config` is not consumed and remains available on
    /// rank 0 for output-writing paths.
    fn from_config(config: &cobre_io::Config, skip_simulation: bool) -> Self {
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

        Self {
            seed,
            forward_passes,
            stopping_rules,
            stopping_mode,
            should_simulate,
            n_scenarios: config.simulation.num_scenarios,
            io_channel_capacity: config.simulation.io_channel_capacity,
            policy_path: config.policy.path.clone(),
        }
    }
}

/// Convert a [`BroadcastConfig`] back into a [`StoppingRuleSet`] for the
/// training loop.
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
}

/// Broadcast a serializable value from rank 0 to all ranks.
///
/// Rank 0 serializes the value with postcard, broadcasts the byte length (as
/// `[u64; 1]`), then broadcasts the bytes. Other ranks receive the length,
/// allocate a buffer, receive the bytes, and deserialize. A length of 0 is
/// used to signal a failure on rank 0, causing all ranks to return an error.
///
/// Root rank keeps its original value (no round-trip deserialization on root)
/// for efficiency; postcard round-trips are exact so both paths produce
/// identical values.
///
/// # Errors
///
/// - [`CliError::Internal`] with "serialization error" if postcard serialization fails.
/// - [`CliError::Internal`] with "broadcast error" if the communicator broadcast fails.
/// - [`CliError::Internal`] with "deserialization error" if postcard deserialization fails.
/// - [`CliError::Internal`] with "rank 0 signaled broadcast failure" if length 0 is received.
fn broadcast_value<T, C>(value: Option<T>, comm: &C) -> Result<T, CliError>
where
    T: serde::Serialize + serde::de::DeserializeOwned,
    C: cobre_comm::Communicator,
{
    let is_root = comm.rank() == 0;

    // Serialize on root; non-root gets an empty placeholder.
    // Root must always supply Some(value); a missing value is a caller contract
    // violation surfaced as CliError::Internal rather than a panic.
    let serialized: Vec<u8> = if is_root {
        let Some(ref v) = value else {
            return Err(CliError::Internal {
                message: "broadcast_value: root rank must supply Some(value)".to_string(),
            });
        };
        postcard::to_allocvec(v).map_err(|e| CliError::Internal {
            message: format!("serialization error: {e}"),
        })?
    } else {
        Vec::new()
    };

    // Broadcast length so all ranks can allocate the receive buffer.
    // The length is stored as u64 (MPI-compatible) and converted to usize for
    // buffer allocation. usize is at least 32 bits on all supported targets;
    // payloads exceeding usize::MAX bytes are not realistically possible.
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
        // Rank 0 broadcasts length 0 to signal a failure before the data broadcast.
        return Err(CliError::Internal {
            message: "rank 0 signaled broadcast failure (length 0)".to_string(),
        });
    }

    // Broadcast the payload bytes.
    let mut bytes = if is_root { serialized } else { vec![0u8; len] };
    comm.broadcast(&mut bytes, 0)
        .map_err(|e| CliError::Internal {
            message: format!("broadcast error (data): {e}"),
        })?;

    // Root keeps the original value; non-root deserializes from received bytes.
    // On root, value is guaranteed Some (checked above in the serialization branch).
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

    let stderr = Term::stderr();

    if !quiet && !args.no_banner && stderr.is_term() {
        crate::banner::print_banner(&stderr);
    }

    // --- Phase 1: rank 0 loads from disk, all ranks receive via broadcast ---
    //
    // Only rank 0 accesses the filesystem. Non-root ranks may not have the
    // case directory mounted (e.g., on cluster nodes without NFS access to the
    // head node's filesystem).
    //
    // The raw Config is NOT broadcast: cobre_io::config::StoppingRuleConfig
    // uses #[serde(tag = "type")] (internally-tagged enum) which postcard
    // cannot deserialize (it refuses deserialize_any). Instead, rank 0
    // converts Config into a BroadcastConfig — a postcard-safe struct holding
    // only the fields every rank needs — and broadcasts that.
    // Rank 0 retains `root_config` for the output-writing paths.
    let (raw_system, raw_bcast_config, root_config) = if is_root {
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
        let bcast = BroadcastConfig::from_config(&config, args.skip_simulation);
        (Some(system), Some(bcast), Some(config))
    } else {
        (None, None, None)
    };

    let system = broadcast_value(raw_system, &comm)?;
    let bcast_config = broadcast_value(raw_bcast_config, &comm)?;

    let stage_templates = build_stage_templates(&system);
    if stage_templates.templates.is_empty() {
        return Err(CliError::Validation {
            report: "system has no study stages — cannot train".to_string(),
        });
    }
    let stage_templates_ref = &stage_templates.templates;
    let base_rows = &stage_templates.base_rows;

    // Build the full indexer with equipment column ranges.
    // Assumption: all stages have the same block count (uniform horizon).
    // The 1dtoy example has 1 block per stage; heterogeneous block counts
    // would require a per-stage indexer (deferred).
    let n_blks_stage0 = system.stages().first().map_or(1, |s| s.blocks.len().max(1));
    let indexer = StageIndexer::with_equipment(
        stage_templates_ref[0].n_hydro,
        stage_templates_ref[0].max_par_order,
        system.thermals().len(),
        system.lines().len(),
        system.buses().len(),
        n_blks_stage0,
    );
    let initial_state = vec![0.0_f64; indexer.n_state];

    let seed = bcast_config.seed;
    let stochastic = build_stochastic_context(&system, seed).map_err(|e| CliError::Internal {
        message: format!("stochastic context error: {e}"),
    })?;

    let forward_passes = bcast_config.forward_passes;
    let stopping_rules = stopping_rules_from_broadcast(&bcast_config);
    let n_stages = stage_templates_ref.len();
    let max_iterations = max_iterations_from_rules(&stopping_rules);

    // The FCF slot stride is the user's total forward_passes. The training
    // loop distributes work among ranks (base/remainder), so each rank's
    // forward_pass_index uses a global offset to map to unique FCF slots.
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
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

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

    // Non-root ranks take the quiet path: drain events without spawning a
    // progress bar thread. The derived `quiet` flag (args.quiet || !is_root)
    // ensures non-root ranks always enter the silent branch.
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
        None,
        None,
        &comm,
    ) {
        Ok(result) => result,
        Err(e) => {
            // event_tx was moved into TrainingConfig and is now dropped, which
            // causes the progress thread to exit its recv loop. Join before
            // returning to avoid orphaned threads.
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

    // Barrier: ensure all ranks have finished training before rank 0 writes
    // the policy checkpoint. Without this, rank 0 could write stale cut data
    // while other ranks are still computing.
    comm.barrier().map_err(|e| CliError::Internal {
        message: format!("post-training barrier error: {e}"),
    })?;

    // should_simulate was resolved on rank 0 and included in BroadcastConfig,
    // so all ranks agree on whether to run the simulation phase.
    let should_simulate = bcast_config.should_simulate;

    // Resolve output_dir once: only rank 0 ever uses it, but we need the
    // `PathBuf` to be available in all branches so borrows of `args` are
    // consistent. Non-root ranks compute a meaningless default but never
    // access the filesystem with it.
    let output_dir: PathBuf = args.output.unwrap_or_else(|| args.case_dir.join("output"));

    // Determine simulation output for summary (populated on rank 0 only).
    let simulation_output;

    if should_simulate {
        let n_scenarios = bcast_config.n_scenarios;
        let io_capacity = bcast_config.io_channel_capacity as usize;

        let sim_config = SimulationConfig {
            n_scenarios,
            io_channel_capacity: io_capacity,
        };

        let entity_counts = build_entity_counts(&system);

        // All ranks create the channel: simulate() sends results through
        // result_tx regardless of rank. Each rank collects its own subset of
        // scenario results from the channel.
        //
        // The channel is bounded at io_capacity. To prevent deadlock when
        // num_scenarios > io_capacity, a background thread drains result_rx
        // concurrently with simulate(). Without concurrent draining, simulate()
        // would block on the 65th send (default capacity 64) while the main
        // thread waits for simulate() to return — a classic bounded-channel
        // deadlock.
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity.max(1));

        // Spawn the drain thread before calling simulate() so it is ready to
        // consume items as soon as the first scenario result is sent.
        let drain_handle = std::thread::spawn(move || {
            result_rx
                .into_iter()
                .collect::<Vec<cobre_sddp::SimulationScenarioResult>>()
        });

        let sim_result = simulate(
            &mut solver,
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
        )
        .map_err(CliError::from);

        // Drop result_tx before joining drain_handle. The background thread's
        // into_iter() loop terminates only when the sender is closed; dropping
        // here ensures the join below does not deadlock regardless of whether
        // simulate() succeeded or failed.
        drop(result_tx);

        // Join the drain thread to collect results. We always join (even on
        // error) to avoid leaking the thread. The join result is only used on
        // the success path; on error the collected Vec is discarded.
        let local_results = drain_handle.join().unwrap_or_default();

        sim_result?;

        // Serialize local results with postcard for MPI transfer.
        let local_bytes =
            postcard::to_allocvec(&local_results).map_err(|e| CliError::Internal {
                message: format!("simulation result serialization error: {e}"),
            })?;

        // Exchange per-rank byte lengths via allgatherv so every rank knows
        // how many bytes each rank will contribute.
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

        // Compute displacements and total buffer size for the byte gather.
        // u64 byte lengths are converted to usize; payloads exceeding usize::MAX
        // are not possible in practice on supported targets.
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

        // Gather all serialized scenario bytes from all ranks.
        let mut all_bytes = vec![0u8; total_bytes];
        comm.allgatherv(&local_bytes, &mut all_bytes, &recv_counts, &recv_displs)
            .map_err(|e| CliError::Internal {
                message: format!("simulation result gather error: {e}"),
            })?;

        // Barrier: ensure all ranks have completed the gather before rank 0
        // proceeds to write results.
        comm.barrier().map_err(|e| CliError::Internal {
            message: format!("post-simulation barrier error: {e}"),
        })?;

        // Only rank 0 deserializes, writes Parquet files, and returns output.
        simulation_output = if is_root {
            // Write the policy checkpoint (cuts + basis as FlatBuffers).
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

            // Deserialize each rank's portion from the gathered byte buffer
            // and combine into a single contiguous result set.
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

            // Create the Parquet writer on the main thread (needs &System), then
            // write all gathered results.
            let parquet_config = cobre_io::ParquetWriterConfig::default();
            let mut sim_writer = cobre_io::output::simulation_writer::SimulationParquetWriter::new(
                &output_dir,
                &system,
                &parquet_config,
            )
            .map_err(CliError::from)?;

            // Write each scenario result through the Parquet writer.
            let mut failed: u32 = 0;
            for scenario_result in all_results {
                let payload = crate::simulation_io::convert_scenario_for_writer(scenario_result);
                if let Err(e) = sim_writer.write_scenario(payload) {
                    tracing::error!("simulation write error: {e}");
                    failed += 1;
                }
            }
            let mut sim_output = sim_writer.finalize();
            sim_output.failed = failed;

            Some(sim_output)
        } else {
            None
        };
    } else {
        simulation_output = None;

        // When skipping simulation, rank 0 still writes the policy checkpoint.
        if is_root {
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
        }
    }

    // Only rank 0 writes training/simulation summaries and prints the summary.
    // root_config is Some on rank 0 (loaded from disk above) and None on
    // non-root ranks. The is_root guard ensures we never unwrap on a non-root rank.
    if is_root {
        // root_config is guaranteed Some on rank 0.
        let config = root_config.ok_or_else(|| CliError::Internal {
            message: "root_config was None on rank 0 — internal invariant violated".to_string(),
        })?;
        write_results(
            &output_dir,
            &training_output,
            simulation_output.as_ref(),
            &system,
            &config,
        )
        .map_err(CliError::from)?;

        if !quiet {
            let summary = RunSummary {
                training: TrainingSummary {
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
                    total_lp_solves: training_output
                        .convergence_records
                        .iter()
                        .map(|r| u64::from(r.lp_solves))
                        .sum(),
                    total_time_ms: training_result.total_time_ms,
                },
                simulation: simulation_output.as_ref().map(|sim| SimulationSummary {
                    n_scenarios: sim.n_scenarios,
                    completed: sim.completed,
                    failed: sim.failed,
                }),
                output_dir: output_dir.clone(),
            };
            crate::summary::print_summary(&stderr, &summary);
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::broadcast_value;

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
}
