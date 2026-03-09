//! `cobre run <CASE_DIR>` subcommand.
//!
//! Executes the full solve lifecycle:
//!
//! 1. Load the case directory (`cobre_io::load_case`).
//! 2. Parse `config.json` (`cobre_io::parse_config`).
//! 3. Build stage LP templates (`cobre_sddp::build_stage_templates`).
//! 4. Build the stochastic context (`cobre_stochastic::build_stochastic_context`).
//! 5. Train the SDDP policy (`cobre_sddp::train`).
//! 6. Optionally run simulation (`cobre_sddp::simulate`).
//! 7. Write all outputs (`cobre_io::write_results`).

use std::path::PathBuf;
use std::sync::mpsc;

use clap::Args;
use console::Term;

use cobre_comm::LocalBackend;
use cobre_core::TrainingEvent;
use cobre_io::{SimulationOutput, write_results};
use cobre_sddp::{
    EntityCounts, FutureCostFunction, HorizonMode, RiskMeasure, SimulationConfig, StageIndexer,
    StoppingMode, StoppingRule, StoppingRuleSet, TrainingConfig, build_stage_templates,
    build_training_output, simulate, train,
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

/// Execute the `run` subcommand.
///
/// Runs the full lifecycle: load → build templates → train → simulate → write.
///
/// # Errors
///
/// Returns [`CliError`] when loading, training, simulation, or I/O fails.
/// The exit code indicates the category of failure.
#[allow(clippy::too_many_lines)]
pub fn execute(args: RunArgs) -> Result<(), CliError> {
    let stderr = Term::stderr();

    if !args.quiet && !args.no_banner && stderr.is_term() {
        crate::banner::print_banner(&stderr);
    }

    if !args.case_dir.exists() {
        return Err(CliError::Io {
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "case directory does not exist",
            ),
            context: args.case_dir.display().to_string(),
        });
    }
    if !args.quiet {
        let _ = stderr.write_line(&format!("Loading case: {}", args.case_dir.display()));
    }
    let system = cobre_io::load_case(&args.case_dir)?;

    let config_path = args.case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path)?;

    let stage_templates = build_stage_templates(&system);
    if stage_templates.templates.is_empty() {
        return Err(CliError::Validation {
            report: "system has no study stages — cannot train".to_string(),
        });
    }
    let stage_templates_ref = &stage_templates.templates;
    let base_rows = &stage_templates.base_rows;

    let indexer = StageIndexer::from_stage_template(&stage_templates_ref[0]);
    let initial_state = vec![0.0_f64; indexer.n_state];

    let seed = config.training.seed.map_or(DEFAULT_SEED, i64::unsigned_abs);
    let stochastic = build_stochastic_context(&system, seed).map_err(|e| CliError::Internal {
        message: format!("stochastic context error: {e}"),
    })?;

    let forward_passes = config
        .training
        .forward_passes
        .unwrap_or(DEFAULT_FORWARD_PASSES);
    let stopping_rules = resolve_stopping_rules(&config.training);
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

    let quiet_rx: Option<mpsc::Receiver<TrainingEvent>>;
    let progress_handle = if args.quiet {
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
        &LocalBackend,
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

    let should_simulate =
        !args.skip_simulation && config.simulation.enabled && config.simulation.num_scenarios > 0;

    let simulation_output = if should_simulate {
        let n_scenarios = config.simulation.num_scenarios;
        let io_capacity = config.simulation.io_channel_capacity as usize;

        let sim_config = SimulationConfig {
            n_scenarios,
            io_channel_capacity: io_capacity,
        };

        let entity_counts = build_entity_counts(&system);
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity.max(1));

        let io_thread = std::thread::spawn(move || {
            let mut completed: u32 = 0;
            while result_rx.recv().is_ok() {
                completed += 1;
            }
            (completed, 0_u32)
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
            &LocalBackend,
            &result_tx,
        )
        .map_err(CliError::from);

        drop(result_tx);

        let (completed, failed) = io_thread.join().map_err(|_| CliError::Internal {
            message: "simulation I/O thread panicked".to_string(),
        })?;

        sim_result?;

        Some(SimulationOutput {
            n_scenarios,
            completed,
            failed,
            partitions_written: vec![],
        })
    } else {
        None
    };

    let output_dir = args.output.unwrap_or_else(|| args.case_dir.join("output"));

    write_results(
        &output_dir,
        &training_output,
        simulation_output.as_ref(),
        &system,
        &config,
    )
    .map_err(CliError::from)?;

    if !args.quiet {
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

    Ok(())
}

/// Resolve the [`StoppingRuleSet`] from the IO training config.
///
/// Falls back to an iteration-limit of [`DEFAULT_MAX_ITERATIONS`] when the
/// config is absent.
fn resolve_stopping_rules(training: &cobre_io::config::TrainingConfig) -> StoppingRuleSet {
    use cobre_io::config::StoppingRuleConfig;

    let configs = match &training.stopping_rules {
        Some(rules) if !rules.is_empty() => rules.clone(),
        _ => vec![StoppingRuleConfig::IterationLimit {
            limit: u32::try_from(DEFAULT_MAX_ITERATIONS).unwrap_or(u32::MAX),
        }],
    };

    let rules: Vec<StoppingRule> = configs
        .into_iter()
        .map(|c| match c {
            StoppingRuleConfig::IterationLimit { limit } => StoppingRule::IterationLimit {
                limit: u64::from(limit),
            },
            StoppingRuleConfig::TimeLimit { seconds } => StoppingRule::TimeLimit { seconds },
            StoppingRuleConfig::BoundStalling {
                iterations,
                tolerance,
            } => StoppingRule::BoundStalling {
                iterations: u64::from(iterations),
                tolerance,
            },
            StoppingRuleConfig::Simulation { .. } => {
                // Simulation stopping rule requires upper-bound evaluation;
                // not implemented in Phase 8 MVP — fall back to a large iteration limit.
                StoppingRule::IterationLimit {
                    limit: DEFAULT_MAX_ITERATIONS,
                }
            }
        })
        .collect();

    let mode = if training.stopping_mode.eq_ignore_ascii_case("all") {
        StoppingMode::All
    } else {
        StoppingMode::Any
    };

    StoppingRuleSet { rules, mode }
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
    EntityCounts {
        hydro_ids: system.hydros().iter().map(|h| h.id.0).collect(),
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
