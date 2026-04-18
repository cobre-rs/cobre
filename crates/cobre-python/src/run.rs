//! Solver execution entry points for the `cobre.run` Python sub-module.
//!
//! Exposes [`run`] — a high-level function that replicates the lifecycle of
//! `cobre run` but without MPI, progress bars, or a terminal banner. The GIL
//! is released for the entire Rust computation so Python threads and the
//! interpreter continue to run alongside the solver.
//!
//! ## Signal handling and Ctrl-C
//!
//! While the GIL is released, Python's signal machinery cannot deliver
//! `SIGINT`. If the user presses Ctrl-C during a long training run, the
//! interrupt will be queued and delivered only after the current iteration
//! completes and control returns to the Python interpreter. This is the
//! expected MVP behaviour — progress callbacks are deferred to a future ticket.
//!
//! ## Single-process only
//!
//! This module uses [`cobre_comm::LocalBackend`] exclusively. MPI is never
//! initialized here. For distributed runs, launch `mpiexec cobre` as a
//! subprocess.

use std::path::PathBuf;
use std::sync::mpsc;

use pyo3::exceptions::{PyOSError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use cobre_comm::LocalBackend;
use cobre_io::output::simulation_writer::{ScenarioWritePayload, SimulationParquetWriter};
use cobre_io::{ParquetWriterConfig, SolverStatsRow};
use cobre_sddp::{
    ArOrderSummary, DEFAULT_SEED, EstimationReport, FutureCostFunction, HydroModelSummary,
    ModelProvenanceReport, SolverStatsDelta, StochasticSource, StochasticSummary, StudySetup,
    build_hydro_model_summary, build_provenance_report, build_stochastic_summary,
    prepare_hydro_models, prepare_stochastic,
};
use cobre_solver::HighsSolver;

/// Summary returned by [`run_inner`] on success.
struct RunSummary {
    converged: bool,
    iterations: u64,
    lower_bound: f64,
    upper_bound: Option<f64>,
    gap_percent: Option<f64>,
    total_time_ms: u64,
    output_dir: PathBuf,
    simulation: Option<SimSummary>,
    stochastic: Option<StochasticSummary>,
    hydro_models: Option<HydroModelSummary>,
    provenance: Option<ModelProvenanceReport>,
}

struct SimSummary {
    n_scenarios: u32,
    completed: u32,
}

fn init_rayon(threads: Option<u32>) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads.map_or(1, |t| t as usize))
        .build_global()
        .unwrap_or(());
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn write_policy_checkpoint(
    policy_dir: &std::path::Path,
    fcf: &FutureCostFunction,
    training_result: &cobre_sddp::TrainingResult,
    max_iterations: u64,
    forward_passes: u32,
    seed: u64,
    export_states: bool,
) -> Result<(), String> {
    use cobre_io::output::policy::{
        PolicyCheckpointMetadata, write_policy_checkpoint as io_write_policy_checkpoint,
    };
    use cobre_sddp::policy_export::{
        build_active_indices, build_stage_basis_records, build_stage_cut_records,
        build_stage_cuts_payloads, build_stage_states_payloads, convert_basis_cache,
    };

    let n_stages = fcf.pools.len();
    let state_dimension = fcf.state_dimension;

    let stage_records = build_stage_cut_records(fcf);
    let stage_active_indices = build_active_indices(&stage_records);
    let stage_cuts = build_stage_cuts_payloads(fcf, &stage_records, &stage_active_indices);

    let (basis_col_u8, basis_row_u8) = convert_basis_cache(training_result);
    let stage_bases = build_stage_basis_records(fcf, training_result, &basis_col_u8, &basis_row_u8);

    let warm_start_counts: Vec<u32> = fcf.pools.iter().map(|p| p.warm_start_count).collect();
    let metadata = PolicyCheckpointMetadata {
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: cobre_io::now_iso8601(),
        completed_iterations: training_result.iterations as u32,
        final_lower_bound: training_result.final_lb,
        best_upper_bound: Some(training_result.final_ub),
        state_dimension: state_dimension as u32,
        num_stages: n_stages as u32,
        max_iterations: max_iterations as u32,
        forward_passes,
        warm_start_cuts: warm_start_counts.iter().copied().max().unwrap_or(0),
        warm_start_counts,
        rng_seed: seed,
        total_visited_states: training_result
            .visited_archive
            .as_ref()
            .map_or(0, |a| (0..a.num_stages()).map(|t| a.count(t) as u64).sum()),
    };

    let stage_states = if export_states {
        build_stage_states_payloads(training_result.visited_archive.as_ref())
    } else {
        Vec::new()
    };

    io_write_policy_checkpoint(
        policy_dir,
        &stage_cuts,
        &stage_bases,
        &metadata,
        &stage_states,
    )
    .map_err(|e| e.to_string())
}

/// Write all applicable stochastic preprocessing artifacts to `{output_dir}/stochastic/`.
///
/// Called when `exports.stochastic` is `true` in `config.json`. Each writer call is
/// independent: failure is logged to stderr as a warning and does not prevent the remaining
/// files or training from proceeding.
///
/// Files written:
/// - `noise_openings.parquet` — always
/// - `inflow_seasonal_stats.parquet` — always
/// - `inflow_ar_coefficients.parquet` — always
/// - `correlation.json` — always
/// - `load_seasonal_stats.parquet` — only when any load model has `std_mw > 0`
/// - `fitting_report.json` — only when `estimation_report` is `Some`
fn export_stochastic_artifacts_py(
    output_dir: &std::path::Path,
    stochastic: &cobre_stochastic::StochasticContext,
    system: &cobre_core::System,
    estimation_report: Option<&EstimationReport>,
) {
    use cobre_core::scenario::LoadModel;
    use cobre_io::output::{
        write_correlation_json, write_fitting_report, write_inflow_ar_coefficients,
        write_inflow_seasonal_stats, write_load_seasonal_stats, write_noise_openings,
    };
    use cobre_io::scenarios::LoadSeasonalStatsRow;
    use cobre_sddp::{
        estimation_report_to_fitting_report, inflow_models_to_ar_rows, inflow_models_to_stats_rows,
    };

    let stochastic_dir = output_dir.join("stochastic");

    if let Err(e) = write_noise_openings(
        &stochastic_dir.join("noise_openings.parquet"),
        stochastic.opening_tree(),
    ) {
        eprintln!("cobre-python: stochastic export warning: noise_openings: {e}");
    }

    let stats_rows = inflow_models_to_stats_rows(system.inflow_models());
    if let Err(e) = write_inflow_seasonal_stats(
        &stochastic_dir.join("inflow_seasonal_stats.parquet"),
        &stats_rows,
    ) {
        eprintln!("cobre-python: stochastic export warning: inflow_seasonal_stats: {e}");
    }

    let ar_rows = inflow_models_to_ar_rows(system.inflow_models());
    if let Err(e) = write_inflow_ar_coefficients(
        &stochastic_dir.join("inflow_ar_coefficients.parquet"),
        &ar_rows,
    ) {
        eprintln!("cobre-python: stochastic export warning: inflow_ar_coefficients: {e}");
    }

    if let Err(e) = write_correlation_json(
        &stochastic_dir.join("correlation.json"),
        system.correlation(),
    ) {
        eprintln!("cobre-python: stochastic export warning: correlation: {e}");
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
            eprintln!("cobre-python: stochastic export warning: load_seasonal_stats: {e}");
        }
    }

    if let Some(report) = estimation_report {
        let fitting = estimation_report_to_fitting_report(report);
        if let Err(e) = write_fitting_report(&stochastic_dir.join("fitting_report.json"), &fitting)
        {
            eprintln!("cobre-python: stochastic export warning: fitting_report: {e}");
        }
    }
}

/// Convert a [`SolverStatsDelta`] into a [`SolverStatsRow`] for Parquet output.
///
/// The `id` parameter is the row identifier: iteration number for training phases,
/// scenario ID for the simulation phase.
#[allow(clippy::cast_possible_truncation)]
fn delta_to_stats_row(
    id: u32,
    phase: &str,
    stage: i32,
    delta: &SolverStatsDelta,
) -> SolverStatsRow {
    SolverStatsRow {
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
        clear_solver_count: delta.clear_solver_count,
        clear_solver_failures: delta.clear_solver_failures,
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

/// Result of the training phase within `run_inner`.
struct TrainingPhaseResult {
    result: cobre_sddp::TrainingResult,
    output: cobre_io::TrainingOutput,
    error: Option<cobre_sddp::SddpError>,
    started_at: String,
}

/// Run the training phase: solver init, train, write outputs.
fn run_training_phase_py(
    setup: &mut StudySetup,
    n_threads: usize,
    warm_start_basis_mode: cobre_io::config::WarmStartBasisMode,
) -> Result<TrainingPhaseResult, String> {
    use cobre_solver::highs::WarmStartBasisMode as SolverMode;

    // Convert from the I/O config type to the solver-crate mirrored type.
    let solver_mode = match warm_start_basis_mode {
        cobre_io::config::WarmStartBasisMode::AlienOnly => SolverMode::AlienOnly,
        cobre_io::config::WarmStartBasisMode::NonAlienFirst => SolverMode::NonAlienFirst,
    };

    let started_at = cobre_io::now_iso8601();
    let mut solver = HighsSolver::new()
        .map_err(|e| format!("HiGHS initialisation failed: {e}"))?
        .with_warm_start_mode(solver_mode);
    let (event_tx, event_rx) = mpsc::channel();
    let training_outcome = setup
        .train(
            &mut solver,
            &LocalBackend,
            n_threads,
            move || HighsSolver::new().map(|s| s.with_warm_start_mode(solver_mode)),
            Some(event_tx),
            None,
        )
        .map_err(|e| format!("training error: {e}"))?;
    let training_result = training_outcome.result;

    let events: Vec<_> = event_rx.try_iter().collect();
    let training_output = setup.build_training_output(&training_result, &events);

    Ok(TrainingPhaseResult {
        result: training_result,
        output: training_output,
        error: training_outcome.error,
        started_at,
    })
}

/// Write all training artifacts: policy checkpoint, training results, solver stats,
/// and cut selection records.
fn write_training_artifacts(
    output_dir: &std::path::Path,
    system: &cobre_core::System,
    config: &cobre_io::Config,
    setup: &StudySetup,
    training: &TrainingPhaseResult,
    seed: u64,
    n_threads: usize,
) -> Result<(), String> {
    write_policy_checkpoint(
        &output_dir.join(setup.policy_path()),
        setup.fcf(),
        &training.result,
        setup.max_iterations(),
        setup.forward_passes(),
        seed,
        config.exports.states,
    )
    .map_err(|e| format!("policy checkpoint error: {e}"))?;

    if !training.result.solver_stats_log.is_empty() {
        let rows: Vec<SolverStatsRow> = training
            .result
            .solver_stats_log
            .iter()
            .map(|(iter, phase, stage, delta)| {
                #[allow(clippy::cast_possible_truncation)]
                delta_to_stats_row(*iter as u32, phase, *stage, delta)
            })
            .collect();
        cobre_io::write_solver_stats(output_dir, &rows)
            .map_err(|e| format!("solver stats output: {e}"))?;
    }

    if !training.output.cut_selection_records.is_empty() {
        cobre_io::write_cut_selection_records(
            output_dir,
            &training.output.cut_selection_records,
            &ParquetWriterConfig::default(),
        )
        .map_err(|e| format!("cut selection output: {e}"))?;
    }

    let training_ctx = cobre_io::OutputContext {
        hostname: cobre_io::get_hostname(),
        solver: "highs".to_string(),
        solver_version: Some(cobre_solver::highs_version()),
        started_at: training.started_at.clone(),
        completed_at: cobre_io::now_iso8601(),
        distribution: cobre_io::DistributionInfo {
            backend: "local".to_string(),
            world_size: 1,
            ranks_participated: 1,
            num_nodes: 1,
            threads_per_rank: u32::try_from(n_threads).unwrap_or(u32::MAX),
            mpi_library: None,
            mpi_standard: None,
            thread_level: None,
            slurm_job_id: None,
        },
    };
    cobre_io::write_training_results(output_dir, &training.output, system, config, &training_ctx)
        .map_err(|e| format!("training results output: {e}"))?;

    Ok(())
}

/// Run the simulation phase: workspace pool, Parquet writing, and output.
fn run_simulation_phase_py(
    setup: &mut StudySetup,
    output_dir: &std::path::Path,
    system: &cobre_core::System,
    training_result: &cobre_sddp::TrainingResult,
    n_threads: usize,
    warm_start_basis_mode: cobre_io::config::WarmStartBasisMode,
) -> Result<SimSummary, String> {
    use cobre_solver::highs::WarmStartBasisMode as SolverMode;

    let solver_mode = match warm_start_basis_mode {
        cobre_io::config::WarmStartBasisMode::AlienOnly => SolverMode::AlienOnly,
        cobre_io::config::WarmStartBasisMode::NonAlienFirst => SolverMode::NonAlienFirst,
    };

    let sim_started_at = cobre_io::now_iso8601();
    let io_capacity = setup.simulation_config().io_channel_capacity;
    let mut sim_pool = setup
        .create_workspace_pool(n_threads, move || {
            HighsSolver::new().map(|s| s.with_warm_start_mode(solver_mode))
        })
        .map_err(|e| format!("HiGHS initialisation failed for simulation pool: {e}"))?;
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity.max(1));

    let sim_writer =
        SimulationParquetWriter::new(output_dir, system, &ParquetWriterConfig::default())
            .map_err(|e| format!("simulation writer initialisation error: {e}"))?;

    let drain_handle = std::thread::spawn(move || {
        let mut writer = sim_writer;
        let mut failed: u32 = 0;
        for scenario_result in result_rx {
            if let Err(e) = writer.write_scenario(ScenarioWritePayload::from(scenario_result)) {
                eprintln!("cobre-python: simulation write warning: {e}");
                failed += 1;
            }
        }
        (writer, failed)
    });

    let sim_result = setup
        .simulate(
            &mut sim_pool.workspaces,
            &LocalBackend,
            &result_tx,
            None,
            training_result.baked_templates.as_deref(),
            &training_result.basis_cache,
        )
        .map_err(|e| format!("simulation error: {e}"));
    drop(result_tx);

    let (sim_writer, write_failures) = drain_handle
        .join()
        .map_err(|_| "drain thread panicked".to_string())?;
    let sim_run_result = sim_result?;

    let mut sim_out = sim_writer.finalize(0);
    sim_out.failed = write_failures;

    if !sim_run_result.solver_stats.is_empty() {
        let rows: Vec<SolverStatsRow> = sim_run_result
            .solver_stats
            .iter()
            .map(|(scenario_id, delta)| delta_to_stats_row(*scenario_id, "simulation", -1, delta))
            .collect();
        cobre_io::write_simulation_solver_stats(output_dir, &rows)
            .map_err(|e| format!("simulation solver stats output: {e}"))?;
    }

    let sim_summary = SimSummary {
        n_scenarios: sim_out.n_scenarios,
        completed: sim_out.completed,
    };
    let sim_ctx = cobre_io::OutputContext {
        hostname: cobre_io::get_hostname(),
        solver: "highs".to_string(),
        solver_version: Some(cobre_solver::highs_version()),
        started_at: sim_started_at,
        completed_at: cobre_io::now_iso8601(),
        distribution: cobre_io::DistributionInfo {
            backend: "local".to_string(),
            world_size: 1,
            ranks_participated: 1,
            num_nodes: 1,
            threads_per_rank: u32::try_from(n_threads).unwrap_or(u32::MAX),
            mpi_library: None,
            mpi_standard: None,
            thread_level: None,
            slurm_job_id: None,
        },
    };
    cobre_io::write_simulation_results(output_dir, &sim_out, &sim_ctx)
        .map_err(|e| format!("simulation results output: {e}"))?;

    Ok(sim_summary)
}

/// Run the full solve lifecycle without MPI or progress bars (GIL released for computation).
fn run_inner(
    case_dir: &std::path::Path,
    output_dir: PathBuf,
    threads: Option<u32>,
    skip_simulation: bool,
) -> Result<RunSummary, String> {
    init_rayon(threads);
    let n_threads = threads.map_or(1, |t| t as usize);

    let system = cobre_io::load_case(case_dir).map_err(|e| e.to_string())?;
    let config = cobre_io::parse_config(&case_dir.join("config.json"))
        .map_err(|e| format!("config parse error: {e}"))?;

    let seed = config
        .training
        .tree_seed
        .map_or(DEFAULT_SEED, i64::unsigned_abs);
    let should_simulate =
        !skip_simulation && config.simulation.enabled && config.simulation.num_scenarios > 0;

    let training_source = config
        .training_scenario_source(&case_dir.join("config.json"))
        .map_err(|e| format!("scenario source error: {e}"))?;

    let result = prepare_stochastic(system, case_dir, &config, seed, &training_source)
        .map_err(|e| format!("stochastic preprocessing error: {e}"))?;
    let system = result.system;
    let estimation_report = result.estimation_report;
    let estimation_path = result.estimation_path;

    let hydro_models_result = prepare_hydro_models(&system, case_dir)
        .map_err(|e| format!("hydro model preprocessing error: {e}"))?;

    let mut setup = StudySetup::new(&system, &config, result.stochastic, hydro_models_result)
        .map_err(|e| e.to_string())?;
    setup.set_export_states(config.exports.states);

    let provenance_report = build_provenance_report(
        estimation_path,
        estimation_report.as_ref(),
        setup.stochastic().provenance(),
        system.hydros().len(),
    );

    if config.exports.stochastic {
        export_stochastic_artifacts_py(
            &output_dir,
            setup.stochastic(),
            &system,
            estimation_report.as_ref(),
        );
    }

    let scaling_path = output_dir.join("training/scaling_report.json");
    cobre_io::write_scaling_report(&scaling_path, setup.scaling_report())
        .map_err(|e| format!("failed to write scaling report: {e}"))?;

    let provenance_path = output_dir.join("training/model_provenance.json");
    if let Err(e) = cobre_io::write_provenance_report(&provenance_path, &provenance_report) {
        eprintln!("cobre-python: provenance output warning: {e}");
    }

    let stochastic_summary = build_stochastic_summary(
        &system,
        setup.stochastic(),
        estimation_report.as_ref(),
        seed,
    );
    let hydro_models_summary = Some(build_hydro_model_summary(setup.hydro_models(), &system));

    let training_enabled = config.training.enabled;

    if training_enabled {
        // Warm-start: load prior policy and inject cuts before training.
        if config.policy.mode == cobre_io::PolicyMode::WarmStart {
            let policy_dir = output_dir.join(setup.policy_path());
            if !policy_dir.exists() {
                return Err(format!(
                    "Policy directory not found: {}. Cannot warm-start \
                     without a prior policy.",
                    policy_dir.display()
                ));
            }

            let checkpoint = cobre_io::output::policy::read_policy_checkpoint(&policy_dir)
                .map_err(|e| format!("failed to read policy checkpoint: {e}"))?;

            if config.policy.validate_compatibility {
                #[allow(clippy::cast_possible_truncation)]
                let n_stages = system.stages().iter().filter(|s| s.id >= 0).count() as u32;
                let state_dim = setup.fcf().state_dimension as u32;
                cobre_sddp::validate_policy_compatibility(
                    &checkpoint.metadata,
                    state_dim,
                    n_stages,
                )
                .map_err(|e| format!("policy validation error: {e}"))?;
            }

            // Reserve one extra slot for cuts added in the final iteration.
            let warm_fcf = cobre_sddp::FutureCostFunction::new_with_warm_start(
                &checkpoint.stage_cuts,
                setup.forward_passes(),
                setup.max_iterations().saturating_add(1),
            )
            .map_err(|e| format!("warm-start FCF construction error: {e}"))?;
            setup.replace_fcf(warm_fcf);
        } else if config.policy.mode == cobre_io::PolicyMode::Resume {
            let policy_dir = output_dir.join(setup.policy_path());
            if !policy_dir.exists() {
                return Err(format!(
                    "Policy directory not found: {}. Cannot resume \
                     without a prior checkpoint.",
                    policy_dir.display()
                ));
            }

            let checkpoint = cobre_io::output::policy::read_policy_checkpoint(&policy_dir)
                .map_err(|e| format!("failed to read policy checkpoint: {e}"))?;

            if config.policy.validate_compatibility {
                #[allow(clippy::cast_possible_truncation)]
                let n_stages = system.stages().iter().filter(|s| s.id >= 0).count() as u32;
                let state_dim = setup.fcf().state_dimension as u32;
                cobre_sddp::validate_policy_compatibility(
                    &checkpoint.metadata,
                    state_dim,
                    n_stages,
                )
                .map_err(|e| format!("policy validation error: {e}"))?;
            }

            let completed = u64::from(checkpoint.metadata.completed_iterations);

            // Reserve one extra slot for cuts added in the final iteration.
            let warm_fcf = cobre_sddp::FutureCostFunction::new_with_warm_start(
                &checkpoint.stage_cuts,
                setup.forward_passes(),
                setup.max_iterations().saturating_add(1),
            )
            .map_err(|e| format!("resume FCF construction error: {e}"))?;
            setup.replace_fcf(warm_fcf);
            setup.set_start_iteration(completed);
        }

        // Boundary cuts — orthogonal to policy mode. Runs after warm-start/resume
        // so that both compose correctly: warm-start replaces the entire FCF first,
        // then boundary cuts overwrite only the terminal pool.
        if let Some(ref bp) = config.policy.boundary {
            let boundary_path = output_dir.join(&bp.path);
            #[allow(clippy::cast_possible_truncation)]
            let state_dim = setup.fcf().state_dimension as u32;
            let boundary_records =
                cobre_sddp::load_boundary_cuts(&boundary_path, bp.source_stage, state_dim)
                    .map_err(|e| format!("boundary cut error: {e}"))?;
            cobre_sddp::inject_boundary_cuts(&mut setup, &boundary_records);
        }

        let training = run_training_phase_py(
            &mut setup,
            n_threads,
            config.training.solver.warm_start_basis_mode,
        )?;

        write_training_artifacts(
            &output_dir,
            &system,
            &config,
            &setup,
            &training,
            seed,
            n_threads,
        )?;

        if let Some(ref e) = training.error {
            return Err(format!(
                "training failed after {} iterations: {e}",
                training.result.iterations
            ));
        }

        let simulation = if should_simulate {
            Some(run_simulation_phase_py(
                &mut setup,
                &output_dir,
                &system,
                &training.result,
                n_threads,
                config.training.solver.warm_start_basis_mode,
            )?)
        } else {
            None
        };

        Ok(RunSummary {
            converged: training.output.converged,
            iterations: training.result.iterations,
            lower_bound: training.result.final_lb,
            upper_bound: Some(training.result.final_ub),
            gap_percent: Some(training.result.final_gap * 100.0),
            total_time_ms: training.result.total_time_ms,
            output_dir,
            simulation,
            stochastic: Some(stochastic_summary),
            hydro_models: hydro_models_summary,
            provenance: Some(provenance_report),
        })
    } else {
        // Training disabled: check if simulation is requested.
        if should_simulate {
            // Simulation-only mode: load policy and run simulation.
            let policy_dir = output_dir.join(setup.policy_path());
            if !policy_dir.exists() {
                return Err(format!(
                    "Policy directory not found: {}. Cannot run simulation-only \
                     mode without a trained policy.",
                    policy_dir.display()
                ));
            }

            let checkpoint = cobre_io::output::policy::read_policy_checkpoint(&policy_dir)
                .map_err(|e| format!("failed to read policy checkpoint: {e}"))?;

            // Validate compatibility if configured.
            if config.policy.validate_compatibility {
                #[allow(clippy::cast_possible_truncation)]
                let n_stages = system.stages().iter().filter(|s| s.id >= 0).count() as u32;
                let state_dim = setup.fcf().state_dimension as u32;
                cobre_sddp::validate_policy_compatibility(
                    &checkpoint.metadata,
                    state_dim,
                    n_stages,
                )
                .map_err(|e| format!("policy validation error: {e}"))?;
            }

            // Replace the empty FCF with the loaded one.
            let loaded_fcf =
                cobre_sddp::FutureCostFunction::from_deserialized(&checkpoint.stage_cuts)
                    .map_err(|e| format!("FCF reconstruction error: {e}"))?;
            setup.replace_fcf(loaded_fcf);

            // Build basis cache from loaded checkpoint.
            let basis_cache = cobre_sddp::build_basis_cache_from_checkpoint(
                setup.num_stages(),
                &checkpoint.stage_bases,
            );

            // Create a minimal TrainingResult for simulation warm-start.
            let training_result = cobre_sddp::TrainingResult {
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
                // simulation path will use the fallback load_model + add_rows
                // path.
                baked_templates: None,
            };

            let simulation = Some(run_simulation_phase_py(
                &mut setup,
                &output_dir,
                &system,
                &training_result,
                n_threads,
                config.training.solver.warm_start_basis_mode,
            )?);

            return Ok(RunSummary {
                converged: false,
                iterations: 0,
                lower_bound: checkpoint.metadata.final_lower_bound,
                upper_bound: checkpoint.metadata.best_upper_bound,
                gap_percent: None,
                total_time_ms: 0,
                output_dir,
                simulation,
                stochastic: Some(stochastic_summary),
                hydro_models: hydro_models_summary,
                provenance: Some(provenance_report),
            });
        }

        // Both training and simulation disabled — return zero-iteration summary.
        Ok(RunSummary {
            converged: false,
            iterations: 0,
            lower_bound: 0.0,
            upper_bound: None,
            gap_percent: None,
            total_time_ms: 0,
            output_dir,
            simulation: None,
            stochastic: Some(stochastic_summary),
            hydro_models: hydro_models_summary,
            provenance: Some(provenance_report),
        })
    }
}

/// Convert a [`StochasticSource`] enum variant to a Python string or `None`.
fn stochastic_source_str(source: &StochasticSource) -> Option<&'static str> {
    match source {
        StochasticSource::Estimated => Some("estimated"),
        StochasticSource::Loaded => Some("loaded"),
        StochasticSource::None => None,
    }
}

/// Convert an [`ArOrderSummary`] to a Python dict.
fn ar_order_to_dict<'py>(
    py: Python<'py>,
    summary: &ArOrderSummary,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("method", &summary.method)?;
    dict.set_item("min_order", summary.min_order)?;
    dict.set_item("max_order", summary.max_order)?;
    dict.set_item("n_hydros", summary.n_hydros)?;
    dict.set_item("order_counts", summary.order_counts.clone())?;
    Ok(dict)
}

/// Convert a [`HydroModelSummary`] to a Python dict.
fn hydro_model_summary_to_dict<'py>(
    py: Python<'py>,
    summary: &HydroModelSummary,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("n_constant", summary.n_constant)?;
    dict.set_item("n_fpha", summary.n_fpha)?;
    dict.set_item("total_planes", summary.total_planes)?;
    dict.set_item("n_evaporation", summary.n_evaporation)?;
    dict.set_item("n_no_evaporation", summary.n_no_evaporation)?;
    Ok(dict)
}

/// Convert a [`StochasticSummary`] to a Python dict.
fn stochastic_summary_to_dict<'py>(
    py: Python<'py>,
    summary: &StochasticSummary,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        "inflow_source",
        stochastic_source_str(&summary.inflow_source),
    )?;
    dict.set_item("n_hydros", summary.n_hydros)?;
    dict.set_item("n_seasons", summary.n_seasons)?;
    if let Some(ar) = &summary.ar_summary {
        let ar_dict = ar_order_to_dict(py, ar)?;
        dict.set_item("ar_order", ar_dict)?;
    } else {
        dict.set_item("ar_order", py.None())?;
    }
    dict.set_item(
        "correlation_source",
        stochastic_source_str(&summary.correlation_source),
    )?;
    dict.set_item("correlation_dim", summary.correlation_dim.as_deref())?;
    dict.set_item(
        "opening_tree_source",
        stochastic_source_str(&summary.opening_tree_source),
    )?;
    dict.set_item("openings_per_stage", summary.openings_per_stage.clone())?;
    dict.set_item("n_stages", summary.n_stages)?;
    dict.set_item("n_load_buses", summary.n_load_buses)?;
    dict.set_item("seed", summary.seed)?;
    Ok(dict)
}

/// Convert a [`ModelProvenanceReport`] to a Python dict.
fn provenance_to_dict<'py>(
    py: Python<'py>,
    report: &ModelProvenanceReport,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("estimation_path", &report.estimation_path)?;
    dict.set_item(
        "seasonal_stats_source",
        report.seasonal_stats_source.to_string(),
    )?;
    dict.set_item(
        "ar_coefficients_source",
        report.ar_coefficients_source.to_string(),
    )?;
    dict.set_item("correlation_source", report.correlation_source.to_string())?;
    dict.set_item(
        "opening_tree_source",
        report.opening_tree_source.to_string(),
    )?;
    dict.set_item("n_hydros", report.n_hydros)?;
    dict.set_item("ar_method", report.ar_method.as_deref())?;
    dict.set_item("ar_max_order", report.ar_max_order)?;
    dict.set_item(
        "white_noise_fallbacks",
        report.white_noise_fallbacks.clone(),
    )?;
    Ok(dict)
}

/// Load a case, train an SDDP policy, optionally simulate, and write results.
/// GIL is released for the entire Rust computation.
/// Returns a dict with keys: `"converged"`, `"iterations"`, `"lower_bound"`, `"upper_bound"`,
/// `"gap_percent"`, `"total_time_ms"`, `"output_dir"`, `"simulation"`, `"stochastic"`,
/// `"hydro_models"`, `"provenance"`.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
#[pyo3(signature = (case_dir, output_dir=None, threads=None, skip_simulation=None))]
pub fn run(
    py: Python<'_>,
    case_dir: PathBuf,
    output_dir: Option<PathBuf>,
    threads: Option<u32>,
    skip_simulation: Option<bool>,
) -> PyResult<Py<PyAny>> {
    if !case_dir.exists() {
        return Err(PyOSError::new_err(format!(
            "case directory does not exist: {}",
            case_dir.display()
        )));
    }

    let resolved_output = output_dir.unwrap_or_else(|| case_dir.join("output"));
    let skip = skip_simulation.unwrap_or(false);

    let result: Result<RunSummary, String> =
        py.detach(move || run_inner(&case_dir, resolved_output, threads, skip));

    match result {
        Ok(summary) => {
            let dict = PyDict::new(py);
            dict.set_item("converged", summary.converged)?;
            dict.set_item("iterations", summary.iterations)?;
            dict.set_item("lower_bound", summary.lower_bound)?;
            dict.set_item("upper_bound", summary.upper_bound)?;
            dict.set_item("gap_percent", summary.gap_percent)?;
            dict.set_item("total_time_ms", summary.total_time_ms)?;
            dict.set_item("output_dir", summary.output_dir.to_string_lossy().as_ref())?;

            dict.set_item(
                "simulation",
                if let Some(sim) = summary.simulation {
                    let sim_dict = PyDict::new(py);
                    sim_dict.set_item("n_scenarios", sim.n_scenarios)?;
                    sim_dict.set_item("completed", sim.completed)?;
                    sim_dict.into()
                } else {
                    py.None()
                },
            )?;

            let stochastic_val = if let Some(stoch) = &summary.stochastic {
                stochastic_summary_to_dict(py, stoch)?.into()
            } else {
                py.None()
            };
            dict.set_item("stochastic", stochastic_val)?;

            let hydro_val = if let Some(hydro) = &summary.hydro_models {
                hydro_model_summary_to_dict(py, hydro)?.into()
            } else {
                py.None()
            };
            dict.set_item("hydro_models", hydro_val)?;

            let provenance_val = if let Some(prov) = &summary.provenance {
                provenance_to_dict(py, prov)?.into()
            } else {
                py.None()
            };
            dict.set_item("provenance", provenance_val)?;

            Ok(dict.into())
        }
        Err(msg) => {
            let err_fn = if msg.as_str().starts_with("output write error")
                || msg.as_str().starts_with("policy checkpoint error")
            {
                PyOSError::new_err
            } else {
                PyRuntimeError::new_err
            };
            Err(err_fn(msg))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use cobre_sddp::setup::prepare_stochastic;

    #[test]
    fn prepare_stochastic_succeeds_for_d01_case_via_python_path() {
        let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("cobre-python parent")
            .parent()
            .expect("crates parent")
            .join("examples/deterministic/d01-thermal-dispatch");

        let system = cobre_io::load_case(&case_dir).expect("load_case must succeed for D01");
        let config = cobre_io::parse_config(&case_dir.join("config.json"))
            .expect("parse_config must succeed for D01");

        let seed = config.training.tree_seed.map_or(42_u64, i64::unsigned_abs);

        let training_source = config
            .training_scenario_source(&case_dir.join("config.json"))
            .expect("training_scenario_source must succeed for D01");

        let result = prepare_stochastic(system, &case_dir, &config, seed, &training_source);
        assert!(
            result.is_ok(),
            "prepare_stochastic failed for D01 via Python path: {:?}",
            result.err()
        );
    }
}
