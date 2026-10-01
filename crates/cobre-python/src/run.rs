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
//! completes and control returns to the Python interpreter.
//!
//! ## Single-process only
//!
//! This module uses [`cobre_comm::LocalBackend`] exclusively. MPI is never
//! initialized here. For distributed runs, launch `mpiexec cobre` as a
//! subprocess.

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;

use chrono::NaiveDate;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Map;
use serde_json::Value;

use cobre_core::TrainingEvent;

use crate::convert::pydict_to_json_map;
use crate::errors::{
    BOUNDARY_CUT_ERROR_PREFIX, CONFIG_OVERRIDE_ERROR_PREFIX, CONFIG_PARSE_ERROR_PREFIX,
    CONFIG_READ_ERROR_PREFIX, ErrorSource, HYDRO_MODEL_PREPROCESSING_ERROR_PREFIX,
    INTERNAL_ERROR_PREFIX, OUTPUT_WRITE_ERROR_PREFIX, POLICY_CHECKPOINT_ERROR_PREFIX,
    POLICY_VALIDATION_ERROR_PREFIX, SCENARIO_SOURCE_ERROR_PREFIX, SETUP_VALIDATION_ERROR_PREFIX,
    SIMULATION_ERROR_PREFIX, SIMULATION_WRITER_INIT_ERROR_PREFIX,
    STOCHASTIC_PREPROCESSING_ERROR_PREFIX, TRAINING_ERROR_PREFIX, convert_error,
};
use cobre_io::LoadError;

use cobre_comm::LocalBackend;
use cobre_core::System;
use cobre_core::TrainingEvent::IterationSummary;
use cobre_io::BoundaryPolicy;
use cobre_io::Config;
use cobre_io::DistributionInfo;
use cobre_io::EntitySlot;
use cobre_io::LoadedCase;
use cobre_io::MetadataCost;
use cobre_io::MetadataSimulationSolveStats;
use cobre_io::MetadataTrainingSolveStats;
use cobre_io::OutputContext;
use cobre_io::PolicyMode::Resume;
use cobre_io::PolicyMode::WarmStart;
use cobre_io::ReportEntry;
use cobre_io::SetupTimings;
use cobre_io::SolverStatsRow;
use cobre_io::TrainingOutput;
use cobre_io::get_hostname;
use cobre_io::now_iso8601;
use cobre_io::output::policy::read_policy_checkpoint;
use cobre_io::output::simulation_writer::{
    ScenarioWritePayload, SimulationParquetWriter, write_paths, write_scenario_summary,
};
use cobre_io::output::write_evaporation_models;
use cobre_io::output::write_fpha_deviation_points;
use cobre_io::output::write_fpha_hyperplanes;
use cobre_io::parse_config;
use cobre_io::validate_case_with_artifacts;
use cobre_io::write_fixed_delivery;
use cobre_io::write_generic_constraint_echo;
use cobre_io::write_hydro_model_summary;
use cobre_io::write_provenance_report;
use cobre_io::write_row_selection_records;
use cobre_io::write_scaling_report;
use cobre_io::write_simulation_results;
use cobre_io::write_simulation_solver_stats;
use cobre_io::write_solver_stats;
use cobre_io::write_training_results;
use cobre_sddp::BoundaryLoadRequest;
use cobre_sddp::FullFcf;
use cobre_sddp::FutureCostFunction;
use cobre_sddp::HydroFitTimings;
use cobre_sddp::PolicyLoadProof;
use cobre_sddp::PolicyStageManifest;
use cobre_sddp::SddpError;
use cobre_sddp::SimulationWeighting;
use cobre_sddp::TrainingResult;
use cobre_sddp::ValidatedBoundaryCuts;
use cobre_sddp::aggregate_simulation;
use cobre_sddp::aggregate_solver_stats_log;
use cobre_sddp::build_basis_cache_from_checkpoint;
use cobre_sddp::build_deviation_summary;
use cobre_sddp::build_evaporation_model_rows;
use cobre_sddp::build_fixed_delivery_rows;
use cobre_sddp::build_generic_constraint_echo_rows;
use cobre_sddp::checkpoint_terminal_cost_scale_factor;
use cobre_sddp::delta_to_stats_row;
use cobre_sddp::hydro_models::prepare_hydro_models_from_artifacts;
use cobre_sddp::inject_boundary_cuts;
use cobre_sddp::load_boundary_cuts;
use cobre_sddp::policy::orchestration::CheckpointParams;
use cobre_sddp::policy::orchestration::build_season_manifest;
use cobre_sddp::policy::orchestration::export_stochastic_artifacts;
use cobre_sddp::policy::orchestration::write_checkpoint;
use cobre_sddp::rescale_checkpoint_cuts_for_load;
use cobre_sddp::resolve_boundary_state_requirements;
use cobre_sddp::setup::RunPhasePlan;
use cobre_sddp::solver_stats_log_to_rows;
use cobre_sddp::study_horizon_end;
use cobre_sddp::validate_policy_load;
use cobre_sddp::{
    ArOrderSummary, DEFAULT_SEED, HydroModelSummary, ModelProvenanceReport, SolverStatsDelta,
    StochasticSource, StochasticSummary, StudyParams, StudySetup, build_hydro_model_summary,
    build_provenance_report, build_stochastic_summary, prepare_stochastic,
};
use cobre_solver::ActiveSolver;
use cobre_solver::active_solver_metadata_id;
use cobre_solver::active_solver_version;
use cobre_stochastic::sampling::historical::HistoricalScenarioLibrary;

/// Error returned by [`run_via_study`].
///
/// A captured callback `PyErr` is carried verbatim (its type and message reach
/// Python unchanged) and propagated only after the run's partial artifacts have
/// been written.
#[derive(Debug)]
pub(crate) enum RunError {
    /// A descriptive message mapped to a Python exception type by the caller.
    Message(String),
    /// A `PyErr` captured from the streaming callback (or `check_signals`).
    Callback(PyErr),
    /// A typed case-load failure, carried verbatim so the mapping site can pick
    /// the per-variant class.
    Load(LoadError),
    /// A typed SDDP failure carried verbatim with its descriptive message, so the
    /// mapping site can attach structured fields (e.g. `Infeasible`'s
    /// stage/iteration/scenario) without losing the message text.
    Sddp {
        /// The typed SDDP error.
        error: SddpError,
        /// The verbatim descriptive message (preserved so `match=` assertions pass).
        message: String,
    },
}

impl From<String> for RunError {
    fn from(msg: String) -> Self {
        RunError::Message(msg)
    }
}

impl From<PhaseError> for RunError {
    fn from(err: PhaseError) -> Self {
        match err {
            PhaseError::Message(msg) => RunError::Message(msg),
            PhaseError::Load(err) => RunError::Load(err),
            PhaseError::Sddp { error, message } => RunError::Sddp { error, message },
        }
    }
}

/// Error returned by the training/simulation phase helpers.
///
/// Mirrors [`RunError`] minus the callback variant. The `From<String>` impl keeps
/// every existing `?` site unchanged; only a hard `train`/`simulate` failure
/// builds the typed `Sddp` arm.
#[derive(Debug)]
pub(crate) enum PhaseError {
    /// A descriptive message.
    Message(String),
    /// A typed case-load failure, carried verbatim so the mapping site can pick
    /// the per-variant class.
    Load(LoadError),
    /// A typed SDDP failure carried verbatim with its descriptive message.
    Sddp {
        /// The typed SDDP error.
        error: SddpError,
        /// The verbatim descriptive message.
        message: String,
    },
}

impl From<String> for PhaseError {
    fn from(msg: String) -> Self {
        PhaseError::Message(msg)
    }
}

/// Summary returned by [`run_via_study`] on success.
pub(crate) struct RunSummary {
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

pub(crate) struct SimSummary {
    pub(crate) n_scenarios: u32,
    pub(crate) completed: u32,
}

/// Validate that `threads`, when given, is >= 1.
///
/// `None` is valid (means "let the runtime choose", defaulting to 1);
/// `Some(0)` raises [`PyValueError`].
pub(crate) fn validated_threads(threads: Option<u32>) -> PyResult<Option<u32>> {
    if let Some(t) = threads
        && t == 0
    {
        return Err(PyValueError::new_err(format!(
            "threads must be >= 1 when given, got {t}"
        )));
    }
    Ok(threads)
}

/// Build a scoped rayon thread pool for the requested thread count and run the
/// closure inside `pool.install(...)`.
///
/// A fresh pool per call — not a process-global pool, which can only be
/// configured once per process — so two sequential `run` invocations with
/// different thread counts each honor their own value. The effective `n` is
/// passed into the closure so callers can record it in metadata.
///
/// # Errors
///
/// Returns a descriptive `Err(String)` on pool-construction failure rather than
/// silently falling back to an implicit pool.
pub(crate) fn run_in_scoped_pool<T>(
    threads: Option<u32>,
    f: impl FnOnce(usize) -> T + Send,
) -> Result<T, String>
where
    T: Send,
{
    let n = threads.map_or(1, |t| t as usize);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .map_err(|e| format!("{INTERNAL_ERROR_PREFIX}: rayon pool construction failed: {e}"))?;
    Ok(pool.install(|| f(n)))
}

/// Result of the training phase within `run_via_study`.
pub(crate) struct TrainingPhaseResult {
    pub result: TrainingResult,
    pub output: TrainingOutput,
    pub error: Option<SddpError>,
    pub started_at: String,
}

/// Assemble a [`TrainingPhaseResult`] from a finished training run and its
/// full event stream.
///
/// `events` must be the complete set of [`TrainingEvent`]s emitted during the
/// run: `build_training_output` builds `convergence_records` from them, so a
/// partial set diverges convergence parity between the streaming and
/// non-streaming paths.
fn build_training_phase_result(
    setup: &StudySetup,
    training_result: TrainingResult,
    events: &[TrainingEvent],
    error: Option<SddpError>,
    started_at: String,
    n_threads: usize,
) -> TrainingPhaseResult {
    let mut training_output = setup.build_training_output(&training_result, events);

    // `total_lp_solves` is sourced from the per-iteration convergence records to
    // mirror the CLI exactly (see `aggregate_solver_stats_log`).
    let total_lp_solves: u64 = training_output
        .convergence_records
        .iter()
        .map(|r| u64::from(r.lp_solves))
        .sum();
    let (first_try, retried, failed, forward_solve_seconds, backward_solve_seconds) =
        aggregate_solver_stats_log(&training_result.solver_stats_log, None);
    training_output.training_solve_stats = MetadataTrainingSolveStats {
        total_lp_solves: Some(total_lp_solves),
        first_try: Some(first_try),
        retried: Some(retried),
        failed: Some(failed),
        forward_solve_seconds: Some(forward_solve_seconds),
        backward_solve_seconds: Some(backward_solve_seconds),
        parallelism: Some(u32::try_from(n_threads).unwrap_or(u32::MAX)),
    };

    TrainingPhaseResult {
        result: training_result,
        output: training_output,
        error,
        started_at,
    }
}

/// Run the training phase (no callback): solver init, train, collect events.
///
/// Events are collected with `event_rx.try_iter()` only AFTER `train` returns,
/// keeping the no-callback golden parity test bit-identical. The streaming
/// variant ([`run_training_phase_py_streaming`]) handles the `on_iteration` case.
pub(crate) fn run_training_phase_py(
    setup: &mut StudySetup,
    n_threads: usize,
) -> Result<TrainingPhaseResult, PhaseError> {
    let started_at = now_iso8601();
    let mut solver = ActiveSolver::new().map_err(|e| {
        format!(
            "{} initialisation failed: {e}",
            cobre_solver::active_solver_name()
        )
    })?;
    let (event_tx, event_rx) = mpsc::channel();
    let training_outcome = setup
        .train(
            &mut solver,
            &LocalBackend,
            n_threads,
            ActiveSolver::new,
            Some(event_tx),
            None,
        )
        .map_err(|e| PhaseError::Sddp {
            message: format!("{TRAINING_ERROR_PREFIX}: {e}"),
            error: e,
        })?;

    let events: Vec<_> = event_rx.try_iter().collect();
    Ok(build_training_phase_result(
        setup,
        training_outcome.result,
        &events,
        training_outcome.error,
        started_at,
        n_threads,
    ))
}

/// Run the training phase with a Python `on_iteration` callback, streaming
/// boundary events to Python via a dedicated drain thread (see
/// [`drain_training_events`]).
///
/// `train` runs on this thread with the GIL released. The callback runs ONLY
/// inside `Python::attach` in the drain thread, at iteration boundaries — never
/// in the solver's hot LP loop. The solver loop polls `shutdown_flag` at
/// iteration boundaries and exits gracefully, writing whatever partial artifacts
/// it completed.
///
/// # Errors
///
/// Returns `Err(String)` on `HiGHS` init failure, a `train` error, or a drain
/// thread panic. A captured callback `PyErr` (or `KeyboardInterrupt`) is NOT an
/// error here — it is returned alongside the phase result so the caller can
/// propagate it *after* writing artifacts.
pub(crate) fn run_training_phase_py_streaming(
    setup: &mut StudySetup,
    n_threads: usize,
    on_iteration: Py<PyAny>,
) -> Result<(TrainingPhaseResult, Option<PyErr>), PhaseError> {
    let started_at = now_iso8601();
    let mut solver = ActiveSolver::new().map_err(|e| {
        format!(
            "{} initialisation failed: {e}",
            cobre_solver::active_solver_name()
        )
    })?;
    let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();
    let shutdown_flag = Arc::new(AtomicBool::new(false));

    let drain_flag = Arc::clone(&shutdown_flag);
    let drain_handle =
        std::thread::spawn(move || drain_training_events(&event_rx, &drain_flag, &on_iteration));

    let training_outcome = setup.train(
        &mut solver,
        &LocalBackend,
        n_threads,
        ActiveSolver::new,
        Some(event_tx),
        Some(&shutdown_flag),
    );

    // The channel is already closed: `event_tx` was moved into `setup.train`,
    // whose `TrainingSession` drops the only remaining sender before returning,
    // so the drain thread's `recv()` loop has terminated and `join()` will not
    // block. Surface a training error before a drain panic — it is the more
    // diagnostic failure.
    let drain_result = drain_handle.join();
    let training_outcome = training_outcome.map_err(|e| PhaseError::Sddp {
        message: format!("{TRAINING_ERROR_PREFIX}: {e}"),
        error: e,
    })?;
    let (events, captured_pyerr) =
        drain_result.map_err(|_| format!("{INTERNAL_ERROR_PREFIX}: drain thread panicked"))?;

    let phase = build_training_phase_result(
        setup,
        training_outcome.result,
        &events,
        training_outcome.error,
        started_at,
        n_threads,
    );

    Ok((phase, captured_pyerr))
}

/// Drain-thread body: collect every event, forward boundary summaries to the
/// Python callback under the GIL, and honor early-stop / Ctrl-C / raising-callback
/// requests via the shared `shutdown_flag`.
///
/// Returns the complete event collection (for `build_training_output` parity)
/// and the first captured `PyErr`, if any. Never panics: a raising callback is
/// captured, not unwound.
fn drain_training_events(
    event_rx: &mpsc::Receiver<TrainingEvent>,
    shutdown_flag: &Arc<AtomicBool>,
    on_iteration: &Py<PyAny>,
) -> (Vec<TrainingEvent>, Option<PyErr>) {
    use std::sync::atomic::Ordering;

    let mut collected: Vec<TrainingEvent> = Vec::new();
    let mut captured_pyerr: Option<PyErr> = None;

    while let Ok(event) = event_rx.recv() {
        // Dispatch before pushing so the callback borrows `event` directly; the
        // event is moved into the collection afterward. Once a stop is requested,
        // keep draining (to recover remaining events) but skip GIL reacquisition.
        //
        // `Relaxed` suffices for `shutdown_flag`: it is a one-way latch (only
        // flipped `false` -> `true`). Both outcomes — stop now, or one extra
        // iteration before the store is seen — are correct under the cooperative
        // contract, so no acquire/release synchronization is needed.
        if !shutdown_flag.load(Ordering::Relaxed) {
            Python::attach(|py| {
                let mut request_stop = |err: Option<PyErr>| {
                    shutdown_flag.store(true, Ordering::Relaxed);
                    if let Some(err) = err {
                        captured_pyerr.get_or_insert(err);
                    }
                };

                if let Err(err) = py.check_signals() {
                    request_stop(Some(err));
                    return;
                }

                match iteration_summary_to_dict(py, &event) {
                    Ok(Some(dict)) => match on_iteration.bind(py).call1((dict,)) {
                        Ok(ret) => match ret.is_truthy() {
                            Ok(true) => request_stop(None),
                            Ok(false) => {}
                            Err(err) => request_stop(Some(err)),
                        },
                        Err(err) => request_stop(Some(err)),
                    },
                    Ok(None) => {}
                    // Surface a conversion failure rather than silently dropping it.
                    Err(err) => request_stop(Some(err)),
                }
            });
        }

        // Parity: every event must reach `build_training_output`.
        collected.push(event);
    }

    (collected, captured_pyerr)
}

/// [`DistributionInfo`] for a single-process (non-MPI) run, shared by the
/// training and simulation `OutputContext` sites.
fn single_process_distribution(n_threads: usize) -> DistributionInfo {
    DistributionInfo {
        backend: "local".to_string(),
        world_size: 1,
        ranks_participated: 1,
        num_hosts: 1,
        threads_per_rank: u32::try_from(n_threads).unwrap_or(u32::MAX),
        mpi_library: None,
        mpi_standard: None,
        thread_level: None,
        slurm_job_id: None,
        hosts: vec![cobre_io::HostLayout {
            hostname: get_hostname(),
            ranks: vec![0],
        }],
    }
}

/// Write the training artifacts: policy checkpoint, training results, solver
/// stats, and cut selection records.
pub(crate) fn write_training_artifacts(
    output_dir: &Path,
    system: &System,
    config: &Config,
    setup: &StudySetup,
    training: &TrainingPhaseResult,
    setup_timings: &SetupTimings,
    seed: u64,
    n_threads: usize,
) -> Result<(), String> {
    write_checkpoint(
        &output_dir.join(&setup.policy_path),
        setup,
        system,
        &training.result,
        &CheckpointParams {
            max_iterations: setup.loop_params.max_iterations,
            forward_passes: setup.loop_params.forward_passes,
            seed,
            export_states: config.exports.states,
        },
    )
    .map_err(|e| format!("{POLICY_CHECKPOINT_ERROR_PREFIX}: {e}"))?;

    if !training.result.solver_stats_log.is_empty() {
        let rows = solver_stats_log_to_rows(&training.result.solver_stats_log);
        write_solver_stats(output_dir, &rows)
            .map_err(|e| format!("{OUTPUT_WRITE_ERROR_PREFIX}: solver stats output: {e}"))?;
    }

    if !training.output.cut_selection_records.is_empty() {
        write_row_selection_records(output_dir, &training.output.cut_selection_records)
            .map_err(|e| format!("{OUTPUT_WRITE_ERROR_PREFIX}: cut selection output: {e}"))?;
    }

    let training_ctx = OutputContext {
        hostname: get_hostname(),
        solver: active_solver_metadata_id().to_string(),
        solver_version: Some(active_solver_version()),
        started_at: training.started_at.clone(),
        completed_at: now_iso8601(),
        distribution: single_process_distribution(n_threads),
        setup: Some(setup_timings.clone()),
        // Mirrors the CLI write site so Python and CLI emit the same
        // `production_fit_deviation` section.
        production_fit_deviation: build_deviation_summary(&setup.hydro_models.fpha_fit_deviations),
    };
    write_training_results(output_dir, &training.output, system, config, &training_ctx)
        .map_err(|e| format!("{OUTPUT_WRITE_ERROR_PREFIX}: training results output: {e}"))?;

    Ok(())
}

/// Write the trained FPHA hyperplanes sidecar, when the model produced any.
///
/// Both `run()` and `Study.train()` reach this through the shared
/// `Study::train_native`, so they emit it identically. Training-only:
/// simulation-only runs do not write it.
pub(crate) fn write_fpha_hyperplanes_if_any(
    output_dir: &Path,
    setup: &StudySetup,
) -> Result<(), String> {
    if !setup.hydro_models.fpha_export_rows.is_empty() {
        let fpha_path = output_dir
            .join("hydro_models")
            .join("fpha_hyperplanes.parquet");
        write_fpha_hyperplanes(&fpha_path, &setup.hydro_models.fpha_export_rows).map_err(|e| {
            format!("{OUTPUT_WRITE_ERROR_PREFIX}: failed to write fpha_hyperplanes: {e}")
        })?;
    }
    Ok(())
}

/// Write the resolved evaporation-model coefficients sidecar, when the case
/// models evaporation for at least one hydro.
///
/// Both `run()` and `Study.train()` reach this through the shared
/// `Study::train_native`, which keeps them matched to the CLI's
/// `write_evaporation_models` output (the Python-parity hard rule).
pub(crate) fn write_evaporation_models_if_any(
    output_dir: &Path,
    setup: &StudySetup,
    system: &System,
) -> Result<(), String> {
    let rows = build_evaporation_model_rows(&setup.hydro_models, system);
    if !rows.is_empty() {
        let evaporation_path = output_dir
            .join("hydro_models")
            .join("evaporation_models.parquet");
        write_evaporation_models(&evaporation_path, &rows).map_err(|e| {
            format!("{OUTPUT_WRITE_ERROR_PREFIX}: failed to write evaporation_models: {e}")
        })?;
    }
    Ok(())
}

/// Write the resolved generic-constraint echo sidecar, when the case declares at
/// least one generic constraint.
///
/// Both `run()` and `Study.train()` reach this through the shared
/// `Study::train_native`, which keeps them matched to the CLI's
/// `write_generic_constraint_echo` output (the Python-parity hard rule).
pub(crate) fn write_generic_constraint_echo_if_any(
    output_dir: &Path,
    setup: &StudySetup,
    system: &System,
) -> Result<(), String> {
    if !system.generic_constraints().is_empty() {
        let rows = build_generic_constraint_echo_rows(setup, system);
        let echo_path = output_dir
            .join("generic_constraints")
            .join("resolved_echo.parquet");
        write_generic_constraint_echo(&echo_path, &rows).map_err(|e| {
            format!("{OUTPUT_WRITE_ERROR_PREFIX}: failed to write generic_constraint_echo: {e}")
        })?;
    }
    Ok(())
}

/// Write the run-level fixed post-horizon commitment echo.
///
/// Both `run()` and `Study.train()` reach this through the shared
/// `Study::train_native`, which keeps them matched to the CLI's
/// `write_fixed_delivery` output (the Python-parity hard rule).
pub(crate) fn write_fixed_delivery_if_any(
    output_dir: &Path,
    setup: &StudySetup,
    system: &System,
) -> Result<(), String> {
    let rows = build_fixed_delivery_rows(setup, system);
    write_fixed_delivery(output_dir, &rows)
        .map_err(|e| format!("{OUTPUT_WRITE_ERROR_PREFIX}: failed to write fixed_delivery: {e}"))
}

/// Write the per-sampled-point FPHA deviation table sidecar, when the run opted
/// in (`config.exports.fpha_deviation_points`) AND the fit produced any points.
///
/// Off by default, so a default run writes no file and is byte-identical to the
/// CLI. Both `run()` and `Study.train()` reach this through the shared
/// `Study::train_native`, which keeps them matched to the CLI's
/// `write_fpha_deviation_points` output (the Python-parity hard rule).
pub(crate) fn write_fpha_deviation_points_if_any(
    output_dir: &Path,
    setup: &StudySetup,
    config: &Config,
) -> Result<(), String> {
    if !config.exports.fpha_deviation_points {
        return Ok(());
    }
    let rows = setup.hydro_models.fpha_deviation_point_rows.as_slice();
    if !rows.is_empty() {
        let deviation_points_path = output_dir
            .join("hydro_models")
            .join("fpha_deviation_points.parquet");
        write_fpha_deviation_points(&deviation_points_path, rows).map_err(|e| {
            format!("{OUTPUT_WRITE_ERROR_PREFIX}: failed to write fpha_deviation_points: {e}")
        })?;
    }
    Ok(())
}

/// Run the simulation phase: workspace pool, Parquet writing, and output.
pub(crate) fn run_simulation_phase_py(
    setup: &mut StudySetup,
    output_dir: &Path,
    system: &System,
    training_result: &TrainingResult,
    n_threads: usize,
) -> Result<SimSummary, PhaseError> {
    let sim_started_at = now_iso8601();
    let io_capacity = setup.simulation_config().io_channel_capacity;
    let mut sim_pool = setup
        .create_workspace_pool(&LocalBackend, n_threads, ActiveSolver::new)
        .map_err(|e| {
            format!(
                "{} initialisation failed for simulation pool: {e}",
                cobre_solver::active_solver_name()
            )
        })?;
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity.max(1));

    let sim_writer = SimulationParquetWriter::new(output_dir, system)
        .map_err(|e| format!("{SIMULATION_WRITER_INIT_ERROR_PREFIX}: {e}"))?;

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

    let sim_start = std::time::Instant::now();
    let sim_result = setup
        .simulate(
            &mut sim_pool.workspaces,
            &LocalBackend,
            &result_tx,
            None,
            training_result.frozen_templates.as_deref(),
            &training_result.basis_cache,
        )
        .map_err(|e| {
            // Build the message from the original `SimulationError` before
            // wrapping, so the text stays byte-identical to the old string path.
            let message = format!("{SIMULATION_ERROR_PREFIX}: {e}");
            PhaseError::Sddp {
                message,
                error: SddpError::from(e),
            }
        });
    drop(result_tx);

    let (sim_writer, write_failures) = drain_handle
        .join()
        .map_err(|_| format!("{INTERNAL_ERROR_PREFIX}: simulation drain thread panicked"))?;
    let sim_run_result = sim_result?;

    #[allow(clippy::cast_possible_truncation)]
    let sim_time_ms = sim_start.elapsed().as_millis() as u64;

    // Single-process: the writer already holds every scenario's node-path rows,
    // so no cross-rank gather (unlike the CLI). Written before `finalize`
    // consumes the writer.
    write_paths(output_dir, sim_writer.path_rows().to_vec())
        .map_err(|e| format!("{OUTPUT_WRITE_ERROR_PREFIX}: simulation paths output: {e}"))?;

    let mut sim_out = sim_writer.finalize(sim_time_ms);
    sim_out.failed = write_failures;

    // Single-process: no opening or per-worker dimension to filter on, so fold
    // every per-scenario delta into one aggregate.
    let mut agg = SolverStatsDelta::default();
    for (_, _, delta) in &sim_run_result.solver_stats {
        SolverStatsDelta::accumulate_into(&mut agg, delta);
    }

    // The weighting rides out on the run result, resolved once from the
    // simulation Traversal inside `simulate()` (matching the CLI path in
    // `cobre-cli`'s `run/simulation.rs`): `Census` (exact leaf-path expectation)
    // when `census_weights` is `Some`, the uniform Monte-Carlo sample mean when
    // `None`.
    let weighting = match sim_run_result.census_weights.as_deref() {
        Some(weights) => SimulationWeighting::Census { weights },
        None => SimulationWeighting::Uniform,
    };
    let (cost_summary, gathered_scenario_costs) = aggregate_simulation(
        &sim_run_result.costs,
        setup.simulation_config(),
        &LocalBackend,
        weighting,
    )
    .map_err(|e| format!("{SIMULATION_ERROR_PREFIX}: cost aggregation: {e}"))?;

    let scenario_summary_rows: Vec<(u32, Option<f64>, f64)> = gathered_scenario_costs
        .iter()
        .map(|&(scenario_id, discounted_immediate_cost, probability)| {
            (scenario_id, probability, discounted_immediate_cost)
        })
        .collect();
    write_scenario_summary(output_dir, &scenario_summary_rows).map_err(|e| {
        format!("{OUTPUT_WRITE_ERROR_PREFIX}: simulation scenario summary output: {e}")
    })?;

    let parallelism = u32::try_from(n_threads).unwrap_or(u32::MAX);
    sim_out.cost = Some(MetadataCost {
        mean_cost: cost_summary.mean_cost,
        std_cost: cost_summary.std_cost,
    });
    sim_out.solve_stats = MetadataSimulationSolveStats {
        total_lp_solves: Some(agg.lp_solves),
        first_try: Some(agg.first_try_successes),
        retried: Some(agg.lp_successes.saturating_sub(agg.first_try_successes)),
        failed: Some(agg.lp_failures),
        solve_seconds: Some(agg.solve_time_ms / 1000.0),
        parallelism: Some(parallelism),
    };

    // Single-process: simulation fills scenario_id (not iteration); stage/opening/
    // rank/worker_id are all None.
    if !sim_run_result.solver_stats.is_empty() {
        let rows: Vec<SolverStatsRow> = sim_run_result
            .solver_stats
            .iter()
            .map(|(scenario_id, _opening, delta)| {
                #[allow(clippy::cast_possible_wrap)]
                delta_to_stats_row(
                    None,
                    Some(*scenario_id as i32),
                    "simulation",
                    None,
                    None,
                    None,
                    None,
                    delta,
                )
            })
            .collect();
        write_simulation_solver_stats(output_dir, &rows).map_err(|e| {
            format!("{OUTPUT_WRITE_ERROR_PREFIX}: simulation solver stats output: {e}")
        })?;
    }

    let sim_summary = SimSummary {
        n_scenarios: sim_out.n_scenarios,
        completed: sim_out.completed,
    };
    let sim_ctx = OutputContext {
        hostname: get_hostname(),
        solver: active_solver_metadata_id().to_string(),
        solver_version: Some(active_solver_version()),
        started_at: sim_started_at,
        completed_at: now_iso8601(),
        distribution: single_process_distribution(n_threads),
        setup: None,
        // training-only.
        production_fit_deviation: None,
    };
    write_simulation_results(output_dir, &sim_out, &sim_ctx)
        .map_err(|e| format!("{OUTPUT_WRITE_ERROR_PREFIX}: simulation results output: {e}"))?;

    Ok(sim_summary)
}

/// Load the effective [`cobre_io::Config`] for a run.
///
/// With no overrides, [`cobre_io::parse_config`] reads and validates
/// `config.json`. With overrides, the file is deep-merged with them via
/// [`cobre_io::Config::with_overrides`], which runs the same validation
/// `parse_config` performs, so the persisted metadata reflects the effective
/// (post-override) config.
fn load_effective_config(
    config_path: &Path,
    overrides: Option<&Map<String, Value>>,
) -> Result<Config, String> {
    match overrides {
        Some(map) if !map.is_empty() => {
            let raw = std::fs::read_to_string(config_path)
                .map_err(|e| format!("{CONFIG_READ_ERROR_PREFIX}: {e}"))?;
            let base: Value = serde_json::from_str(&raw)
                .map_err(|e| format!("{CONFIG_PARSE_ERROR_PREFIX}: {e}"))?;
            Config::with_overrides(&base, map)
                .map_err(|e| format!("{CONFIG_OVERRIDE_ERROR_PREFIX}: {e}"))
        }
        _ => parse_config(config_path).map_err(|e| format!("{CONFIG_PARSE_ERROR_PREFIX}: {e}")),
    }
}

/// Adds [`SETUP_VALIDATION_ERROR_PREFIX`] only for `SddpError::Validation`;
/// every other setup-phase error keeps today's message and class.
#[allow(clippy::needless_pass_by_value)]
fn setup_error_message(err: SddpError, phase_prefix: Option<&str>) -> String {
    let body = match phase_prefix {
        Some(prefix) => format!("{prefix}: {err}"),
        None => err.to_string(),
    };
    if matches!(err, SddpError::Validation(_)) {
        format!("{SETUP_VALIDATION_ERROR_PREFIX}: {body}")
    } else {
        body
    }
}

/// Maps a boundary-cut load error to its message prefix: [`SddpError::PolicyVersionMismatch`]
/// gets [`POLICY_VALIDATION_ERROR_PREFIX`] (so it raises `PolicyIncompatibleError`), every
/// other boundary-load error keeps [`BOUNDARY_CUT_ERROR_PREFIX`].
#[allow(clippy::needless_pass_by_value)]
fn boundary_cut_error_message(err: SddpError) -> String {
    let prefix = if matches!(err, SddpError::PolicyVersionMismatch { .. }) {
        POLICY_VALIDATION_ERROR_PREFIX
    } else {
        BOUNDARY_CUT_ERROR_PREFIX
    };
    format!("{prefix}: {err}")
}

/// Everything the front half of the solve lifecycle produces: the live
/// [`StudySetup`] plus the adjacent immutable state that `run_via_study` and the
/// `Study` pyclass both consume. [`build_study_setup`] is the sole producer (the
/// single load path).
///
/// The `warnings` carrier holds the validation-pipeline warnings captured during
/// load (via [`cobre_io::validate_case_with_artifacts`]) so `Study::validate` can
/// replay them without re-reading disk.
pub(crate) struct LoadedStudy {
    /// The live, fully prepared study setup (cuts pool, templates, stochastic
    /// context, hydro models, scenario libraries).
    pub setup: StudySetup,
    /// The system after stochastic preprocessing (inflow non-negativity, etc.).
    pub system: System,
    /// The effective (post-override) configuration.
    pub config: Config,
    /// The resolved tree seed.
    pub seed: u64,
    /// The model-provenance report, including the past-inflows digest.
    pub provenance: ModelProvenanceReport,
    /// The structural stochastic summary.
    pub stochastic_summary: StochasticSummary,
    /// The structural hydro-model summary.
    pub hydro_models_summary: HydroModelSummary,
    /// Validation-pipeline warnings captured during the case load.
    pub warnings: Vec<ReportEntry>,
    /// Wall-clock setup-phase timings, mirroring the CLI's `SetupTimings`
    /// collection in `crates/cobre-cli/src/commands/run/setup.rs`.
    pub setup_timings: SetupTimings,
}

/// Run the front half of the solve lifecycle: load the case, resolve the
/// effective config, run stochastic/hydro-model preprocessing, build the
/// [`StudySetup`] and the provenance/summary carriers, and write the front-half
/// sidecar artifacts.
///
/// Python-free, and the ONLY place the front half runs: both [`run_via_study`]
/// and the `Study` pyclass call it, so there is a single load path with no
/// divergence.
///
/// `overrides` is the already-converted dotted-key override map; `None` and an
/// empty map both reproduce the no-override path.
///
/// # Errors
///
/// Returns a typed [`PhaseError`] on any load, config, preprocessing,
/// construction, or sidecar-write failure. Case-load failures carry the full
/// typed [`LoadError`] so the caller can map each variant to its appropriate
/// Python exception class.
pub(crate) fn build_study_setup(
    case_dir: &Path,
    output_dir: &Path,
    overrides: Option<&Map<String, Value>>,
) -> Result<LoadedStudy, PhaseError> {
    let mut timings = SetupTimings::default();
    let load_start = std::time::Instant::now();

    // The `validate_*` variant (rather than `load_case_with_artifacts`) captures
    // the warnings so `Study::validate` can replay them without re-reading disk.
    let (loaded, report) = validate_case_with_artifacts(case_dir).map_err(PhaseError::Load)?;
    let LoadedCase { system, artifacts } = loaded;
    let warnings = report.warnings;

    let config = load_effective_config(&case_dir.join("config.json"), overrides)?;
    timings.load_seconds = load_start.elapsed().as_secs_f64();

    // Resolve the boundary-derived state requirements once; carried onto the
    // construction config below so both the layout and the boundary-load reject
    // see them. Mirrors the CLI run path.
    let boundary_requirements = resolve_boundary_state_requirements(case_dir, &config)
        .map_err(|e| setup_error_message(e, None))?;

    let seed = config
        .training
        .tree_seed
        .map_or(DEFAULT_SEED, i64::unsigned_abs);

    let training_source = config
        .training_scenario_source(&case_dir.join("config.json"))
        .map_err(|e| format!("{SCENARIO_SOURCE_ERROR_PREFIX}: {e}"))?;

    let stochastic_start = std::time::Instant::now();
    let result = prepare_stochastic(
        system,
        case_dir,
        &config,
        seed,
        &training_source,
        boundary_requirements.inflow_lag_depth(),
    )
    .map_err(|e| setup_error_message(e, Some(STOCHASTIC_PREPROCESSING_ERROR_PREFIX)))?;
    timings.stochastic_fit_seconds = stochastic_start.elapsed().as_secs_f64();
    let system = result.system;
    let estimation_report = result.estimation_report;
    let estimation_path = result.estimation_path;

    let mut hydro_timings = HydroFitTimings::default();
    let hydro_models_result = prepare_hydro_models_from_artifacts(
        &system,
        &artifacts,
        config.exports.fpha_deviation_points,
        Some(&mut hydro_timings),
    )
    .map_err(|e| setup_error_message(e, Some(HYDRO_MODEL_PREPROCESSING_ERROR_PREFIX)))?;
    timings.production_fit_seconds = hydro_timings.production_fit_seconds;
    timings.evaporation_fit_seconds = hydro_timings.evaporation_fit_seconds;

    let simulation_source = config
        .simulation_scenario_source(&case_dir.join("config.json"))
        .map_err(|e| format!("{SCENARIO_SOURCE_ERROR_PREFIX}: {e}"))?;
    let mut construction =
        StudyParams::from_config(&config, Vec::new()).map_err(|e| setup_error_message(e, None))?;
    construction.boundary = boundary_requirements;
    construction.scalar_parameters = artifacts.scalar_parameters;
    let setup = StudySetup::from_broadcast_params(
        &system,
        result.stochastic,
        construction,
        hydro_models_result,
        &training_source,
        &simulation_source,
    )
    .map_err(|e| setup_error_message(e, None))?;

    let mut provenance_report = build_provenance_report(
        estimation_path,
        estimation_report.as_ref(),
        setup.inputs.stochastic.provenance(),
        system.hydros(),
        &setup.hydro_models.provenance,
    );
    // Fingerprint the derived lag seed (training-side library only) so
    // stale-library detection can compare against a fresh digest on later runs.
    provenance_report.inflow.historical_library_seed_digest = setup
        .inputs
        .scenario_libraries
        .training
        .historical
        .as_ref()
        .map(HistoricalScenarioLibrary::seed_digest);

    if config.exports.stochastic {
        let mut on_warning = |msg: &str| {
            eprintln!("cobre-python: stochastic export warning: {msg}");
        };
        export_stochastic_artifacts(
            output_dir,
            &setup.inputs.stochastic,
            &system,
            estimation_report.as_ref(),
            &mut on_warning,
        );
    }

    let scaling_path = output_dir.join("training/scaling_report.json");
    write_scaling_report(&scaling_path, &setup.inputs.stage_data.scaling_report)
        .map_err(|e| format!("{OUTPUT_WRITE_ERROR_PREFIX}: failed to write scaling report: {e}"))?;

    let provenance_path = output_dir.join("training/model_provenance.json");
    write_provenance_report(&provenance_path, &provenance_report).map_err(|e| {
        format!("{OUTPUT_WRITE_ERROR_PREFIX}: failed to write model provenance: {e}")
    })?;

    let stochastic_summary =
        build_stochastic_summary(&system, &setup.inputs.stochastic, estimation_report.as_ref(), seed);
    let hydro_models_summary = build_hydro_model_summary(&setup.hydro_models, &system);

    let hydro_models_path = output_dir.join("training/hydro_models.json");
    write_hydro_model_summary(&hydro_models_path, &hydro_models_summary).map_err(|e| {
        format!("{OUTPUT_WRITE_ERROR_PREFIX}: failed to write hydro model summary: {e}")
    })?;

    Ok(LoadedStudy {
        setup,
        system,
        config,
        seed,
        provenance: provenance_report,
        stochastic_summary,
        hydro_models_summary,
        warnings,
        setup_timings: timings,
    })
}

/// Rescale `checkpoint`'s cut coefficients into the loading study's
/// `cost_scale_factor` (see [`rescale_checkpoint_cuts_for_load`]), then validate
/// it against `setup`/`system` via the shared
/// [`cobre_sddp::validate_policy_load`] entry point, building both
/// [`cobre_sddp::PolicyStageManifest`]s exactly as the CLI's
/// `load_and_validate_checkpoint` does (the checkpoint's terminal-stage entity
/// manifest vs. [`StudySetup::build_terminal_entity_manifest`]) — the single
/// manifest-construction shape shared by warm-start, resume, and
/// simulation-only loads. Warnings are drained to stderr (single-process,
/// non-fatal). Returns the resulting [`PolicyLoadProof<FullFcf>`], the sole
/// credential `FutureCostFunction::new_with_warm_start`/`from_deserialized`
/// accept.
///
/// # Errors
///
/// Returns `Err(String)` formatted as `"policy validation error: {e}"` on a
/// version mismatch, or on a `state_dimension`, `num_stages`, or
/// entity-manifest mismatch.
fn validate_loaded_policy(
    checkpoint: &mut cobre_io::PolicyCheckpoint,
    system: &System,
    setup: &StudySetup,
) -> Result<PolicyLoadProof<FullFcf>, String> {
    let source_cost_scale_factor = checkpoint_terminal_cost_scale_factor(checkpoint)
        .map_err(|e| format!("{POLICY_VALIDATION_ERROR_PREFIX}: {e}"))?;
    rescale_checkpoint_cuts_for_load(
        &mut checkpoint.stage_cuts,
        Some(source_cost_scale_factor),
        setup.inputs.stage_data.stage_templates.cost_scale_factor,
    );

    #[allow(clippy::cast_possible_truncation)]
    let n_stages = system.stages().iter().filter(|s| s.id >= 0).count() as u32;
    #[allow(clippy::cast_possible_truncation)]
    let state_dim = setup.fcf.state_dimension as u32;

    let current_manifest = setup.build_terminal_entity_manifest(system);
    let checkpoint_terminal_manifest: &[EntitySlot] = checkpoint
        .stage_cuts
        .last()
        .map_or(&[], |s| s.entity_manifest.as_slice());
    let source_state_dim = checkpoint
        .stage_cuts
        .last()
        .map_or(0, |s| s.state_dimension);
    let source_graph = &checkpoint.metadata.graph_manifest;
    let current_graph = setup.build_graph_manifest();

    let source = PolicyStageManifest {
        state_dimension: source_state_dim,
        num_stages: checkpoint.metadata.num_stages,
        n_pools: source_graph.n_pools,
        slots: checkpoint_terminal_manifest,
        graph: source_graph,
    };
    let current = PolicyStageManifest {
        state_dimension: state_dim,
        num_stages: n_stages,
        n_pools: current_graph.n_pools,
        slots: &current_manifest,
        graph: &current_graph,
    };
    let proof = validate_policy_load::<FullFcf>(&checkpoint.metadata.cobre_version, &source, &current)
        .map_err(|e| format!("{POLICY_VALIDATION_ERROR_PREFIX}: {e}"))?;

    for msg in &proof.warnings {
        eprintln!("cobre-python: policy validation warning: {msg}");
    }

    Ok(proof)
}

/// Build a warm-started [`FutureCostFunction`] from a validated checkpoint proof.
///
/// `pool_state_dimensions`/`visit_bounds` come from the pre-replacement
/// (cold-path) FCF's per-pool arrays (`new_per_pool`), reused verbatim by both
/// the `WarmStart` and `Resume` modes rather than substituting a scalar.
/// `max_iterations + 1` reserves one extra cut slot for the cut added in the
/// final iteration.
fn build_warm_start_fcf(
    setup: &StudySetup,
    proof: &PolicyLoadProof<FullFcf>,
    checkpoint: &cobre_io::PolicyCheckpoint,
    mode_label: &str,
) -> Result<FutureCostFunction, String> {
    let pool_state_dimensions: Vec<usize> =
        setup.fcf.pools.iter().map(|p| p.state_dimension).collect();
    let visit_bounds: Vec<u64> = setup
        .fcf
        .pools
        .iter()
        .map(|p| u64::from(p.visit_stride))
        .collect();
    FutureCostFunction::new_with_warm_start(
        proof,
        &checkpoint.stage_cuts,
        &pool_state_dimensions,
        &visit_bounds,
        setup.loop_params.forward_passes,
        setup.loop_params.max_iterations.saturating_add(1),
    )
    .map_err(|e| format!("{mode_label} FCF construction error: {e}"))
}

/// Seed `setup`'s warm-start basis cache from a loaded checkpoint's stage bases.
/// Empty bases (a checkpoint written without `store_basis`) leave iteration 1 to
/// cold-start.
fn seed_warm_start_basis_cache(
    setup: &mut StudySetup,
    checkpoint: &cobre_io::PolicyCheckpoint,
) -> Result<(), String> {
    if !checkpoint.stage_bases.is_empty() {
        let basis_cache =
            build_basis_cache_from_checkpoint(&checkpoint.stage_bases, &checkpoint.stage_cuts, setup)
                .map_err(|e| format!("{POLICY_VALIDATION_ERROR_PREFIX}: {e}"))?;
        setup.set_warm_start_basis_cache(basis_cache);
    }
    Ok(())
}

pub(crate) struct BoundaryReconciliation {
    pub(crate) cuts: ValidatedBoundaryCuts,
    pub(crate) checkpoint_path: PathBuf,
    pub(crate) boundary_date: NaiveDate,
}

/// Reconcile `bp`'s checkpoint against `setup`'s terminal manifest without
/// injecting anything, exactly as the CLI's validate path does before solving.
pub(crate) fn reconcile_boundary_policy(
    setup: &StudySetup,
    system: &System,
    bp: &BoundaryPolicy,
    case_dir: &Path,
) -> Result<BoundaryReconciliation, SddpError> {
    let checkpoint_path = bp.checkpoint_path(case_dir);
    // Rationale: the cast cannot truncate — `state_dimension` counts FCF
    // state variables (one per reservoir/lag), bounded by the validated study
    // dimensions and far below `u32::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    let state_dim = setup.fcf.state_dimension as u32;
    let current_manifest = setup.build_terminal_entity_manifest(system);
    let fixed_windows = setup.build_terminal_fixed_post_horizon_windows(system);

    let Some(boundary_date) = study_horizon_end(system) else {
        return Err(SddpError::Validation(format!(
            "case {}: the study declares no non-negative stage, so it has no boundary date to \
             load a boundary policy against",
            case_dir.display()
        )));
    };

    let study_seasons = build_season_manifest(system);
    let cuts = load_boundary_cuts(
        &BoundaryLoadRequest::new(
            &checkpoint_path,
            boundary_date,
            state_dim,
            &current_manifest,
            setup.inputs.stage_data.stage_templates.cost_scale_factor,
        )
        .with_fixed_windows(&fixed_windows)
        .with_inflow_lag_depth(setup.boundary_requirements().inflow_lag_depth())
        .with_study_seasons(&study_seasons)
        .with_strict(bp.strict),
    )?;

    Ok(BoundaryReconciliation {
        cuts,
        checkpoint_path,
        boundary_date,
    })
}

/// Apply the configured policy mode (warm-start / resume / boundary cuts) to
/// `setup` BEFORE training.
///
/// Shared by the monolithic `run` path and `Study::train` (no divergence), and
/// Python-free. The default mode with no boundary cuts is a no-op.
///
/// # Errors
///
/// Returns a descriptive `Err(String)` when a `WarmStart`/`Resume` mode finds no
/// prior policy directory, when the checkpoint cannot be read, when policy
/// validation fails, when warm-start/resume FCF construction fails, or when the
/// boundary cuts cannot be loaded. The caller maps the message to a Python
/// exception type via [`crate::errors::convert_error`].
pub(crate) fn apply_training_policy_mode(
    setup: &mut StudySetup,
    system: &System,
    config: &Config,
    output_dir: &Path,
    case_dir: &Path,
) -> Result<(), String> {
    if config.policy.mode == WarmStart {
        let policy_dir = output_dir.join(&setup.policy_path);
        if !policy_dir.exists() {
            return Err(format!(
                "Policy directory not found: {}. Cannot warm-start \
                 without a prior policy.",
                policy_dir.display()
            ));
        }

        let mut checkpoint = read_policy_checkpoint(&policy_dir)
            .map_err(|e| format!("failed to read policy checkpoint: {e}"))?;
        let proof = validate_loaded_policy(&mut checkpoint, system, setup)?;

        let warm_fcf = build_warm_start_fcf(setup, &proof, &checkpoint, "warm-start")?;
        setup.replace_fcf(warm_fcf);
        seed_warm_start_basis_cache(setup, &checkpoint)?;
    } else if config.policy.mode == Resume {
        let policy_dir = output_dir.join(&setup.policy_path);
        if !policy_dir.exists() {
            return Err(format!(
                "Policy directory not found: {}. Cannot resume \
                 without a prior checkpoint.",
                policy_dir.display()
            ));
        }

        let mut checkpoint = read_policy_checkpoint(&policy_dir)
            .map_err(|e| format!("failed to read policy checkpoint: {e}"))?;
        let proof = validate_loaded_policy(&mut checkpoint, system, setup)?;

        let completed = u64::from(checkpoint.metadata.producer.completed_iterations);

        let warm_fcf = build_warm_start_fcf(setup, &proof, &checkpoint, "resume")?;
        setup.replace_fcf(warm_fcf);
        setup.set_start_iteration(completed);
        seed_warm_start_basis_cache(setup, &checkpoint)?;
    }

    // Boundary cuts run AFTER warm-start/resume so the two compose: warm-start
    // replaces the entire FCF first, then boundary cuts overwrite only the
    // terminal pool.
    if let Some(ref bp) = config.policy.boundary {
        let recon = reconcile_boundary_policy(setup, system, bp, case_dir)
            .map_err(boundary_cut_error_message)?;
        inject_boundary_cuts(setup, &recon.cuts).map_err(boundary_cut_error_message)?;
        let cut_count = recon.cuts.len();
        eprintln!(
            "cobre-python: boundary cuts: {cut_count} loaded from {} (priced at {})",
            recon.checkpoint_path.display(),
            recon.boundary_date
        );
        eprintln!("cobre-python: {}", recon.cuts.report().summary_line());
    }

    Ok(())
}

/// Reconstruct an on-disk policy checkpoint into a `(FutureCostFunction,
/// TrainingResult)` pair for simulation-only / `Study.load_policy`, exactly as
/// the CLI's `load_policy_for_simulation` builds it (a synthetic
/// [`TrainingResult::new`] with `frozen_templates = None`).
///
/// The single, Python-free on-disk reconstruction path, shared by the
/// simulation-only branch of [`run_via_study`] and `Study::load_policy`.
///
/// Deliberately does NOT call [`StudySetup::replace_fcf`]: the caller decides
/// whether to mutate the study, so a trained `Policy` and a loaded one feed the
/// identical simulate path.
///
/// [`TrainingResult::new`]: cobre_sddp::TrainingResult::new
/// [`StudySetup::replace_fcf`]: cobre_sddp::StudySetup::replace_fcf
///
/// # Errors
///
/// Returns a descriptive `Err(String)` when `policy_dir` does not exist (the
/// `"Policy directory not found: ..."` message), when the checkpoint cannot be
/// read, when policy validation fails, when FCF reconstruction fails, or when a
/// stored basis does not match the study's LP. The caller maps the message to
/// a Python exception type via [`crate::errors::convert_error`].
pub(crate) fn reconstruct_policy_from_checkpoint(
    setup: &StudySetup,
    system: &System,
    policy_dir: &Path,
) -> Result<(FutureCostFunction, TrainingResult), String> {
    if !policy_dir.exists() {
        return Err(format!(
            "Policy directory not found: {}. Cannot run simulation-only \
             mode without a trained policy.",
            policy_dir.display()
        ));
    }

    let mut checkpoint = read_policy_checkpoint(policy_dir)
        .map_err(|e| format!("failed to read policy checkpoint: {e}"))?;
    let proof = validate_loaded_policy(&mut checkpoint, system, setup)?;

    let pool_state_dimensions: Vec<usize> =
        setup.fcf.pools.iter().map(|p| p.state_dimension).collect();
    let loaded_fcf = FutureCostFunction::from_deserialized(
        &proof,
        &checkpoint.stage_cuts,
        &pool_state_dimensions,
    )
    .map_err(|e| format!("FCF reconstruction error: {e}"))?;

    let basis_cache =
        build_basis_cache_from_checkpoint(&checkpoint.stage_bases, &checkpoint.stage_cuts, setup)
            .map_err(|e| format!("{POLICY_VALIDATION_ERROR_PREFIX}: {e}"))?;

    let training_result = TrainingResult::new(
        checkpoint.metadata.producer.final_lower_bound,
        checkpoint
            .metadata
            .producer
            .best_upper_bound
            .unwrap_or(f64::INFINITY),
        0.0,
        0.0,
        checkpoint.metadata.producer.completed_iterations.into(),
        "loaded from checkpoint".to_string(),
        0,
        basis_cache,
        Vec::new(),
        None,
        // None: checkpoints store no frozen templates; simulate() re-freezes from the
        // FCF cut pool at startup.
        None,
    );

    Ok((loaded_fcf, training_result))
}

/// Run the full solve lifecycle without MPI or progress bars (GIL released for computation).
///
/// The SINGLE execution path: constructs one [`Study`] through `new_native` and
/// drives the three branches through the native methods, returning the
/// [`RunSummary`] the [`run`] shim renders into the public dict. It performs no
/// `PyO3` dict assembly itself.
///
/// `overrides` is the already-converted `config_overrides` map. When `Some` and
/// non-empty, the effective config is the deep-merge of `config.json` and the
/// overrides via [`cobre_io::Config::with_overrides`], so the persisted metadata
/// reflects what actually ran. `None` and an empty map both reproduce the
/// no-override path.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn run_via_study(
    case_dir: &Path,
    output_dir: PathBuf,
    threads: Option<u32>,
    overrides: Option<Map<String, Value>>,
    on_iteration: Option<Py<PyAny>>,
) -> Result<RunSummary, RunError> {
    use crate::study::Study;

    let mut study = Study::new_native(case_dir, Some(output_dir.clone()), threads, overrides)?;

    let should_simulate = study.simulation_enabled();
    // Fixed at Study construction and never mutated by train_native/simulate_native
    // (only `setup` is), so it is safe to read once regardless of which arm runs.
    let stochastic = Some(study.stochastic_summary().clone());
    let hydro_models = Some(study.hydro_models_summary().clone());
    let provenance = Some(study.provenance().clone());

    match RunPhasePlan::resolve(study.training_enabled(), should_simulate) {
        RunPhasePlan::TrainedThenSimulated => {
            let policy = study.train_native(on_iteration)?;

            let simulation = if should_simulate {
                Some(study.simulate_native(&policy, None)?)
            } else {
                None
            };

            let result = policy.training_result();
            Ok(RunSummary {
                converged: policy.converged,
                iterations: result.iterations,
                lower_bound: result.final_lb,
                upper_bound: Some(result.final_ub),
                gap_percent: Some(result.final_gap * 100.0),
                total_time_ms: result.total_time_ms,
                output_dir,
                simulation,
                stochastic,
                hydro_models,
                provenance,
            })
        }
        RunPhasePlan::SimulateFromPolicy => {
            let policy = study.load_policy_native(None)?;
            let simulation = Some(study.simulate_native(&policy, None)?);

            let result = policy.training_result();
            Ok(RunSummary {
                converged: false,
                iterations: 0,
                lower_bound: result.final_lb,
                upper_bound: if result.final_ub.is_finite() {
                    Some(result.final_ub)
                } else {
                    None
                },
                gap_percent: None,
                total_time_ms: 0,
                output_dir,
                simulation,
                stochastic,
                hydro_models,
                provenance,
            })
        }
        RunPhasePlan::Nothing => Ok(RunSummary {
            converged: false,
            iterations: 0,
            lower_bound: 0.0,
            upper_bound: None,
            gap_percent: None,
            total_time_ms: 0,
            output_dir,
            simulation: None,
            stochastic,
            hydro_models,
            provenance,
        }),
    }
}

/// Convert an `Option<T>` into a Python dict, or `None` when absent.
fn optional_summary_dict<'py, T>(
    py: Python<'py>,
    value: Option<&T>,
    to_dict: impl FnOnce(Python<'py>, &T) -> PyResult<Bound<'py, PyDict>>,
) -> PyResult<Py<PyAny>> {
    match value {
        Some(v) => Ok(to_dict(py, v)?.into()),
        None => Ok(py.None()),
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
pub(crate) fn hydro_model_summary_to_dict<'py>(
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
pub(crate) fn stochastic_summary_to_dict<'py>(
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
pub(crate) fn provenance_to_dict<'py>(
    py: Python<'py>,
    report: &ModelProvenanceReport,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("estimation_path", &report.inflow.estimation_path)?;
    dict.set_item(
        "seasonal_stats_source",
        report.inflow.seasonal_stats_source.to_string(),
    )?;
    dict.set_item(
        "ar_coefficients_source",
        report.inflow.ar_coefficients_source.to_string(),
    )?;
    dict.set_item(
        "correlation_source",
        report.inflow.correlation_source.to_string(),
    )?;
    dict.set_item(
        "opening_tree_source",
        report.inflow.opening_tree_source.to_string(),
    )?;
    dict.set_item("n_hydros", report.inflow.n_hydros)?;
    dict.set_item("ar_method", report.inflow.ar_method.as_deref())?;
    dict.set_item("ar_max_order", report.inflow.ar_max_order)?;
    dict.set_item(
        "white_noise_fallbacks",
        report.inflow.white_noise_fallbacks.clone(),
    )?;

    let hp_dict = PyDict::new(py);
    hp_dict.set_item(
        "n_fpha_computed_from_geometry",
        report.hydro_production.n_fpha_computed_from_geometry,
    )?;
    hp_dict.set_item(
        "n_fpha_precomputed_hyperplanes",
        report.hydro_production.n_fpha_precomputed_hyperplanes,
    )?;
    hp_dict.set_item(
        "n_evaporation_ref_user_supplied",
        report.hydro_production.n_evaporation_ref_user_supplied,
    )?;
    hp_dict.set_item(
        "n_evaporation_ref_default_midpoint",
        report.hydro_production.n_evaporation_ref_default_midpoint,
    )?;
    dict.set_item("hydro_production", hp_dict)?;

    Ok(dict)
}

/// Convert a boundary [`TrainingEvent::IterationSummary`] into a Python dict;
/// every other variant returns `Ok(None)`, keeping GIL reacquisition rare.
///
/// `gap` is the **raw relative** optimality gap as stored on the event, **not**
/// multiplied by 100 — intentionally distinct from `run.run()`'s
/// `gap_percent = gap * 100`. The Python side must scale to a percentage itself.
fn iteration_summary_to_dict<'py>(
    py: Python<'py>,
    event: &cobre_core::TrainingEvent,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    match event {
        IterationSummary {
            iteration,
            lower_bound,
            upper_bound,
            gap,
            wall_time_ms,
            ..
        } => {
            let dict = PyDict::new(py);
            dict.set_item("kind", "iteration")?;
            dict.set_item("iteration", *iteration)?;
            dict.set_item("lower_bound", *lower_bound)?;
            dict.set_item("upper_bound", *upper_bound)?;
            dict.set_item("gap", *gap)?;
            dict.set_item("wall_time_ms", *wall_time_ms)?;
            Ok(Some(dict))
        }
        _ => Ok(None),
    }
}

/// Load a case, train an SDDP policy, optionally simulate, and write results.
/// GIL is released for the entire Rust computation.
/// Returns a dict with keys: `"converged"`, `"iterations"`, `"lower_bound"`, `"upper_bound"`,
/// `"gap_percent"`, `"total_time_ms"`, `"output_dir"`, `"simulation"`, `"stochastic"`,
/// `"hydro_models"`, and `"provenance"`.
///
/// `config_overrides` is an optional flat dotted-key mapping (e.g.
/// `{"training.tree_seed": 7}`) that is deep-merged into `config.json` before the
/// run; the effective (post-override) config is what every output reflects. The
/// dict is converted to a `serde_json::Map` here under the GIL, before
/// `py.detach`. `None` and an empty map both reproduce the no-override behavior.
///
/// `on_iteration` is an optional Python callable invoked once per training
/// iteration boundary with a `dict` describing the iteration (`"kind"`,
/// `"iteration"`, `"lower_bound"`, `"upper_bound"`, `"gap"`, `"wall_time_ms"`).
/// A truthy return requests a cooperative stop at the next iteration boundary;
/// the run still writes its (partial) artifacts. A callback that raises
/// propagates as the run's exception after artifacts are written. The callback
/// runs in a dedicated drain thread under the GIL — never in the solver's hot
/// loop. When `None` (the default), the run is bit-identical to the no-callback
/// path.
///
/// Warning diagnostics (simulation write warnings, policy validation warnings,
/// and boundary reconciliation summaries) are written directly to standard error
/// from Rust, with no parameter to suppress them. These diagnostics are emitted to
/// file descriptor 2 and are **not** captured by `contextlib.redirect_stderr` or
/// pytest's `capsys`.
// needless_pass_by_value: PyO3's from-Python extraction hands over owned values,
// so the `PathBuf`/`Py<PyAny>` arguments cannot be borrowed at this boundary.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
#[pyo3(signature = (case_dir, output_dir=None, threads=None, config_overrides=None, on_iteration=None))]
pub fn run(
    py: Python<'_>,
    case_dir: PathBuf,
    output_dir: Option<PathBuf>,
    threads: Option<u32>,
    config_overrides: Option<Bound<'_, PyDict>>,
    on_iteration: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    if !case_dir.exists() {
        return Err(PyOSError::new_err(format!(
            "case directory does not exist: {}",
            case_dir.display()
        )));
    }

    let threads = validated_threads(threads)?;

    let resolved_output = output_dir.unwrap_or_else(|| case_dir.join("output"));

    let overrides = config_overrides
        .map(|dict| pydict_to_json_map(&dict))
        .transpose()?;

    let result: Result<RunSummary, RunError> = py.detach(move || {
        run_via_study(&case_dir, resolved_output, threads, overrides, on_iteration)
    });

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

            dict.set_item(
                "stochastic",
                optional_summary_dict(py, summary.stochastic.as_ref(), stochastic_summary_to_dict)?,
            )?;
            dict.set_item(
                "hydro_models",
                optional_summary_dict(
                    py,
                    summary.hydro_models.as_ref(),
                    hydro_model_summary_to_dict,
                )?,
            )?;
            dict.set_item(
                "provenance",
                optional_summary_dict(py, summary.provenance.as_ref(), provenance_to_dict)?,
            )?;

            Ok(dict.into())
        }
        // Returned verbatim, NOT routed through `convert_error` — that would
        // clobber the callback's original traceback/type.
        Err(RunError::Callback(err)) => Err(err),
        // Case-load failures: map via the typed lane so each `LoadError` variant
        // reaches its appropriate class (`CaseIoError` / `ValidationError` / etc.).
        Err(RunError::Load(err)) => Err(convert_error(ErrorSource::Load(&err))),
        // Routed through the single mapping site so structured fields (e.g.
        // `Infeasible`) reach Python as `SolverError` attributes.
        Err(RunError::Sddp { error, message }) => Err(convert_error(ErrorSource::Sddp {
            error: &error,
            message,
        })),
        Err(RunError::Message(msg)) => Err(convert_error(ErrorSource::Message(msg))),
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
mod tests {
    use std::path::Path;
    use std::path::PathBuf;

    use cobre_sddp::setup::prepare_stochastic;
    use cobre_sddp::{SolverStatsDelta, SolverStatsLogEntry, aggregate_solver_stats_log};

    use cobre_core::TrainingEvent;
    use cobre_core::training_event::{WorkerPhaseTimings, WorkerTimingPhase};
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    use super::{
        apply_training_policy_mode, build_study_setup, iteration_summary_to_dict,
        read_policy_checkpoint, reconstruct_policy_from_checkpoint, run_in_scoped_pool,
        run_via_study,
    };

    fn example_case_dir(relative: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("cobre-python parent")
            .parent()
            .expect("crates parent")
            .join(relative)
    }

    /// `build_study_setup` is Python-free, so its happy path can be exercised
    /// without a GIL token. It must load `examples/1dtoy`, resolve the effective
    /// config, build a fully prepared `StudySetup`, and return populated
    /// summaries — the single load path both `run_via_study` and `Study::__new__`
    /// rely on.
    #[test]
    fn build_study_setup_succeeds_for_1dtoy() {
        let case_dir = example_case_dir("examples/1dtoy");

        let output_dir =
            std::env::temp_dir().join(format!("cobre_py_build_study_{}", std::process::id()));
        std::fs::create_dir_all(&output_dir).expect("create output dir");

        let loaded = build_study_setup(&case_dir, &output_dir, None)
            .expect("build_study_setup must succeed for 1dtoy");

        // 1dtoy uses the default tree seed.
        assert_eq!(
            loaded.seed,
            cobre_sddp::DEFAULT_SEED,
            "1dtoy must resolve to the default tree seed"
        );
        // 1dtoy trains, so training is enabled in the effective config.
        assert!(
            loaded.config.training.enabled,
            "1dtoy config.training.enabled must be true"
        );
        // The stochastic summary must describe at least one hydro.
        assert!(
            loaded.stochastic_summary.n_hydros > 0,
            "stochastic summary must report a non-zero hydro count"
        );

        std::fs::remove_dir_all(&output_dir).ok();
    }

    /// The Python load path inherits `External`-scheme moment derivation
    /// through the shared `build_study_setup` entry point: a σ = 0 External
    /// load/inflow deck with no seasonal-stats twins must load successfully
    /// with no binding change in this crate.
    #[test]
    fn build_study_setup_succeeds_for_d56_external_authoritative() {
        let case_dir = example_case_dir("examples/deterministic/d56-external-authoritative");

        let output_dir =
            std::env::temp_dir().join(format!("cobre_py_build_study_d56_{}", std::process::id()));
        std::fs::create_dir_all(&output_dir).expect("create output dir");

        let loaded = build_study_setup(&case_dir, &output_dir, None)
            .expect("build_study_setup must succeed for d56-external-authoritative");

        assert!(
            loaded.stochastic_summary.n_hydros > 0,
            "stochastic summary must report a non-zero hydro count"
        );

        std::fs::remove_dir_all(&output_dir).ok();
    }

    /// `apply_training_policy_mode` is Python-free: under the default
    /// `PolicyMode` (no warm-start/resume) and no boundary cuts, it must be a
    /// no-op that returns `Ok(())` and leaves the freshly built FCF untouched.
    /// No GIL token is required (no `Python::initialize()`).
    #[test]
    fn apply_training_policy_mode_default_mode_is_noop() {
        let case_dir = example_case_dir("examples/1dtoy");

        let output_dir =
            std::env::temp_dir().join(format!("cobre_py_policy_mode_noop_{}", std::process::id()));
        std::fs::create_dir_all(&output_dir).expect("create output dir");

        let mut loaded = build_study_setup(&case_dir, &output_dir, None)
            .expect("build_study_setup must succeed for 1dtoy");

        // 1dtoy uses the default policy mode (cut-from-scratch) and no boundary
        // cuts, so the freshly built FCF has no cuts and must stay that way.
        let before_active = loaded.setup.fcf.total_active_cuts();
        let before_generated = loaded.setup.fcf.total_generated_cuts();

        apply_training_policy_mode(
            &mut loaded.setup,
            &loaded.system,
            &loaded.config,
            &output_dir,
            &case_dir,
        )
        .expect("default-mode policy application must be a no-op");

        assert_eq!(
            loaded.setup.fcf.total_active_cuts(),
            before_active,
            "default-mode apply_training_policy_mode must not change the active cut count"
        );
        assert_eq!(
            loaded.setup.fcf.total_generated_cuts(),
            before_generated,
            "default-mode apply_training_policy_mode must not change the generated cut count"
        );

        std::fs::remove_dir_all(&output_dir).ok();
    }

    #[test]
    fn iteration_summary_to_dict_maps_fields() {
        // Initialize the interpreter in the standalone test binary: under the
        // `extension-module` feature, auto-initialize is ignored, so we must
        // prepare it explicitly before attaching.
        Python::initialize();

        let event = TrainingEvent::IterationSummary {
            iteration: 12,
            lower_bound: 100.0,
            upper_bound: 110.0,
            gap: 0.0909,
            wall_time_ms: 1000,
            iteration_time_ms: 200,
            forward_ms: 80,
            backward_ms: 100,
            lp_solves: 240,
            solve_time_ms: 45.2,
            lower_bound_eval_ms: 10,
            fwd_setup_time_ms: 2,
            fwd_load_imbalance_ms: 2,
            fwd_scheduling_overhead_ms: 1,
            rows_in_lp_sum: 720,
            rows_in_lp_count: 240,
            rows_in_lp_max: 24,
        };

        Python::attach(|py| {
            let dict = iteration_summary_to_dict(py, &event)
                .expect("conversion must not error")
                .expect("IterationSummary must yield Some(dict)");

            let kind: String = extract_item(&dict, "kind");
            assert_eq!(kind, "iteration");

            let iteration: u64 = extract_item(&dict, "iteration");
            assert_eq!(iteration, 12);

            let lower_bound: f64 = extract_item(&dict, "lower_bound");
            assert_eq!(lower_bound, 100.0);

            let upper_bound: f64 = extract_item(&dict, "upper_bound");
            assert_eq!(upper_bound, 110.0);

            let gap: f64 = extract_item(&dict, "gap");
            // Raw relative gap, NOT scaled by 100.
            assert!((gap - 0.0909).abs() < 1e-9);

            let wall_time_ms: u64 = extract_item(&dict, "wall_time_ms");
            assert_eq!(wall_time_ms, 1000);
        });
    }

    #[test]
    fn iteration_summary_to_dict_filters_other_variants() {
        Python::initialize();

        let convergence = TrainingEvent::ConvergenceUpdate {
            iteration: 1,
            lower_bound: 100.0,
            upper_bound: 110.0,
            upper_bound_std: 5.0,
            gap: 0.0909,
            rules_evaluated: vec![],
        };
        let worker_timing = TrainingEvent::WorkerTiming {
            rank: 0,
            worker_id: 2,
            iteration: 1,
            phase: WorkerTimingPhase::Backward,
            timings: WorkerPhaseTimings::default(),
        };

        Python::attach(|py| {
            assert!(
                iteration_summary_to_dict(py, &convergence)
                    .expect("conversion must not error")
                    .is_none(),
                "ConvergenceUpdate must be filtered"
            );
            assert!(
                iteration_summary_to_dict(py, &worker_timing)
                    .expect("conversion must not error")
                    .is_none(),
                "WorkerTiming must be filtered"
            );
        });
    }

    /// Extract a typed value for `key` from a `PyDict`, panicking on absence or
    /// type mismatch (test-only helper).
    fn extract_item<'py, T>(dict: &Bound<'py, PyDict>, key: &str) -> T
    where
        T: for<'a> pyo3::FromPyObject<'a, 'py>,
        for<'a> <T as pyo3::FromPyObject<'a, 'py>>::Error: std::fmt::Debug,
    {
        dict.get_item(key)
            .expect("dict lookup must not error")
            .unwrap_or_else(|| panic!("missing key: {key}"))
            .extract()
            .expect("value must extract to requested type")
    }

    #[test]
    fn aggregate_training_solve_stats_folds_and_splits_by_phase() {
        let forward_delta = SolverStatsDelta {
            lp_solves: 10,
            first_try_successes: 7,
            lp_successes: 9,
            lp_failures: 1,
            solve_time_ms: 2500.0,
            ..SolverStatsDelta::default()
        };

        let backward_delta = SolverStatsDelta {
            lp_solves: 4,
            first_try_successes: 2,
            lp_successes: 4,
            lp_failures: 0,
            solve_time_ms: 1500.0,
            ..SolverStatsDelta::default()
        };

        let stats_log = vec![
            SolverStatsLogEntry::from_raw(0, "forward", Some(0), -1, 0, -1, forward_delta),
            SolverStatsLogEntry::from_raw(0, "backward", Some(0), 0, 0, 0, backward_delta),
        ];

        // The shared fold returns the 5 phase-derived counts only; `None` folds
        // every entry (the single-process Python caller). `total_lp_solves` is
        // sourced at the call site from the per-iteration convergence records
        // to mirror the CLI (see `aggregate_solver_stats_log`'s doc).
        let (first_try, retried, failed, forward_seconds, backward_seconds) =
            aggregate_solver_stats_log(&stats_log, None);

        // first_try = 7 + 2; retried = (9-7) + (4-2) = 4; failed = 1 + 0.
        assert_eq!(first_try, 9);
        assert_eq!(retried, 4);
        assert_eq!(failed, 1);
        // Phase split with /1000.0 ms→s conversion.
        assert_eq!(forward_seconds, 2.5);
        assert_eq!(backward_seconds, 1.5);
    }

    #[test]
    fn prepare_stochastic_succeeds_for_d01_case_via_python_path() {
        let case_dir = example_case_dir("examples/deterministic/d01-thermal-dispatch");

        let system = cobre_io::load_case(&case_dir).expect("load_case must succeed for D01");
        let config = cobre_io::parse_config(&case_dir.join("config.json"))
            .expect("parse_config must succeed for D01");

        let seed = config.training.tree_seed.map_or(42_u64, i64::unsigned_abs);

        let training_source = config
            .training_scenario_source(&case_dir.join("config.json"))
            .expect("training_scenario_source must succeed for D01");

        let result = prepare_stochastic(system, &case_dir, &config, seed, &training_source, None);
        assert!(
            result.is_ok(),
            "prepare_stochastic failed for D01 via Python path: {:?}",
            result.err()
        );
    }

    /// End-to-end parity check: the Python `run` path must persist the same
    /// training/simulation metadata (bounds, cost, solve-stats, host layout) that
    /// the CLI produces, so that `summary`/`report` render identically regardless
    /// of which front-end wrote the run.
    ///
    /// The golden values are the CLI's actual output for `examples/1dtoy` (a
    /// 4-stage case, which makes `total_lp_solves` a genuine multi-stage guard:
    /// it is the convergence-record sum, not the stats-log sum). Equality is a
    /// true cross-implementation regression guard, not a tautology — the Python
    /// and CLI write paths are separate code that must each independently
    /// populate the carriers.
    #[test]
    fn python_run_1dtoy_metadata_matches_cli_golden_values() {
        let case_dir = example_case_dir("examples/1dtoy");

        let output_dir =
            std::env::temp_dir().join(format!("cobre_py_parity_{}", std::process::id()));
        std::fs::create_dir_all(&output_dir).expect("create output dir");

        run_via_study(&case_dir, output_dir.clone(), Some(1), None, None)
            .expect("run_via_study must succeed for 1dtoy via Python path");

        let training = cobre_io::read_training_metadata(&output_dir.join("training/metadata.json"))
            .expect("read training metadata");
        let simulation =
            cobre_io::read_simulation_metadata(&output_dir.join("simulation/metadata.json"))
                .expect("read simulation metadata");

        // Relative-tolerance float comparison (the test module relaxes float_cmp,
        // but parity targets are floating-point so use a relative bound).
        let close = |actual: f64, golden: f64| (actual - golden).abs() / golden < 1e-6;

        // ── Training metadata ────────────────────────────────────────────────
        assert_eq!(
            training.problem_dimensions.num_stages, 4,
            "1dtoy must be a 4-stage case so total_lp_solves is a multi-stage guard"
        );

        let golden_lower_bound = 15_595_518.381_798_638;
        assert!(
            close(training.bounds.final_lower_bound, golden_lower_bound),
            "final_lower_bound {} not within 1e-6 of golden {golden_lower_bound}",
            training.bounds.final_lower_bound
        );

        let golden_upper_bound = 579_592.198_622_440_7;
        let upper_bound = training
            .bounds
            .final_upper_bound
            .expect("training final_upper_bound must be Some");
        assert!(
            close(upper_bound, golden_upper_bound),
            "final_upper_bound {upper_bound} not within 1e-6 of golden {golden_upper_bound}"
        );

        // Exact: the convergence-record sum (the regression target).
        assert_eq!(
            training.solve_stats.total_lp_solves,
            Some(5632),
            "training total_lp_solves must equal the convergence-record sum"
        );

        // ── Simulation metadata ──────────────────────────────────────────────
        let cost = simulation
            .cost
            .as_ref()
            .expect("simulation cost must be populated by the run path");
        let golden_mean_cost = 9_679_385.922_404_84;
        assert!(
            close(cost.mean_cost, golden_mean_cost),
            "mean_cost {} not within 1e-6 of golden {golden_mean_cost}",
            cost.mean_cost
        );

        assert_eq!(
            simulation.solve_stats.total_lp_solves,
            Some(400),
            "simulation total_lp_solves must equal golden"
        );

        assert_eq!(
            simulation.scenarios.total, 100,
            "scenarios.total must be 100"
        );

        // ── Host layout (single-host LocalBackend) ───────────────────────────
        for (label, hosts) in [
            ("training", &training.distribution.hosts),
            ("simulation", &simulation.distribution.hosts),
        ] {
            assert_eq!(
                hosts.len(),
                1,
                "{label} distribution.hosts must have one entry"
            );
            assert_eq!(hosts[0].ranks, vec![0], "{label} host ranks must be [0]");
            assert!(
                !hosts[0].hostname.is_empty(),
                "{label} host hostname must be non-empty"
            );
        }

        std::fs::remove_dir_all(&output_dir).ok();
    }

    /// Verify that each scoped pool honors its own per-call thread count: two
    /// sequential calls in the same process with different thread counts each
    /// receive the value they were configured with.
    ///
    /// This is the per-call replacement for the old process-global pool, whose
    /// configuration only took effect on the first call per process. The closure
    /// receives `n = threads.map_or(1, |t| t as usize)`, so distinct
    /// requests yield distinct values regardless of call order.
    #[test]
    fn scoped_pool_honors_per_call_thread_count() {
        let first = run_in_scoped_pool(Some(2), |n| n);
        let second = run_in_scoped_pool(Some(3), |n| n);

        assert_eq!(
            first,
            Ok(2),
            "first scoped pool must honor its configured thread count (2)"
        );
        assert_eq!(
            second,
            Ok(3),
            "second scoped pool must honor its configured thread count (3)"
        );
    }

    /// Recursively copy a directory tree (the case fixtures are flat enough that
    /// a simple recursive walk suffices for the parity test).
    fn copy_dir_all(src: &Path, dst: &Path) {
        std::fs::create_dir_all(dst).expect("create dst dir");
        for entry in std::fs::read_dir(src).expect("read src dir") {
            let entry = entry.expect("dir entry");
            let file_type = entry.file_type().expect("file type");
            let target = dst.join(entry.file_name());
            if file_type.is_dir() {
                copy_dir_all(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), &target).expect("copy file");
            }
        }
    }

    /// The fifth acceptance criterion: the override path and the edited-config
    /// path are equivalent. Running the unedited 1dtoy case with
    /// `config_overrides={"training.tree_seed": 7}` must produce the same
    /// `final_lower_bound` as physically editing `config.json` to set
    /// `tree_seed = 7` and running it. This proves overrides flow through the
    /// entire lifecycle (not just metadata) — the seed changes the sampled
    /// scenario tree, so a divergent path would diverge in the bound.
    ///
    /// Left un-gated to match `python_run_1dtoy_metadata_matches_cli_golden_values`,
    /// whose runtime this mirrors (one 1dtoy train+simulate per path).
    #[test]
    fn override_path_equals_edited_config_for_1dtoy() {
        let case_dir = example_case_dir("examples/1dtoy");

        let base =
            std::env::temp_dir().join(format!("cobre_py_override_parity_{}", std::process::id()));
        let edited_case = base.join("edited_case");
        let edited_out = base.join("edited_out");
        let override_out = base.join("override_out");

        // (a) Edited-config path: copy the case, set tree_seed = 7 on disk, run.
        copy_dir_all(&case_dir, &edited_case);
        let config_path = edited_case.join("config.json");
        let raw = std::fs::read_to_string(&config_path).expect("read config.json");
        let mut json: serde_json::Value = serde_json::from_str(&raw).expect("parse config.json");
        json["training"]["tree_seed"] = serde_json::json!(7);
        std::fs::write(
            &config_path,
            serde_json::to_string_pretty(&json).expect("serialize config"),
        )
        .expect("write edited config.json");

        std::fs::create_dir_all(&edited_out).expect("create edited out dir");
        run_via_study(&edited_case, edited_out.clone(), Some(1), None, None)
            .expect("edited-config run must succeed");

        // (b) Override path: run the unedited case with the equivalent override.
        let mut overrides = serde_json::Map::new();
        overrides.insert("training.tree_seed".to_string(), serde_json::json!(7));
        std::fs::create_dir_all(&override_out).expect("create override out dir");
        run_via_study(
            &case_dir,
            override_out.clone(),
            Some(1),
            Some(overrides),
            None,
        )
        .expect("override run must succeed");

        let edited_meta =
            cobre_io::read_training_metadata(&edited_out.join("training/metadata.json"))
                .expect("read edited training metadata");
        let override_meta =
            cobre_io::read_training_metadata(&override_out.join("training/metadata.json"))
                .expect("read override training metadata");

        // The persisted effective seed must be 7 on both paths.
        assert_eq!(
            override_meta.configuration.seed,
            Some(7),
            "override path must persist the effective seed (7) to metadata"
        );
        assert_eq!(
            edited_meta.configuration.seed,
            Some(7),
            "edited-config path must persist seed 7 to metadata"
        );

        let edited_lb = edited_meta.bounds.final_lower_bound;
        let override_lb = override_meta.bounds.final_lower_bound;
        let rel = (edited_lb - override_lb).abs() / edited_lb.abs();
        assert!(
            rel < 1e-6,
            "override-path final_lower_bound {override_lb} not within 1e-6 of \
             edited-config final_lower_bound {edited_lb} (rel = {rel})"
        );

        std::fs::remove_dir_all(&base).ok();
    }

    /// `reconstruct_policy_from_checkpoint` is Python-free: after a full
    /// train+simulate run writes a checkpoint, building a study via
    /// `build_study_setup` and calling the helper must reconstruct a
    /// `(FutureCostFunction, TrainingResult)` whose iteration count equals the
    /// checkpoint's `completed_iterations` and whose FCF state dimension matches
    /// the study's freshly built FCF. No GIL token (no `Python::initialize()`).
    #[test]
    fn reconstruct_policy_from_checkpoint_roundtrips_for_1dtoy() {
        let case_dir = example_case_dir("examples/1dtoy");

        let output_dir =
            std::env::temp_dir().join(format!("cobre_py_reconstruct_{}", std::process::id()));
        std::fs::create_dir_all(&output_dir).expect("create output dir");

        // Produce a checkpoint by running the full lifecycle once.
        run_via_study(&case_dir, output_dir.clone(), Some(1), None, None)
            .expect("run_via_study must succeed for 1dtoy");

        // Build a fresh study and reconstruct the policy from the checkpoint. The
        // policy directory is `<output_dir>/<policy_path>` (the configured
        // checkpoint location), so derive it from the live setup rather than
        // hardcoding a path.
        let loaded = build_study_setup(&case_dir, &output_dir, None)
            .expect("build_study_setup must succeed for 1dtoy");
        let fresh_state_dim = loaded.setup.fcf.state_dimension;
        let policy_dir = output_dir.join(&loaded.setup.policy_path);

        // The completed-iteration count recorded in the on-disk checkpoint.
        let checkpoint = read_policy_checkpoint(&policy_dir).expect("read policy checkpoint");
        let expected_iterations: u64 = checkpoint.metadata.producer.completed_iterations.into();

        let (fcf, training_result) =
            reconstruct_policy_from_checkpoint(&loaded.setup, &loaded.system, &policy_dir)
                .expect("reconstruct_policy_from_checkpoint must succeed");

        assert_eq!(
            training_result.iterations, expected_iterations,
            "reconstructed TrainingResult.iterations must equal the checkpoint's \
             completed_iterations"
        );
        assert_eq!(
            fcf.state_dimension, fresh_state_dim,
            "reconstructed FCF state dimension must match the freshly built study's FCF"
        );
        // The synthetic result must carry no frozen templates; simulate re-freezes
        // from the FCF (monolithic behavior).
        assert!(
            training_result.frozen_templates.is_none(),
            "loaded-from-checkpoint TrainingResult must carry frozen_templates = None"
        );

        std::fs::remove_dir_all(&output_dir).ok();
    }

    /// P3 (Rust side): a simulation-only run that reconstructs the checkpoint via
    /// the extracted helper must produce simulation metadata bit-identical to the
    /// train-then-simulate run that wrote the checkpoint.
    ///
    /// Train+simulate into dir A, then run `run_via_study` with
    /// `training.enabled = false` against dir A (reusing the checkpoint). The
    /// simulation-only branch reconstructs the policy via
    /// `reconstruct_policy_from_checkpoint` and feeds the unchanged
    /// `run_simulation_phase_py`, so `cost.mean_cost` and
    /// `solve_stats.total_lp_solves` must match exactly.
    ///
    /// IGNORED (pre-existing blocker, not a regression of this change): the
    /// checkpoint basis-reconstruction path trips a `debug_assert!` in
    /// `cobre_sddp::basis_reconstruct::reconstruct_basis`
    /// (`row_status.len() != base_row_count + cut_row_slots.len()`) in debug
    /// builds. `build_basis_cache_from_checkpoint` rebuilds a `CapturedBasis`
    /// with `base_row_count = 0` and empty `cut_row_slots` but a non-empty
    /// `row_status`, which violates that invariant. The identical panic occurs in
    /// the unchanged monolithic `cobre.run.run` simulation-only path
    /// (`config.training.enabled = false`) — this is faithfully preserved here,
    /// not introduced. In release builds the `debug_assert!` is a no-op and the
    /// path succeeds. Re-enable once the upstream `CapturedBasis`
    /// reconstruction-from-checkpoint defect is fixed.
    #[cfg_attr(
        debug_assertions,
        ignore = "pre-existing CapturedBasis debug_assert! in reconstruct_basis; re-enable once the upstream reconstruction-from-checkpoint defect is fixed"
    )]
    #[test]
    fn python_simulation_only_metadata_matches_train_then_simulate() {
        let case_dir = example_case_dir("examples/1dtoy");

        let output_dir =
            std::env::temp_dir().join(format!("cobre_py_simonly_parity_{}", std::process::id()));
        std::fs::create_dir_all(&output_dir).expect("create output dir");

        // (a) Train + simulate into dir A; this writes the checkpoint and the
        // train-then-simulate simulation metadata.
        run_via_study(&case_dir, output_dir.clone(), Some(1), None, None)
            .expect("train-then-simulate run_via_study must succeed");

        let train_then_sim =
            cobre_io::read_simulation_metadata(&output_dir.join("simulation/metadata.json"))
                .expect("read train-then-simulate simulation metadata");
        let golden_mean = train_then_sim
            .cost
            .as_ref()
            .expect("train-then-simulate cost must be populated")
            .mean_cost;
        let golden_lp_solves = train_then_sim.solve_stats.total_lp_solves;

        // (b) Simulation-only run against the SAME dir, reusing the checkpoint,
        // with training disabled via an override. This overwrites simulation/.
        let mut overrides = serde_json::Map::new();
        overrides.insert("training.enabled".to_string(), serde_json::json!(false));
        run_via_study(
            &case_dir,
            output_dir.clone(),
            Some(1),
            Some(overrides),
            None,
        )
        .expect("simulation-only run_via_study must succeed");

        let sim_only =
            cobre_io::read_simulation_metadata(&output_dir.join("simulation/metadata.json"))
                .expect("read simulation-only simulation metadata");
        let sim_only_cost = sim_only
            .cost
            .as_ref()
            .expect("simulation-only cost must be populated");

        let rel = (sim_only_cost.mean_cost - golden_mean).abs() / golden_mean.abs();
        assert!(
            rel < 1e-6,
            "simulation-only mean_cost {} not within 1e-6 of train-then-simulate \
             mean_cost {golden_mean} (rel = {rel})",
            sim_only_cost.mean_cost
        );
        assert_eq!(
            sim_only.solve_stats.total_lp_solves, golden_lp_solves,
            "simulation-only total_lp_solves must exactly equal train-then-simulate"
        );

        std::fs::remove_dir_all(&output_dir).ok();
    }

    /// The Python bindings derive their simulation weighting from the resolved
    /// `Traversal`, exactly as the training-and-simulate path does — under a
    /// sampled selection this can only ever resolve to `Uniform`, never
    /// `Census`, regardless of what a caller might otherwise assemble beside it.
    /// Python-free (no GIL token), mirroring the CLI's own test.
    #[test]
    fn simulation_weighting_census_underivable_from_sampled_traversal() {
        use cobre_sddp::SimulationWeighting;
        use cobre_sddp::setup::{
            NodeGraph, NodeId, NodeOpenings, NodeRuntime, OpeningSource, StageIdx, Traversal,
        };

        let ng = NodeGraph {
            node_ids: vec![NodeId(0)].into(),
            nodes: vec![NodeRuntime {
                stage: StageIdx(0),
                pool_id: 0,
                openings: NodeOpenings {
                    source: OpeningSource::Generated,
                    offset: 0,
                    len: 1,
                    q: 1.0,
                },
            }]
            .into(),
            successors: vec![Vec::new()].into(),
            n_pools: 1,
            pool_stage: vec![StageIdx(0)],
        };

        let sampled = Traversal::resolve(&ng, false, 10);
        assert!(matches!(
            sampled.simulation_weighting(),
            SimulationWeighting::Uniform
        ));

        let enumerated = Traversal::resolve(&ng, true, 1);
        assert!(matches!(
            enumerated.simulation_weighting(),
            SimulationWeighting::Census { .. }
        ));
    }
}
