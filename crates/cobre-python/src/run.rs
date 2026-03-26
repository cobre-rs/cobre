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
use cobre_io::{write_results, ParquetWriterConfig, SolverStatsRow};
use cobre_sddp::{
    build_hydro_model_summary, build_stochastic_summary, prepare_hydro_models, prepare_stochastic,
    ArOrderSummary, EstimationReport, FutureCostFunction, HydroModelSummary,
    SimulationScenarioResult, SolverStatsDelta, StochasticSource, StochasticSummary, StudySetup,
    DEFAULT_SEED,
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

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]
fn write_policy_checkpoint(
    policy_dir: &std::path::Path,
    fcf: &FutureCostFunction,
    training_result: &cobre_sddp::TrainingResult,
    max_iterations: u64,
    forward_passes: u32,
    seed: u64,
) -> Result<(), String> {
    use cobre_io::output::policy::{
        write_policy_checkpoint as io_write_policy_checkpoint, PolicyBasisRecord,
        PolicyCheckpointMetadata, PolicyCutRecord, StageCutsPayload,
    };

    let n_stages = fcf.pools.len();
    let state_dimension = fcf.state_dimension;

    let stage_records: Vec<Vec<PolicyCutRecord<'_>>> = fcf
        .pools
        .iter()
        .map(|pool| {
            (0..pool.populated_count)
                .filter(|&i| pool.active[i])
                .map(|i| {
                    let meta = &pool.metadata[i];
                    PolicyCutRecord {
                        cut_id: meta.iteration_generated * u64::from(pool.forward_passes)
                            + u64::from(meta.forward_pass_index),
                        slot_index: i as u32,
                        iteration: meta.iteration_generated as u32,
                        forward_pass_index: meta.forward_pass_index,
                        intercept: pool.intercepts[i],
                        coefficients: &pool.coefficients[i],
                        is_active: true,
                        domination_count: meta.active_count as u32,
                    }
                })
                .collect()
        })
        .collect();

    let stage_active_indices: Vec<Vec<u32>> = stage_records
        .iter()
        .map(|records| records.iter().map(|r| r.slot_index).collect())
        .collect();

    let stage_cuts: Vec<StageCutsPayload<'_>> = fcf
        .pools
        .iter()
        .enumerate()
        .map(|(stage_idx, pool)| StageCutsPayload {
            stage_id: stage_idx as u32,
            state_dimension: state_dimension as u32,
            capacity: pool.capacity as u32,
            warm_start_count: pool.warm_start_count,
            cuts: &stage_records[stage_idx],
            active_cut_indices: &stage_active_indices[stage_idx],
            populated_count: pool.populated_count as u32,
        })
        .collect();

    let basis_col_u8: Vec<Vec<u8>> = training_result
        .basis_cache
        .iter()
        .map(|opt| {
            opt.as_ref()
                .map(|b| b.col_status.iter().map(|&v| v as u8).collect())
                .unwrap_or_default()
        })
        .collect();

    let basis_row_u8: Vec<Vec<u8>> = training_result
        .basis_cache
        .iter()
        .map(|opt| {
            opt.as_ref()
                .map(|b| b.row_status.iter().map(|&v| v as u8).collect())
                .unwrap_or_default()
        })
        .collect();

    let stage_bases: Vec<PolicyBasisRecord<'_>> = training_result
        .basis_cache
        .iter()
        .enumerate()
        .filter_map(|(stage_idx, opt)| {
            opt.as_ref().map(|_| {
                let num_cut_rows = fcf
                    .pools
                    .get(stage_idx)
                    .map_or(0, |pool| pool.populated_count.min(pool.capacity) as u32);
                PolicyBasisRecord {
                    stage_id: stage_idx as u32,
                    iteration: training_result.iterations as u32,
                    column_status: &basis_col_u8[stage_idx],
                    row_status: &basis_row_u8[stage_idx],
                    num_cut_rows,
                }
            })
        })
        .collect();

    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|d| format!("{}s-since-epoch", d.as_secs()))
        .unwrap_or_else(|_| "unknown".to_string());

    let metadata = PolicyCheckpointMetadata {
        version: "1.0.0".to_string(),
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at,
        completed_iterations: training_result.iterations as u32,
        final_lower_bound: training_result.final_lb,
        best_upper_bound: Some(training_result.final_ub),
        state_dimension: state_dimension as u32,
        num_stages: n_stages as u32,
        config_hash: String::new(),
        system_hash: String::new(),
        max_iterations: max_iterations as u32,
        forward_passes,
        warm_start_cuts: 0,
        rng_seed: seed,
    };

    io_write_policy_checkpoint(policy_dir, &stage_cuts, &stage_bases, &metadata)
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
        simplex_iterations: delta.simplex_iterations,
        solve_time_ms: delta.solve_time_ms,
        load_model_time_ms: delta.load_model_time_ms,
        add_rows_time_ms: delta.add_rows_time_ms,
        set_bounds_time_ms: delta.set_bounds_time_ms,
        basis_set_time_ms: delta.basis_set_time_ms,
    }
}

/// Run the full solve lifecycle without MPI or progress bars (GIL released for computation).
#[allow(clippy::too_many_lines)]
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

    // Seed must be extracted before config is moved into StudySetup.
    let seed = config.training.seed.map_or(DEFAULT_SEED, i64::unsigned_abs);
    let should_simulate =
        !skip_simulation && config.simulation.enabled && config.simulation.num_scenarios > 0;

    let result = prepare_stochastic(system, case_dir, &config, seed)
        .map_err(|e| format!("stochastic preprocessing error: {e}"))?;
    let system = result.system;
    let estimation_report = result.estimation_report;

    let hydro_models_result = prepare_hydro_models(&system, case_dir)
        .map_err(|e| format!("hydro model preprocessing error: {e}"))?;

    let mut setup = StudySetup::new(&system, &config, result.stochastic, hydro_models_result)
        .map_err(|e| e.to_string())?;

    // Export stochastic artifacts when requested.
    if config.exports.stochastic {
        export_stochastic_artifacts_py(
            &output_dir,
            setup.stochastic(),
            &system,
            estimation_report.as_ref(),
        );
    }

    // Write scaling report (before training starts).
    let scaling_path = output_dir.join("training/scaling_report.json");
    cobre_io::write_scaling_report(&scaling_path, setup.scaling_report())
        .map_err(|e| format!("failed to write scaling report: {e}"))?;

    let mut solver = HighsSolver::new().map_err(|e| format!("HiGHS initialisation failed: {e}"))?;
    let (event_tx, event_rx) = mpsc::channel();
    let training_outcome = setup
        .train(
            &mut solver,
            &LocalBackend,
            n_threads,
            HighsSolver::new,
            Some(event_tx),
            None,
        )
        .map_err(|e| format!("training error: {e}"))?;
    if let Some(ref e) = training_outcome.error {
        return Err(format!(
            "training failed after {} iterations: {e}",
            training_outcome.result.iterations
        ));
    }
    let training_result = training_outcome.result;

    let events: Vec<_> = event_rx.try_iter().collect();
    let training_output = setup.build_training_output(&training_result, &events);

    write_policy_checkpoint(
        &output_dir.join(setup.policy_path()),
        setup.fcf(),
        &training_result,
        setup.max_iterations(),
        setup.forward_passes(),
        seed,
    )
    .map_err(|e| format!("policy checkpoint error: {e}"))?;

    let converged = training_output.converged;
    let iterations = training_result.iterations;
    let lower_bound = training_result.final_lb;
    let gap_percent = Some(training_result.final_gap * 100.0);
    let total_time_ms = training_result.total_time_ms;

    let stochastic_summary = build_stochastic_summary(
        &system,
        setup.stochastic(),
        estimation_report.as_ref(),
        seed,
    );

    let hydro_models_summary = Some(build_hydro_model_summary(setup.hydro_models(), &system));

    // Write training solver stats to training/solver/iterations.parquet.
    if !training_result.solver_stats_log.is_empty() {
        let rows: Vec<SolverStatsRow> = training_result
            .solver_stats_log
            .iter()
            .map(|(iter, phase, stage, delta)| {
                #[allow(clippy::cast_possible_truncation)] // iteration count fits in u32
                delta_to_stats_row(*iter as u32, phase, *stage, delta)
            })
            .collect();
        cobre_io::write_solver_stats(&output_dir, &rows)
            .map_err(|e| format!("solver stats output: {e}"))?;
    }

    // Write per-stage cut selection records (training-only, no simulation dependency).
    if !training_output.cut_selection_records.is_empty() {
        cobre_io::write_cut_selection_records(
            &output_dir,
            &training_output.cut_selection_records,
            &ParquetWriterConfig::default(),
        )
        .map_err(|e| format!("cut selection output: {e}"))?;
    }

    if should_simulate {
        let io_capacity = setup.simulation_config().io_channel_capacity;
        let mut sim_pool = setup
            .create_workspace_pool(n_threads, HighsSolver::new)
            .map_err(|e| format!("HiGHS initialisation failed for simulation pool: {e}"))?;
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity.max(1));
        // Drain thread prevents bounded-channel deadlock when n_scenarios > io_capacity.
        let drain_handle = std::thread::spawn(move || {
            result_rx
                .into_iter()
                .collect::<Vec<SimulationScenarioResult>>()
        });
        let sim_result = setup
            .simulate(
                &mut sim_pool.workspaces,
                &LocalBackend,
                &result_tx,
                None,
                &training_result.basis_cache,
            )
            .map_err(|e| format!("simulation error: {e}"));
        drop(result_tx);
        let local_results = drain_handle
            .join()
            .map_err(|_| "drain thread panicked".to_string())?;
        let sim_run_result = sim_result?;

        // Write simulation solver stats to simulation/solver/scenarios.parquet.
        if !sim_run_result.solver_stats.is_empty() {
            let rows: Vec<SolverStatsRow> = sim_run_result
                .solver_stats
                .iter()
                .map(|(scenario_id, delta)| {
                    delta_to_stats_row(*scenario_id, "simulation", -1, delta)
                })
                .collect();
            cobre_io::write_simulation_solver_stats(&output_dir, &rows)
                .map_err(|e| format!("simulation solver stats output: {e}"))?;
        }

        let mut sim_writer =
            SimulationParquetWriter::new(&output_dir, &system, &ParquetWriterConfig::default())
                .map_err(|e| format!("simulation writer initialisation error: {e}"))?;
        let mut failed: u32 = 0;
        for scenario_result in local_results {
            if let Err(e) = sim_writer.write_scenario(ScenarioWritePayload::from(scenario_result)) {
                eprintln!("cobre-python: simulation write warning: {e}");
                failed += 1;
            }
        }
        let mut sim_out = sim_writer.finalize(0);
        sim_out.failed = failed;
        let sim_summary = SimSummary {
            n_scenarios: sim_out.n_scenarios,
            completed: sim_out.completed,
        };
        write_results(
            &output_dir,
            &training_output,
            Some(&sim_out),
            &system,
            &config,
        )
        .map_err(|e| format!("output write error: {e}"))?;
        Ok(RunSummary {
            converged,
            iterations,
            lower_bound,
            upper_bound: Some(training_result.final_ub),
            gap_percent,
            total_time_ms,
            output_dir,
            simulation: Some(sim_summary),
            stochastic: Some(stochastic_summary),
            hydro_models: hydro_models_summary,
        })
    } else {
        write_results(&output_dir, &training_output, None, &system, &config)
            .map_err(|e| format!("output write error: {e}"))?;
        Ok(RunSummary {
            converged,
            iterations,
            lower_bound,
            upper_bound: Some(training_result.final_ub),
            gap_percent,
            total_time_ms,
            output_dir,
            simulation: None,
            stochastic: Some(stochastic_summary),
            hydro_models: hydro_models_summary,
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

/// Load a case, train an SDDP policy, optionally simulate, and write results.
/// GIL is released for the entire Rust computation.
/// Returns a dict with keys: `"converged"`, `"iterations"`, `"lower_bound"`, `"upper_bound"`,
/// `"gap_percent"`, `"total_time_ms"`, `"output_dir"`, `"simulation"`, `"stochastic"`.
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
