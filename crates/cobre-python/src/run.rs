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
use cobre_io::output::simulation_writer::{
    BusWriteRecord, ContractWriteRecord, CostWriteRecord, ExchangeWriteRecord,
    GenericViolationWriteRecord, HydroWriteRecord, InflowLagWriteRecord,
    NonControllableWriteRecord, PumpingWriteRecord, ScenarioWritePayload, SimulationParquetWriter,
    StageWritePayload, ThermalWriteRecord,
};
use cobre_io::{write_results, ParquetWriterConfig};
use cobre_sddp::{FutureCostFunction, SimulationScenarioResult, SimulationStageResult, StudySetup};
use cobre_solver::HighsSolver;
use cobre_stochastic::build_stochastic_context;

const DEFAULT_SEED: u64 = 42;

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
}

struct SimSummary {
    n_scenarios: u32,
    completed: u32,
}

fn init_rayon(threads: Option<u32>) {
    let n = threads.map_or(1, |t| t as usize);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .unwrap_or(());
}

fn convert_stage(src: SimulationStageResult) -> StageWritePayload {
    StageWritePayload {
        stage_id: src.stage_id,
        costs: src
            .costs
            .into_iter()
            .map(|s| CostWriteRecord {
                stage_id: s.stage_id,
                block_id: s.block_id,
                total_cost: s.total_cost,
                immediate_cost: s.immediate_cost,
                future_cost: s.future_cost,
                discount_factor: s.discount_factor,
                thermal_cost: s.thermal_cost,
                contract_cost: s.contract_cost,
                deficit_cost: s.deficit_cost,
                excess_cost: s.excess_cost,
                storage_violation_cost: s.storage_violation_cost,
                filling_target_cost: s.filling_target_cost,
                hydro_violation_cost: s.hydro_violation_cost,
                inflow_penalty_cost: s.inflow_penalty_cost,
                generic_violation_cost: s.generic_violation_cost,
                spillage_cost: s.spillage_cost,
                fpha_turbined_cost: s.fpha_turbined_cost,
                curtailment_cost: s.curtailment_cost,
                exchange_cost: s.exchange_cost,
                pumping_cost: s.pumping_cost,
            })
            .collect(),
        hydros: src
            .hydros
            .into_iter()
            .map(|s| HydroWriteRecord {
                stage_id: s.stage_id,
                block_id: s.block_id,
                hydro_id: s.hydro_id,
                turbined_m3s: s.turbined_m3s,
                spillage_m3s: s.spillage_m3s,
                evaporation_m3s: s.evaporation_m3s,
                diverted_inflow_m3s: s.diverted_inflow_m3s,
                diverted_outflow_m3s: s.diverted_outflow_m3s,
                incremental_inflow_m3s: s.incremental_inflow_m3s,
                inflow_m3s: s.inflow_m3s,
                storage_initial_hm3: s.storage_initial_hm3,
                storage_final_hm3: s.storage_final_hm3,
                generation_mw: s.generation_mw,
                productivity_mw_per_m3s: s.productivity_mw_per_m3s,
                spillage_cost: s.spillage_cost,
                water_value_per_hm3: s.water_value_per_hm3,
                storage_binding_code: s.storage_binding_code,
                operative_state_code: s.operative_state_code,
                turbined_slack_m3s: s.turbined_slack_m3s,
                outflow_slack_below_m3s: s.outflow_slack_below_m3s,
                outflow_slack_above_m3s: s.outflow_slack_above_m3s,
                generation_slack_mw: s.generation_slack_mw,
                storage_violation_below_hm3: s.storage_violation_below_hm3,
                filling_target_violation_hm3: s.filling_target_violation_hm3,
                evaporation_violation_m3s: s.evaporation_violation_m3s,
                inflow_nonnegativity_slack_m3s: s.inflow_nonnegativity_slack_m3s,
            })
            .collect(),
        thermals: src
            .thermals
            .into_iter()
            .map(|s| ThermalWriteRecord {
                stage_id: s.stage_id,
                block_id: s.block_id,
                thermal_id: s.thermal_id,
                generation_mw: s.generation_mw,
                generation_cost: s.generation_cost,
                is_gnl: s.is_gnl,
                gnl_committed_mw: s.gnl_committed_mw,
                gnl_decision_mw: s.gnl_decision_mw,
                operative_state_code: s.operative_state_code,
            })
            .collect(),
        exchanges: src
            .exchanges
            .into_iter()
            .map(|s| ExchangeWriteRecord {
                stage_id: s.stage_id,
                block_id: s.block_id,
                line_id: s.line_id,
                direct_flow_mw: s.direct_flow_mw,
                reverse_flow_mw: s.reverse_flow_mw,
                exchange_cost: s.exchange_cost,
                operative_state_code: s.operative_state_code,
            })
            .collect(),
        buses: src
            .buses
            .into_iter()
            .map(|s| BusWriteRecord {
                stage_id: s.stage_id,
                block_id: s.block_id,
                bus_id: s.bus_id,
                load_mw: s.load_mw,
                deficit_mw: s.deficit_mw,
                excess_mw: s.excess_mw,
                spot_price: s.spot_price,
            })
            .collect(),
        pumping_stations: src
            .pumping_stations
            .into_iter()
            .map(|s| PumpingWriteRecord {
                stage_id: s.stage_id,
                block_id: s.block_id,
                pumping_station_id: s.pumping_station_id,
                pumped_flow_m3s: s.pumped_flow_m3s,
                power_consumption_mw: s.power_consumption_mw,
                pumping_cost: s.pumping_cost,
                operative_state_code: s.operative_state_code,
            })
            .collect(),
        contracts: src
            .contracts
            .into_iter()
            .map(|s| ContractWriteRecord {
                stage_id: s.stage_id,
                block_id: s.block_id,
                contract_id: s.contract_id,
                power_mw: s.power_mw,
                price_per_mwh: s.price_per_mwh,
                total_cost: s.total_cost,
                operative_state_code: s.operative_state_code,
            })
            .collect(),
        non_controllables: src
            .non_controllables
            .into_iter()
            .map(|s| NonControllableWriteRecord {
                stage_id: s.stage_id,
                block_id: s.block_id,
                non_controllable_id: s.non_controllable_id,
                generation_mw: s.generation_mw,
                available_mw: s.available_mw,
                curtailment_mw: s.curtailment_mw,
                curtailment_cost: s.curtailment_cost,
                operative_state_code: s.operative_state_code,
            })
            .collect(),
        inflow_lags: src
            .inflow_lags
            .into_iter()
            .map(|s| InflowLagWriteRecord {
                stage_id: s.stage_id,
                hydro_id: s.hydro_id,
                lag_index: s.lag_index,
                inflow_m3s: s.inflow_m3s,
            })
            .collect(),
        generic_violations: src
            .generic_violations
            .into_iter()
            .map(|s| GenericViolationWriteRecord {
                stage_id: s.stage_id,
                block_id: s.block_id,
                constraint_id: s.constraint_id,
                slack_value: s.slack_value,
                slack_cost: s.slack_cost,
            })
            .collect(),
    }
}

fn convert_scenario(src: SimulationScenarioResult) -> ScenarioWritePayload {
    ScenarioWritePayload {
        scenario_id: src.scenario_id,
        stages: src.stages.into_iter().map(convert_stage).collect(),
    }
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

    let created_at = {
        use std::time::SystemTime;
        match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
            Ok(d) => format!("{}s-since-epoch", d.as_secs()),
            Err(_) => "unknown".to_string(),
        }
    };

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

    // Seed must be extracted before config is moved into StudySetup.
    let seed = config.training.seed.map_or(DEFAULT_SEED, i64::unsigned_abs);
    let should_simulate =
        !skip_simulation && config.simulation.enabled && config.simulation.num_scenarios > 0;

    let stochastic = build_stochastic_context(&system, seed, &[], None)
        .map_err(|e| format!("stochastic context error: {e}"))?;

    let mut setup = StudySetup::new(&system, &config, stochastic).map_err(|e| e.to_string())?;

    let mut solver = HighsSolver::new().map_err(|e| format!("HiGHS initialisation failed: {e}"))?;
    let (event_tx, event_rx) = mpsc::channel();
    let training_result = setup
        .train(
            &mut solver,
            &LocalBackend,
            n_threads,
            HighsSolver::new,
            Some(event_tx),
            None,
        )
        .map_err(|e| format!("training error: {e}"))?;

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
    let upper_bound = Some(training_result.final_ub);
    let gap_percent = Some(training_result.final_gap * 100.0);
    let total_time_ms = training_result.total_time_ms;

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
            .simulate(&mut sim_pool.workspaces, &LocalBackend, &result_tx, None)
            .map_err(|e| format!("simulation error: {e}"));
        drop(result_tx);
        let local_results = drain_handle
            .join()
            .map_err(|_| "drain thread panicked".to_string())?;
        sim_result?;

        let mut sim_writer =
            SimulationParquetWriter::new(&output_dir, &system, &ParquetWriterConfig::default())
                .map_err(|e| format!("simulation writer initialisation error: {e}"))?;
        let mut failed: u32 = 0;
        for scenario_result in local_results {
            if let Err(e) = sim_writer.write_scenario(convert_scenario(scenario_result)) {
                eprintln!("cobre-python: simulation write warning: {e}");
                failed += 1;
            }
        }
        let mut sim_out = sim_writer.finalize();
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
            upper_bound,
            gap_percent,
            total_time_ms,
            output_dir,
            simulation: Some(sim_summary),
        })
    } else {
        write_results(&output_dir, &training_output, None, &system, &config)
            .map_err(|e| format!("output write error: {e}"))?;
        Ok(RunSummary {
            converged,
            iterations,
            lower_bound,
            upper_bound,
            gap_percent,
            total_time_ms,
            output_dir,
            simulation: None,
        })
    }
}

/// Load a case, train an SDDP policy, optionally simulate, and write results.
/// GIL is released for the entire Rust computation.
/// Returns a dict with keys: "converged", "iterations", "lower_bound", "upper_bound",
/// "gap_percent", "total_time_ms", "output_dir", "simulation".
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

            if let Some(sim) = summary.simulation {
                let sim_dict = PyDict::new(py);
                sim_dict.set_item("n_scenarios", sim.n_scenarios)?;
                sim_dict.set_item("completed", sim.completed)?;
                dict.set_item("simulation", sim_dict)?;
            } else {
                dict.set_item("simulation", py.None())?;
            }

            Ok(dict.into())
        }
        Err(msg) => {
            if msg.as_str().starts_with("output write error")
                || msg.as_str().starts_with("policy checkpoint error")
            {
                Err(PyOSError::new_err(msg))
            } else {
                Err(PyRuntimeError::new_err(msg))
            }
        }
    }
}
