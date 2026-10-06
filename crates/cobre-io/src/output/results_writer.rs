//! Aggregate output writers that combine training and simulation artifacts.

use std::path::Path;

use cobre_core::System;

use super::dictionary::write_dictionaries;
use super::error::OutputError;
use super::manifest::{
    MetadataBounds, MetadataConfiguration, MetadataConvergence, MetadataIterations,
    MetadataProblemDimensions, MetadataRowPool, MetadataScenarios, MetadataSimulationSolveStats,
    OutputContext, RunStatus, SimulationMetadata, TrainingMetadata, write_simulation_metadata,
    write_training_metadata,
};
use super::software::{SOFTWARE_NAME, SOFTWARE_VERSION};
use super::training_writer::TrainingParquetWriter;
use super::{SimulationOutput, TrainingOutput};
use crate::Config;
use crate::config::{ForwardPassesResolution, StoppingRuleConfig};

/// Write all training artifacts to the output directory.
///
/// Also creates an empty `simulation/` directory so downstream code can
/// unconditionally write into it.
///
/// # Errors
///
/// Returns [`OutputError`] if any directory creation or file write fails.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn write_training_results(
    output_dir: &Path,
    training_output: &TrainingOutput,
    system: &System,
    config: &Config,
    ctx: &OutputContext,
) -> Result<(), OutputError> {
    create_output_dir(&output_dir.join("training/dictionaries"))?;
    create_output_dir(&output_dir.join("training/timing"))?;
    create_output_dir(&output_dir.join("simulation"))?;

    write_dictionaries(&output_dir.join("training/dictionaries"), system)?;

    let writer = TrainingParquetWriter::new(output_dir)?;
    writer.write(training_output)?;

    let converged_at = training_output
        .converged
        .then_some(training_output.iterations_completed);

    let max_iterations = extract_max_iterations(config);

    let metadata = TrainingMetadata {
        software: SOFTWARE_NAME.to_string(),
        software_version: SOFTWARE_VERSION.to_string(),
        hostname: ctx.hostname.clone(),
        solver: ctx.solver.clone(),
        solver_version: ctx.solver_version.clone(),
        started_at: ctx.started_at.clone(),
        completed_at: ctx.completed_at.clone(),
        duration_seconds: training_output.total_time_ms as f64 / 1_000.0,
        status: training_output.status,
        configuration: MetadataConfiguration {
            seed: config.training.tree_seed,
            max_iterations,
            forward_passes: match config.resolve_forward_passes() {
                Some(ForwardPassesResolution::Sampled(n)) => Some(n),
                Some(ForwardPassesResolution::Enumerated) | None => None,
            },
            stopping_mode: config.training.stopping_mode.to_string(),
            policy_mode: config.policy.mode.to_string(),
        },
        problem_dimensions: MetadataProblemDimensions {
            num_stages: system.n_stages() as u32,
            num_hydros: system.n_hydros() as u32,
            num_thermals: system.n_thermals() as u32,
            num_buses: system.n_buses() as u32,
            num_lines: system.n_lines() as u32,
        },
        iterations: MetadataIterations {
            completed: training_output.iterations_completed,
            converged_at,
        },
        convergence: MetadataConvergence {
            achieved: training_output.converged,
            final_gap_percent: training_output.final_gap_percent,
            termination_reason: training_output.termination_reason.clone(),
        },
        row_pool: MetadataRowPool {
            total_generated: training_output.cut_stats.total_generated,
            total_active: training_output.cut_stats.total_active,
            peak_active: training_output.cut_stats.peak_active,
            cuts_active: training_output.cut_stats.cuts_active,
            rows_in_lp_total: training_output.cut_stats.rows_in_lp_total,
            rows_in_lp_solve_count: training_output.cut_stats.rows_in_lp_solve_count,
            rows_in_lp_max: training_output.cut_stats.rows_in_lp_max,
            total_loaded: training_output.cut_stats.total_loaded,
        },
        bounds: MetadataBounds {
            final_lower_bound: training_output.final_lower_bound,
            final_upper_bound: training_output.final_upper_bound,
            final_upper_bound_std: training_output.final_upper_bound_std,
            final_upper_bound_kind: training_output.final_upper_bound_kind.clone(),
        },
        solve_stats: training_output.training_solve_stats.clone(),
        setup: ctx.setup.clone(),
        production_fit_deviation: ctx.production_fit_deviation.clone(),
        distribution: ctx.distribution.clone(),
    };
    write_training_metadata(&output_dir.join("training/metadata.json"), &metadata)?;

    Ok(())
}

pub(crate) const SIMULATION_METADATA_FILE: &str = "simulation/metadata.json";

/// Write simulation artifacts to the output directory.
///
/// The `simulation/` directory must already exist (created by
/// [`write_training_results`]).
///
/// # Errors
///
/// Returns [`OutputError`] if metadata serialization or file I/O fails.
pub fn write_simulation_results(
    output_dir: &Path,
    simulation_output: &SimulationOutput,
    ctx: &OutputContext,
) -> Result<(), OutputError> {
    let metadata = simulation_metadata(simulation_output, RunStatus::Complete, ctx);
    write_simulation_metadata(&output_dir.join(SIMULATION_METADATA_FILE), &metadata)?;

    Ok(())
}

/// Write the metadata of a simulation that was skipped before any scenario
/// ran: `status` partial, `n_scenarios` in total and none completed.
///
/// Creates the `simulation/` directory. Writes no `_SUCCESS` marker; that is
/// the caller's last write.
///
/// # Errors
///
/// Returns [`OutputError`] if the directory cannot be created, or if metadata
/// serialization or file I/O fails.
pub fn write_skipped_simulation_results(
    output_dir: &Path,
    n_scenarios: u32,
    ctx: &OutputContext,
) -> Result<(), OutputError> {
    create_output_dir(&output_dir.join("simulation"))?;
    let skipped = SimulationOutput {
        n_scenarios,
        completed: 0,
        failed: 0,
        total_time_ms: 0,
        cost: None,
        solve_stats: MetadataSimulationSolveStats::default(),
    };
    let metadata = simulation_metadata(&skipped, RunStatus::Partial, ctx);
    write_simulation_metadata(&output_dir.join(SIMULATION_METADATA_FILE), &metadata)
}

#[allow(clippy::cast_precision_loss)]
fn simulation_metadata(
    simulation_output: &SimulationOutput,
    status: RunStatus,
    ctx: &OutputContext,
) -> SimulationMetadata {
    SimulationMetadata {
        software: SOFTWARE_NAME.to_string(),
        software_version: SOFTWARE_VERSION.to_string(),
        hostname: ctx.hostname.clone(),
        solver: ctx.solver.clone(),
        solver_version: ctx.solver_version.clone(),
        started_at: ctx.started_at.clone(),
        completed_at: ctx.completed_at.clone(),
        duration_seconds: simulation_output.total_time_ms as f64 / 1_000.0,
        status,
        scenarios: MetadataScenarios {
            total: simulation_output.n_scenarios,
            completed: simulation_output.completed,
            failed: simulation_output.failed,
        },
        cost: simulation_output.cost.clone(),
        solve_stats: simulation_output.solve_stats.clone(),
        distribution: ctx.distribution.clone(),
    }
}

/// Write the training result tables and, when supplied, the simulation
/// completion metadata to the output directory.
///
/// # Errors
///
/// Returns [`OutputError`] if any file I/O or serialization step fails.
pub fn write_results(
    output_dir: &Path,
    training_output: &TrainingOutput,
    simulation_output: Option<&SimulationOutput>,
    system: &System,
    config: &Config,
    ctx: &OutputContext,
) -> Result<(), OutputError> {
    write_training_results(output_dir, training_output, system, config, ctx)?;
    if let Some(sim) = simulation_output {
        write_simulation_results(output_dir, sim, ctx)?;
    }
    Ok(())
}

const SUCCESS_MARKER_FILE: &str = "_SUCCESS";

/// Write the empty `_SUCCESS` marker into a phase directory.
///
/// The marker means its phase finished writing, so each phase-writer calls this
/// as its last write. `phase_dir` is not created: a marker in a directory the
/// phase never wrote to would be false. A simulation scenario whose partition
/// could not be written is counted in `scenarios.failed` in
/// `simulation/metadata.json` and does not withhold the marker.
///
/// # Errors
///
/// Returns [`OutputError::IoError`] when `phase_dir` does not exist or the
/// marker cannot be created.
pub fn write_success_marker(phase_dir: &Path) -> Result<(), OutputError> {
    let marker_path = phase_dir.join(SUCCESS_MARKER_FILE);
    std::fs::write(&marker_path, b"").map_err(|e| OutputError::io(&marker_path, e))
}

/// Remove the `_SUCCESS` marker from a phase directory.
///
/// Callers invoke this before the phase's first write, so a reused output
/// directory never shows the previous run's marker beside files the new run
/// is still writing, and a run that fails mid-phase leaves no marker. A missing
/// marker or a missing `phase_dir` is success.
///
/// # Errors
///
/// Returns [`OutputError::IoError`] when the marker exists but cannot be
/// removed (for example, `_SUCCESS` is a directory): keeping it would keep a
/// false marker.
pub fn remove_success_marker(phase_dir: &Path) -> Result<(), OutputError> {
    let marker_path = phase_dir.join(SUCCESS_MARKER_FILE);
    match std::fs::remove_file(&marker_path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(OutputError::io(&marker_path, e)),
    }
}

fn extract_max_iterations(config: &Config) -> Option<u32> {
    config
        .training
        .stopping_rules
        .as_ref()?
        .iter()
        .find_map(|r| match r {
            StoppingRuleConfig::IterationLimit { limit } => Some(*limit),
            _ => None,
        })
}

fn create_output_dir(dir: &Path) -> Result<(), OutputError> {
    std::fs::create_dir_all(dir).map_err(|e| OutputError::io(dir, e))
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;
    use crate::output::{IterationRecord, RowPoolStatistics, TrainingOutput};
    use crate::test_support::output::read_first_batch;
    use crate::test_support::output::{make_config, make_output_context, make_system};
    use crate::{MetadataSimulationSolveStats, MetadataTrainingSolveStats};

    fn make_iteration_record(iteration: u32) -> IterationRecord {
        IterationRecord {
            iteration,
            lower_bound: 1.0,
            upper_bound: 2.0,
            upper_bound_std: 0.1,
            gap_percent: Some(50.0),
            cuts_added: 10,
            cuts_removed: 2,
            cuts_active: 8,
            time_forward_ms: 100,
            time_backward_ms: 200,
            time_total_ms: 300,
            forward_passes: 4,
            lp_solves: 40,
            time_forward_wall_ms: 100,
            time_backward_wall_ms: 200,
            time_cut_selection_ms: 0,
            time_mpi_allreduce_ms: 0,
            time_cut_sync_ms: 0,
            time_lower_bound_ms: 0,
            time_state_exchange_ms: 0,
            time_cut_batch_build_ms: 0,
            time_bwd_load_imbalance_ms: 0,
            time_bwd_scheduling_overhead_ms: 0,
            time_fwd_load_imbalance_ms: 0,
            time_fwd_scheduling_overhead_ms: 0,
            time_overhead_ms: 0,
            solve_time_ms: 0.0,
            mean_rows_in_lp: 0.0,
        }
    }

    fn make_training_output(n_records: usize) -> TrainingOutput {
        let records = (1..=n_records as u32).map(make_iteration_record).collect();
        TrainingOutput {
            convergence_records: records,
            final_lower_bound: 99.5,
            final_upper_bound: Some(101.0),
            final_gap_percent: Some(1.51),
            final_upper_bound_std: Some(0.5),
            final_upper_bound_kind: "statistical".to_string(),
            iterations_completed: n_records as u32,
            converged: true,
            termination_reason: "gap tolerance reached".to_string(),
            status: RunStatus::Complete,
            total_time_ms: 5_000,
            cut_stats: RowPoolStatistics {
                total_generated: 200,
                total_active: 80,
                peak_active: 95,
                cuts_active: 0,
                rows_in_lp_total: 0,
                rows_in_lp_solve_count: 0,
                rows_in_lp_max: 0,
                total_loaded: 0,
            },
            cut_selection_records: vec![],
            worker_timing_records: vec![],
            training_solve_stats: MetadataTrainingSolveStats::default(),
        }
    }

    fn make_simulation_output() -> SimulationOutput {
        SimulationOutput {
            n_scenarios: 10,
            completed: 10,
            failed: 0,
            total_time_ms: 1_000,
            cost: None,
            solve_stats: MetadataSimulationSolveStats::default(),
        }
    }

    #[test]
    fn write_results_creates_training_directories() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);

        write_results(
            tmp.path(),
            &training,
            None,
            &make_system(),
            &make_config(),
            &make_output_context(),
        )
        .expect("write_results must succeed");

        assert!(tmp.path().join("training").is_dir(), "training/ must exist");
        assert!(
            tmp.path().join("training/dictionaries").is_dir(),
            "training/dictionaries/ must exist"
        );
        assert!(
            tmp.path().join("training/timing").is_dir(),
            "training/timing/ must exist"
        );
    }

    #[test]
    fn write_results_creates_simulation_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);

        write_results(
            tmp.path(),
            &training,
            None,
            &make_system(),
            &make_config(),
            &make_output_context(),
        )
        .expect("write_results must succeed with simulation_output = None");

        assert!(
            tmp.path().join("simulation").is_dir(),
            "simulation/ must exist even when simulation_output is None"
        );
    }

    #[test]
    fn write_results_returns_ok_on_success() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(3);
        let simulation = SimulationOutput {
            n_scenarios: 10,
            completed: 10,
            failed: 0,
            total_time_ms: 1_500,
            cost: None,
            solve_stats: MetadataSimulationSolveStats::default(),
        };

        let result = write_results(
            tmp.path(),
            &training,
            Some(&simulation),
            &make_system(),
            &make_config(),
            &make_output_context(),
        );
        assert!(
            result.is_ok(),
            "write_results must return Ok(()) on success"
        );
    }

    #[test]
    fn write_results_writes_no_success_marker() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);

        write_results(
            tmp.path(),
            &training,
            Some(&make_simulation_output()),
            &make_system(),
            &make_config(),
            &make_output_context(),
        )
        .expect("write_results must succeed");

        assert!(!tmp.path().join("training/_SUCCESS").exists());
        assert!(!tmp.path().join("simulation/_SUCCESS").exists());
    }

    #[test]
    fn write_results_creates_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);

        write_results(
            tmp.path(),
            &training,
            None,
            &make_system(),
            &make_config(),
            &make_output_context(),
        )
        .expect("write_results must succeed");

        let path = tmp.path().join("training/metadata.json");
        assert!(path.is_file(), "training/metadata.json must exist");

        let content = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value =
            serde_json::from_str(&content).expect("metadata.json must contain valid JSON");

        assert_eq!(value["hostname"].as_str(), Some("test-host"));
        assert_eq!(value["solver"].as_str(), Some("highs"));
        assert!(value["started_at"].is_string());
        assert!(value["completed_at"].is_string());
    }

    #[test]
    fn write_results_metadata_row_pool_reports_loaded_boundary_cuts() {
        let tmp = tempfile::tempdir().unwrap();
        let mut training = make_training_output(0);
        training.cut_stats.total_generated = 10_009;
        training.cut_stats.total_active = 10_009;
        training.cut_stats.total_loaded = 10_000;

        write_results(
            tmp.path(),
            &training,
            None,
            &make_system(),
            &make_config(),
            &make_output_context(),
        )
        .expect("write_results must succeed");

        let path = tmp.path().join("training/metadata.json");
        let content = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value =
            serde_json::from_str(&content).expect("metadata.json must contain valid JSON");

        assert_eq!(value["row_pool"]["total_generated"].as_u64(), Some(10_009));
        assert_eq!(value["row_pool"]["total_loaded"].as_u64(), Some(10_000));
    }

    #[test]
    fn write_results_creates_convergence_parquet() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(3);

        write_results(
            tmp.path(),
            &training,
            None,
            &make_system(),
            &make_config(),
            &make_output_context(),
        )
        .expect("write_results must succeed");

        assert!(
            tmp.path().join("training/convergence.parquet").is_file(),
            "training/convergence.parquet must exist"
        );
    }

    #[test]
    fn write_results_convergence_parquet_row_count() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(3);

        write_results(
            tmp.path(),
            &training,
            None,
            &make_system(),
            &make_config(),
            &make_output_context(),
        )
        .expect("write_results must succeed");

        let path = tmp.path().join("training/convergence.parquet");
        let file = std::fs::File::open(&path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();

        let total_rows: usize = reader
            .map(|b| b.expect("batch must be Ok").num_rows())
            .sum();
        assert_eq!(total_rows, 3, "convergence.parquet must have 3 rows");
    }

    #[test]
    fn write_results_empty_training_convergence_parquet_correct_schema() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);

        write_results(
            tmp.path(),
            &training,
            None,
            &make_system(),
            &make_config(),
            &make_output_context(),
        )
        .expect("write_results must succeed");

        let path = tmp.path().join("training/convergence.parquet");
        assert!(path.is_file(), "training/convergence.parquet must exist");

        let file = std::fs::File::open(&path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let schema = builder.schema().clone();
        let reader = builder.build().unwrap();

        let total_rows: usize = reader
            .map(|b| b.expect("batch must be Ok").num_rows())
            .sum();
        assert_eq!(total_rows, 0, "empty training must produce 0 rows");
        assert_eq!(
            schema.fields().len(),
            15,
            "convergence schema must have 15 columns"
        );
    }

    #[test]
    fn write_results_simulation_metadata_scenarios_total() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(3);
        let simulation = SimulationOutput {
            n_scenarios: 10,
            completed: 10,
            failed: 0,
            total_time_ms: 0,
            cost: None,
            solve_stats: MetadataSimulationSolveStats::default(),
        };

        write_results(
            tmp.path(),
            &training,
            Some(&simulation),
            &make_system(),
            &make_config(),
            &make_output_context(),
        )
        .expect("write_results must succeed");

        let path = tmp.path().join("simulation/metadata.json");
        assert!(path.is_file(), "simulation/metadata.json must exist");

        let content = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(
            value["scenarios"]["total"].as_u64(),
            Some(10),
            "$.scenarios.total must equal 10"
        );
    }

    #[test]
    fn write_results_creates_dictionaries() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);

        write_results(
            tmp.path(),
            &training,
            None,
            &make_system(),
            &make_config(),
            &make_output_context(),
        )
        .expect("write_results must succeed");

        assert!(
            tmp.path()
                .join("training/dictionaries/codes.json")
                .is_file(),
            "training/dictionaries/codes.json must exist"
        );
    }

    #[test]
    fn write_results_codes_json_contains_operative_state() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);

        write_results(
            tmp.path(),
            &training,
            None,
            &make_system(),
            &make_config(),
            &make_output_context(),
        )
        .expect("write_results must succeed");

        let path = tmp.path().join("training/dictionaries/codes.json");
        let content = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&content).unwrap();

        assert!(
            value["operative_state"].is_object(),
            "codes.json must contain an operative_state object"
        );
        assert_eq!(
            value["operative_state"]["2"].as_str(),
            Some("operating"),
            r#"codes.json operative_state["2"] must equal "operating""#
        );
    }

    #[test]
    fn write_training_results_produces_complete_output() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(3);

        write_training_results(
            tmp.path(),
            &training,
            &make_system(),
            &make_config(),
            &make_output_context(),
        )
        .expect("write_training_results must succeed");

        assert!(tmp.path().join("training").is_dir());
        assert!(tmp.path().join("training/dictionaries").is_dir());
        assert!(tmp.path().join("training/timing").is_dir());
        assert!(tmp.path().join("training/metadata.json").is_file());
        assert!(!tmp.path().join("training/_SUCCESS").exists());
        assert!(
            tmp.path().join("simulation").is_dir(),
            "simulation/ directory must be created by write_training_results"
        );
    }

    #[test]
    fn write_simulation_results_produces_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();
        let sim = make_simulation_output();

        write_simulation_results(tmp.path(), &sim, &make_output_context())
            .expect("write_simulation_results must succeed");

        assert!(tmp.path().join("simulation/metadata.json").is_file());
        assert!(!tmp.path().join("simulation/_SUCCESS").exists());
    }

    #[test]
    fn training_metadata_records_the_phase_status() {
        use crate::output::manifest::read_training_metadata;

        let tmp = tempfile::tempdir().unwrap();
        let mut training = make_training_output(2);
        training.status = RunStatus::Partial;

        write_training_results(
            tmp.path(),
            &training,
            &make_system(),
            &make_config(),
            &make_output_context(),
        )
        .expect("write_training_results must succeed");

        let metadata = read_training_metadata(&tmp.path().join("training/metadata.json"))
            .expect("read_training_metadata must succeed");
        assert_eq!(metadata.status, RunStatus::Partial);
    }

    #[test]
    fn simulation_metadata_stays_complete_with_failed_scenarios() {
        use crate::output::manifest::read_simulation_metadata;

        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();
        let mut sim = make_simulation_output();
        sim.completed = 7;
        sim.failed = 3;

        write_simulation_results(tmp.path(), &sim, &make_output_context())
            .expect("write_simulation_results must succeed");

        let metadata = read_simulation_metadata(&tmp.path().join("simulation/metadata.json"))
            .expect("read_simulation_metadata must succeed");
        assert_eq!(metadata.scenarios.failed, 3);
        assert_eq!(metadata.status, RunStatus::Complete);
    }

    #[test]
    fn skipped_simulation_metadata_is_partial_with_zero_completed_scenarios() {
        use crate::output::manifest::read_simulation_metadata;

        let tmp = tempfile::tempdir().unwrap();
        assert!(!tmp.path().join("simulation").exists());

        write_skipped_simulation_results(tmp.path(), 100, &make_output_context())
            .expect("write_skipped_simulation_results must succeed");

        let metadata = read_simulation_metadata(&tmp.path().join("simulation/metadata.json"))
            .expect("read_simulation_metadata must succeed");
        assert_eq!(metadata.status, RunStatus::Partial);
        assert_eq!(metadata.scenarios.total, 100);
        assert_eq!(metadata.scenarios.completed, 0);
        assert_eq!(metadata.scenarios.failed, 0);
        assert!(metadata.cost.is_none());
        assert_eq!(metadata.duration_seconds, 0.0);
        assert!(!tmp.path().join("simulation/_SUCCESS").exists());
    }

    #[test]
    fn training_results_writer_leaves_no_success_marker() {
        let tmp = tempfile::tempdir().unwrap();

        write_training_results(
            tmp.path(),
            &make_training_output(2),
            &make_system(),
            &make_config(),
            &make_output_context(),
        )
        .expect("write_training_results must succeed");

        assert!(
            !tmp.path().join("training/_SUCCESS").exists(),
            "training/_SUCCESS belongs to the caller's phase-writer, not write_training_results"
        );
    }

    #[test]
    fn simulation_results_writer_leaves_no_success_marker() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        write_simulation_results(
            tmp.path(),
            &make_simulation_output(),
            &make_output_context(),
        )
        .expect("write_simulation_results must succeed");

        assert!(
            !tmp.path().join("simulation/_SUCCESS").exists(),
            "simulation/_SUCCESS belongs to the caller's phase-writer, not write_simulation_results"
        );
    }

    #[test]
    fn success_marker_writer_creates_an_empty_marker() {
        let tmp = tempfile::tempdir().unwrap();

        write_success_marker(tmp.path()).expect("write_success_marker must succeed");

        let marker = std::fs::metadata(tmp.path().join("_SUCCESS")).expect("_SUCCESS must exist");
        assert!(marker.is_file());
        assert_eq!(marker.len(), 0);
    }

    #[test]
    fn success_marker_writer_fails_without_creating_a_missing_phase_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let phase_dir = tmp.path().join("training");

        let err = write_success_marker(&phase_dir).expect_err("a missing phase dir must fail");

        assert!(
            matches!(&err, OutputError::IoError { path, .. } if path.ends_with("_SUCCESS")),
            "expected an IoError on the marker path, got {err:?}"
        );
        assert!(!phase_dir.exists(), "the phase dir must not be created");
    }

    #[test]
    fn success_marker_remover_deletes_an_existing_marker() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("_SUCCESS"), b"").unwrap();

        remove_success_marker(tmp.path()).expect("remove_success_marker must succeed");

        assert!(!tmp.path().join("_SUCCESS").exists());
    }

    #[test]
    fn success_marker_remover_accepts_a_missing_marker() {
        let tmp = tempfile::tempdir().unwrap();

        remove_success_marker(tmp.path()).expect("a phase dir without a marker must succeed");
        remove_success_marker(&tmp.path().join("training"))
            .expect("a missing phase dir must succeed");
    }

    #[test]
    fn success_marker_remover_reports_a_marker_it_cannot_remove() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("_SUCCESS")).unwrap();

        let err = remove_success_marker(tmp.path()).expect_err("a directory marker must fail");

        assert!(
            matches!(&err, OutputError::IoError { path, .. } if path.ends_with("_SUCCESS")),
            "expected an IoError on the marker path, got {err:?}"
        );
    }

    #[test]
    fn split_functions_match_write_results_output() {
        let tmp_combined = tempfile::tempdir().unwrap();
        let tmp_split = tempfile::tempdir().unwrap();
        let training = make_training_output(2);
        let sim = make_simulation_output();
        let ctx = make_output_context();

        write_results(
            tmp_combined.path(),
            &training,
            Some(&sim),
            &make_system(),
            &make_config(),
            &ctx,
        )
        .expect("write_results must succeed");

        write_training_results(
            tmp_split.path(),
            &training,
            &make_system(),
            &make_config(),
            &ctx,
        )
        .expect("write_training_results must succeed");
        write_simulation_results(tmp_split.path(), &sim, &ctx)
            .expect("write_simulation_results must succeed");

        let combined_metadata = tmp_combined.path().join("training/metadata.json").is_file();
        let split_metadata = tmp_split.path().join("training/metadata.json").is_file();
        assert_eq!(combined_metadata, split_metadata);

        let combined_sim_metadata = tmp_combined
            .path()
            .join("simulation/metadata.json")
            .is_file();
        let split_sim_metadata = tmp_split.path().join("simulation/metadata.json").is_file();
        assert_eq!(combined_sim_metadata, split_sim_metadata);
    }

    #[test]
    fn extract_max_iterations_from_config() {
        let config = make_config();
        assert_eq!(extract_max_iterations(&config), Some(10));
    }

    #[test]
    fn training_metadata_has_max_iterations() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);

        write_training_results(
            tmp.path(),
            &training,
            &make_system(),
            &make_config(),
            &make_output_context(),
        )
        .expect("write_training_results must succeed");

        let path = tmp.path().join("training/metadata.json");
        let content = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&content).unwrap();

        assert_eq!(
            value["configuration"]["max_iterations"].as_u64(),
            Some(10),
            "configuration.max_iterations must be extracted from stopping rules"
        );
    }

    fn read_convergence_kind_and_std_nulls(dir: &std::path::Path) -> (String, usize) {
        use arrow::array::{Array, Float64Array, StringArray};
        let path = dir.join("training/convergence.parquet");
        let batch = read_first_batch(&path);
        let kind = batch
            .column_by_name("upper_bound_kind")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0)
            .to_string();
        let std_nulls = batch
            .column_by_name("upper_bound_std")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .null_count();
        (kind, std_nulls)
    }

    #[test]
    fn exact_bound_writes_exact_kind_and_null_std_in_both_artifacts() {
        use crate::output::manifest::read_training_metadata;
        let tmp = tempfile::tempdir().unwrap();
        let mut training = make_training_output(3);
        training.final_upper_bound_kind = "exact".to_string();
        training.final_upper_bound_std = None;

        write_training_results(
            tmp.path(),
            &training,
            &make_system(),
            &make_config(),
            &make_output_context(),
        )
        .expect("write must succeed");

        let (kind, std_nulls) = read_convergence_kind_and_std_nulls(tmp.path());
        assert_eq!(kind, "exact", "convergence upper_bound_kind must be exact");
        assert_eq!(
            std_nulls, 3,
            "every upper_bound_std must be NULL under exact"
        );

        let metadata = read_training_metadata(&tmp.path().join("training/metadata.json")).unwrap();
        assert_eq!(metadata.bounds.final_upper_bound_kind, "exact");
        assert_eq!(metadata.bounds.final_upper_bound_std, None);
    }

    #[test]
    fn statistical_bound_writes_statistical_kind_and_populated_std_in_both_artifacts() {
        use crate::output::manifest::read_training_metadata;
        let tmp = tempfile::tempdir().unwrap();
        let mut training = make_training_output(3);
        training.final_upper_bound_kind = "statistical".to_string();
        training.final_upper_bound_std = Some(0.5);

        write_training_results(
            tmp.path(),
            &training,
            &make_system(),
            &make_config(),
            &make_output_context(),
        )
        .expect("write must succeed");

        let (kind, std_nulls) = read_convergence_kind_and_std_nulls(tmp.path());
        assert_eq!(kind, "statistical");
        assert_eq!(
            std_nulls, 0,
            "upper_bound_std must be populated under statistical"
        );

        let metadata = read_training_metadata(&tmp.path().join("training/metadata.json")).unwrap();
        assert_eq!(metadata.bounds.final_upper_bound_kind, "statistical");
        assert_eq!(metadata.bounds.final_upper_bound_std, Some(0.5));
    }

    #[test]
    fn training_results_persist_bounds_and_solve_stats() {
        use crate::output::manifest::read_training_metadata;

        let tmp = tempfile::tempdir().unwrap();
        let mut training = make_training_output(2);
        training.final_lower_bound = 48_500.0;
        training.final_upper_bound = Some(49_000.0);
        training.final_upper_bound_std = Some(250.0);
        training.training_solve_stats = MetadataTrainingSolveStats {
            total_lp_solves: Some(120),
            first_try: Some(110),
            retried: Some(10),
            failed: Some(0),
            forward_solve_seconds: Some(3.0),
            backward_solve_seconds: Some(5.0),
            parallelism: Some(4),
        };

        write_training_results(
            tmp.path(),
            &training,
            &make_system(),
            &make_config(),
            &make_output_context(),
        )
        .expect("write_training_results must succeed");

        let metadata = read_training_metadata(&tmp.path().join("training/metadata.json"))
            .expect("read_training_metadata must succeed");

        assert_eq!(metadata.bounds.final_lower_bound, 48_500.0);
        assert_eq!(metadata.bounds.final_upper_bound, Some(49_000.0));
        assert_eq!(metadata.bounds.final_upper_bound_std, Some(250.0));
        assert_eq!(metadata.solve_stats.total_lp_solves, Some(120));
        assert_eq!(metadata.solve_stats.forward_solve_seconds, Some(3.0));
        assert_eq!(metadata.solve_stats.backward_solve_seconds, Some(5.0));
        assert_eq!(metadata.solve_stats.parallelism, Some(4));
    }

    #[test]
    fn simulation_results_persist_cost_and_solve_stats() {
        use crate::output::manifest::read_simulation_metadata;

        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();
        let mut sim = make_simulation_output();
        sim.cost = Some(crate::MetadataCost {
            mean_cost: 12_345.6,
            std_cost: 678.9,
        });
        sim.solve_stats = MetadataSimulationSolveStats {
            total_lp_solves: Some(200),
            first_try: Some(190),
            retried: Some(9),
            failed: Some(1),
            solve_seconds: Some(7.5),
            parallelism: Some(8),
        };

        write_simulation_results(tmp.path(), &sim, &make_output_context())
            .expect("write_simulation_results must succeed");

        let metadata = read_simulation_metadata(&tmp.path().join("simulation/metadata.json"))
            .expect("read_simulation_metadata must succeed");

        let cost = metadata.cost.expect("cost must be persisted");
        assert_eq!(cost.mean_cost, 12_345.6);
        assert_eq!(cost.std_cost, 678.9);
        assert_eq!(metadata.solve_stats.total_lp_solves, Some(200));
        assert_eq!(metadata.solve_stats.solve_seconds, Some(7.5));
        assert_eq!(metadata.solve_stats.parallelism, Some(8));
    }
}
