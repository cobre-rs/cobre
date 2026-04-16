//! Aggregate output writers that combine training and simulation artifacts.
//!
//! [`write_training_results`] creates the full `training/` directory tree
//! (dictionaries, convergence Parquet, metadata, `_SUCCESS` marker)
//! and an empty `simulation/` directory.
//!
//! [`write_simulation_results`] writes the `simulation/metadata.json` and
//! `_SUCCESS` marker.
//!
//! [`write_results`] is a convenience wrapper that calls both in sequence.

use std::path::Path;

use cobre_core::System;

use super::dictionary::write_dictionaries;
use super::error::OutputError;
use super::manifest::{
    MetadataConfiguration, MetadataConvergence, MetadataCuts, MetadataIterations,
    MetadataProblemDimensions, MetadataScenarios, OutputContext, SimulationMetadata,
    TrainingMetadata, write_simulation_metadata, write_training_metadata,
};
use super::parquet_config::ParquetWriterConfig;
use super::training_writer::TrainingParquetWriter;
use super::{SimulationOutput, TrainingOutput};
use crate::Config;
use crate::config::StoppingRuleConfig;

/// Write all training artifacts to the output directory.
///
/// Creates the `training/` subdirectory structure (dictionaries, convergence
/// Parquet, timing, metadata) and writes the `training/_SUCCESS` marker on
/// completion. Also creates an empty `simulation/` directory so downstream
/// code can unconditionally write into it.
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
    std::fs::create_dir_all(output_dir.join("training/dictionaries"))
        .map_err(|e| OutputError::io(output_dir.join("training/dictionaries"), e))?;
    std::fs::create_dir_all(output_dir.join("training/timing"))
        .map_err(|e| OutputError::io(output_dir.join("training/timing"), e))?;
    std::fs::create_dir_all(output_dir.join("simulation"))
        .map_err(|e| OutputError::io(output_dir.join("simulation"), e))?;

    write_dictionaries(&output_dir.join("training/dictionaries"), system, config)?;

    let parquet_config = ParquetWriterConfig::default();
    let writer = TrainingParquetWriter::new(output_dir, &parquet_config)?;
    writer.write(training_output)?;

    let converged_at = training_output
        .converged
        .then_some(training_output.iterations_completed);

    let max_iterations = extract_max_iterations(config);

    let metadata = TrainingMetadata {
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
        hostname: ctx.hostname.clone(),
        solver: ctx.solver.clone(),
        solver_version: ctx.solver_version.clone(),
        started_at: ctx.started_at.clone(),
        completed_at: ctx.completed_at.clone(),
        duration_seconds: training_output.total_time_ms as f64 / 1_000.0,
        status: "complete".to_string(),
        configuration: MetadataConfiguration {
            seed: config.training.tree_seed,
            max_iterations,
            forward_passes: config.training.forward_passes,
            stopping_mode: config.training.stopping_mode.clone(),
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
        cuts: MetadataCuts {
            total_generated: training_output.cut_stats.total_generated,
            total_active: training_output.cut_stats.total_active,
            peak_active: training_output.cut_stats.peak_active,
        },
        distribution: ctx.distribution.clone(),
    };
    write_training_metadata(&output_dir.join("training/metadata.json"), &metadata)?;

    std::fs::write(output_dir.join("training/_SUCCESS"), b"")
        .map_err(|e| OutputError::io(output_dir.join("training/_SUCCESS"), e))?;

    Ok(())
}

/// Write simulation artifacts to the output directory.
///
/// Writes `simulation/metadata.json` and the `simulation/_SUCCESS` marker.
/// The `simulation/` directory must already exist (created by
/// [`write_training_results`]).
///
/// # Errors
///
/// Returns [`OutputError`] if metadata serialization or file I/O fails.
#[allow(clippy::cast_precision_loss)]
pub fn write_simulation_results(
    output_dir: &Path,
    simulation_output: &SimulationOutput,
    ctx: &OutputContext,
) -> Result<(), OutputError> {
    let metadata = SimulationMetadata {
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
        hostname: ctx.hostname.clone(),
        solver: ctx.solver.clone(),
        solver_version: ctx.solver_version.clone(),
        started_at: ctx.started_at.clone(),
        completed_at: ctx.completed_at.clone(),
        duration_seconds: simulation_output.total_time_ms as f64 / 1_000.0,
        status: "complete".to_string(),
        scenarios: MetadataScenarios {
            total: simulation_output.n_scenarios,
            completed: simulation_output.completed,
            failed: simulation_output.failed,
        },
        distribution: ctx.distribution.clone(),
    };
    write_simulation_metadata(&output_dir.join("simulation/metadata.json"), &metadata)?;

    std::fs::write(output_dir.join("simulation/_SUCCESS"), b"")
        .map_err(|e| OutputError::io(output_dir.join("simulation/_SUCCESS"), e))?;

    Ok(())
}

/// Write all output artifacts (training + simulation) to the output directory.
///
/// Convenience wrapper that calls [`write_training_results`] followed by
/// [`write_simulation_results`] (when simulation output is present).
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

/// Extract the iteration limit from the stopping rules configuration.
fn extract_max_iterations(config: &Config) -> Option<u32> {
    config
        .training
        .stopping_rules
        .as_ref()?
        .iter()
        .find_map(|r| {
            if let StoppingRuleConfig::IterationLimit { limit } = r {
                Some(*limit)
            } else {
                None
            }
        })
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
    use crate::output::{CutStatistics, IterationRecord, TrainingOutput};

    fn make_iteration_record(iteration: u32) -> IterationRecord {
        IterationRecord {
            iteration,
            lower_bound: 1.0,
            upper_bound_mean: 2.0,
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
            time_bwd_setup_ms: 0,
            time_bwd_load_imbalance_ms: 0,
            time_bwd_scheduling_overhead_ms: 0,
            time_fwd_setup_ms: 0,
            time_fwd_load_imbalance_ms: 0,
            time_fwd_scheduling_overhead_ms: 0,
            time_overhead_ms: 0,
            solve_time_ms: 0.0,
        }
    }

    fn make_training_output(n_records: usize) -> TrainingOutput {
        let records = (1..=n_records as u32).map(make_iteration_record).collect();
        TrainingOutput {
            convergence_records: records,
            final_lower_bound: 99.5,
            final_upper_bound: Some(101.0),
            final_gap_percent: Some(1.51),
            iterations_completed: n_records as u32,
            converged: true,
            termination_reason: "gap tolerance reached".to_string(),
            total_time_ms: 5_000,
            cut_stats: CutStatistics {
                total_generated: 200,
                total_active: 80,
                peak_active: 95,
            },
            cut_selection_records: vec![],
        }
    }

    fn make_system() -> cobre_core::System {
        cobre_core::SystemBuilder::new()
            .build()
            .expect("empty system must be valid")
    }

    fn make_config() -> crate::Config {
        use crate::config::{
            CheckpointingConfig, CutSelectionConfig, EstimationConfig, ExportsConfig,
            InflowNonNegativityConfig, ModelingConfig, PolicyConfig, PolicyMode, SimulationConfig,
            StoppingRuleConfig, TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
        };
        crate::Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig::default(),
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: None,
                forward_passes: Some(4),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 10 }]),
                stopping_mode: "any".to_string(),
                cut_formulation: None,
                forward_pass: None,
                cut_selection: CutSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                scenario_source: None,
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig {
                path: "./policy".to_string(),
                mode: PolicyMode::Fresh,
                validate_compatibility: true,
                checkpointing: CheckpointingConfig::default(),
                boundary: None,
            },
            simulation: SimulationConfig {
                enabled: false,
                num_scenarios: 0,
                policy_type: "outer".to_string(),
                output_path: None,
                output_mode: None,
                io_channel_capacity: 64,
                scenario_source: None,
            },
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    fn make_simulation_output() -> SimulationOutput {
        SimulationOutput {
            n_scenarios: 10,
            completed: 10,
            failed: 0,
            total_time_ms: 1_000,
            partitions_written: vec!["simulation/costs/part-00.parquet".to_string()],
        }
    }

    fn make_output_context() -> OutputContext {
        use super::super::manifest::DistributionInfo;
        OutputContext {
            hostname: "test-host".to_string(),
            solver: "highs".to_string(),
            solver_version: None,
            started_at: "2026-01-17T08:00:00Z".to_string(),
            completed_at: "2026-01-17T12:30:00Z".to_string(),
            distribution: DistributionInfo {
                backend: "local".to_string(),
                world_size: 1,
                ranks_participated: 1,
                num_nodes: 1,
                threads_per_rank: 1,
                mpi_library: None,
                mpi_standard: None,
                thread_level: None,
                slurm_job_id: None,
            },
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
            partitions_written: vec!["simulation/costs/part-00.parquet".to_string()],
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
    fn write_results_creates_success_marker() {
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
            tmp.path().join("training/_SUCCESS").is_file(),
            "training/_SUCCESS must exist after write_results"
        );
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
            13,
            "convergence schema must have 13 columns"
        );

        assert!(
            tmp.path().join("training/_SUCCESS").is_file(),
            "training/_SUCCESS must exist even with 0 records"
        );
    }

    #[test]
    fn write_results_simulation_success_marker_conditional() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);
        let simulation = SimulationOutput {
            n_scenarios: 10,
            completed: 10,
            failed: 0,
            total_time_ms: 0,
            partitions_written: vec![],
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

        assert!(
            tmp.path().join("simulation/_SUCCESS").is_file(),
            "simulation/_SUCCESS must exist when simulation_output is Some"
        );
        assert!(
            tmp.path().join("training/_SUCCESS").is_file(),
            "training/_SUCCESS must exist"
        );

        let tmp2 = tempfile::tempdir().unwrap();
        write_results(
            tmp2.path(),
            &training,
            None,
            &make_system(),
            &make_config(),
            &make_output_context(),
        )
        .expect("write_results must succeed");

        assert!(
            !tmp2.path().join("simulation/_SUCCESS").exists(),
            "simulation/_SUCCESS must NOT exist when simulation_output is None"
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
            partitions_written: vec![],
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
        assert!(tmp.path().join("training/_SUCCESS").is_file());
        assert!(
            tmp.path().join("simulation").is_dir(),
            "simulation/ directory must be created by write_training_results"
        );
    }

    #[test]
    fn write_simulation_results_produces_metadata_and_success() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();
        let sim = make_simulation_output();

        write_simulation_results(tmp.path(), &sim, &make_output_context())
            .expect("write_simulation_results must succeed");

        assert!(tmp.path().join("simulation/metadata.json").is_file());
        assert!(tmp.path().join("simulation/_SUCCESS").is_file());
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

        let combined_training_success = tmp_combined.path().join("training/_SUCCESS").is_file();
        let split_training_success = tmp_split.path().join("training/_SUCCESS").is_file();
        assert_eq!(combined_training_success, split_training_success);

        let combined_sim_success = tmp_combined.path().join("simulation/_SUCCESS").is_file();
        let split_sim_success = tmp_split.path().join("simulation/_SUCCESS").is_file();
        assert_eq!(combined_sim_success, split_sim_success);

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
}
