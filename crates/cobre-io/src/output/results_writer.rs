//! Aggregate output writers that combine training and simulation artifacts.
//!
//! [`write_training_results`] creates the full `training/` directory tree
//! (dictionaries, convergence Parquet, manifest, metadata, `_SUCCESS` marker)
//! and an empty `simulation/` directory.
//!
//! [`write_simulation_results`] writes the `simulation/` manifest and
//! `_SUCCESS` marker.
//!
//! [`write_results`] is a convenience wrapper that calls both in sequence.

use std::path::Path;

use cobre_core::System;

use super::dictionary::write_dictionaries;
use super::error::OutputError;
use super::manifest::{
    ManifestConvergence, ManifestCuts, ManifestIterations, ManifestMpiInfo, ManifestScenarios,
    MetadataConfigSnapshot, MetadataEnvironment, MetadataProblemDimensions, MetadataRunInfo,
    SimulationManifest, TrainingManifest, TrainingMetadata, write_metadata,
    write_simulation_manifest, write_training_manifest,
};
use super::parquet_config::ParquetWriterConfig;
use super::training_writer::TrainingParquetWriter;
use super::{SimulationOutput, TrainingOutput};
use crate::Config;

/// Write all training artifacts to the output directory.
///
/// Creates the `training/` subdirectory structure (dictionaries, convergence
/// Parquet, timing, manifest, metadata) and writes the `training/_SUCCESS`
/// marker on completion. Also creates an empty `simulation/` directory so
/// downstream code can unconditionally write into it.
///
/// This function is one half of the split from [`write_results`]; it can be
/// called independently to persist training outputs before simulation starts.
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
    let training_manifest = TrainingManifest {
        version: "2.0.0".to_string(),
        status: "complete".to_string(),
        started_at: None,
        completed_at: None,
        iterations: ManifestIterations {
            max_iterations: None,
            completed: training_output.iterations_completed,
            converged_at,
        },
        convergence: ManifestConvergence {
            achieved: training_output.converged,
            final_gap_percent: training_output.final_gap_percent,
            termination_reason: training_output.termination_reason.clone(),
        },
        cuts: ManifestCuts {
            total_generated: training_output.cut_stats.total_generated,
            total_active: training_output.cut_stats.total_active,
            peak_active: training_output.cut_stats.peak_active,
        },
        checksum: None,
        mpi_info: ManifestMpiInfo::default(),
    };
    write_training_manifest(
        &output_dir.join("training/_manifest.json"),
        &training_manifest,
    )?;

    let training_metadata = TrainingMetadata {
        version: "2.0.0".to_string(),
        run_info: MetadataRunInfo {
            run_id: "not-implemented".to_string(),
            started_at: None,
            completed_at: None,
            duration_seconds: Some(training_output.total_time_ms as f64 / 1_000.0),
            cobre_version: env!("CARGO_PKG_VERSION").to_string(),
            solver: None,
            solver_version: None,
            hostname: None,
            user: None,
        },
        configuration_snapshot: MetadataConfigSnapshot {
            seed: config.training.tree_seed,
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
        performance_summary: None,
        data_integrity: None,
        environment: MetadataEnvironment {
            mpi_implementation: None,
            mpi_version: None,
            num_ranks: None,
            cpus_per_rank: None,
            memory_per_rank_gb: None,
        },
    };
    write_metadata(
        &output_dir.join("training/metadata.json"),
        &training_metadata,
    )?;

    std::fs::write(output_dir.join("training/_SUCCESS"), b"")
        .map_err(|e| OutputError::io(output_dir.join("training/_SUCCESS"), e))?;

    Ok(())
}

/// Write simulation artifacts to the output directory.
///
/// Writes `simulation/_manifest.json` and the `simulation/_SUCCESS` marker.
/// The `simulation/` directory must already exist (created by
/// [`write_training_results`]).
///
/// # Errors
///
/// Returns [`OutputError`] if manifest serialization or file I/O fails.
#[allow(clippy::cast_precision_loss)]
pub fn write_simulation_results(
    output_dir: &Path,
    simulation_output: &SimulationOutput,
) -> Result<(), OutputError> {
    let sim_manifest = SimulationManifest {
        version: "2.0.0".to_string(),
        status: "complete".to_string(),
        started_at: None,
        completed_at: None,
        duration_seconds: Some(simulation_output.total_time_ms as f64 / 1_000.0),
        scenarios: ManifestScenarios {
            total: simulation_output.n_scenarios,
            completed: simulation_output.completed,
            failed: simulation_output.failed,
        },
        partitions_written: simulation_output.partitions_written.clone(),
        checksum: None,
        mpi_info: ManifestMpiInfo::default(),
    };
    write_simulation_manifest(&output_dir.join("simulation/_manifest.json"), &sim_manifest)?;

    std::fs::write(output_dir.join("simulation/_SUCCESS"), b"")
        .map_err(|e| OutputError::io(output_dir.join("simulation/_SUCCESS"), e))?;

    Ok(())
}

/// Write all output artifacts (training + simulation) to the output directory.
///
/// Convenience wrapper that calls [`write_training_results`] followed by
/// [`write_simulation_results`] (when simulation output is present).
/// Retained for backward compatibility.
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
) -> Result<(), OutputError> {
    write_training_results(output_dir, training_output, system, config)?;
    if let Some(sim) = simulation_output {
        write_simulation_results(output_dir, sim)?;
    }
    Ok(())
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
            time_forward_solve_ms: 100,
            time_forward_sample_ms: 0,
            time_backward_solve_ms: 200,
            time_backward_cut_ms: 0,
            time_cut_selection_ms: 0,
            time_mpi_allreduce_ms: 0,
            time_mpi_broadcast_ms: 0,
            time_io_write_ms: 0,
            time_state_exchange_ms: 0,
            time_cut_batch_build_ms: 0,
            time_rayon_overhead_ms: 0,
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
            SimulationSamplingConfig, StoppingRuleConfig, TrainingConfig, TrainingSolverConfig,
            UpperBoundEvaluationConfig,
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
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig {
                path: "./policy".to_string(),
                mode: PolicyMode::Fresh,
                validate_compatibility: true,
                checkpointing: CheckpointingConfig::default(),
            },
            simulation: SimulationConfig {
                enabled: false,
                num_scenarios: 0,
                policy_type: "outer".to_string(),
                output_path: None,
                output_mode: None,
                io_channel_capacity: 64,
                sampling_scheme: SimulationSamplingConfig::default(),
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

    #[test]
    fn write_results_creates_training_directories() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);

        write_results(tmp.path(), &training, None, &make_system(), &make_config())
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

        write_results(tmp.path(), &training, None, &make_system(), &make_config())
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

        write_results(tmp.path(), &training, None, &make_system(), &make_config())
            .expect("write_results must succeed");

        assert!(
            tmp.path().join("training/_SUCCESS").is_file(),
            "training/_SUCCESS must exist after write_results"
        );
    }

    #[test]
    fn write_results_creates_training_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);

        write_results(tmp.path(), &training, None, &make_system(), &make_config())
            .expect("write_results must succeed");

        let path = tmp.path().join("training/_manifest.json");
        assert!(path.is_file(), "training/_manifest.json must exist");

        let content = std::fs::read_to_string(&path).unwrap();
        let _parsed: serde_json::Value =
            serde_json::from_str(&content).expect("_manifest.json must contain valid JSON");
    }

    #[test]
    fn write_results_creates_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(0);

        write_results(tmp.path(), &training, None, &make_system(), &make_config())
            .expect("write_results must succeed");

        let path = tmp.path().join("training/metadata.json");
        assert!(path.is_file(), "training/metadata.json must exist");

        let content = std::fs::read_to_string(&path).unwrap();
        let _parsed: serde_json::Value =
            serde_json::from_str(&content).expect("metadata.json must contain valid JSON");
    }

    #[test]
    fn write_results_creates_convergence_parquet() {
        let tmp = tempfile::tempdir().unwrap();
        let training = make_training_output(3);

        write_results(tmp.path(), &training, None, &make_system(), &make_config())
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

        write_results(tmp.path(), &training, None, &make_system(), &make_config())
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

        write_results(tmp.path(), &training, None, &make_system(), &make_config())
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
        write_results(tmp2.path(), &training, None, &make_system(), &make_config())
            .expect("write_results must succeed");

        assert!(
            !tmp2.path().join("simulation/_SUCCESS").exists(),
            "simulation/_SUCCESS must NOT exist when simulation_output is None"
        );
    }

    #[test]
    fn write_results_simulation_manifest_scenarios_total() {
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
        )
        .expect("write_results must succeed");

        let path = tmp.path().join("simulation/_manifest.json");
        assert!(path.is_file(), "simulation/_manifest.json must exist");

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

        write_results(tmp.path(), &training, None, &make_system(), &make_config())
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

        write_results(tmp.path(), &training, None, &make_system(), &make_config())
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

        write_training_results(tmp.path(), &training, &make_system(), &make_config())
            .expect("write_training_results must succeed");

        assert!(tmp.path().join("training").is_dir());
        assert!(tmp.path().join("training/dictionaries").is_dir());
        assert!(tmp.path().join("training/timing").is_dir());
        assert!(tmp.path().join("training/_manifest.json").is_file());
        assert!(tmp.path().join("training/metadata.json").is_file());
        assert!(tmp.path().join("training/_SUCCESS").is_file());
        assert!(
            tmp.path().join("simulation").is_dir(),
            "simulation/ directory must be created by write_training_results"
        );
    }

    #[test]
    fn write_simulation_results_produces_manifest_and_success() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();
        let sim = make_simulation_output();

        write_simulation_results(tmp.path(), &sim).expect("write_simulation_results must succeed");

        assert!(tmp.path().join("simulation/_manifest.json").is_file());
        assert!(tmp.path().join("simulation/_SUCCESS").is_file());
    }

    #[test]
    fn split_functions_match_write_results_output() {
        let tmp_combined = tempfile::tempdir().unwrap();
        let tmp_split = tempfile::tempdir().unwrap();
        let training = make_training_output(2);
        let sim = make_simulation_output();

        write_results(
            tmp_combined.path(),
            &training,
            Some(&sim),
            &make_system(),
            &make_config(),
        )
        .expect("write_results must succeed");

        write_training_results(tmp_split.path(), &training, &make_system(), &make_config())
            .expect("write_training_results must succeed");
        write_simulation_results(tmp_split.path(), &sim)
            .expect("write_simulation_results must succeed");

        let combined_training_success = tmp_combined.path().join("training/_SUCCESS").is_file();
        let split_training_success = tmp_split.path().join("training/_SUCCESS").is_file();
        assert_eq!(combined_training_success, split_training_success);

        let combined_sim_success = tmp_combined.path().join("simulation/_SUCCESS").is_file();
        let split_sim_success = tmp_split.path().join("simulation/_SUCCESS").is_file();
        assert_eq!(combined_sim_success, split_sim_success);

        let combined_manifest = tmp_combined
            .path()
            .join("training/_manifest.json")
            .is_file();
        let split_manifest = tmp_split.path().join("training/_manifest.json").is_file();
        assert_eq!(combined_manifest, split_manifest);

        let combined_sim_manifest = tmp_combined
            .path()
            .join("simulation/_manifest.json")
            .is_file();
        let split_sim_manifest = tmp_split.path().join("simulation/_manifest.json").is_file();
        assert_eq!(combined_sim_manifest, split_sim_manifest);
    }
}
