//! Integration tests for the `cobre summary` subcommand.
//!
//! The fixture is written with the public `cobre_io` writers and `serde_json`
//! (the two optional sidecars), so the JSON shapes track the real schemas without
//! a full live run.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::*;
use cobre_io::{
    DistributionInfo, HostLayout, MetadataBounds, MetadataConfiguration, MetadataConvergence,
    MetadataCost, MetadataIterations, MetadataProblemDimensions, MetadataRowPool,
    MetadataScenarios, MetadataSimulationSolveStats, MetadataTrainingSolveStats, RankAffinity,
    SimulationMetadata, TrainingMetadata, read_training_metadata, write_simulation_metadata,
    write_training_metadata,
};
use predicates::prelude::*;
use serde_json::json;

fn cobre() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

fn local_distribution() -> DistributionInfo {
    DistributionInfo {
        backend: "local".to_string(),
        world_size: 1,
        ranks_participated: 1,
        num_nodes: 1,
        threads_per_rank: 1,
        mpi_library: None,
        mpi_standard: None,
        thread_level: None,
        slurm_job_id: None,
        hosts: vec![HostLayout {
            hostname: "fixture-host".to_string(),
            ranks: vec![0],
        }],
        rank_affinity: Vec::new(),
    }
}

fn write_training_fixture(dir: &Path) {
    let metadata = TrainingMetadata {
        cobre_version: "0.0.0-test".to_string(),
        hostname: "fixture-host".to_string(),
        solver: "highs".to_string(),
        solver_version: Some("1.7.0".to_string()),
        started_at: "2026-01-17T08:00:00Z".to_string(),
        completed_at: "2026-01-17T12:30:00Z".to_string(),
        duration_seconds: 16_200.0,
        status: "complete".to_string(),
        configuration: MetadataConfiguration {
            seed: Some(42),
            max_iterations: Some(100),
            forward_passes: Some(192),
            stopping_mode: "any".to_string(),
            policy_mode: "fresh".to_string(),
        },
        problem_dimensions: MetadataProblemDimensions {
            num_stages: 12,
            num_hydros: 1,
            num_thermals: 1,
            num_buses: 1,
            num_lines: 0,
        },
        iterations: MetadataIterations {
            completed: 10,
            converged_at: Some(10),
        },
        convergence: MetadataConvergence {
            achieved: true,
            final_gap_percent: Some(0.45),
            termination_reason: "gap_tolerance".to_string(),
        },
        row_pool: MetadataRowPool {
            total_generated: 100,
            total_active: 90,
            peak_active: 100,
            cuts_active: 90,
            rows_in_lp_total: 0,
            rows_in_lp_solve_count: 0,
            rows_in_lp_max: 0,
        },
        bounds: MetadataBounds {
            final_lower_bound: 48_500.0,
            final_upper_bound: Some(49_000.0),
            final_upper_bound_std: Some(250.0),
        },
        solve_stats: MetadataTrainingSolveStats {
            total_lp_solves: Some(840),
            first_try: Some(800),
            retried: Some(38),
            failed: Some(2),
            forward_solve_seconds: Some(1.5),
            backward_solve_seconds: Some(4.5),
            parallelism: Some(1),
            forward_phase_wall_seconds: Some(2.0),
            backward_phase_wall_seconds: Some(5.0),
            forward_wait_seconds: Some(0.1),
            backward_wait_seconds: Some(0.2),
            serial_lower_bound_seconds: Some(0.3),
            serial_row_selection_seconds: Some(0.4),
            serial_row_sync_seconds: Some(0.5),
            serial_allreduce_seconds: Some(0.6),
            serial_scheduling_seconds: Some(0.7),
        },
        setup: None,
        production_fit_deviation: None,
        distribution: local_distribution(),
    };

    std::fs::create_dir_all(dir.join("training")).unwrap();
    write_training_metadata(&dir.join("training/metadata.json"), &metadata).unwrap();
}

fn write_simulation_fixture(dir: &Path) {
    let metadata = SimulationMetadata {
        cobre_version: "0.0.0-test".to_string(),
        hostname: "fixture-host".to_string(),
        solver: "highs".to_string(),
        solver_version: Some("1.7.0".to_string()),
        started_at: "2026-01-17T12:30:00Z".to_string(),
        completed_at: "2026-01-17T12:30:12Z".to_string(),
        duration_seconds: 12.5,
        status: "complete".to_string(),
        scenarios: MetadataScenarios {
            total: 192,
            completed: 192,
            failed: 0,
        },
        cost: Some(MetadataCost {
            mean_cost: 1.0e6,
            std_cost: 2.0e4,
            cvar: 1.2e6,
            cvar_alpha: 0.95,
        }),
        solve_stats: MetadataSimulationSolveStats {
            total_lp_solves: Some(5_000),
            first_try: Some(4_900),
            retried: Some(90),
            failed: Some(10),
            solve_seconds: Some(3.0),
            parallelism: Some(1),
        },
        distribution: local_distribution(),
    };

    std::fs::create_dir_all(dir.join("simulation")).unwrap();
    write_simulation_metadata(&dir.join("simulation/metadata.json"), &metadata).unwrap();
}

fn write_hydro_models_fixture(dir: &Path) {
    let summary = json!({
        "n_constant": 1,
        "n_fpha": 0,
        "total_planes": 0,
        "fpha_details": [],
        "n_evaporation": 0,
        "n_no_evaporation": 1,
        "n_user_supplied_ref": 0,
        "n_default_midpoint_ref": 0
    });
    std::fs::create_dir_all(dir.join("training")).unwrap();
    std::fs::write(
        dir.join("training/hydro_models.json"),
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .unwrap();
}

fn write_provenance_fixture(dir: &Path) {
    let report = json!({
        "inflow": {
            "estimation_path": "full_estimation",
            "seasonal_stats_source": "estimated",
            "ar_coefficients_source": "estimated",
            "correlation_source": "estimated",
            "opening_tree_source": "estimated",
            "n_hydros": 1,
            "ar_method": "AIC",
            "ar_max_order": 2,
            "white_noise_fallbacks": []
        },
        "hydro_production": {
            "n_fpha_computed_from_geometry": 0,
            "n_fpha_precomputed_hyperplanes": 0,
            "n_evaporation_ref_user_supplied": 0,
            "n_evaporation_ref_default_midpoint": 0
        }
    });
    std::fs::create_dir_all(dir.join("training")).unwrap();
    std::fs::write(
        dir.join("training/model_provenance.json"),
        serde_json::to_string_pretty(&report).unwrap(),
    )
    .unwrap();
}

fn assert_ordered(haystack: &str, needle_a: &str, needle_b: &str) {
    let pos_a = haystack
        .find(needle_a)
        .unwrap_or_else(|| panic!("expected to find {needle_a:?} in stderr"));
    let pos_b = haystack
        .find(needle_b)
        .unwrap_or_else(|| panic!("expected to find {needle_b:?} in stderr"));
    assert!(
        pos_a < pos_b,
        "expected {needle_a:?} (at {pos_a}) before {needle_b:?} (at {pos_b})"
    );
}

#[test]
fn summary_prints_all_five_sections_in_live_order() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();

    write_training_fixture(path);
    write_hydro_models_fixture(path);
    write_provenance_fixture(path);
    write_simulation_fixture(path);

    let output = cobre()
        .args(["summary", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .clone();

    let stderr = String::from_utf8(output.stderr).unwrap();

    assert_ordered(&stderr, "Execution", "Hydro models");
    assert_ordered(&stderr, "Hydro models", "Model provenance");
    assert_ordered(&stderr, "Model provenance", "Training complete in");
    assert_ordered(&stderr, "Training complete in", "Simulation complete");
}

#[test]
fn summary_skips_model_provenance_when_sidecar_absent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();

    write_training_fixture(path);
    write_hydro_models_fixture(path);
    write_simulation_fixture(path);

    let output = cobre()
        .args(["summary", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .clone();

    let stderr = String::from_utf8(output.stderr).unwrap();

    assert!(!stderr.contains("Model provenance"));
    assert!(stderr.contains("Execution"));
    assert!(stderr.contains("Hydro models"));
    assert!(stderr.contains("Training complete in"));
    assert!(stderr.contains("Simulation complete"));
}

#[test]
fn summary_skips_hydro_models_when_sidecar_absent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();

    write_training_fixture(path);
    write_provenance_fixture(path);

    let output = cobre()
        .args(["summary", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .clone();

    let stderr = String::from_utf8(output.stderr).unwrap();

    assert!(!stderr.contains("Hydro models"));
    assert!(stderr.contains("Execution"));
    assert!(stderr.contains("Model provenance"));
    assert!(stderr.contains("Training complete in"));
}

#[test]
fn summary_only_training_required_minimal_dir() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();

    write_training_fixture(path);

    let output = cobre()
        .args(["summary", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .clone();

    let stderr = String::from_utf8(output.stderr).unwrap();

    assert!(stderr.contains("Execution"));
    assert!(stderr.contains("Training complete in"));
    assert!(!stderr.contains("Hydro models"));
    assert!(!stderr.contains("Model provenance"));
    assert!(!stderr.contains("Simulation complete"));
}

#[test]
fn summary_prints_persisted_cpu_and_memory_placement() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();
    write_training_fixture(path);
    let metadata_path = path.join("training/metadata.json");
    let mut metadata = read_training_metadata(&metadata_path).unwrap();
    metadata.distribution.rank_affinity.push(RankAffinity {
        rank: 0,
        policy: "numa".to_string(),
        online_processing_units: Some(96),
        visible_processing_units: Some(48),
        physical_cores: Some(48),
        numa_nodes: Some(4),
        visible_cpus: vec![0, 2, 4, 6],
        memory_policy: Some("bind".to_string()),
        memory_policy_nodes: vec![0, 1],
        allowed_memory_nodes: vec![0, 1, 2, 3],
        memory_discovery_error: None,
        worker_cpus: vec![0, 2],
        discovery_error: None,
    });
    write_training_metadata(&metadata_path, &metadata).unwrap();

    let output = cobre()
        .args(["summary", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .clone();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("48 physical cores, 4 NUMA node(s), 48/96 logical CPUs visible"));
    assert!(stderr.contains("Affinity:  numa (2 worker bindings)"));
    assert!(stderr.contains("Memory:    bind on nodes 0–1, allowed nodes 0–3"));
}

#[test]
fn summary_malformed_provenance_sidecar_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();

    write_training_fixture(path);
    std::fs::create_dir_all(path.join("training")).unwrap();
    // A corrupt sidecar must error, not be skipped the way an absent one is.
    std::fs::write(
        path.join("training/model_provenance.json"),
        "{ this is not valid json",
    )
    .unwrap();

    cobre()
        .args(["summary", path.to_str().unwrap()])
        .assert()
        .failure()
        .code(predicate::eq(4));
}
