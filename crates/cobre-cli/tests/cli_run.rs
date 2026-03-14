//! Integration tests for the `cobre run` subcommand.
//!
//! Tests the full execution lifecycle: binary invocation → load → train →
//! simulate → write output → exit code. Fixtures are built programmatically
//! in temporary directories.
//!
//! # Coverage
//!
//! - AC-1: Valid case directory → exit 0, `training/convergence.parquet` and
//!   `training/_manifest.json` exist in the default output directory.
//! - AC-2: `--skip-simulation` → exit 0, no `simulation/_manifest.json`.
//! - AC-3: `--output <custom_dir>` → output written to the specified directory.
//! - AC-4: Invalid case directory (missing required files) → exit 1, stderr
//!   contains a validation error message.
//! - AC-5: Nonexistent case directory path → exit 2, stderr contains an I/O
//!   error message.

#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::*;
use predicates::prelude::*;
use tempfile::TempDir;

fn cobre() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

/// Minimal `config.json` with `forward_passes` and `stopping_rules`.
/// Uses a single iteration limit so tests run fast.
/// `training.seed` is absent, so the CLI will use the default seed (42) and
/// emit a warning to stderr.
const CONFIG_JSON: &str = r#"{
    "training": {
        "forward_passes": 1,
        "stopping_rules": [
            { "type": "iteration_limit", "limit": 2 }
        ]
    }
}"#;

/// Variant of `CONFIG_JSON` with an explicit `training.seed`.
/// The CLI must not emit the "no random seed specified" warning.
const CONFIG_WITH_SEED_JSON: &str = r#"{
    "training": {
        "forward_passes": 1,
        "stopping_rules": [
            { "type": "iteration_limit", "limit": 2 }
        ],
        "seed": 99
    }
}"#;

const PENALTIES_JSON: &str = r#"{
    "bus": {
        "deficit_segments": [
            { "depth_mw": 500.0, "cost": 1000.0 },
            { "depth_mw": null,  "cost": 5000.0 }
        ],
        "excess_cost": 100.0
    },
    "line": { "exchange_cost": 2.0 },
    "hydro": {
        "spillage_cost": 0.01,
        "fpha_turbined_cost": 0.05,
        "diversion_cost": 0.1,
        "storage_violation_below_cost": 10000.0,
        "filling_target_violation_cost": 50000.0,
        "turbined_violation_below_cost": 500.0,
        "outflow_violation_below_cost": 500.0,
        "outflow_violation_above_cost": 500.0,
        "generation_violation_below_cost": 1000.0,
        "evaporation_violation_cost": 5000.0,
        "water_withdrawal_violation_cost": 1000.0
    },
    "non_controllable_source": { "curtailment_cost": 0.005 }
}"#;

const STAGES_JSON: &str = r#"{
    "policy_graph": {
        "type": "finite_horizon",
        "annual_discount_rate": 0.06,
        "transitions": []
    },
    "scenario_source": { "sampling_scheme": "in_sample", "seed": 42 },
    "stages": [
        {
            "id": 0,
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 744.0 }],
            "num_scenarios": 2
        },
        {
            "id": 1,
            "start_date": "2024-02-01",
            "end_date": "2024-03-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 672.0 }],
            "num_scenarios": 2
        }
    ]
}"#;

const INITIAL_CONDITIONS_JSON: &str = r#"{ "storage": [], "filling_storage": [] }"#;
const BUSES_JSON: &str = r#"{ "buses": [{ "id": 1, "name": "BUS_1" }] }"#;
const LINES_JSON: &str = r#"{ "lines": [] }"#;
const HYDROS_JSON: &str = r#"{ "hydros": [] }"#;
const THERMALS_JSON: &str = r#"{ "thermals": [] }"#;

fn write_file(root: &Path, relative: &str, content: &str) {
    let full = root.join(relative);
    if let Some(parent) = full.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&full, content).unwrap();
}

fn make_valid_case(dir: &TempDir) {
    let root = dir.path();
    write_file(root, "config.json", CONFIG_JSON);
    write_file(root, "penalties.json", PENALTIES_JSON);
    write_file(root, "stages.json", STAGES_JSON);
    write_file(root, "initial_conditions.json", INITIAL_CONDITIONS_JSON);
    write_file(root, "system/buses.json", BUSES_JSON);
    write_file(root, "system/lines.json", LINES_JSON);
    write_file(root, "system/hydros.json", HYDROS_JSON);
    write_file(root, "system/thermals.json", THERMALS_JSON);
}

#[test]
fn valid_case_exits_0() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();

    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
            "--quiet",
            "--skip-simulation",
        ])
        .assert()
        .success();
}

#[test]
fn valid_case_creates_training_manifest() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();

    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
            "--quiet",
            "--skip-simulation",
        ])
        .assert()
        .success();

    assert!(out.path().join("training/_manifest.json").is_file());
}

#[test]
fn valid_case_creates_convergence_parquet() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();

    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
            "--quiet",
            "--skip-simulation",
        ])
        .assert()
        .success();

    assert!(out.path().join("training/convergence.parquet").is_file());
}

#[test]
fn skip_simulation_does_not_produce_simulation_manifest() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();

    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
            "--quiet",
            "--skip-simulation",
        ])
        .assert()
        .success();

    assert!(!out.path().join("simulation/_manifest.json").exists());
}

#[test]
fn custom_output_dir_receives_training_artifacts() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let custom_out = TempDir::new().unwrap();
    assert_ne!(dir.path(), custom_out.path());

    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            custom_out.path().to_str().unwrap(),
            "--quiet",
            "--skip-simulation",
        ])
        .assert()
        .success();

    assert!(custom_out.path().join("training/_manifest.json").is_file());
    assert!(!dir.path().join("output").exists());
}

#[test]
fn missing_required_file_exits_1() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    fs::remove_file(dir.path().join("system/buses.json")).unwrap();

    cobre()
        .args(["run", dir.path().to_str().unwrap(), "--quiet"])
        .assert()
        .failure()
        .code(1);
}

#[test]
fn missing_required_file_stderr_contains_validation_error() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    fs::remove_file(dir.path().join("system/buses.json")).unwrap();

    cobre()
        .args(["run", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stderr(predicate::str::contains("error"));
}

#[test]
fn nonexistent_path_exits_2() {
    cobre()
        .args(["run", "/nonexistent/path/that/does/not/exist", "--quiet"])
        .assert()
        .failure()
        .code(2);
}

#[test]
fn nonexistent_path_stderr_contains_io_error() {
    cobre()
        .args(["run", "/nonexistent/path/that/does/not/exist"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("I/O error"));
}

#[test]
fn test_run_quiet_no_banner_no_summary() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();
    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
            "--quiet",
            "--skip-simulation",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("COBRE v").not())
        .stderr(predicate::str::contains("Training complete in").not());
}

#[test]
fn test_run_no_banner_flag_suppresses_banner() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();
    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
            "--no-banner",
            "--skip-simulation",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("COBRE v").not())
        .stderr(predicate::str::contains("Training complete in"));
}

/// Write a valid case directory using the supplied `config.json` content.
fn make_valid_case_with_config(dir: &TempDir, config_content: &str) {
    let root = dir.path();
    write_file(root, "config.json", config_content);
    write_file(root, "penalties.json", PENALTIES_JSON);
    write_file(root, "stages.json", STAGES_JSON);
    write_file(root, "initial_conditions.json", INITIAL_CONDITIONS_JSON);
    write_file(root, "system/buses.json", BUSES_JSON);
    write_file(root, "system/lines.json", LINES_JSON);
    write_file(root, "system/hydros.json", HYDROS_JSON);
    write_file(root, "system/thermals.json", THERMALS_JSON);
}

// ---- Seed diagnostic tests --------------------------------------------------

/// AC: When `training.seed` is absent, stderr contains the default-seed hint
/// emitted by the stochastic diagnostics block, and the run exits 0.
#[test]
fn no_seed_in_config_emits_warning_on_stderr() {
    let dir = TempDir::new().unwrap();
    make_valid_case_with_config(&dir, CONFIG_JSON);
    let out = TempDir::new().unwrap();

    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
            "--skip-simulation",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains(
            "set training.seed for reproducibility",
        ));
}

/// AC: When `training.seed` is explicitly set, stderr does NOT contain the
/// default-seed hint (the seed line shows the value without any hint suffix).
#[test]
fn explicit_seed_in_config_suppresses_warning() {
    let dir = TempDir::new().unwrap();
    make_valid_case_with_config(&dir, CONFIG_WITH_SEED_JSON);
    let out = TempDir::new().unwrap();

    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
            "--skip-simulation",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("set training.seed for reproducibility").not());
}
