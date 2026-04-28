//! Integration tests for the `cobre run` subcommand.
//!
//! Tests the full execution lifecycle: binary invocation → load → train →
//! simulate → write output → exit code. Fixtures are built programmatically
//! in temporary directories.
//!
//! # Coverage
//!
//! - AC-1: Valid case directory → exit 0, `training/convergence.parquet` and
//!   `training/metadata.json` exist in the default output directory.
//! - AC-2: `simulation.enabled: false` in config → exit 0, no `simulation/metadata.json`.
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
const CONFIG_JSON: &str = r#"{
    "training": {
        "forward_passes": 1,
        "stopping_rules": [
            { "type": "iteration_limit", "limit": 2 }
        ],
        "scenario_source": { "inflow": { "scheme": "in_sample" }, "seed": 42 }
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
        ])
        .assert()
        .success();
}

#[test]
fn valid_case_creates_training_metadata() {
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
        ])
        .assert()
        .success();

    assert!(out.path().join("training/metadata.json").is_file());
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
        ])
        .assert()
        .success();

    assert!(out.path().join("training/convergence.parquet").is_file());
}

#[test]
fn disabled_simulation_does_not_produce_manifest() {
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
        ])
        .assert()
        .success();

    assert!(!out.path().join("simulation/metadata.json").exists());
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
        ])
        .assert()
        .success();

    assert!(custom_out.path().join("training/metadata.json").is_file());
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
fn test_run_quiet_suppresses_banner_and_summary() {
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
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("COBRE v").not())
        .stderr(predicate::str::contains("Training complete in").not());
}

/// Config with `exports.stochastic = true` and `estimation.order_selection = "pacf_annual"`.
const CONFIG_STOCHASTIC_PAR_A_JSON: &str = r#"{
    "training": {
        "forward_passes": 1,
        "stopping_rules": [
            { "type": "iteration_limit", "limit": 2 }
        ],
        "scenario_source": { "inflow": { "scheme": "in_sample" }, "seed": 42 }
    },
    "exports": { "stochastic": true },
    "estimation": { "order_selection": "pacf_annual" }
}"#;

/// CLI run with `exports.stochastic = true` and `pacf_annual` order selection writes
/// `output/stochastic/inflow_annual_component.parquet`. On a case with no hydros the
/// file is written with zero data rows but a valid Arrow schema.
#[test]
fn cli_run_writes_inflow_annual_component_when_par_a_active() {
    let dir = TempDir::new().unwrap();
    // Reuse the existing fixture helpers; only swap in the stochastic config.
    write_file(dir.path(), "config.json", CONFIG_STOCHASTIC_PAR_A_JSON);
    write_file(dir.path(), "penalties.json", PENALTIES_JSON);
    write_file(dir.path(), "stages.json", STAGES_JSON);
    write_file(
        dir.path(),
        "initial_conditions.json",
        INITIAL_CONDITIONS_JSON,
    );
    write_file(dir.path(), "system/buses.json", BUSES_JSON);
    write_file(dir.path(), "system/lines.json", LINES_JSON);
    write_file(dir.path(), "system/hydros.json", HYDROS_JSON);
    write_file(dir.path(), "system/thermals.json", THERMALS_JSON);

    let out = TempDir::new().unwrap();

    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .success();

    let parquet_path = out
        .path()
        .join("stochastic/inflow_annual_component.parquet");
    assert!(
        parquet_path.is_file(),
        "stochastic/inflow_annual_component.parquet must exist"
    );

    // Round-trip through the parser: zero rows expected (no hydros in fixture).
    let rows = cobre_io::scenarios::parse_inflow_annual_component(&parquet_path).unwrap();
    assert_eq!(
        rows.len(),
        0,
        "expected zero annual component rows for a case with no hydros"
    );
}
