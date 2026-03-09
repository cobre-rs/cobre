//! Integration tests for the `cobre validate` subcommand.
//!
//! Tests the full path: binary invocation → `load_case` → formatted output →
//! exit code.  Fixtures are built programmatically in temporary directories.

#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::*;
use predicates::prelude::*;
use tempfile::TempDir;

// ── fixture helpers ───────────────────────────────────────────────────────────

fn cobre() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

fn write_file(root: &Path, relative: &str, content: &str) {
    let full = root.join(relative);
    if let Some(parent) = full.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&full, content).unwrap();
}

const CONFIG_JSON: &str = r#"{
    "training": {
        "forward_passes": 10,
        "stopping_rules": [
            { "type": "iteration_limit", "limit": 100 }
        ]
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
            "num_scenarios": 50
        }
    ]
}"#;

const INITIAL_CONDITIONS_JSON: &str = r#"{ "storage": [], "filling_storage": [] }"#;
const BUSES_JSON: &str = r#"{ "buses": [{ "id": 1, "name": "BUS_1" }] }"#;
const LINES_JSON: &str = r#"{ "lines": [] }"#;
const HYDROS_JSON: &str = r#"{ "hydros": [] }"#;
const THERMALS_JSON: &str = r#"{ "thermals": [] }"#;

/// Build a minimal valid case directory in `dir`.
///
/// Contains 1 bus, 0 hydros, 0 thermals, 0 lines, 1 stage.
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

// ── acceptance criterion 1: valid case → exit 0, entity counts in stdout ─────

#[test]
fn valid_case_exits_0() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn valid_case_stdout_contains_buses_count() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("buses:"));
}

// ── acceptance criterion 2: missing required file → exit 1, error in stdout ──

#[test]
fn missing_buses_json_exits_1() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    // Remove the required buses.json to trigger a validation error.
    fs::remove_file(dir.path().join("system/buses.json")).unwrap();
    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1);
}

#[test]
fn missing_buses_json_stdout_contains_error() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    fs::remove_file(dir.path().join("system/buses.json")).unwrap();
    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stdout(predicate::str::contains("error"));
}

#[test]
fn missing_buses_json_stdout_mentions_file() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    fs::remove_file(dir.path().join("system/buses.json")).unwrap();
    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stdout(predicate::str::contains("buses.json"));
}

// ── acceptance criterion 3: nonexistent path → exit 2, I/O error in stderr ───

#[test]
fn nonexistent_path_exits_2() {
    cobre()
        .args(["validate", "/nonexistent/path/that/does/not/exist"])
        .assert()
        .failure()
        .code(2);
}

#[test]
fn nonexistent_path_stderr_mentions_path() {
    cobre()
        .args(["validate", "/nonexistent/path/that/does/not/exist"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("nonexistent"));
}

// ── acceptance criterion 4: piped output has no ANSI escape sequences ─────────

#[test]
fn valid_case_piped_stdout_has_no_ansi_escapes() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    // When stdout is not a terminal (as in this test), `console` strips ANSI codes.
    let output = cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    // ANSI escape sequences start with ESC (\x1b) followed by '['.
    assert!(
        !stdout.contains('\x1b'),
        "stdout should contain no ANSI escape sequences when piped, got: {stdout:?}"
    );
}
