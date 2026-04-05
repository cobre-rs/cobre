//! Integration tests for the `cobre report` subcommand.
//!
//! Tests run the compiled `cobre` binary against temporary directories
//! containing minimal metadata JSON fixtures.
//!
//! # Coverage
//!
//! - AC-1: Results directory with `training/metadata.json` → exit 0, stdout is
//!   valid JSON with a `training` key containing `iterations`.
//! - AC-2: Results directory with both training and simulation metadata →
//!   exit 0, stdout JSON contains both `training` and `simulation` keys
//!   (neither is null).
//! - AC-3: Nonexistent results directory → exit 2, stderr contains error
//!   message about the missing path.
//! - AC-4: Valid results directory → `jq .training.status` would yield a valid
//!   status string (verified by parsing the `status` field from stdout).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::*;
use predicates::prelude::*;
use tempfile::TempDir;

// ── Binary helper ─────────────────────────────────────────────────────────────

fn cobre() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

// ── Fixture constants ─────────────────────────────────────────────────────────

const TRAINING_METADATA_JSON: &str = r#"{
    "cobre_version": "0.3.2",
    "hostname": "test-host",
    "solver": "highs",
    "started_at": "2026-01-17T08:00:00Z",
    "completed_at": "2026-01-17T12:30:00Z",
    "duration_seconds": 16200.0,
    "status": "complete",
    "configuration": {
        "seed": 42,
        "max_iterations": 100,
        "forward_passes": 192,
        "stopping_mode": "any",
        "policy_mode": "fresh"
    },
    "problem_dimensions": {
        "num_stages": 12,
        "num_hydros": 160,
        "num_thermals": 200,
        "num_buses": 5,
        "num_lines": 8
    },
    "iterations": {
        "completed": 10,
        "converged_at": 10
    },
    "convergence": {
        "achieved": true,
        "final_gap_percent": 0.45,
        "termination_reason": "bound_stalling"
    },
    "cuts": {
        "total_generated": 1250000,
        "total_active": 980000,
        "peak_active": 1100000
    },
    "mpi": { "world_size": 1, "ranks_participated": 1 }
}"#;

const SIMULATION_METADATA_JSON: &str = r#"{
    "cobre_version": "0.3.2",
    "hostname": "test-host",
    "solver": "highs",
    "started_at": "2026-01-17T13:00:00Z",
    "completed_at": "2026-01-17T13:15:00Z",
    "duration_seconds": 900.0,
    "status": "complete",
    "scenarios": { "total": 100, "completed": 100, "failed": 0 },
    "mpi": { "world_size": 1, "ranks_participated": 1 }
}"#;

// ── Fixture helpers ───────────────────────────────────────────────────────────

fn write_file(root: &Path, relative: &str, content: &str) {
    let full = root.join(relative);
    if let Some(parent) = full.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&full, content).unwrap();
}

/// Build a minimal results directory with only training metadata.
fn make_training_only_results(dir: &TempDir) {
    let root = dir.path();
    write_file(root, "training/metadata.json", TRAINING_METADATA_JSON);
}

/// Build a results directory with both training and simulation metadata.
fn make_full_results(dir: &TempDir) {
    make_training_only_results(dir);
    write_file(
        dir.path(),
        "simulation/metadata.json",
        SIMULATION_METADATA_JSON,
    );
}

// ── AC-1: training metadata → exit 0, valid JSON with `iterations` ────────────

#[test]
fn training_only_exits_0() {
    let dir = TempDir::new().unwrap();
    make_training_only_results(&dir);

    cobre()
        .args(["report", dir.path().to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn training_only_stdout_is_valid_json() {
    let dir = TempDir::new().unwrap();
    make_training_only_results(&dir);

    let output = cobre()
        .args(["report", dir.path().to_str().unwrap()])
        .output()
        .unwrap();

    assert!(output.status.success(), "exit code must be 0");
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: serde_json::Value =
        serde_json::from_str(&stdout).expect("stdout must be valid JSON");

    assert!(
        value["training"].is_object(),
        "stdout JSON must contain a 'training' key"
    );
    assert!(
        value["training"]["iterations"].is_object(),
        "training must have an 'iterations' object"
    );
    assert_eq!(
        value["training"]["iterations"]["completed"].as_u64(),
        Some(10),
        "training.iterations.completed must equal 10"
    );
}

#[test]
fn training_only_simulation_key_is_null() {
    let dir = TempDir::new().unwrap();
    make_training_only_results(&dir);

    let output = cobre()
        .args(["report", dir.path().to_str().unwrap()])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    assert!(
        value["simulation"].is_null(),
        "simulation must be null when only training metadata exists"
    );
}

// ── AC-2: training + simulation → both keys present ───────────────────────────

#[test]
fn full_results_exits_0() {
    let dir = TempDir::new().unwrap();
    make_full_results(&dir);

    cobre()
        .args(["report", dir.path().to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn full_results_both_training_and_simulation_present() {
    let dir = TempDir::new().unwrap();
    make_full_results(&dir);

    let output = cobre()
        .args(["report", dir.path().to_str().unwrap()])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    assert!(
        value["training"].is_object(),
        "training must be an object when both metadata files exist"
    );
    assert!(
        value["simulation"].is_object(),
        "simulation must be an object (not null) when simulation metadata exists"
    );
    assert_eq!(
        value["simulation"]["scenarios"]["total"].as_u64(),
        Some(100),
        "simulation.scenarios.total must equal 100"
    );
}

// ── AC-3: nonexistent directory → exit 2 ─────────────────────────────────────

#[test]
fn nonexistent_directory_exits_2() {
    cobre()
        .args(["report", "/nonexistent/results/path/that/cannot/exist"])
        .assert()
        .failure()
        .code(2);
}

#[test]
fn nonexistent_directory_stderr_contains_error() {
    cobre()
        .args(["report", "/nonexistent/results/path/that/cannot/exist"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("error"));
}

// ── AC-4: valid results → training.status is a non-empty string ──────────────

#[test]
fn status_field_is_valid_string() {
    let dir = TempDir::new().unwrap();
    make_training_only_results(&dir);

    let output = cobre()
        .args(["report", dir.path().to_str().unwrap()])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    let status = value["training"]["status"]
        .as_str()
        .expect("training.status must be a string");
    assert!(
        !status.is_empty(),
        "training.status must be a non-empty string, got empty"
    );
    // The fixture uses "complete"; verify it is forwarded correctly.
    assert_eq!(
        status, "complete",
        "training.status must match the training metadata"
    );
}

// ── Additional edge cases ─────────────────────────────────────────────────────

#[test]
fn output_directory_field_contains_absolute_path() {
    let dir = TempDir::new().unwrap();
    make_training_only_results(&dir);

    let output = cobre()
        .args(["report", dir.path().to_str().unwrap()])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    let out_dir = value["output_directory"]
        .as_str()
        .expect("output_directory must be a string");
    assert!(
        std::path::Path::new(out_dir).is_absolute(),
        "output_directory must be an absolute path, got: {out_dir}"
    );
}

#[test]
fn missing_training_manifest_exits_2() {
    let dir = TempDir::new().unwrap();
    // Create the directory but do not write the required training metadata.
    fs::create_dir_all(dir.path().join("training")).unwrap();

    cobre()
        .args(["report", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(2);
}
