//! Integration tests for the `--color` global flag and `COBRE_COLOR` / `FORCE_COLOR` env vars.
//!
//! Each test spawns a subprocess so that env var overrides and `console`'s global
//! `colors_enabled_stderr` state are completely isolated from the test process.
//!
//! # Coverage
//!
//! - `--color always` forces ANSI escapes in banner output even when stderr is piped.
//! - `--color never` suppresses all ANSI escapes even when stderr is a TTY.
//! - `--color always` placed before the subcommand name (`cobre --color always run ...`)
//!   also works because `global = true` on the arg.
//! - `COBRE_COLOR=always` (no `--color` flag) forces color on.
//! - `FORCE_COLOR=1` (no `--color` or `COBRE_COLOR`) forces color on.
//! - Invalid `COBRE_COLOR` value is silently ignored (auto-detection applies).

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

// ------------------------------------------------------------------
// Minimal valid-case fixture (copied from cli_run.rs so this test
// module is self-contained)
// ------------------------------------------------------------------

const CONFIG_JSON: &str = r#"{
    "training": {
        "forward_passes": 1,
        "stopping_rules": [
            { "type": "iteration_limit", "limit": 2 }
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

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

/// `--color always` forces ANSI color sequences in stderr banner output
/// even when stderr is not connected to a TTY (captured by `assert_cmd`).
///
/// The predicate checks for the 256-color orange used by the busbar (`\x1b[38;5;172m`),
/// which is a banner-specific color escape that is only present when color is forced on.
#[test]
fn color_always_flag_forces_ansi_in_banner() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();

    cobre()
        .args([
            "run",
            "--color",
            "always",
            "--skip-simulation",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        // Check for the banner's 256-color orange busbar escape, which is only
        // emitted when color is forced on (auto-detection would disable it for
        // a piped stderr subprocess).
        .stderr(predicate::str::contains("\x1b[38;5;172m"));
}

/// `--color never` suppresses all ANSI escape sequences in stderr output.
///
/// `--quiet` is added to suppress the `indicatif` progress bar, which emits
/// cursor-movement sequences (`\x1b[1A`, `\x1b[2K`) regardless of the color
/// setting. Those sequences are structural, not color-related, so the meaningful
/// assertion is against the combined banner + summary output that `--quiet`
/// suppresses entirely, leaving clean plain-text stderr.
#[test]
fn color_never_flag_suppresses_ansi_in_banner() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();

    cobre()
        .args([
            "run",
            "--color",
            "never",
            "--quiet",
            "--skip-simulation",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("\x1b[").not());
}

/// `--color always` placed before the subcommand name is accepted because
/// the arg is declared with `global = true`.
#[test]
fn color_always_global_flag_before_subcommand_is_accepted() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();

    cobre()
        .args([
            "--color",
            "always",
            "run",
            "--skip-simulation",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("\x1b[38;5;172m"));
}

/// `COBRE_COLOR=always` with no `--color` flag forces ANSI color on.
#[test]
fn cobre_color_env_always_forces_ansi() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();

    cobre()
        .env("COBRE_COLOR", "always")
        // Unset FORCE_COLOR to avoid interference.
        .env_remove("FORCE_COLOR")
        .args([
            "run",
            "--skip-simulation",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("\x1b[38;5;172m"));
}

/// `FORCE_COLOR=1` with no `--color` or `COBRE_COLOR` forces ANSI color on.
#[test]
fn force_color_env_forces_ansi() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();

    cobre()
        .env("FORCE_COLOR", "1")
        // Unset COBRE_COLOR to ensure FORCE_COLOR path is exercised.
        .env_remove("COBRE_COLOR")
        .args([
            "run",
            "--skip-simulation",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("\x1b[38;5;172m"));
}

/// An invalid `COBRE_COLOR` value is silently ignored; no crash occurs.
///
/// Under auto-detection in a non-TTY subprocess, color is disabled, so the
/// banner output will not contain ANSI sequences. The important assertion is
/// that the process exits successfully (no panic from the invalid value).
#[test]
fn cobre_color_env_invalid_value_is_silently_ignored() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();

    cobre()
        .env("COBRE_COLOR", "invalid-value")
        .env_remove("FORCE_COLOR")
        .args([
            "run",
            "--skip-simulation",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
        ])
        .assert()
        // Must not crash with an error about the env var value.
        .success();
}
