//! Shared harness for `cobre-cli` run-path integration tests: spawning the
//! binary, resolving committed example cases, and building minimal valid-case
//! fixtures in a temp dir.

#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]
// Items may be unused in one binary but used in another.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Spawns the `cobre` binary under test.
pub fn cobre() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

/// Resolves `examples/<name>` relative to the repository root.
pub fn case_dir(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root must be two levels above CARGO_MANIFEST_DIR");
    root.join("examples").join(name)
}

/// Writes `content` to `root/relative`, creating parent directories as needed.
pub fn write_file(root: &Path, relative: &str, content: &str) {
    let full = root.join(relative);
    if let Some(parent) = full.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&full, content).unwrap();
}

/// Penalty config shared by every programmatically-built case fixture.
pub const PENALTIES_JSON: &str = r#"{
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
        "turbined_cost": 0.05,
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

const DEFAULT_CONFIG_JSON: &str = r#"{
    "training": {
        "selection": { "method": "sampled", "forward_passes": 1 },
        "stopping_rules": [
            { "type": "iteration_limit", "limit": 2 }
        ],
        "scenario_source": { "inflow": { "scheme": "in_sample" }, "seed": 42 }
    }
}"#;

const DEFAULT_STAGES_JSON: &str = r#"{
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
            "num_openings": 2
        },
        {
            "id": 1,
            "start_date": "2024-02-01",
            "end_date": "2024-03-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 672.0 }],
            "num_openings": 2
        }
    ]
}"#;

const DEFAULT_INITIAL_CONDITIONS_JSON: &str = r#"{ "storage": [], "filling_storage": [] }"#;
const DEFAULT_BUSES_JSON: &str =
    r#"{ "buses": [{ "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" }] }"#;
const DEFAULT_LINES_JSON: &str = r#"{ "lines": [] }"#;
const DEFAULT_HYDROS_JSON: &str = r#"{ "hydros": [] }"#;
const DEFAULT_THERMALS_JSON: &str = r#"{ "thermals": [] }"#;

/// Rewrites the cobre version recorded in `policy_dir/manifest.bin`.
pub fn restamp_policy_version(policy_dir: &Path, version: &str) {
    let path = policy_dir.join("manifest.bin");
    let mut manifest =
        cobre_io::deserialize_checkpoint_manifest(&fs::read(&path).unwrap()).unwrap();
    manifest.cobre_version = version.to_string();
    fs::write(&path, cobre_io::serialize_checkpoint_manifest(&manifest)).unwrap();
}

/// Writes a minimal valid case fixture under `dir`. Each `Some` override
/// replaces the matching default; buses/lines/hydros use fixed defaults.
pub fn make_valid_case(
    dir: &Path,
    config_json: Option<&str>,
    stages_json: Option<&str>,
    initial_conditions_json: Option<&str>,
    thermals_json: Option<&str>,
) {
    write_file(
        dir,
        "config.json",
        config_json.unwrap_or(DEFAULT_CONFIG_JSON),
    );
    write_file(dir, "penalties.json", PENALTIES_JSON);
    write_file(
        dir,
        "stages.json",
        stages_json.unwrap_or(DEFAULT_STAGES_JSON),
    );
    write_file(
        dir,
        "initial_conditions.json",
        initial_conditions_json.unwrap_or(DEFAULT_INITIAL_CONDITIONS_JSON),
    );
    write_file(dir, "system/buses.json", DEFAULT_BUSES_JSON);
    write_file(dir, "system/lines.json", DEFAULT_LINES_JSON);
    write_file(dir, "system/hydros.json", DEFAULT_HYDROS_JSON);
    write_file(
        dir,
        "system/thermals.json",
        thermals_json.unwrap_or(DEFAULT_THERMALS_JSON),
    );
}
