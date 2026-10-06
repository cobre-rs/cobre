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

/// Recursively copies `src` into `dst` without following symlinks.
pub fn copy_dir_recursive(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for entry in fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir_recursive(&from, &to);
        } else {
            fs::copy(&from, &to).unwrap();
        }
    }
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

/// Rewrites the software version recorded in `policy_dir/manifest.bin`.
pub fn restamp_policy_version(policy_dir: &Path, version: &str) {
    restamp_policy_manifest(policy_dir, |manifest| {
        manifest.software_version = version.to_string();
    });
}

/// Rewrites the software name recorded in `policy_dir/manifest.bin`.
pub fn restamp_policy_software(policy_dir: &Path, software: &str) {
    restamp_policy_manifest(policy_dir, |manifest| {
        manifest.software = Some(software.to_string());
    });
}

fn restamp_policy_manifest(
    policy_dir: &Path,
    edit: impl FnOnce(&mut cobre_io::CheckpointManifest),
) {
    let path = policy_dir.join("manifest.bin");
    let mut manifest =
        cobre_io::deserialize_checkpoint_manifest(&fs::read(&path).unwrap()).unwrap();
    edit(&mut manifest);
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

/// Builds a case under `case` whose stages sample `historical_residuals` and
/// whose `{source: file}` opening tree is the one a first run exported, over a
/// 1dtoy that carries no inflow history.
pub fn write_supplied_opening_tree_case(case: &Path) {
    copy_dir_recursive(&case_dir("1dtoy"), case);
    let mut config = serde_json::json!({
        "training": {
            "selection": { "method": "sampled", "forward_passes": 1 },
            "stopping_rules": [{ "type": "iteration_limit", "limit": 1 }],
            "scenario_source": {
                "seed": 42,
                "inflow": { "scheme": "in_sample" },
                "load": { "scheme": "in_sample" },
                "ncs": { "scheme": "in_sample" }
            }
        },
        "simulation": { "enabled": false },
        "modeling": { "inflow_non_negativity": { "method": "none" } },
        "exports": { "stochastic": true }
    });
    write_file(case, "config.json", &config.to_string());

    let export = tempfile::TempDir::new().unwrap();
    let run = cobre()
        .args(["run", case.to_str().unwrap()])
        .args(["--output", export.path().to_str().unwrap(), "--quiet"])
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "the export run must succeed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    fs::copy(
        export.path().join("stochastic/noise_openings.parquet"),
        case.join("scenarios/noise_openings.parquet"),
    )
    .unwrap();

    let stages = fs::read_to_string(case.join("stages.json")).unwrap();
    let stages = stages.replace(
        "\"num_openings\": 10",
        "\"num_openings\": 10, \"sampling_method\": \"historical_residuals\"",
    );
    assert_eq!(stages.matches("historical_residuals").count(), 4);
    write_file(case, "stages.json", &stages);

    config.as_object_mut().unwrap().remove("exports");
    config["training"]["scenario_source"]["openings"] = serde_json::json!({ "source": "file" });
    write_file(case, "config.json", &config.to_string());
}
