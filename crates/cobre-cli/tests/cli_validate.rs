//! Integration tests for the `cobre validate` subcommand.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use assert_cmd::prelude::*;
use predicates::prelude::*;
use tempfile::TempDir;

mod common;
use common::{PENALTIES_JSON, copy_dir_recursive, write_supplied_opening_tree_case};

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
        "selection": { "method": "sampled", "forward_passes": 10 },
        "stopping_rules": [
            { "type": "iteration_limit", "limit": 100 }
        ],
        "scenario_source": { "inflow": { "scheme": "in_sample" }, "seed": 42 }
    }
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
            "num_openings": 50
        },
        {
            "id": 1,
            "start_date": "2024-02-01",
            "end_date": "2024-03-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 696.0 }],
            "num_openings": 50
        }
    ]
}"#;

const INITIAL_CONDITIONS_JSON: &str = r#"{ "storage": [], "filling_storage": [] }"#;
const BUSES_JSON: &str =
    r#"{ "buses": [{ "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" }] }"#;
const LINES_JSON: &str = r#"{ "lines": [] }"#;
const HYDROS_JSON: &str = r#"{ "hydros": [] }"#;
const THERMALS_JSON: &str = r#"{ "thermals": [] }"#;

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
        .stdout(predicate::str::contains("buses,"));
}

#[test]
fn missing_buses_json_exits_1() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
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

/// A case that fails BEFORE boundary reconciliation (here, the six-layer IO
/// pipeline over a missing required file) must still emit a single
/// parseable JSON object under `--json` — never interleaved human report
/// text — carrying the failing phase and message, with stderr empty and a
/// non-zero exit.
#[test]
fn missing_buses_json_json_mode_emits_parseable_error_object() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    fs::remove_file(dir.path().join("system/buses.json")).unwrap();

    let output = cobre()
        .args(["validate", dir.path().to_str().unwrap(), "--json"])
        .output()
        .unwrap();

    assert_eq!(
        output.status.code(),
        Some(1),
        "expected the validation exit code"
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&stdout);
    assert!(
        parsed.is_ok(),
        "stdout must be a single parseable JSON object, got parse error {:?} for: {stdout:?}",
        parsed.as_ref().err()
    );
    let value = parsed.unwrap();
    assert!(
        value["configured"].is_null(),
        "configured must stay absent on an early abort: {value}"
    );
    assert!(
        value["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("buses.json")),
        "the error object must name the offending file: {value}"
    );

    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.trim().is_empty(),
        "stderr must stay empty (already_rendered): got {stderr:?}"
    );
}

/// stderr must NOT carry the "run `cobre validate`" hint — that would point the
/// user back at the very command they just ran.
#[test]
fn validate_failure_report_in_stdout_not_stderr() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    fs::remove_file(dir.path().join("system/buses.json")).unwrap();
    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stdout(predicate::str::contains("error"))
        .stderr(predicate::str::contains("buses.json").not())
        .stderr(predicate::str::contains("run `cobre validate`").not());
}

/// The offending path must surface exactly once: the relative prefix and the
/// embedded `SchemaError` display both carry it, so the two must be deduped.
#[test]
fn validate_schema_failure_path_appears_once() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    write_file(
        dir.path(),
        "system/buses.json",
        r#"{ "buses": [{ "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" }, { "id": 1, "name": "BUS_2", "operational_start_date": "2024-01-01" }] }"#,
    );

    let output = cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "expected validation failure");
    let stdout = String::from_utf8(output.stdout).unwrap();
    let occurrences = stdout.matches("system/buses.json").count();
    assert_eq!(
        occurrences, 1,
        "offending path must appear exactly once, found {occurrences} in: {stdout:?}"
    );
}

/// A duplicate bus id is collected by the six-layer pipeline into a single
/// `LoadError::ConstraintError`; under `--json` its `phase` must be the shared
/// `cobre-io` kind vocabulary, not a CLI-only classifier string.
#[test]
fn duplicate_bus_id_json_mode_reports_constraint_error_kind() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    write_file(
        dir.path(),
        "system/buses.json",
        r#"{ "buses": [{ "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" }, { "id": 1, "name": "BUS_2", "operational_start_date": "2024-01-01" }] }"#,
    );

    let output = cobre()
        .args(["validate", dir.path().to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "expected validation failure");

    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(
        value["error"]["phase"],
        serde_json::json!("ConstraintError"),
        "kind must be the shared cobre-io vocabulary: {value}"
    );
}

/// A `config.json` parse failure is caught after the six-layer pipeline
/// succeeds; under `--json` it must still emit a single error object.
#[test]
fn config_parse_failure_json_mode_emits_error_object() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    write_file(dir.path(), "config.json", "{ not valid json");

    let output = cobre()
        .args(["validate", dir.path().to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(!output.status.success(), "expected a validation failure");

    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&stdout);
    assert!(
        parsed.is_ok(),
        "stdout must be a single parseable JSON object, got parse error {:?} for: {stdout:?}",
        parsed.as_ref().err()
    );
    let value = parsed.unwrap();
    assert!(
        value["error"]["message"].is_string(),
        "expected an error object with a message, got: {value}"
    );
}

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

#[test]
fn valid_case_piped_stdout_has_no_ansi_escapes() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    // `console` strips ANSI codes when stdout is not a terminal, as it is here.
    let output = cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        !stdout.contains('\x1b'),
        "stdout should contain no ANSI escape sequences when piped, got: {stdout:?}"
    );
}

/// An unknown `cut_selection` key is a hard schema error under
/// `deny_unknown_fields`, not silently ignored.
#[test]
fn removed_cut_selection_field_fails_validate() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);

    let removed_field_config = r#"{
        "training": {
            "selection": { "method": "sampled", "forward_passes": 10 },
            "stopping_rules": [
                { "type": "iteration_limit", "limit": 100 }
            ],
            "scenario_source": { "inflow": { "scheme": "in_sample" }, "seed": 42 },
            "cut_selection": { "basis_activity_window": 100 }
        }
    }"#;
    write_file(dir.path(), "config.json", removed_field_config);

    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .failure();
}

/// An invalid `simulation.scenario_source` must be rejected by both `validate`
/// and `run`, carrying the same rule message — the training half of the
/// config stays valid so the failure is unambiguously the simulation source.
#[test]
fn invalid_simulation_scenario_source_fails_validate_and_run() {
    const MSG: &str = "historical scheme is only valid for the inflow class";

    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);

    let mut config: serde_json::Value = serde_json::from_str(CONFIG_JSON).unwrap();
    config["simulation"] = serde_json::json!({
        "scenario_source": { "seed": 1, "load": { "scheme": "historical" } }
    });
    write_file(dir.path(), "config.json", &config.to_string());

    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .stdout(predicate::str::contains(MSG));

    cobre()
        .args(["run", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains(MSG));
}

#[test]
fn simulation_out_of_sample_with_its_own_seed_passes_validate_and_run() {
    let dir = TempDir::new().unwrap();
    copy_dir_recursive(
        &common::case_dir("deterministic/d29-weekly-par-noise-sharing"),
        dir.path(),
    );
    let config_path = dir.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&config_path).unwrap()).unwrap();
    config["training"]["scenario_source"] =
        serde_json::json!({ "inflow": { "scheme": "in_sample" } });
    config["training"]["stopping_rules"] =
        serde_json::json!([{ "type": "iteration_limit", "limit": 3 }]);
    config["simulation"] = serde_json::json!({
        "enabled": true,
        "selection": { "method": "sampled", "num_scenarios": 2 },
        "scenario_source": { "seed": 42, "inflow": { "scheme": "out_of_sample" } }
    });
    fs::write(&config_path, config.to_string()).unwrap();

    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let output = TempDir::new().unwrap();
    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            output.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .success();
}

#[test]
fn checkpointing_enabled_without_interval_exits_1() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);

    let mut config: serde_json::Value = serde_json::from_str(CONFIG_JSON).unwrap();
    config["policy"] = serde_json::json!({ "checkpointing": { "enabled": true } });
    write_file(dir.path(), "config.json", &config.to_string());

    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stdout(predicate::str::contains(
            "policy.checkpointing.interval_iterations",
        ));
}

/// An FPHA hydro with no `hydro_production_models.json` entry slips past the
/// IO pipeline (the Layer-4 dimensional check skips FPHA hydros when
/// `fpha_hyperplanes.parquet` is absent) and is rejected only at Phase 10 by
/// `prepare_hydro_models_from_artifacts` → `determine_source`.
fn write_fpha_hydro_without_production_models_case(dir: &TempDir) {
    write_file(dir.path(), "config.json", CONFIG_JSON);
    write_file(dir.path(), "penalties.json", PENALTIES_JSON);
    write_file(dir.path(), "stages.json", STAGES_JSON);
    write_file(
        dir.path(),
        "initial_conditions.json",
        INITIAL_CONDITIONS_JSON,
    );
    write_file(dir.path(), "system/buses.json", BUSES_JSON);
    write_file(dir.path(), "system/lines.json", LINES_JSON);
    write_file(dir.path(), "system/thermals.json", THERMALS_JSON);

    let fpha_hydros_json = r#"{
        "hydros": [
            {
                "id": 1,
                "name": "UHE_FPHA",
                "operational_start_date": "2024-01-01",
                "downstream_id": null,
                "reservoir": {
                    "min_storage_hm3": 0.0,
                    "max_storage_hm3": 500.0
                },
                "outflow": {
                    "min_outflow_m3s": 0.0,
                    "max_outflow_m3s": null
                },
                "generation": {
                    "model": "fpha",
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 100.0,
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 300.0
                },
                "unit_groups": [
                    {
                        "id": 0,
                        "name": "UHE_FPHA",
                        "bus_id": 1,
                        "min_generation_mw": 0.0,
                        "max_generation_mw": 300.0,
                        "min_turbined_m3s": 0.0,
                        "max_turbined_m3s": 100.0
                    }
                ]
            }
        ]
    }"#;
    write_file(dir.path(), "system/hydros.json", fpha_hydros_json);
}

#[test]
fn fpha_hydro_without_production_models_json_fails_validate() {
    let dir = TempDir::new().unwrap();
    write_fpha_hydro_without_production_models_case(&dir);

    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1);
}

// ── boundary reconciliation (`policy.boundary`, `--json`) ──────────────────────
//
// A minimal 2-stage, single-hydro case with a `constant_productivity`
// production model — the smallest fixture with a nonzero terminal storage
// slot, so `load_boundary_cuts`'s reconciliation has a slot to tally.

fn boundary_hydro_entry_json(hydro_id: i64) -> String {
    format!(
        r#"{{
                        "id": {hydro_id}, "name": "H", "operational_start_date": "2020-01-01",
                        "downstream_id": null,
                        "reservoir": {{ "min_storage_hm3": 0.0, "max_storage_hm3": 200.0 }},
                        "outflow": {{ "min_outflow_m3s": 0.0, "max_outflow_m3s": 50.0 }},
                        "generation": {{
                            "model": "constant_productivity",
                            "min_turbined_m3s": 0.0, "max_turbined_m3s": 50.0,
                            "min_generation_mw": 0.0, "max_generation_mw": 50.0
                        }},
                        "unit_groups": [
                            {{
                                "id": 0, "name": "H", "bus_id": 0,
                                "min_generation_mw": 0.0, "max_generation_mw": 50.0,
                                "min_turbined_m3s": 0.0, "max_turbined_m3s": 50.0
                            }}
                        ]
                    }}"#
    )
}

fn boundary_production_model_entry_json(hydro_id: i64) -> String {
    format!(
        r#"{{
                        "hydro_id": {hydro_id}, "selection_mode": "stage_ranges",
                        "stage_ranges": [
                            {{
                                "start_stage_id": 0, "end_stage_id": null,
                                "model": "constant_productivity", "productivity_mw_per_m3s": 1.0
                            }}
                        ]
                    }}"#
    )
}

/// Writes one entry per id in `hydro_ids` into `system/hydros.json`,
/// `system/hydro_production_models.json` and `initial_conditions.json`, each
/// hydro on bus 0 with `downstream_id: null`. [`write_boundary_case`] is the
/// one-element form.
fn write_boundary_case_with_hydros(dir: &Path, hydro_ids: &[i64]) {
    write_file(
        dir,
        "config.json",
        r#"{
            "training": {
                "selection": { "method": "sampled", "forward_passes": 1 },
                "stopping_rules": [{ "type": "iteration_limit", "limit": 1 }]
            },
            "simulation": { "enabled": false },
            "modeling": { "inflow_non_negativity": { "method": "none" } }
        }"#,
    );
    write_file(
        dir,
        "stages.json",
        r#"{
            "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0 },
            "stages": [
                {
                    "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
                    "blocks": [{ "id": 0, "name": "SINGLE", "hours": 730 }], "num_openings": 1
                },
                {
                    "id": 1, "start_date": "2024-02-01", "end_date": "2024-03-01",
                    "blocks": [{ "id": 0, "name": "SINGLE", "hours": 730 }], "num_openings": 1
                }
            ]
        }"#,
    );
    let hydro_entries = hydro_ids
        .iter()
        .map(|&id| boundary_hydro_entry_json(id))
        .collect::<Vec<_>>()
        .join(",\n                    ");
    write_file(
        dir,
        "system/hydros.json",
        &format!(
            r#"{{
                "hydros": [
                    {hydro_entries}
                ]
            }}"#
        ),
    );
    let production_model_entries = hydro_ids
        .iter()
        .map(|&id| boundary_production_model_entry_json(id))
        .collect::<Vec<_>>()
        .join(",\n                    ");
    write_file(
        dir,
        "system/hydro_production_models.json",
        &format!(
            r#"{{
                "production_models": [
                    {production_model_entries}
                ]
            }}"#
        ),
    );
    write_file(
        dir,
        "system/buses.json",
        r#"{ "buses": [
            { "id": 0, "name": "B0", "operational_start_date": "2020-01-01",
              "deficit_segments": [{ "depth_mw": null, "cost": 1000.0 }] }
        ] }"#,
    );
    write_file(dir, "system/lines.json", LINES_JSON);
    write_file(dir, "system/thermals.json", THERMALS_JSON);
    let storage_entries = hydro_ids
        .iter()
        .map(|&id| format!(r#"{{ "hydro_id": {id}, "value_hm3": 100.0 }}"#))
        .collect::<Vec<_>>()
        .join(", ");
    write_file(
        dir,
        "initial_conditions.json",
        &format!(r#"{{ "storage": [{storage_entries}], "filling_storage": [] }}"#),
    );
    write_file(
        dir,
        "penalties.json",
        r#"{
            "bus": {
                "deficit_segments": [{ "depth_mw": null, "cost": 1000.0 }],
                "excess_cost": 0.01
            },
            "line": { "exchange_cost": 0.01 },
            "hydro": {
                "spillage_cost": 0.01, "turbined_cost": 0.01, "diversion_cost": 0.01,
                "storage_violation_below_cost": 1500.0, "filling_target_violation_cost": 500.0,
                "turbined_violation_below_cost": 500.0, "outflow_violation_below_cost": 500.0,
                "outflow_violation_above_cost": 500.0, "generation_violation_below_cost": 500.0,
                "evaporation_violation_cost": 500.0, "water_withdrawal_violation_cost": 500.0
            },
            "non_controllable_source": { "curtailment_cost": 0.005 }
        }"#,
    );
}

fn write_boundary_case(dir: &Path, hydro_id: i64) {
    write_boundary_case_with_hydros(dir, &[hydro_id]);
}

/// Materializes the policy checkpoint at `dir/output/policy` that the
/// boundary tests point `policy.boundary.path` at.
fn run_case(dir: &Path) {
    cobre()
        .args(["run", dir.to_str().unwrap()])
        .assert()
        .success();
}

fn append_boundary_policy(dir: &Path, boundary_policy_dir: &Path) {
    let boundary_path = boundary_policy_dir.to_str().unwrap();
    let config = format!(
        r#"{{
            "training": {{
                "selection": {{ "method": "sampled", "forward_passes": 1 }},
                "stopping_rules": [{{ "type": "iteration_limit", "limit": 1 }}]
            }},
            "simulation": {{ "enabled": false }},
            "modeling": {{ "inflow_non_negativity": {{ "method": "none" }} }},
            "policy": {{ "boundary": {{ "path": "{boundary_path}" }} }}
        }}"#
    );
    write_file(dir, "config.json", &config);
}

/// [`append_boundary_policy`] with an explicit `strict` key merged in;
/// `append_boundary_policy` itself keeps writing no `strict` key at all.
fn append_boundary_policy_with_strict(dir: &Path, boundary_policy_dir: &Path, strict: bool) {
    let boundary_path = boundary_policy_dir.to_str().unwrap();
    let config = format!(
        r#"{{
            "training": {{
                "selection": {{ "method": "sampled", "forward_passes": 1 }},
                "stopping_rules": [{{ "type": "iteration_limit", "limit": 1 }}]
            }},
            "simulation": {{ "enabled": false }},
            "modeling": {{ "inflow_non_negativity": {{ "method": "none" }} }},
            "policy": {{ "boundary": {{ "path": "{boundary_path}", "strict": {strict} }} }}
        }}"#
    );
    write_file(dir, "config.json", &config);
}

/// Rewrites `config.json` to train with `policy.mode = mode` (`warm_start` or
/// `resume`) from the case's own just-produced checkpoint at the default
/// `output/policy` path.
fn append_policy_mode(dir: &Path, mode: &str) {
    let config = format!(
        r#"{{
            "training": {{
                "selection": {{ "method": "sampled", "forward_passes": 1 }},
                "stopping_rules": [{{ "type": "iteration_limit", "limit": 1 }}]
            }},
            "simulation": {{ "enabled": false }},
            "modeling": {{ "inflow_non_negativity": {{ "method": "none" }} }},
            "policy": {{ "mode": "{mode}" }}
        }}"#
    );
    write_file(dir, "config.json", &config);
}

fn append_warm_start_policy(dir: &Path) {
    append_policy_mode(dir, "warm_start");
}

fn append_resume_policy(dir: &Path) {
    append_policy_mode(dir, "resume");
}

/// Rewrites `config.json` to simulate from the case's own just-produced
/// checkpoint at the default `output/policy` path, without training.
fn append_simulation_only_policy(dir: &Path) {
    write_file(
        dir,
        "config.json",
        r#"{
            "training": {
                "enabled": false,
                "selection": { "method": "sampled", "forward_passes": 1 },
                "stopping_rules": [{ "type": "iteration_limit", "limit": 1 }]
            },
            "simulation": { "enabled": true, "selection": { "method": "sampled", "num_scenarios": 1 } },
            "modeling": { "inflow_non_negativity": { "method": "none" } }
        }"#,
    );
}

/// Every file under `dir` keyed by its path relative to `dir`, with its bytes:
/// two equal snapshots hold the same relative paths, lengths and contents.
fn snapshot_files(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, files: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(root, &path, files);
            } else {
                let relative = path.strip_prefix(root).unwrap().to_path_buf();
                files.insert(relative, fs::read(&path).unwrap());
            }
        }
    }
    let mut files = BTreeMap::new();
    walk(dir, dir, &mut files);
    files
}

fn validate_json(dir: &Path) -> (std::process::Output, serde_json::Value) {
    let output = cobre()
        .args(["validate", dir.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    let value = serde_json::from_slice(&output.stdout).unwrap();
    (output, value)
}

/// A compatible boundary (the case's own just-produced checkpoint) prints the
/// selected boundary date, then the one-line reconciliation summary, and
/// exits 0, without a solve. The per-family breakdown is gated behind
/// `RUST_LOG=debug`, so it is absent from default stdout.
#[test]
fn boundary_report_summary_prints_and_exits_0() {
    let dir = TempDir::new().unwrap();
    write_boundary_case(dir.path(), 0);
    run_case(dir.path());
    append_boundary_policy(dir.path(), &dir.path().join("output/policy"));

    let output = cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let date_pos = stdout.find("boundary policy priced at 2024-03-01");
    let summary_pos = stdout.find("boundary reconciliation:");
    assert!(
        date_pos.is_some() && summary_pos.is_some() && date_pos < summary_pos,
        "boundary date line must precede the reconciliation summary: {stdout}"
    );
    assert!(!stdout.contains("storage: COPY="));
}

/// A RELATIVE `policy.boundary.path` resolves against the CASE (input) directory,
/// not the run's output directory: `"output/policy"` points at
/// `case_dir/output/policy` (the just-produced checkpoint) and validate exits 0.
#[test]
fn boundary_relative_path_resolves_against_case_dir_not_output_dir() {
    let dir = TempDir::new().unwrap();
    write_boundary_case(dir.path(), 0);
    run_case(dir.path());
    append_boundary_policy(dir.path(), Path::new("output/policy"));

    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("boundary reconciliation:"));
}

/// `--json` emits a single, parseable JSON object carrying the per-family
/// tallies, with no human report text interleaved on stdout.
#[test]
fn boundary_json_mode_emits_parseable_object_with_tallies() {
    let dir = TempDir::new().unwrap();
    write_boundary_case(dir.path(), 0);
    run_case(dir.path());
    append_boundary_policy(dir.path(), &dir.path().join("output/policy"));

    let output = cobre()
        .args(["validate", dir.path().to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["configured"], serde_json::json!(true));
    assert_eq!(value["boundary_date"], serde_json::json!("2024-03-01"));
    assert_eq!(value["report"]["storage"]["copy"], serde_json::json!(1));
}

/// `--json` with no `policy.boundary` configured emits the explicit
/// absent-marker object, never a crash.
#[test]
fn boundary_absent_json_marks_absent_marker() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);

    let output = cobre()
        .args(["validate", dir.path().to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["configured"], serde_json::json!(false));
    assert!(value["boundary_date"].is_null());
    assert!(value["report"].is_null());
}

/// A boundary trained on a different hydro set is a validate failure:
/// non-zero exit, naming the offending hydro.
#[test]
fn boundary_mismatched_hydro_set_exits_nonzero_and_names_hydro() {
    let target_dir = TempDir::new().unwrap();
    write_boundary_case(target_dir.path(), 0);
    run_case(target_dir.path());

    let source_dir = TempDir::new().unwrap();
    write_boundary_case(source_dir.path(), 1);
    run_case(source_dir.path());

    append_boundary_policy(target_dir.path(), &source_dir.path().join("output/policy"));

    cobre()
        .args(["validate", target_dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stdout(predicate::str::contains("hydro 0"))
        .stdout(predicate::str::contains("different set of plants"));
}

/// The same mismatched-hydro-set boundary reject under `--json` emits a
/// single parseable error object whose `kind` is the shared
/// `BoundaryReconciliationError` vocabulary.
#[test]
fn boundary_mismatched_hydro_set_json_mode_reports_boundary_reconciliation_error_kind() {
    let target_dir = TempDir::new().unwrap();
    write_boundary_case(target_dir.path(), 0);
    run_case(target_dir.path());

    let source_dir = TempDir::new().unwrap();
    write_boundary_case(source_dir.path(), 1);
    run_case(source_dir.path());

    append_boundary_policy(target_dir.path(), &source_dir.path().join("output/policy"));

    let output = cobre()
        .args(["validate", target_dir.path().to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "expected validation failure");

    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(
        value["error"]["phase"],
        serde_json::json!("BoundaryReconciliationError"),
        "kind must be the shared boundary-phase vocabulary: {value}"
    );
    assert!(
        value["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("hydro 0")),
        "expected the offending hydro named in the message, got: {value}"
    );
}

/// A SOURCE boundary that prices an extra hydro the target does not model is
/// a superset, not a mismatch: `cobre validate` still exits 0, and `--json`
/// reports the extra hydro's storage slot under `dropped_source_slots`.
#[test]
fn boundary_superset_source_exits_zero_and_reports_the_drop_in_json() {
    let target_dir = TempDir::new().unwrap();
    write_boundary_case(target_dir.path(), 0);
    run_case(target_dir.path());

    let source_dir = TempDir::new().unwrap();
    write_boundary_case_with_hydros(source_dir.path(), &[0, 1]);
    run_case(source_dir.path());

    append_boundary_policy(target_dir.path(), &source_dir.path().join("output/policy"));

    let output = cobre()
        .args(["validate", target_dir.path().to_str().unwrap()])
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        !stdout.contains("warning:") && !stderr.contains("warning:"),
        "boundary path must never warn: stdout={stdout:?} stderr={stderr:?}"
    );
    assert!(
        output.status.success(),
        "stdout={stdout:?} stderr={stderr:?}"
    );

    let json_output = cobre()
        .args(["validate", target_dir.path().to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(json_output.status.success());
    let json_stdout = String::from_utf8(json_output.stdout).unwrap();
    let value: serde_json::Value = serde_json::from_str(&json_stdout).unwrap();
    let dropped = value["report"]["dropped_source_slots"].as_array().unwrap();
    assert!(
        !dropped.is_empty(),
        "dropped_source_slots must be non-empty: {json_stdout}"
    );
    assert_eq!(dropped[0]["family"], serde_json::json!("storage"));
    assert!(value["report"]["straddling_slots"].is_array());
}

/// The same superset under `policy.boundary.strict = true` is a reject:
/// `cobre validate` exits 1 and names the dropping family in its message.
#[test]
fn boundary_superset_source_under_strict_exits_nonzero_and_names_the_family() {
    let target_dir = TempDir::new().unwrap();
    write_boundary_case(target_dir.path(), 0);
    run_case(target_dir.path());

    let source_dir = TempDir::new().unwrap();
    write_boundary_case_with_hydros(source_dir.path(), &[0, 1]);
    run_case(source_dir.path());

    append_boundary_policy_with_strict(
        target_dir.path(),
        &source_dir.path().join("output/policy"),
        true,
    );

    let output = cobre()
        .args(["validate", target_dir.path().to_str().unwrap()])
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        !stdout.contains("warning:") && !stderr.contains("warning:"),
        "boundary path must never warn: stdout={stdout:?} stderr={stderr:?}"
    );
    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout={stdout:?} stderr={stderr:?}"
    );
    assert!(
        stdout.contains("price entities this study does not model"),
        "stdout={stdout:?}"
    );
    assert!(stdout.contains("(storage: 1)"), "stdout={stdout:?}");
    assert!(
        stdout.contains("policy.boundary.strict"),
        "stdout={stdout:?}"
    );
}

#[test]
fn fpha_hydro_without_production_models_json_stdout_mentions_file() {
    let dir = TempDir::new().unwrap();
    write_fpha_hydro_without_production_models_case(&dir);

    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stdout(predicate::str::contains("hydro_production_models.json"));
}

// ── non-boundary scalar-parameter presence guard ───────────────────────────────

/// A non-boundary deck whose scalar-parameter table has a resolution gap (a
/// `seasonal` param with no entry for the resolved season) exits 1, honoring
/// the module contract that a clean `validate` implies a clean pre-solver `run`.
/// The `--json` kind and message are identical to `cobre.io.validate`'s.
#[test]
fn non_boundary_scalar_parameter_gap_is_rejected() {
    let dir = TempDir::new().unwrap();
    copy_dir_recursive(&common::case_dir("1dtoy"), dir.path());
    write_file(
        dir.path(),
        "constraints/generic_parameters.json",
        r#"{"scalar_parameters": [{"id": 1, "name": "p_gap", "kind": "seasonal", "values": [[5, 1.0]]}]}"#,
    );

    let output = cobre()
        .args(["validate", dir.path().to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "gap deck must exit 1");
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(
        value["error"]["phase"],
        serde_json::json!("GenericConstraintValidationError"),
        "kind must match cobre.io.validate: {value}"
    );
    assert_eq!(
        value["error"]["message"],
        serde_json::json!(
            "constraints/: configuration validation error: parameter 'p_gap': no seasonal value for season_id=0 (needed by stage 0)"
        ),
        "message must match cobre.io.validate byte-for-byte: {value}"
    );
}

/// The same non-boundary deck with a fully-resolved scalar parameter still
/// validates and exits 0 — the guard raises no false rejection.
#[test]
fn non_boundary_resolved_scalar_parameter_validates() {
    let dir = TempDir::new().unwrap();
    copy_dir_recursive(&common::case_dir("1dtoy"), dir.path());
    write_file(
        dir.path(),
        "constraints/generic_parameters.json",
        r#"{"scalar_parameters": [{"id": 1, "name": "p_ok", "kind": "constant", "value": 1.0}]}"#,
    );

    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .success();
}

/// A checkpoint written by another cobre version is refused at warm-start
/// load, naming both versions and ending with the re-run remedy.
#[test]
fn warm_start_refuses_a_policy_written_by_another_version() {
    let dir = TempDir::new().unwrap();
    write_boundary_case(dir.path(), 0);
    run_case(dir.path());
    append_warm_start_policy(dir.path());

    cobre()
        .args(["run", dir.path().to_str().unwrap()])
        .assert()
        .success();

    common::restamp_policy_version(&dir.path().join("output/policy"), "0.0.1");

    cobre()
        .args(["run", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stderr(predicate::str::contains("written by cobre 0.0.1"))
        .stderr(predicate::str::contains(format!(
            "this is cobre {}",
            env!("CARGO_PKG_VERSION")
        )))
        .stderr(predicate::str::contains(
            "re-run the program that produced it with cobre",
        ))
        .stderr(predicate::str::contains(
            "for a converted boundary policy, convert it again",
        ))
        .stderr(predicate::str::contains("re-export").not());
}

/// A checkpoint written by another program at this build's exact version is
/// refused at warm-start load, naming that program.
#[test]
fn warm_start_refuses_a_policy_written_by_other_software() {
    let dir = TempDir::new().unwrap();
    write_boundary_case(dir.path(), 0);
    run_case(dir.path());
    append_warm_start_policy(dir.path());

    common::restamp_policy_software(&dir.path().join("output/policy"), "another-program");

    cobre()
        .args(["run", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stderr(predicate::str::contains(format!(
            "written by another-program {}",
            env!("CARGO_PKG_VERSION")
        )));
}

/// A boundary source written by another cobre version is refused at run,
/// naming both versions.
#[test]
fn boundary_policy_written_by_another_version_is_refused_at_run() {
    let dir = TempDir::new().unwrap();
    write_boundary_case(dir.path(), 0);
    run_case(dir.path());
    let boundary_policy_dir = dir.path().join("output/policy");
    append_boundary_policy(dir.path(), &boundary_policy_dir);

    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            dir.path().join("boundary_ok").to_str().unwrap(),
        ])
        .assert()
        .success();

    common::restamp_policy_version(&boundary_policy_dir, "0.0.1");

    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            dir.path().join("boundary_refused").to_str().unwrap(),
        ])
        .assert()
        .failure()
        .code(1)
        .stderr(predicate::str::contains("written by cobre 0.0.1"))
        .stderr(predicate::str::contains(format!(
            "this is cobre {}",
            env!("CARGO_PKG_VERSION")
        )));
}

/// A policy stamped by another program is refused by `cobre run`, and `cobre
/// validate` reproduces the refusal without touching the policy directory.
#[test]
fn validate_refuses_a_warm_start_policy_written_by_other_software() {
    let dir = TempDir::new().unwrap();
    write_boundary_case(dir.path(), 0);
    run_case(dir.path());
    append_warm_start_policy(dir.path());
    common::restamp_policy_software(&dir.path().join("output/policy"), "another-program");

    cobre()
        .args(["run", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stderr(predicate::str::contains("written by another-program"))
        .stderr(predicate::str::contains("run `cobre validate <CASE_DIR>`"));

    let before = snapshot_files(&dir.path().join("output/policy"));
    let (output, value) = validate_json(dir.path());
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(value["error"]["phase"], "WarmStartIncompatible");
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("written by another-program"),
        "got: {value}"
    );
    assert_eq!(snapshot_files(&dir.path().join("output/policy")), before);
}

/// A checkpoint written by another cobre version is refused by `validate` at
/// resume load, under the resume kind.
#[test]
fn validate_refuses_a_resume_policy_written_by_another_version() {
    let dir = TempDir::new().unwrap();
    write_boundary_case(dir.path(), 0);
    run_case(dir.path());
    append_resume_policy(dir.path());
    common::restamp_policy_version(&dir.path().join("output/policy"), "0.0.1");

    let (output, value) = validate_json(dir.path());
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(value["error"]["phase"], "ResumeIncompatible");
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("written by cobre 0.0.1"),
        "got: {value}"
    );
}

/// A simulation-only load shares the warm-start kind: there is no third kind.
#[test]
fn validate_refuses_a_simulation_only_policy_under_the_warm_start_kind() {
    let dir = TempDir::new().unwrap();
    write_boundary_case(dir.path(), 0);
    run_case(dir.path());
    append_simulation_only_policy(dir.path());
    common::restamp_policy_version(&dir.path().join("output/policy"), "0.0.1");

    let (output, value) = validate_json(dir.path());
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(value["error"]["phase"], "WarmStartIncompatible");
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("written by cobre 0.0.1"),
        "got: {value}"
    );
}

#[test]
fn validate_refuses_a_missing_policy_directory_without_creating_it() {
    let dir = TempDir::new().unwrap();
    write_boundary_case(dir.path(), 0);
    append_warm_start_policy(dir.path());

    let (output, value) = validate_json(dir.path());
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(value["error"]["phase"], "WarmStartIncompatible");
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Policy directory not found"),
        "got: {value}"
    );
    assert!(!dir.path().join("output").exists());
}

/// A policy file the process cannot open is an OS failure, not a refusal of the
/// case: exit 2 under the `IoError` phase.
#[cfg(unix)]
#[test]
fn validate_reports_an_unreadable_policy_manifest_as_an_io_error() {
    use std::os::unix::fs::PermissionsExt;

    let dir = TempDir::new().unwrap();
    write_boundary_case(dir.path(), 0);
    run_case(dir.path());
    append_warm_start_policy(dir.path());
    let manifest = dir.path().join("output/policy/manifest.bin");
    fs::set_permissions(&manifest, fs::Permissions::from_mode(0o000)).unwrap();
    if fs::read(&manifest).is_ok() {
        return;
    }

    let (output, value) = validate_json(dir.path());
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(value["error"]["phase"], "IoError");
}

#[test]
fn validate_json_has_no_policy_load_without_a_configured_policy() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);

    let (output, value) = validate_json(dir.path());
    assert_eq!(output.status.code(), Some(0));
    assert!(!value.as_object().unwrap().contains_key("policy_load"));
}

/// An accepted warm-start or resume policy reports its mode and no unused bases.
#[test]
fn validate_json_reports_the_accepted_policy_load_mode() {
    let dir = TempDir::new().unwrap();
    write_boundary_case(dir.path(), 0);
    run_case(dir.path());

    for mode in ["warm_start", "resume"] {
        append_policy_mode(dir.path(), mode);
        let (output, value) = validate_json(dir.path());
        assert_eq!(output.status.code(), Some(0), "{mode}: {value}");
        assert_eq!(
            value["policy_load"],
            serde_json::json!({ "mode": mode, "unused_stored_bases": 0 }),
            "{mode}"
        );
    }
}

/// A study that supplies its opening tree from a file validates although its
/// `historical_residuals` stages have no inflow history to build a library from.
#[test]
fn validate_accepts_a_supplied_opening_tree_with_historical_residuals_stages() {
    let case = TempDir::new().unwrap();
    write_supplied_opening_tree_case(case.path());

    cobre()
        .args(["validate", case.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("historical windows").not())
        .stdout(predicate::str::contains("V2.").not())
        .stderr(predicate::str::contains("historical windows").not())
        .stderr(predicate::str::contains("V2.").not());
}

/// `cobre validate --output <DIR>` checks the policy `cobre run --output <DIR>`
/// loads, where the default `<CASE_DIR>/output/` holds none.
#[test]
fn validate_output_flag_checks_the_policy_run_loads_from_that_directory() {
    let dir = TempDir::new().unwrap();
    let case = dir.path().to_str().unwrap();
    let custom = dir.path().join("custom");
    write_boundary_case(dir.path(), 0);
    cobre()
        .args(["run", case, "--output", custom.to_str().unwrap()])
        .assert()
        .success();
    append_warm_start_policy(dir.path());
    common::restamp_policy_software(&custom.join("policy"), "another-program");

    cobre()
        .args(["run", case, "--output", custom.to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stderr(predicate::str::contains("written by another-program"));

    let output = cobre()
        .args([
            "validate",
            case,
            "--output",
            custom.to_str().unwrap(),
            "--json",
        ])
        .output()
        .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(value["error"]["phase"], "WarmStartIncompatible");
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("written by another-program"),
        "got: {value}"
    );

    let (output, value) = validate_json(dir.path());
    assert_eq!(output.status.code(), Some(1));
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Policy directory not found"),
        "got: {value}"
    );
}

/// Validate reads `--output` and never creates it, whether or not a policy load
/// is configured.
#[test]
fn validate_output_flag_never_creates_the_directory() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let absent = dir.path().join("absent");

    cobre()
        .args([
            "validate",
            dir.path().to_str().unwrap(),
            "--output",
            absent.to_str().unwrap(),
        ])
        .assert()
        .success();
    assert!(!absent.exists());

    append_warm_start_policy(dir.path());
    cobre()
        .args([
            "validate",
            dir.path().to_str().unwrap(),
            "--output",
            absent.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .code(1)
        .stdout(predicate::str::contains("Policy directory not found"));
    assert!(!absent.exists());
}

/// A relative `--output` resolves against the working directory in `validate`,
/// exactly as in `run`.
#[test]
fn validate_output_relative_path_resolves_against_the_working_directory() {
    let case = TempDir::new().unwrap();
    let cwd = TempDir::new().unwrap();
    let case_arg = case.path().to_str().unwrap();
    write_boundary_case(case.path(), 0);
    cobre()
        .current_dir(cwd.path())
        .args(["run", case_arg, "--output", "rel_out"])
        .assert()
        .success();
    append_warm_start_policy(case.path());
    common::restamp_policy_software(&cwd.path().join("rel_out/policy"), "another-program");

    let output = cobre()
        .current_dir(cwd.path())
        .args(["validate", case_arg, "--output", "rel_out", "--json"])
        .output()
        .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("written by another-program"),
        "got: {value}"
    );
}

#[test]
fn validate_help_lists_the_output_flag() {
    cobre()
        .args(["validate", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--output"))
        .stdout(predicate::str::contains("<DIR>"));
}
