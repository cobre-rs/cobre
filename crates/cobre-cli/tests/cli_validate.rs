//! Integration tests for the `cobre validate` subcommand.

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
        "selection": { "method": "sampled", "forward_passes": 10 },
        "stopping_rules": [
            { "type": "iteration_limit", "limit": 100 }
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

// These tests cover failures that pass the six-layer IO pipeline but are caught
// only by the three additional phases (StudyParams::from_config, prepare_stochastic,
// prepare_hydro_models_from_artifacts) that `validate` exercises.

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

fn write_boundary_case(dir: &Path, hydro_id: i64) {
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
    write_file(
        dir,
        "system/hydros.json",
        &format!(
            r#"{{
                "hydros": [
                    {{
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
                    }}
                ]
            }}"#
        ),
    );
    write_file(
        dir,
        "system/hydro_production_models.json",
        &format!(
            r#"{{
                "production_models": [
                    {{
                        "hydro_id": {hydro_id}, "selection_mode": "stage_ranges",
                        "stage_ranges": [
                            {{
                                "start_stage_id": 0, "end_stage_id": null,
                                "model": "constant_productivity", "productivity_mw_per_m3s": 1.0
                            }}
                        ]
                    }}
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
    write_file(
        dir,
        "initial_conditions.json",
        &format!(
            r#"{{ "storage": [{{ "hydro_id": {hydro_id}, "value_hm3": 100.0 }}], "filling_storage": [] }}"#
        ),
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
                "storage_violation_below_cost": 500.0, "filling_target_violation_cost": 500.0,
                "turbined_violation_below_cost": 500.0, "outflow_violation_below_cost": 500.0,
                "outflow_violation_above_cost": 500.0, "generation_violation_below_cost": 500.0,
                "evaporation_violation_cost": 500.0, "water_withdrawal_violation_cost": 500.0
            },
            "non_controllable_source": { "curtailment_cost": 0.005 }
        }"#,
    );
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
            "policy": {{ "boundary": {{ "path": "{boundary_path}", "source_stage": 1 }} }}
        }}"#
    );
    write_file(dir, "config.json", &config);
}

/// A compatible boundary (the case's own just-produced checkpoint) prints the
/// one-line reconciliation summary and exits 0, without a solve. The per-family
/// breakdown is gated behind `RUST_LOG=debug`, so it is absent from default stdout.
#[test]
fn boundary_report_summary_prints_and_exits_0() {
    let dir = TempDir::new().unwrap();
    write_boundary_case(dir.path(), 0);
    run_case(dir.path());
    append_boundary_policy(dir.path(), &dir.path().join("output/policy"));

    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("boundary reconciliation:"))
        .stdout(predicate::str::contains("storage: COPY=").not());
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
