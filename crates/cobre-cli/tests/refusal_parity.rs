//! Table-driven parity between `cobre validate` and `cobre run` refusals.
//!
//! One table, three outcomes: each `ParityRow` in `ROWS` mutates a committed
//! example, declares its `Outcome` (bracketed refusal, plain refusal or
//! warning) and a message fragment, and both commands must report the same
//! line from the outcome's anchor onward. Add a row here and in
//! `crates/cobre-python/tests/test_refusal_parity.py`; the checker does not
//! change when rows are added.

#![allow(clippy::unwrap_used, clippy::panic)]

use std::fs;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float64Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use serde_json::{Value, json};
use tempfile::TempDir;

mod common;
use common::{case_dir, cobre, copy_dir_recursive, restamp_policy_version};

enum Outcome {
    BracketedRefusal { kind: &'static str },
    PlainRefusal,
    Warning,
}

impl Outcome {
    fn exit_code(&self) -> i32 {
        match self {
            Outcome::BracketedRefusal { .. } | Outcome::PlainRefusal => 1,
            Outcome::Warning => 0,
        }
    }

    fn anchor(&self, fragment: &str) -> String {
        match self {
            Outcome::BracketedRefusal { kind } => format!("[{kind}]"),
            Outcome::PlainRefusal | Outcome::Warning => fragment.to_string(),
        }
    }

    fn marker(&self) -> Option<&'static str> {
        match self {
            Outcome::Warning => Some("warning:"),
            Outcome::BracketedRefusal { .. } | Outcome::PlainRefusal => None,
        }
    }
}

struct ParityRow {
    name: &'static str,
    base_case: &'static str,
    mutate: fn(&Path),
    outcome: Outcome,
    fragment: &'static str,
}

const ROWS: &[ParityRow] = &[
    ParityRow {
        name: "travel_time_negative",
        base_case: "deterministic/d44-travel-time-substage",
        mutate: negative_travel_time,
        outcome: Outcome::BracketedRefusal {
            kind: "InvalidValue",
        },
        fragment: "travel_time_hours must be finite and >= 0.0",
    },
    ParityRow {
        name: "travel_time_release_before_downstream_entry",
        base_case: "deterministic/d44-travel-time-substage",
        mutate: release_before_downstream_entry,
        outcome: Outcome::BracketedRefusal {
            kind: "BusinessRuleViolation",
        },
        fragment: "has not reached Operating status there",
    },
    ParityRow {
        name: "pumping_station_active_after_endpoint_exit",
        base_case: "deterministic/d35-pumping-commissioning",
        mutate: pumping_endpoint_exits_while_station_active,
        outcome: Outcome::BracketedRefusal {
            kind: "BusinessRuleViolation",
        },
        fragment: "is not Operating there",
    },
    ParityRow {
        name: "season_overlap_within_one_level",
        base_case: "deterministic/d30-multi-resolution-monthly-quarterly",
        mutate: duplicate_january_season,
        outcome: Outcome::BracketedRefusal {
            kind: "SchemaViolation",
        },
        fragment: "overlap within one resolution level",
    },
    ParityRow {
        name: "generic_constraint_repeated_block_argument",
        base_case: "deterministic/d13-generic-constraint",
        mutate: repeat_generic_constraint_block_argument,
        outcome: Outcome::BracketedRefusal {
            kind: "SchemaViolation",
        },
        fragment: "repeated block argument in variable",
    },
    ParityRow {
        name: "policy_path_empty",
        base_case: "1dtoy",
        mutate: empty_policy_path,
        outcome: Outcome::BracketedRefusal {
            kind: "SchemaViolation",
        },
        fragment: "names the output directory or one of its ancestors",
    },
    ParityRow {
        name: "policy_path_current_directory",
        base_case: "1dtoy",
        mutate: current_directory_policy_path,
        outcome: Outcome::BracketedRefusal {
            kind: "SchemaViolation",
        },
        fragment: "names the output directory or one of its ancestors",
    },
    ParityRow {
        name: "policy_path_parent_directory",
        base_case: "1dtoy",
        mutate: parent_directory_policy_path,
        outcome: Outcome::BracketedRefusal {
            kind: "SchemaViolation",
        },
        fragment: "names the output directory or one of its ancestors",
    },
    ParityRow {
        name: "policy_path_naming_the_output_directory",
        base_case: "1dtoy",
        mutate: output_directory_policy_path,
        outcome: Outcome::PlainRefusal,
        fragment: "names the output directory or one of its ancestors",
    },
    ParityRow {
        name: "policy_path_climbing_back_into_the_output_directory",
        base_case: "1dtoy",
        mutate: climbing_back_into_the_output_directory_policy_path,
        outcome: Outcome::PlainRefusal,
        fragment: "names the output directory or one of its ancestors",
    },
    ParityRow {
        name: "policy_directory_holding_an_unrecognized_file",
        base_case: "1dtoy",
        mutate: unrecognized_file_in_the_policy_directory,
        outcome: Outcome::PlainRefusal,
        fragment: "is not part of a checkpoint",
    },
    ParityRow {
        name: "policy_path_inside_a_simulation_family_directory",
        base_case: "1dtoy",
        mutate: simulation_family_child_policy_path,
        outcome: Outcome::PlainRefusal,
        fragment: "lies inside simulation/costs, which a run removes whole",
    },
    ParityRow {
        name: "policy_path_naming_the_simulation_solver_directory",
        base_case: "1dtoy",
        mutate: simulation_solver_policy_path,
        outcome: Outcome::PlainRefusal,
        fragment: "names simulation/solver, which holds files a run writes",
    },
    ParityRow {
        name: "policy_path_containing_a_cleared_directory",
        base_case: "1dtoy",
        mutate: simulation_policy_path,
        outcome: Outcome::PlainRefusal,
        fragment: "contains simulation/costs, which a run removes whole",
    },
    ParityRow {
        name: "policy_path_containing_the_training_solver_directory",
        base_case: "1dtoy",
        mutate: training_policy_path,
        outcome: Outcome::PlainRefusal,
        fragment: "contains training/solver, which holds files a run writes",
    },
    ParityRow {
        name: "historical_forward_scheme_without_inflow_history",
        base_case: "1dtoy",
        mutate: historical_forward_scheme,
        outcome: Outcome::PlainRefusal,
        fragment: "no valid historical windows found",
    },
    ParityRow {
        name: "historical_forward_scheme_with_a_zero_deviation_season",
        base_case: "deterministic/d26-estimated-par2",
        mutate: historical_forward_scheme_with_a_zero_deviation_season,
        outcome: Outcome::PlainRefusal,
        fragment: "V2.3: historical library contains non-finite eta",
    },
    ParityRow {
        name: "stopping_rules_without_iteration_limit_rule",
        base_case: "deterministic/d01-thermal-dispatch",
        mutate: time_limit_only_stopping_rules,
        outcome: Outcome::BracketedRefusal {
            kind: "SchemaViolation",
        },
        fragment: "field training.stopping_rules: must contain an iteration_limit rule",
    },
    ParityRow {
        name: "empty_stopping_rules",
        base_case: "deterministic/d01-thermal-dispatch",
        mutate: empty_stopping_rules,
        outcome: Outcome::BracketedRefusal {
            kind: "SchemaViolation",
        },
        fragment: "field training.stopping_rules: must contain an iteration_limit rule",
    },
    ParityRow {
        name: "warm_start_policy_from_another_version",
        base_case: "1dtoy",
        mutate: warm_start_policy_from_another_version,
        outcome: Outcome::PlainRefusal,
        fragment: "policy was written by",
    },
    ParityRow {
        name: "resume_policy_from_another_version",
        base_case: "1dtoy",
        mutate: resume_policy_from_another_version,
        outcome: Outcome::PlainRefusal,
        fragment: "policy was written by",
    },
    ParityRow {
        name: "simulation_only_policy_from_another_version",
        base_case: "1dtoy",
        mutate: simulation_only_policy_from_another_version,
        outcome: Outcome::PlainRefusal,
        fragment: "policy was written by",
    },
    ParityRow {
        name: "warm_start_without_a_policy_directory",
        base_case: "1dtoy",
        mutate: warm_start_without_a_policy_directory,
        outcome: Outcome::PlainRefusal,
        fragment: "Policy directory not found",
    },
    ParityRow {
        name: "simulation_only_policy_with_unused_stored_bases",
        base_case: "1dtoy",
        mutate: simulation_only_policy_with_unused_stored_bases,
        outcome: Outcome::Warning,
        fragment: "stored bases not used",
    },
];

fn edit_json(path: &Path, edit: impl FnOnce(&mut Value)) {
    let mut value: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    edit(&mut value);
    fs::write(path, serde_json::to_string_pretty(&value).unwrap()).unwrap();
}

fn negative_travel_time(case: &Path) {
    edit_json(&case.join("system/hydros.json"), |hydros| {
        hydros["hydros"][0]["travel_time_hours"] = json!(-1.0);
    });
}

fn release_before_downstream_entry(case: &Path) {
    edit_json(&case.join("system/hydros.json"), |hydros| {
        hydros["hydros"][1]["entry_stage_id"] = json!(1);
    });
}

fn pumping_endpoint_exits_while_station_active(case: &Path) {
    edit_json(&case.join("system/hydros.json"), |hydros| {
        hydros["hydros"][1]["exit_stage_id"] = json!(1);
    });
}

fn duplicate_january_season(case: &Path) {
    edit_json(&case.join("stages.json"), |stages| {
        stages["season_definitions"]["seasons"]
            .as_array_mut()
            .unwrap()
            .push(json!({
                "id": 16,
                "label": "January bis",
                "month_start": 1,
                "day_start": 1,
                "month_end": 1,
                "day_end": 31
            }));
    });
}

fn repeat_generic_constraint_block_argument(case: &Path) {
    edit_json(
        &case.join("constraints/generic_constraints.json"),
        |constraints| {
            constraints["constraints"][0]["expression"] = json!("thermal_generation(0, 0, 0)");
        },
    );
}

fn set_policy_path(case: &Path, policy_path: &str) {
    edit_json(&case.join("config.json"), |config| {
        config["policy"]["path"] = json!(policy_path);
    });
}

fn empty_policy_path(case: &Path) {
    set_policy_path(case, "");
}

fn current_directory_policy_path(case: &Path) {
    set_policy_path(case, ".");
}

fn parent_directory_policy_path(case: &Path) {
    set_policy_path(case, "..");
}

fn output_directory_policy_path(case: &Path) {
    set_policy_path(case, case.join("output").to_str().unwrap());
}

fn climbing_back_into_the_output_directory_policy_path(case: &Path) {
    set_policy_path(case, "../output");
}

fn unrecognized_file_in_the_policy_directory(case: &Path) {
    let policy_dir = case.join("output/policy");
    fs::create_dir_all(&policy_dir).unwrap();
    fs::write(policy_dir.join("notes.txt"), "kept by the user").unwrap();
}

fn simulation_family_child_policy_path(case: &Path) {
    set_policy_path(case, "simulation/costs/policy");
}

fn simulation_solver_policy_path(case: &Path) {
    set_policy_path(case, "simulation/solver");
}

fn simulation_policy_path(case: &Path) {
    set_policy_path(case, "simulation");
}

fn training_policy_path(case: &Path) {
    set_policy_path(case, "training");
}

fn historical_forward_scheme(case: &Path) {
    edit_json(&case.join("config.json"), |config| {
        config["training"]["scenario_source"]["inflow"] = json!({"scheme": "historical"});
    });
}

fn historical_forward_scheme_with_a_zero_deviation_season(case: &Path) {
    edit_json(&case.join("config.json"), |config| {
        config["training"]["scenario_source"] =
            json!({"seed": 1, "inflow": {"scheme": "historical"}});
        config["estimation"] = json!({"max_order": 0});
    });
    write_inflow_seasonal_stats_with_zero_deviation_at_stage_3(
        &case.join("scenarios/inflow_seasonal_stats.parquet"),
    );
}

fn write_inflow_seasonal_stats_with_zero_deviation_at_stage_3(path: &Path) {
    let stage_ids: Vec<i32> = (-2..=11).collect();
    let rows = stage_ids.len();
    let std_m3s: Vec<f64> = stage_ids
        .iter()
        .map(|&stage| if stage == 3 { 0.0 } else { 50.0 })
        .collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("mean_m3s", DataType::Float64, false),
        Field::new("std_m3s", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![0; rows])),
            Arc::new(Int32Array::from(stage_ids)),
            Arc::new(Float64Array::from(vec![200.0; rows])),
            Arc::new(Float64Array::from(std_m3s)),
        ],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(fs::File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn time_limit_only_stopping_rules(case: &Path) {
    edit_json(&case.join("config.json"), |config| {
        config["training"]["stopping_rules"] = json!([{"type": "time_limit", "seconds": 600}]);
    });
}

fn empty_stopping_rules(case: &Path) {
    edit_json(&case.join("config.json"), |config| {
        config["training"]["stopping_rules"] = json!([]);
    });
}

fn train_one_iteration(case: &Path) {
    edit_json(&case.join("config.json"), |config| {
        config["training"]["stopping_rules"] = json!([{"type": "iteration_limit", "limit": 1}]);
        config["simulation"]["enabled"] = json!(false);
    });
    let trained = cobre()
        .arg("run")
        .arg(case)
        .arg("--quiet")
        .output()
        .unwrap();
    assert!(
        trained.status.success(),
        "training the policy failed: {}",
        String::from_utf8_lossy(&trained.stderr)
    );
}

fn set_policy_mode(case: &Path, mode: &str) {
    edit_json(&case.join("config.json"), |config| {
        config["policy"]["mode"] = json!(mode);
    });
}

fn select_simulation_only(case: &Path) {
    edit_json(&case.join("config.json"), |config| {
        config["training"]["enabled"] = json!(false);
        config["simulation"]["enabled"] = json!(true);
        config["simulation"]["selection"] = json!({"method": "sampled", "num_scenarios": 1});
    });
}

fn warm_start_policy_from_another_version(case: &Path) {
    train_one_iteration(case);
    restamp_policy_version(&case.join("output/policy"), "0.0.1");
    set_policy_mode(case, "warm_start");
}

fn resume_policy_from_another_version(case: &Path) {
    train_one_iteration(case);
    restamp_policy_version(&case.join("output/policy"), "0.0.1");
    set_policy_mode(case, "resume");
}

fn simulation_only_policy_from_another_version(case: &Path) {
    train_one_iteration(case);
    restamp_policy_version(&case.join("output/policy"), "0.0.1");
    select_simulation_only(case);
}

fn warm_start_without_a_policy_directory(case: &Path) {
    set_policy_mode(case, "warm_start");
}

fn simulation_only_policy_with_unused_stored_bases(case: &Path) {
    let thermals = case.join("system/thermals.json");
    edit_json(&thermals, |thermals| {
        thermals["thermals"].as_array_mut().unwrap().push(json!({
            "id": 2,
            "name": "UTE3",
            "operational_start_date": "2020-01-01",
            "bus_id": 0,
            "generation": {"min_mw": 0.0, "max_mw": 15.0},
            "cost_per_mwh": 20.0
        }));
    });
    train_one_iteration(case);
    fs::copy(case_dir("1dtoy").join("system/thermals.json"), thermals).unwrap();
    select_simulation_only(case);
}

struct Observed<'a> {
    code: Option<i32>,
    text: &'a str,
}

fn reported_tail(text: &str, fragment: &str, anchor: &str, marker: Option<&str>) -> Option<String> {
    let line = text
        .lines()
        .find(|line| line.contains(fragment) && marker.is_none_or(|m| line.contains(m)))?;
    let start = line.find(anchor)?;
    Some(line[start..].trim_end().to_string())
}

fn parity_violations(
    outcome: &Outcome,
    fragment: &str,
    validate: &Observed,
    run: &Observed,
) -> Vec<String> {
    let anchor = outcome.anchor(fragment);
    let expected = outcome.exit_code();
    let tail =
        |observed: &Observed| reported_tail(observed.text, fragment, &anchor, outcome.marker());
    let validate_tail = tail(validate);
    let run_tail = tail(run);

    let mut violations = Vec::new();
    if validate.code != Some(expected) {
        violations.push(format!(
            "validate exited {:?}, expected {expected}",
            validate.code
        ));
    }
    if run.code != Some(expected) {
        violations.push(format!("run exited {:?}, expected {expected}", run.code));
    }
    for (command, observed) in [("validate", validate), ("run", run)] {
        if observed.text.contains("report this at") {
            violations.push(format!("{command} printed bug-report text"));
        }
    }
    if run_tail.is_none() {
        violations.push(format!(
            "run reported no line containing {fragment:?} from {anchor:?}"
        ));
    }
    match (&validate_tail, &run_tail) {
        (None, _) => violations.push(format!(
            "validate reported no line containing {fragment:?} from {anchor:?}"
        )),
        (Some(v), Some(r)) if v != r => violations.push(format!(
            "validate and run reported different lines: validate {v:?}, run {r:?}"
        )),
        _ => {}
    }
    if run.text.contains("run `cobre validate")
        && (validate.code != Some(1) || validate_tail.is_none())
    {
        violations.push(
            "run printed the validate hint but validate does not reproduce the refusal".into(),
        );
    }
    violations
}

fn mutated_case(row: &ParityRow) -> TempDir {
    let dir = TempDir::new().unwrap();
    copy_dir_recursive(&case_dir(row.base_case), dir.path());
    (row.mutate)(dir.path());
    dir
}

fn cli_violations(row: &ParityRow, validate_case: &Path, run_case: &Path) -> Vec<String> {
    let validate = cobre()
        .arg("validate")
        .arg(validate_case)
        .output()
        .unwrap_or_else(|e| panic!("{}: cobre validate failed to spawn: {e}", row.name));
    let run = cobre()
        .arg("run")
        .arg(run_case)
        .output()
        .unwrap_or_else(|e| panic!("{}: cobre run failed to spawn: {e}", row.name));
    let validate_text = String::from_utf8_lossy(&validate.stdout);
    let run_text = String::from_utf8_lossy(&run.stderr);
    parity_violations(
        &row.outcome,
        row.fragment,
        &Observed {
            code: validate.status.code(),
            text: &validate_text,
        },
        &Observed {
            code: run.status.code(),
            text: &run_text,
        },
    )
}

#[test]
fn validate_and_run_report_identically() {
    for row in ROWS {
        let case = mutated_case(row);
        let violations = cli_violations(row, case.path(), case.path());
        assert!(
            violations.is_empty(),
            "{}: {}",
            row.name,
            violations.join("; ")
        );
    }
}

#[test]
fn parity_check_flags_a_refusal_validate_does_not_reproduce() {
    let row = &ROWS[0];
    let mutated = mutated_case(row);
    let unmutated = TempDir::new().unwrap();
    copy_dir_recursive(&case_dir(row.base_case), unmutated.path());

    let violations = cli_violations(row, unmutated.path(), mutated.path());

    assert!(
        violations.iter().any(|v| v.contains("does not reproduce")),
        "expected a `does not reproduce` violation, got {violations:?}"
    );
}

const HINT: &str = "  -> run `cobre validate <CASE_DIR>` for a full diagnostic report\n";
const BRACKETED_FRAGMENT: &str = "travel_time_hours must be finite";
const PLAIN_FRAGMENT: &str = "V2.1: seasonless historical stage";
const WARNING_FRAGMENT: &str = "stored basis is stale";

fn synthetic_violations(
    outcome: &Outcome,
    fragment: &str,
    validate: (i32, &str),
    run: (i32, &str),
) -> Vec<String> {
    parity_violations(
        outcome,
        fragment,
        &Observed {
            code: Some(validate.0),
            text: validate.1,
        },
        &Observed {
            code: Some(run.0),
            text: run.1,
        },
    )
}

#[test]
fn parity_check_holds_for_each_outcome_shape() {
    let bracketed = synthetic_violations(
        &Outcome::BracketedRefusal {
            kind: "InvalidValue",
        },
        BRACKETED_FRAGMENT,
        (
            1,
            "Validation: 1 errors, 0 warnings in /case\n\
             error: [InvalidValue] system/hydros.json (Hydro 0): Hydro 0: travel_time_hours must be finite and >= 0.0, got -1\n",
        ),
        (
            1,
            &format!(
                "Loading case: /case\n\
                 error: constraint violation: [InvalidValue] system/hydros.json (Hydro 0): Hydro 0: travel_time_hours must be finite and >= 0.0, got -1\n{HINT}"
            ),
        ),
    );
    assert!(bracketed.is_empty(), "bracketed: {bracketed:?}");

    let plain = synthetic_violations(
        &Outcome::PlainRefusal,
        PLAIN_FRAGMENT,
        (
            1,
            "Validation: 1 errors, 0 warnings in /case\n\
             error: stages.json: stochastic error: insufficient data: V2.1: seasonless historical stages for hydro 3\n",
        ),
        (
            1,
            &format!(
                "Loading case: /case\n\
                 error: stochastic error: insufficient data: V2.1: seasonless historical stages for hydro 3\n{HINT}"
            ),
        ),
    );
    assert!(plain.is_empty(), "plain: {plain:?}");

    let warning = synthetic_violations(
        &Outcome::Warning,
        WARNING_FRAGMENT,
        (
            0,
            "Validation: 0 errors, 1 warnings in /case\n\
             warning: policy/manifest.bin (basis 2): stored basis is stale\n",
        ),
        (0, "Loading case: /case\nwarning: stored basis is stale\n"),
    );
    assert!(warning.is_empty(), "warning: {warning:?}");
}

#[test]
fn parity_check_flags_each_outcome_shape_mismatch() {
    let bracket_missing_from_run = synthetic_violations(
        &Outcome::BracketedRefusal {
            kind: "InvalidValue",
        },
        BRACKETED_FRAGMENT,
        (
            1,
            "error: [InvalidValue] system/hydros.json (Hydro 0): Hydro 0: travel_time_hours must be finite\n",
        ),
        (
            1,
            &format!(
                "error: constraint violation: Hydro 0: travel_time_hours must be finite\n{HINT}"
            ),
        ),
    );
    assert!(
        bracket_missing_from_run
            .iter()
            .any(|v| v.contains("run reported no line")),
        "{bracket_missing_from_run:?}"
    );

    let plain_tails_differ = synthetic_violations(
        &Outcome::PlainRefusal,
        PLAIN_FRAGMENT,
        (
            1,
            "error: stages.json: stochastic error: insufficient data: V2.1: seasonless historical stages for hydro 3\n",
        ),
        (
            1,
            &format!(
                "error: stochastic error: insufficient data: V2.1: seasonless historical stages for hydro 4\n{HINT}"
            ),
        ),
    );
    assert!(
        plain_tails_differ
            .iter()
            .any(|v| v.contains("reported different lines")),
        "{plain_tails_differ:?}"
    );

    let warning_run_fails = synthetic_violations(
        &Outcome::Warning,
        WARNING_FRAGMENT,
        (
            0,
            "warning: policy/manifest.bin (basis 2): stored basis is stale\n",
        ),
        (1, "warning: stored basis is stale\n"),
    );
    assert!(
        warning_run_fails.iter().any(|v| v.contains("run exited")),
        "{warning_run_fails:?}"
    );

    let warning_absent_from_validate = synthetic_violations(
        &Outcome::Warning,
        WARNING_FRAGMENT,
        (
            0,
            "Validation: 0 errors, 0 warnings in /case\nnote: stored basis is stale\n",
        ),
        (0, "warning: stored basis is stale\n"),
    );
    assert!(
        warning_absent_from_validate
            .iter()
            .any(|v| v.contains("validate reported no line")),
        "{warning_absent_from_validate:?}"
    );
}
