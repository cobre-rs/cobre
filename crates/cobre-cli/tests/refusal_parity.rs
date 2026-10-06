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

use serde_json::{Value, json};
use tempfile::TempDir;

mod common;
use common::{case_dir, cobre, copy_dir_recursive};

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
    let output_dir = TempDir::new().unwrap();
    let validate = cobre()
        .arg("validate")
        .arg(validate_case)
        .output()
        .unwrap_or_else(|e| panic!("{}: cobre validate failed to spawn: {e}", row.name));
    let run = cobre()
        .arg("run")
        .arg(run_case)
        .arg("--output")
        .arg(output_dir.path())
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
