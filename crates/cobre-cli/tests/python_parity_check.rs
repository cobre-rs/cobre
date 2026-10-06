//! Integration test: Python parity check.
//!
//! Runs `scripts/ci/check_python_parity.py`, the canonical source-level
//! enforcement of the CLI↔Python output-parity hard rule, against the real tree
//! and against mutated copies of both write paths.

#![allow(clippy::expect_used, clippy::panic)]

use std::fs;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use tempfile::TempDir;

const CLI_OUTPUTS: &str = "crates/cobre-cli/src/commands/run/outputs.rs";
const PYTHON_RUN: &str = "crates/cobre-python/src/run.rs";
const CLI_TRAINING_MARKER: &str = r#"write_success_marker(&args.output_dir.join("training"))"#;
const PYTHON_TRAINING_MARKER: &str = r#"write_success_marker(&output_dir.join("training"))"#;

fn python3_available() -> bool {
    let available = Command::new("python3")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !available {
        eprintln!("python3 not found; skipping");
    }
    available
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("repo root must be two levels above CARGO_MANIFEST_DIR")
        .to_path_buf()
}

fn run_script(root: &Path) -> Output {
    let script = repo_root().join("scripts/ci/check_python_parity.py");
    assert!(
        script.exists(),
        "scripts/ci/check_python_parity.py must exist at {}",
        script.display()
    );
    Command::new("python3")
        .arg(&script)
        .arg("--max")
        .arg("0")
        .arg("--root")
        .arg(root)
        .output()
        .expect("failed to invoke python3")
}

fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).expect("failed to create fixture directory");
    for entry in fs::read_dir(from).expect("failed to read source directory") {
        let entry = entry.expect("failed to read directory entry");
        let target = to.join(entry.file_name());
        if entry.file_type().expect("failed to stat entry").is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), &target).expect("failed to copy fixture file");
        }
    }
}

fn fixture_root() -> TempDir {
    let fixture = TempDir::new().expect("failed to create fixture root");
    for tree in ["crates/cobre-cli/src", "crates/cobre-python/src"] {
        copy_tree(&repo_root().join(tree), &fixture.path().join(tree));
    }
    fixture
}

fn statement_lines(path: &Path, needle: &str) -> (Vec<String>, RangeInclusive<usize>) {
    let text = fs::read_to_string(path).expect("failed to read fixture file");
    let lines: Vec<String> = text.lines().map(str::to_owned).collect();
    let first = lines
        .iter()
        .position(|line| line.contains(needle))
        .unwrap_or_else(|| panic!("`{needle}` not found in {}", path.display()));
    let last = (first..lines.len())
        .find(|&i| lines[i].trim().ends_with(';'))
        .unwrap_or_else(|| panic!("no `;` ends the `{needle}` statement in {}", path.display()));
    (lines, first..=last)
}

fn write_lines(path: &Path, lines: &[String]) {
    fs::write(path, lines.join("\n") + "\n").expect("failed to write fixture file");
}

fn insert_after_statement(path: &Path, needle: &str, line: &str) {
    let (mut lines, statement) = statement_lines(path, needle);
    lines.insert(statement.end() + 1, line.to_owned());
    write_lines(path, &lines);
}

fn remove_statement(path: &Path, needle: &str) {
    let (mut lines, statement) = statement_lines(path, needle);
    lines.drain(statement);
    write_lines(path, &lines);
}

fn append(path: &Path, code: &str) {
    let mut text = fs::read_to_string(path).expect("failed to read fixture file");
    text.push_str(code);
    fs::write(path, text).expect("failed to write fixture file");
}

fn report(output: &Output) -> String {
    format!(
        "--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}

fn assert_accepted(output: &Output) {
    assert_eq!(
        output.status.code(),
        Some(0),
        "Python parity check failed.\n{}",
        report(output)
    );
}

fn assert_rejected_naming(output: &Output, names: &[&str]) {
    assert_eq!(
        output.status.code(),
        Some(1),
        "Python parity check must reject the fixture.\n{}",
        report(output)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    for name in names {
        assert!(
            stdout.contains(name),
            "stdout must name `{name}`.\n{}",
            report(output)
        );
    }
}

#[test]
fn python_parity_script_passes() {
    if !python3_available() {
        return;
    }
    let output = run_script(&repo_root());
    assert_accepted(&output);
    assert!(
        String::from_utf8_lossy(&output.stdout)
            .contains("`write_success_marker` is the last write in each of"),
        "the OK line must report the marker-order check.\n{}",
        report(&output)
    );
}

#[test]
fn parity_script_rejects_a_write_after_the_success_marker() {
    if !python3_available() {
        return;
    }
    let fixture = fixture_root();
    insert_after_statement(
        &fixture.path().join(CLI_OUTPUTS),
        CLI_TRAINING_MARKER,
        "    cobre_io::write_fixed_delivery(args.output_dir, &fixed_rows).map_err(CliError::from)?;",
    );
    assert_rejected_naming(
        &run_script(fixture.path()),
        &["write_training_outputs", "write_fixed_delivery"],
    );
}

#[test]
fn parity_script_follows_a_local_wrapper_called_after_the_success_marker() {
    if !python3_available() {
        return;
    }
    let fixture = fixture_root();
    let run_rs = fixture.path().join(PYTHON_RUN);
    append(
        &run_rs,
        "fn stage_extra_sidecar(dir: &Path) -> Result<(), String> { cobre_io::write_fixed_delivery(dir, &[]).map_err(|e| e.to_string()) }\n",
    );
    insert_after_statement(
        &run_rs,
        PYTHON_TRAINING_MARKER,
        "    stage_extra_sidecar(output_dir)?;",
    );
    assert_rejected_naming(
        &run_script(fixture.path()),
        &[
            "crates/cobre-python/src/run.rs::write_training_outputs",
            "stage_extra_sidecar",
        ],
    );
}

#[test]
fn parity_script_rejects_a_phase_writer_without_the_success_marker() {
    if !python3_available() {
        return;
    }
    let fixture = fixture_root();
    remove_statement(&fixture.path().join(PYTHON_RUN), PYTHON_TRAINING_MARKER);
    assert_rejected_naming(
        &run_script(fixture.path()),
        &["crates/cobre-python/src/run.rs::write_training_outputs"],
    );
}

#[test]
fn parity_script_fails_closed_when_a_phase_writer_is_missing() {
    if !python3_available() {
        return;
    }
    let fixture = fixture_root();
    let run_rs = fixture.path().join(PYTHON_RUN);
    let text = fs::read_to_string(&run_rs).expect("failed to read fixture file");
    fs::write(
        &run_rs,
        text.replace(
            "fn run_simulation_phase_py",
            "fn run_simulation_phase_renamed",
        ),
    )
    .expect("failed to write fixture file");
    assert_rejected_naming(&run_script(fixture.path()), &["run_simulation_phase_py"]);
}

#[test]
fn parity_script_ignores_method_calls_after_the_success_marker() {
    if !python3_available() {
        return;
    }
    let fixture = fixture_root();
    insert_after_statement(
        &fixture.path().join(CLI_OUTPUTS),
        CLI_TRAINING_MARKER,
        r#"    let _ = args.stderr.write_line("done");"#,
    );
    assert_accepted(&run_script(fixture.path()));
}

#[test]
fn parity_script_checks_every_function_that_writes_the_marker() {
    if !python3_available() {
        return;
    }
    let fixture = fixture_root();
    append(
        &fixture.path().join(CLI_OUTPUTS),
        "
fn extra_phase(dir: &Path) -> Result<(), CliError> {
    write_success_marker(dir).map_err(CliError::from)?;
    cobre_io::write_fixed_delivery(dir, &[]).map_err(CliError::from)?;
    Ok(())
}
",
    );
    assert_rejected_naming(&run_script(fixture.path()), &["extra_phase"]);
}
