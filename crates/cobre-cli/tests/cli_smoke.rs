//! Smoke tests for the `cobre` binary using `assert_cmd`.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

fn cobre() -> Command {
    // `cargo_bin!` honors custom build directories via CARGO_BIN_EXE_cobre, unlike
    // the deprecated `Command::cargo_bin`.
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

#[test]
fn help_exits_0_and_lists_subcommands() {
    cobre()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("run"))
        .stdout(predicate::str::contains("validate"))
        .stdout(predicate::str::contains("report"))
        .stdout(predicate::str::contains("version"));
}

#[test]
fn run_help_exits_0_and_lists_flags() {
    cobre()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--output"))
        .stdout(predicate::str::contains("--quiet"))
        .stdout(predicate::str::contains("--threads"))
        .stdout(predicate::str::contains("--cpu-bind"))
        .stdout(predicate::str::contains("CASE_DIR"));
}

#[test]
fn run_cpu_bind_invalid_exits_with_clap_error() {
    cobre()
        .args(["run", "--cpu-bind", "socket", "/some/path"])
        .assert()
        .failure()
        .code(2);
}

#[test]
fn run_cpu_bind_none_is_accepted_by_clap() {
    cobre()
        .args(["run", "--cpu-bind", "none", "/nonexistent/path"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("I/O error"));
}

#[cfg(target_os = "linux")]
#[test]
fn run_cpu_bind_core_initializes_before_case_loading() {
    cobre()
        .args([
            "run",
            "--threads",
            "2",
            "--cpu-bind",
            "core",
            "/nonexistent/path",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("I/O error"));
}

/// Exit 2 here is a clap validation error (the `.range(1..)` parser), not I/O —
/// the case path is never touched.
#[test]
fn run_threads_zero_exits_with_clap_error() {
    cobre()
        .args(["run", "--threads", "0", "/some/path"])
        .assert()
        .failure()
        .code(2);
}

/// A positive `--threads` passes clap; the failure is the missing path (I/O),
/// proving execution proceeded past argument parsing.
#[test]
fn run_threads_positive_is_accepted_by_clap() {
    cobre()
        .args(["run", "--threads", "2", "/nonexistent/path"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("I/O error"));
}

#[test]
fn version_exits_0_and_contains_version_string() {
    let version = env!("CARGO_PKG_VERSION");
    cobre()
        .arg("version")
        .assert()
        .success()
        .stdout(predicate::str::contains(version))
        .stdout(predicate::str::contains(cobre_solver::active_solver_name()));
}

#[test]
fn version_exits_0_and_stdout_contains_cobre_prefix() {
    cobre()
        .arg("version")
        .assert()
        .success()
        .stdout(predicate::str::contains("cobre "));
}

#[test]
fn version_stdout_contains_active_solver() {
    let expected = format!("solver: {}", cobre_solver::active_solver_name());
    cobre()
        .arg("version")
        .assert()
        .success()
        .stdout(predicate::str::contains(expected));
}

#[test]
fn run_nonexistent_path_exits_2_with_io_error() {
    cobre()
        .args(["run", "/nonexistent/path"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("I/O error"));
}

#[test]
fn validate_nonexistent_path_exits_2() {
    cobre()
        .args(["validate", "/nonexistent/path"])
        .assert()
        .failure()
        .code(2);
}

#[test]
fn report_nonexistent_path_exits_2() {
    cobre()
        .args(["report", "/nonexistent/path"])
        .assert()
        .failure()
        .code(2);
}

#[test]
fn unknown_subcommand_exits_nonzero() {
    cobre().arg("unknown-subcommand").assert().failure();
}
