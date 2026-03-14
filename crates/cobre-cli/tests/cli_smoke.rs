//! Smoke tests for the `cobre` binary using `assert_cmd`.
//!
//! These tests verify that the argument-parsing skeleton and subcommand stubs
//! behave correctly without performing any real computation.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

fn cobre() -> Command {
    // `cargo_bin!` resolves via the CARGO_BIN_EXE_cobre environment variable,
    // which is set by cargo for integration tests and is compatible with custom
    // build directories (unlike the deprecated `Command::cargo_bin`).
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
        .stdout(predicate::str::contains("CASE_DIR"));
}

/// `--threads 0` is rejected by clap before execution begins.
///
/// The `value_parser = clap::value_parser!(u32).range(1..)` constraint
/// on `RunArgs::threads` means `0` triggers a clap validation error,
/// producing exit code 2 and a message on stderr, without any I/O.
#[test]
fn run_threads_zero_exits_with_clap_error() {
    cobre()
        .args(["run", "--threads", "0", "/some/path"])
        .assert()
        .failure()
        .code(2);
}

/// `--threads` with a positive value is accepted by clap (argument is valid).
///
/// Even though the case directory does not exist, the argument parsing itself
/// succeeds: the error is an I/O error (exit code 2), not a clap parse error.
/// This test verifies that `--threads 2` passes validation and execution
/// proceeds past argument parsing.
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
        .stdout(predicate::str::contains("HiGHS"));
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
fn version_stdout_contains_solver_highs() {
    cobre()
        .arg("version")
        .assert()
        .success()
        .stdout(predicate::str::contains("solver: HiGHS"));
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
