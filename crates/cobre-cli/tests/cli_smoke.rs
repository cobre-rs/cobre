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
        .stdout(predicate::str::contains("--skip-simulation"))
        .stdout(predicate::str::contains("--quiet"))
        .stdout(predicate::str::contains("--no-banner"))
        .stdout(predicate::str::contains("--verbose"))
        .stdout(predicate::str::contains("CASE_DIR"));
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
fn run_stub_exits_0_and_prints_not_implemented() {
    cobre()
        .args(["run", "/nonexistent/path"])
        .assert()
        .success()
        .stderr(predicate::str::contains("not yet implemented"));
}

#[test]
fn validate_stub_exits_0_and_prints_not_implemented() {
    cobre()
        .args(["validate", "/nonexistent/path"])
        .assert()
        .success()
        .stderr(predicate::str::contains("not yet implemented"));
}

#[test]
fn report_stub_exits_0_and_prints_not_implemented() {
    cobre()
        .args(["report", "/nonexistent/path"])
        .assert()
        .success()
        .stderr(predicate::str::contains("not yet implemented"));
}

#[test]
fn unknown_subcommand_exits_nonzero() {
    cobre().arg("unknown-subcommand").assert().failure();
}
