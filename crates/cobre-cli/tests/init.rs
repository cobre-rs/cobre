//! Integration tests for the `cobre init` subcommand.
//!
//! Tests the full path: binary invocation → template resolution → file
//! creation → exit code. The init-then-validate round-trip test is the
//! highest-value test: it proves that the embedded template files constitute
//! a valid case directory.

#![allow(clippy::unwrap_used)]

use std::fs;

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;
use tempfile::TempDir;

fn cobre() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

#[test]
fn test_init_list_shows_1dtoy() {
    cobre()
        .args(["init", "--list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("1dtoy"));
}

#[test]
fn test_init_1dtoy_creates_valid_case() {
    let dir = TempDir::new().unwrap();
    let dir_str = dir.path().to_str().unwrap();

    cobre()
        .args(["init", "--template", "1dtoy", dir_str])
        .assert()
        .success();

    cobre().args(["validate", dir_str]).assert().success();
}

#[test]
fn test_init_unknown_template_fails() {
    let dir = TempDir::new().unwrap();
    let dir_str = dir.path().to_str().unwrap();

    cobre()
        .args(["init", "--template", "bogus", dir_str])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Available"));
}

#[test]
fn test_init_no_args_fails() {
    cobre().args(["init"]).assert().failure();
}

#[test]
fn test_init_existing_non_empty_dir_fails() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("dummy.txt"), "x").unwrap();

    cobre()
        .args(["init", "--template", "1dtoy", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("force"));
}

#[test]
fn test_init_force_overwrites() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("dummy.txt"), "x").unwrap();

    cobre()
        .args([
            "init",
            "--template",
            "1dtoy",
            dir.path().to_str().unwrap(),
            "--force",
        ])
        .assert()
        .success();
}
