//! Opt-in regression tests: byte-identical comparison vs the v0.4.5 reference map.
//!
//! Every test in this file is `#[ignore]` by default so the full D-suite run at
//! release profile does not block ordinary CI.  To run explicitly:
//!
//! ```sh
//! cargo nextest run -p cobre-sddp --test v045_regression --run-ignored --release
//! ```
//!
//! ## What these tests do
//!
//! Each test invokes the release-mode `cobre` CLI binary on a d-case via
//! `std::process::Command`, then computes SHA256 of the key stable output
//! parquet files and compares them against the hashes recorded in
//! `docs/assessments/v0_4_5_reference.md`.
//!
//! A mismatch is only acceptable when the affected path is in the same
//! expected-drifts allowlist used by `scripts/compare_v045_reference.py`:
//!
//! - `*/training/solver/iterations.parquet`   — ticket-007 schema rename
//! - `*/simulation/solver/iterations.parquet` — ticket-007 schema rename
//! - `*/training/convergence.parquet`          — timing columns (excluded in ticket-001)
//! - `*/training/timing/iterations.parquet`    — pure timing file (excluded in ticket-001)
//!
//! Any other mismatch is an unexpected drift and constitutes a test failure.
//!
//! ## Design notes
//!
//! - The release binary path is located by looking for `target/release/cobre`
//!   relative to the workspace root (two levels up from `crates/cobre-sddp/`).
//! - Output is written to a temporary directory under `target/v045-regression-run/`
//!   (not `target/v045-reference/` to avoid overwriting the reference capture).
//! - The reference SHA256 map is parsed from
//!   `docs/assessments/v0_4_5_reference.md` at test time using the same
//!   marker-based extraction logic as the Python script.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use std::collections::HashMap;
use std::io::{BufRead as _, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// SHA256 helper (pure std — no external crate)
// ---------------------------------------------------------------------------

/// Compute the SHA256 hex digest of the file at `path`.
fn sha256_file(path: &Path) -> String {
    // We shell out to `sha256sum` (Linux / coreutils) because std does not
    // include SHA256.  This keeps the test file free of extra deps.
    // This keeps the test file free of extra deps while staying pure-stdlib for
    // the actual test logic.
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .expect("sha256sum must be available (install GNU coreutils)");
    assert!(
        output.status.success(),
        "sha256sum failed for {}",
        path.display()
    );
    // sha256sum output: "<64-hex>  <path>\n"
    let stdout = String::from_utf8(output.stdout).expect("sha256sum output must be UTF-8");
    stdout
        .split_ascii_whitespace()
        .next()
        .expect("sha256sum must produce at least one token")
        .to_owned()
}

// ---------------------------------------------------------------------------
// Reference map parser
// ---------------------------------------------------------------------------

/// Parse the SHA256 map from `docs/assessments/v0_4_5_reference.md`.
///
/// Extracts every fenced code block whose content lines all match the
/// `<64-hex>  <path>` format and merges them into one map.
fn load_reference_map(repo_root: &Path) -> HashMap<String, String> {
    let doc_path = repo_root.join("docs/assessments/v0_4_5_reference.md");
    let file =
        std::fs::File::open(&doc_path).expect("docs/assessments/v0_4_5_reference.md must exist");
    let reader = BufReader::new(file);

    let mut result: HashMap<String, String> = HashMap::new();
    let mut in_fence = false;
    let mut candidate: Vec<String> = Vec::new();

    for line in reader.lines() {
        let line = line.expect("line must be valid UTF-8");
        let stripped = line.trim();
        if !in_fence {
            if stripped == "```" {
                in_fence = true;
                candidate.clear();
            }
        } else if stripped == "```" {
            // Validate and accept block
            let mut block: HashMap<String, String> = HashMap::new();
            let mut valid = true;
            for cl in &candidate {
                let cl_trimmed = cl.trim();
                if cl_trimmed.is_empty() {
                    continue;
                }
                if let Some((sha, path)) = parse_sha_line(cl_trimmed) {
                    block.insert(path, sha);
                } else {
                    valid = false;
                    break;
                }
            }
            if valid && !block.is_empty() {
                result.extend(block);
            }
            in_fence = false;
        } else {
            candidate.push(line.clone());
        }
    }

    assert!(
        !result.is_empty(),
        "No SHA256 map found in docs/assessments/v0_4_5_reference.md"
    );
    result
}

/// A minimal 64-char hex prefix check implemented without a regex crate.
/// Returns `Some((sha_hex, rel_path))` if the line matches the sha256sum format.
fn parse_sha_line(line: &str) -> Option<(String, String)> {
    // Format: "<64 lowercase hex chars>  <path>"
    if line.len() < 67 {
        // 64 + 2 + at least 1
        return None;
    }
    let sha = &line[..64];
    if !sha.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')) {
        return None;
    }
    // Must be followed by exactly two spaces
    if &line[64..66] != "  " {
        return None;
    }
    let rel_path = line[66..].trim().to_owned();
    if rel_path.is_empty() {
        return None;
    }
    Some((sha.to_owned(), rel_path))
}

// ---------------------------------------------------------------------------
// Expected-drifts allowlist
// ---------------------------------------------------------------------------

/// Returns `true` if a mismatch for `rel_path` is expected (and allowlisted).
///
/// Mirrors the `EXPECTED_DRIFTS` list in `scripts/compare_v045_reference.py`.
fn is_allowlisted_drift(rel_path: &str) -> bool {
    rel_path.ends_with("/training/solver/iterations.parquet")
        || rel_path.ends_with("/simulation/solver/iterations.parquet")
        || rel_path.ends_with("/training/convergence.parquet")
        || rel_path.ends_with("/training/timing/iterations.parquet")
}

// ---------------------------------------------------------------------------
// Core test harness
// ---------------------------------------------------------------------------

/// Locate the workspace root (two `..` from the test crate dir).
fn workspace_root() -> PathBuf {
    // Integration tests run with cwd = workspace root when invoked via nextest.
    // As a fallback we look for Cargo.lock walking up from the manifest dir.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("crates/cobre-sddp must have parent")
        .parent()
        .expect("crates/ must have parent")
        .to_owned()
}

/// Run a single d-case via the release `cobre` binary and compare its stable
/// parquet outputs against the v0.4.5 reference map.
///
/// `case_name` must match the directory name under `examples/deterministic/`
/// (e.g. `"d01-thermal-dispatch"`).
fn assert_v045_compatible(case_name: &str) {
    let root = workspace_root();
    let cobre_bin = root.join("target/release/cobre");
    assert!(
        cobre_bin.exists(),
        "release binary not found at {}; run `cargo build --release --workspace` first",
        cobre_bin.display()
    );

    let case_dir = root.join("examples/deterministic").join(case_name);
    assert!(
        case_dir.exists(),
        "case directory not found: {}",
        case_dir.display()
    );

    let out_dir = root.join("target/v045-regression-run").join(case_name);
    std::fs::create_dir_all(&out_dir).expect("must be able to create output dir");

    // Run the CLI
    let status = Command::new(&cobre_bin)
        .args([
            "run",
            case_dir.to_str().unwrap(),
            "--output",
            out_dir.to_str().unwrap(),
            "--quiet",
        ])
        .status()
        .expect("failed to launch cobre binary");
    assert!(
        status.success(),
        "cobre run failed for case {case_name} (exit code: {status})"
    );

    // Load the reference map
    let reference = load_reference_map(&root);

    // Collect stable parquet files produced for this case
    let mut unexpected_drifts: Vec<(String, String, String)> = Vec::new();
    let mut ok_count = 0usize;
    let mut allowed_count = 0usize;

    for (rel_path, ref_sha) in &reference {
        // Filter to entries belonging to this case only
        let case_prefix = format!("{case_name}/");
        if !rel_path.starts_with(&case_prefix) {
            continue;
        }

        // Strip the case prefix to get the path within the output dir
        let within_case = &rel_path[case_prefix.len()..];
        let file_path = out_dir.join(within_case);

        if !file_path.exists() {
            // A file present in the reference that doesn't exist post-epic-03 is
            // suspicious unless it's timing-bearing (already excluded at capture).
            assert!(
                is_allowlisted_drift(rel_path),
                "v0.4.5 reference file not produced by post-epic-03 build: {rel_path}\n\
                 Expected at: {}",
                file_path.display()
            );
            allowed_count += 1;
            continue;
        }

        let actual_sha = sha256_file(&file_path);
        if actual_sha == *ref_sha {
            ok_count += 1;
        } else if is_allowlisted_drift(rel_path) {
            allowed_count += 1;
        } else {
            unexpected_drifts.push((rel_path.clone(), ref_sha.clone(), actual_sha));
        }
    }

    if !unexpected_drifts.is_empty() {
        let report = unexpected_drifts
            .iter()
            .map(|(path, expected, actual)| {
                format!(
                    "  UNEXPECTED DRIFT: {path}\n    reference: {expected}\n    actual   : {actual}"
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "v0.4.5 regression FAILED for case {case_name} — {} unexpected drift(s)\n\
             ({ok_count} byte-identical, {allowed_count} allowlisted)\n\n{report}",
            unexpected_drifts.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Per-case test functions (one per d-case; all #[ignore] by default)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d01_thermal_dispatch() {
    assert_v045_compatible("d01-thermal-dispatch");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d02_single_hydro() {
    assert_v045_compatible("d02-single-hydro");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d03_two_hydro_cascade() {
    assert_v045_compatible("d03-two-hydro-cascade");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d04_transmission() {
    assert_v045_compatible("d04-transmission");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d05_fpha_constant_head() {
    assert_v045_compatible("d05-fpha-constant-head");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d06_fpha_variable_head() {
    assert_v045_compatible("d06-fpha-variable-head");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d07_fpha_computed() {
    assert_v045_compatible("d07-fpha-computed");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d08_evaporation() {
    assert_v045_compatible("d08-evaporation");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d09_multi_deficit() {
    assert_v045_compatible("d09-multi-deficit");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d10_inflow_nonnegativity() {
    assert_v045_compatible("d10-inflow-nonnegativity");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d11_water_withdrawal() {
    assert_v045_compatible("d11-water-withdrawal");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d13_generic_constraint() {
    assert_v045_compatible("d13-generic-constraint");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d14_block_factors() {
    assert_v045_compatible("d14-block-factors");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d15_non_controllable_source() {
    assert_v045_compatible("d15-non-controllable-source");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d16_par1_lag_shift() {
    assert_v045_compatible("d16-par1-lag-shift");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d19_multi_hydro_par() {
    assert_v045_compatible("d19-multi-hydro-par");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d20_operational_violations() {
    assert_v045_compatible("d20-operational-violations");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d21_min_outflow_regression() {
    assert_v045_compatible("d21-min-outflow-regression");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d22_per_block_min_outflow() {
    assert_v045_compatible("d22-per-block-min-outflow");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d23_bidirectional_withdrawal() {
    assert_v045_compatible("d23-bidirectional-withdrawal");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d24_productivity_override() {
    assert_v045_compatible("d24-productivity-override");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d25_discount_rate() {
    assert_v045_compatible("d25-discount-rate");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d26_estimated_par2() {
    assert_v045_compatible("d26-estimated-par2");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d27_per_stage_thermal_cost() {
    assert_v045_compatible("d27-per-stage-thermal-cost");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d28_decomp_weekly_monthly() {
    assert_v045_compatible("d28-decomp-weekly-monthly");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d29_pattern_c_weekly_par() {
    assert_v045_compatible("d29-pattern-c-weekly-par");
}

#[test]
#[ignore = "Opt-in: run with --run-ignored to check v0.4.5 byte identity"]
fn v045_d30_pattern_d_monthly_quarterly() {
    assert_v045_compatible("d30-pattern-d-monthly-quarterly");
}
