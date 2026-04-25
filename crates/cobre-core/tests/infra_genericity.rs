//! Infrastructure crate genericity gate — integration test.
//!
//! Runs `scripts/check-infra-genericity.sh` and asserts it exits successfully.
//! The script scans the five infrastructure crates (`cobre-core`, `cobre-io`,
//! `cobre-solver`, `cobre-stochastic`, `cobre-comm`) for algorithm-specific
//! vocabulary that must not appear in their production source code.
//!
//! Failing this test means one of the infra crates has acquired an
//! algorithm-specific reference (e.g. `Sddp`, `Benders`, `Cut`, `cut pool`).
//! Fix the violation before committing.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;
use std::process::Command;

#[test]
fn infra_genericity_gate() {
    // Locate the script relative to the workspace root. The test binary is run
    // from the workspace root by `cargo test`, so we can walk up from CARGO_MANIFEST_DIR.
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    // cobre-core manifest is at <repo>/crates/cobre-core/Cargo.toml, so the
    // workspace root is two levels up.
    let workspace_root = manifest_dir
        .parent() // crates/
        .expect("cobre-core has a parent directory")
        .parent() // repo root
        .expect("crates/ has a parent directory");

    let script = workspace_root.join("scripts/check-infra-genericity.sh");

    assert!(
        script.exists(),
        "Gate script not found at {}: run from the workspace root or ensure the script exists",
        script.display()
    );

    let output = Command::new("bash")
        .arg(&script)
        .current_dir(workspace_root)
        .output()
        .expect("failed to execute check-infra-genericity.sh");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "Infra genericity gate FAILED.\n\
         --- stdout ---\n{stdout}\n\
         --- stderr ---\n{stderr}\n\
         Fix the flagged violations before committing."
    );
}
