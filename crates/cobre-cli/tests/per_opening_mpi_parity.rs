//! Integration test wrapper for the per-opening MPI parity shell script.
//!
//! This test shells out to `scripts/test_per_opening_mpi_parity.sh` and asserts
//! exit code 0.  It is marked `#[ignore]` so that default `cargo test` and
//! `cargo nextest run` invocations do not trigger MPI execution.
//!
//! To run explicitly:
//!
//! ```text
//! cargo test -p cobre-cli --features mpi -- --ignored per_opening_mpi_parity
//! ```
//!
//! Prerequisites (same as the shell script):
//!   - `mpirun` on `PATH` (`OpenMPI` or `MPICH`)
//!   - `target/release/cobre` built with `--features mpi`
//!   - Python 3 with `pyarrow` installed

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;
use std::process::Command;

/// Returns the repository root, resolved relative to this test file's location.
///
/// Cargo sets `CARGO_MANIFEST_DIR` to the crate root; the repo root is two
/// levels up (crates/cobre-cli → repo root).
fn repo_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir)
        .parent() // crates/
        .expect("crates/ parent must exist")
        .parent() // repo root
        .expect("repo root must exist")
        .to_path_buf()
}

/// Runs `scripts/test_per_opening_mpi_parity.sh` with the D01 case and
/// asserts that exit code 0 is returned (parity confirmed).
///
/// This test requires MPI (`mpirun`) and a release binary with `--features mpi`.
/// It is `#[ignore]`d by default; run with `-- --ignored` to activate.
#[test]
#[ignore = "requires mpirun and a release build with --features mpi"]
fn per_opening_mpi_parity_d01() {
    let root = repo_root();
    let script = root.join("scripts").join("test_per_opening_mpi_parity.sh");
    let case_dir = root
        .join("examples")
        .join("deterministic")
        .join("d01-thermal-dispatch");

    assert!(
        script.exists(),
        "parity script not found: {}",
        script.display()
    );
    assert!(
        case_dir.exists(),
        "D01 case directory not found: {}",
        case_dir.display()
    );

    let status = Command::new("bash")
        .arg(&script)
        .arg(&case_dir)
        .current_dir(&root)
        .status()
        .expect("failed to launch test_per_opening_mpi_parity.sh");

    assert!(
        status.success(),
        "per-opening MPI parity test failed (exit code {:?}). \
         See target/parity_* directories for the preserved outputs.",
        status.code()
    );
}
