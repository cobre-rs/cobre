//! Linux end-to-end parity gate for opt-in Rayon worker affinity.

#![cfg(target_os = "linux")]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use assert_cmd::prelude::*;
use tempfile::TempDir;

fn cobre() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

fn case_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/deterministic/d01-thermal-dispatch")
}

fn run_case(case: &Path, output: &Path, policy: &str, threads: &str) {
    cobre()
        .args([
            "run",
            case.to_str().expect("case path must be UTF-8"),
            "--output",
            output.to_str().expect("output path must be UTF-8"),
            "--threads",
            threads,
            "--cpu-bind",
            policy,
            "--quiet",
        ])
        .assert()
        .success();
}

fn copy_directory(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).expect("destination directory must be created");
    for entry in fs::read_dir(source).expect("source directory must be readable") {
        let entry = entry.expect("source entry must be readable");
        let target = destination.join(entry.file_name());
        if entry.path().is_dir() {
            copy_directory(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).expect("fixture file must be copied");
        }
    }
}

fn reverse_thermal_declarations(case: &Path) {
    let path = case.join("system/thermals.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).expect("thermals fixture must be readable"))
            .expect("thermals fixture must be JSON");
    value["thermals"]
        .as_array_mut()
        .expect("thermals must be an array")
        .reverse();
    fs::write(
        path,
        serde_json::to_vec_pretty(&value).expect("reversed thermals must encode"),
    )
    .expect("reversed thermals fixture must be written");
}

fn relative_files(root: &Path, current: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(current).expect("directory must be readable") {
        let path = entry.expect("directory entry must be readable").path();
        if path.is_dir() {
            relative_files(root, &path, files);
        } else {
            files.push(
                path.strip_prefix(root)
                    .expect("descendant must strip root")
                    .to_path_buf(),
            );
        }
    }
}

fn assert_directory_bytes_equal(lhs: &Path, rhs: &Path) {
    let mut lhs_files = Vec::new();
    let mut rhs_files = Vec::new();
    relative_files(lhs, lhs, &mut lhs_files);
    relative_files(rhs, rhs, &mut rhs_files);
    lhs_files.sort();
    rhs_files.sort();
    assert_eq!(lhs_files, rhs_files, "directory file sets differ");
    for relative in lhs_files {
        let normalized_artifact = |root: &Path| {
            let bytes = fs::read(root.join(&relative)).expect("artifact must be readable");
            if relative == Path::new("metadata.json") {
                let mut metadata: serde_json::Value =
                    serde_json::from_slice(&bytes).expect("metadata must be JSON");
                metadata
                    .as_object_mut()
                    .expect("metadata must be an object")
                    .remove("created_at");
                serde_json::to_vec(&metadata).expect("metadata must encode")
            } else {
                bytes
            }
        };
        assert_eq!(
            normalized_artifact(lhs),
            normalized_artifact(rhs),
            "artifact differs: {}",
            relative.display(),
        );
    }
}

fn assert_convergence_equal(lhs: &Path, rhs: &Path) {
    let lhs = cobre_io::read_convergence_summary(&lhs.join("training/convergence.parquet"))
        .expect("left convergence must be readable");
    let rhs = cobre_io::read_convergence_summary(&rhs.join("training/convergence.parquet"))
        .expect("right convergence must be readable");
    assert_eq!(lhs.total_lp_solves, rhs.total_lp_solves);
    assert_eq!(
        lhs.final_lower_bound.to_bits(),
        rhs.final_lower_bound.to_bits()
    );
    assert_eq!(
        lhs.final_upper_bound_mean.to_bits(),
        rhs.final_upper_bound_mean.to_bits()
    );
    assert_eq!(
        lhs.final_upper_bound_std.to_bits(),
        rhs.final_upper_bound_std.to_bits()
    );
    assert_eq!(
        lhs.final_gap_percent.map(f64::to_bits),
        rhs.final_gap_percent.map(f64::to_bits)
    );
}

#[test]
fn bound_and_unbound_runs_have_identical_numerical_policy() {
    let unbound = TempDir::new().expect("unbound output tempdir");
    let bound = TempDir::new().expect("bound output tempdir");
    let bound_single = TempDir::new().expect("single-worker output tempdir");
    let bound_repeat = TempDir::new().expect("repeated output tempdir");
    let permuted_case = TempDir::new().expect("permuted case tempdir");
    let permuted_output = TempDir::new().expect("permuted output tempdir");
    let original_case = case_dir();
    copy_directory(&original_case, permuted_case.path());
    reverse_thermal_declarations(permuted_case.path());
    run_case(&original_case, unbound.path(), "none", "2");
    run_case(&original_case, bound.path(), "core", "2");
    run_case(&original_case, bound_single.path(), "core", "1");
    run_case(&original_case, bound_repeat.path(), "core", "2");
    run_case(permuted_case.path(), permuted_output.path(), "core", "2");

    assert_directory_bytes_equal(&unbound.path().join("policy"), &bound.path().join("policy"));
    assert_directory_bytes_equal(
        &bound.path().join("policy"),
        &bound_single.path().join("policy"),
    );
    assert_directory_bytes_equal(
        &bound.path().join("policy"),
        &bound_repeat.path().join("policy"),
    );
    assert_directory_bytes_equal(
        &bound.path().join("policy"),
        &permuted_output.path().join("policy"),
    );
    assert_convergence_equal(unbound.path(), bound.path());
    assert_convergence_equal(bound.path(), bound_single.path());
    assert_convergence_equal(bound.path(), bound_repeat.path());
    assert_convergence_equal(bound.path(), permuted_output.path());

    let unbound_metadata =
        cobre_io::read_training_metadata(&unbound.path().join("training/metadata.json"))
            .expect("unbound metadata must be readable");
    let bound_metadata =
        cobre_io::read_training_metadata(&bound.path().join("training/metadata.json"))
            .expect("bound metadata must be readable");
    assert_eq!(unbound_metadata.distribution.rank_affinity.len(), 1);
    assert_eq!(bound_metadata.distribution.rank_affinity.len(), 1);
    assert_eq!(
        unbound_metadata.distribution.rank_affinity[0].policy,
        "none"
    );
    let placement = &bound_metadata.distribution.rank_affinity[0];
    assert_eq!(placement.policy, "core");
    assert_eq!(placement.worker_cpus.len(), 2);
    assert!(
        placement
            .worker_cpus
            .iter()
            .all(|cpu| placement.visible_cpus.contains(cpu))
    );
}

#[cfg(feature = "mpi")]
fn run_mpi_case(output: &Path, ranks: &str) -> bool {
    match Command::new("mpiexec").arg("--version").output() {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return false,
        Err(error) => {
            assert_eq!(
                error.kind(),
                std::io::ErrorKind::NotFound,
                "mpiexec availability check failed: {error}"
            );
            return false;
        }
    }
    Command::new("mpiexec")
        .args([
            "-n",
            ranks,
            assert_cmd::cargo::cargo_bin!("cobre")
                .to_str()
                .expect("binary path must be UTF-8"),
            "run",
            case_dir().to_str().expect("case path must be UTF-8"),
            "--output",
            output.to_str().expect("output path must be UTF-8"),
            "--comm-backend",
            "mpi",
            "--threads",
            "1",
            "--cpu-bind",
            "core",
            "--quiet",
        ])
        .assert()
        .success();
    true
}

#[cfg(feature = "mpi")]
#[test]
fn bound_local_and_mpi_rank_shapes_have_identical_numerical_policy() {
    let local = TempDir::new().expect("local output tempdir");
    let mpi_one = TempDir::new().expect("single-rank output tempdir");
    let mpi_two = TempDir::new().expect("two-rank output tempdir");
    run_case(&case_dir(), local.path(), "core", "1");
    if !run_mpi_case(mpi_one.path(), "1") {
        return;
    }
    assert!(run_mpi_case(mpi_two.path(), "2"));

    assert_directory_bytes_equal(&local.path().join("policy"), &mpi_one.path().join("policy"));
    assert_directory_bytes_equal(
        &mpi_one.path().join("policy"),
        &mpi_two.path().join("policy"),
    );
    assert_convergence_equal(local.path(), mpi_one.path());
    assert_convergence_equal(mpi_one.path(), mpi_two.path());

    let metadata = cobre_io::read_training_metadata(&mpi_two.path().join("training/metadata.json"))
        .expect("MPI metadata must be readable");
    assert_eq!(metadata.distribution.rank_affinity.len(), 2);
    for placement in metadata.distribution.rank_affinity {
        assert_eq!(placement.policy, "core");
        assert_eq!(placement.worker_cpus.len(), 1);
        assert!(
            placement
                .worker_cpus
                .iter()
                .all(|cpu| placement.visible_cpus.contains(cpu))
        );
    }
}
