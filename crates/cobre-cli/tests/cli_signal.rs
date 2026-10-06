//! Integration tests for how `cobre run` handles SIGTERM and SIGINT, in one
//! process and under `mpiexec -n 2`, and for the exit code every rank takes when
//! rank 0's final writes fail. The tests spawn the binary, signal a process at a
//! stderr readiness line, and check the exit status and outputs.

#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs;
use std::io::{BufRead, BufReader};
use std::os::unix::process::ExitStatusExt;
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use signal_hook::consts::signal::{SIGINT, SIGTERM};
use tempfile::TempDir;

mod common;
use common::{case_dir, cobre, copy_dir_recursive, write_file};

const TIMEOUT: Duration = Duration::from_secs(180);

const STOP_CONFIG: &str = r#"{
  "training": {
    "selection": { "method": "sampled", "forward_passes": 1 },
    "stopping_rules": [{ "type": "iteration_limit", "limit": 1000 }],
    "scenario_source": {
      "seed": 42,
      "inflow": { "scheme": "in_sample" },
      "load": { "scheme": "in_sample" },
      "ncs": { "scheme": "in_sample" }
    }
  },
  "simulation": { "enabled": true, "selection": { "method": "sampled", "num_scenarios": 100 } },
  "modeling": { "inflow_non_negativity": { "method": "none" } }
}"#;

const SLOW_ITERATION_CONFIG: &str = r#"{
  "training": {
    "selection": { "method": "sampled", "forward_passes": 32 },
    "stopping_rules": [{ "type": "iteration_limit", "limit": 1000 }],
    "scenario_source": {
      "seed": 42,
      "inflow": { "scheme": "in_sample" },
      "load": { "scheme": "in_sample" },
      "ncs": { "scheme": "in_sample" }
    }
  },
  "simulation": { "enabled": true, "selection": { "method": "sampled", "num_scenarios": 100 } },
  "modeling": { "inflow_non_negativity": { "method": "none" } }
}"#;

const LONG_SIMULATION_CONFIG: &str = r#"{
  "training": {
    "selection": { "method": "sampled", "forward_passes": 1 },
    "stopping_rules": [{ "type": "iteration_limit", "limit": 2 }],
    "scenario_source": {
      "seed": 42,
      "inflow": { "scheme": "in_sample" },
      "load": { "scheme": "in_sample" },
      "ncs": { "scheme": "in_sample" }
    }
  },
  "simulation": { "enabled": true, "selection": { "method": "sampled", "num_scenarios": 2000 } },
  "modeling": { "inflow_non_negativity": { "method": "none" } }
}"#;

const THREE_ITERATION_CONFIG: &str = r#"{
  "training": {
    "selection": { "method": "sampled", "forward_passes": 1 },
    "stopping_rules": [{ "type": "iteration_limit", "limit": 3 }],
    "scenario_source": {
      "seed": 42,
      "inflow": { "scheme": "in_sample" },
      "load": { "scheme": "in_sample" },
      "ncs": { "scheme": "in_sample" }
    }
  },
  "simulation": { "enabled": true, "selection": { "method": "sampled", "num_scenarios": 100 } },
  "modeling": { "inflow_non_negativity": { "method": "none" } }
}"#;

const SLOW_THREE_ITERATION_CONFIG: &str = r#"{
  "training": {
    "selection": { "method": "sampled", "forward_passes": 32 },
    "stopping_rules": [{ "type": "iteration_limit", "limit": 3 }],
    "scenario_source": {
      "seed": 42,
      "inflow": { "scheme": "in_sample" },
      "load": { "scheme": "in_sample" },
      "ncs": { "scheme": "in_sample" }
    }
  },
  "simulation": { "enabled": true, "selection": { "method": "sampled", "num_scenarios": 100 } },
  "modeling": { "inflow_non_negativity": { "method": "none" } }
}"#;

const NO_SIMULATION_CONFIG: &str = r#"{
  "training": {
    "selection": { "method": "sampled", "forward_passes": 1 },
    "stopping_rules": [{ "type": "iteration_limit", "limit": 1000 }],
    "scenario_source": {
      "seed": 42,
      "inflow": { "scheme": "in_sample" },
      "load": { "scheme": "in_sample" },
      "ncs": { "scheme": "in_sample" }
    }
  },
  "simulation": { "enabled": false },
  "modeling": { "inflow_non_negativity": { "method": "none" } }
}"#;

fn case_with_config(config_json: &str) -> TempDir {
    let case = TempDir::new().unwrap();
    copy_dir_recursive(&case_dir("1dtoy"), case.path());
    write_file(case.path(), "config.json", config_json);
    case
}

fn output_with_a_stale_simulation_partition() -> TempDir {
    let out = TempDir::new().unwrap();
    write_file(
        out.path(),
        "simulation/costs/scenario_id=9999/data.parquet",
        "",
    );
    out
}

fn spawn_run(launcher: Option<&Path>, case: &Path, out: &Path) -> (Child, Receiver<String>) {
    let binary = cobre();
    let mut command = match launcher {
        Some(launcher) => {
            let mut command = Command::new(launcher);
            command.args(["-n", "2"]).arg(binary.get_program());
            command
        }
        None => binary,
    };
    let mut child = command
        .args(["--color", "never", "run"])
        .arg(case)
        .arg("--output")
        .arg(out)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("cobre must spawn");

    let mut stderr = BufReader::new(child.stderr.take().expect("stderr is piped"));
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut line = Vec::new();
        while stderr.read_until(b'\n', &mut line).is_ok_and(|n| n > 0) {
            let _ = tx.send(String::from_utf8_lossy(&line).trim_end().to_string());
            line.clear();
        }
    });
    (child, rx)
}

fn wait_for_line(
    child: &mut Child,
    rx: &Receiver<String>,
    pred: impl Fn(&str) -> bool,
    deadline: Instant,
) -> String {
    loop {
        let failure = match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(line) if pred(&line) => return line,
            Ok(_) => continue,
            Err(RecvTimeoutError::Timeout) => format!("no matching stderr line within {TIMEOUT:?}"),
            Err(RecvTimeoutError::Disconnected) => {
                "the run closed stderr before printing the expected line".to_string()
            }
        };
        let _ = child.kill();
        let status = child.wait();
        panic!("{failure}; the run ended with {status:?}");
    }
}

fn is_progress_line(line: &str) -> bool {
    let mut tokens = line.split_whitespace();
    tokens.next() == Some("Training")
        && tokens
            .next()
            .and_then(|t| t.split_once('/'))
            .is_some_and(|(k, n)| k.parse::<u64>().is_ok() && n.parse::<u64>().is_ok())
        && tokens.next() == Some("iter")
}

/// `kill` exits 0 on an exited but unreaped child, so liveness is checked first.
fn send(child: &mut Child, sig: &str) {
    if let Some(status) = child.try_wait().expect("try_wait must succeed") {
        panic!("the run exited before the signal {sig}: {status}");
    }
    signal_pids(&[child.id()], sig);
}

fn signal_pids(pids: &[u32], sig: &str) {
    let status = Command::new("kill")
        .args(["-s", sig])
        .args(pids.iter().map(u32::to_string))
        .status()
        .expect("kill must spawn");
    assert!(
        status.success(),
        "a process in {pids:?} exited before the signal {sig}"
    );
}

fn wait_until(child: &mut Child, deadline: Instant) -> ExitStatus {
    loop {
        if let Some(status) = child.try_wait().expect("try_wait must succeed") {
            return status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("the run did not exit within {TIMEOUT:?}");
        }
        thread::sleep(Duration::from_millis(20));
    }
}

/// The stderr lines not yet consumed, through the end of the stream.
fn remaining_lines(rx: &Receiver<String>, deadline: Instant) -> Vec<String> {
    let mut lines = Vec::new();
    loop {
        match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(line) => lines.push(line),
            Err(RecvTimeoutError::Disconnected) => return lines,
            Err(RecvTimeoutError::Timeout) => panic!("stderr stayed open after the run exited"),
        }
    }
}

fn read_json(path: &Path) -> Value {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

fn assert_signal_stop_outputs(out: &Path) {
    let training = read_json(&out.join("training/metadata.json"));
    assert_eq!(training["status"], "partial");
    assert_eq!(
        training["convergence"]["termination_reason"],
        "graceful_shutdown"
    );
    let completed = training["iterations"]["completed"].as_u64().unwrap();
    assert!(
        (2..1000).contains(&completed),
        "the stop must land on a boundary after the signalled iteration, got {completed}"
    );

    assert_checkpoint_at(out, completed);
    assert_skipped_simulation(out);
}

fn assert_checkpoint_at(out: &Path, completed: u64) {
    let checkpoint = cobre_io::read_policy_checkpoint(&out.join("policy"))
        .expect("the stopped run must leave a readable policy checkpoint");
    assert_eq!(
        u64::from(checkpoint.metadata.producer.completed_iterations),
        completed
    );
}

fn assert_skipped_simulation(out: &Path) {
    let simulation = read_json(&out.join("simulation/metadata.json"));
    assert_eq!(simulation["status"], "partial");
    assert_eq!(
        simulation["scenarios"],
        json!({ "total": 100, "completed": 0, "failed": 0 })
    );
    assert!(out.join("simulation/_SUCCESS").is_file());
    assert!(!out.join("simulation/costs").exists());
}

fn assert_one_signal_stops_gracefully(sig: &str) {
    let case = case_with_config(STOP_CONFIG);
    let out = output_with_a_stale_simulation_partition();
    let deadline = Instant::now() + TIMEOUT;
    let (mut child, rx) = spawn_run(None, case.path(), out.path());

    wait_for_line(&mut child, &rx, is_progress_line, deadline);
    send(&mut child, sig);
    let status = wait_until(&mut child, deadline);

    assert_eq!(
        status.code(),
        Some(5),
        "the run must stop gracefully: {status}"
    );
    assert_signal_stop_outputs(out.path());
}

#[test]
fn sigterm_stop_skips_the_configured_simulation() {
    assert_one_signal_stops_gracefully("TERM");
}

#[test]
fn sigint_stop_skips_the_configured_simulation() {
    assert_one_signal_stops_gracefully("INT");
}

#[test]
fn repeated_sigterm_stays_graceful_through_the_final_writes() {
    let case = case_with_config(STOP_CONFIG);
    let out = output_with_a_stale_simulation_partition();
    let deadline = Instant::now() + TIMEOUT;
    let (mut child, rx) = spawn_run(None, case.path(), out.path());

    wait_for_line(&mut child, &rx, is_progress_line, deadline);
    send(&mut child, "TERM");
    wait_for_line(
        &mut child,
        &rx,
        |line| line == "Writing training outputs...",
        deadline,
    );
    send(&mut child, "TERM");
    let status = wait_until(&mut child, deadline);

    assert_eq!(
        status.code(),
        Some(5),
        "the run must stop gracefully: {status}"
    );
    assert_signal_stop_outputs(out.path());
    assert!(out.path().join("training/_SUCCESS").is_file());
}

#[test]
fn second_sigint_terminates_a_single_process_run_by_sigint() {
    let case = case_with_config(SLOW_ITERATION_CONFIG);
    let out = TempDir::new().unwrap();
    let deadline = Instant::now() + TIMEOUT;
    let (mut child, rx) = spawn_run(None, case.path(), out.path());

    wait_for_line(&mut child, &rx, is_progress_line, deadline);
    send(&mut child, "INT");
    thread::sleep(Duration::from_millis(20));
    send(&mut child, "INT");
    let status = wait_until(&mut child, deadline);

    assert_eq!(status.signal(), Some(SIGINT), "{status}");
}

#[test]
fn sigterm_during_the_simulation_terminates_by_the_signal() {
    let case = case_with_config(LONG_SIMULATION_CONFIG);
    let out = TempDir::new().unwrap();
    let deadline = Instant::now() + TIMEOUT;
    let (mut child, rx) = spawn_run(None, case.path(), out.path());

    wait_for_line(
        &mut child,
        &rx,
        |line| line.starts_with("Simulation starting..."),
        deadline,
    );
    send(&mut child, "TERM");
    let status = wait_until(&mut child, deadline);

    assert_eq!(status.signal(), Some(SIGTERM), "{status}");
    assert!(!out.path().join("simulation/_SUCCESS").exists());
}

#[test]
fn sigterm_during_the_training_writes_never_completes_the_simulation() {
    let case = case_with_config(THREE_ITERATION_CONFIG);
    let out = TempDir::new().unwrap();
    let deadline = Instant::now() + TIMEOUT;
    let (mut child, rx) = spawn_run(None, case.path(), out.path());

    wait_for_line(
        &mut child,
        &rx,
        |line| line == "Writing training outputs...",
        deadline,
    );
    send(&mut child, "TERM");
    let status = wait_until(&mut child, deadline);

    let out = out.path();
    if status.signal().is_none() {
        println!("branch a");
        assert_eq!(status.code(), Some(5), "{status}");
        let training = read_json(&out.join("training/metadata.json"));
        assert_eq!(training["status"], "complete");
        assert_eq!(
            training["convergence"]["termination_reason"],
            "iteration_limit"
        );
        let simulation = read_json(&out.join("simulation/metadata.json"));
        assert_eq!(simulation["status"], "partial");
        assert_eq!(simulation["scenarios"]["completed"], 0);
        assert!(out.join("simulation/_SUCCESS").is_file());
    } else {
        println!("branch b");
        assert_eq!(status.signal(), Some(SIGTERM), "{status}");
        assert!(!out.join("simulation/_SUCCESS").exists());
    }
}

#[test]
fn signal_coincident_with_the_iteration_limit_exits_5_with_complete_training() {
    let case = case_with_config(SLOW_THREE_ITERATION_CONFIG);
    let out = output_with_a_stale_simulation_partition();
    let deadline = Instant::now() + TIMEOUT;
    let (mut child, rx) = spawn_run(None, case.path(), out.path());

    wait_for_line(
        &mut child,
        &rx,
        |line| line.starts_with("Training   2/3 iter"),
        deadline,
    );
    send(&mut child, "TERM");
    let status = wait_until(&mut child, deadline);
    let timings: Vec<String> = remaining_lines(&rx, deadline)
        .into_iter()
        .filter(|line| line.starts_with("Training   3/3") || line.contains("Output written to"))
        .collect();
    println!("{timings:#?}");

    assert_ne!(
        status.signal(),
        Some(SIGTERM),
        "iteration 3 plus the training writes must outlast the signal's delivery: {timings:#?}"
    );
    assert_eq!(status.code(), Some(5), "{status}");
    let out = out.path();
    let training = read_json(&out.join("training/metadata.json"));
    assert_eq!(training["status"], "complete");
    assert_eq!(
        training["convergence"]["termination_reason"],
        "iteration_limit"
    );
    assert_eq!(training["iterations"]["completed"], 3);
    assert_checkpoint_at(out, 3);
    assert_skipped_simulation(out);
}

#[test]
fn sigterm_stop_without_a_configured_simulation_exits_5() {
    let case = case_with_config(NO_SIMULATION_CONFIG);
    let out = TempDir::new().unwrap();
    let deadline = Instant::now() + TIMEOUT;
    let (mut child, rx) = spawn_run(None, case.path(), out.path());

    wait_for_line(&mut child, &rx, is_progress_line, deadline);
    send(&mut child, "TERM");
    let status = wait_until(&mut child, deadline);

    assert_eq!(status.code(), Some(5), "{status}");
    let out = out.path();
    let training = read_json(&out.join("training/metadata.json"));
    assert_eq!(training["status"], "partial");
    assert_checkpoint_at(out, training["iterations"]["completed"].as_u64().unwrap());
    assert!(!out.join("simulation/metadata.json").exists());
}

#[cfg(feature = "mpi")]
mod mpi {
    use std::os::unix::ffi::OsStrExt;
    use std::path::PathBuf;

    use super::*;

    /// The PID of the `rank` process launched with `--output out`.
    fn rank_pid(out: &Path, rank: u32) -> u32 {
        let out_arg = out.as_os_str().as_bytes();
        let rank_var = format!("PMI_RANK={rank}");
        for entry in fs::read_dir("/proc").unwrap().flatten() {
            let Some(pid) = entry.file_name().to_str().and_then(|n| n.parse().ok()) else {
                continue;
            };
            let (Ok(cmdline), Ok(environ)) = (
                fs::read(entry.path().join("cmdline")),
                fs::read(entry.path().join("environ")),
            ) else {
                continue;
            };
            if cmdline.split(|&b| b == 0).any(|arg| arg == out_arg)
                && environ
                    .split(|&b| b == 0)
                    .any(|var| var == rank_var.as_bytes())
            {
                return pid;
            }
        }
        panic!("no rank {rank} process writes to {}", out.display());
    }

    fn mpiexec() -> PathBuf {
        let mpich = PathBuf::from("/opt/mpich/bin/mpiexec");
        if mpich.is_file() {
            return mpich;
        }
        let on_path = Command::new("mpiexec")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        assert!(
            on_path.is_ok_and(|status| status.success()),
            "the MPI signal tests need /opt/mpich/bin/mpiexec or an mpiexec on PATH"
        );
        PathBuf::from("mpiexec")
    }

    fn assert_graceful_launcher_exit(status: ExitStatus) {
        assert_eq!(
            status.code(),
            Some(5),
            "the launcher must report every rank's exit 5: {status}"
        );
    }

    #[test]
    fn mpi_sigterm_to_one_rank_stops_both_ranks_at_the_same_iteration() {
        let case = case_with_config(STOP_CONFIG);
        let out = output_with_a_stale_simulation_partition();
        let deadline = Instant::now() + TIMEOUT;
        let (mut child, rx) = spawn_run(Some(&mpiexec()), case.path(), out.path());

        wait_for_line(&mut child, &rx, is_progress_line, deadline);
        signal_pids(&[rank_pid(out.path(), 1)], "TERM");
        let status = wait_until(&mut child, deadline);
        let stderr = remaining_lines(&rx, deadline);

        assert_graceful_launcher_exit(status);
        assert!(
            !stderr.iter().any(|line| line.contains("MPI_Abort")),
            "{stderr:#?}"
        );
        assert_signal_stop_outputs(out.path());
    }

    #[test]
    fn mpi_repeated_sigint_to_one_rank_stays_graceful() {
        let case = case_with_config(SLOW_ITERATION_CONFIG);
        let out = output_with_a_stale_simulation_partition();
        let deadline = Instant::now() + TIMEOUT;
        let (mut child, rx) = spawn_run(Some(&mpiexec()), case.path(), out.path());

        wait_for_line(&mut child, &rx, is_progress_line, deadline);
        let rank_1 = rank_pid(out.path(), 1);
        signal_pids(&[rank_1], "INT");
        thread::sleep(Duration::from_millis(20));
        signal_pids(&[rank_1], "INT");
        let status = wait_until(&mut child, deadline);

        assert_graceful_launcher_exit(status);
        assert_signal_stop_outputs(out.path());
    }

    #[test]
    fn mpi_sigterm_during_the_final_write_keeps_every_artifact() {
        let case = case_with_config(STOP_CONFIG);
        let out = output_with_a_stale_simulation_partition();
        let deadline = Instant::now() + TIMEOUT;
        let (mut child, rx) = spawn_run(Some(&mpiexec()), case.path(), out.path());

        wait_for_line(&mut child, &rx, is_progress_line, deadline);
        let ranks = [rank_pid(out.path(), 0), rank_pid(out.path(), 1)];
        signal_pids(&ranks[1..], "TERM");
        wait_for_line(
            &mut child,
            &rx,
            |line| line == "Writing training outputs...",
            deadline,
        );
        signal_pids(&ranks, "TERM");
        let status = wait_until(&mut child, deadline);

        assert_graceful_launcher_exit(status);
        assert_signal_stop_outputs(out.path());
        assert!(out.path().join("training/_SUCCESS").is_file());
    }

    #[test]
    fn mpi_training_write_failure_gives_every_rank_its_exit_code() {
        let case = case_with_config(THREE_ITERATION_CONFIG);
        let out = TempDir::new().unwrap();
        fs::create_dir_all(out.path().join("training/metadata.json.tmp")).unwrap();
        let deadline = Instant::now() + TIMEOUT;
        let (mut child, _rx) = spawn_run(Some(&mpiexec()), case.path(), out.path());

        let status = wait_until(&mut child, deadline);

        println!("launcher exit: {status}");
        assert_eq!(status.code(), Some(2), "{status}");
        assert!(!out.path().join("training/_SUCCESS").exists());
        assert!(!out.path().join("simulation/_SUCCESS").exists());
    }

    #[test]
    fn mpi_skipped_simulation_write_failure_gives_every_rank_its_exit_code() {
        let case = case_with_config(STOP_CONFIG);
        let out = TempDir::new().unwrap();
        fs::create_dir_all(out.path().join("simulation/metadata.json.tmp")).unwrap();
        let deadline = Instant::now() + TIMEOUT;
        let (mut child, rx) = spawn_run(Some(&mpiexec()), case.path(), out.path());

        wait_for_line(&mut child, &rx, is_progress_line, deadline);
        signal_pids(&[rank_pid(out.path(), 1)], "TERM");
        let status = wait_until(&mut child, deadline);

        assert_eq!(status.code(), Some(2), "{status}");
        assert!(out.path().join("training/_SUCCESS").is_file());
        assert!(!out.path().join("simulation/_SUCCESS").exists());
    }
}
