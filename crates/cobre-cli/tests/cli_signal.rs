//! Integration tests for how `cobre run` handles SIGTERM and SIGINT: each test
//! spawns the binary, waits for a progress line on stderr, signals the process,
//! and checks its exit status and outputs.

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

fn wait_for_line(rx: &Receiver<String>, pred: impl Fn(&str) -> bool, deadline: Instant) -> String {
    loop {
        match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(line) if pred(&line) => return line,
            Ok(_) => {}
            Err(RecvTimeoutError::Timeout) => panic!("no matching stderr line within {TIMEOUT:?}"),
            Err(RecvTimeoutError::Disconnected) => {
                panic!("the run closed stderr before printing the expected line")
            }
        }
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
    let status = Command::new("kill")
        .args(["-s", sig, &child.id().to_string()])
        .status()
        .expect("kill must spawn");
    assert!(status.success(), "the run exited before the signal {sig}");
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

    let checkpoint = cobre_io::read_policy_checkpoint(&out.join("policy"))
        .expect("the stopped run must leave a readable policy checkpoint");
    assert_eq!(
        u64::from(checkpoint.metadata.producer.completed_iterations),
        completed
    );

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

    wait_for_line(&rx, is_progress_line, deadline);
    send(&mut child, sig);
    let status = wait_until(&mut child, deadline);

    assert_eq!(
        status.signal(),
        None,
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

    wait_for_line(&rx, is_progress_line, deadline);
    send(&mut child, "TERM");
    wait_for_line(&rx, |line| line == "Writing training outputs...", deadline);
    send(&mut child, "TERM");
    let status = wait_until(&mut child, deadline);

    assert_eq!(
        status.signal(),
        None,
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

    wait_for_line(&rx, is_progress_line, deadline);
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
        &rx,
        |line| line.starts_with("Simulation starting..."),
        deadline,
    );
    send(&mut child, "TERM");
    let status = wait_until(&mut child, deadline);

    assert_eq!(status.signal(), Some(SIGTERM), "{status}");
    assert!(!out.path().join("simulation/_SUCCESS").exists());
}
