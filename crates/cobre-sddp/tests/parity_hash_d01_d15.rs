//! Parity hash harness for deterministic cases D01–D15 (14 cases; D12 is absent).
//!
//! Computes a SHA-256 digest over a whitelist of semantic fields from each
//! case's training + simulation output. On first run with `COBRE_PARITY_REGEN=1`
//! the test **writes** the baseline files; on subsequent runs it **verifies**
//! against the committed baselines.
//!
//! ## Hash whitelist (in fixed order)
//!
//! 1. Per-iteration convergence data: `iteration_u64_le || lower_bound_f64_le
//!    || upper_bound_f64_le || upper_bound_std_f64_le || gap_f64_le`
//!    Captured from [`TrainingEvent::ConvergenceUpdate`] events (one per
//!    completed iteration, ordered 1..=N).
//!
//! 2. Per-stage, per-cut: `stage_u32_le || intercept_f64_le ||
//!    coefficient_count_u32_le || coefficient_f64_le[]`
//!    Iterated over stages 0..num_stages, then active cuts within each stage
//!    in the slot order reported by [`FutureCostFunction::active_cuts`].
//!
//! 3. Simulation primal trajectory per scenario per stage:
//!    `stage_u32_le || num_primals_u32_le || primal_f64_le[]`
//!    Scenarios sorted ascending by `scenario_id`; stages by `stage_id`.
//!    Primals = `storage_final_hm3` for each hydro at each stage, sorted by
//!    `(block_id, hydro_id)`.  For pure-thermal cases the primal vector is
//!    empty (`num_primals = 0`).
//!
//! 4. Simulation dual trajectory per scenario per stage:
//!    `stage_u32_le || num_duals_u32_le || dual_f64_le[]`
//!    Same ordering.  Duals = `water_value_per_hm3` for each hydro record,
//!    sorted by `(block_id, hydro_id)`.
//!
//! ## Field-name translation from ticket spec
//!
//! The ticket spec references generic "primal trajectory" and "dual trajectory"
//! fields.  The actual structs use:
//! - `SimulationHydroResult::storage_final_hm3`  → primal state variable
//! - `SimulationHydroResult::water_value_per_hm3` → dual of storage balance
//! - `TrainingEvent::ConvergenceUpdate::upper_bound_std` → `upper_bound_std_f64_le`
//!
//! ## Timing exclusion
//!
//! No field ending in `_ms`, containing `elapsed`, or containing `wall` is
//! included in the hash. Timing fields are allowed to drift between runs.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use std::path::Path;
use std::sync::mpsc;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::{TrainingEvent, scenario::ScenarioSource};
use cobre_sddp::{
    StudySetup, aggregate_simulation, hydro_models::prepare_hydro_models, setup::prepare_stochastic,
};
use cobre_solver::highs::HighsSolver;
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Stub communicator (single-rank, copied from deterministic.rs)
// ---------------------------------------------------------------------------

struct StubComm;

impl Communicator for StubComm {
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _counts: &[usize],
        _displs: &[usize],
    ) -> Result<(), CommError> {
        recv[..send.len()].clone_from_slice(send);
        Ok(())
    }

    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _op: ReduceOp,
    ) -> Result<(), CommError> {
        recv.clone_from_slice(send);
        Ok(())
    }

    fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
        Ok(())
    }

    fn barrier(&self) -> Result<(), CommError> {
        Ok(())
    }

    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        1
    }

    fn abort(&self, error_code: i32) -> ! {
        std::process::exit(error_code)
    }
}

// ---------------------------------------------------------------------------
// Baseline file path
// ---------------------------------------------------------------------------

fn baseline_path(case: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/parity_baselines")
        .join(format!("{case}.sha256"))
}

// ---------------------------------------------------------------------------
// Hash computation
// ---------------------------------------------------------------------------

/// Compute a SHA-256 parity hash over the semantic whitelist.
///
/// The hash is deterministic: every field is encoded as little-endian bytes,
/// the iteration order is ascending, stages are ascending within cuts, and
/// scenarios are sorted ascending by `scenario_id`.
///
/// # Field translation
///
/// - Per-iteration convergence data comes from the collected
///   `ConvergenceUpdate` events (timing-free variant of `IterationSummary`).
/// - Cut data comes from `setup.fcf().active_cuts(stage)`.
/// - Primal trajectory uses `SimulationHydroResult::storage_final_hm3`.
/// - Dual trajectory uses `SimulationHydroResult::water_value_per_hm3`.
///   Both are sorted by `(block_id, hydro_id)` within each stage.
fn compute_parity_hash(
    convergence_updates: &[(u64, f64, f64, f64, f64)],
    setup: &StudySetup,
    mut scenario_results: Vec<cobre_sddp::SimulationScenarioResult>,
) -> String {
    let mut hasher = Sha256::new();

    // ------------------------------------------------------------------
    // Section 1: Per-iteration convergence data
    // ------------------------------------------------------------------
    for &(iteration, lb, ub, ub_std, gap) in convergence_updates {
        hasher.update(iteration.to_le_bytes());
        hasher.update(lb.to_le_bytes());
        hasher.update(ub.to_le_bytes());
        hasher.update(ub_std.to_le_bytes());
        hasher.update(gap.to_le_bytes());
    }

    // ------------------------------------------------------------------
    // Section 2: Active cuts per stage (ascending stage order, then slot
    //            order within each stage as reported by active_cuts())
    // ------------------------------------------------------------------
    let fcf = setup.fcf();
    let num_stages = fcf.pools.len();
    for stage in 0..num_stages {
        for (_slot, intercept, coefficients) in fcf.active_cuts(stage) {
            hasher.update((stage as u32).to_le_bytes());
            hasher.update(intercept.to_le_bytes());
            hasher.update((coefficients.len() as u32).to_le_bytes());
            for &c in coefficients {
                hasher.update(c.to_le_bytes());
            }
        }
    }

    // ------------------------------------------------------------------
    // Section 3 & 4: Simulation primal and dual trajectories
    //
    // Sort scenarios ascending by scenario_id for determinism.
    // ------------------------------------------------------------------
    scenario_results.sort_by_key(|r| r.scenario_id);

    for scenario in &mut scenario_results {
        // Sort stages ascending by stage_id (pipeline already stage-ordered,
        // but sort defensively). `SimulationStageResult` does not derive Clone,
        // so we sort in-place using the owned Vec.
        scenario.stages.sort_by_key(|s| s.stage_id);

        for stage in &mut scenario.stages {
            // Sort hydro records by (block_id, hydro_id) for determinism.
            // `SimulationHydroResult` does not derive Clone; sort in-place.
            stage.hydros.sort_by_key(|h| (h.block_id, h.hydro_id));

            // Primal trajectory: storage_final_hm3 per hydro record.
            let num_primals = stage.hydros.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_primals.to_le_bytes());
            for h in &stage.hydros {
                hasher.update(h.storage_final_hm3.to_le_bytes());
            }

            // Dual trajectory: water_value_per_hm3 per hydro record.
            let num_duals = stage.hydros.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_duals.to_le_bytes());
            for h in &stage.hydros {
                hasher.update(h.water_value_per_hm3.to_le_bytes());
            }
        }
    }

    format!("{:x}", hasher.finalize())
}

// ---------------------------------------------------------------------------
// Baseline read/write
// ---------------------------------------------------------------------------

/// Read the baseline file for `case` and compare to `hash`, or write the
/// baseline when `COBRE_PARITY_REGEN=1`.
///
/// Returns `Ok(())` on match or successful write; `Err(msg)` on mismatch or
/// missing baseline.
fn read_or_regen_baseline(case: &str, hash: &str) -> Result<(), String> {
    let path = baseline_path(case);

    if std::env::var("COBRE_PARITY_REGEN").as_deref() == Ok("1") {
        // Regeneration mode: write the baseline.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create baseline dir: {e}"))?;
        }
        std::fs::write(&path, format!("{hash}\n"))
            .map_err(|e| format!("cannot write baseline for {case}: {e}"))?;
        eprintln!("REGEN: wrote baseline for {case}: {hash}");
        return Ok(());
    }

    // Verification mode.
    if !path.exists() {
        return Err(format!(
            "baseline file for {case} is missing at {}; \
             run with COBRE_PARITY_REGEN=1 to generate it",
            path.display()
        ));
    }

    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read baseline for {case}: {e}"))?;
    let expected = raw.trim();

    // Validate the baseline is a well-formed 64-char hex string.
    if expected.len() != 64 || !expected.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("baseline file {case} is malformed: {expected:?}"));
    }

    if expected == hash {
        eprintln!("OK: parity hash for {case} matched baseline {hash}");
        Ok(())
    } else {
        Err(format!(
            "parity hash mismatch for {case}:\n  expected (baseline): {expected}\n  actual:              {hash}"
        ))
    }
}

// ---------------------------------------------------------------------------
// Case runner
// ---------------------------------------------------------------------------

/// Map a D-case label (e.g. `"D01"`) to its fixture directory path.
fn case_dir(label: &str) -> std::path::PathBuf {
    let suffix = match label {
        "D01" => "d01-thermal-dispatch",
        "D02" => "d02-single-hydro",
        "D03" => "d03-two-hydro-cascade",
        "D04" => "d04-transmission",
        "D05" => "d05-fpha-constant-head",
        "D06" => "d06-fpha-variable-head",
        "D07" => "d07-fpha-computed",
        "D08" => "d08-evaporation",
        "D09" => "d09-multi-deficit",
        "D10" => "d10-inflow-nonnegativity",
        "D11" => "d11-water-withdrawal",
        "D13" => "d13-generic-constraint",
        "D14" => "d14-block-factors",
        "D15" => "d15-non-controllable-source",
        other => panic!("unknown case label: {other}"),
    };
    // Integration tests run from the crate root; fixtures live at
    // ../../examples/deterministic/<suffix> relative to the crate.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/deterministic")
        .join(suffix)
}

/// Run the full train + simulate pipeline for a D-case and assert parity.
fn run_case(label: &str) {
    let dir = case_dir(label);
    let config_path = dir.join("config.json");

    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    let system = cobre_io::load_case(&dir).expect("load_case must succeed");

    let pr = prepare_stochastic(system, &dir, &config, 42, &ScenarioSource::default())
        .expect("prepare_stochastic must succeed");
    let system = pr.system;
    let stochastic = pr.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, &dir).expect("prepare_hydro_models must succeed");

    // Enable simulation so we get per-scenario stage results.
    let mut config_with_sim = config.clone();
    config_with_sim.simulation.enabled = true;
    // Use a small fixed scenario count for determinism and speed.
    config_with_sim.simulation.num_scenarios = 1;

    let mut setup = StudySetup::new(&system, &config_with_sim, stochastic, hydro_models)
        .expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");

    // Set up event channel to capture per-iteration convergence data.
    let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();

    let outcome = setup
        .train(
            &mut solver,
            &comm,
            1,
            HighsSolver::new,
            Some(event_tx),
            None,
        )
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "{label}: expected no training error"
    );
    let result = outcome.result;

    // Collect ConvergenceUpdate events and sort by iteration number.
    let mut convergence_updates: Vec<(u64, f64, f64, f64, f64)> = event_rx
        .into_iter()
        .filter_map(|ev| {
            if let TrainingEvent::ConvergenceUpdate {
                iteration,
                lower_bound,
                upper_bound,
                upper_bound_std,
                gap,
                ..
            } = ev
            {
                Some((iteration, lower_bound, upper_bound, upper_bound_std, gap))
            } else {
                None
            }
        })
        .collect();
    convergence_updates.sort_by_key(|&(iter, ..)| iter);

    // Run simulation to collect per-scenario stage results.
    let mut pool = setup
        .create_workspace_pool(&comm, 1, HighsSolver::new)
        .expect("simulation workspace pool must build");

    let io_capacity = setup.io_channel_capacity().max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

    let local_costs = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            result.baked_templates.as_deref(),
            &result.basis_cache,
        )
        .expect("simulate must return Ok");

    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");

    let sim_config = setup.simulation_config();
    let _summary = aggregate_simulation(&local_costs.costs, &sim_config, &comm)
        .expect("aggregate_simulation must succeed");

    // Compute parity hash and compare/write baseline.
    let hash = compute_parity_hash(&convergence_updates, &setup, scenario_results);

    read_or_regen_baseline(label, &hash).unwrap_or_else(|msg| panic!("{msg}"));
}

// ---------------------------------------------------------------------------
// Individual test functions — 14 cases, D12 absent
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d01() {
    run_case("D01");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d02() {
    run_case("D02");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d03() {
    run_case("D03");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d04() {
    run_case("D04");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d05() {
    run_case("D05");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d06() {
    run_case("D06");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d07() {
    run_case("D07");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d08() {
    run_case("D08");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d09() {
    run_case("D09");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d10() {
    run_case("D10");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d11() {
    run_case("D11");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d13() {
    run_case("D13");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d14() {
    run_case("D14");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d15() {
    run_case("D15");
}
