//! Shared parity-hash computation for the integration test harnesses — the sole
//! owner of the hash whitelist and its byte layout.
//!
//! The hash is deterministic: every field is little-endian, stages ascending,
//! scenarios sorted by `scenario_id`, hydro/thermal records by
//! `(block_id, id)`. Fields 4–6 are not redundant with storage/dual: each is an
//! `n_blks`-dependent read kept specifically to surface an extraction/cost/base
//! bug a uniform-block case cannot detect — do not drop them as duplicate.
//!
//! ## Hash whitelist (in fixed order)
//!
//! 1. Per-pool, per-cut: `pool_id_u32_le || intercept_f64_le ||
//!    coefficient_count_u32_le || coefficient_f64_le[]`. The hashed key
//!    iterates `fcf`'s `pools` in pool-id order — equal to stage order on a
//!    chain (`pool_id == stage`), the only shape this harness exercises.
//! 2. Primal trajectory (`storage_final_hm3`) per scenario per stage.
//! 3. Dual trajectory (`water_value_per_hm3`) per scenario per stage.
//! 4. Per-block equipment (`spillage_m3s`) — base shifts off stage 0's block
//!    width under a non-uniform schedule (the simulation-extraction base bug).
//! 5. Cost breakdown (`spillage_cost`) — a `range_sum` whose base AND length
//!    shift under a non-uniform schedule (the cost-breakdown bug).
//! 6. Anticipated decision (`anticipated_decision_mw`) — base is the per-stage
//!    `thermal.end` (`n_blks`-dependent). `None` hashes as a 0-flag, so only
//!    anticipated cases (D34) move this field.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    dead_code
)]

use cobre_io::config::SimulationSelection;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use cobre_core::TrainingEvent;
use cobre_sddp::{
    SimulationScenarioResult, SimulationWeighting, StudySetup, aggregate_simulation,
    hydro_models::prepare_hydro_models,
    setup::{StudyParams, prepare_stochastic},
};
use sha2::{Digest, Sha256};
use std::fmt::Write;

use super::StubComm;
use super::permute::permute_case;

/// Compute the SHA-256 parity hash over the module-doc whitelist.
pub fn compute_parity_hash(
    setup: &StudySetup,
    mut scenario_results: Vec<SimulationScenarioResult>,
) -> String {
    let mut hasher = Sha256::new();

    // Active cuts in ascending pool-id order (equal to stage order on a chain),
    // then active_cuts() slot order — fixed iteration order is what makes the
    // cut digest declaration-order-stable.
    let fcf = &setup.fcf;
    let n_pools = fcf.pools.len();
    for pool in 0..n_pools {
        for (_slot, intercept, coefficients) in fcf.active_cuts(pool) {
            hasher.update((pool as u32).to_le_bytes());
            hasher.update(intercept.to_le_bytes());
            hasher.update((coefficients.len() as u32).to_le_bytes());
            for &c in coefficients {
                hasher.update(c.to_le_bytes());
            }
        }
    }

    // Sort into canonical order so the digest is independent of how the pipeline
    // emitted scenarios/stages/records.
    scenario_results.sort_by_key(|r| r.scenario_id);

    for scenario in &mut scenario_results {
        scenario.stages.sort_by_key(|s| s.stage_id);

        for stage in &mut scenario.stages {
            stage.hydros.sort_by_key(|h| (h.block_id, h.hydro_id));

            let num_hydros = stage.hydros.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_hydros.to_le_bytes());
            for h in &stage.hydros {
                hasher.update(h.storage_final_hm3.to_le_bytes());
            }

            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_hydros.to_le_bytes());
            for h in &stage.hydros {
                hasher.update(h.water_value_per_hm3.to_le_bytes());
            }

            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_hydros.to_le_bytes());
            for h in &stage.hydros {
                hasher.update(h.spillage_m3s.to_le_bytes());
            }

            let num_costs = stage.costs.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_costs.to_le_bytes());
            for c in &stage.costs {
                hasher.update(c.spillage_cost.to_le_bytes());
            }

            stage.thermals.sort_by_key(|t| (t.block_id, t.thermal_id));
            let num_thermals = stage.thermals.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_thermals.to_le_bytes());
            for t in &stage.thermals {
                // Hash a presence flag + value so `None` maps to a fixed (0, 0.0):
                // dropping the flag would collide `None` with `Some(0.0)` and break
                // the encoding's injectivity.
                let (flag, value) = t.anticipated_decision_mw.map_or((0u8, 0.0), |v| (1u8, v));
                hasher.update(flag.to_le_bytes());
                hasher.update(value.to_le_bytes());
            }
        }
    }

    hasher
        .finalize()
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

// ---------------------------------------------------------------------------
// Golden-case harness — shared by the per-backend `parity_hash_*` modules
// ---------------------------------------------------------------------------

/// Path to the `<case>.sha256` baseline under the given backend subdir of
/// `tests/fixtures/`.
fn baseline_path(baseline_subdir: &str, case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(baseline_subdir)
        .join(format!("{case}.sha256"))
}

/// Read the baseline for `case` and assert it matches `hash`.
///
/// Returns `Ok(())` on match; `Err(msg)` on mismatch or a missing/malformed
/// baseline. The missing-baseline error names the ignored `parity_regen_*`
/// test as the remedy — see [`write_baseline`].
fn assert_baseline(baseline_subdir: &str, case: &str, hash: &str) -> Result<(), String> {
    let path = baseline_path(baseline_subdir, case);

    if !path.exists() {
        let case_lower = case.to_lowercase();
        return Err(format!(
            "baseline file for {case} is missing at {}; \
             run the ignored `parity_regen_{case_lower}` test \
             (--run-ignored ignored-only) to generate it",
            path.display()
        ));
    }

    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read baseline for {case}: {e}"))?;
    let expected = raw.trim();

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

/// Write `hash` as the baseline for `case`, creating the baseline directory
/// when needed. Only the ignored `parity_regen_*` tests call this.
///
/// Returns `Ok(())` on success; `Err(msg)` on a write failure.
fn write_baseline(baseline_subdir: &str, case: &str, hash: &str) -> Result<(), String> {
    let path = baseline_path(baseline_subdir, case);

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("cannot create baseline dir: {e}"))?;
    }
    std::fs::write(&path, format!("{hash}\n"))
        .map_err(|e| format!("cannot write baseline for {case}: {e}"))?;
    eprintln!("REGEN: wrote baseline for {case}: {hash}");
    Ok(())
}

/// Map a golden-case label (e.g. `"D06"`) to its deterministic fixture directory.
///
/// The five labels are deliberately non-sequential: they are the
/// feature-combined subset of the contiguous `d01..` deterministic suite whose
/// union spans the cross-feature LP / scaling / cut / basis interactions that
/// hide dormant bugs (the tier-1 golden-parity contract). Each pins a distinct
/// piece of machinery:
///
/// - `D06` (`d06-fpha-variable-head`) — variable-head FPHA average-storage
///   coefficient on both incoming and outgoing storage columns.
/// - `D15` (`d15-non-controllable-source`) — NCS availability factor and the
///   lower-bound-patches-NCS contract.
/// - `D30` (`d30-multi-resolution-monthly-quarterly`) — multi-resolution
///   monthly/quarterly decomposition.
/// - `D34` (`d34-anticipated-varying-blocks`) — anticipated dispatch under
///   non-uniform per-stage block counts.
/// - `D41` (`d41-energy-contracts`) — energy import/export contract columns.
///
/// The simpler `d01..d05` single-feature cases are covered at the behavioral
/// tier (`deterministic.rs`); promoting a case into this golden set needs
/// justification.
///
/// ## `D30` re-bless provenance (residual-std-ratio closure derivation)
///
/// `D30`'s golden hash (both backends) was re-blessed when `residual_std_ratio`
/// became closure-derived (unit-variance-exact, from the AR coefficients alone)
/// instead of read from `inflow_ar_coefficients.parquet`. This is NOT the
/// generic mixed-per-season-order shift (~1e-4, see
/// `derive_residual_std_ratios`'s doc): `D30` supplies its AR coefficients
/// directly (the `out_of_sample` scheme never runs the YW fit) with a uniform
/// order of 1 at every season it uses, so the closure is exact there —
/// `r = sqrt(1 - psi^2) = 0.866025403784439` for `psi = 0.5`. The fixture's
/// stored ratio, `0.85`, was simply an arbitrary literal never fit against
/// `psi`, so the ~1.6e-2 gap between it and the derived value reflects the
/// stored-value inconsistency this migration retires, not a closure defect —
/// `D06`/`D15`/`D34`/`D41`'s unchanged hashes confirm the closure is exact
/// wherever a case's per-season orders are uniform. The determinism property
/// (same input -> same output; declaration-order / rank invariance) is
/// untouched; only this one blessed value moved.
pub fn case_dir(label: &str) -> PathBuf {
    let suffix = match label {
        "D06" => "d06-fpha-variable-head",
        "D15" => "d15-non-controllable-source",
        "D30" => "d30-multi-resolution-monthly-quarterly",
        "D34" => "d34-anticipated-varying-blocks",
        "D41" => "d41-energy-contracts",
        other => panic!("unknown case label: {other}"),
    };
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/deterministic")
        .join(suffix)
}

/// Train and simulate the case at `dir` under the given LP solver, returning
/// the built [`StudySetup`] and its (unsorted) simulation scenario results.
pub fn train_and_simulate_at_dir<S, F>(
    dir: &Path,
    make_solver: F,
) -> (StudySetup, Vec<SimulationScenarioResult>)
where
    S: cobre_solver::SolverInterface<Profile = cobre_solver::ActiveProfile> + Send,
    F: Fn() -> Result<S, cobre_solver::SolverError> + Copy,
{
    let config_path = dir.join("config.json");

    let config = cobre_io::parse_config(&config_path).expect("config must parse");
    let system = cobre_io::load_case(dir).expect("load_case must succeed");

    let prep_source = config
        .training_scenario_source(&config_path)
        .expect("training_scenario_source must parse");
    let pr = prepare_stochastic(system, dir, &config, 42, &prep_source, None)
        .expect("prepare_stochastic must succeed");
    let system = pr.system;
    let stochastic = pr.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, dir, false).expect("prepare_hydro_models must succeed");

    let mut config_with_sim = config.clone();
    config_with_sim.simulation.enabled = true;
    config_with_sim.simulation.selection = Some(SimulationSelection::Sampled { num_scenarios: 1 });

    let sentinel = Path::new("config.json");
    let training_source = config_with_sim
        .training_scenario_source(sentinel)
        .expect("training_scenario_source must parse");
    let simulation_source = config_with_sim
        .simulation_scenario_source(sentinel)
        .expect("simulation_scenario_source must parse");

    let params = StudyParams::from_config(&config_with_sim, Vec::new())
        .expect("StudyParams::from_config must succeed");

    let mut setup = StudySetup::from_broadcast_params(
        &system,
        stochastic,
        params,
        hydro_models,
        &training_source,
        &simulation_source,
    )
    .expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = make_solver().expect("solver construction must succeed");

    let (event_tx, _event_rx) = mpsc::channel::<TrainingEvent>();

    let outcome = setup
        .train(&mut solver, &comm, 1, make_solver, Some(event_tx), None)
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "{}: expected no training error",
        dir.display()
    );
    let result = outcome.result;

    let mut pool = setup
        .create_workspace_pool(&comm, 1, make_solver)
        .expect("simulation workspace pool must build");

    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

    let local_costs = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            result.frozen_templates.as_deref(),
            &result.basis_cache,
        )
        .expect("simulate must return Ok");

    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");

    let sim_config = setup.simulation_config();
    let (_summary, _gathered) = aggregate_simulation(
        &local_costs.costs,
        sim_config,
        &comm,
        SimulationWeighting::Uniform,
    )
    .expect("aggregate_simulation must succeed");

    (setup, scenario_results)
}

/// Run the full train + simulate pipeline for the case at `dir` under the
/// given LP solver and return its parity hash.
///
/// Generic over the solver so [`compute_golden_case_hash`],
/// [`compute_permuted_case_hash`], and every backend `parity_hash_*`/
/// `parity_regen_*`/`shuffle_matrix_*` test share one body; `make_solver`
/// supplies fresh worker solvers exactly as the production
/// training/simulation paths do.
fn compute_case_hash_at_dir<S, F>(dir: &Path, make_solver: F) -> String
where
    S: cobre_solver::SolverInterface<Profile = cobre_solver::ActiveProfile> + Send,
    F: Fn() -> Result<S, cobre_solver::SolverError> + Copy,
{
    let (setup, scenario_results) = train_and_simulate_at_dir::<S, F>(dir, make_solver);
    compute_parity_hash(&setup, scenario_results)
}

/// Run [`compute_case_hash_at_dir`] for a golden case's own directory.
fn compute_golden_case_hash<S, F>(label: &str, make_solver: F) -> String
where
    S: cobre_solver::SolverInterface<Profile = cobre_solver::ActiveProfile> + Send,
    F: Fn() -> Result<S, cobre_solver::SolverError> + Copy,
{
    compute_case_hash_at_dir::<S, F>(&case_dir(label), make_solver)
}

/// Run [`compute_case_hash_at_dir`] on one [`permute_case`] copy of `label`'s
/// golden case directory, seeded from `seed` — the declaration-order-shuffle
/// axis's per-permutation hash. See [`super::permute`]'s module doc for the
/// classification map and coverage bound.
fn compute_permuted_case_hash<S, F>(label: &str, seed: u64, make_solver: F) -> String
where
    S: cobre_solver::SolverInterface<Profile = cobre_solver::ActiveProfile> + Send,
    F: Fn() -> Result<S, cobre_solver::SolverError> + Copy,
{
    let permuted = permute_case(&case_dir(label), seed);
    compute_case_hash_at_dir::<S, F>(permuted.path(), make_solver)
}

/// Run [`compute_golden_case_hash`] for a golden case and assert the result
/// against the committed baseline, returning the computed hash.
///
/// `baseline_subdir` selects the per-backend baseline directory under
/// `tests/fixtures/`.
pub fn run_golden_case<S, F>(baseline_subdir: &str, label: &str, make_solver: F) -> String
where
    S: cobre_solver::SolverInterface<Profile = cobre_solver::ActiveProfile> + Send,
    F: Fn() -> Result<S, cobre_solver::SolverError> + Copy,
{
    let hash = compute_golden_case_hash::<S, F>(label, make_solver);
    assert_baseline(baseline_subdir, label, &hash).unwrap_or_else(|msg| panic!("{msg}"));
    hash
}

/// Run [`compute_golden_case_hash`] for a golden case and write the result as
/// the committed baseline, overwriting whatever is there.
///
/// Only the ignored `parity_regen_<case>` tests call this. `baseline_subdir`
/// selects the per-backend baseline directory under `tests/fixtures/`.
pub fn regen_golden_case<S, F>(baseline_subdir: &str, label: &str, make_solver: F)
where
    S: cobre_solver::SolverInterface<Profile = cobre_solver::ActiveProfile> + Send,
    F: Fn() -> Result<S, cobre_solver::SolverError> + Copy,
{
    let hash = compute_golden_case_hash::<S, F>(label, make_solver);
    write_baseline(baseline_subdir, label, &hash).unwrap_or_else(|msg| panic!("{msg}"));
}

/// Assert [`compute_permuted_case_hash`] (`label`, `seed`) equals `base_hash` —
/// the declaration-order-shuffle axis's core assertion, folded into every
/// `parity_hash_<case>`/`shuffle_matrix_<case>` test in `tests/parity.rs`. The
/// comparison target is the caller's in-memory base hash, not the committed
/// baseline: it holds even on a machine where the baseline itself does not
/// reproduce (parity hashes are environment-dependent; base-vs-permuted
/// equality is not).
pub fn assert_permutation_hash<S, F>(label: &str, seed: u64, base_hash: &str, make_solver: F)
where
    S: cobre_solver::SolverInterface<Profile = cobre_solver::ActiveProfile> + Send,
    F: Fn() -> Result<S, cobre_solver::SolverError> + Copy,
{
    let permuted_hash = compute_permuted_case_hash::<S, F>(label, seed, make_solver);
    assert_eq!(
        base_hash, permuted_hash,
        "declaration-order invariance violated for {label} at seed {seed}:\n  \
         base hash:     {base_hash}\n  permuted hash: {permuted_hash}"
    );
}
