//! Shared test utilities for `cobre-sddp` integration tests.
//!
//! [`build_setup_for_case`] is a drop-in replacement for `StudySetup::new` that
//! drives the same construction pipeline as the CLI.

#![allow(clippy::expect_used, dead_code)]

use std::path::Path;
use std::sync::mpsc;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::scenario::SamplingScheme;
use cobre_core::{BlockMode, System};
use cobre_io::Config;
use cobre_sddp::{
    BoundaryStateRequirements, SimulationScenarioResult, StudySetup,
    hydro_models::{PrepareHydroModelsResult, prepare_hydro_models},
    setup::{StudyParams, prepare_stochastic},
    test_support::decks::{SLOW_DECKS, committed_decks},
};
use cobre_solver::ActiveSolver;
use cobre_stochastic::{
    ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context,
};

pub mod anticipated_structural_assertions;
pub mod builders;
pub mod cut_oracles;
pub mod in_code_studies;
pub mod parity_hash;
pub mod parquet_fixtures;
pub mod permute;

/// Single-rank `Communicator` stub: broadcasts/reductions copy data locally;
/// other collectives are no-ops.
pub struct StubComm;

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

/// `size() == 2` sibling of [`StubComm`]: every collective writes only this
/// rank's own slot (`recv[displs[0]..displs[0] + send.len()]`), mirroring
/// `state_exchange.rs`'s own `Rank1Of2` test pattern rather than echoing rank
/// 0's data into rank 1's slot. Faithful — not a dishonest tautology — only
/// when the caller trains with `forward_passes == 1`: `RankDistribution`
/// (`base_fwd=0, remainder=1` for `num_ranks=2`) assigns rank 0 the sole real
/// forward pass and rank 1 exactly zero, so the zero contribution this stub
/// leaves unwritten IS what a genuine rank 1 would also send. Every caller
/// must keep its own fixture at `forward_passes == 1`.
pub struct Rank0Of2;

impl Communicator for Rank0Of2 {
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _counts: &[usize],
        displs: &[usize],
    ) -> Result<(), CommError> {
        let start = displs[0];
        recv[start..start + send.len()].clone_from_slice(send);
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
        2
    }

    fn abort(&self, error_code: i32) -> ! {
        std::process::exit(error_code)
    }
}

/// Resolve the boundary state requirements for a test case, tolerating a
/// placeholder boundary path: a test that installs a fake `policy.boundary` and
/// injects cuts manually must not trigger a read of the missing checkpoint. The
/// boundary is still PRESENT whenever `policy.boundary` is configured — only the
/// depth read is skipped — so a fake/unreadable checkpoint yields `present(0)`
/// (present, no inflow-lag slots), never `none()`. Routes through the production
/// resolver when a real checkpoint exists.
pub fn boundary_requirements(case_dir: &Path, config: &Config) -> BoundaryStateRequirements {
    let Some(bp) = config.policy.boundary.as_ref() else {
        return BoundaryStateRequirements::none();
    };
    if !bp.checkpoint_path(case_dir).join("manifest.bin").exists() {
        return BoundaryStateRequirements::present(0);
    }
    cobre_sddp::resolve_boundary_state_requirements(case_dir, config)
        .unwrap_or_else(|_| BoundaryStateRequirements::present(0))
}

/// Build a [`StudySetup`] for a case directory.
///
/// The caller's `prepare_hydro_models` has already folded the productivity
/// override into `hydro_models`. This helper re-loads `case_dir`'s
/// `CaseArtifacts` (a second full parse) for `scalar_parameters`, because
/// `StudyParams::from_config` leaves them empty (they load from disk artifacts,
/// not `Config`) — every setup caller must patch them in itself (cobre-cli via
/// MPI broadcast, cobre-python directly).
pub fn build_setup_for_case(
    case_dir: &Path,
    config: &Config,
    system: &System,
    stochastic: StochasticContext,
    hydro_models: PrepareHydroModelsResult,
) -> StudySetup {
    let sentinel = Path::new("config.json");
    let training_source = config
        .training_scenario_source(sentinel)
        .expect("training_scenario_source must parse");
    let simulation_source = config
        .simulation_scenario_source(sentinel)
        .expect("simulation_scenario_source must parse");

    let mut construction = StudyParams::from_config(config, Vec::new())
        .expect("StudyParams::from_config must succeed");
    construction.boundary = boundary_requirements(case_dir, config);
    construction.scalar_parameters = cobre_io::load_case_with_artifacts(case_dir)
        .expect("load_case_with_artifacts must succeed")
        .artifacts
        .scalar_parameters;

    StudySetup::from_broadcast_params(
        system,
        stochastic,
        construction,
        hydro_models,
        &training_source,
        &simulation_source,
    )
    .expect("StudySetup::from_broadcast_params must build")
}

/// Build a fresh [`System`] and [`StudySetup`] from `case_dir`'s config:
/// applies `mutate`, then derives the training scenario source from the
/// mutated config. Returns the post-`prepare_stochastic` [`System`] that
/// built the setup — `build_setup_for_case` only borrows it, so both are
/// available to the caller.
pub fn fresh_system_and_setup_with(
    case_dir: &Path,
    mutate: impl FnOnce(&mut Config),
) -> (System, StudySetup) {
    let config_path = case_dir.join("config.json");
    let mut config = cobre_io::parse_config(&config_path).expect("config must parse");
    mutate(&mut config);
    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let training_source = config
        .training_scenario_source(&config_path)
        .expect("training_scenario_source must parse");
    let inflow_lag_depth = boundary_requirements(case_dir, &config).inflow_lag_depth();
    let prepare_result = prepare_stochastic(
        system,
        case_dir,
        &config,
        42,
        &training_source,
        inflow_lag_depth,
    )
    .expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");

    let setup = build_setup_for_case(case_dir, &config, &system, stochastic, hydro_models);
    (system, setup)
}

/// Build a fresh [`StudySetup`] from `case_dir`'s config: applies `mutate`,
/// then derives the training scenario source from the mutated config — the
/// pipeline shared by every `mpi_wire.rs` determinism gate's `fresh_setup`.
pub fn fresh_setup_with(case_dir: &Path, mutate: impl FnOnce(&mut Config)) -> StudySetup {
    fresh_system_and_setup_with(case_dir, mutate).1
}

/// Recursively copies `src` into `dst`, skipping any `output` subdirectory
/// (`permute.rs`'s own copy discipline for that gitignored generated tree).
fn copy_dir_recursive(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst)
        .unwrap_or_else(|e| panic!("create_dir_all {}: {e}", dst.display()));
    for entry in
        std::fs::read_dir(src).unwrap_or_else(|e| panic!("read_dir {}: {e}", src.display()))
    {
        let entry = entry.unwrap_or_else(|e| panic!("dir entry under {}: {e}", src.display()));
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        let file_type = entry
            .file_type()
            .unwrap_or_else(|e| panic!("file_type {}: {e}", src_path.display()));
        if file_type.is_dir() {
            if entry.file_name() == "output" {
                continue;
            }
            copy_dir_recursive(&src_path, &dst_path);
        } else {
            std::fs::copy(&src_path, &dst_path).unwrap_or_else(|e| {
                panic!("copy {} -> {}: {e}", src_path.display(), dst_path.display())
            });
        }
    }
}

/// Builds `case_dir` with every `stages.json` stage forced to `mode`: copies
/// the case into a temporary directory, rewrites its `"block_mode"` fields,
/// and builds through [`fresh_system_and_setup_with`] under `catch_unwind`,
/// turning a build panic into `Err(message)`.
pub fn fresh_system_and_setup_in_block_mode(
    case_dir: &Path,
    mode: BlockMode,
) -> Result<(System, StudySetup), String> {
    let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("tempdir: {e}"));
    copy_dir_recursive(case_dir, tmp.path());

    let stages_path = tmp.path().join("stages.json");
    let text = std::fs::read_to_string(&stages_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", stages_path.display()));
    let mut value: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("parse {}: {e}", stages_path.display()));
    let mode_str = match mode {
        BlockMode::Parallel => "parallel",
        BlockMode::Chronological => "chronological",
    };
    let stages = value
        .get_mut("stages")
        .and_then(serde_json::Value::as_array_mut)
        .unwrap_or_else(|| {
            panic!(
                "{} has no top-level \"stages\" array",
                stages_path.display()
            )
        });
    for stage in stages {
        stage
            .as_object_mut()
            .unwrap_or_else(|| panic!("{} stage entry is not a JSON object", stages_path.display()))
            .insert(
                "block_mode".to_string(),
                serde_json::Value::String(mode_str.to_string()),
            );
    }
    let rendered = serde_json::to_string_pretty(&value)
        .unwrap_or_else(|e| panic!("serialize {}: {e}", stages_path.display()));
    std::fs::write(&stages_path, rendered)
        .unwrap_or_else(|e| panic!("write {}: {e}", stages_path.display()));

    let case_path = tmp.path().to_path_buf();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fresh_system_and_setup_with(&case_path, |_| {})
    }))
    .map_err(|payload| {
        payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("<non-string panic payload>")
            .to_string()
    })
}

/// Deck keys whose block-mode flip is rejected by a `cobre-io` validation
/// rule, paired with the rejection's expected message substring: a deck
/// here must still fail its flip with that substring, or the entry is
/// stale.
pub const BLOCK_MODE_FLIP_REJECTED: &[(&str, &str)] = &[];

/// Resolves [`fresh_system_and_setup_in_block_mode`]'s result against
/// [`BLOCK_MODE_FLIP_REJECTED`]: `None` for a listed, still-failing key
/// (after asserting its substring), the built pair otherwise, and a panic
/// for any other failure.
fn build_flip_or_panic(
    key: &str,
    case_dir: &Path,
    mode: BlockMode,
) -> Option<(System, StudySetup)> {
    match fresh_system_and_setup_in_block_mode(case_dir, mode) {
        Ok(pair) => Some(pair),
        Err(msg) => match BLOCK_MODE_FLIP_REJECTED.iter().find(|(k, _)| *k == key) {
            Some((_, substring)) => {
                assert!(
                    msg.contains(substring),
                    "deck {key} in block mode {mode:?}: BLOCK_MODE_FLIP_REJECTED substring \
                     {substring:?} not found in failure {msg:?}"
                );
                None
            }
            None => panic!("deck {key} in block mode {mode:?} failed to build: {msg}"),
        },
    }
}

/// Visits every committed deck (skipping [`SLOW_DECKS`] unless `slow-tests`
/// is enabled) that has a stage of `n_blks > 1` — read from the [`Parallel`]
/// flip's `block_hours_per_stage` — building both mode flips and visiting
/// `(key, chronological, parallel)`; returns the visited count.
///
/// [`Parallel`]: BlockMode::Parallel
pub fn for_each_deck_in_both_block_modes(
    mut visit: impl FnMut(&str, (&System, &StudySetup), (&System, &StudySetup)),
) -> usize {
    let slow_tests_enabled = cfg!(feature = "slow-tests");
    let mut count = 0;
    for deck in committed_decks() {
        if !slow_tests_enabled && SLOW_DECKS.contains(&deck.key.as_str()) {
            continue;
        }
        let Some((parallel_system, parallel_setup)) =
            build_flip_or_panic(&deck.key, &deck.dir, BlockMode::Parallel)
        else {
            continue;
        };
        let has_multi_block = parallel_setup
            .inputs
            .stage_data
            .stage_templates
            .block_hours_per_stage
            .iter()
            .any(|hours| hours.len() > 1);
        if !has_multi_block {
            continue;
        }
        let Some((chrono_system, chrono_setup)) =
            build_flip_or_panic(&deck.key, &deck.dir, BlockMode::Chronological)
        else {
            continue;
        };
        visit(
            &deck.key,
            (&chrono_system, &chrono_setup),
            (&parallel_system, &parallel_setup),
        );
        count += 1;
    }
    count
}

/// Visits every committed deck (skipping [`SLOW_DECKS`] unless `slow-tests` is
/// enabled), then every [`in_code_studies::structural_studies`] entry, building
/// one [`System`] and [`StudySetup`] at a time; returns the total visit count.
pub fn for_each_study(mut visit: impl FnMut(&str, &System, &StudySetup)) -> usize {
    let slow_tests_enabled = cfg!(feature = "slow-tests");
    let mut count = 0;
    for deck in committed_decks() {
        if !slow_tests_enabled && SLOW_DECKS.contains(&deck.key.as_str()) {
            continue;
        }
        let (system, setup) = fresh_system_and_setup_with(&deck.dir, |_| {});
        visit(&deck.key, &system, &setup);
        count += 1;
    }
    for (key, system, setup) in in_code_studies::structural_studies() {
        visit(&key, &system, &setup);
        count += 1;
    }
    count
}

/// [`for_each_study`]'s visit count, for a caller confirming its sweep saw
/// every study: every committed deck minus its [`SLOW_DECKS`] skips, plus
/// every [`in_code_studies::structural_studies`] entry.
#[must_use]
pub fn expected_study_count() -> usize {
    let slow_tests_enabled = cfg!(feature = "slow-tests");
    let skipped = if slow_tests_enabled {
        0
    } else {
        committed_decks()
            .iter()
            .filter(|deck| SLOW_DECKS.contains(&deck.key.as_str()))
            .count()
    };
    committed_decks().len() - skipped + in_code_studies::structural_studies().len()
}

/// Build a [`StochasticContext`] for an in-code `System`, hermetic (no external
/// scenario files) — the construction shared by [`build_setup_in_code`] and
/// [`try_build_setup_in_code`].
pub fn stochastic_in_code(system: &System) -> StochasticContext {
    build_stochastic_context(
        system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("build_stochastic_context")
}

/// Fallible core shared by every in-code `StudySetup` constructor below: the
/// sole call site of `StudySetup::new` in this module.
#[allow(clippy::needless_pass_by_value)]
fn try_build_setup_in_code_with_models(
    system: System,
    config: &Config,
    hydro_models: PrepareHydroModelsResult,
) -> Result<StudySetup, cobre_sddp::SddpError> {
    let stochastic = stochastic_in_code(&system);
    StudySetup::new(&system, config, stochastic, hydro_models, Vec::new())
}

/// Construct a [`StudySetup`] in-process from an explicit `hydro_models`
/// result, building the stochastic context directly so the test stays
/// hermetic (no external scenario files). Use this over [`build_setup_in_code`]
/// when the fixture needs an active production/evaporation model
/// [`PrepareHydroModelsResult::default_from_system`] cannot express.
#[allow(clippy::needless_pass_by_value)]
pub fn build_setup_in_code_with_models(
    system: System,
    config: &Config,
    hydro_models: PrepareHydroModelsResult,
) -> StudySetup {
    try_build_setup_in_code_with_models(system, config, hydro_models).expect("StudySetup::new")
}

/// Construct a [`StudySetup`] in-process, building the stochastic context
/// directly so the test stays hermetic (no external scenario files).
// Taken by value so callers pass an owned `System` inline without a separate
// binding; the body only borrows it.
#[allow(clippy::needless_pass_by_value)]
pub fn build_setup_in_code(system: System, config: &Config) -> StudySetup {
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    build_setup_in_code_with_models(system, config, hydro_models)
}

/// Fallible sibling of [`build_setup_in_code`]: returns `StudySetup::new`'s
/// `Result` so a test can assert a setup-time rejection (e.g. the admission
/// gate) instead of panicking.
#[allow(clippy::needless_pass_by_value)]
pub fn try_build_setup_in_code(
    system: System,
    config: &Config,
) -> Result<StudySetup, cobre_sddp::SddpError> {
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    try_build_setup_in_code_with_models(system, config, hydro_models)
}

/// Train `iterations`, then run the one-scenario simulation and return the drained
/// per-scenario results.
pub fn run_simulation(setup: &mut StudySetup, iterations: usize) -> Vec<SimulationScenarioResult> {
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new: must succeed");

    let outcome = setup
        .train(
            &mut solver,
            &comm,
            iterations,
            ActiveSolver::new,
            None,
            None,
        )
        .expect("training error: train() must not return Err");
    assert!(
        outcome.error.is_none(),
        "training error: training returned an error: {:?}",
        outcome.error,
    );

    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("workspace pool error: create_workspace_pool must succeed");
    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

    setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            None,
            &outcome.result.basis_cache,
        )
        .expect("simulation error: simulate() must not return Err");
    // Drop the sender before the join below; a live result_tx keeps
    // result_rx.into_iter() blocked forever, deadlocking the join.
    drop(result_tx);
    drain_handle.join().expect("drain thread must not panic")
}
