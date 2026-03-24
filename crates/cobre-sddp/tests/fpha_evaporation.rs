//! End-to-end convergence test for the 4ree-fpha-evap case.
//!
//! Exercises the full pipeline -- case loading, hydro model preprocessing,
//! study setup, training, and simulation -- for a system with 4 hydro plants,
//! 2 FPHA production models, and evaporation modeled on 3 plants.
//!
//! ## Design constraints
//!
//! - Only the public `cobre_sddp::` and `cobre_io::` APIs are used.
//! - `StubComm` is defined locally (same pattern as `integration.rs`).
//! - The test is self-contained with no cross-test shared state.
//! - Path is relative to the crate root: `../../examples/4ree-fpha-evap/`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::path::Path;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_sddp::{
    EvaporationModel, EvaporationReferenceSource, ResolvedProductionModel, StudySetup,
    aggregate_simulation, hydro_models::prepare_hydro_models, setup::prepare_stochastic,
};
use cobre_solver::highs::HighsSolver;

// ===========================================================================
// Shared helpers
// ===========================================================================

/// Single-rank communicator that correctly copies data through `allgatherv`
/// and `allreduce`. Required by the exchange and forward-sync steps so that
/// state is available to the backward pass.
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
}

// ===========================================================================
// E2E convergence test
// ===========================================================================

/// Given the `examples/4ree-fpha-evap/` case directory, the full pipeline
/// -- load case, prepare hydro models, build study setup, train, simulate --
/// must complete without error and produce physically meaningful results.
#[test]
fn fpha_evaporation_case_converges() {
    // Cargo test runs from the crate root; the examples directory is two
    // levels up from `crates/cobre-sddp/`.
    let case_dir = Path::new("../../examples/4ree-fpha-evap");

    // ── Step 1: load config ───────────────────────────────────────────────────

    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    // ── Step 2: load case (system) ────────────────────────────────────────────

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    assert_eq!(
        system.hydros().len(),
        4,
        "system must contain exactly 4 hydro plants"
    );

    // ── Step 3: prepare stochastic context (rank-0 pipeline) ──────────────────

    let prepare_result =
        prepare_stochastic(system, case_dir, &config, 42).expect("prepare_stochastic must succeed");

    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    // ── Step 4: prepare hydro models ──────────────────────────────────────────

    let hydro_models =
        prepare_hydro_models(&system, case_dir).expect("prepare_hydro_models must succeed");

    // Acceptance criterion: 2 FPHA hydros in the production model set.
    let n_fpha = (0..hydro_models.production.n_hydros())
        .filter(|&h| {
            // Stage 0 is representative; FPHA hydros use the Fpha variant.
            matches!(
                hydro_models.production.model(h, 0),
                ResolvedProductionModel::Fpha { .. }
            )
        })
        .count();
    assert_eq!(n_fpha, 2, "production model set must have 2 FPHA hydros");

    // Acceptance criterion: evaporation set must report has_evaporation() == true.
    assert!(
        hydro_models.evaporation.has_evaporation(),
        "evaporation model set must have at least one linearized evaporation model"
    );

    // ── Step 5: build StudySetup ──────────────────────────────────────────────

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup must build");

    // Acceptance criterion: templates for all 12 study stages.
    assert_eq!(
        setup.stage_templates().len(),
        12,
        "study setup must have 12 stage templates"
    );

    // ── Step 6: train ─────────────────────────────────────────────────────────

    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");

    let training_result = setup
        .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
        .expect("train must return Ok");

    // Acceptance criterion: convergence within the 256-iteration budget.
    assert!(
        training_result.iterations <= 256,
        "training must converge within 256 iterations; got {}",
        training_result.iterations
    );

    // ── Step 7: simulate ──────────────────────────────────────────────────────

    let mut pool = setup
        .create_workspace_pool(1, HighsSolver::new)
        .expect("simulation workspace pool must build");

    let io_capacity = setup.io_channel_capacity().max(1);
    let (result_tx, result_rx) = std::sync::mpsc::sync_channel(io_capacity);

    // Drain thread collects all SimulationScenarioResult items from the channel.
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

    let local_costs = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            &training_result.basis_cache,
        )
        .expect("simulate must return Ok");

    // Drop the sender so the drain thread terminates.
    drop(result_tx);
    let _scenario_results = drain_handle.join().expect("drain thread must not panic");

    // Aggregate to obtain SimulationSummary.
    let sim_config = setup.simulation_config();
    let summary = aggregate_simulation(&local_costs.costs, &sim_config, &comm)
        .expect("aggregate_simulation must succeed");

    // Acceptance criterion: n_scenarios == 100 and mean_cost > 0.
    assert_eq!(
        summary.n_scenarios, 100,
        "simulation must produce exactly 100 scenarios"
    );
    assert!(
        summary.mean_cost > 0.0,
        "simulation mean cost must be positive; got {}",
        summary.mean_cost
    );
}

// ===========================================================================
// Provenance integration test
// ===========================================================================

/// Verifies that after loading the updated 4ree-fpha-evap case and running
/// `prepare_hydro_models`, the evaporation reference provenance is correct:
///
/// - Hydro 0 (FPHA + user-supplied `reference_volumes_hm3`) → `UserSupplied`
/// - Hydro 1 (FPHA + user-supplied `reference_volumes_hm3`) → `UserSupplied`
/// - Hydro 2 (constant productivity + no `reference_volumes_hm3`) → `DefaultMidpoint`
///
/// Also verifies that the per-stage reference volumes for hydro 0 match the
/// expected season-to-volume mapping from the `reference_volumes_hm3` array.
/// Stages 0-11 have `season_ids` 0-11 (January-December), so each stage's
/// reference volume must equal `reference_volumes_hm3[season_id]`.
#[test]
fn test_4ree_fpha_evap_seasonal_ref_provenance() {
    let case_dir = Path::new("../../examples/4ree-fpha-evap");

    // ── Load system ───────────────────────────────────────────────────────────

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    assert_eq!(
        system.hydros().len(),
        4,
        "system must contain exactly 4 hydro plants"
    );

    // ── Run hydro model preprocessing ────────────────────────────────────────

    let result =
        prepare_hydro_models(&system, case_dir).expect("prepare_hydro_models must succeed");

    // ── Verify provenance per hydro ───────────────────────────────────────────
    //
    // Hydros are stored in canonical ID order (0, 1, 2, 3).
    // Hydro 3 has no evaporation block; its reference provenance is DefaultMidpoint
    // (irrelevant but consistent, as documented in HydroModelProvenance).

    let ref_sources = &result.provenance.evaporation_reference_sources;
    assert_eq!(
        ref_sources.len(),
        4,
        "must have one reference source entry per hydro"
    );

    // Hydro 0: FPHA with user-supplied reference volumes.
    assert_eq!(
        ref_sources[0].1,
        EvaporationReferenceSource::UserSupplied,
        "hydro 0 must have UserSupplied reference source"
    );

    // Hydro 1: FPHA with user-supplied reference volumes.
    assert_eq!(
        ref_sources[1].1,
        EvaporationReferenceSource::UserSupplied,
        "hydro 1 must have UserSupplied reference source"
    );

    // Hydro 2: constant productivity with evaporation but no reference volumes.
    assert_eq!(
        ref_sources[2].1,
        EvaporationReferenceSource::DefaultMidpoint,
        "hydro 2 must have DefaultMidpoint reference source (no user-supplied volumes)"
    );

    // ── Verify per-stage reference volumes for hydro 0 ───────────────────────
    //
    // The stages in this case run January–December (season_ids 0–11).
    // The reference_volumes_hm3 array for hydro 0 is indexed by month (0-based),
    // so stage t (season_id = t) must use reference_volumes_hm3[t].
    //
    // Expected values from hydros.json hydro 0:
    //   Jan=140000, Feb=150000, Mar=160000, Apr=170000, May=170000, Jun=150000,
    //   Jul=130000, Aug=110000, Sep=100000, Oct=100000, Nov=110000, Dec=125000
    let expected_ref_vols: [f64; 12] = [
        140_000.0, 150_000.0, 160_000.0, 170_000.0, 170_000.0, 150_000.0, 130_000.0, 110_000.0,
        100_000.0, 100_000.0, 110_000.0, 125_000.0,
    ];

    match result.evaporation.model(0) {
        EvaporationModel::Linearized {
            reference_volumes_hm3,
            ..
        } => {
            assert_eq!(
                reference_volumes_hm3.len(),
                12,
                "hydro 0 must have 12 per-stage reference volumes (one per study stage)"
            );
            for (stage, (&actual, &expected)) in reference_volumes_hm3
                .iter()
                .zip(expected_ref_vols.iter())
                .enumerate()
            {
                assert!(
                    (actual - expected).abs() < 1e-9,
                    "hydro 0 stage {stage}: expected reference volume {expected}, got {actual}"
                );
            }
        }
        EvaporationModel::None => {
            panic!("hydro 0 must have a Linearized evaporation model, got None");
        }
    }
}
