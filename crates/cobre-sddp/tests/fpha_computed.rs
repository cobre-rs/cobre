//! End-to-end convergence test for computed-source FPHA hydro models.
//! Modifies 4ree-fpha-evap case to use `source: "computed"` for hydro 0,
//! exercises full pipeline (load, preprocess, train, simulate), and verifies
//! convergence and production model sources.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::{fs, path::Path};

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_sddp::{
    ProductionModelSource, ResolvedProductionModel, StudySetup, aggregate_simulation,
    hydro_models::prepare_hydro_models, setup::prepare_stochastic,
};
use cobre_solver::highs::HighsSolver;
use tempfile::TempDir;

// ===========================================================================
// Shared helpers
// ===========================================================================

/// Single-rank communicator stub (copies data via allgatherv/allreduce).
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
// Temporary case directory helpers
// ===========================================================================

/// Recursively copies a directory tree from `src` to `dst`.
fn copy_dir_recursive(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for entry in fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path);
        } else {
            fs::copy(&src_path, &dst_path).unwrap();
        }
    }
}

/// Build temp case overlay: 4ree-fpha-evap with hydro 0 changed to
/// `source: "computed"` and tailrace/losses/efficiency added.
/// Excludes `output/` so pipeline writes fresh results.
#[allow(clippy::too_many_lines, clippy::unreadable_literal)]
fn setup_computed_case() -> TempDir {
    let src = Path::new("../../examples/4ree-fpha-evap");
    let tmp = TempDir::new().unwrap();
    let dst = tmp.path();

    // Copy flat files at the case root.
    for name in &[
        "config.json",
        "initial_conditions.json",
        "penalties.json",
        "stages.json",
    ] {
        let src_path = src.join(name);
        let dst_path = dst.join(name);
        fs::copy(&src_path, &dst_path).unwrap();
    }

    // Copy the scenarios directory (contains .parquet files).
    copy_dir_recursive(&src.join("scenarios"), &dst.join("scenarios"));

    // Copy the system directory (contains all entity JSON and .parquet files).
    copy_dir_recursive(&src.join("system"), &dst.join("system"));

    // Overwrite hydro_production_models.json so hydro 0 uses `source: "computed"`.
    // Hydro 1 retains `source: "precomputed"` (seasonal mode, two seasons).
    let prod_models_json = serde_json::json!({
        "production_models": [
            {
                "hydro_id": 0,
                "selection_mode": "stage_ranges",
                "stage_ranges": [
                    {
                        "start_stage_id": 0,
                        "end_stage_id": null,
                        "model": "fpha",
                        "fpha_config": { "source": "computed" }
                    }
                ]
            },
            {
                "hydro_id": 1,
                "selection_mode": "seasonal",
                "default_model": "fpha",
                "seasons": [
                    {
                        "season_id": 0,
                        "model": "fpha",
                        "fpha_config": { "source": "precomputed" }
                    },
                    {
                        "season_id": 1,
                        "model": "fpha",
                        "fpha_config": { "source": "precomputed" }
                    }
                ]
            }
        ]
    });
    fs::write(
        dst.join("system/hydro_production_models.json"),
        serde_json::to_string_pretty(&prod_models_json).unwrap(),
    )
    .unwrap();

    // Overwrite hydros.json to add tailrace, hydraulic_losses, and efficiency
    // to hydro 0. These fields are required by the computed FPHA fitting path.
    // All other hydro fields are kept identical to the committed example.
    // Hydros 1, 2, and 3 are unchanged.
    let hydros_json = serde_json::json!({
        "hydros": [
            {
                "id": 0,
                "name": "1",
                "bus_id": 0,
                "downstream_id": null,
                "reservoir": {
                    "min_storage_hm3": 0.0,
                    "max_storage_hm3": 204078.3
                },
                "outflow": {
                    "min_outflow_m3s": 0.0,
                    "max_outflow_m3s": null
                },
                "generation": {
                    "model": "fpha",
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 45578.9,
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 45578.9
                },
                "tailrace": {
                    "type": "polynomial",
                    "coefficients": [350.0]
                },
                "hydraulic_losses": {
                    "type": "factor",
                    "value": 0.03
                },
                "efficiency": {
                    "type": "constant",
                    "value": 0.92
                },
                "evaporation": {
                    "coefficients_mm": [
                        150.0, 130.0, 120.0, 90.0, 60.0, 40.0, 30.0, 40.0, 70.0, 100.0,
                        130.0, 150.0
                    ],
                    "reference_volumes_hm3": [
                        140000, 150000, 160000, 170000, 170000, 150000, 130000, 110000,
                        100000, 100000, 110000, 125000
                    ]
                }
            },
            {
                "id": 1,
                "name": "2",
                "bus_id": 1,
                "downstream_id": null,
                "reservoir": {
                    "min_storage_hm3": 0.0,
                    "max_storage_hm3": 19929.2
                },
                "outflow": {
                    "min_outflow_m3s": 0.0,
                    "max_outflow_m3s": null
                },
                "generation": {
                    "model": "fpha",
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 13967.0,
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 13967.0
                },
                "evaporation": {
                    "coefficients_mm": [
                        140.0, 120.0, 110.0, 80.0, 55.0, 35.0, 25.0, 35.0, 65.0, 95.0,
                        120.0, 140.0
                    ],
                    "reference_volumes_hm3": [
                        14000, 15000, 16000, 17000, 17000, 15000, 12500, 10000, 9000, 9000,
                        10500, 12500
                    ]
                }
            },
            {
                "id": 2,
                "name": "3",
                "bus_id": 2,
                "downstream_id": null,
                "reservoir": {
                    "min_storage_hm3": 0.0,
                    "max_storage_hm3": 51806.1
                },
                "outflow": {
                    "min_outflow_m3s": 0.0,
                    "max_outflow_m3s": null
                },
                "generation": {
                    "model": "constant_productivity",
                    "productivity_mw_per_m3s": 1.0,
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 9573.1,
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 9573.1
                },
                "evaporation": {
                    "coefficients_mm": [
                        130.0, 110.0, 100.0, 75.0, 50.0, 30.0, 20.0, 30.0, 60.0, 90.0,
                        115.0, 135.0
                    ]
                }
            },
            {
                "id": 3,
                "name": "4",
                "bus_id": 3,
                "downstream_id": null,
                "reservoir": {
                    "min_storage_hm3": 0.0,
                    "max_storage_hm3": 15765.5
                },
                "outflow": {
                    "min_outflow_m3s": 0.0,
                    "max_outflow_m3s": null
                },
                "generation": {
                    "model": "constant_productivity",
                    "productivity_mw_per_m3s": 1.0,
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 9666.2,
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 9666.2
                }
            }
        ]
    });
    fs::write(
        dst.join("system/hydros.json"),
        serde_json::to_string_pretty(&hydros_json).unwrap(),
    )
    .unwrap();

    tmp
}

// ===========================================================================
// E2E convergence test
// ===========================================================================

/// Full pipeline with computed-source FPHA: load, prepare, train, simulate.
/// Must converge within 256 iterations and produce valid simulation summary.
#[test]
fn fpha_computed_case_converges() {
    let tmp = setup_computed_case();
    let case_dir = tmp.path();

    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");
    assert_eq!(system.hydros().len(), 4, "system must have 4 hydros");

    let prepare_result =
        prepare_stochastic(system, case_dir, &config, 42).expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir).expect("prepare_hydro_models must succeed");

    // Verify hydro 0 has ComputedFromGeometry provenance.
    let production_sources = &hydro_models.provenance.production_sources;
    assert_eq!(
        production_sources[0].1,
        ProductionModelSource::ComputedFromGeometry,
        "hydro 0 must have ComputedFromGeometry provenance"
    );

    // Verify production model set has 2 FPHA hydros (hydros 0 and 1).
    let n_fpha = (0..hydro_models.production.n_hydros())
        .filter(|&h| {
            matches!(
                hydro_models.production.model(h, 0),
                ResolvedProductionModel::Fpha { .. }
            )
        })
        .count();
    assert_eq!(n_fpha, 2, "production model set must have 2 FPHA hydros");

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup must build");
    assert_eq!(
        setup.stage_templates().len(),
        12,
        "must have 12 study stages"
    );

    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");
    let training_result = setup
        .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
        .expect("train must succeed");
    assert!(
        training_result.iterations <= 256,
        "training convergence within 256 iterations; got {}",
        training_result.iterations
    );

    let mut pool = setup
        .create_workspace_pool(1, HighsSolver::new)
        .expect("workspace pool must build");
    let io_capacity = setup.io_channel_capacity().max(1);
    let (result_tx, result_rx) = std::sync::mpsc::sync_channel(io_capacity);
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());
    let local_costs = setup
        .simulate(&mut pool.workspaces, &comm, &result_tx, None)
        .expect("simulate must succeed");
    drop(result_tx);
    drop(drain_handle.join().expect("drain thread must not panic"));

    let sim_config = setup.simulation_config();
    let summary = aggregate_simulation(&local_costs, &sim_config, &comm)
        .expect("aggregate_simulation must succeed");
    assert_eq!(summary.n_scenarios, 100, "must simulate 100 scenarios");
    assert!(summary.mean_cost > 0.0, "mean cost must be positive");
}
