//! Deterministic test suite for the SDDP pipeline.
//!
//! Each case is fully deterministic — constant load (zero variance), no
//! stochastic inflows — so the optimal cost is hand-computable, which is what
//! licenses the exact cost asserts and tight convergence bounds below.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use cobre_io::config::SimulationSelection;
use std::path::Path;
use std::sync::mpsc;

use cobre_core::scenario::ScenarioSource;
use cobre_core::{BlockMode, EntityId};
use cobre_io::{
    PolicyCutRecord, STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, StageCutsPayload,
    write_policy_checkpoint,
};
use cobre_sddp::{
    SimulationWeighting, StudySetup, aggregate_simulation, hydro_models::prepare_hydro_models,
    indexer::HydroSys, lead_time::resolve_spread, setup::prepare_stochastic,
};
use cobre_solver::{ActiveSolver, SolverInterface};

mod common;
use common::{StubComm, build_setup_for_case, fresh_setup_with};

/// Train a case (`StubComm`, `ActiveSolver`, seed 42, 1 thread) and return the
/// setup, the canonicalized system, the training result, and the live solver.
fn train_deterministic_case(
    case_dir: &Path,
) -> (
    StudySetup,
    cobre_core::System,
    cobre_sddp::TrainingResult,
    ActiveSolver,
) {
    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let training_source = config
        .training_scenario_source(&config_path)
        .expect("training_scenario_source must parse");
    let prepare_result = prepare_stochastic(system, case_dir, &config, 42, &training_source, None)
        .expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");

    let mut setup = build_setup_for_case(case_dir, &config, &system, stochastic, hydro_models);

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");
    (setup, system, outcome.result, solver)
}

/// Train a case and return the result plus the live solver, so callers can
/// inspect `solver.statistics()` after training.
fn run_deterministic_with_solver(case_dir: &Path) -> (cobre_sddp::TrainingResult, ActiveSolver) {
    let (_setup, _system, result, solver) = train_deterministic_case(case_dir);
    (result, solver)
}

/// Train a case and return the setup (for post-train state introspection via
/// `stage_state()`, or driving a subsequent simulation), the canonicalized
/// system, and the training result.
fn run_deterministic_with_setup(
    case_dir: &Path,
) -> (StudySetup, cobre_core::System, cobre_sddp::TrainingResult) {
    let (setup, system, result, _solver) = train_deterministic_case(case_dir);
    (setup, system, result)
}

/// Train a case (`StubComm`, `ActiveSolver`, seed 42, 1 thread) and return the result.
fn run_deterministic(case_dir: &Path) -> cobre_sddp::TrainingResult {
    run_deterministic_with_solver(case_dir).0
}

/// Train with 1-scenario simulation enabled, then simulate, returning the
/// training result, per-scenario results, and the aggregate summary.
fn run_with_simulation(
    case_dir: &Path,
) -> (
    cobre_sddp::TrainingResult,
    Vec<cobre_sddp::SimulationScenarioResult>,
    cobre_sddp::SimulationSummary,
) {
    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let pr = prepare_stochastic(
        system,
        case_dir,
        &config,
        42,
        &ScenarioSource::default(),
        None,
    )
    .expect("prepare_stochastic must succeed");
    let system = pr.system;
    let stochastic = pr.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");

    let mut config_with_sim = config.clone();
    config_with_sim.simulation.enabled = true;
    config_with_sim.simulation.selection = Some(SimulationSelection::Sampled { num_scenarios: 1 });

    let mut setup = build_setup_for_case(
        case_dir,
        &config_with_sim,
        &system,
        stochastic,
        hydro_models,
    );

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");
    let result = outcome.result;

    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
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
    let (summary, _gathered) = aggregate_simulation(
        &local_costs.costs,
        sim_config,
        &comm,
        SimulationWeighting::Uniform,
    )
    .expect("aggregate_simulation must succeed");

    (result, scenario_results, summary)
}

fn assert_cost(actual: f64, expected: f64, tolerance: f64, case_name: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{case_name}: expected cost {expected}, got {actual} (diff={} > tolerance={tolerance})",
        (actual - expected).abs()
    );
}

/// Write a single-row `hydro_energy_productivity.parquet` supplying a per-hydro
/// `ρ_eq` override (`stage_id = NULL`, applies to all stages). Lets FPHA cases
/// without VHA geometry pass the FPHA correctness gate without changing LP economics.
fn write_energy_productivity_override(
    dest: &std::path::Path,
    hydro_id: i32,
    equivalent_productivity_mw_per_m3s: f64,
) {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, true),
        Field::new(
            "equivalent_productivity_mw_per_m3s",
            DataType::Float64,
            true,
        ),
        Field::new("reference_volume_hm3", DataType::Float64, true),
        Field::new("reference_outflow_m3s", DataType::Float64, true),
        Field::new(
            "specific_productivity_mw_per_m3s_per_m",
            DataType::Float64,
            true,
        ),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![hydro_id])),
            Arc::new(Int32Array::from(vec![None::<i32>])),
            Arc::new(Float64Array::from(vec![equivalent_productivity_mw_per_m3s])),
            Arc::new(Float64Array::from(vec![None::<f64>])),
            Arc::new(Float64Array::from(vec![None::<f64>])),
            Arc::new(Float64Array::from(vec![None::<f64>])),
        ],
    )
    .expect("valid RecordBatch for hydro_energy_productivity override");

    let file = std::fs::File::create(dest).expect("create hydro_energy_productivity.parquet");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter for override");
    writer.write(&batch).expect("write override batch");
    writer.close().expect("close override writer");
}

/// Recursively copies `src` into a fresh [`tempfile::TempDir`], skipping the
/// gitignored `output/` subtree. Materializes an alternate bound-file
/// configuration of a committed case without mutating the tracked fixture.
fn copy_case_dir(src: &Path) -> tempfile::TempDir {
    fn copy_recursive(src: &Path, dst: &Path) {
        std::fs::create_dir_all(dst).expect("create_dir_all for case copy");
        for entry in std::fs::read_dir(src).expect("read_dir for case copy") {
            let entry = entry.expect("dir entry for case copy");
            let file_type = entry.file_type().expect("file_type for case copy");
            let src_path = entry.path();
            if file_type.is_dir() {
                if entry.file_name() == "output" {
                    continue;
                }
                copy_recursive(&src_path, &dst.join(entry.file_name()));
            } else {
                std::fs::copy(&src_path, dst.join(entry.file_name()))
                    .expect("copy file for case copy");
            }
        }
    }
    let tmp = tempfile::tempdir().expect("tempdir must succeed");
    copy_recursive(src, tmp.path());
    tmp
}

/// Writes `constraints/thermal_bounds.parquet` at `dest` with a single
/// stage-wide (`block_id = NULL`) row overriding `max_generation_mw` for
/// `thermal_id` at `stage_id` — the hours-weighted-fold configuration (one
/// value applied to every block of the stage), as opposed to a per-block row.
fn write_stage_wide_thermal_bound(
    dest: &Path,
    thermal_id: i32,
    stage_id: i32,
    max_generation_mw: f64,
) {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![
        Field::new("thermal_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("max_generation_mw", DataType::Float64, true),
        Field::new("block_id", DataType::Int32, true),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![thermal_id])),
            Arc::new(Int32Array::from(vec![stage_id])),
            Arc::new(Float64Array::from(vec![max_generation_mw])),
            Arc::new(Int32Array::from(vec![None::<i32>])),
        ],
    )
    .expect("valid RecordBatch for stage-wide thermal bound override");

    let file = std::fs::File::create(dest).expect("create thermal_bounds.parquet override");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter for override");
    writer.write(&batch).expect("write override batch");
    writer.close().expect("close override writer");
}

/// Expected total cost for D02 (single hydro, 2 stages, deterministic inflows).
/// Derivation: κ=2.628, S₀=100hm³, inflows=[40,10]m³/s, demand=80MW.
/// Terminal stage turbines at 50m³/s capacity. Backward pass shows optimal
/// storage at stage boundary = 105.12 hm³.
///
/// Analytical thermal cost is `23_635_000 / 9 ≈ 2_626_111.11 $`. With
/// `turbined_cost = 0.01 $/MWh` applied to every hydro's turbine column
/// (see `fill_turbine_columns` in `lp::builder::columns`), the
/// deterministic LB adds a fixed regularization contribution of
/// `5_785 / 9 ≈ 642.78 $` (= 0.01 · 730 · (25_000/657 + 50) summed across the
/// two stages' turbined flows). Total = `23_640_785 / 9 ≈ 2_626_753.89 $`.
pub const D02_EXPECTED_COST: f64 = 23_640_785.0 / 9.0;

/// Expected total cost for D05 (FPHA constant-head, same physical setup as D02).
///
/// D05's `penalties.json` sets `turbined_cost = 0.0`, so the universal
/// turbined-cost regularization (which lifts D02 by `5_785 / 9 ≈ 642.78 $`)
/// contributes nothing here. The LB collapses to the pre-regularization
/// analytical value `23_635_000 / 9 ≈ 2_626_111.11 $` used to validate the
/// FPHA constant-head encoding against D02's hand-computed cost.
pub const D05_EXPECTED_COST: f64 = 23_635_000.0 / 9.0;

/// Two-stage pure thermal dispatch. Optimal cost is hand-computable.
///
/// ## Case setup
///
/// - 1 bus, 2 thermal plants (merit order), deterministic load 20 MW,
///   2 stages each with 730 hours, no hydro.
///
/// ## Expected cost derivation
///
/// - T0: capacity 15 MW at $5/MWh → dispatched at full capacity
/// - T1: capacity 15 MW at $10/MWh → dispatched at 5 MW to cover residual load
/// - Cost per stage = (15 × 5.0 + 5 × 10.0) × 730 = 125.0 × 730 = 91,250 $
/// - Total (2 stages) = 2 × 91,250 = **182,500 $**
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d01_thermal_dispatch() {
    let case_dir = Path::new("../../examples/deterministic/d01-thermal-dispatch");
    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, 182_500.0, 1e-6, "D01");
    assert!(
        result.iterations <= 10,
        "D01: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D01: gap={:.2e}",
        result.final_gap
    );
}

/// Two-stage hydrothermal dispatch. Optimal cost is hand-computed via LP.
///
/// ## Case setup
///
/// - 1 bus, 1 thermal (T0: 100 MW at $50/MWh), 1 hydro (H0: constant
///   productivity 1.0 MW/(m3/s), max 50 m3/s / 50 MW, storage 0–200 hm3)
/// - Deterministic inflows: 40.0 m3/s (stage 0) and 10.0 m3/s (stage 1)
/// - Deterministic load: 80.0 MW per stage
/// - Initial storage: 100.0 hm3
/// - 2 stages, 730 h each, no discounting
///
/// ## Expected cost
///
/// See [`D02_EXPECTED_COST`] for the full derivation. The optimal cost is
/// 23 635 000 / 9 ≈ 2 626 111.111... $, achieved by:
/// - Stage 0: turb₀ = 100/2.628 ≈ 38.05 m3/s, gen_th₀ ≈ 41.95 MW
/// - Stage 1: turb₁ = 50 m3/s (full capacity), gen_th₁ = 30 MW
/// - Storage at end of stage 0: exactly 40·2.628 = 105.12 hm3
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d02_single_hydro() {
    let case_dir = Path::new("../../examples/deterministic/d02-single-hydro");
    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, D02_EXPECTED_COST, 1e-4, "D02");
    assert!(
        result.iterations <= 10,
        "D02: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D02: gap={:.2e}",
        result.final_gap
    );
}

/// Behavioral: `modeling.cost_scale_factor` is objective conditioning only —
/// D02 trained at a non-default factor converges to the SAME cost as the
/// default-factor run (scale-invariance of the model — the real correctness
/// claim; the LP builder divides
/// by the resolved factor at template build and every reporting boundary
/// multiplies back, so the argmin — and its cost — does not depend on which
/// factor was configured).
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d02_single_hydro_cost_scale_invariant() {
    let case_dir = Path::new("../../examples/deterministic/d02-single-hydro");

    let default_result = run_deterministic(case_dir);
    assert_cost(
        default_result.final_lb,
        D02_EXPECTED_COST,
        1e-4,
        "D02-default-factor",
    );

    for non_default_factor in [10_000.0, 5_000_000.0] {
        let mut setup = fresh_setup_with(case_dir, |config| {
            config.modeling.cost_scale_factor = Some(non_default_factor);
        });
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must return Ok");
        assert!(
            outcome.error.is_none(),
            "cost_scale_factor={non_default_factor}: expected no training error"
        );
        assert_cost(
            outcome.result.final_lb,
            D02_EXPECTED_COST,
            1e-4,
            &format!("D02-factor-{non_default_factor}"),
        );
        assert!(
            (outcome.result.final_lb - default_result.final_lb).abs() < 1e-3,
            "cost_scale_factor={non_default_factor}: final_lb {} must match the \
             default-factor run's final_lb {} to behavioral tolerance",
            outcome.result.final_lb,
            default_result.final_lb
        );
    }
}

/// Three-stage cascade hydrothermal dispatch (2 hydros in series).
/// Combined capacity 70 MW < demand 75 MW. Cascade coupling: H1 receives H0's
/// discharge. Terminal stages: full capacity (thermal = 5 MW). Stage 0 binding
/// storage constraints yield thermal ≈ 28.09 MW.
///
/// Analytical thermal + deficit cost is `4_171_000 / 3 ≈ 1_390_333.33`. With
/// `turbined_cost` applied to every hydro's turbine column,
/// the deterministic LB adds a fixed regularization contribution that lifts
/// the total by `+1_364.4333…` (= turbined_cost × turbined MWh summed over
/// stages and blocks for both hydros).
pub const D03_EXPECTED_COST: f64 = 1_391_697.766_666_667_3;

#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d03_two_hydro_cascade() {
    let case_dir = Path::new("../../examples/deterministic/d03-two-hydro-cascade");
    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, D03_EXPECTED_COST, 1e-4, "D03");
    assert!(
        result.iterations <= 10,
        "D03: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D03: gap={:.2e}",
        result.final_gap
    );
}

/// Two-stage 2-bus transmission dispatch with line export limit.
/// B0 has excess hydro capacity and exports via 15 MW line to B1 (deficit region).
/// H0 covers B0 demand + export (thermal = 0). B1 has 5 MW unmet deficit/stage.
/// H1 depleted to minimize loss. Analytical thermal+deficit cost is
/// `5_263_443_883 / 657 ≈ 8_011_330.11 $`. With the universal `turbined_cost`
/// regularization applied to both hydros' turbine columns, the LB lifts to
/// `5_264_062_704 / 657 ≈ 8_012_272.00 $` (delta `≈ 941.89 $`).
pub const D04_EXPECTED_COST: f64 = 5_264_062_704.0 / 657.0;

#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d04_transmission() {
    let case_dir = Path::new("../../examples/deterministic/d04-transmission");
    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, D04_EXPECTED_COST, 1e-4, "D04");
    assert!(
        result.iterations <= 10,
        "D04: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D04: gap={:.2e}",
        result.final_gap
    );
}

/// Two-stage FPHA hydrothermal dispatch with single hyperplane (constant-productivity equivalent).
/// Identical to D02 except H0 uses FPHA model with one hyperplane per stage encoding
/// `gen = 1.0 × turbined_flow` (γ₀=0, γᵥ=0, γ_q=1.0, γ_s=0.0).
/// LP must match D02 exactly; cost tolerance 1e-6.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d05_fpha_constant_head() {
    let case_dir = Path::new("../../examples/deterministic/d05-fpha-constant-head");

    // ρ_eq = 1.0 is LP-neutral here: D05's FPHA hyperplane encodes
    // gen_h = 1.0 × turbined_m3s, matching the override scalar.
    let case = copy_case_dir(case_dir);
    write_energy_productivity_override(
        &case.path().join("system/hydro_energy_productivity.parquet"),
        0,
        1.0,
    );

    let result = run_deterministic(case.path());
    assert_cost(result.final_lb, D05_EXPECTED_COST, 1e-6, "D05");
    assert!(
        result.iterations <= 10,
        "D05: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D05: gap={:.2e}",
        result.final_gap
    );
}

/// Expected total cost for D06 (variable-head FPHA, 2 planes per stage).
///
/// ## Case setup
///
/// Same physical system as D02/D05: 1 bus, 1 thermal (T0: 100 MW at $50/MWh),
/// 1 hydro (H0: max 50 m3/s, storage 0–200 hm3), demand 80 MW, 2 stages × 730 h,
/// initial storage 100 hm3, deterministic inflows 40/10 m3/s.
///
/// H0 uses 2 precomputed FPHA hyperplanes (stage_id = null, valid for all stages):
/// - Plane 0: γ₀=0.0, γᵥ=0.002, γ_q=0.8, γ_s=0.0, κ_scale=1.0
/// - Plane 1: γ₀=0.0, γᵥ=0.001, γ_q=0.95, γ_s=0.0, κ_scale=1.0
///
/// ## FPHA constraint formulation in cobre-sddp
///
/// The LP builder implements the FPHA constraint using the **average** of
/// incoming and outgoing storage, not solely outgoing storage:
///
/// ```text
/// gen_h ≤ γ₀ + γᵥ/2·V_in + γᵥ/2·V_out + γ_q·q + γ_s·s
/// ```
///
/// This encodes the average forebay head over the stage interval.
/// V_in is pinned by the `storage_in` column bounds at the reference point
/// (previous iteration's trial value or the initial condition).
///
/// ## Analytical derivation (κ = 730·3600/10⁶ = 657/250 = 2.628 hm3 per m3/s)
///
/// ### Stage 1 (terminal, no future value)
///
/// With V_in1 = V_out0, inflow = 10 m3/s, and the turbine cap at 50 m3/s:
///
/// Setting q₁ = 50 m3/s (max) requires V_out0 ≥ 40·κ = 105.12 hm3 so that
/// V_out1 = V_out0 + (10 − 50)·κ ≥ 0.
///
/// At V_out0 = 105.12 hm3 = 40·κ, V_out1 = 0, V_in1 = 40·κ:
/// - Plane 0 bound: 0.002/2·(40·κ + 0) + 0.8·50 = 657/6250 + 40 = 250657/6250 ≈ 40.105 MW
/// - Plane 1 bound: 0.001/2·(40·κ + 0) + 0.95·50 ≈ 47.553 MW
/// - Binding plane: 0 → gen_h1 = 250657/6250 ≈ 40.105 MW
/// - gen_th1 = 80 − 250657/6250 = 249343/6250 ≈ 39.895 MW
/// - Stage 1 cost = (249343/6250) × 730 × 50 = 36404078/25 = 1,456,163.12 $
///
/// ### Stage 0
///
/// The SDDP backward pass places a Benders cut on V_out0. The shadow price
/// of the water-balance constraint drives the optimiser to raise storage
/// from 100 to exactly 40·κ = 105.12 hm3, the minimum needed for q₁ = 50.
///
/// At optimum: q₀ = 25000/657 ≈ 38.0518 m3/s, sp₀ = 0,
/// V_out0 = 2628/25 hm3, V_in0 = 100 hm3.
///
/// Plane 0 binds (with average-storage FPHA):
/// - gen_h0 = 0.002/2·(100 + 2628/25) + 0.8·(25000/657) = 62921137/2053125 ≈ 30.6465 MW
/// - gen_th0 = 80 − gen_h0 = 101328863/2053125 ≈ 49.3535 MW
/// - Stage 0 cost = (101328863/2053125) × 730 × 50 = 405315452/225 ≈ 1,801,402.01 $
///
/// ### Total cost
///
/// Total = 405315452/225 + 36404078/25
///       = 405315452/225 + 327636702/225
///       = **732952154/225 ≈ 3,257,565.1289 $**
///
/// This differs from D02_EXPECTED_COST (≈ 2,626,111.11 $) and D05_EXPECTED_COST
/// because the variable-head FPHA constraints reduce per-unit generation:
/// at q₀ ≈ 38.05 m3/s and mean storage ≈ (100+105.12)/2 hm3, plane 0 gives
/// only ≈ 30.65 MW, and in stage 1 the partial head (from V_in1=105.12, V_out1=0)
/// raises gen_h1 slightly above 40 MW, lowering thermal cost slightly vs D02.
pub const D06_EXPECTED_COST: f64 = 732_952_154.0 / 225.0;

/// Two-stage FPHA hydrothermal dispatch with 2 variable-head hyperplanes.
///
/// H0 uses 2 precomputed hyperplanes encoding storage-dependent generation
/// (γᵥ > 0). Plane 0 (γᵥ=0.002, γ_q=0.8) and plane 1 (γᵥ=0.001, γ_q=0.95)
/// together approximate the concave production function. Cost differs from D02/D05
/// because head variation reduces per-m3/s generation at typical storage levels.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d06_fpha_variable_head() {
    let case_dir = Path::new("../../examples/deterministic/d06-fpha-variable-head");

    // ρ_eq value is irrelevant to D06 economics — assertions depend only on
    // FPHA hyperplane evaluation, not on ρ_eq.
    let case = copy_case_dir(case_dir);
    write_energy_productivity_override(
        &case.path().join("system/hydro_energy_productivity.parquet"),
        0,
        1.0,
    );

    let result = run_deterministic(case.path());
    assert_cost(result.final_lb, D06_EXPECTED_COST, 1e-4, "D06");
    assert!(
        result.iterations <= 10,
        "D06: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D06: gap={:.2e}",
        result.final_gap
    );
    assert!(
        (result.final_lb - D02_EXPECTED_COST).abs() > 1.0,
        "D06: cost must differ from D02 (variable head changes economics)"
    );
}

/// Two-stage FPHA hydrothermal dispatch with computed hyperplanes from VHA geometry.
///
/// ## Case setup
///
/// Same physical system as D02/D05/D06: 1 bus, 1 thermal (T0: 100 MW at $50/MWh),
/// 1 hydro (H0: max 50 m3/s, storage 0–200 hm3), demand 80 MW, 2 stages × 730 h,
/// initial storage 100 hm3, deterministic inflows 40/10 m3/s.
///
/// H0 uses `"source": "computed"` in `hydro_production_models.json`. The fitting
/// pipeline reads the VHA curve from `system/hydro_geometry.parquet`, evaluates
/// the production function φ(V, q) using the tailrace, hydraulic losses, and
/// efficiency fields in `hydros.json`, and fits FPHA hyperplanes automatically.
///
/// ## Geometry
///
/// VHA curve: 5 breakpoints over 0–200 hm3. Forebay heights 350–400 m (well above
/// the constant tailrace at 300 m), giving a net head of ~50–100 m. Factor losses
/// of 3% and constant efficiency of 92% are applied.
///
/// The asserted cost is the converged optimum (LB == UB), distinct from D06's
/// value because the computed fitting uses a different discretization grid and
/// plane count than D06's precomputed planes. Backend-agnostic to tolerance.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d07_fpha_computed() {
    let case_dir = Path::new("../../examples/deterministic/d07-fpha-computed");
    let result = run_deterministic(case_dir);
    assert!(
        result.final_gap.abs() < 1e-6,
        "D07: gap={:.2e}",
        result.final_gap
    );
    assert!(
        result.iterations <= 10,
        "D07: iterations={}",
        result.iterations
    );
    assert!(
        result.final_lb > 0.0,
        "D07: final_lb={} must be positive",
        result.final_lb
    );
    assert_cost(result.final_lb, 3_657_537.696_594_757, 1.0, "D07");
}

// ---------------------------------------------------------------------------
// D-case energy-output correctness sweep (d02, d03)
// ---------------------------------------------------------------------------

/// Conversion factor from hm³·MW/(m³/s) to MWh (= 10⁶ / 3600).
///
/// `stored_energy_mwh = (volume_hm3 − V_min) × ρ_acum_integrated × ENERGY_FACTOR`
const ENERGY_FACTOR: f64 = 1.0e6 / 3600.0;

/// Expected `ρ_eq` and `ρ_acum` for D02 (single hydro, `constant_productivity = 1.0`,
/// no downstream).
///
/// `ρ_eq = 1.0` MW/(m³/s) — read directly from `hydros.json`.
/// `ρ_acum = 1.0` — H0 has no downstream, so accumulated = own `ρ_eq`.
const D02_RHO_EQ: f64 = 1.0;
const D02_RHO_ACUM: f64 = 1.0;

/// Stage-0 initial storage for D02 H0 \[hm³\] from `initial_conditions.json`.
const D02_H0_V_INIT: f64 = 100.0;

/// Deterministic stage-0 inflow for D02 H0 \[m³/s\].
///
/// From the D02 scenario parquet (std = 0): mean = 40.0 m³/s.
const D02_H0_STAGE0_INFLOW: f64 = 40.0;

/// Deterministic stage-1 inflow for D02 H0 \[m³/s\].
///
/// From the D02 scenario parquet (std = 0): mean = 10.0 m³/s.
const D02_H0_STAGE1_INFLOW: f64 = 10.0;

/// Expected `ρ_eq` and `ρ_acum` for D03 H0.
///
/// `ρ_eq(H0) = 1.0` — from `hydros.json`, `constant_productivity`.
/// `ρ_acum(H0) = 2.0` — H0 is upstream of H1; accumulated = ρ_eq(H0) + ρ_eq(H1).
const D03_H0_RHO_EQ: f64 = 1.0;
const D03_H0_RHO_ACUM: f64 = 2.0;

/// Expected `ρ_eq` and `ρ_acum` for D03 H1.
///
/// `ρ_eq(H1) = 1.0` — from `hydros.json`, `constant_productivity`.
/// `ρ_acum(H1) = 1.0` — H1 has no downstream.
const D03_H1_RHO_EQ: f64 = 1.0;
const D03_H1_RHO_ACUM: f64 = 1.0;

/// Stage-0 initial storage for D03 H0 \[hm³\] from `initial_conditions.json`.
const D03_H0_V_INIT: f64 = 80.0;

/// Stage-0 initial storage for D03 H1 \[hm³\] from `initial_conditions.json`.
const D03_H1_V_INIT: f64 = 50.0;

/// Verify the natural-inflow-energy and stored-energy columns in
/// `simulation/hydros` for D02 and D03. Both use `ConstantProductivity` hydros
/// (bypassing the FPHA gate), so `ρ_eq` and `ρ_acum` are directly computable
/// from `hydros.json` without running the LP. Neither deck declares VHA geometry,
/// so the integrated cascade grid `stored_energy_*_mwh` reads is bit-identical to
/// `ρ_acum` and the `*_RHO_ACUM` constants serve both.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d_case_energy_outputs() {
    const TOL: f64 = 1e-6;
    const V_MIN: f64 = 0.0; // both cases have min_storage_hm3 = 0

    // ── D02: single hydro, 2 stages ──────────────────────────────────────────
    {
        let case_dir = Path::new("../../examples/deterministic/d02-single-hydro");
        let (_result, scenario_results, _summary) = run_with_simulation(case_dir);

        assert_eq!(scenario_results.len(), 1, "D02: expected 1 scenario");
        let scenario = &scenario_results[0];
        assert_eq!(scenario.stages.len(), 2, "D02: expected 2 stages");

        for stage_result in &scenario.stages {
            let stage = stage_result.stage_id as usize;
            assert_eq!(
                stage_result.hydros.len(),
                1,
                "D02 stage {stage}: expected 1 hydro result"
            );
            let h = &stage_result.hydros[0];

            let diff_rho_eq = (h.equivalent_productivity_mw_per_m3s - D02_RHO_EQ).abs();
            assert!(
                diff_rho_eq <= TOL,
                "D02 stage {stage} H0: equivalent_productivity_mw_per_m3s = {} (expected {D02_RHO_EQ}, diff = {diff_rho_eq})",
                h.equivalent_productivity_mw_per_m3s,
            );

            let diff_rho_acum = (h.accumulated_productivity_mw_per_m3s - D02_RHO_ACUM).abs();
            assert!(
                diff_rho_acum <= TOL,
                "D02 stage {stage} H0: accumulated_productivity_mw_per_m3s = {} (expected {D02_RHO_ACUM}, diff = {diff_rho_acum})",
                h.accumulated_productivity_mw_per_m3s,
            );

            let expected_energy = h.accumulated_productivity_mw_per_m3s * h.incremental_inflow_m3s;
            let diff_energy = (h.incremental_inflow_energy_mw - expected_energy).abs();
            assert!(
                diff_energy <= TOL,
                "D02 stage {stage} H0: incremental_inflow_energy_mw = {} (ρ_acum × inflow = {expected_energy}, diff = {diff_energy})",
                h.incremental_inflow_energy_mw,
            );

            let expected_earm = (h.storage_initial_hm3 - V_MIN)
                * h.integrated_accumulated_productivity_mw_per_m3s
                * ENERGY_FACTOR;
            let diff_earm = (h.stored_energy_initial_mwh - expected_earm).abs();
            assert!(
                diff_earm <= TOL,
                "D02 stage {stage} H0: stored_energy_initial_mwh = {} (expected {expected_earm}, diff = {diff_earm})",
                h.stored_energy_initial_mwh,
            );
        }

        // Stage 0 against hand-fixed constants (deterministic from inputs),
        // not just field-consistency.
        let h0_stage0 = scenario.stages[0]
            .hydros
            .iter()
            .find(|h| h.hydro_id == 0)
            .expect("D02: H0 missing from stage 0");

        let diff_inflow0 =
            (h0_stage0.incremental_inflow_energy_mw - D02_RHO_ACUM * D02_H0_STAGE0_INFLOW).abs();
        assert!(
            diff_inflow0 <= TOL,
            "D02 stage 0 H0: incremental_inflow_energy_mw = {} (expected {}, diff = {diff_inflow0})",
            h0_stage0.incremental_inflow_energy_mw,
            D02_RHO_ACUM * D02_H0_STAGE0_INFLOW,
        );

        let expected_earm0 = (D02_H0_V_INIT - V_MIN) * D02_RHO_ACUM * ENERGY_FACTOR;
        let diff_earm0 = (h0_stage0.stored_energy_initial_mwh - expected_earm0).abs();
        assert!(
            diff_earm0 <= TOL,
            "D02 stage 0 H0: stored_energy_initial_mwh = {} (expected {expected_earm0}, diff = {diff_earm0})",
            h0_stage0.stored_energy_initial_mwh,
        );

        let h0_stage1 = scenario.stages[1]
            .hydros
            .iter()
            .find(|h| h.hydro_id == 0)
            .expect("D02: H0 missing from stage 1");

        let diff_inflow1 =
            (h0_stage1.incremental_inflow_energy_mw - D02_RHO_ACUM * D02_H0_STAGE1_INFLOW).abs();
        assert!(
            diff_inflow1 <= TOL,
            "D02 stage 1 H0: incremental_inflow_energy_mw = {} (expected {}, diff = {diff_inflow1})",
            h0_stage1.incremental_inflow_energy_mw,
            D02_RHO_ACUM * D02_H0_STAGE1_INFLOW,
        );
    }

    // ── D03: two-hydro cascade, 3 stages ─────────────────────────────────────
    {
        let case_dir = Path::new("../../examples/deterministic/d03-two-hydro-cascade");
        let (_result, scenario_results, _summary) = run_with_simulation(case_dir);

        assert_eq!(scenario_results.len(), 1, "D03: expected 1 scenario");
        let scenario = &scenario_results[0];
        assert_eq!(scenario.stages.len(), 3, "D03: expected 3 stages");

        for stage_result in &scenario.stages {
            let stage = stage_result.stage_id as usize;
            assert_eq!(
                stage_result.hydros.len(),
                2,
                "D03 stage {stage}: expected 2 hydro results"
            );

            for h in &stage_result.hydros {
                let (expected_rho_eq, expected_rho_acum) = if h.hydro_id == 0 {
                    (D03_H0_RHO_EQ, D03_H0_RHO_ACUM)
                } else {
                    (D03_H1_RHO_EQ, D03_H1_RHO_ACUM)
                };

                let diff_rho_eq = (h.equivalent_productivity_mw_per_m3s - expected_rho_eq).abs();
                assert!(
                    diff_rho_eq <= TOL,
                    "D03 stage {stage} H{}: equivalent_productivity_mw_per_m3s = {} (expected {expected_rho_eq}, diff = {diff_rho_eq})",
                    h.hydro_id,
                    h.equivalent_productivity_mw_per_m3s,
                );

                let diff_rho_acum =
                    (h.accumulated_productivity_mw_per_m3s - expected_rho_acum).abs();
                assert!(
                    diff_rho_acum <= TOL,
                    "D03 stage {stage} H{}: accumulated_productivity_mw_per_m3s = {} (expected {expected_rho_acum}, diff = {diff_rho_acum})",
                    h.hydro_id,
                    h.accumulated_productivity_mw_per_m3s,
                );

                let expected_energy =
                    h.accumulated_productivity_mw_per_m3s * h.incremental_inflow_m3s;
                let diff_energy = (h.incremental_inflow_energy_mw - expected_energy).abs();
                assert!(
                    diff_energy <= TOL,
                    "D03 stage {stage} H{}: incremental_inflow_energy_mw = {} (ρ_acum × inflow = {expected_energy}, diff = {diff_energy})",
                    h.hydro_id,
                    h.incremental_inflow_energy_mw,
                );

                let expected_earm = (h.storage_initial_hm3 - V_MIN)
                    * h.integrated_accumulated_productivity_mw_per_m3s
                    * ENERGY_FACTOR;
                let diff_earm = (h.stored_energy_initial_mwh - expected_earm).abs();
                assert!(
                    diff_earm <= TOL,
                    "D03 stage {stage} H{}: stored_energy_initial_mwh = {} (expected {expected_earm}, diff = {diff_earm})",
                    h.hydro_id,
                    h.stored_energy_initial_mwh,
                );
            }
        }

        // Stage 0 against hand-fixed constants (deterministic from initial conditions).
        let h0_s0 = scenario.stages[0]
            .hydros
            .iter()
            .find(|h| h.hydro_id == 0)
            .expect("D03: H0 missing from stage 0");
        let h1_s0 = scenario.stages[0]
            .hydros
            .iter()
            .find(|h| h.hydro_id == 1)
            .expect("D03: H1 missing from stage 0");

        let expected_earm_h0 = (D03_H0_V_INIT - V_MIN) * D03_H0_RHO_ACUM * ENERGY_FACTOR;
        let diff_earm_h0 = (h0_s0.stored_energy_initial_mwh - expected_earm_h0).abs();
        assert!(
            diff_earm_h0 <= TOL,
            "D03 stage 0 H0: stored_energy_initial_mwh = {} (expected {expected_earm_h0}, diff = {diff_earm_h0})",
            h0_s0.stored_energy_initial_mwh,
        );

        let expected_earm_h1 = (D03_H1_V_INIT - V_MIN) * D03_H1_RHO_ACUM * ENERGY_FACTOR;
        let diff_earm_h1 = (h1_s0.stored_energy_initial_mwh - expected_earm_h1).abs();
        assert!(
            diff_earm_h1 <= TOL,
            "D03 stage 0 H1: stored_energy_initial_mwh = {} (expected {expected_earm_h1}, diff = {diff_earm_h1})",
            h1_s0.stored_energy_initial_mwh,
        );
    }
}

/// Expected total cost for D08 (single hydro with linearized evaporation, 2 stages).
///
/// ## Case setup
///
/// Same physical system as D02: 1 bus, 1 thermal (T0: 100 MW at $50/MWh),
/// 1 hydro (H0: constant productivity 1.0 MW/(m3/s), max 50 m3/s / 50 MW,
/// storage 0–200 hm3), demand 80 MW, 2 stages × 730 h, initial storage
/// 100 hm3, deterministic inflows 40/10 m3/s.
///
/// H0 has evaporation enabled: `coefficients_mm = [100.0; 12]` (uniform
/// across all 12 calendar months). No `reference_volumes_hm3` — default
/// midpoint (100 hm3) is used.
///
/// ## Geometry (hydro_geometry.parquet)
///
/// Linear VHA: (0 hm3 → 0.5 km²), (100 hm3 → 1.0 km²), (200 hm3 → 1.5 km²).
/// Uniform slope da/dv = 0.005 km²/hm3.
///
/// ## Evaporation coefficient derivation (calendar-month rate)
///
/// Midpoint reference_volume = (0 + 200) / 2 = 100 hm3; a_ref = 1.0 km²,
/// da/dv = 0.005 km²/hm3; monthly_evaporation_mm = 100.
///
/// The flow is a monthly-average rate: `mm_km2_to_m3s = 1 / (3.6 × month_hours)`,
/// dividing by the stage's CALENDAR month, not its nominal 730 h. The two stages
/// therefore carry DIFFERENT coefficients — stage 0 is January (744 h →
/// `1/2678.4`) and stage 1 is February 2024 (696 h leap → `1/2505.6`) — and each
/// deposits only its `stage_hours / month_hours` share of the month's
/// evaporation, the property this case pins. (A `stage_hours` divisor would
/// cancel against the water-balance coupling and deposit a whole month on any
/// stage; see `resolve_evaporation_models`.)
///
/// ## Expected cost
///
/// The model and optimal policy are D02's (stage-0 `turb₀ = 50 m3/s`, full
/// capacity), plus evaporation drawing the reservoir down. The study converges
/// (`LB == UB`, gap < 1e-6) to the value below — above D02's ≈ 2,626,111.11 $
/// because evaporation consumes water that would otherwise generate, forcing more
/// thermal dispatch. The converged optimum, not a closed-form rational: the
/// calendar-month divisors (744, 696) do not reduce cleanly the way the retired
/// nominal-730 h derivation did.
pub const D08_EXPECTED_COST: f64 = 2_628_380.531_119_177;

#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d08_evaporation() {
    let case_dir = Path::new("../../examples/deterministic/d08-evaporation");
    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, D08_EXPECTED_COST, 1e-4, "D08");
    assert!(
        result.iterations <= 10,
        "D08: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D08: gap={:.2e}",
        result.final_gap
    );
    assert!(
        result.final_lb > D02_EXPECTED_COST,
        "D08: cost {:.6} must exceed D02 cost {:.6}",
        result.final_lb,
        D02_EXPECTED_COST
    );
}

/// Expected total cost for D09 (multi-segment deficit, 2 stages).
///
/// ## Case setup
///
/// - 1 bus (B0) with 2 deficit segments:
///   - Segment 0: depth_mw=10.0, cost=$500/MWh (first 10 MW of deficit)
///   - Segment 1: depth_mw=null (unlimited), cost=$5000/MWh (remaining deficit)
/// - 1 thermal (T0): capacity 30 MW at $10/MWh
/// - Deterministic load: 50 MW per stage
/// - 2 stages, 730 hours each, no hydro
///
/// ## Expected cost derivation
///
/// With supply = 30 MW (thermal at full capacity) and demand = 50 MW:
///   deficit = 20 MW per stage
///   - First 10 MW of deficit → segment 0: 10 × $500/MWh
///   - Next 10 MW of deficit → segment 1: 10 × $5000/MWh
///
/// Cost per stage = (30 × 10 + 10 × 500 + 10 × 5000) × 730
///                = (300 + 5000 + 50000) × 730
///                = 55300 × 730
///                = 40,369,000 $
/// Total (2 stages) = 2 × 40,369,000 = **80,738,000 $**
pub const D09_EXPECTED_COST: f64 = 80_738_000.0;

/// Two-stage pure thermal dispatch with 2-segment tiered deficit pricing.
///
/// ## Case setup
///
/// - 1 bus with 2 deficit segments: [10 MW @ $500/MWh, unlimited @ $5000/MWh]
/// - 1 thermal (T0): 30 MW at $10/MWh, deterministic load 50 MW
/// - 2 stages × 730 h, no hydro
///
/// ## Expected cost
///
/// See [`D09_EXPECTED_COST`] for the derivation. With 20 MW deficit per stage
/// split across both segments, total cost is 80,738,000 $.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d09_multi_deficit() {
    let case_dir = Path::new("../../examples/deterministic/d09-multi-deficit");
    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, D09_EXPECTED_COST, 1e-6, "D09");
    assert!(
        result.iterations <= 10,
        "D09: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D09: gap={:.2e}",
        result.final_gap
    );
}

/// Expected total cost for D10 (inflow non-negativity penalty, 2 stages).
///
/// ## Case setup
///
/// Same physical system as D02: 1 bus (B0), 1 thermal (T0: 100 MW at $50/MWh),
/// 1 hydro (H0: constant productivity 1.0 MW/(m3/s), max 50 m3/s, storage
/// 0–200 hm3), demand 80 MW, 2 stages × 730 h, initial storage 100 hm3.
///
/// Inflows: stage 0 = 40 m3/s (positive), stage 1 = -5 m3/s (negative).
/// Config: `inflow_non_negativity: {method: "penalty", penalty_cost: 500.0}`.
///
/// ## Penalty cost unit (verified from `build_stage_templates`)
///
/// From `build_stage_templates_resolving_layout`, which delegates to it:
/// ```text
/// let obj_coeff = penalty_cost * total_stage_hours;
/// objective[col] = obj_coeff;
/// ```
/// The inflow slack column `sigma_inf_h` has objective coefficient
/// `penalty_cost × total_stage_hours = 500 × 730 = 365,000 $/m3/s`.
///
/// The LP column for sigma has lower bound 0 (not max(0, -inflow)).  The LP
/// freely chooses sigma = 0 when it is cheaper to reduce turbining instead.
///
/// ## Cost derivation (κ = 730·3600/10⁶ = 657/250 hm³/(m³/s))
///
/// ### Stage 1 optimal sigma
///
/// Stage 1 water balance:
///   V_out1 = V_in1 + (sigma − 5 − q1) × κ ≥ 0.
///
/// KKT: increasing sigma by 1 m3/s costs 365,000 $ but adds κ hm3 of water
/// (which at most saves 50 × 730/κ = 36,500 $/m3/s via turbining).
/// Since 365,000 >> 36,500, the optimizer always sets sigma = 0.
///
/// ### Stage 1 dispatch (sigma = 0, effective inflow = −5 m3/s)
///
/// With sigma = 0 the balance is:
///   V_out1 = V_in1 + (−5 − q1) × κ ≥ 0  →  q1 ≤ V_in1/κ − 5.
///
/// Breakpoints:
///   If V_in1 ≥ 55·κ (= 144.54 hm3): q1 = 50, gen_th1 = 30, cost1 = 1,095,000.
///   If 5·κ ≤ V_in1 < 55·κ:           q1 = V_in1/κ − 5.
///
/// Since q0 ≤ 50 and inflow0 = 40: V_out0 ≥ 100 − 10·κ ≈ 73.7 hm3 > 5·κ = 13.1 hm3.
/// So sigma = 0 is always feasible (no infeasibility risk for any q0 ≤ 50).
///
/// ### Stage 0 optimal policy
///
/// Water balance: V_out0 = 100 + (40 − q0) × κ.
///
/// Case A (q0 ≤ 15145/657, V_out0 ≥ 55·κ):
///   Total thermal = 36500 × (110 − q0). Decreases with q0; optimal at boundary.
///
/// Case B (q0 > 15145/657, V_out0 < 55·κ):
///   q1 = V_out0/κ − 5 = 35 + 25000/657 − q0.
///   Total thermal = 36500 × [(80 − q0) + (45 − 25000/657 + q0)]
///                 = 36500 × (125 − 25000/657) — constant in q0.
///
/// At the boundary (Case A → B): total thermal is continuous.
/// The optimizer is indifferent in Case B and the SDDP converges to the
/// same constant total thermal cost for any q0 in that range.
///
/// ### Total cost (no penalty)
///
/// Total = 36500 × (125 − 25000/657)
///       = 36500 × (82125 − 25000)/657
///       = 36500 × 57125/657
///       = **2,085,062,500/657 = 28,562,500/9 ≈ 3,173,611.11 $**
///
/// D10 cost > D02 cost (≈ 2,626,111.11 $): the negative inflow in stage 1
/// effectively reduces available water (turbining is limited to V_in1/κ − 5
/// instead of V_in1/κ + 10), requiring more thermal dispatch in stage 1.
pub const D10_EXPECTED_COST: f64 = 28_562_500.0 / 9.0;

/// Two-stage hydrothermal dispatch with inflow non-negativity penalty.
///
/// ## Case setup
///
/// - 1 bus (B0), 1 thermal (T0: 100 MW at $50/MWh), 1 hydro (H0: constant
///   productivity 1.0 MW/(m3/s), max 50 m3/s, storage 0–200 hm3)
/// - Deterministic inflows: 40.0 m3/s (stage 0), -5.0 m3/s (stage 1)
/// - Deterministic load: 80.0 MW per stage
/// - Initial storage: 100.0 hm3
/// - 2 stages × 730 h, `inflow_non_negativity: {method: "penalty", penalty_cost: 500.0}`
///
/// ## Expected cost
///
/// See [`D10_EXPECTED_COST`] for the full derivation.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d10_inflow_nonnegativity() {
    let case_dir = Path::new("../../examples/deterministic/d10-inflow-nonnegativity");
    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, D10_EXPECTED_COST, 1e-4, "D10");
    assert!(
        result.iterations <= 10,
        "D10: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D10: gap={:.2e}",
        result.final_gap
    );
    assert!(
        result.final_lb > D02_EXPECTED_COST,
        "D10: cost {:.6} must exceed D02 cost {:.6}",
        result.final_lb,
        D02_EXPECTED_COST
    );
}

/// Expected total cost for D11 (single hydro with water withdrawal, 2 stages).
///
/// ## Case setup
///
/// - 1 bus (B0), 1 thermal (T0: 100 MW at $100/MWh), 1 hydro (H0: constant
///   productivity 1.0 MW/(m3/s), max 50 m3/s / 50 MW, storage 0–200 hm3)
/// - Deterministic inflows: 30.0 m3/s per stage
/// - Water withdrawal: 10 m3/s per stage (via `constraints/hydro_bounds.parquet`)
/// - Deterministic load: 80.0 MW per stage
/// - Initial storage: 100.0 hm3
/// - 2 stages × 730 h, `inflow_non_negativity: {method: "none"}`
///
/// ## How withdrawal enters the water balance
///
/// The LP water balance for each stage is:
///   V_out = V_in + κ × (inflow − withdrawal − turbine − spill)
///         = V_in + κ × (30 − 10 − turbine − spill)
///         = V_in + κ × (20 − turbine − spill)
///
/// This is equivalent to a case with 20 m3/s net inflow and no withdrawal.
///
/// ## Expected cost derivation (κ = 730 × 3600 / 10⁶ = 657/250 hm³/(m³/s))
///
/// Let q0 = turbined flow in stage 0.
///
/// Stage 1 water balance: V_out1 = V_in1 + κ × (20 − q1) ≥ 0
///   → q1 ≤ V_in1/κ + 20
///
/// Stage 0 storage: V_out0 = 100 + κ × (20 − q0)
///
/// ### Case A: q0 ≤ 18430/657 (so V_out0 ≥ 30κ, stage 1 can run at full capacity)
///
/// When V_out0 ≥ 30κ, q1 = 50 and gen_th1 = 30 MW.
/// Total thermal = (80 − q0) + 30 = 110 − q0, which decreases with q0.
/// Optimal at q0 = 18430/657 ≈ 28.054 m3/s (boundary).
///
/// ### Case B: q0 > 18430/657 (V_out0 < 30κ, stage 1 is storage-limited)
///
/// q1 = V_out0/κ + 20 = (100 + κ×(20−q0))/κ + 20 = 100/κ + 40 − q0
/// gen_th1 = 80 − q1 = 40 − 100/κ + q0
///
/// Total thermal = (80 − q0) + (40 − 100/κ + q0) = 120 − 100/κ
///   = 120 − 25000/657 = 53840/657 MW (constant — independent of q0)
///
/// The objective is constant in Case B, so SDDP converges to:
///   Total cost = (53840/657) × 100 × 730
///              = 53840 × 73000 / 657
///              = **3,930,320,000 / 657 ≈ 5,982,222.22 $**
///
/// D11 cost > D02 cost (≈ 2,626,111.11 $) because the withdrawal reduces net
/// inflow from 30 to 20 m3/s, leaving less water for generation across both
/// stages and requiring significantly more thermal dispatch.
///
/// The pinned value adds the universal `turbined_cost` regularization
/// (`≈ 569.78 $`) to the analytical thermal+deficit cost
/// `3_930_320_000 / 657 ≈ 5_982_222.22 $`, yielding `3_930_694_344 / 657`.
pub const D11_WATER_WITHDRAWAL_EXPECTED_COST: f64 = 3_930_694_344.0 / 657.0;

/// Two-stage hydrothermal dispatch with water withdrawal applied via hydro bounds.
///
/// ## Case setup
///
/// - 1 bus (B0), 1 thermal (T0: 100 MW at $100/MWh), 1 hydro (H0: constant
///   productivity 1.0 MW/(m3/s), max 50 m3/s / 50 MW, storage 0–200 hm3)
/// - Deterministic inflows: 30.0 m3/s per stage
/// - Water withdrawal: 10 m3/s per stage (from `constraints/hydro_bounds.parquet`)
/// - Deterministic load: 80.0 MW per stage
/// - Initial storage: 100.0 hm3
/// - 2 stages × 730 h
///
/// ## Expected cost
///
/// See [`D11_WATER_WITHDRAWAL_EXPECTED_COST`] for the full derivation. The 10 m3/s
/// withdrawal reduces effective net inflow from 30 to 20 m3/s, increasing thermal dispatch.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d11_water_withdrawal() {
    let case_dir = Path::new("../../examples/deterministic/d11-water-withdrawal");
    let result = run_deterministic(case_dir);
    assert_cost(
        result.final_lb,
        D11_WATER_WITHDRAWAL_EXPECTED_COST,
        1e-4,
        "D11",
    );
    assert!(
        result.iterations <= 10,
        "D11: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D11: gap={:.2e}",
        result.final_gap
    );
    assert!(
        result.final_lb > D02_EXPECTED_COST,
        "D11: cost {:.6} must exceed D02 cost {:.6} (withdrawal increases thermal dispatch)",
        result.final_lb,
        D02_EXPECTED_COST
    );
}

/// Warm-start verification for the D02 system: after training,
/// `SolverStatistics.basis_consistency_failures` must be zero. A non-zero count
/// means silent cold-start fallbacks that degrade performance without an error.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d11_warm_start_verification() {
    let case_dir = Path::new("../../examples/deterministic/d02-single-hydro");
    let (result, solver) = run_deterministic_with_solver(case_dir);

    assert_cost(result.final_lb, D02_EXPECTED_COST, 1e-4, "D11");
    assert!(
        result.final_gap.abs() < 1e-6,
        "D11: gap={:.2e}",
        result.final_gap
    );

    let stats = solver.statistics();
    assert_eq!(
        stats.basis_consistency_failures, 0,
        "D11: expected 0 basis rejections, got {}",
        stats.basis_consistency_failures
    );
}

/// Checkpoint round-trip for the D02 system: exercises the full `FlatBuffers`
/// persistence pipeline (train → write → read → simulate from the loaded FCF).
///
/// ## Why the simulation cost should equal the training LB
///
/// D02 has deterministic (zero-variance) inflows and load. With
/// `num_scenarios = 1`, the single simulation scenario uses the same inflow
/// realization as the training forward pass, and the converged FCF is the true
/// value function — so the simulation cost equals the training LB.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d12_checkpoint_round_trip() {
    let case_dir = Path::new("../../examples/deterministic/d02-single-hydro");

    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let pr = prepare_stochastic(
        system,
        case_dir,
        &config,
        42,
        &ScenarioSource::default(),
        None,
    )
    .expect("prepare_stochastic must succeed");
    let system = pr.system;
    let stochastic = pr.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");

    let mut config_with_sim = config.clone();
    config_with_sim.simulation.enabled = true;
    config_with_sim.simulation.selection = Some(SimulationSelection::Sampled { num_scenarios: 1 });

    let mut setup = StudySetup::new(
        &system,
        &config_with_sim,
        stochastic,
        hydro_models,
        Vec::new(),
    )
    .expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");
    let result = outcome.result;

    assert_cost(result.final_lb, D02_EXPECTED_COST, 1e-4, "D12-train");
    assert!(
        result.final_gap.abs() < 1e-6,
        "D12: gap={:.2e}",
        result.final_gap
    );

    // Required side effect before StageCutsPayload extraction below.
    let _training_output = setup.build_training_output(&result, &[]);

    let tmp = tempfile::tempdir().expect("tempdir must succeed");
    let policy_dir = tmp.path().join("policy");

    let fcf = &setup.fcf;

    let cut_records_per_stage: Vec<Vec<PolicyCutRecord<'_>>> = fcf
        .pools
        .iter()
        .map(|pool| {
            (0..pool.populated())
                .map(|slot| {
                    let meta = pool.metadata(slot);
                    PolicyCutRecord {
                        cut_id: slot as u64,
                        slot_index: slot as u32,
                        iteration: meta.iteration_generated as u32,
                        forward_pass_index: meta.forward_pass_index,
                        intercept: pool.intercept(slot),
                        coefficients: pool.coefficient_row(slot),
                        is_active: pool.is_active(slot),
                    }
                })
                .collect()
        })
        .collect();

    let active_indices_per_stage: Vec<Vec<u32>> = fcf
        .pools
        .iter()
        .map(|pool| {
            (0..pool.populated())
                .filter(|&slot| pool.is_active(slot))
                .map(|slot| slot as u32)
                .collect()
        })
        .collect();

    let stage_cuts_payloads: Vec<StageCutsPayload<'_>> = fcf
        .pools
        .iter()
        .enumerate()
        .map(|(stage_idx, pool)| StageCutsPayload {
            stage_id: stage_idx as u32,
            state_dimension: pool.state_dimension as u32,
            capacity: pool.capacity as u32,
            warm_start_count: pool.warm_start_count,
            cuts: &cut_records_per_stage[stage_idx],
            active_cut_indices: &active_indices_per_stage[stage_idx],
            populated_count: pool.populated() as u32,
            entity_manifest: &[],
            cost_scale_factor: 1_000_000.0,
            node_id: i32::try_from(stage_idx).unwrap_or(-1),
            graph_stage_id: -1,
            priced_state_date: STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
        })
        .collect();

    let n_stages = fcf.pools.len();
    let warm_start_counts: Vec<u32> = fcf.pools.iter().map(|p| p.warm_start_count).collect();
    let policy_metadata = cobre_sddp::test_support::checkpoint_metadata(
        n_stages as u32,
        cobre_io::GraphManifest::default(),
        cobre_io::ProducerBlock {
            completed_iterations: result.iterations as u32,
            final_lower_bound: result.final_lb,
            best_upper_bound: Some(result.final_ub),
            max_iterations: 100,
            forward_passes: 1,
            warm_start_cuts: warm_start_counts.iter().copied().max().unwrap_or(0),
            warm_start_counts,
            rng_seed: 42,
            total_visited_states: 0,
            training_block_mode: "parallel".to_string(),
            training_block_mode_per_stage: vec![],
            cost_scale_factor: None,
        },
    );

    write_policy_checkpoint(
        &policy_dir,
        &stage_cuts_payloads,
        &[],
        &policy_metadata,
        &[],
    )
    .expect("write_policy_checkpoint must succeed");

    let checkpoint =
        cobre_io::read_policy_checkpoint(&policy_dir).expect("read_policy_checkpoint must succeed");

    assert_eq!(
        checkpoint.metadata.num_stages, 2,
        "D12: checkpoint must have 2 stages"
    );
    assert_eq!(
        checkpoint.stage_cuts[0].state_dimension, 1,
        "D12: checkpoint must have per-pool state_dimension == 1 (one hydro = one storage state)"
    );
    assert!(
        !checkpoint.stage_cuts.is_empty(),
        "D12: checkpoint must contain at least one stage_cuts entry"
    );

    let manifest_path = policy_dir.join("manifest.bin");
    assert!(manifest_path.is_file(), "D12: manifest.bin must exist");

    let stage_bin_path = policy_dir.join("cuts/000.bin");
    assert!(stage_bin_path.is_file(), "D12: cuts/000.bin must exist");

    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
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
    let _scenario_results = drain_handle.join().expect("drain thread must not panic");

    let sim_config = setup.simulation_config();
    let (summary, _gathered) = aggregate_simulation(
        &local_costs.costs,
        sim_config,
        &comm,
        SimulationWeighting::Uniform,
    )
    .expect("aggregate_simulation must succeed");

    assert_eq!(
        summary.n_scenarios, 1,
        "D12: simulation must produce exactly 1 scenario"
    );

    assert_cost(summary.mean_cost, D02_EXPECTED_COST, 1e-2, "D12-sim");
}

/// Generic constraint capping thermal dispatch, forcing deficit.
///
/// ## Case setup
///
/// - 1 bus, 1 thermal T0: capacity 30 MW at $50/MWh, deterministic load 20 MW,
///   2 stages each with 730 hours, no hydro.
/// - 1 generic constraint: `thermal_generation(0) <= 10 MW`
///   with slack penalty $5000/MWh (slack is more expensive than deficit at $1000/MWh).
/// - Deficit cost: $1000/MWh (from buses.json).
///
/// ## Expected cost derivation
///
/// The optimizer will dispatch T0 = 10 MW (at the constraint cap) and leave
/// 10 MW of deficit, since deficit ($1000/MWh) is cheaper than violating
/// the generic constraint ($5000/MWh).
///
/// Cost per stage:
///   thermal: 10 MW × $50/MWh × 730 h = $365,000
///   deficit: 10 MW × $1000/MWh × 730 h = $7,300,000
///   total:   $7,665,000
///
/// Total (2 stages) = 2 × $7,665,000 = **$15,330,000**
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d13_generic_constraint() {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let case_dir = Path::new("../../examples/deterministic/d13-generic-constraint");

    let constraints_dir = case_dir.join("constraints");
    std::fs::create_dir_all(&constraints_dir).expect("create constraints dir");

    let schema = Arc::new(Schema::new(vec![
        Field::new("constraint_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("block_id", DataType::Int32, true),
        Field::new("bound_lower", DataType::Float64, true),
        Field::new("bound_upper", DataType::Float64, true),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1])),
            Arc::new(Int32Array::from(vec![0, 1])),
            Arc::new(Int32Array::new_null(2)), // block_id null = all blocks
            Arc::new(Float64Array::new_null(2)), // upper-only: no bound_lower
            Arc::new(Float64Array::from(vec![10.0, 10.0])),
        ],
    )
    .expect("RecordBatch");

    let bounds_path = constraints_dir.join("generic_constraint_bounds.parquet");
    let file = std::fs::File::create(&bounds_path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");

    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, 15_330_000.0, 1e-2, "D13");
    assert!(
        result.iterations <= 10,
        "D13: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-4,
        "D13: gap={:.2e}",
        result.final_gap
    );
}

/// Range constraint whose LOWER bound binds, forcing thermal dispatch above
/// its own physical cap and into a genuine, reported band violation. The
/// range sibling of `d13`: same system, differing only in the constraint.
///
/// ## Case setup
///
/// - 1 bus, 1 thermal T0: capacity 30 MW at $50/MWh, deterministic load 20 MW,
///   2 stages each with 730 hours, no hydro (identical to `d13`'s system).
/// - 1 generic constraint: `35 <= thermal_generation(0) <= 100`
///   with slack penalty $5000/MWh (`d13`'s slack config, unchanged).
/// - Deficit cost: $1000/MWh, excess cost: $0.01/MWh (from `penalties.json`).
///
/// ## Expected cost derivation
///
/// Unconstrained merit-order dispatch sets T0 = 20 MW (matching load; thermal
/// is far cheaper than deficit, and any MW above load only adds excess cost).
/// The band's lower bound of 35 MW exceeds T0's own physical cap of 30 MW, so
/// full compliance is impossible: the LP is forced to trade off dispatch `t`
/// against deficit/excess/violation cost. For `t` in `[20, 30]`:
///
/// ```text
/// cost(t) = 50t + 0.01(t - 20) + 5000(35 - t)   [thermal + excess + violation]
///         = -4949.99 t + 174_999.8
/// ```
///
/// strictly decreasing in `t`, so the minimum sits at the right end of the
/// feasible interval: T0's own 30 MW column cap (never at an interior point,
/// and the `[0, 20]` branch is strictly higher — same shape as `d13`'s
/// deficit-vs-violation trade-off, confirmed by direct substitution below).
///
/// Cost per stage at T0 = 30 MW (excess is generation above the 20 MW load,
/// i.e. 10 MW, not above the band):
///   thermal:    30 MW x $50/MWh x 730 h    = $1,095,000
///   excess:     10 MW x $0.01/MWh x 730 h  = $73
///   violation:  5 MW x $5000/MWh x 730 h   = $18,250,000
///   total:      $19,345,073
///
/// Total (2 stages) = 2 x $19,345,073 = **$38,690,146**
///
/// ## Discriminating power
///
/// A hypothetical implementation that applies only `bound_upper` and silently
/// drops `bound_lower` (the lower half of the band) would build a row with
/// `row_lower = -inf`, `row_upper = 100` — never binding, since T0's own cap
/// (30) is already below 100. Dispatch would then settle back at the
/// unconstrained merit-order level, T0 = 20 MW (no excess, no deficit, no
/// violation): cost per stage = 20 MW x $50/MWh x 730 h = $730,000, total (2
/// stages) = **$1,460,000** — two orders of magnitude below the correct
/// $38,690,146, so the broken implementation fails this case's cost assertion.
///
/// ## Violation record
///
/// The band cannot be fully satisfied (35 MW > T0's 30 MW cap), so the
/// optimum leaves a genuine 5 MW shortfall below the lower bound: each
/// stage's simulation output reports exactly ONE `generic_violations` record
/// for constraint 1, with `slack_value` (the net `s_plus - s_minus`) equal to
/// `5.0` — the fixture is in the VIOLATION-PRESENT case, not the feasible one.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d54_range_constraint() {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let case_dir = Path::new("../../examples/deterministic/d54-range-constraint");

    let constraints_dir = case_dir.join("constraints");
    std::fs::create_dir_all(&constraints_dir).expect("create constraints dir");

    let schema = Arc::new(Schema::new(vec![
        Field::new("constraint_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("block_id", DataType::Int32, true),
        Field::new("bound_lower", DataType::Float64, true),
        Field::new("bound_upper", DataType::Float64, true),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1])),
            Arc::new(Int32Array::from(vec![0, 1])),
            Arc::new(Int32Array::new_null(2)), // block_id null = all blocks
            Arc::new(Float64Array::from(vec![35.0, 35.0])),
            Arc::new(Float64Array::from(vec![100.0, 100.0])),
        ],
    )
    .expect("RecordBatch");

    let bounds_path = constraints_dir.join("generic_constraint_bounds.parquet");
    let file = std::fs::File::create(&bounds_path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");

    let (result, scenario_results, summary) = run_with_simulation(case_dir);

    assert_cost(result.final_lb, 38_690_146.0, 1e-2, "D54");
    assert!(
        result.iterations <= 10,
        "D54: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-4,
        "D54: gap={:.2e}",
        result.final_gap
    );
    assert_cost(summary.mean_cost, 38_690_146.0, 1e-2, "D54-sim");

    assert_eq!(
        scenario_results.len(),
        1,
        "D54: simulation must produce exactly 1 scenario"
    );
    let stages = &scenario_results[0].stages;
    assert_eq!(stages.len(), 2, "D54: expected 2 stages");
    for stage in stages {
        assert_eq!(
            stage.generic_violations.len(),
            1,
            "D54 stage {}: expected exactly one generic constraint violation record",
            stage.stage_id
        );
        let violation = &stage.generic_violations[0];
        assert_eq!(violation.constraint_id, 1, "D54: violation constraint id");
        assert!(
            (violation.slack_value - 5.0).abs() < 1e-2,
            "D54 stage {}: expected a 5 MW band shortfall, got {}",
            stage.stage_id,
            violation.slack_value
        );
    }
}

/// Two-stage thermal dispatch with per-block load factors.
///
/// ## Case setup
///
/// - 1 bus, 2 thermal plants (merit order: T0 at $5/MWh cap 15 MW, T1 at
///   $10/MWh cap 15 MW), deterministic load 20 MW mean, 2 stages each with
///   2 blocks (block 0: 400 hours, block 1: 330 hours), load factors [0.8, 1.2]
///
/// ## Expected cost derivation
///
/// - Block 0: load = 20 * 0.8 = 16 MW.  T0=15 MW, T1=1 MW.
///   cost = (15*5 + 1*10) * 400 = 85 * 400 = 34,000
/// - Block 1: load = 20 * 1.2 = 24 MW.  T0=15 MW, T1=9 MW.
///   cost = (15*5 + 9*10) * 330 = 165 * 330 = 54,450
/// - Cost per stage = 34,000 + 54,450 = 88,450
/// - Total (2 stages) = 2 * 88,450 = **176,900**
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d14_block_factors() {
    use crate::common::parquet_fixtures::write_seasonal_stats;

    let case_dir = Path::new("../../examples/deterministic/d14-block-factors");

    let scenarios_dir = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios_dir).expect("create scenarios dir");

    write_seasonal_stats(
        &scenarios_dir.join("load_seasonal_stats.parquet"),
        "bus_id",
        "mean_mw",
        "std_mw",
        &[(0, 0, 20.0, 0.0), (0, 1, 20.0, 0.0)],
    );

    // Empty inflow stats: D14 has no hydros.
    write_seasonal_stats(
        &scenarios_dir.join("inflow_seasonal_stats.parquet"),
        "hydro_id",
        "mean_m3s",
        "std_m3s",
        &[],
    );

    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, 176_900.0, 1e-4, "D14");
    assert!(
        result.iterations <= 10,
        "D14: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D14: gap={:.2e}",
        result.final_gap
    );
}

/// Two-stage thermal + NCS dispatch.
///
/// ## Case setup
///
/// - 1 bus, 1 thermal (T0 at $10/MWh, cap 100 MW), 1 NCS (curtailment_cost
///   $0.001/MWh, bus 0, max_generation_mw 100 MW), deterministic load 80 MW,
///   2 stages each with 1 block of 730 hours.
/// - NCS available generation = 50 MW per stage (from non_controllable_stats.parquet,
///   mean=0.5, std=0.0 — availability factor 0.5 * 100 MW = 50 MW, deterministic,
///   exercises the stochastic NCS pipeline).
///
/// ## Expected cost derivation
///
/// - NCS generates at full 50 MW (incentivized by negative objective coeff).
/// - Thermal covers remaining 30 MW.
/// - Thermal cost per stage: 30 * 10 * 730 = 219,000
/// - NCS curtailment cost per stage: 0.001 * 50 * 730 = 36.5 (regularization,
///   the LP objective adds -0.001 * block_hours * g_ncs).
///   The NCS contribution to objective = -0.001 * 730 * 50 = -36.5
/// - Total objective per stage = 219,000 + (-36.5) = 218,963.5
/// - Total (2 stages) = 2 * 218,963.5 = **437,927.0**
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d15_non_controllable_source() {
    use crate::common::parquet_fixtures::write_seasonal_stats;

    let case_dir = Path::new("../../examples/deterministic/d15-non-controllable-source");

    let scenarios_dir = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios_dir).expect("create scenarios dir");

    write_seasonal_stats(
        &scenarios_dir.join("load_seasonal_stats.parquet"),
        "bus_id",
        "mean_mw",
        "std_mw",
        &[(0, 0, 80.0, 0.0), (0, 1, 80.0, 0.0)],
    );

    // Empty inflow stats: D15 has no hydros.
    write_seasonal_stats(
        &scenarios_dir.join("inflow_seasonal_stats.parquet"),
        "hydro_id",
        "mean_m3s",
        "std_m3s",
        &[],
    );

    // NCS availability is a factor: mean 0.5 × max 100 MW = 50 MW. std 0 drives
    // the stochastic NCS pipeline with zero noise (deterministic).
    write_seasonal_stats(
        &scenarios_dir.join("non_controllable_stats.parquet"),
        "ncs_id",
        "mean",
        "std",
        &[(0, 0, 0.5, 0.0), (0, 1, 0.5, 0.0)],
    );

    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, 437_927.0, 1e-2, "D15");
    assert!(
        result.iterations <= 10,
        "D15: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-4,
        "D15: gap={:.2e}",
        result.final_gap
    );
}

/// D18 per-stage NCS commissioning dormancy varies across the window.
///
/// The fixture declares two NCS sources on a 3-stage horizon: NCS0 with
/// `entry_stage_id = 1` (commissions at stage 1) and NCS1 with
/// `exit_stage_id = 2` (decommissions at stage 2). Under the dense layout both
/// NCS keep a column at every stage; the per-stage dormancy mask computed by
/// `StudySetup::ncs_stochastic_dormant_for_test` (applying the commissioning
/// predicate to the stored windows, keyed by stochastic slot) marks which is
/// commissioning-dormant:
///
/// - stage 0 → exactly one slot dormant (NCS0 not yet commissioned)
/// - stage 1 → no slot dormant (both active)
/// - stage 2 → exactly one slot dormant (NCS1 decommissioned)
///
/// and the dormant slot at stage 0 differs from the dormant slot at stage 2.
/// This guards the per-stage NCS patch path: a uniform-NCS case (no dormancy at
/// any stage, as in D15) cannot detect an availability bound zeroed onto the
/// wrong stage's columns. Asserting the dormancy genuinely varies across stages
/// confirms the fixture exercises that path rather than collapsing to a uniform
/// (no-dormancy) mask.
///
/// This assertion only builds the stage templates — it does not train — so it
/// runs under the default `cargo test` profile, unconditionally.
#[test]
fn d18_ncs_commissioning_active_set_varies() {
    let case_dir = Path::new("../../examples/deterministic/d18-ncs-commissioning-window");
    let config = cobre_io::parse_config(&case_dir.join("config.json")).expect("config must parse");
    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");
    let prepare_result = prepare_stochastic(
        system,
        case_dir,
        &config,
        42,
        &ScenarioSource::default(),
        None,
    )
    .expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");
    let setup = build_setup_for_case(
        case_dir,
        &config,
        &system,
        prepare_result.stochastic,
        hydro_models,
    );

    let dormant = setup.ncs_stochastic_dormant_for_test();
    assert_eq!(dormant.len(), 3, "D18: expected 3 study stages");

    // The dense column count is the full NCS count (2) at every stage; the
    // dormancy mask, not an active-subset list, carries the per-stage variance.
    let dormant_count = |s: usize| dormant[s].iter().filter(|&&d| d).count();
    assert_eq!(dormant_count(0), 1, "D18 stage 0: exactly one NCS dormant");
    assert_eq!(dormant_count(1), 0, "D18 stage 1: both NCS active");
    assert_eq!(dormant_count(2), 1, "D18 stage 2: exactly one NCS dormant");

    // The load-bearing per-stage property: the dormant set is not uniform across
    // the horizon — stage 0 differs from the fully-commissioned stage, and the
    // dormant slot at stage 0 (NCS0 pre-entry) differs from stage 2 (NCS1 exit).
    assert_ne!(
        dormant[0], dormant[1],
        "D18: dormancy must differ between stage 0 and the commissioned stage"
    );
    assert_ne!(
        dormant[0], dormant[2],
        "D18: the dormant NCS at stage 0 (pre-entry) must differ from stage 2 (post-exit)"
    );
}

/// D33 per-stage block count varies across the horizon.
///
/// The fixture declares three study stages whose `blocks` arrays have different
/// length — 1 / 3 / 2 — so each stage's per-block equipment column stride
/// (turbine/spillage/thermal/bus) diverges from stage 0's. `block_hours_per_stage[t]`
/// carries one entry per block of stage `t`, so its length equals that stage's
/// `blocks.len()` — the per-stage block count threaded through the pipeline.
///
/// This guards the per-stage block-count path: a uniform-block case (every stage
/// shares one block count, as in D14) cannot detect equipment columns read off
/// the wrong stage's block width, because all stages share the same stride.
/// Asserting the block count genuinely differs across stages confirms the
/// fixture exercises that path rather than collapsing to a uniform width.
///
/// This assertion only builds the stage templates — it does not train — so it
/// runs under the default `cargo test` profile, unconditionally.
#[test]
fn d33_per_stage_block_count_varies() {
    let case_dir = Path::new("../../examples/deterministic/d33-per-stage-block-counts");
    let config = cobre_io::parse_config(&case_dir.join("config.json")).expect("config must parse");
    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");
    let prepare_result = prepare_stochastic(
        system,
        case_dir,
        &config,
        42,
        &ScenarioSource::default(),
        None,
    )
    .expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");
    let setup = build_setup_for_case(
        case_dir,
        &config,
        &system,
        prepare_result.stochastic,
        hydro_models,
    );

    // `block_hours_per_stage[t].len()` is the block count of stage `t`
    // (`compute_stage_hours` collects one `duration_hours` per block of the
    // stage), which equals `StageGeometry::n_blks` threaded through the
    // pipeline.
    let block_counts: Vec<usize> = setup
        .inputs
        .stage_data
        .stage_templates
        .block_hours_per_stage
        .iter()
        .map(Vec::len)
        .collect();
    assert_eq!(block_counts, vec![1, 3, 2], "D33: per-stage block counts");

    // The load-bearing per-stage property: the block count is not uniform across
    // the horizon, so the per-block equipment stride genuinely differs per stage.
    let distinct: std::collections::BTreeSet<usize> = block_counts.iter().copied().collect();
    assert!(
        distinct.len() >= 2,
        "D33: per-stage block count must take at least two distinct values, got {block_counts:?}"
    );
    assert_ne!(
        block_counts[0], block_counts[1],
        "D33: block count must differ between stage 0 and stage 1"
    );
}

/// D18: NCS commissioning window — three-stage thermal + NCS dispatch.
///
/// ## Case setup
///
/// - 1 bus, 1 thermal (T0 at $10/MWh, cap 100 MW), deterministic load 80 MW,
///   3 stages each with 1 block of 730 hours, finite horizon, no discount.
/// - Two NCS sources (bus 0, max 100 MW, curtailment_cost $0.001/MWh),
///   availability factor 0.5 (= 50 MW each, std 0 deterministic). NCS0
///   `entry_stage_id = 1`, NCS1 `exit_stage_id = 2`.
///
/// ## Expected cost derivation
///
/// Stages are fully decoupled (no storage, no discount, finite horizon), so the
/// converged lower bound is the sum of per-stage optima. The NCS objective term
/// is `-curtailment_cost × block_hours × g_ncs = -0.73 × g_ncs` (per MW
/// dispatched); thermal is `+7300 × g_thermal`; excess is `+7.3 × g_excess`.
///
/// - Stage 0 (NCS1 active, 50 MW): NCS 50 + thermal 30.
///   `50 × (-0.73) + 30 × 7300 = -36.5 + 219_000 = 218_963.5`.
/// - Stage 1 (NCS0 + NCS1, 100 MW available, load 80): dispatching NCS beyond
///   the 80 MW load would force excess (`+7.3/MW`) against only `-0.73/MW` of
///   NCS incentive, so the optimum dispatches exactly 80 MW of NCS and zero
///   thermal: `80 × (-0.73) = -58.4`.
/// - Stage 2 (NCS0 active, 50 MW): NCS 50 + thermal 30 = `218_963.5`.
///
/// Total = `218_963.5 - 58.4 + 218_963.5 = 437_868.6`.
///
/// The case must stay feasible at stage 0, where NCS0 is not yet commissioned —
/// the unlimited deficit segment plus the 100 MW thermal cap guarantee it.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d18_ncs_commissioning_window() {
    let case_dir = Path::new("../../examples/deterministic/d18-ncs-commissioning-window");
    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, 437_868.6, 1e-2, "D18");
    assert!(
        result.iterations <= 10,
        "D18: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-4,
        "D18: gap={:.2e}",
        result.final_gap
    );
}

/// D16: PAR(1) lag-shift deterministic test.
///
/// ## System
///
/// 1 bus, 1 hydro (H0), 3 stages, constant productivity = 1.0 MW/(m3/s).
/// No storage (min=max=0). Max turbined = 200 m3/s, max generation = 200 MW.
/// Load = 200 MW constant. Deficit cost = 1000 $/MWh. Block = 730 hours.
///
/// ## PAR(1) model
///
/// psi = 0.5 at all stages, mean = 100 m3/s, std ~ 0 (deterministic).
/// Initial lag seed = 200 m3/s.
///
/// ## Expected inflows with correct lag shift
///
/// Z_0 = 100 + 0.5 * (200 - 100) = 150 m3/s
/// Z_1 = 100 + 0.5 * (150 - 100) = 125 m3/s
/// Z_2 = 100 + 0.5 * (125 - 100) = 112.5 m3/s
///
/// ## Expected cost
///
/// Deficit per stage = 200 - Z_t MW.
/// Cost = sum_t[ deficit_t * 1000 * 730 ]
///      = (50 + 75 + 87.5) * 730000
///      = 212.5 * 730000
///      = 155_125_000
///
/// Without lag shift (bug): every stage sees lag=200, Z_t=150 for all t.
/// Cost = 3 * 50 * 730000 = 109_500_000. Fails with correct cost.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d16_par1_lag_shift() {
    let case_dir = Path::new("../../examples/deterministic/d16-par1-lag-shift");
    let result = run_deterministic(case_dir);
    assert!(
        result.final_lb > 0.0,
        "D16: lower bound must be positive, got {}",
        result.final_lb
    );
    // The lag shift makes inflows decrease across stages (Z = 150, 125, 112.5),
    // producing higher deficits than the no-shift bug (Z = 150 at every stage).
    // Pinned value adds the universal `turbined_cost` regularization (`≈ 2_828.75 $`)
    // to the deficit-only cost `7_756_250.0`.
    assert_cost(result.final_lb, 7_759_078.749_993_78, 1.0, "D16");
}

/// Regression guard for the model-persistence optimization.
///
/// Runs D01 (2 stages, 2 thermals, deterministic) and verifies that the
/// solver's `load_model_count` is consistent with per-stage loading, NOT
/// per-scenario loading. With model persistence, `load_model` is called
/// once per stage per iteration (forward + backward + lower bound), not
/// once per (scenario, stage) pair.
///
/// Numerical equivalence is verified by the `d01_thermal_dispatch` test;
/// this test additionally checks the call count invariant.
#[test]
fn model_persistence_regression_d01() {
    let case_dir = Path::new("../../examples/deterministic/d01-thermal-dispatch");
    let (result, solver) = run_deterministic_with_solver(case_dir);

    assert_cost(result.final_lb, 182_500.0, 1e-6, "D01-persistence");

    let stats = solver.statistics();
    let n_stages = 2_u64;
    let forward_passes = 2_u64;
    let iterations = result.iterations;

    let without_persistence_forward = n_stages * forward_passes * iterations;
    let with_persistence_forward = n_stages * iterations;

    // load_model_count (forward + backward + LB) must stay below the
    // per-scenario forward-only count, confirming per-stage persistence is active.
    assert!(
        stats.load_model_count < without_persistence_forward,
        "model persistence regression: load_model_count ({}) should be < {} (per-scenario forward-only count), \
         expected ~{} for persisted forward",
        stats.load_model_count,
        without_persistence_forward,
        with_persistence_forward
    );
}

// ---------------------------------------------------------------------------
// Incremental cut management integration tests
// ---------------------------------------------------------------------------

/// Verify the LB solver's incremental cut management reduces `load_model_count`
/// compared to the full-rebuild baseline.
///
/// Runs D03 (3-stage, 2-hydro cascade) which runs for 10 iterations. The LB
/// solver uses a dedicated CutRowMap and only calls `load_model` once (first
/// iteration). Forward + backward still call `load_model` per stage per
/// iteration.
///
/// Expected load_model breakdown with incremental LB:
/// - Forward: n_stages * iterations = 3 * 10 = 30
/// - Backward: (n_stages - 1) * iterations = 2 * 10 = 20
/// - LB: 1 (first iteration only, incremental thereafter)
/// - Total ~51
///
/// Without incremental LB, the total would be 30 + 20 + 10 = 60.
/// We verify that load_model_count is strictly less than the non-incremental total.
#[test]
fn incremental_lb_reduces_load_model_count() {
    let case_dir = Path::new("../../examples/deterministic/d03-two-hydro-cascade");
    let (result, solver) = run_deterministic_with_solver(case_dir);

    assert_cost(result.final_lb, D03_EXPECTED_COST, 1e-4, "D03-incremental");

    let stats = solver.statistics();
    let n_stages = 3_u64;
    let iterations = result.iterations;

    // Non-incremental LB would call load_model once per iteration; incremental
    // calls it only on the first. Total budget = forward + backward + LB.
    let non_incremental_lb = iterations;
    let forward_count = n_stages * iterations;
    let backward_count = (n_stages - 1) * iterations;
    let total_without_incremental = forward_count + backward_count + non_incremental_lb;

    assert!(
        stats.load_model_count < total_without_incremental,
        "incremental LB should reduce load_model_count: got {} >= {} (non-incremental total), \
         iterations={}, n_stages={}",
        stats.load_model_count,
        total_without_incremental,
        iterations,
        n_stages
    );

    // The LB solver does 1 load_model instead of `iterations`, so the reduction
    // is at least (iterations - 1).
    let expected_savings = iterations.saturating_sub(1);
    let actual_savings = total_without_incremental - stats.load_model_count;
    assert!(
        actual_savings >= expected_savings,
        "LB incremental savings should be >= {} (iterations - 1), got {} savings \
         (total_without={}, actual={})",
        expected_savings,
        actual_savings,
        total_without_incremental,
        stats.load_model_count
    );
}

/// Multi-hydro PAR(2) regression test with inflow truncation.
///
/// ## Case setup
///
/// - 1 bus (B0), 1 thermal (T0: 200 MW at $50/MWh), 2 hydros:
///   - H0: constant productivity 1.0 MW/(m3/s), max turbined 100 m3/s,
///     storage 0–200 hm3, PAR(2) with psi = [0.5, 0.3], mean = 40 m3/s
///   - H1: constant productivity 0.8 MW/(m3/s), max turbined 80 m3/s,
///     storage 0–150 hm3, PAR(2) with psi = [0.4, 0.2], mean = 25 m3/s
/// - Deterministic load: 100 MW per stage
/// - Initial storage: H0 = 100 hm3, H1 = 75 hm3
/// - Pre-study lag seed (Nov/Dec 2023 `recent_observations`): H0 = [50, 45] m3/s,
///   H1 = [30, 28] m3/s
/// - 3 stages × 730 h, `inflow_non_negativity: {method: "truncation"}`
///
/// ## What this tests
///
/// With 2 hydros and PAR(2), the lag state indices are:
///   - `inflow_lags.start + 0*2 + 0` = hydro 0, lag 0
///   - `inflow_lags.start + 0*2 + 1` = hydro 0, lag 1  (BUG: was hydro 1, lag 0)
///   - `inflow_lags.start + 1*2 + 0` = hydro 1, lag 0
///   - `inflow_lags.start + 1*2 + 1` = hydro 1, lag 1  (BUG: was hydro 0, lag 1)
///
/// If the hydro-major/lag-major bug regressed, the wrong lag values would be
/// used in PAR evaluation, producing a different optimal cost.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d19_multi_hydro_par_truncation() {
    let case_dir = Path::new("../../examples/deterministic/d19-multi-hydro-par");
    let result = run_deterministic(case_dir);

    // The system with PAR(2) truncation and 2 hydros must produce a positive cost.
    assert!(
        result.final_lb > 0.0,
        "D19: lower bound must be positive, got {}",
        result.final_lb
    );
    // No iteration bound: the 2-hydro × 2-lag state space need not converge
    // within 50 iterations. The LB match is the lag-major indexing regression check.
    assert_cost(result.final_lb, D19_EXPECTED_COST, 1.0, "D19");
}

/// Expected lower bound for D19 (2-hydro PAR(2) with truncation, 3 stages).
///
/// Empirical, not hand-computable (2-hydro × 2-lag state space). D19 is a
/// 2-hydro AR(2) windowed case whose stage-0 lags reference the pre-study
/// Nov/Dec 2023 seasons (`stage_id = -1, -2`); the seed `[50, 45, 30, 28]`
/// (hydro 0 then hydro 1, most-recent-lag-first) produces this cost, while a
/// zero seed or a wrong-hydro-major seed produces a materially different one
/// — the hydro-major/lag-major lag-indexing regression guard.
pub const D19_EXPECTED_COST: f64 = 1_334_568.013_586_834_3;

/// Operational violation slacks: 1 hydro with active min_outflow, max_outflow,
/// min_turbined, and min_generation bounds.
///
/// ## Case setup (D20)
///
/// - 1 bus, 1 hydro, 0 thermals, 1 block (730h), 2 stages, deterministic.
/// - Hydro: min_outflow=40, max_outflow=50, min_turbined=30, min_generation=20,
///   max_turbined=50, productivity=1.0, max_storage=200, initial_storage=10.
/// - Inflows: stage 0 = 40 m3/s, stage 1 = 10 m3/s (zero std_dev).
/// - Penalty costs: all 4 operational violations = 5000 $/MWh, deficit = 1000 $/MWh.
///
/// ## Expected behaviour
///
/// With low initial storage (10 hm3), the hydro cannot sustain 40 m3/s
/// min_outflow at either stage. At stage 0, total available water is
/// 10 + 40*2.628 = 115.12 hm3, but the optimizer splits water across
/// stages, leading to outflow below 40 m3/s. At stage 1, even less water
/// is available, forcing min_outflow and min_turbined violation slacks.
///
/// The expected cost is recorded empirically and locked for regression.
/// Simulation is also run to verify non-zero operational violation slacks.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d20_operational_violations() {
    let case_dir = Path::new("../../examples/deterministic/d20-operational-violations");
    let (result, scenario_results, summary) = run_with_simulation(case_dir);

    assert!(
        result.iterations <= 20,
        "D20: iterations={} (expected <= 20)",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D20: gap={:.2e} (expected < 1e-6)",
        result.final_gap
    );
    assert_cost(result.final_lb, D20_EXPECTED_COST, 1e-2, "D20");
    assert_eq!(summary.n_scenarios, 1);
    assert_cost(summary.mean_cost, D20_EXPECTED_COST, 1e-2, "D20-sim");

    assert_eq!(scenario_results.len(), 1);
    let scenario = &scenario_results[0];
    assert_eq!(scenario.stages.len(), 2);

    let found_outflow_below = scenario
        .stages
        .iter()
        .flat_map(|s| &s.hydros)
        .any(|h| h.outflow_slack_below_m3s > 1e-10);
    let found_turbine_below = scenario
        .stages
        .iter()
        .flat_map(|s| &s.hydros)
        .any(|h| h.turbined_slack_m3s > 1e-10);
    assert!(
        found_outflow_below,
        "D20: expected non-zero outflow_slack_below_m3s"
    );
    assert!(
        found_turbine_below,
        "D20: expected non-zero turbined_slack_m3s"
    );
}

/// Empirical lower bound, initial_storage=10 hm3 (includes the universal
/// `turbined_cost` regularization).
pub const D20_EXPECTED_COST: f64 = 195_744_837.222_222_24;

/// LP consistency test: cost consistency between outflow violation slacks
/// and `hydro_violation_cost`. 1 hydro (min_outflow=50 m3/s), 1 thermal,
/// inflow=10 m3/s (insufficient), initial_storage=5 hm3, penalty=5000.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d21_min_outflow_regression() {
    use crate::common::parquet_fixtures::write_seasonal_stats;

    let case_dir = Path::new("../../examples/deterministic/d21-min-outflow-regression");

    let scenarios_dir = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios_dir).expect("create scenarios dir");

    write_seasonal_stats(
        &scenarios_dir.join("load_seasonal_stats.parquet"),
        "bus_id",
        "mean_mw",
        "std_mw",
        &[(0, 0, 20.0, 0.0), (0, 1, 20.0, 0.0)],
    );

    write_seasonal_stats(
        &scenarios_dir.join("inflow_seasonal_stats.parquet"),
        "hydro_id",
        "mean_m3s",
        "std_m3s",
        &[(0, 0, 10.0, 0.0), (0, 1, 10.0, 0.0)],
    );

    let (result, scenario_results, summary) = run_with_simulation(case_dir);

    assert!(
        result.iterations <= 20,
        "D21: iterations={} (expected <= 20)",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D21: gap={:.2e} (expected < 1e-6)",
        result.final_gap
    );
    assert_cost(result.final_lb, D21_EXPECTED_COST, 1e-2, "D21");
    assert_eq!(summary.n_scenarios, 1);
    assert_cost(
        summary.mean_cost,
        result.final_lb,
        1e-2,
        "D21-sim-vs-training",
    );

    assert_eq!(scenario_results.len(), 1);
    let scenario = &scenario_results[0];
    assert_eq!(scenario.stages.len(), 2);

    let found_outflow_below = scenario
        .stages
        .iter()
        .flat_map(|s| &s.hydros)
        .any(|h| h.outflow_slack_below_m3s > 1e-10);
    assert!(
        found_outflow_below,
        "D21: expected non-zero outflow_slack_below_m3s"
    );

    // Per-block formulation: slack is in m3/s, objective = penalty * block_hours.
    let penalty = 5000.0_f64;
    let hours = 730.0_f64;

    let mut total_hydro_violation_cost = 0.0;
    for (s, stage_result) in scenario.stages.iter().enumerate() {
        assert_eq!(stage_result.hydros.len(), 1);
        assert_eq!(stage_result.costs.len(), 1);
        let slack_m3s = stage_result.hydros[0].outflow_slack_below_m3s;
        let stage_violation_cost = stage_result.costs[0].hydro_violation_cost;
        total_hydro_violation_cost += stage_violation_cost;

        if slack_m3s > 1e-10 {
            let expected_cost = slack_m3s * penalty * hours;
            let cost_diff = (stage_violation_cost - expected_cost).abs();
            assert!(
                cost_diff < 1e-2,
                "D21 stage {s}: hydro_violation_cost={stage_violation_cost}, \
                 expected={expected_cost}, diff={cost_diff}"
            );
        }
    }
    assert!(
        total_hydro_violation_cost > 0.0,
        "D21: hydro_violation_cost must be positive"
    );

    // Decomposed cost fields. D21 has only min-outflow violations, so every
    // other violation component must be zero and the below-cost must carry the
    // whole hydro_violation_cost.
    for (s, stage_result) in scenario.stages.iter().enumerate() {
        let cost = &stage_result.costs[0];

        let component_sum = cost.outflow_violation_below_cost
            + cost.outflow_violation_above_cost
            + cost.turbined_violation_cost
            + cost.generation_violation_cost
            + cost.evaporation_violation_cost
            + cost.withdrawal_violation_cost;
        assert!(
            (cost.hydro_violation_cost - component_sum).abs() < 1e-6,
            "D21 stage {s}: sum invariant failed: hydro_violation_cost={}, component_sum={}",
            cost.hydro_violation_cost,
            component_sum
        );

        assert!(
            cost.outflow_violation_above_cost.abs() < 1e-10,
            "D21 stage {s}: outflow_above should be 0, got {}",
            cost.outflow_violation_above_cost
        );
        assert!(
            cost.turbined_violation_cost.abs() < 1e-10,
            "D21 stage {s}: turbined should be 0, got {}",
            cost.turbined_violation_cost
        );
        assert!(
            cost.generation_violation_cost.abs() < 1e-10,
            "D21 stage {s}: generation should be 0, got {}",
            cost.generation_violation_cost
        );
        assert!(
            cost.evaporation_violation_cost.abs() < 1e-10,
            "D21 stage {s}: evaporation should be 0, got {}",
            cost.evaporation_violation_cost
        );
        assert!(
            cost.withdrawal_violation_cost.abs() < 1e-10,
            "D21 stage {s}: withdrawal should be 0, got {}",
            cost.withdrawal_violation_cost
        );

        let slack_m3s = stage_result.hydros[0].outflow_slack_below_m3s;
        if slack_m3s > 1e-10 {
            let expected_below_cost = slack_m3s * penalty * hours;
            assert!(
                (cost.outflow_violation_below_cost - expected_below_cost).abs() < 1e-2,
                "D21 stage {s}: outflow_violation_below_cost={}, expected={}",
                cost.outflow_violation_below_cost,
                expected_below_cost
            );
        }
    }

    let found_below_cost = scenario
        .stages
        .iter()
        .flat_map(|s| &s.costs)
        .any(|c| c.outflow_violation_below_cost > 1e-10);
    assert!(
        found_below_cost,
        "D21: expected non-zero outflow_violation_below_cost in at least one stage"
    );
}

/// Empirical lower bound, initial_storage=5 hm3, inflow=10 m3/s (includes the
/// universal `turbined_cost` regularization).
pub const D21_EXPECTED_COST: f64 = 285_716_271.0;

/// D22: Multi-block per-block min outflow regression test.
///
/// 1 hydro (min_outflow=30 m3/s), 1 thermal, 3 blocks per stage (200h, 300h, 230h),
/// inflow=10 m3/s. Violation of 20 m3/s in every block at penalty 5000 $/m3/s.
/// Validates that per-block constraints prevent the optimizer from concentrating
/// flow into one block while starving others.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d22_per_block_min_outflow() {
    use crate::common::parquet_fixtures::write_seasonal_stats;

    let case_dir = Path::new("../../examples/deterministic/d22-per-block-min-outflow");

    let scenarios_dir = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios_dir).expect("create scenarios dir");

    write_seasonal_stats(
        &scenarios_dir.join("load_seasonal_stats.parquet"),
        "bus_id",
        "mean_mw",
        "std_mw",
        &[(0, 0, 20.0, 0.0), (0, 1, 20.0, 0.0)],
    );

    write_seasonal_stats(
        &scenarios_dir.join("inflow_seasonal_stats.parquet"),
        "hydro_id",
        "mean_m3s",
        "std_m3s",
        &[(0, 0, 10.0, 0.0), (0, 1, 10.0, 0.0)],
    );

    let (result, scenario_results, summary) = run_with_simulation(case_dir);

    assert!(
        result.iterations <= 20,
        "D22: iterations={} (expected <= 20)",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D22: gap={:.2e} (expected < 1e-6)",
        result.final_gap
    );
    assert_cost(result.final_lb, D22_EXPECTED_COST, 1e-2, "D22");
    assert_eq!(summary.n_scenarios, 1);
    assert_cost(
        summary.mean_cost,
        result.final_lb,
        1e-2,
        "D22-sim-vs-training",
    );

    let scenario = &scenario_results[0];
    let block_hours = [200.0_f64, 300.0, 230.0];
    let penalty = 5000.0_f64;

    for (s, stage_result) in scenario.stages.iter().enumerate() {
        // 1 hydro × 3 blocks = 3 rows.
        assert_eq!(
            stage_result.hydros.len(),
            3,
            "D22 stage {s}: expected 3 per-block hydro rows"
        );

        for (b, hr) in stage_result.hydros.iter().enumerate() {
            // inflow=10 < min_outflow=30, so every block violates.
            assert!(
                hr.outflow_slack_below_m3s > 1e-6,
                "D22 stage {s} block {b}: outflow_slack_below_m3s should be > 0, got {}",
                hr.outflow_slack_below_m3s
            );
        }

        // Slack differs per block: penalty ∝ block_hours, so the optimizer
        // concentrates outflow into the longer (more expensive) blocks.
        assert_eq!(stage_result.costs.len(), 1);
        let total_violation_cost = stage_result.costs[0].hydro_violation_cost;
        let expected_total: f64 = stage_result
            .hydros
            .iter()
            .enumerate()
            .map(|(b, hr)| hr.outflow_slack_below_m3s * penalty * block_hours[b])
            .sum();
        assert!(
            (total_violation_cost - expected_total).abs() < 1e-2,
            "D22 stage {s}: hydro_violation_cost={total_violation_cost}, expected={expected_total}"
        );
    }
}

/// Empirical lower bound, multi-block (3 blocks) per-block min outflow
/// (includes the universal `turbined_cost` regularization).
pub const D22_EXPECTED_COST: f64 = 140_376_826.555_555_58;

/// D23: Bidirectional withdrawal -- over-withdrawal activation.
///
/// ## Case setup
///
/// - 1 bus (B0), 1 thermal (T0: 200 MW at $100/MWh), 1 hydro (H0: constant
///   productivity 1.0 MW/(m³/s), max turbine 20 m³/s, reservoir 0-10 hm³)
/// - Deterministic inflows: 50.0 m³/s per stage (high, to create water excess)
/// - Water withdrawal target: 5 m³/s per stage
/// - Deterministic load: 80.0 MW per stage
/// - Initial storage: 5.0 hm³
/// - 2 stages x 730 h, `inflow_non_negativity: none`
///
/// ## Penalty structure (asymmetric)
///
/// - `water_withdrawal_violation_pos_cost`: 1.0 (cheap over-withdrawal)
/// - `water_withdrawal_violation_neg_cost`: 10,000.0 (expensive under-withdrawal)
/// - `spillage_cost`: 1,000.0 (expensive spillage)
///
/// ## Why over-withdrawal activates
///
/// kappa = 730 * 3600 / 1e6 = 2.628 hm³/(m³/s)
///
/// Water excess per stage: inflow (50) - withdrawal_target (5) - max_turbine (20) = 25 m³/s
/// Storage fill from excess: 25 * 2.628 = 65.7 hm³ >> max_storage (10 hm³)
///
/// The solver must shed excess water. Two options:
/// 1. Spill: cost = 1,000 * 730 = 730,000 per m³/s
/// 2. Over-withdraw: cost = 1.0 * 730 = 730 per m³/s
///
/// Over-withdrawal is ~1000x cheaper, so the solver strongly prefers `ww_pos > 0`.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d23_bidirectional_withdrawal() {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    use crate::common::parquet_fixtures::write_seasonal_stats;

    let case_dir = Path::new("../../examples/deterministic/d23-bidirectional-withdrawal");

    let scenarios_dir = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios_dir).expect("create scenarios dir");

    write_seasonal_stats(
        &scenarios_dir.join("load_seasonal_stats.parquet"),
        "bus_id",
        "mean_mw",
        "std_mw",
        &[(0, 0, 80.0, 0.0), (0, 1, 80.0, 0.0)],
    );

    write_seasonal_stats(
        &scenarios_dir.join("inflow_seasonal_stats.parquet"),
        "hydro_id",
        "mean_m3s",
        "std_m3s",
        &[(0, 0, 50.0, 0.0), (0, 1, 50.0, 0.0)],
    );

    let constraints_dir = case_dir.join("constraints");
    std::fs::create_dir_all(&constraints_dir).expect("create constraints dir");

    let bounds_schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("water_withdrawal_m3s", DataType::Float64, false),
    ]));
    let bounds_batch = RecordBatch::try_new(
        Arc::clone(&bounds_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 0])),
            Arc::new(Int32Array::from(vec![0, 1])),
            Arc::new(Float64Array::from(vec![5.0, 5.0])),
        ],
    )
    .expect("bounds RecordBatch");
    let file = std::fs::File::create(constraints_dir.join("hydro_bounds.parquet"))
        .expect("create bounds parquet");
    let mut writer = ArrowWriter::try_new(file, bounds_schema, None).expect("ArrowWriter");
    writer.write(&bounds_batch).expect("write bounds batch");
    writer.close().expect("close bounds writer");

    let (result, scenario_results, _summary) = run_with_simulation(case_dir);

    assert!(
        result.iterations <= 20,
        "D23: iterations={} (expected <= 20)",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D23: gap={:.2e} (expected < 1e-6)",
        result.final_gap
    );

    assert_eq!(scenario_results.len(), 1);
    let scenario = &scenario_results[0];
    assert_eq!(scenario.stages.len(), 2);

    let mut found_ww_pos = false;
    for stage_result in &scenario.stages {
        for hydro_result in &stage_result.hydros {
            if hydro_result.water_withdrawal_violation_pos_m3s > 1e-10 {
                found_ww_pos = true;
            }
            // Under-withdrawal must not activate: its neg cost is high.
            assert!(
                hydro_result.water_withdrawal_violation_neg_m3s < 1e-10,
                "D23: unexpected under-withdrawal violation: {}",
                hydro_result.water_withdrawal_violation_neg_m3s
            );
        }
    }
    assert!(
        found_ww_pos,
        "D23: expected non-zero water_withdrawal_violation_pos_m3s (over-withdrawal)"
    );

    // Water balance identity, per stage:
    // V_out = V_in + kappa * (inflow - ww_target + ww_neg - ww_pos - turbined - spillage)
    let kappa = 730.0 * 3600.0 / 1e6; // hm3 per (m3/s)
    let ww_target = 5.0;
    let inflow = 50.0;

    for (s, stage_result) in scenario.stages.iter().enumerate() {
        assert_eq!(stage_result.hydros.len(), 1);
        let h = &stage_result.hydros[0];

        let net_flow = inflow - ww_target + h.water_withdrawal_violation_neg_m3s
            - h.water_withdrawal_violation_pos_m3s
            - h.turbined_m3s
            - h.spillage_m3s;
        let expected_v_out = h.storage_initial_hm3 + kappa * net_flow;
        let diff = (h.storage_final_hm3 - expected_v_out).abs();
        assert!(
            diff < 1e-6,
            "D23 stage {s}: water balance mismatch: V_out={}, expected={expected_v_out}, diff={diff}",
            h.storage_final_hm3
        );
    }
}

/// Verify per-block bus balance: generation + deficit - excess + net_exchange =
/// load. Catches LP-extraction mismatches where a bus-balance coefficient
/// (e.g. productivity) diverges between the LP builder and the extraction pipeline.
///
/// # Limitations
///
/// Entity result structs carry no `bus_id`, so this sums generation across all
/// buses — accurate only for single-bus systems. Multi-bus systems with exchange
/// lines would need a bus-entity mapping for per-bus balance.
fn assert_bus_balance(stage: &cobre_sddp::SimulationStageResult, tolerance: f64, label: &str) {
    let mut block_ids: Vec<u32> = stage.buses.iter().filter_map(|b| b.block_id).collect();
    block_ids.sort_unstable();
    block_ids.dedup();

    for &block_id in &block_ids {
        let hydro_gen: f64 = stage
            .hydros
            .iter()
            .filter(|h| h.block_id == Some(block_id))
            .map(|h| h.generation_mw)
            .sum();
        let thermal_gen: f64 = stage
            .thermals
            .iter()
            .filter(|t| t.block_id == Some(block_id))
            .map(|t| t.generation_mw)
            .sum();
        let ncs_gen: f64 = stage
            .non_controllables
            .iter()
            .filter(|n| n.block_id == Some(block_id))
            .map(|n| n.generation_mw)
            .sum();
        let deficit: f64 = stage
            .buses
            .iter()
            .filter(|b| b.block_id == Some(block_id))
            .map(|b| b.deficit_mw)
            .sum();
        let excess: f64 = stage
            .buses
            .iter()
            .filter(|b| b.block_id == Some(block_id))
            .map(|b| b.excess_mw)
            .sum();
        // Net exchange sign: direct flow enters the bus, reverse flow leaves.
        let net_exchange: f64 = stage
            .exchanges
            .iter()
            .filter(|e| e.block_id == Some(block_id))
            .map(|e| e.direct_flow_mw - e.reverse_flow_mw)
            .sum();
        let load: f64 = stage
            .buses
            .iter()
            .filter(|b| b.block_id == Some(block_id))
            .map(|b| b.load_mw)
            .sum();

        let supply = hydro_gen + thermal_gen + ncs_gen + deficit - excess + net_exchange;
        let mismatch = (supply - load).abs();
        assert!(
            mismatch < tolerance,
            "{label} stage {} block {block_id}: bus balance mismatch: \
             supply={supply:.6} (hydro={hydro_gen:.6} + thermal={thermal_gen:.6} \
             + ncs={ncs_gen:.6} + deficit={deficit:.6} - excess={excess:.6} \
             + exchange={net_exchange:.6}) vs load={load:.6}, diff={mismatch:.2e}",
            stage.stage_id
        );
    }
}

/// Expected cost for D24: per-stage productivity override (rho_0=0.8, rho_1=1.2).
///
/// ## Case setup
///
/// Same physical system as D02: 1 bus, 1 thermal (T0: 100 MW at $50/MWh),
/// 1 hydro (H0: max 50 m3/s, storage 0-200 hm3, max_generation 50 MW),
/// demand 80 MW, 2 stages x 730 h, initial storage 100 hm3, deterministic
/// inflows 40/10 m3/s.
///
/// Unlike D02, H0 uses per-stage productivity overrides via
/// `hydro_production_models.json`: stage 0 rho=0.8, stage 1 rho=1.2.
/// The entity-level `productivity_mw_per_m3s = 1.0` must NOT be used.
///
/// ## Expected cost derivation
///
/// kappa = 730 * 3600 / 1e6 = 657/250 = 2.628 hm3/(m3/s * stage).
///
/// Key constraint interaction: with rho_1=1.2, the generation cap (50 MW)
/// limits effective turbining: gen_h1 = 1.2 * q1 <= 50 => q1 <= 125/3 m3/s.
/// In stage 0, rho_0=0.8 leaves q0 <= 50 (gen_h0 = 0.8*50 = 40 < 50 MW).
///
/// Since rho_1 > rho_0, water is more valuable in stage 1. The optimizer
/// stores water in stage 0 to use it at higher productivity in stage 1.
/// The optimal V1 = (125/3 - 10) * kappa = 4161/50 = 83.22 hm3, making
/// V2 = 0 with q1 = 125/3 m3/s (generation cap binding).
///
/// Stage 0: q0 = 30475/657 m3/s, gen_h0 = 0.8 * q0 = 24380/657 MW,
///   g_th0 = 80 - 24380/657 = 28180/657 MW.
///   Cost_0 = 50 * 730 * 28180/657 = 14090000/9.
///
/// Stage 1: q1 = 125/3 m3/s, gen_h1 = 1.2 * 125/3 = 50 MW (cap),
///   g_th1 = 80 - 50 = 30 MW, V2 = 0 hm3.
///   Cost_1 = 50 * 730 * 30 = 1095000.
///
/// Total = 14090000/9 + 1095000 = 23945000/9 ~ 2660555.56 $.
///
/// ## Bug detection
///
/// If the bug were present (using entity rho=1.0 instead of the overrides),
/// the cost would equal D02: 23635000/9 ~ 2626111.11 $. The difference
/// (~ $34444) is well above the 1e-4 tolerance, so the test catches the bug.
///
/// The pinned value adds D02's universal `turbined_cost` regularization
/// (`5_785 / 9 ≈ 642.78 $`, same single-hydro turbined flows) to the analytical
/// thermal cost `23_945_000 / 9`, yielding `23_950_785 / 9`.
pub const D24_EXPECTED_COST: f64 = 23_950_785.0 / 9.0;

/// D24: Productivity override — per-stage productivity from `hydro_production_models.json`.
///
/// Same physical system as D02 except H0 has per-stage productivity overrides:
/// stage 0 -> rho = 0.8, stage 1 -> rho = 1.2 (entity model says 1.0).
///
/// This test catches the bus balance productivity mismatch bug: if the LP uses
/// the entity-level productivity (1.0) instead of the per-stage override, the
/// optimal cost would differ.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d24_productivity_override() {
    use crate::common::parquet_fixtures::write_seasonal_stats;

    let case_dir = Path::new("../../examples/deterministic/d24-productivity-override");

    let scenarios_dir = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios_dir).expect("create scenarios dir");

    write_seasonal_stats(
        &scenarios_dir.join("inflow_seasonal_stats.parquet"),
        "hydro_id",
        "mean_m3s",
        "std_m3s",
        &[(0, 0, 40.0, 0.0), (0, 1, 10.0, 0.0)],
    );

    write_seasonal_stats(
        &scenarios_dir.join("load_seasonal_stats.parquet"),
        "bus_id",
        "mean_mw",
        "std_mw",
        &[(0, 0, 80.0, 0.0), (0, 1, 80.0, 0.0)],
    );

    let (result, scenario_results, _summary) = run_with_simulation(case_dir);
    assert_cost(result.final_lb, D24_EXPECTED_COST, 1e-4, "D24");
    assert!(
        result.iterations <= 10,
        "D24: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D24: gap={:.2e}",
        result.final_gap
    );
    assert!(
        (result.final_lb - D02_EXPECTED_COST).abs() > 1.0,
        "D24: cost must differ from D02 (per-stage overrides change economics)"
    );

    assert_eq!(
        scenario_results.len(),
        1,
        "D24: expected 1 simulation scenario"
    );
    for stage in &scenario_results[0].stages {
        assert_bus_balance(stage, 1e-3, "D24");
    }
}

// ---------------------------------------------------------------------------
// D25: Discount rate
// ---------------------------------------------------------------------------

/// Expected cost for D25 (single hydro, 2 stages, 12% annual discount rate).
///
/// Same physical system as D02 but with `annual_discount_rate: 0.12`.
/// The one-step discount factor for stage 0 (31-day January) is:
///
/// `d_0 = 1 / (1.12)^(31/365.25) ≈ 0.9904`
///
/// The discount factor multiplies the theta (future cost) coefficient in
/// the stage-0 LP objective, reducing the present value of future costs.
/// This shifts the optimal dispatch toward less water conservation, yielding
/// a lower total present-value cost than the undiscounted D02 case.
/// Includes the universal `turbined_cost` regularization (`≈ 640.12 $`).
const D25_EXPECTED_COST: f64 = 2_612_094.703_543_594_6;

/// D25: Two-stage single-hydro with 12% annual discount rate.
///
/// Verifies that the discounted SDDP lower bound converges to the correct
/// present-value cost, and that it is strictly less than D02's undiscounted LB.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d25_discount_rate() {
    let case_dir = Path::new("../../examples/deterministic/d25-discount-rate");
    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, D25_EXPECTED_COST, 1e-4, "D25");
    assert!(
        result.iterations <= 10,
        "D25: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D25: gap={:.2e}",
        result.final_gap
    );
    assert!(
        result.final_lb < D02_EXPECTED_COST,
        "D25: discounted LB ({}) must be < undiscounted D02 LB ({})",
        result.final_lb,
        D02_EXPECTED_COST,
    );
}

/// D25: Verify simulation discount factors match expected cumulative factors.
///
/// Runs training + simulation on the D25 case and asserts that:
/// - Stage 0 cumulative discount factor = 1.0 (always)
/// - Stage 1 cumulative discount factor = d_0 = 1/(1.12)^(31/365.25)
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d25_simulation_discount_factors() {
    let case_dir = Path::new("../../examples/deterministic/d25-discount-rate");
    let (result, scenarios, _summary) = run_with_simulation(case_dir);

    assert_cost(result.final_lb, D25_EXPECTED_COST, 1e-4, "D25-sim");

    assert_eq!(scenarios.len(), 1, "D25: expected 1 simulation scenario");
    let stages = &scenarios[0].stages;
    assert_eq!(stages.len(), 2, "D25: expected 2 stages");

    // Stage 0 cumulative discount factor is always 1.0.
    let df0 = stages[0].costs[0].discount_factor;
    assert!(
        (df0 - 1.0).abs() < 1e-12,
        "D25: stage 0 discount_factor expected 1.0, got {df0}"
    );

    let d0 = 1.0_f64 / 1.12_f64.powf(31.0 / 365.25);
    let df1 = stages[1].costs[0].discount_factor;
    assert!(
        (df1 - d0).abs() < 1e-10,
        "D25: stage 1 discount_factor expected {d0}, got {df1}"
    );
}

// ---------------------------------------------------------------------------
// D26: Estimated PAR(2) — regression guard for the forward-prediction fix
// ---------------------------------------------------------------------------

/// D26 expected lower bound — regression guard that PAR prediction uses the
/// forward (not backward) lag. Includes the universal `turbined_cost`
/// regularization (`≈ 17_830.06 $`).
pub const D26_EXPECTED_COST: f64 = 50_625_314.970_196_81;

/// D26: PAR(2) estimation from inflow history (regression guard for forward-prediction fix).
/// Exercises full PAR(p) pipeline with PACF order selection and Yule-Walker fitting.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d26_estimated_par2() {
    let case_dir = Path::new("../../examples/deterministic/d26-estimated-par2");
    let result = run_deterministic(case_dir);

    assert!(
        result.final_lb > 0.0,
        "D26: lower bound must be positive, got {}",
        result.final_lb
    );
    assert_cost(result.final_lb, D26_EXPECTED_COST, 1.0, "D26");
    assert!(
        result.iterations <= 100,
        "D26: must converge within 100 iterations, got {}",
        result.iterations
    );
}

/// D26: Verify PACF order selection picks AR order 2.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d26_estimated_par2_order_selection() {
    let case_dir = Path::new("../../examples/deterministic/d26-estimated-par2");
    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");
    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let prepare_result = prepare_stochastic(
        system,
        case_dir,
        &config,
        42,
        &ScenarioSource::default(),
        None,
    )
    .expect("prepare_stochastic must succeed");

    let report = prepare_result
        .estimation_report
        .expect("estimation report must be Some");

    assert_eq!(report.entries.len(), 1, "expected 1 hydro entry");

    let (hydro_id, entry) = report.entries.iter().next().expect("entry exists");
    assert_eq!(
        entry.selected_order, 2,
        "expected AR order 2 for hydro {hydro_id}, got {}",
        entry.selected_order
    );
}

// ---------------------------------------------------------------------------
// D27: Per-stage thermal cost override
// ---------------------------------------------------------------------------

/// Expected cost for D27 (2-thermal system, stage-varying costs).
///
/// ## Case setup
///
/// - 1 bus (B0), 2 thermals, no hydro, deterministic load 100 MW.
/// - T1 (id=0): base cost 50 $/MWh, capacity 0-60 MW.
/// - T2 (id=1): base cost 80 $/MWh, capacity 0-80 MW.
/// - `thermal_bounds.parquet` overrides T1 cost at stage 1 to 120 $/MWh.
/// - 2 stages × 730 h each.
///
/// ## Expected cost derivation
///
/// Stage 0 (T1 at 50 $/MWh, T2 at 80 $/MWh — T1 dispatched first):
/// - T1 at full capacity: 60 MW × 50 $/MWh × 730 h = 2 190 000 $
/// - T2 covers residual: 40 MW × 80 $/MWh × 730 h = 2 336 000 $
/// - Stage 0 cost = 4 526 000 $
///
/// Stage 1 (T1 at 120 $/MWh via override, T2 at 80 $/MWh — T2 dispatched first):
/// - T2 at full capacity: 80 MW × 80 $/MWh × 730 h = 4 672 000 $
/// - T1 covers residual: 20 MW × 120 $/MWh × 730 h = 1 752 000 $
/// - Stage 1 cost = 6 424 000 $
///
/// Total = 4 526 000 + 6 424 000 = **10 950 000 $**
///
/// Compared to the uniform-cost baseline (T1 at 50 $/MWh in both stages):
/// - Uniform total = 2 × 4 526 000 = 9 052 000 $
/// - D27 total must be strictly greater, confirming the override is applied.
pub const D27_EXPECTED_COST: f64 = 10_950_000.0;

/// D27: Per-stage thermal cost override via `constraints/thermal_bounds.parquet`.
///
/// Uses pre-committed parquet fixtures (scenarios + constraints) to verify that
/// the LP objective coefficients use the resolved per-stage cost rather than the
/// entity base cost.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d27_per_stage_thermal_cost() {
    let case_dir = Path::new("../../examples/deterministic/d27-per-stage-thermal-cost");

    let result = run_deterministic(case_dir);

    assert_cost(result.final_lb, D27_EXPECTED_COST, 1e-4, "D27");
    assert!(
        result.iterations <= 10,
        "D27: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D27: gap={:.2e}",
        result.final_gap
    );

    // Exceeding the uniform-cost baseline confirms the override is applied and
    // reorders stage-1 dispatch (see D27_EXPECTED_COST for the baseline).
    let uniform_baseline = 9_052_000.0_f64;
    assert!(
        result.final_lb > uniform_baseline,
        "D27: per-stage cost override must increase total cost vs uniform baseline \
         ({} > {})",
        result.final_lb,
        uniform_baseline
    );
}

/// D28: Mixed-resolution case (5 weekly + 1 monthly stages).
///
/// Smoke test that verifies the full pipeline loads and trains without error
/// on a case with:
/// - Non-uniform `num_scenarios` (1 per weekly stage, 5 for the monthly stage)
/// - `season_definitions` with monthly cycle (12 seasons)
/// - External inflow scenario source
/// - `recent_observations` in initial conditions
///
/// The test only checks that training completes at least 1 iteration; no
/// expected cost is asserted here
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d28_decomp_weekly_monthly_loads_and_trains() {
    let case_dir = Path::new("../../examples/deterministic/d28-decomp-weekly-monthly");

    let result = run_deterministic(case_dir);

    assert!(
        result.iterations > 0,
        "D28: must complete at least 1 iteration"
    );
}

/// D56: a sigma=0 External LOAD deck with no `load_seasonal_stats.parquet`
/// twin realizes its own external value at stage 0, not a seasonal mean —
/// the deck-level proof of
/// `external_load_sigma_zero_reconstructs_external_value_not_seasonal_mean`
/// (`crates/cobre-sddp/src/setup/scenario_libraries.rs`). Also confirms the
/// deck as-authored validates with no `BusinessRuleViolation` (the
/// seasonal-stats-optional-under-External path).
#[test]
fn d56_external_load_reconstructs_external_value_not_seasonal_mean() {
    let case_dir = Path::new("../../examples/deterministic/d56-external-authoritative");

    cobre_io::validate_case(case_dir)
        .expect("d56 as-authored must validate with no BusinessRuleViolation");

    let (setup, _system, _result) = run_deterministic_with_setup(case_dir);

    let mean = setup.inputs.stochastic.normal().mean(0, 0);
    let std = setup.inputs.stochastic.normal().std(0, 0);
    let eta = setup
        .inputs
        .scenario_libraries
        .training
        .external_load
        .as_ref()
        .expect("d56's load scheme is External; a training library must exist")
        .eta_slice(0, 0)[0];
    let realized = (mean + std * eta).max(0.0);

    assert!(
        (realized - 123.0).abs() < 1e-9,
        "D56: stage-0 realized load must equal the external value (123.0), not a \
         seasonal mean; got {realized}"
    );
}

/// D56: a sigma=0 AR(0) External INFLOW deck with no
/// `inflow_seasonal_stats.parquet` twin loads and converges (LB == UB), proving
/// the seasonal-stats-optional-under-External path for inflow end to end.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d56_external_authoritative_loads_and_converges() {
    let case_dir = Path::new("../../examples/deterministic/d56-external-authoritative");

    let result = run_deterministic(case_dir);

    assert!(
        result.iterations >= 1,
        "D56: must complete at least 1 iteration"
    );
    assert!(
        (result.final_lb - result.final_ub).abs() < 1e-6,
        "D56: a sigma=0 deck must converge LB == UB; lb={} ub={}",
        result.final_lb,
        result.final_ub
    );
}

/// D29: weekly stages with PAR(1) noise-group sharing.
///
/// ## System
///
/// 1 bus, 1 hydro (H0), 4 weekly stages in January 2024 (all season_id=0),
/// PAR(1) with psi=0.5, OutOfSample noise, inflow_lags=true.
///
/// ## What this tests
///
/// - All 4 weekly stages share the same noise group ID (group 0).
/// - Training with noise sharing completes without error.
/// - Simulation completes with sensible costs.
///
/// This is the end-to-end verification that noise group precomputation,
/// ForwardSampler integration, opening tree integration, and setup wiring
/// compose correctly for the noise-group-sharing workflow.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d29_weekly_par_noise_sharing() {
    let case_dir = Path::new("../../examples/deterministic/d29-weekly-par-noise-sharing");

    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    // Build the training scenario source from the config so the seed and
    // OutOfSample scheme are propagated to the forward-pass noise generator.
    let training_source = config
        .training_scenario_source(&config_path)
        .expect("training_scenario_source must parse");

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let pr = prepare_stochastic(system, case_dir, &config, 42, &training_source, None)
        .expect("prepare_stochastic must succeed");
    let system = pr.system;
    let stochastic = pr.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");

    let mut setup = StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
        .expect("StudySetup must build");

    let groups = &setup.inputs.stage_data.noise_group_ids;
    assert_eq!(groups.len(), 4, "expected 4 study stages");
    assert!(
        groups.iter().all(|&g| g == groups[0]),
        "all weekly stages in the same month must share the same group ID, got {groups:?}"
    );

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "D29: expected no training error, got: {:?}",
        outcome.error
    );
    let result = outcome.result;

    assert!(
        result.iterations > 0,
        "D29: must complete at least 1 iteration"
    );
    assert!(
        result.final_lb > 0.0,
        "D29: lower bound must be positive, got {}",
        result.final_lb
    );

    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("simulation workspace pool must build");

    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);

    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

    let _local_costs = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            result.frozen_templates.as_deref(),
            &result.basis_cache,
        )
        .expect("simulation must succeed");

    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");

    assert_eq!(
        scenario_results.len(),
        1,
        "D29: expected 1 simulation scenario result"
    );
}

/// D30: monthly-to-quarterly multi-resolution stage transition.
///
/// ## System
///
/// 1 bus, 1 hydro (H0), 6 monthly stages (Jan-Jun 2024, season_id 0-5) followed
/// by 4 quarterly stages (Q3 2024 – Q2 2025, season_id 12-15). Custom SeasonMap
/// with 12 monthly + 4 quarterly season definitions. PAR(1) with psi=0.5,
/// OutOfSample noise, inflow_lags=true for all stages.
///
/// ## What this tests
///
/// - Case loads and trains without error on a Custom-cycle multi-resolution study.
/// - Training completes at least 1 iteration with a positive lower bound.
///
/// Full structural and downstream-lag-transition assertions are in the dedicated
/// `hydro_sim.rs` test file (the `multi_resolution_integration` module), which
/// verifies composition correctness including noise group IDs,
/// accumulate_downstream flags, rebuild_from_downstream, and simulation.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d30_multi_resolution_loads_and_trains() {
    let case_dir = Path::new("../../examples/deterministic/d30-multi-resolution-monthly-quarterly");

    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    // Use the config's OutOfSample training source so PAR noise is correctly seeded.
    let training_source = config
        .training_scenario_source(&config_path)
        .expect("training_scenario_source must parse");

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let pr = prepare_stochastic(system, case_dir, &config, 42, &training_source, None)
        .expect("prepare_stochastic must succeed");
    let system = pr.system;
    let stochastic = pr.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");

    let mut setup = StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
        .expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "D30: expected no training error, got: {:?}",
        outcome.error
    );
    let result = outcome.result;

    assert!(
        result.iterations > 0,
        "D30: must complete at least 1 iteration"
    );
    assert!(
        result.final_lb > 0.0,
        "D30: lower bound must be positive, got {}",
        result.final_lb
    );
}

/// The frozen-template simulation path must produce bit-for-bit identical
/// per-scenario costs to the legacy fallback path (rel error == 0.0, within
/// 1e-12), confirming `freeze_rows_into_template` builds a mathematically
/// equivalent LP to `load_model + add_rows`. Trained on D01, which runs >= 2
/// iterations so `frozen_templates` is `Some`.
#[test]
fn frozen_vs_fallback_simulation_costs_are_identical() {
    let case_dir = Path::new("../../examples/deterministic/d01-thermal-dispatch");
    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let pr = prepare_stochastic(
        system,
        case_dir,
        &config,
        42,
        &ScenarioSource::default(),
        None,
    )
    .expect("prepare_stochastic must succeed");
    let system = pr.system;
    let stochastic = pr.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");

    let mut config_with_sim = config.clone();
    config_with_sim.simulation.enabled = true;
    config_with_sim.simulation.selection = Some(SimulationSelection::Sampled { num_scenarios: 4 });

    let mut setup = StudySetup::new(
        &system,
        &config_with_sim,
        stochastic,
        hydro_models,
        Vec::new(),
    )
    .expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");
    let training_result = outcome.result;

    assert!(
        training_result.frozen_templates.is_some(),
        "D01 training must produce frozen templates (requires >= 2 iterations)"
    );

    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("simulation workspace pool must build");

    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (tx_frozen, rx_frozen) = mpsc::sync_channel(io_capacity);
    let drain_frozen = std::thread::spawn(move || rx_frozen.into_iter().collect::<Vec<_>>());

    let frozen_run = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &tx_frozen,
            None,
            training_result.frozen_templates.as_deref(),
            &training_result.basis_cache,
        )
        .expect("frozen-path simulate must return Ok");
    drop(tx_frozen);
    drop(drain_frozen.join().expect("drain thread must not panic"));

    let (tx_fallback, rx_fallback) = mpsc::sync_channel(io_capacity);
    let drain_fallback = std::thread::spawn(move || rx_fallback.into_iter().collect::<Vec<_>>());

    let fallback_run = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &tx_fallback,
            None,
            None, // force legacy path
            &training_result.basis_cache,
        )
        .expect("fallback-path simulate must return Ok");
    drop(tx_fallback);
    drop(drain_fallback.join().expect("drain thread must not panic"));

    assert_eq!(
        frozen_run.costs.len(),
        fallback_run.costs.len(),
        "both runs must return the same number of scenarios"
    );

    for ((b_id, b_cost, _), (f_id, f_cost, _)) in
        frozen_run.costs.iter().zip(fallback_run.costs.iter())
    {
        assert_eq!(b_id, f_id, "scenario IDs must match between runs");
        let rel_err = if b_cost.abs() > 1e-10 {
            (b_cost - f_cost).abs() / b_cost.abs()
        } else {
            (b_cost - f_cost).abs()
        };
        assert!(
            rel_err < 1e-12,
            "scenario {b_id}: frozen cost {b_cost} != fallback cost {f_cost} (rel_err={rel_err})"
        );
    }
}

/// D43: storage-only (lag-dropped) cut convergence.
///
/// ## System
///
/// 1 bus, 1 hydro (H0, constant productivity 0.8 MW/(m3/s), 0–500 hm3 storage),
/// 1 thermal (300 MW @ 100 $/MWh), 4 monthly stages, load 220 MW constant.
/// PACF with `max_order=2` fits a PAR(2) model (global `n_state = N*(1+L) = 3`).
///
/// ## Storage-only stage
///
/// Stage 2 sets `state_variables.inflow_lags = false`. Pool `t` is sized by stage
/// `t+1`'s config, so stage 2's storage-only config sizes **pool 1** at cut
/// dimension `N = 1` (the AR lags are dropped from that pool's cuts) while the
/// lag-enabled pools stay at `N*(1+L) = 3`. The reduced cut, rendered through the
/// per-pool outgoing projection, touches only the storage column.
///
/// ## Convergence
///
/// The lag-dropped cut is an approximation change; this case proves it still
/// closes the LB/UB gap. The run terminates on `bound_stalling` (the LB
/// stabilizes — it does not plateau below the UB), with LB == UB to machine
/// precision.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d43_storage_only_cut_converges() {
    let case_dir = Path::new("../../examples/deterministic/d43-storage-only-cut");

    // Pool dimensions: stage 2's storage-only config sizes pool 1 down to N,
    // while the lag-enabled pools carry storage + lags.
    let config = cobre_io::parse_config(&case_dir.join("config.json")).expect("config must parse");
    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");
    let pr = prepare_stochastic(
        system,
        case_dir,
        &config,
        42,
        &ScenarioSource::default(),
        None,
    )
    .expect("prepare_stochastic must succeed");
    let system = pr.system;
    let stochastic = pr.stochastic;
    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");
    let mut setup = build_setup_for_case(case_dir, &config, &system, stochastic, hydro_models);
    let n_hydros = setup.stage_state().hydro_count;
    let global_n_state = setup.stage_state().n_state;
    assert!(
        global_n_state > n_hydros,
        "D43: PACF must fit PAR(p>0) so the lag drop is non-trivial (n_state={global_n_state}, N={n_hydros})",
    );
    // ONLY pool 1 is reduced (sized by stage 2's storage-only config); every
    // sibling stays at the full global dimension. Pinning all four catches both an
    // under-reduction (pool 1 not dropped) and an over-eager reduction (a sibling
    // wrongly dropped).
    assert_eq!(
        setup.fcf.pools[1].state_dimension, n_hydros,
        "D43: storage-only pool 1 must have cut dimension N (lags dropped)",
    );
    for t in [0usize, 2, 3] {
        assert_eq!(
            setup.fcf.pools[t].state_dimension, global_n_state,
            "D43: lag-enabled pool {t} must stay at the full global dimension",
        );
    }

    // Convergence: the storage-only cut closes the LB/UB gap and the run stops on
    // bound stalling (not the iteration cap), proving the LB does not plateau.
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "D43: expected no training error");
    let result = outcome.result;
    assert_eq!(
        result.reason, "bound_stalling",
        "D43: must converge via bound_stalling, not exhaust the iteration cap (reason={})",
        result.reason,
    );
    assert!(
        result.final_gap < 1e-6,
        "D43: LB/UB gap must close (gap={})",
        result.final_gap,
    );
    assert_cost(result.final_lb, 11_658_487.253_236_46, 1.0, "D43");

    // The cut(s) stored in the reduced pool carry the reduced length: a cut on a
    // storage-only pool has exactly N coefficients, not the global n_state. This is
    // what discriminates the per-stage-dimension-aware storage from a revert to
    // full-dimension (the bound alone is neutral — an exact reduction stores
    // numerically the same cut a full pool would).
    let reduced_pool = &setup.fcf.pools[1];
    assert!(
        reduced_pool.active_count() > 0,
        "D43: the reduced pool must hold at least one trained cut to inspect",
    );
    for (slot, _intercept, coefficients) in reduced_pool.active_cuts() {
        assert_eq!(
            coefficients.len(),
            n_hydros,
            "D43: stored cut at pool-1 slot {slot} must have reduced length N={n_hydros}",
        );
    }
}

/// D43's stage 0 starts exactly on a month boundary, so the windowed accumulator
/// seed is inert (`accum = weight = 0.0`, per `derive_inflow_seeds`) and the
/// windowed cast collapses to the pre-windowing month lookup — an observation-free
/// case that must reproduce its pre-windowing `final_lb` bit-for-bit. The `to_bits`
/// golden is HiGHS-only (`#[cfg]`-gated): bit-exactness is backend-specific — CLP's
/// simplex reaches a different-but-valid vertex — so removing the gate breaks the
/// CLP suite. D43's `assert_cost` above tolerance-pins the same value for both
/// backends; this adds the bit-for-bit no-drift gate on the golden HiGHS path.
#[cfg(feature = "highs")]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn test_observation_free_case_bit_exact_pre_epic() {
    let case_dir = Path::new("../../examples/deterministic/d43-storage-only-cut");
    let result = run_deterministic(case_dir);

    assert_eq!(
        result.final_lb.to_bits(),
        0x4166_3c9e_e81a_8359u64,
        "D43 final_lb must reproduce its pre-windowing value bit-for-bit: got {} ({:#018x})",
        result.final_lb,
        result.final_lb.to_bits()
    );
}

/// D06 is the variable-head FPHA case carrying the "FPHA uses average storage"
/// contract (`-gammaV/2` on BOTH storage columns) — the FPHA plane rows are
/// where the per-cell apportionment concentrates, so a no-drift pin here covers
/// the partitioned production-row path the synthetic `cell_partition_gates`
/// fixtures cannot: a real fitted hyperplane set on a real single-bus case. The
/// `to_bits` golden is HiGHS-only (`#[cfg]`-gated): bit-exactness is
/// backend-specific — CLP's simplex reaches a different-but-valid vertex — so
/// removing the gate breaks the CLP suite. `d06_fpha_variable_head` above
/// tolerance-pins the same value for both backends; this adds the bit-for-bit
/// no-drift gate on the golden HiGHS path.
#[cfg(feature = "highs")]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn test_fpha_variable_head_case_bit_exact() {
    let case_dir = Path::new("../../examples/deterministic/d06-fpha-variable-head");

    // rho_eq is irrelevant to D06 economics (see d06_fpha_variable_head above);
    // pinned to the same neutral value so this golden's setup matches exactly.
    let case = copy_case_dir(case_dir);
    write_energy_productivity_override(
        &case.path().join("system/hydro_energy_productivity.parquet"),
        0,
        1.0,
    );

    let result = run_deterministic(case.path());

    assert_eq!(
        result.final_lb.to_bits(),
        0x4148_da6e_907f_6e5cu64,
        "D06 final_lb must reproduce bit-for-bit: got {} ({:#018x})",
        result.final_lb,
        result.final_lb.to_bits()
    );
}

/// D44: sub-stage-delay bucket dual and per-stage thermal split.
///
/// ## System
///
/// Cascade `U -> J`, 2 stages of 720 h each, one block, parallel mode. `U` (100
/// hm3 initial storage) declares `travel_time_hours = 360` on its arc to `J`
/// (run-of-river, 0 hm3 storage); both carry toy productivity 1 MWh/hm3 (`U`'s
/// `productivity_mw_per_m3s = 0.0036 = M3S_TO_HM3`, so `generation_mw * hours`
/// equals the turbined volume in hm3 regardless of stage length). One thermal at
/// 10 $/MWh serves 200 MWh/stage demand.
///
/// `k_0 = k_1 = 1/2` (360 h travel time over a 720 h stage), so `L = 1`: one
/// bucket dimension for `J`.
///
/// ## Hand-derived optimum
///
/// Release everything at stage 0 (`x = 100` hm3): the same-stage share (`k_0 =
/// 50` hm3) reaches `J` immediately, and the delayed share (`k_1 = 50` hm3)
/// matures onto `J`'s balance row at stage 1 via the bucket. Any water held back
/// to stage 1 would lose its own `k_1` half past the horizon (no terminal
/// credit) with no compensating gain, so releasing everything at stage 0 weakly
/// dominates every other split.
///
/// Hydro generation: `U` 100 MWh + `J` 50 MWh = 150 MWh at stage 0 (thermal
/// covers the remaining 50 MWh of 200 MWh demand); `J` 50 MWh only at stage 1
/// (thermal covers the remaining 150 MWh). Total cost = (50 + 150) MWh x 10
/// $/MWh = **2000 $**.
///
/// ## Why total cost alone cannot discriminate
///
/// A fold implementation (no bucket at all, the crossing `k_1` half absorbed
/// same-stage instead of delayed) also reaches total cost 2000 — releasing 100
/// hm3 at stage 0 generates 100 MWh at `J` immediately either way, just shifted
/// one stage earlier. The bucket subgradient (a fold has no bucket column to
/// read; a wrong-sign coefficient flips its sign) and the per-stage thermal
/// split (50/150 vs. a folded 100/100) are what a fold gets wrong, so they are
/// asserted explicitly below instead of relying on the fold-blind total.
#[test]
fn d44_travel_time_substage_transit_bucket_dual() {
    // The LP builder divides every non-theta objective coefficient by this
    // factor; `duals_extraction`'s `rc / col_scale` unscaling leaves it in, so
    // a stored cut coefficient must be multiplied back to read real dollars
    // (mirrors `extraction.rs`'s `water_value = dual * COST_SCALE_FACTOR`).
    const COST_SCALE_FACTOR: f64 = 1_000_000.0;
    const HOURS_PER_STAGE: f64 = 720.0;
    const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;
    const TOL: f64 = 1e-6;

    let case_dir = Path::new("../../examples/deterministic/d44-travel-time-substage");

    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");
    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");
    let pr = prepare_stochastic(
        system,
        case_dir,
        &config,
        42,
        &ScenarioSource::default(),
        None,
    )
    .expect("prepare_stochastic must succeed");
    let system = pr.system;
    let stochastic = pr.stochastic;
    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");
    let mut setup = build_setup_for_case(case_dir, &config, &system, stochastic, hydro_models);

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "D44: expected no training error");
    let result = outcome.result;

    assert!(
        result.final_gap.abs() < 1e-6,
        "D44: gap={:.2e}",
        result.final_gap
    );
    assert_cost(result.final_lb, 2000.0, TOL, "D44");

    // Bucket subgradient: rc/col_scale on the incoming bucket column, stored
    // (undivided by COST_SCALE_FACTOR) as pool-0's cut coefficient
    // (StateSpace::state_to_lp_incoming_column's explicit bucket arm resolves
    // the pin; the cut coefficient index is the STATE index, identity to the
    // outgoing column per `transit_buckets_out`).
    //
    // J's own storage entering stage 1 (a non-degenerate corroborating check:
    // both it and the bucket deliver into J's same stage-1 balance row, so
    // both must price at the same -10 $/hm3 marginal value). U's own storage
    // entering stage 1 is NOT asserted here — U fully drains at stage 0, so
    // every term on its stage-1 balance row is forced to exactly zero and its
    // reduced cost is basis-dependent, not a robust, backend-agnostic pin.
    let state = setup.stage_state();
    assert_eq!(
        state.n_buckets, 1,
        "D44: exactly one bucket dimension (single arc, L=1)"
    );
    let transit_bucket_idx = state.transit_buckets_out.start;
    let j_canonical_idx = system
        .hydros()
        .iter()
        .position(|h| h.id == EntityId::from(1))
        .expect("D44: J (hydro id 1) must exist in the canonical hydro order");
    let storage_j_idx = state.storage.start + j_canonical_idx;

    let pool0 = &setup.fcf.pools[0];
    assert!(
        pool0.active_count() > 0,
        "D44: pool 0 must hold at least one trained cut to inspect"
    );
    for (slot, _intercept, coefficients) in pool0.active_cuts() {
        let transit_bucket_dual = coefficients[transit_bucket_idx] * COST_SCALE_FACTOR;
        assert!(
            (transit_bucket_dual - (-10.0)).abs() < TOL,
            "D44: bucket subgradient at pool-0 slot {slot} must be exactly -10 $/hm3 \
             (a wrong-sign coefficient would give +10); got {transit_bucket_dual}"
        );
        let storage_j_dual = coefficients[storage_j_idx] * COST_SCALE_FACTOR;
        assert!(
            (storage_j_dual - (-10.0)).abs() < TOL,
            "D44: J's own storage water value at pool-0 slot {slot} must also be -10 $/hm3 \
             (same stage-1 balance row as the bucket delivery); got {storage_j_dual}"
        );
    }

    // Per-stage thermal split + delivery split: simulate the trained policy
    // (config.json already enables simulation with num_scenarios = 1).
    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("D44: simulation workspace pool must build");
    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());
    let _local_costs = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            result.frozen_templates.as_deref(),
            &result.basis_cache,
        )
        .expect("D44: simulate must return Ok");
    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");

    assert_eq!(scenario_results.len(), 1, "D44: exactly one scenario");
    let scenario = &scenario_results[0];
    assert_eq!(scenario.stages.len(), 2, "D44: exactly two stages");

    let thermal_mwh = |stage_id: usize| -> f64 {
        scenario.stages[stage_id]
            .thermals
            .iter()
            .find(|t| t.thermal_id == 0)
            .unwrap_or_else(|| panic!("D44: T0 missing from stage {stage_id}"))
            .generation_mw
            * HOURS_PER_STAGE
    };
    assert!(
        (thermal_mwh(0) - 50.0).abs() < TOL,
        "D44: stage 0 thermal generation must be 50 MWh, got {}",
        thermal_mwh(0)
    );
    assert!(
        (thermal_mwh(1) - 150.0).abs() < TOL,
        "D44: stage 1 thermal generation must be 150 MWh, got {}",
        thermal_mwh(1)
    );

    // Delivery split: stage 0's same-stage k_0 share arrives as J's own
    // turbined volume (no bucket record for a same-stage deposit); stage 1's
    // k_1 share is the bucket's maturing delivery (`delayed_arrival_hm3`).
    let j_stage0 = scenario.stages[0]
        .hydros
        .iter()
        .find(|h| h.hydro_id == 1)
        .expect("D44: J missing from stage 0");
    let j_stage0_volume_hm3 = j_stage0.turbined_m3s * M3S_TO_HM3 * HOURS_PER_STAGE;
    assert!(
        (j_stage0_volume_hm3 - 50.0).abs() < TOL,
        "D44: J must receive 50 hm3 at stage 0 (same-stage k_0), got {j_stage0_volume_hm3}"
    );

    let transit_bucket_stage1 = scenario.stages[1]
        .transit_buckets
        .iter()
        .find(|b| b.hydro_id == 1)
        .expect("D44: J's bucket record missing from stage 1");
    assert!(
        (transit_bucket_stage1.delayed_arrival_hm3 - 50.0).abs() < TOL,
        "D44: J must receive 50 hm3 at stage 1 via the bucket, got {}",
        transit_bucket_stage1.delayed_arrival_hm3
    );
}

/// Pads `stage_hours` with copies of its trailing duration until the total
/// covers `travel_time_hours` — mirrors the production padding
/// (`bucket_topology::extend_for_resolution`) that lets
/// `resolve_spread` resolve an anchor whose own remaining calendar runs out
/// before its arrival window closes; without it `resolve_spread`'s own
/// `Σ_d k_d = 1` debug_assert panics instead of resolving a real depth.
fn pad_calendar_for_resolution(stage_hours: &[f64], travel_time_hours: f64) -> Vec<f64> {
    let last = *stage_hours
        .last()
        .expect("D45: calendar must have at least one stage");
    let mut padded = stage_hours.to_vec();
    let mut padded_hours = 0.0;
    while padded_hours < travel_time_hours {
        padded.push(last);
        padded_hours += last;
    }
    padded
}

/// D45: mixed-calendar depth-3 counterexample and end-to-end water
/// conservation.
///
/// ## Calendar
///
/// One 720 h monthly stage (0) then three 168 h weekly stages (1, 2, 3). The
/// `U -> J` arc declares `travel_time_hours = 360`. At the monthly anchor the
/// arrival window `[360, 1080)` overlaps four periods — `k_0 = 1/2`
/// (same-stage), `k_1 = k_2 = 7/30`, `k_3 = 1/30` — so `L = 3`, not the
/// closed-form ceiling `ceil(360/720) = 1`, which would drop `8/30` of the
/// release (the k-factor conservation contract). Four stages is
/// exactly deep enough for the monthly anchor's depth-3 delivery to land
/// fully within the horizon (target stage `0 + 3 = 3`, the last one), so this
/// case's release delivers in full with zero horizon drop — the depth-3
/// mechanics are directly observable rather than swallowed by the terminal
/// drop.
///
/// ## What each assertion pins
///
/// 1. `resolve_spread` at the monthly anchor reproduces the hand-derived
///    `stage_weights` and `stage_reach == 3` directly (a ceiling-form
///    regression would report 1).
/// 2. End-to-end conservation: summed over every stage, `U`'s released
///    volume (turbined + spilled) equals `J`'s received volume (turbined +
///    spilled — `J` has zero storage, so its own release equals whatever it
///    receives that stage, same-stage share plus any bucket maturity) plus
///    the horizon drop (whatever is still in transit, unconsumed, at the
///    terminal stage — a bucket slot there targets a stage past the
///    horizon).
/// 3. Global-max sizing with per-stage masking: the bucket block's global
///    depth (`state.n_buckets`) is the deepest per-anchor reach; the
///    documented per-stage cap (the "terminal credit deferred" contract:
///    `active.min(n_stages - 1 - stage)`) shrinks that reach stage by stage
///    down to zero at the terminal stage.
#[test]
fn d45_travel_time_mixed_calendar_conservation() {
    const K_TOL: f64 = 1e-9;
    const CONSERVATION_TOL: f64 = 1e-6;
    const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;
    const TRAVEL_TIME_HOURS: f64 = 360.0;
    const N_STAGES: usize = 4;

    let stage_hours = [720.0, 168.0, 168.0, 168.0];

    let monthly = resolve_spread(TRAVEL_TIME_HOURS, 0, &stage_hours, None);
    assert_eq!(
        monthly.stage_reach, 3,
        "D45: the monthly anchor must resolve to depth 3, not the closed-form \
         ceiling ceil(360/720) = 1 (which would drop 8/30 of the release)"
    );
    let expected_k = [0.5, 7.0 / 30.0, 7.0 / 30.0, 1.0 / 30.0];
    assert_eq!(
        monthly.stage_weights.len(),
        expected_k.len(),
        "D45: stage_weights must have 4 entries"
    );
    for (lag, (&actual, &expected)) in monthly
        .stage_weights
        .iter()
        .zip(expected_k.iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() < K_TOL,
            "D45: stage_weights[{lag}] = {actual}, expected {expected}"
        );
    }

    let case_dir = Path::new("../../examples/deterministic/d45-travel-time-mixed-calendar");
    let (setup, _system, result) = run_deterministic_with_setup(case_dir);
    assert!(
        result.final_gap.abs() < 1e-6,
        "D45: gap={:.2e}",
        result.final_gap
    );

    let state = setup.stage_state();
    assert_eq!(
        state.n_buckets, 3,
        "D45: the global bucket depth must be the deepest per-anchor reach (3)"
    );
    let padded = pad_calendar_for_resolution(&stage_hours, TRAVEL_TIME_HOURS);
    let own_depths: Vec<usize> = (0..N_STAGES)
        .map(|stage| resolve_spread(TRAVEL_TIME_HOURS, stage, &padded, None).stage_reach)
        .collect();
    let capped: Vec<usize> = own_depths
        .iter()
        .enumerate()
        .map(|(stage, &depth)| depth.min(N_STAGES - 1 - stage))
        .collect();
    assert_eq!(
        capped,
        vec![3, 2, 1, 0],
        "D45: per-stage active range must shrink toward the horizon end \
         (active.min(n_stages - 1 - stage)), got {own_depths:?} capped to {capped:?}"
    );
    assert_eq!(
        capped[0], state.n_buckets,
        "D45: stage 0 must reach the full global depth"
    );
    assert!(
        capped[N_STAGES - 1] < state.n_buckets,
        "D45: the terminal stage's active range must be strictly shorter than \
         the global depth (masked), got {}",
        capped[N_STAGES - 1]
    );

    let comm = StubComm;
    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("D45: simulation workspace pool must build");
    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());
    let _local_costs = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            result.frozen_templates.as_deref(),
            &result.basis_cache,
        )
        .expect("D45: simulate must return Ok");
    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");

    assert_eq!(scenario_results.len(), 1, "D45: exactly one scenario");
    let scenario = &scenario_results[0];
    assert_eq!(scenario.stages.len(), N_STAGES, "D45: exactly four stages");

    let mut released_hm3 = 0.0_f64;
    let mut delivered_hm3 = 0.0_f64;
    for stage_result in &scenario.stages {
        let hours = stage_hours[stage_result.stage_id as usize];
        let u = stage_result
            .hydros
            .iter()
            .find(|h| h.hydro_id == 0)
            .unwrap_or_else(|| panic!("D45: U missing from stage {}", stage_result.stage_id));
        let j = stage_result
            .hydros
            .iter()
            .find(|h| h.hydro_id == 1)
            .unwrap_or_else(|| panic!("D45: J missing from stage {}", stage_result.stage_id));
        released_hm3 += (u.turbined_m3s + u.spillage_m3s) * M3S_TO_HM3 * hours;
        // J has zero storage, so its own release equals its total inflow that
        // stage: the same-stage k_0 share plus any bucket maturity (both
        // enter J's water-balance row; there is no separate output field for
        // the same-stage share alone).
        delivered_hm3 += (j.turbined_m3s + j.spillage_m3s) * M3S_TO_HM3 * hours;
    }

    let terminal_stage = scenario.stages.last().expect("D45: at least one stage");
    let terminal_transit_buckets: Vec<_> = terminal_stage
        .transit_buckets
        .iter()
        .filter(|b| b.hydro_id == 1)
        .collect();
    assert_eq!(
        terminal_transit_buckets.len(),
        3,
        "D45: J must carry all 3 globally-sized bucket lag slots even at the terminal stage"
    );
    let horizon_drop_hm3: f64 = terminal_transit_buckets
        .iter()
        .map(|b| b.in_transit_volume_hm3)
        .sum();

    let residual = released_hm3 - (delivered_hm3 + horizon_drop_hm3);
    assert!(
        residual.abs() < CONSERVATION_TOL,
        "D45: conservation violated: released={released_hm3}, delivered={delivered_hm3}, \
         horizon_drop={horizon_drop_hm3}, residual={residual}"
    );
}

/// D46: chronological block-resolved attribution. Cascade `U -> J`,
/// `travel_time_hours = 250` on `U`'s arc; stage 0 is 720 h resolved into 3
/// chronological blocks of 240 h each, stage 1 a single 720 h receiving
/// stage. The block-table / K=1-parity / state-dimension pins live in
/// [`chronological_attribution`]; this proves the same cascade also trains
/// and converges as a real file-based case.
#[test]
fn d46_travel_time_chronological_converges() {
    let case_dir = Path::new("../../examples/deterministic/d46-travel-time-chronological");
    let result = run_deterministic(case_dir);
    assert!(
        result.final_gap.abs() < 1e-6,
        "D46: gap={:.2e}",
        result.final_gap
    );
}

/// D47: confluence aggregation. Two upstreams `U1`
/// (`travel_time_hours = 360`) and `U2` (`travel_time_hours = 1080`) both feed
/// downstream `J` (run-of-river), 3 uniform 720 h stages, parallel mode.
///
/// ## Hand-derived per-arc schedules
///
/// `U1`'s arrival window `[360, 1080)` over 720 h stages gives `k = [1/2, 1/2]`
/// (depth 1): half same-stage, half maturing one stage later. `U2`'s window
/// `[1080, 1800)` gives `k = [0, 1/2, 1/2]` (depth 2): nothing same-stage, half
/// maturing one stage later, half two stages later. `L_j = max(1, 2) = 2`, so
/// `J`'s bucket block must aggregate BOTH arcs into ONE depth-2 block, not one
/// block per arc (a per-arc block would give `n_buckets = 1 + 2 = 3`).
///
/// Releasing everything at stage 0 (weakly optimal — deferring any release
/// risks losing its own deepest share past the horizon, same argument as the
/// two-hydro case) puts every arc's full 100 hm3 through its own `k`-schedule:
/// `J` receives `U1`'s 50 hm3 same-stage share at stage 0; `U1`'s 50 hm3
/// lag-1 share AND `U2`'s 50 hm3 lag-1 share both mature at stage 1 (100 hm3
/// total — the confluence sum); `U2`'s 50 hm3 lag-2 share matures at stage 2.
/// A single-term transition (`Σ_i k_{d,i} D_i^{t−d}`) would mis-time `U2`'s
/// deeper mass instead of this schedule.
#[test]
fn d47_travel_time_confluence_aggregation() {
    const TOL: f64 = 1e-6;
    const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;
    const HOURS_PER_STAGE: f64 = 720.0;

    let stage_hours = [720.0, 720.0, 720.0];

    let u1 = resolve_spread(360.0, 0, &stage_hours, None);
    assert_eq!(u1.stage_reach, 1, "U1: depth must be 1");
    for (lag, (&actual, &expected)) in u1.stage_weights.iter().zip([0.5, 0.5].iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-9,
            "U1: stage_weights[{lag}] = {actual}, expected {expected}"
        );
    }

    let u2 = resolve_spread(1080.0, 0, &stage_hours, None);
    assert_eq!(
        u2.stage_reach, 2,
        "U2: depth must be 2 (arrives at lags 1 and 2)"
    );
    for (lag, (&actual, &expected)) in u2
        .stage_weights
        .iter()
        .zip([0.0, 0.5, 0.5].iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() < 1e-9,
            "U2: stage_weights[{lag}] = {actual}, expected {expected}"
        );
    }

    let case_dir = Path::new("../../examples/deterministic/d47-travel-time-confluence");
    let (setup, system, result) = run_deterministic_with_setup(case_dir);
    assert!(
        result.final_gap.abs() < 1e-6,
        "D47: gap={:.2e}",
        result.final_gap
    );

    let state = setup.stage_state();
    assert_eq!(
        state.n_buckets, 2,
        "D47: n_buckets must be 2 (one merged block of depth max(1,2)); a \
         one-block-per-arc regression would give 1 + 2 = 3"
    );
    let j_canonical_idx = system
        .hydros()
        .iter()
        .position(|h| h.id == EntityId::from(2))
        .expect("D47: J (hydro id 2) must exist in the canonical hydro order");
    assert_eq!(
        state.transit_bucket_column_order,
        vec![
            (HydroSys::new(j_canonical_idx), 1),
            (HydroSys::new(j_canonical_idx), 2)
        ],
        "D47: both bucket slots must belong to J's single block (same plant \
         index, lags 1 and 2), never a separate block per upstream arc"
    );

    let comm = StubComm;
    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("D47: simulation workspace pool must build");
    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());
    let _local_costs = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            result.frozen_templates.as_deref(),
            &result.basis_cache,
        )
        .expect("D47: simulate must return Ok");
    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");

    assert_eq!(scenario_results.len(), 1, "D47: exactly one scenario");
    let scenario = &scenario_results[0];
    assert_eq!(scenario.stages.len(), 3, "D47: exactly three stages");

    let expected_arrivals_hm3 = [50.0, 100.0, 50.0];
    let mut released_hm3 = 0.0_f64;
    let mut delivered_hm3 = 0.0_f64;
    for stage_result in &scenario.stages {
        let stage_id = stage_result.stage_id as usize;
        let u1_hydro = stage_result
            .hydros
            .iter()
            .find(|h| h.hydro_id == 0)
            .unwrap_or_else(|| panic!("D47: U1 missing from stage {stage_id}"));
        let u2_hydro = stage_result
            .hydros
            .iter()
            .find(|h| h.hydro_id == 1)
            .unwrap_or_else(|| panic!("D47: U2 missing from stage {stage_id}"));
        let j_hydro = stage_result
            .hydros
            .iter()
            .find(|h| h.hydro_id == 2)
            .unwrap_or_else(|| panic!("D47: J missing from stage {stage_id}"));

        released_hm3 +=
            (u1_hydro.turbined_m3s + u1_hydro.spillage_m3s) * M3S_TO_HM3 * HOURS_PER_STAGE;
        released_hm3 +=
            (u2_hydro.turbined_m3s + u2_hydro.spillage_m3s) * M3S_TO_HM3 * HOURS_PER_STAGE;

        // J has zero storage, so its own release equals its total arrival that
        // stage: the summed same-stage shares plus the summed bucket maturities
        // from BOTH arcs (there is no separate output field splitting them).
        let j_arrival_hm3 =
            (j_hydro.turbined_m3s + j_hydro.spillage_m3s) * M3S_TO_HM3 * HOURS_PER_STAGE;
        delivered_hm3 += j_arrival_hm3;
        assert!(
            (j_arrival_hm3 - expected_arrivals_hm3[stage_id]).abs() < TOL,
            "D47: stage {stage_id} arrival at J must be {} hm3 (the confluence \
             sum of both arcs' hand-derived shares), got {j_arrival_hm3}",
            expected_arrivals_hm3[stage_id]
        );
    }

    let terminal_stage = scenario.stages.last().expect("D47: at least one stage");
    let horizon_drop_hm3: f64 = terminal_stage
        .transit_buckets
        .iter()
        .filter(|b| b.hydro_id == 2)
        .map(|b| b.in_transit_volume_hm3)
        .sum();
    let residual = released_hm3 - (delivered_hm3 + horizon_drop_hm3);
    assert!(
        residual.abs() < TOL,
        "D47: conservation violated over both arcs: released={released_hm3}, \
         delivered={delivered_hm3}, horizon_drop={horizon_drop_hm3}, residual={residual}"
    );
}

/// D48: a non-zero windowed IC-defluence seed drives the stage-0/1 in-transit
/// buckets and measurably lowers the training cost — the coverage capstone for
/// the windowed-seed derivation.
///
/// ## System
///
/// Cascade `U -> J`, 3 weekly (168 h) stages, one block, parallel mode. `U`
/// (id 0) declares `travel_time_hours = 336` on its arc to `J` (id 1,
/// run-of-river). BOTH hydros carry zero reservoir capacity and zero natural
/// inflow, so `U` releases nothing in-study: the ONLY water in the system is
/// the pre-study defluence already in transit, seeded into `J`'s stage-0
/// buckets from `U`'s `past_defluences`. Both carry toy productivity 1 MWh/hm3
/// (`productivity = 0.0036 = M3S_TO_HM3`, so a turbined hm3 generates 1 MWh
/// regardless of stage length). One thermal at 10 $/MWh serves 200 MWh/stage.
///
/// ## Windowed seed derivation (window -> k -> volume -> cost)
///
/// `U`'s single `past_defluences` window `[2023-12-18, 2024-01-01) = [start_0 -
/// 336h, start_0)` at 100 m3/s covers exactly the arc's in-transit span
/// `[start_0 - t_v, start_0)` (so it passes the io coverage gate). The seed
/// (`build_initial_transit_bucket_state`) unrolls it through
/// `StageCalendar::hour_window_shares`:
///   - `e_off = start_0 - end_date = 0`, `width = end_date - start_date = 336`.
///   - `hour_window_shares` sets `window_start = t_v - e_off - width = 0` and overlaps
///     `[0, 336)` against the weekly stage clock `[168, 168, 168]` -> `[168,
///     168]` -> `k = [1/2, 1/2]`.
///   - `volume = width * M3S_TO_HM3 * value = 336 * 0.0036 * 100 = 120.96 hm3`.
///   - `seed = k * volume = [60.48, 60.48]` hm3. Bucket lag 1 (`b_1^in`)
///     delivers 60.48 hm3 at stage 0; lag 2 shifts to lag 1 across the stage-0
///     ring and delivers 60.48 hm3 at stage 1. Both land inside the horizon
///     (stage 2 is terminal, receives nothing), so no share is dropped.
///
/// ## Hand-derived optimum
///
/// `J` turbines each maturing arrival (1 hm3 -> 1 MWh), displacing thermal:
///   - Stage 0: J 60.48 MWh -> thermal 200 - 60.48 = 139.52 MWh -> 1395.20 $.
///   - Stage 1: J 60.48 MWh -> thermal 139.52 MWh -> 1395.20 $.
///   - Stage 2: no arrival -> thermal 200 MWh -> 2000.00 $.
///   - Total = **4790.40 $**.
///
/// ## Zero-seed contrast (the non-zero seed is load-bearing)
///
/// With the window value set to 0.0 the seed is all-zero, no water reaches J,
/// and every stage is full thermal: `3 * 200 * 10 = 6000.00 $ != 4790.40`. The
/// two costs differ by exactly the delivered energy valued at the thermal price
/// (`120.96 MWh * 10 $/MWh = 1209.60 $`), so the `final_lb == 4790.40`
/// assertion fails the instant the seed is dropped or mis-derived — this is the
/// computed-two-ways check that keeps the case from being a zero-seed tautology.
#[test]
fn d48_travel_time_ic_seed_windowed_defluence_cost() {
    const HOURS_PER_STAGE: f64 = 168.0;
    const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;
    const T_V: f64 = 336.0;
    const WINDOW_WIDTH: f64 = 336.0;
    const VALUE_M3S: f64 = 100.0;
    const DEMAND_MWH: f64 = 200.0;
    const THERMAL_COST: f64 = 10.0;
    const N_STAGES: usize = 3;
    const TOL: f64 = 1e-6;

    // window -> k: recompute the IC-anchor split through the same overlap
    // primitive the seed uses, so the derivation is machine-checked, not a
    // hand-copied constant. e_off = 0 (the window ends at start_0), so
    // window_start = t_v - e_off - width = 0.
    let e_off = 0.0_f64;
    let window_start = T_V - e_off - WINDOW_WIDTH;
    let overlaps =
        cobre_core::window_period_overlaps(window_start, WINDOW_WIDTH, &[168.0, 168.0, 168.0]);
    let k: Vec<f64> = overlaps.iter().map(|o| o / WINDOW_WIDTH).collect();
    assert_eq!(
        k.len(),
        2,
        "D48: the window must split across two study stages"
    );
    for (lag, &kd) in k.iter().enumerate() {
        assert!(
            (kd - 0.5).abs() < TOL,
            "D48: k[{lag}] must be 1/2 (even split over two 168 h weeks), got {kd}"
        );
    }

    // window -> volume -> seed.
    let volume_hm3 = WINDOW_WIDTH * M3S_TO_HM3 * VALUE_M3S;
    assert!(
        (volume_hm3 - 120.96).abs() < TOL,
        "D48: volume={volume_hm3}"
    );
    let seed_hm3: Vec<f64> = k.iter().map(|kd| kd * volume_hm3).collect(); // [60.48, 60.48]

    // seed -> delivered MWh -> cost (1 hm3 turbined at J -> 1 MWh; the seed
    // delivers seed[d] hm3 at study stage d, and nothing at the terminal stage).
    let mut delivered_mwh = vec![0.0_f64; N_STAGES];
    delivered_mwh[0] = seed_hm3[0];
    delivered_mwh[1] = seed_hm3[1];
    let mut expected_lb = 0.0_f64;
    for &delivered in &delivered_mwh {
        expected_lb += (DEMAND_MWH - delivered) * THERMAL_COST;
    }
    assert!(
        (expected_lb - 4790.4).abs() < TOL,
        "D48: hand-derived optimum must be 4790.40, got {expected_lb}"
    );

    // Zero-seed contrast (computed two ways): the all-thermal baseline minus the
    // seed savings must equal the seeded optimum, and the two must differ.
    let expected_zero_seed_lb = (N_STAGES as f64) * DEMAND_MWH * THERMAL_COST; // 6000.0
    let seed_savings = (seed_hm3[0] + seed_hm3[1]) * THERMAL_COST; // 1209.6
    assert!(
        (expected_zero_seed_lb - seed_savings - expected_lb).abs() < TOL,
        "D48: seeded optimum must be the all-thermal baseline less the delivered \
         energy's thermal value: {expected_zero_seed_lb} - {seed_savings} != {expected_lb}"
    );
    assert!(
        (expected_zero_seed_lb - expected_lb).abs() > 1.0,
        "D48: the non-zero seed must MOVE the cost (load-bearing, not a tautology): \
         zero-seed {expected_zero_seed_lb} vs seeded {expected_lb}"
    );

    let case_dir = Path::new("../../examples/deterministic/d48-travel-time-ic-seed");
    let (setup, system, result) = run_deterministic_with_setup(case_dir);

    assert!(
        result.final_gap.abs() < 1e-6,
        "D48: gap={:.2e}",
        result.final_gap
    );
    assert_cost(result.final_lb, expected_lb, TOL, "D48");

    let state = setup.stage_state();
    assert_eq!(
        state.n_buckets, 2,
        "D48: exactly two bucket dimensions (single arc, IC-anchor depth 2)"
    );
    let j_canonical_idx = system
        .hydros()
        .iter()
        .position(|h| h.id == EntityId::from(1))
        .expect("D48: J (hydro id 1) must exist in the canonical hydro order");
    assert_eq!(
        state.transit_bucket_column_order,
        vec![
            (HydroSys::new(j_canonical_idx), 1),
            (HydroSys::new(j_canonical_idx), 2)
        ],
        "D48: both bucket slots must belong to J's single block, lags 1 and 2"
    );

    // End-to-end delivery split: simulate the trained policy and pin the seeded
    // arrivals, the per-stage thermal split, and seed conservation.
    let comm = StubComm;
    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("D48: simulation workspace pool must build");
    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());
    let _local_costs = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            result.frozen_templates.as_deref(),
            &result.basis_cache,
        )
        .expect("D48: simulate must return Ok");
    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");

    assert_eq!(scenario_results.len(), 1, "D48: exactly one scenario");
    let scenario = &scenario_results[0];
    assert_eq!(scenario.stages.len(), N_STAGES, "D48: exactly three stages");

    let j_id = 1;
    let expected_arrivals_hm3 = [seed_hm3[0], seed_hm3[1], 0.0]; // [60.48, 60.48, 0.0]
    let expected_thermal_mwh = [
        DEMAND_MWH - delivered_mwh[0],
        DEMAND_MWH - delivered_mwh[1],
        DEMAND_MWH - delivered_mwh[2],
    ]; // [139.52, 139.52, 200.0]

    let mut delivered_hm3 = 0.0_f64;
    for stage_result in &scenario.stages {
        let stage_id = stage_result.stage_id as usize;

        // The seed's maturing arrival at J this stage (lag-1 bucket b_1^in).
        let j_arrival_hm3 = stage_result
            .transit_buckets
            .iter()
            .find(|b| b.hydro_id == j_id && b.lag == 1)
            .map_or(0.0, |b| b.delayed_arrival_hm3);
        delivered_hm3 += j_arrival_hm3;
        assert!(
            (j_arrival_hm3 - expected_arrivals_hm3[stage_id]).abs() < TOL,
            "D48: stage {stage_id} seeded arrival at J must be {} hm3, got {j_arrival_hm3}",
            expected_arrivals_hm3[stage_id]
        );

        // J turbines every hm3 it receives (zero storage), and 1 hm3 -> 1 MWh,
        // so its generation MUST equal the arrival — the water actually offsets
        // thermal, not merely arrives on paper.
        let j_hydro = stage_result
            .hydros
            .iter()
            .find(|h| h.hydro_id == j_id)
            .unwrap_or_else(|| panic!("D48: J missing from stage {stage_id}"));
        let j_turbined_hm3 = j_hydro.turbined_m3s * M3S_TO_HM3 * HOURS_PER_STAGE;
        assert!(
            (j_turbined_hm3 - expected_arrivals_hm3[stage_id]).abs() < TOL,
            "D48: stage {stage_id} J must turbine the {} hm3 it receives, got {j_turbined_hm3}",
            expected_arrivals_hm3[stage_id]
        );

        let thermal_mwh = stage_result
            .thermals
            .iter()
            .find(|t| t.thermal_id == 0)
            .unwrap_or_else(|| panic!("D48: T0 missing from stage {stage_id}"))
            .generation_mw
            * HOURS_PER_STAGE;
        assert!(
            (thermal_mwh - expected_thermal_mwh[stage_id]).abs() < TOL,
            "D48: stage {stage_id} thermal must be {} MWh, got {thermal_mwh}",
            expected_thermal_mwh[stage_id]
        );
    }

    // Seed conservation: the entire seeded volume is delivered within the
    // horizon (U releases nothing in-study), leaving no residual in transit at
    // the terminal stage.
    let horizon_drop_hm3: f64 = scenario
        .stages
        .last()
        .expect("D48: at least one stage")
        .transit_buckets
        .iter()
        .filter(|b| b.hydro_id == j_id)
        .map(|b| b.in_transit_volume_hm3)
        .sum();
    let residual = volume_hm3 - (delivered_hm3 + horizon_drop_hm3);
    assert!(
        residual.abs() < TOL,
        "D48: seed conservation violated: seeded={volume_hm3}, delivered={delivered_hm3}, \
         horizon_drop={horizon_drop_hm3}, residual={residual}"
    );
}

/// D49: arrival-frame chronological delivery density. Cascade `U -> J`,
/// `travel_time_hours = 200` on `U`'s arc. Stages 0 and 1 are single 168 h
/// weekly PARALLEL senders; stage 2 is a monthly 720 h CHRONOLOGICAL arrival
/// stage split into blocks `[20, 100, 600]`. This is the first deterministic
/// case whose maturing bucket delivers into a chronological stage at a non-zero
/// index — the branch that resolves the delivery density in the arrival stage's
/// own frame rather than collapsing to the same-stage (first-stage) uniform
/// fallback, and rather than the parallel single-row arrival every prior water
/// case used.
///
/// ## What is exercised
///
/// Two PARALLEL source stages both mature into the ONE chronological arrival
/// stage: source stage 1 at lag 1 and source stage 0 at lag 2 — a genuine
/// multi-lag blend (`>= 2` contributing source lags), and the
/// parallel-sender -> chronological-arrival cell that used to resolve to a
/// duration-uniform density.
///
/// ## Hand-derived delivery density (`arrival_density`)
///
/// The delivered per-block split is the fixed, release-independent blend
/// `arrival_density_b = (sum_d source_weight_d * source_density_{d,b})
/// / (sum_d source_weight_d)`, with `source_weight_d` the stage-clock weight of
/// source stage `A - d` (from `resolve_spread`) and `source_density_d` that
/// source's lag-`d` delivery density resolved against the arrival stage's OWN
/// blocks `[20, 100, 600]`:
///
/// - source stage 1, lag 1: `source_weight = 1`,
///   `source_density = [0, 88/168, 80/168]`;
/// - source stage 0, lag 2: `source_weight = 32/168`,
///   `source_density = [20/32, 12/32, 0]`.
///
/// Total source weight `1 + 32/168 = 200/168`; the blend is `[0.1, 0.5, 0.4]`,
/// which conserves to 1. This test recomputes both parts from the public
/// resolvers (`resolve_spread` for the weights, `window_period_overlaps` for
/// each source density) and cross-checks the closed-form `[0.1, 0.5, 0.4]`, so
/// the expected split pins the setup precompute, never the solver output.
///
/// ## What the LP delivers
///
/// `U` starts with 100 hm3 of storage and zero natural inflow; draining it
/// within the horizon pushes water into transit that matures at stage 2. Any
/// stage-2 release from `U` would arrive same-stage (contaminating the split),
/// so releasing early is strictly better (it double-generates — once at `U`,
/// once at `J` inside the horizon) and `U` empties by stage 1, leaving nothing
/// to release at stage 2. `J` is run-of-river (zero storage), so its per-block
/// output equals its per-block arrival, which — since the same-stage release is
/// zero — is exactly `arrival_density_b` times the maturing bucket. The split
/// is a fixed LP coefficient, so it holds regardless of how much water matures,
/// making this backend-agnostic across HiGHS and CLP.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d49_travel_time_chronological_arrival_density() {
    const T_V: f64 = 200.0;
    const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;
    const N_STAGES: usize = 3;
    const ARRIVAL_STAGE_IDX: usize = 2;
    // Exact-rational precompute cross-check vs the closed-form blend.
    const DERIV_TOL: f64 = 1e-9;
    // Delivered split vs the precompute (an LP solve stands between them).
    const SPLIT_TOL: f64 = 1e-6;

    let stage_hours = [168.0, 168.0, 720.0];
    let arrival_blocks = [20.0, 100.0, 600.0];

    // Stage-clock weights (the source weights) via the public spread resolver.
    // Two source stages reach the chronological arrival stage: stage 1 at lag 1
    // and stage 0 at lag 2 — a multi-lag blend.
    let padded = pad_calendar_for_resolution(&stage_hours, T_V);
    let stage_weights_0 = resolve_spread(T_V, 0, &padded, None).stage_weights;
    let stage_weights_1 = resolve_spread(T_V, 1, &padded, None).stage_weights;
    let source_weight_lag1 = stage_weights_1[1];
    let source_weight_lag2 = stage_weights_0[2];
    assert!(
        source_weight_lag1 > 0.0 && source_weight_lag2 > 0.0,
        "D49: the blend must have >= 2 contributing source lags (lag-1 weight \
         {source_weight_lag1}, lag-2 weight {source_weight_lag2})"
    );

    // Lag-`lag` delivery density of `source_stage` resolved against the arrival
    // stage's OWN blocks — the arrival-frame counterpart of `resolve_spread`,
    // recomputed here from `window_period_overlaps` exactly as the setup
    // precompute does, so nothing is read from the solver.
    let arrival_frame_density = |source_stage: usize, lag: usize| -> Vec<f64> {
        let h_source = stage_hours[source_stage];
        let window_start = T_V;
        let window_end = T_V + h_source;
        let arrival_start: f64 = stage_hours[source_stage..source_stage + lag].iter().sum();
        let arrival_end = arrival_start + arrival_blocks.iter().sum::<f64>();
        let overlap_start = window_start.max(arrival_start);
        let overlap_end = window_end.min(arrival_end);
        let width = overlap_end - overlap_start;
        assert!(
            width > 0.0,
            "D49: source stage {source_stage} lag {lag} must reach the arrival stage"
        );
        let local_start = overlap_start - arrival_start;
        let mut row: Vec<f64> =
            cobre_core::window_period_overlaps(local_start, width, &arrival_blocks)
                .iter()
                .map(|overlap| overlap / width)
                .collect();
        row.resize(arrival_blocks.len(), 0.0);
        row
    };
    let source_density_lag1 = arrival_frame_density(1, 1);
    let source_density_lag2 = arrival_frame_density(0, 2);

    let total_source_weight = source_weight_lag1 + source_weight_lag2;
    let expected_density: Vec<f64> = (0..arrival_blocks.len())
        .map(|b| {
            (source_weight_lag1 * source_density_lag1[b]
                + source_weight_lag2 * source_density_lag2[b])
                / total_source_weight
        })
        .collect();

    // Closed-form cross-check: the hand-derived blend is [0.1, 0.5, 0.4],
    // conserving to 1.
    let closed_form = [0.1, 0.5, 0.4];
    for (b, (&got, &want)) in expected_density.iter().zip(&closed_form).enumerate() {
        assert!(
            (got - want).abs() < DERIV_TOL,
            "D49: hand-derived arrival_density[{b}] = {got}, closed-form {want}"
        );
    }
    let derived_sum: f64 = expected_density.iter().sum();
    assert!(
        (derived_sum - 1.0).abs() < DERIV_TOL,
        "D49: hand-derived arrival_density must conserve to 1.0, got {derived_sum}"
    );

    let case_dir = Path::new("../../examples/deterministic/d49-travel-time-chronological-arrival");
    let (setup, system, result) = run_deterministic_with_setup(case_dir);
    assert!(
        result.final_gap.abs() < 1e-6,
        "D49: gap={:.2e}",
        result.final_gap
    );

    // The arrival stage must be chronological at a non-zero index (the branch
    // under test); its senders must be parallel (the parallel-sender cell).
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    assert_eq!(
        study_stages.len(),
        N_STAGES,
        "D49: exactly three study stages"
    );
    // ARRIVAL_STAGE_IDX == 2 > 0: a non-first, chronological arrival stage is the
    // only construction that drives the arrival-frame branch rather than the
    // uniform first-stage fallback.
    assert_eq!(
        study_stages[ARRIVAL_STAGE_IDX].block_mode,
        BlockMode::Chronological,
        "D49: the arrival stage must be chronological to drive the arrival-frame branch"
    );
    assert_eq!(
        study_stages[0].block_mode,
        BlockMode::Parallel,
        "D49: source stage 0 must be parallel"
    );
    assert_eq!(
        study_stages[1].block_mode,
        BlockMode::Parallel,
        "D49: source stage 1 must be parallel"
    );

    let comm = StubComm;
    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("D49: simulation workspace pool must build");
    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());
    let _local_costs = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            result.frozen_templates.as_deref(),
            &result.basis_cache,
        )
        .expect("D49: simulate must return Ok");
    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");

    assert_eq!(scenario_results.len(), 1, "D49: exactly one scenario");
    let scenario = &scenario_results[0];
    assert_eq!(scenario.stages.len(), N_STAGES, "D49: exactly three stages");

    let arrival_stage = scenario
        .stages
        .iter()
        .find(|s| s.stage_id as usize == ARRIVAL_STAGE_IDX)
        .expect("D49: the arrival stage must be simulated");

    // U must release nothing same-stage at the arrival stage: only then is J's
    // per-block arrival purely the maturing bucket, so the split reads the
    // arrival density and not a fresh same-stage deposit.
    let u_release_hm3: f64 = arrival_stage
        .hydros
        .iter()
        .filter(|h| h.hydro_id == 0)
        .map(|h| {
            let b = h
                .block_id
                .expect("D49: chronological records carry a block id") as usize;
            (h.turbined_m3s + h.spillage_m3s) * M3S_TO_HM3 * arrival_blocks[b]
        })
        .sum();
    assert!(
        u_release_hm3.abs() < SPLIT_TOL,
        "D49: U must release nothing at the arrival stage so the per-block split \
         reads the maturing bucket alone, got {u_release_hm3} hm3"
    );

    // J is run-of-river, so its per-block output (turbined + spilled) equals its
    // per-block arrival — here the maturing bucket split by arrival_density.
    let mut delivered_by_block = [0.0_f64; 3];
    for h in arrival_stage.hydros.iter().filter(|h| h.hydro_id == 1) {
        let b = h
            .block_id
            .expect("D49: chronological records carry a block id") as usize;
        delivered_by_block[b] += (h.turbined_m3s + h.spillage_m3s) * M3S_TO_HM3 * arrival_blocks[b];
    }
    let delivered_total: f64 = delivered_by_block.iter().sum();
    assert!(
        delivered_total > 1.0,
        "D49: the maturing bucket must deliver a positive volume at the arrival \
         stage (else the split is unobservable), got {delivered_total} hm3"
    );

    // Cross-check the total against the reported incoming lag-1 bucket.
    let reported_arrival_hm3 = arrival_stage
        .transit_buckets
        .iter()
        .find(|b| b.hydro_id == 1 && b.lag == 1)
        .map_or(0.0, |b| b.delayed_arrival_hm3);
    assert!(
        (delivered_total - reported_arrival_hm3).abs() < SPLIT_TOL,
        "D49: J's summed per-block release ({delivered_total}) must equal the \
         reported maturing arrival ({reported_arrival_hm3})"
    );

    // The delivered per-block split must equal the hand-derived arrival density.
    let mut delivered_split_sum = 0.0_f64;
    for (b, (&delivered, &want)) in delivered_by_block.iter().zip(&expected_density).enumerate() {
        let split = delivered / delivered_total;
        delivered_split_sum += split;
        assert!(
            (split - want).abs() < SPLIT_TOL,
            "D49: delivered arrival split[{b}] = {split}, hand-derived arrival_density {want}"
        );
    }
    assert!(
        (delivered_split_sum - 1.0).abs() < SPLIT_TOL,
        "D49: the delivered arrival split must conserve to 1.0, got {delivered_split_sum}"
    );

    // The parallel-sender -> chronological-arrival cell must deliver the
    // arrival-frame blend, NOT the duration-weighted uniform density the old
    // sender-frame lookup collapsed a parallel sender to.
    let arrival_total: f64 = arrival_blocks.iter().sum();
    let duration_uniform: Vec<f64> = arrival_blocks.iter().map(|&h| h / arrival_total).collect();
    let split_vs_uniform: f64 = delivered_by_block
        .iter()
        .zip(&duration_uniform)
        .map(|(&delivered, &uniform)| (delivered / delivered_total - uniform).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        split_vs_uniform > 1e-3,
        "D49: the delivered split must be the arrival-frame blend, not the \
         duration-weighted uniform density {duration_uniform:?} (max deviation \
         {split_vs_uniform})"
    );
}

/// D50: a PLAIN tributary into a bucketed chronological confluence must not
/// perturb the maturing bucket's arrival-density split. Confluence `J` (id 2)
/// is fed by TWO upstreams: `U` (id 1) carries `travel_time_hours = 200` (a
/// declared travel-time arc, so `J` holds a maturing transit bucket) and `V`
/// (id 0) is a plain tributary with no `travel_time_hours` (a same-stage arc,
/// no maturing bucket). `V` sorts before `U` in the id-ordered cascade upstream
/// list, so the arrival-density resolver visits the plain tributary first.
///
/// `cobre-io`'s `check_chronological_confluence_heterogeneous_travel_time`
/// counts travel-time arcs only, so one travel-time arc plus one plain tributary
/// is `< 2` and passes config validation — the resolver is the only guard, and
/// it must skip `V` (which has no `arc_arrival_density` entry) rather than fold
/// its duration-uniform density into the confluence. The pre-fix resolver
/// derived a uniform density for the plain tributary and then compared it to
/// `U`'s non-uniform arrival density: a false heterogeneous-confluence panic in
/// debug/test, a silent wrong (uniform) split in release. Because `V` sorts
/// first, the pre-fix release path would seed the split with the uniform density
/// and keep it, so the delivered-split assertion below catches the release-mode
/// regression too.
///
/// Same calendar and arc as D49 (`T_V = 200`, senders `[168, 168]` parallel,
/// arrival stage `[20, 100, 600]` chronological), so the hand-derived
/// `arrival_density` is the same `[0.1, 0.5, 0.4]` blend, recomputed here from
/// the public resolvers. `V` carries zero water (zero storage, zero inflow, zero
/// turbine/generation capacity), so `J`'s stage-2 arrival is the maturing bucket
/// alone and its per-block split reads `U`'s arrival density. The split is a
/// fixed LP coefficient, so the assertion is backend-agnostic across HiGHS and
/// CLP.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d50_travel_time_plain_tributary_confluence_arrival_density() {
    const T_V: f64 = 200.0;
    const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;
    const N_STAGES: usize = 3;
    const ARRIVAL_STAGE_IDX: usize = 2;
    const DERIV_TOL: f64 = 1e-9;
    const SPLIT_TOL: f64 = 1e-6;
    // Entity ids: V is the plain tributary, U the travel-time arc, J the
    // bucketed confluence downstream.
    const V_ID: i32 = 0;
    const U_ID: i32 = 1;
    const J_ID: i32 = 2;

    let stage_hours = [168.0, 168.0, 720.0];
    let arrival_blocks = [20.0, 100.0, 600.0];

    // Hand-derived arrival_density = [0.1, 0.5, 0.4] (identical to D49: same arc,
    // same calendar), recomputed from the public resolvers so nothing is read
    // from the solver.
    let padded = pad_calendar_for_resolution(&stage_hours, T_V);
    let stage_weights_0 = resolve_spread(T_V, 0, &padded, None).stage_weights;
    let stage_weights_1 = resolve_spread(T_V, 1, &padded, None).stage_weights;
    let source_weight_lag1 = stage_weights_1[1];
    let source_weight_lag2 = stage_weights_0[2];
    assert!(
        source_weight_lag1 > 0.0 && source_weight_lag2 > 0.0,
        "D50: the blend must have >= 2 contributing source lags (lag-1 weight \
         {source_weight_lag1}, lag-2 weight {source_weight_lag2})"
    );

    let arrival_frame_density = |source_stage: usize, lag: usize| -> Vec<f64> {
        let h_source = stage_hours[source_stage];
        let window_start = T_V;
        let window_end = T_V + h_source;
        let arrival_start: f64 = stage_hours[source_stage..source_stage + lag].iter().sum();
        let arrival_end = arrival_start + arrival_blocks.iter().sum::<f64>();
        let overlap_start = window_start.max(arrival_start);
        let overlap_end = window_end.min(arrival_end);
        let width = overlap_end - overlap_start;
        assert!(
            width > 0.0,
            "D50: source stage {source_stage} lag {lag} must reach the arrival stage"
        );
        let local_start = overlap_start - arrival_start;
        let mut row: Vec<f64> =
            cobre_core::window_period_overlaps(local_start, width, &arrival_blocks)
                .iter()
                .map(|overlap| overlap / width)
                .collect();
        row.resize(arrival_blocks.len(), 0.0);
        row
    };
    let source_density_lag1 = arrival_frame_density(1, 1);
    let source_density_lag2 = arrival_frame_density(0, 2);

    let total_source_weight = source_weight_lag1 + source_weight_lag2;
    let expected_density: Vec<f64> = (0..arrival_blocks.len())
        .map(|b| {
            (source_weight_lag1 * source_density_lag1[b]
                + source_weight_lag2 * source_density_lag2[b])
                / total_source_weight
        })
        .collect();

    let closed_form = [0.1, 0.5, 0.4];
    for (b, (&got, &want)) in expected_density.iter().zip(&closed_form).enumerate() {
        assert!(
            (got - want).abs() < DERIV_TOL,
            "D50: hand-derived arrival_density[{b}] = {got}, closed-form {want}"
        );
    }
    let derived_sum: f64 = expected_density.iter().sum();
    assert!(
        (derived_sum - 1.0).abs() < DERIV_TOL,
        "D50: hand-derived arrival_density must conserve to 1.0, got {derived_sum}"
    );

    let case_dir =
        Path::new("../../examples/deterministic/d50-travel-time-plain-tributary-confluence");
    let (setup, system, result) = run_deterministic_with_setup(case_dir);
    assert!(
        result.final_gap.abs() < 1e-6,
        "D50: gap={:.2e}",
        result.final_gap
    );

    // The triggering topology: J is fed by V (plain, id 0) then U (travel-time,
    // id 1) in id-order, and only U carries a travel-time arc. This is exactly
    // the confluence the pre-fix resolver mishandled.
    let upstream_of_j = system.cascade().upstream(cobre_core::EntityId(J_ID));
    assert_eq!(
        upstream_of_j,
        &[cobre_core::EntityId(V_ID), cobre_core::EntityId(U_ID)],
        "D50: J must be fed by V (id 0, plain) then U (id 1, travel-time)"
    );
    let travel_time_of = |id: i32| {
        system
            .hydros()
            .iter()
            .find(|h| h.id == cobre_core::EntityId(id))
            .unwrap_or_else(|| panic!("D50: hydro {id} must be present"))
            .travel_time_hours
    };
    assert!(
        travel_time_of(V_ID).is_none(),
        "D50: V (id 0) must be a plain tributary with no travel_time_hours"
    );
    assert_eq!(
        travel_time_of(U_ID),
        Some(T_V),
        "D50: U (id 1) must carry the declared travel-time arc"
    );

    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    assert_eq!(
        study_stages.len(),
        N_STAGES,
        "D50: exactly three study stages"
    );
    assert_eq!(
        study_stages[ARRIVAL_STAGE_IDX].block_mode,
        BlockMode::Chronological,
        "D50: the arrival stage must be chronological to drive the arrival-frame branch"
    );
    assert_eq!(
        study_stages[0].block_mode,
        BlockMode::Parallel,
        "D50: source stage 0 must be parallel"
    );
    assert_eq!(
        study_stages[1].block_mode,
        BlockMode::Parallel,
        "D50: source stage 1 must be parallel"
    );

    let comm = StubComm;
    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("D50: simulation workspace pool must build");
    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());
    let _local_costs = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            result.frozen_templates.as_deref(),
            &result.basis_cache,
        )
        .expect("D50: simulate must return Ok");
    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");

    assert_eq!(scenario_results.len(), 1, "D50: exactly one scenario");
    let scenario = &scenario_results[0];
    assert_eq!(scenario.stages.len(), N_STAGES, "D50: exactly three stages");

    let arrival_stage = scenario
        .stages
        .iter()
        .find(|s| s.stage_id as usize == ARRIVAL_STAGE_IDX)
        .expect("D50: the arrival stage must be simulated");

    // The plain tributary V carries zero water, so it releases nothing at the
    // arrival stage — its presence in the confluence is purely topological and
    // must not add a same-stage deposit onto J's arrival split.
    let v_release_hm3: f64 = arrival_stage
        .hydros
        .iter()
        .filter(|h| h.hydro_id == V_ID)
        .map(|h| {
            let b = h
                .block_id
                .expect("D50: chronological records carry a block id") as usize;
            (h.turbined_m3s + h.spillage_m3s) * M3S_TO_HM3 * arrival_blocks[b]
        })
        .sum();
    assert!(
        v_release_hm3.abs() < SPLIT_TOL,
        "D50: the plain tributary V must release nothing at the arrival stage, got {v_release_hm3} hm3"
    );

    // U must release nothing same-stage at the arrival stage: only then is J's
    // per-block arrival purely the maturing bucket, so the split reads the
    // arrival density and not a fresh same-stage deposit.
    let u_release_hm3: f64 = arrival_stage
        .hydros
        .iter()
        .filter(|h| h.hydro_id == U_ID)
        .map(|h| {
            let b = h
                .block_id
                .expect("D50: chronological records carry a block id") as usize;
            (h.turbined_m3s + h.spillage_m3s) * M3S_TO_HM3 * arrival_blocks[b]
        })
        .sum();
    assert!(
        u_release_hm3.abs() < SPLIT_TOL,
        "D50: U must release nothing at the arrival stage so the per-block split \
         reads the maturing bucket alone, got {u_release_hm3} hm3"
    );

    // J is run-of-river, so its per-block output (turbined + spilled) equals its
    // per-block arrival — here the maturing bucket split by arrival_density.
    let mut delivered_by_block = [0.0_f64; 3];
    for h in arrival_stage.hydros.iter().filter(|h| h.hydro_id == J_ID) {
        let b = h
            .block_id
            .expect("D50: chronological records carry a block id") as usize;
        delivered_by_block[b] += (h.turbined_m3s + h.spillage_m3s) * M3S_TO_HM3 * arrival_blocks[b];
    }
    let delivered_total: f64 = delivered_by_block.iter().sum();
    assert!(
        delivered_total > 1.0,
        "D50: the maturing bucket must deliver a positive volume at the arrival \
         stage (else the split is unobservable), got {delivered_total} hm3"
    );

    let reported_arrival_hm3 = arrival_stage
        .transit_buckets
        .iter()
        .find(|b| b.hydro_id == J_ID && b.lag == 1)
        .map_or(0.0, |b| b.delayed_arrival_hm3);
    assert!(
        (delivered_total - reported_arrival_hm3).abs() < SPLIT_TOL,
        "D50: J's summed per-block release ({delivered_total}) must equal the \
         reported maturing arrival ({reported_arrival_hm3})"
    );

    // The delivered per-block split must equal U's hand-derived arrival density —
    // NOT the duration-uniform density the plain tributary would have injected.
    let mut delivered_split_sum = 0.0_f64;
    for (b, (&delivered, &want)) in delivered_by_block.iter().zip(&expected_density).enumerate() {
        let split = delivered / delivered_total;
        delivered_split_sum += split;
        assert!(
            (split - want).abs() < SPLIT_TOL,
            "D50: delivered arrival split[{b}] = {split}, hand-derived arrival_density {want}"
        );
    }
    assert!(
        (delivered_split_sum - 1.0).abs() < SPLIT_TOL,
        "D50: the delivered arrival split must conserve to 1.0, got {delivered_split_sum}"
    );

    let arrival_total: f64 = arrival_blocks.iter().sum();
    let duration_uniform: Vec<f64> = arrival_blocks.iter().map(|&h| h / arrival_total).collect();
    let split_vs_uniform: f64 = delivered_by_block
        .iter()
        .zip(&duration_uniform)
        .map(|(&delivered, &uniform)| (delivered / delivered_total - uniform).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        split_vs_uniform > 1e-3,
        "D50: the delivered split must be U's arrival-frame blend, not the \
         duration-weighted uniform density {duration_uniform:?} the plain \
         tributary would have injected (max deviation {split_vs_uniform})"
    );
}

// ---------------------------------------------------------------------------
// Behavioral convergence coverage for the non-golden (demoted) hashed cases
//
// Each demoted case is single-scenario deterministic, so it converges to a
// hand-stable optimum with LB == UB. The asserted cost is that converged optimum
// and is backend-agnostic — HiGHS and CLP agree to the absolute tolerance below.
// This is the behavioral safety net (tier 2) that lets the bit-exact golden tier
// shrink to a small feature-combined set without losing regression coverage.
// ---------------------------------------------------------------------------

/// Generate a deterministic "converges to a known optimum" regression test: train
/// the case, assert the LB/UB gap closes (`< 1e-6`), and assert the final LB
/// equals the known optimum within `tolerance`. Used only for cases whose body is
/// exactly this gap + cost check; cases with extra setup or assertions stay explicit.
macro_rules! converges_case {
    ($(#[$meta:meta])* $name:ident, $dir:literal, $label:literal, $expected:expr, $tolerance:expr) => {
        $(#[$meta])*
        #[cfg_attr(
            not(feature = "slow-tests"),
            ignore = "slow: run with --features slow-tests"
        )]
        #[test]
        fn $name() {
            let case_dir = Path::new(concat!("../../examples/deterministic/", $dir));
            let result = run_deterministic(case_dir);
            assert!(
                result.final_gap.abs() < 1e-6,
                concat!($label, ": gap={:.2e}"),
                result.final_gap
            );
            assert_cost(result.final_lb, $expected, $tolerance, $label);
        }
    };
}

converges_case!(
    /// D17: mixed-sign (per-month) evaporation.
    d17_converges_to_known_optimum,
    "d17-evaporation-mixed-sign",
    "D17",
    4_380_000.0,
    1.0
);

converges_case!(
    /// D31: backwater reference volume — upstream computed-FPHA plane shift.
    d31_converges_to_known_optimum,
    "d31-backwater-reference-volume",
    "D31",
    1_475_795.556_870_993_5,
    1.0
);

converges_case!(
    /// D32: reversible (pumping) plant.
    d32_converges_to_known_optimum,
    "d32-reversible-plant",
    "D32",
    1_126_109.6,
    1.0
);

converges_case!(
    /// D35: pumping-station commissioning window.
    d35_converges_to_known_optimum,
    "d35-pumping-commissioning",
    "D35",
    1_245_756.166_666_667,
    1.0
);

converges_case!(
    /// D36: thermal + line commissioning window.
    d36_converges_to_known_optimum,
    "d36-thermal-line-commissioning",
    "D36",
    30_713_514.888_888_91,
    1.0
);

converges_case!(
    /// D37: anticipated thermal under a commissioning window.
    d37_converges_to_known_optimum,
    "d37-anticipated-commissioning",
    "D37",
    15_444_634.222_222_22,
    1.0
);

converges_case!(
    /// D38: dead-volume filling hydro.
    d38_converges_to_known_optimum,
    "d38-dead-volume-filling",
    "D38",
    21_821_186.4,
    1.0
);

converges_case!(
    /// D39: PreFilling hydro upstream of a Filling hydro.
    d39_converges_to_known_optimum,
    "d39-prefilling-upstream-of-filling",
    "D39",
    4_882_544.0,
    1.0
);

converges_case!(
    /// D40: filling cascade.
    d40_converges_to_known_optimum,
    "d40-filling-cascade",
    "D40",
    27_503_012.0,
    1.0
);

converges_case!(
    /// D42: non-filling hydro commissioning.
    d42_converges_to_known_optimum,
    "d42-nonfilling-hydro-commissioning",
    "D42",
    10_185_524.0,
    1.0
);

/// D33: per-stage varying block counts (convergence companion to
/// [`d33_per_stage_block_count_varies`], which only checks the structural strides).
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d33_converges_to_known_optimum() {
    let case_dir = Path::new("../../examples/deterministic/d33-per-stage-block-counts");
    let result = run_deterministic(case_dir);
    assert!(
        result.final_gap.abs() < 1e-6,
        "D33: gap={:.2e}",
        result.final_gap
    );
    assert_cost(result.final_lb, 553.333_333_333_3, 1e-2, "D33");
}

/// D51: split-plant two-bus fixture — the repo's first multi-cell hydro plant.
///
/// One FPHA hydro (H0) declares two unit groups on two different buses (B0,
/// B1), so `HydroCellIndex` partitions it into two cells (one per bus). Both
/// groups keep their own declared envelope at every stage (H0-B0: 30.0 MW /
/// 42.0 m3/s; H0-B1: 20.0 MW / 28.0 m3/s — declared sum 50.0 MW / 70.0 m3/s,
/// strictly below the plant's own declared 60.0 MW / 80.0 m3/s, satisfying
/// rule 41 with slack). A single plant-axis `hydro_bounds.parquet` row LOWERS
/// the plant's resolved envelope at stage 0 to 28.0 MW / 32.0 m3/s — strictly
/// between the two groups' declared values — so the per-cell `min(group box,
/// plant envelope)` resolution (spec section 1.3) binds on the PLANT term for
/// H0-B0's cell and on the GROUP term for H0-B1's cell at that stage; rule 43
/// permits this because it only ever lowers, never raises, the plant's own
/// declared value. A connecting line (absolute `capacity.direct_mw`/
/// `reverse_mw`, no `exchange_factors.json`) carries B1's hydro surplus to
/// B0, which cannot meet its own 45 MW load from its local cell alone; a
/// thermal at B0 has a per-block override in `thermal_bounds.parquet` (stage
/// 0's PEAK block caps at 8.0 MW, OFFPEAK at 25.0 MW) covering the residual
/// at stage 1.
///
/// This case is not hand-derived symbolically (the FPHA plane's storage term
/// couples the two stages); instead it is pinned at its observed converged
/// value, matching the established pattern for other structurally complex
/// fixtures in this suite (e.g. D31, D40, D42). `final_lb == final_ub`
/// bit-for-bit on HiGHS; CLP reproduces it to ~5e-10 (both well inside the
/// tolerance below). Deliberately NOT gated behind `slow-tests` — the
/// 2-stage horizon converges in a fraction of a second.
#[test]
fn d51_split_plant_two_bus_converges_to_known_optimum() {
    let case_dir = Path::new("../../examples/deterministic/d51-split-plant-two-bus");
    let result = run_deterministic(case_dir);
    assert!(
        result.final_gap.abs() < 1e-6,
        "D51: gap={:.2e}",
        result.final_gap
    );
    assert!(
        result.iterations <= 10,
        "D51: iterations={}",
        result.iterations
    );
    assert_cost(result.final_lb, 2_277_012.200_056_066_3, 1.0, "D51");
}

/// D51: per-`(stage, block)` bit-exact sum of the split plant's
/// `hydro_bus_generation` rows against its own `hydros` row.
///
/// `extract_hydro_bus_generation` sums each cell's LP column in the SAME
/// ascending-cell order `extract_hydro_per_block`'s own `.sum()` consumes, so
/// summing the per-bus rows in the writer's own row order reproduces the
/// plant row bit-for-bit for both `turbined_m3s` and `generation_mw`. The
/// `generation_mw` equality is an FPHA/single-cell property — each cell reads
/// its OWN independent LP column, with no multiply-then-sum reordering — and
/// is deliberately NOT claimed for a `ConstantProductivity` multi-cell plant,
/// where `generation_mw` is `turbined_m3s * productivity` per cell and
/// `(Σq)·ρ != Σ(q·ρ)` in floating point (spec section 7.10).
#[test]
fn d51_hydro_bus_generation_sums_bit_exactly_to_plant_rows() {
    let case_dir = Path::new("../../examples/deterministic/d51-split-plant-two-bus");
    let (_result, scenario_results, _summary) = run_with_simulation(case_dir);

    assert_eq!(
        scenario_results.len(),
        1,
        "D51: exactly 1 simulated scenario"
    );
    let scenario = &scenario_results[0];
    assert_eq!(scenario.stages.len(), 2, "D51: exactly 2 stages");

    let mut checked_blocks = 0_usize;
    for stage in &scenario.stages {
        assert_eq!(
            stage.hydro_bus_generation.len(),
            2 * stage.hydros.len(),
            "D51 stage {}: 2 bus rows per plant row (one per cell)",
            stage.stage_id
        );
        for plant_row in &stage.hydros {
            let bus_sum_turbined: f64 = stage
                .hydro_bus_generation
                .iter()
                .filter(|r| r.hydro_id == plant_row.hydro_id && r.block_id == plant_row.block_id)
                .map(|r| r.turbined_m3s)
                .sum();
            let bus_sum_generation: f64 = stage
                .hydro_bus_generation
                .iter()
                .filter(|r| r.hydro_id == plant_row.hydro_id && r.block_id == plant_row.block_id)
                .map(|r| r.generation_mw)
                .sum();

            assert_eq!(
                bus_sum_turbined.to_bits(),
                plant_row.turbined_m3s.to_bits(),
                "D51 stage {} block {:?}: bus-row turbined_m3s sum ({bus_sum_turbined}) must \
                 equal the plant row's turbined_m3s ({}) bit-for-bit",
                stage.stage_id,
                plant_row.block_id,
                plant_row.turbined_m3s
            );
            assert_eq!(
                bus_sum_generation.to_bits(),
                plant_row.generation_mw.to_bits(),
                "D51 stage {} block {:?}: bus-row generation_mw sum ({bus_sum_generation}) must \
                 equal the plant row's generation_mw ({}) bit-for-bit (FPHA/single-cell \
                 property, see this test's doc comment)",
                stage.stage_id,
                plant_row.block_id,
                plant_row.generation_mw
            );
            checked_blocks += 1;
        }
    }
    assert_eq!(
        checked_blocks, 3,
        "D51: 2 blocks at stage 0 + 1 at stage 1 = 3 (stage, block) pairs"
    );
}

/// D52: per-block `max_generation_mw` cap binds in the block it targets, not
/// a neighbouring one.
///
/// T-CHEAP (thermal 0, cost 0.0 $/MWh) carries a per-block override at stage 0:
/// PEAK (block 0) caps it at 100.0 MW, OFFPEAK (block 1) at 500.0 MW. The
/// deterministic bus load is 150.0 MW in every block. In PEAK the cap sits
/// strictly below load, so T-CHEAP dispatches at exactly its own cap and the
/// substitute source T-SUB (cost 300.0 $/MWh) covers the 50.0 MW shortfall;
/// in OFFPEAK (cap 500.0 MW, above load) T-CHEAP alone covers the full
/// 150.0 MW and T-SUB stays at zero. A resolver reading a single stage-level
/// bound, or OFFPEAK's override instead of PEAK's, would let T-CHEAP
/// dispatch at 150.0 MW (load) or 500.0 MW (the other block's cap) in PEAK.
#[test]
fn d52_per_block_cap_binds_in_peak_block() {
    let case_dir = Path::new("../../examples/deterministic/d52-per-block-thermal-fold");
    let (_result, scenario_results, _summary) = run_with_simulation(case_dir);

    assert_eq!(
        scenario_results.len(),
        1,
        "D52: exactly 1 simulated scenario"
    );
    let stage0 = &scenario_results[0].stages[0];
    assert_eq!(stage0.stage_id, 0, "D52: first stage result is stage 0");

    let peak_cheap = stage0
        .thermals
        .iter()
        .find(|t| t.thermal_id == 0 && t.block_id == Some(0))
        .expect("D52: T-CHEAP row at (stage 0, block 0 = PEAK) must exist");
    assert!(
        (peak_cheap.generation_mw - 100.0).abs() < 1e-6,
        "D52: PEAK-block T-CHEAP generation must equal its own block's cap \
         (100.0 MW), got {} — a stage-level or off-peak bound would read \
         150.0 (load) or 500.0 (the other block's cap)",
        peak_cheap.generation_mw
    );

    let offpeak_cheap = stage0
        .thermals
        .iter()
        .find(|t| t.thermal_id == 0 && t.block_id == Some(1))
        .expect("D52: T-CHEAP row at (stage 0, block 1 = OFFPEAK) must exist");
    assert!(
        (offpeak_cheap.generation_mw - 150.0).abs() < 1e-6,
        "D52: OFFPEAK-block T-CHEAP generation must equal full load \
         (150.0 MW, under its own 500.0 MW cap), got {}",
        offpeak_cheap.generation_mw
    );

    let peak_sub = stage0
        .thermals
        .iter()
        .find(|t| t.thermal_id == 1 && t.block_id == Some(0))
        .expect("D52: T-SUB row at (stage 0, block 0 = PEAK) must exist");
    assert!(
        (peak_sub.generation_mw - 50.0).abs() < 1e-6,
        "D52: T-SUB must cover the 50.0 MW PEAK-block shortfall \
         (load 150.0 - cap 100.0), got {}",
        peak_sub.generation_mw
    );
}

/// D52's own deck inputs, named once so the fold-delta derivation below is a
/// pure function of them rather than a re-typed literal.
const D52_LOAD_MW: f64 = 150.0;
const D52_PEAK_CAP_MW: f64 = 100.0;
const D52_OFFPEAK_CAP_MW: f64 = 500.0;
const D52_PEAK_HOURS: f64 = 100.0;
const D52_OFFPEAK_HOURS: f64 = 300.0;
const D52_CHEAP_COST_PER_MWH: f64 = 0.0;
const D52_SUBSTITUTE_COST_PER_MWH: f64 = 300.0;

/// D52: the objective difference between the committed per-block bound
/// configuration and its hours-weighted fold to one stage value equals a
/// closed-form expression of the deck's own inputs — never a value read off
/// either run.
///
/// ## Derivation
///
/// The fold collapses stage 0's two per-block caps into one stage-wide value
/// (the same average this suite's helper writes into the folded copy):
/// `fold = (PEAK_HOURS·PEAK_CAP + OFFPEAK_HOURS·OFFPEAK_CAP) /
/// (PEAK_HOURS + OFFPEAK_HOURS) = (100·100 + 300·500) / 400 = 400.0 MW`.
/// Because `fold (400.0) >= LOAD_MW (150.0)`, the folded configuration lets
/// T-CHEAP alone cover the full load in every block of stage 0 — including
/// the former PEAK block — so T-SUB never dispatches there. The per-block
/// configuration instead caps T-CHEAP at `PEAK_CAP_MW` in the PEAK block
/// alone, forcing T-SUB to cover the `shortfall = LOAD_MW - PEAK_CAP_MW =
/// 50.0` MW residual at `SUBSTITUTE_COST_PER_MWH`. OFFPEAK and stage 1 are
/// unaffected by the fold (both configurations' caps there stay >= load), so
/// the entire objective delta is confined to the PEAK block:
///
/// `delta = shortfall_MW × (SUBSTITUTE_COST_PER_MWH − CHEAP_COST_PER_MWH) ×
/// PEAK_HOURS`
///
/// `CHEAP_COST_PER_MWH` is 0.0 by construction (T-CHEAP is the deck's
/// zero-cost source), so this collapses to `shortfall_MW × substitute_cost ×
/// peak_hours` — kept here in the general cost-difference form so the
/// derivation stays correct if T-CHEAP's cost is ever changed off zero.
#[test]
fn d52_hours_weighted_fold_delta_matches_hand_derivation() {
    let case_dir = Path::new("../../examples/deterministic/d52-per-block-thermal-fold");

    let per_block_result = run_deterministic(case_dir);
    assert!(
        per_block_result.final_gap.abs() < 1e-6,
        "D52 per-block: gap={:.2e}",
        per_block_result.final_gap
    );
    assert!(
        (per_block_result.final_lb - per_block_result.final_ub).abs() < 1e-6,
        "D52 per-block: LB ({}) must equal UB ({}) to tolerance",
        per_block_result.final_lb,
        per_block_result.final_ub
    );

    let fold_cap_mw = (D52_PEAK_HOURS * D52_PEAK_CAP_MW + D52_OFFPEAK_HOURS * D52_OFFPEAK_CAP_MW)
        / (D52_PEAK_HOURS + D52_OFFPEAK_HOURS);
    assert!(
        fold_cap_mw >= D52_LOAD_MW,
        "D52: the fold ({fold_cap_mw} MW) must sit at or above load \
         ({D52_LOAD_MW} MW), or the folded configuration would ALSO need \
         T-SUB in the PEAK block, invalidating the single-term delta \
         derivation below"
    );

    let folded_case = copy_case_dir(case_dir);
    write_stage_wide_thermal_bound(
        &folded_case
            .path()
            .join("constraints/thermal_bounds.parquet"),
        0,
        0,
        fold_cap_mw,
    );
    let folded_result = run_deterministic(folded_case.path());
    assert!(
        folded_result.final_gap.abs() < 1e-6,
        "D52 folded: gap={:.2e}",
        folded_result.final_gap
    );
    assert!(
        (folded_result.final_lb - folded_result.final_ub).abs() < 1e-6,
        "D52 folded: LB ({}) must equal UB ({}) to tolerance",
        folded_result.final_lb,
        folded_result.final_ub
    );

    let shortfall_mw = D52_LOAD_MW - D52_PEAK_CAP_MW;
    let expected_delta =
        shortfall_mw * (D52_SUBSTITUTE_COST_PER_MWH - D52_CHEAP_COST_PER_MWH) * D52_PEAK_HOURS;

    let actual_delta = per_block_result.final_lb - folded_result.final_lb;
    assert!(
        (actual_delta - expected_delta).abs() < 1e-6,
        "D52: per-block cost ({}) minus folded cost ({}) = {actual_delta}, \
         expected the hand-derived delta {expected_delta} (shortfall \
         {shortfall_mw} MW x cost delta {} $/MWh x {} peak hours)",
        per_block_result.final_lb,
        folded_result.final_lb,
        D52_SUBSTITUTE_COST_PER_MWH - D52_CHEAP_COST_PER_MWH,
        D52_PEAK_HOURS
    );
}

/// D53: a per-cell min-turbine floor genuinely BINDS under water starvation —
/// the behavioral fixture for the per-cell min-floor reversal (spec §7.9,
/// `.claude/rules/sddp.md`'s min-floor contract).
///
/// Same split-plant topology as D51 (one FPHA hydro, unit group 0 on bus 0 /
/// cell A with no floor, unit group 1 on bus 1 / cell B with
/// `min_turbined_m3s = 27.0`), but WITHOUT D51's max-side group-bounds
/// overrides, and with deliberately low inflow (`3.0`/`2.0 m3/s` at stages
/// 0/1) so the reservoir cannot sustain cell B's floor: total available
/// water is `120.0 hm3` (initial storage) `+ (3.0 + 2.0) * 730 * 0.0036 hm3`
/// (inflow) `= 133.14 hm3`, while cell B's floor alone demands
/// `27.0 * 730 * 0.0036 * 2 = 141.9 hm3` over the two stages — a shortfall
/// that cannot be closed even with cell A turbining nothing (cell A has no
/// floor and is economically free to do so). This is genuine water
/// starvation, not a structural cap: cell B's own `max_turbined_m3s` stays
/// `28.0` at every stage, strictly above the `27.0` floor, so the shortfall
/// below can only come from insufficient water, never an unreachable column
/// bound.
///
/// `final_lb == final_ub` to within `1e-6` (both backends observed
/// bit-identical); pinned at the observed converged value, matching the
/// established pattern for structurally complex FPHA fixtures (D31, D40,
/// D42, D51) — the FPHA plane's storage term couples the two stages, so this
/// case is not hand-derived symbolically. What IS hand-derived and asserted
/// directly is the min-floor contract itself: cell B's actual turbined flow
/// plus its slack reaches EXACTLY the declared floor every block, storage is
/// fully drained to `0.0` by the terminal stage (proving water scarcity, not
/// a capacity artifact), and each stage's `turbined_violation_cost` equals
/// `shortfall * turbined_violation_below_cost * block_hours` — "penalty =
/// shortfall × plant price" — computed from the SAME observed slack the
/// floor assertion above uses, read straight off `SimulationCostResult`
/// rather than re-derived.
#[test]
fn d53_hydro_cell_min_floor_binds_under_water_starvation() {
    const FLOOR_M3S: f64 = 27.0;
    const TURBINED_VIOLATION_BELOW_COST: f64 = 10_000.0;
    const PEAK_HOURS: f64 = 200.0;
    const OFFPEAK_HOURS: f64 = 530.0;
    const STAGE1_HOURS: f64 = 730.0;

    let case_dir = Path::new("../../examples/deterministic/d53-hydro-cell-min-floor");
    let (result, scenario_results, _summary) = run_with_simulation(case_dir);

    assert!(
        result.final_gap.abs() < 1e-6,
        "D53: gap={:.2e}",
        result.final_gap
    );
    assert!(
        (result.final_lb - result.final_ub).abs() < 1e-6,
        "D53: LB ({}) must equal UB ({}) to tolerance",
        result.final_lb,
        result.final_ub
    );
    assert!(
        result.iterations <= 10,
        "D53: iterations={}",
        result.iterations
    );
    assert_cost(result.final_lb, 40_373_560.917_320_274, 1.0, "D53");

    assert_eq!(
        scenario_results.len(),
        1,
        "D53: exactly 1 simulated scenario"
    );
    let scenario = &scenario_results[0];
    assert_eq!(scenario.stages.len(), 2, "D53: exactly 2 stages");

    let mut checked_blocks = 0_usize;
    for stage in &scenario.stages {
        assert_eq!(
            stage.costs.len(),
            1,
            "D53 stage {}: one stage-level cost aggregate",
            stage.stage_id
        );
        let mut expected_turbined_violation_cost = 0.0_f64;

        for h in &stage.hydros {
            assert_eq!(h.hydro_id, 0, "D53: the only hydro is H0");
            let hours = match (stage.stage_id, h.block_id) {
                (0, Some(0)) => PEAK_HOURS,
                (0, Some(1)) => OFFPEAK_HOURS,
                (1, Some(0)) => STAGE1_HOURS,
                other => panic!("D53: unexpected (stage, block) {other:?}"),
            };

            // The floor genuinely binds: actual turbined flow is strictly
            // below the declared floor, and the slack picks up exactly the
            // difference (the soft row's `q_c + slack_c >= floor` holds as an
            // equality at the cost-minimizing optimum).
            assert!(
                h.turbined_m3s < FLOOR_M3S - 1e-6,
                "D53 stage {} block {:?}: turbined ({}) must fall strictly below the \
                 {FLOOR_M3S} m3/s floor — the fixture's whole point is that it starves",
                stage.stage_id,
                h.block_id,
                h.turbined_m3s
            );
            assert!(
                h.turbined_slack_m3s > 1e-6,
                "D53 stage {} block {:?}: turbined_slack ({}) must be strictly positive \
                 — the floor must be active, not merely declared",
                stage.stage_id,
                h.block_id,
                h.turbined_slack_m3s
            );
            assert!(
                (h.turbined_m3s + h.turbined_slack_m3s - FLOOR_M3S).abs() < 1e-6,
                "D53 stage {} block {:?}: turbined ({}) + slack ({}) must reach the \
                 {FLOOR_M3S} m3/s floor exactly",
                stage.stage_id,
                h.block_id,
                h.turbined_m3s,
                h.turbined_slack_m3s
            );
            assert_eq!(
                h.generation_slack_mw, 0.0,
                "D53 stage {} block {:?}: the generation floor is declared 0.0 and must \
                 stay inert — only the turbined floor is exercised by this fixture",
                stage.stage_id, h.block_id
            );

            expected_turbined_violation_cost +=
                h.turbined_slack_m3s * TURBINED_VIOLATION_BELOW_COST * hours;
            checked_blocks += 1;
        }

        // "Penalty = shortfall × plant price": the stage's reported
        // turbined-violation cost equals the SAME slack values asserted
        // above, priced at the plant's declared `turbined_violation_below_cost`
        // (10000.0, `penalties.json`) for each block's own hours — never
        // divided by the plant's cell count (there is only one floor-bearing
        // cell here, so a `1/|cells|` apportionment bug would still show up
        // as a factor-of-2 understatement against cell A's own zero floor
        // contributing nothing).
        assert!(
            (stage.costs[0].turbined_violation_cost - expected_turbined_violation_cost).abs()
                < 1e-3,
            "D53 stage {}: reported turbined_violation_cost ({}) must equal shortfall * \
             price * hours ({expected_turbined_violation_cost})",
            stage.stage_id,
            stage.costs[0].turbined_violation_cost
        );
        assert_eq!(
            stage.costs[0].generation_violation_cost, 0.0,
            "D53 stage {}: the generation floor is inert, so its violation cost must be 0.0",
            stage.stage_id
        );

        // Storage is fully drained by the terminal stage: the shortfall above
        // is genuine water starvation, not a structural cap set below the
        // floor (cell B's own max_turbined_m3s stays 28.0 > 27.0 throughout).
        if stage.stage_id == 1 {
            for h in &stage.hydros {
                assert_eq!(
                    h.storage_final_hm3, 0.0,
                    "D53: storage must be fully drained by the terminal stage, proving the \
                     floor's shortfall comes from water scarcity, not an unreachable cap"
                );
            }
        }

        // Cell A (bus 0) carries no floor and is economically free to turbine
        // nothing, ceding all available water to cell B (bus 1) — confirming
        // the plant-level slack summed above is entirely cell B's own, never
        // apportioned across the plant's two cells.
        for hb in &stage.hydro_bus_generation {
            if hb.bus_id == 0 {
                assert!(
                    hb.turbined_m3s.abs() < 1e-6,
                    "D53 stage {} block {:?}: cell A (bus 0) must turbine ~0.0, got {}",
                    stage.stage_id,
                    hb.block_id,
                    hb.turbined_m3s
                );
            }
        }
    }
    assert_eq!(
        checked_blocks, 3,
        "D53: 2 blocks at stage 0 + 1 at stage 1 = 3 (stage, block) pairs"
    );
}

/// Chronological-blocks telescoping ⇒ parallel bound-agreement anchor.
///
/// Pins the "telescoping ⇒ parallel agreement when interiors are inert" contract
/// at the solved-bound level: with `γᵥ = 0` (constant productivity), no
/// storage-dependent evaporation, and non-binding interior storage bounds, a
/// chronological (`K = 2`) run's converged lower bound equals the matched parallel
/// run's. The water rows telescope unconditionally (`Sᵏ` cancels to `Sᴷ − S⁰`,
/// `Σ τ_k = ζ`); the bound agreement holds only because nothing makes the interior
/// storage path bind.
mod chronological_telescoping {
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, DeficitSegment,
        EntityId, HydroBlockBounds, HydroGenerationModel, HydroPenalties, HydroStageBounds,
        HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSelection, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_solver::ActiveSolver;

    use cobre_io::{PolicyCheckpoint, read_policy_checkpoint};
    use cobre_sddp::FutureCostFunction;
    use cobre_sddp::policy::orchestration::{CheckpointParams, write_checkpoint};
    use cobre_sddp::policy_export::build_stage_cut_records;
    use tempfile::TempDir;

    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage,
    };
    use super::common::{StubComm, build_setup_in_code, try_build_setup_in_code};

    const N_STAGES: usize = 3;
    const N_ITERATIONS: u32 = 12;
    const HYDRO_ID: i32 = 1;

    fn zero_hydro_stage_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 0.0,
        }
    }

    /// Two-block, constant-productivity hydro + backup thermal on one bus, built under
    /// `block_mode`. Constant productivity means `γᵥ = 0`; no evaporation coefficients
    /// means no storage-dependent loss; the wide `[0, 500]` storage bounds never bind —
    /// the three conditions the inert-interior contract requires.
    fn build_system(block_mode: BlockMode) -> cobre_core::System {
        use chrono::NaiveDate;

        let bus = make_bus(
            EntityId(2),
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
            },
        );

        let hydro = make_hydro(
            EntityId(HYDRO_ID),
            HydroSpec {
                name: "H1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(2),
                min_storage_hm3: 0.0,
                max_storage_hm3: 500.0,
                max_turbined_m3s: 100.0,
                generation_model: HydroGenerationModel::ConstantProductivity,
                specific_productivity_mw_per_m3s_per_m: Some(0.5),
                max_generation_mw: 250.0,
                penalties: zero_hydro_stage_penalties(),
                ..Default::default()
            },
        );

        // Two blocks with distinct durations so the chronological chain is a genuine
        // K = 2 chain (τ_0 ≠ τ_1) whose telescoped total still recovers the parallel ζ.
        let blocks = vec![
            Block {
                index: 0,
                name: "B0".to_string(),
                duration_hours: 300.0,
            },
            Block {
                index: 1,
                name: "B1".to_string(),
                duration_hours: 444.0,
            },
        ];

        let stages: Vec<Stage> = (0..N_STAGES)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: NaiveDate::from_ymd_opt(2024, (i % 12 + 1) as u32, 1).unwrap(),
                        end_date: NaiveDate::from_ymd_opt(2024, ((i % 12 + 1) % 12 + 1) as u32, 1)
                            .unwrap(),
                        season_id: Some(0),
                        blocks: blocks.clone(),
                        block_mode,
                        state_config: StageStateConfig {
                            storage: true,
                            inflow_lags: false,
                        },
                        risk_config: StageRiskConfig::Expectation,
                        scenario_config: ScenarioSourceConfig {
                            branching_factor: 1,
                            noise_method: NoiseMethod::Saa,
                        },
                    },
                )
            })
            .collect();

        // Deterministic: zero-variance inflow and load so the optimum is a single
        // scenario and both modes converge to the same bound.
        let inflow_models: Vec<InflowModel> = (0..N_STAGES)
            .map(|i| InflowModel {
                hydro_id: EntityId(HYDRO_ID),
                stage_id: i32::try_from(i).expect("stage index fits i32"),
                mean_m3s: 60.0,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..N_STAGES)
            .map(|i| LoadModel {
                bus_id: EntityId(2),
                stage_id: i32::try_from(i).expect("stage index fits i32"),
                mean_mw: 120.0,
                std_mw: 0.0,
            })
            .collect();

        let default_hydro_bounds = || HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 500.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        };
        let default_hydro_bounds_block = || HydroBlockBounds {
            max_turbined_m3s: 100.0,
            max_generation_mw: 250.0,
            ..Default::default()
        };

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_bounds_block(),
                thermal: ThermalStageBounds {
                    cost_per_mwh: 100.0,
                },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 400.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: zero_hydro_stage_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(HYDRO_ID),
                value_hm3: 200.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![super::common::builders::make_thermal(
                EntityId(3),
                super::common::builders::ThermalSpec {
                    name: "T_backup".to_string(),
                    operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    bus_id: EntityId(2),
                    cost_per_mwh: 100.0,
                    min_generation_mw: 0.0,
                    max_generation_mw: 400.0,
                    anticipated_config: None,
                    ..Default::default()
                },
            )])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("build_system: valid constant-productivity study")
    }

    fn build_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::None,
                },
                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                    limit: N_ITERATIONS,
                }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    fn train_final_lb(block_mode: BlockMode) -> f64 {
        let system = build_system(block_mode);
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );
        outcome.result.final_lb
    }

    #[test]
    fn chronological_telescopes_to_parallel_when_interiors_inert() {
        let parallel_lb = train_final_lb(BlockMode::Parallel);
        let chronological_lb = train_final_lb(BlockMode::Chronological);
        // LP-tolerance agreement: the two are different LPs (chronological adds inert
        // interior columns and chained rows), so compare to an absolute-relative
        // tolerance that absorbs HiGHS/CLP FP noise while catching a genuine
        // divergence (a binding interior path would shift the bound far more).
        let tol = 1e-6 * parallel_lb.abs().max(1.0);
        assert!(
            (chronological_lb - parallel_lb).abs() <= tol,
            "chronological bound {chronological_lb} must equal parallel bound \
             {parallel_lb} within LP tolerance {tol} (inert interiors); a divergence \
             signals the interior storage path bound"
        );
    }

    /// A `count == 1` enumerated study (a chain graph, no stochastic branching)
    /// is admitted, trains to `LB == UB`, and the exact probability-weighted
    /// bound over its single path (`w = 1`) equals that path's cost with a zero
    /// CI half-width — the only enumerated case reachable until the all-paths
    /// forward engine lands.
    #[test]
    fn enumerated_count_one_study_reports_exact_bound_with_zero_ci() {
        use cobre_io::config::TrainingSelection;
        use cobre_sddp::forward::{ForwardBound, ForwardResult, sync_forward};

        let system = build_system(BlockMode::Parallel);
        let mut config = build_config();
        config.training.selection = Some(TrainingSelection::Enumerated {});

        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("enumerated count==1 training must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );
        let result = outcome.result;

        let tol = 1e-6 * result.final_ub.abs().max(1.0);
        assert!(
            (result.final_lb - result.final_ub).abs() <= tol,
            "enumerated count==1 must converge LB {} to the exact UB {} within {tol}",
            result.final_lb,
            result.final_ub,
        );

        // The exact reduction over the single trained path reproduces that path's
        // cost as the upper bound, with the CI half-width zeroed under enumeration.
        let local = ForwardResult {
            scenario_costs: vec![result.final_ub],
            elapsed_ms: 0,
            lp_solves: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
            stage_stats: Vec::new(),
        };
        let sync = sync_forward(
            &local,
            &comm,
            1,
            ForwardBound::Exact {
                path_weights: &[1.0],
            },
        )
        .unwrap();
        assert_eq!(sync.global_ub_mean, result.final_ub);
        assert_eq!(sync.ci_95_half_width, 0.0);
    }

    /// Train the parallel-block system under `config` (built deterministically),
    /// panicking on any training error.
    fn train_result(config: &Config) -> cobre_sddp::TrainingResult {
        let mut setup = build_setup_in_code(build_system(BlockMode::Parallel), config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("training must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );
        outcome.result
    }

    fn enumerated_config_with_rules(rules: Vec<StoppingRuleConfig>) -> Config {
        use cobre_io::config::TrainingSelection;
        let mut config = build_config();
        config.training.selection = Some(TrainingSelection::Enumerated {});
        config.training.stopping_rules = Some(rules);
        config
    }

    /// An enumerated + expectation study with a `Gap { tolerance }` rule stops via
    /// the gap rule — before the iteration cap — once the clamped canonical-R$
    /// `UB_exact − LB` first drops to within the tolerance.
    #[test]
    fn enumerated_gap_rule_stops_when_exact_gap_within_tolerance() {
        // Reference run: iteration limit only → converged exact bounds set the scale.
        let ref_result = train_result(&enumerated_config_with_rules(vec![
            StoppingRuleConfig::IterationLimit { limit: 20 },
        ]));
        assert!(ref_result.final_ub.is_finite() && ref_result.final_lb.is_finite());

        // 0.1% of the converged bound: above the LP-tolerance floor (so the gap
        // reaches it) yet a real threshold the exact gap must cross.
        let tolerance = 1e-3 * ref_result.final_ub.abs().max(1.0);
        let result = train_result(&enumerated_config_with_rules(vec![
            StoppingRuleConfig::IterationLimit { limit: 20 },
            StoppingRuleConfig::Gap {
                tolerance: Some(tolerance),
                relative_tolerance: None,
            },
        ]));

        assert_eq!(
            result.reason, "gap",
            "training must stop via the gap rule, not the iteration cap"
        );
        assert!(
            result.iterations < 20,
            "the gap rule must stop before the iteration cap (ran {})",
            result.iterations
        );
        let gap = (result.final_ub - result.final_lb).max(0.0);
        assert!(
            gap <= tolerance,
            "the clamped canonical-R$ gap {gap} must be within tolerance {tolerance}"
        );
    }

    /// A `Gap` rule under sampled forward selection is rejected at setup by the
    /// admission gate — the exact upper bound the gap needs is produced only by
    /// the enumerated engine.
    #[test]
    fn sampled_gap_rule_rejected_at_setup() {
        let mut config = build_config();
        config.training.stopping_rules = Some(vec![
            StoppingRuleConfig::IterationLimit { limit: 5 },
            StoppingRuleConfig::Gap {
                tolerance: Some(1000.0),
                relative_tolerance: None,
            },
        ]);
        let err = try_build_setup_in_code(build_system(BlockMode::Parallel), &config)
            .expect_err("a Gap rule under sampled selection must be rejected");
        match err {
            cobre_sddp::SddpError::Validation(msg) => {
                assert!(msg.contains("gap"), "names the rule: {msg}");
                assert!(msg.contains("sampled"), "names the selection: {msg}");
            }
            other => panic!("expected SddpError::Validation, got {other:?}"),
        }
    }

    /// Train a policy under `train_mode`, write it to a fresh `TempDir` via the
    /// shared `write_checkpoint` writer, and read it back.
    ///
    /// Returns the trained `StudySetup` (its `fcf` is the source of the written
    /// cut records), the read-back checkpoint, and the `TempDir` whose drop
    /// deletes the on-disk policy — kept alive by returning it. Also asserts the
    /// season descriptor round-trips as the absent default: `build_system`
    /// declares no `season_map`.
    fn train_and_checkpoint(
        train_mode: BlockMode,
    ) -> (cobre_sddp::StudySetup, PolicyCheckpoint, TempDir) {
        let config = build_config();
        let mut setup = build_setup_in_code(build_system(train_mode), &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );
        let result = outcome.result;

        // System is not `Clone`; rebuild the identical train-mode system for the
        // checkpoint writer (`build_system` is deterministic).
        let system = build_system(train_mode);
        let policy_dir = TempDir::new().expect("TempDir::new");
        let params = CheckpointParams {
            max_iterations: setup.loop_params.max_iterations,
            forward_passes: setup.loop_params.forward_passes,
            seed: setup.loop_params.seed,
            export_states: config.exports.states,
        };
        write_checkpoint(policy_dir.path(), &setup, &system, &result, &params)
            .expect("write_checkpoint must succeed");
        let checkpoint =
            read_policy_checkpoint(policy_dir.path()).expect("read_policy_checkpoint must succeed");

        assert_eq!(
            checkpoint.metadata.season_manifest.cycle_code,
            cobre_io::SEASON_CYCLE_CODE_ABSENT,
            "build_system declares no season map; the written descriptor must round-trip absent"
        );
        assert_eq!(checkpoint.metadata.season_manifest.n_seasons, 0);
        assert!(checkpoint.metadata.season_manifest.hydro_orders.is_empty());

        (setup, checkpoint, policy_dir)
    }

    /// Assert every checkpoint cut's coefficients and intercept are bit-for-bit
    /// identical to `scale * (the records written from fcf)` (`f64::to_bits`,
    /// never `==`).
    ///
    /// `scale == 1.0` compares raw internal-scaled values (a `checkpoint`
    /// whose cuts were never routed through the canonical-currency export
    /// transform, e.g. `FutureCostFunction::new_with_warm_start` fed
    /// `checkpoint.stage_cuts` directly). `scale ==
    /// setup.inputs.stage_data.stage_templates.cost_scale_factor` compares against a
    /// checkpoint written by [`write_checkpoint`] (`orchestration.rs`), which
    /// multiplies every value by that same factor at export — a single
    /// multiply, so the comparison is exact, not tolerance-based.
    fn assert_cuts_bit_identical(
        fcf: &FutureCostFunction,
        checkpoint: &PolicyCheckpoint,
        scale: f64,
    ) {
        let written = build_stage_cut_records(fcf);
        assert_eq!(
            written.len(),
            checkpoint.stage_cuts.len(),
            "stage count must match between written FCF and read checkpoint"
        );
        for (stage, (stage_written, stage_read)) in
            written.iter().zip(checkpoint.stage_cuts.iter()).enumerate()
        {
            assert_eq!(
                stage_written.len(),
                stage_read.cuts.len(),
                "stage {stage}: cut count must match between written and read"
            );
            for (cut_idx, (w, r)) in stage_written.iter().zip(stage_read.cuts.iter()).enumerate() {
                assert_eq!(
                    (w.intercept * scale).to_bits(),
                    r.intercept.to_bits(),
                    "stage {stage} cut {cut_idx}: intercept bits differ ({} * {scale} vs {})",
                    w.intercept,
                    r.intercept
                );
                assert_eq!(
                    w.coefficients.len(),
                    r.coefficients.len(),
                    "stage {stage} cut {cut_idx}: coefficient length differs"
                );
                for (k, (wc, rc)) in w.coefficients.iter().zip(r.coefficients.iter()).enumerate() {
                    assert_eq!(
                        (wc * scale).to_bits(),
                        rc.to_bits(),
                        "stage {stage} cut {cut_idx} coeff {k}: bits differ ({wc} * {scale} vs {rc})"
                    );
                }
            }
        }
    }

    /// Train in `train_mode`, checkpoint, then load into a `load_mode` study and
    /// evaluate `theta` against the load-mode LP.
    ///
    /// Asserts (1) the written checkpoint holds the trained FCF's cuts scaled
    /// by the writing study's `cost_scale_factor` (canonical
    /// currency units at rest — exact, a single multiply), (2) the cross-mode
    /// warm-start load succeeds, (3) `FutureCostFunction::new_with_warm_start`
    /// (constructor fidelity only, bypassing the load-side rescale this
    /// narrow test does not exercise) copies the checkpoint's raw bytes
    /// unchanged, and (4) a load-mode simulation runs the cross-mode FCF
    /// without error. Only cut bytes are asserted portable; the persisted
    /// basis is column-count-dependent (hence mode-dependent) and is
    /// intentionally not asserted.
    fn assert_cross_mode_load_preserves_cut_bytes(train_mode: BlockMode, load_mode: BlockMode) {
        let (trained_setup, checkpoint, _policy_dir) = train_and_checkpoint(train_mode);
        let cost_scale_factor = trained_setup
            .inputs
            .stage_data
            .stage_templates
            .cost_scale_factor;
        assert_cuts_bit_identical(&trained_setup.fcf, &checkpoint, cost_scale_factor);

        let config = build_config();
        let mut setup2 = build_setup_in_code(build_system(load_mode), &config);

        let proof = cobre_sddp::test_support::trivial_full_fcf_proof(
            checkpoint.stage_cuts[0].state_dimension,
            checkpoint.metadata.num_stages,
        );
        let pool_state_dimensions: Vec<usize> =
            setup2.fcf.pools.iter().map(|p| p.state_dimension).collect();
        let visit_bounds: Vec<u64> = setup2
            .fcf
            .pools
            .iter()
            .map(|p| u64::from(p.visit_stride))
            .collect();
        let warm_fcf = FutureCostFunction::new_with_warm_start(
            &proof,
            &checkpoint.stage_cuts,
            &pool_state_dimensions,
            &visit_bounds,
            setup2.loop_params.forward_passes,
            setup2.loop_params.max_iterations.saturating_add(1),
        )
        .expect("cross-mode warm-start load must succeed (cuts are n_blks-independent)");

        assert_cuts_bit_identical(&warm_fcf, &checkpoint, 1.0);

        setup2.replace_fcf(warm_fcf);
        setup2.simulation_config.n_scenarios = 1;

        let comm = StubComm;
        let mut pool = setup2
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("simulation workspace pool must build");
        let io_capacity = setup2.simulation_config.io_channel_capacity.max(1);
        let (result_tx, result_rx) = std::sync::mpsc::sync_channel(io_capacity);
        let drain = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

        let run = setup2
            .simulate(&mut pool.workspaces, &comm, &result_tx, None, None, &[])
            .expect("cross-mode simulate must succeed (theta evaluates against the load-mode LP)");

        drop(result_tx);
        drop(drain.join().expect("drain thread must not panic"));

        assert_eq!(
            run.costs.len(),
            1,
            "cross-mode simulate must produce one scenario cost, confirming theta evaluated"
        );
    }

    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    #[test]
    fn cross_mode_policy_load_preserves_cut_bytes_parallel_to_chronological() {
        assert_cross_mode_load_preserves_cut_bytes(BlockMode::Parallel, BlockMode::Chronological);
    }

    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    #[test]
    fn cross_mode_policy_load_preserves_cut_bytes_chronological_to_parallel() {
        assert_cross_mode_load_preserves_cut_bytes(BlockMode::Chronological, BlockMode::Parallel);
    }
}

/// A study whose `SeasonMap` and per-season-fitted `InflowModel`s are actually
/// populated, checkpointed through the real producer path
/// ([`orchestration::write_checkpoint`], never `write_policy_checkpoint`
/// directly) and read back — the round trip `chronological_telescoping`'s
/// `train_and_checkpoint` deliberately does not cover, since its fixture
/// declares no season map and always round-trips the absent descriptor.
mod season_descriptor_checkpoint_round_trip {
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, SeasonCycleType, SeasonDefinition,
        SeasonMap, Stage, StageRiskConfig, StageStateConfig,
    };
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, DeficitSegment,
        EntityId, HorizonGraph, HydroBlockBounds, HydroGenerationModel, HydroPenalties,
        HydroStageBounds, HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties,
        NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds,
        ResolvedBounds, ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSelection, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_solver::ActiveSolver;

    use cobre_io::{SEASON_CYCLE_CODE_MONTHLY, read_policy_checkpoint};
    use cobre_sddp::policy::orchestration::{CheckpointParams, write_checkpoint};
    use tempfile::TempDir;

    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage,
    };
    use super::common::{StubComm, build_setup_in_code};

    const N_STAGES: usize = 3;
    const N_ITERATIONS: u32 = 2;
    const N_SEASONS: usize = 12;
    const BUS_ID: i32 = 1;
    const HYDRO_A_ID: i32 = 1;
    const HYDRO_B_ID: i32 = 2;

    // Dense season ordinals the three stages carry (sparse across the 12-month
    // cycle, so most `orders` slots must round-trip as the missing-pair zero).
    const STAGE_SEASON_IDS: [usize; N_STAGES] = [0, 5, 11];

    fn hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 500.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    fn monthly_season_map() -> SeasonMap {
        let seasons = (0..N_SEASONS)
            .map(|id| SeasonDefinition {
                id,
                label: format!("Month{}", id + 1),
                month_start: u32::try_from(id + 1).expect("month fits u32"),
                day_start: None,
                month_end: None,
                day_end: None,
            })
            .collect();
        SeasonMap {
            cycle_type: SeasonCycleType::Monthly,
            seasons,
        }
    }

    /// Per-`(hydro, stage)` AR-coefficient vectors whose lengths are the
    /// per-season orders the round trip must recover: H1 is `[1, 1, 2]` and H2
    /// is `[2, 1, 1]` at dense ordinals `[0, 5, 11]` (`STAGE_SEASON_IDS`), `0`
    /// everywhere else.
    fn ar_coefficients(hydro_id: i32, stage: usize) -> Vec<f64> {
        match (hydro_id, stage) {
            (HYDRO_A_ID, 0) => vec![0.3],
            (HYDRO_A_ID, 1) => vec![0.25],
            (HYDRO_A_ID, 2) => vec![0.2, 0.1],
            (HYDRO_B_ID, 0) => vec![0.4, 0.05],
            (HYDRO_B_ID, 1) => vec![0.35],
            (HYDRO_B_ID, 2) => vec![0.15],
            _ => unreachable!("only two hydros and three stages are fitted"),
        }
    }

    fn build_system() -> cobre_core::System {
        use chrono::NaiveDate;

        let bus = make_bus(
            EntityId(BUS_ID),
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
            },
        );

        let make_fitted_hydro = |id: i32, name: &str| {
            make_hydro(
                EntityId(id),
                HydroSpec {
                    name: name.to_string(),
                    operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    bus_id: EntityId(BUS_ID),
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 300.0,
                    max_turbined_m3s: 100.0,
                    generation_model: HydroGenerationModel::ConstantProductivity,
                    specific_productivity_mw_per_m3s_per_m: Some(0.5),
                    max_generation_mw: 250.0,
                    penalties: hydro_penalties(),
                    ..Default::default()
                },
            )
        };

        let stages: Vec<Stage> = (0..N_STAGES)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: NaiveDate::from_ymd_opt(2024, (i % 12 + 1) as u32, 1).unwrap(),
                        end_date: NaiveDate::from_ymd_opt(2024, ((i % 12 + 1) % 12 + 1) as u32, 1)
                            .unwrap(),
                        season_id: Some(STAGE_SEASON_IDS[i]),
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: 744.0,
                        }],
                        block_mode: BlockMode::Parallel,
                        state_config: StageStateConfig {
                            storage: true,
                            inflow_lags: true,
                        },
                        risk_config: StageRiskConfig::Expectation,
                        scenario_config: ScenarioSourceConfig {
                            branching_factor: 1,
                            noise_method: NoiseMethod::Saa,
                        },
                    },
                )
            })
            .collect();

        let inflow_models: Vec<InflowModel> = [HYDRO_A_ID, HYDRO_B_ID]
            .into_iter()
            .flat_map(|hid| {
                (0..N_STAGES).map(move |i| InflowModel {
                    hydro_id: EntityId(hid),
                    stage_id: i32::try_from(i).expect("stage index fits i32"),
                    mean_m3s: 80.0,
                    std_m3s: 20.0,
                    ar_coefficients: ar_coefficients(hid, i),
                    residual_std_ratio: 0.8,
                    annual: None,
                })
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..N_STAGES)
            .map(|i| LoadModel {
                bus_id: EntityId(BUS_ID),
                stage_id: i32::try_from(i).expect("stage index fits i32"),
                mean_mw: 50.0,
                std_mw: 0.0,
            })
            .collect();

        let default_hydro_bounds = || HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 300.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        };
        let default_hydro_bounds_block = || HydroBlockBounds {
            max_turbined_m3s: 100.0,
            max_generation_mw: 250.0,
            ..Default::default()
        };

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_bounds_block(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 2,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let initial_conditions = InitialConditions {
            storage: vec![
                HydroStorage {
                    hydro_id: EntityId(HYDRO_A_ID),
                    value_hm3: 100.0,
                },
                HydroStorage {
                    hydro_id: EntityId(HYDRO_B_ID),
                    value_hm3: 100.0,
                },
            ],
            filling_storage: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![
                make_fitted_hydro(HYDRO_A_ID, "H1"),
                make_fitted_hydro(HYDRO_B_ID, "H2"),
            ])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .policy_graph(HorizonGraph {
                season_map: Some(monthly_season_map()),
                ..HorizonGraph::default()
            })
            .build()
            .expect("build_system: valid two-hydro season-fitted study")
    }

    fn build_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::None,
                },
                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                    limit: N_ITERATIONS,
                }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    /// A study with a monthly `SeasonMap` and hydros fitted at per-season
    /// orders, checkpointed through the real producer path and read back,
    /// carries `n_seasons == 12` and each hydro's `orders` vector at the
    /// study's dense season ordinals.
    #[test]
    fn checkpoint_round_trip_carries_the_studys_populated_season_descriptor() {
        let config = build_config();
        let mut setup = build_setup_in_code(build_system(), &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );
        let result = outcome.result;

        // System is not `Clone`; rebuild the identical study for the checkpoint
        // writer (`build_system` is deterministic).
        let system = build_system();
        let policy_dir = TempDir::new().expect("TempDir::new");
        let params = CheckpointParams {
            max_iterations: setup.loop_params.max_iterations,
            forward_passes: setup.loop_params.forward_passes,
            seed: setup.loop_params.seed,
            export_states: config.exports.states,
        };
        write_checkpoint(policy_dir.path(), &setup, &system, &result, &params)
            .expect("write_checkpoint must succeed");
        let checkpoint =
            read_policy_checkpoint(policy_dir.path()).expect("read_policy_checkpoint must succeed");

        let manifest = &checkpoint.metadata.season_manifest;
        assert_eq!(
            manifest.cycle_code, SEASON_CYCLE_CODE_MONTHLY,
            "the study declares a monthly SeasonMap"
        );
        assert_eq!(manifest.n_seasons, N_SEASONS as u32);
        assert_eq!(
            manifest.hydro_orders.len(),
            2,
            "both fitted hydros must carry an orders vector"
        );

        assert_eq!(manifest.hydro_orders[0].hydro_id, HYDRO_A_ID);
        assert_eq!(manifest.hydro_orders[1].hydro_id, HYDRO_B_ID);

        let mut expected_a = vec![0_u32; N_SEASONS];
        expected_a[STAGE_SEASON_IDS[0]] = 1;
        expected_a[STAGE_SEASON_IDS[1]] = 1;
        expected_a[STAGE_SEASON_IDS[2]] = 2;
        assert_eq!(
            manifest.hydro_orders[0].orders, expected_a,
            "H1's orders must equal its ar_coefficients length at each dense season ordinal, 0 elsewhere"
        );

        let mut expected_b = vec![0_u32; N_SEASONS];
        expected_b[STAGE_SEASON_IDS[0]] = 2;
        expected_b[STAGE_SEASON_IDS[1]] = 1;
        expected_b[STAGE_SEASON_IDS[2]] = 1;
        assert_eq!(
            manifest.hydro_orders[1].orders, expected_b,
            "H2's orders must equal its ar_coefficients length at each dense season ordinal, 0 elsewhere"
        );
    }
}

/// A study with a monthly `SeasonMap` and inflow-lag state, checkpointed
/// through the real producer path (`orchestration::write_checkpoint`) and
/// read back through the real consumer path (`load_boundary_cuts` with
/// `with_study_seasons`): the producer's own descriptor and the consumer's
/// own gate must agree end to end, never merely on an in-memory
/// `SeasonManifest` the test builds itself.
mod boundary_season_gate_round_trip {
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, SeasonCycleType, SeasonDefinition,
        SeasonMap, Stage, StageRiskConfig, StageStateConfig,
    };
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, DeficitSegment,
        EntityId, HorizonGraph, HydroBlockBounds, HydroGenerationModel, HydroPenalties,
        HydroStageBounds, HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties,
        NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds,
        ResolvedBounds, ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSelection, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_solver::ActiveSolver;

    use cobre_sddp::policy::orchestration::{
        CheckpointParams, build_season_manifest, write_checkpoint,
    };
    use cobre_sddp::{BoundaryLoadRequest, load_boundary_cuts, study_horizon_end};
    use tempfile::TempDir;

    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage,
    };
    use super::common::{StubComm, build_setup_in_code};

    const N_STAGES: usize = 3;
    const N_ITERATIONS: u32 = 2;
    const N_SEASONS: usize = 12;
    const BUS_ID: i32 = 1;
    const HYDRO_ID: i32 = 1;

    // Dense season ordinals the three stages carry; the terminal stage (index
    // 2) lands on season 11 and carries a fitted 2-lag AR model, so the
    // terminal manifest's inflow-lag family is non-empty.
    const STAGE_SEASON_IDS: [usize; N_STAGES] = [0, 5, 11];

    fn hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 500.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    fn monthly_season_map() -> SeasonMap {
        let seasons = (0..N_SEASONS)
            .map(|id| SeasonDefinition {
                id,
                label: format!("Month{}", id + 1),
                month_start: u32::try_from(id + 1).expect("month fits u32"),
                day_start: None,
                month_end: None,
                day_end: None,
            })
            .collect();
        SeasonMap {
            cycle_type: SeasonCycleType::Monthly,
            seasons,
        }
    }

    /// The fitted AR-coefficient vector at each stage: order 1 at the first
    /// two stages, order 2 at the terminal stage (index 2, season 11).
    fn ar_coefficients(stage: usize) -> Vec<f64> {
        match stage {
            0 => vec![0.3],
            1 => vec![0.25],
            2 => vec![0.2, 0.1],
            _ => unreachable!("only three stages are fitted"),
        }
    }

    fn build_system() -> cobre_core::System {
        use chrono::NaiveDate;

        let bus = make_bus(
            EntityId(BUS_ID),
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
            },
        );

        let hydro = make_hydro(
            EntityId(HYDRO_ID),
            HydroSpec {
                name: "H1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(BUS_ID),
                min_storage_hm3: 0.0,
                max_storage_hm3: 300.0,
                max_turbined_m3s: 100.0,
                generation_model: HydroGenerationModel::ConstantProductivity,
                specific_productivity_mw_per_m3s_per_m: Some(0.5),
                max_generation_mw: 250.0,
                penalties: hydro_penalties(),
                ..Default::default()
            },
        );

        let stages: Vec<Stage> = (0..N_STAGES)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: NaiveDate::from_ymd_opt(2024, (i % 12 + 1) as u32, 1).unwrap(),
                        end_date: NaiveDate::from_ymd_opt(2024, ((i % 12 + 1) % 12 + 1) as u32, 1)
                            .unwrap(),
                        season_id: Some(STAGE_SEASON_IDS[i]),
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: 744.0,
                        }],
                        block_mode: BlockMode::Parallel,
                        state_config: StageStateConfig {
                            storage: true,
                            inflow_lags: true,
                        },
                        risk_config: StageRiskConfig::Expectation,
                        scenario_config: ScenarioSourceConfig {
                            branching_factor: 1,
                            noise_method: NoiseMethod::Saa,
                        },
                    },
                )
            })
            .collect();

        let inflow_models: Vec<InflowModel> = (0..N_STAGES)
            .map(|i| InflowModel {
                hydro_id: EntityId(HYDRO_ID),
                stage_id: i32::try_from(i).expect("stage index fits i32"),
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: ar_coefficients(i),
                residual_std_ratio: 0.8,
                annual: None,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..N_STAGES)
            .map(|i| LoadModel {
                bus_id: EntityId(BUS_ID),
                stage_id: i32::try_from(i).expect("stage index fits i32"),
                mean_mw: 50.0,
                std_mw: 0.0,
            })
            .collect();

        let default_hydro_bounds = || HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 300.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        };
        let default_hydro_bounds_block = || HydroBlockBounds {
            max_turbined_m3s: 100.0,
            max_generation_mw: 250.0,
            ..Default::default()
        };

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_bounds_block(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(HYDRO_ID),
                value_hm3: 100.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .policy_graph(HorizonGraph {
                season_map: Some(monthly_season_map()),
                ..HorizonGraph::default()
            })
            .build()
            .expect("build_system: valid one-hydro season-fitted study")
    }

    fn build_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::None,
                },
                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                    limit: N_ITERATIONS,
                }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    /// Trains one study, writes its checkpoint through the real producer
    /// (`orchestration::write_checkpoint`), then loads it back through the
    /// real consumer (`load_boundary_cuts`) gated on
    /// `with_study_seasons(&build_season_manifest(&system))`: the load
    /// succeeds and the report's `inflow_lag.copy` tally is non-zero — the
    /// producer's own descriptor and the consumer's own gate agree.
    #[test]
    fn boundary_load_accepts_the_studys_own_checkpoint_under_its_own_season_gate() {
        let config = build_config();
        let mut setup = build_setup_in_code(build_system(), &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );
        let result = outcome.result;

        // System is not `Clone`; rebuild the identical study for the checkpoint
        // writer and the boundary-load accessors below (`build_system` is
        // deterministic).
        let system = build_system();
        let policy_dir = TempDir::new().expect("TempDir::new");
        let params = CheckpointParams {
            max_iterations: setup.loop_params.max_iterations,
            forward_passes: setup.loop_params.forward_passes,
            seed: setup.loop_params.seed,
            export_states: config.exports.states,
        };
        write_checkpoint(policy_dir.path(), &setup, &system, &result, &params)
            .expect("write_checkpoint must succeed");

        let state_dim = setup.fcf.state_dimension as u32;
        let current_manifest = setup.build_terminal_entity_manifest(&system);
        let boundary_date =
            study_horizon_end(&system).expect("a non-negative stage exists in this study");
        let study_seasons = build_season_manifest(&system);

        let validated = load_boundary_cuts(
            &BoundaryLoadRequest::new(
                policy_dir.path(),
                boundary_date,
                state_dim,
                &current_manifest,
                setup.inputs.stage_data.stage_templates.cost_scale_factor,
            )
            .with_study_seasons(&study_seasons),
        )
        .expect(
            "the study's own checkpoint must reconcile against its own season descriptor and \
             terminal manifest",
        );

        assert!(
            validated.report().inflow_lag.copy > 0,
            "the terminal inflow-lag slot must reconcile by identity Copy: {:?}",
            validated.report()
        );
    }
}

/// A cascade whose upstream hydro declares a travel-time arc into its
/// downstream hydro, checkpointed through the real producer path
/// ([`orchestration::write_checkpoint`], never `write_policy_checkpoint`
/// directly) and read back: the transit-bucket manifest slot's arrival
/// interval must match on the checkpoint [`cobre_io::read_policy_checkpoint`]
/// actually parses, not merely on the in-memory manifest
/// `build_stage_entity_manifest` produces.
mod transit_arrival_interval_checkpoint_round_trip {
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, DeficitSegment,
        EntityId, HydroBlockBounds, HydroGenerationModel, HydroPenalties, HydroStageBounds,
        HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PostStudyStage, PostStudyStages,
        PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder, ThermalBlockBounds,
        ThermalStageBounds,
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSelection, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_solver::ActiveSolver;

    use cobre_io::{StageCutsReadResult, StateFamily, read_policy_checkpoint};
    use cobre_sddp::policy::orchestration::{CheckpointParams, write_checkpoint};
    use tempfile::TempDir;

    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage,
    };
    use super::common::{StubComm, build_setup_in_code};

    const N_STAGES: usize = 3;
    const N_ITERATIONS: u32 = 1;
    const BUS_ID: i32 = 1;
    const HYDRO_A_ID: i32 = 1;
    const HYDRO_B_ID: i32 = 2;
    const TRAVEL_TIME_HOURS: f64 = 744.0;

    fn hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 500.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    fn build_system() -> cobre_core::System {
        use chrono::NaiveDate;

        let bus = make_bus(
            EntityId(BUS_ID),
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
            },
        );

        let make_cascade_hydro = |id: i32, name: &str, downstream_id: Option<EntityId>| {
            make_hydro(
                EntityId(id),
                HydroSpec {
                    name: name.to_string(),
                    operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    bus_id: EntityId(BUS_ID),
                    downstream_id,
                    travel_time_hours: downstream_id.map(|_| TRAVEL_TIME_HOURS),
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 300.0,
                    max_turbined_m3s: 100.0,
                    generation_model: HydroGenerationModel::ConstantProductivity,
                    specific_productivity_mw_per_m3s_per_m: Some(0.5),
                    max_generation_mw: 250.0,
                    penalties: hydro_penalties(),
                    ..Default::default()
                },
            )
        };

        let stages: Vec<Stage> = (0..N_STAGES)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: NaiveDate::from_ymd_opt(2024, (i % 12 + 1) as u32, 1).unwrap(),
                        end_date: NaiveDate::from_ymd_opt(2024, ((i % 12 + 1) % 12 + 1) as u32, 1)
                            .unwrap(),
                        season_id: Some(0),
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: TRAVEL_TIME_HOURS,
                        }],
                        block_mode: BlockMode::Parallel,
                        state_config: StageStateConfig {
                            storage: true,
                            inflow_lags: true,
                        },
                        risk_config: StageRiskConfig::Expectation,
                        scenario_config: ScenarioSourceConfig {
                            branching_factor: 1,
                            noise_method: NoiseMethod::Saa,
                        },
                    },
                )
            })
            .collect();

        let inflow_models: Vec<InflowModel> = [HYDRO_A_ID, HYDRO_B_ID]
            .into_iter()
            .flat_map(|hid| {
                (0..N_STAGES).map(move |i| InflowModel {
                    hydro_id: EntityId(hid),
                    stage_id: i32::try_from(i).expect("stage index fits i32"),
                    mean_m3s: 80.0,
                    std_m3s: 20.0,
                    ar_coefficients: vec![0.2],
                    residual_std_ratio: 0.8,
                    annual: None,
                })
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..N_STAGES)
            .map(|i| LoadModel {
                bus_id: EntityId(BUS_ID),
                stage_id: i32::try_from(i).expect("stage index fits i32"),
                mean_mw: 50.0,
                std_mw: 0.0,
            })
            .collect();

        let default_hydro_bounds = || HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 300.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        };
        let default_hydro_bounds_block = || HydroBlockBounds {
            max_turbined_m3s: 100.0,
            max_generation_mw: 250.0,
            ..Default::default()
        };

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_bounds_block(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 2,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let initial_conditions = InitialConditions {
            storage: vec![
                HydroStorage {
                    hydro_id: EntityId(HYDRO_A_ID),
                    value_hm3: 100.0,
                },
                HydroStorage {
                    hydro_id: EntityId(HYDRO_B_ID),
                    value_hm3: 100.0,
                },
            ],
            filling_storage: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        let post_study_stages = PostStudyStages {
            stages: vec![PostStudyStage {
                start_date: NaiveDate::from_ymd_opt(2024, 4, 1).unwrap(),
                duration_hours: 720.0,
            }],
            thermal_bounds: Vec::new(),
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![
                make_cascade_hydro(HYDRO_A_ID, "H1", Some(EntityId(HYDRO_B_ID))),
                make_cascade_hydro(HYDRO_B_ID, "H2", None),
            ])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .post_study_stages(Some(post_study_stages))
            .build()
            .expect("build_system: valid two-hydro cascade with a declared post-study calendar")
    }

    fn build_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::None,
                },
                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                    limit: N_ITERATIONS,
                }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    /// The lag-1 transit-bucket slot on the pool named by `stage_id`, read
    /// back from the checkpoint the real reader parses.
    fn lag1_bucket_slot(pool: &StageCutsReadResult) -> &cobre_io::EntitySlot {
        pool.entity_manifest
            .iter()
            .find(|slot| {
                slot.entity_type == StateFamily::HydroTransitBucket.code() && slot.subindex == 1
            })
            .unwrap_or_else(|| {
                panic!(
                    "pool {} must carry a lag-1 transit-bucket slot",
                    pool.stage_id
                )
            })
    }

    /// End-to-end: the arrival-stage resolver (`delivery_stages[current_stage_idx
    /// + lag]`) survives training, [`orchestration::write_checkpoint`], and
    /// [`cobre_io::read_policy_checkpoint`] byte-for-byte. At the last study
    /// stage (`t == 2`), lag 1 resolves to the declared post-study stage
    /// (`2024-04-01`, `720.0` hours -> `[2024-04-01, 2024-05-01)`); at stage 0,
    /// the same slot resolves to study stage 1's own `[2024-02-01, 2024-03-01)`.
    #[test]
    fn transit_arrival_interval_checkpoint_round_trip() {
        let config = build_config();
        let mut setup = build_setup_in_code(build_system(), &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );
        let result = outcome.result;

        // System is not `Clone`; rebuild the identical study for the checkpoint
        // writer (`build_system` is deterministic).
        let system = build_system();
        let policy_dir = TempDir::new().expect("TempDir::new");
        let params = CheckpointParams {
            max_iterations: setup.loop_params.max_iterations,
            forward_passes: setup.loop_params.forward_passes,
            seed: setup.loop_params.seed,
            export_states: config.exports.states,
        };
        write_checkpoint(policy_dir.path(), &setup, &system, &result, &params)
            .expect("write_checkpoint must succeed");
        let checkpoint =
            read_policy_checkpoint(policy_dir.path()).expect("read_policy_checkpoint must succeed");

        assert_eq!(
            checkpoint.stage_cuts.len(),
            N_STAGES,
            "one pool per stage on this non-branching chain"
        );

        let last_stage_pool = checkpoint
            .stage_cuts
            .iter()
            .find(|pool| pool.stage_id == (N_STAGES - 1) as u32)
            .expect("the last study stage must own a pool");
        let last_stage_slot = lag1_bucket_slot(last_stage_pool);
        assert_eq!(
            last_stage_slot.interval_start, 20_240_401,
            "lag 1 from the last study stage resolves to the declared post-study stage"
        );
        assert_eq!(last_stage_slot.interval_end, 20_240_501);

        let stage0_pool = checkpoint
            .stage_cuts
            .iter()
            .find(|pool| pool.stage_id == 0)
            .expect("stage 0 must own a pool");
        let stage0_slot = lag1_bucket_slot(stage0_pool);
        assert_eq!(
            stage0_slot.interval_start, 20_240_201,
            "lag 1 from stage 0 resolves to study stage 1's own interval"
        );
        assert_eq!(stage0_slot.interval_end, 20_240_301);
    }
}

/// Chronological block-resolved attribution for a declared travel-time arc: the
/// resolver's block tables, the `K = 1` chronological-vs-parallel byte-identity
/// anchor, and the parallel-vs-chronological state-dimension equality
/// (mode-independent sizing, the shared-density aggregation identity, and the
/// fixed-delivery-density contract).
mod chronological_attribution {
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{Block, BlockMode, Stage};
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, DeficitSegment,
        EntityId, HydroBlockBounds, HydroGenerationModel, HydroPenalties, HydroStageBounds,
        HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::TrainingSelection;
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_sddp::lead_time::resolve_spread;
    use cobre_sddp::test_support::assert_all_templates_byte_identical;

    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };

    const U_ID: i32 = 1;
    const J_ID: i32 = 2;
    const BUS_ID: i32 = 3;
    const THERMAL_ID: i32 = 4;
    const TRAVEL_TIME_HOURS: f64 = 250.0;
    const N_STAGES: usize = 2;
    const TOL: f64 = 1e-9;

    fn assert_close(actual: &[f64], expected: &[f64], label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: length mismatch, got {actual:?}, expected {expected:?}"
        );
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() < TOL,
                "{label}: value mismatch, got {actual:?}, expected {expected:?}"
            );
        }
    }

    fn zero_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 0.0,
        }
    }

    fn zero_hydro_stage_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 0.0,
        }
    }

    /// Cascade `U -> J` (`U` declares `travel_time_hours`) on one bus with a
    /// backup thermal, `N_STAGES` stages: stage 0 carries `stage0_blocks` under
    /// `block_mode` (the anchor under test); stage 1 is a single default-length
    /// receiving stage for whatever lag the anchor's arrival window reaches.
    fn build_system(block_mode: BlockMode, stage0_blocks: Vec<Block>) -> cobre_core::System {
        use chrono::NaiveDate;

        let bus = make_bus(
            EntityId(BUS_ID),
            BusSpec {
                name: "B0".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 1000.0,
                }],
                excess_cost: 0.0,
            },
        );

        let u = make_hydro(
            EntityId(U_ID),
            HydroSpec {
                name: "U".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
                bus_id: EntityId(BUS_ID),
                downstream_id: Some(EntityId(J_ID)),
                travel_time_hours: Some(TRAVEL_TIME_HOURS),
                min_storage_hm3: 0.0,
                max_storage_hm3: 150.0,
                max_turbined_m3s: 100.0,
                generation_model: HydroGenerationModel::ConstantProductivity,
                max_generation_mw: 10.0,
                penalties: zero_hydro_penalties(),
                ..Default::default()
            },
        );

        let j = make_hydro(
            EntityId(J_ID),
            HydroSpec {
                name: "J".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2020, 1, 2).unwrap(),
                bus_id: EntityId(BUS_ID),
                downstream_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 0.0,
                max_turbined_m3s: 100.0,
                generation_model: HydroGenerationModel::ConstantProductivity,
                max_generation_mw: 10.0,
                penalties: zero_hydro_penalties(),
                ..Default::default()
            },
        );

        let receiving_stage_blocks = vec![Block {
            index: 0,
            name: "SINGLE".to_string(),
            duration_hours: 720.0,
        }];
        let stages: Vec<Stage> = vec![
            make_stage(
                0,
                StageSpec {
                    start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                    blocks: stage0_blocks,
                    block_mode,
                    ..StageSpec::default()
                },
            ),
            make_stage(
                1,
                StageSpec {
                    start_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
                    blocks: receiving_stage_blocks,
                    block_mode: BlockMode::Parallel,
                    ..StageSpec::default()
                },
            ),
        ];

        let inflow_models: Vec<InflowModel> = [U_ID, J_ID]
            .into_iter()
            .flat_map(|hydro_id| {
                (0..N_STAGES).map(move |stage_id| InflowModel {
                    hydro_id: EntityId(hydro_id),
                    stage_id: i32::try_from(stage_id).expect("stage index fits i32"),
                    mean_m3s: 0.0,
                    std_m3s: 0.0,
                    ar_coefficients: vec![],
                    residual_std_ratio: 1.0,
                    annual: None,
                })
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..N_STAGES)
            .map(|stage_id| LoadModel {
                bus_id: EntityId(BUS_ID),
                stage_id: i32::try_from(stage_id).expect("stage index fits i32"),
                mean_mw: 1.0,
                std_mw: 0.0,
            })
            .collect();

        let default_hydro_bounds = || HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 150.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        };
        let default_hydro_bounds_block = || HydroBlockBounds {
            max_turbined_m3s: 100.0,
            max_generation_mw: 10.0,
            ..Default::default()
        };

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_bounds_block(),
                thermal: ThermalStageBounds { cost_per_mwh: 10.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 2,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: zero_hydro_stage_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let initial_conditions = InitialConditions {
            storage: vec![
                HydroStorage {
                    hydro_id: EntityId(U_ID),
                    value_hm3: 100.0,
                },
                HydroStorage {
                    hydro_id: EntityId(J_ID),
                    value_hm3: 0.0,
                },
            ],
            filling_storage: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            // Empty is safe: `build_initial_transit_bucket_state`'s history selection
            // falls back to an empty slice (seed 0.0) rather than panicking —
            // these tests inspect LP structure and state dimension, never a
            // seeded value.
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![make_thermal(
                EntityId(THERMAL_ID),
                ThermalSpec {
                    name: "T0".to_string(),
                    operational_start_date: NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
                    bus_id: EntityId(BUS_ID),
                    cost_per_mwh: 10.0,
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                    anticipated_config: None,
                    ..Default::default()
                },
            )])
            .hydros(vec![u, j])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("build_system: valid cascade with a declared travel-time arc")
    }

    fn build_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::None,
                },
                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    fn build_setup(block_mode: BlockMode, stage0_blocks: Vec<Block>) -> cobre_sddp::StudySetup {
        let system = build_system(block_mode, stage0_blocks);
        let config = build_config();
        build_setup_in_code(system, &config)
    }

    fn single_block(name: &str, duration_hours: f64) -> Vec<Block> {
        vec![Block {
            index: 0,
            name: name.to_string(),
            duration_hours,
        }]
    }

    fn three_chronological_blocks() -> Vec<Block> {
        vec![
            Block {
                index: 0,
                name: "B0".to_string(),
                duration_hours: 240.0,
            },
            Block {
                index: 1,
                name: "B1".to_string(),
                duration_hours: 240.0,
            },
            Block {
                index: 2,
                name: "B2".to_string(),
                duration_hours: 240.0,
            },
        ]
    }

    /// The monthly (720 h) anchor's block-resolved attribution over 3x240 h
    /// chronological blocks with `travel_time_hours = 250`, reproducing the
    /// resolver's own pinned reference values (`lead_time::tests::
    /// test_example_iii_block_factors`) directly — this test's only job is to
    /// confirm the same reference numbers hold for the exact stage lengths
    /// `d46-travel-time-chronological` declares.
    #[test]
    fn resolve_spread_matches_reference_block_tables() {
        let stage_lengths_hours = [720.0, 720.0];
        let block_lengths_hours = [240.0, 240.0, 240.0];
        let resolution = resolve_spread(
            TRAVEL_TIME_HOURS,
            0,
            &stage_lengths_hours,
            Some(&block_lengths_hours),
        );

        assert_eq!(resolution.stage_reach, 1, "depth must be 1 (k_0, k_1 only)");
        assert_close(
            &resolution.stage_weights,
            &[470.0 / 720.0, 250.0 / 720.0],
            "stage_weights",
        );

        assert_close(
            &resolution.block_deposits[0],
            &[1.0, 0.0],
            "block_deposits[0] (B0)",
        );
        assert_close(
            &resolution.block_deposits[1],
            &[230.0 / 240.0, 10.0 / 240.0],
            "block_deposits[1] (B1)",
        );
        assert_close(
            &resolution.block_deposits[2],
            &[0.0, 1.0],
            "block_deposits[2] (B2)",
        );

        // within_stage_routing[b] is self-inclusive (index 0 = block b
        // routing to itself), so within_stage_routing_{B0->B1} =
        // within_stage_routing[0][1], within_stage_routing_{B0->B2} =
        // within_stage_routing[0][2], within_stage_routing_{B1->B2} =
        // within_stage_routing[1][1].
        assert_close(
            &resolution.within_stage_routing[0],
            &[0.0, 230.0 / 240.0, 10.0 / 240.0],
            "within_stage_routing[0] (B0 self-inclusive)",
        );
        assert_close(
            &resolution.within_stage_routing[1],
            &[0.0, 230.0 / 240.0],
            "within_stage_routing[1] (B1 self-inclusive)",
        );
        assert_close(
            &resolution.within_stage_routing[2],
            &[0.0],
            "within_stage_routing[2] (B2 self-inclusive)",
        );

        assert_eq!(resolution.arrival_density.len(), 1);
        assert_close(
            &resolution.arrival_density[0],
            &[240.0 / 250.0, 10.0 / 250.0],
            "arrival_density (96%/4% split)",
        );
    }

    /// `K = 1` chronological (single 720 h block) must be byte-identical to
    /// the parallel build, WITH the travel-time arc declared (`χ_{0,d} = k_d`
    /// — the fixed-delivery-density contract). A single chronological block
    /// has no interior routing to diverge on, so every stage template must
    /// match the parallel LP exactly, bit for bit.
    #[test]
    fn k1_chronological_byte_identical_to_parallel_with_travel_time_on() {
        let parallel = build_setup(BlockMode::Parallel, single_block("B0", 720.0));
        let chronological = build_setup(BlockMode::Chronological, single_block("B0", 720.0));

        let parallel_templates = &parallel.inputs.stage_data.stage_templates.templates;
        let chrono_templates = &chronological.inputs.stage_data.stage_templates.templates;
        assert_all_templates_byte_identical(
            parallel_templates,
            chrono_templates,
            "parallel vs chronological (travel time on)",
        );
    }

    /// Mode-independent sizing: the bucket state is a pure function of stage
    /// lengths, never of `n_blks`/`block_mode`. Builds the
    /// SAME cascade in parallel (`K = 1`) and chronological (`K = 3`) mode and
    /// asserts `state.n_state`/`state.n_buckets` are equal — a state dimension
    /// that instead tracked `n_blks` would silently misalign the trial-state
    /// broadcast and the cut coefficients between the two modes.
    #[test]
    fn state_dimension_equal_across_parallel_and_chronological_with_travel_time_on() {
        let parallel = build_setup(BlockMode::Parallel, single_block("B0", 720.0));
        let chronological = build_setup(BlockMode::Chronological, three_chronological_blocks());

        let parallel_state = parallel.stage_state();
        let chronological_state = chronological.stage_state();

        assert!(
            parallel_state.n_buckets > 0,
            "the declared arc must actually size a bucket dimension"
        );
        assert_eq!(
            parallel_state.n_buckets, chronological_state.n_buckets,
            "n_buckets must not depend on block_mode/n_blks"
        );
        assert_eq!(
            parallel_state.n_state, chronological_state.n_state,
            "n_state must not depend on block_mode/n_blks"
        );
    }
}

/// End-to-end regression: an FPHA hydro's stage-specific `equivalent_productivity`
/// (`rho_eq`) override must resolve by the study's DOMAIN `Stage::id`, never by the
/// 0-based study position. The two coincide on every other deterministic fixture (all
/// declare 0-based, contiguous-from-zero stage ids), so only a fixture whose domain
/// ids are offset from position — here: 10, 11, 12 at positions 0, 1, 2 — can surface
/// a position-keyed read. The fixture also carries VHA geometry + `rho_esp` for the
/// FPHA hydro, so a mis-keyed lookup falls through to a silently WRONG derived
/// coefficient rather than the loud `FphaMissingEquivalentProductivity` error — the
/// hazard this module pins.
mod nonzero_stage_fpha_override_regression {
    use cobre_io::config::SimulationSelection;
    use std::path::{Path, PathBuf};

    use cobre_core::scenario::ScenarioSource;
    use cobre_core::{EntityId, StageId};
    use cobre_io::HydroEnergyProductivityRow;
    use cobre_sddp::energy_conversion::build_hydro_energy_productivity_override;
    use cobre_sddp::hydro_models::prepare_hydro_models;
    use cobre_sddp::setup::prepare_stochastic;
    use cobre_sddp::{SimulationHydroResult, SimulationScenarioResult, StudySetup};

    use super::common::parity_hash::compute_parity_hash;
    use super::common::permute::permute_case;
    use super::common::{build_setup_for_case, run_simulation};

    /// Fixed seed for this fixture's fast, non-golden default-CI declaration-
    /// order-invariance probe; the full seeded-shuffle matrix lives in
    /// `tests/parity.rs`'s `shuffle_matrix_<case>` tests.
    const PERMUTATION_SEED: u64 = 20_260_711;

    /// Domain stage id (`Stage::id`) carrying the override row, and the corresponding
    /// 0-based study position — see `stages.json` / `hydro_energy_productivity.parquet`
    /// under [`case_dir`]. Position 1 is deliberately NOT equal to domain id 11: that
    /// gap is what a position-keyed lookup would miss.
    const OVERRIDE_DOMAIN_STAGE_ID: i32 = 11;
    const OVERRIDE_STAGE_POSITION: u32 = 1;
    const OVERRIDE_RHO_EQ: f64 = 4.2;
    const FPHA_HYDRO_ID: i32 = 0;

    fn case_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/nonzero_stage_fpha_override")
    }

    /// Return hydro `hydro_id`'s result at study `position`. The stage results are
    /// emitted in ascending study order, so the position is the ordinal index —
    /// NOT the `stage_id` column value, which now carries the (non-0-based) domain
    /// id after the output-axis re-key. The fixture trains and simulates exactly
    /// one scenario, so the flattened stage list is that single scenario's ordered
    /// stages.
    fn hydro_result_at(
        scenario_results: &[SimulationScenarioResult],
        position: u32,
        hydro_id: i32,
    ) -> &SimulationHydroResult {
        scenario_results
            .iter()
            .flat_map(|s| &s.stages)
            .nth(position as usize)
            .unwrap_or_else(|| panic!("no stage at position {position}"))
            .hydros
            .iter()
            .find(|h| h.hydro_id == hydro_id)
            .unwrap_or_else(|| panic!("no hydro {hydro_id} result at stage position {position}"))
    }

    /// The override's domain stage resolves the override value; its un-overridden
    /// neighbors resolve the SAME VHA+`rho_esp`-derived value, which differs from the
    /// override — the distinguishing-value pattern the fixture is built to produce
    /// (`[derived, OVERRIDE_RHO_EQ, derived]` across positions 0, 1, 2).
    ///
    /// Negative control (documented here per the discrimination requirement, not
    /// executed): reverting the domain-`StageId`-keyed accessors to a position-keyed
    /// `usize` makes `equivalent_productivity(hydro, stage)` search the override table
    /// for `stage_id == Some(1)` (the study position) instead of `Some(11)` (the domain
    /// id). No row matches position 1, so the lookup would fall through to the
    /// VHA+`rho_esp` derivation exactly like positions 0 and 2 — the `assert_ne`-style
    /// distinctness check below would then compare two equal `derived` values and fail.
    /// [`position_keyed_lookup_misses_the_domain_keyed_override_row`] demonstrates that
    /// exact miss directly, through the same public accessor, without reverting any
    /// production code.
    #[test]
    fn fpha_override_resolves_by_domain_stage_id_end_to_end() {
        let dir = case_dir();
        let (_result, scenario_results, _summary) = super::run_with_simulation(&dir);

        // The output re-key stamps the DOMAIN stage id (not the 0-based position)
        // into every result row: the stage at study position 1 carries stage_id 11.
        let stage_at_override = scenario_results
            .iter()
            .flat_map(|s| &s.stages)
            .nth(OVERRIDE_STAGE_POSITION as usize)
            .expect("override stage present");
        #[allow(clippy::cast_possible_wrap)]
        {
            assert_eq!(
                stage_at_override.stage_id as i32, OVERRIDE_DOMAIN_STAGE_ID,
                "output stage_id must carry the declared domain id, not the study position"
            );
        }

        let at_override =
            hydro_result_at(&scenario_results, OVERRIDE_STAGE_POSITION, FPHA_HYDRO_ID);
        assert_eq!(
            at_override.equivalent_productivity_mw_per_m3s.to_bits(),
            OVERRIDE_RHO_EQ.to_bits(),
            "domain stage_id={OVERRIDE_DOMAIN_STAGE_ID} (position {OVERRIDE_STAGE_POSITION}) \
             must read the override value exactly"
        );

        let before =
            hydro_result_at(&scenario_results, 0, FPHA_HYDRO_ID).equivalent_productivity_mw_per_m3s;
        let after =
            hydro_result_at(&scenario_results, 2, FPHA_HYDRO_ID).equivalent_productivity_mw_per_m3s;
        assert_eq!(
            before.to_bits(),
            after.to_bits(),
            "both un-overridden neighbors must read the SAME VHA+rho_esp-derived value"
        );
        assert!(
            (before - OVERRIDE_RHO_EQ).abs() > 0.5,
            "the derived neighbor value ({before}) must be clearly distinct from the \
             override ({OVERRIDE_RHO_EQ}) — otherwise a position-keyed miss would be \
             indistinguishable from a correct domain-keyed hit"
        );
    }

    /// Direct discrimination through the public accessor
    /// `HydroEnergyProductivityOverride::equivalent_productivity` (never reverting the
    /// production fix): the SAME override row resolves when keyed by its domain
    /// `StageId(11)` and misses when keyed by the study POSITION `StageId(1)` — the read
    /// a position-keyed accessor would have performed pre-fix. This is the negative
    /// control: a `StageId(1)` key returning `Some(_)` here would mean the accessor no
    /// longer discriminates domain id from position, and
    /// [`fpha_override_resolves_by_domain_stage_id_end_to_end`] would then be unable to
    /// fail under a position-keyed regression.
    #[test]
    fn position_keyed_lookup_misses_the_domain_keyed_override_row() {
        let row = HydroEnergyProductivityRow {
            hydro_id: EntityId(FPHA_HYDRO_ID),
            stage_id: Some(OVERRIDE_DOMAIN_STAGE_ID),
            equivalent_productivity_mw_per_m3s: Some(OVERRIDE_RHO_EQ),
            reference_outflow_m3s: None,
            specific_productivity_mw_per_m3s_per_m: None,
        };
        let table = build_hydro_energy_productivity_override(&[row]).expect("override builds");

        let position_as_stage_id =
            StageId(i32::try_from(OVERRIDE_STAGE_POSITION).expect("position fits i32"));
        assert_eq!(
            table.equivalent_productivity(EntityId(FPHA_HYDRO_ID), position_as_stage_id),
            None,
            "a position-keyed lookup (StageId(1)) must miss the row declared at domain id 11"
        );
        assert_eq!(
            table.equivalent_productivity(
                EntityId(FPHA_HYDRO_ID),
                StageId(OVERRIDE_DOMAIN_STAGE_ID)
            ),
            Some(OVERRIDE_RHO_EQ),
            "a domain-id-keyed lookup (StageId(11)) must hit the declared row"
        );
    }

    /// Declaration-order invariance: a seeded permutation of every
    /// [`permute_case`]-classified registry (hydros, production models,
    /// stages, storage/filling entries) must not change the parity hash — the
    /// override fix must not depend on how the study's entities were
    /// declared. The full seeded-shuffle matrix (more registries, more
    /// permutations) lives in `tests/parity.rs`'s `shuffle_matrix_<case>`
    /// tests; this is the fast, non-golden default-CI probe for this fixture.
    #[test]
    fn declaration_order_permutation_parity_hash_is_identical() {
        let base_dir = case_dir();
        let permuted_dir = permute_case(&base_dir, PERMUTATION_SEED);

        let (setup_a, results_a) = train_and_simulate_setup(&base_dir);
        let (setup_b, results_b) = train_and_simulate_setup(permuted_dir.path());

        let hash_a = compute_parity_hash(&setup_a, results_a);
        let hash_b = compute_parity_hash(&setup_b, results_b);

        assert_eq!(
            hash_a, hash_b,
            "declaration-order invariance violated: base hash={hash_a}, permuted hash={hash_b}"
        );
    }

    /// Train + one-scenario-simulate `dir` and return the finished [`StudySetup`]
    /// alongside the drained per-scenario results, so [`compute_parity_hash`] (which
    /// needs both the FCF and the simulation trajectory) has both in scope.
    fn train_and_simulate_setup(dir: &Path) -> (StudySetup, Vec<SimulationScenarioResult>) {
        let config_path = dir.join("config.json");
        let config = cobre_io::parse_config(&config_path).expect("config must parse");
        let system = cobre_io::load_case(dir).expect("load_case must succeed");

        let pr = prepare_stochastic(system, dir, &config, 42, &ScenarioSource::default(), None)
            .expect("prepare_stochastic must succeed");
        let system = pr.system;
        let stochastic = pr.stochastic;

        let hydro_models =
            prepare_hydro_models(&system, dir, false).expect("prepare_hydro_models must succeed");

        let mut config_with_sim = config.clone();
        config_with_sim.simulation.enabled = true;
        config_with_sim.simulation.selection =
            Some(SimulationSelection::Sampled { num_scenarios: 1 });

        let mut setup =
            build_setup_for_case(dir, &config_with_sim, &system, stochastic, hydro_models);
        let scenario_results = run_simulation(&mut setup, 1);
        (setup, scenario_results)
    }
}

/// End-to-end regression for the `CalendarMonth` evaporation-month fix: a
/// `Custom`-cycle stage whose `season_id` deliberately differs from its calendar
/// month must still resolve evaporation by the TRUE calendar month derived from
/// `start_date`, and a `Weekly`-cycle evaporating stage (`season_id >= 12`) must
/// not hard-error at setup. Both fixtures route through the full
/// `cobre_io::load_case` -> `prepare_stochastic` -> `prepare_hydro_models`
/// pipeline every other deterministic case uses — the season-parsing path the
/// fix's own unit tests (`model/temporal/stage_key.rs` in `cobre-core`,
/// `hydro_models/evaporation.rs`) construct `Stage` values directly and never
/// exercise.
mod custom_weekly_evaporation_regression {
    use std::path::{Path, PathBuf};

    use cobre_core::scenario::ScenarioSource;
    use cobre_sddp::hydro_models::{EvaporationModel, EvaporationModelSet, prepare_hydro_models};
    use cobre_sddp::setup::prepare_stochastic;

    fn custom_case_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/custom_cycle_evaporation_month_mismatch")
    }

    fn weekly_case_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/weekly_cycle_evaporation")
    }

    /// Run `dir` through the same `load_case` -> `prepare_stochastic` ->
    /// `prepare_hydro_models` pipeline `run_deterministic` uses internally, and
    /// return the resolved evaporation models. Without the calendar-month
    /// derivation, the Weekly fixture panics here (`season_id >= 12` ->
    /// `SddpError::Validation` surfaces through the `.expect` below).
    fn resolve_evaporation(dir: &Path) -> EvaporationModelSet {
        let config = cobre_io::parse_config(&dir.join("config.json")).expect("config must parse");
        let system = cobre_io::load_case(dir).expect("load_case must succeed");
        let pr = prepare_stochastic(system, dir, &config, 42, &ScenarioSource::default(), None)
            .expect("prepare_stochastic must succeed");
        let hydro_models = prepare_hydro_models(&pr.system, dir, false)
            .expect("prepare_hydro_models must succeed (season_id >= 12 must no longer error)");
        hydro_models.evaporation
    }

    /// Both fixtures' hydro shares the VHA curve
    /// `resolve_evaporation_known_geometry_produces_correct_coefficients`
    /// (`hydro_models/evaporation.rs`) already pins: volumes `[100, 200, 300, 400,
    /// 500]` hm3, areas `[1.0, 1.5, 2.0, 2.5, 3.0]` km2, giving an already-verified
    /// constant slope — so the reference-volume area (`A_REF`) and its derivative
    /// (`DA_DV`) at the reservoir's midpoint volume (`REFERENCE_VOLUME_HM3`) are
    /// known constants here rather than recomputed from geometry.
    const A_REF: f64 = 2.0;
    const DA_DV: f64 = 0.005;
    const REFERENCE_VOLUME_HM3: f64 = 300.0;
    /// Evaporation divides by the CALENDAR month, not the stage's nominal 730 h:
    /// both fixture stages sit in 30-day months (June, November) = 720 h.
    const MONTH_HOURS: f64 = 720.0;

    /// Independent replication (not a call into production code) of the Taylor
    /// linearization `resolve_evaporation_core` computes for a given
    /// `monthly_evaporation_mm`, at this fixture's known geometry constants — the
    /// Tier-4 analytical-derivation pattern.
    fn expected_coefficients(monthly_evaporation_mm: f64) -> (f64, f64) {
        let mm_km2_to_m3s = 1.0 / (3.6 * MONTH_HOURS);
        let slope = mm_km2_to_m3s * monthly_evaporation_mm * DA_DV;
        let intercept =
            mm_km2_to_m3s * monthly_evaporation_mm * A_REF - slope * REFERENCE_VOLUME_HM3;
        (slope, intercept)
    }

    fn linearized_at(models: &EvaporationModelSet, stage_position: usize) -> (f64, f64) {
        match models.model(0) {
            EvaporationModel::Linearized { coefficients, .. } => {
                let c = &coefficients[stage_position];
                (c.volume_slope_m3s_per_hm3, c.intercept_m3s)
            }
            none @ EvaporationModel::None => {
                panic!("expected Linearized evaporation, got {none:?}")
            }
        }
    }

    /// `custom_cycle_evaporation_month_mismatch/system/hydros.json`'s
    /// `evaporation.coefficients_mm`, duplicated here so the expected values
    /// below are self-contained.
    const COEFFICIENTS_MM: [f64; 12] = [
        10.0, 20.0, 1.0, 40.0, 50.0, 60.0, 70.0, 1.0, 90.0, 100.0, 110.0, 120.0,
    ];

    /// Positive assertion plus analytic negative control (the fix is never
    /// reverted; the wrong lookup's value is asserted against instead): stage
    /// 0 (`start_date` 2024-06-01, `season_id = 2`) and stage 1 (`start_date`
    /// 2024-11-01, `season_id = 7`) each resolve evaporation from
    /// `coefficients_mm[month_of(stage)]` (index 5 = June = 60.0, index 10 =
    /// November = 110.0), never `coefficients_mm[season_id]` (index 2 = 1.0,
    /// index 7 = 1.0 — the value a `season_id`-keyed lookup would read instead).
    /// A checkout with `month_of` reverted to `stage.season_id` directly would
    /// make the first two assertions below fail: the resolved intercepts would
    /// equal `wrong_intercept0`/`wrong_intercept1` instead of
    /// `expected_intercept0`/`expected_intercept1`.
    #[test]
    fn custom_cycle_evaporation_indexes_by_calendar_month_not_season_id() {
        let dir = custom_case_dir();
        let models = resolve_evaporation(&dir);

        let (slope0, intercept0) = linearized_at(&models, 0);
        let (slope1, intercept1) = linearized_at(&models, 1);

        let (expected_slope0, expected_intercept0) = expected_coefficients(COEFFICIENTS_MM[5]);
        let (expected_slope1, expected_intercept1) = expected_coefficients(COEFFICIENTS_MM[10]);
        let (_, wrong_intercept0) = expected_coefficients(COEFFICIENTS_MM[2]);
        let (_, wrong_intercept1) = expected_coefficients(COEFFICIENTS_MM[7]);

        assert!(
            (slope0 - expected_slope0).abs() < 1e-12
                && (intercept0 - expected_intercept0).abs() < 1e-9,
            "stage 0 (June, season_id=2) must use coefficients_mm[5]=60.0: expected \
             (slope={expected_slope0}, intercept={expected_intercept0}), got \
             (slope={slope0}, intercept={intercept0})"
        );
        assert!(
            (slope1 - expected_slope1).abs() < 1e-12
                && (intercept1 - expected_intercept1).abs() < 1e-9,
            "stage 1 (November, season_id=7) must use coefficients_mm[10]=110.0: expected \
             (slope={expected_slope1}, intercept={expected_intercept1}), got \
             (slope={slope1}, intercept={intercept1})"
        );

        assert!(
            (intercept0 - wrong_intercept0).abs() > 0.005,
            "stage 0's resolved intercept ({intercept0}) must be far from the season_id=2-keyed \
             value ({wrong_intercept0}) — otherwise a season_id-keyed regression would be \
             indistinguishable from the fix"
        );
        assert!(
            (intercept1 - wrong_intercept1).abs() > 0.005,
            "stage 1's resolved intercept ({intercept1}) must be far from the season_id=7-keyed \
             value ({wrong_intercept1}) — otherwise a season_id-keyed regression would be \
             indistinguishable from the fix"
        );
    }

    /// A Weekly-cycle evaporating study (`season_id` 21 and 26, both `>= 12`)
    /// does not return `SddpError::Validation` at setup, and training
    /// completes and converges. Without the calendar-month derivation,
    /// [`super::run_deterministic`] panics on this fixture inside
    /// `prepare_hydro_models`'s `.expect("prepare_hydro_models must succeed")`.
    #[test]
    fn weekly_cycle_evaporation_no_longer_errors_and_setup_completes() {
        let dir = weekly_case_dir();

        let models = resolve_evaporation(&dir);
        assert!(
            matches!(models.model(0), EvaporationModel::Linearized { .. }),
            "hydro 0 must resolve a Linearized evaporation model, got {:?}",
            models.model(0)
        );

        let result = super::run_deterministic(&dir);
        assert!(
            result.iterations <= 10,
            "weekly evaporation case must converge quickly: iterations={}",
            result.iterations
        );
        assert!(
            result.final_gap.abs() < 1e-6,
            "weekly evaporation case must still converge: gap={:.2e}",
            result.final_gap
        );
    }
}

#[cfg(feature = "test-support")]
mod k_fan_branching_sampled_coverage {
    //! Branching-graph end-to-end sampled coverage on the DECOMP K-fan fixture
    //! (`cobre_sddp::test_support::k_fan_setup`): a declared root fanning into
    //! `K` distinct nodes, each with its own leaf, `num_nodes > n_pools` via
    //! leaf sharing — a shape a chain-only suite cannot reach, where a
    //! canonical-node-position-as-pool-id conflation bug would misroute or
    //! overflow a pool instead of hiding behind `node_index == pool_id`.
    //! Exercises the per-node forward frontier, the reverse-topological
    //! backward sweep, and the per-pool trial-state routing end-to-end.

    use std::collections::HashSet;
    use std::sync::mpsc;

    use cobre_sddp::SimulationWeighting;
    use cobre_sddp::TrainingOutcome;
    use cobre_sddp::aggregate_simulation;
    use cobre_sddp::setup::{NodePos, StageIdx};
    use cobre_sddp::test_support::k_fan_setup;
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    const K: usize = 12;
    const FORWARD_PASSES: u32 = 3;
    const MAX_ITERATIONS: u32 = 5;

    /// Train the K-fan fixture single-rank single-thread `sampled`, panicking
    /// on any training error (a stored-cut slot out-of-bounds panics INSIDE
    /// `train` via `CutPool::add_cut`'s own `debug_assert`, before this
    /// function returns).
    fn train_k_fan() -> (cobre_sddp::test_support::KFanFixture, TrainingOutcome) {
        let mut fixture = k_fan_setup(K, FORWARD_PASSES, MAX_ITERATIONS);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let outcome = fixture
            .setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("k_fan training must return Ok");
        assert!(
            outcome.error.is_none(),
            "k_fan training must not error: {:?}",
            outcome.error
        );
        (fixture, outcome)
    }

    /// Cuts appended to `pool_id` during `iteration`, read from the trained
    /// [`cobre_sddp::CutPool`]'s own per-slot metadata — never a hard-coded
    /// count. Every appended cut is 1:1 with a trial point routed to this
    /// node this iteration (`compute_one_backward_node`'s
    /// `debug_assert_eq!(staged_cuts_buf.len(), trial_points.len())`), so
    /// this count IS the seed-deterministic sampled visit count.
    fn cuts_in_iteration(
        fixture: &cobre_sddp::test_support::KFanFixture,
        pool_id: usize,
        iteration: u64,
    ) -> u64 {
        let pool = &fixture.setup.fcf.pools[pool_id];
        let mut count = 0u64;
        for slot in 0..pool.populated() {
            if pool.is_active(slot) && pool.metadata(slot).iteration_generated == iteration {
                count += 1;
            }
        }
        count
    }

    /// Load-bearing coverage gate: trains without panic (proving no
    /// out-of-bounds cut slot — `CutPool::add_cut`'s
    /// `debug_assert!(slot < capacity)` would panic first), then cross-checks
    /// the routing structurally: the root (the sole stage-0 node) gets
    /// exactly `forward_passes` cuts every iteration; the K fan nodes'
    /// per-iteration cut counts sum to exactly `forward_passes` (every trial
    /// point routes to exactly one fan node — neither dropped nor
    /// double-routed); and the shared leaf pool never receives a cut (a leaf
    /// has no successor to generate one for).
    #[test]
    fn k_fan_sampled_routing_sums_to_forward_passes_with_no_oob() {
        let (fixture, _outcome) = train_k_fan();
        let node_graph = &fixture.setup.inputs.node_graph;

        let cut_generating: Vec<NodePos> = (0..node_graph.nodes.len())
            .map(NodePos)
            .filter(|&pos| !node_graph.successors[pos].is_empty())
            .collect();
        assert_eq!(
            cut_generating.len(),
            1 + K,
            "K-fan power precondition: exactly 1 root + K fan nodes must be cut-generating"
        );
        let root_pos = *cut_generating
            .iter()
            .find(|&&pos| node_graph.nodes[pos].stage == StageIdx(0))
            .expect("the root is cut-generating and is the fixture's sole stage-0 node");
        let fan_positions: Vec<NodePos> = cut_generating
            .iter()
            .copied()
            .filter(|&pos| pos != root_pos)
            .collect();
        assert_eq!(
            fan_positions.len(),
            K,
            "K distinct fan nodes must be cut-generating"
        );

        let leaf_positions: Vec<NodePos> = (0..node_graph.nodes.len())
            .map(NodePos)
            .filter(|&pos| node_graph.successors[pos].is_empty())
            .collect();
        assert_eq!(leaf_positions.len(), K, "K leaves, one per fan branch");
        let leaf_pool = node_graph.nodes[leaf_positions[0]].pool_id;
        for &pos in &leaf_positions {
            assert_eq!(
                node_graph.nodes[pos].pool_id, leaf_pool,
                "every leaf must share the one leaf-sharing pool"
            );
        }
        assert_eq!(
            fixture.setup.fcf.pools[leaf_pool].populated(),
            0,
            "a leaf has no successor, so it must never generate a cut"
        );

        let mut touched_fan_nodes = HashSet::new();
        let root_pool = node_graph.nodes[root_pos].pool_id;
        for iteration in 1..=u64::from(MAX_ITERATIONS) {
            assert_eq!(
                cuts_in_iteration(&fixture, root_pool, iteration),
                u64::from(FORWARD_PASSES),
                "iteration {iteration}: the root is the sole stage-0 node, so every forward \
                 trial point reaches it — its per-iteration cut count must always equal \
                 forward_passes"
            );

            let mut fan_total = 0u64;
            for &pos in &fan_positions {
                let pool_id = node_graph.nodes[pos].pool_id;
                let count = cuts_in_iteration(&fixture, pool_id, iteration);
                assert!(
                    count <= u64::from(FORWARD_PASSES),
                    "iteration {iteration}: fan node at position {pos} (pool {pool_id}) got \
                     {count} cuts, exceeding forward_passes ({FORWARD_PASSES}) — over-routed"
                );
                if count > 0 {
                    touched_fan_nodes.insert(pos);
                }
                fan_total += count;
            }
            assert_eq!(
                fan_total,
                u64::from(FORWARD_PASSES),
                "iteration {iteration}: the K fan nodes' per-iteration cut counts must sum to \
                 forward_passes — every forward trial point routes to EXACTLY one fan node, \
                 never zero (dropped) and never more than one (double-routed)"
            );
        }

        assert!(
            touched_fan_nodes.len() >= 2,
            "power precondition: the run must genuinely touch >= 2 distinct fan nodes over \
             {MAX_ITERATIONS} iterations (touched {}), or cross-node misrouting has nothing \
             to be caught against",
            touched_fan_nodes.len()
        );
    }

    /// Sampled scale gate: sampled solves-per-iteration equals the sampled
    /// node-visit work — every one of `forward_passes` trial points visits
    /// exactly `path_length` nodes (root, its sampled fan node, that fan
    /// node's leaf), so the forward-phase LP-solve count is
    /// `forward_passes * path_length` every iteration, strictly below the
    /// per-path enumeration `enumerated_scenario_count` — never the exact
    /// per-node enumerated visit count (unwired: enumerated execution beyond
    /// a derived count of 1 is not yet admitted).
    #[test]
    fn k_fan_sampled_forward_solves_equal_visit_work_below_enumerated() {
        let (fixture, outcome) = train_k_fan();

        let path_length = 1 + fixture
            .setup
            .inputs
            .node_graph
            .nodes
            .iter()
            .map(|n| n.stage.0)
            .max()
            .expect("the K-fan graph has at least one node") as u64;
        assert_eq!(
            path_length, 3,
            "the K-fan is root -> fan -> leaf, 3 stages deep"
        );

        let expected = u64::from(fixture.forward_passes) * path_length;
        assert!(
            expected < fixture.enumerated_scenario_count,
            "power precondition: forward_passes * path_length ({expected}) must be strictly \
             below enumerated_scenario_count ({}) — otherwise sampled and enumerated scale \
             are indistinguishable on this fixture",
            fixture.enumerated_scenario_count
        );

        for iteration in 1..=u64::from(MAX_ITERATIONS) {
            let forward_solves: u64 = outcome
                .result
                .solver_stats_log
                .iter()
                .filter(|e| e.iteration == iteration && e.phase == "forward")
                .map(|e| e.delta.lp_solves)
                .sum();
            assert_eq!(
                forward_solves, expected,
                "iteration {iteration}: forward-phase LP solves must equal forward_passes * \
                 path_length — a regression toward per-path enumeration would instead scale \
                 with K"
            );
            assert!(
                forward_solves < fixture.enumerated_scenario_count,
                "iteration {iteration}: sampled forward work ({forward_solves}) must stay \
                 strictly below the per-path enumeration ({})",
                fixture.enumerated_scenario_count
            );
        }
    }

    /// A declared branching graph went infeasible past ~2 iterations before the
    /// per-pool frozen overlay landed (a leaf stage's frozen template embedded a
    /// fan node's cut, and the terminal leaf's pinned theta then made the LP
    /// infeasible). This asserts the K-fan trains the full run without an
    /// `Infeasible { .. }`, that the frozen overlay is per POOL (not per stage),
    /// and that each node genuinely warm-starts from its OWN node basis — the
    /// offered bases are reused and every one is accepted (no slot-identity
    /// reconstruction failure), not silently all-cold.
    #[test]
    fn k_fan_trains_per_pool_frozen_with_reused_node_bases() {
        let (fixture, outcome) = train_k_fan();
        let result = outcome.result;

        assert!(
            result.iterations >= 3,
            "the K-fan must train past the pre-fix ~2-iteration infeasibility point (ran {})",
            result.iterations
        );
        assert!(result.final_lb.is_finite(), "final_lb must be finite");

        // The frozen overlay is one template per POOL, and this fixture has
        // strictly more pools than stages (leaf sharing gives n_pools = K+2 over
        // 3 stages) — so a per-stage overlay could not have produced this length.
        let node_graph = &fixture.setup.inputs.node_graph;
        let n_pools = node_graph.n_pools;
        let n_stages: usize = node_graph
            .nodes
            .iter()
            .map(|n| n.stage)
            .collect::<HashSet<_>>()
            .len();
        assert!(
            n_pools > n_stages,
            "the K-fan must have more pools ({n_pools}) than stages ({n_stages}) via leaf sharing"
        );
        let frozen = result
            .frozen_templates
            .as_ref()
            .expect("training always returns frozen templates");
        assert_eq!(
            frozen.len(),
            n_pools,
            "frozen overlay must carry one template per pool, not per stage"
        );

        // Warm-start reuse: forward+backward solves offer stored node bases, and
        // every offer is accepted. A broken (m, stage) key would alias sibling
        // nodes, the node_id guard would turn each into a cold solve (no offer),
        // and `basis_offered` would collapse; a broken slot-identity
        // reconstruction would instead raise `basis_consistency_failures`.
        let (offered, rejected) = result
            .solver_stats_log
            .iter()
            .filter(|e| e.phase == "forward" || e.phase == "backward")
            .fold((0u64, 0u64), |(o, r), e| {
                (
                    o + e.delta.basis_offered,
                    r + e.delta.basis_consistency_failures,
                )
            });
        assert!(
            offered > 0,
            "branching warm-start must reuse node bases (basis_offered > 0), not solve all-cold"
        );
        assert_eq!(
            rejected, 0,
            "every offered node basis must reconstruct by slot and be accepted \
             (basis_consistency_failures must be 0), got {rejected}"
        );
    }

    /// Branching SIMULATION: each scenario's sampled walk visits a fan node and
    /// its leaf, and each stage solve must load the VISITED node's own pool
    /// frozen template (the per-pool overlay), not a per-stage mix. Pre-fix this
    /// went infeasible the same way training did. Asserts every scenario
    /// simulates to a finite cost.
    #[test]
    fn k_fan_simulation_loads_visited_node_pool_cuts() {
        let (mut fixture, outcome) = train_k_fan();
        let result = outcome.result;

        // Simulate one scenario per enumerated fan path so several distinct
        // visited nodes (hence distinct pools) are exercised.
        let n_scenarios = u32::try_from(K).expect("K fits u32");
        fixture.setup.simulation_config.n_scenarios = n_scenarios;

        let comm = StubComm;
        let mut pool = fixture
            .setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("simulation workspace pool must build");

        let io_capacity = fixture.setup.simulation_config.io_channel_capacity.max(1);
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
        let drain = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

        let local_costs = fixture
            .setup
            .simulate(
                &mut pool.workspaces,
                &comm,
                &result_tx,
                None,
                result.frozen_templates.as_deref(),
                &result.basis_cache,
            )
            .expect("branching simulation must not go infeasible");
        drop(result_tx);
        let scenario_results = drain.join().expect("drain thread must not panic");

        assert_eq!(
            scenario_results.len(),
            n_scenarios as usize,
            "every simulated scenario must produce a result"
        );
        for r in &scenario_results {
            assert!(
                r.total_cost.is_finite(),
                "scenario {} cost must be finite (a per-stage frozen mix would infeasible)",
                r.scenario_id
            );
        }
        assert_eq!(
            local_costs.costs.len(),
            n_scenarios as usize,
            "compact cost buffer must carry one entry per scenario"
        );
        let (summary, _gathered) = aggregate_simulation(
            &local_costs.costs,
            fixture.setup.simulation_config(),
            &comm,
            SimulationWeighting::Uniform,
        )
        .expect("aggregate_simulation must succeed");
        assert!(
            summary.mean_cost.is_finite(),
            "aggregate mean simulation cost must be finite"
        );
    }
}

mod k_fan_enumerated_exact_bound {
    //! Enumerated all-paths forward on the DECOMP K-fan (`K >= 2`): the exhaustive
    //! engine visits every distinct node once, reconstructs each root->leaf path's
    //! cost, and reduces the exact bound `Σ_ℓ P(ℓ)·C(ℓ)`. The lower bound (an
    //! independent root Bellman evaluation) is the reference the exact upper bound
    //! must close to; the forward LP-solve total pins the dedup scale gate.

    use cobre_sddp::TrainingOutcome;
    use cobre_sddp::test_support::{k_fan_setup_enumerated, try_k_fan_simulation_enumerated};
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    const K: usize = 4;
    const MAX_ITERATIONS: u32 = 15;

    fn train() -> (cobre_sddp::test_support::KFanFixture, TrainingOutcome) {
        let mut fixture = k_fan_setup_enumerated(K, MAX_ITERATIONS);
        assert_eq!(
            u64::from(fixture.forward_passes),
            fixture.enumerated_scenario_count,
            "enumerated forward_passes must resolve to the graph's path count"
        );
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let outcome = fixture
            .setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("enumerated K-fan training must return Ok");
        assert!(
            outcome.error.is_none(),
            "enumerated K-fan training must not error: {:?}",
            outcome.error
        );
        (fixture, outcome)
    }

    /// The exact upper bound `Σ_ℓ P(ℓ)·C(ℓ)` closes to the independent lower-bound
    /// root evaluation (the reference), with a zero CI half-width (a deduplicated
    /// enumeration carries no sampling distribution). Trains `>= 3` iterations
    /// without an `Infeasible`.
    #[test]
    fn enumerated_k_fan_closes_lb_to_exact_ub_with_zero_ci() {
        let (_fixture, outcome) = train();
        let result = outcome.result;

        assert!(
            result.iterations >= 3,
            "must train >= 3 iterations (ran {})",
            result.iterations
        );
        assert!(result.final_lb.is_finite() && result.final_ub.is_finite());

        let tol = 1e-6 * result.final_ub.abs().max(1.0);
        assert!(
            (result.final_ub - result.final_lb).abs() <= tol,
            "the exact UB {} must close to the independent LB {} within {tol}",
            result.final_ub,
            result.final_lb,
        );
        assert_eq!(
            result.final_ub_std, 0.0,
            "an enumerated bound has no sampling distribution: std must be 0"
        );
    }

    /// Dedup scale gate: the enumerated forward solves each distinct node
    /// exactly once, so the per-iteration forward LP-solve total equals
    /// `Σ forward_solve_counts` (the node count on a `|Ω|=1` tree), strictly
    /// below the naive per-path total `enumerated_scenario_count * path_length`.
    #[test]
    fn enumerated_k_fan_forward_solves_equal_dedup_total_below_naive() {
        let (fixture, outcome) = train();
        let node_graph = &fixture.setup.inputs.node_graph;

        // Σ forward_solve_counts on a |Ω|=1 tree == the node count (π(n) = 1).
        let dedup_total = node_graph.nodes.len() as u64;
        let path_length = 1 + node_graph
            .nodes
            .iter()
            .map(|n| n.stage.0)
            .max()
            .expect("the K-fan has nodes") as u64;
        assert_eq!(path_length, 3, "root -> fan -> leaf");
        let naive_total = fixture.enumerated_scenario_count * path_length;
        assert!(
            dedup_total < naive_total,
            "dedup total ({dedup_total}) must be strictly below naive per-path re-solving \
             ({naive_total})"
        );

        for iteration in 1..=u64::from(MAX_ITERATIONS).min(outcome.result.iterations) {
            let forward_solves: u64 = outcome
                .result
                .solver_stats_log
                .iter()
                .filter(|e| e.iteration == iteration && e.phase == "forward")
                .map(|e| e.delta.lp_solves)
                .sum();
            assert_eq!(
                forward_solves, dedup_total,
                "iteration {iteration}: enumerated forward must solve each node once \
                 (total {dedup_total}), not re-solve shared prefixes per path"
            );
        }
    }

    /// The simulation-enumerated admissibility guard admits any derived leaf-path
    /// count on a single-predecessor tree: an enumerated `K >= 2` simulation setup
    /// resolves `Ok`, with `n_scenarios` set to the derived count `K`.
    #[test]
    fn enumerated_k_ge_2_simulation_is_admitted() {
        let setup = try_k_fan_simulation_enumerated(K)
            .expect("enumerated K>=2 simulation must be admitted on a single-predecessor tree");
        assert_eq!(
            setup.simulation_config.n_scenarios,
            u32::try_from(K).expect("K fits u32"),
            "n_scenarios must resolve to the derived leaf-path count K"
        );
    }
}

mod visit_bound_overflow_guard {
    //! The release-active visit-bound overflow guard
    //! (`check_visit_bound` in `training/backward_pass_state.rs`): a named
    //! rejection when a realized routed count exceeds a pool's statistical
    //! `pool_cut_stride` floor, and structural non-firing on chain and
    //! enumerated graphs.

    use std::path::Path;

    use cobre_sddp::SddpError;
    use cobre_sddp::setup::NodePos;
    use cobre_sddp::test_support::{k_fan_setup, k_fan_setup_enumerated};
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;
    use super::run_deterministic_with_setup;

    /// K=2, F=8: the smallest combination found by a documented sweep over
    /// `(k_fan branch count, forward_passes, max_iterations)` whose
    /// SAMPLED routing — under `k_fan_setup`'s fixed training seed — realizes
    /// a routed count above a fan pool's capped stride within the iteration
    /// budget (the sweep tried k in 2..=30, forward_passes in 3..=20, and
    /// max_iterations up to 50; every other combination trained clean the
    /// whole way). Overflow first occurs at iteration <= 27; 30 sits
    /// comfortably past that.
    const K: usize = 2;
    const FORWARD_PASSES: u32 = 8;
    const MAX_ITERATIONS: u32 = 30;

    /// Constructs a fixture whose capped stride sits strictly below a
    /// realizable routed count, drives training past it, and asserts the
    /// named rejection.
    #[test]
    fn overflow_guard_rejects_realized_visit_count_above_stride() {
        let mut fixture = k_fan_setup(K, FORWARD_PASSES, MAX_ITERATIONS);

        // Precondition (asserted, not assumed): some fan pool's capped
        // stride sits STRICTLY below forward_passes, the realizable routed
        // ceiling — otherwise an overflow cannot occur on this fixture at
        // all and the test below would pass vacuously.
        let node_graph = &fixture.setup.inputs.node_graph;
        let capped_pool = (0..node_graph.nodes.len())
            .map(NodePos)
            .filter(|&pos| !node_graph.successors[pos].is_empty())
            .map(|pos| node_graph.nodes[pos].pool_id)
            .find(|&pool_id| fixture.setup.fcf.pools[pool_id].visit_stride < FORWARD_PASSES)
            .expect(
                "power precondition: some fan pool's capped stride must sit strictly below \
                 forward_passes on this K/forward_passes combination, or an overflow cannot be \
                 realized on this fixture at all",
            );
        assert!(
            fixture.setup.fcf.pools[capped_pool].visit_stride < FORWARD_PASSES,
            "power precondition restated: pool {capped_pool}'s stride ({}) must be strictly \
             below forward_passes ({FORWARD_PASSES})",
            fixture.setup.fcf.pools[capped_pool].visit_stride
        );

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome = fixture
            .setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect(
                "train() itself must return Ok; a mid-training error surfaces as outcome.error",
            );

        match outcome.error {
            Some(SddpError::Validation(msg)) => {
                assert!(
                    msg.contains("visit-bound overflow"),
                    "names the failure: {msg}"
                );
                assert!(
                    msg.contains(&format!("pool {capped_pool}")),
                    "names the overflowing pool: {msg}"
                );
                assert!(msg.contains("stride"), "names the stride: {msg}");
                assert!(msg.contains("routed"), "names the routed count: {msg}");
            }
            other => panic!(
                "expected a named SddpError::Validation visit-bound rejection on this \
                 (K={K}, forward_passes={FORWARD_PASSES}, max_iterations={MAX_ITERATIONS}) \
                 fixture, got {other:?}"
            ),
        }
    }

    /// Chain half: every shipped chain deterministic case already routes
    /// through the guard on every level; `run_deterministic_with_setup`
    /// asserts `outcome.error.is_none()` internally, so a misfiring guard
    /// would already fail this call. Additionally asserts the arithmetic
    /// identity the non-firing falls out of: a chain pool's reach
    /// probability is exactly `1`, so its `pool_cut_stride` margin equals
    /// `forward_passes` bit-for-bit (never merely `>=`).
    #[test]
    fn overflow_guard_never_fires_on_chain() {
        let case_dir = Path::new("../../examples/deterministic/d01-thermal-dispatch");
        let (setup, _system, result) = run_deterministic_with_setup(case_dir);
        assert!(
            result.iterations > 0,
            "must train at least one iteration for the guard to have run at all"
        );
        for (pool_id, pool) in setup.fcf.pools.iter().enumerate() {
            assert_eq!(
                pool.visit_stride, setup.loop_params.forward_passes,
                "pool {pool_id}: a chain pool's stride must equal forward_passes exactly \
                 (reach probability 1, pool_cut_stride's variance-0 identity)"
            );
        }
    }

    /// Enumerated half: the K-fan trained fully enumerated
    /// (`forward_passes` resolved to the graph's exact path count) must
    /// never trip the guard — every reachable pool's `pool_cut_stride`
    /// margin structurally upper-bounds its exact, deterministic enumerated
    /// visit count (no shape predicate: the margin's `+3σ` term is always
    /// `>= 0` and its cap never falls below the mean).
    #[test]
    fn overflow_guard_never_fires_on_enumerated_k_fan() {
        const K: usize = 4;
        const MAX_ITERATIONS: u32 = 8;

        let mut fixture = k_fan_setup_enumerated(K, MAX_ITERATIONS);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome = fixture
            .setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("enumerated K-fan training must return Ok");
        assert!(
            outcome.error.is_none(),
            "the overflow guard must never fire on an enumerated graph: {:?}",
            outcome.error
        );
    }
}

mod heterogeneous_visit_bound_resume {
    //! `FutureCostFunction::new_with_warm_start` threads the study's
    //! per-pool state-dimension and visit-bound arrays — the same per-pool
    //! contract `new_per_pool` takes — so a resumed run's pool geometry
    //! matches the cold-start run's. Exercised on the K-fan's HETEROGENEOUS
    //! per-pool strides (declared non-uniform branch weights): the shape the
    //! pre-fix scalar substitution (every resumed pool given `forward_passes`
    //! uniformly as its stride) silently broke.

    use cobre_io::{
        STAGE_CUTS_NODE_ID_SENTINEL, STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, StageCutsReadResult,
    };
    use cobre_sddp::FutureCostFunction;
    use cobre_sddp::setup::NodeId;
    use cobre_sddp::test_support::{k_fan_setup, trivial_full_fcf_proof};

    const K: usize = 4;
    const FORWARD_PASSES: u32 = 3;
    const MAX_ITERATIONS: u32 = 5;

    #[test]
    fn resume_matches_cold_start_stride_and_slots_on_heterogeneous_bounds() {
        let cold = k_fan_setup(K, FORWARD_PASSES, MAX_ITERATIONS);
        let cold_strides: Vec<u32> = cold
            .setup
            .fcf
            .pools
            .iter()
            .map(|p| p.visit_stride)
            .collect();
        let cold_dims: Vec<usize> = cold
            .setup
            .fcf
            .pools
            .iter()
            .map(|p| p.state_dimension)
            .collect();

        // Power precondition: the K-fan's declared non-uniform branch weights
        // give distinct pools distinct strides — the shape a scalar
        // substitution (every pool given forward_passes uniformly) cannot be
        // distinguished from the fix on.
        assert!(
            cold_strides.iter().any(|&s| s != cold_strides[0]),
            "power precondition: the K-fan fixture must carry heterogeneous per-pool visit \
             strides (got {cold_strides:?}), or this test cannot distinguish the fix from the \
             pre-fix scalar substitution"
        );

        let proof = trivial_full_fcf_proof(
            u32::try_from(cold_dims[0]).expect("state dimension fits u32"),
            u32::try_from(cold.setup.num_stages()).expect("stage count fits u32"),
        );

        // Mirrors `sole_pool_owner_node_id` (the production writer's
        // algorithm in `policy_export.rs`): every pool has exactly one
        // owning node EXCEPT the K-fan's shared terminal leaf pool, which
        // `k` distinct leaf nodes own jointly and must therefore keep the
        // sentinel.
        let pool_owner_node_id = |pool: usize| -> i32 {
            let mut owner: Option<i32> = None;
            for (pos, node) in cold.setup.inputs.node_graph.nodes.iter_indexed() {
                if node.pool_id == pool {
                    if owner.is_some() {
                        return STAGE_CUTS_NODE_ID_SENTINEL;
                    }
                    owner = Some(cold.setup.inputs.node_graph.node_ids[pos].0);
                }
            }
            owner.unwrap_or(STAGE_CUTS_NODE_ID_SENTINEL)
        };

        // A checkpoint with no warm-start cuts: this test's concern is the
        // constructor's per-pool dimension/stride threading, not cut-byte
        // fidelity (already covered by
        // `assert_cross_mode_load_preserves_cut_bytes` in
        // `chronological_telescoping`).
        let stage_results: Vec<StageCutsReadResult> = cold_dims
            .iter()
            .enumerate()
            .map(|(p, &dim)| StageCutsReadResult {
                stage_id: u32::try_from(p).expect("pool index fits u32"),
                state_dimension: u32::try_from(dim).expect("state dimension fits u32"),
                capacity: 0,
                warm_start_count: 0,
                populated_count: 0,
                cuts: Vec::new(),
                entity_manifest: Vec::new(),
                cost_scale_factor: None,
                node_id: pool_owner_node_id(p),
                graph_stage_id: -1,
                priced_state_date: STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
            })
            .collect();
        let visit_bounds: Vec<u64> = cold_strides.iter().map(|&s| u64::from(s)).collect();

        let mut resumed = FutureCostFunction::new_with_warm_start(
            &proof,
            &stage_results,
            &cold_dims,
            &visit_bounds,
            FORWARD_PASSES,
            u64::from(MAX_ITERATIONS),
        )
        .expect("new_with_warm_start must build from the injected per-pool arrays");

        let resumed_strides: Vec<u32> = resumed.pools.iter().map(|p| p.visit_stride).collect();
        assert_eq!(
            resumed_strides, cold_strides,
            "resumed per-pool strides must equal the cold-start run's on a heterogeneous-bound \
             graph — the deleted scalar substitution replaced every pool's stride with \
             forward_passes ({FORWARD_PASSES}) uniformly"
        );

        // Slot-index equivalence: the resumed pool's warm_start_count is 0
        // (no warm-start cuts here) and its iteration_base defaults to 0,
        // exactly like a fresh cold pool — so a cut at (iteration=1,
        // forward_pass_index=0) must land at the documented formula's slot,
        // `warm_start_count + iteration * stride + forward_pass_index`,
        // using the COLD-START run's own per-pool stride.
        for (pool_id, &stride) in cold_strides.iter().enumerate() {
            resumed.add_cut(
                NodeId(0),
                pool_id,
                1,
                0,
                1.0,
                &vec![0.0; cold_dims[pool_id]],
            );
            let expected_slot = stride as usize;
            assert!(
                resumed.pools[pool_id].is_active(expected_slot),
                "pool {pool_id}: a cut at iteration 1, forward_pass_index 0 must land at slot \
                 {expected_slot} (warm_start_count 0 + iteration 1 * stride {stride} + \
                 forward_pass_index 0) — the cold-start run's own stride"
            );
        }
    }
}

/// Enumerated + external activation: a `|Ω| = 1` single-path chain whose nodes
/// each declare an external scenario column. The forward pin and the backward
/// `OpeningSource` match make both passes read the same `eta_slice(stage, k)`,
/// so the exact upper bound over the declared columns meets the lower bound.
mod enumerated_external {
    #![allow(clippy::cast_possible_wrap)]

    use std::collections::BTreeMap;
    use std::sync::mpsc;

    use chrono::NaiveDate;
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, DeficitSegment,
        EntityId, HorizonGraph, HydroBlockBounds, HydroPenalties, HydroStageBounds,
        LineBlockBounds, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec,
        PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, System,
        SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
        entities::hydro::HydroGenerationModel,
        scenario::{ExternalScenarioRow, InflowModel, LoadModel, SamplingScheme, ScenarioSource},
        temporal::{
            Node as PolicyNode, NoiseMethod, PolicyGraphType, ScenarioSourceConfig,
            StageStateConfig, Transition,
        },
    };
    use cobre_sddp::{
        DEFAULT_COST_SCALE_FACTOR, InflowNonNegativityMethod, SimulationScenarioResult,
        StoppingMode, StoppingRule, StoppingRuleSet, StudySetup,
        hydro_models::PrepareHydroModelsResult,
        setup::{SimulationEnumeratedRequest, StudyParams},
    };
    use cobre_solver::ActiveSolver;
    use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

    use crate::common::StubComm;
    use crate::common::builders::{
        BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage,
    };

    const N_STAGES: usize = 3;
    /// Raw external inflow (m³/s) each node's declared column realizes — distinct
    /// per stage, so the pinned column genuinely discriminates: were the backward
    /// pass to read the generated tree instead, its cut would price a different
    /// inflow and the lower bound would not meet the exact upper bound.
    const EXTERNAL_INFLOW_M3S: [f64; N_STAGES] = [90.0, 40.0, 70.0];
    /// A second external column (`scenario_id = 1`) the nodes do NOT declare.
    /// Distinct enough that pricing it yields a different converged bound, so a
    /// simulation walk that hash-selected (as `select_external_scenario` is
    /// stage-independent, the hash picks one column for every stage of a scenario)
    /// instead of honoring the node pin would evaluate a different policy — the
    /// discriminator against a dropped pin.
    const DECOY_INFLOW_M3S: [f64; N_STAGES] = [0.0, 0.0, 0.0];
    const INFLOW_MEAN_M3S: f64 = 60.0;
    const INFLOW_STD_M3S: f64 = 10.0;
    const DEMAND_MW: f64 = 50.0;

    fn build_system() -> System {
        let bus_id = EntityId(1);
        let hydro_id = EntityId(2);

        let bus = make_bus(
            bus_id,
            BusSpec {
                name: "B".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
            },
        );

        let hydro = make_hydro(
            hydro_id,
            HydroSpec {
                name: "H".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id,
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                specific_productivity_mw_per_m3s_per_m: Some(0.5),
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                ..HydroSpec::default()
            },
        );

        let stages: Vec<_> = (0..N_STAGES)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                        season_id: None,
                        state_config: StageStateConfig {
                            storage: true,
                            inflow_lags: false,
                        },
                        scenario_config: ScenarioSourceConfig {
                            branching_factor: 1,
                            noise_method: NoiseMethod::Saa,
                        },
                        ..StageSpec::default()
                    },
                )
            })
            .collect();

        let inflow_models: Vec<_> = (0..N_STAGES)
            .map(|i| InflowModel {
                hydro_id,
                stage_id: i as i32,
                mean_m3s: INFLOW_MEAN_M3S,
                std_m3s: INFLOW_STD_M3S,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        // Deterministic demand (std 0 ⇒ zero stochastic-load noise dimension), so
        // the only noise class is the external inflow.
        let load_models: Vec<_> = (0..N_STAGES)
            .map(|i| LoadModel {
                bus_id,
                stage_id: i as i32,
                mean_mw: DEMAND_MW,
                std_mw: 0.0,
            })
            .collect();

        // Two external columns per stage: column 0 (declared by the nodes here)
        // and a decoy column 1 the pin must never select.
        let external_scenarios: Vec<_> = (0..N_STAGES)
            .flat_map(|i| {
                [
                    ExternalScenarioRow {
                        stage_id: i as i32,
                        scenario_id: 0,
                        hydro_id,
                        value_m3s: EXTERNAL_INFLOW_M3S[i],
                    },
                    ExternalScenarioRow {
                        stage_id: i as i32,
                        scenario_id: 1,
                        hydro_id,
                        value_m3s: DECOY_INFLOW_M3S[i],
                    },
                ]
            })
            .collect();

        // Single-successor chain; every node declares external column 0, so the
        // graph is node-native (External openings), a single enumerated path. The
        // library also carries a decoy column 1 the nodes never declare — a
        // dropped pin would hash-select it and price a different policy.
        let nodes: Vec<_> = (0..N_STAGES)
            .map(|i| PolicyNode {
                id: i as i32,
                stage_id: i as i32,
                scenario_id: Some(0),
                label: None,
            })
            .collect();
        let transitions: Vec<_> = (0..N_STAGES - 1)
            .map(|i| Transition {
                source_id: i as i32,
                target_id: (i + 1) as i32,
                probability: 1.0,
                annual_discount_rate_override: None,
            })
            .collect();
        let policy_graph = HorizonGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions,
            nodes,
            stage_discount_rate_overrides: BTreeMap::new(),
            season_map: None,
        };

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 250.0,
                    ..Default::default()
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 500.0,
                    filling_target_violation_cost: 0.0,
                    turbined_violation_below_cost: 0.0,
                    outflow_violation_below_cost: 0.0,
                    outflow_violation_above_cost: 0.0,
                    generation_violation_below_cost: 0.0,
                    evaporation_violation_cost: 0.0,
                    water_withdrawal_violation_cost: 0.0,
                    water_withdrawal_violation_pos_cost: 0.0,
                    water_withdrawal_violation_neg_cost: 0.0,
                    evaporation_violation_pos_cost: 0.0,
                    evaporation_violation_neg_cost: 0.0,
                    inflow_nonnegativity_cost: 1000.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .external_scenarios(external_scenarios)
            .policy_graph(policy_graph)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("enumerated-external chain system must build")
    }

    fn construction_config(
        backward_scheduler: cobre_io::config::BackwardScheduler,
        n_scenarios: u32,
    ) -> StudyParams {
        StudyParams {
            seed: 42,
            forward_passes: 1,
            training_enumerated: true,
            stopping_rule_set: StoppingRuleSet {
                rules: vec![StoppingRule::IterationLimit { limit: 40 }],
                mode: StoppingMode::Any,
            },
            n_scenarios,
            simulation_enumerated: SimulationEnumeratedRequest::Sampled,
            io_channel_capacity: 0,
            policy_path: String::new(),
            inflow_method: InflowNonNegativityMethod::None,
            cut_selection: None,
            cut_activity_tolerance: 0.0,
            budget: None,
            export_states: false,
            scalar_parameters: Vec::new(),
            training_solver_backward: None,
            training_solver_forward: None,
            simulation_solver: None,
            backward_scheduler,
            cost_scale_factor: DEFAULT_COST_SCALE_FACTOR,
            boundary: cobre_sddp::BoundaryStateRequirements::none(),
        }
    }

    /// Train the enumerated-external chain under `backward_scheduler` and return
    /// the converged `(LB, UB)`. Both backward schedulers route an External
    /// successor node through the same `eta_slice(stage, k)` read.
    fn train_final_bounds(backward_scheduler: cobre_io::config::BackwardScheduler) -> (f64, f64) {
        let system = build_system();
        let source = ScenarioSource {
            inflow_scheme: SamplingScheme::External,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            seed: Some(42),
            historical_years: None,
        };
        let stochastic = build_stochastic_context(
            &system,
            42,
            Some(42),
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::External),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .expect("build_stochastic_context must succeed");
        assert!(
            stochastic.n_stochastic_ncs() == 0,
            "fixture must carry no NCS noise dimension"
        );

        let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
        let mut setup = StudySetup::from_broadcast_params(
            &system,
            stochastic,
            construction_config(backward_scheduler, 0),
            hydro_models,
            &source,
            &source,
        )
        .expect("StudySetup::from_broadcast_params must build");
        assert!(
            setup
                .inputs
                .scenario_libraries
                .training
                .external_inflow
                .is_some(),
            "external inflow library must be present under the External scheme"
        );

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("enumerated-external training must return Ok");
        assert!(outcome.error.is_none(), "expected no training error");
        (outcome.result.final_lb, outcome.result.final_ub)
    }

    /// By-scenario backward (default): forward and backward both read
    /// `eta_slice(stage, k)` for the pinned column, so the exact upper bound over
    /// the declared path meets the lower bound.
    #[test]
    fn enumerated_external_chain_lb_equals_ub_by_scenario() {
        let (lb, ub) = train_final_bounds(cobre_io::config::BackwardScheduler::ByScenario {});
        assert!(
            (ub - lb).abs() < 1e-6,
            "by-scenario enumerated-external forward≡backward must converge LB==UB: \
             LB = {lb}, UB = {ub}",
        );
    }

    /// By-node backward: the opening-block scheduler's claim loop takes the same
    /// External `eta_slice(stage, k)` read, so it converges to the identical
    /// LB==UB as the by-scenario path.
    #[test]
    fn enumerated_external_chain_lb_equals_ub_by_node() {
        let (lb, ub) =
            train_final_bounds(cobre_io::config::BackwardScheduler::ByNode { block_size: None });
        assert!(
            (ub - lb).abs() < 1e-6,
            "by-node enumerated-external forward≡backward must converge LB==UB: \
             LB = {lb}, UB = {ub}",
        );
    }

    /// Train the enumerated-external chain, then run `n_sim` sampled simulation
    /// scenarios over the same node graph, returning `(LB, UB, per-scenario
    /// simulation results sorted by scenario id)`.
    fn train_and_simulate(n_sim: u32) -> (f64, f64, Vec<SimulationScenarioResult>) {
        let system = build_system();
        let source = ScenarioSource {
            inflow_scheme: SamplingScheme::External,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            seed: Some(42),
            historical_years: None,
        };
        let stochastic = build_stochastic_context(
            &system,
            42,
            Some(42),
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::External),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .expect("build_stochastic_context must succeed");

        let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
        let mut setup = StudySetup::from_broadcast_params(
            &system,
            stochastic,
            construction_config(cobre_io::config::BackwardScheduler::ByScenario {}, n_sim),
            hydro_models,
            &source,
            &source,
        )
        .expect("StudySetup::from_broadcast_params must build");

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("enumerated-external training must return Ok");
        assert!(outcome.error.is_none(), "expected no training error");
        let (lb, ub) = (outcome.result.final_lb, outcome.result.final_ub);

        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("create_workspace_pool must succeed");
        let (result_tx, result_rx) = mpsc::sync_channel(usize::try_from(n_sim.max(1)).unwrap());
        let drain = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());
        setup
            .simulate(
                &mut pool.workspaces,
                &comm,
                &result_tx,
                None,
                None,
                &outcome.result.basis_cache,
            )
            .expect("simulate must return Ok");
        drop(result_tx);
        let mut results = drain.join().expect("drain thread must not panic");
        results.sort_by_key(|r| r.scenario_id);
        (lb, ub, results)
    }

    /// Simulation honors the node's external pin: every simulated scenario replays
    /// the DECLARED column 0 (cost == UB == LB), even though the library carries a
    /// distinct decoy column 1. A dropped pin would hash-select over both columns
    /// (`select_external_scenario` is stage-independent, so a scenario reads one
    /// column for all its stages) and any scenario landing on the decoy would price
    /// a different, higher-inflow-deprived cost — so `cost == UB` for every scenario
    /// is the replay proof, not a hash-drawn value.
    #[test]
    fn enumerated_external_simulation_replays_the_declared_column() {
        // The decoy column must differ from the declared one, else the fixture
        // could not discriminate a pinned replay from a hash-drawn one.
        assert_ne!(
            EXTERNAL_INFLOW_M3S, DECOY_INFLOW_M3S,
            "declared and decoy external columns must differ for the test to have power",
        );
        let (lb, ub, results) = train_and_simulate(8);
        assert!(
            (ub - lb).abs() < 1e-6,
            "enumerated-external forward≡backward must converge LB==UB: LB = {lb}, UB = {ub}",
        );
        assert!(
            !results.is_empty(),
            "simulation must produce at least one scenario"
        );
        // The realized incremental inflow each stage is the pinned column's raw
        // value (no PAR lags), so it discriminates the declared column from the
        // decoy independently of the deficit-dominated cost: a dropped pin would
        // hash-select and the scenarios landing on the decoy would report its
        // inflow instead. The hydro is the sole entity, so `hydros[0]`.
        for r in &results {
            assert_eq!(
                r.stages.len(),
                N_STAGES,
                "each simulated scenario must span every stage",
            );
            for (t, stage) in r.stages.iter().enumerate() {
                let realized = stage.hydros[0].incremental_inflow_m3s;
                assert!(
                    (realized - EXTERNAL_INFLOW_M3S[t]).abs() < 1e-6,
                    "scenario {} stage {t}: simulation must replay declared column \
                     {} m³/s, got {realized} (decoy is {})",
                    r.scenario_id,
                    EXTERNAL_INFLOW_M3S[t],
                    DECOY_INFLOW_M3S[t],
                );
            }
            assert!(
                (r.total_cost - ub).abs() < 1e-6,
                "simulated scenario cost must match the exact bound UB {ub}, got {}",
                r.total_cost,
            );
        }
    }
}

/// A reduced-projection pool (storage-only cut state, `n_slots < n_state`) must
/// score cuts against trial states read in its OWN projected space. The
/// visited-states archive is packed at the global StateDim stride; striding it
/// by the pool's projected dimension — the pre-fix behavior — reads misaligned
/// memory and deactivates the wrong cuts. This pins the projected read against
/// that misaligned read on the storage-only cut-state axis, where they differ.
#[test]
fn cut_selection_scores_reduced_projection_in_projected_space() {
    use cobre_core::temporal::StageStateConfig;
    use cobre_sddp::cut::CutPool;
    use cobre_sddp::cut_selection::CutSelectionStrategy;
    use cobre_sddp::indexer::{CutSlot, CutStateProjection, StateSpace};
    use cobre_sddp::lead_time::AnticipatedResolution;
    use cobre_sddp::setup::NodeId;

    // 1 hydro, PAR(1): global state = [storage, lag] → n_state = 2.
    let global = StateSpace::new(
        1,
        1,
        Vec::new(),
        Vec::new(),
        AnticipatedResolution::default(),
        &[1],
    );
    assert_eq!(global.n_state, 2);

    let proj = CutStateProjection::new(
        &global,
        StageStateConfig {
            storage: true,
            inflow_lags: false,
        },
    );
    let n_slots = proj.n_slots();
    assert!(
        n_slots < global.n_state,
        "precondition: strictly reduced projection (n_slots {n_slots} < n_state {})",
        global.n_state
    );

    // 1-D cuts: cut0 = x, cut1 = 3, cut2 = 0.5. Over the storage values only,
    // cut1 is always at-max, so cut0 and cut2 are dominated.
    let mut pool = CutPool::new(3, n_slots, 1, 0);
    pool.add_cut(NodeId(0), 0, 0, 0.0, &[1.0]);
    pool.add_cut(NodeId(0), 1, 0, 3.0, &[0.0]);
    pool.add_cut(NodeId(0), 2, 0, 0.5, &[0.0]);

    let strategy = CutSelectionStrategy::Level1 {
        check_frequency: 1,
        tie_tolerance: 0.0,
    };

    // Two trial points, packed at the global StateDim stride [storage, lag]:
    // small storage, large lag so the misalignment is observable.
    let n_trials = 2;
    let global_states = [1.0, 10.0, 2.0, 20.0];

    let mut projected = Vec::new();
    for m in 0..n_trials {
        for s in 0..n_slots {
            let g = proj.global_state_index(CutSlot::new(s)).get();
            projected.push(global_states[m * global.n_state + g]);
        }
    }
    assert_eq!(
        projected,
        vec![1.0, 2.0],
        "storage-only projection selects the storage StateDim, not a prefix stride"
    );

    let mut projected_deact = strategy
        .select_for_stage(&pool, &projected, n_trials, 10, 0)
        .updates;
    projected_deact.sort_unstable();

    // Pre-fix misaligned read: stride the global buffer by the pool dim,
    // treating the 4 packed values as 4 one-dim states {1, 10, 2, 20}. cut0 is
    // at-max at the large lag values, so it survives — the wrong deactivation set.
    let derived_trials = global_states.len() / n_slots;
    let mut misaligned_deact = strategy
        .select_for_stage(&pool, &global_states, derived_trials, 10, 0)
        .updates;
    misaligned_deact.sort_unstable();

    assert_eq!(
        projected_deact,
        vec![0, 2],
        "projected read deactivates cut0 (never at-max among the storage states) and cut2"
    );
    assert_eq!(
        misaligned_deact,
        vec![2],
        "misaligned read wrongly keeps cut0 (at-max at lag values read as storage)"
    );
    assert_ne!(
        projected_deact, misaligned_deact,
        "the projection must change the deactivation set on a reduced pool"
    );
}

/// Dual-folding verification.
///
/// On a deterministic PAR(1) trunk the inflow lag is redundant state, so the two
/// representations the engine supports — folded (`inflow_lags: false`, storage-only
/// trunk cuts, the production default; the lag is bound-fixed and priced through the
/// storage future cost) and unfolded (`inflow_lags: true`, the lag carried as an
/// explicit cut-state dimension) — must agree on the deterministic trunk.
///
/// Empirically the fold-invariant quantities are:
///   * the trunk **cut coefficients**: the two builds share one stage LP (identical
///     columns, pins, and successor solves) and one deterministic forward
///     trajectory, so the incoming-storage reduced cost — the storage subgradient —
///     is **bit-identical** across every trunk cut; only the cut PROJECTION differs
///     (the unfolded build additionally carries a nonzero lag coefficient the folded
///     build absorbs into the intercept);
///   * the **first-stage policy**: both builds realize the same optimal policy, so
///     the policy cost (`final_ub`) agrees to a stated tolerance — the folded and
///     unfolded LPs are different, so the simulation may settle on a
///     different-but-valid vertex at the degenerate optimum (~1 ULP here), the
///     hot != cold divergence the determinism contract permits (never asserted equal
///     bit-for-bit).
///
/// `final_lb` is deliberately NOT compared between the builds: a non-vacuous fold
/// (nonzero lag coefficient) forces the unfolded build to carry a redundant lag
/// dimension the deterministic forward under-samples, so its lower bound stays loose
/// while the folded build converges (`lb == ub`). That is the correct finding — it
/// is why folding is the default — not an agreement quantity.
mod dual_folding_f34 {
    use cobre_sddp::StudySetup;
    use cobre_sddp::test_support::{LagFold, dual_folding_setup, pool_cut_state_dimensions};
    use cobre_solver::ActiveSolver;

    use crate::common::StubComm;

    const FORWARD_PASSES: u32 = 3;
    const MAX_ITERATIONS: u32 = 12;
    /// Relative tolerance for the converged-value / policy-cost comparisons: a
    /// different-but-valid vertex at the degenerate optimum agrees only to solver
    /// optimality tolerance, never to the bit (the observed `final_ub` gap is ~1 ULP).
    const VALUE_REL_TOL: f64 = 1e-6;
    /// A coefficient magnitude above this counts as genuinely nonzero — the trunk
    /// storage/lag coefficients are ~0.07/0.14, far above solver noise.
    const NONZERO_EPS: f64 = 1e-9;

    struct Trained {
        setup: StudySetup,
        lb: f64,
        ub: f64,
    }

    fn train(fold: LagFold) -> Trained {
        let mut setup = dual_folding_setup(fold, FORWARD_PASSES, MAX_ITERATIONS);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("dual-folding training must return Ok");
        assert!(
            outcome.error.is_none(),
            "dual-folding training must not error: {:?}",
            outcome.error
        );
        Trained {
            lb: outcome.result.final_lb,
            ub: outcome.result.final_ub,
            setup,
        }
    }

    /// Pool ids of the trunk nodes (nodes with a successor own their own pool).
    fn trunk_pool_ids(setup: &StudySetup) -> Vec<usize> {
        let mut ids = Vec::new();
        for (pos, node) in setup.inputs.node_graph.nodes.iter_indexed() {
            if !setup.inputs.node_graph.successors[pos].is_empty() && !ids.contains(&node.pool_id) {
                ids.push(node.pool_id);
            }
        }
        ids.sort_unstable();
        ids
    }

    /// Widest successor count over all nodes — the terminal fan's width.
    fn fan_width(setup: &StudySetup) -> usize {
        setup
            .inputs
            .node_graph
            .nodes
            .iter_indexed()
            .map(|(pos, _)| setup.inputs.node_graph.successors[pos].len())
            .max()
            .unwrap_or(0)
    }

    /// Bit pattern of the storage coefficient (cut-state slot 0) of every active cut
    /// in `pool`, in the canonical append-only order.
    fn storage_coeff_bits(setup: &StudySetup, pool: usize) -> Vec<u64> {
        setup
            .fcf
            .active_cuts(pool)
            .map(|(_slot, _intercept, coeffs)| coeffs[0].to_bits())
            .collect()
    }

    /// Largest absolute value of coefficient `slot` over `pool`'s active cuts.
    fn max_abs_coeff(setup: &StudySetup, pool: usize, slot: usize) -> f64 {
        setup
            .fcf
            .active_cuts(pool)
            .map(|(_s, _i, coeffs)| coeffs.get(slot).copied().unwrap_or(0.0).abs())
            .fold(0.0_f64, f64::max)
    }

    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: trains two policies; run with --features slow-tests"
    )]
    #[test]
    fn f34_dual_folding_trunk_agreement() {
        let folded = train(LagFold::Folded);
        let unfolded = train(LagFold::Unfolded);

        let trunk_a = trunk_pool_ids(&folded.setup);
        let trunk_b = trunk_pool_ids(&unfolded.setup);
        assert_eq!(
            trunk_a, trunk_b,
            "folded and unfolded share one graph, so their trunk pools coincide"
        );

        // ── Power preconditions — the agreement must not be vacuous ──
        assert!(
            fan_width(&folded.setup) >= 2,
            "terminal fan must carry >= 2 successors (got {})",
            fan_width(&folded.setup)
        );
        assert!(
            folded.setup.stage_state().max_par_order > 0,
            "trunk must carry a genuine inflow lag (max_par_order > 0)"
        );

        let dims_a = pool_cut_state_dimensions(&folded.setup);
        let dims_b = pool_cut_state_dimensions(&unfolded.setup);
        for &p in &trunk_a {
            assert!(
                dims_b[p] > dims_a[p],
                "fold must be structurally non-vacuous: unfolded trunk pool {p} must \
                 project more cut-state dims ({}) than folded ({})",
                dims_b[p],
                dims_a[p]
            );
        }

        // Numerical non-vacuity: the unfolded build carries a nonzero lag coefficient
        // (the fold folds something real), and the shared storage coefficient is
        // nonzero (so its bit-identity below has power).
        for &p in &trunk_b {
            assert!(
                max_abs_coeff(&unfolded.setup, p, 1) > NONZERO_EPS,
                "unfolded trunk pool {p} must carry a nonzero lag coefficient"
            );
        }
        for &p in &trunk_a {
            assert!(
                max_abs_coeff(&folded.setup, p, 0) > NONZERO_EPS,
                "folded trunk pool {p} must carry a nonzero storage coefficient"
            );
        }

        // ── Trunk agreement ──
        // (1) Trunk cut coefficients: the storage subgradient is bit-identical. The
        //     two builds solve one identical stage LP over one identical deterministic
        //     trajectory; only the cut projection differs, so the incoming-storage
        //     reduced cost is fixed bit-for-bit by the determinism contract.
        for (&pa, &pb) in trunk_a.iter().zip(trunk_b.iter()) {
            assert_eq!(
                storage_coeff_bits(&folded.setup, pa),
                storage_coeff_bits(&unfolded.setup, pb),
                "trunk storage cut coefficients must be bit-identical (pool {pa})"
            );
        }

        // (2) The folded (default) build converges: its lower bound reaches the
        //     policy cost, so `final_ub` below is the true optimum.
        assert!(
            (folded.lb - folded.ub).abs() <= VALUE_REL_TOL * folded.ub.abs(),
            "folded build must converge: lb {} vs ub {}",
            folded.lb,
            folded.ub
        );

        // (3) First-stage policy: both builds realize the same optimum, compared to a
        //     stated tolerance because the different LPs may reach a
        //     different-but-valid vertex (never bit-for-bit; the determinism contract
        //     forbids asserting hot == cold across a representation change).
        assert!(
            (folded.ub - unfolded.ub).abs() <= VALUE_REL_TOL * folded.ub.abs(),
            "first-stage policy cost must agree within tolerance: folded {} vs unfolded {}",
            folded.ub,
            unfolded.ub
        );

        // (4) The unfolded lower bound is valid (never exceeds the shared optimum),
        //     even though it stays loose — the redundant deterministic lag is
        //     under-sampled, so its gap does not close.
        assert!(
            unfolded.lb <= unfolded.ub * (1.0 + VALUE_REL_TOL),
            "unfolded lower bound {} must not exceed the optimum {}",
            unfolded.lb,
            unfolded.ub
        );
    }
}

mod f36_gap_attainability {
    //! Fixture-based gap-attainability check on the enumerated DECOMP K-fan.
    //! Under enumerated forwards + an expectation measure, the `Gap { tolerance }`
    //! stopping rule halts training once the EXACT canonical-R$ gap
    //! `final_ub − final_lb` closes to `<= tolerance`. `final_ub` is exact under
    //! enumerated (`final_ub_std == 0`), so the gap is a hard quantity, not a
    //! statistical estimate. `Gap` is admissible only under enumerated +
    //! expectation (rejected elsewhere at setup — that admission test is owned
    //! elsewhere and not re-authored here).

    use cobre_sddp::stopping_rule::RULE_GAP;
    use cobre_sddp::test_support::k_fan_setup_gap;
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    /// Fan width and iteration budget. The enumerated K-fan learns its exact policy
    /// in a single backward pass, so the gap closes well within the budget; the
    /// budget only bounds a hypothetical non-converging run.
    const K: usize = 4;
    const MAX_ITERATIONS: u32 = 30;

    /// Absolute gap tolerance, canonical R$. The fixture's optimum is ~8.928e7 R$
    /// and the converged gap sits at the LP-solver ULP floor (~1e-8 R$), so 10.0 R$
    /// is comfortably above the feasibility-noise floor (~1e-9 × the objective
    /// magnitude ≈ 0.09 R$) yet far below the objective — a genuinely attainable
    /// absolute tolerance. A tolerance below the noise floor would leave the gap
    /// `> tolerance`, so the run would stop on a NON-`gap` rule instead — the
    /// mandatory `iteration_limit`, the only other rule this fixture declares — and
    /// the `reason == gap` assertion below is that unattainability failure signal.
    const GAP_TOLERANCE_R: f64 = 10.0;

    /// Train the gap-configured fixture at `cost_scale_factor`, returning
    /// `(final_lb, final_ub, final_ub_std, reason)`.
    fn train_gap(cost_scale_factor: Option<f64>) -> (f64, f64, f64, String) {
        let mut fixture = k_fan_setup_gap(K, GAP_TOLERANCE_R, MAX_ITERATIONS, cost_scale_factor);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let outcome = fixture
            .setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("gap-configured K-fan training must return Ok");
        assert!(
            outcome.error.is_none(),
            "gap-configured K-fan training must not error: {:?}",
            outcome.error
        );
        let r = &outcome.result;
        (r.final_lb, r.final_ub, r.final_ub_std, r.reason.clone())
    }

    /// Attainability (+ expectation-only): on the enumerated + expectation fixture,
    /// the achieved canonical-R$ gap closes to `<= GAP_TOLERANCE_R` and the run
    /// stops on the `gap` rule (not the mandatory `iteration_limit`) — the
    /// attainability signal.
    #[test]
    fn gap_tolerance_attained_under_enumerated_expectation() {
        let (lb, ub, ub_std, reason) = train_gap(None);

        assert_eq!(
            ub_std, 0.0,
            "enumerated forwards give an exact UB: final_ub_std must be 0"
        );
        let gap = ub - lb;
        assert!(
            gap <= GAP_TOLERANCE_R,
            "achieved canonical-R$ gap {gap} must close to <= tolerance {GAP_TOLERANCE_R}"
        );
        assert_eq!(
            reason, RULE_GAP,
            "must stop on the gap rule (attainable tolerance); a non-gap stop \
             ({reason}) with gap > tolerance is the unattainability failure signal"
        );
    }

    /// Canonical-units pinning: the absolute gap comparison is on the UNSCALED
    /// (canonical-R$) side. The same fixture at `cost_scale_factor == 1.0`
    /// (unscaled LP) and at a non-unit factor converges to the same
    /// gap-in-canonical-R$ and the same lower bound. A comparison performed on the
    /// SCALED bounds would shift the reported bounds by the factor (the non-unit
    /// run's `final_lb` would be `× 1/factor`) and fail the lb-match. Mirrors
    /// `d02_single_hydro_cost_scale_invariant`.
    #[test]
    fn gap_comparison_is_canonical_units_invariant_to_cost_scale() {
        let (lb_unit, ub_unit, _, reason_unit) = train_gap(Some(1.0));
        let (lb_scaled, ub_scaled, _, reason_scaled) = train_gap(Some(1e6));

        assert_eq!(
            reason_unit, RULE_GAP,
            "unit-factor run must stop on the gap rule"
        );
        assert_eq!(
            reason_scaled, RULE_GAP,
            "non-unit-factor run must stop on the gap rule"
        );

        let gap_unit = ub_unit - lb_unit;
        let gap_scaled = ub_scaled - lb_scaled;
        assert!(
            gap_unit <= GAP_TOLERANCE_R && gap_scaled <= GAP_TOLERANCE_R,
            "both runs' canonical-R$ gap must close to <= tolerance \
             (unit {gap_unit}, scaled {gap_scaled})"
        );

        // Same gap-in-canonical-R$: both converge to the ~0 gap and must agree far
        // tighter than the stopping tolerance (a genuine equality, not a restatement
        // of `both <= tolerance`). A scaled-side comparison would let the non-unit
        // run stop at a canonical gap up to tolerance × factor, diverging here.
        let gap_match_tol = 1e-3;
        assert!(
            (gap_unit - gap_scaled).abs() <= gap_match_tol,
            "the two runs must close to the same gap-in-canonical-R$ \
             (unit {gap_unit}, scaled {gap_scaled}, tol {gap_match_tol})"
        );
        // The lower bound is scale-invariant in canonical R$ — this carries the
        // pin's power: a scaled-bounds comparison/report would shift it by the
        // factor (the ~8.928e7 R$ optimum vs a ~1/factor-shifted value). The
        // magnitude-relative tolerance admits scaling roundoff yet rejects the
        // factor-sized shift.
        let lb_match_tol = 1e-6 * ub_unit.abs().max(1.0);
        assert!(
            (lb_unit - lb_scaled).abs() <= lb_match_tol,
            "canonical-R$ lower bound must be invariant to cost_scale_factor \
             (unit {lb_unit}, scaled {lb_scaled}, tol {lb_match_tol})"
        );
    }
}

/// Enumerated-study checkpoint export→re-load round-trip under the stride-1
/// `cut_id` numbering. The sampled half is `d12_checkpoint_round_trip`; this pins
/// the enumerated half — an enumerated policy's dense stride-1 `cut_id`
/// (`iteration_generated * visit_stride + forward_pass_index` with `visit_stride
/// == 1` and `forward_pass_index == 0`, so `cut_id == iteration` for a non-leaf
/// pool, never `iteration ×` a larger stride) survives write→read intact. The
/// trunk+fan enumerated graph carries several non-leaf pools (root plus each
/// fan node) and one shared leaf pool, so the round-trip is pinned across both
/// stride-1 and stride-0 pools on both backends.
#[cfg(feature = "test-support")]
mod enumerated_checkpoint {
    use std::collections::HashSet;

    use cobre_io::{
        GraphManifest, ProducerBlock, STAGE_CUTS_NODE_ID_SENTINEL,
        STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, StageCutsPayload, read_policy_checkpoint,
        write_policy_checkpoint,
    };
    use cobre_sddp::policy_export::build_stage_cut_records;
    use cobre_sddp::setup::NodePos;
    use cobre_sddp::test_support::k_fan_setup_enumerated;
    use cobre_solver::ActiveSolver;

    use crate::common::StubComm;

    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    #[test]
    fn enumerated_checkpoint_round_trips_under_stride_one_cut_ids() {
        const K: usize = 3;
        const MAX_ITERATIONS: u32 = 4;

        let mut fixture = k_fan_setup_enumerated(K, MAX_ITERATIONS);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let outcome = fixture
            .setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("enumerated training must return Ok");
        assert!(
            outcome.error.is_none(),
            "enumerated training error: {:?}",
            outcome.error
        );
        let result = outcome.result;
        // Required side effect before the cut extraction below (mirrors d12).
        let _training_output = fixture.setup.build_training_output(&result, &[]);

        // Partition pools into leaf (stride 0) and non-leaf (stride 1).
        let node_graph = &fixture.setup.inputs.node_graph;
        let mut leaf_pools: HashSet<usize> = HashSet::new();
        let mut nonleaf_pools: HashSet<usize> = HashSet::new();
        for i in 0..node_graph.nodes.len() {
            let pos = NodePos(i);
            let pool = node_graph.nodes[pos].pool_id;
            if node_graph.successors[pos].is_empty() {
                leaf_pools.insert(pool);
            } else {
                nonleaf_pools.insert(pool);
            }
        }
        assert!(
            !nonleaf_pools.is_empty() && !leaf_pools.is_empty(),
            "the trunk+fan fixture must exercise both non-leaf and leaf pools"
        );

        let fcf = &fixture.setup.fcf;
        // The stride that drives the enumerated cut_id numbering.
        for &p in &nonleaf_pools {
            assert_eq!(
                fcf.pools[p].visit_stride, 1,
                "non-leaf pool {p} sizes at stride 1"
            );
        }
        for &p in &leaf_pools {
            assert_eq!(
                fcf.pools[p].visit_stride, 0,
                "shared leaf pool {p} sizes at stride 0"
            );
        }

        // The real export path: cut_id = iteration_generated * visit_stride + fp.
        let cut_records_per_stage = build_stage_cut_records(fcf);

        // Write-side stride-1 pin: a non-leaf pool's cut_id is dense (== its
        // iteration), never iteration × a larger stride; the leaf pool emits none.
        let mut nonleaf_cut_count = 0usize;
        for (pool_idx, records) in cut_records_per_stage.iter().enumerate() {
            if nonleaf_pools.contains(&pool_idx) {
                for rec in records {
                    assert_eq!(
                        rec.cut_id,
                        u64::from(rec.iteration),
                        "non-leaf pool {pool_idx}: stride-1 cut_id must equal its iteration"
                    );
                    nonleaf_cut_count += 1;
                }
            } else {
                assert!(
                    records.is_empty(),
                    "leaf pool {pool_idx} generates no cut (stride 0)"
                );
            }
        }
        assert!(
            nonleaf_cut_count > 0,
            "training must produce non-leaf cuts for the round-trip to pin"
        );

        let active_indices_per_stage: Vec<Vec<u32>> = fcf
            .pools
            .iter()
            .map(|pool| {
                (0..pool.populated())
                    .filter(|&slot| pool.is_active(slot))
                    .map(|slot| slot as u32)
                    .collect()
            })
            .collect();

        // Mirrors `sole_pool_owner_node_id` (the production writer's
        // algorithm in `policy_export.rs`): every non-leaf pool has exactly
        // one owning node; the shared leaf pool the fixture doc above
        // names keeps the sentinel.
        let pool_owner_node_id = |pool: usize| -> i32 {
            let mut owner: Option<i32> = None;
            for (pos, node) in node_graph.nodes.iter_indexed() {
                if node.pool_id == pool {
                    if owner.is_some() {
                        return STAGE_CUTS_NODE_ID_SENTINEL;
                    }
                    owner = Some(node_graph.node_ids[pos].0);
                }
            }
            owner.unwrap_or(STAGE_CUTS_NODE_ID_SENTINEL)
        };

        let stage_cuts_payloads: Vec<StageCutsPayload<'_>> = fcf
            .pools
            .iter()
            .enumerate()
            .map(|(pool_idx, pool)| StageCutsPayload {
                stage_id: pool_idx as u32,
                state_dimension: pool.state_dimension as u32,
                capacity: pool.capacity as u32,
                warm_start_count: pool.warm_start_count,
                cuts: &cut_records_per_stage[pool_idx],
                active_cut_indices: &active_indices_per_stage[pool_idx],
                populated_count: pool.populated() as u32,
                entity_manifest: &[],
                cost_scale_factor: 1_000_000.0,
                node_id: pool_owner_node_id(pool_idx),
                graph_stage_id: -1,
                priced_state_date: STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
            })
            .collect();

        let n_pools = fcf.pools.len();
        let warm_start_counts: Vec<u32> = fcf.pools.iter().map(|p| p.warm_start_count).collect();
        let policy_metadata = cobre_sddp::test_support::checkpoint_metadata(
            n_pools as u32,
            GraphManifest::default(),
            ProducerBlock {
                completed_iterations: result.iterations as u32,
                final_lower_bound: result.final_lb,
                best_upper_bound: Some(result.final_ub),
                max_iterations: MAX_ITERATIONS,
                forward_passes: fixture.forward_passes,
                warm_start_cuts: warm_start_counts.iter().copied().max().unwrap_or(0),
                warm_start_counts,
                rng_seed: 42,
                total_visited_states: 0,
                training_block_mode: "parallel".to_string(),
                training_block_mode_per_stage: vec![],
                cost_scale_factor: None,
            },
        );

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let policy_dir = tmp.path().join("policy");
        write_policy_checkpoint(
            &policy_dir,
            &stage_cuts_payloads,
            &[],
            &policy_metadata,
            &[],
        )
        .expect("write_policy_checkpoint must succeed");
        let checkpoint =
            read_policy_checkpoint(&policy_dir).expect("read_policy_checkpoint must succeed");

        assert_eq!(
            checkpoint.metadata.num_stages as usize, n_pools,
            "num_stages round-trips"
        );
        assert_eq!(checkpoint.stage_cuts.len(), n_pools, "one payload per pool");

        // Every pool's cuts round-trip byte-for-byte, and every reloaded non-leaf
        // cut still carries the dense stride-1 cut_id (== its iteration).
        for (pool_idx, stage) in checkpoint.stage_cuts.iter().enumerate() {
            let written = &cut_records_per_stage[pool_idx];
            assert_eq!(
                stage.cuts.len(),
                written.len(),
                "pool {pool_idx} cut count round-trips"
            );
            assert_eq!(
                stage.state_dimension as usize, fcf.pools[pool_idx].state_dimension,
                "pool {pool_idx} state_dimension round-trips"
            );
            for (loaded, w) in stage.cuts.iter().zip(written) {
                assert_eq!(loaded.cut_id, w.cut_id, "cut_id round-trips");
                assert_eq!(loaded.slot_index, w.slot_index, "slot_index round-trips");
                assert_eq!(loaded.iteration, w.iteration, "iteration round-trips");
                assert_eq!(
                    loaded.intercept.to_bits(),
                    w.intercept.to_bits(),
                    "intercept round-trips bit-for-bit"
                );
                assert_eq!(
                    loaded.coefficients.as_slice(),
                    w.coefficients,
                    "coefficients round-trip bit-for-bit"
                );
            }
            if nonleaf_pools.contains(&pool_idx) {
                for loaded in &stage.cuts {
                    assert_eq!(
                        loaded.cut_id,
                        u64::from(loaded.iteration),
                        "reloaded non-leaf pool {pool_idx}: stride-1 cut_id survives the round-trip"
                    );
                }
            }
        }
    }
}

/// Terminal in-transit water-bucket keep-live + FCF valuation when a boundary
/// FCF is present: `build_transit_bucket_topology`'s terminal-lag mask
/// un-caps only when `config.policy.boundary` is declared, so a
/// zero-terminal-value study (no boundary) keeps the masked layout
/// byte-for-byte (Terminal credit deferred), while a boundary-loaded study
/// carries the deep-lag terminal slots live and prices them through the
/// already-generic cut-state projection (the Delivery-family right-boundary
/// pricing contract). Reuses d45's already-vetted
/// mixed-calendar arc (depth 3 into a 4-stage horizon, so the terminal's own
/// uncapped reach is nonzero) rather than a fresh fixture.
#[cfg(feature = "test-support")]
// Rationale (cast_sign_loss): `StageTemplate`'s CSC arrays (`col_starts`,
// `row_indices`) are solver-side `i32` indices that are always non-negative
// by construction (row/column counts, never a sentinel or delta).
#[allow(clippy::cast_sign_loss)]
mod water_terminal_fcf_valuation {
    use std::path::Path;

    use chrono::NaiveDate;
    use cobre_core::EntityId;
    use cobre_core::temporal::StageStateConfig;
    use cobre_io::BoundaryPolicy;
    use cobre_sddp::indexer::CutStateProjection;
    use cobre_sddp::setup::{NodeId, StageIdx};
    use cobre_sddp::test_support::{patch_backward_opening_for_probe, solve_stage_for_probe};
    use cobre_sddp::workspace::SolverWorkspace;
    use cobre_sddp::{
        BoundaryLoadRequest, CutPool, StudySetup, build_cut_row_batch_into, inject_boundary_cuts,
        load_boundary_cuts,
    };
    use cobre_solver::{
        ActiveSolver, FreezeScratch, RowBatch, SolverInterface, StageTemplate,
        freeze_rows_into_template,
    };

    use crate::common::{StubComm, fresh_setup_with};

    const CASE_DIR: &str = "../../examples/deterministic/d45-travel-time-mixed-calendar";
    const TRAVEL_TIME_HOURS: f64 = 360.0;
    const STAGE_HOURS: [f64; 4] = [720.0, 168.0, 168.0, 168.0];
    const DOWNSTREAM_ID: EntityId = EntityId(1);
    /// Boundary cut intercept `alpha`, positive so the injected cut binds `theta`
    /// above its zero floor for every in-transit volume `x >= 0`.
    const ALPHA: f64 = 5.0;
    /// Boundary cut coefficient `beta` on the terminal bucket slot.
    const BETA: f64 = 3.0;
    const TOL: f64 = 1e-6;

    fn case_dir() -> &'static Path {
        Path::new(CASE_DIR)
    }

    fn boundary_policy() -> BoundaryPolicy {
        BoundaryPolicy {
            path: "unused".to_string(),
            strict: false,
        }
    }

    /// This gate's power precondition: the terminal stage's own uncapped
    /// reach (`resolve_spread` at the last stage, padded past the horizon)
    /// must be nonzero, or un-capping the mask would be a no-op.
    fn terminal_uncapped_reach() -> usize {
        let padded = super::pad_calendar_for_resolution(&STAGE_HOURS, TRAVEL_TIME_HOURS);
        let terminal = STAGE_HOURS.len() - 1;
        super::resolve_spread(TRAVEL_TIME_HOURS, terminal, &padded, None).stage_reach
    }

    /// J's (the downstream plant's) global bucket state index for `lag`
    /// (`state.transit_buckets_out.start` plus `column_order`'s position),
    /// resolved by canonical hydro index — the single derivation site every
    /// helper below reads through.
    fn bucket_state_index(setup: &StudySetup, lag: usize) -> usize {
        let system = cobre_io::load_case(case_dir()).expect("load_case must succeed");
        let j_idx = system
            .hydros()
            .iter()
            .position(|h| h.id == DOWNSTREAM_ID)
            .expect("downstream plant must exist in the canonical hydro order");
        let state = setup.stage_state();
        let pos = state
            .transit_bucket_column_order
            .iter()
            .position(|&(p, l)| p.get() == j_idx && l == lag)
            .unwrap_or_else(|| panic!("lag {lag} must be a declared bucket slot"));
        state.transit_buckets_out.start + pos
    }

    /// Count of nonzero entries in `template` landing on `row` — the
    /// structural half of the keep-live pin (no solve).
    fn row_nnz(template: &StageTemplate, row: usize) -> usize {
        (0..template.num_cols)
            .flat_map(|col| {
                let start = template.col_starts[col] as usize;
                let end = template.col_starts[col + 1] as usize;
                template.row_indices[start..end].iter().map(|&r| r as usize)
            })
            .filter(|&r| r == row)
            .count()
    }

    /// `col`'s own CSC entries in `template` — a live bucket outgoing column
    /// carries exactly one (its own `emit_shift_rows` `+1.0` definition row);
    /// a masked/frozen one carries none.
    fn col_entries(template: &StageTemplate, col: usize) -> &[i32] {
        let start = template.col_starts[col] as usize;
        let end = template.col_starts[col + 1] as usize;
        &template.row_indices[start..end]
    }

    /// Pool `pool`'s fixture `priced_state_date`: `2030-01-01` plus `pool`
    /// months.
    fn fixture_priced_date(pool: u32) -> NaiveDate {
        cobre_sddp::test_support::fixture_priced_date(
            cobre_sddp::test_support::ymd(2030, 1, 1),
            pool,
        )
    }

    /// Load a bucket-only boundary (`BETA` on `bucket_col`, zero elsewhere,
    /// intercept `ALPHA`) and inject it into `setup`'s terminal pool.
    fn inject_bucket_boundary(setup: &mut StudySetup, dir: &Path, bucket_col: usize) {
        let state_dimension = setup.fcf.state_dimension as u32;
        let mut coefficients = vec![0.0_f64; state_dimension as usize];
        coefficients[bucket_col] = BETA;
        cobre_sddp::test_support::write_synthetic_boundary(
            dir,
            state_dimension,
            ALPHA,
            &coefficients,
            fixture_priced_date(0),
        );

        let boundary_cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
            dir,
            fixture_priced_date(0),
            state_dimension,
            &[],
            1.0,
        ))
        .expect("boundary cut must load");
        inject_boundary_cuts(setup, &boundary_cuts).unwrap();
    }

    /// Build the terminal pool's FROZEN LP template: the base structural
    /// template plus one literal row per active cut, exactly as production
    /// freezes a pool once per iteration.
    fn freeze_terminal_template(setup: &StudySetup, pool_id: usize) -> StageTemplate {
        let state = setup.stage_state();
        let terminal_stage = setup.num_stages() - 1;
        let ctx = setup.stage_ctx();
        let base = &ctx.templates[terminal_stage];

        let cut_state = CutStateProjection::new(
            state,
            StageStateConfig {
                storage: true,
                inflow_lags: true,
            },
        );

        let mut batch = RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        };
        build_cut_row_batch_into(
            &mut batch,
            &setup.fcf,
            pool_id,
            state,
            &cut_state,
            &base.col_scale,
        );

        let mut frozen = StageTemplate::empty();
        let mut scratch = FreezeScratch::new();
        freeze_rows_into_template(base, &batch, &mut frozen, &mut scratch);
        frozen
    }

    /// Solve the terminal stage's frozen `template` against `pool` with the
    /// state vector pinned to `pinned_state`, returning `(theta, priced_col)`'s
    /// primal values — `priced_col` is read back from the SOLVED outgoing
    /// column (never assumed equal to a pin), since the ring's shift-row
    /// definition (`b_d^out = b_{d+1}^in + k_d * D`) makes the outgoing value
    /// an affine, not identity, function of whichever incoming slot is
    /// pinned. `raw_noise` is sized from `training_ctx.stochastic.dim()`
    /// (never hand-picked), so an all-zero draw is always the correct shape
    /// regardless of which fixture calls this.
    fn terminal_theta(
        setup: &StudySetup,
        template: &StageTemplate,
        pool: &CutPool,
        node_id: NodeId,
        pinned_state: &[f64],
        priced_col: usize,
    ) -> (f64, f64) {
        let comm = StubComm;
        let mut workspace_pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("create_workspace_pool");
        let ws: &mut SolverWorkspace<ActiveSolver> = &mut workspace_pool.workspaces[0];
        let ctx = setup.stage_ctx();
        let training_ctx = setup.training_ctx();
        let theta_col = setup.stage_state().theta;
        let terminal_stage = setup.num_stages() - 1;

        let raw_noise = vec![0.0_f64; training_ctx.stochastic.dim()];

        ws.solver.reset_solver_state();
        ws.solver.load_model(template);
        patch_backward_opening_for_probe(
            ws,
            &ctx,
            &training_ctx,
            StageIdx(terminal_stage),
            pinned_state,
            &raw_noise,
        );

        let view =
            solve_stage_for_probe(ws, &ctx, pool, None, StageIdx(terminal_stage), 0, node_id)
                .expect("terminal stage solve must not error");
        (view.primal[theta_col], view.primal[priced_col])
    }

    /// Pin the state vector with `x` on `bucket_col` and `0.0` elsewhere.
    fn bucket_pin(setup: &StudySetup, bucket_col: usize, x: f64) -> Vec<f64> {
        let mut pin = vec![0.0_f64; setup.stage_state().n_state];
        pin[bucket_col] = x;
        pin
    }

    /// The gated-off build reproduces today's capped mask (structural
    /// byte-neutrality anchor), the boundary-present build un-caps the
    /// terminal stage's own reach, sizing (`n_state`, bucket region ranges)
    /// stays identical across the gate, and the newly-live terminal row
    /// carries real nonzero coupling (not an empty shell).
    #[test]
    fn terminal_deep_lag_rows_and_columns_are_live_only_with_boundary_present() {
        let uncapped_terminal_reach = terminal_uncapped_reach();
        assert!(
            uncapped_terminal_reach >= 1,
            "gate has no power unless the terminal's own reach is nonzero (got \
             {uncapped_terminal_reach})"
        );

        let setup_off = fresh_setup_with(case_dir(), |_cfg| {});
        let setup_on = fresh_setup_with(case_dir(), |cfg| {
            cfg.policy.boundary = Some(boundary_policy());
        });

        let state_off = setup_off.stage_state();
        let state_on = setup_on.stage_state();
        assert_eq!(
            state_off.n_state, state_on.n_state,
            "n_state must be gate-invariant"
        );
        assert_eq!(state_off.n_buckets, state_on.n_buckets);
        assert_eq!(
            state_off.transit_bucket_column_order,
            state_on.transit_bucket_column_order
        );
        assert_eq!(state_off.transit_buckets_out, state_on.transit_buckets_out);
        assert_eq!(state_off.transit_buckets_in, state_on.transit_buckets_in);

        let terminal_stage = setup_off.num_stages() - 1;
        let template_off = &setup_off.stage_ctx().templates[terminal_stage];
        let template_on = &setup_on.stage_ctx().templates[terminal_stage];

        assert_eq!(
            template_on.num_rows,
            template_off.num_rows + uncapped_terminal_reach,
            "boundary-present must emit exactly the un-masked bucket-definition rows"
        );

        // The lag-1 slot's outgoing column follows row_pos automatically
        // (frozen [0,0] off, live [0, inf) on) with no separate column edit.
        let lag1_col = bucket_state_index(&setup_off, 1);
        assert_eq!(template_off.col_lower[lag1_col], 0.0);
        assert_eq!(template_off.col_upper[lag1_col], 0.0);
        assert_eq!(template_on.col_lower[lag1_col], 0.0);
        assert_eq!(template_on.col_upper[lag1_col], f64::INFINITY);

        // Off, the masked column defines no row at all; on, it defines
        // exactly one — a genuine equality carrying real coupling (the
        // shift-row identity, plus the release column whenever the arc
        // deposits at this lag/stage) — never an empty shell.
        assert!(
            col_entries(template_off, lag1_col).is_empty(),
            "a masked bucket outgoing column must define no row"
        );
        let on_entries = col_entries(template_on, lag1_col);
        assert_eq!(
            on_entries.len(),
            1,
            "a live bucket outgoing column must define exactly its own row"
        );
        let new_row = on_entries[0] as usize;
        assert_eq!(template_on.row_lower[new_row], 0.0);
        assert_eq!(template_on.row_upper[new_row], 0.0);
        assert!(
            row_nnz(template_on, new_row) >= 2,
            "a live bucket-definition row must carry at least its own shift-row identity"
        );
    }

    /// With no `policy.boundary`, `final_lb` reproduces its pre-boundary-gate
    /// value bit-for-bit — the gate's sole divergence is `config.policy.
    /// boundary` presence. The `to_bits` literal is HiGHS-only (`#[cfg]`-gated):
    /// bit-exactness is backend-specific, and CLP legitimately reaches a
    /// different-but-valid vertex; the cross-backend byte-neutrality guarantee
    /// is covered by the parity-hash reproduction in `parity.rs`.
    #[cfg(feature = "highs")]
    #[test]
    fn gated_off_final_lb_is_byte_stable() {
        let mut setup = fresh_setup_with(case_dir(), |_cfg| {});
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must return Ok");
        assert!(outcome.error.is_none(), "expected no training error");
        assert_eq!(
            outcome.result.final_lb.to_bits(),
            GATED_OFF_FINAL_LB_BITS,
            "gated-off final_lb must stay bit-identical to the pre-boundary-gate tree; a \
             mismatch means the boundary_present gate perturbed the no-boundary path"
        );
    }

    /// The gated-off `final_lb` on this exact case (seed 42, 1 thread,
    /// `StubComm`, `ActiveSolver`) — the HiGHS regression sentinel for
    /// `gated_off_final_lb_is_byte_stable`.
    #[cfg(feature = "highs")]
    const GATED_OFF_FINAL_LB_BITS: u64 = 0x40c4_0000_0000_0000;

    /// A terminal boundary cut carrying a known coefficient `BETA` on the
    /// terminal lag-1 bucket slot prices theta by exactly `BETA * x`
    /// (closed-form), mirroring `right_boundary_pricing.rs`'s thermal
    /// `beta * x` pin on the bucket family.
    ///
    /// The probe pins lag **2**'s incoming column, not lag 1's: the ring's
    /// shift-row definition is `b_d^out = b_{d+1}^in + k_d * D` (never an
    /// identity), so lag 1's outgoing value — what the cut and theta actually
    /// price — is controlled by lag 2's incoming, one hop deeper. The arc's
    /// own release `D` adds a constant, cost-optimal offset independent of
    /// this pin, so it cancels in the delta between the two solves below.
    #[test]
    fn boundary_cut_prices_the_terminal_bucket_state_by_beta_times_x() {
        let mut setup = fresh_setup_with(case_dir(), |cfg| {
            cfg.policy.boundary = Some(boundary_policy());
            cfg.modeling.cost_scale_factor = Some(1.0);
        });
        let bucket_col = bucket_state_index(&setup, 1);
        let pin_col = bucket_state_index(&setup, 2);

        let tmp = tempfile::tempdir().expect("tempdir");
        inject_bucket_boundary(&mut setup, &tmp.path().join("boundary"), bucket_col);

        let terminal_pool_id = setup
            .inputs
            .node_graph
            .terminal_pool(setup.num_stages())
            .unwrap();
        let template = freeze_terminal_template(&setup, terminal_pool_id);
        let pool = &setup.fcf.pools[terminal_pool_id];
        let terminal_node = NodeId(i32::try_from(setup.num_stages() - 1).expect("fits i32"));

        let x0 = 0.0;
        let x1 = 2.0;
        let (theta0, outgoing0) = terminal_theta(
            &setup,
            &template,
            pool,
            terminal_node,
            &bucket_pin(&setup, pin_col, x0),
            bucket_col,
        );
        let (theta1, outgoing1) = terminal_theta(
            &setup,
            &template,
            pool,
            terminal_node,
            &bucket_pin(&setup, pin_col, x1),
            bucket_col,
        );

        assert!(
            theta0 > 0.0 && theta1 > 0.0,
            "both solves must land on the injected cut's binding regime \
             (theta0={theta0}, theta1={theta1})"
        );
        assert!(
            (outgoing1 - outgoing0 - (x1 - x0)).abs() < TOL,
            "the deeper pin must move the priced outgoing lag 1:1 through the \
             shift-row identity (b_1^out = b_2^in + k_1*D, D cost-optimal and \
             pin-independent): outgoing0={outgoing0}, outgoing1={outgoing1}"
        );

        // Closed form: theta = alpha + beta * outgoing, using the SOLVED
        // outgoing value (never the pin) — exact regardless of the arc's own
        // cost-optimal release D, which the pin does not determine.
        assert!(
            (theta0 - (ALPHA + BETA * outgoing0)).abs() < TOL
                && (theta1 - (ALPHA + BETA * outgoing1)).abs() < TOL,
            "theta must equal alpha + beta*outgoing: theta0={theta0} \
             (expected {}), theta1={theta1} (expected {})",
            ALPHA + BETA * outgoing0,
            ALPHA + BETA * outgoing1
        );

        let expected_delta = BETA * (outgoing1 - outgoing0);
        let actual_delta = theta1 - theta0;
        assert!(
            (actual_delta - expected_delta).abs() < TOL,
            "theta must respond to the in-transit volume by beta*x: expected \
             delta {expected_delta}, got {actual_delta}"
        );
    }

    /// The backward pass propagates the boundary coefficient inward —
    /// after one training iteration, pool 0's (the earliest stage's)
    /// generated cut carries a nonzero coefficient on the terminal bucket
    /// state, reached through the shift-row chain connecting every stage's
    /// outgoing column to the next stage's incoming column at the same
    /// global index.
    #[test]
    fn backward_pass_propagates_boundary_coefficient_to_pool_zero() {
        let mut setup = fresh_setup_with(case_dir(), |cfg| {
            cfg.policy.boundary = Some(boundary_policy());
        });
        let bucket_col = bucket_state_index(&setup, 1);

        let tmp = tempfile::tempdir().expect("tempdir");
        inject_bucket_boundary(&mut setup, &tmp.path().join("boundary"), bucket_col);

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error must be None; got {:?}",
            outcome.error
        );

        let (_slot, _intercept, coefficients) = setup
            .fcf
            .active_cuts(0)
            .next()
            .expect("pool 0 must carry a generated cut after training");
        assert!(
            coefficients[bucket_col].abs() > TOL,
            "pool 0's cut must carry a nonzero coefficient on the terminal bucket \
             state; got {}",
            coefficients[bucket_col]
        );
    }
}

/// Enumerated forward selection makes the exact risk-adjusted upper bound
/// computable, so a `gap` stopping rule is admissible under a uniform `CVaR` —
/// the DECOMP-parity capability. These integration tests drive the whole
/// pipeline (`StudySetup::new` admission gate → enumerated forward UB → gap).
mod enumerated_cvar_gap {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::HydroGenerationModel;
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, DeficitSegment,
        EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds, HydroStorage,
        InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, System, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod, ModelingConfig, PolicyConfig, RowSelectionConfig,
        SimulationConfig as IoSimulationConfig, StoppingMode, StoppingRuleConfig, TrainingConfig,
        TrainingSelection, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_solver::ActiveSolver;

    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage,
    };
    use super::common::{StubComm, build_setup_in_code, try_build_setup_in_code};

    const BUS_ID: i32 = 1;
    const HYDRO_ID: i32 = 1;
    const N_STAGES: usize = 2;
    const CVAR: StageRiskConfig = StageRiskConfig::CVaR {
        alpha: 0.15,
        lambda: 0.4,
    };

    fn zero_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 0.0,
        }
    }

    fn stages() -> Vec<Stage> {
        let start = NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date");
        (0..N_STAGES)
            .map(|i| {
                let s = start + chrono::Duration::days(30 * i64::try_from(i).unwrap_or(0));
                make_stage(
                    i,
                    StageSpec {
                        start_date: s,
                        end_date: s + chrono::Duration::days(30),
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: 730.0,
                        }],
                        block_mode: BlockMode::Parallel,
                        state_config: StageStateConfig {
                            storage: true,
                            inflow_lags: false,
                        },
                        risk_config: StageRiskConfig::Expectation,
                        scenario_config: ScenarioSourceConfig {
                            branching_factor: 1,
                            noise_method: NoiseMethod::Saa,
                        },
                        ..Default::default()
                    },
                )
            })
            .collect()
    }

    /// One bus (deterministic 100 MW load served entirely by deficit — the
    /// default zero productivity leaves the hydro non-generating) and one
    /// reservoir hydro, with `risk_config` applied per stage.
    fn build_system(risk_config: &[StageRiskConfig]) -> System {
        assert_eq!(risk_config.len(), N_STAGES);
        let bus = make_bus(
            EntityId(BUS_ID),
            BusSpec {
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );
        let hydro = make_hydro(
            EntityId(HYDRO_ID),
            HydroSpec {
                bus_id: EntityId(BUS_ID),
                max_storage_hm3: 200.0,
                max_turbined_m3s: 100.0,
                max_generation_mw: 100.0,
                generation_model: HydroGenerationModel::ConstantProductivity,
                ..Default::default()
            },
        );

        let mut stages = stages();
        for (stage, rc) in stages.iter_mut().zip(risk_config) {
            stage.risk_config = *rc;
        }

        let inflow_models: Vec<InflowModel> = (0..N_STAGES)
            .map(|i| InflowModel {
                hydro_id: EntityId(HYDRO_ID),
                stage_id: i32::try_from(i).unwrap_or(0),
                mean_m3s: 50.0,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();
        let load_models: Vec<LoadModel> = (0..N_STAGES)
            .map(|i| LoadModel {
                bus_id: EntityId(BUS_ID),
                stage_id: i32::try_from(i).unwrap_or(0),
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 100.0,
                    ..Default::default()
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: zero_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(InitialConditions {
                storage: vec![HydroStorage {
                    hydro_id: EntityId(HYDRO_ID),
                    value_hm3: 100.0,
                }],
                ..InitialConditions::default()
            })
            .build()
            .expect("enumerated_cvar_gap: valid single-hydro study")
    }

    fn enumerated_gap_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: InflowNonNegativityMethod::Penalty,
                },
                cost_scale_factor: Some(1.0),
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![
                    StoppingRuleConfig::Gap {
                        tolerance: Some(1.0e9),
                        relative_tolerance: None,
                    },
                    StoppingRuleConfig::IterationLimit { limit: 20 },
                ]),
                stopping_mode: StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Enumerated {}),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig {
                enabled: false,
                ..IoSimulationConfig::default()
            },
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    /// A gap rule under enumerated forwards + a UNIFORM CVaR builds (the gate now
    /// admits it), trains without error, and reports a NON-NEGATIVE gap — the
    /// risk-adjusted upper bound brackets the risk-adjusted lower bound. Before
    /// the change `StudySetup::new` rejected this pairing outright.
    #[test]
    fn uniform_cvar_gap_admits_trains_and_brackets() {
        let system = build_system(&[CVAR, CVAR]);
        let config = enumerated_gap_config();

        assert!(
            try_build_setup_in_code(build_system(&[CVAR, CVAR]), &config).is_ok(),
            "enumerated + uniform CVaR + gap rule must be admitted by StudySetup::new"
        );

        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must return Ok");
        assert!(
            outcome.error.is_none(),
            "training must not error: {:?}",
            outcome.error
        );

        let r = &outcome.result;
        assert!(r.final_ub.is_finite(), "final_ub must be finite");
        assert!(
            r.final_ub >= r.final_lb - 1e-6,
            "the risk-adjusted UB ({}) must bracket the LB ({}); a risk-neutral UB \
             under a CVaR LB is the negative-gap defect this change fixes",
            r.final_ub,
            r.final_lb
        );
        assert!(
            r.final_gap >= -1e-6,
            "reported gap must be non-negative, got {}",
            r.final_gap
        );
    }

    /// A gap rule under enumerated forwards but a NON-UNIFORM measure (stage 0
    /// expectation, stage 1 CVaR) stays rejected: the static path-cost bound has
    /// no single measure to apply.
    #[test]
    fn nonuniform_risk_gap_is_rejected() {
        let config = enumerated_gap_config();
        let result =
            try_build_setup_in_code(build_system(&[StageRiskConfig::Expectation, CVAR]), &config);
        match result {
            Err(cobre_sddp::SddpError::Validation(msg)) => {
                assert!(
                    msg.contains("uniform") && msg.contains("gap"),
                    "reject message must name the uniformity requirement: {msg}"
                );
            }
            other => panic!("expected a uniformity Validation reject, got {other:?}"),
        }
    }
}

/// JSON fixture text shared verbatim by
/// [`security_curve_integrated_productivity_equivalence`] and
/// [`stored_energy_columns_determinism`]: one bus, default penalties, and a
/// 2-stage/730h-block calendar with hydro 0 seeded at 500 hm3 and hydro 1 at
/// 100 hm3.
mod stored_energy_fixture_json {
    pub(super) const PENALTIES_JSON: &str = r#"{
  "bus": { "deficit_segments": [ { "depth_mw": null, "cost": 1000.0 } ], "excess_cost": 0.01 },
  "line": { "exchange_cost": 0.01 },
  "hydro": {
    "spillage_cost": 0.01, "turbined_cost": 0.01, "diversion_cost": 0.01,
    "storage_violation_below_cost": 10000.0, "filling_target_violation_cost": 10000.0,
    "turbined_violation_below_cost": 10000.0, "outflow_violation_below_cost": 10000.0,
    "outflow_violation_above_cost": 10000.0, "generation_violation_below_cost": 10000.0,
    "evaporation_violation_cost": 10000.0, "water_withdrawal_violation_cost": 10000.0
  },
  "non_controllable_source": { "curtailment_cost": 0.005 }
}"#;

    pub(super) const STAGES_JSON: &str = r#"{
  "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0 },
  "stages": [
    { "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
      "blocks": [ { "id": 0, "name": "SINGLE", "hours": 730 } ], "num_openings": 1 },
    { "id": 1, "start_date": "2024-02-01", "end_date": "2024-03-01",
      "blocks": [ { "id": 0, "name": "SINGLE", "hours": 730 } ], "num_openings": 1 }
  ]
}"#;

    pub(super) const INITIAL_CONDITIONS_JSON: &str = r#"{
  "storage": [
    { "hydro_id": 0, "value_hm3": 500.0 },
    { "hydro_id": 1, "value_hm3": 100.0 }
  ],
  "filling_storage": []
}"#;

    pub(super) const BUSES_JSON: &str = r#"{
  "buses": [
    { "id": 0, "name": "B0", "operational_start_date": "2020-01-01",
      "deficit_segments": [ { "depth_mw": null, "cost": 1000.0 } ] }
  ]
}"#;

    pub(super) const LINES_JSON: &str = r#"{ "lines": [] }"#;
}

/// In-code deck with real VHA geometry, an operative `hydro_bounds` ceiling
/// below the entity's physical range, and both productivity override columns.
mod security_curve_integrated_productivity_equivalence {
    use std::path::Path;
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    use cobre_sddp::{
        build_generic_constraint_echo_rows, hydro_models::prepare_hydro_models,
        setup::prepare_stochastic,
    };

    use crate::common::build_setup_for_case;
    use crate::common::parquet_fixtures::{write_hydro_geometry, write_seasonal_stats};

    use super::stored_energy_fixture_json::{
        BUSES_JSON, INITIAL_CONDITIONS_JSON, LINES_JSON, PENALTIES_JSON, STAGES_JSON,
    };

    /// `(hydro_id, stage_id, equivalent_productivity_mw_per_m3s override,
    /// specific_productivity_mw_per_m3s_per_m override)` — `reference_outflow_m3s`
    /// stays `NULL` for every row (Q_ref override is out of this fixture's scope).
    fn write_hydro_energy_productivity(dest: &Path, rows: &[(i32, i32, Option<f64>, Option<f64>)]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, true),
            Field::new(
                "equivalent_productivity_mw_per_m3s",
                DataType::Float64,
                true,
            ),
            Field::new("reference_outflow_m3s", DataType::Float64, true),
            Field::new(
                "specific_productivity_mw_per_m3s_per_m",
                DataType::Float64,
                true,
            ),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(
                    rows.iter().map(|r| r.0).collect::<Vec<_>>(),
                )),
                Arc::new(Int32Array::from(
                    rows.iter().map(|r| Some(r.1)).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    rows.iter().map(|r| r.2).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(vec![None::<f64>; rows.len()])),
                Arc::new(Float64Array::from(
                    rows.iter().map(|r| r.3).collect::<Vec<_>>(),
                )),
            ],
        )
        .expect("valid RecordBatch for hydro_energy_productivity");
        let file = std::fs::File::create(dest).expect("create hydro_energy_productivity.parquet");
        let mut writer = ArrowWriter::try_new(file, schema, None)
            .expect("ArrowWriter for productivity override");
        writer.write(&batch).expect("write productivity batch");
        writer.close().expect("close productivity writer");
    }

    fn write_hydro_bounds_max_storage(
        dest: &Path,
        hydro_id: i32,
        stage_id: i32,
        max_storage_hm3: f64,
    ) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("max_storage_hm3", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![hydro_id])),
                Arc::new(Int32Array::from(vec![stage_id])),
                Arc::new(Float64Array::from(vec![max_storage_hm3])),
            ],
        )
        .expect("valid RecordBatch for hydro_bounds");
        let file = std::fs::File::create(dest).expect("create hydro_bounds.parquet");
        let mut writer =
            ArrowWriter::try_new(file, schema, None).expect("ArrowWriter for hydro bounds");
        writer.write(&batch).expect("write hydro bounds batch");
        writer.close().expect("close hydro bounds writer");
    }

    /// `(constraint_id, stage_id, bound_lower)` — `block_id` and `bound_upper`
    /// stay `NULL` for every row; every constraint in this fixture is stage-level
    /// and lower-bounded (or driven entirely by its own inline affine remainder).
    fn write_generic_constraint_bounds(dest: &Path, rows: &[(i32, i32, Option<f64>)]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("constraint_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("block_id", DataType::Int32, true),
            Field::new("bound_lower", DataType::Float64, true),
            Field::new("bound_upper", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(
                    rows.iter().map(|r| r.0).collect::<Vec<_>>(),
                )),
                Arc::new(Int32Array::from(
                    rows.iter().map(|r| r.1).collect::<Vec<_>>(),
                )),
                Arc::new(Int32Array::new_null(rows.len())),
                Arc::new(Float64Array::from(
                    rows.iter().map(|r| r.2).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::new_null(rows.len())),
            ],
        )
        .expect("valid RecordBatch for generic_constraint_bounds");
        let file = std::fs::File::create(dest).expect("create generic_constraint_bounds.parquet");
        let mut writer =
            ArrowWriter::try_new(file, schema, None).expect("ArrowWriter for constraint bounds");
        writer.write(&batch).expect("write constraint bounds batch");
        writer.close().expect("close constraint bounds writer");
    }

    const CONFIG_JSON: &str = r#"{
  "training": { "selection": { "method": "sampled", "forward_passes": 1 },
    "stopping_rules": [ { "type": "iteration_limit", "limit": 1 } ] },
  "simulation": { "enabled": false },
  "modeling": { "inflow_non_negativity": { "method": "none" } }
}"#;

    const THERMALS_JSON: &str = r#"{
  "thermals": [
    { "id": 0, "name": "T0", "operational_start_date": "2020-01-01", "bus_id": 0,
      "generation": { "min_mw": 0.0, "max_mw": 100.0 }, "cost_per_mwh": 50.0 }
  ]
}"#;

    /// Hydro 0 carries the exactly-representable VHA range `[256, 1280]` (a
    /// power-of-two multiple of 256, per the exact-geometry hand-oracle
    /// convention) and the entity `ρ_esp = 0.125`; hydro 1 is a bare
    /// non-FPHA plant whose `ρ_eq` is only ever supplied by the
    /// `hydro_energy_productivity.parquet` override at stage 1.
    const HYDROS_JSON: &str = r#"{
  "hydros": [
    { "id": 0, "name": "H0", "operational_start_date": "2020-01-01", "downstream_id": null,
      "reservoir": { "min_storage_hm3": 256.0, "max_storage_hm3": 1280.0 },
      "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": 50.0 },
      "generation": { "model": "constant_productivity", "min_turbined_m3s": 0.0,
        "max_turbined_m3s": 50.0, "min_generation_mw": 0.0, "max_generation_mw": 50.0 },
      "unit_groups": [ { "id": 0, "name": "H0", "bus_id": 0, "min_generation_mw": 0.0,
        "max_generation_mw": 50.0, "min_turbined_m3s": 0.0, "max_turbined_m3s": 50.0 } ],
      "specific_productivity_mw_per_m3s_per_m": 0.125 },
    { "id": 1, "name": "H1", "operational_start_date": "2020-01-01", "downstream_id": null,
      "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 200.0 },
      "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": 50.0 },
      "generation": { "model": "constant_productivity", "min_turbined_m3s": 0.0,
        "max_turbined_m3s": 50.0, "min_generation_mw": 0.0, "max_generation_mw": 50.0 },
      "unit_groups": [ { "id": 0, "name": "H1", "bus_id": 0, "min_generation_mw": 0.0,
        "max_generation_mw": 50.0, "min_turbined_m3s": 0.0, "max_turbined_m3s": 50.0 } ] }
  ]
}"#;

    /// Hydro 1's `stage_ranges` covers ONLY stage 0, leaving stage 1 with no JSON
    /// productivity source — the `equivalent_productivity_mw_per_m3s` override at
    /// stage 1 is then the sole supplier there (the Layer-6 exactly-one-source rule).
    const PRODUCTION_MODELS_JSON: &str = r#"{
  "production_models": [
    { "hydro_id": 0, "selection_mode": "stage_ranges", "stage_ranges": [
      { "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity",
        "productivity_mw_per_m3s": 1.0 } ] },
    { "hydro_id": 1, "selection_mode": "stage_ranges", "stage_ranges": [
      { "start_stage_id": 0, "end_stage_id": 0, "model": "constant_productivity",
        "productivity_mw_per_m3s": 2.0 } ] }
  ]
}"#;

    const GENERIC_PARAMETERS_JSON: &str = r#"{
  "scalar_parameters": [
    { "id": 1, "name": "int_rho", "kind": "computed",
      "computed_spec": { "tag": "integrated_accumulated_productivity", "hydro_id": 0 } },
    { "id": 2, "name": "max_energy", "kind": "computed",
      "computed_spec": { "tag": "max_stored_energy", "hydro_id": 0 } },
    { "id": 3, "name": "rho_point", "kind": "computed",
      "computed_spec": { "tag": "accumulated_productivity", "hydro_id": 0 } },
    { "id": 4, "name": "h1_ref", "kind": "computed",
      "computed_spec": { "tag": "equivalent_productivity", "hydro_id": 1 } },
    { "id": 5, "name": "h1_int", "kind": "computed",
      "computed_spec": { "tag": "integrated_equivalent_productivity", "hydro_id": 1 } }
  ]
}"#;

    /// Builds the fixture deck in a fresh `TempDir`. `int_rho_stage1` is baked in
    /// as the "before" form's literal coefficient (`hydro_storage_final`, no
    /// `@name` reference — the hand-folded form the bridge injects today);
    /// `bound_stage1`/`max_storage_ceiling` similarly drive the "before" form's
    /// literal RHS and the operative storage ceiling below the entity max.
    fn build_fixture(int_rho_stage1: f64, bound_stage1: f64) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let root = tmp.path();
        std::fs::create_dir_all(root.join("system")).expect("create system dir");
        std::fs::create_dir_all(root.join("constraints")).expect("create constraints dir");
        std::fs::create_dir_all(root.join("scenarios")).expect("create scenarios dir");

        std::fs::write(root.join("config.json"), CONFIG_JSON).expect("write config.json");
        std::fs::write(root.join("penalties.json"), PENALTIES_JSON).expect("write penalties.json");
        std::fs::write(root.join("stages.json"), STAGES_JSON).expect("write stages.json");
        std::fs::write(
            root.join("initial_conditions.json"),
            INITIAL_CONDITIONS_JSON,
        )
        .expect("write initial_conditions.json");
        std::fs::write(root.join("system/buses.json"), BUSES_JSON).expect("write buses.json");
        std::fs::write(root.join("system/lines.json"), LINES_JSON).expect("write lines.json");
        std::fs::write(root.join("system/thermals.json"), THERMALS_JSON)
            .expect("write thermals.json");
        std::fs::write(root.join("system/hydros.json"), HYDROS_JSON).expect("write hydros.json");
        std::fs::write(
            root.join("system/hydro_production_models.json"),
            PRODUCTION_MODELS_JSON,
        )
        .expect("write hydro_production_models.json");
        std::fs::write(
            root.join("constraints/generic_parameters.json"),
            GENERIC_PARAMETERS_JSON,
        )
        .expect("write generic_parameters.json");

        let constraints_json = format!(
            r#"{{
  "constraints": [
    {{ "id": 0, "name": "after_form", "slack": {{ "enabled": false }},
       "expression": "@int_rho * hydro_useful_volume_final(0) >= 0.5 * @max_energy" }},
    {{ "id": 1, "name": "before_form", "slack": {{ "enabled": false }},
       "expression": "{int_rho_stage1} * hydro_storage_final(0)" }},
    {{ "id": 2, "name": "hydro0_reference_point_probe", "slack": {{ "enabled": false }},
       "expression": "@rho_point * hydro_storage(1)" }},
    {{ "id": 3, "name": "hydro1_reference_point_override_probe", "slack": {{ "enabled": false }},
       "expression": "@h1_ref * hydro_storage(1)" }},
    {{ "id": 4, "name": "hydro1_integrated_override_probe", "slack": {{ "enabled": false }},
       "expression": "@h1_int * hydro_storage(1)" }},
    {{ "id": 5, "name": "hydro0_max_stored_energy_probe", "slack": {{ "enabled": false }},
       "expression": "@max_energy * hydro_storage(1)" }}
  ]
}}"#
        );
        std::fs::write(
            root.join("constraints/generic_constraints.json"),
            constraints_json,
        )
        .expect("write generic_constraints.json");

        write_hydro_geometry(
            &root.join("system/hydro_geometry.parquet"),
            &[
                (0, 256.0, 300.0, 1.0),
                (0, 768.0, 340.0, 1.0),
                (0, 1280.0, 348.0, 1.0),
            ],
        );
        write_hydro_energy_productivity(
            &root.join("system/hydro_energy_productivity.parquet"),
            &[
                // Hydro 0: ρ_esp override at stage 1 only; stage 0 uses the entity value.
                (0, 1, None, Some(0.375)),
                // Hydro 1: ρ_eq override at stage 1, the sole productivity source there.
                (1, 1, Some(0.75), None),
            ],
        );
        write_hydro_bounds_max_storage(&root.join("constraints/hydro_bounds.parquet"), 0, 0, 700.0);
        write_generic_constraint_bounds(
            &root.join("constraints/generic_constraint_bounds.parquet"),
            &[
                (0, 0, None),
                (0, 1, None),
                (1, 1, Some(bound_stage1)),
                (2, 0, Some(0.0)),
                (3, 1, Some(0.0)),
                (4, 1, Some(0.0)),
                (5, 0, Some(0.0)),
            ],
        );

        write_seasonal_stats(
            &root.join("scenarios/inflow_seasonal_stats.parquet"),
            "hydro_id",
            "mean_m3s",
            "std_m3s",
            &[
                (0, 0, 40.0, 0.0),
                (0, 1, 10.0, 0.0),
                (1, 0, 0.0, 0.0),
                (1, 1, 0.0, 0.0),
            ],
        );
        write_seasonal_stats(
            &root.join("scenarios/load_seasonal_stats.parquet"),
            "bus_id",
            "mean_mw",
            "std_mw",
            &[(0, 0, 80.0, 0.0), (0, 1, 80.0, 0.0)],
        );

        tmp
    }

    /// The security curve `useful >= pct * useful_max` authored both as
    /// primitives (`hydro_useful_volume_final`, coefficient `@int_rho`, bound
    /// `pct * @max_energy`) and hand-folded (a literal `int_rho` on
    /// `hydro_storage_final`, bound `pct*int_rho*(V_hi-V_lo) + int_rho*V_lo`)
    /// resolve to the same coefficient and the same folded bound, at both an
    /// unoverridden and a ρ_esp-overridden stage, unaffected by an operative
    /// `hydro_bounds` ceiling below the entity range, and distinct from the
    /// plant's reference-point productivity; a non-FPHA `ρ_eq` override
    /// separately wins both evaluators for its own (hydro, stage).
    #[test]
    fn after_and_before_forms_match_the_hand_oracle() {
        // VHA breakpoints spanning hydro 0's entity range exactly (no fractional
        // interpolation in `ForebayTable::mean_height`): volumes at multiples of
        // 256, heights non-uniform so the mean genuinely integrates the curve.
        const V_LO: f64 = 256.0;
        const V_MID: f64 = 768.0;
        const V_HI: f64 = 1280.0;
        const H_LO: f64 = 300.0;
        const H_MID: f64 = 340.0;
        const H_HI: f64 = 348.0;
        const ENTITY_RHO_ESP: f64 = 0.125;
        const OVERRIDE_RHO_ESP: f64 = 0.375;
        const PCT: f64 = 0.5;
        const HYDRO1_OVERRIDE_RHO_EQ: f64 = 0.75;

        let mean_height = (0.5 * (H_LO + H_MID) * (V_MID - V_LO)
            + 0.5 * (H_MID + H_HI) * (V_HI - V_MID))
            / (V_HI - V_LO);

        let int_rho_stage0 = ENTITY_RHO_ESP * mean_height;
        let int_rho_stage1 = OVERRIDE_RHO_ESP * mean_height;
        let max_energy_stage0 = int_rho_stage0 * (V_HI - V_LO);
        let max_energy_stage1 = int_rho_stage1 * (V_HI - V_LO);
        let bound_stage0 = PCT * max_energy_stage0 + int_rho_stage0 * V_LO;
        let bound_stage1 = PCT * max_energy_stage1 + int_rho_stage1 * V_LO;

        let tmp = build_fixture(int_rho_stage1, bound_stage1);
        let case_dir = tmp.path();

        let config_path = case_dir.join("config.json");
        let config = cobre_io::parse_config(&config_path).expect("config must parse");
        let system = cobre_io::load_case(case_dir).expect("load_case must succeed");
        let training_source = config
            .training_scenario_source(&config_path)
            .expect("training_scenario_source must parse");
        let prepare_result =
            prepare_stochastic(system, case_dir, &config, 42, &training_source, None)
                .expect("prepare_stochastic must succeed");
        let system = prepare_result.system;
        let stochastic = prepare_result.stochastic;
        let hydro_models = prepare_hydro_models(&system, case_dir, false)
            .expect("prepare_hydro_models must succeed");
        let setup = build_setup_for_case(case_dir, &config, &system, stochastic, hydro_models);

        let echo_rows = build_generic_constraint_echo_rows(&setup, &system);
        let find = |constraint_id: i32, stage_id: i32| {
            echo_rows
                .iter()
                .find(|r| r.constraint_id == constraint_id && r.stage_id == stage_id)
                .unwrap_or_else(|| {
                    panic!("no echo row for constraint {constraint_id} at stage {stage_id}")
                })
        };
        let close =
            |value: f64, expected: f64| (value - expected).abs() <= 1e-9 * expected.abs().max(1.0);

        // after == before == hand oracle at the overridden stage, to tolerance —
        // never `==` on a trapezoid-derived value.
        let after1 = find(0, 1);
        let before1 = find(1, 1);
        let after1_coef = after1.coefficient.expect("after1 coefficient");
        let after1_bound = after1.bound_lower.expect("after1 bound_lower");
        let before1_coef = before1.coefficient.expect("before1 coefficient");
        let before1_bound = before1.bound_lower.expect("before1 bound_lower");
        assert!(
            close(after1_coef, int_rho_stage1),
            "after-form coefficient {after1_coef} != hand oracle {int_rho_stage1}"
        );
        assert!(
            close(after1_bound, bound_stage1),
            "after-form bound_lower {after1_bound} != hand oracle {bound_stage1}"
        );
        assert!(
            close(before1_coef, int_rho_stage1),
            "before-form coefficient {before1_coef} != hand oracle {int_rho_stage1}"
        );
        assert!(
            close(before1_bound, bound_stage1),
            "before-form bound_lower {before1_bound} != hand oracle {bound_stage1}"
        );
        assert!(
            close(after1_coef, before1_coef),
            "after coefficient {after1_coef} != before coefficient {before1_coef}"
        );
        assert!(
            close(after1_bound, before1_bound),
            "after bound_lower {after1_bound} != before bound_lower {before1_bound}"
        );
        // `HydroStorageFinal` and `HydroUsefulVolumeFinal` both resolve through
        // `resolve_hydro_storage_boundary` to the same fixed boundary column
        // (crates/cobre-sddp/src/lp/generic_constraints.rs); the coefficient and
        // bound_lower agreement above is that shared-column fact's only
        // observable surface at this build-time, no-solve echo layer.
        assert_eq!(
            after1.variable_kind.as_deref(),
            Some("hydro_useful_volume_final")
        );
        assert_eq!(
            before1.variable_kind.as_deref(),
            Some("hydro_storage_final")
        );

        // The after-form at stage 0 — under an operative `hydro_bounds` ceiling
        // (700) below the entity max (1280) — still reflects the entity range,
        // and so does `max_stored_energy` itself.
        let after0 = find(0, 0);
        let after0_coef = after0.coefficient.expect("after0 coefficient");
        let after0_bound = after0.bound_lower.expect("after0 bound_lower");
        assert!(
            close(after0_bound, bound_stage0),
            "hydro_bounds ceiling leaked into the folded bound: {after0_bound} != {bound_stage0}"
        );
        let max_energy0 = find(5, 0)
            .coefficient
            .expect("max_energy probe coefficient");
        assert!(
            close(max_energy0, max_energy_stage0),
            "hydro_bounds ceiling leaked into max_stored_energy: {max_energy0} != {max_energy_stage0}"
        );

        // The integrated cascade productivity genuinely differs from the
        // plant's reference-point `accumulated_productivity`.
        let rho_point0 = find(2, 0).coefficient.expect("rho_point coefficient");
        assert!(
            (after0_coef - rho_point0).abs() > 1.0,
            "integrated ({after0_coef}) must differ from reference-point ({rho_point0})"
        );

        // The ρ_esp override shifts int_rho only at its own stage;
        // exactly-representable geometry makes `to_bits()` provable, not
        // coincidental.
        assert_eq!(after0_coef.to_bits(), int_rho_stage0.to_bits());
        assert_eq!(after1_coef.to_bits(), int_rho_stage1.to_bits());

        // A non-FPHA ρ_eq override wins both evaluators (the resolved
        // reference-point and integrated own terms) for its own (hydro, stage).
        let h1_ref = find(3, 1).coefficient.expect("h1_ref coefficient");
        let h1_int = find(4, 1).coefficient.expect("h1_int coefficient");
        assert_eq!(h1_ref.to_bits(), HYDRO1_OVERRIDE_RHO_EQ.to_bits());
        assert_eq!(h1_int.to_bits(), HYDRO1_OVERRIDE_RHO_EQ.to_bits());
    }
}

/// The six simulation-output hydro columns outside the parity hash's field
/// whitelist (`tests/common/parity_hash.rs`'s module doc) — the four
/// integrated-grid/stored-energy-power columns and the moved
/// `stored_energy_{initial,final}_mwh` pair — must still be declaration-order
/// invariant and run-to-run reproducible, exactly like every hashed column.
/// The fixture's VHA-geometry hydro gives the gate power: its integrated
/// productivity genuinely differs from its reference-point productivity, so a
/// permutation that left the two coincidentally equal could not hide a bug.
mod stored_energy_columns_determinism {
    use std::path::Path;

    use cobre_core::scenario::ScenarioSource;
    use cobre_io::config::SimulationSelection;
    use cobre_sddp::hydro_models::prepare_hydro_models;
    use cobre_sddp::setup::prepare_stochastic;
    use cobre_sddp::{SimulationHydroResult, SimulationScenarioResult};

    use crate::common::parquet_fixtures::{write_hydro_geometry, write_seasonal_stats};
    use crate::common::permute::permute_case;
    use crate::common::{build_setup_for_case, run_simulation};

    use super::stored_energy_fixture_json::{
        BUSES_JSON, INITIAL_CONDITIONS_JSON, LINES_JSON, PENALTIES_JSON, STAGES_JSON,
    };

    /// Fixed seed for the declaration-order-invariance probe, matching the
    /// sibling `nonzero_stage_fpha_override_regression` idiom.
    const PERMUTATION_SEED: u64 = 20_260_922;

    const CONFIG_JSON: &str = r#"{
  "training": { "selection": { "method": "sampled", "forward_passes": 1 },
    "stopping_rules": [ { "type": "iteration_limit", "limit": 10 } ] },
  "simulation": { "enabled": false },
  "modeling": { "inflow_non_negativity": { "method": "none" } }
}"#;

    const THERMALS_JSON: &str = r#"{
  "thermals": [
    { "id": 0, "name": "T0", "operational_start_date": "2020-01-01", "bus_id": 0,
      "generation": { "min_mw": 0.0, "max_mw": 200.0 }, "cost_per_mwh": 50.0 }
  ]
}"#;

    /// Hydro 0 carries the exactly-representable VHA range `[256, 1280]` and the
    /// entity `ρ_esp = 0.125`, so its integrated equivalent productivity
    /// (`ρ_esp * mean_height`) differs from its `ConstantProductivity`
    /// reference-point value (`1.0`) — the fixture's "gate has power" property.
    /// Hydro 1 carries no geometry (its two grids coincide); its only role is
    /// giving the declaration-order permutation a second entity to reorder.
    const HYDROS_JSON: &str = r#"{
  "hydros": [
    { "id": 0, "name": "H0", "operational_start_date": "2020-01-01", "downstream_id": null,
      "reservoir": { "min_storage_hm3": 256.0, "max_storage_hm3": 1280.0 },
      "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": 100.0 },
      "generation": { "model": "constant_productivity", "min_turbined_m3s": 0.0,
        "max_turbined_m3s": 100.0, "min_generation_mw": 0.0, "max_generation_mw": 100.0 },
      "unit_groups": [ { "id": 0, "name": "H0", "bus_id": 0, "min_generation_mw": 0.0,
        "max_generation_mw": 100.0, "min_turbined_m3s": 0.0, "max_turbined_m3s": 100.0 } ],
      "specific_productivity_mw_per_m3s_per_m": 0.125 },
    { "id": 1, "name": "H1", "operational_start_date": "2020-01-01", "downstream_id": null,
      "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 500.0 },
      "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": 80.0 },
      "generation": { "model": "constant_productivity", "min_turbined_m3s": 0.0,
        "max_turbined_m3s": 80.0, "min_generation_mw": 0.0, "max_generation_mw": 160.0 },
      "unit_groups": [ { "id": 0, "name": "H1", "bus_id": 0, "min_generation_mw": 0.0,
        "max_generation_mw": 160.0, "min_turbined_m3s": 0.0, "max_turbined_m3s": 80.0 } ] }
  ]
}"#;

    const PRODUCTION_MODELS_JSON: &str = r#"{
  "production_models": [
    { "hydro_id": 0, "selection_mode": "stage_ranges", "stage_ranges": [
      { "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity",
        "productivity_mw_per_m3s": 1.0 } ] },
    { "hydro_id": 1, "selection_mode": "stage_ranges", "stage_ranges": [
      { "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity",
        "productivity_mw_per_m3s": 2.0 } ] }
  ]
}"#;

    /// Builds the fixture deck in a fresh `TempDir`: two `ConstantProductivity`
    /// hydros on one bus, one carrying VHA geometry so its integrated grid
    /// genuinely differs from its reference-point grid.
    fn build_fixture() -> tempfile::TempDir {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let root = tmp.path();
        std::fs::create_dir_all(root.join("system")).expect("create system dir");
        std::fs::create_dir_all(root.join("scenarios")).expect("create scenarios dir");

        std::fs::write(root.join("config.json"), CONFIG_JSON).expect("write config.json");
        std::fs::write(root.join("penalties.json"), PENALTIES_JSON).expect("write penalties.json");
        std::fs::write(root.join("stages.json"), STAGES_JSON).expect("write stages.json");
        std::fs::write(
            root.join("initial_conditions.json"),
            INITIAL_CONDITIONS_JSON,
        )
        .expect("write initial_conditions.json");
        std::fs::write(root.join("system/buses.json"), BUSES_JSON).expect("write buses.json");
        std::fs::write(root.join("system/lines.json"), LINES_JSON).expect("write lines.json");
        std::fs::write(root.join("system/thermals.json"), THERMALS_JSON)
            .expect("write thermals.json");
        std::fs::write(root.join("system/hydros.json"), HYDROS_JSON).expect("write hydros.json");
        std::fs::write(
            root.join("system/hydro_production_models.json"),
            PRODUCTION_MODELS_JSON,
        )
        .expect("write hydro_production_models.json");

        write_hydro_geometry(
            &root.join("system/hydro_geometry.parquet"),
            &[
                (0, 256.0, 300.0, 1.0),
                (0, 768.0, 340.0, 1.0),
                (0, 1280.0, 348.0, 1.0),
            ],
        );

        write_seasonal_stats(
            &root.join("scenarios/inflow_seasonal_stats.parquet"),
            "hydro_id",
            "mean_m3s",
            "std_m3s",
            &[
                (0, 0, 30.0, 0.0),
                (0, 1, 20.0, 0.0),
                (1, 0, 15.0, 0.0),
                (1, 1, 10.0, 0.0),
            ],
        );
        write_seasonal_stats(
            &root.join("scenarios/load_seasonal_stats.parquet"),
            "bus_id",
            "mean_mw",
            "std_mw",
            &[(0, 0, 40.0, 0.0), (0, 1, 35.0, 0.0)],
        );

        tmp
    }

    /// Train + one-scenario-simulate `dir` and return the drained per-scenario
    /// results — the same `build_setup_for_case`/`run_simulation` harness the
    /// sibling permutation probe (`nonzero_stage_fpha_override_regression`) uses.
    fn train_and_simulate(dir: &Path) -> Vec<SimulationScenarioResult> {
        let config_path = dir.join("config.json");
        let config = cobre_io::parse_config(&config_path).expect("config must parse");
        let system = cobre_io::load_case(dir).expect("load_case must succeed");

        let pr = prepare_stochastic(system, dir, &config, 42, &ScenarioSource::default(), None)
            .expect("prepare_stochastic must succeed");
        let system = pr.system;
        let stochastic = pr.stochastic;

        let hydro_models =
            prepare_hydro_models(&system, dir, false).expect("prepare_hydro_models must succeed");

        let mut config_with_sim = config.clone();
        config_with_sim.simulation.enabled = true;
        config_with_sim.simulation.selection =
            Some(SimulationSelection::Sampled { num_scenarios: 1 });

        let mut setup =
            build_setup_for_case(dir, &config_with_sim, &system, stochastic, hydro_models);
        run_simulation(&mut setup, 1)
    }

    /// Flattens every scenario's per-stage hydro rows (the stage aggregate row,
    /// `block_id: None`, and each per-block row) into one canonically ordered
    /// list — the comparison unit the assertions below key on.
    fn sorted_hydro_rows(results: &[SimulationScenarioResult]) -> Vec<SimulationHydroResult> {
        let mut rows: Vec<SimulationHydroResult> = results
            .iter()
            .flat_map(|s| &s.stages)
            .flat_map(|stage| stage.hydros.iter().cloned())
            .collect();
        rows.sort_by_key(|h| (h.stage_id, h.block_id, h.hydro_id));
        rows
    }

    /// Asserts `to_bits()` equality of the six parity-hash-excluded columns at
    /// every `(hydro_id, stage_id, block_id)` key, between two comparably
    /// ordered row lists.
    fn assert_six_columns_bit_identical(
        label: &str,
        base: &[SimulationHydroResult],
        other: &[SimulationHydroResult],
    ) {
        assert_eq!(base.len(), other.len(), "{label}: row count differs");
        for (b, o) in base.iter().zip(other) {
            assert_eq!(
                (b.hydro_id, b.stage_id, b.block_id),
                (o.hydro_id, o.stage_id, o.block_id),
                "{label}: row key mismatch"
            );
            let key = format!(
                "hydro {} stage {} block {:?}",
                b.hydro_id, b.stage_id, b.block_id
            );
            assert_eq!(
                b.integrated_equivalent_productivity_mw_per_m3s.to_bits(),
                o.integrated_equivalent_productivity_mw_per_m3s.to_bits(),
                "{label}: integrated_equivalent_productivity_mw_per_m3s differs at {key}"
            );
            assert_eq!(
                b.integrated_accumulated_productivity_mw_per_m3s.to_bits(),
                o.integrated_accumulated_productivity_mw_per_m3s.to_bits(),
                "{label}: integrated_accumulated_productivity_mw_per_m3s differs at {key}"
            );
            assert_eq!(
                b.stored_energy_initial_mw.to_bits(),
                o.stored_energy_initial_mw.to_bits(),
                "{label}: stored_energy_initial_mw differs at {key}"
            );
            assert_eq!(
                b.stored_energy_final_mw.to_bits(),
                o.stored_energy_final_mw.to_bits(),
                "{label}: stored_energy_final_mw differs at {key}"
            );
            assert_eq!(
                b.stored_energy_initial_mwh.to_bits(),
                o.stored_energy_initial_mwh.to_bits(),
                "{label}: stored_energy_initial_mwh differs at {key}"
            );
            assert_eq!(
                b.stored_energy_final_mwh.to_bits(),
                o.stored_energy_final_mwh.to_bits(),
                "{label}: stored_energy_final_mwh differs at {key}"
            );
        }
    }

    /// Permuting hydro (and every other whitelisted registry's) declaration
    /// order must not change the six parity-hash-excluded columns.
    #[test]
    fn declaration_order_is_invariant() {
        let base_dir = build_fixture();
        let permuted_dir = permute_case(base_dir.path(), PERMUTATION_SEED);

        let base_rows = sorted_hydro_rows(&train_and_simulate(base_dir.path()));
        let permuted_rows = sorted_hydro_rows(&train_and_simulate(permuted_dir.path()));

        assert_six_columns_bit_identical(
            "declaration-order permutation",
            &base_rows,
            &permuted_rows,
        );

        let h0_stage0 = base_rows
            .iter()
            .find(|h| h.hydro_id == 0 && h.stage_id == 0 && h.block_id == Some(0))
            .expect("hydro 0 stage 0 block 0 present");
        assert_ne!(
            h0_stage0
                .integrated_equivalent_productivity_mw_per_m3s
                .to_bits(),
            h0_stage0.equivalent_productivity_mw_per_m3s.to_bits(),
            "fixture must exercise the integrated grid: hydro 0's integrated \
             productivity must differ from its reference-point productivity"
        );
    }

    /// Two fresh training + simulation runs over the SAME (unpermuted) deck must
    /// produce `to_bits()`-identical values for the six columns — the
    /// `clp_determinism.rs` fresh-instance idiom applied to the output layer.
    #[test]
    fn run_to_run_is_reproducible() {
        let dir = build_fixture();
        let run_a = sorted_hydro_rows(&train_and_simulate(dir.path()));
        let run_b = sorted_hydro_rows(&train_and_simulate(dir.path()));

        assert_six_columns_bit_identical("run-to-run reproducibility", &run_a, &run_b);
    }
}
