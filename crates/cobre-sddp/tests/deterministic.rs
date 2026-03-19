//! Deterministic test suite for the SDDP pipeline.
//!
//! Each test case in this suite is fully deterministic: load is constant
//! (zero variance), there are no stochastic inflows, and the optimal cost is
//! hand-computable. This makes it possible to assert exact cost values and
//! tight convergence bounds.
//!
//! ## Design philosophy
//!
//! - Cases live under `examples/deterministic/<case-id>/` in the workspace root.
//! - Every test calls `run_deterministic` with the case directory path and then
//!   asserts on `TrainingResult.final_lb` and `TrainingResult.iterations`.
//! - Expected costs are derived analytically in the ticket specification; the
//!   derivation is documented in each test's doc comment.
//! - `StubComm` provides a single-rank communicator that faithfully copies data
//!   through `allgatherv` and `allreduce` so the pipeline runs without MPI.

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

use std::path::Path;
use std::sync::mpsc;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_io::{
    PolicyCheckpointMetadata, PolicyCutRecord, StageCutsPayload, write_policy_checkpoint,
};
use cobre_sddp::{
    StudySetup, aggregate_simulation, hydro_models::prepare_hydro_models, setup::prepare_stochastic,
};
use cobre_solver::SolverInterface;
use cobre_solver::highs::HighsSolver;

/// Single-rank communicator stub for deterministic testing.
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

/// Execute the full training pipeline for a case directory and return both the
/// `TrainingResult` and the `HighsSolver`. Uses `StubComm`, seed 42, and 1 thread.
///
/// Unlike `run_deterministic`, this helper keeps the solver alive so callers can
/// inspect `solver.statistics()` (e.g. `basis_rejections`) after training completes.
fn run_deterministic_with_solver(case_dir: &Path) -> (cobre_sddp::TrainingResult, HighsSolver) {
    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let prepare_result =
        prepare_stochastic(system, case_dir, &config, 42).expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir).expect("prepare_hydro_models must succeed");

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");

    let result = setup
        .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
        .expect("train must return Ok");
    (result, solver)
}

/// Execute the full training pipeline for a case directory and return the
/// `TrainingResult`. Uses `StubComm`, `HighsSolver`, seed 42, and 1 thread.
fn run_deterministic(case_dir: &Path) -> cobre_sddp::TrainingResult {
    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let prepare_result =
        prepare_stochastic(system, case_dir, &config, 42).expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir).expect("prepare_hydro_models must succeed");

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");

    setup
        .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
        .expect("train must return Ok")
}

/// Assert that `actual` is within `tolerance` of `expected`.
///
/// Panics with a diagnostic message identifying the case and the actual vs
/// expected values when the assertion fails.
fn assert_cost(actual: f64, expected: f64, tolerance: f64, case_name: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tolerance,
        "{case_name}: expected cost {expected}, got {actual} (diff={diff} > tolerance={tolerance})"
    );
}

/// Expected total cost for D02 (single hydro, 2 stages, deterministic inflows).
/// Derivation: κ=2.628, S₀=100hm³, inflows=[40,10]m³/s, demand=80MW.
/// Terminal stage turbines at 50m³/s capacity. Backward pass shows optimal
/// storage at stage boundary = 105.12 hm³. Total cost = 23,635,000/9 $.
pub const D02_EXPECTED_COST: f64 = 23_635_000.0 / 9.0;

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

/// Three-stage cascade hydrothermal dispatch (2 hydros in series).
/// Combined capacity 70 MW < demand 75 MW. Cascade coupling: H1 receives H0's
/// discharge. Terminal stages: full capacity (thermal = 5 MW). Stage 0 binding
/// storage constraints yield thermal ≈ 28.09 MW. Total cost = 4,171,000/3 $.
pub const D03_EXPECTED_COST: f64 = 4_171_000.0 / 3.0;

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
/// H1 depleted to minimize loss. Total cost ≈ 8,011,330 $.
pub const D04_EXPECTED_COST: f64 = 5_263_443_883.0 / 657.0;

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
#[test]
fn d05_fpha_constant_head() {
    let case_dir = Path::new("../../examples/deterministic/d05-fpha-constant-head");
    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, D02_EXPECTED_COST, 1e-6, "D05");
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
/// V_in is fixed by the Benders storage-fixing row at the reference point
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
#[test]
fn d06_fpha_variable_head() {
    let case_dir = Path::new("../../examples/deterministic/d06-fpha-variable-head");
    let result = run_deterministic(case_dir);
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
/// ## Assertions
///
/// - Convergence: `final_gap.abs() < 1e-6` (tight gap).
/// - Iterations: `<= 10` (fast convergence on deterministic case).
/// - Sanity: `final_lb > 0.0` (positive cost; system requires thermal dispatch).
/// - The computed cost differs from D06 because the fitting pipeline uses a
///   different discretization grid and number of planes; no exact match is asserted.
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
/// ## Evaporation coefficient derivation
///
/// Midpoint v_ref = (0 + 200) / 2 = 100 hm3.
/// a_ref = 1.0 km², da/dv = 0.005 km²/hm3.
/// stage_hours = 730 h → zeta_evap = 1 / (3.6 × 730) = 1/2628.
/// c_ev = 100 mm.
/// k_evap_v = (1/2628) × 100 × 0.005 = 1/5256.
/// k_evap0  = (1/2628) × 100 × 1.0 − (1/5256) × 100 = 25/1314.
///
/// ## Water-balance model (α = κ × k_evap_v / 2 = 1/4000)
///
/// Substituting Q_ev = k_evap0 + k_evap_v/2 × (V_out + V_in) into the LP:
///   V_out × (1 + α) = V_in × (1 − α) + κ × (q_in − q_turb) − κ × k_evap0
///
/// ## Expected cost derivation (κ = 657/250 hm3/(m3/s))
///
/// The gradient of total cost w.r.t. turb₀ is proportional to
/// `−1 + (1−α)/(1+α) < 0`, so increasing turb₀ decreases total cost.
/// The optimal stage-0 policy is therefore turb₀ = 50 m3/s (full capacity).
///
/// ### Stage 0 (turb₀ = 50, V_in₀ = 100 hm3)
///
/// V_out₀ = [100×(1−α) + κ×(40 − 50) − κ×k_evap0] / (1+α)
///         = 294580/4001 ≈ 73.627 hm3.
/// gen_h₀ = 50 MW, gen_th₀ = 30 MW.
/// Stage 0 cost = 30 × 50 × 730 = 1,095,000 $.
///
/// ### Stage 1 (terminal, V_in₁ = V_out₀ = 294580/4001 hm3)
///
/// turb₁_max = V_in₁ × (1−α) / κ + 10 − k_evap0
///           = 399452585/10514628 ≈ 37.990 m3/s  (<50, so binding).
/// At turb₁ = turb₁_max: V_out₁ = 0.
///
/// gen_th₁ = 80 − turb₁_max = 441717655/10514628 ≈ 42.010 MW.
/// Stage 1 cost = gen_th₁ × 50 × 730 = 55214706875/36009 ≈ 1,533,358.52 $.
///
/// ### Total cost
///
/// Total = 1,095,000 + 55214706875/36009
///       = 39439545000/36009 + 55214706875/36009
///       = **94644561875/36009 ≈ 2,628,358.52 $**
///
/// D08 cost > D02 cost (≈ 2,626,111.11 $): evaporation consumes additional water
/// in the reservoir, leaving less for stage-1 generation and requiring more
/// thermal dispatch.
pub const D08_EXPECTED_COST: f64 = 94_644_561_875.0 / 36_009.0;

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
/// ## Penalty cost unit (verified from lp_builder.rs)
///
/// From `lp_builder.rs` (`build_stage_templates`):
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
/// See [`D10_EXPECTED_COST`] for the full derivation. Stage 1's negative inflow
/// activates a 5 m3/s slack (cost 1,825,000 $). The total cost is
/// 3,164,185,000/657 ≈ 4,815,350.08 $.
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
pub const D11_WATER_WITHDRAWAL_EXPECTED_COST: f64 = 3_930_320_000.0 / 657.0;

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
/// withdrawal reduces effective net inflow from 30 to 20 m3/s, increasing thermal
/// dispatch and pushing total cost to 3,930,320,000 / 657 ≈ 5,982,222.22 $.
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

/// Warm-start verification for the D02 system.
///
/// Validates that basis transfer via `solve_with_basis` works end-to-end: after
/// the training loop completes, `SolverStatistics.basis_rejections` must be zero.
/// A non-zero count would indicate silent cold-start fallbacks, which would
/// degrade performance without surfacing an error.
///
/// Reuses the D02 example directory (no new case data needed).
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
        stats.basis_rejections, 0,
        "D11: expected 0 basis rejections, got {}",
        stats.basis_rejections
    );
}

/// Checkpoint round-trip for the D02 system.
///
/// Exercises the full FlatBuffers persistence pipeline end-to-end:
/// 1. Train D02 to convergence.
/// 2. Build training output and write the policy checkpoint.
/// 3. Read the checkpoint back and verify metadata fields.
/// 4. Run simulation with the trained FCF (reloaded cuts in `setup`).
/// 5. Assert mean simulation cost matches the training LB within 1e-2.
///
/// ## Why the simulation cost should equal the training LB
///
/// D02 has deterministic (zero-variance) inflows and a deterministic load.
/// With `num_scenarios = 1`, the single simulation scenario uses the same
/// inflow realization as the training forward pass. Because the system is
/// fully deterministic and the FCF converges to the true value function, the
/// simulation cost equals the optimal cost captured by the training LB.
///
/// Reuses the D02 example directory (no new case data needed).
#[test]
fn d12_checkpoint_round_trip() {
    let case_dir = Path::new("../../examples/deterministic/d02-single-hydro");

    // ── Step 1: load config and case ─────────────────────────────────────────

    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    // ── Step 2: prepare stochastic and hydro models ───────────────────────────

    let pr =
        prepare_stochastic(system, case_dir, &config, 42).expect("prepare_stochastic must succeed");
    let system = pr.system;
    let stochastic = pr.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir).expect("prepare_hydro_models must succeed");

    // ── Step 3: override simulation config ────────────────────────────────────

    let mut config_with_sim = config.clone();
    config_with_sim.simulation.enabled = true;
    config_with_sim.simulation.num_scenarios = 1;

    // ── Step 4: build StudySetup and train ────────────────────────────────────

    let mut setup = StudySetup::new(&system, &config_with_sim, stochastic, hydro_models)
        .expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");

    let result = setup
        .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
        .expect("train must return Ok");

    assert_cost(result.final_lb, D02_EXPECTED_COST, 1e-4, "D12-train");
    assert!(
        result.final_gap.abs() < 1e-6,
        "D12: gap={:.2e}",
        result.final_gap
    );

    // ── Step 5: build training output (needed for StageCutsPayload extraction) ─

    let _training_output = setup.build_training_output(&result, &[]);

    // ── Step 6: write policy checkpoint ──────────────────────────────────────

    let tmp = tempfile::tempdir().expect("tempdir must succeed");
    let policy_dir = tmp.path().join("policy");

    // Access the FCF pools to construct StageCutsPayload references.
    let fcf = setup.fcf();

    let cut_records_per_stage: Vec<Vec<PolicyCutRecord<'_>>> = fcf
        .pools
        .iter()
        .map(|pool| {
            (0..pool.populated_count)
                .map(|slot| {
                    let meta = &pool.metadata[slot];
                    PolicyCutRecord {
                        cut_id: slot as u64,
                        slot_index: slot as u32,
                        iteration: meta.iteration_generated as u32,
                        forward_pass_index: meta.forward_pass_index,
                        intercept: pool.intercepts[slot],
                        coefficients: &pool.coefficients[slot],
                        is_active: pool.active[slot],
                        domination_count: meta.domination_count as u32,
                    }
                })
                .collect()
        })
        .collect();

    let active_indices_per_stage: Vec<Vec<u32>> = fcf
        .pools
        .iter()
        .map(|pool| {
            (0..pool.populated_count)
                .filter(|&slot| pool.active[slot])
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
            populated_count: pool.populated_count as u32,
        })
        .collect();

    let n_stages = fcf.pools.len();
    let policy_metadata = PolicyCheckpointMetadata {
        version: "1.0.0".to_string(),
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: "2026-03-16T00:00:00Z".to_string(),
        completed_iterations: result.iterations as u32,
        final_lower_bound: result.final_lb,
        best_upper_bound: Some(result.final_ub),
        state_dimension: fcf.state_dimension as u32,
        num_stages: n_stages as u32,
        config_hash: "d12-config-hash".to_string(),
        system_hash: "d12-system-hash".to_string(),
        max_iterations: 100,
        forward_passes: 1,
        warm_start_cuts: 0,
        rng_seed: 42,
    };

    write_policy_checkpoint(&policy_dir, &stage_cuts_payloads, &[], &policy_metadata)
        .expect("write_policy_checkpoint must succeed");

    // ── Step 7: read checkpoint back and verify metadata ─────────────────────

    let checkpoint =
        cobre_io::read_policy_checkpoint(&policy_dir).expect("read_policy_checkpoint must succeed");

    assert_eq!(
        checkpoint.metadata.num_stages, 2,
        "D12: checkpoint must have 2 stages"
    );
    assert_eq!(
        checkpoint.metadata.state_dimension, 1,
        "D12: checkpoint must have state_dimension == 1 (one hydro = one storage state)"
    );
    assert!(
        !checkpoint.stage_cuts.is_empty(),
        "D12: checkpoint must contain at least one stage_cuts entry"
    );

    let metadata_path = policy_dir.join("metadata.json");
    assert!(metadata_path.is_file(), "D12: metadata.json must exist");

    let stage_bin_path = policy_dir.join("cuts/stage_000.bin");
    assert!(
        stage_bin_path.is_file(),
        "D12: cuts/stage_000.bin must exist"
    );

    // ── Step 8: simulate using the FCF already in setup ───────────────────────

    let mut pool = setup
        .create_workspace_pool(1, HighsSolver::new)
        .expect("simulation workspace pool must build");

    let io_capacity = setup.io_channel_capacity().max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);

    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

    let local_costs = setup
        .simulate(&mut pool.workspaces, &comm, &result_tx, None)
        .expect("simulate must return Ok");

    drop(result_tx);
    let _scenario_results = drain_handle.join().expect("drain thread must not panic");

    // ── Step 9: compute mean simulation cost and compare to training LB ───────

    let sim_config = setup.simulation_config();
    let summary = aggregate_simulation(&local_costs, &sim_config, &comm)
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
#[test]
fn d13_generic_constraint() {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let case_dir = Path::new("../../examples/deterministic/d13-generic-constraint");

    // Create the generic_constraint_bounds.parquet before running the case.
    let constraints_dir = case_dir.join("constraints");
    std::fs::create_dir_all(&constraints_dir).expect("create constraints dir");

    let schema = Arc::new(Schema::new(vec![
        Field::new("constraint_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("block_id", DataType::Int32, true),
        Field::new("bound", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1])), // constraint_id
            Arc::new(Int32Array::from(vec![0, 1])), // stage_id
            Arc::new(Int32Array::new_null(2)),      // block_id = null (all blocks)
            Arc::new(Float64Array::from(vec![10.0, 10.0])), // bound = 10 MW
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
