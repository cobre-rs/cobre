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
    clippy::doc_markdown
)]

use std::path::Path;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
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
///
/// ## Derivation
///
/// Parameters:
/// - Block hours: h = 730
/// - Conversion: κ = 730 × 3600 / 1e6 = 2.628 hm3 per (m3/s · stage)
/// - Initial storage: S₀ = 100.0 hm3
/// - Inflows: q₀ = 40.0 m3/s, q₁ = 10.0 m3/s
/// - Thermal: capacity 100 MW, cost 50 $/MWh
/// - Hydro: productivity 1.0 MW/(m3/s), max_turbined 50 m3/s, max_storage 200 hm3
/// - Demand: D = 80 MW (> 50 MW hydro max → thermal always non-zero)
///
/// ### Stage 1 (terminal, zero future value)
///
/// Turbine all available water up to capacity:
///   turb₁_max = (S₁ + q₁·κ) / κ = S₁/κ + q₁
///
/// When S₁ = q_boundary = (50 − 10)·κ = 40·κ = 105.12 hm3:
///   turb₁ = 50 m3/s (capacity-constrained), gen_th₁ = 30 MW, spill₁ = 0
///
/// Water value (∂cost₁/∂S₁) in the unconstrained region S₁ < 105.12:
///   = −50·730/κ = −50·730/2.628 ≈ −13,889 $/hm3
///
/// ### Stage 0 (Benders cut from stage 1)
///
/// Cost is flat in S₁ ≤ 105.12 because turbine savings in stage 1 exactly
/// offset thermal costs in stage 0 (turb₀ terms cancel).  For S₁ > 105.12
/// the extra water is spilled (tiny spillage cost 0.01 $/hm3) with no
/// reduction in stage-1 thermal.  Hence the optimum is S₁ = 105.12 exactly.
///
/// Water balance without spillage:
///   S₁ = 100 + (40 − turb₀)·κ = 105.12  →  turb₀ = 100/κ = 25000/657 m3/s
///
/// Dispatch:
///   gen_th₀ = 80 − 25000/657 = 27560/657 MW
///   gen_th₁ = 30 MW
///
/// ### Total cost
///
///   cost = (27560/657 × 50 + 30 × 50) × 730
///        = (27560 × 50 × 730) / 657 + 30 × 50 × 730
///        = 1 005 940 000 / 657 + 1 095 000
///        = 23 635 000 / 9
///        ≈ 2 626 111.111... $
///
/// Verification invariants:
///   - Thermal > 0 in both stages (demand 80 > hydro max 50)
///   - Water value (Benders cut coefficient for storage) is non-zero
///   - No spillage at optimum
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

/// Three-stage cascade hydrothermal dispatch. Optimal cost is hand-computed via LP.
///
/// ## Case setup
///
/// - 1 bus, 1 thermal (T0: 100 MW at $50/MWh)
/// - H0 (id=0, upstream): productivity 1.0 MW/(m3/s), max 30 m3/s / 30 MW,
///   storage 0–150 hm3, initial 80 hm3, `downstream_id = 1`
/// - H1 (id=1, downstream): productivity 1.0 MW/(m3/s), max 40 m3/s / 40 MW,
///   storage 0–100 hm3, initial 50 hm3, `downstream_id = null`
/// - Combined hydro capacity = 70 MW < demand 75 MW → thermal always needed
/// - Deterministic inflows (std = 0): H0 = [25, 15, 5] m3/s, H1 = [10, 5, 2] m3/s
/// - Deterministic load: 75.0 MW per stage (3 stages, 730 h each, no discounting)
///
/// H1 water balance receives H0's turbined outflow: `ΔsH1 = (q_H1 + tH0 + spH0 − tH1 − spH1) × κ`.
/// This cascade coupling is the feature exercised by this test.
///
/// ## Expected cost derivation
///
/// **Conversion:** κ = 730 × 3600 / 1 000 000 = 657/250 = 2.628 hm3 per (m3/s · stage)
///
/// **Combined hydro capacity = 70 MW < demand 75 MW → thermal ≥ 5 MW every stage.**
///
/// ### Terminal stage (stage 2, zero future value)
///
/// Maximize hydro; both plants turbine at full capacity:
/// - tH0₂ = 30 m3/s, tH1₂ = 40 m3/s → hydro = 70 MW, thermal₂ = 5 MW
/// - Stage-2 cost = 5 × 50 × 730 = 182 500 $
/// - Water balance requires: sH0₁ ≥ 25κ = 65.7 hm3, sH1₁ ≥ 8κ = 21.024 hm3
///
/// ### Stage 1 (Benders cut from stage 2)
///
/// Again turbine at full capacity: tH0₁ = 30, tH1₁ = 40, thermal₁ = 5 MW
/// - Stage-1 cost = 182 500 $
/// - Water balance requires: sH0₀ ≥ 40κ = 105.12 hm3, sH1₀ ≥ 13κ = 34.164 hm3
///
/// ### Stage 0 (Benders cut from stage 1)
///
/// Minimize thermal₀ = 75 − tH0₀ − tH1₀ subject to end-of-stage storage targets.
///
/// **H0 storage constraint** (sH0₀ ≥ 40κ):
/// ```text
/// 80 + (25 − tH0₀) × κ ≥ 40κ
/// tH0₀ ≤ 80/κ − 15 = 80 × 250/657 − 15 = 10145/657 ≈ 15.4414 m3/s
/// ```
/// Binding: tH0₀ = 10145/657 m3/s (saves enough H0 water for stages 1 and 2).
///
/// **H1 storage constraint** (sH1₀ ≥ 13κ):
/// ```text
/// 50 + (10 + tH0₀ − tH1₀) × κ ≥ 13κ
/// tH1₀ ≤ 50/κ + 10 + tH0₀ − 13 = 50 × 250/657 + tH0₀ − 3
///       = 12500/657 + 10145/657 − 3 = 20674/657 ≈ 31.4673 m3/s
/// ```
/// Binding: tH1₀ = 20674/657 m3/s (saves enough H1 water for stages 1 and 2).
///
/// **Thermal in stage 0:**
/// ```text
/// thermal₀ = 75 − tH0₀ − tH1₀
///           = 75 − 10145/657 − 20674/657
///           = (75 × 657 − 10145 − 20674) / 657
///           = (49275 − 30819) / 657
///           = 18456 / 657
///           = 6152 / 219 ≈ 28.0913 MW
/// ```
/// Stage-0 cost = (6152/219) × 50 × 730 = 3 076 000/3 ≈ 1 025 333.333 $
///
/// ### Total cost
///
/// ```text
/// total = 3 076 000/3 + 182 500 + 182 500
///       = 3 076 000/3 + 365 000
///       = 3 076 000/3 + 1 095 000/3
///       = 4 171 000/3
///       ≈ 1 390 333.333... $
/// ```
///
/// ### Verification invariants
///
/// - Thermal > 0 in every stage (demand 75 > hydro max 70)
/// - Stage 0 thermal ≈ 28.09 MW; stages 1 and 2 thermal = 5 MW exactly
/// - sH0 and sH1 both reach exactly 0 at end of stage 2 (all water depleted)
/// - H1's inflow includes H0's turbined outflow (cascade coupling)
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

/// Two-stage 2-bus transmission dispatch. Optimal cost is hand-computed via LP.
///
/// ## Case setup
///
/// - 2 buses (B0, B1), 1 line (B0→B1, capacity 15 MW bidirectional, exchange_cost 0.01 $/MWh)
/// - T0 (bus=0): 100 MW at 50 $/MWh
/// - H0 (bus=0): productivity 1.0 MW/(m3/s), max 50 m3/s / 50 MW, storage 0–200 hm3, initial 100 hm3
/// - H1 (bus=1): productivity 1.0 MW/(m3/s), max 20 m3/s / 20 MW, storage 0–100 hm3, initial 50 hm3
/// - Deterministic inflows: H0 = [35, 20] m3/s, H1 = [15, 5] m3/s (std=0)
/// - Deterministic load: B0 = 30 MW, B1 = 40 MW (both stages, std=0)
/// - 2 stages, 730 h each, no discounting
///
/// ## Expected cost derivation
///
/// **Conversion:** κ = 730 × 3600 / 1 000 000 = 657/250 = 2.628 hm3 per (m3/s · stage)
///
/// ### Bus power balance
///
/// B1 needs 40 MW per stage. H1 max = 20 MW. Line capacity = 15 MW.
/// B1 can receive at most 35 MW, so **B1 has deficit ≥ 5 MW every stage**.
/// The line always flows B0→B1 at full capacity (B1 is deficit; B0 has excess hydro).
/// B0 must produce 30 + 15 = 45 MW per stage (local demand + export).
///
/// ### H0 capacity check
///
/// H0 initial = 100 hm3. Stage 0: inflow 35 m3/s, turbine 45 m3/s (generation 45 MW).
///   S_H0_1 = 100 + (35 − 45) × κ = 100 − 10 × 2.628 = 73.72 hm3
/// Stage 1: inflow 20 m3/s. Max turbine = min(73.72/κ + 20, 50) = min(48.05, 50) = 48.05 m3/s.
/// Need only 45 m3/s → T0 = 0 MW both stages. **Thermal cost = 0.**
///
/// ### H1 total turbined (water budget)
///
/// Total H1 water available = 50 + (15 + 5) × κ = 102.56 hm3 = (50/κ + 20) × κ
/// Total H1 turbined = 50/κ + 20 = 12500/657 + 20 m3/s (summed over 2 stages)
///
/// The split between stages is indeterminate (degenerate LP: water value identical
/// at both stages since deficit cost is constant). SDDP converges to the same
/// **total** cost regardless of intra-period split.
///
/// ### B1 total deficit
///
/// B1 needs 2 × 25 = 50 MW total from H1 (since line provides 2 × 15 MW).
/// H1 provides 50/κ + 20 = 12500/657 + 20 MW total.
/// Total deficit = 50 − (12500/657 + 20) = 30 − 12500/657 = 7210/657 MW (summed)
///
/// ### Cost components
///
/// Deficit cost:  7210/657 × 1000 × 730 = 5 263 300 000/657 ≈ 8 011 111.11 $
/// Line cost:     2 × 15 × 0.01 × 730  = 219.00 $
/// Thermal cost:  0 (H0 turbines cover B0 demand both stages)
///
/// Total = 5 263 300 000/657 + 219 = 5 263 443 883/657 ≈ **8 011 330.11 $**
///
/// ### Verification invariants
///
/// - B1 deficit > 0 in at least one stage (line capacity binding at 15 MW)
/// - Line flow ≈ 15 MW both stages (always at capacity)
/// - T0 = 0 MW (H0 surplus covers B0 demand plus export)
/// - H1 depleted to 0 at end of stage 1 (all water used to minimize deficit)
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
