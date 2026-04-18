//! Integration tests for `CanonicalStateStrategy` Disabled and ClearSolver variants.
//!
//! Validates end-to-end correctness of the A.1 mechanism:
//!
//! - `Disabled`: legacy per-trial-point `load_model` path. `clear_solver_count` must
//!   stay at zero throughout the training run.
//! - `ClearSolver`: per-(worker, stage) `load_model` with per-trial-point
//!   `clear_solver_state`. Counter must increment; failure counter must stay zero.
//! - Misconfigured backend: using `ClearSolver` with a solver that returns
//!   `Err(Unsupported)` from `clear_solver_state` must panic with the message
//!   established in ticket-004.
//!
//! These tests use a minimal 3-stage, 1-hydro, 0-PAR system (no D-case fixtures).
//! Each test is self-contained and must complete well under 5 seconds.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]

use std::collections::BTreeMap;

use chrono::NaiveDate;
use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::{
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, SamplingScheme,
    },
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
    Bus, DeficitSegment, EntityId,
};
use cobre_solver::{
    Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
};
use cobre_stochastic::{
    build_stochastic_context, ClassSchemes, OpeningTreeInputs, StochasticContext,
};

use cobre_sddp::{
    cut::fcf::FutureCostFunction, train, CanonicalStateStrategy, CutManagementConfig, EventConfig,
    HorizonMode, InflowNonNegativityMethod, LoopConfig, RiskMeasure, SolverStatsEntry,
    StageContext, StageIndexer, StoppingMode, StoppingRule, StoppingRuleSet, TrainingConfig,
    TrainingContext,
};

use cobre_solver::highs::HighsSolver;

// ---------------------------------------------------------------------------
// Shared communicator stub
// ---------------------------------------------------------------------------

/// Single-rank communicator that correctly copies data through `allgatherv`
/// and `allreduce`. Mirrors the `StubComm` from `integration.rs`.
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
// Mock solver for the noop-backend panic test
// ---------------------------------------------------------------------------

/// Minimal mock solver that tracks row count for dual buffer sizing.
///
/// Does NOT override `clear_solver_state`, so the default implementation
/// returns `Err(SolverError::Unsupported(...))`, which triggers the
/// `CanonicalStateStrategy::ClearSolver` panic in the backward pass.
struct NoopSolver {
    current_num_rows: usize,
    dual_buf: Vec<f64>,
    primal_buf: Vec<f64>,
}

impl NoopSolver {
    fn new() -> Self {
        Self {
            current_num_rows: 0,
            dual_buf: vec![0.0_f64; 64],
            primal_buf: vec![0.0_f64; 4],
        }
    }
}

impl SolverInterface for NoopSolver {
    fn solver_name_version(&self) -> String {
        "NoopSolver 0.0.0".to_string()
    }

    fn load_model(&mut self, template: &StageTemplate) {
        self.current_num_rows = template.num_rows;
    }

    fn add_rows(&mut self, cuts: &RowBatch) {
        self.current_num_rows += cuts.num_rows;
        if self.current_num_rows > self.dual_buf.len() {
            self.dual_buf.resize(self.current_num_rows, 0.0);
        }
    }

    fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

    fn solve(&mut self) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
        if self.dual_buf.len() < self.current_num_rows {
            self.dual_buf.resize(self.current_num_rows, 0.0);
        }
        Ok(cobre_solver::SolutionView {
            objective: 0.0,
            primal: &self.primal_buf,
            dual: &self.dual_buf[..self.current_num_rows],
            reduced_costs: &self.primal_buf,
            iterations: 0,
            solve_time_seconds: 0.0,
        })
    }

    fn reset(&mut self) {}

    fn get_basis(&mut self, _out: &mut Basis) {}

    fn solve_with_basis(
        &mut self,
        _basis: &Basis,
    ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
        self.solve()
    }

    fn statistics(&self) -> SolverStatistics {
        SolverStatistics::default()
    }

    fn name(&self) -> &'static str {
        "Noop"
    }
    // `clear_solver_state` is NOT overridden: the default returns
    // `Err(SolverError::Unsupported("clear_solver_state not implemented for this backend"))`.
}

// ---------------------------------------------------------------------------
// Stochastic context and LP template helpers
// ---------------------------------------------------------------------------

/// Build a `StochasticContext` for `n_stages` stages with a single hydro,
/// zero PAR order, and 1 branching opening per stage.
///
/// Mirrors `make_stochastic_context` from `integration.rs`.
#[allow(clippy::too_many_lines)]
fn make_stochastic_context(n_stages: usize) -> StochasticContext {
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::InflowModel;

    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };

    let hydro = Hydro {
        id: EntityId(1),
        name: "H1".to_string(),
        bus_id: EntityId(0),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 100.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity {
            productivity_mw_per_m3s: 1.0,
        },
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            fpha_turbined_cost: 0.0,
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
            inflow_nonnegativity_cost: 1000.0,
        },
    };

    let make_stage = |idx: usize| Stage {
        index: idx,
        id: idx as i32,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks: vec![Block {
            index: 0,
            name: "S".to_string(),
            duration_hours: 744.0,
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
    };

    let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();

    let inflow_models: Vec<_> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(1),
            stage_id: i as i32,
            mean_m3s: 100.0,
            std_m3s: 30.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        })
        .collect();

    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: vec![CorrelationEntity {
                    entity_type: "inflow".to_string(),
                    id: EntityId(1),
                }],
                matrix: vec![vec![1.0]],
            }],
        },
    );
    let correlation = CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    };

    let system = cobre_core::SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .build()
        .unwrap();

    build_stochastic_context(
        &system,
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
    .unwrap()
}

/// Minimal stage template for N=1 hydro, L=0 PAR, compatible with `HighsSolver`.
///
/// Row layout (3 rows):
/// - Row 0: storage-fixing row — `x_si = state`            (Category 1 patches)
/// - Row 1: z-inflow definition at `N*(1+L)=1` — `x_z = 0` (Category 5 patches)
/// - Row 2: water-balance at `base_rows[s]=2` — `x_s - x_z - x_si = 0` (Category 3)
///
/// All three rows have at least one non-zero coefficient so HiGHS builds a
/// non-degenerate basis structure. Rows 1 and 2 are patched to RHS=0 throughout
/// the test (Category 5 gives `z_inflow_rhs=0`; Category 3 uses `noise_scale=0`),
/// so the only feasible solution is the all-zero point.
///
/// `base_rows[s] = 2` is valid because `num_rows = 3 > 2`.
///
/// Column layout:
/// - Col 0 `x_s`:   outgoing storage  [0, 100]  — state variable
/// - Col 1 `x_z`:   z_inflow          [0, 200]  — inflow variable
/// - Col 2 `x_si`:  incoming storage  [0, 100]  — patched to `state` by Category 1
/// - Col 3 `theta`: future cost       [0, ∞]    — objective variable
///
/// CSC matrix (5 NNZ):
/// - Col 0 → row 2, coeff +1.0  (`x_s` in water-balance)
/// - Col 1 → row 1, coeff +1.0  (`x_z` in z-inflow def)
/// - Col 1 → row 2, coeff −1.0  (`x_z` in water-balance)
/// - Col 2 → row 0, coeff +1.0  (`x_si` in storage-fixing)
/// - Col 2 → row 2, coeff −1.0  (`x_si` in water-balance)
fn minimal_template() -> StageTemplate {
    StageTemplate {
        num_cols: 4,
        num_rows: 3,
        num_nz: 5,
        // col_starts: [0, 1, 3, 5, 5]  (Col0: 1 NNZ, Col1: 2 NNZ, Col2: 2 NNZ, Col3: 0)
        col_starts: vec![0_i32, 1, 3, 5, 5],
        // col0→row2, col1→row1, col1→row2, col2→row0, col2→row2
        row_indices: vec![2_i32, 1, 2, 0, 2],
        values: vec![1.0_f64, 1.0, -1.0, 1.0, -1.0],
        col_lower: vec![0.0, 0.0, 0.0, 0.0],
        col_upper: vec![100.0, 200.0, 100.0, f64::INFINITY],
        objective: vec![0.0, 0.0, 0.0, 1.0],
        row_lower: vec![0.0, 0.0, 0.0],
        row_upper: vec![0.0, 0.0, 0.0],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 1,
        n_hydro: 1,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

fn iteration_limit(limit: u64) -> StoppingRuleSet {
    StoppingRuleSet {
        rules: vec![StoppingRule::IterationLimit { limit }],
        mode: StoppingMode::Any,
    }
}

// ---------------------------------------------------------------------------
// Aggregation helper
// ---------------------------------------------------------------------------

/// Sum `clear_solver_count` across all log entries.
fn total_clear_solver_count(log: &[SolverStatsEntry]) -> u64 {
    log.iter().map(|(_, _, _, d)| d.clear_solver_count).sum()
}

/// Sum `clear_solver_failures` across all log entries.
fn total_clear_solver_failures(log: &[SolverStatsEntry]) -> u64 {
    log.iter().map(|(_, _, _, d)| d.clear_solver_failures).sum()
}

// ---------------------------------------------------------------------------
// Shared training run builder
// ---------------------------------------------------------------------------

/// Configuration for the minimal 3-stage, 5-scenario, 3-iteration training run.
const N_STAGES: usize = 3;
const FORWARD_PASSES: u32 = 5;
const MAX_ITERATIONS: u64 = 3;
const FCF_CAPACITY: u64 = 50;

/// Run a minimal training pass with the given `CanonicalStateStrategy` and
/// the provided solver and factory. Returns the `TrainingOutcome`.
///
/// Panics propagate to the caller — used directly by `#[should_panic]` tests.
fn run_minimal_training<S>(
    solver: &mut S,
    strategy: CanonicalStateStrategy,
    solver_factory: impl Fn() -> Result<S, cobre_solver::SolverError>,
) -> cobre_sddp::TrainingOutcome
where
    S: SolverInterface + Send,
{
    let indexer = StageIndexer::new(1, 0); // N=1 hydro, L=0 PAR
    let n_state = indexer.n_state;
    let templates = vec![minimal_template(); N_STAGES];
    // base_rows[s] = 2: Category 3 patches LP row 2 (water-balance).
    // Template has num_rows=3, so row 2 is valid for HighsSolver.
    let base_rows = vec![2usize; N_STAGES];
    // noise_scale: one entry per (stage, hydro). With N=1 hydro and N_STAGES stages,
    // length = N_STAGES * 1.
    //
    // Use 0.0 so that noise[h] = row_lower[base_row+h] + 0.0*eta = 0.0 for all eta.
    // This keeps the water-balance row (Category 3) a trivial 0=0 equality that
    // HighsSolver accepts as feasible, regardless of the sampled inflow noise.
    // A non-zero noise_scale would produce a non-zero RHS in an empty row
    // (no variable entries in the minimal template), making HighsSolver declare
    // the LP infeasible.
    let noise_scale = vec![0.0_f64; N_STAGES];
    let initial_state = vec![0.0_f64; n_state];
    let stochastic = make_stochastic_context(N_STAGES);
    let horizon = HorizonMode::Finite {
        num_stages: N_STAGES,
    };
    let risk_measures = vec![RiskMeasure::Expectation; N_STAGES];
    let block_counts: Vec<usize> = vec![1; N_STAGES];

    let mut fcf = FutureCostFunction::new(
        N_STAGES,
        n_state,
        FORWARD_PASSES,
        FCF_CAPACITY,
        &vec![0; N_STAGES],
    );

    let stage_ctx = StageContext {
        templates: &templates,
        base_rows: &base_rows,
        noise_scale: &noise_scale,
        n_hydros: 1,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &block_counts,
        ncs_max_gen: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
    };

    train(
        solver,
        TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: FORWARD_PASSES,
                max_iterations: MAX_ITERATIONS,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit(MAX_ITERATIONS),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures,
                canonical_state_strategy: strategy,
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        },
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            stages: &[],
        },
        &StubComm,
        solver_factory,
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Under `CanonicalStateStrategy::Disabled`, `clear_solver_state` is never
/// called, so the aggregated `clear_solver_count` must be zero for the entire
/// training run.
///
/// Uses real `HighsSolver` workers to exercise the actual backward LP path.
#[test]
fn backward_pass_under_disabled_has_zero_clear_solver_count() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");
    let outcome = run_minimal_training(
        &mut solver,
        CanonicalStateStrategy::Disabled,
        HighsSolver::new,
    );
    assert!(
        outcome.error.is_none(),
        "expected no training error; got: {:?}",
        outcome.error
    );
    let count = total_clear_solver_count(&outcome.result.solver_stats_log);
    assert_eq!(
        count, 0,
        "Disabled strategy must never call clear_solver_state; got clear_solver_count={count}"
    );
}

/// Under `CanonicalStateStrategy::ClearSolver`, `clear_solver_state` must be
/// called at least once (once per backward trial point after the first per
/// worker/stage), and no calls must fail.
///
/// Uses real `HighsSolver` workers because `MockSolver` does not implement
/// `clear_solver_state` and would panic on the first call.
#[test]
fn backward_pass_under_clear_solver_increments_counter() {
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");
    let outcome = run_minimal_training(
        &mut solver,
        CanonicalStateStrategy::ClearSolver,
        HighsSolver::new,
    );
    assert!(
        outcome.error.is_none(),
        "expected no training error; got: {:?}",
        outcome.error
    );
    let count = total_clear_solver_count(&outcome.result.solver_stats_log);
    let failures = total_clear_solver_failures(&outcome.result.solver_stats_log);
    assert!(
        count > 0,
        "ClearSolver strategy must call clear_solver_state at least once; got clear_solver_count=0"
    );
    assert_eq!(
        failures, 0,
        "ClearSolver strategy must have zero failures; got clear_solver_failures={failures}"
    );
}

/// When `CanonicalStateStrategy::ClearSolver` is active but the solver backend
/// does not implement `clear_solver_state` (returns `Err(Unsupported)`), the
/// backward pass must panic with a message containing
/// `"CanonicalStateStrategy::ClearSolver"`.
///
/// This guards the programming error detection established in ticket-004.
///
/// Implementation note: `NoopSolver` inherits the default `clear_solver_state`
/// which returns `Err(SolverError::Unsupported(...))`. No end-to-end D-case is
/// needed — the panic fires on the first backward trial point.
#[test]
#[should_panic(expected = "CanonicalStateStrategy::ClearSolver")]
fn noop_solver_backend_rejects_clear_solver_strategy() {
    let mut solver = NoopSolver::new();
    // The panic propagates out of `train()` via rayon, through the `?` in the
    // backward worker loop, and surfaces as a panic before `unwrap()` in
    // `run_minimal_training`. The `#[should_panic]` harness catches it.
    let _ = run_minimal_training(&mut solver, CanonicalStateStrategy::ClearSolver, || {
        Ok(NoopSolver::new())
    });
}
