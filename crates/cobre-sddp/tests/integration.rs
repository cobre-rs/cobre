//! End-to-end integration tests for the SDDP training loop.
//!
//! Exercises the full [`cobre_sddp::train`] function with a small toy system
//! (1 hydro, 0 PAR order, 2 stages), verifying convergence behaviour,
//! determinism, lower-bound monotonicity, event emission order, stopping rule
//! termination, and error propagation.
//!
//! ## Design constraints
//!
//! - Only the public `cobre_sddp::` API is used (no `#[cfg(test)]` items).
//! - All test helpers are defined locally in this file.
//! - Each test is self-contained with no cross-test shared state.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

// External crate imports

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;

use chrono::NaiveDate;
use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::{
    Bus, DeficitSegment, EntityId, TrainingEvent,
    scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_solver::{
    Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
};
use cobre_stochastic::{
    OpeningTree, StochasticContext, build_stochastic_context,
    correlation::resolve::DecomposedCorrelation, tree::generate::generate_opening_tree,
};

use cobre_sddp::{
    HorizonMode, InflowNonNegativityMethod, RiskMeasure, SddpError, StageIndexer, StoppingMode,
    StoppingRule, StoppingRuleSet, TrainingConfig, cut::fcf::FutureCostFunction, train,
};

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
        // Single rank: copy send into recv (identity gather).
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

/// Communicator wrapper that sets `flag` to `true` on the first `allreduce`
/// call, simulating a shutdown signal arriving mid-iteration-1. On
/// subsequent calls it behaves identically to [`StubComm`].
///
/// The `allreduce` is called during `sync_forward` (step 2), so by the time
/// iteration 2's convergence check runs the shutdown flag is already set.
struct ShutdownComm {
    flag: Arc<AtomicBool>,
    /// Count of allreduce calls so we only flip the flag once.
    allreduce_calls: AtomicUsize,
}

impl ShutdownComm {
    fn new(flag: Arc<AtomicBool>) -> Self {
        Self {
            flag,
            allreduce_calls: AtomicUsize::new(0),
        }
    }
}

impl Communicator for ShutdownComm {
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
        // Set the shutdown flag on the very first allreduce call (iteration 1
        // forward sync). The training loop checks the flag at the start of each
        // iteration, so iteration 2 will trigger GracefulShutdown.
        if self.allreduce_calls.fetch_add(1, Ordering::Relaxed) == 0 {
            self.flag.store(true, Ordering::Relaxed);
        }
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

/// Mock solver that returns objectives from a repeating sequence.
///
/// - `objectives` cycles with `call_count % len` on each `solve` call.
/// - `infeasible_on_call` triggers `SolverError::Infeasible` at that call index.
/// - `reset()` resets `call_count` to 0 for determinism tests.
struct MockSolver {
    objectives: Vec<f64>,
    call_count: usize,
    infeasible_on_call: Option<usize>,
}

impl MockSolver {
    fn with_objectives(objectives: Vec<f64>) -> Self {
        Self {
            objectives,
            call_count: 0,
            infeasible_on_call: None,
        }
    }

    fn with_fixed(objective: f64) -> Self {
        Self::with_objectives(vec![objective])
    }

    fn infeasible_on_first() -> Self {
        Self {
            objectives: vec![0.0],
            call_count: 0,
            infeasible_on_call: Some(0),
        }
    }
}

impl SolverInterface for MockSolver {
    fn load_model(&mut self, _template: &StageTemplate) {}
    fn add_rows(&mut self, _cuts: &RowBatch) {}
    fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

    fn solve(&mut self) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
        let call = self.call_count;
        self.call_count += 1;
        if self.infeasible_on_call == Some(call) {
            return Err(SolverError::Infeasible);
        }
        let obj = self.objectives[call % self.objectives.len()];
        Ok(cobre_solver::SolutionView {
            objective: obj,
            primal: &[0.0, 0.0, 0.0],
            dual: &[0.0],
            reduced_costs: &[0.0, 0.0, 0.0],
            iterations: 0,
            solve_time_seconds: 0.0,
        })
    }

    fn reset(&mut self) {
        self.call_count = 0;
    }

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
        "MockIntegration"
    }
}

/// Build an `OpeningTree` with `n_openings` at stage 0 using seed 42.
fn make_opening_tree(n_openings: usize) -> OpeningTree {
    let stage = Stage {
        index: 0,
        id: 0,
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
            branching_factor: n_openings,
            noise_method: NoiseMethod::Saa,
        },
    };

    let entity_id = EntityId(1);
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: vec![CorrelationEntity {
                    entity_type: "inflow".to_string(),
                    id: entity_id,
                }],
                matrix: vec![vec![1.0]],
            }],
        },
    );
    let corr_model = CorrelationModel {
        method: "cholesky".to_string(),
        profiles,
        schedule: vec![],
    };
    let mut decomposed = DecomposedCorrelation::build(&corr_model).unwrap();
    let entity_order = vec![entity_id];

    generate_opening_tree(42, &[stage], 1, &mut decomposed, &entity_order)
}

/// Build a `StochasticContext` with `n_stages` stages, 1 hydro, and seed 42.
#[allow(clippy::cast_possible_wrap, clippy::too_many_lines)]
fn make_stochastic_context(n_stages: usize, n_openings: usize) -> StochasticContext {
    use cobre_core::SystemBuilder;
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
            branching_factor: n_openings,
            noise_method: NoiseMethod::Saa,
        },
    };

    let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
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
        method: "cholesky".to_string(),
        profiles,
        schedule: vec![],
    };

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .build()
        .unwrap();

    build_stochastic_context(&system, 42).unwrap()
}

/// Minimal stage template for N=1 hydro, L=0 PAR.
fn minimal_template() -> StageTemplate {
    StageTemplate {
        num_cols: 3,
        num_rows: 1,
        num_nz: 1,
        col_starts: vec![0, 0, 1, 1],
        row_indices: vec![0],
        values: vec![1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
        objective: vec![0.0, 0.0, 1.0],
        row_lower: vec![0.0],
        row_upper: vec![0.0],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 1,
        n_hydro: 1,
        max_par_order: 0,
    }
}

fn make_fcf(n_stages: usize) -> FutureCostFunction {
    FutureCostFunction::new(n_stages, 1, 1, FCF_CAPACITY_ITERATIONS, 0)
}

fn iteration_limit(limit: u64) -> StoppingRuleSet {
    StoppingRuleSet {
        rules: vec![StoppingRule::IterationLimit { limit }],
        mode: StoppingMode::Any,
    }
}

/// All training parameters for a 2-stage, N=1 toy system.
struct Fixture {
    n_stages: usize,
    templates: Vec<StageTemplate>,
    base_rows: Vec<usize>,
    indexer: StageIndexer,
    initial_state: Vec<f64>,
    opening_tree: OpeningTree,
    stochastic: StochasticContext,
    horizon: HorizonMode,
    risk_measures: Vec<RiskMeasure>,
}

const FCF_CAPACITY_ITERATIONS: u64 = 50;

impl Fixture {
    fn new(n_stages: usize) -> Self {
        let indexer = StageIndexer::new(1, 0); // N=1, L=0
        let templates = vec![minimal_template(); n_stages];
        // base_row: the AR-dynamics row offset is 1 (1 dual-relevant row)
        let base_rows = vec![1usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let opening_tree = make_opening_tree(1);
        let stochastic = make_stochastic_context(n_stages, 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        Self {
            n_stages,
            templates,
            base_rows,
            indexer,
            initial_state,
            opening_tree,
            stochastic,
            horizon,
            risk_measures,
        }
    }
}

// Tests

/// Verify the full training loop runs to completion under `IterationLimit`.
///
/// Exercises: `run_forward_pass` → `sync_forward` → `ExchangeBuffers::exchange`
/// → `run_backward_pass` → `CutSyncBuffers::sync_cuts` → `evaluate_lower_bound`
/// → `ConvergenceMonitor::update` → `TrainingResult` fields.
#[test]
fn train_converges_with_mock_solver() {
    let fx = Fixture::new(2);
    let mut fcf = make_fcf(fx.n_stages);
    let mut solver = MockSolver::with_fixed(100.0);
    let comm = StubComm;

    let config = TrainingConfig {
        forward_passes: 1,
        max_iterations: 10,
        checkpoint_interval: None,
        warm_start_cuts: 0,
        event_sender: None,
    };

    let result = train(
        &mut solver,
        config,
        &mut fcf,
        &fx.templates,
        &fx.base_rows,
        &fx.indexer,
        &fx.initial_state,
        &fx.opening_tree,
        &fx.stochastic,
        &fx.horizon,
        &fx.risk_measures,
        iteration_limit(10),
        None,
        None,
        &comm,
        1,
        || Ok(MockSolver::with_fixed(100.0)),
        &InflowNonNegativityMethod::None,
        &[],
        0,
    )
    .unwrap();

    assert!(
        result.iterations <= 10,
        "iterations must be <= 10, got {}",
        result.iterations
    );
    assert!(
        result.final_lb >= 0.0,
        "final_lb must be >= 0.0, got {}",
        result.final_lb
    );
    assert!(
        result.final_ub >= 0.0,
        "final_ub must be >= 0.0, got {}",
        result.final_ub
    );
    assert!(
        result.final_gap >= 0.0 || result.final_gap < 0.0,
        "final_gap is a real number"
    );
    assert!(
        !result.reason.is_empty(),
        "reason must be a non-empty string"
    );
}

/// Run `train` twice with identical configuration and verify bit-for-bit
/// identical bounds and identical iteration counts.
#[test]
fn train_deterministic_with_same_seed() {
    let fx = Fixture::new(2);

    let mut fcf1 = make_fcf(fx.n_stages);
    let mut solver1 = MockSolver::with_fixed(50.0);
    let comm = StubComm;

    let result1 = train(
        &mut solver1,
        TrainingConfig {
            forward_passes: 1,
            max_iterations: 10,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
        },
        &mut fcf1,
        &fx.templates,
        &fx.base_rows,
        &fx.indexer,
        &fx.initial_state,
        &fx.opening_tree,
        &fx.stochastic,
        &fx.horizon,
        &fx.risk_measures,
        iteration_limit(5),
        None,
        None,
        &comm,
        1,
        || Ok(MockSolver::with_fixed(50.0)),
        &InflowNonNegativityMethod::None,
        &[],
        0,
    )
    .unwrap();

    let mut fcf2 = make_fcf(fx.n_stages);
    let mut solver2 = MockSolver::with_fixed(50.0);
    let opening_tree2 = make_opening_tree(1);
    let stochastic2 = make_stochastic_context(fx.n_stages, 1);

    let result2 = train(
        &mut solver2,
        TrainingConfig {
            forward_passes: 1,
            max_iterations: 10,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
        },
        &mut fcf2,
        &fx.templates,
        &fx.base_rows,
        &fx.indexer,
        &fx.initial_state,
        &opening_tree2,
        &stochastic2,
        &fx.horizon,
        &fx.risk_measures,
        iteration_limit(5),
        None,
        None,
        &comm,
        1,
        || Ok(MockSolver::with_fixed(50.0)),
        &InflowNonNegativityMethod::None,
        &[],
        0,
    )
    .unwrap();

    assert_eq!(
        result1.final_lb.to_bits(),
        result2.final_lb.to_bits(),
        "final_lb must be bit-for-bit identical: {} vs {}",
        result1.final_lb,
        result2.final_lb
    );
    assert_eq!(
        result1.final_ub.to_bits(),
        result2.final_ub.to_bits(),
        "final_ub must be bit-for-bit identical: {} vs {}",
        result1.final_ub,
        result2.final_ub
    );
    assert_eq!(
        result1.iterations, result2.iterations,
        "iteration count must be identical"
    );
}

/// Verify `lb[k] >= lb[k-1]` for all consecutive iterations from
/// `ConvergenceUpdate` events.
#[test]
fn train_lb_monotonically_nondecreasing() {
    let fx = Fixture::new(2);
    let mut fcf = make_fcf(fx.n_stages);
    // Solver returns a fixed objective so LB stays constant (non-decreasing
    // trivially). The key property is that it never decreases.
    let mut solver = MockSolver::with_fixed(80.0);
    let comm = StubComm;

    let (tx, rx) = mpsc::channel::<TrainingEvent>();
    let config = TrainingConfig {
        forward_passes: 1,
        max_iterations: 20,
        checkpoint_interval: None,
        warm_start_cuts: 0,
        event_sender: Some(tx),
    };

    train(
        &mut solver,
        config,
        &mut fcf,
        &fx.templates,
        &fx.base_rows,
        &fx.indexer,
        &fx.initial_state,
        &fx.opening_tree,
        &fx.stochastic,
        &fx.horizon,
        &fx.risk_measures,
        iteration_limit(6),
        None,
        None,
        &comm,
        1,
        || Ok(MockSolver::with_fixed(100.0)),
        &InflowNonNegativityMethod::None,
        &[],
        0,
    )
    .unwrap();

    // Collect all ConvergenceUpdate events and extract lower bounds.
    let events: Vec<TrainingEvent> = rx.try_iter().collect();
    let lower_bounds: Vec<f64> = events
        .iter()
        .filter_map(|e| {
            if let TrainingEvent::ConvergenceUpdate { lower_bound, .. } = e {
                Some(*lower_bound)
            } else {
                None
            }
        })
        .collect();

    assert!(
        lower_bounds.len() >= 5,
        "expected at least 5 ConvergenceUpdate events, got {}",
        lower_bounds.len()
    );

    for window in lower_bounds.windows(2) {
        let (prev, curr) = (window[0], window[1]);
        assert!(
            curr >= prev,
            "LB must be non-decreasing: lb[k]={curr} < lb[k-1]={prev}"
        );
    }
}

/// Verify the exact event sequence emitted by `train` for 3 iterations:
/// 1 `TrainingStarted` + 3 * 6 per-iteration events + 1 `TrainingFinished` = 20 total.
#[test]
fn train_emits_correct_event_sequence() {
    let fx = Fixture::new(2);
    let mut fcf = make_fcf(fx.n_stages);
    let mut solver = MockSolver::with_fixed(100.0);
    let comm = StubComm;

    let (tx, rx) = mpsc::channel::<TrainingEvent>();
    let config = TrainingConfig {
        forward_passes: 1,
        max_iterations: 10,
        checkpoint_interval: None,
        warm_start_cuts: 0,
        event_sender: Some(tx),
    };

    train(
        &mut solver,
        config,
        &mut fcf,
        &fx.templates,
        &fx.base_rows,
        &fx.indexer,
        &fx.initial_state,
        &fx.opening_tree,
        &fx.stochastic,
        &fx.horizon,
        &fx.risk_measures,
        // Limit to exactly 3 iterations.
        iteration_limit(3),
        None,
        None,
        &comm,
        1,
        || Ok(MockSolver::with_fixed(100.0)),
        &InflowNonNegativityMethod::None,
        &[],
        0,
    )
    .unwrap();

    let events: Vec<TrainingEvent> = rx.try_iter().collect();

    assert_eq!(
        events.len(),
        20,
        "expected 20 events (1 started + 3*6 per-iter + 1 finished), got {} events: {events:?}",
        events.len()
    );

    assert!(
        matches!(events[0], TrainingEvent::TrainingStarted { .. }),
        "events[0] must be TrainingStarted, got {:?}",
        events[0]
    );

    assert!(
        matches!(events[19], TrainingEvent::TrainingFinished { .. }),
        "events[19] must be TrainingFinished, got {:?}",
        events[19]
    );

    let per_iter_types: &[fn(&TrainingEvent) -> bool] = &[
        |e| matches!(e, TrainingEvent::ForwardPassComplete { .. }),
        |e| matches!(e, TrainingEvent::ForwardSyncComplete { .. }),
        |e| matches!(e, TrainingEvent::BackwardPassComplete { .. }),
        |e| matches!(e, TrainingEvent::CutSyncComplete { .. }),
        |e| matches!(e, TrainingEvent::ConvergenceUpdate { .. }),
        |e| matches!(e, TrainingEvent::IterationSummary { .. }),
    ];

    for iter_idx in 0..3usize {
        let offset = 1 + iter_idx * 6;
        for (step, &check_fn) in per_iter_types.iter().enumerate() {
            let event = &events[offset + step];
            assert!(
                check_fn(event),
                "iteration {}, step {}: unexpected event {:?}",
                iter_idx + 1,
                step,
                event
            );
        }
    }
}

/// Verify `train` terminates at `IterationLimit { limit: 3 }` and reports
/// `reason == "iteration_limit"`.
#[test]
fn train_stops_at_iteration_limit() {
    let fx = Fixture::new(2);
    let mut fcf = make_fcf(fx.n_stages);
    let mut solver = MockSolver::with_fixed(100.0);
    let comm = StubComm;

    let result = train(
        &mut solver,
        TrainingConfig {
            forward_passes: 1,
            max_iterations: 10,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
        },
        &mut fcf,
        &fx.templates,
        &fx.base_rows,
        &fx.indexer,
        &fx.initial_state,
        &fx.opening_tree,
        &fx.stochastic,
        &fx.horizon,
        &fx.risk_measures,
        iteration_limit(3),
        None,
        None,
        &comm,
        1,
        || Ok(MockSolver::with_fixed(100.0)),
        &InflowNonNegativityMethod::None,
        &[],
        0,
    )
    .unwrap();

    assert_eq!(
        result.iterations, 3,
        "expected exactly 3 iterations, got {}",
        result.iterations
    );
    assert_eq!(
        result.reason, "iteration_limit",
        "expected reason 'iteration_limit', got '{}'",
        result.reason
    );
}

/// Verify `train` terminates with `reason == "graceful_shutdown"` when an
/// external shutdown flag is set.
#[test]
fn train_stops_on_graceful_shutdown() {
    let fx = Fixture::new(2);
    let mut fcf = make_fcf(fx.n_stages);
    let mut solver = MockSolver::with_fixed(100.0);

    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let comm = ShutdownComm::new(Arc::clone(&shutdown_flag));

    let rules = StoppingRuleSet {
        rules: vec![
            StoppingRule::GracefulShutdown,
            StoppingRule::IterationLimit { limit: 20 },
        ],
        mode: StoppingMode::Any,
    };

    let result = train(
        &mut solver,
        TrainingConfig {
            forward_passes: 1,
            max_iterations: 20,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
        },
        &mut fcf,
        &fx.templates,
        &fx.base_rows,
        &fx.indexer,
        &fx.initial_state,
        &fx.opening_tree,
        &fx.stochastic,
        &fx.horizon,
        &fx.risk_measures,
        rules,
        None,
        Some(&shutdown_flag),
        &comm,
        1,
        || Ok(MockSolver::with_fixed(100.0)),
        &InflowNonNegativityMethod::None,
        &[],
        0,
    )
    .unwrap();

    assert_eq!(
        result.reason, "graceful_shutdown",
        "expected reason 'graceful_shutdown', got '{}' after {} iterations",
        result.reason, result.iterations
    );
    assert!(
        result.iterations <= 2,
        "expected at most 2 iterations before shutdown, got {}",
        result.iterations
    );
}

/// Verify `train` propagates `SddpError::Infeasible` when the solver returns
/// `SolverError::Infeasible` on the first forward-pass solve.
#[test]
fn train_propagates_infeasible_error() {
    let fx = Fixture::new(2);
    let mut fcf = make_fcf(fx.n_stages);
    let mut solver = MockSolver::infeasible_on_first();
    let comm = StubComm;

    let result = train(
        &mut solver,
        TrainingConfig {
            forward_passes: 1,
            max_iterations: 10,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
        },
        &mut fcf,
        &fx.templates,
        &fx.base_rows,
        &fx.indexer,
        &fx.initial_state,
        &fx.opening_tree,
        &fx.stochastic,
        &fx.horizon,
        &fx.risk_measures,
        iteration_limit(10),
        None,
        None,
        &comm,
        1,
        || Ok(MockSolver::infeasible_on_first()),
        &InflowNonNegativityMethod::None,
        &[],
        0,
    );

    assert!(
        matches!(result, Err(SddpError::Infeasible { stage: 0, .. })),
        "expected Err(SddpError::Infeasible {{ stage: 0, .. }}), got: {result:?}"
    );
}
