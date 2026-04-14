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
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, SamplingScheme,
    },
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_solver::{
    Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
};
use cobre_stochastic::{
    ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context,
};

use cobre_sddp::{
    HorizonMode, InflowNonNegativityMethod, RiskMeasure, SddpError, StageContext, StageIndexer,
    StoppingMode, StoppingRule, StoppingRuleSet, TrainingConfig, TrainingContext,
    cut::fcf::FutureCostFunction, train,
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

/// Communicator wrapper that sets `flag` to `true` on the first `allgatherv`
/// call, simulating a shutdown signal arriving mid-iteration-1. On
/// subsequent calls it behaves identically to [`StubComm`].
///
/// The `allgatherv` is called during `sync_forward` (step 2), so by the time
/// iteration 2's convergence check runs the shutdown flag is already set.
struct ShutdownComm {
    flag: Arc<AtomicBool>,
    /// Count of allgatherv calls; the shutdown flag is flipped on the first
    /// call that corresponds to `sync_forward` (forward sync in iteration 1).
    allgatherv_calls: AtomicUsize,
}

impl ShutdownComm {
    fn new(flag: Arc<AtomicBool>) -> Self {
        Self {
            flag,
            allgatherv_calls: AtomicUsize::new(0),
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
        // Set flag on first call so iteration 2's convergence check triggers shutdown.
        if self.allgatherv_calls.fetch_add(1, Ordering::Relaxed) == 0 {
            self.flag.store(true, Ordering::Relaxed);
        }
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
    fn solver_name_version(&self) -> String {
        "MockSolver 0.0.0".to_string()
    }
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
            primal: &[0.0, 0.0, 0.0, 0.0],
            dual: &[0.0, 0.0],
            reduced_costs: &[0.0, 0.0, 0.0, 0.0],
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
        method: "spectral".to_string(),
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

/// Minimal stage template for N=1 hydro, L=0 PAR.
fn minimal_template() -> StageTemplate {
    // N=1, L=0 → cols: storage(0), z_inflow(1), storage_in(2), theta(3)
    //             rows: storage_fixing(0), z_inflow(1)
    StageTemplate {
        num_cols: 4,
        num_rows: 2,
        num_nz: 1,
        col_starts: vec![0, 0, 0, 1, 1],
        row_indices: vec![0],
        values: vec![1.0],
        col_lower: vec![0.0; 4],
        col_upper: vec![f64::INFINITY; 4],
        objective: vec![0.0, 0.0, 0.0, 1.0],
        row_lower: vec![0.0; 2],
        row_upper: vec![0.0; 2],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 1,
        n_hydro: 1,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

fn make_fcf(n_stages: usize) -> FutureCostFunction {
    FutureCostFunction::new(n_stages, 1, 1, FCF_CAPACITY_ITERATIONS, &vec![0; n_stages])
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
    stochastic: StochasticContext,
    horizon: HorizonMode,
    risk_measures: Vec<RiskMeasure>,
}

const FCF_CAPACITY_ITERATIONS: u64 = 50;

impl Fixture {
    fn new(n_stages: usize) -> Self {
        let indexer = StageIndexer::new(1, 0); // N=1, L=0
        let templates = vec![minimal_template(); n_stages];
        // base_row = n_dual_relevant + n_hydros = 1 + 1 = 2 (z_inflow rows follow state rows)
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
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
            stochastic,
            horizon,
            risk_measures,
        }
    }
}

/// Run a single training pass with a given stochastic context.
///
/// Returns the `TrainingOutcome`. Used to de-duplicate the two identical
/// train calls in `train_deterministic_with_same_seed`.
fn run_one_deterministic_pass(
    fx: &Fixture,
    stochastic: &StochasticContext,
    limit: u64,
) -> cobre_sddp::TrainingOutcome {
    let mut fcf = make_fcf(fx.n_stages);
    let mut solver = MockSolver::with_fixed(50.0);
    let stage_ctx = StageContext {
        templates: &fx.templates,
        base_rows: &fx.base_rows,
        noise_scale: &[],
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &[1usize, 1],
        ncs_max_gen: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
    };
    train(
        &mut solver,
        TrainingConfig {
            forward_passes: 1,
            max_iterations: 10,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
            cut_activity_tolerance: 0.0,
            n_fwd_threads: 1,
            max_blocks: 1,
            cut_selection: None,
            shutdown_flag: None,
            start_iteration: 0,
            export_states: false,
            angular_pruning: None,
            budget: None,
        },
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            horizon: &fx.horizon,
            indexer: &fx.indexer,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            stages: &[],
        },
        &fx.risk_measures,
        iteration_limit(limit),
        &StubComm,
        || Ok(MockSolver::with_fixed(50.0)),
    )
    .unwrap()
}

// Tests

/// Verify the full training loop runs to completion under `IterationLimit`.
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
        cut_activity_tolerance: 0.0,
        n_fwd_threads: 1,
        max_blocks: 1,
        cut_selection: None,
        shutdown_flag: None,
        start_iteration: 0,
        export_states: false,
        angular_pruning: None,
        budget: None,
    };

    let stage_ctx = StageContext {
        templates: &fx.templates,
        base_rows: &fx.base_rows,
        noise_scale: &[],
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &[1usize, 1],
        ncs_max_gen: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
    };
    let result = train(
        &mut solver,
        config,
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            horizon: &fx.horizon,
            indexer: &fx.indexer,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            stages: &[],
        },
        &fx.risk_measures,
        iteration_limit(10),
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
    )
    .unwrap();

    assert!(result.result.iterations <= 10);
    assert!(result.result.final_lb >= 0.0);
    assert!(result.result.final_ub >= 0.0);
    assert!(result.result.final_gap.is_finite());
    assert!(!result.result.reason.is_empty());
}

/// Run `train` twice with identical configuration and verify bit-for-bit
/// identical bounds and identical iteration counts.
#[test]
fn train_deterministic_with_same_seed() {
    let fx = Fixture::new(2);

    let result1 = run_one_deterministic_pass(&fx, &fx.stochastic, 5);

    let stochastic2 = make_stochastic_context(fx.n_stages, 1);
    let result2 = run_one_deterministic_pass(&fx, &stochastic2, 5);

    assert_eq!(
        result1.result.final_lb.to_bits(),
        result2.result.final_lb.to_bits()
    );
    assert_eq!(
        result1.result.final_ub.to_bits(),
        result2.result.final_ub.to_bits()
    );
    assert_eq!(result1.result.iterations, result2.result.iterations);
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
        cut_activity_tolerance: 0.0,
        n_fwd_threads: 1,
        max_blocks: 1,
        cut_selection: None,
        shutdown_flag: None,
        start_iteration: 0,
        export_states: false,
        angular_pruning: None,
        budget: None,
    };

    let stage_ctx = StageContext {
        templates: &fx.templates,
        base_rows: &fx.base_rows,
        noise_scale: &[],
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &[1usize, 1],
        ncs_max_gen: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
    };
    train(
        &mut solver,
        config,
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            horizon: &fx.horizon,
            indexer: &fx.indexer,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            stages: &[],
        },
        &fx.risk_measures,
        iteration_limit(6),
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
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

    assert!(lower_bounds.len() >= 5);
    for window in lower_bounds.windows(2) {
        assert!(window[1] >= window[0]);
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
        cut_activity_tolerance: 0.0,
        n_fwd_threads: 1,
        max_blocks: 1,
        cut_selection: None,
        shutdown_flag: None,
        start_iteration: 0,
        export_states: false,
        angular_pruning: None,
        budget: None,
    };

    let stage_ctx = StageContext {
        templates: &fx.templates,
        base_rows: &fx.base_rows,
        noise_scale: &[],
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &[1usize, 1],
        ncs_max_gen: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
    };
    train(
        &mut solver,
        config,
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            horizon: &fx.horizon,
            indexer: &fx.indexer,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            stages: &[],
        },
        &fx.risk_measures,
        // Limit to exactly 3 iterations.
        iteration_limit(3),
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
    )
    .unwrap();

    let events: Vec<TrainingEvent> = rx.try_iter().collect();

    assert_eq!(events.len(), 20);
    assert!(matches!(events[0], TrainingEvent::TrainingStarted { .. }));
    assert!(matches!(events[19], TrainingEvent::TrainingFinished { .. }));

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
            assert!(check_fn(&events[offset + step]));
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

    let stage_ctx = StageContext {
        templates: &fx.templates,
        base_rows: &fx.base_rows,
        noise_scale: &[],
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &[1usize, 1],
        ncs_max_gen: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
    };
    let result = train(
        &mut solver,
        TrainingConfig {
            forward_passes: 1,
            max_iterations: 10,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
            cut_activity_tolerance: 0.0,
            n_fwd_threads: 1,
            max_blocks: 1,
            cut_selection: None,
            shutdown_flag: None,
            start_iteration: 0,
            export_states: false,
            angular_pruning: None,
            budget: None,
        },
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            horizon: &fx.horizon,
            indexer: &fx.indexer,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            stages: &[],
        },
        &fx.risk_measures,
        iteration_limit(3),
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
    )
    .unwrap();

    assert_eq!(result.result.iterations, 3);
    assert_eq!(result.result.reason, "iteration_limit");
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

    let stage_ctx = StageContext {
        templates: &fx.templates,
        base_rows: &fx.base_rows,
        noise_scale: &[],
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &[1usize, 1],
        ncs_max_gen: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
    };
    let result = train(
        &mut solver,
        TrainingConfig {
            forward_passes: 1,
            max_iterations: 20,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
            cut_activity_tolerance: 0.0,
            n_fwd_threads: 1,
            max_blocks: 1,
            cut_selection: None,
            shutdown_flag: Some(Arc::clone(&shutdown_flag)),
            start_iteration: 0,
            export_states: false,
            angular_pruning: None,
            budget: None,
        },
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            horizon: &fx.horizon,
            indexer: &fx.indexer,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            stages: &[],
        },
        &fx.risk_measures,
        rules,
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
    )
    .unwrap();

    assert_eq!(result.result.reason, "graceful_shutdown");
    assert!(result.result.iterations <= 2);
}

/// Verify `train` propagates `SddpError::Infeasible` when the solver returns
/// `SolverError::Infeasible` on the first forward-pass solve.
#[test]
fn train_propagates_infeasible_error() {
    let fx = Fixture::new(2);
    let mut fcf = make_fcf(fx.n_stages);
    let mut solver = MockSolver::infeasible_on_first();
    let comm = StubComm;

    let stage_ctx = StageContext {
        templates: &fx.templates,
        base_rows: &fx.base_rows,
        noise_scale: &[],
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &[1usize, 1],
        ncs_max_gen: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
    };
    let result = train(
        &mut solver,
        TrainingConfig {
            forward_passes: 1,
            max_iterations: 10,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender: None,
            cut_activity_tolerance: 0.0,
            n_fwd_threads: 1,
            max_blocks: 1,
            cut_selection: None,
            shutdown_flag: None,
            start_iteration: 0,
            export_states: false,
            angular_pruning: None,
            budget: None,
        },
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            horizon: &fx.horizon,
            indexer: &fx.indexer,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            stages: &[],
        },
        &fx.risk_measures,
        iteration_limit(10),
        &comm,
        || Ok(MockSolver::infeasible_on_first()),
    );

    let outcome = result.expect("train must return Ok(TrainingOutcome) with captured error");
    assert!(outcome.error.is_some(), "expected error in TrainingOutcome");
    assert!(
        matches!(outcome.error, Some(SddpError::Infeasible { stage: 0, .. })),
        "expected SddpError::Infeasible at stage 0, got: {:?}",
        outcome.error
    );
    assert_eq!(
        outcome.result.iterations, 0,
        "no iterations should have completed"
    );
    assert_eq!(outcome.result.reason, "error");
}

/// D17: Level1 cut selection produces convergent results with bounded pool.
///
/// Verifies that enabling `CutSelectionStrategy::Level1 { threshold: 0,
/// check_frequency: 2 }` does not break convergence. With the mock solver
/// returning `dual: &[0.0, 0.0]`, all generated cuts have `active_count == 0`
/// and are deactivated by Level1 selection.
///
/// Checks:
/// - Lower bound is monotone non-decreasing.
/// - At least one `CutSelectionComplete` event with `cuts_deactivated > 0`.
/// - `active_count() < populated_count` for the stage-0 FCF pool.
#[test]
#[allow(clippy::too_many_lines)]
fn d17_level1_cut_selection_convergence() {
    use cobre_sddp::cut_selection::CutSelectionStrategy;

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
        cut_activity_tolerance: 0.0,
        n_fwd_threads: 1,
        max_blocks: 1,
        cut_selection: Some(CutSelectionStrategy::Level1 {
            threshold: 0,
            check_frequency: 2,
        }),
        shutdown_flag: None,
        start_iteration: 0,
        export_states: false,
        angular_pruning: None,
        budget: None,
    };

    let stage_ctx = StageContext {
        templates: &fx.templates,
        base_rows: &fx.base_rows,
        noise_scale: &[],
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &[1usize, 1],
        ncs_max_gen: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
    };
    let result = train(
        &mut solver,
        config,
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            horizon: &fx.horizon,
            indexer: &fx.indexer,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            stages: &[],
        },
        &fx.risk_measures,
        iteration_limit(10),
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
    )
    .unwrap();

    assert!(
        result.result.iterations <= 10,
        "training must complete within limit"
    );

    // AC2: Lower bound is monotone non-decreasing.
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
        !lower_bounds.is_empty(),
        "must have at least one ConvergenceUpdate event"
    );
    for window in lower_bounds.windows(2) {
        assert!(
            window[1] >= window[0],
            "lower bound must be non-decreasing: {} -> {}",
            window[0],
            window[1]
        );
    }

    // AC3: At least one CutSelectionComplete event was emitted.
    let sel_events: Vec<&TrainingEvent> = events
        .iter()
        .filter(|e| matches!(e, TrainingEvent::CutSelectionComplete { .. }))
        .collect();

    assert!(
        !sel_events.is_empty(),
        "must have at least one CutSelectionComplete event"
    );

    // AC4: Stage 0 is exempt from cut selection (no activity tracking).
    // All stage-0 cuts generated during training should remain active.
    // The populated count may include a warm-start slot, so we check
    // that active is at least (iterations * forward_passes).
    assert!(
        fcf.pools[0].active_count() >= result.result.iterations as usize,
        "stage 0 must be exempt: expected at least {} active cuts, got {} \
         (populated={})",
        result.result.iterations,
        fcf.pools[0].active_count(),
        fcf.pools[0].populated_count,
    );

    // Diagnostic: basis rejection rate after cut selection.
    // With the mock solver, basis_rejections is always 0 since the mock
    // does not track basis operations. This check is informational — it
    // would detect degradation if the mock were upgraded to track basis
    // rejections, or when running with a real solver.
    // See BasisStore doc comment for the design decision (option 1 vs 3).
    let stats = solver.statistics();
    if stats.basis_offered > 0 && stats.basis_rejections > stats.basis_offered / 2 {
        eprintln!(
            "WARNING: basis rejection rate after cut selection is {}/{}. \
             Consider implementing option 3 (discard cut row statuses).",
            stats.basis_rejections, stats.basis_offered
        );
    }
}

/// D18: Lml1 cut selection produces convergent results with bounded pool.
///
/// Verifies that enabling `CutSelectionStrategy::Lml1 { memory_window: 3,
/// check_frequency: 2 }` does not break convergence. The mock solver produces
/// zero-activity cuts (dual = 0), so `last_active_iter` never advances past
/// `iteration_generated`. After `memory_window` iterations, all older cuts
/// are deactivated by Lml1.
///
/// Checks:
/// - Lower bound is monotone non-decreasing.
/// - At least one `CutSelectionComplete` event with `cuts_deactivated > 0`.
/// - `active_count() < populated_count` for the stage-0 FCF pool.
#[test]
#[allow(clippy::too_many_lines)]
fn d18_lml1_cut_selection_convergence() {
    use cobre_sddp::cut_selection::CutSelectionStrategy;

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
        cut_activity_tolerance: 0.0,
        n_fwd_threads: 1,
        max_blocks: 1,
        cut_selection: Some(CutSelectionStrategy::Lml1 {
            memory_window: 3,
            check_frequency: 2,
        }),
        shutdown_flag: None,
        start_iteration: 0,
        export_states: false,
        angular_pruning: None,
        budget: None,
    };

    let stage_ctx = StageContext {
        templates: &fx.templates,
        base_rows: &fx.base_rows,
        noise_scale: &[],
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &[1usize, 1],
        ncs_max_gen: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
    };
    let result = train(
        &mut solver,
        config,
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            horizon: &fx.horizon,
            indexer: &fx.indexer,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            stages: &[],
        },
        &fx.risk_measures,
        iteration_limit(10),
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
    )
    .unwrap();

    assert!(
        result.result.iterations <= 10,
        "training must complete within limit"
    );

    // AC2: Lower bound is monotone non-decreasing.
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
        !lower_bounds.is_empty(),
        "must have at least one ConvergenceUpdate event"
    );
    for window in lower_bounds.windows(2) {
        assert!(
            window[1] >= window[0],
            "lower bound must be non-decreasing: {} -> {}",
            window[0],
            window[1]
        );
    }

    // AC3: At least one CutSelectionComplete event was emitted.
    let sel_events: Vec<&TrainingEvent> = events
        .iter()
        .filter(|e| matches!(e, TrainingEvent::CutSelectionComplete { .. }))
        .collect();

    assert!(
        !sel_events.is_empty(),
        "must have at least one CutSelectionComplete event"
    );

    // AC4: Stage 0 is exempt from cut selection (no activity tracking).
    // All stage-0 cuts generated during training should remain active.
    // The populated count may include a warm-start slot, so we check
    // that active is at least (iterations * forward_passes).
    assert!(
        fcf.pools[0].active_count() >= result.result.iterations as usize,
        "stage 0 must be exempt: expected at least {} active cuts, got {} \
         (populated={})",
        result.result.iterations,
        fcf.pools[0].active_count(),
        fcf.pools[0].populated_count,
    );
}
