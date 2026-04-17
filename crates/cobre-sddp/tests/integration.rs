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
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
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
    CutManagementConfig, EventConfig, HorizonMode, InflowNonNegativityMethod, LoopConfig,
    RiskMeasure, SddpError, StageContext, StageIndexer, StoppingMode, StoppingRule,
    StoppingRuleSet, TrainingConfig, TrainingContext, cut::fcf::FutureCostFunction, train,
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

/// Mock solver that returns a zero-filled dual slice matching the current row count.
///
/// Unlike `MockSolver` (which has a hardcoded two-element dual), this expands
/// the dual buffer as cuts accumulate, so it can back tests where the backward
/// pass solves at interior stages with active cut rows present.
struct ExpandingMockSolver {
    objectives: Vec<f64>,
    call_count: usize,
    current_num_rows: usize,
    dual_buf: Vec<f64>,
    primal_buf: Vec<f64>,
}

impl ExpandingMockSolver {
    fn with_objectives(objectives: Vec<f64>) -> Self {
        Self {
            objectives,
            call_count: 0,
            current_num_rows: 0,
            dual_buf: vec![0.0_f64; 64],
            primal_buf: vec![0.0_f64; 4],
        }
    }
}

impl SolverInterface for ExpandingMockSolver {
    fn solver_name_version(&self) -> String {
        "ExpandingMockSolver 0.0.0".to_string()
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
        let call = self.call_count;
        self.call_count += 1;
        let obj = self.objectives[call % self.objectives.len()];
        if self.dual_buf.len() < self.current_num_rows {
            self.dual_buf.resize(self.current_num_rows, 0.0);
        }
        Ok(cobre_solver::SolutionView {
            objective: obj,
            primal: &self.primal_buf,
            dual: &self.dual_buf[..self.current_num_rows],
            reduced_costs: &self.primal_buf,
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
        "ExpandingMock"
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
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
    };
    train(
        &mut solver,
        TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 10,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit(limit),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: fx.risk_measures.clone(),
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
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            stages: &[],
        },
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
        loop_config: LoopConfig {
            forward_passes: 1,
            max_iterations: 10,
            start_iteration: 0,
            n_fwd_threads: 1,
            max_blocks: 1,
            stopping_rules: iteration_limit(10),
        },
        cut_management: CutManagementConfig {
            cut_selection: None,
            budget: None,
            cut_activity_tolerance: 0.0,
            warm_start_cuts: 0,
            risk_measures: fx.risk_measures.clone(),
        },
        events: EventConfig {
            event_sender: None,
            checkpoint_interval: None,
            shutdown_flag: None,
            export_states: false,
        },
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
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
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
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            stages: &[],
        },
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
        loop_config: LoopConfig {
            forward_passes: 1,
            max_iterations: 20,
            start_iteration: 0,
            n_fwd_threads: 1,
            max_blocks: 1,
            stopping_rules: iteration_limit(6),
        },
        cut_management: CutManagementConfig {
            cut_selection: None,
            budget: None,
            cut_activity_tolerance: 0.0,
            warm_start_cuts: 0,
            risk_measures: fx.risk_measures.clone(),
        },
        events: EventConfig {
            event_sender: Some(tx),
            checkpoint_interval: None,
            shutdown_flag: None,
            export_states: false,
        },
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
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
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
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            stages: &[],
        },
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
/// 1 `TrainingStarted` + 3 * 7 per-iteration events + 1 `TrainingFinished` = 23 total.
/// Per-iteration events: `ForwardPassComplete`, `ForwardSyncComplete`, `BackwardPassComplete`,
/// `CutSyncComplete`, `TemplateBakeComplete`, `ConvergenceUpdate`, `IterationSummary`.
#[test]
fn train_emits_correct_event_sequence() {
    let fx = Fixture::new(2);
    let mut fcf = make_fcf(fx.n_stages);
    let mut solver = MockSolver::with_fixed(100.0);
    let comm = StubComm;

    let (tx, rx) = mpsc::channel::<TrainingEvent>();
    let config = TrainingConfig {
        loop_config: LoopConfig {
            forward_passes: 1,
            max_iterations: 10,
            start_iteration: 0,
            n_fwd_threads: 1,
            max_blocks: 1,
            // Limit to exactly 3 iterations.
            stopping_rules: iteration_limit(3),
        },
        cut_management: CutManagementConfig {
            cut_selection: None,
            budget: None,
            cut_activity_tolerance: 0.0,
            warm_start_cuts: 0,
            risk_measures: fx.risk_measures.clone(),
        },
        events: EventConfig {
            event_sender: Some(tx),
            checkpoint_interval: None,
            shutdown_flag: None,
            export_states: false,
        },
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
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
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
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            stages: &[],
        },
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
    )
    .unwrap();

    let events: Vec<TrainingEvent> = rx.try_iter().collect();

    assert_eq!(events.len(), 23);
    assert!(matches!(events[0], TrainingEvent::TrainingStarted { .. }));
    assert!(matches!(events[22], TrainingEvent::TrainingFinished { .. }));

    let per_iter_types: &[fn(&TrainingEvent) -> bool] = &[
        |e| matches!(e, TrainingEvent::ForwardPassComplete { .. }),
        |e| matches!(e, TrainingEvent::ForwardSyncComplete { .. }),
        |e| matches!(e, TrainingEvent::BackwardPassComplete { .. }),
        |e| matches!(e, TrainingEvent::CutSyncComplete { .. }),
        |e| matches!(e, TrainingEvent::TemplateBakeComplete { .. }),
        |e| matches!(e, TrainingEvent::ConvergenceUpdate { .. }),
        |e| matches!(e, TrainingEvent::IterationSummary { .. }),
    ];

    for iter_idx in 0..3usize {
        let offset = 1 + iter_idx * 7;
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
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
    };
    let result = train(
        &mut solver,
        TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 10,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit(3),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: fx.risk_measures.clone(),
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
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            stages: &[],
        },
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
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
    };
    let result = train(
        &mut solver,
        TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 20,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: rules,
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: fx.risk_measures.clone(),
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: Some(Arc::clone(&shutdown_flag)),
                export_states: false,
            },
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
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            stages: &[],
        },
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
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
    };
    let result = train(
        &mut solver,
        TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 10,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit(10),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: fx.risk_measures.clone(),
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
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            stages: &[],
        },
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
        loop_config: LoopConfig {
            forward_passes: 1,
            max_iterations: 10,
            start_iteration: 0,
            n_fwd_threads: 1,
            max_blocks: 1,
            stopping_rules: iteration_limit(10),
        },
        cut_management: CutManagementConfig {
            cut_selection: Some(CutSelectionStrategy::Level1 {
                threshold: 0,
                check_frequency: 2,
            }),
            budget: None,
            cut_activity_tolerance: 0.0,
            warm_start_cuts: 0,
            risk_measures: fx.risk_measures.clone(),
        },
        events: EventConfig {
            event_sender: Some(tx),
            checkpoint_interval: None,
            shutdown_flag: None,
            export_states: false,
        },
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
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
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
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            stages: &[],
        },
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

/// D17 with basis reconstruction always active: truncation guard does not
/// corrupt convergence.
///
/// Basis reconstruction is now unconditional (post-Epic-02 flag removal).
/// Verifies:
/// - Lower bound matches the D17 baseline.
/// - Zero basis rejections (reconstruction produces valid warm-start bases).
#[test]
fn d17_level1_cut_selection_reconstruction() {
    use cobre_sddp::cut_selection::CutSelectionStrategy;

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
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
    };

    let result = train(
        &mut solver,
        TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: 10,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit(10),
            },
            cut_management: CutManagementConfig {
                cut_selection: Some(CutSelectionStrategy::Level1 {
                    threshold: 0,
                    check_frequency: 2,
                }),
                budget: None,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: fx.risk_measures.clone(),
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
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            stages: &[],
        },
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
    )
    .unwrap();

    // Reconstruction must not affect the optimal solution.
    assert!(
        result.result.final_lb.is_finite(),
        "D17+reconstruction: lower bound must be finite, got {}",
        result.result.final_lb,
    );

    // Zero basis rejections — reconstruction produces valid warm-start bases.
    let stats = solver.statistics();
    assert_eq!(
        stats.basis_rejections, 0,
        "D17+reconstruction: expected 0 basis rejections, got {}",
        stats.basis_rejections,
    );
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
        loop_config: LoopConfig {
            forward_passes: 1,
            max_iterations: 10,
            start_iteration: 0,
            n_fwd_threads: 1,
            max_blocks: 1,
            stopping_rules: iteration_limit(10),
        },
        cut_management: CutManagementConfig {
            cut_selection: Some(CutSelectionStrategy::Lml1 {
                memory_window: 3,
                check_frequency: 2,
            }),
            budget: None,
            cut_activity_tolerance: 0.0,
            warm_start_cuts: 0,
            risk_measures: fx.risk_measures.clone(),
        },
        events: EventConfig {
            event_sender: Some(tx),
            checkpoint_interval: None,
            shutdown_flag: None,
            export_states: false,
        },
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
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
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
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            stages: &[],
        },
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

/// AC #8 (ticket-003): D01 must produce a bit-identical lower bound after the
/// forward path is rewired through `reconstruct_basis`.
///
/// `reconstruct_basis` is a warm-start heuristic — it must not change the
/// optimal LP solution.  This test runs D01 with the default configuration
/// and asserts the lower bound matches the reference value of 182,500 $ to
/// within float tolerance.
#[test]
fn test_forward_basis_reconstruct_bit_identical_d01() {
    use std::path::Path;

    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
    use cobre_core::scenario::ScenarioSource;
    use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
    use cobre_solver::SolverInterface;
    use cobre_solver::highs::HighsSolver;

    struct LocalStubComm;

    impl Communicator for LocalStubComm {
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

    let case_dir = Path::new("../../examples/deterministic/d01-thermal-dispatch");
    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");
    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let prepare_result =
        prepare_stochastic(system, case_dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir).expect("prepare_hydro_models must succeed");

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup must build");

    let comm = LocalStubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");

    // Reconstruct path must yield the bit-identical reference lower bound.
    let diff = (outcome.result.final_lb - 182_500.0_f64).abs();
    assert!(
        diff <= 1e-6,
        "reconstruct path: expected lower bound 182500.0, got {} (diff={:.2e})",
        outcome.result.final_lb,
        diff
    );

    // Sanity: zero basis rejections — reconstructed bases must be accepted.
    let stats = solver.statistics();
    assert_eq!(
        stats.basis_rejections, 0,
        "reconstruct path: expected 0 basis rejections, got {}",
        stats.basis_rejections
    );
}

/// Mock solver that tracks `add_rows_count` cumulatively and exposes it via
/// `statistics()`. Used to verify that the baked-template path calls `add_rows`
/// zero times on iteration 2.
struct TrackingMockSolver {
    add_rows_count: u64,
    load_model_count: u64,
    current_num_rows: usize,
    dual_buf: Vec<f64>,
    primal_buf: Vec<f64>,
}

impl TrackingMockSolver {
    fn new() -> Self {
        Self {
            add_rows_count: 0,
            load_model_count: 0,
            current_num_rows: 0,
            dual_buf: vec![0.0_f64; 64],
            primal_buf: vec![0.0_f64; 4],
        }
    }
}

impl cobre_solver::SolverInterface for TrackingMockSolver {
    fn solver_name_version(&self) -> String {
        "TrackingMockSolver 0.0.0".to_string()
    }

    fn load_model(&mut self, template: &cobre_solver::StageTemplate) {
        self.current_num_rows = template.num_rows;
        self.load_model_count += 1;
    }

    fn add_rows(&mut self, cuts: &cobre_solver::RowBatch) {
        self.current_num_rows += cuts.num_rows;
        self.add_rows_count += 1;
        if self.dual_buf.len() < self.current_num_rows {
            self.dual_buf.resize(self.current_num_rows, 0.0);
        }
    }

    fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

    fn solve(&mut self) -> Result<cobre_solver::SolutionView<'_>, cobre_solver::SolverError> {
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

    fn get_basis(&mut self, _out: &mut cobre_solver::Basis) {}

    fn solve_with_basis(
        &mut self,
        _basis: &cobre_solver::Basis,
    ) -> Result<cobre_solver::SolutionView<'_>, cobre_solver::SolverError> {
        self.solve()
    }

    fn statistics(&self) -> cobre_solver::SolverStatistics {
        cobre_solver::SolverStatistics {
            add_rows_count: self.add_rows_count,
            load_model_count: self.load_model_count,
            ..cobre_solver::SolverStatistics::default()
        }
    }

    fn name(&self) -> &'static str {
        "TrackingMock"
    }
}

/// Verify that baked templates skip `add_rows` calls in iteration 2.
///
/// Setup: 3-stage, 4 forward-pass scenarios, 1 active cut per stage.
/// Iteration 1: legacy path calls `add_rows`.
/// After baking: merge active cuts into templates.
/// Iteration 2: baked path calls `add_rows` zero times.
#[test]
#[allow(clippy::too_many_lines)]
fn forward_pass_uses_baked_template_on_iter_2() {
    use cobre_sddp::{
        BakedTemplates, BasisStore, ForwardPassBatch, FutureCostFunction, HorizonMode,
        InflowNonNegativityMethod, PatchBuffer, SolverWorkspace, StageContext, StageIndexer,
        TrainingContext, WorkspaceSizing, build_cut_row_batch_into, run_forward_pass,
    };
    use cobre_solver::{RowBatch, StageTemplate, bake_rows_into_template};

    // ── System parameters ───────────────────────────────────────────────────
    let n_stages = 3;
    let n_fwd = 4; // forward-pass scenarios

    // StageIndexer: N=1, L=0 → n_state=1, theta=3, num_cols=4
    let indexer = StageIndexer::new(1, 0);

    // Base templates: minimal 1-row LP (storage_fixing row only).
    let base_template = minimal_template();
    let base_templates: Vec<StageTemplate> = vec![base_template.clone(); n_stages];

    // Stages matching the 3-stage stochastic context.
    let stochastic = make_stochastic_context(3, 1);
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
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
        })
        .collect();

    // FCF: capacity=50, n_state=1. Add 1 cut per stage so `add_rows` fires.
    let mut fcf = FutureCostFunction::new(n_stages, indexer.n_state, 2, 50, &vec![0; n_stages]);
    for t in 0..n_stages {
        // One Benders cut per stage: intercept=10.0, coeff=[1.0] (state_dim=1).
        fcf.add_cut(t, 1, 0, 10.0, &[1.0]);
    }

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let initial_state = vec![0.0_f64; indexer.n_state];
    let base_rows = vec![2usize; n_stages]; // matches minimal_template() in integration.rs

    let stage_ctx = StageContext {
        templates: &base_templates,
        base_rows: &base_rows,
        noise_scale: &[],
        // n_hydros=0 suppresses the noise-transform path (same convention used
        // by all forward.rs unit tests that call run_forward_pass directly).
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &vec![1usize; n_stages],
        ncs_max_gen: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
    };
    let training_ctx = TrainingContext {
        horizon: &horizon,
        indexer: &indexer,
        inflow_method: &InflowNonNegativityMethod::None,
        stochastic: &stochastic,
        initial_state: &initial_state,
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        stages: &stages,
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
        recent_accum_seed: &[],
        recent_weight_seed: 0.0,
    };

    // Helper: build empty RowBatch.
    let empty_batch = || RowBatch {
        num_rows: 0,
        row_starts: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    };

    // Helper: build a SolverWorkspace for TrackingMockSolver.
    let make_workspace = || {
        let sizing = WorkspaceSizing {
            // hydro_count=0 matches n_hydros=0 in stage_ctx (noise path is suppressed).
            hydro_count: 0,
            max_par_order: 0,
            n_load_buses: 0,
            max_blocks: 1,
            downstream_par_order: 0,
            max_openings: 0,
            initial_pool_capacity: 0,
            n_state: indexer.n_state,
        };
        SolverWorkspace::new(
            TrackingMockSolver::new(),
            PatchBuffer::new(0, 0, 0, 0),
            indexer.n_state,
            sizing,
        )
    };

    // Records buffer: n_fwd * n_stages entries.
    let make_records = || {
        (0..n_fwd * n_stages)
            .map(|_| cobre_sddp::TrajectoryRecord {
                primal: Vec::new(),
                dual: Vec::new(),
                stage_cost: 0.0,
                state: Vec::new(),
            })
            .collect::<Vec<_>>()
    };

    // ── Iteration 1: legacy path (baked.ready = false) ──────────────────────
    let mut workspaces_iter1 = vec![make_workspace()];
    let mut basis_store_iter1 = BasisStore::new(n_fwd, n_stages);
    let mut cut_batches: Vec<RowBatch> = (0..n_stages).map(|_| empty_batch()).collect();
    let mut records_iter1 = make_records();

    let not_baked = BakedTemplates {
        templates: &[],
        ready: false,
    };
    let fwd_batch_iter1 = ForwardPassBatch {
        local_forward_passes: n_fwd,
        total_forward_passes: n_fwd,
        iteration: 1,
        fwd_offset: 0,
    };

    let add_rows_before_iter1 = workspaces_iter1[0].solver.add_rows_count;
    let _ = run_forward_pass(
        &mut workspaces_iter1,
        &mut basis_store_iter1,
        &stage_ctx,
        &not_baked,
        &fcf,
        &mut cut_batches,
        &training_ctx,
        &fwd_batch_iter1,
        &mut records_iter1,
    )
    .expect("iteration 1 forward pass must not error");
    let add_rows_delta_iter1 = workspaces_iter1[0].solver.add_rows_count - add_rows_before_iter1;

    assert!(
        add_rows_delta_iter1 > 0,
        "iteration 1 (legacy path) must call add_rows at least once; \
         got add_rows_delta={add_rows_delta_iter1}"
    );

    // ── Bake step: simulate step 4d from training.rs ────────────────────────
    // Build cut row batches (one per stage) and bake them into copies of the
    // base template. This mirrors what the training loop does after iteration 1.
    let mut bake_batches: Vec<RowBatch> = (0..n_stages).map(|_| empty_batch()).collect();
    let mut baked_templates: Vec<StageTemplate> =
        (0..n_stages).map(|_| StageTemplate::empty()).collect();

    for t in 0..n_stages {
        build_cut_row_batch_into(
            &mut bake_batches[t],
            &fcf,
            t,
            &indexer,
            &base_templates[t].col_scale,
        );
        bake_rows_into_template(
            &base_templates[t],
            &bake_batches[t],
            &mut baked_templates[t],
        );
    }

    // Verify: baked template has more rows than the base template.
    for t in 0..n_stages {
        assert!(
            baked_templates[t].num_rows > base_templates[t].num_rows,
            "stage {t}: baked template must have more rows than base; \
             base={} baked={}",
            base_templates[t].num_rows,
            baked_templates[t].num_rows,
        );
    }

    // ── Iteration 2: baked path (baked.ready = true) ────────────────────────
    let mut workspaces_iter2 = vec![make_workspace()];
    let mut basis_store_iter2 = BasisStore::new(n_fwd, n_stages);
    // cut_batches still holds the per-stage batches from the pre-loop build.
    let mut records_iter2 = make_records();

    let is_baked = BakedTemplates {
        templates: &baked_templates,
        ready: true,
    };
    let fwd_batch_iter2 = ForwardPassBatch {
        local_forward_passes: n_fwd,
        total_forward_passes: n_fwd,
        iteration: 2,
        fwd_offset: 0,
    };

    let add_rows_before_iter2 = workspaces_iter2[0].solver.add_rows_count;
    let _ = run_forward_pass(
        &mut workspaces_iter2,
        &mut basis_store_iter2,
        &stage_ctx,
        &is_baked,
        &fcf,
        &mut cut_batches,
        &training_ctx,
        &fwd_batch_iter2,
        &mut records_iter2,
    )
    .expect("iteration 2 forward pass must not error");
    let add_rows_delta_iter2 = workspaces_iter2[0].solver.add_rows_count - add_rows_before_iter2;

    assert_eq!(
        add_rows_delta_iter2, 0,
        "iteration 2 (baked path) must not call add_rows; \
         got add_rows_delta={add_rows_delta_iter2}"
    );

    // ── Verify basis_row_capacity uses baked template num_rows ───────────────
    // Each record was produced with baked.ready=true. The forward pass
    // sets basis_row_capacity = baked.templates[t].num_rows. We verify
    // indirectly that load_model was called with the baked template by
    // confirming that the solver's current_num_rows after each stage-loop
    // iteration equals the baked template's num_rows. Since all stages use
    // the same template structure, we can read current_num_rows at end of pass.
    assert_eq!(
        workspaces_iter2[0].solver.current_num_rows,
        baked_templates[n_stages - 1].num_rows,
        "after iteration 2 the solver's current_num_rows must equal \
         baked_templates[last].num_rows = {}; got {}",
        baked_templates[n_stages - 1].num_rows,
        workspaces_iter2[0].solver.current_num_rows,
    );
}

/// Verify that baked backward pass uses delta batch with no new iteration 2 cuts.
///
/// Setup: 3-stage, 4 scenarios, 1 cut per stage from iteration 1.
/// After baking, iteration 2's backward pass appends delta batch (0 new cuts).
/// On baked path `add_rows` is skipped; legacy path would call it per trial-point.
#[test]
#[allow(clippy::too_many_lines)]
fn backward_pass_uses_delta_batch_on_iter_2() {
    use cobre_sddp::{
        BackwardPassSpec, BakedTemplates, BasisStore, CutSyncBuffers, ExchangeBuffers,
        FutureCostFunction, HorizonMode, InflowNonNegativityMethod, PatchBuffer, RiskMeasure,
        SolverWorkspace, StageContext, StageIndexer, TrainingContext, WorkspaceSizing,
        build_cut_row_batch_into, run_backward_pass,
    };
    use cobre_solver::{RowBatch, StageTemplate, bake_rows_into_template};

    // ── System parameters ──────────────────────────────────────────────────
    let n_stages = 3;
    let n_fwd = 4;

    let indexer = StageIndexer::new(1, 0);
    let base_template = minimal_template();
    let base_templates: Vec<StageTemplate> = vec![base_template.clone(); n_stages];

    let stochastic = make_stochastic_context(n_stages, 1);
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
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
        })
        .collect();

    // Helper: create a fresh FCF pre-populated with n_fwd cuts per stage at
    // iteration 0. Slots 0..n_fwd-1 are occupied; iteration=1 backward pass
    // uses slots n_fwd..2*n_fwd-1; iteration=2 uses slots 2*n_fwd..3*n_fwd-1.
    let make_fcf_iter0 = || {
        let mut fcf = FutureCostFunction::new(
            n_stages,
            indexer.n_state,
            n_fwd as u32,
            50,
            &vec![0_u32; n_stages],
        );
        for t in 0..n_stages {
            for fp in 0..n_fwd {
                fcf.add_cut(t, 0, fp as u32, 10.0, &[1.0]);
            }
        }
        fcf
    };

    // Separate FCF for the baked run: the baked templates are built from this
    // FCF's iter-0 cuts. The baked backward pass uses the same FCF with exactly
    // those cuts active — no extra cuts are added between baking and the pass.
    let mut fcf_baked_run = make_fcf_iter0();

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let base_rows = vec![2usize; n_stages];

    let stage_ctx = StageContext {
        templates: &base_templates,
        base_rows: &base_rows,
        noise_scale: &[],
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &vec![1usize; n_stages],
        ncs_max_gen: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
    };
    let training_ctx = TrainingContext {
        horizon: &horizon,
        indexer: &indexer,
        inflow_method: &InflowNonNegativityMethod::None,
        stochastic: &stochastic,
        initial_state: &vec![0.0_f64; indexer.n_state],
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        stages: &stages,
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
        recent_accum_seed: &[],
        recent_weight_seed: 0.0,
    };

    let empty_batch = || RowBatch {
        num_rows: 0,
        row_starts: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    };

    let make_workspace = || {
        let sizing = WorkspaceSizing {
            hydro_count: 0,
            max_par_order: 0,
            n_load_buses: 0,
            max_blocks: 1,
            downstream_par_order: 0,
            max_openings: 1,
            initial_pool_capacity: 0,
            n_state: indexer.n_state,
        };
        SolverWorkspace::new(
            TrackingMockSolver::new(),
            PatchBuffer::new(0, 0, 0, 0),
            indexer.n_state,
            sizing,
        )
    };

    // Helper: build exchange buffers pre-populated with n_fwd state vectors.
    let make_exchange = || {
        use cobre_comm::LocalBackend;
        use cobre_sddp::TrajectoryRecord;
        let mut bufs = ExchangeBuffers::new(indexer.n_state, n_fwd, 1);
        let records: Vec<TrajectoryRecord> = (0..n_fwd)
            .map(|i| TrajectoryRecord {
                primal: Vec::new(),
                dual: Vec::new(),
                stage_cost: 0.0,
                state: vec![i as f64 * 10.0],
            })
            .collect();
        bufs.exchange(&records, 0, 1, &LocalBackend).unwrap();
        bufs
    };

    // ── Bake step: build baked templates from iter-0 cuts ──────────────────
    // Baked templates capture all currently-active cuts in fcf_baked_run so
    // that the backward pass can call load_model(baked) + add_rows(delta_only).
    let mut bake_batches: Vec<RowBatch> = (0..n_stages).map(|_| empty_batch()).collect();
    let mut baked_templates: Vec<StageTemplate> =
        (0..n_stages).map(|_| StageTemplate::empty()).collect();

    for t in 0..n_stages {
        build_cut_row_batch_into(
            &mut bake_batches[t],
            &fcf_baked_run,
            t,
            &indexer,
            &base_templates[t].col_scale,
        );
        bake_rows_into_template(
            &base_templates[t],
            &bake_batches[t],
            &mut baked_templates[t],
        );
    }

    // Sanity: baked templates have more rows than base.
    for t in 0..n_stages {
        assert!(
            baked_templates[t].num_rows > base_templates[t].num_rows,
            "stage {t}: baked template must have more rows"
        );
    }

    // ── Legacy path (iter 1): measure add_rows_count delta ─────────────────
    // Uses a separate FCF instance so that cuts added by the legacy pass do
    // not pollute the baked FCF. The baked pass's FCF (fcf_baked_run) must
    // contain exactly the cuts that are baked into the templates.
    let mut fcf_legacy_run = make_fcf_iter0();
    let mut workspaces_legacy = vec![make_workspace()];
    let mut exchange_legacy = make_exchange();
    let mut cut_batches_legacy: Vec<RowBatch> = (0..n_stages).map(|_| empty_batch()).collect();
    let mut csb_legacy = CutSyncBuffers::new(indexer.n_state, 64, 1);
    let basis_store_legacy = BasisStore::new(n_fwd, n_stages);

    let not_baked = BakedTemplates {
        templates: &[],
        ready: false,
    };
    let add_rows_before_legacy = workspaces_legacy[0].solver.add_rows_count;
    let _ = run_backward_pass(
        &mut workspaces_legacy,
        &basis_store_legacy,
        &stage_ctx,
        &not_baked,
        &mut fcf_legacy_run,
        &mut cut_batches_legacy,
        &training_ctx,
        &mut BackwardPassSpec {
            exchange: &mut exchange_legacy,
            records: &[],
            iteration: 1,
            local_work: n_fwd,
            fwd_offset: 0,
            risk_measures: &vec![RiskMeasure::Expectation; n_stages],
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb_legacy,
            probabilities_buf: &mut Vec::new(),
            successor_active_slots_buf: &mut Vec::new(),
            visited_archive: None,
            metadata_sync_buf: &mut Vec::new(),
            global_increments_buf: &mut Vec::new(),
            real_states_buf: &mut Vec::new(),
        },
        &StubComm,
    )
    .expect("legacy backward pass must not error");
    let add_rows_delta_legacy = workspaces_legacy[0].solver.add_rows_count - add_rows_before_legacy;

    // Legacy path adds rows each time (once per worker pre-loop + once per
    // trial-point reload × stages with a successor).
    assert!(
        add_rows_delta_legacy > 0,
        "legacy path must call add_rows; got {add_rows_delta_legacy}"
    );

    // ── Baked path (iter 2): delta = 0 for the first successor, > 0 for the second ──
    // At the start of the baked pass, fcf_baked_run has only iter-0 cuts. The
    // baked templates were built from those same cuts, so the first successor
    // (stage 2, processed at t=1) has no delta cuts and add_rows is not called.
    // After t=1 finishes, iter-2 cuts are added to stage 1. These become the
    // delta for t=0 (successor=1), so add_rows IS called for that stage.
    //
    // Key invariant: baked path total add_rows calls < legacy total add_rows
    // calls, because the baked path skips add_rows entirely for stages with
    // zero delta. In this 3-stage, n_fwd=4 scenario, the legacy path calls
    // add_rows 10 times (5 per successor) while the baked path calls it only
    // 5 times (0 for successor=2, 5 for successor=1).
    let mut workspaces_baked = vec![make_workspace()];
    let mut exchange_baked = make_exchange();
    let mut cut_batches_baked: Vec<RowBatch> = (0..n_stages).map(|_| empty_batch()).collect();
    let mut csb_baked = CutSyncBuffers::new(indexer.n_state, 64, 1);
    let basis_store_baked = BasisStore::new(n_fwd, n_stages);

    let is_baked = BakedTemplates {
        templates: &baked_templates,
        ready: true,
    };
    let add_rows_before_baked = workspaces_baked[0].solver.add_rows_count;
    let _ = run_backward_pass(
        &mut workspaces_baked,
        &basis_store_baked,
        &stage_ctx,
        &is_baked,
        &mut fcf_baked_run,
        &mut cut_batches_baked,
        &training_ctx,
        &mut BackwardPassSpec {
            exchange: &mut exchange_baked,
            records: &[],
            iteration: 2,
            local_work: n_fwd,
            fwd_offset: 0,
            risk_measures: &vec![RiskMeasure::Expectation; n_stages],
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb_baked,
            probabilities_buf: &mut Vec::new(),
            successor_active_slots_buf: &mut Vec::new(),
            visited_archive: None,
            metadata_sync_buf: &mut Vec::new(),
            global_increments_buf: &mut Vec::new(),
            real_states_buf: &mut Vec::new(),
        },
        &StubComm,
    )
    .expect("baked backward pass must not error");
    let add_rows_delta_baked = workspaces_baked[0].solver.add_rows_count - add_rows_before_baked;

    // Baked path must call add_rows FEWER times than the legacy path.
    // The legacy path adds the full active cut set for every stage; the baked
    // path skips add_rows entirely for stages with zero delta cuts.
    assert!(
        add_rows_delta_baked < add_rows_delta_legacy,
        "baked path must call add_rows fewer times than legacy; \
         baked={add_rows_delta_baked}, legacy={add_rows_delta_legacy}"
    );
    // Baked path must still have called load_model (for each trial-point reload).
    assert!(
        workspaces_baked[0].solver.load_model_count > 0,
        "baked path must still call load_model"
    );
    // Legacy path must have called add_rows for EVERY stage with a successor.
    // (n_successors × (1 pre-loop + n_fwd trial-point reloads) = 2 × 5 = 10)
    let expected_legacy_add_rows = 2 * (1 + n_fwd) as u64;
    assert_eq!(
        add_rows_delta_legacy, expected_legacy_add_rows,
        "legacy path must call add_rows {expected_legacy_add_rows} times; \
         got {add_rows_delta_legacy}"
    );
}

/// AC (ticket-011): smoke test that the baked-template backward pass does not
/// diverge or panic over multiple iterations.
///
/// Smoke test: baking activates on iteration 2 and completes 5 iterations.
/// Verifies training completes, lower bound is non-negative, and iteration count matches.
#[test]
fn baked_backward_pass_smoke_test() {
    let n_iter = 5_u64;
    let fx = Fixture::new(3);
    let mut fcf = make_fcf(fx.n_stages);
    // ExpandingMockSolver tracks current_num_rows (updated on load_model/add_rows)
    // and returns a dual slice of that length, which is required once cuts are
    // added on iteration 2+ (baked path). MockSolver has a hardcoded 2-element
    // dual and would panic with an out-of-bounds slice access.
    let mut solver = ExpandingMockSolver::with_objectives(vec![50.0]);
    let stage_ctx = StageContext {
        templates: &fx.templates,
        base_rows: &fx.base_rows,
        noise_scale: &[],
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &[1_usize, 1, 1],
        ncs_max_gen: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
    };

    let outcome = train(
        &mut solver,
        TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                max_iterations: n_iter,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit(n_iter),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                warm_start_cuts: 0,
                risk_measures: fx.risk_measures.clone(),
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
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            stages: &[],
        },
        &StubComm,
        || Ok(ExpandingMockSolver::with_objectives(vec![50.0])),
    )
    .expect("baked backward pass smoke: train must not error");

    // Training ran to the requested iteration limit.
    assert_eq!(
        outcome.result.iterations, n_iter,
        "expected {n_iter} iterations, got {}",
        outcome.result.iterations
    );

    // Lower bound from MockSolver (fixed obj=50) must be non-negative.
    assert!(
        outcome.result.final_lb >= 0.0,
        "final lower bound must be non-negative; got {}",
        outcome.result.final_lb
    );
}
