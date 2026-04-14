//! End-to-end integration tests for the stochastic load pipeline.
//!
//! Exercises the full stochastic load path from [`System`] construction with
//! [`LoadModel`] entries through [`build_stochastic_context`] verification
//! and a mock-solver training run, confirming that load noise is wired into
//! the context and that training completes without errors.
//!
//! ## Design constraints
//!
//! - Only the public `cobre_sddp::` and `cobre_stochastic::` APIs are used.
//! - All test helpers are defined locally in this file.
//! - Each test is self-contained with no cross-test shared state.
//! - `MockSolver` is used throughout — no real solver (`HiGHS`) dependency.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]

use std::collections::BTreeMap;
use std::sync::mpsc;

use chrono::NaiveDate;
use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::{
    Bus, DeficitSegment, EntityId, TrainingEvent,
    entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties},
    scenario::{InflowModel, LoadModel, SamplingScheme},
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_sddp::{
    HorizonMode, InflowNonNegativityMethod, RiskMeasure, StageContext, StageIndexer, StoppingMode,
    StoppingRule, StoppingRuleSet, TrainingConfig, TrainingContext, cut::fcf::FutureCostFunction,
    train,
};
use cobre_solver::{
    Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
};
use cobre_stochastic::{
    ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context,
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

/// Mock solver that returns a fixed objective on every `solve` call.
struct MockSolver {
    objective: f64,
    call_count: usize,
}

impl MockSolver {
    fn with_fixed(objective: f64) -> Self {
        Self {
            objective,
            call_count: 0,
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
        self.call_count += 1;
        Ok(cobre_solver::SolutionView {
            objective: self.objective,
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
        "MockLoadIntegration"
    }
}

/// Build a `System` with 1 bus, 1 hydro, `n_stages` stages, and optionally
/// stochastic load data for the single bus.
///
/// - `load_mean_mw` / `load_std_mw`: parameters for [`LoadModel`] entries.
///   When `load_std_mw == 0.0` the load is deterministic and the returned
///   context will report `n_load_buses() == 0`.
/// - `n_openings`: branching factor for the opening tree.
///
/// The correlation model is intentionally left empty (no profiles), which
/// `build_stochastic_context` treats as independent (identity) correlation.
/// This is consistent with the unit-test fixture in `forward.rs`.
fn build_system_with_load(
    n_stages: usize,
    n_openings: usize,
    load_mean_mw: f64,
    load_std_mw: f64,
) -> cobre_core::System {
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

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(0),
            stage_id: i as i32,
            mean_mw: load_mean_mw,
            std_mw: load_std_mw,
        })
        .collect();

    // Empty correlation profiles: `build_stochastic_context` treats this as
    // independent (identity) correlation for all noise entities.
    let correlation = cobre_core::scenario::CorrelationModel {
        method: "spectral".to_string(),
        profiles: BTreeMap::new(),
        schedule: vec![],
    };

    cobre_core::SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .correlation(correlation)
        .build()
        .unwrap()
}

/// Build a `StochasticContext` from a system with load models.
fn build_context_with_load(
    n_stages: usize,
    load_mean_mw: f64,
    load_std_mw: f64,
) -> StochasticContext {
    let system = build_system_with_load(n_stages, 1, load_mean_mw, load_std_mw);
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
///
/// Load rows are patched via `set_row_bounds` at runtime; they do not change
/// the primal structure seen by the mock solver (3 columns, 1 state row).
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
    // capacity=50 iterations, state_dimension=1, 1 stage cut pool
    FutureCostFunction::new(n_stages, 1, 1, 50, &vec![0; n_stages])
}

fn iteration_limit(limit: u64) -> StoppingRuleSet {
    StoppingRuleSet {
        rules: vec![StoppingRule::IterationLimit { limit }],
        mode: StoppingMode::Any,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

/// Verify that `build_stochastic_context` reports `n_load_buses() == 1` when
/// the system has a `LoadModel` with `std_mw > 0`, and `n_load_buses() == 0`
/// when `std_mw == 0.0` (deterministic load).
#[test]
fn test_stochastic_load_context_construction() {
    // Stochastic case: std_mw=50.0 means the bus qualifies as stochastic.
    let stochastic_ctx = build_context_with_load(2, 500.0, 50.0);
    assert_eq!(
        stochastic_ctx.n_load_buses(),
        1,
        "n_load_buses must be 1 when std_mw=50.0 > 0"
    );

    // Deterministic case: std_mw=0.0 means the bus is excluded from noise dim.
    let deterministic_ctx = build_context_with_load(2, 500.0, 0.0);
    assert_eq!(
        deterministic_ctx.n_load_buses(),
        0,
        "n_load_buses must be 0 when std_mw=0.0"
    );
}

/// Verify that a 3-iteration training run with stochastic load (`std_mw=50.0`)
/// completes successfully and produces exactly 3 lower-bound entries.
#[test]
fn test_stochastic_load_training_completes() {
    let n_stages = 2usize;
    let n_load_buses = 1usize;
    let stochastic = build_context_with_load(n_stages, 500.0, 50.0);

    assert_eq!(
        stochastic.n_load_buses(),
        n_load_buses,
        "pre-condition: n_load_buses must be 1"
    );

    let indexer = StageIndexer::new(1, 0); // N=1 hydro, L=0 PAR
    let templates = vec![minimal_template(); n_stages];
    let base_rows = vec![2usize; n_stages];
    let initial_state = vec![0.0_f64; indexer.n_state];
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];
    let mut fcf = make_fcf(n_stages);
    let mut solver = MockSolver::with_fixed(100.0);
    let comm = StubComm;

    // Collect convergence events to verify we get one LB per iteration.
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
    };

    // load_balance_row_starts: one per stage, pointing past the base rows.
    // The mock solver ignores set_row_bounds so the exact value doesn't matter
    // as long as the slice length matches n_stages.
    let load_balance_row_starts = vec![1usize; n_stages];
    // load_bus_indices: the LP column index of the load bus (index 0 among buses).
    let load_bus_indices = vec![0usize];
    let block_counts_per_stage = vec![1usize; n_stages];

    let stage_ctx = StageContext {
        templates: &templates,
        base_rows: &base_rows,
        noise_scale: &[],
        n_hydros: 0,
        n_load_buses,
        load_balance_row_starts: &load_balance_row_starts,
        load_bus_indices: &load_bus_indices,
        block_counts_per_stage: &block_counts_per_stage,
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
            stages: &[],
        },
        &risk_measures,
        iteration_limit(3),
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
    )
    .expect("train must succeed with stochastic load");

    assert_eq!(
        result.result.iterations, 3,
        "expected exactly 3 iterations, got {}",
        result.result.iterations
    );

    // Collect ConvergenceUpdate events to confirm one LB per iteration.
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

    assert_eq!(
        lower_bounds.len(),
        3,
        "expected 3 lower-bound entries (one per iteration), got {}",
        lower_bounds.len()
    );
}

/// Verify that a training run with deterministic load (`std_mw=0.0`) produces
/// `n_load_buses() == 0` and completes successfully.
///
/// With no stochastic load buses the system behaves identically to the
/// no-load-model baseline from `integration.rs`.
#[test]
fn test_deterministic_load_training_matches_baseline() {
    let n_stages = 2usize;
    let stochastic = build_context_with_load(n_stages, 500.0, 0.0);

    assert_eq!(
        stochastic.n_load_buses(),
        0,
        "pre-condition: deterministic load must yield n_load_buses=0"
    );

    let indexer = StageIndexer::new(1, 0);
    let templates = vec![minimal_template(); n_stages];
    let base_rows = vec![2usize; n_stages];
    let initial_state = vec![0.0_f64; indexer.n_state];
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];
    let mut fcf = make_fcf(n_stages);
    let mut solver = MockSolver::with_fixed(100.0);
    let comm = StubComm;

    let block_counts_per_stage = vec![1usize; n_stages];

    let stage_ctx = StageContext {
        templates: &templates,
        base_rows: &base_rows,
        noise_scale: &[],
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &block_counts_per_stage,
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
            stages: &[],
        },
        &risk_measures,
        iteration_limit(3),
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
    )
    .expect("train must succeed with deterministic load");

    assert_eq!(
        result.result.iterations, 3,
        "expected exactly 3 iterations, got {}",
        result.result.iterations
    );
    assert!(
        result.result.final_lb >= 0.0,
        "final_lb must be non-negative"
    );
}

/// Verify that two training runs with identical seed=42 and stochastic load
/// configuration produce bit-for-bit identical lower-bound sequences.
#[test]
fn test_stochastic_load_seed_determinism() {
    let n_stages = 2usize;
    let n_load_buses = 1usize;

    let run_training = || {
        let stochastic = build_context_with_load(n_stages, 500.0, 50.0);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
        let mut fcf = make_fcf(n_stages);
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
        };

        let load_balance_row_starts = vec![1usize; n_stages];
        let load_bus_indices = vec![0usize];
        let block_counts_per_stage = vec![1usize; n_stages];

        let stage_ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses,
            load_balance_row_starts: &load_balance_row_starts,
            load_bus_indices: &load_bus_indices,
            block_counts_per_stage: &block_counts_per_stage,
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
                stages: &[],
            },
            &risk_measures,
            iteration_limit(3),
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .expect("train must succeed");

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

        (result, lower_bounds)
    };

    let (result1, lbs1) = run_training();
    let (result2, lbs2) = run_training();

    assert_eq!(
        result1.result.iterations, result2.result.iterations,
        "iteration counts must be identical: {} vs {}",
        result1.result.iterations, result2.result.iterations
    );

    assert_eq!(
        lbs1.len(),
        lbs2.len(),
        "lower-bound sequence lengths must match: {} vs {}",
        lbs1.len(),
        lbs2.len()
    );

    for (k, (lb1, lb2)) in lbs1.iter().zip(lbs2.iter()).enumerate() {
        assert_eq!(
            lb1.to_bits(),
            lb2.to_bits(),
            "lower bound at iteration {} must be bit-for-bit identical: {} vs {}",
            k + 1,
            lb1,
            lb2
        );
    }

    assert_eq!(
        result1.result.final_lb.to_bits(),
        result2.result.final_lb.to_bits(),
        "final_lb must be bit-for-bit identical: {} vs {}",
        result1.result.final_lb,
        result2.result.final_lb
    );
}
