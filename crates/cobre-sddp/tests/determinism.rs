//! Determinism verification tests for the SDDP training and simulation loops.
//!
//! Verifies that the rayon-parallelized forward pass, backward pass, and
//! simulation produce bit-identical results regardless of thread count.
//! This property is guaranteed by:
//!
//! 1. Deterministic SipHash-1-3 seed derivation per scenario (deterministic SipHash-1-3 seed derivation).
//! 2. Declaration-order invariance in entity processing.
//! 3. Static work partitioning (not rayon default chunking).
//! 4. Deterministic cut merging order (sorted by trial point index).
//!
//! Each test runs with 1 workspace, then with 4 workspaces, and asserts
//! bit-exact equality on all outputs.

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

use std::collections::{BTreeMap, HashMap};
use std::sync::mpsc;

use chrono::NaiveDate;
use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::{
    Bus, DeficitSegment, EntityId,
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, SamplingScheme,
    },
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_sddp::{
    CutManagementConfig, EntityCounts, EventConfig, ForwardResult, FutureCostFunction, HorizonMode,
    InflowNonNegativityMethod, LoopConfig, PatchBuffer, RiskMeasure, SimulationConfig,
    SimulationOutputSpec, SolverWorkspace, StageContext, StageIndexer, StoppingMode, StoppingRule,
    StoppingRuleSet, TrainingConfig, TrainingContext, WorkspaceSizing, simulate, sync_forward,
    train,
};
use cobre_solver::{
    Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
};
use cobre_stochastic::{
    ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context,
};

// ===========================================================================
// Shared communicator stub
// ===========================================================================

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

// ===========================================================================
// Mock solver for N=3 hydros, L=0 PAR
//
// Column layout for StageIndexer::new(3, 0):
//   storage      = 0..3
//   inflow_lags  = 3..3  (empty, L=0)
//   z_inflow     = 3..6
//   storage_in   = 6..9
//   theta        = 9
//   num_cols     = 10
//
// The primal must have 10 entries so `view.primal[indexer.theta]` (index 9)
// is valid. The dual must have at least n_dual_relevant = 3 entries so the
// backward pass can extract dual values for the 3 storage-fixing rows.
// ===========================================================================

const PRIMAL_3H: &[f64] = &[0.0; 10];
// The dual must cover: n_dual_relevant (3) + max cuts per stage (10 iterations × 1 pass = 10).
// Use 64 to cover any reasonable iteration count without tight sizing.
const DUAL_3H: &[f64] = &[0.0; 64];
const REDUCED_COSTS_3H: &[f64] = &[0.0; 10];

/// Mock solver for a 3-hydro, PAR(0) stage LP.
///
/// Returns a fixed objective on every solve. Thread counts and parallel
/// scheduling do not affect the solver result — this is the baseline for
/// verifying that the orchestration layer is itself deterministic.
struct MockSolver3H {
    objective: f64,
}

impl MockSolver3H {
    fn new(objective: f64) -> Self {
        Self { objective }
    }
}

impl SolverInterface for MockSolver3H {
    fn solver_name_version(&self) -> String {
        "MockSolver 0.0.0".to_string()
    }
    fn load_model(&mut self, _template: &StageTemplate) {}
    fn add_rows(&mut self, _cuts: &RowBatch) {}
    fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

    fn solve(&mut self) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
        Ok(cobre_solver::SolutionView {
            objective: self.objective,
            primal: PRIMAL_3H,
            dual: DUAL_3H,
            reduced_costs: REDUCED_COSTS_3H,
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
        "MockDeterminism3H"
    }
}

// ===========================================================================
// Fixture construction
// ===========================================================================

/// Build a `StochasticContext` for a 3-hydro, 5-stage system with seed 42.
///
/// Uses PAR(0) (no AR lags) and a single opening per stage so the fixture
/// remains small while still exercising more code paths than a 1-hydro system.
fn make_stochastic_context_3h(n_stages: usize) -> StochasticContext {
    use cobre_core::SystemBuilder;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::InflowModel;

    let zero_penalties = || HydroPenalties {
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
    };

    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };

    let make_hydro = |id_val: i32, name: &str| Hydro {
        id: EntityId(id_val),
        name: name.to_string(),
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
        penalties: zero_penalties(),
    };

    let hydros = vec![
        make_hydro(1, "H1"),
        make_hydro(2, "H2"),
        make_hydro(3, "H3"),
    ];

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

    // One InflowModel per (hydro, stage) pair.
    let mut inflow_models: Vec<InflowModel> = Vec::new();
    for stage_idx in 0..n_stages {
        for hydro_id in [1i32, 2, 3] {
            inflow_models.push(InflowModel {
                hydro_id: EntityId(hydro_id),
                stage_id: stage_idx as i32,
                mean_m3s: 50.0 + f64::from(hydro_id) * 10.0,
                std_m3s: 15.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            });
        }
    }

    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: vec![
                    CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(1),
                    },
                    CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(2),
                    },
                    CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(3),
                    },
                ],
                matrix: vec![
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 1.0],
                ],
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
        .hydros(hydros)
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

/// Build a `StageTemplate` for a 3-hydro, PAR(0) stage LP.
///
/// Column layout (N=3, L=0):
/// ```text
/// 0..3  storage_out  (outgoing storage, N=3)
/// 3..6  z_inflow     (realized inflow variables, N=3)
/// 6..9  storage_in   (incoming storage, N=3, L=0 → no lag cols)
/// 9     theta
/// ```
///
/// Row layout (N=3, L=0):
/// ```text
/// 0..3  storage-fixing rows  (one per hydro)
/// 3..6  z_inflow rows        (one per hydro, at N*(1+L)=3)
/// ```
///
/// The matrix has one nonzero per storage-fixing row (column = `storage_in[h]`,
/// coefficient = 1.0) so the patch buffer has something to patch.
fn template_3h() -> StageTemplate {
    // 3 storage-fixing rows, one NZ each → 3 NZ total.
    // col_starts: CSC format. 10 columns + 1 sentinel = 11 entries.
    // Columns 0..3 (storage_out): no NZ
    // Columns 3..6 (z_inflow):    no NZ
    // Columns 6..9 (storage_in):  one NZ each (rows 0, 1, 2 respectively)
    // Column  9    (theta):       no NZ
    let col_starts = vec![
        0_i32, // col 0 (storage_out[0])
        0,     // col 1 (storage_out[1])
        0,     // col 2 (storage_out[2])
        0,     // col 3 (z_inflow[0])
        0,     // col 4 (z_inflow[1])
        0,     // col 5 (z_inflow[2])
        0,     // col 6 (storage_in[0]) — NZ starts here
        1,     // col 7 (storage_in[1])
        2,     // col 8 (storage_in[2])
        3,     // col 9 (theta)
        3,     // sentinel
    ];
    let row_indices = vec![0_i32, 1, 2]; // row 0, 1, 2 for storage_in cols
    let values = vec![1.0_f64, 1.0, 1.0];

    let mut objective = vec![0.0_f64; 10];
    objective[9] = 1.0; // theta at col 9

    StageTemplate {
        num_cols: 10,
        num_rows: 6,
        num_nz: 3,
        col_starts,
        row_indices,
        values,
        col_lower: vec![0.0; 10],
        col_upper: vec![f64::INFINITY; 10],
        objective,
        row_lower: vec![0.0; 6],
        row_upper: vec![0.0; 6],
        n_state: 3,
        n_transfer: 0,
        n_dual_relevant: 3,
        n_hydro: 3,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

fn make_fcf_3h(n_stages: usize) -> FutureCostFunction {
    // state_dimension = 3, forward_passes = 1, capacity = 50 iterations, 0 warm-start cuts
    FutureCostFunction::new(n_stages, 3, 1, 50, &vec![0; n_stages])
}

fn iteration_limit(limit: u64) -> StoppingRuleSet {
    StoppingRuleSet {
        rules: vec![StoppingRule::IterationLimit { limit }],
        mode: StoppingMode::Any,
    }
}

/// All training parameters for the 3-hydro, 5-stage test system.
struct Fixture3H {
    n_stages: usize,
    templates: Vec<StageTemplate>,
    base_rows: Vec<usize>,
    indexer: StageIndexer,
    initial_state: Vec<f64>,
    stochastic: StochasticContext,
    horizon: HorizonMode,
    risk_measures: Vec<RiskMeasure>,
}

impl Fixture3H {
    fn new() -> Self {
        let n_stages = 5;
        // N=3 hydros, L=0 PAR order
        let indexer = StageIndexer::new(3, 0);
        let templates = vec![template_3h(); n_stages];
        // base_row: water-balance rows start at row_water_balance_start = n_state + n_hydros = 3 + 3 = 6.
        let base_rows = vec![6usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context_3h(n_stages);
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

// ===========================================================================
// Helper: run training with a given number of forward-pass workspaces
// ===========================================================================

/// Run `train()` on the 3-hydro fixture with `n_workspaces` forward-pass threads
/// and return the `(TrainingResult, FutureCostFunction)` pair.
///
/// An isolated rayon thread pool with exactly `n_workspaces` threads is used
/// to prevent interaction with the global pool or other parallel tests.
fn run_training(
    n_workspaces: usize,
    fx: &Fixture3H,
    n_iterations: u64,
) -> (cobre_sddp::TrainingResult, FutureCostFunction) {
    // Each run gets a fresh FCF and a fresh primary solver.
    // train() now returns TrainingOutcome; unwrap the result field.
    let mut fcf = make_fcf_3h(fx.n_stages);
    let mut primary_solver = MockSolver3H::new(100.0);
    let comm = StubComm;

    let config = TrainingConfig {
        loop_config: LoopConfig {
            forward_passes: 1,
            max_iterations: n_iterations,
            start_iteration: 0,
            n_fwd_threads: 1,
            max_blocks: 1,
            stopping_rules: iteration_limit(n_iterations),
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

    // Use an isolated thread pool so that tests with different workspace counts
    // do not share the global rayon pool and do not interfere with each other.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_workspaces)
        .build()
        .unwrap();

    let stage_ctx = StageContext {
        templates: &fx.templates,
        base_rows: &fx.base_rows,
        noise_scale: &[],
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &[1usize; 5],
        ncs_max_gen: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
    };
    let result = pool
        .install(|| {
            train(
                &mut primary_solver,
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
                    recent_accum_seed: &[],
                    recent_weight_seed: 0.0,
                },
                &comm,
                || Ok(MockSolver3H::new(100.0)),
            )
        })
        .unwrap();

    (result.result, fcf)
}

// ===========================================================================
// Helper: run simulation with a given number of workspaces
// ===========================================================================

/// Run `simulate()` on the trained FCF with `n_workspaces` worker threads.
///
/// Returns the sorted cost buffer `Vec<(scenario_id, total_cost, category_costs)>`.
fn run_simulation(
    n_workspaces: usize,
    fx: &Fixture3H,
    fcf: &FutureCostFunction,
    n_scenarios: u32,
) -> Vec<(u32, f64, cobre_sddp::ScenarioCategoryCosts)> {
    let sim_config = SimulationConfig {
        n_scenarios,
        io_channel_capacity: 64,
    };
    let entity_counts = EntityCounts {
        hydro_ids: vec![1, 2, 3],
        hydro_productivities: vec![1.0, 1.0, 1.0],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![0],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    // Build a workspace pool of `n_workspaces` independently allocated workspaces.
    let mut workspaces: Vec<SolverWorkspace<MockSolver3H>> = (0..n_workspaces)
        .map(|_| {
            SolverWorkspace::new(
                MockSolver3H::new(100.0),
                PatchBuffer::new(fx.indexer.hydro_count, fx.indexer.max_par_order, 0, 0),
                fx.indexer.n_state,
                WorkspaceSizing {
                    hydro_count: fx.indexer.hydro_count,
                    max_par_order: fx.indexer.max_par_order,
                    n_load_buses: 0,
                    max_blocks: 0,
                    downstream_par_order: 0,
                    ..WorkspaceSizing::default()
                },
            )
        })
        .collect();

    let comm = StubComm;
    let (result_tx, result_rx) = mpsc::sync_channel(64);

    // Drain the channel in a background thread to avoid blocking simulate().
    let drain_thread = std::thread::spawn(move || {
        let mut results = Vec::new();
        while let Ok(r) = result_rx.recv() {
            results.push(r);
        }
        results
    });

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_workspaces)
        .build()
        .unwrap();

    let cost_buffer = pool
        .install(|| {
            simulate(
                &mut workspaces,
                &StageContext {
                    templates: &fx.templates,
                    base_rows: &fx.base_rows,
                    noise_scale: &[],
                    n_hydros: 0,
                    n_load_buses: 0,
                    load_balance_row_starts: &[],
                    load_bus_indices: &[],
                    block_counts_per_stage: &[],
                    ncs_max_gen: &[],
                    discount_factors: &[],
                    cumulative_discount_factors: &[],
                    stage_lag_transitions: &[],
                    noise_group_ids: &[],
                    downstream_par_order: 0,
                },
                fcf,
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
                    recent_accum_seed: &[],
                    recent_weight_seed: 0.0,
                },
                &sim_config,
                SimulationOutputSpec {
                    result_tx: &result_tx,
                    zeta_per_stage: &[],
                    block_hours_per_stage: &[],
                    entity_counts: &entity_counts,
                    generic_constraint_row_entries: &[],
                    ncs_col_starts: &[],
                    n_ncs_per_stage: &[],
                    ncs_entity_ids_per_stage: &[],
                    diversion_upstream: &HashMap::new(),
                    hydro_productivities_per_stage: &vec![vec![1.0, 1.0, 1.0]; fx.n_stages],
                    event_sender: None,
                },
                &[],
                &comm,
            )
        })
        .unwrap();

    drop(result_tx);
    let _ = drain_thread.join().unwrap();

    cost_buffer.costs
}

// ===========================================================================
// Test: training determinism across thread counts
// ===========================================================================

/// Verify that `train()` produces bit-identical outputs when run with 1
/// workspace versus 4 workspaces.
///
/// The fixture uses 3 hydros and 5 stages, which exercises enough
/// parallelisation code paths (forward pass partitioning, backward pass
/// synchronisation, cut merging order) to catch ordering bugs.
#[test]
fn test_training_determinism_across_thread_counts() {
    const N_ITERATIONS: u64 = 10;

    let fx = Fixture3H::new();
    let (result_1, fcf_1) = run_training(1, &fx, N_ITERATIONS);
    let (result_4, fcf_4) = run_training(4, &fx, N_ITERATIONS);

    assert_eq!(result_1.iterations, result_4.iterations);
    assert_eq!(result_1.final_lb.to_bits(), result_4.final_lb.to_bits());
    assert_eq!(result_1.final_ub.to_bits(), result_4.final_ub.to_bits());
    assert_eq!(result_1.final_gap.to_bits(), result_4.final_gap.to_bits());

    assert_eq!(fcf_1.pools.len(), fcf_4.pools.len());
    for t in 0..fcf_1.pools.len() {
        let pool_1 = &fcf_1.pools[t];
        let pool_4 = &fcf_4.pools[t];
        assert_eq!(pool_1.populated_count, pool_4.populated_count);

        for s in 0..pool_1.populated_count {
            assert_eq!(pool_1.active[s], pool_4.active[s]);
            assert_eq!(
                pool_1.intercepts[s].to_bits(),
                pool_4.intercepts[s].to_bits()
            );
            let sd = pool_1.state_dimension;
            let start = s * sd;
            let c1 = &pool_1.coefficients[start..start + sd];
            let c4 = &pool_4.coefficients[start..start + sd];
            assert_eq!(c1.len(), c4.len());
            for (&coeff_1, &coeff_4) in c1.iter().zip(c4.iter()) {
                assert_eq!(coeff_1.to_bits(), coeff_4.to_bits());
            }
        }
    }
}

// ===========================================================================
// Multi-rank mock communicator
// ===========================================================================

use std::any::Any;
use std::cell::RefCell;
use std::sync::Arc;

/// Type-erased wrapper that lets `allgatherv<T>` retrieve a `Vec<T>` stored
/// under the mock's thread-local gather buffer.
///
/// The thread-local holds `Box<dyn Any + Send>`. Before each `sync_forward`
/// call the test stores `GatherBuffer(pre_assembled_vec_f64)`. Inside
/// `allgatherv<T>` we call `downcast_ref::<GatherBuffer<T>>()`: when
/// `T = f64` the downcast succeeds and we get a `&Vec<f64>` that can be
/// `clone_from_slice`d directly into `recv: &mut [f64]` — no unsafe, no
/// transmutation.
struct GatherBuffer<T>(Vec<T>);

thread_local! {
    /// Stores the pre-assembled flat gather buffer as a type-erased `Any`.
    ///
    /// Set by [`MultiRankMockComm::set_gather_buffer`] before each call to
    /// `sync_forward`. Read by `allgatherv<T>` via `downcast_ref::<GatherBuffer<T>>`.
    static MOCK_GATHER_BUFFER: RefCell<Arc<dyn Any + Send + Sync>> =
        RefCell::new(Arc::new(GatherBuffer::<f64>(Vec::new())));
}

/// Multi-rank mock communicator for testing canonical summation.
///
/// Simulates a single rank within a multi-rank group. `allgatherv` fills the
/// entire `recv` buffer from a pre-loaded gather buffer set via
/// [`MultiRankMockComm::set_gather_buffer`], faithfully reproducing what real
/// MPI `allgatherv` delivers to every rank — all ranks receive all data.
///
/// The test orchestrator stores the full flat cost vector before each
/// `sync_forward` call. When `allgatherv<f64>` runs inside `sync_forward`,
/// the type-erased buffer is downcast back to `GatherBuffer<f64>` via
/// `Any::downcast_ref` (a safe operation), and its contents are
/// `clone_from_slice`d into `recv: &mut [f64]` (also safe). This pattern
/// avoids unsafe transmutations while correctly simulating multi-rank gather.
struct MultiRankMockComm {
    rank: usize,
    total_size: usize,
}

impl MultiRankMockComm {
    /// Create a mock for virtual `rank` within a group of `total_size` ranks.
    fn new(rank: usize, total_size: usize) -> Self {
        Self { rank, total_size }
    }

    /// Pre-load the full flat gathered cost buffer used by `allgatherv`.
    ///
    /// `global_costs` must contain all ranks' scenario costs in rank order:
    /// rank 0's costs first, then rank 1's, etc. This mirrors what real MPI
    /// `allgatherv` places in `recv` on every participating rank.
    ///
    /// Must be called on the same thread that will call `sync_forward`.
    fn set_gather_buffer(global_costs: Vec<f64>) {
        MOCK_GATHER_BUFFER.with(|cell| {
            *cell.borrow_mut() = Arc::new(GatherBuffer(global_costs));
        });
    }
}

impl Communicator for MultiRankMockComm {
    /// Simulate `allgatherv` by filling `recv` from the pre-loaded buffer.
    ///
    /// The pre-loaded buffer is type-erased as `Box<dyn Any>`. This method
    /// attempts `downcast_ref::<GatherBuffer<T>>()` to retrieve a `&Vec<T>`.
    /// When `T = f64` (which is always the case when called from
    /// `sync_forward`), the downcast succeeds and the buffer contents are
    /// copied into `recv` via `clone_from_slice` — a fully safe operation.
    ///
    /// If the downcast fails (e.g., some other `T` is used), the method falls
    /// back to filling only the local rank's slot from `_send` and leaving
    /// other slots at `T::default()`. This fallback is unreachable in the
    /// tests in this file.
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        counts: &[usize],
        displs: &[usize],
    ) -> Result<(), CommError> {
        MOCK_GATHER_BUFFER.with(|cell| {
            let arc = cell.borrow();
            if let Some(buf) = arc.downcast_ref::<GatherBuffer<T>>() {
                // Happy path: T = f64 (always true in sync_forward calls).
                // Fill recv from the pre-assembled flat buffer in rank order.
                for rank in 0..self.total_size {
                    let start = displs[rank];
                    let count = counts[rank];
                    recv[start..start + count].clone_from_slice(&buf.0[start..start + count]);
                }
            } else {
                // Fallback: only fill this rank's local slot.
                // Unreachable in the determinism tests (T is always f64).
                let local_start = displs[self.rank];
                let local_count = counts[self.rank];
                recv[local_start..local_start + local_count].clone_from_slice(send);
            }
        });
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
        self.rank
    }

    fn size(&self) -> usize {
        self.total_size
    }

    fn abort(&self, error_code: i32) -> ! {
        std::process::exit(error_code)
    }
}

// ===========================================================================
// Test: canonical upper bound determinism across rank counts
// ===========================================================================

/// Verify that `sync_forward` produces bit-identical `SyncResult` statistics
/// when the same 8 scenario costs are partitioned across 1, 2, and 4 virtual
/// ranks.
///
/// This is the CI-compatible regression gate for the canonical upper bound
/// summation fix (ticket-009). After that fix, `sync_forward` uses `allgatherv`
/// to assemble a flat global cost vector in rank order, then sums it
/// sequentially. The sequential summation order is identical regardless of how
/// many ranks contribute (because the costs always appear in global scenario
/// index order: rank 0's costs first, rank 1's next, etc.).
///
/// The test uses [`MultiRankMockComm`] to simulate 2-rank and 4-rank
/// `allgatherv` without requiring an MPI installation. The mock pre-loads the
/// full flat gather buffer via [`MultiRankMockComm::set_gather_buffer`] and
/// correctly fills the entire `recv` slice using a type-erased `Any`-based
/// downcast — no unsafe code required.
#[test]
fn test_canonical_ub_determinism_across_rank_counts() {
    // Known cost vector with distinct non-trivial values.
    // Chosen so that partial sums group differently across partition boundaries,
    // exercising the canonical summation property.
    const ALL_COSTS: &[f64] = &[100.0, 200.0, 150.0, 175.0, 125.0, 190.0, 160.0, 180.0];
    const N: usize = 8;
    const TOTAL_FWD_PASSES: usize = N;

    let result_1rank = {
        let local = ForwardResult {
            scenario_costs: ALL_COSTS.to_vec(),
            elapsed_ms: 0,
            lp_solves: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
        };
        sync_forward(&local, &StubComm, TOTAL_FWD_PASSES).unwrap()
    };

    let result_2rank = {
        MultiRankMockComm::set_gather_buffer(ALL_COSTS.to_vec());
        let comm = MultiRankMockComm::new(0, 2);
        let local = ForwardResult {
            scenario_costs: ALL_COSTS[..4].to_vec(),
            elapsed_ms: 0,
            lp_solves: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
        };
        sync_forward(&local, &comm, TOTAL_FWD_PASSES).unwrap()
    };

    let result_4rank = {
        MultiRankMockComm::set_gather_buffer(ALL_COSTS.to_vec());
        let comm = MultiRankMockComm::new(0, 4);
        let local = ForwardResult {
            scenario_costs: ALL_COSTS[..2].to_vec(),
            elapsed_ms: 0,
            lp_solves: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
        };
        sync_forward(&local, &comm, TOTAL_FWD_PASSES).unwrap()
    };

    assert_eq!(
        result_1rank.global_ub_mean.to_bits(),
        result_2rank.global_ub_mean.to_bits()
    );
    assert_eq!(
        result_1rank.global_ub_mean.to_bits(),
        result_4rank.global_ub_mean.to_bits()
    );
    assert_eq!(
        result_1rank.global_ub_std.to_bits(),
        result_2rank.global_ub_std.to_bits()
    );
    assert_eq!(
        result_1rank.global_ub_std.to_bits(),
        result_4rank.global_ub_std.to_bits()
    );
    assert_eq!(
        result_1rank.ci_95_half_width.to_bits(),
        result_2rank.ci_95_half_width.to_bits()
    );
    assert_eq!(
        result_1rank.ci_95_half_width.to_bits(),
        result_4rank.ci_95_half_width.to_bits()
    );
}

// ===========================================================================
// Test: simulation determinism across thread counts
// ===========================================================================

/// Verify that `simulate()` produces bit-identical cost buffers when run with
/// 1 workspace versus 4 workspaces on the same trained FCF.
///
/// Uses the same 3-hydro, 5-stage fixture with 20 scenarios, which is enough
/// to require work distribution across multiple workers when 4 workspaces are
/// active.
#[test]
fn test_simulation_determinism_across_thread_counts() {
    const N_ITERATIONS: u64 = 10;
    const N_SCENARIOS: u32 = 20;

    let fx = Fixture3H::new();
    // Train once with 1 workspace to get a stable FCF for simulation.
    let (_training_result, fcf) = run_training(1, &fx, N_ITERATIONS);

    let costs_1 = run_simulation(1, &fx, &fcf, N_SCENARIOS);
    let costs_4 = run_simulation(4, &fx, &fcf, N_SCENARIOS);

    assert_eq!(costs_1.len(), costs_4.len());
    for ((id_1, cost_1, cats_1), (id_4, cost_4, cats_4)) in costs_1.iter().zip(costs_4.iter()) {
        assert_eq!(id_1, id_4);
        assert_eq!(cost_1.to_bits(), cost_4.to_bits());
        assert_eq!(
            cats_1.resource_cost.to_bits(),
            cats_4.resource_cost.to_bits()
        );
        assert_eq!(
            cats_1.recourse_cost.to_bits(),
            cats_4.recourse_cost.to_bits()
        );
        assert_eq!(
            cats_1.violation_cost.to_bits(),
            cats_4.violation_cost.to_bits()
        );
        assert_eq!(
            cats_1.regularization_cost.to_bits(),
            cats_4.regularization_cost.to_bits()
        );
        assert_eq!(cats_1.imputed_cost.to_bits(), cats_4.imputed_cost.to_bits());
    }
}
