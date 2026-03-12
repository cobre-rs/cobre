//! Determinism verification tests for the SDDP training and simulation loops.
//!
//! Verifies that the rayon-parallelized forward pass, backward pass, and
//! simulation produce bit-identical results regardless of thread count.
//! This property is guaranteed by:
//!
//! 1. Deterministic SipHash-1-3 seed derivation per scenario (DEC-017).
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

use std::collections::BTreeMap;
use std::sync::mpsc;

use chrono::NaiveDate;
use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::{
    Bus, DeficitSegment, EntityId,
    scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_sddp::{
    EntityCounts, FutureCostFunction, HorizonMode, InflowNonNegativityMethod, PatchBuffer,
    RiskMeasure, SimulationConfig, SolverWorkspace, StageIndexer, StoppingMode, StoppingRule,
    StoppingRuleSet, TrainingConfig, simulate, train,
};
use cobre_solver::{
    Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
};
use cobre_stochastic::{
    OpeningTree, StochasticContext, build_stochastic_context,
    correlation::resolve::DecomposedCorrelation, tree::generate::generate_opening_tree,
};

// ===========================================================================
// Shared communicator stub
// ===========================================================================

/// Single-rank communicator: correctly copies data through allgatherv and allreduce.
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
// Mock solver for N=3 hydros, L=0 PAR
//
// Column layout for StageIndexer::new(3, 0):
//   storage      = 0..3
//   inflow_lags  = 3..3  (empty, L=0)
//   storage_in   = 3..6
//   theta        = 6
//   num_cols     = 7
//
// The primal must have 7 entries so `view.primal[indexer.theta]` (index 6)
// is valid. The dual must have at least n_dual_relevant = 3 entries so the
// backward pass can extract dual values for the 3 storage-fixing rows.
// ===========================================================================

const PRIMAL_3H: &[f64] = &[0.0; 7];
// The dual must cover: n_dual_relevant (3) + max cuts per stage (10 iterations × 1 pass = 10).
// Use 64 to cover any reasonable iteration count without tight sizing.
const DUAL_3H: &[f64] = &[0.0; 64];
const REDUCED_COSTS_3H: &[f64] = &[0.0; 7];

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

/// Build an opening tree with `n_openings` at stage 0 using seed 42.
/// Uses 3 hydro correlation entities.
fn make_opening_tree_3h(n_openings: usize) -> OpeningTree {
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

    let id_h1 = EntityId(1);
    let id_h2 = EntityId(2);
    let id_h3 = EntityId(3);

    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: vec![
                    CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: id_h1,
                    },
                    CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: id_h2,
                    },
                    CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: id_h3,
                    },
                ],
                // 3x3 identity matrix — uncorrelated inflows.
                matrix: vec![
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 1.0],
                ],
            }],
        },
    );
    let corr_model = CorrelationModel {
        method: "cholesky".to_string(),
        profiles,
        schedule: vec![],
    };
    let mut decomposed = DecomposedCorrelation::build(&corr_model).unwrap();
    let entity_order = vec![id_h1, id_h2, id_h3];

    generate_opening_tree(42, &[stage], 3, &mut decomposed, &entity_order)
}

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
        method: "cholesky".to_string(),
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

    build_stochastic_context(&system, 42).unwrap()
}

/// Build a `StageTemplate` for a 3-hydro, PAR(0) stage LP.
///
/// Column layout:
/// ```text
/// 0..3  storage (outgoing, N=3)
/// 3..6  storage_in (incoming, N=3, L=0 → no lag cols)
/// 6     theta
/// ```
///
/// Rows: 3 storage-fixing constraints.
///
/// The matrix has one nonzero per storage-fixing row (column = `storage_in[h]`,
/// coefficient = 1.0) so the patch buffer has something to patch.
fn template_3h() -> StageTemplate {
    // 3 storage-fixing rows, one NZ each → 3 NZ total.
    // col_starts: CSC format. 7 columns + 1 sentinel = 8 entries.
    // Columns 0..3 (storage_out): no NZ
    // Columns 3..6 (storage_in): one NZ each (row 0, 1, 2 respectively)
    // Column 6 (theta): no NZ
    let col_starts = vec![
        0_i32, // col 0 (storage_out[0])
        0,     // col 1 (storage_out[1])
        0,     // col 2 (storage_out[2])
        0,     // col 3 (storage_in[0]) — NZ starts here
        1,     // col 4 (storage_in[1])
        2,     // col 5 (storage_in[2])
        3,     // col 6 (theta)
        3,     // sentinel
    ];
    let row_indices = vec![0_i32, 1, 2]; // row 0, 1, 2 for storage_in cols
    let values = vec![1.0_f64, 1.0, 1.0];

    StageTemplate {
        num_cols: 7,
        num_rows: 3,
        num_nz: 3,
        col_starts,
        row_indices,
        values,
        col_lower: vec![0.0; 7],
        col_upper: vec![f64::INFINITY; 7],
        objective: vec![0.0; 7],
        row_lower: vec![0.0; 3],
        row_upper: vec![0.0; 3],
        n_state: 3,
        n_transfer: 0,
        n_dual_relevant: 3,
        n_hydro: 3,
        max_par_order: 0,
    }
}

fn make_fcf_3h(n_stages: usize) -> FutureCostFunction {
    // state_dimension = 3, forward_passes = 1, capacity = 50 iterations, 0 warm-start cuts
    FutureCostFunction::new(n_stages, 3, 1, 50, 0)
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
    opening_tree: OpeningTree,
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
        // base_row: the AR-dynamics row offset equals n_dual_relevant = 3.
        let base_rows = vec![3usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let opening_tree = make_opening_tree_3h(1);
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
            opening_tree,
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
    let mut fcf = make_fcf_3h(fx.n_stages);
    let mut primary_solver = MockSolver3H::new(100.0);
    let comm = StubComm;

    let config = TrainingConfig {
        forward_passes: 1,
        max_iterations: n_iterations,
        checkpoint_interval: None,
        warm_start_cuts: 0,
        event_sender: None,
    };

    // Use an isolated thread pool so that tests with different workspace counts
    // do not share the global rayon pool and do not interfere with each other.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_workspaces)
        .build()
        .unwrap();

    let result = pool
        .install(|| {
            train(
                &mut primary_solver,
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
                iteration_limit(n_iterations),
                None,
                None,
                &comm,
                n_workspaces,
                || Ok(MockSolver3H::new(100.0)),
                &InflowNonNegativityMethod::None,
                &[],
                0,
            )
        })
        .unwrap();

    (result, fcf)
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
                PatchBuffer::new(fx.indexer.hydro_count, fx.indexer.max_par_order),
                fx.indexer.n_state,
                fx.indexer.hydro_count,
                fx.indexer.max_par_order,
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
                &fx.templates,
                &fx.base_rows,
                fcf,
                &fx.stochastic,
                &sim_config,
                &fx.horizon,
                &fx.initial_state,
                &fx.indexer,
                &entity_counts,
                &comm,
                &result_tx,
                &InflowNonNegativityMethod::None,
                &[],
                0,
                &[],
                &[],
                None,
            )
        })
        .unwrap();

    drop(result_tx);
    let _ = drain_thread.join().unwrap();

    cost_buffer
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

    assert_eq!(
        result_1.iterations, result_4.iterations,
        "iteration count must be identical: 1-thread={}, 4-thread={}",
        result_1.iterations, result_4.iterations
    );

    assert_eq!(
        result_1.final_lb.to_bits(),
        result_4.final_lb.to_bits(),
        "final_lb must be bit-identical: 1-thread={:.17e}, 4-thread={:.17e}",
        result_1.final_lb,
        result_4.final_lb
    );

    assert_eq!(
        result_1.final_ub.to_bits(),
        result_4.final_ub.to_bits(),
        "final_ub must be bit-identical: 1-thread={:.17e}, 4-thread={:.17e}",
        result_1.final_ub,
        result_4.final_ub
    );

    assert_eq!(
        result_1.final_gap.to_bits(),
        result_4.final_gap.to_bits(),
        "final_gap must be bit-identical: 1-thread={:.17e}, 4-thread={:.17e}",
        result_1.final_gap,
        result_4.final_gap
    );

    // Compare FCF cut pools stage by stage.
    // The FCF holds n_stages - 1 non-trivial pools (the last stage in Finite
    // horizon mode has no successor, so pool[n_stages-1] may be empty).
    assert_eq!(
        fcf_1.pools.len(),
        fcf_4.pools.len(),
        "FCF pool count must be identical"
    );

    for t in 0..fcf_1.pools.len() {
        let pool_1 = &fcf_1.pools[t];
        let pool_4 = &fcf_4.pools[t];

        assert_eq!(
            pool_1.populated_count, pool_4.populated_count,
            "populated_count mismatch at stage {t}: 1-thread={}, 4-thread={}",
            pool_1.populated_count, pool_4.populated_count
        );

        for s in 0..pool_1.populated_count {
            assert_eq!(
                pool_1.active[s], pool_4.active[s],
                "active flag mismatch at stage {t}, slot {s}"
            );

            assert_eq!(
                pool_1.intercepts[s].to_bits(),
                pool_4.intercepts[s].to_bits(),
                "intercept mismatch at stage {t}, slot {s}: \
                 1-thread={:.17e}, 4-thread={:.17e}",
                pool_1.intercepts[s],
                pool_4.intercepts[s]
            );

            assert_eq!(
                pool_1.coefficients[s].len(),
                pool_4.coefficients[s].len(),
                "coefficient length mismatch at stage {t}, slot {s}"
            );

            for (c, (&coeff_1, &coeff_4)) in pool_1.coefficients[s]
                .iter()
                .zip(pool_4.coefficients[s].iter())
                .enumerate()
            {
                assert_eq!(
                    coeff_1.to_bits(),
                    coeff_4.to_bits(),
                    "coefficient mismatch at stage {t}, slot {s}, dim {c}: \
                     1-thread={coeff_1:.17e}, 4-thread={coeff_4:.17e}"
                );
            }
        }
    }
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

    assert_eq!(
        costs_1.len(),
        costs_4.len(),
        "cost buffer length must be identical: 1-worker={}, 4-worker={}",
        costs_1.len(),
        costs_4.len()
    );

    for (idx, ((id_1, cost_1, cats_1), (id_4, cost_4, cats_4))) in
        costs_1.iter().zip(costs_4.iter()).enumerate()
    {
        assert_eq!(
            id_1, id_4,
            "scenario_id mismatch at position {idx}: 1-worker={id_1}, 4-worker={id_4}"
        );

        assert_eq!(
            cost_1.to_bits(),
            cost_4.to_bits(),
            "total_cost mismatch at scenario {id_1}: \
             1-worker={cost_1:.17e}, 4-worker={cost_4:.17e}"
        );

        assert_eq!(
            cats_1.resource_cost.to_bits(),
            cats_4.resource_cost.to_bits(),
            "resource_cost mismatch at scenario {id_1}"
        );
        assert_eq!(
            cats_1.recourse_cost.to_bits(),
            cats_4.recourse_cost.to_bits(),
            "recourse_cost mismatch at scenario {id_1}"
        );
        assert_eq!(
            cats_1.violation_cost.to_bits(),
            cats_4.violation_cost.to_bits(),
            "violation_cost mismatch at scenario {id_1}"
        );
        assert_eq!(
            cats_1.regularization_cost.to_bits(),
            cats_4.regularization_cost.to_bits(),
            "regularization_cost mismatch at scenario {id_1}"
        );
        assert_eq!(
            cats_1.imputed_cost.to_bits(),
            cats_4.imputed_cost.to_bits(),
            "imputed_cost mismatch at scenario {id_1}"
        );
    }
}
