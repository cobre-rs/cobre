//! End-to-end integration test for the Phase 7 train + simulate + write cycle.
//!
//! Exercises the full pipeline: training loop → `build_training_output` →
//! `write_policy_checkpoint` → `simulate` → `write_results`.

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

use cobre_io::{
    Config, PolicyCheckpointMetadata, PolicyCutRecord, SimulationOutput, StageCutsPayload,
    write_policy_checkpoint, write_results,
};
use cobre_sddp::{
    EntityCounts, FutureCostFunction, HorizonMode, InflowNonNegativityMethod, PatchBuffer,
    RiskMeasure, SimulationConfig, SimulationOutputSpec, SolverWorkspace, StageContext,
    StageIndexer, StoppingMode, StoppingRule, StoppingRuleSet, TrainingConfig, TrainingContext,
    build_training_output, simulate, train,
};

/// Single-rank communicator for testing.
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

/// Mock solver that cycles through objectives on each `solve` call.
struct MockSolver {
    objectives: Vec<f64>,
    call_count: usize,
}

impl MockSolver {
    fn with_fixed(objective: f64) -> Self {
        Self {
            objectives: vec![objective],
            call_count: 0,
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

#[allow(clippy::cast_possible_wrap)]
fn make_stochastic_context(n_stages: usize, n_openings: usize) -> StochasticContext {
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

    let system = cobre_core::SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .build()
        .unwrap();

    build_stochastic_context(&system, 42, &[], &[], None).unwrap()
}

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
        let base_rows = vec![2usize; n_stages];
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

fn make_config() -> Config {
    use cobre_io::config::{
        CheckpointingConfig, CutSelectionConfig, ExportsConfig, InflowNonNegativityConfig,
        ModelingConfig, PolicyConfig, SimulationConfig as IoSimulationConfig,
        SimulationSamplingConfig, StoppingRuleConfig, TrainingConfig as IoTrainingConfig,
        TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig::default(),
        },
        training: IoTrainingConfig {
            enabled: true,
            seed: None,
            forward_passes: Some(1),
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 3 }]),
            stopping_mode: "any".to_string(),
            cut_formulation: None,
            forward_pass: None,
            cut_selection: CutSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig {
            path: "./policy".to_string(),
            mode: "fresh".to_string(),
            validate_compatibility: true,
            checkpointing: CheckpointingConfig::default(),
        },
        simulation: IoSimulationConfig {
            enabled: false,
            num_scenarios: 0,
            policy_type: "outer".to_string(),
            output_path: None,
            output_mode: None,
            io_channel_capacity: 64,
            sampling_scheme: SimulationSamplingConfig::default(),
        },
        exports: ExportsConfig::default(),
        estimation: cobre_io::EstimationConfig::default(),
    }
}

fn make_system() -> cobre_core::System {
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
        },
    };

    // Two stages are needed for the 2-stage fixture system.
    let make_stage = |idx: usize| {
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };
        Stage {
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
        }
    };

    let stages: Vec<_> = (0..2usize).map(make_stage).collect();

    let inflow_models: Vec<InflowModel> = (0..2usize)
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

    cobre_core::SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .build()
        .unwrap()
}

#[test]
fn train_simulate_write_cycle() {
    let fx = Fixture::new(2);
    let mut fcf = make_fcf(fx.n_stages);
    let mut solver = MockSolver::with_fixed(100.0);
    let comm = StubComm;

    let (tx, rx) = mpsc::channel::<TrainingEvent>();
    let training_config = TrainingConfig {
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
    };

    let block_counts_per_stage = vec![1usize; fx.n_stages];
    let stage_ctx = StageContext {
        templates: &fx.templates,
        base_rows: &fx.base_rows,
        noise_scale: &[],
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &block_counts_per_stage,
        ncs_max_gen: &[],
    };
    let result = train(
        &mut solver,
        training_config,
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            horizon: &fx.horizon,
            indexer: &fx.indexer,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
        },
        &fx.opening_tree,
        &fx.risk_measures,
        iteration_limit(3),
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
    )
    .expect("train must succeed");

    assert_eq!(result.result.iterations, 3);

    let events: Vec<TrainingEvent> = rx.try_iter().collect();

    let training_output = build_training_output(&result.result, &events, &fcf);

    assert_eq!(training_output.convergence_records.len(), 3);

    let tmp = tempfile::tempdir().expect("tempdir must succeed");
    let policy_dir = tmp.path().join("policy");

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

    let policy_metadata = PolicyCheckpointMetadata {
        version: "1.0.0".to_string(),
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: "2026-03-08T00:00:00Z".to_string(),
        completed_iterations: result.result.iterations as u32,
        final_lower_bound: result.result.final_lb,
        best_upper_bound: Some(result.result.final_ub),
        state_dimension: fcf.state_dimension as u32,
        num_stages: fx.n_stages as u32,
        config_hash: "test-config-hash".to_string(),
        system_hash: "test-system-hash".to_string(),
        max_iterations: 3,
        forward_passes: 1,
        warm_start_cuts: 0,
        rng_seed: 42,
    };

    write_policy_checkpoint(&policy_dir, &stage_cuts_payloads, &[], &policy_metadata)
        .expect("write_policy_checkpoint must succeed");

    let sim_solver = MockSolver::with_fixed(100.0);
    let sim_comm = StubComm;

    let sim_config = SimulationConfig {
        n_scenarios: 2,
        io_channel_capacity: 4,
    };

    let entity_counts = EntityCounts {
        hydro_ids: vec![1],
        hydro_productivities: vec![1.0],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![0],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let (result_tx, result_rx) = mpsc::sync_channel(4);

    let io_thread = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

    let mut sim_workspaces = vec![SolverWorkspace::new(
        sim_solver,
        PatchBuffer::new(fx.indexer.hydro_count, fx.indexer.max_par_order, 0, 0),
        fx.indexer.n_state,
        fx.indexer.hydro_count,
        fx.indexer.max_par_order,
        0,
        0,
    )];

    simulate(
        &mut sim_workspaces,
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
        },
        &fcf,
        &TrainingContext {
            horizon: &fx.horizon,
            indexer: &fx.indexer,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
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
            hydro_productivities_per_stage: &vec![vec![1.0]; fx.n_stages],
            event_sender: None,
        },
        &[],
        &sim_comm,
    )
    .expect("simulate must succeed");

    drop(result_tx);

    let simulation_results = io_thread.join().expect("I/O thread must not panic");

    assert_eq!(simulation_results.len(), 2);

    let sim_output = SimulationOutput {
        n_scenarios: 2,
        completed: 2,
        failed: 0,
        total_time_ms: 0,
        partitions_written: vec![],
    };

    let system = make_system();
    let config = make_config();
    let output_dir = tmp.path();

    write_results(
        output_dir,
        &training_output,
        Some(&sim_output),
        &system,
        &config,
    )
    .expect("write_results must succeed");

    let convergence_path = output_dir.join("training/convergence.parquet");
    assert!(convergence_path.is_file());
    {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        let file = std::fs::File::open(&convergence_path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let total_rows: usize = reader
            .map(|b| b.expect("batch must be Ok").num_rows())
            .sum();
        assert_eq!(total_rows, 3);
    }

    assert!(
        output_dir
            .join("training/timing/iterations.parquet")
            .is_file()
    );

    let manifest_path = output_dir.join("training/_manifest.json");
    assert!(manifest_path.is_file());
    {
        let content = std::fs::read_to_string(&manifest_path).unwrap();
        let value: serde_json::Value =
            serde_json::from_str(&content).expect("_manifest.json must be valid JSON");
        assert_eq!(value["status"].as_str(), Some("complete"));
    }

    let metadata_path = output_dir.join("training/metadata.json");
    assert!(metadata_path.is_file());
    {
        let content = std::fs::read_to_string(&metadata_path).unwrap();
        let value: serde_json::Value =
            serde_json::from_str(&content).expect("metadata.json must be valid JSON");
        assert_eq!(value["problem_dimensions"]["num_hydros"].as_u64(), Some(1));
    }

    assert!(output_dir.join("training/_SUCCESS").is_file());

    let codes_path = output_dir.join("training/dictionaries/codes.json");
    assert!(codes_path.is_file());
    {
        let content = std::fs::read_to_string(&codes_path).unwrap();
        let _value: serde_json::Value =
            serde_json::from_str(&content).expect("codes.json must be valid JSON");
    }

    let sim_manifest_path = output_dir.join("simulation/_manifest.json");
    assert!(sim_manifest_path.is_file());

    assert!(output_dir.join("simulation/_SUCCESS").is_file());

    let policy_meta_path = policy_dir.join("metadata.json");
    assert!(policy_meta_path.is_file());
    {
        let content = std::fs::read_to_string(&policy_meta_path).unwrap();
        let value: serde_json::Value =
            serde_json::from_str(&content).expect("policy/metadata.json must be valid JSON");
        assert_eq!(value["completed_iterations"].as_u64(), Some(3));
    }

    let stage_bin_path = policy_dir.join("cuts/stage_000.bin");
    assert!(stage_bin_path.is_file());
    {
        let metadata = std::fs::metadata(&stage_bin_path).unwrap();
        assert!(metadata.len() > 0);
    }

    assert!(policy_dir.join("basis").is_dir());
}

/// Mock solver that returns a configurable primal vector sized to match a
/// real LP template. Used to verify the extraction path reads slack columns.
struct SizedMockSolver {
    primal: Vec<f64>,
    dual: Vec<f64>,
}

impl SizedMockSolver {
    fn new(num_cols: usize, num_rows: usize) -> Self {
        Self {
            primal: vec![0.0; num_cols],
            dual: vec![0.0; num_rows],
        }
    }

    fn set_primal(&mut self, index: usize, value: f64) {
        self.primal[index] = value;
    }
}

impl SolverInterface for SizedMockSolver {
    fn load_model(&mut self, template: &StageTemplate) {
        self.primal.resize(template.num_cols, 0.0);
        self.dual.resize(template.num_rows, 0.0);
    }

    fn add_rows(&mut self, cuts: &RowBatch) {
        self.dual.resize(self.dual.len() + cuts.num_rows, 0.0);
    }

    fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

    fn solve(&mut self) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
        Ok(cobre_solver::SolutionView {
            objective: 1000.0,
            primal: &self.primal,
            dual: &self.dual,
            reduced_costs: &self.primal,
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
        "SizedMockSolver"
    }
}

/// Build a 1-hydro, 1-bus system with `min_outflow_m3s` > 0 for integration testing.
#[allow(clippy::cast_possible_wrap)]
fn make_min_outflow_system() -> cobre_core::System {
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::InflowModel;
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractStageBounds, HydroStageBounds,
        HydroStagePenalties, LineStageBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingStageBounds, ResolvedBounds,
        ResolvedPenalties, ThermalStageBounds,
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

    let hydro = Hydro {
        id: EntityId(1),
        name: "H1".to_string(),
        bus_id: EntityId(0),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 50.0,
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
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            fpha_turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 5000.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
        },
    };

    let n_stages = 2;
    let make_stage = |idx: usize| {
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };
        Stage {
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
        }
    };

    let stages: Vec<_> = (0..n_stages).map(make_stage).collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(1),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        })
        .collect();

    let load_models: Vec<cobre_core::scenario::LoadModel> = (0..n_stages)
        .map(|i| cobre_core::scenario::LoadModel {
            bus_id: EntityId(0),
            stage_id: i as i32,
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
            n_stages,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                min_outflow_m3s: 50.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                max_diversion_m3s: None,
                filling_inflow_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
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
            n_stages,
        },
        &PenaltiesDefaults {
            hydro: HydroStagePenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 5000.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
            },
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

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

    cobre_core::SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .correlation(correlation)
        .build()
        .unwrap()
}

/// Integration test: simulation with `min_outflow_m3s` > 0 produces non-zero
/// `outflow_slack_below_m3s` when the primal vector has non-zero slack values.
///
/// This test uses the real LP template builder (`build_stage_templates`) to
/// construct correctly-sized templates, then a `SizedMockSolver` whose primal
/// vector has sentinel non-zero values at the `outflow_below_slack` column.
/// The simulation extracts results from the primal, and we verify that the
/// operational violation slack propagates correctly to the output.
#[test]
fn simulation_min_outflow_slack_extracted_from_primal() {
    use cobre_sddp::lp_builder::build_stage_templates;

    let system = make_min_outflow_system();
    let n_stages = 2;

    let stochastic = make_stochastic_context(n_stages, 1);

    let hydro_models =
        cobre_sddp::hydro_models::PrepareHydroModelsResult::default_from_system(&system);

    let templates_result = build_stage_templates(
        &system,
        &InflowNonNegativityMethod::None,
        stochastic.par(),
        stochastic.normal(),
        &hydro_models.production,
        &hydro_models.evaporation,
    )
    .expect("build_stage_templates must succeed");

    let t0 = &templates_result.templates[0];

    // Build indexer matching the template layout.
    let indexer = StageIndexer::with_equipment(
        &cobre_sddp::indexer::EquipmentCounts {
            hydro_count: 1,
            max_par_order: 0,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 1,
            n_blks: 1,
            has_inflow_penalty: false,
            max_deficit_segments: 1,
        },
        &cobre_sddp::indexer::FphaColumnLayout {
            hydro_indices: vec![],
            planes_per_hydro: vec![],
        },
    );

    assert!(indexer.has_operational_violations);
    assert!(!indexer.outflow_below_slack.is_empty());

    let slack_col = indexer.outflow_below_slack.start;
    assert!(
        slack_col < t0.num_cols,
        "outflow_below_slack col {} must be within template cols {}",
        slack_col,
        t0.num_cols
    );
    assert_eq!(
        t0.col_upper[slack_col],
        f64::INFINITY,
        "outflow_below_slack col_upper must be +inf when min_outflow > 0"
    );

    let min_outflow_row = indexer.min_outflow_rows.start;
    let total_hours = 744.0_f64;
    let m3s_to_hm3 = 3_600.0 / 1_000_000.0;
    let zeta = total_hours * m3s_to_hm3;
    // Per-block formulation: row_lower is in rate units (m3/s), not volume.
    let expected_row_lower = 50.0;
    assert!(
        (t0.row_lower[min_outflow_row] - expected_row_lower).abs() < 1e-10,
        "min_outflow row_lower = {}, expected {} (rate units m3/s)",
        t0.row_lower[min_outflow_row],
        expected_row_lower
    );

    // Inject a sentinel non-zero value at the slack column in the primal.
    // Per-block: the slack column value IS in m3/s, no conversion needed.
    let sentinel_m3s = 5.0;
    let expected_slack_m3s = sentinel_m3s;
    let mut solver = SizedMockSolver::new(t0.num_cols, t0.num_rows);
    solver.set_primal(slack_col, sentinel_m3s);

    let templates = vec![t0.clone(); n_stages];
    let base_rows = vec![templates_result.base_rows[0]; n_stages];
    let initial_state = vec![100.0_f64; indexer.n_state];
    let opening_tree = make_opening_tree(1);
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };

    let mut fcf = make_fcf(n_stages);

    let block_counts = vec![1usize; n_stages];
    let stage_ctx = StageContext {
        templates: &templates,
        base_rows: &base_rows,
        noise_scale: &templates_result.noise_scale,
        n_hydros: 1,
        n_load_buses: 0,
        load_balance_row_starts: &templates_result.load_balance_row_starts,
        load_bus_indices: &[],
        block_counts_per_stage: &block_counts,
        ncs_max_gen: &[],
    };

    let training_config = TrainingConfig {
        forward_passes: 1,
        max_iterations: 1,
        checkpoint_interval: None,
        warm_start_cuts: 0,
        event_sender: None,
        cut_activity_tolerance: 0.0,
        n_fwd_threads: 1,
        max_blocks: 1,
        cut_selection: None,
        shutdown_flag: None,
        start_iteration: 0,
    };

    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    train(
        &mut solver,
        training_config,
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
        },
        &opening_tree,
        &risk_measures,
        iteration_limit(1),
        &StubComm,
        || Ok(SizedMockSolver::new(t0.num_cols, t0.num_rows)),
    )
    .expect("training must succeed");

    let sim_config = SimulationConfig {
        n_scenarios: 1,
        io_channel_capacity: 4,
    };

    let entity_counts = EntityCounts {
        hydro_ids: vec![1],
        hydro_productivities: vec![1.0],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![0],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let zeta_per_stage = vec![zeta; n_stages];
    let block_hours_per_stage = vec![vec![total_hours]; n_stages];
    let hydro_productivities_per_stage = vec![vec![1.0]; n_stages];

    let (result_tx, result_rx) = mpsc::sync_channel(4);

    let io_thread = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

    let mut sim_solver = SizedMockSolver::new(t0.num_cols, t0.num_rows);
    sim_solver.set_primal(slack_col, sentinel_m3s);

    let mut sim_workspaces = vec![SolverWorkspace::new(
        sim_solver,
        PatchBuffer::new(indexer.hydro_count, indexer.max_par_order, 0, 0),
        indexer.n_state,
        indexer.hydro_count,
        indexer.max_par_order,
        0,
        0,
    )];

    simulate(
        &mut sim_workspaces,
        &stage_ctx,
        &fcf,
        &TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
        },
        &sim_config,
        SimulationOutputSpec {
            result_tx: &result_tx,
            zeta_per_stage: &zeta_per_stage,
            block_hours_per_stage: &block_hours_per_stage,
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &[],
            ncs_col_starts: &[],
            n_ncs_per_stage: &[],
            ncs_entity_ids_per_stage: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hydro_productivities_per_stage,
            event_sender: None,
        },
        &[],
        &StubComm,
    )
    .expect("simulate must succeed");

    drop(result_tx);

    let results = io_thread.join().expect("I/O thread must not panic");
    assert_eq!(results.len(), 1, "expected exactly 1 scenario result");

    let scenario = &results[0];
    let mut found_nonzero_slack = false;
    for stage_result in &scenario.stages {
        for hydro_result in &stage_result.hydros {
            if (hydro_result.outflow_slack_below_m3s - expected_slack_m3s).abs() < 1e-6 {
                found_nonzero_slack = true;
            }
        }
    }
    assert!(
        found_nonzero_slack,
        "Expected at least one hydro result with outflow_slack_below_m3s = {expected_slack_m3s:.6} \
         (sentinel_m3s={sentinel_m3s} / zeta={zeta}), but all were zero. \
         This indicates the extraction path does not read from the slack column.",
    );
}
