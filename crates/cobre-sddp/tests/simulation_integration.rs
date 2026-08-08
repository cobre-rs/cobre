//! End-to-end integration test for the train + simulate + write cycle.

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
// `..Default::default()` in the make_* Spec calls is the intentional future-field
// seam from `common::builders` — a no-op today, not dead code.
#![allow(clippy::needless_update)]

use std::collections::{BTreeMap, HashMap};
use std::sync::mpsc;

use chrono::NaiveDate;
use cobre_core::{
    DeficitSegment, EntityId, SystemBuilder, TrainingEvent,
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, LoadModel,
        SamplingScheme,
    },
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_solver::{
    ActiveProfile, Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
};
use cobre_stochastic::{
    ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context,
};

use cobre_io::{
    Config, EstimationConfig, MetadataSimulationSolveStats, PolicyCheckpointMetadata,
    PolicyCutRecord, PolicyMode, SimulationOutput, StageCutsPayload, write_policy_checkpoint,
    write_results,
};
use cobre_sddp::{
    Phase, PrepareHydroModelsResult, ResolvedParameters, SolverProfiles, StoppingMode,
    StoppingRule, StoppingRuleSet, TrainingConfig, build_training_output,
    config::{CutManagementConfig, EventConfig, LoopConfig},
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    energy_conversion::{EnergyConversion, EnergyConversionSet},
    horizon_mode::HorizonMode,
    indexer::{CutStateProjection, StateSpace, StudyDimensions},
    inflow_method::InflowNonNegativityMethod,
    lp_builder::PatchBuffer,
    risk_measure::RiskMeasure,
    simulate,
    simulation::{EntityCounts, SimulationConfig, SimulationOutputSpec},
    train,
    workspace::{SolverWorkspace, WorkspaceSizing},
};

mod common;
use common::StubComm;
use common::builders::{BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage};

/// Mirrors the gated `test_support::state_layout_for` via the public
/// [`StateSpace::new`] constructor: this external test crate cannot see the parent
/// crate's `#[cfg(test)]` surface, so it rebuilds byte-identical patch columns here.
fn state_layout_for(hydro_count: usize, max_par_order: usize) -> StateSpace {
    StateSpace::new(
        hydro_count,
        max_par_order,
        0,
        Vec::new(),
        0,
        0,
        vec![],
        &vec![max_par_order; hydro_count],
    )
}

/// Carries the non-state study shape directly: this external test crate cannot see
/// the parent crate's `#[cfg(test)]`/`test-support` surface. `max_deficit_segments`
/// is `1`; `n_pumping`/`has_ncs`/anticipated are empty for these fixtures.
fn study_dims_for(
    n_thermals: usize,
    n_lines: usize,
    n_buses: usize,
    hydro_count: usize,
    has_inflow_penalty: bool,
) -> StudyDimensions {
    StudyDimensions {
        n_thermals,
        n_lines,
        n_buses,
        max_deficit_segments: 1,
        has_ncs: false,
        has_inflow_penalty,
        has_withdrawal: hydro_count > 0,
        has_operational_violations: hydro_count != 0,
        anticipated_thermal_indices: vec![],
        n_pumping: 0,
    }
}

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
    type Profile = ActiveProfile;

    fn apply_profile(&mut self, _profile: &ActiveProfile) {}

    fn solver_name_version(&self) -> String {
        "MockSolver 0.0.0".to_string()
    }
    fn load_model(&mut self, _template: &StageTemplate) {}
    fn add_rows(&mut self, _cuts: &RowBatch) {}
    fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn solve(
        &mut self,
        _basis: Option<&Basis>,
    ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
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

    fn get_basis(&mut self, out: &mut Basis) {
        cobre_sddp::test_support::fill_consistent_basis(out);
    }

    fn statistics(&self) -> SolverStatistics {
        SolverStatistics::default()
    }

    fn statistics_into(&self, out: &mut SolverStatistics) {
        *out = self.statistics();
    }

    fn name(&self) -> &'static str {
        "MockIntegration"
    }
}

#[allow(clippy::cast_possible_wrap)]
fn make_stochastic_context(n_stages: usize, n_openings: usize) -> StochasticContext {
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::InflowModel;

    let bus = make_bus(
        EntityId(0),
        BusSpec {
            name: "B0".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );
    let hydro = make_hydro(
        EntityId(1),
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(0),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
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
                turbined_cost: 0.0,
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
            ..Default::default()
        },
    );

    let stages: Vec<Stage> = (0..n_stages)
        .map(|idx| {
            make_stage(
                idx,
                StageSpec {
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
                    ..Default::default()
                },
            )
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(1),
            stage_id: i as i32,
            mean_m3s: 100.0,
            std_m3s: 30.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
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
    state: StateSpace,
    initial_state: Vec<f64>,
    stochastic: StochasticContext,
    horizon: HorizonMode,
    risk_measures: Vec<RiskMeasure>,
}

const FCF_CAPACITY_ITERATIONS: u64 = 50;

impl Fixture {
    fn new(n_stages: usize) -> Self {
        let state = state_layout_for(1, 0);
        let templates = vec![minimal_template(); n_stages];
        // base_row: the AR-dynamics row offset is 1 (1 dual-relevant row)
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; state.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        Self {
            n_stages,
            templates,
            base_rows,
            state,
            initial_state,
            stochastic,
            horizon,
            risk_measures,
        }
    }
}

fn make_config() -> Config {
    use cobre_io::config::{
        CheckpointingConfig, ExportsConfig, InflowNonNegativityConfig, ModelingConfig,
        PolicyConfig, RowSelectionConfig, SimulationConfig as IoSimulationConfig,
        StoppingRuleConfig, TrainingConfig as IoTrainingConfig, TrainingSolverConfig,
        UpperBoundEvaluationConfig,
    };
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig::default(),
            cost_scale_factor: None,
        },
        training: IoTrainingConfig {
            enabled: true,
            tree_seed: None,
            forward_passes: Some(1),
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 3 }]),
            stopping_mode: "any".to_string(),
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            parallelism: cobre_io::config::ParallelismConfig::default(),
            scenario_source: None,
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig {
            path: "./policy".to_string(),
            mode: PolicyMode::Fresh,
            checkpointing: CheckpointingConfig::default(),
            boundary: None,
        },
        simulation: IoSimulationConfig {
            enabled: false,
            num_scenarios: 0,
            io_channel_capacity: 64,
            scenario_source: None,
            solver: None,
        },
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

fn make_system() -> cobre_core::System {
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::InflowModel;

    let bus = make_bus(
        EntityId(0),
        BusSpec {
            name: "B0".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );
    let hydro = make_hydro(
        EntityId(1),
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(0),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
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
                turbined_cost: 0.0,
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
            ..Default::default()
        },
    );

    let stages: Vec<_> = (0..2usize)
        .map(|idx| {
            make_stage(
                idx,
                StageSpec {
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
                    ..Default::default()
                },
            )
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..2usize)
        .map(|i| InflowModel {
            hydro_id: EntityId(1),
            stage_id: i as i32,
            mean_m3s: 100.0,
            std_m3s: 30.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
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

    SystemBuilder::new()
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
            event_sender: Some(tx),
            checkpoint_interval: None,
            shutdown_flag: None,
            export_states: false,
        },
    };

    let block_counts_per_stage = vec![1usize; fx.n_stages];
    let stage_ctx = StageContext {
        geometry_per_stage: &[],
        templates: &fx.templates,
        base_rows: &fx.base_rows,
        noise_scale: &[],
        n_hydros: 0,
        cost_scale_factor: 1_000_000.0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &block_counts_per_stage,
        ncs_col_starts: &[],
        n_ncs: 0,
        ncs_stochastic_dense_col: &[],
        ncs_stochastic_windows: &[],
        anticipated_windows: &[],
        study_stage_ids: &[],
        ncs_max_gen: &[],
        ncs_allow_curtailment: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
    };
    let cut_state_layouts = all_enabled_cut_state_layouts(&fx.state, fx.n_stages);
    let study_dims = study_dims_for(0, 0, 0, 0, false);
    let training_context = TrainingContext {
        horizon: &fx.horizon,
        state: &fx.state,
        cut_state_layouts: &cut_state_layouts,
        study_dims: &study_dims,
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
        lag_accum_seed: &[],
        lag_weight_seed: &[],
        dcs: None,
        stages: &[],
    };
    let result = train(
        &mut solver,
        training_config,
        &mut fcf,
        &stage_ctx,
        &training_context,
        &comm,
        || Ok(MockSolver::with_fixed(100.0)),
        None,
        SolverProfiles::default(),
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
            (0..pool.populated())
                .map(|slot| {
                    let meta = pool.metadata(slot);
                    PolicyCutRecord {
                        cut_id: slot as u64,
                        slot_index: slot as u32,
                        iteration: meta.iteration_generated as u32,
                        forward_pass_index: meta.forward_pass_index,
                        intercept: pool.intercept(slot),
                        coefficients: pool.coefficient_row(slot),
                        is_active: pool.is_active(slot),
                    }
                })
                .collect()
        })
        .collect();

    let active_indices_per_stage: Vec<Vec<u32>> = fcf
        .pools
        .iter()
        .map(|pool| {
            (0..pool.populated())
                .filter(|&slot| pool.is_active(slot))
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
            populated_count: pool.populated() as u32,
            entity_manifest: &[],
        })
        .collect();

    let warm_start_counts: Vec<u32> = fcf.pools.iter().map(|p| p.warm_start_count).collect();
    let policy_metadata = PolicyCheckpointMetadata {
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: "2026-03-08T00:00:00Z".to_string(),
        completed_iterations: result.result.iterations as u32,
        final_lower_bound: result.result.final_lb,
        best_upper_bound: Some(result.result.final_ub),
        state_dimension: fcf.state_dimension as u32,
        num_stages: fx.n_stages as u32,
        max_iterations: 3,
        forward_passes: 1,
        warm_start_cuts: warm_start_counts.iter().copied().max().unwrap_or(0),
        warm_start_counts,
        rng_seed: 42,
        total_visited_states: 0,
        training_block_mode: "parallel".to_string(),
        training_block_mode_per_stage: vec![],
        cost_scale_factor: None,
    };

    write_policy_checkpoint(
        &policy_dir,
        &stage_cuts_payloads,
        &[],
        &policy_metadata,
        &[],
    )
    .expect("write_policy_checkpoint must succeed");

    let sim_solver = MockSolver::with_fixed(100.0);
    let sim_comm = StubComm;

    let sim_config = SimulationConfig {
        n_scenarios: 2,
        io_channel_capacity: 4,
        profile: Phase::Simulation.profile(),
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
        0,
        0,
        sim_solver,
        PatchBuffer::new(fx.state.hydro_count, fx.state.max_par_order, 0, 0, 0, 0, 0),
        fx.state.n_state,
        WorkspaceSizing {
            hydro_count: fx.state.hydro_count,
            max_par_order: fx.state.max_par_order,
            n_load_buses: 0,
            max_blocks: 0,
            downstream_par_order: 0,
            ..WorkspaceSizing::default()
        },
    )];

    let zero_ec = EnergyConversion {
        equivalent_productivity_mw_per_m3s: 0.0,
        reference_volume_hm3: 0.0,
        reference_outflow_m3s: 0.0,
    };
    let ec = EnergyConversionSet::new(
        vec![vec![zero_ec; fx.n_stages]; 1],
        vec![vec![0.0_f64; fx.n_stages]; 1],
        1,
        fx.n_stages,
    );

    simulate(
        &mut sim_workspaces,
        &StageContext {
            geometry_per_stage: &[],
            templates: &fx.templates,
            base_rows: &fx.base_rows,
            noise_scale: &[],
            n_hydros: 0,
            cost_scale_factor: 1_000_000.0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            n_ncs: 0,
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            anticipated_windows: &[],
            study_stage_ids: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        },
        &fcf,
        &training_context,
        &sim_config,
        SimulationOutputSpec {
            result_tx: &result_tx,
            zeta_per_stage: &[],
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &[],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &[],
            ncs_col_starts: &[],
            n_ncs: 0,
            pumping_col_starts: &[],
            n_pumping: 0,
            geometry_per_stage: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &[],
            contract_is_import: &[],
            ncs_entity_ids_per_stage: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &vec![vec![1.0]; fx.n_stages],
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
        },
        None,
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
        cost: None,
        solve_stats: MetadataSimulationSolveStats::default(),
    };

    let system = make_system();
    let config = make_config();
    let output_dir = tmp.path();

    let output_ctx = cobre_io::OutputContext {
        hostname: "test-host".to_string(),
        solver: "highs".to_string(),
        solver_version: None,
        started_at: "2026-01-17T08:00:00Z".to_string(),
        completed_at: "2026-01-17T12:30:00Z".to_string(),
        distribution: cobre_io::DistributionInfo {
            backend: "local".to_string(),
            world_size: 1,
            ranks_participated: 1,
            num_nodes: 1,
            threads_per_rank: 1,
            mpi_library: None,
            mpi_standard: None,
            thread_level: None,
            slurm_job_id: None,
            hosts: Vec::new(),
            rank_affinity: Vec::new(),
        },
        setup: None,
        production_fit_deviation: None,
    };
    write_results(
        output_dir,
        &training_output,
        Some(&sim_output),
        &system,
        &config,
        &output_ctx,
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

    let metadata_path = output_dir.join("training/metadata.json");
    assert!(metadata_path.is_file());
    {
        let content = std::fs::read_to_string(&metadata_path).unwrap();
        let value: serde_json::Value =
            serde_json::from_str(&content).expect("metadata.json must be valid JSON");
        assert_eq!(value["status"].as_str(), Some("complete"));
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

    let sim_metadata_path = output_dir.join("simulation/metadata.json");
    assert!(sim_metadata_path.is_file());

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
    type Profile = ActiveProfile;

    fn apply_profile(&mut self, _profile: &ActiveProfile) {}

    fn solver_name_version(&self) -> String {
        "MockSolver 0.0.0".to_string()
    }
    fn load_model(&mut self, template: &StageTemplate) {
        self.primal.resize(template.num_cols, 0.0);
        self.dual.resize(template.num_rows, 0.0);
    }

    fn add_rows(&mut self, cuts: &RowBatch) {
        self.dual.resize(self.dual.len() + cuts.num_rows, 0.0);
    }

    fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn solve(
        &mut self,
        _basis: Option<&Basis>,
    ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
        Ok(cobre_solver::SolutionView {
            objective: 1000.0,
            primal: &self.primal,
            dual: &self.dual,
            reduced_costs: &self.primal,
            iterations: 0,
            solve_time_seconds: 0.0,
        })
    }

    fn get_basis(&mut self, out: &mut Basis) {
        cobre_sddp::test_support::fill_consistent_basis(out);
    }

    fn statistics(&self) -> SolverStatistics {
        SolverStatistics::default()
    }

    fn statistics_into(&self, out: &mut SolverStatistics) {
        *out = self.statistics();
    }

    fn name(&self) -> &'static str {
        "SizedMockSolver"
    }
}

/// Build a 1-hydro, 1-bus system with `min_outflow_m3s` > 0 for integration testing.
#[allow(clippy::cast_possible_wrap)]
fn make_min_outflow_system() -> cobre_core::System {
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::InflowModel;
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, HydroBlockBounds,
        HydroStageBounds, HydroStagePenalties, LineBlockBounds, LineStagePenalties,
        NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds,
        ResolvedBounds, ResolvedPenalties, ThermalBlockBounds, ThermalStageBounds,
    };

    let bus = make_bus(
        EntityId(0),
        BusSpec {
            name: "B0".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let hydro = make_hydro(
        EntityId(1),
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(0),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 50.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
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
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 5000.0,
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
            ..Default::default()
        },
    );

    let n_stages = 2;
    let stages: Vec<_> = (0..n_stages)
        .map(|idx| {
            make_stage(
                idx,
                StageSpec {
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
                    ..Default::default()
                },
            )
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(1),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
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
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                min_outflow_m3s: 50.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                max_diversion_m3s: None,
            },
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
            },
            line_block: LineBlockBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping_block: PumpingBlockBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract_block: ContractBlockBounds {
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
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 5000.0,
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
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    };

    SystemBuilder::new()
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

/// A sentinel value injected at the `outflow_below_slack` primal column (via a
/// `SizedMockSolver` over a real `build_stage_templates_resolving_layout` template) must propagate
/// to `outflow_slack_below_m3s` in the simulation output.
#[test]
fn simulation_min_outflow_slack_extracted_from_primal() {
    use cobre_sddp::build_stage_templates_resolving_layout;

    let system = make_min_outflow_system();
    let n_stages = 2;

    let stochastic = make_stochastic_context(n_stages, 1);

    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);

    let templates_result = build_stage_templates_resolving_layout(
        &system,
        InflowNonNegativityMethod::None,
        stochastic.par(),
        stochastic.normal(),
        &hydro_models.production,
        &hydro_models.evaporation,
        &ResolvedParameters::default(),
    )
    .expect("build_stage_templates_resolving_layout must succeed");

    let t0 = &templates_result.templates[0];

    let study_dims = study_dims_for(0, 0, 1, 1, false);
    // The operational-violation constraint *row* range is owned by `StageLayout` and
    // pinned by `stage_layout_operational_violation_rows_are_contiguous_blocks`; this
    // end-to-end test covers only the slack-*column* extraction path.
    let geometry = &templates_result.geometry_per_stage[0];
    let state = state_layout_for(1, 0);

    assert!(study_dims.has_operational_violations);
    assert!(!geometry.outflow_below_slack.is_empty());

    let slack_col = geometry.outflow_below_slack.start;
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

    let total_hours = 744.0_f64;
    let m3s_to_hm3 = 3_600.0 / 1_000_000.0;
    let zeta = total_hours * m3s_to_hm3;

    // The slack column value is in m3/s, so no zeta conversion is applied.
    let sentinel_m3s = 5.0;
    let expected_slack_m3s = sentinel_m3s;
    let mut solver = SizedMockSolver::new(t0.num_cols, t0.num_rows);
    solver.set_primal(slack_col, sentinel_m3s);

    let templates = vec![t0.clone(); n_stages];
    let base_rows = vec![templates_result.base_rows[0]; n_stages];
    // Every stage clones `t0`, so stage-0 geometry must be replicated across all
    // stages for extraction to read the stage-correct slack columns.
    let equipment_geometry = vec![templates_result.geometry_per_stage[0].clone(); n_stages];
    let initial_state = vec![100.0_f64; state.n_state];
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };

    let mut fcf = make_fcf(n_stages);

    let block_counts = vec![1usize; n_stages];
    let stage_ctx = StageContext {
        geometry_per_stage: &[],
        templates: &templates,
        base_rows: &base_rows,
        noise_scale: &templates_result.noise_scale,
        n_hydros: 1,
        cost_scale_factor: 1_000_000.0,
        n_load_buses: 0,
        load_balance_row_starts: &templates_result.load_balance_row_starts,
        load_bus_indices: &[],
        block_counts_per_stage: &block_counts,
        ncs_col_starts: &[],
        n_ncs: 0,
        ncs_stochastic_dense_col: &[],
        ncs_stochastic_windows: &[],
        anticipated_windows: &[],
        study_stage_ids: &[],
        ncs_max_gen: &[],
        ncs_allow_curtailment: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
    };

    let training_config = TrainingConfig {
        loop_config: LoopConfig {
            forward_passes: 1,
            max_iterations: 1,
            start_iteration: 0,
            n_fwd_threads: 1,
            max_blocks: 1,
            stopping_rules: iteration_limit(1),
        },
        cut_management: CutManagementConfig {
            cut_selection: None,
            budget: None,
            cut_activity_tolerance: 0.0,
            warm_start_cuts: 0,
            risk_measures: vec![RiskMeasure::Expectation; n_stages],
        },
        events: EventConfig {
            event_sender: None,
            checkpoint_interval: None,
            shutdown_flag: None,
            export_states: false,
        },
    };

    let cut_state_layouts = all_enabled_cut_state_layouts(&state, n_stages);
    let training_context = TrainingContext {
        horizon: &horizon,
        state: &state,
        cut_state_layouts: &cut_state_layouts,
        study_dims: &study_dims,
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
        lag_accum_seed: &[],
        lag_weight_seed: &[],
        dcs: None,
        stages: &[],
    };
    train(
        &mut solver,
        training_config,
        &mut fcf,
        &stage_ctx,
        &training_context,
        &StubComm,
        || Ok(SizedMockSolver::new(t0.num_cols, t0.num_rows)),
        None,
        SolverProfiles::default(),
    )
    .expect("training must succeed");

    let sim_config = SimulationConfig {
        n_scenarios: 1,
        io_channel_capacity: 4,
        profile: Phase::Simulation.profile(),
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
        0,
        0,
        sim_solver,
        PatchBuffer::new(state.hydro_count, state.max_par_order, 0, 0, 0, 0, 0),
        state.n_state,
        WorkspaceSizing {
            hydro_count: state.hydro_count,
            max_par_order: state.max_par_order,
            n_load_buses: 0,
            max_blocks: 0,
            downstream_par_order: 0,
            ..WorkspaceSizing::default()
        },
    )];

    let zero_ec2 = EnergyConversion {
        equivalent_productivity_mw_per_m3s: 0.0,
        reference_volume_hm3: 0.0,
        reference_outflow_m3s: 0.0,
    };
    let ec2 = EnergyConversionSet::new(
        vec![vec![zero_ec2; n_stages]; 1],
        vec![vec![0.0_f64; n_stages]; 1],
        1,
        n_stages,
    );

    simulate(
        &mut sim_workspaces,
        &stage_ctx,
        &fcf,
        &training_context,
        &sim_config,
        SimulationOutputSpec {
            result_tx: &result_tx,
            zeta_per_stage: &zeta_per_stage,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &block_hours_per_stage,
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &[],
            ncs_col_starts: &[],
            n_ncs: 0,
            pumping_col_starts: &[],
            n_pumping: 0,
            geometry_per_stage: &equipment_geometry,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &[],
            contract_is_import: &[],
            ncs_entity_ids_per_stage: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hydro_productivities_per_stage,
            energy_conversion: &ec2,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
        },
        None,
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

/// Local mirror of the gated `test_support::all_enabled_cut_state_layouts`
/// via the public `CutStateProjection::new`, so this external test crate (which cannot
/// see the parent crate's `#[cfg(test)]` surface) builds the default all-enabled
/// per-pool projection. Every pool projects the full global state, keeping the
/// extracted subgradient bit-identical to the global-loop result.
fn all_enabled_cut_state_layouts(global: &StateSpace, n_stages: usize) -> Vec<CutStateProjection> {
    let full = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    (0..n_stages)
        .map(|_| CutStateProjection::new(global, full))
        .collect()
}
