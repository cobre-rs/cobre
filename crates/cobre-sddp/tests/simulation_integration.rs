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

use cobre_io::config::{SimulationSelection, TrainingSelection};
use std::collections::{BTreeMap, HashMap};
use std::sync::mpsc;

use chrono::NaiveDate;
use cobre_comm::Communicator;
use cobre_core::{
    DeficitSegment, EntityId, SystemBuilder, TrainingEvent,
    entities::bus::Bus,
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
    ActiveProfile, ActiveSolver, Basis, RowBatch, SolverError, SolverInterface, SolverStatistics,
    StageTemplate,
};
use cobre_stochastic::{
    ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context,
};

use cobre_io::output::simulation_writer::{
    ScenarioWritePayload, SimulationParquetWriter, write_scenario_summary,
};
use cobre_io::{
    Config, EstimationConfig, MetadataSimulationSolveStats, PolicyCutRecord, PolicyMode,
    STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, SimulationOutput, StageCutsPayload,
    read_policy_checkpoint, write_policy_checkpoint, write_results,
};
use cobre_sddp::{
    CapturedBasis, Phase, PrepareHydroModelsResult, ResolvedParameters, SimulationSummary,
    SolverProfiles, StoppingMode, StoppingRule, StoppingRuleSet, TrainingConfig,
    aggregate_simulation, build_training_output,
    config::{CutManagementConfig, EventConfig, LoopConfig},
    context::TrainingContext,
    cut::FutureCostFunction,
    energy_conversion::{EnergyConversion, EnergyConversionSet},
    horizon_mode::HorizonMode,
    indexer::{AnticipatedPlants, CutStateProjection, StateSpace, StudyDimensions},
    inflow_method::InflowNonNegativityMethod,
    lead_time::AnticipatedResolution,
    lp::builder::{PatchBuffer, StageGeometry},
    risk_measure::RiskMeasure,
    setup::{
        SimulationEnumeratedRequest, StudySetup,
        node_graph::{NodePos, Traversal},
    },
    simulate,
    simulation::{
        EntityCounts, SimulationConfig, SimulationOutputSpec, SimulationScenarioResult,
        aggregation::GatheredScenarioCosts,
    },
    solver_stats::SolverStatsDelta,
    test_support::{
        StageContextFixture, branching_tree_setup_enumerated, extensive_form_optimum,
        hydro_only_bus_geometry, k_fan_setup_enumerated, node_prefix_counts, node_scenario_count,
        permissive_state_boxes, single_path_enumerated_setup, trunk_fan_setup_enumerated,
        water_binding_external_fan_setup,
    },
    train,
    workspace::{SolverWorkspace, WorkspaceSizing},
};

mod common;
use common::Rank0Of2;
use common::StubComm;
use common::builders::{BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage};

/// Mirrors the gated `test_support::state_layout_for` via the public
/// [`StateSpace::new`] constructor: this external test crate cannot see the parent
/// crate's `#[cfg(test)]` surface, so it rebuilds byte-identical patch columns here.
fn state_layout_for(hydro_count: usize, max_par_order: usize) -> StateSpace {
    StateSpace::new(
        hydro_count,
        max_par_order,
        Vec::new(),
        vec![],
        AnticipatedResolution::default(),
        &vec![max_par_order; hydro_count],
    )
}

/// Carries the non-state study shape directly: this external test crate cannot see
/// the parent crate's `#[cfg(test)]`/`test-support` surface. `max_deficit_segments`
/// is `1`; anticipated is empty for these fixtures.
fn study_dims_for(has_inflow_penalty: bool) -> StudyDimensions {
    StudyDimensions {
        max_deficit_segments: 1,
        inflow_method: if has_inflow_penalty {
            InflowNonNegativityMethod::Penalty
        } else {
            InflowNonNegativityMethod::None
        },
        anticipated_plants: AnticipatedPlants::default(),
        downstream_par_order: 0,
    }
}

fn default_bus() -> Bus {
    make_bus(
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
    )
}

fn default_correlation_model() -> CorrelationModel {
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
    CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
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
            primal: &[0.0; 15],
            dual: &[0.0; 7],
            reduced_costs: &[0.0; 15],
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

    let bus = default_bus();
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

    let correlation = default_correlation_model();

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
    // N=1, L=0 → cols: storage(0), z_inflow(1), storage_in(2), theta(3), then
    // the decoupled hydro+bus equipment padding `test_support::geometry`
    // addresses for N=1 hydro, 1 bus, 1 block (turbine..generation_below_slack,
    // cols 4-14); `MockSolver::solve` mirrors this column count.
    //             rows: z_inflow(0), mock_pin(1), then that same
    //             layout's water_balance/load_balance/oper_violation rows (2-6)
    let num_cols = 15;
    let num_rows = 7;
    let mut col_lower = vec![0.0; num_cols];
    col_lower[1] = f64::NEG_INFINITY;
    let mut objective = vec![0.0; num_cols];
    objective[3] = 1.0;
    StageTemplate {
        num_cols,
        num_rows,
        num_nz: 2,
        col_starts: {
            let mut v = vec![2_i32; num_cols + 1];
            v[0] = 0;
            v[1] = 0;
            v[2] = 1;
            v
        },
        row_indices: vec![0, 1],
        values: vec![1.0, 1.0],
        col_lower,
        col_upper: vec![f64::INFINITY; num_cols],
        objective,
        row_lower: vec![0.0; num_rows],
        row_upper: vec![0.0; num_rows],
        n_state: 1,
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

fn default_single_hydro_entity_counts() -> EntityCounts {
    EntityCounts {
        hydro_ids: vec![1],
        hydro_productivities: vec![1.0],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![0],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    }
}

fn zero_energy_conversion_set(n_stages: usize) -> EnergyConversionSet {
    let zero_ec = EnergyConversion {
        equivalent_productivity_mw_per_m3s: 0.0,
        reference_volume_hm3: 0.0,
        reference_outflow_m3s: 0.0,
    };
    EnergyConversionSet::new(
        vec![vec![zero_ec; n_stages]; 1],
        vec![vec![0.0_f64; n_stages]; 1],
        &cobre_sddp::test_support::minimal_hydros(1),
        n_stages,
    )
}

/// All training parameters for a 2-stage, N=1 toy system.
struct Fixture {
    n_stages: usize,
    templates: Vec<StageTemplate>,
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
        let initial_state = vec![0.0_f64; state.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        Self {
            n_stages,
            templates,
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
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 3 }]),
            stopping_mode: cobre_io::config::StoppingMode::Any,
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            parallelism: cobre_io::config::ParallelismConfig::default(),
            scenario_source: None,
            selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
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
            io_channel_capacity: 64,
            scenario_source: None,
            solver: None,
            selection: Some(SimulationSelection::Sampled { num_scenarios: 0 }),
        },
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

fn make_system() -> cobre_core::System {
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::InflowModel;

    let bus = default_bus();
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

    let correlation = default_correlation_model();

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
            training_enumerated: false,
            max_iterations: 10,
            start_iteration: 0,
            n_fwd_threads: 1,
            stopping_rules: iteration_limit(3),
        },
        cut_management: CutManagementConfig {
            cut_selection: None,
            budget: None,
            cut_activity_tolerance: 0.0,
            risk_measures: fx.risk_measures.clone(),
        },
        events: EventConfig {
            event_sender: Some(tx),
            checkpoint_interval: None,
            shutdown_flag: None,
            export_states: false,
        },
    };

    let geometry = vec![hydro_only_bus_geometry(); fx.n_stages];
    let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
    let stage_ctx_fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
    let stage_ctx = stage_ctx_fixture.ctx();
    let cut_state_layouts = all_enabled_cut_state_layouts(&fx.state, fx.n_stages);
    let study_dims = study_dims_for(false);
    let training_context = TrainingContext {
        node_graph: &cobre_sddp::test_support::chain_node_graph(&fx.stochastic),
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

    let training_output = build_training_output(&result.result, &events, &fcf, false);

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
            cost_scale_factor: 1_000_000.0,
            node_id: i32::try_from(stage_idx).unwrap_or(-1),
            graph_stage_id: -1,
            priced_state_date: STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
        })
        .collect();

    let warm_start_counts: Vec<u32> = fcf.pools.iter().map(|p| p.warm_start_count).collect();
    let policy_metadata = cobre_sddp::test_support::checkpoint_metadata(
        fx.n_stages as u32,
        cobre_io::GraphManifest::default(),
        cobre_io::ProducerBlock {
            completed_iterations: result.result.iterations as u32,
            final_lower_bound: result.result.final_lb,
            best_upper_bound: Some(result.result.final_ub),
            max_iterations: 3,
            forward_passes: 1,
            warm_start_cuts: warm_start_counts.iter().copied().max().unwrap_or(0),
            warm_start_counts,
            rng_seed: 42,
            total_visited_states: 0,
            training_block_mode: "parallel".to_string(),
            training_block_mode_per_stage: vec![],
            cost_scale_factor: None,
        },
    );

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

    let entity_counts = default_single_hydro_entity_counts();

    let (result_tx, result_rx) = mpsc::sync_channel(4);

    let io_thread = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

    let mut sim_workspaces = vec![SolverWorkspace::new(
        0,
        0,
        sim_solver,
        PatchBuffer::new(&fx.state, &[], &[]),
        &training_context,
        &stage_ctx,
        WorkspaceSizing::default(),
    )];

    let ec = zero_energy_conversion_set(fx.n_stages);

    simulate(
        &mut sim_workspaces,
        &stage_ctx,
        &fcf,
        &training_context,
        &sim_config,
        SimulationOutputSpec {
            result_tx: &result_tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &vec![vec![744.0]; fx.n_stages],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); fx.n_stages],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); fx.n_stages],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &vec![vec![1.0]; fx.n_stages],
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &sim_comm,
        &Traversal::default(),
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
            num_hosts: 1,
            threads_per_rank: 1,
            mpi_library: None,
            mpi_standard: None,
            thread_level: None,
            slurm_job_id: None,
            hosts: Vec::new(),
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

    let policy_manifest_path = policy_dir.join("manifest.bin");
    assert!(policy_manifest_path.is_file());
    {
        let checkpoint =
            read_policy_checkpoint(&policy_dir).expect("policy manifest.bin must read back");
        assert_eq!(checkpoint.metadata.producer.completed_iterations, 3);
    }

    let stage_bin_path = policy_dir.join("cuts/000.bin");
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
    use cobre_core::entities::hydro::HydroGenerationModel;
    use cobre_core::scenario::InflowModel;
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, HydroBlockBounds,
        HydroPenalties, HydroStageBounds, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, ThermalBlockBounds, ThermalStageBounds,
    };

    let bus = default_bus();

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
                max_turbined_m3s: 100.0,
                min_outflow_m3s: 50.0,
                max_generation_mw: 100.0,
                ..Default::default()
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
            hydro: HydroPenalties {
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

    let correlation = default_correlation_model();

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

    let study_dims = study_dims_for(false);
    // The operational-violation constraint *row* range is owned by `StageLayout` and
    // pinned by `stage_layout_operational_violation_rows_are_contiguous_blocks`; this
    // end-to-end test covers only the slack-*column* extraction path.
    let geometry = &templates_result.geometry_per_stage[0];
    let state = state_layout_for(1, 0);

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
    // Every stage clones `t0`, so stage-0 geometry must be replicated across all
    // stages for extraction to read the stage-correct slack columns.
    let equipment_geometry = vec![templates_result.geometry_per_stage[0].clone(); n_stages];
    let initial_state = vec![100.0_f64; state.n_state];
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };

    let mut fcf = make_fcf(n_stages);

    let state_boxes = permissive_state_boxes(state.n_state, n_stages);
    let stage_ctx_fixture = StageContextFixture::new(&templates, &state_boxes, &equipment_geometry);
    let stage_ctx = stage_ctx_fixture.ctx();

    let training_config = TrainingConfig {
        loop_config: LoopConfig {
            forward_passes: 1,
            training_enumerated: false,
            max_iterations: 1,
            start_iteration: 0,
            n_fwd_threads: 1,
            stopping_rules: iteration_limit(1),
        },
        cut_management: CutManagementConfig {
            cut_selection: None,
            budget: None,
            cut_activity_tolerance: 0.0,
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
        node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
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

    let entity_counts = default_single_hydro_entity_counts();

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
        PatchBuffer::new(&state, &[], &[]),
        &training_context,
        &stage_ctx,
        WorkspaceSizing::default(),
    )];

    let ec2 = zero_energy_conversion_set(n_stages);

    simulate(
        &mut sim_workspaces,
        &stage_ctx,
        &fcf,
        &training_context,
        &sim_config,
        SimulationOutputSpec {
            result_tx: &result_tx,
            hydro_cell_index: &cobre_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &block_hours_per_stage,
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![Vec::new(); n_stages],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); n_stages],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &hydro_productivities_per_stage,
            energy_conversion: &ec2,
            hydro_min_storage_hm3: &[0.0],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &StubComm,
        &Traversal::default(),
    )
    .expect("simulate must succeed");

    drop(result_tx);

    let results = io_thread.join().expect("I/O thread must not panic");
    assert_eq!(results.len(), 1, "expected exactly 1 scenario result");

    let scenario = &results[0];
    let found_nonzero_slack = scenario.stages.iter().any(|stage_result| {
        stage_result.hydros.iter().any(|hydro_result| {
            (hydro_result.outflow_slack_below_m3s - expected_slack_m3s).abs() < 1e-6
        })
    });
    assert!(
        found_nonzero_slack,
        "Expected at least one hydro result with outflow_slack_below_m3s = {expected_slack_m3s:.6} \
         (sentinel_m3s={sentinel_m3s} / zeta={zeta}), but all were zero. \
         This indicates the extraction path does not read from the slack column.",
    );
}

/// A `SimulationSelection::Enumerated` study with derived
/// `K == 1` (the [`Fixture`]'s single-opening chain) dispatches
/// `SimulationState::run` to `run_enumerated_simulation`, and its single
/// `SimulationScenarioResult` is bit-for-bit equal to the sampled
/// simulation's single scenario on the same trained study — both walks visit
/// the graph's one and only root-to-leaf path, drawing noise from the same
/// `(scenario = 0, iteration = 0, stage)` seed either way, so the two must
/// coincide exactly.
#[test]
fn enumerated_census_k1_matches_sampled_single_scenario() {
    let fx = Fixture::new(2);
    let mut fcf = make_fcf(fx.n_stages);
    let mut solver = MockSolver::with_fixed(100.0);
    let comm = StubComm;

    let training_config = TrainingConfig {
        loop_config: LoopConfig {
            forward_passes: 1,
            training_enumerated: false,
            max_iterations: 3,
            start_iteration: 0,
            n_fwd_threads: 1,
            stopping_rules: iteration_limit(3),
        },
        cut_management: CutManagementConfig {
            cut_selection: None,
            budget: None,
            cut_activity_tolerance: 0.0,
            risk_measures: fx.risk_measures.clone(),
        },
        events: EventConfig {
            event_sender: None,
            checkpoint_interval: None,
            shutdown_flag: None,
            export_states: false,
        },
    };

    // `default_single_hydro_entity_counts` (used below) declares both a hydro
    // and a bus, so the geometry must address both families, matching
    // `minimal_template`'s N=1 hydro, 1 bus, 1-block layout.
    let geometry: Vec<StageGeometry> = vec![hydro_only_bus_geometry(); fx.n_stages];
    let state_boxes = permissive_state_boxes(fx.state.n_state, fx.n_stages);
    let stage_ctx_fixture = StageContextFixture::new(&fx.templates, &state_boxes, &geometry);
    let stage_ctx = stage_ctx_fixture.ctx();
    let cut_state_layouts = all_enabled_cut_state_layouts(&fx.state, fx.n_stages);
    let study_dims = study_dims_for(false);
    let node_graph = cobre_sddp::test_support::chain_node_graph(&fx.stochastic);
    let training_context = TrainingContext {
        node_graph: &node_graph,
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
    train(
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

    let derived_k = cobre_sddp::test_support::node_scenario_count(&node_graph)
        .expect("node_scenario_count must not overflow on this trivial fixture");
    assert_eq!(
        derived_k, 1,
        "the fixture's single-opening chain must derive exactly one enumerated path"
    );

    let entity_counts = default_single_hydro_entity_counts();
    let ec = zero_energy_conversion_set(fx.n_stages);
    let sim_config = SimulationConfig {
        n_scenarios: 1,
        io_channel_capacity: 4,
        profile: Phase::Simulation.profile(),
    };
    let hydro_cell_index = cobre_sddp::test_support::identity_hydro_cell_index(256);
    let hydro_productivities_per_stage = vec![vec![1.0]; fx.n_stages];
    let block_hours_per_stage = vec![vec![1.0]; fx.n_stages];

    let run_sim =
        |traversal: &Traversal| -> cobre_sddp::simulation::types::SimulationScenarioResult {
            let sim_solver = MockSolver::with_fixed(100.0);
            let mut sim_workspaces = vec![SolverWorkspace::new(
                0,
                0,
                sim_solver,
                PatchBuffer::new(&fx.state, &[], &[]),
                &training_context,
                &stage_ctx,
                WorkspaceSizing::default(),
            )];
            let (result_tx, result_rx) = mpsc::sync_channel(4);
            simulate(
                &mut sim_workspaces,
                &stage_ctx,
                &fcf,
                &training_context,
                &sim_config,
                SimulationOutputSpec {
                    result_tx: &result_tx,
                    hydro_cell_index: &hydro_cell_index,
                    block_hours_per_stage: &block_hours_per_stage,
                    entity_counts: &entity_counts,
                    generic_constraint_row_entries: &vec![Vec::new(); fx.n_stages],
                    pumping_consumption_mw_per_m3s: &[],
                    contract_prices_per_stage: &vec![Vec::new(); fx.n_stages],
                    contract_slots: &[],
                    diversion_upstream: &HashMap::new(),
                    hydro_productivities_per_stage: &hydro_productivities_per_stage,
                    energy_conversion: &ec,
                    hydro_min_storage_hm3: &[0.0],
                    event_sender: None,
                    extended_delivery_anchors: &[],
                    transit_seed_arcs: &[],
                    past_defluences: &[],
                    study_stage_dates: &[],
                },
                None,
                &[],
                &comm,
                traversal,
            )
            .expect("simulate must succeed");
            drop(result_tx);
            let mut results: Vec<_> = result_rx.into_iter().collect();
            assert_eq!(results.len(), 1, "K=1 must produce exactly one scenario");
            results.remove(0)
        };

    let enumerated_plan = Traversal::resolve(&node_graph, true, 1);
    let enum_scenario = run_sim(&enumerated_plan);
    let sampled_scenario = run_sim(&Traversal::default());

    assert_eq!(
        enum_scenario, sampled_scenario,
        "the enumerated K=1 census's single scenario must be bit-for-bit equal to the \
         sampled simulation's single scenario on the same trained study"
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

// ── Census execution: value oracle, exact mean/variance, dedup, thread
//    determinism ────────────────────────────────────────────────────────────

/// Relative + absolute LP tolerance for census-mean-vs-extensive-form-value
/// comparison. Mirrors `extensive_form_oracle.rs`'s `REL_TOL`/`ABS_TOL`
/// unchanged (not re-derived): the backend primal/dual feasibility tolerance is
/// `≈ 1e-9`, scaled by the objective magnitude, bounding the gap between the
/// extensive-form LP and the trained policy's own accumulated solves.
const CENSUS_REL_TOL: f64 = 1e-6;
const CENSUS_ABS_TOL: f64 = 1e-4;

/// `true` when `a` and `b` agree within [`CENSUS_REL_TOL`]·|scale| + [`CENSUS_ABS_TOL`].
fn census_close(a: f64, b: f64) -> bool {
    (a - b).abs() <= CENSUS_ABS_TOL + CENSUS_REL_TOL * a.abs().max(b.abs())
}

/// Train `setup` single-rank/single-thread to convergence (mirrors
/// `extensive_form_oracle.rs`'s `train_bounds`, dropping the returned bounds —
/// callers here read the trained policy's simulated cost instead).
fn train_census_fixture_to_convergence(mut setup: StudySetup) -> StudySetup {
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("training must return Ok");
    assert!(
        outcome.error.is_none(),
        "training must not error: {:?}",
        outcome.error
    );
    setup
}

/// Convert a trained `setup` to a census (`enumerated`) simulation: derives
/// `K` from the graph itself ([`node_scenario_count`]) and installs it onto
/// `simulation_config.n_scenarios` — the field `Traversal::resolve`'s
/// `SimulationSelection::Enumerated` arm derives its own plan from
/// `node_graph` regardless, but `assign_scenarios`/`aggregate_simulation`
/// read `simulation_config.n_scenarios` directly, so it must match `K`.
fn as_enumerated_census(mut setup: StudySetup) -> StudySetup {
    let derived_k: u32 = node_scenario_count(&setup.inputs.node_graph)
        .expect("node_scenario_count must not overflow on these fixtures")
        .try_into()
        .expect("K must fit u32 on these fixtures");
    setup.simulation_enumerated = SimulationEnumeratedRequest::Enumerated;
    setup.simulation_config.n_scenarios = derived_k;
    setup
}

/// One census `simulate()` call's full outcome: the per-scenario detailed
/// results (canonical `scenario_id` order), the aggregated summary and
/// gathered `(scenario_id, cost, probability)` rows, and the total realized
/// LP-solve count across every workspace this run used (the dedup-scale
/// invariant checks) — measured via the solver-statistics delta around
/// `simulate()`, the same source `run_worker_scenarios` uses, since
/// `SimulationRunResult::solver_stats` is per-LEAF (a shared node's stats are
/// replicated into every leaf that visits it) and cannot answer "how many
/// solves actually happened."
struct CensusRun {
    results: Vec<SimulationScenarioResult>,
    summary: SimulationSummary,
    gathered: GatheredScenarioCosts,
    actual_lp_solves: u64,
}

/// Run `setup`'s (already-census-converted) simulation under `n_threads`
/// workers on `comm`, aggregating with the traversal-derived
/// [`cobre_sddp::SimulationWeighting`]. Generic over the communicator so the
/// same path runs under a single-rank stub and the 2-rank `Rank0Of2` shape.
fn run_census<C: Communicator>(setup: &StudySetup, comm: &C, n_threads: usize) -> CensusRun {
    let mut pool = setup
        .create_workspace_pool(comm, n_threads, ActiveSolver::new)
        .expect("workspace pool");
    let stats_before: Vec<SolverStatistics> = pool
        .workspaces
        .iter()
        .map(|ws| ws.solver.statistics())
        .collect();

    let n_scenarios = setup.simulation_config().n_scenarios.max(1) as usize;
    let (result_tx, result_rx) = mpsc::sync_channel(n_scenarios);
    let sim_run_result = setup
        .simulate(&mut pool.workspaces, comm, &result_tx, None, None, &[])
        .expect("census simulate must succeed");
    drop(result_tx);
    let mut results: Vec<SimulationScenarioResult> = result_rx.into_iter().collect();
    results.sort_by_key(|r| r.scenario_id);

    let actual_lp_solves: u64 = stats_before
        .iter()
        .zip(pool.workspaces.iter().map(|ws| ws.solver.statistics()))
        .map(|(before, after)| SolverStatsDelta::from_snapshots(before, &after).lp_solves)
        .sum();

    let traversal = Traversal::resolve(
        &setup.inputs.node_graph,
        true,
        setup.simulation_config().n_scenarios,
    );
    let weighting = traversal.simulation_weighting();
    let (summary, gathered) = aggregate_simulation(
        &sim_run_result.costs,
        setup.simulation_config(),
        comm,
        weighting,
    )
    .expect("aggregate_simulation must succeed");

    CensusRun {
        results,
        summary,
        gathered,
        actual_lp_solves,
    }
}

/// One census run's warm-start accounting: per-scenario cost as raw bits (for
/// bit-identity comparison), the summed warm-start basis offers, and the summed
/// basis-consistency rejections across every workspace.
struct WarmStartRun {
    per_scenario_cost_bits: Vec<(u32, u64)>,
    basis_offered: u64,
    basis_consistency_failures: u64,
}

/// Census `simulate()` with a caller-supplied warm-start cache, measuring the
/// warm-start offer/reject deltas across a fresh workspace pool.
fn run_census_with_bases<C: Communicator>(
    setup: &StudySetup,
    comm: &C,
    n_threads: usize,
    stage_bases: &[Option<CapturedBasis>],
) -> WarmStartRun {
    let mut pool = setup
        .create_workspace_pool(comm, n_threads, ActiveSolver::new)
        .expect("workspace pool");
    let before: Vec<SolverStatistics> = pool
        .workspaces
        .iter()
        .map(|ws| ws.solver.statistics())
        .collect();

    let n_scenarios = setup.simulation_config().n_scenarios.max(1) as usize;
    let (result_tx, result_rx) = mpsc::sync_channel(n_scenarios);
    setup
        .simulate(
            &mut pool.workspaces,
            comm,
            &result_tx,
            None,
            None,
            stage_bases,
        )
        .expect("census simulate must succeed");
    drop(result_tx);
    let mut results: Vec<SimulationScenarioResult> = result_rx.into_iter().collect();
    results.sort_by_key(|r| r.scenario_id);

    let (basis_offered, basis_consistency_failures) = before
        .iter()
        .zip(pool.workspaces.iter().map(|ws| ws.solver.statistics()))
        .map(|(b, a)| SolverStatsDelta::from_snapshots(b, &a))
        .fold((0_u64, 0_u64), |(off, fail), d| {
            (off + d.basis_offered, fail + d.basis_consistency_failures)
        });

    WarmStartRun {
        per_scenario_cost_bits: results
            .iter()
            .map(|r| (r.scenario_id, r.total_cost.to_bits()))
            .collect(),
        basis_offered,
        basis_consistency_failures,
    }
}

/// Pool-fill warms the previously-cold terminal leaves of an enumerated census:
/// training captures a basis only at the scenario-0 leaf, so every sibling leaf
/// starts `None`; `StudySetup::simulate` fills them from the shared pool. Stripping
/// the leaf pool of every capture (`stripped`) removes the fill's source, so its
/// leaves stay cold — the offered-basis delta between the two is the fill's work.
/// The filled bases must be ACCEPTED (0 consistency failures) and must not move
/// the objective (per-scenario cost bit-identical warm-vs-cold), and the run must
/// be bit-reproducible across thread shapes (never `hot == cold`).
#[test]
fn enumerated_census_pool_fill_warms_previously_cold_leaves() {
    let mut setup = branching_tree_setup_enumerated(30);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("training must return Ok");
    assert!(
        outcome.error.is_none(),
        "training must not error: {:?}",
        outcome.error
    );
    let sparse = outcome.result.basis_cache;

    let setup = as_enumerated_census(setup);

    let leaves: Vec<usize> = (0..setup.inputs.node_graph.successors.len())
        .filter(|&i| setup.inputs.node_graph.successors[NodePos(i)].is_empty())
        .collect();
    assert!(
        leaves.len() >= 2,
        "fixture must fan into >= 2 same-pool terminal leaves; got {}",
        leaves.len()
    );
    let captured_leaves = leaves.iter().filter(|&&i| sparse[i].is_some()).count();
    assert!(
        captured_leaves >= 1,
        "training must capture >= 1 terminal-leaf basis to seed the fill"
    );

    let mut stripped = sparse.clone();
    for &i in &leaves {
        stripped[i] = None;
    }

    let filled = run_census_with_bases(&setup, &comm, 1, &sparse);
    let no_leaf_source = run_census_with_bases(&setup, &comm, 1, &stripped);

    assert!(
        filled.basis_offered >= no_leaf_source.basis_offered + leaves.len() as u64,
        "pool-fill must warm every terminal leaf: offered {} vs stripped {} (>= +{} leaves)",
        filled.basis_offered,
        no_leaf_source.basis_offered,
        leaves.len()
    );
    assert_eq!(
        filled.basis_consistency_failures, 0,
        "every re-tagged filled basis must be accepted (same-pool template, own node_id)"
    );
    assert_eq!(
        filled.per_scenario_cost_bits, no_leaf_source.per_scenario_cost_bits,
        "warm != cold: the moved dispatch vertex must not move the per-scenario cost"
    );

    let filled_4 = run_census_with_bases(&setup, &comm, 4, &sparse);
    assert_eq!(
        filled.per_scenario_cost_bits, filled_4.per_scenario_cost_bits,
        "per-scenario cost must be bit-identical at threads=1 vs threads=4"
    );
}

/// Value oracle. `branching_tree_setup_enumerated` branches at BOTH
/// interior stages under non-uniform weights (the shape a shape-based
/// admission clause would have rejected — see `extensive_form_oracle.rs`);
/// trained to convergence, its census `mean_cost` must close to the
/// independently-computed extensive-form optimum, never `hot == cold`.
#[test]
fn census_mean_cost_closes_to_extensive_form_value() {
    let setup = branching_tree_setup_enumerated(30);
    let optimum = extensive_form_optimum(&setup);
    let setup = as_enumerated_census(train_census_fixture_to_convergence(setup));

    let comm = StubComm;
    let run = run_census(&setup, &comm, 1);

    assert!(
        census_close(run.summary.mean_cost, optimum),
        "census mean_cost {} must close to the extensive-form optimum {optimum} (gap {})",
        run.summary.mean_cost,
        run.summary.mean_cost - optimum
    );
}

/// Exact mean + variance on the DECOMP K-fan's known per-leaf weights.
/// `k_fan_policy_graph`'s leaf `i` (`1..=k`) carries the declared, non-uniform
/// edge probability `i / Σj` — hand-derived here from the fixture's own
/// documented construction, independent of the engine's plan weights — and
/// `summary.mean_cost`/`std_cost` must equal the weighted `Σ w·c` /
/// `√Σ w(c−μ)²` computed independently in this test from the per-path costs
/// `aggregate_simulation` gathers.
#[test]
fn census_mean_and_variance_match_hand_computed_weighted_formula() {
    let k = 4usize;
    let fixture = k_fan_setup_enumerated(k, 30);
    let setup = as_enumerated_census(train_census_fixture_to_convergence(fixture.setup));

    let comm = StubComm;
    let run = run_census(&setup, &comm, 1);

    assert_eq!(
        run.gathered.len(),
        k,
        "gathered rows must have exactly k entries"
    );
    let total_weight: f64 = (1..=k).map(|i| i as f64).sum();
    let mut costs = Vec::with_capacity(k);
    let mut weights = Vec::with_capacity(k);
    for (idx, &(scenario_id, cost, weight)) in run.gathered.iter().enumerate() {
        assert_eq!(
            scenario_id, idx as u32,
            "gathered rows must be canonical ascending scenario_id"
        );
        let w = weight.expect("census weight must be populated (Some) under Census weighting");
        let expected_w = (idx + 1) as f64 / total_weight;
        assert!(
            (w - expected_w).abs() < 1e-9,
            "leaf {idx}'s weight {w} must equal the fixture's declared edge probability \
             {expected_w} (= (idx+1)/Σj)"
        );
        costs.push(cost);
        weights.push(w);
    }

    let weight_sum: f64 = weights.iter().sum();
    assert!(
        (weight_sum - 1.0).abs() < 1e-9,
        "weights must sum to 1.0, got {weight_sum}"
    );

    let hand_mean: f64 = costs.iter().zip(&weights).map(|(c, w)| c * w).sum();
    let hand_var: f64 = costs
        .iter()
        .zip(&weights)
        .map(|(c, w)| w * (c - hand_mean).powi(2))
        .sum();
    let hand_std = hand_var.sqrt();

    assert!(
        (run.summary.mean_cost - hand_mean).abs() < 1e-9,
        "summary.mean_cost {} must equal the hand-computed Σw·c {hand_mean} within 1e-9",
        run.summary.mean_cost
    );
    assert!(
        (run.summary.std_cost - hand_std).abs() < 1e-9,
        "summary.std_cost {} must equal the hand-computed √Σw(c−μ)² {hand_std} within 1e-9",
        run.summary.std_cost
    );
}

/// Exact mean + variance census oracle on genuinely DISTINCT per-leaf costs.
/// `census_mean_and_variance_match_hand_computed_weighted_formula` above runs on
/// `k_fan_setup_enumerated`, whose deficit-dominated leaves are bit-identical
/// (`ρ = 0` placeholder) — so `Σ w·c = c` for any weight vector and the weighted
/// variance is `≈ 0`, neither assertion has power to catch a mis-weighted or
/// uniform-weighted census. `water_binding_external_fan_setup`'s scarce reservoir
/// and non-zero productivity (`ρ = 0.95`) make each leaf's own inflow bind,
/// producing distinct costs; its leaf `i` (`1..=k`) carries the same declared edge
/// probability `i / Σj` as the k-fan above, hand-derived here independent of the
/// engine's plan weights.
#[test]
fn census_distinct_cost_mean_and_variance_match_hand_computed_weighted_formula() {
    let k = 3usize;
    let fixture = water_binding_external_fan_setup(k, 30);
    let setup = as_enumerated_census(train_census_fixture_to_convergence(fixture));

    let comm = StubComm;
    let run = run_census(&setup, &comm, 1);

    assert_eq!(
        run.gathered.len(),
        k,
        "gathered rows must have exactly k entries"
    );
    let total_weight: f64 = (1..=k).map(|i| i as f64).sum();
    let mut costs = Vec::with_capacity(k);
    let mut weights = Vec::with_capacity(k);
    for (idx, &(scenario_id, cost, weight)) in run.gathered.iter().enumerate() {
        assert_eq!(
            scenario_id, idx as u32,
            "gathered rows must be canonical ascending scenario_id"
        );
        let w = weight.expect("census weight must be populated (Some) under Census weighting");
        let expected_w = (idx + 1) as f64 / total_weight;
        assert!(
            (w - expected_w).abs() < 1e-9,
            "leaf {idx}'s weight {w} must equal the fixture's declared edge probability \
             {expected_w} (= (idx+1)/Σj)"
        );
        costs.push(cost);
        weights.push(w);
    }

    let weight_sum: f64 = weights.iter().sum();
    assert!(
        (weight_sum - 1.0).abs() < 1e-9,
        "weights must sum to 1.0, got {weight_sum}"
    );

    // Non-degeneracy self-check: without this, a fixture that silently degenerated
    // to equal per-leaf costs would leave the mean/variance assertions below vacuous.
    let max_cost = costs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_cost = costs.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        max_cost - min_cost > 1.0,
        "per-leaf costs must be genuinely distinct (max {max_cost} - min {min_cost} must exceed \
         1.0); a fixture that degenerated to equal costs would make the mean/variance \
         assertions below vacuous: {costs:?}"
    );

    let hand_mean: f64 = costs.iter().zip(&weights).map(|(c, w)| c * w).sum();
    let hand_var: f64 = costs
        .iter()
        .zip(&weights)
        .map(|(c, w)| w * (c - hand_mean).powi(2))
        .sum();
    let hand_std = hand_var.sqrt();

    assert!(
        (run.summary.mean_cost - hand_mean).abs() < 1e-9,
        "summary.mean_cost {} must equal the hand-computed Σw·c {hand_mean} within 1e-9",
        run.summary.mean_cost
    );
    assert!(
        (run.summary.std_cost - hand_std).abs() < 1e-9,
        "summary.std_cost {} must equal the hand-computed √Σw(c−μ)² {hand_std} within 1e-9",
        run.summary.std_cost
    );
    assert!(
        run.summary.std_cost > 0.0,
        "summary.std_cost must be strictly positive on this distinct-cost fixture, got {}",
        run.summary.std_cost
    );
}

/// Dedup correctness + solve count, on a trunk-then-fan graph whose
/// `t_trunk` trunk nodes are shared by every one of the `k` leaves.
///
/// (a) Extract-once identity: every leaf's stage-`t` per-entity row for a
/// shared trunk stage must equal every OTHER leaf's — both are copies of the
/// SAME node's single extracted result, so they must be bit-identical.
/// (b) Dedup-scale: the single-rank solve count must equal the enumerated
/// forward's own `Σ forward_solve_counts` dedup-scale assertion — that
/// quantity is `pub(crate)`-only, but on any admitted enumerated fixture
/// (single-predecessor, one opening per node) `node_prefix_counts` gives
/// exactly `1` per node (`assert_reachable_prefixes_match`'s precondition,
/// `extensive_form_oracle.rs`) — the same equality
/// `forward_solve_counts_k_fan_matches_hand_computed_prefix_counts`
/// (`setup/node_graph.rs`) pins from inside the crate — so their sum equals
/// the node count, proving no per-path re-solve.
#[test]
fn census_shared_trunk_rows_extract_once_and_solve_count_matches_dedup() {
    let t_trunk = 3usize;
    let k = 3usize;
    let fixture = trunk_fan_setup_enumerated(t_trunk, k, 5);
    let setup = as_enumerated_census(train_census_fixture_to_convergence(fixture.setup));

    let comm = StubComm;
    let run = run_census(&setup, &comm, 1);
    assert_eq!(run.results.len(), k, "census must produce exactly k leaves");

    for t in 0..t_trunk {
        let first = &run.results[0].stages[t];
        for other in &run.results[1..] {
            assert_eq!(
                &other.stages[t], first,
                "leaf {}'s stage {t} row must equal leaf {}'s — both extract the same \
                 shared trunk node's single result",
                other.scenario_id, run.results[0].scenario_id
            );
        }
    }

    let prefix_counts =
        node_prefix_counts(&setup.inputs.node_graph).expect("node_prefix_counts must not overflow");
    assert!(
        prefix_counts.iter().all(|&c| c == 1),
        "every node on this admitted (single-predecessor, |Ω|=1) fixture must have exactly \
         one root-to-node prefix: {prefix_counts:?}"
    );
    let expected_solves: u64 = prefix_counts.iter().sum();
    assert_eq!(
        run.actual_lp_solves, expected_solves,
        "single-rank census solve count must equal Σ forward_solve_counts (no per-path re-solve)"
    );
}

/// A [`cobre_core::System`] whose only purpose is driving
/// [`SimulationParquetWriter::new`]'s directory/block-hours setup for the
/// byte-comparison — the writer reads only `system.stages()` (block hours)
/// and entity counts, never the policy graph, so this need not reproduce a
/// census fixture's branching structure. Mirrors the single-hydro/single-bus,
/// one-744h-block-per-stage shape every K-fan/branching-tree/trunk-fan fixture
/// in `test_support.rs` documents, parameterized only by stage count.
fn writer_shape_system(n_stages: usize) -> cobre_core::System {
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
                },
            )
        })
        .collect();
    SystemBuilder::new()
        .stages(stages)
        .build()
        .expect("writer_shape_system: SystemBuilder::build must succeed")
}

/// Write `results`/`gathered`'s `scenario_summary.parquet` and per-entity
/// (hydros) Parquet under a fresh `TempDir`, returning the raw bytes of both
/// — `scenario_summary.parquet` verbatim, the hydros files concatenated in
/// sorted-path order (one file per `scenario_id=NNNN/` Hive partition).
fn write_census_parquet_outputs(
    n_stages: usize,
    results: Vec<SimulationScenarioResult>,
    gathered: &GatheredScenarioCosts,
) -> (Vec<u8>, Vec<u8>) {
    let tmp = tempfile::TempDir::new().expect("tempdir must succeed");
    let system = writer_shape_system(n_stages);
    let mut writer = SimulationParquetWriter::new(tmp.path(), &system)
        .expect("SimulationParquetWriter::new must succeed");
    for scenario in results {
        writer
            .write_scenario(ScenarioWritePayload::from(scenario))
            .expect("write_scenario must succeed");
    }
    let _ = writer.finalize(0);

    let rows: Vec<(u32, Option<f64>, f64)> = gathered
        .iter()
        .map(|&(id, cost, weight)| (id, weight, cost))
        .collect();
    write_scenario_summary(tmp.path(), &rows).expect("write_scenario_summary must succeed");

    let summary_bytes = std::fs::read(tmp.path().join("simulation/scenario_summary.parquet"))
        .expect("read scenario_summary.parquet");

    let hydros_dir = tmp.path().join("simulation/hydros");
    let mut hydro_files: Vec<std::path::PathBuf> = if hydros_dir.is_dir() {
        std::fs::read_dir(&hydros_dir)
            .expect("read_dir simulation/hydros")
            .filter_map(std::result::Result::ok)
            .flat_map(|partition| {
                std::fs::read_dir(partition.path())
                    .expect("read_dir a hydros scenario partition")
                    .filter_map(std::result::Result::ok)
                    .map(|e| e.path())
                    .collect::<Vec<_>>()
            })
            .collect()
    } else {
        Vec::new()
    };
    hydro_files.sort();
    assert!(
        !hydro_files.is_empty(),
        "the writer must have produced at least one simulation/hydros/ parquet file"
    );
    let mut hydro_bytes = Vec::new();
    for f in hydro_files {
        hydro_bytes.extend(std::fs::read(&f).expect("read a hydros parquet file"));
    }

    (summary_bytes, hydro_bytes)
}

/// Thread bit-invariance. For a fixed `K >= 2` census, `mean_cost`/
/// `std_cost` (`to_bits()`), the `scenario_summary.parquet` bytes (including
/// the `probability` column), and a per-entity (hydros) Parquet file must be
/// bit-identical across `--threads 1`, `2`, and `4` — the replicate model's
/// canonical claim-order-independent scatter plus the fixed-order gather-then-
/// sum reduction make the whole output topology-invariant, never merely the
/// aggregate scalars.
#[test]
fn census_output_is_bit_identical_across_thread_counts() {
    let k = 4usize;
    let fixture = k_fan_setup_enumerated(k, 30);
    let setup = as_enumerated_census(train_census_fixture_to_convergence(fixture.setup));
    let n_stages = setup.num_stages();
    let comm = StubComm;

    let mut summaries = Vec::with_capacity(3);
    let mut parquet_outputs = Vec::with_capacity(3);
    for &n_threads in &[1usize, 2, 4] {
        let run = run_census(&setup, &comm, n_threads);
        summaries.push((
            run.summary.mean_cost.to_bits(),
            run.summary.std_cost.to_bits(),
        ));
        parquet_outputs.push(write_census_parquet_outputs(
            n_stages,
            run.results,
            &run.gathered,
        ));
    }

    for n_threads in [2usize, 4] {
        let idx = if n_threads == 2 { 1 } else { 2 };
        assert_eq!(
            summaries[idx], summaries[0],
            "mean_cost/std_cost bit patterns must be identical between threads=1 and \
             threads={n_threads}"
        );
        assert_eq!(
            parquet_outputs[idx].0, parquet_outputs[0].0,
            "scenario_summary.parquet bytes must be identical between threads=1 and \
             threads={n_threads}"
        );
        assert_eq!(
            parquet_outputs[idx].1, parquet_outputs[0].1,
            "the hydros Parquet bytes must be identical between threads=1 and \
             threads={n_threads}"
        );
    }
}

/// Rank-shape invariance of the census SWEEP. Mirrors the forward engine's
/// `enumerated_single_path_2rank_stub_matches_single_rank`: a single-path
/// (`K == 1`) census driven end-to-end through `StudySetup::simulate` must
/// produce bit-identical `mean_cost`/`std_cost` under a single-rank stub and the
/// 2-rank `Rank0Of2` shape. Unlike `census_output_is_bit_identical_across_thread_counts`
/// (threads only, `world_size == 1`) and the `mpi_wire.rs` rank-shape gate
/// (`aggregate_simulation` on hand-built costs, never the sweep), this exercises
/// the full `world_size == 2` sweep pipeline — `assign_scenarios` /
/// `mark_own_paths` / `run_sweep` / `re_expand`, then the gather-then-sum
/// aggregate. `Rank0Of2` is faithful ONLY at `K == 1`, where rank 1's
/// `assign_scenarios` share is empty (the zero it leaves unwritten IS what a
/// genuine rank 1 would send); genuine cross-rank trunk replication is covered
/// by the real-MPI SLURM job, the same ceiling the forward engine's stub carries.
#[test]
fn census_single_path_sweep_bit_identical_across_rank_shapes() {
    let setup = as_enumerated_census(train_census_fixture_to_convergence(
        single_path_enumerated_setup(30),
    ));

    // Power self-checks: `Rank0Of2` is faithful only when rank 1 does zero work,
    // i.e. the census resolves to a single path, and the stub genuinely presents
    // a 2-rank world (never a vacuous size-1 comparison).
    assert_eq!(
        setup.simulation_config().n_scenarios,
        1,
        "single_path fixture must resolve to K == 1 for Rank0Of2 faithfulness"
    );
    assert_eq!(Rank0Of2.size(), 2, "Rank0Of2 must present a 2-rank world");

    let single = run_census(&setup, &StubComm, 1);
    let two_rank = run_census(&setup, &Rank0Of2, 1);

    assert_eq!(
        single.summary.mean_cost.to_bits(),
        two_rank.summary.mean_cost.to_bits(),
        "census mean_cost must be bitwise identical between world_size 1 and 2"
    );
    assert_eq!(
        single.summary.std_cost.to_bits(),
        two_rank.summary.std_cost.to_bits(),
        "census std_cost must be bitwise identical between world_size 1 and 2"
    );
    assert_eq!(
        single.summary.std_cost, 0.0,
        "a single-path census is a degenerate zero-variance distribution"
    );
}
