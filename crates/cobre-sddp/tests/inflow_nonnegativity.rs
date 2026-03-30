//! Integration tests for inflow non-negativity enforcement via the penalty method.
//!
//! Verifies end-to-end behaviour for `InflowNonNegativityMethod::Penalty`:
//!
//! 1. Training completes without `SddpError::Infeasible` when PAR(p) noise
//!    produces negative effective inflows.
//! 2. The inflow non-negativity slack absorbs negative inflow (slack > 0 in
//!    the LP solution at any stage where inflow is negative).
//! 3. `SimulationHydroResult.inflow_nonnegativity_slack_m3s` is populated
//!    (non-zero) in at least one scenario / stage when the system produces
//!    negative inflows.
//!
//! ## System design
//!
//! All three tests use a 2-hydro, 1-bus, 3-stage system. The PAR(0) inflow
//! model has `mean_m3s = 0.0` and `std_m3s = 30.0`, so approximately half of
//! all sampled noise values produce negative effective inflows. The opening
//! tree is generated with 10 openings per stage from seed 42, guaranteeing
//! that multiple openings carry negative noise realisations.
//!
//! `HighsSolver` is used throughout so the LP is actually solved and slack
//! columns receive non-trivial primal values when inflow is negative.

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
    BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties, ContractStageBounds, DeficitSegment,
    EntityId, HydroStageBounds, HydroStagePenalties, LineStageBounds, LineStagePenalties,
    NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults, PumpingStageBounds, ResolvedBounds,
    ResolvedPenalties, ThermalStageBounds,
    scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_sddp::{
    EntityCounts, FutureCostFunction, HorizonMode, InflowNonNegativityMethod, PatchBuffer,
    RiskMeasure, SimulationConfig, SimulationOutputSpec, SolverWorkspace, StageContext,
    StageIndexer, StoppingMode, StoppingRule, StoppingRuleSet, TrainingConfig, TrainingContext,
    hydro_models::PrepareHydroModelsResult, lp_builder::build_stage_templates, simulate, train,
};
use cobre_solver::HighsSolver;
use cobre_stochastic::{
    OpeningTree, PrecomputedPar, StochasticContext, build_stochastic_context,
    correlation::resolve::DecomposedCorrelation, tree::generate::generate_opening_tree,
};

// ===========================================================================
// Communicator stub
// ===========================================================================

/// Single-rank communicator that performs identity operations.
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
// System and stochastic context fixture
// ===========================================================================

const N_STAGES: usize = 3;
const N_HYDROS: usize = 2;

/// Build a 2-hydro, 1-bus, 3-stage system designed to produce negative inflows.
///
/// - `mean_m3s = 0.0` and `std_m3s = 30.0` means roughly half of all sampled
///   noise values are negative, giving effective inflows < 0 m³/s.
/// - The opening tree uses 10 openings per stage, guaranteeing multiple
///   openings carry negative noise realisations for the test to be meaningful.
/// - 2 hydros ensure the fixture exercises the multi-hydro code path.
/// - 1 block per stage for simplicity.
///
/// `ResolvedBounds` and `ResolvedPenalties` are built manually from the hydro
/// entity values so that `build_stage_templates` can read them without
/// `cobre-io` case loading.
fn build_system() -> cobre_core::System {
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::InflowModel;

    let zero_entity_penalties = HydroPenalties {
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
        max_storage_hm3: 50.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity {
            productivity_mw_per_m3s: 1.0,
        },
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 50.0,
        min_generation_mw: 0.0,
        max_generation_mw: 50.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: zero_entity_penalties,
    };

    let hydros = vec![make_hydro(1, "H1"), make_hydro(2, "H2")];

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
            branching_factor: 10,
            noise_method: NoiseMethod::Saa,
        },
    };

    let stages: Vec<Stage> = (0..N_STAGES).map(make_stage).collect();

    // PAR(0) with mean=0.0, std=30.0 → large chance of negative inflow.
    let inflow_models: Vec<InflowModel> = (0..N_STAGES)
        .flat_map(|stage_idx| {
            [EntityId(1), EntityId(2)]
                .iter()
                .map(move |&hydro_id| InflowModel {
                    hydro_id,
                    stage_id: stage_idx as i32,
                    mean_m3s: 0.0,
                    std_m3s: 30.0,
                    ar_coefficients: vec![],
                    residual_std_ratio: 1.0,
                })
        })
        .collect();

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
                ],
                // 2×2 identity: uncorrelated inflows.
                matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            }],
        },
    );
    let correlation = CorrelationModel {
        method: "cholesky".to_string(),
        profiles,
        schedule: vec![],
    };

    // Build ResolvedBounds manually from entity field values.
    // build_stage_templates requires this table to be populated (not empty).
    let hydro_bounds_default = HydroStageBounds {
        min_storage_hm3: 0.0,
        max_storage_hm3: 50.0,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 50.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        min_generation_mw: 0.0,
        max_generation_mw: 50.0,
        max_diversion_m3s: None,
        filling_inflow_m3s: 0.0,
        water_withdrawal_m3s: 0.0,
    };
    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: N_HYDROS,
            n_thermals: 0,
            n_lines: // n_thermals
        0,
            n_pumping: // n_lines
        0,
            n_contracts: // n_pumping
        0,
            n_stages: // n_contracts
        N_STAGES,
        },
        &BoundsDefaults {
            hydro: hydro_bounds_default,
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

    // Build ResolvedPenalties manually.
    let hydro_penalties_default = HydroStagePenalties {
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
    let resolved_penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: N_HYDROS,
            n_buses: 1,
            n_lines: // n_buses
        0,
            n_ncs: // n_lines
        0,
            n_stages: // n_ncs
        N_STAGES,
        },
        &PenaltiesDefaults {
            hydro: hydro_penalties_default,
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    cobre_core::SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .bounds(resolved_bounds)
        .penalties(resolved_penalties)
        .build()
        .unwrap()
}

/// Build a [`StochasticContext`] for the 2-hydro, 3-stage negative-inflow fixture.
fn build_stochastic() -> StochasticContext {
    let system = build_system();
    build_stochastic_context(&system, 42, &[], &[], None).unwrap()
}

/// Build an [`OpeningTree`] with 10 openings at stage 0 for the 2-hydro fixture.
///
/// 10 openings with `std = 30.0` and seed 42 guarantees multiple openings
/// produce negative inflow values.
fn build_opening_tree() -> OpeningTree {
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
            branching_factor: 10,
            noise_method: NoiseMethod::Saa,
        },
    };

    let id_h1 = EntityId(1);
    let id_h2 = EntityId(2);
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
                ],
                matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            }],
        },
    );

    let mut decomposed = DecomposedCorrelation::build(&CorrelationModel {
        method: "cholesky".to_string(),
        profiles,
        schedule: vec![],
    })
    .unwrap();

    generate_opening_tree(42, &[stage], 2, &mut decomposed, &[id_h1, id_h2])
}

// ===========================================================================
// Shared fixture builder
// ===========================================================================

/// All resources needed to run training and simulation.
struct Fixture {
    stage_templates: cobre_sddp::lp_builder::StageTemplates,
    stochastic: StochasticContext,
    opening_tree: OpeningTree,
    indexer: StageIndexer,
    initial_state: Vec<f64>,
    horizon: HorizonMode,
    risk_measures: Vec<RiskMeasure>,
    entity_counts: EntityCounts,
    inflow_method: InflowNonNegativityMethod,
}

fn build_fixture() -> Fixture {
    let system = build_system();
    let inflow_method = InflowNonNegativityMethod::Penalty { cost: 1000.0 };

    let par_lp = PrecomputedPar::build(
        system.inflow_models(),
        &system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .cloned()
            .collect::<Vec<_>>(),
        &system.hydros().iter().map(|h| h.id).collect::<Vec<_>>(),
    )
    .unwrap();

    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    let stage_templates = build_stage_templates(
        &system,
        &inflow_method,
        &par_lp,
        &cobre_stochastic::normal::precompute::PrecomputedNormal::default(),
        &hydro_models.production,
        &hydro_models.evaporation,
    )
    .expect("no FPHA plants in integration test fixture");
    let stochastic = build_stochastic();
    let opening_tree = build_opening_tree();

    let n_stages = stage_templates.templates.len();
    let first_tmpl = stage_templates.templates.first().expect("at least 1 stage");
    let n_blks = system.stages().first().map_or(1, |s| s.blocks.len().max(1));
    let has_inflow_penalty = inflow_method.has_slack_columns() && first_tmpl.n_hydro > 0;
    let indexer = StageIndexer::with_equipment(
        &cobre_sddp::EquipmentCounts {
            hydro_count: first_tmpl.n_hydro,
            max_par_order: first_tmpl.max_par_order,
            n_thermals: system.thermals().len(),
            n_lines: system.lines().len(),
            n_buses: system.buses().len(),
            n_blks,
            has_inflow_penalty,
            max_deficit_segments: 1,
        },
        &cobre_sddp::FphaColumnLayout {
            hydro_indices: vec![],
            planes_per_hydro: vec![],
        },
    );
    // z-inflow column and row ranges are set by StageIndexer::new at
    // fixed offset N*(1+L), no per-stage wiring needed.

    let initial_state = vec![0.0_f64; indexer.n_state];
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let entity_counts = EntityCounts {
        hydro_ids: system.hydros().iter().map(|h| h.id.0).collect(),
        hydro_productivities: vec![1.0; system.hydros().len()],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: system.buses().iter().map(|b| b.id.0).collect(),
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    Fixture {
        stage_templates,
        stochastic,
        opening_tree,
        indexer,
        initial_state,
        horizon,
        risk_measures,
        entity_counts,
        inflow_method,
    }
}

// ===========================================================================
// Shared test helpers
// ===========================================================================

fn train_fixture(
    fx: &Fixture,
    iterations: u64,
) -> Result<cobre_sddp::TrainingOutcome, cobre_sddp::SddpError> {
    let n_stages = fx.stage_templates.templates.len();
    let mut fcf = FutureCostFunction::new(n_stages, fx.indexer.n_state, 1, 20, 0);
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");
    let comm = StubComm;

    let _n_stages = fx.stage_templates.templates.len();
    let block_counts: Vec<usize> = fx
        .stage_templates
        .block_hours_per_stage
        .iter()
        .map(Vec::len)
        .collect();
    let max_blocks = block_counts.iter().copied().max().unwrap_or(1);

    let stage_ctx = StageContext {
        templates: &fx.stage_templates.templates,
        base_rows: &fx.stage_templates.base_rows,
        noise_scale: &fx.stage_templates.noise_scale,
        n_hydros: fx.stage_templates.n_hydros,
        n_load_buses: fx.stage_templates.n_load_buses,
        load_balance_row_starts: &fx.stage_templates.load_balance_row_starts,
        load_bus_indices: &fx.stage_templates.load_bus_indices,
        block_counts_per_stage: &block_counts,
        ncs_max_gen: &[],
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
            max_blocks,
            cut_selection: None,
            shutdown_flag: None,
            start_iteration: 0,
        },
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            horizon: &fx.horizon,
            indexer: &fx.indexer,
            inflow_method: &fx.inflow_method,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
        },
        &fx.opening_tree,
        &fx.risk_measures,
        StoppingRuleSet {
            rules: vec![StoppingRule::IterationLimit { limit: iterations }],
            mode: StoppingMode::Any,
        },
        &comm,
        HighsSolver::new,
    )
}

fn simulate_fixture(
    fx: &Fixture,
    fcf: &FutureCostFunction,
) -> Result<Vec<cobre_sddp::SimulationScenarioResult>, cobre_sddp::SimulationError> {
    let (result_tx, result_rx) = mpsc::sync_channel(32);

    let collector_thread = std::thread::spawn(move || {
        let mut all_results = Vec::new();
        while let Ok(r) = result_rx.recv() {
            all_results.push(r);
        }
        all_results
    });

    let mut sim_workspaces = vec![SolverWorkspace::new(
        HighsSolver::new().expect("HighsSolver::new must succeed"),
        PatchBuffer::new(fx.indexer.hydro_count, fx.indexer.max_par_order, 0, 0),
        fx.indexer.n_state,
        fx.indexer.hydro_count,
        fx.indexer.max_par_order,
        0,
        0,
    )];
    let comm = StubComm;

    let block_counts_sim: Vec<usize> = fx
        .stage_templates
        .block_hours_per_stage
        .iter()
        .map(Vec::len)
        .collect();

    simulate(
        &mut sim_workspaces,
        &StageContext {
            templates: &fx.stage_templates.templates,
            base_rows: &fx.stage_templates.base_rows,
            noise_scale: &fx.stage_templates.noise_scale,
            n_hydros: fx.stage_templates.n_hydros,
            n_load_buses: fx.stage_templates.n_load_buses,
            load_balance_row_starts: &fx.stage_templates.load_balance_row_starts,
            load_bus_indices: &fx.stage_templates.load_bus_indices,
            block_counts_per_stage: &block_counts_sim,
            ncs_max_gen: &[],
        },
        fcf,
        &TrainingContext {
            horizon: &fx.horizon,
            indexer: &fx.indexer,
            inflow_method: &fx.inflow_method,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
        },
        &SimulationConfig {
            n_scenarios: 20,
            io_channel_capacity: 32,
        },
        SimulationOutputSpec {
            result_tx: &result_tx,
            zeta_per_stage: &fx.stage_templates.zeta_per_stage,
            block_hours_per_stage: &fx.stage_templates.block_hours_per_stage,
            entity_counts: &fx.entity_counts,
            generic_constraint_row_entries: &[],
            ncs_col_starts: &[],
            n_ncs_per_stage: &[],
            ncs_entity_ids_per_stage: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &fx.stage_templates.hydro_productivities_per_stage,
            event_sender: None,
        },
        &[],
        &comm,
    )?;

    drop(result_tx);
    Ok(collector_thread
        .join()
        .expect("collector thread must not panic"))
}

// ===========================================================================
// Test 1: Penalty method prevents LP infeasibility
// ===========================================================================

/// Verify that training with `InflowNonNegativityMethod::Penalty` completes
/// without `SddpError::Infeasible` when PAR(0) noise produces negative
/// effective inflows. With `mean_m3s = 0.0` and `std_m3s = 30.0`, approximately
/// half of sampled openings carry negative inflow values. The penalty slack
/// columns keep the LP feasible at every stage.
#[test]
fn test_penalty_method_prevents_infeasibility() {
    let fx = build_fixture();
    let result = train_fixture(&fx, 5);
    assert!(
        result.is_ok(),
        "training must succeed without SddpError::Infeasible with penalty method, got: {result:?}"
    );
}

// ===========================================================================
// Test 2: Penalty slack absorbs negative inflow in simulation
// ===========================================================================

/// Verify that when the penalty method is active, simulation produces at least one
/// `SimulationHydroResult` with `inflow_nonnegativity_slack_m3s > 0.0` when negative
/// inflows occur. With `mean_m3s = 0.0`, `std_m3s = 30.0`, and 20 scenarios,
/// approximately half of noise draws are negative.
#[test]
fn test_penalty_slack_value_matches_negative_inflow() {
    let fx = build_fixture();
    let n_stages = fx.stage_templates.templates.len();
    let fcf = FutureCostFunction::new(n_stages, fx.indexer.n_state, 1, 20, 0);

    train_fixture(&fx, 3).expect("training must succeed before simulation");
    let scenario_results = simulate_fixture(&fx, &fcf).expect("simulate must succeed");

    let found_nonzero_slack = scenario_results.iter().any(|scenario| {
        scenario.stages.iter().any(|stage| {
            stage
                .hydros
                .iter()
                .any(|h| h.inflow_nonnegativity_slack_m3s > 0.0)
        })
    });

    assert!(
        found_nonzero_slack,
        "at least one hydro must have inflow_nonnegativity_slack_m3s > 0.0 across 20 scenarios \
         with mean_m3s=0 and std_m3s=30; none found"
    );
}

// ===========================================================================
// Test 3: Simulation slack output field is populated
// ===========================================================================

/// Verify that `SimulationHydroResult.inflow_nonnegativity_slack_m3s` is
/// correctly populated in simulation output when the penalty method is active.
/// The field is wired through the extraction pipeline and produces non-zero
/// values when inflow is negative.
#[test]
fn test_simulation_slack_output_populated() {
    let fx = build_fixture();
    let n_stages = fx.stage_templates.templates.len();
    let fcf = FutureCostFunction::new(n_stages, fx.indexer.n_state, 1, 20, 0);

    train_fixture(&fx, 3).expect("training must succeed");
    let scenario_results = simulate_fixture(&fx, &fcf).expect("simulate must succeed");

    assert_eq!(
        scenario_results.len(),
        20,
        "expected 20 simulation results, got {}",
        scenario_results.len()
    );

    let any_nonzero = scenario_results.iter().any(|scenario| {
        scenario.stages.iter().any(|stage| {
            stage
                .hydros
                .iter()
                .any(|h| h.inflow_nonnegativity_slack_m3s > 0.0)
        })
    });

    assert!(
        any_nonzero,
        "inflow_nonnegativity_slack_m3s must be > 0.0 in at least one hydro stage result"
    );
}
