//! Chronological inflow noise must reach the water balance of the hydro it was
//! drawn for, and only that hydro.
//!
//! Two independent hydros on a two-block chronological stage, neither able to
//! release water profitably (zero load, positive turbining and spillage costs, a
//! wide storage box), so each hydro's end-of-stage storage is its initial storage
//! plus the stage inflow volume. A one-hot standardized draw on hydro 1 must then
//! raise hydro 1's end storage by `ζ · σ₁` hm³ and leave hydro 0's untouched. The
//! assertions read the solved LP, not row positions, so they hold for any
//! encoding of the inflow in the water balance.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]

mod common;

use std::path::Path;

use chrono::NaiveDate;
use cobre_core::scenario::{InflowModel, LoadModel};
use cobre_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
};
use cobre_core::{
    BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, DeficitSegment,
    EntityId, HydroBlockBounds, HydroGenerationModel, HydroPenalties, HydroStageBounds,
    HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
    PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds, ResolvedPenalties,
    SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
};
use cobre_io::config::{
    Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
    InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig, RowSelectionConfig,
    SimulationConfig as IoSimulationConfig, StoppingRuleConfig, TrainingConfig, TrainingSelection,
    TrainingSolverConfig, UpperBoundEvaluationConfig,
};
use cobre_sddp::SddpError;
use cobre_sddp::StudySetup;
use cobre_sddp::build_stage_templates_resolving_layout;
use cobre_sddp::hydro_models::PrepareHydroModelsResult;
use cobre_sddp::indexer::StateDim;
use cobre_sddp::inflow_method::InflowNonNegativityMethod;
use cobre_sddp::resolved_parameters::ResolvedParameters;
use cobre_sddp::setup::{NodePos, StageIdx};
use cobre_sddp::test_support::capture_patched_node_template_with_inflow_noise;
use cobre_sddp::test_support::chronological_multi_block_inflow_noise;
use cobre_solver::{ActiveSolver, SolverInterface};

use common::build_setup_in_code;
use common::builders::{
    BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
};
use common::try_build_setup_in_code;

const N_STAGES: usize = 2;
const BUS_ID: i32 = 10;
const HYDRO_IDS: [i32; 2] = [1, 2];
const INFLOW_MEAN_M3S: f64 = 60.0;
const INFLOW_STD_M3S: [f64; 2] = [10.0, 20.0];
const BLOCK_HOURS: [f64; 2] = [300.0, 444.0];
const INITIAL_STORAGE_HM3: f64 = 1000.0;
const M3S_TO_HM3: f64 = 0.0036;

fn hydro_penalties() -> HydroPenalties {
    HydroPenalties {
        spillage_cost: 0.01,
        diversion_cost: 0.0,
        turbined_cost: 0.01,
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
        inflow_nonnegativity_cost: 0.0,
    }
}

fn build_system() -> cobre_core::System {
    build_system_with(
        [BlockMode::Chronological; N_STAGES],
        &BLOCK_HOURS,
        [INFLOW_STD_M3S; N_STAGES],
        [None, None],
    )
}

/// `inflow_std_m3s[stage][hydro]` is hydro `hydro`'s inflow standard deviation at
/// stage `stage`; `entry_stage_ids[hydro]` is that hydro's `HydroSpec::entry_stage_id`.
fn build_system_with(
    block_modes: [BlockMode; N_STAGES],
    block_hours: &[f64],
    inflow_std_m3s: [[f64; 2]; N_STAGES],
    entry_stage_ids: [Option<i32>; 2],
) -> cobre_core::System {
    let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();

    let bus = make_bus(
        EntityId(BUS_ID),
        BusSpec {
            name: "B".to_string(),
            operational_start_date: start,
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        },
    );

    let hydros = HYDRO_IDS
        .iter()
        .enumerate()
        .map(|(k, &id)| {
            make_hydro(
                EntityId(id),
                HydroSpec {
                    name: format!("H{id}"),
                    operational_start_date: start,
                    bus_id: EntityId(BUS_ID),
                    entry_stage_id: entry_stage_ids[k],
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 10_000.0,
                    max_turbined_m3s: 100.0,
                    generation_model: HydroGenerationModel::ConstantProductivity,
                    specific_productivity_mw_per_m3s_per_m: Some(0.5),
                    max_generation_mw: 250.0,
                    penalties: hydro_penalties(),
                    ..Default::default()
                },
            )
        })
        .collect();

    let blocks: Vec<Block> = block_hours
        .iter()
        .enumerate()
        .map(|(index, &duration_hours)| Block {
            index,
            name: format!("B{index}"),
            duration_hours,
        })
        .collect();

    let stages = (0..N_STAGES)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
                    start_date: NaiveDate::from_ymd_opt(2024, (i % 12 + 1) as u32, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(2024, ((i % 12 + 1) % 12 + 1) as u32, 1)
                        .unwrap(),
                    season_id: Some(0),
                    blocks: blocks.clone(),
                    block_mode: block_modes[i],
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

    let inflow_models = HYDRO_IDS
        .iter()
        .enumerate()
        .flat_map(|(h, &id)| {
            (0..N_STAGES).map(move |i| InflowModel {
                hydro_id: EntityId(id),
                stage_id: i32::try_from(i).expect("stage index fits i32"),
                mean_m3s: INFLOW_MEAN_M3S,
                std_m3s: inflow_std_m3s[i][h],
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
        })
        .collect();

    let load_models = (0..N_STAGES)
        .map(|i| LoadModel {
            bus_id: EntityId(BUS_ID),
            stage_id: i32::try_from(i).expect("stage index fits i32"),
            mean_mw: 0.0,
            std_mw: 0.0,
        })
        .collect();

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: HYDRO_IDS.len(),
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: N_STAGES,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 10_000.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 250.0,
                ..Default::default()
            },
            thermal: ThermalStageBounds {
                cost_per_mwh: 100.0,
            },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 400.0,
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
            n_hydros: HYDRO_IDS.len(),
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: N_STAGES,
        },
        &PenaltiesDefaults {
            hydro: hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let initial_conditions = InitialConditions {
        storage: HYDRO_IDS
            .iter()
            .map(|&id| HydroStorage {
                hydro_id: EntityId(id),
                value_hm3: INITIAL_STORAGE_HM3,
            })
            .collect(),
        filling_storage: vec![],
        past_anticipated_commitments: vec![],
        recent_observations: vec![],
        past_defluences: vec![],
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![make_thermal(
            EntityId(20),
            ThermalSpec {
                name: "T".to_string(),
                operational_start_date: start,
                bus_id: EntityId(BUS_ID),
                cost_per_mwh: 100.0,
                min_generation_mw: 0.0,
                max_generation_mw: 400.0,
                anticipated_config: None,
                ..Default::default()
            },
        )])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .build()
        .expect("two independent hydros on a two-stage system")
}

fn build_config() -> Config {
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: CfgInflowMethod::None,
            },
            cost_scale_factor: None,
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: Some(42),
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
            stopping_mode: cobre_io::config::StoppingMode::Any,
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            parallelism: cobre_io::config::ParallelismConfig::default(),
            scenario_source: None,
            selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig::default(),
        simulation: IoSimulationConfig::default(),
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

fn root_node(setup: &StudySetup) -> NodePos {
    let graph = &setup.node_graph;
    (0..graph.nodes.len())
        .map(NodePos)
        .find(|&pos| graph.nodes[pos].stage == StageIdx(0))
        .expect("study must have a stage-0 node")
}

fn end_storage_hm3(setup: &StudySetup, inflow_eta: &[f64]) -> Vec<f64> {
    let template =
        capture_patched_node_template_with_inflow_noise(setup, root_node(setup), inflow_eta);
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
    solver.load_model(&template);
    let view = solver.solve(None).expect("root stage LP must solve");
    let state = setup.stage_state();
    (0..HYDRO_IDS.len())
        .map(|h| {
            let col = state.lp_column_for_state(StateDim::new(h)).get();
            let scale = template.col_scale.get(col).copied().unwrap_or(1.0);
            view.primal[col] * scale
        })
        .collect()
}

#[test]
#[ignore = "chronological multi-block inflow noise is rejected at setup; this study sets a positive inflow standard deviation on such a stage"]
fn chronological_inflow_noise_moves_only_its_own_hydro() {
    let setup = build_setup_in_code(build_system(), &build_config());
    let zeta_hm3_per_m3s: f64 = BLOCK_HOURS.iter().sum::<f64>() * M3S_TO_HM3;

    let baseline = end_storage_hm3(&setup, &[0.0, 0.0]);
    for (h, &storage) in baseline.iter().enumerate() {
        let expected = INITIAL_STORAGE_HM3 + zeta_hm3_per_m3s * INFLOW_MEAN_M3S;
        assert!(
            (storage - expected).abs() < 1e-6,
            "hydro {h}: with a zero draw the stage must store its mean inflow, \
             expected {expected}, got {storage}"
        );
    }

    let shocked = end_storage_hm3(&setup, &[0.0, 1.0]);
    let delta: Vec<f64> = shocked.iter().zip(&baseline).map(|(s, b)| s - b).collect();
    let expected_delta = [0.0, zeta_hm3_per_m3s * INFLOW_STD_M3S[1]];
    for h in 0..HYDRO_IDS.len() {
        assert!(
            (delta[h] - expected_delta[h]).abs() < 1e-6,
            "a one-hot draw on hydro 1 must change hydro {h}'s end storage by \
             {} hm³, got {} (all deltas: {delta:?})",
            expected_delta[h],
            delta[h]
        );
    }
}

#[test]
fn chronological_multi_block_inflow_noise_is_rejected_at_setup() {
    match try_build_setup_in_code(build_system(), &build_config()) {
        Err(SddpError::Validation(msg)) => {
            assert!(
                msg.contains("not implemented")
                    && msg.contains("stage 0")
                    && msg.contains("hydro 1"),
                "reject message must name the rejection, stage, and hydro: {msg}"
            );
        }
        other => panic!(
            "expected a chronological multi-block inflow-noise Validation reject, got {other:?}"
        ),
    }
}

#[test]
fn chronological_multi_block_inflow_noise_is_flagged() {
    let system = build_system();
    let stochastic = common::stochastic_in_code(&system);
    assert_eq!(
        chronological_multi_block_inflow_noise(&system, stochastic.par()),
        Some((0, 1))
    );
}

#[test]
fn single_block_chronological_inflow_noise_is_not_flagged() {
    let system = build_system_with(
        [BlockMode::Chronological; N_STAGES],
        &[744.0],
        [INFLOW_STD_M3S; N_STAGES],
        [None, None],
    );
    let stochastic = common::stochastic_in_code(&system);
    assert_eq!(
        chronological_multi_block_inflow_noise(&system, stochastic.par()),
        None
    );
    assert!(try_build_setup_in_code(system, &build_config()).is_ok());
}

#[test]
fn parallel_multi_block_inflow_noise_is_not_flagged() {
    let system = build_system_with(
        [BlockMode::Parallel; N_STAGES],
        &BLOCK_HOURS,
        [INFLOW_STD_M3S; N_STAGES],
        [None, None],
    );
    let stochastic = common::stochastic_in_code(&system);
    assert_eq!(
        chronological_multi_block_inflow_noise(&system, stochastic.par()),
        None
    );
    assert!(try_build_setup_in_code(system, &build_config()).is_ok());
}

#[test]
fn inflow_noise_is_flagged_only_on_the_chronological_stage() {
    let block_modes = [BlockMode::Parallel, BlockMode::Chronological];

    let not_flagged = build_system_with(
        block_modes,
        &BLOCK_HOURS,
        [[10.0, 20.0], [0.0, 0.0]],
        [None, None],
    );
    let not_flagged_stochastic = common::stochastic_in_code(&not_flagged);
    assert_eq!(
        chronological_multi_block_inflow_noise(&not_flagged, not_flagged_stochastic.par()),
        None
    );
    assert!(try_build_setup_in_code(not_flagged, &build_config()).is_ok());

    let flagged = build_system_with(
        block_modes,
        &BLOCK_HOURS,
        [[0.0, 0.0], [10.0, 0.0]],
        [None, None],
    );
    let flagged_stochastic = common::stochastic_in_code(&flagged);
    assert_eq!(
        chronological_multi_block_inflow_noise(&flagged, flagged_stochastic.par()),
        Some((1, 1))
    );
}

#[test]
fn prefilling_hydro_inflow_noise_is_not_flagged() {
    let system = build_system_with(
        [BlockMode::Chronological; N_STAGES],
        &BLOCK_HOURS,
        [[0.0, 20.0], [0.0, 0.0]],
        [None, Some(1)],
    );
    let stochastic = common::stochastic_in_code(&system);
    assert_eq!(
        chronological_multi_block_inflow_noise(&system, stochastic.par()),
        None
    );
    assert!(try_build_setup_in_code(system, &build_config()).is_ok());
}

#[test]
fn committed_chronological_decks_still_load() {
    let deterministic_decks = [
        "d46-travel-time-chronological",
        "d49-travel-time-chronological-arrival",
        "d50-travel-time-plain-tributary-confluence",
    ];
    let deterministic_root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/deterministic");
    for deck in deterministic_decks {
        common::fresh_setup_with(&deterministic_root.join(deck), |_| {});
    }

    let fixture_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/chronological_storage");
    common::fresh_setup_with(&fixture_dir, |_| {});
}

#[test]
fn builder_accepts_chronological_multi_block_inflow_noise() {
    let system = build_system();
    let stochastic = common::stochastic_in_code(&system);
    assert!(
        chronological_multi_block_inflow_noise(&system, stochastic.par()).is_some(),
        "fixture must be the same system the setup-level rejection test builds"
    );

    let models = PrepareHydroModelsResult::default_from_system(&system);
    let templates = build_stage_templates_resolving_layout(
        &system,
        InflowNonNegativityMethod::None,
        stochastic.par(),
        stochastic.normal(),
        &models.production,
        &models.evaporation,
        &ResolvedParameters::default(),
    )
    .expect("the builder must accept chronological multi-block inflow noise");
    assert_eq!(
        templates.geometry_per_stage[0].water_balance.len(),
        4,
        "two hydros times two chronological blocks"
    );
}
