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
use cobre_sddp::StudySetup;
use cobre_sddp::indexer::StateDim;
use cobre_sddp::setup::{NodePos, StageIdx};
use cobre_sddp::test_support::capture_patched_node_template_with_inflow_noise;
use cobre_solver::{ActiveSolver, SolverInterface};

use common::build_setup_in_code;
use common::builders::{
    BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
};

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
        .map(|&id| {
            make_hydro(
                EntityId(id),
                HydroSpec {
                    name: format!("H{id}"),
                    operational_start_date: start,
                    bus_id: EntityId(BUS_ID),
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

    let blocks: Vec<Block> = BLOCK_HOURS
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
                    block_mode: BlockMode::Chronological,
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
        .zip(INFLOW_STD_M3S)
        .flat_map(|(&id, std_m3s)| {
            (0..N_STAGES).map(move |i| InflowModel {
                hydro_id: EntityId(id),
                stage_id: i32::try_from(i).expect("stage index fits i32"),
                mean_m3s: INFLOW_MEAN_M3S,
                std_m3s,
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
        .expect("two independent hydros on a chronological two-block stage")
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
#[ignore = "known defect: chronological inflow noise is patched onto another hydro's water-balance rows"]
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
