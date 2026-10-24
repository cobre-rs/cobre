//! Runtime drift-absorption regressions: a stored state value seeded a hair
//! outside its admissible box must train (or simulate) to completion rather
//! than abort.
//!
//! Each fixture is synthetic: the deck that first surfaced the idle-thermal
//! shape below is unavailable, so the case is reproduced by mechanism, not by
//! the original data. Every fixture is built via `build_setup_in_code`
//! (`tests/common/mod.rs`), which bypasses `cobre-io` the same way
//! `anticipated_commitment_drifted_over_cap_is_absorbed`
//! (`tests/anticipated_scenarios.rs`) does. Most are within-drift seeds
//! `cobre-io` would accept; `anticipated_commitment_over_cap_seed_is_clamped`
//! injects a genuine over-commitment in-code (one `cobre-io` would reject at
//! load time) to prove the setup-time seed clamp absorbs it too.
//!
//! Each test asserts a positive completion signal — `Ok` with a finite
//! `final_lb`, or a finite simulated cost — never merely the absence of a
//! panic.
//!
//! The initial storage is the one seed whose box is not the stage-0 box: it is
//! projected onto the hydro's physical range, so a stage-0 end-of-stage storage
//! bound that excludes the seed must leave the seed where it was.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::items_after_statements,
    clippy::needless_pass_by_value,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

mod common;

use chrono::{NaiveDate, TimeDelta};
use cobre_core::entities::hydro::HydroGenerationModel;
use cobre_core::entities::thermal::AnticipatedConfig;
use cobre_core::scenario::InflowModel;
use cobre_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
};
use cobre_core::{
    AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
    ContractBlockBounds, EntityId, HydroBlockBounds, HydroPastDefluence, HydroPenalties,
    HydroStageBounds, HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties,
    NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
    ResolvedPenalties, System, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
};
use cobre_io::config::{
    Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig, InflowNonNegativityMethod,
    ModelingConfig, ParallelismConfig, PolicyConfig, RowSelectionConfig,
    SimulationConfig as IoSimulationConfig, SimulationSelection, StoppingMode, StoppingRuleConfig,
    TrainingConfig, TrainingSelection, TrainingSolverConfig, UpperBoundEvaluationConfig,
};
use cobre_sddp::test_support::{oracle_initial_state, stage_state_box_bounds};
use cobre_solver::ActiveSolver;

use common::builders::{
    BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
};

const TRAIN_ITERATIONS: u32 = 4;
const N_THREADS: usize = 1;

fn daily_stage_dates(
    anchor: NaiveDate,
    n_stages: usize,
    days_per_stage: i64,
) -> Vec<(NaiveDate, NaiveDate)> {
    (0..n_stages)
        .map(|i| {
            (
                anchor + TimeDelta::days(days_per_stage * i as i64),
                anchor + TimeDelta::days(days_per_stage * (i as i64 + 1)),
            )
        })
        .collect()
}

/// One `AnticipatedCommitmentHistory` window per `(slot, value_mw)` pair,
/// covering slot `i`'s own study-stage date span exactly (same
/// `anchor`/`days_per_stage` as [`daily_stage_dates`]) so `StageCalendar::coverage`
/// resolves full coverage and the seed lands in `commit_out`'s slot `i`.
fn commitment_seed_windows(
    thermal_id: EntityId,
    anchor: NaiveDate,
    days_per_stage: i64,
    seeds: &[(usize, f64)],
) -> Vec<AnticipatedCommitmentHistory> {
    seeds
        .iter()
        .map(|&(slot, value_mw)| AnticipatedCommitmentHistory {
            thermal_id,
            start_date: anchor + TimeDelta::days(days_per_stage * slot as i64),
            end_date: anchor + TimeDelta::days(days_per_stage * (slot as i64 + 1)),
            value_mw,
        })
        .collect()
}

fn zero_hydro_penalties() -> HydroPenalties {
    HydroPenalties {
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
        inflow_nonnegativity_cost: 0.0,
    }
}

fn zero_penalties(n_hydros: usize, n_stages: usize) -> ResolvedPenalties {
    ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages,
        },
        &PenaltiesDefaults {
            hydro: zero_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    )
}

fn build_config(iteration_limit: u32) -> Config {
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: InflowNonNegativityMethod::Penalty,
            },
            cost_scale_factor: None,
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: Some(42),
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                limit: iteration_limit,
            }]),
            stopping_mode: StoppingMode::Any,
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            parallelism: ParallelismConfig::default(),
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

fn build_config_with_simulation(iteration_limit: u32) -> Config {
    Config {
        simulation: IoSimulationConfig {
            enabled: true,
            selection: Some(SimulationSelection::Sampled { num_scenarios: 1 }),
            ..IoSimulationConfig::default()
        },
        ..build_config(iteration_limit)
    }
}

fn assert_trains_to_completion(system: System, config: &Config) {
    let mut setup = common::build_setup_in_code(system, config);
    let comm = common::StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

    let outcome = setup
        .train(&mut solver, &comm, N_THREADS, ActiveSolver::new, None, None)
        .expect("train must not return Err");

    assert!(
        outcome.error.is_none(),
        "a state seeded a hair outside its admissible box must train to completion: the \
         drift is numerical noise the outgoing-state seam absorbs, not an over-commitment, \
         and aborting on it is a false infeasibility. Got: {:?}",
        outcome.error
    );
    assert!(
        outcome.result.final_lb.is_finite(),
        "a completed training run must report a finite final_lb, got {}",
        outcome.result.final_lb
    );
}

/// A single anticipated thermal on its own bus (no hydro): the bus's default
/// unbounded, zero-cost deficit/excess segments absorb any load/generation
/// mismatch, so the only feasibility question under test is the commitment
/// ring's fishing equality against `[floor_mw, cap_mw]` (or the `stage_override`
/// cell). `seeds` maps ring slot `i` (matures at study stage `i`) to its
/// pre-study committed value; an unlisted slot stays at its `0.0` default.
fn build_commitment_system(
    n_stages: usize,
    k_max: u32,
    floor_mw: f64,
    cap_mw: f64,
    seeds: &[(usize, f64)],
    stage_override: Option<(usize, f64, f64)>,
) -> System {
    let bus_id = EntityId(1);
    let ant_id = EntityId(2);
    const DAYS_PER_STAGE: i64 = 30;

    let bus = make_bus(bus_id, BusSpec::default());

    let thermal = make_thermal(
        ant_id,
        ThermalSpec {
            name: "T_ant".to_string(),
            bus_id,
            min_generation_mw: floor_mw,
            max_generation_mw: cap_mw,
            cost_per_mwh: 10.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(k_max)),
            ..Default::default()
        },
    );

    let anchor = NaiveDate::from_ymd_opt(2024, 1, 1).expect("2024-01-01 is a valid date");
    let stage_dates = daily_stage_dates(anchor, n_stages, DAYS_PER_STAGE);
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
                    start_date: stage_dates[i].0,
                    end_date: stage_dates[i].1,
                    blocks: vec![Block {
                        index: 0,
                        name: "S".to_string(),
                        duration_hours: (DAYS_PER_STAGE * 24) as f64,
                    }],
                    block_mode: BlockMode::Parallel,
                    state_config: StageStateConfig {
                        storage: false,
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

    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
            k_max: k_max as usize,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 0.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds::default(),
            thermal: ThermalStageBounds { cost_per_mwh: 10.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: floor_mw,
                max_generation_mw: cap_mw,
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
    if let Some((stage_idx, min_mw, max_mw)) = stage_override {
        *bounds.thermal_block_base_mut(0, stage_idx) = ThermalBlockBounds {
            min_generation_mw: min_mw,
            max_generation_mw: max_mw,
        };
    }

    let penalties = zero_penalties(0, n_stages);

    let past_anticipated_commitments =
        commitment_seed_windows(ant_id, anchor, DAYS_PER_STAGE, seeds);

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .stages(stages)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(InitialConditions {
            past_anticipated_commitments,
            ..Default::default()
        })
        .build()
        .expect("build_commitment_system: valid system")
}

#[test]
fn reporter_shape_idle_thermal_negative_hair_at_zero_floor_trains() {
    const K_MAX: u32 = 6;
    let system = build_commitment_system(K_MAX as usize, K_MAX, 0.0, 50.0, &[(0, -1.5e-4)], None);
    assert_trains_to_completion(system, &build_config(TRAIN_ITERATIONS));
}

#[test]
fn commitment_hair_below_positive_must_run_floor_trains() {
    const FLOOR_MW: f64 = 20.0;
    let seed_mw = FLOOR_MW * (1.0 - 1e-12);
    let system = build_commitment_system(3, 2, FLOOR_MW, 100.0, &[(0, seed_mw)], None);
    assert_trains_to_completion(system, &build_config(TRAIN_ITERATIONS));
}

#[test]
fn commitment_hair_above_cap_trains() {
    const CAP_MW: f64 = 100.0;
    let seed_mw = CAP_MW * (1.0 + 1e-12);
    let system = build_commitment_system(3, 2, 0.0, CAP_MW, &[(0, seed_mw)], None);
    assert_trains_to_completion(system, &build_config(TRAIN_ITERATIONS));
}

/// A genuine over-cap seed — 50% past the cap, orders of magnitude beyond
/// `envelope_tolerance` — is projected onto `[floor_mw, cap_mw]` by
/// `build_initial_state`'s seed-time clamp before training ever starts, so it
/// trains rather than aborting — a genuine over-commitment, unlike the
/// sub-tolerance hair `commitment_hair_above_cap_trains` exercises.
#[test]
fn anticipated_commitment_over_cap_seed_is_clamped() {
    const FLOOR_MW: f64 = 0.0;
    const CAP_MW: f64 = 100.0;
    let seed_mw = CAP_MW * 1.5;
    let system = build_commitment_system(3, 2, FLOOR_MW, CAP_MW, &[(0, seed_mw)], None);
    let config = build_config(TRAIN_ITERATIONS);

    let mut setup = common::build_setup_in_code(system, &config);
    let commit_out = setup.stage_state().commit_out.clone();
    let initial_state = oracle_initial_state(&setup);

    for j in commit_out.clone() {
        assert!(
            (FLOOR_MW - 1e-6..=CAP_MW + 1e-6).contains(&initial_state[j]),
            "commit_out[{j}] = {} must be clamped into [{FLOOR_MW}, {CAP_MW}] by \
             build_initial_state's seed-time projection, not left at the raw \
             over-cap seed {seed_mw}",
            initial_state[j]
        );
    }
    assert!(
        (initial_state[commit_out.start] - seed_mw).abs() > 1.0,
        "fixture sanity: the seed must actually be moved by the clamp, else the \
         box check above is vacuous"
    );

    let comm = common::StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
    let outcome = setup
        .train(&mut solver, &comm, N_THREADS, ActiveSolver::new, None, None)
        .expect("train must not return Err");

    assert!(
        outcome.error.is_none(),
        "a genuine over-cap commitment seed is clamped at setup time and must train \
         to completion — the retired runtime verdict must not resurface. Got: {:?}",
        outcome.error
    );
    assert!(
        outcome.result.final_lb.is_finite(),
        "a completed training run must report a finite final_lb, got {}",
        outcome.result.final_lb
    );
}

#[test]
fn dormant_zero_zero_delivery_stage_trains() {
    let system = build_commitment_system(3, 2, 0.0, 100.0, &[(1, 1.0e-4)], Some((1, 0.0, 0.0)));
    assert_trains_to_completion(system, &build_config(TRAIN_ITERATIONS));
}

/// A two-hydro upstream-to-downstream cascade declaring a travel-time arc:
/// the only way to seed the in-transit water-bucket family. Both hydros carry
/// a permissive `[0, cap]` storage box and no load, so the bus's own
/// unbounded, zero-cost deficit/excess segments absorb generation freely; the
/// only feasibility question under test is the seeded bucket's own
/// `[0, inf)` box.
fn build_transit_bucket_system(n_stages: usize, travel_time_hours: f64, seed_m3s: f64) -> System {
    let bus_id = EntityId(1);
    let upstream_id = EntityId(2);
    let downstream_id = EntityId(3);
    const CAP_HM3: f64 = 500.0;
    const STAGE_HOURS: f64 = 24.0;

    let bus = make_bus(bus_id, BusSpec::default());

    let downstream = make_hydro(
        downstream_id,
        HydroSpec {
            bus_id,
            max_storage_hm3: CAP_HM3,
            max_turbined_m3s: 500.0,
            max_generation_mw: 1_000.0,
            generation_model: HydroGenerationModel::ConstantProductivity,
            ..Default::default()
        },
    );
    let upstream = make_hydro(
        upstream_id,
        HydroSpec {
            bus_id,
            downstream_id: Some(downstream_id),
            travel_time_hours: Some(travel_time_hours),
            max_storage_hm3: CAP_HM3,
            max_turbined_m3s: 500.0,
            max_generation_mw: 1_000.0,
            generation_model: HydroGenerationModel::ConstantProductivity,
            ..Default::default()
        },
    );

    let base = NaiveDate::from_ymd_opt(2024, 1, 1).expect("2024-01-01 is a valid date");
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| {
            let start = base + TimeDelta::days(i as i64);
            make_stage(
                i,
                StageSpec {
                    start_date: start,
                    end_date: start + TimeDelta::days(1),
                    blocks: vec![Block {
                        index: 0,
                        name: "S".to_string(),
                        duration_hours: STAGE_HOURS,
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

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
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
                max_storage_hm3: CAP_HM3,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 500.0,
                max_generation_mw: 1_000.0,
                ..HydroBlockBounds::default()
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

    let penalties = zero_penalties(2, n_stages);

    let past_defluences = vec![HydroPastDefluence {
        hydro_id: upstream_id,
        start_date: base - TimeDelta::days(1),
        end_date: base,
        value_m3s: seed_m3s,
    }];

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![downstream, upstream])
        .stages(stages)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(InitialConditions {
            storage: vec![
                HydroStorage {
                    hydro_id: downstream_id,
                    value_hm3: 100.0,
                },
                HydroStorage {
                    hydro_id: upstream_id,
                    value_hm3: 100.0,
                },
            ],
            past_defluences,
            ..Default::default()
        })
        .build()
        .expect("build_transit_bucket_system: valid system")
}

#[test]
fn transit_bucket_drifting_negative_trains() {
    let system = build_transit_bucket_system(4, 24.0, -1.0e-4);
    assert_trains_to_completion(system, &build_config(TRAIN_ITERATIONS));
}

/// A single hydro with zero inflow and no load: the reservoir has nothing
/// forcing it off its seeded value, so the only feasibility question under
/// test is the seeded initial storage against `[0, cap_hm3]`.
fn build_storage_system(n_stages: usize, cap_hm3: f64, seed_hm3: f64) -> System {
    build_storage_system_with(n_stages, cap_hm3, seed_hm3, 0.0, (0.0, cap_hm3))
}

/// [`build_storage_system`] with a constant `inflow_m3s` and stage 0's
/// end-of-stage storage bound replaced by `stage0_storage_hm3` (`(min, max)`,
/// hm³); every later stage keeps `[0, cap_hm3]`.
fn build_storage_system_with(
    n_stages: usize,
    cap_hm3: f64,
    seed_hm3: f64,
    inflow_m3s: f64,
    stage0_storage_hm3: (f64, f64),
) -> System {
    let bus_id = EntityId(1);
    let hydro_id = EntityId(2);
    const DAYS_PER_STAGE: i64 = 30;

    let bus = make_bus(bus_id, BusSpec::default());
    let hydro = make_hydro(
        hydro_id,
        HydroSpec {
            bus_id,
            max_storage_hm3: cap_hm3,
            max_turbined_m3s: 100.0,
            max_generation_mw: 100.0,
            generation_model: HydroGenerationModel::ConstantProductivity,
            ..Default::default()
        },
    );

    let anchor = NaiveDate::from_ymd_opt(2024, 1, 1).expect("2024-01-01 is a valid date");
    let stage_dates = daily_stage_dates(anchor, n_stages, DAYS_PER_STAGE);
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
                    start_date: stage_dates[i].0,
                    end_date: stage_dates[i].1,
                    blocks: vec![Block {
                        index: 0,
                        name: "S".to_string(),
                        duration_hours: (DAYS_PER_STAGE * 24) as f64,
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
            hydro_id,
            stage_id: i as i32,
            mean_m3s: inflow_m3s,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let mut bounds = ResolvedBounds::new(
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
                max_storage_hm3: cap_hm3,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 100.0,
                ..HydroBlockBounds::default()
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

    let first_stage_bounds = bounds.hydro_bounds_mut(0, 0);
    (
        first_stage_bounds.min_storage_hm3,
        first_stage_bounds.max_storage_hm3,
    ) = stage0_storage_hm3;

    let penalties = zero_penalties(1, n_stages);

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(InitialConditions {
            storage: vec![HydroStorage {
                hydro_id,
                value_hm3: seed_hm3,
            }],
            ..Default::default()
        })
        .build()
        .expect("build_storage_system: valid system")
}

#[test]
fn storage_drifting_above_cap_simulates() {
    const CAP_HM3: f64 = 100.0;
    let seed_hm3 = CAP_HM3 * (1.0 + 1e-9);
    let system = build_storage_system(3, CAP_HM3, seed_hm3);
    let config = build_config_with_simulation(TRAIN_ITERATIONS);
    let mut setup = common::build_setup_in_code(system, &config);

    let results = common::run_simulation(&mut setup, N_THREADS);

    assert!(
        !results.is_empty(),
        "simulation must produce at least one scenario result after absorbing the storage drift"
    );
    for result in &results {
        assert!(
            result.total_cost.is_finite(),
            "scenario {} total_cost must be finite after absorbing the storage drift, got {}",
            result.scenario_id,
            result.total_cost
        );
    }
}

const STORAGE_DIM: usize = 0;

/// Seeds 50 hm³ in a `[0, 100]` reservoir fed 20 m³/s, with stage 0's
/// end-of-stage storage bound set to `stage0_storage_hm3`, which excludes the
/// seed: the setup must keep the seed and training must complete from it.
fn assert_initial_storage_kept(stage0_storage_hm3: (f64, f64)) {
    const CAP_HM3: f64 = 100.0;
    const SEED_HM3: f64 = 50.0;
    let system = build_storage_system_with(3, CAP_HM3, SEED_HM3, 20.0, stage0_storage_hm3);
    let mut setup = common::build_setup_in_code(system, &build_config(TRAIN_ITERATIONS));

    let (lower, upper) = stage_state_box_bounds(&setup, 0);
    assert!(
        SEED_HM3 < lower[STORAGE_DIM] || SEED_HM3 > upper[STORAGE_DIM],
        "fixture: the seed must lie outside stage 0's end-of-stage storage box [{}, {}]",
        lower[STORAGE_DIM],
        upper[STORAGE_DIM]
    );
    assert_eq!(
        oracle_initial_state(&setup)[STORAGE_DIM],
        SEED_HM3,
        "an initial storage inside the physical range must not be projected onto stage 0's \
         end-of-stage storage bound: that creates or removes water before the first solve"
    );

    let comm = common::StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
    let outcome = setup
        .train(&mut solver, &comm, N_THREADS, ActiveSolver::new, None, None)
        .expect("train must not return Err");
    assert!(
        outcome.error.is_none(),
        "training from the declared initial storage must complete, got {:?}",
        outcome.error
    );
    assert!(
        outcome.result.final_lb.is_finite(),
        "a completed training run must report a finite final_lb, got {}",
        outcome.result.final_lb
    );
}

#[test]
fn initial_storage_below_stage0_storage_floor_is_not_raised() {
    assert_initial_storage_kept((80.0, 100.0));
}

#[test]
fn initial_storage_above_stage0_storage_ceiling_is_not_lowered() {
    assert_initial_storage_kept((0.0, 40.0));
}
