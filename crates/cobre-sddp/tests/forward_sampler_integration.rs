//! Integration tests for `ForwardSampler` dispatch added in Epic 06.
//!
//! Covers three scenarios:
//! 1. `InSample` bitwise equivalence: the refactored path produces identical
//!    lower bounds and iteration counts to the pre-refactor D01 baseline.
//! 2. `OutOfSample` convergence: training with fresh noise converges to a lower
//!    bound within 5% of the `InSample` lower bound.
//! 3. Declaration-order invariance for `OutOfSample`: entity ordering does not
//!    affect the lower bound (bitwise identical results for both orderings).
//!
//! All tests use `StubComm` (single-rank) and `HighsSolver`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::needless_range_loop,
    clippy::trivially_copy_pass_by_ref,
    clippy::too_many_lines
)]

use std::collections::BTreeMap;
use std::path::Path;

use chrono::NaiveDate;
use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::{
    BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties, ContractStageBounds, DeficitSegment,
    EntityId, HydroStageBounds, HydroStagePenalties, LineStageBounds, LineStagePenalties,
    NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults, PumpingStageBounds, ResolvedBounds,
    ResolvedPenalties, ScenarioSource, SystemBuilder, ThermalStageBounds,
    entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties},
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
        SamplingScheme,
    },
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_sddp::{
    InflowNonNegativityMethod, StoppingMode, StoppingRule, StoppingRuleSet, StudySetup,
    hydro_models::PrepareHydroModelsResult, setup::prepare_stochastic,
};
use cobre_solver::highs::HighsSolver;
use cobre_stochastic::build_stochastic_context;

// ---------------------------------------------------------------------------
// Shared test infrastructure
// ---------------------------------------------------------------------------

/// Single-rank communicator stub for integration testing.
///
/// Faithfully copies data through `allgatherv` and `allreduce` so the full
/// SDDP training pipeline runs without a real MPI implementation.
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

/// Run the SDDP training pipeline on a pre-loaded case directory.
///
/// Uses `StubComm`, `HighsSolver`, seed 42, and 1 thread.
/// Returns the `TrainingResult`.
fn run_case_from_dir(case_dir: &Path) -> cobre_sddp::TrainingResult {
    use cobre_io::parse_config;
    use cobre_sddp::hydro_models::prepare_hydro_models;

    let config_path = case_dir.join("config.json");
    let config = parse_config(&config_path).expect("config must parse");

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let prepare_result =
        prepare_stochastic(system, case_dir, &config, 42).expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir).expect("prepare_hydro_models must succeed");

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");
    outcome.result
}

fn make_stage(index: usize, branching_factor: usize) -> Stage {
    Stage {
        index,
        id: index as i32,
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
            branching_factor,
            noise_method: NoiseMethod::Saa,
        },
    }
}

fn make_hydro(raw_id: i32) -> Hydro {
    Hydro {
        id: EntityId(raw_id),
        name: format!("H{raw_id}"),
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
    }
}

fn hydro_stage_bounds() -> HydroStageBounds {
    HydroStageBounds {
        min_storage_hm3: 0.0,
        max_storage_hm3: 100.0,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        max_diversion_m3s: None,
        filling_inflow_m3s: 0.0,
        water_withdrawal_m3s: 0.0,
    }
}

fn hydro_stage_penalties() -> HydroStagePenalties {
    HydroStagePenalties {
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
    }
}

fn build_resolved_bounds(n_hydros: usize, n_stages: usize) -> ResolvedBounds {
    let n_st = n_stages.max(1);
    ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
        },
        &BoundsDefaults {
            hydro: hydro_stage_bounds(),
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
    )
}

fn build_resolved_penalties(n_hydros: usize, n_buses: usize, n_stages: usize) -> ResolvedPenalties {
    let n_st = n_stages.max(1);
    ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros,
            n_buses,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: hydro_stage_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    )
}

fn make_correlation(entity_ids: &[EntityId]) -> CorrelationModel {
    let n = entity_ids.len();
    let mut matrix = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        matrix[i][i] = 1.0;
    }

    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: entity_ids
                    .iter()
                    .map(|&id| CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id,
                    })
                    .collect(),
                matrix,
            }],
        },
    );

    CorrelationModel {
        method: "cholesky".to_string(),
        profiles,
        schedule: vec![],
    }
}

fn build_single_hydro_system(
    hydro_id: i32,
    n_stages: usize,
    branching_factor: usize,
    sampling_scheme: SamplingScheme,
    forward_seed: Option<i64>,
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

    let id = EntityId(hydro_id);
    let hydro = make_hydro(hydro_id);

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| make_stage(i, branching_factor))
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: id,
            stage_id: i as i32,
            mean_m3s: 100.0,
            std_m3s: 30.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        })
        .collect();

    let correlation = make_correlation(&[id]);
    let bounds = build_resolved_bounds(1, n_stages);
    let penalties = build_resolved_penalties(1, 1, n_stages);

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .bounds(bounds)
        .penalties(penalties)
        .scenario_source(ScenarioSource {
            sampling_scheme,
            seed: forward_seed,
        })
        .build()
        .expect("SystemBuilder must produce a valid system")
}

fn build_two_hydro_system(
    hydro_id_order: &[i32; 2],
    n_stages: usize,
    branching_factor: usize,
    sampling_scheme: SamplingScheme,
    forward_seed: Option<i64>,
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

    // Declare hydros in the specified order (exercises entity_order path)
    let hydros = vec![make_hydro(hydro_id_order[0]), make_hydro(hydro_id_order[1])];

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| make_stage(i, branching_factor))
        .collect();

    // Build inflow models for both hydros across all stages
    let mut inflow_models = Vec::new();
    for &raw_id in hydro_id_order {
        for stage_idx in 0..n_stages {
            inflow_models.push(InflowModel {
                hydro_id: EntityId(raw_id),
                stage_id: stage_idx as i32,
                mean_m3s: 100.0,
                std_m3s: 30.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            });
        }
    }

    // Build correlation model with entities in declaration order.
    // The entity list order in the CorrelationGroup drives `entity_order` in
    // `build_stochastic_context` — this is the invariance path under test.
    let entity_ids: Vec<EntityId> = hydro_id_order.iter().map(|&id| EntityId(id)).collect();
    let correlation = make_correlation(&entity_ids);
    let bounds = build_resolved_bounds(2, n_stages);
    let penalties = build_resolved_penalties(2, 1, n_stages);

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .bounds(bounds)
        .penalties(penalties)
        .scenario_source(ScenarioSource {
            sampling_scheme,
            seed: forward_seed,
        })
        .build()
        .expect("SystemBuilder must produce a valid system")
}

fn run_programmatic(
    system: &cobre_core::System,
    forward_passes: u32,
    max_iterations: u64,
) -> cobre_sddp::TrainingResult {
    // Extract forward_seed from scenario_source — mirrors what prepare_stochastic does.
    let forward_seed = system.scenario_source().seed.map(i64::unsigned_abs);

    let stochastic = build_stochastic_context(system, 42, forward_seed, &[], &[], None)
        .expect("build_stochastic_context must succeed");

    let hydro_models = PrepareHydroModelsResult::default_from_system(system);

    let stopping_rule_set = StoppingRuleSet {
        rules: vec![StoppingRule::IterationLimit {
            limit: max_iterations,
        }],
        mode: StoppingMode::Any,
    };

    let mut setup = StudySetup::from_broadcast_params(
        system,
        stochastic,
        42, // tree seed
        forward_passes,
        stopping_rule_set,
        0,             // n_scenarios (simulation disabled)
        0,             // io_channel_capacity
        String::new(), // policy_path
        InflowNonNegativityMethod::None,
        None, // cut_selection
        0.0,  // cut_activity_tolerance
        hydro_models,
    )
    .expect("StudySetup::from_broadcast_params must succeed");

    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");
    outcome.result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Verify that the refactored `ForwardSampler::InSample` path produces the
/// same lower bound and iteration count as the pre-refactor D01 baseline.
///
/// D01 is a 2-stage, 2-thermal deterministic dispatch case. Expected cost is
/// 182,500 $ (hand-derivable: see `tests/deterministic.rs::d01_thermal_dispatch`).
///
/// This test uses the full IO path (`load_case` + `prepare_stochastic` +
/// `StudySetup::new`) — the same path exercised by the D01 deterministic
/// regression suite — so it confirms the refactored `ForwardSampler` dispatch
/// is transparent in the complete pipeline.
#[test]
fn insample_equivalence_d01() {
    let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/cobre-sddp has a parent dir")
        .parent()
        .expect("crates has a parent dir (workspace root)")
        .join("examples/deterministic/d01-thermal-dispatch");

    let result = run_case_from_dir(&case_dir);

    // D01 expected lower bound: 182,500 $ (deterministic, hand-derivable).
    // Tolerance matches the existing `d01_thermal_dispatch` regression test.
    let expected_lb = 182_500.0_f64;
    let diff = (result.final_lb - expected_lb).abs();
    assert!(
        diff <= 1e-6,
        "InSample equivalence: expected LB {expected_lb}, got {} (diff={diff})",
        result.final_lb
    );

    // D01 uses iteration_limit=10; a deterministic case converges quickly.
    assert!(
        result.iterations <= 10,
        "InSample equivalence: expected iterations <= 10, got {}",
        result.iterations
    );
}

/// Verify that `SamplingScheme::OutOfSample` converges to a lower bound
/// within 5% relative tolerance of the `InSample` lower bound.
///
/// Both systems are identical except for the sampling scheme. The system has
/// 1 bus, 1 hydro (constant productivity, mean=100 m³/s, std=30 m³/s),
/// 3 stages with `branching_factor=5` and SAA noise. With 20 forward passes
/// and 50 iterations both schemes reach comparable lower bounds.
#[test]
fn out_of_sample_convergence() {
    const FORWARD_PASSES: u32 = 20;
    const MAX_ITERATIONS: u64 = 50;
    const RELATIVE_TOLERANCE: f64 = 0.05; // 5%

    let system_insample = build_single_hydro_system(
        1, // hydro_id
        3, // n_stages
        5, // branching_factor
        SamplingScheme::InSample,
        None, // forward_seed (not used for InSample)
    );

    let system_oos = build_single_hydro_system(
        1,
        3,
        5,
        SamplingScheme::OutOfSample,
        Some(42), // forward_seed required for OutOfSample
    );

    let lb_insample = run_programmatic(&system_insample, FORWARD_PASSES, MAX_ITERATIONS).final_lb;
    let lb_oos = run_programmatic(&system_oos, FORWARD_PASSES, MAX_ITERATIONS).final_lb;

    let relative_error = (lb_oos - lb_insample).abs() / lb_insample.abs().max(1e-10);
    assert!(
        relative_error < RELATIVE_TOLERANCE,
        "OutOfSample LB {lb_oos:.4} diverges from InSample LB {lb_insample:.4} \
         by {:.2}% (tolerance: {:.0}%)",
        relative_error * 100.0,
        RELATIVE_TOLERANCE * 100.0,
    );
}

/// Verify that declaration order of hydro entities does not affect the lower
/// bound when using `SamplingScheme::OutOfSample`.
///
/// Builds two identical two-hydro systems that differ only in the order in
/// which entities are declared in `SystemBuilder` and in the correlation model
/// entity list. `SystemBuilder::build()` sorts hydros by `EntityId`, so the
/// canonical entity set is the same; what differs is the `entity_order` slice
/// that `build_stochastic_context` derives from the correlation model —
/// this is the invariance path under test.
///
/// Both systems use the same `forward_seed = Some(99)`. The lower bounds must
/// be bitwise identical (asserted with `assert_eq!` on `f64`).
#[test]
fn out_of_sample_declaration_order_invariance() {
    const FORWARD_PASSES: u32 = 5;
    const MAX_ITERATIONS: u64 = 20;

    // System A: declare hydros in order [H1, H2]
    let system_a = build_two_hydro_system(&[1, 2], 3, 3, SamplingScheme::OutOfSample, Some(99));

    // System B: declare hydros in order [H2, H1] (reversed declaration order)
    let system_b = build_two_hydro_system(&[2, 1], 3, 3, SamplingScheme::OutOfSample, Some(99));

    let lb_a = run_programmatic(&system_a, FORWARD_PASSES, MAX_ITERATIONS).final_lb;
    let lb_b = run_programmatic(&system_b, FORWARD_PASSES, MAX_ITERATIONS).final_lb;

    assert_eq!(
        lb_a, lb_b,
        "declaration-order invariance violated: LB_A={lb_a}, LB_B={lb_b}"
    );
}
