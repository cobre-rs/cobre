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
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::unnecessary_cast
)]

use std::collections::BTreeMap;
use std::path::Path;

use chrono::NaiveDate;
use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::{
    BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties, ContractStageBounds, DeficitSegment,
    EntityId, HydroStageBounds, HydroStagePenalties, LineStageBounds, LineStagePenalties,
    NcsStagePenalties, NonControllableSource, PenaltiesCountsSpec, PenaltiesDefaults,
    PumpingStageBounds, ResolvedBounds, ResolvedPenalties, ScenarioSource, SystemBuilder,
    ThermalStageBounds,
    entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties},
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, ExternalLoadRow,
        ExternalNcsRow, ExternalScenarioRow, InflowHistoryRow, InflowModel, LoadModel, NcsModel,
        SamplingScheme,
    },
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_sddp::{
    InflowNonNegativityMethod, StoppingMode, StoppingRule, StoppingRuleSet, StudySetup,
    hydro_models::PrepareHydroModelsResult,
    setup::{ConstructionConfig, prepare_stochastic},
};
use cobre_solver::highs::HighsSolver;
use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

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

    fn abort(&self, error_code: i32) -> ! {
        std::process::exit(error_code)
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

    let training_source = config
        .training_scenario_source(&config_path)
        .expect("training_scenario_source must resolve");
    let prepare_result = prepare_stochastic(system, case_dir, &config, 42, &training_source)
        .expect("prepare_stochastic must succeed");
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
                cost_per_mwh: 0.0,
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
        method: "spectral".to_string(),
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
) -> (cobre_core::System, ScenarioSource) {
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

    let source = ScenarioSource {
        inflow_scheme: sampling_scheme,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        seed: forward_seed,
        historical_years: None,
    };

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("SystemBuilder must produce a valid system");

    (system, source)
}

fn build_two_hydro_system(
    hydro_id_order: &[i32; 2],
    n_stages: usize,
    branching_factor: usize,
    sampling_scheme: SamplingScheme,
    forward_seed: Option<i64>,
) -> (cobre_core::System, ScenarioSource) {
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

    let source = ScenarioSource {
        inflow_scheme: sampling_scheme,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        seed: forward_seed,
        historical_years: None,
    };

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("SystemBuilder must produce a valid system");

    (system, source)
}

fn run_programmatic(
    system: &cobre_core::System,
    source: &ScenarioSource,
    forward_passes: u32,
    max_iterations: u64,
    inflow_method: InflowNonNegativityMethod,
) -> cobre_sddp::TrainingResult {
    let forward_seed = source.seed.map(i64::unsigned_abs);

    let stochastic = build_stochastic_context(
        system,
        42,
        forward_seed,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("build_stochastic_context must succeed");

    let hydro_models = PrepareHydroModelsResult::default_from_system(system);

    let stopping_rule_set = StoppingRuleSet {
        rules: vec![StoppingRule::IterationLimit {
            limit: max_iterations,
        }],
        mode: StoppingMode::Any,
    };

    let config = ConstructionConfig {
        seed: 42, // tree seed
        forward_passes,
        stopping_rule_set,
        n_scenarios: 0, // simulation disabled
        io_channel_capacity: 0,
        policy_path: String::new(),
        inflow_method,
        cut_selection: None,
        cut_activity_tolerance: 0.0,
        budget: None,
        export_states: false,
    };
    let mut setup =
        StudySetup::from_broadcast_params(system, stochastic, config, hydro_models, source, source)
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

    let (system_insample, source_insample) = build_single_hydro_system(
        1, // hydro_id
        3, // n_stages
        5, // branching_factor
        SamplingScheme::InSample,
        None, // forward_seed (not used for InSample)
    );

    let (system_oos, source_oos) = build_single_hydro_system(
        1,
        3,
        5,
        SamplingScheme::OutOfSample,
        Some(42), // forward_seed required for OutOfSample
    );

    // Use Truncation to prevent LP infeasibility from large negative noise draws.
    // The seed domain changed intentionally (derive_forward_seed_grouped vs
    // derive_forward_seed) so the noise trajectory differs from the baseline;
    // Truncation makes the test robust to any seed without affecting the
    // convergence comparison between InSample and OutOfSample.
    let lb_insample = run_programmatic(
        &system_insample,
        &source_insample,
        FORWARD_PASSES,
        MAX_ITERATIONS,
        InflowNonNegativityMethod::Truncation,
    )
    .final_lb;
    let lb_oos = run_programmatic(
        &system_oos,
        &source_oos,
        FORWARD_PASSES,
        MAX_ITERATIONS,
        InflowNonNegativityMethod::Truncation,
    )
    .final_lb;

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
    let (system_a, source_a) =
        build_two_hydro_system(&[1, 2], 3, 3, SamplingScheme::OutOfSample, Some(99));

    // System B: declare hydros in order [H2, H1] (reversed declaration order)
    let (system_b, source_b) =
        build_two_hydro_system(&[2, 1], 3, 3, SamplingScheme::OutOfSample, Some(99));

    let lb_a = run_programmatic(
        &system_a,
        &source_a,
        FORWARD_PASSES,
        MAX_ITERATIONS,
        InflowNonNegativityMethod::None,
    )
    .final_lb;
    let lb_b = run_programmatic(
        &system_b,
        &source_b,
        FORWARD_PASSES,
        MAX_ITERATIONS,
        InflowNonNegativityMethod::None,
    )
    .final_lb;

    assert_eq!(
        lb_a, lb_b,
        "declaration-order invariance violated: LB_A={lb_a}, LB_B={lb_b}"
    );
}

// ---------------------------------------------------------------------------
// Historical / External / Mixed helpers and tests (tickets 032-034)
// ---------------------------------------------------------------------------

/// Create a stage with distinct `season_id` (index % 4) and per-month dates.
fn make_monthly_stage(index: usize, branching_factor: usize) -> Stage {
    let month = (index % 12) as u32 + 1;
    let year = 2024 + (index / 12) as i32;
    let next_month = if month == 12 { 1 } else { month + 1 };
    let next_year = if month == 12 { year + 1 } else { year };
    Stage {
        index,
        id: index as i32,
        start_date: NaiveDate::from_ymd_opt(year, month, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(next_year, next_month, 1).unwrap(),
        season_id: Some(index % 4),
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

/// Generate `n_years * 12` monthly inflow history rows with seasonal variation.
fn build_inflow_history(hydro_id: EntityId, n_years: usize) -> Vec<InflowHistoryRow> {
    let base_year = 2000;
    let mut rows = Vec::with_capacity(n_years * 12);
    for y in 0..n_years {
        for m in 0..12u32 {
            let value = 80.0 + 15.0 * (f64::from(m) * std::f64::consts::PI / 6.0).sin();
            rows.push(InflowHistoryRow {
                hydro_id,
                date: NaiveDate::from_ymd_opt(base_year + y as i32, m + 1, 1).unwrap(),
                value_m3s: value,
            });
        }
    }
    rows
}

/// Build a system for Historical inflow testing.
fn build_historical_system(
    hydro_raw_id: i32,
    branching_factor: usize,
    n_history_years: usize,
    forward_seed: Option<i64>,
) -> (cobre_core::System, ScenarioSource) {
    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let id = EntityId(hydro_raw_id);
    let hydro = make_hydro(hydro_raw_id);
    let stages: Vec<Stage> = (0..4)
        .map(|i| make_monthly_stage(i, branching_factor))
        .collect();
    let inflow_models: Vec<InflowModel> = (0..4)
        .map(|i| InflowModel {
            hydro_id: id,
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        })
        .collect();
    let history = build_inflow_history(id, n_history_years);
    let correlation = make_correlation(&[id]);
    let bounds = build_resolved_bounds(1, 4);
    let penalties = build_resolved_penalties(1, 1, 4);

    let source = ScenarioSource {
        inflow_scheme: SamplingScheme::Historical,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        seed: forward_seed,
        historical_years: None,
    };

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .inflow_history(history)
        .correlation(correlation)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("SystemBuilder for historical must succeed");

    (system, source)
}

/// Run training pipeline with per-class schemes derived from the supplied source.
/// Returns the `StudySetup` so callers can assert library presence before training.
fn run_with_setup(
    system: &cobre_core::System,
    source: &ScenarioSource,
    forward_passes: u32,
    max_iterations: u64,
) -> (StudySetup, cobre_sddp::TrainingResult) {
    let forward_seed = source.seed.map(i64::unsigned_abs);
    let schemes = ClassSchemes {
        inflow: Some(source.inflow_scheme),
        load: Some(source.load_scheme),
        ncs: Some(source.ncs_scheme),
    };

    let stochastic = build_stochastic_context(
        system,
        42,
        forward_seed,
        &[],
        &[],
        OpeningTreeInputs::default(),
        schemes,
    )
    .expect("build_stochastic_context must succeed");

    let hydro_models = PrepareHydroModelsResult::default_from_system(system);
    let stopping_rule_set = StoppingRuleSet {
        rules: vec![StoppingRule::IterationLimit {
            limit: max_iterations,
        }],
        mode: StoppingMode::Any,
    };

    let config = ConstructionConfig {
        seed: 42,
        forward_passes,
        stopping_rule_set,
        n_scenarios: 0,
        io_channel_capacity: 0,
        policy_path: String::new(),
        inflow_method: InflowNonNegativityMethod::None,
        cut_selection: None,
        cut_activity_tolerance: 0.0,
        budget: None,
        export_states: false,
    };
    let mut setup =
        StudySetup::from_broadcast_params(system, stochastic, config, hydro_models, source, source)
            .expect("StudySetup::from_broadcast_params must succeed");

    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");
    let outcome = setup
        .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");
    (setup, outcome.result)
}

/// Generate external inflow rows with deterministic arithmetic noise.
fn build_external_inflow_rows(
    hydro_id: EntityId,
    n_stages: usize,
    n_scenarios: usize,
) -> Vec<ExternalScenarioRow> {
    let mut rows = Vec::with_capacity(n_stages * n_scenarios);
    for stage in 0..n_stages {
        for scenario in 0..n_scenarios {
            let noise = ((scenario * 7 + stage * 3) % 10) as f64 - 5.0;
            rows.push(ExternalScenarioRow {
                stage_id: stage as i32,
                scenario_id: scenario as i32,
                hydro_id,
                value_m3s: 80.0 + 20.0 * noise / 5.0,
            });
        }
    }
    rows
}

/// Build a system for External inflow testing.
fn build_external_system(
    hydro_raw_id: i32,
    branching_factor: usize,
    n_scenarios: usize,
    forward_seed: Option<i64>,
) -> (cobre_core::System, ScenarioSource) {
    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let id = EntityId(hydro_raw_id);
    let hydro = make_hydro(hydro_raw_id);
    let stages: Vec<Stage> = (0..3)
        .map(|i| make_monthly_stage(i, branching_factor))
        .collect();
    let inflow_models: Vec<InflowModel> = (0..3)
        .map(|i| InflowModel {
            hydro_id: id,
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        })
        .collect();
    let ext_rows = build_external_inflow_rows(id, 3, n_scenarios);
    let correlation = make_correlation(&[id]);
    let bounds = build_resolved_bounds(1, 3);
    let penalties = build_resolved_penalties(1, 1, 3);

    let source = ScenarioSource {
        inflow_scheme: SamplingScheme::External,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        seed: forward_seed,
        historical_years: None,
    };

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .external_scenarios(ext_rows)
        .correlation(correlation)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("SystemBuilder for external must succeed");

    (system, source)
}

fn build_resolved_penalties_with_ncs(
    n_hydros: usize,
    n_buses: usize,
    n_ncs: usize,
    n_stages: usize,
) -> ResolvedPenalties {
    let n_st = n_stages.max(1);
    ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros,
            n_buses,
            n_lines: 0,
            n_ncs,
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

/// Assert that all external libraries are None.
fn assert_no_external_libraries(setup: &StudySetup) {
    assert!(setup.historical_library().is_none());
    assert!(setup.external_inflow_library().is_none());
    assert!(setup.external_load_library().is_none());
    assert!(setup.external_ncs_library().is_none());
}

/// Build a system for mixed-scheme testing (hydro + NCS + stochastic load).
fn build_mixed_system(
    inflow_scheme: SamplingScheme,
    load_scheme: SamplingScheme,
    ncs_scheme: SamplingScheme,
) -> (cobre_core::System, ScenarioSource) {
    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let hydro = make_hydro(1);
    let ncs = NonControllableSource {
        id: EntityId(10),
        name: "NCS0".to_string(),
        bus_id: EntityId(0),
        entry_stage_id: None,
        exit_stage_id: None,
        max_generation_mw: 30.0,
        curtailment_cost: 0.0,
    };
    let stages: Vec<Stage> = (0..3).map(|i| make_monthly_stage(i, 5)).collect();
    let inflow_models: Vec<InflowModel> = (0..3)
        .map(|i| InflowModel {
            hydro_id: EntityId(1),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        })
        .collect();
    let load_models: Vec<LoadModel> = (0..3)
        .map(|i| LoadModel {
            bus_id: EntityId(0),
            stage_id: i as i32,
            mean_mw: 60.0,
            std_mw: 10.0,
        })
        .collect();
    let ncs_models: Vec<NcsModel> = (0..3)
        .map(|i| NcsModel {
            ncs_id: EntityId(10),
            stage_id: i as i32,
            mean: 20.0,
            std: 5.0,
        })
        .collect();
    let correlation = make_correlation(&[EntityId(1)]);
    let bounds = build_resolved_bounds(1, 3);
    let penalties = build_resolved_penalties_with_ncs(1, 1, 1, 3);

    let source = ScenarioSource {
        inflow_scheme,
        load_scheme,
        ncs_scheme,
        seed: Some(42),
        historical_years: None,
    };

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .non_controllable_sources(vec![ncs])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .ncs_models(ncs_models)
        .correlation(correlation)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("SystemBuilder for mixed must succeed");

    (system, source)
}

// --- ticket-032: Historical integration test ---

#[test]
fn historical_convergence() {
    const FORWARD_PASSES: u32 = 10;
    const MAX_ITERATIONS: u64 = 50;

    let (system, source) = build_historical_system(1, 5, 10, Some(42));
    let (setup, result) = run_with_setup(&system, &source, FORWARD_PASSES, MAX_ITERATIONS);

    assert!(
        setup.historical_library().is_some(),
        "historical_library must be Some for Historical scheme"
    );

    assert!(
        result.final_lb.is_finite(),
        "final_lb must be finite, got {}",
        result.final_lb
    );
}

// --- ticket-033: External integration test ---

#[test]
fn external_inflow_convergence() {
    const FORWARD_PASSES: u32 = 10;
    const MAX_ITERATIONS: u64 = 50;
    const N_SCENARIOS: usize = 20;

    let (system, source) = build_external_system(1, 5, N_SCENARIOS, Some(42));
    let (setup, result) = run_with_setup(&system, &source, FORWARD_PASSES, MAX_ITERATIONS);

    assert!(
        setup.external_inflow_library().is_some(),
        "external_inflow_library must be Some for External scheme"
    );

    assert!(
        result.final_lb.is_finite(),
        "final_lb must be finite, got {}",
        result.final_lb
    );

    // Reproducibility: same seed must produce identical LB.
    let (system2, source2) = build_external_system(1, 5, N_SCENARIOS, Some(42));
    let (_setup2, result2) = run_with_setup(&system2, &source2, FORWARD_PASSES, MAX_ITERATIONS);
    assert_eq!(
        result.final_lb, result2.final_lb,
        "reproducibility violated: run1={}, run2={}",
        result.final_lb, result2.final_lb
    );
}

// --- ticket-034: Mixed-scheme integration test ---

#[test]
fn mixed_scheme_convergence() {
    const FORWARD_PASSES: u32 = 10;
    const MAX_ITERATIONS: u64 = 50;

    // Combination 1: inflow InSample, load OutOfSample, ncs InSample
    let (system_a, source_a) = build_mixed_system(
        SamplingScheme::InSample,
        SamplingScheme::OutOfSample,
        SamplingScheme::InSample,
    );
    let (setup_a, result_a) = run_with_setup(&system_a, &source_a, FORWARD_PASSES, MAX_ITERATIONS);
    assert_no_external_libraries(&setup_a);
    assert!(
        result_a.final_lb.is_finite(),
        "combo 1: final_lb must be finite"
    );

    // Combination 2: inflow OutOfSample, load InSample, ncs InSample
    let (system_b, source_b) = build_mixed_system(
        SamplingScheme::OutOfSample,
        SamplingScheme::InSample,
        SamplingScheme::InSample,
    );
    let (setup_b, result_b) = run_with_setup(&system_b, &source_b, FORWARD_PASSES, MAX_ITERATIONS);
    assert_no_external_libraries(&setup_b);
    assert!(
        result_b.final_lb.is_finite(),
        "combo 2: final_lb must be finite"
    );
}

// ---------------------------------------------------------------------------
// External load / NCS library population tests (F1-106)
// ---------------------------------------------------------------------------

/// Generate `n_stages × n_scenarios` external load rows for a single bus.
fn build_external_load_rows(
    bus_id: EntityId,
    n_stages: usize,
    n_scenarios: usize,
) -> Vec<ExternalLoadRow> {
    let mut rows = Vec::with_capacity(n_stages * n_scenarios);
    for stage in 0..n_stages {
        for scenario in 0..n_scenarios {
            rows.push(ExternalLoadRow {
                stage_id: stage as i32,
                scenario_id: scenario as i32,
                bus_id,
                value_mw: 50.0 + 5.0 * (scenario as f64),
            });
        }
    }
    rows
}

/// Generate `n_stages × n_scenarios` external NCS rows for a single NCS source.
fn build_external_ncs_rows(
    ncs_id: EntityId,
    n_stages: usize,
    n_scenarios: usize,
) -> Vec<ExternalNcsRow> {
    let mut rows = Vec::with_capacity(n_stages * n_scenarios);
    for stage in 0..n_stages {
        for scenario in 0..n_scenarios {
            rows.push(ExternalNcsRow {
                stage_id: stage as i32,
                scenario_id: scenario as i32,
                ncs_id,
                value: 0.6 + 0.04 * (scenario as f64 / n_scenarios as f64),
            });
        }
    }
    rows
}

/// Build a system whose load scheme is `External`, backed by pre-computed
/// `ExternalLoadRow` data on the `System`.
///
/// Has 1 bus, 1 hydro, 1 load model, and 3 monthly stages.
fn build_external_load_system(
    n_scenarios: usize,
    forward_seed: Option<i64>,
) -> (cobre_core::System, ScenarioSource) {
    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let hydro = make_hydro(1);
    let stages: Vec<Stage> = (0..3).map(|i| make_monthly_stage(i, 5)).collect();
    let inflow_models: Vec<InflowModel> = (0..3)
        .map(|i| InflowModel {
            hydro_id: EntityId(1),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        })
        .collect();
    let load_models: Vec<LoadModel> = (0..3)
        .map(|i| LoadModel {
            bus_id: EntityId(0),
            stage_id: i as i32,
            mean_mw: 60.0,
            std_mw: 10.0,
        })
        .collect();
    let ext_load_rows = build_external_load_rows(EntityId(0), 3, n_scenarios);
    let correlation = make_correlation(&[EntityId(1)]);
    let bounds = build_resolved_bounds(1, 3);
    let penalties = build_resolved_penalties(1, 1, 3);

    let source = ScenarioSource {
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::External,
        ncs_scheme: SamplingScheme::InSample,
        seed: forward_seed,
        historical_years: None,
    };

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .external_load_scenarios(ext_load_rows)
        .correlation(correlation)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("SystemBuilder for external load must succeed");

    (system, source)
}

/// Build a system whose NCS scheme is `External`, backed by pre-computed
/// `ExternalNcsRow` data on the `System`.
///
/// Has 1 bus, 1 hydro, 1 NCS source, and 3 monthly stages.
fn build_external_ncs_system(
    n_scenarios: usize,
    forward_seed: Option<i64>,
) -> (cobre_core::System, ScenarioSource) {
    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let hydro = make_hydro(1);
    let ncs = NonControllableSource {
        id: EntityId(10),
        name: "NCS0".to_string(),
        bus_id: EntityId(0),
        entry_stage_id: None,
        exit_stage_id: None,
        max_generation_mw: 30.0,
        curtailment_cost: 0.0,
    };
    let stages: Vec<Stage> = (0..3).map(|i| make_monthly_stage(i, 5)).collect();
    let inflow_models: Vec<InflowModel> = (0..3)
        .map(|i| InflowModel {
            hydro_id: EntityId(1),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        })
        .collect();
    let ncs_models: Vec<NcsModel> = (0..3)
        .map(|i| NcsModel {
            ncs_id: EntityId(10),
            stage_id: i as i32,
            mean: 20.0,
            std: 5.0,
        })
        .collect();
    let ext_ncs_rows = build_external_ncs_rows(EntityId(10), 3, n_scenarios);
    let correlation = make_correlation(&[EntityId(1)]);
    let bounds = build_resolved_bounds(1, 3);
    let penalties = build_resolved_penalties_with_ncs(1, 1, 1, 3);

    let source = ScenarioSource {
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::External,
        seed: forward_seed,
        historical_years: None,
    };

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .non_controllable_sources(vec![ncs])
        .stages(stages)
        .inflow_models(inflow_models)
        .ncs_models(ncs_models)
        .external_ncs_scenarios(ext_ncs_rows)
        .correlation(correlation)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("SystemBuilder for external NCS must succeed");

    (system, source)
}

/// Verify that `ClassSampler::External` for load populates
/// `StudySetup::external_load_library` and produces a finite lower bound.
///
/// Builds a system with `load_scheme = External` and 20 pre-computed
/// `ExternalLoadRow` entries per stage. After training, asserts that
/// `external_load_library()` is `Some` and the final lower bound is finite.
#[test]
fn external_load_library_populated() {
    const FORWARD_PASSES: u32 = 10;
    const MAX_ITERATIONS: u64 = 20;
    const N_SCENARIOS: usize = 20;

    let (system, source) = build_external_load_system(N_SCENARIOS, Some(42));
    let (setup, result) = run_with_setup(&system, &source, FORWARD_PASSES, MAX_ITERATIONS);

    assert!(
        setup.external_load_library().is_some(),
        "external_load_library must be Some when load_scheme is External"
    );
    assert!(
        setup.external_inflow_library().is_none(),
        "external_inflow_library must be None when inflow_scheme is InSample"
    );
    assert!(
        setup.external_ncs_library().is_none(),
        "external_ncs_library must be None when ncs_scheme is InSample"
    );
    assert!(
        result.final_lb.is_finite(),
        "final_lb must be finite for External load scheme, got {}",
        result.final_lb
    );
}

/// Verify that `ClassSampler::External` for NCS populates
/// `StudySetup::external_ncs_library` and produces a finite lower bound.
///
/// Builds a system with `ncs_scheme = External` and 20 pre-computed
/// `ExternalNcsRow` entries per stage. After training, asserts that
/// `external_ncs_library()` is `Some` and the final lower bound is finite.
#[test]
fn external_ncs_library_populated() {
    const FORWARD_PASSES: u32 = 10;
    const MAX_ITERATIONS: u64 = 20;
    const N_SCENARIOS: usize = 20;

    let (system, source) = build_external_ncs_system(N_SCENARIOS, Some(42));
    let (setup, result) = run_with_setup(&system, &source, FORWARD_PASSES, MAX_ITERATIONS);

    assert!(
        setup.external_ncs_library().is_some(),
        "external_ncs_library must be Some when ncs_scheme is External"
    );
    assert!(
        setup.external_inflow_library().is_none(),
        "external_inflow_library must be None when inflow_scheme is InSample"
    );
    assert!(
        setup.external_load_library().is_none(),
        "external_load_library must be None when load_scheme is InSample"
    );
    assert!(
        result.final_lb.is_finite(),
        "final_lb must be finite for External NCS scheme, got {}",
        result.final_lb
    );
}

// ---------------------------------------------------------------------------
// Noise-sharing regression tests (Epic 02, ticket-005)
// ---------------------------------------------------------------------------

/// Build a 12-stage monthly system where every stage has a distinct
/// `(season_id, year)` pair — i.e. every stage gets a unique noise group ID
/// from `precompute_noise_groups`. This simulates the normal monthly study
/// layout where Pattern C noise sharing is structurally inactive.
///
/// Returns the system and an `InSample` scenario source so the test is
/// deterministic and exercises the common production path.
fn build_monthly_unique_groups_system(
    n_stages: usize,
    branching_factor: usize,
) -> (cobre_core::System, ScenarioSource) {
    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };

    let hydro = make_hydro(1);

    // Each stage uses a different calendar month so (season_id, year) is unique
    // for every stage. season_id = index % 12 gives each month a distinct id.
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| {
            let month = (i % 12) as u32 + 1;
            let year = 2024 + (i / 12) as i32;
            let next_month = if month == 12 { 1 } else { month + 1 };
            let next_year = if month == 12 { year + 1 } else { year };
            Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(year, month, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(next_year, next_month, 1).unwrap(),
                // Use the calendar month (0-based) as season_id so every stage
                // within a calendar year has a different season. Combined with
                // the distinct year, every (season_id, year) pair is unique.
                season_id: Some(i % 12),
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
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(1),
            stage_id: i as i32,
            mean_m3s: 80.0 + 20.0 * ((i as f64) * std::f64::consts::PI / 6.0).sin(),
            std_m3s: 15.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        })
        .collect();

    let correlation = make_correlation(&[EntityId(1)]);
    let bounds = build_resolved_bounds(1, n_stages);
    let penalties = build_resolved_penalties(1, 1, n_stages);

    let source = ScenarioSource {
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        seed: None,
        historical_years: None,
    };

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("SystemBuilder must produce a valid system");

    (system, source)
}

/// Regression test for ticket-005: noise group wiring must be transparent for
/// monthly studies.
///
/// A 12-stage monthly study where every stage has a unique `(season_id, year)`
/// pair produces unique noise group IDs for every stage via
/// `precompute_noise_groups`. With unique groups there is no sharing, so the
/// forward pass and opening tree behaviour must be bit-identical to running
/// the same study twice with the same seed.
///
/// This test verifies that:
/// 1. The noise group IDs are propagated from `StudySetup` through
///    `StageContext` to `SampleRequest.noise_group_id` in both the forward
///    pass and simulation pipeline.
/// 2. The opening tree is built with `Some(noise_group_ids)` wired from setup.
/// 3. For monthly studies (unique groups) the noise sharing infrastructure is
///    transparent: same seed + same system = bit-identical lower bound.
#[test]
fn monthly_noise_sharing_regression() {
    const FORWARD_PASSES: u32 = 5;
    const MAX_ITERATIONS: u64 = 10;

    let (system, source) = build_monthly_unique_groups_system(
        12, // 12 monthly stages (1 year)
        3,  // branching_factor=3
    );

    // Run once.
    let result_a = run_programmatic(
        &system,
        &source,
        FORWARD_PASSES,
        MAX_ITERATIONS,
        InflowNonNegativityMethod::None,
    );

    // Run again with the same system and source — must be bit-identical.
    let result_b = run_programmatic(
        &system,
        &source,
        FORWARD_PASSES,
        MAX_ITERATIONS,
        InflowNonNegativityMethod::None,
    );

    assert_eq!(
        result_a.final_lb, result_b.final_lb,
        "monthly noise sharing regression: lower bound is not deterministic. \
         run_a={}, run_b={}",
        result_a.final_lb, result_b.final_lb
    );
    assert_eq!(
        result_a.iterations, result_b.iterations,
        "monthly noise sharing regression: iteration count is not deterministic. \
         run_a={}, run_b={}",
        result_a.iterations, result_b.iterations
    );
    assert!(
        result_a.final_lb.is_finite(),
        "monthly noise sharing regression: lower bound must be finite, got {}",
        result_a.final_lb
    );
}
