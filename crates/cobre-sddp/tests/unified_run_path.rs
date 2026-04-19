//! Integration tests for the unified `run_stage_solve` entry point.
//!
//! guards: unified-path
//!
//! Verifies that the forward and backward passes delegate correctly to
//! `run_stage_solve` after tickets 003 and 004 rewired `forward.rs` and
//! `backward.rs`. The primary assertion for each pass is that the LP solve
//! counter increments by exactly the expected number of solves, proving each
//! stage/scenario pair issues exactly one solve through the unified path.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
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
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, SamplingScheme,
    },
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_solver::highs::HighsSolver;
use cobre_solver::{RowBatch, SolverInterface, SolverStatistics, StageTemplate};
use cobre_stochastic::{
    ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context,
};

use cobre_sddp::basis_reconstruct::{HIGHS_BASIS_STATUS_BASIC as B, ReconstructionStats};
use cobre_sddp::cut::pool::CutPool;
use cobre_sddp::stage_solve::{Phase, StageInputs, StageOutcome, run_stage_solve};
use cobre_sddp::workspace::CapturedBasis;
use cobre_sddp::{
    BackwardPassSpec, BakedTemplates, BasisStore, CanonicalStateStrategy, CutManagementConfig,
    CutSyncBuffers, EntityCounts, EventConfig, ExchangeBuffers, ForwardPassBatch,
    FutureCostFunction, HorizonMode, InflowNonNegativityMethod, LoopConfig, PatchBuffer,
    RiskMeasure, SimulationConfig, SimulationOutputSpec, SolverWorkspace, StageContext,
    StageIndexer, StoppingMode, StoppingRule, StoppingRuleSet, TrainingConfig, TrainingContext,
    TrajectoryRecord, WorkspaceSizing, run_backward_pass, run_forward_pass, simulate, train,
};

// ---------------------------------------------------------------------------
// Stub communicator
// ---------------------------------------------------------------------------

#[allow(dead_code)]
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Minimal LP template for N=1 hydro, L=0 PAR, compatible with `HighsSolver`.
///
/// Same template used in `canonical_state_strategy.rs` — 3 rows, 4 cols,
/// with a water-balance structure that `HighsSolver` accepts as feasible with
/// all-zero RHS and zero noise scale.
///
/// Row layout:
/// - Row 0: storage-fixing
/// - Row 1: z-inflow definition
/// - Row 2: water-balance
///
/// `base_rows[s] = 2` is valid because `num_rows = 3 > 2`.
fn minimal_template() -> StageTemplate {
    StageTemplate {
        num_cols: 4,
        num_rows: 3,
        num_nz: 5,
        col_starts: vec![0_i32, 1, 3, 5, 5],
        row_indices: vec![2_i32, 1, 2, 0, 2],
        values: vec![1.0_f64, 1.0, -1.0, 1.0, -1.0],
        col_lower: vec![0.0, 0.0, 0.0, 0.0],
        col_upper: vec![100.0, 200.0, 100.0, f64::INFINITY],
        objective: vec![0.0, 0.0, 0.0, 1.0],
        row_lower: vec![0.0, 0.0, 0.0],
        row_upper: vec![0.0, 0.0, 0.0],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 1,
        n_hydro: 1,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

/// Build a minimal `StochasticContext` for `n_stages` stages with 1 hydro,
/// zero-mean inflow, and SAA sampling. Mirrors `make_stochastic_context` in
/// `canonical_state_strategy.rs`.
#[allow(clippy::too_many_lines)]
fn make_stochastic_context(n_stages: usize) -> StochasticContext {
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
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
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
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    };

    let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();

    // Zero-mean, zero-std inflow keeps the LP feasible with all-zero state.
    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(1),
            stage_id: i as i32,
            mean_m3s: 0.0,
            std_m3s: 0.0,
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
        method: "spectral".to_string(),
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

fn empty_row_batch() -> RowBatch {
    RowBatch {
        num_rows: 0,
        row_starts: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

/// After ticket-003, the forward pass delegates every stage/scenario LP solve
/// through `run_stage_solve`. Verify that a 2-stage, 2-scenario, 1-iteration
/// forward pass:
///   1. Completes without error.
///   2. Issues exactly `n_stages * n_scenarios` LP solves — one per
///      (stage, scenario) pair, matching the pre-rewire counter progression.
///
/// Uses `HighsSolver` which properly increments `statistics().solve_count`.
///
/// guards: unified-path
#[test]
fn forward_uses_stage_solve() {
    let n_stages = 2_usize;
    let n_scenarios = 2_usize;

    let indexer = StageIndexer::new(1, 0); // N=1, L=0 → n_state=1
    let templates: Vec<StageTemplate> = vec![minimal_template(); n_stages];
    // base_rows[s] = 2: water-balance row is at index 2 in the template.
    let base_rows = vec![2_usize; n_stages];
    // noise_scale=0 keeps z_inflow_rhs=0 → LP always feasible with zero-state.
    let noise_scale = vec![0.0_f64; n_stages];
    let initial_state = vec![0.0_f64; indexer.n_state];
    let stochastic = make_stochastic_context(n_stages);

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
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
        })
        .collect();

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let fcf = FutureCostFunction::new(n_stages, indexer.n_state, 2, 50, &vec![0; n_stages]);

    let stage_ctx = StageContext {
        templates: &templates,
        base_rows: &base_rows,
        noise_scale: &noise_scale,
        // n_hydros=1 enables the noise-transform path that patches the LP RHS.
        // Combined with noise_scale=0, the patch is a no-op but the path is
        // exercised (same convention as canonical_state_strategy.rs).
        n_hydros: 1,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &vec![1_usize; n_stages],
        ncs_max_gen: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
    };

    let training_ctx = TrainingContext {
        horizon: &horizon,
        indexer: &indexer,
        inflow_method: &InflowNonNegativityMethod::None,
        stochastic: &stochastic,
        initial_state: &initial_state,
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        stages: &stages,
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
        recent_accum_seed: &[],
        recent_weight_seed: 0.0,
    };

    // Use HighsSolver so solve_count is properly tracked.
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");
    solver.load_model(&templates[0]);

    let sizing = WorkspaceSizing {
        hydro_count: 1,
        max_par_order: 0,
        n_load_buses: 0,
        max_blocks: 1,
        downstream_par_order: 0,
        max_openings: 0,
        initial_pool_capacity: 0,
        n_state: indexer.n_state,
    };
    let workspace = SolverWorkspace::new(
        solver,
        PatchBuffer::new(indexer.n_state, 0, 1, 0),
        indexer.n_state,
        sizing,
    );
    let mut workspaces = vec![workspace];

    let mut basis_store = BasisStore::new(n_scenarios, n_stages);
    let mut cut_batches: Vec<RowBatch> = (0..n_stages).map(|_| empty_row_batch()).collect();
    let mut records: Vec<cobre_sddp::TrajectoryRecord> = (0..n_scenarios * n_stages)
        .map(|_| cobre_sddp::TrajectoryRecord {
            primal: Vec::new(),
            dual: Vec::new(),
            stage_cost: 0.0,
            state: Vec::new(),
        })
        .collect();

    let not_baked = BakedTemplates {
        templates: &[],
        ready: false,
    };
    let batch = ForwardPassBatch {
        local_forward_passes: n_scenarios,
        total_forward_passes: n_scenarios,
        iteration: 1,
        fwd_offset: 0,
    };

    let solve_count_before = workspaces[0].solver.statistics().solve_count;

    let result = run_forward_pass(
        &mut workspaces,
        &mut basis_store,
        &stage_ctx,
        &not_baked,
        &fcf,
        &mut cut_batches,
        &training_ctx,
        &batch,
        &mut records,
    );

    // 1. Pass must complete without error.
    let _ = result.expect("forward pass must complete Ok");

    // 2. solve_count must increase by exactly n_stages * n_scenarios.
    let solve_count_after = workspaces[0].solver.statistics().solve_count;
    let delta = solve_count_after - solve_count_before;
    assert_eq!(
        delta,
        (n_stages * n_scenarios) as u64,
        "expected exactly {} LP solves (n_stages={} × n_scenarios={}), got {}",
        n_stages * n_scenarios,
        n_stages,
        n_scenarios,
        delta,
    );
}

/// After ticket-004, the backward pass delegates every stage LP solve through
/// `run_stage_solve`. Verify that a 2-stage, 2-scenario, 1-iteration forward +
/// backward pass:
///   1. Backward completes without error.
///   2. Backward produces > 0 cuts (at least one cut per trial point at stage 0).
///   3. Total LP solve count equals forward + backward solves:
///      forward: `n_stages * n_scenarios`
///      backward: `(n_stages - 1) * n_scenarios * n_openings_per_stage`
///      where `n_openings_per_stage = 1` (SAA with `branching_factor=1`).
///
/// guards: unified-path
#[test]
fn backward_uses_stage_solve() {
    let n_stages = 2_usize;
    let n_scenarios = 2_usize;

    let indexer = StageIndexer::new(1, 0); // N=1, L=0 → n_state=1
    let templates: Vec<StageTemplate> = vec![minimal_template(); n_stages];
    let base_rows = vec![2_usize; n_stages];
    let noise_scale = vec![0.0_f64; n_stages];
    let initial_state = vec![0.0_f64; indexer.n_state];
    let stochastic = make_stochastic_context(n_stages);

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
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
        })
        .collect();

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let mut fcf = FutureCostFunction::new(n_stages, indexer.n_state, 2, 50, &vec![0; n_stages]);

    let stage_ctx = StageContext {
        templates: &templates,
        base_rows: &base_rows,
        noise_scale: &noise_scale,
        n_hydros: 1,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &vec![1_usize; n_stages],
        ncs_max_gen: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
    };

    let training_ctx = TrainingContext {
        horizon: &horizon,
        indexer: &indexer,
        inflow_method: &InflowNonNegativityMethod::None,
        stochastic: &stochastic,
        initial_state: &initial_state,
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        stages: &stages,
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
        recent_accum_seed: &[],
        recent_weight_seed: 0.0,
    };

    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");
    solver.load_model(&templates[0]);

    let sizing = WorkspaceSizing {
        hydro_count: 1,
        max_par_order: 0,
        n_load_buses: 0,
        max_blocks: 1,
        downstream_par_order: 0,
        max_openings: 1,
        initial_pool_capacity: 0,
        n_state: indexer.n_state,
    };
    let workspace = SolverWorkspace::new(
        solver,
        PatchBuffer::new(indexer.n_state, 0, 1, 0),
        indexer.n_state,
        sizing,
    );
    let mut workspaces = vec![workspace];

    let mut basis_store = BasisStore::new(n_scenarios, n_stages);
    let mut cut_batches: Vec<RowBatch> = (0..n_stages).map(|_| empty_row_batch()).collect();
    let mut records: Vec<TrajectoryRecord> = (0..n_scenarios * n_stages)
        .map(|_| TrajectoryRecord {
            primal: Vec::new(),
            dual: Vec::new(),
            stage_cost: 0.0,
            state: Vec::new(),
        })
        .collect();

    let not_baked = BakedTemplates {
        templates: &[],
        ready: false,
    };
    let batch = ForwardPassBatch {
        local_forward_passes: n_scenarios,
        total_forward_passes: n_scenarios,
        iteration: 1,
        fwd_offset: 0,
    };

    // --- Forward pass ---
    let fwd_solves_before = workspaces[0].solver.statistics().solve_count;

    let _ = run_forward_pass(
        &mut workspaces,
        &mut basis_store,
        &stage_ctx,
        &not_baked,
        &fcf,
        &mut cut_batches,
        &training_ctx,
        &batch,
        &mut records,
    )
    .expect("forward pass must complete Ok");

    let fwd_solves_after = workspaces[0].solver.statistics().solve_count;
    let fwd_delta = fwd_solves_after - fwd_solves_before;
    assert_eq!(
        fwd_delta,
        (n_stages * n_scenarios) as u64,
        "forward: expected {} solves, got {}",
        n_stages * n_scenarios,
        fwd_delta,
    );

    // --- Backward pass ---
    // Pre-populate exchange with states from the forward records.
    let mut exchange = ExchangeBuffers::new(indexer.n_state, n_scenarios, 1);
    // Stage 0 backward solves stage 1 (successor), so exchange must hold the
    // state entering stage 1, i.e., records[m * n_stages + 0].state for each m.
    // The BackwardPassSpec.records path calls exchange.exchange(records, t, ...)
    // once per stage, so we pass the records directly instead of pre-populating.
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];
    let mut csb = CutSyncBuffers::new(indexer.n_state, 64, 1);

    let comm = StubComm;
    let bwd_solves_before = workspaces[0].solver.statistics().solve_count;

    let bwd_result = run_backward_pass(
        &mut workspaces,
        &basis_store,
        &stage_ctx,
        &not_baked,
        &mut fcf,
        &mut cut_batches,
        &training_ctx,
        &mut BackwardPassSpec {
            exchange: &mut exchange,
            records: &records,
            iteration: 1,
            local_work: n_scenarios,
            fwd_offset: 0,
            risk_measures: &risk_measures,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            probabilities_buf: &mut Vec::new(),
            successor_active_slots_buf: &mut Vec::new(),
            visited_archive: None,
            metadata_sync_buf: &mut Vec::new(),
            global_increments_buf: &mut Vec::new(),
            real_states_buf: &mut Vec::new(),
            canonical_state_strategy: CanonicalStateStrategy::default(),
        },
        &comm,
    )
    .expect("backward pass must complete Ok");

    let bwd_solves_after = workspaces[0].solver.statistics().solve_count;
    let bwd_delta = bwd_solves_after - bwd_solves_before;

    // 1. Backward must produce at least one cut (one per trial point at stage 0).
    assert!(
        bwd_result.cuts_generated > 0,
        "backward pass must generate at least one cut, got 0"
    );

    // 2. Backward solve count: (n_stages-1) backward stages * n_scenarios
    //    trial points * 1 opening (branching_factor=1).
    let expected_bwd_solves = ((n_stages - 1) * n_scenarios) as u64;
    assert_eq!(
        bwd_delta,
        expected_bwd_solves,
        "backward: expected {expected_bwd_solves} LP solves \
         ((n_stages-1)={} x n_scenarios={} x n_openings=1), got {bwd_delta}",
        n_stages - 1,
        n_scenarios,
    );

    // 3. Total solves = forward + backward.
    let total_delta = fwd_delta + bwd_delta;
    let expected_total = fwd_delta + expected_bwd_solves;
    assert_eq!(
        total_delta, expected_total,
        "total LP solves must equal forward ({fwd_delta}) + backward ({expected_bwd_solves}), \
         got {total_delta}",
    );
}

/// After ticket-005, the simulation pipeline delegates every stage LP solve
/// through `run_stage_solve(Phase::Simulation)`. The unified path applies
/// `enforce_basic_count_invariant` uniformly across both the baked and
/// fallback arms, eliminating the 900 non-alien rejections observed on
/// convertido when the stored basis is stale relative to the current cut pool.
///
/// This test exercises the warm-start path end-to-end:
///   1. Trains for 3 iterations with `HighsSolver`, producing a populated
///      `basis_cache` in `TrainingResult`.
///   2. Runs simulation against the trained FCF with those stored bases.
///   3. Asserts `SolverStatsDelta::basis_non_alien_rejections == 0` across
///      all simulation scenarios — the preemptive fix for the observability
///      finding.
///
/// guards: unified-path
#[test]
fn simulation_zero_rejections_on_cut_churn() {
    let n_stages = 2_usize;
    let n_scenarios = 2_usize;

    let indexer = StageIndexer::new(1, 0); // N=1, L=0 → n_state=1
    let templates: Vec<StageTemplate> = vec![minimal_template(); n_stages];
    let base_rows = vec![2_usize; n_stages];
    let noise_scale = vec![0.0_f64; n_stages];
    let initial_state = vec![0.0_f64; indexer.n_state];
    let stochastic = make_stochastic_context(n_stages);

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![cobre_core::temporal::Block {
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
        })
        .collect();

    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let mut fcf = FutureCostFunction::new(
        n_stages,
        indexer.n_state,
        n_scenarios as u32,
        50,
        &vec![0; n_stages],
    );

    let stage_ctx = StageContext {
        templates: &templates,
        base_rows: &base_rows,
        noise_scale: &noise_scale,
        n_hydros: 1,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &vec![1_usize; n_stages],
        ncs_max_gen: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
    };

    let training_ctx = TrainingContext {
        horizon: &horizon,
        indexer: &indexer,
        inflow_method: &InflowNonNegativityMethod::None,
        stochastic: &stochastic,
        initial_state: &initial_state,
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        stages: &stages,
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
        recent_accum_seed: &[],
        recent_weight_seed: 0.0,
    };

    let comm = StubComm;

    // --- Training: 3 iterations to populate the basis_cache with warm-start
    // bases and add cuts to the FCF pools (creates the "stale basis" scenario
    // the invariant enforcement must handle).
    let training_config = TrainingConfig {
        loop_config: LoopConfig {
            forward_passes: n_scenarios as u32,
            max_iterations: 10,
            start_iteration: 0,
            n_fwd_threads: 1,
            max_blocks: 1,
            stopping_rules: StoppingRuleSet {
                rules: vec![StoppingRule::IterationLimit { limit: 3 }],
                mode: StoppingMode::Any,
            },
        },
        cut_management: CutManagementConfig {
            cut_selection: None,
            budget: None,
            cut_activity_tolerance: 0.0,
            warm_start_cuts: 0,
            risk_measures: vec![RiskMeasure::Expectation; n_stages],
            ..CutManagementConfig::default()
        },
        events: EventConfig {
            event_sender: None,
            checkpoint_interval: None,
            shutdown_flag: None,
            export_states: false,
        },
    };

    let training_outcome = train(
        &mut HighsSolver::new().expect("HighsSolver::new"),
        training_config,
        &mut fcf,
        &stage_ctx,
        &training_ctx,
        &comm,
        HighsSolver::new,
    )
    .expect("training must succeed");

    assert!(
        training_outcome.result.iterations >= 2,
        "need at least 2 iterations so cuts accumulate and the basis becomes potentially stale"
    );

    // basis_cache contains the warm-start bases from the final iteration.
    // These will be passed to simulate as stage_bases.
    let stage_bases = training_outcome.result.basis_cache;

    // --- Simulation: run against the trained FCF with stored bases.
    let sim_config = SimulationConfig {
        n_scenarios: n_scenarios as u32,
        io_channel_capacity: 8,
    };

    let entity_counts = EntityCounts {
        hydro_ids: vec![1_i32],
        hydro_productivities: vec![1.0],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![0_i32],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let (result_tx, result_rx) = mpsc::sync_channel(16);
    // Drain the channel in a background thread so it never fills.
    let _io_thread = std::thread::spawn(move || result_rx.into_iter().for_each(|_| {}));

    let mut sim_solver = HighsSolver::new().expect("HighsSolver::new for simulation");
    sim_solver.load_model(&templates[0]);
    let mut sim_workspaces = vec![SolverWorkspace::new(
        sim_solver,
        PatchBuffer::new(indexer.n_state, 0, 1, 0),
        indexer.n_state,
        WorkspaceSizing {
            hydro_count: 1,
            max_par_order: 0,
            n_load_buses: 0,
            max_blocks: 1,
            downstream_par_order: 0,
            max_openings: 0,
            initial_pool_capacity: 0,
            n_state: indexer.n_state,
        },
    )];

    let sim_result = simulate(
        &mut sim_workspaces,
        &stage_ctx,
        &fcf,
        &training_ctx,
        &sim_config,
        SimulationOutputSpec {
            result_tx: &result_tx,
            zeta_per_stage: &vec![1.0; n_stages],
            block_hours_per_stage: &vec![vec![744.0]; n_stages],
            entity_counts: &entity_counts,
            generic_constraint_row_entries: &vec![vec![]; n_stages],
            ncs_col_starts: &vec![0; n_stages],
            n_ncs_per_stage: &vec![0; n_stages],
            ncs_entity_ids_per_stage: &vec![vec![]; n_stages],
            diversion_upstream: &std::collections::HashMap::new(),
            hydro_productivities_per_stage: &vec![vec![1.0]; n_stages],
            event_sender: None,
        },
        None, // no baked templates — exercises the fallback (legacy) warm-start arm
        &stage_bases,
        &comm,
    )
    .expect("simulation must complete without error");

    // Primary assertion: the unified run_stage_solve path with uniform
    // enforce_basic_count_invariant must produce zero non-alien rejections.
    let total_rejections: u64 = sim_result
        .solver_stats
        .iter()
        .map(|(_, delta)| delta.basis_non_alien_rejections)
        .sum();

    assert_eq!(
        total_rejections, 0,
        "simulation must produce 0 non-alien basis rejections after ticket-005 rewire, \
         got {total_rejections} (preemptive fix for warm-start-observability-findings.md)"
    );
}

// ===========================================================================
// Ticket-006 cross-phase equivalence tests
// ===========================================================================
//
// These tests exercise `run_stage_solve` directly (not through the full
// forward/backward/simulation pipelines) to lock in the "unified path"
// guarantee: given identical inputs, every Phase produces identical
// basis reconstruction and LP solutions.
//
// `HighsSolver` is not `Clone`, so each phase gets a fresh workspace built
// from the same template and loaded with the same basis state.  Cloning the
// workspace would be cheaper but is not possible; rebuilding from the same
// deterministic template guarantees byte-identical solver state at entry.
// ---------------------------------------------------------------------------

/// Minimal 3-col / 2-row LP for cross-phase fixture use.
///
///   min  0*x0 + 1*x1 + 50*x2
///   s.t. x0            = 6   (state-fixing)
///        2*x0 + x2     = 14  (power balance)
///   x0 ∈ [0,10], x1 ∈ [0,+∞), x2 ∈ [0,8]
///
/// Optimal: x0=6, x1=8, x2=2 → objective=8+100=108.
fn eq_template() -> StageTemplate {
    StageTemplate {
        num_cols: 3,
        num_rows: 2,
        num_nz: 3,
        col_starts: vec![0_i32, 2, 2, 3],
        row_indices: vec![0_i32, 1, 1],
        values: vec![1.0, 2.0, 1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![10.0, f64::INFINITY, 8.0],
        objective: vec![0.0, 1.0, 50.0],
        row_lower: vec![6.0, 14.0],
        row_upper: vec![6.0, 14.0],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 1,
        n_hydro: 1,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

/// Build a fresh `SolverWorkspace<HighsSolver>` with the fixture LP loaded.
///
/// `pool_capacity` sets `WorkspaceSizing::initial_pool_capacity`, which
/// pre-sizes `ScratchBuffers::recon_slot_lookup`.  Pass a value large enough
/// to cover the maximum cut-slot index that `reconstruct_basis` will look up;
/// `0` is correct when no stored basis is provided (`stored_basis: None`).
fn eq_workspace(template: &StageTemplate, pool_capacity: usize) -> SolverWorkspace<HighsSolver> {
    let mut solver = HighsSolver::new().expect("HighsSolver::new");
    solver.load_model(template);
    SolverWorkspace::new(
        solver,
        PatchBuffer::new(0, 0, 0, 0),
        0,
        WorkspaceSizing {
            initial_pool_capacity: pool_capacity,
            ..WorkspaceSizing::default()
        },
    )
}

/// Build a `CutPool` with `n` active cuts at iteration 0, forward-pass indices
/// 0..n-1.  The pool has capacity 16 and `forward_passes = n` so that the slot
/// formula `0 + 0 * n + fp` maps fp ∈ [0, n) to slots [0, n).
///
/// The cut coefficients are `[1.0]` (state dimension = 1) and intercepts are
/// `0.0`.  These values are never actually evaluated by the tests; only the
/// slot identities and active count matter.
fn eq_pool_with_cuts(n: usize) -> CutPool {
    let mut pool = CutPool::new(16, 1, n as u32, 0);
    for fp in 0..n {
        pool.add_cut(0, fp as u32, 0.0, &[1.0_f64]);
    }
    pool
}

/// Add `n` placeholder cut rows to a solver workspace so the LP dimension
/// matches the basis that `eq_excess_basis` produces.
///
/// Each row is `(-∞, +∞)` with no non-zero coefficients — always satisfied and
/// irrelevant to the LP optimum, but they make the solver's internal row count
/// equal to `template.num_rows + n`, which is what `cobre_highs_set_basis_non_alien`
/// checks when validating the basis.
///
/// This mirrors what the forward/backward passes do when they call `add_rows`
/// before delegating to `run_stage_solve`.
fn eq_add_cut_rows(ws: &mut SolverWorkspace<HighsSolver>, n: usize) {
    let batch = RowBatch {
        num_rows: n,
        row_starts: vec![0_i32; n + 1],
        col_indices: vec![],
        values: vec![],
        row_lower: vec![-f64::INFINITY; n],
        row_upper: vec![f64::INFINITY; n],
    };
    ws.solver.add_rows(&batch);
}

/// Build a minimal `StageContext` wrapping a single template slice.
fn eq_context(templates: &[StageTemplate]) -> StageContext<'_> {
    StageContext {
        templates,
        base_rows: &[],
        noise_scale: &[],
        n_hydros: 0,
        n_load_buses: 0,
        load_balance_row_starts: &[],
        load_bus_indices: &[],
        block_counts_per_stage: &[],
        ncs_max_gen: &[],
        discount_factors: &[],
        cumulative_discount_factors: &[],
        stage_lag_transitions: &[],
        noise_group_ids: &[],
        downstream_par_order: 0,
    }
}

/// Build a `CapturedBasis` with `total_basic == num_row + excess`.
///
/// `num_cols = 3` (matches `eq_template`).
///
/// `cut_slots` holds the slot indices of the cut rows that were active at
/// capture time.  `excess` must equal `cut_slots.len()`: each slot contributes
/// one BASIC cut row, so `total_basic = num_cols + base_rows + excess` while
/// `num_row = base_rows + excess`.  That leaves `col_basic = num_cols = 3`
/// excess basic entries that `enforce_basic_count_invariant` demotes from the
/// cut-row section (indices `base_rows .. base_rows + excess`).
///
/// Column statuses: all `num_cols` BASIC (so the `col_basic` contribution is
/// `num_cols = 3` regardless of `base_rows`, giving `excess = 3` demotions).
/// Row statuses: all `base_rows + excess` entries BASIC.
fn eq_excess_basis(base_rows: usize, excess: usize, cut_slots: &[u32]) -> CapturedBasis {
    assert_eq!(
        cut_slots.len(),
        excess,
        "eq_excess_basis: cut_slots.len() must equal excess"
    );
    let num_cols = 3_usize;
    let total_rows = base_rows + excess;
    let mut cb = CapturedBasis::new(num_cols, total_rows, base_rows, excess, 1);
    // All columns BASIC so col_basic == num_cols == 3.
    cb.basis.col_status.clear();
    cb.basis.col_status.resize(num_cols, B);
    // All rows BASIC; cut rows will be demoted by enforce_basic_count_invariant.
    cb.basis.row_status.clear();
    cb.basis.row_status.resize(total_rows, B);
    // Record the captured cut-row slot indices.
    cb.cut_row_slots.extend_from_slice(cut_slots);
    cb.state_at_capture.push(0.0_f64);
    cb
}

/// Build `StageInputs` referencing the shared fixture objects.
fn eq_inputs<'a>(
    ctx: &'a StageContext<'a>,
    indexer: &'a StageIndexer,
    pool: &'a CutPool,
    basis: Option<&'a CapturedBasis>,
) -> StageInputs<'a> {
    StageInputs {
        stage_context: ctx,
        indexer,
        pool,
        current_state: &[0.0_f64],
        stored_basis: basis,
        baked_template: None,
        stage_index: 0,
        scenario_index: 0,
        horizon_is_terminal: false,
        terminal_has_boundary_cuts: false,
        iteration: None,
    }
}

// ---------------------------------------------------------------------------
// Helper: extract (primal, dual, recon_stats) from any StageOutcome variant.
// ---------------------------------------------------------------------------

fn unpack_outcome(outcome: &StageOutcome<'_>) -> (Vec<f64>, Vec<f64>, ReconstructionStats) {
    match outcome {
        StageOutcome::Forward { view, recon_stats }
        | StageOutcome::Backward { view, recon_stats }
        | StageOutcome::Simulation { view, recon_stats } => {
            (view.primal.to_vec(), view.dual.to_vec(), *recon_stats)
        }
    }
}

// ---------------------------------------------------------------------------
// Test 1: cross_phase_identical_inputs_identical_reconstruction
// ---------------------------------------------------------------------------

/// Verify that `run_stage_solve` produces byte-identical `recon_stats`,
/// `view.primal`, and `view.dual` regardless of `Phase`.
///
/// Setup: excess basis (2 base rows, excess=3) so that
/// `enforce_basic_count_invariant` performs real demotion work on every call.
/// Each phase gets its own fresh workspace built from the same deterministic
/// template — determinism guarantees byte-identical solver state at entry.
///
/// guards: unified-path
#[test]
fn cross_phase_identical_inputs_identical_reconstruction() {
    let template = eq_template();
    let templates = std::slice::from_ref(&template);
    // 3 active cuts at slots [0,1,2] so the LP has 2+3=5 rows and
    // enforce_basic_count_invariant can demote 3 BASIC cut rows.
    let pool = eq_pool_with_cuts(3);
    let indexer = StageIndexer::new(1, 0);
    // excess=3 cut rows (slots [0,1,2]), all BASIC at capture; all 3 cols BASIC
    // gives col_basic=3 so total_basic=3+5=8, num_row=5, excess=3 demotions.
    let captured = eq_excess_basis(2, 3, &[0, 1, 2]);

    let mut results: Vec<(Vec<f64>, Vec<f64>, ReconstructionStats)> = Vec::new();

    for phase in [Phase::Forward, Phase::Backward, Phase::Simulation] {
        let ctx = eq_context(templates);
        // pool_capacity=16 pre-sizes recon_slot_lookup to cover slots [0,1,2].
        let mut ws = eq_workspace(&template, 16);
        // Add 3 cut rows so the LP dimension matches the 5-row basis produced
        // by eq_excess_basis. This mirrors what forward/backward callers do
        // (via add_rows) before delegating to run_stage_solve.
        eq_add_cut_rows(&mut ws, 3);
        let inputs = eq_inputs(&ctx, &indexer, &pool, Some(&captured));
        let outcome =
            run_stage_solve(&mut ws, phase, &inputs).expect("solve must succeed for all phases");
        results.push(unpack_outcome(&outcome));
    }

    let (fwd_primal, fwd_dual, fwd_stats) = &results[0];
    let (bwd_primal, bwd_dual, bwd_stats) = &results[1];
    let (sim_primal, sim_dual, sim_stats) = &results[2];

    assert_eq!(
        fwd_stats,
        bwd_stats,
        "Forward vs Backward recon_stats mismatch: \
         preserved={}/{}, new_tight={}/{}, new_slack={}/{}",
        fwd_stats.preserved,
        bwd_stats.preserved,
        fwd_stats.new_tight,
        bwd_stats.new_tight,
        fwd_stats.new_slack,
        bwd_stats.new_slack,
    );
    assert_eq!(
        bwd_stats,
        sim_stats,
        "Backward vs Simulation recon_stats mismatch: \
         preserved={}/{}, new_tight={}/{}, new_slack={}/{}",
        bwd_stats.preserved,
        sim_stats.preserved,
        bwd_stats.new_tight,
        sim_stats.new_tight,
        bwd_stats.new_slack,
        sim_stats.new_slack,
    );

    assert_eq!(
        fwd_primal, bwd_primal,
        "Forward vs Backward view.primal mismatch: {fwd_primal:?} vs {bwd_primal:?}",
    );
    assert_eq!(
        bwd_primal, sim_primal,
        "Backward vs Simulation view.primal mismatch: {bwd_primal:?} vs {sim_primal:?}",
    );

    assert_eq!(
        fwd_dual, bwd_dual,
        "Forward vs Backward view.dual mismatch: {fwd_dual:?} vs {bwd_dual:?}",
    );
    assert_eq!(
        bwd_dual, sim_dual,
        "Backward vs Simulation view.dual mismatch: {bwd_dual:?} vs {sim_dual:?}",
    );
}

// ---------------------------------------------------------------------------
// Test 2: phase_only_affects_outcome_variant_not_solve_path
// ---------------------------------------------------------------------------

/// Verify that `SolverStatistics` deltas are identical across all three phases.
///
/// Expected deltas per phase (one call to `run_stage_solve` with a stored basis):
///   - `solve_count` +1  (one LP solve)
///   - `basis_offered` +1 (one warm-start attempt)
///   - `load_model_count` +0 (no model reload inside `run_stage_solve`)
///
/// guards: unified-path
#[test]
fn phase_only_affects_outcome_variant_not_solve_path() {
    let template = eq_template();
    let templates = std::slice::from_ref(&template);
    // 3 active cuts at slots [0,1,2] so the LP has 2+3=5 rows and
    // enforce_basic_count_invariant can demote 3 BASIC cut rows.
    let pool = eq_pool_with_cuts(3);
    let indexer = StageIndexer::new(1, 0);
    // excess=3 cut rows (slots [0,1,2]), all BASIC at capture.
    let captured = eq_excess_basis(2, 3, &[0, 1, 2]);

    // Collect (solve_count_delta, basis_offered_delta, load_model_count_delta).
    let mut deltas: Vec<(u64, u64, u64)> = Vec::new();

    for phase in [Phase::Forward, Phase::Backward, Phase::Simulation] {
        let ctx = eq_context(templates);
        // pool_capacity=16 pre-sizes recon_slot_lookup to cover slots [0,1,2].
        let mut ws = eq_workspace(&template, 16);
        // Add 3 cut rows so the LP dimension matches the 5-row basis produced
        // by eq_excess_basis. This mirrors what forward/backward callers do
        // (via add_rows) before delegating to run_stage_solve.
        eq_add_cut_rows(&mut ws, 3);
        // Snapshot statistics before the call.
        let before: SolverStatistics = ws.solver.statistics();

        let inputs = eq_inputs(&ctx, &indexer, &pool, Some(&captured));
        run_stage_solve(&mut ws, phase, &inputs).expect("solve must succeed for all phases");

        let after: SolverStatistics = ws.solver.statistics();
        deltas.push((
            after.solve_count - before.solve_count,
            after.basis_offered - before.basis_offered,
            after.load_model_count - before.load_model_count,
        ));
    }

    let (fwd_solves, fwd_offered, fwd_loads) = deltas[0];
    let (bwd_solves, bwd_offered, bwd_loads) = deltas[1];
    let (sim_solves, sim_offered, sim_loads) = deltas[2];

    // Each phase must issue exactly one LP solve.
    assert_eq!(
        fwd_solves, 1,
        "Phase::Forward: expected solve_count delta=1, got {fwd_solves}"
    );
    assert_eq!(
        bwd_solves, 1,
        "Phase::Backward: expected solve_count delta=1, got {bwd_solves}"
    );
    assert_eq!(
        sim_solves, 1,
        "Phase::Simulation: expected solve_count delta=1, got {sim_solves}"
    );

    // Each phase must offer the stored basis exactly once.
    assert_eq!(
        fwd_offered, 1,
        "Phase::Forward: expected basis_offered delta=1, got {fwd_offered}"
    );
    assert_eq!(
        bwd_offered, 1,
        "Phase::Backward: expected basis_offered delta=1, got {bwd_offered}"
    );
    assert_eq!(
        sim_offered, 1,
        "Phase::Simulation: expected basis_offered delta=1, got {sim_offered}"
    );

    // No model reload should occur inside a single run_stage_solve call.
    assert_eq!(
        fwd_loads, 0,
        "Phase::Forward: expected load_model_count delta=0, got {fwd_loads}"
    );
    assert_eq!(
        bwd_loads, 0,
        "Phase::Backward: expected load_model_count delta=0, got {bwd_loads}"
    );
    assert_eq!(
        sim_loads, 0,
        "Phase::Simulation: expected load_model_count delta=0, got {sim_loads}"
    );

    // Delta tuples must be identical across all three phases.
    assert_eq!(
        (fwd_solves, fwd_offered, fwd_loads),
        (bwd_solves, bwd_offered, bwd_loads),
        "Forward vs Backward SolverStatistics deltas differ: \
         (solves,offered,loads)=({fwd_solves},{fwd_offered},{fwd_loads}) \
         vs ({bwd_solves},{bwd_offered},{bwd_loads})"
    );
    assert_eq!(
        (bwd_solves, bwd_offered, bwd_loads),
        (sim_solves, sim_offered, sim_loads),
        "Backward vs Simulation SolverStatistics deltas differ: \
         (solves,offered,loads)=({bwd_solves},{bwd_offered},{bwd_loads}) \
         vs ({sim_solves},{sim_offered},{sim_loads})"
    );
}

// ---------------------------------------------------------------------------
// Test 4: cold_start_zero_recon_stats
// ---------------------------------------------------------------------------

/// Verify that `stored_basis: None` produces `recon_stats == default()` for
/// all three phases, and that the cold solve still returns a valid solution.
///
/// guards: unified-path
#[test]
fn cold_start_zero_recon_stats() {
    let template = eq_template();
    let templates = std::slice::from_ref(&template);
    let pool = CutPool::new(16, 1, 1, 0);
    let indexer = StageIndexer::new(1, 0);

    for phase in [Phase::Forward, Phase::Backward, Phase::Simulation] {
        let ctx = eq_context(templates);
        let mut ws = eq_workspace(&template, 0);

        let inputs = eq_inputs(&ctx, &indexer, &pool, None);
        let outcome = run_stage_solve(&mut ws, phase, &inputs)
            .expect("cold start must succeed for all phases");

        let (primal, _dual, recon_stats) = unpack_outcome(&outcome);

        assert_eq!(
            recon_stats,
            ReconstructionStats::default(),
            "Phase::{phase:?}: cold start must produce recon_stats == default(), \
             got preserved={}, new_tight={}, new_slack={}",
            recon_stats.preserved,
            recon_stats.new_tight,
            recon_stats.new_slack,
        );

        // Cold solve must still produce a valid (non-empty) primal solution.
        assert!(
            !primal.is_empty(),
            "Phase::{phase:?}: cold start must return a non-empty primal solution"
        );
    }
}
