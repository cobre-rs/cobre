use std::collections::BTreeMap;

use chrono::NaiveDate;
use cobre_comm::{CommData, Communicator, ReduceOp};
use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
use cobre_core::scenario::{
    CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
    LoadModel, SamplingScheme,
};
use cobre_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
};
use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder, WorkerPhaseTimings};
use cobre_solver::{
    Basis, LpSolution, ProfiledSolver, RowBatch, SolverError, SolverInterface, SolverStatistics,
    StageTemplate,
};
use cobre_stochastic::StochasticContext;
use cobre_stochastic::context::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

use cobre_comm::LocalBackend;

use super::stats_aggregation::weighted_cost_reduction;
use super::{
    ForwardBound, ForwardPassBatch, ForwardResult, SyncResult, build_delta_cut_row_batch_into,
    run_forward_pass, sync_forward,
};
use crate::cut::row::build_cut_row_batch_into;
use crate::solve::partition;
use crate::{
    CutPool, SddpError, StoppingMode, StoppingRule, StoppingRuleSet, TrainingConfig,
    config::{CutManagementConfig, EventConfig, LoopConfig},
    context::TrainingContext,
    cut::FutureCostFunction,
    horizon_mode::HorizonMode,
    inflow_method::InflowNonNegativityMethod,
    lp::builder::PatchBuffer,
    lp::indexer::StateSpace,
    risk_measure::RiskMeasure,
    setup::{NodeId, NodePos},
    test_support::{self, StageContextFixture, equipment_free_geometry, permissive_state_boxes},
    trajectory::TrajectoryRecord,
    workspace::{BackwardAccumulators, BasisStore, ScratchBuffers, SolverWorkspace},
};

// ── Mock solver ──────────────────────────────────────────────────────────

/// Mock solver that returns a configurable fixed `LpSolution` on every `solve()`.
///
/// Optionally returns `SolverError::Infeasible` at the n-th solve call (0-indexed).
struct MockSolver {
    solution: LpSolution,
    /// If `Some(n)`, the n-th solve call (0-indexed, counting both cold-start
    /// and warm-start calls) returns infeasible.
    infeasible_at: Option<usize>,
    call_count: usize,
    /// Number of times `solve(Some(&basis))` has been called.
    warm_start_calls: usize,
    /// Internal buffers that `SolutionView` borrows from.
    buf_primal: Vec<f64>,
    buf_dual: Vec<f64>,
    buf_reduced_costs: Vec<f64>,
}

impl MockSolver {
    fn always_ok(solution: LpSolution) -> Self {
        let buf_primal = solution.primal.clone();
        let buf_dual = solution.dual.clone();
        let buf_reduced_costs = solution.reduced_costs.clone();
        Self {
            solution,
            infeasible_at: None,
            call_count: 0,
            warm_start_calls: 0,
            buf_primal,
            buf_dual,
            buf_reduced_costs,
        }
    }

    fn infeasible_on(solution: LpSolution, n: usize) -> Self {
        let buf_primal = solution.primal.clone();
        let buf_dual = solution.dual.clone();
        let buf_reduced_costs = solution.reduced_costs.clone();
        Self {
            solution,
            infeasible_at: Some(n),
            call_count: 0,
            warm_start_calls: 0,
            buf_primal,
            buf_dual,
            buf_reduced_costs,
        }
    }

    fn do_solve(&mut self) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
        let call = self.call_count;
        self.call_count += 1;
        if self.infeasible_at == Some(call) {
            return Err(SolverError::Infeasible);
        }
        self.buf_primal.clone_from(&self.solution.primal);
        self.buf_dual.clone_from(&self.solution.dual);
        self.buf_reduced_costs
            .clone_from(&self.solution.reduced_costs);
        Ok(cobre_solver::SolutionView {
            objective: self.solution.objective,
            primal: &self.buf_primal,
            dual: &self.buf_dual,
            reduced_costs: &self.buf_reduced_costs,
            iterations: self.solution.iterations,
            solve_time_seconds: self.solution.solve_time_seconds,
        })
    }
}

impl SolverInterface for MockSolver {
    type Profile = cobre_solver::ActiveProfile;

    fn apply_profile(&mut self, _profile: &cobre_solver::ActiveProfile) {}

    fn solver_name_version(&self) -> String {
        "MockSolver 0.0.0".to_string()
    }
    fn load_model(&mut self, _template: &StageTemplate) {}

    fn add_rows(&mut self, _cuts: &RowBatch) {}

    fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

    fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

    fn solve(
        &mut self,
        basis: Option<&Basis>,
    ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
        if basis.is_some() {
            self.warm_start_calls += 1;
        }
        self.do_solve()
    }

    fn get_basis(&mut self, out: &mut Basis) {
        crate::test_support::fill_consistent_basis(out);
    }

    fn statistics(&self) -> SolverStatistics {
        SolverStatistics {
            solve_count: self.call_count as u64,
            ..SolverStatistics::default()
        }
    }

    fn statistics_into(&self, out: &mut SolverStatistics) {
        out.copy_from(&self.statistics());
    }

    fn name(&self) -> &'static str {
        "Mock"
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Minimal N=1, L=0 template: `[storage_out, z_inflow, storage_in, theta]`.
/// row 0: z-inflow definition (`z_inflow[0]` = rhs); row 1: pins `storage_in`.
fn minimal_template_1_0() -> StageTemplate {
    StageTemplate {
        num_cols: 4,
        num_rows: 2,
        num_nz: 2,
        col_starts: vec![0_i32, 0, 1, 2, 2], // col 1 (z_inflow) NZ at row 0; col 2 (storage_in) NZ at row 1
        row_indices: vec![0_i32, 1],
        values: vec![1.0, 1.0],
        col_lower: vec![0.0, f64::NEG_INFINITY, 0.0, 0.0],
        col_upper: vec![f64::INFINITY; 4],
        objective: vec![0.0, 0.0, 0.0, 1.0], // minimise theta (at col 3)
        row_lower: vec![0.0, 0.0],
        row_upper: vec![0.0, 0.0],
        n_state: 1,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

fn fixed_solution(num_cols: usize, objective: f64, theta_col: usize, theta_val: f64) -> LpSolution {
    let mut primal = vec![0.0_f64; num_cols];
    primal[theta_col] = theta_val;
    LpSolution {
        objective,
        primal,
        dual: vec![0.0; 1], // forward pass never reads dual; length is arbitrary
        reduced_costs: vec![0.0; num_cols],
        iterations: 0,
        solve_time_seconds: 0.0,
    }
}

fn empty_records(n: usize) -> Vec<TrajectoryRecord> {
    (0..n)
        .map(|_| TrajectoryRecord {
            primal: Vec::new(),
            dual: Vec::new(),
            stage_cost: 0.0,
            node_id: NodeId(0),
            state: Vec::new(),
        })
        .collect()
}

/// Build a minimal `StochasticContext` for a single-hydro, `n_stages`-stage
/// system; a matching `default` self-correlation profile when `with_profile`,
/// an empty profile map otherwise.
///
/// Used by integration tests that call `run_forward_pass`. The `MockSolver`
/// ignores the noise values produced by `sample_forward`, so the exact
/// stochastic parameterisation does not affect correctness; it only needs
/// to be structurally valid for the sampling API.
#[allow(clippy::too_many_lines)]
fn make_stochastic_context_1_hydro(n_stages: usize, with_profile: bool) -> StochasticContext {
    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(1),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
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
    };
    hydro.declare_mirror_unit_group(EntityId(0));
    let make_stage = |idx: usize, id: i32| Stage {
        index: idx,
        id,
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
            branching_factor: 3,
            noise_method: NoiseMethod::Saa,
        },
    };
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let stages: Vec<Stage> = (0..n_stages)
        .map(|idx| make_stage(idx, idx as i32))
        .collect();
    let inflow = |stage_id: i32| InflowModel {
        hydro_id: EntityId(1),
        stage_id,
        mean_m3s: 100.0,
        std_m3s: 30.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    };
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let inflow_models: Vec<InflowModel> = (0..n_stages).map(|s| inflow(s as i32)).collect();
    let mut profiles = BTreeMap::new();
    if with_profile {
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
    }
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

// ── Unit tests: ForwardResult ────────────────────────────────────────────

#[test]
fn forward_result_field_access() {
    let r = ForwardResult {
        scenario_costs: vec![60.0, 70.0, 80.0, 90.0],
        elapsed_ms: 123,
        lp_solves: 0,
        setup_time_ms: 0,
        load_imbalance_ms: 0,
        scheduling_overhead_ms: 0,
        stage_stats: Vec::new(),
    };
    assert_eq!(r.scenario_costs.len(), 4);
    assert_eq!(r.scenario_costs[0], 60.0);
    assert_eq!(r.elapsed_ms, 123);
}

#[test]
fn forward_result_clone_and_debug() {
    let r = ForwardResult {
        scenario_costs: vec![1.0, 2.0],
        elapsed_ms: 5,
        lp_solves: 0,
        setup_time_ms: 0,
        load_imbalance_ms: 0,
        scheduling_overhead_ms: 0,
        stage_stats: Vec::new(),
    };
    let c = r.clone();
    assert_eq!(c.scenario_costs.len(), r.scenario_costs.len());
    assert_eq!(c.scenario_costs[0].to_bits(), r.scenario_costs[0].to_bits());
    let s = format!("{r:?}");
    assert!(s.contains("ForwardResult"));
}

// ── Unit tests: forward overhead decomposition ───────────────────────────

#[test]
fn forward_overhead_decomposition_four_workers() {
    use cobre_solver::SolverStatistics;

    use crate::solver_stats::SolverStatsDelta;

    fn make_stats(
        solve_s: f64,
        load_model_s: f64,
        set_bounds_s: f64,
        basis_set_s: f64,
    ) -> SolverStatistics {
        SolverStatistics {
            total_solve_time_seconds: solve_s,
            total_load_model_time_seconds: load_model_s,
            total_set_bounds_time_seconds: set_bounds_s,
            total_basis_set_time_seconds: basis_set_s,
            ..SolverStatistics::default()
        }
    }

    let befores = [
        make_stats(0.0, 0.0, 0.0, 0.0),
        make_stats(0.0, 0.0, 0.0, 0.0),
        make_stats(0.0, 0.0, 0.0, 0.0),
        make_stats(0.0, 0.0, 0.0, 0.0),
    ];
    let afters = [
        make_stats(0.500, 0.050, 0.0, 0.0),
        make_stats(0.600, 0.060, 0.0, 0.0),
        make_stats(0.550, 0.045, 0.0, 0.0),
        make_stats(0.580, 0.055, 0.0, 0.0),
    ];

    let deltas: Vec<SolverStatsDelta> = befores
        .iter()
        .zip(&afters)
        .map(|(b, a)| SolverStatsDelta::from_snapshots(b, a))
        .collect();

    let setup_ms: f64 = deltas
        .iter()
        .map(|d| d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms)
        .sum();

    let worker_totals: Vec<f64> = deltas
        .iter()
        .map(|d| {
            d.solve_time_ms + d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms
        })
        .collect();

    let n_workers_f = 4.0_f64;
    let max_ms = worker_totals.iter().copied().fold(0.0_f64, f64::max);
    let avg_ms = worker_totals.iter().sum::<f64>() / n_workers_f;
    let imbalance_ms = (max_ms - avg_ms).max(0.0);
    let parallel_wall_ms = 700_u64; // 40 ms above the slowest worker
    #[allow(clippy::cast_precision_loss)]
    let scheduling_ms = (parallel_wall_ms as f64 - max_ms).max(0.0);

    // setup_time_ms = 50 + 60 + 45 + 55 = 210
    assert!(
        (setup_ms - 210.0).abs() < 0.001,
        "setup_time_ms should be 210, got {setup_ms}"
    );
    // max_worker total = 660 (worker 1: 600+60)
    assert!(
        (max_ms - 660.0).abs() < 0.001,
        "max_worker_ms should be 660, got {max_ms}"
    );
    // avg = (550+660+595+635)/4 = 610
    assert!(
        (avg_ms - 610.0).abs() < 0.001,
        "avg_worker_ms should be 610, got {avg_ms}"
    );
    // imbalance = 660 - 610 = 50
    assert!(
        (imbalance_ms - 50.0).abs() < 0.001,
        "load_imbalance_ms should be 50, got {imbalance_ms}"
    );
    // scheduling = 700 - 660 = 40
    assert!(
        (scheduling_ms - 40.0).abs() < 0.001,
        "scheduling_overhead_ms should be 40, got {scheduling_ms}"
    );
}

#[test]
fn forward_overhead_decomposition_single_worker_zero_imbalance() {
    use cobre_solver::SolverStatistics;

    use crate::solver_stats::SolverStatsDelta;

    let before = SolverStatistics::default();
    let after = SolverStatistics {
        total_solve_time_seconds: 1.0,
        total_load_model_time_seconds: 0.1,
        ..SolverStatistics::default()
    };

    let deltas = [SolverStatsDelta::from_snapshots(&before, &after)];
    let worker_totals: Vec<f64> = deltas
        .iter()
        .map(|d| {
            d.solve_time_ms + d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms
        })
        .collect();

    let n_workers_f = 1.0_f64;
    let max_ms = worker_totals.iter().copied().fold(0.0_f64, f64::max);
    let avg_ms = worker_totals.iter().sum::<f64>() / n_workers_f;
    let imbalance_ms = (max_ms - avg_ms).max(0.0);

    assert_eq!(
        imbalance_ms, 0.0,
        "load_imbalance_ms must be 0.0 for a single worker"
    );
}

#[test]
fn forward_overhead_scheduling_clamped_to_zero_on_clock_skew() {
    use cobre_solver::SolverStatistics;

    use crate::solver_stats::SolverStatsDelta;

    let before = SolverStatistics::default();
    let after = SolverStatistics {
        total_solve_time_seconds: 1.0, // 1000 ms
        ..SolverStatistics::default()
    };

    let deltas = [SolverStatsDelta::from_snapshots(&before, &after)];
    let max_ms = deltas
        .iter()
        .map(|d| d.solve_time_ms)
        .fold(0.0_f64, f64::max);

    // Wall time (800 ms) < max worker total (1000 ms) — clock skew.
    let parallel_wall_ms = 800_u64;
    #[allow(clippy::cast_precision_loss)]
    let scheduling_ms = (parallel_wall_ms as f64 - max_ms).max(0.0);

    assert_eq!(
        scheduling_ms, 0.0,
        "scheduling_overhead_ms must clamp to 0.0 on clock skew"
    );
}

fn single_workspace(solver: MockSolver, state: &StateSpace) -> SolverWorkspace<MockSolver> {
    SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(solver),
        patch_buf: PatchBuffer::new(state, &[], &[]),
        current_state: Vec::with_capacity(state.n_state),
        scratch: ScratchBuffers {
            inflow_m3s_buf: Vec::with_capacity(state.hydro_count),
            lag_matrix_buf: Vec::with_capacity(state.max_par_order * state.hydro_count),
            par_inflow_buf: Vec::with_capacity(state.hydro_count),
            eta_floor_buf: Vec::with_capacity(state.hydro_count),
            zero_targets_buf: vec![0.0_f64; state.hydro_count],
            ncs_col_upper_buf: Vec::new(),
            ncs_col_lower_buf: Vec::new(),
            ncs_col_indices_buf: Vec::new(),
            ncs_col_lower_active_buf: Vec::new(),
            ncs_col_upper_active_buf: Vec::new(),
            last_ncs_col_start: usize::MAX,
            ncs_col_upper_extract_buf: Vec::new(),
            load_rhs_buf: Vec::new(),
            row_lower_buf: Vec::new(),
            z_inflow_rhs_buf: Vec::new(),
            effective_eta_buf: Vec::new(),
            unscaled_primal: Vec::new(),
            unscaled_dual: Vec::new(),
            lag_accumulator: vec![],
            lag_weight_accum: vec![],
            downstream_accumulator: Vec::new(),
            downstream_weight_accum: 0.0,
            downstream_completed_lags: Vec::new(),
            downstream_n_completed: 0,
            recon_slot_lookup: Vec::new(),
            trajectory_costs_buf: Vec::new(),
            raw_noise_buf: Vec::new(),
            corr_scratch: Vec::new(),
            current_node_buf: Vec::new(),
        },
        scratch_basis: Basis::new(0, 0),
        backward_accum: BackwardAccumulators::default(),
        worker_timing_buf: WorkerPhaseTimings::default(),
    }
}

/// Build 3 minimal [`Stage`] values matching `make_stochastic_context_1_hydro(3, true)`.
///
/// Provides the `stages` slice required by [`TrainingContext`] so that
/// [`cobre_stochastic::build_forward_sampler`] can read per-stage noise methods.
fn make_stages_3() -> Vec<Stage> {
    let make_stage = |idx: usize, id: i32| Stage {
        index: idx,
        id,
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
            branching_factor: 3,
            noise_method: NoiseMethod::Saa,
        },
    };
    vec![make_stage(0, 0), make_stage(1, 1), make_stage(2, 2)]
}

// ── Acceptance criteria integration tests ───────────────────────────────

#[test]
#[allow(clippy::too_many_lines)]
fn ac_two_scenarios_three_stages_fixed_solution() {
    // State layout: N=1, L=0 → n_state=1, theta=3, num_cols=4
    let state = test_support::state_layout(1, 0);
    let solution = fixed_solution(4, 100.0, state.theta, 30.0);
    let solver = MockSolver::always_ok(solution);
    let fcf = FutureCostFunction::new(3, state.n_state, 2, 100, &[0; 3]);
    let config = TrainingConfig {
        loop_config: LoopConfig {
            forward_passes: 2,
            training_enumerated: false,
            max_iterations: 100,
            start_iteration: 0,
            resume_lower_bound_history: Vec::new(),
            n_fwd_threads: 1,
            stopping_rules: StoppingRuleSet {
                rules: vec![StoppingRule::IterationLimit { limit: 100 }],
                mode: StoppingMode::Any,
            },
        },
        cut_management: CutManagementConfig {
            cut_selection: None,
            budget: None,
            cut_activity_tolerance: 0.0,
            risk_measures: vec![RiskMeasure::Expectation],
        },
        events: EventConfig {
            event_sender: None,
            periodic_checkpoint: None,
            shutdown_flag: None,
            export_states: false,
        },
    };

    let horizon = HorizonMode::Finite { num_stages: 3 };

    let templates = vec![
        minimal_template_1_0(),
        minimal_template_1_0(),
        minimal_template_1_0(),
    ];
    let initial_state = vec![0.0_f64; state.n_state];
    let mut records = empty_records(2 * 3);
    let stochastic = make_stochastic_context_1_hydro(3, true);
    let stages = make_stages_3();
    let mut ws = single_workspace(solver, &state);
    let mut basis_store =
        BasisStore::new(config.loop_config.forward_passes as usize, templates.len());

    let state_boxes = permissive_state_boxes(state.n_state, templates.len());
    let geometry = equipment_free_geometry(&[1usize, 1, 1]);
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let ctx = fixture.ctx();
    let result = run_forward_pass(
        std::slice::from_mut(&mut ws),
        &mut basis_store,
        &ctx,
        &templates,
        &fcf,
        &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(
                &state,
                horizon.num_stages(),
            ),
            study_dims: &test_support::study_dims(),
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
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &ForwardPassBatch {
            local_forward_passes: config.loop_config.forward_passes as usize,
            total_forward_passes: config.loop_config.forward_passes as usize,
            iteration: 0,
            fwd_offset: 0,
            event_sender: None,
        },
        &mut records,
    )
    .unwrap();

    assert_eq!(result.scenario_costs.len(), 2);
    // stage_cost = (100 - 30) * COST_SCALE_FACTOR = 70_000_000.
    for (i, record) in records.iter().enumerate() {
        assert_eq!(
            record.stage_cost, 70_000_000.0,
            "record[{i}].stage_cost should be 70_000_000.0 ((objective - theta) * COST_SCALE_FACTOR)"
        );
    }
    // each scenario cost = 70_000_000 * 3 stages = 210_000_000.
    assert_eq!(result.scenario_costs[0], 210_000_000.0);
    assert_eq!(result.scenario_costs[1], 210_000_000.0);
}

#[test]
#[allow(clippy::too_many_lines)]
fn ac_infeasible_at_stage_1_scenario_0_returns_infeasible_error() {
    let state = test_support::state_layout(1, 0);
    let solution = fixed_solution(4, 100.0, state.theta, 30.0);
    // Stage-first loop: with 2 scenarios and 3 stages, the solve order is
    // (s0,t0), (s1,t0), (s0,t1), (s1,t1), ... — the 3rd call (index 2)
    // is stage 1 of scenario 0.
    let solver = MockSolver::infeasible_on(solution, 2);
    let fcf = FutureCostFunction::new(3, state.n_state, 2, 100, &[0; 3]);
    let config = TrainingConfig {
        loop_config: LoopConfig {
            forward_passes: 2,
            training_enumerated: false,
            max_iterations: 100,
            start_iteration: 0,
            resume_lower_bound_history: Vec::new(),
            n_fwd_threads: 1,
            stopping_rules: StoppingRuleSet {
                rules: vec![StoppingRule::IterationLimit { limit: 100 }],
                mode: StoppingMode::Any,
            },
        },
        cut_management: CutManagementConfig {
            cut_selection: None,
            budget: None,
            cut_activity_tolerance: 0.0,
            risk_measures: vec![RiskMeasure::Expectation],
        },
        events: EventConfig {
            event_sender: None,
            periodic_checkpoint: None,
            shutdown_flag: None,
            export_states: false,
        },
    };

    let horizon = HorizonMode::Finite { num_stages: 3 };

    let templates = vec![
        minimal_template_1_0(),
        minimal_template_1_0(),
        minimal_template_1_0(),
    ];
    let initial_state = vec![0.0_f64; state.n_state];
    let mut records = empty_records(2 * 3);
    let stochastic = make_stochastic_context_1_hydro(3, true);
    let stages = make_stages_3();
    let mut ws = single_workspace(solver, &state);
    let mut basis_store =
        BasisStore::new(config.loop_config.forward_passes as usize, templates.len());

    let state_boxes = permissive_state_boxes(state.n_state, templates.len());
    let geometry = equipment_free_geometry(&[1usize, 1, 1]);
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let ctx = fixture.ctx();
    let result = run_forward_pass(
        std::slice::from_mut(&mut ws),
        &mut basis_store,
        &ctx,
        &templates,
        &fcf,
        &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(
                &state,
                horizon.num_stages(),
            ),
            study_dims: &test_support::study_dims(),
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
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &ForwardPassBatch {
            local_forward_passes: config.loop_config.forward_passes as usize,
            total_forward_passes: config.loop_config.forward_passes as usize,
            iteration: 0,
            fwd_offset: 0,
            event_sender: None,
        },
        &mut records,
    );

    match result {
        Err(SddpError::Infeasible {
            stage, scenario, ..
        }) => {
            assert_eq!(stage, 1, "expected stage=1");
            assert_eq!(scenario, 0, "expected scenario=0");
        }
        other => panic!("expected Infeasible, got {other:?}"),
    }
}

#[test]
fn ac_global_scenario_index_rank1_scenario0() {
    // global_scenario = rank * forward_passes + m = 1 * 3 + 0 = 3
    let rank = 1usize;
    let forward_passes = 3usize;
    let m = 0usize;
    let global_scenario = rank * forward_passes + m;
    assert_eq!(global_scenario, 3);
}

#[test]
#[allow(clippy::too_many_lines)]
fn cost_statistics_accumulated_correctly() {
    let state = test_support::state_layout(1, 0);
    let solution = fixed_solution(4, 100.0, state.theta, 30.0);
    let solver = MockSolver::always_ok(solution);
    let fcf = FutureCostFunction::new(3, state.n_state, 2, 100, &[0; 3]);
    let config = TrainingConfig {
        loop_config: LoopConfig {
            forward_passes: 2,
            training_enumerated: false,
            max_iterations: 100,
            start_iteration: 0,
            resume_lower_bound_history: Vec::new(),
            n_fwd_threads: 1,
            stopping_rules: StoppingRuleSet {
                rules: vec![StoppingRule::IterationLimit { limit: 100 }],
                mode: StoppingMode::Any,
            },
        },
        cut_management: CutManagementConfig {
            cut_selection: None,
            budget: None,
            cut_activity_tolerance: 0.0,
            risk_measures: vec![RiskMeasure::Expectation],
        },
        events: EventConfig {
            event_sender: None,
            periodic_checkpoint: None,
            shutdown_flag: None,
            export_states: false,
        },
    };

    let horizon = HorizonMode::Finite { num_stages: 3 };

    let templates = vec![
        minimal_template_1_0(),
        minimal_template_1_0(),
        minimal_template_1_0(),
    ];
    let initial_state = vec![0.0_f64; state.n_state];
    let mut records = empty_records(2 * 3);
    let stochastic = make_stochastic_context_1_hydro(3, true);
    let stages = make_stages_3();
    let mut ws = single_workspace(solver, &state);
    let mut basis_store =
        BasisStore::new(config.loop_config.forward_passes as usize, templates.len());

    let state_boxes = permissive_state_boxes(state.n_state, templates.len());
    let geometry = equipment_free_geometry(&[1usize, 1, 1]);
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let ctx = fixture.ctx();
    let result = run_forward_pass(
        std::slice::from_mut(&mut ws),
        &mut basis_store,
        &ctx,
        &templates,
        &fcf,
        &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(
                &state,
                horizon.num_stages(),
            ),
            study_dims: &test_support::study_dims(),
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
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &ForwardPassBatch {
            local_forward_passes: config.loop_config.forward_passes as usize,
            total_forward_passes: config.loop_config.forward_passes as usize,
            iteration: 0,
            fwd_offset: 0,
            event_sender: None,
        },
        &mut records,
    )
    .unwrap();

    // stage_cost per solve = (100 - 30) * COST_SCALE_FACTOR = 70_000_000
    // total_cost per scenario = 70_000_000 * 3 stages = 210_000_000
    assert_eq!(result.scenario_costs.len(), 2);
    assert_eq!(result.scenario_costs[0], 210_000_000.0);
    assert_eq!(result.scenario_costs[1], 210_000_000.0);
    // Derived statistics: sum = 420_000_000, sum_sq = 210_000_000^2 * 2.
    let cost_sum: f64 = result.scenario_costs.iter().sum();
    let cost_sum_sq: f64 = result.scenario_costs.iter().map(|c| c * c).sum();
    assert_eq!(cost_sum, 420_000_000.0);
    assert_eq!(cost_sum_sq, 210_000_000.0_f64.powi(2) * 2.0);
}

// ── Unit tests: SyncResult ───────────────────────────────────────────────

#[test]
fn sync_result_field_access() {
    let r = SyncResult {
        global_ub_mean: 75.0,
        global_ub_std: 12.909,
        ci_95_half_width: 12.651,
        sync_time_ms: 7,
    };
    assert_eq!(r.global_ub_mean, 75.0);
    assert_eq!(r.global_ub_std, 12.909);
    assert_eq!(r.ci_95_half_width, 12.651);
    assert_eq!(r.sync_time_ms, 7);
}

#[test]
fn sync_result_clone_and_debug() {
    let r = SyncResult {
        global_ub_mean: 2.0,
        global_ub_std: 3.0,
        ci_95_half_width: 4.0,
        sync_time_ms: 5,
    };
    let c = r.clone();
    assert_eq!(c.global_ub_mean, r.global_ub_mean);
    assert_eq!(c.global_ub_std, r.global_ub_std);
    let s = format!("{r:?}");
    assert!(s.contains("SyncResult"));
}

// ── Unit tests: UB statistics computation ───────────────────────────────

/// 4 scenarios with costs [60, 70, 80, 90].
///
/// `cost_sum` = 300, `cost_sum_sq` = 60²+70²+80²+90² = 23000, count = 4.
/// mean = 75.0
/// variance = (23000 - 4 * 75^2) / 3 = (23000 - 22500) / 3 = 500/3
/// std = sqrt(500/3) ≈ 12.910
/// `ci_95` = 1.96 * std / sqrt(4)
#[test]
fn ub_statistics_four_scenarios_correct_mean_and_std() {
    let local = ForwardResult {
        scenario_costs: vec![60.0, 70.0, 80.0, 90.0],
        elapsed_ms: 0,
        lp_solves: 0,
        setup_time_ms: 0,
        load_imbalance_ms: 0,
        scheduling_overhead_ms: 0,
        stage_stats: Vec::new(),
    };
    let comm = LocalBackend;
    let result = sync_forward(&local, &comm, 4, ForwardBound::Statistical).unwrap();

    assert_eq!(result.global_ub_mean, 75.0, "mean must be 300/4 = 75");

    let expected_std = (500.0_f64 / 3.0).sqrt();
    let tolerance = 1e-9;
    assert!(
        (result.global_ub_std - expected_std).abs() < tolerance,
        "std deviation {got} should be ≈ {expected_std}",
        got = result.global_ub_std,
    );

    let expected_ci = 1.96_f64 * expected_std / 4.0_f64.sqrt();
    assert!(
        (result.ci_95_half_width - expected_ci).abs() < tolerance,
        "ci_95 {got} should be ≈ {expected_ci}",
        got = result.ci_95_half_width,
    );
}

/// 4 scenarios, costs [60,70,80,90].
///
/// Matches the exact acceptance criterion values: `global_ub_mean` = 75.0,
/// `global_ub_std` > 0.
///
/// The sequential summation of [60, 70, 80, 90] gives:
/// `cost_sum` = 300, `cost_sum_sq` = 23000, N = 4, mean = 75.
/// std = sqrt((23000 - 4*75^2) / 3) = sqrt(500/3) ≈ 12.910.
#[test]
fn acceptance_criterion_ub_mean() {
    let local = ForwardResult {
        scenario_costs: vec![60.0, 70.0, 80.0, 90.0],
        elapsed_ms: 0,
        lp_solves: 0,
        setup_time_ms: 0,
        load_imbalance_ms: 0,
        scheduling_overhead_ms: 0,
        stage_stats: Vec::new(),
    };
    let comm = LocalBackend;
    let result = sync_forward(&local, &comm, 4, ForwardBound::Statistical).unwrap();

    assert_eq!(result.global_ub_mean, 75.0);
    let expected_std = (500.0_f64 / 3.0).sqrt();
    assert!(
        (result.global_ub_std - expected_std).abs() < 1e-9,
        "std deviation {got} should be ≈ {expected_std}",
        got = result.global_ub_std,
    );
}

/// Canonical summation: [1.0, 2.0, 3.0, 4.0] produces identical mean/std
/// regardless of how the vector is split across "ranks".
///
/// Verifies that the `allgatherv` + sequential summation approach produces
/// bit-identical statistics for the full vector `[1, 2, 3, 4]` whether it
/// is presented as a single rank with 4 scenarios or simulated as two
/// ranks with 2 scenarios each.
#[test]
fn canonical_summation_identical_regardless_of_partition() {
    let single_rank = ForwardResult {
        scenario_costs: vec![1.0, 2.0, 3.0, 4.0],
        elapsed_ms: 0,
        lp_solves: 0,
        setup_time_ms: 0,
        load_imbalance_ms: 0,
        scheduling_overhead_ms: 0,
        stage_stats: Vec::new(),
    };
    let comm = LocalBackend;
    let result_single = sync_forward(&single_rank, &comm, 4, ForwardBound::Statistical).unwrap();

    // Build the full global buffer manually; sequential summation must yield
    // the same statistics as the single-rank result above.
    let global_costs = [1.0_f64, 2.0, 3.0, 4.0];
    let global_n = global_costs.len();
    #[allow(clippy::cast_precision_loss)]
    let global_n_f64 = global_n as f64;
    let cost_sum: f64 = global_costs.iter().sum();
    let cost_sum_sq: f64 = global_costs.iter().map(|c| c * c).sum();
    let mean = cost_sum / global_n_f64;
    let variance = (cost_sum_sq - global_n_f64 * mean * mean) / (global_n_f64 - 1.0);
    let expected_std = variance.max(0.0).sqrt();
    let expected_mean = mean;

    assert_eq!(
        result_single.global_ub_mean.to_bits(),
        expected_mean.to_bits(),
        "mean must be bit-identical to sequential summation of [1,2,3,4]"
    );
    assert_eq!(
        result_single.global_ub_std.to_bits(),
        expected_std.to_bits(),
        "std must be bit-identical to sequential summation of [1,2,3,4]"
    );
}

#[test]
fn bessel_correction_single_scenario_zero_std_and_ci() {
    let local = ForwardResult {
        scenario_costs: vec![500.0],
        elapsed_ms: 0,
        lp_solves: 0,
        setup_time_ms: 0,
        load_imbalance_ms: 0,
        scheduling_overhead_ms: 0,
        stage_stats: Vec::new(),
    };
    let comm = LocalBackend;
    let result = sync_forward(&local, &comm, 1, ForwardBound::Statistical).unwrap();

    assert_eq!(
        result.global_ub_std, 0.0,
        "std must be 0.0 for a single scenario (N=1 Bessel correction)"
    );
    assert_eq!(
        result.ci_95_half_width, 0.0,
        "ci_95 must be 0.0 for a single scenario"
    );
}

/// Guard: negative variance from floating-point cancellation → std = 0.0, not NaN.
#[test]
fn negative_variance_guard_produces_zero_std_not_nan() {
    // Two identical large values: the single-pass Bessel formula
    // (sum_sq - N*mean^2)/(N-1) can yield a tiny negative variance from
    // floating-point rounding, which the max(0, .).sqrt() guard must clamp.
    let v = 1.0e15_f64;
    let local = ForwardResult {
        scenario_costs: vec![v, v],
        elapsed_ms: 0,
        lp_solves: 0,
        setup_time_ms: 0,
        load_imbalance_ms: 0,
        scheduling_overhead_ms: 0,
        stage_stats: Vec::new(),
    };
    let comm = LocalBackend;
    let result = sync_forward(&local, &comm, 2, ForwardBound::Statistical).unwrap();

    assert!(
        !result.global_ub_std.is_nan(),
        "std must not be NaN even when floating-point variance is slightly negative"
    );
    // Both costs are exactly equal so true variance = 0.
    // The max(0, variance).sqrt() guard must clamp any tiny negative value.
    assert_eq!(
        result.global_ub_std, 0.0,
        "std must be 0.0 when variance is zero (or clamps from tiny negative)"
    );
}

// ── Integration tests: sync_forward with LocalBackend ────────────────────

#[test]
fn sync_forward_local_backend_global_equals_local() {
    // Two scenarios each costing 420.0 → mean = 420.0.
    let local = ForwardResult {
        scenario_costs: vec![420.0, 420.0],
        elapsed_ms: 5,
        lp_solves: 0,
        setup_time_ms: 0,
        load_imbalance_ms: 0,
        scheduling_overhead_ms: 0,
        stage_stats: Vec::new(),
    };
    let comm = LocalBackend;
    let result = sync_forward(&local, &comm, 2, ForwardBound::Statistical).unwrap();

    // In single-rank mode, allgatherv is an identity copy.
    assert_eq!(
        result.global_ub_mean, 420.0,
        "global_ub_mean must equal the arithmetic mean of the cost vector"
    );
}

#[test]
fn sync_forward_sync_time_ms_is_valid_u64() {
    let local = ForwardResult {
        scenario_costs: vec![50.0, 50.0],
        elapsed_ms: 0,
        lp_solves: 0,
        setup_time_ms: 0,
        load_imbalance_ms: 0,
        scheduling_overhead_ms: 0,
        stage_stats: Vec::new(),
    };
    let comm = LocalBackend;
    let result = sync_forward(&local, &comm, 2, ForwardBound::Statistical).unwrap();
    // sync_time_ms is u64 — any value is a valid non-negative u64.
    // We just verify the field exists and doesn't overflow to something absurd.
    let _ = result.sync_time_ms;
}

#[test]
fn sync_forward_comm_error_wraps_as_sddp_communication() {
    use cobre_comm::CommError;

    /// Communicator that always returns `CommError::InvalidCommunicator`.
    struct FailingComm;

    impl Communicator for FailingComm {
        fn allgatherv<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            Err(CommError::InvalidCommunicator)
        }

        fn allreduce<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            Err(CommError::InvalidCommunicator)
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            Err(CommError::InvalidCommunicator)
        }

        fn barrier(&self) -> Result<(), CommError> {
            Err(CommError::InvalidCommunicator)
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

    let local = ForwardResult {
        scenario_costs: vec![100.0],
        elapsed_ms: 0,
        lp_solves: 0,
        setup_time_ms: 0,
        load_imbalance_ms: 0,
        scheduling_overhead_ms: 0,
        stage_stats: Vec::new(),
    };
    let comm = FailingComm;
    let err = sync_forward(&local, &comm, 1, ForwardBound::Statistical).unwrap_err();

    assert!(
        matches!(err, SddpError::Communication(_)),
        "CommError must be wrapped as SddpError::Communication, got: {err:?}"
    );
}

// ── Unit tests: exact probability-weighted upper bound ───────────────────

/// `Σ wᵢ·cᵢ` on a hand-checked vector: costs [10, 20, 30] with weights
/// [0.5, 0.25, 0.25] → 5.0 + 5.0 + 7.5 = 17.5 (every product exact in f64).
#[test]
fn weighted_cost_reduction_matches_hand_computed_sum() {
    let costs = [10.0_f64, 20.0, 30.0];
    let weights = [0.5_f64, 0.25, 0.25];
    assert_eq!(weighted_cost_reduction(&costs, &weights), 17.5);
}

/// Uniform `wᵢ = 1/n` reproduces the arithmetic mean bit-for-bit: for
/// [1, 2, 3, 4] the reduction equals `10/4 = 2.5`.
#[test]
fn weighted_cost_reduction_uniform_weights_reproduce_mean() {
    let costs = [1.0_f64, 2.0, 3.0, 4.0];
    let uniform = vec![1.0_f64 / 4.0; 4];
    let mean = costs.iter().sum::<f64>() / 4.0;
    assert_eq!(
        weighted_cost_reduction(&costs, &uniform).to_bits(),
        mean.to_bits(),
        "uniform-weighted reduction must reproduce the arithmetic mean bit-for-bit"
    );
}

/// A single element returns its own `w·c`: `w = 1` yields the cost verbatim,
/// and a non-unit weight scales it — pinning that the reduction is `Σ w·c`,
/// never a weight-ignoring special case at `n == 1`.
#[test]
fn weighted_cost_reduction_single_element_is_weighted_cost() {
    assert_eq!(weighted_cost_reduction(&[42.0], &[1.0]), 42.0);
    assert_eq!(weighted_cost_reduction(&[42.0], &[0.5]), 21.0);
}

/// `sync_forward` under [`ForwardBound::Exact`] reports the probability-weighted
/// `Σ w·c` — distinct from the Welford mean the same costs would give — with the
/// standard deviation and CI half-width both zeroed.
#[test]
fn sync_forward_exact_reduces_weighted_sum_and_zeroes_ci() {
    let local = ForwardResult {
        scenario_costs: vec![100.0, 200.0],
        elapsed_ms: 0,
        lp_solves: 0,
        setup_time_ms: 0,
        load_imbalance_ms: 0,
        scheduling_overhead_ms: 0,
        stage_stats: Vec::new(),
    };
    let comm = LocalBackend;
    let weights = [0.75_f64, 0.25];
    let result = sync_forward(
        &local,
        &comm,
        2,
        ForwardBound::Exact {
            path_weights: &weights,
        },
    )
    .unwrap();

    // 0.75*100 + 0.25*200 = 125.0, whereas the sampled Welford mean would be 150.0.
    assert_eq!(result.global_ub_mean, 125.0);
    assert_eq!(result.global_ub_std, 0.0);
    assert_eq!(result.ci_95_half_width, 0.0);
}

/// The degenerate single-path enumeration (`w = 1`) reports that path's cost as
/// the exact bound with a zero CI half-width.
#[test]
fn sync_forward_exact_single_path_returns_cost_with_zero_ci() {
    let local = ForwardResult {
        scenario_costs: vec![500.0],
        elapsed_ms: 0,
        lp_solves: 0,
        setup_time_ms: 0,
        load_imbalance_ms: 0,
        scheduling_overhead_ms: 0,
        stage_stats: Vec::new(),
    };
    let comm = LocalBackend;
    let weights = [1.0_f64];
    let result = sync_forward(
        &local,
        &comm,
        1,
        ForwardBound::Exact {
            path_weights: &weights,
        },
    )
    .unwrap();

    assert_eq!(result.global_ub_mean, 500.0);
    assert_eq!(result.global_ub_std, 0.0);
    assert_eq!(result.ci_95_half_width, 0.0);
}

/// The nested backward risk recursion is the NESTED (time-consistent) `CVaR` of
/// the realized costs, strictly above the end-of-horizon `CVaR` of the whole-path
/// totals on a tree whose worst branch compounds. Fixture: a 3-stage binary tree;
/// the "bad" stage-1 branch (node 2, immediate 100) leads to the "bad-bad" leaf
/// (node 6, immediate 100); all other immediates are 0. Path totals are
/// {0, 0, 100, 200} at weight 0.25.
///
/// - Pure `CVaR_0.5` nested → `V(root) = 200`: the recursion concentrates on the
///   worst branch at every stage (the α² ≈ worst-quarter protection).
/// - End-of-horizon `CVaR_0.5` of the totals → `150` (worst-half of the totals).
/// - `Expectation` collapses to `Σ wᵢ·total = 75`, matching a plain weighted sum.
#[test]
fn nested_ub_recursion_is_nested_not_end_of_horizon() {
    use super::stats_aggregation::nested_ub_recursion;
    use crate::setup::node_graph::{NestedUbTopology, NodePos, TypedVec};

    // parent map: 0=root; 1,2 = stage-1 children of 0; 3,4 = leaves of 1; 5,6 = leaves of 2.
    let parent: TypedVec<NodePos, Option<NodePos>> = vec![
        None,
        Some(NodePos(0)),
        Some(NodePos(0)),
        Some(NodePos(1)),
        Some(NodePos(1)),
        Some(NodePos(2)),
        Some(NodePos(2)),
    ]
    .into();
    let leaf = [NodePos(3), NodePos(4), NodePos(5), NodePos(6)];
    let weight = [0.25_f64; 4];
    // path-major [c_root, c_stage1, c_leaf] per path (paths in leaf order 3,4,5,6):
    #[rustfmt::skip]
    let global = [
        0.0, 0.0, 0.0,   // path via leaf 3 (good branch)
        0.0, 0.0, 0.0,   // path via leaf 4 (good branch)
        0.0, 100.0, 0.0, // path via leaf 5 (bad stage-1, good leaf)
        0.0, 100.0, 100.0, // path via leaf 6 (bad stage-1, bad leaf)
    ];
    let cum_d = [1.0_f64, 1.0, 1.0];
    let topology = NestedUbTopology::new(&parent, &leaf, &weight);

    let cvar = RiskMeasure::CVaR {
        alpha: 0.5,
        lambda: 1.0,
    };
    let mut scratch = super::stats_aggregation::NestedUbRecursionScratch::default();
    let nested = nested_ub_recursion(&topology, &global, 3, &cum_d, cvar, &mut scratch);
    assert!(
        (nested - 200.0).abs() < 1e-12,
        "nested pure CVaR_0.5 must be 200.0, got {nested}"
    );

    // End-of-horizon CVaR of the whole-path totals is only 150 — strictly below
    // the nested 200, the exact gap the spec's negative-gap defect turned on.
    let totals = [0.0_f64, 0.0, 100.0, 200.0];
    let end_of_horizon = cvar.evaluate_risk(&totals, &weight);
    assert!(
        (end_of_horizon - 150.0).abs() < 1e-12,
        "end-of-horizon CVaR_0.5 must be 150.0, got {end_of_horizon}"
    );
    assert!(
        nested > end_of_horizon,
        "the nested bound ({nested}) must exceed the end-of-horizon bound ({end_of_horizon})"
    );

    // Expectation collapses the recursion to the plain probability-weighted total.
    let expectation = nested_ub_recursion(
        &topology,
        &global,
        3,
        &cum_d,
        RiskMeasure::Expectation,
        &mut scratch,
    );
    assert!(
        (expectation - 75.0).abs() < 1e-12,
        "Expectation must collapse to Σ wᵢ·total = 75.0, got {expectation}"
    );
}

#[test]
fn nested_ub_recursion_applies_the_probability_floor() {
    use super::stats_aggregation::{NestedUbRecursionScratch, nested_ub_recursion};
    use crate::setup::node_graph::{NestedUbTopology, NodePos, TypedVec};

    let parent: TypedVec<NodePos, Option<NodePos>> = vec![
        None,
        Some(NodePos(0)),
        Some(NodePos(0)),
        Some(NodePos(0)),
        Some(NodePos(0)),
    ]
    .into();
    let leaf = [NodePos(1), NodePos(2), NodePos(3), NodePos(4)];
    let weight = [0.25_f64; 4];
    let global = [0.0, 10.0, 0.0, 20.0, 0.0, 30.0, 0.0, 40.0];
    let cum_d = [1.0_f64, 1.0];
    let topology = NestedUbTopology::new(&parent, &leaf, &weight);

    let cvar = RiskMeasure::CVaR {
        alpha: 0.5,
        lambda: 0.5,
    };
    let nested = nested_ub_recursion(
        &topology,
        &global,
        2,
        &cum_d,
        cvar,
        &mut NestedUbRecursionScratch::default(),
    );
    assert!(
        (nested - 30.0).abs() < 1e-12,
        "the floored nested bound over the 4-leaf fan must be 30.0, got {nested}"
    );
}

// ── Unit tests: warm-start basis caching ─────────────────────────────────

/// Helper: run one iteration of `run_forward_pass` with a single scenario
/// and a 3-stage horizon. The workspace is passed mutably; the basis store
/// is returned so callers can inspect per-scenario, per-stage cached bases.
fn run_one_iteration(
    ws: &mut SolverWorkspace<MockSolver>,
    basis_store: &mut BasisStore,
) -> Result<(), crate::SddpError> {
    let state = test_support::state_layout(1, 0);
    let fcf = FutureCostFunction::new(3, state.n_state, 1, 100, &[0; 3]);
    let config = TrainingConfig {
        loop_config: LoopConfig {
            forward_passes: 1,
            training_enumerated: false,
            max_iterations: 100,
            start_iteration: 0,
            resume_lower_bound_history: Vec::new(),
            n_fwd_threads: 1,
            stopping_rules: StoppingRuleSet {
                rules: vec![StoppingRule::IterationLimit { limit: 100 }],
                mode: StoppingMode::Any,
            },
        },
        cut_management: CutManagementConfig {
            cut_selection: None,
            budget: None,
            cut_activity_tolerance: 0.0,
            risk_measures: vec![RiskMeasure::Expectation],
        },
        events: EventConfig {
            event_sender: None,
            periodic_checkpoint: None,
            shutdown_flag: None,
            export_states: false,
        },
    };

    let horizon = HorizonMode::Finite { num_stages: 3 };

    let templates = vec![
        minimal_template_1_0(),
        minimal_template_1_0(),
        minimal_template_1_0(),
    ];
    let initial_state = vec![0.0_f64; state.n_state];
    let mut records = empty_records(3);
    let stochastic = make_stochastic_context_1_hydro(3, true);
    let stages = make_stages_3();

    let state_boxes = permissive_state_boxes(state.n_state, templates.len());
    let geometry = equipment_free_geometry(&[1usize, 1, 1]);
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let ctx = fixture.ctx();
    run_forward_pass(
        std::slice::from_mut(ws),
        basis_store,
        &ctx,
        &templates,
        &fcf,
        &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(
                &state,
                horizon.num_stages(),
            ),
            study_dims: &test_support::study_dims(),
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
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &ForwardPassBatch {
            local_forward_passes: config.loop_config.forward_passes as usize,
            total_forward_passes: config.loop_config.forward_passes as usize,
            iteration: 0,
            fwd_offset: 0,
            event_sender: None,
        },
        &mut records,
    )
    .map(|_| ())
}

#[test]
fn warm_start_first_iteration_cold_second_iteration_warm() {
    let state = test_support::state_layout(1, 0);
    let solution = fixed_solution(4, 100.0, state.theta, 30.0);
    let solver = MockSolver::always_ok(solution);
    // Single workspace and a shared basis store (1 scenario × 3 stages).
    let mut ws = single_workspace(solver, &state);
    let mut basis_store = BasisStore::new(1, 3);

    run_one_iteration(&mut ws, &mut basis_store).unwrap();
    assert_eq!(
        ws.solver.inner().warm_start_calls,
        0,
        "first iteration must use cold-start for all stages (warm_start_calls == 0)"
    );

    // After first iteration, all 3 stages for scenario 0 have a cached basis.
    assert!(
        (0..3).all(|t| basis_store.get(0, NodePos(t)).is_some()),
        "basis_store must be fully populated for scenario 0 after the first iteration"
    );

    run_one_iteration(&mut ws, &mut basis_store).unwrap();
    assert!(
        ws.solver.inner().warm_start_calls > 0,
        "second iteration must use warm-start for at least one stage \
         (warm_start_calls > 0, got {})",
        ws.solver.inner().warm_start_calls
    );
}

#[test]
fn basis_invalidated_on_solver_error() {
    let state = test_support::state_layout(1, 0);
    let solution = fixed_solution(4, 100.0, state.theta, 30.0);
    // Call 4 = second iteration, stage 1 (calls 0-2 = first iteration
    // stages 0,1,2; calls 3,4,5 = second iteration stages 0,1,2).
    let solver = MockSolver::infeasible_on(solution, 4);
    // Single workspace and a shared basis store (1 scenario × 3 stages).
    let mut ws = single_workspace(solver, &state);
    let mut basis_store = BasisStore::new(1, 3);

    run_one_iteration(&mut ws, &mut basis_store).unwrap();
    assert!(
        (0..3).all(|t| basis_store.get(0, NodePos(t)).is_some()),
        "basis_store must be fully populated for scenario 0 after iteration 1"
    );

    // Second iteration: stage 0 warm-starts (call 3 OK), stage 1 infeasible (call 4).
    let err = run_one_iteration(&mut ws, &mut basis_store).unwrap_err();
    assert!(
        matches!(err, SddpError::Infeasible { stage: 1, .. }),
        "expected Infeasible at stage 1, got: {err:?}"
    );

    assert!(
        basis_store.get(0, NodePos(1)).is_none(),
        "basis_store.get(0, NodePos(1)) must be None after solver error at stage 1"
    );

    // Stage 0 succeeded in iteration 2 — its basis was re-extracted.
    assert!(
        basis_store.get(0, NodePos(0)).is_some(),
        "basis_store.get(0, NodePos(0)) must be Some (stage 0 succeeded before error)"
    );
}

// ── New test: parallel cost agreement ────────────────────────────────────

/// With 1-workspace and 4-workspace pools producing the same `cost_sum`.
///
/// Given the same input data, `run_forward_pass` with a single workspace
/// must produce identical `cost_sum` and `cost_sum_sq` values compared to
/// running with 4 workspaces. This verifies the static partitioning
/// produces deterministic results regardless of workspace count.
#[test]
#[allow(clippy::too_many_lines)]
fn test_forward_pass_parallel_cost_agreement() {
    let state = test_support::state_layout(1, 0);
    let solution = fixed_solution(4, 100.0, state.theta, 30.0);
    let stochastic = make_stochastic_context_1_hydro(3, true);
    let stages = make_stages_3();
    let fcf = FutureCostFunction::new(3, state.n_state, 2, 100, &[0; 3]);
    let horizon = HorizonMode::Finite { num_stages: 3 };
    let templates = vec![
        minimal_template_1_0(),
        minimal_template_1_0(),
        minimal_template_1_0(),
    ];
    let initial_state = vec![0.0_f64; state.n_state];
    let n_scenarios = 10;

    let state_boxes = permissive_state_boxes(state.n_state, templates.len());
    let geometry = equipment_free_geometry(&[1usize, 1, 1]);
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let ctx = fixture.ctx();

    let mut ws1 = single_workspace(MockSolver::always_ok(solution.clone()), &state);
    let mut records1 = empty_records(n_scenarios * 3);
    let mut basis_store1 = BasisStore::new(n_scenarios, templates.len());
    let result1 = run_forward_pass(
        std::slice::from_mut(&mut ws1),
        &mut basis_store1,
        &ctx,
        &templates,
        &fcf,
        &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(
                &state,
                horizon.num_stages(),
            ),
            study_dims: &test_support::study_dims(),
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
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &ForwardPassBatch {
            local_forward_passes: n_scenarios,
            total_forward_passes: n_scenarios,
            iteration: 0,
            fwd_offset: 0,
            event_sender: None,
        },
        &mut records1,
    )
    .unwrap();

    let mut workspaces4: Vec<SolverWorkspace<MockSolver>> = (0..4)
        .map(|_| single_workspace(MockSolver::always_ok(solution.clone()), &state))
        .collect();
    let mut records4 = empty_records(n_scenarios * 3);
    let mut basis_store4 = BasisStore::new(n_scenarios, templates.len());
    let result4 = run_forward_pass(
        &mut workspaces4,
        &mut basis_store4,
        &ctx,
        &templates,
        &fcf,
        &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(
                &state,
                horizon.num_stages(),
            ),
            study_dims: &test_support::study_dims(),
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
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &ForwardPassBatch {
            local_forward_passes: n_scenarios,
            total_forward_passes: n_scenarios,
            iteration: 0,
            fwd_offset: 0,
            event_sender: None,
        },
        &mut records4,
    )
    .unwrap();

    assert_eq!(
        result1.scenario_costs.len(),
        result4.scenario_costs.len(),
        "scenario_costs length must be identical for 1 and 4 workspaces"
    );
    // Each scenario cost must be bit-identical regardless of workspace count.
    for (i, (c1, c4)) in result1
        .scenario_costs
        .iter()
        .zip(result4.scenario_costs.iter())
        .enumerate()
    {
        assert_eq!(
            c1.to_bits(),
            c4.to_bits(),
            "scenario_costs[{i}] must be bit-identical: 1-workspace={c1:.17e}, 4-workspace={c4:.17e}"
        );
    }
}

// ── New test: work distribution across 4 workspaces ──────────────────────

#[allow(clippy::too_many_lines)]
#[test]
fn test_forward_pass_work_distribution() {
    let state = test_support::state_layout(1, 0);
    let solution = fixed_solution(4, 100.0, state.theta, 30.0);
    let stochastic = make_stochastic_context_1_hydro(3, true);
    let stages = make_stages_3();
    let fcf = FutureCostFunction::new(3, state.n_state, 2, 100, &[0; 3]);
    let horizon = HorizonMode::Finite { num_stages: 3 };
    let num_stages = 3usize;
    let templates = vec![
        minimal_template_1_0(),
        minimal_template_1_0(),
        minimal_template_1_0(),
    ];
    let initial_state = vec![0.0_f64; state.n_state];
    let n_scenarios = 10usize;
    let n_workers = 4usize;

    let mut workspaces: Vec<SolverWorkspace<MockSolver>> = (0..n_workers)
        .map(|_| single_workspace(MockSolver::always_ok(solution.clone()), &state))
        .collect();
    let mut records = empty_records(n_scenarios * num_stages);
    let mut basis_store = BasisStore::new(n_scenarios, num_stages);

    let state_boxes = permissive_state_boxes(state.n_state, templates.len());
    let geometry = equipment_free_geometry(&[1usize, 1, 1]);
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let ctx = fixture.ctx();
    let _result = run_forward_pass(
        &mut workspaces,
        &mut basis_store,
        &ctx,
        &templates,
        &fcf,
        &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(
                &state,
                horizon.num_stages(),
            ),
            study_dims: &test_support::study_dims(),
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
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &ForwardPassBatch {
            local_forward_passes: n_scenarios,
            total_forward_passes: n_scenarios,
            iteration: 0,
            fwd_offset: 0,
            event_sender: None,
        },
        &mut records,
    )
    .unwrap();

    // Verify each workspace performed the expected number of LP solves.
    // partition(10, 4, w): base=2, remainder=2.
    // Workers 0,1: 3 scenarios × 3 stages = 9 solves each.
    // Workers 2,3: 2 scenarios × 3 stages = 6 solves each.
    for (w, ws) in workspaces.iter().enumerate() {
        let (start_m, end_m) = partition(n_scenarios, n_workers, w);
        let assigned_scenarios = end_m - start_m;
        let expected_solves = assigned_scenarios * num_stages;

        let floor_scenarios = n_scenarios / n_workers;
        let ceil_scenarios = n_scenarios.div_ceil(n_workers);
        assert!(
            assigned_scenarios == floor_scenarios || assigned_scenarios == ceil_scenarios,
            "worker {w} assigned {assigned_scenarios} scenarios, expected {floor_scenarios} or {ceil_scenarios}"
        );

        let actual_solves = usize::try_from(ws.solver.statistics().solve_count)
            .expect("solve_count fits in usize in tests");
        assert_eq!(
            actual_solves, expected_solves,
            "worker {w} (scenarios [{start_m}, {end_m})) performed {actual_solves} solves, expected {expected_solves}"
        );
    }

    let total_solves: usize = workspaces
        .iter()
        .map(|ws| {
            usize::try_from(ws.solver.statistics().solve_count)
                .expect("solve_count fits in usize in tests")
        })
        .sum();
    assert_eq!(
        total_solves,
        n_scenarios * num_stages,
        "total solve count {total_solves} must equal n_scenarios * num_stages = {}",
        n_scenarios * num_stages
    );
}

// ── Truncation unit tests ────────────────────────────────────────────────

/// Build a `StochasticContext` for 1 hydro and 1 stage with the given
/// `mean_m3s` and `std_m3s`. Used by truncation tests.
#[allow(clippy::too_many_lines)]
fn make_stochastic_1h_1s(mean_m3s: f64, std_m3s: f64) -> StochasticContext {
    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(1),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
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
    };
    hydro.declare_mirror_unit_group(EntityId(0));
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
            branching_factor: 3,
            noise_method: NoiseMethod::Saa,
        },
    };
    let inflow_model = InflowModel {
        hydro_id: EntityId(1),
        stage_id: 0,
        mean_m3s,
        std_m3s,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    };
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
        .stages(vec![stage])
        .inflow_models(vec![inflow_model])
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

/// Helper that runs `run_forward_pass` with 1 scenario, 1 stage, and returns
/// the `z_inflow_rhs_buf` from the workspace after the call.
fn run_single_stage_forward(
    stochastic: &StochasticContext,
    inflow_method: InflowNonNegativityMethod,
) -> Vec<f64> {
    let state = test_support::state_layout(1, 0);
    let solution = fixed_solution(4, 0.0, state.theta, 0.0);
    let solver = MockSolver::always_ok(solution);
    let fcf = FutureCostFunction::new(1, state.n_state, 1, 10, &[0; 1]);
    let horizon = HorizonMode::Finite { num_stages: 1 };
    let template = minimal_template_1_0();
    let templates = vec![template];
    let initial_state = vec![0.0_f64; state.n_state];
    let mut records = empty_records(1);
    let mut ws = single_workspace(solver, &state);
    let mut basis_store = BasisStore::new(1, 1);

    let state_boxes = permissive_state_boxes(state.n_state, templates.len());
    let geometry = equipment_free_geometry(&[1usize]);
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let ctx = fixture.ctx();
    let stages = vec![Stage {
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
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    }];
    let _ = run_forward_pass(
        std::slice::from_mut(&mut ws),
        &mut basis_store,
        &ctx,
        &templates,
        &fcf,
        &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(
                &state,
                horizon.num_stages(),
            ),
            study_dims: &test_support::study_dims(),
            inflow_method: &inflow_method,
            stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &stages,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &ForwardPassBatch {
            local_forward_passes: 1,
            total_forward_passes: 1,
            iteration: 0,
            fwd_offset: 0,
            event_sender: None,
        },
        &mut records,
    )
    .unwrap();

    ws.scratch.z_inflow_rhs_buf.clone()
}

#[test]
fn truncation_clamps_negative_inflow_noise() {
    // Deterministic base = -1000 m³/s (always produces negative inflow).
    let mean_m3s = -1000.0_f64;
    let sigma = 1.0_f64;

    let stochastic = make_stochastic_1h_1s(mean_m3s, sigma);

    let z_inflow_truncation =
        run_single_stage_forward(&stochastic, InflowNonNegativityMethod::Truncation);

    assert_eq!(
        z_inflow_truncation.len(),
        1,
        "z_inflow_rhs_buf must have 1 entry"
    );
    // After truncation: z_inflow_rhs[0] = mean + sigma * eta_clamped, the
    // realized inflow in m3/s (no zeta). eta_clamped = max(eta, eta_min) where
    // eta_min = (0 - mean) / sigma = 1000, so z_inflow_rhs[0] = 0.0 exactly.
    assert!(
        z_inflow_truncation[0] >= 0.0,
        "after truncation, z_inflow_rhs[0] must be >= 0 (inflow cannot be negative), got {}",
        z_inflow_truncation[0]
    );
}

/// Truncation does not clamp when inflow is positive.
///
/// With a very large positive mean (`mean_m3s = 1000.0`) and small sigma,
/// the PAR inflow is always positive for any sampled noise. The z-inflow
/// buffer must be identical to the no-truncation path.
#[test]
fn truncation_no_clamp_when_inflow_positive() {
    let mean_m3s = 1000.0_f64;
    let sigma = 1.0_f64;

    let stochastic = make_stochastic_1h_1s(mean_m3s, sigma);

    let z_inflow_truncation =
        run_single_stage_forward(&stochastic, InflowNonNegativityMethod::Truncation);
    let z_inflow_none = run_single_stage_forward(&stochastic, InflowNonNegativityMethod::None);

    assert_eq!(z_inflow_truncation.len(), 1);
    assert_eq!(z_inflow_none.len(), 1);
    assert_eq!(
        z_inflow_truncation[0].to_bits(),
        z_inflow_none[0].to_bits(),
        "when inflow is positive, truncation must not alter the z-inflow buffer (expected identical bits)"
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn none_method_unchanged_with_truncation_code_present() {
    let state = test_support::state_layout(1, 0);
    let solution = fixed_solution(4, 100.0, state.theta, 30.0);
    let solver = MockSolver::always_ok(solution);
    let fcf = FutureCostFunction::new(3, state.n_state, 2, 100, &[0; 3]);
    let config = TrainingConfig {
        loop_config: LoopConfig {
            forward_passes: 2,
            training_enumerated: false,
            max_iterations: 100,
            start_iteration: 0,
            resume_lower_bound_history: Vec::new(),
            n_fwd_threads: 1,
            stopping_rules: StoppingRuleSet {
                rules: vec![StoppingRule::IterationLimit { limit: 100 }],
                mode: StoppingMode::Any,
            },
        },
        cut_management: CutManagementConfig {
            cut_selection: None,
            budget: None,
            cut_activity_tolerance: 0.0,
            risk_measures: vec![RiskMeasure::Expectation],
        },
        events: EventConfig {
            event_sender: None,
            periodic_checkpoint: None,
            shutdown_flag: None,
            export_states: false,
        },
    };

    let horizon = HorizonMode::Finite { num_stages: 3 };
    let templates = vec![
        minimal_template_1_0(),
        minimal_template_1_0(),
        minimal_template_1_0(),
    ];
    let initial_state = vec![0.0_f64; state.n_state];
    let mut records = empty_records(2 * 3);
    let stochastic = make_stochastic_context_1_hydro(3, true);
    let stages = make_stages_3();
    let mut ws = single_workspace(solver, &state);
    let mut basis_store =
        BasisStore::new(config.loop_config.forward_passes as usize, templates.len());

    let state_boxes = permissive_state_boxes(state.n_state, templates.len());
    let geometry = equipment_free_geometry(&[1usize, 1, 1]);
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let ctx = fixture.ctx();
    let result = run_forward_pass(
        std::slice::from_mut(&mut ws),
        &mut basis_store,
        &ctx,
        &templates,
        &fcf,
        &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(
                &state,
                horizon.num_stages(),
            ),
            study_dims: &test_support::study_dims(),
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
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &ForwardPassBatch {
            local_forward_passes: config.loop_config.forward_passes as usize,
            total_forward_passes: config.loop_config.forward_passes as usize,
            iteration: 0,
            fwd_offset: 0,
            event_sender: None,
        },
        &mut records,
    )
    .unwrap();

    assert_eq!(result.scenario_costs.len(), 2);
    for (i, record) in records.iter().enumerate() {
        assert_eq!(
            record.stage_cost, 70_000_000.0,
            "none_method: record[{i}].stage_cost should be 70_000_000.0 ((objective - theta) * COST_SCALE_FACTOR)"
        );
    }
}

// ── Load noise test helpers ─────────────────────────────────────────────

/// Build a `StochasticContext` with 1 hydro and 1 stochastic load bus over
/// a single stage.
///
/// `mean_mw` and `std_mw` control the load bus noise model.  An empty
/// correlation model is used so the two noise entities are treated as
/// independent standard normals.
#[allow(clippy::too_many_lines)]
fn make_stochastic_context_1_hydro_1_load_bus(mean_mw: f64, std_mw: f64) -> StochasticContext {
    let bus0 = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let bus1 = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(10),
        name: "H10".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
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
    };
    hydro.declare_mirror_unit_group(EntityId(0));
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
            branching_factor: 3,
            noise_method: NoiseMethod::Saa,
        },
    };
    let inflow_model = InflowModel {
        hydro_id: EntityId(10),
        stage_id: 0,
        mean_m3s: 100.0,
        std_m3s: 20.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    };
    let load_model = LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw,
        std_mw,
    };
    let correlation = CorrelationModel {
        method: "spectral".to_string(),
        profiles: std::collections::BTreeMap::new(),
        schedule: vec![],
    };
    let system = SystemBuilder::new()
        .buses(vec![bus0, bus1])
        .hydros(vec![hydro])
        .stages(vec![stage])
        .inflow_models(vec![inflow_model])
        .load_models(vec![load_model])
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

// ── New test: parallel infeasibility propagation ──────────────────────────

#[test]
fn test_forward_pass_parallel_infeasibility() {
    let state = test_support::state_layout(1, 0);
    let solution = fixed_solution(4, 100.0, state.theta, 30.0);
    let stochastic = make_stochastic_context_1_hydro(3, true);
    let stages = make_stages_3();
    let fcf = FutureCostFunction::new(3, state.n_state, 2, 100, &[0; 3]);
    let horizon = HorizonMode::Finite { num_stages: 3 };
    let num_stages = 3usize;
    let templates = vec![
        minimal_template_1_0(),
        minimal_template_1_0(),
        minimal_template_1_0(),
    ];
    let initial_state = vec![0.0_f64; state.n_state];
    let n_scenarios = 10usize;
    let n_workers = 4usize;

    let mut workspaces: Vec<SolverWorkspace<MockSolver>> = (0..n_workers)
        .map(|w| {
            let solver = if w == 1 {
                // Fail on the first solve call this worker makes.
                MockSolver::infeasible_on(solution.clone(), 0)
            } else {
                MockSolver::always_ok(solution.clone())
            };
            single_workspace(solver, &state)
        })
        .collect();

    let mut records = empty_records(n_scenarios * num_stages);
    let mut basis_store = BasisStore::new(n_scenarios, num_stages);

    let state_boxes = permissive_state_boxes(state.n_state, templates.len());
    let geometry = equipment_free_geometry(&[1usize, 1, 1]);
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let ctx = fixture.ctx();
    let result = run_forward_pass(
        &mut workspaces,
        &mut basis_store,
        &ctx,
        &templates,
        &fcf,
        &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(
                &state,
                horizon.num_stages(),
            ),
            study_dims: &test_support::study_dims(),
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
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &ForwardPassBatch {
            local_forward_passes: n_scenarios,
            total_forward_passes: n_scenarios,
            iteration: 0,
            fwd_offset: 0,
            event_sender: None,
        },
        &mut records,
    );

    match result {
        Err(SddpError::Infeasible {
            stage,
            scenario,
            iteration,
        }) => {
            assert_eq!(
                stage, 0,
                "infeasible stage must be 0 (first stage of worker 1)"
            );
            assert_eq!(
                scenario, 3,
                "infeasible scenario must be 3 (start_m of worker 1)"
            );
            assert_eq!(
                iteration, 0,
                "iteration must be 0 (first training iteration)"
            );
        }
        Err(other) => panic!("expected SddpError::Infeasible, got: {other:?}"),
        Ok(_) => panic!("expected Err(SddpError::Infeasible), got Ok"),
    }
}

// ── Load noise wiring tests ──────────────────────────────────────────────

/// Verify that the load balance row is patched to a positive value when
/// `mean_mw + std_mw * eta > 0`.
///
/// With `mean_mw = 300.0` and `std_mw = 30.0`, any standard-normal draw `eta`
/// satisfying `|eta| <= 10` produces a positive realization, which holds with
/// overwhelming probability.  After a single-scenario forward pass the
/// `load_rhs_buf` must contain a positive value equal to
/// `max(0, 300 + 30 * eta) * block_factor`.  Since no load factors file is
/// supplied, `block_factor = 1.0`, so `load_rhs_buf[0] = max(0, 300 + 30 * eta)`.
#[test]
#[allow(clippy::too_many_lines)]
fn forward_pass_load_noise_positive_realization() {
    let n_load_buses = 1usize;
    let stochastic = make_stochastic_context_1_hydro_1_load_bus(300.0, 30.0);
    let state = test_support::state_layout(1, 0);
    let load_bus_indices = vec![0usize];
    let geometry_per_stage = vec![test_support::geometry_with_load_balance(10, 1, 1)];
    let patch_buf = PatchBuffer::new(&state, &load_bus_indices, &geometry_per_stage);
    let mut ws = SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(MockSolver::always_ok(fixed_solution(
            4,
            100.0,
            state.theta,
            30.0,
        ))),
        patch_buf,
        current_state: Vec::with_capacity(state.n_state),
        scratch: ScratchBuffers {
            inflow_m3s_buf: Vec::with_capacity(1),
            lag_matrix_buf: Vec::with_capacity(0),
            par_inflow_buf: Vec::with_capacity(1),
            eta_floor_buf: Vec::with_capacity(1),
            zero_targets_buf: vec![0.0_f64; 1],
            ncs_col_upper_buf: Vec::new(),
            ncs_col_lower_buf: Vec::new(),
            ncs_col_indices_buf: Vec::new(),
            ncs_col_lower_active_buf: Vec::new(),
            ncs_col_upper_active_buf: Vec::new(),
            last_ncs_col_start: usize::MAX,
            ncs_col_upper_extract_buf: Vec::new(),
            load_rhs_buf: Vec::with_capacity(n_load_buses),
            row_lower_buf: Vec::new(),
            z_inflow_rhs_buf: Vec::new(),
            effective_eta_buf: Vec::new(),
            unscaled_primal: Vec::new(),
            unscaled_dual: Vec::new(),
            lag_accumulator: vec![],
            lag_weight_accum: vec![],
            downstream_accumulator: Vec::new(),
            downstream_weight_accum: 0.0,
            downstream_completed_lags: Vec::new(),
            downstream_n_completed: 0,
            recon_slot_lookup: Vec::new(),
            trajectory_costs_buf: Vec::new(),
            raw_noise_buf: Vec::new(),
            corr_scratch: Vec::new(),
            current_node_buf: Vec::new(),
        },
        scratch_basis: Basis::new(0, 0),
        backward_accum: BackwardAccumulators::default(),
        worker_timing_buf: WorkerPhaseTimings::default(),
    };

    let templates = vec![minimal_template_1_0()];
    let initial_state = vec![0.0_f64; state.n_state];
    let mut records = empty_records(1);
    let fcf = FutureCostFunction::new(1, state.n_state, 1, 10, &[0; 1]);
    let horizon = HorizonMode::Finite { num_stages: 1 };
    let mut basis_store = BasisStore::new(1, 1);

    let state_boxes = permissive_state_boxes(state.n_state, templates.len());
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry_per_stage)
        .load_bus_indices(&load_bus_indices);
    let ctx = fixture.ctx();
    let _fwd = run_forward_pass(
        std::slice::from_mut(&mut ws),
        &mut basis_store,
        &ctx,
        &templates,
        &fcf,
        &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(
                &state,
                horizon.num_stages(),
            ),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &ForwardPassBatch {
            local_forward_passes: 1,
            total_forward_passes: 1,
            iteration: 0,
            fwd_offset: 0,
            event_sender: None,
        },
        &mut records,
    )
    .unwrap();

    assert_eq!(
        ws.scratch.load_rhs_buf.len(),
        n_load_buses,
        "load_rhs_buf must have 1 entry (1 load bus x 1 block)"
    );
    assert!(
        ws.scratch.load_rhs_buf[0] > 0.0,
        "load realization must be positive with mean=300, std=30: got {}",
        ws.scratch.load_rhs_buf[0]
    );

    let load_start = 0;
    assert_eq!(
        ws.patch_buf.lower[load_start], ws.scratch.load_rhs_buf[0],
        "patch_buf lower must equal load_rhs_buf[0]"
    );
    assert_eq!(
        ws.patch_buf.upper[load_start], ws.scratch.load_rhs_buf[0],
        "patch_buf upper must equal load_rhs_buf[0] (equality constraint)"
    );
    assert_eq!(
        ws.patch_buf.indices[load_start], 10,
        "patch index must be geometry_per_stage[0].load_balance.start() + 0 * n_blks"
    );
}

/// Verify that a load realization that would be negative is clamped to zero.
///
/// With `mean_mw = -1000.0` and `std_mw = 1.0`, any standard-normal draw
/// (bounded to roughly +-5 in practice) produces `mean + std * eta ~= -1000`,
/// which must be clamped to `0.0` before block factor scaling.
#[test]
#[allow(clippy::too_many_lines)]
fn forward_pass_load_noise_clamped_to_zero() {
    let n_load_buses = 1usize;
    let stochastic = make_stochastic_context_1_hydro_1_load_bus(-1000.0, 1.0);
    let state = test_support::state_layout(1, 0);
    let load_bus_indices = vec![0usize];
    let geometry_per_stage = vec![test_support::geometry_with_load_balance(10, 1, 1)];
    let patch_buf = PatchBuffer::new(&state, &load_bus_indices, &geometry_per_stage);
    let mut ws = SolverWorkspace {
        rank: 0,
        worker_id: 0,
        solver: ProfiledSolver::new(MockSolver::always_ok(fixed_solution(
            4,
            100.0,
            state.theta,
            30.0,
        ))),
        patch_buf,
        current_state: Vec::with_capacity(state.n_state),
        scratch: ScratchBuffers {
            inflow_m3s_buf: Vec::with_capacity(1),
            lag_matrix_buf: Vec::with_capacity(0),
            par_inflow_buf: Vec::with_capacity(1),
            eta_floor_buf: Vec::with_capacity(1),
            zero_targets_buf: vec![0.0_f64; 1],
            ncs_col_upper_buf: Vec::new(),
            ncs_col_lower_buf: Vec::new(),
            ncs_col_indices_buf: Vec::new(),
            ncs_col_lower_active_buf: Vec::new(),
            ncs_col_upper_active_buf: Vec::new(),
            last_ncs_col_start: usize::MAX,
            ncs_col_upper_extract_buf: Vec::new(),
            load_rhs_buf: Vec::with_capacity(n_load_buses),
            row_lower_buf: Vec::new(),
            z_inflow_rhs_buf: Vec::new(),
            effective_eta_buf: Vec::new(),
            unscaled_primal: Vec::new(),
            unscaled_dual: Vec::new(),
            lag_accumulator: vec![],
            lag_weight_accum: vec![],
            downstream_accumulator: Vec::new(),
            downstream_weight_accum: 0.0,
            downstream_completed_lags: Vec::new(),
            downstream_n_completed: 0,
            recon_slot_lookup: Vec::new(),
            trajectory_costs_buf: Vec::new(),
            raw_noise_buf: Vec::new(),
            corr_scratch: Vec::new(),
            current_node_buf: Vec::new(),
        },
        scratch_basis: Basis::new(0, 0),
        backward_accum: BackwardAccumulators::default(),
        worker_timing_buf: WorkerPhaseTimings::default(),
    };

    let templates = vec![minimal_template_1_0()];
    let initial_state = vec![0.0_f64; state.n_state];
    let mut records = empty_records(1);
    let fcf = FutureCostFunction::new(1, state.n_state, 1, 10, &[0; 1]);
    let horizon = HorizonMode::Finite { num_stages: 1 };
    let mut basis_store = BasisStore::new(1, 1);

    let state_boxes = permissive_state_boxes(state.n_state, templates.len());
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry_per_stage)
        .load_bus_indices(&load_bus_indices);
    let ctx = fixture.ctx();
    let _fwd = run_forward_pass(
        std::slice::from_mut(&mut ws),
        &mut basis_store,
        &ctx,
        &templates,
        &fcf,
        &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(
                &state,
                horizon.num_stages(),
            ),
            study_dims: &test_support::study_dims(),
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &ForwardPassBatch {
            local_forward_passes: 1,
            total_forward_passes: 1,
            iteration: 0,
            fwd_offset: 0,
            event_sender: None,
        },
        &mut records,
    )
    .unwrap();

    assert_eq!(
        ws.scratch.load_rhs_buf.len(),
        n_load_buses,
        "load_rhs_buf must have 1 entry (1 load bus x 1 block)"
    );
    assert_eq!(
        ws.scratch.load_rhs_buf[0], 0.0,
        "realization with mean=-1000 must be clamped to 0.0, got {}",
        ws.scratch.load_rhs_buf[0]
    );

    let load_start = 0;
    assert_eq!(
        ws.patch_buf.lower[load_start], 0.0,
        "patch lower must be 0.0 (clamped)"
    );
    assert_eq!(
        ws.patch_buf.upper[load_start], 0.0,
        "patch upper must be 0.0 (clamped)"
    );
}

#[test]
fn forward_pass_no_load_buses_unchanged() {
    let stochastic = make_stochastic_context_1_hydro(3, true);
    let stages = make_stages_3();
    let state = test_support::state_layout(1, 0);
    let solution = fixed_solution(4, 100.0, state.theta, 30.0);
    let mut ws = single_workspace(MockSolver::always_ok(solution), &state);

    let templates = vec![
        minimal_template_1_0(),
        minimal_template_1_0(),
        minimal_template_1_0(),
    ];
    let initial_state = vec![0.0_f64; state.n_state];
    let mut records = empty_records(3);
    let fcf = FutureCostFunction::new(3, state.n_state, 1, 10, &[0; 3]);
    let horizon = HorizonMode::Finite { num_stages: 3 };
    let mut basis_store = BasisStore::new(1, 3);

    let state_boxes = permissive_state_boxes(state.n_state, templates.len());
    let geometry = equipment_free_geometry(&[1, 1, 1]);
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry);
    let ctx = fixture.ctx();
    let _fwd = run_forward_pass(
        std::slice::from_mut(&mut ws),
        &mut basis_store,
        &ctx,
        &templates,
        &fcf,
        &TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(
                &state,
                horizon.num_stages(),
            ),
            study_dims: &test_support::study_dims(),
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
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &ForwardPassBatch {
            local_forward_passes: 1,
            total_forward_passes: 1,
            iteration: 0,
            fwd_offset: 0,
            event_sender: None,
        },
        &mut records,
    )
    .unwrap();

    // With n_load_buses=0, active_load_patches stays 0; the sole patch is the
    // one z-inflow row `minimal_template_1_0`'s single hydro implies.
    assert_eq!(
        ws.patch_buf.forward_patch_count(),
        1,
        "forward_patch_count must equal the single hydro's z-inflow row when \
         n_load_buses=0, got {}",
        ws.patch_buf.forward_patch_count()
    );
    assert!(
        ws.scratch.load_rhs_buf.is_empty(),
        "load_rhs_buf must be empty when n_load_buses=0"
    );
}

// ── Tests for build_delta_cut_row_batch_into ─────────────────────────

fn empty_delta_batch() -> RowBatch {
    RowBatch {
        num_rows: 0,
        row_starts: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    }
}

#[test]
fn test_build_delta_empty_pool() {
    let fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
    let state = test_support::state_layout(1, 0);
    let mut batch = empty_delta_batch();

    build_delta_cut_row_batch_into(
        &mut batch,
        &fcf,
        0,
        &state,
        &test_support::cut_state_projection(&state),
        &[],
        1,
    );

    assert_eq!(batch.num_rows, 0);
    assert_eq!(batch.row_starts, vec![0_i32]);
    assert!(batch.col_indices.is_empty());
    assert!(batch.values.is_empty());
    assert!(batch.row_lower.is_empty());
    assert!(batch.row_upper.is_empty());
}

#[test]
fn test_build_delta_single_iteration_filter() {
    // Pool has cuts at iterations 1, 2, 3; calling with current_iteration=2
    // emits only the iteration-2 cut.
    let mut fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
    // iteration=1, fwd_idx=0: slot = 0 + 1*1 + 0 = 1
    fcf.add_cut(NodeId(0), 0, 1, 0, 10.0, &[1.0]);
    // iteration=2, fwd_idx=0: slot = 0 + 2*1 + 0 = 2
    fcf.add_cut(NodeId(0), 0, 2, 0, 20.0, &[2.0]);
    // iteration=3, fwd_idx=0: slot = 0 + 3*1 + 0 = 3
    fcf.add_cut(NodeId(0), 0, 3, 0, 30.0, &[3.0]);

    let state = test_support::state_layout(1, 0);
    let mut batch = empty_delta_batch();

    build_delta_cut_row_batch_into(
        &mut batch,
        &fcf,
        0,
        &state,
        &test_support::cut_state_projection(&state),
        &[],
        2,
    );

    assert_eq!(batch.num_rows, 1);
    assert_eq!(batch.row_lower, vec![20.0]);
    assert_eq!(batch.row_starts, vec![0_i32, 2_i32]);
    // The cut emitted must carry iteration-2's coefficient (-2.0).
    assert_eq!(batch.values[0], -2.0);
}

#[test]
fn test_build_delta_skips_deactivated_cuts() {
    // Pool has cuts at iteration 1, some deactivated; only active
    // iteration-1 cuts are emitted.
    let mut fcf = FutureCostFunction::new(2, 1, 2, 10, &[0; 2]);
    // iteration=1, fwd_idx=0: slot = 0 + 1*2 + 0 = 2
    fcf.add_cut(NodeId(0), 0, 1, 0, 10.0, &[1.0]);
    // iteration=1, fwd_idx=1: slot = 0 + 1*2 + 1 = 3
    fcf.add_cut(NodeId(0), 0, 1, 1, 20.0, &[2.0]);

    // Deactivate slot 2 (the first iteration-1 cut).
    fcf.pools[0].deactivate(&[2]);

    let state = test_support::state_layout(1, 0);
    let mut batch = empty_delta_batch();

    build_delta_cut_row_batch_into(
        &mut batch,
        &fcf,
        0,
        &state,
        &test_support::cut_state_projection(&state),
        &[],
        1,
    );

    // Only slot 3 (intercept=20.0) should appear.
    assert_eq!(batch.num_rows, 1);
    assert_eq!(batch.row_lower, vec![20.0]);
}

#[test]
fn test_build_delta_excludes_warm_start_cuts() {
    // Pool seeded with a warm-start cut AND one training iteration cut.
    // Delta call with current_iteration=1 must exclude the warm-start row.
    use cobre_io::OwnedPolicyCutRecord;

    let warm_record = OwnedPolicyCutRecord {
        cut_id: 0,
        slot_index: 0,
        coefficients: vec![5.0],
        intercept: 99.0,
        iteration: 0,
        forward_pass_index: 0,
        is_active: true,
    };
    let mut pool = CutPool::new_with_warm_start(1, 2, 10, &[warm_record]);
    // Now add a training cut at iteration=1, fwd_idx=0:
    // slot = warm_start_count(1) + 1*2 + 0 = 3
    pool.add_cut(NodeId(0), 1, 0, 7.0, &[1.0]);

    // Build an FCF with 2 stages (n_state=1, 2 fwd passes, 10 max iters).
    let mut fcf = FutureCostFunction::new(2, 1, 2, 10, &[0; 2]);
    fcf.pools[0] = pool;

    let state = test_support::state_layout(1, 0);
    let mut batch = empty_delta_batch();

    build_delta_cut_row_batch_into(
        &mut batch,
        &fcf,
        0,
        &state,
        &test_support::cut_state_projection(&state),
        &[],
        1,
    );

    // Warm-start cut (intercept=99.0) must be excluded; training cut
    // (intercept=7.0) must be present.
    assert_eq!(batch.num_rows, 1);
    assert_eq!(batch.row_lower, vec![7.0]);
}

#[test]
fn test_build_delta_matches_full_batch_when_pool_has_only_current_iter() {
    // When the pool contains only cuts from current_iteration, delta and
    // full builders must produce byte-identical output.
    let mut fcf = FutureCostFunction::new(2, 1, 2, 10, &[0; 2]);
    // iteration=1, fwd_idx=0: slot = 1*2+0 = 2
    fcf.add_cut(NodeId(0), 0, 1, 0, 10.0, &[1.0]);
    // iteration=1, fwd_idx=1: slot = 1*2+1 = 3
    fcf.add_cut(NodeId(0), 0, 1, 1, 20.0, &[3.0]);

    let state = test_support::state_layout(1, 0);

    let mut batch_full = empty_delta_batch();
    build_cut_row_batch_into(
        &mut batch_full,
        &fcf,
        0,
        &state,
        &test_support::cut_state_projection(&state),
        &[],
    );

    let mut batch_delta = empty_delta_batch();
    build_delta_cut_row_batch_into(
        &mut batch_delta,
        &fcf,
        0,
        &state,
        &test_support::cut_state_projection(&state),
        &[],
        1,
    );

    assert_eq!(batch_delta.num_rows, batch_full.num_rows);
    assert_eq!(batch_delta.row_starts, batch_full.row_starts);
    assert_eq!(batch_delta.col_indices, batch_full.col_indices);
    assert_eq!(batch_delta.values, batch_full.values);
    assert_eq!(batch_delta.row_lower, batch_full.row_lower);
    assert_eq!(batch_delta.row_upper, batch_full.row_upper);
}

#[test]
fn test_build_delta_sparse_path() {
    // n_hydro=1, n_lag=1 gives a non-empty nonzero_state_indices mask (n_state=2);
    // full sparse-path correctness is covered by build_cut_row_batch_into's own
    // tests — this only checks col_indices.len() == mask.len() + 1 per row.
    let state = test_support::state_layout(1, 1);
    let mask_len = state.nonzero_state_indices.len();

    if mask_len == 0 {
        return;
    }

    let mut fcf = FutureCostFunction::new(2, state.n_state, 1, 10, &[0; 2]);
    fcf.add_cut(NodeId(0), 0, 1, 0, 5.0, &vec![1.0; state.n_state]);

    let mut batch = empty_delta_batch();
    build_delta_cut_row_batch_into(
        &mut batch,
        &fcf,
        0,
        &state,
        &test_support::cut_state_projection(&state),
        &[],
        1,
    );

    assert_eq!(batch.num_rows, 1);
    // Each row: mask_len state entries + 1 theta entry.
    assert_eq!(batch.col_indices.len(), mask_len + 1);
}

#[test]
fn test_build_delta_reuses_out_buffer() {
    // Call twice; second call must produce correct output even when `batch`
    // had stale data from the first call.
    let mut fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
    fcf.add_cut(NodeId(0), 0, 1, 0, 11.0, &[1.0]);
    fcf.add_cut(NodeId(0), 0, 2, 0, 22.0, &[2.0]);

    let state = test_support::state_layout(1, 0);
    let mut batch = empty_delta_batch();

    // First call: iteration 1 → should yield the iteration-1 cut.
    build_delta_cut_row_batch_into(
        &mut batch,
        &fcf,
        0,
        &state,
        &test_support::cut_state_projection(&state),
        &[],
        1,
    );
    assert_eq!(batch.num_rows, 1);
    assert_eq!(batch.row_lower, vec![11.0]);

    // Second call: iteration 2 → stale data from first call must be gone.
    build_delta_cut_row_batch_into(
        &mut batch,
        &fcf,
        0,
        &state,
        &test_support::cut_state_projection(&state),
        &[],
        2,
    );
    assert_eq!(batch.num_rows, 1);
    assert_eq!(batch.row_lower, vec![22.0]);
    assert_eq!(batch.row_starts.len(), 2); // [0, 2]
}

#[test]
fn test_build_delta_clears_row_starts() {
    // batch.row_starts[0] must be 0 regardless of prior state.
    let mut fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
    fcf.add_cut(NodeId(0), 0, 1, 0, 5.0, &[1.0]);

    let state = test_support::state_layout(1, 0);

    // Pre-populate batch with garbage.
    let mut batch = RowBatch {
        num_rows: 5,
        row_starts: vec![0_i32, 2, 4, 6, 8, 10],
        col_indices: vec![0_i32; 10],
        values: vec![99.0_f64; 10],
        row_lower: vec![0.0_f64; 5],
        row_upper: vec![0.0_f64; 5],
    };

    build_delta_cut_row_batch_into(
        &mut batch,
        &fcf,
        0,
        &state,
        &test_support::cut_state_projection(&state),
        &[],
        1,
    );

    assert_eq!(batch.row_starts[0], 0_i32);
    assert_eq!(batch.num_rows, 1);
    // Prior garbage must be gone.
    assert_eq!(batch.row_starts.len(), 2);
}

/// `build_delta_cut_row_batch_into` with a pool containing only warm-start
/// cuts must emit zero rows regardless of the requested iteration.
#[test]
fn build_delta_cut_row_batch_into_skips_warm_start_slots() {
    use cobre_io::OwnedPolicyCutRecord;

    // One warm-start cut at slot 0.
    let ws_record = OwnedPolicyCutRecord {
        cut_id: 0,
        slot_index: 0,
        coefficients: vec![1.0],
        intercept: 99.0,
        iteration: 0,
        forward_pass_index: 0,
        is_active: true,
    };
    let mut pool = CutPool::new_with_warm_start(1, 1, 10, &[ws_record]);
    // One iteration-1 cut at slot 1 (warm_start_count=1, so slot = 1+1*1+0 = 2,
    // but new_with_warm_start sets warm_start_count=1 → slot = 1+1*1+0 = 2).
    pool.add_cut(NodeId(0), 1, 0, 7.0, &[1.0]);

    let mut fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
    fcf.pools[0] = pool;

    let state = test_support::state_layout(1, 0);
    let mut batch = empty_delta_batch();

    build_delta_cut_row_batch_into(
        &mut batch,
        &fcf,
        0,
        &state,
        &test_support::cut_state_projection(&state),
        &[],
        1,
    );

    // Warm-start slot must be excluded; only the iteration-1 cut appears.
    assert_eq!(batch.num_rows, 1);
    assert_eq!(batch.row_lower[0], 7.0);
}

// -----------------------------------------------------------------------
// Forward DCS integration tests (real ActiveSolver)
// -----------------------------------------------------------------------
//
// These exercise the forward DCS branch in `run_forward_stage`: load the
// cut-free base, patch the pinned incoming state, solve the cut pool lazily,
// and extract the primal. The LP shapes mirror the backward DCS fixture
// (a coupling row ties outgoing storage col0 to the pinned incoming storage
// col2, and cuts constrain theta against col0), so the primal/objective are
// determinate at the pinned state.
mod dcs_forward {
    use cobre_core::scenario::SamplingScheme;
    use cobre_solver::{ActiveSolver, SolverInterface, StageTemplate};

    use super::super::{StageKey, run_forward_stage};
    use crate::context::TrainingContext;
    use crate::cut::FutureCostFunction;
    use crate::cut_selection::CutMetadata;
    use crate::dcs::DcsParams;
    use crate::horizon_mode::HorizonMode;

    use crate::DEFAULT_COST_SCALE_FACTOR;
    use crate::inflow_method::InflowNonNegativityMethod;
    use crate::lp::builder::{PatchBuffer, StateBox};
    use crate::setup::{NodeId, NodePos, StageIdx};
    use crate::test_support;
    use crate::test_support::{StageContextFixture, equipment_free_geometry};
    use crate::trajectory::TrajectoryRecord;
    use crate::workspace::{BasisStore, NoisePreallocation, SolverWorkspace, WorkspaceSizing};

    const X_HAT: f64 = 2.0;

    /// Cut-free base template for the N=1, L=0 state layout:
    /// cols `[storage_out=0, z_inflow=1, storage_in=2, theta=3]`. Row 0 is the
    /// z-inflow definition row (`z_inflow[0]` = rhs); row 1 is the coupling
    /// row `storage_out - storage_in = 0`. Minimise theta. The incoming
    /// state (col 2) is pinned to `x_hat` by the patch; the coupling row ties
    /// col0 to it; cuts constrain theta against col0.
    fn fwd_core_template() -> StageTemplate {
        StageTemplate {
            num_cols: 4,
            num_rows: 2,
            num_nz: 3,
            col_starts: vec![0_i32, 1, 2, 3, 3],
            row_indices: vec![1_i32, 0, 1],
            values: vec![1.0, 1.0, -1.0],
            col_lower: vec![0.0, f64::NEG_INFINITY, 0.0, -1.0e6],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY, 1.0e6],
            objective: vec![0.0, 0.0, 0.0, 1.0],
            row_lower: vec![0.0, 0.0],
            row_upper: vec![0.0, 0.0],
            n_state: 1,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// All-cuts frozen template: the cut-free base plus the three pool cuts
    /// frozen as structural rows (rows 2..5), in pool slot order:
    ///   slot 0: -0*col0 + theta >= 1 ; slot 1: -2*col0 + theta >= 0 ;
    ///   slot 2: -0*col0 + theta >= 3.
    /// Row 0 is the z-inflow definition row; row 1 is the coupling row.
    /// `num_rows = 5 = 1 (z) + template_num_rows(1) + 3 cuts`.
    fn fwd_all_cuts_frozen() -> StageTemplate {
        // CSC by column. col0 entries: coupling (row1,+1), slot1 cut (row3,-2).
        // col1 (z_inflow): row0,+1. col2: coupling (row1,-1).
        // col3 (theta): rows 2,3,4 each +1.
        StageTemplate {
            num_cols: 4,
            num_rows: 5,
            num_nz: 7,
            // col0: rows [1,3] vals [1,-2]; col1: row[0] val[1];
            // col2: row[1] val[-1]; col3: rows [2,3,4] vals [1,1,1].
            col_starts: vec![0_i32, 2, 3, 4, 7],
            row_indices: vec![1_i32, 3, 0, 1, 2, 3, 4],
            values: vec![1.0, -2.0, 1.0, -1.0, 1.0, 1.0, 1.0],
            col_lower: vec![0.0, f64::NEG_INFINITY, 0.0, -1.0e6],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY, 1.0e6],
            objective: vec![0.0, 0.0, 0.0, 1.0],
            // row0 z-inflow (=rhs); row1 coupling (=0); rows2..4 cuts (>= intercept).
            row_lower: vec![0.0, 0.0, 1.0, 0.0, 3.0],
            row_upper: vec![0.0, 0.0, f64::INFINITY, f64::INFINITY, f64::INFINITY],
            n_state: 1,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Frozen template carrying a single DOMINATING spurious cut
    /// (`-5*col0 + theta >= 0`, floor 10 at `x_hat = 2`, NOT in the pool),
    /// used to prove the DCS path loads the cut-free base and ignores this
    /// row. Row 0 is the z-inflow definition row; row 1 is the coupling row.
    fn fwd_frozen_dominating_cut() -> StageTemplate {
        StageTemplate {
            num_cols: 4,
            num_rows: 3,
            num_nz: 5,
            col_starts: vec![0_i32, 2, 3, 4, 5],
            row_indices: vec![1_i32, 2, 0, 1, 2],
            values: vec![1.0, -5.0, 1.0, -1.0, 1.0],
            col_lower: vec![0.0, f64::NEG_INFINITY, 0.0, -1.0e6],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY, 1.0e6],
            objective: vec![0.0, 0.0, 0.0, 1.0],
            row_lower: vec![0.0, 0.0, 0.0],
            row_upper: vec![0.0, 0.0, f64::INFINITY],
            n_state: 1,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Pool of three cuts on the incoming-storage state, seeded so the DCS
    /// initial set omits the binding slot 1 (stale `last_active_iter`).
    fn fwd_pool() -> FutureCostFunction {
        let mut fcf = FutureCostFunction::new(1, 1, 8, 10, &[0]);
        fcf.add_cut(NodeId(0), 0, 0, 0, 1.0, &[0.0]);
        fcf.add_cut(NodeId(0), 0, 0, 1, 0.0, &[2.0]); // binding: floor 2*x_hat = 4
        fcf.add_cut(NodeId(0), 0, 0, 2, 3.0, &[0.0]);
        let meta = |generated: u64, last: u64| CutMetadata {
            iteration_generated: generated,
            forward_pass_index: 0,
            node: NodeId(0),
            active_count: 0,
            last_active_iter: last,
        };
        fcf.pools[0].set_metadata_for_test(0, meta(1, 5));
        fcf.pools[0].set_metadata_for_test(1, meta(1, 1)); // stale → outside k2=2 window at iter 5
        fcf.pools[0].set_metadata_for_test(2, meta(1, 5));
        fcf
    }

    fn fwd_active_workspace() -> SolverWorkspace<ActiveSolver> {
        let sizing = WorkspaceSizing {
            max_openings: 1,
            initial_pool_capacity: 16,
            max_local_fwd: 1,
            noise: NoisePreallocation::StochasticDim,
        };
        let solver = ActiveSolver::new().expect("ActiveSolver::new()");
        let state = test_support::state_layout(1, 0);
        let stochastic = test_support::hydro_free_stochastic_context(1, 1);
        let node_graph = crate::test_support::chain_node_graph(&stochastic);
        let study_dims = test_support::study_dims();
        let horizon = HorizonMode::Finite { num_stages: 1 };
        let cut_state_layouts = test_support::all_enabled_cut_state_layouts(&state, 1);
        let initial_state: Vec<f64> = Vec::new();
        let training_ctx = TrainingContext {
            node_graph: &node_graph,
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
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };
        let stage_ctx_fixture = StageContextFixture::new(&[], &[], &[]);
        SolverWorkspace::new(
            0,
            0,
            solver,
            PatchBuffer::new(&state, &[], &[]),
            &training_ctx,
            &stage_ctx_fixture.ctx(),
            sizing,
        )
    }

    fn dcs_params(start_iteration: u64) -> DcsParams {
        DcsParams {
            k1: None,
            k2: 2,
            nadic: 10,
            epsilon_viol: 1e-10,
            start_iteration,
            max_inner_iterations: 50,
        }
    }

    /// Run one forward stage (stage 0 of a 2-stage horizon, so theta is not
    /// terminal-zeroed) with the given `dcs` option and `frozen` template,
    /// returning `(stage_cost, advanced_state, scoring_time_seconds)`. The
    /// cut-free base is `ctx.templates[0]`; on the frozen path the
    /// caller-equivalent `load_model(frozen)` is performed here (mirroring
    /// `run_forward_worker`).
    ///
    /// The production `run_forward_worker` filter is reproduced verbatim:
    /// `dcs.filter(|p| p.is_active(iteration))`. When `iteration <
    /// start_iteration` this collapses `dcs` to `None`, so the frozen path is
    /// taken — exactly as the worker does. The returned
    /// `scoring_time_seconds` (read from the workspace's lazy-solve scratch)
    /// is `0.0` iff the lazy path was never entered, giving callers a faithful
    /// witness for which branch ran.
    fn run_one_forward_stage(
        dcs: Option<DcsParams>,
        frozen: &StageTemplate,
        iteration: u64,
    ) -> (f64, Vec<f64>, f64) {
        // Mirror run_forward_worker's per-pass gate: DCS is `Some` only when
        // configured AND active at this iteration. This is the production
        // suppression site; reproducing it here (rather than passing `dcs`
        // straight through) is what makes the inactive-iteration assertion a
        // real witness instead of a coincidence.
        let dcs = dcs.filter(|p| p.is_active(iteration));
        let state = test_support::state_layout(1, 0);
        let core = fwd_core_template();
        let templates = vec![core.clone(), core.clone()];
        let stochastic = super::make_stochastic_context_1_hydro(2, true);
        let horizon = HorizonMode::Finite { num_stages: 2 };
        let fcf = fwd_pool();

        let mut ws = fwd_active_workspace();
        ws.current_state.clear();
        ws.current_state.push(X_HAT);
        let mut basis_store = BasisStore::new(1, 2);

        // Discount factor 0 at this stage makes stage_cost = objective * SCALE
        // = theta * SCALE (no other objective term), so the observable is
        // directly sensitive to the converged theta — letting the
        // dominating-frozen-cut test distinguish theta=4 (correct) from
        // theta=10 (a wrong frozen load).
        let discount_factors = [0.0_f64, 0.0];
        let state_boxes = vec![
            StateBox {
                lower: vec![f64::NEG_INFINITY; state.n_state],
                upper: vec![f64::INFINITY; state.n_state],
            };
            2
        ];
        let geometry = equipment_free_geometry(&[1usize, 1]);
        let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry)
            .discount_factors(&discount_factors);
        let ctx = fixture.ctx();
        let study_dims = test_support::study_dims();
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(
                &state,
                horizon.num_stages(),
            ),
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs,
        };

        // Frozen path: mirror run_forward_worker's per-scenario frozen load.
        // DCS path: run_forward_stage loads the cut-free base itself.
        if dcs.is_none() {
            ws.solver.load_model(frozen);
        }

        let mut records = vec![TrajectoryRecord {
            primal: Vec::new(),
            dual: Vec::new(),
            stage_cost: 0.0,
            node_id: NodeId(0),
            state: Vec::new(),
        }];
        let raw_noise = vec![0.0; stochastic.dim()];
        let key = StageKey {
            t: StageIdx(0),
            m: 0,
            local_m: 0,
            iteration,
            raw_noise: &raw_noise,
            basis_row_capacity: frozen.num_rows,
            pool: &fcf.pools[0],
            dcs,
            node: NodePos(0),
        };
        let mut slices = basis_store.split_workers_mut(1);
        let stage_cost = run_forward_stage(
            &mut ws,
            &mut slices[0],
            &ctx,
            &training_ctx,
            &key,
            &mut records,
        )
        .expect("forward stage solve must succeed");
        // The lazy-solve scratch accumulates scoring wall time only when
        // `lazy_solve_preloaded` runs (the DCS branch). It stays exactly
        // `0.0` on the frozen path, so it witnesses which branch executed.
        let scoring_time_seconds = ws.backward_accum.dcs_solve.scoring_time_seconds;
        (stage_cost, records[0].state.clone(), scoring_time_seconds)
    }

    /// DCS branch (binding cut omitted from the seed) yields the same
    /// stage cost and advanced state as the frozen all-cuts path within 1e-9.
    #[test]
    fn forward_dcs_exact_matches_all_cuts() {
        let all_cuts = fwd_all_cuts_frozen();
        // iteration 5 >= start_iteration 2 → DCS active; the filter is a pass-through.
        let (frozen_cost, frozen_state, frozen_scoring) = run_one_forward_stage(None, &all_cuts, 5);
        let (dcs_cost, dcs_state, dcs_scoring) =
            run_one_forward_stage(Some(dcs_params(2)), &all_cuts, 5);

        // Frozen path never scores; the active DCS path does at least one pass.
        assert_eq!(
            frozen_scoring, 0.0,
            "frozen path must not enter the lazy solve"
        );
        assert!(
            dcs_scoring > 0.0,
            "active DCS path must enter the lazy solve (scoring time accumulated)"
        );

        assert!(
            (frozen_cost - dcs_cost).abs() < 1e-9,
            "stage cost: frozen {frozen_cost} vs DCS {dcs_cost}"
        );
        // The binding cut floor is 4 at x_hat=2; minimise-theta gives theta=4.
        // With discount 0, stage_cost = objective * SCALE = theta * SCALE, so
        // both paths land on 4 * DEFAULT_COST_SCALE_FACTOR.
        assert!((dcs_cost - 4.0 * DEFAULT_COST_SCALE_FACTOR).abs() < 1e-3);
        assert_eq!(frozen_state.len(), dcs_state.len());
        for (b, d) in frozen_state.iter().zip(&dcs_state) {
            assert!((b - d).abs() < 1e-9, "state: frozen {b} vs DCS {d}");
        }
        // Sanity: advanced storage state equals the pinned x_hat (coupling
        // row forces storage_out = storage_in = x_hat).
        assert!((dcs_state[0] - X_HAT).abs() < 1e-9);
    }

    /// A frozen template with a DOMINATING embedded cut (floor 10,
    /// gradient 5, NOT in the pool) must NOT change the DCS result — proving
    /// the cut-free `ctx.templates[t]` is loaded, not `params.frozen[t]`. If
    /// the DCS path erroneously loaded the dominating frozen template, the
    /// advanced state / cost would reflect theta=10, diverging from all-cuts.
    #[test]
    fn forward_dcs_frozen_cuts_present_uses_cut_free_core() {
        let all_cuts = fwd_all_cuts_frozen();
        let dominating = fwd_frozen_dominating_cut();
        let (allcuts_cost, allcuts_state, _) = run_one_forward_stage(None, &all_cuts, 5);
        // DCS path is handed the dominating frozen template, but must ignore it
        // and load the cut-free base, recovering the all-cuts result.
        let (dcs_cost, dcs_state, _) = run_one_forward_stage(Some(dcs_params(2)), &dominating, 5);

        assert!(
            (allcuts_cost - dcs_cost).abs() < 1e-9,
            "stage cost: all-cuts {allcuts_cost} vs DCS {dcs_cost} (DCS must \
             ignore the dominating frozen cut)"
        );
        assert_eq!(allcuts_state.len(), dcs_state.len());
        for (a, d) in allcuts_state.iter().zip(&dcs_state) {
            assert!(
                (a - d).abs() < 1e-9,
                "state: all-cuts {a} vs DCS {d} (DCS must load the cut-free base)"
            );
        }
    }

    /// The `run_forward_worker` `is_active` filter actually suppresses
    /// DCS before `start_iteration` and lets it through at/after it.
    ///
    /// This is a real witness, not a coincidence: the suppression is observed
    /// directly via the `scoring_time_seconds` counter, which is `0.0` iff the
    /// lazy solve was never entered. `run_one_forward_stage` reproduces the
    /// production filter (`dcs.filter(|p| p.is_active(iteration))`) verbatim.
    ///
    /// - `start_iteration = 4`, `iteration = 1` (inactive): the filter
    ///   collapses `dcs` to `None`, so the frozen path runs — proven by
    ///   `scoring_time_seconds == 0.0` AND a bit-for-bit match with the
    ///   `dcs = None` frozen run.
    /// - `iteration = 4` (active): the filter lets DCS through — proven by
    ///   `scoring_time_seconds > 0.0`.
    ///
    /// The `DcsParams::is_active` boundary is also asserted directly so the
    /// filter's flip point is pinned independently of the stage solve.
    #[test]
    fn forward_dcs_inactive_before_start_iteration() {
        // Boundary semantics of the filter predicate itself.
        let params = dcs_params(4);
        assert!(
            !params.is_active(1),
            "iteration 1 < start_iteration 4 must be inactive"
        );
        assert!(
            params.is_active(4),
            "iteration 4 == start_iteration 4 must be active"
        );

        let all_cuts = fwd_all_cuts_frozen();

        // Inactive iteration: filter suppresses DCS → frozen path. The
        // scoring counter must be exactly 0.0 (lazy solve never entered),
        // and the result must match the dcs=None frozen run bit-for-bit.
        let (frozen_cost, frozen_state, frozen_scoring) = run_one_forward_stage(None, &all_cuts, 1);
        let (early_cost, early_state, early_scoring) =
            run_one_forward_stage(Some(dcs_params(4)), &all_cuts, 1);
        assert_eq!(
            frozen_scoring, 0.0,
            "dcs=None frozen run must not score (sanity)"
        );
        assert_eq!(
            early_scoring, 0.0,
            "iteration 1 < start_iteration 4: filter must suppress DCS, so \
             the frozen path runs and no lazy scoring occurs"
        );
        assert_eq!(frozen_cost.to_bits(), early_cost.to_bits());
        assert_eq!(frozen_state.len(), early_state.len());
        for (b, e) in frozen_state.iter().zip(&early_state) {
            assert_eq!(b.to_bits(), e.to_bits());
        }

        // Active iteration (== start_iteration): filter lets DCS through, so
        // the lazy solve runs and accumulates scoring time.
        let (_active_cost, _active_state, active_scoring) =
            run_one_forward_stage(Some(dcs_params(4)), &all_cuts, 4);
        assert!(
            active_scoring > 0.0,
            "iteration 4 >= start_iteration 4: filter must let DCS through, \
             so the lazy solve runs and scoring time accumulates"
        );
    }
}

// -----------------------------------------------------------------------
// Bucket copy-gap regression (forward pass)
// -----------------------------------------------------------------------
//
// Exercises `run_forward_stage` over a bucket-aware state layout (storage +
// one AR lag + one travel-time bucket + one anticipated-thermal slot) with a
// `MockSolver` returning a fixed primal, proving the bucket state rides the
// state-assembly plain copy: the lag-shift overwrite lands on index 1 and the
// anticipated-shift overwrite lands on index 3, never on the bucket index 2.
mod transit_bucket_copy_gap {
    use cobre_core::scenario::SamplingScheme;
    use cobre_solver::{LpSolution, SolverInterface, StageTemplate};

    use super::super::{StageKey, run_forward_stage};
    use super::MockSolver;
    use crate::context::TrainingContext;
    use crate::cut::FutureCostFunction;
    use crate::horizon_mode::HorizonMode;
    use crate::inflow_method::InflowNonNegativityMethod;
    use crate::lp::builder::{PatchBuffer, StateBox};
    use crate::lp::indexer::HydroSys;
    use crate::setup::{NodeId, NodePos, StageIdx};
    use crate::test_support;
    use crate::test_support::StageContextFixture;
    use crate::trajectory::TrajectoryRecord;
    use crate::workspace::{BasisStore, NoisePreallocation, SolverWorkspace, WorkspaceSizing};

    /// Column layout for `N=1, L=1, B=1, A=1, K_max=1`:
    /// `[storage(0), lag0(1), bucket_out(2), ant_slot0(3), z_inflow(4),
    /// storage_in(5), bucket_in(6), ant_state_in(7), theta(8)]`.
    /// `n_state = 4` (storage, lag0, `bucket_out`, `ant_slot0`) — the state
    /// region is the LP's first `n_state` columns by construction. With
    /// `k_max = 1`, `ant_slot0` is both the ring's only slot and its newest
    /// (`k_i - 1 = 0`), so it resolves by identity like `bucket_out` — no
    /// separate decision-aliased column exists anymore.
    const NUM_COLS: usize = 9;
    const TRANSIT_BUCKET_COL: usize = 2;
    const TRANSIT_BUCKET_VALUE: f64 = 777.0;
    const ANTICIPATED_SLOT_COL: usize = 3;
    const ANTICIPATED_SLOT_VALUE: f64 = 400.0;
    const Z_INFLOW_COL: usize = 4;
    const Z_INFLOW_VALUE: f64 = 55.0;

    fn transit_bucket_template() -> StageTemplate {
        StageTemplate {
            num_cols: NUM_COLS,
            num_rows: 0,
            num_nz: 0,
            col_starts: vec![0_i32; NUM_COLS + 1],
            row_indices: Vec::new(),
            values: Vec::new(),
            col_lower: vec![f64::NEG_INFINITY; NUM_COLS],
            col_upper: vec![f64::INFINITY; NUM_COLS],
            objective: vec![0.0; NUM_COLS],
            row_lower: Vec::new(),
            row_upper: Vec::new(),
            n_state: 4,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Canned primal: `storage=100`, `lag0=200` (pre-overwrite), `bucket_out=777`,
    /// `ant_slot0=400`, `z_inflow=55`, `storage_in=0`, `bucket_in=0`,
    /// `ant_state_in=0`, `theta=0`. The `MockSolver` returns this verbatim
    /// regardless of the bounds `run_forward_stage` patches.
    fn transit_bucket_solution() -> LpSolution {
        let mut primal = vec![0.0_f64; NUM_COLS];
        primal[0] = 100.0;
        primal[1] = 200.0;
        primal[TRANSIT_BUCKET_COL] = TRANSIT_BUCKET_VALUE;
        primal[ANTICIPATED_SLOT_COL] = ANTICIPATED_SLOT_VALUE;
        primal[Z_INFLOW_COL] = Z_INFLOW_VALUE;
        LpSolution {
            objective: 0.0,
            primal,
            dual: Vec::new(),
            reduced_costs: vec![0.0; NUM_COLS],
            iterations: 0,
            solve_time_seconds: 0.0,
        }
    }

    fn transit_bucket_workspace() -> SolverWorkspace<MockSolver> {
        let sizing = WorkspaceSizing {
            max_openings: 1,
            initial_pool_capacity: 16,
            max_local_fwd: 1,
            noise: NoisePreallocation::StochasticDim,
        };
        let state = test_support::state_layout_with_transit_buckets(
            1,
            1,
            vec![(HydroSys::new(0), 0)],
            vec![1],
        );
        let stochastic = super::make_stochastic_context_1_hydro(1, false);
        let node_graph = crate::test_support::chain_node_graph(&stochastic);
        let study_dims = test_support::study_dims();
        let horizon = HorizonMode::Finite { num_stages: 1 };
        let cut_state_layouts = test_support::all_enabled_cut_state_layouts(&state, 1);
        let initial_state: Vec<f64> = Vec::new();
        let training_ctx = TrainingContext {
            node_graph: &node_graph,
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
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };
        SolverWorkspace::new(
            0,
            0,
            MockSolver::always_ok(transit_bucket_solution()),
            PatchBuffer::new(&state, &[], &[]),
            &training_ctx,
            &StageContextFixture::new(&[], &[], &[]).ctx(),
            sizing,
        )
    }

    /// Run one forward stage over the bucket-aware layout, returning the
    /// captured advanced state (`records[0].state`).
    fn run_transit_bucket_forward_stage() -> Vec<f64> {
        let state = test_support::state_layout_with_transit_buckets(
            1,
            1,
            vec![(HydroSys::new(0), 0)],
            vec![1],
        );
        let template = transit_bucket_template();
        let templates = vec![template.clone()];
        let stochastic = super::make_stochastic_context_1_hydro(1, false);
        let horizon = HorizonMode::Finite { num_stages: 1 };
        let fcf = FutureCostFunction::new(1, state.n_state, 1, 1, &[0]);

        let mut ws = transit_bucket_workspace();
        ws.current_state.clear();
        ws.current_state
            .extend_from_slice(&[10.0, 20.0, 30.0, 40.0]);

        let geometry_per_stage = test_support::equipment_free_geometry(&[1]);
        let state_boxes = vec![StateBox {
            lower: vec![f64::NEG_INFINITY; state.n_state],
            upper: vec![f64::INFINITY; state.n_state],
        }];

        let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry_per_stage);
        let ctx = fixture.ctx();
        let study_dims = test_support::study_dims();
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &test_support::all_enabled_cut_state_layouts(&state, 1),
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };

        ws.solver.load_model(&template);

        let mut basis_store = BasisStore::new(1, 1);
        let mut records = vec![TrajectoryRecord {
            primal: Vec::new(),
            dual: Vec::new(),
            stage_cost: 0.0,
            node_id: NodeId(0),
            state: Vec::new(),
        }];
        let key = StageKey {
            t: StageIdx(0),
            m: 0,
            local_m: 0,
            iteration: 1,
            raw_noise: &[0.0],
            basis_row_capacity: template.num_rows,
            pool: &fcf.pools[0],
            dcs: None,
            node: NodePos(0),
        };
        let mut slices = basis_store.split_workers_mut(1);
        run_forward_stage(
            &mut ws,
            &mut slices[0],
            &ctx,
            &training_ctx,
            &key,
            &mut records,
        )
        .expect("bucket forward stage solve must succeed");

        records[0].state.clone()
    }

    /// The bucket and anticipated-ring state (`state[transit_buckets_out]`,
    /// `state[commit_out]`) ride the state-assembly plain copy: each
    /// equals its own LP primal column, untouched by the lag-shift (the only
    /// remaining state-assembly overwrite, at index 1).
    #[test]
    fn transit_bucket_state_survives_lag_and_anticipated_overwrites() {
        let advanced = run_transit_bucket_forward_stage();
        assert_eq!(
            advanced[TRANSIT_BUCKET_COL], TRANSIT_BUCKET_VALUE,
            "bucket state must equal the LP primal's bucket_out column"
        );
        assert_eq!(
            advanced[ANTICIPATED_SLOT_COL], ANTICIPATED_SLOT_VALUE,
            "anticipated-ring slot must equal the LP primal's ant_slot0 column, \
             identity-resolved like the bucket"
        );
        // The lag overwrite genuinely ran (not a vacuous no-op): the lag slot
        // picks up the accumulated z_inflow value.
        assert_eq!(
            advanced[1], Z_INFLOW_VALUE,
            "lag0 must be overwritten by the lag shift"
        );
    }
}
