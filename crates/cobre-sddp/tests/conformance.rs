//! Conformance test suite for `cobre-sddp` component contracts.
//!
//! Verifies that each abstraction point satisfies its documented contract when
//! exercised with known inputs. This file is an integration test: it uses only
//! the public `cobre_sddp::` API and reimplements all test helpers locally
//! (integration tests cannot access `#[cfg(test)]` items from the main crate).
//!
//! Test groups:
//! - [`risk_measure_conformance`] — `RiskMeasure` aggregation and evaluation.
//! - [`stopping_rule_conformance`] — `StoppingRule` / `StoppingRuleSet` semantics.
//! - [`cut_conformance`] — `CutPool` round-trip and `CutWireHeader` serialization.
//! - [`convergence_conformance`] — `ConvergenceMonitor` gap formula and history.
//! - [`lb_conformance`] — `evaluate_lower_bound` monotonicity property.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
// `..Default::default()` in the make_* Spec calls is the intentional future-field
// seam from `common::builders` — a no-op today, not dead code.
#![allow(clippy::needless_update)]

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::BlockMode;
use cobre_sddp::indexer::{BlockRowFamily, HydroSys};
use cobre_sddp::lp::builder::StageGeometry;
use cobre_sddp::{FutureCostFunction, SyncResult};
use cobre_solver::{
    Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
};

mod common;
use common::builders::{StageSpec, make_stage};

/// Single-rank stub communicator for tests.
struct LocalComm;

impl Communicator for LocalComm {
    fn allgatherv<T: CommData>(
        &self,
        _send: &[T],
        _recv: &mut [T],
        _counts: &[usize],
        _displs: &[usize],
    ) -> Result<(), CommError> {
        Ok(())
    }

    fn allreduce<T: CommData>(
        &self,
        _send: &[T],
        _recv: &mut [T],
        _op: ReduceOp,
    ) -> Result<(), CommError> {
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

/// Mock solver that returns configurable objective values in sequence.
struct MockSolver {
    objectives: Vec<f64>,
    call_count: usize,
}

impl MockSolver {
    fn with_objectives(objectives: Vec<f64>) -> Self {
        Self {
            objectives,
            call_count: 0,
        }
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
        _basis: Option<&Basis>,
    ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
        let call = self.call_count;
        self.call_count += 1;
        let obj = self.objectives[call % self.objectives.len()];
        Ok(cobre_solver::SolutionView {
            objective: obj,
            primal: &[0.0, 0.0, 0.0],
            dual: &[0.0],
            reduced_costs: &[0.0, 0.0, 0.0],
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
        "MockConformance"
    }
}

/// Minimal stage template for a hydro-free system (single `theta` column, no
/// rows): pairs with `wrap_opening_tree`'s empty-system stochastic context.
fn minimal_template() -> StageTemplate {
    StageTemplate {
        num_cols: 1,
        num_rows: 0,
        num_nz: 0,
        col_starts: vec![0, 0],
        row_indices: vec![],
        values: vec![],
        col_lower: vec![0.0],
        col_upper: vec![f64::INFINITY],
        objective: vec![1.0],
        row_lower: vec![],
        row_upper: vec![],
        n_state: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

/// Build an `OpeningTree` with `n_openings` openings at stage 0.
fn simple_opening_tree(n_openings: usize) -> cobre_stochastic::OpeningTree {
    use chrono::NaiveDate;
    use cobre_core::{
        EntityId,
        scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
        },
    };
    use cobre_stochastic::correlation::resolve::DecomposedCorrelation;
    use cobre_stochastic::generate_opening_tree;
    use cobre_stochastic::tree::OpeningTreeGenerationInputs;
    use std::collections::BTreeMap;

    let stage = make_stage(
        0,
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
    );

    let entity_id = EntityId(1);
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: vec![CorrelationEntity {
                    entity_type: "inflow".to_string(),
                    id: entity_id,
                }],
                matrix: vec![vec![1.0]],
            }],
        },
    );
    let corr_model = CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    };
    let entity_order = vec![entity_id];
    let dims = cobre_stochastic::ClassDimensions {
        n_hydros: 1,
        n_load_buses: 0,
        n_ncs: 0,
    };
    let decomposed = DecomposedCorrelation::build(&corr_model, &entity_order, dims).unwrap();

    generate_opening_tree(
        42,
        &[stage],
        &decomposed,
        &entity_order,
        dims,
        &OpeningTreeGenerationInputs::default(),
    )
    .unwrap()
}

fn make_sync_result(global_ub_mean: f64) -> SyncResult {
    SyncResult {
        global_ub_mean,
        global_ub_std: 0.0,
        ci_95_half_width: 0.0,
        sync_time_ms: 0,
    }
}

fn make_fcf(n_stages: usize, state_dimension: usize) -> cobre_sddp::FutureCostFunction {
    FutureCostFunction::new(n_stages, state_dimension, 2, 100, &vec![0; n_stages])
}

// ===========================================================================
// Test groups
// ===========================================================================

mod risk_measure_conformance {
    //! Conformance tests for `RiskMeasure` aggregation and risk evaluation.

    use cobre_sddp::risk_measure::{BackwardOutcome, RiskMeasure};

    fn outcome(intercept: f64, obj: f64, coefficients: Vec<f64>) -> BackwardOutcome {
        BackwardOutcome {
            intercept,
            coefficients,
            objective_value: obj,
        }
    }

    const RISK_MEASURE_CASES: &[(&str, f64, f64)] = &[
        ("expectation_weighted_mean", 0.0, 0.0), // alpha=0 signals Expectation variant
        ("cvar_alpha_one_equals_expectation", 1.0, 1.0),
        ("cvar_alpha_half_concentrates_on_worst", 0.5, 1.0),
        ("cvar_weights_sum_to_one", 0.3, 0.8),
    ];

    #[allow(clippy::too_many_lines)]
    #[test]
    fn risk_measure_parameter_sweep() {
        for (idx, &(desc, alpha, lambda)) in RISK_MEASURE_CASES.iter().enumerate() {
            match idx {
                0 => {
                    let outcomes = vec![
                        outcome(10.0, 10.0, vec![1.0]),
                        outcome(20.0, 20.0, vec![2.0]),
                        outcome(30.0, 30.0, vec![3.0]),
                        outcome(40.0, 40.0, vec![4.0]),
                    ];
                    let probs = vec![0.1, 0.2, 0.3, 0.4];
                    let (intercept, coeffs) =
                        RiskMeasure::Expectation.aggregate_cut(&outcomes, &probs);
                    let expected_intercept = 0.1 * 10.0 + 0.2 * 20.0 + 0.3 * 30.0 + 0.4 * 40.0;
                    assert!(
                        (intercept - expected_intercept).abs() < 1e-10,
                        "case_index = {idx}, desc = {desc}: intercept must equal weighted mean \
                         {expected_intercept}, got {intercept}"
                    );
                    let expected_coeff = 0.1 * 1.0 + 0.2 * 2.0 + 0.3 * 3.0 + 0.4 * 4.0;
                    assert_eq!(
                        coeffs.len(),
                        1,
                        "case_index = {idx}, desc = {desc}: coefficient vector must have length 1"
                    );
                    assert!(
                        (coeffs[0] - expected_coeff).abs() < 1e-10,
                        "case_index = {idx}, desc = {desc}: coefficient[0] must equal \
                         {expected_coeff}, got {}",
                        coeffs[0]
                    );
                }
                1 => {
                    let outcomes = vec![
                        outcome(10.0, 10.0, vec![1.0]),
                        outcome(20.0, 20.0, vec![2.0]),
                        outcome(30.0, 30.0, vec![3.0]),
                        outcome(40.0, 40.0, vec![4.0]),
                    ];
                    let probs = vec![0.25; 4];
                    let (int_exp, coeffs_exp) =
                        RiskMeasure::Expectation.aggregate_cut(&outcomes, &probs);
                    let (int_cvar, coeffs_cvar) =
                        RiskMeasure::CVaR { alpha, lambda }.aggregate_cut(&outcomes, &probs);
                    assert!(
                        (int_exp - int_cvar).abs() < 1e-10,
                        "case_index = {idx}, desc = {desc}: CVaR(alpha={alpha}, lambda={lambda}) \
                         intercept {int_cvar} must equal Expectation {int_exp}"
                    );
                    assert_eq!(coeffs_exp.len(), coeffs_cvar.len());
                    for (ci, (e, c)) in coeffs_exp.iter().zip(&coeffs_cvar).enumerate() {
                        assert!(
                            (e - c).abs() < 1e-10,
                            "case_index = {idx}, desc = {desc}: coefficient[{ci}]: CVaR={c} must \
                             equal Expectation={e}"
                        );
                    }
                    let costs = vec![10.0, 20.0, 30.0, 40.0];
                    let risk_exp = RiskMeasure::Expectation.evaluate_risk(&costs, &probs);
                    let risk_cvar =
                        RiskMeasure::CVaR { alpha, lambda }.evaluate_risk(&costs, &probs);
                    assert!(
                        (risk_exp - risk_cvar).abs() < 1e-10,
                        "case_index = {idx}, desc = {desc}: evaluate_risk CVaR={risk_cvar} must \
                         equal Expectation={risk_exp}"
                    );
                }
                2 => {
                    let outcomes = vec![
                        outcome(10.0, 10.0, vec![]),
                        outcome(20.0, 20.0, vec![]),
                        outcome(30.0, 30.0, vec![]),
                        outcome(40.0, 40.0, vec![]),
                    ];
                    let probs = vec![0.25; 4];
                    let rm = RiskMeasure::CVaR { alpha, lambda };
                    let (intercept, _) = rm.aggregate_cut(&outcomes, &probs);
                    // Per-scenario upper bound = p / alpha = 0.25 / 0.5 = 0.5.
                    // Greedy fills worst two (40, 30) → 0.5*40 + 0.5*30 = 35.0.
                    let expected = 35.0_f64;
                    assert!(
                        (intercept - expected).abs() < 1e-10,
                        "case_index = {idx}, desc = {desc}: CVaR(alpha={alpha}, lambda={lambda}) \
                         must concentrate on worst 50%: expected {expected}, got {intercept}"
                    );
                }
                3 => {
                    let obj_values = [15.0_f64, 5.0, 25.0, 35.0];
                    let probs = vec![0.3, 0.2, 0.3, 0.2];
                    let rm = RiskMeasure::CVaR { alpha, lambda };
                    // If weights sum to 1, aggregating unit intercepts yields intercept=1.
                    let unit_outcomes: Vec<BackwardOutcome> = obj_values
                        .iter()
                        .map(|&obj| BackwardOutcome {
                            intercept: 1.0,
                            coefficients: vec![1.0],
                            objective_value: obj,
                        })
                        .collect();
                    let (intercept, coeffs) = rm.aggregate_cut(&unit_outcomes, &probs);
                    assert!(
                        (intercept - 1.0).abs() < 1e-10,
                        "case_index = {idx}, desc = {desc}: CVaR(alpha={alpha}, lambda={lambda}) \
                         weights must sum to 1.0 (intercept check): got {intercept}"
                    );
                    assert!(
                        (coeffs[0] - 1.0).abs() < 1e-10,
                        "case_index = {idx}, desc = {desc}: CVaR(alpha={alpha}, lambda={lambda}) \
                         weights must sum to 1.0 (coeff check): got {}",
                        coeffs[0]
                    );
                }
                _ => unreachable!("unexpected case_index = {idx}"),
            }
        }
    }
}

mod stopping_rule_conformance {
    //! Conformance tests for `StoppingRule` and `StoppingRuleSet` semantics.

    use cobre_sddp::StopMask;
    use cobre_sddp::stopping_rule::{MonitorState, StoppingMode, StoppingRule, StoppingRuleSet};

    fn make_state(iteration: u64, lb: f64, history: Vec<f64>, shutdown: bool) -> MonitorState {
        MonitorState {
            iteration,
            wall_time_seconds: 0.0,
            lower_bound: lb,
            upper_bound: 0.0,
            lower_bound_history: history,
            shutdown_requested: shutdown,
        }
    }

    const STOPPING_RULE_CASES: &[(&str, f64, u64, u64, usize, bool)] = &[
        ("bound_stalling_max_guard", 0.01, 3, 0, 4, false),
        ("all_mode_requires_simultaneous", 0.001, 5, 10, 2, false),
        ("graceful_shutdown_bypasses_all", 0.001, 5, 100, 0, true),
    ];

    #[test]
    fn stopping_rule_parameter_sweep() {
        for (idx, &(desc, tolerance, stalling_iters, iter_limit, history_len, shutdown)) in
            STOPPING_RULE_CASES.iter().enumerate()
        {
            match idx {
                // History [0.0, 0.0, 0.0, 0.001], window 3 → lb_window_start = 0.0;
                // delta = 0.001 / max(1.0, 0.001) = 0.001 < tolerance 0.01 → triggers.
                0 => {
                    let rule = StoppingRule::BoundStalling {
                        tolerance,
                        iterations: stalling_iters,
                    };
                    let history = vec![0.0_f64; history_len - 1]
                        .into_iter()
                        .chain(std::iter::once(0.001_f64))
                        .collect();
                    let state = make_state(history_len as u64, 0.001, history, shutdown);
                    assert!(
                        rule.is_triggered(&state),
                        "case_index = {idx}, desc = {desc}: BoundStalling must trigger when \
                         |delta|={:.6} < tolerance={tolerance} using max guard",
                        0.001_f64 / 1.0_f64
                    );
                    assert_eq!(
                        rule.name(),
                        "bound_stalling",
                        "case_index = {idx}, desc = {desc}: rule_name must be bound_stalling"
                    );
                }
                1 => {
                    let rule_set = StoppingRuleSet {
                        rules: vec![
                            StoppingRule::IterationLimit { limit: iter_limit },
                            StoppingRule::TimeLimit { seconds: 3600.0 },
                            StoppingRule::BoundStalling {
                                tolerance,
                                iterations: stalling_iters,
                            },
                        ],
                        mode: StoppingMode::All,
                    };
                    // IterationLimit triggers at iter_limit=10.
                    // TimeLimit(3600) does NOT trigger (wall_time=1000s < 3600s).
                    // BoundStalling needs stalling_iters=5 entries; only 2 provided.
                    let history = vec![99.0_f64, 100.0];
                    let state = MonitorState {
                        iteration: iter_limit,
                        wall_time_seconds: 1000.0,
                        lower_bound: 100.0,
                        upper_bound: 0.0,
                        lower_bound_history: history,
                        shutdown_requested: shutdown,
                    };
                    let decision = rule_set.evaluate(&state);
                    assert!(
                        !decision.should_stop(),
                        "case_index = {idx}, desc = {desc}: All mode must not stop when neither \
                         conjunct triggers"
                    );
                    assert!(
                        decision.mask().contains(StopMask::ITERATION_LIMIT),
                        "case_index = {idx}, desc = {desc}: IterationLimit({iter_limit}) must \
                         trigger at iteration {iter_limit}"
                    );
                    assert!(
                        !decision.mask().contains(StopMask::TIME_LIMIT),
                        "case_index = {idx}, desc = {desc}: TimeLimit(3600) must not trigger at \
                         1000s"
                    );
                    assert!(
                        !decision.mask().contains(StopMask::BOUND_STALLING),
                        "case_index = {idx}, desc = {desc}: BoundStalling must not trigger with \
                         only {history_len} history entries"
                    );
                }
                // shutdown_requested=true forces should_stop=true even though
                // IterationLimit(100) has not triggered at iteration 1.
                2 => {
                    let rule_set = StoppingRuleSet {
                        rules: vec![StoppingRule::IterationLimit { limit: iter_limit }],
                        mode: StoppingMode::All,
                    };
                    let state = make_state(1, 0.0, vec![], shutdown);
                    assert!(
                        rule_set.evaluate(&state).should_stop(),
                        "case_index = {idx}, desc = {desc}: a shutdown request must bypass All \
                         mode and force should_stop=true (shutdown_requested={shutdown})"
                    );
                }
                _ => unreachable!("unexpected case_index = {idx}"),
            }
        }
    }
}

mod cut_conformance {
    //! Conformance tests for `CutPool` and `CutWireHeader` round-trip.

    use cobre_sddp::cut::{
        CutPool,
        wire::{CutWireHeader, cut_wire_size, deserialize_cut, serialize_cut},
    };
    use cobre_sddp::setup::NodeId;

    /// Verify `CutWireHeader` serialize/deserialize round-trip with `n_state=3`.
    #[test]
    fn cut_wire_record_serialize_deserialize_round_trip() {
        let n_state = 3;
        let intercept = 42.5_f64;
        let coefficients = [1.0_f64, 2.0, 3.0];

        let slot_index = 7_u32;
        let node_id = -5_i32;
        let iteration = 2_u32;
        let forward_pass_index = 1_u32;

        let mut buf = vec![0u8; cut_wire_size(n_state)];
        serialize_cut(
            &mut buf,
            slot_index,
            node_id,
            iteration,
            forward_pass_index,
            intercept,
            &coefficients,
        );

        let (header, recovered_coeffs) = deserialize_cut(&buf, n_state).unwrap();

        assert_eq!(
            header,
            CutWireHeader {
                slot_index,
                node_id,
                iteration,
                forward_pass_index,
                intercept,
            },
            "deserialized header must match serialized header"
        );

        // Coefficients must match exactly (bit-for-bit).
        assert_eq!(
            recovered_coeffs.len(),
            n_state,
            "recovered coefficient count must be {n_state}"
        );
        for (i, (&orig, &got)) in coefficients.iter().zip(&recovered_coeffs).enumerate() {
            assert_eq!(
                orig.to_bits(),
                got.to_bits(),
                "coefficient[{i}] must round-trip exactly: orig={orig}, got={got}"
            );
        }
    }

    /// Verify `CutPool::active_cuts()` returns all 3 added cuts with correct
    /// intercepts and coefficients.
    #[test]
    fn cut_pool_add_then_active_cuts_returns_correct_data() {
        let mut pool = CutPool::new(10, 1, 1, 0);

        pool.add_cut(NodeId(0), 0, 0, 10.0, &[1.0]);
        pool.add_cut(NodeId(0), 1, 0, 20.0, &[2.0]);
        pool.add_cut(NodeId(0), 2, 0, 30.0, &[3.0]);

        let active: Vec<(usize, f64, &[f64])> = pool.active_cuts().collect();
        assert_eq!(active.len(), 3, "must return all 3 active cuts");

        let mut slot_to_intercept = std::collections::HashMap::new();
        let mut slot_to_coeff = std::collections::HashMap::new();
        for (slot, intercept, coeffs) in &active {
            slot_to_intercept.insert(*slot, *intercept);
            slot_to_coeff.insert(*slot, coeffs[0]);
        }

        assert!(
            (slot_to_intercept[&0] - 10.0).abs() < 1e-10,
            "slot 0 intercept must be 10.0, got {}",
            slot_to_intercept[&0]
        );
        assert!(
            (slot_to_intercept[&1] - 20.0).abs() < 1e-10,
            "slot 1 intercept must be 20.0, got {}",
            slot_to_intercept[&1]
        );
        assert!(
            (slot_to_intercept[&2] - 30.0).abs() < 1e-10,
            "slot 2 intercept must be 30.0, got {}",
            slot_to_intercept[&2]
        );

        assert!(
            (slot_to_coeff[&0] - 1.0).abs() < 1e-10,
            "slot 0 coefficient must be 1.0, got {}",
            slot_to_coeff[&0]
        );
        assert!(
            (slot_to_coeff[&1] - 2.0).abs() < 1e-10,
            "slot 1 coefficient must be 2.0, got {}",
            slot_to_coeff[&1]
        );
        assert!(
            (slot_to_coeff[&2] - 3.0).abs() < 1e-10,
            "slot 2 coefficient must be 3.0, got {}",
            slot_to_coeff[&2]
        );
    }
}

mod convergence_conformance {
    //! Conformance tests for `ConvergenceMonitor` gap formula and history.

    use cobre_sddp::stopping_rule::{StoppingMode, StoppingRule, StoppingRuleSet};
    use cobre_sddp::{ConvergenceMonitor, StopMask};

    use super::make_sync_result;

    fn make_monitor(limit: u64) -> ConvergenceMonitor {
        let rule_set = StoppingRuleSet {
            rules: vec![StoppingRule::IterationLimit { limit }],
            mode: StoppingMode::Any,
        };
        ConvergenceMonitor::new(rule_set)
    }

    const CONVERGENCE_CASES: &[(&str, f64, f64, f64)] = &[
        ("gap_formula_large_lb", 100.0, 110.0, 10.0 / 100.0),
        ("gap_formula_small_lb_max_guard", 0.3, 0.5, 0.2),
        ("lb_history_monotonic", 0.0, 0.0, 0.0), // sentinel; logic below uses lb_values
        ("iteration_limit_exact_trigger", 0.0, 0.0, 0.0), // sentinel; logic below uses loop
    ];

    #[test]
    fn convergence_monitor_parameter_sweep() {
        for (idx, &(desc, lb, ub, expected_gap)) in CONVERGENCE_CASES.iter().enumerate() {
            match idx {
                // gap = (UB - LB) / max(1, |LB|).
                0 | 1 => {
                    let mut monitor = make_monitor(100);
                    monitor.update(lb, &make_sync_result(ub), 0.0);
                    let got = monitor.gap();
                    assert!(
                        (got - expected_gap).abs() < 1e-10,
                        "case_index = {idx}, desc = {desc}: gap(UB={ub}, LB={lb}) must equal \
                         {expected_gap}, got {got}"
                    );
                }
                2 => {
                    let lb_values = [10.0_f64, 20.0, 30.0, 40.0, 50.0];
                    let sync = make_sync_result(110.0);
                    let mut monitor = make_monitor(100);
                    for &v in &lb_values {
                        monitor.update(v, &sync, 0.0);
                    }
                    assert_eq!(
                        monitor.iteration_count(),
                        5,
                        "case_index = {idx}, desc = {desc}: iteration count must equal 5"
                    );
                    assert!(
                        (monitor.lower_bound() - 50.0).abs() < 1e-10,
                        "case_index = {idx}, desc = {desc}: lower_bound() must return latest LB \
                         50.0, got {}",
                        monitor.lower_bound()
                    );
                    // Verify history depth indirectly via BoundStalling requiring 5 entries.
                    let stalling_rule = StoppingRule::BoundStalling {
                        tolerance: 1000.0, // huge tolerance → triggers if enough history
                        iterations: 5,
                    };
                    let rule_set = StoppingRuleSet {
                        rules: vec![stalling_rule],
                        mode: StoppingMode::Any,
                    };
                    let mut monitor2 = ConvergenceMonitor::new(rule_set);
                    for &v in &lb_values {
                        monitor2.update(v, &make_sync_result(110.0), 0.0);
                    }
                    let decision = monitor2.update(50.0, &make_sync_result(110.0), 0.0);
                    assert!(
                        decision.should_stop()
                            && decision.mask().contains(StopMask::BOUND_STALLING),
                        "case_index = {idx}, desc = {desc}: after 6 updates, BoundStalling must \
                         have sufficient history and trigger"
                    );
                }
                3 => {
                    let rule_set = StoppingRuleSet {
                        rules: vec![StoppingRule::IterationLimit { limit: 10 }],
                        mode: StoppingMode::Any,
                    };
                    let mut monitor = ConvergenceMonitor::new(rule_set);
                    let sync = make_sync_result(110.0);
                    for i in 1..10 {
                        let decision = monitor.update(100.0, &sync, 0.0);
                        assert!(
                            !decision.should_stop(),
                            "case_index = {idx}, desc = {desc}: IterationLimit(10) must not \
                             trigger at iteration {i}"
                        );
                        assert!(
                            !decision.mask().contains(StopMask::ITERATION_LIMIT),
                            "case_index = {idx}, desc = {desc}: IterationLimit rule must not be \
                             triggered at iteration {i}"
                        );
                    }
                    let decision = monitor.update(100.0, &sync, 0.0);
                    assert!(
                        decision.should_stop(),
                        "case_index = {idx}, desc = {desc}: IterationLimit(10) must trigger at \
                         iteration 10"
                    );
                    assert!(
                        decision.mask().contains(StopMask::ITERATION_LIMIT),
                        "case_index = {idx}, desc = {desc}: IterationLimit rule must be triggered \
                         at iteration 10"
                    );
                }
                _ => unreachable!("unexpected case_index = {idx}"),
            }
        }
    }
}

mod lb_conformance {
    //! LB monotonicity conformance: adding cuts can only increase the lower bound.

    use cobre_core::SystemBuilder;
    use cobre_core::scenario::SamplingScheme;
    use cobre_sddp::{
        context::TrainingContext,
        horizon_mode::HorizonMode,
        indexer::{StateSpace, StudyDimensions},
        inflow_method::InflowNonNegativityMethod,
        lead_time::AnticipatedResolution,
        lower_bound::{LbEvalScratch, LbEvalScratchBundle, evaluate_lower_bound},
        lp::builder::PatchBuffer,
        risk_measure::RiskMeasure,
        test_support::{StageContextFixture, cut_state_projection, equipment_free_geometry},
        workspace::{ScratchBuffers, WorkspaceSizing},
    };
    use cobre_solver::RowBatch;
    use cobre_stochastic::{
        ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context,
    };

    use super::{LocalComm, MockSolver, make_fcf, minimal_template, simple_opening_tree};

    /// Mirrors the gated `test_support::state_layout_for` body via the
    /// public [`StateSpace::new`] constructor, so this external test crate (which
    /// does not see the parent crate's `#[cfg(test)]` surface) resolves
    /// byte-identical patch columns on the default feature set.
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

    /// Wrap a pre-built stage-0 `OpeningTree` into a degenerate, entity-free
    /// `StochasticContext` (`n_stochastic_ncs() == 0`, `par().n_stages() == 0`),
    /// so this public-API test can supply a real `&StochasticContext` in place
    /// of the retired `stochastic: None` field. `user_tree` bypasses generation
    /// entirely, so the injected tree's shape is preserved verbatim.
    fn wrap_opening_tree(tree: cobre_stochastic::OpeningTree) -> StochasticContext {
        let system = SystemBuilder::new().build().expect("empty system is valid");
        build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs {
                user_tree: Some(tree),
                ..Default::default()
            },
            ClassSchemes {
                inflow: None,
                load: None,
                ncs: None,
            },
        )
        .expect("degenerate stochastic context must build")
    }

    /// Conformance contract: `evaluate_lower_bound` returns a higher (or equal)
    /// value when the mock solver produces higher objectives, simulating the
    /// effect of tighter cuts added to the FCF.
    ///
    /// This replicates the monotonicity property from `lower_bound.rs` as a
    /// public-API integration test.
    #[test]
    fn evaluate_lower_bound_monotonicity_with_additional_cuts() {
        let state_layout = state_layout_for(0, 0);
        let template = minimal_template();
        let templates = vec![template];
        let fcf = make_fcf(2, state_layout.n_state);
        let initial_state = vec![0.0_f64; state_layout.n_state];
        let mut patch_buf = PatchBuffer::new(&state_layout, &[], &[]);
        let opening_tree = simple_opening_tree(2);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;
        let stochastic = wrap_opening_tree(opening_tree);

        let geometry = equipment_free_geometry(&[1]);
        let fixture = StageContextFixture::new(&templates, &[], &geometry);
        let ctx = fixture.ctx();
        let horizon = HorizonMode::Finite { num_stages: 2 };
        let study_dims = StudyDimensions::default();
        let cut_state_layouts = vec![
            cut_state_projection(&state_layout),
            cut_state_projection(&state_layout),
        ];
        let inflow_method = InflowNonNegativityMethod::None;
        let training_ctx = TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state_layout,
            cut_state_layouts: &cut_state_layouts,
            study_dims: &study_dims,
            inflow_method: &inflow_method,
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

        let mut lb_cut_batch = RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        };
        let mut lb_scratch = LbEvalScratch::new();
        let mut noise_scratch =
            ScratchBuffers::new(&training_ctx, &ctx, WorkspaceSizing::default());

        // First call: solver returns [50, 100] → LB = E[50, 100] = 75 (scaled).
        // After unscaling by COST_SCALE_FACTOR (1_000_000), LB = 75_000_000.
        let mut solver1 = MockSolver::with_objectives(vec![50.0, 100.0]);
        let lb1 = {
            let mut bundle = LbEvalScratchBundle::from_scratch_fields(
                &mut patch_buf,
                &mut lb_cut_batch,
                None,
                &mut noise_scratch,
                &mut lb_scratch,
            );
            evaluate_lower_bound(
                &mut solver1,
                &fcf,
                &ctx,
                &training_ctx,
                &rm,
                &mut bundle,
                &comm,
            )
        }
        .expect("first evaluate_lower_bound must succeed");

        assert!(
            (lb1 - 75_000_000.0).abs() < 1e-1,
            "lb1 must equal 75_000_000.0, got {lb1}"
        );

        // Second call: solver returns [80, 120] → LB = E[80, 120] = 100 (scaled).
        // After unscaling by COST_SCALE_FACTOR (1_000_000), LB = 100_000_000.
        // This simulates the effect of tighter cuts (higher stage-0 LP objectives).
        let mut solver2 = MockSolver::with_objectives(vec![80.0, 120.0]);
        let lb2 = {
            let mut bundle = LbEvalScratchBundle::from_scratch_fields(
                &mut patch_buf,
                &mut lb_cut_batch,
                None,
                &mut noise_scratch,
                &mut lb_scratch,
            );
            evaluate_lower_bound(
                &mut solver2,
                &fcf,
                &ctx,
                &training_ctx,
                &rm,
                &mut bundle,
                &comm,
            )
        }
        .expect("second evaluate_lower_bound must succeed");

        assert!(
            (lb2 - 100_000_000.0).abs() < 1e-1,
            "lb2 must equal 100_000_000.0, got {lb2}"
        );

        assert!(
            lb2 >= lb1,
            "lb2 ({lb2}) must be >= lb1 ({lb1}) when cuts are tighter"
        );
    }
}

// ---------------------------------------------------------------------------
// Constraint inventory conformance
// ---------------------------------------------------------------------------

/// Mirrors the gated `test_support::geometry` body through the public
/// [`StageGeometry`](StageGeometry) literal, so this
/// external test crate (which does not see the parent crate's `#[cfg(test)]`
/// surface) resolves byte-identical column ranges on the default feature set. The
/// operational-violation constraint *row* ranges are NOT reproduced here — they
/// are owned internally by `StageLayout` and pinned by a crate-internal test;
/// this helper covers only the column families `StageGeometry` exposes.
#[allow(clippy::too_many_arguments)]
fn build_geometry(
    hydro_count: usize,
    max_par_order: usize,
    n_thermals: usize,
    n_lines: usize,
    n_buses: usize,
    n_blks: usize,
    has_inflow_penalty: bool,
    max_deficit_segments: usize,
    fpha_hydro_indices: Vec<HydroSys>,
    _fpha_planes: &[usize],
) -> StageGeometry {
    // theta = N*(3+L); control region starts at theta + 1 (no anticipated thermals).
    let theta = hydro_count * (3 + max_par_order);
    let turbine_start = theta + 1;
    let spillage_start = turbine_start + hydro_count * n_blks;
    let diversion_start = spillage_start + hydro_count * n_blks;
    let thermal_start = diversion_start + hydro_count * n_blks;
    let thermal_end = thermal_start + n_thermals * n_blks;
    let line_fwd_start = thermal_end;
    let line_rev_start = line_fwd_start + n_lines * n_blks;
    let deficit_start = line_rev_start + n_lines * n_blks;
    let excess_start = deficit_start + n_buses * max_deficit_segments * n_blks;
    let excess_end = excess_start + n_buses * n_blks;
    let (inflow_slack, active_penalty) = if has_inflow_penalty && hydro_count > 0 {
        (excess_end..excess_end + hydro_count, true)
    } else {
        (0..0, false)
    };
    let n_fpha = fpha_hydro_indices.len();
    let generation_start = if active_penalty {
        inflow_slack.end
    } else {
        excess_end
    };
    let generation_end = generation_start + n_fpha * n_blks;
    let generation = if n_fpha > 0 {
        generation_start..generation_end
    } else {
        0..0
    };
    // No evaporation columns in these fixtures, so the evap block is empty.
    let evap_col_end = generation_end;
    let (withdrawal_slack_neg, withdrawal_slack_pos) = if hydro_count > 0 {
        let neg = evap_col_end..evap_col_end + hydro_count;
        let pos = neg.end..neg.end + hydro_count;
        (neg, pos)
    } else {
        (0..0, 0..0)
    };
    let ws_end = withdrawal_slack_pos.end;
    let (ob, oa, tb, gb) = if hydro_count == 0 {
        (0..0, 0..0, 0..0, 0..0)
    } else {
        let n_op = hydro_count * n_blks;
        let ob = ws_end..ws_end + n_op;
        let oa = ob.end..ob.end + n_op;
        let tb = oa.end..oa.end + n_op;
        let gb = tb.end..tb.end + n_op;
        (ob, oa, tb, gb)
    };
    // Rows: z_inflow → water_balance → load_balance; evap/op-violation rows stay
    // internal (`StageGeometry` does not expose them).
    let water_balance_start = hydro_count;
    let load_balance_start = water_balance_start + hydro_count;
    let load_balance_end = load_balance_start + n_buses * n_blks;

    StageGeometry {
        turbine: turbine_start..spillage_start,
        spillage: spillage_start..diversion_start,
        diversion: diversion_start..thermal_start,
        thermal: thermal_start..thermal_end,
        anticipated_decision: 0..0,
        line_fwd: line_fwd_start..line_rev_start,
        line_rev: line_rev_start..deficit_start,
        deficit: deficit_start..excess_start,
        excess: excess_start..excess_end,
        generation,
        // This conformance geometry models no NCS or pumping columns.
        ncs_generation: 0..0,
        pumping_flow: 0..0,
        evap_indices: Vec::new(),
        inflow_slack,
        withdrawal_slack_neg,
        withdrawal_slack_pos,
        outflow_below_slack: ob,
        outflow_above_slack: oa,
        turbine_below_slack: tb,
        generation_below_slack: gb,
        // This conformance geometry models no contract columns; the production
        // `start..start`-at-pumping-end anchoring is owned by
        // `StageGeometry::from_layout`.
        contract_import: 0..0,
        contract_export: 0..0,
        water_balance: BlockRowFamily::one_per_entity(
            water_balance_start..water_balance_start + hydro_count,
        ),
        load_balance: BlockRowFamily::per_block(load_balance_start..load_balance_end),
        fpha: load_balance_end..load_balance_end,
        // This conformance geometry models no filling hydros, so the
        // terminal-target and operating-floor blocks are empty — including the
        // sparse `σ_fill` / `σ^{v-}` system→slot index vectors.
        filling_target: 0..0,
        filling_target_col: 0..0,
        filled_min_storage_floor: 0..0,
        filled_min_storage_floor_col: 0..0,
        n_blks,
        storage_internal_start: 0,
        block_mode: BlockMode::Parallel,
        fpha_hydro_indices,
        evap_hydro_indices: Vec::new(),
        filling_target_hydro_indices: Vec::new(),
        filled_min_storage_floor_hydro_indices: Vec::new(),
    }
}

/// Verify that the per-stage geometry produces non-empty ranges for every
/// equipment/slack column family when all features are active.
///
/// This catches bugs where the layout arithmetic silently produces empty (0..0)
/// ranges for a column family that should be present. The operational-violation
/// constraint *row* ranges (`min_outflow_rows` etc.) are owned internally by
/// `StageLayout` and are pinned by the crate-internal
/// `stage_layout_operational_violation_rows_are_contiguous_blocks` test; this
/// public-API test covers the column families `StageGeometry` exposes.
#[test]
fn indexer_constraint_inventory() {
    // N=3, L=1, T=2, Ln=1, B=2, K=2, penalty on, S=2; FPHA hydro 0 with 3 planes.
    let geometry = build_geometry(3, 1, 2, 1, 2, 2, true, 2, vec![HydroSys::new(0)], &[3]);

    assert!(
        !geometry.outflow_below_slack.is_empty(),
        "outflow_below_slack must be non-empty"
    );
    assert!(
        !geometry.outflow_above_slack.is_empty(),
        "outflow_above_slack must be non-empty"
    );
    assert!(
        !geometry.turbine_below_slack.is_empty(),
        "turbine_below_slack must be non-empty"
    );
    assert!(
        !geometry.generation_below_slack.is_empty(),
        "generation_below_slack must be non-empty"
    );

    assert!(
        !geometry.inflow_slack.is_empty(),
        "inflow_slack must be non-empty when has_inflow_penalty=true"
    );

    assert!(
        !geometry.withdrawal_slack_neg.is_empty(),
        "withdrawal_slack_neg must be non-empty when hydro_count > 0"
    );
    assert!(
        !geometry.withdrawal_slack_pos.is_empty(),
        "withdrawal_slack_pos must be non-empty when hydro_count > 0"
    );

    assert_eq!(
        geometry.outflow_above_slack.start, geometry.outflow_below_slack.end,
        "outflow_above must start where outflow_below ends"
    );
    assert_eq!(
        geometry.turbine_below_slack.start, geometry.outflow_above_slack.end,
        "turbine_below must start where outflow_above ends"
    );
    assert_eq!(
        geometry.generation_below_slack.start, geometry.turbine_below_slack.end,
        "generation_below must start where turbine_below ends"
    );

    let hydro_count = 3;
    let n_blks = 2;
    let n_op = hydro_count * n_blks;
    assert_eq!(geometry.outflow_below_slack.len(), n_op);
    assert_eq!(geometry.outflow_above_slack.len(), n_op);
    assert_eq!(geometry.turbine_below_slack.len(), n_op);
    assert_eq!(geometry.generation_below_slack.len(), n_op);
    assert_eq!(geometry.withdrawal_slack_neg.len(), hydro_count);
    assert_eq!(geometry.withdrawal_slack_pos.len(), hydro_count);
}

/// CI regression guard: verifies the expected number of hydro-related slack
/// column families in the per-stage geometry and checks for range overlaps.
///
/// When a developer adds a new constraint type to the LP builder, they must
/// also:
/// 1. Add the corresponding slack column range to `StageGeometry`/`StageLayout`.
/// 2. Add extraction logic in `accumulate_category_costs`.
/// 3. Add a new field to `SimulationCostResult`.
/// 4. Update this guard count.
#[test]
fn constraint_extraction_regression_guard() {
    use std::ops::Range;

    // N=2, L=1, T=1, Ln=1, B=1, K=1, penalty on, S=1; FPHA hydro 0 with 2 planes.
    let geometry = build_geometry(2, 1, 1, 1, 1, 1, true, 1, vec![HydroSys::new(0)], &[2]);

    // The families that feed the hydro violation cost decomposition in
    // accumulate_category_costs().
    let slack_families: Vec<(&str, &Range<usize>)> = vec![
        ("outflow_below_slack", &geometry.outflow_below_slack),
        ("outflow_above_slack", &geometry.outflow_above_slack),
        ("turbine_below_slack", &geometry.turbine_below_slack),
        ("generation_below_slack", &geometry.generation_below_slack),
        ("withdrawal_slack_neg", &geometry.withdrawal_slack_neg),
        ("withdrawal_slack_pos", &geometry.withdrawal_slack_pos),
        ("inflow_slack", &geometry.inflow_slack),
    ];

    assert_eq!(
        slack_families.len(),
        7,
        "Expected exactly 7 hydro-related slack column families. \
         If you added a new constraint type, update: (a) accumulate_category_costs, \
         (b) SimulationCostResult, (c) CostWriteRecord + costs_schema, \
         (d) this regression guard."
    );

    for i in 0..slack_families.len() {
        let (name_a, range_a) = &slack_families[i];
        if range_a.is_empty() {
            continue;
        }
        for (name_b, range_b) in &slack_families[i + 1..] {
            if range_b.is_empty() {
                continue;
            }
            let overlaps = range_a.start < range_b.end && range_b.start < range_a.end;
            assert!(
                !overlaps,
                "Slack column range overlap detected between {name_a} ({range_a:?}) and \
                 {name_b} ({range_b:?}). This indicates a layout arithmetic bug in \
                 the per-stage geometry."
            );
        }
    }
}
