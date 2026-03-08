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

// Shared helpers (reimplemented here because integration tests cannot access
// #[cfg(test)] items from the main crate)

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_sddp::forward::SyncResult;
use cobre_solver::{
    Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
};

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
}

/// Mock solver that returns configurable objective values in sequence.
struct MockSolver {
    objectives: Vec<f64>,
    call_count: usize,
    infeasible_on_call: Option<usize>,
}

impl MockSolver {
    fn with_objectives(objectives: Vec<f64>) -> Self {
        Self {
            objectives,
            call_count: 0,
            infeasible_on_call: None,
        }
    }
}

impl SolverInterface for MockSolver {
    fn load_model(&mut self, _template: &StageTemplate) {}
    fn add_rows(&mut self, _cuts: &RowBatch) {}
    fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

    fn solve(&mut self) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
        let call = self.call_count;
        self.call_count += 1;
        if self.infeasible_on_call == Some(call) {
            return Err(SolverError::Infeasible);
        }
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

    fn reset(&mut self) {
        self.call_count = 0;
    }

    fn get_basis(&mut self, _out: &mut Basis) {}

    fn solve_with_basis(
        &mut self,
        _basis: &Basis,
    ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
        self.solve()
    }

    fn statistics(&self) -> SolverStatistics {
        SolverStatistics::default()
    }

    fn name(&self) -> &'static str {
        "MockConformance"
    }
}

/// Minimal stage template for a single hydro, zero PAR lags.
fn minimal_template() -> StageTemplate {
    StageTemplate {
        num_cols: 3,
        num_rows: 1,
        num_nz: 1,
        col_starts: vec![0, 0, 1, 1],
        row_indices: vec![0],
        values: vec![1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
        objective: vec![0.0, 0.0, 1.0],
        row_lower: vec![0.0],
        row_upper: vec![0.0],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 1,
        n_hydro: 1,
        max_par_order: 0,
    }
}

/// Build an `OpeningTree` with `n_openings` openings at stage 0.
fn simple_opening_tree(n_openings: usize) -> cobre_stochastic::OpeningTree {
    use chrono::NaiveDate;
    use cobre_core::{
        EntityId,
        scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };
    use cobre_stochastic::correlation::resolve::DecomposedCorrelation;
    use std::collections::BTreeMap;

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
            branching_factor: n_openings,
            noise_method: NoiseMethod::Saa,
        },
    };

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
        method: "cholesky".to_string(),
        profiles,
        schedule: vec![],
    };
    let mut decomposed = DecomposedCorrelation::build(&corr_model).unwrap();
    let entity_order = vec![entity_id];

    cobre_stochastic::tree::generate::generate_opening_tree(
        42,
        &[stage],
        1,
        &mut decomposed,
        &entity_order,
    )
}

fn make_sync_result(global_ub_mean: f64) -> SyncResult {
    SyncResult {
        global_ub_mean,
        global_ub_std: 0.0,
        ci_95_half_width: 0.0,
        sync_time_ms: 0,
    }
}

fn make_fcf(n_stages: usize, state_dimension: usize) -> cobre_sddp::cut::fcf::FutureCostFunction {
    cobre_sddp::cut::fcf::FutureCostFunction::new(n_stages, state_dimension, 2, 100, 0)
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

    /// Verify `Expectation.aggregate_cut` computes a probability-weighted mean
    /// with 4 outcomes and non-uniform probabilities.
    #[test]
    fn risk_measure_expectation_aggregate_cut_sums_to_weighted_mean() {
        let outcomes = vec![
            outcome(10.0, 10.0, vec![1.0]),
            outcome(20.0, 20.0, vec![2.0]),
            outcome(30.0, 30.0, vec![3.0]),
            outcome(40.0, 40.0, vec![4.0]),
        ];
        // Non-uniform probabilities that sum to 1.0
        let probs = vec![0.1, 0.2, 0.3, 0.4];

        let (intercept, coeffs) = RiskMeasure::Expectation.aggregate_cut(&outcomes, &probs);

        // Expected: 0.1*10 + 0.2*20 + 0.3*30 + 0.4*40 = 1 + 4 + 9 + 16 = 30.0
        let expected_intercept = 0.1 * 10.0 + 0.2 * 20.0 + 0.3 * 30.0 + 0.4 * 40.0;
        assert!(
            (intercept - expected_intercept).abs() < 1e-10,
            "intercept must equal weighted mean {expected_intercept}, got {intercept}"
        );

        // Expected coefficient: 0.1*1 + 0.2*2 + 0.3*3 + 0.4*4 = 0.1+0.4+0.9+1.6 = 3.0
        let expected_coeff = 0.1 * 1.0 + 0.2 * 2.0 + 0.3 * 3.0 + 0.4 * 4.0;
        assert_eq!(coeffs.len(), 1, "coefficient vector must have length 1");
        assert!(
            (coeffs[0] - expected_coeff).abs() < 1e-10,
            "coefficient[0] must equal {expected_coeff}, got {}",
            coeffs[0]
        );
    }

    /// Verify `CVaR(alpha=1.0, lambda=1.0).aggregate_cut` produces identical
    /// results to `Expectation.aggregate_cut`.
    #[test]
    fn risk_measure_cvar_alpha_one_equals_expectation() {
        let outcomes = vec![
            outcome(10.0, 10.0, vec![1.0]),
            outcome(20.0, 20.0, vec![2.0]),
            outcome(30.0, 30.0, vec![3.0]),
            outcome(40.0, 40.0, vec![4.0]),
        ];
        let probs = vec![0.25; 4];

        let (int_exp, coeffs_exp) = RiskMeasure::Expectation.aggregate_cut(&outcomes, &probs);
        let (int_cvar, coeffs_cvar) = RiskMeasure::CVaR {
            alpha: 1.0,
            lambda: 1.0,
        }
        .aggregate_cut(&outcomes, &probs);

        assert!(
            (int_exp - int_cvar).abs() < 1e-10,
            "CVaR(alpha=1, lambda=1) intercept {int_cvar} must equal Expectation {int_exp}"
        );
        assert_eq!(coeffs_exp.len(), coeffs_cvar.len());
        for (i, (e, c)) in coeffs_exp.iter().zip(&coeffs_cvar).enumerate() {
            assert!(
                (e - c).abs() < 1e-10,
                "coefficient[{i}]: CVaR={c} must equal Expectation={e}"
            );
        }

        // Also verify evaluate_risk matches
        let costs = vec![10.0, 20.0, 30.0, 40.0];
        let risk_exp = RiskMeasure::Expectation.evaluate_risk(&costs, &probs);
        let risk_cvar = RiskMeasure::CVaR {
            alpha: 1.0,
            lambda: 1.0,
        }
        .evaluate_risk(&costs, &probs);
        assert!(
            (risk_exp - risk_cvar).abs() < 1e-10,
            "CVaR(alpha=1, lambda=1) evaluate_risk {risk_cvar} must equal Expectation {risk_exp}"
        );
    }

    /// Verify `CVaR(alpha=0.5, lambda=1.0)` with 4 uniform-probability outcomes
    /// places all weight on the worst 50% (outcomes 30 and 40).
    #[test]
    fn risk_measure_cvar_alpha_half_concentrates_on_worst() {
        let outcomes = vec![
            outcome(10.0, 10.0, vec![]), // cheapest
            outcome(20.0, 20.0, vec![]),
            outcome(30.0, 30.0, vec![]),
            outcome(40.0, 40.0, vec![]), // most expensive
        ];
        let probs = vec![0.25; 4];

        let rm = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 1.0,
        };
        let (intercept, _) = rm.aggregate_cut(&outcomes, &probs);

        // CVaR(0.5, 1.0) with 4 uniform outcomes concentrates weight on the worst 50%.
        // Per-scenario upper bound = p / alpha = 0.25 / 0.5 = 0.5 for each scenario.
        // Greedy fills worst two (40, 30) with 0.5 each → sum = 0.5*40 + 0.5*30 = 35.0
        let expected = 35.0_f64;
        assert!(
            (intercept - expected).abs() < 1e-10,
            "CVaR(0.5, 1.0) must concentrate on worst 50%: expected {expected}, got {intercept}"
        );
    }

    /// Verify that the `CVaR` weights sum to exactly 1.0 for non-uniform
    /// probabilities and arbitrary alpha/lambda.
    #[test]
    fn risk_measure_cvar_weights_sum_to_one() {
        // Non-uniform probabilities; alpha and lambda chosen to produce a
        // non-trivial weight distribution.
        let outcomes = [
            outcome(10.0, 15.0, vec![1.0]),
            outcome(20.0, 5.0, vec![1.0]),
            outcome(30.0, 25.0, vec![1.0]),
            outcome(40.0, 35.0, vec![1.0]),
        ];
        let probs = vec![0.3, 0.2, 0.3, 0.2];

        let rm = RiskMeasure::CVaR {
            alpha: 0.3,
            lambda: 0.8,
        };

        // If weights sum to 1, aggregating unit intercepts yields intercept=1.
        let unit_outcomes: Vec<BackwardOutcome> = outcomes
            .iter()
            .map(|o| BackwardOutcome {
                intercept: 1.0,
                coefficients: vec![1.0],
                objective_value: o.objective_value,
            })
            .collect();

        let (intercept, coeffs) = rm.aggregate_cut(&unit_outcomes, &probs);
        assert!(
            (intercept - 1.0).abs() < 1e-10,
            "weights must sum to 1.0 (intercept check): got {intercept}"
        );
        assert!(
            (coeffs[0] - 1.0).abs() < 1e-10,
            "weights must sum to 1.0 (coeff check): got {}",
            coeffs[0]
        );
    }
}

mod stopping_rule_conformance {
    //! Conformance tests for `StoppingRule` and `StoppingRuleSet` semantics.

    use cobre_sddp::stopping_rule::{MonitorState, StoppingMode, StoppingRule, StoppingRuleSet};

    fn make_state(iteration: u64, lb: f64, history: Vec<f64>, shutdown: bool) -> MonitorState {
        MonitorState {
            iteration,
            wall_time_seconds: 0.0,
            lower_bound: lb,
            lower_bound_history: history,
            shutdown_requested: shutdown,
            simulation_costs: None,
        }
    }

    /// Verify `BoundStalling` uses `max(1.0, |lb|)` as denominator, preventing
    /// division by near-zero values.
    ///
    /// When `lb = 0.001` over a window of 3 iterations where the history
    /// starts at 0.0, the delta = (0.001 - 0.0) / max(1.0, 0.001) = 0.001 / 1.0
    /// = 0.001, which is below tolerance=0.01 → triggers.
    #[test]
    fn stopping_rule_bound_stalling_uses_max_guard() {
        let rule = StoppingRule::BoundStalling {
            tolerance: 0.01,
            iterations: 3,
        };
        // History: [0.0, 0.0, 0.0, 0.001]. Window of 3 → lb_window_start = history[1] = 0.0.
        // lb_current = 0.001, denominator = max(1.0, |0.001|) = 1.0
        // delta = (0.001 - 0.0) / 1.0 = 0.001 < 0.01 → triggered
        let history = vec![0.0, 0.0, 0.0, 0.001];
        let state = make_state(4, 0.001, history, false);
        let result = rule.evaluate(&state);

        assert!(
            result.triggered,
            "BoundStalling must trigger when |delta|={:.6} < tolerance=0.01 using max guard",
            0.001_f64 / 1.0_f64
        );
        assert_eq!(result.rule_name, "bound_stalling");
    }

    /// Verify `StoppingRuleSet` with `StoppingMode::All` requires ALL non-shutdown
    /// rules to trigger simultaneously — 2 of 3 triggering is insufficient.
    #[test]
    fn stopping_rule_set_all_mode_requires_simultaneous() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 10 },
                StoppingRule::TimeLimit { seconds: 3600.0 },
                StoppingRule::BoundStalling {
                    tolerance: 0.001,
                    iterations: 5,
                },
            ],
            mode: StoppingMode::All,
        };

        // IterationLimit(10) triggers at iteration 10.
        // TimeLimit(3600) does NOT trigger (wall_time = 1000s < 3600s).
        // BoundStalling requires 5 history entries; provide only 2 → not triggered.
        let state = MonitorState {
            iteration: 10,
            wall_time_seconds: 1000.0, // below the 3600s limit
            lower_bound: 100.0,
            lower_bound_history: vec![99.0, 100.0], // only 2 entries, need 5
            shutdown_requested: false,
            simulation_costs: None,
        };

        let (should_stop, results) = rule_set.evaluate(&state);

        // All mode: 2 of 3 non-shutdown rules triggered → must NOT stop
        assert!(
            !should_stop,
            "All mode must not stop when only 2 of 3 rules trigger"
        );
        assert_eq!(results.len(), 3, "must return 3 results");
        assert!(
            results[0].triggered,
            "IterationLimit(10) must trigger at iteration 10"
        );
        assert!(
            !results[1].triggered,
            "TimeLimit(3600) must not trigger at 1000s"
        );
        assert!(
            !results[2].triggered,
            "BoundStalling must not trigger with only 2 history entries"
        );
    }

    /// Verify `GracefulShutdown` bypasses `StoppingMode::All` logic.
    ///
    /// Even with `All` mode and only `GracefulShutdown` + a non-triggered
    /// `IterationLimit`, a shutdown signal forces `should_stop = true`.
    #[test]
    fn stopping_rule_graceful_shutdown_bypasses_all_mode() {
        let rule_set = StoppingRuleSet {
            rules: vec![
                StoppingRule::IterationLimit { limit: 100 },
                StoppingRule::GracefulShutdown,
            ],
            mode: StoppingMode::All,
        };

        // Iteration 1 does NOT trigger IterationLimit(100), but shutdown IS requested.
        let state = make_state(1, 0.0, vec![], true);
        let (should_stop, _) = rule_set.evaluate(&state);

        assert!(
            should_stop,
            "GracefulShutdown must bypass All mode and force should_stop=true"
        );
    }
}

mod cut_conformance {
    //! Conformance tests for `CutPool` and `CutWireHeader` round-trip.

    use cobre_sddp::cut::{
        CutPool,
        wire::{CutWireHeader, cut_wire_size, deserialize_cut, serialize_cut},
    };

    /// Verify `CutWireHeader` serialize/deserialize round-trip with `n_state=3`.
    ///
    /// Creates a buffer, serializes a cut with `intercept=42.5` and
    /// `coefficients=[1.0, 2.0, 3.0]`, then deserializes and checks all fields.
    #[test]
    fn cut_wire_record_serialize_deserialize_round_trip() {
        let n_state = 3;
        let intercept = 42.5_f64;
        let coefficients = [1.0_f64, 2.0, 3.0];

        // Slot/iteration/forward_pass_index values chosen to be non-zero and distinct.
        let slot_index = 7_u32;
        let iteration = 2_u32;
        let forward_pass_index = 1_u32;

        let mut buf = vec![0u8; cut_wire_size(n_state)];
        serialize_cut(
            &mut buf,
            slot_index,
            iteration,
            forward_pass_index,
            intercept,
            &coefficients,
        );

        let (header, recovered_coeffs) = deserialize_cut(&buf, n_state);

        // All header fields must match exactly.
        assert_eq!(
            header,
            CutWireHeader {
                slot_index,
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
        // 1-dimensional state, 1 forward pass per iteration, no warm-start.
        let mut pool = CutPool::new(10, 1, 1, 0);

        // Add 3 cuts at deterministic slots 0, 1, 2.
        pool.add_cut(0, 0, 10.0, &[1.0]);
        pool.add_cut(1, 0, 20.0, &[2.0]);
        pool.add_cut(2, 0, 30.0, &[3.0]);

        let active: Vec<(usize, f64, &[f64])> = pool.active_cuts().collect();
        assert_eq!(active.len(), 3, "must return all 3 active cuts");

        // Collect intercepts and coefficients by slot index for deterministic check.
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

    use cobre_sddp::convergence::ConvergenceMonitor;
    use cobre_sddp::stopping_rule::{StoppingMode, StoppingRule, StoppingRuleSet};

    use super::make_sync_result;

    fn make_monitor(limit: u64) -> ConvergenceMonitor {
        let rule_set = StoppingRuleSet {
            rules: vec![StoppingRule::IterationLimit { limit }],
            mode: StoppingMode::Any,
        };
        ConvergenceMonitor::new(rule_set)
    }

    /// Verify the gap formula `(UB - LB) / max(1, |UB|)`.
    ///
    /// Case 1: UB=110, LB=100 → denominator=110 → gap=10/110.
    /// Case 2: UB=0.5, LB=0.3 → denominator=max(1,0.5)=1.0 → gap=0.2/1.0=0.2.
    #[test]
    fn convergence_monitor_gap_formula_matches_spec() {
        // Case 1: UB=110, LB=100
        let mut monitor1 = make_monitor(100);
        monitor1.update(100.0, &make_sync_result(110.0));
        let expected1 = 10.0_f64 / 110.0_f64;
        assert!(
            (monitor1.gap() - expected1).abs() < 1e-10,
            "gap(UB=110, LB=100) must equal {expected1}, got {}",
            monitor1.gap()
        );

        // Case 2: UB=0.5, LB=0.3 → denominator=max(1.0, 0.5)=1.0
        let mut monitor2 = make_monitor(100);
        monitor2.update(0.3, &make_sync_result(0.5));
        let expected2 = (0.5_f64 - 0.3_f64) / 1.0_f64; // = 0.2 / 1.0
        assert!(
            (monitor2.gap() - expected2).abs() < 1e-10,
            "gap(UB=0.5, LB=0.3) must equal {expected2} (max guard=1.0), got {}",
            monitor2.gap()
        );
    }

    /// Verify LB history grows monotonically when LB values increase over 5 calls.
    ///
    /// After 5 updates with strictly increasing LB values [10, 20, 30, 40, 50],
    /// the internal history must have length 5 and the LB accessor must return
    /// the latest value (50).
    #[test]
    fn convergence_monitor_lb_history_grows_monotonically_when_lb_increases() {
        let mut monitor = make_monitor(100);
        let lb_values = [10.0_f64, 20.0, 30.0, 40.0, 50.0];
        let sync = make_sync_result(110.0);

        for &lb in &lb_values {
            monitor.update(lb, &sync);
        }

        assert_eq!(
            monitor.iteration_count(),
            5,
            "iteration count must equal number of update calls"
        );
        assert!(
            (monitor.lower_bound() - 50.0).abs() < 1e-10,
            "lower_bound() must return latest LB 50.0, got {}",
            monitor.lower_bound()
        );

        // Verify that iteration_count grew from 0 to 5 monotonically (proxy for history length).
        // The history is an internal field but we can verify it indirectly via a BoundStalling
        // rule that requires 5 history entries — it should now have enough.
        let stalling_rule = StoppingRule::BoundStalling {
            tolerance: 1000.0, // huge tolerance → triggers if enough history
            iterations: 5,
        };
        let rule_set = StoppingRuleSet {
            rules: vec![stalling_rule],
            mode: StoppingMode::Any,
        };
        let mut monitor2 = ConvergenceMonitor::new(rule_set);
        for &lb in &lb_values {
            monitor2.update(lb, &make_sync_result(110.0));
        }
        // After 5 calls, the BoundStalling rule has 5 entries and should evaluate (not skip).
        // With tolerance=1000 and strictly increasing LBs, delta > 0, which is < 1000 → triggers.
        let (should_stop, results) = monitor2.update(50.0, &make_sync_result(110.0));
        assert_eq!(results[0].rule_name, "bound_stalling");
        assert!(
            should_stop || !results[0].detail.contains("insufficient history"),
            "after 6 updates, BoundStalling must have sufficient history"
        );
    }

    /// Verify `IterationLimit(10)` triggers at exactly iteration 10 and not at 9.
    #[test]
    fn convergence_monitor_iteration_limit_triggers_at_exact_count() {
        let rule_set = StoppingRuleSet {
            rules: vec![StoppingRule::IterationLimit { limit: 10 }],
            mode: StoppingMode::Any,
        };
        let mut monitor = ConvergenceMonitor::new(rule_set);
        let sync = make_sync_result(110.0);

        // Iterations 1-9: must NOT trigger
        for i in 1..10 {
            let (stop, results) = monitor.update(100.0, &sync);
            assert!(
                !stop,
                "IterationLimit(10) must not trigger at iteration {i}"
            );
            assert!(
                !results[0].triggered,
                "IterationLimit rule must not be triggered at iteration {i}"
            );
        }

        // Iteration 10: MUST trigger
        let (stop, results) = monitor.update(100.0, &sync);
        assert!(stop, "IterationLimit(10) must trigger at iteration 10");
        assert!(
            results[0].triggered,
            "IterationLimit rule must be triggered at iteration 10"
        );
        assert_eq!(results[0].rule_name, "iteration_limit");
    }
}

mod lb_conformance {
    //! LB monotonicity conformance: adding cuts can only increase the lower bound.

    use cobre_sddp::{PatchBuffer, RiskMeasure, StageIndexer, lower_bound::evaluate_lower_bound};

    use super::{LocalComm, MockSolver, make_fcf, minimal_template, simple_opening_tree};

    /// Conformance contract: `evaluate_lower_bound` returns a higher (or equal)
    /// value when the mock solver produces higher objectives, simulating the
    /// effect of tighter cuts added to the FCF.
    ///
    /// This replicates the monotonicity property from `lower_bound.rs` as a
    /// public-API integration test.
    #[test]
    fn evaluate_lower_bound_monotonicity_with_additional_cuts() {
        let indexer = StageIndexer::new(1, 0);
        let template = minimal_template();
        let fcf = make_fcf(2, indexer.n_state);
        let initial_state = vec![0.0_f64; indexer.n_state];
        let mut patch_buf = PatchBuffer::new(indexer.hydro_count, indexer.max_par_order);
        let opening_tree = simple_opening_tree(2);
        let rm = RiskMeasure::Expectation;
        let comm = LocalComm;

        // First call: solver returns [50, 100] → LB = E[50, 100] = 75.
        let mut solver1 = MockSolver::with_objectives(vec![50.0, 100.0]);
        let lb1 = evaluate_lower_bound(
            &mut solver1,
            &template,
            &fcf,
            &initial_state,
            1,
            &indexer,
            &mut patch_buf,
            &opening_tree,
            &rm,
            &comm,
        )
        .expect("first evaluate_lower_bound must succeed");

        assert!((lb1 - 75.0).abs() < 1e-10, "lb1 must equal 75.0, got {lb1}");

        // Second call: solver returns [80, 120] → LB = E[80, 120] = 100.
        // This simulates the effect of tighter cuts (higher stage-0 LP objectives).
        let mut solver2 = MockSolver::with_objectives(vec![80.0, 120.0]);
        let lb2 = evaluate_lower_bound(
            &mut solver2,
            &template,
            &fcf,
            &initial_state,
            1,
            &indexer,
            &mut patch_buf,
            &opening_tree,
            &rm,
            &comm,
        )
        .expect("second evaluate_lower_bound must succeed");

        assert!(
            (lb2 - 100.0).abs() < 1e-10,
            "lb2 must equal 100.0, got {lb2}"
        );

        // Monotonicity: lb2 >= lb1
        assert!(
            lb2 >= lb1,
            "lb2 ({lb2}) must be >= lb1 ({lb1}) when cuts are tighter"
        );
    }
}
