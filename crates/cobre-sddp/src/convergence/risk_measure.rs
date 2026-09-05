//! Risk measure for cut aggregation and risk-adjusted cost evaluation.
//!
//! [`RiskMeasure`] aggregation replaces opening probabilities `p(ω)` with
//! risk-adjusted weights `μ*_ω`. For `Expectation`, `μ*_ω = p(ω)`; for `CVaR`,
//! a greedy allocation places maximum mass on the highest-cost scenarios
//! after reserving `(1 - λ)·p(ω)` for every scenario, realizing
//! `ρ^{λ,α}[Z] = (1 - λ)·E[Z] + λ·CVaR_α[Z]`.
//!
//! ## Examples
//!
//! ```rust
//! use cobre_sddp::risk_measure::{BackwardOutcome, RiskMeasure};
//!
//! // Expectation: weighted average of intercepts
//! let outcomes = vec![
//!     BackwardOutcome { intercept: 10.0, coefficients: vec![], objective_value: 10.0 },
//!     BackwardOutcome { intercept: 20.0, coefficients: vec![], objective_value: 20.0 },
//!     BackwardOutcome { intercept: 30.0, coefficients: vec![], objective_value: 30.0 },
//! ];
//! let probs = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
//! let (intercept, _) = RiskMeasure::Expectation.aggregate_cut(&outcomes, &probs);
//! assert!((intercept - 20.0).abs() < 1e-10);
//! ```

use cobre_core::StageRiskConfig;
use cobre_core::StageRiskConfig::CVaR;
use cobre_core::StageRiskConfig::Expectation;
/// Per-worker scratch buffers for `CVaR` weight computation, reused across
/// backward-pass stages so the allocation is paid once. Owned exclusively per
/// rayon worker (a field of `BackwardAccumulators`), so no synchronisation.
#[derive(Debug, Default, Clone)]
pub struct RiskMeasureScratch {
    /// Per-scenario caps on additional `CVaR` mass `λ p_ω / α`.
    pub upper_bounds: Vec<f64>,
    /// Scenario indices sorted descending by objective/cost value.
    pub order: Vec<usize>,
    /// Computed risk weights `μ*_ω`.
    pub mu: Vec<f64>,
}

impl RiskMeasureScratch {
    /// Create an empty scratch; capacities grow lazily on first use.
    #[must_use]
    pub fn new() -> Self {
        Self {
            upper_bounds: Vec::new(),
            order: Vec::new(),
            mu: Vec::new(),
        }
    }
}

/// Results from solving one backward pass opening at a single stage. The
/// intercept and coefficients derive from the LP dual variables (Cut Management
/// SS2); `objective_value` ranks scenarios for `CVaR` allocation (Risk Measures
/// SS7).
#[derive(Debug, Clone)]
pub struct BackwardOutcome {
    /// Per-scenario cut intercept `α_t(ω)`.
    pub intercept: f64,

    /// Per-scenario cut coefficients `π_t(ω)`, one per state variable. Must be
    /// the same length across all outcomes in one `aggregate_cut` call.
    pub coefficients: Vec<f64>,

    /// Optimal objective value `Q_t(x̂, ω)`; higher means a worse scenario.
    pub objective_value: f64,
}

/// Risk measure for stage-level cut aggregation: how opening-level outcomes are
/// weighted into a single cut. Enum dispatch over a closed variant set.
///
/// ## Examples
///
/// ```rust
/// use cobre_sddp::risk_measure::{BackwardOutcome, RiskMeasure};
///
/// let rm = RiskMeasure::CVaR { alpha: 0.5, lambda: 1.0 };
/// let costs = vec![10.0, 20.0, 30.0, 40.0];
/// let probs = vec![0.25; 4];
/// let result = rm.evaluate_risk(&costs, &probs);
/// assert!((result - 35.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RiskMeasure {
    /// Risk-neutral expected value: weights equal the opening probabilities
    /// `μ*_ω = p(ω)` (Cut Management SS3).
    Expectation,

    /// Convex combination of expectation and `CVaR`:
    /// `ρ^{λ,α}[Z] = (1 - λ) E[Z] + λ · CVaR_α[Z]` (Risk Measures SS3, SS7).
    CVaR {
        /// `CVaR` upper-tail fraction `α ∈ (0, 1]`; `α = 1` equals expectation,
        /// smaller `α` concentrates weight on the worst `α`-fraction.
        alpha: f64,

        /// Risk aversion weight `λ ∈ [0, 1]`; `λ = 0` reduces to `Expectation`
        /// (normalised at config load time), `λ = 1` gives pure `CVaR`.
        lambda: f64,
    },
}

impl From<StageRiskConfig> for RiskMeasure {
    fn from(config: StageRiskConfig) -> Self {
        match config {
            Expectation => Self::Expectation,
            CVaR { alpha, lambda } => Self::CVaR { alpha, lambda },
        }
    }
}

impl RiskMeasure {
    /// Aggregate per-opening backward pass results into a single cut: the
    /// weighted sum of per-opening intercepts and coefficients under `μ*_ω`.
    ///
    /// ## Preconditions
    ///
    /// - `outcomes.len() == probabilities.len() > 0`
    /// - `probabilities` sum to `1.0` within floating-point tolerance
    /// - all `outcomes[i].coefficients` have equal length
    #[must_use]
    pub fn aggregate_cut(
        &self,
        outcomes: &[BackwardOutcome],
        probabilities: &[f64],
    ) -> (f64, Vec<f64>) {
        debug_assert_eq!(
            outcomes.len(),
            probabilities.len(),
            "aggregate_cut: outcomes and probabilities must have the same length"
        );
        debug_assert!(
            !outcomes.is_empty(),
            "aggregate_cut: at least one outcome required"
        );

        match self {
            RiskMeasure::Expectation => aggregate_weighted(outcomes, probabilities),
            RiskMeasure::CVaR { alpha, lambda } => {
                let mu = compute_cvar_weights(outcomes, probabilities, *alpha, *lambda);
                aggregate_weighted(outcomes, &mu)
            }
        }
    }

    /// No-allocation buffer variant of [`aggregate_cut`](RiskMeasure::aggregate_cut):
    /// writes into caller-provided buffers and reuses `scratch`. `Expectation`
    /// does not touch `scratch`.
    ///
    /// ## Preconditions
    ///
    /// - `outcomes.len() == probabilities.len() > 0`
    /// - `coefficients_out.len() == outcomes[0].coefficients.len()`
    /// - all `outcomes[i].coefficients` have equal length
    pub(crate) fn aggregate_cut_into(
        &self,
        outcomes: &[BackwardOutcome],
        probabilities: &[f64],
        intercept_out: &mut f64,
        coefficients_out: &mut [f64],
        scratch: &mut RiskMeasureScratch,
    ) {
        debug_assert_eq!(
            outcomes.len(),
            probabilities.len(),
            "aggregate_cut_into: outcomes and probabilities must have the same length"
        );
        debug_assert!(
            !outcomes.is_empty(),
            "aggregate_cut_into: at least one outcome required"
        );

        match self {
            RiskMeasure::Expectation => {
                aggregate_weighted_into(outcomes, probabilities, intercept_out, coefficients_out);
            }
            RiskMeasure::CVaR { alpha, lambda } => {
                compute_cvar_weights_into(outcomes, probabilities, *alpha, *lambda, scratch);
                aggregate_weighted_into(outcomes, &scratch.mu, intercept_out, coefficients_out);
            }
        }
    }

    /// Evaluate the risk-adjusted scalar cost from a vector of cost values, used
    /// for convergence bound computation. `Expectation` is the probability-weighted
    /// mean; `CVaR` is the convex combination `(1-λ) E[Z] + λ · CVaR_α[Z]`.
    ///
    /// ## Preconditions
    ///
    /// - `costs.len() == probabilities.len() > 0`
    /// - `probabilities` sum to `1.0` within floating-point tolerance
    #[must_use]
    pub fn evaluate_risk(&self, costs: &[f64], probabilities: &[f64]) -> f64 {
        let mut scratch = RiskMeasureScratch::new();
        self.evaluate_risk_into(costs, probabilities, &mut scratch)
    }

    /// [`Self::evaluate_risk`] reusing `scratch` for the `CVaR` weight allocation;
    /// prefer this on a hot path that evaluates many vectors (e.g. the per-node
    /// nested upper-bound recursion), so the allocation is paid once.
    #[must_use]
    pub(crate) fn evaluate_risk_into(
        &self,
        costs: &[f64],
        probabilities: &[f64],
        scratch: &mut RiskMeasureScratch,
    ) -> f64 {
        debug_assert_eq!(
            costs.len(),
            probabilities.len(),
            "evaluate_risk: costs and probabilities must have the same length"
        );
        debug_assert!(
            !costs.is_empty(),
            "evaluate_risk: at least one cost required"
        );

        match self {
            RiskMeasure::Expectation => costs.iter().zip(probabilities).map(|(c, p)| c * p).sum(),
            RiskMeasure::CVaR { alpha, lambda } => {
                // EAVaR = E_μ*[Z]: by the dual representation (Risk Measures SS4.2)
                // the greedy allocation in compute_cvar_weights_from_costs_into is
                // the optimal μ*, so the weighted sum below equals (1-λ)E[Z]+λCVaR_α[Z].
                compute_cvar_weights_from_costs_into(
                    costs,
                    probabilities,
                    *alpha,
                    *lambda,
                    scratch,
                );
                costs
                    .iter()
                    .zip(scratch.mu.iter())
                    .map(|(c, w)| c * w)
                    .sum()
            }
        }
    }

    /// Collapse the documented `CVaR { lambda: 0 }` ≡ `Expectation` equivalence
    /// to `Expectation` so a zero-risk-aversion `CVaR` compares and aggregates as
    /// the risk-neutral measure it is. A positive-`lambda` `CVaR` is returned
    /// unchanged. The `lambda > 0` predicate matches `is_effective_non_expectation`.
    #[must_use]
    pub(crate) fn effective(self) -> RiskMeasure {
        if matches!(self, RiskMeasure::CVaR { lambda, .. } if lambda > 0.0) {
            self
        } else {
            RiskMeasure::Expectation
        }
    }
}

/// The single risk measure shared by every stage, or `None` when they differ.
///
/// Compared on the [`effective`](RiskMeasure::effective) form, so a mix of
/// `Expectation` and `CVaR { lambda: 0 }` counts as uniform. This is the measure
/// the enumerated risk-adjusted upper bound applies once to the path costs, and
/// the uniformity a `gap` stopping rule requires under `CVaR` (a per-stage
/// varying measure has no single static bound).
#[must_use]
pub(crate) fn uniform_effective_measure(measures: &[RiskMeasure]) -> Option<RiskMeasure> {
    let first = measures.first()?.effective();
    measures
        .iter()
        .all(|m| m.effective() == first)
        .then_some(first)
}

/// Compute `CVaR` weights via continuous-knapsack greedy allocation on objective
/// values, reusing `scratch`. After this call, `scratch.mu[i]` is scenario `i`'s
/// risk weight.
pub fn compute_cvar_weights_into(
    outcomes: &[BackwardOutcome],
    probabilities: &[f64],
    alpha: f64,
    lambda: f64,
    scratch: &mut RiskMeasureScratch,
) {
    let n = outcomes.len();

    scratch.upper_bounds.clear();
    scratch
        .upper_bounds
        .extend(probabilities.iter().map(|&p| lambda * p / alpha));

    scratch.order.clear();
    scratch.order.extend(0..n);
    scratch.order.sort_by(|&i, &j| {
        outcomes[j]
            .objective_value
            .total_cmp(&outcomes[i].objective_value)
    });

    scratch.mu.clear();
    scratch
        .mu
        .extend(probabilities.iter().map(|&p| (1.0 - lambda) * p));
    let mut remaining = lambda;
    for &idx in &scratch.order {
        if remaining <= 0.0 {
            break;
        }
        let alloc = scratch.upper_bounds[idx].min(remaining);
        scratch.mu[idx] += alloc;
        remaining -= alloc;
    }
}

/// [`compute_cvar_weights_into`] over raw `costs: &[f64]` rather than
/// `&[BackwardOutcome]`, used by [`RiskMeasure::evaluate_risk`].
pub fn compute_cvar_weights_from_costs_into(
    costs: &[f64],
    probabilities: &[f64],
    alpha: f64,
    lambda: f64,
    scratch: &mut RiskMeasureScratch,
) {
    let n = costs.len();

    scratch.upper_bounds.clear();
    scratch
        .upper_bounds
        .extend(probabilities.iter().map(|&p| lambda * p / alpha));

    scratch.order.clear();
    scratch.order.extend(0..n);
    scratch
        .order
        .sort_by(|&i, &j| costs[j].total_cmp(&costs[i]));

    scratch.mu.clear();
    scratch
        .mu
        .extend(probabilities.iter().map(|&p| (1.0 - lambda) * p));
    let mut remaining = lambda;
    for &idx in &scratch.order {
        if remaining <= 0.0 {
            break;
        }
        let alloc = scratch.upper_bounds[idx].min(remaining);
        scratch.mu[idx] += alloc;
        remaining -= alloc;
    }
}

/// Allocating wrapper around [`compute_cvar_weights_into`]; prefer the `_into`
/// form on hot paths.
fn compute_cvar_weights(
    outcomes: &[BackwardOutcome],
    probabilities: &[f64],
    alpha: f64,
    lambda: f64,
) -> Vec<f64> {
    let mut scratch = RiskMeasureScratch::new();
    compute_cvar_weights_into(outcomes, probabilities, alpha, lambda, &mut scratch);
    scratch.mu
}

fn aggregate_weighted(outcomes: &[BackwardOutcome], weights: &[f64]) -> (f64, Vec<f64>) {
    let state_dim = outcomes.first().map_or(0, |o| o.coefficients.len());

    let mut agg_intercept = 0.0_f64;
    let mut agg_coefficients = vec![0.0_f64; state_dim];

    aggregate_weighted_into(outcomes, weights, &mut agg_intercept, &mut agg_coefficients);

    (agg_intercept, agg_coefficients)
}

/// Write weighted-aggregation results into caller-provided buffers, bit-identical
/// to [`aggregate_weighted`] but without allocating.
///
/// ## Preconditions
///
/// - `outcomes.len() == weights.len()`
/// - `coefficients_out.len() == outcomes[0].coefficients.len()`
pub(crate) fn aggregate_weighted_into(
    outcomes: &[BackwardOutcome],
    weights: &[f64],
    intercept_out: &mut f64,
    coefficients_out: &mut [f64],
) {
    coefficients_out.fill(0.0);
    *intercept_out = 0.0;
    for (outcome, &w) in outcomes.iter().zip(weights) {
        *intercept_out += w * outcome.intercept;
        for (agg, &coeff) in coefficients_out.iter_mut().zip(&outcome.coefficients) {
            *agg += w * coeff;
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)] // test helpers use small n values
mod tests {
    use cobre_core::StageRiskConfig;

    use super::{BackwardOutcome, RiskMeasure};

    fn outcome(intercept: f64, obj: f64) -> BackwardOutcome {
        BackwardOutcome {
            intercept,
            coefficients: vec![],
            objective_value: obj,
        }
    }

    fn outcome_with_coeffs(intercept: f64, obj: f64, coeffs: Vec<f64>) -> BackwardOutcome {
        BackwardOutcome {
            intercept,
            coefficients: coeffs,
            objective_value: obj,
        }
    }

    fn uniform(n: usize) -> Vec<f64> {
        let p = 1.0_f64 / (n as f64);
        vec![p; n]
    }

    #[test]
    fn expectation_aggregate_cut_equal_probs_mean_intercept() {
        let outcomes = vec![
            outcome(10.0, 10.0),
            outcome(20.0, 20.0),
            outcome(30.0, 30.0),
        ];
        let probs = uniform(3);
        let (intercept, _) = RiskMeasure::Expectation.aggregate_cut(&outcomes, &probs);
        assert!(
            (intercept - 20.0).abs() < 1e-10,
            "expected 20.0, got {intercept}"
        );
    }

    #[test]
    fn expectation_aggregate_cut_nonuniform_probs() {
        let outcomes = vec![
            outcome(10.0, 10.0),
            outcome(20.0, 20.0),
            outcome(30.0, 30.0),
        ];
        let probs = vec![0.5, 0.3, 0.2];
        let (intercept, _) = RiskMeasure::Expectation.aggregate_cut(&outcomes, &probs);
        let expected = 0.5 * 10.0 + 0.3 * 20.0 + 0.2 * 30.0; // 17.0
        assert!(
            (intercept - expected).abs() < 1e-10,
            "expected {expected}, got {intercept}"
        );
    }

    #[test]
    fn expectation_aggregate_cut_coefficients_weighted() {
        let outcomes = vec![
            outcome_with_coeffs(0.0, 0.0, vec![1.0, 2.0]),
            outcome_with_coeffs(0.0, 0.0, vec![3.0, 4.0]),
        ];
        let probs = vec![0.5, 0.5];
        let (_, coeffs) = RiskMeasure::Expectation.aggregate_cut(&outcomes, &probs);
        assert_eq!(coeffs.len(), 2);
        assert!((coeffs[0] - 2.0).abs() < 1e-10); // 0.5*1 + 0.5*3
        assert!((coeffs[1] - 3.0).abs() < 1e-10); // 0.5*2 + 0.5*4
    }

    #[test]
    fn expectation_evaluate_risk_equal_probs() {
        let costs = vec![10.0, 20.0, 30.0];
        let probs = uniform(3);
        let result = RiskMeasure::Expectation.evaluate_risk(&costs, &probs);
        assert!((result - 20.0).abs() < 1e-10, "expected 20.0, got {result}");
    }

    #[test]
    fn expectation_evaluate_risk_nonuniform_probs() {
        let costs = vec![100.0, 200.0];
        let probs = vec![0.7, 0.3];
        let result = RiskMeasure::Expectation.evaluate_risk(&costs, &probs);
        let expected = 0.7 * 100.0 + 0.3 * 200.0; // 130.0
        assert!(
            (result - expected).abs() < 1e-10,
            "expected {expected}, got {result}"
        );
    }

    #[test]
    fn cvar_evaluate_risk_pure_cvar_alpha_half() {
        let rm = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 1.0,
        };
        let costs = vec![10.0, 20.0, 30.0, 40.0];
        let probs = vec![0.25; 4];
        let result = rm.evaluate_risk(&costs, &probs);
        assert!((result - 35.0).abs() < 1e-10, "expected 35.0, got {result}");
    }

    #[test]
    fn cvar_evaluate_risk_alpha_one_equals_expectation() {
        let rm_cvar = RiskMeasure::CVaR {
            alpha: 1.0,
            lambda: 1.0,
        };
        let costs = vec![10.0, 20.0, 30.0, 40.0];
        let probs = vec![0.25; 4];
        let result_cvar = rm_cvar.evaluate_risk(&costs, &probs);
        let result_exp = RiskMeasure::Expectation.evaluate_risk(&costs, &probs);
        assert!(
            (result_cvar - result_exp).abs() < 1e-10,
            "CVaR with alpha=1 should equal Expectation: {result_cvar} vs {result_exp}"
        );
    }

    #[test]
    fn cvar_evaluate_risk_lambda_zero_equals_expectation() {
        let rm_cvar = RiskMeasure::CVaR {
            alpha: 0.2,
            lambda: 0.0,
        };
        let costs = vec![5.0, 15.0, 25.0, 35.0];
        let probs = vec![0.25; 4];
        let result_cvar = rm_cvar.evaluate_risk(&costs, &probs);
        let result_exp = RiskMeasure::Expectation.evaluate_risk(&costs, &probs);
        assert!(
            (result_cvar - result_exp).abs() < 1e-10,
            "CVaR with lambda=0 should equal Expectation: {result_cvar} vs {result_exp}"
        );
    }

    #[test]
    fn cvar_mixture_preserves_expectation_floor_in_value_and_cut() {
        let rm = RiskMeasure::CVaR {
            alpha: 0.15,
            lambda: 0.4,
        };
        let outcomes = [
            outcome_with_coeffs(10.0, 0.0, vec![1.0, 0.0]),
            outcome_with_coeffs(20.0, 100.0, vec![0.0, 1.0]),
        ];
        let probabilities = [0.5, 0.5];
        assert!((rm.evaluate_risk(&[0.0, 100.0], &probabilities) - 70.0).abs() < 1e-10);
        let (intercept, coefficients) = rm.aggregate_cut(&outcomes, &probabilities);
        assert!((intercept - 17.0).abs() < 1e-10);
        assert!((coefficients[0] - 0.3).abs() < 1e-10);
        assert!((coefficients[1] - 0.7).abs() < 1e-10);
        let mut scratch = super::RiskMeasureScratch::new();
        let mut buffered_intercept = 0.0;
        let mut buffered_coefficients = [0.0; 2];
        rm.aggregate_cut_into(
            &outcomes,
            &probabilities,
            &mut buffered_intercept,
            &mut buffered_coefficients,
            &mut scratch,
        );
        assert_eq!(buffered_intercept, intercept);
        assert_eq!(buffered_coefficients.as_slice(), coefficients.as_slice());
    }

    #[test]
    fn cvar_mixture_twenty_equiprobable_costs() {
        let costs: Vec<_> = (1..=20).map(f64::from).collect();
        let rm = RiskMeasure::CVaR {
            alpha: 0.15,
            lambda: 0.4,
        };
        assert!((rm.evaluate_risk(&costs, &[0.05; 20]) - 13.9).abs() < 1e-10);
    }

    #[test]
    fn cvar_mixture_matches_primal_tail_formula_and_envelope() {
        let costs = [-20.0, 10.0, 10.0, 80.0, 1000.0];
        let probabilities = [0.1, 0.2, 0.5, 0.2, 0.0];
        let outcomes: Vec<_> = costs.iter().map(|&c| outcome(c, c)).collect();
        let expectation: f64 = costs.iter().zip(probabilities).map(|(c, p)| c * p).sum();
        let mut scratch = super::RiskMeasureScratch::new();
        let mut outcome_scratch = super::RiskMeasureScratch::new();
        for alpha in [0.01, 0.15, 0.3, 0.5, 1.0] {
            let cvar = costs
                .iter()
                .map(|&eta| {
                    eta + costs
                        .iter()
                        .zip(probabilities)
                        .map(|(&cost, p)| p * (cost - eta).max(0.0))
                        .sum::<f64>()
                        / alpha
                })
                .fold(f64::INFINITY, f64::min);
            for lambda in [0.0, 0.1, 0.4, 0.5, 0.9, 1.0] {
                let rm = RiskMeasure::CVaR { alpha, lambda };
                let actual = rm.evaluate_risk_into(&costs, &probabilities, &mut scratch);
                let expected = (1.0 - lambda) * expectation + lambda * cvar;
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "alpha={alpha}, lambda={lambda}: {actual} != {expected}"
                );
                super::compute_cvar_weights_into(
                    &outcomes,
                    &probabilities,
                    alpha,
                    lambda,
                    &mut outcome_scratch,
                );
                assert_eq!(scratch.mu, outcome_scratch.mu);
                assert!((scratch.mu.iter().sum::<f64>() - 1.0).abs() < 1e-10);
                for (&mu, p) in scratch.mu.iter().zip(probabilities) {
                    let floor = (1.0 - lambda) * p;
                    assert!(mu >= floor - 1e-12);
                    assert!(mu <= floor + lambda * p / alpha + 1e-12);
                }
                assert_eq!(scratch.mu[4], 0.0);
            }
        }
        super::compute_cvar_weights_from_costs_into(
            &[0.0, 100.0],
            &[0.5, 0.5],
            0.15,
            0.4,
            &mut scratch,
        );
        assert_eq!(scratch.mu.len(), 2);
        assert!((scratch.mu[0] - 0.3).abs() < 1e-10);
        assert!((scratch.mu[1] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn cvar_evaluate_risk_convex_combination() {
        let rm = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 0.5,
        };
        let costs = vec![0.0, 100.0];
        let probs = vec![0.5, 0.5];
        let result = rm.evaluate_risk(&costs, &probs);
        assert!((result - 75.0).abs() < 1e-10);
    }

    #[test]
    fn cvar_aggregate_cut_pure_cvar_selects_worst() {
        let outcomes = vec![
            outcome(10.0, 10.0),
            outcome(20.0, 20.0),
            outcome(30.0, 30.0),
            outcome(40.0, 40.0),
        ];
        let probs = vec![0.25; 4];
        let rm = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 1.0,
        };
        let (intercept, _) = rm.aggregate_cut(&outcomes, &probs);
        assert!((intercept - 35.0).abs() < 1e-10);
    }

    #[test]
    fn cvar_aggregate_cut_with_coefficients() {
        let outcomes = vec![
            outcome_with_coeffs(10.0, 10.0, vec![1.0, 0.0]),
            outcome_with_coeffs(20.0, 20.0, vec![0.0, 1.0]),
        ];
        let probs = vec![0.5, 0.5];
        let rm = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 1.0,
        };
        let (intercept, coeffs) = rm.aggregate_cut(&outcomes, &probs);
        assert!((intercept - 20.0).abs() < 1e-10);
        assert_eq!(coeffs.len(), 2);
        assert!((coeffs[0] - 0.0).abs() < 1e-10);
        assert!((coeffs[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cvar_aggregate_cut_alpha_one_equals_expectation() {
        let outcomes = vec![
            outcome(10.0, 10.0),
            outcome(20.0, 20.0),
            outcome(30.0, 30.0),
        ];
        let probs = uniform(3);
        let rm_exp = RiskMeasure::Expectation;
        let rm_cvar = RiskMeasure::CVaR {
            alpha: 1.0,
            lambda: 1.0,
        };
        let (int_exp, _) = rm_exp.aggregate_cut(&outcomes, &probs);
        let (int_cvar, _) = rm_cvar.aggregate_cut(&outcomes, &probs);
        assert!(
            (int_exp - int_cvar).abs() < 1e-10,
            "alpha=1 CVaR should equal Expectation: {int_exp} vs {int_cvar}"
        );
    }

    #[test]
    fn cvar_aggregate_cut_lambda_zero_equals_expectation() {
        // lambda=0: the expectation floor is the entire mass.
        let outcomes = vec![
            outcome(10.0, 10.0),
            outcome(20.0, 20.0),
            outcome(30.0, 30.0),
        ];
        let probs = uniform(3);
        let rm_exp = RiskMeasure::Expectation;
        let rm_cvar = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 0.0,
        };
        let (int_exp, _) = rm_exp.aggregate_cut(&outcomes, &probs);
        let (int_cvar, _) = rm_cvar.aggregate_cut(&outcomes, &probs);
        assert!(
            (int_exp - int_cvar).abs() < 1e-10,
            "lambda=0 CVaR should equal Expectation: {int_exp} vs {int_cvar}"
        );
    }

    #[test]
    fn cvar_aggregate_cut_weights_sum_to_one() {
        let outcomes = [
            outcome(10.0, 15.0),
            outcome(20.0, 5.0),
            outcome(30.0, 25.0),
            outcome(40.0, 35.0),
        ];
        let probs = vec![0.3, 0.2, 0.3, 0.2];
        let rm = RiskMeasure::CVaR {
            alpha: 0.3,
            lambda: 0.8,
        };
        // Compute weights indirectly: aggregate scalar-1 intercepts and sum
        // (not directly accessible, but we verify via a single-coefficient outcome)
        let unit_outcomes: Vec<_> = (0..4)
            .map(|i| super::BackwardOutcome {
                intercept: 1.0,
                coefficients: vec![1.0],
                objective_value: outcomes[i].objective_value,
            })
            .collect();
        let (intercept, coeffs) = rm.aggregate_cut(&unit_outcomes, &probs);
        // If weights sum to 1, both intercept and coeff[0] should equal 1.0
        assert!(
            (intercept - 1.0).abs() < 1e-10,
            "weight sum must be 1.0, got intercept={intercept}"
        );
        assert!(
            (coeffs[0] - 1.0).abs() < 1e-10,
            "weight sum must be 1.0 (coeff check), got {}",
            coeffs[0]
        );
    }

    #[test]
    fn risk_measure_debug_copy_eq() {
        let rm = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 0.8,
        };
        let copied = rm;
        assert_eq!(copied, rm);
        assert_ne!(copied, RiskMeasure::Expectation);
        let debug_str = format!("{rm:?}");
        assert!(debug_str.contains("CVaR"));
    }

    #[test]
    fn backward_outcome_debug_and_clone() {
        let o = BackwardOutcome {
            intercept: 1.0,
            coefficients: vec![2.0, 3.0],
            objective_value: 5.0,
        };
        let cloned = o.clone();
        let debug_str = format!("{o:?}");
        assert!(debug_str.contains("BackwardOutcome"));
        assert!((cloned.intercept - o.intercept).abs() < f64::EPSILON);
    }

    #[test]
    fn test_from_stage_risk_config_expectation() {
        let config = StageRiskConfig::Expectation;
        let rm = RiskMeasure::from(config);
        assert!(matches!(rm, RiskMeasure::Expectation));
    }

    #[test]
    fn test_from_stage_risk_config_cvar() {
        let config = StageRiskConfig::CVaR {
            alpha: 0.95,
            lambda: 0.5,
        };
        let rm = RiskMeasure::from(config);
        assert!(matches!(
            rm,
            RiskMeasure::CVaR {
                alpha: 0.95,
                lambda: 0.5
            }
        ));
    }

    #[test]
    fn aggregate_weighted_into_matches_aggregate_weighted() {
        use super::aggregate_weighted_into;

        let outcomes = vec![
            outcome_with_coeffs(10.0, 10.0, vec![1.0, 2.0, 3.0]),
            outcome_with_coeffs(20.0, 20.0, vec![4.0, 5.0, 6.0]),
            outcome_with_coeffs(30.0, 30.0, vec![7.0, 8.0, 9.0]),
        ];
        let weights = vec![0.5, 0.3, 0.2];

        let (ref_intercept, ref_coeffs) =
            RiskMeasure::Expectation.aggregate_cut(&outcomes, &weights);

        let mut intercept_out = 0.0_f64;
        let mut coefficients_out = vec![0.0_f64; 3];
        aggregate_weighted_into(
            &outcomes,
            &weights,
            &mut intercept_out,
            &mut coefficients_out,
        );

        assert_eq!(
            intercept_out, ref_intercept,
            "intercept must be bit-identical"
        );
        assert_eq!(
            coefficients_out, ref_coeffs,
            "coefficients must be bit-identical"
        );
    }

    #[test]
    fn aggregate_cut_into_matches_aggregate_cut_expectation() {
        use super::RiskMeasureScratch;

        let outcomes = vec![
            outcome_with_coeffs(5.0, 5.0, vec![1.0, 0.0]),
            outcome_with_coeffs(15.0, 15.0, vec![0.0, 1.0]),
        ];
        let probs = vec![0.6, 0.4];

        let (ref_intercept, ref_coeffs) = RiskMeasure::Expectation.aggregate_cut(&outcomes, &probs);

        let mut intercept_out = 0.0_f64;
        let mut coefficients_out = vec![0.0_f64; 2];
        let mut scratch = RiskMeasureScratch::new();
        RiskMeasure::Expectation.aggregate_cut_into(
            &outcomes,
            &probs,
            &mut intercept_out,
            &mut coefficients_out,
            &mut scratch,
        );

        assert_eq!(intercept_out, ref_intercept, "intercept bit-identical");
        assert_eq!(coefficients_out, ref_coeffs, "coefficients bit-identical");
    }

    #[test]
    fn aggregate_cut_into_matches_aggregate_cut_cvar() {
        use super::RiskMeasureScratch;

        let outcomes = vec![
            outcome_with_coeffs(10.0, 10.0, vec![1.0, 0.0]),
            outcome_with_coeffs(20.0, 20.0, vec![0.0, 1.0]),
            outcome_with_coeffs(30.0, 30.0, vec![1.0, 1.0]),
        ];
        let probs = vec![1.0 / 3.0; 3];
        let rm = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 1.0,
        };

        let (ref_intercept, ref_coeffs) = rm.aggregate_cut(&outcomes, &probs);

        let mut intercept_out = 0.0_f64;
        let mut coefficients_out = vec![0.0_f64; 2];
        let mut scratch = RiskMeasureScratch::new();
        rm.aggregate_cut_into(
            &outcomes,
            &probs,
            &mut intercept_out,
            &mut coefficients_out,
            &mut scratch,
        );

        assert_eq!(intercept_out, ref_intercept, "CVaR intercept bit-identical");
        assert_eq!(
            coefficients_out, ref_coeffs,
            "CVaR coefficients bit-identical"
        );
    }

    /// The backward cut applies the risk measure ONCE over the joint
    /// successor×opening outcome vector, never per-child-then-averaged. On a
    /// 2-child × 2-opening pure-CVaR fan the two disagree by a closed-form margin:
    /// the joint `CVaR₀.₅` over `[10, 20, 30, 40]` (weights `0.25`) concentrates on
    /// the worst two (`30`, `40`) → `35`; the nested measure (`CVaR` per child, then
    /// probability-average the two children) gives `max` per child then averages →
    /// `(20 + 40)/2 = 30`. The engine must produce the joint `35` (`aggregate_cut_into`).
    #[test]
    fn joint_cvar_differs_from_nested_per_child_then_average() {
        use super::RiskMeasureScratch;

        let rm = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 1.0,
        };

        // Child A openings [10, 20]; child B openings [30, 40]; intercept == objective.
        let joint_outcomes = vec![
            outcome_with_coeffs(10.0, 10.0, vec![0.0]),
            outcome_with_coeffs(20.0, 20.0, vec![0.0]),
            outcome_with_coeffs(30.0, 30.0, vec![0.0]),
            outcome_with_coeffs(40.0, 40.0, vec![0.0]),
        ];
        // Joint weights P(child)·q = 0.5·0.5 = 0.25 per outcome, canonical order.
        let joint_weights = vec![0.25_f64; 4];

        let mut joint_intercept = 0.0_f64;
        let mut joint_coeffs = vec![0.0_f64; 1];
        let mut scratch = RiskMeasureScratch::new();
        rm.aggregate_cut_into(
            &joint_outcomes,
            &joint_weights,
            &mut joint_intercept,
            &mut joint_coeffs,
            &mut scratch,
        );
        assert!(
            (joint_intercept - 35.0).abs() < 1e-10,
            "joint CVaR over the 4-outcome vector must be 35.0, got {joint_intercept}"
        );

        // Mutation control: the nested measure — CVaR within each child, then the
        // probability average across children — the semantics the joint contract rejects.
        let child_weights = vec![0.5_f64; 2];
        let (cvar_a, _) = rm.aggregate_cut(&joint_outcomes[0..2], &child_weights);
        let (cvar_b, _) = rm.aggregate_cut(&joint_outcomes[2..4], &child_weights);
        let nested = 0.5 * cvar_a + 0.5 * cvar_b;
        assert!(
            (nested - 30.0).abs() < 1e-10,
            "nested per-child-then-average CVaR must be 30.0, got {nested}"
        );
        assert!(
            (joint_intercept - nested).abs() > 1.0,
            "joint ({joint_intercept}) and nested ({nested}) must differ — the pin is vacuous otherwise"
        );
    }

    #[test]
    fn compute_cvar_weights_into_matches_allocating_variant() {
        use super::{RiskMeasureScratch, compute_cvar_weights_into};

        let outcomes = vec![
            outcome(10.0, 10.0),
            outcome(20.0, 20.0),
            outcome(30.0, 30.0),
            outcome(40.0, 40.0),
        ];
        let probs = vec![0.25; 4];

        let rm = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 1.0,
        };
        let (ref_intercept, _) = rm.aggregate_cut(&outcomes, &probs);

        let mut scratch = RiskMeasureScratch::new();
        compute_cvar_weights_into(&outcomes, &probs, 0.5, 1.0, &mut scratch);

        let weighted_intercept: f64 = outcomes
            .iter()
            .zip(scratch.mu.iter())
            .map(|(o, w)| o.intercept * w)
            .sum();
        assert!(
            (weighted_intercept - ref_intercept).abs() < 1e-10,
            "into variant must produce identical weighted result: got {weighted_intercept}, expected {ref_intercept}"
        );
        let weight_sum: f64 = scratch.mu.iter().sum();
        assert!(
            (weight_sum - 1.0).abs() < 1e-10,
            "weights must sum to 1.0, got {weight_sum}"
        );
    }

    #[test]
    fn risk_measure_cvar_aggregate_cut_into_reuses_scratch() {
        use super::RiskMeasureScratch;

        let outcomes = vec![
            outcome_with_coeffs(10.0, 10.0, vec![1.0, 0.0]),
            outcome_with_coeffs(20.0, 20.0, vec![0.0, 1.0]),
            outcome_with_coeffs(30.0, 30.0, vec![1.0, 1.0]),
        ];
        let probs = vec![1.0 / 3.0; 3];
        let rm = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 1.0,
        };

        let mut scratch = RiskMeasureScratch::new();

        let mut intercept1 = 0.0_f64;
        let mut coefficients1 = vec![0.0_f64; 2];
        rm.aggregate_cut_into(
            &outcomes,
            &probs,
            &mut intercept1,
            &mut coefficients1,
            &mut scratch,
        );
        let cap_after_first = scratch.mu.capacity();

        let mut intercept2 = 0.0_f64;
        let mut coefficients2 = vec![0.0_f64; 2];
        rm.aggregate_cut_into(
            &outcomes,
            &probs,
            &mut intercept2,
            &mut coefficients2,
            &mut scratch,
        );
        let cap_after_second = scratch.mu.capacity();

        assert_eq!(
            intercept1, intercept2,
            "results must be identical across calls"
        );
        assert_eq!(
            coefficients1, coefficients2,
            "coefficients must be identical across calls"
        );
        assert!(
            cap_after_second >= cap_after_first,
            "scratch capacity must not shrink: first={cap_after_first}, second={cap_after_second}"
        );
    }
}
