//! Risk measure for cut aggregation and risk-adjusted cost evaluation.
//!
//! [`RiskMeasure`] aggregation replaces opening probabilities `p(ω)` with
//! risk-adjusted weights `μ*_ω`. For `Expectation`, `μ*_ω = p(ω)`; for `CVaR`,
//! `μ*_ω = (1 - λ)·p(ω) + λ·ν_ω`, where the pure `CVaR_α` allocation `ν` places
//! maximum mass (cap `p(ω)/α`) on the highest-cost scenarios (Risk Measures SS7),
//! realizing `ρ^{λ,α}[Z] = (1 - λ)·E[Z] + λ·CVaR_α[Z]`.
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
/// Reusable `CVaR` weight-computation buffers, so the allocation is paid once.
/// Every consumer (each backward worker, the lower bound, the nested upper
/// bound) owns its own, so no synchronisation.
#[derive(Debug, Default, Clone)]
pub struct RiskMeasureScratch {
    /// Per-scenario pure-`CVaR` caps `p_ω/α`.
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
        /// `CVaR` confidence level `α ∈ (0, 1]`; `α = 1` equals expectation,
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
/// the enumerated risk-adjusted upper bound applies at every node of its nested
/// recursion, and the uniformity a `gap` stopping rule requires under `CVaR` (a
/// per-stage varying measure has no single static bound).
#[must_use]
pub(crate) fn uniform_effective_measure(measures: &[RiskMeasure]) -> Option<RiskMeasure> {
    let first = measures.first()?.effective();
    measures
        .iter()
        .all(|m| m.effective() == first)
        .then_some(first)
}

/// Write the `CVaR` weights of `outcomes`, ranked by objective value, into
/// `scratch.mu`: `μ = (1−λ)·p + λ·ν`, with `ν` the pure `CVaR_α` allocation.
pub fn compute_cvar_weights_into(
    outcomes: &[BackwardOutcome],
    probabilities: &[f64],
    alpha: f64,
    lambda: f64,
    scratch: &mut RiskMeasureScratch,
) {
    cvar_weights_kernel(
        outcomes.len(),
        |i| outcomes[i].objective_value,
        probabilities,
        alpha,
        lambda,
        scratch,
    );
}

/// The `μ = (1−λ)·p + λ·ν` of [`compute_cvar_weights_into`] over raw `costs`
/// rather than `&[BackwardOutcome]`, used by [`RiskMeasure::evaluate_risk`].
pub fn compute_cvar_weights_from_costs_into(
    costs: &[f64],
    probabilities: &[f64],
    alpha: f64,
    lambda: f64,
    scratch: &mut RiskMeasureScratch,
) {
    cvar_weights_kernel(
        costs.len(),
        |i| costs[i],
        probabilities,
        alpha,
        lambda,
        scratch,
    );
}

/// `μ_ω = (1−λ)·p_ω + λ·ν_ω`, where `ν` greedily fills mass 1 at cap `p_ω/α`
/// from the costliest opening down; a single greedy at cap `(1−λ)·p_ω + λ·p_ω/α`
/// with no floor is the wrong-but-compiling alternative. The index tie-break
/// keeps the unstable sort declaration-order deterministic.
fn cvar_weights_kernel(
    n: usize,
    value: impl Fn(usize) -> f64,
    probabilities: &[f64],
    alpha: f64,
    lambda: f64,
    scratch: &mut RiskMeasureScratch,
) {
    scratch.order.clear();
    scratch.order.extend(0..n);
    scratch
        .order
        .sort_unstable_by(|&i, &j| value(j).total_cmp(&value(i)).then(i.cmp(&j)));

    scratch.upper_bounds.clear();
    scratch
        .upper_bounds
        .extend(probabilities.iter().map(|&p| p / alpha));

    scratch.mu.clear();
    scratch.mu.resize(n, 0.0);
    let mut remaining = 1.0_f64;
    for &idx in &scratch.order {
        if remaining <= 0.0 {
            break;
        }
        let alloc = scratch.upper_bounds[idx].min(remaining);
        scratch.mu[idx] = alloc;
        remaining -= alloc;
    }

    for (mu, &p) in scratch.mu.iter_mut().zip(probabilities) {
        *mu = (1.0 - lambda) * p + lambda * *mu;
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

    #[test]
    fn cvar_weights_match_analytic_table() {
        use super::{
            RiskMeasureScratch, compute_cvar_weights_from_costs_into, compute_cvar_weights_into,
        };

        struct Row {
            probs: &'static [f64],
            costs: &'static [f64],
            alpha: f64,
            lambda: f64,
            weights: &'static [f64],
            risk: f64,
        }
        let rows = [
            Row {
                probs: &[0.25; 4],
                costs: &[10.0, 20.0, 30.0, 40.0],
                alpha: 0.5,
                lambda: 0.5,
                weights: &[0.125, 0.125, 0.375, 0.375],
                risk: 30.0,
            },
            Row {
                probs: &[0.1, 0.2, 0.3, 0.4],
                costs: &[40.0, 30.0, 20.0, 10.0],
                alpha: 0.25,
                lambda: 0.5,
                weights: &[0.25, 0.40, 0.15, 0.20],
                risk: 27.0,
            },
            Row {
                probs: &[1.0],
                costs: &[42.0],
                alpha: 0.3,
                lambda: 0.6,
                weights: &[1.0],
                risk: 42.0,
            },
        ];

        for Row {
            probs,
            costs,
            alpha,
            lambda,
            weights,
            risk,
        } in rows
        {
            let outcomes: Vec<BackwardOutcome> = costs.iter().map(|&c| outcome(c, c)).collect();
            let mut from_outcomes = RiskMeasureScratch::new();
            compute_cvar_weights_into(&outcomes, probs, alpha, lambda, &mut from_outcomes);
            let mut from_costs = RiskMeasureScratch::new();
            compute_cvar_weights_from_costs_into(costs, probs, alpha, lambda, &mut from_costs);

            let rho = RiskMeasure::CVaR { alpha, lambda }.evaluate_risk(costs, probs);
            assert!(
                (rho - risk).abs() < 1e-12,
                "costs {costs:?} at alpha={alpha}, lambda={lambda}: risk must be {risk}, got {rho}"
            );
            assert_eq!(
                from_outcomes.mu, from_costs.mu,
                "both entry points must give identical weights for costs {costs:?}"
            );
            assert_eq!(from_costs.mu.len(), weights.len());
            for (i, (&got, &want)) in from_costs.mu.iter().zip(weights).enumerate() {
                assert!(
                    (got - want).abs() < 1e-12,
                    "costs {costs:?} at alpha={alpha}, lambda={lambda}: weight {i} must be \
                     {want}, got {got} (all weights {:?})",
                    from_costs.mu
                );
            }
        }
    }

    #[test]
    fn cvar_cost_ties_break_by_canonical_index() {
        use super::RiskMeasureScratch;

        let outcomes = vec![
            outcome_with_coeffs(10.0, 10.0, vec![0.0, 0.0, 0.0]),
            outcome_with_coeffs(30.0, 30.0, vec![1.0, 0.0, 0.0]),
            outcome_with_coeffs(30.0, 30.0, vec![0.0, 1.0, 0.0]),
            outcome_with_coeffs(30.0, 30.0, vec![0.0, 0.0, 1.0]),
        ];
        let probs = vec![0.25; 4];
        let rm = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 0.5,
        };

        let mut fresh = RiskMeasureScratch::new();
        let mut intercept = 0.0_f64;
        let mut coefficients = vec![0.0_f64; 3];
        rm.aggregate_cut_into(
            &outcomes,
            &probs,
            &mut intercept,
            &mut coefficients,
            &mut fresh,
        );

        let expected_weights = [0.125, 0.375, 0.375, 0.125];
        for (i, (&got, &want)) in fresh.mu.iter().zip(&expected_weights).enumerate() {
            assert!(
                (got - want).abs() < 1e-12,
                "tied weight {i} must be {want}, got {got} (all weights {:?})",
                fresh.mu
            );
        }
        assert!(
            (intercept - 27.5).abs() < 1e-12,
            "tied intercept must be 27.5, got {intercept}"
        );
        for (i, (&got, &want)) in coefficients.iter().zip(&[0.375, 0.375, 0.125]).enumerate() {
            assert!(
                (got - want).abs() < 1e-12,
                "tied coefficient {i} must be {want}, got {got} (all {coefficients:?})"
            );
        }

        let mut reused = RiskMeasureScratch::new();
        let wider: Vec<BackwardOutcome> = [5.0, 60.0, 15.0, 60.0, 25.0, 35.0]
            .iter()
            .map(|&c| outcome_with_coeffs(c, c, vec![c, -c, 1.0]))
            .collect();
        let mut wider_intercept = 0.0_f64;
        let mut wider_coefficients = vec![0.0_f64; 3];
        rm.aggregate_cut_into(
            &wider,
            &uniform(6),
            &mut wider_intercept,
            &mut wider_coefficients,
            &mut reused,
        );
        let mut reused_intercept = 0.0_f64;
        let mut reused_coefficients = vec![0.0_f64; 3];
        rm.aggregate_cut_into(
            &outcomes,
            &probs,
            &mut reused_intercept,
            &mut reused_coefficients,
            &mut reused,
        );
        assert_eq!(
            reused_intercept, intercept,
            "a scratch reused from a wider input must give the fresh-scratch intercept"
        );
        assert_eq!(
            reused_coefficients, coefficients,
            "a scratch reused from a wider input must give the fresh-scratch coefficients"
        );
    }

    #[test]
    fn aggregate_cut_into_applies_the_probability_floor() {
        use super::RiskMeasureScratch;

        let outcomes = vec![
            outcome_with_coeffs(10.0, 10.0, vec![1.0, 0.0]),
            outcome_with_coeffs(20.0, 20.0, vec![0.0, 1.0]),
            outcome_with_coeffs(30.0, 30.0, vec![1.0, 1.0]),
            outcome_with_coeffs(40.0, 40.0, vec![2.0, 0.0]),
        ];
        let probs = vec![0.25; 4];
        let rm = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 0.5,
        };

        let mut scratch = RiskMeasureScratch::new();
        let mut intercept = 0.0_f64;
        let mut coefficients = vec![0.0_f64; 2];
        rm.aggregate_cut_into(
            &outcomes,
            &probs,
            &mut intercept,
            &mut coefficients,
            &mut scratch,
        );

        assert!(
            (intercept - 30.0).abs() < 1e-12,
            "the floored cut intercept must be 30.0, got {intercept}"
        );
        for (i, (&got, &want)) in coefficients.iter().zip(&[1.25, 0.5]).enumerate() {
            assert!(
                (got - want).abs() < 1e-12,
                "floored cut coefficient {i} must be {want}, got {got} (all {coefficients:?})"
            );
        }
    }

    #[test]
    fn cvar_weights_reduce_bitwise_at_lambda_endpoints() {
        use super::{
            RiskMeasureScratch, compute_cvar_weights_from_costs_into, compute_cvar_weights_into,
        };

        let costs = [12.0, 40.0, 7.5, 40.0, 23.0, -3.0];
        let probs = [0.05_f64, 0.3, 0.1, 0.15, 0.25, 0.15];
        let alpha = 0.35;
        let outcomes: Vec<BackwardOutcome> = costs.iter().map(|&c| outcome(c, c)).collect();

        let mut order: Vec<usize> = (0..costs.len()).collect();
        order.sort_by(|&i, &j| costs[j].total_cmp(&costs[i]));
        let mut pure_greedy = vec![0.0_f64; costs.len()];
        let mut remaining = 1.0_f64;
        for &i in &order {
            if remaining <= 0.0 {
                break;
            }
            let alloc = (probs[i] / alpha).min(remaining);
            pure_greedy[i] = alloc;
            remaining -= alloc;
        }

        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
        let mut scratch = RiskMeasureScratch::new();
        for (lambda, expected, what) in [
            (
                1.0,
                &pure_greedy[..],
                "the pure CVaR greedy over caps p/alpha",
            ),
            (0.0, &probs[..], "the probabilities"),
        ] {
            compute_cvar_weights_from_costs_into(&costs, &probs, alpha, lambda, &mut scratch);
            assert_eq!(
                bits(&scratch.mu),
                bits(expected),
                "lambda = {lambda} weights over costs must be {what}, bit for bit"
            );
            compute_cvar_weights_into(&outcomes, &probs, alpha, lambda, &mut scratch);
            assert_eq!(
                bits(&scratch.mu),
                bits(expected),
                "lambda = {lambda} weights over outcomes must be {what}, bit for bit"
            );
        }
    }
}

#[cfg(test)]
mod proptests {
    use proptest::prelude::*;
    use proptest::test_runner::RngSeed;

    use super::{RiskMeasureScratch, compute_cvar_weights_from_costs_into};

    /// Fixed cases/seed so a failing shrink is reproducible run-to-run.
    fn fixed_config() -> ProptestConfig {
        ProptestConfig {
            cases: 256,
            rng_seed: RngSeed::Fixed(42),
            ..ProptestConfig::default()
        }
    }

    /// `3..=8` openings: tie-prone costs, and positive raw weights normalised by
    /// their sum into probabilities.
    fn openings() -> impl Strategy<Value = (Vec<f64>, Vec<f64>)> {
        (3..=8usize)
            .prop_flat_map(|n| {
                (
                    prop::collection::vec(
                        prop_oneof![-1e3..1e3, (-3i32..=3).prop_map(f64::from)],
                        n,
                    ),
                    prop::collection::vec(1e-3..1.0_f64, n),
                )
            })
            .prop_map(|(costs, raw)| {
                let total: f64 = raw.iter().sum();
                (costs, raw.iter().map(|w| w / total).collect())
            })
    }

    /// `CVaR_α[z] = min_η {η + E[(z − η)⁺]/α}`, the minimum attained at some `η = z_i`.
    fn rockafellar_uryasev_cvar(costs: &[f64], probs: &[f64], alpha: f64) -> f64 {
        costs
            .iter()
            .map(|&eta| {
                let excess: f64 = costs
                    .iter()
                    .zip(probs)
                    .map(|(&z, &p)| p * (z - eta).max(0.0))
                    .sum();
                eta + excess / alpha
            })
            .fold(f64::INFINITY, f64::min)
    }

    proptest! {
        #![proptest_config(fixed_config())]

        #[test]
        fn cvar_weights_match_rockafellar_uryasev_oracle(
            (costs, probs) in openings(),
            alpha in prop_oneof![Just(1.0_f64), 1e-4..1.0_f64],
            lambda in 1e-6..1.0_f64,
        ) {
            let mut scratch = RiskMeasureScratch::new();
            compute_cvar_weights_from_costs_into(&costs, &probs, alpha, lambda, &mut scratch);
            let mu = &scratch.mu;

            let cost_scale = costs.iter().fold(1.0_f64, |m, z| m.max(z.abs()));
            let expectation: f64 = costs.iter().zip(&probs).map(|(z, p)| z * p).sum();
            let oracle = (1.0 - lambda) * expectation
                + lambda * rockafellar_uryasev_cvar(&costs, &probs, alpha);
            let weighted: f64 = costs.iter().zip(mu).map(|(z, m)| z * m).sum();
            prop_assert!(
                (weighted - oracle).abs() <= 1e-12 * cost_scale,
                "weighted cost {weighted} must equal the oracle {oracle} (mu {mu:?})"
            );

            for (i, (&m, &p)) in mu.iter().zip(&probs).enumerate() {
                let floor = (1.0 - lambda) * p;
                let cap = floor + lambda * p / alpha;
                prop_assert!(
                    m >= floor - 1e-12 && m <= cap + 1e-12 * cap.max(1.0),
                    "weight {i} = {m} must lie in [{floor}, {cap}]"
                );
            }

            let mass: f64 = mu.iter().sum();
            prop_assert!((mass - 1.0).abs() <= 1e-12, "weights must sum to 1, got {mass}");
        }
    }
}
