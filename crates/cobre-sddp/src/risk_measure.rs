//! Risk measure abstraction for cut aggregation and risk-adjusted cost evaluation.
//!
//! [`RiskMeasure`] is a flat enum with two variants — [`RiskMeasure::Expectation`]
//! and [`RiskMeasure::CVaR`] — dispatched via `match` at each backward pass stage.
//! This follows the enum dispatch pattern (DEC-001).
//!
//! ## Aggregation semantics
//!
//! The primary method [`RiskMeasure::aggregate_cut`] replaces opening probabilities
//! `p(ω)` with risk-adjusted weights `μ*_ω` and computes weighted sums of intercepts
//! and coefficients. For `Expectation`, `μ*_ω = p(ω)`. For `CVaR`, the weights are
//! computed via a sorting-based greedy allocation that places maximum mass on the
//! highest-cost scenarios (Risk Measures SS7).
//!
//! ## Risk evaluation
//!
//! [`RiskMeasure::evaluate_risk`] aggregates a vector of cost realizations into a
//! scalar risk-adjusted cost. For `CVaR`, the formula is:
//! `(1 - λ) · E[Z] + λ · CVaR_α[Z]`.
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

/// Results from solving one backward pass opening at a single stage.
///
/// Each opening produces an intercept and a coefficient vector derived
/// from the LP dual variables as described in Cut Management SS2.
/// The `objective_value` is used by [`RiskMeasure::CVaR`] to rank scenarios
/// by cost for the greedy weight allocation (Risk Measures SS7).
#[derive(Debug, Clone)]
pub struct BackwardOutcome {
    /// Per-scenario cut intercept `α_t(ω)`.
    pub intercept: f64,

    /// Per-scenario cut coefficients `π_t(ω)`, one per state variable.
    ///
    /// Length equals `state_dimension`. Must be the same length across
    /// all outcomes passed to a single `aggregate_cut` call.
    pub coefficients: Vec<f64>,

    /// Optimal objective value `Q_t(x̂, ω)` of the stage subproblem.
    ///
    /// Used to rank scenarios by cost when computing `CVaR` risk weights.
    /// A higher value indicates a worse (more expensive) scenario.
    pub objective_value: f64,
}

/// Risk measure for cut aggregation and risk-adjusted cost evaluation.
///
/// Each stage in the training loop holds one `RiskMeasure` value, resolved
/// from the `risk_measure` field in `stages.json` during configuration
/// loading. The enum is matched at each backward pass stage to select the
/// aggregation behaviour.
///
/// ## Variants
///
/// - [`RiskMeasure::Expectation`]: risk-neutral expected value. Aggregation
///   weights equal the opening probabilities: `μ*_ω = p(ω)`.
/// - [`RiskMeasure::CVaR`]: convex combination of expectation and `CVaR`.
///   Weights are computed by the greedy allocation described in Risk
///   Measures SS7.
///
/// ## Dispatch
///
/// Both variants are dispatched via `match` (enum dispatch, DEC-001).
/// `aggregate_cut` and `evaluate_risk` are pure query methods — they do
/// not return `Result` because all inputs are validated at configuration
/// load time.
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
#[derive(Debug, Clone)]
pub enum RiskMeasure {
    /// Risk-neutral expected value.
    ///
    /// Aggregation weights equal the opening probabilities: `μ*_ω = p(ω)`.
    /// Reduces to the standard single-cut aggregation from Cut Management SS3.
    Expectation,

    /// Convex combination of expectation and `CVaR`:
    /// `ρ^{λ,α}[Z] = (1 - λ) E[Z] + λ · CVaR_α[Z]`.
    ///
    /// See Risk Measures SS3 for the definition and Risk Measures SS7 for
    /// the weight computation procedure.
    CVaR {
        /// `CVaR` confidence level `α ∈ (0, 1]`.
        ///
        /// `α = 1` is equivalent to expectation. Smaller `α` values
        /// produce more risk-averse behaviour by concentrating weight on
        /// the worst `α`-fraction of scenarios.
        alpha: f64,

        /// Risk aversion weight `λ ∈ [0, 1]`.
        ///
        /// `λ = 0` reduces to `Expectation` (normalised at config load
        /// time). `λ = 1` gives pure `CVaR`.
        lambda: f64,
    },
}

impl RiskMeasure {
    /// Aggregate per-opening backward pass results into a single cut.
    ///
    /// Replaces the opening probabilities `p(ω)` with risk-adjusted weights
    /// `μ*_ω` and computes the weighted sum of per-opening intercepts and
    /// coefficients. This is the only difference from risk-neutral aggregation
    /// — the cut structure and LP insertion are identical.
    ///
    /// ## Preconditions
    ///
    /// - `outcomes.len() == probabilities.len()` (one probability per opening)
    /// - `outcomes.len() > 0` (at least one opening)
    /// - `probabilities` sum to `1.0` within floating-point tolerance
    /// - All `outcomes[i].coefficients` have equal length
    ///
    /// ## Returns
    ///
    /// `(aggregated_intercept, aggregated_coefficients)` where
    /// `aggregated_coefficients.len() == state_dimension`.
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

    /// Evaluate the risk-adjusted scalar cost from a vector of cost values.
    ///
    /// Used for convergence bound computation during the forward pass.
    /// For risk-neutral, this is the probability-weighted mean. For `CVaR`,
    /// this is the convex combination `(1-λ) E[Z] + λ · CVaR_α[Z]`.
    ///
    /// ## Preconditions
    ///
    /// - `costs.len() == probabilities.len()` (one probability per realization)
    /// - `costs.len() > 0` (at least one realization)
    /// - `probabilities` sum to `1.0` within floating-point tolerance
    ///
    /// ## Returns
    ///
    /// The risk-adjusted scalar cost (finite when all inputs are finite).
    #[must_use]
    pub fn evaluate_risk(&self, costs: &[f64], probabilities: &[f64]) -> f64 {
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
            RiskMeasure::Expectation => {
                // E[Z] = Σ p(ω) · Z(ω)
                costs.iter().zip(probabilities).map(|(c, p)| c * p).sum()
            }
            RiskMeasure::CVaR { alpha, lambda } => {
                // EAVaR = (1 - λ)·E[Z] + λ·CVaR_α[Z]
                //
                // By the dual representation (Risk Measures SS4.2), EAVaR equals
                // E_μ*[Z] where μ* maximises the weighted cost subject to the
                // per-scenario upper bounds μ̄_ω = (1-λ)·p_ω + λ·p_ω/α. The
                // greedy allocation (continuous knapsack on costs sorted descending)
                // produces this optimal μ*. Therefore:
                //   EAVaR = Σ μ*_ω · Z(ω)
                // where μ* is computed by `compute_cvar_weights_from_costs`.
                let mu = compute_cvar_weights_from_costs(costs, probabilities, *alpha, *lambda);
                costs.iter().zip(mu.iter()).map(|(c, w)| c * w).sum()
            }
        }
    }
}

/// Compute `CVaR` weights via greedy allocation (continuous knapsack on objective values).
fn compute_cvar_weights(
    outcomes: &[BackwardOutcome],
    probabilities: &[f64],
    alpha: f64,
    lambda: f64,
) -> Vec<f64> {
    let n = outcomes.len();
    let upper_bounds: Vec<f64> = probabilities
        .iter()
        .map(|&p| (1.0 - lambda) * p + lambda * p / alpha)
        .collect();

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        outcomes[j]
            .objective_value
            .partial_cmp(&outcomes[i].objective_value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut mu = vec![0.0_f64; n];
    let mut remaining = 1.0_f64;
    for &idx in &order {
        if remaining <= 0.0 {
            break;
        }
        let alloc = upper_bounds[idx].min(remaining);
        mu[idx] = alloc;
        remaining -= alloc;
    }
    mu
}

/// Compute `CVaR` weights via greedy allocation on scalar cost values.
fn compute_cvar_weights_from_costs(
    costs: &[f64],
    probabilities: &[f64],
    alpha: f64,
    lambda: f64,
) -> Vec<f64> {
    let n = costs.len();
    let upper_bounds: Vec<f64> = probabilities
        .iter()
        .map(|&p| (1.0 - lambda) * p + lambda * p / alpha)
        .collect();

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        costs[j]
            .partial_cmp(&costs[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut mu = vec![0.0_f64; n];
    let mut remaining = 1.0_f64;
    for &idx in &order {
        if remaining <= 0.0 {
            break;
        }
        let alloc = upper_bounds[idx].min(remaining);
        mu[idx] = alloc;
        remaining -= alloc;
    }
    mu
}

fn aggregate_weighted(outcomes: &[BackwardOutcome], weights: &[f64]) -> (f64, Vec<f64>) {
    let state_dim = outcomes.first().map_or(0, |o| o.coefficients.len());

    let mut agg_intercept = 0.0_f64;
    let mut agg_coefficients = vec![0.0_f64; state_dim];

    for (outcome, &w) in outcomes.iter().zip(weights) {
        agg_intercept += w * outcome.intercept;
        for (agg, &coeff) in agg_coefficients.iter_mut().zip(&outcome.coefficients) {
            *agg += w * coeff;
        }
    }

    (agg_intercept, agg_coefficients)
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)] // test helpers use small n values
mod tests {
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
            outcome(10.0, 10.0), // cheapest
            outcome(20.0, 20.0),
            outcome(30.0, 30.0),
            outcome(40.0, 40.0), // most expensive
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
            outcome_with_coeffs(10.0, 10.0, vec![1.0, 0.0]), // cheapest
            outcome_with_coeffs(20.0, 20.0, vec![0.0, 1.0]), // expensive
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
        // CVaR with alpha=1, lambda=1 should give same result as Expectation
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
        // lambda=0: upper bounds equal p, weights = p, same as Expectation
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
        // Verify that the computed risk weights always sum to 1.0
        // (This is an invariant of the greedy allocation.)
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
    fn risk_measure_debug_and_clone() {
        let rm = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 0.8,
        };
        let cloned = rm.clone();
        let debug_str = format!("{rm:?}");
        assert!(debug_str.contains("CVaR"));
        let _ = cloned;
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
}
