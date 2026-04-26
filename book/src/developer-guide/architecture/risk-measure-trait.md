# Risk Measure Trait

## Purpose

This spec defines the `RiskMeasure` abstraction -- the enum-based trait through which the SDDP backward pass aggregates per-opening outcomes into cut coefficients and evaluates risk-adjusted costs for convergence monitoring. The risk measure is the mechanism by which scenario probabilities $p(\omega)$ are replaced with risk-adjusted weights $\mu^*_\omega$ during cut aggregation, as defined in [Risk Measures SS7](../math/risk-measures.md). Because the risk measure can vary per stage (configured via the `risk_measure` field in `stages.json`), the abstraction uses enum dispatch rather than compile-time monomorphization, following the analysis in [Extension Points SS7](./extension-points.md).

> **Convention: Rust traits as specification guidelines.** The Rust trait definitions, method signatures, and struct declarations throughout this specification corpus serve as _guidelines for implementation_, not as absolute source-of-truth contracts that must be reproduced verbatim. Their purpose is twofold: (1) to express behavioral contracts, preconditions, postconditions, and type-level invariants more precisely than prose alone, and (2) to anchor conformance test suites that verify backend interchangeability (see [Backend Testing §1](../hpc/backend-testing.md)). Implementation may diverge in naming, parameter ordering, error representation, or internal organization when practical considerations demand it -- provided the behavioral contracts and conformance tests continue to pass. When a trait signature and a prose description conflict, the prose description (which captures the domain intent) takes precedence; the conflict should be resolved by updating the trait signature. This convention applies to all trait-bearing specification documents in `src/specs/`.

## 1. Trait Definition

The risk measure is modeled as a flat enum with two variants, matching the two risk measure types supported by Cobre ([Extension Points SS2.1](./extension-points.md)):

```rust
/// Risk measure for cut aggregation and risk-adjusted cost evaluation.
///
/// Each stage in the training loop holds one `RiskMeasure` value, resolved
/// from the `risk_measure` field in `stages.json` during configuration
/// loading (see Extension Points SS6). The enum is matched at each backward
/// pass stage to select the aggregation behavior.
#[derive(Debug, Clone)]
pub enum RiskMeasure {
    /// Risk-neutral expected value.
    ///
    /// Aggregation weights equal the opening probabilities: μ*_ω = p(ω).
    /// This reduces to the standard single-cut aggregation from
    /// [Cut Management SS3](../math/cut-management.md).
    Expectation,

    /// Convex combination of expectation and CVaR:
    /// ρ^{λ,α}[Z] = (1 - λ) E[Z] + λ · CVaR_α[Z]
    ///
    /// See [Risk Measures SS3](../math/risk-measures.md) for the definition
    /// and [Risk Measures SS7](../math/risk-measures.md) for the weight
    /// computation procedure.
    CVaR {
        /// CVaR confidence level α ∈ (0, 1].
        /// α = 1 is equivalent to expectation.
        alpha: f64,

        /// Risk aversion weight λ ∈ [0, 1].
        /// λ = 0 reduces to Expectation (normalized at config load time).
        lambda: f64,
    },
}

impl RiskMeasure {
    /// Aggregate per-opening backward pass results into a single cut.
    ///
    /// Replaces the opening probabilities p(ω) with risk-adjusted weights
    /// μ*_ω and computes the weighted sum of per-opening intercepts and
    /// coefficients. This is the ONLY difference from risk-neutral
    /// aggregation -- the cut structure and LP insertion are identical.
    pub fn aggregate_cut(
        &self,
        outcomes: &[BackwardOutcome],
        probabilities: &[f64],
    ) -> (f64, Vec<f64>) {
        // Implementation dispatches on variant:
        // - Expectation: μ*_ω = p(ω) (direct weighted sum)
        // - CVaR: μ*_ω computed via sorting-based greedy allocation
        todo!()
    }

    /// Evaluate the risk-adjusted scalar cost from a vector of cost values.
    ///
    /// Used for convergence bound computation during the forward pass.
    /// For risk-neutral, this is the probability-weighted mean. For CVaR,
    /// this is the convex combination (1-λ)E[Z] + λ·CVaR_α[Z].
    pub fn evaluate_risk(
        &self,
        costs: &[f64],
        probabilities: &[f64],
    ) -> f64 {
        todo!()
    }
}
```

The `BackwardOutcome` type holds the per-opening results from the backward pass:

```rust
/// Results from solving one backward pass opening at a single stage.
///
/// Each opening produces an intercept and a coefficient vector, derived
/// from the LP dual variables as described in [Cut Management SS2].
pub struct BackwardOutcome {
    /// Per-scenario cut intercept α_t(ω).
    pub intercept: f64,

    /// Per-scenario cut coefficients π_t(ω), one per state variable.
    /// Length equals `state_dimension`.
    pub coefficients: Vec<f64>,

    /// Optimal objective value Q_t(x̂, ω) of the stage subproblem.
    /// Used for risk weight computation (sorting by cost).
    pub objective_value: f64,
}
```

## 2. Method Contracts

### 2.1 aggregate_cut

`aggregate_cut` is the primary method consumed by the backward pass. It is called once per visited state per stage, after all openings at that stage have been solved. The method replaces the uniform scenario probabilities $p(\omega)$ with risk-adjusted weights $\mu^*_\omega$ and computes the weighted aggregation. The resulting cut coefficients are written into the FCF cut pool at the deterministic slot ([Cut Management Implementation SS1.2](./cut-management-impl.md)).

**Preconditions:**

| Condition                                        | Description                                |
| ------------------------------------------------ | ------------------------------------------ |
| `outcomes.len() == probabilities.len()`          | One probability per opening                |
| `outcomes.len() > 0`                             | At least one opening                       |
| `probabilities` sum to 1.0 (within tolerance)    | Valid probability distribution             |
| All `outcomes[i].coefficients` have equal length | Consistent state dimension across openings |
| All `outcomes[i].objective_value` are finite     | No NaN or infinity from LP solves          |

**Postconditions:**

| Condition                                                                                  | Description                                                                                                                                                                                       |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Return `(intercept, coefficients)` where `coefficients.len() == state_dimension`           | Aggregated cut has correct dimension                                                                                                                                                              |
| For `Expectation`: result equals probability-weighted sum                                  | $\bar{\alpha} = \sum_\omega p(\omega) \cdot \hat{\alpha}(\omega)$, $\bar{\pi}_h = \sum_\omega p(\omega) \cdot \pi_h(\omega)$ -- identical to [Cut Management SS3](../math/cut-management.md)      |
| For `CVaR`: result equals risk-weighted sum using $\mu^*_\omega$                           | $\bar{\alpha} = \sum_\omega \mu^*_\omega \cdot \hat{\alpha}(\omega)$, $\bar{\pi}_h = \sum_\omega \mu^*_\omega \cdot \pi_h(\omega)$ -- as defined in [Risk Measures SS7](../math/risk-measures.md) |
| Risk-adjusted weights $\mu^*_\omega$ satisfy $\sum_\omega \mu^*_\omega = 1$                | Weights form a valid probability distribution                                                                                                                                                     |
| Risk-adjusted weights satisfy $0 \leq \mu^*_\omega \leq \bar{\mu}_\omega$ for all $\omega$ | Each weight is bounded by the per-scenario upper bound from [Risk Measures SS7](../math/risk-measures.md)                                                                                         |

**Behavioral contract for weight computation:** When the variant is `CVaR { alpha, lambda }`, the risk-adjusted weights $\mu^*_\omega$ must be the optimal solution to the dual LP from [Risk Measures SS4.2](../math/risk-measures.md). The per-scenario upper bound is $\bar{\mu}_\omega = (1 - \lambda) p_\omega + \lambda p_\omega / \alpha$, and the optimal weights place maximum mass on the highest-cost scenarios. The math spec establishes that a sorting-based greedy allocation produces the same result as the dual LP ([Risk Measures SS7, equivalence note](../math/risk-measures.md)). The implementation may use either approach.

**Infallibility:** This method does not return `Result`. All inputs are validated at configuration load time ([Extension Points SS6](./extension-points.md)), and the LP solver guarantees finite objective values when relatively complete recourse holds ([Cut Management SS4](../math/cut-management.md)).

### 2.2 evaluate_risk

`evaluate_risk` computes a scalar risk-adjusted cost from a vector of cost realizations. It is used during convergence bound computation: the forward pass produces cost samples across scenarios, and `evaluate_risk` aggregates them into the risk-adjusted objective value reported in convergence logs.

**Preconditions:**

| Condition                                     | Description                          |
| --------------------------------------------- | ------------------------------------ |
| `costs.len() == probabilities.len()`          | One probability per cost realization |
| `costs.len() > 0`                             | At least one realization             |
| `probabilities` sum to 1.0 (within tolerance) | Valid probability distribution       |
| All `costs[i]` are finite                     | No NaN or infinity                   |

**Postconditions:**

| Condition                                                                                     | Description                                                                    |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Return is finite                                                                              | Valid scalar result                                                            |
| For `Expectation`: result equals $\sum_\omega p(\omega) \cdot Z_\omega$                       | Probability-weighted mean                                                      |
| For `CVaR`: result equals $(1 - \lambda) \mathbb{E}[Z] + \lambda \cdot \text{CVaR}_\alpha[Z]$ | Convex combination as defined in [Risk Measures SS3](../math/risk-measures.md) |
| `Expectation` result $\leq$ `CVaR` result (when $\alpha < 1$)                                 | CVaR places more weight on worst outcomes                                      |

**Bound validity warning:** When CVaR is active on any stage, the first-stage LP objective is a convergence indicator only -- it is NOT a valid lower bound on the true risk-averse optimal cost. See [Risk Measures SS10](../math/risk-measures.md) for the full explanation.

**Infallibility:** Same rationale as `aggregate_cut` -- inputs are validated at configuration load time.

## 3. Supporting Types

### 3.1 StageRiskConfig

The `StageRiskConfig` enum represents the deserialized form of the `risk_measure` field in `stages.json` ([Input Scenarios SS1.7](../data-model/input-scenarios.md)). It maps directly to the JSON schema:

```rust
/// Configuration representation of the risk measure, matching the
/// `risk_measure` field in `stages.json`.
///
/// Deserialized from either the string `"expectation"` or the object
/// `{"cvar": {"alpha": ..., "lambda": ...}}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageRiskConfig {
    /// Risk-neutral expected value.
    Expectation,

    /// CVaR with parameters alpha and lambda.
    CVaR {
        alpha: f64,
        lambda: f64,
    },
}
```

**Conversion:** `StageRiskConfig` is validated and converted to `RiskMeasure` during configuration loading (step 3 of the variant selection pipeline in [Extension Points SS6](./extension-points.md)). The validation rules are specified in SS5.

### 3.2 Per-Stage Risk Measure Array

Because the risk measure can vary per stage ([Risk Measures SS8](../math/risk-measures.md)), the training loop holds a per-stage array:

```rust
/// Per-stage risk measure configuration, indexed by stage ID.
///
/// Created during initialization by mapping each stage's `risk_measure`
/// field to a validated `RiskMeasure` enum value.
pub type StageRiskMeasures = Vec<RiskMeasure>;
```

The backward pass indexes into this array at each stage to select the appropriate aggregation behavior.

## 4. Dispatch Mechanism

The risk measure uses **enum dispatch** -- a `match` on the `RiskMeasure` variant at each call site in the backward pass. This is the natural choice for a small, fixed set of variants where per-stage variation is required.

**Why not compile-time monomorphization:** The risk measure can differ between stages (e.g., stage 0 uses `CVaR { alpha: 0.95, lambda: 0.5 }`, stage 1 uses `Expectation`). A generic type parameter `R: RiskMeasureTrait` would fix a single risk measure for the entire training loop, which cannot represent per-stage variation. Encoding the per-stage array as `Vec<Box<dyn RiskMeasureTrait>>` would introduce heap allocation and virtual dispatch on the hot path. Enum dispatch avoids both limitations: the `Vec<RiskMeasure>` is a flat vector of enum-sized elements, and the `match` is inlineable by the compiler.

**Why not trait objects:** The variant set is closed (Expectation and CVaR only, with no additional variants planned -- see [Extension Points SS2.5](./extension-points.md)). Trait objects add indirection cost without the extensibility benefit. The enum approach is consistent with the dispatch analysis in [Extension Points SS7](./extension-points.md).

**Performance characteristics:** The `match` statement in `aggregate_cut` executes once per visited state per backward stage. At production scale (192 forward passes, 60 stages), this is at most ~11,520 match dispatches per iteration -- negligible compared to the LP solve cost that dominates the backward pass.

## 5. Validation Rules

The following validation rules apply to `StageRiskConfig` during configuration loading. These reproduce rules R1-R3 from [Extension Points SS2.3](./extension-points.md):

| Rule | Condition                                   | Error                                                                     |
| ---- | ------------------------------------------- | ------------------------------------------------------------------------- |
| R1   | `alpha` must be in $(0, 1]$                 | `alpha=0` produces undefined CVaR; `alpha=1` is equivalent to expectation |
| R2   | `lambda` must be in $[0, 1]$                | Weight outside valid range                                                |
| R3   | `lambda=0` is equivalent to `"expectation"` | Accepted (no error), but normalized to `Expectation` internally           |

Validation is performed once during the variant selection pipeline ([Extension Points SS6](./extension-points.md), step 5). After validation, the `RiskMeasure` enum values are guaranteed to satisfy these constraints for the entire training run. This is why `aggregate_cut` and `evaluate_risk` are infallible -- they operate on validated inputs.

**R3 normalization detail:** When `lambda=0` is detected in a `CVaR` config, the conversion step produces `RiskMeasure::Expectation` rather than `RiskMeasure::CVaR { alpha, lambda: 0.0 }`. This avoids the unnecessary sorting overhead in the CVaR weight computation path when the result would be identical to the expectation.

## 6. Special Cases

The three special cases from [Risk Measures SS7](../math/risk-measures.md) govern the weight computation behavior:

### 6.1 Risk-Neutral ($\lambda = 0$)

When $\lambda = 0$, the per-scenario weight upper bound reduces to $\bar{\mu}_\omega = p_\omega$, so the risk-adjusted weights equal the original probabilities: $\mu^*_\omega = p(\omega)$. The aggregation is identical to the standard single-cut formula from [Cut Management SS3](../math/cut-management.md). Per validation rule R3 (SS5), this case is normalized to the `Expectation` variant at config load time, so the CVaR code path is never entered.

### 6.2 Pure CVaR ($\lambda = 1$)

When $\lambda = 1$, the weight upper bound is $\bar{\mu}_\omega = p_\omega / \alpha$. Since $\alpha < 1$ implies $\sum_\omega \bar{\mu}_\omega = 1/\alpha > 1$, only the worst $\alpha$-fraction of scenarios receive non-zero weight. All remaining scenarios get $\mu^*_\omega = 0$ -- they do not contribute to the aggregated cut at all. This produces the most risk-averse cuts, focusing entirely on tail scenarios.

### 6.3 Convex Combination ($0 < \lambda < 1$)

The general case: every scenario receives at least the floor weight $(1 - \lambda) p_\omega > 0$ from the expectation component, but the worst scenarios receive additional weight up to $\bar{\mu}_\omega = (1 - \lambda) p_\omega + \lambda p_\omega / \alpha$. The sorting-based greedy allocation distributes the excess capacity (above the floor) to the highest-cost scenarios first.

This is the most common configuration in practice. It balances tail-risk protection (from the CVaR component) with stability (from the expectation floor that ensures all scenarios contribute).

## Cross-References

- [Risk Measures](../math/risk-measures.md) -- Mathematical definitions: CVaR (SS2), convex combination (SS3), dual representation (SS4), risk-averse subgradient theorem (SS5), cut generation with risk measures (SS7), per-stage profiles (SS8), lower bound invalidity (SS10)
- [Cut Management](../math/cut-management.md) -- Dual extraction (SS2), single-cut aggregation (SS3) that risk-neutral weights reproduce, cut validity conditions (SS4)
- [Cut Management Implementation](./cut-management-impl.md) -- FCF runtime structure (SS1), deterministic slot assignment (SS1.2), cross-rank cut synchronization (SS4)
- [Extension Points](./extension-points.md) -- Risk measure variant table (SS2.1), configuration mapping (SS2.2), validation rules R1-R3 (SS2.3), behavioral contract (SS2.4), dispatch mechanism analysis (SS7), variant selection pipeline (SS6)
- [Training Loop](./training-loop.md) -- Backward pass (SS6) where `aggregate_cut` is invoked; behavioral contracts for abstraction points (SS3)
- [Input Scenarios SS1.7](../data-model/input-scenarios.md) -- JSON schema for `risk_measure` field: `"expectation"` or `{"cvar": {"alpha": ..., "lambda": ...}}`
- [Communicator Trait](../hpc/communicator-trait.md) -- Reference pattern for trait specification structure, convention blockquote, and method contract format
- [Solver Abstraction SS10](./solver-abstraction.md) -- Compile-time solver selection pattern (contrasted with the enum dispatch used here)
