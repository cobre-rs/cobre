# Extension Points

## Purpose

This spec defines how Cobre selects, composes, and validates algorithm variants at configuration time. The training loop is parameterized by four abstraction points — risk measure, cut formulation, horizon mode, and sampling scheme — each with a defined set of variants. This spec maps each variant to its configuration source, lists validation rules, and documents cross-variant compatibility constraints.

For the behavioral contracts that the training loop imposes on each abstraction point, see [Training Loop SS3](./training-loop.md). For the mathematical definitions of each variant, see the referenced math specs.

## 1. Variant Architecture Overview

The training loop has a fixed structure (forward pass, backward pass, MPI synchronization, convergence monitoring) parameterized by four configurable abstraction points:

| Abstraction Point   | Determines                                                 | Configured In | Current Variants                                  |
| ------------------- | ---------------------------------------------------------- | ------------- | ------------------------------------------------- |
| **Risk Measure**    | How backward outcomes are aggregated into cut coefficients | `stages.json` | Expectation, CVaR (SS2)                           |
| **Cut Formulation** | Structure of cuts added to the FCF                         | Fixed         | Single-cut (SS3)                                  |
| **Horizon Mode**    | Stage transitions, terminal conditions, discount factors   | `stages.json` | Finite (SS4); Cyclic — deferred                   |
| **Sampling Scheme** | How the forward pass selects scenario realizations         | `config.json` | InSample, External, Historical, OutOfSample (SS5) |

**Fixed components** (not configurable — same behavior regardless of variant selection):

- Forward pass execution and parallel distribution ([Training Loop SS4](./training-loop.md))
- Backward pass execution and stage synchronization ([Training Loop SS6](./training-loop.md))
- MPI cut synchronization ([Cut Management Implementation SS4](./cut-management-impl.md))
- Convergence monitoring and stopping rules ([Convergence Monitoring](./convergence-monitoring.md))
- LP construction and solver interaction ([Solver Abstraction](./solver-abstraction.md))

## 2. Risk Measure Variants

The risk measure determines how backward pass outcomes (one per opening) are aggregated into a single cut. It can vary per stage (configured in each stage's `risk_measure` field in `stages.json`).

### 2.1 Variant Table

| Variant         | Config Value                              | Behavior                                                                                                                                                                            | Math Reference                                     |
| --------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| **Expectation** | `"expectation"`                           | Probability-weighted average of per-outcome intercepts and gradients. Weights equal the uniform opening probabilities $p(\omega)$.                                                  | [Risk Measures SS1](../math/risk-measures.md)      |
| **CVaR**        | `{"cvar": {"alpha": ..., "lambda": ...}}` | Convex combination: $(1-\lambda)\mathbb{E}[\cdot] + \lambda \cdot \text{CVaR}_\alpha[\cdot]$. Sorting-based greedy weight allocation places maximum weight on worst-cost scenarios. | [Risk Measures SS3, SS7](../math/risk-measures.md) |

### 2.2 Configuration Mapping

The `risk_measure` field appears per stage in `stages.json` (see [Input Scenarios SS1.7](../data-model/input-scenarios.md)):

```json
{
  "stages": [
    { "id": 0, "risk_measure": { "cvar": { "alpha": 0.95, "lambda": 0.5 } } },
    { "id": 1, "risk_measure": "expectation" }
  ]
}
```

### 2.3 Validation Rules

| Rule | Condition                                   | Error                                                                     |
| ---- | ------------------------------------------- | ------------------------------------------------------------------------- |
| R1   | `alpha` must be in $(0, 1]$                 | `alpha=0` produces undefined CVaR; `alpha=1` is equivalent to expectation |
| R2   | `lambda` must be in $[0, 1]$                | Weight outside valid range                                                |
| R3   | `lambda=0` is equivalent to `"expectation"` | Accepted (no error), but normalized to `"expectation"` internally         |

### 2.4 Behavioral Contract

Each risk measure variant must provide two operations consumed by the training loop:

1. **Cut aggregation** — Given $N_{\text{openings}}$ backward outcomes with their probabilities, produce a single set of cut coefficients $(\alpha, \pi)$. For Expectation, the weights are uniform. For CVaR, the weights are computed via the sorting-based greedy allocation from [Risk Measures SS7](../math/risk-measures.md).

2. **Risk evaluation** — Given a set of cost values and probabilities, produce a scalar risk-adjusted cost. Used for convergence bound computation.

> **Bound validity warning**: When CVaR is active, the first-stage LP objective is a convergence indicator only — it is NOT a valid lower bound. See [Risk Measures SS10](../math/risk-measures.md).

### 2.5 Deferred Variants

No additional risk measure variants are currently planned. The Expectation + CVaR convex combination covers the standard risk-averse SDDP formulation. Entropic risk measures would require a different dual representation and are not in scope.

## 3. Cut Formulation Variants

The cut formulation determines the structure of cuts added to the FCF at each backward pass stage.

### 3.1 Variant Table

| Variant        | Status   | Behavior                                                                                                       | Math Reference                                    |
| -------------- | -------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| **Single-cut** | Current  | One aggregated cut per iteration per stage. $\theta \geq \bar{\alpha} + \bar{\pi}^\top x$.                     | [SDDP Algorithm SS6.1](../math/sddp-algorithm.md) |
| **Multi-cut**  | Deferred | One cut per opening per iteration. $\theta_\omega \geq \alpha(\omega) + \pi(\omega)^\top x$ for each $\omega$. | [Deferred Features SSC.3](../deferred.md)         |

### 3.2 Configuration

No configuration required — single-cut is the only available formulation. When multi-cut is implemented, it will be selectable via `config.json` → `training.cut_formulation`.

## 4. Horizon Mode Variants

The horizon mode determines stage traversal, terminal conditions, and discount factors. It is derived from the `policy_graph` field in `stages.json`.

### 4.1 Variant Table

| Variant    | Status   | Config Trigger                                     | Behavior                                                                                                                                    | Math Reference                                    |
| ---------- | -------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| **Finite** | Current  | `policy_graph.type = "finite_horizon"` (or absent) | Linear chain $1 \to 2 \to \cdots \to T$. Terminal value $V_{T+1} = 0$. No cycle, no mandatory discount.                                     | [SDDP Algorithm SS4.1](../math/sddp-algorithm.md) |
| **Cyclic** | Deferred | `policy_graph.type = "cyclic"`                     | At least one transition creates a cycle (source_id > target_id or equal). Cut pools organized by season. Discount required for convergence. | [Infinite Horizon](../math/infinite-horizon.md)   |

### 4.2 Configuration Mapping

The horizon mode is implicitly selected by the `policy_graph` structure in `stages.json` (see [Input Scenarios SS1.2](../data-model/input-scenarios.md)):

**Finite horizon:**

```json
{
  "policy_graph": {
    "type": "finite_horizon",
    "annual_discount_rate": 0.06,
    "transitions": [
      { "source_id": 0, "target_id": 1, "probability": 1.0 },
      { "source_id": 1, "target_id": 2, "probability": 1.0 }
    ]
  }
}
```

**Cyclic (infinite periodic) horizon:**

```json
{
  "policy_graph": {
    "type": "cyclic",
    "annual_discount_rate": 0.06,
    "transitions": [
      { "source_id": 0, "target_id": 1, "probability": 1.0 },
      { "source_id": 59, "target_id": 48, "probability": 1.0 }
    ]
  }
}
```

### 4.3 Validation Rules

| Rule | Condition                                                                                                 | Error                                        |
| ---- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| H1   | At least one stage must exist                                                                             | Empty stage set                              |
| H2   | For `cyclic`: cumulative cycle discount $d_{\text{cycle}} = \prod_{t \in \text{cycle}} d_{t \to t+1} < 1$ | Cycle discount $\geq 1$ prevents convergence |
| H3   | For `cyclic`: cycle start stage must exist in the stage set                                               | Cycle start out of bounds                    |
| H4   | All transition target stages must exist                                                                   | Dangling transition                          |

### 4.4 Behavioral Contract

Each horizon mode variant must provide:

1. **Successor computation** — Given a stage, return successor stages with transition probabilities
2. **Terminal detection** — Whether a stage has no successors (finite only; cyclic stages are never terminal)
3. **Discount factor** — The discount factor $d_{t \to t+1}$ for each transition. For finite horizon, this comes from the per-transition or global `annual_discount_rate` (may be 1.0 if undiscounted). For cyclic, the back-edge transition always applies the discount factor.
4. **Validation** — Verify stage configuration is valid for the selected mode

> **Forward pass termination in cyclic mode**: The forward pass simulates multiple cycles until the cumulative discount drops below a tolerance threshold or a `max_horizon_length` safety bound is reached. See [Infinite Horizon SS6](../math/infinite-horizon.md).

## 5. Sampling Scheme Variants

The sampling scheme determines how the forward pass selects scenario realizations at each stage. This is the fourth abstraction point defined in [Training Loop SS3.4](./training-loop.md).

### 5.1 Variant Table

| Variant         | Config Value      | Forward Noise Source                                         | Backward Noise Source                         | Math Reference                                      |
| --------------- | ----------------- | ------------------------------------------------------------ | --------------------------------------------- | --------------------------------------------------- |
| **InSample**    | `"in_sample"`     | Random index from fixed opening tree                         | Same opening tree                             | [Scenario Generation SS3](./scenario-generation.md) |
| **External**    | `"external"`      | User-provided `external_scenarios.parquet`                   | Opening tree from PAR fitted to external data | [Scenario Generation SS4](./scenario-generation.md) |
| **Historical**  | `"historical"`    | `inflow_history.parquet` mapped to stages                    | Opening tree from PAR fitted to history       | [Scenario Generation SS3](./scenario-generation.md) |
| **OutOfSample** | `"out_of_sample"` | Fresh PAR-generated scenarios (independent of training tree) | Opening tree from PAR model                   | [Scenario Generation SS3](./scenario-generation.md) |

### 5.2 Configuration Mapping

The `scenario_source` field in `config.json` (see [Configuration Reference](../configuration/configuration-reference.md)):

```json
{ "scenario_source": { "sampling_scheme": "in_sample", "seed": 42 } }
```

```json
{
  "scenario_source": {
    "sampling_scheme": "external",
    "selection_mode": "random"
  }
}
```

```json
{ "scenario_source": { "sampling_scheme": "historical" } }
```

### 5.3 Validation Rules

| Rule | Condition                                                             | Error                                  |
| ---- | --------------------------------------------------------------------- | -------------------------------------- |
| S1   | `in_sample` requires `seed`                                           | Reproducibility requires explicit seed |
| S2   | `external` requires `external_scenarios.parquet` in input directory   | Missing external scenario file         |
| S3   | `historical` requires `inflow_history.parquet` in input directory     | Missing historical inflow file         |
| S4   | `external` with `selection_mode` must be `"random"` or `"sequential"` | Invalid selection mode                 |

### 5.4 Key Invariant

The backward pass always uses the fixed opening tree, regardless of the forward sampling scheme. The separation of forward and backward noise sources is fundamental to SDDP correctness — see [Scenario Generation SS3.1](./scenario-generation.md).

## 6. Variant Selection Pipeline

At startup, the variant selection pipeline resolves configuration into concrete algorithm behavior. This happens during the initialization phase described in [CLI and Lifecycle](./cli-and-lifecycle.md):

| Step | Action                                | Input                                   | Output                           |
| ---- | ------------------------------------- | --------------------------------------- | -------------------------------- |
| 1    | Parse `config.json` and `stages.json` | Raw JSON files                          | Typed configuration structures   |
| 2    | Select horizon mode                   | `policy_graph.type`                     | Finite or Cyclic mode            |
| 3    | Select per-stage risk measures        | `stages[].risk_measure`                 | Risk measure per stage           |
| 4    | Select sampling scheme                | `scenario_source.sampling_scheme`       | Sampling scheme                  |
| 5    | Validate individual variants          | Per-variant rules (SS2.3, SS4.3, SS5.3) | Reject invalid configurations    |
| 6    | Validate cross-variant composition    | Composition rules (SS8)                 | Reject incompatible combinations |
| 7    | Construct training loop               | All resolved variants                   | Parameterized training loop      |

The risk measure is per-stage (step 3 produces a mapping from stage ID to risk measure variant). All other abstraction points are global (one selection for the entire run).

## 7. Dispatch Mechanism

**Open design point**: The dispatch mechanism — how variant-specific behavior is invoked at runtime — is an implementation decision deferred to coding time.

| Option                            | Description                                                                                                                                | Trade-off                                                                                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Compile-time monomorphization** | Generic training loop `TrainingLoop<R, C, H, S>` instantiated with concrete types. Variant selected via `cfg` features or top-level match. | Zero overhead. No dynamic dispatch cost. But: per-stage risk measure variation requires a different approach (can't monomorphize per stage).    |
| **Enum dispatch**                 | Each abstraction point uses a flat enum (e.g., `enum RiskMeasure { Expectation, CVaR { alpha, lambda } }`). Match at each call site.       | No heap allocation, inlineable, supports per-stage variation naturally. Small match overhead at each backward pass iteration.                   |
| **Trait objects**                 | `Box<dyn RiskMeasure>` with dynamic dispatch.                                                                                              | Maximum flexibility, supports arbitrary nesting (e.g., convex combination of arbitrary inner risk measures). Virtual call overhead on hot path. |

The per-stage risk measure requirement (SS2.2) rules out pure compile-time monomorphization for the risk measure dimension. Enum dispatch is the natural fit for a small, fixed set of variants.

> **Note**: The solver abstraction uses compile-time `cfg`-feature selection for the LP solver backend (CLP vs HiGHS) — see [Solver Abstraction SS10](./solver-abstraction.md). The algorithm variant dispatch mechanism is a separate decision and need not use the same approach.

## 8. Variant Composition Validation

Some variant combinations have specific interactions that require validation or special handling.

### 8.1 Cross-Variant Rules

| Rule | Combination                             | Constraint                                                              | Rationale                                                                           |
| ---- | --------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| X1   | CVaR + Cyclic horizon                   | Convergence indicator only (not a valid lower bound)                    | [Risk Measures SS10](../math/risk-measures.md): risk-averse LB is not a valid bound |
| X2   | CVaR + Simulation stopping rule         | Simulation-based stopping must use risk-adjusted forward costs          | The upper bound estimate requires risk-adjusted evaluation                          |
| X3   | External/Historical + Cyclic            | Forward pass cycle wrapping must handle external scenario index mapping | External scenarios may not align with cycle boundaries                              |
| X4   | Any risk measure + Multi-cut (deferred) | Per-opening risk adjustment not applicable with multi-cut               | Multi-cut produces one cut per opening, bypassing risk aggregation                  |

### 8.2 Default Configuration

When fields are omitted, the following defaults apply:

| Abstraction Point | Default            | Rationale                                     |
| ----------------- | ------------------ | --------------------------------------------- |
| Risk measure      | `"expectation"`    | Risk-neutral is the standard SDDP formulation |
| Cut formulation   | Single-cut         | Only available formulation                    |
| Horizon mode      | `"finite_horizon"` | Standard finite horizon, $V_{T+1} = 0$        |
| Sampling scheme   | `"in_sample"`      | PAR-generated opening tree                    |

## Cross-References

- [Training Loop SS3](./training-loop.md) — Behavioral contracts for the four abstraction points that this spec maps variants to
- [Training Loop SS2](./training-loop.md) — Training orchestrator components and iteration lifecycle
- [Risk Measures](../math/risk-measures.md) — Mathematical definitions: CVaR (SS2), convex combination (SS3), cut weight computation (SS7), lower bound invalidity (SS10)
- [SDDP Algorithm SS4](../math/sddp-algorithm.md) — Policy graph topologies (finite SS4.1, cyclic SS4.2)
- [SDDP Algorithm SS6](../math/sddp-algorithm.md) — Single-cut (SS6.1) and multi-cut (SS6.2) formulations
- [Infinite Horizon](../math/infinite-horizon.md) — Cyclic policy graph: periodic structure, cut sharing, forward/backward pass behavior, convergence
- [Discount Rate](../math/discount-rate.md) — Discount factor mechanics for horizon modes
- [Scenario Generation SS3](./scenario-generation.md) — Sampling scheme abstraction, forward/backward noise source separation
- [Input Scenarios SS1.2](../data-model/input-scenarios.md) — `policy_graph` schema for horizon mode configuration
- [Input Scenarios SS1.7](../data-model/input-scenarios.md) — `risk_measure` field schema for per-stage risk configuration
- [Input Scenarios SS2.1](../data-model/input-scenarios.md) — `scenario_source` schema for sampling scheme configuration
- [Configuration Reference SS5, SS6.2-SS6.3](../configuration/configuration-reference.md) — Horizon mode, risk measure, and scenario source config tables
- [Solver Abstraction SS10](./solver-abstraction.md) — Compile-time solver selection (contrasted with algorithm variant dispatch)
- [Deferred Features](../deferred.md) — Multi-cut (C.3), Monte Carlo backward sampling (C.14)
