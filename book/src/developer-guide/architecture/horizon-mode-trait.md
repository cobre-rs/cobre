# Horizon Mode Trait

## Purpose

This spec defines the `HorizonMode` abstraction -- the enum-based trait through which the SDDP training loop determines stage traversal order, successor transitions with associated probabilities and discount factors, terminal conditions, forward pass termination logic, and cut pool organization. The horizon mode is a global property of the training run (one mode per execution), derived from the `policy_graph` field in `stages.json` ([Input Scenarios SS1.2](../data-model/input-scenarios.md)). Unlike the risk measure (which varies per stage), the horizon mode governs the structural topology of the entire policy graph, so a single enum value suffices for the full run. The two supported modes -- Finite and Cyclic -- correspond to the acyclic and infinite periodic horizon topologies described in [SDDP Algorithm SS4](../math/sddp-algorithm.md) and [Infinite Horizon](../math/infinite-horizon.md).

> **Convention: Rust traits as specification guidelines.** The Rust trait definitions, method signatures, and struct declarations throughout this specification corpus serve as _guidelines for implementation_, not as absolute source-of-truth contracts that must be reproduced verbatim. Their purpose is twofold: (1) to express behavioral contracts, preconditions, postconditions, and type-level invariants more precisely than prose alone, and (2) to anchor conformance test suites that verify backend interchangeability (see [Backend Testing §1](../hpc/backend-testing.md)). Implementation may diverge in naming, parameter ordering, error representation, or internal organization when practical considerations demand it -- provided the behavioral contracts and conformance tests continue to pass. When a trait signature and a prose description conflict, the prose description (which captures the domain intent) takes precedence; the conflict should be resolved by updating the trait signature. This convention applies to all trait-bearing specification documents in `src/specs/`.

## 1. Trait Definition

The horizon mode is modeled as a flat enum with two variants, matching the two horizon types supported by Cobre ([Extension Points SS4.1](./extension-points.md)):

```rust
/// Horizon mode controlling stage traversal, terminal conditions,
/// discount factors, and cut pool organization.
///
/// A single `HorizonMode` value is resolved from the `policy_graph`
/// field in `stages.json` during configuration loading (see
/// Extension Points SS6). The enum is global to the training run --
/// all stages share the same horizon mode.
#[derive(Debug, Clone)]
pub enum HorizonMode {
    /// Finite (acyclic) horizon.
    ///
    /// Linear chain 1 → 2 → ··· → T with terminal condition V_{T+1} = 0.
    /// The current implementation uses a linear-chain stage progression
    /// where each stage transitions to stage+1 (implicit transitions).
    /// Each stage has a unique cut pool indexed by stage ID.
    ///
    /// See [SDDP Algorithm SS4.1](../math/sddp-algorithm.md).
    Finite {
        /// Total number of stages T in the finite chain.
        /// Must be at least 2 (a single-stage problem has no predecessor
        /// to generate cuts for, making SDDP degenerate). Validated by
        /// `HorizonMode::validate`.
        num_stages: usize,
    },

    /// Cyclic (infinite periodic) horizon.
    ///
    /// At least one transition creates a cycle (source_id ≥ target_id).
    /// Cut pools are organized by season τ ∈ {1, ..., P}, not by
    /// absolute stage ID. Discount factor required for convergence:
    /// d_cycle = ∏_{t ∈ cycle} d_{t→t+1} < 1.
    ///
    /// See [Infinite Horizon](../math/infinite-horizon.md).
    ///
    /// > **Status: Not Implemented** -- The Cyclic variant is deferred to
    /// > a future release. The enum variant is defined here to reserve the
    /// > design space and document the intended contract. The current
    /// > implementation supports only `Finite` horizon.
    Cyclic {
        /// Precomputed transitions, same structure as Finite.
        transitions: TransitionMap,

        /// Number of stages per cycle P (e.g., 12 for monthly stages).
        cycle_length: usize,

        /// First stage ID in the cycle (the back-edge target).
        cycle_start: usize,

        /// Cumulative discount factor around one full cycle:
        /// d_cycle = ∏_{t ∈ cycle} d_{t→t+1}.
        /// Must be strictly less than 1 (validated by rule H2).
        cycle_discount: f64,

        /// Maximum number of stages the forward pass may traverse
        /// before forced termination (safety bound).
        /// E.g., 240 stages = 20 years for monthly stages.
        max_horizon_length: usize,

        /// Cumulative discount threshold below which the forward
        /// pass terminates (the remaining contribution is negligible).
        discount_threshold: f64,
    },
}
```

The `TransitionMap` type is defined in SS3. Note that the current `Finite` implementation uses implicit linear-chain transitions (`stage -> stage+1`) rather than an explicit `TransitionMap`. The `TransitionMap` is used only by the aspirational `Cyclic` variant.

## 2. Method Contracts

The four operations from [Extension Points SS4.4](./extension-points.md) are formalized below.

### 2.1 successors

`successors` returns the set of successor stages reachable from a given stage, together with the transition probability and per-transition discount factor. This is the primary method consumed by both the forward pass (to determine the next stage) and the backward pass (to determine which stages contribute cuts to the current stage).

```rust
impl HorizonMode {
    /// Return successor stages with transition probabilities and
    /// discount factors.
    ///
    /// For Finite mode, terminal stages return an empty vector.
    /// For Cyclic mode, the back-edge transition (where source_id ≥
    /// target_id) is included with its discount factor.
    pub fn successors(&self, stage_id: usize) -> Vec<Successor> {
        todo!()
    }
}

/// A single successor transition from a source stage.
#[derive(Debug, Clone, Copy)]
pub struct Successor {
    /// Target stage ID.
    pub target_id: usize,

    /// Transition probability. For deterministic chains, this is 1.0.
    pub probability: f64,

    /// Discount factor d_{source → target} for this transition.
    /// Derived from `annual_discount_rate` and stage duration,
    /// or from a per-transition override.
    /// See [Discount Rate SS3](../math/discount-rate.md).
    pub discount_factor: f64,
}
```

**Preconditions:**

| Condition                          | Description                                                                                                      |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `stage_id` exists in the stage set | The stage ID was loaded from `stages.json` and passed validation                                                 |
| Transitions have been precomputed  | `HorizonMode` was constructed via the variant selection pipeline ([Extension Points SS6](./extension-points.md)) |

**Postconditions:**

| Condition                                                            | Description                                                                                 |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Probabilities sum to 1.0 (within tolerance) when result is non-empty | Outgoing transitions from a non-terminal stage form a valid probability distribution        |
| All `target_id` values exist in the stage set                        | No dangling transitions (guaranteed by validation rule H4)                                  |
| All `discount_factor` values are in $(0, 1]$                         | Discount factors are positive and at most 1.0                                               |
| For `Finite`: terminal stages return an empty `Vec`                  | The last stage in the chain has no successors; $V_{T+1} = 0$                                |
| For `Cyclic`: result is always non-empty                             | Cyclic stages are never terminal; the back-edge ensures at least one successor              |
| For `Cyclic`: the back-edge transition has `target_id <= source_id`  | The back-edge is identifiable by the target being at or before the source in stage ordering |

**Determinism guarantee:** The returned successors are in a deterministic order (sorted by `target_id`). This ensures that all MPI ranks iterate over successors identically, preserving the reproducibility invariant.

**Infallibility:** This method does not return `Result`. All stage IDs and transitions are validated at configuration load time (rules H1-H4 in SS5).

### 2.2 is_terminal

`is_terminal` tests whether a stage has no successors. In finite horizon, the last stage in the chain is terminal ($V_{T+1} = 0$). In cyclic horizon, no stage is ever terminal -- the back-edge ensures perpetual traversal.

```rust
impl HorizonMode {
    /// Whether the given stage has no successors.
    ///
    /// Finite: true for the last stage in the chain.
    /// Cyclic: always false (the cycle never terminates).
    pub fn is_terminal(&self, stage_id: usize) -> bool {
        todo!()
    }
}
```

**Preconditions:**

| Condition                          | Description          |
| ---------------------------------- | -------------------- |
| `stage_id` exists in the stage set | Same as `successors` |

**Postconditions:**

| Condition                                                                     | Description                                                          |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Return value is consistent with `successors`                                  | `is_terminal(id) == true` if and only if `successors(id).is_empty()` |
| For `Finite`: exactly one stage (or more, in branching graphs) returns `true` | At least the final stage in each path is terminal                    |
| For `Cyclic`: always returns `false`                                          | No stage in a cyclic graph is terminal                               |

**Infallibility:** Same rationale as `successors`.

### 2.3 discount_factor

`discount_factor` returns the per-transition discount factor $d_{s \to t}$ between a source stage and a target stage. The discount factor is derived from the `annual_discount_rate` (global or per-transition override) and the source stage's duration, following the conversion formula in [Discount Rate SS3](../math/discount-rate.md):

$$
d_{s \to t} = \frac{1}{(1 + r_{\text{annual}})^{\Delta t_s}}
$$

where $\Delta t_s$ is the duration of the source stage expressed in years.

```rust
impl HorizonMode {
    /// Return the discount factor for the transition from `source` to
    /// `target`.
    ///
    /// The discount factor is applied to the future cost variable θ in
    /// the stage subproblem objective: c_t + d_{t→t+1} · θ.
    /// See [Discount Rate SS4](../math/discount-rate.md).
    pub fn discount_factor(&self, source: usize, target: usize) -> f64 {
        todo!()
    }
}
```

**Preconditions:**

| Condition                                | Description                                                   |
| ---------------------------------------- | ------------------------------------------------------------- |
| `source` exists in the stage set         | Valid source stage ID                                         |
| `target` exists in the stage set         | Valid target stage ID                                         |
| `(source, target)` is a valid transition | The transition exists in the `policy_graph.transitions` array |

**Postconditions:**

| Condition                                                                | Description                                                                                               |
| ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| Return value is in $(0, 1]$                                              | Discount factor is positive and at most 1.0                                                               |
| For undiscounted problems (`annual_discount_rate = 0`): returns 1.0      | No discounting applied                                                                                    |
| Consistent with `successors`                                             | `discount_factor(s, t)` equals the `discount_factor` field of the matching `Successor` in `successors(s)` |
| For `Cyclic` back-edge: the cumulative product around the cycle is $< 1$ | Guaranteed by validation rule H2                                                                          |

**Infallibility:** Same rationale as `successors`. The transition is guaranteed to exist after validation.

### 2.4 validate

`validate` verifies that the horizon mode configuration is consistent. It is called once during initialization -- never during training. The current implementation enforces rule H1 (non-empty stage set) for the `Finite` variant. Rules H2-H4 are deferred until the `Cyclic` variant is implemented.

```rust
impl HorizonMode {
    /// Post-construction validation of horizon mode invariants.
    ///
    /// Called once during initialization. Returns Ok(()) if all rules
    /// pass, or Err(SddpError::Validation(msg)) on failure.
    pub fn validate(&self) -> Result<(), SddpError> {
        todo!()
    }
}
```

> **Design intent:** When the `Cyclic` variant is implemented, `validate` should be extended with structured error accumulation (collecting all violated rules rather than failing on the first). The planned `ValidationError` enum for rules H2-H4 is documented below for reference but does not yet exist in the codebase:
>
> - **H2** `CycleDiscountNotConvergent` — cumulative cycle discount must be strictly less than 1
> - **H3** `CycleStartOutOfBounds` — cycle start stage must exist in the stage set
> - **H4** `DanglingTransition` — a transition targets a stage that does not exist

**Preconditions:**

| Condition                                                    | Description                                                |
| ------------------------------------------------------------ | ---------------------------------------------------------- |
| `stages` is the parsed stage array from `stages.json`        | Raw configuration data                                     |
| `policy_graph` is the parsed policy graph from `stages.json` | Contains `type`, `annual_discount_rate`, and `transitions` |

**Postconditions:**

| Condition                                                          | Description                                                          |
| ------------------------------------------------------------------ | -------------------------------------------------------------------- |
| On `Ok(horizon_mode)`: all rules H1-H4 are satisfied               | The returned `HorizonMode` is fully validated and ready for training |
| On `Err(errors)`: every violated rule produces a `ValidationError` | All violations are reported (not just the first)                     |
| H1 enforced: at least one stage exists                             | `stages.len() > 0`                                                   |
| H2 enforced: cycle discount $< 1$ (Cyclic only)                    | $d_{\text{cycle}} = \prod_{t \in \text{cycle}} d_{t \to t+1} < 1$    |
| H3 enforced: cycle start stage exists (Cyclic only)                | The back-edge target ID is present in the stage set                  |
| H4 enforced: no dangling transitions                               | Every `target_id` in `transitions` exists in the stage set           |

**Error accumulation:** `validate` collects all violations before returning, rather than failing on the first error. This allows the user to fix all configuration issues in a single iteration.

## 3. Supporting Types

### 3.1 PolicyGraphConfig

`PolicyGraphConfig` represents the deserialized form of the `policy_graph` field in `stages.json` ([Input Scenarios SS1.2](../data-model/input-scenarios.md)). It maps directly to the JSON schema:

```rust
/// Configuration representation of the policy graph, matching the
/// `policy_graph` field in `stages.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct PolicyGraphConfig {
    /// Horizon type: "finite_horizon" or "cyclic".
    #[serde(rename = "type")]
    pub graph_type: PolicyGraphType,

    /// Global annual discount rate.
    /// Converted to per-transition factors using stage durations.
    /// A value of 0.0 means no discounting (d = 1.0).
    pub annual_discount_rate: f64,

    /// Stage transitions with probabilities and optional per-transition
    /// discount rate overrides.
    pub transitions: Vec<TransitionConfig>,
}

/// Policy graph type tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyGraphType {
    FiniteHorizon,
    Cyclic,
}

/// A single transition in the policy graph.
#[derive(Debug, Clone, Deserialize)]
pub struct TransitionConfig {
    pub source_id: usize,
    pub target_id: usize,
    pub probability: f64,

    /// Per-transition annual discount rate override.
    /// When absent, the global `annual_discount_rate` is used.
    pub annual_discount_rate: Option<f64>,
}
```

### 3.2 TransitionMap

> **Status: Aspirational** -- The `TransitionMap` type is defined for the `Cyclic` variant, which is not yet implemented. The `Finite` variant uses implicit linear-chain transitions (`stage -> stage+1`) and does not require an explicit `TransitionMap`.

The `TransitionMap` is the precomputed runtime representation of transitions, built from `PolicyGraphConfig` during validation. It enables O(1) lookup of successors by stage ID.

```rust
/// Precomputed transition lookup: stage_id → Vec<Successor>.
///
/// Built once during `HorizonMode::validate()` from the raw
/// `PolicyGraphConfig`. Sorted by target_id within each entry
/// to ensure deterministic iteration order across MPI ranks.
pub type TransitionMap = HashMap<usize, Vec<Successor>>;
```

### 3.3 Season Mapping (Cyclic Mode)

For cyclic mode, the season function $\tau(t) = (t - 1) \bmod P + 1$ from [Infinite Horizon SS2](../math/infinite-horizon.md) maps absolute stage IDs to season indices. This mapping determines which cut pool a stage reads from and writes to.

```rust
impl HorizonMode {
    /// Map a stage ID to its season index τ ∈ {1, ..., P}.
    ///
    /// For Finite mode, each stage has a unique "season" equal to
    /// its stage ID (no sharing). For Cyclic mode, the season is
    /// computed as τ(t) = (t - cycle_start) mod P + 1, where P
    /// is the cycle_length.
    ///
    /// See [Infinite Horizon SS2](../math/infinite-horizon.md).
    pub fn season(&self, stage_id: usize) -> usize {
        todo!()
    }
}
```

## 4. Dispatch Mechanism

The horizon mode uses **enum dispatch** -- a `match` on the `HorizonMode` variant at each call site in the training loop. The horizon mode is global (one per run) because it defines the structural topology of the policy graph, which is a property of the problem formulation, not of individual stages.

**Why global, not per-stage:** The horizon mode determines the graph structure -- whether stages form an acyclic chain or a cycle. This is inherently a global property: either the graph has a cycle or it does not. Unlike the risk measure (where stage 0 might use CVaR while stage 1 uses Expectation), the horizon mode applies uniformly to all stages. A "per-stage horizon mode" would be semantically incoherent -- a stage cannot be simultaneously in a finite chain and in a cycle.

**Why enum dispatch, not compile-time monomorphization:** Although the horizon mode is global (making monomorphization technically feasible), enum dispatch is preferred for consistency with the other abstraction points ([Extension Points SS7](./extension-points.md)). The match overhead is negligible: `successors` and `is_terminal` are called once per stage per forward/backward pass iteration. At production scale (60 stages, 200 iterations), this is ~24,000 match dispatches per training run -- vanishingly small compared to the LP solve cost.

**Why not trait objects:** The variant set is closed (Finite and Cyclic only, with no additional variants planned). Trait objects add indirection cost without extensibility benefit. The enum approach avoids heap allocation and is consistent with the `RiskMeasure` dispatch pattern ([Risk Measure Trait SS4](./risk-measure-trait.md)).

## 5. Forward Pass Termination

In finite horizon, the forward pass terminates naturally when it reaches a terminal stage ($V_{T+1} = 0$). In cyclic horizon, there is no terminal stage -- the forward pass must be explicitly terminated to avoid infinite traversal.

The cyclic forward pass termination logic follows [Infinite Horizon SS6](../math/infinite-horizon.md). At each stage $t$ along the forward trajectory, the cumulative discount factor from the trajectory start is tracked:

$$
d_{1 \to t} = \prod_{s=1}^{t-1} d_{s \to s+1}
$$

The forward pass terminates when **either** of two conditions is met:

**Condition 1 -- Cumulative discount below threshold:** The remaining contribution to the total cost is negligible:

$$
d_{1 \to t} < \varepsilon_d
$$

where $\varepsilon_d$ is the `discount_threshold` parameter. When this condition holds, future stage costs are discounted by a factor so small that they cannot materially affect the total trajectory cost. Typical value: $\varepsilon_d = 10^{-6}$.

**Condition 2 -- Maximum horizon length reached:** A safety bound prevents unbounded traversal:

$$
t > L_{\max}
$$

where $L_{\max}$ is the `max_horizon_length` parameter. This protects against pathological cases where the discount factor per cycle is close to 1 (valid but slowly converging). Typical value: $L_{\max} = 240$ stages (20 years for monthly stages).

Both parameters are stored in the `Cyclic` variant of the `HorizonMode` enum (see SS1). The training loop checks these conditions after each stage solve in the forward pass:

```rust
/// Determine whether the cyclic forward pass should terminate at
/// the current stage.
///
/// Returns true if either the cumulative discount has fallen below
/// the threshold or the maximum horizon length has been exceeded.
impl HorizonMode {
    pub fn should_terminate_forward(
        &self,
        stages_traversed: usize,
        cumulative_discount: f64,
    ) -> bool {
        match self {
            HorizonMode::Finite { .. } => false, // Finite uses is_terminal instead
            HorizonMode::Cyclic {
                max_horizon_length,
                discount_threshold,
                ..
            } => {
                cumulative_discount < *discount_threshold
                    || stages_traversed > *max_horizon_length
            }
        }
    }
}
```

**Upper bound computation with cyclic termination:** The forward pass trajectory cost for the upper bound estimate ([Discount Rate SS7](../math/discount-rate.md)) sums discounted stage costs over all traversed stages, including multiple cycle repetitions:

$$
\bar{z}^{k,m} = \sum_{t=1}^{t_{\text{stop}}} d_{1 \to t} \cdot c_t(\hat{x}_t^{k,m})
$$

where $t_{\text{stop}}$ is the stage at which either termination condition was triggered.

## 6. Cut Pool Organization

The horizon mode determines how cut pools are indexed and shared across stages.

### 6.1 Finite Mode

In finite mode, each stage has its own independent cut pool, indexed by the absolute stage ID. Cut $k$ added to stage $t$ is only used by stage $t$:

$$
\underline{V}_t(x) = \max_{k \in \mathcal{K}_t} \left\{ \alpha_k + \pi_k^\top x \right\}
$$

where $\mathcal{K}_t$ is the cut set for stage $t$. There are $T$ independent cut pools, one per stage.

### 6.2 Cyclic Mode

In cyclic mode, cut pools are organized by **season** $\tau \in \{1, \ldots, P\}$ rather than by absolute stage ID, following [Infinite Horizon SS5](../math/infinite-horizon.md). Let $\mathcal{C}_\tau = \{t : \tau(t) = \tau\}$ be all stages with season $\tau$. A cut generated at any stage $t \in \mathcal{C}_\tau$ is valid for all stages in $\mathcal{C}_\tau$:

$$
\underline{V}_\tau(x) = \max_{k \in \mathcal{K}_\tau} \left\{ \alpha_k + \pi_k^\top x \right\}
$$

There are only $P$ cut pools (one per season), regardless of how many cycles the forward pass traverses. This means a single cycle's worth of cut pools represents the entire infinite horizon.

**Practical consequence:** When the backward pass generates a cut at stage $t$ in cycle $n$, the cut is added to cut pool $\tau(t)$, making it immediately available to all future stages at the same position in any cycle. This is the mechanism by which information propagates across cycles in the infinite horizon formulation.

**Season mapping:** The season function from [Infinite Horizon SS2](../math/infinite-horizon.md) is:

$$
\tau(t) = (t - t_{\text{start}}) \bmod P + 1
$$

where $t_{\text{start}}$ is the cycle start stage ID (the `cycle_start` field in the `Cyclic` variant) and $P$ is the `cycle_length`.

## Cross-References

- [Infinite Horizon](../math/infinite-horizon.md) -- Periodic structure (SS2), cyclic policy graph (SS3), discount requirement (SS4), cut sharing within cycles (SS5), forward pass termination (SS6), backward pass behavior (SS7)
- [Discount Rate](../math/discount-rate.md) -- Discounted Bellman equation (SS2), annual-rate-to-factor conversion (SS3), discount on $\theta$ (SS4), cumulative discounting (SS5), infinite horizon considerations (SS9)
- [SDDP Algorithm SS4](../math/sddp-algorithm.md) -- Policy graph topologies: finite horizon (SS4.1), cyclic graph (SS4.2)
- [Extension Points](./extension-points.md) -- Horizon mode variant table (SS4.1), configuration mapping (SS4.2), validation rules H1-H4 (SS4.3), behavioral contract (SS4.4), dispatch mechanism analysis (SS7), variant selection pipeline (SS6)
- [Training Loop SS3.3](./training-loop.md) -- Behavioral contract for the horizon mode abstraction point
- [Input Scenarios SS1.2](../data-model/input-scenarios.md) -- `policy_graph` JSON schema: `type`, `annual_discount_rate`, `transitions` array, per-transition overrides
- [Cut Management](../math/cut-management.md) -- Cut generation (SS2), single-cut aggregation (SS3); cuts remain undiscounted, discount applied to $\theta$
- [Stopping Rules](../math/stopping-rules.md) -- Convergence criteria using discounted bounds, cycle convergence tolerance
- [Risk Measure Trait](./risk-measure-trait.md) -- Sibling trait specification following the same enum dispatch pattern
- [Communicator Trait](../hpc/communicator-trait.md) -- Reference pattern for trait specification structure and convention blockquote
- [Solver Abstraction SS10](./solver-abstraction.md) -- Compile-time solver selection pattern (contrasted with the enum dispatch used here)
- [Configuration Reference](../configuration/configuration-reference.md) -- Horizon and cycle configuration parameters: `max_horizon_length`, `discount_threshold`, `cycle_convergence_tolerance`
