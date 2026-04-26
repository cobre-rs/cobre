# Stopping Rule Trait

## Purpose

This spec defines the `StoppingRule` and `StoppingRuleSet` abstractions -- a two-layer enum-based design through which the SDDP training loop evaluates termination conditions after each iteration. The first layer models individual stopping rules as enum variants, each with an `evaluate` method that inspects convergence monitor state and returns a boolean. The second layer composes multiple rules into a `StoppingRuleSet` with configurable combination logic ("any" for OR, "all" for AND), returning both a termination decision and the triggering reason(s). The composition layer is unique among Cobre's trait abstractions: unlike the single-variant enums used for [RiskMeasure](./risk-measure-trait.md), [HorizonMode](./horizon-mode-trait.md), [SamplingScheme](./sampling-scheme-trait.md), and [CutSelectionStrategy](./cut-selection-trait.md), the stopping rule abstraction requires evaluating a _set_ of rules and combining their results according to a configurable boolean logic. The five supported rules -- IterationLimit, TimeLimit, BoundStalling, SimulationBased, and GracefulShutdown -- correspond to the stopping rules defined in [Stopping Rules](../math/stopping-rules.md) and evaluated by the convergence monitor in [Convergence Monitoring SS2.3](./convergence-monitoring.md).

> **Convention: Rust traits as specification guidelines.** The Rust trait definitions, method signatures, and struct declarations throughout this specification corpus serve as _guidelines for implementation_, not as absolute source-of-truth contracts that must be reproduced verbatim. Their purpose is twofold: (1) to express behavioral contracts, preconditions, postconditions, and type-level invariants more precisely than prose alone, and (2) to anchor conformance test suites that verify backend interchangeability (see [Backend Testing §1](../hpc/backend-testing.md)). Implementation may diverge in naming, parameter ordering, error representation, or internal organization when practical considerations demand it -- provided the behavioral contracts and conformance tests continue to pass. When a trait signature and a prose description conflict, the prose description (which captures the domain intent) takes precedence; the conflict should be resolved by updating the trait signature. This convention applies to all trait-bearing specification documents in `src/specs/`.

## 1. Trait Definition

The stopping rule abstraction has two layers: individual rules (enum variants with an `evaluate` method) and a composition layer (`StoppingRuleSet`) that holds a list of rules and combines their evaluations.

### 1.1 Individual Rule Enum

```rust
/// Individual stopping rule for SDDP training termination.
///
/// Each variant corresponds to one of the five stopping rules defined in
/// [Stopping Rules](../math/stopping-rules.md). The enum is stored inside
/// a `StoppingRuleSet` (SS1.2), which manages the composition logic.
///
/// The `IterationLimit` variant is mandatory -- every `StoppingRuleSet`
/// must contain at least one `IterationLimit` rule as a safety bound
/// against infinite loops.
#[derive(Debug, Clone)]
pub enum StoppingRule {
    /// Terminate when the iteration count reaches a fixed limit.
    ///
    /// This is a safety bound -- it must always be present in the rule set.
    /// See [Stopping Rules SS2](../math/stopping-rules.md).
    IterationLimit {
        /// Maximum iteration count k_max. Training stops when k >= limit.
        limit: u64,
    },

    /// Terminate when cumulative wall-clock time exceeds a threshold.
    ///
    /// Wall-clock time is checked at the end of each iteration (after the
    /// convergence update step in [Training Loop SS2.1](./training-loop.md)).
    /// See [Stopping Rules SS3](../math/stopping-rules.md).
    TimeLimit {
        /// Maximum wall-clock time in seconds.
        seconds: f64,
    },

    /// Terminate when the lower bound improvement over a sliding window
    /// falls below a relative tolerance.
    ///
    /// See [Stopping Rules SS4](../math/stopping-rules.md) for the formula.
    BoundStalling {
        /// Relative tolerance for the windowed improvement test.
        tolerance: f64,

        /// Number of past iterations over which to measure improvement (tau).
        iterations: u64,
    },

    /// Terminate when both the lower bound and simulated policy costs
    /// have stabilized. Checked only every `period` iterations.
    ///
    /// This is the most complex rule, involving a two-phase evaluation:
    /// first a bound stability check, then a Monte Carlo simulation
    /// comparison. See [Stopping Rules SS5](../math/stopping-rules.md).
    SimulationBased {
        /// Evaluate this rule every `period` iterations (not every iteration).
        period: u64,

        /// Normalized Euclidean distance threshold for simulation cost
        /// comparison between consecutive evaluations.
        distance_tolerance: f64,

        /// Number of Monte Carlo forward simulations to run when the bound
        /// stability pre-filter passes (executed by the convergence monitor).
        replications: u32,

        /// Number of past iterations for the bound stability pre-check.
        bound_stability_window: u64,
    },

    /// Terminate when an external signal (SIGTERM, SIGINT) is received.
    ///
    /// This rule is not configured via JSON -- it is always implicitly
    /// present and driven by an OS signal flag. The training loop
    /// checkpoints the last **completed** iteration before exiting.
    /// See [Convergence Monitoring SS1](./convergence-monitoring.md).
    GracefulShutdown,
}
```

### 1.2 Composition Layer

```rust
/// A composed set of stopping rules with configurable combination logic.
///
/// The `StoppingRuleSet` holds a list of individual rules and a
/// `stopping_mode` that determines how individual rule evaluations combine:
/// - `"any"` (OR): stop when ANY rule triggers
/// - `"all"` (AND): stop when ALL rules trigger simultaneously
///
/// The set always includes at least one `IterationLimit` rule (validated
/// during configuration loading) and always includes an implicit
/// `GracefulShutdown` rule (injected at construction time, not configured
/// in JSON).
///
/// See [Stopping Rules SS6](../math/stopping-rules.md) for the
/// combination logic specification.
pub struct StoppingRuleSet {
    /// The individual stopping rules, resolved from `config.json` during
    /// configuration loading. Must contain at least one `IterationLimit`.
    /// The `GracefulShutdown` rule is always appended implicitly.
    pub rules: Vec<StoppingRule>,

    /// Combination mode: `Any` (OR logic) or `All` (AND logic).
    pub mode: StoppingMode,
}

/// Combination mode for the stopping rule set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoppingMode {
    /// Stop when any rule triggers (OR logic). This is the default.
    Any,

    /// Stop when all rules trigger simultaneously (AND logic).
    All,
}

impl StoppingRuleSet {
    /// Evaluate all stopping rules against the current monitor state.
    ///
    /// Returns `(should_stop, all_results)` where `should_stop` is the
    /// combined termination decision and `all_results` lists the evaluation
    /// result for every rule.
    ///
    /// `GracefulShutdown` is always evaluated first. If the shutdown flag
    /// is set, the method returns `(true, results)` immediately, regardless
    /// of the configured `mode`.
    ///
    /// For the remaining rules:
    /// - `Any`: stop if any rule triggered (OR logic).
    /// - `All`: stop if all rules triggered (AND logic).
    pub fn evaluate(&self, state: &MonitorState) -> (bool, Vec<StoppingRuleResult>) {
        todo!()
    }
}
```

**Two-layer rationale:** The individual rule enum (`StoppingRule`) models each termination criterion as a self-contained evaluator with its own parameters and logic. The composition layer (`StoppingRuleSet`) provides the combination semantics that are unique to stopping rules among Cobre's abstractions. Other enum-dispatch traits (RiskMeasure, HorizonMode, SamplingScheme, CutSelectionStrategy) operate on a single active variant per call site. Stopping rules operate on a _set_ of active variants combined by boolean logic, requiring the additional composition layer.

## 2. Individual Rule Contracts

Each stopping rule variant implements an `evaluate` method that reads from the convergence monitor state and returns a boolean indicating whether the rule's termination condition is met.

```rust
impl StoppingRule {
    /// Evaluate this individual rule against the current monitor state.
    ///
    /// Returns a `StoppingRuleResult` with the rule's identifier, whether
    /// the rule's termination condition is satisfied, and a human-readable
    /// description of the current state.
    ///
    /// The method is pure -- it reads from `MonitorState` but does not
    /// modify it. Side effects (running Monte Carlo simulations for the
    /// SimulationBased rule) are performed by the convergence monitor
    /// before calling `evaluate`, and the results are stored in the
    /// monitor state.
    pub fn evaluate(&self, state: &MonitorState) -> StoppingRuleResult {
        match self {
            Self::IterationLimit { limit } => { /* SS2.1 */ }
            Self::TimeLimit { seconds } => { /* SS2.2 */ }
            Self::BoundStalling { tolerance, iterations } => { /* SS2.3 */ }
            Self::SimulationBased { period, .. } => { /* SS2.4 */ }
            Self::GracefulShutdown => { /* SS2.5 */ }
        }
    }
}
```

**Infallibility:** The `evaluate` method does not return `Result`. All rule parameters are validated at configuration load time (SS4), and the `MonitorState` is populated by the convergence monitor each iteration before rule evaluation. This follows the same upstream-validation-guarantees-infallibility pattern established in [RiskMeasure SS2](./risk-measure-trait.md) and [SamplingScheme SS2](./sampling-scheme-trait.md).

### 2.1 IterationLimit

**Evaluation:**

$$
\text{STOP} \iff k \geq k_{\max}
$$

where $k$ is `state.iteration` (1-based) and $k_{\max}$ is the `limit` parameter.

**Preconditions:**

| Condition              | Description                          |
| ---------------------- | ------------------------------------ |
| `state.iteration >= 1` | At least one iteration has completed |
| `limit >= 1`           | Validated at config load time (SS4)  |

**Postconditions:**

| Condition                                                            | Description                                       |
| -------------------------------------------------------------------- | ------------------------------------------------- |
| Returns `true` iff `state.iteration >= limit`                        | Deterministic evaluation based on iteration count |
| Monotonic: once `true`, remains `true` for all subsequent iterations | Iteration count is monotonically increasing       |

**Monitor state consumed:** `state.iteration` only.

**Mandatory presence:** The `IterationLimit` rule serves as a safety bound to prevent infinite loops. Every `StoppingRuleSet` must contain at least one `IterationLimit` rule. This is enforced during configuration validation (SS4).

### 2.2 TimeLimit

**Evaluation:**

$$
\text{STOP} \iff t_{\text{elapsed}} \geq t_{\max}
$$

where $t_{\text{elapsed}}$ is `state.wall_time_seconds` (cumulative wall-clock time since training start) and $t_{\max}$ is the `seconds` parameter.

**Preconditions:**

| Condition                        | Description                         |
| -------------------------------- | ----------------------------------- |
| `state.wall_time_seconds >= 0.0` | Non-negative elapsed time           |
| `seconds > 0.0`                  | Validated at config load time (SS4) |

**Postconditions:**

| Condition                                                            | Description                                    |
| -------------------------------------------------------------------- | ---------------------------------------------- |
| Returns `true` iff `state.wall_time_seconds >= seconds`              | Deterministic evaluation based on elapsed time |
| Monotonic: once `true`, remains `true` for all subsequent iterations | Wall-clock time is monotonically increasing    |

**Monitor state consumed:** `state.wall_time_seconds` only.

### 2.3 BoundStalling

**Evaluation:**

Track the deterministic lower bound $\underline{z}^k$ over iterations. Compute relative improvement over a window of $\tau$ iterations (the `iterations` parameter):

$$
\Delta_k = \frac{\underline{z}^k - \underline{z}^{k-\tau}}{\max(1, |\underline{z}^k|)}
$$

**Stopping condition:**

$$
\text{STOP} \iff |\Delta_k| < \text{tolerance}
$$

This formula is defined in [Stopping Rules SS4](../math/stopping-rules.md) and reproduced in [Convergence Monitoring SS2.2](./convergence-monitoring.md).

**Preconditions:**

| Condition                                       | Description                                                       |
| ----------------------------------------------- | ----------------------------------------------------------------- |
| `state.iteration >= iterations`                 | Enough history exists for the windowed comparison                 |
| `state.lower_bound_history.len() >= iterations` | The monitor has recorded at least `iterations` lower bound values |
| `iterations >= 1`                               | Validated at config load time (SS4)                               |
| `tolerance > 0.0`                               | Validated at config load time (SS4)                               |

**Postconditions:**

| Condition                                                                                     | Description                                                              |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Returns `false` if `state.iteration < iterations`                                             | Not enough history to evaluate; rule is not triggered                    |
| Returns `true` iff the absolute relative improvement is below `tolerance`                     | Matches the formula from [Stopping Rules SS4](../math/stopping-rules.md) |
| The denominator uses $\max(1, \lvert\underline{z}^k\rvert)$ to guard against near-zero bounds | Numerical robustness against division by small values                    |

**Monitor state consumed:** `state.iteration`, `state.lower_bound` (current), and `state.lower_bound_history[k - iterations]` (the bound from $\tau$ iterations ago).

**Not monotonic:** Unlike IterationLimit and TimeLimit, BoundStalling can return `true` at iteration $k$ and `false` at iteration $k+1$ if a late cut causes a significant bound jump. The composition logic in SS3 handles this correctly -- it evaluates all rules at each iteration, not just once.

### 2.4 SimulationBased

The simulation-based rule is the most complex, involving a two-phase evaluation that runs only every `period` iterations. It combines a bound stability pre-check with a Monte Carlo policy comparison.

**Phase 1 -- Bound stability check:**

$$
\text{Bound stable} \iff \left| \underline{z}^k - \underline{z}^{k - w} \right| < \text{bound\_tol} \times \max(1, |\underline{z}^k|)
$$

where $w$ is the `bound_stability_window` parameter. If the bound is not stable, the rule returns `false` without running simulations (saving computational cost).

**Phase 2 -- Monte Carlo policy comparison:**

If the bound is stable, the convergence monitor runs `replications` Monte Carlo forward simulations using the current policy. The per-stage mean costs $c_t^{\text{new}}$ are compared to the previous simulation's costs $c_t^{\text{old}}$ via normalized Euclidean distance:

$$
d = \sqrt{\sum_{t} \left( \frac{c_t^{\text{new}} - c_t^{\text{old}}}{\max(1, |c_t^{\text{old}}|)} \right)^2}
$$

**Stopping condition:**

$$
\text{STOP} \iff \text{Bound stable} \land d < \text{distance\_tol}
$$

This is the algorithm defined in [Stopping Rules SS5](../math/stopping-rules.md).

**Preconditions:**

| Condition                                                   | Description                                                                                             |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `state.iteration >= period`                                 | At least one evaluation period has elapsed                                                              |
| `state.iteration % period == 0`                             | Current iteration is a check point                                                                      |
| `state.lower_bound_history.len() >= bound_stability_window` | Enough bound history for stability check                                                                |
| Previous simulation results available in `state`            | Required for distance computation (first evaluation has no previous results and always returns `false`) |
| `replications >= 1`                                         | Validated at config load time (SS4)                                                                     |
| `period >= 1`                                               | Validated at config load time (SS4)                                                                     |
| `bound_stability_window >= 1`                               | Validated at config load time (SS4)                                                                     |
| `distance_tolerance > 0.0`                                  | Validated at config load time (SS4)                                                                     |

**Postconditions:**

| Condition                                                                           | Description                                                             |
| ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Returns `false` if `state.iteration % period != 0`                                  | Not an evaluation iteration                                             |
| Returns `false` if bound stability check fails (Phase 1)                            | No simulation is run; no computational cost beyond the bound comparison |
| Returns `false` on the first evaluation (no previous simulation to compare against) | The distance metric requires two consecutive simulation snapshots       |
| Returns `true` iff both bound stability and simulation distance checks pass         | Both phases must pass simultaneously                                    |

**Monitor state consumed:** `state.iteration`, `state.lower_bound`, `state.lower_bound_history[k - bound_stability_window]`, and `state.simulation_costs` (per-stage mean costs from the most recent simulation evaluation, populated by the convergence monitor before `evaluate` is called).

**Computational cost:** Running `replications` Monte Carlo simulations is expensive (comparable to a forward pass). This is why the rule evaluates only every `period` iterations and gates the simulation behind the cheap bound stability pre-check (Phase 1). The convergence monitor is responsible for executing the simulations and populating `state.simulation_costs`; the `evaluate` method itself only performs the distance comparison.

**Side-effect separation:** The `evaluate` method is pure -- it reads simulation results from `MonitorState` but does not trigger the simulations itself. The convergence monitor detects that a SimulationBased rule exists in the set, checks whether the current iteration is a multiple of `period`, and if so, runs the Phase 1 check internally. Only if Phase 1 passes does the monitor execute the Monte Carlo simulations and store the results in `state.simulation_costs`. This separation ensures that `evaluate` remains a stateless, testable function.

### 2.5 GracefulShutdown

**Evaluation:**

$$
\text{STOP} \iff \text{signal\_flag} = \text{true}
$$

The `signal_flag` is set by an OS signal handler (SIGTERM or SIGINT) and read via `state.shutdown_requested`.

**Preconditions:**

| Condition                                     | Description                                |
| --------------------------------------------- | ------------------------------------------ |
| `state.shutdown_requested` is a valid boolean | Set by the signal handler, read atomically |

**Postconditions:**

| Condition                                             | Description                                            |
| ----------------------------------------------------- | ------------------------------------------------------ |
| Returns `true` iff `state.shutdown_requested == true` | Direct flag check                                      |
| Monotonic: once `true`, remains `true`                | The signal flag is never cleared during a training run |

**Monitor state consumed:** `state.shutdown_requested` only.

**Not configurable:** This rule is not configured via JSON. It is always implicitly present in every `StoppingRuleSet` and is injected at construction time. The GracefulShutdown rule always participates in "any" mode semantics regardless of the configured `stopping_mode` -- if a shutdown signal is received, training terminates unconditionally. This is implemented by evaluating GracefulShutdown separately from the configured rules, before applying the composition logic (SS3).

## 3. Composition Contract

The `StoppingRuleSet::evaluate` method evaluates all individual rules and combines their results according to the `stopping_mode`.

### 3.1 Evaluation Semantics

```rust
impl StoppingRuleSet {
    pub fn evaluate(&self, state: &MonitorState) -> (bool, Vec<StoppingRuleResult>) {
        // Step 1: Evaluate GracefulShutdown unconditionally.
        // If shutdown is requested, evaluate all rules and return
        // (true, results) immediately regardless of mode.
        if state.shutdown_requested {
            let results: Vec<StoppingRuleResult> =
                self.rules.iter().map(|r| r.evaluate(state)).collect();
            return (true, results);
        }

        // Step 2: Evaluate all configured rules.
        let results: Vec<StoppingRuleResult> =
            self.rules.iter().map(|r| r.evaluate(state)).collect();

        // Step 3: Apply combination logic (GracefulShutdown already handled).
        let non_shutdown_triggered: Vec<bool> = self.rules
            .iter()
            .zip(results.iter())
            .filter(|(rule, _)| !matches!(rule, StoppingRule::GracefulShutdown))
            .map(|(_, result)| result.triggered)
            .collect();

        let should_stop = match self.mode {
            StoppingMode::Any => {
                // OR logic: stop if any rule triggered.
                non_shutdown_triggered.iter().any(|&t| t)
            }
            StoppingMode::All => {
                // AND logic: stop only if all rules triggered.
                !non_shutdown_triggered.is_empty()
                    && non_shutdown_triggered.iter().all(|&t| t)
            }
        };

        (should_stop, results)
    }
}
```

### 3.2 "Any" Mode (OR Logic)

$$
\text{STOP} \iff \text{Rule}_1 \lor \text{Rule}_2 \lor \ldots \lor \text{Rule}_n
$$

The first rule (in configuration order) whose `evaluate` returns `true` is reported as the stop reason. If multiple rules trigger simultaneously, only the first is reported. This matches [Stopping Rules SS6](../math/stopping-rules.md): "First rule to trigger causes termination."

**Postconditions:**

| Condition                                                                                  | Description                              |
| ------------------------------------------------------------------------------------------ | ---------------------------------------- |
| Returns `(true, results)` if any non-shutdown rule's `triggered` field is `true`           | At least one rule satisfied              |
| Returns `(false, results)` if no non-shutdown rule's `triggered` field is `true`           | Training continues                       |
| `results` contains one `StoppingRuleResult` per rule in the set, regardless of `triggered` | Full evaluation transparency for logging |

### 3.3 "All" Mode (AND Logic)

$$
\text{STOP} \iff \text{Rule}_1 \land \text{Rule}_2 \land \ldots \land \text{Rule}_n
$$

All configured rules must evaluate to `true` simultaneously for termination. The stop reason reports all triggered rules.

**Postconditions:**

| Condition                                                                                  | Description                              |
| ------------------------------------------------------------------------------------------ | ---------------------------------------- |
| Returns `(true, results)` if all non-shutdown rules' `triggered` fields are `true`         | All rules satisfied simultaneously       |
| Returns `(false, results)` if any non-shutdown rule's `triggered` field is `false`         | Training continues                       |
| `results` contains one `StoppingRuleResult` per rule in the set, regardless of `triggered` | Full evaluation transparency for logging |

### 3.4 GracefulShutdown Override

The `GracefulShutdown` rule is evaluated before the configured rules and bypasses the composition logic entirely. If `state.shutdown_requested` is `true`, `evaluate` returns `(true, results)` regardless of the `stopping_mode` and regardless of whether other rules have triggered. This ensures that external termination signals are always honored immediately.

**Rationale:** A shutdown signal represents an external constraint (e.g., job scheduler timeout, operator intervention) that supersedes algorithmic convergence criteria. Requiring all rules to trigger before honoring a shutdown signal (in "all" mode) would be incorrect -- the system must exit promptly.

## 4. Supporting Types

### 4.1 StoppingRuleConfig

The `StoppingRuleConfig` enum represents the deserialized form of individual entries in the `stopping_rules` array in `config.json` ([Configuration Reference](../configuration/configuration-reference.md)):

```rust
/// Configuration representation of a stopping rule, matching the
/// `stopping_rules[]` entries in `config.json`.
///
/// Deserialized from JSON objects with a `type` discriminator field.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoppingRuleConfig {
    IterationLimit {
        limit: u32,
    },
    TimeLimit {
        seconds: f64,
    },
    BoundStalling {
        tolerance: f64,
        iterations: u32,
    },
    Simulation {
        period: u32,
        distance_tol: f64,
        bound_tol: f64,
        replications: u32,
        bound_window: u32,
    },
}
```

**Conversion:** `StoppingRuleConfig` entries are validated and converted to `StoppingRule` variants during configuration loading. The `GracefulShutdown` variant has no config representation -- it is injected at `StoppingRuleSet` construction time.

**Validation rules:**

| Rule | Condition                                   | Error                                                                                   |
| ---- | ------------------------------------------- | --------------------------------------------------------------------------------------- |
| V1   | `limit >= 1` for IterationLimit             | Zero iteration limit prevents any training                                              |
| V2   | `seconds > 0.0` for TimeLimit               | Non-positive time limit                                                                 |
| V3   | `iterations >= 1` for BoundStalling         | Zero window is undefined                                                                |
| V4   | `tolerance > 0.0` for BoundStalling         | Non-positive tolerance                                                                  |
| V5   | `replications >= 1` for Simulation          | Zero replications is undefined                                                          |
| V6   | `period >= 1` for Simulation                | Zero period (every iteration) is computationally prohibitive; use BoundStalling instead |
| V7   | `bound_window >= 1` for Simulation          | Zero bound stability window is undefined                                                |
| V8   | `distance_tol > 0.0` for Simulation         | Non-positive distance tolerance                                                         |
| V9   | `bound_tol > 0.0` for Simulation            | Non-positive bound tolerance                                                            |
| V9   | At least one IterationLimit rule in the set | Safety bound is mandatory                                                               |

### 4.2 StoppingRuleResult

```rust
/// Result of evaluating a single stopping rule, returned as part of
/// the `Vec<StoppingRuleResult>` from `StoppingRuleSet::evaluate`.
///
/// Defined in `cobre-core` so that the result type is available to both
/// the stopping rule module and the convergence monitor without a
/// circular dependency.
#[derive(Debug, Clone)]
pub struct StoppingRuleResult {
    /// Rule identifier matching the variant name (e.g., `"iteration_limit"`,
    /// `"time_limit"`, `"bound_stalling"`, `"simulation_based"`,
    /// `"graceful_shutdown"`).
    pub rule_name: String,

    /// Whether this rule's condition is satisfied at the current iteration.
    pub triggered: bool,

    /// Human-readable description of the rule's current state
    /// (e.g., `"iteration 100/100"`, `"elapsed 3600.0s / 3600.0s limit"`).
    pub detail: String,
}
```

The `rule_name` values correspond to the termination reason strings in [Convergence Monitoring SS4.2](./convergence-monitoring.md): `"iteration_limit"`, `"time_limit"`, `"bound_stalling"`, `"simulation_based"`, and `"graceful_shutdown"`. Each `StoppingRule` variant produces a fixed `rule_name` string via named constants.

### 4.3 MonitorState

The `MonitorState` struct holds the convergence monitor quantities consumed by the stopping rules. It is populated by the convergence monitor each iteration and passed to `StoppingRuleSet::evaluate` as a read-only reference.

```rust
/// Convergence monitor state consumed by stopping rule evaluation.
///
/// This is a read-only view of the quantities tracked by the convergence
/// monitor ([Convergence Monitoring SS2.1](./convergence-monitoring.md)).
/// The monitor populates this state after each iteration's forward
/// synchronization step, before calling `evaluate`.
pub struct MonitorState {
    /// Current iteration index (1-based).
    pub iteration: u64,

    /// Cumulative wall-clock time since training start, in seconds.
    pub wall_time_seconds: f64,

    /// Current lower bound (stage-1 LP objective value).
    pub lower_bound: f64,

    /// History of lower bounds from past iterations (chronological order).
    /// `lower_bound_history[i]` is the lower bound at iteration `i + 1`.
    /// Populated by the convergence monitor; appended each iteration.
    pub lower_bound_history: Vec<f64>,

    /// Whether an external shutdown signal has been received.
    /// Set by an OS signal handler (SIGTERM / SIGINT) and read atomically.
    pub shutdown_requested: bool,

    /// Per-stage mean costs from the most recent simulation evaluation.
    /// `None` if no simulation has been run yet, or if the convergence
    /// monitor has not yet run a `SimulationBased` check.
    pub simulation_costs: Option<Vec<f64>>,
}
```

**Which rules consume which fields:**

| MonitorState Field    | IterationLimit | TimeLimit | BoundStalling | SimulationBased | GracefulShutdown |
| --------------------- | :------------: | :-------: | :-----------: | :-------------: | :--------------: |
| `iteration`           |      Yes       |           |      Yes      |       Yes       |                  |
| `wall_time_seconds`   |                |    Yes    |               |                 |                  |
| `lower_bound`         |                |           |      Yes      |       Yes       |                  |
| `lower_bound_history` |                |           |      Yes      |       Yes       |                  |
| `shutdown_requested`  |                |           |               |                 |       Yes        |
| `simulation_costs`    |                |           |               |       Yes       |                  |

## 5. Interaction with Convergence Monitor

The convergence monitor ([Convergence Monitoring](./convergence-monitoring.md)) owns the `StoppingRuleSet` and calls `evaluate` once per iteration. The interaction follows a fixed protocol:

### 5.1 Per-Iteration Protocol

1. **Forward pass completes.** The training loop produces per-rank upper bound statistics (cost sum, cost sum-of-squares, scenario count).

2. **Forward synchronization.** `allreduce` with `ReduceOp::Sum` aggregates global upper bound statistics across ranks ([Convergence Monitoring SS3.1](./convergence-monitoring.md)).

3. **State exchange.** Each rank broadcasts its visited states to all other ranks so that every rank has the full set of trial points for cut generation ([Training Loop SS5](./training-loop.md)).

4. **Backward pass.** Generate cuts from visited states ([Training Loop SS6](./training-loop.md)).

5. **Cut synchronization.** Distribute new cuts to all ranks via `allgatherv`.

6. **Lower bound evaluation.** Rank 0 solves all stage-0 openings with the latest FCF cuts, applies the risk measure, and broadcasts the LB to all ranks ([Training Loop SS4.3b](./training-loop.md)).

7. **Monitor update.** The convergence monitor updates `MonitorState`:
   - Increments `iteration`
   - Updates `wall_time_seconds`
   - Appends the new lower bound to `lower_bound_history`
   - Sets `lower_bound` to the current value
   - Checks `shutdown_requested` flag

8. **Simulation pre-check (conditional).** If a `SimulationBased` rule exists in the set and `iteration % period == 0`:
   - The monitor performs the Phase 1 bound stability check internally
   - If Phase 1 passes, the monitor runs `replications` Monte Carlo simulations
   - The monitor stores per-stage mean costs in `state.simulation_costs`

9. **Rule evaluation.** The monitor calls `self.rule_set.evaluate(&self.state)`, receiving `(should_stop, results)`.

10. **Decision.** If `evaluate` returns `(true, results)` with at least one triggered result, the monitor records the termination reason and signals the training loop to exit. If `(false, results)` with no triggered results, the training loop proceeds to the next iteration.

### 5.2 Ownership Boundaries

The stopping rule abstraction is deliberately separate from the convergence monitor. The monitor is responsible for:

- Tracking all convergence quantities (`MonitorState`)
- Executing Monte Carlo simulations when needed by the SimulationBased rule
- Emitting training log events ([Convergence Monitoring SS4](./convergence-monitoring.md))
- Persisting convergence history to Parquet

The stopping rules are responsible for:

- Evaluating termination conditions given a snapshot of monitor state
- Composing individual evaluations into a combined decision
- Reporting the stop reason

This separation ensures that stopping rules are testable in isolation (given a `MonitorState`, the output is deterministic) and that the convergence monitor can be extended with new tracked quantities without modifying the stopping rule enum.

## 6. Risk-Averse Considerations

When CVaR risk measures are active on any stage, two stopping rules are affected:

### 6.1 BoundStalling under CVaR

The lower bound $\underline{z}^k$ monitored by the BoundStalling rule is the stage-1 LP objective value. As documented in [Risk Measures SS10](../math/risk-measures.md), this value is **not a valid lower bound** on the true risk-averse optimal cost when CVaR is active -- it is a convergence indicator only. However, BoundStalling remains useful because it detects stabilization of the outer approximation. When the convergence indicator plateaus (relative improvement below tolerance), further iterations are unlikely to produce meaningful policy improvement.

The BoundStalling formula itself does not change under CVaR. The interpretation changes: instead of "the gap between LB and the true optimum is small," the interpretation is "the outer approximation has converged to its limit, and additional cuts provide diminishing returns."

> **Recommendation:** When risk measures are enabled, convergence reports should label the stalling detection as "convergence indicator stabilized" rather than "lower bound converged." See [Risk Measures SS10, Recommendations](../math/risk-measures.md).

### 6.2 SimulationBased under CVaR

The simulation-based rule's Phase 2 compares per-stage mean costs from Monte Carlo simulations. This comparison measures **policy stability** directly: it checks whether the decisions produced by the current policy have stabilized across iterations. Policy stability is independent of bound validity -- even when the lower bound is not a valid bound (under CVaR), the simulated costs reflect the actual quality of the policy.

The Phase 1 bound stability check monitors the same convergence indicator discussed in SS6.1. Under CVaR, Phase 1 detects stabilization of the risk-adjusted outer approximation, which is a necessary precondition for policy convergence.

The cross-variant composition rule X2 from [Extension Points SS8.1](./extension-points.md) applies: when CVaR is active and the SimulationBased rule is configured, the simulation forward costs must use risk-adjusted evaluation. The convergence monitor is responsible for applying the risk measure's `evaluate_risk` method ([RiskMeasure SS2.2](./risk-measure-trait.md)) when computing the simulation cost summary.

## 7. Dispatch Mechanism

The stopping rule abstraction uses **enum dispatch** -- a `match` on the `StoppingRule` variant in the `evaluate` method. This is consistent with the pattern established in [RiskMeasure SS4](./risk-measure-trait.md), [HorizonMode SS4](./horizon-mode-trait.md), [SamplingScheme SS4](./sampling-scheme-trait.md), and [CutSelectionStrategy SS4](./cut-selection-trait.md).

**Why not compile-time monomorphization:** The stopping rule set contains multiple rules evaluated simultaneously, with the composition result depending on all of them. A generic type parameter `S: StoppingRuleTrait` would fix a single rule type, which cannot represent a heterogeneous set of rules. Encoding the set as `Vec<Box<dyn StoppingRuleTrait>>` would introduce heap indirection without benefit, since the variant set is small and closed.

**Why not trait objects:** The variant set is closed (five rules, with no additional variants planned). Trait objects add indirection cost without extensibility benefit. The enum representation allows the entire `StoppingRuleSet` to be stack-friendly (`Vec<StoppingRule>` is a flat vector of enum-sized elements).

**Performance characteristics:** The `evaluate` method is called once per rule per iteration. With a typical set of 2-3 rules, this is 2-3 match dispatches per iteration -- negligible compared to the LP solve cost that dominates each iteration.

## Cross-References

- [Stopping Rules](../math/stopping-rules.md) -- Mathematical definitions of all five stopping rules: iteration limit (SS2), time limit (SS3), bound stalling formula (SS4), simulation-based two-phase algorithm (SS5), combination logic (SS6)
- [Convergence Monitoring](./convergence-monitoring.md) -- Convergence monitor architecture (SS2), tracked quantities (SS2.1), bound stalling detection (SS2.2), evaluation protocol (SS2.3), termination event schema (SS4.2)
- [Risk Measures SS10](../math/risk-measures.md) -- Lower bound invalidity under CVaR: the stage-1 LP objective is a convergence indicator, not a valid bound
- [Extension Points](./extension-points.md) -- Variant architecture overview (SS1), cross-variant composition rules (SS8.1, rule X2: CVaR + simulation stopping), dispatch mechanism analysis (SS7)
- [Training Loop](./training-loop.md) -- Iteration lifecycle (SS2.1) where convergence update and stopping rule evaluation occur (step 5), termination conditions (SS2.2)
- [Configuration Reference](../configuration/configuration-reference.md) -- JSON schema for `stopping_rules` array and `stopping_mode` field in `config.json`
- [Upper Bound Evaluation](../math/upper-bound-evaluation.md) -- Monte Carlo simulation for upper bound estimation, consumed by the simulation-based stopping rule
- [RiskMeasure Trait](./risk-measure-trait.md) -- Sibling enum-dispatch trait pattern (SS4 dispatch rationale); `evaluate_risk` method (SS2.2) used for risk-adjusted simulation costs
- [HorizonMode Trait](./horizon-mode-trait.md) -- Sibling enum-dispatch trait pattern (SS4 dispatch rationale)
- [SamplingScheme Trait](./sampling-scheme-trait.md) -- Sibling enum-dispatch trait pattern (SS4 dispatch rationale)
- [CutSelectionStrategy Trait](./cut-selection-trait.md) -- Sibling enum-dispatch trait pattern (SS4 dispatch rationale)
- [Communicator Trait](../hpc/communicator-trait.md) -- Reference pattern for trait specification structure, convention blockquote, and method contract format
