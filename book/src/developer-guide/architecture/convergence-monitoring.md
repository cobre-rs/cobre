# Convergence Monitoring

## Purpose

This spec defines the Cobre SDDP convergence monitoring architecture: the convergence criteria and stopping rules, the convergence monitor's tracked quantities and evaluation logic, bound computation including cross-rank aggregation, and the training log format for progress reporting. For the mathematical foundations, see [Stopping Rules](../math/stopping-rules.md) and [Upper Bound Evaluation](../math/upper-bound-evaluation.md).

## 1. Convergence Criteria

SDDP convergence is determined by the gap between lower and upper bounds on the optimal objective:

**Lower Bound (LB):**

- The objective value of the stage-1 LP, which includes both the immediate stage-1 cost and the future cost approximation $\theta_2$ from accumulated cuts: $LB = \min_x \{ c_1^\top x_1 + \theta_2 : \text{constraints} \}$
- Deterministic: all scenarios share the same initial state $x_0$, so the stage-1 LP is identical regardless of scenario — only one solve is needed
- Monotonically non-decreasing across iterations, since cuts only tighten the FCF approximation

**Upper Bound (UB):**

- Statistical estimate from forward simulation: the mean total cost across all $N$ scenario trajectories
- $UB_k = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} c_t^{(i)}$
- Includes 95% confidence interval: $UB \pm 1.96 \cdot \sigma / \sqrt{N}$
- Not monotonic — depends on the scenarios sampled in each iteration
- For risk-averse policies (CVaR), the upper bound computation incorporates the risk measure weighting; see [Upper Bound Evaluation](../math/upper-bound-evaluation.md)

**Convergence Gap:**

$$
\text{gap} = \frac{UB - LB}{\max(1, |UB|)}
$$

The $\max(1, |UB|)$ denominator smoothly handles the near-zero case by clamping the denominator to at least 1, matching the formula in [SDDP Algorithm SS3.3](../math/sddp-algorithm.md). The gap is computed and logged each iteration for progress reporting, but is **not** itself a stopping criterion in standard SDDP — convergence is determined by the stopping rules below.

> **Complete tree mode note**: In [complete tree mode](./scenario-generation.md) (deferred — see [C.12](../deferred.md)), all scenarios from the stochastic tree are enumerated exhaustively. The upper bound becomes **deterministic** (exact expected cost, not a statistical estimate), and the gap becomes an exact convergence measure. In this case, a gap-based stopping criterion becomes meaningful and should be added when complete tree mode is implemented. This differs from standard SDDP where the upper bound is a noisy Monte Carlo estimate.

**Stopping Rules:**

The convergence monitor evaluates the stopping rules defined in [Stopping Rules](../math/stopping-rules.md). The available rules and their config parameters are:

| Rule              | Condition                                                                     | Config Type       |
| ----------------- | ----------------------------------------------------------------------------- | ----------------- |
| Iteration limit   | Iteration count $\geq$ `limit`                                                | `iteration_limit` |
| Time limit        | Wall-clock time $\geq$ `seconds`                                              | `time_limit`      |
| Bound stalling    | LB relative improvement over `iterations` window $<$ `tolerance`              | `bound_stalling`  |
| Simulation-based  | Bound stable AND simulated policy costs stable (checked every `period` iters) | `simulation`      |
| Graceful shutdown | External signal received (checkpoints last **completed** iteration)           | OS signal         |

Rules combine via `stopping_mode`: `"any"` (default, OR logic) or `"all"` (AND logic). See [Stopping Rules](../math/stopping-rules.md) for full mathematical definitions and configuration schema.

## 2. Convergence Monitor

The convergence monitor is updated once per iteration (after the backward pass, cut synchronization, and lower bound evaluation) and evaluates all stopping rules. The UB statistics come from the post-forward `allreduce`; the LB comes from a separate post-backward computation (rank-0 solves all stage-0 openings and broadcasts — see [SS3.2](#32-lower-bound-properties)).

### 2.1 Tracked Quantities

The monitor maintains the following per-iteration history:

| Quantity                  | Type       | Description                                                                |
| ------------------------- | ---------- | -------------------------------------------------------------------------- |
| Lower bound               | `f64`      | Risk-adjusted aggregation of stage-0 LP objectives across all openings     |
| Upper bound               | `f64`      | Mean total forward cost across all scenarios                               |
| Upper bound std           | `f64`      | Standard deviation of total forward costs                                  |
| Gap                       | `f64`      | Relative gap $(UB - LB) / \max(1, \lvert UB \rvert)$                       |
| Iteration wall-clock time | `Duration` | Elapsed time for the full iteration (forward + backward + synchronization) |

### 2.2 Bound Stalling Detection

The bound stalling rule tracks the relative improvement of the lower bound over a configurable window. Given the `bound_stalling` configuration with parameters `iterations` ($\tau$) and `tolerance`:

$$
\Delta_k = \frac{\underline{z}^k - \underline{z}^{k-\tau}}{\max(1, |\underline{z}^k|)}
$$

The rule triggers when $|\Delta_k| < \text{tolerance}$.

This is the same formula defined in [Stopping Rules SS4](../math/stopping-rules.md). The convergence monitor maintains the bound history needed to evaluate this windowed comparison.

### 2.3 Convergence Evaluation

At each iteration, the monitor evaluates all configured stopping rules:

1. **Iteration limit** — If iteration count $\geq$ `limit`, report terminated
2. **Time limit** — If cumulative wall-clock time $\geq$ `seconds`, report terminated
3. **Bound stalling** — If LB relative improvement over the last `iterations` window is below `tolerance`, report converged
4. **Simulation-based** — If the check period has elapsed: test bound stability, then run Monte Carlo simulations and compare to previous; if both stable, report converged. See [Stopping Rules SS5](../math/stopping-rules.md)
5. **Graceful shutdown** — If an external signal flag is set, report terminated

The `stopping_mode` determines whether the first satisfied rule terminates (mode `"any"`) or all must be satisfied (mode `"all"`). The termination reason is recorded in the training log and output metadata.

### 2.3a Simulation-Based Stopping Rule Integration

This subsection specifies how the simulation-based stopping rule ([Stopping Rules SS5](../math/stopping-rules.md)) interacts with the training loop. It resolves the execution model, workspace reuse, scenario sourcing, parallel distribution, and first-check semantics.

#### Execution Model

The simulation check runs **synchronously** — it blocks the training loop iteration. When the check period triggers (iteration $k$ is a multiple of `period`), the convergence monitor first evaluates bound stability. If the bound is stable, the monitor initiates a simulation forward pass before returning control to the training loop. No background or asynchronous execution occurs. This design avoids concurrent access to solver workspaces and eliminates the need for additional synchronization primitives.

The simulation check executes as part of step 5 ("Convergence update") in the [Training Loop SS2.1](./training-loop.md) iteration lifecycle.

#### Solver Workspace Reuse

The simulation forward pass reuses the same thread-local solver workspaces as the training forward pass ([Solver Workspaces SS1](./solver-workspaces.md)). No separate LP instances are created. Each thread uses its existing solver instance, RHS patch buffer, primal buffer, and dual buffer to solve simulation scenarios exactly as it would for training scenarios. The per-stage basis cache is **not** updated by simulation solves — the cache retains the training iteration's basis to preserve warm-start quality for the next training iteration.

#### Scenario Source

Simulation scenarios are **fresh InSample draws** from the fixed opening tree. At each stage, a random opening index $j \in \{0, \ldots, N_{\text{openings}} - 1\}$ is sampled to select the noise vector for that stage's realization, following the same InSample mechanism as the training forward pass ([Scenario Generation SS3.2](./scenario-generation.md)).

The simulation uses a **separate seed** to ensure its draws are independent of the training forward pass draws for the same iteration. The simulation seed is derived as:

```
simulation_seed = seed_derive(base_seed + iteration, sim_scenario, stage)
```

where `base_seed + iteration` serves as the effective base seed for the simulation check at iteration $k$, and `seed_derive` is the SipHash-1-3 derivation function specified in [Scenario Generation SS2.2a](./scenario-generation.md). The addition `base_seed + iteration` (wrapping `u64` addition) produces a base seed distinct from the training forward pass base seed for the same iteration, guaranteeing independent draws without requiring a separate seed derivation input layout.

> **Convention:** The simulation check MUST NOT use External or Historical scenario sources. It always draws from the fixed opening tree via the InSample scheme.

#### Parallel Distribution

The `replications` simulation scenarios (typically 100) are distributed across MPI ranks using **contiguous block assignment**, the same distribution strategy as the training forward pass ([Training Loop SS4.3](./training-loop.md)). Within each rank, **all Rayon worker threads** participate in the simulation forward pass using the same thread-trajectory affinity pattern: each thread owns complete trajectories and solves all stages sequentially for its assigned scenarios.

Given $R$ ranks and $N_{\text{rep}}$ replications, each rank processes $\lfloor N_{\text{rep}} / R \rfloor$ or $\lceil N_{\text{rep}} / R \rceil$ scenarios. After all ranks complete, a single `allreduce` aggregates the per-stage cost sums across ranks, yielding the global per-stage mean costs $c_t^{new}$ used in the normalized distance comparison.

#### Comparison Metric

The per-stage mean costs $c_t^{new}$ from the current simulation check are compared to the previous simulation check's per-stage costs $c_t^{old}$ using the normalized distance formula from [Stopping Rules SS5](../math/stopping-rules.md) step 2:

$$
d = \sqrt{\sum_{t=1}^{T} \left( \frac{c_t^{new} - c_t^{old}}{\max(1, |c_t^{old}|)} \right)^2}
$$

Convergence triggers when the bound is stable AND $d < \text{distance\_tol}$.

#### First-Check Behavior

On the **first** simulation check (the first iteration where the period condition triggers and the bound is stable), there is no previous simulation result to compare against. The simulation runs and stores its per-stage costs as the baseline $c_t^{old}$, but does **not** trigger convergence. A valid distance comparison requires at least two simulation checks.

#### Performance Note

At production scale ($T = 120$ stages, $N_{\text{rep}} = 100$ replications, $R = 16$ ranks), each rank solves approximately $\lceil 100 / 16 \rceil \times 120 = 7 \times 120 = 840$ LP subproblems during the simulation check. Using the ~2 ms per LP solve estimate from [Production Scale Reference](../overview/production-scale-reference.md), the wall-clock time is approximately $840 \times 2\text{ms} = 1.68\text{s} \approx 1.7\text{s}$ per rank. Since all ranks execute in parallel, the total wall-clock cost of a simulation check is ~1.7 seconds — modest relative to the full training iteration time.

### 2.4 Per-Iteration Output Record

Each iteration produces a record with the following fields, used for logging (SS4) and output persistence:

| Field             | Description                                                         |
| ----------------- | ------------------------------------------------------------------- |
| `iteration`       | Iteration index (1-based)                                           |
| `lower_bound`     | LB value                                                            |
| `upper_bound`     | UB value (mean forward cost)                                        |
| `upper_bound_std` | Standard deviation of forward costs                                 |
| `ci_95`           | 95% confidence interval half-width ($1.96 \cdot \sigma / \sqrt{N}$) |
| `gap`             | Relative gap (informational, not a stopping criterion)              |
| `wall_time`       | Cumulative wall-clock time                                          |
| `iteration_time`  | Wall-clock time for this iteration                                  |

## 3. Bound Computation

![Convergence bound evolution — lower bound increases monotonically as cuts are added, upper bound decreases with confidence band, gap narrows until stopping rule triggers](../../images/d21-convergence-bounds.svg)

### 3.1 Cross-Rank Aggregation

Forward pass scenarios are distributed across MPI ranks (see [Training Loop SS4.3](./training-loop.md)). After the forward pass, **upper bound** statistics must be aggregated globally. This is done via a single `MPI_Allreduce` with `ReduceOp::Sum` that collects three sufficient statistics from each rank:

| Statistic        | Per-rank value                                       | Reduction operation |
| ---------------- | ---------------------------------------------------- | ------------------- |
| Scenario count   | Number of trajectories solved by this rank           | Sum                 |
| Cost sum         | Sum of total costs across local trajectories         | Sum                 |
| Cost sum-squares | Sum of squared total costs across local trajectories | Sum                 |

From the global aggregates, the upper bound statistics are computed:

- **Mean**: $\bar{c} = \text{sum} / N$
- **Variance** (Bessel-corrected): $s^2 = \frac{\text{sum\_sq} - N \cdot \bar{c}^2}{N - 1}$
- **Standard deviation**: $\sigma = \sqrt{s^2}$
- **95% CI half-width**: $1.96 \cdot \sigma / \sqrt{N}$

This single allreduce (3 doubles) is sufficient — no per-scenario data needs to be communicated. See [SS3.1a](#31a-upper-bound-variance-aggregation) for the complete aggregation protocol, numerical stability analysis, and edge cases.

### 3.1a Upper Bound Variance Aggregation

This subsection specifies the complete protocol for computing the upper bound mean, sample variance, standard deviation, and 95% confidence interval from distributed forward pass results. It resolves [GAP-038](../overview/spec-gap-inventory.md).

#### Per-Rank Local Computation

After the forward pass, each rank $r$ has solved $M_r$ scenario trajectories and recorded their total costs $\{c_1, c_2, \ldots, c_{M_r}\}$. Each rank computes three local sufficient statistics:

| Statistic      | Formula                  | Type                |
| -------------- | ------------------------ | ------------------- |
| `local_count`  | $M_r$                    | `u64` cast to `f64` |
| `local_sum`    | $\sum_{i=1}^{M_r} c_i$   | `f64`               |
| `local_sum_sq` | $\sum_{i=1}^{M_r} c_i^2$ | `f64`               |

#### Allreduce Payload

The three local statistics are packed into a contiguous `[f64; 3]` array in the order:

```
[sum, sum_sq, count]
```

and aggregated via a single `allreduce` with `ReduceOp::Sum`. After the reduction, every rank holds the global aggregates:

$$
N = \sum_r M_r, \qquad S = \sum_r \text{local\_sum}_r, \qquad Q = \sum_r \text{local\_sum\_sq}_r
$$

> **Separation from LB evaluation.** The UB statistics `allreduce` runs post-forward. The lower bound is evaluated separately post-backward by rank 0 (see [SS3.2](#32-lower-bound-properties) and [Training Loop SS4.3b](./training-loop.md)).

#### Global Variance Formula

From the global aggregates, compute:

$$
\bar{c} = \frac{S}{N}
$$

$$
s^2 = \frac{Q - N \cdot \bar{c}^2}{N - 1}
$$

$$
\sigma = \sqrt{s^2}
$$

$$
\text{CI}_{95} = 1.96 \cdot \frac{\sigma}{\sqrt{N}}
$$

The denominator $N - 1$ is **Bessel's correction**, producing the unbiased sample variance. This is required because the $N$ forward pass scenarios are a sample from the underlying stochastic process, not the full population. The 95% confidence interval assumes approximate normality of the sample mean by the central limit theorem (valid for typical SDDP forward pass counts $N \geq 20$).

#### Numerical Stability

The single-pass formula $Q - N \cdot \bar{c}^2$ can suffer from **catastrophic cancellation** when the coefficient of variation $\sigma / \bar{c}$ is small — i.e., when costs are large in magnitude but tightly distributed. The relative error is bounded by:

$$
\varepsilon_{\text{rel}} \lesssim \varepsilon_{\text{mach}} \cdot N \cdot \left(\frac{\bar{c}}{\sigma}\right)^2
$$

where $\varepsilon_{\text{mach}} \approx 2.2 \times 10^{-16}$ for `f64`. At production scale ($N = 100$ scenarios, costs $\sim 10^6$, $\sigma / \bar{c} \approx 0.05\text{--}0.10$), this bound evaluates to:

$$
\varepsilon_{\text{rel}} \lesssim 2.2 \times 10^{-16} \times 100 \times (20)^2 \approx 10^{-12}
$$

This is well within acceptable precision for convergence monitoring. **The single-pass formula is the baseline.** If higher numerical stability is required (e.g., for post-processing validation with extreme cost distributions), a two-pass approach (first allreduce for the mean, second allreduce for centered sum-of-squares $\sum (c_i - \bar{c})^2$) can be used — but this doubles the synchronization cost and is deferred as an optimization.

#### Edge Case: Single Scenario ($N = 1$)

When $N = 1$, the Bessel-corrected variance involves division by $N - 1 = 0$. In this case:

- Set $\sigma = 0$
- Set $\text{CI}_{95} = 0$
- Log a **warning** that the upper bound has no statistical validity with a single scenario (no variance estimate is possible)

The upper bound mean $\bar{c}$ is still valid as a point estimate but carries no confidence interval.

#### Cross-References

- [Training Loop SS4.3b](./training-loop.md) — Two-call allreduce baseline (LB `Min` + UB `Sum`)
- [Convergence Monitoring SS1](#1-convergence-criteria) — Upper bound definition and 95% CI usage in convergence gap
- [Convergence Monitoring SS2.4](#24-per-iteration-output-record) — `upper_bound_std` and `ci_95` output fields consuming these computed values

### 3.2 Lower Bound Properties

The lower bound is evaluated **after the backward pass** by rank 0. Rank 0 iterates over all stage-0 openings in the fixed opening tree, solves the stage-0 LP for each opening with the latest FCF cuts (including new cuts from the current iteration's backward pass), and aggregates the per-opening objectives via the stage-0 risk measure. The scalar result is broadcast to all ranks via `comm.broadcast()`. See [Training Loop SS4.3b](./training-loop.md) for the full algorithm.

Key properties:

- **Monotonically non-decreasing** — Each iteration adds cuts that can only tighten the FCF under-approximation, so $LB_{k+1} \geq LB_k$
- **Valid lower bound** — The stage-0 objective with the current FCF under-approximates the true expected cost, so $LB_k \leq z^*$ for all $k$ (under Expectation; under CVaR this is a convergence indicator — see [Risk Measures SS10](../math/risk-measures.md))
- **Risk-measure-aware** — The LB applies the stage-0 risk measure (Expectation or CVaR) with uniform opening probabilities, correctly handling non-Expectation risk measures
- **Post-backward timing** — The LB uses the latest cuts, producing the tightest available bound at each iteration. The previous forward pass UB is stale after the backward pass adds new cuts; gap comparison requires a new forward pass

### 3.3 Infinite Horizon Considerations

In cyclic (infinite horizon) mode, the discount factor $d < 1$ ensures that the infinite sum of stage costs converges. The bound computation must account for this:

- Stage costs in the forward pass are discounted: the cost at stage $t$ is weighted by $d^{t-1}$
- The upper bound is the mean of discounted total trajectory costs
- The lower bound naturally incorporates discounting through the $\theta$ variable, which carries the discounted future cost approximation

See [Infinite Horizon](../math/infinite-horizon.md) for the mathematical treatment of discount factors in the SDDP context.

## 4. Training Log Format

The convergence monitor emits a structured log each iteration. The log includes a header (emitted once at training start) and per-iteration lines:

**Header:**

```
═══════════════════════════════════════════════════════════════════
Cobre SDDP Training
Case: <case_name>
Started: <timestamp>
Ranks: <R> | Threads/rank: <T> | Stages: <S> | Hydros: <H>
═══════════════════════════════════════════════════════════════════
```

**Per-iteration line:**

```
Iter <n> | LB: <value> | UB: <value> ± <ci_95> | Gap: <value>%
```

**Termination summary:**

```
═══════════════════════════════════════════════════════════════════
<TERMINATION_REASON> after <n> iterations (<reason detail>)
Total time: <elapsed> | Avg iteration: <avg>
Final LB: <value> | Final UB: <value> ± <ci_95>
Total cuts: <count> | Cuts/stage: ~<avg>
═══════════════════════════════════════════════════════════════════
```

The termination reason is one of: `BOUND_STALLING`, `SIMULATION`, `ITERATION_LIMIT`, `TIME_LIMIT`, or `SHUTDOWN` (graceful signal).

### 4.1 JSON-Lines Streaming Schema

When `--output-format json-lines` is specified, the training log is emitted as newline-delimited JSON instead of the text format above. Each line is a self-describing JSON object with a `type` field. The field values match the per-iteration output record defined in SS2.4.

**Progress event** (one per iteration):

```json
{
  "type": "progress",
  "iteration": 1,
  "lower_bound": 1234567.89,
  "upper_bound": 1345678.9,
  "upper_bound_std": 12345.67,
  "ci_95": 2420.73,
  "gap": 0.0899,
  "wall_time_ms": 45200,
  "iteration_time_ms": 45200
}
```

**Started event** (emitted once, replaces the text header):

```json
{
  "type": "started",
  "case": "/data/case_001",
  "stages": 120,
  "hydros": 164,
  "thermals": 130,
  "ranks": 8,
  "threads_per_rank": 16,
  "timestamp": "2026-02-25T10:00:00Z"
}
```

The text log and JSON-lines are **mutually exclusive output modes** for the same event stream. When `--output-format human` (default), the text log is emitted. When `--output-format json-lines`, the JSON-lines stream is emitted. The Parquet convergence log (`training/convergence.parquet`) is always written to disk regardless of output mode. See [Structured Output SS3](../interfaces/structured-output.md) for the complete JSON-lines streaming protocol.

### 4.2 Termination Event Schema

When the training loop terminates, a structured termination event is emitted (replaces the text termination summary):

```json
{
  "type": "terminated",
  "reason": "bound_stalling",
  "iterations": 87,
  "final_lb": 72105.4,
  "final_ub": 73211.8,
  "total_time_ms": 3912000,
  "total_cuts": 16704
}
```

The `reason` field uses the same values as the text termination reason: `bound_stalling`, `simulation`, `iteration_limit`, `time_limit`, or `shutdown`. After the termination event, a final `result` event provides the response envelope for the overall command outcome. See [Structured Output SS3](../interfaces/structured-output.md) for the `result` event schema.

## Cross-References

- [Stopping Rules](../math/stopping-rules.md) — Mathematical definitions of the stopping rules implemented by the convergence monitor
- [Upper Bound Evaluation](../math/upper-bound-evaluation.md) — Statistical upper bound theory, confidence interval construction, and bias corrections
- [Risk Measures](../math/risk-measures.md) — CVaR risk measure, which affects upper bound computation for risk-averse policies
- [Infinite Horizon](../math/infinite-horizon.md) — Discount factor treatment for cyclic stage graphs
- [Training Loop](./training-loop.md) — The SDDP training loop that invokes this convergence monitor each iteration (SS2.1 step 5 convergence update, SS4.3 forward pass distribution)
- [Solver Workspaces](./solver-workspaces.md) — Thread-local solver infrastructure reused by the simulation-based stopping rule (SS1)
- [Scenario Generation](./scenario-generation.md) — Seed derivation function (SS2.2a) and InSample sampling scheme (SS3.2) used for simulation check draws
- [Synchronization](../hpc/synchronization.md) — MPI synchronization points including `MPI_Allreduce` for convergence statistics (§1.3)
- [Configuration Reference](../configuration/configuration-reference.md) — JSON schema for `stopping_rules` and `stopping_mode`
- [Structured Output](../interfaces/structured-output.md) — JSON-lines streaming protocol and response envelope schema
- [Terminal UI](../interfaces/terminal-ui.md) — TUI convergence plot consuming the same per-iteration record
