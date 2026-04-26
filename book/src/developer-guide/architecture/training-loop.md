# Training Loop

## Purpose

This spec defines the Cobre SDDP training loop architecture: the core training components, their configurable abstraction points, forward pass execution with sampling scheme parameterization and parallel distribution, backward pass execution with opening tree evaluation and cut generation, state management, and dual extraction for cut coefficients.

## 1. SDDP Algorithm Overview

The training phase implements the Stochastic Dual Dynamic Programming (SDDP) algorithm, iteratively constructing piecewise-linear approximations of the expected future cost function (FCF) through forward simulation and backward cut generation.

Each iteration consists of three phases:

1. **Forward pass** — Sample $M$ scenarios via the configured **sampling scheme**, solve the LP at each stage with the current FCF, record visited states and stage costs for the upper bound
2. **Backward pass** — For each stage $T$ down to 2, evaluate the cost-to-go from each visited state under **all** openings from the fixed opening tree, extract LP duals, and compute new cuts via the risk measure
3. **Convergence check** — Update the upper bound estimate (mean forward cost), compute the gap $(UB - LB) / |UB|$, and test stopping rules (gap tolerance, stable LB, iteration/time limits)

The loop terminates when converged or a limit is reached, outputting the FCF cuts and bound history.

## 2. Training Orchestrator Components

The training orchestrator manages the iterative SDDP loop and coordinates the following components:

| Component               | Responsibility                                                                                            | Configuration Source                        |
| ----------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **Risk Measure**        | Determines how backward outcomes are aggregated into cut coefficients (expectation vs CVaR)               | `risk_measure` per stage in `stages.json`   |
| **Cut Formulation**     | Determines cut structure (single-cut; multi-cut is deferred)                                              | Fixed: single-cut                           |
| **Horizon Mode**        | Determines stage transitions, terminal conditions, and discount factors                                   | `policy_graph` in `stages.json`             |
| **Sampling Scheme**     | Determines how the forward pass selects scenario realizations at each stage                               | `training.scenario_source` in `config.json` |
| **FCF**                 | Stores accumulated Benders cuts per stage; queried during LP construction and updated after backward pass | Built incrementally across iterations       |
| **Convergence Monitor** | Tracks lower/upper bounds, gap history, and evaluates stopping rules                                      | `stopping_rules` in `config.json`           |

### 2.1 Iteration Lifecycle

Each iteration follows a fixed sequence:

1. **Forward pass** — Execute $M$ scenario trajectories (SS4)
2. **Forward synchronization** — `allreduce` ([Communicator Trait SS2.2](../hpc/communicator-trait.md)) aggregates upper bound statistics across ranks
3. **State exchange** — Exchange visited state data between ranks so all ranks have the full set of trial points for the backward pass
4. **Backward pass** — Generate cuts from visited states (SS6). After each per-stage state exchange, archive gathered states into the `VisitedStatesArchive` (see SS6.4a).
5. **Cut synchronization** — `allgatherv` ([Communicator Trait SS2.1](../hpc/communicator-trait.md)) distributes new cuts to all ranks
   5a. **Cut selection** (conditional: `should_run(iteration)`) — Stage 0 is exempt (see SS6.4a); stages $1 \ldots T-1$ distributed across threads via `into_par_iter()`, each thread calls `select_for_stage` on its assigned stages, deactivations applied sequentially (see [Cut Selection Strategy Trait SS2.2a](./cut-selection-trait.md) and [Cut Selection Strategy Trait SS6.4.4](./cut-selection-trait.md))
   5b. **Lower bound evaluation** — Rank 0 iterates all stage-0 openings, solves the LP for each with the current FCF cuts, aggregates per-opening objectives via the stage-0 risk measure, and broadcasts the scalar LB to all ranks via `comm.broadcast()` (see [Convergence Monitoring SS3.2](./convergence-monitoring.md))
6. **Convergence update** — Update bound estimates, evaluate stopping rules (see [Convergence Monitoring](./convergence-monitoring.md))
7. **Checkpoint** — If the checkpoint interval has elapsed, persist current FCF and iteration state (see [Checkpointing](../hpc/checkpointing.md))
8. **Logging** — Emit iteration summary (bounds, gap, timings)

### 2.1a Event Emission Points

Each step in the iteration lifecycle (SS2.1) emits a typed event to the shared event channel when an event sender is registered. These events feed all runtime consumers: text logger, JSON-lines writer, TUI renderer, MCP progress notifications, and Parquet convergence writer. Event types are defined in `cobre-core`.

| Step | Lifecycle Phase         | Event Type             | Payload Summary                                                                                                                                   |
| ---- | ----------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | Forward pass            | `ForwardPassComplete`  | iteration, scenarios, ub_mean, ub_std, elapsed_ms                                                                                                 |
| 2    | Forward synchronization | `ForwardSyncComplete`  | iteration, global_ub_mean, global_ub_std, sync_time_ms                                                                                            |
| 3    | State exchange          | _(no dedicated event)_ | State exchange timing is reported within `BackwardPassComplete.state_exchange`                                                                    |
| 4    | Backward pass           | `BackwardPassComplete` | iteration, cuts_generated, stages_processed, elapsed_ms, state_exchange, cut_batch_build, rayon_overhead                                          |
| 5    | Cut synchronization     | `CutSyncComplete`      | iteration, cuts_distributed, cuts_active, cuts_removed, sync_time_ms                                                                              |
| 5a   | Cut selection           | `CutSelectionComplete` | iteration, cuts_deactivated, stages_processed, selection_time_ms, allgatherv_time_ms (only emitted when `should_run` is true)                     |
| 6    | Convergence update      | `ConvergenceUpdate`    | iteration, lower_bound, upper_bound, upper_bound_std, gap, rules_evaluated[] (also carries lower bound evaluation results; see SS5b in lifecycle) |
| 7    | Checkpoint              | `CheckpointComplete`   | iteration, checkpoint_path, elapsed_ms (only when checkpoint interval triggers)                                                                   |
| 8    | Logging                 | `IterationSummary`     | iteration, lower_bound, upper_bound, gap, wall_time_ms, iteration_time_ms, forward_ms, backward_ms, lp_solves, solve_time_ms                      |

**Lifecycle events** (emitted once per training/simulation run, not per iteration):

| Event Type           | Trigger                     | Payload Summary                                                                          |
| -------------------- | --------------------------- | ---------------------------------------------------------------------------------------- |
| `TrainingStarted`    | Training loop entry         | case_name, stages, hydros, thermals, ranks, threads_per_rank, timestamp                  |
| `TrainingFinished`   | Training loop exit          | reason, iterations, final_lb, final_ub, total_time_ms, total_cuts                        |
| `SimulationProgress` | Simulation batch completion | scenarios_complete, scenarios_total, elapsed_ms, scenario_cost, solve_time_ms, lp_solves |
| `SimulationFinished` | Simulation completion       | scenarios, output_dir, elapsed_ms                                                        |

The event channel uses an `Option<std::sync::mpsc::Sender<TrainingEvent>>` pattern: when `None`, no events are emitted (zero overhead for library-mode callers). When `Some(sender)`, events are emitted at each step boundary. The channel has a single receiver. Multiple output sinks (text logger, JSON-lines writer, TUI renderer, Parquet convergence writer) are served by a single consumer thread that dispatches each received event to all registered sinks. This fan-out is internal to the consumer, not a property of the channel. See [Convergence Monitoring SS4.1](./convergence-monitoring.md) for the JSON-lines schema, [Terminal UI](../interfaces/terminal-ui.md) for TUI consumption, and [MCP Server](../interfaces/mcp-server.md) for MCP progress notifications.

> **Design note.** The event channel uses `std::sync::mpsc` from the Rust standard library. This avoids introducing `tokio` or `crossbeam` as dependencies in `cobre-sddp` or `cobre-core`. The training loop is synchronous -- it runs inside an MPI process with no async runtime. The channel is unbounded (`mpsc::channel()`) because events are small (< 1 KB each) and emitted at most 8 times per iteration (one per lifecycle step in SS2.1a), so memory pressure from buffered events is negligible. Deferred async interface crates (`cobre-python`, `cobre-mcp`) may bridge to `tokio::sync::broadcast` or equivalent async channels in their own event adapters.

### 2.1b TrainingEvent Type Definitions

This subsection provides the concrete Rust type definitions for the `TrainingEvent` enum and its payload structs. The enum lives in **cobre-core** (not cobre-sddp) because event types are consumed by cobre-cli, cobre-tui, and cobre-mcp -- all of which depend on cobre-core but not on cobre-sddp. This placement avoids a reverse dependency from interface crates back into the algorithm crate.

#### Derive traits

`TrainingEvent` and all payload structs derive `Clone` and `Debug`. They do **not** require `Send + Sync` because the event channel transfers ownership: the sender moves events into the channel, and each consumer receives an owned clone. There is no shared mutable access to event values.

#### Timestamp policy

Events do **not** carry wall-clock timestamps. The consumer (text logger, JSON-lines writer, TUI renderer) is responsible for capturing `Instant::now()` or `SystemTime::now()` upon receipt. This avoids `clock_gettime` syscall overhead in the hot path (forward and backward pass events are emitted thousands of times per training run). The single exception is the `timestamp` field in `TrainingStarted`, which records the training start wall-clock time once at entry -- this is not a per-event timestamp but a run-level metadata field.

#### Helper struct: StoppingRuleResult

The `ConvergenceUpdate` variant carries a vector of stopping rule evaluation results. Each element reports one rule's outcome for the current iteration:

```rust
/// Result of evaluating a single stopping rule at a given iteration.
#[derive(Clone, Debug)]
pub struct StoppingRuleResult {
    /// Rule identifier matching the variant name in the stopping rules config
    /// (e.g., "graceful_shutdown", "bound_stalling", "iteration_limit", "time_limit", "simulation_based").
    pub rule_name: String,
    /// Whether this rule's condition is satisfied at the current iteration.
    pub triggered: bool,
    /// Human-readable description of the rule's current state
    /// (e.g., "gap 0.42% <= 1.00%", "LB stable for 12/10 iterations").
    pub detail: String,
}
```

See [Convergence Monitoring](./convergence-monitoring.md) SS2 for the stopping rule definitions and evaluation logic.

#### TrainingEvent enum

The enum has exactly 12 variants: 8 per-iteration events (one per lifecycle step in SS2.1a) and 4 lifecycle events (emitted once per training or simulation run).

```rust
/// Typed events emitted by the SDDP training loop and simulation runner.
/// Defined in cobre-core. Consumed by cobre-cli, cobre-tui, and cobre-mcp.
#[derive(Clone, Debug)]
pub enum TrainingEvent {
    // ── Per-iteration events (8) ────────────────────────────────────

    /// Step 1: Forward pass completed for this iteration on the local rank.
    ForwardPassComplete {
        iteration: u64,
        /// Number of forward scenarios evaluated on this rank.
        scenarios: u32,
        /// Mean total forward cost across local scenarios.
        ub_mean: f64,
        /// Standard deviation of total forward cost across local scenarios.
        ub_std: f64,
        /// Wall-clock time for the forward pass on this rank, in milliseconds.
        elapsed_ms: u64,
    },

    /// Step 2: Forward synchronization (allreduce) completed.
    ForwardSyncComplete {
        iteration: u64,
        /// Global upper bound mean after allreduce.
        global_ub_mean: f64,
        /// Global upper bound standard deviation after allreduce.
        global_ub_std: f64,
        /// Wall-clock time for the MPI synchronization, in milliseconds.
        sync_time_ms: u64,
    },

    /// Step 4: Backward pass completed for this iteration (includes step 3 state exchange timing).
    BackwardPassComplete {
        iteration: u64,
        /// Number of new cuts generated across all stages.
        cuts_generated: u32,
        /// Number of stages processed in the backward sweep.
        stages_processed: u32,
        /// Wall-clock time for the backward pass, in milliseconds.
        elapsed_ms: u64,
        /// Time spent exchanging state data between ranks.
        state_exchange: Duration,
        /// Time spent building cut batches.
        cut_batch_build: Duration,
        /// Rayon parallelism overhead.
        rayon_overhead: Duration,
    },

    /// Step 5: Cut synchronization (allgatherv) completed.
    CutSyncComplete {
        iteration: u64,
        /// Number of cuts distributed to all ranks via allgatherv.
        cuts_distributed: u32,
        /// Total number of active cuts in the FCF after synchronization.
        cuts_active: u32,
        /// Number of cuts removed by cut selection after synchronization.
        cuts_removed: u32,
        /// Wall-clock time for the MPI synchronization, in milliseconds.
        sync_time_ms: u64,
    },

    /// Step 5a: Cut selection completed (only emitted when `should_run` is true).
    CutSelectionComplete {
        iteration: u64,
        /// Number of cuts deactivated by the selection strategy.
        cuts_deactivated: u32,
        /// Number of stages processed by the selection strategy.
        stages_processed: u32,
        /// Wall-clock time for the cut selection pass, in milliseconds.
        selection_time_ms: u64,
        /// Wall-clock time for the allgatherv of deactivation masks, in milliseconds.
        allgatherv_time_ms: u64,
    },

    /// Step 6: Convergence check completed.
    ConvergenceUpdate {
        iteration: u64,
        /// Current lower bound (non-decreasing).
        lower_bound: f64,
        /// Current upper bound (statistical estimate).
        upper_bound: f64,
        /// Standard deviation of the upper bound estimate.
        upper_bound_std: f64,
        /// Relative optimality gap: (UB - LB) / |UB|.
        gap: f64,
        /// Evaluation result for each configured stopping rule.
        rules_evaluated: Vec<StoppingRuleResult>,
    },

    /// Step 7: Checkpoint written (only emitted when the checkpoint interval triggers).
    CheckpointComplete {
        iteration: u64,
        /// Filesystem path where the checkpoint was written.
        checkpoint_path: String,
        /// Wall-clock time for the checkpoint write, in milliseconds.
        elapsed_ms: u64,
    },

    /// Step 8: Full iteration summary with aggregated timings.
    IterationSummary {
        iteration: u64,
        lower_bound: f64,
        upper_bound: f64,
        /// Relative optimality gap: (UB - LB) / |UB|.
        gap: f64,
        /// Cumulative wall-clock time since training started, in milliseconds.
        wall_time_ms: u64,
        /// Wall-clock time for this iteration only, in milliseconds.
        iteration_time_ms: u64,
        /// Forward pass time for this iteration, in milliseconds.
        forward_ms: u64,
        /// Backward pass time for this iteration, in milliseconds.
        backward_ms: u64,
        /// Total number of LP solves in this iteration (forward + backward).
        lp_solves: u64,
        /// Total solver time in this iteration, in milliseconds.
        solve_time_ms: f64,
    },

    // ── Lifecycle events (4) ────────────────────────────────────────

    /// Emitted once when the training loop begins.
    TrainingStarted {
        /// Case study name from the input data directory.
        case_name: String,
        /// Total number of stages in the horizon.
        stages: u32,
        /// Number of hydro plants in the system.
        hydros: u32,
        /// Number of thermal plants in the system.
        thermals: u32,
        /// Number of MPI ranks participating in training.
        ranks: u32,
        /// Number of threads per rank (rayon thread pool size per [Hybrid Parallelism §2](../hpc/hybrid-parallelism.md)).
        threads_per_rank: u32,
        /// Wall-clock time at training start (run-level metadata, not a per-event timestamp).
        timestamp: String,
    },

    /// Emitted once when the training loop exits (converged or limit reached).
    TrainingFinished {
        /// Termination reason: which stopping rule(s) triggered, or "iteration_limit", "time_limit".
        reason: String,
        /// Total number of iterations completed.
        iterations: u64,
        /// Final lower bound at termination.
        final_lb: f64,
        /// Final upper bound at termination.
        final_ub: f64,
        /// Total wall-clock time for the training run, in milliseconds.
        total_time_ms: u64,
        /// Total number of cuts in the FCF at termination.
        total_cuts: u64,
    },

    /// Emitted periodically during policy simulation (not during training).
    SimulationProgress {
        /// Number of simulation scenarios completed so far.
        scenarios_complete: u32,
        /// Total number of simulation scenarios to run.
        scenarios_total: u32,
        /// Wall-clock time since simulation started, in milliseconds.
        elapsed_ms: u64,
        /// Cost for this scenario.
        scenario_cost: f64,
        /// Solver time for this scenario, in milliseconds.
        solve_time_ms: f64,
        /// Number of LP solves for this scenario.
        lp_solves: u64,
    },

    /// Emitted once when policy simulation completes.
    SimulationFinished {
        /// Total number of simulation scenarios evaluated.
        scenarios: u32,
        /// Directory where simulation output files were written.
        output_dir: String,
        /// Total wall-clock time for the simulation run, in milliseconds.
        elapsed_ms: u64,
    },
}
```

#### Cross-references

- **SS2.1a** (above): Event emission points and payload summary table that this subsection formalizes.
- **[Convergence Monitoring SS4.1](./convergence-monitoring.md)**: JSON-lines streaming schema consumed by the text logger and JSON-lines writer. The `IterationSummary` and `ConvergenceUpdate` events are the primary data sources for each JSON-lines record.
- **[Structured Output](../interfaces/structured-output.md)**: Streaming protocol for external consumers (MCP server, programmatic callers). `TrainingEvent` variants map to the structured output event types.
- **[Convergence Monitoring SS2](./convergence-monitoring.md)**: Stopping rule definitions referenced by `StoppingRuleResult.rule_name`.

### 2.2 Termination Conditions

The loop terminates based on the configured `stopping_mode` (`"any"` or `"all"`) applied to the following conditions:

| Condition         | Description                                                         | Configuration              |
| ----------------- | ------------------------------------------------------------------- | -------------------------- |
| Bound stalling    | LB relative improvement over window below tolerance                 | `bound_stalling` rule      |
| Simulation-based  | Bound stable AND simulated policy costs stable                      | `simulation` rule          |
| Iteration limit   | Maximum iteration count reached                                     | `iteration_limit` rule     |
| Time limit        | Wall-clock time exceeded                                            | `time_limit` rule          |
| Graceful shutdown | External signal received (checkpoints last **completed** iteration) | OS signal (SIGTERM/SIGINT) |

For the full stopping rule specification, see [Stopping Rules](../math/stopping-rules.md).

## 3. Abstraction Points

The training loop is parameterized by four abstraction points. Each is a behavioral contract — the training loop interacts with each through a defined interface, independent of the specific variant.

### 3.1 Risk Measure

Given a set of backward outcomes (one per opening) with probabilities, the risk measure aggregates them into a single cut. The two variants are:

- **Expectation** (risk-neutral) — Probability-weighted average of outcomes. The cut intercept and gradient are the weighted means of the per-outcome intercepts and gradients.
- **CVaR** (risk-averse) — Convex combination of expectation and conditional value-at-risk: $(1 - \lambda) \cdot \mathbb{E}[\cdot] + \lambda \cdot \text{CVaR}_\alpha[\cdot]$. Cut coefficients are computed via sorting-based greedy weight allocation. See [Risk Measures](../math/risk-measures.md).

The risk measure can vary by stage (configured per stage in `stages.json`).

### 3.2 Cut Formulation

Determines the structure of cuts added to the FCF:

- **Single-cut** (current) — One aggregated cut per iteration per stage. The future cost variable $\theta$ receives a single constraint per backward pass evaluation.
- **Multi-cut** (deferred) — One cut per opening per iteration. See [Deferred Features SSC.3](../deferred.md).

### 3.3 Horizon Mode

Determines stage traversal and terminal conditions:

- **Finite horizon** — Linear chain $1 \to 2 \to \cdots \to T$. Terminal value $V_{T+1} = 0$.
- **Cyclic** — Stage $T$ transitions back to a cycle start stage. Requires discount factor $d < 1$ for convergence. Cuts at equivalent cycle positions are shared.

See [SDDP Algorithm SS4](../math/sddp-algorithm.md) and [Infinite Horizon](../math/infinite-horizon.md).

### 3.4 Sampling Scheme

Determines how the forward pass selects scenario realizations. This is one of three orthogonal SDDP concerns formalized in [Scenario Generation SS3](./scenario-generation.md):

| Scheme       | Forward Noise Source                       | Description                                                    |
| ------------ | ------------------------------------------ | -------------------------------------------------------------- |
| `InSample`   | Fixed opening tree                         | Sample random index from pre-generated noise vectors (default) |
| `External`   | User-provided `external_scenarios.parquet` | Draw from external data (random or sequential selection)       |
| `Historical` | `inflow_history.parquet` mapped to stages  | Replay historical inflow sequences in order                    |

The backward pass noise source is **always** the fixed opening tree, regardless of the forward sampling scheme. This separation means the forward and backward passes may use different noise distributions — see [Scenario Generation SS3.1](./scenario-generation.md).

## 4. Forward Pass

### 4.1 Overview

The forward pass simulates $M$ independent scenario trajectories through the full stage horizon, solving the stage LP at each step with the current FCF approximation. The purpose is twofold:

1. **Generate trial points** — The visited states $\{\hat{x}_t\}$ at each stage become the evaluation points for the backward pass
2. **Estimate upper bound** — The mean total forward cost across all trajectories provides a statistical upper bound estimate

### 4.2 Scenario Trajectory

For each forward trajectory:

1. **Initialize** — Start from the known initial state $x_0$: initial storage volumes from [Input Constraints SS1](../data-model/input-constraints.md) and inflow lag values from historical data or pre-study stages
2. **Stage loop** ($t = 1, \ldots, T$):
   a. **Select scenario realization** — The sampling scheme selects the noise vector for this stage:
   - _InSample_: Sample random index $j$ from the opening tree, retrieve noise vector $\eta_{t,j}$
   - _External_: Select scenario from external data (by random sampling or sequential iteration). The external inflow values are **inverted to noise terms** via the PAR model (see [Scenario Generation SS3.2](./scenario-generation.md))
   - _Historical_: Look up historical inflow for this stage. The historical values are similarly inverted to noise terms
     b. **Compute inflows and fix noise** — The PAR model evaluates with the selected noise to produce inflow values. The noise terms $\varepsilon_{h,t}$ (whether sampled, inverted from external data, or inverted from historical data) are fixed into the LP via fixing constraints on the AR dynamics equation — the LP always receives noise, never raw inflow values directly (see [Scenario Generation SS3.2](./scenario-generation.md))
     c. **Build stage LP** — Construct the stage LP with incoming state $\hat{x}_{t-1}$, scenario realization, and all current FCF cuts as constraints on $\theta$
     d. **Solve** — Solve the LP. Feasibility is guaranteed by the recourse slack system (see [Penalty System](../data-model/penalty-system.md))
     e. **Record** — Populate a `TrajectoryRecord` (see SS4.2b) with the primal solution, dual solution, stage cost, and end-of-stage state
     f. **Transition** — Pass $\hat{x}_t$ as the incoming state to stage $t+1$
3. **Aggregate** — Compute total trajectory cost $\sum_{t=1}^{T} c_t$

### 4.2a Forward Pass Patch Sequence

Step c above ("Build stage LP") decomposes into the LP rebuild sequence ([Solver Abstraction SS11.2](./solver-abstraction.md)): load template, add active cuts, patch scenario-dependent RHS values, and warm-start. This subsection specifies the **exact `set_row_bounds` calls** for the forward pass — the patches that transform a generic stage template into the LP for a specific (incoming state, scenario realization) pair. All patches are applied in a single call using three parallel arrays: an `indices` array of row indices, a `lower` array of new lower bounds, and an `upper` array of new upper bounds (SoA parameter style per [Solver Interface Trait SS2.3](./solver-interface-trait.md)).

Three categories of patches are applied, all targeting constraint RHS values:

**Category 1 — Incoming state (storage fixing RHS)**

For each operating hydro $h \in [0, N)$, fix the incoming storage in the storage fixing constraint:

```
patch(row = h, value = state[h])
```

This sets $\hat{v}_{h}$ (the incoming storage from the previous stage) as the RHS of the storage fixing constraint at row $h$ ([Solver Abstraction SS2.2](./solver-abstraction.md)). The fixing constraint binds the incoming storage LP variable $v^{in}_h$ to this value; $v^{in}_h$ then propagates through the water balance, FPHA, and generic constraints as an LP variable (see [LP Formulation §4a](../math/lp-formulation.md)).

**Category 2 — Incoming state (AR lag fixing RHS)**

For each operating hydro $h \in [0, N)$ and each lag $\ell \in [0, L)$, fix the inflow lag value:

```
patch(row = N + ℓ·N + h, value = state[N + ℓ·N + h])
```

This sets $\hat{a}_{h,\ell}$ (the inflow lag from the incoming state) as the RHS of the lag fixing constraint at row $N + \ell \cdot N + h$ ([Solver Abstraction SS2.2](./solver-abstraction.md)). The row index formula mirrors the column index formula for the lag state variable — this symmetry is by design (see [Solver Abstraction SS2.2](./solver-abstraction.md)).

**Category 3 — Noise innovation (AR dynamics RHS)**

For each operating hydro $h \in [0, N)$, fix the stochastic innovation term $\varepsilon_{h,t}$ in the AR dynamics equation:

```
patch(row = ar_dynamics_row(h), value = εₕ)
```

where `ar_dynamics_row(h)` is the row index of hydro $h$'s inflow AR dynamics constraint in the middle region of the row layout ([Solver Abstraction SS2.2](./solver-abstraction.md)). The noise value $\varepsilon_{h,t}$ comes from the sampling scheme's realization (step a) — whether sampled from the opening tree, inverted from external data, or inverted from historical data.

**Patch count formula:**

$$n_{patches} = \underbrace{N}_{\text{storage fixing}} + \underbrace{N \cdot L}_{\text{lag fixing}} + \underbrace{N}_{\text{noise fixing}} = N \cdot (2 + L)$$

At production scale ($N = 160$, $L = 12$): $n_{patches} = 160 \times (2 + 12) = 2{,}240$.

**Worked example (3-hydro AR(2) system):**

Using the system from [Solver Abstraction SS2.4](./solver-abstraction.md) ($N = 3$, $L = 2$):

| Patch # | Category       | Row Formula             |  Row | Value                          |
| ------: | -------------- | ----------------------- | ---: | ------------------------------ |
|       0 | Storage fixing | $h = 0$                 |    0 | $\hat{v}_0$ (storage H0)       |
|       1 | Storage fixing | $h = 1$                 |    1 | $\hat{v}_1$ (storage H1)       |
|       2 | Storage fixing | $h = 2$                 |    2 | $\hat{v}_2$ (storage H2)       |
|       3 | AR lag fixing  | $N + 0 \cdot N + 0 = 3$ |    3 | $\hat{a}_{0,0}$ (H0 lag 0)     |
|       4 | AR lag fixing  | $N + 0 \cdot N + 1 = 4$ |    4 | $\hat{a}_{1,0}$ (H1 lag 0)     |
|       5 | AR lag fixing  | $N + 0 \cdot N + 2 = 5$ |    5 | $\hat{a}_{2,0}$ (H2 lag 0)     |
|       6 | AR lag fixing  | $N + 1 \cdot N + 0 = 6$ |    6 | $\hat{a}_{0,1}$ (H0 lag 1)     |
|       7 | AR lag fixing  | $N + 1 \cdot N + 1 = 7$ |    7 | $\hat{a}_{1,1}$ (H1 lag 1)     |
|       8 | AR lag fixing  | $N + 1 \cdot N + 2 = 8$ |    8 | $\hat{a}_{2,1}$ (H2 lag 1)     |
|       9 | Noise fixing   | `ar_dynamics_row(0)`    | (\*) | $\varepsilon_{0,t}$ (H0 noise) |
|      10 | Noise fixing   | `ar_dynamics_row(1)`    | (\*) | $\varepsilon_{1,t}$ (H1 noise) |
|      11 | Noise fixing   | `ar_dynamics_row(2)`    | (\*) | $\varepsilon_{2,t}$ (H2 noise) |

(\*) AR dynamics rows are in the static non-dual region ([Solver Abstraction SS2.2](./solver-abstraction.md)). The exact row indices depend on the system's bus and block counts.

Total: $n_{patches} = 3 \times (2 + 2) = 12$ patches, matching the formula.

**Backward pass similarity:** The backward pass applies the same three patch categories with different values: the incoming state is the trial point $\hat{x}_{t-1}$ from the forward pass, and the noise innovations are drawn from the fixed opening tree rather than the sampling scheme. The patch count formula and row indices are identical.

### 4.2b TrajectoryRecord Type

The `TrajectoryRecord` struct captures the complete LP solution for one scenario at one stage. It is the unit of data produced by step (e) of the forward pass (SS4.2) and consumed by both the backward pass and the simulation output writer.

```rust
/// Complete LP solution for one scenario trajectory, stored per stage.
///
/// This struct is a superset shared between the training forward pass and
/// simulation. Training consumes only `state` and `stage_cost` for the
/// backward pass; simulation uses all fields for output writing.
///
/// Memory layout: records for a full trial are stored contiguously in a flat
/// buffer indexed by `scenario * n_stages + stage`, giving cache-friendly
/// strided access during the backward pass sweep (stride = n_stages).
struct TrajectoryRecord {
    /// Full primal solution vector for this stage's LP.
    /// Length equals the stage's column count from `StageTemplate`.
    primal: Vec<f64>,
    /// Full dual solution vector for this stage's LP (constraint duals).
    /// Length equals the stage's row count (static + dynamic constraint rows).
    dual: Vec<f64>,
    /// LP objective value at this stage (stage cost contribution).
    stage_cost: f64,
    /// End-of-stage state vector (storage levels + AR inflow lags).
    /// Length equals `state_dimension` from SS5.1.
    state: Vec<f64>,
}
```

**Field descriptions:**

| Field        | Type       | Description                                                                                                                                                                                                      |
| ------------ | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `primal`     | `Vec<f64>` | Full primal solution vector for the stage LP. Length equals the stage's column count from `StageTemplate` ([Solver Abstraction SS2.1](./solver-abstraction.md)). Includes state variables, controls, and slacks. |
| `dual`       | `Vec<f64>` | Full dual solution vector (constraint shadow prices). Length equals the stage's row count, including both static constraints and active FCF dynamic constraint rows.                                             |
| `stage_cost` | `f64`      | LP objective value at this stage, representing the immediate stage cost contribution (excluding the future cost variable $\theta$).                                                                              |
| `state`      | `Vec<f64>` | End-of-stage state vector: storage volumes followed by AR inflow lags, in the LP column prefix layout from SS5.1. Length equals $n_{state} = N \cdot (1 + L)$.                                                   |

**Memory layout.** During a training trial, one `TrajectoryRecord` is created per (scenario, stage) pair. The full trial's records are stored in a flat `Vec<TrajectoryRecord>` of length $M \times T$ (where $M$ is the number of forward scenarios on this rank and $T$ is the number of stages), indexed as `records[i * n_stages + t]` for scenario $i$ at stage $t$. The `[scenario][stage]` indexing order gives stride-$T$ access during the backward pass, which iterates stage-by-stage across all scenarios simultaneously: for a given stage $t$, the backward pass reads `records[0 * T + t], records[1 * T + t], \ldots, records[(M-1) * T + t]`, which are spaced $T$ elements apart. This stride is small enough (each `TrajectoryRecord` is on the order of kilobytes) that hardware prefetchers handle the access pattern efficiently.

**Dual-use design note.** The `TrajectoryRecord` struct serves as both the training forward pass record and the simulation output record. Training uses `state` (for patching the next stage's incoming state and for cut gradient computation in the backward pass) and `stage_cost` (for the upper bound estimate and future cost function update). Simulation additionally reads `primal` and `dual` for per-entity result extraction and Parquet output writing ([Output Schemas SS5](../data-model/output-schemas.md)). No separate simulation record type is needed — the simulation forward pass (see [Simulation Architecture SS3.2](./simulation-architecture.md)) populates the same `TrajectoryRecord` and streams it to the output writer. The `primal` and `dual` fields are allocated but unused during training; this is an acceptable memory trade-off because the training forward pass processes only $M$ scenarios per iteration (typically 1--20), so the overhead is bounded by $M \times T \times (n_{cols} + n_{rows}) \times 8$ bytes, which is negligible relative to the cut pool and LP workspace memory.

### 4.3 Parallel Distribution

Scenarios are distributed across MPI ranks in contiguous blocks. Within each rank, scenarios are parallelized across rayon threads with **thread-trajectory affinity**: each thread owns one or more complete trajectories and solves all stages sequentially for its assigned trajectories. This preserves cache locality — the solver basis, scenario data, and LP coefficients remain warm in the thread's cache lines across stages.

The training loop is generic over `C: Communicator` (see [Communicator Trait SS3](../hpc/communicator-trait.md) for the function signature pattern), enabling compile-time specialization to any communication backend.

When $M > N_{\text{threads}}$, threads process multiple trajectories in batches. Between batches, the thread saves and restores forward pass state (solver basis, visited states, scenario realization) at stage boundaries. This is analogous to context switching, but only occurs at well-defined stage boundaries.

After all ranks complete their trajectories, a single `allreduce` with `ReduceOp::Sum` aggregates upper bound statistics:

- **Upper bound statistics** — Sum, sum-of-squares, and trajectory count for computing the mean and variance of total forward costs across all trajectories

The lower bound is evaluated separately after the backward pass — see [Convergence Monitoring SS3.2](./convergence-monitoring.md).

### 4.3a Single-Rank Forward Pass Variant

When `comm.size() == 1` (single-process mode, used by `cobre-python` and `cobre-mcp`, or single-rank MPI execution), all scenarios are assigned to the single rank. The `allreduce` for bound aggregation becomes a local computation -- the rank's local statistics are the global statistics. For the `LocalBackend`, this is an identity copy operation (see [Local Backend SS2.2](../hpc/backend-local.md)). No inter-rank communication occurs. Rayon thread-level parallelism remains active: scenarios are distributed across threads within the single rank using the same thread-trajectory affinity pattern (SS4.3). See [Hybrid Parallelism §1](../hpc/hybrid-parallelism.md) for the single-process mode initialization sequence.

### 4.3b Lower Bound Evaluation

The lower bound (LB) is evaluated **after the backward pass** by rank 0 only. It is computed by iterating over all stage-0 openings (noise innovations) in the fixed opening tree, solving the stage-0 LP for each opening with the latest FCF cuts, and aggregating the per-opening objectives via the stage-0 risk measure.

$$
LB = \rho_0\left(\left\{ \min_{x_0} \left\{ c_0^\top x_0 + \theta_1 \;\middle|\; \text{stage-0 constraints for opening } \omega \right\} \right\}_{\omega \in \Omega_0}\right)
$$

where $\rho_0$ is the stage-0 risk measure (Expectation or CVaR) applied with uniform opening probabilities $p(\omega) = 1/N_{\text{openings}}$.

#### Algorithm

1. Rank 0 iterates over all $N_{\text{openings}}$ openings at stage 0
2. For each opening: rebuild the stage-0 LP with all current FCF cuts, patch with initial state $x_0$ and opening noise, solve, record the objective value
3. Apply the risk measure to aggregate the per-opening objectives into a scalar LB value
4. Rank 0 broadcasts the LB to all other ranks via `comm.broadcast()`

#### Correctness

- The LB must be evaluated **after** the backward pass adds new cuts so the FCF has the latest approximation. Evaluating during the forward pass would use stale cuts, producing a weaker bound.
- Only rank 0 needs to solve because the scenario tree openings are identical everywhere and all ranks share the same initial state $x_0$.
- The risk measure must be applied (not just expectation) because stage 0 can have a risk measure different from Expectation (e.g., CVaR).
- No cut is generated — this is purely an evaluation step, not a backward pass step.

#### Single-Rank Mode

In single-rank mode ([SS4.3a](#43a-single-rank-forward-pass-variant)), the broadcast is an identity operation. The single rank is always rank 0 and performs the full LB evaluation.

### 4.4 Warm-Starting

The forward pass LP solution at stage $t$ provides a near-optimal basis for the backward pass solves at the same stage. The solver retains this basis after the forward solve so that the backward pass at stage $t$ can warm-start from it, significantly reducing solve times. See [Solver Workspaces](./solver-workspaces.md).

## 5. State Management

### 5.1 State Vector

The state vector carries all information needed to transition between stages and generate valid cuts. It consists of:

| Component       | Dimension          | Source at Stage $t$                                 |
| --------------- | ------------------ | --------------------------------------------------- |
| Storage volumes | $N_{\text{hydro}}$ | End-of-stage storage from LP solution ($v_{h,T_k}$) |
| AR inflow lags  | $\sum_h P_h$       | Updated lag buffer after inflow computation         |

Future extensions (batteries, GNL pipeline) may add additional state dimensions — see [SDDP Algorithm SS5](../math/sddp-algorithm.md).

#### 5.1.1 Concrete Type Definition

The state vector is a flat `[f64]` array whose layout matches the LP column prefix defined in [Solver Abstraction SS2.1](./solver-abstraction.md):

```rust
/// Flat state vector matching the LP column prefix layout.
/// Position i corresponds to cut coefficient i.
/// Layout: [v_0, v_1, ..., v_{N-1}, a_{0,0}, a_{1,0}, ..., a_{N-1,0}, a_{0,1}, ..., a_{N-1,L-1}]
/// where v_h = storage for hydro h, a_{h,l} = inflow lag l for hydro h.
/// Total dimension: N * (1 + L)
///
/// Aligned to 64 bytes for AVX-512 SIMD dot product operations.
type StateVector = Vec<f64>;  // len = n_state, allocated with 64-byte alignment
```

The state dimension is $n_{state} = N \cdot (1 + L)$, where $N$ is the number of operating hydros and $L$ is the maximum PAR order across all operating hydros — both defined in [Solver Abstraction SS2.1](./solver-abstraction.md).

**Memory alignment**: State vectors are allocated with 64-byte alignment (the AVX-512 register width, accommodating 8 `f64` values per SIMD lane). This alignment constraint also applies to cut coefficient arrays in the cut pool ([Solver Abstraction SS2.5](./solver-abstraction.md)), ensuring that the primary operation on state vectors — the dot product — can use aligned SIMD loads.

#### 5.1.2 Dot Product as Primary Operation

The primary numerical operation on state vectors is the dot product between a state vector $\mathbf{x}$ and a cut coefficient vector $\boldsymbol{\pi}$:

$$\theta_k = \alpha_k + \boldsymbol{\pi}_k^\top \mathbf{x}$$

This operation occurs in two critical contexts:

1. **Forward pass** — Evaluating the current FCF at a visited state to determine the value of $\theta$. For each active cut $k$, compute $\alpha_k + \boldsymbol{\pi}_k^\top \mathbf{x}$ and take the maximum. This determines the lower bound contribution from the future cost approximation.
2. **Backward pass** — Computing the cut intercept after solving a backward LP:

$$\alpha = Q_t(\hat{x}_{t-1}, \omega) - \boldsymbol{\pi}^\top \hat{x}_{t-1}$$

Both contexts involve a dense dot product of length $n_{state}$ between two 64-byte-aligned `f64` arrays. Implementations should use BLAS-like vectorized routines when available (e.g., SIMD-accelerated `ddot`). At production scale ($n_{state} = 2{,}080$), this is a 16.6 KB operation that fits entirely in L1 data cache — see [Solver Abstraction SS2.5](./solver-abstraction.md) for cache locality analysis.

### 5.2 State Extraction from LP Solution

After solving the stage LP, the state vector is extracted from the LP primal solution. Because the state variables occupy the contiguous prefix of the column layout ([Solver Abstraction SS2.1](./solver-abstraction.md)), extraction is a single contiguous memory copy:

```rust
// Extract state from LP solution — single contiguous memcpy
let state: &[f64] = &solution.primal[0..n_state];
```

This copies:

- `state[0..N]` — Storage volumes $v_0, v_1, \ldots, v_{N-1}$ from the LP primal vector
- `state[N..N*(1+L)]` — Inflow lag values $a_{0,0}, a_{1,0}, \ldots, a_{N-1,L-1}$ from the LP primal vector

No index gathering or scattering is required — the LP column layout is designed so that state extraction is a single contiguous slice read.

### 5.3 State Transfer Between Stages

Transferring state from stage $t$ to stage $t+1$ requires patching the next stage's LP with the outgoing state values. This uses the `set_row_bounds` interface ([Solver Interface Trait SS2.3](./solver-interface-trait.md)) with two categories of patches:

**Storage transfer**: For each operating hydro $h \in [0, N)$, patch the storage fixing constraint RHS:

```
patch(row = h, value = state[h])
```

**Inflow lag transfer**: For each hydro $h \in [0, N)$ and lag $\ell \in [0, L)$, patch the lag fixing constraint RHS:

```
patch(row = N + ℓ·N + h, value = state[N + ℓ·N + h])
```

Both use the row index formulas from [Solver Abstraction SS2.2](./solver-abstraction.md). The patch row index for state variable $r$ matches column $r$ — the row-column symmetry is exact for the entire fixing constraint region $[0, n_{state})$.

The state transfer patches are a subset of the full forward pass patch sequence (SS4.2a, categories 1 and 2). The noise innovation patches (category 3) are separate because they depend on the scenario realization, not the incoming state.

### 5.4 State Lifecycle

1. **Initialization** — The initial state $x_0$ is constructed from:
   - Storage: initial reservoir volumes from `initial_conditions.json` (see [Input Constraints SS1](../data-model/input-constraints.md))
   - AR lags: historical inflow values from `inflow_history.parquet` or pre-study stages in `stages.json`, ordered newest-first (lag 1 = most recent)

2. **Update** — At each stage, the state is updated in two steps:
   - **Inflow computation**: The PAR model (or external/historical lookup) produces the stage inflow $a_{h,t}$. The lag buffer is shifted: the oldest lag drops off, all remaining lags shift by one position, and $a_{h,t}$ becomes the new lag-1 value
   - **Storage extraction**: End-of-stage storage volumes are read from the LP solution's state variable values (SS5.2)

3. **Extraction for backward pass** — After the forward pass, the visited states at each stage are collected across all ranks via `allgatherv`. State deduplication (merging duplicate visited states to reduce backward pass LP solves) is a potential optimization deferred to [Deferred Features](../deferred.md).

### 5.4a State Vector Wire Format

Step 3 above collects visited states from all MPI ranks via `allgatherv` ([Communicator Trait SS2.1](../hpc/communicator-trait.md)). This subsection specifies the **exact wire format** for state vector exchange -- the byte-level layout, indexing scheme, and collective operation parameters that all ranks must agree on.

**Serialization: raw `[f64]` reinterpretation.** State vectors are transmitted as raw `f64` arrays reinterpreted as bytes -- **not** serialized via `postcard` or any structured format. This is consistent with the hot-path convention for cut wire format ([Cut Management Implementation SS4.2](./cut-management-impl.md)): data that flows through per-iteration collective operations uses raw reinterpretation for zero-copy semantics and minimal latency. The `postcard` serialization path ([Input Loading Pipeline SS6](./input-loading-pipeline.md)) is reserved for initialization-time broadcast of heterogeneous structures, not for hot-path homogeneous `f64` arrays.

**Granularity: one `allgatherv` per stage $t$.** The state vector exchange issues one `allgatherv` call per stage, not a single call for all stages combined. Each call at stage $t$ gathers the $M$ visited states for that stage from all ranks:

$$\text{allgatherv}_t : \quad \text{rank } r \text{ sends } M_r \text{ state vectors of dimension } n_{state}$$

where $M_r$ is the number of forward trajectories assigned to rank $r$ by the contiguous block assignment ([Work Distribution SS3.1](../hpc/work-distribution.md)), and $n_{state} = N \cdot (1 + L)$ is the state dimension ([Solver Abstraction SS2.1](./solver-abstraction.md)).

> **Rationale for per-stage granularity.** A single `allgatherv` for all stages would require either (a) a stage index tag per state vector (adding overhead) or (b) a fixed stage ordering assumption that prevents future per-stage deduplication or variable-count extensions. Per-stage calls are simpler, naturally composable with per-stage backward pass processing (SS6.2), and allow future per-stage count variation without protocol changes.

**Indexing: scenario-major within each stage.** Within each rank's send buffer for stage $t$, state vectors are packed in scenario order -- the state from scenario $m$ occupies positions $[m \cdot n_{state}, \; (m+1) \cdot n_{state})$. Across ranks, the `allgatherv` receive buffer is populated in rank order (rank 0's states first, then rank 1's, etc.), matching the rank-ordered receive semantics of [Communicator Trait SS2.1](../hpc/communicator-trait.md).

**Send buffer layout (rank $r$, stage $t$):**

```
send_buf[0 .. M_r * n_state]  =  [x_{t}^{(s_0)}, x_{t}^{(s_1)}, ..., x_{t}^{(s_{M_r - 1})}]
```

where $s_0, s_1, \ldots, s_{M_r - 1}$ are the scenario indices assigned to rank $r$, and each $x_t^{(s)}$ is a contiguous `[f64; n_state]` array in the LP column prefix layout ([Solver Abstraction SS2.1](./solver-abstraction.md)).

**Collective operation parameters:**

| Parameter    | Formula                             | Description                                             |
| ------------ | ----------------------------------- | ------------------------------------------------------- |
| `counts[r]`  | $M_r \times n_{state}$              | Number of `f64` elements rank $r$ contributes           |
| `displs[r]`  | $\sum_{j=0}^{r-1} \text{counts}[j]$ | Offset into receive buffer where rank $r$'s data begins |
| `send.len()` | $M_r \times n_{state}$              | This rank's send buffer length (in `f64` elements)      |
| `recv.len()` | $M \times n_{state}$                | Total receive buffer length (in `f64` elements)         |

where $M = \sum_{r=0}^{R-1} M_r$ is the total number of forward trajectories across all ranks.

**Counts and displacements derivation.** The counts array is computed from the contiguous block assignment ([Work Distribution SS3.1](../hpc/work-distribution.md)):

$$\text{counts}[r] = M_r \times n_{state}$$

The displacements are the exclusive prefix sum of counts:

$$\text{displs}[r] = \sum_{j=0}^{r-1} \text{counts}[j]$$

These arrays are computed once at training initialization (since $M$, $R$, and $n_{state}$ are fixed for the entire training run) and reused for every stage's `allgatherv` call.

**Receive buffer indexing.** After the `allgatherv` completes, the state vector for global scenario $m$ at stage $t$ is located at:

```
recv_buf[m * n_state .. (m + 1) * n_state]
```

This flat indexing works because the rank-ordered receive layout and the contiguous block assignment together produce a globally contiguous scenario ordering in the receive buffer.

**No alignment padding in wire format.** The wire format contains no padding bytes between state vectors. The 64-byte alignment requirement for SIMD dot products (SS5.1.1) is a **local** concern: each rank copies received state vectors into locally aligned buffers before use in the backward pass. The wire format prioritizes minimal bandwidth and simple indexing over alignment.

**Production-scale sizing.** At production scale ($M = 192$ trajectories, $n_{state} = 2{,}080 = 160 \times (1 + 12)$):

| Metric                       | Value                                                |
| ---------------------------- | ---------------------------------------------------- |
| Bytes per state vector       | $2{,}080 \times 8 = 16{,}640$ bytes                  |
| Total `f64` values per stage | $192 \times 2{,}080 = 399{,}360$                     |
| Total bytes per stage        | $399{,}360 \times 8 = 3{,}194{,}880 \approx 3.19$ MB |
| Total bytes across 60 stages | $3.19 \times 60 \approx 191$ MB                      |

With $R = 4$ ranks: $M_r = 48$ trajectories per rank, `counts[r]` $= 48 \times 2{,}080 = 99{,}840$ `f64` elements, `send.len()` $= 99{,}840 \times 8 = 798{,}720$ bytes $\approx 0.80$ MB per rank per stage.

### 5.5 StageIndexer

The `StageIndexer` provides a read-only index map for accessing LP primal and dual positions by semantic name. It eliminates magic index numbers from the training loop and centralizes all LP layout arithmetic in one place.

```rust
/// Read-only index map for accessing LP primal/dual positions by semantic name.
/// Built once at initialization from the stage definition.
/// Shared across all threads within an MPI rank (`Send + Sync`).
/// Equal on all ranks (since LPs differ only by noise innovations, not structure).
pub struct StageIndexer {
    /// Column range for outgoing storage volumes: [0, N).
    pub storage: Range<usize>,
    /// Column range for inflow lag variables: [N, N*(1+L)).
    pub inflow_lags: Range<usize>,
    /// Column range for incoming storage variables: [N*(1+L), N*(2+L)).
    pub storage_in: Range<usize>,
    /// Column index of the future cost variable θ: N*(2+L).
    pub theta: usize,
    /// Total state dimension: N*(1+L). Equal to storage.len() + inflow_lags.len().
    pub n_state: usize,
    /// Row range for storage fixing constraints: [0, N).
    /// Dual of row h gives the storage cut coefficient π^v_h directly.
    pub storage_fixing: Range<usize>,
    /// Row range for AR lag fixing constraints: [N, N+N*L).
    /// Dual of row (N + ℓ*N + h) gives the lag cut coefficient π^lag_{h,ℓ} directly.
    pub lag_fixing: Range<usize>,
    /// Number of operating hydros at this stage.
    pub hydro_count: usize,
    /// Maximum PAR order across all operating hydros at this stage.
    pub max_par_order: usize,
}
```

#### 5.5.1 Indexer Properties

- **Built at initialization**: The indexer is constructed once per stage from the `System` struct ([Internal Structures SS1](../data-model/internal-structures.md)) and the stage configuration. The construction cost is negligible — pure arithmetic on system dimensions.
- **Immutable after construction** (`Send + Sync`): The indexer contains only `usize` values and `Range<usize>` values. It is never mutated after construction, making it safe to share across all threads within an MPI rank without synchronization.
- **Equal across all ranks**: All MPI ranks construct the same LP structure for each stage — the LP layout depends only on the system definition and stage configuration, not on the rank's assigned scenarios. Only noise innovation values differ across ranks and scenarios. Therefore, all ranks produce identical indexers.
- **Owned by the stage definition**: The indexer is associated with the stage template ([Solver Interface Trait SS4.4](./solver-interface-trait.md)), not with any solver instance. It outlives individual solver invocations and is shared read-only.
- **Row–column symmetry for cut extraction**: The dual-extraction region contains exactly `n_state` rows (storage fixing + lag fixing), and the state prefix contains `n_state` columns (outgoing storage + lags). Row $r$'s dual is the cut coefficient for state variable at column $r$. This symmetry eliminates all index translation — `cut_coefficients[0..n_state] = dual[0..n_state]`.

#### 5.5.2 Indexer Usage Examples

```rust
// Extract full state vector from LP solution (single contiguous slice)
let state = &solution.primal[indexer.storage.start..indexer.inflow_lags.end];
assert_eq!(state.len(), indexer.n_state);

// Extract cut coefficients directly from dual solution (single contiguous slice)
let cut_coeffs = &solution.dual[indexer.storage_fixing.start..indexer.lag_fixing.end];
assert_eq!(cut_coeffs.len(), indexer.n_state);

// Access a specific hydro's storage value (hydro 3)
let h3_storage = solution.primal[indexer.storage.start + 3];

// Access a specific lag value (hydro 2, lag 1)
// Formula: inflow_lags.start + lag * hydro_count + hydro
let h2_lag1 = solution.primal[indexer.inflow_lags.start + 1 * indexer.hydro_count + 2];

// Patch storage fixing RHS for incoming storage (hydro h)
let fix_row = indexer.storage_fixing.start + h;

// Patch lag fixing RHS for (hydro h, lag ℓ)
let lag_row = indexer.lag_fixing.start + l * indexer.hydro_count + h;

// Access θ variable value
let theta_value = solution.primal[indexer.theta];
```

**Lag indexing verification**: The formula `inflow_lags.start + l * hydro_count + h` produces the correct index given the LP column layout `[..., a_{0,0}, a_{1,0}, ..., a_{N-1,0}, a_{0,1}, ..., a_{N-1,L-1}]`. For hydro $h$ at lag $\ell$: the column index is $N + \ell \cdot N + h$, and since `inflow_lags.start = N` and `hydro_count = N`, the formula gives $N + \ell \cdot N + h$ — matching [Solver Abstraction SS2.1](./solver-abstraction.md).

#### 5.5.3 Worked Example (3-Hydro AR(2) System)

Using the system from [Solver Abstraction SS2.4](./solver-abstraction.md) ($N = 3$, $L = 2$):

```rust
let indexer = StageIndexer {
    storage: 0..3,           // columns 0, 1, 2 (outgoing storage)
    inflow_lags: 3..9,       // columns 3, 4, 5, 6, 7, 8
    storage_in: 9..12,       // columns 9, 10, 11 (incoming storage)
    theta: 12,               // column 12
    n_state: 9,              // 3 * (1 + 2)
    storage_fixing: 0..3,    // rows 0, 1, 2
    lag_fixing: 3..9,        // rows 3, 4, 5, 6, 7, 8
    hydro_count: 3,
    max_par_order: 2,
};

// Extract state: primal[0..9] — a single contiguous slice of 9 f64 values
let state = &solution.primal[indexer.storage.start..indexer.inflow_lags.end];
// state = [v_0, v_1, v_2, a_{0,0}, a_{1,0}, a_{2,0}, a_{0,1}, a_{1,1}, a_{2,1}]

// Extract cut coefficients: dual[0..9] — a single contiguous slice of 9 f64 values
let cut_coeffs = &solution.dual[indexer.storage_fixing.start..indexer.lag_fixing.end];
// cut_coeffs = [π^fix_0, π^fix_1, π^fix_2, π^lag_{0,0}, ..., π^lag_{2,1}]
// Row r's dual IS the cut coefficient for state variable at column r

// H1 storage (hydro 1): primal[0 + 1] = primal[1]
let h1_storage = solution.primal[indexer.storage.start + 1];

// H2 lag 1 (hydro 2, lag 1): primal[3 + 1*3 + 2] = primal[8]
let h2_lag1 = solution.primal[indexer.inflow_lags.start + 1 * indexer.hydro_count + 2];
```

## 6. Backward Pass

### 6.1 Overview

The backward pass improves the FCF by generating new Benders cuts. It walks stages in reverse order from $T$ down to 2. The trial points $\{\hat{x}_t\}$ used here are the visited states from **all** forward scenarios across **all** MPI ranks (gathered via `allgatherv` in SS5.4). At each stage, the cost-to-go from each trial point is evaluated under **all** openings from the fixed opening tree.

### 6.2 Cut Generation per Stage

At each stage $t$, for each trial point $\hat{x}_{t-1}$ collected during the forward pass:

1. **Retrieve openings** — Get all $N_{\text{openings}}$ noise vectors for stage $t$ from the fixed opening tree (see [Scenario Generation SS2.3](./scenario-generation.md)). This is the **Complete** backward sampling scheme — all openings are always evaluated. A deferred `MonteCarlo(n)` variant would sample a subset; see [Deferred Features SSC.14](../deferred.md).

2. **Evaluate each opening** — For each noise vector $\eta_{t,j}$ ($j = 0, \ldots, N_{\text{openings}} - 1$):
   a. Compute realized inflows via the PAR model with the trial state's lag buffer and the opening's noise vector
   b. Build the backward LP at stage $t$: the incoming state $\hat{x}_{t-1}$ is fixed (storage and lag values set as constraints), and the scenario realization uses the computed inflows
   c. Solve the LP and extract:
   - Objective value $Q_t(\hat{x}_{t-1}, \omega_j)$
   - Dual variables of state-linking constraints (water balance for storage, fixing constraints for AR lags)
   - The fixing constraint duals capture all downstream effects (water balance, FPHA hyperplanes, generic constraints) automatically via the LP envelope theorem — no manual dual combination is needed (see [SS7.2](#72-derivation-from-lp-duality) and [Cut Management SS2](../math/cut-management.md))

3. **Aggregate into cut** — The risk measure aggregates the per-opening outcomes into a single cut:
   - Probabilities are uniform: $p(\omega_j) = 1/N_{\text{openings}}$
   - For Expectation: weighted average of intercepts and gradients
   - For CVaR: sorting-based greedy weight allocation (see [Risk Measures](../math/risk-measures.md))

4. **Add cut** — The new cut is added to stage $t-1$'s cut pool in the FCF

### 6.3 Parallel Distribution

Trial states at each stage are distributed across MPI ranks. Within each rank, each thread evaluates its assigned states sequentially, reusing the warm solver basis saved from the forward pass at that stage (SS4.4). The branching scenarios (openings) for each state are evaluated sequentially by the same thread, keeping the solver state hot.

**Contiguous block assignment.** The backward pass distributes trial states using the same **contiguous block assignment** as the forward pass (SS4.3). After the forward pass, visited states from all ranks are gathered via `allgatherv` (SS5.4a), producing a receive buffer ordered by rank. The $M$ total trial points are then assigned to $R$ ranks: the first $M \bmod R$ ranks each receive $\lceil M/R \rceil$ trial points, and the remaining ranks each receive $\lfloor M/R \rfloor$ trial points. Each rank $r$ receives a contiguous subset $[\text{start}_r, \text{start}_r + M_r)$ into the gathered buffer, where $M_r$ and $\text{start}_r$ are computed by the contiguous block formula in [Work Distribution SS3.1](../hpc/work-distribution.md). Because the `allgatherv` receive buffer is populated in rank order and the block assignment uses the same rank ordering, trial points are directly indexable from the receive buffer without any reindexing or redistribution. State deduplication (reducing the trial point set before distribution) is deferred to [Deferred Features](../deferred.md).

**Stage synchronization barrier**: All threads across all ranks must complete cut generation at stage $t$ before any thread proceeds to stage $t-1$. This is because the new cuts at stage $t$ must be available to all ranks before they solve backward LPs at stage $t-1$ (which include stage $t$'s cuts in their FCF approximation).

After processing each stage, `allgatherv` collects all new cuts from all ranks and distributes them, so every rank has the complete set of new cuts.

### 6.3a Single-Rank Backward Pass Variant

When `comm.size() == 1`, the `allgatherv` for cut synchronization becomes an identity operation -- all cuts generated by the single rank are immediately available locally (for the `LocalBackend`, this is a memcpy; see [Local Backend SS2.2](../hpc/backend-local.md)). The per-stage synchronization barrier reduces to a rayon join barrier only (ensuring all threads complete cut generation at stage $t$ before proceeding to stage $t-1$). All trial states are local, so no state broadcasting is needed. The backward pass logic is otherwise identical to the multi-rank case.

### 6.4 LP Rebuild Considerations

Memory constraints prevent keeping all stage LPs with their full cut sets resident simultaneously. The StageLpCache architecture ([Solver Abstraction SS11.4](./solver-abstraction.md)) addresses this by pre-assembling complete LPs per stage in CSC format. Stage transitions use `passModel` to bulk-load the complete LP including active cuts (~8.6 ms at ~44 GB/s NUMA-interleaved bandwidth). The between-iterations StageLpCache update (~5 ms) absorbs new cuts and deactivates old ones off the critical path, performed by the leader rank on the SharedRegion.

Key mechanisms that minimize rebuild cost:

- **StageLpCache** — Complete pre-assembled LP per stage via SharedRegion, eliminating per-thread CSR assembly buffers
- **Basis persistence** — Reuse the forward pass basis as a warm-start for the backward LP
- **Cut preallocation** — 15K cut slots pre-allocated in the CSC structure; new cuts fill existing slots without structural change

See [Solver Abstraction SS11.2–SS11.4](./solver-abstraction.md) and [Solver Workspaces](./solver-workspaces.md).

### 6.4a Cut Selection Step

> **Decision [DEC-016](../overview/decision-log.md#dec-016) (active):** Cut selection uses deferred parallel execution — stages distributed across ranks and threads, with DeactivationSet allgatherv and leader-only SharedRegion write.

#### VisitedStatesArchive Allocation

The `VisitedStatesArchive` is **always** allocated at training start, regardless of which cut selection strategy is active or whether cut selection is enabled at all. Pre-allocation uses `max_iterations * total_forward_passes` as the capacity per stage, so no heap allocation occurs during the training loop. The archive records all forward-pass trial points for two purposes:

1. **Dominated cut selection** — The `Dominated` variant reads `archive.states_for_stage(t)` during the selection phase.
2. **Export and analysis** — The archive is returned in `TrainingResult.visited_archive` at training completion, and the caller may persist it to the policy checkpoint directory as `states/stage_NNN.bin` FlatBuffers files (see [Binary Formats SS3.1](../data-model/binary-formats.md)).

#### State Archival in Backward Pass

States are archived during the backward pass, after each per-stage `allgatherv` exchange produces the gathered state buffer. For each stage $t$ (iterated in reverse order), the call `archive.archive_gathered_states(t, gathered, total_fwd)` appends the gathered states into the stage's flat storage. This happens before cut generation at stage $t$, so the archive grows incrementally as the backward pass sweeps from stage $T-1$ down to stage $0$.

#### Cut Selection Execution

After the backward pass completes and new cuts have been synchronized (step 4 in SS2.1), the training loop conditionally executes the cut selection phase (step 4a). This step only runs when `should_run(iteration)` returns `true` ([Cut Selection Strategy Trait SS2.1](./cut-selection-trait.md)) — i.e., at multiples of `check_frequency`. On non-selection iterations, the loop proceeds directly to convergence update.

**Stage 0 exemption.** Stage 0 is exempt from cut selection. Its cuts are never the "successor" in the backward pass, so their binding activity metadata is never updated by `update_activity`. Deactivating them based on stale metadata would weaken the lower bound approximation. The training loop emits a no-op `StageSelectionRecord` for stage 0 (cuts_deactivated=0) and only processes stages $1 \ldots T-1$.

**Execution sequence:**

1. **Check** — Evaluate `strategy.should_run(iteration)`. If `false`, skip to step 5 (convergence update).
2. **Exempt stage 0** — Record a no-op selection record for stage 0 (active count unchanged).
3. **Parallel select** — Stages $1 \ldots T-1$ are distributed across threads via Rayon `into_par_iter()`. Each thread calls `strategy.select_for_stage(pool, states, iteration, stage_index)` on its assigned stages. The archive provides visited states: `archive.states_for_stage(stage)` for the `Dominated` variant, or `&[]` for `Level1`/`Lml1`.
4. **Sequential apply** — The `DeactivationSet` results are collected and applied sequentially because `pool.deactivate(&indices)` requires `&mut` access to the cut pool.
5. **Emit event** — `CutSelectionComplete` event with total deactivations, per-stage records, and timing breakdown.

**Multi-rank variant.** In the multi-rank case (SS2.2a), stages are partitioned across ranks before the within-rank Rayon parallelism. After all ranks complete, `allgatherv` gathers per-stage `DeactivationSet` payloads. The leader rank applies deactivations to the SharedRegion StageLpCache, followed by `fence()` + barrier. Wire format: [Synchronization §1.4a](../hpc/synchronization.md).

**Single-rank variant.** When `comm.size() == 1`, no `allgatherv` is needed. The sequence simplifies to: parallel `select_for_stage` all stages → sequential apply deactivations → `fence()`.

**Interaction with StageLpCache update (SS6.4).** The StageLpCache update consists of two logically independent writes: new cut insertion and cut deactivation. New cut insertion runs on every iteration (leader writes coefficients and intercepts for cuts generated in the backward pass). Cut deactivation runs only on selection iterations and uses the `DeactivationSet` from the parallel selection phase. Both writes are performed by the leader rank before the `fence()` + barrier.

#### TrainingResult

The `TrainingResult` struct includes a `visited_archive: Option<VisitedStatesArchive>` field that is always `Some` when training completes (or when training terminates early due to error -- the archive is moved out via `take()`). The caller uses this to persist visited states to the policy checkpoint when the `exports.states` configuration flag is set.

## 7. Dual Extraction for Cut Coefficients

### 7.1 Cut Structure

A Benders cut for stage $t-1$ has the form:

$$
\theta_t \geq \alpha + \sum_{h \in \mathcal{H}} \pi^v_h \cdot v_{h,t-1} + \sum_{h \in \mathcal{H}} \sum_{\ell=1}^{P_h} \pi^{lag}_{h,\ell} \cdot a_{h,t-1-\ell}
$$

where:

| Symbol               | Description                                                      |
| -------------------- | ---------------------------------------------------------------- |
| $\alpha$             | Cut intercept (constant term)                                    |
| $\pi^v_h$            | Cut coefficient for hydro $h$'s storage state variable           |
| $\pi^{lag}_{h,\ell}$ | Cut coefficient for hydro $h$'s inflow lag $\ell$ state variable |
| $v_{h,t-1}$          | End-of-stage storage at stage $t-1$ (state variable)             |
| $a_{h,t-1-\ell}$     | Inflow lag $\ell$ at stage $t-1$ (state variable)                |

### 7.2 Derivation from LP Duality

The cut coefficients are derived from the dual variables of the **fixing constraints** — the equality constraints that bind each incoming state variable to its trial value. Both storage and inflow lags use the same pattern:

- **Storage**: The storage fixing constraint $v^{in}_h = \hat{v}_h$ binds the incoming storage LP variable to its trial value. Its dual $\pi^{fix}_h$ is the storage cut coefficient directly: $\pi^v_h = \pi^{fix}_h$. By the LP envelope theorem, this dual automatically captures all downstream effects — water balance, FPHA hyperplanes, and any generic constraints that reference $v^{in}_h$ — without manual combination. See [LP Formulation §4a](../math/lp-formulation.md).
- **AR inflow lags**: The lag fixing constraint $a^{in}_{h,\ell} = \hat{a}_{h,\ell}$ binds each lag variable to its incoming value. Its dual $\pi^{lag}_{h,\ell}$ is the lag cut coefficient directly (the AR autoregressive coefficients $\psi_\ell$ appear in the dynamics constraint on the LP variable, not on the incoming state).

Cut coefficient extraction is a single contiguous slice read from the dual solution: `cut_coefficients[0..n_state] = dual[0..n_state]`, where the first $N$ duals are storage fixing duals and the remaining $N \cdot L$ are lag fixing duals. See [Solver Abstraction SS2.2](./solver-abstraction.md) for the row layout and [Cut Management §2](../math/cut-management.md) for the mathematical derivation.

The intercept $\alpha$ is computed from the LP objective value and the state-dependent terms:

$$
\alpha = Q_t(\hat{x}_{t-1}, \omega) - \sum_h \pi^v_h \cdot \hat{v}_{h,t-1} - \sum_h \sum_\ell \pi^{lag}_{h,\ell} \cdot \hat{a}_{h,t-1-\ell}
$$

### 7.3 Cut Metadata

Each cut carries metadata for cut management:

| Field        | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| Stage        | Which stage's FCF this cut belongs to                        |
| Iteration    | The iteration when this cut was generated                    |
| Active count | Number of times this cut was binding in subsequent LP solves |

The active count is used by cut selection strategies to prune dominated or inactive cuts. See [Cut Management Implementation](./cut-management-impl.md).

## Cross-References

- [SDDP Algorithm](../math/sddp-algorithm.md) — Mathematical definition of the SDDP algorithm that this training loop implements
- [Cut Management (Math)](../math/cut-management.md) — Mathematical foundations for cut coefficients, selection theory, and dominance criteria
- [Cut Management Implementation](./cut-management-impl.md) — FCF structure, cut selection strategies, serialization, cross-rank cut synchronization, parallel selection phase (SS7.1a), StageLpCache update phase (SS7.1b)
- [Cut Selection Strategy Trait](./cut-selection-trait.md) — Cut selection calling convention (SS2.2), parallel work distribution (SS2.2a), conditional execution via `should_run` (SS2.1)
- [Stopping Rules](../math/stopping-rules.md) — Convergence criteria and termination conditions
- [Risk Measures](../math/risk-measures.md) — CVaR mathematical formulation and cut weight computation
- [Work Distribution](../hpc/work-distribution.md) — Detailed communication+rayon parallelism patterns for forward and backward pass distribution
- [Convergence Monitoring](./convergence-monitoring.md) — Convergence criteria, bound computation, and stopping rules applied within this loop
- [Input Loading Pipeline](./input-loading-pipeline.md) — How case data and warm-start policy cuts are loaded before training begins
- [Input Constraints](../data-model/input-constraints.md) — Initial conditions (SS1) that provide the starting state $x_0$
- [Input Scenarios](../data-model/input-scenarios.md) — Scenario source configuration (SS2.1), external scenarios (SS2.5)
- [Scenario Generation](./scenario-generation.md) — Sampling scheme abstraction (SS3), fixed opening tree lifecycle (SS2.3), external scenario integration (SS4)
- [Penalty System](../data-model/penalty-system.md) — Recourse slacks guaranteeing LP feasibility
- [Solver Abstraction](./solver-abstraction.md) — Solver interface and LP construction
- [Solver Workspaces](./solver-workspaces.md) — Solver state management, basis persistence, and warm-starting
- [Synchronization](../hpc/synchronization.md) — Barrier semantics, collective operations via Communicator trait, and stage-boundary synchronization patterns
- [Checkpointing](../hpc/checkpointing.md) — Checkpoint format and graceful shutdown
- [Deferred Features](../deferred.md) — Multi-cut (C.3), alternative forward pass (C.13), Monte Carlo backward sampling (C.14), policy compatibility validation (C.9)
- [Structured Output](../interfaces/structured-output.md) — JSON-lines streaming protocol consuming events from this training loop
- [Terminal UI](../interfaces/terminal-ui.md) — TUI renderer consuming events from this training loop
- [MCP Server](../interfaces/mcp-server.md) — MCP progress notifications consuming events from this training loop
- [Python Bindings](../interfaces/python-bindings.md) — Single-process execution mode for Python library callers
- [Communicator Trait](../hpc/communicator-trait.md) — Communicator trait definition, method contracts, generic parameterization
- [Local Backend](../hpc/backend-local.md) — LocalBackend identity/no-op operations for single-rank execution
