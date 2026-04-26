# Simulation Architecture

## Purpose

This spec defines the simulation phase of the Cobre SDDP solver: how trained policies are evaluated on large scenario sets, how simulation statistics are computed, and how results are streamed to Parquet output files across distributed MPI ranks. It also covers the simulation-only non-convex extensions that refine LP solutions during policy evaluation.

For the simulation output Parquet schemas, see [Output Schemas](../data-model/output-schemas.md). For the output infrastructure (manifests, MPI partitioning, crash recovery), see [Output Infrastructure](../data-model/output-infrastructure.md).

## 1. Simulation Overview

The simulation phase evaluates the trained SDDP policy on a large number of scenarios to assess:

1. **Policy quality** — Expected cost, variance, and risk metrics
2. **Operational behavior** — Storage trajectories, generation mix, deficit frequency
3. **Robustness** — Performance across diverse hydrological conditions

```mermaid
flowchart TB
    IN["<b>Input</b><br/>trained FCF (cuts) + simulation scenarios config"]
    SEL["<b>Scenario Selection</b><br/><i>per-class: InSample · External · Historical</i>"]
    subgraph PE ["Parallel execution — MPI × Rayon (ranks concurrent)"]
        direction TB
        R0["<b>Rank 0</b> · scenarios 0…S/R<br/><br/>per scenario, stage 1…T:<br/>1. realize uncertainties  ·  2. build stage LP + cuts  ·  3. solve LP<br/>4. non-convex refine <i>(optional)</i>  ·  5. stream → writer"]
        R1["<b>Rank 1</b> · scenarios S/R…2S/R<br/><i>same pipeline, different scenarios</i>"]
        RD["<b>Rank R−1</b> · scenarios (R−1)S/R…S<br/><i>…</i>"]
    end
    OUT["<b>Output Aggregation</b><br/>partitioned Parquet per rank → merge manifest<br/><i>cost stats · storage trajectories · deficit frequency</i>"]

    IN --> SEL --> PE --> OUT
```

No inter-rank communication during simulation — scenarios are statically distributed across ranks; within a rank, work is stolen across Rayon threads.

### 1.1 Simulation Configuration

Simulation behavior is configured via the `simulation` section of `config.json`. Key parameters include:

| Parameter       | Config Path                        | Description                                                 | Reference                                                              |
| --------------- | ---------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------- |
| `enabled`       | `simulation.enabled`               | Whether the simulation phase executes after training        | [CLI and Lifecycle SS5.3](./cli-and-lifecycle.md)                      |
| `num_scenarios` | `simulation.num_scenarios`         | Number of scenarios to evaluate                             | [Configuration Reference](../configuration/configuration-reference.md) |
| Sampling scheme | `scenario_source` in `config.json` | How scenarios are selected (InSample, External, Historical) | [Scenario Generation SS3](./scenario-generation.md)                    |

The sampling scheme is defined in `config.json`. See [Configuration Reference SS18.11](../configuration/configuration-reference.md).

For the full `config.json` schema, see [Configuration Reference](../configuration/configuration-reference.md).

### 1.2 Relationship to Training

The simulation phase reuses the same LP construction infrastructure as training. At each stage, the simulation LP includes:

- All constraints from the training LP (water balance, load balance, generation bounds, etc.)
- The trained FCF cuts as lower-bounding constraints on the future cost variable $\theta$
- The same block structure (parallel or chronological) as defined per stage — see [Block Formulations](../math/block-formulations.md)

The key differences from training are:

| Aspect              | Training                         | Simulation                                 |
| ------------------- | -------------------------------- | ------------------------------------------ |
| Direction           | Forward + backward passes        | Forward pass only (no cut generation)      |
| Objective           | Build policy (cuts)              | Evaluate policy quality                    |
| Scenario count      | Moderate (1–20 per iteration)    | Large (hundreds to thousands)              |
| Non-convex features | Excluded (LP must remain convex) | Optionally included (post-LP refinement)   |
| Output              | Convergence log, FCF cuts        | Per-scenario results, aggregate statistics |

## 2. Policy Compatibility Validation

Before simulation begins, the system validates that the current input data is compatible with the trained policy. The FCF cuts encode LP structural information (state variable dimensions, constraint structure, dual coefficients) that becomes invalid if the system configuration changes between training and simulation.

**Validation checks include** (non-exhaustive):

| Property                             | Why It Matters                                                                      |
| ------------------------------------ | ----------------------------------------------------------------------------------- |
| Block mode per stage                 | Cut duals come from different water balance structures (parallel vs. chronological) |
| Block count and durations per stage  | LP column/row dimensions change                                                     |
| Number of hydro plants               | State variable dimension changes                                                    |
| AR orders per hydro                  | State variable dimension changes (inflow lags are state)                            |
| Cascade topology                     | Water balance constraint structure changes                                          |
| Number of buses, lines               | Load balance constraint structure changes                                           |
| Number of thermals, contracts        | LP column count changes                                                             |
| Production model per hydro per stage | FPHA vs. constant affects generation constraint and dual structure                  |

Any mismatch results in a **hard error** — the simulation does not proceed with an incompatible policy.

Warm-start compatibility validation rules are specified in [Input Loading Pipeline SS7.1](./input-loading-pipeline.md).

The full policy compatibility validation specification, including the metadata format persisted by training and the validation algorithm, is documented in [Deferred Features SSC.9](../deferred.md).

## 3. Simulation Execution

### 3.1 Scenario Distribution

Simulation scenarios are distributed across MPI ranks using the same two-level work distribution as training:

1. **MPI rank level — deterministic distribution.** If $S$ total scenarios are distributed across $R$ ranks, the first $S \bmod R$ ranks receive $\lceil S/R \rceil$ scenarios and the remaining ranks receive $\lfloor S/R \rfloor$. Each rank stores only its assigned scenarios in memory.

2. **Thread level — dynamic work-stealing.** Within each rank, scenarios are processed by a thread pool with dynamic work-stealing. Per-scenario costs are not uniform — LP solve time varies with noise realization, active constraints, and warm-start quality — so dynamic scheduling absorbs this variability.

For details on the distribution strategy and NUMA-aware allocation, see [Scenario Generation SS5](./scenario-generation.md).

### 3.2 Per-Scenario Forward Pass

For each assigned scenario, the simulation executes a complete forward pass through all stages:

1. **Initialize state** — Set the initial state (storage volumes, inflow lags) from the case data

2. **For each stage $t = 1, \ldots, T$:**

   a. **Realize uncertainties** — Obtain the stage realization from the configured sampling scheme. For InSample, sample from the opening tree. For External or Historical, use the external/historical values with noise inversion to obtain $\varepsilon$ values for the LP's AR dynamics constraint. See [Scenario Generation SS3.2](./scenario-generation.md).

   b. **Build stage LP** — Construct the LP with the current state, realized uncertainties, and all FCF cuts accumulated during training. The LP includes the block structure for this stage (parallel or chronological, per-stage definition). See [Block Formulations](../math/block-formulations.md).

   c. **Solve LP** — Solve the stage LP. The simulation LP should always be feasible due to recourse slack variables (deficit, excess). If infeasibility occurs, it indicates a system error.

   d. **Apply non-convex refinements** — If non-convex extensions are configured (see SS5), refine the LP solution through post-processing.

   e. **Extract results** — Record stage-level outputs: generation, storage, flows, costs, violations, marginal values. See [Output Schemas SS5](../data-model/output-schemas.md) for column definitions.

   f. **Propagate state** — Extract end-of-stage storage volumes and updated inflow lag buffer as the initial state for stage $t+1$.

3. **Compute scenario cost** — Sum immediate costs across all stages, applying discount factors. See [Discount Rate](../math/discount-rate.md).

4. **Stream results** — Send the scenario results to the output writer for streaming to disk (see SS6).

> **Shared record type.** The simulation forward pass reuses the [TrajectoryRecord](./training-loop.md) type (SS4.2b) from the training loop rather than defining a separate simulation record struct. Training populates only the `state` and `stage_cost` fields for backward pass consumption; simulation additionally reads `primal` and `dual` for per-entity result extraction and Parquet output.

### 3.3 Memory Management

Individual scenario results are streamed to the output writer immediately upon completion (see SS6). This prevents accumulation of all scenario results in memory, which is critical when simulating thousands of scenarios. Each thread releases the scenario's LP workspace and result buffers before proceeding to its next scenario. Per-scenario scalar costs (total and per-category) are retained in a compact buffer for final statistics computation (see SS4.4).

### 3.4 Result Types

This section defines the Rust types that carry simulation results from the SDDP forward pass loop to the output writer. These types are the payload of the bounded channel described in SS6.1 and the return type of the `fn simulate()` entry point.

**Design rationale — nested vs. flat layout.** The result types use a nested design (per-entity-type `Vec`s grouped by stage) rather than a flat columnar layout. The simulation loop produces results stage-by-stage; the output writer is responsible for columnar conversion during Parquet serialization (see [Output Schemas SS5](../data-model/output-schemas.md)). The nested layout matches the production flow and avoids a premature columnar transpose inside the simulation hot path.

**Design rationale — streaming vs. batch return.** The `fn simulate()` function does **not** return `Vec<SimulationScenarioResult>`. Instead, it accepts a `Sender<SimulationScenarioResult>` channel handle (the sending end of the bounded channel from SS6.1). Each scenario result is sent through the channel as soon as it completes, enabling the background I/O thread to begin writing before all scenarios finish. The function returns `Result<SimulationSummary, SimulationError>` containing only the aggregate statistics from SS4.

**Derived columns excluded.** The result structs contain only values produced directly by the LP solve and state propagation. Columns that are computed during output writing are excluded:

| Excluded Column          | Derivation                                       | Entity   |
| ------------------------ | ------------------------------------------------ | -------- |
| `outflow_m3s`            | `turbined_m3s + spillage_m3s`                    | Hydro    |
| `generation_mwh`         | `generation_mw × block_duration_hours`           | Hydro    |
| `net_flow_mw`            | `direct_flow_mw − reverse_flow_mw`               | Exchange |
| `net_flow_mwh`           | `net_flow_mw × block_duration_hours`             | Exchange |
| `losses_mw`              | $(1 - \eta)(f^+ + f^-)$                          | Exchange |
| `losses_mwh`             | `losses_mw × block_duration_hours`               | Exchange |
| `generation_mwh`         | `generation_mw × block_duration_hours`           | Thermal  |
| `load_mwh`               | `load_mw × block_duration_hours`                 | Bus      |
| `deficit_mwh`            | `deficit_mw × block_duration_hours`              | Bus      |
| `excess_mwh`             | `excess_mw × block_duration_hours`               | Bus      |
| `pumped_volume_hm3`      | `pumped_flow_m3s × duration × conversion_factor` | Pumping  |
| `energy_consumption_mwh` | `power_consumption_mw × block_duration_hours`    | Pumping  |
| `energy_mwh`             | `power_mw × block_duration_hours`                | Contract |
| `curtailment_mwh`        | `curtailment_mw × block_duration_hours`          | NCS      |
| `generation_mwh`         | `generation_mw × block_duration_hours`           | NCS      |

Block duration is known to the output writer from `Stage::blocks` ([Internal Structures SS12.2](../data-model/internal-structures.md)). Keeping the conversion in the writer avoids redundant `f64` fields in the hot-path result payload.

#### 3.4.1 Per-Entity Result Sub-Structs

Each entity type has a dedicated result struct holding the LP solution values for one entity at one (stage, block) pair. All structs implement `Send` to support transfer across the bounded channel.

```rust
/// Cost breakdown for one (stage, block) pair.
/// Corresponds to one row in the costs output schema
/// (output-schemas.md SS5.1).
pub struct SimulationCostResult {
    pub stage_id: u32,
    /// Block index within the stage, or None for stage-level aggregates.
    pub block_id: Option<u32>,
    pub total_cost: f64,
    pub immediate_cost: f64,
    pub future_cost: f64,
    pub discount_factor: f64,
    // Resource costs
    pub thermal_cost: f64,
    pub contract_cost: f64,
    // Category 1 — Recourse
    pub deficit_cost: f64,
    pub excess_cost: f64,
    // Category 2 — Violations
    pub storage_violation_cost: f64,
    pub filling_target_cost: f64,
    pub hydro_violation_cost: f64,
    pub turbined_min_violation_cost: f64,
    pub turbined_max_violation_cost: f64,
    pub storage_min_violation_cost: f64,
    pub storage_max_violation_cost: f64,
    pub ramp_up_violation_cost: f64,
    pub ramp_down_violation_cost: f64,
    pub inflow_penalty_cost: f64,
    pub generic_violation_cost: f64,
    // Category 3 — Regularization
    pub spillage_cost: f64,
    pub fpha_turbined_cost: f64,
    pub curtailment_cost: f64,
    pub exchange_cost: f64,
    // Imputed costs
    pub pumping_cost: f64,
}

/// Hydro plant result for one (stage, block, hydro) tuple.
/// Corresponds to one row in the hydros output schema
/// (output-schemas.md SS5.2). Energy columns (generation_mwh)
/// are derived by the output writer.
pub struct SimulationHydroResult {
    pub stage_id: u32,
    pub block_id: Option<u32>,
    pub hydro_id: i32,
    pub turbined_m3s: f64,
    pub spillage_m3s: f64,
    pub evaporation_m3s: Option<f64>,
    pub diverted_inflow_m3s: Option<f64>,
    pub diverted_outflow_m3s: Option<f64>,
    pub incremental_inflow_m3s: f64,
    pub inflow_m3s: f64,
    pub storage_initial_hm3: f64,
    pub storage_final_hm3: f64,
    pub generation_mw: f64,
    pub productivity_mw_per_m3s: Option<f64>,
    pub spillage_cost: f64,
    pub water_value_per_hm3: f64,
    pub storage_binding_code: i8,
    pub operative_state_code: i8,
    // Violation slacks
    pub turbined_slack_m3s: f64,
    pub outflow_slack_below_m3s: f64,
    pub outflow_slack_above_m3s: f64,
    pub generation_slack_mw: f64,
    pub storage_violation_below_hm3: f64,
    pub filling_target_violation_hm3: f64,
    pub evaporation_violation_m3s: f64,
    pub inflow_nonnegativity_slack_m3s: f64,
}

/// Thermal unit result for one (stage, block, thermal) tuple.
/// Corresponds to one row in the thermals output schema
/// (output-schemas.md SS5.3). Energy columns (generation_mwh)
/// are derived by the output writer.
pub struct SimulationThermalResult {
    pub stage_id: u32,
    pub block_id: Option<u32>,
    pub thermal_id: i32,
    pub generation_mw: f64,
    pub generation_cost: f64,
    pub is_gnl: bool,
    pub gnl_committed_mw: Option<f64>,
    pub gnl_decision_mw: Option<f64>,
    pub operative_state_code: i8,
}

/// Exchange (transmission line) result for one (stage, block, line) tuple.
/// Corresponds to one row in the exchanges output schema
/// (output-schemas.md SS5.4). Derived columns (net_flow_mw, losses_mw,
/// and all MWh energy columns) are computed by the output writer.
pub struct SimulationExchangeResult {
    pub stage_id: u32,
    pub block_id: Option<u32>,
    pub line_id: i32,
    pub direct_flow_mw: f64,
    pub reverse_flow_mw: f64,
    pub exchange_cost: f64,
    pub operative_state_code: i8,
}

/// Bus result for one (stage, block, bus) tuple.
/// Corresponds to one row in the buses output schema
/// (output-schemas.md SS5.5). Energy columns (load_mwh, deficit_mwh,
/// excess_mwh) are derived by the output writer.
pub struct SimulationBusResult {
    pub stage_id: u32,
    pub block_id: Option<u32>,
    pub bus_id: i32,
    pub load_mw: f64,
    pub deficit_mw: f64,
    pub excess_mw: f64,
    pub spot_price: f64,
}

/// Pumping station result for one (stage, block, station) tuple.
/// Corresponds to one row in the pumping_stations output schema
/// (output-schemas.md SS5.6). Volume and energy columns are derived
/// by the output writer.
pub struct SimulationPumpingResult {
    pub stage_id: u32,
    pub block_id: Option<u32>,
    pub pumping_station_id: i32,
    pub pumped_flow_m3s: f64,
    pub power_consumption_mw: f64,
    pub pumping_cost: f64,
    pub operative_state_code: i8,
}

/// Contract result for one (stage, block, contract) tuple.
/// Corresponds to one row in the contracts output schema
/// (output-schemas.md SS5.7). Energy column (energy_mwh) is
/// derived by the output writer.
pub struct SimulationContractResult {
    pub stage_id: u32,
    pub block_id: Option<u32>,
    pub contract_id: i32,
    pub power_mw: f64,
    pub price_per_mwh: f64,
    pub total_cost: f64,
    pub operative_state_code: i8,
}

/// Non-controllable source result for one (stage, block, source) tuple.
/// Corresponds to one row in the non_controllables output schema
/// (output-schemas.md SS5.8). Energy columns (generation_mwh,
/// curtailment_mwh) are derived by the output writer.
pub struct SimulationNonControllableResult {
    pub stage_id: u32,
    pub block_id: Option<u32>,
    pub non_controllable_id: i32,
    pub generation_mw: f64,
    pub available_mw: f64,
    pub curtailment_mw: f64,
    pub curtailment_cost: f64,
    pub operative_state_code: i8,
}

/// Inflow lag state for one (stage, hydro, lag_index) tuple.
/// Corresponds to one row in the inflow_lags output schema
/// (output-schemas.md SS5.10). Only populated when the hydro
/// has AR order > 0.
pub struct SimulationInflowLagResult {
    pub stage_id: u32,
    pub hydro_id: i32,
    pub lag_index: u32,
    pub inflow_m3s: f64,
}

/// Generic constraint violation for one (stage, block, constraint) tuple.
/// Corresponds to one row in the violations/generic output schema
/// (output-schemas.md SS5.11).
pub struct SimulationGenericViolationResult {
    pub stage_id: u32,
    pub block_id: Option<u32>,
    pub constraint_id: i32,
    pub slack_value: f64,
    pub slack_cost: f64,
}
```

#### 3.4.2 Per-Stage Result

Groups all entity-type results for a single stage into one struct. The simulation loop produces one `SimulationStageResult` per stage per scenario; the per-entity `Vec`s contain one entry per (block, entity) pair within that stage.

```rust
/// All simulation results for a single stage within one scenario.
/// Contains per-entity-type result vectors, each holding one entry
/// per (block, entity) pair. Optional entity types are empty Vecs
/// when the entity type does not exist in the system.
pub struct SimulationStageResult {
    pub stage_id: u32,
    pub costs: Vec<SimulationCostResult>,
    pub hydros: Vec<SimulationHydroResult>,
    pub thermals: Vec<SimulationThermalResult>,
    pub exchanges: Vec<SimulationExchangeResult>,
    pub buses: Vec<SimulationBusResult>,
    /// Empty if no pumping stations exist in the system.
    pub pumping_stations: Vec<SimulationPumpingResult>,
    /// Empty if no contracts exist in the system.
    pub contracts: Vec<SimulationContractResult>,
    /// Empty if no non-controllable sources exist in the system.
    pub non_controllables: Vec<SimulationNonControllableResult>,
    /// Empty if no hydros have AR order > 0.
    pub inflow_lags: Vec<SimulationInflowLagResult>,
    /// Empty if no generic constraints exist or no violations occurred.
    /// Only non-zero violations may be included.
    pub generic_violations: Vec<SimulationGenericViolationResult>,
}
```

#### 3.4.3 SimulationScenarioResult

The top-level per-scenario payload sent through the bounded channel (SS6.1). Each completed scenario produces exactly one `SimulationScenarioResult` instance.

```rust
/// Complete simulation result for one scenario. This is the payload
/// type of the bounded channel connecting simulation threads to the
/// background I/O thread (SS6.1).
///
/// # Send bound
///
/// `SimulationScenarioResult` implements `Send` because it is
/// transferred across a thread boundary: the simulation thread
/// that produces it sends it through the channel to the dedicated
/// I/O thread. All constituent types are `Send`-safe (plain data,
/// no Rc, no raw pointers, no non-Send interior mutability).
///
/// # Memory lifetime
///
/// Each instance is produced by a simulation thread, sent through
/// the channel, consumed by the I/O thread for Parquet writing,
/// and then dropped. At most `channel_capacity` instances exist
/// simultaneously (bounded by channel backpressure). See SS3.3.
pub struct SimulationScenarioResult {
    /// 0-based scenario identifier. Unique across all MPI ranks.
    /// Determines the Hive partition path:
    /// `{entity}/scenario_id={scenario_id:04d}/data.parquet`.
    pub scenario_id: u32,

    /// Total discounted cost for this scenario, summed across all
    /// stages: $C_s = \sum_{t=1}^{T} d_t \cdot c_t$ where $d_t$
    /// is the cumulative discount factor and $c_t$ the stage
    /// immediate cost.
    pub total_cost: f64,

    /// Per-category cost components for this scenario, summed
    /// across all stages. Used for per-category statistics (SS4.2)
    /// and retained in the compact cost buffer (SS3.3) even after
    /// per-stage detail is streamed to the output writer.
    pub per_category_costs: ScenarioCategoryCosts,

    /// Per-stage detailed results. Present when the output detail
    /// level (SS6.2) is Stage-level or Full. Empty (zero-length Vec)
    /// when detail level is Summary — in that case, only
    /// `scenario_id`, `total_cost`, and `per_category_costs` are
    /// populated.
    pub stages: Vec<SimulationStageResult>,
}

/// Per-category cost totals for one scenario, summed across all stages.
/// Matches the category breakdown in SS4.2.
pub struct ScenarioCategoryCosts {
    /// thermal_cost + contract_cost
    pub resource_cost: f64,
    /// deficit_cost + excess_cost
    pub recourse_cost: f64,
    /// storage_violation_cost + filling_target_cost + hydro_violation_cost
    /// + inflow_penalty_cost + generic_violation_cost
    pub violation_cost: f64,
    /// spillage_cost + fpha_turbined_cost + curtailment_cost + exchange_cost
    pub regularization_cost: f64,
    /// pumping_cost (imputed)
    pub imputed_cost: f64,
}
```

#### 3.4.4 SimulationSummary

Aggregate statistics returned by `fn simulate()` after all scenarios complete and MPI aggregation (SS4.4) finishes. This is the `Ok` variant of the function's return type.

```rust
/// Aggregate simulation statistics computed after all scenarios
/// complete. Produced by MPI aggregation on rank 0 (SS4.4) and
/// returned as the Ok value of fn simulate().
pub struct SimulationSummary {
    // -- SS4.1 Cost Statistics --

    /// Mean total cost across all scenarios:
    /// $\bar{C} = \frac{1}{S} \sum_{s=1}^{S} C_s$
    pub mean_cost: f64,

    /// Sample standard deviation of total cost:
    /// $\sigma_C = \sqrt{\frac{1}{S-1} \sum_{s=1}^{S} (C_s - \bar{C})^2}$
    pub std_cost: f64,

    /// Minimum total cost across all scenarios.
    pub min_cost: f64,

    /// Maximum total cost across all scenarios.
    pub max_cost: f64,

    /// CVaR at the configured confidence level alpha.
    /// Mean of the worst (1 - alpha) fraction of scenario costs.
    /// See [Risk Measures](../math/risk-measures.md).
    pub cvar: f64,

    /// Confidence level used for CVaR computation.
    pub cvar_alpha: f64,

    // -- SS4.2 Per-Category Cost Statistics --

    /// Per-category cost statistics (mean, max, frequency) for each
    /// of the five cost categories defined in SS4.2.
    pub category_stats: Vec<CategoryCostStats>,

    // -- SS4.3 Operational Statistics --

    /// Fraction of scenarios with at least one stage having deficit > 0.
    pub deficit_frequency: f64,

    /// Total deficit energy (MWh) summed across all scenarios and stages.
    pub total_deficit_mwh: f64,

    /// Total spillage energy (MWh) summed across all scenarios and stages.
    pub total_spillage_mwh: f64,

    /// Number of scenarios simulated (across all ranks).
    pub n_scenarios: u32,

    /// Optional per-stage aggregate statistics. Present when the output
    /// detail level is Stage-level or Full.
    pub stage_stats: Option<Vec<StageSummaryStats>>,
}

/// Per-category cost statistics for one cost category.
pub struct CategoryCostStats {
    /// Category name (matches SS4.2 table: "resource", "recourse",
    /// "violation", "regularization", "imputed").
    pub category: String,
    /// Mean cost across all scenarios.
    pub mean: f64,
    /// Maximum cost across all scenarios.
    pub max: f64,
    /// Fraction of scenarios where category cost > 0.
    pub frequency: f64,
}

/// Per-stage aggregate statistics (optional, when detail >= stage-level).
pub struct StageSummaryStats {
    pub stage_id: u32,
    /// Mean total cost at this stage across all scenarios.
    pub mean_cost: f64,
    /// Mean total storage (hm3) across all hydros and scenarios.
    pub mean_storage_hm3: f64,
    /// Mean total generation (MW) across all sources and scenarios.
    pub mean_generation_mw: f64,
}
```

#### 3.4.5 SimulationError

Error conditions that can cause `fn simulate()` to fail. Variants cover LP-level failures that should not occur in a well-formed simulation (recourse slacks should prevent infeasibility) and I/O failures from the output writer.

```rust
/// Errors that can occur during simulation execution.
pub enum SimulationError {
    /// LP infeasibility at a simulation stage. This indicates a system
    /// error — recourse slack variables (deficit, excess) should always
    /// make the LP feasible. The error includes the scenario, stage,
    /// and solver diagnostic to aid debugging.
    LpInfeasible {
        scenario_id: u32,
        stage_id: u32,
        solver_message: String,
    },

    /// LP solver returned an unexpected status (e.g., numerical
    /// difficulties, unbounded). Includes solver-specific diagnostics.
    SolverError {
        scenario_id: u32,
        stage_id: u32,
        solver_message: String,
    },

    /// I/O failure during output writing (disk full, permission denied,
    /// Parquet encoding error). The simulation cannot continue if the
    /// output writer fails because results would be lost.
    IoError {
        message: String,
    },

    /// Policy compatibility validation failed (SS2). The trained policy
    /// is incompatible with the current system configuration.
    PolicyIncompatible {
        message: String,
    },

    /// Channel send failure — the receiving end (I/O thread) has
    /// dropped unexpectedly. Indicates a panic or crash in the
    /// output writer.
    ChannelClosed,
}
```

#### 3.4.6 Function Signature: `fn simulate()`

The `fn simulate()` entry point executes the simulation phase. It is generic over `S: SolverInterface` and `C: Communicator`, following the same compile-time monomorphization pattern as `fn train()` ([Solver Interface Trait SS5](./solver-interface-trait.md), [Communicator Trait §3](../hpc/communicator-trait.md)).

```rust
/// Execute the SDDP simulation phase: evaluate a trained policy on
/// a set of scenarios, stream per-scenario results through a bounded
/// channel to the output writer, and return aggregate statistics.
///
/// # Parameters
///
/// - `system`: Shared reference to the fully loaded and validated
///   system representation ([Internal Structures SS1](../data-model/internal-structures.md)).
///   Immutable and shared across all threads within this rank.
///
/// - `policy`: The trained FCF (Future Cost Function) policy to
///   evaluate. Contains the optimality cuts accumulated during
///   training, loaded from the policy directory
///   ([Binary Formats SS3.2](../data-model/binary-formats.md)).
///
/// - `config`: Simulation configuration parameters (number of
///   scenarios, output detail level, sampling scheme, etc.).
///   See SS1.1 for the configuration surface.
///
/// - `solver_factory`: Factory closure that creates one
///   `SolverInterface` instance per thread. Each simulation thread
///   owns its solver for the duration of the simulation.
///   See [Solver Interface Trait SS5](./solver-interface-trait.md).
///
/// - `comm`: Communicator backend for MPI collective operations
///   (scenario distribution SS3.1, statistics aggregation SS4.4).
///   See [Communicator Trait §3](../hpc/communicator-trait.md).
///
/// - `result_tx`: Sending end of the bounded channel connecting
///   simulation threads to the background I/O thread (SS6.1).
///   Each completed scenario is sent as a `SimulationScenarioResult`
///   through this channel. Channel backpressure throttles simulation
///   when the I/O thread falls behind.
///
/// # Returns
///
/// - `Ok(SimulationSummary)`: Aggregate statistics (SS4.1–4.4) after
///   all scenarios complete and MPI aggregation finishes. On
///   non-rank-0 processes, the summary contains only locally
///   computed statistics (min/max from allreduce); the full
///   statistics (mean, std, CVaR, per-category) are authoritative
///   only on rank 0 (SS4.4).
///
/// - `Err(SimulationError)`: A fatal error prevented simulation
///   from completing. Partial results already sent through
///   `result_tx` remain valid and may have been written to disk.
///
/// # Thread safety
///
/// This function spawns a thread pool internally for scenario-level
/// parallelism (SS3.1). The `system` and `policy` references are
/// shared read-only across threads. Each thread creates its own
/// solver via `solver_factory`. The `result_tx` sender is cloned
/// per thread (standard mpsc/crossbeam channel semantics).
pub fn simulate<S: SolverInterface, C: Communicator>(
    system: &System,
    policy: &Policy,
    config: &SimulationConfig,
    solver_factory: impl Fn() -> S + Send + Sync,
    comm: &C,
    result_tx: Sender<SimulationScenarioResult>,
) -> Result<SimulationSummary, SimulationError> {
    // 1. Validate policy compatibility (SS2)
    // 2. Compute local scenario assignment (SS3.1)
    // 3. Spawn thread pool, distribute scenarios with work-stealing
    // 4. Per scenario: forward pass (SS3.2), send result through channel
    // 5. After all local scenarios complete, participate in MPI
    //    aggregation (SS4.4)
    // 6. Return SimulationSummary
    todo!()
}
```

**Consistency with `internal-structures.md`**: The `fn simulate()` signature above refines the placeholder in [Internal Structures SS1.7](../data-model/internal-structures.md) (`cobre_sddp::simulate(system: &System, policy: &Policy, config: &SimulationConfig, comm: &C) -> Result<SimulationResult, SimError>`) by adding the `solver_factory` parameter (matching the `fn train()` pattern), the `result_tx` channel handle, and renaming the return types to their concrete definitions (`SimulationSummary`, `SimulationError`).

## 4. Simulation Statistics

The simulation phase computes aggregate statistics across all scenarios. Each rank stores per-scenario total costs and per-category cost components in a local buffer during execution. After all ranks complete, costs are gathered to rank 0 for final statistics computation (see SS4.4).

### 4.1 Cost Statistics

| Statistic                   | Description                                                                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Mean cost                   | $\bar{C} = \frac{1}{S} \sum_{s=1}^{S} C_s$                                                                                           |
| Standard deviation          | $\sigma_C = \sqrt{\frac{1}{S-1} \sum_{s=1}^{S} (C_s - \bar{C})^2}$                                                                   |
| Minimum / Maximum cost      | Range of total scenario costs                                                                                                        |
| CVaR at confidence $\alpha$ | Conditional Value-at-Risk: mean of the worst $(1-\alpha)$ fraction of scenario costs. See [Risk Measures](../math/risk-measures.md). |

CVaR computation requires all individual scenario costs (for sorting and tail averaging), which are gathered to rank 0 via `MPI_Gatherv` after all ranks complete (see SS4.3). Since the gathered array is already available, the mean and standard deviation are also computed directly from it — avoiding the numerical instability of running sum-of-squares formulas. Min and max are computed via `MPI_Allreduce` without gathering.

### 4.2 Per-Category Cost Statistics

In addition to aggregate cost statistics, the simulation reports mean cost broken down by the cost categories defined in [Penalty System SS2](../data-model/penalty-system.md). This allows the user to understand the composition of total cost and diagnose policy behavior.

| Category                           | Components Summed                                                                                                            |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Resource costs                     | `thermal_cost` + `contract_cost`                                                                                             |
| Category 1 — Recourse              | `deficit_cost` + `excess_cost`                                                                                               |
| Category 2 — Constraint violations | `storage_violation_cost` + `filling_target_cost` + `hydro_violation_cost` + `inflow_penalty_cost` + `generic_violation_cost` |
| Category 3 — Regularization        | `spillage_cost` + `fpha_turbined_cost` + `curtailment_cost` + `exchange_cost`                                                |
| Imputed costs                      | `pumping_cost`                                                                                                               |

For each category, the simulation computes:

- **Mean** across all scenarios
- **Maximum** across all scenarios (identifies worst-case contributors)
- **Frequency** — fraction of scenarios where the category cost is non-zero (particularly relevant for deficit and constraint violations)

These statistics are computed from the gathered cost arrays on rank 0 alongside the aggregate statistics in SS4.1. The per-scenario, per-stage cost breakdown is always available in the detailed output files — see [Output Schemas SS5.1](../data-model/output-schemas.md) for the full column definitions.

### 4.3 Operational Statistics

| Statistic             | Description                                                                                  |
| --------------------- | -------------------------------------------------------------------------------------------- |
| Deficit frequency     | Fraction of scenarios with at least one stage having deficit > 0                             |
| Total deficit energy  | Sum of deficit (MWh) across all scenarios and stages                                         |
| Total spillage energy | Sum of spillage (MWh) across all scenarios and stages                                        |
| Stage-level stats     | Optional per-stage aggregates (mean storage, mean generation, mean cost) when detail ≥ stage |

### 4.4 MPI Aggregation

Each rank stores its local scenario costs in a per-rank buffer. After all ranks complete, results are aggregated:

- **Min/Max** (`min_cost`, `max_cost`) — `MPI_Allreduce` with `MPI_MIN` / `MPI_MAX`
- **Scenario costs** — Individual scenario total costs and per-category cost components gathered to rank 0 via `MPI_Gatherv`
- **Mean, std, CVaR, per-category stats** — Computed on rank 0 from the gathered arrays
- **Deficit/spillage sums** (`deficit_mwh_sum`, `spill_mwh_sum`, `deficit_scenarios`) — `MPI_Allreduce` with `MPI_SUM`

## 5. Non-Convex Extensions (Deferred)

SDDP produces an optimal policy for the convex relaxation. During simulation, the LP solution can optionally be refined with non-convex operational constraints that are excluded from training (because they would break the LP convexity required for valid cut generation).

All non-convex extensions are **deferred**. This section documents _what_ they are and _why_ they are simulation-only, not _how_ they will be implemented.

| Feature                 | Why Simulation-Only                                             | Status   | Reference                                 |
| ----------------------- | --------------------------------------------------------------- | -------- | ----------------------------------------- |
| Linearized head model   | Bilinear dependency (head × flow) changes LP between iterations | Deferred | [Deferred Features SSC.6](../deferred.md) |
| Thermal unit commitment | Binary on/off variables incompatible with LP relaxation         | Deferred | [Deferred Features SSC.1](../deferred.md) |
| Minimum generation      | Big-M or indicator constraints break LP convexity               | Deferred | —                                         |
| Startup/shutdown costs  | Multi-period linking constraints across blocks                  | Deferred | —                                         |

**Key invariant**: Non-convex refinements do **not** affect state propagation to the next stage. State transition always uses the original LP solution to maintain consistency with the trained policy. The refined solution replaces the LP solution for output purposes only.

## 6. Output Streaming

### 6.1 Streaming Architecture

With potentially thousands of scenarios, storing all results in memory before writing is impractical. The output writer uses a streaming architecture:

- A **bounded channel** of type `Sender<SimulationScenarioResult>` / `Receiver<SimulationScenarioResult>` connects the simulation threads to a dedicated **background I/O thread**. The `SimulationScenarioResult` type (SS3.4.3) is the channel payload.
- Simulation threads send completed scenario results through the channel as they finish. Each scenario produces exactly one `SimulationScenarioResult` instance.
- The I/O thread receives `SimulationScenarioResult` values and writes them to Parquet files asynchronously via the `SimulationParquetWriter` ([Output Infrastructure SS6.2](../data-model/output-infrastructure.md)), converting the nested per-entity-type layout to the columnar Parquet schemas defined in [Output Schemas SS5](../data-model/output-schemas.md).
- The channel backpressure prevents simulation from running too far ahead of I/O. At most `channel_capacity` `SimulationScenarioResult` instances exist simultaneously in the channel buffer.

> **Placeholder** — The output streaming pipeline diagram (`../../diagrams/exports/svg/data/output-streaming-pipeline.svg`) will be revised after the text review is complete.

### 6.2 Output Detail Levels

The amount of data written per scenario depends on the configured output detail level:

| Detail Level | What Is Written                               | Use Case                        |
| ------------ | --------------------------------------------- | ------------------------------- |
| Summary      | Only aggregate cost per scenario              | Quick policy quality assessment |
| Stage-level  | Per-stage aggregates (cost, deficit, storage) | Storage trajectory analysis     |
| Full         | Per-scenario, per-stage, per-entity detail    | Detailed operational analysis   |

For the complete column definitions at each detail level, see [Output Schemas SS5](../data-model/output-schemas.md).

### 6.3 Distributed Output

Each MPI rank writes its simulation results independently using **per-rank output files**. This avoids MPI coordination during I/O and enables each rank to write at full local disk bandwidth.

The output uses Hive-style partitioning by `scenario_id`, where each rank writes exclusively to the partitions corresponding to its assigned scenarios. No coordination between ranks is needed during writing — the partition assignment is determined by the deterministic scenario distribution (SS3.1).

After all ranks complete, rank 0 writes the simulation manifest (`_manifest.json`) with checksums, row counts, and partition listings. The `_SUCCESS` marker file is written atomically on successful completion. See [Output Infrastructure SS1.1](../data-model/output-infrastructure.md).

## Cross-References

- [CLI and Lifecycle](./cli-and-lifecycle.md) — Execution phases and conditional simulation mode (SS5.3)
- [Scenario Generation](./scenario-generation.md) — Sampling scheme abstraction (SS3), scenario distribution (SS5), noise inversion for external scenarios (SS4.3)
- [Training Loop](./training-loop.md) — Training phase that produces the policy evaluated by simulation
- [Solver Interface Trait](./solver-interface-trait.md) — Compile-time monomorphization pattern for `fn simulate()` (SS5)
- [Communicator Trait](../hpc/communicator-trait.md) — Generic communicator backend for MPI operations (§3)
- [Internal Structures](../data-model/internal-structures.md) — `System` struct (SS1), `EntityId` type (SS1.8), `Stage`/`Block` definitions (SS12)
- [Binary Formats](../data-model/binary-formats.md) — Policy file format for FCF cuts (SS3.2)
- [Block Formulations](../math/block-formulations.md) — Block structure (parallel/chronological) within simulation LP stages
- [Risk Measures](../math/risk-measures.md) — CVaR computation for simulation cost statistics
- [Discount Rate](../math/discount-rate.md) — Discount factors applied to scenario costs
- [LP Formulation](../math/lp-formulation.md) — Stage LP construction shared between training and simulation
- [Penalty System](../data-model/penalty-system.md) — Three-category cost taxonomy referenced by `ScenarioCategoryCosts` (SS2)
- [Output Schemas](../data-model/output-schemas.md) — Parquet column definitions for all simulation output files (SS5.1--5.11)
- [Output Infrastructure](../data-model/output-infrastructure.md) — Manifests and crash recovery (SS1), MPI Hive partitioning (SS2-SS4), `SimulationParquetWriter` output writer API (SS6.2)
- [Deferred Features SSC.1](../deferred.md) — GNL / thermal unit commitment (simulation-only MIP)
- [Deferred Features SSC.6](../deferred.md) — FPHA enhancements including head-dependent refinement
- [Deferred Features SSC.9](../deferred.md) — Policy compatibility validation specification
- [Deferred Features SSC.13](../deferred.md) — Alternative forward pass model (simulation-only LP in training)
