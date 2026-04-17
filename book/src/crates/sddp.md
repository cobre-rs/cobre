# cobre-sddp

<span class="status-alpha">alpha</span>

`cobre-sddp` implements the Stochastic Dual Dynamic Programming (SDDP) algorithm
(Pereira & Pinto, 1991) for long-term hydrothermal dispatch and energy planning.
It is the first algorithm vertical in the Cobre ecosystem: a training loop that
iteratively improves a piecewise-linear approximation of the value function for
multi-stage stochastic linear programs.

For the mathematical foundations — including the Benders decomposition,
cut coefficient derivation, and risk measure theory — see the
[methodology reference](https://cobre-rs.github.io/cobre-docs/).

This crate depends on `cobre-core` for system data types, `cobre-stochastic` for
inflow scenario generation and load noise parameters, `cobre-solver` for LP
subproblem solving, and `cobre-comm` for distributed communication.

## Iteration lifecycle

Each training iteration follows a fixed eight-step sequence. The ordering
reflects the correction introduced in the lower bound plan fix (F-019): the
lower bound is evaluated **after** the backward pass and cut synchronization,
not during forward synchronization.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 1  Forward pass                                                   │
│          Each rank simulates config.forward_passes scenarios through     │
│          all stages, solving the LP at each (scenario, stage) pair with  │
│          the current FCF approximation.                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 2  Forward sync                                                   │
│          allreduce (sum + broadcast) aggregates local UB statistics into │
│          a global mean, standard deviation, and 95% CI half-width.      │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 3  State exchange                                                 │
│          allgatherv gathers all ranks' trial point state vectors so     │
│          every rank can solve the backward pass at ALL trial points.    │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 4  Backward pass                                                  │
│          Sweeps stages T-2 down to 0, solving the successor LP under    │
│          every opening from the fixed tree, extracting LP duals to form  │
│          Benders cut coefficients, and inserting one cut per trial point  │
│          per stage into the Future Cost Function (FCF).                  │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 5  Cut sync                                                       │
│          allgatherv shares each rank's newly generated cuts so that all  │
│          ranks maintain an identical FCF at the end of each iteration.  │
│                                                                         │
│  Step 5a Cut management pipeline (optional, two stages)                 │
│          S1: Strategy-based selection (Level1/LML1/Dominated) —         │
│              runs at multiples of check_frequency.                      │
│          S2: Budget enforcement — hard cap on active cuts per stage,    │
│              runs every iteration when max_active_per_stage is set.     │
│                                                                         │
│  Step 5b LB evaluation                                                  │
│          Rank 0 solves the stage-0 LP for every opening in the tree    │
│          and aggregates the objectives via the stage-0 risk measure.    │
│          The scalar lower bound is broadcast to all ranks.              │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 6  Convergence check                                              │
│          The ConvergenceMonitor updates bound statistics and evaluates   │
│          the configured stopping rules to determine whether to stop.    │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 7  Checkpoint                                                     │
│          The FlatBuffers policy checkpoint infrastructure is             │
│          implemented in cobre-io (write_policy_checkpoint). The CLI     │
│          writes a final snapshot after training completes. Periodic     │
│          in-loop writes via checkpoint_interval are not yet wired       │
│          into the training loop.                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 8  Event emission                                                 │
│          TrainingEvent values are sent to the optional event channel    │
│          for real-time monitoring by the CLI or TUI layer.              │
└─────────────────────────────────────────────────────────────────────────┘
```

The convergence gap is computed as:

```text
gap = (UB - LB) / max(1.0, |UB|)
```

The `max(1.0, |UB|)` guard prevents division by zero when the upper bound is
near zero.

## Module overview

| Module                | Responsibility                                                                                                                                                                                                                                                                     |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `training`            | `train`: the top-level loop orchestrator; wires all steps together                                                                                                                                                                                                                 |
| `forward`             | `run_forward_pass`, `sync_forward`: step 1 and step 2                                                                                                                                                                                                                              |
| `state_exchange`      | `ExchangeBuffers`: step 3 allgatherv of trial point state vectors                                                                                                                                                                                                                  |
| `backward`            | `run_backward_pass`: step 4 Benders cut generation with work-stealing parallelism                                                                                                                                                                                                  |
| `cut_sync`            | `CutSyncBuffers`: step 5 allgatherv of new cut wire records                                                                                                                                                                                                                        |
| `cut_selection`       | `CutSelectionStrategy`, `CutMetadata`, `DeactivationSet`: step 5a Stage 1 pool pruning                                                                                                                                                                                             |
| `lower_bound`         | `evaluate_lower_bound`: step 5b risk-adjusted LB computation (parallelized across openings)                                                                                                                                                                                        |
| `convergence`         | `ConvergenceMonitor`: step 6 bound tracking and stopping rule evaluation                                                                                                                                                                                                           |
| `cut`                 | `CutPool`, `FutureCostFunction`, `CutRowMap`, `WARM_START_ITERATION`: cut data structures, wire format, and LP row mapping                                                                                                                                                         |
| `basis_padding`       | `pad_basis_for_cuts`: Strategy S3 basis-aware warm-start padding — evaluates new cuts at warm-start state and assigns informed basis status                                                                                                                                        |
| `config`              | `TrainingConfig`: algorithm parameters                                                                                                                                                                                                                                             |
| `context`             | `StageContext`, `TrainingContext`: hot-path argument bundles that absorb parameters into context structs                                                                                                                                                                           |
| `stopping_rule`       | `StoppingRule`, `StoppingRuleSet`, `MonitorState`: termination criteria                                                                                                                                                                                                            |
| `risk_measure`        | `RiskMeasure`, `BackwardOutcome`: risk-neutral and CVaR aggregation                                                                                                                                                                                                                |
| `horizon_mode`        | `HorizonMode`: finite vs. cyclic stage traversal (only `Finite` currently)                                                                                                                                                                                                         |
| `indexer`             | `StageIndexer`, `EquipmentCounts`, `FphaColumnLayout`: LP column/row offset arithmetic for stage subproblems                                                                                                                                                                       |
| `lp_builder`          | `build_stage_templates`, `StageTemplates`, `PatchBuffer`: stage template construction, LP scaling, and row-bound patch arrays                                                                                                                                                      |
| `workspace`           | `SolverWorkspace`, `WorkspacePool`, `BasisStore`: per-worker solver instances with pre-allocated scratch buffers and basis storage                                                                                                                                                 |
| `trajectory`          | `TrajectoryRecord`: forward pass LP solution record (primal, dual, state, cost)                                                                                                                                                                                                    |
| `noise`               | Noise-to-RHS-patch logic shared across forward, backward, and simulation passes; includes `accumulate_and_shift_lag_state` for sub-monthly lag accumulation                                                                                                                        |
| `lag_transition`      | `precompute_stage_lag_transitions`: builds per-stage `StageLagTransition` configs from stage dates and lag period boundaries; accumulator seeding from `RecentObservation` for mid-season starts                                                                                   |
| `solver_stats`        | `SolverStatsEntry`, `SolverStatsDelta`, `aggregate_solver_statistics`: per-phase solver statistics delta computation and cross-worker aggregation                                                                                                                                  |
| `scaling_report`      | `ScalingReport`, `StageScalingReport`, `CoefficientRange`: LP prescaling diagnostics written to JSON                                                                                                                                                                               |
| `simulation`          | Full simulation pipeline with stage-major loop; all result types (`SimulationHydroResult`, etc.); `simulate`, `aggregate_simulation`                                                                                                                                               |
| `error`               | `SddpError`: unified error type aggregating solver, comm, stochastic, and I/O errors                                                                                                                                                                                               |
| `fpha_fitting`        | FPHA fitting pipeline — computes piecewise-linear hydroelectric production hyperplanes from reservoir geometry                                                                                                                                                                     |
| `hydro_models`        | `prepare_hydro_models`, `EvaporationModel`, `FphaPlane`, `ResolvedProductionModel`: hydro model preprocessing at initialization                                                                                                                                                    |
| `generic_constraints` | Generic constraint row entries — user-defined linear constraints with 20 variable types                                                                                                                                                                                            |
| `inflow_method`       | `InflowNonNegativityMethod`: Truncation, Penalty, TruncationWithPenalty, and None strategies                                                                                                                                                                                       |
| `estimation`          | `EstimationReport`, `LagScaleWarning`, `StdRatioDivergence`: PAR estimation pipeline outputs                                                                                                                                                                                       |
| `provenance`          | `ModelProvenanceReport`, `build_provenance_report`: round-trip audit trail for model preprocessing                                                                                                                                                                                 |
| `stochastic_summary`  | `StochasticSummary`, `build_stochastic_summary`: human-readable summary of stochastic preprocessing                                                                                                                                                                                |
| `visited_states`      | `VisitedStatesArchive`: forward-pass trial point storage for cut selection and policy diagnostics                                                                                                                                                                                  |
| `policy_export`       | Policy checkpoint writing (FlatBuffers cuts, basis, states, metadata)                                                                                                                                                                                                              |
| `policy_load`         | `build_basis_cache_from_checkpoint`, `validate_policy_compatibility`, `load_boundary_cuts`, `inject_boundary_cuts`: policy loading for warm-start, resume, and terminal boundary cut injection from external checkpoints                                                           |
| `training_output`     | `build_training_output`: assembles all training results for the output writers                                                                                                                                                                                                     |
| `conversion`          | Type conversion utilities between internal and I/O representations                                                                                                                                                                                                                 |
| `setup`               | `StudySetup`, `StudyParams`, `prepare_stochastic`: pre-built study state; holds four optional scenario libraries (`historical_library`, `external_inflow_library`, `external_load_library`, `external_ncs_library`) built conditionally from per-class `SamplingScheme` selections |

## Configuration

### `TrainingConfig`

`TrainingConfig` controls the training loop parameters. All fields are public
and must be set explicitly — there is no `Default` implementation, preventing
silent misconfigurations.

| Field                   | Type                            | Description                                               |
| ----------------------- | ------------------------------- | --------------------------------------------------------- |
| `forward_passes`        | `u32`                           | Scenarios per rank per iteration (must be >= 1)           |
| `max_iterations`        | `u64`                           | Safety bound on total iterations; also sizes the cut pool |
| `checkpoint_interval`   | `Option<u64>`                   | Write checkpoint every N iterations; `None` = disabled    |
| `warm_start_cuts`       | `Vec<u32>`                      | Per-stage pre-loaded cut counts from a policy file        |
| `event_sender`          | `Option<Sender<TrainingEvent>>` | Channel for real-time monitoring events; `None` = silent  |
| `cut_selection`         | `Option<CutSelectionStrategy>`  | Stage 1 cut selection strategy; `None` = no selection     |
| `budget`                | `Option<u32>`                   | Stage 2 max active cuts per stage; `None` = no budget     |
| `basis_padding_enabled` | `bool`                          | Strategy S3 basis-aware warm-start padding                |

### `StoppingRuleSet`

The stopping rule set composes one or more termination criteria. Every set
must include an `IterationLimit` rule as a safety bound against infinite loops.

| Rule variant       | Trigger condition                                                   |
| ------------------ | ------------------------------------------------------------------- |
| `IterationLimit`   | `iteration >= limit`                                                |
| `TimeLimit`        | `wall_time_seconds >= seconds`                                      |
| `BoundStalling`    | Relative LB improvement over a sliding window falls below tolerance |
| `SimulationBased`  | Periodic Monte Carlo simulation costs stabilize                     |
| `GracefulShutdown` | External SIGTERM / SIGINT received (always evaluated first)         |

The `mode` field controls how multiple rules combine:

- `StoppingMode::Any` (OR): stop when any rule triggers.
- `StoppingMode::All` (AND): stop when all rules trigger simultaneously.

```rust,ignore
use cobre_sddp::stopping_rule::{StoppingMode, StoppingRule, StoppingRuleSet};

let stopping_rules = StoppingRuleSet {
    rules: vec![
        StoppingRule::IterationLimit { limit: 500 },
        StoppingRule::BoundStalling {
            tolerance: 0.001,
            iterations: 20,
        },
        StoppingRule::GracefulShutdown,
    ],
    mode: StoppingMode::Any,
};
```

### `RiskMeasure`

`RiskMeasure` controls how per-opening backward pass outcomes are aggregated
into Benders cuts and how the lower bound is computed.

| Variant       | Description                                                                           |
| ------------- | ------------------------------------------------------------------------------------- |
| `Expectation` | Risk-neutral expected value. Weights equal opening probabilities.                     |
| `CVaR`        | Convex combination `(1 - λ)·E[Z] + λ·CVaR_α[Z]`. `alpha` ∈ (0, 1], `lambda` ∈ [0, 1]. |

`alpha = 1` with `CVaR` is equivalent to `Expectation`. `lambda = 0` with
`CVaR` is also equivalent to `Expectation`. One `RiskMeasure` value is
assigned per stage from the `stages.json` configuration field `risk_measure`.

### `CutSelectionStrategy`

Cut selection is optional. When configured, it forms Stage 1 of the
two-stage cut management pipeline that also includes budget enforcement
(Stage 2). See the user-facing
[Performance Accelerators](../guide/performance-accelerators.md#cut-management-pipeline)
guide for the full pipeline description.

| Variant     | Deactivation condition                                               |
| ----------- | -------------------------------------------------------------------- |
| `Level1`    | `active_count <= threshold` (never active; least aggressive)         |
| `Lml1`      | `iteration - last_active_iter > memory_window` (outside time window) |
| `Dominated` | Dominated at all visited forward pass states (most aggressive)       |

All variants respect a `check_frequency` parameter: selection only runs at
iterations that are multiples of `check_frequency` and never at iteration 0.
Stage 0 is always exempt.

`Dominated` selection performs `O(|active cuts| x |visited states|)` work
per stage per check. It uses the `VisitedStatesArchive` (always collected
during training) and the `domination_epsilon` tolerance parameter.

## Key data structures

### `StudySetup`

`StudySetup` is constructed once by `StudySetup::new` from a validated `System` and `Config`. It owns all precomputed state — stage templates, stochastic context, FCF, indexer, initial state, risk measures, and entity counts — and holds it across async boundaries without lifetime issues.

Four optional library fields are built conditionally based on per-class `SamplingScheme` selections:

| Field                     | Type                                | Built when                                    |
| ------------------------- | ----------------------------------- | --------------------------------------------- |
| `historical_library`      | `Option<HistoricalScenarioLibrary>` | `inflow_scheme == SamplingScheme::Historical` |
| `external_inflow_library` | `Option<ExternalScenarioLibrary>`   | `inflow_scheme == SamplingScheme::External`   |
| `external_load_library`   | `Option<ExternalScenarioLibrary>`   | `load_scheme == SamplingScheme::External`     |
| `external_ncs_library`    | `Option<ExternalScenarioLibrary>`   | `ncs_scheme == SamplingScheme::External`      |

Callers borrow `StudySetup` to construct `TrainingContext` and `StageContext`; the public accessor methods (`historical_library()`, `external_inflow_library()`, etc.) return `Option<&T>` and are `None` for sampling schemes that do not use those libraries.

### `FutureCostFunction`

The Future Cost Function (FCF) holds one `CutPool` per stage. Each `CutPool`
is a pre-allocated flat array of cut slots. Cuts are inserted deterministically
by `(iteration, forward_pass_index)` to guarantee bit-for-bit identical FCF
state across all MPI ranks.

The FCF is built once before training begins. Total slot capacity is
`warm_start_cuts + max_iterations * forward_passes` per stage.

### `PatchBuffer`

A `PatchBuffer` holds the three parallel arrays consumed by the LP solver's
`set_row_bounds` call. It is sized for `N * (2 + L) + M * B` patches, where
N is the number of hydro plants, L is the maximum PAR order, M is the number
of stochastic load buses, and B is the maximum block count across stages:

- **Category 1** `[0, N)` — storage-fixing: equality constraint at incoming storage.
- **Category 2** `[N, N*(1+L))` — lag-fixing: equality constraint at AR lagged inflows.
- **Category 3** `[N*(1+L), N*(2+L))` — noise-fixing: equality constraint at scenario noise.
- **Category 4** `[N*(2+L), N*(2+L) + M*B_active)` — load balance row patches: equality
  constraint at stochastic load demand per bus per block (optional; empty when
  `n_load_buses == 0`).

The backward pass uses only categories 1 and 2 (`fill_state_patches`) for
the state-fixing rows, then applies Category 4 (`fill_load_patches`) to set
the stochastic load demand at each bus before solving the successor LP.
The forward pass uses all four categories (`fill_forward_patches` followed by
`fill_load_patches`).

When `n_load_buses == 0`, Category 4 is empty and `forward_patch_count`
returns `N*(2+L)` unchanged, making load noise an optional zero-cost extension.

### `ExchangeBuffers` and `CutSyncBuffers`

Both types pre-allocate all communication buffers once at construction time and
reuse them across all stages and iterations. This keeps the per-stage exchange
allocation-free on the hot path.

`ExchangeBuffers` handles the state vector allgatherv (step 3):

- Send buffer: `local_count * n_state` floats.
- Receive buffer: `local_count * num_ranks * n_state` floats (rank-major order).

`CutSyncBuffers` handles the cut wire allgatherv (step 5):

- Send buffer: `max_cuts_per_rank * cut_wire_size(n_state)` bytes.
- Receive buffer: `max_cuts_per_rank * num_ranks * cut_wire_size(n_state)` bytes.

## Load noise integration

When `load_seasonal_stats.parquet` is present in the case directory, the
`cobre-io` loader populates a `PrecomputedNormal` (from `cobre-stochastic`)
alongside the PAR model. This object stores the per-stage, per-bus mean and
standard deviation for stochastic bus demand and the per-block load factors
derived from the seasonal statistics.

The forward and backward passes apply stochastic load noise as follows:

1. **Noise drawing**: for each stochastic load bus `i` at stage `t`, the pass
   draws a standard normal variate `eta` (from the shared noise vector whose
   first `n_hydros` entries are inflow innovations and next `n_load_buses`
   entries are load innovations). The realized demand is:

   ```text
   load_rhs[i * K + blk] = max(0, mean(t, i) + std(t, i) * eta) * block_factor(t, i, blk)
   ```

   The `max(0, ...)` clamp prevents negative demand. `block_factor` scales the
   base realization by the per-block load profile.

2. **Load patching**: `fill_load_patches` writes each `load_rhs` entry into
   Category 4 of the `PatchBuffer`, targeting the load balance row for that
   bus and block. Row indices are provided by `load_balance_row_starts` (one
   per stage) and `load_bus_indices` (position of each stochastic bus within
   the LP row layout).

3. **State independence**: load noise realizations do not produce additional
   state variables. The Benders cut coefficients cover only the hydro state
   dimensions (storage volumes and AR lags). Load noise enters the subproblem
   purely as a right-hand side perturbation of the bus power balance constraints.

Load noise follows the same PAR(p) framework used for inflow noise — the
combined noise vector `[inflow_noise | load_noise]` is drawn from the
correlated multivariate normal defined by the `StochasticContext`. For details
on the PAR(p) model and correlation structure, see the `cobre-stochastic`
crate page.

## Convergence monitoring

`ConvergenceMonitor` tracks bound statistics and evaluates stopping rules. It
is constructed once before the loop begins and updated at the end of each
iteration via `update(lb, &sync_result)`.

```rust
use cobre_sddp::convergence::ConvergenceMonitor;
use cobre_sddp::forward::SyncResult;
use cobre_sddp::stopping_rule::{StoppingMode, StoppingRule, StoppingRuleSet};

let rule_set = StoppingRuleSet {
    rules: vec![StoppingRule::IterationLimit { limit: 100 }],
    mode: StoppingMode::Any,
};

let mut monitor = ConvergenceMonitor::new(rule_set);

let sync = SyncResult {
    global_ub_mean: 110.0,
    global_ub_std: 5.0,
    ci_95_half_width: 2.0,
    sync_time_ms: 10,
};

let (stop, results) = monitor.update(100.0, &sync);
assert!(!stop);
assert_eq!(monitor.iteration_count(), 1);
// gap = (110 - 100) / max(1.0, 110.0) = 10/110
assert!((monitor.gap() - 10.0 / 110.0).abs() < 1e-10);
```

Accessor methods on `ConvergenceMonitor`:

| Method               | Returns                                        |
| -------------------- | ---------------------------------------------- |
| `lower_bound()`      | Latest LB value                                |
| `upper_bound()`      | Latest UB mean                                 |
| `upper_bound_std()`  | Latest UB standard deviation                   |
| `ci_95_half_width()` | Latest 95% CI half-width                       |
| `gap()`              | Convergence gap: (UB - LB) / max(1.0, abs(UB)) |
| `iteration_count()`  | Number of completed `update` calls             |
| `set_shutdown()`     | Signal a graceful shutdown before next update  |

## Event system

The training loop emits `TrainingEvent` values (from `cobre-core`) at each
lifecycle step boundary when `config.event_sender` is `Some`. Events carry
structured data for real-time display in the TUI or CLI layers.

Key events emitted during training:

| Event variant               | When emitted                                                  |
| --------------------------- | ------------------------------------------------------------- |
| `ForwardPassComplete`       | After step 1 completes for all local scenarios                |
| `ForwardSyncComplete`       | After step 2 global UB statistics are merged                  |
| `BackwardPassComplete`      | After step 4 cut generation for all trial points              |
| `CutSyncComplete`           | After step 5 cut allgatherv                                   |
| `CutSelectionComplete`      | After step 5a Stage 1 selection (when strategy is set)        |
| `BudgetEnforcementComplete` | After step 5a Stage 2 budget enforcement (when budget is set) |
| `ConvergenceUpdate`         | After step 6 stopping rules evaluated                         |
| `IterationSummary`          | At the end of each iteration (LB, UB, gap, timing)            |
| `TrainingFinished`          | When a stopping rule triggers                                 |

## Quick start (pseudocode)

The following shows the shape of a `train` call. All arguments must be built
from the upstream pipeline (`cobre-io` for system data, `cobre-stochastic` for
the opening tree, `cobre-solver` for the LP solver instance).

```rust,ignore
use cobre_sddp::{
    FutureCostFunction, HorizonMode, RiskMeasure, StageIndexer,
    TrainingConfig, TrainingResult,
    stopping_rule::{StoppingMode, StoppingRule, StoppingRuleSet},
    train,
};

// Build the FCF for num_stages stages, n_state state dimensions,
// forward_passes scenarios per rank, max_iterations iterations.
let mut fcf = FutureCostFunction::new(num_stages, n_state, forward_passes, max_iterations, &vec![0; num_stages]);

let config = TrainingConfig {
    forward_passes: 10,
    max_iterations: 500,
    checkpoint_interval: None,
    warm_start_cuts: 0,
    event_sender: None,
};

let stopping_rules = StoppingRuleSet {
    rules: vec![
        StoppingRule::IterationLimit { limit: 500 },
        StoppingRule::GracefulShutdown,
    ],
    mode: StoppingMode::Any,
};

let horizon = HorizonMode::Finite { num_stages };

let result: TrainingResult = train(
    &mut solver,        // SolverInterface impl (e.g., HiGHS)
    config,
    &mut fcf,
    &templates,         // one StageTemplate per stage
    &base_rows,         // AR dynamics base row index per stage
    &indexer,           // StageIndexer from StageIndexer::new(n_hydro, max_par_order)
    &initial_state,     // known initial storage volumes
    &opening_tree,      // from cobre_stochastic::build_stochastic_context
    &stochastic,        // StochasticContext
    &horizon,
    &risk_measures,     // one RiskMeasure per stage
    stopping_rules,
    None,               // no cut selection in this example
    None,               // no external shutdown flag
    &comm,              // Communicator (LocalBackend or FerrompiBackend)
)?;

println!(
    "Converged in {} iterations: LB={:.2}, UB={:.2}, gap={:.4}",
    result.iterations, result.final_lb, result.final_ub, result.final_gap
);
```

## Error handling

All fallible operations return `Result<T, SddpError>`. The error type is
`Send + Sync + 'static` and can be propagated across thread boundaries or
wrapped by `anyhow`.

| `SddpError` variant | Trigger                                                  |
| ------------------- | -------------------------------------------------------- |
| `Solver`            | LP solve failed for numerical or timeout reasons         |
| `Communication`     | MPI collective operation failed                          |
| `Stochastic`        | Scenario generation or PAR model validation failed       |
| `Io`                | Case directory loading or validation failed              |
| `Validation`        | Algorithm configuration is semantically invalid          |
| `Infeasible`        | LP has no feasible solution (stage, iteration, scenario) |
| `Simulation`        | Simulation phase error (LP failure, I/O, policy issue)   |

## Performance notes

For a comprehensive user-facing guide to all performance optimizations, see the
[Performance Accelerators](../guide/performance-accelerators.md) chapter.

### Pre-allocation discipline

The training loop makes no heap allocations on the hot path inside the
iteration loop. All workspace buffers are allocated once before the loop:

- `WorkspacePool`: one `SolverWorkspace` per thread (solver + PatchBuffer + ScratchBuffers + Basis).
- `TrajectoryRecord` flat vec: `forward_passes * num_stages` records.
- `PatchBuffer`: `N * (2 + L) + M * max_blocks` entries per worker.
- `ExchangeBuffers`: `local_count * num_ranks * n_state` floats.
- `CutSyncBuffers`: `max_cuts_per_rank * num_ranks * cut_wire_size(n_state)` bytes.
- `ScratchBuffers`: noise, inflow, lag matrix, PAR, eta, load, z-inflow buffers per worker.
- `BasisStore`: `forward_passes * num_stages` basis slots.

### Backward pass work-stealing

The inner trial-point loop in the backward pass uses atomic counter
work-stealing (`AtomicUsize::fetch_add(1, Relaxed)`) instead of static
partitioning. Staged cuts are sorted by `trial_point_idx` after the parallel
region to preserve bit-for-bit determinism across thread counts.

### Model persistence and incremental cuts

`CutRowMap` provides O(1) slot-to-row lookup for the persistent lower-bound
LP so the append path skips cuts that are already present. The LB LP is
strictly append-only: cuts are never removed from it, which keeps the lower
bound monotonically non-decreasing. The shared cut pool's active/inactive
bit is not propagated to the LB LP — pool-deactivated cuts remain as LP
rows in the LB solver.

### Cut wire format

The cut wire format used by `CutSyncBuffers` is a fixed-size record:
24 bytes of header (slot index, iteration, forward pass index, intercept)
followed by `n_state * 8` bytes of coefficients. The record size is
`cut_wire_size(n_state) = 24 + n_state * 8` bytes.

### Communication-free parallelism

Forward pass noise is generated without inter-rank communication. Each rank
independently derives its noise seed from `(base_seed, iteration, scenario, stage_id)`
using deterministic SipHash-1-3 seed derivation from `cobre-stochastic`. The opening tree is
pre-generated once before training and shared read-only across all iterations.

### Solver statistics instrumentation

Per-call, per-phase timing and counting of all solver operations is tracked
in `SolverStatistics` (18 fields) and written to `training/solver/iterations.parquet`
and `training/solver/retry_histogram.parquet`. In multi-threaded runs,
per-worker statistics are aggregated via `aggregate_solver_statistics()` which
sums all fields across workers.

## Testing

```
cargo test -p cobre-sddp --all-features
```

The crate requires no external system libraries beyond what is needed by the
workspace (HiGHS is always available; MPI is optional via the `mpi` feature
of `cobre-comm`).

### Test suite overview

The test suite covers:

- Unit tests for each module's core logic.
- Integration tests using `LocalBackend` (single-rank) for the
  communication-involving modules (`forward`, `backward`, `cut_sync`,
  `state_exchange`, `lower_bound`, `training`).
- Doc-tests for all public types and functions with constructible examples.

## Feature flags

`cobre-sddp` has no optional feature flags of its own. Feature flag propagation
from `cobre-comm` (the `mpi` feature) controls whether MPI-based distributed
training is available at link time.

```toml
# Cargo.toml
cobre-sddp = { version = "0.1" }
```
