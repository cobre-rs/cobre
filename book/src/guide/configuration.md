# Configuration

All runtime parameters for `cobre run` are controlled by `config.json` in the
case directory. This page documents every section and field.

---

## Minimal Config

```json
{
  "training": {
    "forward_passes": 50,
    "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }]
  }
}
```

All other sections are optional with defaults documented below.

---

## `training`

Controls the SDDP training phase.

### Mandatory Fields

| Field            | Type    | Description                                                                                                                           |
| ---------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `forward_passes` | integer | Number of scenario trajectories per iteration. Larger values reduce variance in each iteration's cut but increase cost per iteration. |
| `stopping_rules` | array   | At least one stopping rule (see below). The rule set must contain at least one `iteration_limit` rule.                                |

### Optional Fields

| Field           | Type               | Default | Description                                                                                                                                                                                             |
| --------------- | ------------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`       | boolean            | `true`  | Set to `false` to skip training and proceed directly to simulation (requires a pre-trained policy).                                                                                                     |
| `tree_seed`     | integer            | `null`  | Random seed for the opening scenario tree. When `null`, a default seed of 42 is used (deterministic but arbitrary). See [Stochastic Modeling](./stochastic-modeling.md) for the dual-seed architecture. |
| `stopping_mode` | `"any"` or `"all"` | `"any"` | How multiple stopping rules combine: `"any"` stops when the first rule is satisfied; `"all"` requires all rules to be satisfied simultaneously.                                                         |

For the per-class `scenario_source` configuration, see the
[`scenario_source` sub-section](#scenario_source) below and
[Stochastic Modeling](./stochastic-modeling.md).

### `scenario_source`

Controls where the forward-pass noise comes from for each entity class during
training. When absent, all classes default to `in_sample` (reusing the
pre-generated opening tree).

| Field              | Type              | Default       | Description                                                                                             |
| ------------------ | ----------------- | ------------- | ------------------------------------------------------------------------------------------------------- |
| `seed`             | integer or `null` | `null`        | Shared forward-pass seed for `out_of_sample`, `historical`, and `external` schemes.                     |
| `inflow`           | object            | `in_sample`   | Sampling scheme for hydro inflow. Object with `"scheme"` key.                                           |
| `load`             | object            | `in_sample`   | Sampling scheme for bus load. Object with `"scheme"` key.                                               |
| `ncs`              | object            | `in_sample`   | Sampling scheme for NCS availability. Object with `"scheme"` key.                                       |
| `historical_years` | array or object   | auto-discover | Restrict the pool of historical windows. List (`[1940, 1953]`) or range (`{"from": 1940, "to": 2010}`). |

Valid values for `"scheme"`: `"in_sample"`, `"out_of_sample"`, `"historical"`, `"external"`.

Example — out-of-sample inflow with in-sample load and NCS:

```json
{
  "training": {
    "tree_seed": 42,
    "forward_passes": 50,
    "stopping_rules": [{ "type": "iteration_limit", "limit": 200 }],
    "scenario_source": {
      "seed": 99,
      "inflow": { "scheme": "out_of_sample" },
      "load": { "scheme": "in_sample" },
      "ncs": { "scheme": "in_sample" }
    }
  }
}
```

See [Stochastic Modeling — Sampling Schemes](./stochastic-modeling.md#sampling-schemes)
for a full description of each scheme and the `historical_years` field.

### Stopping Rules

Each entry in `stopping_rules` is a JSON object with a `"type"` discriminator.

#### `iteration_limit`

Stop after a fixed number of training iterations.

```json
{ "type": "iteration_limit", "limit": 200 }
```

| Field   | Type    | Description                               |
| ------- | ------- | ----------------------------------------- |
| `limit` | integer | Maximum number of SDDP iterations to run. |

#### `time_limit`

Stop after a wall-clock time budget is exhausted.

```json
{ "type": "time_limit", "seconds": 3600.0 }
```

| Field     | Type  | Description                       |
| --------- | ----- | --------------------------------- |
| `seconds` | float | Maximum training time in seconds. |

#### `bound_stalling`

Stop when the relative improvement in the lower bound falls below a threshold.

```json
{ "type": "bound_stalling", "iterations": 20, "tolerance": 0.0001 }
```

| Field        | Type    | Description                                                                                              |
| ------------ | ------- | -------------------------------------------------------------------------------------------------------- |
| `iterations` | integer | Window size: the number of past iterations over which to compute the relative improvement.               |
| `tolerance`  | float   | Relative improvement threshold. Training stops when the improvement over the window is below this value. |

#### `simulation`

Stop when both the lower bound and a Monte Carlo policy cost estimate have
stabilized. Periodically runs a batch of forward simulations and compares
the result against previous evaluations.

```json
{
  "type": "simulation",
  "replications": 100,
  "period": 10,
  "bound_window": 5,
  "distance_tol": 0.01,
  "bound_tol": 0.0001
}
```

| Field          | Type    | Description                                                           |
| -------------- | ------- | --------------------------------------------------------------------- |
| `replications` | integer | Number of Monte Carlo forward simulations per check.                  |
| `period`       | integer | Iterations between simulation checks.                                 |
| `bound_window` | integer | Number of past iterations for bound stability check.                  |
| `distance_tol` | float   | Normalized distance threshold between consecutive simulation results. |
| `bound_tol`    | float   | Relative tolerance for bound stability.                               |

### `stopping_mode`

When multiple stopping rules are listed, `stopping_mode` controls how they
combine:

- `"any"` (default): stop when any one rule is satisfied.
- `"all"`: stop only when every rule is satisfied simultaneously.

```json
{
  "training": {
    "forward_passes": 50,
    "stopping_mode": "all",
    "stopping_rules": [
      { "type": "iteration_limit", "limit": 500 },
      { "type": "bound_stalling", "iterations": 20, "tolerance": 0.0001 }
    ]
  }
}
```

---

## `simulation`

Controls the optional post-training simulation phase.

| Field           | Type      | Default   | Description                                                                       |
| --------------- | --------- | --------- | --------------------------------------------------------------------------------- |
| `enabled`       | boolean   | `false`   | Enable the simulation phase after training.                                       |
| `num_scenarios` | integer   | `2000`    | Number of independent Monte Carlo simulation scenarios to evaluate.               |
| `policy_type`   | `"outer"` | `"outer"` | Policy representation for simulation. `"outer"` uses the cut pool (Benders cuts). |

When `simulation.enabled` is `false` or `num_scenarios` is `0`, the simulation
phase is skipped entirely.

Example:

```json
{
  "simulation": {
    "enabled": true,
    "num_scenarios": 1000
  }
}
```

### `scenario_source`

Controls where the forward-pass noise comes from during the simulation phase.
When absent, simulation falls back to the scheme configured under
`training.scenario_source`. This allows you to train with in-sample noise and
simulate with a different scheme (for example, out-of-sample or historical)
without modifying the training configuration.

The fields are identical to `training.scenario_source`:

| Field              | Type              | Default       | Description                                                                                             |
| ------------------ | ----------------- | ------------- | ------------------------------------------------------------------------------------------------------- |
| `seed`             | integer or `null` | `null`        | Shared forward-pass seed for `out_of_sample`, `historical`, and `external` schemes.                     |
| `inflow`           | object            | `in_sample`   | Sampling scheme for hydro inflow. Object with `"scheme"` key.                                           |
| `load`             | object            | `in_sample`   | Sampling scheme for bus load. Object with `"scheme"` key.                                               |
| `ncs`              | object            | `in_sample`   | Sampling scheme for NCS availability. Object with `"scheme"` key.                                       |
| `historical_years` | array or object   | auto-discover | Restrict the pool of historical windows. List (`[1940, 1953]`) or range (`{"from": 1940, "to": 2010}`). |

Example — simulate with out-of-sample inflow while training uses in-sample:

```json
{
  "training": {
    "forward_passes": 50,
    "stopping_rules": [{ "type": "iteration_limit", "limit": 200 }]
  },
  "simulation": {
    "enabled": true,
    "num_scenarios": 2000,
    "scenario_source": {
      "seed": 77,
      "inflow": { "scheme": "out_of_sample" },
      "load": { "scheme": "in_sample" },
      "ncs": { "scheme": "in_sample" }
    }
  }
}
```

---

## `modeling`

Controls physical modeling options.

| Field                   | Type   | Default   | Description                                            |
| ----------------------- | ------ | --------- | ------------------------------------------------------ |
| `inflow_non_negativity` | object | see below | Strategy for handling negative PAR model inflow draws. |

### `inflow_non_negativity`

| Field          | Type   | Default     | Description                                                                                                       |
| -------------- | ------ | ----------- | ----------------------------------------------------------------------------------------------------------------- |
| `method`       | string | `"penalty"` | One of `"none"`, `"penalty"`, `"truncation"`, or `"truncation_with_penalty"`.                                     |
| `penalty_cost` | float  | `1000.0`    | Penalty coefficient applied to negative inflow slack when `method` is `"penalty"` or `"truncation_with_penalty"`. |

- `"none"` -- no treatment; negative inflows are passed through to the LP.
- `"penalty"` -- adds a penalty variable to the LP that penalizes negative inflow
  draws at the specified cost per unit.
- `"truncation"` -- clamps negative PAR model draws to zero before applying noise.
- `"truncation_with_penalty"` -- combines both: clamps the inflow to zero and adds
  a bounded slack variable penalised at `penalty_cost`, providing a smooth backstop
  for extreme tail realisations.

Example:

```json
{
  "modeling": {
    "inflow_non_negativity": {
      "method": "penalty",
      "penalty_cost": 100.0
    }
  }
}
```

---

## `cut_selection`

Controls the cut management pipeline for managing cut pool growth. The
pipeline has up to two stages: strategy-based selection and budget
enforcement. Cut management periodically scans the cut pool and
deactivates cuts that are unlikely to improve the policy, reducing LP
size without sacrificing convergence quality. For a detailed explanation
of each stage, see
[Performance Accelerators](./performance-accelerators.md#cut-management-pipeline).

### Core Fields

| Field                    | Type    | Default | Description                                                                                                                                                                                                                                                                                                                             |
| ------------------------ | ------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`                | boolean | `false` | Enable cut pruning.                                                                                                                                                                                                                                                                                                                     |
| `method`                 | string  | --      | Selection method: `"level1"`, `"lml1"`, or `"domination"`.                                                                                                                                                                                                                                                                              |
| `threshold`              | integer | `0`     | Activity threshold for Level1: deactivate cuts with `active_count <= threshold`.                                                                                                                                                                                                                                                        |
| `memory_window`          | integer | `null`  | Sliding window for LML1: deactivate cuts not active within this many iterations. Overrides `threshold`.                                                                                                                                                                                                                                 |
| `domination_epsilon`     | float   | `1e-6`  | Tolerance for domination comparisons (Dominated method).                                                                                                                                                                                                                                                                                |
| `check_frequency`        | integer | `1`     | Iterations between pruning checks (Stage 1).                                                                                                                                                                                                                                                                                            |
| `cut_activity_tolerance` | float   | `1e-6`  | Minimum dual multiplier for a cut to count as binding.                                                                                                                                                                                                                                                                                  |
| `basis_activity_window`  | integer | `5`     | Sliding-window size (1-31 iterations) for tracking recent cut binding activity. Controls the warm-start classifier: cuts bound within the last `basis_activity_window` iterations are guessed tight on basis reconstruction. See [Performance Accelerators — Basis Reconstruction](./performance-accelerators.md#basis-reconstruction). |
| `max_active_per_stage`   | integer | `null`  | Hard cap on active cuts per stage (Stage 2 budget enforcement). `null` = no budget.                                                                                                                                                                                                                                                     |

**Methods:**

- `"level1"` -- deactivates cuts that have never been binding (cumulative
  binding count at or below `threshold`). Least aggressive; preserves
  convergence guarantee.
- `"lml1"` -- deactivates cuts that have not been binding within a sliding
  window of `memory_window` iterations (falls back to `threshold` if
  `memory_window` is not set).
- `"domination"` -- deactivates cuts that are dominated at every visited
  forward-pass trial point. Most aggressive; requires the visited-states
  archive (always collected during training). The `domination_epsilon`
  parameter controls the tolerance for domination comparisons.

Example with both pipeline stages:

```json
{
  "training": {
    "cut_selection": {
      "enabled": true,
      "method": "level1",
      "threshold": 0,
      "check_frequency": 5,
      "cut_activity_tolerance": 1e-6,
      "max_active_per_stage": 500
    }
  }
}
```

---

## `estimation`

Controls the PAR(p) model estimation pipeline. When the case provides
`inflow_history.parquet`, Cobre can automatically estimate AR coefficients
instead of requiring pre-computed `inflow_ar_coefficients.parquet`.

| Field                         | Type    | Default  | Description                                                                      |
| ----------------------------- | ------- | -------- | -------------------------------------------------------------------------------- |
| `max_order`                   | integer | `6`      | Maximum lag order considered during autoregressive model fitting.                |
| `order_selection`             | string  | `"pacf"` | Order selection criterion: `"pacf"` (PACF-based) or `"fixed"` (use `max_order`). |
| `min_observations_per_season` | integer | `30`     | Minimum observations per (entity, season) group to proceed with estimation.      |
| `max_coefficient_magnitude`   | float   | `null`   | Safety net: reduce to order 0 if any coefficient exceeds this magnitude.         |

Example:

```json
{
  "estimation": {
    "max_order": 6,
    "order_selection": "pacf",
    "min_observations_per_season": 30
  }
}
```

---

## `policy`

Controls policy persistence (checkpoint saving and warm-start loading).

| Field                    | Type                                     | Default      | Description                                                                                                                                 |
| ------------------------ | ---------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `path`                   | string                                   | `"./policy"` | Directory where policy data (cuts, states) is stored.                                                                                       |
| `mode`                   | `"fresh"`, `"warm_start"`, or `"resume"` | `"fresh"`    | Initialization mode. `"fresh"` starts from scratch; `"warm_start"` loads cuts from a previous run; `"resume"` continues an interrupted run. |
| `validate_compatibility` | boolean                                  | `true`       | When loading a policy, verify that entity counts, stage counts, and cut dimensions match the current system.                                |
| `boundary`               | object or null                           | `null`       | Terminal boundary cut configuration for coupling with an outer model's FCF. See below.                                                      |

### `checkpointing`

| Field                 | Type    | Default | Description                                     |
| --------------------- | ------- | ------- | ----------------------------------------------- |
| `enabled`             | boolean | `false` | Enable periodic checkpointing during training.  |
| `initial_iteration`   | integer | `null`  | First iteration to write a checkpoint.          |
| `interval_iterations` | integer | `null`  | Iterations between checkpoints.                 |
| `store_basis`         | boolean | `false` | Include LP basis in checkpoints for warm-start. |
| `compress`            | boolean | `false` | Compress checkpoint files.                      |

### `boundary`

Optional configuration for loading terminal-stage boundary cuts from a
different Cobre policy checkpoint. When present, the solver loads cuts from
the source checkpoint and injects them as fixed boundary conditions at the
terminal stage of the current study. The imported cuts are not updated by
training — they remain fixed throughout.

This enables Cobre-to-Cobre model coupling: a monthly study produces a
policy checkpoint, and a weekly+monthly DECOMP study loads that checkpoint's
cuts as its terminal-stage future cost function.

| Field          | Type    | Description                                                     |
| -------------- | ------- | --------------------------------------------------------------- |
| `path`         | string  | Path to the source policy checkpoint directory.                 |
| `source_stage` | integer | 0-based stage index in the source checkpoint to load cuts from. |

Example — load stage 2's cuts from a monthly policy as terminal boundary:

```json
{
  "policy": {
    "mode": "fresh",
    "boundary": {
      "path": "../monthly_study/policy",
      "source_stage": 2
    }
  }
}
```

See [Policy Management — Boundary Cuts](./policy-management.md#boundary-cuts)
for a full explanation of the coupling workflow.

---

## Temporal Resolution

Cobre does not have dedicated `config.json` fields for temporal resolution. The
resolution of each stage is determined entirely by the date boundaries in
`stages.json`. However, when `stages.json` defines stages at different temporal
resolutions — for example, four weekly stages within a month followed by monthly
stages, or monthly stages transitioning to quarterly stages — three mechanisms
activate automatically that users should understand.

### Noise Group Sharing

When multiple SDDP stages share the same `season_id` within the same calendar
period (for example, four weekly stages all assigned `season_id: 0` for
January), they receive identical PAR noise draws. This ensures that sub-monthly
stages present an inflow trajectory consistent with the monthly PAR model they
were fitted from, rather than fabricating independent weekly variability that
the historical record does not support.

### Observation Aggregation

When the study includes stages at different resolutions (for example, monthly
and quarterly), Cobre automatically aggregates fine-grained historical
observations into coarser season buckets before PAR fitting. A user supplying
monthly `inflow_history.parquet` for a study that includes quarterly stages does
not need to pre-aggregate the data; Cobre derives one observation per
(entity, season, year) at the appropriate coarser resolution. Aggregating in the
opposite direction (disaggregating coarser observations to a finer resolution)
is not supported and will produce a validation error at case load time.

### Lag Resolution Transition

For studies that transition from monthly to quarterly stages, the PAR lag state
changes resolution at the boundary. During the monthly phase, each monthly
inflow is accumulated into a ring buffer indexed by the downstream (quarterly)
lag. When the first quarterly stage is reached, the ring buffer contains a
complete set of duration-weighted monthly contributions, and the lag state is
rebuilt automatically. This transition is transparent to the LP and the cut
representation; it introduces no additional LP variables.

### Example: Weekly Stages Within a Month

The following `stages.json` excerpt shows four weekly stages within January
(stages 0-3, all with `season_id: 0`) followed by a normal monthly stage for
February (`season_id: 1`). Stages 0-3 share the same `season_id` and will
therefore receive identical PAR noise draws during training:

```json
[
  {
    "id": 0,
    "start_date": "2024-01-01",
    "end_date": "2024-01-08",
    "season_id": 0,
    "num_scenarios": 50
  },
  {
    "id": 1,
    "start_date": "2024-01-08",
    "end_date": "2024-01-15",
    "season_id": 0,
    "num_scenarios": 50
  },
  {
    "id": 2,
    "start_date": "2024-01-15",
    "end_date": "2024-01-22",
    "season_id": 0,
    "num_scenarios": 50
  },
  {
    "id": 3,
    "start_date": "2024-01-22",
    "end_date": "2024-02-01",
    "season_id": 0,
    "num_scenarios": 50
  },
  {
    "id": 4,
    "start_date": "2024-02-01",
    "end_date": "2024-03-01",
    "season_id": 1,
    "num_scenarios": 50
  }
]
```

### Recommended Alternative: Weekly Blocks Within a Monthly Stage

When weekly dispatch granularity is needed but true weekly-resolution noise
data is unavailable, the recommended approach is to use a single monthly SDDP
stage with chronological blocks rather than four separate weekly SDDP stages.
This provides weekly LP granularity while keeping one noise realization per
month — consistent with the data resolution — and avoids the lag-accumulation
complications that arise with multiple independent weekly stages. See
[Stochastic Modeling — Temporal Resolution and PAR](./stochastic-modeling.md#temporal-resolution-and-par)
for the full explanation and a `stages.json` example of the block pattern.

### See Also (Temporal Resolution)

- [Stochastic Modeling — Multi-Resolution Studies](./stochastic-modeling.md#multi-resolution-studies) — detailed mechanism descriptions including the noise group precomputation algorithm, observation aggregation internals, and lag ring buffer design
- [Stochastic Modeling — Temporal Resolution and PAR](./stochastic-modeling.md#temporal-resolution-and-par) — the honest representation principle, the recommended weekly-block pattern, and validation rules 27-31

---

## `exports`

Controls which outputs are written to the results directory.

| Field             | Type                           | Default | Description                                                                     |
| ----------------- | ------------------------------ | ------- | ------------------------------------------------------------------------------- |
| `training`        | boolean                        | `true`  | Write training convergence data (Parquet).                                      |
| `cuts`            | boolean                        | `true`  | Write the cut pool (FlatBuffers).                                               |
| `states`          | boolean                        | `false` | Write visited forward-pass trial points to the policy checkpoint (FlatBuffers). |
| `vertices`        | boolean                        | `true`  | Write inner approximation vertices when applicable (Parquet).                   |
| `simulation`      | boolean                        | `true`  | Write per-entity simulation results (Parquet).                                  |
| `forward_detail`  | boolean                        | `false` | Write per-scenario forward-pass detail (large; disabled by default).            |
| `backward_detail` | boolean                        | `false` | Write per-scenario backward-pass detail (large; disabled by default).           |
| `compression`     | `"zstd"`, `"lz4"`, or `"none"` | `null`  | Output Parquet compression algorithm. `null` uses the crate default (zstd).     |
| `stochastic`      | boolean                        | `false` | Export stochastic preprocessing artifacts to `output/stochastic/`.              |

---

## Full Example

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/config.schema.json",
  "training": {
    "tree_seed": 42,
    "forward_passes": 50,
    "stopping_rules": [
      { "type": "iteration_limit", "limit": 200 },
      { "type": "bound_stalling", "iterations": 20, "tolerance": 0.0001 }
    ],
    "stopping_mode": "any",
    "scenario_source": {
      "seed": 99,
      "inflow": { "scheme": "out_of_sample" },
      "load": { "scheme": "in_sample" },
      "ncs": { "scheme": "in_sample" }
    },
    "cut_selection": {
      "enabled": true,
      "method": "level1",
      "threshold": 0,
      "check_frequency": 5,
      "cut_activity_tolerance": 1e-6,
      "max_active_per_stage": null
    }
  },
  "modeling": {
    "inflow_non_negativity": {
      "method": "penalty",
      "penalty_cost": 1000.0
    }
  },
  "simulation": {
    "enabled": true,
    "num_scenarios": 2000
  },
  "policy": {
    "path": "./policy",
    "mode": "fresh"
  },
  "exports": {
    "training": true,
    "cuts": true,
    "states": false,
    "simulation": true,
    "compression": "zstd"
  }
}
```

---

## Advanced Fields

The `Config` struct supports additional sections not documented on this page.
These fields are deserialized from `config.json` when present but are intended
for advanced use cases and may change between releases:

| Section                           | Purpose                                                                                                  |
| --------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `upper_bound_evaluation`          | Inner approximation upper-bound evaluation settings                                                      |
| `training.cut_formulation`        | Cut formulation variant (single-cut or multi-cut)                                                        |
| `training.forward_pass.pass_type` | Forward pass strategy selection                                                                          |
| `training.solver`                 | LP solver options (see [Solver Safeguards](./performance-accelerators.md#solver-safeguards) for details) |
| `simulation.io_channel_capacity`  | Async I/O channel buffer size for simulation output writing                                              |

All fields have defaults and can be omitted. For the complete list of fields
and their types, see the `Config` struct in the
[cobre-io API docs](https://docs.rs/cobre-io).

---

## See Also

- [Case Directory Format](../reference/case-format.md) — full schema for all input files
- [Running Studies](./running-studies.md) — end-to-end workflow guide
- [Error Codes](../reference/error-codes.md) — validation errors including `SchemaError` for config fields
- [Stochastic Modeling — Temporal Resolution](./stochastic-modeling.md#temporal-resolution-and-par) — how stage resolution affects PAR noise sharing and validation rules
