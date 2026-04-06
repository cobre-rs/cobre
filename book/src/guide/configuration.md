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

| Field           | Type               | Default | Description                                                                                                                                                                             |
| --------------- | ------------------ | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`       | boolean            | `true`  | Set to `false` to skip training and proceed directly to simulation (requires a pre-trained policy).                                                                                     |
| `tree_seed`     | integer            | `null`  | Random seed for the opening scenario tree. When `null`, derived from OS entropy (non-reproducible). See [Stochastic Modeling](./stochastic-modeling.md) for the dual-seed architecture. |
| `stopping_mode` | `"any"` or `"all"` | `"any"` | How multiple stopping rules combine: `"any"` stops when the first rule is satisfied; `"all"` requires all rules to be satisfied simultaneously.                                         |

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

Controls the cut selection strategy for managing cut pool growth. Cut
selection periodically scans the cut pool and deactivates cuts that are
unlikely to improve the policy, reducing LP size without sacrificing
convergence quality.

| Field                    | Type    | Default | Description                                                                       |
| ------------------------ | ------- | ------- | --------------------------------------------------------------------------------- |
| `enabled`                | boolean | `false` | Enable cut pruning.                                                               |
| `method`                 | string  | --      | Selection method: `"level1"`, `"lml1"`, or `"domination"`.                        |
| `threshold`              | integer | `0`     | Activity threshold. For Level1: deactivate cuts with `active_count <= threshold`. |
| `check_frequency`        | integer | `1`     | Iterations between pruning checks.                                                |
| `cut_activity_tolerance` | float   | `1e-6`  | Minimum dual multiplier for a cut to count as binding.                            |

**Methods:**

- `"level1"` -- deactivates cuts that have never been binding (cumulative
  binding count at or below `threshold`). Least aggressive; preserves
  convergence guarantee.
- `"lml1"` -- deactivates cuts that have not been binding within a sliding
  window of `threshold` iterations.
- `"domination"` -- deactivates cuts that are dominated at every visited
  forward-pass trial point. Most aggressive; requires the visited-states
  archive (always collected during training). The `threshold` parameter
  controls how many consecutive domination checks a cut must fail before
  deactivation.

Example:

```json
{
  "training": {
    "cut_selection": {
      "enabled": true,
      "method": "level1",
      "threshold": 0,
      "check_frequency": 5,
      "cut_activity_tolerance": 1e-6
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

### `checkpointing`

| Field                 | Type    | Default | Description                                     |
| --------------------- | ------- | ------- | ----------------------------------------------- |
| `enabled`             | boolean | `false` | Enable periodic checkpointing during training.  |
| `initial_iteration`   | integer | `null`  | First iteration to write a checkpoint.          |
| `interval_iterations` | integer | `null`  | Iterations between checkpoints.                 |
| `store_basis`         | boolean | `false` | Include LP basis in checkpoints for warm-start. |
| `compress`            | boolean | `false` | Compress checkpoint files.                      |

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
      "cut_activity_tolerance": 1e-6
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

| Section                           | Purpose                                                     |
| --------------------------------- | ----------------------------------------------------------- |
| `upper_bound_evaluation`          | Inner approximation upper-bound evaluation settings         |
| `training.cut_formulation`        | Cut formulation variant (single-cut or multi-cut)           |
| `training.forward_pass.pass_type` | Forward pass strategy selection                             |
| `training.solver`                 | LP solver retry budget and attempt limits                   |
| `simulation.sampling_scheme`      | Simulation scenario sampling strategy                       |
| `simulation.io_channel_capacity`  | Async I/O channel buffer size for simulation output writing |

All fields have defaults and can be omitted. For the complete list of fields
and their types, see the `Config` struct in the
[cobre-io API docs](https://docs.rs/cobre-io).

---

## See Also

- [Case Directory Format](../reference/case-format.md) — full schema for all input files
- [Running Studies](./running-studies.md) — end-to-end workflow guide
- [Error Codes](../reference/error-codes.md) — validation errors including `SchemaError` for config fields
