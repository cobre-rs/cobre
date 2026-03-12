# Configuration

All runtime parameters for `cobre run` are controlled by `config.json` in the
case directory. This page documents every section and field.

---

## Minimal Config

```json
{
  "version": "2.0.0",
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

| Field           | Type               | Default | Description                                                                                                                                     |
| --------------- | ------------------ | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`       | boolean            | `true`  | Set to `false` to skip training and proceed directly to simulation (requires a pre-trained policy).                                             |
| `seed`          | integer            | `42`    | Random seed for reproducible scenario generation.                                                                                               |
| `stopping_mode` | `"any"` or `"all"` | `"any"` | How multiple stopping rules combine: `"any"` stops when the first rule is satisfied; `"all"` requires all rules to be satisfied simultaneously. |

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
phase is skipped regardless of the `--skip-simulation` flag.

Example:

```json
{
  "simulation": {
    "enabled": true,
    "num_scenarios": 1000
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

---

## `exports`

Controls which outputs are written to the results directory.

| Field             | Type                           | Default | Description                                                                 |
| ----------------- | ------------------------------ | ------- | --------------------------------------------------------------------------- |
| `training`        | boolean                        | `true`  | Write training convergence data (Parquet).                                  |
| `cuts`            | boolean                        | `true`  | Write the cut pool (FlatBuffers).                                           |
| `states`          | boolean                        | `true`  | Write visited state vectors (Parquet).                                      |
| `vertices`        | boolean                        | `true`  | Write inner approximation vertices when applicable (Parquet).               |
| `simulation`      | boolean                        | `true`  | Write per-entity simulation results (Parquet).                              |
| `forward_detail`  | boolean                        | `false` | Write per-scenario forward-pass detail (large; disabled by default).        |
| `backward_detail` | boolean                        | `false` | Write per-scenario backward-pass detail (large; disabled by default).       |
| `compression`     | `"zstd"`, `"lz4"`, or `"none"` | `null`  | Output Parquet compression algorithm. `null` uses the crate default (zstd). |

---

## Full Example

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/config.schema.json",
  "version": "2.0.0",
  "training": {
    "seed": 42,
    "forward_passes": 50,
    "stopping_rules": [
      { "type": "iteration_limit", "limit": 200 },
      { "type": "bound_stalling", "iterations": 20, "tolerance": 0.0001 }
    ],
    "stopping_mode": "any"
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
    "states": true,
    "simulation": true,
    "compression": "zstd"
  }
}
```

---

## See Also

- [Case Directory Format](../reference/case-format.md) — full schema for all input files
- [Running Studies](./running-studies.md) — end-to-end workflow guide
- [Error Codes](../reference/error-codes.md) — validation errors including `SchemaError` for config fields
