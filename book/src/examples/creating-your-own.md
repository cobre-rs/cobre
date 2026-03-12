# Creating Your Own Case

This page explains how to create a Cobre case directory from scratch, without
using `cobre init`. It lists the minimum required files, the optional files, the
`$schema` URL pattern for editor validation, and the exact steps to go from an
empty directory to a validated, runnable study.

If you prefer to start from a working template and modify it, use:

```bash
cobre init --template 1dtoy my_study
```

For a field-by-field explanation of each file, see
[Anatomy of a Case](../tutorial/anatomy-of-a-case.md) and the
[Case Format Reference](../reference/case-format.md).

---

## Minimum Required Files

A Cobre case directory requires exactly these files to pass validation:

```
my_case/
  config.json               # Solver configuration (required)
  penalties.json            # Global penalty defaults (required)
  stages.json               # Stage sequence and policy graph (required)
  initial_conditions.json   # Reservoir storage at study start (required)
  system/
    buses.json              # Electrical bus registry (required)
    lines.json              # Transmission line registry (required, may be empty)
    hydros.json             # Hydro plant registry (required, may be empty)
    thermals.json           # Thermal plant registry (required, may be empty)
```

All eight files must be present. `lines.json`, `hydros.json`, and `thermals.json`
may contain empty arrays (`"lines": []`, `"hydros": []`, `"thermals": []`), but
the files themselves must exist. A case with no hydro plants and no thermals will
fail physically — there is nothing to dispatch — but it will pass schema validation
and is useful for testing the load pipeline.

---

## Optional Files

The following files extend the case with additional data. The validator reads each
one if it exists and ignores it if it does not:

| File                                       | Purpose                                              |
| ------------------------------------------ | ---------------------------------------------------- |
| `scenarios/inflow_seasonal_stats.parquet`  | PAR(p) seasonal statistics for hydro inflow modeling |
| `scenarios/load_seasonal_stats.parquet`    | PAR(p) seasonal statistics for bus load modeling     |
| `scenarios/inflow_ar_coefficients.parquet` | Autoregressive lag coefficients for PAR(p) inflow    |
| `scenarios/inflow_history.parquet`         | Historical inflow series for model calibration       |
| `scenarios/load_factors.json`              | Stage-varying load scaling factors                   |
| `scenarios/correlation.json`               | Cross-series correlation structure                   |
| `system/non_controllable_sources.json`     | Wind and solar generators                            |
| `system/pumping_stations.json`             | Pumped-storage facilities                            |
| `system/energy_contracts.json`             | Bilateral energy contracts                           |
| `constraints/thermal_bounds.parquet`       | Stage-varying thermal generation bounds              |
| `constraints/hydro_bounds.parquet`         | Stage-varying hydro dispatch bounds                  |

When the `scenarios/` files are absent, Cobre generates white-noise inflow and load
scenarios using only the stage mean and standard deviation values from `stages.json`
(if those fields are present) or generates zero-uncertainty scenarios. For
stochastic studies, supply the `inflow_seasonal_stats.parquet` and
`load_seasonal_stats.parquet` files.

---

## Editor Validation with `$schema`

Every Cobre JSON file supports the `$schema` field. When present, editors that
understand JSON Schema (VS Code with the JSON Language Features extension, Neovim
with `jsonls`, JetBrains IDEs) use the schema to provide autocompletion and
inline error highlighting.

The URL pattern is:

```
https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/<filename>.schema.json
```

The available schema files are:

| File                      | Schema URL                                                                |
| ------------------------- | ------------------------------------------------------------------------- |
| `config.json`             | `https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/config.schema.json`             |
| `penalties.json`          | `https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/penalties.schema.json`          |
| `stages.json`             | `https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/stages.schema.json`             |
| `initial_conditions.json` | `https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/initial_conditions.schema.json` |
| `system/buses.json`       | `https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/buses.schema.json`              |
| `system/lines.json`       | `https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/lines.schema.json`              |
| `system/hydros.json`      | `https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/hydros.schema.json`             |
| `system/thermals.json`    | `https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/thermals.schema.json`           |

Add the `$schema` field as the first key in each file to activate editor support:

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/config.schema.json",
  "training": { ... }
}
```

For the complete list of schema URLs, see [Schemas](../reference/schemas.md).

---

## Step-by-Step: A Minimal 1-Bus, 1-Thermal Case

This walkthrough creates the smallest possible runnable case: one bus, one thermal
plant, no hydro, four monthly stages, and deterministic load (zero standard
deviation). Run these steps from your terminal.

### Step 1: Create the directory

```bash
mkdir my_case
cd my_case
mkdir system
```

### Step 2: Write `config.json`

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/config.schema.json",
  "training": {
    "forward_passes": 1,
    "stopping_rules": [{ "type": "iteration_limit", "limit": 50 }]
  }
}
```

The `simulation` block is omitted, so no post-training simulation runs. Add it
when your case is working and you want dispatch results.

### Step 3: Write `stages.json`

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/stages.schema.json",
  "policy_graph": {
    "type": "finite_horizon",
    "annual_discount_rate": 0.0
  },
  "stages": [
    {
      "id": 0,
      "start_date": "2024-01-01",
      "end_date": "2024-02-01",
      "blocks": [{ "id": 0, "name": "SINGLE", "hours": 744 }],
      "num_scenarios": 5
    },
    {
      "id": 1,
      "start_date": "2024-02-01",
      "end_date": "2024-03-01",
      "blocks": [{ "id": 0, "name": "SINGLE", "hours": 696 }],
      "num_scenarios": 5
    },
    {
      "id": 2,
      "start_date": "2024-03-01",
      "end_date": "2024-04-01",
      "blocks": [{ "id": 0, "name": "SINGLE", "hours": 744 }],
      "num_scenarios": 5
    },
    {
      "id": 3,
      "start_date": "2024-04-01",
      "end_date": "2024-05-01",
      "blocks": [{ "id": 0, "name": "SINGLE", "hours": 720 }],
      "num_scenarios": 5
    }
  ]
}
```

`annual_discount_rate: 0.0` disables discounting, keeping costs in nominal terms.
`num_scenarios: 5` draws 5 scenario trajectories per iteration during training.

### Step 4: Write `penalties.json`

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/penalties.schema.json",
  "bus": {
    "deficit_segments": [{ "depth_mw": null, "cost": 3000.0 }],
    "excess_cost": 100.0
  },
  "line": {
    "exchange_cost": 2.0
  },
  "hydro": {
    "spillage_cost": 0.01,
    "fpha_turbined_cost": 0.05,
    "diversion_cost": 0.1,
    "storage_violation_below_cost": 10000.0,
    "filling_target_violation_cost": 50000.0,
    "turbined_violation_below_cost": 500.0,
    "outflow_violation_below_cost": 500.0,
    "outflow_violation_above_cost": 500.0,
    "generation_violation_below_cost": 1000.0,
    "evaporation_violation_cost": 5000.0,
    "water_withdrawal_violation_cost": 1000.0
  },
  "non_controllable_source": {
    "curtailment_cost": 0.005
  }
}
```

All `hydro` and `non_controllable_source` penalty fields are required by the
schema even if your case has no hydro plants or non-controllable sources. Copy
the values above verbatim; they only take effect when those element types exist.

### Step 5: Write `initial_conditions.json`

```json
{
  "storage": [],
  "filling_storage": []
}
```

Both arrays are empty because this case has no hydro plants. The file must still
be present.

### Step 6: Write `system/buses.json`

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/buses.schema.json",
  "buses": [
    {
      "id": 0,
      "name": "GRID"
    }
  ]
}
```

A bus with no `deficit_segments` block inherits the global defaults from
`penalties.json`. Add `"deficit_segments"` inside the bus object to override them
for this bus only.

### Step 7: Write `system/lines.json`

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/lines.schema.json",
  "lines": []
}
```

An empty lines file is required. A single-bus case never needs lines.

### Step 8: Write `system/hydros.json`

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/hydros.schema.json",
  "hydros": []
}
```

### Step 9: Write `system/thermals.json`

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/thermals.schema.json",
  "thermals": [
    {
      "id": 0,
      "name": "PLANT1",
      "bus_id": 0,
      "cost_segments": [{ "capacity_mw": 100.0, "cost_per_mwh": 20.0 }],
      "generation": {
        "min_mw": 0.0,
        "max_mw": 100.0
      }
    }
  ]
}
```

One thermal plant with 100 MW capacity at 20 $/MWh. The `bus_id: 0` connects it
to the `GRID` bus defined in `buses.json`. IDs must match across files — if you
define a thermal with `bus_id: 1` but no bus with `id: 1` exists, validation
will fail with a referential integrity error.

### Step 10: Validate

```bash
cobre validate my_case
```

A clean case prints a validation summary with no errors. If `cobre validate`
reports errors, read the error message carefully — it includes the file name,
the field path, and a description of what is wrong.

Common validation errors on a new case:

| Error message                                    | Cause                                                     |
| ------------------------------------------------ | --------------------------------------------------------- |
| `missing required file: system/lines.json`       | The file does not exist; create it with an empty array    |
| `hydro_id 0 not found in registry`               | `initial_conditions.json` references a non-existent plant |
| `bus_id 1 does not exist`                        | A generator references a bus that is not in `buses.json`  |
| `stopping_rules must contain at least one entry` | The `stopping_rules` array in `config.json` is empty      |

### Step 11: Run

```bash
cobre run my_case --output my_case/output
```

The output directory is created automatically. The solver prints a progress bar
to stderr during training and a summary when complete.

---

## Adding Stochastic Load

The minimal case above runs with deterministic (zero-variance) scenarios because
no `scenarios/` files are present. To add stochastic load, create
`scenarios/load_seasonal_stats.parquet` with one row per (bus, stage) pair.

The file must contain these columns:

| Column     | Type   | Description                                           |
| ---------- | ------ | ----------------------------------------------------- |
| `bus_id`   | INT32  | Bus identifier (matches `id` in `buses.json`)         |
| `stage_id` | INT32  | Stage identifier (matches `id` in `stages.json`)      |
| `mean_mw`  | DOUBLE | Seasonal mean load in MW                              |
| `std_mw`   | DOUBLE | Seasonal standard deviation in MW (0 = deterministic) |
| `ar_order` | INT32  | Number of AR lags (0 = white noise, no correlation)   |

For a 1-bus, 4-stage case with a mean load of 60 MW and 10% standard deviation:

```python
import polars as pl

df = pl.DataFrame({
    "bus_id":   [0, 0, 0, 0],
    "stage_id": [0, 1, 2, 3],
    "mean_mw":  [60.0, 60.0, 60.0, 60.0],
    "std_mw":   [6.0,  6.0,  6.0,  6.0],
    "ar_order": [0, 0, 0, 0],
})
df.write_parquet("my_case/scenarios/load_seasonal_stats.parquet")
```

```bash
mkdir -p my_case/scenarios
# run the Python script above, then validate and run:
cobre validate my_case
cobre run my_case --output my_case/output
```

For the inflow stochastic model, create `scenarios/inflow_seasonal_stats.parquet`
with the same structure but using `hydro_id` instead of `bus_id` and
`mean_m3s` / `std_m3s` instead of `mean_mw` / `std_mw`.

---

## Where to Go Next

- [Case Format Reference](../reference/case-format.md) — complete field-by-field schema for every file
- [Configuration](../guide/configuration.md) — all `config.json` options including convergence rules and warm-start
- [1dtoy Example](./1dtoy.md) — annotated walkthrough of a complete working case
- [Understanding Results](../tutorial/understanding-results.md) — how to interpret the output directory
