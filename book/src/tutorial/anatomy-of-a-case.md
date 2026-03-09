# Anatomy of a Case

A Cobre case directory is a self-contained folder of input files. When you run
`cobre run` or `cobre validate`, the first thing Cobre does is call `load_case`
on that directory. `load_case` reads every file, runs the five-layer validation
pipeline (schema, references, physical feasibility, stochastic consistency, solver
feasibility), and produces a fully-validated `System` object ready for the solver.

This page walks through every file in the `1dtoy` example, explaining what each
field controls and why it matters. The example lives in `examples/1dtoy/` in the
repository and is also available via `cobre init --template 1dtoy`.

For the complete field-by-field schema reference, see
[Case Format Reference](../reference/case-format.md).

---

## Directory Structure

The `1dtoy` case contains 10 input files across three directories:

```
1dtoy/
  config.json
  initial_conditions.json
  penalties.json
  stages.json
  system/
    buses.json
    hydros.json
    lines.json
    thermals.json
  scenarios/
    inflow_seasonal_stats.parquet
    load_seasonal_stats.parquet
```

The four root-level files configure the solver and define the time horizon. The
`system/` subdirectory holds the power system entities. The `scenarios/`
subdirectory holds the stochastic input data that drives scenario generation.

---

## Root-Level Files

### `config.json`

`config.json` controls all solver parameters: how many training iterations to run,
when to stop, whether to follow training with a simulation pass, and more.

```json
{
  "version": "1.0.0",
  "training": {
    "forward_passes": 1,
    "stopping_rules": [
      {
        "type": "iteration_limit",
        "limit": 128
      }
    ]
  },
  "simulation": {
    "enabled": true,
    "num_scenarios": 100
  }
}
```

`version` is an informational string; it does not affect behavior.

The `training` section is mandatory. `forward_passes: 1` means each training
iteration draws one scenario trajectory. The `stopping_rules` array must contain
at least one `iteration_limit` rule. Here the solver stops after 128 iterations.
For production studies you would typically also add a convergence-based stopping
rule such as `bound_stalling`, but for a small tutorial case an iteration limit
is sufficient.

The `simulation` section is optional and defaults to disabled. Here it is enabled
with 100 scenarios. After training completes, Cobre evaluates the trained policy
over 100 independently sampled scenarios and writes the results to the output
directory.

For the full list of configuration options, see
[Configuration](../guide/configuration.md).

---

### `penalties.json`

`penalties.json` defines the global penalty cost defaults. These costs are added
to the LP objective whenever a physical constraint is violated in a soft-constraint
sense — for example, when demand cannot be fully served (deficit) or when a
reservoir bound is violated. Setting these costs high relative to actual generation
costs ensures that violations are used as a last resort rather than a cheap
dispatch option.

```json
{
  "bus": {
    "deficit_segments": [
      {
        "depth_mw": 500.0,
        "cost": 1000.0
      },
      {
        "depth_mw": null,
        "cost": 5000.0
      }
    ],
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

The `bus.deficit_segments` array defines a piecewise-linear deficit cost curve.
The first segment covers the first 500 MW of unserved energy at 1000 $/MWh.
Beyond 500 MW, the cost rises to 5000 $/MWh (the segment with `depth_mw: null`
is always the final unbounded tier). The two-tier structure mimics a typical
Value of Lost Load model where the first tranche represents interruptible load
and the second represents non-interruptible load. `excess_cost` penalizes
over-injection at 100 $/MWh.

Hydro penalty costs cover a range of operational constraint violations. The low
`spillage_cost` (0.01 $/hm3) makes spillage the cheapest way to release water
when turbine capacity is exhausted. The high `storage_violation_below_cost`
(10,000 $/hm3) and `filling_target_violation_cost` (50,000 $/hm3) make reservoir
bound violations extremely costly, ensuring the solver strongly avoids them.

Individual entities can override these global defaults in their own JSON files
using a `penalties` block. The reference page documents all override options.

---

### `stages.json`

`stages.json` defines the temporal structure of the study: the sequence of
planning stages, the load blocks within each stage, the number of scenarios to
sample at each stage during training, and the policy graph horizon type.

```json
{
  "policy_graph": {
    "type": "finite_horizon",
    "annual_discount_rate": 0.12
  },
  "stages": [
    {
      "id": 0,
      "start_date": "2024-01-01",
      "end_date": "2024-02-01",
      "blocks": [
        {
          "id": 0,
          "name": "SINGLE",
          "hours": 744
        }
      ],
      "num_scenarios": 10
    },
    {
      "id": 1,
      "start_date": "2024-02-01",
      "end_date": "2024-03-01",
      "blocks": [
        {
          "id": 0,
          "name": "SINGLE",
          "hours": 696
        }
      ],
      "num_scenarios": 10
    },
    {
      "id": 2,
      "start_date": "2024-03-01",
      "end_date": "2024-04-01",
      "blocks": [
        {
          "id": 0,
          "name": "SINGLE",
          "hours": 744
        }
      ],
      "num_scenarios": 10
    },
    {
      "id": 3,
      "start_date": "2024-04-01",
      "end_date": "2024-05-01",
      "blocks": [
        {
          "id": 0,
          "name": "SINGLE",
          "hours": 720
        }
      ],
      "num_scenarios": 10
    }
  ]
}
```

`policy_graph.type: "finite_horizon"` means the planning horizon is a linear
sequence of stages with no cyclic structure and zero terminal value after the
last stage. The `annual_discount_rate: 0.12` applies a 12% annual discount to
future stage costs.

The `stages` array defines four monthly stages covering January through April 2024.
Each stage has a single load block named `SINGLE` that spans the entire month. The
`hours` values match the actual number of hours in each calendar month (744 for
January, 696 for February in 2024, and so on). These hours are used when converting
power (MW) to energy (MWh) in the LP objective.

`num_scenarios: 10` means 10 scenario trajectories are sampled at each stage during
training forward passes. A small number like 10 is sufficient for a tutorial; real
studies typically use 50 or more.

---

### `initial_conditions.json`

`initial_conditions.json` provides the reservoir storage levels at the beginning
of the study. Every hydro plant that participates in the study must have an entry
here.

```json
{
  "storage": [
    {
      "hydro_id": 0,
      "value_hm3": 83.222
    }
  ],
  "filling_storage": []
}
```

`storage` covers operating reservoirs: plants that both generate power and store
water between stages. `hydro_id: 0` corresponds to `UHE1` defined in
`system/hydros.json`. The initial storage is 83.222 hm³, which is about 8.3% of
the 1000 hm³ maximum capacity — a low-storage starting condition that forces the
solver to balance generation against the risk of running dry.

`filling_storage` covers filling reservoirs — reservoirs that do not generate power
but feed downstream plants. The `1dtoy` case has no filling reservoirs, so this
array is empty. It must still be present (even if empty) to satisfy the schema.

---

## `system/` Files

### `system/buses.json`

Buses are the nodes of the electrical network. Every generator and load is
connected to a bus. The bus balance constraint ensures that injections equal
withdrawals at every bus in every LP solve.

```json
{
  "buses": [
    {
      "id": 0,
      "name": "SIN",
      "deficit_segments": [
        {
          "depth_mw": null,
          "cost": 1000.0
        }
      ]
    }
  ]
}
```

The `1dtoy` case has a single bus named `SIN` (Sistema Interligado Nacional,
the Brazilian interconnected system). A single-bus model treats the entire system
as one copper-plate node: there are no transmission constraints.

The bus-level `deficit_segments` here overrides the global default from
`penalties.json` with a simpler single-tier structure: unlimited deficit at
1000 $/MWh. When an entity-level override is present, it takes precedence over
the global default.

---

### `system/lines.json`

Transmission lines connect pairs of buses and carry power flows subject to
capacity limits. In a single-bus model, no lines are needed.

```json
{
  "lines": []
}
```

The file must be present even if the `lines` array is empty. The validator
checks for the file and would raise a schema error if it were absent.

---

### `system/hydros.json`

Hydro plants have a reservoir (water storage), a turbine (converts water flow to
electricity), and optional cascade linkage to downstream plants.

```json
{
  "hydros": [
    {
      "id": 0,
      "name": "UHE1",
      "bus_id": 0,
      "downstream_id": null,
      "reservoir": {
        "min_storage_hm3": 0.0,
        "max_storage_hm3": 1000.0
      },
      "outflow": {
        "min_outflow_m3s": 0.0,
        "max_outflow_m3s": 50.0
      },
      "generation": {
        "model": "constant_productivity",
        "productivity_mw_per_m3s": 1.0,
        "min_turbined_m3s": 0.0,
        "max_turbined_m3s": 50.0,
        "min_generation_mw": 0.0,
        "max_generation_mw": 50.0
      }
    }
  ]
}
```

`UHE1` connects to bus 0 (SIN). `downstream_id: null` means it is a tailwater
plant — there is no plant downstream that receives its outflow.

The `reservoir` block defines storage bounds in hm³ (cubic hectometres). UHE1
can hold between 0 and 1000 hm³. The minimum of 0 means the reservoir can be
fully emptied, which is common for run-of-river-adjacent plants.

The `outflow` block limits total outflow (turbined + spilled) to 50 m³/s maximum.
This is a physical constraint representing the river channel capacity below the dam.

The `generation` block uses `"constant_productivity"`, the simplest turbine model:
generation (MW) equals turbined flow (m³/s) times the `productivity_mw_per_m3s`
factor. Here the factor is 1.0, so 1 m³/s of turbined flow yields 1 MW. The
turbine can pass between 0 and 50 m³/s, and the resulting generation is bounded
between 0 and 50 MW.

---

### `system/thermals.json`

Thermal plants are dispatchable generators with a fixed cost per MWh. The
piecewise cost structure allows modeling fuel cost curves by defining multiple
capacity segments at increasing costs.

```json
{
  "thermals": [
    {
      "id": 0,
      "name": "UTE1",
      "bus_id": 0,
      "cost_segments": [
        {
          "capacity_mw": 15.0,
          "cost_per_mwh": 5.0
        }
      ],
      "generation": {
        "min_mw": 0.0,
        "max_mw": 15.0
      }
    },
    {
      "id": 1,
      "name": "UTE2",
      "bus_id": 0,
      "cost_segments": [
        {
          "capacity_mw": 15.0,
          "cost_per_mwh": 10.0
        }
      ],
      "generation": {
        "min_mw": 0.0,
        "max_mw": 15.0
      }
    }
  ]
}
```

Both thermal plants connect to bus 0. `UTE1` is the cheaper unit at 5 $/MWh and
`UTE2` costs 10 $/MWh. Both are limited to 15 MW maximum dispatch. In the LP,
Cobre will always prefer UTE1 over UTE2 and prefer both over deficit (1000 $/MWh),
creating a natural merit-order dispatch.

Each thermal has a single cost segment covering its entire capacity. For plants
with variable heat rates you would add additional segments — for example,
`{ "capacity_mw": 10.0, "cost_per_mwh": 8.0 }` followed by
`{ "capacity_mw": 5.0, "cost_per_mwh": 12.0 }` to model a plant that becomes
progressively more expensive at higher output.

---

## `scenarios/` Files

The `scenarios/` directory holds Parquet files that parameterize the stochastic
models used to generate inflow and load scenarios during training and simulation.
Unlike the JSON files, these are binary columnar files that cannot be inspected
with a text editor.

### `scenarios/inflow_seasonal_stats.parquet`

This file contains the seasonal mean and standard deviation of historical inflows
for each (hydro plant, stage) pair, plus the autoregressive order for the PAR(p)
model. Cobre uses these statistics to fit a periodic autoregressive model that
generates correlated inflow scenarios across stages.

Expected columns:

| Column     | Type   | Description                                             |
| ---------- | ------ | ------------------------------------------------------- |
| `hydro_id` | INT32  | Hydro plant identifier (matches `id` in `hydros.json`)  |
| `stage_id` | INT32  | Stage identifier (matches `id` in `stages.json`)        |
| `mean_m3s` | DOUBLE | Seasonal mean inflow in m³/s                            |
| `std_m3s`  | DOUBLE | Seasonal standard deviation in m³/s (must be >= 0)      |
| `ar_order` | INT32  | Number of AR lags in the PAR(p) model (0 = white noise) |

The 1dtoy file has 4 rows, one for each stage, for the single hydro plant `UHE1`
(hydro_id = 0). When `ar_order > 0`, Cobre also looks for an
`inflow_ar_coefficients.parquet` file containing the lag coefficients. The 1dtoy
case uses `ar_order = 0` (white noise), so no coefficients file is needed.

To inspect a Parquet file on your machine, use any of:

```python
import polars as pl
df = pl.read_parquet("scenarios/inflow_seasonal_stats.parquet")
print(df)
```

```python
import pandas as pd
df = pd.read_parquet("scenarios/inflow_seasonal_stats.parquet")
print(df)
```

```sql
-- DuckDB
SELECT * FROM read_parquet('scenarios/inflow_seasonal_stats.parquet');
```

### `scenarios/load_seasonal_stats.parquet`

This file contains the seasonal statistics for electrical load at each bus. It
drives the stochastic load model that generates demand scenarios during training
and simulation.

Expected columns:

| Column     | Type   | Description                                             |
| ---------- | ------ | ------------------------------------------------------- |
| `bus_id`   | INT32  | Bus identifier (matches `id` in `buses.json`)           |
| `stage_id` | INT32  | Stage identifier (matches `id` in `stages.json`)        |
| `mean_mw`  | DOUBLE | Seasonal mean load in MW                                |
| `std_mw`   | DOUBLE | Seasonal standard deviation in MW (must be >= 0)        |
| `ar_order` | INT32  | Number of AR lags in the PAR(p) model (0 = white noise) |

The 1dtoy file has 4 rows, one for each stage, for the single bus `SIN`
(bus_id = 0). The load mean and standard deviation determine how much demand
the system must serve in each scenario and how uncertain that demand is.

---

## What's Next

Now that you understand what each file does, the next page walks you through
creating a case from scratch:

- [Building a System](./building-a-system.md) — step-by-step guide to creating every file
- [Case Format Reference](../reference/case-format.md) — complete field-by-field schema
- [Configuration](../guide/configuration.md) — all `config.json` fields documented
