# Building a System

This page walks you through creating a minimal case directory from scratch,
explaining why each file exists and what each field controls. The target is a
single-bus hydrothermal system identical to the `1dtoy` template: one bus, one
hydro plant, two thermal units, and a four-month planning horizon.

If you want to start from a working template instead, use:

```bash
cobre init --template 1dtoy my_study
```

This page is for users who want to understand the structure of every file before
touching real data.

---

## Prerequisites

Create an empty directory and enter it:

```bash
mkdir my_study
cd my_study
mkdir system
```

You will need 8 JSON files. By the end of this guide your directory will look like:

```
my_study/
  config.json
  initial_conditions.json
  penalties.json
  stages.json
  system/
    buses.json
    hydros.json
    lines.json
    thermals.json
```

The `scenarios/` subdirectory is optional for a minimal case. Cobre can generate
white-noise inflow and load scenarios using only the stage definitions, without
Parquet statistics files.

---

## Step 1: Create `config.json`

`config.json` tells Cobre how to run the study. At minimum it needs a `training`
section with a `forward_passes` count and at least one `stopping_rules` entry.

Create `my_study/config.json`:

```json
{
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

`forward_passes` controls how many scenario trajectories are drawn per training
iteration. Start with 1 for fast iteration during case development; increase to
50 or more for production runs where you want lower variance per iteration.

`stopping_rules` must contain at least one `iteration_limit` entry. The solver
will run until one of the configured rules triggers. Here it stops after 128
iterations regardless of convergence. You can add a second rule — for example,
`{ "type": "time_limit", "seconds": 300 }` — and the solver will stop when
either condition is met.

The `simulation` block is optional. When `enabled: true`, Cobre runs a
post-training simulation pass using `num_scenarios` independently sampled
scenarios and writes dispatch results to Parquet files.

For the full list of configuration options including warm-start, cut selection,
and output controls, see [Configuration](../guide/configuration.md).

---

## Step 2: Create `stages.json`

`stages.json` defines the time horizon. Each stage represents a planning period.
The solver builds one LP sub-problem per stage per scenario trajectory.

Create `my_study/stages.json`:

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

`policy_graph.type: "finite_horizon"` is the correct choice for a planning
horizon with a definite end date and no cycling. The `annual_discount_rate` is
applied to discount future stage costs back to present value. A rate of 0.12
means costs one year in the future are worth 88% of present costs.

Each stage entry needs an `id` (0-indexed integer), a `start_date` and `end_date`
in ISO 8601 format, an array of `blocks`, and a `num_scenarios` count.

The `blocks` array subdivides a stage into load periods. A single block named
`SINGLE` that spans all the hours of the month is the simplest choice. More
detailed studies use two or three blocks (peak/off-peak/overnight) to capture
intra-stage load variation. The `hours` value must equal the actual number of
hours in the stage: these hours convert MW dispatch levels to MWh costs in the
LP objective.

`num_scenarios` is the number of inflow/load scenario trajectories sampled at
each stage during training. More scenarios per iteration produce less-noisy
cut estimates at the cost of more LP solves per iteration.

---

## Step 3: Create `penalties.json`

Penalty costs define how much the solver pays when it cannot satisfy a constraint
without violating a physical bound. High penalties make violations expensive so
the solver avoids them; low penalties on minor constraints (like spillage) allow
the solver to use flexibility when needed.

Create `my_study/penalties.json`:

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

The `bus.deficit_segments` array must end with a segment where `depth_mw` is
`null`. This unbounded final segment ensures the LP always has a feasible solution
even when generation capacity is insufficient to cover load. All four top-level
sections (`bus`, `line`, `hydro`, `non_controllable_source`) are required even
if your system contains none of that entity type.

Individual penalty values can be overridden per entity by adding a `penalties`
block inside any entity definition in the `system/` files. The global values
here serve as the default for any entity that does not specify its own.

---

## Step 4: Create `system/buses.json`

A bus is an electrical node. All generators and loads connect to a bus. Every
system needs at least one bus.

Create `my_study/system/buses.json`:

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

`id` must be a unique non-negative integer. `name` is a human-readable label
used in output files and validation messages. The `deficit_segments` override
here replaces the global deficit curve from `penalties.json` for this specific
bus. A single unbounded segment at 1000 $/MWh is the simplest possible deficit
model.

If you omit `deficit_segments` from a bus, Cobre uses the global default from
`penalties.json` for that bus. Explicit overrides are useful when different buses
have different Value of Lost Load characteristics.

---

## Step 5: Create `system/lines.json`

Transmission lines connect pairs of buses and impose flow limits between them.
A single-bus system has no lines.

Create `my_study/system/lines.json`:

```json
{
  "lines": []
}
```

The file must exist even with an empty array. The validator checks that the file
is present and that its schema is valid. If you later add a second bus, you can
add lines here by specifying `source_bus_id`, `target_bus_id`, `direct_mw`, and
`reverse_mw` for each line.

---

## Step 6: Create `system/thermals.json`

Thermal plants are dispatchable generators. They have a fixed cost per MWh of
generation and physical capacity bounds. Add them in increasing cost order as
a matter of convention, though the LP will find the optimal merit order regardless.

Create `my_study/system/thermals.json`:

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

`bus_id: 0` connects both plants to the SIN bus. The `cost_segments` array
defines a piecewise-linear cost curve. Each segment has a `capacity_mw` and a
`cost_per_mwh`. With a single segment, the entire capacity is available at the
same cost. The segment capacities should sum to `generation.max_mw`.

`generation.min_mw: 0.0` means the plant can be turned off completely. A
non-zero minimum would represent a must-run commitment constraint. `max_mw`
caps the generation level and should equal the sum of all `cost_segments`
capacities.

The `bus_id` must reference a bus `id` defined in `buses.json`. The validator
will catch any broken reference and report it as a reference integrity error.

---

## Step 7: Create `system/hydros.json`

Hydro plants have three components: a reservoir (state variable between stages),
a turbine (converts water flow to electricity), and optional cascade linkage to
downstream plants.

Create `my_study/system/hydros.json`:

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

`downstream_id: null` marks UHE1 as a tailwater plant. To model a cascade where
plant A flows into plant B, you would set `downstream_id: <B's id>` on plant A.
Cobre enforces that the downstream graph is acyclic.

The `reservoir` block uses hm³ (cubic hectometres) as the unit for water volume.
`min_storage_hm3: 0.0` allows the reservoir to empty completely. If your plant
has a dead storage (volume below the turbine intake), set `min_storage_hm3` to
that value.

The `outflow` block limits total outflow (turbined flow plus spillage). The upper
bound `max_outflow_m3s: 50.0` models the river channel capacity. Setting a
non-zero `min_outflow_m3s` would represent a minimum ecological flow requirement.

The `generation` block uses `"constant_productivity"`, the simplest of the three
supported turbine models. The other two — `"linearized_head"` and `"fpha"` (four-
piece hyperplane approximation) — model head-dependent productivity for variable-
head plants. The `productivity_mw_per_m3s` factor converts turbined flow to
generated power. Here 1 m³/s yields 1 MW. Real plants typically have productivity
factors between 0.5 and 10 depending on the head height. For details on all three
models, see [Hydro Plants](../guide/hydro-plants.md).

---

## Step 8: Create `initial_conditions.json`

Every hydro plant needs an initial reservoir storage value at the start of the
study. This is the state the solver uses for stage 0's water balance equation.

Create `my_study/initial_conditions.json`:

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

`hydro_id: 0` matches UHE1 defined in `system/hydros.json`. Every hydro plant
in the system must have exactly one entry in either `storage` or
`filling_storage` — not both, not neither. The validator checks this.

`value_hm3: 83.222` sets the initial reservoir at about 8.3% of its 1000 hm³
capacity. Choosing a realistic initial condition matters for short horizons
because the first few stages will be heavily influenced by whether the reservoir
starts full or nearly empty. For multi-year studies the initial condition has
less impact on later stages.

`filling_storage` is for filling reservoirs — reservoirs that accumulate water
but do not generate power. The 1dtoy system has none, so this array is empty.
It must be present even when empty.

---

## Step 9: Validate Your Case

With all 8 files in place, validate the case to confirm every layer passes:

```bash
cobre validate my_study
```

On success, Cobre prints the entity counts:

```
Valid case: 1 buses, 1 hydros, 2 thermals, 0 lines
  buses: 1
  hydros: 1
  thermals: 2
  lines: 0
```

If any validation layer fails, each error is prefixed with `error:` and the
exit code is 1. Common errors at this stage:

- `reference error: hydro 0 references bus 99 which does not exist` — a
  `bus_id` in `hydros.json` does not match any `id` in `buses.json`.
- `initial conditions: hydro 0 has no initial storage entry` — a hydro plant
  in `hydros.json` is missing from `initial_conditions.json`.
- `penalties.json: non_controllable_source section missing` — a required top-level
  section is absent from `penalties.json`, even if the system has no NCS plants.

Fix each reported error and re-run `cobre validate` until the exit code is 0.

---

## What's Next

Your hand-built case is functionally identical to the `1dtoy` template. You can
run it directly:

```bash
cobre run my_study --output my_study/results
```

To compare your files against the template at any point:

```bash
cobre init --template 1dtoy 1dtoy_reference
diff -r my_study 1dtoy_reference
```

From here, the natural next steps are:

- [Understanding Results](./understanding-results.md) — how to read the Parquet output files
- [Anatomy of a Case](./anatomy-of-a-case.md) — detailed explanation of every field in these files
- [Case Format Reference](../reference/case-format.md) — complete schema with all optional fields
- [Configuration](../guide/configuration.md) — advanced `config.json` options including warm-start and cut selection
