# Hydro Plants

Hydroelectric power plants are the central dispatchable resource in Cobre's system
model. Unlike thermal units, which simply convert fuel into electricity at a cost,
hydro plants manage a reservoir — a state variable that persists between stages and
couples the dispatch decisions of today to the feasibility of tomorrow. This
intertemporal coupling is precisely why hydrothermal scheduling requires stochastic
dynamic programming rather than a simple merit-order dispatch.

A hydro plant in Cobre is composed of three physical components: a **reservoir**
that stores water between stages, a **turbine** that converts water flow into
electrical generation, and a **spillway** that releases excess water without
producing power. Each stage's LP sub-problem contains one water balance constraint
per plant: inflow plus beginning storage equals turbined flow plus spillage plus
ending storage. The solver decides how much to turbine and how much to store,
trading off present-stage generation against future-stage optionality.

Plants can be linked into a **cascade** via the `downstream_id` field. When plant A
has `downstream_id` pointing to plant B, all water released from A (turbined flow
plus spillage) enters B's reservoir at the same stage. Cascade topology is validated
to be acyclic — no chain of downstream references may loop back to an earlier plant.

For a step-by-step introduction to writing `hydros.json`, see
[Building a System](../tutorial/building-a-system.md) and
[Anatomy of a Case](../tutorial/anatomy-of-a-case.md). This page provides the
complete field reference with all optional fields documented.

> **Theory reference**: For the mathematical formulation of hydro modeling and the
> SDDP algorithm that drives dispatch decisions, see
> [SDDP Theory](https://cobre-rs.github.io/cobre-docs/theory/sddp-theory.html)
> in the methodology reference.

---

## JSON Schema

Hydro plants are defined in `system/hydros.json`. The top-level object has a single
key `"hydros"` containing an array of plant objects. The following example shows
all fields — required and optional — for a single plant:

```json
{
  "hydros": [
    {
      "id": 1,
      "name": "UHE Tucuruí",
      "bus_id": 0,
      "downstream_id": null,
      "entry_stage_id": null,
      "exit_stage_id": null,
      "reservoir": {
        "min_storage_hm3": 50.0,
        "max_storage_hm3": 45000.0
      },
      "outflow": {
        "min_outflow_m3s": 1000.0,
        "max_outflow_m3s": 100000.0
      },
      "generation": {
        "model": "constant_productivity",
        "productivity_mw_per_m3s": 0.8765,
        "min_turbined_m3s": 500.0,
        "max_turbined_m3s": 22500.0,
        "min_generation_mw": 0.0,
        "max_generation_mw": 8370.0
      },
      "tailrace": {
        "type": "polynomial",
        "coefficients": [5.0, 0.001]
      },
      "hydraulic_losses": {
        "type": "factor",
        "value": 0.03
      },
      "efficiency": {
        "type": "constant",
        "value": 0.93
      },
      "evaporation_coefficients_mm": [
        80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0
      ],
      "diversion": {
        "downstream_id": 2,
        "max_flow_m3s": 200.0
      },
      "filling": {
        "start_stage_id": 48,
        "filling_inflow_m3s": 100.0
      },
      "penalties": {
        "spillage_cost": 0.01,
        "diversion_cost": 0.1,
        "fpha_turbined_cost": 0.05,
        "storage_violation_below_cost": 10000.0,
        "filling_target_violation_cost": 50000.0,
        "turbined_violation_below_cost": 500.0,
        "outflow_violation_below_cost": 500.0,
        "outflow_violation_above_cost": 500.0,
        "generation_violation_below_cost": 1000.0,
        "evaporation_violation_cost": 5000.0,
        "water_withdrawal_violation_cost": 1000.0
      }
    }
  ]
}
```

The `1dtoy` template uses a minimal hydro definition that omits all optional fields.
Only `id`, `name`, `bus_id`, `downstream_id`, `reservoir`, `outflow`, and `generation`
are required. All other top-level keys (`tailrace`, `hydraulic_losses`, `efficiency`,
`evaporation_coefficients_mm`, `diversion`, `filling`, `penalties`) are optional and
default to off when absent.

---

## Core Fields

These fields appear at the top level of each hydro plant object.

| Field            | Type            | Required | Description                                                                                                                                                      |
| ---------------- | --------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`             | integer         | Yes      | Unique non-negative integer identifier. Must be unique across all hydro plants. Referenced by `initial_conditions.json` and by other plants via `downstream_id`. |
| `name`           | string          | Yes      | Human-readable plant name. Used in output files, validation messages, and log output.                                                                            |
| `bus_id`         | integer         | Yes      | Identifier of the electrical bus to which this plant's generation is injected. Must match an `id` in `buses.json`.                                               |
| `downstream_id`  | integer or null | Yes      | Identifier of the plant that receives this plant's outflow. `null` means the plant is at the bottom of its cascade — outflow leaves the system.                  |
| `entry_stage_id` | integer or null | No       | Stage index at which the plant enters service (inclusive). `null` means the plant is available from stage 0.                                                     |
| `exit_stage_id`  | integer or null | No       | Stage index at which the plant is decommissioned (inclusive). `null` means the plant is never decommissioned.                                                    |

---

## Reservoir

The `reservoir` block defines the operational storage bounds for the plant. Storage
is tracked in **hm³** (cubic hectometres; 1 hm³ = 10⁶ m³). The beginning-of-stage
storage is the state variable that links consecutive stages in the LP.

```json
"reservoir": {
  "min_storage_hm3": 0.0,
  "max_storage_hm3": 1000.0
}
```

| Field             | Type   | Description                                                                                                                                                                   |
| ----------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `min_storage_hm3` | number | Minimum operational storage (dead volume). Water below this level cannot reach the turbine intakes. For plants that can empty completely, use `0.0`.                          |
| `max_storage_hm3` | number | Maximum operational storage (flood control level). When the reservoir reaches this level, all excess inflow must be spilled. Must be strictly greater than `min_storage_hm3`. |

Setting `min_storage_hm3` to the dead volume of your reservoir is important for
correctly computing the usable storage range. A reservoir with 500 hm³ total
physical capacity but 100 hm³ below the turbine intakes should be modeled as
`min_storage_hm3: 100.0, max_storage_hm3: 500.0`.

---

## Outflow Constraints

The `outflow` block constrains total outflow from the plant. Total outflow equals
turbined flow plus spillage. These constraints are enforced by soft penalties
when they cannot be satisfied due to extreme scenario conditions.

```json
"outflow": {
  "min_outflow_m3s": 0.0,
  "max_outflow_m3s": 50.0
}
```

| Field             | Type           | Description                                                                                                                                                         |
| ----------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `min_outflow_m3s` | number         | Minimum total outflow required at all times [m³/s]. Set to the ecological flow requirement or minimum riparian right. Use `0.0` if there is no minimum requirement. |
| `max_outflow_m3s` | number or null | Maximum total outflow [m³/s]. Models the physical capacity of the river channel below the dam. `null` means no upper bound on outflow.                              |

Minimum outflow is a hard lower bound on the sum of turbined flow and spillage.
When the solver cannot meet this bound (for example, because the reservoir is
nearly empty and inflow is very low), a violation slack variable is added to the
LP at the cost specified by `outflow_violation_below_cost` in the penalties block.

---

## Generation Models

The `generation` block configures the turbine model for dispatch purposes. It
provides the default production function used when no `hydro_production_models.json`
file is present, or for any plant not listed there. All variants share the core
turbine bounds (`min_turbined_m3s`, `max_turbined_m3s`) and generation bounds
(`min_generation_mw`, `max_generation_mw`). The `model` key selects which
production function converts flow to power.

```json
"generation": {
  "model": "constant_productivity",
  "productivity_mw_per_m3s": 1.0,
  "min_turbined_m3s": 0.0,
  "max_turbined_m3s": 50.0,
  "min_generation_mw": 0.0,
  "max_generation_mw": 50.0
}
```

| Field                     | Type   | Description                                                                                                |
| ------------------------- | ------ | ---------------------------------------------------------------------------------------------------------- |
| `model`                   | string | Production function variant. See the model table below.                                                    |
| `productivity_mw_per_m3s` | number | Power output per unit of turbined flow [MW/(m³/s)]. Used by `constant_productivity` and `linearized_head`. |
| `min_turbined_m3s`        | number | Minimum turbined flow [m³/s]. Non-zero values model a minimum stable turbine operation.                    |
| `max_turbined_m3s`        | number | Maximum turbined flow (installed turbine capacity) [m³/s].                                                 |
| `min_generation_mw`       | number | Minimum electrical generation [MW].                                                                        |
| `max_generation_mw`       | number | Maximum electrical generation (installed capacity) [MW].                                                   |

### Available Production Function Models

| Model                 | `model` value             | Status            | Description                                                                                                                                          |
| --------------------- | ------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Constant productivity | `"constant_productivity"` | Available         | `power = productivity * turbined_flow`. Independent of reservoir head. Requires only `productivity_mw_per_m3s`.                                      |
| FPHA                  | `"fpha"`                  | Available         | Piecewise-linear outer approximation of the nonlinear production function. Head-dependent. Configured via `hydro_production_models.json`. See below. |
| Linearized head       | `"linearized_head"`       | Not yet available | Head-dependent productivity linearized around an operating point at each stage. Will be documented when released.                                    |

For the `1dtoy` example and for most initial studies, `constant_productivity` is
the correct choice. The `productivity_mw_per_m3s` factor encodes the plant's
average efficiency and net head. For a plant with 80 m net head and 90% efficiency,
the theoretical productivity is approximately `9.81 * 80 * 0.90 / 1000 ≈ 0.706`
MW/(m³/s).

---

## FPHA Production Model

The FPHA (Forebay-Height Production Approximation) model represents the nonlinear
relationship between reservoir volume, turbined flow, spillage, and electrical
generation as a piecewise-linear outer approximation. It captures the head
dependence of hydro production — plants with high reservoir levels generate more
power for the same turbined flow.

FPHA is configured per plant and per stage via `system/hydro_production_models.json`.
A plant not listed in that file uses the `model` specified in its `generation` block
in `hydros.json`.

### Configuration File

`system/hydro_production_models.json` maps each hydro plant to a production model
selection strategy. The file is optional; when absent, all plants use their
`generation.model` from `hydros.json`.

Two selection strategies are supported:

**`stage_ranges`** — assigns a model to each contiguous stage interval:

```json
{
  "$schema": "../schemas/production_models.schema.json",
  "production_models": [
    {
      "hydro_id": 1,
      "selection_mode": "stage_ranges",
      "stage_ranges": [
        {
          "start_stage_id": 0,
          "end_stage_id": null,
          "model": "fpha",
          "fpha_config": {
            "source": "precomputed"
          }
        }
      ]
    }
  ]
}
```

Each stage range and season entry may include an optional `productivity_override`
field (see [Productivity Override](#productivity-override) below).

**`seasonal`** — assigns a model based on season index, with a fallback for seasons
not explicitly listed:

```json
{
  "$schema": "../schemas/production_models.schema.json",
  "production_models": [
    {
      "hydro_id": 1,
      "selection_mode": "seasonal",
      "default_model": "constant_productivity",
      "seasons": [
        {
          "season_id": 0,
          "model": "fpha",
          "fpha_config": {
            "source": "computed",
            "volume_discretization_points": 7,
            "turbine_discretization_points": 7
          }
        }
      ]
    }
  ]
}
```

Season indices are 0-based and match the season map defined in `stages.json`.

### Hyperplane Sources

When a plant is configured with `model: "fpha"`, the `fpha_config.source` field
selects where the hyperplane coefficients come from.

#### `source: "precomputed"`

Hyperplanes are loaded directly from `system/fpha_hyperplanes.parquet`. Use this
source when you have pre-fitted hyperplanes from a previous run or from an external
tool.

```json
"fpha_config": {
  "source": "precomputed"
}
```

The `fpha_config` block for `"precomputed"` requires no additional fields. The
discretization and fitting options are ignored — the hyperplanes are used as-is.

The Parquet file must be present at `system/fpha_hyperplanes.parquet`. Its schema is:

| Column            | Type    | Required | Description                                            |
| ----------------- | ------- | -------- | ------------------------------------------------------ |
| `hydro_id`        | INT32   | Yes      | Hydro plant identifier                                 |
| `stage_id`        | INT32?  | No       | Stage the plane applies to (`null` = all stages)       |
| `plane_id`        | INT32   | Yes      | Plane index within this hydro                          |
| `gamma_0`         | DOUBLE  | Yes      | Intercept coefficient (MW)                             |
| `gamma_v`         | DOUBLE  | Yes      | Volume coefficient (MW/hm³). Must be positive.         |
| `gamma_q`         | DOUBLE  | Yes      | Turbined flow coefficient (MW per m³/s)                |
| `gamma_s`         | DOUBLE  | Yes      | Spillage coefficient (MW per m³/s). Must be ≤ 0.       |
| `kappa`           | DOUBLE? | No       | Correction factor (default: 1.0)                       |
| `valid_v_min_hm3` | DOUBLE? | No       | Minimum volume where this plane is valid (hm³)         |
| `valid_v_max_hm3` | DOUBLE? | No       | Maximum volume where this plane is valid (hm³)         |
| `valid_q_max_m3s` | DOUBLE? | No       | Maximum turbined flow where this plane is valid (m³/s) |

Each `(hydro_id, stage_id)` group must have at least 3 planes. Rows are sorted by
`(hydro_id, stage_id, plane_id)` ascending; null `stage_id` sorts before any
non-null value.

#### `source: "computed"`

Hyperplanes are fitted at runtime from the plant's physical geometry. Cobre reads
the VHA (Volume-Height-Area) curve from `system/hydro_geometry.parquet`, evaluates
the production function `phi(v, q, s)` over a discretization grid, and fits a
piecewise-linear outer approximation.

This source requires:

1. The hydro plant must have `tailrace`, `hydraulic_losses`, and `efficiency` models
   defined in `hydros.json`.
2. `system/hydro_geometry.parquet` must contain at least 2 rows for the plant, with
   strictly increasing `volume_hm3` values and non-decreasing `height_m` and
   `area_km2` values.

```json
"fpha_config": {
  "source": "computed",
  "volume_discretization_points": 5,
  "turbine_discretization_points": 5,
  "spillage_discretization_points": 5,
  "max_planes_per_hydro": 10,
  "fitting_window": null
}
```

All fields except `source` are optional:

| Field                            | Default | Description                                                                                                            |
| -------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------- |
| `volume_discretization_points`   | 5       | Number of volume grid points for fitting. Must be >= 2.                                                                |
| `turbine_discretization_points`  | 5       | Number of turbined-flow grid points. Must be >= 2.                                                                     |
| `spillage_discretization_points` | 5       | Number of spillage grid points. Must be >= 2.                                                                          |
| `max_planes_per_hydro`           | 10      | Maximum planes retained after heuristic selection. Must be >= 1.                                                       |
| `fitting_window`                 | null    | Optional volume range for fitting. When absent, the full operating range `[min_storage_hm3, max_storage_hm3]` is used. |

The `fitting_window` field restricts which portion of the operating range is used to
construct the grid. Use it when the plant rarely operates near one extreme and you
want the planes to be tighter in the operating region. Two bound variants are
supported per dimension, and they are mutually exclusive:

```json
"fitting_window": {
  "volume_min_hm3": 1000.0,
  "volume_max_hm3": 40000.0
}
```

```json
"fitting_window": {
  "volume_min_percentile": 5.0,
  "volume_max_percentile": 95.0
}
```

Do not mix absolute (`_hm3`) and percentile (`_percentile`) bounds for the same
limit — the validator will reject the configuration.

### Kappa Correction Factor

The FPHA envelope is an outer approximation: by construction it never underestimates
generation. To ensure the LP does not systematically overestimate production, a
correction factor kappa (κ) is applied to each hyperplane's intercept:

```
gamma_0_effective = gamma_0 * kappa
```

Kappa is computed automatically during fitting by finding the tightest scalar
multiplier such that the scaled envelope is valid. It satisfies `0 < kappa <= 1.0`.

A kappa value below 0.95 indicates that the hyperplane envelope deviates
noticeably from the true production function over the fitted grid. When this
occurs, a warning is emitted during case loading:

```
Warning: hydro 'UHE Example' FPHA envelope has kappa = 0.87 (< 0.95).
Consider increasing discretization points or narrowing the fitting window.
```

For `source: "precomputed"`, kappa is read from the optional `kappa` column.
When absent or null, kappa defaults to 1.0 (the stored intercepts are used
unchanged).

### Parquet Export for Round-Trip Use

When hyperplanes are fitted at runtime (`source: "computed"`), the fitted
coefficients — including the computed kappa values — are automatically written to:

```
output/hydro_models/fpha_hyperplanes.parquet
```

This file uses the same 11-column schema as the input `system/fpha_hyperplanes.parquet`.
To switch from computed to precomputed fitting on a subsequent run, copy this file
to `system/fpha_hyperplanes.parquet` and change `source` to `"precomputed"` in
`hydro_production_models.json`.

### Productivity Override

When a plant uses the `constant_productivity` or `linearized_head` model, the
productivity value normally comes from the entity's `productivity_mw_per_m3s` field
in `hydros.json`. The optional `productivity_override` field on a stage range or
season entry replaces this base value for the stages covered by that entry.

This is useful when external data (e.g., NEWAVE MODIF.DAT temporal overrides of
tailrace or forebay elevations) changes the effective head drop at specific stages,
requiring a different productivity coefficient.

```json
{
  "start_stage_id": 12,
  "end_stage_id": 24,
  "model": "constant_productivity",
  "productivity_override": 0.72
}
```

| Field                   | Type           | Default | Description                                                                                           |
| ----------------------- | -------------- | ------- | ----------------------------------------------------------------------------------------------------- |
| `productivity_override` | number or null | `null`  | When present, replaces `productivity_mw_per_m3s` for this entry. Must be positive. Not valid on FPHA. |

**Validation rules:**

- `productivity_override` must be strictly positive when present.
- `productivity_override` is rejected when `model` is `"fpha"` (FPHA computes
  productivity from hyperplanes, not a scalar coefficient).
- When absent or `null`, the entity's base `productivity_mw_per_m3s` is used.

---

## Cascade Topology

The `downstream_id` field creates a directed chain of hydro plants. Water released
from an upstream plant — whether turbined or spilled — enters the downstream
plant's reservoir in the same stage.

To model a three-plant cascade where plant 0 flows into plant 1, which flows into
plant 2:

```json
{ "id": 0, "downstream_id": 1, ... }
{ "id": 1, "downstream_id": 2, ... }
{ "id": 2, "downstream_id": null, ... }
```

Cobre validates that the downstream graph is **acyclic**: no chain of
`downstream_id` references may return to a plant already in the chain. A cycle
would make the water balance equation unsolvable. The validator reports the cycle
as a topology error with the full chain of plant IDs.

Plants with `downstream_id: null` are **tailwater plants** — their outflow leaves
the basin. Each connected component of the cascade graph must have exactly one
tailwater plant (the chain's end node). A cascade component with no tailwater plant
would be a cycle, which the validator rejects.

---

## Advanced Fields

The following fields enable higher-fidelity physical modeling. They are all optional.
For most system planning studies, these fields can be omitted; they become relevant
when calibrating a model against historical dispatch data or when the head variation
at a plant is significant.

### Tailrace Model

The `tailrace` block models the downstream water level as a function of total
outflow. The tailrace elevation affects the net hydraulic head and is used by the
`linearized_head` and `fpha` generation models. When absent, tailrace elevation
is treated as zero.

Two variants are supported:

**Polynomial** — `height = a₀ + a₁·Q + a₂·Q² + …`

```json
"tailrace": {
  "type": "polynomial",
  "coefficients": [5.0, 0.001]
}
```

`coefficients` is an array of polynomial coefficients in ascending power order.
`coefficients[0]` is the constant term (height at zero outflow in metres),
`coefficients[1]` is the coefficient for Q¹, and so on.

**Piecewise** — linearly interpolated between (outflow, height) breakpoints.

```json
"tailrace": {
  "type": "piecewise",
  "points": [
    { "outflow_m3s": 0.0, "height_m": 3.0 },
    { "outflow_m3s": 5000.0, "height_m": 4.5 },
    { "outflow_m3s": 15000.0, "height_m": 6.2 }
  ]
}
```

Points must be sorted in ascending `outflow_m3s` order. The solver interpolates
linearly between adjacent points.

### Hydraulic Losses

The `hydraulic_losses` block models head loss in the penstock and draft tube.
Hydraulic losses reduce the effective head available at the turbine. When absent,
the penstock is modeled as lossless.

**Factor** — loss as a fraction of net head:

```json
"hydraulic_losses": { "type": "factor", "value": 0.03 }
```

`value` is a dimensionless fraction (e.g., `0.03` = 3% of net head).

**Constant** — fixed head loss regardless of flow:

```json
"hydraulic_losses": { "type": "constant", "value_m": 2.5 }
```

`value_m` is the fixed head loss in metres.

### Efficiency Model

The `efficiency` block scales the power output from the hydraulic power available.
When absent, 100% efficiency is assumed.

Currently only the `"constant"` variant is supported:

```json
"efficiency": { "type": "constant", "value": 0.93 }
```

`value` is a dimensionless fraction in the range (0, 1]. A value of `0.93` means
the turbine converts 93% of available hydraulic power to electrical output.

### Evaporation Coefficients

The `evaporation_coefficients_mm` field models water loss from the reservoir
surface due to evaporation. When present, it must be an array of exactly 12
values, one per calendar month:

```json
"evaporation_coefficients_mm": [
  80.0, 75.0, 70.0, 65.0, 60.0, 55.0,
  60.0, 65.0, 70.0, 75.0, 80.0, 85.0
]
```

Index 0 is January, index 11 is December. Values are in mm/month. The evaporated
volume is computed from the surface area of the reservoir at each stage. When
absent, no evaporation is modeled.

### Diversion Channel

The `diversion` block models a water diversion channel that routes flow directly
from this plant's reservoir to a downstream plant's reservoir, bypassing turbines
and spillways. When absent, no diversion is modeled.

```json
"diversion": {
  "downstream_id": 2,
  "max_flow_m3s": 200.0
}
```

| Field           | Description                                                         |
| --------------- | ------------------------------------------------------------------- |
| `downstream_id` | Identifier of the plant whose reservoir receives the diverted flow. |
| `max_flow_m3s`  | Maximum diversion flow capacity [m³/s].                             |

### Filling Configuration

The `filling` block enables a filling operation mode, where the reservoir is
intentionally filled from an external, fixed inflow source (such as a diversion
works from an unrelated basin) during a defined stage window. When absent, no
filling operation is active.

```json
"filling": {
  "start_stage_id": 48,
  "filling_inflow_m3s": 100.0
}
```

| Field                | Description                                                     |
| -------------------- | --------------------------------------------------------------- |
| `start_stage_id`     | Stage index at which filling begins (inclusive).                |
| `filling_inflow_m3s` | Constant inflow applied to the reservoir during filling [m³/s]. |

---

## Penalties

The `penalties` block inside a hydro plant definition overrides the global defaults
from `penalties.json` for that specific plant. When the block is absent, all penalty
values fall back to the global defaults. When it is present, it must contain all 11
fields.

Penalty costs are added to the LP objective when soft constraint violations occur.
They do not represent physical costs — they are optimization weights that guide the
solver to avoid infeasible or undesirable operating states.

```json
"penalties": {
  "spillage_cost": 0.01,
  "diversion_cost": 0.1,
  "fpha_turbined_cost": 0.05,
  "storage_violation_below_cost": 10000.0,
  "filling_target_violation_cost": 50000.0,
  "turbined_violation_below_cost": 500.0,
  "outflow_violation_below_cost": 500.0,
  "outflow_violation_above_cost": 500.0,
  "generation_violation_below_cost": 1000.0,
  "evaporation_violation_cost": 5000.0,
  "water_withdrawal_violation_cost": 1000.0
}
```

| Field                             | Unit   | Description                                                                                                                                                                                        |
| --------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `spillage_cost`                   | $/m³/s | Penalty per m³/s of water spilled. Setting this low (e.g., 0.01) makes spillage the least-cost way to relieve a flood situation. Setting it high penalizes wasted water in water-scarce scenarios. |
| `diversion_cost`                  | $/m³/s | Penalty per m³/s of diverted flow exceeding the diversion channel capacity.                                                                                                                        |
| `fpha_turbined_cost`              | $/MWh  | Penalty per MWh of turbined generation in the FPHA approximation. Not used by `constant_productivity`.                                                                                             |
| `storage_violation_below_cost`    | $/hm³  | Penalty per hm³ of storage below `min_storage_hm3`. Should be set high (thousands) to make violations a last resort.                                                                               |
| `filling_target_violation_cost`   | $/hm³  | Penalty per hm³ of storage below the filling target. Only active when a `filling` block is present.                                                                                                |
| `turbined_violation_below_cost`   | $/m³/s | Penalty per m³/s of turbined flow below `min_turbined_m3s`.                                                                                                                                        |
| `outflow_violation_below_cost`    | $/m³/s | Penalty per m³/s of total outflow below `min_outflow_m3s`. Set high to enforce ecological flow requirements.                                                                                       |
| `outflow_violation_above_cost`    | $/m³/s | Penalty per m³/s of total outflow above `max_outflow_m3s`. Set high to enforce flood channel capacity limits.                                                                                      |
| `generation_violation_below_cost` | $/MW   | Penalty per MW of generation below `min_generation_mw`.                                                                                                                                            |
| `evaporation_violation_cost`      | $/mm   | Penalty per mm of evaporation constraint violation. Only active when `evaporation_coefficients_mm` is present.                                                                                     |
| `water_withdrawal_violation_cost` | $/m³/s | Penalty per m³/s of water withdrawal constraint violation.                                                                                                                                         |

### Three-Tier Resolution Cascade

Penalty values are resolved from the most specific to the most general source:

1. **Stage-level override** (defined in stage-specific penalty files, when present)
2. **Entity-level override** (the `penalties` block inside the plant's JSON object)
3. **Global default** (the `hydro` section of `penalties.json`)

The `penalties` block on a plant replaces the global default for that plant alone.
All plants that do not have a `penalties` block use the global values from
`penalties.json`. The global `penalties.json` file must always be present and must
contain all 11 hydro penalty fields.

---

## Validation Rules

Cobre's five-layer validation pipeline checks the following conditions on hydro
plants. Violations are reported as error messages with the failing plant's `id`
and the nature of the problem.

| Rule                            | Error Class          | Description                                                                                                                      |
| ------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Bus reference integrity         | Reference error      | Every `bus_id` must match an `id` in `buses.json`.                                                                               |
| Downstream reference integrity  | Reference error      | Every non-null `downstream_id` must match an `id` in `hydros.json`.                                                              |
| Cascade acyclicity              | Topology error       | The directed graph of `downstream_id` links must be acyclic.                                                                     |
| Storage bounds ordering         | Physical feasibility | `min_storage_hm3` must be less than `max_storage_hm3`.                                                                           |
| Outflow bounds ordering         | Physical feasibility | When `max_outflow_m3s` is present, it must be greater than or equal to `min_outflow_m3s`.                                        |
| Turbine bounds ordering         | Physical feasibility | `min_turbined_m3s` must be less than or equal to `max_turbined_m3s`.                                                             |
| Generation bounds consistency   | Physical feasibility | `min_generation_mw` must be less than or equal to `max_generation_mw`.                                                           |
| Initial conditions completeness | Reference error      | Every hydro plant must have exactly one entry in `initial_conditions.json` (either in `storage` or `filling_storage`, not both). |
| Evaporation array length        | Schema error         | When `evaporation_coefficients_mm` is present, it must have exactly 12 values.                                                   |
| FPHA geometry coverage          | Dimensional error    | Every plant configured with `fpha` or `linearized_head` must have at least 2 rows in `system/hydro_geometry.parquet`.            |
| FPHA plane coverage             | Dimensional error    | Every `(hydro_id, stage_id)` group in `system/fpha_hyperplanes.parquet` must have at least 3 planes.                             |
| FPHA coefficient signs          | Semantic error       | `gamma_v` must be positive; `gamma_s` must be non-positive.                                                                      |
| Geometry monotonicity           | Semantic error       | `volume_hm3` must be strictly increasing; `height_m` and `area_km2` must be non-decreasing.                                      |

---

## Related Pages

- [Anatomy of a Case](../tutorial/anatomy-of-a-case.md) — walks through the complete `1dtoy` hydro definition
- [Building a System](../tutorial/building-a-system.md) — step-by-step guide to writing `hydros.json` from scratch
- [System Modeling](./system-modeling.md) — overview of all entity types and how they interact
- [Case Format Reference](../reference/case-format.md) — complete JSON schema for all input files
