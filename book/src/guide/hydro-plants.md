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

The `generation` block configures the turbine model (internally stored as the
`generation_model` field on the `Hydro` struct). All variants share the core
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

| Model                 | `model` value             | Status            | Description                                                                                                                                    |
| --------------------- | ------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Constant productivity | `"constant_productivity"` | Available         | `power = productivity * turbined_flow`. Independent of reservoir head. The only model supported in the current release.                        |
| Linearized head       | `"linearized_head"`       | Not yet available | Head-dependent productivity linearized around an operating point at each stage. Will be documented when released.                              |
| FPHA                  | `"fpha"`                  | Not yet available | Full production function with head-area-productivity tables. Requires forebay and tailrace elevation tables. Will be documented when released. |

For the `1dtoy` example and for most initial studies, `constant_productivity` is
the correct choice. The `productivity_mw_per_m3s` factor encodes the plant's
average efficiency and net head. For a plant with 80 m net head and 90% efficiency,
the theoretical productivity is approximately `9.81 * 80 * 0.90 / 1000 ≈ 0.706`
MW/(m³/s).

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

---

## Related Pages

- [Anatomy of a Case](../tutorial/anatomy-of-a-case.md) — walks through the complete `1dtoy` hydro definition
- [Building a System](../tutorial/building-a-system.md) — step-by-step guide to writing `hydros.json` from scratch
- [System Modeling](./system-modeling.md) — overview of all entity types and how they interact
- [Case Format Reference](../reference/case-format.md) — complete JSON schema for all input files
