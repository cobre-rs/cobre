# Thermal Units

Thermal power plants are the dispatchable generation assets that complement hydro
in Cobre's system model. The term "thermal" covers any generator whose output is
bounded by installed capacity and whose dispatch incurs an explicit cost per MWh:
combustion turbines, combined-cycle plants, coal-fired units, nuclear plants, and
diesel generators all map onto the same Cobre `Thermal` entity type.

Unlike hydro plants, thermal units carry no state between stages. Each stage's
LP sub-problem treats a thermal unit as a simple bounded generation variable with
a marginal cost. The solver dispatches thermal units in merit order — from cheapest
to most expensive — to meet any residual demand not covered by hydro generation.
In a hydrothermal system, the long-run value of stored water is compared against
the short-run cost of thermal dispatch at each stage, which is the fundamental
trade-off the SDDP algorithm optimizes.

The cost structure of a thermal unit is modeled with a **piecewise-linear cost
curve** (`cost_segments`). A single-segment plant dispatches all its capacity at a
flat cost. A multi-segment plant has increasing marginal costs at higher output
levels, reflecting the physical reality that a plant becomes less fuel-efficient
as it approaches its rated capacity.

For an introductory walkthrough of writing `thermals.json`, see
[Building a System](../tutorial/building-a-system.md) and
[Anatomy of a Case](../tutorial/anatomy-of-a-case.md). This page provides the
complete field reference, including multi-segment cost curves and GNL configuration.

---

## JSON Schema

Thermal units are defined in `system/thermals.json`. The top-level object has a
single key `"thermals"` containing an array of unit objects. The following example
shows all fields for a two-segment plant with GNL configuration:

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
      "name": "Angra 1",
      "bus_id": 0,
      "entry_stage_id": null,
      "exit_stage_id": null,
      "cost_segments": [
        {
          "capacity_mw": 300.0,
          "cost_per_mwh": 50.0
        },
        {
          "capacity_mw": 357.0,
          "cost_per_mwh": 80.0
        }
      ],
      "generation": {
        "min_mw": 0.0,
        "max_mw": 657.0
      },
      "gnl_config": {
        "lag_stages": 2
      }
    }
  ]
}
```

The first plant (`UTE1`) matches the `1dtoy` template format: a single cost
segment with no optional fields. The second plant (`Angra 1`) shows the complete
schema with a two-segment cost curve and GNL dispatch anticipation. The fields
`entry_stage_id`, `exit_stage_id`, and `gnl_config` are optional and can be omitted.

---

## Core Fields

These fields appear at the top level of each thermal unit object.

| Field            | Type            | Required | Description                                                                                                       |
| ---------------- | --------------- | -------- | ----------------------------------------------------------------------------------------------------------------- |
| `id`             | integer         | Yes      | Unique non-negative integer identifier. Must be unique across all thermal units.                                  |
| `name`           | string          | Yes      | Human-readable plant name. Used in output files, validation messages, and log output.                             |
| `bus_id`         | integer         | Yes      | Identifier of the electrical bus to which this unit's generation is injected. Must match an `id` in `buses.json`. |
| `entry_stage_id` | integer or null | No       | Stage index at which the unit enters service (inclusive). `null` means the unit is available from stage 0.        |
| `exit_stage_id`  | integer or null | No       | Stage index at which the unit is decommissioned (inclusive). `null` means the unit is never decommissioned.       |

---

## Generation Bounds

The `generation` block sets the output limits for the unit (stored internally as
`min_generation_mw` and `max_generation_mw` on the `Thermal` struct). These are
enforced as hard bounds on the generation variable in each stage LP.

```json
"generation": {
  "min_mw": 0.0,
  "max_mw": 657.0
}
```

| Field    | Type   | Description                                                                                                                                                                                                    |
| -------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `min_mw` | number | Minimum electrical generation (minimum stable load) [MW]. A non-zero value represents a must-run commitment: the solver is required to dispatch at least this much generation whenever the unit is in service. |
| `max_mw` | number | Maximum electrical generation (installed capacity) [MW]. This must equal the sum of all `capacity_mw` values in `cost_segments`.                                                                               |

A `min_mw` of `0.0` means the unit can be turned off completely — it is treated as
an interruptible resource. A non-zero `min_mw` (for example, `100.0` for a plant
whose turbine must spin continuously for mechanical reasons) means the LP must
always dispatch at least that amount whenever the plant is active.

The `max_mw` field caps total generation and must equal the sum of all segment
capacities in `cost_segments`. The validator checks this constraint and reports an
error if the values do not match.

---

## Cost Segments

The `cost_segments` array defines the piecewise-linear generation cost curve. Each
segment represents a range of generation capacity and its associated marginal cost.
Segments are applied in order: the first `capacity_mw` MW of output uses the first
segment's cost, the next `capacity_mw` MW uses the second segment's cost, and so on.

```json
"cost_segments": [
  {
    "capacity_mw": 300.0,
    "cost_per_mwh": 50.0
  },
  {
    "capacity_mw": 357.0,
    "cost_per_mwh": 80.0
  }
]
```

| Field          | Type   | Description                                                 |
| -------------- | ------ | ----------------------------------------------------------- |
| `capacity_mw`  | number | Generation capacity of this segment [MW]. Must be positive. |
| `cost_per_mwh` | number | Marginal cost in this segment [$/MWh].                      |

### Single-Segment Plants

Most thermal units in planning studies use a single cost segment, which treats the
entire capacity as available at a uniform marginal cost:

```json
"cost_segments": [
  { "capacity_mw": 15.0, "cost_per_mwh": 5.0 }
]
```

The LP will dispatch this plant at any level between `min_mw` and `max_mw`, with
the generation cost equal to `dispatched_mw * hours_in_block * cost_per_mwh`.

### Multi-Segment Plants

A multi-segment curve models a plant whose heat rate increases at higher output,
which is common for steam turbines and combined-cycle units. For example, a 657 MW
plant that is efficient at partial load but increasingly expensive above 300 MW:

```json
"cost_segments": [
  { "capacity_mw": 300.0, "cost_per_mwh": 50.0 },
  { "capacity_mw": 357.0, "cost_per_mwh": 80.0 }
]
```

The LP sees this as two separate generation variables that are constrained to be
dispatched in order: the cheaper 300 MW segment fills first before the solver
uses any of the 357 MW higher-cost segment. The total capacity is 300 + 357 = 657 MW,
which must equal `generation.max_mw`.

Segments must be listed in **ascending cost order** as a convention, though the
optimizer will find the merit-order dispatch regardless of the ordering in the file.

### Capacity Sum Constraint

The sum of all `capacity_mw` values must equal `generation.max_mw`. This is
validated by Cobre and reported as a physical feasibility error if violated:

```
physical error: thermal 1 cost_segments capacity sum (657.0 MW) does not match
  max_generation_mw (700.0 MW)
```

---

## GNL Configuration

> **Not yet implemented.** The `gnl_config` field is parsed and validated but has
> no effect on the LP formulation in the current version. GNL dispatch anticipation
> is a planned feature — see the [CHANGELOG](https://github.com/cobre-rs/cobre/blob/main/CHANGELOG.md)
> for the implementation timeline.

The optional `gnl_config` block is intended to enable GNL (Gás Natural Liquefeito,
or liquefied natural gas) dispatch anticipation. This will model thermal units that
require advance scheduling over multiple stages due to commitment lead times — for
example, an LNG-fired plant that must be booked several weeks before the dispatch
occurs.

```json
"gnl_config": {
  "lag_stages": 2
}
```

| Field        | Type    | Description                                                                                                                               |
| ------------ | ------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `lag_stages` | integer | Number of stages of dispatch anticipation. A value of `2` means the generation commitment for stage `t` must be decided at stage `t - 2`. |

When implemented, `lag_stages` greater than zero will couple the commitment
decision at an earlier stage to the dispatch variable at a later stage. For now,
the field is accepted by the parser but silently ignored during LP construction.

---

## Validation Rules

Cobre's five-layer validation pipeline checks the following conditions on thermal
units. Violations are reported as error messages with the failing unit's `id`.

| Rule                       | Error Class          | Description                                                                                           |
| -------------------------- | -------------------- | ----------------------------------------------------------------------------------------------------- |
| Bus reference integrity    | Reference error      | Every `bus_id` must match an `id` in `buses.json`.                                                    |
| Cost segment capacity sum  | Physical feasibility | The sum of all `capacity_mw` values in `cost_segments` must equal `max_mw` in the `generation` block. |
| Generation bounds ordering | Physical feasibility | `min_mw` must be less than or equal to `max_mw`.                                                      |
| Non-empty cost segments    | Schema error         | The `cost_segments` array must contain at least one segment.                                          |
| Positive segment capacity  | Physical feasibility | Each segment's `capacity_mw` must be strictly positive.                                               |
| GNL lag validity           | Physical feasibility | When `gnl_config` is present, `lag_stages` must be a non-negative integer.                            |

---

## Related Pages

- [Anatomy of a Case](../tutorial/anatomy-of-a-case.md) — walks through the complete `1dtoy` thermal definitions
- [Building a System](../tutorial/building-a-system.md) — step-by-step guide to writing `thermals.json` from scratch
- [System Modeling](./system-modeling.md) — overview of all entity types and how they interact
- [Case Format Reference](../reference/case-format.md) — complete JSON schema for all input files
