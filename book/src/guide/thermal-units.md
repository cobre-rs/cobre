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

The cost structure of a thermal unit is modeled with a **scalar marginal cost**
(`cost_per_mwh`). The LP dispatches the unit at any level between `min_mw` and
`max_mw`, with the generation cost equal to `dispatched_mw * hours_in_block * cost_per_mwh`.

For an introductory walkthrough of writing `thermals.json`, see
[Building a System](../tutorial/building-a-system.md) and
[Anatomy of a Case](../tutorial/anatomy-of-a-case.md). This page provides the
complete field reference, including GNL configuration.

---

## JSON Schema

Thermal units are defined in `system/thermals.json`. The top-level object has a
single key `"thermals"` containing an array of unit objects. The following example
shows all fields, including the optional `entry_stage_id`, `exit_stage_id`, and
`gnl_config`:

```json
{
  "thermals": [
    {
      "id": 0,
      "name": "UTE1",
      "bus_id": 0,
      "cost_per_mwh": 5.0,
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
      "cost_per_mwh": 50.0,
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

The first plant (`UTE1`) matches the `1dtoy` template format: a cost per MWh with
no optional fields. The second plant (`Angra 1`) shows the complete schema with GNL
dispatch anticipation. The fields `entry_stage_id`, `exit_stage_id`, and
`gnl_config` are optional and can be omitted.

---

## Core Fields

These fields appear at the top level of each thermal unit object.

| Field            | Type            | Required | Description                                                                                                       |
| ---------------- | --------------- | -------- | ----------------------------------------------------------------------------------------------------------------- |
| `id`             | integer         | Yes      | Unique non-negative integer identifier. Must be unique across all thermal units.                                  |
| `name`           | string          | Yes      | Human-readable plant name. Used in output files, validation messages, and log output.                             |
| `bus_id`         | integer         | Yes      | Identifier of the electrical bus to which this unit's generation is injected. Must match an `id` in `buses.json`. |
| `cost_per_mwh`   | number          | Yes      | Marginal cost of generation [$/MWh]. Must be ≥ 0.0.                                                               |
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
| `max_mw` | number | Maximum electrical generation (installed capacity) [MW].                                                                                                                                                       |

A `min_mw` of `0.0` means the unit can be turned off completely — it is treated as
an interruptible resource. A non-zero `min_mw` (for example, `100.0` for a plant
whose turbine must spin continuously for mechanical reasons) means the LP must
always dispatch at least that amount whenever the plant is active.

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

| Rule                       | Error Class          | Description                                                                |
| -------------------------- | -------------------- | -------------------------------------------------------------------------- |
| Bus reference integrity    | Reference error      | Every `bus_id` must match an `id` in `buses.json`.                         |
| Non-negative cost          | Schema error         | `cost_per_mwh` must be ≥ 0.0.                                              |
| Generation bounds ordering | Physical feasibility | `min_mw` must be less than or equal to `max_mw`.                           |
| GNL lag validity           | Physical feasibility | When `gnl_config` is present, `lag_stages` must be a non-negative integer. |

---

## Related Pages

- [Anatomy of a Case](../tutorial/anatomy-of-a-case.md) — walks through the complete `1dtoy` thermal definitions
- [Building a System](../tutorial/building-a-system.md) — step-by-step guide to writing `thermals.json` from scratch
- [System Modeling](./system-modeling.md) — overview of all entity types and how they interact
- [Case Format Reference](../reference/case-format.md) — complete JSON schema for all input files
