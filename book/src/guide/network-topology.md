# Network Topology

The electrical network in Cobre describes how generators and loads are connected
and how power can move between regions. At the heart of the network model is the
**bus**: a named node at which power balance must be maintained every stage and
every load block. Generators inject power into buses; loads withdraw power from
buses; transmission lines transfer power between buses.

The simplest possible model is a **single-bus (copper-plate)** system: one bus
that aggregates all generation and all load into a single node. In a copper-plate
model there are no flow limits, no transmission losses, and no geographical
differentiation in price or dispatch. The `1dtoy` template uses a single-bus
configuration. This is the right starting point for system-level capacity planning
studies where the internal transmission network is not the focus.

A **multi-bus** system introduces two or more buses connected by transmission lines.
Lines impose flow limits between buses. When a line's capacity is binding, each
bus has its own locational marginal price, and the dispatch in one region cannot
freely substitute for a deficit in another. Multi-bus models are appropriate when
regional subsystems have constrained interconnections that influence dispatch,
investment decisions, or price formation.

---

## Buses

Every generator and every load must be attached to a bus. Buses are defined in
`system/buses.json` under a top-level `"buses"` array.

### JSON Schema

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

This is the complete `buses.json` from the `1dtoy` example: one bus with a single
unbounded deficit segment at 1000 $/MWh. The `excess_cost` field is optional and
comes from the global `penalties.json` when not specified per-bus.

### Core Fields

| Field              | Type    | Required | Description                                                                                                                                       |
| ------------------ | ------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`               | integer | Yes      | Unique non-negative integer identifier. Must be unique across all buses.                                                                          |
| `name`             | string  | Yes      | Human-readable bus name. Used in output files, validation messages, and log output.                                                               |
| `deficit_segments` | array   | No       | Piecewise-linear deficit cost curve. Overrides the global defaults from `penalties.json` for this bus. See [Deficit Modeling](#deficit-modeling). |
| `excess_cost`      | number  | No       | Penalty per MWh of surplus generation absorbed by this bus ($/MWh). Overrides the global default from `penalties.json`.                           |

### Bus Balance Constraint

For every bus `b`, every stage `t`, and every load block `k`, the LP enforces:

```
  generation_injected(b, t, k)
  + imports_from_lines(b, t, k)
  + deficit(b, t, k)
  = load_demand(b, t, k)
  + exports_to_lines(b, t, k)
  + excess(b, t, k)
```

`deficit` and `excess` are non-negative slack variables added to the LP objective
at their respective penalty costs. The deficit slack makes the problem feasible
when there is not enough generation to meet demand. The excess slack absorbs
surplus generation when more power is produced than can be consumed or transmitted
away.

---

## Deficit Modeling

Deficit represents unserved load — demand that the solver cannot cover with
available generation. The deficit cost is the **Value of Lost Load (VoLL)** from the
solver's perspective: the penalty the LP pays per MWh of unserved demand.

### Deficit Segments

Rather than a single flat VoLL, Cobre models deficit costs as a **piecewise-linear
curve**: a sequence of segments with increasing costs. The segments are cumulative.
The first segment covers the first `depth_mw` MW of deficit at the lowest cost,
the second segment covers the next `depth_mw` MW at a higher cost, and so on.

```json
"deficit_segments": [
  { "depth_mw": 500.0, "cost": 1000.0 },
  { "depth_mw": null,  "cost": 5000.0 }
]
```

In this two-segment example, the first 500 MW of deficit costs 1000 $/MWh. Any
deficit above 500 MW costs 5000 $/MWh. The final segment must have `depth_mw: null`
(unbounded), which guarantees the LP can always find a feasible solution regardless
of the generation shortfall.

| Field      | Type           | Description                                                                                                      |
| ---------- | -------------- | ---------------------------------------------------------------------------------------------------------------- |
| `depth_mw` | number or null | MW of deficit covered by this segment. `null` for the final unbounded segment.                                   |
| `cost`     | number         | Penalty cost per MWh of deficit in this segment [$/MWh]. Must be positive. Segments should be in ascending cost. |

### Three-Tier Penalty Resolution

Deficit segment values are resolved from the most specific to the most general source:

1. **Stage-level override** — penalty files for individual stages, when present
2. **Bus-level override** — the `deficit_segments` array inside the bus's JSON object
3. **Global default** — the `bus.deficit_segments` section of `penalties.json`

When `deficit_segments` is omitted from a bus definition, Cobre uses the global
default from `penalties.json`. This makes it easy to set a system-wide VoLL and
then override it for specific buses with different reliability requirements.

### Choosing Deficit Costs

A typical two-tier configuration uses a moderate cost for the first tier (to allow
partial deficit in extreme scenarios without distorting the optimality cuts too
much) and a very high cost for the unbounded final tier (to make full deficit a
last resort). Values of 1000–5000 $/MWh for the first tier and 5000–20000 $/MWh
for the final tier are common in practice.

Setting the deficit cost too low relative to thermal generation costs will cause
the solver to prefer deficit over building reserves, which misrepresents the cost
of unserved energy. Setting it too high can cause numerical conditioning issues in
the LP; in practice, values above 100 000 $/MWh are rarely necessary.

---

## Lines

Transmission lines connect pairs of buses and impose flow limits on power transfer
between them. Lines are defined in `system/lines.json` under a top-level `"lines"`
array. A single-bus system has an empty lines array.

### JSON Schema

The following example shows a two-bus system with a single connecting line:

```json
{
  "lines": [
    {
      "id": 0,
      "name": "North-South Interconnection",
      "source_bus_id": 0,
      "target_bus_id": 1,
      "entry_stage_id": null,
      "exit_stage_id": null,
      "capacity": {
        "direct_mw": 1000.0,
        "reverse_mw": 800.0
      },
      "losses_percent": 2.5,
      "exchange_cost": 1.0
    }
  ]
}
```

This line allows up to 1000 MW to flow from bus 0 to bus 1, and up to 800 MW in
the reverse direction. A 2.5% transmission loss is applied to all flow. The
`exchange_cost` is an optional per-line override of the global value from
`penalties.json` — it is a regularization penalty, not a physical cost.

### Core Fields

| Field                 | Type            | Required | Description                                                                                                                 |
| --------------------- | --------------- | -------- | --------------------------------------------------------------------------------------------------------------------------- |
| `id`                  | integer         | Yes      | Unique non-negative integer identifier. Must be unique across all lines.                                                    |
| `name`                | string          | Yes      | Human-readable line name. Used in output files, validation messages, and log output.                                        |
| `source_bus_id`       | integer         | Yes      | Bus ID at the source end. Defines the "direct" flow direction. Must match an `id` in `buses.json`.                          |
| `target_bus_id`       | integer         | Yes      | Bus ID at the target end. Must match an `id` in `buses.json`. Must differ from `source_bus_id`.                             |
| `entry_stage_id`      | integer or null | No       | Stage at which the line enters service (inclusive). `null` means available from stage 0.                                    |
| `exit_stage_id`       | integer or null | No       | Stage at which the line is decommissioned (inclusive). `null` means never decommissioned.                                   |
| `capacity.direct_mw`  | number          | Yes      | Maximum flow from source to target [MW]. Hard upper bound on the flow variable.                                             |
| `capacity.reverse_mw` | number          | Yes      | Maximum flow from target to source [MW]. Hard upper bound on the reverse flow variable.                                     |
| `losses_percent`      | number          | No       | Transmission losses as a percentage of transmitted power (e.g., `2.5` means 2.5%). Defaults to `0.0` for lossless transfer. |
| `exchange_cost`       | number          | No       | Regularization penalty per MWh of flow [$/MWh]. Overrides the global default from `penalties.json`. See note below.         |

### Exchange Cost Note

The `exchange_cost` is not a tariff or a physical transmission cost — it is a
**regularization penalty** added to the LP objective to give the solver a strict
preference between equivalent dispatch solutions. Without any exchange cost, the
solver is indifferent between using or not using a lossless, uncongested line,
which can cause oscillations between equivalent solutions across iterations.

A small exchange cost (0.5--2.0 $/MWh) breaks this degeneracy without meaningfully
distorting the economic dispatch. The global default is set in `penalties.json`
under `line.exchange_cost`. Per-line overrides are supported via the optional
`exchange_cost` field on each line object, which takes precedence over the global
default. Lines without an explicit `exchange_cost` use the global value.

---

## Transmission Losses

When `losses_percent` is non-zero, the power arriving at the target bus is less
than the power leaving the source bus. If bus A sends `F` MW to bus B over a line
with 2.5% losses, then:

- Bus A's balance sees an outflow of `F` MW
- Bus B's balance sees an inflow of `F * (1 - 0.025) = 0.975 * F` MW

The lost power (0.025 \* F MW) does not appear anywhere in the network — it
represents heat dissipated in the conductor. From the LP's perspective, losses
increase the effective cost of transferring power: the source bus must generate
more to deliver the same amount at the target bus.

Setting `losses_percent: 0.0` models a lossless (superconductive) connection.
This is appropriate for short, high-voltage DC links or for cases where transmission
losses are not a modeling concern.

---

## Single-Bus vs Multi-Bus

### When to use a single-bus model

A single bus (copper-plate) is appropriate when:

- You are building an initial case and want to isolate dispatch economics from
  network effects
- Transmission constraints are not binding in the scenarios you are studying
- The system is geographically compact with ample interconnection capacity
- You are validating the stochastic model before adding network complexity

The `1dtoy` template is a single-bus case. All generators and loads connect to
bus 0 (`SIN`), and `lines.json` contains an empty array.

### When to use a multi-bus model

A multi-bus model is appropriate when:

- Different regions have distinct generation mixes and load profiles
- Transmission capacity is a binding constraint that affects dispatch or pricing
- You need locational marginal prices for investment decisions or contract pricing
- You are modeling a system where curtailment of cheap generation (wind in one
  region, hydro in another) is caused by transmission congestion

### Adding a second bus

To extend the `1dtoy` template to two buses, add a second bus to `buses.json`:

```json
{
  "buses": [
    { "id": 0, "name": "North" },
    { "id": 1, "name": "South" }
  ]
}
```

Then add a line to `lines.json`:

```json
{
  "lines": [
    {
      "id": 0,
      "name": "North-South",
      "source_bus_id": 0,
      "target_bus_id": 1,
      "capacity": {
        "direct_mw": 500.0,
        "reverse_mw": 500.0
      },
      "losses_percent": 1.0,
      "exchange_cost": 1.0
    }
  ]
}
```

Assign each generator and load to the appropriate bus by setting its `bus_id`.
When you run `cobre validate`, the validator will confirm that all `bus_id`
references resolve to existing buses.

---

## Validation Rules

Cobre's five-layer validation pipeline checks the following conditions for buses
and lines. Violations are reported as error messages with the failing entity's `id`.

| Rule                      | Error Class          | Description                                                                                                   |
| ------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------- |
| Bus reference integrity   | Reference error      | Every `bus_id` on any entity (hydro, thermal, contract, line, etc.) must match an `id` in `buses.json`.       |
| Line source bus existence | Reference error      | `source_bus_id` on each line must match an `id` in `buses.json`.                                              |
| Line target bus existence | Reference error      | `target_bus_id` on each line must match an `id` in `buses.json`.                                              |
| No self-loops             | Physical feasibility | `source_bus_id` and `target_bus_id` must differ on every line. A line from a bus to itself is not meaningful. |
| Deficit segment ordering  | Physical feasibility | Deficit segments must be listed with ascending costs. The final segment must have `depth_mw: null`.           |
| Unbounded final segment   | Physical feasibility | The last entry in every `deficit_segments` array must have `depth_mw: null` to guarantee LP feasibility.      |
| Non-negative capacity     | Physical feasibility | `capacity.direct_mw` and `capacity.reverse_mw` must be non-negative.                                          |
| Non-negative losses       | Physical feasibility | `losses_percent` must be in the range [0, 100).                                                               |

When a bus ID referenced by a generator does not exist in `buses.json`, the
validator reports the error as:

```
reference error: thermal 2 references bus 99 which does not exist
```

Fix the `bus_id` or add the missing bus and re-run `cobre validate` until the
exit code is 0.

---

## Related Pages

- [System Modeling](./system-modeling.md) — overview of all entity types and how they compose the LP
- [Anatomy of a Case](../tutorial/anatomy-of-a-case.md) — walkthrough of the complete `1dtoy` case including `buses.json` and `lines.json`
- [Building a System](../tutorial/building-a-system.md) — step-by-step guide to creating buses and lines from scratch
- [Case Format Reference](../reference/case-format.md) — complete JSON schema for all input files
