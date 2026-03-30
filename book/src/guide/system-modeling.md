# System Modeling

A Cobre case describes a power system as a collection of **entities**. Each entity
represents a physical component — a bus, a generator, a transmission line — or a
contractual obligation. Together, they form the complete model that the solver turns
into a sequence of LP sub-problems, one per stage per scenario trajectory.

The fundamental organizing principle is simple: every generator and every load
connects to a **bus**. A bus is an electrical node at which the power balance
constraint must hold. At each stage and each load block, the LP enforces that the
total power injected into a bus equals the total power withdrawn from it. When the
constraint cannot be satisfied by physical generation alone, deficit slack variables
absorb the gap at a penalty cost, ensuring the LP always has a feasible solution.

Entities are grouped by type and stored in a `System` object. The `System` is built
from the case directory by `load_case`, which runs a five-layer validation pipeline
before handing the model to the solver. Within the `System`, all entity collections
are kept in canonical ID-sorted order. This ordering is an invariant: it guarantees
that simulation results are bit-for-bit identical regardless of the order entities
appear in the input files.

---

## Entity Types

Cobre models seven entity types. Five are fully implemented and contribute LP
variables and constraints. Two are registered stubs that appear in the entity
model but do not yet contribute LP variables in the current release.

| Entity Type      | Status | JSON File                              | Description                                                                                                                                        |
| ---------------- | ------ | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Bus              | Full   | `system/buses.json`                    | Electrical node. Power balance constraint per stage per block. See [Network Topology](./network-topology.md).                                      |
| Line             | Full   | `system/lines.json`                    | Transmission interconnection between two buses with flow limits and losses. See [Network Topology](./network-topology.md).                         |
| Hydro            | Full   | `system/hydros.json`                   | Reservoir-turbine-spillway system with cascade linkage. See [Hydro Plants](./hydro-plants.md).                                                     |
| Thermal          | Full   | `system/thermals.json`                 | Dispatchable generator with piecewise-linear cost curve. See [Thermal Units](./thermal-units.md).                                                  |
| Contract         | Stub   | `system/energy_contracts.json`         | Energy purchase or sale obligation. Entity exists in registry; no LP variables in this release.                                                    |
| Pumping Station  | Stub   | `system/pumping_stations.json`         | Pumped-storage or water-transfer station. Entity exists in registry; no LP variables in this release.                                              |
| Non-Controllable | Full   | `system/non_controllable_sources.json` | Variable renewable source (wind, solar, run-of-river). Generation variable bounded by available capacity × block factor, with curtailment penalty. |

The two remaining stub types (Contract and Pumping Station) are registered in
the entity model so that LP construction code can iterate over all seven types
consistently. Adding LP contributions for these entities is planned for future
releases.

---

## Non-Controllable Sources

A non-controllable source (NCS) represents a variable renewable generator whose
output is externally specified rather than optimized by the solver. Typical
examples include wind farms, utility-scale solar arrays, and run-of-river hydro
units without significant storage. The solver dispatches the NCS at its full
available capacity unless doing so would oversupply the bus, in which case
curtailment occurs and the solver pays a curtailment penalty.

Each NCS contributes one generation LP variable per block, bounded by:

```
0 <= generation_mw <= available_generation_mw * block_factor
```

where `available_generation_mw` comes from `constraints/ncs_bounds.parquet`
(with `system/non_controllable_sources.json` providing the base value) and
`block_factor` from `scenarios/non_controllable_factors.json` (default 1.0).

When `scenarios/non_controllable_stats.parquet` is present, NCS availability
becomes stochastic: each forward and backward pass scenario draws a random
availability factor and the LP column upper bound varies per scenario. See
[Stochastic Modeling](./stochastic-modeling.md#stochastic-ncs-availability)
for details.

The objective coefficient is `-curtailment_cost * block_hours`, making it
cheaper to generate than to curtail. The NCS generation variable injects +1.0 MW
at its connected bus in the power balance constraint, identical to a thermal plant.

Simulation output is written to `simulation/non_controllables/` with columns
for `generation_mw`, `available_mw`, `curtailment_mw`, and `curtailment_cost`
per (stage, block, source) triplet. See the
[Output Format Reference](../reference/output-format.md) for the complete schema.

---

## How Entities Connect

The network is **bus-centric**. Every entity that produces or consumes power is
attached to a bus via a `bus_id` field:

```
   Hydro ──┐
           │ inject
  Thermal ─┤
           ├──> Bus <──── Line ────> Bus
  NCS ─────┘
                │
               load
                │
           Contract
         Pumping Station
```

At each stage and load block, the LP enforces the bus balance constraint:

```
  sum(generation at bus) + sum(imports from lines) + deficit
    = load_demand + sum(exports to lines) + excess
```

Deficit and excess slack variables absorb imbalance at a penalty cost, ensuring
the LP is always feasible. When the deficit penalty is high enough relative to
the cost of available generation, the solver will prefer to generate rather than
incur deficit.

**Cascade topology** governs hydro plant interactions. A hydro plant with a non-null
`downstream_id` sends all of its outflow — turbined flow plus spillage — into the
downstream plant's reservoir at the same stage. The cascade forms a directed forest:
multiple upstream plants may flow into a single downstream plant, but no cycles
are allowed. Water balance is computed in topological order — upstream plants first,
downstream plants last — in a single pass per stage.

---

## Declaration-Order Invariance

The order in which entities appear in the JSON input files does not affect results.
Cobre reads all entities from their files, then sorts each collection by entity ID
before building the `System`. Every function that processes entity collections
operates on this canonical sorted order.

This invariant has a practical consequence: you can rearrange entries in
`buses.json`, `hydros.json`, or any other entity file without changing the
simulation output. You can also add new entities with lower IDs than existing ones
without disturbing results for the existing entities.

---

## Penalties and Soft Constraints

LP solvers require feasible problems. Physical constraints — minimum outflow, minimum
turbined flow, reservoir bounds — can become infeasible under extreme stochastic
scenarios (very low inflow, very high load). Cobre handles this by making nearly
every physical constraint **soft**: instead of a hard infeasibility, the solver pays
a penalty cost to violate the constraint by a small amount.

Penalties are set at three levels, resolved from most specific to most general:

1. **Stage-level override** — penalty files for individual stages, when present
2. **Entity-level override** — a `penalties` block inside the entity's JSON object
3. **Global default** — the top-level `penalties.json` file in the case directory

This three-tier cascade gives you precise control: you can set a strict global
spillage penalty and then relax it for a specific plant that is known to spill
frequently in wet years. For details on the penalty fields for each entity type,
see the [Configuration](./configuration.md) guide and the
[Case Format Reference](../reference/case-format.md).

The bus deficit segments are the most important penalty to configure correctly.
A deficit cost that is too low makes the solver prefer deficit over building
generation capacity; a cost that is too high (or an unbounded segment that is
absent) can cause numerical instability. The final deficit segment must always
have `depth_mw: null` (unbounded) to guarantee LP feasibility.

---

## Entity Lifecycle

Entities can enter service or be decommissioned at specified stages using
`entry_stage_id` and `exit_stage_id` fields:

| Field            | Type            | Meaning                                                                                      |
| ---------------- | --------------- | -------------------------------------------------------------------------------------------- |
| `entry_stage_id` | integer or null | Stage index at which the entity enters service (inclusive). `null` = available from stage 0  |
| `exit_stage_id`  | integer or null | Stage index at which the entity is decommissioned (inclusive). `null` = never decommissioned |

These fields are available on `Hydro`, `Thermal`, and `Line` entities. When a plant
has `entry_stage_id: 12`, the LP does not include any variables for that plant in
stages 0 through 11. From stage 12 onward, the plant appears in every sub-problem
as normal.

Lifecycle fields are useful for planning studies that span commissioning or retirement
events: new thermal plants coming online mid-horizon, or aging hydro units being
decommissioned. Each lifecycle event is validated to ensure that `entry_stage_id`
falls within the stage range defined in `stages.json`.

---

## Related Pages

- [Hydro Plants](./hydro-plants.md) — complete field reference for `system/hydros.json`
- [Thermal Units](./thermal-units.md) — complete field reference for `system/thermals.json`
- [Network Topology](./network-topology.md) — buses, lines, deficit modeling, and transmission
- [Anatomy of a Case](../tutorial/anatomy-of-a-case.md) — walkthrough of every file in the `1dtoy` example
- [Case Format Reference](../reference/case-format.md) — complete JSON schema for all input files
