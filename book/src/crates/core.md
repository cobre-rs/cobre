# cobre-core

<span class="status-experimental">experimental</span>

`cobre-core` is the shared data model for the Cobre ecosystem. It defines the
fundamental entity types used across all crates: buses, transmission lines,
hydro plants, thermal units, energy contracts, pumping stations, and
non-controllable sources. Every other Cobre crate consumes `cobre-core` types
by shared reference; no crate other than `cobre-io` constructs `System` values.

The crate has no solver, optimizer, or I/O dependencies. It holds pure data
structures, the `System` container that groups them, derived topology graphs,
and penalty resolution utilities.

## Design principles

**Clarity-first representation.** `cobre-core` stores entities in the form most
readable to a human engineer: nested JSON concepts are flattened into named
fields with explicit unit suffixes, optional sub-models appear as `Option<Enum>`
variants, and every `f64` field carries a unit in its name and doc comment.
Performance-adapted views (packed arrays, LP variable indices) live in `cobre-sddp`,
not here.

**Validate at construction.** The `SystemBuilder` catches invalid states during
construction -- duplicate IDs, broken cross-references, cascade cycles, and
invalid filling configurations -- so the rest of the system receives a
structurally sound `System` with no need for defensive checks at solve time.

**Declaration-order invariance.** Entity collections are stored in canonical
ID-sorted order. Any `System` built from the same entities produces bit-for-bit
identical results regardless of the order in which entities were supplied to
`SystemBuilder`. Integration tests verify this property explicitly.

**Thread-safe and immutable after construction.** `System` is `Send + Sync`.
After `SystemBuilder::build()` returns `Ok`, the `System` is immutable and can
be shared across threads without synchronization.

## Entity types

### Fully modeled entities

These four entity types contribute LP variables and constraints in the SDDP
training and simulation passes.

#### Bus

An electrical network node where power balance is maintained.

| Field              | Type                  | Description                                      |
| ------------------ | --------------------- | ------------------------------------------------ |
| `id`               | `EntityId`            | Unique bus identifier                            |
| `name`             | `String`              | Human-readable name                              |
| `deficit_segments` | `Vec<DeficitSegment>` | Pre-resolved piecewise-linear deficit cost curve |
| `excess_cost`      | `f64`                 | Cost per MWh for surplus generation absorption   |

`DeficitSegment` has two fields: `depth_mw: Option<f64>` (the MW capacity of
the segment; `None` for the final unbounded segment) and `cost_per_mwh: f64`
(the marginal cost in that segment). Segments are ordered by ascending cost.
The final segment always has `depth_mw = None` to ensure LP feasibility.

#### Line

A transmission interconnection between two buses.

| Field                 | Type          | Description                                     |
| --------------------- | ------------- | ----------------------------------------------- |
| `id`                  | `EntityId`    | Unique line identifier                          |
| `name`                | `String`      | Human-readable name                             |
| `source_bus_id`       | `EntityId`    | Source bus for the direct flow direction        |
| `target_bus_id`       | `EntityId`    | Target bus for the direct flow direction        |
| `entry_stage_id`      | `Option<i32>` | Stage when line enters service; `None` = always |
| `exit_stage_id`       | `Option<i32>` | Stage when line is retired; `None` = never      |
| `direct_capacity_mw`  | `f64`         | Maximum MW flow from source to target           |
| `reverse_capacity_mw` | `f64`         | Maximum MW flow from target to source           |
| `losses_percent`      | `f64`         | Transmission losses as a percentage             |
| `exchange_cost`       | `f64`         | Regularization cost per MWh exchanged           |

Line flow is a hard constraint; the `exchange_cost` is a regularization term,
not a violation penalty.

#### Thermal

A thermal power plant with a piecewise-linear generation cost curve.

| Field               | Type                      | Description                                       |
| ------------------- | ------------------------- | ------------------------------------------------- |
| `id`                | `EntityId`                | Unique thermal plant identifier                   |
| `name`              | `String`                  | Human-readable name                               |
| `bus_id`            | `EntityId`                | Bus receiving this plant's generation             |
| `entry_stage_id`    | `Option<i32>`             | Stage when plant enters service; `None` = always  |
| `exit_stage_id`     | `Option<i32>`             | Stage when plant is retired; `None` = never       |
| `cost_segments`     | `Vec<ThermalCostSegment>` | Piecewise-linear cost curve, ascending cost order |
| `min_generation_mw` | `f64`                     | Minimum stable load                               |
| `max_generation_mw` | `f64`                     | Installed capacity                                |
| `gnl_config`        | `Option<GnlConfig>`       | GNL dispatch anticipation; `None` = no lag        |

`ThermalCostSegment` holds `capacity_mw: f64` and `cost_per_mwh: f64`.
`GnlConfig` holds `lag_stages: i32` (number of stages of dispatch anticipation
for liquefied natural gas units that require advance scheduling).

#### Hydro

The most complex entity type: a hydroelectric plant with a reservoir, turbines,
and optional cascade connectivity. It has 22 fields.

**Identity and connectivity:**

| Field            | Type               | Description                                         |
| ---------------- | ------------------ | --------------------------------------------------- |
| `id`             | `EntityId`         | Unique plant identifier                             |
| `name`           | `String`           | Human-readable name                                 |
| `bus_id`         | `EntityId`         | Bus receiving this plant's electrical generation    |
| `downstream_id`  | `Option<EntityId>` | Downstream plant in cascade; `None` = terminal node |
| `entry_stage_id` | `Option<i32>`      | Stage when plant enters service; `None` = always    |
| `exit_stage_id`  | `Option<i32>`      | Stage when plant is retired; `None` = never         |

**Reservoir and outflow:**

| Field             | Type          | Description                                       |
| ----------------- | ------------- | ------------------------------------------------- |
| `min_storage_hm3` | `f64`         | Minimum operational storage (dead volume)         |
| `max_storage_hm3` | `f64`         | Maximum operational storage (flood control level) |
| `min_outflow_m3s` | `f64`         | Minimum total outflow at all times                |
| `max_outflow_m3s` | `Option<f64>` | Maximum total outflow; `None` = no upper bound    |

**Turbine:**

| Field               | Type                   | Description                                        |
| ------------------- | ---------------------- | -------------------------------------------------- |
| `generation_model`  | `HydroGenerationModel` | Production function variant                        |
| `min_turbined_m3s`  | `f64`                  | Minimum turbined flow                              |
| `max_turbined_m3s`  | `f64`                  | Maximum turbined flow (installed turbine capacity) |
| `min_generation_mw` | `f64`                  | Minimum electrical generation                      |
| `max_generation_mw` | `f64`                  | Maximum electrical generation (installed capacity) |

**Optional hydraulic sub-models:**

| Field                         | Type                           | Description                                             |
| ----------------------------- | ------------------------------ | ------------------------------------------------------- |
| `tailrace`                    | `Option<TailraceModel>`        | Downstream water level model; `None` = zero             |
| `hydraulic_losses`            | `Option<HydraulicLossesModel>` | Penstock loss model; `None` = lossless                  |
| `efficiency`                  | `Option<EfficiencyModel>`      | Turbine efficiency model; `None` = 100%                 |
| `evaporation_coefficients_mm` | `Option<[f64; 12]>`            | Monthly evaporation [mm/month]; `None` = no evaporation |
| `diversion`                   | `Option<DiversionChannel>`     | Diversion channel; `None` = no diversion                |
| `filling`                     | `Option<FillingConfig>`        | Filling operation config; `None` = no filling           |

**Penalties:**

| Field       | Type             | Description                                               |
| ----------- | ---------------- | --------------------------------------------------------- |
| `penalties` | `HydroPenalties` | Pre-resolved penalty costs from the global-entity cascade |

### Stub entities

These three entity types are data-complete in Phase 1 but contribute no LP
variables or constraints in the minimal viable solver. Their type definitions
exist in the registry so that LP construction code can iterate over all entity
types from the start without requiring special cases.

#### PumpingStation

Transfers water between hydro reservoirs while consuming electrical power.
Fields: `id`, `name`, `bus_id`, `source_hydro_id`, `destination_hydro_id`,
`entry_stage_id`, `exit_stage_id`, `consumption_mw_per_m3s`, `min_flow_m3s`,
`max_flow_m3s`.

#### EnergyContract

A bilateral energy agreement with an entity outside the modeled system.
Fields: `id`, `name`, `bus_id`, `contract_type` (`ContractType::Import` or
`ContractType::Export`), `entry_stage_id`, `exit_stage_id`, `price_per_mwh`,
`min_mw`, `max_mw`. Negative `price_per_mwh` represents export revenue.

#### NonControllableSource

Intermittent generation (wind, solar, run-of-river) that cannot be dispatched.
Fields: `id`, `name`, `bus_id`, `entry_stage_id`, `exit_stage_id`,
`max_generation_mw`, `curtailment_cost` (pre-resolved).

## Supporting types

### Enums

| Enum                   | Variants                                                                                                 | Purpose                                               |
| ---------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `HydroGenerationModel` | `ConstantProductivity { productivity_mw_per_m3s }`, `LinearizedHead { productivity_mw_per_m3s }`, `Fpha` | Production function for turbine power computation     |
| `TailraceModel`        | `Polynomial { coefficients: Vec<f64> }`, `Piecewise { points: Vec<TailracePoint> }`                      | Downstream water level as a function of total outflow |
| `HydraulicLossesModel` | `Factor { value }`, `Constant { value_m }`                                                               | Head loss in penstock and draft tube                  |
| `EfficiencyModel`      | `Constant { value }`                                                                                     | Turbine-generator efficiency                          |
| `ContractType`         | `Import`, `Export`                                                                                       | Energy flow direction for bilateral contracts         |

`ConstantProductivity` is used in both SDDP training and simulation.
`LinearizedHead` is used in simulation only (not SDDP training), because the
head-dependent term introduces a bilinear product in the training LP.
`Fpha` is the full production function with head-area-productivity tables.

### Structs

| Struct               | Fields                                           | Purpose                                                |
| -------------------- | ------------------------------------------------ | ------------------------------------------------------ |
| `TailracePoint`      | `outflow_m3s: f64`, `height_m: f64`              | One breakpoint on a piecewise tailrace curve           |
| `DeficitSegment`     | `depth_mw: Option<f64>`, `cost_per_mwh: f64`     | One segment of a piecewise deficit cost curve          |
| `ThermalCostSegment` | `capacity_mw: f64`, `cost_per_mwh: f64`          | One segment of a thermal generation cost curve         |
| `GnlConfig`          | `lag_stages: i32`                                | Dispatch anticipation lag for GNL thermal units        |
| `DiversionChannel`   | `downstream_id: EntityId`, `max_flow_m3s: f64`   | Water diversion bypassing turbines and spillways       |
| `FillingConfig`      | `start_stage_id: i32`, `filling_inflow_m3s: f64` | Reservoir filling operation from a fixed inflow source |
| `HydroPenalties`     | 11 `f64` fields (see Penalty resolution section) | Pre-resolved penalty costs for one hydro plant         |

## EntityId

`EntityId` is a newtype wrapper around `i32`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityId(pub i32);
```

**Why `i32`, not `String`.** All JSON entity schemas use integer IDs. Integer
keys are cheaper to hash, compare, and copy than strings. `EntityId` appears in
every lookup index and cross-reference field, so this is a high-frequency type.
If a future input format requires string IDs, the newtype boundary isolates the
change to `EntityId`'s internal representation and its `From`/`Into` impls.

**Why no `Ord`.** Entity ordering is always by inner `i32` value (canonical
ID order), but the spec deliberately omits `Ord` to prevent accidental use of
lexicographic ordering in contexts that expect ID-based ordering. Sort sites use
`sort_by_key(|e| e.id.0)` explicitly, making the intent visible at each call
site.

Construction and conversion:

```rust
use cobre_core::EntityId;

let id: EntityId = EntityId::from(42);
let raw: i32 = i32::from(id);
assert_eq!(id.to_string(), "42");
```

## System and SystemBuilder

`System` is the top-level in-memory representation of a validated, resolved
case. It is produced by `SystemBuilder` (directly in tests) and by
`cobre-io::load_case()` in production. It is consumed read-only by `cobre-sddp`
and `cobre-stochastic`.

```rust
use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};

let system = SystemBuilder::new()
    .buses(vec![Bus {
        id: EntityId(1),
        name: "Main Bus".to_string(),
        deficit_segments: vec![],
        excess_cost: 0.0,
    }])
    .build()
    .expect("valid system");

assert_eq!(system.n_buses(), 1);
assert!(system.bus(EntityId(1)).is_some());
```

### Validation in SystemBuilder::build()

`SystemBuilder::build()` runs four validation phases in order:

1. **Duplicate check.** Each of the 7 entity collections is scanned for
   duplicate `EntityId` values. All collections are checked before returning.
   If any duplicates are found, `build()` returns early with the error list.

2. **Cross-reference validation.** Every foreign-key field is verified against
   the appropriate collection index. Checked fields include `bus_id` on hydros,
   thermals, pumping stations, energy contracts, and non-controllable sources;
   `source_bus_id` and `target_bus_id` on lines; `downstream_id` and
   `diversion.downstream_id` on hydros; and `source_hydro_id` and
   `destination_hydro_id` on pumping stations. All broken references across all
   entity types are collected; `build()` returns early after this phase if any
   are found.

3. **Cascade topology and cycle detection.** `CascadeTopology` is built from
   the validated hydro `downstream_id` fields. If the topological sort
   (Kahn's algorithm) does not reach all hydros, the unvisited hydros form a
   cycle. Their IDs are reported in a `ValidationError::CascadeCycle` error.
   Filling configurations are also validated in this phase.

4. **Filling config validation.** Each hydro with a `FillingConfig` must have
   a positive `filling_inflow_m3s` and a non-`None` `entry_stage_id`. Violations
   produce `ValidationError::InvalidFillingConfig` errors.

If all phases pass, `build()` constructs `NetworkTopology`, builds O(1) lookup
indices for all 7 collections, and returns the immutable `System`.

The `build()` signature collects and returns all errors found across all
collections rather than short-circuiting on the first failure:

```rust
pub fn build(self) -> Result<System, Vec<ValidationError>>
```

### Canonical ordering

Before building indices, `SystemBuilder::build()` sorts every entity collection
by `entity.id.0`. The resulting `System` stores entities in this canonical order.
All accessor methods (`buses()`, `hydros()`, etc.) return slices in canonical
order. This guarantees declaration-order invariance: two `System` values built
from the same entities in different input orders are structurally identical.

## Topology

### CascadeTopology

`CascadeTopology` represents the directed forest of hydro plant cascade
relationships. It is built from the `downstream_id` fields of all hydro plants
and stored on `System`.

```rust
let cascade = system.cascade();

// Downstream plant for a given hydro (None if terminal).
let ds: Option<EntityId> = cascade.downstream(EntityId(1));

// All upstream plants for a given hydro (empty slice if headwater).
let upstream: &[EntityId] = cascade.upstream(EntityId(3));

// Topological ordering: every upstream plant appears before its downstream.
let order: &[EntityId] = cascade.topological_order();

cascade.is_headwater(EntityId(1)); // true if no upstream plants
cascade.is_terminal(EntityId(3));  // true if no downstream plant
```

The topological order is computed using Kahn's algorithm with a sorted ready
queue, ensuring determinism: within the same topological level, hydros appear
in ascending ID order.

### NetworkTopology

`NetworkTopology` provides O(1) lookups for bus-line incidence and bus-to-entity
maps. It is built from all entity collections and stored on `System`.

```rust
let network = system.network();

// Lines connected to a bus.
let connections: &[BusLineConnection] = network.bus_lines(EntityId(1));
// BusLineConnection has `line_id: EntityId` and `is_source: bool`.

// Generators connected to a bus.
let generators: &BusGenerators = network.bus_generators(EntityId(1));
// BusGenerators has `hydro_ids`, `thermal_ids`, `ncs_ids` (all Vec<EntityId>).

// Load entities connected to a bus.
let loads: &BusLoads = network.bus_loads(EntityId(1));
// BusLoads has `contract_ids` and `pumping_station_ids` (both Vec<EntityId>).
```

All ID lists in `BusGenerators` and `BusLoads` are in canonical ascending-ID
order for determinism.

## Penalty resolution

Penalty values are resolved from a three-tier cascade: global defaults,
entity-level overrides, and stage-level overrides. Phase 1 implements the first
two tiers. Stage-varying overrides are a Phase 2 concern.

`GlobalPenaltyDefaults` holds system-wide fallback values for all penalty fields:

```rust
pub struct GlobalPenaltyDefaults {
    pub bus_deficit_segments: Vec<DeficitSegment>,
    pub bus_excess_cost: f64,
    pub line_exchange_cost: f64,
    pub hydro: HydroPenalties,
    pub ncs_curtailment_cost: f64,
}
```

The five resolution functions each accept an optional entity-level override and
the global defaults, returning the resolved value:

```rust
// Returns entity segments if present, else global defaults.
let segments = resolve_bus_deficit_segments(&entity_override, &global);

// Returns entity value if Some, else global default.
let cost    = resolve_bus_excess_cost(entity_override, &global);
let cost    = resolve_line_exchange_cost(entity_override, &global);
let cost    = resolve_ncs_curtailment_cost(entity_override, &global);

// Resolves all 11 hydro penalty fields field-by-field.
let hydro_p = resolve_hydro_penalties(&entity_overrides, &global);
```

`HydroPenalties` holds 11 pre-resolved `f64` fields:

| Field                             | Unit   | Description                                        |
| --------------------------------- | ------ | -------------------------------------------------- |
| `spillage_cost`                   | $/m³/s | Penalty per m³/s of spillage                       |
| `diversion_cost`                  | $/m³/s | Penalty per m³/s exceeding diversion channel limit |
| `fpha_turbined_cost`              | $/MWh  | Regularization cost for FPHA turbined flow         |
| `storage_violation_below_cost`    | $/hm³  | Penalty per hm³ of storage below minimum           |
| `filling_target_violation_cost`   | $/hm³  | Penalty per hm³ below filling target               |
| `turbined_violation_below_cost`   | $/m³/s | Penalty per m³/s of turbined flow below minimum    |
| `outflow_violation_below_cost`    | $/m³/s | Penalty per m³/s of total outflow below minimum    |
| `outflow_violation_above_cost`    | $/m³/s | Penalty per m³/s of total outflow above maximum    |
| `generation_violation_below_cost` | $/MW   | Penalty per MW of generation below minimum         |
| `evaporation_violation_cost`      | $/mm   | Penalty per mm of evaporation constraint violation |
| `water_withdrawal_violation_cost` | $/m³/s | Penalty per m³/s of water withdrawal violation     |

The optional `HydroPenaltyOverrides` struct mirrors `HydroPenalties` with all
fields as `Option<f64>`. It is an intermediate type used during case loading;
the resolved `HydroPenalties` (with no `Option`s) is what is stored on each
`Hydro` entity.

## Validation errors

`ValidationError` is the error type returned by `SystemBuilder::build()`:

| Variant                | Meaning                                                                        |
| ---------------------- | ------------------------------------------------------------------------------ |
| `DuplicateId`          | Two entities in the same collection share an `EntityId`                        |
| `InvalidReference`     | A cross-reference field points to an ID that does not exist                    |
| `CascadeCycle`         | The hydro `downstream_id` graph contains a cycle                               |
| `InvalidFillingConfig` | A hydro's filling configuration has non-positive inflow or no `entry_stage_id` |
| `DisconnectedBus`      | A bus has no lines, generators, or loads (reserved for Phase 2 validation)     |
| `InvalidPenalty`       | An entity-level penalty value is invalid (e.g., negative cost)                 |

All variants implement `Display` and the standard `Error` trait. The error
message includes the entity type, the offending ID, and (for reference errors)
the field name and the missing referenced ID.

```rust
use cobre_core::{EntityId, ValidationError};

let err = ValidationError::InvalidReference {
    source_entity_type: "Hydro",
    source_id: EntityId(3),
    field_name: "bus_id",
    referenced_id: EntityId(99),
    expected_type: "Bus",
};
// "Hydro with id 3 has invalid cross-reference in field 'bus_id': referenced Bus id 99 does not exist"
println!("{err}");
```

## Public API summary

`System` exposes four categories of methods:

**Collection accessors** (return `&[T]` in canonical ID order):
`buses()`, `lines()`, `hydros()`, `thermals()`, `pumping_stations()`,
`contracts()`, `non_controllable_sources()`

**Count queries** (return `usize`):
`n_buses()`, `n_lines()`, `n_hydros()`, `n_thermals()`,
`n_pumping_stations()`, `n_contracts()`, `n_non_controllable_sources()`

**Entity lookup by ID** (return `Option<&T>`):
`bus(id)`, `line(id)`, `hydro(id)`, `thermal(id)`, `pumping_station(id)`,
`contract(id)`, `non_controllable_source(id)` -- each is O(1) via a
`HashMap<EntityId, usize>` index into the canonical collection.

**Topology accessors** (return references to derived structures):
`cascade()` returns `&CascadeTopology`,
`network()` returns `&NetworkTopology`.

For full method signatures and rustdoc, run:

```
cargo doc --workspace --no-deps --open
```

For the theoretical underpinning of the entity model, generation models, and
penalty system, see the
[methodology reference](https://cobre-rs.github.io/cobre-docs/).
