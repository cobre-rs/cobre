# cobre-core

<span class="status-alpha">alpha</span>

`cobre-core` is the shared data model for the Cobre ecosystem. It defines the
fundamental entity types used across all crates: buses, transmission lines,
hydro plants, thermal units, energy contracts, pumping stations, and
non-controllable sources. Every other Cobre crate consumes `cobre-core` types
by shared reference; no crate other than `cobre-io` constructs `System` values.

The crate has no solver, optimizer, or I/O dependencies. It holds pure data
structures, the `System` container that groups them, derived topology graphs,
penalty resolution utilities, temporal types, scenario pipeline types, initial
conditions, generic constraints, and pre-resolved penalty/bound tables.

## Module overview

| Module               | Purpose                                                      |
| -------------------- | ------------------------------------------------------------ |
| `entities`           | Entity types: Bus, Line, Hydro, Thermal, and stub types      |
| `entity_id`          | `EntityId` newtype wrapper                                   |
| `error`              | `ValidationError` enum                                       |
| `generic_constraint` | User-defined linear constraints over LP variables            |
| `initial_conditions` | Reservoir storage levels at study start                      |
| `penalty`            | Global defaults, entity overrides, and resolution functions  |
| `resolved`           | Pre-resolved penalty/bound tables with O(1) lookup           |
| `scenario`           | PAR model parameters, load statistics, and correlation model |
| `system`             | `System` container and `SystemBuilder`                       |
| `temporal`           | Stages, blocks, seasons, and the policy graph                |
| `topology`           | `CascadeTopology` and `NetworkTopology` derived structures   |

## Design principles

**Clarity-first representation.** `cobre-core` stores entities in the form most
readable to a human engineer: nested JSON concepts are flattened into named
fields with explicit unit suffixes, optional sub-models appear as `Option<Enum>`
variants, and every `f64` field carries a unit in its name and doc comment.
Performance-adapted views (packed arrays, LP variable indices) live in downstream solver crates,
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

These four entity types contribute LP variables and constraints in optimization
and simulation procedures.

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

These three entity types are data-complete but do not contribute LP variables or
constraints in the minimal viable implementation. Their type definitions exist in
the registry so analysis code can iterate over all entity types uniformly.

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

`ConstantProductivity` is used universally and is the minimal viable model.
`LinearizedHead` is for high-fidelity analyses where head-dependent terms matter.
`Fpha` is the full production function with head-area-productivity tables for detailed modeling.

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
`cobre-io::load_case()` in production. It is consumed read-only by downstream solver and analysis crates.

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
entity-level overrides, and stage-level overrides. The first two tiers are
implemented in Phase 1. Stage-varying overrides are deferred to Phase 2.

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

## Temporal model

The `temporal` module defines the time structure of a multi-stage stochastic
optimization problem. These types are loaded from `stages.json` by `cobre-io`
and stored on `System`.

There are 13 types in total: 5 enums and 8 structs.

### Enums

| Enum              | Variants                                           | Purpose                                                   |
| ----------------- | -------------------------------------------------- | --------------------------------------------------------- |
| `BlockMode`       | `Parallel`, `Chronological`                        | How blocks within a stage relate in the LP                |
| `SeasonCycleType` | `Monthly`, `Weekly`, `Custom`                      | How season IDs map to calendar periods                    |
| `NoiseMethod`     | `Saa`, `Lhs`, `QmcSobol`, `QmcHalton`, `Selective` | Opening tree noise generation algorithm                   |
| `PolicyGraphType` | `FiniteHorizon`, `Cyclic`                          | Whether the study horizon is acyclic or infinite-periodic |
| `StageRiskConfig` | `Expectation`, `CVaR { alpha, lambda }`            | Per-stage risk measure configuration                      |

`BlockMode::Parallel` is the default: blocks are independent sub-periods solved
simultaneously, with water balance aggregated across all blocks in the stage.
`BlockMode::Chronological` enables intra-stage storage dynamics (daily cycling).

`PolicyGraphType::FiniteHorizon` is the minimal viable solver choice: an acyclic
stage chain with zero terminal value. `Cyclic` requires a positive
`annual_discount_rate` for convergence.

### Block

A load block within a stage, representing a sub-period with uniform demand and
generation characteristics.

| Field            | Type     | Description                                            |
| ---------------- | -------- | ------------------------------------------------------ |
| `index`          | `usize`  | 0-based index within the parent stage (0, 1, ..., n-1) |
| `name`           | `String` | Human-readable block label (e.g., "PEAK", "OFF-PEAK")  |
| `duration_hours` | `f64`    | Duration of this block in hours; must be positive      |

The block weight (fraction of stage duration) is derived on demand as
`duration_hours / sum(all block hours in stage)` and is not stored.

### StageStateConfig

Flags controlling which variables carry state between stages.

| Field         | Type   | Default | Description                                                    |
| ------------- | ------ | ------- | -------------------------------------------------------------- |
| `storage`     | `bool` | `true`  | Whether reservoir storage volumes are state variables          |
| `inflow_lags` | `bool` | `false` | Whether past inflow realizations (AR lags) are state variables |

`inflow_lags` must be `true` when the PAR model order `p > 0` and inflow lag
cuts are enabled.

### ScenarioSourceConfig

Per-stage scenario generation configuration.

| Field              | Type          | Description                                                |
| ------------------ | ------------- | ---------------------------------------------------------- |
| `branching_factor` | `usize`       | Number of noise realizations per stage; must be positive   |
| `noise_method`     | `NoiseMethod` | Algorithm for generating noise vectors in the opening tree |

`branching_factor` is the per-stage branching factor for both the opening tree
and the forward pass. `noise_method` is orthogonal to `SamplingScheme` (which
selects the forward-pass noise source); it governs how the backward-pass opening
tree is produced.

### Stage

A single stage in the multi-stage stochastic problem, partitioning the study
horizon into decision periods.

| Field             | Type                   | Description                                                      |
| ----------------- | ---------------------- | ---------------------------------------------------------------- |
| `index`           | `usize`                | 0-based array position after canonical sort                      |
| `id`              | `i32`                  | Domain-level identifier from `stages.json`; negative = pre-study |
| `start_date`      | `NaiveDate`            | Stage start date (inclusive), ISO 8601                           |
| `end_date`        | `NaiveDate`            | Stage end date (exclusive), ISO 8601                             |
| `season_id`       | `Option<usize>`        | Index into `SeasonMap::seasons`; `None` = no seasonal structure  |
| `blocks`          | `Vec<Block>`           | Ordered load blocks; sum of `duration_hours` = stage duration    |
| `block_mode`      | `BlockMode`            | Parallel or chronological block formulation                      |
| `state_config`    | `StageStateConfig`     | State variable flags                                             |
| `risk_config`     | `StageRiskConfig`      | Risk measure for this stage                                      |
| `scenario_config` | `ScenarioSourceConfig` | Branching factor and noise method                                |

Pre-study stages (negative `id`) carry only `id`, `start_date`, `end_date`, and
`season_id`. Their `blocks`, `risk_config`, and `scenario_config` fields are
unused.

```rust
use chrono::NaiveDate;
use cobre_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage,
    StageRiskConfig, StageStateConfig,
};

let stage = Stage {
    index: 0,
    id: 1,
    start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
    end_date:   NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
    season_id:  Some(0),
    blocks: vec![Block {
        index: 0,
        name: "SINGLE".to_string(),
        duration_hours: 744.0,
    }],
    block_mode: BlockMode::Parallel,
    state_config: StageStateConfig { storage: true, inflow_lags: false },
    risk_config: StageRiskConfig::Expectation,
    scenario_config: ScenarioSourceConfig {
        branching_factor: 50,
        noise_method: NoiseMethod::Saa,
    },
};
```

### SeasonDefinition and SeasonMap

Season definitions map season IDs to calendar periods for PAR model coefficient
lookup and inflow history aggregation.

`SeasonDefinition` fields:

| Field         | Type          | Description                                              |
| ------------- | ------------- | -------------------------------------------------------- |
| `id`          | `usize`       | 0-based season index (0-11 for monthly, 0-51 for weekly) |
| `label`       | `String`      | Human-readable label (e.g., "January", "Wet Season")     |
| `month_start` | `u32`         | Calendar month where the season starts (1-12)            |
| `day_start`   | `Option<u32>` | Calendar day start; only used for `Custom` cycle type    |
| `month_end`   | `Option<u32>` | Calendar month end; only used for `Custom` cycle type    |
| `day_end`     | `Option<u32>` | Calendar day end; only used for `Custom` cycle type      |

`SeasonMap` groups the definitions with a cycle type:

| Field        | Type                    | Description                                                |
| ------------ | ----------------------- | ---------------------------------------------------------- |
| `cycle_type` | `SeasonCycleType`       | `Monthly` (12 seasons), `Weekly` (52 seasons), or `Custom` |
| `seasons`    | `Vec<SeasonDefinition>` | Season entries sorted by `id`                              |

### Transition and PolicyGraph

`Transition` represents a directed edge in the policy graph:

| Field                           | Type          | Description                                                    |
| ------------------------------- | ------------- | -------------------------------------------------------------- |
| `source_id`                     | `i32`         | Source stage ID                                                |
| `target_id`                     | `i32`         | Target stage ID                                                |
| `probability`                   | `f64`         | Transition probability; outgoing probabilities must sum to 1.0 |
| `annual_discount_rate_override` | `Option<f64>` | Per-transition rate override; `None` = use global rate         |

`PolicyGraph` is the top-level clarity-first representation of the stage graph
loaded from `stages.json`:

| Field                  | Type                | Description                                                     |
| ---------------------- | ------------------- | --------------------------------------------------------------- |
| `graph_type`           | `PolicyGraphType`   | `FiniteHorizon` (acyclic) or `Cyclic` (infinite periodic)       |
| `annual_discount_rate` | `f64`               | Global discount rate; `0.0` = no discounting                    |
| `transitions`          | `Vec<Transition>`   | Stage transitions forming a linear chain or DAG                 |
| `season_map`           | `Option<SeasonMap>` | Season definitions; `None` when no seasonal structure is needed |

For finite horizon, transitions form a linear chain. For cyclic horizon, at
least one transition has `source_id >= target_id` (a back-edge) and the
`annual_discount_rate` must be positive for convergence.

```rust
use cobre_core::temporal::{PolicyGraph, PolicyGraphType, Transition};

let graph = PolicyGraph {
    graph_type: PolicyGraphType::FiniteHorizon,
    annual_discount_rate: 0.06,
    transitions: vec![
        Transition { source_id: 1, target_id: 2, probability: 1.0,
                     annual_discount_rate_override: None },
        Transition { source_id: 2, target_id: 3, probability: 1.0,
                     annual_discount_rate_override: Some(0.08) },
    ],
    season_map: None,
};
assert_eq!(graph.graph_type, PolicyGraphType::FiniteHorizon);
```

The solver-level `HorizonMode` enum in `cobre-sddp` is built from a `PolicyGraph`
at initialization time; it precomputes transition maps, cycle detection, and
discount factors for efficient runtime dispatch. The `PolicyGraph` in `cobre-core`
is the user-facing clarity-first representation.

## Scenario pipeline types

The `scenario` module holds clarity-first data containers for the raw scenario
pipeline parameters loaded from input files. These are raw input-facing types;
performance-adapted views (pre-computed LP arrays, Cholesky-decomposed matrices)
belong in downstream crates (`cobre-stochastic`, `cobre-sddp`).

### SamplingScheme and ScenarioSource

`SamplingScheme` selects the forward-pass noise source:

| Variant      | Description                                                          |
| ------------ | -------------------------------------------------------------------- |
| `InSample`   | Forward pass reuses the opening tree generated for the backward pass |
| `External`   | Forward pass draws from an externally supplied scenario file         |
| `Historical` | Forward pass replays historical inflow realizations                  |

`InSample` is the default and the minimal viable solver choice.

`ScenarioSource` is the top-level scenario configuration loaded from `stages.json`:

| Field             | Type                            | Description                                                  |
| ----------------- | ------------------------------- | ------------------------------------------------------------ |
| `sampling_scheme` | `SamplingScheme`                | Noise source for the forward pass                            |
| `seed`            | `Option<i64>`                   | Random seed for reproducible generation; `None` = OS entropy |
| `selection_mode`  | `Option<ExternalSelectionMode>` | Only used when `sampling_scheme` is `External`               |

`ExternalSelectionMode` has two variants: `Random` (draw uniformly at random)
and `Sequential` (replay in file order, cycling when the end is reached).

### InflowModel

Raw PAR(p) model parameters for a single (hydro, stage) pair, loaded from
`inflow_seasonal_stats.parquet` and `inflow_ar_coefficients.parquet`.

| Field             | Type       | Description                                              |
| ----------------- | ---------- | -------------------------------------------------------- |
| `hydro_id`        | `EntityId` | Hydro plant this model belongs to                        |
| `stage_id`        | `i32`      | Stage index this model applies to                        |
| `mean_m3s`        | `f64`      | Seasonal mean inflow μ [m³/s]                            |
| `std_m3s`         | `f64`      | Seasonal standard deviation σ [m³/s]                     |
| `ar_order`        | `usize`    | AR model order p; zero means white-noise inflow          |
| `ar_coefficients` | `Vec<f64>` | AR lag coefficients [ψ₁, ψ₂, …, ψₚ]; length = `ar_order` |

```rust
use cobre_core::{EntityId, scenario::InflowModel};

let model = InflowModel {
    hydro_id: EntityId(1),
    stage_id: 3,
    mean_m3s: 150.0,
    std_m3s: 30.0,
    ar_order: 2,
    ar_coefficients: vec![0.45, 0.22],
};
assert_eq!(model.ar_order, 2);
assert_eq!(model.ar_coefficients.len(), 2);
```

`System` holds a `Vec<InflowModel>` sorted by `(hydro_id, stage_id)` for
declaration-order invariance.

### LoadModel

Raw load seasonal statistics for a single (bus, stage) pair, loaded from
`load_seasonal_stats.parquet`.

| Field      | Type       | Description                                     |
| ---------- | ---------- | ----------------------------------------------- |
| `bus_id`   | `EntityId` | Bus this load model belongs to                  |
| `stage_id` | `i32`      | Stage index this model applies to               |
| `mean_mw`  | `f64`      | Seasonal mean load demand [MW]                  |
| `std_mw`   | `f64`      | Seasonal standard deviation of load demand [MW] |

Load typically has no AR structure, so no lag coefficients are stored.
`System` holds a `Vec<LoadModel>` sorted by `(bus_id, stage_id)`.

### CorrelationModel

`CorrelationModel` is the top-level correlation configuration loaded from
`correlation.json`. It holds named profiles and an optional stage-to-profile
schedule.

The type hierarchy is:

```
CorrelationModel
  └── profiles: BTreeMap<String, CorrelationProfile>
        └── groups: Vec<CorrelationGroup>
              ├── entities: Vec<CorrelationEntity>
              └── matrix: Vec<Vec<f64>>   (symmetric, row-major)
```

`CorrelationEntity` carries `entity_type: String` (currently always `"inflow"`)
and `id: EntityId`. Using `String` rather than an enum preserves forward
compatibility when additional stochastic variable types are added.

`profiles` uses `BTreeMap` rather than `HashMap` to preserve deterministic
iteration order (declaration-order invariance). Cholesky decomposition of the
correlation matrices is NOT performed here; that belongs to `cobre-stochastic`.

```rust
use std::collections::BTreeMap;
use cobre_core::{EntityId, scenario::{
    CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
}};

let mut profiles = BTreeMap::new();
profiles.insert("default".to_string(), CorrelationProfile {
    groups: vec![CorrelationGroup {
        name: "All".to_string(),
        entities: vec![
            CorrelationEntity { entity_type: "inflow".to_string(), id: EntityId(1) },
            CorrelationEntity { entity_type: "inflow".to_string(), id: EntityId(2) },
        ],
        matrix: vec![vec![1.0, 0.8], vec![0.8, 1.0]],
    }],
});

let model = CorrelationModel {
    method: "cholesky".to_string(),
    profiles,
    schedule: vec![],
};
assert!(model.profiles.contains_key("default"));
```

When `schedule` is empty, a single profile (typically named `"default"`) applies
to all stages. When `schedule` is non-empty, each entry maps a stage index to an
active profile name.

## Initial conditions and constraints

### InitialConditions

`InitialConditions` holds the reservoir storage levels at the start of the study.
It is loaded from `initial_conditions.json` by `cobre-io` and stored on `System`.

Two arrays are kept separate because filling hydros can have an initial volume
below dead storage (`min_storage_hm3`), which is not a valid operating level
for regular hydros:

| Field             | Type                | Description                                                 |
| ----------------- | ------------------- | ----------------------------------------------------------- |
| `storage`         | `Vec<HydroStorage>` | Initial storage for operating hydros [hm³]                  |
| `filling_storage` | `Vec<HydroStorage>` | Initial storage for filling hydros [hm³]; below dead volume |

`HydroStorage` carries `hydro_id: EntityId` and `value_hm3: f64`. A hydro must
appear in exactly one of the two arrays. Both arrays are sorted by `hydro_id`
after loading for declaration-order invariance.

```rust
use cobre_core::{EntityId, InitialConditions, HydroStorage};

let ic = InitialConditions {
    storage: vec![
        HydroStorage { hydro_id: EntityId(0), value_hm3: 15_000.0 },
        HydroStorage { hydro_id: EntityId(1), value_hm3:  8_500.0 },
    ],
    filling_storage: vec![
        HydroStorage { hydro_id: EntityId(10), value_hm3: 200.0 },
    ],
};

assert_eq!(ic.storage.len(), 2);
assert_eq!(ic.filling_storage.len(), 1);
```

### GenericConstraint

`GenericConstraint` represents a user-defined linear constraint over LP
variables, loaded from `generic_constraints.json` and stored in
`System::generic_constraints`. The expression parser (string to
`ConstraintExpression`) and referential validation live in `cobre-io`, not here.

| Field         | Type                   | Description                                            |
| ------------- | ---------------------- | ------------------------------------------------------ |
| `id`          | `EntityId`             | Unique constraint identifier                           |
| `name`        | `String`               | Short name used in reports and log output              |
| `description` | `Option<String>`       | Optional human-readable description                    |
| `expression`  | `ConstraintExpression` | Parsed left-hand-side linear expression                |
| `sense`       | `ConstraintSense`      | Comparison sense: `GreaterEqual`, `LessEqual`, `Equal` |
| `slack`       | `SlackConfig`          | Slack variable configuration                           |

`ConstraintExpression` holds a `Vec<LinearTerm>`. Each `LinearTerm` has a
`coefficient: f64` and a `variable: VariableRef`.

### VariableRef

`VariableRef` is an enum with 20 variants covering all LP variable types
defined in the data model. Each variant names the variable type and carries the
entity ID. For block-specific variables, `block_id` is `None` to sum over all
blocks or `Some(i)` to reference block `i` specifically.

| Category | Variants                                                                                                                                     |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Hydro    | `HydroStorage`, `HydroTurbined`, `HydroSpillage`, `HydroDiversion`, `HydroOutflow`, `HydroGeneration`, `HydroEvaporation`, `HydroWithdrawal` |
| Thermal  | `ThermalGeneration`                                                                                                                          |
| Line     | `LineDirect`, `LineReverse`, `LineExchange`                                                                                                  |
| Bus      | `BusDeficit`, `BusExcess`                                                                                                                    |
| Pumping  | `PumpingFlow`, `PumpingPower`                                                                                                                |
| Contract | `ContractImport`, `ContractExport`                                                                                                           |
| NCS      | `NonControllableGeneration`, `NonControllableCurtailment`                                                                                    |

`HydroStorage`, `HydroEvaporation`, and `HydroWithdrawal` are stage-level
variables (no `block_id`). All other hydro variables and all thermal, line, bus,
pumping, contract, and NCS variables are block-specific (`block_id` field present).

`LineExchange` represents the net flow on a line (direct - reverse). Its resolver
returns two LP column entries: `(fwd_col, +1.0)` and `(rev_col, -1.0)`. This
simplifies generic constraints that reference net exchange between buses.

### SlackConfig

Controls whether a soft constraint with a penalty cost is added to the LP:

| Field     | Type          | Description                                                        |
| --------- | ------------- | ------------------------------------------------------------------ |
| `enabled` | `bool`        | If `true`, adds a slack variable allowing constraint violation     |
| `penalty` | `Option<f64>` | Penalty per unit of violation; must be `Some(positive)` if enabled |

```rust
use cobre_core::{
    EntityId, GenericConstraint, ConstraintExpression, ConstraintSense,
    LinearTerm, SlackConfig, VariableRef,
};

let expr = ConstraintExpression {
    terms: vec![
        LinearTerm {
            coefficient: 1.0,
            variable: VariableRef::HydroGeneration {
                hydro_id: EntityId(10),
                block_id: None,   // sum over all blocks
            },
        },
        LinearTerm {
            coefficient: 1.0,
            variable: VariableRef::HydroGeneration {
                hydro_id: EntityId(11),
                block_id: None,
            },
        },
    ],
};

let gc = GenericConstraint {
    id: EntityId(0),
    name: "min_hydro_total".to_string(),
    description: Some("Minimum total hydro generation".to_string()),
    expression: expr,
    sense: ConstraintSense::GreaterEqual,
    slack: SlackConfig { enabled: true, penalty: Some(5_000.0) },
};

assert_eq!(gc.expression.terms.len(), 2);
```

## Resolved penalties and bounds

The `resolved` module holds pre-resolved penalty and bound tables that provide
O(1) lookup for LP builders and solvers.

### Design: flat Vec with 2D indexing

During input loading, the three-tier cascade (global defaults -> entity overrides
-> stage overrides) is evaluated once by `cobre-io`. The results are stored in
flat `Vec<T>` arrays with manual 2D indexing:

```
data[entity_idx * n_stages + stage_idx]
```

This layout gives cache-friendly sequential access when iterating over stages
for a fixed entity (the common inner loop pattern in LP construction). No
re-evaluation of the cascade is ever required at solve time; every penalty or
bound lookup is a single array index operation.

### ResolvedPenalties

`ResolvedPenalties` holds per-(entity, stage) penalty values for all four
entity types that carry stage-varying penalties: hydros, buses, lines, and
non-controllable sources.

Per-(entity, stage) penalty structs:

| Struct                | Fields                  | Description                                              |
| --------------------- | ----------------------- | -------------------------------------------------------- |
| `HydroStagePenalties` | 11 `f64` fields         | All hydro penalty costs for one (hydro, stage) pair      |
| `BusStagePenalties`   | `excess_cost: f64`      | Bus excess cost for one (bus, stage) pair                |
| `LineStagePenalties`  | `exchange_cost: f64`    | Line flow regularization cost for one (line, stage) pair |
| `NcsStagePenalties`   | `curtailment_cost: f64` | NCS curtailment cost for one (ncs, stage) pair           |

Bus deficit segments are NOT stage-varying. The piecewise-linear deficit
structure is fixed at the entity or global level, so `BusStagePenalties`
contains only `excess_cost`.

All four per-stage penalty structs implement `Copy`, so they can be passed by
value on hot paths.

```rust
use cobre_core::resolved::{
    BusStagePenalties, HydroStagePenalties, LineStagePenalties,
    NcsStagePenalties, ResolvedPenalties,
};

// Allocate a 3-hydro, 2-bus, 1-line, 1-ncs table for 5 stages.
let table = ResolvedPenalties::new(
    3, 2, 1, 1, 5,
    HydroStagePenalties { spillage_cost: 0.01, diversion_cost: 0.02,
                          fpha_turbined_cost: 0.03,
                          storage_violation_below_cost: 1000.0,
                          filling_target_violation_cost: 5000.0,
                          turbined_violation_below_cost: 500.0,
                          outflow_violation_below_cost: 500.0,
                          outflow_violation_above_cost: 500.0,
                          generation_violation_below_cost: 500.0,
                          evaporation_violation_cost: 500.0,
                          water_withdrawal_violation_cost: 500.0 },
    BusStagePenalties { excess_cost: 100.0 },
    LineStagePenalties { exchange_cost: 5.0 },
    NcsStagePenalties { curtailment_cost: 50.0 },
);

// O(1) lookup: hydro 1, stage 3
let p = table.hydro_penalties(1, 3);
assert!((p.spillage_cost - 0.01).abs() < f64::EPSILON);
```

### ResolvedBounds

`ResolvedBounds` holds per-(entity, stage) bound values for five entity types:
hydros, thermals, lines, pumping stations, and energy contracts.

Per-(entity, stage) bound structs:

| Struct                | Fields                                   | Description                                  |
| --------------------- | ---------------------------------------- | -------------------------------------------- |
| `HydroStageBounds`    | 11 fields (see table below)              | All hydro bounds for one (hydro, stage) pair |
| `ThermalStageBounds`  | `min_generation_mw`, `max_generation_mw` | Thermal generation bounds [MW]               |
| `LineStageBounds`     | `direct_mw`, `reverse_mw`                | Transmission capacity bounds [MW]            |
| `PumpingStageBounds`  | `min_flow_m3s`, `max_flow_m3s`           | Pumping flow bounds [m³/s]                   |
| `ContractStageBounds` | `min_mw`, `max_mw`, `price_per_mwh`      | Contract bounds [MW] and effective price     |

`HydroStageBounds` has 11 fields:

| Field                  | Unit | Description                                                          |
| ---------------------- | ---- | -------------------------------------------------------------------- |
| `min_storage_hm3`      | hm³  | Dead volume (soft lower bound)                                       |
| `max_storage_hm3`      | hm³  | Physical reservoir capacity (hard upper bound)                       |
| `min_turbined_m3s`     | m³/s | Minimum turbined flow (soft lower bound)                             |
| `max_turbined_m3s`     | m³/s | Maximum turbined flow (hard upper bound)                             |
| `min_outflow_m3s`      | m³/s | Environmental flow requirement (soft lower bound)                    |
| `max_outflow_m3s`      | m³/s | Flood-control limit (soft upper bound); `None` = unbounded           |
| `min_generation_mw`    | MW   | Minimum electrical generation (soft lower bound)                     |
| `max_generation_mw`    | MW   | Maximum electrical generation (hard upper bound)                     |
| `max_diversion_m3s`    | m³/s | Diversion channel capacity (hard upper bound); `None` = no diversion |
| `filling_inflow_m3s`   | m³/s | Filling inflow retained during filling stages; default 0.0           |
| `water_withdrawal_m3s` | m³/s | Water withdrawal per stage; positive = removed, negative = added     |

```rust
use cobre_core::resolved::{
    ContractStageBounds, HydroStageBounds, LineStageBounds,
    PumpingStageBounds, ResolvedBounds, ThermalStageBounds,
};

// Allocate a table for 2 hydros, 1 thermal, 1 line, 0 pumping, 0 contracts, 3 stages.
let table = ResolvedBounds::new(
    2, 1, 1, 0, 0, 3,
    HydroStageBounds { min_storage_hm3: 10.0, max_storage_hm3: 200.0,
                       min_turbined_m3s: 0.0,  max_turbined_m3s: 500.0,
                       min_outflow_m3s: 5.0,   max_outflow_m3s: None,
                       min_generation_mw: 0.0, max_generation_mw: 100.0,
                       max_diversion_m3s: None,
                       filling_inflow_m3s: 0.0, water_withdrawal_m3s: 0.0 },
    ThermalStageBounds { min_generation_mw: 50.0, max_generation_mw: 400.0 },
    LineStageBounds { direct_mw: 1000.0, reverse_mw: 800.0 },
    PumpingStageBounds { min_flow_m3s: 0.0, max_flow_m3s: 0.0 },
    ContractStageBounds { min_mw: 0.0, max_mw: 0.0, price_per_mwh: 0.0 },
);

// O(1) lookup: hydro 0, stage 2
let b = table.hydro_bounds(0, 2);
assert!((b.max_storage_hm3 - 200.0).abs() < f64::EPSILON);
assert!(b.max_outflow_m3s.is_none());
```

Both tables expose `_mut` accessor variants (e.g., `hydro_penalties_mut`,
`hydro_bounds_mut`) that return `&mut T` for in-place updates during case
loading. These are used exclusively by `cobre-io`; all other crates use the
immutable read accessors.

## Serde feature flag

`cobre-core` ships with an optional `serde` feature that enables
`serde::Serialize` and `serde::Deserialize` for all public types. The feature
is disabled by default to keep the minimal build free of serialization
dependencies.

### When to enable

| Use case                                          | Enable? |
| ------------------------------------------------- | ------- |
| Reading `cobre-core` as a pure data model library | No      |
| Building `cobre-io` (JSON input loading)          | Yes     |
| MPI broadcast via `postcard` in `cobre-comm`      | Yes     |
| Checkpoint serialization in `cobre-sddp`          | Yes     |
| Python bindings in `cobre-python`                 | Yes     |
| Writing tests that inspect values as JSON         | Yes     |

### Enabling the feature

```toml
# Cargo.toml
[dependencies]
cobre-core = { version = "0.x", features = ["serde"] }
```

Or from the command line:

```
cargo build --features cobre-core/serde
```

Enabling `serde` also activates `chrono/serde`, which is required because
`Stage` carries `NaiveDate` fields that must be serializable for JSON input
loading and MPI broadcast.

### How it works

Every public type in `cobre-core` carries a `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`
attribute. When the feature is inactive, the derive is omitted entirely and the
`serde` dependency is not compiled. There is no runtime cost and no API surface
change when the feature is disabled.

All downstream Cobre crates that perform serialization declare
`cobre-core/serde` as a required dependency. The workspace ensures that only
one copy of `cobre-core` is compiled, with the feature union of all crates that
request it.

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
