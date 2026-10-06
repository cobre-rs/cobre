# cobre-core

Shared data model for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.

This crate defines the fundamental types used across all Cobre tools: buses, branches,
generators (hydro, thermal, renewable), loads, and the top-level
`System` struct. A power system described with `cobre-core` types can be used for
stochastic optimization, steady-state analysis, and any other procedure in the
ecosystem. The crate carries no solver or algorithm dependencies and enforces
declaration-order invariance so that results are identical regardless of input ordering.

## When to Use

Depend on `cobre-core` directly when you are building a new analysis tool or
algorithm that needs to consume a validated power system description without
pulling in solver or I/O logic. If you are writing test utilities or fixtures
that construct small `System` instances, `cobre-core` is the only dependency
you need.

## Key Types

- **`System`** — Immutable, fully-validated power system description built by `SystemBuilder`
- **`SystemBuilder`** — Validates and assembles all entities into a `System`
- **`Hydro`** — Hydroelectric plant with storage, spillage, and productivity parameters
- **`Thermal`** — Thermal generation unit with cost segments and operational bounds
- **`Bus`** — Network bus carrying load, deficit penalties, and connected generators
- **`GenericConstraint`** — Linear constraint over named variables for custom coupling

## Module overview

| Module                            | Purpose                                                                                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `entities`                        | Entity types: `Bus`, `Line`, `Hydro`, `Thermal`, `PumpingStation`, `NonControllableSource`, `EnergyContract`                          |
| `entity_id`                       | `EntityId` newtype wrapper around `i32`                                                                                               |
| `error`                           | `ValidationError` enum returned by `SystemBuilder::build()`                                                                           |
| `model::temporal`                 | Stages, blocks, seasons, and the policy graph                                                                                         |
| `model::scenario`                 | PAR model parameters, load/NCS statistics, correlation model, sampling scheme, historical/external scenario row types                 |
| `model::penalty`                  | Global penalty defaults, entity-level overrides, and the `resolve_*` cascade functions                                                |
| `model::resolved`                 | Pre-resolved penalty/bound/factor tables with O(1) array-indexed lookup for solver crates                                             |
| `model::parameters`               | `ScalarParameter` / `ComputedParameter` model for user-defined scalar and derived parameters                                          |
| `constraints::generic_constraint` | User-defined linear constraints (`GenericConstraint`, `VariableRef`) over LP variables                                                |
| `constraints::initial_conditions` | Reservoir storage, AR inflow lags, and anticipated-commitment history at study start                                                  |
| `constraints::training_event`     | `TrainingEvent` enum: the typed event stream consumed by loggers, the TUI, MCP progress, and Parquet writers                          |
| `stats::welford`                  | `WelfordAccumulator` — running mean and sample standard deviation for streaming statistics                                            |
| `system`                          | `System` container and `SystemBuilder`                                                                                                |
| `topology`                        | `CascadeTopology` derived structure                                                                                                   |

## `SystemBuilder` validation pipeline

`SystemBuilder::build()` (`src/system/builder.rs`) sorts, validates, and
assembles the immutable `System` in one pass, collecting every error found in
a phase before deciding whether to return early — it never short-circuits on
the first individual error, only between phases:

1. **Canonical sort.** Buses, lines, hydros, thermals, pumping stations,
   contracts, and non-controllable sources sort by
   `(operational_start_date, id)`; stages and generic constraints sort by `id`
   alone. The `id` tiebreak is the stable canonical key (entity names are
   user-chosen and vary between authors of the same system).
2. **Duplicate check.** Every entity collection (and the stage collection) is
   scanned for duplicate `EntityId`/`id` values across all collections before
   returning early with the accumulated error list.
3. **Cross-reference validation.** Every foreign-key field (`bus_id`,
   `source_bus_id`/`target_bus_id`, `downstream_id`, `source_hydro_id`/
   `destination_hydro_id`, etc.) is checked against the appropriate index.
4. **Cascade cycle + filling config validation.** `CascadeTopology` is built
   from the validated hydro `downstream_id` fields; a topological sort that
   does not reach all hydros reports the unvisited set as a
   `ValidationError::CascadeCycle`. Each hydro `FillingConfig` is checked for
   a non-negative `filling_min_rate_m3s` and a non-`None` `entry_stage_id`.

If all phases pass, `build()` builds O(1) lookup indices for all seven entity
collections and returns the immutable `System`. This guarantees
declaration-order invariance: two `System` values built from the same
entities in different input orders are structurally identical.

## Feature flags

| Feature  | Default | Description                                                                                                                                                                                                                                                                               |
| -------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `serde`  | off     | Enables `serde::Serialize`/`Deserialize` for all public types (and `chrono/serde`, needed because `Stage` carries `NaiveDate` fields). Required by `cobre-io` (JSON loading), MPI broadcast via `postcard` in `cobre-comm`, checkpoint serialization in `cobre-sddp`, and `cobre-python`. |
| `schema` | off     | Enables `schemars::JsonSchema` for public types referenced from auto-generated JSON Schemas (e.g. `ComputedParameter` embedded in `constraints/generic_parameters.json`). Implies `serde`.                                                                                                |

Every public type carries a `#[cfg_attr(feature = "serde", derive(...))]`
attribute, so the derive — and the `serde`/`schemars` dependencies — are
compiled out entirely when the feature is disabled, with no runtime cost and
no API surface change.

## Links

| Resource   | URL                                                      |
| ---------- | -------------------------------------------------------- |
| Docs site  | <https://docs.cobre-rs.dev/>                             |
| API Docs   | https://docs.rs/cobre-core/latest/cobre_core/            |
| Repository | https://github.com/cobre-rs/cobre                        |
| CHANGELOG  | https://github.com/cobre-rs/cobre/blob/main/CHANGELOG.md |

## Status

Alpha — API is functional but not yet stable.

## License

Apache-2.0
