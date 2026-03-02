# Master Plan: Phase 1 -- cobre-core Data Model and Registries

## Executive Summary

Phase 1 implements the `cobre-core` crate -- the foundation data model for the Cobre SDDP solver ecosystem. This crate defines all entity types (Bus, Line, Thermal, Hydro, PumpingStation, EnergyContract, NonControllableSource), supporting enums and structs, the `EntityId` newtype, the `System` container with O(1) lookup indices, cascade and network topology structures, three-tier penalty resolution at the entity level, and canonical ordering enforcement. The crate has zero in-workspace dependencies and is consumed read-only by all other Cobre crates.

## Goals & Non-Goals

### Goals

- Define all 7 entity struct types with all fields from the internal-structures spec (section 1.9)
- Define all supporting enums (`HydroGenerationModel`, `TailraceModel`, `HydraulicLossesModel`, `EfficiencyModel`, `ContractType`) and supporting structs (`DeficitSegment`, `ThermalCostSegment`, `GnlConfig`, `DiversionChannel`, `FillingConfig`, `HydroPenalties`, `TailracePoint`)
- Define the `EntityId` newtype wrapping `i32` with `Debug, Clone, Copy, PartialEq, Eq, Hash`
- Define the `System` struct with entity collections, lookup indices, and topology
- Implement `CascadeTopology` (downstream/upstream adjacency, topological ordering)
- Implement `NetworkTopology` (bus-line incidence, bus generation/load maps)
- Implement three-tier penalty resolution at the entity level (global defaults -> entity overrides)
- Enforce canonical ID ordering in all entity collections
- Ensure `System` is `Send + Sync` and immutable after construction
- Provide the full public API surface: slice accessors, count queries, entity lookup by ID
- Validate cross-references (bus_id exists, downstream_id exists), cascade DAG (no cycles), bus connectivity
- Achieve full `cargo test`, `cargo clippy`, and `cargo doc` compliance

### Non-Goals

- JSON/Parquet parsing or file I/O (Phase 2: cobre-io)
- Stage-varying penalty resolution (requires stages from Phase 2)
- `ResolvedPenalties` and `ResolvedBounds` per (entity, stage) tables (Phase 2)
- LP construction or solver integration (Phase 3/6)
- MPI communication (Phases 3-4)
- Scenario generation or PAR models (Phase 5)
- Performance-adapted views or SIMD layouts (cobre-sddp, Phase 6)
- `PolicyGraph`, `Stage`, `ParModel`, `CorrelationModel`, `InitialConditions`, `GenericConstraint` types (Phase 2+)
- Serde serialization/deserialization (Phase 2 concern for cobre-io)

## Architecture Overview

### Current State

The `cobre-core` crate is an empty stub: `src/lib.rs` contains only doc comments. `Cargo.toml` has no dependencies beyond workspace defaults.

### Target State

A fully populated crate with the following module structure:

```
crates/cobre-core/src/
  lib.rs              -- Public API re-exports, crate-level docs
  entity_id.rs        -- EntityId newtype
  entities/
    mod.rs            -- Module declarations and re-exports
    bus.rs            -- Bus, DeficitSegment
    line.rs           -- Line
    hydro.rs          -- Hydro, HydroGenerationModel, TailraceModel, TailracePoint,
                         HydraulicLossesModel, EfficiencyModel, DiversionChannel,
                         FillingConfig, HydroPenalties
    thermal.rs        -- Thermal, ThermalCostSegment, GnlConfig
    pumping_station.rs -- PumpingStation
    energy_contract.rs -- EnergyContract, ContractType
    non_controllable.rs -- NonControllableSource
  topology/
    mod.rs            -- Module declarations and re-exports
    cascade.rs        -- CascadeTopology
    network.rs        -- NetworkTopology
  system.rs           -- System struct, builder, public API
  error.rs            -- Error types for validation
  penalty.rs          -- Three-tier penalty resolution logic (global -> entity)
```

### Key Design Decisions

1. **Module per entity type**: Each entity gets its own file for clarity and to prevent monolithic files.
2. **No serde in Phase 1**: Entity types do not derive `Serialize`/`Deserialize` yet -- that is cobre-io's concern. This keeps cobre-core dependency-free.
3. **System construction via `SystemBuilder`**: A builder pattern that takes entity collections (pre-sorted or sorts them), resolves entity-level penalties, builds topology, validates cross-references, and produces an immutable `System`. The builder returns `Result<System, ValidationError>`.
4. **No `Ord` on `EntityId`**: Per spec, canonical ordering uses collection position. Sorting happens once during construction. If needed for sorting during construction, a private helper can compare the inner `i32`.
5. **Stub entities have full structs**: PumpingStation, EnergyContract, and NonControllableSource are complete struct definitions (not empty stubs) -- they just contribute no LP variables in the minimal viable solver.

## Technical Approach

### Tech Stack

- Rust 2024 edition, MSRV 1.85
- No external dependencies (cobre-core is dependency-free)
- Workspace lints: `unsafe_code = "forbid"`, `missing_docs = "warn"`, `clippy::pedantic`, `unwrap_used = "deny"`

### Component/Module Breakdown

| Module                       | Responsibility                     | Key Types                                                                                                                                                           |
| ---------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `entity_id`                  | Entity identification              | `EntityId`                                                                                                                                                          |
| `entities::bus`              | Bus node definition                | `Bus`, `DeficitSegment`                                                                                                                                             |
| `entities::line`             | Transmission line definition       | `Line`                                                                                                                                                              |
| `entities::hydro`            | Hydro plant definition             | `Hydro`, `HydroGenerationModel`, `TailraceModel`, `TailracePoint`, `HydraulicLossesModel`, `EfficiencyModel`, `DiversionChannel`, `FillingConfig`, `HydroPenalties` |
| `entities::thermal`          | Thermal plant definition           | `Thermal`, `ThermalCostSegment`, `GnlConfig`                                                                                                                        |
| `entities::pumping_station`  | Pumping station definition         | `PumpingStation`                                                                                                                                                    |
| `entities::energy_contract`  | Energy contract definition         | `EnergyContract`, `ContractType`                                                                                                                                    |
| `entities::non_controllable` | Non-controllable source definition | `NonControllableSource`                                                                                                                                             |
| `topology::cascade`          | Hydro cascade graph                | `CascadeTopology`                                                                                                                                                   |
| `topology::network`          | Transmission network graph         | `NetworkTopology`                                                                                                                                                   |
| `system`                     | Top-level container                | `System`, `SystemBuilder`                                                                                                                                           |
| `error`                      | Validation errors                  | `ValidationError`                                                                                                                                                   |
| `penalty`                    | Penalty resolution logic           | `GlobalPenaltyDefaults`, resolution functions                                                                                                                       |

### Data Flow

```
Entity data (from cobre-io in Phase 2, from test helpers in Phase 1)
    |
    v
SystemBuilder::new()
    .buses(Vec<Bus>)
    .lines(Vec<Line>)
    .hydros(Vec<Hydro>)
    .thermals(Vec<Thermal>)
    ...
    .global_penalties(GlobalPenaltyDefaults)
    .build()
    |
    v (sorts, validates, resolves, builds topology)
    |
    v
Result<System, ValidationError>
    |
    v
System (immutable, Send + Sync, shared via &System)
```

### Testing Strategy

- **Unit tests**: Per-module tests for each entity type (construction, field access), `EntityId` behavior, topology construction, penalty resolution
- **Integration tests**: Full `SystemBuilder` round-trip tests with realistic multi-entity configurations
- **Declaration-order invariance tests**: Same entities in different orders produce identical `System` (bit-for-bit)
- **Validation error tests**: Invalid cross-references, cascade cycles, missing buses produce correct errors
- **Property-based considerations**: Canonical ordering holds for any input permutation (can use simple permutation tests rather than proptest to avoid dependencies)

## Phases & Milestones

### Epic 1: Foundation Types (Detailed)

Define `EntityId`, all 7 entity structs, all supporting enums/structs, and the error module. This establishes the type system that everything else builds on.

### Epic 2: System Struct and Topology (Detailed)

Implement `CascadeTopology`, `NetworkTopology`, `GlobalPenaltyDefaults`, entity-level penalty resolution, `SystemBuilder`, and the `System` struct with its full public API.

### Epic 3: Validation and Testing (Outline)

Implement cross-reference validation, cascade DAG validation, bus connectivity checks, declaration-order invariance tests, and comprehensive integration tests.

## Risk Analysis

| Risk                                                   | Likelihood | Impact | Mitigation                                                                  |
| ------------------------------------------------------ | ---------- | ------ | --------------------------------------------------------------------------- |
| Spec ambiguity in penalty resolution scope for Phase 1 | Medium     | Low    | Phase 1 resolves global -> entity only; stage-varying is Phase 2            |
| Module structure diverges from cobre-io expectations   | Low        | Medium | cobre-io reads from files and constructs System; public API is well-defined |
| Topology validation logic is complex                   | Medium     | Medium | Decompose into cascade (DAG) and network (bus-line incidence) separately    |
| clippy pedantic produces many warnings initially       | High       | Low    | Address systematically per ticket; all pub items get doc comments           |

## Success Metrics

- `cargo build -p cobre-core` succeeds with zero errors
- `cargo test -p cobre-core` passes all tests (aim for 90%+ line coverage)
- `cargo clippy -p cobre-core` produces zero warnings
- `cargo doc -p cobre-core` builds clean documentation
- All entity types match spec section 1.9 field-by-field
- Declaration-order invariance test passes for at least one multi-entity scenario
- System construction validates and rejects invalid cascade cycles and missing bus references
