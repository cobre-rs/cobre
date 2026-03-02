# Epic 01: Foundation Types

## Goal

Define the complete type system for `cobre-core`: the `EntityId` newtype, all 7 entity struct types with all fields from the internal-structures spec (section 1.9), all supporting enums and structs, the error module, and the crate module structure. After this epic, every type that the `System` struct (Epic 2) needs to reference exists and compiles.

## Scope

- `EntityId` newtype wrapping `i32` with `Debug, Clone, Copy, PartialEq, Eq, Hash`
- Supporting enums: `HydroGenerationModel`, `TailraceModel`, `HydraulicLossesModel`, `EfficiencyModel`, `ContractType`
- Supporting structs: `DeficitSegment`, `ThermalCostSegment`, `GnlConfig`, `DiversionChannel`, `FillingConfig`, `HydroPenalties`, `TailracePoint`
- Entity structs: `Bus`, `Line`, `Hydro`, `Thermal`, `PumpingStation`, `EnergyContract`, `NonControllableSource`
- Error types: `ValidationError` enum with all validation failure variants
- Module structure: `entity_id.rs`, `entities/`, `error.rs`, crate `lib.rs` re-exports
- Full doc comments on every public item (workspace lint: `missing_docs = "warn"`)
- All types compile under `unsafe_code = "forbid"`, `unwrap_used = "deny"`, `clippy::pedantic`

## Non-Goals

- System struct or builder (Epic 2)
- Topology structs (Epic 2)
- Penalty resolution logic (Epic 2)
- Serde derives (Phase 2: cobre-io)
- Any test beyond basic compilation and unit tests for the types themselves

## Tickets

1. **ticket-001**: Scaffold crate module structure, define `EntityId` and error types
2. **ticket-002**: Define Bus, Line, and supporting types (DeficitSegment)
3. **ticket-003**: Define Hydro entity and all hydro supporting types
4. **ticket-004**: Define Thermal, PumpingStation, EnergyContract, NonControllableSource and remaining supporting types

## Dependencies

- No external dependencies. This is the first epic.

## Success Criteria

- `cargo build -p cobre-core` compiles with zero errors
- `cargo clippy -p cobre-core` produces zero warnings
- `cargo doc -p cobre-core` builds clean documentation
- `cargo test -p cobre-core` passes all unit tests for entity types
- All public items have doc comments
