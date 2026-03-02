# ticket-004 Define Thermal, PumpingStation, EnergyContract, and NonControllableSource Entities

## Context

### Background

After Bus, Line, and Hydro are defined (tickets 002 and 003), the remaining four entity types complete the entity type system. Thermal is a full entity with a piecewise cost curve. PumpingStation, EnergyContract, and NonControllableSource are complete struct definitions -- they exist as full entities in the registry but will contribute no LP variables in the minimal viable solver (NO-OP stubs for LP purposes, but complete data model definitions).

### Relation to Epic

This is the fourth and final ticket in Epic 01 (Foundation Types). It completes the entity type system by populating four entity modules. After this ticket, every type needed by the `System` struct (Epic 2) is defined and compiling.

### Current State

- `crates/cobre-core/src/entities/thermal.rs` is a stub (from ticket-001)
- `crates/cobre-core/src/entities/pumping_station.rs` is a stub (from ticket-001)
- `crates/cobre-core/src/entities/energy_contract.rs` is a stub (from ticket-001)
- `crates/cobre-core/src/entities/non_controllable.rs` is a stub (from ticket-001)
- `EntityId`, `Bus`, `Line`, `Hydro`, and all hydro supporting types are defined (tickets 001-003)

## Specification

### Requirements

1. Define supporting types for Thermal in `entities/thermal.rs`:

   **`ThermalCostSegment`**:

   ```rust
   #[derive(Debug, Clone, Copy, PartialEq)]
   pub struct ThermalCostSegment {
       /// Generation capacity of this segment [MW].
       pub capacity_mw: f64,
       /// Marginal cost in this segment [$/MWh].
       pub cost_per_mwh: f64,
   }
   ```

   **`GnlConfig`**:

   ```rust
   #[derive(Debug, Clone, Copy, PartialEq, Eq)]
   pub struct GnlConfig {
       /// Number of stages of dispatch anticipation.
       pub lag_stages: i32,
   }
   ```

2. Define `Thermal` in `entities/thermal.rs`:

   ```rust
   #[derive(Debug, Clone, PartialEq)]
   pub struct Thermal {
       pub id: EntityId,
       pub name: String,
       pub bus_id: EntityId,
       pub entry_stage_id: Option<i32>,
       pub exit_stage_id: Option<i32>,
       pub cost_segments: Vec<ThermalCostSegment>,
       pub min_generation_mw: f64,
       pub max_generation_mw: f64,
       pub gnl_config: Option<GnlConfig>,
   }
   ```

3. Define `PumpingStation` in `entities/pumping_station.rs`:

   ```rust
   #[derive(Debug, Clone, PartialEq)]
   pub struct PumpingStation {
       pub id: EntityId,
       pub name: String,
       pub bus_id: EntityId,
       pub source_hydro_id: EntityId,
       pub destination_hydro_id: EntityId,
       pub entry_stage_id: Option<i32>,
       pub exit_stage_id: Option<i32>,
       pub consumption_mw_per_m3s: f64,
       pub min_flow_m3s: f64,
       pub max_flow_m3s: f64,
   }
   ```

4. Define `ContractType` enum and `EnergyContract` in `entities/energy_contract.rs`:

   **`ContractType`**:

   ```rust
   #[derive(Debug, Clone, Copy, PartialEq, Eq)]
   pub enum ContractType {
       Import,
       Export,
   }
   ```

   **`EnergyContract`**:

   ```rust
   #[derive(Debug, Clone, PartialEq)]
   pub struct EnergyContract {
       pub id: EntityId,
       pub name: String,
       pub bus_id: EntityId,
       pub contract_type: ContractType,
       pub entry_stage_id: Option<i32>,
       pub exit_stage_id: Option<i32>,
       pub price_per_mwh: f64,
       pub min_mw: f64,
       pub max_mw: f64,
   }
   ```

5. Define `NonControllableSource` in `entities/non_controllable.rs`:

   ```rust
   #[derive(Debug, Clone, PartialEq)]
   pub struct NonControllableSource {
       pub id: EntityId,
       pub name: String,
       pub bus_id: EntityId,
       pub entry_stage_id: Option<i32>,
       pub exit_stage_id: Option<i32>,
       pub max_generation_mw: f64,
       pub curtailment_cost: f64,
   }
   ```

6. Update `entities/mod.rs` to declare and re-export all new types.

7. Update `lib.rs` re-exports to include all new entity types.

### Inputs/Props

- `EntityId` from `entity_id.rs` (ticket-001)

### Outputs/Behavior

- All four entity types and their supporting types are accessible from crate root
- All types derive the exact traits specified in the spec
- All public items have doc comments

### Error Handling

No error handling in this ticket -- pure data structs.

## Acceptance Criteria

- [ ] Given the file `crates/cobre-core/src/entities/thermal.rs`, when inspecting `Thermal` fields, then it has exactly 9 fields: id, name, bus_id, entry_stage_id, exit_stage_id, cost_segments, min_generation_mw, max_generation_mw, gnl_config
- [ ] Given the file `crates/cobre-core/src/entities/pumping_station.rs`, when inspecting `PumpingStation` fields, then it has exactly 10 fields matching the spec section 1.9.6
- [ ] Given the file `crates/cobre-core/src/entities/energy_contract.rs`, when inspecting `EnergyContract` fields, then it has exactly 9 fields and `ContractType` has exactly 2 variants (Import, Export)
- [ ] Given the file `crates/cobre-core/src/entities/non_controllable.rs`, when inspecting `NonControllableSource` fields, then it has exactly 7 fields matching the spec section 1.9.8
- [ ] Given a clean checkout, when running `cargo clippy -p cobre-core`, then zero warnings are produced

## Implementation Guide

### Suggested Approach

1. Implement `ThermalCostSegment`, `GnlConfig`, and `Thermal` in `thermal.rs`
2. Implement `PumpingStation` in `pumping_station.rs`
3. Implement `ContractType` and `EnergyContract` in `energy_contract.rs`
4. Implement `NonControllableSource` in `non_controllable.rs`
5. Update `entities/mod.rs` with all module declarations and re-exports
6. Update `lib.rs` with all entity type re-exports
7. Write unit tests for each type
8. Run full build pipeline: `cargo build`, `cargo clippy`, `cargo doc`, `cargo test`

### Key Files to Modify

- `crates/cobre-core/src/entities/thermal.rs` (populate from stub)
- `crates/cobre-core/src/entities/pumping_station.rs` (populate from stub)
- `crates/cobre-core/src/entities/energy_contract.rs` (populate from stub)
- `crates/cobre-core/src/entities/non_controllable.rs` (populate from stub)
- `crates/cobre-core/src/entities/mod.rs` (add module decls and re-exports)
- `crates/cobre-core/src/lib.rs` (add entity re-exports)

### Patterns to Follow

- Match the spec sections 1.9.5 (Thermal), 1.9.6 (PumpingStation), 1.9.7 (EnergyContract), 1.9.8 (NonControllableSource) field-by-field
- Doc comments include units in brackets: `/// Power consumption rate [MW/(m3/s)]`
- `ThermalCostSegment` derives `Copy` (spec: `Debug, Clone, Copy, PartialEq`)
- `GnlConfig` derives `Copy` and `Eq` (all fields are `i32`)
- `ContractType` derives `Copy` and `Eq` (no data fields in variants)

### Pitfalls to Avoid

- `PumpingStation.source_hydro_id` and `destination_hydro_id` are `EntityId` (not `i32`) for type safety
- `EnergyContract.price_per_mwh` can be negative (export revenue) -- this is intentional, not a bug
- `NonControllableSource.curtailment_cost` is a resolved field (defaults applied during loading), not optional
- Do NOT add any `demand_mw` or `load_mw` field to Bus -- demand comes from external data in Phase 2
- `GnlConfig.lag_stages` is `i32` matching the JSON schema, with `Eq` derive (integer comparison)

## Testing Requirements

### Unit Tests

In `entities/thermal.rs`:

- `test_thermal_construction`: Create a Thermal with 2 cost segments, verify all fields
- `test_thermal_cost_segment`: Create a `ThermalCostSegment`, verify `capacity_mw` and `cost_per_mwh`
- `test_thermal_with_gnl`: Create a Thermal with `gnl_config: Some(GnlConfig { lag_stages: 2 })`, verify
- `test_thermal_without_gnl`: Create a Thermal with `gnl_config: None`, verify

In `entities/pumping_station.rs`:

- `test_pumping_station_construction`: Create a PumpingStation, verify all 10 fields

In `entities/energy_contract.rs`:

- `test_import_contract`: Create an EnergyContract with `ContractType::Import`, verify `price_per_mwh` is positive
- `test_export_contract`: Create an EnergyContract with `ContractType::Export`, verify `price_per_mwh` is negative
- `test_contract_type_equality`: `ContractType::Import == ContractType::Import` is true, `Import != Export`

In `entities/non_controllable.rs`:

- `test_non_controllable_construction`: Create a NonControllableSource, verify all 7 fields
- `test_non_controllable_curtailment_cost`: Verify `curtailment_cost` field is accessible and has expected value

### Integration Tests

Not applicable for this ticket.

### E2E Tests

Not applicable for this ticket.

## Dependencies

- **Blocked By**: ticket-001 (EntityId and module structure)
- **Blocks**: ticket-005 (SystemBuilder needs all entity types)

## Effort Estimate

**Points**: 2
**Confidence**: High
