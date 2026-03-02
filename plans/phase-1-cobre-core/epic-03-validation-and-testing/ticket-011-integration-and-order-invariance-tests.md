# ticket-011 Implement Integration Tests and Declaration-Order Invariance Tests

## Context

### Background

The `cobre-core` crate now has all entity types, topology structures, penalty resolution, `SystemBuilder`, and full validation (cross-references, cascade cycles, filling configs) implemented. What remains is a comprehensive integration test suite that exercises the full `SystemBuilder::build()` pipeline with realistic multi-entity configurations and verifies the declaration-order invariance guarantee: the same set of entities, provided in any input order, must produce an identical `System`.

### Relation to Epic

This is the final ticket in Epic 03 and the final ticket in Phase 1. It serves as the acceptance test suite for the entire `cobre-core` crate. Successful completion of these tests demonstrates that Phase 1 is ready for downstream consumers (`cobre-io`, `cobre-sddp`, etc.).

### Current State

- `System` struct in `crates/cobre-core/src/system.rs` does NOT derive `PartialEq`. All of its field types (`Vec<Bus>`, `HashMap<EntityId, usize>`, `CascadeTopology`, `NetworkTopology`) do implement `PartialEq`, so deriving it is straightforward.
- No `crates/cobre-core/tests/` directory exists yet. Integration tests will be the first files in this directory.
- Test helpers (`make_bus`, `make_hydro`, etc.) exist in unit test modules within `system.rs`, `cascade.rs`, and `network.rs`, but they are module-private. Integration tests will need their own helpers or a shared test utility.
- `SystemBuilder` supports all 7 entity types with builder methods.
- All validation (duplicates, cross-references, cascade cycles, filling configs) is implemented.

## Specification

### Requirements

1. **Add `PartialEq` derive to `System`**: Add `#[derive(Debug, PartialEq)]` to the `System` struct so that two `System` instances can be compared with `==`.

2. **Create integration test file**: Create `crates/cobre-core/tests/integration.rs` with tests that exercise the full System construction pipeline.

3. **Declaration-order invariance test**: Build the same multi-entity system twice with entities provided in different orders. Assert that the two resulting `System` instances are identical via `PartialEq`.

4. **Realistic multi-entity test**: Build a system with all 7 entity types, a multi-node cascade, multiple buses with lines, and verify all lookups, topology queries, and entity counts are correct.

5. **Validation rejection tests**: Build systems with known-invalid configurations and verify the correct `ValidationError` variants are returned.

### Inputs/Props

Integration tests use `cobre_core` as a library dependency (external test crate). All types are accessed via `use cobre_core::{...}`.

### Outputs/Behavior

All tests pass with `cargo test --package cobre-core`. No test depends on another test. Each test constructs its own data.

### Error Handling

Tests that verify validation errors use `assert!(result.is_err())` and inspect the error vector contents with pattern matching.

## Acceptance Criteria

- [ ] Given the `System` struct in `crates/cobre-core/src/system.rs`, when its derive list is inspected, then it includes `PartialEq` (i.e., `#[derive(Debug, PartialEq)]`).
- [ ] Given a set of 2 buses, 1 line, 2 hydros (cascade A->B), 1 thermal, 1 pumping station, 1 contract, and 1 NCS, when `SystemBuilder::build()` is called with entities in forward ID order and then again with entities in reverse ID order, then both resulting `System` values are equal (`system_forward == system_reverse`).
- [ ] Given a system with all 7 entity types constructed via `SystemBuilder`, when entity counts, ID lookups, cascade topology, and network topology are queried, then all values match the input data.
- [ ] Given a `Hydro` referencing a non-existent `Bus` via `bus_id`, when `SystemBuilder::build()` is called in the integration test, then the result is `Err` containing at least one `ValidationError::InvalidReference`.
- [ ] Given `cargo test --package cobre-core` is run, then all tests pass and `cargo clippy --package cobre-core` reports no warnings.

## Implementation Guide

### Suggested Approach

1. In `crates/cobre-core/src/system.rs`, change `#[derive(Debug)]` on `System` to `#[derive(Debug, PartialEq)]`.

2. Create `crates/cobre-core/tests/integration.rs`. This file acts as an external test crate and can only access public API.

3. At the top of `integration.rs`, define test helper functions for creating entities. These are similar to the unit test helpers but use the public API:

   ```rust
   use cobre_core::{
       Bus, DeficitSegment, DiversionChannel, EnergyContract, EntityId,
       FillingConfig, Hydro, HydroGenerationModel, HydroPenalties, Line,
       NonControllableSource, PumpingStation, SystemBuilder, Thermal,
       ThermalCostSegment, ValidationError,
   };
   // plus ContractType if needed
   ```

4. Write the declaration-order invariance test:
   - Create a full set of entities (all 7 types)
   - Call `SystemBuilder::new().<setters>.build()` with entities in ascending ID order
   - Clone all entity vecs and reverse them
   - Call `SystemBuilder::new().<setters>.build()` with reversed vecs
   - Assert `system_asc == system_desc`

5. Write a realistic multi-entity lookup test:
   - Build a 3-bus, 2-line, 3-hydro (cascade), 2-thermal, 1-PS, 1-contract, 1-NCS system
   - Verify `n_buses()`, `n_hydros()`, etc.
   - Verify `system.bus(EntityId(X)).is_some()`
   - Verify `system.cascade().downstream(...)` returns expected values
   - Verify `system.network().bus_generators(...)` returns expected hydro/thermal IDs

6. Write a validation rejection test:
   - Build a system with a hydro referencing a non-existent bus
   - Assert `build()` returns `Err` with `InvalidReference`

### Key Files to Modify

- `crates/cobre-core/src/system.rs` -- add `PartialEq` derive to `System` struct
- `crates/cobre-core/tests/integration.rs` -- new file with integration tests

### Patterns to Follow

- Use `.to_string()` for string construction in tests (per Epic 01 conventions).
- Use `make_bus(id)`, `make_hydro(id)` helper pattern from existing unit tests, but adapted for integration test scope (using public types only).
- Use `assert_eq!` for count and ID comparisons; use pattern matching with `matches!` for error variant checks.
- Sort `EntityId` comparisons by inner `i32` for determinism.

### Pitfalls to Avoid

- Do NOT make unit test helpers in `system.rs` public -- integration tests define their own helpers.
- Do NOT test internal module details (private fields, private functions) -- integration tests only use the public API.
- Do NOT add `Copy` to `System` -- only add `PartialEq`.
- Ensure `PumpingStation` test entities reference valid hydro IDs and bus IDs when testing the happy path.
- Remember that `make_hydro(id)` helpers must set `bus_id` to a valid bus ID for cross-reference validation to pass.

## Testing Requirements

### Unit Tests

None -- this ticket adds the `PartialEq` derive but does not add new unit tests in `system.rs`.

### Integration Tests

All tests are in `crates/cobre-core/tests/integration.rs`:

1. `test_declaration_order_invariance` -- same entities in forward vs. reverse order produce identical `System`
2. `test_realistic_multi_entity_system` -- all 7 entity types, verify counts, lookups, topology
3. `test_invalid_cross_reference_rejected` -- hydro with bad `bus_id` rejected
4. `test_cascade_cycle_rejected` -- cyclic cascade rejected in integration context

### E2E Tests

Not applicable.

## Dependencies

- **Blocked By**: ticket-009-cross-reference-validation.md, ticket-010-cascade-and-filling-validation.md
- **Blocks**: None -- this is the final ticket in Phase 1

## Effort Estimate

**Points**: 2
**Confidence**: High
