# ticket-009 Implement Cross-Reference Validation in SystemBuilder

## Context

### Background

All entity types in `cobre-core` have fields that reference other entities by `EntityId` (e.g., `Hydro.bus_id` references a `Bus`, `Line.source_bus_id` references a `Bus`, `PumpingStation.source_hydro_id` references a `Hydro`). Currently, `SystemBuilder::build()` only validates duplicate IDs within each collection. It does not verify that cross-reference fields point to entities that actually exist. An invalid reference (e.g., a hydro plant connected to bus ID 99 when no bus with ID 99 exists) would silently produce a `System` with broken topology, causing panics or incorrect results downstream in `cobre-sddp`.

### Relation to Epic

This is the first ticket in Epic 03 (Validation and Testing). It adds the first layer of semantic validation to `SystemBuilder::build()`, ensuring all inter-entity references are valid before topology construction. Ticket-010 adds cascade-specific and filling-config validation. Ticket-011 exercises both validation layers via integration tests.

### Current State

- `SystemBuilder::build()` in `crates/cobre-core/src/system.rs` sorts entities, checks for duplicate IDs, builds `CascadeTopology` and `NetworkTopology`, and constructs lookup indices.
- The `ValidationError::InvalidReference` variant already exists in `crates/cobre-core/src/error.rs` with fields: `source_entity_type`, `source_id`, `field_name`, `referenced_id`, `expected_type`.
- The `bus_index`, `hydro_index` (and other indices) `HashMap<EntityId, usize>` are constructed after duplicate checking. These can be used for O(1) reference lookups.
- The comment on `SystemBuilder::build()` line 347 says: "Cross-reference and topology validation is added in Epic 3."

## Specification

### Requirements

Add cross-reference validation to `SystemBuilder::build()` that checks every entity field referencing another entity. Validation runs **after** duplicate checking passes and **after** indices are built, but **before** topology construction. All invalid references across all entity types are collected into the error vector (no short-circuiting on first error).

The following cross-references must be validated:

| Source Entity           | Field                                                  | Must Reference   |
| ----------------------- | ------------------------------------------------------ | ---------------- |
| `Line`                  | `source_bus_id`                                        | existing `Bus`   |
| `Line`                  | `target_bus_id`                                        | existing `Bus`   |
| `Hydro`                 | `bus_id`                                               | existing `Bus`   |
| `Hydro`                 | `downstream_id` (when `Some`)                          | existing `Hydro` |
| `Hydro`                 | `diversion.downstream_id` (when `diversion` is `Some`) | existing `Hydro` |
| `Thermal`               | `bus_id`                                               | existing `Bus`   |
| `PumpingStation`        | `bus_id`                                               | existing `Bus`   |
| `PumpingStation`        | `source_hydro_id`                                      | existing `Hydro` |
| `PumpingStation`        | `destination_hydro_id`                                 | existing `Hydro` |
| `EnergyContract`        | `bus_id`                                               | existing `Bus`   |
| `NonControllableSource` | `bus_id`                                               | existing `Bus`   |

### Inputs/Props

The validation function receives the already-built `HashMap<EntityId, usize>` indices for buses and hydros, plus the sorted entity slices.

### Outputs/Behavior

For each invalid reference, push a `ValidationError::InvalidReference` to the error vector with:

- `source_entity_type`: the entity type containing the bad reference (e.g., `"Hydro"`)
- `source_id`: the `EntityId` of the entity containing the bad reference
- `field_name`: the field name (e.g., `"bus_id"`, `"downstream_id"`, `"diversion.downstream_id"`)
- `referenced_id`: the `EntityId` that does not exist
- `expected_type`: the entity type that was expected (e.g., `"Bus"`, `"Hydro"`)

If any invalid references are found, `build()` returns `Err(errors)` (combined with any other validation errors from later tickets). If all references are valid, processing continues to topology construction.

### Error Handling

This ticket does not introduce new error variants. It uses the existing `ValidationError::InvalidReference` variant. Multiple invalid references across different entity types are all collected and returned together.

## Acceptance Criteria

- [ ] Given a `Hydro` with `bus_id = EntityId(99)` and no `Bus` with id 99 exists, when `SystemBuilder::build()` is called, then the result is `Err` containing `ValidationError::InvalidReference { source_entity_type: "Hydro", source_id: EntityId(_), field_name: "bus_id", referenced_id: EntityId(99), expected_type: "Bus" }`.
- [ ] Given a `Hydro` with `downstream_id = Some(EntityId(50))` and no `Hydro` with id 50 exists, when `SystemBuilder::build()` is called, then the result is `Err` containing `ValidationError::InvalidReference` with `field_name: "downstream_id"` and `expected_type: "Hydro"`.
- [ ] Given a `PumpingStation` with `source_hydro_id = EntityId(77)` and no `Hydro` with id 77 exists, when `SystemBuilder::build()` is called, then the result is `Err` containing `ValidationError::InvalidReference` with `field_name: "source_hydro_id"` and `expected_type: "Hydro"`.
- [ ] Given multiple entities with invalid references in different collections (e.g., a `Line` with bad `source_bus_id` AND a `Thermal` with bad `bus_id`), when `SystemBuilder::build()` is called, then the `Err` vector contains one `InvalidReference` error for each invalid reference (all are reported, not just the first).
- [ ] Given all cross-references are valid, when `SystemBuilder::build()` is called, then the result is `Ok(System)` with the same behavior as before this ticket.

## Implementation Guide

### Suggested Approach

1. In `SystemBuilder::build()`, after the existing duplicate-checking block and index construction, add a call to a new private function `validate_cross_references(...)`.
2. The function signature should be:
   ```rust
   fn validate_cross_references(
       lines: &[Line],
       hydros: &[Hydro],
       thermals: &[Thermal],
       pumping_stations: &[PumpingStation],
       contracts: &[EnergyContract],
       non_controllable_sources: &[NonControllableSource],
       bus_index: &HashMap<EntityId, usize>,
       hydro_index: &HashMap<EntityId, usize>,
       errors: &mut Vec<ValidationError>,
   )
   ```
3. Inside the function, iterate over each entity collection and check each cross-reference field against the appropriate index using `index.contains_key(&referenced_id)`.
4. Move the `if !errors.is_empty() { return Err(errors); }` check to AFTER cross-reference validation (it currently gates only duplicate errors). Alternatively, keep the duplicate-error early return and add a second early return after cross-reference validation.
5. Update the doc comment on `build()` to document the new validation.

### Key Files to Modify

- `crates/cobre-core/src/system.rs` -- add `validate_cross_references` function and call it from `build()`

### Patterns to Follow

- Follow the `check_duplicates` pattern: a private function that takes entity slices and an `&mut Vec<ValidationError>`, pushing errors without returning early.
- Use `bus_index.contains_key(&id)` for O(1) existence checks, following the `build_index` pattern already used.
- Use `.to_string()` for string construction in tests (not `String::from()`), per Epic 01 conventions.

### Pitfalls to Avoid

- Do NOT validate `downstream_id` when it is `None` -- only check `Some(id)` values.
- Do NOT validate `diversion.downstream_id` when `diversion` is `None`.
- Do NOT check topology (cycles, connectivity) in this ticket -- that is ticket-010.
- Do NOT add `PartialEq` to `System` in this ticket -- that is ticket-011.
- Remember that `bus_index` only contains bus IDs and `hydro_index` only contains hydro IDs. Do not confuse them.

## Testing Requirements

### Unit Tests

Add tests in the existing `#[cfg(test)] mod tests` block in `system.rs`:

1. `test_invalid_bus_reference_hydro` -- hydro with bad `bus_id`
2. `test_invalid_downstream_reference` -- hydro with bad `downstream_id`
3. `test_invalid_pumping_station_hydro_refs` -- pumping station with bad `source_hydro_id` or `destination_hydro_id`
4. `test_multiple_invalid_references_collected` -- multiple bad refs across entity types, all reported
5. `test_valid_cross_references_pass` -- all refs valid, `build()` succeeds

### Integration Tests

None in this ticket. Integration tests are in ticket-011.

### E2E Tests

Not applicable.

## Dependencies

- **Blocked By**: ticket-008-system-builder-and-system-struct.md
- **Blocks**: ticket-011-integration-and-order-invariance-tests.md

## Effort Estimate

**Points**: 2
**Confidence**: High
