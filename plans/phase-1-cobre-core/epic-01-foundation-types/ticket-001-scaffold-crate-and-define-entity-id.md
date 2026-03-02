# ticket-001 Scaffold Crate Module Structure and Define EntityId

## Context

### Background

The `cobre-core` crate is the foundation data model for the Cobre SDDP solver ecosystem. It currently contains only doc comments in `src/lib.rs` and has no dependencies. This ticket establishes the module structure that all subsequent tickets build upon, and defines the two most fundamental types: `EntityId` (used by every entity struct) and `ValidationError` (used by the System builder in Epic 2).

### Relation to Epic

This is the first ticket in Epic 01 (Foundation Types). It creates the file/module scaffolding and the types that all other entity definition tickets depend on. Without `EntityId`, no entity struct can define its `id` field or cross-reference fields.

### Current State

- `crates/cobre-core/src/lib.rs` contains only doc comments (lines 1-24), no code
- `crates/cobre-core/Cargo.toml` has no dependencies
- Workspace lints: `unsafe_code = "forbid"`, `missing_docs = "warn"`, `clippy::pedantic`, `unwrap_used = "deny"`

## Specification

### Requirements

1. Create the following module files (all initially containing only module-level doc comments and necessary declarations):
   - `crates/cobre-core/src/entity_id.rs`
   - `crates/cobre-core/src/entities/mod.rs`
   - `crates/cobre-core/src/entities/bus.rs`
   - `crates/cobre-core/src/entities/line.rs`
   - `crates/cobre-core/src/entities/hydro.rs`
   - `crates/cobre-core/src/entities/thermal.rs`
   - `crates/cobre-core/src/entities/pumping_station.rs`
   - `crates/cobre-core/src/entities/energy_contract.rs`
   - `crates/cobre-core/src/entities/non_controllable.rs`
   - `crates/cobre-core/src/error.rs`
   - `crates/cobre-core/src/topology/mod.rs`
   - `crates/cobre-core/src/topology/cascade.rs`
   - `crates/cobre-core/src/topology/network.rs`
   - `crates/cobre-core/src/system.rs`
   - `crates/cobre-core/src/penalty.rs`

2. Update `crates/cobre-core/src/lib.rs` to:
   - Declare all modules (`mod entity_id; mod entities; mod error; mod topology; mod system; mod penalty;`)
   - Re-export key types at crate root (`pub use entity_id::EntityId; pub use error::ValidationError;`)
   - Re-export entity types from `entities` module
   - Keep the existing crate-level doc comments (updated for accuracy)

3. Define `EntityId` in `entity_id.rs`:

   ```rust
   /// Strongly-typed entity identifier.
   ///
   /// Wraps the i32 identifier from JSON input files. The newtype pattern prevents
   /// accidental confusion between entity IDs and collection indices (usize).
   /// EntityId is used as the key in HashMap<EntityId, usize> lookup tables
   /// and as the value in cross-reference fields (e.g., Hydro::bus_id, Line::source_bus_id).
   #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
   pub struct EntityId(pub i32);
   ```

   - Implement `Display` for `EntityId` (shows the inner i32)
   - Implement `From<i32>` for `EntityId` and `From<EntityId>` for `i32`
   - Do NOT implement `Ord` (per spec: "No `Ord` -- Intentional")
   - Add unit tests for `EntityId`: equality, hashing (two EntityIds with same i32 hash equally), Copy semantics, Display output, From conversions

4. Define `ValidationError` in `error.rs`:
   ```rust
   /// Errors produced during System construction and validation.
   #[derive(Debug, Clone)]
   pub enum ValidationError {
       /// A cross-reference field (e.g., bus_id, downstream_id) refers to
       /// an entity ID that does not exist in the system.
       InvalidReference {
           /// The entity type containing the invalid reference.
           source_entity_type: &'static str,
           /// The ID of the entity containing the invalid reference.
           source_id: EntityId,
           /// The name of the field containing the invalid reference.
           field_name: &'static str,
           /// The referenced ID that does not exist.
           referenced_id: EntityId,
           /// The entity type that was expected.
           expected_type: &'static str,
       },
       /// Duplicate entity ID within a single entity collection.
       DuplicateId {
           entity_type: &'static str,
           id: EntityId,
       },
       /// The hydro cascade contains a cycle.
       CascadeCycle {
           /// IDs of hydros forming the cycle.
           cycle_ids: Vec<EntityId>,
       },
       /// A hydro's filling configuration is invalid.
       InvalidFillingConfig {
           hydro_id: EntityId,
           reason: String,
       },
       /// A bus has no connections (no lines, generators, or loads).
       DisconnectedBus {
           bus_id: EntityId,
       },
       /// Entity-level penalty value is invalid (e.g., negative cost).
       InvalidPenalty {
           entity_type: &'static str,
           entity_id: EntityId,
           field_name: &'static str,
           reason: String,
       },
   }
   ```

   - Implement `std::fmt::Display` for `ValidationError`
   - Implement `std::error::Error` for `ValidationError`
   - Add unit tests verifying Display output for each variant

### Inputs/Props

None -- this ticket creates new files from scratch.

### Outputs/Behavior

- All module files exist and compile
- `EntityId` type is usable from crate root: `cobre_core::EntityId`
- `ValidationError` type is usable from crate root: `cobre_core::ValidationError`
- Entity module files exist but are initially empty (just module-level doc comments)

### Error Handling

Not applicable for this ticket -- it defines the error types themselves.

## Acceptance Criteria

- [ ] Given a clean checkout, when running `cargo build -p cobre-core`, then the build succeeds with zero errors
- [ ] Given a clean checkout, when running `cargo clippy -p cobre-core`, then zero warnings are produced
- [ ] Given a clean checkout, when running `cargo doc -p cobre-core --no-deps`, then documentation builds without warnings
- [ ] Given the file `crates/cobre-core/src/entity_id.rs`, when inspecting its content, then it contains `pub struct EntityId(pub i32)` with derives `Debug, Clone, Copy, PartialEq, Eq, Hash` and NO `Ord` derive
- [ ] Given test code `let a = EntityId(1); let b = EntityId(1);`, when comparing `a == b`, then the assertion passes; and `let c = EntityId(2);` when comparing `a == c`, then the assertion fails

## Implementation Guide

### Suggested Approach

1. Create the directory structure: `src/entities/` and `src/topology/`
2. Create `entity_id.rs` with the `EntityId` struct and its impls (`Display`, `From<i32>`, `From<EntityId> for i32`)
3. Create `error.rs` with the `ValidationError` enum, `Display` impl, and `Error` impl
4. Create stub files for all entity modules (just `//! Module doc comment` and empty content)
5. Create stub files for topology modules
6. Create stub files for `system.rs` and `penalty.rs`
7. Update `lib.rs` with module declarations and public re-exports
8. Run `cargo build`, `cargo clippy`, `cargo doc`, and `cargo test`
9. Fix any clippy pedantic warnings (e.g., needless pass by value, missing docs)

### Key Files to Modify

- `crates/cobre-core/src/lib.rs` (rewrite)
- `crates/cobre-core/src/entity_id.rs` (new)
- `crates/cobre-core/src/error.rs` (new)
- `crates/cobre-core/src/entities/mod.rs` (new)
- `crates/cobre-core/src/entities/bus.rs` (new, stub)
- `crates/cobre-core/src/entities/line.rs` (new, stub)
- `crates/cobre-core/src/entities/hydro.rs` (new, stub)
- `crates/cobre-core/src/entities/thermal.rs` (new, stub)
- `crates/cobre-core/src/entities/pumping_station.rs` (new, stub)
- `crates/cobre-core/src/entities/energy_contract.rs` (new, stub)
- `crates/cobre-core/src/entities/non_controllable.rs` (new, stub)
- `crates/cobre-core/src/topology/mod.rs` (new, stub)
- `crates/cobre-core/src/topology/cascade.rs` (new, stub)
- `crates/cobre-core/src/topology/network.rs` (new, stub)
- `crates/cobre-core/src/system.rs` (new, stub)
- `crates/cobre-core/src/penalty.rs` (new, stub)

### Patterns to Follow

- Use `//!` doc comments for module-level documentation
- Use `///` doc comments for every `pub` item (workspace lint: `missing_docs = "warn"`)
- Follow the spec's derive lists exactly: `EntityId` gets `Debug, Clone, Copy, PartialEq, Eq, Hash` -- no more, no less
- Use `#[must_use]` on pure accessor methods returning non-unit types
- For `Display` impl on `ValidationError`, produce messages suitable for end-user error reporting
- The `cycle_ids` field in `CascadeCycle` uses `Vec<EntityId>` (not `Vec<i32>`) for type safety

### Pitfalls to Avoid

- Do NOT add `Ord` or `PartialOrd` to `EntityId` -- the spec explicitly forbids it
- Do NOT add `serde` derives -- no serde dependency in cobre-core for Phase 1
- Do NOT use `.unwrap()` anywhere -- workspace lint `unwrap_used = "deny"`
- Do NOT use `unsafe` -- workspace lint `unsafe_code = "forbid"`
- Stub module files must have at least a `//!` doc comment to avoid `missing_docs` warnings
- The `ValidationError::InvalidFillingConfig.reason` and `ValidationError::InvalidPenalty.reason` use `String` (not `&str`) because the error must be `'static` for `std::error::Error`

## Testing Requirements

### Unit Tests

- `entity_id::tests`:
  - `test_equality`: Two `EntityId` with same `i32` are equal; different `i32` are not equal
  - `test_copy`: `EntityId` implements `Copy` (assign to new variable, both still usable)
  - `test_hash_consistency`: Two equal `EntityId` produce the same hash (use `std::hash::Hash` + `DefaultHasher`)
  - `test_display`: `EntityId(42).to_string()` produces `"42"`
  - `test_from_i32`: `EntityId::from(5)` equals `EntityId(5)`
  - `test_into_i32`: `i32::from(EntityId(7))` equals `7`

- `error::tests`:
  - `test_display_invalid_reference`: Display output contains source entity type, field name, and referenced ID
  - `test_display_duplicate_id`: Display output contains entity type and ID
  - `test_display_cascade_cycle`: Display output contains the cycle IDs
  - `test_error_trait`: `ValidationError` can be used as `&dyn std::error::Error`

### Integration Tests

Not applicable for this ticket.

### E2E Tests

Not applicable for this ticket.

## Dependencies

- **Blocked By**: None -- this is the first ticket
- **Blocks**: ticket-002, ticket-003, ticket-004 (all entity definition tickets depend on `EntityId` and the module structure)

## Effort Estimate

**Points**: 2
**Confidence**: High
