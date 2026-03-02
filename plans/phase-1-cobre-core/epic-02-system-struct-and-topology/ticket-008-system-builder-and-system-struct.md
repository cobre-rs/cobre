# ticket-008 Implement SystemBuilder and System Struct with Public API

## Context

### Background

The `System` struct is the top-level container for the entire power system data model. It holds entity collections, lookup indices, and topology. It is produced by cobre-io in Phase 2 and consumed read-only by cobre-sddp. In Phase 1, the `SystemBuilder` provides a programmatic construction path (used by tests and eventually by cobre-io). The builder accepts entity Vecs, sorts them by ID (canonical ordering), checks for duplicate IDs, constructs topology, and produces an immutable `System`.

### Relation to Epic

This is the fourth and final ticket in Epic 02 (System Struct and Topology). It ties together all entity types (Epic 01), penalty resolution (ticket-005), and both topology structures (tickets 006-007) into the `System` container. After this ticket, a complete `System` can be constructed and queried through the spec-defined API surface.

### Current State

- `crates/cobre-core/src/system.rs` is a stub file (from ticket-001)
- All 7 entity types are defined (tickets 002-004)
- `GlobalPenaltyDefaults`, `HydroPenaltyOverrides`, and resolution functions are defined (ticket-005)
- `CascadeTopology` is defined with `build()` (ticket-006)
- `NetworkTopology` is defined with `build()` (ticket-007)
- `ValidationError` is defined with `DuplicateId` variant (ticket-001)

## Specification

### Requirements

1. Define the `System` struct in `system.rs` matching spec section 1.3 (Phase 1 subset):

   ```rust
   /// Top-level system representation.
   ///
   /// Produced by cobre-io (Phase 2) or SystemBuilder (Phase 1 tests).
   /// Consumed by cobre-sddp and cobre-stochastic by shared reference.
   /// Immutable after construction. Shared read-only across threads.
   ///
   /// Entity collections are in canonical order (sorted by EntityId's inner i32).
   /// Lookup indices provide O(1) access by EntityId.
   pub struct System {
       // Entity collections (canonical ordering by ID) -- public
       buses: Vec<Bus>,
       lines: Vec<Line>,
       hydros: Vec<Hydro>,
       thermals: Vec<Thermal>,
       pumping_stations: Vec<PumpingStation>,
       contracts: Vec<EnergyContract>,
       non_controllable_sources: Vec<NonControllableSource>,

       // O(1) lookup indices (entity ID -> position in collection) -- private
       bus_index: HashMap<EntityId, usize>,
       line_index: HashMap<EntityId, usize>,
       hydro_index: HashMap<EntityId, usize>,
       thermal_index: HashMap<EntityId, usize>,
       pumping_station_index: HashMap<EntityId, usize>,
       contract_index: HashMap<EntityId, usize>,
       non_controllable_source_index: HashMap<EntityId, usize>,

       // Topology
       cascade: CascadeTopology,
       network: NetworkTopology,
   }
   ```

   Note: The spec's `System` includes `stages`, `policy_graph`, `penalties`, `bounds`, `par_models`, `correlation`, `initial_conditions`, and `generic_constraints`. These are Phase 2+ fields and are NOT included in Phase 1.

2. Implement the spec's public API surface (section 1.4) as methods on `System`:

   **Entity collection accessors**:

   ```rust
   impl System {
       pub fn buses(&self) -> &[Bus];
       pub fn lines(&self) -> &[Line];
       pub fn hydros(&self) -> &[Hydro];
       pub fn thermals(&self) -> &[Thermal];
       pub fn pumping_stations(&self) -> &[PumpingStation];
       pub fn contracts(&self) -> &[EnergyContract];
       pub fn non_controllable_sources(&self) -> &[NonControllableSource];
   }
   ```

   **Entity count queries**:

   ```rust
   impl System {
       pub fn n_buses(&self) -> usize;
       pub fn n_lines(&self) -> usize;
       pub fn n_hydros(&self) -> usize;
       pub fn n_thermals(&self) -> usize;
       pub fn n_pumping_stations(&self) -> usize;
       pub fn n_contracts(&self) -> usize;
       pub fn n_non_controllable_sources(&self) -> usize;
   }
   ```

   **Entity lookup by ID** (O(1) via HashMap):

   ```rust
   impl System {
       pub fn bus(&self, id: EntityId) -> Option<&Bus>;
       pub fn line(&self, id: EntityId) -> Option<&Line>;
       pub fn hydro(&self, id: EntityId) -> Option<&Hydro>;
       pub fn thermal(&self, id: EntityId) -> Option<&Thermal>;
       pub fn pumping_station(&self, id: EntityId) -> Option<&PumpingStation>;
       pub fn contract(&self, id: EntityId) -> Option<&EnergyContract>;
       pub fn non_controllable_source(&self, id: EntityId) -> Option<&NonControllableSource>;
   }
   ```

   **Topology accessors**:

   ```rust
   impl System {
       pub fn cascade(&self) -> &CascadeTopology;
       pub fn network(&self) -> &NetworkTopology;
   }
   ```

3. Implement `SystemBuilder`:

   ```rust
   /// Builder for constructing a validated, immutable System.
   ///
   /// Accepts entity collections and global penalty defaults, sorts entities
   /// by ID, checks for duplicate IDs, builds topology, and returns the System.
   pub struct SystemBuilder {
       buses: Vec<Bus>,
       lines: Vec<Line>,
       hydros: Vec<Hydro>,
       thermals: Vec<Thermal>,
       pumping_stations: Vec<PumpingStation>,
       contracts: Vec<EnergyContract>,
       non_controllable_sources: Vec<NonControllableSource>,
   }

   impl SystemBuilder {
       pub fn new() -> Self;
       pub fn buses(self, buses: Vec<Bus>) -> Self;
       pub fn lines(self, lines: Vec<Line>) -> Self;
       pub fn hydros(self, hydros: Vec<Hydro>) -> Self;
       pub fn thermals(self, thermals: Vec<Thermal>) -> Self;
       pub fn pumping_stations(self, stations: Vec<PumpingStation>) -> Self;
       pub fn contracts(self, contracts: Vec<EnergyContract>) -> Self;
       pub fn non_controllable_sources(self, sources: Vec<NonControllableSource>) -> Self;

       /// Build the System.
       ///
       /// Sorts all entity collections by EntityId (canonical ordering).
       /// Checks for duplicate IDs within each collection.
       /// Builds CascadeTopology and NetworkTopology.
       /// Constructs lookup indices.
       ///
       /// Returns Err with a list of all validation errors found.
       /// Currently only checks for duplicate IDs. Cross-reference and
       /// topology validation is added in Epic 3.
       pub fn build(self) -> Result<System, Vec<ValidationError>>;
   }
   ```

4. Add a compile-time `Send + Sync` assertion for `System`:

   ```rust
   // Compile-time check that System is Send + Sync.
   const _: () = {
       fn assert_send_sync<T: Send + Sync>() {}
       fn check() { assert_send_sync::<System>(); }
   };
   ```

5. Update `lib.rs` to re-export `System` and `SystemBuilder`.

### Inputs/Props

- Entity Vecs (possibly unsorted, possibly with duplicates)
- Types from all previous tickets

### Outputs/Behavior

- `SystemBuilder::new().buses(v).lines(v)...build()` returns `Ok(System)` for valid input
- Entity collections in the returned System are sorted by `EntityId.0` (canonical ordering)
- `system.bus(EntityId(5))` returns `Some(&bus)` if a bus with id=5 exists, `None` otherwise
- `system.n_hydros()` returns the number of hydros
- `system.cascade()` returns the CascadeTopology
- `System` is `Send + Sync` (compile-time verified)
- If duplicate IDs exist within any collection, `build()` returns `Err` with `DuplicateId` errors

### Error Handling

- `build()` returns `Result<System, Vec<ValidationError>>` to collect multiple errors
- Duplicate ID detection: for each entity collection after sorting, check adjacent elements for equal IDs
- Cross-reference validation and cycle detection are NOT in this ticket (Epic 3)

## Acceptance Criteria

- [ ] Given a `SystemBuilder` with 2 buses (id=1, id=0) provided in non-canonical order, when calling `build()`, then `Ok(system)` is returned and `system.buses()[0].id == EntityId(0)` and `system.buses()[1].id == EntityId(1)` (sorted)
- [ ] Given a `SystemBuilder` with a hydro having `id=5`, when calling `build()` and then `system.hydro(EntityId(5))`, then `Some(&hydro)` is returned with `id == EntityId(5)`
- [ ] Given a `SystemBuilder` with two buses having the same id=0, when calling `build()`, then `Err(errors)` is returned with at least one `ValidationError::DuplicateId` for buses
- [ ] Given the file `crates/cobre-core/src/system.rs`, when it compiles, then the `Send + Sync` assertion passes (System is thread-safe)
- [ ] Given a clean checkout, when running `cargo clippy -p cobre-core`, then zero warnings are produced

## Implementation Guide

### Suggested Approach

1. Define the `System` struct with all fields (entity Vecs, lookup HashMaps, topology)
2. Implement all accessor methods on `System` (straightforward: return slice/len/HashMap get)
3. Define `SystemBuilder` with builder methods (each stores the Vec)
4. Implement `build()`:
   a. Sort each entity Vec by `entity.id.0` (inner i32). Use `sort_by_key(|e| e.id.0)` -- this is efficient if already sorted per the spec note.
   b. Check for duplicate IDs: after sorting, iterate with `.windows(2)` and check if adjacent elements have equal ids
   c. If duplicates found, collect errors but continue checking other collections
   d. Build lookup indices: iterate each sorted Vec, insert `(entity.id, index)` into HashMap
   e. Build `CascadeTopology::build(&hydros)`
   f. Build `NetworkTopology::build(&buses, &lines, &hydros, &thermals, &ncs, &contracts, &pumping_stations)`
   g. Construct `System` and return
5. Add the `Send + Sync` compile-time assertion
6. Add `Default` impl for `SystemBuilder`
7. Write tests

### Key Files to Modify

- `crates/cobre-core/src/system.rs` (populate from stub)
- `crates/cobre-core/src/lib.rs` (add System and SystemBuilder re-exports)

### Patterns to Follow

- Builder pattern: each setter takes `self` by value and returns `Self` for chaining
- `sort_by_key(|e| e.id.0)` for canonical ordering (EntityId lacks Ord but i32 has Ord)
- `HashMap::with_capacity(vec.len())` for pre-allocated lookup indices
- Accessor methods use `#[must_use]` attribute
- Lookup by ID: `self.bus_index.get(&id).map(|&i| &self.buses[i])`
- `n_buses()` returns `self.buses.len()`
- `buses()` returns `&self.buses`

### Pitfalls to Avoid

- Do NOT make entity collections `pub` -- they are accessed through the accessor methods. The spec shows them as `pub` but the lookup indices are private; for Phase 1, keep collections accessible through accessors (the struct fields themselves can be `pub(crate)` or private with pub accessor methods)
- Actually, re-reading the spec: entity collections ARE `pub` and lookup indices are private. Match the spec: make entity collection fields `pub` and index fields private.
- Wait -- if entity collections are `pub`, then `system.buses` returns `Vec<Bus>` directly. But the accessor `system.buses()` returns `&[Bus]`. Having both is redundant. The spec defines both pub fields AND accessor methods. Follow the spec: make collections pub, implement accessor methods as well (they return slices of the pub fields). This allows both `system.buses` (field access) and `system.buses()` (method returning slice).
- [ASSUMPTION] Making entity collections `pub` matches the spec sketch, but it allows mutation through `system.buses.push(...)`. Since the System is conceptually immutable, the accessor method approach (private fields + pub fn) is safer. However, the spec explicitly says "Public for read access" on entity collections. Resolution: use `pub` fields as the spec says. The SystemBuilder pattern ensures construction is controlled, and Rust's ownership model prevents mutation through shared references (`&System`). If a user has `mut System`, they own it and can modify it -- this is acceptable.
- Do NOT skip empty collections -- a System with zero hydros is valid
- Do NOT add stages, policy_graph, or other Phase 2+ fields
- The `build()` method signature returns `Vec<ValidationError>` (plural) to report all errors at once

## Testing Requirements

### Unit Tests

In `system.rs` (`#[cfg(test)] mod tests`):

Create test helper functions to build minimal entities (e.g., `fn make_bus(id: i32, name: &str) -> Bus`).

- `test_empty_system`: SystemBuilder with no entities builds successfully, all counts are 0
- `test_canonical_ordering`: Provide buses in reverse order (id=2, id=1, id=0), verify after build they are sorted [0, 1, 2]
- `test_lookup_by_id`: Build system with 3 hydros, verify `system.hydro(EntityId(X))` returns correct hydro for each
- `test_lookup_missing_id`: `system.hydro(EntityId(999))` returns `None`
- `test_count_queries`: Build with known entity counts, verify `n_buses()`, `n_hydros()`, etc.
- `test_slice_accessors`: `system.buses()` returns a slice with the correct length and content
- `test_duplicate_id_error`: Two buses with id=0 -> `build()` returns `Err` with `DuplicateId`
- `test_multiple_duplicate_errors`: Duplicates in both buses and thermals -> `Err` contains errors for both
- `test_send_sync`: Compile-time assertion passes (tested by compilation, not runtime)
- `test_cascade_accessible`: Build system with hydros, verify `system.cascade().topological_order()` is non-empty
- `test_network_accessible`: Build system with buses and lines, verify `system.network().bus_lines(bus_id)` returns connections

### Integration Tests

Not applicable for this ticket (full integration tests are in Epic 3).

### E2E Tests

Not applicable for this ticket.

## Dependencies

- **Blocked By**: ticket-005 (penalty resolution), ticket-006 (CascadeTopology), ticket-007 (NetworkTopology)
- **Blocks**: ticket-009, ticket-010, ticket-011 (Epic 3 validation tickets)

## Effort Estimate

**Points**: 3
**Confidence**: High
