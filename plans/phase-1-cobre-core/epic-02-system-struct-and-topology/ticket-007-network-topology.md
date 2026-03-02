# ticket-007 Implement NetworkTopology

## Context

### Background

The `NetworkTopology` struct holds the resolved transmission network graph for buses and lines. It provides bus-line incidence (which lines are connected to each bus), bus generation maps (which generators are connected to each bus), and bus load maps (which loads/contracts/pumping stations are connected to each bus). This enables single-pass power balance constraint generation during LP construction in Phase 6.

### Relation to Epic

This is the third ticket in Epic 02 (System Struct and Topology). `NetworkTopology` is the second of two topology structures required by the `System` struct. It depends on `Bus`, `Line`, `Hydro`, `Thermal`, `NonControllableSource`, `EnergyContract`, and `PumpingStation` entities. The `SystemBuilder` (ticket-008) will construct `NetworkTopology` during System construction.

### Current State

- `crates/cobre-core/src/topology/network.rs` is a stub file (from ticket-001)
- All 7 entity types are defined (tickets 002-004)
- `EntityId` is defined (ticket-001)

## Specification

### Requirements

1. Define `NetworkTopology` in `topology/network.rs`:

   ```rust
   /// Resolved transmission network topology for buses and lines.
   ///
   /// Provides O(1) lookup for bus-line incidence, bus-to-generator maps,
   /// and bus-to-load maps. Built from entity collections during System
   /// construction and immutable thereafter.
   ///
   /// Used by cobre-sddp for power balance constraint generation.
   #[derive(Debug, Clone, PartialEq)]
   pub struct NetworkTopology {
       /// Bus-line incidence: bus_id -> list of (line_id, is_source).
       /// `is_source` is true when the bus is the source (direct flow direction).
       bus_lines: HashMap<EntityId, Vec<BusLineConnection>>,

       /// Bus generation map: bus_id -> list of generator IDs by type.
       bus_generators: HashMap<EntityId, BusGenerators>,

       /// Bus load map: bus_id -> list of load/demand entity IDs.
       bus_loads: HashMap<EntityId, BusLoads>,
   }

   /// A line connection from a bus perspective.
   #[derive(Debug, Clone, Copy, PartialEq, Eq)]
   pub struct BusLineConnection {
       /// The line's entity ID.
       pub line_id: EntityId,
       /// True if this bus is the line's source (direct flow direction).
       /// False if this bus is the line's target (reverse flow direction).
       pub is_source: bool,
   }

   /// Generator entities connected to a bus.
   #[derive(Debug, Clone, PartialEq, Default)]
   pub struct BusGenerators {
       /// Hydro plant IDs connected to this bus.
       pub hydro_ids: Vec<EntityId>,
       /// Thermal plant IDs connected to this bus.
       pub thermal_ids: Vec<EntityId>,
       /// Non-controllable source IDs connected to this bus.
       pub ncs_ids: Vec<EntityId>,
   }

   /// Load/demand entities connected to a bus.
   #[derive(Debug, Clone, PartialEq, Default)]
   pub struct BusLoads {
       /// Energy contract IDs at this bus.
       pub contract_ids: Vec<EntityId>,
       /// Pumping station IDs consuming power at this bus.
       pub pumping_station_ids: Vec<EntityId>,
   }
   ```

2. Implement a constructor:

   ```rust
   impl NetworkTopology {
       /// Build network topology from entity collections.
       ///
       /// Constructs bus-line incidence, bus generation maps, and bus load maps
       /// from the entity collections. Does not validate (no bus existence checks) --
       /// validation is separate.
       ///
       /// All entity slices are assumed to be in canonical ID order.
       pub fn build(
           buses: &[Bus],
           lines: &[Line],
           hydros: &[Hydro],
           thermals: &[Thermal],
           non_controllable_sources: &[NonControllableSource],
           contracts: &[EnergyContract],
           pumping_stations: &[PumpingStation],
       ) -> Self;
   }
   ```

3. Implement public accessor methods:

   ```rust
   impl NetworkTopology {
       /// Returns the lines connected to a bus.
       /// Returns an empty slice if the bus has no connected lines.
       pub fn bus_lines(&self, bus_id: EntityId) -> &[BusLineConnection];

       /// Returns the generators connected to a bus.
       /// Returns a default (empty) BusGenerators if the bus has no generators.
       pub fn bus_generators(&self, bus_id: EntityId) -> &BusGenerators;

       /// Returns the loads connected to a bus.
       /// Returns a default (empty) BusLoads if the bus has no loads.
       pub fn bus_loads(&self, bus_id: EntityId) -> &BusLoads;
   }
   ```

4. Update `topology/mod.rs` to declare and re-export the new types.

5. Update `lib.rs` to re-export `NetworkTopology`, `BusLineConnection`, `BusGenerators`, `BusLoads`.

### Inputs/Props

- All 7 entity type slices (all in canonical ID order)
- `EntityId` for all references

### Outputs/Behavior

- `bus_lines(bus_id)` returns all lines connected to that bus with direction info
- `bus_generators(bus_id)` returns hydro, thermal, and NCS IDs connected to that bus
- `bus_loads(bus_id)` returns contract and pumping station IDs at that bus
- For buses with no connections, accessor methods return empty collections
- All ID lists within each map entry are in canonical (i32-ascending) order for determinism

### Error Handling

No validation in the `build` method. Bus existence checks are performed in Epic 3. The topology simply records what the entity data says.

## Acceptance Criteria

- [ ] Given 2 buses (id=0, id=1) and 1 line (id=0, source_bus_id=0, target_bus_id=1), when calling `NetworkTopology::build`, then `bus_lines(EntityId(0))` returns one connection with `line_id=EntityId(0), is_source=true` and `bus_lines(EntityId(1))` returns one connection with `line_id=EntityId(0), is_source=false`
- [ ] Given a bus (id=0) with 2 hydros (bus_id=0) and 1 thermal (bus_id=0), when calling `NetworkTopology::build`, then `bus_generators(EntityId(0)).hydro_ids` has length 2 and `bus_generators(EntityId(0)).thermal_ids` has length 1
- [ ] Given a bus with no generators or loads, when calling `bus_generators(bus_id)`, then all ID lists are empty
- [ ] Given a clean checkout, when running `cargo clippy -p cobre-core`, then zero warnings are produced

## Implementation Guide

### Suggested Approach

1. Define `BusLineConnection`, `BusGenerators`, `BusLoads` supporting structs
2. Define `NetworkTopology` struct with private fields
3. Implement `build`:
   a. Initialize empty HashMaps
   b. Iterate lines: for each line, add a `BusLineConnection` to both `source_bus_id` (is_source=true) and `target_bus_id` (is_source=false)
   c. Iterate hydros: add each `hydro.id` to `bus_generators[hydro.bus_id].hydro_ids`
   d. Iterate thermals: add each `thermal.id` to `bus_generators[thermal.bus_id].thermal_ids`
   e. Iterate NCS: add each `ncs.id` to `bus_generators[ncs.bus_id].ncs_ids`
   f. Iterate contracts: add each `contract.id` to `bus_loads[contract.bus_id].contract_ids`
   g. Iterate pumping stations: add each `station.id` to `bus_loads[station.bus_id].pumping_station_ids`
   h. Sort all ID lists by inner i32 for determinism
4. Implement accessor methods with empty-slice fallback for missing keys
5. Write unit tests

### Key Files to Modify

- `crates/cobre-core/src/topology/network.rs` (populate from stub)
- `crates/cobre-core/src/topology/mod.rs` (add module decl and re-exports)
- `crates/cobre-core/src/lib.rs` (add NetworkTopology re-exports)

### Patterns to Follow

- Use `HashMap::entry(key).or_default()` to initialize map entries on first access
- Sort all Vec entries by inner i32 after construction for determinism (declaration-order invariance)
- Accessor methods return `&[T]` or `&Struct` with fallback to empty/default values
- For `bus_generators` and `bus_loads`, use a module-level `static` or `lazy_static` default, or return a reference to a locally-owned default -- the simplest approach is to store a default in the struct or use `HashMap::get().unwrap_or(&DEFAULT)` with a const default

### Pitfalls to Avoid

- The `build` method must handle bus_id references that don't exist in the `buses` slice (store them anyway; validation catches this)
- Must sort all internal Vec by EntityId's inner i32 for deterministic behavior
- EntityId lacks Ord, so sort by `.0` (the inner i32)
- The `bus_generators` and `bus_loads` accessors need a fallback for missing keys -- consider storing a static default or using an approach that returns `&BusGenerators` without allocating
- `BusLineConnection` needs `Eq` (not just `PartialEq`) because all its fields are `Eq` (`EntityId` is `Eq`, `bool` is `Eq`)

## Testing Requirements

### Unit Tests

In `topology/network.rs` (`#[cfg(test)] mod tests`):

To construct test entities, create small helper functions that build minimal Bus/Line/Hydro/Thermal/etc. instances with only the fields needed (all other fields use sensible defaults).

- `test_empty_network`: All entity slices empty -> all accessors return empty
- `test_single_line`: One line connecting two buses -> each bus sees one connection with correct direction
- `test_multiple_lines_same_bus`: Bus with 3 lines -> `bus_lines` returns 3 connections
- `test_generators_per_bus`: Two hydros and one thermal on bus 0, one NCS on bus 1 -> correct generator maps
- `test_loads_per_bus`: One contract and one pumping station on bus 0 -> correct load map
- `test_bus_no_connections`: Bus exists but nothing connected -> empty generators, loads, lines
- `test_deterministic_ordering`: Generator and load ID lists are sorted by inner i32

### Integration Tests

Not applicable for this ticket.

### E2E Tests

Not applicable for this ticket.

## Dependencies

- **Blocked By**: ticket-002 (Bus, Line), ticket-003 (Hydro), ticket-004 (Thermal, PumpingStation, EnergyContract, NonControllableSource)
- **Blocks**: ticket-008 (SystemBuilder constructs NetworkTopology)

## Effort Estimate

**Points**: 2
**Confidence**: High
