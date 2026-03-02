# Epic 02: System Struct and Topology -- Learnings

## What Worked Well

1. **Clear separation of concerns**: Penalty resolution, cascade topology, network topology, and SystemBuilder were cleanly independent tickets with well-defined interfaces.

2. **Builder pattern with Result**: `SystemBuilder::build() -> Result<System, Vec<ValidationError>>` collects all errors at once, giving users a complete error report rather than failing on the first issue.

3. **Compile-time Send + Sync assertion**: The `const _: ()` trick caught thread-safety issues at compile time rather than runtime.

4. **OnceLock-backed static defaults**: For `NetworkTopology` accessors (`bus_generators`, `bus_loads`), using `std::sync::OnceLock` for module-level default values avoids allocation on the read path while keeping the API clean.

5. **Kahn's algorithm with deterministic ordering**: Using a sorted `Vec<i32>` as the ready queue for topological sort (since EntityId lacks Ord) maintains determinism without requiring Ord on the public type.

## Issues Encountered

1. **Rust 2024 edition dereference patterns**: `|(_, &deg)| deg == 0` is disallowed in closures under edition 2024. Must use `|&(_, deg)| *deg == 0` instead. This affected cascade topology code.

2. **clippy::struct_field_names**: NetworkTopology has fields `bus_lines`, `bus_generators`, `bus_loads` which all share the `bus_` prefix, triggering clippy. Suppressed with `#[allow(clippy::struct_field_names)]` and documented the reason.

3. **Pub fields vs accessor methods on System**: The spec says entity collections are `pub` AND provides accessor methods. This redundancy is intentional -- `pub` fields allow direct access for internal crate use, while accessor methods provide the public API for external consumers.

## Patterns Established

- **HasId trait**: Private trait for generic sort/dedup/index operations on entity collections
- **build_index helper**: Generic `fn build_index<T: HasId>(items: &[T]) -> HashMap<EntityId, usize>`
- **check_duplicates helper**: Generic duplicate detection using `.windows(2)` on sorted slices
- **Topology construction**: `Topology::build(&[Entity])` pattern -- takes borrowed slices, returns owned topology
- **Entity ID sorting**: `sort_by_key(|e| e.id.0)` since EntityId lacks Ord
- **Test helpers**: `make_bus(id, name)`, `make_hydro(id, downstream)` etc. for minimal test entity construction

## Metrics

| Ticket     | Readiness | Quality | Tests Added |
| ---------- | --------- | ------- | ----------- |
| ticket-005 | 0.98      | 1.00    | 17          |
| ticket-006 | 1.00      | 1.00    | 10          |
| ticket-007 | 1.00      | 1.00    | 7           |
| ticket-008 | 0.98      | 1.00    | 15          |

Mean readiness: 0.99 | Mean quality: 1.00 | Total tests: 49

## Recommendations for Epic 03

1. **Cross-reference validation (ticket-009)**: Should use the HasId pattern and lookup indices from System. Check bus_id on hydros/thermals/etc, downstream_id on hydros, source/target bus on lines.

2. **Cascade validation (ticket-010)**: Cycle detection should use CascadeTopology's existing structure. Consider DFS-based cycle detection using the downstream adjacency map.

3. **Integration tests (ticket-011)**: Test declaration-order invariance by building the same System with entities in different orders and comparing the result. Use the PartialEq derive on System for comparison.
