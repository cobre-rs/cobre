# Epic 02: System Struct and Topology

## Goal

Implement the `System` container struct with its full public API, the `CascadeTopology` and `NetworkTopology` structures, entity-level penalty resolution logic (`GlobalPenaltyDefaults`), and the `SystemBuilder` that constructs an immutable, validated `System` from entity collections. After this epic, a valid `System` can be constructed programmatically (from test helpers), passed around as `&System`, and queried through the spec-defined API surface.

## Scope

- `CascadeTopology`: downstream/upstream adjacency, travel times, topological ordering of hydros
- `NetworkTopology`: bus-line incidence, bus generation map (hydros/thermals/NCS per bus), bus load map (contracts/pumping per bus)
- `GlobalPenaltyDefaults`: global default penalty values for all entity types (matching `penalties.json` structure)
- Entity-level penalty resolution: global defaults -> entity overrides for Bus (deficit_segments, excess_cost), Line (exchange_cost), Hydro (HydroPenalties), NonControllableSource (curtailment_cost)
- `SystemBuilder`: accepts entity Vecs, global penalty defaults; sorts by ID, resolves entity-level penalties, builds topology, validates, returns `Result<System, Vec<ValidationError>>`
- `System` struct with:
  - 7 entity collection Vecs (public)
  - 7 HashMap lookup indices (private)
  - CascadeTopology and NetworkTopology (public)
  - Slice accessors, count queries, entity lookup by ID methods
  - `Send + Sync` guarantee (compile-time check)

## Non-Goals

- Stage-varying penalty resolution (Phase 2)
- `ResolvedPenalties` and `ResolvedBounds` per (entity, stage) (Phase 2)
- Stages, PolicyGraph, ParModel, CorrelationModel (Phase 2+)
- JSON/Parquet parsing (Phase 2)
- Cross-reference validation (Epic 3)
- Cascade cycle detection (Epic 3)
- Bus connectivity validation (Epic 3)

## Tickets

1. **ticket-005**: Implement GlobalPenaltyDefaults and entity-level penalty resolution
2. **ticket-006**: Implement CascadeTopology
3. **ticket-007**: Implement NetworkTopology
4. **ticket-008**: Implement SystemBuilder and System struct with public API

## Dependencies

- Blocked by: Epic 01 (all entity types must be defined)
- Blocks: Epic 03 (validation requires System construction)

## Success Criteria

- A `System` can be constructed from entity Vecs via `SystemBuilder`
- Entity collections are sorted by ID in the constructed System
- `system.hydro(EntityId(5))` returns the correct `&Hydro` via O(1) HashMap lookup
- `system.n_buses()` returns the count
- `CascadeTopology` provides topological ordering of hydros
- `NetworkTopology` provides bus-line incidence
- Entity-level penalties are resolved correctly (global defaults with entity overrides)
- `System` is `Send + Sync` (verified by compile-time assertion)
