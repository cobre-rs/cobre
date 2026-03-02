# ticket-002 Define Bus and Line Entity Structs

## Context

### Background

With the crate module structure and `EntityId` type in place (ticket-001), we can now define the first entity types. Bus and Line are the simplest entity types in the system and form the transmission network. They are grouped together because they are closely related (lines connect buses) and share similar complexity.

### Relation to Epic

This is the second ticket in Epic 01 (Foundation Types). It populates the `bus.rs` and `line.rs` entity modules with complete struct definitions matching the internal-structures spec section 1.9.2 (Bus) and 1.9.3 (Line).

### Current State

- `crates/cobre-core/src/entities/bus.rs` is a stub file (module-level doc comment only, from ticket-001)
- `crates/cobre-core/src/entities/line.rs` is a stub file (module-level doc comment only, from ticket-001)
- `EntityId` is defined and usable from crate root

## Specification

### Requirements

1. Define `DeficitSegment` in `entities/bus.rs`:

   ```rust
   /// A single segment of the piecewise-linear deficit cost curve.
   ///
   /// Segments are cumulative: the first depth_mw MW of deficit costs cost_per_mwh,
   /// the next segment's depth_mw MW costs that segment's cost_per_mwh, and so on.
   /// The final segment has depth_mw = None (extends to infinity).
   #[derive(Debug, Clone, PartialEq)]
   pub struct DeficitSegment {
       /// MW of deficit covered by this segment [MW]. None for the final unbounded segment.
       pub depth_mw: Option<f64>,
       /// Cost per MWh of deficit in this segment [$/MWh].
       pub cost_per_mwh: f64,
   }
   ```

2. Define `Bus` in `entities/bus.rs`:

   ```rust
   /// Electrical network node where energy balance is maintained.
   ///
   /// Buses represent aggregation points in the transmission network -- regional
   /// subsystems, substations, or any user-defined grouping. Each bus has a
   /// piecewise-linear deficit cost curve that ensures LP feasibility when demand
   /// cannot be met.
   ///
   /// Source: system/buses.json. See Input System Entities SS1.
   #[derive(Debug, Clone, PartialEq)]
   pub struct Bus {
       /// Unique bus identifier.
       pub id: EntityId,
       /// Human-readable bus name.
       pub name: String,
       /// Pre-resolved piecewise-linear deficit cost segments.
       /// Segments are ordered by ascending cost. The final segment has depth_mw = None
       /// (unbounded) to ensure LP feasibility.
       pub deficit_segments: Vec<DeficitSegment>,
       /// Cost per MWh for surplus generation absorption [$/MWh].
       pub excess_cost: f64,
   }
   ```

3. Define `Line` in `entities/line.rs`:

   ```rust
   /// Transmission interconnection between two buses.
   ///
   /// Lines allow bidirectional power transfer subject to capacity limits and
   /// transmission losses. Line flow is a hard constraint (no slack variables) --
   /// the exchange_cost is a regularization penalty, not a violation penalty.
   ///
   /// Source: system/lines.json. See Input System Entities SS2.
   #[derive(Debug, Clone, PartialEq)]
   pub struct Line {
       /// Unique line identifier.
       pub id: EntityId,
       /// Human-readable line name.
       pub name: String,
       /// Source bus for direct flow direction.
       pub source_bus_id: EntityId,
       /// Target bus for direct flow direction.
       pub target_bus_id: EntityId,
       /// Stage when line enters service. None = always exists.
       pub entry_stage_id: Option<i32>,
       /// Stage when line is decommissioned. None = never decommissioned.
       pub exit_stage_id: Option<i32>,
       /// Maximum flow from source to target [MW]. Hard bound.
       pub direct_capacity_mw: f64,
       /// Maximum flow from target to source [MW]. Hard bound.
       pub reverse_capacity_mw: f64,
       /// Transmission losses as percentage (e.g., 2.5 means 2.5%).
       pub losses_percent: f64,
       /// Regularization cost per MWh exchanged [$/MWh].
       pub exchange_cost: f64,
   }
   ```

4. Update `entities/mod.rs` to declare and re-export:
   - `pub mod bus;` and `pub use bus::{Bus, DeficitSegment};`
   - `pub mod line;` and `pub use line::Line;`

5. Update `lib.rs` re-exports to include `Bus`, `DeficitSegment`, `Line` from the entities module.

6. Add unit tests for each type covering construction and field access.

### Inputs/Props

- `EntityId` from `entity_id.rs` (ticket-001)

### Outputs/Behavior

- `Bus`, `DeficitSegment`, and `Line` are accessible as `cobre_core::Bus`, `cobre_core::DeficitSegment`, `cobre_core::Line`
- All types derive `Debug, Clone, PartialEq`
- All public fields and types have `///` doc comments

### Error Handling

No error handling in this ticket -- these are pure data structs. Validation is in Epic 3.

## Acceptance Criteria

- [ ] Given the file `crates/cobre-core/src/entities/bus.rs`, when inspecting its content, then it contains `pub struct Bus` with fields `id: EntityId`, `name: String`, `deficit_segments: Vec<DeficitSegment>`, `excess_cost: f64` and `pub struct DeficitSegment` with fields `depth_mw: Option<f64>`, `cost_per_mwh: f64`
- [ ] Given the file `crates/cobre-core/src/entities/line.rs`, when inspecting its content, then it contains `pub struct Line` with exactly 10 fields matching the spec (id, name, source_bus_id, target_bus_id, entry_stage_id, exit_stage_id, direct_capacity_mw, reverse_capacity_mw, losses_percent, exchange_cost)
- [ ] Given test code constructing a `Bus` with 3 deficit segments where the last has `depth_mw: None`, when accessing `bus.deficit_segments[2].depth_mw`, then the result is `None`
- [ ] Given a clean checkout, when running `cargo clippy -p cobre-core`, then zero warnings are produced
- [ ] Given a clean checkout, when running `cargo test -p cobre-core`, then all tests pass including the new Bus and Line tests

## Implementation Guide

### Suggested Approach

1. Implement `DeficitSegment` struct in `bus.rs` with derives and doc comments
2. Implement `Bus` struct in `bus.rs` with derives and doc comments
3. Implement `Line` struct in `line.rs` with derives and doc comments
4. Update `entities/mod.rs` with module declarations and re-exports
5. Update `lib.rs` to re-export the new entity types
6. Write unit tests for `Bus` and `Line` construction
7. Run `cargo build`, `cargo clippy`, `cargo doc`, `cargo test`

### Key Files to Modify

- `crates/cobre-core/src/entities/bus.rs` (populate from stub)
- `crates/cobre-core/src/entities/line.rs` (populate from stub)
- `crates/cobre-core/src/entities/mod.rs` (add module decls and re-exports)
- `crates/cobre-core/src/lib.rs` (add entity re-exports)

### Patterns to Follow

- Match the spec section 1.9.2 and 1.9.3 field-by-field -- do not add or remove fields
- Doc comments must include units in brackets: `/// Minimum storage (dead volume) [hm3]`
- Use `Option<i32>` for nullable stage IDs (not `Option<EntityId>` -- stage IDs are not entity IDs)
- The `excess_cost` field on `Bus` is a "Resolved" field per the spec (defaults applied during loading) but is defined as a plain `f64` in the struct -- resolution happens in the SystemBuilder

### Pitfalls to Avoid

- Do NOT derive `Serialize`/`Deserialize` -- no serde in Phase 1
- Do NOT add `Ord` to any type (unless clippy pedantic requires it for a specific reason; if so, document why)
- Do NOT use `.unwrap()` in tests -- use `assert_eq!` directly or pattern matching
- `DeficitSegment` derives `Clone` (not `Copy`) because `Option<f64>` is `Copy` but `Vec` would not be if we needed it later -- actually `DeficitSegment` is small and could be `Copy`, but the spec shows `#[derive(Debug, Clone, PartialEq)]` so match that exactly
- Bus name is `String` not `&str` -- the struct owns its data

## Testing Requirements

### Unit Tests

In `entities/bus.rs` (`#[cfg(test)] mod tests`):

- `test_bus_construction`: Create a `Bus` with 3 deficit segments (last unbounded), verify all fields accessible
- `test_deficit_segment_unbounded`: Create a `DeficitSegment` with `depth_mw: None`, verify it is the unbounded final segment
- `test_bus_equality`: Two `Bus` structs with identical fields are equal; changing one field makes them not equal

In `entities/line.rs` (`#[cfg(test)] mod tests`):

- `test_line_construction`: Create a `Line` with all fields, verify field access
- `test_line_lifecycle_always`: Line with `entry_stage_id: None, exit_stage_id: None` represents "always exists"
- `test_line_lifecycle_bounded`: Line with `entry_stage_id: Some(5), exit_stage_id: Some(120)` represents bounded lifecycle
- `test_line_equality`: Two identical `Line` structs are equal

### Integration Tests

Not applicable for this ticket.

### E2E Tests

Not applicable for this ticket.

## Dependencies

- **Blocked By**: ticket-001 (EntityId and module structure)
- **Blocks**: ticket-005 (SystemBuilder needs Bus and Line), ticket-007 (NetworkTopology needs Bus and Line)

## Effort Estimate

**Points**: 1
**Confidence**: High
