# ticket-006 Implement CascadeTopology

## Context

### Background

The hydro cascade is a directed graph where each hydro plant has at most one downstream plant. The `CascadeTopology` struct holds the resolved cascade graph: downstream adjacency, upstream adjacency (derived from downstream references), and a topological ordering that enables single-pass cascade traversal for water balance computation. This topology is built once during System construction and is immutable thereafter.

### Relation to Epic

This is the second ticket in Epic 02 (System Struct and Topology). `CascadeTopology` is one of two topology structures required by the `System` struct. It depends only on `Hydro` entities (specifically their `id` and `downstream_id` fields). The `SystemBuilder` (ticket-008) will construct `CascadeTopology` during System construction.

### Current State

- `crates/cobre-core/src/topology/cascade.rs` is a stub file (from ticket-001)
- `Hydro` entity is defined with `downstream_id: Option<EntityId>` (ticket-003)
- `EntityId` is defined (ticket-001)

## Specification

### Requirements

1. Define `CascadeTopology` in `topology/cascade.rs`:

   ```rust
   /// Resolved hydro cascade graph for water balance traversal.
   ///
   /// The cascade is a directed forest (collection of trees) where each hydro has
   /// at most one downstream plant. Terminal nodes have no downstream plant.
   /// The topology is built from hydro `downstream_id` fields during System
   /// construction and is immutable thereafter.
   ///
   /// The topological order enables single-pass water balance computation:
   /// processing hydros in topological order guarantees that upstream inflows
   /// are computed before the downstream plant that receives them.
   #[derive(Debug, Clone, PartialEq)]
   pub struct CascadeTopology {
       /// Downstream adjacency: hydro_id -> downstream hydro_id.
       /// Terminal nodes are not present in the map.
       downstream: HashMap<EntityId, EntityId>,

       /// Upstream adjacency: hydro_id -> list of upstream hydro_ids.
       /// Hydros with no upstream plants are not present in the map.
       upstream: HashMap<EntityId, Vec<EntityId>>,

       /// Topological ordering of all hydro IDs.
       /// Every upstream plant appears before its downstream plant.
       /// Within the same topological level, order is by EntityId's inner i32
       /// to ensure determinism.
       topological_order: Vec<EntityId>,
   }
   ```

2. Implement a constructor:

   ```rust
   impl CascadeTopology {
       /// Build cascade topology from hydro entities.
       ///
       /// Constructs the downstream adjacency map, derives upstream adjacency,
       /// and computes topological order. Does not validate (no cycle detection) --
       /// validation is separate.
       ///
       /// # Arguments
       /// * `hydros` - Slice of hydro entities, assumed to be in canonical ID order.
       pub fn build(hydros: &[Hydro]) -> Self;
   }
   ```

3. Implement public accessor methods:

   ```rust
   impl CascadeTopology {
       /// Returns the downstream hydro for the given hydro, if any.
       pub fn downstream(&self, hydro_id: EntityId) -> Option<EntityId>;

       /// Returns the upstream hydros for the given hydro, if any.
       /// Returns an empty slice if the hydro has no upstream plants.
       pub fn upstream(&self, hydro_id: EntityId) -> &[EntityId];

       /// Returns the topological ordering of all hydro IDs.
       /// Every upstream plant appears before its downstream plant.
       pub fn topological_order(&self) -> &[EntityId];

       /// Returns true if the given hydro is a headwater (no upstream plants).
       pub fn is_headwater(&self, hydro_id: EntityId) -> bool;

       /// Returns true if the given hydro is a terminal node (no downstream plant).
       pub fn is_terminal(&self, hydro_id: EntityId) -> bool;

       /// Returns the number of hydros in the cascade.
       pub fn len(&self) -> usize;

       /// Returns true if the cascade has no hydros.
       pub fn is_empty(&self) -> bool;
   }
   ```

4. Update `topology/mod.rs` with module declaration and re-exports.

5. Update `lib.rs` to re-export `CascadeTopology`.

### Inputs/Props

- `&[Hydro]` -- slice of hydro entities in canonical ID order
- `EntityId` for all ID references

### Outputs/Behavior

- `CascadeTopology::build` constructs the topology from hydro entities
- `downstream(id)` returns `Some(downstream_id)` or `None` for terminal nodes
- `upstream(id)` returns a slice of upstream hydro IDs (empty if none)
- `topological_order()` returns all hydro IDs such that upstream always precedes downstream
- For an empty hydros slice, all methods return empty/None/0 results
- For hydros with no downstream references (all terminal), topological order is canonical ID order

### Error Handling

The `build` method does NOT validate (no cycle detection, no reference validation). It assumes the input is valid. Validation is performed separately in Epic 3. If a `downstream_id` references a non-existent hydro, the downstream entry is simply stored as-is (the validation layer catches this).

## Acceptance Criteria

- [ ] Given hydros A(id=0, downstream=Some(2)), B(id=1, downstream=Some(2)), C(id=2, downstream=None), when calling `CascadeTopology::build`, then `downstream(EntityId(0))` returns `Some(EntityId(2))`, `upstream(EntityId(2))` returns a slice containing EntityId(0) and EntityId(1), and `topological_order()` has C after both A and B
- [ ] Given an empty hydros slice, when calling `CascadeTopology::build`, then `len()` returns 0, `is_empty()` returns true, and `topological_order()` returns an empty slice
- [ ] Given hydros where all have `downstream_id: None`, when calling `CascadeTopology::build`, then all are headwaters and terminal nodes, and `topological_order()` returns them in canonical ID order
- [ ] Given a clean checkout, when running `cargo clippy -p cobre-core`, then zero warnings are produced
- [ ] Given a clean checkout, when running `cargo test -p cobre-core`, then all cascade topology tests pass

## Implementation Guide

### Suggested Approach

1. Build the downstream HashMap by iterating hydros and inserting `(hydro.id, downstream_id)` for each hydro with `downstream_id.is_some()`
2. Build the upstream HashMap by iterating the downstream map: for each `(from, to)`, append `from` to `upstream[to]`
3. Sort the upstream lists by inner i32 for determinism
4. Compute topological order using Kahn's algorithm:
   - Start with all hydros that have no upstream plants (headwaters)
   - Use a BinaryHeap or sorted Vec to process headwaters in i32 order for determinism (since EntityId has no Ord, sort by `.0`)
   - For each processed hydro, "remove" it and add its downstream to the ready set if all its upstream are processed
   - Collect the processing order as the topological order
5. For the topological sort, since EntityId lacks Ord, use a helper that compares `.0` values for deterministic ordering
6. Implement all accessor methods

### Key Files to Modify

- `crates/cobre-core/src/topology/cascade.rs` (populate from stub)
- `crates/cobre-core/src/topology/mod.rs` (add module decl and re-export)
- `crates/cobre-core/src/lib.rs` (add CascadeTopology re-export)

### Patterns to Follow

- Use `HashMap<EntityId, EntityId>` for downstream (not Vec of pairs)
- Use `HashMap<EntityId, Vec<EntityId>>` for upstream (allows multiple upstream plants)
- Return `&[EntityId]` from `upstream()` by returning the Vec's slice, or an empty slice constant for missing entries
- The `build` method takes `&[Hydro]` (borrowed, non-consuming)
- Use `#[must_use]` on `is_headwater`, `is_terminal`, `is_empty`, `len`

### Pitfalls to Avoid

- EntityId does NOT implement `Ord`, so you cannot use it directly in `BTreeMap` or `BinaryHeap`. For deterministic sorting within the topological sort, extract the inner `i32` via `.0`
- The upstream accessor must handle missing keys gracefully (return empty slice, not panic)
- Do NOT implement cycle detection in `build` -- that is Epic 3's validation concern
- Do NOT skip hydros that have downstream_id referencing non-existent hydros -- just store the reference. Validation catches this.
- The `std::collections::HashMap` is the correct choice (not BTreeMap) since we need `EntityId: Hash + Eq` lookup, not ordered iteration
- Import `HashMap` with `use std::collections::HashMap;`

## Testing Requirements

### Unit Tests

In `topology/cascade.rs` (`#[cfg(test)] mod tests`):

- `test_empty_cascade`: Empty hydros slice produces empty topology
- `test_single_hydro_terminal`: One hydro with `downstream_id: None` -- is headwater, is terminal, topo order has one element
- `test_linear_chain`: Three hydros A->B->C -- topo order is [A, B, C], upstream(C) = [B], downstream(A) = B
- `test_fork_merge`: A->C, B->C (two headwaters merge) -- topo order has A and B before C, upstream(C) = [A, B]
- `test_parallel_chains`: A->B and C->D (two independent chains) -- all represented, topo order has A before B and C before D
- `test_all_terminal`: Three hydros, all terminal -- all are headwaters, topo order is canonical ID order
- `test_deterministic_ordering`: Upstream lists for a given hydro are in consistent order across multiple builds with the same input
- `test_is_headwater`: Verify `is_headwater` returns true only for hydros with no upstream
- `test_is_terminal`: Verify `is_terminal` returns true only for hydros with no downstream
- `test_len`: Verify `len()` returns the number of hydros

### Integration Tests

Not applicable for this ticket.

### E2E Tests

Not applicable for this ticket.

## Dependencies

- **Blocked By**: ticket-003 (Hydro entity definition)
- **Blocks**: ticket-008 (SystemBuilder constructs CascadeTopology)

## Effort Estimate

**Points**: 3
**Confidence**: High
