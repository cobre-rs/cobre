# ticket-010 Implement Cascade Cycle Detection and Filling Config Validation

## Context

### Background

The hydro cascade graph must be a directed acyclic graph (DAG). A cycle (e.g., A->B->C->A) would cause infinite water balance loops in the SDDP training loop. Currently, `CascadeTopology::build()` uses Kahn's algorithm for topological sorting but does not check whether all nodes were processed -- nodes in a cycle never reach in-degree 0 and are silently omitted from the topological order. Additionally, hydro filling configurations have invariants that must be validated: a filling config requires the hydro to have an `entry_stage_id`, and `filling_inflow_m3s` must be positive.

### Relation to Epic

This is the second validation ticket in Epic 03. It adds cascade DAG validation and filling config validation to `SystemBuilder::build()`, building on the cross-reference validation from ticket-009. Ticket-011 exercises these validations in integration tests.

### Current State

- `CascadeTopology::build()` in `crates/cobre-core/src/topology/cascade.rs` uses Kahn's algorithm. If the graph has a cycle, `topological_order.len() < hydros.len()` because nodes in the cycle are never processed. The function does not detect or report this.
- `ValidationError::CascadeCycle` exists in `crates/cobre-core/src/error.rs` with field `cycle_ids: Vec<EntityId>`.
- `ValidationError::InvalidFillingConfig` exists with fields `hydro_id: EntityId` and `reason: String`.
- `FillingConfig` in `crates/cobre-core/src/entities/hydro.rs` has fields `start_stage_id: i32` and `filling_inflow_m3s: f64`.
- `Hydro.entry_stage_id` is `Option<i32>`.

## Specification

### Requirements

Add two validation passes to `SystemBuilder::build()`, after cross-reference validation (ticket-009) and before topology construction succeeds:

**1. Cascade cycle detection:** After `CascadeTopology::build()` is called, check whether `topological_order.len() == hydros.len()`. If not, identify the hydros that were not included in the topological order (these are the cycle participants) and emit a `ValidationError::CascadeCycle` error. The `cycle_ids` field should contain the IDs of all hydros not in the topological order, sorted by inner `i32` for determinism.

**2. Filling config validation:** For each hydro that has `filling: Some(config)`:

- If `filling_inflow_m3s <= 0.0`, emit `ValidationError::InvalidFillingConfig` with reason `"filling_inflow_m3s must be positive"`.
- If the hydro has `entry_stage_id: None`, emit `ValidationError::InvalidFillingConfig` with reason `"filling requires entry_stage_id to be set"`.

### Inputs/Props

Cycle detection uses the already-built `CascadeTopology` and the hydro slice. Filling validation uses only the hydro slice.

### Outputs/Behavior

For cycle detection: a single `ValidationError::CascadeCycle` containing all cycle participant IDs. For filling validation: one `ValidationError::InvalidFillingConfig` per invalid filling config, with a human-readable reason string. All errors are collected and returned together with any cross-reference errors.

### Error Handling

Uses existing `ValidationError::CascadeCycle` and `ValidationError::InvalidFillingConfig` variants. No new error variants needed.

## Acceptance Criteria

- [ ] Given hydros A(0)->B(1)->C(2)->A(0) forming a 3-node cycle, when `SystemBuilder::build()` is called, then the result is `Err` containing `ValidationError::CascadeCycle { cycle_ids }` where `cycle_ids` contains `EntityId(0)`, `EntityId(1)`, and `EntityId(2)` sorted ascending.
- [ ] Given a hydro with `filling: Some(FillingConfig { start_stage_id: 10, filling_inflow_m3s: -5.0 })`, when `SystemBuilder::build()` is called, then the result is `Err` containing `ValidationError::InvalidFillingConfig` with reason containing `"filling_inflow_m3s must be positive"`.
- [ ] Given a hydro with `filling: Some(FillingConfig { .. })` and `entry_stage_id: None`, when `SystemBuilder::build()` is called, then the result is `Err` containing `ValidationError::InvalidFillingConfig` with reason containing `"entry_stage_id"`.
- [ ] Given a valid acyclic cascade with no filling configs, when `SystemBuilder::build()` is called, then the result is `Ok(System)` and `system.cascade().topological_order().len()` equals the number of hydros.
- [ ] Given a system with both a cascade cycle AND an invalid filling config, when `SystemBuilder::build()` is called, then the `Err` vector contains both a `CascadeCycle` and an `InvalidFillingConfig` error.

## Implementation Guide

### Suggested Approach

1. In `SystemBuilder::build()`, after calling `CascadeTopology::build(&self.hydros)`, add cascade cycle detection:
   ```rust
   if cascade.topological_order().len() < self.hydros.len() {
       let in_topo: HashSet<EntityId> = cascade.topological_order().iter().copied().collect();
       let mut cycle_ids: Vec<EntityId> = self.hydros.iter()
           .map(|h| h.id)
           .filter(|id| !in_topo.contains(id))
           .collect();
       cycle_ids.sort_by_key(|id| id.0);
       errors.push(ValidationError::CascadeCycle { cycle_ids });
   }
   ```
2. Add a private function `validate_filling_configs(hydros: &[Hydro], errors: &mut Vec<ValidationError>)` that iterates over hydros and checks each filling config.
3. Call both validation passes after cross-reference validation. The order should be: duplicate check -> cross-reference validation -> topology build + cycle check -> filling validation -> final error check.
4. Add `use std::collections::HashSet;` to imports if not already present.

### Key Files to Modify

- `crates/cobre-core/src/system.rs` -- add cycle detection after `CascadeTopology::build()`, add `validate_filling_configs` function

### Patterns to Follow

- Follow the `check_duplicates` / `validate_cross_references` pattern: private functions that push to `&mut Vec<ValidationError>`.
- Sort `cycle_ids` by `id.0` for determinism, following the `sort_by_key(|id| id.0)` pattern from cascade topology.
- Use `std::collections::HashSet` for O(1) membership testing in cycle detection.

### Pitfalls to Avoid

- Do NOT modify `CascadeTopology::build()` to return `Result` -- keep it infallible. The cycle detection is done in `SystemBuilder::build()` by comparing lengths.
- Do NOT attempt to identify the exact cycle path (which specific edges form the cycle). The `cycle_ids` field contains ALL hydros not in the topological order, which is sufficient for error reporting.
- Do NOT validate `filling.start_stage_id` against actual stage data -- stage validation requires Phase 2 data.
- Do NOT validate penalty values in this ticket -- penalty validation is a separate concern that can be added later.

## Testing Requirements

### Unit Tests

Add tests in the existing `#[cfg(test)] mod tests` block in `system.rs`:

1. `test_cascade_cycle_detected` -- 3-node cycle, check `CascadeCycle` error with correct IDs
2. `test_cascade_self_loop_detected` -- single hydro pointing to itself
3. `test_filling_without_entry_stage` -- filling config present but `entry_stage_id` is `None`
4. `test_filling_negative_inflow` -- filling config with `filling_inflow_m3s <= 0.0`
5. `test_valid_filling_config_passes` -- valid filling config does not produce errors

### Integration Tests

None in this ticket. Integration tests are in ticket-011.

### E2E Tests

Not applicable.

## Dependencies

- **Blocked By**: ticket-008-system-builder-and-system-struct.md, ticket-009-cross-reference-validation.md
- **Blocks**: ticket-011-integration-and-order-invariance-tests.md

## Effort Estimate

**Points**: 2
**Confidence**: High
