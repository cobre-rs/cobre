# ticket-005 Implement GlobalPenaltyDefaults and Entity-Level Penalty Resolution

## Context

### Background

The Cobre penalty system uses a three-tier cascade: global defaults -> entity overrides -> stage overrides. Phase 1 only implements the first two tiers (global -> entity). Stage-varying overrides require stages, which are a Phase 2 concern. This ticket defines `GlobalPenaltyDefaults` (the structure holding global penalty values from `penalties.json`) and implements the resolution logic that applies entity-level overrides on top of globals to produce resolved penalty values on each entity.

### Relation to Epic

This is the first ticket in Epic 02 (System Struct and Topology). The penalty resolution logic is needed by `SystemBuilder` (ticket-008) to resolve penalty fields during System construction. It is independent of topology construction and can be implemented first.

### Current State

- `crates/cobre-core/src/penalty.rs` is a stub file (from ticket-001)
- All entity types are defined with their penalty fields: `Bus.excess_cost`, `Bus.deficit_segments`, `Line.exchange_cost`, `Hydro.penalties: HydroPenalties`, `NonControllableSource.curtailment_cost`
- `HydroPenalties` struct is defined in `entities/hydro.rs` (ticket-003) with 11 f64 fields

## Specification

### Requirements

1. Define `GlobalPenaltyDefaults` in `penalty.rs` -- the global default penalty values matching `penalties.json` structure:

   ```rust
   /// Global default penalty values for all entity types.
   ///
   /// Mirrors the structure of `penalties.json`. These values are used as
   /// fallbacks when entity-level overrides are not specified.
   /// See Penalty System spec section 3.
   #[derive(Debug, Clone, PartialEq)]
   pub struct GlobalPenaltyDefaults {
       // Bus defaults
       /// Default piecewise-linear deficit cost segments for buses.
       pub bus_deficit_segments: Vec<DeficitSegment>,
       /// Default excess cost for buses [$/MWh].
       pub bus_excess_cost: f64,

       // Line defaults
       /// Default exchange cost for lines [$/MWh].
       pub line_exchange_cost: f64,

       // Hydro defaults
       /// Default hydro penalty values. Applied to any hydro field not
       /// overridden at the entity level.
       pub hydro: HydroPenalties,

       // Non-controllable source defaults
       /// Default curtailment cost for non-controllable sources [$/MWh].
       pub ncs_curtailment_cost: f64,
   }
   ```

2. Implement penalty resolution functions in `penalty.rs`:

   ```rust
   /// Resolve a bus's deficit segments: use entity override if present, else global default.
   pub fn resolve_bus_deficit_segments(
       entity_deficit_segments: &Option<Vec<DeficitSegment>>,
       global: &GlobalPenaltyDefaults,
   ) -> Vec<DeficitSegment>;

   /// Resolve a bus's excess cost: use entity override if present, else global default.
   pub fn resolve_bus_excess_cost(
       entity_excess_cost: Option<f64>,
       global: &GlobalPenaltyDefaults,
   ) -> f64;

   /// Resolve a line's exchange cost: use entity override if present, else global default.
   pub fn resolve_line_exchange_cost(
       entity_exchange_cost: Option<f64>,
       global: &GlobalPenaltyDefaults,
   ) -> f64;

   /// Resolve a hydro's penalty values: for each of the 11 penalty fields,
   /// use entity override if present, else global default.
   pub fn resolve_hydro_penalties(
       entity_overrides: &Option<HydroPenaltyOverrides>,
       global: &GlobalPenaltyDefaults,
   ) -> HydroPenalties;

   /// Resolve a non-controllable source's curtailment cost.
   pub fn resolve_ncs_curtailment_cost(
       entity_curtailment_cost: Option<f64>,
       global: &GlobalPenaltyDefaults,
   ) -> f64;
   ```

3. Define `HydroPenaltyOverrides` -- optional entity-level overrides (all fields are `Option<f64>` because any field may or may not be overridden at the entity level):

   ```rust
   /// Optional entity-level hydro penalty overrides.
   ///
   /// Each field corresponds to a field in HydroPenalties. A value of None means
   /// "use global default". A value of Some(x) means "override with x".
   /// This is an intermediate type used during System construction; the resolved
   /// HydroPenalties (with no Options) is stored on the Hydro entity.
   #[derive(Debug, Clone, Copy, PartialEq, Default)]
   pub struct HydroPenaltyOverrides {
       pub spillage_cost: Option<f64>,
       pub diversion_cost: Option<f64>,
       pub fpha_turbined_cost: Option<f64>,
       pub storage_violation_below_cost: Option<f64>,
       pub filling_target_violation_cost: Option<f64>,
       pub turbined_violation_below_cost: Option<f64>,
       pub outflow_violation_below_cost: Option<f64>,
       pub outflow_violation_above_cost: Option<f64>,
       pub generation_violation_below_cost: Option<f64>,
       pub evaporation_violation_cost: Option<f64>,
       pub water_withdrawal_violation_cost: Option<f64>,
   }
   ```

4. Update `lib.rs` to re-export `GlobalPenaltyDefaults`, `HydroPenaltyOverrides`, and the resolution functions from `penalty` module.

### Inputs/Props

- `DeficitSegment` from `entities/bus.rs` (ticket-002)
- `HydroPenalties` from `entities/hydro.rs` (ticket-003)

### Outputs/Behavior

- `GlobalPenaltyDefaults` is constructable with all penalty defaults
- `resolve_hydro_penalties` with `None` overrides returns the global defaults exactly
- `resolve_hydro_penalties` with `Some(0.05)` for `spillage_cost` returns global defaults with `spillage_cost` overridden to `0.05`
- All scalar resolution functions follow the same pattern: entity override present -> use it; absent -> use global

### Error Handling

Penalty resolution functions do not validate values (e.g., non-negative costs). Validation is Epic 3. Resolution functions are pure transforms.

## Acceptance Criteria

- [ ] Given a `GlobalPenaltyDefaults` with `hydro.spillage_cost = 0.01` and `HydroPenaltyOverrides { spillage_cost: Some(0.05), ..Default::default() }`, when calling `resolve_hydro_penalties`, then the result has `spillage_cost == 0.05` and all other fields match the global defaults
- [ ] Given a `GlobalPenaltyDefaults` with `bus_excess_cost = 100.0` and an entity override of `None`, when calling `resolve_bus_excess_cost(None, &global)`, then the result is `100.0`
- [ ] Given a `GlobalPenaltyDefaults` and `HydroPenaltyOverrides::default()` (all None), when calling `resolve_hydro_penalties`, then the result equals `global.hydro` exactly
- [ ] Given a clean checkout, when running `cargo clippy -p cobre-core`, then zero warnings are produced
- [ ] Given a clean checkout, when running `cargo test -p cobre-core`, then all penalty resolution tests pass

## Implementation Guide

### Suggested Approach

1. Define `GlobalPenaltyDefaults` struct with doc comments
2. Define `HydroPenaltyOverrides` struct with `Default` derive
3. Implement `resolve_bus_deficit_segments` -- clone entity override if present, else clone global default
4. Implement `resolve_bus_excess_cost` -- unwrap_or pattern
5. Implement `resolve_line_exchange_cost` -- same pattern
6. Implement `resolve_hydro_penalties` -- field-by-field resolution, 11 fields
7. Implement `resolve_ncs_curtailment_cost` -- same pattern
8. Write unit tests for each resolution function
9. Update `lib.rs` with re-exports

### Key Files to Modify

- `crates/cobre-core/src/penalty.rs` (populate from stub)
- `crates/cobre-core/src/lib.rs` (add penalty re-exports)

### Patterns to Follow

- Use `entity_override.unwrap_or(global_default)` pattern for scalar fields
- Use `entity_override.clone().unwrap_or_else(|| global_default.clone())` for Vec fields (DeficitSegment)
- For `resolve_hydro_penalties`, resolve each of the 11 fields individually against the global hydro defaults
- All resolution functions are pure (no mutation, no side effects)
- `HydroPenaltyOverrides` derives `Default` so all fields default to `None`

### Pitfalls to Avoid

- Do NOT implement stage-varying resolution -- that is Phase 2
- Do NOT validate penalty values (non-negative, ordering) -- that is Epic 3
- The `resolve_bus_deficit_segments` function takes `&Option<Vec<DeficitSegment>>` (entity may not have overridden deficit segments) and returns `Vec<DeficitSegment>` (owned, resolved)
- `HydroPenaltyOverrides` is an intermediate type for construction only -- the resolved `HydroPenalties` (no Options) is what gets stored on the Hydro entity
- Do NOT use `.unwrap()` -- use `.unwrap_or()` or `.unwrap_or_else()` which are not banned by the lint

## Testing Requirements

### Unit Tests

In `penalty.rs` (`#[cfg(test)] mod tests`):

- `test_resolve_bus_excess_cost_global`: No entity override -> returns global default
- `test_resolve_bus_excess_cost_override`: Entity override present -> returns override value
- `test_resolve_bus_deficit_segments_global`: No entity override -> returns global default segments
- `test_resolve_bus_deficit_segments_override`: Entity override present -> returns override segments
- `test_resolve_line_exchange_cost_global`: No entity override -> returns global default
- `test_resolve_line_exchange_cost_override`: Entity override present -> returns override value
- `test_resolve_hydro_penalties_all_global`: All overrides None -> result equals global hydro defaults
- `test_resolve_hydro_penalties_partial_override`: Some overrides present -> those fields overridden, rest from global
- `test_resolve_hydro_penalties_all_override`: All overrides present -> all fields from overrides
- `test_resolve_ncs_curtailment_cost_global`: No override -> global default
- `test_resolve_ncs_curtailment_cost_override`: Override present -> override value

### Integration Tests

Not applicable for this ticket.

### E2E Tests

Not applicable for this ticket.

## Dependencies

- **Blocked By**: ticket-002 (DeficitSegment), ticket-003 (HydroPenalties)
- **Blocks**: ticket-008 (SystemBuilder uses penalty resolution)

## Effort Estimate

**Points**: 2
**Confidence**: High
