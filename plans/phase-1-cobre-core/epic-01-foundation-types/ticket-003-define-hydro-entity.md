# ticket-003 Define Hydro Entity and All Hydro Supporting Types

## Context

### Background

The Hydro entity is the most complex type in the Cobre data model. It requires multiple supporting enums and structs for its generation model, tailrace, hydraulic losses, efficiency, diversion, filling, and penalty overrides. This ticket defines the complete Hydro type family according to the internal-structures spec section 1.9.4.

### Relation to Epic

This is the third ticket in Epic 01 (Foundation Types). It populates the `hydro.rs` entity module with the `Hydro` struct and all its supporting types. This is the largest single entity definition ticket due to the hydro plant's complexity.

### Current State

- `crates/cobre-core/src/entities/hydro.rs` is a stub file (module-level doc comment only, from ticket-001)
- `EntityId` is defined and usable from crate root (ticket-001)
- `Bus` and `Line` are defined (ticket-002)

## Specification

### Requirements

1. Define the following supporting enums in `entities/hydro.rs`:

   **`HydroGenerationModel`** -- tagged union for production function selection:

   ```rust
   #[derive(Debug, Clone, PartialEq)]
   pub enum HydroGenerationModel {
       ConstantProductivity { productivity_mw_per_m3s: f64 },
       LinearizedHead { productivity_mw_per_m3s: f64 },
       Fpha,
   }
   ```

   - Doc comments on each variant explaining its role (constant productivity for training+simulation, linearized head for simulation-only, FPHA for training+simulation)

   **`TailraceModel`** -- downstream water level computation:

   ```rust
   #[derive(Debug, Clone, PartialEq)]
   pub enum TailraceModel {
       Polynomial { coefficients: Vec<f64> },
       Piecewise { points: Vec<TailracePoint> },
   }
   ```

   **`HydraulicLossesModel`**:

   ```rust
   #[derive(Debug, Clone, Copy, PartialEq)]
   pub enum HydraulicLossesModel {
       Factor { value: f64 },
       Constant { value_m: f64 },
   }
   ```

   **`EfficiencyModel`**:

   ```rust
   #[derive(Debug, Clone, Copy, PartialEq)]
   pub enum EfficiencyModel {
       Constant { value: f64 },
   }
   ```

2. Define the following supporting structs in `entities/hydro.rs`:

   **`TailracePoint`**:

   ```rust
   #[derive(Debug, Clone, Copy, PartialEq)]
   pub struct TailracePoint {
       pub outflow_m3s: f64,
       pub height_m: f64,
   }
   ```

   **`DiversionChannel`**:

   ```rust
   #[derive(Debug, Clone, PartialEq)]
   pub struct DiversionChannel {
       pub downstream_id: EntityId,
       pub max_flow_m3s: f64,
   }
   ```

   **`FillingConfig`**:

   ```rust
   #[derive(Debug, Clone, Copy, PartialEq)]
   pub struct FillingConfig {
       pub start_stage_id: i32,
       pub filling_inflow_m3s: f64,
   }
   ```

   **`HydroPenalties`** -- entity-level penalty overrides (resolved from global -> entity cascade):

   ```rust
   #[derive(Debug, Clone, Copy, PartialEq)]
   pub struct HydroPenalties {
       pub spillage_cost: f64,
       pub diversion_cost: f64,
       pub fpha_turbined_cost: f64,
       pub storage_violation_below_cost: f64,
       pub filling_target_violation_cost: f64,
       pub turbined_violation_below_cost: f64,
       pub outflow_violation_below_cost: f64,
       pub outflow_violation_above_cost: f64,
       pub generation_violation_below_cost: f64,
       pub evaporation_violation_cost: f64,
       pub water_withdrawal_violation_cost: f64,
   }
   ```

3. Define the `Hydro` struct with all fields from spec section 1.9.4:

   ```rust
   #[derive(Debug, Clone, PartialEq)]
   pub struct Hydro {
       // Core identity and topology
       pub id: EntityId,
       pub name: String,
       pub bus_id: EntityId,
       pub downstream_id: Option<EntityId>,
       pub entry_stage_id: Option<i32>,
       pub exit_stage_id: Option<i32>,
       // Reservoir
       pub min_storage_hm3: f64,
       pub max_storage_hm3: f64,
       // Outflow bounds
       pub min_outflow_m3s: f64,
       pub max_outflow_m3s: Option<f64>,
       // Generation model
       pub generation_model: HydroGenerationModel,
       pub min_turbined_m3s: f64,
       pub max_turbined_m3s: f64,
       pub min_generation_mw: f64,
       pub max_generation_mw: f64,
       // Optional data
       pub tailrace: Option<TailraceModel>,
       pub hydraulic_losses: Option<HydraulicLossesModel>,
       pub efficiency: Option<EfficiencyModel>,
       pub evaporation_coefficients_mm: Option<[f64; 12]>,
       // Diversion and filling
       pub diversion: Option<DiversionChannel>,
       pub filling: Option<FillingConfig>,
       // Resolved penalties
       pub penalties: HydroPenalties,
   }
   ```

4. Update `entities/mod.rs` to declare and re-export all new types from `hydro` module.

5. Update `lib.rs` re-exports to include all hydro types.

6. Add unit tests for each type.

### Inputs/Props

- `EntityId` from `entity_id.rs` (ticket-001)

### Outputs/Behavior

- All hydro-related types are accessible from crate root (e.g., `cobre_core::Hydro`, `cobre_core::HydroGenerationModel`)
- `Hydro` struct has exactly the fields from spec section 1.9.4
- All derive lists match the spec exactly

### Error Handling

No error handling in this ticket -- pure data structs.

## Acceptance Criteria

- [ ] Given the file `crates/cobre-core/src/entities/hydro.rs`, when counting the fields on the `Hydro` struct, then there are exactly 21 fields matching the spec: id, name, bus_id, downstream_id, entry_stage_id, exit_stage_id, min_storage_hm3, max_storage_hm3, min_outflow_m3s, max_outflow_m3s, generation_model, min_turbined_m3s, max_turbined_m3s, min_generation_mw, max_generation_mw, tailrace, hydraulic_losses, efficiency, evaporation_coefficients_mm, diversion, filling, penalties
- [ ] Given test code creating a `Hydro` with `HydroGenerationModel::ConstantProductivity { productivity_mw_per_m3s: 0.8765 }`, when accessing `hydro.generation_model`, then it matches the variant and value
- [ ] Given test code creating a `HydroPenalties` struct with all 11 penalty fields, when accessing each field, then all values are correct
- [ ] Given a clean checkout, when running `cargo clippy -p cobre-core`, then zero warnings are produced
- [ ] Given a clean checkout, when running `cargo test -p cobre-core`, then all tests pass including the new hydro tests

## Implementation Guide

### Suggested Approach

1. Define `TailracePoint` first (used by `TailraceModel`)
2. Define all supporting enums: `HydroGenerationModel`, `TailraceModel`, `HydraulicLossesModel`, `EfficiencyModel`
3. Define supporting structs: `DiversionChannel`, `FillingConfig`, `HydroPenalties`
4. Define the `Hydro` struct with all 21 fields
5. Update `entities/mod.rs` and `lib.rs` with re-exports
6. Write unit tests for each type
7. Run `cargo build`, `cargo clippy`, `cargo doc`, `cargo test`

### Key Files to Modify

- `crates/cobre-core/src/entities/hydro.rs` (populate from stub)
- `crates/cobre-core/src/entities/mod.rs` (add hydro module and re-exports)
- `crates/cobre-core/src/lib.rs` (add hydro type re-exports)

### Patterns to Follow

- Match spec section 1.9.1 (supporting enums) and 1.9.4 (Hydro) exactly
- Doc comments must include units in brackets where applicable: `/// Minimum storage (dead volume) [hm3]`
- The `evaporation_coefficients_mm` field is `Option<[f64; 12]>` (fixed-size array, not `Vec`)
- `HydroPenalties` is `Copy` (all fields are `f64`)
- `DiversionChannel` is NOT `Copy` because it contains `EntityId` which is `Copy`, but the spec shows `#[derive(Debug, Clone, PartialEq)]` -- actually `DiversionChannel` with two fields (EntityId and f64) could be Copy. Follow the spec derive list exactly.
- `TailraceModel` is NOT `Copy` because `Polynomial` contains `Vec<f64>` and `Piecewise` contains `Vec<TailracePoint>`

### Pitfalls to Avoid

- The `Hydro` struct has 21 fields (NOT 22) -- count carefully against the spec
- `downstream_id` is `Option<EntityId>` (not `Option<i32>`) because it references another hydro entity
- `max_outflow_m3s` is `Option<f64>` (None = no upper bound constraint)
- `evaporation_coefficients_mm` is `Option<[f64; 12]>` with the fixed array inside the Option
- Do NOT add `water_withdrawal_m3s` field to Hydro -- that comes from per-stage bounds in Phase 2
- The `penalties` field is NOT Optional -- it is always resolved (if no entity override exists, the global default is used)

## Testing Requirements

### Unit Tests

In `entities/hydro.rs` (`#[cfg(test)] mod tests`):

- `test_hydro_constant_productivity`: Create a Hydro with `ConstantProductivity` model, verify model variant and productivity value
- `test_hydro_fpha`: Create a Hydro with `Fpha` model, verify variant is `Fpha`
- `test_hydro_optional_fields_none`: Create a Hydro with all optional fields as `None`, verify construction succeeds
- `test_hydro_optional_fields_some`: Create a Hydro with all optional fields populated, verify all accessible
- `test_tailrace_polynomial`: Create `TailraceModel::Polynomial` with 3 coefficients, verify coefficients vector
- `test_tailrace_piecewise`: Create `TailraceModel::Piecewise` with 3 points, verify points
- `test_hydraulic_losses_factor`: Create `HydraulicLossesModel::Factor { value: 0.03 }`, verify
- `test_filling_config`: Create `FillingConfig` with `start_stage_id: 48, filling_inflow_m3s: 100.0`, verify
- `test_hydro_penalties_all_fields`: Create `HydroPenalties` with all 11 fields, verify each is accessible
- `test_diversion_channel`: Create `DiversionChannel`, verify `downstream_id` and `max_flow_m3s`

### Integration Tests

Not applicable for this ticket.

### E2E Tests

Not applicable for this ticket.

## Dependencies

- **Blocked By**: ticket-001 (EntityId and module structure)
- **Blocks**: ticket-005 (SystemBuilder needs Hydro), ticket-006 (CascadeTopology needs Hydro)

## Effort Estimate

**Points**: 2
**Confidence**: High
