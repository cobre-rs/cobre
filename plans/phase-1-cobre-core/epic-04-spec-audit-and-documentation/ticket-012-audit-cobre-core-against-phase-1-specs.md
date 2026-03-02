# ticket-012 Audit cobre-core Implementation Against Phase 1 Specs

## Context

### Background

Phase 1 of the Cobre project implements the `cobre-core` crate -- the foundation data model for the SDDP solver ecosystem. Across epics 01-03 (11 tickets), the following was built: 7 entity types (Bus, Line, Hydro, Thermal, PumpingStation, EnergyContract, NonControllableSource), supporting enums and structs, `EntityId` newtype, `System` container with `SystemBuilder`, `CascadeTopology`, `NetworkTopology`, penalty resolution (global -> entity), cross-reference validation, cascade cycle detection, filling config validation, and declaration-order invariance tests. All 11 tickets scored EXCELLENT or ACCEPTABLE quality.

Before marking Phase 1 complete and documenting it in the software book, a systematic audit must verify that the implementation conforms to every relevant section in the Phase 1 spec reading list. The audit ensures no spec requirements were missed, no fields were omitted, and no validation rules were left unimplemented.

### Relation to Epic

This is the first ticket of epic-04 (Spec Audit and Documentation). Its output -- the audit report -- informs the documentation tickets (013-014) by confirming what was implemented and flagging any deviations. If the audit finds gaps, those would be tracked as separate tickets outside this epic.

### Current State

The implementation exists in `crates/cobre-core/src/` with the following module structure:

- `entity_id.rs` -- `EntityId` newtype
- `entities/bus.rs` -- `Bus`, `DeficitSegment`
- `entities/line.rs` -- `Line`
- `entities/hydro.rs` -- `Hydro`, `HydroGenerationModel`, `TailraceModel`, `TailracePoint`, `HydraulicLossesModel`, `EfficiencyModel`, `DiversionChannel`, `FillingConfig`, `HydroPenalties`
- `entities/thermal.rs` -- `Thermal`, `ThermalCostSegment`, `GnlConfig`
- `entities/pumping_station.rs` -- `PumpingStation`
- `entities/energy_contract.rs` -- `EnergyContract`, `ContractType`
- `entities/non_controllable.rs` -- `NonControllableSource`
- `topology/cascade.rs` -- `CascadeTopology`
- `topology/network.rs` -- `NetworkTopology`, `BusLineConnection`, `BusGenerators`, `BusLoads`
- `system.rs` -- `System`, `SystemBuilder`
- `error.rs` -- `ValidationError`
- `penalty.rs` -- `GlobalPenaltyDefaults`, `HydroPenaltyOverrides`, resolution functions

The 9 spec files in the Phase 1 reading list are in `~/git/cobre-docs/src/specs/`:

1. `overview/notation-conventions.md`
2. `overview/design-principles.md`
3. `math/hydro-production-models.md`
4. `data-model/input-system-entities.md`
5. `data-model/input-hydro-extensions.md`
6. `data-model/penalty-system.md`
7. `data-model/internal-structures.md`
8. `math/system-elements.md`
9. `math/equipment-formulations.md`

## Specification

### Requirements

Produce a structured audit report (`plans/phase-1-cobre-core/epic-04-spec-audit-and-documentation/audit-report.md`) that:

1. **For each of the 9 spec files**, enumerate every Phase 1-relevant requirement and assess conformance against the implementation. Phase 1 scope is limited to the `cobre-core` data model -- requirements about LP construction, scenario generation, I/O loading, or runtime behavior are noted as "out of Phase 1 scope" and not audited.

2. **For each entity type** (Bus, Line, Hydro, Thermal, PumpingStation, EnergyContract, NonControllableSource), compare the spec's field table against the Rust struct field-by-field and report:
   - Fields present and matching: field name, type, optionality all match
   - Fields with name differences: field exists but Rust name differs from spec (document the mapping)
   - Fields missing from implementation: field in spec but not in Rust struct
   - Fields extra in implementation: field in Rust struct but not in spec (document rationale)

3. **For each supporting type** (enums and structs), verify the variants/fields match the spec.

4. **For each validation rule** specified in the internal-structures spec and penalty-system spec, verify whether it is implemented in `SystemBuilder::build()` and the validation functions.

5. **For the public API** (`System` methods), verify that the methods specified in internal-structures section 1.4 are present.

6. **For design principles**, verify that declaration-order invariance, canonical ordering, `Send + Sync`, and immutability-after-construction are implemented.

7. **Summary section** with:
   - Total number of spec requirements audited
   - Number conforming
   - Number with minor deviations (naming, documented as intentional)
   - Number with gaps (if any)
   - Overall conformance assessment

### Inputs/Props

- All 9 spec files listed above (read from `~/git/cobre-docs/src/specs/`)
- All source files in `crates/cobre-core/src/` (read from `/home/rogerio/git/cobre/`)
- The 3 epic learnings files in `plans/phase-1-cobre-core/learnings/`

### Outputs/Behavior

A single markdown file `plans/phase-1-cobre-core/epic-04-spec-audit-and-documentation/audit-report.md` containing the structured audit report.

### Error Handling

- If a spec section is ambiguous about whether a requirement applies to Phase 1, document the ambiguity and the decision made (in-scope or out-of-scope) with rationale.
- If a gap is found, document it clearly with the spec section reference, what was expected, and what exists (or does not exist). Do NOT attempt to fix it in this ticket.

## Acceptance Criteria

- [ ] Given the 9 Phase 1 spec files exist at the paths listed above, when the agent reads all 9 spec files and all `crates/cobre-core/src/` source files, then the file `plans/phase-1-cobre-core/epic-04-spec-audit-and-documentation/audit-report.md` is created.
- [ ] Given the audit report is created, when the report is inspected, then it contains a section for each of the 9 spec files with per-requirement conformance assessment.
- [ ] Given the audit report is created, when the entity field comparison tables are inspected, then every field in every entity type from `internal-structures.md` section 1.9 is either listed as "conforming", "naming difference" (with mapping), "missing", or "extra" (with rationale).
- [ ] Given the audit report is created, when the summary section is inspected, then it contains the total count of requirements audited and the conformance percentages.
- [ ] Given the audit report is created, when the validation rules section is inspected, then each validation rule from `internal-structures.md` and `penalty-system.md` that is in Phase 1 scope has a conformance status.

## Implementation Guide

### Suggested Approach

1. Read all 9 spec files from `~/git/cobre-docs/src/specs/` in the order listed above.
2. For each spec, identify requirements that fall within Phase 1 scope (data model types, entity fields, validation rules, design principles). Skip requirements about LP construction, runtime behavior, I/O, or features explicitly deferred beyond Phase 1 scope.
3. For entity types, systematically compare the spec field tables (from `internal-structures.md` section 1.9 and `input-system-entities.md`) against the Rust struct definitions.
4. For validation rules, check whether `SystemBuilder::build()` and its helper functions implement the rule.
5. For design principles, check the `System` struct's derives, `Send + Sync` assertion, canonical ordering in the builder, and integration tests.
6. Write the audit report in a structured format with clear section headers matching the spec files.

### Key Files to Modify

- **Create**: `plans/phase-1-cobre-core/epic-04-spec-audit-and-documentation/audit-report.md`

### Key Files to Read

- `~/git/cobre-docs/src/specs/overview/notation-conventions.md`
- `~/git/cobre-docs/src/specs/overview/design-principles.md`
- `~/git/cobre-docs/src/specs/math/hydro-production-models.md`
- `~/git/cobre-docs/src/specs/data-model/input-system-entities.md`
- `~/git/cobre-docs/src/specs/data-model/input-hydro-extensions.md`
- `~/git/cobre-docs/src/specs/data-model/penalty-system.md`
- `~/git/cobre-docs/src/specs/data-model/internal-structures.md`
- `~/git/cobre-docs/src/specs/math/system-elements.md`
- `~/git/cobre-docs/src/specs/math/equipment-formulations.md`
- `crates/cobre-core/src/lib.rs`
- `crates/cobre-core/src/entity_id.rs`
- `crates/cobre-core/src/entities/bus.rs`
- `crates/cobre-core/src/entities/line.rs`
- `crates/cobre-core/src/entities/hydro.rs`
- `crates/cobre-core/src/entities/thermal.rs`
- `crates/cobre-core/src/entities/pumping_station.rs`
- `crates/cobre-core/src/entities/energy_contract.rs`
- `crates/cobre-core/src/entities/non_controllable.rs`
- `crates/cobre-core/src/topology/cascade.rs`
- `crates/cobre-core/src/topology/network.rs`
- `crates/cobre-core/src/system.rs`
- `crates/cobre-core/src/error.rs`
- `crates/cobre-core/src/penalty.rs`
- `plans/phase-1-cobre-core/learnings/epic-01-summary.md`
- `plans/phase-1-cobre-core/learnings/epic-02-summary.md`
- `plans/phase-1-cobre-core/learnings/epic-03-summary.md`

### Patterns to Follow

- Use tables for field-by-field comparisons (Spec Field | Rust Field | Status | Notes)
- Use checkmark/cross symbols for quick visual scanning
- Group findings by spec file, then by entity type within each spec file
- Keep the "out of Phase 1 scope" notes brief -- just list the section number and a one-line reason

### Pitfalls to Avoid

- Do NOT modify any source code in this ticket -- the audit is read-only
- Do NOT modify any spec files -- cobre-docs is the source of truth
- Do NOT conflate Phase 1 scope with the full spec scope -- many spec requirements are for Phase 2+ (e.g., `ResolvedPenalties`, `ResolvedBounds`, `Stage`, `PolicyGraph`, stage-varying overrides)
- Do NOT treat JSON schema field names as the authoritative Rust field names -- the `internal-structures.md` section 1.9 Rust structs are the authoritative field names for `cobre-core`

## Testing Requirements

### Unit Tests

Not applicable -- this ticket produces a documentation artifact, not code.

### Integration Tests

Not applicable.

### E2E Tests (if applicable)

Not applicable.

## Dependencies

- **Blocked By**: ticket-011 (integration and order-invariance tests -- last ticket of epic-03, completed)
- **Blocks**: ticket-013 (software book page uses audit findings to describe what was implemented)

## Effort Estimate

**Points**: 3
**Confidence**: High
