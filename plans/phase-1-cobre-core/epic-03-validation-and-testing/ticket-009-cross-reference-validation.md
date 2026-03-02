# ticket-009 Implement Cross-Reference Validation in SystemBuilder

> **[OUTLINE]** This ticket requires refinement before execution.
> It will be refined with learnings from earlier epics.

## Objective

Add cross-reference validation to `SystemBuilder::build()` so that all entity fields referencing other entities (bus_id, downstream_id, source_hydro_id, destination_hydro_id, diversion downstream_id) are verified to point to existing entities. Invalid references produce `ValidationError::InvalidReference` errors.

## Anticipated Scope

- **Files likely to be modified**: `crates/cobre-core/src/system.rs` (SystemBuilder::build method)
- **Key decisions needed**: Whether validation runs before or after topology construction; whether to short-circuit on first error or collect all errors; how to efficiently check references (HashSet of existing IDs vs. lookup in the index HashMap)
- **Open questions**: Should the validation also check that `entry_stage_id < exit_stage_id` when both are present (lifecycle sanity)? This is an entity-level invariant, not a cross-reference, but it is a natural addition.

## Dependencies

- **Blocked By**: ticket-008 (SystemBuilder and System struct)
- **Blocks**: ticket-011 (integration tests exercise validation)

## Effort Estimate

**Points**: 2
**Confidence**: Low (will be re-estimated during refinement)
