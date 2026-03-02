# ticket-010 Implement Cascade Cycle Detection and Filling Config Validation

> **[OUTLINE]** This ticket requires refinement before execution.
> It will be refined with learnings from earlier epics.

## Objective

Add cascade DAG validation (cycle detection) and filling configuration validation to `SystemBuilder::build()`. A cycle in the hydro cascade (A->B->C->A) is a fatal error because it would cause infinite water balance loops. Filling config validation ensures `start_stage_id < entry_stage_id` and that filling is only defined for hydros with `entry_stage_id`.

## Anticipated Scope

- **Files likely to be modified**: `crates/cobre-core/src/system.rs` (SystemBuilder::build), possibly `crates/cobre-core/src/topology/cascade.rs` (if cycle detection is a method on CascadeTopology)
- **Key decisions needed**: Whether cycle detection is a method on CascadeTopology (`validate_acyclic()`) or a standalone function called by SystemBuilder; algorithm choice for cycle detection (Kahn's algorithm already used in topological sort -- if all nodes are not processed, there is a cycle); how to report which hydros form the cycle
- **Open questions**: Should the topological sort in CascadeTopology::build already detect cycles (since Kahn's algorithm naturally detects them by checking if all nodes were processed)? If so, should CascadeTopology::build return Result instead of assuming valid input?

## Dependencies

- **Blocked By**: ticket-008 (SystemBuilder), ticket-006 (CascadeTopology)
- **Blocks**: ticket-011 (integration tests exercise validation)

## Effort Estimate

**Points**: 2
**Confidence**: Low (will be re-estimated during refinement)
