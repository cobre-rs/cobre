# ticket-011 Implement Integration Tests and Declaration-Order Invariance Tests

> **[OUTLINE]** This ticket requires refinement before execution.
> It will be refined with learnings from earlier epics.

## Objective

Create comprehensive integration tests that exercise the full System construction pipeline with realistic multi-entity configurations. Include declaration-order invariance tests that verify bit-for-bit identical System construction regardless of entity input ordering. These tests serve as the validation contract for Phase 1 completion.

## Anticipated Scope

- **Files likely to be modified**: `crates/cobre-core/tests/` (new integration test files), possibly `crates/cobre-core/src/` (test helper module)
- **Key decisions needed**: How to compare two System instances for equality (derive PartialEq on System, or compare field-by-field); whether to create a shared test fixture module with helper functions for building entities; how many entity configurations to test (minimal 2-bus system, medium 4-bus system, large-ish system with all entity types)
- **Open questions**: Should we create a test helper crate or just an internal test module? Should the declaration-order invariance test use all 7 entity types or focus on the most complex (hydro cascade)?

## Dependencies

- **Blocked By**: ticket-008 (SystemBuilder), ticket-009 (cross-reference validation), ticket-010 (cascade validation)
- **Blocks**: None -- this is the final ticket in Phase 1

## Effort Estimate

**Points**: 3
**Confidence**: Low (will be re-estimated during refinement)
