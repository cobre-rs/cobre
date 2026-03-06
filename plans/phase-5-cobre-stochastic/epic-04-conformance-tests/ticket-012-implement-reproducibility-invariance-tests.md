# ticket-012 Implement reproducibility and invariance tests

> **[OUTLINE]** This ticket requires refinement before execution.
> It will be refined with learnings from earlier epics.

## Objective

Create integration tests that validate the cross-concern properties of the stochastic pipeline: deterministic reproducibility (same seed produces bit-identical results across repeated runs), declaration-order invariance (reordered hydro IDs produce identical results), seed sensitivity (different seeds produce different results), and infrastructure genericity compliance (zero SDDP references in crate source). These tests complement the functional conformance tests in ticket-011 by validating invariants rather than specific output values.

## Anticipated Scope

- **Files likely to be modified**:
  - `crates/cobre-stochastic/tests/reproducibility.rs` (new integration test file)
  - `crates/cobre-stochastic/tests/invariance.rs` (new integration test file, or combined with reproducibility)
- **Key decisions needed**:
  - Whether to use a shell-out `grep` command for the SDDP reference check or implement it as a Rust source-scanning test (depends on CI integration patterns from prior phases)
  - Whether declaration-order invariance tests should compare full OpeningTree data or just selected slices (depends on tree size and comparison cost)
- **Open questions**:
  - Should the reproducibility test run the pipeline N times and compare all N results, or compare pairs? What is N?
  - For declaration-order invariance, what constitutes "reordered"? Should the test use reversed IDs, shuffled IDs, or both?
  - Should the infrastructure genericity gate test be in this file or in a separate `tests/genericity.rs`?

## Dependencies

- **Blocked By**: ticket-010-organize-public-api.md
- **Blocks**: ticket-013-write-documentation.md

## Effort Estimate

**Points**: 2
**Confidence**: Low (will be re-estimated during refinement)
