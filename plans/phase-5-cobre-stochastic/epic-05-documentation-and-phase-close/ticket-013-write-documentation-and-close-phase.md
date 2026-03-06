# ticket-013 Write documentation and close phase

> **[OUTLINE]** This ticket requires refinement before execution.
> It will be refined with learnings from earlier epics.

## Objective

Complete Phase 5 by writing the software book chapter for `cobre-stochastic`, updating CONTRIBUTING.md with crate-specific guidelines, running the final infrastructure genericity audit (`grep -riE 'sddp' crates/cobre-stochastic/`), and updating the phase tracker in CLAUDE.md to mark Phase 5 as complete. This follows the same phase-close pattern established in Phases 3 and 4.

## Anticipated Scope

- **Files likely to be modified**:
  - `book/src/crates/cobre-stochastic.md` (new book chapter)
  - `book/src/SUMMARY.md` (add chapter entry)
  - `CONTRIBUTING.md` (add crate-specific section)
  - `CLAUDE.md` (update phase tracker table)
- **Key decisions needed**:
  - Book chapter structure: should it mirror the cobre-solver and cobre-comm chapters, or follow a different organization given the mathematical nature of the content?
  - How much mathematical detail to include in the book chapter vs. deferring to the spec in cobre-docs
- **Open questions**:
  - What is the final test count (unit + integration + doc-tests)?
  - Were there any deviations from the spec that need to be documented as ADRs?
  - Should the book chapter include worked examples with concrete numbers (e.g., the conformance test fixture)?

## Dependencies

- **Blocked By**: ticket-011-implement-pipeline-conformance-tests.md, ticket-012-implement-reproducibility-invariance-tests.md
- **Blocks**: None

## Effort Estimate

**Points**: 2
**Confidence**: Low (will be re-estimated during refinement)
