# ticket-011 Implement end-to-end pipeline conformance tests

> **[OUTLINE]** This ticket requires refinement before execution.
> It will be refined with learnings from earlier epics.

## Objective

Create integration tests in `crates/cobre-stochastic/tests/conformance.rs` that exercise the full stochastic pipeline from raw `InflowModel` input through to InSample forward sampling output. The tests use the hand-computable fixture from `sampling-scheme-testing.md` (3 stages, 2 hydros, 5 openings, known correlation matrix) and verify that the pipeline produces correct results at each intermediate step: PAR preprocessing outputs, Cholesky factors, seed-derived noise vectors, correlated noise, and sampled indices.

## Anticipated Scope

- **Files likely to be modified**:
  - `crates/cobre-stochastic/tests/conformance.rs` (new integration test file)
  - Possibly `crates/cobre-stochastic/tests/fixtures/` (test data if needed)
- **Key decisions needed**:
  - Whether to use the exact hand-computed reference values from the spec or compute them independently (depends on whether golden values are available after Epic 02 implementation)
  - Whether to test intermediate steps (PAR -> Cholesky -> noise -> tree) individually or only the end-to-end result (depends on API visibility decisions from ticket-010)
  - What statistical tolerance to use for correlated noise validation (depends on sample size and the correlation strength in the fixture)
- **Open questions**:
  - Are the golden reference values in sampling-scheme-testing.md sufficient, or do we need to compute additional reference values using a third-party tool (e.g., Python/NumPy)?
  - Should the conformance test also validate memory layout properties (stage-major ordering, contiguous allocation)?

## Dependencies

- **Blocked By**: ticket-010-organize-public-api.md
- **Blocks**: ticket-013-write-documentation.md

## Effort Estimate

**Points**: 3
**Confidence**: Low (will be re-estimated during refinement)
