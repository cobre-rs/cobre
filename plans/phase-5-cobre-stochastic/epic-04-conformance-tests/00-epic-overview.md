# Epic 04: Conformance Tests

## Goal

Implement a dedicated conformance test suite that validates the end-to-end stochastic pipeline against hand-computed reference values from the sampling-scheme-testing.md spec. These tests exercise the full pipeline (PAR preprocessing -> seed derivation -> noise generation -> correlation -> opening tree -> InSample sampling) with known inputs and verify bit-exact or statistically bounded outputs. The suite also validates cross-concern properties: deterministic reproducibility, declaration-order invariance, and infrastructure genericity compliance.

## Scope

- End-to-end pipeline tests with the 3-stage, 2-hydro, 5-opening fixture from sampling-scheme-testing.md
- Deterministic reproducibility tests: same seed produces identical output across runs
- Declaration-order invariance tests: reordered hydro IDs produce identical results
- Seed sensitivity tests: different seeds produce different trees
- Statistical validation: correlated noise vectors match target correlation within tolerance
- Infrastructure genericity gate: zero SDDP references in crate source

## Out of Scope

- Performance benchmarks (deferred to post-MVP)
- External and Historical sampling scheme conformance (deferred variants)
- Cross-platform reproducibility verification (CI validates single platform)

## Tickets

| Ticket | Title | Points | Blocks |
|--------|-------|--------|--------|
| ticket-011 | Implement end-to-end pipeline conformance tests | 3 | Epic 05 |
| ticket-012 | Implement reproducibility and invariance tests | 2 | Epic 05 |

## Dependencies

- **Blocked By**: Epic 03 (needs complete public API)
- **Blocks**: Epic 05 (documentation references test results)

## Success Criteria

- All conformance tests pass with `cargo test -p cobre-stochastic`
- End-to-end test uses the exact fixture from sampling-scheme-testing.md
- `grep -riE 'sddp' crates/cobre-stochastic/` returns zero matches
