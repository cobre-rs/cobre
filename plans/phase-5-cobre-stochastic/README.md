# Phase 5: cobre-stochastic -- Scenario Generation & PAR Preprocessing

## Overview

Phase 5 implements the `cobre-stochastic` crate, the scenario generation layer of the Cobre ecosystem. It preprocesses PAR(p) autoregressive inflow models into cache-friendly layouts, generates deterministic opening trees using SipHash-1-3 seeded RNG (DEC-017), produces correlated noise vectors via Cholesky decomposition, and implements the InSample sampling scheme.

**Tech Stack**: Rust 2024 edition, MSRV 1.85
**Dependencies**: cobre-core, siphasher 1.x, rand 0.10, rand_distr 0.6, rand_pcg 0.10, thiserror 2
**Plan Type**: Progressive (epics 1-2 detailed, epics 3-5 outline)

## Epic Summary

| Epic | Name | Tickets | Status | Phase |
|------|------|---------|--------|-------|
| 01 | Crate Scaffold and PAR Types | 3 (detailed) | completed | completed |
| 02 | Noise Generation and Opening Tree | 4 (detailed) | completed | completed |
| 03 | InSample Sampling and Integration | 3 (refined) | completed | completed |
| 04 | Conformance Tests | 2 (refined) | pending | executing |
| 05 | Documentation and Phase-Close | 1 (outline) | pending | outline |

## Dependency Graph

```
ticket-001 (scaffold)
  |
  +---> ticket-002 (PrecomputedParLp) ---> ticket-003 (PAR validation)
  |
  +---> ticket-004 (SipHash seed) --------+
  |                                        |
  +---> ticket-005 (Cholesky/correlation) -+---> ticket-007 (tree generation)
  |                                        |
  +---> ticket-006 (OpeningTree types) ----+
                                           |
                  ticket-008 (InSample) <--+
                  ticket-009 (context)  <--+
                        |
                        v
                  ticket-010 (public API)
                        |
                        v
            ticket-011 (conformance tests)
            ticket-012 (reproducibility tests)
                        |
                        v
              ticket-013 (documentation)
```

## Progress Tracking

| Ticket | Title | Epic | Status | Detail Level | Readiness | Quality | Badge |
|--------|-------|------|--------|--------------|-----------|---------|-------|
| ticket-001 | Scaffold crate structure | epic-01 | completed | Detailed | 0.93 | 0.75 | ACCEPTABLE |
| ticket-002 | Implement PrecomputedParLp | epic-01 | completed | Detailed | 0.91 | 1.00 | EXCELLENT |
| ticket-003 | Implement PAR validation | epic-01 | completed | Detailed | 0.94 | 0.98 | EXCELLENT |
| ticket-004 | Implement SipHash seed derivation | epic-02 | completed | Detailed | 1.00 | 1.00 | EXCELLENT |
| ticket-005 | Implement Cholesky and correlation | epic-02 | completed | Detailed | 0.92 | 0.90 | EXCELLENT |
| ticket-006 | Implement OpeningTree types | epic-02 | completed | Detailed | 0.98 | 1.00 | EXCELLENT |
| ticket-007 | Implement opening tree generation | epic-02 | completed | Detailed | 1.00 | 0.95 | EXCELLENT |
| ticket-008 | Implement InSample forward sampling | epic-03 | completed | Refined | 1.00 | 0.88 | ACCEPTABLE |
| ticket-009 | Implement StochasticContext initialization | epic-03 | completed | Refined | 1.00 | 0.75 | ACCEPTABLE |
| ticket-010 | Organize public API and re-exports | epic-03 | completed | Refined | 1.00 | 0.95 | EXCELLENT |
| ticket-011 | Implement pipeline conformance tests | epic-04 | completed | Refined | 1.00 | 0.75 | ACCEPTABLE |
| ticket-012 | Implement reproducibility/invariance tests | epic-04 | completed | Refined | 1.00 | 0.96 | EXCELLENT |
| ticket-013 | Write documentation and close phase | epic-05 | pending | Outline | -- | -- | -- |

## Resolved Decisions

All pending decisions from the original plan are resolved by the PAR Coefficient Storage Redesign (DEC-020, `docs/design/PAR-COEFFICIENT-REDESIGN.md`):

1. **Sigma derivation formula** (was ticket-002): Resolved. sigma = s_m * residual_std_ratio. No autocorrelation reconstruction needed. `residual_std_ratio` is stored directly in the AR coefficients file and passed through `InflowModel`.
2. **Stationarity check** (was ticket-003): Resolved. Stationarity check is deferred for the minimal viable implementation per design doc section 8. Stored coefficients are in standardized form (psi*), so no reverse-standardization is needed for any future check.
3. **Cholesky PSD vs PD** (was ticket-005): Resolved. Per DEC-020 section 8, positive definite matrices are required. PSD matrices with zero eigenvalues produce a CholeskyDecompositionFailed error.

## Epic 03 Refinement Decisions

The following open questions from the outline tickets were resolved during refinement using learnings from epics 01-02:

1. **Return type for `sample_forward`** (was ticket-008): Returns `(usize, &[f64])` -- the index for diagnostics and a borrowed slice from the `OpeningTreeView`. No `NoiseVector` newtype needed; a plain slice is the simplest correct interface.
2. **`OpeningTreeView` vs `&OpeningTree`** (was ticket-008): Takes `OpeningTreeView<'_>` to support shared-memory scenarios.
3. **RNG signature** (was ticket-008): Uses `derive_forward_seed` + `rng_from_seed` + `rng.random::<u64>() % n` -- reuses existing infrastructure, caller does not provide a pre-initialized RNG.
4. **`build_stochastic_context` input** (was ticket-009): Takes `&System` + explicit `u64` seed. No `cobre-io` dependency added. The `Option<i64>` -> `u64` conversion is the caller's responsibility.
5. **Validation inside builder** (was ticket-009): `validate_par_parameters` is called inside `build_stochastic_context` (mandatory validation, not optional).
6. **`StochasticContext` ownership** (was ticket-009): Owns all components. `tree_view()` returns a borrowed view.
7. **`CholeskyFactor` / `GroupFactor` visibility** (was ticket-010): Remain in flat re-exports for advanced inspection.

## Epic 04 Refinement Decisions

The following open questions from the outline tickets were resolved during refinement using learnings from epics 01-03:

1. **Golden reference values** (was ticket-011): Do NOT use golden RNG output values. The opening tree is generated by the SipHash+Pcg64 chain, and exact values depend on crate versions. Test structural properties (dimensions, finiteness) and statistical properties (N(0,1) marginals) instead.
2. **Intermediate step testing** (was ticket-011): Test intermediate steps (PAR coefficients) via `StochasticContext` accessors, which expose `par_lp()`, `correlation()`, and `opening_tree()`. No need for additional API visibility.
3. **Statistical tolerance** (was ticket-011): Use the established bounds from epic-02: `|mean| < 0.15` and `|std - 1| < 0.15` over 500+ samples.
4. **Reproducibility test design** (was ticket-012): Compare pairs (build twice, compare). No need for N repeated runs.
5. **Declaration-order invariance** (was ticket-012): Test reversed hydro IDs only. `SystemBuilder` canonically sorts by EntityId internally, so both orders produce identical results.
6. **Genericity gate implementation** (was ticket-012): Use `std::process::Command` to run grep within a `#[test]` function. The grep targets `src/` only (not `tests/`).
7. **Test file organization** (was ticket-012): Single file `reproducibility.rs` for all invariant tests. No separate `invariance.rs` or `genericity.rs`.

## Key Specs

- `~/git/cobre-docs/src/specs/architecture/scenario-generation.md` -- Core scenario generation spec
- `~/git/cobre-docs/src/specs/math/par-inflow-model.md` -- PAR(p) mathematical formulation
- `~/git/cobre-docs/src/specs/math/inflow-nonnegativity.md` -- Inflow non-negativity methods
- `~/git/cobre-docs/src/specs/architecture/sampling-scheme-trait.md` -- SamplingScheme enum
- `~/git/cobre-docs/src/specs/architecture/sampling-scheme-testing.md` -- Conformance test fixtures
- `~/git/cobre-docs/src/specs/data-model/input-scenarios.md` -- Input data formats
