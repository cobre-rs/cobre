# Phase 5: cobre-stochastic -- Scenario Generation & PAR Preprocessing

## Overview

Phase 5 implements the `cobre-stochastic` crate, the scenario generation layer of the Cobre ecosystem. It preprocesses PAR(p) autoregressive inflow models into cache-friendly layouts, generates deterministic opening trees using SipHash-1-3 seeded RNG (DEC-017), produces correlated noise vectors via Cholesky decomposition, and implements the InSample sampling scheme.

**Tech Stack**: Rust 2024 edition, MSRV 1.85
**Dependencies**: cobre-core, siphasher 1.x, rand 0.9, rand_distr 0.5, rand_pcg 0.4, thiserror 2
**Plan Type**: Progressive (epics 1-2 detailed, epics 3-5 outline)

## Epic Summary

| Epic | Name | Tickets | Status | Phase |
|------|------|---------|--------|-------|
| 01 | Crate Scaffold and PAR Types | 3 (detailed) | pending | executing |
| 02 | Noise Generation and Opening Tree | 4 (detailed) | pending | executing |
| 03 | InSample Sampling and Integration | 3 (outline) | pending | outline |
| 04 | Conformance Tests | 2 (outline) | pending | outline |
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
| ticket-001 | Scaffold crate structure | epic-01 | pending | Detailed | 0.93 | -- | -- |
| ticket-002 | Implement PrecomputedParLp | epic-01 | pending | Detailed | 0.91 | -- | -- |
| ticket-003 | Implement PAR validation | epic-01 | pending | Detailed | 0.94 | -- | -- |
| ticket-004 | Implement SipHash seed derivation | epic-02 | pending | Detailed | 1.00 | -- | -- |
| ticket-005 | Implement Cholesky and correlation | epic-02 | pending | Detailed | 0.92 | -- | -- |
| ticket-006 | Implement OpeningTree types | epic-02 | pending | Detailed | 0.98 | -- | -- |
| ticket-007 | Implement opening tree generation | epic-02 | pending | Detailed | 1.00 | -- | -- |
| ticket-008 | Implement InSample forward sampling | epic-03 | pending | Outline | -- | -- | -- |
| ticket-009 | Implement StochasticContext initialization | epic-03 | pending | Outline | -- | -- | -- |
| ticket-010 | Organize public API and re-exports | epic-03 | pending | Outline | -- | -- | -- |
| ticket-011 | Implement pipeline conformance tests | epic-04 | pending | Outline | -- | -- | -- |
| ticket-012 | Implement reproducibility/invariance tests | epic-04 | pending | Outline | -- | -- | -- |
| ticket-013 | Write documentation and close phase | epic-05 | pending | Outline | -- | -- | -- |

## Resolved Decisions

All pending decisions from the original plan are resolved by the PAR Coefficient Storage Redesign (DEC-020, `docs/design/PAR-COEFFICIENT-REDESIGN.md`):

1. **Sigma derivation formula** (was ticket-002): Resolved. sigma = s_m * residual_std_ratio. No autocorrelation reconstruction needed. `residual_std_ratio` is stored directly in the AR coefficients file and passed through `InflowModel`.
2. **Stationarity check** (was ticket-003): Resolved. Stationarity check is deferred for the minimal viable implementation per design doc section 8. Stored coefficients are in standardized form (ψ*), so no reverse-standardization is needed for any future check.
3. **Cholesky PSD vs PD** (was ticket-005): Resolved. Per DEC-020 section 8, positive definite matrices are required. PSD matrices with zero eigenvalues produce a CholeskyDecompositionFailed error.

## Key Specs

- `~/git/cobre-docs/src/specs/architecture/scenario-generation.md` -- Core scenario generation spec
- `~/git/cobre-docs/src/specs/math/par-inflow-model.md` -- PAR(p) mathematical formulation
- `~/git/cobre-docs/src/specs/math/inflow-nonnegativity.md` -- Inflow non-negativity methods
- `~/git/cobre-docs/src/specs/architecture/sampling-scheme-trait.md` -- SamplingScheme enum
- `~/git/cobre-docs/src/specs/architecture/sampling-scheme-testing.md` -- Conformance test fixtures
- `~/git/cobre-docs/src/specs/data-model/input-scenarios.md` -- Input data formats
