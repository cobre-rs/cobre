# PAR Coefficient Storage Redesign

## Overview

Emergency implementation plan to redesign PAR coefficient storage across the Cobre ecosystem, changing from original-unit coefficients (psi) to standardized coefficients (psi*) with explicit `residual_std_ratio`. This must be completed BEFORE Phase 5 (cobre-stochastic) begins.

## Tech Stack

- Rust 2024 edition, MSRV 1.85
- Markdown specs (cobre-docs)
- Apache Parquet (arrow + parquet crates)

## Design Reference

`/home/rogerio/git/cobre/docs/design/PAR-COEFFICIENT-REDESIGN.md`

## Epic Summary

| Epic | Name | Tickets | Status |
| --- | --- | --- | --- |
| 01 | Spec Updates (cobre-docs) | 2 | pending |
| 02 | Core Type and I/O Updates (cobre-core + cobre-io) | 3 | pending |
| 03 | Validation Updates and Phase 5 Plan Patch | 2 | pending |

## Progress Tracking

| Ticket | Title | Epic | Status | Detail Level | Readiness | Quality | Badge |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ticket-001 | Update PAR Inflow Model and Input Scenarios Specs | epic-01 | completed | Detailed | 0.95 | 0.96 | EXCELLENT |
| ticket-002 | Add DEC-020 and Update Spec Gap Inventory | epic-01 | completed | Detailed | 0.93 | 0.75 | ACCEPTABLE |
| ticket-003 | Add residual_std_ratio to InflowModel in cobre-core | epic-02 | completed | Detailed | 0.96 | 0.95 | EXCELLENT |
| ticket-004 | Add residual_std_ratio to InflowArCoefficientRow Parser | epic-02 | completed | Detailed | 0.96 | 0.88 | ACCEPTABLE |
| ticket-005 | Update Assembly Logic for residual_std_ratio | epic-02 | completed | Detailed | 0.96 | 0.90 | EXCELLENT |
| ticket-006 | Update Validation Layers for residual_std_ratio | epic-03 | pending | Detailed | 0.94 | -- | -- |
| ticket-007 | Update Phase 5 Plan Tickets to Remove ASSUMPTION Flags | epic-03 | pending | Detailed | 0.94 | -- | -- |

## Dependency Graph

```
ticket-001 (specs)
    |
    +---> ticket-002 (DEC-020, gap inventory)
    |
    +---> ticket-003 (cobre-core InflowModel)
              |
              +---> ticket-004 (cobre-io AR parser)
              |         |
              |         +---> ticket-005 (assembly)
              |         |         |
              |         +---> ticket-006 (validation)
              |                   |
              |                   +---> ticket-007 (Phase 5 plan patch)
              |
              +---> ticket-005 (assembly) [also blocked by 004]
```

## Quality Gates

- `cargo test --workspace --all-features` must pass
- `cargo clippy --all-targets --all-features -- -D warnings` must pass
- `grep -riE 'sddp' crates/cobre-core/src/ crates/cobre-io/src/` must return empty
- All 3 [ASSUMPTION] flags in Phase 5 plan must be resolved

## Key Decisions (Pre-Resolved)

All decisions documented in `docs/design/PAR-COEFFICIENT-REDESIGN.md` section 8:

1. Store psi* (standardized by seasonal std) -- NOT original-unit psi
2. Store residual_std_ratio -- NOT sigma_m directly
3. Skip stationarity check -- deferred
4. Require positive definite for Cholesky -- not PSD
5. Remove ar_order from seasonal stats -- derive from coefficient count (no redundant data)
