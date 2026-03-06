# Epic 02: Noise Generation and Opening Tree

## Goal

Implement the noise generation infrastructure (SipHash-1-3 seed derivation, RNG, N(0,1) sampling), Cholesky decomposition for spatial correlation, and the `OpeningTree` struct that holds the fixed pre-generated noise realizations consumed by the backward pass.

## Scope

- SipHash-1-3 deterministic seed derivation per DEC-017
- Pcg64 RNG initialization from derived seed
- Standard normal sampling
- Cholesky decomposition of correlation matrices
- Profile-to-stage correlation resolution
- `OpeningTree` and `OpeningTreeView` types with stage-major layout
- Opening tree generation from seed + correlation

## Out of Scope

- InSample forward sampling (Epic 03)
- LHS, QMC, Selective sampling methods (deferred)
- SharedRegion-based tree sharing (cobre-sddp integration concern)
- PAR model fitting from history (deferred)

## Tickets

| Ticket | Title | Points | Blocks |
|--------|-------|--------|--------|
| ticket-004 | Implement SipHash-1-3 seed derivation | 2 | ticket-007 |
| ticket-005 | Implement Cholesky decomposition and correlation resolution | 3 | ticket-007 |
| ticket-006 | Implement OpeningTree and OpeningTreeView types | 2 | ticket-007 |
| ticket-007 | Implement opening tree generation | 3 | Epic 03 tickets |

## Dependencies

- **Blocked By**: Epic 01 (needs StochasticError, PrecomputedParLp)
- **Blocks**: Epic 03 (InSample sampling reads from OpeningTree)

## Success Criteria

- `derive_seed(42, 0, 0, 0)` produces the same u64 value on every invocation
- Cholesky decomposition of a known 3x3 PSD matrix matches hand-computed lower triangular factor
- `OpeningTree` for 3 stages x 5 openings x 2 entities contains exactly `3 * 5 * 2 = 30` f64 values
- Opening tree generation with seed=42 produces identical output across repeated runs
