# Epic 01: Crate Scaffold and PAR Types

## Goal

Establish the `cobre-stochastic` crate with proper dependencies, error types, and the PAR preprocessing pipeline that transforms raw `InflowModel` parameters from `cobre-core` into the cache-friendly `PrecomputedParLp` struct consumed by downstream solvers.

## Scope

- Update `Cargo.toml` with dependencies (`cobre-core`, `siphasher`, `rand`, `rand_distr`, `rand_pcg`)
- Define `StochasticError` enum for all error conditions
- Implement `PrecomputedParLp` with flat contiguous arrays (deterministic_base, sigma, psi)
- Implement PAR parameter validation (positive residual variance, AR order consistency)
- Set up module structure (`par/`, `correlation/`, `noise/`, `tree/`, `sampling/`)

## Out of Scope

- Cholesky decomposition (Epic 02)
- Opening tree generation (Epic 02)
- Noise generation and seed derivation (Epic 02)
- InSample sampling (Epic 03)

## Tickets

| Ticket | Title | Points | Blocks |
|--------|-------|--------|--------|
| ticket-001 | Scaffold crate structure and dependencies | 2 | ticket-002, ticket-003 |
| ticket-002 | Implement PrecomputedParLp struct and builder | 3 | ticket-003 |
| ticket-003 | Implement PAR parameter validation | 2 | Epic 02 tickets |

## Dependencies

- **Blocked By**: None (cobre-core Phase 1 is complete)
- **Blocks**: Epic 02 (needs PrecomputedParLp for opening tree generation)

## Success Criteria

- `cargo build -p cobre-stochastic` compiles cleanly
- `cargo clippy -p cobre-stochastic --all-targets -- -D warnings` passes
- PrecomputedParLp correctly computes deterministic_base and sigma from hand-verifiable inputs
- All unit tests pass
