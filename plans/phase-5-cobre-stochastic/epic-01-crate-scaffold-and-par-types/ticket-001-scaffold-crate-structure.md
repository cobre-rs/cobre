# ticket-001 Scaffold crate structure and dependencies

## Context

### Background

Phase 5 implements `cobre-stochastic`, the scenario generation layer of the Cobre ecosystem. The crate currently exists as an empty stub with only module-level documentation in `lib.rs` and a bare `Cargo.toml`. This ticket establishes the full module structure, adds all required dependencies, defines the error type, and sets up the crate for implementation.

### Relation to Epic

This is the first ticket in Epic 01 (Crate Scaffold and PAR Types). All subsequent tickets depend on the module structure and error types established here.

### Current State

- `crates/cobre-stochastic/src/lib.rs`: 22 lines of doc comments only, no code
- `crates/cobre-stochastic/Cargo.toml`: Bare workspace member with no dependencies

## Specification

### Requirements

1. Update `Cargo.toml` to add dependencies:
   - `cobre-core = { path = "../cobre-core" }` (mandatory)
   - `siphasher = "1"` (DEC-017: SipHash-1-3 for deterministic seed derivation)
   - `rand = "0.9"` (PRNG framework)
   - `rand_distr = "0.5"` (StandardNormal distribution)
   - `rand_pcg = "0.4"` (Pcg64 generator)
   - `thiserror = "2"` (error derive macro, consistent with cobre-solver and cobre-comm)
2. Create module structure under `src/`:
   - `error.rs` -- `StochasticError` enum
   - `par/mod.rs` -- PAR module (re-exports)
   - `par/precompute.rs` -- placeholder
   - `par/validation.rs` -- placeholder
   - `correlation/mod.rs` -- Correlation module (re-exports)
   - `correlation/cholesky.rs` -- placeholder
   - `correlation/resolve.rs` -- placeholder
   - `noise/mod.rs` -- Noise module (re-exports)
   - `noise/seed.rs` -- placeholder
   - `noise/rng.rs` -- placeholder
   - `tree/mod.rs` -- Tree module (re-exports)
   - `tree/opening_tree.rs` -- placeholder
   - `tree/generate.rs` -- placeholder
   - `sampling/mod.rs` -- Sampling module (re-exports)
   - `sampling/insample.rs` -- placeholder
3. Define `StochasticError` in `error.rs`:
   - `InvalidParParameters { hydro_id: i32, stage_id: i32, reason: String }` -- PAR validation failure
   - `NonPositiveResidualVariance { hydro_id: i32, stage_id: i32, sigma_squared: f64 }` -- sigma^2 <= 0
   - `CholeskyDecompositionFailed { profile_name: String, reason: String }` -- non-PSD matrix
   - `InvalidCorrelation { profile_name: String, reason: String }` -- correlation validation error
   - `InsufficientData { context: String }` -- missing required input data
   - `SeedDerivationError { reason: String }` -- seed computation failure
4. Update `lib.rs`:
   - Remove stale doc comment references to "Monte Carlo" and "SDDP"
   - Use generic language: "stochastic process models", "iterative optimization algorithms", "scenario generation"
   - Add `#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]`
   - Re-export `StochasticError` and module contents

### Inputs/Props

None -- this is a pure scaffold ticket.

### Outputs/Behavior

- `cargo build -p cobre-stochastic` succeeds
- `cargo clippy -p cobre-stochastic --all-targets -- -D warnings` passes
- All placeholder modules compile (empty or with minimal stub content)

### Error Handling

`StochasticError` implements `std::error::Error + Send + Sync` via `thiserror::Error` derive.

## Acceptance Criteria

- [ ] Given the updated `Cargo.toml`, when `cargo build -p cobre-stochastic` is run, then it compiles without errors
- [ ] Given the `StochasticError` enum in `crates/cobre-stochastic/src/error.rs`, when each variant is constructed, then it implements `std::error::Error + Send + Sync + 'static`
- [ ] Given `crates/cobre-stochastic/src/lib.rs`, when inspected, then it contains zero occurrences of "SDDP", "sddp", "forward pass", or "backward pass"
- [ ] Given the module structure, when `cargo clippy -p cobre-stochastic --all-targets -- -D warnings` is run, then zero warnings are emitted
- [ ] Given the module structure, when each of `par`, `correlation`, `noise`, `tree`, `sampling` is imported in a test, then the import resolves successfully

## Implementation Guide

### Suggested Approach

1. Update `Cargo.toml` first, verify dependency versions compile
2. Create `error.rs` with `StochasticError` enum using `#[derive(Debug, thiserror::Error)]`
3. Create all module directories and `mod.rs` files with minimal content (doc comment + empty re-exports)
4. Create placeholder `.rs` files with module-level doc comment only
5. Update `lib.rs` with module declarations, re-exports, and generic doc comments
6. Run `cargo clippy` and fix any issues

### Key Files to Modify

- `crates/cobre-stochastic/Cargo.toml`
- `crates/cobre-stochastic/src/lib.rs`
- `crates/cobre-stochastic/src/error.rs` (new)
- `crates/cobre-stochastic/src/par/mod.rs` (new)
- `crates/cobre-stochastic/src/par/precompute.rs` (new)
- `crates/cobre-stochastic/src/par/validation.rs` (new)
- `crates/cobre-stochastic/src/correlation/mod.rs` (new)
- `crates/cobre-stochastic/src/correlation/cholesky.rs` (new)
- `crates/cobre-stochastic/src/correlation/resolve.rs` (new)
- `crates/cobre-stochastic/src/noise/mod.rs` (new)
- `crates/cobre-stochastic/src/noise/seed.rs` (new)
- `crates/cobre-stochastic/src/noise/rng.rs` (new)
- `crates/cobre-stochastic/src/tree/mod.rs` (new)
- `crates/cobre-stochastic/src/tree/opening_tree.rs` (new)
- `crates/cobre-stochastic/src/tree/generate.rs` (new)
- `crates/cobre-stochastic/src/sampling/mod.rs` (new)
- `crates/cobre-stochastic/src/sampling/insample.rs` (new)

### Patterns to Follow

- Error type pattern: follow `cobre-comm/src/error.rs` (`CommError`) using `thiserror::Error` derive
- Module layout: follow `cobre-comm/src/` (one file per concern, `mod.rs` for re-exports)
- Test suppression: `#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]` in `lib.rs` (follows cobre-solver and cobre-comm)
- Doc-markdown lint: use backtick-wrapped brackets for units (`[`m3/s`]`)

### Pitfalls to Avoid

- Do NOT reference "SDDP", "forward pass", or "backward pass" in code, types, or doc comments -- this is an infrastructure crate (see CLAUDE.md "Infrastructure crate genericity")
- Do NOT add `rand` features beyond what is needed -- `rand` 0.9 changed the feature structure from 0.8; verify the correct feature set
- Do NOT use `workspace = true` with per-lint overrides in `[lints]` -- if `unsafe_code = "forbid"` is inherited from workspace, it cannot be overridden (cobre-stochastic should NOT need unsafe, so workspace lints are fine)
- Do NOT add serde derives to `StochasticError` -- error types are not serialized
- Verify that `siphasher = "1"` resolves to 1.x (not 0.x) -- the `SipHasher13` type is in the 1.x series

## Testing Requirements

### Unit Tests

- `error.rs`: Test that each `StochasticError` variant can be constructed and implements `std::error::Error`
- `lib.rs`: Compile-time assertion that `StochasticError: std::error::Error + Send + Sync`

### Integration Tests

None for this ticket (scaffold only).

### E2E Tests

Not applicable.

## Dependencies

- **Blocked By**: None
- **Blocks**: ticket-002-implement-precomputed-par-lp.md, ticket-003-implement-par-validation.md

## Effort Estimate

**Points**: 2
**Confidence**: High
