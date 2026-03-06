# ticket-004 Implement SipHash-1-3 seed derivation

## Context

### Background

DEC-017 mandates communication-free parallel noise generation: every rank and thread independently derives identical noise via deterministic SipHash-1-3 seed derivation. The seed derivation function takes a base seed and context tuple (iteration, scenario, stage for forward pass; opening_index, stage for opening tree) and produces a deterministic 64-bit seed that initializes a PRNG. The `siphasher` crate (version 1.x) provides the `SipHasher13` type with guaranteed output stability across versions.

### Relation to Epic

This is the first ticket in Epic 02. The seed derivation function is the foundation of all noise generation -- both the opening tree (ticket-007) and forward pass noise use this function to ensure cross-rank reproducibility.

### Current State

- `crates/cobre-stochastic/src/noise/seed.rs` is a placeholder file (from ticket-001)
- `siphasher = "1"` is a dependency in Cargo.toml (from ticket-001)
- No seed derivation code exists

## Specification

### Requirements

1. Implement two seed derivation functions in `crates/cobre-stochastic/src/noise/seed.rs`:

```rust
use siphasher::sip::SipHasher13;
use std::hash::Hasher;

/// Derive a deterministic 64-bit seed for forward pass noise generation.
///
/// The derived seed is identical for the same `(base_seed, iteration,
/// scenario, stage)` tuple regardless of MPI rank, thread ID, or
/// process restart. Uses SipHash-1-3 per DEC-017.
///
/// # Wire format
///
/// The hash input is a 20-byte little-endian concatenation:
/// ```text
/// base_seed (u64, 8 bytes) ++ iteration (u32, 4 bytes)
///   ++ scenario (u32, 4 bytes) ++ stage (u32, 4 bytes)
/// ```
pub fn derive_forward_seed(
    base_seed: u64,
    iteration: u32,
    scenario: u32,
    stage: u32,
) -> u64

/// Derive a deterministic 64-bit seed for opening tree generation.
///
/// The derived seed is identical for the same `(base_seed,
/// opening_index, stage)` tuple. Uses SipHash-1-3 per DEC-017.
///
/// # Wire format
///
/// The hash input is a 16-byte little-endian concatenation:
/// ```text
/// base_seed (u64, 8 bytes) ++ opening_index (u32, 4 bytes)
///   ++ stage (u32, 4 bytes)
/// ```
pub fn derive_opening_seed(
    base_seed: u64,
    opening_index: u32,
    stage: u32,
) -> u64
```

2. Both functions must:
   - Use `SipHasher13::new()` (zero-key) for the hasher
   - Encode all integers as little-endian bytes via `.to_le_bytes()`
   - Concatenate byte arrays in the exact order specified (no separators, no length prefixes)
   - Feed the concatenated bytes to the hasher via `Hasher::write()`
   - Return the `Hasher::finish()` value as the derived seed

3. Implement a helper for RNG initialization in `crates/cobre-stochastic/src/noise/rng.rs`:

```rust
use rand::SeedableRng;
use rand_pcg::Pcg64;

/// Initialize a Pcg64 RNG from a derived 64-bit seed.
///
/// Expands the 64-bit seed to the 256-bit state required by Pcg64
/// using the `seed_from_u64` method, which applies a deterministic
/// expansion algorithm.
pub fn rng_from_seed(seed: u64) -> Pcg64
```

### Inputs/Props

- `base_seed: u64` -- user-configured base seed from `ScenarioSource.seed`
- Context integers: `iteration: u32`, `scenario: u32`, `stage: u32`, `opening_index: u32`

### Outputs/Behavior

- Returns a deterministic `u64` seed
- Same inputs always produce the same output (cross-platform, cross-build)

### Error Handling

These functions are infallible (pure computation, no I/O, no allocation failure).

## Acceptance Criteria

- [ ] Given `derive_forward_seed(42, 0, 0, 0)` called twice, when the results are compared, then they are bitwise identical
- [ ] Given `derive_forward_seed(42, 0, 0, 0)` and `derive_forward_seed(42, 0, 0, 1)`, when the results are compared, then they differ (different stage produces different seed)
- [ ] Given `derive_forward_seed(42, 0, 0, 0)` and `derive_forward_seed(43, 0, 0, 0)`, when the results are compared, then they differ (different base seed produces different seed)
- [ ] Given `derive_opening_seed(42, 0, 0)` and `derive_forward_seed(42, 0, 0, 0)`, when the results are compared, then they differ (different wire format lengths: 16 vs 20 bytes)
- [ ] Given `rng_from_seed(12345)`, when `rng.gen::<f64>()` is called twice across two separate `rng_from_seed(12345)` initializations, then the sequence of values is identical

## Implementation Guide

### Suggested Approach

1. Implement `derive_forward_seed`:
   ```rust
   pub fn derive_forward_seed(base_seed: u64, iteration: u32, scenario: u32, stage: u32) -> u64 {
       let mut hasher = SipHasher13::new();
       hasher.write(&base_seed.to_le_bytes());
       hasher.write(&iteration.to_le_bytes());
       hasher.write(&scenario.to_le_bytes());
       hasher.write(&stage.to_le_bytes());
       hasher.finish()
   }
   ```
2. Implement `derive_opening_seed` similarly with 16-byte input
3. Implement `rng_from_seed`:
   ```rust
   pub fn rng_from_seed(seed: u64) -> Pcg64 {
       Pcg64::seed_from_u64(seed)
   }
   ```
4. Add tests verifying determinism, sensitivity, and cross-function differentiation

### Key Files to Modify

- `crates/cobre-stochastic/src/noise/seed.rs` (primary implementation)
- `crates/cobre-stochastic/src/noise/rng.rs` (RNG helper)
- `crates/cobre-stochastic/src/noise/mod.rs` (re-exports)

### Patterns to Follow

- Pure function pattern: no state, no side effects, no I/O
- Explicit `.to_le_bytes()` on every integer (DEC-017 mandates little-endian)
- `SipHasher13::new()` uses zero key -- no key derivation needed

### Pitfalls to Avoid

- Do NOT use `std::hash::Hash` trait implementations on primitive types -- they may pad or format differently; use raw `.to_le_bytes()` concatenation
- Do NOT use `std::collections::hash_map::DefaultHasher` -- its output is not stable across Rust versions (DEC-017 explicitly prohibits this)
- Do NOT use `Pcg64::from_seed()` directly (requires 32-byte array) -- use `Pcg64::seed_from_u64()` which handles the expansion
- Verify `siphasher` version resolves to 1.x, not 0.x
- Verify `rand_pcg` version is compatible with `rand` 0.9 -- `rand_pcg` 0.4 may need `rand` 0.9 feature

## Testing Requirements

### Unit Tests

- `derive_forward_seed` determinism: same inputs produce same output
- `derive_forward_seed` sensitivity: varying each parameter changes the output
- `derive_opening_seed` determinism and sensitivity
- Cross-function differentiation: same partial inputs produce different seeds between forward and opening functions
- `rng_from_seed` determinism: same seed produces identical RNG sequence
- Golden value test: pin one specific `derive_forward_seed(42, 0, 0, 0)` output as a regression test to catch accidental hash algorithm changes

### Integration Tests

None for this ticket.

### E2E Tests

Not applicable.

## Dependencies

- **Blocked By**: ticket-001-scaffold-crate-structure.md
- **Blocks**: ticket-007-implement-opening-tree-generation.md

## Effort Estimate

**Points**: 2
**Confidence**: High
