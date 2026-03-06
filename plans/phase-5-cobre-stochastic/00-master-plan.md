# Master Plan: Phase 5 -- cobre-stochastic (Scenario Generation & PAR Preprocessing)

## Executive Summary

Phase 5 implements the `cobre-stochastic` crate -- the scenario generation layer that preprocesses PAR(p) autoregressive inflow models into cache-friendly layouts, generates deterministic opening trees using SipHash-1-3 seeded RNG (DEC-017), produces correlated noise vectors via Cholesky decomposition, and implements the InSample sampling scheme. This is an infrastructure crate with zero algorithm-specific references.

## Goals & Non-Goals

### Goals

1. **PAR preprocessing**: Transform raw `InflowModel` from `cobre-core` into a `PrecomputedParLp` struct with contiguous arrays for hot-path access (deterministic base, sigma, psi coefficients)
2. **Cholesky decomposition**: Decompose correlation matrices from `CorrelationModel` into lower-triangular factors, cached per profile for runtime correlation transforms
3. **Opening tree generation**: Build a fixed `OpeningTree` with stage-major memory layout using deterministic SipHash-1-3 seed derivation per (opening_index, stage)
4. **Correlated noise generation**: Generate independent N(0,1) samples and transform via Cholesky factor to produce spatially correlated noise vectors
5. **InSample forward sampling**: Implement the minimal-viable sampling scheme -- sample a random index into the opening tree at each stage
6. **Deterministic reproducibility**: Identical results given the same base seed, regardless of MPI rank count, thread count, or restart

### Non-Goals

- External and Historical sampling schemes (deferred variants)
- LHS, QMC-Sobol, QMC-Halton, Selective sampling methods (deferred)
- PAR model fitting from historical data (Yule-Walker, BIC order selection) -- Phase 5 only preprocesses user-provided parameters
- Load scenario generation (implemented but minimal -- mean + std + block factors)
- SharedRegion allocation for opening tree (depends on cobre-comm's SharedMemoryProvider, which is a cobre-sddp integration concern)
- Noise inversion for External/Historical schemes
- Complete tree mode

## Architecture Overview

### Current State

- `cobre-stochastic` is an empty stub crate (`lib.rs` with only module-level docs, `Cargo.toml` with no dependencies beyond workspace defaults)
- `cobre-core` provides: `InflowModel`, `LoadModel`, `CorrelationModel`, `CorrelationProfile`, `CorrelationGroup`, `ScenarioSource`, `SamplingScheme`, `Stage`, `ScenarioSourceConfig`, `NoiseMethod`, `SeasonMap`
- `cobre-io` provides: `ScenarioData` (assembled inflow models, load models, correlation, history, external scenarios, load factors)

### Target State

```
cobre-stochastic/
  src/
    lib.rs                    -- Public re-exports, crate docs
    error.rs                  -- StochasticError enum
    par/
      mod.rs                  -- PAR module re-exports
      precompute.rs           -- PrecomputedParLp: deterministic_base, sigma, psi
      validation.rs           -- PAR parameter validation (positive sigma, stationarity)
    correlation/
      mod.rs                  -- Correlation module re-exports
      cholesky.rs             -- Cholesky decomposition and correlated noise transform
      resolve.rs              -- Profile-to-stage resolution, DecomposedCorrelation
    noise/
      mod.rs                  -- Noise module re-exports
      seed.rs                 -- SipHash-1-3 seed derivation (DEC-017)
      rng.rs                  -- RNG initialization from derived seed, N(0,1) sampling
    tree/
      mod.rs                  -- Tree module re-exports
      opening_tree.rs         -- OpeningTree struct and OpeningTreeView
      generate.rs             -- Opening tree generation (all stages x all openings)
    sampling/
      mod.rs                  -- Sampling module re-exports
      insample.rs             -- InSample forward sampling (index into opening tree)
  Cargo.toml                  -- Dependencies: cobre-core, siphasher, rand, rand_distr, rand_pcg
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `siphasher` 1.x for seed derivation | DEC-017: SipHash-1-3 output stability across builds; `std::collections::hash_map::DefaultHasher` is explicitly prohibited |
| `rand_pcg::Pcg64` for PRNG | Spec SS2.2a recommends Pcg64; fast, statistically well-tested, deterministic from 64-bit seed |
| `rand_distr::StandardNormal` for N(0,1) | Standard approach; ziggurat algorithm is fast |
| `Box<[f64]>` for immutable arrays | OpeningTree and PrecomputedParLp data never resized after construction; `Box<[f64]>` communicates this invariant |
| Flat contiguous arrays with stage-major layout | Cache-friendly for sequential stage iteration; no nested Vec allocations on hot path |
| No unsafe code | cobre-stochastic performs no FFI; `unsafe_code = "forbid"` inherited from workspace |
| Infrastructure genericity | Zero SDDP references in code, types, docs, or comments; use "iterative optimization", "the calling algorithm", etc. |

## Technical Approach

### Tech Stack

- **Language**: Rust 2024 edition, MSRV 1.85
- **Dependencies**: `cobre-core`, `siphasher` 1.x, `rand` 0.9.x, `rand_distr` 0.5.x, `rand_pcg` 0.4.x (verify exact versions before ticket dispatch)
- **Workspace lints**: `unsafe_code = "forbid"`, `unwrap_used = "deny"`, clippy pedantic

### Component Breakdown

1. **PAR Preprocessing** (`par/`): Takes `&[InflowModel]` + `&[Stage]`, produces `PrecomputedParLp` with flat arrays indexed `[stage][hydro]`
2. **Correlation** (`correlation/`): Takes `&CorrelationModel` + `&[Stage]`, produces `DecomposedCorrelation` with Cholesky factors per profile and stage-to-profile lookup
3. **Noise Infrastructure** (`noise/`): Seed derivation via SipHash-1-3, RNG initialization, N(0,1) sampling
4. **Opening Tree** (`tree/`): Generates fixed opening tree before the optimization loop; stage-major layout with offset-based access
5. **Sampling** (`sampling/`): InSample scheme: sample random index into opening tree per stage

### Data Flow

```
InflowModel[] ──> PrecomputedParLp (deterministic_base, sigma, psi)
                       │
CorrelationModel ──> DecomposedCorrelation (Cholesky factors per profile)
                       │
base_seed ──> SipHash-1-3 seed derivation ──> Pcg64 RNG ──> N(0,1) samples
                       │                                         │
                       └──> Cholesky transform ──> correlated noise η
                                                        │
                                              ──> OpeningTree (fixed, stage-major)
                                                        │
                              InSample sampling ──> opening index ──> noise vector
```

### Testing Strategy

- **Unit tests**: Per-module `#[cfg(test)] mod tests` with hand-computable examples
- **Integration tests**: End-to-end PAR preprocessing + opening tree generation + InSample sampling with known seed, verified against hand-computed values
- **Conformance tests**: Dedicated epic (Epic 04) with reproducibility, determinism, and correctness tests
- **Declaration-order invariance**: Tests with reordered hydro IDs verifying bit-identical results
- **Infrastructure genericity gate**: `grep -riE 'sddp' crates/cobre-stochastic/` at epic boundaries

## Phases & Milestones

| Epic | Name | Scope | Duration |
|------|------|-------|----------|
| 01 | Crate Scaffold and PAR Types | Cargo.toml, error types, PrecomputedParLp, PAR validation | 1-2 weeks |
| 02 | Noise Generation and Opening Tree | SipHash seed, Cholesky, correlated noise, OpeningTree struct | 2-3 weeks |
| 03 | InSample Sampling and Integration | InSample forward sampling, public API, integration | 1-2 weeks |
| 04 | Conformance Tests | Reproducibility, determinism, hand-computed reference values | 1 week |
| 05 | Documentation and Phase-Close | Book chapter, CONTRIBUTING.md, neutral language audit | 1 week |

## Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| `rand` 0.9 API changes from 0.8 | Medium | Verify API before ticket dispatch; pin exact versions |
| Cholesky decomposition for non-PSD matrices | Low | Validation rejects non-PSD matrices; return error on decomposition failure |
| SipHash output mismatch across platforms | Low | DEC-017 mandates little-endian encoding; siphasher 1.x guarantees output stability |
| Numerical precision in sigma computation | Medium | Add tolerance-based validation; warn when sigma^2 is near zero |
| Scope creep into PAR fitting | High | PAR fitting is explicitly a non-goal; only preprocessing of user-provided params |

## Success Metrics

1. All tests pass: `cargo test -p cobre-stochastic`
2. Zero clippy warnings: `cargo clippy -p cobre-stochastic --all-targets -- -D warnings`
3. Zero SDDP references: `grep -riE 'sddp' crates/cobre-stochastic/` returns empty
4. Deterministic output: Same seed produces identical OpeningTree across runs
5. Performance: PrecomputedParLp construction completes in < 1ms for 60 stages x 160 hydros
6. All acceptance criteria met across all tickets
