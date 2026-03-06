# Accumulated Learnings — Through Epic 05

## Epic 01: Crate Scaffold and PAR Types
- Ticket-001 scope was large (scaffold + types + tests); future scaffolds should be split
- `PrecomputedParLp` layout: flat `Vec<f64>` with month-major indexing for cache locality
- PAR validation uses iterative checks (NaN, dimension, sigma positivity) with early return

## Epic 02: Noise Generation and Opening Tree
- SipHash-1-3 seed derivation: deterministic, communication-free (DEC-017)
- Cholesky packed lower-triangular storage: `data[i*(i+1)/2 + j]` — O(n²/2) memory
- `OpeningTree` stores flat `Vec<f64>` with `(scenarios × stages × entities)` layout
- `OpeningTreeView` borrows the tree for shared-memory scenarios

## Epic 03: InSample Sampling and Integration
- `sample_forward` returns `(usize, &[f64])` — index + borrowed noise slice
- `StochasticContext` owns all components; `tree_view()` returns borrowed view
- `build_stochastic_context` takes `&System` + `u64` seed, calls `validate_par_parameters` internally

## Epic 04: Conformance Tests
- Golden RNG values are fragile across crate versions; test structural + statistical properties
- Statistical tolerance: `|mean| < 0.15`, `|std - 1| < 0.15` over 500+ samples
- Declaration-order invariance: reversed hydro IDs must produce identical results
- Genericity gate as `#[test]` using `std::process::Command` + grep

## Epic 05: Documentation and Phase-Close
- Self-referential grep patterns: never spell out the forbidden pattern literally in docs
- Phase tracker, "Current phase", and "Parallelizable phases" must be updated together
- Documentation-only epics are efficient as single-ticket scopes

## Performance Optimizations (applied during epic-04 boundary)
- Pre-computed entity position maps via `resolve_positions()` eliminate HashMap lookups in hot loop
- Stack-allocated buffers (`[0.0_f64; 64]`) for groups ≤64 entities avoid heap allocation
- Incremental `row_base` offset in Cholesky `transform` avoids redundant `i*(i+1)/2` computation
- Combined effect: eliminated ~720,000 Vec heap allocations per tree generation
