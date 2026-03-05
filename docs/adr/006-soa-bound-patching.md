# ADR-006: SoA Bound Patching for Solver Hot Path

**Status:** Accepted
**Date:** 2026-03-05
**Spec reference:** DEC-019

## Context

The `set_row_bounds` and `set_col_bounds` methods on `SolverInterface` originally
accepted a single slice of tuples: `&[(usize, f64, f64)]`. Callers naturally hold
bounds as three separate arrays (indices, lower, upper) because the HiGHS C API
expects separate pointers. The AoS signature forced a zip-then-unzip round-trip on
the hottest path in iterative LP solving: at production scale the bounds-patching
path is called 11,520 times per iteration (M scenarios × T stages), making
unnecessary allocations and copies measurable.

Three scratch buffers (`scratch_lower`, `scratch_upper`, `scratch_col_starts_i32`)
were maintained inside `HighsSolver` solely to support the unzip step, adding
heap state that must be managed across calls.

## Decision

Change `set_row_bounds` and `set_col_bounds` to accept three parallel slices:
`indices: &[usize]`, `lower: &[f64]`, `upper: &[f64]`. Remove the three scratch
buffers from `HighsSolver`. All call sites pass their naturally separate arrays
directly.

## Consequences

- Eliminated the zip-then-unzip round-trip on the bounds-patching hot path
- Removed three scratch buffers from the solver struct, simplifying its lifecycle
- Rust API now aligns directly with the HiGHS C API calling convention
- FFI boundary is simpler: no intermediate owned collections at the call site
- Callers that previously constructed tuples must be updated to pass parallel slices
