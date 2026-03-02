# ADR-003: Compile-Time Solver Selection

**Status:** Accepted
**Date:** 2026-03-02
**Spec reference:** DEC-005

## Context

The LP solver interface wraps foreign C libraries (HiGHS, CLP). Unlike the algorithm
variant traits, the solver backend is fixed for the entire lifetime of a binary -- you
don't switch solvers mid-execution. Runtime polymorphism would add vtable overhead on
every LP solve call (millions per training run).

## Decision

Use Cargo feature flags for compile-time solver selection. Exactly one solver backend
is active per binary. The `SolverInterface` trait uses compile-time monomorphization
(generic `<S: SolverInterface>` throughout `cobre-sddp`), not enum dispatch or dynamic
dispatch.

Feature flags: `highs` (default), `clp`.

## Consequences

- Zero-overhead solver calls -- the compiler monomorphizes all solver interactions
- Cannot switch solver at runtime (acceptable: solver choice is a deployment decision)
- Two separate test binaries needed (one per solver) for full coverage
- HiGHS is the primary backend; CLP is the secondary reference implementation
