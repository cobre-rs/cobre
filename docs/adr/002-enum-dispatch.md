# ADR-002: Enum Dispatch Over Dynamic Dispatch

**Status:** Accepted
**Date:** 2026-03-02
**Spec reference:** DEC-006

## Context

The SDDP algorithm has multiple parameterization points (risk measure, cut selection
strategy, horizon mode, sampling scheme, stopping rules). Each has a closed set of
variants known at compile time. The dispatch mechanism at these points affects both
performance and extensibility.

## Decision

Use flat enum dispatch (`match` at call sites) for all closed variant sets. Reject
`Box<dyn Trait>` entirely. The only trait using compile-time monomorphization is
`SolverInterface`, which wraps FFI calls and is fixed at build time.

Four dispatch patterns are used:

1. Single active variant, global scope -- flat enum (e.g., `CutSelectionStrategy`)
2. Single active variant, per-stage scope -- flat enum (e.g., `RiskMeasure`)
3. Multiple active variants, simultaneous -- `Vec<EnumVariant>` (e.g., `StoppingRuleSet`)
4. Single active variant, FFI wrapper -- Rust trait with monomorphization (e.g., `SolverInterface`)

## Consequences

- No heap allocation or vtable indirection on the hot path
- Adding a new variant requires modifying the enum and all match arms
- No runtime plugin loading (acceptable for the current variant set)
- See `solver-interface-trait.md` SS5 in cobre-docs for the full rationale
