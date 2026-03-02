# ADR-005: FlatBuffers for Policy Data Persistence

**Status:** Accepted
**Date:** 2026-03-02
**Spec reference:** DEC-003

## Context

The SDDP solver produces a Future Cost Function (FCF) consisting of Benders cuts
that must be persisted for checkpointing and simulation replay. The FCF can contain
millions of cuts, each with coefficient vectors proportional to the number of state
variables. The persistence format must support zero-copy reads for fast checkpoint
resume and SIMD-friendly memory layout.

## Decision

Use FlatBuffers for policy data persistence (cuts, states, vertices, checkpoint data).
FlatBuffers provides zero-copy deserialization and allows SIMD-aligned access to the
coefficient arrays.

## Consequences

- Zero-copy deserialization for fast checkpoint resume
- SIMD-friendly flat memory layout for coefficient arrays
- Requires FlatBuffers schema files (`.fbs`) and code generation step
- Schema evolution supported via FlatBuffers' forward/backward compatibility rules
- Not human-readable (acceptable: policy data is machine-consumed)
