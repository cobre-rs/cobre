# ADR-004: postcard for MPI Broadcast Serialization

**Status:** Accepted
**Date:** 2026-03-02
**Spec reference:** DEC-002

## Context

The SDDP training loop broadcasts the `System` struct from rank 0 to all worker ranks
at startup. This requires serializing Rust structs into a byte buffer suitable for
`MPI_Bcast`. The serialization format must be compact, fast, and well-maintained.

## Decision

Use `postcard` (serde-based, no-std compatible, varint encoding) for MPI broadcast
serialization. This replaces an earlier consideration of `bincode`, which is
unmaintained. `rkyv` was also considered but rejected due to alignment requirements
that complicate MPI buffer management.

## Consequences

- Compact wire format with varint encoding
- Maintained crate with active development
- Requires `serde::Serialize`/`Deserialize` on all broadcast types
- Not zero-copy (acceptable: broadcast happens once at startup, not on the hot path)
