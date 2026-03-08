# ADR-007: Raw i32 Basis Status Codes

**Status:** Accepted
**Date:** 2026-03-08
**Spec reference:** DEC-002 (compile-time solver monomorphization)

## Context

The original `SolverInterface` design included a `BasisStatus` enum with four variants
(`Basic`, `Lower`, `Upper`, `SuperBasic`) used for simplex basis representation. Each
`get_basis` call required a per-element enum translation from the solver-native integer
codes, and each `solve_with_basis` call required the reverse translation. On the SDDP
hot path (~6M LP solves per training run), this round-trip conversion added measurable
overhead and complexity without providing any value: the calling algorithm never inspects
or modifies individual basis status codes — it treats the basis as an opaque token
extracted from one solve and injected into the next.

## Decision

Store basis status codes as raw solver-native `i32` values in the `Basis` struct. The
`BasisStatus` enum is removed entirely. `get_basis` writes `i32` status codes directly
into caller-provided buffers via FFI memcpy. `solve_with_basis` injects them back via
`copy_from_slice` — a memcpy, not a per-element conversion.

The `Basis` struct has two fields:

- `col_status: Vec<i32>` — one status code per LP column
- `row_status: Vec<i32>` — one status code per LP row

Status code values are solver-specific (HiGHS uses 0=Basic, 1=Lower, 2=Upper,
3=SuperBasic, 4=NonBasic) and opaque to the calling algorithm.

## Consequences

- Zero per-element conversion overhead on the hot path
- `Basis` is a simple data buffer with no enum variant matching
- Basis is not portable across solver backends (HiGHS codes differ from CLP codes);
  this is acceptable because a binary uses exactly one solver backend (DEC-002)
- The calling algorithm cannot inspect individual column/row statuses; if future
  features require this, a translation function can be added as a cold-path utility
- Dimension mismatch handling (when new cut rows are added after a basis was saved)
  fills extra row slots with the solver-native "Basic" code (HiGHS: 1)
