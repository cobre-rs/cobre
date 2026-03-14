# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the Cobre project.
ADRs capture significant decisions that affect the codebase, along with their context
and consequences.

## Relationship to the Decision Log

The full specification corpus in [cobre-docs](https://github.com/cobre-rs/cobre-docs)
maintains a detailed [Decision Log](https://cobre-rs.github.io/cobre-docs/specs/overview/decision-log.html)
(DEC-001 through DEC-017) that tracks cross-cutting architectural decisions across
the specification files. That log is the authoritative source for spec-level decisions.

The ADRs in this directory serve a different purpose: they capture implementation-level
decisions made during development that aren't covered by the spec corpus. When a
decision originates from the spec's Decision Log, the ADR references the corresponding
DEC-NNN identifier.

## Format

Each ADR follows this template:

```markdown
# ADR-NNN: Title

**Status:** Proposed | Accepted | Deprecated | Superseded by ADR-NNN
**Date:** YYYY-MM-DD
**Spec reference:** DEC-NNN (if applicable)

## Context

What is the issue that we're seeing that motivates this decision?

## Decision

What is the change that we're proposing and/or doing?

## Consequences

What becomes easier or more difficult because of this change?
```

## Index

| ADR                                           | Status   | Title                                                          |
| --------------------------------------------- | -------- | -------------------------------------------------------------- |
| [001](001-documentation-split.md)             | Accepted | Documentation split between cobre and cobre-docs               |
| [002](002-enum-dispatch.md)                   | Accepted | Enum dispatch over dynamic dispatch for closed variant sets    |
| [003](003-compile-time-solver.md)             | Accepted | Compile-time solver selection via Cargo feature flags          |
| [004](004-postcard-mpi-serialization.md)      | Accepted | postcard for MPI broadcast serialization                       |
| [005](005-flatbuffers-policy.md)              | Accepted | FlatBuffers for policy data persistence                        |
| [006](006-soa-bound-patching.md)              | Accepted | SoA bound patching for solver hot path                         |
| [007](007-raw-basis-status-codes.md)          | Accepted | Raw i32 basis status codes for zero-copy warm-start            |
| [008](008-user-supplied-opening-tree.md)      | Accepted | User-supplied opening tree via Parquet file                    |
| [009](009-stochastic-artifact-export.md)      | Accepted | Stochastic artifact export to output/stochastic/               |
| [010](010-complete-tree-work-distribution.md) | Accepted | Complete tree work distribution with stage-by-stage allgatherv |
| [011](011-per-stage-warm-start-cuts.md)       | Accepted | Per-stage warm-start counts and terminal-stage boundary cuts   |
