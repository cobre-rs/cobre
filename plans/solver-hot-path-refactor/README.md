# Solver Hot-Path Interface Refactor

Refactor the `cobre-solver` trait interface and HiGHS backend to eliminate three
categories of unnecessary work on the SDDP training loop hot path.

## Tech Stack

- Rust 2024 edition, MSRV 1.85
- Primary crates: `cobre-solver`, `cobre-sddp`
- Testing: `cargo test --workspace --all-features`

## Epics

| Epic | Name | Tickets | Detail Level |
|------|------|---------|--------------|
| epic-01 | i32 CSC Indices | 3 | Detailed |
| epic-02 | SolutionView | 4 | Detailed |
| epic-03 | RawBasis | 3 | Refined |

## Progress

| Ticket | Title | Epic | Status | Detail Level | Readiness | Quality | Badge |
|--------|-------|------|--------|--------------|-----------|---------|-------|
| ticket-001 | Change StageTemplate and RowBatch index types to i32 | epic-01 | completed | Detailed | 1.00 | 0.73 | BELOW GATE |
| ticket-002 | Simplify HighsSolver load_model and add_rows | epic-01 | completed | Detailed | 1.00 | 0.95 | EXCELLENT |
| ticket-003 | Update cobre-sddp template and row batch construction | epic-01 | completed | Detailed | 0.98 | 0.93 | EXCELLENT |
| ticket-004 | Add SolutionView type and trait methods | epic-02 | completed | Detailed | 1.00 | 1.00 | EXCELLENT |
| ticket-005 | Implement solve_view in HighsSolver | epic-02 | completed | Detailed | 1.00 | 0.88 | ACCEPTABLE |
| ticket-006 | Conformance tests for SolutionView equivalence | epic-02 | completed | Detailed | 0.97 | 1.00 | EXCELLENT |
| ticket-007 | Migrate cobre-sddp hot path to solve_view | epic-02 | completed | Detailed | 0.96 | 0.83 | ACCEPTABLE |
| ticket-008 | Add RawBasis type and trait methods | epic-03 | completed | Refined | 1.00 | 0.88 | ACCEPTABLE |
| ticket-009 | Implement get_raw_basis and solve_with_raw_basis_view in HighsSolver | epic-03 | completed | Refined | 1.00 | 0.95 | EXCELLENT |
| ticket-010 | Conformance tests for RawBasis round-trip | epic-03 | completed | Refined | 1.00 | 1.00 | EXCELLENT |

## Dependency Graph

```
ticket-001 --> ticket-002 --> ticket-003

ticket-004 --> ticket-005 --> ticket-006
                    |              |
                    +--> ticket-007 <--+ (soft)
                    |
                    v
              ticket-008 --> ticket-009 --> ticket-010
```

Epic 01 and Epic 02 are independent and can execute in parallel.
Epic 03 depends on Epic 02 (for `SolutionView` type).

## Reference

- Analysis document: `/home/rogerio/git/cobre/solver-hot-path-refactor.md`
- Master plan: `plans/solver-hot-path-refactor/00-master-plan.md`
