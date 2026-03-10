# Plan: v0.1.0 Release Blockers

Complete the remaining feature gaps and documentation work required for a credible Cobre v0.1.0 release.

## Tech Stack

- Rust 2024 edition (MSRV 1.85)
- rayon (new dependency for cobre-sddp)
- mdBook (documentation sites)
- VHS / asciinema (terminal recordings)

## Epics

| Epic | Name | Tickets | Detail Level |
|------|------|---------|--------------|
| 01 | Rayon Thread Parallelism | 6 | Detailed |
| 02 | Inflow Non-Negativity | 3 | Detailed |
| 03 | ~~4ree Example Case~~ (deferred) | 2 | Outline |
| 04 | Documentation Completion | 6 | Outline |
| 05 | Polish & Recordings | 8 | Outline |

## Progress

| Ticket | Title | Epic | Status | Detail Level | Readiness | Quality | Badge |
|--------|-------|------|--------|--------------|-----------|---------|-------|
| ticket-001 | Create SolverWorkspace struct and workspace pool | epic-01 | completed | Detailed | 0.97 | 1.00 | EXCELLENT |
| ticket-002 | Parallelize forward pass with rayon | epic-01 | completed | Detailed | 1.00 | 0.95 | EXCELLENT |
| ticket-003 | Parallelize backward pass with per-thread cut staging | epic-01 | completed | Detailed | 1.00 | 0.93 | EXCELLENT |
| ticket-004 | Parallelize simulation pipeline | epic-01 | completed | Detailed | 1.00 | 0.88 | ACCEPTABLE |
| ticket-005 | Add --threads CLI flag and thread pool initialization | epic-01 | completed | Detailed | 1.00 | 0.95 | EXCELLENT |
| ticket-006 | Add determinism verification tests | epic-01 | completed | Detailed | 0.96 | 1.00 | EXCELLENT |
| ticket-007 | Add inflow non-negativity penalty method to LP builder | epic-02 | completed | Detailed | 1.00 | 1.00 | EXCELLENT |
| ticket-008 | Wire inflow non-negativity config into training and simulation | epic-02 | completed | Detailed | 0.98 | 0.93 | EXCELLENT |
| ticket-009 | ~~Implement inflow truncation method~~ (deferred — requires external AR eval) | epic-02 | skipped | Detailed | -- | -- | -- |
| ticket-010 | Add inflow non-negativity integration tests | epic-02 | completed | Detailed | 0.96 | 0.97 | EXCELLENT |
| ticket-011 | ~~Translate 4ree case data to Cobre format~~ (deferred) | epic-03 | skipped | Outline | -- | -- | -- |
| ticket-012 | ~~Embed 4ree template and validate end-to-end~~ (deferred) | epic-03 | skipped | Outline | -- | -- | -- |
| ticket-013 | Promote specs/math content into theory pages | epic-04 | pending | Outline | -- | -- | -- |
| ticket-014 | Demote specs section in cobre-docs navigation | epic-04 | pending | Outline | -- | -- | -- |
| ticket-015 | Fix cobre-docs metadata and add mdbook-admonish | epic-04 | pending | Outline | -- | -- | -- |
| ticket-016 | Fix software book stubs | epic-04 | pending | Outline | -- | -- | -- |
| ticket-017 | ~~Add examples section to software book~~ (deferred — depends on Epic 03) | epic-04 | skipped | Outline | -- | -- | -- |
| ticket-026 | Document deferred truncation and statistics invariance | epic-04 | pending | Outline | -- | -- | -- |
| ticket-018 | ~~Generate and calibrate VHS recordings~~ (deferred — depends on Epic 03) | epic-05 | skipped | Outline | -- | -- | -- |
| ticket-019 | Create broken-case validation tape | epic-05 | pending | Outline | -- | -- | -- |
| ticket-020 | ~~Embed recordings in README and book~~ (deferred — depends on ticket-018) | epic-05 | skipped | Outline | -- | -- | -- |
| ticket-021 | Add mdbook-mermaid and update CI | epic-05 | pending | Outline | -- | -- | -- |
| ticket-022 | Update PROJECT-STATUS.md | epic-05 | pending | Outline | -- | -- | -- |
| ticket-023 | Final review pass | epic-05 | pending | Outline | -- | -- | -- |
| ticket-024 | Fix banner color/style loss under mpiexec | epic-05 | pending | Outline | -- | -- | -- |
| ticket-025 | Document deferred HPC optimizations in roadmap | epic-05 | pending | Outline | -- | -- | -- |

## Dependency Graph

```
ticket-001 ──> ticket-002 ──> ticket-005 ──> ticket-006
           ├──> ticket-003 ──┘
           └──> ticket-004 ──┘

ticket-007 ──> ticket-008 ──> ticket-010
ticket-009 (SKIPPED — truncation deferred to post-v0.1.0)

ticket-011, ticket-012 (SKIPPED — 4ree example deferred to post-v0.1.0)
ticket-017 (SKIPPED — depends on Epic 03)
ticket-018, ticket-020 (SKIPPED — depends on Epic 03)

ticket-013 ──> ticket-014
ticket-026 (depends on Epic 02)
ticket-019 (independent — broken-case validation tape)
ticket-022 (depends on Epics 01-02) ──> ticket-023
ticket-019 ────────────────────────────┘
ticket-024 (independent — fix banner color under mpiexec)
ticket-025 (depends on ticket-005, ticket-022 — roadmap HPC optimizations)
```
