# Epic 05: Documentation and Phase-Close

## Goal

Complete the Phase 5 deliverables by writing the software book chapter for cobre-stochastic, updating CONTRIBUTING.md with crate-specific guidelines, running the final infrastructure genericity audit, and updating the phase tracker in CLAUDE.md.

## Scope

- Software book chapter: `book/src/crates/cobre-stochastic.md` with architecture overview, usage examples, and API reference pointers
- CONTRIBUTING.md update: crate-specific build/test instructions for cobre-stochastic
- Final infrastructure genericity audit: `grep -riE 'sddp' crates/cobre-stochastic/`
- Phase tracker update in CLAUDE.md: mark Phase 5 as complete
- Final test count documentation

## Out of Scope

- API documentation beyond doc comments (rustdoc is auto-generated)
- Performance benchmarking documentation
- User-facing tutorials (deferred to post-MVP)

## Tickets

| Ticket | Title | Points | Blocks |
|--------|-------|--------|--------|
| ticket-013 | Write documentation and close phase | 2 | None |

## Dependencies

- **Blocked By**: Epic 04 (all tests must pass before documenting final state)
- **Blocks**: None (final epic in Phase 5)

## Success Criteria

- Book chapter exists at `book/src/crates/cobre-stochastic.md` with non-trivial content
- `grep -riE 'sddp' crates/cobre-stochastic/` returns zero matches
- CLAUDE.md phase tracker shows Phase 5 as complete
- All test counts documented
