# Epic 05: Polish and Recordings

## Goal

Generate actual terminal recordings (VHS/asciinema), embed them in the README and software book, create a broken-case validation tape, add mdbook preprocessors, and perform a final review pass.

## Scope

- Generate VHS recordings with calibrated timings
- Create broken-case validation tape
- Embed recordings in README.md and book pages
- Add mdbook-mermaid preprocessor
- Update PROJECT-STATUS.md
- Final cross-link and content review
- Fix CLI banner color loss under mpiexec
- Document deferred HPC optimizations in roadmap

## Out of Scope

- CI automation for recording generation
- Automated performance benchmarks in recordings
- Multi-rank MPI demo recordings

## Tickets

| ID | Title | Effort | Dependencies |
|----|-------|--------|--------------|
| ticket-018 | Generate and calibrate VHS recordings | 2 pts | Epic 03 |
| ticket-019 | Create broken-case validation tape | 2 pts | None |
| ticket-020 | Embed recordings in README and book | 2 pts | ticket-018, ticket-019 |
| ticket-021 | Add mdbook-mermaid and update CI | 1 pts | None |
| ticket-022 | Update PROJECT-STATUS.md | 1 pts | Epic 01, Epic 02, Epic 03 |
| ticket-023 | Final review pass | 3 pts | All previous tickets |
| ticket-024 | Fix banner color/style loss under mpiexec | 1 pts | None |
| ticket-025 | Document deferred HPC optimizations in roadmap | 1 pts | ticket-005, ticket-022 |
