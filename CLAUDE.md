# Cobre — Development Guidelines

## Project Overview

Cobre is a Rust ecosystem for power system optimization. The first solver
vertical is SDDP-based hydrothermal dispatch.

- **Language**: Rust 2024 edition, MSRV 1.86
- **License**: Apache-2.0
- **Workspace**: 11 crates (10 workspace + 1 external `ferrompi`)
- **Build**: `cargo build --workspace`
- **Test**: `cargo test --workspace --all-features`
- **Format**: `cargo fmt --all` (CI enforces `--check`)

See `CONTRIBUTING.md` for build prerequisites and commit message format.

---

## Current State (v0.1.11)

The SDDP solver is fully functional: case loading, stochastic scenario
generation, training, simulation, policy checkpointing, and output writing.
2,747 tests, including 16 deterministic regression cases (D01–D16).

**Implemented:** constant-productivity and FPHA hydro models, evaporation,
cascade coupling, water withdrawal, inflow non-negativity, multi-segment
deficit, generic constraints (20 variable types), NCS stochastic availability,
block factors, per-stage productivity override, CVaR risk measure, PAR(p)
estimation (periodic YW, PACF), LP scaling, solver statistics instrumentation,
LP setup optimisation (model persistence, incremental cuts, sparse cuts),
simulation basis warm-start, MPI distribution, Python bindings with Arrow
zero-copy, CLI with 6 subcommands.

**Known gaps:** GNL thermals, batteries (entity stubs exist, no LP contribution).

---

## Hard Rules

These are non-negotiable. Violations must be fixed before committing.

- `unsafe_code = "forbid"` — no unsafe anywhere
- `unwrap_used = "deny"` — no `.unwrap()` in library code (ok in tests)
- `clippy::all` and `clippy::pedantic` at `warn` level, zero warnings in CI
- **Never use `Box<dyn Trait>`** — enum dispatch for closed variant sets
- **Never allocate on hot paths** — pre-allocate workspaces, reuse buffers
- **Never add `#[allow(clippy::too_many_arguments)]`** without first absorbing
  the parameter into an existing context struct. See `.claude/architecture-rules.md`
- **Declaration-order invariance** — results must be bit-for-bit identical
  regardless of input entity ordering
- **Infrastructure crate genericity** — `cobre-core`, `cobre-io`, `cobre-solver`,
  `cobre-stochastic`, `cobre-comm` must contain zero algorithm-specific references
  (no "sddp", "SDDP", "Benders" in types, functions, or doc comments)
- **Python parity** — every output file the CLI writes must also be written by
  the Python bindings in `cobre-python`. When adding a new output, wire it in both.
- Do not use `bincode` — use `postcard` for MPI, `FlatBuffers` for policy
- Do not commit secrets, `.env` files, or credentials
- Do not force-push to `main`

---

## Pre-Commit Check

Before committing changes to `crates/`, run:

```
python3 scripts/check_suppressions.py --max 0
```

This checks that no `#[allow(clippy::too_many_arguments)]` exists in production
code (code before `#[cfg(test)]`). If the check fails, absorb the parameter
into an existing context struct instead of adding a suppression. Read
`.claude/architecture-rules.md` for the specific struct patterns.

The current codebase has legacy suppressions that are being worked down.
Until they reach zero, use `--max 15` as the threshold. The count must
never increase.

---

## Architecture Guides (Read When Relevant)

When modifying hot-path code (`forward.rs`, `backward.rs`, `training.rs`,
`simulation/pipeline.rs`, `lower_bound.rs`), read:
→ `.claude/architecture-rules.md`

When adding new LP variables, constraints, or entity types, read:
→ `crates/cobre-sddp/src/lp_builder.rs` module docs and `crates/cobre-sddp/src/indexer.rs`

When adding new output files, check both CLI and Python write paths:
→ `crates/cobre-cli/src/commands/run.rs` (`write_outputs` function)
→ `crates/cobre-python/src/run.rs` (`run_inner` function)

---

## Key References

| Resource | Location | Purpose |
|----------|----------|---------|
| Software book | `book/` | User-facing documentation (mdBook) |
| Methodology reference | `~/git/cobre-docs/` | Specs, theory, math |
| CHANGELOG | `CHANGELOG.md` | Per-release feature list |
| Design docs | `docs/design/` | Future feature designs (not yet implemented) |
