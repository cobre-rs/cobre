# Cobre — Development Guidelines

## Project Overview

Cobre is a Rust ecosystem for power system optimization. The first solver
vertical is SDDP-based hydrothermal dispatch.

- **Language**: Rust 2024 edition, MSRV 1.86
- **License**: Apache-2.0
- **Workspace**: 11 crates (8 workspace + 3 excluded: `cobre-mcp` stub, `cobre-tui` stub, `cobre-python`)
- **Build**: `cargo build --workspace`
- **Test**: `cargo test --workspace --all-features`
- **Format**: `cargo fmt --all` (CI enforces `--check`)

## Hard Rules

These are non-negotiable. Violations must be fixed before committing.

- `unsafe_code = "forbid"` workspace default — `cobre-solver`, `cobre-comm`, and `cobre-python` override to `allow` for FFI/MPI/PyO3
- `unwrap_used = "deny"` — no `.unwrap()` in library code (ok in tests)
- `clippy::all` and `clippy::pedantic` at `warn` level, zero warnings in CI
- **Never use `Box<dyn Trait>`** — enum dispatch for closed variant sets
- **Never allocate on hot paths** — pre-allocate workspaces, reuse buffers
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
- **`slow-tests` feature** — long-running tests (D-case sweep, FPHA plane-selection, forward-sampler convergence) are gated behind `#[cfg_attr(not(feature = "slow-tests"), ignore = ...)]`. Default `cargo test --workspace` skips them; pass `--features slow-tests` to include them.
- **No plan-structure references in user-facing artifacts** — identifiers such
  as `Epic 09`, `ticket-007`, or `architecture-unification plan` must not
  appear in `CHANGELOG.md`, release notes, `book/`, public rustdoc, or
  comments in shipped code. Plans live in `plans/` (gitignored); public
  artifacts describe behavior, not how the work was organized. Git commit
  messages may reference plan structure — they target git-log readers, not
  release consumers. Existing rustdoc/comment references predating this
  rule are tech debt; clean up opportunistically when touching the
  surrounding code.
  
---

## Architecture Guides (Read When Relevant)

When modifying hot-path code (`forward.rs`, `backward.rs`, `training.rs`,
`simulation/pipeline.rs`, `lower_bound.rs`), read:
→ `.claude/architecture-rules.md`

When applying a stored basis at any call site, read:
→ `crates/cobre-sddp/src/basis_reconstruct.rs` module docs.
`reconstruct_basis` is the single hot-path entry point; never
bypass it.

When changing the MPI basis-cache wire format, read:
→ `crates/cobre-sddp/src/workspace.rs` —
`CapturedBasis::to_broadcast_payload` and
`CapturedBasis::try_from_broadcast_payload` are the sole
owners of the byte layout. Any layout change must update
both methods together; the `broadcast_basis_cache` helper
in `training.rs` only owns the four MPI broadcast calls.

When adding new LP variables, constraints, or entity types, read:
→ `crates/cobre-sddp/src/lp_builder.rs` module docs and `crates/cobre-sddp/src/indexer.rs`

When modifying study setup construction or scenario library building, note that
`setup.rs` is now a directory module (`setup/mod.rs`) with six sub-modules:
→ `setup/params.rs` — `StudyParams`, `ConstructionConfig`, constants
→ `setup/stochastic_pipeline.rs` — `PrepareStochasticResult`, `prepare_stochastic`, helpers
→ `setup/template_postprocess.rs` — `postprocess_templates`
→ `setup/scenario_libraries.rs` — 4 scenario library builder functions
→ `setup/accessors.rs` — 33 accessor methods and context builders
→ `setup/orchestration.rs` — `train`, `simulate`, `build_training_output`, `create_workspace_pool`
The `StudySetup` struct, its two constructors (`new`, `from_broadcast_params`), and three
private helpers remain in `setup/mod.rs`.

When adding new output files, check both CLI and Python write paths:
→ `crates/cobre-cli/src/commands/run.rs` (`write_outputs` function)
→ `crates/cobre-python/src/run.rs` (`run_inner` function)

---

## Key References

| Resource              | Location            | Purpose                                      |
| --------------------- | ------------------- | -------------------------------------------- |
| Software book         | `book/`             | User-facing documentation (mdBook)           |
| Methodology reference | `~/git/cobre-docs/` | Specs, theory, math                          |
| CHANGELOG             | `CHANGELOG.md`      | Per-release feature list                     |
| Design docs           | `docs/design/`      | Future feature designs (not yet implemented) |
