# Cobre — Development Guidelines

## Project Overview

Cobre is a Rust ecosystem for power system optimization. The first solver
vertical is SDDP-based hydrothermal dispatch.

- **Language**: Rust 2024 edition, MSRV 1.88
- **License**: Apache-2.0
- **Workspace**: Cargo workspace members (`cobre-mcp`, `cobre-tui`, `cobre-flow`, `cobre-uc`, `cobre-emt` are reserved stubs) plus the maturin-built `cobre-python` (excluded from the workspace so `cargo test --workspace` does not require a Python interpreter); `ARCHITECTURE.md` owns the full crate map
- **Build**: `cargo build --workspace`
- **Test**: `cargo test --workspace --features "mpi numa shared-memory serde schema slow-tests flatc-conformance test-support"`
- **Format**: `cargo fmt --all` (CI enforces `--check`)

## Hard Rules

These are non-negotiable. Violations must be fixed before committing.

- `unsafe_code = "forbid"` workspace default — `cobre-solver`, `cobre-comm`, and `cobre-python` override to `allow` for FFI/MPI/PyO3; `cobre-sddp` overrides for the `matrixmultiply::dgemm` call its cut-selection kernel needs (isolated in `src/gemm.rs`)
- `unwrap_used = "deny"` — no `.unwrap()` in library code (ok in tests)
- `clippy::all` and `clippy::pedantic` at `warn` level, zero warnings in CI
- **Never use `Box<dyn Trait>`** — enum dispatch for closed variant sets
- **Never allocate on hot paths** — pre-allocate workspaces, reuse buffers
- **Declaration-order invariance** — results must be bit-for-bit identical
  regardless of input entity ordering. Together with run-to-run
  reproducibility (same inputs → bit-for-bit same outputs, across fresh
  solver instances) this defines Cobre determinism. Cross-algorithm
  equivalence is NOT part of the contract: a hot/warm-started solve may
  report a different-but-equally-valid optimal vertex than a cold solve
  (same objective and primals, different duals). Assert reproducibility
  and order-invariance — never hot == cold
  (`crates/cobre-solver/tests/clp_determinism.rs` is the reference harness)
- **Unwired config is reserved, not dead** — several config sections are
  loaded, validated, and schema-exported without yet being consumed (e.g.
  the vertex-based upper-bound-evaluation config `LipschitzConfig.mode`, a
  one-valued enum with no LP consumer). They
  reserve seams for planned features — do not remove unconsumed config in
  a dead-code sweep without owner sign-off
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
  appear in `CHANGELOG.md`, release notes, public rustdoc, or
  comments in shipped code. Plans live in `plans/` (gitignored); public
  artifacts describe behavior, not how the work was organized. Git commit
  messages may reference plan structure — they target git-log readers, not
  release consumers. Existing rustdoc/comment references predating this
  rule are tech debt; clean up opportunistically when touching the
  surrounding code.
- **Comment discipline — default-off** — Code ships with **no comments** unless a
  comment survives the Deletion Test (`.claude/rules/comments.md` §1): delete it;
  if a competent reader of the code alone would then introduce a bug, "simplify"
  something correct into something wrong, or lose a fact that lives outside the
  file, keep the **single clause** that triggers it — otherwise leave it deleted.
  Refactor (rename / extract / introduce a type) and relocate (commit message /
  `rules/*.md` / a test name) **before** commenting. **Never delete or weaken a
  load-bearing correctness contract** — those are pinned to a named regression
  test and a `rules/*.md` entry (`.claude/rules/sddp.md`); tighten an inline copy
  to a pointer, never lose the invariant.
---

## Architecture Guides (Read When Relevant)

SDDP correctness contracts (Benders cut sign, column-bound state pinning, FPHA
average storage, append-only cut pool / slot-identity basis, NCS availability
factors) are codified in `.claude/rules/sddp.md`, which auto-loads when editing
`crates/cobre-sddp/**/*.rs`. Each is a contract, not a style preference — a
plausible deviation produces wrong bounds or rejected warm-starts that still
compile.

Comment & documentation rules for all `.rs` files are codified in `.claude/rules/comments.md`, which auto-loads when editing any `**/*.rs` file. It governs the default-off Deletion Test, the refactor/relocate/hoist gates, the Four Voices, and directives D1–D5 / N1–N6.

Prose documentation integrity rules (scope matrix, the single adaptation, and the six prose-only failure modes) are codified in `.claude/rules/doc-integrity.md`, which auto-loads when editing Markdown files in `CONTRIBUTING.md`, `CHANGELOG.md`, and root-level `*.md`.

When modifying hot-path code (`training/forward/`, `training/backward/`,
`training/training.rs`, `simulation/pipeline.rs`, `training/lower_bound.rs`),
read:
→ `.claude/architecture-rules.md`

When applying a stored basis at any call site, read:
→ `crates/cobre-sddp/src/cut/basis_reconstruct.rs` module docs — the authoritative
statement of the two entry points and when each applies (`reconstruct_basis` on the
frozen hot path; `reconstruct_basis_uniform_basic` on the DCS path). Use the correct
one for the path; never bypass `reconstruct_basis` on the frozen path.

When changing the MPI basis-cache wire format, read:
→ `crates/cobre-sddp/src/workspace/workspace.rs` —
`CapturedBasis::to_broadcast_payload` and
`CapturedBasis::try_from_broadcast_payload` are the sole
owners of the byte layout. Any layout change must update
both methods together; the `broadcast_basis_cache` helper
in `training/training.rs` only owns the four MPI broadcast calls.

When changing the stage-LP builder (`lp/builder/`, `lp/indexer/`, or the `setup/`
code that resolves the builder's inputs), read:
→ `docs/design/lp-builder-contract.md` — the four jobs the builder exists for and
the questions every change to it must answer

When adding new LP variables, constraints, or entity types, read:
→ `crates/cobre-sddp/src/lp/builder/mod.rs` module docs and `crates/cobre-sddp/src/lp/indexer/mod.rs`

When modifying study setup construction or scenario library building, read:
→ `crates/cobre-sddp/src/setup/mod.rs` — `setup/` is a directory module whose
`mod.rs` owns the `StudySetup` struct and its two constructors. The sub-struct
layout and which sub-module owns each piece is mapped in
`.claude/architecture-rules.md` → "StudySetup Sub-Structs".

When adding new output files, check both CLI and Python write paths:
→ `crates/cobre-cli/src/commands/run/outputs.rs` (`write_training_outputs` / `write_simulation_outputs` functions)
→ `crates/cobre-python/src/run.rs` (`run_via_study` / `run_training_phase_py` functions)

When changing schema-bearing `cobre-io` types (fields, `#[derive(JsonSchema)]`
types, or schemars-visible doc comments), regenerate the committed schemas —
CI's `schemas` job diffs `schemas/` against the live export and fails on drift:
→ `cargo build --release --bin cobre && ./target/release/cobre schema export --output-dir schemas`

---

## Key References

| Resource              | Location            | Purpose                                      |
| --------------------- | ------------------- | -------------------------------------------- |
| Workspace map         | `ARCHITECTURE.md`   | Crate responsibilities, dependency boundaries, build-time choices |
| Unified docs site     | `https://docs.cobre-rs.dev/`                     | User-facing documentation (methodology + software) |
| Methodology reference | `~/git/cobre-docs/` | Specs, theory, math                          |
| CHANGELOG             | `CHANGELOG.md`      | Per-release feature list                     |
| Design docs           | `docs/design/`      | Live specs, decision records & proposals — `docs/design/README.md` is the status index |
