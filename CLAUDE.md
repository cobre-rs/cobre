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

See `CONTRIBUTING.md` for build prerequisites and commit message format.

---

## Current State (v0.4.4)

The SDDP solver is fully functional: case loading, stochastic scenario
generation, training, simulation, policy checkpointing, and output writing.
3,600+ tests, including 27 deterministic regression cases (D01–D11, D13–D16,
D19–D30) and 2 cut selection integration tests (D17–D18).

**Implemented:** constant-productivity and FPHA hydro models, evaporation,
cascade coupling, water withdrawal, inflow non-negativity (truncation, penalty,
truncation-with-penalty), multi-segment deficit, generic constraints (20
variable types), NCS stochastic availability, block factors, per-stage
productivity override, per-stage thermal cost override, CVaR risk measure,
PAR(p) estimation (periodic YW, PACF), LP scaling, solver statistics
instrumentation, LP setup optimisation (model persistence, incremental cuts,
sparse cuts), simulation basis warm-start, two-stage cut management pipeline
(strategy-based selection, active cut budget enforcement),
slot-tracked basis reconstruction with capture-time state metadata (CapturedBasis), backward pass work-stealing
parallelism, parallel lower bound evaluation, solver safeguards (12-level retry
escalation with wall-clock budgets), MPI distribution with execution topology
reporting, Python bindings with Arrow zero-copy, CLI with 7 subcommands, policy
warm-start and resume-from-checkpoint with per-stage cut counts, cost
decomposition, per-block operational violations, bidirectional
withdrawal/evaporation slacks, per-plant inflow penalty via cascade, discount
rate, visited state persistence, per-class scenario sampling (Historical,
External, InSample, OutOfSample per entity class), composite ForwardSampler with
ClassSampler dispatch, HistoricalScenarioLibrary and ExternalScenarioLibrary,
HistoricalResiduals noise method, historical window discovery, per-class
external scenario files, same-type correlation enforcement, sub-monthly stage
lag accumulation (StageLagTransition precomputation with frozen-lag semantics),
recent_observations input for mid-season study starts, terminal boundary cuts
(BoundaryPolicy for Cobre-to-Cobre FCF coupling), non-uniform scenario count
support (V3.4 relaxation with padding for DECOMP weekly+monthly studies).
Simulation pipeline bakes per-stage templates end-to-end when training produced
baked templates; the full CapturedBasis metadata propagates to non-root MPI
ranks via a 4-broadcast wire format.

**Known gaps:** GNL thermals, batteries (entity stubs exist, no LP contribution).

---

## Hard Rules

These are non-negotiable. Violations must be fixed before committing.

- `unsafe_code = "forbid"` workspace default — `cobre-solver`, `cobre-comm`, and `cobre-python` override to `allow` for FFI/MPI/PyO3
- `unwrap_used = "deny"` — no `.unwrap()` in library code (ok in tests)
- `clippy::all` and `clippy::pedantic` at `warn` level, zero warnings in CI
- **Never use `Box<dyn Trait>`** — enum dispatch for closed variant sets
- **Never allocate on hot paths** — pre-allocate workspaces, reuse buffers
- **`CapturedBasis` metadata integrity** — `basis_cache` is
  `Vec<Option<CapturedBasis>>` (not bare `Basis`);
  `broadcast_basis_cache` uses a 4-broadcast wire format
  (2 x i32, 2 x f64). Both are invariants established by the
  basis-reconstruction plan; regressing either silently
  degrades warm-start quality without a compile error.
- **`TrainingResult` struct-literal parity** — any new field
  added to `TrainingResult` MUST be listed explicitly at every
  struct literal site in `crates/cobre-cli/src/commands/run.rs`
  and `crates/cobre-python/src/run.rs`. `Option<T>` fields with
  a struct spread silently adopt `None` — grep `TrainingResult {`
  after any struct change.
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
Until they reach zero, use `--max 10` as the threshold. The count must
never increase.

---

## Architecture Guides (Read When Relevant)

When modifying hot-path code (`forward.rs`, `backward.rs`, `training.rs`,
`simulation/pipeline.rs`, `lower_bound.rs`), read:
→ `.claude/architecture-rules.md`

When applying a stored basis at any call site, read:
→ `crates/cobre-sddp/src/basis_reconstruct.rs` module docs.
`reconstruct_basis` is the single hot-path entry point; never
bypass it. The `stored_cut_row_offset` parameter is non-zero
only on the baked-template backward path (pass `0` everywhere
else).

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
