# Cobre -- Development Guidelines

This file is loaded by Claude Code when working in this repository. It captures
the conventions, source-of-truth references, and implementation ordering that
govern all development on the Cobre ecosystem.

---

## Project Overview

Cobre is an ecosystem of Rust crates for power system analysis and optimization.
The first solver vertical is SDDP-based hydrothermal dispatch -- a production-grade
distributed solver for long-term energy planning.

- **Language**: Rust 2024 edition, MSRV 1.85
- **License**: Apache-2.0
- **Workspace**: 11 crates (10 workspace + 1 external `ferrompi`)
- **Build**: `cargo build --workspace`
- **Test**: `cargo test --workspace --all-features`

See `CONTRIBUTING.md` for build prerequisites, project structure, commit message
format, and crate-specific coding guidelines.

---

## Two Repositories, Two Roles

Development spans two repositories with distinct roles:

| Repository     | Location                  | Role                                  | Contains                                                                                                               |
| -------------- | ------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **cobre-docs** | `~/git/cobre-docs`        | **What we want** -- the specification | 74 spec files defining architecture, algorithms, data model, HPC layer, testing contracts, and cross-cutting decisions |
| **cobre**      | `~/git/cobre` (this repo) | **What exists** -- the implementation | Rust source code, software book, ADRs, CI, and everything that ships                                                   |

**cobre-docs is the source of truth for design.** Every implementation decision must
trace back to a spec section. When coding a feature, read the relevant spec first.
If the spec is silent on a detail, that is a spec gap -- escalate it, do not guess.

**cobre is the source of truth for what has been built.** The code, tests, and
software book in this repo reflect the current state of the implementation. When
a spec describes a feature that hasn't been implemented yet, the code is the
authority on what actually works today.

### Implementation progress tracking

As each phase is implemented, update the "Current phase" section in this file to
reflect what has been completed and what is next. This gives any agent working in
this repo an accurate picture of implementation status.

The implementation phases below track what has been **specified** (in cobre-docs)
versus what has been **implemented** (in cobre). At any point in time:

- Specs in cobre-docs describe the **complete target** for the minimal viable solver
- Code in cobre represents the **partially completed** implementation
- The gap between them is the remaining work

When a phase is complete (all types, traits, and tests from the spec reading list
are implemented and passing), mark it as done in the phase tracker below and
advance the "Current phase" pointer.

### Key paths in cobre-docs

| Category              | Path                                            | Purpose                                              |
| --------------------- | ----------------------------------------------- | ---------------------------------------------------- |
| Implementation order  | `src/specs/overview/implementation-ordering.md` | 8-phase build sequence, per-phase spec reading lists |
| Spec gap inventory    | `src/specs/overview/spec-gap-inventory.md`      | 39 resolved gaps with resolution paths               |
| Decision log          | `src/specs/overview/decision-log.md`            | DEC-001 through DEC-017, cross-cutting decisions     |
| Cross-reference index | `src/specs/cross-reference-index.md`            | Spec-to-crate mappings, dependency order             |
| Design principles     | `src/specs/overview/design-principles.md`       | Format selection, declaration-order invariance       |
| Architecture specs    | `src/specs/architecture/`                       | Trait specs, testing specs, impl specs               |
| HPC specs             | `src/specs/hpc/`                                | Communication, parallelism, memory, backends         |
| Math specs            | `src/specs/math/`                               | Algorithm formulations, LP structure                 |
| Data model specs      | `src/specs/data-model/`                         | Input/output schemas, internal structures            |

### Spec consultation workflow

Before implementing any module:

1. **Read the phase entry** in `implementation-ordering.md` section 5 for the crate
2. **Read every spec** in that phase's reading list
3. **Check the decision log** for DEC-NNN entries that affect the crate
4. **Check the spec gap inventory** for gaps in the crate's specs (all 39 are resolved
   but the resolutions document important design details)
5. **Read the cross-reference index** sections 1-2 for the crate's spec mapping

When you encounter a spec reference like "SS4.2" or "§1.3", these are section
identifiers in the spec files:

- **SS** prefix = architecture spec sections (e.g., SS4 in `training-loop.md`)
- **§** prefix = HPC spec sections (e.g., §1 in `communicator-trait.md`)
- Plain numbered sections = overview/planning specs (e.g., `## 3.` in `implementation-ordering.md`)

---

## Implementation Ordering

The minimal viable SDDP solver is built in 8 phases. Each phase produces a testable
intermediate. Dependencies flow bottom-up:

```
Phase 1 (core) ──────┬──> Phase 2 (io) ──────────────────────────────────────┐
                      │                                                      │
                      ├──> Phase 3 (ferrompi + solver) ──> Phase 4 (comm) ──┐│
                      │                                                     ││
                      └──> Phase 5 (stochastic) ──────────────────────────┐ ││
                                                                          v vv
                                                                     Phase 6 (sddp training)
                                                                          │
                                                                          v
                                                                     Phase 7 (simulation + output)
                                                                          │
                                                                          v
                                                                     Phase 8 (cli)
```

### Phase summary

| Phase | Crate(s)                    | What becomes testable                                                        |
| ----- | --------------------------- | ---------------------------------------------------------------------------- |
| 1     | `cobre-core`                | Entity model, registries, topology validation, penalty resolution            |
| 2     | `cobre-io`                  | Case directory loading, 5-layer validation pipeline, JSON/Parquet parsing    |
| 3     | `ferrompi` + `cobre-solver` | MPI bindings, LP solver abstraction, HiGHS backend, warm-starting            |
| 4     | `cobre-comm`                | Communicator trait, MPI backend, local backend, backend selection            |
| 5     | `cobre-stochastic`          | PAR(p) preprocessing, noise generation, opening trees, InSample sampling     |
| 6     | `cobre-sddp`                | Training loop, forward/backward pass, cut management, convergence monitoring |
| 7     | `cobre-sddp` + `cobre-io`   | Simulation pipeline, Parquet output, FlatBuffers FCF, checkpointing          |
| 8     | `cobre-cli`                 | Execution lifecycle, config resolution, exit codes                           |

### Phase tracker

<!-- UPDATE THIS TABLE as phases are completed -->

| Phase | Status      | Notes                                                                                                                                                                     |
| ----- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | complete    | Entity model, System, topology, validation, penalty resolution -- 177 tests (137 unit + 7 integration + 33 doc)                                                           |
| 2     | complete    | load_case pipeline, 5-layer validation, 33-file JSON/Parquet loading, penalty/bound resolution -- 622 tests                                                               |
| 3     | complete    | LP solver abstraction, HiGHS backend, 30 conformance tests, ferrompi audit -- 67 tests (35 unit + 30 integration + 2 doc)                                                 |
| 4     | complete    | Communicator trait, LocalBackend, FerrompiBackend, factory, conformance tests -- 90 tests (54 unit + 28 integration + 8 doc)                                              |
| 5     | complete    | PAR(p) preprocessing, SipHash seed derivation, Cholesky correlation, opening tree, InSample sampling -- 125 tests (105 unit + 5 conformance + 4 reproducibility + 11 doc) |
| 6     | not started | Blocked by Phases 1-5                                                                                                                                                     |
| 7     | not started | Blocked by Phase 6                                                                                                                                                        |
| 8     | not started | Blocked by Phase 7                                                                                                                                                        |

### Current phase

**Phase 5: cobre-stochastic -- Complete.** The stochastic process model layer is implemented: PAR(p) coefficient preprocessing with original-unit conversion and stage-major flat array layout; deterministic noise generation via SipHash-1-3 seed derivation (DEC-017) and Pcg64 RNG; hand-rolled Cholesky decomposition with packed lower-triangular storage and pre-computed entity positions; opening scenario tree with sentinel offset arrays; InSample scenario selection via `sample_forward`; and `StochasticContext` as the single integration entry point. Neutral language enforced throughout -- zero SDDP references in source. All 125 tests pass (105 unit + 5 conformance + 4 reproducibility + 11 doc). Next candidate is Phase 6 (`cobre-sddp`).

### Parallelizable phases

Phase 6 (`cobre-sddp`) is the next candidate. Phases 1–5 are all complete.

### Per-phase spec reading lists

For the complete spec reading list for each phase, see
`~/git/cobre-docs/src/specs/overview/implementation-ordering.md` section 5.

Quick reference for Phase 1 (complete):

- `src/specs/overview/notation-conventions.md`
- `src/specs/overview/design-principles.md`
- `src/specs/math/hydro-production-models.md`
- `src/specs/data-model/input-system-entities.md`
- `src/specs/data-model/input-hydro-extensions.md`
- `src/specs/data-model/penalty-system.md`
- `src/specs/data-model/internal-structures.md`
- `src/specs/math/system-elements.md`
- `src/specs/math/equipment-formulations.md`

---

## Trait Variant Selection (Minimal Viable)

Each trait abstraction has one variant for the minimal viable solver:

| Trait                  | Variant           | Rationale                                           |
| ---------------------- | ----------------- | --------------------------------------------------- |
| `RiskMeasure`          | Expectation       | Risk-neutral baseline                               |
| `CutSelectionStrategy` | Level-1           | Preserves convergence while controlling pool growth |
| `HorizonMode`          | Finite            | Linear stage chain, zero terminal value             |
| `SamplingScheme`       | InSample          | Same openings for forward and backward pass         |
| `StoppingRule`         | All 5 (composite) | Full stopping rule set via `StoppingRuleSet`        |
| `SolverInterface`      | HiGHS             | Open-source LP solver with warm-start support       |
| `Communicator`         | MPI (ferrompi)    | Production distributed backend                      |

Deferred variants are documented in `implementation-ordering.md` section 6.

---

## System Element Scope

Four element types are fully modeled. Three are NO-OP stubs (type exists in registry
but contributes zero LP variables/constraints):

| Element          | Status | Notes                                                    |
| ---------------- | ------ | -------------------------------------------------------- |
| Bus              | Full   | Power balance constraint per bus per block               |
| Line             | Full   | Flow variable with MW capacity bounds                    |
| Thermal          | Full   | Generation variable with MW bounds and cost              |
| Hydro            | Full   | Reservoir, turbine, spillage. Constant productivity only |
| Contract         | Stub   | NO-OP -- entity exists, no LP contribution               |
| Pumping Station  | Stub   | NO-OP -- entity exists, no LP contribution               |
| Non-Controllable | Stub   | NO-OP -- entity exists, no LP contribution               |

Stubs must exist from Phase 1 so that LP construction code iterates over all element
types from the start (avoids first-time integration surprises).

---

## Coding Standards

### Hard rules (enforced by workspace lints)

- `unsafe_code = "forbid"` -- no unsafe at all
- `unwrap_used = "deny"` -- no `.unwrap()` in library code (ok in tests)
- `clippy::all` and `clippy::pedantic` at `warn` level
- `missing_docs = "warn"` -- all public items should have doc comments

### Performance rules

- **No allocation on hot paths.** The SDDP training loop solves millions of LPs.
  Pre-allocate workspaces; reuse buffers.
- **Cache-friendly layout.** Group LP variables that are accessed together (e.g., all
  storage levels before all inflow lags) for contiguous memory access.
- **SIMD-friendly arrays.** Prefer flat `Vec<f64>` over nested structs on hot paths.
- **Compile-time dispatch for FFI.** The `SolverInterface` trait uses monomorphization
  (not `dyn`) because it wraps FFI calls. All other traits use enum dispatch.

### Dispatch patterns

The Cobre architecture uses four dispatch patterns. **Never use `Box<dyn Trait>`** --
closed variant sets always prefer enum dispatch.

| Pattern                                           | Mechanism                                 | Examples                                                |
| ------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------- |
| Single active variant, global scope               | Flat enum, `match`                        | `CutSelectionStrategy`, `HorizonMode`, `SamplingScheme` |
| Single active variant, per-stage scope            | Flat enum, `match`                        | `RiskMeasure`                                           |
| Multiple active variants, simultaneous evaluation | `Vec<EnumVariant>` + composition struct   | `StoppingRuleSet`                                       |
| Single active variant, fixed at build time, FFI   | Rust trait, compile-time monomorphization | `SolverInterface`                                       |

### Error handling

- **Pure query methods** (`should_run`, `select`, `evaluate`): never return `Result`.
  They rely on upstream validation guarantees.
- **FFI-wrapping methods** (`solve`, `solve_with_basis`): return `Result` with split
  Ok/Err postcondition tables as documented in the trait specs.
- **Err recovery**: the **caller** calls `reset()` before reusing an instance after error.

### Serialization

- **MPI broadcast**: use `postcard` (not `bincode` -- it is unmaintained)
- **Policy persistence** (FCF cuts): use `FlatBuffers`
- **Input data**: JSON + Parquet
- **Output data**: Hive-partitioned Parquet

### Declaration-order invariance

Results must be **bit-for-bit identical** regardless of input entity ordering. Entity
collections are always stored in canonical ID-sorted order. Every function that
processes entity collections must be tested with reordered input.

### Infrastructure crate genericity (ENFORCED)

The infrastructure crates (`cobre-core`, `cobre-io`, `cobre-solver`, `cobre-stochastic`,
`cobre-comm`) are **deliberately generic** and must contain **zero algorithm-specific
references** (no SDDP, no algorithm names in function/struct/type names, docs, or
comments). This is a first-class requirement from the ecosystem design (see
`cobre-docs/src/specs/overview/ecosystem-vision.md` §6).

**Rules:**

- No function, struct, enum, or type may include "sddp", "SDDP", or any other
  algorithm name in infrastructure crates
- Doc comments must use generic language ("the calling algorithm", "iterative LP
  solving", "optimization algorithms") instead of algorithm-specific references
- Test code may mention algorithms only in comments explaining the fixture's origin,
  never in function or variable names
- Application crates (`cobre-cli`, `cobre-mcp`, `cobre-tui`, `cobre-python`) and the
  solver vertical (`cobre-sddp`) **may** reference SDDP freely

**Quality gate:** At the end of each implementation phase, all modified infrastructure
crate files must pass `grep -riE 'sddp' crates/<crate>/src/` with zero matches. This
check is part of the plan completion protocol.

---

## Key Architectural Decisions

These are from the decision log (`DEC-001` through `DEC-017` in cobre-docs). The most
implementation-critical ones:

| DEC     | Decision                                                               |
| ------- | ---------------------------------------------------------------------- |
| DEC-001 | Enum dispatch over `Box<dyn Trait>` for closed variant sets            |
| DEC-002 | Compile-time monomorphization for `SolverInterface` (FFI wrapper)      |
| DEC-003 | `postcard` for MPI serialization (not `bincode`)                       |
| DEC-004 | FlatBuffers for FCF policy persistence                                 |
| DEC-005 | Five-layer validation pipeline in `cobre-io`                           |
| DEC-006 | Three-tier penalty resolution (global/entity/stage)                    |
| DEC-009 | Per-stage `allgatherv` in backward pass for cut synchronization        |
| DEC-016 | Deferred parallel cut selection with `allgatherv` for DeactivationSets |
| DEC-017 | Communication-free parallel noise via deterministic SipHash-1-3 seeds  |

Always check the full decision log before making architectural choices.

---

## Documentation Sites

| Site                  | Repository        | Content                               | When to update                                 |
| --------------------- | ----------------- | ------------------------------------- | ---------------------------------------------- |
| Software book         | `cobre/book/`     | Installation, guides, crate overviews | During/after implementation                    |
| Methodology reference | `cobre-docs/`     | Specs, theory, math, formats          | Before implementation (specs define contracts) |
| ADRs                  | `cobre/docs/adr/` | Architecture Decision Records         | When cross-cutting decisions are made          |

The software book in `book/` uses mdBook. It contains user-facing documentation that
describes what the software does. Theory pages in cobre-docs are written **ahead** of
implementation (they define the contract); software pages in `book/` are written
**during or after** implementation (they describe what exists).

---

## Agent Delegation

When delegating implementation tasks to specialist agents, use these mappings:

| Crate / Task                      | Agent                               |
| --------------------------------- | ----------------------------------- |
| `cobre-core`, `cobre-sddp`        | `hpc-rust-developer`                |
| `cobre-solver` (FFI, HiGHS, CLP)  | `hpc-rust-developer`                |
| `cobre-comm`, `ferrompi` (MPI)    | `hpc-rust-developer`                |
| `cobre-stochastic` (scenarios)    | `hpc-rust-developer`                |
| `cobre-io` (JSON, Parquet)        | `hpc-rust-developer`                |
| `cobre-cli` (clap, lifecycle)     | `hpc-rust-developer`                |
| `cobre-python` (PyO3)             | `hpc-rust-developer`                |
| `cobre-tui` (ratatui)             | `hpc-rust-developer`                |
| `cobre-mcp` (MCP server)          | `hpc-rust-developer`                |
| Parallel algorithm design         | `hpc-parallel-computing-specialist` |
| SDDP algorithm questions          | `sddp-specialist`                   |
| Data model / serialization format | `data-model-format-specialist`      |
| Test planning                     | `monorepo-test-planner`             |
| Test implementation               | `monorepo-test-developer`           |
| Documentation (book pages)        | `open-source-documentation-writer`  |

### Agent dispatch protocol

When dispatching any implementation agent:

1. **Always provide the spec file path** from `~/git/cobre-docs/` -- the agent reads
   the spec before writing code
2. **Tell the agent what already exists** -- point to the current crate source files
   in this repo so it builds on existing work, not from scratch
3. **Include the phase context** -- tell the agent which implementation phase this
   work belongs to, so it knows what dependencies are available
4. **Specify the target crate and module** -- e.g., "implement in `crates/cobre-core/src/entities/hydro.rs`"

Example dispatch:

```
Implement the Hydro entity type in cobre-core.

Spec: ~/git/cobre-docs/src/specs/data-model/input-system-entities.md (SS3: Hydro Plants)
Also read: ~/git/cobre-docs/src/specs/data-model/input-hydro-extensions.md
Also read: ~/git/cobre-docs/src/specs/math/hydro-production-models.md

Current state: crates/cobre-core/src/lib.rs is an empty stub.
Phase: 1 (cobre-core foundation)
Target: crates/cobre-core/src/entities/hydro.rs (new file)
```

---

## Do NOT

- Implement anything without first reading the relevant spec in cobre-docs
- Use `Box<dyn Trait>` -- prefer enum dispatch for closed variant sets
- Use `unsafe` -- it is `forbid` at workspace level
- Use `.unwrap()` in library code -- it is `deny` at workspace level
- Use `bincode` -- use `postcard` for MPI, `FlatBuffers` for policy
- Allocate on the hot path inside the training loop
- Break declaration-order invariance (results must be identical regardless of input ordering)
- Reference SDDP (or any algorithm name) in infrastructure crate code, types, or docs
- Make architectural decisions not specified in the specs -- escalate gaps
- Commit secrets, `.env` files, or credentials
- Force-push to `main`
