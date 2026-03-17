# Cobre -- Development Guidelines

This file is loaded by Claude Code when working in this repository. It captures
the conventions, source-of-truth references, and architectural decisions that
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

## Current State (v0.1.3)

The SDDP solver is fully functional. The pipeline covers case loading, stochastic
scenario generation, training, simulation, policy checkpointing, and output writing.
2,624 tests across the workspace, including a deterministic regression suite (D01-D12)
with hand-computed expected costs.

**What's implemented:**

- Training loop with forward/backward pass, Benders cut management, 5 stopping rules
- Constant-productivity and FPHA hydro production models (precomputed + computed from geometry)
- Cascade hydro coupling, evaporation linearization, inflow non-negativity penalties
- PAR(p) fitting (Levinson-Durbin, AIC order selection), stochastic load demand
- Simulation pipeline with FlatBuffers policy checkpoint and Parquet output
- Multi-bus transmission with line flow limits
- MPI distribution (ferrompi) and intra-rank thread parallelism (rayon, `--threads N`)
- CLI: `init`, `run`, `validate`, `report`, `summary`, `version`
- Python bindings (PyO3, tested on 3.12/3.13/3.14)

**Known gaps:**

- Multi-segment deficit pricing (LP builder uses single deficit variable per bus)
- CVaR risk measure (enum variant exists, LP modification not implemented)
- GNL thermals, batteries, non-controllable sources (entity stubs exist, no LP contribution)

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

| Element          | Status | Notes                                                                                                                       |
| ---------------- | ------ | --------------------------------------------------------------------------------------------------------------------------- |
| Bus              | Full   | Power balance constraint per bus per block                                                                                  |
| Line             | Full   | Flow variable with MW capacity bounds                                                                                       |
| Thermal          | Full   | Generation variable with MW bounds and cost                                                                                 |
| Hydro            | Full   | Reservoir, turbine, spillage. Constant productivity, FPHA (precomputed + computed from geometry), evaporation linearization |
| Contract         | Stub   | NO-OP -- entity exists, no LP contribution                                                                                  |
| Pumping Station  | Stub   | NO-OP -- entity exists, no LP contribution                                                                                  |
| Non-Controllable | Stub   | NO-OP -- entity exists, no LP contribution                                                                                  |

Stubs exist in the registry so LP construction code iterates over all element
types uniformly. Implementing an element means adding LP variables/constraints.

---

## Coding Standards

### Hard rules (enforced by workspace lints)

- `unsafe_code = "forbid"` -- no unsafe at all
- `unwrap_used = "deny"` -- no `.unwrap()` in library code (ok in tests)
- `clippy::all` and `clippy::pedantic` at `warn` level
- `missing_docs = "warn"` -- all public items should have doc comments
- **`cargo fmt` must pass** -- run `cargo fmt --all` after every implementation ticket
  and before committing. CI enforces `cargo fmt --all -- --check`.

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

**Quality gate:** All infrastructure crate files must pass
`grep -riE 'sddp' crates/<crate>/src/` with zero matches.

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
3. **Specify the target crate and module** -- e.g., "implement in `crates/cobre-sddp/src/risk.rs`"

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
