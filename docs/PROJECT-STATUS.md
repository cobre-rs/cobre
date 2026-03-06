# Cobre — Project Status

This document provides a quick overview of implementation progress across all
crates. It is updated at the completion of each implementation phase.

Last updated: Phase 4 complete (2026-03-05).

---

## Overall Status

**Phases 1 through 4 are complete.** The foundation data model (`cobre-core`), the
I/O layer (`cobre-io`), the LP solver abstraction (`cobre-solver`/`ferrompi`), and
the communication abstraction layer (`cobre-comm`) are fully implemented and passing
all tests. Phase 5 (`cobre-stochastic`) is the next candidate.

### Phase Progress

| Phase | Crates                      | Status      | Completed  |
| ----- | --------------------------- | ----------- | ---------- |
| 1     | `cobre-core`                | complete    | 2026-02-21 |
| 2     | `cobre-io`                  | complete    | 2026-03-04 |
| 3     | `ferrompi` + `cobre-solver` | complete    | 2026-03-04 |
| 4     | `cobre-comm`                | complete    | 2026-03-05 |
| 5     | `cobre-stochastic`          | not started | --         |
| 6     | `cobre-sddp` (training)     | not started | --         |
| 7     | `cobre-sddp` (simulation)   | not started | --         |
| 8     | `cobre-cli`                 | not started | --         |

---

## Per-Crate Implementation Status

| Crate              | Phase | Status   | Notes                                                                |
| ------------------ | ----- | -------- | -------------------------------------------------------------------- |
| `cobre-core`       | 1     | complete | Entity model, registries, topology, penalty resolution               |
| `cobre-io`         | 2     | complete | load_case pipeline, 5-layer validation, 33-file JSON/Parquet loading |
| `cobre-solver`     | 3     | complete | LP solver abstraction, HiGHS backend, FFI bindings, warm-start       |
| `ferrompi`         | 3     | external | MPI 4.x bindings -- transferred from rjmalves/ferrompi, v0.2.0       |
| `cobre-comm`       | 4     | complete | Communicator trait, LocalBackend, FerrompiBackend, factory function  |
| `cobre-stochastic` | 5     | stub     | PAR(p) models, scenario generation -- not yet implemented            |
| `cobre-sddp`       | 6+7   | stub     | SDDP training loop and simulation -- not yet implemented             |
| `cobre-cli`        | 8     | stub     | Binary entry point (clap) -- not yet implemented                     |
| `cobre-mcp`        | post  | stub     | MCP server for AI agent integration -- not yet implemented           |
| `cobre-python`     | post  | stub     | PyO3 Python bindings (cdylib) -- not yet implemented                 |
| `cobre-tui`        | post  | stub     | ratatui terminal UI -- not yet implemented                           |

---

## Test Coverage Summary

Test counts reflect `cargo test -p <crate> --all-features` output as of the last
phase completion.

| Crate            | Unit | Integration | Invariance | Doc-tests | Total | Notes                                                     |
| ---------------- | ---- | ----------- | ---------- | --------- | ----- | --------------------------------------------------------- |
| `cobre-core`     | 137  | 7           | --         | 33        | 177   | Includes serde round-trip tests under `--features serde`  |
| `cobre-io`       | 514  | 6           | 3          | 99        | 622   | 2 doc-tests ignored (require testdata not in CI path)     |
| `cobre-solver`   | 35   | 30          | --         | 2         | 67    | 30 HiGHS conformance tests; mpi feature tests excluded    |
| `cobre-comm`     | 54   | 28          | --         | 8         | 90    | 18 local conformance + 10 factory tests; no-feature build |
| All other crates | --   | --          | --         | --        | 0     | Stub crates with no tests yet                             |

**Workspace total (phases 1–4): 956 tests passing.**

Run the full suite with:

```bash
cargo test --workspace --all-features
```

---

## Key Milestones Achieved

### Phase 1 — cobre-core (complete)

- Entity model: `Bus`, `Line`, `Thermal`, `Hydro`, `Contract`, `PumpingStation`, `NonControllable`
- `System` type with typed registries for all 7 entity types
- Topology validation (bus connectivity, reference bus checks)
- Three-tier penalty resolution (global / entity / stage)
- Serde support for all core types (optional feature)
- Declaration-order invariance: results identical regardless of input ordering
- 177 tests: 137 unit + 7 integration + 33 doc-tests

### Phase 2 — cobre-io (complete)

- `load_case(path)` — single-call entry point that loads a full case directory
- 5-layer validation pipeline: structural → schema → referential → dimensional → semantic
- 33-file JSON/Parquet loading covering all entity types and temporal data
- Penalty resolution: merges global, entity-level, and stage-level penalty tables
- Bound resolution: resolves entity bounds from raw input to validated `ResolvedBounds`
- Postcard broadcast serialization: `Case` implements `postcard::Serialize`/`Deserialize` for MPI distribution
- Validation report writer: structured error and warning output
- Entity registry loading: populates `cobre-core` registries from parsed input
- 622 tests: 514 unit + 6 integration + 3 invariance + 99 doc-tests

### Phase 3 — cobre-solver + ferrompi (complete)

- `SolverInterface` trait with compile-time monomorphization (DEC-002)
- `HighsSolver` backend with custom FFI bindings to HiGHS LP solver
- 5-level retry escalation for numerically difficult LPs
- Dual normalization and warm-start basis management
- 30 HiGHS conformance tests covering all solver states and edge cases
- ferrompi v0.2.0 audited for Phase 4 readiness (READY WITH ADAPTATIONS)
- 67 tests: 35 unit + 30 integration + 2 doc-tests

### Phase 4 — cobre-comm (complete)

- `Communicator` trait: `allgatherv`, `allreduce`, `broadcast`, `barrier`, `rank`, `size`
- `SharedMemoryProvider` and `SharedRegion` traits for intra-node shared memory
- `LocalCommunicator` object-safe sub-trait for initialization coordination
- `LocalBackend`: single-process no-op backend with heap-fallback shared memory (always available)
- `FerrompiBackend`: MPI 4.x backend wrapping ferrompi v0.2.0 with `i32`↔`usize` conversion layer
- `BackendKind` enum and `create_communicator` factory with `COBRE_COMM_BACKEND` environment override
- Local backend conformance test suite (18 tests) and factory/error integration tests (10 tests)
- Infrastructure crate genericity enforced: zero algorithm-specific references in source
- 90 tests: 54 unit + 28 integration + 8 doc-tests

---

## Dependency Graph (Workspace)

```
cobre-core
    └── cobre-io
            └── (feeds into cobre-sddp in Phase 6)

ferrompi (external crate)
    └── cobre-solver  [Phase 3]
            └── cobre-comm  [Phase 4]
                    └── (feeds into cobre-sddp in Phase 6)

cobre-stochastic  [Phase 5, depends on cobre-core]
    └── (feeds into cobre-sddp in Phase 6)

cobre-sddp  [Phase 6+7, depends on cobre-core + cobre-io + cobre-solver + cobre-comm + cobre-stochastic]
    └── cobre-cli  [Phase 8]

cobre-core
    ├── cobre-mcp   [post-Phase 8, single-process]
    ├── cobre-python  [post-Phase 8, single-process]
    └── cobre-tui   [post-Phase 8, UI only]
```

---

## Links

- Specification corpus: [cobre-docs](https://github.com/cobre-rs/cobre-docs) / [deployed](https://cobre-rs.github.io/cobre-docs/)
- Implementation ordering: `~/git/cobre-docs/src/specs/overview/implementation-ordering.md`
- Architecture Decision Records: `docs/adr/`
- Software book: `book/` (mdBook, built with `mdbook build`)
- Phase tracker (AI context): `CLAUDE.md` — Phase tracker section
