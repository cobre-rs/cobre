# Cobre — Project Status

Implementation progress across all crates (updated: Phase 8 complete, 2026-03-09).

---

## Overall Status

**All 8 phases complete.** Workspace total: 1851 tests.

### Phase Progress

| Phase | Crates                      | Status   | Completed  |
| ----- | --------------------------- | -------- | ---------- |
| 1     | `cobre-core`                | complete | 2026-02-21 |
| 2     | `cobre-io`                  | complete | 2026-03-04 |
| 3     | `ferrompi` + `cobre-solver` | complete | 2026-03-04 |
| 4     | `cobre-comm`                | complete | 2026-03-05 |
| 5     | `cobre-stochastic`          | complete | 2026-03-07 |
| 6     | `cobre-sddp` (training)     | complete | 2026-03-08 |
| 7     | `cobre-sddp` (simulation)   | complete | 2026-03-08 |
| 8     | `cobre-cli`                 | complete | 2026-03-09 |

---

## Per-Crate Implementation Status

| Crate              | Phase | Status   | Notes                                                                                       |
| ------------------ | ----- | -------- | ------------------------------------------------------------------------------------------- |
| `cobre-core`       | 1     | complete | Entity model, registries, topology, penalty resolution                                      |
| `cobre-io`         | 2     | complete | load_case pipeline, 5-layer validation, 33-file JSON/Parquet loading                        |
| `cobre-solver`     | 3     | complete | LP solver abstraction, HiGHS backend, FFI bindings, warm-start                              |
| `ferrompi`         | 3     | external | MPI 4.x bindings -- transferred from rjmalves/ferrompi, v0.2.0                              |
| `cobre-comm`       | 4     | complete | Communicator trait, LocalBackend, FerrompiBackend, factory function                         |
| `cobre-stochastic` | 5     | complete | PAR(p) preprocessing, SipHash seed derivation, Cholesky correlation, opening tree, InSample |
| `cobre-sddp`       | 6+7   | complete | SDDP training loop, simulation pipeline, Parquet/FlatBuffers output                         |
| `cobre-cli`        | 8     | complete | run/validate/report/version subcommands, progress bars, terminal banner, post-run summary   |
| `cobre-mcp`        | post  | stub     | MCP server for AI agent integration -- not yet implemented                                  |
| `cobre-python`     | post  | stub     | PyO3 Python bindings (cdylib) -- not yet implemented                                        |
| `cobre-tui`        | post  | stub     | ratatui terminal UI -- not yet implemented                                                  |

---

## Test Coverage Summary

| Crate              | Unit | Integration | Other | Doc-tests | Total | Notes                                                      |
| ------------------ | ---- | ----------- | ----- | --------- | ----- | ---------------------------------------------------------- |
| `cobre-core`       | 147  | 7           | --    | 34        | 188   | Includes serde round-trip tests under `--features serde`   |
| `cobre-io`         | 631  | 10          | --    | 108       | 749   | 2 doc-tests ignored (require testdata not in CI path)      |
| `cobre-solver`     | 47   | 39          | --    | 2         | 88    | 39 HiGHS conformance tests; mpi feature tests excluded     |
| `cobre-comm`       | 74   | 18+11       | --    | 9         | 112   | 18 local conformance + 11 factory tests; no-feature build  |
| `cobre-stochastic` | 105  | 5+4         | --    | 11        | 125   | 5 conformance + 4 reproducibility (declaration-order gate) |
| `cobre-sddp`       | 418  | 13+7+1      | --    | 40        | 479   | 13 conformance + 7 integration + 1 genericity gate         |
| `cobre-cli`        | 70   | 12+11+9+8   | --    | 0         | 110   | run/validate/report/version integration via assert_cmd     |

**Workspace total (phases 1–8): 1851 tests passing.**

Run the full suite with:

```bash
cargo test --workspace --all-features
```

---

## Key Milestones Achieved

### Phase 1 — cobre-core

- Entity model (Bus, Line, Thermal, Hydro, Contract, PumpingStation, NonControllable)
- System registry with topology validation and three-tier penalty resolution
- Declaration-order invariance; serde support
- 188 tests

### Phase 2 — cobre-io

- `load_case(path)` with 5-layer validation pipeline (structural → schema → referential → dimensional → semantic)
- 33-file JSON/Parquet loading; penalty/bound resolution; postcard serialization for MPI
- Validation report writer with structured error output
- 749 tests

### Phase 3 — cobre-solver + ferrompi

- `SolverInterface` trait with compile-time monomorphization; HiGHS backend with FFI bindings
- 5-level retry escalation for difficult LPs; dual normalization; warm-start basis management
- ferrompi v0.2.0 audited (READY WITH ADAPTATIONS)
- 88 tests

### Phase 4 — cobre-comm

- `Communicator` trait with `allgatherv`, `allreduce`, `broadcast`, `barrier`, `rank`, `size`
- `LocalBackend` (always available) and `FerrompiBackend` (MPI 4.x) with factory selection
- `BackendKind` enum with `COBRE_COMM_BACKEND` override; zero algorithm-specific references
- 112 tests

### Phase 5 — cobre-stochastic

- PAR(p) preprocessing with Yule-Walker estimation; SipHash-1-3 seed derivation for parallel noise
- Cholesky correlation; opening tree construction; InSample sampling scheme
- Declaration-order invariance enforced
- 125 tests

### Phase 6 — cobre-sddp training

- SDDP training loop with forward/backward passes and per-stage `allgatherv`
- Level-1 cut selection with deactivation sets; convergence monitoring via `StoppingRuleSet`
- Risk-neutral Expectation measure
- 351 tests

### Phase 7 — cobre-sddp simulation + output

- Simulation pipeline with full scenario distribution and MPI result aggregation
- Hive-partitioned Parquet writers for convergence, timing, and per-entity results
- JSON manifest/metadata; dictionary writers; FlatBuffers policy checkpoint
- TrainingOutput bridge
- 1228 tests (cobre-io: 749, cobre-sddp: 479)

### Phase 8 — cobre-cli

- `run`, `validate`, `report`, `version` subcommands
- Progress bars; terminal banner; post-run summary table
- Config resolution via env vars, defaults, and CLI flags
- Exit codes: 0 (success), 1 (validation), 2 (I/O), 3 (solver), 4 (internal)
- 110 tests

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
