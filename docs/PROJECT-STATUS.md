# Cobre — Project Status

Implementation progress across all crates (updated: v0.1.0 release, 2026-03-10).

---

## Overall Status

**All 8 phases complete. v0.1.0 release blockers resolved.** Workspace total: 1955 tests.

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

| Crate              | Phase | Status   | Notes                                                                                         |
| ------------------ | ----- | -------- | --------------------------------------------------------------------------------------------- |
| `cobre-core`       | 1     | complete | Entity model, registries, topology, penalty resolution                                        |
| `cobre-io`         | 2     | complete | load_case pipeline, 5-layer validation, 33-file JSON/Parquet loading                          |
| `cobre-solver`     | 3     | complete | LP solver abstraction, HiGHS backend, FFI bindings, warm-start                                |
| `ferrompi`         | 3     | external | MPI 4.x bindings -- transferred from rjmalves/ferrompi, v0.2.0                                |
| `cobre-comm`       | 4     | complete | Communicator trait, LocalBackend, FerrompiBackend, factory function                           |
| `cobre-stochastic` | 5     | complete | PAR(p) preprocessing, SipHash seed derivation, Cholesky correlation, opening tree, InSample   |
| `cobre-sddp`       | 6+7   | complete | SDDP training loop, simulation pipeline, rayon thread parallelism, Parquet/FlatBuffers output |
| `cobre-cli`        | 8     | complete | run/validate/report/version subcommands, --threads flag, progress bars, terminal banner       |
| `cobre-mcp`        | post  | stub     | MCP server for AI agent integration -- not yet implemented                                    |
| `cobre-python`     | post  | stub     | PyO3 Python bindings (cdylib) -- not yet implemented                                          |
| `cobre-tui`        | post  | stub     | ratatui terminal UI -- not yet implemented                                                    |

---

## Test Coverage Summary

| Crate              | Unit | Integration    | Other | Doc-tests | Total | Notes                                                                                                       |
| ------------------ | ---- | -------------- | ----- | --------- | ----- | ----------------------------------------------------------------------------------------------------------- |
| `cobre-core`       | 147  | 7              | --    | 34        | 188   | Includes serde round-trip tests under `--features serde`                                                    |
| `cobre-io`         | 631  | 10             | --    | 108       | 749   | 2 doc-tests ignored (require testdata not in CI path)                                                       |
| `cobre-solver`     | 47   | 39             | --    | 2         | 88    | 39 HiGHS conformance tests; mpi feature tests excluded                                                      |
| `cobre-comm`       | 74   | 18+11          | --    | 9         | 112   | 18 local conformance + 11 factory tests; no-feature build                                                   |
| `cobre-stochastic` | 106  | 5+4            | --    | 11        | 126   | 5 conformance + 4 reproducibility (declaration-order gate)                                                  |
| `cobre-sddp`       | 470  | 13+7+2+3+1     | --    | 49        | 545   | 13 conformance + 7 integration + 2 determinism + 3 inflow_nonnegativity + 1 simulation; 2 doc-tests ignored |
| `cobre-cli`        | 93   | 6+12+11+11+8+6 | --    | 0         | 147   | color/report/run/smoke/validate/init integration via assert_cmd                                             |

**Workspace total (phases 1–8 + v0.1.0 blockers): 1955 tests passing.**

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
- 126 tests

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
- 147 tests

### v0.1.0 Release Blockers

Work completed after Phase 8 to reach the v0.1.0 release tag:

- **Epic 01 — Rayon thread parallelism**: Intra-rank thread parallelism implemented in
  `cobre-sddp` for the forward pass, backward pass, and simulation pipeline. Each thread
  owns its own `SolverInterface` instance (HiGHS is not thread-safe). Controlled via the
  `--threads N` CLI flag or the `COBRE_THREADS` environment variable; default is 1 thread.
  Determinism tests verify bit-for-bit identical results regardless of thread count.

- **Epic 02 — Inflow non-negativity**: Penalty-method handling of negative PAR(p)
  realisations implemented in `cobre-sddp`. The `InflowNonNegativityMethod` enum exposes
  `Penalty { cost }` and `None` variants. Truncation variants are deferred to post-v0.1.0.
  See `docs/deferred-truncation-design.md` for the architectural analysis.

- **Epic 04 — Documentation**: Software book chapters, ADR entries, and API reference
  updated to reflect the v0.1.0 feature set, including the two-level parallelism model,
  the inflow non-negativity configuration, and the `--color` / `--threads` CLI flags.

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

## Critical Gaps for v0.1.0

### Intra-rank thread parallelism (RESOLVED)

The SDDP solver uses a two-level parallelism model:

1. **Inter-rank (MPI)**: forward passes and simulation scenarios are distributed
   across MPI ranks via base/remainder assignment. This is implemented.
2. **Intra-rank (threads)**: within each MPI rank, the assigned forward passes
   (training) or scenarios (simulation) are processed in parallel using rayon's
   work-stealing thread pool. **This is implemented as of Epic 01.**

The forward pass, backward pass, and simulation pipeline in `cobre-sddp` all use
`rayon::iter` to parallelize over their assigned work. Each thread owns a private
`SolverInterface` (HiGHS) instance and a private `PatchBuffer` workspace. The FCF
cut pool is read-only during the forward pass and updated with a locking merge
after the backward pass.

The thread count is controlled at runtime via:

- `--threads N` CLI flag on the `run` subcommand
- `COBRE_THREADS` environment variable
- Default: 1 thread (conservative; explicit opt-in for parallelism)

The `threads_per_rank` field in `TrainingEvent::TrainingStarted` reflects the
actual configured thread count.

### Inflow non-negativity: penalty method only

PAR(p) models can produce negative inflow realisations. The SDDP literature
describes three methods for handling this: penalty (slack variable with high
cost), truncation (external AR evaluation with modified LP bounds), and
truncation-with-penalty (bounded slack combined with modified bounds).

**v0.1.0 implements only the penalty method.** The `InflowNonNegativityMethod`
enum has `Penalty { cost }` and `None` variants; the `Truncation` and
`TruncationWithPenalty` variants are deferred.

**Why truncation is deferred**: Cobre's LP formulation encodes AR dynamics
implicitly in the water balance row bounds. Truncation requires evaluating
the full inflow value `a_h` as a scalar before LP patching, which means
threading AR coefficients from `StochasticContext` through to the forward
pass — a non-trivial architectural change. The penalty method gives correct
results at the cost of allowing virtual inflow at the penalty rate.

**Full analysis**: `docs/deferred-truncation-design.md`

---

## Links

- Specification corpus: [cobre-docs](https://github.com/cobre-rs/cobre-docs) / [deployed](https://cobre-rs.github.io/cobre-docs/)
- Implementation ordering: `~/git/cobre-docs/src/specs/overview/implementation-ordering.md`
- Architecture Decision Records: `docs/adr/`
- Software book: `book/` (mdBook, built with `mdbook build`)
- Phase tracker (AI context): `CLAUDE.md` — Phase tracker section
