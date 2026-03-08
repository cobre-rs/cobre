# Phase 7 Assessment: Simulation Pipeline + Output Infrastructure

**Date:** 2026-03-08
**Scope:** cobre-rs/cobre main branch after Phase 7 implementation

---

## 1. Verdict

**Proceed to Phase 8 (cobre-cli).** Phase 7 is complete and well-implemented. The simulation pipeline, Parquet output writers, FlatBuffers policy checkpoint, training output bridge, and dictionary/manifest writers are all in place. The full train→simulate→write cycle is covered by an end-to-end integration test. No blockers.

---

## 2. Previous Observations: Resolved

All three items flagged in the post-refactor assessment have been addressed:

| Item | Status | Evidence |
|------|--------|----------|
| `interpret_terminal_status` return type | **Fixed** | Now returns `Option<SolverError>` (highs.rs:337). Call sites use `if let Some(terminal_err)` not `if let Some(Err(e))`. |
| Missing ADR for basis migration | **Fixed** | ADR-007 created (docs/adr/007-raw-basis-status-codes.md). Clean, correct, references DEC-002. |
| Backward pass `binding_slots` allocation | **Fixed** | `binding_slots.clear()` reuse pattern (backward.rs:209) replaces per-opening `Vec` allocation. |

The backward pass coefficient `Vec<f64>` allocation (backward.rs:204) remains per-opening. This is structurally necessary — each opening's coefficients move into `BackwardOutcome` for risk measure aggregation. Eliminating it would require a fundamentally different aggregation API. Not worth pursuing.

---

## 3. Phase 7 Deliverables: Complete

### 3.1 Simulation Pipeline (cobre-sddp/src/simulation/)

The simulation module implements forward-only policy evaluation with correct solver interaction patterns:

- **Basis warm-starting across scenarios** (pipeline.rs:171-275): Same `Vec<Option<Basis>>` cache pattern as the training loop. First scenario cold-starts; subsequent scenarios warm-start at each stage.
- **Zero-copy solution extraction** (pipeline.rs:221-225): Uses `SolutionView` from `solve()` / `solve_with_basis()`. Extracts what it needs before the next solver mutation.
- **Bounded channel streaming** (pipeline.rs:302-304): Per-scenario results are sent through `SyncSender<SimulationScenarioResult>` for background I/O. The sender blocks when the channel is full, providing natural backpressure against the solver outpacing the writer.
- **Scenario distribution** (extraction.rs:99): `assign_scenarios(n_scenarios, rank, world_size)` splits scenarios across MPI ranks with remainder distribution.
- **Seed domain separation** (pipeline.rs:181): `SIMULATION_SEED_OFFSET` ensures simulation and training use disjoint regions of the SipHash seed space.

### 3.2 Result Extraction (cobre-sddp/src/simulation/extraction.rs)

Per-entity result extraction from `SolutionView` is implemented for all entity types. The extraction reads column positions from `StageIndexer` — same indexer used by the training loop. Storage, inflow lags, and state variables are extracted at known positions. Equipment columns (turbined, spillage, generation) are stubbed pending LP builder equipment column range exposure. This is expected — the LP builder currently handles equipment columns internally.

### 3.3 Training Output Bridge (cobre-sddp/src/training_output.rs)

The `build_training_output` function converts `TrainingResult` + `Vec<TrainingEvent>` into `TrainingOutput` using a `BTreeMap<u64, PartialRecord>` to correlate events by iteration number. This is a clean post-hoc reconstruction that doesn't touch the hot-path training code.

`TrainingEvent` lives in `cobre-core` (training_event.rs, 21K) — a deliberate placement that allows the event type to be shared without creating a circular dependency between `cobre-sddp` and `cobre-io`.

### 3.4 Parquet Output Writers (cobre-io/src/output/)

- **Training writer** (training_writer.rs): Writes convergence records and timing data as Parquet with Arrow RecordBatch construction.
- **Simulation writer** (simulation_writer.rs): Hive-partitioned Parquet output (`simulation/costs/scenario_id=NNNN/data.parquet`, etc.). One Parquet file per entity type per scenario. `build_*_batch` functions construct Arrow arrays from write records.
- **Dictionary writer** (dictionary.rs): JSON files for codes, entities, variables, bounds, state mappings.
- **Manifest/metadata writers** (manifest.rs): `_manifest.json` and `metadata.json` with commit-point semantics — `_SUCCESS` sentinel file written last.

### 3.5 FlatBuffers Policy Checkpoint (cobre-io/src/output/policy.rs)

Uses the `flatbuffers` runtime builder API directly (no `.fbs` codegen). Per-stage cut files (`cuts/stage_NNN.bin`) and basis files (`basis/stage_NNN.bin`). `metadata.json` is written last as the commit signal — presence of metadata means the checkpoint is complete. `PolicyCutRecord<'a>` borrows coefficient slices to avoid copying the 2,080-element coefficient vectors.

### 3.6 End-to-End Integration Test

`simulation_integration.rs::train_simulate_write_cycle` exercises the complete pipeline: train with mock solver → collect events → build training output → write policy checkpoint → simulate → stream results → write Parquet/manifests → verify file existence and content. This is the single most valuable test in the codebase for Phase 7 correctness.

---

## 4. Observations (Non-Blocking)

### 4.1 `docs/PROJECT-STATUS.md` is stale

Still says "Phase 4 complete" (line 6). `CLAUDE.md` is the authoritative source and is correct (Phase 7 complete). Either update PROJECT-STATUS.md or remove it — having two status trackers that disagree is worse than having one.

### 4.2 Simulation extraction stubs for equipment columns

`extract_stage_result` returns zero values for equipment columns (`turbined_m3s`, `spillage_m3s`, `generation_mw`, etc.) across all entity types. Comments indicate this is deferred until "LP builder exposes equipment column ranges." This is correct for the MVP — the important state variables (storage, inflow lags) are extracted. But the simulation output will have limited usefulness for operational analysis until these are filled in. Flag this for a post-v0.1.0 iteration or for when the LP builder's column layout is stabilized.

### 4.3 Per-scenario Parquet partitioning may not scale

The simulation writer creates one directory per scenario (`scenario_id=0000/`, `scenario_id=0001/`, ...) with one Parquet file per entity type per scenario. At production scale (2,000 scenarios × 10 entity types), this produces 20,000 small Parquet files. Most Parquet readers (DuckDB, Spark, Pandas) handle this via partition discovery, but filesystem overhead from many small files can be significant on network filesystems.

An alternative is to batch scenarios into larger Parquet files (e.g., 100 scenarios per file) and use row groups for per-scenario access. This is a future optimization — the current approach is correct and interoperable with standard Hive partition conventions.

### 4.4 Simulation pipeline is single-threaded per rank

The simulation inner loop (pipeline.rs:200-276) is sequential: one scenario at a time, one stage at a time. This is correct for the MVP — solver instances are thread-local and not `Sync`, so parallelizing across scenarios within a rank requires multiple solver instances (the Rayon-based parallelism planned for later phases). MPI-level parallelism across ranks is already in place via `assign_scenarios`.

### 4.5 `SolverError` variants lost diagnostic data during cleanup

The previous version had `Infeasible { ray: Option<Vec<f64>> }` and `NumericalDifficulty { partial_solution: Option<LpSolution> }`. The current version simplified these to unit variants or scalar-only payloads. This was the right call for the hot path (no heap allocation on error), but if you ever need to debug why an LP is infeasible in production, you'll need to add a diagnostic mode that re-extracts the ray. Not a concern for Phase 8.

---

## 5. Test Coverage

| Crate | #[test] count | Delta from Phase 6 |
|-------|---------------|---------------------|
| cobre-solver | 86 | 0 (unchanged) |
| cobre-sddp | 419 | +97 |
| cobre-io | 641 | +117 |
| cobre-core | 154 | 0 |
| cobre-stochastic | 114 | 0 |
| cobre-comm | 114 | 0 |
| **Total** | **1,528** | **+214** |

CLAUDE.md reports 1,718 — difference (~190) is doc tests, consistent with previous pattern.

---

## 6. Recommendation

**Proceed to Phase 8 (cobre-cli).** The simulation and output infrastructure is complete. The integration test validates the full lifecycle. The remaining observations (stale PROJECT-STATUS, equipment column stubs, partition scaling) are all post-v0.1.0 concerns.

Phase 8 scope from the roadmap: `cobre validate`, `cobre run`, `cobre report` subcommands with `clap` argument parsing, config resolution, and exit codes. This is the final phase before v0.1.0.
