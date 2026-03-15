# Cobre v0.1.3 — Pre-Release Assessment

**Date:** 2026-03-15
**Scope:** `cobre` repository (main branch, post feat/v0.1.3-stochastic-io merge), assessed against the v0.1.1 post-release assessment and v0.1.3 progress assessment
**Repository stats:** 123,682 lines of Rust across 200 source files, 2,072 `#[test]` functions, 11 ADRs
**Base documents:** `cobre-v0.1.1-post-release-assessment.md`, `cobre-v0.1.3-progress-assessment.md`

---

## Table of Contents

1. [Scorecard — v0.1.3 Progress Assessment Resolution](#1-scorecard)
2. [What Shipped Well](#2-what-shipped-well)
3. [Structural Issues — The Incomplete Pipeline Extraction](#3-structural-issues)
4. [Python Bindings — Feature Parity Gap Widening](#4-python-bindings)
5. [Documentation Staleness](#5-documentation-staleness)
6. [Minor Issues](#6-minor-issues)
7. [Architecture Quality](#7-architecture-quality)
8. [Consolidated Task List](#8-consolidated-task-list)
9. [Summary](#9-summary)

---

## 1. Scorecard — v0.1.3 Progress Assessment Resolution

### Phase 1: CLI Cleanup — Complete

| Task | Status | Evidence |
|------|--------|---------|
| C1: Remove `--skip-simulation` | **Done** | `RunArgs` has 3 flags only: `--output`, `--quiet`, `--threads` |
| C2: Remove `--no-banner` | **Done** | Not in `RunArgs` struct |
| C3: Remove `--verbose` | **Done** | No tracing subscriber setup in `execute()` |
| C4: Remove `--export-stochastic` flag | **Done** | Config-only via `exports.stochastic` in `config.json` |
| C5: Remove `format_stochastic_diagnostics` | **Done** | Zero `[stochastic]` eprintln in cobre-sddp |
| C6: Simplify `BroadcastConfig` | **Done** | No `should_simulate` or `export_stochastic` fields |

### Phase 2: Stochastic Summary + Simulation Progress — Complete

| Task | Status | Evidence |
|------|--------|---------|
| S1-S5: Stochastic summary types and formatting | **Done** | `summary.rs` 1,175 lines with 3-tier AR display, 20+ tests |
| W1: `WelfordAccumulator` to cobre-core | **Done** | `welford.rs` in cobre-core, 147 lines, 3 unit tests |
| W2: `SimulationProgress` simplified | **Done** | `scenario_cost: f64` field replaces 3 stats fields |
| W3: Per-worker accumulators removed | **Done** | Zero `WelfordAccumulator` references in `simulation/pipeline.rs` |
| W4: Progress thread accumulates globally | **Done** | `sim_acc: Option<WelfordAccumulator>` in progress thread event loop |
| W5: Tests updated | **Done** | 5 accumulator-aware tests in `progress.rs` |

### Phase 3: Pipeline Extraction — Partial

| Task | Status | Evidence |
|------|--------|---------|
| P1: `StudySetup` struct | **Done** | `setup.rs` 1,384 lines with 18 fields, 30+ accessor methods |
| P2: `from_system()` constructor | **Done** | `new()` + `from_broadcast_params()` dual-path constructor |
| P3: `train()` method | **Done** | `setup.train()` with 7 arguments |
| P4: `simulate()` method | **Done** | `setup.simulate()` with 4 arguments |
| P5: `stochastic_summary()` / `export_stochastic()` | **Partial** | Summary built as free function in CLI, not a `StudySetup` method; export is a free function in CLI's `run.rs` |
| P6: Rewrite CLI `execute()` | **Incomplete** | Still **1,956 lines** in `run.rs` (target was ~60 for `execute()`) |
| P7: Rewrite Python `run_inner()` | **Done** | `run_inner()` is ~125 lines, uses `StudySetup::new()` cleanly |

### v0.1.1 Documentation Fixes (from previous assessment)

| Task | Status |
|------|--------|
| DEC-XYZ references removed from book | **Done** — 0 matches in `book/src/crates/*.md` |
| Status badges updated | **Done** — all crate pages use `status-alpha` |
| README roadmap rewritten | **Done** — links to implementation-ordering instead of inline checkboxes |
| `book/src/introduction.md` current | **Done** — reflects all 8 phases, 6 subcommands |
| `cobre.dev` schema URLs fixed | **Done** — 0 matches in `book/src/` |
| 4ree example page | **Done** — `book/src/examples/4ree.md` in SUMMARY.md |

### v0.1.1 Code Quality Refactoring (Q1-Q5)

| Task | v0.1.1 | v0.1.3 | Status |
|------|--------|--------|--------|
| Q1: `StageContext` struct | 8 raw args | 1 struct ref | **Done** |
| Q2: Noise extraction | 2 copies | 1 `noise.rs` | **Done** |
| Q3: `ScratchBuffers` separation | 11 flat fields | Nested struct | **Done** |
| Q4: `TrainingContext` bundle | — | 5-field struct | **Done** |
| Q5: Re-enable clippy lint | 8+ suppressions | 2 remaining | **Nearly done** |

---

## 2. What Shipped Well

### 2.1 Hot-Path Argument Counts — Dramatically Improved

| Function | v0.1.1 | v0.1.3 | Plan target |
|---|---|---|---|
| `run_forward_pass()` | 20 | **7** | ~10 |
| `run_backward_pass()` | 20 | **7** | ~10 |
| `simulate()` | 22 | **7** | ~12 |
| `setup.train()` (public) | — | **7** | ~12 |
| `setup.simulate()` (public) | — | **4** | — |
| `train()` (internal free fn) | 25 | **14** | ~13 |

The forward/backward/simulate hot paths exceeded the refactoring target by a wide margin. `StageContext` and `TrainingContext` are clean, well-documented structs in a dedicated `context.rs` module (48 lines). The internal `train()` free function is at 14 args — close to target and reasonable given the generic type parameters (`S: SolverInterface`, `C: Communicator`) and the solver factory closure.

This is the single most important structural improvement in the release.

### 2.2 Noise Transformation — Single Source of Truth

`noise.rs` (725 lines) consolidates all noise-to-RHS-patch logic. Both `transform_inflow_noise()` and `transform_load_noise()` are called from forward pass, backward pass, and simulation. The module has its own test suite. The forward/backward duplication bug class identified in the v0.1.1 assessment is structurally eliminated — there is now no way for one call site to receive a fix and others to be forgotten.

The function signatures are tight:

```
transform_inflow_noise(raw_noise, stage, current_state, &StageContext, &TrainingContext, &mut ScratchBuffers)
transform_load_noise(raw_noise, stage, &StageContext, &mut ScratchBuffers)
```

No dissolved context. Every argument is necessary. The scratch buffers are passed by mutable reference — zero allocation.

### 2.3 Simulation Progress — Correctly Fixed

The per-worker `WelfordAccumulator` bug is resolved by the textbook fix:

1. `WelfordAccumulator` lives in `cobre-core` (correct architectural placement — it has no dependencies).
2. The simulation pipeline emits `scenario_cost: f64` per event — raw data, no pre-aggregation.
3. The progress thread owns the sole accumulator in `sim_acc: Option<WelfordAccumulator>`.
4. Display shows `mean: X  std: Y  CI95: +/-Z` using the global accumulator.

Five tests in `progress.rs` validate the accumulator integration, including a known-mean/known-variance check. This is solid.

### 2.4 CLI Visual Language — Polished

The stochastic summary replaces the homebrew `[stochastic]` `eprintln!` pattern with `Term::stderr()` + `console::style`. The `ArOrderSummary` type has a three-tier display method:

- ≤10 hydros: `AIC (3x order-1, 2x order-2)` — per-order compact form
- 11–30 hydros: `AIC (orders 1-4, 15 hydros)` — range summary
- 31+ hydros: `AIC (order 1: 12, order 2: 8, 31 hydros)` — histogram

Boundary tests at 10/11/30/31 hydros confirm the tier transitions. The `format_openings_per_stage()` helper handles uniform vs. variable branching factors. This scales correctly from 1dtoy (1 hydro) to NEWAVE-scale (165+ hydros).

### 2.5 User-Supplied Opening Trees and Stochastic Export

Both ADR features shipped end-to-end:

- **ADR-008 (User opening tree):** `noise_openings.rs` (573 lines in cobre-io) handles Parquet parsing, validation, and assembly. The CLI loads, validates, broadcasts (via `BroadcastOpeningTree`), and passes the tree to `build_stochastic_context()`. Validation checks dimension, stage coverage, and opening completeness.

- **ADR-009 (Stochastic export):** `output/stochastic.rs` (1,909 lines in cobre-io) writes 6 artifact files. `export_stochastic_artifacts()` in the CLI writes them all, with per-file error handling that logs warnings without aborting.

The round-trip invariant holds: exported `noise_openings.parquet` has the same schema as the input file, so `output/stochastic/noise_openings.parquet` → `scenarios/noise_openings.parquet` is a copy operation.

### 2.6 `StudySetup` — The Right Abstraction

`StudySetup` owns all precomputed study state (templates, indexer, FCF, stochastic context, risk measures, entity counts, block layout, config-derived scalars). Its public API is clean:

```rust
// Construction
StudySetup::new(&system, &config, stochastic) -> Result<Self>
StudySetup::from_broadcast_params(system, stochastic, ...) -> Result<Self>

// Training
setup.train(&mut solver, &comm, n_threads, factory, events, flag) -> Result<TrainingResult>

// Simulation
setup.simulate(&mut workspaces, &comm, &result_tx, events) -> Result<Vec<...>>

// Helpers
setup.create_workspace_pool(n_threads, factory) -> Result<WorkspacePool>
setup.simulation_config() -> SimulationConfig
setup.build_training_output(&result, &events) -> TrainingOutput
```

The Python bindings already benefit: `run_inner()` dropped from ~250 to ~125 lines. The dual-constructor pattern (`new` for local, `from_broadcast_params` for MPI broadcast) correctly handles the architectural constraint that non-root MPI ranks receive pre-extracted scalars.

---

## 3. Structural Issues — The Incomplete Pipeline Extraction

### 3.1 CLI `execute()` Is Still a Monolith

The v0.1.3 plan's primary goal was reducing CLI `execute()` from 530 to ~60 lines. The file is now **1,956 lines** (grew from 530, not shrunk). `execute()` itself is ~400 lines plus `#[allow(clippy::too_many_lines)]`. The reason: `StudySetup` absorbed the computation orchestration, but `execute()` still contains all of:

| Concern | Approximate lines | Should be |
|---|---|---|
| MPI broadcast machinery | ~350 | Separate `broadcast` module |
| Stochastic export | ~90 | `StudySetup` method or separate module |
| Stochastic summary builder | ~100 | `StudySetup` method |
| Simulation result allgatherv + deserialization | ~90 | Separate function |
| Output writing (2 branches) | ~100 | Separate function |
| Test code | ~600 | Separate test file |

The test code alone is 600+ lines in the same file, mixed with production code. The stochastic summary builder (`build_stochastic_summary`, `inflow_models_to_stats_rows`, `inflow_models_to_ar_rows`, `estimation_report_to_fitting_report`) adds ~200 lines of conversion logic that belongs in cobre-sddp or cobre-io, not in the CLI. Also, the `build_stochastic_summary` can't know about the opening tree provenance and puts in a default value with a conservative approach. Actually, we don't track provenance properly for the stochastic elements, we should do it.

### 3.2 Config Parsing Duplicated

Both `StudySetup::new()` (setup.rs:126-211) and `BroadcastConfig::from_config()` (run.rs:162-234) contain identical logic for:

- Seed extraction: `config.training.seed.map_or(DEFAULT_SEED, i64::unsigned_abs)`
- Forward passes: `config.training.forward_passes.unwrap_or(DEFAULT_FORWARD_PASSES)`
- Stopping rule conversion: `StoppingRuleConfig` → `StoppingRule` / `BroadcastStoppingRule` (~30 lines, identical match arms)
- Stopping mode: `eq_ignore_ascii_case("all")` → `StoppingMode::All`
- n_scenarios conditional: `if config.simulation.enabled { ... } else { 0 }`
- Cut selection: `parse_cut_selection_config(&config.training.cut_selection)`

This is the same "dissolved-context" antipattern the refactoring was designed to eliminate — it just migrated up one level from function arguments to config-to-params conversion. When a new config field is added (e.g., `checkpoint_interval`), it must be added in both `StudySetup::new()` and `BroadcastConfig::from_config()`.

**Fix:** Extract a `StudyParams` struct (seed, forward_passes, stopping_rules, n_scenarios, etc.) with a single `from_config(&Config) -> Self` method. Both `StudySetup::new()` and `BroadcastConfig` derive from `StudyParams`.

### 3.3 Constants Duplicated Across Three Files

`DEFAULT_SEED`, `DEFAULT_FORWARD_PASSES`, `DEFAULT_MAX_ITERATIONS` are defined identically in:

| File | Constants defined |
|---|---|
| `cobre-sddp/src/setup.rs` | All three |
| `cobre-cli/src/commands/run.rs` | All three |
| `cobre-python/src/run.rs` | `DEFAULT_SEED` only |

These should live in exactly one place — `cobre-sddp::setup` — and be re-exported. The Python crate can import from `cobre-sddp`.

### 3.4 `StageContext` Construction Not Factored Into a Method

The v0.1.3 plan specified `stage_ctx(&self) -> StageContext<'_>` and `training_ctx(&self) -> TrainingContext<'_>` methods on `StudySetup`. Instead, both structs are built inline in `train()` (lines 533-542) and `simulate()` (lines 592-601) — two copies of the same 8-field struct literal for `StageContext` and two copies of the 5-field struct for `TrainingContext`. This is trivial to factor but wasn't done.

---

## 4. Python Bindings — Feature Parity Gap Widening

### 4.1 Missing Features

| Feature | CLI | Python | Impact |
|---|---|---|---|
| PAR estimation from history | `estimate_from_history()` called before setup | Not called | Users with `inflow_history.parquet` get raw noise, not fitted PAR |
| User-supplied opening tree | `load_user_opening_tree()` + broadcast | Hardcoded `None` | Users cannot supply custom noise realizations |
| Stochastic artifact export | `export_stochastic_artifacts()` | Not implemented | No way to inspect fitted model from Python |
| Stochastic summary | `build_stochastic_summary()` printed to stderr | Not returned in result dict | No diagnostic info in Jupyter notebooks |

Each release adds features to the CLI that the Python bindings don't receive. v0.1.1 added estimation; v0.1.3 added user trees and export. The gap compounds. A user who tries `import cobre; cobre.run.run("case/")` after following the CLI tutorial will get different behavior — estimation won't run, custom trees won't load.

### 4.2 `convert_scenario` Duplicated

The field-for-field type mapping from `SimulationScenarioResult` → `ScenarioWritePayload` exists in two places:

- `crates/cobre-cli/src/simulation_io.rs` — 211 lines, well-structured with per-entity conversion functions
- `crates/cobre-python/src/run.rs` — ~230 lines of inline `.map()` chains in `convert_stage()` and `convert_scenario()`

Both copies are structurally identical. If a field is added to `SimulationStageResult`, both must be updated. The conversion should be a `From` impl in `cobre-io` or `cobre-sddp`.

### 4.3 Policy Checkpoint Also Duplicated

`write_policy_checkpoint` is reimplemented as a standalone function in `cobre-python/src/run.rs` (lines 302-374) because it needs to construct `PolicyCheckpointMetadata` — the same thing the CLI does in `cobre-cli/src/policy_io.rs`. This is a third instance of the duplication pattern.

---

## 5. Documentation Staleness

### 5.1 `docs/ROADMAP.md` — Stale (MODERATE)

The opening section says:

> **Status**: Deferred from v0.1.0. Only the penalty method is available.

Truncation was delivered in v0.1.1. The entire "Inflow Truncation Methods" section is factually wrong. The HPC Optimizations section still references "the rayon baseline for v0.1.0" without acknowledging that v0.1.1 added estimation or v0.1.2 added canonical summation.

### 5.2 `CLAUDE.md` — Missing v0.1.3 (MODERATE)

The phase tracker ends at v0.1.2. The "Current phase" section should document:
- `StudySetup` extraction (pipeline deduplication)
- Noise transformation deduplication (`noise.rs`)
- CLI flag cleanup (4 flags removed)
- Stochastic summary (`summary.rs`)
- Simulation progress fix (WelfordAccumulator moved, per-worker bug fixed)
- User-supplied opening tree implementation (ADR-008)
- Stochastic artifact export implementation (ADR-009)
- `ScratchBuffers` separation in workspace

### 5.3 `CHANGELOG.md` — Empty `[Unreleased]` Section (CRITICAL for release)

All v0.1.3 work is merged but the changelog has only `## [Unreleased]` with no content below it. This must be populated before tagging.

### 5.4 Test Count Discrepancy (LOW)

CLAUDE.md says "Workspace total: 2179 tests" (set during v0.1.1). The current count is 2,072 `#[test]` functions. The delta (-107) is likely from test consolidation during the refactoring — some tests that exercised the old argument-passing patterns were replaced by `StudySetup` integration tests. The tracker should reflect the actual count.

Per-crate breakdown for the record:

| Crate | #[test] count |
|---|---|
| cobre-core | 159 |
| cobre-io | 745 |
| cobre-stochastic | 204 |
| cobre-solver | 86 |
| cobre-comm | 114 |
| cobre-sddp | 576 |
| cobre-cli | 188 |
| cobre-python | 0 |
| **Total** | **2,072** |

### 5.5 `reference/roadmap.md` (Book) — No v0.1.3 Section (LOW)

Lists v0.1.1 and v0.1.2 deliverables but has no v0.1.3 section.

### 5.6 Cargo Version Not Bumped (CRITICAL for release)

`Cargo.toml` workspace version is `0.1.2`. All inter-crate dependency versions reference `0.1.2`. Must be bumped to `0.1.3` before tagging.

---

## 6. Minor Issues

### 6.1 Progress Bar Display Condition Mismatch

In `progress.rs` line 230:

```rust
if scenarios_complete >= 2 {
    // show std and CI
```

`scenarios_complete` is the global atomic counter from the event (counting completions across all 24 workers). The accumulator may have only 1 observation when this condition first triggers — e.g., worker 12 completes first and emits `scenarios_complete=1`, then worker 5 completes and emits `scenarios_complete=2`, but the progress thread's accumulator has only seen 2 events at that point, so `std_dev()` returns a valid but very noisy estimate. This is cosmetic — the display converges quickly — but the condition should be `acc.count() >= 2` (requires adding a `pub fn count(&self) -> u64` method to `WelfordAccumulator`).

### 6.2 `eprintln!` Bypasses `--quiet` Flag

`export_stochastic_artifacts()` uses `eprintln!` for 6 warning messages (lines 1223-1276 of run.rs) instead of `Term::stderr()` or `tracing::warn!`. These warnings print even when `--quiet` is set, breaking the quietness contract. The function already receives `quiet` and `stderr` parameters — it just doesn't use them for error reporting.

### 6.3 `#[allow(clippy::too_many_lines)]` on `execute()`

Line 514: `execute()` still needs this suppression. This is a symptom of §3.1 — the function does too much. Not a release blocker, but a clear signal that the extraction is incomplete.

### 6.4 `BroadcastConfig` Postcard Workaround Adds Complexity

The CLI defines its own `BroadcastStoppingRule`, `BroadcastStoppingMode`, `BroadcastCutSelection` types because the `cobre-sddp` equivalents don't implement `serde::Serialize + serde::Deserialize` (to avoid adding serde to cobre-sddp's public API). This results in ~160 lines of boilerplate conversion code (BroadcastConfig, its `from_config`, its `into_strategy`, `stopping_rules_from_broadcast`). A cleaner approach would be `#[cfg(feature = "serde")]` on the sddp types, or a dedicated `cobre-sddp::broadcast` module.

### 6.5 Opening Tree Broadcast Wrapper

Similarly, `BroadcastOpeningTree` (run.rs:244-248) exists because `OpeningTree` doesn't implement serde. The wrapper deconstructs into `(data, openings_per_stage, dim)` and reconstructs via `OpeningTree::from_parts`. This is correct but adds 40 lines of glue that could be eliminated by `#[cfg(feature = "serde")]` on `OpeningTree`.

---

## 7. Architecture Quality

### 7.1 What's Structurally Sound

**Infrastructure crate genericity:** `grep -riE 'sddp' crates/cobre-core/src/` returns zero matches. Same for cobre-solver, cobre-stochastic, cobre-comm. The firewall holds.

**Dispatch patterns:** No `Box<dyn Trait>` anywhere. `SolverInterface` uses monomorphization. `RiskMeasure`, `CutSelectionStrategy`, `HorizonMode` use enum dispatch. Consistent with DEC-001 and DEC-002.

**Serialization choices:** postcard for MPI broadcast, FlatBuffers for policy, JSON+Parquet for I/O. No protocol drift. The `BroadcastConfig` workaround is ugly but doesn't violate the serialization architecture.

**Error handling:** Hot-path functions (`run_forward_pass`, `run_backward_pass`, `simulate`) return `Result<_, SddpError>`. No panics in the training loop. The `SolverWorkspace` pattern (solver + scratch buffers) prevents solver state corruption on error recovery.

### 7.2 The Dependency Graph

```
StudySetup::new(&System, &Config, StochasticContext)
    ├── build_stage_templates(system, inflow_method, par_lp, normal_lp)
    ├── StageIndexer::with_equipment(...)
    ├── build_initial_state(system, &indexer)
    ├── FutureCostFunction::new(n_stages, n_state, fwd_passes, capacity, 0)
    ├── RiskMeasure::from(stage.risk_config) per stage
    └── build_entity_counts(system)

StudySetup::train(solver, comm, n_threads, factory, events, flag)
    └── crate::train(solver, config, &mut fcf, &stage_ctx, &training_ctx,
                     opening_tree, risk_measures, stopping_rules,
                     cut_selection, flag, comm, n_threads, factory, max_blocks)

StudySetup::simulate(workspaces, comm, result_tx, events)
    └── crate::simulate(workspaces, &stage_ctx, &fcf, &training_ctx,
                        &sim_config, output, comm)
```

The ownership is clean: `StudySetup` owns everything the solver needs, borrows are temporary for each pass. The `&mut self` on `train()` (because FCF is mutated) and `&self` on `simulate()` (read-only FCF) correctly express the mutation contract.

### 7.3 Test Architecture

The 6 `StudySetup` integration tests in `setup.rs` form a good smoke-test suite:
- `from_broadcast_params_minimal` — construction succeeds
- `new_from_config` — config round-trip
- `cut_selection_disabled` — config default
- `train_completes_within_iteration_limit` — end-to-end training
- `train_generates_cuts_in_fcf` — FCF mutation verified
- `simulate_after_train_returns_nonempty_costs` — full lifecycle

The CLI tests (`188 #[test]` functions) cover broadcast helpers, thread resolution, stochastic summary formatting, and the stochastic summary builder with multiple system configurations (with hydros, without hydros, with estimation, without estimation).

---

## 8. Consolidated Task List

### Before Tagging v0.1.3 (2-3 hours)

| # | Task | Effort | Severity |
|---|------|--------|----------|
| R1 | Populate `CHANGELOG.md [Unreleased]` with all v0.1.3 changes | 30 min | **Blocking** |
| R2 | Bump `Cargo.toml` workspace version to `0.1.3` (and all inter-crate deps) | 15 min | **Blocking** |
| R3 | Update CLAUDE.md phase tracker with v0.1.3 status | 15 min | Moderate |
| R4 | Fix test count in CLAUDE.md: 2,072, not 2,179 | 5 min | Low |
| R5 | Add v0.1.3 section to `book/src/reference/roadmap.md` | 15 min | Low |
| R6 | Fix `docs/ROADMAP.md` — Truncation is delivered, not deferred | 15 min | Moderate |

### Short-Term Debt — Extraction Completion (2-3 days)

| # | Task | Effort | Impact |
|---|------|--------|--------|
| D1 | Add `stage_ctx(&self)` and `training_ctx(&self)` methods on `StudySetup` | 30 min | Eliminates 2 copies of struct literal |
| D2 | Deduplicate `DEFAULT_*` constants to `cobre-sddp::setup` | 30 min | Single source of truth |
| D3 | Extract `StudyParams` struct from config parsing, used by both `StudySetup::new()` and `BroadcastConfig::from_config()` | 2h | Eliminates 80 lines of duplicated conversion |
| D4 | Move `convert_scenario` to `cobre-io` as `From<SimulationScenarioResult> for ScenarioWritePayload` | 2h | Eliminates 440 lines across 2 files |
| D5 | Replace `eprintln!` in `export_stochastic_artifacts` with conditional `Term::stderr()` or `tracing::warn!` | 15 min | `--quiet` respected |
| D6 | Add `WelfordAccumulator::count()`, fix progress bar condition to use accumulator count | 15 min | Correct display guard |
| D7 | Extract `build_stochastic_summary` and conversion helpers from CLI to `cobre-sddp` | 2h | Python bindings can reuse |

### Medium-Term — Python Parity (3-5 days)

| # | Task | Effort | Impact |
|---|------|--------|--------|
| P1 | Call `estimate_from_history()` in Python `run_inner()` | 1d | Users get fitted PAR models |
| P2 | Add user-supplied opening tree loading in Python `run_inner()` | 0.5d | Feature parity with CLI |
| P3 | Add stochastic artifact export in Python `run_inner()` | 0.5d | Jupyter model inspection |
| P4 | Return `StochasticSummary` in Python `run()` result dict | 0.5d | Diagnostic info in notebooks |
| P5 | Factor pre-setup orchestration (estimation + tree loading) into a shared function in `cobre-sddp` callable from both CLI and Python | 1d | Structural fix for parity gap |

### Medium-Term — CLI Modularization (2-3 days)

| # | Task | Effort | Impact |
|---|------|--------|--------|
| E1 | Extract MPI broadcast machinery (`BroadcastConfig`, helpers) into `cobre-cli::broadcast` module | 1d | `run.rs` drops ~350 lines |
| E2 | Extract simulation result allgatherv + deserialization into a function | 0.5d | `execute()` drops ~90 lines |
| E3 | Extract output writing branches into a function | 0.5d | `execute()` drops ~100 lines |
| E4 | Move test code from `run.rs` to `tests/cli_run_stochastic.rs` | 0.5d | Source file drops ~600 lines |

After E1-E4, `execute()` should be under 120 lines — not the originally planned 60, but close enough given the inherent MPI complexity.

### Recommended Ordering

```
Day 1:    R1-R6 (release prep) + D1-D2 (trivial dedup)
Day 2:    D3, D5, D6 (config dedup, quiet fix, accumulator fix)
Day 3:    D4, D7 (shared conversion, shared summary)
Day 4-5:  P1-P5 (Python parity — uses D4 and D7 outputs)
Day 6-8:  E1-E4 (CLI modularization)
```

The Python parity work (P1-P5) should come before CLI modularization (E1-E4) because the parity gap affects users, while CLI line counts are an internal quality concern.

---

## 9. Summary

**The v0.1.3 branch is release-ready after the 6 documentation/versioning items in §8.** The hot-path refactoring exceeded targets, the simulation progress bug is correctly fixed, the CLI visual language is professional, and two non-trivial ADR features shipped end-to-end.

**The pipeline extraction is architecturally correct but half-finished.** `StudySetup` is the right abstraction and already delivers value (Python bindings at 125 lines). But the CLI `execute()` grew instead of shrinking because MPI broadcast, simulation gathering, output writing, and stochastic summary building weren't extracted alongside computation orchestration. The config parsing duplication between `StudySetup::new()` and `BroadcastConfig::from_config()` is the most structurally concerning leftover — it's the v0.1.1 "dissolved context" antipattern at a new level.

**The Python feature parity gap is the most user-facing concern.** Every CLI feature (estimation, user trees, stochastic export) that isn't wired in the Python bindings means Python users get a degraded experience. The fix is to push more orchestration into `StudySetup` or a shared pre-setup layer, which aligns with completing the pipeline extraction.

**The codebase quality trajectory is positive.** The v0.1.1 assessment found "actively degrading" code with 25-argument functions, duplicated noise logic, and dissolved context everywhere. The v0.1.3 state is "structurally sound with well-characterized debt." The remaining work is mechanical extraction, not architectural redesign.

---

**End of Assessment.**
