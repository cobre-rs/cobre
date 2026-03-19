# Cobre v0.1.4 — State Assessment & Roadmap Alignment

**Date:** 2026-03-17
**Scope:** Full codebase review (`main` at `4a80e2d`), cross-referenced against all prior assessments (v0.1.1 post-release, v0.1.3 progress, v0.1.3 pre-release, v0.1.4 quality assessment) and both roadmap documents (definitive roadmap, revised roadmap)
**Base documents:** `cobre-v0_1_3-pre-release-assessment.md`, `definitive-roadmap.md`, `revised-roadmap.md`, `2026-03-17-v0_1_4-quality-assessment.md`

---

## Table of Contents

1. [Metrics Evolution Across Releases](#1-metrics-evolution)
2. [What v0.1.4 Delivered](#2-what-v014-delivered)
3. [v0.1.3 Debt Resolution Scorecard](#3-v013-debt-resolution)
4. [Remaining Structural Debt](#4-remaining-structural-debt)
5. [Roadmap Alignment — Where We Are vs Where We Planned](#5-roadmap-alignment)
6. [Recommended Next Steps](#6-recommended-next-steps)
7. [Summary](#7-summary)

---

## 1. Metrics Evolution

| Metric | v0.1.0 | v0.1.1 | v0.1.3 | v0.1.4 | Delta v0.1.3→v0.1.4 |
|--------|--------|--------|--------|--------|----------------------|
| Source lines (`.rs`) | ~91,000 | ~100,000 | 123,682 | ~128,000 | +4,318 |
| Integration test lines | — | — | — | 13,735 | — |
| Source files | — | — | 200 | 209 | +9 |
| `#[test]` count | 1,982 | ~2,000 | 2,072 | 2,394 | +322 |
| ADRs | — | — | 11 | 11 | 0 |
| Crates (workspace + external) | 10+1 | 10+1 | 10+1 | 10+1 | 0 |
| Clippy warnings | 0 | 0 | 0 | 0 | 0 |
| Duplicate deps | — | — | 1 | 1 | 0 |

**Interpretation.** The +4,318 lines are almost entirely FPHA fitting (~4,631 lines including tests), hydro model preprocessing (~4,502 lines), and deterministic regression tests. The +322 tests include 138 in `fpha_fitting.rs`, 62 in `hydro_models.rs`, and the 12-case D01–D12 regression suite. This is a feature-heavy release with a proportionally high test investment.

### Per-Crate Distribution (v0.1.4)

| Crate | Lines | Role |
|-------|-------|------|
| cobre-sddp | 53,849 | Algorithm implementation (42% of total) |
| cobre-io | 48,933 | I/O, validation, output (38%) |
| cobre-stochastic | 10,610 | PAR fitting, scenario generation |
| cobre-core | 9,572 | Shared types, system model |
| cobre-cli | 7,690 | CLI commands, progress, summary |
| cobre-solver | 4,818 | HiGHS bindings, solver interface |
| cobre-comm | 3,938 | MPI/local communication backend |
| cobre-python | 2,320 | Python bindings |
| cobre-mcp | 25 | Stub |

---

## 2. What v0.1.4 Delivered

### 2.1 FPHA Hydro Production Model

This is the headline feature. The four-piece hyperplane approximation models variable-head hydroelectric plants where power output depends nonlinearly on both turbined flow and reservoir head. Two modes are supported:

- **Precomputed hyperplanes** — supplied via `fpha_hyperplanes.parquet`, for users who have their own fitting pipeline
- **Computed from geometry** — fitted from forebay volume-elevation and tailrace flow-elevation curves in `hydro_geometry.parquet`, which is the production workflow

The fitting pipeline in `fpha_fitting.rs` (4,631 lines, 138 tests) generates least-squares hyperplanes from the production function surface. This is preprocessing-only code — it runs once before training, not on the hot path.

**LP builder impact:** FPHA adds hyperplane cut constraints and segment bounds per FPHA-enabled hydro. The backward pass extracts FPHA/evaporation duals for cut coefficient computation. The indexer was extended with FPHA and evaporation index maps.

### 2.2 Evaporation Linearization

Reservoir surface evaporation modeled as a linearized function of stored volume, with per-season evaporation reference volumes for seasonal variation. Evaporation variables and constraints are integrated into the LP water balance. This interacts with FPHA because the reservoir level affects both the production function and the evaporation rate.

### 2.3 Deterministic Regression Test Suite

12 hand-computed test cases (D01–D12) covering: thermal dispatch, single hydro, cascade, transmission, FPHA (constant head, variable head, computed), evaporation, multi-deficit, and inflow non-negativity. Each case has an analytically derived expected cost. These are regression anchors — if the LP formulation changes, the expected cost will change and the test will catch it. This is an important maturity step. Before this suite, correctness was validated indirectly through convergence behavior. Now there are fixed-point assertions.

### 2.4 CLI Modularization (Carried from v0.1.3 Debt)

The monolithic `run.rs` (1,956 lines in v0.1.3) was restructured:

| Component | v0.1.3 | v0.1.4 | Status |
|-----------|--------|--------|--------|
| `commands/run.rs` | 1,956 (monolithic) | 828 | Extracted broadcast, gathering |
| `commands/broadcast.rs` | — | 439 | MPI broadcast machinery |
| `execute()` function | embedded in 1,956 | ~407 | Functional but not at target |

The target from the v0.1.3 assessment was "under 120 lines" for `execute()`. At 407 lines, it's a 4x reduction from the monolithic version but still carries significant orchestration weight. The inherent MPI rank-0-vs-non-root branching, the progress thread lifecycle, and simulation gathering all live in `execute()`. Given MPI's intrinsic complexity, 407 is a defensible size — the important thing is that the *broadcast data construction* and *simulation result gathering* are now in dedicated functions with clear boundaries.

---

## 3. v0.1.3 Debt Resolution Scorecard

The v0.1.3 pre-release assessment identified 24 tasks across four categories. Here is their resolution status in v0.1.4.

### Short-Term Debt (D1–D7)

| # | Task | Status | Evidence |
|---|------|--------|----------|
| D1 | `stage_ctx()` / `training_ctx()` accessors on `StudySetup` | **Done** | Both methods present in `setup.rs` with tests |
| D2 | Deduplicate `DEFAULT_*` constants to `cobre-sddp::setup` | **Done** | All three constants in `setup.rs`, re-exported via `lib.rs` |
| D3 | `StudyParams` struct for config dedup | **Done** | `StudyParams::from_config()` used by both `StudySetup::new()` and `BroadcastConfig` |
| D4 | `convert_scenario` → `From<SimulationScenarioResult>` | **Done** | `conversion.rs:205`, used by both CLI and Python |
| D5 | Replace `eprintln!` in stochastic export with `Term::stderr()` | **Done** | Zero `eprintln!` in cobre-sddp production code |
| D6 | `WelfordAccumulator::count()` fix | **Done** | (incorporated in prior release) |
| D7 | `build_stochastic_summary` in cobre-sddp | **Done** | `stochastic_summary.rs` in cobre-sddp, callable from CLI and Python |

**All 7 items resolved.** The config-to-domain conversion duplication — the most structurally concerning item from v0.1.3 — is cleanly eliminated by `StudyParams`.

### Python Parity (P1–P5)

| # | Task | Status | Evidence |
|---|------|--------|----------|
| P1 | Estimation in Python `run_inner()` | **Done** | `prepare_stochastic()` handles estimation; called from Python |
| P2 | User-supplied opening tree in Python | **Done** | `prepare_stochastic()` loads user tree when present |
| P3 | Stochastic artifact export in Python | **Done** | `export_stochastic_artifacts_py()` in `run.rs` |
| P4 | `StochasticSummary` in Python result dict | **Done** | `build_stochastic_summary()` called, returned via `stochastic_summary_to_dict()` |
| P5 | Factor pre-setup into shared function | **Done** | `prepare_stochastic()` and `prepare_hydro_models()` in cobre-sddp, used by both CLI and Python |

**All 5 items resolved.** The Python parity gap flagged as "widening" in the v0.1.3 pre-release assessment is now closed. `run_inner()` is at 595 lines (up from the v0.1.3 post-extraction 125 lines, but this includes the full simulation pipeline, stochastic export, and hydro model summary — it's doing more, not regressing structurally).

### CLI Modularization (E1–E4)

| # | Task | Status | Evidence |
|---|------|--------|----------|
| E1 | Extract broadcast machinery to `broadcast` module | **Done** | `commands/broadcast.rs` (439 lines) |
| E2 | Extract simulation result allgatherv | **Done** | `gather_simulation_results()` in `run.rs` |
| E3 | Extract output writing branches | **Partial** | `write_outputs()` exists but `execute()` still at 407 lines |
| E4 | Move test code from `run.rs` to integration tests | **Partial** | Only 2 `#[test]` remain in `commands/run.rs`; bulk tests in `tests/cli_run.rs` |

### Release Prep (R1–R6)

| # | Task | Status |
|---|------|--------|
| R1 | Populate CHANGELOG | **Done** — v0.1.3 and v0.1.4 entries present |
| R2 | Version bump to 0.1.4 | **Done** — workspace version is `0.1.4` |
| R3 | CLAUDE.md update | **Done** — "Current State (v0.1.4)" section accurate |
| R4 | Fix test count | **Done** — 2,624 in CLAUDE.md (matches your assessment) |
| R5 | Roadmap in book | **Done** — v0.1.1 through v0.1.4 deliverables listed |
| R6 | Fix docs/ROADMAP.md | **Done** — Truncation listed correctly |

**Overall: 21 of 24 items fully resolved, 3 partially resolved.** The remaining partial items (E3, E4) are minor — `execute()` at 407 lines is not a problem, and the CLI test extraction is cosmetic.

---

## 4. Remaining Structural Debt

### 4.1 Hot-Path Allocation Opportunities (From Your Assessment)

Your v0.1.4 assessment correctly identified four allocation sites in the training hot path. I agree with all four findings and your priority assignments. Reiterating here for the consolidated view:

| Priority | Site | Impact |
|----------|------|--------|
| P1 | Coefficient vector clone in `collect_local_cuts_for_stage` (training.rs:137) | Largest per-cut allocation. Every cut's `Vec<f64>` is cloned for sync. |
| P1 | Sort in backward pass (backward.rs:504) | O(n log n) per stage per iteration. Likely redundant given contiguous index ranges. |
| P2 | Uniform probabilities vec per stage (backward.rs:477) | Per-stage `Vec<f64>` allocation. Constant across iterations. |
| P2 | Active cuts collect per stage (backward.rs:482–485) | Per-stage `Vec<usize>` allocation. Grows over iterations. |

**When to address these:** Not now. These are optimization opportunities for when profiling on production-scale systems (165 hydros, 120 stages, 2000 iterations) shows that allocation is a meaningful fraction of total time. The sort replacement (P1) is nearly free to verify and could be done opportunistically, but the coefficient clone restructuring requires changing the `sync_cuts` interface and should be done alongside the first real-scale performance campaign.

### 4.2 Infrastructure Crate Genericity Drift (From Your Assessment)

**"Benders" in documentation (7 occurrences across cobre-io and cobre-solver):** These leak through FlatBuffers schema cross-references. The fix is simple — replace "Benders cuts" with "cutting planes" or "cuts" in doc comments. The FlatBuffers table name `BendersCut` can stay since it lives in the application layer.

**"Forward pass" / "backward pass" in types and docs (20+ occurrences in cobre-core):** The `TrainingEvent` enum uses `ForwardPassComplete` / `BackwardPassComplete` and `time_forward_ms` / `time_backward_ms`. This is baked into the core event model. The rename to phase-generic terminology (`EvaluationPhaseComplete` / `RefinementPhaseComplete`) is the right fix but is a breaking change for event consumers (CLI progress, Python bindings).

**When to address:** The "Benders" doc comments can be fixed anytime (30 minutes of work). The `TrainingEvent` rename should be bundled with a minor version bump since it changes the public API of cobre-core.

### 4.3 `train()` Free Function — 14 Arguments

The internal `train()` function in `training.rs` has 14 parameters and retains `#[allow(clippy::too_many_arguments)]`. This was at 25 in v0.1.1 and 21 in the v0.1.3 progress check. At 14, it includes 2 generic type parameters (`S`, `C`), 5 context/state references, a closure, and bookkeeping args. The public API through `StudySetup::train()` has 6 args — the 14-arg signature is internal only.

This is close to irreducible given the generic type parameters and the closure. The remaining path to reduce it would be bundling `opening_tree`, `risk_measures`, `stopping_rules`, `cut_selection`, and `shutdown_flag` into a single "training run spec" struct. Whether that improves clarity is debatable — the current signature reads linearly and each argument is used. I'd leave this alone.

### 4.4 `from_broadcast_params()` — Construction Complexity

`StudySetup::from_broadcast_params()` (setup.rs:306) also has `#[allow(clippy::too_many_arguments)]`. This is the non-root MPI rank construction path where parameters arrive via broadcast rather than from config files. The argument count reflects the broadcast payload structure. Since this is construction-time code (called once per run), not hot-path, the complexity is acceptable.

### 4.5 Production Clippy Suppressions

| Suppression | Count (production) | Assessment |
|------------|-------------------|------------|
| `too_many_arguments` | 4 (`train`, `from_broadcast_params`, 2 indexer helpers) | Acceptable — see §4.3 and §4.4 |
| `too_many_lines` | 2 (`train`, `build_stage_template_lp`) | LP builder function is inherently long; train function carries MPI+threading+convergence logic |
| `cast_possible_truncation` / `cast_precision_loss` | ~15 | Inherent to u64/usize/f64 LP solver interface work |

The structural suppressions (`too_many_arguments`, `too_many_lines`) are at their practical minimum. The cast suppressions are interface noise between Rust's type system and LP solver FFI. No action needed.

---

## 5. Roadmap Alignment — Where We Are vs Where We Planned

### 5.1 Definitive Roadmap (2026-03-11) Cross-Reference

| Milestone | Planned | Actual | Status |
|-----------|---------|--------|--------|
| **Immediate fixes** | Introduction.md, schema URLs, README | v0.1.1 | **Done** |
| **v0.1.1** | 4ree example, `cobre summary`, inflow truncation | v0.1.1 | **Done** — plus CVaR, PAR estimation, stochastic load noise |
| **v0.2.0 Track A** | `pip install cobre`, Python book section | **Done** | PyPI published, Python section in book, README updated |
| **v0.2.0 Track B** | `cobre-newave` converter | **Not started** | Converter repo not created |
| **v0.3.0** | CVaR, multi-cut, NUMA, benchmarks | CVaR done (v0.1.1) | Multi-cut, NUMA, benchmarks pending |
| **v0.4.0** | Checkpoint/resume, SPA editor, warm-start | **Not started** | ADR-011 exists for warm-start cuts |

The definitive roadmap's ordering was: distribution first (v0.2.0), then algorithm depth (v0.3.0). What actually happened was mostly the reverse: v0.1.1 through v0.1.4 went deep on algorithm quality (CVaR, FPHA, evaporation, regression tests, code quality) while the NEWAVE converter remained untouched. However, the Python distribution track (Track A) is now complete — `pip install cobre` works and the software book covers Python usage.

### 5.2 Revised Roadmap (2026-03-11) Cross-Reference

| Milestone | Planned | Actual | Status |
|-----------|---------|--------|--------|
| **v0.1.1** | PAR simulation, truncation, scenario hardening, 4ree, `cobre summary` | v0.1.1–v0.1.3 | **Done** (spread across 3 releases) |
| **v0.1.2** | Noise methods (LHS, QMC), checkpoint/resume, external/historical sampling | **Not started** | None of these features implemented |
| **v0.2.0** | Infinite horizon, SIDP upper bound, cobre-bridge | **Not started** | None implemented |

The revised roadmap's v0.1.2 (noise methods + checkpoint) was displaced by v0.1.2 (cut selection wiring, documentation) → v0.1.3 (StudySetup extraction, CLI cleanup) → v0.1.4 (FPHA, evaporation). This displacement was reasonable — FPHA and evaporation are higher-value modeling features than alternative noise methods — but it means the revised roadmap's dependency chain (PAR simulation → noise methods → checkpoint → infinite horizon) hasn't been followed.

### 5.3 What Actually Happened (The Implicit Roadmap)

Looking at the actual release sequence:

```
v0.1.0 — Base SDDP solver, Python bindings, JSON schemas
v0.1.1 — PAR estimation, CVaR, inflow truncation, stochastic load noise, cobre summary
v0.1.2 — Cut selection wiring, documentation accuracy, design ADRs
v0.1.3 — StudySetup extraction, CLI cleanup, noise dedup, stochastic export, user opening trees
v0.1.4 — FPHA, evaporation, hydro model pipeline, deterministic regression suite
```

The pattern is clear: you prioritized **modeling completeness and code quality** over **distribution and alternative algorithms**. Each release either added a modeling feature needed for realistic hydro dispatch (FPHA, evaporation, CVaR) or paid down structural debt (StudySetup, noise dedup, CLI modularization). The PyPI distribution track was completed in parallel — `pip install cobre` works. This is a defensible path — you can't credibly compare against NEWAVE if you don't model variable-head production and evaporation, and you can't attract Python users if the package isn't on PyPI.

### 5.4 The Gap

The gap is entirely on the distribution side:

| Item | Priority (definitive roadmap) | Status |
|------|-------------------------------|--------|
| `pip install cobre` | v0.2.0 Track A — "the adoption release" | **Done** |
| `cobre-newave` converter | v0.2.0 Track B — "the most important thing for adoption" | Not started |
| Python section in software book | v0.2.0 Track A | **Done** |
| Jupyter notebook tutorial | Adoption checklist | Not started |
| Benchmark suite with published results | Credibility checklist | Not started (D01–D12 is internal, not published) |

The PyPI distribution gap is closed — `pip install cobre` works and the software book covers Python usage. The remaining adoption blockers are the NEWAVE converter (which creates new users from the existing Brazilian power systems community), the Jupyter notebook tutorial (which lowers the barrier for Python users who already found the package), and the published benchmark suite (which provides external credibility).

The definitive roadmap explicitly warned: "Don't implement CVaR before Python is on PyPI." CVaR shipped in v0.1.1 and PyPI shipped later — the ordering was inverted, but both are now done. The remaining gap is narrower than it was: it's the converter and the credibility artifacts, not the entire distribution stack.

---

## 6. Recommended Next Steps

### 6.1 Decision Point: v0.1.5 vs v0.2.0

The codebase is in a mature state. The modeling surface covers realistic hydro dispatch (constant productivity, FPHA, evaporation, cascade coupling, multi-bus transmission, inflow non-negativity, CVaR risk). The code quality is high — the deterministic regression suite, the 2,394 tests, zero clippy warnings, and clean architecture demonstrate this. And with PyPI live, `pip install cobre` works today.

The question is: **what comes next — more features or the remaining distribution pieces?**

I recommend a **distribution-and-validation-focused v0.2.0** as the next milestone. The algorithm is mature enough to produce meaningful results on realistic cases, and Python users can already install the package. What's missing is the bridge from NEWAVE users to Cobre (the converter) and the external credibility artifacts (benchmarks, Jupyter tutorial). Continuing to add features (noise methods, checkpoint, infinite horizon) before demonstrating Cobre against NEWAVE is engineering depth without reach.

### 6.2 Recommended v0.2.0 Scope

**Track A — Remaining Python Adoption (3 days)**

PyPI is live and the software book covers Python usage. What remains:

| Task | Effort | Deliverable |
|------|--------|-------------|
| Jupyter notebook tutorial | 1 day | End-to-end Python workflow: load case, run, analyze results with polars/matplotlib |
| Arrow zero-copy result loading | 1 day | Return results as Arrow RecordBatches; `polars.from_arrow()` for free |
| Example notebook in repo | 0.5 day | `examples/notebooks/quickstart.ipynb` checked into the repo |

**Track B — NEWAVE Converter (2 weeks)**

| Task | Effort | Deliverable |
|------|--------|-------------|
| Create `cobre-rs/cobre-bridge` repo | 0.5 day | pyproject.toml, depends on inewave |
| Entity conversion (HIDR→hydros, TERM→thermals, CONFHD→cascade, SISTEMA→buses/lines) | 3 days | Core mapping layer |
| Temporal conversion (NEWAVE dates→stages, load files→demand) | 2 days | Complete temporal structure |
| Stochastic conversion (inflow history→parquet, PAR coefficients→parquet) | 2 days | Full scenario pipeline |
| CLI: `cobre-bridge convert newave <SRC> <DST>` | 0.5 day | One-command conversion |
| Validation against known NEWAVE case | 1 day | Published convergence bound comparison |

**Track C — Published Benchmark (1 week, parallelizable with Track B)**

| Task | Effort | Deliverable |
|------|--------|-------------|
| Benchmark runner script | 0.5 day | Automated wall-time + bound-quality measurement |
| 1dtoy baseline | 0.5 day | Reference result, Cobre vs sddp-lab |
| 4ree baseline | 1 day | Reference result, Cobre vs sddp-lab, multi-thread scaling |
| NEWAVE case comparison | 2 days | Cobre vs NEWAVE on converted case |
| Results page in software book | 1 day | Public-facing credibility artifact |

**Total: ~3.5 weeks for all three tracks (partially parallelizable).** Track A (3 days) can run in parallel with Track B (2 weeks). Track C (1 week) depends on Track B for the NEWAVE case comparison but the 1dtoy/4ree baselines can start earlier.

### 6.3 What to Defer

Following the principle "distribution before algorithm depth":

- **Noise methods (LHS, QMC)** — Defer to v0.3.0. SAA is sufficient for meaningful comparisons.
- **Checkpoint/resume** — Defer to v0.3.0. Only needed for very long HPC runs.
- **Infinite horizon** — Defer to v0.3.0. Brazilian hydro dispatch uses finite horizon.
- **SIDP upper bound** — Defer to v0.3.0. Statistical UB from simulation is sufficient for now.
- **Multi-cut formulation** — Defer to v0.3.0+. Single-cut converges fine for the validation cases.
- **Hot-path allocations** — Defer until first production-scale profiling session.
- **TrainingEvent rename** — Defer to v0.3.0, bundle with a breaking-change release.

### 6.4 The Only Features Worth Squeezing Into v0.2.0

If you need a modeling feature before the distribution push, the only candidate is **multi-segment deficit pricing**. NEWAVE uses tiered deficit costs (segment depths × costs), and if your converter produces a case that uses multi-segment deficit, Cobre will silently use a single deficit variable and produce different results. This is the one gap that could undermine a NEWAVE comparison.

Check whether the NEWAVE cases you plan to convert use multi-segment deficit. If they do, implement it before v0.2.0. If they don't (many small cases don't), defer it.

### 6.5 Minor Cleanup (Can Be Done Anytime, Low Effort)

| Task | Effort | Impact |
|------|--------|--------|
| Replace "Benders" with "cutting plane" in infra crate doc comments | 30 min | Genericity compliance |
| Verify backward pass sort is redundant, replace with `debug_assert!` | 1 hour | Free hot-path optimization |

---

## 7. Summary

### What's Strong

The codebase is in the best shape it's ever been. The trajectory from v0.1.1 (25-argument functions, duplicated noise logic, dissolved context) to v0.1.4 (clean context structs, shared preprocessing pipeline, 12 deterministic regression tests, FPHA, evaporation) is impressive. Every prior assessment's critical findings have been addressed. The Python parity gap that was "widening" in v0.1.3 is fully closed. The CLI is modularized. The test coverage is thorough and structurally sound — the D01–D12 regression suite provides fixed-point correctness guarantees that didn't exist before. And `pip install cobre` works — the Python distribution track is complete.

### What's Missing

Two things: the NEWAVE converter and the credibility artifacts. The converter creates new users from the existing Brazilian power systems community — every NEWAVE user becomes a potential Cobre user overnight. The credibility artifacts (published benchmarks, Jupyter tutorial) lower the barrier for users who find the package and need to trust it before investing time. These are the remaining blockers between a mature solver and a usable product.

### The Recommendation

Ship v0.2.0 as a validation-and-adoption release. The NEWAVE converter is the centerpiece (~2 weeks). Surround it with the Jupyter tutorial, Cobre-vs-sddp-lab baselines, and a Cobre-vs-NEWAVE comparison on a converted case. The engineering foundation is solid, the modeling surface is rich, and Python distribution is live. Further algorithm features (noise methods, infinite horizon, checkpoint) add depth for power users, but the project needs reach first. Get it in front of the Brazilian power systems community — the code speaks for itself once people can run it on their own cases.

---

**End of Assessment.**