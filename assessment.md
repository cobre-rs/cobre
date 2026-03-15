# Cobre v0.1.1 — Post-Release Assessment & Design Analysis

**Date:** 2026-03-13  
**Scope:** Both repositories (cobre v0.1.1 + cobre-docs), documentation cleanup, code quality audit, reproducibility analysis, and four forward-looking design topics  
**Base document:** `definitive-roadmap.md` (2026-03-11)

---

## Table of Contents

1. [Documentation Audit — Software Book (`cobre/book/`)](#1-documentation-audit--software-book)
2. [Documentation Audit — Methodology Reference (`cobre-docs/`)](#2-documentation-audit--methodology-reference)
3. [Crate Abstraction Leaks — Cross-Crate Naming Analysis](#3-crate-abstraction-leaks--cross-crate-naming-analysis)
4. [Code Quality & HPC Maintainability — The Degradation Pattern](#4-code-quality--hpc-maintainability--the-degradation-pattern)
5. [CRITICAL: Training/Simulation Mismatch and Unwired Features](#5-critical-trainingsimulation-mismatch-and-unwired-features)
6. [Reproducibility Across Parallelization Strategies](#6-reproducibility-across-parallelization-strategies)
7. [Design: User-Supplied Opening Trees](#7-design-user-supplied-opening-trees)
8. [Design: Exporting Internally-Computed Stochastic Artifacts](#8-design-exporting-internally-computed-stochastic-artifacts)
9. [Design: Complete Tree Execution and Work Distribution](#9-design-complete-tree-execution-and-work-distribution)
10. [Design: Per-Stage Warm-Start Cuts and End-of-World Boundary Conditions](#10-design-per-stage-warm-start-cuts-and-end-of-world-boundary-conditions)
11. [FPHA: Scope Assessment and NEWAVE Comparison Roadmap](#11-fpha-scope-assessment-and-newave-comparison-roadmap)
12. [Consolidated Task List](#12-consolidated-task-list)

---

## 1. Documentation Audit — Software Book

### 1.1 DEC-XYZ References

The software book contains **14 references to DEC-XYZ decision identifiers** scattered across four crate documentation pages (`crates/stochastic.md` ×4, `crates/comm.md` ×4, `crates/solver.md` ×3, `crates/sddp.md` ×1). These belong to the old cobre-docs spec numbering system, superseded by the ADR system in `docs/adr/`. **Action:** Replace each with a plain-language description. The rationale is already in the surrounding prose.

### 1.2 Outdated Status Badges

All book crate pages say `<span class="status-experimental">experimental</span>`. The README says `alpha` for all crates except `cobre-python`. **Action:** Update all book badges to match the README.

### 1.3 Outdated Roadmap in README

The README roadmap lists `cobre-python` as unimplemented under v0.2, and v0.3 lists "Newton-Raphson AC power flow" and "DC power flow and OPF" — items from a previous draft not in the definitive roadmap. **Action:** Rewrite to match the definitive roadmap.

### 1.4 Outdated Content in `reference/roadmap.md`

Lists "CVaR risk measure" as future (already implemented), references nonexistent `docs/ROADMAP.md`. **Action:** Rewrite to reflect v0.1.1.

### 1.5 Missing 4ree Example Page

`SUMMARY.md` lists only `1dtoy` under Examples. The 4ree case has no book page. **Action:** Create `book/src/examples/4ree.md`.

### 1.6 Outdated Crate Page Content

- **`crates/stochastic.md`:** References `evaluate_par_inflow` naming (§3). Test count "220 tests" unverified post-v0.1.1.
- **`crates/sddp.md`:** Step 7 says checkpoints are deferred — they're shipped. Version string `0.0.1`. Test count "590 tests" unverified.
- **`crates/overview.md`:** CLI description says `run/validate/report/compare/serve` — actual: `run`, `validate`, `report`, `init`, `schema`, `summary`, `version`. Mermaid graph missing `io → stochastic` edge.
- **`crates/comm.md`:** TCP and SHM described as "reserved" — should be "deferred."

---

## 2. Documentation Audit — Methodology Reference

### 2.1 DEC-XYZ References in specs/

The `specs/` section retains ~20 DEC-XYZ references. Lower priority since specs is labeled "(Internal)." Keep if the decision log is maintained; remove if abandoned.

### 2.2 Roadmap Pages

Accurate against v0.1.1. No stale references.

### 2.3 Missing BibTeX Citation

The introduction lacks the `@techreport` citation block from the documentation migration plan.

---

## 3. Crate Abstraction Leaks — Cross-Crate Naming Analysis

### 3.1 The Core Problem

`cobre-stochastic` leaks hydro-specific naming into generic stochastic infrastructure:

| Current name | Problem | Better name |
|---|---|---|
| `evaluate_par_inflow()` | PAR evaluates a value, not necessarily "inflow" | `evaluate_par()` |
| `evaluate_par_inflows()` | Batch version | `evaluate_par_batch()` |
| `PrecomputedParLp` | "Lp" suffix implies solver coupling | `PrecomputedPar` or `ParCache` |
| `PrecomputedNormalLp` | Same "Lp" problem | `PrecomputedNormal` or `NormalCache` |

### 3.2 Recommended Renames

**Phase 1 (non-breaking):** Add aliases, deprecate old names, fix docstrings.
**Phase 2 (next minor):** Rename types, remove aliases.
**Phase 3:** Update all documentation.

### 3.3 Other Boundaries — Clean

All other crate boundaries (`cobre-solver`, `cobre-comm`, `cobre-core`, `cobre-io`) are clean after systematic audit. The stochastic crate is the only one with naming leaks.

---

## 4. Code Quality & HPC Maintainability — The Degradation Pattern

This is the most structurally important finding in the audit.

### 4.1 The Core Problem: Parameter Explosion

| Function | Arguments at v0.1.1 |
|---|---|
| `train()` | **25** |
| `run_forward_pass()` | **20** |
| `run_backward_pass()` | **20** |
| `simulate()` | **22** |

Every feature addition (inflow truncation, stochastic load noise, load balance patching) contributed 3-5 new parameters threaded through the entire call chain. `#[allow(clippy::too_many_arguments)]` and `#[allow(clippy::too_many_lines)]` are used 8+ times across the hot-path modules.

### 4.2 What's Specifically Wrong

**4.2.1 Dissolved context.** The forward pass receives `noise_scale`, `n_hydros`, `n_load_buses`, `load_balance_row_starts`, `load_bus_indices`, `block_counts_per_stage` as 6 separate arguments. All are fields of `StageTemplates`. The CLI `run.rs` destructures the struct, passes the pieces to `train()`, which passes them to `run_forward_pass()`, which passes them into the worker closure. Three layers of manual destructuring for no benefit.

**4.2.2 Noise transformation duplicated THREE times.** The "convert raw η to RHS patch value" block is copy-pasted between:

1. `forward.rs` lines 536-610 (with truncation path)
2. `backward.rs` lines 288-311 (without truncation)
3. `simulation/pipeline.rs` lines 308-331 (without truncation)

The load noise computation (another ~15 lines) is identically triplicated across all three files. Any change to noise handling requires updating three files identically. This is not just a maintenance burden — it has already produced a correctness bug (see §5).

**4.2.3 `SolverWorkspace` has become a bag of scratch buffers.** 11 fields, of which 7 are `Vec<f64>` scratch buffers added incrementally for specific features. None are logically related to the solver.

**4.2.4 Simulation pipeline is a parallel codebase.** `simulation/pipeline.rs` (2,552 lines) is essentially a fork of the training forward pass with extraction logic bolted on. It shares no code with `forward.rs` despite having the identical LP rebuild sequence, noise transformation, warm-start basis logic, and patch sequence.

**4.2.5 Three orchestration sites.** The pipeline assembly logic (building templates, indexer, stochastic context, threading 25 parameters to `train()` and 22 to `simulate()`) is duplicated in:
- `cobre-cli/src/commands/run.rs` (471 lines, the CLI entry point)
- `cobre-python/src/run.rs` (488 lines, the Python entry point)
- Any future entry point (TUI, MCP) will require a third copy

Both currently hardcode the same bugs (see §5).

**4.2.6 Monolith functions.** `build_stage_templates` (491 lines) and `extract_stage_result` (467 lines) both have `#[allow(clippy::too_many_lines)]`. The LP builder handles column layout, row layout, coefficient assembly, bounds, objectives, water balance, load balance, transmission, and penalty slacks in one function. FPHA will add another ~100 lines to it.

### 4.3 The HPC Impact

**Stack frame pressure.** 20 arguments = 320+ bytes of pointer/length pairs per frame. Rayon closures capturing 15+ references create equally large capture structs.

**Optimizer inhibition.** LLVM handles 20+ argument functions less aggressively. Register pressure forces spills. The noise transformation loop reads from 4 separate pointer arguments.

**Cache coherence.** The 11-field `SolverWorkspace` (264 bytes) exceeds two cache lines. Accessing `ws.solver` then `ws.noise_buf` straddles boundaries.

**Instruction cache duplication.** Three copies of the noise transformation means three copies in L1 icache (32 KB per core on Zen 4), displacing solver hot-path instructions.

### 4.4 The Refactoring Plan

**Phase 1 — `StageContext` bundle (2-3 days):**

```rust
pub struct StageContext<'a> {
    pub templates: &'a [StageTemplate],
    pub base_rows: &'a [usize],
    pub noise_scale: &'a [f64],
    pub n_hydros: usize,
    pub n_load_buses: usize,
    pub load_balance_row_starts: &'a [usize],
    pub load_bus_indices: &'a [usize],
    pub block_counts_per_stage: &'a [usize],
}
```

Collapses 8 arguments into 1 across all hot-path functions.

**Phase 2 — Noise transformation extraction (1-2 days):**

```rust
fn transform_noise_to_patches(
    ws: &mut SolverWorkspace<S>,
    raw_noise: &[f64],
    stage: usize,
    ctx: &StageContext,
    inflow_method: &InflowNonNegativityMethod,
    stochastic: &StochasticContext,
    indexer: &StageIndexer,
)
```

**Eliminates all three copies.** Single source of truth. When truncation is added to backward/simulation (which it must be — see §5), it's implemented once.

**Phase 3 — Workspace restructuring (1 day):**

```rust
pub struct SolverWorkspace<S: SolverInterface> {
    pub solver: S,
    pub patch_buf: PatchBuffer,
    pub current_state: Vec<f64>,
    pub scratch: NoiseScratch,
}

pub(crate) struct NoiseScratch {
    pub noise_buf: Vec<f64>,
    pub inflow_m3s_buf: Vec<f64>,
    pub lag_matrix_buf: Vec<f64>,
    pub par_inflow_buf: Vec<f64>,
    pub eta_floor_buf: Vec<f64>,
    pub zero_targets_buf: Vec<f64>,
    pub load_rhs_buf: Vec<f64>,
    pub row_lower_buf: Vec<f64>,
}
```

**Phase 4 — `TrainingContext` bundle (1 day):**

```rust
pub struct TrainingContext<'a> {
    pub stage_ctx: StageContext<'a>,
    pub horizon: &'a HorizonMode,
    pub risk_measures: &'a [RiskMeasure],
    pub indexer: &'a StageIndexer,
    pub initial_state: &'a [f64],
    pub inflow_method: &'a InflowNonNegativityMethod,
}
```

**Result:**

| Function | Before | After |
|---|---|---|
| `train()` | 25 | ~12 |
| `run_forward_pass()` | 20 | ~10 |
| `run_backward_pass()` | 20 | ~10 |
| `simulate()` | 22 | ~12 |

### 4.5 The Rule Going Forward

Every new feature that adds data to the hot path must answer: *does this belong in an existing context struct, or does it justify a new parameter?* If the answer is "thread it through 4 function signatures as a bare slice," the answer is wrong. The context structs are the firewall against parameter explosion. New fields go into the context; the function signatures don't change.

---

## 5. CRITICAL: Training/Simulation Mismatch and Unwired Features

**This section documents bugs and silent failures that produce wrong results without any error or warning. These must be fixed before any new feature work.**

### 5.1 CRITICAL — Simulation Does Not Apply Inflow Truncation

**Location:** `cobre-sddp/src/simulation/pipeline.rs` line 214

```rust
_inflow_method: &InflowNonNegativityMethod,
```

The leading underscore is the evidence: `_inflow_method` is explicitly unused. The simulation pipeline never applies inflow truncation. A policy trained with `method: "truncation"` (which clamps negative PAR inflows to zero in the forward pass) is simulated with raw unclamped inflows. The simulation can encounter negative inflows that the training forward pass never saw, producing dispatch decisions outside the policy's training domain.

**Impact:** Simulation cost statistics (mean, std, CI) may be systematically biased when truncation is active. The 2000-scenario simulation output — the main deliverable users care about — is computed with a different stochastic model than the one used to train the policy.

**Fix:** The noise transformation extraction (§4.4 Phase 2) fixes this automatically: once `transform_noise_to_patches()` is a shared function that accepts `inflow_method`, the simulation calls it with the same method as training. Until the extraction is done, the simulation pipeline must manually apply the same truncation logic as the forward pass.

### 5.2 CRITICAL — CVaR Risk Measure Is Implemented But Unreachable

**Locations:**
- `cobre-cli/src/commands/run.rs` line 505
- `cobre-python/src/run.rs` line 571

Both entry points hardcode:
```rust
let risk_measures = vec![RiskMeasure::Expectation; n_stages];
```

The per-stage `StageRiskConfig` from the loaded `System` (which supports `CVaR { alpha, lambda }`) is completely ignored. A user setting `risk_config: { type: "cvar", alpha: 0.5, lambda: 0.5 }` in their `stages.json` gets risk-neutral Expectation silently. No error, no warning.

The algorithm implementation (`cobre-sddp/src/risk_measure.rs`) is complete with 16 tests. The `StageRiskConfig` enum in `cobre-core` has the CVaR variant. The stages JSON parser reads it. Everything works except the final wiring step.

**Impact:** CVaR is a mandatory feature for any serious comparison with NEWAVE, which uses CVaR on all stages. The algorithm code is done — this is a 15-minute fix.

**Fix:** In both `run.rs` files, replace the hardcoded vector with:
```rust
let risk_measures: Vec<RiskMeasure> = system.stages()
    .iter()
    .filter(|s| s.id >= 0)
    .map(|s| match &s.risk_config {
        StageRiskConfig::Expectation => RiskMeasure::Expectation,
        StageRiskConfig::CVaR { alpha, lambda } => RiskMeasure::CVaR {
            alpha: *alpha,
            lambda: *lambda,
        },
    })
    .collect();
```

### 5.3 CRITICAL — Cut Selection Is Implemented But Not Wired

**Location:** `cobre-cli/src/commands/run.rs` line 555

The CLI passes `None` for the `cut_selection` parameter to `train()`. The `CutSelectionStrategy` code exists in `cobre-sddp/src/cut_selection.rs` with complete implementation and tests (Level1, Lml1 strategies), but there is no config path to enable it. Users cannot prune inactive cuts.

**Impact:** Without cut selection, the cut pool grows linearly with iterations. For production runs (50+ iterations × 200 forward passes), this means 10,000+ cuts per stage, most of which are inactive. LP solve time scales with cut count. This is a performance bug that becomes a wall at production scale.

**Fix:** Add a `cut_selection` section to `config.json` schema, parse it in both CLI and Python entry points, and pass the resolved strategy to `train()`.

### 5.4 CRITICAL — FPHA Returns Zero Generation Without Warning

**Location:** `cobre-sddp/src/lp_builder.rs` line 1015

```rust
HydroGenerationModel::Fpha => 0.0, // FPHA not supported in v0.1.0
```

An FPHA hydro in the input produces a valid LP that solves with zero generation from that plant. The user gets physically wrong results (the hydro contributes nothing to the system) with no error or warning. The input validation pipeline does not flag FPHA as unsupported.

**Impact:** Any case imported from NEWAVE (where all 165 hydros use FPHA) would produce meaningless results. The silent failure is the worst kind — the software runs to completion and writes output files that look normal.

**Immediate fix (before FPHA implementation):** Add a validation error or at minimum a loud warning when any hydro uses `HydroGenerationModel::Fpha`. The case should fail at validation time, not produce wrong results silently.

### 5.5 MODERATE — Backward Pass Does Not Apply Truncation

**Location:** `cobre-sddp/src/backward.rs` lines 288-294

The backward pass uses the non-truncation noise transformation path regardless of `inflow_method`. This is less critical than the simulation mismatch (§5.1) because the backward pass operates on the opening tree noise (not forward-pass sampled noise), but it creates a theoretical inconsistency: the cut coefficients are computed at points that may include negative inflows even when the training forward pass would have clamped them.

**Impact:** Potentially suboptimal cuts when truncation is active. The cut approximation at negative-inflow regions is never visited by the forward pass, so wasted computational effort. The convergence bound is still valid (the cuts are still valid Benders cuts), but the approximation quality may be slightly reduced.

**Fix:** Addressed automatically by the noise transformation extraction (§4.4 Phase 2).

### 5.6 Summary of Critical Fixes

| # | Issue | Severity | Fix effort | Dependency |
|---|-------|----------|-----------|------------|
| B1 | Simulation ignores inflow truncation | **CRITICAL** | 1 day (manual) or free with Q2 | None |
| B2 | CVaR unreachable from CLI and Python | **CRITICAL** | 15 min per entry point | None |
| B3 | Cut selection not wired to config | **CRITICAL** | 1 day (config schema + wiring) | None |
| B4 | FPHA silent zero generation | **CRITICAL** | 30 min (validation error) | None |
| B5 | Backward pass skips truncation | MODERATE | Free with Q2 | Q2 (noise extraction) |

**These fixes must be completed before ANY new feature development.** Shipping a version where training and simulation use different stochastic models is a correctness failure, not a missing feature. The CVaR and cut selection wiring are features that are already implemented and tested — they just need the last 15 minutes of plumbing to become available to users.

---

## 6. Reproducibility Across Parallelization Strategies

### 6.1 Current State

Strong within-rank determinism: SipHash seeds, static partitioning, sorted cut merge. Existing determinism tests verify bit-identical results between 1 and 4 workspaces.

### 6.2 The Multi-Rank Problem

`sync_forward` uses `allreduce(Sum)` on `[cost_sum, cost_sum_sq, scenario_count]`. Different rank counts produce different partial sum groupings. Floating-point addition is not associative. The **upper bound** will diverge at the bit level across MPI configurations. The **lower bound is safe** — computed by rank 0 alone.

### 6.3 Vulnerability Points

1. **Worker-local cost summation** (`forward.rs` lines 700-722): partial sums differ with different worker counts
2. **`sync_forward` allreduce** (`forward.rs` lines 156-158): grouping across ranks differs

Backward pass `pi_dot_x` and risk measure aggregation are deterministic (fixed iterator order). Not vulnerable.

### 6.4 Mitigation: Canonical Summation Order

Instead of reducing partial sums, `allgatherv` individual scenario costs and sum in global scenario index order. One `allgatherv` of `forward_passes` doubles per iteration. Negligible cost, guaranteed bit-identical UB.

### 6.5 Tasks

1. Empirical verification: 4ree in 3 configs, compare `convergence.parquet` bit-by-bit
2. Implement canonical summation if UB diverges (almost certain)
3. Extend `mpi_smoke.sh` for bit-identical comparison
4. Add multi-rank mock test to `determinism.rs`

---

## 7. Design: User-Supplied Opening Trees

### 7.1 Context

The opening tree is always generated internally. Users can supply scenario data but not backward-pass noise realizations. This matches the override pattern for other scenario inputs.

### 7.2 Design

Add optional `scenarios/noise_openings.parquet`. Schema: `stage_id` (i32), `opening_index` (u32), `entity_index` (u32), `value` (f64). When present, bypass internal generation.

### 7.3 Forward/Backward Separation

No additional mechanism needed. `sample_forward()` already indexes into `OpeningTree` regardless of how it was populated.

### 7.4 Effort

~2.5 days.

---

## 8. Design: Exporting Internally-Computed Stochastic Artifacts

### 8.1 The Problem

The estimation pipeline is a black box. When the user supplies only `inflow_history.parquet`, Cobre fits parameters and generates the opening tree — nothing is persisted. The user cannot inspect, reuse, compare, or debug the fitted model.

### 8.2 What Gets Exported

**Fitted model (conditional on estimation):** `output/stochastic/inflow_seasonal_stats.parquet`, `inflow_ar_coefficients.parquet`, `correlation.json`, `fitting_report.json`.

**Runtime infrastructure (always):** `output/stochastic/noise_openings.parquet`, `load_seasonal_stats.parquet`.

### 8.3 The Round-Trip Invariant

Every exported artifact uses the exact same schema as the corresponding input file. Copy output to input directory → Cobre loads it directly, bypassing estimation. This closes the loop with the user-supplied opening tree design from §7.

### 8.4 When Export Happens

After stochastic context is built, before training. Controlled by `--export-stochastic` flag. Default off in v0.1.x, on in v0.2+.

### 8.5 Effort

~5 days.

---

## 9. Design: Complete Tree Execution and Work Distribution

### 9.1 The Problem

In complete tree (C.12), node count grows exponentially. The static `forward_passes / n_ranks` partition doesn't work.

### 9.2 Architecture

Stage-by-stage work distribution with barrier synchronization and per-stage `allgatherv` of states.

### 9.3 Effort

3-4 weeks. Implementation depends on the code quality refactoring being complete.

---

## 10. Design: Per-Stage Warm-Start Cuts and End-of-World Boundary Conditions

### 10.1 The Generalization

Any stage can start with any number of cuts. The last stage can have cuts (DECOMP boundary condition). Change `FutureCostFunction::new` to accept per-stage counts. Modify terminal-stage theta handling.

### 10.2 Effort

~5 days. Overlaps with checkpoint/resume.

---

## 11. FPHA: Scope Assessment and NEWAVE Comparison Roadmap

### 11.1 FPHA Is Not a Simple Feature

After reviewing the CEPEL official documentation, FPHA has two substantial phases:

**Phase A — Hyperplane Fitting (preprocessing, 3-4 weeks):**

The hydroelectric production function is nonlinear: `P = ρ · g · η_turbine · h_net(V, Q) · Q`, where net head depends on storage volume (cota-volume polynomial), total outflow (tailwater polynomial), and hydraulic losses. The FPHA approximates this nonlinear surface with tangent hyperplanes.

The fitting requires:
1. Physical input data: cota-volume polynomial, tailwater polynomial, loss coefficient, turbine efficiency curves, number of generating units, unit capacities
2. Grid sampling: evaluate the nonlinear FPH at a grid of `(volume, turbined_flow)` operating points
3. Hyperplane computation: at each grid point, compute the tangent plane coefficients via partial derivatives of the head function
4. Hyperplane selection: from potentially hundreds of candidates, select a minimal subset that provides a good piecewise-linear outer approximation
5. Per-stage adaptation when plant physical characteristics change

This is comparable in scope to the entire PAR fitting pipeline.

**Phase B — LP Integration (2 weeks):**

Each hyperplane becomes a constraint: `P_h ≤ a₀ + a₁·V + a₂·Q_turb + a₃·Q_vert`. With 165 hydros × ~4 hyperplanes × 3 blocks = ~2000 additional constraints per stage LP. This changes:

- `build_stage_templates` (the 491-line monolith — must be decomposed first)
- `StageIndexer` (FPHA rows carry duals that affect `n_dual_relevant` and cut coefficients)
- Backward pass dual extraction (state dimension changes)
- `extract_stage_result` (the 467-line monolith — must extract FPHA-specific results)

**Revised estimate: 6-7 weeks total**, not the 2-3 weeks initially projected.

### 11.2 NEWAVE Production Comparison — Gap Analysis

Target NEWAVE configuration: 165 hydros, 130 thermals, 5 buses, 120 stages × 3 blocks, FPHA, CVaR, 200 forward passes, MPI across 3 nodes, 2000 simulation scenarios, selective sampling, PAR fitting from history.

| Feature | NEWAVE | Cobre status | Blocker? |
|---|---|---|---|
| 165 hydros, 130 thermals | ✓ | ✓ No structural limit | No |
| 5 buses with exchange | ✓ | ✓ Already in 4ree | No |
| 120 stages × 3 blocks | ✓ | ✓ Supported | No |
| FPHA production function | ✓ | **Stub returns 0.0** | **YES — physically meaningless without it** |
| CVaR risk measure | ✓ | **Implemented but not wired** | **YES — 15-minute fix** |
| Cut selection | ✓ | **Implemented but not wired** | YES — performance wall at scale |
| 200 forward passes, MPI | ✓ | ✓ Supported | No |
| 2000 simulation scenarios | ✓ | ✓ Supported | No |
| Selective sampling | ✓ | **Not implemented** (only SAA) | No — SAA with more openings is valid |
| PAR fitting from history | ✓ | ✓ v0.1.1 estimation pipeline | No |
| Seasonal correlations | ✓ | ✓ Cholesky per season | No |
| Inflow truncation | ✓ | ✓ Forward pass only (**simulation broken**) | YES — must fix |

### 11.3 Honest Timeline to Credible Comparison

```
Weeks 1-2:   Code quality refactoring (Q1-Q5)
              + CRITICAL bug fixes (B1-B4): truncation, CVaR, cut selection, FPHA warning
              + Reproducibility verification (R1)

Weeks 3-4:   FPHA hyperplane fitting module
              (geometry functions, grid sampling, selection)

Weeks 5-6:   FPHA LP integration
              (builder decomposition, indexer, duals, extraction)

Week 7:      FPHA validation against reference hyperplanes

Weeks 8-9:   cobre-newave conversion + case conversion

Week 10:     Run comparison, write results document
```

**10 weeks to a credible NEWAVE comparison.** Not 7. The FPHA fitting procedure alone is as complex as the PAR estimation pipeline that took several weeks.

The comparison document would acknowledge remaining methodology differences (SAA vs. selective sampling, standard PAR vs. PAR-A) while demonstrating that the LP formulation, risk measure, and production function produce equivalent results on the same physical system.

---

## 12. Consolidated Task List

### URGENT — Critical Bug Fixes (2-3 days)

**These must be fixed before ANY other work. They produce wrong results silently.**

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| B1 | Simulation ignores inflow truncation — train/test mismatch | **CRITICAL** | Apply truncation in simulation noise transform; permanent fix via Q2 |
| B2 | CVaR hardcoded to Expectation in CLI and Python | **CRITICAL** | Read `StageRiskConfig` from system, map to `RiskMeasure` |
| B3 | Cut selection not wired to config | **CRITICAL** | Add config schema section, parse, pass to `train()` |
| B4 | FPHA silently returns zero generation | **CRITICAL** | Add validation error when `Fpha` variant is used before implementation |

### Code Quality Refactoring (5-7 days)

**Must happen before any new features are added to the hot path.**

| # | Task | Effort |
|---|------|--------|
| Q1 | Create `StageContext` struct, collapse 8 args into 1 | 2-3d |
| Q2 | Extract shared `transform_noise_to_patches()` — eliminates 3-way duplication AND fixes B1/B5 | 1-2d |
| Q3 | Split `SolverWorkspace` into solver + scratch sub-structs | 1d |
| Q4 | Create `TrainingContext` bundle, reduce `train()` to ~12 args | 1d |
| Q5 | Re-enable `clippy::too_many_arguments` lint | 0.5d |

### Documentation Fixes (1-2 days)

| # | Task | Effort |
|---|------|--------|
| D1 | Remove all DEC-XYZ references from book crate pages | 1h |
| D2 | Update status badges experimental → alpha | 30m |
| D3 | Rewrite README roadmap | 30m |
| D4 | Rewrite `reference/roadmap.md` | 30m |
| D5 | Create `examples/4ree.md` book page | 2h |
| D6 | Fix `crates/overview.md` CLI subcommand list and dependency graph | 15m |
| D7 | Fix `crates/sddp.md` checkpoint description, version | 30m |
| D8 | Add BibTeX citation to cobre-docs introduction | 10m |
| D9 | Verify/update test counts in stochastic.md and sddp.md | 30m |

### Abstraction Cleanup (1-2 days)

| # | Task | Effort |
|---|------|--------|
| A1 | Add `evaluate_par`/`evaluate_par_batch` aliases, deprecate `_inflow` names | 2h |
| A2 | Update docstrings to say "PAR model" not "PAR inflow model" | 1h |
| A3 | Plan `PrecomputedParLp` → `PrecomputedPar` rename for next minor | Design |
| A4 | Update book crate pages | 1h |

### Reproducibility (3-5 days)

| # | Task | Effort |
|---|------|--------|
| R1 | Empirical test: 4ree in 3 configs, compare convergence.parquet | 1d |
| R2 | Implement canonical UB summation via allgatherv | 2d |
| R3 | Extend mpi_smoke.sh for bit-identical comparison | 0.5d |
| R4 | Multi-rank mock test in determinism.rs | 1d |

### Design — New Features (ADRs)

| # | Task | Effort |
|---|------|--------|
| F1 | User-supplied opening tree spec | 1d |
| F2 | Stochastic artifact export spec | 1d |
| F3 | Complete tree work distribution spec | 1d |
| F4 | Per-stage warm-start + boundary cuts spec | 1d |

### Mandatory Ordering

```
IMMEDIATE (before anything else):
  B1-B4: Critical bug fixes — these produce wrong results NOW

Week 1-2:
  Q1-Q5: Code quality refactoring (the structural firewall)
  D1-D9: Documentation fixes (parallel)

Week 3:
  A1-A4: Abstraction cleanup
  R1-R2: Reproducibility verification + fix

Week 4:
  R3-R4: Reproducibility CI tests
  F1-F4: Design ADRs for upcoming features

Weeks 5-10:
  FPHA implementation (on clean foundation)
  → NEWAVE comparison
```

The bug fixes (B1-B4) are sequenced first because they affect result correctness. The code quality refactoring (Q1-Q5) is second because every subsequent feature will add data to the hot path — without context structs, each feature adds 2-4 more arguments to every function. FPHA in particular will add hyperplane table references, FPHA row offsets, and production function mode flags that would become 4-5 bare-slice parameters without the `StageContext`. The refactoring is not optional — it is the prerequisite for implementing FPHA without making the degradation irreversible.

---

**End of Assessment.**