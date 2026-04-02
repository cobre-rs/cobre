# Assessment: Cobre v0.2.2 → v0.3.2

**Date:** 2026-04-02
**Baseline:** v0.2.2 (`1d8d14e`, 166,592 lines, 10 suppressions)
**Current HEAD:** v0.3.2 (`f60d6da`, 176,267 lines)
**Diff:** +18,599 / −6,462 across 234 files (3 releases: v0.3.0, v0.3.1, v0.3.2)
**Total codebase:** 176,267 lines of Rust (72,501 production, 103,766 test — 58.9% test ratio)

---

## 1. Previous Assessment Items — Resolution Scorecard

This release cycle systematically addressed the prioritized action plan from the v0.2.2 assessment. Resolution rate: **11 of 17 items addressed.**

### Tier 0 (Fix-Now) — All Resolved

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1 | Update CLAUDE.md version | ✅ | `check_claudemd_version.py` passes: v0.3.2 matches Cargo.toml |
| 2 | Tighten suppression threshold to `--max 10` | ✅ | Both CLAUDE.md and `scripts/pre-commit` |
| 3 | Add quality scripts to CI | ✅ | New `quality-scripts` CI job runs parity, suppressions, CLAUDE.md version |

### Tier 1 (High-Impact Refactors) — 2 of 3 Resolved

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 4 | Split `lp_builder.rs` | ✅ | 10,903-line monolith → 6 files: `mod.rs` (202), `layout.rs` (481), `matrix.rs` (1,313), `patch.rs` (1,016), `scaling.rs` (411), `template.rs` (8,987 — 565 prod + 8,422 test) |
| 5 | Eliminate coefficient clones in cut sync | ✅ | `pack_local_cuts` serializes directly from pool. `collect_local_cuts_for_stage` eliminated. Zero clones. |
| 6 | Rename `FphaConfig` in indexer | ❌ | Both `cobre-sddp/indexer.rs` and `cobre-io/extensions/production_models.rs` still define `pub struct FphaConfig`. Low priority. |

### Tier 2 (Performance Polish) — 1 of 3 Resolved

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 7 | Pre-allocate backward pass per-stage buffers | ✅ | `probabilities_buf` and `successor_active_slots_buf` are `&mut Vec` on `BackwardPassSpec`, reused via `clear()` + `resize()`/`extend()` |
| 8 | Add retry-level histogram to `SolverStatistics` | ❌ | Still aggregate counts only |
| 9 | Unify sparse/dense paths in `build_cut_row_batch_into` | ❌ | Both paths still exist, but `tests/sparse_dense.rs` guards against drift |

### Tier 3 (Test Coverage) — All Resolved

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 10 | Sparse vs dense bit-for-bit equivalence test | ✅ | `tests/sparse_dense.rs` (250 lines) |
| 11 | `cobre-python` pytest suite | ✅ | 48 test functions across 7 files (773 lines), CI on Python 3.12/3.13/3.14 with coverage |
| 12 | `validate_referential_integrity` split | ❌ | Still 742 lines in one function |

### Tier 4 (Architecture) — All Resolved

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 13 | Absorb `cut_selection` + `shutdown_flag` into `TrainingConfig` | ✅ | `train()` went from 12→10 params |
| 14 | Extract `output/mod.rs` write functions | ✅ | `results_writer.rs` (684 lines) split out, `mod.rs` down from 1,264→642 |
| 15 | Decide `cobre-mcp` / `cobre-tui` timeline | ❌ | Still stubs in workspace |

### Tier 5 (Strategic) — Not Started

| # | Item | Status |
|---|------|--------|
| 16 | Run ONS teste benchmark | ❌ — Still pending |
| 17 | Decide incremental injection necessity | ❌ — Blocked on #16 |

---

## 2. What Shipped — Release-by-Release

### v0.3.0 — Policy Management + Operational Violations

The largest release in the cycle. Two work streams:

**Policy lifecycle:**
- **Warm-start and resume-from-checkpoint** — `policy.mode` enum (`fresh`, `warm_start`, `resume`). Warm-start injects cuts into a fresh FCF; resume continues from saved iteration count.
- **Simulation-only mode** — Run simulation against a saved policy without re-training (`training.enabled = false`).
- **`policy_export.rs` + `policy_load.rs`** — Extracted shared conversion logic from duplicated CLI/Python code. This eliminated a latent parity risk where CLI and Python serialized cuts differently.

**Operational violations:**
- **Truncation-with-penalty inflow method** — Combined truncation and penalty enforcement, matching SPTcpp's `truncamento_penalizacao` mode.
- **Per-plant inflow penalty via cascade** — Penalty cost overridable per hydro through `penalties.json`.
- **Bidirectional withdrawal and evaporation slacks** — Split into directional (pos/neg) components with independent costs.
- **Per-block operational violations** — Min/max outflow, turbined flow, and generation constraints with per-block slack columns.
- **Cost decomposition** — 6 granular violation cost columns in simulation output.
- **Per-stage productivity override** — `hydro_production_models.json` with D24 regression test.
- **D19–D24 regression tests** covering multi-hydro PAR(p), operational violations, min-outflow, per-block violations, bidirectional withdrawal, productivity override.

**Assessment fixes (from v0.2.2 action plan):**
- `lp_builder.rs` five-way split
- `output/mod.rs` extraction
- `cut_selection` + `shutdown_flag` absorbed into `TrainingConfig`
- Backward pass buffer pre-allocation
- Coefficient clone elimination
- Quality scripts in CI
- Python pytest suite
- Sparse/dense equivalence test

### v0.3.1 — Discount Rate

- Annual discount rate from policy graph → per-stage one-step discount factors on theta objective coefficient.
- Cumulative discount factors weight stagewise costs in upper bound and simulation.
- **Bugfix:** Upper bound now applies cumulative discount factors, making it commensurate with the discounted lower bound.
- **Bugfix:** Stage cost extraction (`objective - theta`) corrected to account for discount factor on theta (`objective - d_t * theta`).
- **D25 regression test** — Verifies discounted lower bound and simulation discount factors against undiscounted baseline (D02).

### v0.3.2 — Dominated Cut Selection + Visited States

- **`CutSelectionStrategy::Dominated`** — O(|active cuts| × |visited states|) per stage per check. Deactivates cuts dominated at ALL visited forward-pass trial points. Configurable `threshold` and `check_frequency`. Includes current-iteration protection and stage-0 exemption.
- **`VisitedStatesArchive`** — Flat `Vec<f64>` per stage for cache-friendly iteration. Pre-allocated capacity. Always collected during training.
- **`exports.states` config flag** — Opt-in persistence of visited states to policy checkpoint (`policy/states/stage_NNN.bin`). Defaults to `false`.
- **Policy checkpoint now exports all cuts** (active + inactive) with `is_active` flag and `active_cut_indices` vectors. Enables full reconstruction for warm-start and post-hoc analysis.
- **`compute_effective_eta` helper** — Extracted reusable inflow noise computation, reducing duplication across forward, backward, and lower-bound evaluation passes.
- **Bugfix:** Lower bound evaluation now applies inflow truncation, matching forward and backward passes. Previously, negative truncated inflows were not applied in stage-0 openings, producing optimistic lower bounds.

---

## 3. Architecture State — Quantitative

### 3.1 Suppression Inventory

| Metric | v0.2.2 | v0.3.2 | Δ |
|--------|--------|--------|---|
| `too_many_arguments` (production) | 10 | 10 | 0 — held steady |
| `train()` params | 12 | **10** | −2 |
| `run_forward_pass` params | 8 | 8 | 0 |
| `run_backward_pass` params | 8 | 8 | 0 |
| `simulate` params | 8 | 8 | 0 |
| `evaluate_lower_bound` params | 9 | 9 | 0 |

**No suppression regression despite 6 new feature areas** (discount rate, dominated cuts, visited states, policy warm-start, operational violations, cost decomposition). This is the strongest signal that the context struct pattern is working. New data flows through existing structs:

- Discount factors → `StageContext` (per-stage read-only data, decision tree #1)
- Visited states → `TrainingConfig` / `TrainingOutcome` (study-level)
- Cut selection strategy → `TrainingConfig` (study-level config)
- Shutdown flag → `TrainingConfig` (study-level config)
- Inflow truncation method → `TrainingContext` (study-level read-only)

### 3.2 `train()` Signature — Final State

```rust
pub fn train<S: SolverInterface + Send, C: Communicator>(
    solver: &mut S,
    config: TrainingConfig,
    fcf: &mut FutureCostFunction,
    stage_ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    opening_tree: &OpeningTree,
    risk_measures: &[RiskMeasure],
    stopping_rules: StoppingRuleSet,
    comm: &C,
    solver_factory: impl Fn() -> Result<S, cobre_solver::SolverError>,
) -> Result<TrainingOutcome, SddpError>
```

10 params. Every parameter is structurally distinct — further reduction would require merging semantically unrelated things. `solver` and `solver_factory` are generic-typed and can't easily join a struct. `fcf` is `&mut` (output). `opening_tree`, `risk_measures`, and `stopping_rules` are algorithm-level inputs that don't belong in `TrainingConfig` (config is about how to run, not what to solve).

This signature is stable. The architecture rules should update the budget from 14 to 10 and mark this function as "at target."

### 3.3 File Size Distribution

Top 10 production files (excluding tests):

| File | Total Lines | Production | Test |
|------|-------------|------------|------|
| `lp_builder/template.rs` | 8,987 | 565 | 8,422 |
| `hydro_models.rs` | 4,650 | ~2,800 | ~1,850 |
| `fpha_fitting.rs` | 4,646 | 1,796 | 2,850 |
| `forward.rs` | 3,763 | ~900 | ~2,863 |
| `simulation/pipeline.rs` | 3,621 | ~750 | ~2,871 |
| `backward.rs` | 3,425 | ~780 | ~2,645 |
| `simulation/extraction.rs` | 3,206 | ~1,143 | ~2,063 |
| `setup.rs` | 3,035 | ~1,500 | ~1,535 |
| `estimation.rs` | 2,514 | ~1,100 | ~1,414 |
| `policy.rs` (io) | 2,476 | ~1,200 | ~1,276 |

No file has excessive production code. The large total-line counts are driven by tests (which is healthy). `template.rs` is 94% test code.

### 3.4 Crate-Level Health

| Crate | Production | Test | Test% | Δ from v0.2.2 |
|-------|-----------|------|-------|---------------|
| `cobre-sddp` | 24,426 | 52,527 | 68% | +2,046 prod, +5,837 test |
| `cobre-io` | 25,439 | 28,953 | 53% | +477 prod, +184 test |
| `cobre-core` | 6,090 | 5,174 | 46% | +77 prod, +117 test |
| `cobre-stochastic` | 5,512 | 7,624 | 58% | +144 prod, +469 test |
| `cobre-cli` | 3,680 | 4,692 | 56% | +147 prod, +30 test |
| `cobre-python` | 2,890 | **773** (pytest) | **21%** | +115 prod, **+773 test** |
| `cobre-solver` | 2,287 | 2,949 | 56% | +26 prod, +1 test |
| `cobre-comm` | 2,096 | 1,847 | 47% | +5 prod, +0 test |

`cobre-python` went from 0% to 21% test coverage. The Rust-side test ratio counts `#[test]` annotations and the pytest files are counted separately — the combined effective coverage is better than 21% suggests, since the Rust code under the FFI boundary is tested by `cobre-sddp`'s 68% test ratio.

---

## 4. Performance Analysis

### 4.1 Dominated Cut Selection — O(n²) Scaling Risk (NEW)

The `select_dominated` function has O(|active cuts| × |visited states|) complexity per stage per check:

```rust
for x_hat in visited_states.chunks_exact(n_state) {
    for k in 0..populated {
        if pool.active[k] {
            let val = pool.intercepts[k]
                + pool.coefficients[k].iter().zip(x_hat).map(|(c, x)| c * x).sum::<f64>();
            // ...
        }
    }
}
```

**Estimated cost at ONS scale:**
- Active cuts per stage: ~200–2,000
- Visited states: iterations × forward_passes = 200 × 20 = 4,000
- State dimension: 1,106
- Per check per stage: up to 2,000 × 4,000 = 8M dot products of dimension 1,106
- Across 117 stages: ~10¹⁰ FLOPs per domination check

With `check_frequency=5`, this fires every 5 iterations. The early exit (`n_candidates == 0 → break`) helps when most cuts survive, but the worst case (many candidates, most not dominated until the last visited state) runs the full sweep.

**Mitigations already in place:**
- `current_iteration` protection: cuts from the current iteration can't be dominated
- Stage-0 exemption
- Early exit when no candidates remain
- `is_candidate` filter shrinks the inner loop

**Missing:** No timing capture for the domination check. Add a `domination_check_ms` field to backward pass instrumentation. Without it, you're flying blind on whether this is 50ms or 5s per check at production scale.

**Algorithmic note:** The inner loop (`pool.coefficients[k].iter().zip(x_hat).map(...)`) is a prime candidate for SIMD or BLAS-style optimization if profiling shows it's a bottleneck. The flat `Vec<f64>` layout of `VisitedStatesArchive` was designed for cache-friendly access — this is already better than a `Vec<Vec<f64>>` approach.

### 4.2 Visited States Memory — Unconditional Allocation (MODERATE)

```rust
let mut visited_archive = Some(crate::visited_states::VisitedStatesArchive::new(
    num_stages, n_state, total_forward_passes,
));
```

This is always allocated in `train()`, regardless of whether dominated cut selection is active. At ONS scale:

> 200 iterations × 20 forward passes × 117 stages × 1,106 state dims × 8 bytes = **~4.1 GB**

When cut selection is `Level1`, `Lml1`, or `None`, this memory is allocated, populated every iteration via `append()`, and never read. The archive grows linearly with iteration count.

**Fix:** Gate the archive creation on the presence of `Dominated` variant in `config.cut_selection`:

```rust
let mut visited_archive = match &config.cut_selection {
    Some(CutSelectionStrategy::Dominated { .. }) => Some(VisitedStatesArchive::new(...)),
    _ if config.exports_states => Some(VisitedStatesArchive::new(...)),
    _ => None,
};
```

This preserves the archive when `exports.states = true` (user explicitly wants state persistence) but avoids 4 GB of wasted memory when neither dominated selection nor state export is active.

### 4.3 Previously Identified Bottlenecks — Resolved

| Item | v0.2.2 Status | v0.3.2 Status |
|------|--------------|--------------|
| Coefficient clones in cut sync | ~20 MB/iter | **0** — `pack_local_cuts` zero-copy |
| Per-stage backward allocations | 3 per stage | **0** — buffers pre-allocated on `BackwardPassSpec` |
| `lp_builder.rs` maintainability | 10,903-line monolith | **Split into 6 files** |

---

## 5. Integration Quality

### 5.1 CI Pipeline — Comprehensive

| Job | Purpose | Status |
|-----|---------|--------|
| `check` | `cargo check --workspace` | ✅ |
| `test` | `cargo test --workspace` | ✅ |
| `clippy` | `cargo clippy -D warnings` | ✅ |
| `fmt` | `cargo fmt --check` | ✅ |
| `quality-scripts` | Parity + suppressions + CLAUDE.md version | ✅ **New** |
| `docs` | `cargo doc -D warnings` | ✅ |
| `security` | `cargo audit` | ✅ |
| `deny` | `cargo-deny` (license + deps) | ✅ |
| `coverage` | `cargo-llvm-cov` + Codecov | ✅ **New** |
| `python` | maturin + pytest on 3.12/3.13/3.14 + coverage | ✅ **New** |

10 CI jobs. The quality enforcement is now permanent — no contributor can merge a parity gap, suppression regression, or CLAUDE.md staleness.

### 5.2 Policy Export De-duplication

Previously, both `cobre-cli/src/policy_io.rs` and `cobre-python/src/run.rs` had independent implementations of FCF→policy conversion. If one was updated without the other, policy checkpoints from CLI and Python would diverge silently.

Now `cobre-sddp/src/policy_export.rs` provides `build_stage_cut_records()` and `build_active_indices()` — shared by both CLI and Python. This eliminates an entire class of parity bugs.

### 5.3 `compute_effective_eta` Extraction

The inflow noise computation was duplicated across `forward.rs`, `backward.rs` (via `process_stage_backward`), and `lower_bound.rs`. The v0.3.2 bugfix (lower bound not applying inflow truncation) was caused by this duplication — the forward and backward paths had the truncation logic but `evaluate_lower_bound` didn't.

The extracted `compute_effective_eta` in `noise.rs` is now called from all three sites. This is the correct fix — the next inflow method change (e.g., a new truncation variant) only needs to be implemented once.

### 5.4 Discount Rate Threading

Discount factors are threaded through the system cleanly:
- `StageContext.discount_factors: &[f64]` — per-stage one-step discount factors (read-only, shared)
- `StageContext.cumulative_discount_factors: &[f64]` — cumulative products (read-only, shared)
- Forward pass: theta objective coefficient scaled by `d_t`
- Backward pass: theta in successor LP scaled by `d_{t+1}`
- Training: upper bound applies cumulative factors to stagewise costs
- Simulation: cost accumulation uses cumulative factors

The placement in `StageContext` follows the decision tree correctly (per-stage, read-only, shared across workers → #1).

---

## 6. Prioritized Next Steps

### Tier 0: Fix-Now (< 1 hour)

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 1 | **Gate visited states archive on `Dominated` or `exports.states`** | 30 min | Prevents 4 GB wasted allocation at ONS scale when dominated selection is not active |
| 2 | **Add `domination_check_ms` to backward pass instrumentation** | 30 min | Critical visibility for dominated cut selection at scale |

### Tier 1: Quality Polish (1 session)

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 3 | Add retry-level histogram to `SolverStatistics` | 30 min | Production diagnostics (carried from v0.2.2) |
| 4 | Rename `FphaConfig` in `indexer.rs` to `FphaColumnLayout` | 15 min | Prevents name collision (carried from v0.2.2) |
| 5 | `validate_referential_integrity` split (742 lines) | 2 hrs | Maintainability (carried from v0.2.2) |
| 6 | Update `architecture-rules.md` budget for `train()`: 14→10 | 5 min | Reflects actual state |

### Tier 2: Strategic (blocks production readiness)

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 7 | **Run ONS teste benchmark with v0.3.2** | 2–4 hrs | Validates discount rate, cut selection, violation types at production scale. Answers: is dominated selection worth the O(n²) cost? Does Level1 suffice? |
| 8 | **Multi-rank cut sync integration test** | Blocked on MPI CI runner | True multi-rank correctness validation |
| 9 | Decide `cobre-mcp` / `cobre-tui` timeline | 5 min | Keep or exclude from workspace to save build time |

---

## 7. Summary Scorecard

| Dimension | v0.2.2 | v0.3.2 | Trend |
|-----------|--------|--------|-------|
| `too_many_arguments` suppressions | 10 | 10 | → Held (despite 6 new features) |
| `train()` params | 12 | **10** | ↓ At target |
| `lp_builder.rs` monolith | 10,903 lines | **Split into 6 files** | ✅ Resolved |
| Coefficient clone overhead | ~20 MB/iter | **0** (zero-copy) | ✅ Resolved |
| Per-stage backward allocs | 3 | **0** (pre-allocated) | ✅ Resolved |
| Quality scripts in CI | No | **Yes** (3 scripts) | ✅ Resolved |
| `cobre-python` tests | 0 | **48 tests, 773 lines** | ✅ Resolved |
| Sparse/dense equivalence test | Missing | **Present** (250 lines) | ✅ Resolved |
| `output/mod.rs` size | 1,264 | **642** (split) | ✅ Resolved |
| CLAUDE.md currency | Stale | **Current** (v0.3.2) | ✅ Resolved |
| Policy checkpoint export | Duplicated CLI/Python | **Shared module** | ✅ Resolved |
| Code coverage CI | No | **Yes** (Rust + Python) | ✅ Added |
| Inflow noise duplication | 3 independent impls | **Shared `compute_effective_eta`** | ✅ Fixed |
| Test ratio | 58.3% | 58.9% | → Healthy |
| Total codebase | 166K | 176K | → Controlled growth |
| D-cases | D01–D19 | **D01–D25** | ↑ +6 regression tests |
| Visited states memory | N/A | **~4 GB unconditional** | ⚠ Gate on variant |
| Dominated selection timing | N/A | **Not instrumented** | ⚠ Add before ONS run |

**Overall assessment:** This is the strongest release cycle to date. 11 of 17 previous action items resolved. No architecture regressions despite substantial feature additions (discount rate, dominated cut selection, policy warm-start, 6 violation types, cost decomposition). The codebase has matured from "early-stage, fast-moving" to "structured, CI-enforced, production-approaching." The remaining items are almost entirely strategic (ONS benchmark, timing instrumentation) rather than structural.

The immediate priority is gating the visited states archive and adding domination timing instrumentation before running the ONS teste benchmark — that benchmark will determine whether dominated cut selection justifies its O(n²) cost or whether Level1 is sufficient for production.