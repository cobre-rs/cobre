# Refined Assessment: Cobre v0.3.2

**Date:** 2026-04-02
**Baseline assessment:** `claude-web-assessment.md` (Claude Web, same date)
**Refinement by:** Claude Code (deep codebase verification)
**HEAD:** `f60d6da` (v0.3.2)

---

## 1. Corrections to the Baseline Assessment

The baseline assessment was ~90% accurate. Three claims were factually wrong,
three had minor inaccuracies, and one architectural detail was missed entirely.

### 1.1 Factual Errors

**Error 1 — Item #8 (retry-level histogram): marked unresolved, but exists.**

The baseline says "❌ Still aggregate counts only." This is wrong.
`retry_level_histogram: [u64; 12]` exists on `SolverStatistics`
(`cobre-solver/src/types.rs:198`), is computed in `SolverStatsDelta::from_snapshots`
(`cobre-sddp/src/solver_stats.rs:97-108`), and is written to Parquet as 12 flat
columns `retry_l0`..`retry_l11` (`cobre-io/src/output/schemas.rs:300-311`).

However, the implementation has a design problem — see Section 3.1.

**Error 2 — Item #12 (`validate_referential_integrity`): marked as 742-line monolith, but already split.**

The baseline says "❌ Still 742 lines in one function." The function is actually
35 lines at `cobre-io/src/validation/referential.rs:34-68` — a dispatcher that
delegates to 13 `check_*_references` helpers. The total _file_ is 2,090 lines.
The baseline confused file size with function size.

**Error 3 — Item #6 (FphaConfig location): claims two locations, only one exists.**

The baseline says `FphaConfig` exists in both `cobre-sddp/indexer.rs` AND
`cobre-io/extensions/production_models.rs`. It only exists in
`cobre-io/extensions/production_models.rs:141`. The rename is still worth doing
but is simpler than described.

### 1.2 Minor Inaccuracies

| Claim                          | Baseline | Actual  | Delta       |
| ------------------------------ | -------- | ------- | ----------- |
| CI jobs                        | 10       | 11      | Off by 1    |
| `template.rs` production lines | 565      | 564     | Off by 1    |
| Total codebase lines           | 176,267  | 176,627 | +360 (0.2%) |

### 1.3 Omission — Top-10 File List

The baseline's top-10 production files missed
`cobre-io/src/validation/semantic.rs` at 4,091 lines, which ranks 4th overall.

### 1.4 Missed Architectural Detail — Cut Selection Parallelization

The baseline analyzes `select_dominated` as O(|active cuts| x |visited states|)
per stage and warns about the cost at ONS scale. This analysis is correct
per-stage, but misses that the selection loop is **parallelized via rayon**
across stages (`training.rs:623-633`):

```rust
let deactivations: Vec<(usize, DeactivationSet)> = (1..num_sel_stages)
    .into_par_iter()
    .map(|stage| {
        let pool = &fcf.pools[stage];
        let states = archive_ref.map_or(&[] as &[f64], |a| a.states_for_stage(stage));
        let deact = strategy.select_for_stage(pool, states, iteration, stage as u32);
        (stage, deact)
    })
    .collect();
```

At ONS scale (117 stages), the O(n^2) per-stage cost is distributed across all
available cores. Wall-clock impact is significantly less than the raw FLOP
estimate suggests.

### 1.5 Overstated Concern — "Flying Blind" on Domination Timing

The baseline says "you're flying blind on whether this is 50ms or 5s per check."
This is too strong. Aggregate `selection_time_ms` already exists
(`training.rs:656`), emitted in `CutSelectionComplete`. The first ONS run will
tell us the total wall-clock cost.

What IS missing is **per-stage timing** within the selection — if a few stages
with many cuts dominate the total, we won't know which ones.
`StageSelectionRecord` currently has per-stage cut counts but no per-stage
wall-clock field.

---

## 2. Corrected Resolution Scorecard: 13 of 17

The baseline reported 11/17. With the corrections above, the actual count is
**13 of 17** items addressed from the v0.2.2 assessment.

### Resolved (13)

| #   | Item                                                           | Evidence                                                                                         |
| --- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1   | CLAUDE.md version                                              | `check_claudemd_version.py` passes                                                               |
| 2   | Suppression threshold `--max 10`                               | CLAUDE.md + `scripts/pre-commit`                                                                 |
| 3   | Quality scripts in CI                                          | `quality-scripts` CI job                                                                         |
| 4   | Split `lp_builder.rs`                                          | 6 files: mod (202), layout (481), matrix (1,313), patch (1,016), scaling (411), template (8,987) |
| 5   | Eliminate coefficient clones                                   | `pack_local_cuts` zero-copy serialization                                                        |
| 7   | Pre-allocate backward pass buffers                             | `probabilities_buf` and `successor_active_slots_buf` as `&mut Vec` on `BackwardPassSpec`         |
| 8   | Retry-level histogram                                          | `retry_level_histogram: [u64; 12]` on `SolverStatistics` (types.rs:198)                          |
| 10  | Sparse/dense equivalence test                                  | `tests/sparse_dense.rs` (250 lines)                                                              |
| 11  | `cobre-python` pytest suite                                    | 48 tests across 7 files (773 lines)                                                              |
| 12  | `validate_referential_integrity` split                         | 35-line dispatcher + 13 `check_*_references` helpers                                             |
| 13  | Absorb `cut_selection` + `shutdown_flag` into `TrainingConfig` | `train()` went from 12 to 10 params                                                              |
| 14  | Extract `output/mod.rs` write functions                        | `results_writer.rs` (684 lines), `mod.rs` 1,264 to 642                                           |
| 15  | CLAUDE.md currency                                             | v0.3.2 matches Cargo.toml                                                                        |

### Remaining (4)

| #   | Item                         | Status                                                                      |
| --- | ---------------------------- | --------------------------------------------------------------------------- |
| 6   | Rename `FphaConfig`          | Only in `cobre-io/extensions/production_models.rs` (not in indexer.rs)      |
| 9   | Unify sparse/dense paths     | Both paths exist in `forward.rs:312-355`, guarded by `sparse_dense.rs` test |
| 16  | Run ONS teste benchmark      | Pending                                                                     |
| 17  | Decide incremental injection | Blocked on #16                                                              |

### Excluded from count (items #15 and #6 from baseline differ from above numbering)

- **cobre-mcp / cobre-tui timeline (#15 in baseline):** Still stubs. Both are
  `todo!()`/empty libs with zero dependencies. Not counted as resolved or
  remaining — this is a product decision, not a code issue.

---

## 3. New Findings — Issues Not in the Baseline

### 3.1 Retry Histogram: Implicit Coupling on Magic Number `12`

The retry-level histogram was implemented (correcting the baseline's claim), but
the implementation creates an implicit dependency chain that will break when a
second solver backend (CLP) is added.

**Dependency chain:**

```
cobre-solver (origin — generic interface)
  types.rs:198           — pub retry_level_histogram: [u64; 12]
  highs.rs:534           — let num_retry_levels = 12_u32;  (HiGHS-specific)
  highs.rs:1076          — stats.retry_level_histogram[outcome.level as usize] += 1;

cobre-sddp (consumer)
  solver_stats.rs:62     — pub retry_level_histogram: [u64; 12]
  solver_stats.rs:98     — let mut h = [0u64; 12];

cobre-io (output)
  solver_stats_writer.rs:55   — pub retry_level_histogram: [u64; 12]
  solver_stats_writer.rs:136  — (0..12).map(...)
  schemas.rs:300-311           — 12 hardcoded Field::new("retry_l0") ... "retry_l11"

cobre-cli + cobre-python (passthrough)
  run.rs                       — retry_level_histogram: delta.retry_level_histogram
```

**Problem:** `SolverStatistics` is part of the generic `SolverInterface` trait
contract. `[u64; 12]` bakes a HiGHS implementation detail into the abstraction.
CLP may have 5 or 6 retry levels. The Parquet output adds 12 mostly-zero columns
to every output file.

**Recommended fix:**

1. `SolverStatistics.retry_level_histogram` becomes `Vec<u64>`. Each solver impl
   decides its own level count. HiGHS pushes 12 entries, CLP pushes 5.
2. `SolverStatsDelta` mirrors: `Vec<u64>`. Delta computation zips to the shorter
   side.
3. Parquet output uses a single `List<UInt64>` column `retry_histogram` instead
   of 12 flat columns. Polars/pandas read list columns natively.
4. Schema drops the 12 `retry_l*` fields, replaced by one list field.

Result: 28 Parquet columns become 17. No code outside `highs.rs` knows how many
retry levels exist. Zero cost on the hot path (stats accumulation is per-solve,
not per-iteration).

### 3.2 `architecture-rules.md` Is Stale

The budget table at `.claude/architecture-rules.md:49` shows:

```
| train | 12 | 12 | training.rs | at budget |
```

Actual: 10 parameters. Both `Max args` and `Current` should be updated to 10.

### 3.3 Visited States Unconditional Allocation — Intentional but Wasteful

The baseline correctly identifies the ~4 GB unconditional allocation. The code
comment at `training.rs:303-305` reveals this is intentional:

```rust
// Visited-states archive: always allocated so forward-pass trial points
// are recorded for analysis and export regardless of cut selection method.
// Dominated cut selection also reads from this archive at pruning time.
```

The "analysis and export" rationale is weak: `exports.states` defaults to
`false`, and no analysis path reads the archive unless dominated selection is
active. The allocation should be gated on either `Dominated` variant or
`exports.states = true`.

### 3.4 Per-Stage Domination Timing Missing from `StageSelectionRecord`

`StageSelectionRecord` (`cobre-core/src/training_event.rs:62-73`) has per-stage
**cut counts** but no per-stage **wall-clock timing**:

```rust
pub struct StageSelectionRecord {
    pub stage: u32,
    pub cuts_populated: u32,
    pub cuts_active_before: u32,
    pub cuts_deactivated: u32,
    pub cuts_active_after: u32,
    // No selection_time_ms field
}
```

The rayon `par_iter` closure at `training.rs:625-632` already runs per-stage.
Adding `Instant::now()` / `elapsed()` inside the closure and recording
`selection_time_ms` on `StageSelectionRecord` gives per-stage granularity at
near-zero cost. This replaces the baseline's recommendation for a separate
`domination_check_ms` field on backward pass instrumentation — per-stage timing
on the selection event is more useful and architecturally cleaner.

---

## 4. Confirmed Findings from the Baseline

These claims were verified as accurate and need no correction:

- **10 `too_many_arguments` suppressions** — exact via `check_suppressions.py`
- **`train()` signature** — 10 params, character-for-character match
- **`TrainingConfig` fields** — `cut_selection` and `shutdown_flag` confirmed
- **lp_builder 6-file split** — all line counts match within 1 line
- **output/mod.rs reduction** — 1,264 to 642, `results_writer.rs` at 684
- **Visited states flat `Vec<f64>` layout** — `StageStates` with
  cache-friendly row-major storage
- **`select_dominated` O(n^2)** — loop structure, early exits,
  current-iteration protection, `is_candidate` filter all verified
- **Policy export de-duplication** — `build_stage_cut_records` +
  `build_active_indices` shared by CLI and Python
- **`compute_effective_eta` extraction** — in `noise.rs`, called from
  `lower_bound.rs` directly, `forward.rs`/`backward.rs` indirectly
- **Discount factors in `StageContext`** — both `discount_factors` and
  `cumulative_discount_factors` as `&[f64]`
- **cobre-python tests** — 48 functions, 7 files, 773 lines (all exact)
- **D01-D25 test suite** — 23 deterministic + 2 cut selection
- **cobre-mcp/cobre-tui are stubs** — `todo!()` / empty libs, zero deps
- **BackwardPassSpec buffer pre-allocation** — `probabilities_buf` and
  `successor_active_slots_buf` as `&mut Vec`, reused via `clear()` +
  `resize()`/`extend()`
- **Sparse/dense paths guarded** — both paths in `forward.rs:312-355`,
  `sparse_dense.rs` test prevents drift
- **No `FphaConfig` rename** — still `FphaConfig` in `cobre-io`
- **ONS benchmark not run** — no benchmark data or scripts in repo
- **Crate-level line counts** — within 0.2% of claimed totals

---

## 5. Revised Priority List

### Tier 0 — Before the ONS Benchmark (< 2 hours total)

| #   | Item                                                                       | Effort | Rationale                                                                                                                                                                                                                     |
| --- | -------------------------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Gate visited states archive** on `Dominated` variant or `exports.states` | 30 min | Prevents ~4 GB wasted allocation at ONS scale when neither dominated selection nor state export is active                                                                                                                     |
| 2   | **Add `selection_time_ms` to `StageSelectionRecord`**                      | 20 min | The rayon `par_iter` closure already runs per-stage — wrap with `Instant::now()` / `elapsed()`. Gives per-stage granularity for diagnosing domination cost at ONS scale                                                       |
| 3   | **Decouple retry histogram from magic `12`**                               | 1 hr   | `SolverStatistics` uses `Vec<u64>` instead of `[u64; 12]`. Parquet output uses `List<UInt64>` column instead of 12 flat columns. Eliminates implicit HiGHS coupling before CLP arrives. Reduces Parquet from 28 to 17 columns |
| 4   | **Update `architecture-rules.md`** — `train` row: 12 to 10, "at target"    | 5 min  | Housekeeping — budget table is stale                                                                                                                                                                                          |

### Tier 1 — Quality Polish (1 session, independent of ONS)

| #   | Item                                                                                    | Effort | Rationale                                                                                                              |
| --- | --------------------------------------------------------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------- |
| 5   | Rename `FphaConfig` to `FphaColumnLayout` in `cobre-io/extensions/production_models.rs` | 15 min | Prevents confusion with FPHA fitting config. One location only                                                         |
| 6   | Decide `cobre-mcp` / `cobre-tui` — keep or exclude from workspace                       | 5 min  | Both are `todo!()` stubs adding noise to `cargo check --workspace`. Either timeline them or move outside the workspace |

### Tier 2 — Strategic (determines production readiness)

| #   | Item                                    | Effort                   | Rationale                                                                                                                                                                                                                                                                       |
| --- | --------------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 7   | **Run ONS teste benchmark with v0.3.2** | 2-4 hrs                  | The single most important next step. Answers: (a) is dominated cut selection worth its O(n^2) cost with rayon parallelization? (b) does Level1 suffice? (c) are discount factors + operational violations correct at scale? Tier 0 fixes should land first for full diagnostics |
| 8   | Multi-rank cut sync integration test    | Blocked on MPI CI runner | True multi-rank correctness validation. Infrastructure dependency                                                                                                                                                                                                               |

### Items Removed from the Baseline Plan

| Baseline Item                                          | Why Removed                                                                                                                   |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| Retry-level histogram (baseline #8)                    | Already implemented — replaced by the decoupling fix (Tier 0 #3)                                                              |
| `validate_referential_integrity` split (baseline #12)  | Already split into 35-line dispatcher + 13 helpers                                                                            |
| `domination_check_ms` on backward pass instrumentation | Replaced by per-stage `selection_time_ms` on `StageSelectionRecord` (Tier 0 #2)                                               |
| Sparse/dense unification (baseline #9)                 | Guarded by test, acceptable as-is. Unifying requires benchmarking to prove no perf regression — not worth the risk before ONS |

### Items Kept Unchanged

| Baseline Item                                         | Why Kept                                                             |
| ----------------------------------------------------- | -------------------------------------------------------------------- |
| Gate visited states (baseline Tier 0 #1)              | Still the top fix — verified unconditional at `training.rs:303-311`  |
| ONS benchmark (baseline Tier 2 #7)                    | Still the strategic priority — no change                             |
| MPI integration test (baseline Tier 2 #8)             | Still infrastructure-blocked — no change                             |
| `FphaConfig` rename (baseline Tier 1 #4)              | Simpler than described (one location, not two) but still worth doing |
| `cobre-mcp`/`cobre-tui` decision (baseline Tier 2 #9) | Still stubs, still needs a decision                                  |

---

## 6. Summary Scorecard

| Dimension                         | Baseline Claim    | Verified State                               | Correction                                   |
| --------------------------------- | ----------------- | -------------------------------------------- | -------------------------------------------- |
| Resolution rate                   | 11/17             | **13/17**                                    | +2 (retry histogram + referential integrity) |
| `too_many_arguments` suppressions | 10                | 10                                           | None                                         |
| `train()` params                  | 10                | 10                                           | None                                         |
| Retry histogram                   | Missing           | **Exists, but coupled to `[u64; 12]`**       | Implementation has design debt               |
| `validate_referential_integrity`  | 742-line monolith | **35-line dispatcher + 13 helpers**          | Already split                                |
| `FphaConfig` locations            | 2 (indexer + io)  | **1 (io only)**                              | Simpler rename                               |
| Cut selection parallelization     | Not mentioned     | **Rayon `par_iter` across stages**           | Baseline missed this                         |
| Domination timing                 | "Flying blind"    | **Aggregate `selection_time_ms` exists**     | Per-stage timing missing, not all timing     |
| `architecture-rules.md`           | Not checked       | **Stale** (`train` shows 12, actual 10)      | New finding                                  |
| CI jobs                           | 10                | 11                                           | Off by 1                                     |
| Parquet column count              | Not analyzed      | **28 (12 are retry noise)**                  | New finding                                  |
| Top-10 files                      | 10 listed         | **Missed `semantic.rs` at #4 (4,091 lines)** | Omission                                     |
