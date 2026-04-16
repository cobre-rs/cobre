# Post-v0.4.2 Assessment Report

**Date:** 2026-04-15
**Scope:** All changes since v0.4.2 (50 commits, 131 source files, +18,230 / -1,747 lines)
**Branch:** `feat/temporal-resolution-cd` (includes v0.4.3, v0.4.4, and unreleased work)

---

## Summary

The mathematical and algorithmic implementation is **correct** -- the SDDP-level
analysis found zero correctness bugs across angular pruning, frozen-lag semantics,
multi-resolution PAR aggregation, boundary cuts, and the three-stage cut management
pipeline. All 3,600+ tests pass.

However, the rapid feature velocity has created structural debt in four areas:

1. **God-module growth** -- `setup.rs` at 6,138 lines mixing 5+ responsibilities
2. **Hot-path allocations** -- per-stage heap allocations in the backward pass rayon closure
3. **Lint hygiene decay** -- stale `dead_code` suppressions, `too_many_arguments` at ceiling
4. **Documentation drift** -- CHANGELOG duplication, stale version references, broken rustdoc links

None are blocking. Several will compound if left unaddressed across the next feature cycle.

---

## 1. Architectural Degradations

### A1. `setup.rs` god-module (CRITICAL)

**Location:** `crates/cobre-sddp/src/setup.rs` (6,138 lines, 72 `pub` items)

**Problem:** The file mixes at least 5 distinct responsibilities:

| Responsibility                                                | Approximate lines |
| ------------------------------------------------------------- | ----------------- |
| Config-to-domain conversion (`StudyParams::from_config`)      | 134-264           |
| LP template construction (`from_broadcast_params`)            | 514-1360          |
| Accessor/getter methods (42 methods)                          | 1362-1960         |
| Stochastic preprocessing orchestration (`prepare_stochastic`) | 2240+             |
| Run orchestration (train/simulate)                            | 1694-1850         |

Every new feature (entity type, scenario source, preprocessing step) forces
modification of this single file. The 90 methods make it difficult to reason about
struct invariants.

**Anti-pattern:** `from_broadcast_params` takes 15 positional parameters with
`#[allow(clippy::too_many_arguments)]` and uses two-phase initialization:

```rust
// Inside from_broadcast_params (line 1352-1354):
budget: None,                    // hardcoded sentinel
basis_padding_enabled: false,    // hardcoded sentinel
export_states: false,            // hardcoded sentinel

// Every caller must then fix up:
setup.budget = budget;                     // setup.rs:482
setup.basis_padding_enabled = padding;     // setup.rs:483
setup.set_budget(bcast_config.budget);     // run.rs:846
```

If a future caller omits the fixup, `budget` silently defaults to `None` (no cap)
and `basis_padding_enabled` to `false` -- plausible but incorrect values that
cause no error, just silently wrong behavior. `StudyParams` already bundles these
fields but `from_broadcast_params` does not accept it.

**Recommendation:** Extract into focused modules:

- `study_params.rs` -- config-to-domain conversion and `StudyParams`
- `study_setup_builder.rs` -- LP template construction (the current `from_broadcast_params`)
- `study_accessors.rs` or keep accessors as an `impl` block in a separate file
- `stochastic_setup.rs` -- `prepare_stochastic` and library loading
- `study_setup.rs` -- thin orchestration shell

Accept `StudyParams` in `from_broadcast_params` to eliminate the two-phase init.

---

### A2. Infrastructure crate genericity violation (HIGH)

**Location:** `crates/cobre-solver/src/trait_def.rs:168`

**Problem:** The doc comment reads:

```rust
/// Called by the SDDP algorithm after each `pad_basis_for_cuts` call.
```

This violates the hard rule: _"Infrastructure crate genericity -- cobre-core,
cobre-io, cobre-solver, cobre-stochastic, cobre-comm must contain zero
algorithm-specific references (no 'sddp', 'SDDP', 'Benders' in types, functions,
or doc comments)."_

**Fix:** Replace with algorithm-neutral wording:

```rust
/// Called after each basis padding operation to record padding statistics.
```

---

### A3. `TrainingConfig` structural bloat (HIGH)

**Location:** `crates/cobre-sddp/src/config.rs:94-214`

**Problem:** `TrainingConfig` has 15 fields spanning three concerns:

| Concern              | Fields                                                                                          |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| Loop control         | `forward_passes`, `max_iterations`, `start_iteration`, `convergence_tol`                        |
| Event infrastructure | `event_sender`, `checkpoint_interval`, `shutdown_flag`                                          |
| Cut management       | `cut_selection`, `angular_pruning`, `budget`, `basis_padding_enabled`, `cut_activity_tolerance` |

The no-`Default` design prevents silent misconfiguration in production, but every
test must specify all 15 fields. Six test functions in the same file each repeat the
same 15-field literal. Adding one field requires touching every test site.

**Recommendation:** Group into sub-structs (`LoopConfig`, `CutManagementConfig`,
`EventConfig`) inside `TrainingConfig`. Tests can then use `..Default::default()`
on individual sub-structs while the parent remains non-Default.

---

### A4. Stale `#![allow(dead_code)]` suppressions (MEDIUM)

**Locations:**

| File                                      | Line | Status                                                                                                                                                          |
| ----------------------------------------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `crates/cobre-sddp/src/lag_transition.rs` | 13   | Module-level blanket. Comment says "used starting in Epic 2" but exports are already wired into `setup.rs` lines 851, 856, 864. Masks genuinely dead functions. |
| `crates/cobre-sddp/src/noise.rs`          | 288  | `accumulate_and_shift_lag_state` has `#[allow(dead_code)]` but is called from `forward.rs:912` and `pipeline.rs:458`.                                           |
| `crates/cobre-sddp/src/noise.rs`          | 194  | `shift_lag_state_from_inflows` has `#[allow(dead_code)]` but is transitively called.                                                                            |
| `crates/cobre-sddp/src/noise.rs`          | 161  | `shift_lag_state` is genuinely dead production code (only used in tests), superseded by `accumulate_and_shift_lag_state`. Kept alive by suppression.            |
| `crates/cobre-io/src/output/schemas.rs`   | 4    | Module-level blanket covering 16 schema functions. At least one (`rank_timing_schema`) appears unused outside tests.                                            |

**Fix:** Remove all blanket `#![allow(dead_code)]`. Remove the stale per-function
annotations. For `shift_lag_state`, either delete (refactor tests to use
`accumulate_and_shift_lag_state` with monthly-identity transitions) or move to
`#[cfg(test)]`.

---

### A5. `noise_group_id_at` silent fallback on hot path (MEDIUM)

**Location:** `crates/cobre-sddp/src/context.rs:77-82`

**Problem:**

```rust
self.noise_group_ids.get(t).copied()
    .unwrap_or(u32::try_from(t).unwrap_or(u32::MAX))
```

Called once per stage per trajectory. The `get(t)` should never fail
(`noise_group_ids` is always stage-count sized), but the silent fallback masks
invariant violations and adds a branch the compiler may not elide.

**Fix:**

```rust
debug_assert!(t < self.noise_group_ids.len(), "stage index {t} out of bounds");
self.noise_group_ids[t]
```

---

### A6. `too_many_arguments` suppression count at ceiling (LOW)

The pre-commit check (`scripts/check_suppressions.py`) reports 10 suppressions
against a threshold of 10. The count has not increased since v0.4.2, but it is at
ceiling. Any new function requiring the suppression must first absorb parameters into
a context struct (per `.claude/architecture-rules.md`).

Current locations:

- `backward.rs:679`
- `estimation.rs:1264`
- `forward.rs:1038`
- `lower_bound.rs:116`
- `noise.rs:458`
- `setup.rs:509`
- `simulation/pipeline.rs:189, 866`
- `training.rs:357`
- `tree/generate.rs:104` (cobre-stochastic)

---

## 2. Performance Bottlenecks

### P1. Per-rayon-worker `TrialAccumulators` allocation in backward pass (CRITICAL)

**Location:** `crates/cobre-sddp/src/backward.rs:609-620`

**Problem:** Every rayon worker allocates fresh heap storage per stage per iteration:

```rust
let mut staged: Vec<StagedCut> = Vec::new();             // no capacity hint
let mut accum = TrialAccumulators {
    outcomes: (0..n_openings)
        .map(|_| BackwardOutcome {
            coefficients: vec![0.0; n_state],             // n_openings allocs
            ...
        })
        .collect(),
    slot_increments: vec![0u64; succ.successor_populated_count],
};
```

With 8 threads, 100 stages, 100 openings, n_state=200: ~80,000 `Vec<f64>`
allocations per iteration. This is the highest-impact allocation pattern in the
codebase.

**Fix:** Pre-allocate `TrialAccumulators` as a field in `SolverWorkspace`. Before
each rayon closure body, zero-fill:

```rust
accum.slot_increments.fill(0);
for o in &mut accum.outcomes { o.coefficients.fill(0.0); o.intercept = 0.0; }
```

Also pre-allocate `staged` with `Vec::with_capacity(local_work / n_workers + 1)`.

**Estimated impact:** Eliminates O(n_workers x n_stages x n_openings) heap
allocations per iteration.

---

### P2. Per-trial-point `binding_increments` Vec (CRITICAL)

**Location:** `crates/cobre-sddp/src/backward.rs:564-570`

**Problem:**

```rust
let binding_increments: Vec<(usize, u64)> = accum
    .slot_increments.iter().enumerate()
    .filter(|&(_, c)| *c > 0)
    .map(|(s, c)| (s, *c))
    .collect();   // new heap allocation every trial point
```

1,980 small allocations per iteration (20 passes x 99 stages), all inside the rayon
parallel region creating allocator pressure across threads.

**Fix options:**

1. Replace with `SmallVec<[(usize, u64); 8]>` -- zero heap cost for typical cases
2. Restructure the merge loop to iterate `slot_increments` directly from the
   accumulator, eliminating the intermediate collection entirely

---

### P3. `aggregate_solver_statistics` forces `.collect()` (HIGH)

**Locations:**

- `backward.rs:747, 873` -- 2 x (num_stages - 1) Vecs per iteration
- `training.rs:628-632, 652-656` -- 4 Vecs per iteration

**Problem:** The function signature takes `&[SolverStatistics]`, forcing every call
site to `.collect::<Vec<_>>()`. With 100 stages, this is ~200 small Vecs per
iteration just for diagnostics.

**Fix:** Change signature to `impl Iterator<Item = SolverStatistics>`. All four
call sites can pass the iterator directly, eliminating the intermediate allocation.

---

### P4. `deltas.clone().collect()` in training loop (HIGH)

**Location:** `crates/cobre-sddp/src/training.rs:732-736`

**Problem:**

```rust
let deltas: Vec<_> = backward_result.stage_stats.iter()
    .map(|(_, d)| d.clone())   // clone per stage
    .collect();                 // Vec allocation
let agg = SolverStatsDelta::aggregate(&deltas);
```

99 struct clones + 1 allocation per iteration to satisfy a `&[T]` API.

**Fix:** Change `SolverStatsDelta::aggregate` to accept
`impl Iterator<Item = &SolverStatsDelta>`.

---

### P5. Angular pruning O(n_cuts x n_state) allocations (HIGH)

**Location:** `crates/cobre-sddp/src/angular_pruning.rs:192-242`

**Problem:** `cluster_by_angular_similarity` allocates:

- `Vec<Option<Vec<f64>>>` of length n_eligible -- one `Vec<f64>` per cut for unit normals
- `Vec<Vec<usize>>` for cluster membership
- `Vec<Vec<f64>>` for cluster representatives, with `normal.clone()` per new cluster

At 1,000 cuts x 200 state dimensions = ~1.6 MB per pruning call per stage.

**Fix:** Flatten `unit_normals` into a contiguous `Vec<f64>` (sized
n_eligible x n_state) with `f64::NAN` sentinel rows for zero-norm. Add an
`AngularPruningWorkspace` struct to `SolverWorkspace` holding pre-allocated buffers.
Replace `cluster_reps.push(normal.clone())` with `copy_from_slice` into the flat
buffer.

---

### P6. O(n^2) setup-time loops in lag_transition.rs (MEDIUM)

**Locations:**

- `lag_transition.rs:256-260` -- `compute_monthly_transition` iterates full suffix
  of `all_stages` per stage for `is_last_in_period`
- `lag_transition.rs:464-470` -- `compute_downstream_transitions` has inner loop
  over `stages[stage_idx+1..transition_idx]` for `is_last_of_quarter`

**Impact:** Negligible for typical studies (128 stages), but O(n^2) for 50-year
studies (600+ stages).

**Fix:** Build `HashMap<(season_id, year), last_stage_index>` in a single O(n) pass.
Then `is_last_in_period` becomes an O(1) lookup.

---

## 3. Algorithmic Correctness

### Verdict: SOUND -- No Bugs Found

Exhaustive analysis confirmed:

| Component               | Verdict | Detail                                                                                                                                              |
| ----------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Angular pruning         | Correct | Cluster-wide max dominance is slightly more aggressive than pairwise but still preserves Assumption (H2) from Guigues 2017. Lower bound guaranteed. |
| Frozen-lag semantics    | Correct | accumulate/finalize/spillover in `noise.rs:289-422` produces bit-identical results to `shift_lag_state` for monthly identity case.                  |
| Observation aggregation | Correct | Duration-weighted averaging with leap-year handling via chrono.                                                                                     |
| PAR forward/inverse     | Correct | Equations are mathematically dual. sigma=0 edge case handled.                                                                                       |
| Cut management pipeline | Correct | All three stages only deactivate (never modify coefficients). Finite convergence maintained.                                                        |
| Scenario padding        | Correct | Modular wrap-around for non-uniform counts; clamp order is right (HistoricalResiduals then External, tighter wins via `.min()`).                    |
| Boundary cuts           | Correct | State dimension validated on load; warm-start counts per-stage.                                                                                     |
| Noise group sharing     | Correct | `copy_within` for shared groups; identity mapping for uniform monthly (no behavioral change).                                                       |

### Defensive assertions recommended (LOW)

These invariants are unreachable in practice but would catch future violations:

| Location                       | Missing guard                                                       |
| ------------------------------ | ------------------------------------------------------------------- |
| `noise.rs:350-351`             | `debug_assert!(n_completed <= par_order)` on downstream ring-buffer |
| `cut/pool.rs:245-250`          | Release-mode guard on double-insert (currently debug-only)          |
| `sampling/external.rs:334-336` | `debug_assert!(row.stage_id >= 0)` before `as usize` cast           |

---

## 4. Documentation Degradation

### D1. CHANGELOG duplicated 19 times (CRITICAL)

**Location:** `CHANGELOG.md`

The "Pattern C & D temporal resolution support" block (25 lines) appears identically
in **every version section** from `[Unreleased]` through `[0.1.5]` -- 19 copies.
This is a copy-paste accident that pollutes release history. Removing 18 spurious
copies would cut ~450 lines from the 1,529-line file.

**Fix:** Keep the block only under `[Unreleased]` (or whichever version actually
ships it). Delete the 18 other copies.

---

### D2. Design docs reference non-existent versions (HIGH)

**Locations and incorrect references:**

| File                                             | Claims | Actual     |
| ------------------------------------------------ | ------ | ---------- |
| `docs/design/per-stage-thermal-costs.md:5`       | v0.5.0 | v0.4.4     |
| `docs/design/cut-budget-per-stage.md:5`          | v0.5.0 | v0.4.4     |
| `docs/design/lp-scalability-cut-management.md:5` | v0.5.0 | v0.4.4     |
| `docs/design/per-stage-warm-start-cuts.md:5`     | v0.5.0 | v0.4.4     |
| `docs/design/temporal-resolution-debts.md:16-26` | v0.4.5 | Unreleased |

**Fix:** Update the four v0.5.0 references to v0.4.4. Update the v0.4.5 references
to "Unreleased (feat/temporal-resolution-cd)" or to the actual version once merged.

---

### D3. CLAUDE.md stale test count (HIGH)

**Location:** `CLAUDE.md:23`

States _"28 deterministic regression cases (D01-D16, D19-D28)"_ but D29 and D30 now
exist. The correct range is D01-D16, D19-D30 (27 directories; D12 absent).

**Fix:** Update the count and range.

---

### D4. Rustdoc broken links (MEDIUM)

`cargo doc --workspace --no-deps` produces 8 warnings in `cobre-sddp`:

1. `stage_lag_transitions` links to private `crate::lag_transition::precompute_stage_lag_transitions`
2. Unresolved link to `StudySetup::noise_group_ids`
3. `downstream_par_order` links to private `crate::noise::DownstreamAccumState`
4. `downstream_par_order` links to private `crate::setup::StudySetup::downstream_par_order`
   5-8. `WorkspaceSizing`/`new` link to private `ScratchBuffers` and `WorkspacePool::resize_scratch_bases`

Root cause: `lag_transition` is `pub(crate)` (lib.rs:56) so its public items are not
externally visible despite being referenced from public doc comments.

**Fix:** Either make the referenced items `pub` or rewrite doc comments to avoid
cross-referencing private items (use prose descriptions instead of intra-doc links).

---

### D5. Book lacks temporal resolution configuration guidance (MEDIUM)

The book's `guide/configuration.md` documents boundary cuts (lines 385-428) but
does not document Pattern C/D configuration aspects: noise group sharing, observation
aggregation, and lag transitions are implicit behaviors triggered by stage layout
(sub-monthly resolution), not explicit config fields. Users reading the config
reference will not discover these behaviors.

The `stochastic-modeling` page does cover runtime behavior (lines 258-313), so
partial coverage exists.

**Fix:** Add a subsection to `guide/configuration.md` explaining that sub-monthly
stage layouts automatically trigger noise sharing, observation aggregation, and lag
accumulation, with a cross-reference to the stochastic-modeling page.

---

## 5. Priority Matrix

### Tier 1: Quick wins with high impact (< 1 day combined)

| ID  | Category | Fix                                                            |
| --- | -------- | -------------------------------------------------------------- |
| P3  | Perf     | Change `aggregate_solver_statistics` to accept iterator        |
| P4  | Perf     | Change `SolverStatsDelta::aggregate` to accept iterator        |
| D1  | Docs     | Remove 18 duplicated CHANGELOG blocks                          |
| D2  | Docs     | Fix version references in 5 design docs                        |
| D3  | Docs     | Update CLAUDE.md test count                                    |
| A2  | Arch     | Remove "SDDP" from solver trait doc comment                    |
| A4  | Arch     | Remove stale `dead_code` suppressions                          |
| A5  | Arch     | Replace `noise_group_id_at` fallback with debug_assert + index |

### Tier 2: Targeted improvements (1-3 days combined)

| ID  | Category | Fix                                                   |
| --- | -------- | ----------------------------------------------------- |
| P1  | Perf     | Pre-allocate `TrialAccumulators` in `SolverWorkspace` |
| P2  | Perf     | SmallVec or direct iteration for `binding_increments` |
| P5  | Perf     | Flatten angular pruning into workspace buffers        |
| A3  | Arch     | Group `TrainingConfig` into sub-structs               |
| D4  | Docs     | Fix 8 rustdoc broken links                            |
| D5  | Docs     | Add temporal resolution section to config guide       |

### Tier 3: Strategic investment (1 week)

| ID  | Category | Fix                                                   |
| --- | -------- | ----------------------------------------------------- |
| A1  | Arch     | Extract `setup.rs` into focused modules               |
| P6  | Perf     | O(n) precomputed HashMap in lag_transition.rs         |
| A6  | Arch     | Work down `too_many_arguments` count from 10 toward 0 |

---

## Appendix: Files Analyzed

The following files were read in full by the specialist agents:

**Hot-path (cobre-sddp):**
`forward.rs`, `backward.rs`, `training.rs`, `simulation/pipeline.rs`,
`lp_builder/mod.rs`, `lp_builder/patch.rs`, `workspace.rs`

**New modules (cobre-sddp):**
`lag_transition.rs`, `angular_pruning.rs`, `basis_padding.rs`, `noise.rs`,
`cut/pool.rs`, `cut/fcf.rs`, `cut/wire.rs`, `cut_selection.rs`, `config.rs`,
`context.rs`, `setup.rs`, `policy_load.rs`

**Stochastic (cobre-stochastic):**
`tree/generate.rs`, `sampling/external.rs`, `sampling/historical.rs`,
`sampling/window.rs`, `sampling/class_sampler.rs`, `sampling/out_of_sample.rs`,
`sampling/mod.rs`, `par/aggregate.rs`, `par/evaluate.rs`, `par/fitting.rs`,
`par/precompute.rs`, `par/validation.rs`

**I/O and infra:**
`cobre-solver/src/trait_def.rs`, `cobre-io/src/validation/semantic.rs`,
`cobre-io/src/initial_conditions.rs`, `cobre-io/src/config.rs`,
`cobre-io/src/output/schemas.rs`

**Documentation:**
`CHANGELOG.md`, `CLAUDE.md`, `book/src/SUMMARY.md`, `book/src/guide/configuration.md`,
all files under `docs/design/`

**Tests:**
`tests/deterministic.rs`, `tests/integration.rs`, `tests/decomp_integration.rs`,
`tests/pattern_d_integration.rs`, `tests/boundary_cuts.rs`,
`tests/forward_sampler_integration.rs`
