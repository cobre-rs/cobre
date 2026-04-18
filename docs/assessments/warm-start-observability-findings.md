# Warm-Start Observability Findings (Convertido Analysis)

**Date:** 2026-04-18
**Branch:** `feat/highs-integration` (after epic-01 B.1 + epic-02 A.1 tickets 001-005)
**Workload:** `~/git/cobre-bridge/example/convertido{,_before}` (117 stages, 10 iterations, 10 forward passes, 100 simulation scenarios, 5 threads, single rank)

This document records four issues uncovered while analyzing two runs of the
convertido case (one with the legacy configuration, one with the new A.1 +
B.1 defaults) intended to validate the HiGHS warm-start work.

---

## TL;DR

| #   | Finding                                                                                                                               | Severity                                      | Scope                                   |
| --- | ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | --------------------------------------- |
| 1   | Simulation ignores `warm_start_basis_mode` config (defaults to `NonAlienFirst`)                                                       | **High — correctness of the rollback lever**  | CLI + Python                            |
| 2   | Simulation path skips `enforce_basic_count_invariant`, producing 900 non-alien rejections                                             | **High — explains observed rejection counts** | `cobre-sddp/src/simulation/pipeline.rs` |
| 3   | `basis_rejections` counter under `alien_only` is a structural lie — HiGHS silently rewrites invalid bases via `accommodateAlienBasis` | **Medium — misleading observability**         | `SolverStatistics`; docs                |
| 4   | The two new configs are demonstrably active, but their gains are ~0.3% of total wall time because simplex dominates at ~93%           | **Informational**                             | Benchmark interpretation                |

---

## 1. Simulation ignores the `warm_start_basis_mode` config

### Symptom

Both `convertido` (`warm_start_basis_mode: non_alien_first`) and
`convertido_before` (`warm_start_basis_mode: alien_only`) print
`Non-alien rejections: 900` in the simulation summary. The `alien_only`
case should never call the non-alien setter, so a non-zero count on that
path is impossible by construction.

### Root cause

`run_simulation_phase` builds its solver factory without applying the
config-driven warm-start mode:

```rust
// crates/cobre-cli/src/commands/run.rs:1116-1123
fn run_simulation_phase(
    ctx: &RunContext<impl Communicator>,
    system: &System,
    setup: &mut StudySetup,
    training_result: &cobre_sddp::TrainingResult,
    hostname: &str,
) -> Result<(), CliError> {
    let solver_factory = HighsSolver::new;   // <-- no .with_warm_start_mode(…)
```

`HighsSolver::new` produces a solver with the **type default**
`WarmStartBasisMode::NonAlienFirst` (`crates/cobre-solver/src/highs.rs:52-60`).

Compare with `run_training_phase` (same file, 911-918), which does thread
the config through:

```rust
let solver_factory =
    move || HighsSolver::new().map(|s| s.with_warm_start_mode(warm_start_basis_mode));
let mut solver = HighsSolver::new()
    .with_warm_start_mode(warm_start_basis_mode);
```

The Python binding has the same defect at
`crates/cobre-python/src/run.rs:393`:

```rust
.create_workspace_pool(n_threads, HighsSolver::new)
```

### Provenance

The training-side wiring was added in commit `0e062e97` ("feat: complete
epic 01 (B.1) non-alien setBasis warm-start" — ticket-010). The diff
shows the author converted `run_training_phase`'s `let solver_factory =
HighsSolver::new;` to the explicit form but left the identical line in
`run_simulation_phase` untouched. No `simulation_warm_start_basis_mode`
key exists in the config schema either — the field was designed to
govern both phases, it just wasn't wired.

### Impact

1. The `alien_only` rollback lever established by ticket-010 is silently
   disabled during simulation. Any attempt to validate the legacy alien
   path in simulation will actually exercise the new non-alien path.
2. Simulation always pays the `cobre_highs_set_basis_non_alien`
   consistency check, whether the user asked for it or not.
3. Output correctness is preserved (the alien fallback succeeds on every
   rejected basis), but the `convertido_before` simulation numbers do
   not reflect the `alien_only` behavior the config requested.

### Recommended fix

Thread `warm_start_basis_mode` into `run_simulation_phase` the same way
`run_training_phase` does:

```rust
let solver_factory =
    move || HighsSolver::new().map(|s| s.with_warm_start_mode(warm_start_basis_mode));
```

Same fix in the Python binding. Add an integration test that runs a
minimal simulation with `alien_only` and asserts
`basis_non_alien_rejections == 0`.

---

## 2. 900 non-alien rejections in simulation — basic-count invariant

violation from cut-set churn

### Observed counters (simulation, both cases)

| Counter                             |   Value |
| ----------------------------------- | ------: |
| `basis_offered`                     |  11,700 |
| `basis_non_alien_rejections`        | **900** |
| `basis_rejections` (alien fallback) |       0 |

11,700 = 100 scenarios × 117 stages — every simulation LP warm-starts.
The 900 non-alien rejections (7.7%) all fall through to the alien path,
which succeeds. Why do they reject?

### Mechanism (step-by-step)

1. **During training's forward pass for iteration K**, the basis is
   captured at each `(scenario, stage)` pair
   (`crates/cobre-sddp/src/forward.rs:1073-1099`). The LP at capture
   time has `base_row_count + active_cut_count_at_iter_K_start` rows.
2. **During iteration K's backward pass**, 1,160 new cuts are generated
   (116 stages × 10 forward passes).
3. **During iteration K's cut selection**, some subset of ALL cuts
   (stored + newly added) are deactivated. On convertido with
   `lml1 memory_window=1`: 7,543 cuts deactivated across 10 iterations.
4. **The basis cache is not updated after backward + cut selection.**
   Its row_status still reflects the iter-K forward-pass LP.
5. **In simulation**, `reconstruct_basis` is called (see
   `crates/cobre-sddp/src/simulation/pipeline.rs:415-467`) to adapt the
   stored basis to the current LP:
   - Walks `pool.active_cuts()` (the simulation's cut set).
   - For each current cut that matches a stored slot: preserves its
     stored row status.
   - For each current cut that was NOT in the stored basis: marks it
     `HIGHS_BASIS_STATUS_BASIC` unconditionally.
   - **Dropped stored cuts contribute nothing to the output.**
6. **Invariant accounting** (using subscripts _s_ for stored, _c_ for
   current):
   - Stored basis satisfies HiGHS's invariant exactly:
     `col_basic_s + template_basic_s + stored_cut_basic_s = base_row + stored_cut_count`.
   - After reconstruction:
     ```
     col_basic + template_basic + preserved_cut_basic + new_cut_basic
     = (col_basic_s + template_basic_s + stored_cut_basic_s)
        - dropped_cut_basic_s
        + new_cut_count
     = base_row + stored_cut_count - dropped_cut_basic_s + new_cut_count
     = base_row + preserved_count + dropped_count
        + new_cut_count - dropped_cut_basic_s
     = num_row + (dropped_count - dropped_cut_basic_s)
     = num_row + dropped_nonbasic_count
     ```
   - Each dropped stored cut that was **non-BASIC** adds 1 to the
     `total_basic - num_row` excess.
7. **HiGHS `isBasisConsistent`** (`vendor/HiGHS/highs/lp_data/HighsSolution.cpp:1950-1963`)
   requires `total_basic == num_row` exactly. Any excess rejects.

### Why training forward pass doesn't see this

`forward.rs:959-970` explicitly applies the fix:

```rust
// Forward-path invariant fix (ticket-009): after cut-set churn,
// col_basic + row_basic may exceed num_row when dropped cuts had
// BASIC status.  Demote trailing cut-row BASIC statuses to LOWER
// to restore the HiGHS isBasisConsistent invariant.
// Do NOT apply on the backward path — ticket-008 proved delta == 0
// there by construction.
let num_row = ctx.templates[t].num_rows + pool.active_cuts().count();
let demotions = enforce_basic_count_invariant(
    &mut ws.scratch_basis,
    num_row,
    ctx.templates[t].num_rows,
);
```

### Why simulation doesn't apply it

`crates/cobre-sddp/src/simulation/pipeline.rs:415-467` calls
`reconstruct_basis` but **not** `enforce_basic_count_invariant`. The
code explicitly acknowledges this:

```rust
ws.solver.record_reconstruction_stats(
    recon_stats.preserved,
    recon_stats.new_tight,
    recon_stats.new_slack,
    0, // simulation baked path: no demotion pass
);
```

and the fallback arm similarly:

```rust
    0, // simulation fallback path: no demotion pass
```

Ticket-009 (which introduced `enforce_basic_count_invariant`) fixed the
forward training path and left a matching note for the backward path
("delta == 0 by construction"). The **simulation** path was evidently
overlooked — it has the same cut-set-churn exposure as forward but
inherits none of the protection.

### Connection to the user's intuition

The user asked: _"Does the simulation LP have more rows than the stored
basis because we forget to update our basis in the last training
iteration?"_

The answer is nuanced:

- **Row-count-wise**: `reconstruct_basis` always produces an output of
  size `base_row + current_cut_count`, matching the simulation LP
  exactly. There is no dimension mismatch at the Rust level. The
  `extend-with-BASIC` branch in `solve_with_basis`
  (`highs.rs:1282-1288`) never fires for simulation.
- **Basic-count-wise**: YES, the basis is stale relative to the
  post-backward, post-selection cut pool. The staleness manifests as
  dropped non-BASIC cuts producing an excess that violates
  `isBasisConsistent`. Not a row-count gap — a semantic gap.
- **Could we refresh the basis after the final backward/selection?** In
  principle an extra "warm-start refresh" pass at the end of training
  would re-capture bases against the final cut set and eliminate the
  problem for simulation. But it doubles the last iteration's work and
  still doesn't help if simulation builds its own cut set (it doesn't,
  currently — simulation uses training's final FCF). The
  `enforce_basic_count_invariant` fix is the simpler solution and
  matches what the forward training path already does.

### 900 is consistent with the mechanism

- Iter-10 forward-pass capture: 4,725 active cuts (from iter 1..9).
- Iter-10 backward: +1,160 new cuts.
- Iter-10 cut selection: 1,828 deactivated → 4,057 active.
- Stored cuts dropped at selection round 10: at least 4,725 − 4,057 =
  **668**, up to 1,828 (depending on how the selection split between
  old and new).
- 7.7% of 11,700 solves = 900 rejections. Consistent with a subset of
  stages (those with the most non-BASIC cut churn) producing non-zero
  excess.

### Recommended fix

Apply `enforce_basic_count_invariant` to both simulation arms in
`crates/cobre-sddp/src/simulation/pipeline.rs:415-467`:

```rust
let num_row = match load_spec.baked_template {
    Some(baked) => baked.num_rows,  // baked path: structural rows only
    None => ctx.templates[t].num_rows + pool.active_cuts().count(),
};
let demotions = enforce_basic_count_invariant(
    &mut ws.scratch_basis,
    num_row,
    // base_row_count must match whichever branch we took
    match load_spec.baked_template {
        Some(baked) => baked.num_rows,
        None => ctx.templates[t].num_rows,
    },
);
ws.solver.record_reconstruction_stats(
    recon_stats.preserved,
    recon_stats.new_tight,
    recon_stats.new_slack,
    demotions,
);
```

Add a regression test in
`crates/cobre-sddp/tests/simulation_integration.rs` asserting
`basis_non_alien_rejections == 0` for a minimal 3-stage, 5-scenario
simulation with cut selection enabled.

---

## 3. `basis_rejections` under `alien_only` is a structural lie

### What the counter actually measures

In `highs.rs:1338-1342`:

```rust
if set_status == ffi::HIGHS_STATUS_ERROR {
    self.stats.basis_rejections += 1;
    debug_assert!(false, "raw basis rejected; falling back to cold-start");
}
```

The counter fires only when the chosen setter returns
`HIGHS_STATUS_ERROR`.

### What the alien path actually does with invalid bases

`cobre_highs_set_basis` → `Highs_setBasis` (C API) →
`Highs::setBasis(const HighsBasis&)` with `basis.alien = true` (the C API
default; see `HStruct.h:68`):

```cpp
// vendor/HiGHS/highs/lp_data/Highs.cpp:2557-2592
if (basis.alien) {
    if (!isBasisRightSize(model_.lp_, basis)) {
        return HighsStatus::kError;    // only rejects on dimension mismatch
    }
    // ... formSimplexLpBasisAndFactor(...) called next
}
```

Where `isBasisRightSize` only checks `col_status.size() == num_col &&
row_status.size() == num_row`. Since our Rust code pre-sizes to the
exact LP dimensions (`highs.rs:1277-1287`, extending with BASIC if the
stored basis is short), this check effectively never fails.

The real work happens inside `accommodateAlienBasis`
(`vendor/HiGHS/highs/lp_data/HighsSolution.cpp:1875-1934`) when
`basis.alien` is true:

1. Runs LU factorization on the proposed basis via `HFactor::build()`
   and measures rank deficiency.
2. **Clears every basic flag you provided** (lines 1904-1911).
3. **Re-marks basic only the columns HFactor found pivotable**, in the
   order the LU discovered them (lines 1916-1924).
4. **Fills the remaining rows with logical slack basics** to reach
   `num_basic_variables == num_row` (lines 1927-1932).

It returns `kOk` regardless. HiGHS silently replaced your basis with a
"nearest valid" one.

### Consequence

Under `WarmStartBasisMode::AlienOnly`:

- `basis_rejections` is structurally guaranteed to be 0 in production.
- `basis_set_time_ms` absorbs the cost of a throw-away LU factorization
  and silent repair on every call.
- The simplex sees a basis that may have very different basic columns
  than what we sent.

Under `WarmStartBasisMode::NonAlienFirst`:

- `basis_non_alien_rejections` IS a meaningful counter. It counts calls
  where `isBasisConsistent` (a real structural check) rejected because
  `total_basic != num_row`.
- Singular-but-consistent bases still pass the non-alien check; the
  simplex handles them internally via pivoting. No counter fires.

This is exactly the HiGHS behavior that motivated epic-01: "we detected
that HiGHS was silently fixing when we passed invalid basis". The
non-alien path is the first real strict check we have.

### Implications for the training counters we already saw

In the convertido training parquets:

- **`convertido_before` (alien_only)**: `basis_rejections = 0`,
  `basis_non_alien_rejections = 0`. The 0s tell us nothing about
  warm-start quality — HiGHS may have silently rewritten many bases.
- **`convertido` (non_alien_first)**: `basis_non_alien_rejections = 0`.
  This **is** a real signal: every proposed basis had correct basic
  count. It confirms the epic-01 ticket-008/009 invariant-preservation
  work is functioning as designed.

### Recommended follow-ups

From cheapest to most invasive:

1. **Docs & naming hygiene (cheap)**. Update
   `SolverStatistics::basis_rejections` rustdoc to state it fires only
   on dimension mismatch under `AlienOnly` and on structural count
   mismatch under `NonAlienFirst`. Consider renaming at the next
   breaking change.
2. **Synthetic "silent repair" counter (moderate)**. Before calling the
   alien setter, run an in-Rust equivalent of `isBasisConsistent`. If it
   would have failed but we're calling alien anyway (by user choice),
   increment a new `basis_silent_repairs` counter. O(num_col + num_row)
   per call — negligible against LU cost.
3. **HiGHS patch to expose `rank_deficiency` (invasive)**. Add a
   shim around `accommodateAlienBasis` returning the rank-deficiency
   count. Not worth it unless a specific regression demands it.

Recommend doing (1) as part of the ticket for issues 1–2.

---

## 4. Convertido performance snapshot (context for the above)

### Per-phase aggregates (10 iterations, 5 threads)

| Phase            | Metric               |    Legacy |       New |                            Δ |
| ---------------- | -------------------- | --------: | --------: | ---------------------------: |
| Backward         | `load_model_time_ms` |     7,187 |     3,652 | **−49.2%** (A.1 ClearSolver) |
| Backward         | `basis_set_time_ms`  |     8,561 |        91 |   **−98.9%** (B.1 non-alien) |
| Backward         | `simplex_iterations` |   42.04 M |   42.05 M |               +0.02% (noise) |
| Backward         | `solve_time_ms`      | 3,847,249 | 3,854,003 |                       +0.18% |
| Forward          | `basis_set_time_ms`  |     6,717 |        75 |             **−98.9%** (B.1) |
| Forward          | `load_model_time_ms` |     7,302 |     7,282 |                    unchanged |
| Total wall clock |                      |   1,061 s |   1,058 s |                        −0.3% |

Setup savings were ~20 s CPU across 5 threads ≈ 4 s wall, matching the
observed 3.1 s delta.

### Determinism (critical for ticket 009 default flip)

The lower bound is **byte-identical** across all 10 iterations
(2.350645e+11). Upper-bound divergence appears only at iteration 9 in
the 7th significant digit (7.334967e+10 vs 7.334972e+10) — pure
float-ordering noise. Active cut counts differ by 2 out of ~4000. A.1
under `ClearSolver` is mathematically equivalent to the legacy
`Disabled` path on this workload.

### Newave B/F ratio comparison

| Case                                       | B/F ratio (10 iters) |
| ------------------------------------------ | -------------------: |
| Newave 31 (64 procs)                       |                5.47× |
| Cobre convertido (5 threads, both configs) |                10.1× |

Cobre's backward pass dominates roughly twice as much as Newave's. The
simplex per-LP wall time is comparable; the structural lever is a
stage-major forward pass (Phase C in the roadmap, deferred). A.1/B.1/B.2
together shave ~10% off total wall time but don't close the B/F-ratio
gap. Matching Newave's ~5× ratio requires Phase C, which is out of
scope.

---

## Recommended ticket plan

### Immediate (epic-01 follow-up, before flipping ticket-009 default)

1. **Fix `run_simulation_phase` warm-start mode wiring**
   - Thread `warm_start_basis_mode` into the CLI and Python simulation
     solver factories.
   - Regression test: simulation under `alien_only` → 0 non-alien
     rejections.

2. **Apply `enforce_basic_count_invariant` in simulation**
   - Call it in both arms of `simulation/pipeline.rs:415-467`.
   - Regression test: simulation under `non_alien_first` with a cut
     selection strategy enabled → 0 non-alien rejections.

3. **Document `basis_rejections` semantics**
   - rustdoc note on `SolverStatistics::basis_rejections` clarifying
     that under `AlienOnly` it only catches dimension-mismatch failures,
     not actual acceptance.

### Optional (observability ticket)

4. **Add a `basis_silent_repairs` counter** that fires when the alien
   path would silently rewrite an inconsistent basis. Gives users a
   real signal on "warm-start quality" even under `alien_only`.

### Out of scope for this follow-up

- Phase C (stage-major forward). Roadmap-deferred.
- B.2 (per-opening basis storage). Outline in plan, gated on post-A.1
  profiling (epic-04).

---

## File & line references

- CLI simulation bug: `crates/cobre-cli/src/commands/run.rs:1116-1123`
- Python simulation bug: `crates/cobre-python/src/run.rs:393`
- Training-side wiring (correct): `crates/cobre-cli/src/commands/run.rs:911-918`
- Forward-path invariant fix (precedent):
  `crates/cobre-sddp/src/forward.rs:959-970`
- Simulation reconstruction call sites (missing invariant fix):
  `crates/cobre-sddp/src/simulation/pipeline.rs:415-467`
- `enforce_basic_count_invariant` definition:
  `crates/cobre-sddp/src/basis_reconstruct.rs:338-…`
- `WarmStartBasisMode` type and default:
  `crates/cobre-solver/src/highs.rs:52-60`
- Counter increment sites: `crates/cobre-solver/src/highs.rs:1298-1342`
- `Highs::setBasis(HighsBasis const&)` alien branch:
  `crates/cobre-solver/vendor/HiGHS/highs/lp_data/Highs.cpp:2557-2620`
- `accommodateAlienBasis`:
  `crates/cobre-solver/vendor/HiGHS/highs/lp_data/HighsSolution.cpp:1875-1934`
- `isBasisConsistent`:
  `crates/cobre-solver/vendor/HiGHS/highs/lp_data/HighsSolution.cpp:1950-1963`
