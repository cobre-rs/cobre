# Epic 06 — classifier refinement gaps (follow-up review)

**Status:** Design-review artifact — input for additional Epic 06 tickets
**Authored:** 2026-04-21, during walkthrough of epic-06 scope before T3 dispatch
**Scope:** Critique of the activity-driven classifier (AD-2) and Scheme 1 demotion (AD-3) as specified in `docs/design/epic-06-cut-basis-reconstruction.md` and implemented in T1 (shipped, commit `1546b95`) and T2 (in-progress, uncommitted `basis_reconstruct.rs`)
**Relation to plan:** does NOT replace Epic 06 overview; supplements it with findings that emerged during close reading of the spec + code. Each "Gap" below maps to a candidate ticket extension.
**Blocks / gates:** T3 (route `active_cuts()` through `stage_solve.rs`) activates these mechanisms in production. The findings below are legitimate candidates for land-before-T3, because merging Parts A + B into production before addressing them risks A/B #7 producing a null or negative result for reasons that are fixable without changing architecture.

---

## 1. Executive summary

Four independent design gaps were identified in the activity-driven classifier + Scheme 1 symmetric demotion as currently specified. Each is small (≤ 5 lines of source change for G1-G3; G4 is a wider plumbing change) and can be tested independently. Together they appear to reverse the classifier's effectiveness for a substantial fraction of the cut population it is meant to help.

| #   | Gap                                                                                                                                                                                                                                                                     | Affects                                                                                                           | Candidate fix                                                                                                                                                                                                                                              |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| G1  | Freshly-generated cuts enter their first LP with `active_window = 0` → classifier falls back to BASIC-default despite strongest domain signal that they are likely tight                                                                                                | ~192 × num_stages cuts per iteration (~22,500 on convertido)                                                      | Seed `active_window: 1` in `CutPool::add_cut`                                                                                                                                                                                                              |
| G2  | Demotion sort uses `active_window.count_ones()` over the full 32-bit window, while the classifier uses only the low 5 bits as informative — these two sites disagree on what counts as "recent"                                                                         | Every demotion decision where the candidate pool mixes fresh-and-recent cuts with old-and-stale cuts              | Either mask the popcount by `RECENT_WINDOW_BITS`, or replace the key with a recency-weighted scalar                                                                                                                                                        |
| G3  | popcount is recency-blind: bit 0 (current iter) and bit 31 (32 iterations ago) contribute equally to the sort key                                                                                                                                                       | Same population as G2, pathological whenever pool contains cuts with "old loud" activity and "new quiet" activity | Use `last_active_iter` directly, or `active_window.leading_zeros()`, or a lexicographic `(popcount_in_window, last_active_iter)`                                                                                                                           |
| G4  | `RECENT_WINDOW_K = 5` is a hardcoded `pub const` in `basis_reconstruct.rs`. 5 was inherited from CEPEL's calibration and never validated against a cobre-scale study; different study shapes (horizon length, convergence rate, cut pool size) may prefer different `k` | Every study run — a compile-time constant dictates a user-policy decision                                         | Promote to a configurable parameter on `CutSelectionConfig` (TOML) / `CutManagementConfig` (runtime), plumbed through `StudyParams` → `ConstructionConfig` → `StudySetup` → `StageInputs` → `ReconstructionSource`, same shape as `cut_activity_tolerance` |

Gaps G2 and G3 are closely related — both are symptoms of discarding the recency dimension that the classifier itself treats as primary. They may be solvable by a single change to the demotion key. G1 is independent. **G4 composes with all of them**: once the window is a runtime value, G2's fix automatically derives its mask from the same `k`, and A/B #7 can sub-test `k ∈ {3, 5, 10}` without recompilation (design-doc AD-7 originally reserved this as an optional sub-test, but only if the constant were configurable).

None of these require a `CapturedBasis` wire format change (Epic 06 AD-6 preserved) or a runtime on/off flag (AD-5 preserved — `k` is a tunable magnitude, not an enable switch).

---

## 2. Background — mechanism recap

Epic 06 adds a `u32` sliding-window bitmap to `CutMetadata` at `crates/cobre-sddp/src/cut_selection.rs:86-97`:

```rust
pub active_window: u32,
```

- Bit i = 1 iff the cut was binding (dual > `cut_activity_tolerance = 1e-6`) at iteration `current_iter - i`, OR-reduced across MPI ranks and openings.
- Set to bit 0 at the per-trial-point binding site (`backward.rs:741`: `metadata_sync_window_contribution[slot] |= 1u32`).
- MPI unified via `allreduce(BitwiseOr)` on a `u32` buffer (`backward.rs:1118-1154`).
- Shifted left by 1 at end-of-iteration in `training.rs:1062-1066` — bit 0 becomes bit 1, creating a fresh bit 0 for the next iteration's recording.

Two consumers:

**Classifier** (new cut → LOWER vs BASIC) at `basis_reconstruct.rs:362-376`:

```rust
if active_window & RECENT_WINDOW_BITS != 0 {
    stats.new_tight += 1;
    HIGHS_BASIS_STATUS_LOWER
} else {
    stats.new_slack += 1;
    HIGHS_BASIS_STATUS_BASIC
}
```

where `RECENT_WINDOW_BITS = 0b11111` (low 5 bits, `RECENT_WINDOW_K = 5`).

**Demotion sort** (which preserved-BASIC to flip to LOWER) at `basis_reconstruct.rs:355-358, 389-407`:

```rust
let popcount = cut_metadata.get(target_slot)
    .map_or(0, |m| m.active_window.count_ones());
demotion_candidates.push((out_row_index, popcount));
// ... later ...
demotion_candidates.sort_by_key(|&(_, popcount)| popcount);  // ascending
// .take(scheme1_count) flips the lowest-popcount candidates to LOWER
```

The key observation: the two consumers do not agree on which bits of `active_window` matter.

---

## 3. Gap G1 — the 192-cut-per-stage invisible cohort

### 3.1 The cut lifecycle during one backward pass

The backward pass at iteration `k` iterates stages in reverse at `backward.rs:948`:

```
for t in (0..num_stages-1).rev():
    successor = t + 1
    build delta batch for pool[successor] from this iter's earlier backward steps
    load baked template for stage `successor`
    append delta rows
    solve LP(successor) at 192 trial points × num_openings
    record binding events into slot_increments for pool[successor]
    per-stage MPI allreduce(BOR) → update pool[successor].metadata.active_window
    aggregate 192 new cuts
    fcf.add_cut(t, ...) → adds the 192 cuts to pool[t]  # NOT pool[successor]
```

So the 192 cuts added at step `t` go into `pool[t]`. Those cuts become visible to the solver for the first time at **step `t-1` of the same iteration**, where `successor = t` and the delta batch for `pool[t]` now includes them.

### 3.2 What the classifier sees at that first encounter

At step `t-1` of iteration `k`, the warm-start basis read for the (m, s=t) solve was captured at the same (m, s=t) solve during iteration `k-1`. That stored basis's `cut_row_slots` reflects `pool[t]` at the end of iteration `k-1` — **before the 192 fresh iter-k cuts existed**.

So every one of those 192 slots misses the `slot_lookup` lookup → they all hit the "new cut" branch of the classifier:

```rust
let active_window = cut_metadata.get(target_slot).map_or(0, |m| m.active_window);
if active_window & RECENT_WINDOW_BITS != 0 { LOWER } else { BASIC }
```

Because `CutPool::add_cut` at `cut/pool.rs:256` initialises `active_window: 0`, and no shift or bind event has fired on these slots yet in iteration `k` at this point in the backward loop, every fresh cut satisfies `active_window & RECENT_WINDOW_BITS == 0` → **BASIC**.

### 3.3 The domain signal this discards

A cut produced by the backward aggregation at step `t` is built from 192 duals on the LP at stage `t+1`, solved at trial points `x_hat_t` from iteration k's forward pass. By construction, **each generating cut is active (tight) at its generating trial point**. At the immediately-preceding stage (step `t-1`), the trial points `x_hat_{t-1}` feed into LP(t) and the new cuts are evaluated in that LP.

- Early in training (before convergence): `x_hat_{t-1}` and `x_hat_t` explore different state-space regions; correlation is imperfect but nonzero.
- Near convergence: forward-pass states vary smoothly across stages; `x_hat_{t-1}` and `x_hat_t` are in nearby regions → a cut tight at one is usually tight at the other.

In both regimes, "likely tight" is a stronger prior for the fresh cohort than for preserved-but-unobserved cuts. The current classifier explicitly throws this prior away.

### 3.4 Scale of the miss on convertido

- 50 forward scenarios → 50 new cuts per stage per iteration (the convertido case in Epic 03's A/B #3 baseline).
- 117 stages → ~5,850 fresh cuts per iteration.
- 5 iterations → ~29,250 fresh cuts over a run.
- The user's question hypothesised 192 forward scenarios — that would yield ~22,500 per iteration, ~112,500 over a 5-iter run.

Every one of these cuts takes the BASIC-default path on its first LP encounter. Epic 06's classifier provides zero benefit for this cohort today.

### 3.5 The current spec's only acknowledgement

Design doc §11 R4:

> **R4 — Activity-window at iter=1 is empty**, making the classifier no better than the default on the first iteration. Acceptable — iter-1 pivot count is small compared to steady-state iterations; the window fills quickly (within 5 iterations for k=5).

R4 frames the problem as a one-time iter-1 warm-up issue. It is not. The 192-cohort miss recurs every iteration for the newly-generated slots, regardless of how mature the rest of the pool is. The rationalisation does not cover the steady-state case.

### 3.6 Proposed fix G1

Change `CutPool::add_cut` at `cut/pool.rs:251-257` to seed `active_window: 1`:

```rust
self.metadata[slot] = CutMetadata {
    iteration_generated: iteration,
    forward_pass_index,
    active_count: 0,
    last_active_iter: iteration,
    active_window: 1,   // Epic 06 G1: cut is generated tight at x_hat; seed bit 0
};
```

Semantics change: bit 0 now means "bound at, or generated in, the current iteration." This is a minor overload of the bitmap — the generating event is treated as a bind-signal. CEPEL's SC adoption paper (`cepel-sc-adoption.md`, referenced by Epic 06 AD-7) uses the same intuition.

Additional update at `cut_selection.rs` to reflect the new semantics in the rustdoc.

**Interaction with end-of-iteration shift:** immediately after `add_cut`, the shift at `training.rs:1064` runs (`active_window <<= 1`), so `0b1` → `0b10`. That's actually correct — a cut generated at iter `k` sees its "generating" bit in position 1 at iter `k+1`, which still satisfies `active_window & 0b11111 != 0` → LOWER at every LP appearance during the next 5 iterations if no further bind events accumulate. Good default.

**But** — the first LP appearance is in the same iteration `k`, at backward step `t-1`, which happens BEFORE the end-of-iteration shift. At that moment the seeded bit 0 is still set → classifier fires LOWER → fresh cut gets the tight guess. Correct.

**Risk:** over-classification of LOWER on fresh cuts that happen to be slack at the adjacent stage's trial point. Scheme 2 fallback covers the invariant-preservation case when this happens en masse. But from the simplex perspective, a wrongly-guessed LOWER costs roughly the same as a wrongly-guessed BASIC — the pivot budget is symmetric.

### 3.7 Alternative considered — seed by generating iteration only

Another option: keep `active_window: 0` at creation, but change the classifier rule to treat `iteration_generated == current_iteration - 1` (just-created cuts) as a separate "fresh" bucket that defaults to LOWER. This requires the classifier to consume two metadata fields instead of one, increasing coupling. Rejected in favour of the single-field seed because the semantic meaning of bit 0 naturally covers "was binding or generated."

---

## 4. Gap G2 — window asymmetry between classifier and demotion sort

### 4.1 The asymmetry

| Consumer      | Bit range consulted          | Constant                    |
| ------------- | ---------------------------- | --------------------------- |
| Classifier    | Low 5 bits (`0b11111`)       | `RECENT_WINDOW_BITS` (AD-7) |
| Demotion sort | All 32 bits (`count_ones()`) | none — implicit default     |

The classifier has already made a conscious decision that **activity older than 5 iterations is not informative enough to justify a tight guess**. Yet the demotion sort — which ranks preserved-BASIC candidates on the same underlying data — weighs every bit equally. A cut that bound heavily 20 iterations ago but has been quiet since has its ancient activity counted toward its demotion priority.

### 4.2 Why this is a design inconsistency, not a performance shortcut

One might argue "the sort uses the wider window because it wants more signal." But:

- The classifier's window was calibrated to what's informative (AD-7 cites CEPEL's calibration at `k = 5`).
- Extending that window by 27 bits for a different site on the same data without justifying the widening introduces internal inconsistency.
- The sort does not gain accuracy from the wider window — it gains noise from activity that the classifier has already decided is stale.

### 4.3 Proposed fix G2

Option A (minimal): mask the popcount by `RECENT_WINDOW_BITS` at the popcount site (`basis_reconstruct.rs:355-358`):

```rust
let popcount = cut_metadata.get(target_slot)
    .map_or(0, |m| (m.active_window & RECENT_WINDOW_BITS).count_ones());
```

Range collapses from 0..=32 to 0..=5, matching the classifier's effective window. The stable-sort tie-break (slot order) still preserves full determinism across the five popcount bins.

Option B (bolder): drop popcount entirely in favour of a recency-weighted key. See Gap G3.

---

## 5. Gap G3 — popcount is recency-blind

### 5.1 The worked example

Two preserved BASIC cuts that need to be ranked for demotion:

| Cut   | `active_window` (bit 31 ← → bit 0)        | `popcount` (all 32 bits) | `popcount` (low 5 bits only) | `last_active_iter` | Interpretation                          |
| ----- | ----------------------------------------- | ------------------------ | ---------------------------- | ------------------ | --------------------------------------- |
| **A** | `1111_1111_1111_0000_0000_0000_0000_0000` | 12                       | 0                            | `k - 20`           | 12 binds all ≥ 20 iters ago (old/stale) |
| **B** | `0000_0000_0000_0000_0000_0000_0000_0010` | 1                        | 1                            | `k - 1`            | 1 bind, 1 iteration ago (fresh)         |

**Under the current design** (`count_ones()` over all 32 bits, ascending):

- Sorted order: B (popcount 1), A (popcount 12).
- `lower_deficit = 1` → B is demoted. A stays BASIC.

**Under domain reality:**

- A hasn't bound for 20 iterations. Its stored BASIC status is almost certainly current-correct — the cut is dead weight that cut selection will eventually deactivate.
- B bound last iteration. Its stored BASIC is a capture-time snapshot; recent evidence says the cut is usually tight.
- The correct demotion flips A, keeps B.

The current design picks exactly the opposite cut.

### 5.2 Why this isn't caught by tests

The T2 unit tests in `basis_reconstruct.rs` (spec: `scheme_1_symmetric_demotion_preserves_total_basic`) exercise **popcount 1, 3, 5** — all in the same bit positions. They verify the sort correctly picks the lowest popcount. They do not test the mixed case where `popcount_full > popcount_recent`.

A regression test that would catch this bug:

```rust
#[test]
fn scheme_1_recency_aware_sort_preserves_fresh_cuts() {
    // Cut A: stale but heavy — bound 12 iterations in a row, then silent for 20.
    // Cut B: fresh but light — bound once last iteration.
    // Expectation: A demoted, B preserved.
    // Current design: B demoted, A preserved (THIS FAILS, confirming the bug).
    ...
}
```

This test is a direct translation of the worked example. It should be added to whichever ticket adopts the fix.

### 5.3 Proposed fix G3

**Option 1: `last_active_iter` as the sort key.** Already a `u64` on `CutMetadata`, MPI-synced via the existing `allreduce(Max)` or similar. Sort ascending → oldest last-binding time demoted first.

```rust
let last_iter = cut_metadata.get(target_slot)
    .map_or(0, |m| m.last_active_iter);
demotion_candidates.push((out_row_index, last_iter));
// ...
demotion_candidates.sort_by_key(|&(_, last_iter)| last_iter);
```

Pros: simplest, uses existing field. Captures exactly "when did this cut last matter?"
Cons: coarse-grained (single data point). Two cuts that last bound at the same iteration sort by insertion order (slot). If this is the only signal, it may be too flat for fine discrimination.

**Option 2: `active_window.leading_zeros()` as the key.**

```rust
// leading_zeros(0) = 32 (never bound); leading_zeros(0b1) = 31; leading_zeros(0b1<<5) = 26; etc.
let staleness = cut_metadata.get(target_slot)
    .map_or(32, |m| m.active_window.leading_zeros());
demotion_candidates.push((out_row_index, staleness));
// ...
demotion_candidates.sort_by_key(|&(_, staleness)| std::cmp::Reverse(staleness));
```

Pros: captures "how long since the most recent bind" on a per-iteration resolution, using the bitmap we already have. Highest leading zeros → most stale → demote first.
Cons: maximum resolution is 32 iterations; after that, all cuts look equally stale. For an `N ≤ 5` iteration study (convertido), the low 5 bits already saturate useful information anyway.

**Option 3: lexicographic `(popcount_in_window, last_active_iter)`.**

```rust
let metadata = cut_metadata.get(target_slot);
let pop_in_window = metadata.map_or(0, |m| (m.active_window & RECENT_WINDOW_BITS).count_ones());
let last_iter = metadata.map_or(0, |m| m.last_active_iter);
demotion_candidates.push((out_row_index, pop_in_window, last_iter));
demotion_candidates.sort_by_key(|&(_, p, i)| (p, i));
```

Pros: best-of-both. Primary = activity density in the informative window (matches classifier). Secondary = recency (discriminates within a bin). Strict total order on `(popcount_in_window, last_active_iter, slot-via-stable-sort)` — determinism is guaranteed.
Cons: largest code change; needs two metadata lookups per candidate.

### 5.4 Revisiting the worked example under each option

| Cut | `popcount_full` | `popcount_low5` | `leading_zeros` | `last_active_iter` |
| --- | --------------- | --------------- | --------------- | ------------------ |
| A   | 12              | 0               | 0               | `k - 20`           |
| B   | 1               | 1               | 30              | `k - 1`            |

| Rule                                      | Sort 1st (demoted) | Sort 2nd (kept) | Correct? |
| ----------------------------------------- | ------------------ | --------------- | -------- |
| Current: `popcount_full` ascending        | **B**              | A               | no       |
| G2 alone: `popcount_low5` ascending       | **A (0)**          | B (1)           | yes      |
| G3 Option 1: `last_active_iter` ascending | **A** (k-20)       | B (k-1)         | yes      |
| G3 Option 2: `leading_zeros` descending   | **A** (lz=0)       | B (lz=30)       | yes      |
| G3 Option 3: lex `(pop_low5, last_iter)`  | **A** (0, k-20)    | B (1, k-1)      | yes      |

G2 alone (masking the popcount) fixes this specific example. The other options offer more discriminating power in cases where multiple cuts have the same masked-popcount bin — which happens often at steady state when many cuts have identical low-5-bits activity.

---

## 6. Gap G4 — `RECENT_WINDOW_K` is a compile-time constant

### 6.1 Current state

`RECENT_WINDOW_K = 5` is declared at `basis_reconstruct.rs:83-95` as a `pub const`:

```rust
pub const RECENT_WINDOW_K: u32 = 5;
pub const RECENT_WINDOW_BITS: u32 = (1u32 << RECENT_WINDOW_K) - 1;
```

Every consumer — the classifier at `basis_reconstruct.rs:365` and (after T7 lands) the demotion mask at G2 — uses `RECENT_WINDOW_BITS` directly. The value `5` was inherited from CEPEL's NEWAVE calibration, cited by the Epic 06 overview (AD-7) and the retired `cepel-sc-adoption.md` design doc. It has never been tuned on a cobre-scale study.

### 6.2 Why a fixed default is not defensible long-term

Different studies should probably prefer different `k`:

- **Short-horizon / fast-converging studies** (e.g. 60-stage problems that converge in 3-5 iterations): the bitmap barely fills. `k = 5` throws away bits 0..2 that DO carry signal early. `k = 3` or even `k = 2` would make the classifier less noisy while the policy is still stabilising.
- **Long-horizon / slow-converging studies** (convertido-scale, 117 stages, 20+ iterations to convergence): a cut that bound heavily in iters 1-10 and has been quiet since is genuinely stale — `k = 5` correctly ignores it. But a cut that only binds every 7-8 iterations (a "periodic" cut tied to a seasonal pattern in inflow/demand) is invisible to `k = 5`. `k = 10` would catch it.
- **Cut-selection-heavy studies** (aggressive `Level1` or `Lml1`): cut churn is high, the pool turns over, and a narrower window matches the effective age of the live pool. `k = 3` might be more honest than `k = 5`.
- **Cut-selection-off studies** (no deactivation, uncapped budget): cut lifetimes are unbounded, the pool carries history from every iteration. A wider `k` is justified because stale cuts stay in the pool.

None of these interact in a way that "5 is always right." AD-7 in the Epic 06 overview explicitly leaves `k ∈ {3, 5, 10}` as an A/B sub-test; A/B #7 cannot perform that sub-test today without re-editing the constant and rebuilding, because the value is compile-time. Making it configurable unblocks that sub-test, and allows end users to tune per study without patching the source.

### 6.3 Closest analogs in the codebase — which template to follow

The user's question explicitly names the **active cut budget** (`max_active_per_stage` / `budget`) as a candidate template. It's worth comparing this side-by-side with `cut_activity_tolerance`, the other strong candidate:

| Aspect                          | `budget` (`max_active_per_stage`)                  | `cut_activity_tolerance`           | `basis_activity_window` (proposed)                                                  |
| ------------------------------- | -------------------------------------------------- | ---------------------------------- | ----------------------------------------------------------------------------------- |
| TOML-facing type (`cobre-io`)   | `Option<u32>`                                      | `Option<f64>`                      | `Option<u32>`                                                                       |
| Runtime type (`cobre-sddp`)     | `Option<u32>` (None = uncapped)                    | `f64` (default applied)            | `u32` (default applied — see §6.5)                                                  |
| Semantics of `None` in TOML     | "disabled"                                         | "use default" (`1e-6`)             | "use default" (5)                                                                   |
| Validation                      | None beyond `> 0`                                  | `> 0.0` implied                    | `1..=32` (bitmap width)                                                             |
| Default when unset              | No cap                                             | `1e-6`                             | `5`                                                                                 |
| Flows via `StudyParams`?        | Yes                                                | Yes                                | Yes                                                                                 |
| Flows via `ConstructionConfig`? | Yes                                                | Yes                                | Yes                                                                                 |
| Held on `StudySetup`?           | Yes                                                | Yes                                | Yes                                                                                 |
| Consumer phases                 | Training orchestration (post-backward)             | Backward pass (bind detection)     | **Forward + backward + simulation** (all three call `stage_solve::run_stage_solve`) |
| Consumer file                   | `training.rs` (cut_selection loop)                 | `backward.rs:673` (bind threshold) | `basis_reconstruct.rs:365` (classifier) + (post-T7) the demotion mask               |
| Needs MPI broadcast             | Yes (rebuilt on non-root via `ConstructionConfig`) | Yes (same path)                    | Yes (same path)                                                                     |

**Verdict: `cut_activity_tolerance` is the closer template in every dimension except one (consumer phase count).**

- The SHAPE (scalar, Option-wrapped TOML with runtime default, broadcast via rebuild) is identical.
- The VALIDATION profile (a scalar with a sensible default, always applied) matches.
- The INFORMATIONAL semantic (tunable magnitude, not an on/off toggle) matches.

`budget` differs in that `None` is load-bearing — it means "disabled." For `basis_activity_window`, `None` only ever means "no user override, use the default"; there is no "disable the classifier" semantic associated with any `k`. So copying `budget`'s `Option<u32>` runtime representation would create a spurious "disabled" interpretation on the runtime side.

The one dimension where the new parameter differs from `cut_activity_tolerance` is consumer phase count: `cut_activity_tolerance` is a backward-only input (packed into `BackwardPassSpec` and read at `backward.rs:673`), whereas `basis_activity_window` is consumed by `reconstruct_basis` which is called from forward, backward, AND simulation paths via `stage_solve::run_stage_solve`. This means the parameter must be reachable from `StageInputs` (the struct passed to `run_stage_solve`), not just the backward spec. That's a one-field addition on `StageInputs` and one field addition on `ReconstructionSource`, plus one scalar carried on `StudySetup`.

### 6.4 Proposed plumbing — full flow diagram

Mirror `cut_activity_tolerance` with the small extension noted above. Full path:

```
        1. TOML / JSON entry point
           crates/cobre-io/src/config.rs :: CutSelectionConfig
              pub basis_activity_window: Option<u32>   #[serde(default)]
                      │
                      ▼
        2. Rank-0 parse + default application
           crates/cobre-sddp/src/setup/params.rs :: StudyParams::try_from
              let basis_activity_window = config
                  .training.cut_selection.basis_activity_window
                  .unwrap_or(DEFAULT_BASIS_ACTIVITY_WINDOW);   // 5
              validate: (1..=32).contains(&basis_activity_window)
                      │
                      ▼
        3. MPI broadcast payload
           crates/cobre-sddp/src/setup/params.rs :: ConstructionConfig
              pub basis_activity_window: u32
              (rebuilt on non-root ranks by the CLI / Python runner with
               the same field value — existing config-broadcast pattern)
                      │
                      ▼
        4. Study-wide holding pen
           crates/cobre-sddp/src/setup/mod.rs :: StudySetup
              pub(crate) basis_activity_window: u32
           crates/cobre-sddp/src/setup/accessors.rs
              pub(crate) fn basis_activity_window(&self) -> u32
                      │
                      ▼
        5. Orchestration hand-off
           crates/cobre-sddp/src/setup/orchestration.rs
              training_inputs.basis_activity_window = setup.basis_activity_window();
              (same line as `cut_activity_tolerance` at :51)
                      │
                      ▼
        6. Runtime config held on training/simulation structs
           crates/cobre-sddp/src/training.rs (+ simulation/pipeline.rs)
              read from config once, store on TrainingState / SimulationPipeline
                      │
                      ▼
        7. Stage-solve call boundary
           crates/cobre-sddp/src/stage_solve.rs :: StageInputs
              pub basis_activity_window: u32
                      │
                      ▼
        8. Consumer boundary
           crates/cobre-sddp/src/basis_reconstruct.rs :: ReconstructionSource
              pub basis_activity_window: u32
           derive mask at function entry:
              let recent_window_bits = (1u32 << source.basis_activity_window) - 1;
           replace every use of `RECENT_WINDOW_BITS` with `recent_window_bits`
                      │
                      ▼
        9. Classifier site (basis_reconstruct.rs:365)
           active_window & recent_window_bits != 0
        9. Demotion mask site (post-T7, same file)
           (m.active_window & recent_window_bits).count_ones()
```

The only new wire-format concern is that step 3 (`ConstructionConfig`) is reconstructed on non-root ranks by the CLI / Python layer. That path already carries `cut_activity_tolerance`, `budget`, and 20+ other scalar fields; adding one more `u32` is a single extra line in the reconstruction code (grep for `cut_activity_tolerance:` in `crates/cobre-cli/` and `crates/cobre-python/` to find the broadcast-rebuild sites).

### 6.5 Validation rules

**Range:** `1..=32`. `0` would produce a zero mask and disable the classifier entirely; `>32` overflows the u32 bitmap. Validation lives at the `StudyParams::try_from` site (step 2 above), alongside the existing `budget` and `cut_activity_tolerance` checks, returning `SddpError::Validation` on violation.

**Default:** `5`, matching the current hardcoded value. Introduce as `pub const DEFAULT_BASIS_ACTIVITY_WINDOW: u32 = 5;` near the top of `basis_reconstruct.rs` (keeping the old constant `RECENT_WINDOW_K` as an alias or removing it entirely once all sites migrate). A visible constant in the code lets tests and doc examples reference it without duplicating the magic number.

**Warnings (optional):** if `basis_activity_window > training.max_iterations`, emit a warning — the window is wider than the study will ever run, so high bits are unreachable. Not an error (still correct), but usually a misconfiguration.

### 6.6 Naming

Two candidates:

1. **`basis_activity_window`** — describes the semantic purpose ("how wide is the activity window the basis-reuse classifier consults"). Matches the spirit of `cut_activity_tolerance` naming — both are "cut/basis + activity + \<scalar>". TOML-friendly.
2. **`classifier_window_k`** / **`recent_window_k`** — literal mirror of the internal constant name. Shorter but less self-describing to end users who don't read source.

**Recommended: `basis_activity_window`.** It is the user-facing name; the internal constant can remain `RECENT_WINDOW_K` inside `basis_reconstruct.rs` for continuity with the existing rustdoc and CEPEL reference, fed from the config.

### 6.7 Composition with G1, G2, G3

- **G1** (seed `active_window: 1` at creation): independent of `k`. Seeding bit 0 is correct for any window size ≥ 1.
- **G2** (align demotion mask to the classifier's window): **fused by G4**. Once `recent_window_bits` is derived from the runtime `basis_activity_window`, both the classifier and the demotion site use the same mask automatically. G2 becomes a one-line change at the demotion site ("use `recent_window_bits` not `count_ones()` over the full u32") rather than requiring a separate constant.
- **G3** (recency-aware sort key): composes cleanly. Whichever option is adopted (`last_active_iter`, `leading_zeros`, or lex-pair), the `popcount`-in-window component uses the runtime mask.

The landing order that minimises redundant test churn:

```
T9 (G4: plumb the window as config)
  │
  ├─ T7 (G2: demotion mask uses the same runtime mask)
  │
  └─ T8 (G3: add secondary recency key)

T6 (G1) parallel to the above — no shared code path.
```

### 6.8 Burden assessment — to answer the user's "it has to be done" framing directly

Yes, the plumbing is wide: ~15 files touched, ~70 lines added, one new config field through `cobre-io` / `cobre-sddp` / `cobre-cli` / `cobre-python`. But:

1. **The pattern is already proven.** `cut_activity_tolerance` went through the exact same chain. Copying its shape is mechanical — grep for the field name in the workspace, add a sibling at every hit.
2. **The Python parity hard rule is easy to satisfy.** Every output file the CLI writes must also be writable by `cobre-python`; for config fields, every field the CLI reads must also be readable by `cobre-python::run`. One sibling field addition in `crates/cobre-python/src/run.rs` is enough.
3. **The MPI determinism hard rule is automatically preserved.** The field is a `u32` scalar copied verbatim into the broadcast payload — no floating-point, no collection ordering, no order-dependent computation. Rank-invariant by construction.
4. **It unblocks AD-7's original intent.** Epic 06 AD-7 said "A/B sub-test `k ∈ {3, 5, 10}` is optional; not a gate." Today that sub-test is impossible without three rebuilds; after G4 it's a config toggle. A/B #7 can run the sub-test in-line with the primary A/B.

The alternative — keeping the constant hardcoded — means that whenever a downstream user finds their study wants a different `k`, the only path is to fork the source. That is not a defensible posture for a production solver.

---

## 7. Interaction of G1, G2, G3, G4 with determinism

All four proposed fixes preserve the hard rules in `CLAUDE.md`:

1. **Bit-for-bit identical regardless of input ordering** — every fix uses deterministic inputs (u32 bitmap, u64 iteration counter, u32 slot number) plus stable sort.
2. **MPI parity at 1/2/4 ranks** — `active_window` is `allreduce(BOR)`-unified (`backward.rs:1118-1154`); `last_active_iter` is already MPI-synced via the existing per-slot metadata allreduce. Both are rank-invariant.
3. **No hot-path allocation** — fixes G2 and G3 change the sort key construction but do not change the pre-allocation shape. G1 changes a struct initialiser, not an allocation site.
4. **No `CapturedBasis` wire format change** — Epic 06 AD-6 preserved.
5. **No runtime flag** — Epic 06 AD-5 preserved.

Clippy will happily accept any of the options. The test fixture impact is limited to (i) the `scheme_1_*` test family (5 tests introduced by T2) which need updated expected counts, and (ii) adding the missing recency-aware test case.

---

## 8. Testing strategy additions

### 8.1 Additional unit tests needed

For any Gap addressed:

```rust
// G1 — fresh cut receives LOWER guess on its first LP encounter
#[test]
fn classifier_returns_lower_for_freshly_generated_cut_same_iteration() {
    let mut pool = CutPool::new(10, 2, 1, 0);
    pool.add_cut(0, /*iteration=*/5, /*forward_pass_index=*/0, 1.0, &[1.0, 0.5]);
    // add_cut must initialise active_window = 1 (G1 fix).
    assert_eq!(pool.metadata[0].active_window, 1);
    // Classifier must see LOWER because bit 0 is set.
    ...
}
```

```rust
// G2/G3 — demotion picks stale-but-heavy, keeps fresh-but-light
#[test]
fn scheme_1_recency_aware_sort_preserves_fresh_cuts() {
    // Build stored basis with two preserved-BASIC cuts:
    //   A: active_window = 0xFFF0_0000 (12 binds, all >= 20 iters ago)
    //   B: active_window = 0x0000_0002 (1 bind, last iter)
    // Need one demotion.
    // Assert: A demoted, B preserved.
    ...
}
```

```rust
// G4 — configurable window affects classifier and demotion consistently
#[test]
fn classifier_respects_runtime_basis_activity_window() {
    // Seed active_window = 0b10_0000 (bit 5, outside k=5 but inside k=10).
    // With basis_activity_window=5: classifier returns BASIC (out-of-window).
    // With basis_activity_window=10: classifier returns LOWER (in-window).
    ...
}

#[test]
fn study_params_rejects_basis_activity_window_out_of_range() {
    // basis_activity_window = 0 → SddpError::Validation.
    // basis_activity_window = 33 → SddpError::Validation.
    // basis_activity_window = 5 → Ok.
    // basis_activity_window = None → Ok (defaults to 5).
    ...
}
```

### 8.2 Integration test impact

- `test_backward_cache_hit_rate.rs` — threshold ≥ 0.95. All fixes preserve this by construction (they only change which cuts are tight-guessed, not which slots are preserved).
- `test_backward_cache_reduces_pivots.rs` — threshold unchanged, but with G1 + G2 + G3, should tighten further. If A/B #7 shows sizeable improvement, this is the place to lock it in.
- `reconstructed_basis_preserves_invariant_on_baked_truncation` — unaffected. The truncation path does not touch the classifier.

### 8.3 D-suite byte-identity

At iter=1 with G1 applied, the classifier fires LOWER on the first-iteration generated cohort, whereas without G1 it fires BASIC. This **does** cause iter=1 drift on the `simplex_iterations` and `basis_consistency_failures` columns. Since these are observability columns already allowlisted for Epic 06, the drift is acceptable. Operational columns (hydro, thermal, cost) remain byte-identical because the LP optimum is unchanged — only the warm-start quality differs.

G4 alone (with default `basis_activity_window = 5`) produces byte-identical output to the pre-G4 tree — the default matches the current compile-time constant. D-suite drift is therefore zero for G4 unless a test fixture explicitly sets a non-default value.

Update `scripts/compare_v045_reference.py`'s `EXPECTED_DRIFTS` with a 1-line reference to Gap G1 if G1 ticket lands. G4 does not need an allowlist entry.

---

## 9. Ticket candidates

Proposed atomic tickets, following the Epic 06 ticket format:

### T6 — Seed `active_window: 1` on cut generation (Gap G1)

**Scope:** change `CutPool::add_cut` at `cut/pool.rs:251-257` to initialise `active_window: 1`. Update rustdoc on `CutMetadata.active_window` (`cut_selection.rs:86-97`) to describe the "bound or generated" semantics. Update `add_cut_initialises_active_window_to_zero` test (currently asserts 0) to assert the new value. Add `classifier_returns_lower_for_freshly_generated_cut_same_iteration` test.

**Files:** `cut/pool.rs` (1 line), `cut_selection.rs` (rustdoc, ~5 lines), `basis_reconstruct.rs` tests (~30 lines), D-suite drift allowlist (1 line).

**Estimated effort:** 1 day. Independent of T2/T3 — lands at any point after T1.

### T7 — Align demotion window with classifier window (Gap G2)

**Scope:** at `basis_reconstruct.rs:355-358`, mask the popcount computation by `RECENT_WINDOW_BITS`. Add `demotion_sort_ignores_bits_outside_recent_window` regression test. No other code changes.

**Files:** `basis_reconstruct.rs` (1 line body change, ~30 lines of tests).

**Estimated effort:** 0.5 day. Independent of T6. Lands alongside or after T2 / T3.

### T8 — Recency-aware demotion sort (Gap G3)

**Scope:** adopt one of three options per §5.3. Recommendation: Option 3 (lex `(pop_low5, last_active_iter)`) for best discrimination with minimal risk. If A/B #7 gates show no meaningful improvement from T7 alone, T8 is the next step.

**Files:** `basis_reconstruct.rs` (~10 lines of body change, ~40 lines of tests).

**Estimated effort:** 1.5 days. Depends on T7 (shared code path). Dispatch order: T7 → measure → T8 if needed.

### T9 — Promote `RECENT_WINDOW_K` to a configurable parameter (Gap G4)

**Scope:** plumb `basis_activity_window: Option<u32>` (TOML) → `u32` (runtime, default 5) through the same chain as `cut_activity_tolerance`. Validate range `1..=32` at `StudyParams::try_from`. Pipe to `ReconstructionSource` via `StageInputs`. Replace `RECENT_WINDOW_BITS` in `basis_reconstruct.rs` with a locally-derived `(1u32 << window) - 1` mask. Preserve `DEFAULT_BASIS_ACTIVITY_WINDOW = 5` as a named constant.

**Files:**

- `crates/cobre-io/src/config.rs` — add `basis_activity_window: Option<u32>` to `CutSelectionConfig` with `#[serde(default)]` (~5 lines + schema doc).
- `crates/cobre-sddp/src/config.rs` — add `basis_activity_window: u32` to `CutManagementConfig`, default 5 (~3 lines).
- `crates/cobre-sddp/src/setup/params.rs` — parse in `StudyParams::try_from`, validate range, forward through `into_construction_config` (~10 lines).
- `crates/cobre-sddp/src/setup/mod.rs` — store on `StudySetup` (~3 lines).
- `crates/cobre-sddp/src/setup/accessors.rs` — `fn basis_activity_window(&self) -> u32` (~3 lines).
- `crates/cobre-sddp/src/setup/orchestration.rs` — hand-off to training / simulation (~2 lines).
- `crates/cobre-sddp/src/training.rs` + `crates/cobre-sddp/src/simulation/pipeline.rs` — carry on relevant state structs, populate `StageInputs` (~5 lines each).
- `crates/cobre-sddp/src/stage_solve.rs` — add `basis_activity_window: u32` field to `StageInputs`, pass into `ReconstructionSource` (~3 lines + test-fixture updates).
- `crates/cobre-sddp/src/basis_reconstruct.rs` — add `basis_activity_window: u32` to `ReconstructionSource`, derive mask at function entry, replace all references to `RECENT_WINDOW_BITS` (~10 lines body + ~40 lines of test-fixture updates across the ~15 existing test call sites).
- `crates/cobre-cli/src/commands/*.rs` + `crates/cobre-python/src/run.rs` — forward the new field on the broadcast/rebuild path (~5 lines).
- `crates/cobre-sddp/tests/` — add `classifier_respects_runtime_basis_activity_window` and `study_params_rejects_basis_activity_window_out_of_range` tests (~60 lines).
- `book/` — document the new config field alongside `cut_activity_tolerance` (~15 lines of docs).

**Estimated effort:** 2 days. Independent of T6/T7/T8 in scope but should land BEFORE T7 (G2's fix is "use the runtime mask at the demotion site", which requires G4's mask to exist). Dispatch order: T9 → T7 → T8. T6 parallel.

**Test fixture update volume:** ~15 existing `reconstruct_basis` test call sites will need `basis_activity_window: DEFAULT_BASIS_ACTIVITY_WINDOW` added to their `ReconstructionSource` literal — same shape as the T2 fixture updates for `cut_metadata`. Mechanical.

### Ticket sequencing options

**Option A — fix before T3:** land T6 + T9 + T7 (+ optionally T8) before T3 activates the classifier in production. A/B #7 measures the full stack AND can sub-test `k ∈ {3, 5, 10}` per AD-7's original intent. Pros: one clean measurement; AD-7 sub-test becomes possible. Cons: delays T3 by ~4-5 days.

**Option B — T3 first, then T6/T7/T8/T9 as follow-ups:** A/B #7 measures the current classifier design. If it meets the 5% threshold, the fixes are optional polish. If it doesn't (or worse, regresses), T6/T7/T8/T9 are candidate interventions before declaring null-result and reverting. Pros: faster path to production signal. Cons: may produce a null-result A/B that prompts rollback of work that was fixable; AD-7 sub-test remains impossible until G4 lands anyway.

**Recommended:** **Option A**, specifically **T6 + T9 + T7** as a bundled "classifier refinement" ticket sequence before T3. T8 deferred until A/B #7 shows whether the simpler G2 fix alone is sufficient. T9 is load-bearing — without it, T7 has to invent its own runtime mask or remain hardcoded, and AD-7's sub-test is locked out.

---

## 10. Risks and open questions

### 10.1 Risks

**R-G1-1 — Over-classification of LOWER on generating cohort.** If fresh cuts are systematically slack at the adjacent stage's trial point (contra our prior), G1 trades one wrong-default for another wrong-default. Pivot cost is roughly symmetric, so no correctness risk — only wall-time neutrality or minor regression.

**R-G1-2 — Interaction with Scheme 2 fallback.** G1 increases the count of activity-driven LOWER guesses per solve. If the preserved-BASIC pool is small, Scheme 2 fires more often. The fallback logic is well-tested (`scheme_2_fallback_when_deficit_exceeds_candidates`), but the rate at which it fires affects telemetry interpretation.

**R-G2-1 — Masked popcount reduces discrimination power.** The key range collapses from 0..=32 to 0..=5, so more ties. Stable sort + slot-order tie-break still guarantees determinism. Not a correctness risk, but empirically more cuts will be demoted "in ties" by slot-order rather than by popcount.

**R-G3-1 — `last_active_iter` stale for cuts that have never bound.** Cuts that have never bound have `last_active_iter = iteration_generated`. If many cuts land with the same `iteration_generated`, sorting by `last_active_iter` ascending collapses them all to the same bin. Slot-order tie-break still determinises.

**R-G4-1 — Runtime-derived mask is slightly slower than compile-time constant.** Deriving `(1u32 << k) - 1` on function entry is one shift + one subtract per `reconstruct_basis` call. On convertido (~50,000 calls per run) this is negligible microseconds in the noise floor; `cargo flamegraph` will not resolve it. No hot-path impact.

**R-G4-2 — Config-field drift between `cobre-io` and `cobre-sddp`.** Historically every new `CutSelectionConfig` field has also needed a sibling on `CutManagementConfig` and a parse site in `StudyParams`. Missing one of the three leaves the field silently ignored. Mitigation: the TOML round-trip test at `crates/cobre-io/src/config.rs:1776-1790` already covers serialisation parity for this struct; extend it with `basis_activity_window`. An integration test that writes a TOML with `basis_activity_window = 10`, runs training, and reads back the effective value from `StudySetup::basis_activity_window()` catches end-to-end drift.

**R-G4-3 — Python-parity hard rule.** `crates/cobre-python/src/run.rs` must accept the new config field in its kwargs-to-Config conversion, otherwise `cobre-python` users would silently run with the default. One sibling line alongside the existing `cut_activity_tolerance` handling.

### 10.2 Open questions for the user

1. **Should G1 ticket seed `active_window: 1` (bit 0 only) or `0b11111` (full recent window)?** The latter is more aggressive — it forces LOWER for the next 5 iterations without relying on bind events. Less sensitive to first-encounter timing but more aggressive when the fresh cut is actually slack.

2. **Is the A/B #7 Arm design flexible enough to accommodate an Option C "Parts A+B with G1+G2" and Option D "Parts A+B with G1+G2+G3"?** The current spec at design-doc §12 lists Arms A-D but not these variants. If they run, T4 scope grows.

3. **Are there additional cuts-related metadata fields that could feed a richer demotion key?** `iteration_generated`, `forward_pass_index`, and `active_count` are all available. For example, `active_count / (current_iter - iteration_generated + 1)` is a per-iteration bind-rate that doesn't wrap at 32 iterations like `active_window` does. Worth considering if Epic 06 gets a v2 pass.

4. **Should the rustdoc on `CutMetadata.active_window` call out the two consumer sites' windows explicitly?** Currently it describes the shift/update rules but not how consumers use the data. A sentence noting the classifier/demotion-sort asymmetry (or its resolution after T7) would prevent future regressions.

5. **For G4, should `basis_activity_window` live under `training.cut_selection` (alongside `cut_activity_tolerance`) or under a new `training.basis_reuse` sub-section?** The former matches existing locations and keeps related knobs together; the latter foreshadows a future "basis reuse strategy" section if more knobs are added (e.g., demotion-strategy selection per §5.3). Recommendation: stay under `training.cut_selection` for now — avoid premature structure.

6. **For G4, should the default 5 live in `cobre-sddp` (consumer side) or `cobre-io` (TOML side)?** Today `cut_activity_tolerance`'s default `1e-6` lives in `CutManagementConfig::default()` on the sddp side, with `cobre-io` treating it as `Option<f64>` defaulting to `None` (unwrap_or in `StudyParams`). Consistency argues for the same pattern. Recommendation: default lives on `CutManagementConfig`, mirrored by a `pub const DEFAULT_BASIS_ACTIVITY_WINDOW: u32 = 5;` in `basis_reconstruct.rs` for test/doc convenience.

---

## 11. Artifacts referenced

- **Authoritative spec:** `docs/design/epic-06-cut-basis-reconstruction.md`
- **Epic overview:** `plans/backward-basis-cache/epic-06-cut-basis-reconstruction/00-epic-overview.md`
- **T1 (shipped):** commit `1546b95` — `feat(sddp): add active_window to CutMetadata (Epic 06 T1)`
- **T2 (in progress):** uncommitted changes on `basis_reconstruct.rs` (adds classifier + Scheme 1/2 demotion)
- **T3 (pending):** ticket `plans/backward-basis-cache/epic-06-cut-basis-reconstruction/ticket-003-route-active-cuts-through-stage-solve.md`

**Key source locations cited in this report:**

- `crates/cobre-io/src/config.rs:183-238` — `CutSelectionConfig` (TOML-facing, G4 adds a sibling field)
- `crates/cobre-sddp/src/config.rs:150-202` — `CutManagementConfig` (runtime, G4 adds a sibling field)
- `crates/cobre-sddp/src/setup/params.rs:131-189` — `cut_activity_tolerance` parse + forward pattern (G4 template)
- `crates/cobre-sddp/src/setup/params.rs:201-230` — `ConstructionConfig` (MPI broadcast-reconstructed struct)
- `crates/cobre-sddp/src/setup/mod.rs:165` — `StudySetup.cut_activity_tolerance` storage
- `crates/cobre-sddp/src/setup/accessors.rs:272-274` — `StudySetup::cut_activity_tolerance()` accessor
- `crates/cobre-sddp/src/setup/orchestration.rs:51` — orchestration-level hand-off
- `crates/cobre-sddp/src/cut_selection.rs:57-97` — `CutMetadata`
- `crates/cobre-sddp/src/cut/pool.rs:245-262` — `CutPool::add_cut`
- `crates/cobre-sddp/src/cut/pool.rs:282-295` — `CutPool::active_cuts`
- `crates/cobre-sddp/src/backward.rs:670-677` — per-opening bind detection
- `crates/cobre-sddp/src/backward.rs:735-743` — per-trial-point OR aggregation
- `crates/cobre-sddp/src/backward.rs:948-1060` — reverse backward loop + cut insertion
- `crates/cobre-sddp/src/backward.rs:1118-1154` — MPI allreduce(BOR)
- `crates/cobre-sddp/src/training.rs:1057-1066` — end-of-iteration shift
- `crates/cobre-sddp/src/basis_reconstruct.rs:83-95` — `RECENT_WINDOW_K` / `RECENT_WINDOW_BITS`
- `crates/cobre-sddp/src/basis_reconstruct.rs:338-378` — classifier loop
- `crates/cobre-sddp/src/basis_reconstruct.rs:386-407` — Scheme 1 demotion
- `crates/cobre-sddp/src/basis_reconstruct.rs:410-428` — Scheme 2 fallback
- `crates/cobre-sddp/src/stage_solve.rs:170-198` — call site (T3 pending)
- `crates/cobre-solver/src/highs.rs:1144-1150` — HiGHS dimension-extension BASIC fill

---

## 12. Summary — what each gap costs if shipped unfixed

| Gap | Population affected per iter (convertido 50-scenario baseline)                                                                  | Signal lost                                                                                         | Simplex / study consequence                                                                                                                                            |
| --- | ------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| G1  | ~5,850 fresh cuts (50 × 117 stages)                                                                                             | Domain prior: "generating event = bound"                                                            | Warm-start defaults to BASIC for exactly the cohort where LOWER is most defensible. Pivot overhead on these cuts is fully unmitigated.                                 |
| G2  | Every demotion decision where fresh and stale cuts coexist                                                                      | "What counts as recent" alignment with classifier                                                   | Demotion occasionally picks cuts the classifier has already decided are irrelevant (bits 5..31). Wasted flips.                                                         |
| G3  | Same as G2, pathological whenever pool has varied bind ages                                                                     | Recency vs. cumulative volume                                                                       | Demotion can swap fresh-active cuts for stale-inactive cuts. Direct pivot regression when the fresh cut was correctly BASIC and the stale cut's demotion was harmless. |
| G4  | Every study run (hardcoded `k = 5` never tuned against cobre scale); every end user who wants to tune `k` for their study shape | AD-7's explicit sub-test intent (`k ∈ {3, 5, 10}`); end-user ability to tune without forking source | A/B #7 cannot evaluate window-size sensitivity. Downstream consumers have no path to adapt the solver to their study's convergence profile short of patching source.   |

If A/B #7 produces a null result, **any of these four gaps could be responsible**. Investigating after the fact means re-running the full measurement stack; fixing before means one clean A/B with AD-7's sub-test embedded.
