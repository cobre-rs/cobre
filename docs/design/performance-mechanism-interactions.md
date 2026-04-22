# Cobre SDDP — Performance-Mechanism Interactions

**Status:** Living document — updated 2026-04-22 after the transient-G1 A/B on
`convertido_backward_basis`.

**Scope:** How cut selection, activity budget, basis reconstruction (classifier

- Scheme 1/2 invariant preservation), and the sliding-window activity bitmap
  interact across a training iteration. Includes empirical findings from the
  Epic 06 A/B series that validated (or falsified) our design hypotheses.

**Audience:** Maintainers touching any of `cut/pool.rs`, `cut_selection.rs`,
`basis_reconstruct.rs`, `training.rs`, or `stage_solve.rs`. Read before
changing the `active_window` semantics, the classifier predicate, or the
end-of-iter shift.

---

## 1. The four mechanisms at a glance

| Mechanism                | What it decides                | Field(s) consulted                               | When it fires                  | Can remove cuts? | Can change LP status? |
| ------------------------ | ------------------------------ | ------------------------------------------------ | ------------------------------ | ---------------- | --------------------- |
| **Cut selection**        | _Which cuts exist_             | `active_count`, `last_active_iter`               | End of each iter backward      | Yes (deactivate) | No                    |
| **Activity budget**      | _How many cuts exist_          | `last_active_iter`                               | Right after cut selection      | Yes (evict)      | No                    |
| **Activity bitmap**      | _Warm-start hint signal_       | Binding observations via MPI BitwiseOr           | Per stage in backward pass     | No               | No                    |
| **Basis reconstruction** | _Per-row LP status at warm-up_ | `active_window & (RECENT_WINDOW_BITS｜SEED_BIT)` | Per LP solve with cached basis | No               | Yes (LOWER/BASIC)     |

Key invariant: the first three mechanisms run **once per iteration**, always in
the same order. Basis reconstruction runs **once per LP solve** and reads the
state the first three leave behind.

---

## 2. Life of a cut (per-iteration timeline)

The ordered sequence inside one SDDP iteration `i`:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ITER i FORWARD PASS                                                    │
│    for each (fp, stage t):                                              │
│      solve LP with baked_templates[t] (built at end of iter i-1)        │
│        → if stored basis exists:                                        │
│            reconstruct_basis(…)  — classifier + Scheme 1/2              │
│        → simplex pivots to optimum                                      │
│      record x̂_{t, fp, i}  (for backward pass)                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ITER i BACKWARD PASS  (walks t = T−1 → 0)                              │
│    for each stage t (top-down):                                         │
│      for each (fp, ω=0..19 openings):                                   │
│        solve LP ← reconstruct_basis(…) if cached at ω=0                 │
│      allreduce(BitwiseOr) → OR bit 0 of active_window                   │
│        on slots in pool[t] that were binding at any opening             │
│      (if fp at stage t+1 backward pass branch): add_cut to pool[t]      │
│        → NEW cut gets active_window = SEED_BIT (G1 transient)           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ITER i END-OF-BACKWARD BOOKKEEPING                                     │
│    1. cut selection    → mark active[slot] = false for pruned cuts      │
│       (uses active_count, last_active_iter)                             │
│    2. activity budget  → evict to max_active_per_stage                  │
│       (uses last_active_iter, typically oldest-first)                   │
│    3. active_window shift: active_window = (aw & !SEED_BIT) << 1        │
│       → ages recent-binding bits; clears G1 transient seed              │
│    4. template baking  → rebuild baked_templates[t] from active cuts    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                              iter i+1 …
```

---

## 3. The sliding-window activity bitmap

`CutMetadata.active_window: u32` is a 32-bit sliding bitmap of per-iteration
binding activity.

> **Mask terminology.** Throughout this document, `RECENT_WINDOW_BITS` refers
> to the runtime-derived mask
> `recent_window_bits = (1u32 << basis_activity_window) - 1`, computed at the
> top of `reconstruct_basis` from the validated TOML parameter (default 5,
> range 1..=31). Before T5b (commit `93cc69f`) this was the compile-time
> constant `RECENT_WINDOW_BITS = 0b11111`; the bit semantics are unchanged.

Two read consumers:

1. **Classifier** (`basis_reconstruct.rs`): a cut with `aw & (RECENT_WINDOW_BITS | SEED_BIT) != 0` is classified LOWER (tight guess) for the NEW-cut branch.
2. **Scheme 1 sort key** (`basis_reconstruct.rs`): `(aw & RECENT_WINDOW_BITS).count_ones()` — the popcount in the recent window. Only the masked bits count; SEED_BIT is excluded from the popcount deliberately (it's a within-iter hint, not a history record).

Two write consumers:

1. **`add_cut`** (`cut/pool.rs`): seeds `active_window = SEED_BIT` (bit 31) on every freshly generated cut. The G1 _transient_ seed — see §6.
2. **Backward pass** (`backward.rs`): `allreduce(BitwiseOr)` on bit 0 across ranks, per stage. Any rank observing the cut binding at any opening contributes; the result OR's into the pool's `active_window` after each stage in the backward walk.

One transformation:

- **End-of-iter shift** (`training.rs`): `active_window = (aw & !SEED_BIT) << 1`. Clears SEED_BIT first (so the G1 seed does not carry into iter i+1), then left-shifts the binding-observation bits by one position (bit 0 → bit 1, etc.) so the next iteration's bit 0 starts empty.

### Encoding summary

| Bit position               | Source                              | Lifetime                                                                                                              |
| -------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Bit 0                      | Observed binding in current iter    | Until end-of-iter shift                                                                                               |
| Bits 1–4 (RECENT, default) | Observed binding in iters i-1..i-4  | Shifted out after `basis_activity_window` iters (runtime-configurable, default 5 via `DEFAULT_BASIS_ACTIVITY_WINDOW`) |
| Bits 5–30                  | Observed binding in iters i-5..i-30 | Not consulted by classifier or sort                                                                                   |
| Bit 31 (SEED_BIT)          | G1 transient seed (`add_cut`)       | Cleared at end-of-iter before shift                                                                                   |

The classifier thus "sees" both real recent activity (bits 0–4) and the
within-iter generating event (bit 31). The sort uses real activity only,
because promoting a cut by its generating event alone is information-poor.

---

## 4. Basis reconstruction: the classifier and Scheme 1/2

When a stored basis exists for a stage solve, `reconstruct_basis` walks every
cut row in the target LP and assigns it a status:

### Preserved cuts

A cut whose slot appears in the stored basis's `cut_row_slots` is _preserved_:
its stored row status is copied verbatim. The classifier is **not consulted**
for preserved cuts — their status comes from the LP that captured the basis.

### New cuts

A cut whose slot is not in the stored basis is _new_. The classifier runs:

```rust
if active_window & (RECENT_WINDOW_BITS | SEED_BIT) != 0 {
    // Activity-guided tight guess → NONBASIC_LOWER (new_tight++)
} else {
    // No recent binding signal → BASIC (new_slack++)
}
```

### Scheme 1: symmetric promotion

Each LOWER-classified new cut decrements the running basic count by 1
(relative to HiGHS's BASIC-default extension). To preserve
`col_basic + row_basic == num_row`, we **promote** one preserved-LOWER cut
to BASIC for each new-LOWER guess.

Candidates are the _preserved-LOWER_ cuts collected during the walk. The
lowest-ranked `scheme1_count` candidates by a three-component lex key are
promoted; the intuition is that the least-binding, most-stale preserved
cut is the lowest-cost wrong guess to move to BASIC.

**Sort key**: `(popcount, last_active_iter, insertion_idx)` ascending:

1. **popcount** (`u32`, range 0..=5): `(aw & RECENT_WINDOW_BITS).count_ones()`.
   Primary. Cuts not bound in the recent window promoted first.
2. **last_active_iter** (`u64`): from `CutMetadata`. Secondary. Among
   popcount-tied candidates, the cut with the oldest most-recent binding is
   more stale → promoted first.
3. **insertion_idx** (`u32`): position in the candidates vec at push time,
   strictly increasing per reconstruction. Tertiary. Guarantees no key ties
   → the unstable partial-select algorithm is deterministic by construction.

**Algorithm**: `select_nth_unstable_by_key` (O(n) average) instead of a full
stable sort (O(n log n)). We only need the smallest `scheme1_count` (bounded
by `forward_passes`) entries, not a fully-ordered vec. The partial-select
call is guarded: skipped when `scheme1_count == available_candidates`
(promote all, no selection needed) or zero.

**Correctness**: the _set_ of promoted rows is deterministic because all lex
keys are unique via the tertiary `insertion_idx`. The promotion loop writes
`HIGHS_BASIS_STATUS_BASIC` to distinct `row_status[row_idx]` entries, so the
order of writes does not affect the final basis. MPI parity preserved by
construction — every rank builds the same candidates list and selects the
same set.

**Scale rationale**: at convertido scale (~150 cuts/stage, 11,800
reconstructions) the cumulative sort cost is ~24 ms — noise. At production
scale (~10k cuts/stage, ~1.2M reconstructions per 200 FP × 50 iter run),
partial selection saves roughly 10× cumulative sort time (hundreds of
seconds → tens of seconds). See commit `beb2ea3` (T5a) for implementation.

### Scheme 2: tail fallback

If there are **more** LOWER-classified new cuts than preserved-LOWER candidates
available for promotion (adversarial edge case, common in early iterations
before the preserved-LOWER pool has grown), Scheme 1 alone cannot preserve the
invariant. Scheme 2 walks the most-recently-classified new-LOWER rows in
reverse order and flips them back to BASIC until the deficit is zero.

### Forward vs backward invariant paths

- **Backward path**: Scheme 1 + Scheme 2 preserve the invariant by construction; the downstream `enforce_basic_count_invariant` is a safety-net no-op on the happy path.
- **Forward path**: cut selection may have dropped cuts whose stored row status was BASIC, creating an _excess_ of BASIC (not a deficit). The enforcer demotes trailing BASIC rows to LOWER until the invariant holds. This is the only case where the enforcer does real work.

---

## 5. Why the forward/backward asymmetry matters

The A/B data revealed a stark asymmetry that is consequential for any future
design changes to the classifier:

| Phase        | When classifier runs                      | Why it matters                                                                                                                                                                                               |
| ------------ | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Forward**  | Every stage solve with cached basis       | Rarely fires on "new cuts" — by end-of-iter i-1, all currently-active cuts are in the stored basis, so the "new cut" branch is hit by ≈0 rows per solve                                                      |
| **Backward** | Every stage solve with cached basis (ω=0) | Frequently fires on "new cuts" — cuts added earlier in this same backward walk (at stages above t) are _new_ relative to the stored basis (which was captured at end of iter i-1, before these cuts existed) |

Cuts added during iter `i`'s backward pass do not exist in the stored basis
from end of iter `i-1`. When iter `i`'s backward walk reaches stage t, cuts
generated at stages above t during this iter are in pool[t] but not in
`cut_row_slots` of the stored basis ⇒ they enter the classifier's new-cut
branch.

Additional amplification in backward: at stage t, ω=0's reconstructed basis
seeds the warm-start for all 20 subsequent openings (ω=1..19) via HiGHS's
retained factorization. A poor reconstruction at ω=0 cascades through 20 LP
solves. The same classifier call therefore has roughly 20× the effective impact
in backward vs forward.

Concrete numbers from `convertido_backward_basis` (118 stages × 50 FP × 3
iters):

- Forward simplex-iterations delta between designs: ±9k (out of 17.4M)
- Backward simplex-iterations delta between designs: ±2.1M (out of 67M)

The backward pass therefore dominates any classifier-design comparison. Future
A/B studies should report backward LP-time separately.

---

## 6. Activity-bitmap G1 seed: the "generating event"

A cut generated at state x̂*t via stage t+1's dual is \_tight at that x̂_t by
construction*. If stage t's LP at (fp=f, ω=0) is solved at the same x̂_t
(which it is, immediately after in the same backward walk), the cut is very
likely still tight.

The **G1 seed** encodes this by setting a bit in `active_window` at `add_cut`
time, so the classifier fires LOWER on the cut at its first LP encounter
_within the same iteration's remaining backward stages_.

### Why the seed must be transient

If G1's bit persists across iterations (the original design — seed bit 0,
shift advances it through the recent window), the classifier treats the cut
as "recently active" in iter i+1 even though no real binding observation has
occurred. For a cut to actually still be tight at a _different_ x̂_t (the
iter i+1 forward pass samples new noise), it must survive independent of its
generation point.

- **Deterministic / low-variance cases** (e.g., 1 scenario per stage): iter i+1 visits approximately the same x̂_t as iter i, so the generating tightness transfers. G1 persistence helps.
- **Stochastic / high-FP cases** (e.g., 50 forward passes with real noise): each iter visits distinct x̂_t values. Generating tightness does not transfer. G1 persistence biases the classifier toward spurious LOWER guesses on preserved cuts' sort order, and costs extra simplex pivots.

### The transient design (current HEAD)

- **`SEED_BIT = 1 << 31`**, positioned outside `RECENT_WINDOW_BITS` so it does not contribute to the Scheme 1 popcount sort.
- `add_cut` sets `active_window = SEED_BIT` (not `active_window = 1`).
- Classifier predicate: `aw & (RECENT_WINDOW_BITS | SEED_BIT) != 0`. The seed fires the classifier within-iter.
- End-of-iter shift: `active_window = (aw & !SEED_BIT) << 1`. Clears the seed before advancing the window.
- From iter i+1 onward, only genuine `allreduce(BitwiseOr)` binding observations drive the classifier.

---

## 7. Empirical findings (Epic 06 A/B series)

### Test case: `convertido_backward_basis`

118 stages, 50 forward passes, 3 iterations, local backend, 5 threads.

| Design                                   | Wall-clock           | Backward LP time          | Total simplex     |
| ---------------------------------------- | -------------------- | ------------------------- | ----------------- |
| T3 (pre-G1 baseline, classifier dormant) | 1005.6s              | 4,239,334 ms              | 82.6M             |
| T4+T5 (G1 persistent, bit 0)             | 1062.5s (+5.7%)      | 4,483,893 ms (+5.8%)      | 84.7M (+2.5%)     |
| **Transient-G1 (SEED_BIT)**              | **1006.4s (+0.08%)** | **4,241,880 ms (+0.06%)** | **84.7M (+2.5%)** |

Convergence (LB/UB trajectory) is bit-identical across designs — the policy
does not depend on warm-start basis.

Key insight: transient-G1 has the **same simplex count** as persistent G1 but
**lower solve time** (0.0630 ms/pivot vs 0.0666 ms/pivot). The difference is
in per-pivot cost — a healthier basis factorization because preserved-LOWER
cuts' sort keys are not biased by G1 legacy bits. Scheme 1 promotes cuts
whose only recent-window signal is a real binding observation, which is a
more reliable "this cut is stale" indicator than the G1-legacy bit.

### Test case: D03 churn fixture

3 stages, 3 forward passes, 8 iterations, LML1 + budget=6, 1 scenario per
stage (fully deterministic).

| Design                         | Simplex iterations (forward aggregate) |
| ------------------------------ | -------------------------------------- |
| T1/T2 (classifier dormant)     | 126                                    |
| T3a (classifier active, no G1) | 191                                    |
| T4 (G1 persistent)             | 123                                    |
| **Transient-G1 (SEED_BIT)**    | **184**                                |

On this deterministic fixture, G1 persistence is a clear win (−68 simplex vs
no-seed baseline). Transient-G1 loses this benefit because iter-to-iter x̂
drift is near-zero — the cross-iter G1 signal is genuinely informative here.

The D03 pin in `crates/cobre-sddp/tests/basis_reconstruct_churn.rs` was
updated from 120 to 184 to track the transient-G1 design. The pin's original
purpose (catch the `padding_state = x_hat` regression) is preserved — that
bug raises the count well outside the ±5 % band regardless of baseline.

### Design trade-off captured

Transient-G1 trades deterministic-case performance for stochastic-case
performance. The production workload (convertido and similar case types) is
stochastic, so the trade is net positive. D03 remains valuable as a
correctness fixture but is no longer the performance benchmark.

---

## 8. Interaction summary table

| Interaction                         | What happens                                                                                                                                                                                                                      | Where it lives                                                                  |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Cut selection → reconstruction      | Deactivated cuts excluded from `pool.active_cuts()`. Classifier never sees them.                                                                                                                                                  | `cut_selection.rs` deactivates; `basis_reconstruct.rs` iterates `active_cuts()` |
| Activity budget → reconstruction    | Evicted cuts excluded from `pool.active_cuts()`. Identical to cut selection's effect from the reconstruction's point of view.                                                                                                     | `training.rs` enforcement loop                                                  |
| Reconstruction → cut selection      | No direct coupling. Classifier does not touch `active_count` or `last_active_iter`. Warm-start hint is independent of cut lifecycle.                                                                                              | N/A (intentional separation)                                                    |
| Activity bitmap → cut selection     | No direct coupling. Cut selection reads `active_count` / `last_active_iter`, not `active_window`.                                                                                                                                 | N/A (intentional separation)                                                    |
| Reconstruction → activity bitmap    | Consumer only — classifier reads `active_window & (RECENT_WINDOW_BITS｜SEED_BIT)`; Scheme 1 sort reads masked popcount + `last_active_iter`. Neither writes.                                                                      | `basis_reconstruct.rs`                                                          |
| Backward pass → activity bitmap     | Sole writer during iteration. `allreduce(BitwiseOr)` on bit 0 per stage.                                                                                                                                                          | `backward.rs:1147-1149`                                                         |
| End-of-iter shift → activity bitmap | Sole writer between iterations. Clears SEED_BIT, then shifts.                                                                                                                                                                     | `training.rs:1064-1068`                                                         |
| `add_cut` → activity bitmap         | Sole writer outside the backward-pass path. Sets SEED_BIT.                                                                                                                                                                        | `cut/pool.rs:256`                                                               |
| Scheme 1 → HiGHS invariant          | Promotes preserved-LOWER → BASIC per new-LOWER classification via lex `(popcount, last_active_iter, insertion_idx)` partial selection (`select_nth_unstable_by_key`, O(n) average). Maintains `col_basic + row_basic == num_row`. | `basis_reconstruct.rs` Scheme 1 block                                           |
| Scheme 2 → HiGHS invariant          | Last-resort: overrides latest new-LOWER back to BASIC when Scheme 1 candidates are exhausted.                                                                                                                                     | `basis_reconstruct.rs:468-490`                                                  |
| Forward enforcer → HiGHS invariant  | Demotes trailing BASIC cut rows when cut selection dropped BASIC cuts (creating an excess). Backward path is a no-op by construction.                                                                                             | `basis_reconstruct.rs::enforce_basic_count_invariant`                           |

---

## 9. Design constants (current HEAD)

| Constant / runtime value        | Value                       | Location                                                          | Notes                                                                                                                                        |
| ------------------------------- | --------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `basis_activity_window`         | runtime, validated `1..=31` | TOML `training.cut_selection.basis_activity_window` (Option<u32>) | Classifier/sort-key mask width. `None` → `DEFAULT_BASIS_ACTIVITY_WINDOW`. Threaded end-to-end via `ConstructionConfig`.                      |
| `DEFAULT_BASIS_ACTIVITY_WINDOW` | `5`                         | `basis_reconstruct.rs`                                            | Inherited from CEPEL NEWAVE calibration; no cobre-scale empirical validation. Used when the TOML field is absent.                            |
| `DEFAULT_RECENT_WINDOW_BITS`    | `0b11111`                   | `basis_reconstruct.rs`                                            | Derived from `DEFAULT_BASIS_ACTIVITY_WINDOW`; used only in tests that want the default mask.                                                 |
| `SEED_BIT`                      | `1u32 << 31`                | `basis_reconstruct.rs`                                            | Transient G1 seed, cleared at end-of-iter. Must stay outside the configured `basis_activity_window` window — validation bounds enforce this. |
| `cut_activity_tolerance`        | config-driven               | `ConstructionConfig`                                              | Threshold for the binding-bit OR (`dual > cut_activity_tolerance`).                                                                          |

At `reconstruct_basis` entry, the effective mask is derived as
`recent_window_bits = (1u32 << source.basis_activity_window) - 1`. It is
a per-call, register-resident value — no per-cut overhead vs the former
compile-time constant.

---

## 10. Open questions and future work

### G3 — Recency-aware Scheme 1 sort tie-break — **shipped `beb2ea3` (T5a)**

Sort key upgraded from `popcount` (with slot-order tie-break) to lex
`(popcount, last_active_iter, insertion_idx)`. The tertiary `insertion_idx`
makes every key unique, which unlocked the second half of T5a: replacing
stable `sort_by_key` with `select_nth_unstable_by_key` for partial
selection. See §4 ("Scheme 1: symmetric promotion") for the full design.

**Expected impact:**

- **Correctness of heuristic**: Scheme 1 now promotes the more-stale cut
  when popcounts tie, instead of relying on insertion-order accident. Tied
  cases are common under transient-G1 (real binding observations are sparse
  in the 5-bit window).
- **Performance**: partial selection is O(n) average vs O(n log n) full
  sort. Zero-cost at convertido scale (~24 ms → ~2 ms cumulative sort time);
  ~10× reduction at production scale (~10k cuts, ~1.2M reconstructions).

**Tests added:** `promotion_sort_breaks_popcount_ties_by_last_active_iter`,
its reversed companion, and
`promotion_select_nth_produces_deterministic_promotion_set_at_n_200`.
All 1338 tests (was 1335) pass.

### G4 — Runtime-configurable `basis_activity_window` — **shipped `93cc69f` (T5b)**

The classifier/sort-key activity window is now a runtime TOML parameter:
`training.cut_selection.basis_activity_window: Option<u32>`, validated
`1..=31` at `StudyParams::from_config`, defaulting to `5` (the former
`RECENT_WINDOW_K`) when absent. Plumbed end-to-end through
`ConstructionConfig` → `StudySetup` → `CutManagementConfig` /
`SimulationConfig` → `StageInputs` → `ReconstructionSource`, mirroring the
proven `cut_activity_tolerance` chain. Wire format unchanged (travels via
the existing broadcast payload).

**Why this unlocks work:** the AD-7 sub-test `k ∈ {3, 5, 10}` from the
classifier-refinement-gaps review is now executable without a rebuild.
Different study shapes (short horizons vs long seasonal binding patterns)
can be tuned in config.

**Tests added:** `study_params_rejects_basis_activity_window_out_of_range`
(covers 0, 32, 5, None, 1, 31) and
`reconstruct_basis_honors_runtime_basis_activity_window` (same fixture
yields BASIC at window=5 vs LOWER at window=10). 3441 workspace tests
pass.

### Smarter G1 than the transient seed

The transient seed is a binary "generated this iter" flag. A richer signal
would weight the generating event by properties of the cut (e.g., steepness
of the gradient, proximity of the generating x̂ to recent forward-pass states).
The current design is intentionally simple; richer signals are speculative
until we observe a concrete workload where the transient seed underperforms.

### Classifier under cut-pool churn (LML1 deactivation)

When LML1 deactivates cuts, their slots remain in the pool but `active[slot]
= false` so they are excluded from `active_cuts()`. The stored basis may still
reference these slots. `reconstruct_basis`'s slot-identity path correctly
skips them via `slot_lookup.get(slot)` returning `None` for deactivated slots.
No known interaction pathology, but this path is less exercised on production
workloads; D03-churn remains our main regression fixture.

### Cross-opening basis aggregation

ω=0 solves populate the cached basis; ω=1..19 hot-start from it via HiGHS's
retained factorization. A future enhancement could aggregate basis information
across openings (median status per row?) for a more robust warm-start. Orthogonal
to the classifier design; see `docs/design/cross-opening-basis-aggregation.md`.

---

## 11. Required reading before modifying these mechanisms

- `.claude/architecture-rules.md` — the context-struct and arg-budget rules
- `crates/cobre-sddp/src/basis_reconstruct.rs` module docs
- `crates/cobre-sddp/src/workspace.rs` — `CapturedBasis` wire format ownership
- `crates/cobre-sddp/src/cut_selection.rs` — `CutMetadata` field semantics
- `docs/design/epic-06-cut-basis-reconstruction.md` — original design doc
- `docs/design/epic-06-classifier-refinement-gaps.md` — post-T3 review that produced G1–G4

When in doubt: the four write-sites of `active_window` (`add_cut`, the
backward-pass allreduce loop, the end-of-iter shift, and Scheme 1/2 local
overrides on `row_status`) are the complete surface. Any change to the field's
semantics must update all four consistently — and should be paired with an A/B
on both a deterministic fixture (D03 churn) and a stochastic workload
(convertido_backward_basis) before shipping.
