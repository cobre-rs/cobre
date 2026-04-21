# Cross-opening basis-status aggregation — design proposal

**Status:** Future optimization (logged 2026-04-21 during Epic 05 scoping)
**Origin:** User idea raised during Epic 05 ω-convention discussion, 2026-04-21
**Related:** `docs/design/smarter-cut-initialization.md`, `docs/design/cepel-sc-adoption.md`, `docs/design/backward-basis-as-single-source.md`, `plans/backward-basis-cache/epic-05-unified-basis-store/`

---

## Problem

Each backward pass at `(m, t)` solves `N` LPs — one per opening ω ∈ [0, N-1] — at
the same state `x_hat` but different noise realizations. Today's design (Epic 03
shipped, Epic 05 continuing) picks a single opening's basis (default ω=0) for
the unified `BasisStore[m, t]` slot. The remaining N-1 bases are discarded
after use.

Since HiGHS' `setBasis` C API accepts statuses only (not LU, primals, or
duals — confirmed via the 2026-04-21 API spike), the useful information
content of the discarded bases is "what was the optimal basis status pattern
for this (state, noise) pair". Epic 03 A/B #3 showed ω=1..9 mean pivots
converge to 969 with cv=3.3%, indicating the ω-to-ω basis variation is
small — but non-zero.

Can we aggregate the N opening-specific basis status patterns into a single
representative basis that warm-starts next iteration's forward solve better
than any single ω's basis?

## The blocking technical constraint

Any valid basis must contain **exactly `num_rows` BASIC variables** (plus
`num_cols - num_rows` non-BASIC, accounting for slacks). HiGHS rejects
dimension-inconsistent bases (`BasisInconsistent` at
`cobre-solver/src/highs.rs:1117-1205`, already tracked in the
`basis_rejections` counter).

A naive per-element aggregation — "mark each element BASIC if it was BASIC in
the majority of openings" — almost certainly violates the global count
constraint. A valid aggregation scheme must preserve `sum(basic) == num_rows`.

## Proposal sketches

### Scheme 1 — Frequency-ranked BASIC selection

1. For each basis element (col or row), compute `basic_count = sum(status_ω
== BASIC for ω ∈ 0..N-1)`.
2. Rank all elements by `basic_count` descending.
3. Mark the top `num_rows` as BASIC.
4. For the rest, pick LOWER/UPPER by last-seen value (or plurality vote).

Guarantees count validity. May produce a synthetic basis that was never
optimal for any single opening's LP.

### Scheme 2 — Intersection-first, frequency-fill

1. Start with `I = {elements BASIC in all N openings}`.
2. If `|I| < num_rows`, fill the remaining slots from the next-highest
   `basic_count` ranks.
3. If `|I| > num_rows` (shouldn't occur if individual bases are valid), trim by
   lowest `basic_count`.
4. Non-BASIC elements: pick LOWER/UPPER by majority-last-vote.

Biases toward elements that were stable across all noise realizations. Still
produces a synthetic basis.

### Scheme 3 — Median-opening by Hamming distance

1. For each pair `(ω, ω')`, compute `d(ω, ω') = sum(status_ω[i] != status_ω'[i]
for all elements i)`.
2. For each ω, compute `total_distance(ω) = sum_{ω'} d(ω, ω')`.
3. Pick the ω that minimizes `total_distance` — the basis "most central" to
   the N-opening cloud.
4. Use that ω's basis verbatim.

No synthetic basis — always selects a real opening's actual optimum. Validity
is guaranteed by construction (HiGHS produced it).

## Expected benefits

- **Reduced pivot count on next forward solve.** A "representative" basis
  should require fewer pivots than any single-point basis when the next solve
  is at a different noise realization. Magnitude is bounded by the
  opening-to-opening variance — on convertido that variance is small (cv=3.3%
  on mean pivots), so the expected improvement is ~1–3% pivot reduction on
  the forward read side.
- **Complementary to Epic 05's unified-store win.** Epic 05 eliminates
  cut-row reconstruction on forward reads (0 new cut rows, dimensions match).
  Aggregation further reduces per-solve pivot work on the column statuses.
  The two stack multiplicatively.
- **No wire-format change needed.** Aggregation happens at capture time
  (before the basis is stored in `BasisStore[m, t]`). Storage and broadcast
  are unchanged from Epic 05.

## Expected costs / risks

- **Scheme 1/2 — synthetic-basis rejection risk.** A basis constructed by
  per-element aggregation may satisfy the count constraint but violate
  deeper HiGHS consistency checks (e.g., basis-matrix non-singularity).
  HiGHS falls back to cold-start on rejection; the capture effort is wasted.
  An A/B would show this directly via the `basis_rejections` counter.
- **Scheme 3 — marginal gain.** The median opening is approximately any
  opening in a near-stationary ω-cloud. On convertido's cv=3.3% case, the
  information gain over "pick ω=0" is probably ≤ 1%.
- **Capture-time cost.** Accumulate `num_cols + num_rows` i8 values × N
  openings per (m, t). On convertido: 8,000 elements × 10 openings ≈ 80 KB
  scratch per capture (thread-local, reusable). Aggregation work: O(N ×
  (num_cols + num_rows)) ≈ 80k ops per capture. Negligible.
- **Memory footprint.** Scheme 1/2 need the full N × (num_cols + num_rows) i8
  cube per worker's in-flight (m, t). Scheme 3 needs only pairwise-distance
  counters (N² distance values per (m, t)). Both fit in pre-allocated
  scratch.
- **Implementation complexity.** Scheme 1/2 add ~150 lines of aggregation
  logic + validity checks. Scheme 3 adds ~50 lines. Tests, clippy, and
  D-suite byte-identity gates must all pass.
- **Debuggability.** Synthetic bases (Scheme 1/2) are hard to reason about
  manually — "why did this element become BASIC?" requires inspecting the N
  underlying opening bases. Scheme 3's traceability is strong (always points
  at a specific ω's real basis).

## Ordering

Run **after** Epic 05 completes and A/B #6 publishes its result.

- If Epic 05's A/B #6 achieves the ≥ 5% wall-time reduction target (unified
  store closes the pivot gap), aggregation adds zero marginal value — close
  as not-pursued.
- If A/B #6 shows residual pivot overhead concentrated on the **forward read
  side** (i.e., forward solves with cached basis still pivot more than
  baseline), aggregation becomes a plausible follow-up.
- Scheme 3 is the recommended first attempt — no synthetic-basis risk,
  simplest to implement, clear null-result path (it reduces to "pick ω=0" in
  near-stationary cases).

## A/B #8 design (placeholder)

- Arms: (A) Epic 05 HEAD (ω=0 capture); (B) Scheme 3 (median-opening
  capture); optional (C) Scheme 2 (intersection+fill).
- Metric: forward-read mean pivot count, `basis_rejections` counter,
  total LP wall time.
- Cases: `convertido_backward_basis` at production scale, D01 for edge
  coverage.
- Decision threshold: ≥ 2% additional LP wall-time reduction vs Epic 05
  HEAD AND rejection rate ≤ 0.1%.

## Open questions

1. **Does HiGHS' `isBasisConsistent` catch Scheme-1/2 synthetic-basis
   violations at setBasis time?** If yes, rejection rate on those schemes
   is the dominant signal. If no, the synthetic basis may be silently
   accepted and HiGHS' simplex may take extra pivots to repair. Needs
   exploratory spike.
2. **Does Scheme 3 reduce to "pick ω=0" in practice?** If the Hamming
   distance between openings is uniformly small, the median selection has
   no preference. Needs data from Epic 05's captured bases on D01 and
   convertido.
3. **Interaction with Epic 06's activity-window classifier.** Epic 06
   improves _cut-row_ classification for new cuts added since capture.
   Cross-opening aggregation improves _column-status_ classification for
   the current opening. The two mechanisms target different dimensions
   and compose cleanly.
4. **Wire-format implications for cross-MPI parity.** Aggregation happens
   at capture time on a single worker (per AD-2 in Epic 05); no
   broadcast-time aggregation. Determinism: Scheme 3 picks a specific ω
   based on a deterministic distance computation; if ties exist (possible
   on highly-symmetric cases), tie-break by `ω` index to preserve
   bit-identity across MPI configs.
