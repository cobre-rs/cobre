# Smarter cut-row basis initialization — design proposal

**Status:** Future recommendation (logged 2026-04-21 during Epic 04 planning)
**Origin:** User idea raised during Epic 04 scoping, 2026-04-21
**Related:** `crates/cobre-sddp/src/basis_reconstruct.rs`

---

## Problem

When `reconstruct_basis` applies a stored basis to a stage LP with
more cut rows than existed at capture time, the newly-added cut
rows have no status in the stored basis. The current heuristic
(`basis_reconstruct.rs`) classifies them as:

- `NONBASIC_LOWER` (tight) if the cut's slack at the captured
  `state_at_capture` is ≤ `1e-7`
- `BASIC` (slack) otherwise

This is a single-shot classification based on slack-at-capture.
It ignores:

- Whether the cut was binding in _recent_ iterations
- How often the cut has been active historically
- The dual value (if the cut had been part of a previous solve,
  its dual indicates activity)
- Cuts produced at adjacent stages that haven't been solved yet

HiGHS then has to repair primal/dual feasibility from this guess.
Better guesses mean fewer pivots. The +65% ω=0 pivot increase with
the cache-ON (Epic 03 A/B #3) partly reflects imperfect cut-row
initialization — the cache wins via LU retention despite this
overhead.

## Proposal sketch

Augment `CapturedBasis` with a per-cut activity history snippet,
and use it to classify new cuts more intelligently.

### Option A — Activity counter on `CutMetadata`

Each `CutMetadata` entry gains a `recent_activity: u32` field
incremented each time the cut's dual value is positive (binding)
in a backward-pass solve. Decayed (shifted right by 1) every N
iterations.

On basis reconstruction, new cuts with `recent_activity > threshold`
are initialized as `NONBASIC_LOWER` (tight); others as `BASIC`.

- **Cheap:** one u32 per cut, one counter update per solve.
- **Implementable:** piggy-back on the existing cut-selection
  strategy's `DeactivationSet`.

### Option B — Activity bit-vector per (stage, state-bucket)

For each stage, maintain a bit-vector of length "num cuts" recording
"cut i was active in the last k₂ iterations at a nearby state."
Lookup by quantized state bucket on `state_at_capture`.

- **Richer signal:** matches CEPEL's SC strategy's initial-cut-set
  definition (§3.1 of `cepel-cut-selection-approach.md`).
- **Memory:** per-stage × num-cut bits × k₂ buckets. For convertido
  (117 stages × 8000 cuts × 5 buckets) ≈ 4.7 MB total.
- **Implementation complexity:** state bucketing requires a
  scheme (quantile binning? L1-nearest bucket?). Non-trivial.

### Option C — Use recent dual at recorded solve

During the backward pass, `stage_solve.rs` can capture the full
cut-row duals alongside the basis status. On reconstruction, seed
tight cuts as those whose dual at last solve was > small-epsilon.

- **Directly relevant:** dual > 0 means the cut was binding at
  that state, which is a strong predictor.
- **Implementation:** `CapturedBasis.cut_row_duals: Vec<f64>` of
  length matching `cut_row_slots`. Memory: one f64 per cut row
  per captured basis. Negligible.
- **Cleaner than Option A:** doesn't need a new counter; uses
  existing solver output.

## Expected benefits

- **Pivot reduction at cache hits.** Currently ω=0 iter≥2 solves
  use +65% more pivots with cache-ON (Epic 03 A/B #3). Better cut
  initialization could bring this closer to parity (a few percent
  above baseline instead of 65%). Per-solve wall time should drop
  proportionally.
- **Composable with Epic 04.** The all-ω cache extension (Epic 04)
  would benefit from the same improvement at every ω.
- **Diagnostic value.** Option C's dual-tracking exposes cut
  activity directly, informing future cut-selection strategy
  work (aligned with CEPEL's SC analysis).

## Expected costs / risks

- **Capture-time overhead.** Option C writes ~117 × 10 × 1000 =
  ~1M f64 values per rank per backward capture (worst case).
  One copy per broadcast; could add 1-2% to capture cost.
- **Wire format change.** `CapturedBasis::to_broadcast_payload`
  and `try_from_broadcast_payload` need to serialize the new
  field. Epic 06 invariant says those are the sole owners of the
  byte layout — this is a schema change that requires a version
  bump and both functions updated together.
- **Regression risk.** If the new classification is wrong more
  often than the current slack-based heuristic, pivot counts
  increase rather than decrease. Gate behind an A/B measurement.
- **Test coverage.** `basis_reconstruct.rs` has thorough unit
  tests for the slack-classification path; those need extended
  variants for the activity-based path.

## Ordering

- **Prerequisite:** Epic 04 ships (all-ω cache). Gives a larger
  target for measurement.
- **Could overlap with CEPEL SC proposal** (`docs/design/cepel-sc-adoption.md`)
  if SC is pursued — both rely on cut-activity tracking. If SC is
  pursued first, activity data is available for free.

## A/B #7 design (placeholder)

- Arms: (A) current slack-based classification;
  (B) activity-based classification (Option C recommended).
- Metric: ω=0 iter≥2 mean pivot count (target: reduce from
  2623 toward 1588 cache-OFF baseline);
  total backward LP solve time (target: further reduction
  beyond Epic 02's −13.3%).
- Cases: convertido (production scale), D01 (edge case).
- Decision threshold: ≥5% additional backward LP time reduction
  over v0.5.0 baseline AND no regression on D-suite byte
  identity.

## Open questions

1. Does HiGHS `setBasis()` accept per-row duals, or only statuses?
   If statuses only, Option C reduces to "use dual > eps to set
   status to NONBASIC_LOWER" — which is Option A in different
   framing.
2. How much of the +65% pivot increase is cut-row classification
   vs. column-status classification? If columns dominate, this
   proposal's upside is smaller.
3. Memory/wire-format cost of per-cut-row dual in the broadcast
   payload — is there a compact encoding (e.g., quantized to
   8 bits) that preserves enough signal?
