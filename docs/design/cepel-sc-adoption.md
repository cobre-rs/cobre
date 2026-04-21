# CEPEL SC (Seleção de Cortes) adoption — design proposal

**Status:** Future recommendation (logged 2026-04-21 during Epic 04 planning)
**Origin:** User idea raised during Epic 04 scoping, 2026-04-21
**Source:** `cepel-cut-selection-approach.md` (technical reading of CEPEL RT 11138/2017)

---

## Problem

Every backward LP in Cobre currently solves against the full
accumulated cut pool. After 50 iterations at 117 stages with ~100
cuts/iter accumulated, each LP carries ~5000 cut rows. Most are not
active at the current trial point — CEPEL's data on NEWAVE v23 shows
40–90% of cuts are dominated after 20 iterations.

Cobre's `cut_selection` module (`crates/cobre-sddp/src/cut_selection.rs`)
already implements a basic deactivation strategy via `DeactivationSet`,
but this is activity-count based, not relevance-per-solve. CEPEL's SC
(Seleção de Cortes) is a delayed-constraint-generation wrapper that
solves the same LP as the full-cut formulation but typically with
37–56% less wall time by avoiding cuts that wouldn't be active.

## The SC algorithm (summary)

For each backward stage-scenario-opening solve:

1. **Seed.** Start with "initial cuts" = cuts active in the NLEQ
   subproblems at this state in iterations [iter-1, iter-k₂].
2. **Solve.** Get optimum `f*` and argument `x*`.
3. **Activity check.** For each cut _not_ in the LP, compute
   `α_i = π_i^T · x*`. If `α_i > f*`, the cut is violated.
4. **Add top nadic.** Take up to `nadic` most-violated cuts,
   re-solve from the prior basis (dual-simplex warm start).
5. **Terminate** when no cuts are violated.

CEPEL's calibrated parameters: `k₂=5`, `nadic=10`, exclude lateral
cuts. A 30% end-to-end reduction in backward-pass wall time is
plausible for Cobre (vs CEPEL's 46–53% on NEWAVE, discounted per
§4.3.3 reasoning: Cobre already has HiGHS retention and forward/
backward basis caches, so SC's marginal gain is over an already-
optimized baseline).

## Proposal sketches (three alternatives)

### Option A — Literal SC implementation

Implement the full SC wrapper as a new module `cut_selection/sc.rs`.
Each backward solve enters an inner loop. Maintain per-(stage,
state-bucket) cut-activity history to produce the initial seed set.
Parameters exposed as config.

- **Estimated effort:** 2–3 weeks.
- **Expected benefit:** 25–40% backward-pass wall-time reduction.
- **Integrates with:** solver observability instrumentation; new
  metrics needed for inner-iteration counts.
- **Breaks invariant:** declaration-order invariance. CEPEL's §3.5.1
  shows SC produces the same optimal value as TC but different
  per-entity operational trajectories (multiple-optima divergence).
  The CLAUDE.md rule "results must be bit-for-bit identical
  regardless of input entity ordering" fails.

### Option B — Extract ideas for current cut selection

Keep Cobre's current cut-pool management, but add the SC activity
check as a complementary mechanism for cut-deactivation guidance.

- Track per-cut activity counter incremented when the cut is binding
  (dual > eps) in any backward solve.
- On every cut deactivation round, deactivate cuts with lowest
  activity count rather than (or in addition to) age-based rules.
- No inner loop; the LP still includes all active cuts on entry.

- **Estimated effort:** 3–5 days.
- **Expected benefit:** 5–10% improvement over current deactivation
  strategy; directly measurable via existing cut-selection metrics.
- **Preserves invariant:** no change to LP optimum selection;
  operational trajectories remain deterministic.
- **Low risk:** wraps an existing mechanism.

### Option C — Use SC ideas for basis initialization only

Adopt SC's activity-tracking data as the input to smarter cut-row
basis initialization (see `docs/design/smarter-cut-initialization.md`),
without changing the LP structure or solve count.

- Track per-cut activity like Option B.
- On `reconstruct_basis`, initialize cuts with `recent_activity > 0`
  as `NONBASIC_LOWER` (tight), others as `BASIC` (slack).
- No inner loop; LP solve structure unchanged.

- **Estimated effort:** 1–2 weeks (mostly wire-format extension).
- **Expected benefit:** reduce Epic 03's +65% ω=0 pivot increase
  toward baseline; potentially 5–15% additional wall-time drop on
  top of v0.5.0.
- **Preserves invariant:** no optimum selection change.
- **Risk:** wire format change; A/B required.

## Expected benefits (by option)

| Option                           | Effort    | Back-pass speedup (est.) | Invariance safe?     |
| -------------------------------- | --------- | ------------------------ | -------------------- |
| A — full SC                      | 2–3 weeks | 25–40%                   | NO (multiple-optima) |
| B — activity-guided deactivation | 3–5 days  | 5–10%                    | YES                  |
| C — activity-guided basis init   | 1–2 weeks | 5–15%                    | YES                  |

## Expected costs / risks

- **Option A breaks an architectural invariant.** Cobre's declaration-order
  invariance rule (CLAUDE.md: "results must be bit-for-bit identical
  regardless of input entity ordering") currently rests on deterministic
  LP solve selection. SC introduces CEPEL-documented multiple-optima
  divergence at the per-entity trajectory level. Resolution requires
  either narrowing the invariance guarantee (honest but requires docs),
  implementing deterministic LP tie-breaking (expensive, solver-level
  change), or accepting the drift (CEPEL's implicit choice).
- **All options require cut-activity tracking.** New memory: per-cut
  counter (Option A/B) or per-cut-row activity history (Option C).
  Scales with cut pool size.
- **Option A is complex.** Inner-loop state machine, convergence proofs,
  interaction with the forward/backward basis caches all need design
  attention. CEPEL's §6.5 explicitly flags this integration as an
  open design point.
- **Parameters are tuneable.** `k₂=5`, `nadic=10` are CEPEL's
  calibration on NEWAVE. Cobre should validate on convertido before
  committing; parameters may differ for Cobre's LP shape.
- **Expected gains shrink when composed.** If Epic 04 (all-ω cache)
  - basis-init improvements (`docs/design/smarter-cut-initialization.md`)
    ship first, SC's marginal gain shrinks further. CEPEL's own §4.3.3
    argues this is why their 60% toy-problem speedup became 46–56% on
    NEWAVE — Cobre has even more baseline optimizations already.

## Ordering

CEPEL's §6.5 recommends: ship the basis cache first, measure, then
decide on SC. Epic 01/02 shipped in v0.5.0. Natural next:

1. **Epic 04** (all-ω cache extension) — measure residual backward-pass
   cost after A/B #5.
2. **Option B** — activity-guided cut deactivation, as a targeted low-risk
   improvement. Measure via existing cut-selection metrics.
3. **Option C** — activity-guided basis initialization, composable with
   Epic 04's 2D cache and Option B's activity tracking. A/B measure.
4. **Option A** — full SC, only if Options B + C leave enough backward-pass
   cost to justify the invariance-rule resolution. Likely not for v0.5.x;
   potentially v0.6.0 discussion.

## A/B #8 design (placeholder, Option A specifically)

- Arms: (A) v0.5.x with current cut pool; (B) SC with `k₂=5, nadic=10`,
  excluding lateral cuts.
- Cases: convertido at production scale; D-suite for invariance check
  (expected to fail on B if SC is doing its job — the invariance
  failure is the measurement).
- Metrics: backward LP solve total time, inner-iteration counts per
  solve, per-entity trajectory drift magnitude, aggregate SIN
  statistics (should match).
- Decision threshold: ≥ 20% backward LP time reduction AND aggregate
  statistics match to < 1% AND user accepts the declaration-order
  invariance narrowing.

## Open questions

1. **CVaR integration.** CEPEL's activity formula `α_i = π_i^T · x*`
   may need modification for CVaR cuts (additional dual variables).
   Review Cobre's cut structure before assuming the formula ports.
2. **Forward-pass SC.** CEPEL's experiments cover backward only.
   Forward-pass SC requires a different initial-cut-set definition
   since out-of-sample scenarios don't have a discrete state index.
3. **Memory footprint at Cobre scale.** CEPEL's storage estimate
   (§4 of the reading) is "200 × 120 × 45 × 8000 bits ≈ 10 GB" for
   NEWAVE. Cobre's comparable number on convertido: 500 scenarios
   × 117 stages × 50 iterations × ~5000 cuts × bits ≈ 18 GB. Needs
   streaming or windowed representation.
4. **Interaction with forward/backward basis caches.** Under SC,
   the cut set at (stage, ω=0) in iteration k may differ from
   iter k+1's set. A cached basis from iter k references cuts
   that don't exist in iter k+1's LP. `reconstruct_basis`'s
   slot-identity mechanism must be extended or the cache must be
   reformulated.
5. **Declaration-order invariance resolution.** If Option A proceeds,
   the rule needs a decision: narrow to aggregates only,
   lexicographic simplex tie-breaking, or accept the drift. This is
   a cross-cutting architectural decision, not an SC-local one.
