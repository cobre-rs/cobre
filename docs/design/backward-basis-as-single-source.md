# Backward basis as single source of truth — design proposal

**Status:** Future recommendation (logged 2026-04-21 during Epic 04 planning)
**Origin:** User idea raised during Epic 04 scoping, 2026-04-21
**Related:** `docs/assessments/backward-basis-cache-ab3-convertido.md`, `docs/assessments/backward-basis-cache-decision.md`

---

## Problem

After Epic 02 shipped (v0.5.0), Cobre maintains two basis storage layers:

- `BasisStore` — per-(scenario, stage) forward basis captured during
  the forward pass. Allocated once per run; grows with scenario count.
- `BackwardBasisStore` — per-stage ω=0 basis populated by rank 0's
  m=0 worker during the backward pass, broadcast end-of-iteration.

At the backward read site `resolve_backward_basis`
(`crates/cobre-sddp/src/backward.rs:642-661`), both are consulted:
backward-cache first, forward-store as fallback. On iter=1 the backward
cache is empty so every ω=0 solve falls back to forward. The A/B #3
measurement on `convertido_a` shows 100% cache-hit rate on iter≥2 ω=0
solves — the forward-store fallback is effectively a cold-start-only
path on convertido.

Maintaining two caches duplicates memory, storage-layer code, and
invariants. If the backward cache is sufficient by itself, we can
simplify the architecture.

## Proposal sketch

Remove `BasisStore` as a mandatory path. Concretely:

1. **Iter=1 cold start** — solve without a warm-start basis. HiGHS
   will cold-solve the first ω=0 LP at each stage; this is a
   one-off cost per run (117 LPs on convertido) that amortizes over
   4+ subsequent iterations' reads from `BackwardBasisStore`.
2. **Simulation warm-start** — simulation currently uses `BasisStore`
   to warm-start policy-sim LPs. Replace with a backward-cache read
   at the same (stage, ω=0) slots (simulation is ω=0 by construction
   when inflows are realized). This works as long as the training
   phase has populated `BackwardBasisStore` before simulation begins,
   which it does by construction (training → checkpoint → simulation).
3. **`resolve_backward_basis` simplification** — the fallback chain
   collapses to a direct backward-store lookup plus a `None` branch
   on iter=1. Signature stays, body shrinks.

## Expected benefits

- **Memory savings.** `BasisStore` holds per-scenario slots — on
  convertido with ~500 scenarios × 117 stages that's ~58k slots.
  At ~50 KB/slot that's ~3 GB per rank saved.
- **Simpler architecture.** One cache to reason about. One set of
  invariants. One broadcast wire format (already owned by
  `CapturedBasis::to_broadcast_payload`).
- **Code reduction.** `BasisStore`-specific capture in forward.rs
  (~100 lines), simulation-specific basis read paths, and the
  forward-side capture in `training.rs` can be removed.
- **Memory budget for Epic 04.** Freeing `BasisStore` offsets the
  cost of extending `BackwardBasisStore` to 2D (stage × omega) in
  Epic 04. Net: likely still negative on memory balance vs. Epic 02
  alone but much better than Epic 04 + BasisStore.

## Expected costs / risks

- **Iter=1 wall-time regression.** Epic 03 A/B #3 did not measure
  the iter=1 cold-start cost in isolation. At production scale
  (117 stages × 10 openings × 50 fwd passes × iter 1 only ≈ 58k
  solves on iter=1 alone) a ~10% cold-start penalty on those
  solves could cost ~1-2% of total wall time. Might be acceptable
  or might negate the Epic 02 −13.5% win at small iteration counts.
- **Simulation coupling.** Simulation becomes dependent on the
  training's `BackwardBasisStore` checkpoint. A simulation run
  that reuses a training-only checkpoint must load the backward
  cache from the checkpoint (currently it loads the forward
  `BasisStore` from the policy file). Requires a new section in
  the policy checkpoint format — breaking change.
- **Policy file format change.** The FlatBuffers-based policy
  checkpoint (`crates/cobre-sddp/src/policy_load.rs`) currently
  serializes `BasisStore` for simulation warm-start. Adding
  `BackwardBasisStore` and removing `BasisStore` is a
  schema-breaking change — requires a version bump and migration
  path for existing policy files.
- **MPI broadcast cost at simulation start.** The backward cache
  must be broadcast from rank 0 to all ranks at the start of
  simulation (not just at end-of-training). Adds one broadcast
  call to the simulation entry point. ~60 MB on convertido.
- **Null-result possibility.** If iter=1 cold-start cost dominates
  the Epic 02 savings on small studies (e.g. D-suite with 3–5
  iterations), this change is a wall-time regression for those
  users. Must A/B before shipping.

## Ordering

Should run after Epic 04 (all-ω extension) — the 2D cache gives
a clearer picture of what to measure and whether the memory savings
are load-bearing. Could also become the Epic 05 proposal if A/B
#5 shows the all-ω cache is a win.

## A/B #6 design (placeholder)

- Arms: (A) current v0.5.0 with forward + backward caches;
  (B) backward-only with iter=1 cold-start.
- Metric: iter=1 cold-start wall time, iter≥2 wall time (should
  be unchanged from A), total training wall time, memory
  high-water mark (should drop significantly in B).
- Cases: D-suite (to measure small-iter regression risk),
  convertido (production scale).
- Decision threshold: memory savings ≥ 1 GB/rank AND total wall
  time regression ≤ 2 % on D-suite AND 0 regression on convertido.

## Open questions

1. Does simulation truly never need non-ω=0 backward basis data?
   Check the simulation LP shape at convertido non-scenario-0
   paths — some stages may have ω≥1-style openings for
   stochastic simulation.
2. How large is the forward-store memory footprint in practice?
   The 3 GB/rank estimate assumes uniform slot sizes; actual
   profile via `cargo-bloat` or heaptrack would verify.
3. Policy checkpoint migration: do any downstream consumers
   (cobre-python, external scripts) read the forward basis
   directly from the policy file? If yes, the breaking change
   needs coordinated updates.
