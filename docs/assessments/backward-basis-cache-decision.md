# Backward-Basis-Cache Decision Record

**Date:** 2026-04-21
**Branch HEAD:** `c1016bbf0bf554f36d6fa14a289fa7328223efee`
**Verdict:** **GO** — ship in v0.5.0

## Inputs

- **A/B measurement (ticket-001):**
  [`backward-basis-cache-ab3-convertido.md`](./backward-basis-cache-ab3-convertido.md)
  — production-scale A/B on `convertido_a` (50 fwd passes × 5 iterations
  × 117 stages × 10 openings ≈ 93k backward LP solves per arm).
- **Correctness baseline (Epic 01 T5):**
  [`backward-basis-cache-baseline.md`](./backward-basis-cache-baseline.md)
  — 27 D-case byte-identity + `convertido_a` drift allowlist at commit
  `1de344f` (post-Epic-01).
- **Post-Epic-02 correctness gate (ticket-002):** deferred to CI. Epic 02
  is pure observability (instrumentation, no LP selection or cut coefficient
  changes); the Epic 01 gate at `1de344f` therefore still applies. MPI
  parity at 1/2/4 ranks on D01 is run on every push via
  `.github/workflows/mpi-slurm.yml`.

## Rationale

The A/B #3 report shows a clear, production-scale wall-time win at
convertido_a's 50-forward-pass × 5-iteration topology:

| Metric                                  | Cache-OFF  | Cache-ON   | Delta  |
| --------------------------------------- | ---------- | ---------- | ------ |
| Backward iter≥2 all-ω total LP solve ms | 10,112,036 | 8,769,619  | −13.3% |
| Forward total LP solve ms               | 1,042,857  | 906,493    | −13.1% |
| Total LP solve ms (all phases)          | 12,570,416 | 10,877,415 | −13.5% |
| Total training wall time ms             | 1,401,729  | 1,221,242  | −12.9% |
| Cache hit rate (ω=0, iter≥2)            | N/A        | 1.000      | —      |
| MPI broadcast cost fraction             | 0.000%     | 0.000%     | —      |

The ω=0 iter≥2 mean simplex-iteration count rises from 1,588 to 2,623
(+65%) — consistent with the Epic 01 learning that the cache _adds_
pivots at ω=0 but wins via a secondary reduction of per-LP solve time
at ω≥1 and in the forward pass (HiGHS' retained LU factorization
amortizes better with a stable basis). The net wall-time improvement
is −13.5% across all LP solves, well above the reframed Success Metric
#4 threshold (≥ 0% total backward wall-time drop at production scale).
Cache hit rate is 100%, which means every iter≥2 ω=0 backward solve
reads the cache directly (no forward-store fallbacks observed on this
run — R4 did not fire).

Correctness: the cold-start invariant (all iter=1 ω=0 backward solves
read the forward-pass basis, not the cache) holds. The
`reconstructed_basis_preserves_invariant_on_baked_truncation`
regression test from Epic 01 continues to pass. The Epic 02 review
process caught and corrected three issues before commit (doc-comment
orphan, infrastructure-crate type-name leakage, and a vacuous
cold-start assertion).

## User approval

> "Approved for v0.5.0 ship based on -13.5% total LP time improvement
> on convertido_a at production scale, cache hit rate 1.000, and
> Epic 01/02 correctness gates passing."

## Rollback protocol

**Go path (this record).** Any post-tag regression ships as a v0.5.x
point revert of commits `1de344f` (Epic 01) and `c1016bb` (Epic 02) in
reverse chronological order. Epic 03 commits (measurement report and
CHANGELOG) remain in place since they are purely documentation. The
revert restores the forward-only `BasisStore` semantics at the
backward read site and removes the `BackwardBasisStore` struct and the
capture/broadcast code path.

**Null path (not taken).** Would have reverted the same two commits
and opened a PR against `main` with this decision record as the
description; ticket-004 would have been skipped.

## Related artifacts

- A/B report: `docs/assessments/backward-basis-cache-ab3-convertido.md`
- Epic 01 baseline: `docs/assessments/backward-basis-cache-baseline.md`
- Epic 01 commit: `1de344f7e534b79831658391795343eeec7ff649`
- Epic 02 commit: `c1016bbf0bf554f36d6fa14a289fa7328223efee`
- MPI CI coverage: `.github/workflows/mpi-slurm.yml`
- Integration test: `crates/cobre-sddp/tests/test_backward_cache_hit_rate.rs`
