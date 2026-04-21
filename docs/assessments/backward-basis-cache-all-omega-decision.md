# Backward-Basis-Cache All-ω Extension Decision Record

**Date:** 2026-04-21
**Branch HEAD:** `7d67ea093a1b71555a2bcd4a914d27bd619aae68`
**Verdict:** **NOT-PURSUED** — all-ω extension closes as null-result

## Scope

Epic 04 of `plans/backward-basis-cache` proposed extending the ω=0-only backward
basis cache (shipped in v0.5.0 at commit `7d67ea0`) to all ω slots. Tickets T1
(storage geometry: 1D → 2D `(stage, omega)`) and T2 (capture + read semantics at
every ω) were implemented on a working tree. A pre-A/B smoke run on
`~/git/cobre-bridge/example/convertido_backward_basis_all_omega` (release mode,
10 threads) surfaced a clear wall-time regression before iteration 5 completed;
the user interrupted at iter 3/5. The full A/B #5 was not committed. Epic 04
closes without shipping T1 or T2.

## Inputs

- **Baseline run** (ω=0-only cache, v0.5.0 HEAD):
  `target/release/cobre run ~/git/cobre-bridge/example/convertido_backward_basis/ --threads 10`
  - 5 iterations, 20m 21s total
  - LP time: 10,877.4s total, **17.7ms avg**
  - 614,600 LP solves
  - Simplex iterations: 143,871,368
  - Basis reuse: 100.0% hit (0 rejected / 52,850 offered)
- **All-ω smoke run** (T1+T2 working tree, interrupted):
  `target/release/cobre run ~/git/cobre-bridge/example/convertido_backward_basis_all_omega/ --threads 10`
  - 3/5 iterations, 23m 19s elapsed (extrapolated ~39m total, ~1.9× baseline)
  - LP time: **36.2ms avg** (+105% vs baseline)
  - Run terminated by user before completion
- **Epic 04 overview a-priori reasoning** (`plans/backward-basis-cache/epic-04-all-omega-followup/00-epic-overview.md:71-74`):
  > "An equally plausible null-result is that ω≥1 is already close to the
  > LU-amortization ceiling — the -13.3% was almost entirely from the ω=0
  > first-opening effect, and extending to ω≥1 only adds broadcast cost and
  > capture overhead."

## Rationale

The ω=0-only cache (Epics 01-03) wins because HiGHS pays the LU factorization
cost once per `(stage, iteration)` solve chain at ω=0, then amortizes that
factorization across ω=1..9 within the chain. Epic 03's A/B #3 entry signal
confirmed ω=1..9 sits at a near-fixed-point ~969 pivots (cv 3.3%) with the
ω=0-only cache active — ω≥1 is already warm from HiGHS' retained LU.

Forcing a stored basis at every ω breaks that amortization: HiGHS must load
and re-factorize on every backward LP solve instead of reusing its existing
LU. The +105% average LP time measured on the all-ω smoke run is consistent
with "re-factorize every solve" as the dominant cost. The user summarized the
architectural conclusion during the decision chat:

> "By setting the basis for every opening, our times are much worse...
> I think setting the basis for every omega is not going to result in a good
> performance anyway, reusing the HiGHS state among openings seems to be a
> better idea."

"Reusing HiGHS state among openings" is precisely what the ω=0-only cache
already does: set the basis once per solve chain at ω=0 and let HiGHS'
internal LU amortization carry ω=1..9 for free. The Epic 04 extension would
have replaced that free amortization with a paid per-ω basis load.

## Implementation sanity check

Before reverting, a targeted inspection of the T1+T2 working tree confirmed
the implementation matched the ticket specifications:

- Capture at every ω (`backward.rs:884-886`): no `omega == 0` guard, fires
  for every opening on rank-0 m=0.
- Read at every ω (`backward.rs:738-745`): `resolve_backward_basis` called
  unconditionally with the current `omega`.
- 2D indexing (`backward.rs:639+`): `s * num_openings + omega` consistent at
  capture, read, and merge sites.
- Merge loop (`backward.rs:1276-1294`): drains all `(worker, omega)` slots
  into the 2D store.
- Unit tests `resolve_backward_basis_hits_at_omega_positive` and
  `resolve_backward_basis_fallback_to_forward_at_positive_omega` pass.
- Integration test `test_backward_cache_hit_rate_multi_omega` on D29 (4
  stages × 5 openings) clears the `cache_hit_rate >= 0.95` assertion on
  iter≥2 ω≥1 rows.

The slowdown is architectural, not implementation-level: the code works as
designed, but the design fights HiGHS' LU amortization.

## User approval

> "Null-result: revert T1+T2, close Epic 04 as not-pursued"

## Rollback action taken

Working-tree changes for T1 and T2 were reverted via `git restore` on:

- `crates/cobre-sddp/src/workspace.rs`
- `crates/cobre-sddp/src/backward.rs`
- `crates/cobre-sddp/src/training.rs`
- `crates/cobre-sddp/src/cli/commands/run.rs` (cargo-fmt carry-over)
- `crates/cobre-io/src/output/dictionary.rs` (cargo-fmt carry-over)
- `crates/cobre-sddp/tests/test_backward_cache_hit_rate.rs`

The branch HEAD stays at `7d67ea0` (Epic 03 ship commit). No `git revert` was
needed because T1 and T2 were never committed. Subsequent `cargo build
--workspace --all-features` and `cargo test --package cobre-sddp
--all-features --lib` complete with 0 errors / 1,228 lib tests passing.

## Decision

Epic 04 closes as **not-pursued**. The 2D `BackwardBasisStore` extension is
not retained in the main line — the ω=0-only cache shipped in v0.5.0 is the
final design.

## Related artifacts

- A/B #3 report (ω=0-only cache): `docs/assessments/backward-basis-cache-ab3-convertido.md`
- Epic 01 baseline: `docs/assessments/backward-basis-cache-baseline.md`
- Go-decision (ω=0-only ship): `docs/assessments/backward-basis-cache-decision.md`
- Epic 03 ship commit: `7d67ea093a1b71555a2bcd4a914d27bd619aae68`
- Epic 04 overview: `plans/backward-basis-cache/epic-04-all-omega-followup/00-epic-overview.md`
