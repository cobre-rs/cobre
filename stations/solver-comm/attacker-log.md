# cobre-solver + cobre-comm attacker dispatch log

Baseline: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (register pin; HEAD's evaluated surfaces are byte-identical to it under the
harness drift rule, so `grep` over the worktree and `git show 077dbe2c:<path>` read the same bytes).
Dispatched: 2026-09-18. Agents: `adversarial-attacker` (read-only; Write/Edit denied), model Opus, one
worker per lens sweeping BOTH crates. The dispatching session is the sole writer of every artifact.

## Scope check (before dispatch)

The sweep target is the 30-file set `inventory.json` freezes (23 `crates/cobre-solver/src`, 7
`crates/cobre-comm/src`; set equality against the tree at the baseline is asserted by
`InventoryTests.test_file_set_equals_the_tree_at_the_baseline`). The test-bloat lens additionally
reaches `crates/cobre-solver/tests` (8 binaries) and `crates/cobre-comm/tests` (2). No sub-station
partition exists at this station: one worker per lens covers both crates, so `subStation` is the
station token `solver-comm` in every envelope.

## Prompt and validator (before dispatch)

One shared worker contract, `attacker-prompt.md`, rendered per lens into `prompts/sc-<lens>.md`
(`{{BASELINE_SHA}}` from the register header, `{{LENS}}`). It carries the eight rules — read-only,
one bare JSON object, the anchor-scope rule with the single `dupOf` exception into
`crates/cobre-sddp/src/cut/cut_sync.rs`, the L0 purity test, reserved seams FIRST as ingest filters,
perf layout-never-a-number, prose fix-shapes, no re-raise of the prior register — the closed
Alignment vocabulary, fifteen named probes across the four lenses (P1–P15), the two ingest filters,
the known-and-intended HiGHS-loud / CLP-silent asymmetry, and the frozen E4 envelope with its
`measurementLayout` / `exercisingCallSites` fields. Every line anchor in the probes was re-resolved
at the baseline before rendering; the ticket text's `profiled.rs:137` gate resolves to `:138`, the
mirror's cut-sync heading to `:334`, and the test surface to 3973 integration lines
(`conformance.rs` 1768). `tools/validate-envelope.py --role attacker --station solver-comm` was the
shape gate; it requires `subStation`, a `mechanism` on every perf candidate and rejects any timing
number, diff-shaped fix-shape or off-set `alignmentHint`.

Hand-over mechanics (the truncation lesson from the earlier stations): each worker wrote its
envelope to a scratch file OUTSIDE the repository (`/tmp/sc-attackers/sc-<lens>.json`), validated it
itself, and replied `WRITTEN <bytes> <path>`; the session validated each file again, copied it
verbatim to `raw/sc-<lens>.json`, and merged it.

## Workers (4 lenses, one worker each)

| worker | lens | agent | envelope | candidates | positives | needsHuman | merge | raw size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sc-architecture | architecture | Opus adversarial-attacker, read-only | valid (exit 0, first dispatch) | 9 | 7 | 4 | 9 kept / 0 dropped | 38171 B |
| sc-performance | performance | Opus adversarial-attacker, read-only | valid (exit 0, first dispatch) | 4 | 6 | 2 | 4 kept / 0 dropped | 21896 B |
| sc-over-engineering | over-engineering | Opus adversarial-attacker, read-only | valid (exit 0, first dispatch) | 5 | 6 | 3 | 5 kept / 0 dropped | 20298 B |
| sc-test-bloat | test-bloat | Opus adversarial-attacker, read-only | valid (exit 0, first dispatch) | 6 | 6 | 2 | 6 kept / 0 dropped | 27465 B |

All four envelopes were shape-valid on the first dispatch; no worker was re-dispatched.

## Merge (`candidates.<lens>.json`)

Merge contract: `candidateRef` = `SC-<ARCH|PERF|OE|TB>-NNN` assigned after sorting the kept
candidates by first anchor path then title (so the file is diffable); `lens` stamped on every
candidate; `lenses` = `{"sc-<lens>": "merged"}`; `positives` and `_needsHuman` carried verbatim;
`dropped[]` records every guard rejection with its code. Three guards ran in order over every raw
envelope:

1. **anchor-scope** — a candidate whose anchors all sit outside `crates/cobre-solver/` and
   `crates/cobre-comm/` is dropped `out-of-scope-anchor` unless it carries `dupOf` and anchors only in
   `crates/cobre-sddp/src/cut/cut_sync.rs`; a mixed candidate keeps its in-scope anchors and records
   the trimmed ones (`anchorsTrimmed`).
2. **layout** (performance) — a candidate with `measurementLayout` outside `{4t, 2x2}` is dropped
   `layout-missing`; any timing, speedup ratio or percentage in title, mechanism, fix-shape or
   evidence drops it `timing-number` (a `0usize` literal is not a number in this sense).
3. **fixture** (test-bloat) — a fix-shape proposing a NEW fixture crate is rewritten to the
   `test_support` hoist and the original kept on the candidate (`fixShapeRewritten`).

Result: 24 candidates kept, 0 dropped, 0 trimmed, 0 rewritten — every worker respected the scope
rule, every perf candidate carried its layout (three `4t` solver-side, one `2x2` collective) with
exercising call sites and no number, and the test-bloat worker proposed the `test_support` hoist
itself. The guards are bite-proven on tampered inputs by `CandidateEnvelopeTests` rather than by
this run. Refs: architecture SC-ARCH-001..SC-ARCH-009; performance SC-PERF-001..SC-PERF-004; over-engineering SC-OE-001..SC-OE-005; test-bloat SC-TB-001..SC-TB-006.

## Prior-register screen

Every candidate's title and anchor set was matched against the `reraiseKey` tokens of the four
`prior-register.md` entries. Three collisions, all adjudicated as intended:

- `SC-ARCH-001` (lint tables) collides with the SEEDED candidate — by design: the seeded entry hands
  the observation to the attackers and owes them the fix-shape and severity; the candidate adds the
  drift evidence and two fix-shapes. Not a re-raise.
- `SC-ARCH-002` and `SC-OE-002` both carry `dupOf` = "Superseded cut-sync public methods" with the
  `cut_sync.rs` anchors — the sanctioned E5 handoff, emitted by two lenses. Ingest merges them into
  ONE dup-of record (no id); neither is a re-raise.
- No candidate names the shared-memory hierarchy (every worker routed it to `positives` citing the
  mirror entry at line 79) and none proposes equalizing the basis-validation asymmetry (three
  workers cite it as intended behaviour in `positives`).

## Positives carried into ingest as pre-Cleared subjects

- `architecture` — crates/cobre-comm/src/traits.rs:372 SharedMemoryProvider hierarchy (with SharedRegion:320, LocalCommunicator:254, LocalCommKind:275, local.rs:155 HeapRegion, ferrompi.rs:264 split_local) — sanctioned by: docs/design/reserved-seams-and-deferred-debt.md:79 — Shared-memory communicator trait hierarchy
- `architecture` — crates/cobre-solver/src/lib.rs:52 `pub(crate) mod ffi` with the single documented escape hatch at lib.rs:164
- `architecture` — crates/cobre-solver/src/lib.rs:44 and :50 — the both-backends and no-backend `compile_error!` pair
- `architecture` — crates/cobre-solver/src/basis_status.rs:101 to_discriminant_code / :116 from_discriminant_code
- `architecture` — crates/cobre-comm/src/lib.rs:83 per_rank_counts / :94 prefix_displs
- `architecture` — crates/cobre-comm/src production doc comments
- `architecture` — crates/cobre-solver/src/backends/highs/interface.rs:487 loud BasisInconsistent vs crates/cobre-solver/src/backends/clp/solver.rs:214-216 silent accept-and-repair
- `performance` — crates/cobre-comm/src/traits.rs:372 SharedMemoryProvider hierarchy, with SharedRegion at traits.rs:320, LocalCommunicator at :254, LocalCommKind at :275, HeapRegion at local.rs:155 and FerrompiBackend::split_local at ferrompi.rs:264 — sanctioned by: docs/design/reserved-seams-and-deferred-debt.md:79 — Shared-memory communicator trait hierarchy
- `performance` — crates/cobre-solver/src/backends/highs/interface.rs set_row_bounds:303, set_col_bounds:348, add_rows:253 and get_basis:500
- `performance` — crates/cobre-solver/src/freeze.rs FreezeScratch:21 and freeze_rows_into_template:57
- `performance` — crates/cobre-solver/src/backends/profiled.rs set_profile:52 and solve:100
- `performance` — crates/cobre-comm/src/local.rs allgatherv:49, allreduce:97, broadcast:127 and barrier:134
- `performance` — crates/cobre-comm/src/factory.rs
- `over-engineering` — crates/cobre-comm/src/traits.rs:372 SharedMemoryProvider hierarchy (with SharedRegion<T> :320, LocalCommunicator :254, LocalCommKind :275, HeapRegion<T> local.rs:155, FerrompiBackend::split_local ferrompi.rs:264) — sanctioned by: docs/design/reserved-seams-and-deferred-debt.md:79 - Shared-memory communicator trait hierarchy (BACKLOG OD-001 KEEP-RESERVED)
- `over-engineering` — crates/cobre-solver/src/backends/profiled.rs:30 ProfiledSolver and its delta gate at :52-58 — sanctioned by: P9, judged by workspace-wide consumer count at the baseline
- `over-engineering` — crates/cobre-solver/src/lib.rs:164 pub mod test_support and its verbatim ffi pass-throughs — sanctioned by: P12, with lib.rs:52 pub(crate) mod ffi as the reason the forwards are not redundant
- `over-engineering` — crates/cobre-comm/src/ferrompi.rs:75-76 unsafe impl Send / Sync for FerrompiBackend — sanctioned by: P12, with the RAII/ownership argument recorded inline at ferrompi.rs:64-74
- `over-engineering` — crates/cobre-solver/src/backends/clp/retry.rs:95 const RUNGS table (and the no-time-branching contract documented at :4-5) — sanctioned by: Baseline reading of clp/retry.rs:4-5 and :95, used as the contrast case for the HiGHS candidate
- `over-engineering` — HiGHS-loud / CLP-silent basis validation (highs/interface.rs:487 SolverError::BasisInconsistent against clp/solver.rs:213-216) — sanctioned by: plans/architecture-debt-audit/stations/solver-comm/prior-register.md - Intended behaviour, never a finding
- `test-bloat` — crates/cobre-solver/tests/_q1_sign_convention_probe.rs — sanctioned by: plans/architecture-debt-audit/stations/solver-comm/prompts/sc-test-bloat.md — P14 Protected
- `test-bloat` — crates/cobre-solver/tests/_clp_sign_convention_probe.rs — sanctioned by: plans/architecture-debt-audit/stations/solver-comm/prompts/sc-test-bloat.md — P14 Protected
- `test-bloat` — crates/cobre-comm/src/traits.rs:372 SharedMemoryProvider hierarchy (with SharedRegion, LocalCommunicator, LocalCommKind, HeapRegion, split_local) — sanctioned by: docs/design/reserved-seams-and-deferred-debt.md:79 — Shared-memory communicator trait hierarchy
- `test-bloat` — crates/cobre-solver/tests/conformance.rs:1601 and :1636 against crates/cobre-solver/src/backends/highs/tests.rs:548 and crates/cobre-solver/src/backends/clp/tests.rs:609
- `test-bloat` — crates/cobre-solver/tests/profile_retry_composition.rs:23 whole-file #[cfg(feature = "test-support")] gate
- `test-bloat` — crates/cobre-solver/src/backends/highs/tests.rs:796 mod research_tests (the six asserting tests)

## Needs-human items surfaced by the workers (for the owner gate)

- `architecture` — Where does the shed geometry land: the fields' owners today (StateSpace, StageRowLayout) both live in crates/cobre-sddp/src/lp/, which is exactly the engine-neutral region phase 0b carves into cobre-model, so the owner must say whether Part-I item 8 sheds into the engine and moves again at 0b, or waits for the carve and sheds once.
- `architecture` — Should the LocalCommKind enum live in traits.rs at all: it makes the trait-definition module import both concrete backends (traits.rs:21,23, both behind the shared-memory feature) while the sibling factory.rs:62 CommBackend does the identical enum-dispatch job in the module that already owns concrete-backend enumeration — an ownership question about placement only, distinct from the sanctioned existence-of-the-seam ruling, which nothing here proposes to reopen.
- `architecture` — Does the layering brief's ban on multistage vocabulary in L0 bind production doc comments today, or only at the phase-1 purification: the project's hard genericity rule enumerates sddp/SDDP/Benders and none of the seven solver-side doc sites contains any of them, so the answer decides whether that candidate is a now-fix or a purification rider.
- `architecture` — Is giving up `unsafe_code = "forbid"` at the workspace root acceptable in exchange for a single lint table: the second fix shape for the lint drift removes the drift surface entirely but trades an unoverridable prohibition for a per-crate audited allowance, which is an owner call and not an engineering preference.
- `performance` — Whether the upstream simplex library exposes an index-scoped bound writer and a bulk basis-status accessor that preserve the factorization the same way the current full-array bound calls do decides whether the first two candidates are cobre-side shim additions or an intrinsic property of that library; the shim already reaches C++-class-only methods, which suggests the former, but the owner should confirm before the fix shapes are scheduled.
- `performance` — Whether the retained column-major mirror on the CLP path must remain a full merged mirror, or whether the append path could keep the base and the appended blocks separately and merge only when the reload path actually needs a contiguous mirror, is an owner call about the retained-mirror contract rather than something the code alone settles.
- `over-engineering` — The HiGHS escalation branches on wall-clock time (overall_budget at highs/retry.rs:40, the elapsed break at :52, the budget_exceeded comparison at :84) while the CLP ladder documents at clp/retry.rs:4-5 that it has no time-dependent branching so results stay bit-for-bit identical across thread and rank counts -- whether the HiGHS ladder is deliberately exempt from that property or this is a live determinism divergence is an owner call outside the over-engineering lens.
- `over-engineering` — Whether ExecutionTopology::is_homogeneous is intended for a planned heterogeneous-layout guard (in which case it needs a reserved-seam entry with an activating milestone, not deletion) or is simply left over -- the register carries no entry either way.
- `over-engineering` — Whether the CLP hot-start half should be wired onto the re-solve path or registered as a reserved seam is an owner decision; the candidate deliberately proposes no removal.
- `test-bloat` — crates/cobre-comm/tests/local_conformance.rs:4 cites backend-testing.md for the SS1.1-SS1.8 contract sections, and no file of that name exists anywhere in the tree at the baseline; whether the stale spec citation is this station's to fix or the docs station's (E07) needs the owner.
- `test-bloat` — Whether clp_only_smoke.rs was created as a deliberate belt-and-braces guard against a future re-gating of conformance.rs, in which case the binary stays despite its assertion being already covered and only its false module doc and its fixture copy need fixing.

## Gaps carried forward

None: every lens merged. Two lenses independently emitted the cut-sync `dupOf` handoff; the ingest
ticket folds them into a single dup-of record. Anchor RESOLUTION at the pinned SHA is deliberately
not checked here — it is `check-anchors.py`'s job at ingest, where a stale anchor is rejected once
with the anchor-missing code instead of being silently dropped by the shape gate.
