# Ingest log — station cobre-solver + cobre-comm

Baseline: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`  ·  ingested 2026-09

Sources screened: `candidates.architecture.json`, `candidates.performance.json`,
`candidates.over-engineering.json`, `candidates.test-bloat.json` (the four merged attacker files,
each re-validated with `tools/validate-envelope.py --role attacker --station solver-comm`, exit 0).
candidateRef re-keyed to `<lens>-<nn>` where `nn` is the attacker ref's own `NNN`
(`SC-ARCH-006` ↔ `architecture-06`); the attacker ref is preserved on every row below and as
`attackerRef` in `verdicts.json`. Every candidate is accounted for exactly once below and in
`verdicts.json`; the screens ran in the fixed order anchor → scope → reserved seams → prior
register → informational routing → defender.

## Candidate census

| lens | file | candidates |
|---|---|---|
| architecture | candidates.architecture.json | 9 |
| performance | candidates.performance.json | 4 |
| over-engineering | candidates.over-engineering.json | 5 |
| test-bloat | candidates.test-bloat.json | 6 |
| **total** |  | **24** |

No lens returned zero candidates; there is no no-finding line to record.

Perf routing rule: the four performance candidates arrived with letter severities (`C`, `B`, `B`,
`B`) and no numeric severity, timing, ratio or percentage anywhere in title, mechanism, fixShape or
evidence (the E04-3 merge guard's `TIMING_NUMBER` regex, re-run here); nothing had to be stripped.
`measurementLayout` is preserved on each: `performance-01` `2x2`, `performance-02` `4t`,
`performance-03` `4t`, `performance-04` `4t`.

## Anchor and scope rejections

None. `anchor-probe.md` renders all 110 anchors of the 24 candidates (69 as `path::symbol`
declaration anchors, 40 as `path:line` field / manifest / call-site anchors, 1 listed separately);
`check-anchors.py "INGEST ANCHOR PROBE — solver-comm (2026-09, baseline)" --register
anchor-probe.md --baseline 077dbe2c…` reports `checked 109 anchors, 0 failing`. The one anchor the
harness `ANCHOR_RE` cannot parse — `crates/cobre-solver/csrc/clp_wrapper.c` `cobre_clp_chg_bounds`
L208 on `performance-02` (no `.c` in the regex's extension set) — resolves directly
(`git show 077dbe2c:crates/cobre-solver/csrc/clp_wrapper.c | grep -n cobre_clp_chg_bounds` →
`208:static void cobre_clp_chg_bounds(`), recorded in the probe's last section rather than by
editing the harness.

Station scope: 80 anchors under `crates/cobre-solver/`, 23 under `crates/cobre-comm/`, 7 under
`crates/cobre-sddp/src/cut/cut_sync.rs` — the 7 all belong to the two `dupOf` candidates the next
screen routes (`architecture-02` 4, `over-engineering-02` 3), the single legal exception. No
`out-of-station` disposition (`anchorRejected = 0`, `outOfStation = 0`).

| candidateRef | failing anchor | checker diagnostic |
|---|---|---|
| — | — | — |

## Cleared (sanctioned)

None reached this screen as a candidate (`sanctionedCleared = 0`). The shared-memory communicator
hierarchy — `SharedMemoryProvider`, `SharedRegion<T>`, `LocalCommunicator`, `LocalCommKind`,
`HeapRegion<T>`, `FerrompiBackend::split_local` — is the mirror entry
**`Shared-memory communicator trait hierarchy`**
(`docs/design/reserved-seams-and-deferred-debt.md:79` at this baseline) and was applied as an
ingest FILTER in the attacker prompt (rule 5): all four lenses
emitted it in `positives` with `sanctionedBy` = that heading, none as a candidate, so nothing
proposes its removal, nothing reaches a defender and no OD id exists for it. `test-bloat-02`
(duplicated Communicator contract tests) explicitly excludes the shared-memory tests from its
scope for the same reason; its defender was told not to touch them.

## Re-raise and dup-of rejections

**Dup-of (2, `dupOf = 2`).** `architecture-02` (SC-ARCH-002) and `over-engineering-02` (SC-OE-002)
both name `CutSyncBuffers::sync_cuts`, `pack_local_records`, `sync_packed_records` as superseded
public API. Verified at baseline with `git show 077dbe2c:crates/cobre-sddp/src/cut/cut_sync.rs |
grep -nE '^\s*pub fn (sync_cuts|pack_local_records|sync_packed_records|sync_level_records)\b'` →
`243`, `400`, `495`, `581` (`pub struct CutSyncBuffers` at `71`): all three superseded methods
plus the superseding `sync_level_records` resolve. Both are recorded disposition `dup-of` the
mirror entry **`Superseded cut-sync public methods`** (`:334` at this baseline; register `CD-019`),
one handoff record each appended to `handoffs.json` (`records[]`, `toEpic E5`, `kind dup-of`,
`assignedId null`; `E5.ingestRecords` lists both), no defender dispatched, no new id — a new id on
this route is a `check-reraise.py` failure the verification ticket catches. Two lenses raising the
same dup-of is folded onto the single pre-existing `E5` block, not two.

**Re-raise rejections: none (`reRaiseRejected = 0`).** Every candidate was screened against the
four `## Do not re-raise` items of `prior-register.md`; five tripped a coarse substring and were
adjudicated **distinct**:

| candidateRef | prior item tripped | adjudication |
|---|---|---|
| architecture-01 | per-crate `[lints]` tables (seeded) | delivers what the seed owes — the fix-shape (two shapes, a checker or a shared include) and severity `B` with drift evidence — not the bare observation; distinct |
| architecture-05 | basis-validation divergence (`silent`) | CLP basis-status *code* spellings, not the HiGHS/CLP validation asymmetry; distinct |
| over-engineering-05 | basis-validation divergence (`silently`) | HiGHS retry-ladder duplication; the word is about a tolerance typo; distinct |
| test-bloat-02 | shared-memory hierarchy (symbol names) | names the six symbols only to EXCLUDE their tests from scope under the ingest filter; distinct |
| test-bloat-05 | basis-validation divergence (`silent`) | `clp_only_smoke.rs` binary redundancy; distinct |

## Informational (recorded, no severity)

| candidateRef | claim | roadmapRef | Alignment |
|---|---|---|---|
| architecture-09 (SC-ARCH-009) | `SolverInterface` (`crates/cobre-solver/src/trait_def.rs:41`) carries no capability or feature query; the `compile_error!` pair in `crates/cobre-solver/src/lib.rs:44,50` makes exactly one of `highs`/`clp` compile and `ActiveSolver`/`ActiveProfile` alias the winner; the trait's only backend-varying surface is the opaque `Profile` associated type | `plans/generalizing/beyond-sddp-generalization.md` § III.6 — Solver-paradigm expansion; per-feature capability traits arrive with their second consumer | neutral |

Disposition `informational`, no defender, no severity, no id.

## Held as conflicts (L0 purity test)

| candidateRef | condition hit | held claim | roadmap-consistent alternative |
|---|---|---|---|
| — (no build-it-now variant arrived) | abstraction with exactly one consumer | build the capability trait now | keep the informational record above; the trait earns its consumer when two backends must coexist in one binary (III.6) |

No candidate proposed building the trait at this baseline; SC-ARCH-009's own fixShape pre-states
the hold ("any variant that builds the capability trait now has exactly one implementation … should
be tagged `conflicts` and held, with this record cited as the reason"). No defender verdict carries
`alignmentHint: conflicts` (`conflicts = 0`).

## Part-I cross-references

Three candidates carry `partIRef: I.3-8` (attacker hint `advances-1` on all three), all confirmed
by their defender and tagged so the calibration ticket merges them with the per-field dispositions
in `partI-handoff.json` instead of double-recording the shed:

- `architecture-06` (SC-ARCH-006) — the field-by-field re-verification plus the fixture-collateral
  census (22 in-src + 18 integration mentions) and two widened assertion spans
- `test-bloat-03` (SC-TB-003) — the SS1.1 stage-template fixture re-declared eight times; the
  defender narrows the unjustified duplication to the four in-src `#[cfg(test)]` declarations
- `test-bloat-06` (SC-TB-006) — the two conformance.rs fixture-contract tests asserting a literal
  against its own transcription

## Defender pass

21 survivors (24 received − 2 dup-of − 1 informational), one read-only Opus `adversarial-defender`
each, dispatched in three parallel batches (7 architecture / 8 over-engineering + performance /
6 test-bloat) with `defender-prompt.md` verbatim plus the single candidate object, envelopes handed
back through a scratch file outside the repository (`WRITTEN <bytes> <path>`). Every envelope
validated on first dispatch — `tools/validate-envelope.py --role defender --station solver-comm`
exit 0 on all 21, plus the four E4 clauses (`survivingClaim` strictly narrower than the title,
`sanctionedBy` ∈ the two mirror headings, `intendedBehaviour` present on any dismissal resting on
the basis-validation asymmetry, no timing number on a perf verdict): **0 re-dispatches, 0
malformed, 0 non-narrowing**. Result: **20 confirmed, 1 dismissed**, 3 defender-raised
`_needsHuman` notes for the owner gate. The defender fields are carried verbatim into
`verdicts.json` (`defender` object per candidate); no raw defender copies are kept, matching the
core-io and stochastic stations.

- **Dismissed — `performance-01`** (partition helpers `per_rank_counts` / `prefix_displs`, `2x2`):
  the title's load-bearing clause (recomputed at every collective) does not hold at baseline —
  `cut_sync.rs:182/:186` sit inside `CutSyncBuffers::with_distribution` (`:173`), the
  pre-allocating constructor called once per training session (`training/session/mod.rs:252`),
  and `rank_distribution.rs:61` is likewise once-per-session; setup-time recomputation is
  acceptable by project rule. No `sanctionedBy` (not a reserved-seam dismissal); no
  `intendedBehaviour` (does not rest on the basis asymmetry).
- **No dismissal rests on the HiGHS-loud / CLP-silent basis-validation asymmetry**, so no
  `intendedBehaviour` field was required; `architecture-05` and `performance-04` (the two CLP basis
  candidates) were confirmed on the code-spelling and per-element-FFI mechanisms respectively.
- **Narrowing examples.** `architecture-01` → the drift mechanism only: a crate-level `[lints]`
  table *replaces* `[workspace.lints]` and no `scripts/ci/` checker compares them, so a dropped
  `deny` cannot make CI red. `performance-02` → the allocation only: `cobre_clp_chg_bounds`
  (`clp_wrapper.c:208`) heap-allocates, fills and frees a full-dimension sentinel-translation
  buffer on each of the four `chg_*` crossings. `performance-03` → Rust side only, the CSC trio on
  the per-iteration append path. `over-engineering-04` → the missing guard alone: the 12-value
  agreement between `HighsProfile::default()` and `default_options()` is load-bearing for the
  delta-only dispatch in `ProfiledSolver` and nothing pins it. `test-bloat-05` → two anchors: the
  `clp_only_smoke.rs` module-doc guarantee is false under the very feature combination it names.
- **Needs-human (3), for the owner gate:**
  - `architecture-07` — direction of the residue: keep the canonical basis-status mapping
    unconditional and correct `README.md:106-107`, or name the CLP codes in `ffi::clp` and gate
    each half (which also makes `BasisStatus::to_highs_code`/`from_highs_code` and the
    legacy-checkpoint agreement test HiGHS-only).
  - `over-engineering-03` — the CLP hot-start acquire/solve pair (`cobre_clp_mark_hot_start`,
    `cobre_clp_solve_from_hot_start`, `backends/clp/solver.rs:350/:394`) has no production caller
    and no register entry; the owner supplies the activating milestone or rules it retired at the
    next licensed public-API break — the station cannot invent either.
  - `test-bloat-03` — the four integration binaries: a `tests/common` shared-fixture module (the
    cobre-sddp idiom, no feature gate) or accept that a `test-support`-gated fixture makes the
    documented clp-only invocations (`CONTRIBUTING.md:71`, `crates/cobre-solver/README.md:84`) run
    zero cobre-solver integration tests.

Alignment over all 24: {neutral: 22, advances-1: 2 (`architecture-06`, `test-bloat-06`; the
third I.3-8 candidate `test-bloat-03` was narrowed to in-crate `#[cfg(test)]` duplication with no
feature gate and re-hinted `neutral` by its defender — the `partIRef` travels regardless)}.
Confirmed by lens and proposed severity: architecture 3 B / 4 C, over-engineering
2 B / 2 C, performance 3 B, test-bloat 1 B / 5 C — the calibration ticket owns the house A/B/C
call; these are the attackers' proposals.

## Per-candidate roster

Every one of the 24 candidateRefs, exactly once (mirrors `verdicts.json`; `state` is the ticket's
test vocabulary — accepted / rejected-anchor / rejected-reserved-seam / rejected-defender / merged
— derived from disposition + verdict).

| candidateRef | attackerRef | disposition | verdict | state | partIRef |
|---|---|---|---|---|---|
| architecture-01 | SC-ARCH-001 | defended | confirmed | accepted |  |
| architecture-02 | SC-ARCH-002 | dup-of | merged → CD-019 | merged |  |
| architecture-03 | SC-ARCH-003 | defended | confirmed | accepted |  |
| architecture-04 | SC-ARCH-004 | defended | confirmed | accepted |  |
| architecture-05 | SC-ARCH-005 | defended | confirmed | accepted |  |
| architecture-06 | SC-ARCH-006 | defended | confirmed | accepted | I.3-8 |
| architecture-07 | SC-ARCH-007 | defended | confirmed | accepted |  |
| architecture-08 | SC-ARCH-008 | defended | confirmed | accepted |  |
| architecture-09 | SC-ARCH-009 | informational | recorded, no severity | accepted |  |
| over-engineering-01 | SC-OE-001 | defended | confirmed | accepted |  |
| over-engineering-02 | SC-OE-002 | dup-of | merged → CD-019 | merged |  |
| over-engineering-03 | SC-OE-003 | defended | confirmed | accepted |  |
| over-engineering-04 | SC-OE-004 | defended | confirmed | accepted |  |
| over-engineering-05 | SC-OE-005 | defended | confirmed | accepted |  |
| performance-01 | SC-PERF-001 | defended | dismissed | rejected-defender |  |
| performance-02 | SC-PERF-002 | defended | confirmed | accepted |  |
| performance-03 | SC-PERF-003 | defended | confirmed | accepted |  |
| performance-04 | SC-PERF-004 | defended | confirmed | accepted |  |
| test-bloat-01 | SC-TB-001 | defended | confirmed | accepted |  |
| test-bloat-02 | SC-TB-002 | defended | confirmed | accepted |  |
| test-bloat-03 | SC-TB-003 | defended | confirmed | accepted | I.3-8 |
| test-bloat-04 | SC-TB-004 | defended | confirmed | accepted |  |
| test-bloat-05 | SC-TB-005 | defended | confirmed | accepted |  |
| test-bloat-06 | SC-TB-006 | defended | confirmed | accepted | I.3-8 |

Counts (`verdicts.json.counts`): received 24 = anchorRejected 0 + outOfStation 0 +
sanctionedCleared 0 + dupOf 2 + reRaiseRejected 0 + informational 1 + defended 21;
defended 21 = confirmed 20 + dismissed 1; needsHuman 3.
