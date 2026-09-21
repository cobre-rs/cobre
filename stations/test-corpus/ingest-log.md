# Ingest log — station test corpus and test-support

Baseline: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (register pin; the ticket's `a136840d` figures — cobre-sddp 37 binaries vs 53 files, cobre-io 12 vs 13, the StubComm census lines 3326 / 1655 / 2768 — are the superseded scaffold pin's; at the pin they are 40 / 56, 12 / 13 and 3555 / 1654 / 2767 and every difference is recorded, never edited)  ·  ingested 2026-09-19  ·  yardstick: docs/design/testing-architecture.md (Proposal — §5 is a target; §2 a present-tense claim)  ·  sources: `candidates-test-bloat.json` (primary), `candidates-architecture.json`, `candidates-over-engineering.json`, `candidates-performance.json`. Read-only over the repository: nothing under `crates/`, `.github/` or `docs/` is modified; no TD id is assigned and no BACKLOG section is written here.

## Candidate census

| source | lens | candidates | shared validator | station profile |
|---|---|---|---|---|
| candidates-test-bloat.json | test-bloat | 73 | exit 0 | exit 0 |
| candidates-architecture.json | architecture | 3 | exit 0 | exit 0 |
| candidates-over-engineering.json | over-engineering | 2 | exit 0 | exit 0 |
| candidates-performance.json | performance | 3 | exit 0 | exit 0 |

Received 81 candidates (test-bloat 73, architecture 3, over-engineering 2, performance 3). No lens is empty at this station; an empty lens would be written here as a count-0 row, not an error. Every candidate carries `measurementDefinition` (the profile validator rejects a `measuredValue` without one), so the definition screen can run over all of them. Refs are `<lens>-<nn>`, zero-based in file order; every later table, the defender dispatch and `verdicts.json` key on them.

## Anchor screen

`anchor-probe.md` renders every anchor of every candidate (`path::symbol` for declarations, `path:line` otherwise) as a register-shaped stub under the heading `INGEST ANCHOR PROBE — test-corpus (2026-09, baseline)`; `check-anchors.py "INGEST ANCHOR PROBE — test-corpus (2026-09, baseline)" --register stations/test-corpus/anchor-probe.md --baseline 077dbe2c` resolved 424 anchors with the register's own parser and declaration regex (exit 0; 0 failing). The three anchor shapes the ticket names all occur in the probe and resolve: integration path + free fn (for example `crates/cobre-sddp/tests/anticipated_core.rs::build_system`, `crates/cobre-cli/tests/cli_run.rs::cobre`), harness path + declaration (`crates/cobre-sddp/tests/common/mod.rs:32`, the `Rank0Of2` line, cited by line because the attackers anchored the harness by line; `crates/cobre-stochastic/tests/common/mod.rs::deficit_bus` in symbol form) and workflow path + line (`.github/workflows/ci.yml:32`). Deviation from the ticket text: its example instance for the first shape, `crates/cobre-sddp/tests/extensive_form_oracle.rs::close`, is not a candidate anchor at the pin — no attacker anchored the oracle `close` helper, because the oracle-harness duplication item arrived as a dup-of (test-bloat-04) whose sharpened anchor pair is recorded on the dup-of row of verdicts.json, not in the probe.

### Anchor rejections

None: all 424 anchors resolve at the pin. The ticket's load-bearing case — a §5.8-derived candidate anchoring `StubComm` / `Rank0Of2` at `crates/cobre-comm` — did not arise: the attacker prompt (E08-3, rule 11 and P6) told the cobre-comm worker that the symbols are not there, so its §5.8 candidate (test-bloat-30) anchors the doc sentence (`docs/design/testing-architecture.md:524` / `:415`), the manifest that declares no `test-support` feature (`crates/cobre-comm/Cargo.toml:20`) and the `Communicator` trait it would implement (`crates/cobre-comm/src/traits.rs::Communicator`), while the cobre-sddp worker's twin (test-bloat-43) anchors the canonical home `crates/cobre-sddp/tests/common/mod.rs:32` / `:86` and the private copies. Counter-evidence at the pin, re-run here: `git grep -n 'test-support' 077dbe2c -- crates/cobre-comm/Cargo.toml` → no hit; `git grep -n 'pub struct StubComm\|pub struct Rank0Of2' 077dbe2c -- crates/cobre-sddp/tests/common/mod.rs` → :32 and :86.

## Definition screen (binary vs file)

Every candidate carrying a `measuredValue` was re-resolved against both labelled inventory keys where a pair exists (`figures.int-binaries.perCrate.<crate>` = depth-1 `tests/*.rs`, what Cargo links; `figures.int-binaries.altDefinition.perCrate.<crate>` = recursive `tests/**/*.rs`, what a grep sees; the pin's live pairs are cobre-sddp 40 vs 56, cobre-io 12 vs 13, cobre-stochastic 9 vs 10, workspace 62 solver-linking vs 105 files). A claim flips only when a comparative or threshold in its title changes truth between the two values; a claim that uses the definition as a LABEL ("N binaries re-declare X") is stable and carries its label forward for the calibration ticket. Unpaired keys (homing-split, doctests, nextest, manifests, workflow facts) have one definition and are stable by construction.

| candidateRef | definition | key | value | paired key | paired value | comparative in title | result |
|---|---|---|---|---|---|---|---|
| test-bloat-07 | binary | `figures.int-binaries.perCrate.cobre-io` | 12 | `figures.int-binaries.altDefinition.perCrate.cobre-io` | 13 | — | stable (definition is a label; the asserted fact does not depend on the denominator) |
| test-bloat-13 | binary | `figures.int-binaries.perCrate.cobre-solver` | 8 | `figures.int-binaries.altDefinition.perCrate.cobre-solver` | 8 | — | stable (definition is a label; the asserted fact does not depend on the denominator) |
| test-bloat-15 | binary | `figures.int-binaries-solver-linking.perCrate.cobre-solver` | 8 | `figures.int-binaries.altDefinition.perCrate.cobre-solver` | 8 | — | stable (definition is a label; the asserted fact does not depend on the denominator) |
| test-bloat-24 | binary | `figures.int-binaries.perCrate.cobre-stochastic` | 9 | `figures.int-binaries.altDefinition.perCrate.cobre-stochastic` | 10 | — | stable (definition is a label; the asserted fact does not depend on the denominator) |
| test-bloat-28 | binary | `figures.int-binaries.perCrate.cobre-comm` | 2 | `figures.int-binaries.altDefinition.perCrate.cobre-comm` | 2 | — | stable (definition is a label; the asserted fact does not depend on the denominator) |
| test-bloat-33 | binary | `figures.int-binaries-solver-linking.perCrate.cobre-sddp` | 40 | `figures.int-binaries.altDefinition.perCrate.cobre-sddp` | 56 | — | stable (definition is a label; the asserted fact does not depend on the denominator) |
| test-bloat-35 | binary | `figures.int-binaries-solver-linking.perCrate.cobre-sddp` | 40 | `figures.int-binaries.altDefinition.perCrate.cobre-sddp` | 56 | — | stable (definition is a label; the asserted fact does not depend on the denominator) |
| test-bloat-41 | binary | `figures.int-binaries-solver-linking.perCrate.cobre-sddp` | 40 | `figures.int-binaries.altDefinition.perCrate.cobre-sddp` | 56 | — | stable (definition is a label; the asserted fact does not depend on the denominator) |
| test-bloat-43 | binary | `figures.int-binaries.perCrate.cobre-sddp` | 40 | `figures.int-binaries.altDefinition.perCrate.cobre-sddp` | 56 | — | stable (definition is a label; the asserted fact does not depend on the denominator) |
| test-bloat-47 | binary | `figures.int-binaries.perCrate.cobre-cli` | 14 | `figures.int-binaries.altDefinition.perCrate.cobre-cli` | 14 | — | stable (definition is a label; the asserted fact does not depend on the denominator) |
| test-bloat-51 | binary | `figures.int-binaries.perCrate.cobre-cli` | 14 | `figures.int-binaries.altDefinition.perCrate.cobre-cli` | 14 | — | stable (definition is a label; the asserted fact does not depend on the denominator) |
| test-bloat-52 | binary | `figures.int-binaries.perCrate.cobre-cli` | 14 | `figures.int-binaries.altDefinition.perCrate.cobre-cli` | 14 | — | stable (definition is a label; the asserted fact does not depend on the denominator) |
| test-bloat-54 | binary | `figures.int-binaries.perCrate.cobre-cli` | 14 | `figures.int-binaries.altDefinition.perCrate.cobre-cli` | 14 | — | stable (definition is a label; the asserted fact does not depend on the denominator) |
| test-bloat-65 | binary | `figures.int-binaries.perCrate.cobre-python` | 0 | `figures.int-binaries.altDefinition.perCrate.cobre-python` | 0 | — | stable (definition is a label; the asserted fact does not depend on the denominator) |
| architecture-00 | binary | `figures.int-binaries.perCrate.cobre-sddp` | 40 | `figures.int-binaries.altDefinition.perCrate.cobre-sddp` | 56 | — | stable (definition is a label; the asserted fact does not depend on the denominator) |
| architecture-01 | file | `figures.int-binaries.altDefinition.perCrate.cobre-stochastic` | 10 | `figures.int-binaries.perCrate.cobre-stochastic` | 9 | — | stable (definition is a label; the asserted fact does not depend on the denominator) |
| architecture-02 | file | `figures.int-binaries.altDefinition.perCrate.cobre-io` | 13 | `figures.int-binaries.perCrate.cobre-io` | 12 | — | stable (definition is a label; the asserted fact does not depend on the denominator) |
| over-engineering-01 | binary | `figures.int-binaries.perCrate.cobre-sddp` | 40 | `figures.int-binaries.altDefinition.perCrate.cobre-sddp` | 56 | — | stable (definition is a label; the asserted fact does not depend on the denominator) |
| performance-00 | binary | `figures.int-binaries-solver-linking.value` | 62 | `figures.int-binaries.altDefinition.integrationFiles` | 105 | — | stable (definition is a label; the asserted fact does not depend on the denominator) |
| performance-01 | binary | `figures.int-binaries-solver-linking.perCrate.cobre-sddp` | 40 | `figures.int-binaries.altDefinition.perCrate.cobre-sddp` | 56 | most of | stable under both definitions (most of: 40/62 = 0.65 binaries vs 56/78 = 0.72 files) |

Unpaired (single-definition) keys, stable by construction: test-bloat-03, test-bloat-05, test-bloat-06, test-bloat-08, test-bloat-09, test-bloat-11, test-bloat-17, test-bloat-18, test-bloat-19, test-bloat-20, test-bloat-21, test-bloat-22, test-bloat-25, test-bloat-30, test-bloat-31, test-bloat-37, test-bloat-44, test-bloat-48, test-bloat-49, test-bloat-50, test-bloat-53, test-bloat-55, test-bloat-56, test-bloat-62, test-bloat-67, test-bloat-68, test-bloat-69, test-bloat-70, performance-02.

### Definition-flip downgrades

None. No candidate states a threshold or comparative whose truth differs between the depth-1 and the recursive count; every paired candidate uses the count as a definition label (`raisedBy` workers cited the inventory key, rule 9). The mechanics are exercised by `IngestTests.test_definition_flip_screen_downgrades_a_threshold_claim` on a synthetic 'more than 45 binaries' claim (40 vs 56 → flip → needs-human, severity cleared).

## Positives (coverage-reduction dismissals)

Every `fixShape` in every lens was scanned before any defender was dispatched with the pattern `delet|remov|drop|prune|trim|skip|#\[ignore\]|shrink the suite|fewer tests`; each hit was classified by the sentence it sits in: NEGATED ("no test is deleted, skipped or weakened", "nothing is removed", "instead of deleting", "recorded under positives"), NON-TEST OBJECT (the verb acts on a helper copy, a duplicate declaration, an import, a `cfg` gate, a solve count, a module re-declaration or a helper directory such as `tests/helpers` — a path token, not a test — coverage-neutral de-duplication), or TEST-DELETION (the verb acts on a test, assertion, probe or case). Only TEST-DELETION dismisses, citing `docs/design/testing-architecture.md:602-604` (§7 'Reducing coverage to shrink the suite — the cost is per-binary and per-feature-combo, never per-test') and `.claude/rules/testing.md:113` ('Cost discipline — Cobre links a solver into every test binary': add the case to an existing domain binary, group with `mod`, hoist the helper into a test-support surface).

Regex hits: 98 across 47 candidates — negated 91, non-test object 7, test-deletion 0.

| candidateRef | fixShape excerpt | dismissedBy | coverage-neutral alternative |
|---|---|---|---|
| — | — | — | no candidate proposes a test deletion: the attackers applied rule 7 (coverage-neutral fix-shapes only) and disposed of the four deletion-shaped upstream seeds (TD-011 / TD-012 / TD-015 / TD-019) as dropped-with-reason under positives before they became candidates; the two cli candidates on subset tests (test-bloat-53 / -54) state the fold as an owner-gate question rather than a deletion |

Hits reviewed and kept (classification per hit):

- `test-bloat-00`: [delet] negated; [skip] negated
- `test-bloat-01`: [remov] negated; [delet] negated; [skip] negated
- `test-bloat-03`: [delet] negated; [skip] negated
- `test-bloat-07`: [delet] non-test-object; [remov] non-test-object
- `test-bloat-09`: [drop] negated; [remov] negated
- `test-bloat-10`: [remov] negated
- `test-bloat-12`: [remov] negated
- `test-bloat-13`: [delet] negated; [delet] negated; [skip] negated
- `test-bloat-14`: [delet] negated; [skip] negated
- `test-bloat-15`: [Delet] negated
- `test-bloat-16`: [delet] negated
- `test-bloat-17`: [Drop] non-test-object; [delet] negated; [skip] negated
- `test-bloat-19`: [delet] negated; [skip] negated
- `test-bloat-20`: [delet] negated; [skip] negated
- `test-bloat-21`: [delet] negated; [skip] negated
- `test-bloat-25`: [delet] negated; [skip] negated; [remov] negated
- `test-bloat-28`: [remov] negated; [delet] negated; [skip] negated; [delet] negated; [delet] negated
- `test-bloat-29`: [remov] negated; [remov] negated; [remov] negated; [delet] negated
- `test-bloat-30`: [delet] negated
- `test-bloat-32`: [delet] negated; [skip] negated
- `test-bloat-43`: [delet] negated
- `test-bloat-45`: [remov] negated; [delet] negated; [skip] negated
- `test-bloat-47`: [drop] non-test-object; [delet] negated; [skip] negated
- `test-bloat-48`: [remov] non-test-object
- `test-bloat-49`: [remov] negated; [drop] negated
- `test-bloat-50`: [delet] negated; [skip] negated; [remov] negated
- `test-bloat-51`: [drop] non-test-object; [delet] negated; [skip] negated
- `test-bloat-52`: [delet] negated; [skip] negated
- `test-bloat-53`: [remov] negated
- `test-bloat-54`: [remov] negated; [remov] negated
- `test-bloat-55`: [delet] negated; [drop] negated; [remov] negated
- `test-bloat-56`: [delet] negated
- `test-bloat-57`: [remov] negated
- `test-bloat-59`: [remov] negated; [skip] negated
- `test-bloat-61`: [remov] negated
- `test-bloat-63`: [remov] negated
- `test-bloat-64`: [drop] non-test-object
- `test-bloat-67`: [delet] negated; [skip] negated; [delet] negated
- `test-bloat-68`: [drop] negated; [remov] negated; [skip] negated
- `test-bloat-70`: [remov] negated; [skip] negated
- `test-bloat-71`: [remov] negated; [skip] negated
- `test-bloat-72`: [skip] negated; [remov] negated; [skip] negated
- `architecture-00`: [remov] negated; [skip] negated
- `over-engineering-00`: [delet] negated; [skip] negated
- `performance-00`: [delet] negated; [skip] negated
- `performance-01`: [Delet] negated; [skip] negated
- `performance-02`: [delet] negated; [skip] negated

## Prior register, do-not-touch and re-raise screen

`check-reraise.py "INGEST ANCHOR PROBE — test-corpus (2026-09, baseline)" --register stations/test-corpus/anchor-probe.md --baseline 077dbe2c` screened all 81 titles against the retired corpus (BACKLOG do-not-touch list, BACKLOG cleared verdicts, the mirror's H3 headings at the pin, the resolved Part-VI forks): exit 0, 0 unjustified re-raise(s). A `dupOf` candidate renders a `Re-raise-of:` line naming its mirror heading, so its restatement is a JUSTIFIED hit (a merge, not a re-raise).

### Dup-of merges (no fresh TD id)

| candidateRef | merges into (mirror heading) | sharpened baseline anchor | related register ids |
|---|---|---|---|
| test-bloat-04 | Oracle test-harness duplication | `crates/cobre-sddp/tests/extensive_form_oracle.rs::close` + `crates/cobre-sddp/tests/branching_value_oracle.rs::close` (four carriers at the pin: also `crates/cobre-sddp/tests/node_native_backward_gate.rs:76`, `crates/cobre-sddp/tests/mpi_wire.rs:2816`) | none — mirror-only item (prior-register.md) |
| test-bloat-37 | Mega-file / inline-test-giant asymmetry | `crates/cobre-sddp/src/lp/builder/entries.rs:1680` and `crates/cobre-sddp/src/lp/builder/columns.rs:1305` inline `#[cfg(test)]` vs the extracted `crates/cobre-sddp/src/lp/builder/template/tests.rs` (`template.rs:1328`) and `crates/cobre-sddp/src/lp/builder/layout/tests.rs` (`layout.rs:1962`) | CD-007, TD-047, CD-021 |
| test-bloat-65 | Python-binding Rust tests invisible to CI | `crates/cobre-python/Cargo.toml:34` ([dev-dependencies] names no cobre-sddp test-support) — the mirror's 'never compiled in CI' premise is superseded at the pin by `.github/workflows/ci.yml:567` (cargo test --manifest-path crates/cobre-python/Cargo.toml; SN-06); residue (a)-(c) per prior-register.md | CD-103 |

### Keyword hits reviewed as distinct instances (defended, not merged)

- `architecture-00` matched `Python-binding Rust tests invisible to CI` vocabulary but is a distinct instance: The test-support surface has two activation regimes: cobre-core, cobre-io and cobre-stochastic pin the feature with a se…
- `architecture-02` matched `Oracle test-harness duplication` vocabulary but is a distinct instance: One shared-harness role carries two directory names and three compositions, and the differently-named cobre-io harness i…

### Re-raise and dup-of rejections

None: no title restates a do-not-touch, Cleared or resolved-fork item; the mirror-heading hits are the three dup-of merges above, each justified by its `Re-raise-of:` line. The superseded 'Rust tests invisible to CI' premise (SN-06) reached no candidate as a claim.

## Screen summary before defender dispatch

| disposition | count |
|---|---|
| defended | 78 |
| dup-of | 3 |
| anchor-missing | 0 |
| needs-human | 0 |
| coverage-reduction | 0 |
| re-raise | 0 |

78 survivors go to one read-only Opus `adversarial-defender` each (defender-prompt.md verbatim plus the candidate input object); the mandatory adjudications are `test-bloat-30` (§5.8 doc side), `test-bloat-43` (§5.8 tree side) and `architecture-00` (§5.2 uniformity narrative).

## Defender summary

78 survivors (81 received − 3 dup-of − 0 anchor-missing − 0 coverage-reduction − 0 definition-flip − 0 re-raise), one read-only Opus `adversarial-defender` each, dispatched in parallel batches with `defender-prompt.md` read verbatim plus the single candidate input object (`/tmp/test-corpus-defenders/in/<candidateRef>.json`: the candidate as filed, the four screen results, the claim-class rows for its yardstick section, the upstream seed context when it carries a `seedRef`, sibling candidates on the same anchors, and the attacker's needs-human items); envelopes handed back through a scratch file outside the repository (`WRITTEN <bytes> <path>`). Every envelope was gated with `tools/validate-envelope.py --role defender --station test-corpus` plus the station clauses (`/tmp/test-corpus-ingest/ingest.py validate <ref>`): exactly one verdict keyed by the candidateRef; `claimKind` in tree-fact | target-gap | prose-drift and `targetNotDefect` a boolean on EVERY verdict; `yardstickRef` and `measurementDefinition` present; `survivingClaim` strictly narrower than the title (not equal after normalisation, token-Jaccard < 0.85, ≥ 40 chars) and a `coverageNeutralShape` starting with consolidation | re-homing | feature-surface unification | cadence tiering on a confirmation; `dismissalBasis` (six-value vocabulary incl. `target-not-defect`) + `basisCitation` on a dismissal, `sanctionedBy` from the closed set iff `sanctioned-seam`, `contractCited` iff `contract`; the three mandatory adjudications return confirmed prose-drift or dismissed target-not-defect; `measurement` UNMEASURED on the performance lens (no timing literal) and n/a elsewhere; `alignmentHint` in vocabulary with `conflicts` equal to (`alignmentHint == conflicts`); `seedRef` copied verbatim; no diff or code block. One re-dispatch on failure; a second failure records `needs-human`.

**Retry history:**

- none — every envelope passed on the first dispatch. One defender (`test-bloat-14`) ran for roughly half an hour against a few minutes for its peers and received a mid-run time-check message asking it to decide on the evidence already read and put any open point in `_needsHuman`; that message was not a re-dispatch, and its single envelope passed the gate unchanged.

Result: **58 confirmed, 20 dismissed, 0 unresolved**; 73 entries carry a `_needsHuman` note for the owner gate; 9 verdicts set `targetNotDefect: true` (roadmap items, never defects); 0 set `conflicts: true`; 3 mandatory proposal-vs-tree adjudications recorded.

### Proposal-vs-tree adjudications

| candidateRef | sentence | verdict | claimKind | targetNotDefect | side that won |
|---|---|---|---|---|---|
| test-bloat-30 | §5.8 StubComm/Rank0Of2 sentence — doc side (docs/design/testing-architecture.md:524, restated :415) against crates/cobre-comm/Cargo.toml | dismissed | target-gap | True | proposal |
| test-bloat-43 | §5.8 StubComm/Rank0Of2 sentence — tree side (canonical crates/cobre-sddp/tests/common/mod.rs:32/:86 and the private copies) | dismissed | target-gap | True | proposal |
| architecture-00 | §5.2 test-support uniformity narrative (docs/design/testing-architecture.md:398-399 and :425-431) against the two activation regimes at the pin | confirmed | prose-drift | False | tree |

### Dismissals

| candidateRef | basis | citation |
|---|---|---|
| test-bloat-03 | target-not-defect | docs/design/testing-architecture.md:253-256 (section 5.1, 'One deterministic unit-test homing rule') |
| test-bloat-06 | premise-false-at-pin | crates/cobre-io/src/scenarios/correlation.rs:411 |
| test-bloat-08 | premise-false-at-pin | crates/cobre-io/src/output/stochastic.rs:786-789 |
| test-bloat-17 | premise-false-at-pin | crates/cobre-solver/src/backends/highs/solver.rs:580 |
| test-bloat-18 | target-not-defect | docs/design/testing-architecture.md:253-256 (section 5.1, 'One deterministic unit-test homing rule' -- 'proposal: ~500 test-LOC or ~40 test fns'; 'Pick the thre |
| test-bloat-25 | target-not-defect | docs/design/testing-architecture.md:253-256 (section 5.1, 'One deterministic unit-test homing rule', threshold self-labelled a proposal under the 'Proposed stan |
| test-bloat-26 | premise-false-at-pin | crates/cobre-stochastic/src/seeds.rs:104 |
| test-bloat-27 | premise-false-at-pin | scripts/ci/check-infra-genericity.sh:98-100 |
| test-bloat-30 | target-not-defect | docs/design/testing-architecture.md:3 (Status line: Partially adopted for section 5.2, the rest Proposal, target standard, not yet implemented), with the true c |
| test-bloat-35 | premise-false-at-pin | crates/cobre-sddp/src/lp/builder/test_support.rs:65 |
| test-bloat-43 | target-not-defect | docs/design/testing-architecture.md:524-526 (5.8 MPI & determinism testing standard; Status line at :3) |
| test-bloat-46 | premise-false-at-pin | crates/cobre-sddp/src/simulation/state.rs:412 |
| test-bloat-67 | deliberate-and-documented | .github/workflows/ci.yml:26-27 |
| test-bloat-68 | target-not-defect | docs/design/testing-architecture.md:476-479 — 5.5 Runner standard: nextest everywhere, configured (5. Proposed standard, :233) |
| test-bloat-69 | target-not-defect | docs/design/testing-architecture.md:478-489 (5.5 Runner standard - nextest everywhere, configured); phase 2 at :578 |
| test-bloat-70 | target-not-defect | docs/design/testing-architecture.md:476-489 - section heading '5.5 Runner standard - nextest everywhere, configured' (target); corroborated by the Tier 1 row at |
| over-engineering-01 | deliberate-and-documented | crates/cobre-sddp/tests/permute_helpers.rs:1-3 |
| performance-00 | cost-accepted-by-rule | .claude/rules/testing.md:113 |
| performance-01 | target-not-defect | docs/design/testing-architecture.md:260 |
| performance-02 | deliberate-and-documented | .github/workflows/ci.yml:155-157 |

## Needs-human items surfaced by the defenders

| candidateRef | question |
|---|---|
| test-bloat-00 | Census scope: a second identical out-of-owning-module pair exists at the pin that this candidate does not name, entities/hydro.rs:612 and model/penalty.rs:353 (its cfg(test) module opens at model/penalty.rs:335), sharing their own sixteen-field sequence; the filed reading of four full literals with  |
| test-bloat-00 | Fixture-value locality: whether both sites consume the new named constructor, or only the system-level postcard copy delegates while the value-asserting resolved-penalties module keeps its local fixture, is an owner call; both variants keep every assertion and the nextest count. |
| test-bloat-01 | The fold in coverageNeutralShape merges two test functions into one, so cargo nextest list reports one fewer test; lens-rules.md Rule 2 Boundary records that count carve-out as an owner-gate decision, not a defender's. The owner picks the fold or the count-neutral in-place variant named in the same  |
| test-bloat-01 | Carried from the attacker, still live after this read: whether strengthening a retained assertion counts as coverage-neutral under Rule 2's second question, given that the retained test then asserts a strict superset of what it asserted before. Removal is inadmissible here and leaving the two tautol |
| test-bloat-01 | The candidate files yardstickRef ta-3.2, but none of the eight claim-class rows supplied for section 3.2 covers tautological assertions; the verdict therefore rests on the tree fact and on the testing.md Contracts tautology bullet (:42-45) rather than on any section 3.2 sentence. The owner should co |
| test-bloat-02 | The consolidation fold changes the cargo nextest list count by the number of folded functions; lens-rules.md Rule 2 Boundary records that carve-out (an exact-subset assertion folded into a named retained superset on an identical fixture) as an owner-gate decision, not this lens's. |
| test-bloat-02 | Narrowed from the attacker's item: strengthening in place is moot for the five members that already carry a complete struct literal, but test_contract_type_equality at crates/cobre-core/src/entities/energy_contract.rs:87 constructs nothing, so its only non-fold shape is to make a retained test asser |
| test-bloat-03 | If the owner ratifies the section 5.1 homing threshold, decide whether cobre-core system/mod.rs (43 test fns from line 615) and model/resolved/bounds.rs (37 from line 1193) enter the already-registered re-homing scope as additional instances, or stay out because the crate is uniformly inline and sho |
| test-bloat-05 | Destination form is an owner call, not settled here: the 5.1 inline-versus-sibling threshold is unratified at the pin (claim row ta-5.1-homing-threshold is class target, measured as no ratified threshold, TD-031), so whether the re-homed checkpoint tests land inline in checkpoint.rs or in an extract |
| test-bloat-05 | Where the two facade-crossing version-marker tests at :1457 and :1487 belong: checkpoint.rs can import super::codec::serialize_checkpoint_manifest exactly as it already does at :13-17, or the pair can stay at the facade as the only genuinely cross-submodule tests in the module. |
| test-bloat-06 | TD-008 is a registered entry whose claim is that the scenarios/ homing inconsistency is the durable threshold-INDEPENDENT residue; this dismissal is of the sharpening only (rule 12 bars re-litigating the upstream severity), but the owner should note that the directory measures size-ordered at the pi |
| test-bloat-06 | Carried and re-scoped from the attacker: 'which homing direction should the scenarios directory adopt' decides nothing for scenarios/ after this read, since the directory already satisfies any threshold between its largest inline module and its sibling; the live decision is the single ~500 test-LOC  |
| test-bloat-07 | Direction for the single-consumer builder: inline make_referential_violation_case into its only caller (integration.rs:159) per the documented blast-radius principle at crates/cobre-io/src/test_support.rs:8-9, or promote it to test_support anyway for symmetry with make_multi_entity_case. The two rea |
| test-bloat-08 | Residue for the owner to fold or drop, not a new entry: four byte-identical unwrap schema-only preludes remain at hydro_models.rs:515-517, :705-707, :816-818 and training_writer.rs:987-989; TD-020 already dispositions that class as optional homing cleanup, so the owner decides whether to note them u |
| test-bloat-09 | Owner call on the expected-set source: tightening :258 to set equality needs an authoritative source for the 18 names, and the project's documented choice at crates/cobre-cli/tests/cli_schema.rs:43-44 is to derive both sides from artifacts rather than freeze a literal. Decide whether cobre-io's unit |
| test-bloat-09 | The attacker's question of whether the CI schemas diff is authoritative for the export set is ANSWERED at the pin, not open: scripts/ci/check_schemas.sh runs a recursive diff of schemas/ against a fresh export (absent-file-as-empty, so removals fail) from .github/workflows/ci.yml:338-339, and crates |
| test-bloat-10 | Direction call on TD-022: the register frames the count-only bodies as a redundant surface to remove, this verdict confirms the opposite direction (strengthen them to the name vector); only the removal direction changes the cargo nextest list count, so the owner must pick one before a fix-shape is a |
| test-bloat-11 | Destination pick for the shared make_pumping: cobre-io own test_support (which widens that crate test-support export surface, against the blast-radius sentence at crates/cobre-io/src/test_support.rs:8-9) versus cobre-core beside the PumpingStation type it builds, where crates/cobre-core/src/entities |
| test-bloat-11 | Whether the two further identical zero-penalty bodies I read at crates/cobre-io/src/broadcast.rs:199 and crates/cobre-io/src/output/simulation_writer.rs:2252 are folded under this entry at calibration or minted separately -- naming them in the surviving claim would widen the candidate beyond its fil |
| test-bloat-12 | Does the MPI broadcast payload have a ratified compactness budget? No budget, const or doc exists anywhere in cobre-io at the pin (the ceiling literal at broadcast.rs:399 is its only occurrence), so whether the tightened ceiling is stated as a named budget - or whether the canary is instead re-homed |
| test-bloat-13 | Scope call on the third fixture family: TD-037's fix-shape names the empty batch as one of the three fixtures to give a single declaration, but the three zero-row RowBatch bodies at src/freeze.rs:286, tests/sentinel_inf_row_probe.rs:76 and src/backends/profiled.rs:262 are not byte-identical (vec![]  |
| test-bloat-14 | Whether the name-versus-doc mismatch on test_highs_solve_iterations_positive (crates/cobre-solver/src/backends/highs/tests.rs:342 claims positive while its own doc at :339-340 states any u64 is valid) earns a separate register entry: its body is correct and documented, so the only repair is a rename |
| test-bloat-14 | Whether TD-038's ratified severity C moves now that the surviving widening is one carrier rather than the two the attacker filed; rule 12 leaves registered severity to the owner gate. |
| test-bloat-15 | After the fold the conformance binary holds two clp tests asserting the same objective and primals at two tolerances, 1e-9 from the moved test and 1e-8 at conformance.rs:1450. Whether both are retained (count-neutral and admissible) or the tighter bound replaces the looser assertion in the existing  |
| test-bloat-16 | TD-040's registered fix-shape removes both tests under either branch, a cargo nextest list delta of two, while this confirmation's residue keeps both: the owner must decide at the E08 gate whether the ratified removal direction or lens-rules.md Rule 2 Boundary governs this entry. |
| test-bloat-18 | figures.homing-split.perCrate.cobre-solver.inlineCfgTestModules = 8 counts files carrying the literal #[cfg(test)], which for this crate is the wrong set on both sides: it includes three files with no inline test body (backends/clp/mod.rs:14 and backends/highs/mod.rs:20, which DECLARE the extracted  |
| test-bloat-19 | Destination for the three consolidated undated presets: crates/cobre-stochastic/src/test_support.rs, whose module doc scopes it to this crate's OpeningTree, SeasonMap and InflowModel shapes so entity presets widen that declared scope, or cobre_core::test_support beside the existing BusSpec / HydroSp |
| test-bloat-19 | Scope call TD-032 does not cover: whether the four-site dated make_hydro variant (seeds.rs:180, sampling/external.rs:2403, sampling/historical.rs:1665, par/lag_transition.rs:1141) folds into the same shared preset behind operational_start_date and mirror_unit_group parameters, or stays a separate da |
| test-bloat-20 | Owner must pick ONE destination for this crate's duplicated entity presets: crates/cobre-stochastic/src/test_support.rs (crate-local, reachable from inline unit tests with no feature change) or cobre_core::test_support beside the existing HydroSpec and StageSpec builders (serves other crates but wid |
| test-bloat-20 | The registered TD-028 claim describes the external/historical make_hydro pair as 'one-cosmetic-line' and calls mod.rs's copy a '2024 plus mirror-unit-group' divergent variant; neither holds at this pin (the pair is byte-identical, and src/sampling/mod.rs:893 is the undated short variant). Owner deci |
| test-bloat-22 | Which home the folded generator takes: crates/cobre-stochastic/src/test_support.rs, already gated at src/lib.rs:32 on cfg(any(test, feature = test-support)) and therefore reachable by downstream crates, or a private cfg(test) helper under src/par/fitting/ that stays intra-crate. The pick decides whe |
| test-bloat-22 | A third copy of the same generator exists outside this candidate's declared crate scope at crates/cobre-io/src/scenarios/estimation/tests.rs:1172, identical to the cobre-stochastic estimation copy except its doc-comment first line. cobre-io already depends on cobre-stochastic at crates/cobre-io/Carg |
| test-bloat-24 | Home for the two new statistical assertion helpers: `cobre-stochastic`'s `tests/common/mod.rs` (crate-local, neutral, three consumers) or `cobre_core::test_support` beside the generic `norm_cdf` and `approx_erf` it already owns (`crates/cobre-core/src/test_support.rs:657-666`), which would make the  |
| test-bloat-25 | Ratify or reject the section 5.1 homing threshold (roughly 500 test-LOC or 40 test fns). Until it is ratified no candidate can name which inline tails must move, and if it is ratified the files it selects in cobre-stochastic are sampling/external.rs, sampling/historical.rs, sampling/mod.rs and tree/ |
| test-bloat-26 | TD-033 stays registered upstream and is not re-litigated here: the owner decides whether its recorded zero-behavioral-coverage basis survives the caller-tier literals at crates/cobre-stochastic/src/seeds.rs:195 and crates/cobre-stochastic/src/seeds.rs:333, which pin the same k walk through the same  |
| test-bloat-27 | The INVERSE of this finding is real and is an owner call: crates/cobre-stochastic/src/test_support.rs is a feature-gated public module that the canonical gate excludes on the basename test heuristic (scripts/ci/check-infra-genericity.sh:98-100), and only the one-token in-binary grep binds there, so  |
| test-bloat-27 | Carried from the attacker: whether the two private one-token gates (crates/cobre-stochastic/tests/reproducibility.rs:195 and crates/cobre-io/tests/genericity_gate.rs:6) keep their unexcluded whole-src scope or fold onto the canonical runner is a cross-crate owner pick, and it can only become a cover |
| test-bloat-28 | Count parity: the shared-body fold above is count-neutral, but TD-036's own direction retires the inline copies, which changes the cargo nextest list count by the number of folded duplicates; whether the ratified duplicate-deletion carve-out extends to body-identical inline copies of a public-contra |
| test-bloat-28 | Reconciling the one same-name divergent pair, test_local_broadcast_root0_noop, by widening the inline fixture (src/local.rs:444, two elements against a literal) to the integration one (tests/local_conformance.rs:94, three elements against a captured copy) removes no assertion but does change what th |
| test-bloat-28 | Sequencing with the cobre-comm test-support surface: crates/cobre-comm/Cargo.toml declares only mpi, numa and shared-memory features and no [dev-dependencies] at the pin, so this fold's shared body would land on the same new test-support surface test-bloat-30 adjudicates, and a self dev-dependency e |
| test-bloat-29 | Does the ratified duplicate-deletion carve-out recorded in lens-rules.md Rule 2 Boundary (TD-011, TD-015, TD-051, TD-067, TD-068) reach compile-time assertion duplicates, so TD-035's removal half may execute as written against src/factory.rs:420 and tests/factory_tests.rs:90, or must that half be re |
| test-bloat-29 | Because pub trait Communicator: Send + Sync at crates/cobre-comm/src/traits.rs:90 makes assert_communicator::<FerrompiBackend>() at tests/factory_tests.rs:97 already entail the bound its same-file sibling at :90 asserts, that copy is redundant to a test six lines below it and not only to the inline  |
| test-bloat-30 | The section 5.2 Adoption paragraph at docs/design/testing-architecture.md:425-431 lists only two bullets as not yet carried out and omits the bullet-1 placements (StubComm/Rank0Of2 to cobre-comm, generic scaffolding to cobre-core); whether that omission is itself prose-drift is row ta-5.2-adoption-u |
| test-bloat-31 | How far the penalties fold reaches: the widened constructor could be pub(crate) for just these src homes, or pub and adopted by the roughly twenty further byte-identical all-zero penalty literals elsewhere in the crate (tests/common/builders.rs:207, tests/anticipated_core.rs and nine siblings). The  |
| test-bloat-32 | The TD-042 handover carries two incompatible fix shapes for the same entry: its registered claim states the residue adds no constructor to cobre-sddp's test_support and spreads the base at the call sites, while its upstreamFixShape promotes crates/cobre-sddp/tests/common/builders.rs:190 to a pub fn  |
| test-bloat-32 | Whether the 7 carriers equal to the HydroSpec::default() penalties at crates/cobre-core/src/test_support.rs:219-221 should adopt cobre_core::test_support::make_hydro itself, which cobre-sddp imports in 0 files at the pin, is an adoption decision that widens the change past the penalties literal into |
| test-bloat-32 | The third bullet of the cost-discipline rule states that the one-place property is scoped to the integration-test layer and does not hold workspace-wide for inline cfg(test) literals. Whether the 30 src-side carriers are in scope for the same de-duplication as the 26 integration-layer ones is an own |
| test-bloat-33 | Destination of the folded constructors: this verdict names the currently wired tests/common/ home, since src/test_support.rs holds no neutral resolved-layer constructor and its existing ones are private. Whether the fold lands there or waits for the section 5.2 collapse into the crate's exported tes |
| test-bloat-34 | Scope of the shared home beyond the anticipated pair: crates/cobre-sddp/tests/filling_commissioning.rs carries one further copy in each of the same three classes (:1532 in the minority default_hydro_bounds digest, :1541 in the minority default_hydro_block_bounds digest, :1549 in the single default_h |
| test-bloat-35 | TD-045's fix-shape wording says the local doc comment's two stated reasons are both false at the pin; at the baseline both check out (the named symbol exists at crates/cobre-sddp/src/lp/builder/test_support.rs:65 and an external test binary genuinely cannot see a bare #[cfg(test)] item), so the owne |
| test-bloat-36 | Fix-shape fork this defense surfaced and did not settle: the production rustdoc at noise.rs:211 links the oracle by bracketed intra-doc link, so the owner picks between moving the oracle into mod tests and demoting that link to plain backticks, or narrowing the residue to the pub(crate) modifier alo |
| test-bloat-39 | Whether the anticipated-hydro pair, named as a collapsible family in TD-049's ratified direction but shown here to differ on inflow_lags and block layout, is dropped from that direction or re-filed as its own item: removing a family from a ratified direction is an owner call. |
| test-bloat-40 | TD-050's upstream fix shape names crates/cobre-sddp/tests/common/parity_hash.rs:230 as a caller of the new helper while the same entry's claim concedes that site as out of scope; whether the golden label mapper rebases onto the shared base-path helper touches the sanctioned golden parity_hash roster |
| test-bloat-41 | Folding the weaker count-mismatch probe into its retained sibling drops the cargo nextest list --features test-support count by one; lens-rules.md Rule 2 'Boundary' leaves that carve-out for the ratified duplicate class (TD-051) to the E08 owner gate, not to this verdict. |
| test-bloat-41 | Where the shared 47-line two-rank echo stub is homed: crates/cobre-sddp/tests/common/mod.rs (this verdict's shape, neutral under Part IV/V) versus cobre-comm's test-support surface named by the section 5.8 target at docs/design/testing-architecture.md:524-526, which is one of the station's adjudicat |
| test-bloat-43 | Destination for a single shared rank-shape Communicator double remains an owner call: testing-architecture.md:415-416 and :524-526 assign StubComm/Rank0Of2 to cobre-comm's test-support surface, but cobre-comm declares no such feature at the pin, while cobre-sddp/src/lib.rs:46-47 already gates one. T |
| test-bloat-44 | Whether the ninth home (crates/cobre-sddp/src/simulation/pipeline/tests.rs:160) is inside the ratified TD-054 scope at all: it is a counting/recording double like the four recording doubles the entry excludes by name, so the nine-site blast radius is a fact while the nine-site fold is an owner scope |
| test-bloat-45 | Severity: TD-055 is ratified at the lowest rank and the attacker escalates to the top rank. After this read the escalation rests on one stale roster entry out of 46 plus a guard whose substantive content is asserted elsewhere in the same binary, so the owner decides whether that still meets the top- |
| test-bloat-45 | Whether a zero-assertion test fn may be removed outright rather than strengthened: the upstream direction deletes k1_byte_stability_verdict, which carries no coverage, while Rule 2 forbids deletion as a fix-shape and its Boundary paragraph reserves such carve-outs to the E08 owner gate. This verdict |
| test-bloat-46 | TD-056 registers the rank field of the local stub as a dead-field defect; at the pin crates/cobre-sddp/src/simulation/state.rs:409 reads comm.rank() and passes it to assign_scenarios at :412, so rank is a consumed input fixed at 0, not a dead field. Correcting the wording of a ratified entry is an o |
| test-bloat-46 | Where a shared parameterized stub communicator would live (the sddp test harness at tests/common/mod.rs versus the communication crate's exported test-support surface) remains the station's adjudicate row ta-5.8-stubcomm-home. TD-056's registered fix shape still needs that destination and this dismi |
| test-bloat-47 | Whether cobre-cli may gain a crate-local tests/common helper module now, or whether its shared harness home must wait for the section 5.2 collapse of tests/common into the test-support surface so the crate does not build a convention that collapse would undo: the residue's shape is settled here, its |
| test-bloat-47 | Whether the cobre-core test-support activation at crates/cobre-cli/Cargo.toml:62 is a reserved seam or a stale dev-dependency feature activation: git grep for test_support over crates/cobre-cli returns nothing at the pin and no cobre-cli test file references cobre_core at all, so the shared helper h |
| test-bloat-48 | Whether an intra-crate module re-home counts as a 'rename' under the section 5.1 migration invariant at docs/design/testing-architecture.md:351-354: the two function names and the total count are unchanged, but the fully qualified listed ids move from commands::run::tests::* to commands::run::setup: |
| test-bloat-49 | TD-063's upstream fix-shape also folds the five format_report_* cases into one table-driven test, which lowers the cargo nextest list count; this verdict confirms only the count-neutral repoint and leaves that fold to the owner-gate carve-out recorded in lens-rules.md Rule 2 Boundary. |
| test-bloat-49 | After the repoint, format_report_summary_header_present cannot keep asserting both counts nonzero (no shipped path emits it); whether it is split into the two reachable header cases, which raises the listed count by one, is an owner call under Rule 2's same-count question. |
| test-bloat-50 | Whether the eight panic-only smoke tests over the four printers may be retired once the content assertions land on the shipped path: the upstream fix shape collapses the eleven does_not_panic tests, which lowers the cargo nextest list count, so lens-rules.md Rule 2 Boundary makes that an owner-gate  |
| test-bloat-50 | Whether the re-homed assertions may be strengthened from contains() to a whole-render equality over the returned lines, which is the only shape that would pin the set and order of lines, since that is additive coverage beyond re-homing and would make any future deliberate wording change fail the tes |
| test-bloat-51 | Feature-unification caveat on the recommended shape: cobre-cli takes cobre-io as a normal dependency at crates/cobre-cli/Cargo.toml:22 (schema feature) and as a dev-dependency at :63, so one cobre-io instance is shared during a test build and the cobre binary built by cargo test would compile the te |
| test-bloat-51 | Sharpens the attacker's third item: whether cobre-cli's shared fixture home is the existing cobre-io test-support surface now or must wait for the section 5.2 collapse. My read shows the surface already carries seven of the eight consts plus write_file plus the valid-case writer, so the objection ab |
| test-bloat-51 | Whether the deliberate-duplication comment at crates/cobre-cli/tests/cli_color.rs:20-21 is overruled. It states the run-versus-colour fixture duplication is an intentional self-containment choice, so folding that pair retires a recorded decision and needs the owner, not the defender, to withdraw it. |
| test-bloat-51 | Whether the cobre-core test-support activation at crates/cobre-cli/Cargo.toml:62 is a reserved seam for planned CLI fixture use or a stale activation: no file under crates/cobre-cli/tests/ references cobre_core at the pin, so it has zero consumers, but CLAUDE.md Hard Rules reserve unwired config, an |
| test-bloat-52 | Whether an escape-absence assertion may stand as the --color never integration probe at all: cli_color.rs:130-131 records that auto-detection disables colour on a piped stderr, so the absence proves nothing about the flag unless the child environment forces colour on. The owner decides whether the r |
| test-bloat-52 | Whether the colour suite's flag-to-global wiring assertion may be re-homed onto the solve-free init subcommand path, which changes which subcommand owns the assertion while keeping the predicate identical (carried from the attacker). |
| test-bloat-52 | Whether TD-066's ratified removal half (dropping the two remaining run-based colour cases because banner.rs:70-73 and main.rs:127-132 pin both claims at the unit tier) is admitted under the lens-rules Rule 2 Boundary carve-out. It is declined in this verdict as not coverage-neutral, and admitting it |
| test-bloat-53 | The consolidation fold removes exact-duplicate assertions and therefore changes the cargo nextest list count; lens-rules.md section Rule 2 Boundary records that carve-out for the ratified duplication class (TD-067 here, and the sibling routed seed) as an owner-gate decision, not a lens decision. |
| test-bloat-53 | Whether the accumulator mean and sample-standard-deviation assertions at crates/cobre-cli/src/progress.rs:987-1007 may be re-homed to the crate that declares WelfordAccumulator, since moving a test across a crate boundary shifts which crate's test count carries it. |
| test-bloat-54 | The fold changes the cargo nextest list count: by four under TD-068's registered direction, or by five once the newly verified domination of missing_buses_json_stdout_contains_error (crates/cobre-cli/tests/cli_validate.rs:131-142) by validate_failure_report_in_stdout_not_stderr (:207-220) is folded  |
| test-bloat-54 | Whether the exit-code-only layer is a wanted coarse-to-fine diagnostic split: an assert_cmd Assert chain stops at the first failing predicate, so a merged test reports the exit-code failure and the message failure one at a time, while the present split gives an independent pass or fail signal for th |
| test-bloat-54 | Whether cobre-cli may gain a tests/common helper module now, or whether its shared harness home must wait for the section 5.2 collapse of tests/common into the test-support surface. That decision gates the count-neutral alternative shape (one shared fixture builder), which is the only variant availa |
| test-bloat-55 | TD-069's upstream fix-shape also deletes test_1dtoy_files_have_descriptions (templates.rs:183-188) and test_1dtoy_files_have_relative_paths (:191-197); lens-rules.md Rule 2 forbids deletion as a fix-shape at this station, so whether that ratified removal is admitted under the duplicate-class carve-o |
| test-bloat-55 | Whether the duplicate count pin at scripts/ci/check-docs-examples.sh:35 (EXPECTED_INPUT_FILES=11) is retired once the unit test derives the set: that script's header at :5-7 documents its constants as deliberate bump-on-change and calls itself the single source of truth for the count, so touching a  |
| test-bloat-56 | Whether the surviving definition stays a named private helper or the call sites adopt the inline standard-formatter form already used at summary.rs:858: the body is a no-op re-derivation of that formatter, so the choice decides whether the added table-driven test has a private symbol to target or wh |
| test-bloat-57 | Owner picks between the two coverage-neutral resolutions of crates/cobre-cli/Cargo.toml:62 - drop the unconsumed cobre-core/test-support activation, or keep it with an explicit reservation note beside the crate's slow-tests declaration; the tree gives no evidence either way, since the activation nev |
| test-bloat-58 | TD-058's other half (test_run.py:38 asserting a subset of test_outputs.py:33-50 on the shared 1dtoy deck) falls outside this candidate's scope; the owner decides whether a sibling candidate carries it or calibration folds it into this entry, since removing that test would change the collected pytest |
| test-bloat-58 | Carried from the attacker: whether the shared fixture belongs in test_study.py as a module fixture (matching test_outputs.py:24-31, and the only trained-policy consumers sit in that one module) or in conftest.py as TD-058's registered fix-shape words it. The two differ in blast radius, not in covera |
| test-bloat-59 | This verdict replaces a ratified seed's direction rather than forwarding a conflict: TD-059's fix-shape drops the hand-copied CLI golden literals, which weakens a retained test's only assertions, so the confirmation keeps them and folds the solve instead. The owner must confirm that re-shaping a rat |
| test-bloat-59 | A once-initialised shared fixture in a LazyLock static never drops, so the owner must decide whether the fold may hold the shared directory in a counted guard, or whether the Contracts bullet that test output goes to TempDir and generated artifacts self-delete forbids any once-init holder that outli |
| test-bloat-60 | TD-060's ratified fix-shape directs one Hive-partitioned parquet collector parameterised by entity directory; this verdict withdraws that clause because the empty-tolerant collector's return is the asserted value at test_anticipated_lanes_output_parity.py:236 and :243, so parameterising it would mov |
| test-bloat-60 | The fold replaces four divergent builder docstrings with one contract statement plus per-call-site notes. Each current docstring names a different case-specific reason (D41 is training-only; the anticipated cases ship simulation disabled; several deterministic cases do). Owner must decide whether th |
| test-bloat-61 | Carried from the attacker: TD-061 was ratified with a direction that deletes test_run_via_study_emits_contract_output. This verdict refuses that half as inadmissible under cost discipline and confirms only the fixture fold. The owner must ratify the re-shape of a ratified seed's direction rather tha |
| test-bloat-61 | Not adjudicated here: the seed also observed that cobre.run.run solves the same deck twice (:224 and :264). Sharing one module-scoped python-output fixture (the d02_python_output shape) would remove one full train-and-simulate, but that broadens past this candidate's deck-only claim. The owner must  |
| test-bloat-62 | The shape is additive, so the crate's cargo nextest list count rises rather than staying equal; the owner must say whether section 6 count parity and lens-rules Rule 2's count test admit added coverage or need an explicit allowance for it. |
| test-bloat-62 | Calibration must choose which figure it records for perCrate.cobre-python.inlineCfgTestModules: 5 under the inventory's file census, or 4 files that actually declare a module, since convert.rs:104 and errors.rs:209 are prose matches and not declarations. |
| test-bloat-63 | Owner call: closing the second borrow site needs an import-line edit inside test_parity_filling_sigma.py, but docs/design/testing-architecture.md:563 (section 5.11) directs that the pytest output-parity suite be kept unchanged and TD-072s fix shape freezes all fifteen parity modules on that basis. D |
| test-bloat-64 | Owner call on fixture isolation: routing test_load_stochastic_missing_artifacts_raises onto the shared module-scoped run_output tree makes an assertion about an ABSENT artifact depend on no sibling test ever enabling exports.stochastic into that shared tree, where the per-test tmp_path at :342 guara |
| test-bloat-64 | Wording call for calibration: this station's count-neutrality bar is stated as cargo nextest list parity, which never lists the cobre-python pytest suite, so neutrality for this candidate can only be checked with a pytest collection count. Confirm the entry may state the bar in pytest terms for this |
| test-bloat-66 | Whether the doc-integrity default of no frozen count without a guard extends to Python docstrings and Rust doc comments is an owner call; it decides whether this fix-shape generalizes to sibling prose or stays a single-clause correction in this module. |
| test-bloat-67 | Cadence policy, already owner-gated by the yardstick (§5.6 should be ratified explicitly; §6 phase 4 Ratify the tradeoff first): whether to split NON_SOLVER_FEATURES into a per-PR set without slow-tests plus a full set on push-to-develop or nightly, at the cost of a second feature-combo build and sl |
| test-bloat-68 | Owner call at the 5.5 adoption rather than at this station: the paired cargo test --doc step must land in the same change as the nextest switch because .github/workflows/ci.yml:114 is the only doctest execution at the pin, and the owner should state whether the primary job stays on cargo's default f |
| test-bloat-69 | Sequencing only, no defect: if the owner ratifies section 6 phase 2 (commit a .config/nextest.toml and move the fail-fast, slow-timeout and retry knobs off the three inline run lines), decide whether that profile lands before or after CD-101's edit to the same two invariance-shuffle.yml run lines, s |
| test-bloat-70 | If the owner ratifies the §5.5 runner adoption, the paired cargo test --doc step must land in the same change: nextest runs no doctests, so unifying the runner without that step removes the 329-passing doctest tier from CI. Carried over from the attacker's runner-unification item because it decides  |
| test-bloat-70 | Whether the two CLP-gated doctest bodies (crates/cobre-solver/src/backends/clp/solver.rs:37 and :520) should ever be compiled under the CLP feature combination is an owner call on doctest-tier scope: their assertions are already covered by CLP-combination unit tests at crates/cobre-solver/src/backen |
| test-bloat-71 | Trigger pick for full mode is an owner call among push to the release branch, tags only, and a nightly schedule; it spends the same cluster-minutes budget as CD-101's shuffle-cadence needs-human (BACKLOG.md:6500), so the two should be decided together rather than separately. |
| test-bloat-71 | Whether the reduced path's cluster build and health-wait at mpi-slurm.yml:107-108 is intended as a cluster-boot smoke check on every automatic run; the answer decides whether trimming the provisioning on reduced triggers is admissible at all or whether it would remove the only coverage that path has |
| test-bloat-72 | Trigger shape: enumerate the omitted module or widen both lists to the whole solver crate source tree. Enumerating keeps the cluster job rare but rots the next time a module moves, which is the failure mode that produced this gap; widening is durable but spends cluster minutes on far more events. Th |
| test-bloat-72 | Ordering against the 5.6 / 5.8 cadence targets: if the owner ratifies moving this deck off the pull-request tier, only the push list needs the added pattern and the pull_request block changes shape anyway, so the owner decides whether the filter fix lands before or after that cadence decision so the |
| architecture-00 | Whether cobre-sddp gains the self dev-dependency now or command-line activation is accepted until the 5.2 collapse lands (attacker item 1, still live): the self dev-dependency makes the five feature-gated items in crates/cobre-sddp/tests/deterministic.rs:8708, :10426, :10667, crates/cobre-sddp/tests |
| architecture-00 | Settling claim-class row ta-5.2-adoption-uniform on the doc side: whether the Adoption paragraph's uncarried-bullet list at docs/design/testing-architecture.md:427-430 is corrected to include bullet 1's two placements (generic scaffolding to cobre-core, StubComm/Rank0Of2 to cobre-comm) or the paragr |
| architecture-00 | Whether cobre-solver's narrow gate (crates/cobre-solver/src/lib.rs:163, crates/cobre-solver/src/backends/highs/solver.rs:6 and :580) is recorded in 5.2 as a sanctioned exception to the uniform gate (attacker item 3, still live): the rationale is documented at solver.rs:575-579 and lib.rs:165-171 and |
| architecture-01 | Carried from the attacker and narrowed by this read: whether the §5.2 plan text should still name crates/cobre-io/tests/helpers/ and crates/cobre-stochastic/tests/common/ as collapse sources even though both already re-export their crate's test_support surface, or whether the two shims stay unnamed  |
| architecture-01 | Recording detail for the ingest owner: the §2.2 sentence spans :76 and :78 because a line-final hyphen renders 'behavioral only, constructing fixtures per-file' as a stray list item. Whether that render break is a second, separable prose defect or part of this one changes how the item is filed, not  |
| architecture-02 | Whether the §5.2 collapse source list (docs/design/testing-architecture.md:417-421, adjudicate row ta-5.2-adoption-uniform at :425-431) extends to crates/cobre-io/tests/helpers/ and crates/cobre-stochastic/tests/common/, which decides whether this residue is a standalone rename now or is absorbed by |
| over-engineering-00 | Scope call beyond this candidate: `eq` at crates/cobre-sddp/src/test_support.rs:152-171 has exactly one caller at the pin (crates/cobre-sddp/tests/basis_trajectory_probe.rs:282), so retiring only its zero-caller sibling leaves a one-consumer positional helper behind — the owner decides whether the a |
| over-engineering-01 | The module doc at crates/cobre-sddp/tests/common/anticipated_structural_assertions.rs:1 calls the helpers shared by the anticipated-thermals integration tests, plural, while at the pin each of its three public helpers has exactly one call site, in one test function of crates/cobre-sddp/tests/anticip |
| performance-00 | Cadence boundary: does a pull request keep all three link-bearing jobs (Test at .github/workflows/ci.yml:114, CLP at :223 and :233, Coverage at :497), or does one backend gate pull requests while the second backend's test run and the instrumented pass move to a merge-queue or scheduled tier? Section |
| performance-00 | Target selection inside the CLP job: the build at .github/workflows/ci.yml:223 passes --all-targets, which also builds the criterion bench targets declared at crates/cobre-sddp/Cargo.toml:92-110; whether the CLP job narrows its target selection to the test surface is an owner call on the same anchor |
| performance-01 | Owner call: is the #[path] submodule shape already working at crates/cobre-sddp/tests/template_integration.rs:4400-4420 the ratified Layer 1 shape, and what sets the grouping boundary - docs/design/testing-architecture.md:327-328 leaves exact membership a starting proposal refined during migration. |
| performance-02 | Cadence only, not shape: does the sole codegen-and-link gate over the bench and example targets (the build step at .github/workflows/ci.yml:223) stay on the pull-request tier, or move to merge-to-develop or nightly with the rest of the link-bearing passes? Section 5.6 of the yardstick marks cadence  |
| performance-02 | Owner note, not a finding: if the linked-executable count of the criterion targets is ever revisited, it needs its own entry, because this candidate records the bench declarations as sanctioned (.claude/rules/testing.md:12, docs/design/testing-architecture.md:246) and scopes itself away from them. |

## Per-candidate roster

Every one of the 81 candidateRefs, exactly once (mirrors `verdicts.json`).

| candidateRef | disposition | verdict | state | claimKind | targetNotDefect | definition | alignment | seed | NH |
|---|---|---|---|---|---|---|---|---|---|
| test-bloat-00 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[1] | yes |
| test-bloat-01 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[0] | yes |
| test-bloat-02 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[2] | yes |
| test-bloat-03 | defended | dismissed (target-not-defect) | rejected-defender | target-gap | True | file | neutral | - | yes |
| test-bloat-04 | dup-of | - | dup-of | - | - | n/a | - | - |  |
| test-bloat-05 | defended | confirmed | accepted | tree-fact | False | file | neutral | queue[20] | yes |
| test-bloat-06 | defended | dismissed (premise-false-at-pin) | rejected-defender | tree-fact | False | file | neutral | queue[7] | yes |
| test-bloat-07 | defended | confirmed | accepted | tree-fact | False | binary | neutral | queue[9] | yes |
| test-bloat-08 | defended | dismissed (premise-false-at-pin) | rejected-defender | tree-fact | False | file | neutral | queue[19] | yes |
| test-bloat-09 | defended | confirmed | accepted | tree-fact | False | file | neutral | queue[12] | yes |
| test-bloat-10 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[21] | yes |
| test-bloat-11 | defended | confirmed | accepted | tree-fact | False | file | neutral | queue[15] | yes |
| test-bloat-12 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[5] | yes |
| test-bloat-13 | defended | confirmed | accepted | tree-fact | False | binary | neutral | queue[2] | yes |
| test-bloat-14 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[3] | yes |
| test-bloat-15 | defended | confirmed | accepted | tree-fact | False | binary | neutral | queue[4] | yes |
| test-bloat-16 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[5] | yes |
| test-bloat-17 | defended | dismissed (premise-false-at-pin) | rejected-defender | tree-fact | False | binary | neutral | - |  |
| test-bloat-18 | defended | dismissed (target-not-defect) | rejected-defender | target-gap | True | file | neutral | - | yes |
| test-bloat-19 | defended | confirmed | accepted | tree-fact | False | file | neutral | queue[8] | yes |
| test-bloat-20 | defended | confirmed | accepted | tree-fact | False | file | neutral | queue[4] | yes |
| test-bloat-21 | defended | confirmed | accepted | tree-fact | False | file | neutral | queue[1] |  |
| test-bloat-22 | defended | confirmed | accepted | tree-fact | False | file | neutral | queue[2] | yes |
| test-bloat-23 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[3] |  |
| test-bloat-24 | defended | confirmed | accepted | tree-fact | False | binary | neutral | queue[6] | yes |
| test-bloat-25 | defended | dismissed (target-not-defect) | rejected-defender | target-gap | True | file | neutral | queue[7] | yes |
| test-bloat-26 | defended | dismissed (premise-false-at-pin) | rejected-defender | tree-fact | False | n/a | neutral | queue[9] | yes |
| test-bloat-27 | defended | dismissed (premise-false-at-pin) | rejected-defender | tree-fact | False | n/a | neutral | - | yes |
| test-bloat-28 | defended | confirmed | accepted | tree-fact | False | binary | neutral | queue[1] | yes |
| test-bloat-29 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[0] | yes |
| test-bloat-30 | defended | dismissed (target-not-defect) | proposal-adjudicated | target-gap | True | file | neutral | - | yes |
| test-bloat-31 | defended | confirmed | accepted | tree-fact | False | file | neutral | queue[0] | yes |
| test-bloat-32 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[1] | yes |
| test-bloat-33 | defended | confirmed | accepted | tree-fact | False | binary | neutral | queue[2] | yes |
| test-bloat-34 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[3] | yes |
| test-bloat-35 | defended | dismissed (premise-false-at-pin) | rejected-defender | tree-fact | False | binary | neutral | queue[4] | yes |
| test-bloat-36 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[5] | yes |
| test-bloat-37 | dup-of | - | dup-of | - | - | file | - | queue[6] |  |
| test-bloat-38 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[7] |  |
| test-bloat-39 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[8] | yes |
| test-bloat-40 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[9] | yes |
| test-bloat-41 | defended | confirmed | accepted | tree-fact | False | binary | neutral | queue[10] | yes |
| test-bloat-42 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[11] |  |
| test-bloat-43 | defended | dismissed (target-not-defect) | proposal-adjudicated | target-gap | True | binary | neutral | queue[12] | yes |
| test-bloat-44 | defended | confirmed | accepted | tree-fact | False | file | neutral | queue[13] | yes |
| test-bloat-45 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[14] | yes |
| test-bloat-46 | defended | dismissed (premise-false-at-pin) | rejected-defender | tree-fact | False | n/a | neutral | queue[15] | yes |
| test-bloat-47 | defended | confirmed | accepted | tree-fact | False | binary | neutral | queue[0] | yes |
| test-bloat-48 | defended | confirmed | accepted | tree-fact | False | file | neutral | queue[5] | yes |
| test-bloat-49 | defended | confirmed | accepted | tree-fact | False | file | neutral | queue[6] | yes |
| test-bloat-50 | defended | confirmed | accepted | tree-fact | False | file | neutral | queue[7] | yes |
| test-bloat-51 | defended | confirmed | accepted | tree-fact | False | binary | neutral | queue[8] | yes |
| test-bloat-52 | defended | confirmed | accepted | tree-fact | False | binary | neutral | queue[9] | yes |
| test-bloat-53 | defended | confirmed | accepted | tree-fact | False | file | neutral | queue[10] | yes |
| test-bloat-54 | defended | confirmed | accepted | tree-fact | False | binary | neutral | queue[11] | yes |
| test-bloat-55 | defended | confirmed | accepted | tree-fact | False | file | neutral | queue[12] | yes |
| test-bloat-56 | defended | confirmed | accepted | tree-fact | False | file | neutral | queue[13] | yes |
| test-bloat-57 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | - | yes |
| test-bloat-58 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[1] | yes |
| test-bloat-59 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[2] | yes |
| test-bloat-60 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[3] | yes |
| test-bloat-61 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[4] | yes |
| test-bloat-62 | defended | confirmed | accepted | tree-fact | False | file | neutral | queue[14] | yes |
| test-bloat-63 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[15] | yes |
| test-bloat-64 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | queue[16] | yes |
| test-bloat-65 | dup-of | - | dup-of | - | - | binary | - | - |  |
| test-bloat-66 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | - | yes |
| test-bloat-67 | defended | dismissed (deliberate-and-documented) | rejected-defender | tree-fact | False | file | neutral | - | yes |
| test-bloat-68 | defended | dismissed (target-not-defect) | rejected-defender | target-gap | True | file | neutral | - | yes |
| test-bloat-69 | defended | dismissed (target-not-defect) | rejected-defender | target-gap | True | file | neutral | - | yes |
| test-bloat-70 | defended | dismissed (target-not-defect) | rejected-defender | target-gap | True | file | neutral | - | yes |
| test-bloat-71 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | - | yes |
| test-bloat-72 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | - | yes |
| architecture-00 | defended | confirmed | proposal-adjudicated | prose-drift | False | binary | neutral | - | yes |
| architecture-01 | defended | confirmed | accepted | prose-drift | False | file | neutral | - | yes |
| architecture-02 | defended | confirmed | accepted | tree-fact | False | file | neutral | - | yes |
| over-engineering-00 | defended | confirmed | accepted | tree-fact | False | n/a | neutral | - | yes |
| over-engineering-01 | defended | dismissed (deliberate-and-documented) | rejected-defender | tree-fact | False | binary | neutral | - | yes |
| performance-00 | defended | dismissed (cost-accepted-by-rule) | rejected-defender | tree-fact | False | binary | neutral | - | yes |
| performance-01 | defended | dismissed (target-not-defect) | rejected-defender | target-gap | True | binary | neutral | - | yes |
| performance-02 | defended | dismissed (deliberate-and-documented) | rejected-defender | tree-fact | False | file | neutral | - | yes |

## Reconciliation

Counts by state: {"accepted": 57, "rejected-defender": 18, "dup-of": 3, "proposal-adjudicated": 3}. Received 81; verdicts.json keys 81; every received candidateRef accounted for exactly once; the counts block sums (0 + 0 + 0 + 3 + 0 + 78 = 81). Read-only check: `git status --porcelain -- crates .github docs schemas scripts Cargo.toml Cargo.lock` → empty; the station's writes sit under `stations/test-corpus/`, which is TRACKED in this repository (the ticket's 'untracked files under plans/' and 'carried-in .gitignore' premises are stale and recorded as a deviation). No TD id was assigned and no BACKLOG section was written.
