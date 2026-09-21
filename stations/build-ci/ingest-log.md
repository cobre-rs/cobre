# build-ci ingest log (candidates → verdicts)

Baseline: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`   Station: `build-ci` (label `build-ci-docs`)   Ingested: 2026-09-19

Inputs: the four merged lens files `candidates-{architecture,over-engineering,drift,performance}.json`
(gated in the attacker ticket; `attacker-log.md` is the attacker-side record), `gate-census.json` (the wiring
class of all 17 gate files), `prior-register.md` (CD-061 anchored here; the do-not-touch list; three inbound
handoffs; four recorded observations), `inventory.json`, `tools/target-layering-brief.md`, the committed
mirror `docs/design/reserved-seams-and-deferred-debt.md` at the pin, `CLAUDE.md` / `ARCHITECTURE.md` /
`.claude/rules/doc-integrity.md` at the pin, and `plans/generalizing/beyond-sddp-generalization.md` (worktree;
the roadmap file is not tracked at the pin). Every figure below is measured at `077dbe2c` (`git show`), never on
HEAD. Scratch: `/tmp/build-ci/ingest.py` (`screens`, `validate`, `assemble`), defender inputs
`/tmp/build-ci-defenders/in/<candidateRef>.json`, envelopes `/tmp/build-ci-defenders/out/<candidateRef>.json`.

Attacker envelopes re-validated at ingest: candidates-architecture.json: exit 0; candidates-over-engineering.json: exit 0; candidates-drift.json: exit 0; candidates-performance.json: exit 0.

## Candidate census

| lens | candidates | positives | needs-human (attacker) |
|---|---|---|---|
| architecture | 6 | 7 | 5 |
| over-engineering | 6 | 8 | 4 |
| drift | 10 | 10 | 3 |
| performance | 7 | 6 | 4 |
| **total** | 29 | 31 | 16 |

Every lens returned at least one candidate; no no-finding line applies (the performance lens is informational for this station and may legitimately be empty — it was not).

29 candidates received; candidateRef `<lens>-<nn>` with nn the zero-based index among the lens
file's candidates in file order (the scheme every downstream artifact keys on).

## Anchor screen (non-Rust anchor form)

Every anchor of every candidate — plus the paired `docClaim` / `treeFact` evidence and the `twoSite` sites — was
rendered into `anchor-probe.md` (register-shaped, one block per candidate, `CD-9nn · probe · <ref>` headings) and
resolved with `tools/check-anchors.py "INGEST ANCHOR PROBE — build-ci (2026-09, baseline)" --register anchor-probe.md --baseline 077dbe2c`: exit
0. Shape guard applied first: `{path, symbol}` is admissible only on
`crates/cobre-solver/build.rs`, `crates/cobre-sddp/build.rs`, `crates/cobre-cli/build.rs` (the checker's symbol
resolver is a Rust-item regex); a symbol anchor on a YAML / shell / JSON / Markdown path is `anchor-missing`, never
a silent downgrade. Symbol anchors seen: `crates/cobre-cli/build.rs::TEMPLATE_FILES` (architecture-04),
`crates/cobre-solver/build.rs::main` (performance-06) — both legal, both resolve. Own line check (1 ≤ line ≤ file
length at the pin) over every `{path, line}` anchor: 0 failures.

### Anchor rejections

| candidateRef | offending anchor(s) | checker diagnostic |
|---|---|---|
| — | — | none: every anchor resolves at the pin and every symbol anchor sits on a build.rs |

## Cleared (sanctioned)

The sanctioned-advisory clearance — this station's analogue of the reserved-seams check — ran over every candidate
before any defender was dispatched. Three sanctioned classes clear without a defender, each with a path-and-line
citation of the sanctioning text (measured at the pin; the ticket's line figures are a136840d-era):

- **advisory-by-design gates** (`gate-census.json` class `advisory-by-design`): the header comment proving `exit 0`
  is intentional is `# Exit code: ALWAYS 0 (advisory — never fails the build).` at `scripts/ci/check-comment-line-refs.sh:35`
  (ticket :34), `scripts/ci/check-comment-banners.sh:40`, `scripts/ci/check-comment-bloat.sh:39`, and
  `# Exit code: ALWAYS 0 (advisory — a report, never a gate).` at `scripts/ci/quality-report.sh:42`; the `ci.yml`
  step comments repeat it at `:278-279` (line refs), `:282-283` (banners) and `:298-299` (hotspot report); terminal
  `exit 0` at `:157`, `:174`, `:114`, `:134`.
- **the entry-free allowlist** `scripts/ci/allow-rationale-allowlist.txt`: `:1` `# E4 rationale-on-suppression allowlist
  — intentionally EMPTY.`, `:9-13` the burned-down backlog, `:15` `# KEEP THIS FILE EMPTY.`; 0 active entries at the
  pin — emptiness is the designed state, not an unwired gate.
- **the five reserved stub crates** `cobre-mcp`, `cobre-tui`, `cobre-flow`, `cobre-uc`, `cobre-emt` (`Cargo.toml:12-16`):
  `CLAUDE.md:10` names them reserved stubs; `ARCHITECTURE.md:108` `### Reserved crates (not yet implemented)` with the
  bullets at `:115-125`; the roadmap keeps the named algorithm-crate stubs (`plans/generalizing/beyond-sddp-generalization.md:84`)
  and leaves `cobre-flow` / `cobre-emt` deliberately out of scope (`:1236`).

Transitive invocation resolved in the same pass: `check-comment-bloat.sh` has no direct wiring in `.github/workflows/`
or `scripts/pre-commit` (the census negative grep names it) yet runs from the tail of `scripts/ci/quality-report.sh:132`,
which `ci.yml:300` runs in the `quality-scripts` job (`:248`) — its class is `transitively-advisory`. No candidate
claimed it never executes, so nothing needed sharpening; the resolution is recorded so the claim cannot be filed later.

| candidateRef | title | sanctioning citation |
|---|---|---|
| — | — | none cleared: the three sanctioned classes appear only in the attackers' `positives` (allowlist :1/:15, the stubs, the three advisory gates' exit 0); no candidate claims a sanctioned fact is a defect |

Candidates that anchor a sanctioned item but claim something else about it (screen result `touches-sanctioned-anchor`;
the sanction travels to the defender in `sanctionedAdvisoryScreen`, the claim is judged on its own evidence):

| candidateRef | title | sanctioned item touched | screen result |
|---|---|---|---|
| over-engineering-02 | Five comment and doc gates each hand-maintain their own copy of the crate-source scan-directory list | scripts/ci/check-comment-line-refs.sh (advisory-by-design; header `# Exit code: ALWAYS 0` at :35); scripts/ci/check-comment-banners.sh (advisory-by-design; header `# Exit code: ALWAYS 0` at :40) | touches-sanctioned-anchor |
| drift-07 | Four gate headers cite rule-file sections that do not resolve, and a sibling gate sanctions the file | scripts/ci/check-comment-line-refs.sh (advisory-by-design; header `# Exit code: ALWAYS 0` at :35) | touches-sanctioned-anchor |
| performance-04 | Ten of the fourteen quality-scripts steps each enumerate the crates tree themselves, and the shared  | scripts/ci/check-comment-line-refs.sh (advisory-by-design; header `# Exit code: ALWAYS 0` at :35); scripts/ci/quality-report.sh (advisory-by-design; header `# Exit code: ALWAYS 0` at :42) | touches-sanctioned-anchor |

## Prior register, do-not-touch and re-raise screen

`tools/check-reraise.py "INGEST RERAISE PROBE — build-ci (2026-09, baseline)" --register <stub> --baseline 077dbe2c --json` over the 29 candidates
(stub `/tmp/build-ci/reraise-probe.md`, each sharpen carrying its `Re-raise-of:` line): exit 0.
Do-not-touch ids (CD-008, PD-001, PD-004 — BACKLOG.md L1942-1943 at HEAD) named by a candidate: 0.

### Re-raise rejections

| candidateRef | reason |
|---|---|
| — | none |

### Sharpens of a live entry (kept as `reRaiseOf`, judged on the delta)

| candidateRef | prior id | register line / owner | title |
|---|---|---|---|
| architecture-01 | CD-061 | L2545 — core-io gate (ratified 2026-09-08) | The infra-genericity gate matches one engine's vocabulary rather than the engine-concept rule it enf |
| drift-05 | CD-061 | L2545 — core-io gate (ratified 2026-09-08) | The genericity hard rule names three forbidden tokens while the gate it is enforced by rejects eleve |

CD-061 (core-io station, ratified 2026-09-08) records the register/oracle-coverage gap in cobre-io's output column
vocabulary and is anchored on `scripts/ci/check-infra-genericity.sh:79`; `prior-register.md` allows this station to add
the GATE half only. Both sharpens are the gate half (the `PATTERN` vocabulary-vs-concept limitation; the `CLAUDE.md:39-41`
prose naming three tokens against eleven alternatives) — neither restates the column-vocabulary finding, so neither is
a dup-of; calibration folds or mints.

### Related prior items and inbound handoffs (context for the defender, not a merge)

| candidateRef | relatedTo |
|---|---|
| architecture-03 | TD-045 (L4717, cites invariance-shuffle.yml); sddp gate NH9 (R6-nh-9) → E07 |
| architecture-05 | cli-python gate NH8 (R5-nh-8, queue=build-ci); ROADMAP X4 (`doc = false` seed) |
| drift-06 | solver-comm gate R16 (backend-testing.md cite) → drift lens |
| drift-08 | partI-handoff I.3-6 (this station's evidence package for epic 9) |
| drift-09 | sddp gate NH40 (R6-nh-40) → E07/E11 |
| performance-06 | CD-080 (L3886, build.rs:30/:97 link-demand facts) |

## Dup-of merges (cross-lens overlaps)

| candidateRef | merged into | title | why |
|---|---|---|---|
| performance-01 | performance-00 | The python job carries the MPICH Cache/Build/Set triple on every instance of its version matrix alth | cross-lens twin of performance-00 — the same mechanism (the MPICH prelude configured on a job whose feature resolution never enables `mpi`) on a second job; the attacker itself asks that the two be carried as one item |

### Cross-lens overlaps — decisions

| candidates | decision |
|---|---|
| architecture-00 + architecture-01 + drift-05 | kept separate, `relatedCandidates` handed to each defender |
| over-engineering-00 + performance-00 + performance-02 | kept separate, `relatedCandidates` handed to each defender |
| over-engineering-02 + performance-04 | kept separate, `relatedCandidates` handed to each defender |
| drift-00 + drift-01 + drift-02 | kept separate, `relatedCandidates` handed to each defender |
| over-engineering-03 + over-engineering-04 | kept separate, `relatedCandidates` handed to each defender |
| over-engineering-05 + architecture-05 | kept separate, `relatedCandidates` handed to each defender |
| performance-01 + performance-00 | performance-01 merged into performance-00 (dup-of; the defender of performance-00 judges both jobs) |

## Pull-don't-push and two-site screen

Every fix-shape was screened for a NEW gate framework, a new shared scan library, or a composite action with a single
consumer (→ `conflicts`, held for the owner gate with the amend-in-place alternative, never dropped). The MPICH
Cache/Build/Set triple is exempt: consumers measured at the pin below (eight in `ci.yml`, one each in `mpi-slurm.yml`
and `release-mpi.yml`; no `.github/actions/` directory), so a composite action for it is a normal candidate.

| candidateRef | screen result | MPICH consumers (pin) | title |
|---|---|---|---|
| over-engineering-00 | exempt-many-consumers | {"ci.yml": 8, "mpi-slurm.yml": 1, "release-mpi.yml": 1, "total": 10, "githubActionsDir": false} | The MPICH Cache/Build/Set step triple is copied byte-for-byte eight times inside ci.yml, w |
| over-engineering-01 | exempt-many-consumers | {"ci.yml": 8, "mpi-slurm.yml": 1, "release-mpi.yml": 1, "total": 10, "githubActionsDir": false} | The MPICH version literal 4.2.3 has three independent owners in the SLURM image path (work |

Conflicts held for the owner gate: 0.

Two-site rule (a fix-shape touching the infrastructure-genericity hard rule must name the prose at `CLAUDE.md:39-41` AND
`check-infra-genericity.sh`'s `SCAN_DIRS` / `PATTERN` with its `ci.yml:261-262` `quality-scripts` step):

| candidateRef | result | sites resolved at the pin |
|---|---|---|
| architecture-00 | both-sites-named | {"prose": true, "script": true, "step": true} |
| architecture-01 | both-sites-named | {"prose": true, "script": true, "step": true} |
| drift-05 | both-sites-named | {"prose": true, "script": true, "step": true} |

Returned for sharpening: 0.

## Part-I item-6 evidence package

`partI-handoff.json` — `partIRef` I.3-6; the roadmap claim (`plans/generalizing/beyond-sddp-generalization.md:282`, read in
the worktree because the file is untracked at the pin; the tracked mirror of the claim is
`tools/target-layering-brief.md:91`) versus the gate at the baseline: `EXCLUDED_FILES=()` at `check-infra-genericity.sh:74`,
the comment block `:70-73` and the header `:38-44` (the value-function-artifact format bump landed, so the exemption is
retired and `output/policy/` is scanned like every other infra source file). `stationVerdict: claim-stale`; no
disposition, severity, id, Alignment or recommendation key (guarded by assertion before the file is written; a handoff
with an empty `baselineEvidence` is refused). The same evidence is drift candidate `drift-08`, defended below with its
`partIRef` copied verbatim.

## Defender summary

28 survivors (29 received − 1 dup-of − 0 anchor-missing − 0 cleared − 0 re-raise − 0 conflicts-held − 0 returned), one read-only Opus
`adversarial-defender` each, dispatched in parallel batches with `defender-prompt.md` read verbatim plus the single
candidate input object (`/tmp/build-ci-defenders/in/<candidateRef>.json`: the candidate, its census row, prior context,
related ids and candidates, merged twin, the three screens, ingest notes and the attacker lens's needs-human items);
envelopes handed back through a scratch file outside the repository (`WRITTEN <bytes> <path>`). Every envelope was
checked with `tools/validate-envelope.py --role defender --station build-ci` plus the station clauses: exactly one
verdict keyed by the candidateRef; `survivingClaim` strictly narrower than the title (not equal after normalisation,
token-Jaccard < 0.85, ≥ 40 chars) and absent on a dismissal; a dismissal carries `dismissalBasis` (vocabulary of five) +
`basisCitation`, `sanctionedBy` from the closed set iff `sanctioned-advisory`, `contractCited` iff `contract`;
`enforcementStrength` in vocabulary and equal to the census class for a gate candidate (`unwired` never without a
census row); `measurement` UNMEASURED on the performance lens and n/a elsewhere; `alignmentHint` in vocabulary with
`conflicts` a boolean equal to (`alignmentHint == conflicts`) and `conflictsRule` iff true; `partIRef` / `reRaiseOf`
copied verbatim; no diff or code block; no timing number and no PD id / deck on a performance verdict. One re-dispatch
on failure; a second failure records `unresolved` + `_needsHuman`.

**Retry history:**

- none — every envelope passed on the first dispatch.

Result: **24 confirmed, 4 dismissed, 0 unresolved**;
20 defended candidates carry a `_needsHuman` note for the owner gate; 0 verdicts set
`conflicts: true`; 2 sharpen a live id (CD-061).

## Needs-human items surfaced by the defenders

| candidateRef | question |
|---|---|
| architecture-00 | Owner must choose where the single owner of the infra-crate set sits: CLAUDE.md:39-41 keeps its enumeration pinned by a new divergence check in the gate, or the bullet becomes a shape-only pointer to the gate's list, since doc-integrity section 3 mode 1 admits either and the choice decides which site an amendment edits first. |
| architecture-01 | Should the gate's token scope ever widen past standalone tokens, given that an underscore-separator boundary reds this blocking gate on the cobre-io timing and row-selection vocabulary the live entry ratified as needing no rename? |
| architecture-01 | Should the widening the live entry calls for also name the cobre-solver FreezeScratch field site at crates/cobre-solver/src/freeze.rs:22, which sits outside the cobre-io Parquet and IterationRecord scope that entry summarizes? |
| architecture-02 | Owner call on the trigger list: extend ci.yml's pull_request filter to [main, develop], accepting a full ci.yml run on every develop pull request, or ratify post-merge detection on develop as the intended trade and record that rationale in the workflow or a design doc, since no text at the pin states it. |
| architecture-03 | Shuffle-matrix reachability is an owner call: restore the commented-out nightly schedule at invariance-shuffle.yml:5-7, or convert the bare-ignore shuffle tests to the crate's slow-tests cfg_attr form so the automatically triggered suite runs them and the dispatch workflow becomes a convenience. |
| architecture-03 | Whether the compile-only guard against this class (a target-compiling step over the parity test binary under the shuffle feature set) is wanted at all, given that ci.yml:114 and :233 already compile that target under a feature set that works. |
| architecture-04 | Is examples/1dtoy/ contractually EQUAL to what cobre init materializes, or a superset allowed to carry non-input files as examples/4ree/README.md does? The answer decides whether the missing assertion is set equality or one-way canonical-subset inclusion. |
| architecture-05 | Should the manifest-scoped clippy pass live beside the existing manifest-scoped test step in the python job, inheriting its interpreter and backend selection on the single `3.12` matrix entry, or somewhere that does not need an interpreter present? |
| architecture-05 | Does the documentation-warning bar apply to the bindings crate's rustdoc, given `doc = false` at `crates/cobre-python/Cargo.toml:18` and that the `--workspace` docs job at `ci.yml:399` cannot see the crate at all? |
| architecture-05 | Once CI covers the crate, should the contributor checklist at `CONTRIBUTING.md:386` and the release checklist at `:602-604` also carry the manifest-scoped clippy command, or does automated coverage discharge the obligation `CONTRIBUTING.md:216` states? |
| over-engineering-00 | Whether a first .github/actions/ directory is acceptable for the eight intra-ci.yml copies alone, since GitHub Actions workflows cannot share steps by YAML anchor, leaving a composite action or a checked-in setup script invoked by one step as the only two shapes. |
| over-engineering-02 | Scope call for whoever holds the list: must a guard over the hoisted directory list also cover quality-report.sh:53-64, the sixth copy, given that the library's holdout array at :46-50 already tracks that script for the boundary regex while the header at :11-14 names only two holdouts? |
| over-engineering-03 | Owner call on direction, not on the fact: delete the inert cobre-cli declaration (and its cobre-io / cobre-stochastic twins), or keep the convention deliberately and then either restate the :55-56 rationale without the workspace-wide claim or extend the declaration to cobre-core:15, cobre-solver:15 and cobre-comm:20 so the stated property becomes true. |
| over-engineering-04 | Owner call on disposition, not on the fact: keep the three unconsumed slow-tests entries (cobre-cli:57, cobre-io:19, cobre-stochastic:18) as a declared convention, in which case the scope word in the cobre-io comment at :17-18 needs correcting and the three feature-table members that lack the entry decide whether the convention extends to them, or delete them; the same call governs sibling candidate over-engineering-03 and the cobre-stochastic instance, which is additionally documented at cobre-stochastic/README.md:81. |
| over-engineering-05 | Owner policy call: are the five reserved crate names meant to be re-published at every release so they keep holding their place on crates.io (their manifests inherit the workspace version and set no publish key), or is their absence from the publish list the intended policy that the manifests should state with publish = false? The tree states neither, so the fix direction is an owner decision, not a reading. |
| drift-02 | Whether a doc's own status line must use one of the five index vocabulary values or only be present is an owner call: at the pin two docs (enumerated-traversal-distribution.md:3 and backward-warm-start-channels.md:10) carry descriptive status prose instead of a vocabulary token, and docs/design/README.md:4 states only that a status exists, so the answer decides whether the dismissed half of this candidate is a defect at all. |
| drift-06 | Whether the private spec-SS citation family is re-anchored or published is the owner's call: crates/cobre-comm/src/ferrompi.rs:279 cites backend-ferrompi.md SS5.2 and :241 and :272 cite bare spec SS4.7 and SS3.1, and neither backend doc exists in the tree, in any ref's add history, or in the sibling methodology repo, so the fix at local_conformance.rs:4 sets the pattern for those sites too. |
| drift-06 | Whether check-comment-refs.sh's scan set should be aligned to the sibling gate is an owner decision this verdict does not make: its comment at :56 says the set mirrors check-no-plan-leaks.sh SCAN_DIRS, but that gate appends crates/*/tests and crates/*/benches at :148 and its header at :76-80 records an incident where exactly this scan-set asymmetry let leaks ship. |
| drift-07 | Owner must decide whether `.claude/rules/*.md` section numbers are a citable coordinate at all, since five gate headers cite them and no gate scans shell-script headers; that decision sets whether the residue is re-pointing the five citations alone or re-pointing plus a suffix check on the `.claude/rules/` token class check-comment-refs.sh already parses at :75. |
| drift-08 | The Part-I source document plans/generalizing/beyond-sddp-generalization.md is untracked at the pin, so no in-repo guard and no baseline read can see its copy of the exemption parenthetical; the owner decides whether a correction is expected in the untracked source as well as in the tracked mirror at target-layering-brief.md:91. |
| performance-00 | Owner call carried over from the attacker: whether every ci.yml job keeps a byte-identical MPICH prelude for readability even where its own feature string excludes `mpi`, or the prelude is derived per job from that feature string. It still decides the disposition of this confirmed residue for both the schemas and the python job, and it is the only attacker item that survives this read (the shared-key populator, the single-producer release binary and the license-tool version pin belong to other candidates). |
| performance-02 | Whether a populator job and `needs:` edges for the shared MPICH key are worth adding a serialization point to every warm run in order to collapse the rare cold-key fan-out is an owner trade-off; the workflow text supports both shapes and neither is required by a pinned rule. |
| performance-03 | Owner call: whether the release cobre binary gets one producer plus a publish/consume edge — which would be the first needs: edge in `ci.yml` (none at the pin, so every job is independent today) and would let one broken build block the consumers instead of each gate surfacing its own — or stays duplicated per job; the candidate's second shape, a shared rust-cache scope, is not an alternative to that decision, since a cache scope does not hand a built binary to another job. |
| performance-05 | Whether a prebuilt distribution of cargo-about exists for the exact version pinned at ci.yml:443 together with its cli feature selection is an external fact unverifiable at the baseline, and it decides whether that third site can join the amend at all. |
| performance-05 | Whether the rust-cache action wired at ci.yml:412 and :495 already restores compiled cargo binaries between runs is a property of an action implemented outside this repository, and it decides how much repeated work the two confirmed sites actually carry. |
| performance-06 | Whether a job-tailored submodule set is acceptable at all: actions/checkout exposes no per-submodule input, so the only tailored shape drops the input and hand-lists submodule paths in a following step on each affected job, creating a second copy of the four-entry .gitmodules registry inside the workflow with no guard pinning the two together; the owner decides whether that duplication is worth the avoided fetch, and the pointer is inert until they do. |

## Per-candidate roster

Every one of the 29 candidateRefs, exactly once (mirrors `verdicts.json`; `state` is the station test vocabulary —
accepted / rejected-anchor / rejected-sanctioned-advisory / rejected-re-raise / rejected-defender / merged / held-conflicts /
unresolved — derived from disposition + verdict + dismissalBasis).

| candidateRef | disposition | verdict | state | enforcement | alignment | measurement | partIRef | NH |
|---|---|---|---|---|---|---|---|---|
| architecture-00 | defended | confirmed | accepted | blocking | advances-0b | n/a |  | yes |
| architecture-01 | defended | dismissed (deliberate-and-documented) (sharpens CD-061) | rejected-defender | blocking | neutral | n/a |  | yes |
| architecture-02 | defended | confirmed | accepted | blocking | neutral | n/a |  | yes |
| architecture-03 | defended | confirmed | accepted | blocking | neutral | n/a |  | yes |
| architecture-04 | defended | confirmed | accepted | blocking | neutral | n/a |  | yes |
| architecture-05 | defended | confirmed | accepted | blocking | advances-0a | n/a |  | yes |
| over-engineering-00 | defended | confirmed | accepted | blocking | neutral | n/a |  | yes |
| over-engineering-01 | defended | confirmed | accepted | not-a-gate | neutral | n/a |  |  |
| over-engineering-02 | defended | confirmed | accepted | blocking | neutral | n/a |  | yes |
| over-engineering-03 | defended | confirmed | accepted | not-a-gate | neutral | n/a |  | yes |
| over-engineering-04 | defended | confirmed | accepted | not-a-gate | neutral | n/a |  | yes |
| over-engineering-05 | defended | confirmed | accepted | blocking | neutral | n/a |  | yes |
| drift-00 | defended | dismissed (premise-false-at-pin) | rejected-defender | not-a-gate | neutral | n/a |  |  |
| drift-01 | defended | confirmed | accepted | not-a-gate | neutral | n/a |  |  |
| drift-02 | defended | confirmed | accepted | not-a-gate | neutral | n/a |  | yes |
| drift-03 | defended | confirmed | accepted | not-a-gate | neutral | n/a |  |  |
| drift-04 | defended | confirmed | accepted | blocking | neutral | n/a |  |  |
| drift-05 | defended | dismissed (premise-false-at-pin) (sharpens CD-061) | rejected-defender | blocking | neutral | n/a |  |  |
| drift-06 | defended | confirmed | accepted | blocking | neutral | n/a |  | yes |
| drift-07 | defended | confirmed | accepted | blocking | neutral | n/a |  | yes |
| drift-08 | defended | confirmed | accepted | blocking | neutral | n/a | I.3-6 | yes |
| drift-09 | defended | confirmed | accepted | not-a-gate | neutral | n/a |  |  |
| performance-00 | defended | confirmed | accepted | blocking | neutral | UNMEASURED |  | yes |
| performance-01 | dup-of | — | merged | blocking | — | — |  |  |
| performance-02 | defended | confirmed | accepted | blocking | neutral | UNMEASURED |  | yes |
| performance-03 | defended | confirmed | accepted | blocking | neutral | UNMEASURED |  | yes |
| performance-04 | defended | dismissed (premise-false-at-pin) | rejected-defender | blocking | neutral | UNMEASURED |  |  |
| performance-05 | defended | confirmed | accepted | blocking | neutral | UNMEASURED |  | yes |
| performance-06 | defended | confirmed | accepted | blocking | neutral | UNMEASURED |  | yes |

## Reconciliation

Counts by state: {"accepted": 24, "rejected-defender": 4, "merged": 1}. Received 29; verdicts.json keys 29; every received
candidateRef accounted for exactly once. Read-only check: `git status --porcelain -- . ':!plans/'` is empty (nothing under
`crates/`, `scripts/`, `.github/` or `schemas/` is modified; the station's writes sit under `stations/build-ci/`, which is
TRACKED in this repository — the ticket's "gitignored plans directory" premise is stale and is recorded as a deviation).
No IDs were assigned and no BACKLOG section was written.
