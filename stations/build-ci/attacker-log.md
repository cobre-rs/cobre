# build-ci attacker dispatch log

Baseline: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (register pin; the ticket text's `a136840d` is the superseded scaffold pin — every figure the
workers were handed was re-measured at the pin, see `attacker-prompt.md` § Frozen figures). Every finding below is a
snapshot of that tree (G-SNAPSHOT), not a standing claim. Dispatched: 2026-09-18 (workers launched 23:43 local; envelopes
landed 23:58–00:03; the performance retry and the gate ran 2026-09-19). Agent type: `adversarial-attacker` (Opus,
read-only — no Write/Edit tools; each worker wrote its envelope to `/tmp/build-ci-attackers/out/<lens>.json` with one
heredoc and replied `WRITTEN <bytes> <path>`). Dispatcher: the main session, sole writer of every artifact under
`stations/build-ci/`.

## Prompt and validator (before dispatch)

One shared contract, `attacker-prompt.md` (45,963 bytes, generated from inventory.json / gate-census.json /
prior-register.md by the dispatcher's generator; the core-io template's structure carried through the sddp and cli-python
refinements): preamble (sweep list, inputs), 14 rules (read-only + scratch-file hand-over; one bare JSON object; ANCHOR
FORM `{path, line}` with `{path, symbol}` only on the three build.rs; ENFORCEMENT STRENGTH mandatory with the
advisory-by-design exit-0 quote and the census-owned `unwired` verdict; the TWO-SITE rule with the genericity rule as the
worked instance; PULL-DON'T-PUSH with the ten-consumer MPICH carve-out; performance INFORMATIONAL / `UNMEASURED` with the
allowed-numeral rule; anchors inside the sweep list with crate-side facts cited never anchored; prose fix shapes; never
re-raise; evidence as a command; reserved seams first with `reservedSeamsCheck`; drift pairing `docClaim` ↔ `treeFact`;
empty-is-legitimate with `cleanVerdict`), the alignment vocabulary, the frozen-figure table (26 rows: ticket figure → pin
figure → command; the seven `ci.yml` figures that moved — MPICH Build steps :377/:468/:522, seven `timeout-minutes` lines,
the `docs-examples`…`python` job anchors — the `check-docs-examples.sh` `EXPECTED_INPUT_FILES` line 39 → 32, the
`cobre-cli/Cargo.toml` slow-tests block 54-56 → 55-57, the third out-of-vocabulary README row :24 and the un-indexed
`post-horizon-input-unification.md`), the five-stub table, the gate-wiring census (17 rows with the wiring anchor and the
exit-0 anchors; the naive negative grep quoted and marked insufficient), the do-not-raise set (do-not-touch ids, sanctioned
seams, four recorded observations, the live entries owned elsewhere as a sharpen-only table, the three inbound handoffs as
probes), four lens blocks (architecture P1–P6, over-engineering P1–P5, drift P1–P5, performance P1–P3), the frozen envelope
and the dispatch rules.

The shape gate is `tools/validate-envelope.py --role attacker --station build-ci`. Two facts of the shared validator
predate a non-crate station and conflicted with the ticket's frozen contract; both were put to the owner on 2026-09-19
and extended with the owner's approval (recorded as `extra` file changes on the ticket, never silently):

- its closed lens set lacked `drift` — `LENSES` now includes `drift` (the ticket freezes four lenses and requires every
  candidates file, `candidates-drift.json` included, to pass the validator with exit 0);
- its repo-relative path rule accepted only `crates/ scripts/ docs/ schemas/ examples/ tests/ .github/ .claude/ plans/`
  prefixes, so the three root files the sweep list names (`Cargo.toml`, `ARCHITECTURE.md`, `CLAUDE.md`) failed as
  "must be repo-relative" — exactly those three root files are now accepted (AC 3 itself requires a candidate citing
  `CLAUDE.md:39-41`). Three of the four first-attempt envelopes had failed only on such anchors; after the extension
  they pass unchanged (no re-dispatch, no hand-repair).

One cite in the prompt was disambiguated after dispatch: `Cargo.toml:112-121` (drift P4) read as the root manifest while
it names `crates/cobre-sddp/Cargo.toml:112-121`; the drift worker had read it correctly (its NH40 candidate anchors
`crates/cobre-sddp/Cargo.toml:112`), and the prompt now spells the full path so every tracked `path:line` cite in it
resolves at the pin (`CandidateEnvelopeTests.test_prompt_freezes_the_e7_contract`).

## Gate policies (fixed before the first merge so the verdicts are reproducible)

- **Five named guards, in order, each rejecting with its code:** `anchor-form` (a `{path, symbol}` anchor outside the
  three build.rs — the candidate is returned and the lens re-dispatched once); `out-of-scope-anchor` (an anchor under
  `crates/*/src` or `crates/*/tests` — the candidate is dropped and its title logged under § Out-of-scope for the owning
  station); `missing-enforcement` (a gate/workflow subject — `script` set, or, outside the drift lens, an anchor under
  `.github/workflows/`, `scripts/ci/`, `scripts/pre-commit` — without a strength in `blocking | advisory-by-design |
  unwired | transitively-advisory`; a doc/manifest/build-script subject carries `n-a`); `drift-unpaired` (a drift
  candidate without both `docClaim` and `treeFact`, each with a `{path, line}` that resolves at the pin); `perf-number`
  (a performance entry whose `measured` is not the literal `UNMEASURED`, or whose title / mechanism / fixShape /
  evidence.reading carries a numeral outside the allowed contexts: a line reference `L38` `:38` `lines 38-41`, a
  key-quoted configured value `timeout-minutes: 30` `MPICH_VERSION: "4.2.3"` `--unified=0` `@v5`, a version string, or a
  structural count followed by jobs / steps / workflows / scripts / consumers / copies / files / gates).
- **Other codes:** `anchor-missing` (no anchor survives at the pin inside the sweep list; a non-resolving or out-of-sweep
  anchor is first demoted to `citedContext` with its reason, per anchor), `out-of-station` (the `cobre` umbrella crate
  named in the over-engineering file — the cli-python station's subject), `settled` (a do-not-touch id named),
  `handoff-routed` (a Part-I re-verification record outside the drift envelope), `handoff-shape`.
- **Census join.** Every candidate that names a gate script (`script`, or the first `scripts/ci/*.sh|*.py` anchor) is
  joined to `gate-census.json`: an `unwired` assertion on a script with a `transitiveInvocation` is rewritten to
  `transitively-advisory` and the invocation site (`scripts/ci/quality-report.sh:132` for `check-comment-bloat.sh`) is
  appended as an anchor; any other strength that contradicts the census class is rewritten to the class; an
  `advisory-by-design` candidate that does not quote the `ADVISORY` step comment / `exit 0` gets the census quote and
  both exit-0 anchors appended (`evidence.censusQuote`). A negative grep alone never establishes `unwired`.
- **Stamps are mechanical, never prose edits;** every stamp, demotion, header fix and dropped key is listed below.
- **Two-site (rule 5).** An architecture candidate whose fix touches `CLAUDE.md` must fill `twoSite.prose/script/step`,
  each resolving at the pin; an incomplete one is a `two-site-incomplete` sharpen request bundled into the lens's single
  re-dispatch.
- **Envelope spelling.** `station: "build-ci"` (the slug every station artifact and the validator use), `stationLabel:
  "build-ci-docs"` (the ticket's spelling), `subStation: "build-ci"` (the validator requires the key; the station has one
  sub-surface).
- **Guard calibration.** The `perf-number` numeral-context rule was corrected during the first pass: its first form
  judged a numeral by a window that included the numeral itself, so the endpoint of a line range (`:344-346`) and a
  key=value flag (`--unified=0`) were mis-flagged; the corrected rule (judge by the context before and after the token)
  is the one stated above, it was validated on a scratch copy of the first performance envelope (never merged) and it
  fires on nothing in the final envelopes.
- **Re-dispatch.** A shape-invalid envelope or an `anchor-form` return is re-dispatched exactly once with the validator
  diagnostic quoted verbatim and the instruction to change nothing else; a lens still invalid after the retry is a Gap
  and its envelope is discarded whole.

## Dispatch table

| worker | lens | envelope valid | retries used | candidates | positives | needs-human | handoffs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `adversarial-attacker` (Opus, read-only) `att-build-ci-architecture` | architecture | yes | 0 | 6 (raw 6) | 7 | 5 | 0 |
| `adversarial-attacker` (Opus, read-only) `att-build-ci-over-engineering` | over-engineering | yes | 0 | 6 (raw 6) | 8 | 4 | 0 |
| `adversarial-attacker` (Opus, read-only) `att-build-ci-drift` | drift | yes | 0 | 10 (raw 10) | 10 | 3 | 1 |
| `adversarial-attacker` (Opus, read-only) `att-build-ci-performance` | performance | yes (attempt 1: fail — /tmp/build-ci-attackers/out/performance.json: $.candidates[6].anchors[6].path: must be repo-relative); see the retry note below | 1 | 7 (raw 7) | 6 | 4 | 0 |
- **performance retry note.** attempt 2 removed the `.gitmodules:1` anchor as instructed (moved to `citedContext`, joined by a second cited line, `crates/cobre-sddp/build.rs:11`) and, against the change-nothing-else instruction, also reworded that candidate's `fixShape` and `evidence.reading`; every other candidate, positive and needs-human item is byte-identical to attempt 1 (checked field by field). The reworded prose is the worker's own and is carried as returned.

## Coverage (named probes → what the lens returned)

- **architecture** (6 candidates, 7 positives):
  - P1 SCAN_DIRS two-site: 1 — The infra-genericity gate's crate list is a hand-kept literal in two unpaired places, so t
  - P2 PATTERN vocabulary: 2 — The infra-genericity gate matches one engine's vocabulary rather than the engine-concept r; The invariance-shuffle workflow cannot compile the test target it dispatches, and the auto
  - P3 trigger asymmetry: 2 — No CI workflow runs on a pull request into develop, so every blocking gate first sees a de; The invariance-shuffle workflow cannot compile the test target it dispatches, and the auto
  - P4 hand-mirrored templates: 1 — The CLI template mirror is guarded in one direction only: a file added to the canonical ex
  - P5 invariance-shuffle features: 1 — The invariance-shuffle workflow cannot compile the test target it dispatches, and the auto
  - P6 CI coverage / doc = false: 1 — No clippy invocation in CI reaches the published Python bindings crate, so the lint bar th
- **over-engineering** (6 candidates, 8 positives):
  - P1 MPICH triple: 2 — The MPICH Cache/Build/Set step triple is copied byte-for-byte eight times inside ci.yml, w; The MPICH version literal 4.2.3 has three independent owners in the SLURM image path (work
  - P2 slow-tests: 2 — cobre-cli declares an empty slow-tests feature with no cfg consumer anywhere in the crate,; cobre-io declares an empty slow-tests feature with no cfg consumer anywhere in the crate, 
  - P3 reserved stubs: 2 — (positive) The five reserved stub members impose no measurable workspace cost worth raisin; (positive) cargo-dist cannot ship a reserved stub binary because the dist member selection
  - P4 allowlist: 2 — (positive) scripts/ci/allow-rationale-allowlist.txt is entry-free by design; (positive) EXCLUDED_FILES=() in the genericity gate is an emptied exemption, not an unused
  - P5 gate-corpus shape: 1 — Five comment and doc gates each hand-maintain their own copy of the crate-source scan-dire
- **drift** (10 candidates, 10 positives):
  - P1 README vocabulary: 3 — docs/design/README.md declares a closed five-value status vocabulary and a delete-on-ship ; docs/design/README.md calls itself the map of the directory while post-horizon-input-unifi; Two index rows disagree with the status the doc itself carries: one doc declares a status 
  - P2 crate-map claims: 2 — ARCHITECTURE.md's crate map counts five dependencies for cobre-python while naming six on ; A shipped conformance test states it verifies the contracts of a backend-testing.md that h
  - P3 Part-I item 6: 1 — The roadmap's Part-I item 6 records a genericity-gate exemption for the four cobre-io poli
  - P4 inbound handoffs: 2 — A shipped conformance test states it verifies the contracts of a backend-testing.md that h; The unsafe-code hard rule enumerates one unsafe island in cobre-sddp while the crate's own
  - P5 rule-prose vs enforcement: 3 — (positive) CLAUDE.md:12's test command and ci.yml:32's NON_SOLVER_FEATURES value are the s; (positive) CLAUDE.md:117's schema-regeneration command matches the remediation the schema ; (positive) The three reserved slow-tests declarations describe themselves accurately
- **performance** (7 candidates, 6 positives):
  - P1 CI wall-time / cache: 3 — The schemas job carries the MPICH Cache/Build/Set triple although its only compile is the ; The python job carries the MPICH Cache/Build/Set triple on every instance of its version m; The eight MPICH consumers in ci.yml share one immutable cache key with no dependency edge 
  - P2 gate-runtime shape: 2 — The eight MPICH consumers in ci.yml share one immutable cache key with no dependency edge ; Ten of the fourteen quality-scripts steps each enumerate the crates tree themselves, and t
  - P3 build-graph shape: 4 — The schemas job carries the MPICH Cache/Build/Set triple although its only compile is the ; The python job carries the MPICH Cache/Build/Set triple on every instance of its version m; Three job definitions each compile the release cobre binary independently, with no artifac; Nine of the fourteen job definitions check out every vendored submodule recursively, and o

## Rejections

| title | lens | rejection code | detail |
| --- | --- | --- | --- |
| handoff I.3-6 | architecture (attempt 1) | `handoff-routed` | Part-I re-verification records travel in the drift envelope only (prompt: drift P3) |

## Out-of-scope drops (for the owning stations)

- none: no candidate anchored under `crates/*/src` or `crates/*/tests`; crate-side facts were cited in `citedContext` / `evidence` as rule 8 requires

## Census join

| script (from candidates) | lens | class asserted | census class | census anchor | exit-0 comment |
| --- | --- | --- | --- | --- | --- |
| `scripts/ci/check-infra-genericity.sh` | architecture | blocking | blocking | `.github/workflows/ci.yml:262` | — |
| `scripts/ci/check-infra-genericity.sh` | architecture | blocking | blocking | `.github/workflows/ci.yml:262` | — |
| `scripts/ci/check-allow-rationale.sh` | architecture | blocking | blocking | `.github/workflows/ci.yml:292` | — |
| `scripts/ci/check-docs-examples.sh` | architecture | blocking | blocking | `.github/workflows/ci.yml:360` | — |
| `scripts/ci/check-comment-refs.sh` | over-engineering | blocking | blocking | `.github/workflows/ci.yml:276` | — |
| `scripts/ci/check-infra-genericity.sh` | drift | blocking | blocking | `.github/workflows/ci.yml:262` | — |
| `scripts/ci/check-comment-refs.sh` | drift | blocking | blocking | `.github/workflows/ci.yml:276` | — |
| `scripts/ci/check-doc-paths.sh` | drift | blocking | blocking | `.github/workflows/ci.yml:274` | — |
| `scripts/ci/check-infra-genericity.sh` | drift | blocking | blocking | `.github/workflows/ci.yml:262` | — |
| `scripts/ci/check_schemas.sh` | performance | blocking | blocking | `.github/workflows/ci.yml:339` | — |
| `scripts/ci/check-docs-examples.sh` | performance | blocking | blocking | `.github/workflows/ci.yml:360` | — |
| `scripts/ci/lib/comment_scan.sh` | performance | blocking | blocking | `.github/workflows/ci.yml:268` | — |

No candidate asserted `unwired`, so the `check-comment-bloat.sh` → `transitively-advisory` rewrite had nothing to fire on;
the census row (`scripts/ci/quality-report.sh:132`, invoker class advisory-by-design) stays the station's verdict for that
script and travels to ingest through `gate-census.json`. The three advisory-by-design gates (`check-comment-line-refs.sh`,
`check-comment-banners.sh`, `quality-report.sh`) appear only in `positives`, each citing the `ADVISORY` step comments
(`ci.yml:278-279`, `:282-283`, `:298-299`) and the census exit-0 anchors.

## Stamps, demotions, header fixes, dropped keys, gate notes

- **over-engineering** (attempt 1) — gate note on “Five comment and doc gates each hand-maintain their own copy of the crate-source”: mentions CD-061 without reRaiseOf — ingest decides sharpen vs new gate-half

## Handoffs

- **drift** `I.3-6` — verdict `claim-stale`; claim: cobre-io's policy checkpoint format is literally cut records, with the parenthetical that the genericity gate exempts the four policy files (target-layering-bri; evidence: `scripts/ci/check-infra-genericity.sh:74` — EXCLUDED_FILES=() — empty; no file is exempt from the genericity scan at this baseline; `scripts/ci/check-infra-genericity.sh:70` — the comment block :70-73 states the value-function-artifact format break retired the former output/p. No disposition, no Alignment value (epic 9 decides).
- **architecture** — the worker also emitted a `handoff I.3-6` record; dropped here because Part-I re-verification records travel in the drift envelope only (`handoff-routed`).

## Needs-human items surfaced for the gate

- **architecture NH1** — Should the pull_request trigger cover develop as well as main, accepting a CI run on every develop pull request, or is post-merge detection on develop the intended trade?
- **architecture NH2** — For the shuffle matrix, should the commented-out nightly schedule be restored, or should the five shuffle tests move to the crate's slow-tests gating form so the automatically triggered suite runs them and the dispatch workflow becomes a convenience?
- **architecture NH3** — Should the clippy pass for the Python bindings crate live beside the existing manifest-scoped test step in the python job, which adds minutes to one matrix entry, or somewhere that does not need an interpreter present?
- **architecture NH4** — Does the documentation-warning bar apply to the bindings crate's rustdoc, given that its readers are Python users and the workspace docs job cannot see the crate?
- **architecture NH5** — Does the underscore-boundary change to the genericity pattern need owner sign-off before it lands, since it will start flagging identifiers in cobre-solver that the register entry deliberately left un-renamed?
- **over-engineering NH1** — Are the five reserved crate names meant to be re-published at every release so they keep holding their place on crates.io, given that their manifests carry crates.io-facing Reserved crate name descriptions and none sets publish = false, or is their absence from the publish list the intended policy that the manifests should be made to state?
- **over-engineering NH2** — Should the three empty slow-tests declarations be kept as a deliberate seam despite no mirror entry and a falsified consistency rationale, in which case the feature also belongs in the three members that have a feature table and lack it, or deleted as the candidates propose?
- **over-engineering NH3** — Is a first composite action under .github/actions/ acceptable, given the directory does not exist at the pin and the change would reshape the release-critical MPI workflows as well as ci.yml?
- **over-engineering NH4** — Routing, not a finding for this lens: scripts/ci/lib/comment_scan.sh:11-14 says two gates cannot use the streaming helpers and names check-comment-banners.sh and check-allow-rationale.sh, while its own drift-guard array at :46-50 carries three holdouts including quality-report.sh; the enumeration mismatch belongs to the drift lens if it is wanted.
- **drift NH1** — Whether the two docs that self-declare Implemented should be folded into their authoritative homes and deleted, as docs/design/README.md:31 prescribes, or whether the status vocabulary should gain a sixth value for the retained-pending-fold state, is the owner's decision and it governs three of the eight index rows.
- **drift NH2** — Whether section numbers in .claude/rules/*.md are a citable coordinate at all is the owner's call: five gates cite them, four citations do not resolve, and the alternative is to require directive ids and heading titles instead, as check-allow-rationale.sh:5 already does.
- **drift NH3** — Whether `doc = false` on the bindings' lib target at crates/cobre-python/Cargo.toml:18 is a sanctioned seam or a rustdoc-coverage gap needs the owner, per the inbound cli-python handoff; this lens found no in-sweep document that claims rustdoc coverage for the bindings, so there is no paired drift to raise and the fact is left to the architecture lens.
- **performance NH1** — Whether the MPICH prelude should stay uniform across every ci.yml job for readability even where the job's feature set does not demand it, or be derived per job from that feature set, is a workflow-style decision the owner owns; both MPICH entries above are written to be actionable either way.
- **performance NH2** — Whether introducing a populator job and dependency edges for the shared MPICH key is worth serializing the workflow behind one job on a warm key is an owner trade-off, since today every consumer restores in parallel and nothing waits.
- **performance NH3** — Whether the release cobre binary should have one producer and be passed between jobs, at the cost of losing the independence that stops one job's broken build from masking another's, is the owner's call between the two shapes named in that entry.
- **performance NH4** — Whether the license tool's exact version pin and its feature selection are available as a prebuilt distribution is an external fact this read-only station cannot check at the baseline, and it decides whether that one call site can join the install-action amend.

## Gaps

- none: all four lenses produced a valid envelope (one after its single retry)

## Read-only verification

Three commands, run from the repo root after the merge (outputs verbatim):

```
$ for l in architecture over-engineering drift performance; do python3 plans/architecture-debt-audit/tools/validate-envelope.py --role attacker --station build-ci plans/architecture-debt-audit/stations/build-ci/candidates-$l.json && echo "valid: $l"; done
valid: architecture
valid: over-engineering
valid: drift
valid: performance
$ grep -nE '^\s*[-+]{3} |^\s*@@ |```(diff|patch|yaml)' plans/architecture-debt-audit/stations/build-ci/candidates-*.json; echo "grep exit $?"
grep exit 1
$ git status --porcelain -- . ':!plans/'
(empty — no tracked file outside plans/ is modified, none untracked)
```

The ticket's third check reads `git status --porcelain` "lists nothing, since every artifact sits under the gitignored plans/ tree". In this repository `plans/` is TRACKED (the ticket text is stale; recorded as a deviation, not a spec edit), so the station artifacts show as untracked until the ticket commit and the read-only check is the `':!plans/'` form above plus `CleanTreeTests.test_no_tracked_file_is_modified`, which asserts no tracked file under `crates/ docs/ schemas/ scripts/ .github/ Cargo.toml` changed.

## Per-lens records

```json
{
 "architecture": {
  "lens": "architecture",
  "attempt": 1,
  "gatedAt": "2026-09-19",
  "rawPath": "/tmp/build-ci-attackers/out/architecture.json",
  "validator": "pass",
  "validatorErrors": [],
  "rawCandidates": 6,
  "candidates": 6,
  "positives": 7,
  "needsHuman": 5,
  "handoffs": 0,
  "rejections": [
   {
    "title": "handoff I.3-6",
    "code": "handoff-routed",
    "detail": "Part-I re-verification records travel in the drift envelope only (prompt: drift P3)"
   }
  ],
  "demotedAnchors": [],
  "stamps": [],
  "droppedKeys": [],
  "headerFixes": [],
  "sharpen": [],
  "gateNotes": [],
  "cleanVerdict": false
 },
 "over-engineering": {
  "lens": "over-engineering",
  "attempt": 1,
  "gatedAt": "2026-09-19",
  "rawPath": "/tmp/build-ci-attackers/out/over-engineering.json",
  "validator": "pass",
  "validatorErrors": [],
  "rawCandidates": 6,
  "candidates": 6,
  "positives": 8,
  "needsHuman": 4,
  "handoffs": 0,
  "rejections": [],
  "demotedAnchors": [],
  "stamps": [],
  "droppedKeys": [],
  "headerFixes": [],
  "sharpen": [],
  "gateNotes": [
   {
    "title": "Five comment and doc gates each hand-maintain their own copy of the crate-source scan-directory list",
    "note": "mentions CD-061 without reRaiseOf — ingest decides sharpen vs new gate-half"
   }
  ],
  "cleanVerdict": false
 },
 "drift": {
  "lens": "drift",
  "attempt": 1,
  "gatedAt": "2026-09-19",
  "rawPath": "/tmp/build-ci-attackers/out/drift.json",
  "validator": "pass",
  "validatorErrors": [],
  "rawCandidates": 10,
  "candidates": 10,
  "positives": 10,
  "needsHuman": 3,
  "handoffs": 1,
  "rejections": [],
  "demotedAnchors": [],
  "stamps": [],
  "droppedKeys": [],
  "headerFixes": [],
  "sharpen": [],
  "gateNotes": [],
  "cleanVerdict": false
 },
 "performance": {
  "lens": "performance",
  "attempt": 2,
  "gatedAt": "2026-09-19",
  "rawPath": "/tmp/build-ci-attackers/out/performance.json",
  "validator": "pass",
  "validatorErrors": [],
  "rawCandidates": 7,
  "candidates": 7,
  "positives": 6,
  "needsHuman": 4,
  "handoffs": 0,
  "rejections": [],
  "demotedAnchors": [],
  "stamps": [],
  "droppedKeys": [],
  "headerFixes": [],
  "sharpen": [],
  "gateNotes": [],
  "cleanVerdict": false,
  "retryDiff": "attempt 2 removed the `.gitmodules:1` anchor as instructed (moved to `citedContext`, joined by a second cited line, `crates/cobre-sddp/build.rs:11`) and, against the change-nothing-else instruction, also reworded that candidate's `fixShape` and `evidence.reading`; every other candidate, positive and needs-human item is byte-identical to attempt 1 (checked field by field). The reworded prose is the worker's own and is carried as returned."
 },
 "performance@1": {
  "lens": "performance",
  "attempt": 1,
  "gatedAt": "2026-09-19",
  "rawPath": "/tmp/build-ci-attackers/out/performance.json (attempt-1 content, overwritten by the retry; reconstructed from the gate's scratch copy)",
  "validator": "fail",
  "validatorErrors": [
   "/tmp/build-ci-attackers/out/performance.json: $.candidates[6].anchors[6].path: must be repo-relative"
  ],
  "rawCandidates": 0,
  "candidates": 0,
  "positives": 0,
  "needsHuman": 0,
  "handoffs": 0,
  "rejections": [],
  "demotedAnchors": [],
  "stamps": [],
  "droppedKeys": [],
  "headerFixes": [],
  "sharpen": [],
  "gateNotes": [],
  "cleanVerdict": false
 }
}
```
