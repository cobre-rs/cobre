# build-ci attacker worker prompt (shared preamble; four lenses, one sub-surface)

Baseline: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (the register pin; the ticket text's `a136840d` is the superseded scaffold pin —
every figure below was re-measured at `077dbe2c`, see §Frozen figures). Station: `build-ci` (label:
`build-ci-docs` — the build, CI, scripts, schemas and design-docs surface; the validator and every station
artifact use the slug `build-ci`). Your lens is `<LENS>` — one of `architecture`, `over-engineering`, `drift`,
`performance`; the dispatcher names it in your directive.

Sweep ONLY: `.github/workflows/` (8 files), `scripts/` (`scripts/ci/*.sh|*.py`, `scripts/ci/lib/comment_scan.sh`,
`scripts/ci/allow-rationale-allowlist.txt`, `scripts/pre-commit`), `schemas/` (18 JSON exports),
`crates/cobre-io/schemas/policy.fbs`, the three build scripts `crates/cobre-solver/build.rs`,
`crates/cobre-sddp/build.rs`, `crates/cobre-cli/build.rs`, the root `Cargo.toml`, every `crates/*/Cargo.toml`,
`tests/slurm/`, `docs/design/` (README + 9 docs), `ARCHITECTURE.md`, `CLAUDE.md`. Every path is read at the baseline:
`git show 077dbe2c:<path>` / `git grep <pat> 077dbe2c -- <paths>` / `git ls-tree -r --name-only 077dbe2c <dir>`; never
the working tree, never `cargo`, never `bash scripts/ci/...`, never a workflow run (the gate results you need are
pre-measured in §Gate-wiring census).

Inputs (read them first, in this order):
- `plans/architecture-debt-audit/stations/build-ci/inventory.json` — the enforcement-surface inventory at the pin:
  13 workspace members (5 reserved stubs), the `cobre-python` exclusion, 3 build.rs, 4 vendored submodules, 8
  workflows / 1,610 lines, the 14 `ci.yml` jobs with anchors, the 10 `Build MPICH from source` steps, the gate
  corpus (14 `.sh` + 2 `.py` + the lib helper), the 18 schemas, the design-doc index facts, 10 ticket-figure deviations
- `plans/architecture-debt-audit/stations/build-ci/gate-census.json` — the wiring class of all 17 gate files with
  the anchor that proves it (13 blocking / 3 advisory-by-design / 1 unwired-but-transitively-invoked)
- `plans/architecture-debt-audit/stations/build-ci/prior-register.md` — what the register and the mirror already say
  about these surfaces (CD-061 is anchored here; 78 evidence citers; three inbound handoffs; the do-not-touch list;
  four observations already recorded)
- `plans/architecture-debt-audit/tools/target-layering-brief.md` — the layering (L0–L4), the phase split, the four
  `conflicts` triggers and the closed Alignment vocabulary; its §4 row 6 is the Part-I item-6 claim the drift lens
  re-verifies
- `plans/architecture-debt-audit/stations/solver-comm/partI-handoff.json` → `genericityGateBlindSpot` and
  `handoffs[to=E7]` — the pre-seeded blind-spot record for the architecture lens
- `docs/design/reserved-seams-and-deferred-debt.md` at the pin (`git show 077dbe2c:docs/design/reserved-seams-and-deferred-debt.md`)
  — reserved seams and cleared items (the mirror); `CLAUDE.md` § Hard Rules at the pin (`Unwired config is reserved,
  not dead`; the genericity rule at :39-41; the `unsafe_code` enumeration at :19; the test command at :12)

You are one of four read-only Opus attacker workers, one per lens over the same surface. Stay inside your lens; the
other three lenses are siblings and no worker sees another's output. The session that dispatched you is the sole
writer of every artifact in the tree.

## RULES (each is a guardrail; a violated rule voids the envelope and costs the lens its one re-dispatch)

1. **Read-only.** Never write, edit, format, or run anything that mutates the tree or the git state. `git show
   077dbe2c:<path>`, `git grep <pat> 077dbe2c -- <paths>`, `git ls-tree`, `wc`, `sed -n`, `awk`, `diff` on
   `git show` output, `jq` are fine; `sed -i`, `cargo` (any), `bash scripts/ci/...`, `python3 scripts/ci/...`,
   `git checkout`, `git stash` are not. The only file you may create is the scratch file the dispatcher names,
   OUTSIDE the repository (`/tmp/build-ci-attackers/out/<LENS>.json`), written with a single `cat > <path> <<'EOF' … EOF`.
2. **One JSON object and nothing else.** Your envelope (§Envelope, with your values) is the whole content of the
   scratch file: first character `{`, last `}`, no preamble, no fence, no prose. Your chat reply is the single line
   `WRITTEN <bytes> <path>`.
3. **ANCHOR FORM.** Every anchor is `{"path": ..., "line": <n>}` for YAML, shell, Python, TOML, JSON, `.fbs` and
   Markdown — `line` is the 1-based line of the fact at the baseline blob. `{"path": ..., "symbol": ...}` is legal
   ONLY for `crates/cobre-solver/build.rs`, `crates/cobre-sddp/build.rs` and `crates/cobre-cli/build.rs`
   (`check-anchors.py` resolves symbols with a Rust declaration regex: `fn|struct|enum|const|static|mod|...`).
   Workflow job names and step names go in `evidence` or in the `wiring` object, NEVER in a symbol field. A
   `{path, symbol}` anchor on any other file is `anchor-form`: the candidate is returned and the lens is
   re-dispatched once with the diagnostic.
4. **ENFORCEMENT STRENGTH is mandatory.** Every candidate carries `enforcementStrength`: for a gate or workflow
   subject one of `blocking` | `advisory-by-design` | `unwired` | `transitively-advisory`; for a subject that is not
   a gate or a workflow (a manifest, a doc row, a build script) the literal `n-a`. `advisory-by-design` MUST quote the
   step comment that proves `exit 0` is intentional (the `ADVISORY` step comment in `ci.yml` plus the script's
   terminal `exit 0` — both anchors are in §Gate-wiring census; that is sanctioned, not a defect). `unwired` MUST
   show a negative grep over `.github/workflows/`, `scripts/pre-commit` AND every `scripts/ci/*.sh` caller — and at
   this baseline the census already proves that no gate is unwired: `check-comment-bloat.sh` is invoked by
   `scripts/ci/quality-report.sh:132` and is `transitively-advisory`. A gate candidate names its script in `script`
   (repo path) so the merge can join it to the census; a strength that contradicts the census is rewritten to the
   census class and the rewrite is logged.
5. **TWO-SITE RULE.** A fix touching a hard rule names BOTH the prose rule in `CLAUDE.md` AND the enforcing script
   plus its `ci.yml` step, in the `twoSite` object (`prose`, `script`, `step`, each `{path, line}`). One site only
   is incomplete and is returned for sharpening. The genericity rule is the worked instance: `CLAUDE.md:39-41` ↔
   `scripts/ci/check-infra-genericity.sh:62-68` (`SCAN_DIRS`) ↔ `.github/workflows/ci.yml:261-262` (job
   `quality-scripts` at :248, step `Infra genericity gate`).
6. **PULL, DON'T PUSH.** Amend an existing gate, workflow or rule. A new gate framework, a new shared scan library
   with one caller, or a one-consumer composite action → `alignmentHint: "conflicts"` (trigger 4 of the layering
   brief §3) plus the roadmap-consistent alternative in `fixShape`. The MPICH Cache/Build/Set triple has TEN
   consumers (8 byte-identical copies in `ci.yml` + 2 near-variants) and is EXEMPT from the one-consumer objection
   — do not tag it `conflicts` on that ground.
7. **Performance is INFORMATIONAL here.** This station measures nothing: no timing number, no ratio, no
   percentage, no wall-clock figure, no speedup, no byte or size figure, no PD id request, no `cobre_reduzido_2`
   run. Every performance candidate carries `"measured": "UNMEASURED"` and a `mechanism` (the structural cost
   mechanism in words). The ONLY numerals allowed anywhere in a performance entry are line references (`L38`,
   `:38`, `lines 38-41`), configured values quoted with their key (`timeout-minutes: 30`, `MPICH_VERSION: "4.2.3"`,
   `actions/cache@v5`) and structural counts of jobs / steps / workflows / scripts / consumers / copies (`8 of 14
   jobs`). A `timeout-minutes` value is a CONFIGURED CEILING; quoting it as a duration ("the job takes 30
   minutes") is a rejected claim (`perf-number`). Anything that would drive real perf work is written as a POINTER
   for the cross-cutting perf epic, never as a measured claim of this station.
8. **Anchors stay inside the sweep list.** A candidate anchored under `crates/*/src` or `crates/*/tests` belongs to
   another station: the candidate is dropped (`out-of-scope-anchor`) and its title is logged so the owning station
   can be told. Cite a crate-source fact (for example `crates/cobre-solver/src/freeze.rs:22`) in `evidence`, in
   `docClaim` / `treeFact`, or in `citedContext` — never in `anchors`. `crates/*/Cargo.toml` and the three build.rs
   ARE in the sweep list.
9. **Fix shapes are prose.** `fixShape` describes the shape of a fix in sentences; never a diff, a patch, a code
   block, a `---`/`+++`/`@@` marker or a YAML snippet presented as a replacement. This evaluation ships no fixes.
10. **Never re-raise.** §Do-not-raise lists the settled, sanctioned, recorded and live items. A live entry owned by
    another station's gate (CD-061) may be SHARPENED only through `reRaiseOf: "CD-061"` with NEW evidence; an
    observation the inventory already recorded (§Do-not-raise → recorded) is cited, not re-emitted as a finding.
11. **Evidence is a command you ran.** Every candidate carries `evidence.command` (verbatim, runnable from the repo
    root against `077dbe2c`), `evidence.output` (trimmed) and `evidence.reading` (what the output proves). Counts,
    line numbers and names, not adjectives. §Frozen figures are pre-measured — cite them, do not re-derive them
    (re-deriving is allowed only to check the figure you cite).
12. **Reserved seams first (G-RESERVED).** Before ANY over-engineering, dead-config or unused-declaration candidate,
    check the mirror at the pin, `ARCHITECTURE.md` § Reserved crates (:108-125) and `CLAUDE.md` § Hard Rules
    (`Unwired config is reserved, not dead`), and record `reservedSeamsCheck: {checked: true, result:
    sanctioned|not-found|not-applicable, citation, rule}` on every over-engineering candidate. A ratified reserved
    seam (the five stub crates, the allowlist's self-declared emptiness, the `slow-tests` declarations if you judge
    their comment a sanction) goes in `positives` with `sanctionedBy`, never in `candidates`; a finding about a
    reserved crate is about workspace COST, never about deleting it.
13. **Drift is paired (drift lens only).** Every drift candidate carries BOTH `docClaim` (what the document asserts,
    with its own `{path, line}`) AND `treeFact` (what the tree shows, with its own `{path, line}`); each resolves at
    the baseline blob. `docClaim` / `treeFact` may point anywhere tracked at the pin (a crate source line is a legal
    tree fact); `anchors` still obey rule 8. One side only → rejected, code `drift-unpaired`. A claim that merely
    reads stale is not drift; name the tree location that contradicts it (the precedent for HOW a doc claim is
    falsified is `scripts/ci/check-doc-paths.sh` / `scripts/ci/check_doc_voice.py`).
14. **Empty is legitimate; blank is a bug.** If your lens is clean, return `candidates: []`, a `cleanVerdict` object
    naming the surfaces you examined and why they are clean, AND at least one `positives` entry. A lens with neither
    candidates nor positives is a failed lens. Put anything only the owner can decide in `_needsHuman` as one
    sentence each; prefer fewer, well-evidenced candidates over many weak ones.

## Alignment vocabulary (closed set; `target-layering-brief.md` §1–3)

`advances-0a` (the Engine seam and dispatch in cobre-cli / cobre-python; the `study` config block; shared output
orchestration in cobre-io; rank-0-executes MPI semantics; engine-tagged setup stages), `advances-0b` (carving
`cobre-model` out of the engine-neutral lp/), `advances-1` (purify the data model; the genericity rule extends to
the future L1 crates `cobre-model` / `cobre-network` ONLY by amending the CI grep's crate list — brief §1 L1 row,
roadmap IV.1), `neutral` (advances no phase), `conflicts` (the fix shape pushes an engine concept into L0/L1,
couples engines, places the `Engine` enum below L4, or builds a one-consumer abstraction — tag it and give the
roadmap-consistent alternative in `fixShape`). Cite the roadmap section in `alignmentCitation` (`Part IV.1 | IV.4 |
IV.5 | V.0 | V.1 | V.2`) whenever the hint is not `neutral`.

## Frozen figures (measured at the pin; the ticket's a136840d figures are superseded — cite, never re-derive)

| figure | ticket (a136840d) | pin (077dbe2c) | command |
| --- | --- | --- | --- |
| `check-infra-genericity.sh` header rationale (word boundaries avoid false positives on `execution`, `cuts_active`, `cutting_edge`) | :9-11 | :9-11 (unchanged) | `git show 077dbe2c:scripts/ci/check-infra-genericity.sh \| sed -n '9,11p'` |
| `SCAN_DIRS` — five crates: cobre-core, cobre-io, cobre-solver, cobre-stochastic, cobre-comm | :62-68 | :62-68 (unchanged) | `… \| sed -n '62,68p'` |
| `EXCLUDED_FILES=()` and its retirement rationale ("the value-function-artifact format break retired the former output/policy/ exemption. Add an entry here only with owner sign-off") | :74 / :70-73 | :74 / :70-73 (unchanged) | `… \| sed -n '70,74p'` |
| `PATTERN` (one engine's nouns: `Sddp|sddp|SDDP|Benders|Cut|cut|cut pool|cutting-plane|outer approximation|CutSync*|CutSelection*|MetadataCuts`) | :79 | :79 (unchanged) | `… \| sed -n '79p'` |
| Blind spot pre-seeded by the solver+comm station: `FreezeScratch.cut_nz_per_col` declared `crates/cobre-solver/src/freeze.rs:22`, production uses :137 :138 :141 :188 (cfg(test) boundary :244), evades `\bcut\b` because `_` is a word character; the gate exits 0 at the pin | — | as stated | `stations/solver-comm/partI-handoff.json → genericityGateBlindSpot` |
| `CLAUDE.md` genericity hard rule | :39-41 | :39-41 (unchanged) | `git show 077dbe2c:CLAUDE.md \| sed -n '39,41p'` |
| `ci.yml` `quality-scripts` job / `Infra genericity gate` step | :248 / :261-262 | :248 / :261-262 (unchanged) | `git show 077dbe2c:.github/workflows/ci.yml \| sed -n '248p;261,262p'` |
| `ci.yml` triggers `push: [main, develop]` / `pull_request: [main]` | :3-7 | :3-7 (unchanged) | `… \| sed -n '3,7p'` |
| `ci.yml` `env.MPICH_VERSION: "4.2.3"` | :12 | :12 (unchanged) | `… \| sed -n '12p'` |
| `Build MPICH from source` steps in `ci.yml` (Cache/Build/Set triples, byte-identical) | Build :49 :87 :130 :180 :316 :378 :469 :523 | Build :49 :87 :130 :180 :316 **:377 :468 :522**; triples span :43-68, :81-106, :124-149, :174-199, :310-335, :371-396, :462-487, :516-541 (26 lines each, `diff` of any two is empty) | `git show 077dbe2c:.github/workflows/ci.yml \| grep -n 'Build MPICH from source'` |
| Near-variants outside `ci.yml` | `mpi-slurm.yml:63`, `release-mpi.yml:75` | `mpi-slurm.yml:63` (triple :56-84: adds `set -euo pipefail`, drops the `LD_LIBRARY_PATH` export) and `release-mpi.yml:75` (triple :67-100: every step guarded by `matrix.mpi_source == 'source'`, a four-line ch3:nemesis comment, step named `Set MPICH environment (Linux)`, drops `LD_LIBRARY_PATH`) | `git show 077dbe2c:.github/workflows/mpi-slurm.yml \| sed -n '56,84p'`; `… release-mpi.yml \| sed -n '67,100p'` |
| `.github/` top level | `workflows/` only, no `actions/` | `workflows/` only | `git ls-tree --name-only 077dbe2c .github/` |
| `ci.yml` `timeout-minutes` ceilings (14) | :38 :76 :119 :169 :240 :251 :305 :353 :367 :409 :423 :439 :458 :508 | :38 :76 :119 :169 :240 :251 :305 **:352 :366 :408 :422 :438 :457 :507** (values 30 60 30 60 10 15 30 30 30 15 15 20 60 45 — configured ceilings) | `… \| grep -n 'timeout-minutes'` |
| `ci.yml` jobs | 14 | 14 (`check test clippy clp fmt quality-scripts schemas docs-examples docs security deny license-notices coverage python`); the seven from `docs-examples` on moved up one line | `inventory.json → ci.ciJobs` |
| `crates/cobre-cli/build.rs` hand-mirror note / `TEMPLATE_FILES` | :8-9 / :16-28 | :8-9 / `const TEMPLATE_FILES` at :16 (unchanged) | `git show 077dbe2c:crates/cobre-cli/build.rs \| sed -n '8,9p;16p'` |
| `check-docs-examples.sh` `EXPECTED_INPUT_FILES=11` (a COUNT, not content) | :39 | **:32** (invariant 1 stated at :6-7; the assert at :114-115) | `git show 077dbe2c:scripts/ci/check-docs-examples.sh \| grep -n EXPECTED_INPUT_FILES` |
| `slow-tests = []` no-op declarations with the comment "no slow tests currently in this crate; declaration reserved for workspace consistency" | `cobre-cli/Cargo.toml:54-56`, `cobre-io/Cargo.toml:17-19` | `crates/cobre-cli/Cargo.toml:55-57` (**shifted +1**), `crates/cobre-io/Cargo.toml:17-19`; the four `slow-tests` declarations are cli, io, sddp, stochastic (`inventory.json → workspace.features.slowTestsDeclarations`); `CLAUDE.md:12` puts `slow-tests` in the workspace test command and `ci.yml:32` in `NON_SOLVER_FEATURES` | `git show 077dbe2c:crates/cobre-cli/Cargo.toml \| sed -n '55,57p'` |
| Five reserved stub members | `Cargo.toml:12-16` | `Cargo.toml:12-16` (unchanged); sources 1–6 lines each (table below); `ARCHITECTURE.md:108-125` declares them reserved; `CLAUDE.md:10` names them | `git show 077dbe2c:Cargo.toml \| sed -n '3,23p'` |
| `cobre-python` workspace exclusion | `Cargo.toml:21-23` | `Cargo.toml:21-23` (comment :18-20) | same |
| `allow-rationale-allowlist.txt` self-declaration | :1 "intentionally EMPTY", :15 "KEEP THIS FILE EMPTY" | :1 / :15 (unchanged); 21 lines, 0 active entries | `git show 077dbe2c:scripts/ci/allow-rationale-allowlist.txt` |
| `docs/design/README.md` status vocabulary (five values) / maintenance convention | :4-16 / :29-36 | :4-16 (values at :6 :8 :10 :12 :14) / :29-36 (heading :29, body :31-36) | `git show 077dbe2c:docs/design/README.md \| sed -n '4,16p;29,36p'` |
| README out-of-vocabulary rows | :26-27 "Implemented (retained pending fold into the live spec)" | :26 (`anticipated-fixed-post-horizon-commitments.md`), :27 (`external-scenarios-are-authoritative.md`) AND :24 (`testing-architecture.md`, "Partially adopted (§5.2); the rest Proposal") — three, not two | `… \| sed -n '24p;26,27p'` |
| Design docs indexed | README + 9 docs, all indexed | README + 9 docs, **8 indexed**: `docs/design/post-horizon-input-unification.md` has no row in the status table | `git ls-tree -r --name-only 077dbe2c docs/design` vs README rows :20-27 |
| `ARCHITECTURE.md` reserved crates / detached-crates sentence | :115-125 / :133 | :115-125 (bullets) under `### Reserved crates (not yet implemented)` :108; the detached sentence spans :132-135 (the `cobre` umbrella crate is named at :134 and described at :102-106) | `git show 077dbe2c:ARCHITECTURE.md \| sed -n '100,135p'` |
| Gate corpus | 17 gates + lib helper | 14 `.sh` + 2 `.py` + `scripts/ci/lib/comment_scan.sh` = 17 files (18 tree entries with the allowlist data file) | `git ls-tree -r --name-only 077dbe2c scripts/ci` |
| `quality-report.sh` invokes `check-comment-bloat.sh` | :132 | :132 (unchanged; `exit 0` at :134) | `git show 077dbe2c:scripts/ci/quality-report.sh \| sed -n '128,134p'` |
| Roadmap Part-I item 6 claim ("the genericity gate exempts the four policy files") | brief §4 row 6 | `plans/architecture-debt-audit/tools/target-layering-brief.md:91` at the pin (the roadmap file itself, `plans/generalizing/beyond-sddp-generalization.md`, is not tracked at the pin — cite the brief) | `git show 077dbe2c:plans/architecture-debt-audit/tools/target-layering-brief.md \| sed -n '91p'` |
| `schemas/policy.fbs` | cited by the register and the mirror | does not exist at the root; the FlatBuffers schema is `crates/cobre-io/schemas/policy.fbs` — an observation already recorded in `prior-register.md`, not a finding | `git ls-tree -r --name-only 077dbe2c schemas \| grep fbs` |

Five reserved stub members (`Cargo.toml:12-16`):

| stub | `Cargo.toml` line | sources (lines) |
| --- | --- | --- |
| `cobre-mcp` | 12 | `crates/cobre-mcp/src/main.rs` (6) |
| `cobre-tui` | 13 | `crates/cobre-tui/src/lib.rs` (1) |
| `cobre-flow` | 14 | `crates/cobre-flow/src/lib.rs` (1) |
| `cobre-uc` | 15 | `crates/cobre-uc/src/lib.rs` (1) |
| `cobre-emt` | 16 | `crates/cobre-emt/src/lib.rs` (1) |

## Gate-wiring census (measured; the merge joins every gate candidate against this table)

Classes: `blocking` = a `run:` step with no advisory marker in a workflow job (or `scripts/pre-commit`);
`advisory-by-design` = the `ci.yml` step comment says `ADVISORY` and the script ends in `exit 0` (sanctioned by
design — a candidate about such a script quotes both anchors and is never "the gate is broken");
`transitively-advisory` = no direct wiring site, invoked by an advisory-by-design script. Nothing at the pin is
`unwired`. The naive negative grep (`for f in scripts/ci/*.sh scripts/ci/*.py scripts/ci/lib/*.sh; do n=$(basename "$f"); grep -rq "$n" .github/workflows/ scripts/pre-commit || echo "UNWIRED: $n"; done`) reports exactly one name — `check-comment-bloat.sh` — and is
NOT sufficient evidence for `unwired`.

| script | class (census) | wiring anchor | exit-0 comment (advisory only) | note |
| --- | --- | --- | --- | --- |
| `scripts/ci/check-allow-rationale.sh` | blocking | `.github/workflows/ci.yml:292` (job `quality-scripts`, step `Check rationale on new suppressions (E4 gate)`) |  |  |
| `scripts/ci/check-comment-banners.sh` | advisory-by-design | `.github/workflows/ci.yml:284` (job `quality-scripts`, step `Advisory — in-fn box-drawing banners / N5 candidates (E7)`) | step comment `.github/workflows/ci.yml:282`; terminal `exit 0` at `scripts/ci/check-comment-banners.sh:174` |  |
| `scripts/ci/check-comment-bloat.sh` | unwired | none in `.github/workflows/` or `scripts/pre-commit`; invoked by `scripts/ci/quality-report.sh:132` (advisory-by-design) |  | carry as `transitively-advisory` with that anchor — never `unwired` on a negative grep |
| `scripts/ci/check-comment-line-refs.sh` | advisory-by-design | `.github/workflows/ci.yml:280` (job `quality-scripts`, step `Advisory — drift-prone line references in comments (E2)`) | step comment `.github/workflows/ci.yml:278`; terminal `exit 0` at `scripts/ci/check-comment-line-refs.sh:157` |  |
| `scripts/ci/check-comment-refs.sh` | blocking | `.github/workflows/ci.yml:276` (job `quality-scripts`, step `Check comment references (un-rottable-ref gate)`) |  |  |
| `scripts/ci/check-cut-selection-determinism.sh` | blocking | `.github/workflows/ci.yml:264` (job `quality-scripts`, step `Cut-selection determinism gate`) |  |  |
| `scripts/ci/check-doc-paths.sh` | blocking | `.github/workflows/ci.yml:274` (job `quality-scripts`, step `Check repo-relative doc paths/links`) |  |  |
| `scripts/ci/check-doc-placeholders.sh` | blocking | `.github/workflows/ci.yml:286` (job `quality-scripts`, step `Check for doc placeholders (E5 placeholder gate)`) |  |  |
| `scripts/ci/check-docs-examples.sh` | blocking | `.github/workflows/ci.yml:360` (job `docs-examples`, step `Verify init/run structural invariants`) |  |  |
| `scripts/ci/check-infra-genericity.sh` | blocking | `.github/workflows/ci.yml:262` (job `quality-scripts`, step `Infra genericity gate`) |  |  |
| `scripts/ci/check-no-past-inflows.sh` | blocking | `.github/workflows/ci.yml:272` (job `quality-scripts`, step `Check for past_inflows regression`) |  |  |
| `scripts/ci/check-no-plan-leaks.sh` | blocking | `.github/workflows/ci.yml:270` (job `quality-scripts`, step `Check for plan-structure leaks`); also `scripts/pre-commit:11` |  |  |
| `scripts/ci/check_doc_voice.py` | blocking | `.github/workflows/ci.yml:296` (job `quality-scripts`, step `Check doc voice (no hype / unpinned numbers)`) |  |  |
| `scripts/ci/check_python_parity.py` | blocking | `.github/workflows/ci.yml:294` (job `quality-scripts`, step `Check Python parity`); also `scripts/pre-commit:14` |  |  |
| `scripts/ci/check_schemas.sh` | blocking | `.github/workflows/ci.yml:339` (job `schemas`, step `Verify schema freshness`) |  |  |
| `scripts/ci/lib/comment_scan.sh` | blocking | `.github/workflows/ci.yml:268` (job `quality-scripts`, step `cfg(test) boundary drift guard (shared scan lib)`) |  | shared scan library, sourced by 4 gates; wired directly as the cfg(test) boundary guard |
| `scripts/ci/quality-report.sh` | advisory-by-design | `.github/workflows/ci.yml:300` (job `quality-scripts`, step `Advisory — code-quality hotspot report`) | step comment `.github/workflows/ci.yml:298`; terminal `exit 0` at `scripts/ci/quality-report.sh:134` |  |

Structural shape of the `quality-scripts` job (`ci.yml:248-300`), for the performance lens: `check-allow-rationale.sh`
(:292) is scoped to `git diff --unified=0 "$BASE"...HEAD` against `git merge-base HEAD origin/main` (script :57-60,
:365-370); the comment scans (`check-comment-banners.sh` :148, `check-comment-line-refs.sh` :118/:126,
`check-no-plan-leaks.sh` :180, `check-infra-genericity.sh` :127, `check-comment-bloat.sh` :47, `lib/comment_scan.sh`)
walk the whole tree with `find … -name '*.rs'`.

## Do-not-raise (settled, sanctioned, recorded, live)

**Do-not-touch (BACKLOG.md L1942-1943 at HEAD) — never emit:** CD-008 (retracted), PD-001 (refuted), PD-004 (deferred
pending a profile), the sanctioned reserved-seam / `#[allow]` census; the mirror's `Cleared` section; the resolved
generalization forks D1/D2/D7–D15.

**Sanctioned (positives, never candidates):** the five reserved stub crates (`ARCHITECTURE.md:108-125`, `CLAUDE.md:10`);
the entry-free allowlist (`scripts/ci/allow-rationale-allowlist.txt:1` and `:15`); the three advisory-by-design gates'
`exit 0` (the `ADVISORY` step comments at `ci.yml:278-279`, `:282-283`, `:298-299`); the `cobre-python` workspace
exclusion (`Cargo.toml:18-20` states why); `EXCLUDED_FILES=()` (its emptiness is the retired exemption, `:70-73`).

**Recorded already (cite; do not re-emit as a finding):** the `schemas/policy.fbs` path drift (register L2013 / L3182
at HEAD, mirror :614 at the pin) — the mirror row is epic 11's to write; the README out-of-vocabulary row at :24 and
the un-indexed `post-horizon-input-unification.md` are inventory facts the DRIFT lens turns into candidates (that is the
one place they may appear); the gate-corpus double count in the epic body.

**Live, owned elsewhere (sharpen only, `reRaiseOf`):**

| id | owner | what it already says | this station may add |
| --- | --- | --- | --- |
| CD-061 (L2545) | core-io gate, ratified 2026-09-08 | the register/oracle-coverage gap: training-loop column names the word-boundary gate deliberately cannot see; anchored `scripts/ci/check-infra-genericity.sh:79`; NOT an enforced-contract violation; widen the I.3-6 disposition, no rename now | the GATE half only — `PATTERN`'s vocabulary-versus-concept limitation and the `SCAN_DIRS` two-site pairing are build-ci facts; a vocabulary candidate about cobre-io's column names is CD-061 itself and is not re-emitted |
| CD-025 (L917), CD-027 (L1012), CD-039 (L1515), OD-009 (L1853) | other stations | cite this surface as evidence (`check_python_parity.py`, the mirror, `schemas/policy.fbs`) | nothing — the cites are evidence inside another station's entry |
| CD-075 (L3826) | solver-comm | crate-level `[lints]` replaces `[workspace.lints]` and no checker in `scripts/ci/` compares the copies (`crates/cobre-solver/Cargo.toml:37-47`, `crates/cobre-comm/Cargo.toml:35-45`) | the missing-checker half is already the entry's mechanism; do not re-emit a "no lints-table gate" candidate without `reRaiseOf: "CD-075"` and new evidence |
| CD-080 (L3886) | solver-comm | `cobre-solver/build.rs:30` / `:97` link-demand facts | nothing new at the build.rs level unless you have new evidence |
| TD-045 (L4717) | sddp | cites `invariance-shuffle.yml` | the workflow fact below (NH9) is this station's |

**Inbound handoffs to PROBE (not to dispose):** sddp NH9 — `.github/workflows/invariance-shuffle.yml:34` and `:55` build
the cobre-sddp parity binary with `--features slow-tests` / `"clp slow-tests"` and no `test-support` (architecture lens);
sddp NH40 — `CLAUDE.md:19` enumerates one `unsafe` island in cobre-sddp (`src/gemm.rs`) while `crates/cobre-sddp/src/hull/ffi.rs:27`
and `hull/mod.rs:82` `:152` `:192` are a second, and `crates/cobre-sddp/Cargo.toml:112-121` states the override (drift
lens: `docClaim` CLAUDE.md:19 ↔ `treeFact` hull); cli-python NH8 — the parity script and the python job are owned here
(`check_python_parity.py --max 0` at `ci.yml:294` and `scripts/pre-commit:14`; ROADMAP X4: the bindings' Rust tests
ARE run at `ci.yml:562-567`, the CI-invisibility premise is superseded; the replacement seed is `crates/cobre-python/Cargo.toml:18`
`doc = false`, which hides the bindings crate from `cargo doc` and its intra-doc-link check — architecture or drift
lens, your call); solver-comm R16 — `crates/cobre-comm/tests/local_conformance.rs:4` cites a `backend-testing.md`
that exists nowhere at the pin (`git ls-tree -r --name-only 077dbe2c | grep -c backend-testing.md` → 0; drift lens:
`docClaim` that line ↔ `treeFact` the `docs/design/README.md` index at :18-27).

## Lens: architecture — named probes

**P1 Genericity gate, crate list vs L0–L4 (two-site).** `SCAN_DIRS` (:62-68) hardcodes five crates: core, solver,
comm (L0), stochastic (L1), io (L2). The layering brief §1 L1 row says the genericity rule extends to the future
`cobre-model` / `cobre-network` ONLY by amending the CI grep's crate list; neither crate exists at the pin. Judge the
gate against the TARGET, not today's tree: covering them is a TWO-SITE edit — the rule prose at `CLAUDE.md:39-41`
(which enumerates the same five crates) AND `SCAN_DIRS` plus the `ci.yml` step at :261-262 (job `quality-scripts`,
:248). No register entry records the pairing. `alignmentHint: "advances-1"`, `alignmentCitation: "Part IV.1"`, fill
`twoSite`. `fixShape` AMENDS the script and the rule; a new gate framework is `conflicts`.

**P2 Genericity gate, vocabulary vs concept.** `PATTERN` (:79) enumerates ONE engine's nouns. A future `cobre-direct`
vocabulary would pass unflagged, and the solver+comm station pre-seeded a live evader: `FreezeScratch.cut_nz_per_col`
(`crates/cobre-solver/src/freeze.rs:22`, production uses :137 :138 :141 :188) — `\bcut\b` misses it because `_` is
a word character. MANDATORY: quote the script's OWN rationale at :9-11 (word boundaries are deliberate, to avoid
false positives on `execution`, `cuts_active`, `cutting_edge`) and then say why `cut_nz_per_col` is an engine concept
(a per-column cut non-zero count of the frozen LP) and not such a false positive. Omitting that rationale is a
strawman and the candidate is returned. Anchor the gate (`:79`, `:9-11`); cite the leak site in `evidence` and
`citedContext` (rule 8) — the FIX lives here, the vocabulary half is the solver-comm station's. `reRaiseOf: "CD-061"`
only if you frame it as sharpening the register entry; otherwise it is the new gate-half candidate.

**P3 Trigger asymmetry.** `ci.yml:4-5` `push: [main, develop]`, `:6-7` `pull_request: [main]`. A pull request into
`develop` gets no `ci.yml` run (only the post-merge push does). CLASSIFY with `enforcementStrength` and state the
consequence (which gates a develop PR bypasses until merge). Do NOT propose the trigger edit as a fix shape beyond
one prose sentence — the owner gate decides.

**P4 Hand-mirrored templates.** `crates/cobre-cli/build.rs:8-9` states `templates/1dtoy/` must be kept in sync with
`examples/1dtoy/` BY HAND (`const TEMPLATE_FILES` at :16). `scripts/ci/check-docs-examples.sh:32` pins
`EXPECTED_INPUT_FILES=11` — a COUNT, not content. Say what guards content equality tomorrow (`git grep` for any
`diff -r` / `cmp` between the two trees in `.github/workflows/` and `scripts/`), and propose the amend-in-place shape.

**P5 Workflow feature drift (NH9).** `invariance-shuffle.yml:34` / `:55` run the cobre-sddp parity test with
`slow-tests` only; state, from the workflow text and `crates/cobre-sddp/Cargo.toml` (`[features]`), whether the
`parity` test target needs `test-support` and what the run does without it. Classify the strength.

**P6 CI coverage shape (NH8 / X4).** The python job (`ci.yml:504-593`) builds the CLI (:560-561), runs the bindings' Rust
tests (:562-567) and the parity script (:294, blocking, also `scripts/pre-commit:14`). Probe the remaining gap the
roadmap seeded: `crates/cobre-python/Cargo.toml:18` `doc = false` versus the `docs` job's (:363) `cargo doc --workspace
--no-deps` at :399 with `RUSTDOCFLAGS: -Dwarnings` (:401) (the crate is workspace-excluded at `Cargo.toml:21-23`, so the docs job
never sees it either way). Name what is and is not checked; `n-a` or the docs job's strength as appropriate.

## Lens: over-engineering — named probes (rule 12 pre-check FIRST)

**P1 MPICH block (8× + 2 variants).** In `ci.yml` the Cache/Build/Set MPICH triple repeats 8 times — triples at
:43-68, :81-106, :124-149, :174-199, :310-335, :371-396, :462-487, :516-541, Build steps :49 :87 :130 :180 :316 :377
:468 :522 — and the blocks are BYTE-IDENTICAL (`diff <(git show 077dbe2c:.github/workflows/ci.yml | sed -n '43,68p')
<(… | sed -n '81,106p')` is empty; do it, do not assume). Two near-variants live outside: `mpi-slurm.yml:63` (triple
:56-84) and `release-mpi.yml:75` (triple :67-100, guarded by `matrix.mpi_source == 'source'`). `.github/` contains ONLY
`workflows/`; there is no `actions/`. `fixShape`: one composite action under `.github/actions/` with an input for the
matrix guard (and one for the `LD_LIBRARY_PATH` export the variants drop). TEN consumers — rule 6's one-consumer
objection does NOT apply; do not tag `conflicts` on that ground. List all eight Build-step lines and name the two
variants as variants, not copies.

**P2 No-op `slow-tests` features.** `crates/cobre-cli/Cargo.toml:55-57` and `crates/cobre-io/Cargo.toml:17-19` declare
`slow-tests = []` with the comment "no slow tests currently in this crate; declaration reserved for workspace
consistency". Decide: reserved seam (positives, cite the comment and `CLAUDE.md:12`'s workspace-wide
`--features "... slow-tests ..."` command, which fails on a member that lacks the feature only when passed with
`-p`, not with `--workspace`) or a feature flag that widens `NON_SOLVER_FEATURES` (`ci.yml:32`) for nothing. Two
separate candidates or two positives — one per manifest — each quoting the comment; fill `reservedSeamsCheck`.

**P3 Five reserved stub members.** `Cargo.toml:12-16` lists `cobre-mcp`, `cobre-tui`, `cobre-flow`, `cobre-uc`,
`cobre-emt`; their sources are 1–6 lines (table above). G-RESERVED binds: `ARCHITECTURE.md:108-125` declares them
reserved and `CLAUDE.md:10` names them. Any finding is about workspace COST (every `--workspace` build, test, clippy
and doc walk in the 14 jobs compiles them; `cargo-dist` / `cargo-deny` resolve them), never about deleting a reserved
crate; the alternative shape is prose (an opt-in `default-members` or an explicit member list in the heavy jobs). The
empty `cobre` umbrella crate (`ARCHITECTURE.md:102-106`, :134) is the cli-python station's — OUT of scope: name it in
neither `candidates` nor `positives`.

**P4 Entry-free allowlist.** `scripts/ci/allow-rationale-allowlist.txt:1` says "intentionally EMPTY", `:15` says "KEEP
THIS FILE EMPTY"; `check-allow-rationale.sh` still reads it. This goes in `positives` with `sanctionedBy` citing both
lines. A candidate proposing its deletion re-raises a sanctioned decision and violates G-NORERAISE.

**P5 Gate-corpus shape.** Seventeen files for the quality job: are any two gates the same scan with a different
pattern (the four `lib/comment_scan.sh` sourcers versus `check-comment-refs.sh` / `check-no-plan-leaks.sh`, which
open-code their own `find … -name '*.rs'` walks)? A consolidation candidate must name the shared library that already
exists (pull, don't push — a second scan library is `conflicts`); an observation that the corpus is simply large is
not a finding.

## Lens: drift — named probes (rule 13: every candidate pairs `docClaim` with `treeFact`)

**P1 Status vocabulary and delete-on-ship.** `docs/design/README.md:4-16` defines exactly five statuses (Live spec,
Decision record, Living register, Proposal, Design brief); `:29-36` says a proposal that ships is DELETED once its
content lands in its authoritative home. Rows `:26` (`anticipated-fixed-post-horizon-commitments.md`) and `:27`
(`external-scenarios-are-authoritative.md`) carry "Implemented (retained pending fold into the live spec)" — outside
the vocabulary AND against the convention; row `:24` (`testing-architecture.md`) carries "Partially adopted (§5.2);
the rest Proposal" — outside the vocabulary. Record the drift with both anchors (`docClaim` the vocabulary/convention
lines, `treeFact` the rows and the two files they point at), or dismiss it with a citation. Do NOT touch the mirror row
(`reserved-seams-and-deferred-debt.md`, `:22`) — epic 11 owns that file. The un-indexed
`docs/design/post-horizon-input-unification.md` (no row in `:18-27`) is a second pair: `docClaim` README `:4` ("this
index is the map") ↔ `treeFact` the file's own status line.

**P2 Crate-map claims.** Check `ARCHITECTURE.md:100-135` and the `CLAUDE.md:10` workspace line, claim by claim
against `Cargo.toml:3-23`: 13 members, `cobre-python` excluded (:21-23), five reserved stubs (:12-16), the `cobre`
umbrella crate's `[dependencies]`. Use `check-doc-paths.sh` / `check_doc_voice.py` as the precedent for HOW a doc
claim is falsified (a repo-relative path that does not resolve; a pinned count with no guard). A claim that merely
reads stale is not drift; name the tree location that contradicts it. A count that is TRUE today but unguarded is a
positive with a note, not a candidate (`.claude/rules/doc-integrity.md` §2 is the house rule, but `.claude/` is
outside the sweep — cite, do not anchor).

**P3 Part-I item 6 (re-verify, do not dispose).** The roadmap records a genericity-gate exemption for the cobre-io
policy checkpoint (`target-layering-brief.md:91`, Part I.3 item 6). At the baseline `check-infra-genericity.sh:74` is
`EXCLUDED_FILES=()` and `:70-73` states that the value-function-artifact format break retired it, so
`crates/cobre-io/src/output/policy/` is scanned normally and the gate exits 0. Emit BOTH: (a) a drift candidate
(`docClaim` the brief row :91 ↔ `treeFact` `:74`, anchors `:74` and `:70`, `partIRef: "I.3-6"`) and (b) a
`handoffs` record with `partIRef: "I.3-6"`, `claim`, `baselineEvidence` (:74 and :70), `stationVerdict:
"claim-stale"`, a `note` — and NO `disposition`, NO final `alignment` value: epic 9 decides. This evidence appears
in the drift envelope, never in the architecture one.

**P4 Inbound doc-drift handoffs.** NH40 (`CLAUDE.md:19` ↔ `crates/cobre-sddp/src/hull/ffi.rs:27`, `crates/cobre-sddp/Cargo.toml:112-121`)
and R16 (`crates/cobre-comm/tests/local_conformance.rs:4` ↔ no `backend-testing.md` anywhere at the pin). Pair each;
anchors on `CLAUDE.md` / `docs/design/README.md`, the crate-side line in `treeFact` or `docClaim`.

**P5 Rule-prose ↔ enforcement drift.** For each `CLAUDE.md` § Hard Rules bullet that names a script or a command
(:12 the test command, :19 unsafe islands, :39-41 genericity, :47 the `slow-tests` feature, :117 the schema-regeneration command),
does the enforcing surface still match the prose (feature string, crate list, file names)? Pair any mismatch.

## Lens: performance — INFORMATIONAL ONLY (rule 7)

MANDATE. This station measures nothing. No timing number, no ratio, no wall-clock figure, no PD id request, no
`cobre_reduzido_2` run. Tag every entry `"measured": "UNMEASURED"` and state a `mechanism` in words. The only
numerals allowed are line references, configured values quoted with their key, and structural counts of
jobs/steps/scripts/consumers.

**P1 CI wall-time shape.** The MPICH source build sits on 8 of `ci.yml`'s 14 jobs, behind `actions/cache@v5` keyed on
`mpich-${{ env.MPICH_VERSION }}-${{ runner.os }}-${{ runner.arch }}` (`env` :12; cache steps :45-48 and
siblings) — so the cost is a CACHE-MISS cost paid once per key change, and any claim that ignores the cache is wrong.
The 14 `timeout-minutes` ceilings (:38 :76 :119 :169 :240 :251 :305 :352 :366 :408 :422 :438 :457 :507) are
CONFIGURED CEILINGS, not observed durations; quote one as a duration and the entry is rejected (`perf-number`). The
structural finding, if any, is which jobs share a cache key and which cannot (matrix `runner.os` / `runner.arch`).

**P2 Gate-runtime shape.** Say STRUCTURALLY which `quality-scripts` entries walk the whole tree and which are scoped to
a merge-base diff (`check-allow-rationale.sh` :57-60 / :365-370 is diff-scoped; the six comment/plan/genericity scans
`find` every `.rs` under `crates/`; `check_python_parity.py` and `check_doc_voice.py` read fixed file sets). No
timings — the shape is the finding, and the pointer for the perf epic is "which scans could share one walk".

**P3 Build-graph shape.** Which of the 14 jobs compile the workspace from scratch versus restore
`Swatinem/rust-cache@v2` (grep `rust-cache` and `prefix-key`), and which jobs compile the five stub members and the
vendored submodules (`submodules: recursive` at the checkout steps) without needing them. Structural only.

## Envelope (frozen contract — the merge guard and the ingest ticket read exactly this)

```json
{
  "station": "build-ci",
  "stationLabel": "build-ci-docs",
  "subStation": "build-ci",
  "baseline": "077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c",
  "lens": "<LENS>",
  "candidates": [
    {
      "title": "one line naming the smell and its subject; no ID (IDs are assigned at calibration)",
      "anchors": [
        { "path": ".github/workflows/ci.yml", "line": 49 },
        { "path": "crates/cobre-cli/build.rs", "symbol": "TEMPLATE_FILES" }
      ],
      "enforcementStrength": "blocking|advisory-by-design|unwired|transitively-advisory|n-a (rule 4)",
      "script": "scripts/ci/<gate>.sh — gate candidates only, for the census join; otherwise null",
      "wiring": { "workflow": ".github/workflows/ci.yml", "job": "quality-scripts", "step": "Infra genericity gate", "line": 262 },
      "evidence": {
        "command": "git show 077dbe2c:scripts/ci/check-infra-genericity.sh | sed -n '62,68p'",
        "output": "…trimmed…",
        "reading": "what the output proves; counts, line numbers and names"
      },
      "proposedSeverity": "A|B|C",
      "fixShape": "prose only, never a diff; a rule fix names BOTH sites (rule 5); pull, don't push (rule 6)",
      "twoSite": { "prose": { "path": "CLAUDE.md", "line": 39 }, "script": { "path": "scripts/ci/check-infra-genericity.sh", "line": 62 }, "step": { "path": ".github/workflows/ci.yml", "line": 262 } },
      "alignmentHint": "advances-0a|advances-0b|advances-1|neutral|conflicts",
      "alignmentCitation": "Part IV.1|IV.4|IV.5|V.0|V.1|V.2 — required when alignmentHint != neutral",
      "partIRef": "I.3-6 | null",
      "reRaiseOf": "CD-061 | CD-075 | null (rule 10)",
      "reservedSeamsCheck": { "checked": true, "result": "sanctioned|not-found|not-applicable", "citation": "mirror section / ARCHITECTURE.md line / null", "rule": "which rule was checked" },
      "docClaim": { "text": "drift lens only: what the doc asserts", "path": "docs/design/README.md", "line": 31 },
      "treeFact": { "text": "drift lens only: what the tree shows", "path": "docs/design/README.md", "line": 26 },
      "measured": "performance lens: \"UNMEASURED\"; otherwise null",
      "mechanism": "performance lens: the structural cost mechanism in words, no numbers; otherwise null",
      "citedContext": [ { "path": "crates/cobre-solver/src/freeze.rs", "line": 22, "why": "a crate-side fact cited, not anchored (rule 8)" } ]
    }
  ],
  "positives": [
    { "subject": "scripts/ci/allow-rationale-allowlist.txt is entry-free", "why": "correct and worth protecting / sanctioned / examined and clean", "sanctionedBy": "scripts/ci/allow-rationale-allowlist.txt:1 and :15" }
  ],
  "handoffs": [
    {
      "partIRef": "I.3-6",
      "claim": "drift lens only: the roadmap claim being re-verified",
      "baselineEvidence": [ { "path": "scripts/ci/check-infra-genericity.sh", "line": 74, "shows": "EXCLUDED_FILES=() — empty" } ],
      "stationVerdict": "claim-stale|claim-holds",
      "note": "no disposition and no Alignment value — epic 9 decides"
    }
  ],
  "cleanVerdict": "null, or {\"why\": \"the surfaces examined and why the lens is clean\"} when candidates is empty (rule 14)",
  "_needsHuman": [ "a question only the owner can answer; empty list when none" ]
}
```

Severity: A = a gate that silently passes what its rule forbids, a workflow that skips a hard rule on a shipped
branch, or a structural block on the roadmap; B = real debt with a bounded fix and a named blast radius; C = local
quality. `proposedSeverity` is the attacker's rating; the house calibrates. Keys not in the schema are dropped at the
gate; `verdict` is not yours to set. Optional keys (`script`, `wiring`, `twoSite`, `partIRef`, `reRaiseOf`,
`reservedSeamsCheck`, `docClaim`, `treeFact`, `measured`, `mechanism`, `citedContext`, `handoffs`) are `null` /
omitted when the rule that requires them does not fire. The envelope is the whole content of your scratch file.

## Dispatch

Four workers, one per lens, dispatched concurrently in one message; each directive names `<LENS>` and the scratch
path `/tmp/build-ci-attackers/out/<LENS>.json`. The gate is `tools/validate-envelope.py --role attacker --station build-ci`
(shape) plus the five station guards (`anchor-form`, `out-of-scope-anchor`, `missing-enforcement`, `drift-unpaired`,
`perf-number`) and the census join. A shape-invalid envelope or an `anchor-form` return is re-dispatched exactly once
with the diagnostic quoted; a lens still invalid after the retry is recorded under `## Gaps` in `attacker-log.md`
and its envelope is discarded whole — never partially merged, never hand-repaired. `attacker-log.md` is the resume
point.
