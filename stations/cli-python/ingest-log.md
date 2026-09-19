# cli-python ingest log (candidates → verdicts)

Baseline: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`   Station: `cli-python`   Ingested: 2026-09-18

Inputs: the four merged lens files `candidates-{architecture,performance,over-engineering,test-bloat}.json`
(gated in E06-3; `attacker-log.md` is the attacker-side record), `wave-dispositions.json` (the four owned
live ids CD-025 / CD-029 / CD-002 / CD-009 and the not-owned set), `partI-handoff.json` (the I.5 queue),
`inventory.json` (writer boundary, parity enforcement, test surface), the committed mirror
`docs/design/reserved-seams-and-deferred-debt.md` at the pin, and `.claude/rules/*` at the pin. Every figure below is measured at
`077dbe2c` (`git show`), never on HEAD. Scratch: `/tmp/cli-ingest/ingest.py` (`screens`, then `assemble`),
defender inputs `/tmp/cli-defenders/in/<candidateRef>.json`, envelopes `/tmp/cli-defenders/out/<candidateRef>.json`.

Attacker envelopes re-validated at ingest: candidates-architecture.json: exit 0; candidates-performance.json: exit 0; candidates-over-engineering.json: exit 0; candidates-test-bloat.json: exit 0.

## Candidate census

Cells read `candidates / positives`.

| lens | S6a | S6b | S6c | total |
|---|---|---|---|---|
| architecture | 7 / 7 | 7 / 5 | 7 / 10 | 21 / 22 |
| performance | 8 / 8 | 2 / 8 | 8 / 10 | 18 / 26 |
| over-engineering | 6 / 8 | 6 / 13 | 2 / 6 | 14 / 27 |
| test-bloat | 8 / 6 | 10 / 7 | 5 / 6 | 23 / 19 |
| **total** | 29 / 29 | 25 / 33 | 22 / 32 | 76 / 94 |

Every one of the twelve lens × sub-surface cells holds at least one candidate; no no-finding line applies.

76 candidates received; candidateRef `<subSurface>-<lens>-<nn>` with nn the zero-based index among the
lens file's candidates of that sub-surface in file order (the scheme every downstream artifact keys on).
Required-field screen (claim-based — title + fixShape vocabulary, the same rule the E06-3 gate stamped by):
a CLI→engine coupling subject must carry `partIRef` I.5 / I.3-7; an output-orchestration or boundary/PrepPhase
subject must carry `waveRef` CD-025 / CD-029. Malformed: 0. The screen was first run
with evidence-wide regexes and flagged 7 false positives (a CD-025 symbol quoted as context in a candidate
about another defect); it was narrowed to the claim (title / fix-shape) before this run — the same lesson as
the gate's stamps.

## Anchor rejections

`anchor-probe.md` (register-shaped stub, one block per candidate) → `tools/check-anchors.py "INGEST ANCHOR PROBE — cli-python (2026-09, baseline)"
--register stations/cli-python/anchor-probe.md --baseline 077dbe2c --json`: **251 anchors checked, 0 failing** → anchor-rejected candidates: 0.
Every anchor is a declaration `path::symbol` (Rust `fn/struct/enum/trait/type/const/static/mod/impl` or a struct
field; Python `def`/`class` or a column-0 module constant — the `.py` form the E06-3 checker extension added) or
`path:line` for the two-line facade `crates/cobre/src/lib.rs`. Anchors the gate had already demoted to
`citedContext` (line-only or non-declaration hits) are listed in `attacker-log.md` → "Anchors demoted to
citedContext" and were not re-probed.

## Re-route and dup-of

Route screen: a candidate whose anchors all fall outside `crates/cobre-cli`, `crates/cobre-python`, `crates/cobre`
is re-routed to the owning station (`sddp`, `core-io`, `stochastic`, `solver-comm`) with a hand-off note, never
defended here; a candidate carrying at least one station anchor survives even when its other anchors point into
cobre-sddp (the CLI-side half of a coupling finding is this station's). **0 re-routed, 6 dup-of.**
The E06-3 gate had already dropped the out-of-station raises (`attacker-log.md` → "Prior-register screen and
re-routes"), so nothing reached ingest with a foreign anchor set. Retired / not-owned ids screened against the live
vocabulary: CD-001 (RETIRED, Cleared at the sddp gate), CD-003 Construction hop (RETIRED), CD-004 (sddp station,
ratified), CD-026 / CD-082 (not this station's) — no candidate re-opened one; `S6a-performance-05` argues a perf
mechanism on CD-001's sanctioned successor and carries an ingest note saying so.

| candidateRef | outcome | owner | title |
|---|---|---|---|
| — | re-route | none | no candidate's anchors fall entirely outside the station (every cobre-sddp anchor is paired with a cobre-cli / cobre-python one) |
| S6b-performance-00 | dup-of | `S6b-architecture-03` (this station) | cobre validate resolves the boundary-state requirements twice per invocation, fully deserializing the boundary policy ch |
| S6b-over-engineering-03 | dup-of | `S6b-architecture-04` (this station) | Seven doc examples in the CLI shell are written against two library roots that resolve nowhere, in a crate with no libra |
| S6b-test-bloat-04 | dup-of | `S6b-architecture-04` (this station) | Seven executable-marked doctests in cobre-cli never compile or run, and every one names a crate path that does not exist |
| S6b-over-engineering-05 | dup-of | `S6b-architecture-05` (this station) | The summary module publicly re-exports six engine types under a rationale whose final claim is false, and three of the s |
| S6b-test-bloat-08 | dup-of | `S6b-architecture-05` (this station) | Three of the six engine types the summary module re-exports have zero production uses and exist only so its inline test  |
| S6b-over-engineering-01 | dup-of | `S6b-test-bloat-01` (this station) | Four `#[cfg(test)] pub fn format_*_string` renderers duplicate their `print_*` twins line for line, in the same module w |

**Prior-register restatement screen** (no `reRaiseOf`, shared anchor symbol or title overlap with a live owned id):

| candidateRef | live id | shared symbols | title Jaccard | decision |
|---|---|---|---|---|
| S6a-architecture-04 | CD-002 | BroadcastConfig, load_case_and_config | 0.04 | kept distinct — shares BroadcastConfig / load_case_and_config with CD-002 (positional tuple) and CD-004 (the projection twin, sddp station, ratified) — the claim here is the CALLER shape (the in-process path also reaches StudyParams::from_config only through the wire type); C |
| S6a-performance-05 | CD-002 | broadcast_and_build_setup, reconstruct_stochastic_context_non_root | 0.04 | kept distinct — CD-001 (the CLI's hand-rolled mirror of the rank-0 stochastic pipeline) is RETIRED — resolved by b051c410 and ratified Cleared at the sddp gate 2026-09-18; `reconstruct_stochastic_context_non_root` is the sanctioned thin caller of the single owner `build_stoch |
| S6a-performance-06 | CD-009 | apply_training_policy, load_policy_for_simulation | 0.12 | kept distinct — shares apply_training_policy / load_policy_for_simulation with CD-009 (the triplicated policy_dir guard, live, C) — a distinct claim: every rank rebuilding the whole checkpoint in simulation-only mode is a collective-cost mechanism, not the guard duplication;  |
| S6a-over-engineering-00 | CD-002 | BroadcastConfig, broadcast_and_build_setup | 0.03 | kept distinct — shares BroadcastConfig / broadcast_and_build_setup with CD-002 and CD-004; the projection CONTENT is CD-004's (sanctioned, mirror :563) — only the module's placement under commands/ is on trial |

Restatements folded onto a live id without a defender (`priorRelation: restates`): 0.

## Cleared (sanctioned)

Reserved-seams screen over every over-engineering candidate, against the committed mirror at the pin (Reserved-seam
register, Verified NOT reserved, Cleared, the `#[allow(...)]` census), CLAUDE.md `Unwired config is reserved, not
dead`, ARCHITECTURE.md's crate map and the register. **Cleared at ingest: 0** — every over-engineering candidate
argues a placement, a premise or a residue the sanctioning entry does not cover, so the screen result travels to
the defender as `reservedSeamsScreen` and the closed `sanctionedBy` set (six strings, `defender-prompt.md` rule 5)
is applied there (`dismissalBasis: sanctioned-seam`, counted under Defender summary).

| candidateRef | title | attacker check | ingest result | note |
|---|---|---|---|---|
| S6a-over-engineering-00 | commands/broadcast.rs is a 1,088-line postcard wire-type module filed under commands/, who | sanctioned | no-sanctioning-entry | the projection CONTENT is sanctioned (mirror :563 `Setup config-projection sprawl + CLI non-root reconstruction`, CD-004 twin) — the attacker's own check says so; the PLACEMENT of a wire-type module under commands/ has n |
| S6a-over-engineering-01 | WriteTrainingArgs carries hydro_models next to setup, and at its one construction site the | not-applicable | no-sanctioning-entry | searched: Reserved-seam register (:30), Verified NOT reserved (:125), Cleared (:1049), the census (:1272), CLAUDE.md `Unwired config is reserved, not dead`, ARCHITECTURE.md crate map — no entry names mod.rs, outputs.rs o |
| S6a-over-engineering-02 | write_sim_outputs_on_root is a decision-free pass-through layer, and the root guard its na | not-applicable | no-sanctioning-entry | the subject is CD-025's shape (mirror :587) — a live sharpen, not a seam; defended with priorContext |
| S6a-over-engineering-03 | The per-artifact write guards are mirrored across the CLI/Python boundary by four hand-wri | not-applicable | no-sanctioning-entry | the subject is CD-025's shape (mirror :587) — a live sharpen, not a seam; defended with priorContext |
| S6a-over-engineering-04 | Thirteen production #[allow] suppressions in the cell carry no written rationale, includin | not-found | no-sanctioning-entry | the `#[allow(...)]` census (mirror :1272) sanctions the Load-bearing class ON THE PREMISE that each site carries a `// Rationale:` comment (and `.claude/rules/comments.md` D4 scopes the mandatory rationale to a closed li |
| S6a-over-engineering-05 | RunSummary is declared pub(crate) but has no consumer outside its own file, unlike the Sim | not-applicable | no-sanctioning-entry | searched: Reserved-seam register (:30), Verified NOT reserved (:125), Cleared (:1049), the census (:1272), CLAUDE.md `Unwired config is reserved, not dead`, ARCHITECTURE.md crate map — no entry names run.rs or the item t |
| S6b-over-engineering-00 | The prep-phase error ladder threads a redundant `json: bool` beside `stdout_sink` while th | not-applicable | no-sanctioning-entry | searched: Reserved-seam register (:30), Verified NOT reserved (:125), Cleared (:1049), the census (:1272), CLAUDE.md `Unwired config is reserved, not dead`, ARCHITECTURE.md crate map — no entry names validate.rs or the i |
| S6b-over-engineering-01 | Four `#[cfg(test)] pub fn format_*_string` renderers duplicate their `print_*` twins line  | not-found | no-sanctioning-entry | searched: Reserved-seam register (:30), Verified NOT reserved (:125), Cleared (:1049), the census (:1272), CLAUDE.md `Unwired config is reserved, not dead`, ARCHITECTURE.md crate map — no entry names summary.rs or the it |
| S6b-over-engineering-02 | Three subcommand entry points take their clap args by value and suppress `needless_pass_by | not-found | no-sanctioning-entry | the `#[allow(...)]` census (mirror :1272) sanctions the Load-bearing class ON THE PREMISE that each site carries a `// Rationale:` comment (and `.claude/rules/comments.md` D4 scopes the mandatory rationale to a closed li |
| S6b-over-engineering-03 | Seven doc examples in the CLI shell are written against two library roots that resolve now | not-found | no-sanctioning-entry | searched: Reserved-seam register (:30), Verified NOT reserved (:125), Cleared (:1049), the census (:1272), CLAUDE.md `Unwired config is reserved, not dead`, ARCHITECTURE.md crate map — no entry names banner.rs, error.rs, |
| S6b-over-engineering-04 | Nine production cast suppressions in the summary module carry no `// Rationale:` line, aga | not-found | no-sanctioning-entry | the `#[allow(...)]` census (mirror :1272) sanctions the Load-bearing class ON THE PREMISE that each site carries a `// Rationale:` comment (and `.claude/rules/comments.md` D4 scopes the mandatory rationale to a closed li |
| S6b-over-engineering-05 | The summary module publicly re-exports six engine types under a rationale whose final clai | not-found | no-sanctioning-entry | the `#[allow(...)]` census (mirror :1272) sanctions the Load-bearing class ON THE PREMISE that each site carries a `// Rationale:` comment (and `.claude/rules/comments.md` D4 scopes the mandatory rationale to a closed li |
| S6c-over-engineering-00 | Three cobre.model entity twins are self-labelled minimal stubs: full pyclass scaffolding p | not-found | no-sanctioning-entry | searched: Reserved-seam register (:30), Verified NOT reserved (:125), Cleared (:1049), the census (:1272), CLAUDE.md `Unwired config is reserved, not dead`, ARCHITECTURE.md crate map — no entry names model.rs or the item |
| S6c-over-engineering-01 | Allow-attribute proliferation in the bindings crate: the errors test module re-declares th | not-applicable | no-sanctioning-entry | searched: Reserved-seam register (:30), Verified NOT reserved (:125), Cleared (:1049), the census (:1272), CLAUDE.md `Unwired config is reserved, not dead`, ARCHITECTURE.md crate map — no entry names errors.rs, results.r |

**Facade rows** (`crates/cobre/src/lib.rs`, two doc lines; the crate map's reservation, no mirror row):

- positive from `over-engineering.S6b` — crates/cobre/src/lib.rs (the facade crate's emptiness, cited as context only): sanctioned by ARCHITECTURE.md:102-106 at the pin (`The umbrella crate. Currently an empty skeleton ... reserved for a future single-dependency convenience re-export`); no mirror entry exists (prior-register, Normative). Not a candidate; the residual owner question is carried as needs-human.

- positive from `over-engineering.S6c` — crates/cobre/src/lib.rs (line 1) and crates/cobre/Cargo.toml -- the umbrella facade crate: sanctioned by ARCHITECTURE.md:102-106 and ARCHITECTURE.md:134 (the crate map declares the reservation); prior-register.md:46 (the reserved-seams precondition that routes the residual question to the owner gate). Not a candidate; the residual owner question is carried as needs-human.

- needs-human from `over-engineering.S6b` — The facade crate: keep `crates/cobre` as a documented reserved seam or retire it until the re-export actually exists? It is sanctioned by the crate map and has no mirror entry, and while it re-exports nothing the two `use cobre::...` doc examples in banner.rs and templates.rs cannot be made to resolve.

- needs-human from `over-engineering.S6c` — The umbrella facade crate crates/cobre: keep it as a documented reserved seam and add the missing row to docs/design/reserved-seams-and-deferred-debt.md as an E11 mirror write-back, or retire the crate from the workspace members list until the convenience re-export it reserves actually exists? Two facts bear on the decision and neither is the attacker's to weigh. First, ARCHITECTURE.md:102-106 documents the reservati

- verdict: the facade reservation is `Umbrella crate reserved for a future single-dependency convenience re-export`
  (ARCHITECTURE.md:102-106 at the pin) in the closed set; the mirror has NO row — an E11 write-back the log owes;
  the keep-or-retire question goes to the owner gate as `_needsHuman`, never pre-judged here.

**`commands/broadcast.rs` row** (`S6a-over-engineering-00`): the projection CONTENT is CD-004's, sanctioned at mirror :563 and ratified at the sddp gate; only the module's PLACEMENT under `commands/` is on trial → defended, not cleared.

**Superseded premises** (never a sanction, never a finding — a candidate resting on one is dismissed `premise-false-at-pin`):

- Python-binding Rust tests invisible to CI — mirror :347 at the pin; superseded — ci.yml:562-567 runs `cargo test --manifest-path crates/cobre-python/Cargo.toml`

- the parity gate sees 4 of 17 writers (use-line skip + allowlist) — superseded by fc81427a — import-resolving script, 18 shared names, floor 18

- cobre-python/src/io.rs has zero boundary references — superseded by fc81427a — phase 11 of cobre.io.validate (io.rs:167, :305-325)

- the report / summary subcommands — removed by 797ba443 — commands/report.rs and commands/summary.rs do not exist at the pin

## Dup-of merges

Intra-station folds decided from the overlap analysis (shared anchor symbols ≥ 2, or a shared symbol with title
Jaccard ≥ 0.2, or Jaccard ≥ 0.3): **6 candidates folded** into a surviving twin
(`priorRelation: intra-station`, no defender, verdict null). The survivor's defender input carries every folded
text under `mergedFrom` (title, anchors, evidence, fixShape, severity, measurementRequest) and its verdict must say
which parts of each survive; a folded `reRaiseOf` / `waveRef` is inherited by the survivor (`S6b-architecture-03`
inherits CD-029 sharpens from `S6b-performance-00` and carries its 4t measurementRequest on `perfQueue`).

| folded | lens | into | lens | why |
|---|---|---|---|---|
| S6b-performance-00 | performance | S6b-architecture-03 | architecture | same defect from two lenses: cobre validate resolves the boundary-state requirements and reads the boundary checkpoint twice per invocation (architecture: duplicated engine work + the Python mirror does it once; performance: the deserialisation cost); the architecture framing is the broader claim, the performance measurementRequ |
| S6b-over-engineering-03 | over-engineering | S6b-architecture-04 | architecture | same finding: the seven doc examples across the CLI shell import crate paths that resolve nowhere because the crate has no lib target — architecture (no doctest ever compiles), over-engineering (written against two library roots) and test-bloat (seven executable-marked doctests never run) are three lenses on one defect; the arch |
| S6b-test-bloat-04 | test-bloat | S6b-architecture-04 | architecture | same finding as S6b-over-engineering-03: the seven never-compiled doctests in a crate with no lib target; folded onto the architecture survivor with its anchors |
| S6b-over-engineering-05 | over-engineering | S6b-architecture-05 | architecture | same subject: summary.rs's `pub use cobre_sddp::{6 types}` re-export — architecture (three provenance types imported into production scope only for the module's own tests, under allow(unused_imports)), over-engineering (the re-export rationale's final claim is false; three of six unused) and test-bloat (three have zero productio |
| S6b-test-bloat-08 | test-bloat | S6b-architecture-05 | architecture | same subject as S6b-over-engineering-05: the summary.rs engine-type re-export with three unused members; folded onto the architecture survivor |
| S6b-over-engineering-01 | over-engineering | S6b-test-bloat-01 | test-bloat | same subject: the four #[cfg(test)] pub fn format_*_string renderers that duplicate their print_* twins line for line — over-engineering names the duplication, test-bloat names the consequence (tests assert the twin, so the shipped renderer can drift while tests stay green), which is the sharper claim; folded onto the test-bloat |

## Re-raise rejections

`tools/check-reraise.py "INGEST RERAISE PROBE — cli-python (2026-09, baseline)" --register <stub> --baseline 077dbe2c --json` over the 70 live candidates (in-station or alignment-xref, anchor resolved): exit 0, **0 unjustified re-raise hits** → re-raise rejections: 0. Candidates whose attacker filed a
`retiredOverlaps` row were rendered with a `Re-raise-of:` justification (a sharpen of a live cli-python
disposition or an anchor-overlap with a mirror section that records the very finding), which the checker accepted.

## Blocked pending measurement

Measure-then-claim: 2 candidates claim a parity-gate enforcement gap (`S6a-architecture-06`, `S6a-over-engineering-03`); both were admitted because every quantity the claim rests on was measured at
ingest (next section). **Blocked: 0.**

## Measure-then-claim

`enforcement-measurements.json` (re-measurable; `measuredOn` records the tree). Source layer: `python3 scripts/ci/check_python_parity.py --max 0 --root .` → exit
0, "OK: 0 parity mismatch(es) (max allowed: 0). 18 write functions in both paths."; per-side name sets 18 / 18, in both
18, floor 18, cli-only [], python-only []; invisible to the
gate by name: ['write_checkpoint', 'write_scenario'] (measured external writer names absent from the gate's per-side sets BY NAME, each covered under another name: write_chec); CI step .github/workflows/ci.yml:294 (quality-scripts job).
Writer surface: CLI 39 `write_*(` call sites (16 terminal `write_line`,
23 writer sites), Python 23 (5 `_if_any` helpers);
external writer names 17 / 17, identical
True — the ticket's a136840d figures (41 / 29 / 26) are superseded.
Runtime layer: `crates/cobre-python/tests/test_cli_python_file_set_parity.py` → **passed** (5 passed, 0 skipped) with
target/release/cobre (present, built 2026-09-17); guard conftest.py::cli_binary (session fixture) → _cobre_cli.resolve_cli_binary(repo_root, required=--require-cli-binary): absence of target/release/cobre or target/d…; ciExecutes **yes** — .github/workflows/ci.yml job `python` (:505): `maturin develop --release` (:559), `cargo build --release -p cobre-cli` (:561, step 'Build the CLI for the parity tests'), then `pytest crates/cobre-python/tests/ … --require-cli-binary` (:572) — the runtime layer…
Third layer: `crates/cobre-cli/tests/python_parity_check.rs::python_parity_script_passes` (cargo test --workspace (and the `test` CI job); skip only when python3 absent (eprintln + return)).
CI visibility of the cobre-python Rust tests: 19 `#[test]` ({'errors.rs': 3, 'policy.rs': 1, 'run.rs': 13, 'schema.rs': 2}), executed by
.github/workflows/ci.yml:562-568 `Run Rust tests for the bindings crate` (matrix python 3.12) — CI-VISIBLE; the ticket's 'never executes … no job runs cargo test for it' is superseded (see inventory.j… — the ticket's "22 never execute" is superseded.
Ticket premise "the runtime layer may skip in CI" is therefore superseded at the pin: CI builds the CLI and passes
`--require-cli-binary`; the ticket's `_cli_binary()` is the a136840d spelling of `conftest.py::cli_binary` /
`_cobre_cli.resolve_cli_binary`. Defenders of the two claims copy these figures into `measuredEvidence`
(rule 12) and never re-run anything.

## Contract dismissals

Defender dismissals resting on a pinned rule (`dismissalBasis: contract`, `contractCited` = the rule heading or
bullet): **2**. Ingest policy: a dismissal may also rest on `premise-false-at-pin`, `deliberate-and-documented`,
`sanctioned-seam` (closed set) or `cost-accepted-by-rule` — every basis carries a `basisCitation`; the ticket's
"cite sanctionedBy" wording is the sanctioned-seam case only.

| candidateRef | contractCited | basisCitation |
|---|---|---|
| S6a-over-engineering-02 | CLAUDE.md § Hard Rules — `clippy::all` and `clippy::pedantic` at `warn` level, zero warnings in CI | clippy.toml:1 (too-many-lines-threshold = 150), with Cargo.toml:65-67 and Cargo.toml:71 |
| S6b-over-engineering-04 | `.claude/rules/comments.md` D4 — Rationale above suppression: "Every `#[allow(...)]` for a refactor-decision lint (`clippy::too_many_arguments`, `too_many_lines`, `type_complexity`, `dead_code`, `unused_*`) and every borrow-checker workaround carries a rationale: why the refactor that removes the lint is inappropriate." | .claude/rules/comments.md:365 |

`contractCited` present on 27 verdict(s): `S6a-architecture-02` → CLAUDE.md § Hard Rules — Python parity, `S6b-architecture-00` → the `cobre validate` module contract (`commands/validate.rs` module doc: exit 0 => `cobre run` will not fail before the solver iterates; `--json` stdout is exactly one object), `S6b-architecture-01` → cobre validate module contract (crates/cobre-cli/src/commands/validate.rs module doc: --json stdout is exactly one object), `S6b-architecture-02` → CLAUDE.md § Hard Rules — Python parity, `S6b-architecture-04` → CLAUDE.md § Hard Rules — Comment discipline — default-off, `S6b-architecture-05` → .claude/rules/comments.md D4 — Rationale above suppression, `S6c-architecture-02` → .claude/rules/comments.md § Contract-mirroring beats DRY (but a mirror is shape, never a number), `S6c-architecture-05` → CLAUDE.md § Hard Rules — Python parity, `S6c-performance-02` → CLAUDE.md § Hard Rules — Never allocate on hot paths — pre-allocate workspaces, reuse buffers, `S6a-over-engineering-00` → CLAUDE.md § Hard Rules — Infrastructure crate genericity, `S6a-over-engineering-02` → CLAUDE.md § Hard Rules — `clippy::all` and `clippy::pedantic` at `warn` level, zero warnings in CI, `S6a-over-engineering-03` → CLAUDE.md § Hard Rules — Python parity, `S6a-over-engineering-04` → .claude/rules/comments.md D4 - Rationale above suppression, `S6b-over-engineering-00` → the `cobre validate` module contract (`commands/validate.rs` module doc: exit 0 implies `cobre run` will not fail before the solver iterates; `--json` stdout is exactly one object), `S6b-over-engineering-02` → .claude/rules/comments.md D4 — Rationale above suppression: Every `#[allow(...)]` for a refactor-decision lint (`clippy::too_many_arguments`, `too_many_lines`, `type_complexity`, `dead_code`, `unused_*`) and every borrow-checker workaround carries a rationale, `S6b-over-engineering-04` → `.claude/rules/comments.md` D4 — Rationale above suppression: "Every `#[allow(...)]` for a refactor-decision lint (`clippy::too_many_arguments`, `too_many_lines`, `type_complexity`, `dead_code`, `unused_*`) and every borrow-checker workaround carries a rationale: why the refactor that removes the lint is inappropriate.", `S6c-over-engineering-01` → `.claude/rules/comments.md` D4 — Rationale above suppression. Every `#[allow(...)]` for a refactor-decision lint (`clippy::too_many_arguments`, `too_many_lines`, `type_complexity`, `dead_code`, `unused_*`) and every borrow-checker workaround carries a rationale: why the refactor that removes the lint is inappropriate., `S6a-test-bloat-00` → .claude/rules/testing.md § Cost discipline — Cobre links a solver into every test binary, `S6a-test-bloat-01` → .claude/rules/testing.md § Cost discipline — Cobre links a solver into every test binary, `S6a-test-bloat-03` → .claude/rules/testing.md § Tiers — use the cheapest tier that catches the regression, `S6b-test-bloat-00` → .claude/rules/testing.md § Tiers — use the cheapest tier that catches the regression, `S6b-test-bloat-02` → .claude/rules/testing.md § Cost discipline — Cobre links a solver into every test binary, `S6b-test-bloat-03` → .claude/rules/testing.md § Tiers — use the cheapest tier that catches the regression, `S6b-test-bloat-05` → .claude/rules/testing.md § Cost discipline — Cobre links a solver into every test binary, `S6b-test-bloat-06` → .claude/rules/testing.md § Cost discipline — Cobre links a solver into every test binary, `S6c-test-bloat-00` → .claude/rules/testing.md § Cost discipline — Cobre links a solver into every test binary, `S6c-test-bloat-02` → .claude/rules/testing.md § Cost discipline — Cobre links a solver into every test binary.

## Cross-lens overlaps — decisions

Every pair the overlap analysis surfaced (the `attacker-log.md` cross-cell table plus the cross-lens pairs only
ingest can see), with the decision: merged (→ Dup-of merges) or kept distinct with the reason.

| a | b | shared symbols | Jaccard | decision |
|---|---|---|---|---|
| S6a-architecture-01 | S6a-performance-02 | run_simulation_phase_py, write_simulation_outputs | 0.17 | kept distinct — the architecture claim is the reversed emission order of the four simulation writes across the two front ends (a hand-mirror drift, CD-025 sharpen); the performance claim is the ownership-taking cobre-io writer forcing both front ends to clone the whole simulation result — fixing the order does not  |
| S6a-architecture-01 | S6a-performance-03 | run_simulation_phase_py, write_simulation_outputs | 0.08 | kept distinct — reversed write order (architecture) versus the per-scenario reorder vector both front ends allocate before the path-row write (performance) — two mechanisms on one call chain; the perf sweep measures the allocation, the disposition ticket fixes the order |
| S6a-architecture-02 | S6a-over-engineering-03 | write_training_artifacts, write_training_outputs | 0.05 | kept distinct — both sharpen CD-025 from different angles: the architecture claim is the absence of a common composition unit for the training write set (nine inline CLI writes against one Python helper); the over-engineering claim is the four hand-written per-artifact guards mirrored across the boundary — the guar |
| S6b-architecture-00 | S6b-over-engineering-00 | run_boundary_check | 0.23 | kept distinct — the architecture claim is a contract defect (no `--json` error object for a boundary-phase failure); the over-engineering claim is a redundant `json: bool` threaded beside `stdout_sink` through the prep-phase error ladder — removing the redundant flag does not produce the missing object, emitting th |
| S6b-architecture-03 | S6b-performance-00 | execute, reconcile_boundary | 0.21 | merged: `S6b-performance-00` → `S6b-architecture-03` |
| S6b-architecture-04 | S6b-over-engineering-03 | CliError, find_template, render_banner_string, run_progress_thread | 0.29 | merged: `S6b-over-engineering-03` → `S6b-architecture-04` |
| S6b-architecture-04 | S6b-test-bloat-04 | CliError, find_template, run_progress_thread | 0.04 | merged: `S6b-test-bloat-04` → `S6b-architecture-04` |
| S6b-architecture-05 | S6b-over-engineering-05 | print_hydro_model_summary | 0.21 | merged: `S6b-over-engineering-05` → `S6b-architecture-05` |
| S6a-performance-02 | S6a-performance-03 | run_simulation_phase_py, write_simulation_outputs | 0.17 | kept distinct — two allocations in the same simulation write chain: the whole-result clone forced by the ownership-taking writer, and the per-scenario reorder vector — the perf sweep measures them separately at the candidate's layout; neither subsumes the other |
| S6b-performance-01 | S6b-test-bloat-05 | ProgressHandle, run_progress_thread | 0.07 | kept distinct — the performance claim is the progress thread retaining every received training event in a vector the simulation never drains (unbounded retention); the test-bloat claim is fifteen inline progress tests spawning a real renderer thread to re-assert formatting — same module, different defects |
| S6b-over-engineering-01 | S6b-test-bloat-01 | format_boundary_summary_string, format_hydro_model_summary_string, format_provenance_summary_string, format_setup_summary_string, training_summary_lines | 0.2 | merged: `S6b-over-engineering-01` → `S6b-test-bloat-01` |
| S6b-over-engineering-03 | S6b-test-bloat-04 | CliError, find_template, run_progress_thread | 0.04 | both folded into `S6b-architecture-04` |
| S6b-over-engineering-05 | S6b-test-bloat-08 | print_provenance_summary | 0.22 | both folded into `S6b-architecture-05` |

## Defender summary

70 survivors (76 received − 6 dup-of), one read-only Opus
`adversarial-defender` each, dispatched in parallel batches of six (a slot refilled as soon as its defender
returned) with `defender-prompt.md` read verbatim plus the single candidate input object
(`/tmp/cli-defenders/in/<candidateRef>.json`: the candidate, its prior context, merged twins, related ids,
reserved-seams screen, enforcement measurement, ingest notes and the attacker's needs-human items); envelopes
handed back through a scratch file outside the repository (`WRITTEN <bytes> <path>`). Every envelope was checked
with `tools/validate-envelope.py --role defender --station cli-python` plus the station clauses: exactly one
verdict keyed by the candidateRef; `survivingClaim` strictly narrower than the title (not equal after
normalisation, token-Jaccard < 0.85, ≥ 40 chars) and absent on a dismissal; a dismissal carries `dismissalBasis`
(vocabulary of five) + `basisCitation`, `sanctionedBy` from the closed set iff `sanctioned-seam`, `contractCited`
iff `contract`, `byteNeutral: n/a`; `measuredEvidence` complete iff the input carried `enforcementMeasurement`;
`byteNeutral` / `alignmentHint` in vocabulary; `conflicts` a boolean equal to (`alignmentHint == conflicts`) with
`conflictsRule` iff true; `partIRef` / `waveRef` copied verbatim; no diff or code block; no timing number on a
performance verdict. One re-dispatch on failure; a second failure records `unresolved` + `_needsHuman`.

**Retry history:**

- none — every envelope passed on the first dispatch.

Result: **50 confirmed, 20 dismissed, 0 unresolved**;
37 defended candidates carry a `_needsHuman` note for the owner gate; 0
verdicts set `conflicts: true`; 13 sharpen a live id (merged into it at calibration, no new number).

| lens | confirmed | dismissed | unresolved |
|---|---|---|---|
| architecture | 16 | 5 | 0 |
| over-engineering | 8 | 3 | 0 |
| performance | 7 | 10 | 0 |
| test-bloat | 19 | 2 | 0 |

Dismissal bases: sanctioned-seam 1, premise-false-at-pin 5, deliberate-and-documented 4, contract 2, cost-accepted-by-rule 8.

**Dismissed:**

- `S6a-architecture-04` (architecture) — The CLI run path reaches the engine's parameter projection only through the MPI wire type, unconditionally, and both front ends then enter t… — `sanctioned-seam`, `sanctionedBy`: Setup config-projection sprawl + CLI non-root reconstruction; docs/design/reserved-seams-and-deferred-debt.md:563. The mirror entry the candidate itself cites reserves exactly this caller shape. At docs/design/reserved-seams-and-deferred-debt.md:563 the open half is stated as 'the local StudyParams and the wire BroadcastConfig projection are still two structs kept in step by hand', the blocker is named (the post…

- `S6b-architecture-06` (architecture) — The two L4 classifications of cobre_sddp::SddpError are not jointly owned and only the CLI's is compiler-enforced, so a new engine variant i… — `deliberate-and-documented`; crates/cobre-python/src/errors.rs:13-17. The asymmetry is deliberate, documented at crates/cobre-python/src/errors.rs:13-17, and its load-bearing premise holds at the pin. ErrorSource::Sddp is constructed at exactly four production sites (run.rs:362, run.rs:428, run.rs:745, study.rs:387) and every one is a training or simulation phase help…

- `S6c-architecture-00` (architecture) — The bindings carry no Engine seam: Study and Policy are typed directly on six cobre_sddp types, and two separately registered public entry p… — `premise-false-at-pin`; crates/cobre-python/src/run.rs:1395-1444. Judging only the station delta the ingest note names (two separately registered public entry points over one engine), not the I.5 seam claim. The delta's load-bearing premise is that cobre.run.run and cobre.Study are two engine-reaching surfaces, each needing its own engine-admission site, and that…

- `S6c-architecture-03` (architecture) — Both front ends bypass cobre-io's two typed metadata readers in production and read the same files untyped, leaving the L2 reader exercised … — `deliberate-and-documented`; crates/cobre-python/src/results.rs:117-135 (load_results returns the metadata files' contents as a dict; ValueError only on malformed JSON) with crates/cobre-io/tests/metadata_back_compat.rs:1-7 (the metadata back-compat contract the typed readers assert). The two reads are not the same read, so there is no bypassed reuse. load_results is documented at crates/cobre-python/src/results.rs:117 and :123 to hand Python the CONTENTS of training/metadata.json and simulation/metadata.json, and it does exactly that: read_json_file (:86) parses to serde_json::V…

- `S6c-architecture-06` (architecture) — write_policy_checkpoint in the bindings re-declares the engine's cost-scale default as a bare literal while cobre-sddp exports the named con… — `premise-false-at-pin`; crates/cobre-sddp/src/policy/policy_load.rs:55-61. The load-bearing blast-radius claim does not hold at the pin. cobre-sddp declares TWO public constants at this value with opposite lifetimes: DEFAULT_COST_SCALE_FACTOR (crates/cobre-sddp/src/setup/params.rs:43), the movable default for an absent modeling.cost_scale_factor CONFIG FIELD, consumed only…

- `S6a-performance-00` (performance) — broadcast_value postcard-encodes and discards every payload on rank 0 even when the world has a single rank — `cost-accepted-by-rule`; crates/cobre-cli/src/commands/run/setup.rs:319-380 (the once-per-run broadcast region, timed into SetupTimings) with its single callers crates/cobre-cli/src/commands/run/mod.rs:150 and :167. The candidate's facts hold at the pin. In broadcast.rs, the is_root arm at lines 417-421 calls postcard::to_allocvec with no comm.size() guard, the encoded buffer is broadcast at lines 445-449, and the root arm at lines 451-455 returns the original value, so on a single-rank world the buffer is buil…

- `S6a-performance-01` (performance) — Both simulation gathers use allgatherv although only rank 0 consumes the gathered rows — `cost-accepted-by-rule`; crates/cobre-cli/src/commands/run/simulation.rs:155-158 — both gathers' only call sites, post-simulate teardown after the drain join at :113, with the bulk payload already streamed to Parquet per :80-81. Every factual premise holds at the pin. Both helpers allgatherv their payload (simulation.rs:363, :408) after exchange_gather_plan allgathers the per-rank length vector (:324), and the sole consumer of global_path_rows and global_scenario_stats is write_sim_outputs_on_root behind if ctx.is_root (:20…

- `S6a-performance-02` (performance) — The cobre-io path-row writer takes ownership, so both front ends clone the full simulation path-row set at the write call — `cost-accepted-by-rule`; CLAUDE.md § Architecture Guides — hot-path list (training/forward, training/backward, training/training.rs, simulation/pipeline.rs, training/lower_bound.rs); the write path is not in it. The premise is factually present at the pin and the dismissal rests on cost, not on premise: write_paths takes rows by value and sorts them in place (crates/cobre-io/src/output/simulation_writer.rs:1168-1172), while both front ends hold a borrow and copy at the call (crates/cobre-cli/src/commands/ru…

- `S6a-performance-03` (performance) — Both front ends allocate a per-scenario vector only to reorder two tuple fields before the scenario-summary write — `cost-accepted-by-rule`; CLAUDE.md § Hard Rules — "Never allocate on hot paths" (hot-path list in CLAUDE.md § Architecture Guides); the write site is rank-0 once-per-run at crates/cobre-cli/src/commands/run/simulation.rs:204. The named mechanism is one allocation per run, not per scenario, per stage or per call in a loop. The CLI reshape at outputs.rs:214-220 is reachable only through write_sim_outputs_on_root under the is_root guard (simulation.rs:204-205), after the collective aggregation has completed, and the Python…

- `S6a-performance-04` (performance) — Every per-scenario solver-stats delta is cloned, heap histogram included, only to drop one tuple field before packing — `cost-accepted-by-rule`; crates/cobre-cli/src/commands/run/simulation.rs:156. The mechanism is factually real at the pin and I am not disputing it. simulation.rs:397-402 builds local_stats_stripped by cloning every local (u32, i32, SolverStatsDelta) entry into a two-tuple purely to shed the opening field, and the only reason the clone exists is the packer signature at solver_…

- `S6a-performance-05` (performance) — Non-root ranks re-read the case directory and redo the hydro-model fit rank 0 already holds — `cost-accepted-by-rule`; crates/cobre-cli/src/commands/run/setup.rs:352-363 (sole production call of prepare_hydro_models, setup phase); crates/cobre-sddp/src/production/hydro_models/mod.rs:66-68. The mechanism is real but it is entirely setup-phase work, which the station's perf rule accepts by name (case load, hydro-model fit, validation). prepare_hydro_models has exactly one production caller at crates/cobre-cli/src/commands/run/setup.rs:360, inside broadcast_and_build_setup, which runs on…

- `S6a-performance-06` (performance) — In simulation-only mode every rank reads and rebuilds the whole policy checkpoint, while the sibling boundary-cut path in the same module br… — `cost-accepted-by-rule`; crates/cobre-cli/src/commands/run/mod.rs:228. load_policy_for_simulation has exactly one call site (crates/cobre-cli/src/commands/run/mod.rs:228), reached once per run in the non-training branch of execute_inner immediately before run_simulation_phase, so the directory listing, the per-stage reads, the cut rescale, the manifest validation and t…

- `S6a-performance-07` (performance) — The serial rank-0 training-output write chain sits on every rank's critical path with nothing overlapping it — `deliberate-and-documented`; crates/cobre-cli/src/commands/run/mod.rs:172-173. Dismissed on the in-code rationale at crates/cobre-cli/src/commands/run/mod.rs:172-173 ('Write training outputs before simulation so they persist even if simulation fails'), which the crate's failure model makes contingent on COMPLETION, not merely on ordering: execute at mod.rs:128-136 renders the…

- `S6b-performance-01` (performance) — The progress thread retains every received training event in a vector that the simulation caller discards — `deliberate-and-documented`; crates/cobre-cli/src/progress.rs:7-9 (module doc: both strategies collect every received event verbatim and return the full sequence via ProgressHandle::join). The receiver is taken by value at progress.rs:156 and moved into the spawned closure at :161, so once a progress thread exists the returned Vec is the only path by which any caller can observe the event stream; training.rs:107 then :111 depends on exactly that, feeding the joined Vec into build_trai…

- `S6c-performance-03` (performance) — load_results converts the same training metadata JSON tree into Python twice, once for the manifest key and once for the metadata alias — `cost-accepted-by-rule`; CLAUDE.md Hard Rules - 'Never allocate on hot paths' (hot-path scope list under CLAUDE.md Architecture Guides); crates/cobre-python/src/results.rs:147-197. Premise is true at the pin: crates/cobre-python/src/results.rs:184-185 call json_value_to_py on the same &metadata_val read at :160, so the recursive per-node conversion really does run twice. The mechanism is nonetheless once-per-call post-run work off every hot path, which rule 11's carve-out acce…

- `S6a-over-engineering-02` (over-engineering) — write_sim_outputs_on_root is a decision-free pass-through layer, and the root guard its name asserts lives at the call site instead — `contract`, `contractCited`: CLAUDE.md § Hard Rules — `clippy::all` and `clippy::pedantic` at `warn` level, zero warnings in CI; clippy.toml:1 (too-many-lines-threshold = 150), with Cargo.toml:65-67 and Cargo.toml:71. The candidate's load-bearing step — inline the seven-parameter wrapper into its single call site — is unavailable at the pin, because the project's own configured complexity bar is what produces the extraction. clippy.toml:1 sets too-many-lines-threshold = 150 and Cargo.toml:65-67 puts clippy::all a…

- `S6a-over-engineering-05` (over-engineering) — RunSummary is declared pub(crate) but has no consumer outside its own file, unlike the SimSummary declared 14 lines below it — `premise-false-at-pin`; crates/cobre-python/src/run.rs:1412. The grep in the candidate is accurate, but its load-bearing inference is not: the consumer that asks for crate visibility here is a signature, not a call site. RunSummary is the success arm of the return type of run_via_study, itself declared pub(crate) at crates/cobre-python/src/run.rs:1406 with th…

- `S6b-over-engineering-04` (over-engineering) — Nine production cast suppressions in the summary module carry no `// Rationale:` line, against the mirror's own Load-bearing class definitio… — `contract`, `contractCited`: `.claude/rules/comments.md` D4 — Rationale above suppression: "Every `#[allow(...)]` for a refactor-decision lint (`clippy::too_many_arguments`, `too_many_lines`, `type_complexity`, `dead_code`, `unused_*`) and every borrow-checker workaround carries a rationale: why the refactor that removes the lint is inappropriate."; .claude/rules/comments.md:365. The factual half of the candidate holds at the pin: the nine production cast suppressions exist as filed (summary.rs:602, 648, 713, 757, 777, 821, 851, 866, 870) and none is preceded by a Rationale line, while the test-scoped one at :1436 is correctly excluded. The normative half does not hold. The…

- `S6c-test-bloat-01` (test-bloat) — Schema export is asserted twice in two languages, with one test name byte-identical across the Rust and pytest suites — `premise-false-at-pin`; crates/cobre-python/src/schema.rs:69. The load-bearing premise (the Rust test is a strict superset of the pytest pair) does not hold at the pin. Both Rust tests call the plain Rust export with an already-built PathBuf (schema.rs:69, schema.rs:92), so they exercise none of the public surface the pytest tests reach: the wrap_pyfunction re…

- `S6c-test-bloat-04` (test-bloat) — testing-architecture section 5.11 still asks for the bindings crate's Rust tests to be wired into CI and section 5.2 still lists an unadopte… — `premise-false-at-pin`; docs/design/testing-architecture.md:430 (section 5.2, Adoption 2026-09-15: the bindings manifest does not yet name cobre-sddp with that feature). Both conjuncts of the title fail at the pin. The section 5.2 conjunct is refuted by the doc itself: the dated Adoption paragraph at docs/design/testing-architecture.md:425-431 states that the bindings crate's dev-dependencies do not yet name cobre-sddp with the test-support feature, and that this no…

**Sharpenings (merge into the prior id at calibration, no new number):**

| candidateRef | priorId | verdict | survivingClaim / basis |
|---|---|---|---|
| S6a-architecture-01 | CD-025 | confirmed | Delta over CD-025's surviving claim, narrowed on three counts. (1) The simulation write ORDER diverges one-sidedly: write_simulation_results emits simulation/metadata.json with status complete plus the empty simulation/_ |
| S6a-architecture-02 | CD-025 | confirmed | Delta over CD-025: OutputContext ownership is split with a consequence the register entry does not name, namely that the CLI caller captures completed_at before any write (run/mod.rs:170 into :183) while the bindings wri |
| S6a-architecture-03 | CD-002 | confirmed | Delta over CD-002: the carrier has a second positional level the register row does not record. `type LoadedCase` (setup.rs:68-75) is itself a six-slot positional tuple, constructed positionally at :139-146, re-destructur |
| S6a-architecture-06 | CD-025 | confirmed | Delta over the live hand-mirror claim: the CLI half of the mirror carries a falsified in-code rationale at outputs.rs:124-127, which tells the reader the echo writer at :134 must be called cobre_io-qualified rather than  |
| S6b-architecture-00 | CD-029 | confirmed | Delta over CD-029's surviving claim: the PrepPhase bypass has a machine-readable consequence in cobre-cli that the register does not record. Because the boundary step has no PrepPhase variant and hence no prep_phase_meta |
| S6b-architecture-03 | CD-029 | confirmed | Delta over CD-029: the two hand-mirrored boundary-reconcile copies are not the same body. The CLI copy re-resolves the boundary state requirements inside itself (validate.rs:232) although execute already holds the identi |
| S6a-performance-02 | CD-025 | dismissed | cost-accepted-by-rule |
| S6a-performance-03 | CD-025 | dismissed | cost-accepted-by-rule |
| S6a-performance-07 | CD-025 | dismissed | deliberate-and-documented |
| S6a-over-engineering-02 | CD-025 | dismissed | contract |
| S6a-over-engineering-03 | CD-025 | confirmed | Delta over CD-025's surviving claim: the hand-mirror's per-artifact EMIT CONDITION is carried by prose only - three CLI comments (outputs.rs:99-100, :110-111, :124-127) naming a Python twin, covering 3 of the 5 guarded t |
| S6b-over-engineering-00 | CD-029 | confirmed | Delta over CD-029's surviving claim: the fourth-phase bypass is observable at the `--json` surface, not only structural. Because `run_boundary_check`'s `--json` arm returns `CliError::from(err)` (validate.rs:299) with `a |
| S6a-test-bloat-07 | CD-025 | confirmed | Delta over CD-025's surviving claim, which covers only the writer hand-mirror: the census assertion now also exists in its owning crate at cobre-sddp/src/setup/node_graph.rs:3005 over a strictly more general fixture than |

**Conflicts (0), for the alignment epic:**

- none

**Measured evidence carried (2 enforcement-gap verdicts):**

- `S6a-architecture-06` — confirmed; names 18/18, call sites {'cli': 39, 'cliTerminalWriteLine': 16, 'python': 23}, runtime passed (ciExecutes yes), third layer `crates/cobre-cli/tests/python_parity_check.rs::python_parity_script_passes`.

- `S6a-over-engineering-03` — confirmed; names 18/18, call sites {'cli': 39, 'cliTerminalWriteLine': 16, 'python': 23}, runtime passed (ciExecutes yes), third layer `crates/cobre-cli/tests/python_parity_check.rs::python_parity_script_passes`.

Alignment over the defended set: advances-0a: 9, neutral: 61.

Byte-neutrality over the defended set: asserted: 27, n/a: 38, needs-rebaseline: 5.

**Needs-human (37 candidates), for the owner gate:**

- `S6a-architecture-00` — C1 (kept, narrowed): this verdict dismisses the no-op arm as per-front-end presentation, so if the owner holds that the CLI stderr line and the bindings' zeroed RunSummary are meant to be one behaviour, the surviving claim widens to cover that arm; both are public surface and neither writes a file, so the file-set parity test cannot decide it.

- `S6a-architecture-01` — C6 (carried from the attacker, still live after this read): the solver-stats fold cannot be homed at L2 as filed, because delta_to_stats_row and SolverStatsDelta are declared in cobre-sddp and imported by outputs.rs:29 and :35, so a cobre-io home would make L2 depend on an engine crate; the owner must choose between a cobre-sddp (L3) home for that mapping and leaving the two call sites duplicated while only the four 

- `S6a-architecture-02` — C6 (carried, narrowed): five of the nine training writes fold cobre-sddp-typed rows before writing (outputs.rs:32-38), so the owner must decide where those folds live before a single L2 entry point can own the whole set; an L2 home for the folds would make cobre-io depend on an engine crate, and a cobre-sddp home is outside the writer boundary the register calls conflicts.

- `S6a-architecture-04` — C5 (carried from the attacker): does the owner want the transport-flavoured name of cobre-sddp's sole StudySetup construction core (from_broadcast_params, entered by both front ends) filed as a rename against the station that owns cobre-sddp's public API, or left to the setup-layer redesign that the mirror :563 entry already triggers?

- `S6a-architecture-05` — Owner confirmation only, narrowed from the attacker's C6: confirm that this pure fold sits outside CD-025's writer set-plus-guards scope (mirror :587), so a cobre-sddp home next to its input types is not the held-for-owner L3 writer case. Evidence already on the record: solver_stats_log_to_rows (cobre-sddp solver_stats.rs:316) is shared by both front ends over the same type, and the cobre-sddp to cobre-io dependency 

- `S6b-architecture-00` — Which vocabulary supplies the boundary error object's user-visible `phase` string: the binding's literal `BoundaryReconciliationError` (cobre-python io.rs:305-328) or a fourth prep_phase_metadata row produced by the folded engine-owned phase? The two answers differ in a documented machine-readable field, and the second depends on CD-029's held destination.

- `S6b-architecture-01` — May the CLI's --json phase value CaseValidationError be retired in favour of the cobre-io/Python vocabulary? It is referenced nowhere outside its own declaration and doc (validate.rs:90 and :361), so the change costs nothing mechanically, but it is a user-visible field in a documented machine-readable contract and an unknown downstream consumer may match on it.

- `S6b-architecture-02` — May the CLI --json kind string CaseValidationError be retired in favour of the cobre-io vocabulary the bindings already use? It occurs only at validate.rs:90 and :361 and no test, script or binding matches on it, but it is a field in a documented machine-readable contract, so retiring it is an owner call, not a refactor detail.

- `S6b-architecture-02` — Where should a shared LoadError-to-kind classification live? cobre-io (L2) owns LoadError and both front ends depend on it, but the station's committed destination rule is written for output orchestration only; confirm that the L2 destination binds an error-classification map or name a different owner.

- `S6b-architecture-03` — Sequencing only, not the destination (already CD-029's owner question): patch this duplicate resolve as a standalone cobre-cli-local threading fix, or leave it to be absorbed when the boundary check folds into a phase that resolves the requirements once as a phase input?

- `S6b-architecture-04` — Should cobre-cli gain a [lib] target? Carried from the attacker because it still decides the residue's shape: with a library the four cobre_cli:: examples become compiler-verified, without one they must be de-annotated or de-fenced. Publishing the CLI's renderers, error enum and progress types as a maintained surface with no current consumer is the cost, so the choice is the owner's, not the station's.

- `S6c-architecture-00` — Coverage residue left by this dismissal: study.rs:583-585 / :694-696 admit that whole-tree byte-identity is unasserted for the cobre.Study methods, while the whole-tree goldens reach the same native calls only through run_via_study and the census golden compares selected columns of selected files. Owner call: does that warrant its own test-corpus coverage item, or is it already covered in substance by the shared sing

- `S6c-architecture-06` — Owner call: whether the terminal fallback at crates/cobre-python/src/policy.rs:510 should spell the already-exported cobre_sddp::LEGACY_COST_SCALE_FACTOR (reachable with no new dependency) and whether write_policy_checkpoint's doc should state the both-markers-absent case, since the field doc at policy.rs:104-107 points at that doc and the doc is silent on it.

- `S6a-performance-01` — Rooted gather at L0: closing the non-root replication needs a gatherv the cobre-comm Communicator trait does not expose (four collectives at traits.rs:64). Adding it grows the L0 FFI surface across ferrompi, LocalBackend, the factory enum and the documented precondition table for two call sites in one CLI file. Owner call: admit the primitive, or keep allgatherv and accept that non-root ranks rebuild a payload they d

- `S6a-performance-05` — Owner call, documentation only: should the shared-filesystem re-read on non-root ranks become a ratified deployment assumption with a docs/design entry? Two module docs state the behaviour (setup.rs:1-8, hydro_models/mod.rs:66-68) and no design doc records it, so the record is thinner than the assumption it rests on — not a perf defect at the pin.

- `S6a-performance-06` — Owner call: is the shared-filesystem re-read on non-root ranks a ratified deployment assumption? setup.rs:4-8 states the behaviour with no rationale and no docs/design entry covers it, and the answer decides whether the per-rank policy-checkpoint read is ever revisited at large rank counts.

- `S6a-performance-07` — Owner call: overlapping the rank-0 training-artefact write with rank 0 entering simulation trades the documented 'training outputs persist even if simulation fails' guarantee (mod.rs:172-173, contingent on completion because any rank's error reaches comm.abort at mod.rs:135) plus the stderr progress ordering for the stall. If the owner accepts that trade, the item re-enters the perf sweep at layout 2x2; otherwise the

- `S6c-performance-01` — API posture the station cannot settle: is the dict-shaped cobre.results.load_simulation a supported bulk reader, or is load_simulation_arrow the intended path for large result trees? load_simulation's rustdoc at results.rs:1081-:1126 never points at the Arrow reader, so the owner decides whether the per-column hoist is worth doing or whether the dict reader is documented as a small-result convenience instead.

- `S6c-performance-03` — Carried from the attacker and live only if the owner overrides this dismissal: is object identity between result['training']['manifest'] and result['training']['metadata'] part of the published cobre.results contract, or may one converted dict be bound to both keys (making the two views alias on mutation)?

- `S6c-performance-04` — A per-collection wrapper memo makes element object identity stable across reads (today system.buses[0] is system.buses[0] is False, since every read builds new objects). No test or stub pins that, so the owner must say whether per-read element identity is part of the published cobre.model contract before the memo shape is accepted.

- `S6a-over-engineering-00` — BroadcastNodeGraph (crates/cobre-cli/src/commands/broadcast.rs:256): keep as a documented reserved seam or retire? Verified at the pin that its only references workspace-wide are its own two From impls (:273, :319) and the inline tests at :583-710 — no production caller — and the mirror's opposite precedent for an identically shaped zero-caller postcard pair sits at docs/design/reserved-seams-and-deferred-debt.md:683

- `S6a-over-engineering-03` — Sequencing call: close the two uncovered emit conditions now with CLI-vs-Python coverage on an existing evaporation-modeling case plus a deviation-points-enabled case (pytest additions, no new integration binary, so cost discipline is not engaged), or wait for CD-025's shared cobre-io emit list to remove the duplicated condition entirely and gate it there?

- `S6a-over-engineering-04` — The committed mirror's allow census (docs/design/reserved-seams-and-deferred-debt.md:1324-1330) asserts that every Load-bearing numeric-cast and needless_pass_by_value allow carries a // Rationale: comment; twelve cast sites in this cell carry none, so the owner must decide whether the E11 write-back corrects that census prose to match D4's narrower closed list, or whether D4 is extended to cast lints (which would ma

- `S6b-over-engineering-00` — Should `cobre validate --json` emit an error object for a boundary-reconciliation reject? Today stdout is empty and stderr carries the self-referential hint that cli_validate.rs:219 forbids on the human path; no test pins either shape, so adding the object is a deliberate extension of the `--json` contract on an untested path and needs an owner call plus a new reject test in `--json` mode.

- `S6b-over-engineering-02` — The mirror's allow census (:1272, Load-bearing class) asserts that every refactor-decision allow including `needless_pass_by_value` carries a `// Rationale:`, which is false at schema.rs:49, validate.rs:329 and init.rs:53, while comments.md D4's closed list omits that lint: the owner must decide whether the census prose is corrected to match D4 or D4's list is extended to cover `needless_pass_by_value`, since that ch

- `S6b-over-engineering-04` — E11 census write-back owner call: correct the mirror's Load-bearing class clause (`each carrying a // Rationale: comment`) so it matches comments.md D4's closed lint list, which excludes numeric-cast lints, or else extend D4 to cast lints — the clause overclaims workspace-wide (496 cast allows under crates/*/src against 153 `Rationale:` lines in total), and until it is settled this candidate re-arises at every statio

- `S6c-over-engineering-00` — Direction for the three cobre.model twins (EnergyContract, PumpingStation, NonControllableSource): fill the getters to the house pattern the other four twins follow, or withdraw the three classes and have the PySystem getters return untyped mappings until a caller needs typed objects. The station confirms the asymmetry and the unpinned status; only the owner can pick which way it resolves, and the mirror needs the re

- `S6a-test-bloat-00` — Sequencing only: docs/design/testing-architecture.md already names cli_metadata as the cobre-cli domain binary that would absorb both files under its #[path] submodule mechanics, but that section is a Proposal, so the owner must decide whether to fold this pair now under the standing Cost-discipline rule or defer it to the Layer-1 grouping so the move is made once rather than twice.

- `S6a-test-bloat-01` — Sequencing: is a cobre-cli-local consolidation sanctioned now, or does it wait for the proposed Layer-1 migration in docs/design/testing-architecture.md §5.1, whose §6 phase 1 orders that consolidation first workspace-wide?

- `S6a-test-bloat-01` — Destination: rule 14 names tests/common/ as the cobre-cli helper home, while docs/design/testing-architecture.md §6 phase 3 proposes collapsing tests/common/ into cobre-sddp's test-support feature; an owner must pick one so the helper home is not built twice.

- `S6a-test-bloat-02` — Is the per-test-isolation cost acceptance recorded in the two module docstrings (test_study.py:12-14, test_run.py:9-11) standing policy for the pytest corpus? It is the only stated rationale, and if the owner keeps it as policy the shared read-only-Policy fixture is unwanted even for the five named tests.

- `S6a-test-bloat-04` — Does section 5.11's keep the pytest output-parity suite unchanged clause freeze the four output-parity modules against fixture hoisting, or does it only exempt them from the Layer-1 Rust binary migration while leaving harness hygiene in scope? The confirmed residue touches exactly the files that clause names.

- `S6a-test-bloat-05` — Owner call: does docs/design/testing-architecture.md section 5.11's 'keep the pytest output-parity suite unchanged' freeze the eleven parity modules against fixture hygiene, or only exempt them from the section 5.1 Layer-1 binary migration? That answer decides whether this assertion-preserving shared-fixture residue is actionable at all.

- `S6a-test-bloat-06` — Owner call: whether the one unique stats-row assertion (the simulation shape) should be rehomed next to the owner's existing solver_stats_log_to_rows test in cobre-sddp, or reshaped into a front-end assertion that covers both outputs.rs:197 and cobre-python run.rs:824 under the Python parity hard rule; the answer decides whether the residue is a pure move or a new assertion.

- `S6b-test-bloat-05` — The inline module's 608 test lines exceed the unratified ~500 test-LOC inline homing threshold in docs/design/testing-architecture.md 5.1 while its 22 fns sit under the ~40-fn threshold, so whether the consolidation lands inline or in an extracted src/progress/tests.rs sibling is the test-corpus station's ratification call, not this station's.

- `S6c-test-bloat-00` — Carried from the attacker: correcting the two rationales (convert.rs:104-106, test_convert.py:5-9) and adding a Unit-tier oracle are separable. The owner decides whether this lands as a rationale correction only, as an added cfg(test) module that keeps the seven pytest merge-landing assertions in place, or as both in one change; my read rules out deleting the seven pytest tests, because no other test asserts per-type

- `S6c-test-bloat-02` — Fix scope for the case-path split: crates/cobre-python/pyproject.toml:52 sets testpaths = ["tests"], which makes a bare pytest run from the crate directory a configured invocation the nine cwd-relative modules cannot resolve, while every module docstring and ci.yml:572 name the repo-root invocation -- the owner decides whether the residue normalizes all nine onto a resolved-from-__file__ case path or pins the repo-ro

- `S6c-test-bloat-03` — Anchor vocabulary for pytest files: five of this candidate's six anchors are pytest def names, which tools/check-anchors.py resolves only under its Rust declaration vocabulary; the owner decides whether the checker grows a def/class branch for .py paths or whether pytest anchors are demoted to evidence.

- `S6c-test-bloat-04` — Confirm the E11 write-back's scope covers every design-doc trace of the superseded invisibility premise, not only reserved-seams-and-deferred-debt.md:347: testing-architecture.md:101-102 still lists any cargo test run for the workspace-excluded bindings crate as Absent and asserts its Rust unit tests never compile in CI (the unit-test half is false at the pin, the doctest half still holds because the lib target sets 

## Per-candidate roster

Every one of the 76 candidateRefs, exactly once (mirrors `verdicts.json`; `state` is the station
test vocabulary — accepted / rejected-anchor / rejected-reserved-seam / rejected-defender / merged / unresolved —
derived from disposition + verdict + dismissalBasis).

| candidateRef | disposition | verdict | state | partIRef | waveRef | alignment | byteNeutral |
|---|---|---|---|---|---|---|---|
| S6a-architecture-00 | defended | confirmed | accepted | I.5 |  | advances-0a | asserted |
| S6a-architecture-01 | defended | confirmed (sharpens CD-025) | accepted | I.5 | CD-025 | advances-0a | asserted |
| S6a-architecture-02 | defended | confirmed (sharpens CD-025) | accepted | I.5 | CD-025 | advances-0a | needs-rebaseline |
| S6a-architecture-03 | defended | confirmed (sharpens CD-002) | accepted | I.5 |  | neutral | asserted |
| S6a-architecture-04 | defended | dismissed (sanctioned-seam) | rejected-reserved-seam | I.3-7 |  | neutral | n/a |
| S6a-architecture-05 | defended | confirmed | accepted | I.5 |  | neutral | asserted |
| S6a-architecture-06 | defended | confirmed (sharpens CD-025) | accepted |  | CD-025 | neutral | asserted |
| S6a-over-engineering-00 | defended | confirmed | accepted |  |  | neutral | asserted |
| S6a-over-engineering-01 | defended | confirmed | accepted |  |  | neutral | asserted |
| S6a-over-engineering-02 | defended | dismissed (contract) (sharpens CD-025) | rejected-defender | I.5 | CD-025 | neutral | n/a |
| S6a-over-engineering-03 | defended | confirmed (sharpens CD-025) | accepted | I.5 | CD-025 | advances-0a | asserted |
| S6a-over-engineering-04 | defended | confirmed | accepted |  |  | neutral | asserted |
| S6a-over-engineering-05 | defended | dismissed (premise-false-at-pin) | rejected-defender |  |  | neutral | n/a |
| S6a-performance-00 | defended | dismissed (cost-accepted-by-rule) | rejected-defender |  |  | neutral | n/a |
| S6a-performance-01 | defended | dismissed (cost-accepted-by-rule) | rejected-defender |  |  | neutral | n/a |
| S6a-performance-02 | defended | dismissed (cost-accepted-by-rule) (sharpens CD-025) | rejected-defender | I.5 | CD-025 | neutral | n/a |
| S6a-performance-03 | defended | dismissed (cost-accepted-by-rule) (sharpens CD-025) | rejected-defender | I.5 | CD-025 | neutral | n/a |
| S6a-performance-04 | defended | dismissed (cost-accepted-by-rule) | rejected-defender |  |  | neutral | n/a |
| S6a-performance-05 | defended | dismissed (cost-accepted-by-rule) | rejected-defender |  |  | neutral | n/a |
| S6a-performance-06 | defended | dismissed (cost-accepted-by-rule) | rejected-defender |  |  | neutral | n/a |
| S6a-performance-07 | defended | dismissed (deliberate-and-documented) (sharpens CD-025) | rejected-defender |  | CD-025 | neutral | n/a |
| S6a-test-bloat-00 | defended | confirmed | accepted |  |  | neutral | n/a |
| S6a-test-bloat-01 | defended | confirmed | accepted |  |  | neutral | n/a |
| S6a-test-bloat-02 | defended | confirmed | accepted |  |  | neutral | n/a |
| S6a-test-bloat-03 | defended | confirmed | accepted |  |  | neutral | n/a |
| S6a-test-bloat-04 | defended | confirmed | accepted |  |  | neutral | n/a |
| S6a-test-bloat-05 | defended | confirmed | accepted |  |  | neutral | n/a |
| S6a-test-bloat-06 | defended | confirmed | accepted | I.5 |  | neutral | n/a |
| S6a-test-bloat-07 | defended | confirmed (sharpens CD-025) | accepted | I.5 |  | neutral | n/a |
| S6b-architecture-00 | defended | confirmed (sharpens CD-029) | accepted | I.5 | CD-029 | advances-0a | needs-rebaseline |
| S6b-architecture-01 | defended | confirmed | accepted | I.5 |  | advances-0a | needs-rebaseline |
| S6b-architecture-02 | defended | confirmed | accepted | I.5 | CD-029 | advances-0a | needs-rebaseline |
| S6b-architecture-03 | defended | confirmed (sharpens CD-029) | accepted | I.3-7 | CD-029 | neutral | asserted |
| S6b-architecture-04 | defended | confirmed | accepted |  |  | neutral | n/a |
| S6b-architecture-05 | defended | confirmed | accepted | I.5 |  | neutral | asserted |
| S6b-architecture-06 | defended | dismissed (deliberate-and-documented) | rejected-defender | I.5 |  | neutral | n/a |
| S6b-over-engineering-00 | defended | confirmed (sharpens CD-029) | accepted |  | CD-029 | advances-0a | needs-rebaseline |
| S6b-over-engineering-01 | dup-of | dup-of → S6b-test-bloat-01 | merged |  |  | neutral |  |
| S6b-over-engineering-02 | defended | confirmed | accepted |  |  | neutral | asserted |
| S6b-over-engineering-03 | dup-of | dup-of → S6b-architecture-04 | merged |  |  | neutral |  |
| S6b-over-engineering-04 | defended | dismissed (contract) | rejected-defender |  |  | neutral | n/a |
| S6b-over-engineering-05 | dup-of | dup-of → S6b-architecture-05 | merged | I.5 |  | neutral |  |
| S6b-performance-00 | dup-of | dup-of → S6b-architecture-03 | merged |  | CD-029 | neutral |  |
| S6b-performance-01 | defended | dismissed (deliberate-and-documented) | rejected-defender |  |  | neutral | n/a |
| S6b-test-bloat-00 | defended | confirmed | accepted |  |  | neutral | n/a |
| S6b-test-bloat-01 | defended | confirmed | accepted |  |  | neutral | asserted |
| S6b-test-bloat-02 | defended | confirmed | accepted |  |  | neutral | n/a |
| S6b-test-bloat-03 | defended | confirmed | accepted |  |  | neutral | n/a |
| S6b-test-bloat-04 | dup-of | dup-of → S6b-architecture-04 | merged |  |  | neutral |  |
| S6b-test-bloat-05 | defended | confirmed | accepted |  |  | neutral | n/a |
| S6b-test-bloat-06 | defended | confirmed | accepted |  |  | neutral | n/a |
| S6b-test-bloat-07 | defended | confirmed | accepted |  |  | neutral | n/a |
| S6b-test-bloat-08 | dup-of | dup-of → S6b-architecture-05 | merged | I.5 |  | advances-0a |  |
| S6b-test-bloat-09 | defended | confirmed | accepted |  |  | neutral | asserted |
| S6c-architecture-00 | defended | dismissed (premise-false-at-pin) | rejected-defender | I.5 |  | neutral | n/a |
| S6c-architecture-01 | defended | confirmed | accepted |  |  | advances-0a | asserted |
| S6c-architecture-02 | defended | confirmed | accepted |  |  | neutral | asserted |
| S6c-architecture-03 | defended | dismissed (deliberate-and-documented) | rejected-defender |  |  | neutral | n/a |
| S6c-architecture-04 | defended | confirmed | accepted |  |  | neutral | asserted |
| S6c-architecture-05 | defended | confirmed | accepted |  |  | neutral | asserted |
| S6c-architecture-06 | defended | dismissed (premise-false-at-pin) | rejected-defender | I.5 |  | neutral | n/a |
| S6c-over-engineering-00 | defended | confirmed | accepted |  |  | neutral | asserted |
| S6c-over-engineering-01 | defended | confirmed | accepted |  |  | neutral | asserted |
| S6c-performance-00 | defended | confirmed | accepted |  |  | neutral | asserted |
| S6c-performance-01 | defended | confirmed | accepted |  |  | neutral | asserted |
| S6c-performance-02 | defended | confirmed | accepted |  |  | neutral | asserted |
| S6c-performance-03 | defended | dismissed (cost-accepted-by-rule) | rejected-defender |  |  | neutral | n/a |
| S6c-performance-04 | defended | confirmed | accepted |  |  | neutral | asserted |
| S6c-performance-05 | defended | confirmed | accepted |  |  | neutral | asserted |
| S6c-performance-06 | defended | confirmed | accepted |  |  | neutral | asserted |
| S6c-performance-07 | defended | confirmed | accepted |  |  | neutral | asserted |
| S6c-test-bloat-00 | defended | confirmed | accepted |  |  | neutral | n/a |
| S6c-test-bloat-01 | defended | dismissed (premise-false-at-pin) | rejected-defender |  |  | neutral | n/a |
| S6c-test-bloat-02 | defended | confirmed | accepted |  |  | neutral | n/a |
| S6c-test-bloat-03 | defended | confirmed | accepted |  |  | neutral | n/a |
| S6c-test-bloat-04 | defended | dismissed (premise-false-at-pin) | rejected-defender |  |  | neutral | n/a |

Counts (`verdicts.json.counts`): received 76 = malformed 0 + anchorRejected
0 + outOfStation 0 + dupOf 6 + blocked 0 + defended
70; defended 70 = confirmed 50 + dismissed 20 + unresolved
0; needsHuman 37; conflicts 0; sharpens 13.
