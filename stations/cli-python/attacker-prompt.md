# cli-python attacker worker prompt (shared preamble; twelve cells = four lenses × three sub-surfaces)

Baseline: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (the register pin; the ticket text's `a136840d` is the superseded pin — every
figure below was re-measured at `077dbe2c`, see §Frozen figures). Station: `cli-python` (label:
cobre-cli + cobre-python + facade; the ticket's envelope spells it `cobre-cli+cobre-python+facade`,
the validator and every artifact use the slug `cli-python`). Your cell is `<LENS>.<SUB>`; the dispatcher
names it in your directive.

Sweep ONLY the paths in your sub-surface manifest (§Manifests) — and, for the test-bloat lens, the test
corpus named in that lens block. Every path you cite is read at the baseline: `git show 077dbe2c:<path>`;
never the working tree, never `cargo`, never `pytest`, never `python3 scripts/...` (the parity gate's result is
pre-measured for you).

Inputs (read them first, in this order):
- `plans/architecture-debt-audit/stations/cli-python/inventory.json` — the 30-file partition (S6a 8 / S6b 11 / S6c 11) with per-file total / non-test lines, inline `#[test]` counts and checker-resolvable top symbols; the command surface; the cobre_sddp coupling ranking; the writer boundary (17/17); the parity-enforcement stack with measured coverage; the test surface with CI visibility; 7 supersessions and 50 ticket-figure deviations
- `plans/architecture-debt-audit/stations/cli-python/prior-register.md` — the do-not-re-raise briefing (the four owned ids, the L2 destination rule, the normative constraints, the supersessions)
- `plans/architecture-debt-audit/stations/cli-python/wave-dispositions.json` — the four live dispositions (all `sharpen`) with symbol anchors, register → ticket → pin drift, CD-029's three halves and the ticketDeviations list
- `plans/architecture-debt-audit/stations/cli-python/partI-handoff.json` — Part-I I.5 re-verified claim by claim (8 rows) + the item-7 config-coupling fragment
- `plans/architecture-debt-audit/tools/target-layering-brief.md` — the layering (L0–L4) and the closed Alignment vocabulary
- `docs/design/reserved-seams-and-deferred-debt.md` at the pin (`git show 077dbe2c:docs/design/reserved-seams-and-deferred-debt.md`) — reserved seams and cleared items (the mirror); `CLAUDE.md` § Hard Rules (the Python-parity rule, line 42; `Unwired config is reserved, not dead`); `docs/design/testing-architecture.md` (§5.1–5.3, §5.11) and `.claude/rules/testing.md` (§ Tiers, § Cost discipline) — the test-bloat yardsticks

You are one of twelve read-only Opus attacker workers. The other three lenses over your sub-surface and
the other two sub-surfaces under your lens are covered by siblings; stay inside your lens and your
manifest. No worker sees another worker's output; cross-cell merging is the dispatcher's job. The session
that dispatched you is the sole writer of every artifact in the tree.

## RULES (each is a guardrail; a violated rule voids the envelope and costs the cell its one re-dispatch)

1. **Read-only.** Never write, edit, format, or run anything that mutates the tree or the git state.
   `git show 077dbe2c:<path>`, `git grep <pat> 077dbe2c -- <paths>`, `git ls-tree`, `wc`, `sed -n`, `awk`, `jq` are
   fine; `sed -i`, `cargo` / `maturin` / `pytest` / `python3 scripts/...` (any), `git checkout`, `git stash` are not.
   The only file you may create is the scratch file the dispatcher names, OUTSIDE the repository (`/tmp/cli-attackers/out/`).
2. **One JSON object and nothing else.** Your envelope (§Envelope, with your values) is the whole content
   of the scratch file: first character `{`, last `}`, no preamble, no fence, no prose. Your chat reply is
   the single line `WRITTEN <bytes> <path>`.
3. **SYMBOL ANCHORS ONLY — one exception.** Every anchor is `{"path": ..., "symbol": ...}` where `symbol` is
   a declared `fn|struct|enum|trait|type|const|static|mod|impl` name present in that file at the baseline
   (a struct FIELD name is cited in evidence, not anchored: the register checker resolves declarations). Every
   inherited line anchor of this station has moved (validate.rs +35, cobre-python run.rs +37..+53 since
   a136840d), so a line anchor is either stale or coincidental — it is dropped as `anchor-missing`. The one
   exception: `crates/cobre/src/lib.rs` (two doc lines, no declaration) and `crates/cobre/Cargo.toml` are
   anchored as `{"path": ..., "line": <n>}`. Every anchor path starts with `crates/cobre-cli/`,
   `crates/cobre-python/` or `crates/cobre/` AND lies inside your manifest (test-bloat lens: also the test
   corpus named in its block). A `crates/cobre-sddp` / `cobre-io` / `cobre-core` symbol may be CITED in
   evidence or positives as context, never anchored; a candidate whose anchors would ALL lie outside the
   three crates is emitted with `dupOf: {"station": "sddp|core-io|stochastic|solver-comm", "why": ...}`
   and ONE cited context anchor inside your manifest — the dispatcher re-routes it, it never enters the
   station arrays.
4. **Evidence is a command you ran.** Every candidate carries `evidence.command` (verbatim, runnable from the
   repo root at `077dbe2c`), `evidence.output` (trimmed) and `evidence.reading` (what the output proves). Counts
   and symbol names, not adjectives. The §Frozen figures are pre-measured — cite them, do not re-derive them.
5. **L2 DESTINATION RULE.** Any fix-shape that touches the CLI/Python writer hand-mirror (the
   `write_training_outputs` / `write_simulation_outputs` orchestrators ↔ `write_training_artifacts` + the
   five `*_if_any` helpers ↔ `Study::train_native`; the scenario-summary reshape copied at outputs.rs ↔
   run.rs) names **cobre-io (L2)** — the shared output-orchestration entry point of roadmap V.1 / III.7 — as
   owner, carries `waveRef: "CD-025"` and `reRaiseOf: "CD-025"` (it is a SHARPEN of a live disposition),
   and states NEW evidence. A cobre-sddp home or a cobre-cli-local helper is `alignmentHint: "conflicts"`
   citing Part IV.1 (L2 owns shared output orchestration; forbidden at L2: an engine-specific output tree
   hard-wired in the CLI). The boundary-reconciliation duplicate (`validate.rs::reconcile_boundary` ↔
   cobre-python `run.rs::reconcile_boundary_policy`, both outside `PrepPhase`) carries `waveRef: "CD-029"`;
   its fold destination is cobre-sddp `PrepPhase` (a validation phase, not output orchestration — held as a
   needs-human for the gate in wave-dispositions.json); do not restate it as a cobre-io item.
6. **PART-I I.5 TAG.** Any candidate about cobre-cli / cobre-python coupling to `cobre_sddp` types — the
   run typed on `StudySetup` (`training.rs` `setup.train(`, `simulation.rs` `.simulate(`), the `use
   cobre_sddp` surface, `BroadcastConfig::from_config` → `StudyParams::from_config`, `validate.rs` reaching
   `StudySetup::new_with_boundary_requirements`, the `summary.rs` re-export of six engine types, the
   bindings' `Study` typed on `StudySetup` — carries `partIRef: "I.5"` (`I.3-7` when it is specifically the
   config projection). SDDP vocabulary in L4 is the I.5 finding itself, not a genericity leak: L4 is the
   one layer allowed to name engines (Part IV.1 / IV.4). The missing Engine seam is `advances-0a`.
7. **RESERVED-SEAMS PRE-CHECK (over-engineering lens, every candidate).** Before raising anything, check the
   mirror at the pin, `CLAUDE.md` § Unwired config is reserved, not dead, and any in-code `#[allow(...)]` /
   'reserved' rationale; record `reservedSeamsCheck: {checked: true, result: sanctioned|not-found|not-applicable,
   citation, rule}` on EVERY over-engineering candidate. A sanctioned seam is a `positives` entry with
   `sanctionedBy`, never a candidate. The facade crate has NO mirror entry (prior-register § Normative) but
   `ARCHITECTURE.md:102-106` at the pin declares it 'The umbrella crate. Currently an empty skeleton … reserved for a future single-dependency convenience re-export': record result `sanctioned` (crate map) / `not-found` (mirror) with that citation and carry a `_needsHuman` question (keep as a documented seam, or
   retire the crate until the re-export exists) — it is never pre-judged either way. `commands/broadcast.rs` / `BroadcastConfig` is
   CD-004's twin (sddp station, ratified 2026-09-18, mirror `:563` / `:233`): only the module's PLACEMENT under
   `commands/` is this station's question; the projection itself is never re-raised.
8. **Byte-neutrality bar.** Every fix-shape states how the output tree stays byte-identical from both entry
   points against (1) the CLI-vs-Python golden `crates/cobre-python/tests/test_cli_python_determinism_parity.py`
   (examples/1dtoy through both entry points, whole tree value-for-value), (2) the file-set parity test
   `tests/test_cli_python_file_set_parity.py`, (3) the import-resolving gate `scripts/ci/check_python_parity.py`
   (`--min-shared 18`, 18 shared names at the pin) and (4) the `cobre validate --json`
   single-object stdout contract — or names the golden it would move and why. This station executes no
   measurement and no fix.
9. **Fix shapes are prose.** `fixShape` describes the shape of a fix in sentences; never a diff, patch, code
   block or edit.
10. **Perf candidates carry a layout, never a number.** Every performance candidate sets `measured: false`,
    `unmeasured: {tag: "UNMEASURED", reason}`, `measurementRequest: {layout: "4t"|"2x2", deck, phase,
    claimType: "single-process"|"collective"}` (`4t` = `--threads 4` single process; `2x2` = `mpiexec -n 2` ×
    `--threads 2` for anything crossing ranks — the broadcast path, the rank-0-only writes), `mechanism` (the
    cost mechanism: allocation per call, redundant pass, serial writes, postcard round-trip, Python object
    conversion per element, …), `profiledSymbol` and `queuedTo: "perf-sweep"`. Do not time anything; do not
    quote or estimate timings, speedups or percentages.
11. **Empty is legitimate; blank is a bug.** If your cell is clean, return `candidates: []`, a `cleanVerdict`
    object naming the modules you examined and why they are clean, AND at least one `positives` entry. A cell
    with neither candidates nor positives is a failed cell. The dispatcher records a clean cell as an explicit
    clean verdict for your sub-surface in the merged lens file — never a silent absence.
12. **Never re-raise.** §Do-not-raise lists the settled, superseded, sanctioned, disowned and live items. A live
    disposition (all four are `sharpen`) may be SHARPENED only: the candidate carries `reRaiseOf: "<id>"` and
    NEW evidence; without new evidence, do not emit it. A superseded premise (the `report`/`summary`
    subcommands, the parity gate's 4-of-17 coverage, CI-invisible Python Rust tests, the zero boundary references
    in io.rs) is never emitted as a finding.
13. **LAYERING (L4 entry points).** The `Engine` enum and the dispatch BELONG here (cobre-cli / cobre-python,
    Part IV.4). A fix-shape that puts engine dispatch below L4, makes cobre-io depend on an engine crate, or
    invents an abstraction with one hypothetical consumer (pull, don't push, V.0) is emitted with
    `alignmentHint: "conflicts"` plus a roadmap-consistent alternative in `fixShape`; it is not silently
    dropped and not softened to `neutral`.
14. **Cross-sub-surface duplication is raised ONCE.** The output hand-mirror lives entirely in S6a (both
    `outputs.rs` and cobre-python `run.rs`/`study.rs` are S6a/S6c files — S6a owns it, S6c cites it). The
    validate pipeline mirror (S6b `commands/validate.rs` ↔ S6c cobre-python `io.rs::validate`, phases 8–11)
    is anchored in BOTH files by whichever cell sees it; the dispatcher merges the pair and records the second
    as `merged`. S6c may anchor `crates/cobre-python/src/run.rs` (an S6a file) ONLY for the Rust-test census
    question of the test-bloat lens and for citing the validate mirror's `reconcile_boundary_policy`.

## Alignment vocabulary (closed set; `target-layering-brief.md`)

`advances-0a` (the Engine seam and dispatch in cobre-cli / cobre-python; the `study` config block and its
typed admission gate — an unsupported combination is a rejection, never a silent ignore; shared output
orchestration in cobre-io; rank-0-executes MPI semantics for a direct study (D14); engine-tagged setup
stages; the seam covering `validate`, not only `run`), `advances-0b` (the `cobre-model` carve — not this
station's), `advances-1` (purify the data model — not this station's), `neutral` (advances no phase),
`conflicts` (fix-shape fights the target layering; tag it and give the roadmap-consistent alternative).
Cite the roadmap section in `alignmentCitation` (`Part III.7|IV.1|IV.4|V.0|V.1`) when the hint is not `neutral`.

## Frozen figures (measured at the pin; the ticket's a136840d figures are superseded — cite, never re-derive)

| figure | ticket (a136840d) | pin (077dbe2c) | command |
| ------ | ----------------- | -------------- | ------- |
| `.rs` files / lines in the three crates | 32 / 18,210 | 30 / 16,493 | `git ls-tree -r --name-only 077dbe2c -- crates/cobre-cli/src crates/cobre-python/src crates/cobre/src \| grep '\.rs$'` |
| `Command` enum variants | 7 (Init, Run, Validate, Report, Summary, Schema, Version) | 5 (Init, Run, Validate, Schema, Version) at `main.rs:67` — `report`/`summary` removed by `797ba443` | `git show 077dbe2c:crates/cobre-cli/src/main.rs \| grep -n -A 12 'enum Command'` |
| `cobre_sddp::` occurrences / files / `use cobre_sddp` lines in cobre-cli/src | 95 / 11 / 68 | 97 / 10 / 70 (ranking: policy.rs 20, error.rs 16, simulation.rs 15, outputs.rs 11, setup.rs 8, …) | `git grep -o 'cobre_sddp::' 077dbe2c -- 'crates/cobre-cli/src/*.rs' 'crates/cobre-cli/src/**/*.rs' \| wc -l` |
| `summary.rs` re-export of engine types | `:19`, six types | `:23`, 6 types | `git grep -n 'pub use cobre_sddp' 077dbe2c -- crates/cobre-cli/src/summary.rs` |
| external writer names, CLI side / Python side | 17 / 17 | 17 / 17 — identical sets; CLI split outputs.rs 13 / setup.rs 3 / simulation.rs 1; 3 CLI orchestrators (`write_training_outputs` :58, `write_simulation_outputs` :178, `write_sim_outputs_on_root` simulation.rs:220); 6 Python-local helpers (`write_training_artifacts` :532 + five `*_if_any` :594/614/636/658/675) | `inventory.json → writerBoundary` |
| distinct `write_*(` names in cobre-python/src/run.rs | 26 | 23 | `git show 077dbe2c:crates/cobre-python/src/run.rs \| grep -o 'write_[a-z_0-9]*(' \| sort -u \| wc -l` |
| parity gate (`scripts/ci/check_python_parity.py`) shared writer names | 4 of 17 (two hiding mechanisms) | 18 shared, floor `--min-shared 18`, import-resolving since `fc81427a`; 16 of 17 measured names covered by name, `write_scenario` through `SimulationParquetWriter` | pre-measured (`inventory.json → parityEnforcement.layer1_source`); do NOT run the script |
| CLI integration binaries / their `#[test]` fns / lines; inline `#[test]` in cobre-cli/src | 16 / 84 / 4,035; 199 (283 total) | 14 / 66 / 3,244; 170 (236 total; summary.rs 61, error.rs 28, progress.rs 22, broadcast.rs 19) | `git ls-tree -r --name-only 077dbe2c -- crates/cobre-cli/tests`; `inventory.json → testSurface.cobreCli` |
| pytest files / `def test_` fns / lines | 32 / 197 / 6,512 | 37 / 218 / 7,305 | `git ls-tree -r --name-only 077dbe2c -- crates/cobre-python/tests` |
| Rust `#[test]` fns in cobre-python/src | 22 (run 13, results 5, errors 2, schema 2), 'compiled and run by nothing in CI' | 19 (run.rs 13, errors.rs 3, schema.rs 2, policy.rs 1); the crate IS workspace-excluded (`Cargo.toml:21-23`, so `cargo test --workspace` skips it) BUT `.github/workflows/ci.yml:562` `Run Rust tests for the bindings crate` runs `cargo test --manifest-path crates/cobre-python/Cargo.toml --no-default-features --features highs` (matrix.python-version == '3.12') — CI-VISIBLE; the mirror's `### Python-binding Rust tests invisible to CI` (`:347` at the pin) and testing-architecture §5.11 are STALE (E11 write-back) | `git show 077dbe2c:.github/workflows/ci.yml \| grep -n 'manifest-path crates/cobre-python'` |
| `StudyParams::from_config` non-test call sites | 1 | 5 (setup/mod.rs:378 the one inside cobre-sddp; broadcast.rs:144, validate.rs:384, cobre-python io.rs:254, run.rs:994) | `git grep -n 'StudyParams::from_config' 077dbe2c -- 'crates/*/src/**/*.rs' \| grep -v 'tests\.rs'` |
| cobre-python `io.rs` boundary references (CD-029 parity half) | 0 | 14 — phase 11 of `cobre.io.validate` (`io.rs:167` doc, `:305-325` code) since `fc81427a`; the CLI's `validate.rs::reconcile_boundary` (:219) and cobre-python `run.rs::reconcile_boundary_policy` (:1192) are two copies of the same body | `git show 077dbe2c:crates/cobre-python/src/io.rs \| grep -c boundary` |
| `resolve_thread_count` | setup.rs:58 | setup.rs:58 (unchanged) | `git show 077dbe2c:crates/cobre-cli/src/commands/run/setup.rs \| grep -n 'fn resolve_thread_count'` |
| `commands/broadcast.rs` | 1,063-line pub(crate) wire-type module, not a subcommand | 1,088 lines, `pub(crate) mod broadcast;` at `commands/mod.rs:5`, not a `Command` variant | `git show 077dbe2c:crates/cobre-cli/src/commands/mod.rs` |
| facade crate | 2-line `crates/cobre/src/lib.rs`, no `[dependencies]` | unchanged: two doc lines ('re-exports nothing yet'), `crates/cobre/Cargo.toml` has no `[dependencies]` section; NO mirror entry | `git show 077dbe2c:crates/cobre/src/lib.rs; git show 077dbe2c:crates/cobre/Cargo.toml` |

## Do-not-raise (settled, superseded, sanctioned, disowned, live)

**Settled — never emit:**
- CD-001 closed by `b051c410` (the non-root stochastic rebuild is owned by `cobre-sddp::setup::stochastic_pipeline::build_stochastic_context_for_study`; `run/setup.rs::reconstruct_stochastic_context_non_root` is the thin caller — cite as a positive if you examine it); CD-003 Construction hop closed by `4075c4e8`; CD-006 closed by `3f4c3db3` — all three ratified Cleared at the sddp gate 2026-09-18
- CD-008 retracted · PD-001 refuted · PD-004 deferred pending a profile (sddp) — the BACKLOG do-not-touch list
- CD-019 superseded cut-sync API (sddp/solver-comm); Wave 0/1/2 executed items (CD-010 typed `StateFamily`, CD-017, CD-020, CD-032, CD-033, CD-036, PD-005, OD-002…OD-008)

**Superseded premises — never emit as findings (record as a `positives` entry with the superseding commit if you examine them):**
- the `report` / `summary` subcommands and their Python and cobre-io mirrors — removed by `797ba443 feat: remove the report and summary subcommands and their Python and cobre-io mirrors`; `commands/report.rs` and `commands/summary.rs` do not exist at the pin (never cite them)
- the parity gate 'sees 4 of 17 writers' with a `use`-line skip and a twelve-name allowlist — the script is import-resolving since `fc81427a` (18 shared names, floor 18); a coverage candidate must name a CONCRETE writer the gate misses at the pin
- 'cobre-python's Rust tests are invisible to CI' — ci.yml:562-568 runs them; the surviving fact is two-tier visibility (`cargo test --workspace` skips the excluded crate; CI has a dedicated step) — a doc-drift note for E07/E11, not a finding
- 'cobre-python/src/io.rs has zero boundary references' — CD-029's parity half is FIXED (`fc81427a`); the surviving CD-029 claim is the duplicated reconciler outside `PrepPhase` (rule 5)
- the E06-1 inventory's own deviations list (`inventory.json → ticketFigureDeviations`, 50 rows) — a stale ticket figure is not a finding

**Reserved / sanctioned — emit as `positives` with `sanctionedBy`, never as candidates:**
- the Python-parity HARD RULE (`CLAUDE.md:42`) — normative; its enforcement coverage is measurable (rule 4) but the rule is not a candidate
- `BroadcastConfig` / the `Config → BroadcastConfig → StudyParams::from_config` projection twin — CD-004, sddp station, ratified (mirror `docs/design/reserved-seams-and-deferred-debt.md:563` `#### Setup config-projection sprawl + CLI non-root reconstruction`, audit bullet `:233`); only `commands/broadcast.rs`'s placement is this station's
- `LipschitzConfig.mode` / `UpperBoundEvaluationConfig` (cobre-io, loaded never read) — mirror `:54`; the Legacy cost-scale branch (`cobre-sddp policy_load.rs::rescale_cut_records_for_load`) — BACKLOG L2272; the boundary state-family channels (mirror `:871`) — none is this station's, cite if reached
- unwired / reserved config sections (`CLAUDE.md` § Unwired config is reserved, not dead) — a section the CLI loads and the engine never consumes is a seam, not dead code

**Disowned — anchors live elsewhere; cite as context only:** CD-004 (sddp: the projection twin), CD-026 (core-io: `SimulationParquetWriter::write_scenario` repetition — the CLI/Python call sites are consumers only), CD-082 (sddp: `scalar_parameters` placeholder patched by callers — its validate.rs context is the sddp entry's), CD-011 (core-io: `PolicyStageManifest` naming), the sddp candidates 5a-architecture-01/02 (the wire config superset and the scalar_parameters placeholder cite `broadcast.rs::BroadcastConfig` and `validate.rs::reconcile_boundary` as context — those contexts are CD-002 / CD-009 owners' here, the findings are sddp's).

**Live dispositions (sharpen-only, `reRaiseOf: "<id>"` + `waveRef` where marked + NEW evidence; otherwise do not emit):**

| id | wave | disposition | surviving claim (one line) | primary anchor | tags |
| -- | ---- | ----------- | -------------------------- | -------------- | ---- |
| CD-025 | 5 | sharpen (B) | the cross-crate hand-mirror is intact at 077dbe2c: write_training_outputs (outputs.rs:58) re-inlines the fpha/evaporation/deviation/echo/fixed-delivery writes (:95, :107, :119, :134, :138) that Python factors into five `_if_any… | `crates/cobre-cli/src/commands/run/outputs.rs::write_training_outputs` | `waveRef: "CD-025"`, `partIRef: "I.5"`, destination cobre-io (L2) |
| CD-029 | 5 | sharpen (B) | PrepPhase's prose (validate_phases.rs:20) still claims four preparation steps over a three-row table and a three-variant enum, and the fourth step — boundary reconciliation — still bypasses PrepPhase / prep_phase_metadata in bo… | `crates/cobre-sddp/src/validate_phases.rs::PrepPhase` | `waveRef: "CD-029"`, destination cobre-sddp `PrepPhase` (held for the gate) |
| CD-002 | 4 | sharpen (B → proposed C) | broadcast_and_build_setup (setup.rs:246) destructures a positional 10-tuple (:253-263: raw_system, raw_bcast_config, root_config, root_stochastic, root_estimation_report, root_estimation_path, raw_bcast_tree, root_hydro_models,… | `crates/cobre-cli/src/commands/run/setup.rs::broadcast_and_build_setup` | Wave-4 carrier: sharpen the tuple's contents only, never a second carrier (CD-004 / D14 own the Phase-0a shape) |
| CD-009 | 7 | sharpen (C) | the resolve + `!exists()` guard is repeated exactly three times at policy.rs:172/201/334 (WarmStart :171, Resume :200, load_policy_for_simulation :323) and differs only in the trailing hint — 'Cannot warm-start without a prior … | `crates/cobre-cli/src/commands/run/policy.rs::load_policy_for_simulation` | re-anchor only; cli-local `resolve_policy_dir` is the right home |

## Lens: architecture

Answer each question with symbols or with an explicit `positives` entry.

1. **THE MISSING ENGINE SEAM (S6a; S6c for the bindings' half).** The run is typed on `cobre_sddp::StudySetup`
   end to end: `commands/run/mod.rs::execute` → `::execute_inner` → `run/setup.rs::broadcast_and_build_setup` →
   `run/training.rs::run_training_phase` (`setup.train(` at :88) → `run/simulation.rs::run_simulation_phase`
   (`.simulate(` at :98); the bindings mirror it in `study.rs::Study` (`new_native` :278, `train_native` :315)
   and the monolithic `run.rs::run` (:1669, `run_via_study` :1406). Where would the L4 `Engine` dispatch of
   Part IV.4 sit, what is engine-generic today and what is SDDP-shaped (the two-phase `training_enabled` /
   `n_scenarios` gate at run/mod.rs:166/:224/:227, the per-phase solver profiles, the output tree)? One
   candidate per genuinely distinct seam, `partIRef: "I.5"`, `advances-0a`, Part IV.4 / V.1 cited.
2. **THE CONFIG-PROJECTION CALL SITES (S6a, S6b).** `broadcast.rs::BroadcastConfig::from_config` (:143) wraps
   `StudyParams::from_config` (:144) for the run path (`run/setup.rs::load_case_and_config` :112); `validate.rs`
   calls it directly (:384) and again through `StudySetup::new_with_boundary_requirements` (:234); the
   bindings call it twice more (io.rs:254, run.rs:994). The projection itself is CD-004 (sddp, ratified — cite
   never raise); this station's question is the CALLER shape: five front-end call sites, two paths per front
   end, the validate path building a full `StudySetup` to check a boundary. `partIRef: "I.3-7"` for the
   projection callers, `"I.5"` otherwise.
3. **THE NON-ROOT MPI PATH (S6a).** `run/setup.rs::broadcast_and_build_setup` (:246) — the positional 10-tuple
   (:253-263, `(None×9, Some(e))` :303-313, `(None×10)` :317), `broadcast_value` postcard round-trips, the
   `LoadBroadcastResult` output struct (:151), `run_pre_training` (:468) / `run_root_exports` (:525) as the
   register's positive contrast, `is_root` gating of every write in `run/mod.rs`, `run/simulation.rs::write_sim_outputs_on_root`.
   CD-002 is live (sharpen-only, `reRaiseOf: "CD-002"`); D14 rank-0-executes (Part V.1) is the Phase-0a shape
   the sddp station already restated for the carrier — a second carrier or a tuple redesign is `conflicts`
   with that restatement; sharpen the tuple's CONTENTS or the `is_root` fan-out, cite CD-004 for the config half.
4. **THE DUPLICATED OUTPUT ORCHESTRATION (S6a owns; S6c cites).** Rule 5. `outputs.rs::write_training_outputs`
   re-inlines the fpha/evaporation/deviation/echo/fixed-delivery writes (:95/:107/:119/:134/:138) that
   cobre-python factors into five `*_if_any` helpers called only from `study.rs::train_native` (:369-373);
   `write_simulation_outputs` ↔ `run.rs::run_simulation_phase_py` copy the scenario-summary reshape (:221 ↔ :799);
   `write_sim_outputs_on_root` is the third CLI orchestrator. What EXACTLY differs between the two sides
   (guards, error mapping, progress sink, ordering)? That delta is the typed variation point the single
   cobre-io entry point needs — name it as symbols. `reRaiseOf: "CD-025"`, `waveRef: "CD-025"`.
5. **THE VALIDATE MIRROR (S6b owns; S6c anchors its half).** `commands/validate.rs` runs the six-layer pipeline
   + `StudyParams::from_config` (:384, `PrepPhase::Config` :387) + `prepare_stochastic` (:413) +
   `prepare_hydro_models_from_artifacts` (:422) + `run_boundary_check` (:283) → `reconcile_boundary` (:219) with
   its own `format_boundary_error` (:204); cobre-python `io.rs::validate` (:200) runs the same eleven phases
   (doc :150-175) and calls `run.rs::reconcile_boundary_policy` (:1192) — the same ten steps as the CLI's
   reconciler. Is the phase driver itself hand-mirrored beyond the boundary step (error classification through
   `prep_phase_metadata`, warning rendering, JSON contract)? `reRaiseOf: "CD-029"`, `waveRef: "CD-029"` for the
   boundary half; a NEW candidate for any other duplicated phase driver, anchored in both files (rule 14).
6. **S6b shell and S6c binding surface.** `error.rs::CliError` (16 `cobre_sddp::` refs — an L4 error type that
   enumerates engine errors), `summary.rs` (2,302 lines, `pub use cobre_sddp::{6 types}` at :23 — a renderer
   re-exporting engine types onto the CLI surface), `progress.rs` (1,140 lines, `TrainingEvent` rendering),
   `templates.rs` (one template `1dtoy`, `ALL_TEMPLATES`, `find_template`); S6c `errors.rs` (`LeafClass` /
   `BuiltinBase` mapping), `results.rs` (1,612 lines of parquet/JSON readers re-implemented for Python — do they
   duplicate cobre-io readers?), `policy.rs` (`PyEntitySlot`, `PyCutRecord` — cobre-io record twins), `model.rs`
   (`PyBus`… entity twins of cobre-core), `convert.rs`. Duplicated readers/record twins are candidates with the
   cobre-io or cobre-core owner named in the fix-shape; a `positives` entry where the twin is the PyO3 boundary itself.

## Lens: performance (tag, do not time)

Named targets, re-resolved at the pin; every worker re-runs the greps and reports drift rather than trusting
the list. Each target gets a candidate or an explicit `positives` entry saying why it is not a smell.

1. **Output writing (S6a).** `outputs.rs::write_training_outputs` (:58), `::write_simulation_outputs` (:178),
   `simulation.rs::write_sim_outputs_on_root` (:220) and their Python mirrors `run.rs::write_training_artifacts`
   (:532), `::run_simulation_phase_py` (:696), `study.rs::train_native` (:315): serial per-artifact writes on
   rank 0, per-write `create_dir_all`, row-vector materialisation before each writer, the scenario-summary
   reshape (`write_scenario_summary` :221 / :799). Claim type single-process (`4t`) unless the write sits behind
   a rank-0 gate that other ranks wait on (`collective`, `2x2`).
2. **Thread resolution (S6a).** `run/setup.rs::resolve_thread_count` (:58) and every consumer of the resolved
   count (`RunContext`, `training.rs` `n_fwd_threads`, `simulation.rs`): is the count resolved once and threaded,
   or re-derived? Rayon pool construction per phase?
3. **The broadcast path (S6a, collective).** `broadcast_value` round-trips (`raw_system`, `raw_bcast_config`,
   `raw_bcast_tree`, `raw_scalar_parameters` — four postcard serialisations in `broadcast_and_build_setup`),
   `BroadcastOpeningTree` `to_vec()` copies (:275-281), `reconstruct_stochastic_context_non_root` re-reading
   from disk on non-roots. `claimType: "collective"`, `layout: "2x2"`.
4. **The Python boundary (S6c).** `results.rs` readers (per-row Python object construction, `json_value_to_py`),
   `convert.rs::py_to_json_value` / `pydict_to_json_map`, `model.rs` entity wrappers (`PyBus`… cloning the
   cobre-core entity per access?), `policy.rs` record twins (`PyCutRecord` from `CutRecord` per record). Mechanism:
   per-element conversion, clone-per-access, GIL held across a bulk read.
5. **S6b shell.** `progress.rs` event rendering per iteration (string formatting on the hot training event
   channel?), `summary.rs` formatting, `validate.rs` running `prepare_stochastic` for validation (`the most
   expensive step … validate runs it anyway`, doc :370-371) — a deliberate contract (exit 0 ⇒ run will not fail)
   to record as a positive unless the cost mechanism is avoidable without weakening the contract.
Every candidate: `measured: false`, `unmeasured`, `measurementRequest {layout, deck, phase, claimType}`,
`mechanism`, `profiledSymbol`, `queuedTo: "perf-sweep"`; no numbers.

## Lens: over-engineering (reserved-is-not-dead precondition FIRST — rule 7)

Named targets (each gets a candidate with `reservedSeamsCheck` or a `positives` entry with the citation):
- **The facade crate (S6c).** `crates/cobre/src/lib.rs` (two doc lines: 're-exports nothing yet'),
  `crates/cobre/Cargo.toml` (no `[dependencies]`). Anchor as `{path, line}` (rule 3 exception). The mirror has
  NO entry for it; `ARCHITECTURE.md:102-106` at the pin declares the reservation ("reserved for a future single-dependency convenience re-export") — record both in `reservedSeamsCheck` and raise the owner question in `_needsHuman` (rule 7); the E11 mirror write-back owes the seam its mirror entry.
- **`commands/broadcast.rs` placement (S6a).** 1,088 lines, `pub(crate) mod broadcast;` (`commands/mod.rs:5`),
  not a `Command` variant, a postcard wire-type module (`BroadcastConfig`, `BroadcastOpeningTree`,
  `BroadcastScalarParameter`, `BroadcastStoppingRule`, …, `From` impls) with 19 inline tests. The projection
  content is CD-004's (sanctioned — rule 7); the PLACEMENT under `commands/` is the question.
- **Wrapper layers between clap args and engine calls (S6a, S6b).** `main.rs::Command` → `RunArgs` →
  `run/mod.rs::execute` → `RunContext` → `execute_inner` → `setup_communicator` / `broadcast_and_build_setup` /
  `run_pre_training` → `run_training_phase` → `setup.train(`; `ValidateArgs` → `validate.rs::execute` →
  `run_prep_phase` → `prep_error_to_cli_error` → `describe_prep_error`; `schema.rs` / `version.rs` / `init.rs`
  (`execute` → `execute_scaffold`, `pin_schema_urls`). Which layers decide something and which forward
  without deciding? A layer that exists for testability is a `positives` entry if a test exercises it.
- **The five `*_if_any` helpers (S6a).** They are CD-025's shape (rule 5): a candidate about them is a SHARPEN
  (`reRaiseOf: "CD-025"`, `waveRef: "CD-025"`), never a standalone over-engineering finding; the question
  is whether the guard-per-artifact structure is the typed variation point cobre-io should own.
- **Census with numbers (every cell):** `git grep -c '#\[allow(' 077dbe2c -- <your manifest paths>` — an `#[allow]`
  with a written rationale is sanctioned (`positives`), one without is a candidate; every `enum` in your
  manifest with its variant count (a one-variant enum is a candidate only if it is not a reserved seam);
  structs whose only method is a constructor; `pub` items with no consumer outside their file.
- **S6b / S6c module earning:** `banner.rs` (96 lines), `version.rs` (39), `commands/version.rs`, `templates.rs`
  (a registry with ONE template and a `find_template` lookup — one-consumer registry?), `summary.rs` (2,302
  lines, 61 inline tests for a renderer), cobre-python `version.rs` (57), `schema.rs` (97), `convert.rs` (106),
  `lib.rs` (170) — say explicitly whether each earns its module.

## Lens: test-bloat (yardsticks: `docs/design/testing-architecture.md` §5.1–5.3 + §5.11; `.claude/rules/testing.md` § Tiers, § Cost discipline)

Census at the pin (cite, do not re-derive): cobre-cli `tests/` = 14 integration binaries /
66 `#[test]` fns / 3,244 lines — each binary links the solver (§ Cost discipline);
inline `#[test]` in cobre-cli/src = 170 (summary.rs 61, error.rs 28, progress.rs 22, commands/broadcast.rs 19, commands/run/mod.rs 9, commands/init.rs 8, …). cobre-python `tests/` =
37 files / 218 `def test_` / 7,305 lines (`conftest.py`, `_cobre_cli.py` helpers); Rust `#[test]` in
cobre-python/src = 19 (run.rs 13, errors.rs 3, schema.rs 2, policy.rs 1) — workspace-excluded, CI-visible (§Frozen figures).

Binaries: `cli_color.rs` (3/178), `cli_e2e_run_end_block.rs` (1/153), `cli_run.rs` (12/350), `cli_run_anticipated.rs` (1/423), `cli_run_anticipated_k2.rs` (1/498), `cli_run_evaporation.rs` (1/113), `cli_run_generic_echo.rs` (2/127), `cli_schema.rs` (4/171), `cli_smoke.rs` (10/112), `cli_validate.rs` (22/812), `init.rs` (6/83), `output_metadata_active_backend.rs` (1/82), `python_parity_check.rs` (1/51), `setup_timings_metadata.rs` (1/91).

Pytest files: `_cobre_cli.py` (0), `conftest.py` (0), `test_anticipated_lanes_output_parity.py` (2), `test_anticipated_output_parity.py` (1), `test_boundary_load.py` (3), `test_callback.py` (3), `test_chronological_storage_parity.py` (5), `test_cli_python_determinism_parity.py` (6), `test_cli_python_file_set_parity.py` (5), `test_contract_output_parity.py` (4), `test_convert.py` (11), `test_d48_ic_seed_load.py` (3), `test_enumerated_census_parity.py` (4), `test_errors.py` (9), `test_generic_constraint_echo_parity.py` (1), `test_hydro_inflow_auto_parity.py` (1), `test_import.py` (7), `test_in_transit_output_parity.py` (3), `test_inflow_annual_component_parity.py` (1), `test_io.py` (7), `test_model.py` (8), `test_outputs.py` (9), `test_parity_filling_sigma.py` (7), `test_parity_hydros.py` (7), `test_parity_scalar_parameters.py` (1), `test_policy_load_validation.py` (2), `test_pumping_output_parity.py` (3), `test_readme.py` (2), `test_results.py` (22), `test_run.py` (13), `test_schema.py` (4), `test_solver_stats_parity.py` (6), `test_study.py` (20), `test_types.py` (1), `test_validate.py` (17), `test_version.py` (5), `test_write_policy_checkpoint.py` (15).

Corpus per cell (anchors allowed there in addition to your manifest): **S6a** — the run-path binaries
(`cli_run*.rs`, `cli_e2e_run_end_block.rs`, `output_metadata_active_backend.rs`, `setup_timings_metadata.rs`,
`python_parity_check.rs`) and the Python run/Study/parity suites (`test_run.py`, `test_study.py`,
`test_cli_python_*_parity.py`, `test_*_parity.py`, `test_outputs.py`, `test_callback.py`,
`test_write_policy_checkpoint.py`); **S6b** — `cli_validate.rs`, `cli_schema.rs`, `cli_smoke.rs`, `cli_color.rs`,
`init.rs` and the inline `#[cfg(test)]` modules of your manifest (summary.rs 61, error.rs 28, progress.rs 22,
init.rs 8, validate.rs 7, templates.rs 7, banner.rs 5, main.rs 2); **S6c** — the remaining pytest files
(`test_convert.py`, `test_errors.py`, `test_import.py`, `test_io.py`, `test_model.py`, `test_policy_load_validation.py`,
`test_results.py`, `test_schema.py`, `test_types.py`, `test_validate.py`, `test_version.py`, `test_boundary_load.py`,
`test_d48_ic_seed_load.py`, `test_readme.py`, `conftest.py`, `_cobre_cli.py`) and the Rust `#[test]` census
(rule 14 exception: `crates/cobre-python/src/run.rs` test-module symbols may be anchored for that question).

Three questions per cell:
1. **Inline vs integration split.** Is the split principled (§5.1 canonical layout, §5.3 tier decision rule) or
   accidental? summary.rs carries 61 inline tests for a renderer; error.rs 28; broadcast.rs 19 (S6a);
   cobre-python `run.rs` 13 Rust tests beside a 37-file pytest corpus. Which tier (§ Tiers 1-4) does each
   suite occupy, and is any tier occupied twice for the same claim?
2. **Harness duplication.** Which binaries / pytest files rebuild fixtures another already provides (case-dir
   helpers, `1dtoy` copies, output-tree readers, CLI invocation wrappers vs `_cobre_cli.py`, checkpoint
   writers)? Per-file symbol anchors; a `tests/common` or `conftest.py` consolidation as fix-shape — a NEW
   fixture crate is rejected; § Cost discipline counts every new binary as a full solver link.
3. **The parity suite triangle.** `python_parity_check.rs` (1 test shelling the script), `test_cli_python_file_set_parity.py`
   (5), `test_cli_python_determinism_parity.py` (6) and the eleven `test_*_parity.py` output-parity files: which
   claims overlap, which are load-bearing (the golden is tier 1; §5.11 names the parity suite as 'keep
   unchanged'), and does any file re-run a full study for a claim a cheaper tier covers? Crate-wide totals
   belong to the test-corpus station (E08): report only what is anchored in these three crates' tests.

## Manifests (partition proven by the dispatching session before any worker runs)

Union of the three == every `.rs` under `crates/cobre-cli/src`, `crates/cobre-python/src`, `crates/cobre/src` at the baseline (30 files),
pairwise disjoint (`inventory.json` coverage: findCount 30, assignedCount 30, unassigned/doubleAssigned/phantom []).

### `S6a` — writer/run boundary

8 files, 6,149 lines (4,464 non-test). Lines are total / non-test / inline `#[test]` fns at the pin.

- `crates/cobre-cli/src/commands/broadcast.rs` 1,088 / 461 / 19 — `BroadcastStoppingRule`, `BroadcastStoppingMode`, `BroadcastBackwardScheduler`, `From`
- `crates/cobre-cli/src/commands/run/mod.rs` 520 / 322 / 9 — `outputs`, `policy`, `setup`, `simulation`
- `crates/cobre-cli/src/commands/run/outputs.rs` 229 / 229 / 0 — `WriteTrainingArgs`, `write_training_outputs`, `WriteSimulationArgs`, `write_simulation_outputs`
- `crates/cobre-cli/src/commands/run/policy.rs` 383 / 383 / 0 — `load_and_validate_checkpoint`, `load_checkpoint_into_setup`, `apply_training_policy`, `load_policy_for_simulation`
- `crates/cobre-cli/src/commands/run/setup.rs` 669 / 600 / 1 — `resolve_thread_count`, `LoadedCase`, `load_case_and_config`, `LoadBroadcastResult`
- `crates/cobre-cli/src/commands/run/simulation.rs` 468 / 418 / 1 — `run_simulation_phase`, `write_sim_outputs_on_root`, `print_sim_summary`, `merge_simulation_metadata`
- `crates/cobre-cli/src/commands/run/training.rs` 294 / 294 / 0 — `TrainingPhaseResult`, `GlobalTrainingStats`, `run_training_phase`, `aggregate_solver_stats`
- `crates/cobre-python/src/run.rs` 2,498 / 1,757 / 13 — `RunError`, `From`, `from`, `From`

### `S6b` — diagnostics + CLI shell

11 files, 5,652 lines (2,783 non-test). Lines are total / non-test / inline `#[test]` fns at the pin.

- `crates/cobre-cli/src/banner.rs` 96 / 55 / 5 — `render_banner_string`, `print_banner`, `tests`, `test_render_banner_colored_contains_ansi_escapes`
- `crates/cobre-cli/src/commands/init.rs` 376 / 203 / 8 — `InitArgs`, `execute`, `execute_scaffold`, `pin_schema_urls`
- `crates/cobre-cli/src/commands/mod.rs` 10 / 10 / 0 — `broadcast`, `init`, `run`, `schema`
- `crates/cobre-cli/src/commands/schema.rs` 88 / 88 / 0 — `SchemaArgs`, `SchemaCommand`, `ExportArgs`, `execute`
- `crates/cobre-cli/src/commands/validate.rs` 611 / 479 / 7 — `ValidateArgs`, `ValidateBoundaryOutput`, `BoundaryOutcome`, `ValidateErrorOutput`
- `crates/cobre-cli/src/commands/version.rs` 39 / 39 / 0 — `execute`
- `crates/cobre-cli/src/error.rs` 634 / 288 / 28 — `CliError`, `CliError`, `exit_code`, `validation_lines`
- `crates/cobre-cli/src/main.rs` 134 / 112 / 2 — `banner`, `commands`, `error`, `progress`
- `crates/cobre-cli/src/progress.rs` 1,140 / 532 / 22 — `TRAINING_TEMPLATE`, `SIMULATION_TEMPLATE`, `fmt_sci`, `fmt_hms`
- `crates/cobre-cli/src/summary.rs` 2,302 / 824 / 61 — `print_hydro_model_summary`, `format_rank_list`, `print_execution_topology`, `format_production_line`
- `crates/cobre-cli/src/templates.rs` 222 / 153 / 7 — `TemplateFile`, `Template`, `DTOY1_FILES`, `DTOY1_TEMPLATE`

### `S6c` — bindings + facade

11 files, 4,692 lines (4,512 non-test). Lines are total / non-test / inline `#[test]` fns at the pin.

- `crates/cobre-python/src/convert.rs` 106 / 106 / 0 — `py_to_json_value`, `pydict_to_json_map`
- `crates/cobre-python/src/errors.rs` 472 / 357 / 3 — `LeafClass`, `BuiltinBase`, `LeafClass`, `fn`
- `crates/cobre-python/src/io.rs` 335 / 335 / 0 — `load_error_kind`, `load_validate_config`, `convert_load_error`, `build_warnings_list`
- `crates/cobre-python/src/lib.rs` 170 / 170 / 0 — `convert`, `errors`, `io`, `model`
- `crates/cobre-python/src/model.rs` 526 / 526 / 0 — `PyBus`, `PyBus`, `id`, `name`
- `crates/cobre-python/src/policy.rs` 583 / 556 / 1 — `PyEntitySlot`, `From`, `from`, `PyCutRecord`
- `crates/cobre-python/src/results.rs` 1,612 / 1,612 / 0 — `canonicalize_dir`, `json_value_to_py`, `read_json_file`, `open_parquet_file`
- `crates/cobre-python/src/schema.rs` 97 / 59 / 2 — `export`, `tests`, `test_export_writes_all_schemas_as_valid_json`, `test_export_creates_missing_directory`
- `crates/cobre-python/src/study.rs` 732 / 732 / 0 — `phase_error_to_pyerr`, `Study`, `Policy`, `Policy`
- `crates/cobre-python/src/version.rs` 57 / 57 / 0 — `version_info`
- `crates/cobre/src/lib.rs` 2 / 2 / 0 — (no declarations — two doc lines)

## Envelope (frozen contract — `tools/validate-envelope.py --role attacker --station cli-python` reads exactly this)

```json
{
  "station": "cli-python",
  "subStation": "<SUB>",
  "baseline": "077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c",
  "lens": "<LENS>",
  "candidates": [
    {
      "title": "one line naming the smell and its subject; no ID (IDs are assigned at calibration)",
      "anchors": [
        {
          "path": "crates/cobre-cli/src/<file>.rs",
          "symbol": "<declared symbol in that file>"
        }
      ],
      "evidence": {
        "command": "git show 077dbe2c:crates/cobre-cli/src/<file>.rs | sed -n '<a>,<b>p'",
        "output": "…trimmed…",
        "reading": "why that output supports the claim; counts and symbol names"
      },
      "proposedSeverity": "A|B|C",
      "fixShape": "prose only; names the destination crate; states the byte-neutrality bar (rule 8)",
      "alignmentHint": "advances-0a|advances-0b|advances-1|neutral|conflicts",
      "alignmentCitation": "Part III.7|IV.1|IV.4|V.0|V.1 — required when alignmentHint != neutral",
      "partIRef": "I.5 | I.3-7 | null (rule 6)",
      "waveRef": "CD-025 | CD-029 | null (rule 5)",
      "reRaiseOf": "CD-025 | CD-029 | CD-002 | CD-009 when sharpening a live disposition; otherwise null",
      "reservedSeamsCheck": {
        "checked": true,
        "result": "sanctioned|not-found|not-applicable",
        "citation": "mirror section or null",
        "rule": "which rule was checked"
      },
      "measured": "performance lens: false; otherwise null",
      "unmeasured": {
        "tag": "UNMEASURED",
        "reason": "station tickets do not time claims; queued to the cross-cutting sweep"
      },
      "measurementRequest": {
        "layout": "4t|2x2",
        "deck": "examples/1dtoy or examples/deterministic/<case>",
        "phase": "which phase",
        "claimType": "single-process|collective"
      },
      "mechanism": "performance lens: the cost mechanism, no numbers; otherwise null",
      "profiledSymbol": "performance lens: the symbol the perf epic profiles; otherwise null",
      "queuedTo": "performance lens: perf-sweep; otherwise null",
      "dupOf": "null, or {\"station\": \"sddp|core-io|stochastic|solver-comm\", \"why\": \"...\"} when every real anchor lies outside the three crates (rule 3)"
    }
  ],
  "positives": [
    {
      "subject": "crates/cobre-cli/src/… (a module, a symbol, or a sanctioned seam)",
      "why": "correct and worth protecting / sanctioned / examined and clean",
      "sanctionedBy": "mirror section, CLAUDE.md rule or register id when applicable, else null"
    }
  ],
  "cleanVerdict": "null, or {\"why\": \"the modules examined and why the cell is clean\"} when candidates is empty (rule 11)",
  "_needsHuman": [
    "a question only the owner can answer; empty list when none"
  ]
}
```

Severity: A = wrong results, lost determinism, a user-visible abort, or a structural block on the roadmap;
B = real debt with a bounded fix and a named blast radius; C = local quality. `proposedSeverity` is the
attacker's rating; the house calibrates. Keys not in the schema are dropped at the gate; `verdict` is not
yours to set (the defender's). The envelope is the whole content of your scratch file.

## Dispatch

Twelve cells in two waves of six concurrent workers, two lenses per wave (architecture + performance, then
over-engineering + test-bloat) — the manifests are 8 / 11 / 11 files and this prompt is a third of the sddp
station's, so six concurrent cells do not starve. Each cell's directive names `<LENS>`, `<SUB>` and the scratch
path `/tmp/cli-attackers/out/<LENS>.<SUB>.json`. A shape-invalid envelope is re-dispatched exactly once with the validator
error quoted; a second failure is `needs-human`. `attacker-log.md` is the resume point.
