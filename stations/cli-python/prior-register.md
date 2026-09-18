# Prior register — cobre-cli + cobre-python + facade (baseline `077dbe2c`)

What the register and the committed mirror already say about the L4 surface, so the attacker fan-outs over
the three sub-surfaces (S6a writer/run boundary, S6b diagnostics + CLI shell, S6c bindings + facade) receive
a real do-not-re-raise list instead of re-discovering ratified work. Every code anchor below is `path::symbol`,
resolved against the baseline blob (`git show 077dbe2c:<path>`) with the declaration regex `tools/check-anchors.py`
uses — never `path:line`, because every inherited line anchor has moved (the stale-line column records the
register's line, the ticket's a136840d line and the line at this baseline). Mirror and `CLAUDE.md` citations
are `path:line` on the baseline blob. Each block carries a `reraiseKey` list the attacker screen and
`check-reraise.py` match candidate titles and anchors against.

Four dispositions: an **owned prior id** is a roster row the disposition ticket (E06-2) fills with keep /
retire-with-commit / sharpen — attackers may _sharpen_ a row, never re-derive it as new; a **destination rule**
binds the fix-shape of every S6a candidate; a **normative constraint** is a rule whose enforcement may be a
finding but whose existence is not a candidate; a **supersession** is an inherited figure the baseline contradicts,
recorded old → new with the command so no candidate re-derives the stale number.

---

## Owned prior IDs (disposition pending — ticket 2 fills the Disposition column)

| ID | Wave | Sev (register) | Claim (one line) | Baseline anchor (symbol) | Stale-line note (register → ticket → pin) | Destination rule | Disposition |
| -- | ---- | -------------- | ---------------- | ------------------------ | ---------------------------------------- | ---------------- | ----------- |
| CD-025 | 5 | B — OWNER-AGREED, register L917; Note 2026-09-17: still OPEN, structure unchanged, detection improved | Training + simulation output orchestration hand-mirrored across the CLI/Python boundary; the five `*_if_any` helpers live in cobre-python so the CLI re-inlines its own copies; kept aligned only by the Python-parity hard rule | `crates/cobre-cli/src/commands/run/outputs.rs::write_training_outputs`; `crates/cobre-python/src/run.rs::write_training_artifacts`; `crates/cobre-python/src/run.rs::write_fpha_hyperplanes_if_any`; `crates/cobre-python/src/run.rs::write_evaporation_models_if_any`; `crates/cobre-python/src/run.rs::write_generic_constraint_echo_if_any`; `crates/cobre-python/src/run.rs::write_fixed_delivery_if_any`; `crates/cobre-python/src/run.rs::write_fpha_deviation_points_if_any` | outputs.rs :57 → :58 → :58; run.rs :478 → :479 → :532; `*_if_any` 554/574/597 → 555/575/598 (+621/638) → 594/614/636/658/675 | cobre-io (L2) owns the shared entry point; the entry's own "cobre-sddp or cobre-io" wording is half-stale; an L3 (cobre-sddp) home or a cobre-cli-local helper is `conflicts` (Part IV target layering) and is held, never recommended | _pending_ |
| CD-029 | 5 | B — Status partial 2026-09-17: the Python-parity half is FIXED (`cobre.io.validate` runs the boundary reconciliation as its phase 11); the abstraction half is OPEN, register L1244 | `PrepPhase` doc claims four preparation steps over a three-variant enum, and both front ends run the boundary check outside `PrepPhase` / `prep_phase_metadata` | `crates/cobre-sddp/src/validate_phases.rs::PrepPhase`; `crates/cobre-cli/src/commands/validate.rs::run_boundary_check` | validate.rs :255 → :256 → :283; validate_phases.rs `PrepPhase` enum at :33; io.rs boundary references 0 → 14 (parity half fixed) | fold the boundary check into `PrepPhase` (or correct the doc to three) — the remaining half; the binding mirror is done; Alignment advances-0a | _pending_ |
| CD-002 | 1 | B (register L168) — the ticket calls it C-tier; the register rating is B | Positional 10-tuple straddling three concerns in `broadcast_and_build_setup` with a parallel `(None×10)` non-root arm; direction: named `RootLoadArtifacts` struct | `crates/cobre-cli/src/commands/run/setup.rs::broadcast_and_build_setup`; `crates/cobre-cli/src/commands/broadcast.rs::BroadcastConfig` (the wire twin — CD-004's half, sddp station) | setup.rs :262 → :248 → :246 | the positional tuple is this station's; the `Config → BroadcastConfig` projection twin is CD-004 (ratified 2026-09-18 at the sddp gate, advances-0a) and is not re-raised here | _pending_ |
| CD-009 | 1 | C (register L312) | The `policy_dir` resolve + `!exists()` guard is repeated three times (WarmStart, Resume, `load_policy_for_simulation`), differing only in the trailing hint; direction: extract `resolve_policy_dir(ctx, setup, missing_hint)` | `crates/cobre-cli/src/commands/run/policy.rs::load_policy_for_simulation` (the three `let policy_dir` sites are policy.rs L172, L201, L334 — spans, no declaration) | policy.rs :151/180/273 → :167/196/316 → :172/201/334 | cli-local extraction is the right home (a path helper, not output orchestration) — the L2 rule below does not apply | _pending_ |

**Owned elsewhere — do not raise here:** CD-004 (sddp station, the `Config → StudyParams / BroadcastConfig` twin, ratified 2026-09-18); CD-001 and the CD-003 Construction hop (retired at the sddp gate, commits `b051c410` / `4075c4e8`); CD-026 (`SimulationParquetWriter::write_scenario` repetition — cobre-io, core-io station); CD-019 (superseded cut-sync API — E5 dup-of, solver-comm); CD-082 (sddp: `scalar_parameters` placeholder — its `validate.rs` context is cited, the entry is sddp's).

- **reraiseKey**: `CD-025`, `CD-029`, `CD-002`, `CD-009`, `write_training_outputs`, `write_training_artifacts`, `_if_any`, `PrepPhase`, `run_boundary_check`, `broadcast_and_build_setup`, `BroadcastConfig`, `policy_dir`, `load_policy_for_simulation`, `hand-mirror`

---

## Destination rule (binding on every S6a fix-shape)

- Owner of any shared output-orchestration entry point: **cobre-io (L2)** — the crate both the CLI and the bindings already depend on; the mirror's own target (`docs/design/reserved-seams-and-deferred-debt.md:587`) says "a crate both the CLI and Python depend on", and Part IV of the roadmap places output orchestration below the front ends.
- `conflicts` + Part IV citation + held for owner override: a **cobre-sddp (L3)** home (the engine would own front-end output policy) or a **cobre-cli-local** helper (keeps the double-mirror). Neither is ever recorded as the recommended direction; CD-025's own "cobre-sddp or cobre-io" wording is superseded on the cobre-sddp half.
- A candidate whose fix-shape names cobre-sddp or a cli-local helper for the writer boundary is tagged `conflicts` at ingest with the rule above cited; the owner gate asks it alone.

---

## Normative — not candidates

- **Python-parity hard rule** — `CLAUDE.md:42`: every output file the CLI writes must also be written by the Python bindings; when adding a new output, wire it in both. Its *enforcement coverage* is measurable and may be a finding; the rule itself is not a candidate. Measured at the pin (`inventory.json → parityEnforcement`): `python3 scripts/ci/check_python_parity.py --max 0 --root .` exits 0 with "OK: 0 parity mismatch(es) (max allowed: 0). 18 write functions in both paths."; the script resolves bare imports and matches `write_*` / `export_*` / `*Writer`, floor `--min-shared 18`; 16 of the 17 measured writer names are enforced by name and the last (`write_scenario`) through its writer type. A coverage candidate must name a concrete writer the gate misses — the ticket's "4 of 17" is superseded.
- **Reserved-seams check before ANY OD entry on the facade crate or on `commands/broadcast.rs`.** The mirror's reserved-seam register (`docs/design/reserved-seams-and-deferred-debt.md:30`) has **no entry for the facade crate** (`crates/cobre/src/lib.rs` is two doc lines: "re-exports nothing yet"; `crates/cobre/Cargo.toml` has no `[dependencies]` section) — an OD on the facade cannot be Cleared-with-citation and goes to the owner gate as an open question, not as a pre-judged finding. `commands/broadcast.rs` / `BroadcastConfig` is covered by the deferred-debt entry `docs/design/reserved-seams-and-deferred-debt.md:563` and the audit bullet at `docs/design/reserved-seams-and-deferred-debt.md:233` — the projection twin is CD-004's business (sddp station); a candidate about the module's *placement* under `commands/` (a 1,088-line pub(crate) wire-type module that is not a subcommand) is a legitimate over-engineering question, a candidate about the twin is dup-of CD-004.
- **CD-025's mirror entry** `docs/design/reserved-seams-and-deferred-debt.md:587` — any candidate restating the hand-mirror is dup-of CD-025 (sharpen the row, no new id).
- **Stale mirror entry** `docs/design/reserved-seams-and-deferred-debt.md:347` — its premise no longer holds at the pin: `.github/workflows/ci.yml:562` (`Run Rust tests for the bindings crate`, `cargo test --manifest-path crates/cobre-python/Cargo.toml --no-default-features --features highs`) compiles and runs the 19 Rust `#[test]` functions of the bindings crate. A candidate raising CI-invisibility is a supersession routed to the E11 mirror write-back (and to `docs/design/testing-architecture.md:559` §5.11), never a finding.
- **Part-I I.5 sharpening** — `validate`, `report` and `summary` are named SDDP-shaped; at the pin only `validate` exists (`report` and `summary` were removed with their Python and cobre-io mirrors, `797ba443 feat: remove the report and summary subcommands and their Python and cobre-io mirrors`); the claim holds for `validate.rs` and the `src/summary.rs` module (its `pub use cobre_sddp::{...}` re-export at `crates/cobre-cli/src/summary.rs` L23 carries 6 engine types). An E9/E11 doc correction, not a station finding.

- **reraiseKey**: `Python parity`, `check_python_parity`, `--min-shared`, `facade`, `umbrella`, `re-exports nothing`, `broadcast.rs`, `BroadcastConfig`, `cdylib`, `manifest-path`, `invisible to CI`, `report`, `summary`

---

## Supersessions (old → new, with command)

| Fact | Inherited | Baseline | Command |
| ---- | --------- | -------- | ------- |
| `StudyParams::from_config` non-test call sites (prior audit) | 1 | 5 — the params.rs hits sit inside its #[cfg(test)] module and are excluded | `git grep -n 'StudyParams::from_config' 077dbe2c -- 'crates/*/src/**/*.rs' \| grep -v 'tests\.rs'` |
| distinct `write_*(` names in cobre-python/src/run.rs (prior audit) | 26 | 23 | `git show 077dbe2c:crates/cobre-python/src/run.rs \| grep -o 'write_[a-z_0-9]*(' \| sort -u \| wc -l` |
| cobre-cli production (non-test) lines (prior audit / ticket) | register 2026-08-22: 4522; ticket @a136840d: 5419 | 5490 | `station_checks.classify_lines over every git-show blob of crates/cobre-cli/src/**/*.rs (brace+indent #[cfg(test)] subtraction)` |
| Command enum variants (ticket / Part-I I.5) | 7 | 5 — report and summary removed by 797ba443 feat: remove the report and summary subcommands and their Python and cobre-io mirrors | `git show 077dbe2c:crates/cobre-cli/src/main.rs \| grep -n -A 12 'enum Command'` |
| parity gate shared writer names (ticket) | 4 | 18 — import-resolving script since fc81427a; the two hiding mechanisms are closed | `python3 scripts/ci/check_python_parity.py --max 0 --root .` |
| cobre-python Rust #[test] CI visibility (ticket / mirror `Python-binding Rust tests invisible to CI`) | compiled and run by nothing in CI (22 tests) | run by ci.yml `Run Rust tests for the bindings crate` (19 tests) | `git show 077dbe2c:.github/workflows/ci.yml \| grep -n 'manifest-path crates/cobre-python'` |
| cobre-python/src/io.rs boundary references (CD-029 parity half) (prior audit (CD-029 `grep-confirmed: no boundary references`)) | 0 | 14 — CD-029's Status bullet (2026-09-17) already records the parity half as FIXED | `git show 077dbe2c:crates/cobre-python/src/io.rs \| grep -c boundary` |

The five `StudyParams::from_config` non-test call sites: crates/cobre-cli/src/commands/broadcast.rs L144, crates/cobre-cli/src/commands/validate.rs L384, crates/cobre-python/src/io.rs L254, crates/cobre-python/src/run.rs L994, crates/cobre-sddp/src/setup/mod.rs L378 (the `params.rs` hits sit inside its `#[cfg(test)]` module).

**Ticket-figure deviations.** 50 figures the ticket minted at a136840d differ at `077dbe2c` (full list: `inventory.json → ticketFigureDeviations`): the census (18 CLI files / 9303 lines, not 20 / 10,873; 30 files / 16493 lines in total, not 32 / 18,210), the command surface (5 variants, not 7), the coupling ranking (97 occurrences over 10 files, not 95 over 11), the test surface (14 CLI binaries / 236 tests, not 16 / 283; 37 pytest files / 218 tests, not 32 / 197) and every inherited line anchor.

---

## Do not re-raise

- CD-025, CD-029, CD-002, CD-009 as new findings — sharpen the row (E06-2 owns the disposition)
- the `Config → StudyParams / BroadcastConfig` twin — CD-004 (sddp station, ratified); CD-001 / CD-003 hop — retired
- the CLI/Python hand-mirror as such — CD-025 (mirror `docs/design/reserved-seams-and-deferred-debt.md:587`)
- `SimulationParquetWriter::write_scenario` repetition — CD-026 (core-io station)
- the Python-parity hard rule itself, or the parity gate's "4 of 17" coverage — the rule is normative, the figure is superseded
- cobre-python Rust tests as CI-invisible — superseded at the pin (ci.yml runs them); E11 mirror write-back
- `report` / `summary` as SDDP-shaped subcommands — they no longer exist; E9/E11 doc correction
- a fix-shape that homes output orchestration in cobre-sddp or in a cli-local helper — `conflicts`, held for the owner

## Re-derive

```sh
git ls-tree -r --name-only 077dbe2c -- crates/cobre-cli/src | grep '\.rs$' | wc -l
git ls-tree -r --name-only 077dbe2c -- crates/cobre-python/src | grep '\.rs$' | wc -l
git ls-tree -r --name-only 077dbe2c -- crates/cobre/src | grep '\.rs$' | wc -l
git ls-tree -r --name-only 077dbe2c -- crates/cobre-cli/src | grep '\.rs$' | while read f; do git show 077dbe2c:"$f"; done | wc -l
git ls-tree -r --name-only 077dbe2c -- crates/cobre-python/src | grep '\.rs$' | while read f; do git show 077dbe2c:"$f"; done | wc -l
git ls-tree -r --name-only 077dbe2c -- crates/cobre/src | grep '\.rs$' | while read f; do git show 077dbe2c:"$f"; done | wc -l
git grep -o 'cobre_sddp::' 077dbe2c -- 'crates/cobre-cli/src/*.rs' 'crates/cobre-cli/src/**/*.rs' | wc -l
git grep -l 'cobre_sddp::' 077dbe2c -- 'crates/cobre-cli/src/*.rs' 'crates/cobre-cli/src/**/*.rs' | wc -l
git grep -n '^use cobre_sddp' 077dbe2c -- 'crates/cobre-cli/src/*.rs' 'crates/cobre-cli/src/**/*.rs' | wc -l
git grep -n 'pub use cobre_sddp' 077dbe2c -- crates/cobre-cli/src/summary.rs
git show 077dbe2c:crates/cobre-cli/src/main.rs | grep -n -A 12 'enum Command'
git show 077dbe2c:crates/cobre-cli/src/main.rs | grep -n 'match cli.command'
git ls-tree -r --name-only 077dbe2c -- crates/cobre-cli/src/commands
git show 077dbe2c:crates/cobre-cli/src/commands/mod.rs | grep -n 'mod broadcast'
git show 077dbe2c:crates/cobre-python/src/run.rs | grep -o 'write_[a-z_0-9]*(' | sort -u | wc -l
python3 scripts/ci/check_python_parity.py --max 0 --root .
git grep -n 'StudyParams::from_config' 077dbe2c -- 'crates/*/src/**/*.rs' | grep -v 'tests\.rs'
git show 077dbe2c:.github/workflows/ci.yml | grep -n 'manifest-path crates/cobre-python'
git show 077dbe2c:crates/cobre-python/src/io.rs | grep -c boundary
```
