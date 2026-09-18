# cli-python supersession notes (prior figures contradicted at the baseline)

Baseline: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`   Station: `cli-python`   Written: 2026-09-18

One note per baseline measurement that contradicts BACKLOG.md, the committed mirror or Part-I I.5: the old figure with its source document, the new figure, and the exact command that produced it — never a silent replacement. Every measurement is taken from inventory.json / enforcement-measurements.json (measured at the pin), not re-derived here. Each note has a stable anchor id (SN-nn); every sharpen disposition in calibration.json links the note(s) that justify it. Notes are evidence, not findings — a note never mints an id on its own.

## SN-01 — Part-I I.5: the CLI's cobre_sddp import surface

- **Old figure (source):** '~70 imports' (plans/generalizing/beyond-sddp-generalization.md Part I §I.5, v0.12 snapshot)
- **New figure (baseline `077dbe2c`):** 97 `cobre_sddp` occurrences in 10 files, 70 of them `use` lines (88 distinct paths), at 077dbe2c; ranking policy.rs 20, error.rs 16, simulation.rs 15, outputs.rs 11
- **Command:** `git grep -c 'cobre_sddp' 077dbe2c -- crates/cobre-cli/src | sort -t: -k2 -nr; git grep -n '^use cobre_sddp' 077dbe2c -- crates/cobre-cli/src | wc -l`
- **Measured in:** inventory.json → sddpCoupling; partI-handoff.json i5Queue
- **Justifies:** I.5-2 (sharpen), CD-025, CD-029

## SN-02 — per-side writer call-site figures

- **Old figure (source):** 41 CLI / 29 Python `write_*(` call sites and 26 distinct Python writer names (the epic/ticket text, minted at a136840d)
- **New figure (baseline `077dbe2c`):** 39 CLI call sites (16 terminal `write_line`, 23 writer sites) / 23 Python (5 `_if_any` helpers); external writer names 17 / 17, identical
- **Command:** `git grep -o 'write_[a-z_0-9]*(' 077dbe2c -- 'crates/cobre-cli/src/commands/run/*.rs' | wc -l; … | grep -c 'write_line('; git grep -o 'write_[a-z_0-9]*(' 077dbe2c -- crates/cobre-python/src/run.rs | wc -l; … | grep -c '_if_any('`
- **Measured in:** enforcement-measurements.json → writerSurface; inventory.json → writerBoundary
- **Justifies:** CD-025

## SN-03 — the parity script's per-side name set

- **Old figure (source):** '4 of 17 writers seen; use-line skip + allowlist' (register CD-025 2026-08 text; ticket premise 'source layer covers only its allowlist name set')
- **New figure (baseline `077dbe2c`):** import-resolving since fc81427a: 18 CLI / 18 Python names, 18 in both, floor --min-shared 18, no allowlist; exit 0, "OK: 0 parity mismatch(es) (max allowed: 0). 18 write functions in both paths."; invisible by name only: write_checkpoint, write_scenario (each covered under another name)
- **Command:** `python3 scripts/ci/check_python_parity.py --max 0 --root .`
- **Measured in:** enforcement-measurements.json → sourceLayer; inventory.json → parityEnforcement.layer1_source
- **Justifies:** CD-025

## SN-04 — CD-025 line anchors

- **Old figure (source):** write_training_outputs at outputs.rs:57 (register) / :58 (ticket a136840d); write_training_artifacts at run.rs:478 / :479; the three `_if_any` helpers at run.rs:554, :574, :597 (register)
- **New figure (baseline `077dbe2c`):** outputs.rs:58; run.rs:532; five `_if_any` helpers at run.rs:594, :614, :636, :658, :675 — every symbol still resolves; an anchor refresh, not a claim change (the helper family grew from three to five)
- **Command:** `git grep -n 'fn write_training_outputs\|fn write_training_artifacts\|fn write_.*_if_any' 077dbe2c -- crates/cobre-cli/src/commands/run/outputs.rs crates/cobre-python/src/run.rs`
- **Measured in:** inventory.json → priorAnchors
- **Justifies:** CD-025

## SN-05 — CD-029's Python-parity half

- **Old figure (source):** 'zero equivalent in cobre-python/src/io.rs' (register CD-029; ticket AC `grep -c boundary` returns 0)
- **New figure (baseline `077dbe2c`):** 14 boundary references in io.rs — cobre.io.validate reconciles the boundary as phase 11 since fc81427a (the CD-029 Status bullet of 2026-09-17 already records the parity half as FIXED)
- **Command:** `git show 077dbe2c:crates/cobre-python/src/io.rs | grep -c boundary`
- **Measured in:** inventory.json → supersessions; wave-dispositions.json CD-029.halves
- **Justifies:** CD-029

## SN-06 — cobre-python Rust `#[test]` CI visibility

- **Old figure (source):** 'compiled and run by nothing in CI (22 tests)' (ticket; mirror `### Python-binding Rust tests invisible to CI` docs/design/reserved-seams-and-deferred-debt.md:347; docs/design/testing-architecture.md §5.11)
- **New figure (baseline `077dbe2c`):** 19 tests (errors.rs 3, policy.rs 1, run.rs 13, schema.rs 2) run by ci.yml `Run Rust tests for the bindings crate` (:562-568, python 3.12 matrix) — CI-VISIBLE; the mirror entry and §5.11 are stale (E11 write-back; E07 build/CI cross-reference)
- **Command:** `git grep -c '#\[test\]' 077dbe2c -- 'crates/cobre-python/src/*.rs'; git show 077dbe2c:.github/workflows/ci.yml | grep -n 'manifest-path crates/cobre-python'`
- **Measured in:** enforcement-measurements.json → ciVisibility; inventory.json → testSurface.ciVisibility
- **Justifies:** Cleared cross-reference (no TD)

## SN-07 — the runtime file-set parity layer

- **Old figure (source):** 'its _cli_binary() fixture skips when no compiled cobre binary exists … a runtime layer that skips in CI does not close a gap' (ticket premise)
- **New figure (baseline `077dbe2c`):** guard is conftest.py::cli_binary → _cobre_cli.resolve_cli_binary(required=--require-cli-binary); test_cli_python_file_set_parity.py passed (5 passed, 0 skipped) with target/release/cobre (present, built 2026-09-17); CI builds the CLI (ci.yml:561) and passes --require-cli-binary (:572) before pytest — ciExecutes yes; third layer crates/cobre-cli/tests/python_parity_check.rs::python_parity_script_passes
- **Command:** `python3 -m pytest crates/cobre-python/tests/test_cli_python_file_set_parity.py -v -rs -p no:cacheprovider; git show 077dbe2c:.github/workflows/ci.yml | grep -n 'require-cli-binary\|cargo build --release -p cobre-cli'`
- **Measured in:** enforcement-measurements.json → runtimeLayer / thirdLayer
- **Justifies:** CD-025 (the measure-then-claim residues S6a-architecture-06 / S6a-over-engineering-03)

## SN-08 — CD-002's 'shrinks once CD-001 lands' expectation and its line anchors

- **Old figure (source):** 'Shrinks once CD-001 lands' (register CD-002); broadcast_and_build_setup at setup.rs:262 (register) / :248 (ticket)
- **New figure (baseline `077dbe2c`):** CD-001 landed (b051c410, retired at the sddp gate 2026-09-18) and the tuple did NOT shrink: broadcast_and_build_setup at setup.rs:246 still destructures a positional 10-tuple (:253-263, :290-301, :303-314); the E06-4 delta adds the six-slot `type LoadedCase` inner level (:68-75, :139-146, :266, :631)
- **Command:** `git show 077dbe2c:crates/cobre-cli/src/commands/run/setup.rs | sed -n '68,75p;246,263p;303,317p'`
- **Measured in:** wave-dispositions.json CD-002; verdicts.json S6a-architecture-03
- **Justifies:** CD-002

## SN-09 — CD-009 line anchors

- **Old figure (source):** policy.rs:151, :180, :273 (register) / :167, :196, :316 (ticket a136840d)
- **New figure (baseline `077dbe2c`):** policy.rs:172, :201, :334 (WarmStart, Resume, load_policy_for_simulation :323); the shape is unchanged — an anchor refresh only
- **Command:** `git grep -n 'let policy_dir = ctx.output_dir.join(&setup.policy_path)' 077dbe2c -- crates/cobre-cli/src/commands/run/policy.rs`
- **Measured in:** inventory.json → priorAnchors.policy_dir
- **Justifies:** CD-009

## SN-10 — Part-I I.5: the `report` / `summary` subcommands and the Command enum

- **Old figure (source):** seven Command variants incl. `report` and `summary` (Part-I I.5 v0.12; the E06 ticket texts)
- **New figure (baseline `077dbe2c`):** five variants (Init, Run, Validate, Schema, Version) at main.rs:67 — `report` and `summary` removed by 797ba443 together with their Python and cobre-io mirrors; the I.5 claim survives for validate.rs and the src/summary.rs MODULE (`pub use cobre_sddp::{6 types}` at summary.rs:23) only
- **Command:** `git show 077dbe2c:crates/cobre-cli/src/main.rs | grep -n -A 12 'enum Command'`
- **Measured in:** inventory.json → supersessions; partI-handoff.json i5Queue
- **Justifies:** I.5-1a/1b (retire), I.5 queue rows
