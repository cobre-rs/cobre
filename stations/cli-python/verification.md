# Station verification — cli-python

Station baseline: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (inventory.json; every entry, anchor and figure is measured there).
Register pin: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (the drift rule ties HEAD to it; provenance bullets resolve there).
Declared section title (argv[2]): `★ QUALITY EVALUATION (2026-09, baseline a136840d) — cli-python`
Resolved by slug `cli-python` via the em-dash tail: `★ QUALITY EVALUATION (2026-09, baseline a136840d) — cli-python`

The section is resolved by the station slug, not by the spelled-out title, so the six
later stations reuse this verifier unchanged and this record is byte-stable across runs
at the pin with the same invocation (a re-run leaves the committed file unmodified). The
ticket's test commands quote an older `★ STATION: …` title; the heading on disk is the
`★ QUALITY EVALUATION (…) — <slug>` form the calibration ticket wrote, and the slug
resolves it either way.

| # | Check | Command | Exit | Result |
| --- | --- | --- | --- | --- |
| 1 | check-anchors | `python3 plans/architecture-debt-audit/tools/check-anchors.py cli-python` | 0 | PASS |
| 2 | check-reraise | `python3 plans/architecture-debt-audit/tools/check-reraise.py cli-python` | 0 | PASS |
| 3 | fields-check | `python3 plans/architecture-debt-audit/tools/fields-check.py cli-python` | 0 | PASS |
| 4 | register | `python3 plans/architecture-debt-audit/tools/station_verify.py register . cli-python` | 0 | PASS |
| 5 | inventory-set-equality | `python3 plans/architecture-debt-audit/tools/station_verify.py inventory inventory.json .` | 0 | PASS |
| 6 | infra-genericity | `python3 plans/architecture-debt-audit/tools/station_verify.py genericity . partI-handoff.json` | 0 | PASS |
| 7 | read-only-workspace | `python3 plans/architecture-debt-audit/tools/station_verify.py readonly . 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` | 0 | PASS |

Reserved-seam re-raises (no entry re-raises a ratified seam without a sanction
citation) are enforced twice above: `check-reraise` against the mirror corpus, and
the `register` denylist against the Reserved-seam register.

Test suite (`python3 -m unittest stations/cli-python/tests/test_station.py`): PASS (exit 0)

Per-check logs: `plans/architecture-debt-audit/stations/cli-python/.<check>.log`.

## Station-specific checks — cli-python

| Check | Command | Exit | Result |
| --- | --- | --- | --- |
| shared-verifier | `bash plans/architecture-debt-audit/tools/verify-station.sh cli-python '★ QUALITY EVALUATION (2026-09, baseline a136840d) — cli-python'` | 0 | PASS |
| heading-resolution | dated scaffold title matched exactly once (find_section rule); inner `★ STATION 6 — cobre-cli / cobre-python / facade (2026-09, baseline)` once; legacy `Station 5 — Outputs (`run/outputs.rs`, cobre-io writers, …` reported unmatched; bare 'STATION 6' refused as a selector (exit 2) | 0 | PASS |
| figures | `bash plans/architecture-debt-audit/stations/cli-python/verify-figures.sh` → figures.tsv (46 figures re-measured at 077dbe2c against inventory.json / enforcement-measurements.json / verdicts.json / calibration.json) | 0 | PASS |
| cross-check | 16 figure-specific patterns over the parsed section (every quoted `<n>` located and compared to figures.tsv; the section carries no `(measured-by: …)` tags) | 0 | PASS |
| supersession | calibration.json supersessionNotes (old figure + source → new figure → command) ↔ supersession-notes.md `## SN-nn` blocks ↔ the section's Supersession notes block; every sharpen row links ≥1 note | 0 | PASS |
| wave-5-dispositions | wave-dispositions.json (4 owned ids) through `python3 plans/architecture-debt-audit/tools/check-anchors.py 'DISPOSITION ANCHOR PROBE — cli-python (verify)' --register <stub> --baseline 077dbe2c --json`; alignment, fix-shape owner, conflict variants and the disposition table + SUPERSEDED marker over the parsed section | 0 | PASS |
| i5-handoff | partI-handoff.json i5Queue rows (disposition ∈ keep|retire|sharpen, non-empty re-measure command) ↔ the section's Part-I block | 0 | PASS |
| no-timing | unit-anchored duration/throughput/speedup regexes over the section; PD entries' Measurement bullets (UNMEASURED, claim-type, layout, perf-queue.json marker); perf-queue.json rows | 0 | PASS |
| read-only-snapshot | `git status --porcelain` minus the pre-station snapshot, `.gitignore` exempt, filtered to the evaluated surfaces; `git diff --stat HEAD -- crates docs schemas scripts .github Cargo.toml` | 0 | PASS |
| py-build | `cargo check --manifest-path crates/cobre-python/Cargo.toml` (no libpython RUSTFLAGS override; cargo check never links) | 0 | PASS |
| ci-visibility | ci.yml at 077dbe2c: python job = maturin + pytest (--require-cli-binary) + `cargo test --manifest-path crates/cobre-python/Cargo.toml`; workflows running the bindings' Rust tests = 1 | 0 | PASS |

- shared-verifier: exit 0
- heading-resolution: 1 match for the dated title (BACKLOG.md L5297); inner station heading ×1; legacy 2026-08 heading ×1 deliberately unmatched; 40 entries parsed
- figures: 46 figures, 0 drift; the ticket's 17-figure list is covered and extended (census, partition, coupling triple, double-mirror, parity gate, from_config, test corpus, facade, register-side counts, build + CI evidence)
- cross-check: 43 quoted values located across 16 passages, 0 mismatches; a figure copied from the epic body, BACKLOG.md or Part-I I.5 would fail here
- supersession: 10 notes (SN-01 I.5 coupling, SN-02 writer call sites 41/29 → 39/23, SN-03 parity 4-of-17 → 18/18, SN-04 CD-025 anchors, SN-05 CD-029 parity half, SN-06 CI visibility 22 → 19, SN-07 runtime layer, SN-08 CD-002, SN-09 CD-009, SN-10 report/summary); the from_config figure (1 → 5 non-test sites) is inventory.json → supersessions with its command
- wave-5-dispositions: 4 rows (4 sharpen); 4 baseline anchors resolved; CD-025 fix-shape owner cobre-io with both cobre-sddp and cli-local variants tagged conflicts (never a direction); CD-029's L3 PrepPhase destination held as an owner question; the three parity enforcement layers named under CD-025
- i5-handoff: 8 I.5 rows (Counter({'sharpen': 5, 'keep': 2, 'retire': 1})); the parity-coverage claim is not an I.5 row at this station — its two enforcement layers plus the Rust companion are asserted on the CD-025 disposition prose above
- no-timing: 999 section lines scanned, 0 figures; 7 PD entries UNMEASURED with claim-type + layout; 3 queued to perf-queue.json (the perf lens is queue-only)
- read-only-snapshot: no pre-station snapshot on disk (the tree was clean when the station opened); carried-in exemption `.gitignore` (dirty at a136840d per the ticket, clean at 077dbe2c); 8 untracked plans/ path(s) are the station's own writes; diff over the evaluated surfaces empty
- py-build: exit 0; the RUSTFLAGS override the ticket describes is needed only to RUN the crate's Rust tests locally, and CI runs them with `--no-default-features --features highs` (ci.yml:567).
- ci-visibility: the ticket's premise ('no workflow invokes cargo test or nextest against that manifest') is SUPERSEDED at the pin — ci.yml:562-568 runs the 19 Rust tests; the block therefore asserts the measured CI-visible state and fails if that step disappears (the Cleared cross-reference in the section rests on it; supersession-notes.md SN-06).
- the perf lens is queue-only at this station: every figure is owned by the performance sweep (`perf-queue.json`), none is asserted in the register.

Superseded figures (old → new, command in supersession-notes.md): SN-01 Part-I I.5: the CLI's cobre_sddp import surface: '~70 imports' (plans/generalizing/beyond-sddp-generalization… → 97 `cobre_sddp` occurrences in 10 files, 70 of them `use` lines (88 di…; SN-02 per-side writer call-site figures: 41 CLI / 29 Python `write_*(` call sites and 26 distinct Pyt… → 39 CLI call sites (16 terminal `write_line`, 23 writer sites) / 23 Pyt…; SN-03 the parity script's per-side name set: '4 of 17 writers seen; use-line skip + allowlist' (register… → import-resolving since fc81427a: 18 CLI / 18 Python names, 18 in both,…; SN-04 CD-025 line anchors: write_training_outputs at outputs.rs:57 (register) / :58 (ti… → outputs.rs:58; run.rs:532; five `_if_any` helpers at run.rs:594, :614,…; SN-05 CD-029's Python-parity half: 'zero equivalent in cobre-python/src/io.rs' (register CD-029… → 14 boundary references in io.rs — cobre.io.validate reconciles the bou…; SN-06 cobre-python Rust `#[test]` CI visibility: 'compiled and run by nothing in CI (22 tests)' (ticket; mirr… → 19 tests (errors.rs 3, policy.rs 1, run.rs 13, schema.rs 2) run by ci.…; SN-07 the runtime file-set parity layer: 'its _cli_binary() fixture skips when no compiled cobre bina… → guard is conftest.py::cli_binary → _cobre_cli.resolve_cli_binary(requi…; SN-08 CD-002's 'shrinks once CD-001 lands' expectation and its line anchors: 'Shrinks once CD-001 lands' (register CD-002); broadcast_and… → CD-001 landed (b051c410, retired at the sddp gate 2026-09-18) and the…; SN-09 CD-009 line anchors: policy.rs:151, :180, :273 (register) / :167, :196, :316 (tic… → policy.rs:172, :201, :334 (WarmStart, Resume, load_policy_for_simulati…; SN-10 Part-I I.5: the `report` / `summary` subcommands and the Command enum: seven Command variants incl. `report` and `summary` (Part-I… → five variants (Init, Run, Validate, Schema, Version) at main.rs:67 — `….
Carried-in dirty tracked path: none at 077dbe2c (the ticket's `.gitignore` premise is stale; the exemption rule stays).
