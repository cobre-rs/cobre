# Station verification — build-ci

Station baseline: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (inventory.json; every entry, anchor and figure is measured there).
Register pin: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (the drift rule ties HEAD to it; provenance bullets resolve there).
Declared section title (argv[2]): `STATION 7 — build/CI/scripts/schemas/docs (2026-09)`
Resolved by slug `build-ci` via the em-dash tail: `★ QUALITY EVALUATION (2026-09, baseline a136840d) — build-ci`

The section is resolved by the station slug, not by the spelled-out title, so the six
later stations reuse this verifier unchanged and this record is byte-stable across runs
at the pin with the same invocation (a re-run leaves the committed file unmodified). The
ticket's test commands quote an older `★ STATION: …` title; the heading on disk is the
`★ QUALITY EVALUATION (…) — <slug>` form the calibration ticket wrote, and the slug
resolves it either way.

| # | Check | Command | Exit | Result |
| --- | --- | --- | --- | --- |
| 1 | check-anchors | `python3 plans/architecture-debt-audit/tools/check-anchors.py build-ci` | 0 | PASS |
| 2 | check-reraise | `python3 plans/architecture-debt-audit/tools/check-reraise.py build-ci` | 0 | PASS |
| 3 | fields-check | `python3 plans/architecture-debt-audit/tools/fields-check.py build-ci` | 0 | PASS |
| 4 | register | `python3 plans/architecture-debt-audit/tools/station_verify.py register . build-ci` | 0 | PASS |
| 5 | inventory-set-equality | `python3 plans/architecture-debt-audit/tools/station_verify.py inventory inventory.json .` | 0 | PASS |
| 6 | infra-genericity | `python3 plans/architecture-debt-audit/tools/station_verify.py genericity . partI-handoff.json` | 0 | PASS |
| 7 | read-only-workspace | `python3 plans/architecture-debt-audit/tools/station_verify.py readonly . 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` | 0 | PASS |

Reserved-seam re-raises (no entry re-raises a ratified seam without a sanction
citation) are enforced twice above: `check-reraise` against the mirror corpus, and
the `register` denylist against the Reserved-seam register.

Test suite (`python3 -m unittest stations/build-ci/tests/test_station.py`): PASS (exit 0)

Per-check logs: `plans/architecture-debt-audit/stations/build-ci/.<check>.log`.

## Station-specific checks — build-ci

Station baseline `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (inventory.json); section `STATION 7 — build/CI/scripts/schemas/docs (2026-09)` resolved once at BACKLOG.md:6299.

| Check | Command | Exit | Result |
| --- | --- | --- | --- |
| check-anchors | `python3 plans/architecture-debt-audit/tools/check-anchors.py 'STATION 7 — build/CI/scripts/schemas/docs (2026-09)'` | 0 | PASS |
| check-reraise | `python3 plans/architecture-debt-audit/tools/check-reraise.py 'STATION 7 — build/CI/scripts/schemas/docs (2026-09)'` | 0 | PASS |
| fields-check | `python3 plans/architecture-debt-audit/tools/fields-check.py 'STATION 7 — build/CI/scripts/schemas/docs (2026-09)'` | 0 | PASS |
| fields-check-alignment | `python3 plans/architecture-debt-audit/tools/fields-check.py --require Alignment 'STATION 7 — build/CI/scripts/schemas/docs (2026-09)'` | 0 | PASS |
| verify-station | `bash plans/architecture-debt-audit/tools/verify-station.sh build-ci 'STATION 7 — build/CI/scripts/schemas/docs (2026-09)'` | 0 | PASS |
| census | `re-measured at 077dbe2c: MPICH grep, gate glob, schemas/*.json + policy.fbs, build.rs, workflows, jobs: block awk, negative-grep loop, quality-report.sh invoker; gate-census.json completeness; inventory.json equality; section cross-check` | 0 | PASS |
| oracle-genericity | `bash scripts/ci/check-infra-genericity.sh` | 0 | PASS |
| oracle-doc-paths | `bash scripts/ci/check-doc-paths.sh` | 0 | PASS |
| oracle-doc-voice | `python3 scripts/ci/check_doc_voice.py` | 0 | PASS |
| premises | `EXCLUDED_FILES=() · SCAN_DIRS = 5 crates · CLAUDE.md names no cobre-model/cobre-network (at 077dbe2c); section body: one Alignment per entry, two-site fix-shapes, reviewer rating on every downgrade` | 0 | PASS |
| read-only | `git status --porcelain minus the pre-station snapshot, guarded prefixes .github/ scripts/ schemas/ docs/ Cargo.toml; git diff --stat HEAD over the same` | 0 | PASS |

Re-measured at `077dbe2c`: 14 ci.yml jobs (the raw `grep -nE '^  [a-z0-9-]+:$'` census command matches 15, the extra hit being the `push:` trigger under `on:`) · 17 gate files from `scripts/ci/*.sh scripts/ci/*.py scripts/ci/lib/*.sh` · 1 without direct wiring (check-comment-bloat.sh, invoked by scripts/ci/quality-report.sh:132) · 8x `Build MPICH from source` in ci.yml (10 across the workflows, no `.github/actions/`) · 18 `schemas/*.json` + `crates/cobre-io/schemas/policy.fbs` (no root `schemas/policy.fbs`) · 3 build.rs · 8 workflows.

Census re-measurement: OK 17 gate files / 1 without direct wiring / 14 ci.yml jobs (raw grep 15) / 8x MPICH in ci.yml (10 total) / 18 schemas + policy.fbs / 3 build.rs / 8 workflows

Premises: OK 19 entries: one Alignment each, two-site fix-shapes complete, 5 downgrades carry the reviewer's rating; EXCLUDED_FILES=() present; SCAN_DIRS holds 5 crates; CLAUDE.md names neither cobre-model nor cobre-network.

Carried-in dirty tracked paths (not station-caused): `none — no pre-station snapshot; the tree was clean when the station opened`; the `.gitignore` exemption the ticket describes stays in the rule but nothing exercises it at the pin.
