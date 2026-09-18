# Station verification — solver-comm

Station baseline: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (inventory.json; every entry, anchor and figure is measured there).
Register pin: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (the drift rule ties HEAD to it; provenance bullets resolve there).
Declared section title (argv[2]): `★ QUALITY EVALUATION (2026-09, baseline a136840d) — solver-comm`
Resolved by slug `solver-comm` via the em-dash tail: `★ QUALITY EVALUATION (2026-09, baseline a136840d) — solver-comm`

The section is resolved by the station slug, not by the spelled-out title, so the six
later stations reuse this verifier unchanged and this record is byte-stable across runs
at the pin with the same invocation (a re-run leaves the committed file unmodified). The
ticket's test commands quote an older `★ STATION: …` title; the heading on disk is the
`★ QUALITY EVALUATION (…) — <slug>` form the calibration ticket wrote, and the slug
resolves it either way.

| # | Check | Command | Exit | Result |
| --- | --- | --- | --- | --- |
| 1 | check-anchors | `python3 plans/architecture-debt-audit/tools/check-anchors.py solver-comm` | 0 | PASS |
| 2 | check-reraise | `python3 plans/architecture-debt-audit/tools/check-reraise.py solver-comm` | 0 | PASS |
| 3 | fields-check | `python3 plans/architecture-debt-audit/tools/fields-check.py solver-comm` | 0 | PASS |
| 4 | register | `python3 plans/architecture-debt-audit/tools/station_verify.py register . solver-comm` | 0 | PASS |
| 5 | inventory-set-equality | `python3 plans/architecture-debt-audit/tools/station_verify.py inventory inventory.json .` | 0 | PASS |
| 6 | infra-genericity | `python3 plans/architecture-debt-audit/tools/station_verify.py genericity . partI-handoff.json` | 0 | PASS |
| 7 | read-only-workspace | `python3 plans/architecture-debt-audit/tools/station_verify.py readonly . 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` | 0 | PASS |

Reserved-seam re-raises (no entry re-raises a ratified seam without a sanction
citation) are enforced twice above: `check-reraise` against the mirror corpus, and
the `register` denylist against the Reserved-seam register.

Test suite (`python3 -m unittest stations/solver-comm/tests/test_station.py`): PASS (exit 0)

Per-check logs: `plans/architecture-debt-audit/stations/solver-comm/.<check>.log`.

## Station-specific checks — solver-comm

| Check | Command | Exit | Result |
| --- | --- | --- | --- |
| sanctioned-polarity | shared-memory only in Cleared w/ `docs/design/reserved-seams-and-deferred-debt.md` citation; cut-sync only as the E5 dup-of, no id | 0 | PASS |
| cut-sync-anchors | `git show 077dbe2c:crates/cobre-sddp/src/cut/cut_sync.rs` L243/400/495/581 | 0 | PASS |
| handoff-shape | `python3 plans/architecture-debt-audit/stations/solver-comm/verify-handoffs.py` (E5/E7/E9/E10, perf-queue.json) | 0 | PASS |
| blind-spot | `bash scripts/ci/check-infra-genericity.sh` exit 0 AND `grep -n cut_nz_per_col crates/cobre-solver/src/freeze.rs` → 22 137 138 141 188 (cfg(test) at 244) | 0 | PASS |
| read-only-workspace | `git status --porcelain --untracked-files=no` minus the carried-in paths, filtered to the evaluated surfaces | 0 | PASS |

- sanctioned-polarity: 20 entries scanned for the 6 shared-memory identifiers and the 3 superseded methods
- cut-sync-anchors: sync_cuts L243, pack_local_records L400, sync_packed_records L495, sync_level_records L581
- handoff-shape: perf rows carry layout ∈ {4t, 2x2}, exercising call sites, no measured number; E9 covers the five StageTemplate fields with resolving types.rs anchors; E7 names gateScript/evadingIdentifier/patternEvaded
- blind-spot: gate exit 0; production hits above cfg(test) L244: 22 137 138 141 188
- read-only-workspace: carried-in: none (the shared verifier's row 7 already ties HEAD to the register pin; this row is the porcelain-vs-carried-in restatement).
- perf claims are queued to the perf sweep UNMEASURED, each with a layout and its exercising call sites: see `perf-queue.json`.
