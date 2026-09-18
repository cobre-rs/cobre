# Station verification — sddp

Station baseline: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (inventory.json; every entry, anchor and figure is measured there).
Register pin: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (the drift rule ties HEAD to it; provenance bullets resolve there).
Declared section title (argv[2]): `★ QUALITY EVALUATION (2026-09, baseline a136840d) — sddp`
Resolved by slug `sddp` via the em-dash tail: `★ QUALITY EVALUATION (2026-09, baseline a136840d) — sddp`

The section is resolved by the station slug, not by the spelled-out title, so the six
later stations reuse this verifier unchanged and this record is byte-stable across runs
at the pin with the same invocation (a re-run leaves the committed file unmodified). The
ticket's test commands quote an older `★ STATION: …` title; the heading on disk is the
`★ QUALITY EVALUATION (…) — <slug>` form the calibration ticket wrote, and the slug
resolves it either way.

| # | Check | Command | Exit | Result |
| --- | --- | --- | --- | --- |
| 1 | check-anchors | `python3 plans/architecture-debt-audit/tools/check-anchors.py sddp` | 0 | PASS |
| 2 | check-reraise | `python3 plans/architecture-debt-audit/tools/check-reraise.py sddp` | 0 | PASS |
| 3 | fields-check | `python3 plans/architecture-debt-audit/tools/fields-check.py sddp` | 0 | PASS |
| 4 | register | `python3 plans/architecture-debt-audit/tools/station_verify.py register . sddp` | 0 | PASS |
| 5 | inventory-set-equality | `python3 plans/architecture-debt-audit/tools/station_verify.py inventory inventory.json .` | 0 | PASS |
| 6 | infra-genericity | `python3 plans/architecture-debt-audit/tools/station_verify.py genericity . partI-handoff.json` | 0 | PASS |
| 7 | read-only-workspace | `python3 plans/architecture-debt-audit/tools/station_verify.py readonly . 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` | 0 | PASS |

Reserved-seam re-raises (no entry re-raises a ratified seam without a sanction
citation) are enforced twice above: `check-reraise` against the mirror corpus, and
the `register` denylist against the Reserved-seam register.

Test suite (`python3 -m unittest stations/sddp/tests/test_station.py`): PASS (exit 0)

Per-check logs: `plans/architecture-debt-audit/stations/sddp/.<check>.log`.

## Station-specific checks — sddp

| Check | Command | Exit | Result |
| --- | --- | --- | --- |
| shared-verifier | `bash plans/architecture-debt-audit/tools/verify-station.sh sddp '★ QUALITY EVALUATION (2026-09, baseline a136840d) — sddp'` | 0 | PASS |
| partition | inventory.json sub-stations 5a-5d as a multiset vs `find crates/cobre-sddp/src -name '*.rs'` | 0 | PASS |
| dispositions | wave-dispositions.json (22 rows) through `python3 plans/architecture-debt-audit/tools/check-anchors.py 'DISPOSITION ANCHOR PROBE — sddp (verify)' --register <stub> --baseline 077dbe2c --json` and `git rev-parse --verify <sha>^{commit}`; polarity over the parsed section | 0 | PASS |
| lp-reconciliation | measurements/lp-inventory.json vs `find crates/cobre-sddp/src/lp -name '*.rs'` | 0 | PASS |
| no-timing | unit-anchored duration/throughput/speedup regexes over the section; PD entries' Measurement bullets; perf-queue.json layouts | 0 | PASS |
| read-only-snapshot | `git status --porcelain` minus the pre-station snapshot, filtered to the evaluated surfaces; `git diff --stat HEAD -- crates docs schemas scripts` | 0 | PASS |

- shared-verifier: folded per-check exits: check-anchors 0, check-reraise 0, fields-check 0, register 0, inventory-set-equality 0, infra-genericity 0, read-only-workspace 0, test-suite 0
- partition: 163 modules, each claimed once: 5a 28 / 5b 30 / 5c 58 / 5d 47; doubly-assigned 0, unassigned 0, phantom 0
- dispositions: 22 rows: 11 keep / 3 retire / 8 sharpen; 23 anchors resolved through check-anchors.py at 077dbe2c; retire commits CD-001 -> b051c410, CD-003-construction-hop -> 4075c4e8, CD-006 -> 3f4c3db3; live entries 44, none headed by CD-001/CD-003/CD-006/PD-004
- lp-reconciliation: 30 lp/ modules listed once each; totals 44682 lines / 11655 non-test; every record carries top_symbols and non_test_lines <= total_lines == wc -l
- no-timing: 967 section lines scanned by 3 unit-anchored regexes, 0 figures; 14 PD entries UNMEASURED with layout in 4t|2x2; 9 queued to perf-queue.json, plus the PD-004 existence row
- read-only-snapshot: no pre-station porcelain snapshot on disk (the tree was clean when the station opened); carried-in exemption `.gitignore`; porcelain lines outside the snapshot are the station's own plans/ writes; `git diff --stat HEAD -- crates docs schemas scripts` empty
- the perf lens is queue-only at this station: every figure is owned by the performance sweep (`perf-queue.json`), none is asserted in the register.
