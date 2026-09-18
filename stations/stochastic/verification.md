# Station verification — stochastic

Station baseline: `a136840d4f2ea137f685f0af6dac04254b983b60` (inventory.json; every entry, anchor and figure is measured there).
Register pin: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (the drift rule ties HEAD to it; provenance bullets resolve there).
Declared section title (argv[2]): `★ QUALITY EVALUATION (2026-09, baseline a136840d) — stochastic`
Resolved by slug `stochastic` via the em-dash tail: `★ QUALITY EVALUATION (2026-09, baseline a136840d) — stochastic`

The section is resolved by the station slug, not by the spelled-out title, so the six
later stations reuse this verifier unchanged and this record is byte-stable across runs
at the pin with the same invocation (a re-run leaves the committed file unmodified). The
ticket's test commands quote an older `★ STATION: …` title; the heading on disk is the
`★ QUALITY EVALUATION (…) — <slug>` form the calibration ticket wrote, and the slug
resolves it either way.

| # | Check | Command | Exit | Result |
| --- | --- | --- | --- | --- |
| 1 | check-anchors | `python3 plans/architecture-debt-audit/tools/check-anchors.py stochastic` | 0 | PASS |
| 2 | check-reraise | `python3 plans/architecture-debt-audit/tools/check-reraise.py stochastic` | 0 | PASS |
| 3 | fields-check | `python3 plans/architecture-debt-audit/tools/fields-check.py stochastic` | 0 | PASS |
| 4 | register | `python3 plans/architecture-debt-audit/tools/station_verify.py register . stochastic` | 0 | PASS |
| 5 | inventory-set-equality | `python3 plans/architecture-debt-audit/tools/station_verify.py inventory inventory.json .` | 0 | PASS |
| 6 | infra-genericity | `python3 plans/architecture-debt-audit/tools/station_verify.py genericity . partI-handoff.json` | 0 | PASS |
| 7 | read-only-workspace | `python3 plans/architecture-debt-audit/tools/station_verify.py readonly . 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` | 0 | PASS |

Reserved-seam re-raises (no entry re-raises a ratified seam without a sanction
citation) are enforced twice above: `check-reraise` against the mirror corpus, and
the `register` denylist against the Reserved-seam register.

Test suite (`python3 -m unittest stations/stochastic/tests/test_station.py`): FAIL (exit 1)

Per-check logs: `plans/architecture-debt-audit/stations/stochastic/.<check>.log`.
