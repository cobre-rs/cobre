# Station verification — test-corpus

Station baseline: `{'sha': '077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c', 'describe': 'v0.15.0-100-g077dbe2c', 'measuredOn': '2026-09-19', 'head': 'fca70afcf6586dfe8a68b4d04e500bf576932b17', 'source': "plans/architecture-debt-audit/BACKLOG.md header `Baseline: <sha40> (pinned …)` — the register pin; the ticket's a136840d (v0.15.0-1-ga136840d) is the superseded scaffold pin", 'assertion': "every evaluated surface at HEAD is identical to the pin (git diff --quiet <pin> HEAD -- crates .github scripts schemas examples tests docs Cargo.toml Cargo.lock, the ID-free mirror exempt); censuses read the pin's tree, toolchain figures run on the worktree"}` (inventory.json; every entry, anchor and figure is measured there).
Register pin: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (the drift rule ties HEAD to it; provenance bullets resolve there).
Declared section title (argv[2]): `★ QUALITY EVALUATION (2026-09, baseline a136840d) — test-corpus`
Resolved by slug `test-corpus` via the em-dash tail: `★ QUALITY EVALUATION (2026-09, baseline a136840d) — test-corpus`

The section is resolved by the station slug, not by the spelled-out title, so the six
later stations reuse this verifier unchanged and this record is byte-stable across runs
at the pin with the same invocation (a re-run leaves the committed file unmodified). The
ticket's test commands quote an older `★ STATION: …` title; the heading on disk is the
`★ QUALITY EVALUATION (…) — <slug>` form the calibration ticket wrote, and the slug
resolves it either way.

| # | Check | Command | Exit | Result |
| --- | --- | --- | --- | --- |
| 1 | check-anchors | `python3 plans/architecture-debt-audit/tools/check-anchors.py test-corpus` | 0 | PASS |
| 2 | check-reraise | `python3 plans/architecture-debt-audit/tools/check-reraise.py test-corpus` | 0 | PASS |
| 3 | fields-check | `python3 plans/architecture-debt-audit/tools/fields-check.py test-corpus` | 0 | PASS |
| 4 | register | `python3 plans/architecture-debt-audit/tools/station_verify.py register . test-corpus` | 0 | PASS |
| 5 | inventory-set-equality | `python3 plans/architecture-debt-audit/tools/station_verify.py inventory inventory.json .` | 0 | PASS |
| 6 | infra-genericity | `python3 plans/architecture-debt-audit/tools/station_verify.py genericity . partI-handoff.json` | 0 | PASS |
| 7 | read-only-workspace | `python3 plans/architecture-debt-audit/tools/station_verify.py readonly . 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` | 0 | PASS |

Reserved-seam re-raises (no entry re-raises a ratified seam without a sanction
citation) are enforced twice above: `check-reraise` against the mirror corpus, and
the `register` denylist against the Reserved-seam register.

Test suite (`python3 -m unittest stations/test-corpus/tests/test_station.py`): PASS (exit 0)

Per-check logs: `plans/architecture-debt-audit/stations/test-corpus/.<check>.log`.

## Station-specific checks — test-corpus

Section resolved by title `★ QUALITY EVALUATION (2026-09, baseline a136840d) — test-corpus` to exactly one heading (BACKLOG.md line 6842); station baseline `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (inventory.json; every recount below is measured in that tree, and the worktree `find` is asserted equal to it). The five station assertions the shared verifier cannot know, then the read-only proof.

| # | Check | Command | Exit | Result |
| --- | --- | --- | --- | --- |
| 1 | check-anchors | `python3 plans/architecture-debt-audit/tools/check-anchors.py '★ QUALITY EVALUATION (2026-09, baseline a136840d) — test-corpus'` | 0 | PASS |
| 2 | check-reraise | `python3 plans/architecture-debt-audit/tools/check-reraise.py '★ QUALITY EVALUATION (2026-09, baseline a136840d) — test-corpus'` | 0 | PASS |
| 3 | fields-check | `python3 plans/architecture-debt-audit/tools/fields-check.py '★ QUALITY EVALUATION (2026-09, baseline a136840d) — test-corpus'` | 0 | PASS |
| 4 | fields-check-alignment | `python3 plans/architecture-debt-audit/tools/fields-check.py --require Alignment '★ QUALITY EVALUATION (2026-09, baseline a136840d) — test-corpus'` | 0 | PASS |
| 5 | verify-station | `bash plans/architecture-debt-audit/tools/verify-station.sh test-corpus '★ QUALITY EVALUATION (2026-09, baseline a136840d) — test-corpus'` | 0 | PASS |
| 6 | figure-completeness | `verify-test-corpus.py: 15 figure records, 9 ta-2.1 + 6 prose, command / value / reason` | 0 | PASS |
| 7 | both-definitions-ran | `verify-test-corpus.py: recount depth-1 vs recursive tests/*.rs per crate at the station baseline; cobre-sddp and cobre-io pairs must differ` | 0 | PASS |
| 8 | slow-tests-census-tracked | `verify-test-corpus.py: git grep census vs `grep -rl slow-tests crates/` (target/ artifacts excluded non-vacuously)` | 0 | PASS |
| 9 | entry-definition-claimkind-provenance | `verify-test-corpus.py: every entry names its measurement definition and claim kind; every yardstick-quoted integer resolves to inventory.json` | 0 | PASS |
| 10 | mirror-items-dup-of-only | `verify-test-corpus.py: the three open mirror items appear exactly once, as dup-of merges with resolving anchors and no fresh TD id` | 0 | PASS |
| 11 | read-only-workspace | `git status --porcelain (only the station's own untracked run artifacts) && git diff --stat HEAD -- crates .github docs schemas scripts (empty)` | 0 | PASS |

Measured at the station baseline:

- figures: 15 records, 9 ta-2.1 + 6 ta-2.3/ta-3.2, {'measured': 15}
- definition pairs (binaries vs files) at the pin: cobre-io 12 vs 13, cobre-sddp 40 vs 56
- slow-tests census: 13 git-tracked .rs files at the pin vs naive recursive grep 75 files (56 under target/)
- entries: 8 minted (TD-074, TD-075, TD-076, TD-077, CD-116, CD-117, CD-118, OD-051); claim kinds ['prose-drift', 'tree-fact']
- dup-of merges: 3 mirror items, each once, no fresh TD id

Ticket premises superseded at this baseline (recorded, not edited): the cobre-sddp pair is 40 binaries vs 56 files (ticket: 37 vs 53); the naive `grep -rl slow-tests crates/` sees 75 files with 56 under target/ (ticket: 73 / 54) against 13 tracked; no pre-station porcelain snapshot exists and the worktree carries no modified .gitignore, so cleanliness is judged as 'nothing beyond the station's own directory'; entries carry the definition inside `- **Measurement:**` and the kind inside `- **Claim kind:**` (the register's bold-bullet field shape), and the dup-of merges are bullets under their own heading rather than id-less entries; the figure-provenance rule is applied to integers that appear in the yardstick (the frozen-snapshot hazard) so a defender-measured line count is not a false failure.

Station-specific result: PASS (11/11 checks).
