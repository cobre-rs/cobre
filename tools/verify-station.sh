#!/usr/bin/env bash
# verify-station.sh <station-slug> "<section-title>" — reusable per-station verifier.
#
# Resolves the register section by the SLUG (argv[1]) through
# backlog_parse.find_section's em-dash-tail match, so the six later stations
# (stochastic, solver-comm, sddp, cli-python, build-ci, test-corpus) invoke this
# file unchanged with only their own slug. argv[2] is the human-facing section
# title recorded in verification.md; the checkers never need it because the slug
# resolves the '★ QUALITY EVALUATION (…) — <slug>' heading regardless of its date
# or SHA. No cobre-core/cobre-io path is hardcoded here — every tree root comes
# from the station's own inventory.json and every retired id from the register.
#
# Seven checks: the three harness checkers plus the four station_verify.py
# subcommands (register, inventory, genericity, readonly). Read-only w.r.t. the
# codebase: writes only under plans/architecture-debt-audit/stations/<slug>/ (the
# per-check logs and verification.md). Exits non-zero if any check fails.
# -e is deliberately omitted: this is a multi-check runner that must execute all seven
# checks and aggregate their exit codes (captured per check), not abort on the first one.
set -uo pipefail

STATION="${1:?station slug, e.g. core-io}"
SECTION="${2:?BACKLOG section title (recorded verbatim in verification.md)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
AUDIT="$ROOT/plans/architecture-debt-audit"
TOOLS="$AUDIT/tools"
DIR="$AUDIT/stations/$STATION"
BACKLOG="$AUDIT/BACKLOG.md"
REL="plans/architecture-debt-audit"

[[ -d "$DIR" ]] || { echo "FAIL: no station dir $DIR"; exit 2; }
[[ -f "$DIR/inventory.json" ]] || { echo "FAIL: no inventory.json in $DIR"; exit 2; }
[[ -f "$DIR/partI-handoff.json" ]] || { echo "FAIL: no partI-handoff.json in $DIR"; exit 2; }

BASE="$(python3 -c 'import sys,pathlib; sys.path.insert(0,sys.argv[1]); from lib import backlog_parse as b; print(b.parse_baseline(b.read_register(pathlib.Path(sys.argv[2]))))' "$TOOLS" "$BACKLOG")"

FAILED=0
RESULTS=()

record() { # record <label> <display-command> <exit-code>
  local label="$1" cmd="$2" code="$3"
  if [[ "$code" -eq 0 ]]; then
    RESULTS+=("PASS|$label|$cmd|$code")
  else
    RESULTS+=("FAIL|$label|$cmd|$code")
    FAILED=1
  fi
}

# Checks 1-3: the three harness checkers, resolved by SLUG (argv[1]).
for c in check-anchors check-reraise fields-check; do
  python3 "$TOOLS/$c.py" "$STATION" >"$DIR/.$c.log" 2>&1
  record "$c" "python3 $REL/tools/$c.py $STATION" "$?"
done

# Check 4: register integrity — Alignment vocabulary (independent of fields-check),
# a non-empty/parseable section, and the do-not-touch denylist (hard-retired ids +
# ratified reserved seams). See station_verify.py::_cmd_register.
python3 "$TOOLS/station_verify.py" register "$AUDIT" "$STATION" >"$DIR/.register.log" 2>&1
record register "python3 $REL/tools/station_verify.py register . $STATION" "$?"

# Check 5: inventory set equality — the frozen module census reconstructs the exact
# .rs set; diff it against the live tree, both directions reported separately.
python3 "$TOOLS/station_verify.py" inventory "$DIR/inventory.json" "$ROOT" >"$DIR/.inventory.log" 2>&1
record inventory-set-equality "python3 $REL/tools/station_verify.py inventory inventory.json ." "$?"

# Check 6: the genericity gate as the executable oracle behind Part-I item 6 — the
# gate exits 0, EXCLUDED_FILES=() still holds, and I.3-6 is retire/sharpen.
python3 "$TOOLS/station_verify.py" genericity "$ROOT" "$DIR/partI-handoff.json" >"$DIR/.genericity.log" 2>&1
record infra-genericity "python3 $REL/tools/station_verify.py genericity . partI-handoff.json" "$?"

# Check 7: read-only workspace — no NEW worktree modification and no commit since the
# baseline touches an evaluated surface (crates, docs, scripts, .github, schemas,
# Cargo.toml/lock, examples, tests); the carried-in .gitignore and the station's own
# plans/ writes are exempt.
python3 "$TOOLS/station_verify.py" readonly "$ROOT" "$BASE" >"$DIR/.read-only.log" 2>&1
record read-only-workspace "python3 $REL/tools/station_verify.py readonly . $BASE" "$?"

# The station's own test module is the executable proof of the seven checks; the six
# later stations reuse this verifier by running it against their own slug. Run it from
# the audit dir so the reuse command is literally `python3 -m unittest
# stations/<slug>/tests/test_station.py`.
( cd "$AUDIT" && python3 -m unittest "stations/$STATION/tests/test_station.py" ) \
  >"$DIR/.test-suite.log" 2>&1
test_rc=$?
[[ "$test_rc" -eq 0 ]] || FAILED=1
test_result="$([[ "$test_rc" -eq 0 ]] && echo PASS || echo FAIL)"

# Render the recorded result table into verification.md.
resolved_heading="$(python3 -c 'import sys,pathlib; sys.path.insert(0,sys.argv[1]); from lib import backlog_parse as b; print(b.find_section(b.read_register(pathlib.Path(sys.argv[2])), sys.argv[3]).heading)' "$TOOLS" "$BACKLOG" "$STATION" 2>/dev/null || echo "$SECTION")"
{
  echo "# Station verification — $STATION"
  echo
  echo "Baseline: \`$BASE\` (the register pin; the drift rule ties HEAD to it)."
  echo "Declared section title (argv[2]): \`$SECTION\`"
  echo "Resolved by slug \`$STATION\` via the em-dash tail: \`$resolved_heading\`"
  echo
  echo "The section is resolved by the station slug, not by the spelled-out title, so the six"
  echo "later stations reuse this verifier unchanged and this record is byte-stable across runs"
  echo "at the pin with the same invocation (a re-run leaves the committed file unmodified). The"
  echo "ticket's test commands quote an older \`★ STATION: …\` title; the heading on disk is the"
  echo "\`★ QUALITY EVALUATION (…) — <slug>\` form the calibration ticket wrote, and the slug"
  echo "resolves it either way."
  echo
  echo "| # | Check | Command | Exit | Result |"
  echo "| --- | --- | --- | --- | --- |"
  i=0
  for row in "${RESULTS[@]}"; do
    i=$((i + 1))
    IFS='|' read -r res label cmd code <<<"$row"
    echo "| $i | $label | \`$cmd\` | $code | $res |"
  done
  echo
  echo "Reserved-seam re-raises (no entry re-raises a ratified seam without a sanction"
  echo "citation) are enforced twice above: \`check-reraise\` against the mirror corpus, and"
  echo "the \`register\` denylist against the Reserved-seam register."
  echo
  echo "Test suite (\`python3 -m unittest stations/$STATION/tests/test_station.py\`): $test_result (exit $test_rc)"
  echo
  echo "Per-check logs: \`$REL/stations/$STATION/.<check>.log\`."
} >"$DIR/verification.md"

if [[ "$FAILED" -eq 0 ]]; then
  echo "verify-station.sh $STATION: PASS (7/7 checks)"
else
  echo "verify-station.sh $STATION: FAIL"
  printf '%s\n' "${RESULTS[@]}" | awk -F'|' '$1=="FAIL"{print "  FAIL "$2" (exit "$4")"}'
fi
exit "$FAILED"
