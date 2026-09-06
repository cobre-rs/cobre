#!/usr/bin/env bash
# E01 acceptance gate: proves every harness instrument works and that building it
# touched no tracked evaluation surface. Exits 0 only when every check passes;
# writes the per-check table to stations/harness-verification.md either way.
set -euo pipefail
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
AUDIT="$ROOT/plans/architecture-debt-audit"
T="$AUDIT/tools"; FX="$T/fixtures"
cd "$ROOT"
FAIL=0; RESULTS=()

record() { RESULTS+=("$1|$2|$3"); }                     # name|expected|actual

expect_exit() {                                          # expect_exit <name> <want> <substr> <cmd...>
  local name="$1" want="$2" substr="$3"; shift 3
  local out got=0
  out="$("$@" 2>&1)" || got=$?
  if [ "$got" -ne "$want" ] || { [ -n "$substr" ] && ! printf '%s\n' "$out" | grep -qF -- "$substr"; }; then
    printf 'FAIL %s: exit=%s want=%s substr=%q\n' "$name" "$got" "$want" "$substr" >&2
    FAIL=1
  fi
  record "$name" "$want" "$got"
}

# 1. every checker passes its own --self-test
for c in check-anchors check-reraise check-roadmap-dag fields-check; do
  expect_exit "$c.py --self-test" 0 "" python3 "$T/$c.py" --self-test
done

# 2. good fixture passes, seeded-bad fixture fails with the documented code and message
expect_exit "check-anchors.py good-section.md"          0 ""                     python3 "$T/check-anchors.py"     --register "$FX/good-section.md" good-section
expect_exit "check-anchors.py bad-anchor.md"            1 "anchor-missing"       python3 "$T/check-anchors.py"     --register "$FX/bad-anchor.md" bad-anchor
expect_exit "check-anchors.py NO SUCH SECTION"          3 "section-not-found"    python3 "$T/check-anchors.py"     --register "$FX/good-section.md" "NO SUCH SECTION"
expect_exit "check-reraise.py good-section.md"          0 ""                     python3 "$T/check-reraise.py"     "$FX/good-section.md"
expect_exit "check-reraise.py reraise-seeded.md"        1 "unjustified re-raise" python3 "$T/check-reraise.py"     "$FX/reraise-seeded.md"
expect_exit "fields-check.py reraise-seeded.md"         0 ""                     python3 "$T/fields-check.py"      "$FX/reraise-seeded.md"
expect_exit "fields-check.py missing-alignment.md"      1 "alignment-invalid"    python3 "$T/fields-check.py"      "$FX/missing-alignment.md"
DUP="$T/tests/.dup-fixture.md"
sed '0,/CD-903/s//CD-902/' "$FX/missing-alignment.md" > "$DUP"
expect_exit "fields-check.py duplicate-id (derived)"    4 "duplicate-id"         python3 "$T/fields-check.py"      "$DUP"
rm -f "$DUP"
expect_exit "check-roadmap-dag.py good-roadmap.md"      0 ""                     python3 "$T/check-roadmap-dag.py" --backlog "$FX/good-roadmap.md"
expect_exit "check-roadmap-dag.py cyclic-roadmap.md"    1 "cycle"                python3 "$T/check-roadmap-dag.py" --backlog "$FX/cyclic-roadmap.md"
seeded=0; for r in "${RESULTS[@]}"; do IFS='|' read -r _ want _ <<<"$r"; [ "$want" -ne 0 ] && seeded=$((seeded + 1)); done
[ "$seeded" -ge 5 ] || { echo "FAIL: fewer than five seeded-failure cases ran" >&2; FAIL=1; }

# 3. perf-run.sh --dry-run prints the invocation, writes nothing, touches no deck
DECK="${COBRE_PERF_DECK:-$HOME/git/cobre-bridge/example/cobre_reduzido_2}"
deck_digest() { find "$1" -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1; }   # same recipe as perf-run.sh
tree_digest() { find "$AUDIT/measurements" -type f -printf '%p %s\n' 2>/dev/null | sort | sha256sum | cut -d' ' -f1; }
deck_before="$(deck_digest "$DECK")"; tree_before="$(tree_digest)"
dry_rc=0; dry="$(bash "$T/perf-run.sh" --dry-run CAL 4t 2>&1)" || dry_rc=$?
record "perf-run.sh --dry-run CAL 4t" 0 "$dry_rc"; [ "$dry_rc" -eq 0 ] || FAIL=1
printf '%s\n' "$dry" | grep -qE 'cobre run .*--threads 4( |$)' || { echo "FAIL dry-run: no '--threads 4' invocation" >&2; FAIL=1; }
printf '%s\n' "$dry" | grep -qF -- '--quiet'                   || { echo "FAIL dry-run: no --quiet" >&2; FAIL=1; }
printf '%s\n' "$dry" | grep -qF 'measurements/CAL'             || { echo "FAIL dry-run: target dir not printed" >&2; FAIL=1; }
[ "$(tree_digest)" = "$tree_before" ]        || { echo "FAIL dry-run created or changed files under measurements/" >&2; FAIL=1; }
[ "$(deck_digest "$DECK")" = "$deck_before" ] || { echo "FAIL dry-run mutated the source deck" >&2; FAIL=1; }
DECK_NOTE="sha256 $deck_before"
if grep -q 'deck_sha256' "$AUDIT/measurements/CAL/env.txt" 2>/dev/null; then
  grep -qF "$deck_before" "$AUDIT/measurements/CAL/env.txt" \
    && DECK_NOTE="$DECK_NOTE, byte-identical to the calibration stamp" \
    || { echo "FAIL deck-mutated since calibration (env.txt deck_sha256 differs)" >&2; FAIL=1; }
else
  echo "WARN deck-digest-unrecorded: measurements/CAL/env.txt carries no deck_sha256" >&2
  DECK_NOTE="$DECK_NOTE (WARN: calibration stamp absent)"
fi
expect_exit "perf-run.sh deck-missing (HOME=/nonexistent)" 3 "missing" env HOME=/nonexistent bash "$T/perf-run.sh" --dry-run CAL 4t

# 4. register header: Baseline pin, Milestones block, calibration bound
BACKLOG="$AUDIT/BACKLOG.md"
base_line="$(grep -m1 -E '^Baseline: [0-9a-f]{8,40} ' "$BACKLOG" || true)"
pinned=""
if [ -z "$base_line" ]; then
  echo "FAIL baseline-line-missing in $BACKLOG" >&2; FAIL=1
else
  pinned="$(awk '{print $2}' <<<"$base_line")"
  head_sha="$(git rev-parse HEAD)"
  case "$head_sha" in
    "$pinned"*) : ;;
    *) if git merge-base --is-ancestor "$pinned" HEAD \
          && git diff --quiet "$pinned" HEAD -- crates docs scripts .github schemas Cargo.toml Cargo.lock examples tests; then
         : # ledger-only commits above the pin: the evaluated surfaces are the pinned ones
       else
         echo "FAIL baseline-drift pinned=$pinned head=$head_sha" >&2; FAIL=1
       fi ;;
  esac
fi
grep -qE '^#+ .*Milestones' "$BACKLOG"                        || { echo "FAIL milestones-block-missing" >&2; FAIL=1; }
grep -qE '`0a` *< *`0b` *< *`1`' "$BACKLOG"                    || { echo "FAIL milestone-order 0a<0b<1 not declared" >&2; FAIL=1; }
grep -qF 'gnl-import' "$BACKLOG"                              || { echo "FAIL gnl-import DAG trigger missing" >&2; FAIL=1; }
grep -qE '^Protocol bound: [0-9]+(\.[0-9]+)? s' "$BACKLOG"    || { echo "FAIL calibration-bound-missing" >&2; FAIL=1; }
BOUND="$(cat "$AUDIT/measurements/CAL/median.txt" 2>/dev/null || echo unrecorded)"

# 5. the executable proof: the unittest module
ut_rc=0; ut_out="$(python3 -m unittest discover -s "$T/tests" -p 'test_*.py' 2>&1)" || ut_rc=$?
record "python3 -m unittest discover tools/tests" 0 "$ut_rc"
[ "$ut_rc" -eq 0 ] || { printf '%s\n' "$ut_out" >&2; FAIL=1; }
UT_SUMMARY="$(printf '%s\n' "$ut_out" | grep -E '^(Ran [0-9]+ tests|OK|FAILED)' | tr '\n' ' ')"

# 6. no tracked evaluation surface touched by the whole harness epic
dirty="$(git status --porcelain -- crates docs scripts .github schemas Cargo.toml)"
[ -z "$dirty" ] || { printf 'FAIL tracked-path-dirty:\n%s\n' "$dirty" >&2; FAIL=1; }

# 7. record the run, then exit on the aggregate verdict
STATION="$AUDIT/stations/harness-verification.md"
mkdir -p "$(dirname "$STATION")"
{
  printf '# Harness verification\n\n'
  printf '## Run %s\n\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf -- '- Baseline: `%s` (register pin `%s`)\n' "$(git rev-parse HEAD)" "$pinned"
  printf -- '- Deck: `%s` (%s)\n' "$DECK" "$DECK_NOTE"
  printf -- '- Calibration bound: `%s s` (from `measurements/CAL/median.txt`)\n' "$BOUND"
  printf -- '- Unit suite: %s\n' "${UT_SUMMARY:-not run}"
  printf -- '- Status: **%s**\n\n' "$([ "$FAIL" -eq 0 ] && echo PASS || echo FAIL)"
  printf '### Checks\n\n| check | expected exit | actual exit |\n| --- | --- | --- |\n'
  for r in "${RESULTS[@]}"; do
    IFS='|' read -r name want got <<<"$r"
    printf '| `%s` | %s | %s |\n' "$name" "$want" "$got"
  done
  printf '\n### Tracked-tree assertion\n\n`git status --porcelain -- crates docs scripts .github schemas Cargo.toml` -> %s\n' "$([ -z "$dirty" ] && echo empty || echo DIRTY)"
  printf '\n### Not exercised\n\n'
  printf -- '- `perf-run.sh` exit 4 (`timeout-3x`): needs a runaway solve; exercised ad hoc during the calibration ticket with `--bound 1`, not by this gate.\n'
  printf -- '- `perf-run.sh` exit 5 (`mpi-unavailable`): layout `2x2` is outside the harness epic; exercised ad hoc during the calibration ticket with `mpiexec` off PATH.\n'
  printf -- '- `perf-run.sh` exit 6 (deck mutated): would require corrupting the reference deck.\n'
} > "$STATION"
echo "verify-harness: $([ "$FAIL" -eq 0 ] && echo PASS || echo FAIL) -> $STATION"
exit "$FAIL"
