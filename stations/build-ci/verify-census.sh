#!/usr/bin/env bash
# verify-census.sh — build-ci station verification. Resolves the register heading, runs the three
# harness checkers plus the shared verify-station.sh, RE-MEASURES the whole gate census from the tree
# at the station baseline (nothing is copied from the inventory ticket: a copied figure is an
# unverified figure), runs the three scripts/ci oracles, re-asserts the premises the genericity entry
# rests on, and proves the read-only workspace. Writes only verification.md beside this script
# (scratch under $TMPDIR).
#
#   bash plans/architecture-debt-audit/stations/build-ci/verify-census.sh ["<section title>"] [--no-shared]
#
# argv[1] defaults to the ratified station heading; exactly one heading at any level must carry it
# (other spellings circulated during planning), else exit 2 listing the candidates rather than
# verifying the wrong section or an empty one. --no-shared skips verify-station.sh and the report
# write and prints the station block to stdout: the station test module invokes this script, and
# verify-station.sh runs that module, so the flag is what cuts the recursion.
#
# -e is deliberately omitted: this runs every check and aggregates one exit code.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
AUDIT="$ROOT/plans/architecture-debt-audit"
TOOLS="$AUDIT/tools"
BACKLOG="$AUDIT/BACKLOG.md"
REPORT="$HERE/verification.md"
REL="plans/architecture-debt-audit"
SLUG="build-ci"
DEFAULT_SECTION="STATION 7 — build/CI/scripts/schemas/docs (2026-09)"
MARKER="## Station-specific checks — build-ci"
FAILED=0
SECTION=""
RUN_SHARED=1
for arg in "$@"; do
  case "$arg" in
    --no-shared) RUN_SHARED=0 ;;
    --*) echo "usage: verify-census.sh [\"<section title>\"] [--no-shared]" >&2; exit 2 ;;
    *) SECTION="$arg" ;;
  esac
done
SECTION="${SECTION:-$DEFAULT_SECTION}"

SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/verify-census-build-ci.XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT

# One row per check: parallel arrays so a command containing `|` renders intact.
declare -a R_RES=() R_LABEL=() R_CMD=() R_CODE=()
record() { # record <PASS|FAIL> <label> <display-command> <exit-code>
  R_RES+=("$1"); R_LABEL+=("$2"); R_CMD+=("$3"); R_CODE+=("$4")
  [[ "$1" == FAIL ]] && FAILED=1
  return 0
}

# --- preflight: the heading must resolve exactly once (census_checks.py lists candidates, exit 2) --
CHECKS="$HERE/census_checks.py"
HEADING_LINE="$(python3 "$CHECKS" heading "$BACKLOG" "$SECTION")" || exit 2

# --- the three harness checkers over the resolved title (Alignment required as a fourth pass) ----
run_check() { # run_check <label> <display-command> <cmd...>
  local label="$1" display="$2" rc
  shift 2
  "$@" >"$SCRATCH/$label.log" 2>&1
  rc=$?
  if ((rc == 0)); then record PASS "$label" "$display" 0; else record FAIL "$label" "$display" "$rc"; fi
}
for c in check-anchors check-reraise fields-check; do
  run_check "$c" "python3 $REL/tools/$c.py '$SECTION'" python3 "$TOOLS/$c.py" "$SECTION"
done
run_check fields-check-alignment "python3 $REL/tools/fields-check.py --require Alignment '$SECTION'" \
  python3 "$TOOLS/fields-check.py" --require Alignment "$SECTION"

# --- the shared verifier, verbatim (slug + title); it writes the head of verification.md ---------
SHARED_RC=""
if ((RUN_SHARED)); then
  bash "$TOOLS/verify-station.sh" "$SLUG" "$SECTION"
  SHARED_RC=$?
  if ((SHARED_RC == 0)); then
    record PASS verify-station "bash $REL/tools/verify-station.sh $SLUG '$SECTION'" 0
  else
    record FAIL verify-station "bash $REL/tools/verify-station.sh $SLUG '$SECTION'" "$SHARED_RC"
  fi
fi

# --- re-measure the census from the tree at the station baseline ---------------------------------
# The surfaces are exported from git at inventory.json's `baseline`, never read from the worktree,
# so every figure below is the pin's; the read-only block proves the worktree equals it.
BASE="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["baseline"])' "$HERE/inventory.json")"
TREE="$SCRATCH/tree"
mkdir -p "$TREE"
git -C "$ROOT" archive "$BASE" .github scripts schemas docs/design CLAUDE.md Cargo.toml crates/cobre-io/schemas | tar -x -C "$TREE" \
  || { echo "verify-census.sh: cannot export the station surfaces at $BASE" >&2; exit 1; }
CI="$TREE/.github/workflows/ci.yml"

M_MPICH_CI="$(grep -c 'Build MPICH from source' "$CI")"
M_MPICH_TOTAL="$(cat "$TREE"/.github/workflows/*.yml | grep -c 'Build MPICH from source')"
mapfile -t GATE_FILES < <(cd "$TREE" && printf '%s\n' scripts/ci/*.sh scripts/ci/*.py scripts/ci/lib/*.sh)
M_GATEFILES="${#GATE_FILES[@]}"
SCHEMA_FILES=("$TREE"/schemas/*.json)
M_SCHEMAS="${#SCHEMA_FILES[@]}"
M_FBS=0; [[ -f "$TREE/crates/cobre-io/schemas/policy.fbs" ]] && M_FBS=1
M_ROOT_FBS=0; [[ -f "$TREE/schemas/policy.fbs" ]] && M_ROOT_FBS=1
M_BUILDRS="$(git -C "$ROOT" ls-tree -r --name-only "$BASE" | grep '/build\.rs$' | grep -vc '/vendor/')"
WORKFLOW_FILES=("$TREE"/.github/workflows/*.yml)
M_WORKFLOWS="${#WORKFLOW_FILES[@]}"
M_ACTIONS_DIR=0; [[ -d "$TREE/.github/actions" ]] && M_ACTIONS_DIR=1
# ci.yml job ids are the 2-space keys inside the `jobs:` block only. The raw census command also
# matches `push:` under `on:` (one hit more); the scoped count is the figure the section must quote.
M_JOBS_RAW="$(grep -cE '^  [a-z0-9-]+:$' "$CI")"
mapfile -t JOB_IDS < <(awk '/^jobs:$/{j=1;next} /^[a-z]/{j=0} j&&/^  [a-z0-9_-]+:$/{sub(/^  /,""); sub(/:$/,""); print}' "$CI")
M_JOBS="${#JOB_IDS[@]}"
mapfile -t UNWIRED < <(cd "$TREE" && for f in scripts/ci/*.sh scripts/ci/*.py scripts/ci/lib/*.sh; do
  n="$(basename "$f")"; grep -rq "$n" .github/workflows/ scripts/pre-commit || echo "$n"; done)
mapfile -t TRANSITIVE < <(cd "$TREE" && grep -n 'check-comment-bloat.sh' scripts/ci/*.sh | grep -v '^scripts/ci/check-comment-bloat.sh:' | sed 's/:[^:]*$//')

JOIN=","
join() { local IFS="$JOIN"; echo "$*"; }
python3 "$CHECKS" census "$HERE" "$BACKLOG" "$HEADING_LINE" \
  "mpichCi=$M_MPICH_CI" "mpichTotal=$M_MPICH_TOTAL" "schemas=$M_SCHEMAS" "fbs=$M_FBS" "rootFbs=$M_ROOT_FBS" \
  "buildRs=$M_BUILDRS" "workflows=$M_WORKFLOWS" "actionsDir=$M_ACTIONS_DIR" "jobsRaw=$M_JOBS_RAW" \
  "jobIds=$(join "${JOB_IDS[@]}")" "unwired=$(join "${UNWIRED[@]}")" "transitive=$(join "${TRANSITIVE[@]}")" \
  "gateFiles=$(join "${GATE_FILES[@]}")" >"$SCRATCH/census.out" 2>&1
CENSUS_RC=$?
if ((CENSUS_RC == 0)); then
  record PASS census "re-measured at ${BASE:0:8}: MPICH grep, gate glob, schemas/*.json + policy.fbs, build.rs, workflows, jobs: block awk, negative-grep loop, quality-report.sh invoker; gate-census.json completeness; inventory.json equality; section cross-check" 0
else
  record FAIL census "re-measured at ${BASE:0:8}: gate-census.json completeness; inventory.json equality; section cross-check" "$CENSUS_RC"
fi

# --- executable oracles: a red oracle invalidates the station's claims rather than becoming a finding
run_oracle() { # run_oracle <label> <cmd...>
  local label="$1" rc
  shift
  (cd "$ROOT" && "$@") >"$SCRATCH/oracle-$label.log" 2>&1
  rc=$?
  if ((rc == 0)); then record PASS "oracle-$label" "$*" 0; else record FAIL "oracle-$label" "$*" "$rc"; fi
}
run_oracle genericity bash scripts/ci/check-infra-genericity.sh
run_oracle doc-paths bash scripts/ci/check-doc-paths.sh
run_oracle doc-voice python3 scripts/ci/check_doc_voice.py

# --- premises behind the genericity entry, re-asserted at the baseline ---------------------------
G="$TREE/scripts/ci/check-infra-genericity.sh"
PREMISE_FAILS=()
grep -q '^EXCLUDED_FILES=()' "$G" || PREMISE_FAILS+=("EXCLUDED_FILES is no longer empty: the I.3-6 retirement evidence is stale")
SCAN_N="$(awk '/^SCAN_DIRS=\(/,/^\)/' "$G" | grep -c 'crates/cobre-')"
[[ "$SCAN_N" == 5 ]] || PREMISE_FAILS+=("SCAN_DIRS holds $SCAN_N crates; the entry claims exactly the five infra crates")
if grep -qE 'cobre-(model|network)' "$TREE/CLAUDE.md"; then
  PREMISE_FAILS+=("CLAUDE.md already names cobre-model/cobre-network; re-derive the two-site claim")
fi
python3 "$CHECKS" body "$BACKLOG" "$HEADING_LINE" "$HERE/calibration.json" >"$SCRATCH/premises.out" 2>&1
BODY_RC=$?
if ((${#PREMISE_FAILS[@]} == 0 && BODY_RC == 0)); then
  record PASS premises "EXCLUDED_FILES=() · SCAN_DIRS = 5 crates · CLAUDE.md names no cobre-model/cobre-network (at ${BASE:0:8}); section body: one Alignment per entry, two-site fix-shapes, reviewer rating on every downgrade" 0
else
  record FAIL premises "EXCLUDED_FILES=() · SCAN_DIRS · CLAUDE.md L1 names · section-body assertions" 1
  ((${#PREMISE_FAILS[@]})) && printf 'FAIL premise: %s\n' "${PREMISE_FAILS[@]}" >&2
fi

# --- read-only workspace -----------------------------------------------------------------------
# Compared against the pre-station porcelain snapshot when one exists (none was taken: the tree was
# clean when the station opened — the ticket's dirty .gitignore premise dates from a136840d), so only
# a NEW modification under a guarded prefix fails; the station's own untracked plans/ writes and the
# carried-in .gitignore are exempt.
SNAP="$AUDIT/tree-baseline.porcelain"
git -C "$ROOT" status --porcelain | sort >"$SCRATCH/porcelain.txt"
if [[ -f "$SNAP" ]]; then
  NEW="$(comm -13 <(sort "$SNAP") "$SCRATCH/porcelain.txt")"
else
  NEW="$(cat "$SCRATCH/porcelain.txt")"
fi
GUARDED="$(grep -E '^.. (\.github/|scripts/|schemas/|docs/|Cargo\.(toml|lock))' <<<"$NEW" | grep -v '^.. \.gitignore$' || true)"
DIFFSTAT="$(git -C "$ROOT" diff --stat HEAD -- .github scripts schemas docs Cargo.toml)"
if [[ -z "$GUARDED" && -z "$DIFFSTAT" ]]; then
  record PASS read-only "git status --porcelain minus the pre-station snapshot, guarded prefixes .github/ scripts/ schemas/ docs/ Cargo.toml; git diff --stat HEAD over the same" 0
else
  record FAIL read-only "git status --porcelain minus the pre-station snapshot, guarded prefixes .github/ scripts/ schemas/ docs/ Cargo.toml; git diff --stat HEAD over the same" 1
  [[ -z "$GUARDED" ]] || printf '  station-modified: %s\n' "$GUARDED" >&2
  [[ -z "$DIFFSTAT" ]] || printf '  diff --stat: %s\n' "$DIFFSTAT" >&2
fi
CARRIED="$(cut -c4- "$SNAP" 2>/dev/null | paste -sd' ')"

# --- render the station block --------------------------------------------------------------------
BLOCK="$SCRATCH/block.md"
{
  echo "$MARKER"
  echo
  echo "Station baseline \`${BASE}\` (inventory.json); section \`$SECTION\` resolved once at BACKLOG.md:$HEADING_LINE."
  echo
  echo "| Check | Command | Exit | Result |"
  echo "| --- | --- | --- | --- |"
  if ((RUN_SHARED == 0)); then
    echo "| verify-station | \`bash $REL/tools/verify-station.sh $SLUG '$SECTION'\` | - | SKIP (--no-shared: the station test module invokes this script from inside the shared verifier) |"
  fi
  for i in "${!R_LABEL[@]}"; do
    echo "| ${R_LABEL[$i]} | \`${R_CMD[$i]}\` | ${R_CODE[$i]} | ${R_RES[$i]} |"
  done
  echo
  echo "Re-measured at \`${BASE:0:8}\`: ${M_JOBS} ci.yml jobs (the raw \`grep -nE '^  [a-z0-9-]+:\$'\` census command matches ${M_JOBS_RAW}, the extra hit being the \`push:\` trigger under \`on:\`) · ${M_GATEFILES} gate files from \`scripts/ci/*.sh scripts/ci/*.py scripts/ci/lib/*.sh\` · ${#UNWIRED[@]} without direct wiring (${UNWIRED[*]}, invoked by ${TRANSITIVE[*]}) · ${M_MPICH_CI}x \`Build MPICH from source\` in ci.yml (${M_MPICH_TOTAL} across the workflows, no \`.github/actions/\`) · ${M_SCHEMAS} \`schemas/*.json\` + \`crates/cobre-io/schemas/policy.fbs\` (no root \`schemas/policy.fbs\`) · ${M_BUILDRS} build.rs · ${M_WORKFLOWS} workflows."
  echo
  echo "Census re-measurement: $(tail -1 "$SCRATCH/census.out")"
  echo
  echo "Premises: $(tail -1 "$SCRATCH/premises.out"); EXCLUDED_FILES=() present; SCAN_DIRS holds ${SCAN_N} crates; CLAUDE.md names neither cobre-model nor cobre-network."
  echo
  echo "Carried-in dirty tracked paths (not station-caused): \`${CARRIED:-none — no pre-station snapshot; the tree was clean when the station opened}\`; the \`.gitignore\` exemption the ticket describes stays in the rule but nothing exercises it at the pin."
  if ((FAILED)); then
    echo
    echo "Failures:"
    grep -h '^FAIL' "$SCRATCH"/census.out "$SCRATCH"/premises.out 2>/dev/null | sed 's/^/- /'
    ((${#PREMISE_FAILS[@]})) && printf -- '- premise: %s\n' "${PREMISE_FAILS[@]}"
  fi
} >"$BLOCK"

if ((RUN_SHARED)); then
  if [[ -f "$REPORT" ]]; then
    python3 - "$REPORT" "$BLOCK" "$MARKER" <<'PY'
import pathlib, sys
report, block, marker = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"), sys.argv[3]
head = report.read_text(encoding="utf-8").split(marker, 1)[0].rstrip() + "\n\n"
report.write_text(head + block, encoding="utf-8")
PY
  else
    cp "$BLOCK" "$REPORT"
  fi
else
  cat "$BLOCK"
fi

if ((FAILED == 0)); then
  echo "verify-census.sh build-ci: PASS (${#R_LABEL[@]} checks${SHARED_RC:+; shared verifier exit $SHARED_RC})"
else
  echo "verify-census.sh build-ci: FAIL" >&2
  for i in "${!R_LABEL[@]}"; do
    [[ "${R_RES[$i]}" == FAIL ]] && echo "  FAIL ${R_LABEL[$i]} (exit ${R_CODE[$i]})" >&2
  done
  for f in "$SCRATCH"/*.log "$SCRATCH"/*.out; do
    [[ -s "$f" ]] && grep -q 'FAIL\|Error\|error' "$f" && { echo "--- $(basename "$f")" >&2; tail -20 "$f" >&2; }
  done
fi
exit "$FAILED"
