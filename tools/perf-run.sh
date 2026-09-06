#!/usr/bin/env bash
# Perf Measurement Protocol runner: one warm-up plus three timed runs of the
# profiling-profile `cobre` binary over a scratch copy of the reference deck.
#
# usage: perf-run.sh [--dry-run] [--perf] [--bound <sec>] <ID> <4t|2x2|2t>
#   4t  : cobre_reduzido_2, --threads 4, pinned to P-cores 0,2,4,6
#   2x2 : cobre_reduzido_2, mpiexec -n 2 x --threads 2 (rank pinning by the sweep's wrapper)
#   2t  : cobre-mar-26-rv2-reduced (enumerated), --threads 2, pinned to 0,2
# exit: 0 measured · 2 dirty worktree · 3 deck or binary missing · 4 killed at
#       3x the bound (UNMEASURED timeout-3x) · 5 mpiexec absent for 2x2
#       (UNMEASURED mpi-unavailable) · 6 reference deck mutated · 64 usage
set -euo pipefail

AUDIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(git -C "$AUDIT_DIR" rev-parse --show-toplevel)"
DECK_SAMPLED="$HOME/git/cobre-bridge/example/cobre_reduzido_2"
DECK_ENUMERATED="$HOME/git/cobre-bridge/example/cobre-mar-26-rv2-reduced"
COBRE_BIN="$REPO_ROOT/target/profiling/cobre"
RANK_WRAPPER="$AUDIT_DIR/measurements/_wrap/rank-wrapper.sh"
SAFETY_CEILING_SEC=3600

usage() {
  echo "usage: perf-run.sh [--dry-run] [--perf] [--bound <sec>] <ID> <4t|2x2|2t>"
}

DRY_RUN=0; WITH_PERF=0; BOUND=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --perf)    WITH_PERF=1; shift ;;
    --bound)   BOUND="${2:?--bound needs a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*)       echo "perf-run.sh: unknown flag $1" >&2; usage >&2; exit 64 ;;
    *)         break ;;
  esac
done
[[ $# -eq 2 ]] || { usage >&2; exit 64; }
ID="$1"; LAYOUT="$2"
case "$LAYOUT" in
  4t)  DECK_SRC="$DECK_SAMPLED";    BOUND_LINE='Protocol bound:';               PCORES="0,2,4,6"; THREADS=4 ;;
  2x2) DECK_SRC="$DECK_SAMPLED";    BOUND_LINE='Protocol bound:';               PCORES="0,2 | 4,6 (per rank)"; THREADS=2 ;;
  2t)  DECK_SRC="$DECK_ENUMERATED"; BOUND_LINE='Protocol bound (enumerated):';  PCORES="0,2"; THREADS=2 ;;
  *) echo "perf-run.sh: layout must be 4t, 2x2 or 2t" >&2; exit 64 ;;
esac
OUT_DIR="$AUDIT_DIR/measurements/$ID"
BASELINE_SHA="$(grep -m1 -E '^Baseline: [0-9a-f]{40}' "$AUDIT_DIR/BACKLOG.md" | cut -d' ' -f2 || true)"

if [[ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no -- . ':!plans/architecture-debt-audit')" ]]; then
  echo "perf-run.sh: refusing to measure a dirty worktree (tracked modifications outside plans/architecture-debt-audit; exit 2)" >&2
  exit 2
fi
[[ -d "$DECK_SRC" ]] || { echo "perf-run.sh: deck for layout $LAYOUT missing at $DECK_SRC (exit 3)" >&2; exit 3; }
[[ -x "$COBRE_BIN" ]] || { echo "perf-run.sh: build target/profiling/cobre first (cargo build --profile profiling --features mpi --bin cobre; exit 3)" >&2; exit 3; }

if [[ "$LAYOUT" == "2x2" ]] && ! command -v mpiexec >/dev/null 2>&1; then
  if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$OUT_DIR"
    echo "UNMEASURED mpi-unavailable" > "$OUT_DIR/median.txt"
    printf 'skip_reason\tmpi-unavailable\nlayout\t%s\nbaseline_sha\t%s\n' "$LAYOUT" "$BASELINE_SHA" >> "$OUT_DIR/env.txt"
  fi
  echo "perf-run.sh: mpiexec absent, 2x2 skipped (exit 5)" >&2
  exit 5
fi

if [[ -z "$BOUND" ]]; then
  case "$ID" in
    CAL|CAL-ENUM) BOUND="$SAFETY_CEILING_SEC" ;;
    *) BOUND="$(grep -m1 "^$BOUND_LINE" "$AUDIT_DIR/BACKLOG.md" | sed 's/^[^:]*: *//' | grep -oE '^[0-9]+(\.[0-9]+)?' || true)"
       [[ -n "$BOUND" ]] || { echo "perf-run.sh: no numeric '$BOUND_LINE' in BACKLOG.md; run the calibration or pass --bound (exit 64)" >&2; exit 64; } ;;
  esac
fi
case "$ID" in
  CAL|CAL-ENUM) TIMEOUT_SEC="$SAFETY_CEILING_SEC" ;;
  *) TIMEOUT_SEC="$(awk -v b="$BOUND" 'BEGIN{t=3*b; printf "%d", (t==int(t)) ? t : int(t)+1}')" ;;
esac


build_cmd() {                          # $1 = case dir, $2 = run output dir
  case "$LAYOUT" in
    4t|2t) CMD=(taskset -c "$PCORES" "$COBRE_BIN" run "$1" --output "$2" --threads "$THREADS" --comm-backend local --quiet) ;;
    2x2)   if [[ -x "$RANK_WRAPPER" ]]; then
             CMD=(mpiexec -n 2 "$RANK_WRAPPER" "$COBRE_BIN" run "$1" --output "$2" --threads 2 --comm-backend mpi --quiet)
           else
             CMD=(mpiexec -n 2 "$COBRE_BIN" run "$1" --output "$2" --threads 2 --comm-backend mpi --quiet)
           fi ;;
  esac
}

if [[ $DRY_RUN -eq 1 ]]; then
  build_cmd "<scratch>/deck" "<scratch>/out"
  echo "deck:      $DECK_SRC"
  echo "binary:    $COBRE_BIN"
  echo "bound:     ${BOUND}s (timeout ${TIMEOUT_SEC}s)"
  echo "target:    $OUT_DIR"
  echo "command:   ${CMD[*]}"
  echo "dry run: nothing created"
  exit 0
fi

deck_checksum() {
  find "$1" -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
}

SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/perf-run-$ID-XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
SRC_SUM_BEFORE="$(deck_checksum "$DECK_SRC")"
mkdir -p "$SCRATCH/deck" "$SCRATCH/out"
cp -a "$DECK_SRC/." "$SCRATCH/deck/"
CASE_DIR="$SCRATCH/deck"; RUN_OUT="$SCRATCH/out"
build_cmd "$CASE_DIR" "$RUN_OUT"

mkdir -p "$OUT_DIR"
: > "$OUT_DIR/runs.tsv"
printf '%s\n' "${CMD[@]}" > "$OUT_DIR/cmd.txt"
{
  printf 'baseline_sha\t%s\n'   "$BASELINE_SHA"
  printf 'head_sha\t%s\n'       "$(git -C "$REPO_ROOT" rev-parse HEAD)"
  printf 'measured_at\t%s\n'    "$(date -Iseconds)"
  printf 'id\t%s\n'             "$ID"
  printf 'layout\t%s\n'         "$LAYOUT"
  printf 'bound_sec\t%s\n'      "$BOUND"
  printf 'timeout_sec\t%s\n'    "$TIMEOUT_SEC"
  printf 'cpu_model\t%s\n'      "$(awk -F': ' '/model name/{print $2; exit}' /proc/cpuinfo)"
  printf 'nproc\t%s\n'          "$(nproc)"
  printf 'taskset_mask\t%s\n'   "$PCORES"
  printf 'threads\t%s\n'        "$THREADS"
  printf 'binary\t%s\n'         "$COBRE_BIN"
  printf 'cargo_profile\tprofiling\n'
  printf 'cargo_features\tmpi (cobre-cli), highs (default backend)\n'
  printf 'binary_mtime\t%s\n'   "$(stat -c %y "$COBRE_BIN")"
  printf 'deck\t%s\n'           "$DECK_SRC"
  printf 'deck_sha256\t%s\n'    "$SRC_SUM_BEFORE"
  [[ "$LAYOUT" == "2x2" ]] && printf 'mpiexec\t%s\n' "$(command -v mpiexec)"
  true
} > "$OUT_DIR/env.txt"

assert_source_untouched() {
  local after; after="$(deck_checksum "$DECK_SRC")"
  if [[ "$after" != "$SRC_SUM_BEFORE" ]]; then
    echo "perf-run.sh: reference deck mutated during the run — measurement void (exit 6)" >&2
    echo "UNMEASURED deck-mutated" > "$OUT_DIR/median.txt"
    exit 6
  fi
}

RUN_INDEX=0
TIMED=()
time_one_run() {                       # $1 = role (warmup|timed)
  local role="$1" start end rc=0
  rm -rf "$RUN_OUT"; mkdir -p "$RUN_OUT"
  start=$(date +%s.%N)
  if [[ "$role" == "timed" && $WITH_PERF -eq 1 && ${#TIMED[@]} -eq 0 ]] && command -v perf >/dev/null 2>&1; then
    timeout --kill-after=10s "$TIMEOUT_SEC" perf record -F 99 -g --call-graph dwarf -o "$SCRATCH/perf.data" -- "${CMD[@]}" >/dev/null 2>"$OUT_DIR/run-$RUN_INDEX.log" || rc=$?
    perf report --stdio --no-children -i "$SCRATCH/perf.data" > "$OUT_DIR/perf.txt" 2>/dev/null || true
    printf 'perf_profile\tmeasurements/%s/perf.txt (run %s; sampled, excluded from nothing: its wall time is still a timed row)\n' "$ID" "$RUN_INDEX" >> "$OUT_DIR/env.txt"
  else
    timeout --kill-after=10s "$TIMEOUT_SEC" "${CMD[@]}" >/dev/null 2>"$OUT_DIR/run-$RUN_INDEX.log" || rc=$?
  fi
  end=$(date +%s.%N)
  local wall; wall=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.3f", b-a}')
  local status=ok
  [[ $rc -eq 124 || $rc -eq 137 ]] && status=killed
  [[ $rc -ne 0 && $status == ok ]] && status=failed
  printf '%s\t%s\t%s\t%s\t%s\n' "$RUN_INDEX" "$role" "$wall" "$rc" "$status" >> "$OUT_DIR/runs.tsv"
  if [[ $status == killed ]]; then
    echo "UNMEASURED timeout-3x" > "$OUT_DIR/median.txt"
    printf 'killed_at_run\t%s (timeout %ss)\n' "$RUN_INDEX" "$TIMEOUT_SEC" >> "$OUT_DIR/env.txt"
    assert_source_untouched
    echo "perf-run.sh: run $RUN_INDEX killed after ${TIMEOUT_SEC}s — UNMEASURED timeout-3x (exit 4)" >&2
    exit 4
  fi
  if [[ $status == failed ]]; then
    echo "UNMEASURED run-failed" > "$OUT_DIR/median.txt"
    assert_source_untouched
    echo "perf-run.sh: run $RUN_INDEX failed with exit $rc (see run-$RUN_INDEX.log); exit 1" >&2
    exit 1
  fi
  [[ $role == timed ]] && TIMED+=("$wall")
  RUN_INDEX=$((RUN_INDEX + 1))
}

time_one_run warmup
time_one_run timed
time_one_run timed
time_one_run timed
if [[ $WITH_PERF -eq 1 ]] && ! command -v perf >/dev/null 2>&1; then
  printf 'perf_skip\tperf-not-installed\n' >> "$OUT_DIR/env.txt"
fi

printf '%s\n' "${TIMED[@]}" | sort -g | sed -n '2p' > "$OUT_DIR/median.txt"
assert_source_untouched
echo "perf-run.sh: $ID ($LAYOUT) median $(cat "$OUT_DIR/median.txt") s over runs ${TIMED[*]} -> $OUT_DIR"
