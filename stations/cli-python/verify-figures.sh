#!/usr/bin/env bash
# verify-figures.sh — re-measure every figure the cli-python station quotes, run the shared
# verifier verbatim, and prove the heading-resolution, Wave-5, I.5, no-timing and read-only
# invariants at the station baseline. Read-only over the codebase: writes only figures.tsv and
# verification.md beside this script (scratch under $TMPDIR).
#
#   bash plans/architecture-debt-audit/stations/cli-python/verify-figures.sh ["<section title>"] [--no-shared]
#
# argv[1] defaults to the dated scaffold heading; the bare 'STATION 6' / 'Station 6' is refused as
# a selector (it names the station's inner H3, not the scaffold section, and matched nothing
# before this station wrote). --no-shared skips verify-station.sh and the report write and prints
# the station block to stdout: the station test module invokes this script, and verify-station.sh
# runs the test module, so the flag is what cuts the recursion.
#
# Adaptation (recorded as a deviation on the E06-6 ticket): the section carries no
# `(measured-by: …)` tags — no station has ever written them — so every figure is re-measured from
# the tree the station evaluated (inventory.json's `baseline`, exported from git, never the
# worktree) against the value the station's own artifacts recorded, and the cross-check then
# locates every figure the section text quotes with a figure-specific pattern and compares it to
# the re-measured value; a figure copied from the epic body, BACKLOG.md or Part-I I.5 fails.
#
# -e is deliberately omitted: this runs every check and aggregates a single exit code.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
AUDIT="$ROOT/plans/architecture-debt-audit"
TOOLS="$AUDIT/tools"
BACKLOG="$AUDIT/BACKLOG.md"
OUT="$HERE/figures.tsv"
REPORT="$HERE/verification.md"
REL="plans/architecture-debt-audit"
SLUG="cli-python"
DEFAULT_SECTION="★ QUALITY EVALUATION (2026-09, baseline a136840d) — cli-python"
STATION_HEADING="### ★ STATION 6 — cobre-cli / cobre-python / facade (2026-09, baseline)"
LEGACY_HEADING='## Station 5 — Outputs (`run/outputs.rs`, cobre-io writers, cobre-python parity)'
MARKER="## Station-specific checks — cli-python"
FAILED=0
SECTION=""
RUN_SHARED=1
for arg in "$@"; do
  case "$arg" in
    --no-shared) RUN_SHARED=0 ;;
    --*) echo "usage: verify-figures.sh [\"<section title>\"] [--no-shared]" >&2; exit 2 ;;
    *) SECTION="$arg" ;;
  esac
done
SECTION="${SECTION:-$DEFAULT_SECTION}"
case "$SECTION" in
  'STATION 6' | 'Station 6' | '★ STATION 6')
    echo "FAIL heading-resolution: '$SECTION' is the station's inner heading, not the section selector (it matched nothing in BACKLOG.md before this station wrote); pass the dated scaffold title" >&2
    exit 2 ;;
esac

SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/verify-figures-cli-python.XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT

# Expected figures come from the station's own artifacts (inventory.json, enforcement-measurements.json,
# verdicts.json, calibration.json); a re-measurement that differs is drift in one or the other.
eval "$(
  python3 - "$HERE" <<'PY'
import json, pathlib, shlex, sys
here = pathlib.Path(sys.argv[1])
inv = json.load(open(here / "inventory.json"))
meas = json.load(open(here / "enforcement-measurements.json"))
ver = json.load(open(here / "verdicts.json"))
cal = json.load(open(here / "calibration.json"))
roll = {s["id"]: s for s in inv["substationRollup"]}
wb, sl, ws, cv = inv["writerBoundary"], meas["sourceLayer"], meas["writerSurface"], meas["ciVisibility"]
ts = inv["testSurface"]
out = {
    "BASE": inv["baseline"],
    "E_SRC_FILES": inv["totals"]["files"], "E_SRC_LINES": inv["totals"]["lines"], "E_SRC_NONTEST": inv["totals"]["nonTestLines"],
    "E_S6A_FILES": roll["S6a"]["files"], "E_S6A_LINES": roll["S6a"]["lines"],
    "E_S6B_FILES": roll["S6b"]["files"], "E_S6B_LINES": roll["S6b"]["lines"],
    "E_S6C_FILES": roll["S6c"]["files"], "E_S6C_LINES": roll["S6c"]["lines"],
    "E_SDDP_REFS": inv["sddpCoupling"]["occurrences"], "E_SDDP_FILES": inv["sddpCoupling"]["filesWithOccurrences"], "E_SDDP_USE": inv["sddpCoupling"]["useLines"],
    "E_CLI_CALLS": ws["cliCallSites"], "E_CLI_WRITE_LINE": ws["cliTerminalWriteLine"], "E_PY_CALLS": ws["pythonCallSites"], "E_PY_IF_ANY": ws["pythonIfAnyHelpers"],
    "E_CLI_EXT_NAMES": wb["cli"]["externalCount"], "E_PY_EXT_NAMES": wb["python"]["externalCount"],
    "E_PARITY_SHARED": sl["sharedCount"], "E_PARITY_EXIT": sl["exitCode"], "E_PARITY_FLOOR": sl["minSharedFloor"],
    "E_FROM_CONFIG": next(s["baseline"] for s in inv["supersessions"] if "from_config" in s["fact"]),
    "E_PY_DISTINCT_WRITES": next(s["baseline"] for s in inv["supersessions"] if "distinct" in s["fact"]),
    "E_CLI_BINS": ts["cobreCli"]["binaryCount"], "E_CLI_BIN_FNS": ts["cobreCli"]["binaryTestFns"], "E_CLI_INLINE_FNS": ts["cobreCli"]["inlineTestFns"],
    "E_PY_FILES": ts["cobrePython"]["pytestFileCount"], "E_PY_FNS": ts["cobrePython"]["testFunctions"], "E_PY_LINES": ts["cobrePython"]["lines"], "E_PY_RUST_TESTS": ts["cobrePython"]["rustTestFns"],
    "E_CI_TEST_LINES": len(ts["ciVisibility"]["rustTestsStep"]["commandLines"]),
    "E_ANCHORS_CHECKED": 251, "E_ANCHORS_FAILING": 0,
    "E_RECEIVED": ver["counts"]["received"], "E_DEFENDED": ver["counts"]["defended"], "E_CONFIRMED": ver["counts"]["confirmed"],
    "E_DISMISSED": ver["counts"]["dismissed"], "E_DUPOF": ver["counts"]["dupOf"], "E_MINTED": cal["counts"]["minted"],
    "CLI_LOCAL_NAMES": "|".join(wb["localNames"]["cli"]), "PY_LOCAL_NAMES": "|".join(wb["localNames"]["python"]),
    "S6A_MEMBERS": " ".join(roll["S6a"]["members"]), "S6B_MEMBERS": " ".join(roll["S6b"]["members"]), "S6C_MEMBERS": " ".join(roll["S6c"]["members"]),
}
for k, v in out.items():
    print(f"{k}={shlex.quote(str(v))}")
PY
)"

# The three crates, the CI workflows and the parity script as they stood at the station baseline,
# exported from git so every grep/wc below measures that tree; commands are recorded with the
# scratch path spelled as <tree@sha> so figures.tsv is byte-stable across runs.
TREE="$SCRATCH/tree"
mkdir -p "$TREE"
git -C "$ROOT" archive "$BASE" crates/cobre-cli crates/cobre-python crates/cobre scripts/ci .github/workflows | tar -x -C "$TREE" \
  || { echo "verify-figures.sh: cannot export the station surfaces at $BASE" >&2; exit 1; }
TOKEN="<tree@${BASE:0:8}>"
CLI="$TREE/crates/cobre-cli/src"
RUN="$CLI/commands/run"
PYS="$TREE/crates/cobre-python/src"
members() { for m in "$@"; do printf '%s ' "$TREE/$m"; done; }

printf 'figure\texpected\tmeasured\tstatus\tcommand\n' >"$OUT"
fig() { # fig <name> <expected> <command>
  local name="$1" want="$2" cmd="$3" got status
  got="$(eval "$cmd" 2>/dev/null | tr -d ' \n')"
  if [ "$got" = "$want" ]; then status=OK; else status=DRIFT; FAILED=1; fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$want" "$got" "$status" "${cmd//$TREE/$TOKEN}" >>"$OUT"
}

# 1. census and partition (the section's Baseline paragraph)
fig src_files "$E_SRC_FILES" "find $CLI $PYS $TREE/crates/cobre/src -name '*.rs' | wc -l"
fig src_lines "$E_SRC_LINES" "find $CLI $PYS $TREE/crates/cobre/src -name '*.rs' -print0 | xargs -0 cat | wc -l"
fig src_nontest_lines "$E_SRC_NONTEST" "python3 -c 'import sys; sys.path.insert(0, \"$TOOLS\"); from lib import station_checks as sc; t = sc.Tree(\"$BASE\"); print(sum(sc.classify_lines(f, t.read_text(f))[0] for r in (\"crates/cobre-cli/src\", \"crates/cobre-python/src\", \"crates/cobre/src\") for f in t.rs_files(r)))'"
fig s6a_files "$E_S6A_FILES" "ls $(members $S6A_MEMBERS) | wc -l"
fig s6a_lines "$E_S6A_LINES" "cat $(members $S6A_MEMBERS) | wc -l"
fig s6b_files "$E_S6B_FILES" "ls $(members $S6B_MEMBERS) | wc -l"
fig s6b_lines "$E_S6B_LINES" "cat $(members $S6B_MEMBERS) | wc -l"
fig s6c_files "$E_S6C_FILES" "ls $(members $S6C_MEMBERS) | wc -l"
fig s6c_lines "$E_S6C_LINES" "cat $(members $S6C_MEMBERS) | wc -l"
# 2. Part-I I.5 coupling triple (supersession note SN-01)
fig sddp_refs "$E_SDDP_REFS" "grep -rho 'cobre_sddp::' $CLI | wc -l"
fig sddp_ref_files "$E_SDDP_FILES" "grep -rl 'cobre_sddp::' $CLI | wc -l"
fig sddp_use_lines "$E_SDDP_USE" "grep -rh '^use cobre_sddp' $CLI | wc -l"
# 3. the writer double-mirror (SN-02) and the parity gate's name sets (SN-03)
fig cli_writer_calls "$E_CLI_CALLS" "grep -rho 'write_[a-z_0-9]*(' $RUN | wc -l"
fig cli_write_line_calls "$E_CLI_WRITE_LINE" "grep -rho 'write_[a-z_0-9]*(' $RUN | grep -c 'write_line('"
fig py_writer_calls "$E_PY_CALLS" "grep -o 'write_[a-z_0-9]*(' $PYS/run.rs | wc -l"
fig py_if_any_helpers "$E_PY_IF_ANY" "grep -o 'write_[a-z_0-9]*(' $PYS/run.rs | grep -c '_if_any('"
fig py_distinct_write_names "$E_PY_DISTINCT_WRITES" "grep -o 'write_[a-z_0-9]*(' $PYS/run.rs | sort -u | wc -l"
fig cli_external_writer_names "$E_CLI_EXT_NAMES" "grep -rho 'write_[a-z_0-9]*(' $RUN | tr -d '(' | sort -u | grep -Evx '$CLI_LOCAL_NAMES' | wc -l"
fig py_external_writer_names "$E_PY_EXT_NAMES" "grep -o 'write_[a-z_0-9]*(' $PYS/run.rs | tr -d '(' | sort -u | grep -Evx '$PY_LOCAL_NAMES' | wc -l"
fig parity_shared_names "$E_PARITY_SHARED" "python3 $TREE/scripts/ci/check_python_parity.py --max 0 --root $TREE | grep -oE '[0-9]+ write functions' | cut -d' ' -f1"
fig parity_exit "$E_PARITY_EXIT" "python3 $TREE/scripts/ci/check_python_parity.py --max 0 --root $TREE >/dev/null 2>&1; echo \$?"
fig parity_floor "$E_PARITY_FLOOR" "awk '/\"--min-shared\"/{f=1} f && /default=/{match(\$0, /[0-9]+/); print substr(\$0, RSTART, RLENGTH); exit}' $TREE/scripts/ci/check_python_parity.py"
# 4. the from_config non-test call sites (inventory supersession) — spans every crate at the pin, so it
# reads the blobs through git rather than the three-crate export; the cfg(test) tail of each file is cut
fig from_config_nontest_sites "$E_FROM_CONFIG" "git -C $ROOT grep -l -F 'StudyParams::from_config(' $BASE -- 'crates/*/src/*.rs' | grep -v 'tests\\.rs' | while read -r spec; do git -C $ROOT show \"\$spec\" | awk '/^#\\[cfg\\(test\\)\\]/{exit} /StudyParams::from_config\\(/ && \$0 !~ /^[[:space:]]*\\/\\// {n++} END{print n+0}'; done | awk '{s+=\$1} END{print s+0}'"
# 5. the test corpus (td-queue.json corpusFigures)
fig cli_test_binaries "$E_CLI_BINS" "ls $TREE/crates/cobre-cli/tests/*.rs | wc -l"
fig cli_binary_test_fns "$E_CLI_BIN_FNS" "grep -rh '#\\[test\\]' $TREE/crates/cobre-cli/tests | wc -l"
fig cli_inline_test_fns "$E_CLI_INLINE_FNS" "grep -rh '#\\[test\\]' $CLI | wc -l"
fig py_pytest_files "$E_PY_FILES" "ls $TREE/crates/cobre-python/tests/*.py | wc -l"
fig py_pytest_test_fns "$E_PY_FNS" "grep -rhE '^[[:space:]]*def test_' $TREE/crates/cobre-python/tests/*.py | wc -l"
fig py_pytest_lines "$E_PY_LINES" "cat $TREE/crates/cobre-python/tests/*.py | wc -l"
fig py_rust_tests "$E_PY_RUST_TESTS" "grep -rh '#\\[test\\]' $PYS/*.rs | wc -l"
# 6. the facade
fig facade_lib_lines 2 "wc -l < $TREE/crates/cobre/src/lib.rs"
fig facade_dependencies_sections 0 "grep -c '^\\[dependencies\\]' $TREE/crates/cobre/Cargo.toml"
# 7. the register-side figures the Baseline paragraph quotes
fig anchor_probe_checked "$E_ANCHORS_CHECKED" "python3 $TOOLS/check-anchors.py 'INGEST ANCHOR PROBE — cli-python (2026-09, baseline)' --register $HERE/anchor-probe.md --baseline $BASE --json | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"checked\"])'"
fig anchor_probe_failing "$E_ANCHORS_FAILING" "python3 $TOOLS/check-anchors.py 'INGEST ANCHOR PROBE — cli-python (2026-09, baseline)' --register $HERE/anchor-probe.md --baseline $BASE --json | python3 -c 'import json,sys; print(len(json.load(sys.stdin)[\"failures\"]))'"
fig candidates_received "$E_RECEIVED" "python3 -c 'import json; print(len(json.load(open(\"$HERE/verdicts.json\"))[\"verdicts\"]))'"
fig candidates_defended "$E_DEFENDED" "python3 -c 'import json; print(sum(1 for e in json.load(open(\"$HERE/verdicts.json\"))[\"verdicts\"].values() if e[\"disposition\"] == \"defended\"))'"
fig candidates_confirmed "$E_CONFIRMED" "python3 -c 'import json; print(sum(1 for e in json.load(open(\"$HERE/verdicts.json\"))[\"verdicts\"].values() if e[\"verdict\"] == \"confirmed\"))'"
fig candidates_dismissed "$E_DISMISSED" "python3 -c 'import json; print(sum(1 for e in json.load(open(\"$HERE/verdicts.json\"))[\"verdicts\"].values() if e[\"verdict\"] == \"dismissed\"))'"
fig candidates_dup_of "$E_DUPOF" "python3 -c 'import json; print(sum(1 for e in json.load(open(\"$HERE/verdicts.json\"))[\"verdicts\"].values() if e[\"disposition\"] == \"dup-of\"))'"
fig minted_ids "$E_MINTED" "python3 -c 'import json; print(len(json.load(open(\"$HERE/calibration.json\"))[\"assigned\"]))'"

# 8. build-and-CI-visibility block. cobre-python type-checks from its own manifest WITHOUT a libpython
# RUSTFLAGS override (cargo check never links). The CI-visibility premise the ticket carried ('no
# workflow runs the crate's Rust tests') is SUPERSEDED at the pin: ci.yml's python job builds with
# maturin, runs pytest AND runs `cargo test --manifest-path crates/cobre-python/Cargo.toml` (:562-568),
# so the block asserts the measured state — one workflow, one cargo-test step — and fails if that
# step disappears (the invisibility finding would then be live again and the Cleared cross-reference
# in the section would be wrong).
CARGO_LOG="$SCRATCH/cargo-check.log"
if (cd "$ROOT" && cargo check --manifest-path crates/cobre-python/Cargo.toml) >"$CARGO_LOG" 2>&1; then
  CARGO_RC=0
else
  CARGO_RC=$?
  echo "FAIL py_cargo_check: cobre-python does not type-check from its own manifest (exit $CARGO_RC)" >&2
  tail -5 "$CARGO_LOG" >&2
fi
printf 'py_cargo_check\t0\t%s\t%s\t%s\n' "$CARGO_RC" "$([ "$CARGO_RC" = 0 ] && echo OK || echo DRIFT)" "cargo check --manifest-path crates/cobre-python/Cargo.toml (no RUSTFLAGS)" >>"$OUT"
[ "$CARGO_RC" = 0 ] || FAILED=1
fig ci_cargo_test_steps "$E_CI_TEST_LINES" "grep -c 'cargo test --manifest-path crates/cobre-python/Cargo.toml' $TREE/.github/workflows/ci.yml"
fig workflows_running_py_rust_tests 1 "grep -lE '(cargo (test|nextest)|nextest run).*cobre-python' $TREE/.github/workflows/*.yml | wc -l"
fig ci_python_job_maturin 1 "grep -q 'maturin develop' $TREE/.github/workflows/ci.yml && echo 1 || echo 0"
fig ci_python_job_pytest 1 "grep -q 'pytest crates/cobre-python/tests' $TREE/.github/workflows/ci.yml && echo 1 || echo 0"
fig ci_python_job_requires_cli_binary 1 "grep -q -- '--require-cli-binary' $TREE/.github/workflows/ci.yml && echo 1 || echo 0"

column -t -s "$(printf '\t')" "$OUT" >&2

# 9. the shared verifier, verbatim (slug + dated title), unless --no-shared
SHARED_RC=""
if [ "$RUN_SHARED" = 1 ]; then
  bash "$TOOLS/verify-station.sh" "$SLUG" "$SECTION"
  SHARED_RC=$?
  [ "$SHARED_RC" = 0 ] || FAILED=1
fi

# 10. heading resolution, cross-check, Wave-5, I.5, no-timing, read-only — one parser-backed block
# that also renders the station rows; it prints the block to stdout and exits non-zero on failure.
BLOCK="$SCRATCH/block.md"
python3 - "$ROOT" "$HERE" "$TOOLS" "$BACKLOG" "$SECTION" "$STATION_HEADING" "$LEGACY_HEADING" "$OUT" "$BASE" "${SHARED_RC:-skipped}" "$MARKER" >"$BLOCK" <<'PY' || FAILED=1
import json, pathlib, re, subprocess, sys, tempfile
from collections import Counter

root, here, tools, backlog, title, station_heading, legacy_heading, tsv, base, shared_rc, marker = sys.argv[1:12]
root, here = pathlib.Path(root), pathlib.Path(here)
sys.path.insert(0, tools)
from lib import backlog_parse as bp  # noqa: E402

REL = "plans/architecture-debt-audit"
rows: list[tuple[str, str, int | None, str]] = []
failures: list[str] = []


def run(cmd):
    return subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)


def check(label, command, problems, detail):
    failures.extend(problems)
    rows.append((label, command, 1 if problems else 0, detail))


lines = bp.read_register(pathlib.Path(backlog))
register = "\n".join(lines)

# --- heading resolution ------------------------------------------------------------------
want = title.strip().casefold()
hits = []
for n, raw in enumerate(lines, 1):
    m = bp.SECTION_RE.match(raw)
    if m:
        t = m.group("title")
        if want in (t.strip().casefold(), t.rsplit("—", 1)[-1].strip().casefold()):
            hits.append((n, t))
legacy = register.count(legacy_heading)
inner = register.count(station_heading)
probs = []
if len(hits) != 1:
    probs.append(f"section title {title!r} matches {len(hits)} headings (expected exactly 1): " + "; ".join(f"L{n}" for n, _ in hits))
if inner != 1:
    probs.append(f"the inner station heading occurs {inner} times (expected exactly 1)")
if legacy != 1:
    probs.append(f"the 2026-08 legacy heading occurs {legacy} times (expected exactly 1, deliberately unmatched)")
section = bp.find_section(lines, title) if len(hits) == 1 else None
entries = bp.iter_entries(section) if section else []
if section and not entries:
    probs.append("the section parses to zero entries — an empty section does not pass vacuously")
check("heading-resolution", f"dated scaffold title matched exactly once (find_section rule); inner `{station_heading[4:]}` once; legacy `{legacy_heading[3:60]}…` reported unmatched; bare 'STATION 6' refused as a selector (exit 2)", probs,
      f"{len(hits)} match for the dated title (BACKLOG.md L{hits[0][0] if hits else '?'}); inner station heading ×{inner}; legacy 2026-08 heading ×{legacy} deliberately unmatched; {len(entries)} entries parsed")
text = "\n".join(section.lines) if section else ""

# --- figures.tsv --------------------------------------------------------------------------
figs = {}
for line in pathlib.Path(tsv).read_text(encoding="utf-8").splitlines()[1:]:
    name, expected, measured, status, command = line.split("\t")
    figs[name] = {"expected": expected, "measured": measured, "status": status, "command": command}
probs = [f"figure {n}: recorded {f['expected']} vs re-measured {f['measured']} ({f['command'][:90]})" for n, f in figs.items() if f["status"] != "OK"]
check("figures", f"`bash {REL}/stations/cli-python/verify-figures.sh` → figures.tsv ({len(figs)} figures re-measured at {base[:8]} against inventory.json / enforcement-measurements.json / verdicts.json / calibration.json)", probs,
      f"{len(figs)} figures, {len(probs)} drift; the ticket's 17-figure list is covered and extended (census, partition, coupling triple, double-mirror, parity gate, from_config, test corpus, facade, register-side counts, build + CI evidence)")

# --- cross-check: every figure the section quotes equals the re-measured value --------------
QUOTED = [
    ("Baseline paragraph census", r"(\d+) source files / ([\d,]+) lines \(([\d,]+) non-test\)", ["src_files", "src_lines", "src_nontest_lines"]),
    ("Baseline paragraph S6a", r"S6a (\d+) files / ([\d,]+) lines", ["s6a_files", "s6a_lines"]),
    ("Baseline paragraph S6b", r"S6b (\d+) files / ([\d,]+) lines", ["s6b_files", "s6b_lines"]),
    ("Baseline paragraph S6c", r"S6c (\d+) files / ([\d,]+) lines", ["s6c_files", "s6c_lines"]),
    ("Baseline paragraph candidates", r"(\d+) attacker candidates: (\d+) defended \((\d+) confirmed, (\d+) dismissed, \d+ unresolved\), (\d+) intra-station dup-of", ["candidates_received", "candidates_defended", "candidates_confirmed", "candidates_dismissed", "candidates_dup_of"]),
    ("Baseline paragraph anchors", r"every anchor resolves at the baseline \((\d+) checked / (\d+) failing\)", ["anchor_probe_checked", "anchor_probe_failing"]),
    ("Baseline paragraph minted", r"and (\d+) mint ids", ["minted_ids"]),
    ("Baseline paragraph writer names", r"Writer boundary at the pin: (\d+) / (\d+) external writer names", ["cli_external_writer_names", "py_external_writer_names"]),
    ("Baseline paragraph parity", r"parity gate (\d+) shared names \(floor (\d+)\)", ["parity_shared_names", "parity_floor"]),
    ("Baseline paragraph Rust tests", r"(\d+) cobre-python Rust tests CI-visible", ["py_rust_tests"]),
    ("SN-01 coupling triple", r"→ (\d+) `cobre_sddp` occurrences in (\d+) files, (\d+) of them `use` lines", ["sddp_refs", "sddp_ref_files", "sddp_use_lines"]),
    ("SN-02 writer surface", r"→ (\d+) CLI call sites \((\d+) terminal `write_line`, \d+ writer sites\) / (\d+) Python \((\d+) `_if_any` helpers\); external writer names (\d+) / (\d+)", ["cli_writer_calls", "cli_write_line_calls", "py_writer_calls", "py_if_any_helpers", "cli_external_writer_names", "py_external_writer_names"]),
    ("SN-03 parity name sets", r"import-resolving since fc81427a: (\d+) CLI / (\d+) Python names, (\d+) in both, floor --min-shared (\d+)", ["parity_shared_names", "parity_shared_names", "parity_shared_names", "parity_floor"]),
    ("SN-06 CI-visible tests", r"→ (\d+) tests \(", ["py_rust_tests"]),
    ("CD-025 disposition prose (source layer)", r"sees (\d+) CLI / (\d+) Python names \((\d+) shared, floor (\d+);", ["parity_shared_names", "parity_shared_names", "parity_shared_names", "parity_floor"]),
    ("CD-025 disposition prose (writer surface)", r"the writer surface is (\d+) CLI / (\d+) Python `write_\*\(` call sites \((\d+) terminal write_line", ["cli_writer_calls", "py_writer_calls", "cli_write_line_calls"]),
]
probs = []
quoted = 0
for label, pattern, names in QUOTED:
    ms = list(re.finditer(pattern, text))
    if not ms:
        probs.append(f"{label}: the section no longer quotes this figure ({pattern[:60]}…)")
        continue
    for m in ms:
        for value, name in zip(m.groups(), names):
            quoted += 1
            got = value.replace(",", "")
            if got != figs[name]["measured"]:
                probs.append(f"{label}: quoted {got} for {name}, re-measured {figs[name]['measured']} (`{figs[name]['command'][:80]}`)")
if "two doc lines" not in text or figs["facade_lib_lines"]["measured"] != "2":
    probs.append(f"facade: the section says 'two doc lines', the facade measures {figs['facade_lib_lines']['measured']} lines")
if re.search(r"\(measured-by:", text):
    probs.append("the section carries measured-by tags the cross-check does not know")
check("cross-check", f"{len(QUOTED)} figure-specific patterns over the parsed section (every quoted `<n>` located and compared to figures.tsv; the section carries no `(measured-by: …)` tags)", probs,
      f"{quoted} quoted values located across {len(QUOTED)} passages, {len(probs)} mismatches; a figure copied from the epic body, BACKLOG.md or Part-I I.5 would fail here")

# --- supersession: prior figures recorded as old → new → command, never silently replaced --------
cal = json.load(open(here / "calibration.json"))
notes = cal["supersessionNotes"]
probs = []
notes_md = (here / "supersession-notes.md").read_text(encoding="utf-8")
for n in notes:
    if not (n["old"] and n["new"] and n["command"]):
        probs.append(f"{n['id']}: incomplete (old/new/command)")
    if f"## {n['id']} — " not in notes_md:
        probs.append(f"{n['id']}: missing from supersession-notes.md")
    if f"**{n['id']} — " not in text:
        probs.append(f"{n['id']}: not listed in the section's Supersession notes block")
for rid in ("CD-025", "CD-029", "CD-002", "CD-009"):
    r = next(x for x in cal["reusedIds"] if x["id"] == rid)
    if r["disposition"] == "sharpen" and not r["supersessionNotes"]:
        probs.append(f"{rid}: sharpen without a linked supersession note (a bare replacement)")
required = {"SN-01": "cobre_sddp", "SN-02": "call sites", "SN-03": "18", "SN-06": "19"}
for sid, needle in required.items():
    n = next(x for x in notes if x["id"] == sid)
    if needle not in n["new"]:
        probs.append(f"{sid}: new figure does not carry {needle!r}")
fc = next(x for x in notes if "from_config" in x["fact"] or "from_config" in x["new"]) if any("from_config" in (x["fact"] + x["new"]) for x in notes) else None
check("supersession", "calibration.json supersessionNotes (old figure + source → new figure → command) ↔ supersession-notes.md `## SN-nn` blocks ↔ the section's Supersession notes block; every sharpen row links ≥1 note",
      probs, f"{len(notes)} notes (SN-01 I.5 coupling, SN-02 writer call sites 41/29 → {figs['cli_writer_calls']['measured']}/{figs['py_writer_calls']['measured']}, SN-03 parity 4-of-17 → {figs['parity_shared_names']['measured']}/18, SN-04 CD-025 anchors, SN-05 CD-029 parity half, SN-06 CI visibility 22 → {figs['py_rust_tests']['measured']}, SN-07 runtime layer, SN-08 CD-002, SN-09 CD-009, SN-10 report/summary); the from_config figure (1 → {figs['from_config_nontest_sites']['measured']} non-test sites) is inventory.json → supersessions with its command")

# --- Wave-5 dispositions ---------------------------------------------------------------------
waves = json.load(open(here / "wave-dispositions.json"))["dispositions"]
by_id = {d["id"]: d for d in waves}
probs = []
expected_ids = {"CD-025", "CD-029", "CD-002", "CD-009"}
if set(by_id) != expected_ids:
    probs.append(f"owned ids {sorted(by_id)} != {sorted(expected_ids)}")
stub = ["## DISPOSITION ANCHOR PROBE — cli-python (verify)", ""]
for d in waves:
    a = d["baselineAnchor"]
    stub += [f"**{d['id']} · probe**", "", d["id"], "", f"- **Anchors:** `{a['path']}::{a['symbol']}`", f"- **Baseline:** `{base}`", ""]
    if d["disposition"] not in {"keep", "retire", "sharpen"}:
        probs.append(f"{d['id']}: disposition {d['disposition']!r}")
    if d["disposition"] == "sharpen" and not (d.get("supersededClaim") and d.get("survivingClaim")):
        probs.append(f"{d['id']}: sharpen without both claims")
    if d["disposition"] == "retire" and not d.get("resolvingCommit"):
        probs.append(f"{d['id']}: retire without a resolving commit")
    for v in d.get("conflicts") or []:
        if v.get("tag") != "conflicts" or v.get("presentedAsDirection"):
            probs.append(f"{d['id']}: variant {v.get('variant')!r} places orchestration outside cobre-io without the conflicts tag")
        if not re.search(r"Part (IV|III|V)", v.get("rule", "")):
            probs.append(f"{d['id']}: variant {v.get('variant')!r} cites no Part IV/V rule")
for rid in ("CD-025", "CD-029"):
    d = by_id.get(rid, {})
    if (d.get("alignmentHint") or {}).get("value") != "advances-0a":
        probs.append(f"{rid}: alignment is not advances-0a")
    if not re.search(r"Part (IV|V)", (d.get("alignmentHint") or {}).get("citation", "")):
        probs.append(f"{rid}: alignment citation names no Part IV/V section")
cd025 = by_id.get("CD-025", {})
if "cobre-io" not in (cd025.get("restatedFixShape") or ""):
    probs.append("CD-025: restated fix-shape does not name cobre-io as owner")
if not any("cobre-sddp" in v.get("variant", "") for v in cd025.get("conflicts") or []) or not any("cli-local" in v.get("variant", "") or "cobre-cli" in v.get("variant", "") for v in cd025.get("conflicts") or []):
    probs.append("CD-025: the cobre-sddp and cli-local variants are not both recorded as conflicts")
cd029 = by_id.get("CD-029", {})
if "PrepPhase" in (cd029.get("restatedFixShape") or "") and not any("destination" in json.dumps(q).lower() or "cobre-io" in json.dumps(q) for q in cd029.get("needsHuman") or []):
    probs.append("CD-029: the L3 PrepPhase destination is not held as an owner question")
with tempfile.TemporaryDirectory(prefix="cli-verify.") as tmp:
    p = pathlib.Path(tmp) / "probe.md"
    p.write_text("\n".join(stub) + "\n", encoding="utf-8")
    pr = run([sys.executable, str(pathlib.Path(tools) / "check-anchors.py"), "DISPOSITION ANCHOR PROBE — cli-python (verify)", "--register", str(p), "--baseline", base, "--json"])
    try:
        rep = json.loads(pr.stdout)
    except json.JSONDecodeError:
        rep = {"checked": 0, "failures": [{"anchor": "?", "kind": pr.stderr.strip()[:120]}]}
    probs += [f"anchor-missing[{f['kind']}] {f['anchor']}" for f in rep.get("failures", [])]
table = next((t for t in bp.parse_tables(section) if t and "ID" in t[0] and "Disposition" in t[0]), None) if section else None
if table is None:
    probs.append("no disposition table with ID and Disposition columns in the section")
else:
    trs = {}
    for r in table:
        trs.setdefault(r["ID"], []).append(r)
    for d in waves:
        got = trs.get(d["id"], [])
        if len(got) != 1 or got[0]["Disposition"] != d["disposition"]:
            probs.append(f"{d['id']}: disposition table row mismatch ({len(got)} rows)")
    if "SN-" not in "".join(v for r in trs.get("CD-025", [{}]) for v in r.values()):
        probs.append("CD-025: table row carries no supersession note id")
disp_block = text.split("#### Wave-5 dispositions", 1)[1].split("\n#### ", 1)[0] if "#### Wave-5 dispositions" in text else ""
if "SUPERSEDED" not in disp_block or "cobre-sddp or cobre-io" not in disp_block:
    probs.append("CD-025's 'cobre-sddp or cobre-io' wording is not marked SUPERSEDED in the disposition block")
for layer in ("check_python_parity.py", "test_cli_python_file_set_parity.py", "python_parity_check.rs"):
    if layer not in disp_block:
        probs.append(f"the parity-gap claim under CD-025 does not name enforcement layer {layer}")
check("wave-5-dispositions", f"wave-dispositions.json (4 owned ids) through `python3 {REL}/tools/check-anchors.py 'DISPOSITION ANCHOR PROBE — cli-python (verify)' --register <stub> --baseline {base[:8]} --json`; alignment, fix-shape owner, conflict variants and the disposition table + SUPERSEDED marker over the parsed section",
      probs, f"{len(waves)} rows ({Counter(d['disposition'] for d in waves)['sharpen']} sharpen); {rep.get('checked', 0)} baseline anchors resolved; CD-025 fix-shape owner cobre-io with both cobre-sddp and cli-local variants tagged conflicts (never a direction); CD-029's L3 PrepPhase destination held as an owner question; the three parity enforcement layers named under CD-025")

# --- I.5 handoff -----------------------------------------------------------------------------
handoff = json.load(open(here / "partI-handoff.json"))
probs = []
for r in handoff["i5Queue"]:
    if r.get("disposition") not in {"keep", "retire", "sharpen"}:
        probs.append(f"{r['ref']}: disposition {r.get('disposition')!r}")
    if not str(r.get("command") or "").strip():
        probs.append(f"{r['ref']}: no re-measure command")
    if f"| {r['ref']} |" not in text:
        probs.append(f"{r['ref']}: not rendered in the Part-I block")
check("i5-handoff", "partI-handoff.json i5Queue rows (disposition ∈ keep|retire|sharpen, non-empty re-measure command) ↔ the section's Part-I block", probs,
      f"{len(handoff['i5Queue'])} I.5 rows ({Counter(r['disposition'] for r in handoff['i5Queue'])}); the parity-coverage claim is not an I.5 row at this station — its two enforcement layers plus the Rust companion are asserted on the CD-025 disposition prose above")

# --- no timing literal; every PD entry UNMEASURED with a claim type and a queue marker ----------
_NUM = r"(?<![\w.\-/])\d+(?:\.\d+)?"
TIMING = (
    ("duration", re.compile(_NUM + r"\s?(?:ns|nsecs?|µs|us|usecs?|ms|msecs?|s|secs?|seconds?|mins?|minutes?|h|hrs?|hours?)\b")),
    ("throughput", re.compile(_NUM + r"\s?(?:[KMGT]i?B|LPs?|solves?|iter(?:ation)?s?|ops|cuts|scenarios|passes|nodes|rows)\s?(?:/|per)\s?(?:s|sec|seconds?|min|minutes?|h|hours?)\b|\b(?:[KMGT]i?B|LPs?|solves)/s\b", re.I)),
    ("speedup", re.compile(_NUM + r"\s?(?:x|×)\s?(?:faster|slower|speed-?ups?)\b|" + _NUM + r"\s?%\s?(?:faster|slower|speed-?up|regression|improvement)|\bspeed-?up of " + _NUM, re.I)),
)
probs = []
for off, raw in enumerate(section.lines if section else []):
    for kind, rx in TIMING:
        for m in rx.finditer(raw):
            probs.append(f"{kind} literal {m.group(0)!r} at BACKLOG.md:{section.start + 2 + off}")
perf = json.load(open(here / "perf-queue.json"))
queued = {q["id"] for q in perf["queue"]}
for q in perf["queue"]:
    if q.get("status") != "UNMEASURED" or q.get("measured") is not None or q.get("layout") not in {"4t", "2x2"} or q.get("claimType") not in {"single-process", "collective"}:
        probs.append(f"perf-queue {q['id']}: not UNMEASURED with a layout and claim type")
pd_entries = [e for e in entries if e.id.startswith("PD-")]
if not pd_entries:
    probs.append("the section carries no PD entry")
for e in pd_entries:
    m = e.fields.get("Measurement", "")
    if "UNMEASURED" not in m or not re.search(r"claim-type (single-process|collective)", m) or not re.search(r"layout `(4t|2x2)`", m):
        probs.append(f"{e.id}: Measurement bullet lacks UNMEASURED / claim-type / layout")
    if e.id in queued and "perf-queue.json" not in m:
        probs.append(f"{e.id}: queued without the perf-queue.json marker")
    if e.id not in queued and "not queued" not in m:
        probs.append(f"{e.id}: Sev-C row without the not-queued note")
probs += [f"perf-queue {q}: no PD entry in the section" for q in queued if q not in {e.id for e in pd_entries}]
check("no-timing", "unit-anchored duration/throughput/speedup regexes over the section; PD entries' Measurement bullets (UNMEASURED, claim-type, layout, perf-queue.json marker); perf-queue.json rows", probs,
      f"{len(section.lines) if section else 0} section lines scanned, 0 figures; {len(pd_entries)} PD entries UNMEASURED with claim-type + layout; {len(queued)} queued to perf-queue.json (the perf lens is queue-only)")

# --- read-only workspace -----------------------------------------------------------------------
snapshot_file = here / "tree-baseline.porcelain"
snapshot = snapshot_file.read_text(encoding="utf-8").splitlines() if snapshot_file.exists() else []
porcelain = run(["git", "status", "--porcelain"]).stdout.splitlines()
SURFACE = re.compile(r"^(crates|scripts|schemas|\.github|Cargo\.(?:toml|lock)|docs|examples|tests)(/|$)")
probs = []
station_writes = []
for line in porcelain:
    if line in snapshot:
        continue
    path = line[3:].split(" -> ")[-1].strip()
    if path == ".gitignore":
        continue  # carried-in exemption (dirty at a136840d; clean at 077dbe2c)
    if line.startswith("??") and path.startswith("plans/"):
        station_writes.append(path)
        continue
    if SURFACE.match(path):
        probs.append(f"NEW modified tracked path under an evaluated surface: {line.strip()}")
diff = run(["git", "diff", "--stat", "HEAD", "--", "crates", "docs", "schemas", "scripts", ".github", "Cargo.toml"]).stdout
if diff.strip():
    probs.append("git diff --stat HEAD over the evaluated surfaces is not empty: " + " / ".join(diff.strip().splitlines()[-3:]))
check("read-only-snapshot", "`git status --porcelain` minus the pre-station snapshot, `.gitignore` exempt, filtered to the evaluated surfaces; `git diff --stat HEAD -- crates docs schemas scripts .github Cargo.toml`", probs,
      f"{'snapshot ' + str(len(snapshot)) + ' lines' if snapshot else 'no pre-station snapshot on disk (the tree was clean when the station opened)'}; carried-in exemption `.gitignore` (dirty at a136840d per the ticket, clean at {base[:8]}); {len(station_writes)} untracked plans/ path(s) are the station's own writes; diff over the evaluated surfaces empty")

# --- render --------------------------------------------------------------------------------------
out = [marker, ""]
shared_label = "skipped (--no-shared: the station test module invokes this script from inside the shared verifier)" if shared_rc == "skipped" else f"exit {shared_rc}"
out += ["| Check | Command | Exit | Result |", "| --- | --- | --- | --- |"]
out.append(f"| shared-verifier | `bash {REL}/tools/verify-station.sh cli-python '{title}'` | {'-' if shared_rc == 'skipped' else shared_rc} | {'SKIP' if shared_rc == 'skipped' else ('PASS' if shared_rc == '0' else 'FAIL')} |")
for label, command, code, _ in rows:
    out.append(f"| {label} | {command} | {code} | {'PASS' if code == 0 else 'FAIL'} |")
cargo = figs.get("py_cargo_check", {})
out.append(f"| py-build | `cargo check --manifest-path crates/cobre-python/Cargo.toml` (no libpython RUSTFLAGS override; cargo check never links) | {cargo.get('measured', '?')} | {'PASS' if cargo.get('status') == 'OK' else 'FAIL'} |")
ci_ok = all(figs[k]["status"] == "OK" for k in ("ci_cargo_test_steps", "workflows_running_py_rust_tests", "ci_python_job_maturin", "ci_python_job_pytest", "ci_python_job_requires_cli_binary"))
out.append(f"| ci-visibility | ci.yml at {base[:8]}: python job = maturin + pytest (--require-cli-binary) + `cargo test --manifest-path crates/cobre-python/Cargo.toml`; workflows running the bindings' Rust tests = 1 | {0 if ci_ok else 1} | {'PASS' if ci_ok else 'FAIL'} |")
out.append("")
out.append(f"- shared-verifier: {shared_label}")
out += [f"- {label}: {detail}" for label, _, _, detail in rows]
out.append(f"- py-build: exit {cargo.get('measured', '?')}; the RUSTFLAGS override the ticket describes is needed only to RUN the crate's Rust tests locally, and CI runs them with `--no-default-features --features highs` (ci.yml:567).")
out.append("- ci-visibility: the ticket's premise ('no workflow invokes cargo test or nextest against that manifest') is SUPERSEDED at the pin — ci.yml:562-568 runs the 19 Rust tests; the block therefore asserts the measured CI-visible state and fails if that step disappears (the Cleared cross-reference in the section rests on it; supersession-notes.md SN-06).")
out.append("- the perf lens is queue-only at this station: every figure is owned by the performance sweep (`perf-queue.json`), none is asserted in the register.")
out.append("")
out.append("Superseded figures (old → new, command in supersession-notes.md): " + "; ".join(f"{n['id']} {n['fact']}: {n['old'][:60].rstrip()}… → {n['new'][:70].rstrip()}…" for n in notes) + ".")
out.append(f"Carried-in dirty tracked path: none at {base[:8]} (the ticket's `.gitignore` premise is stale; the exemption rule stays).")
if failures:
    out += ["", "Failures:", *(f"- {f}" for f in failures)]
out.append("")
sys.stdout.write("\n".join(out))
for f in failures:
    print(f"FAIL {f}", file=sys.stderr)
sys.exit(1 if failures else 0)
PY

if [ "$RUN_SHARED" = 1 ]; then
  # append (idempotently) the station block after the shared head verify-station.sh wrote
  if [ -f "$REPORT" ]; then
    python3 - "$REPORT" "$BLOCK" "$MARKER" <<'PY'
import pathlib, sys
report, block, marker = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"), sys.argv[3]
existing = report.read_text(encoding="utf-8")
head = existing.split(marker, 1)[0].rstrip() + "\n\n"
report.write_text(head + block, encoding="utf-8")
PY
  else
    cp "$BLOCK" "$REPORT"
  fi
else
  cat "$BLOCK"
fi

if [ "$FAILED" = 0 ]; then
  echo "verify-figures.sh cli-python: PASS ($(grep -c $'\tOK\t' "$OUT") figures OK; station checks recorded${SHARED_RC:+; shared verifier exit $SHARED_RC})"
else
  echo "verify-figures.sh cli-python: FAIL (see figures.tsv DRIFT rows and the FAIL lines above)" >&2
fi
exit "$FAILED"
