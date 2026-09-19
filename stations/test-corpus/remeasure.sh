#!/usr/bin/env bash
# remeasure.sh — test-corpus station: re-run every snapshot figure docs/design/testing-architecture.md
# froze ("Snapshot figures are calibration-time context; re-measure before acting"), at the register
# pin, and assemble stations/test-corpus/inventory.json before any attacker lens opens. Read-only
# over the codebase: writes only remeasure.log and inventory.json beside this script (scratch under
# $TMPDIR); the three toolchain figures compile into the untracked target/ directories only.
#
#   bash plans/architecture-debt-audit/stations/test-corpus/remeasure.sh
#
# Baseline: the register pin (BACKLOG.md header `Baseline: <sha40>`), not the ticket's superseded
# scaffold pin. HEAD sits past the pin on plans/ commits, so the assertion is "every evaluated surface
# at HEAD is identical to the pin" (the ID-free mirror the evaluation itself writes is exempt), and
# every census is taken from the pin's tree through `git grep <pin>` / `git ls-tree <pin>` /
# `git show <pin>:` so no untracked build artifact can enter a count. The toolchain figures (nextest
# list, doctests, pytest collection) run on the worktree, which the same assertion proves equal.
#
# A figure whose command exits non-zero is recorded UNMEASURED with the stderr excerpt as reason; it
# is never retried, never mutated and never back-filled from the doc.
#
# -e is deliberately omitted: every figure runs and the assembler records each outcome.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
AUDIT="$ROOT/plans/architecture-debt-audit"
BACKLOG="$AUDIT/BACKLOG.md"
LOG="$HERE/remeasure.log"
INVENTORY="$HERE/inventory.json"
MIRROR="docs/design/reserved-seams-and-deferred-debt.md"
SURFACES=(crates .github scripts schemas examples tests docs Cargo.toml Cargo.lock)
RAW="$(mktemp -d "${TMPDIR:-/tmp}/remeasure-test-corpus.XXXXXX")"
trap 'rm -rf "$RAW"' EXIT
cd "$ROOT" || exit 2

PIN="$(sed -n 's/^Baseline: \([0-9a-f]\{40\}\) (pinned .*$/\1/p' "$BACKLOG" | head -1)"
[[ "$PIN" =~ ^[0-9a-f]{40}$ ]] || { echo "remeasure.sh: no 40-hex Baseline line in $BACKLOG" >&2; exit 2; }

assert_baseline() {
  git rev-parse --verify --quiet "$PIN^{commit}" >/dev/null || { echo "remeasure.sh: pin $PIN is not a commit here" >&2; exit 2; }
  if ! git diff --quiet "$PIN" HEAD -- "${SURFACES[@]}" ":(exclude)$MIRROR"; then
    echo "remeasure.sh: evaluated surfaces at HEAD ($(git rev-parse --short HEAD)) differ from the pin ${PIN:0:8}:" >&2
    git diff --stat "$PIN" HEAD -- "${SURFACES[@]}" ":(exclude)$MIRROR" >&2
    exit 2
  fi
  local dirty
  dirty="$(git status --porcelain -- "${SURFACES[@]}")"
  [[ -z "$dirty" ]] || { echo "remeasure.sh: worktree modifies an evaluated surface:" >&2; echo "$dirty" >&2; exit 2; }
}

# emit <figure-id> <source> <definition> <command-string>
# Runs the command verbatim through bash -c, stamps one labelled block into remeasure.log (figure id,
# source section, definition, the verbatim command, exit code, raw stdout, stderr excerpt) and keeps
# the raw stdout / exit code for the assembler. A non-zero exit is stamped as UNMEASURED and returned
# as-is: nothing retries or mutates the command.
emit() {
  local id="$1" src="$2" def="$3" cmd="$4" rc
  bash -c "$cmd" >"$RAW/$id.out" 2>"$RAW/$id.err"
  rc=$?
  printf '%s' "$rc" >"$RAW/$id.rc"
  printf '%s\n%s\n%s\n' "$src" "$def" "$cmd" >"$RAW/$id.meta"
  {
    printf '=== figure=%s source=%s definition=%s\n' "$id" "$src" "$def"
    printf 'command: %s\n' "$cmd"
    printf 'exit: %s\n' "$rc"
    if ((rc == 0)); then
      printf 'status: measured\nstdout:\n'
      cat "$RAW/$id.out"
    else
      printf 'status: unmeasured (non-zero exit; not retried, not mutated, not back-filled from the doc)\nstdout:\n'
      cat "$RAW/$id.out"
      printf 'stderr(excerpt):\n'
      head -c 600 "$RAW/$id.err"
      printf '\n'
    fi
    printf '=== end %s\n\n' "$id"
  } >>"$LOG"
  return "$rc"
}

assert_baseline
DESCRIBE="$(git describe --tags "$PIN")"
{
  printf '# test-corpus re-measurement — baseline %s (%s), run %s\n' "$PIN" "$DESCRIBE" "$(date -I)"
  printf '# HEAD %s; evaluated surfaces identical to the pin (mirror exempt); censuses read the pin, toolchain figures run on the worktree\n\n' "$(git rev-parse HEAD)"
} >"$LOG"

MEMBERS=(cobre cobre-core cobre-io cobre-stochastic cobre-solver cobre-comm cobre-sddp cobre-cli cobre-mcp cobre-tui cobre-flow cobre-uc cobre-emt cobre-python)
CRATES="${MEMBERS[*]}"

# --- §2.1 rows 1-2: integration binaries (depth-1 tests/*.rs, what Cargo links) and the solver-linking subset
emit int-binaries ta-2.1 binary \
  "for c in $CRATES; do printf '%s %s\n' \"\$c\" \"\$(git ls-tree --name-only $PIN crates/\$c/tests/ 2>/dev/null | grep -c '\.rs\$')\"; done"
emit int-files ta-2.1 file \
  "for c in $CRATES; do printf '%s %s\n' \"\$c\" \"\$(git ls-tree -r --name-only $PIN crates/\$c/tests/ 2>/dev/null | grep -c '\.rs\$')\"; done"
emit int-binaries-solver-linking ta-2.1 binary \
  "for c in cobre-sddp cobre-cli cobre-solver; do printf '%s %s\n' \"\$c\" \"\$(git ls-tree --name-only $PIN crates/\$c/tests/ | grep -c '\.rs\$')\"; done"

# --- §2.1 rows 3-5: the toolchain figures (worktree; may legitimately come back UNMEASURED)
emit nextest-list ta-2.1 test "cargo nextest list --features test-support"
emit doctests ta-2.1 doctest "cargo test --doc"
emit pytest-collect ta-2.1 test "pytest crates/cobre-python --collect-only -q"

# --- §2.1 row 6: golden bit-exact cases (the dual-backend parity decks) and the roster behind them
emit golden-cases ta-2.1 deck \
  "for d in parity_baselines parity_baselines_clp; do printf '%s %s\n' \"\$d\" \"\$(git ls-tree --name-only $PIN crates/cobre-sddp/tests/fixtures/\$d/ | wc -l)\"; done; git ls-tree --name-only $PIN crates/cobre-sddp/tests/fixtures/parity_baselines/ crates/cobre-sddp/tests/fixtures/parity_baselines_clp/"
emit golden-roster ta-2.1 function \
  "git grep -ho 'fn parity_hash_[a-z0-9_]*' $PIN -- crates/cobre-sddp/tests | sort -u; printf 'occurrences %s\n' \"\$(git grep -o 'parity_hash_' $PIN -- crates/cobre-sddp/tests | wc -l)\""

# --- §2.1 rows 7-9: git-tracked censuses at the pin (files under one definition, occurrences under the other)
emit to-bits ta-2.1 file "git grep -c 'to_bits' $PIN -- 'crates/**/*.rs' | wc -l"
emit to-bits-occurrences ta-2.1 occurrence "git grep -o 'to_bits' $PIN -- 'crates/**/*.rs' | wc -l"
emit proptest-sites ta-2.1 file "git grep -l 'proptest!' $PIN -- 'crates/**/*.rs' | wc -l"
emit proptest-invocations ta-2.1 invocation "git grep -c 'proptest!' $PIN -- 'crates/**/*.rs'; printf 'total %s\n' \"\$(git grep -o 'proptest!' $PIN -- 'crates/**/*.rs' | wc -l)\""
emit slow-tests-attrs ta-2.1 file "git grep -c 'slow-tests' $PIN -- 'crates/**/*.rs' | wc -l"
emit slow-tests-tracked ta-2.1 file "git grep -l 'slow-tests' $PIN -- 'crates/**' | sed 's#^$PIN:##' | sed 's#crates/\([^/]*\)/.*#\1#' | sort | uniq -c; printf 'tracked-files %s\n' \"\$(git grep -l 'slow-tests' $PIN -- 'crates/**' | wc -l)\"; printf 'occurrences-rs %s\n' \"\$(git grep -o 'slow-tests' $PIN -- 'crates/**/*.rs' | wc -l)\""

# --- §2.3 / §3.2 prose figures, commands authored here
emit nextest-config ta-2.3 n/a "git cat-file -e $PIN:.config/nextest.toml 2>/dev/null && echo present || echo absent"
emit ci-runner-split ta-2.3 line "git show $PIN:.github/workflows/ci.yml | grep -n 'cargo test --workspace\|cargo nextest run'"
emit slow-tests-in-pr-features ta-2.3 line "git show $PIN:.github/workflows/ci.yml | grep -n 'NON_SOLVER_FEATURES'"
emit shuffle-cadence ta-2.3 line "git show $PIN:.github/workflows/invariance-shuffle.yml | grep -n 'workflow_dispatch\|schedule\|cron'"
emit test-support-manifests ta-3.2 line "git grep -n 'test-support' $PIN -- 'crates/*/Cargo.toml' | sed 's#^$PIN:##'"
emit stubcomm-home ta-3.2 line \
  "git grep -n 'struct StubComm\b\|struct Rank0Of2\b' $PIN -- crates | sed 's#^$PIN:##'; printf 'cobre-comm-hits %s\n' \"\$(git grep -l 'StubComm\|Rank0Of2' $PIN -- 'crates/cobre-comm/**' | wc -l)\""
emit sibling-tests-rs ta-3.2 file \
  "git ls-tree -r --name-only $PIN crates | grep '/src/.*tests\.rs\$' | sed 's#crates/\([^/]*\)/.*#\1#' | sort | uniq -c"
emit inline-cfg-test-files ta-3.2 file \
  "for c in $CRATES; do printf '%s %s\n' \"\$c\" \"\$(git grep -l '#\[cfg(test)\]' $PIN -- \"crates/\$c/src/*.rs\" \"crates/\$c/src/**/*.rs\" 2>/dev/null | wc -l)\"; done"
emit homing-item4-anchors ta-3.2 line \
  "git ls-tree -r --name-only $PIN crates/cobre-sddp/src/lp/builder/ | grep -E '/(entries|columns|template|layout)(/tests)?\.rs\$'; git grep -c '#\[cfg(test)\]' $PIN -- crates/cobre-sddp/src/lp/builder/entries.rs crates/cobre-sddp/src/lp/builder/columns.rs crates/cobre-sddp/src/lp/builder/template.rs crates/cobre-sddp/src/lp/builder/layout.rs | sed 's#^$PIN:##'"

# --- harness (§2.2): the cobre-sddp-only machinery, recorded so later lenses know it exists
emit harness-common ta-2.2 file "git ls-tree --name-only $PIN crates/cobre-sddp/tests/common/"
emit harness-fixtures ta-2.2 deck "git ls-tree --name-only $PIN crates/cobre-sddp/tests/fixtures/"
emit harness-benches ta-2.2 file "git ls-tree -r --name-only $PIN crates | grep '/benches/' ; printf 'bench-dirs %s\n' \"\$(git ls-tree -r --name-only $PIN crates | grep '/benches/' | sed 's#\(crates/[^/]*\)/.*#\1#' | sort -u | wc -l)\""
emit harness-mpi-wire ta-2.2 file "git ls-tree --name-only $PIN crates/cobre-sddp/tests/ | grep 'mpi_wire'"
emit oracle-close-anchors ta-2.2 line "git grep -n 'fn close' $PIN -- crates/cobre-sddp/tests/extensive_form_oracle.rs crates/cobre-sddp/tests/branching_value_oracle.rs | sed 's#^$PIN:##'"

# --- read-only proof, run last, then the baseline assertion again
assert_baseline
git status --porcelain -- "${SURFACES[@]}" >"$RAW/porcelain-surfaces.txt"
git status --porcelain >"$RAW/porcelain-all.txt"
LEAK="$(grep -v '^?? plans/' "$RAW/porcelain-all.txt" | grep -v '^ M plans/' | grep -v '^ M \.gitignore$' || true)"
{
  printf '=== read-only proof\n'
  printf 'pin: %s (%s); HEAD: %s\n' "$PIN" "$DESCRIBE" "$(git rev-parse HEAD)"
  printf 'git status --porcelain -- %s:\n%s\n' "${SURFACES[*]}" "$(cat "$RAW/porcelain-surfaces.txt")"
  printf 'git status --porcelain (all):\n%s\n' "$(cat "$RAW/porcelain-all.txt")"
  printf 'leak outside plans/ and .gitignore: %s\n' "${LEAK:-none}"
  printf '=== end read-only proof\n'
} >>"$LOG"

# --- assemble inventory.json from the raw blocks
python3 - "$RAW" "$INVENTORY" "$PIN" "$DESCRIBE" "$(git rev-parse HEAD)" "$LEAK" <<'PY'
import json
import pathlib
import re
import sys
from collections import Counter

raw, out_path, pin, describe, head, leak = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
MEMBERS = ["cobre", "cobre-core", "cobre-io", "cobre-stochastic", "cobre-solver", "cobre-comm", "cobre-sddp", "cobre-cli", "cobre-mcp", "cobre-tui", "cobre-flow", "cobre-uc", "cobre-emt"]
ALL = MEMBERS + ["cobre-python"]
SOLVER_LINKING = ["cobre-sddp", "cobre-cli", "cobre-solver"]


def block(fid):
    src, definition, cmd = (raw / f"{fid}.meta").read_text(encoding="utf-8").split("\n")[:3]
    rc = int((raw / f"{fid}.rc").read_text())
    out = (raw / f"{fid}.out").read_text(encoding="utf-8", errors="replace")
    err = (raw / f"{fid}.err").read_text(encoding="utf-8", errors="replace")
    return {"source": src, "definition": definition, "command": cmd, "rc": rc, "out": out, "err": err}


def pairs(text):
    return {ln.split()[0]: int(ln.split()[1]) for ln in text.splitlines() if re.match(r"^\S+ \d+$", ln.strip())}


def counted(text):
    """`uniq -c` output → {name: count}."""
    return {ln.split()[1]: int(ln.split()[0]) for ln in text.splitlines() if re.match(r"^\s*\d+ \S+$", ln)}


def figure(fid, text, *, value=None, extra=None, status_override=None, blocks=None):
    """One figure record; a composed figure passes the blocks it is built from (first = primary)."""
    parts = blocks or [block(fid)]
    b = next((x for x in parts if x["rc"] != 0), parts[0])
    rec = {"id": fid, "source": parts[0]["source"], "figure": text, "command": " ; ".join(x["command"] for x in parts), "definition": parts[0]["definition"]}
    if b["rc"] != 0 or status_override == "unmeasured":
        rec.update(value=None, status="unmeasured", reason=f"exit {b['rc']}: " + " ".join(b["err"].strip().split())[:600] or "no stderr")
    else:
        rec.update(value=value, status="measured")
    if extra:
        rec.update(extra)
    return rec


figures = []
deviations = []

# 1-2: integration binaries / files / solver-linking subset
bins = pairs(block("int-binaries")["out"])
files = pairs(block("int-files")["out"])
solver = pairs(block("int-binaries-solver-linking")["out"])
figures.append(figure("int-binaries", "Integration-test binaries, per crate (depth-1 tests/*.rs — what Cargo links)", value=sum(bins.values()), extra={"perCrate": bins, "altDefinition": {"integrationFiles": sum(files.values()), "perCrate": files, "note": "recursive tests/**/*.rs — what a grep-based lens sees; sweeps in tests/common/*.rs and tests/fixtures/**"}, "docValue": None, "docNote": "the §2.1 table gives the command, not a frozen number"}))
figures.append(figure("int-binaries-solver-linking", "…that statically link the solver (crates depending on cobre-solver: cobre-sddp, cobre-cli, cobre-solver)", value=sum(solver.values()), extra={"perCrate": solver, "ofTotalBinaries": sum(bins.values())}))

# 3: nextest list — count listed tests (indented lines under each binary header)
nb = block("nextest-list")
if nb["rc"] == 0:
    tests = [ln for ln in nb["out"].splitlines() if re.match(r"^\S+ \S+$", ln)]
    binaries = sorted({ln.split()[0] for ln in tests})
    per_bin_crate = Counter(b.split("::", 1)[0] for b in binaries)
    per_test_crate = Counter(ln.split()[0].split("::", 1)[0] for ln in tests)
    figures.append(figure("nextest-list", "Unit + integration test count (cargo nextest list --features test-support)", value=len(tests), extra={"listedBinaries": len(binaries), "perCrateTests": dict(sorted(per_test_crate.items())), "perCrateBinaries": dict(sorted(per_bin_crate.items())), "parse": "non-TTY list output: one `<binary-id> <test>` line per test (a lib unit-test binary id is the bare crate name, an integration binary is `<crate>::<file>`); binaries are the distinct first tokens"}))
else:
    figures.append(figure("nextest-list", "Unit + integration test count (cargo nextest list --features test-support)"))

# 4: doctests — sum of `running N tests` per Doc-tests block
db = block("doctests")
if db["rc"] == 0:
    running = [int(m.group(1)) for m in re.finditer(r"^running (\d+) tests?", db["out"], re.M)]
    headers = re.findall(r"^\s*Doc-tests (\S+)", db["err"], re.M)
    results = [tuple(int(x) for x in m.groups()) for m in re.finditer(r"^test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored", db["out"], re.M)]
    per = Counter(m.group(1) for m in re.finditer(r"^test crates/([^/]+)/", db["out"], re.M))
    figures.append(figure("doctests", "Doctests (cargo test --doc)", value=sum(running), extra={"perCrate": dict(sorted(per.items())), "testLines": sum(per.values()), "blocks": len(running), "docTestHeadersOnStderr": len(headers), "passed": sum(r[0] for r in results), "failed": sum(r[1] for r in results), "ignored": sum(r[2] for r in results), "parse": "sum of `running N tests` lines on stdout; per-crate attribution from the `test crates/<crate>/... - ... ... ok|ignored` lines (the `Doc-tests <crate>` headers go to stderr and are fewer than the blocks, so they are not zipped)"}))
else:
    figures.append(figure("doctests", "Doctests (cargo test --doc)"))

# 5: pytest collection
pb = block("pytest-collect")
if pb["rc"] == 0:
    m = re.search(r"(\d+) tests? collected", pb["out"] + pb["err"])
    if m:
        figures.append(figure("pytest-collect", "pytest (cobre-python) collected tests", value=int(m.group(1)), extra={"parse": "the `N tests collected` summary line"}))
    else:
        figures.append(figure("pytest-collect", "pytest (cobre-python) collected tests", status_override="unmeasured"))
        figures[-1]["reason"] = "exit 0 but no `N tests collected` summary in the output"
else:
    figures.append(figure("pytest-collect", "pytest (cobre-python) collected tests"))

# 6: golden bit-exact cases (decks × backends) + the roster behind them
gb = block("golden-cases")
decks = pairs("\n".join(ln for ln in gb["out"].splitlines() if not ln.startswith("crates/")))
deck_files = [ln for ln in gb["out"].splitlines() if ln.startswith("crates/")]
rb = block("golden-roster")
roster = [ln.split()[1] for ln in rb["out"].splitlines() if ln.startswith("fn ")]
occ = int(re.search(r"occurrences (\d+)", rb["out"]).group(1)) if rb["rc"] == 0 and re.search(r"occurrences (\d+)", rb["out"]) else None
figures.append(figure("golden-cases", "Golden bit-exact cases: the tests/fixtures/parity_baselines* decks × two backends (HiGHS / CLP)", blocks=[gb, rb], value=sum(decks.values()), extra={"decks": decks, "deckFiles": deck_files, "rosterFunctions": roster, "goldenRosterSize": len(roster), "prefixOccurrences": occ, "note": "roster size is the count of distinct parity_hash_* test functions, not the occurrence count of the prefix"}))

# 7: to_bits (files) with occurrences as the alternate reading
tb = block("to-bits")
tbo = block("to-bits-occurrences")
figures.append(figure("to-bits", "to_bits/ULP determinism assertions (files carrying `to_bits`)", blocks=[tb, tbo], value=int(tb["out"].strip()) if tb["rc"] == 0 else None, extra={"altDefinition": {"occurrences": int(tbo["out"].strip()) if tbo["rc"] == 0 else None}}))

# 8: proptest! — both readings, drift against §3.2 item 7 ("only 5 sites")
pf = block("proptest-sites")
pi = block("proptest-invocations")
inv_total = int(re.search(r"total (\d+)", pi["out"]).group(1)) if pi["rc"] == 0 and re.search(r"total (\d+)", pi["out"]) else None
per_file = {ln.rsplit(":", 1)[0].split(":", 1)[-1]: int(ln.rsplit(":", 1)[1]) for ln in pi["out"].splitlines() if re.search(r":\d+$", ln)}
pfv = int(pf["out"].strip()) if pf["rc"] == 0 else None
figures.append(figure("proptest-sites", "proptest! sites", blocks=[pf, pi], value=pfv, extra={"altDefinition": {"invocations": inv_total, "perFile": per_file}, "docValue": 5, "docAnchor": "ta-3.2 item 7 ('property testing at only 5 sites')", "driftAgainstDoc": pfv is not None and pfv != 5 and inv_total != 5, "driftNote": "the doc's 5 matches neither reading; handed to ingest as a prose-drift candidate, not adjudicated here"}))

# 9: slow-tests — .rs files (the §2.1 command), tracked files of any type, occurrences, per-crate distribution
sa = block("slow-tests-attrs")
st = block("slow-tests-tracked")
dist = counted("\n".join(ln for ln in st["out"].splitlines() if not ln.startswith(("tracked-files", "occurrences-rs"))))
tracked = int(re.search(r"tracked-files (\d+)", st["out"]).group(1)) if st["rc"] == 0 else None
occ_rs = int(re.search(r"occurrences-rs (\d+)", st["out"]).group(1)) if st["rc"] == 0 else None
sav = int(sa["out"].strip()) if sa["rc"] == 0 else None
figures.append(figure("slow-tests-attrs", "Slow-gated (`slow-tests`) attributes — .rs files carrying the token (git-tracked census at the pin)", blocks=[sa, st], value=sav, extra={"altDefinition": {"trackedFilesAnyType": tracked, "occurrencesInRs": occ_rs, "perCrateTrackedFiles": dist}, "docClaim": "concentrated entirely in cobre-sddp", "docClaimHolds": set(dist) == {"cobre-sddp"}, "censusHygiene": "git grep at the pin — a naive recursive grep over the worktree also sweeps the untracked build directories under crates/*/ and is never used here"}))

# 10-15: unpaired prose figures
nc = block("nextest-config")
figures.append(figure("nextest-config", ".config/nextest.toml presence (§2.3: no profiles, retries, partitioning, JUnit or archive)", value=nc["out"].strip() or None))
cr = block("ci-runner-split")
runner_lines = {int(ln.split(":", 1)[0]): ln.split(":", 1)[1].strip() for ln in cr["out"].splitlines() if re.match(r"^\d+:", ln)}
figures.append(figure("ci-runner-split", "Per-job runner split in .github/workflows/ci.yml (cargo test in the HiGHS Test job vs cargo nextest run in the CLP job)", value=len(runner_lines), extra={"anchors": {f".github/workflows/ci.yml:{k}": v for k, v in runner_lines.items()}}))
sp = block("slow-tests-in-pr-features")
nsf = {int(ln.split(":", 1)[0]): ln.split(":", 1)[1].strip() for ln in sp["out"].splitlines() if re.match(r"^\d+:", ln)}
defn = next((v for v in nsf.values() if v.startswith("NON_SOLVER_FEATURES:")), "")
figures.append(figure("slow-tests-in-pr-features", "slow-tests inside NON_SOLVER_FEATURES (the PR feature set), so the slow suite runs on every PR", value="slow-tests" in defn, extra={"definitionAnchor": f".github/workflows/ci.yml:{min(nsf)}" if nsf else None, "definition_line": defn, "consumerAnchors": [f".github/workflows/ci.yml:{k}" for k, v in nsf.items() if not v.startswith("NON_SOLVER_FEATURES:") and "run:" in v]}))
sc_ = block("shuffle-cadence")
cad = {int(ln.split(":", 1)[0]): ln.split(":", 1)[1].strip() for ln in sc_["out"].splitlines() if re.match(r"^\d+:", ln)}
figures.append(figure("shuffle-cadence", "invariance-shuffle.yml cadence: workflow_dispatch only, nightly cron commented out", value="workflow_dispatch-only" if any("workflow_dispatch" in v for v in cad.values()) and all(v.lstrip().startswith("#") for v in cad.values() if "cron" in v) else "other", extra={"anchors": {f".github/workflows/invariance-shuffle.yml:{k}": v for k, v in cad.items()}}))
ts = block("test-support-manifests")
decl, cons = {}, {}
for ln in ts["out"].splitlines():
    m = re.match(r"^crates/([^/]+)/Cargo\.toml:(\d+):(.*)$", ln)
    if not m:
        continue
    crate, line, text = m.group(1), int(m.group(2)), m.group(3).strip()
    if re.match(r"^test-support\s*=", text):
        decl[crate] = f"crates/{crate}/Cargo.toml:{line}"
    elif 'features = ["test-support"]' in text or "features = [\"test-support\"]" in text:
        cons.setdefault(crate, []).append(f"crates/{crate}/Cargo.toml:{line}")
sh = block("stubcomm-home")
homes = [ln for ln in sh["out"].splitlines() if ":" in ln and not ln.startswith("cobre-comm-hits")]
comm_hits = int(re.search(r"cobre-comm-hits (\d+)", sh["out"]).group(1)) if sh["rc"] == 0 else None
harness_home = [h for h in homes if "tests/common/mod.rs" in h and "pub struct" in h]
figures.append(figure("test-support-declarers-vs-consumers", "test-support declarers versus consumers across crates/*/Cargo.toml; StubComm/Rank0Of2 home", blocks=[ts, sh], value={"declarers": sorted(decl), "consumers": sorted(cons), "neither": sorted(set(ALL) - set(decl) - set(cons))}, extra={"declarerAnchors": decl, "consumerAnchors": cons, "stubCommRank0Of2": {"harnessHome": harness_home, "otherDefinitions": [h for h in homes if h not in harness_home], "cobreCommHits": comm_hits, "note": "§5.8 names cobre-comm as the intended home; at the pin the harness pair lives in crates/cobre-sddp/tests/common/mod.rs and every other definition is a per-file or cfg(test)-local stub"}}))
sib = counted(block("sibling-tests-rs")["out"])
inl = pairs(block("inline-cfg-test-files")["out"])
h4 = block("homing-item4-anchors")
h4_files = [ln for ln in h4["out"].splitlines() if ln.startswith("crates/") and ":" not in ln]
h4_counts = {ln.rsplit(":", 1)[0]: int(ln.rsplit(":", 1)[1]) for ln in h4["out"].splitlines() if re.search(r"\.rs:\d+$", ln)}
figures.append(figure("homing-split", "Inline-giant versus extracted-sibling homing split (§3.2 item 4): sibling src/**/tests.rs files vs source files carrying #[cfg(test)], per crate", blocks=[block("sibling-tests-rs"), block("inline-cfg-test-files"), h4], value={"siblingTestsRsTotal": sum(sib.values()), "inlineCfgTestFilesTotal": sum(inl.values())}, extra={"perCrate": {c: {"siblingTestsRs": sib.get(c, 0), "inlineCfgTestModules": inl.get(c, 0)} for c in ALL}, "item4Anchors": {"lpBuilderFiles": h4_files, "cfgTestLinesPerFile": h4_counts, "note": "entries.rs and columns.rs keep inline #[cfg(test)] modules while template.rs and layout.rs have extracted tests.rs siblings"}}))

# per-crate array (six keys, each under its stated definition)
per_crate = []
for c in ALL:
    per_crate.append({"crate": c, "integrationBinaries": bins.get(c, 0), "integrationFiles": files.get(c, 0), "siblingTestsRs": sib.get(c, 0), "inlineCfgTestModules": inl.get(c, 0), "declaresTestSupport": c in decl, "consumesTestSupport": c in cons})

# harness record
hc = [ln.rsplit("/", 1)[-1] for ln in block("harness-common")["out"].splitlines() if ln.strip()]
hf = [ln.rsplit("/", 1)[-1] for ln in block("harness-fixtures")["out"].splitlines() if ln.strip()]
hb_block = block("harness-benches")
hb = [ln for ln in hb_block["out"].splitlines() if "/benches/" in ln]
bench_dirs = int(re.search(r"bench-dirs (\d+)", hb_block["out"]).group(1)) if hb_block["rc"] == 0 else None
mpi_wire = block("harness-mpi-wire")["out"].strip().splitlines()
close = {ln.split(":")[0]: int(ln.split(":")[1]) for ln in block("oracle-close-anchors")["out"].splitlines() if re.match(r"^crates/.*:\d+:", ln)}
harness = {
    "home": "crates/cobre-sddp (the only crate with tests/common/, tests/fixtures/ and benches/ at the pin — §2.2)",
    "commonModules": hc,
    "fixtureDecks": {"count": len(hf), "names": hf, "parityBaselines": decks.get("parity_baselines"), "parityBaselinesClp": decks.get("parity_baselines_clp")},
    "benches": {"count": len(hb), "files": hb, "benchDirectoriesWorkspaceWide": bench_dirs},
    "goldenRosterSize": len(roster), "goldenRoster": roster, "parityHashPrefixOccurrences": occ,
    "mpiWireBinary": mpi_wire,
    "oracleDuplicationAnchors": {"fn close": close, "note": "the mirror already tracks the extensive_form_oracle / branching_value_oracle `fn close` duplication; sharpened to pin line numbers for the ingest dup-of merge"},
    "permuteHelper": "permute.rs" in hc,
}

# read-only proof and per-figure status roll-up
statuses = Counter(f["status"] for f in figures)
inventory = {
    "station": "test-corpus",
    "baseline": {"sha": pin, "describe": describe, "measuredOn": __import__("datetime").date.today().isoformat(), "head": head, "source": "plans/architecture-debt-audit/BACKLOG.md header `Baseline: <sha40> (pinned …)` — the register pin; the ticket's a136840d (v0.15.0-1-ga136840d) is the superseded scaffold pin", "assertion": "every evaluated surface at HEAD is identical to the pin (git diff --quiet <pin> HEAD -- crates .github scripts schemas examples tests docs Cargo.toml Cargo.lock, the ID-free mirror exempt); censuses read the pin's tree, toolchain figures run on the worktree"},
    "yardstick": {"path": "docs/design/testing-architecture.md", "statusLine": "Partially adopted (§5.2); the rest Proposal — Snapshot figures are calibration-time context; re-measure before acting", "sections": {"ta-2.1": "Shape table (metric → how to re-measure)", "ta-2.2": "Where the sophistication lives", "ta-2.3": "Tooling & CI topology", "ta-3.2": "Sustainability & uniformity problems"}},
    "definitions": {"binary": "depth-1 crates/<c>/tests/*.rs — one linked integration binary each (the §2.1 command's find -maxdepth 1)", "file": "git-tracked files matching the census (a recursive reading sweeps tests/common/*.rs and tests/fixtures/**)", "occurrence": "git grep -o token count", "invocation": "proptest! macro invocations", "test": "tests as enumerated by the runner", "doctest": "doc examples cargo test --doc compiled and ran", "deck": "fixture directories", "function": "distinct fn definitions", "line": "resolved file:line anchors at the pin", "n/a": "presence / boolean"},
    "figures": figures,
    "figureStatus": dict(statuses),
    "perCrate": per_crate,
    "harness": harness,
    "readOnlyProof": {"surfacesPorcelain": (raw / "porcelain-surfaces.txt").read_text().strip(), "leakOutsidePlans": leak.strip() or None, "porcelainLines": [ln for ln in (raw / "porcelain-all.txt").read_text().splitlines()]},
}
out_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
core = [f for f in figures if f["id"] in ("int-binaries", "int-binaries-solver-linking", "nextest-list", "doctests", "pytest-collect", "golden-cases", "to-bits", "proptest-sites", "slow-tests-attrs", "nextest-config", "ci-runner-split", "slow-tests-in-pr-features", "shuffle-cadence", "test-support-declarers-vs-consumers", "homing-split")]
print(f"inventory: {len(figures)} figure records ({len(core)} yardstick figures), statuses {dict(statuses)}; perCrate {len(per_crate)}; leak={'none' if not leak.strip() else leak.strip()}")
PY
rc=$?
printf '=== assembler exit %s\n' "$rc" >>"$LOG"
exit "$rc"
