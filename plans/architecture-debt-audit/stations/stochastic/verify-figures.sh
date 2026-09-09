#!/usr/bin/env bash
# verify-figures.sh — re-measure every figure the cobre-stochastic station records and
# prove its partition, perf-handoff, heading-resolution and read-only invariants at the
# pinned baseline. Read-only over the codebase: writes only figures.tsv beside this script.
#
# Adaptation (owner ruling): the register section quotes no inventory figures and carries
# no `(measured-by: …)` tags — those figures live in inventory.json — so every figure is
# re-measured from the tree and asserted against inventory.json's recorded value (drift in
# either direction fails). The QMC/LHS fixture-prelude diffs (0 and 52) are re-measured as
# figures but are OUT OF THIS STATION (owned by the test-corpus station / Epic 8 per the
# ratified TD-024 scoping), so no in-station entry quotes them.
#
# -e is deliberately omitted: this runs every check and aggregates a single exit code.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
SRC="$ROOT/crates/cobre-stochastic/src"
TESTS="$ROOT/crates/cobre-stochastic/tests"
INV="$HERE/inventory.json"
OUT="$HERE/figures.tsv"
BACKLOG="$ROOT/plans/architecture-debt-audit/BACKLOG.md"
TOOLS="$ROOT/plans/architecture-debt-audit/tools"
SECTION="★ QUALITY EVALUATION (2026-09, baseline a136840d) — stochastic"
FAIL=0

# Expected figures come from inventory.json (this station's single source of truth for
# figures); a re-measurement that differs from the recorded value is drift in one or the
# other and fails the run.
eval "$(
  python3 - "$INV" <<'PY'
import json
import sys

inv = json.load(open(sys.argv[1]))
t = inv["totals"]
sub = {s["id"]: s["files"] for s in inv["subStations"]}
ext = next(
    c["measured"]["rawLines"]
    for c in inv["corrections"]
    if "external.rs" in (c.get("claim") or "")
)
print(f"E_SRC={t['srcFiles']}")
print(f"E_NONTEST={t['nonTestLines']}")
print(f"E_INLINE={t['inlineTestFiles']}")
print(f"E_INTEG={t['integrationBinaries']}")
print(f"E_PAR={sub['par']}")
print(f"E_SAMP={sub['sampling']}")
print(f"E_TREE={sub['tree-noise']}")
print(f"E_SEAM={sub['seam']}")
print(f"E_EXT={ext}")
PY
)"

printf 'figure\texpected\tmeasured\tstatus\tcommand\n' >"$OUT"
fig() { # fig <name> <expected> <command>
  local name="$1" want="$2" cmd="$3" got status
  got="$(eval "$cmd" 2>/dev/null | tr -d ' \n')"
  if [ "$got" = "$want" ]; then status=OK; else status=DRIFT; FAIL=1; fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$want" "$got" "$status" "$cmd" >>"$OUT"
}

fig src_files "$E_SRC" "find $SRC -name '*.rs' | wc -l"
fig sub_par "$E_PAR" "find $SRC/par -name '*.rs' | wc -l"
fig sub_sampling "$E_SAMP" "find $SRC/sampling -name '*.rs' | wc -l"
fig sub_tree_noise "$E_TREE" "find $SRC/tree $SRC/noise $SRC/normal $SRC/correlation -name '*.rs' | wc -l"
fig sub_seam "$E_SEAM" "{ find $SRC -maxdepth 1 -name '*.rs'; find $SRC/season_cast -name '*.rs'; } | wc -l"
fig nontest_loc "$E_NONTEST" "find $SRC -name '*.rs' ! -name tests.rs -print0 | xargs -0 awk 'FNR==1{k=1} /^#\\[cfg\\(test\\)\\]/{k=0} k{n++} END{print n}'"
fig inline_test_files "$E_INLINE" "grep -rl '^#\\[cfg(test)\\]' $SRC | wc -l"
fig integration_bins "$E_INTEG" "find $TESTS -name '*.rs' | wc -l"
fig external_rs_lines "$E_EXT" "wc -l < $SRC/sampling/external.rs"
fig workspace_deps 1 "cargo tree --manifest-path $ROOT/Cargo.toml -p cobre-stochastic --depth 1 | grep '^[├└]' | grep -o 'cobre-[a-z-]*' | sort -u | wc -l"
# Out-of-station (test-corpus / Epic 8): re-measured here as evidence, not quoted in any
# in-station entry — the QMC/LHS prelude duplication belongs to the test-corpus station.
fig qmc_prelude_diff 0 "diff <(sed -n '1,317p' $TESTS/halton_integration.rs | sed 's/halton/QMC/g;s/Halton/QMC/g') <(sed -n '1,317p' $TESTS/sobol_integration.rs | sed 's/sobol/QMC/g;s/Sobol/QMC/g') | wc -l"
fig lhs_prelude_diff 52 "diff <(sed -n '1,317p' $TESTS/halton_integration.rs | sed 's/halton/QMC/g;s/Halton/QMC/g') <(sed -n '1,317p' $TESTS/lhs_integration.rs | sed 's/lhs/QMC/g;s/Lhs/QMC/g;s/LHS/QMC/g') | wc -l"

column -t -s "$(printf '\t')" "$OUT" >&2

# Partition, heading-resolution, perf-handoff, quoted-count and read-only — all the
# data/logic checks in one parser-backed block.
python3 - "$ROOT" "$INV" "$BACKLOG" "$SECTION" "$TOOLS" "$OUT" <<'PY' || FAIL=1
import collections
import json
import pathlib
import re
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
inv = json.load(open(sys.argv[2]))
backlog = pathlib.Path(sys.argv[3])
section_title = sys.argv[4]
sys.path.insert(0, sys.argv[5])
figures = {
    r["figure"]: r
    for r in (
        dict(zip(("figure", "expected", "measured", "status", "command"), line.split("\t")))
        for line in pathlib.Path(sys.argv[6]).read_text().splitlines()[1:]
    )
}
from lib import backlog_parse as bp  # noqa: E402

bad: list[str] = []

# 1. Sub-station partition: every source file falls in exactly one sweep. The per-file
# assignment lives in files[].subStation; subStations[].files is the recorded count.
tree = {
    str(p.relative_to(root))
    for p in (root / "crates/cobre-stochastic/src").rglob("*.rs")
}
seen: collections.Counter[str] = collections.Counter(f["path"] for f in inv["files"])
unswept = sorted(tree - set(seen))
not_in_tree = sorted(set(seen) - tree)
multi_swept = sorted(p for p, n in seen.items() if n > 1)
if unswept or not_in_tree or multi_swept:
    bad.append(
        f"PARTITION: unswept={unswept} not-in-tree={not_in_tree} multi-swept={multi_swept}"
    )
else:
    counts = collections.Counter(f["subStation"] for f in inv["files"])
    recorded = {s["id"]: s["files"] for s in inv["subStations"]}
    if dict(counts) != recorded:
        bad.append(f"PARTITION COUNTS: files[] {dict(counts)} != subStations {recorded}")
    else:
        print(f"partition OK: {len(tree)} files, disjoint and covering: {recorded}")

# 2. Every re-measured figure matches its inventory-recorded value.
drift = [f"{r['figure']} (want {r['expected']}, got {r['measured']})"
         for r in figures.values() if r["status"] != "OK"]
if drift:
    bad.append("FIGURE DRIFT: " + "; ".join(drift))
else:
    print(f"figures OK: {len(figures)} re-measured, 0 drift")

# 3. Heading resolution: the dated section matches exactly once; the 2026-08 'Station 3'
# headings are deliberately not matched.
lines = bp.read_register(backlog)
hits = sum(1 for line in lines if line.strip() == f"## {section_title}")
if hits != 1:
    bad.append(f"HEADING: '## {section_title}' matched {hits} times, expected 1")
legacy = [
    line for line in lines
    if re.match(r"^## .*station 3", line, re.I) and section_title not in line
]
print(f"heading OK: dated section matched once; {len(legacy)} legacy 'Station 3' heading(s) not matched")

# 4. Quoted counts in the section prose equal verdicts.json (nothing copied from the epic).
section = bp.find_section(lines, "stochastic")
body = "\n".join(section.lines)
verdicts = json.loads(
    (root / "plans/architecture-debt-audit/stations/stochastic/verdicts.json").read_text()
)["counts"]
m = re.search(r"(\d+) attacker candidates, (\d+) defended, (\d+) confirmed, (\d+) dismissed", body)
if not m:
    bad.append("QUOTED-COUNT: section states no 'N attacker candidates …' line")
else:
    got = tuple(int(x) for x in m.groups())
    want = (verdicts["received"], verdicts["defended"], verdicts["confirmed"], verdicts["dismissed"])
    if got != want:
        bad.append(f"QUOTED-COUNT: section says {got}, verdicts.json says {want}")
    else:
        print(f"quoted counts OK: section {got} == verdicts.json {want}")

# 5. Perf handoff completeness: every entry names claimType/layout/profiledSymbol/anchor
# and asserts no timing (measured is false); measurement belongs to the perf-sweep epic.
perf = json.loads(
    (root / "plans/architecture-debt-audit/stations/stochastic/perf-queue.json").read_text()
)
for e in perf["queue"]:
    missing = [f for f in ("claimType", "layout", "profiledSymbol", "anchor") if not e.get(f)]
    if missing:
        bad.append(f"PERF {e.get('id', '?')}: missing {missing}")
    if e.get("measured") is not False or e.get("assertedTiming") is not None:
        bad.append(f"PERF {e.get('id', '?')}: a timing number is asserted in a read-only station")
if not any(b.startswith("PERF") for b in bad):
    print(f"perf handoff OK: {len(perf['queue'])} entries complete, no timing asserted")

# 6. Read-only workspace: no NEW tracked modification under an evaluated surface (the
# carried-in .gitignore and the station's own plans/ writes are exempt).
porcelain = subprocess.run(
    ["git", "status", "--porcelain", "--untracked-files=no"],
    cwd=root, capture_output=True, text=True, check=True,
).stdout.splitlines()
roots = ("crates/", "docs/", "scripts/", "schemas/", ".github/")
files = ("Cargo.toml", "Cargo.lock")
offenders = []
for line in porcelain:
    path = line[3:].split(" -> ")[-1].strip()
    if path == ".gitignore":
        continue
    if path in files or any(path.startswith(r) for r in roots):
        offenders.append(line.strip())
if offenders:
    bad.append("READ-ONLY: station modified an evaluated surface: " + "; ".join(offenders))
else:
    print("read-only OK: no tracked modification under an evaluated surface")

if bad:
    print("\nFAIL:")
    for b in bad:
        print(f"  - {b}")
    sys.exit(1)
print("\nverify-figures.sh: all checks passed")
PY

exit $FAIL
