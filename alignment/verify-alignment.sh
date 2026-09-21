#!/usr/bin/env bash
# verify-alignment.sh - verification for the generalization-alignment epic.
# Verifies the Part-I dispositions, the lp/ classification and the Alignment
# ledger. Deliberately does NOT call tools/verify-station.sh: that verifier is
# keyed on a station slug + stations/<slug>/inventory.json, and this epic has neither.
#
# Usage: verify-alignment.sh [--report PATH]
#   --report PATH   also render the result table (verification.md) from this run
#   VERIFY_ALIGNMENT_NO_TESTS=1   skip the test-module block (the module itself runs
#                                 this script, so the module invokes it with the flag)
# Exit: 0 every block passed, 1 at least one block failed, 2 cannot locate the repo.
# shellcheck disable=SC2317  # the check functions are invoked indirectly through block()
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)" || exit 2
cd "$ROOT" || exit 2
ALIGN=plans/architecture-debt-audit/alignment
TOOLS=plans/architecture-debt-audit/tools
REGISTER=plans/architecture-debt-audit/BACKLOG.md
# The epic's block lives under the register scaffold's own heading, minted at the register
# pin; the checkers resolve a section by its em-dash tail. (The tickets quote a
# '★ GENERALIZATION ALIGNMENT (…)' title that never existed in the register.)
SECTION=generalization-alignment
TICKET_BASE=a136840d4f2ea137f685f0af6dac04254b983b60   # the scaffold pin the tickets quote
LP_ROOT=crates/cobre-sddp/src/lp
REPORT=""
if [ "${1:-}" = "--report" ]; then REPORT="${2:?--report needs a path}"; fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
RESULTS="$TMP/results.tsv"; : > "$RESULTS"
fail=0

# The register pin, read through the shared parser (never re-implemented here).
BASE="$(python3 - "$REGISTER" <<'PY'
import pathlib, sys
sys.path.insert(0, "plans/architecture-debt-audit/tools")
from lib import backlog_parse as bp
print(bp.parse_baseline(bp.read_register(pathlib.Path(sys.argv[1]))))
PY
)" || exit 2
echo "register pin ${BASE:0:8}; ticket scaffold pin ${TICKET_BASE:0:8}; section '— $SECTION'"

# block NAME CMD... : run CMD, echo its output, record PASS/FAIL with the command's last line.
block() {
  local name="$1"; shift
  printf '\n--- %s\n' "$name"
  if "$@" > "$TMP/out" 2>&1; then
    cat "$TMP/out"; printf '%s\t%s\tPASS\n' "$name" "$(tail -n1 "$TMP/out")" >> "$RESULTS"
  else
    cat "$TMP/out"; printf '%s\t%s\tFAIL\n' "$name" "$(tail -n1 "$TMP/out")" >> "$RESULTS"
    echo "FAIL: $name"; fail=1
  fi
}

# --- the three canonical harness checkers over the epic's section ------------------------
block "anchors" python3 "$TOOLS/check-anchors.py" "$SECTION"
block "re-raise" python3 "$TOOLS/check-reraise.py" "$SECTION"
block "fields (section)" python3 "$TOOLS/fields-check.py" --require Alignment "$SECTION"

# --- nine Part-I dispositions, one each, with the item-specific traps ---------------------
block "Part-I dispositions" python3 - "$ALIGN/part-i-dispositions.json" "$REGISTER" "$SECTION" "$BASE" "$TICKET_BASE" <<'PY'
import json, pathlib, re, subprocess, sys
sys.path.insert(0, "plans/architecture-debt-audit/tools")
from lib import backlog_parse as bp

env_path, register_path, section_tail, base, ticket_base = sys.argv[1:6]
env = json.loads(pathlib.Path(env_path).read_text(encoding="utf-8"))
recs = env["dispositions"]
lines = bp.read_register(pathlib.Path(register_path))
section = bp.find_section(lines, section_tail)
entries = {e.id: e for e in bp.iter_entries(section)}
register_text = "\n".join(lines)
DECL = r"^\s*(pub(\([^)]*\))?\s+)?(async\s+)?(fn|struct|enum|trait|type|const|static|mod|impl)\s+{sym}\b"

def show(path):
    p = subprocess.run(["git", "show", f"{base}:{path}"], capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None

def resolves(a):
    text = show(a["path"])
    if text is None:
        return False
    if a.get("symbol"):
        return re.search(DECL.format(sym=re.escape(a["symbol"])), text, re.M) is not None
    return a.get("line") is not None and 1 <= a["line"] <= text.count("\n") + 1

assert env["baseline"] == base, f"envelope baseline {env['baseline'][:8]} != register pin {base[:8]}"
want = {"1", "2", "3", "4", "5", "6", "7", "8", "I.5"}
got = [str(r["item"]) for r in recs]
assert recs, "vacuous: parsed zero dispositions"
assert set(got) == want and len(got) == 9, f"expected nine dispositions, got {sorted(got)}"
for r in recs:
    assert r["disposition"] in {"keep", "retire", "sharpen"}, f"item {r['item']}: disposition {r['disposition']!r}"
    assert r["fixShapePhase"] in {"0a", "0b", "1"}, f"item {r['item']}: phase {r['fixShapePhase']!r}"
    if r["disposition"] == "retire":
        assert r.get("closedBy"), f"item {r['item']}: retire without a closing version/commit"
        assert r.get("clearedLine") and register_text.count(r["clearedLine"]) == 1, f"item {r['item']}: no single Cleared line"
        assert r.get("registerId") is None, f"item {r['item']}: a retire must not mint an id"
    else:
        rid = r.get("registerId") or ""
        assert re.fullmatch(r"CD-\d{3}", rid), f"item {r['item']}: keep/sharpen without a CD id"
        assert rid in entries, f"{rid} (item {r['item']}) is not an entry of the section"
        assert register_text.count(f"**{rid} ·") == 1, f"{rid}: not exactly one entry heading in the register"
    if r["disposition"] == "sharpen":
        assert r.get("sharpenedClaim") and r.get("baselineAnchors"), f"item {r['item']}: sharpen without claim/anchors"
    for a in r["baselineAnchors"]:
        assert resolves(a), f"anchor-missing: item {r['item']} {a} does not resolve at {base[:8]}"

by = {str(r["item"]): r for r in recs}
for sha in (base, ticket_base):
    rc = subprocess.run(["git", "grep", "-q", "-e", "struct PolicyGraph", sha, "--", "crates/"]).returncode
    assert rc != 0, f"struct PolicyGraph resolves at {sha[:8]}: re-check item 3"
i3 = by["3"]
assert i3["disposition"] in {"sharpen", "retire"}, "item 3 cannot be kept on a dead anchor"
assert all(a.get("symbol") != "PolicyGraph" for a in i3["baselineAnchors"]), "anchor-missing: item 3 restates the dead PolicyGraph anchor"
a3 = " ".join(a["path"] for a in i3["baselineAnchors"])
assert "model/horizon.rs" in a3 or "model/temporal.rs" in a3, f"item 3 lacks a surviving anchor: {a3}"
assert re.search(r"HorizonGraph|PolicyGraphType", i3["sharpenedClaim"]), "item 3's sharpened claim names no surviving type"
i4 = by["4"]
assert "past_defluences" in i4["sharpenedClaim"] and "past_anticipated_commitments" in i4["sharpenedClaim"], "item 4 must cite both surviving warm-start fields"
assert "0.13" in json.dumps(i4), "item 4 must cite the v0.13.0 past_inflows removal"
i5 = by["5"]
assert i5["disposition"] in {"keep", "sharpen"} and i5["registerId"], "item 5 (training_event.rs) is owned by no station and must still be dispositioned"
assert any(a["path"].endswith("constraints/training_event.rs") for a in i5["baselineAnchors"]), "item 5 must anchor training_event.rs"
i6 = by["6"]
assert i6.get("reRaiseOf"), "item 6: the gate exemption is withdrawn at the baseline; the sharpened entry must carry reRaiseOf"
body6 = "\n".join(entries[i6["registerId"]].body)
assert "**Re-raise-of:**" in body6, f"{i6['registerId']} (item 6) lacks its Re-raise-of line"
gate = show("scripts/ci/check-infra-genericity.sh") or ""
assert re.search(r"^EXCLUDED_FILES=\(\)", gate, re.M), "the genericity-gate exemption is not withdrawn at the baseline"
i7 = by["7"]
assert i7["fixShapePhase"] == "0a", "item 7 must be a Phase-0a fix-shape"
assert any(a["path"].endswith("broadcast.rs") for a in i7["baselineAnchors"]), "item 7 must cite a broadcast.rs anchor"
i8 = by["8"]
assert i8["disposition"] == "keep" and i8["alignment"] == "advances-1", f"item 8: {i8['disposition']} / {i8['alignment']}"
assert any(a["path"] == "crates/cobre-solver/src/types.rs" for a in i8["baselineAnchors"]), "item 8 must anchor cobre-solver/src/types.rs"
i5r = by["I.5"]
assert i5r["fixShapePhase"] == "0a", "I.5 must be a Phase-0a fix-shape"
assert i5r.get("measurements"), "I.5 states no measured figures"
def sh(cmd):
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout
rerun = quoted = 0
for m in i5r["measurements"]:
    cmd = m["command"]
    if "<file>" in cmd:
        # the writer-name comparison: one templated command, run for both front ends at the pin
        sets = {}
        for path in ("crates/cobre-cli/src/commands/run/outputs.rs", "crates/cobre-python/src/run.rs"):
            sets[path] = {ln.rstrip("(") for ln in sh(cmd.replace("<file>", path)).split() if ln}
        cli, py = sets["crates/cobre-cli/src/commands/run/outputs.rs"], sets["crates/cobre-python/src/run.rs"]
        assert str(m["figure"]) == f"{len(cli)} vs {len(py)}", f"I.5 writer figure {m['figure']} does not reproduce: {len(cli)} vs {len(py)}"
        assert sorted(cli - py) == m["cliOnly"] and sorted(py - cli) == m["pythonOnly"] and sorted(cli & py) == m["shared"], "I.5 writer sets drifted"
        rerun += 1
    elif cmd.split(" ", 1)[0] in {"git", "grep", "find", "wc"}:
        out = sh(cmd).strip()
        assert out == str(m["figure"]), f"I.5 figure {m['figure']} ({m['unit']}) does not reproduce: command gives {out!r}"
        rerun += 1
    else:
        # a station-quoted figure: its provenance pointer must name a handoff that carries the figure
        handoff = pathlib.Path("plans/architecture-debt-audit/stations/cli-python/partI-handoff.json")
        assert "partI-handoff.json" in cmd and handoff.exists(), f"I.5 figure {m['figure']} has no runnable command and no handoff provenance"
        assert str(m["figure"]) in handoff.read_text(encoding="utf-8"), f"I.5 figure {m['figure']} is not in {handoff}"
        quoted += 1
assert rerun >= 8, f"I.5 re-ran only {rerun} commands"
n = int(sh("grep -rn 'use cobre_sddp' crates/cobre-cli/src | wc -l").strip())
assert any(str(n) == str(m["figure"]) for m in i5r["measurements"]), f"I.5 must state the working-tree import count ({n}) next to its command"
print(f"dispositions OK: 9 items ({sum(1 for r in recs if r['disposition']=='sharpen')} sharpen / "
      f"{sum(1 for r in recs if r['disposition']=='keep')} keep / {sum(1 for r in recs if r['disposition']=='retire')} retire); "
      f"I.5: {rerun} figures re-run from their commands, {quoted} station-quoted with handoff provenance, `use cobre_sddp` = {n} in the working tree")
PY

# --- lp/ classification completeness: rows vs find, LOC recomputed with the SAME slicer --
find "$LP_ROOT" -name '*.rs' | sort > "$TMP/find.txt"
git ls-tree -r --name-only "$BASE" -- "$LP_ROOT" | grep '\.rs$' | sort > "$TMP/tree.txt"
# shellcheck disable=SC2016  # the backticks are literal Markdown, not command substitution
grep -o '^| `crates/cobre-sddp/src/lp/[^`]*\.rs`' "$ALIGN/lp-classification.md" \
  | sed 's/^| `//; s/`$//' | sort -u > "$TMP/rows.txt"
lp_rows() {
  test "$(wc -l < "$TMP/find.txt")" -eq 30 || { echo "lp/ universe drifted from 30 modules: $(wc -l < "$TMP/find.txt")"; return 1; }
  diff -u "$TMP/tree.txt" "$TMP/find.txt" || { echo "working tree differs from the pin under $LP_ROOT"; return 1; }
  diff -u "$TMP/find.txt" "$TMP/rows.txt" || { echo "classification rows != find (missing-module / extra-row)"; return 1; }
  echo "rows OK: $(wc -l < "$TMP/rows.txt") classification rows == $(wc -l < "$TMP/find.txt") find paths == the tree at ${BASE:0:8}"
}
block "lp/ rows vs find" lp_rows

block "LOC recount" python3 - "$TMP/find.txt" "$ALIGN/lp-classification.json" "$ALIGN/lp-grep-proof.json" <<'PY'
import importlib.util, json, pathlib, sys
spec = importlib.util.spec_from_file_location("slicer", "plans/architecture-debt-audit/alignment/lp-nontest-slice.py")
slicer = importlib.util.module_from_spec(spec); spec.loader.exec_module(slicer)
cls = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
proof = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
rows = {r["module"]: r for r in cls["rows"]}
paths = pathlib.Path(sys.argv[1]).read_text().split()
assert rows and paths, "vacuous: zero rows or zero paths"
total = naive = gross = 0
for path in paths:
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    gross += len(lines)
    if slicer.is_sibling(path):
        loc = 0
    else:
        loc = len(slicer.nontest_slice(lines)[0])
    naive += 0 if slicer.is_sibling(path) else slicer.naive_first_marker_loc(lines)
    total += loc
    assert rows[path]["nonTestLoc"] == loc, f"{path}: recorded {rows[path]['nonTestLoc']}, recomputed {loc}"
    assert proof["modules"][path]["nonTestLoc"] == loc, f"{path}: grep-proof LOC drifted"
recorded = cls["totals"]["corpusNonTestLoc"]
assert total == recorded == proof["totals"]["nonTestLoc"], f"trap-aware non-test universe is {total}, recorded {recorded} / proof {proof['totals']['nonTestLoc']}"
assert naive == proof["totals"]["naiveFirstMarkerLoc"], f"naive total {naive} != proof {proof['totals']['naiveFirstMarkerLoc']}"
assert naive != total, "naive first-marker total equals the trap-aware total: the slicer did not run"
assert gross == proof["totals"]["grossLoc"], f"gross {gross} != proof {proof['totals']['grossLoc']}"
print(f"LOC OK: brace-aware {total} vs naive {naive} (differ) over {gross} gross lines; "
      f"the ticket's 11675 / 11396 / 44374 were quoted at the scaffold pin and are recorded as a deviation in lp-classification.json")
PY

# --- grep proof re-run, non-vacuous; classes, seams, dismissals; the measured share -------
block "grep proof" python3 - "$ALIGN/lp-classification.json" "$ALIGN/lp-grep-proof.json" "$ALIGN/lp-classification.md" <<'PY'
import importlib.util, json, pathlib, re, sys
spec = importlib.util.spec_from_file_location("slicer", "plans/architecture-debt-audit/alignment/lp-nontest-slice.py")
slicer = importlib.util.module_from_spec(spec); spec.loader.exec_module(slicer)
cls = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
proof = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))["modules"]
md = pathlib.Path(sys.argv[3]).read_text(encoding="utf-8")
rows = cls["rows"]
assert rows, "vacuous: parsed zero classification rows"
assert slicer.VOCAB == "state_space|cost_to_go|theta|cut|ring", slicer.VOCAB
hits_anywhere = neutral_loc = 0
for r in rows:
    assert r["class"] in {"engine-neutral", "sddp-geometry", "mixed", "test-sibling"}, r["module"]
    if r["class"] == "test-sibling":
        assert slicer.is_sibling(r["module"]) and r["nonTestLoc"] == 0, r["module"]
        continue
    kept, _ = slicer.nontest_slice(pathlib.Path(r["module"]).read_text(encoding="utf-8").splitlines())
    hits = slicer.vocabulary_hits(kept)
    sub = [n for n, _ in hits["substringHits"]]
    word = [n for n, _ in hits["wordBoundaryHits"]]
    hits_anywhere += len(sub)
    rec = proof[r["module"]]
    assert [h[0] for h in rec["substringHits"]] == sub and [h[0] for h in rec["wordBoundaryHits"]] == word, f"{r['module']}: grep proof drifted"
    assert r["grepPass"] == {"substring": len(sub), "wordBoundary": len(word)}, f"{r['module']}: grepPass drifted"
    if r["class"] == "mixed":
        assert r.get("splitNote") and r.get("neutralHalf"), f"{r['module']}: mixed without seam/range"
        if r["module"].endswith("/mod.rs"):
            # a module root has no function seam: the split is the set of `mod` / re-export lines
            assert re.search(r"`mod\b|re-export|pub use", r["splitNote"]), f"{r['module']}: module-root seam names no mod/re-export lines"
        else:
            assert r.get("partition") and r["partition"]["neutral"] and r["partition"]["geometry"], f"{r['module']}: mixed seam names no functions on one side"
        continue
    if r["module"].endswith("builder/patch.rs"):
        assert r["class"] == "sddp-geometry" and not sub and r.get("dismissal"), "patch.rs: zero-hit needs its CamelCase dismissal"
    if r["class"] != "engine-neutral":
        continue
    neutral_loc += r["nonTestLoc"]
    assert not word, f"grep-contradiction {r['module']}: word-boundary hits {word}"
    if sub:
        d = r.get("dismissal") or ""
        quoted = {int(n) for n in re.findall(r"line (\d+)", d)}
        assert quoted == set(sub), f"grep-contradiction {r['module']}: substring hits {sub} not argued away line-by-line (dismissal quotes {sorted(quoted)})"
assert hits_anywhere > 0, "vacuous pattern: no hit anywhere in the 30-module universe"
scaling = proof["crates/cobre-sddp/src/lp/builder/scaling.rs"]
assert scaling["substringHits"] and scaling["wordBoundaryHits"], "builder/scaling.rs no longer carries hits: the pattern is suspect"
rc = proof["crates/cobre-sddp/src/lp/indexer/range_cursor.rs"]
assert [h[0] for h in rc["substringHits"]] == [10] and not rc["wordBoundaryHits"], "range_cursor.rs:10 is no longer the argued substring-only hit"
for zero in ("indexer/entity_index.rs", "indexer/layout.rs", "indexer/storage_boundary_grid.rs", "indexer/study_dimensions.rs"):
    m = "crates/cobre-sddp/src/lp/" + zero
    assert not proof[m]["substringHits"] and not proof[m]["wordBoundaryHits"], f"{zero} is no longer zero-hit"
t = cls["totals"]
assert t["engineNeutralLoc"] == neutral_loc, f"engine-neutral LOC {neutral_loc} != recorded {t['engineNeutralLoc']}"
corpus = t["corpusNonTestLoc"]
lo, hi = round(corpus / 5), round(corpus / 4)
assert t["roadmapBand"] == [lo, hi], f"roadmap band {t['roadmapBand']} != fifth-to-a-quarter of {corpus} = {lo}-{hi}"
assert t["measuredBand"] == [neutral_loc + t["mixedNeutralHalf"][0], neutral_loc + t["mixedNeutralHalf"][1]], "measured band drifted"
assert re.search(r"(fifth|quarter|IV\.2)", md), "lp-classification.md states no share against the fifth-to-a-quarter estimate"
assert t["verdict"] in {"agreement", "amended"}, t["verdict"]
if t["verdict"] == "amended":
    assert re.search(r"\d{4}-\d{2}-\d{2}", t["amendedFigure"]) and "amended" in md, "amended verdict without a dated figure in the table"
print(f"grep proof OK: {hits_anywhere} substring hits across the universe; engine-neutral {neutral_loc} + mixed half "
      f"{t['mixedNeutralHalf'][0]}-{t['mixedNeutralHalf'][1]} = {t['measuredBand'][0]}-{t['measuredBand'][1]} of {corpus} "
      f"vs IV.2 band {lo}-{hi} -> {t['verdict']} (ticket band 2335-2919 over 11675 quoted at the scaffold pin)")
PY

# --- register-wide Alignment, ledger integrity, tests, read-only proof --------------------
block "Alignment presence" python3 "$TOOLS/fields-check.py" --require Alignment --all

block "ledger integrity" python3 - "$REGISTER" "$ALIGN/alignment-ledger.json" "$BASE" <<'PY'
import json, pathlib, sys
sys.path.insert(0, "plans/architecture-debt-audit/tools")
from lib.backlog_parse import ENTRY_RE, find_section, read_register   # shared parser; never re-implemented here
VOCAB = {"advances-0a", "advances-0b", "advances-1", "neutral", "conflicts"}
STATIONS = ["core-io", "stochastic", "solver-comm", "sddp", "cli-python", "build-ci", "test-corpus"]
lines = read_register(pathlib.Path(sys.argv[1]))
ids = [m.group("id") for st in STATIONS for raw in find_section(lines, st).lines for m in [ENTRY_RE.match(raw)] if m]
assert ids, "vacuous: parsed zero register entries across the station sections"
assert len(ids) == len(set(ids)), "duplicate entry id across the station sections"
led = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
assert led["baseline"] == sys.argv[3], f"ledger baseline {led['baseline'][:8]} != register pin {sys.argv[3][:8]}"
dec = {a["entryId"]: a for a in led["ledger"]}
assert set(dec) == set(ids), f"unadjudicated entries: {sorted(set(ids) - set(dec))}; stale: {sorted(set(dec) - set(ids))}"
held = []
for a in dec.values():
    assert a["decided"] in VOCAB, f"{a['entryId']}: decided {a['decided']!r}"
    assert len(a.get("rationale") or "") >= 40, f"{a['entryId']}: thin rationale"
    assert "Part IV" in (a.get("cites") or "") or "Part V" in (a.get("cites") or ""), f"{a['entryId']}: no Part IV/V citation"
    if a["decided"] == "conflicts":
        assert a.get("guardrailViolated"), f"{a['entryId']}: conflicts without a guardrail"
        assert a.get("alternativeFixShape") or a.get("needsHuman"), f"{a['entryId']}: conflicts-without-alternative"
        assert a.get("held"), f"{a['entryId']}: conflicts not held"
        held.append(a["entryId"])
    else:
        assert not a.get("held"), f"{a['entryId']}: held without conflicts"
assert sorted(led["held"]) == sorted(held), "ledger.held disagrees with the conflicts rows"
assert led["coverage"]["registerOnly"] == [] and led["coverage"]["calibrationOnly"] == [], led["coverage"]
assert led["verification"]["passed"] is True and led["selfCheck"]["passed"] is True, "the adjudication's own verification did not pass"
retagged = [r for r in led["ledger"] if r["retagged"]]
print(f"alignment ledger OK: {len(dec)} entries, {len(retagged)} retagged, {len(held)} held for owner override"
      + (f" ({', '.join(held)})" if held else ""))
PY

if [ "${VERIFY_ALIGNMENT_NO_TESTS:-0}" = "1" ]; then
  printf '%s\t%s\t%s\n' "alignment tests" "skipped: VERIFY_ALIGNMENT_NO_TESTS=1 (invoked from the test module itself)" "SKIP" >> "$RESULTS"
else
  block "alignment tests" python3 -m pytest -q -p no:cacheprovider "$ALIGN/tests/test_alignment.py"
fi

read_only() {
  # plans/ is tracked in this repository, so a committed epic leaves nothing in the
  # porcelain; the pre-audit ' M .gitignore' line the ticket anticipates is tolerated if present.
  git status --porcelain --untracked-files=no | grep -v '^ M \.gitignore$' > "$TMP/dirty" || true
  if [ -s "$TMP/dirty" ]; then echo "tree dirty beyond the pre-audit .gitignore edit:"; cat "$TMP/dirty"; return 1; fi
  git diff --quiet HEAD -- crates scripts .github schemas docs Cargo.toml || { echo "tracked source touched by a read-only epic"; return 1; }
  echo "read-only OK: porcelain empty (modulo .gitignore); git diff HEAD over crates scripts .github schemas docs Cargo.toml is empty"
}
block "read-only" read_only

# --- report ------------------------------------------------------------------------------
if [ -n "$REPORT" ]; then
  python3 - "$RESULTS" "$REPORT" "$BASE" "$TICKET_BASE" "$fail" <<'PY'
import pathlib, sys
rows = [ln.split("\t") for ln in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if ln]
base, ticket_base, failed = sys.argv[3], sys.argv[4], sys.argv[5] == "1"
out = [
    f"# Alignment verification (register pin {base[:8]}; the tickets' scaffold pin {ticket_base[:8]})",
    "",
    "Produced by `bash plans/architecture-debt-audit/alignment/verify-alignment.sh --report plans/architecture-debt-audit/alignment/verification.md`. "
    "The three harness checkers run over the register section `★ QUALITY EVALUATION (2026-09, baseline a136840d) — generalization-alignment` "
    "(the scaffold heading minted at the register pin; the tickets' `★ GENERALIZATION ALIGNMENT (…)` title never existed). "
    "Nothing is fixed; every write of the epic lives under plans/architecture-debt-audit/alignment/, which is tracked in this repository "
    "(the tickets' 'gitignored plan artifacts' wording is stale), so the read-only proof is that the committed tree carries no change "
    "outside plans/.",
    "",
    "| check | detail | result |",
    "| --- | --- | --- |",
]
for name, detail, status in rows:
    out.append(f"| {name} | {detail.replace('|', chr(92) + '|')} | {status} |")
held = [r for r in rows if r[0] == "ledger integrity"]
held_ids = held[0][1].split("(", 1)[1].rstrip(")") if held and "(" in held[0][1] else ""
out += [
    "",
    f"**Overall:** {'FAIL' if failed else 'PASS'}.",
    "",
    f"Held for the owner gate: {held_ids or 'none (no station entry violates a Part IV.1 guardrail; see alignment/conflicts-docket.md)'}.",
    "",
    "Figures the tickets quote at the scaffold pin and how they read at the register pin (each recorded as a deviation in the "
    "producing artifact, never edited into the spec): non-test universe 11,675 → 11,701 (the ticket's own rule does not reproduce "
    "its total; 11,685 at a136840d), naive first-marker 11,396 → 11,411, gross 44,374 → 44,682, IV.2 band 2,335-2,919 → 2,340-2,925; "
    "I.5 `use cobre_sddp` lines 79 → 81 and `cobre_sddp::` references 95 → 97 (the envelope states both pins' figures with their commands). "
    "The engine-neutral share is an amended figure (42-56% measured against the fifth-to-a-quarter estimate), dated 2026-09-19.",
]
pathlib.Path(sys.argv[2]).write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"report written: {sys.argv[2]}")
PY
fi

echo
if [ "$fail" -eq 0 ]; then echo "verify-alignment: PASS"; else echo "verify-alignment: FAIL"; fi
exit "$fail"
