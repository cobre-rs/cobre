#!/usr/bin/env python3
"""Station driver for cobre-sddp: the shared verifier plus the assertions it cannot know.

    python3 plans/architecture-debt-audit/stations/sddp/verify-sddp.py "<section title or slug>" [--no-shared]

argv[1] must resolve to exactly one heading of BACKLOG.md under backlog_parse.find_section's rule
(the exact title, or the station slug after the last em dash); zero or several matches exit
non-zero naming the count instead of passing on an empty section. The driver then runs
tools/verify-station.sh sddp "<title>" verbatim — the seven shared checks and the station test
module, which render the head of verification.md — folds its exit code and per-check results into
the result table, and appends, idempotently, the station-specific rows:

1. partition — every module under crates/cobre-sddp/src is claimed by exactly one of the four
   sub-stations of inventory.json; membership is counted as a multiset over the file records and
   the sub-station roll-ups, and doubly-assigned, unassigned and phantom paths are three separate
   lists, so a double-walk cannot hide behind a correct union;
2. dispositions — wave-dispositions.json carries exactly the 22 owned rows (19 prior ids plus the
   three retire items), one disposition each from keep|retire|sharpen, every anchor resolved through
   check-anchors.py at the station baseline, every retire commit resolved by git rev-parse with
   CD-001 and the CD-003 Construction hop pinned literally, every sharpen carrying both claims, and
   the positive polarity that the three retired items and PD-004 appear only as disposition-table
   rows, queue markers or Re-raise-of justifications — never as live entries;
3. lp-reconciliation — measurements/lp-inventory.json against the lp/ tree, drift reported per
   direction, totals and per-record invariants;
4. no-timing — the section body carries no duration, throughput or speedup literal, every PD entry
   is UNMEASURED with a layout tag, and every queued PD entry carries the perf-queue marker;
5. read-only-snapshot — `git status --porcelain` minus the pre-station snapshot names no new
   modification under an evaluated surface, and `git diff --stat HEAD -- crates docs schemas
   scripts` is empty outright.

--no-shared skips the shared verifier and the report write and prints the block to stdout: the
station test module invokes this script, and verify-station.sh runs the test module, so the flag
is what cuts the recursion.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUDIT = ROOT / "plans/architecture-debt-audit"
sys.path.insert(0, str(AUDIT / "tools"))

from lib import backlog_parse as bp  # noqa: E402

SLUG = "sddp"
REL = "plans/architecture-debt-audit"
USAGE = f'python3 {REL}/stations/{SLUG}/verify-{SLUG}.py "<section title or slug>" [--no-shared]'
SRC = "crates/cobre-sddp/src"
LP = "crates/cobre-sddp/src/lp"
PD_004_FILE = "crates/cobre-sddp/src/training/backward_pass_state.rs"
EXPECTED_SRC_FILES = 163
EXPECTED_LP_FILES = 30
OWNED = {
    4: ("CD-004", "CD-005", "CD-024-successor"),
    6: ("CD-015", "CD-022", "CD-018", "CD-035", "CD-034", "CD-023"),
    7: (
        "CD-007",
        "CD-012",
        "CD-014-remnant",
        "CD-016",
        "CD-021",
        "CD-028",
        "CD-030",
        "CD-037",
        "CD-038",
        "OD-009",
    ),
}
RETIRED = ("CD-001", "CD-003-construction-hop", "CD-006")
RETIRE_PINS = {"CD-001": "b051c410", "CD-003-construction-hop": "4075c4e8"}
EXPECTED_IDS = frozenset(i for ids in OWNED.values() for i in ids) | frozenset(RETIRED)
DISPOSITIONS = frozenset({"keep", "retire", "sharpen"})
LAYOUTS = frozenset({"4t", "2x2"})
# Register ids as an entry heading would spell them; the hop's suffix is a disposition-table id.
DO_NOT_RERAISE = ("CD-001", "CD-003", "CD-006", "PD-004")
PROBE_TITLE = "DISPOSITION ANCHOR PROBE — sddp (verify)"
MARKER = "## Station-specific checks — sddp"
SNAPSHOT = HERE / "tree-baseline.porcelain"
CARRIED_IN = frozenset({".gitignore"})
SURFACE_RE = re.compile(
    r"^(crates|scripts|schemas|\.github|Cargo\.(?:toml|lock)|docs|examples|tests)(/|$)"
)
DIFF_STAT_ROOTS = ("crates", "docs", "schemas", "scripts")
STATION_CHECKS = (
    "partition",
    "dispositions",
    "lp-reconciliation",
    "no-timing",
    "read-only-snapshot",
)

# A figure is a number not glued to an identifier, SHA, line tag or range (`L1727`, `077dbe2c`,
# `PD-035`, `-n 1/2`), followed by a time or throughput unit or a speed word — so `163`, `30`,
# `2x2`, `4t`, `--threads 4` and `19 × B` never fire.
_NUM = r"(?<![\w.\-/])\d+(?:\.\d+)?"
DURATION_RE = re.compile(
    _NUM
    + r"\s?(?:ns|nsecs?|µs|us|usecs?|ms|msecs?|s|secs?|seconds?|mins?|minutes?|h|hrs?|hours?)\b"
)
THROUGHPUT_RE = re.compile(
    _NUM
    + r"\s?(?:[KMGT]i?B|LPs?|solves?|iter(?:ation)?s?|ops|cuts|scenarios|passes|nodes|rows)"
    r"\s?(?:/|per)\s?(?:s|sec|seconds?|min|minutes?|h|hours?)\b"
    r"|\b(?:[KMGT]i?B|LPs?|solves)/s\b",
    re.I,
)
SPEEDUP_RE = re.compile(
    _NUM
    + r"\s?(?:x|×)\s?(?:faster|slower|speed-?ups?)\b"
    + r"|"
    + _NUM
    + r"\s?%\s?(?:faster|slower|speed-?up|regression|improvement|of (?:the )?(?:wall|run ?time|solve time))"
    + r"|\bspeed-?up of "
    + _NUM,
    re.I,
)
TIMING_RES = (
    ("duration", DURATION_RE),
    ("throughput", THROUGHPUT_RE),
    ("speedup", SPEEDUP_RE),
)

SHARED_ROW_RE = re.compile(
    r"^\| (?P<n>\d+) \| (?P<label>[^|]+?) \| `(?P<cmd>[^`]+)` \| (?P<code>\d+) \| (?P<res>PASS|FAIL) \|$",
    re.M,
)
TEST_LINE_RE = re.compile(
    r"^Test suite \(`[^`]+`\): (?P<res>PASS|FAIL) \(exit (?P<code>\d+)\)$", re.M
)


class Row:
    def __init__(
        self, label: str, command: str, code: int | None, detail: str = ""
    ) -> None:
        self.label, self.command, self.code, self.detail = label, command, code, detail

    @property
    def ok(self) -> bool:
        return self.code in (0, None)


def run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd), cwd=ROOT, capture_output=True, text=True, check=False
    )


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def find_rs(rel: str) -> list[str]:
    return sorted(run(["find", rel, "-name", "*.rs"]).stdout.split())


def matching_headings(lines: Sequence[str], name: str) -> list[tuple[int, str]]:
    """Every heading find_section would accept for `name`: exact title or em-dash tail."""
    want = name.strip().casefold()
    hits: list[tuple[int, str]] = []
    for lineno, raw in enumerate(lines, 1):
        m = bp.SECTION_RE.match(raw)
        if not m:
            continue
        title = m.group("title")
        tail = title.rsplit("—", 1)[-1].strip().casefold()
        if want in (title.strip().casefold(), tail):
            hits.append((lineno, title))
    return hits


def subsection(section_text: str, heading: str) -> str:
    if heading not in section_text:
        return ""
    return section_text.split(heading, 1)[1].split("\n#### ", 1)[0]


def run_shared(title: str) -> tuple[Row, list[str]]:
    display = f"bash {REL}/tools/verify-station.sh {SLUG} '{title}'"
    proc = run(["bash", str(AUDIT / "tools/verify-station.sh"), SLUG, title])
    report_path = HERE / "verification.md"
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    checks = [(m["label"], int(m["code"])) for m in SHARED_ROW_RE.finditer(report)]
    test = TEST_LINE_RE.search(report)
    if test:
        checks.append(("test-suite", int(test["code"])))
    problems: list[str] = []
    if proc.returncode != 0:
        tail = (proc.stdout.strip() or proc.stderr.strip()).splitlines()[-3:]
        problems.append(
            f"verify-station.sh exited {proc.returncode}: {' / '.join(tail)}"
        )
    if len(checks) != 8:
        problems.append(
            f"verify-station.sh recorded {len(checks)} results in verification.md, expected 7 checks + the test suite"
        )
    problems += [
        f"shared check {label} exited {code}" for label, code in checks if code
    ]
    code = proc.returncode if proc.returncode or not problems else 1
    detail = "folded per-check exits: " + ", ".join(
        f"{label} {code}" for label, code in checks
    )
    return Row("shared-verifier", f"`{display}`", code, detail), problems


def module_of(path: str) -> str:
    return path[len(SRC) + 1 :].split("/", 1)[0]


def partition_lists(
    inv: dict[str, Any], tree: Sequence[str]
) -> tuple[list[str], list[str], list[str]]:
    """(doubly-assigned, unassigned, phantom); membership is a multiset over records and roll-ups."""
    module_claims: dict[str, set[str]] = {}
    for sub, roll in inv["substation_rollup"].items():
        for mod in roll["modules"]:
            module_claims.setdefault(mod, set()).add(sub)
    claims: dict[str, list[str]] = {}
    for rec in inv["src"]["files"]:
        subs = claims.setdefault(rec["path"], [])
        subs.append(rec["substation"])
        subs.extend(
            sorted(
                module_claims.get(module_of(rec["path"]), set()) - {rec["substation"]}
            )
        )
    doubly = [
        f"{path} <- "
        + ", ".join(f"{s} x{n}" if n > 1 else s for s, n in Counter(subs).items())
        for path, subs in sorted(claims.items())
        if len(subs) > 1
    ]
    unassigned = sorted(set(tree) - claims.keys())
    phantom = sorted(claims.keys() - set(tree))
    return doubly, unassigned, phantom


def check_partition(inv: dict[str, Any], tree: Sequence[str]) -> tuple[list[str], str]:
    problems: list[str] = []
    doubly, unassigned, phantom = partition_lists(inv, tree)
    problems += [f"doubly-assigned: {d}" for d in doubly]
    problems += [
        f"unassigned (in the tree, in no sub-station): {p}" for p in unassigned
    ]
    problems += [f"phantom (listed, not in the tree): {p}" for p in phantom]
    counts = Counter(rec["substation"] for rec in inv["src"]["files"])
    for sub, roll in sorted(inv["substation_rollup"].items()):
        if roll["file_count"] != counts[sub]:
            problems.append(
                f"roll-up {sub} says {roll['file_count']} files, {counts[sub]} records carry it"
            )
    if inv["src"]["file_count"] != len(tree):
        problems.append(
            f"inventory file_count {inv['src']['file_count']} != {len(tree)} .rs files under {SRC}"
        )
    if len(tree) != EXPECTED_SRC_FILES:
        problems.append(
            f"{len(tree)} .rs files under {SRC}, the ticket pins {EXPECTED_SRC_FILES} at the baseline"
        )
    detail = (
        f"{len(tree)} modules, each claimed once: "
        + " / ".join(f"{sub} {counts[sub]}" for sub in sorted(counts))
        + f"; doubly-assigned {len(doubly)}, unassigned {len(unassigned)}, phantom {len(phantom)}"
    )
    return problems, detail


def render_stub(title: str, rows: Sequence[dict[str, Any]], baseline: str) -> str:
    out = [f"## {title}", ""]
    for d in rows:
        rid = d["id"].split("-construction-hop")[0].split("-successor")[0]
        rid = rid.split("-remnant")[0]
        claim = d.get("survivingClaim") or d.get("priorTitle") or d["id"]
        a = d["baselineAnchor"]
        out += [
            f"**{rid} · probe · {d['id']}**",
            "",
            " ".join(str(claim).split()),
            "",
            f"- **Anchors:** `{a['path']}::{a['symbol']}`",
            f"- **Baseline:** `{baseline}`",
            "",
        ]
    return "\n".join(out) + "\n"


def anchors_through_checker(
    rows: Sequence[dict[str, Any]], baseline: str
) -> tuple[int, list[str]]:
    with tempfile.TemporaryDirectory(prefix="sddp-verify.") as tmp:
        stub = pathlib.Path(tmp) / "disposition-probe.md"
        stub.write_text(render_stub(PROBE_TITLE, rows, baseline), encoding="utf-8")
        proc = run(
            [
                sys.executable,
                str(AUDIT / "tools/check-anchors.py"),
                PROBE_TITLE,
                "--register",
                str(stub),
                "--baseline",
                baseline,
                "--json",
            ]
        )
    try:
        report = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except json.JSONDecodeError:
        report = {}
    problems = [
        f"anchor-missing[{f['kind']}] {f['anchor']} ({f['entry']})"
        for f in report.get("failures", [])
    ]
    if proc.returncode and not problems:
        problems.append(
            f"check-anchors.py exited {proc.returncode} over the disposition probe: {proc.stderr.strip()[:200]}"
        )
    return int(report.get("checked", 0)), problems


def column(row: dict[str, str], prefix: str) -> str:
    return next((v for k, v in row.items() if k.startswith(prefix)), "")


def check_dispositions(
    env: dict[str, Any],
    baseline: str,
    section: bp.Section,
    entries: Sequence[bp.Entry],
) -> tuple[list[str], str]:
    problems: list[str] = []
    rows = env["dispositions"]
    ids = [r["id"] for r in rows]
    if len(rows) != len(EXPECTED_IDS):
        problems.append(f"{len(rows)} dispositions, expected {len(EXPECTED_IDS)}")
    for rid, n in sorted(Counter(ids).items()):
        if n > 1:
            problems.append(f"duplicated disposition id {rid} ({n} rows)")
    problems += [
        f"missing disposition: {rid}" for rid in sorted(EXPECTED_IDS - set(ids))
    ]
    problems += [f"extra disposition: {rid}" for rid in sorted(set(ids) - EXPECTED_IDS)]
    resolved: dict[str, str] = {}
    for r in rows:
        rid, d = r["id"], r.get("disposition")
        if d not in DISPOSITIONS:
            problems.append(f"{rid}: disposition {d!r} not in keep|retire|sharpen")
        anchor = r.get("baselineAnchor") or {}
        if not anchor.get("path") or not anchor.get("symbol"):
            problems.append(f"{rid}: baselineAnchor lacks path::symbol")
        if d == "retire":
            sha = r.get("resolvingCommit")
            if not sha:
                problems.append(f"{rid}: retire without resolvingCommit")
            else:
                rp = run(["git", "rev-parse", "--verify", "-q", f"{sha}^{{commit}}"])
                if rp.returncode:
                    problems.append(f"{rid}: resolvingCommit {sha} does not resolve")
                else:
                    resolved[rid] = rp.stdout.strip()
                pin = RETIRE_PINS.get(rid)
                if pin and sha != pin:
                    problems.append(
                        f"{rid}: resolvingCommit {sha}, the ticket pins {pin}"
                    )
        elif d == "sharpen":
            if not str(r.get("supersededClaim") or "").strip():
                problems.append(f"{rid}: sharpen without supersededClaim")
            if not str(r.get("survivingClaim") or "").strip():
                problems.append(f"{rid}: sharpen without survivingClaim")
    existence = env.get("existenceChecks", [])
    checked, anchor_problems = anchors_through_checker([*rows, *existence], baseline)
    problems += anchor_problems

    live = {e.id for e in entries}
    for rid in DO_NOT_RERAISE:
        if rid in live:
            problems.append(f"{rid} surfaces as a live entry")
    for e in entries:
        for raw in bp.finding_lines(e.body):
            f = bp.FIELD_RE.match(raw)
            if f and f.group("label").strip() == "Re-raise-of":
                continue
            for rid in DO_NOT_RERAISE:
                if re.search(rf"\b{rid}\b", raw):
                    problems.append(
                        f"{rid} named inside live entry {e.id} outside a Re-raise-of bullet"
                    )
    table = next(
        (
            t
            for t in bp.parse_tables(section)
            if t and "ID" in t[0] and "Disposition" in t[0]
        ),
        None,
    )
    if table is None:
        problems.append(
            "no disposition table with ID and Disposition columns in the section"
        )
    else:
        by_id: dict[str, list[dict[str, str]]] = {}
        for row in table:
            by_id.setdefault(row["ID"], []).append(row)
        for r in rows:
            trs = by_id.get(r["id"], [])
            if len(trs) != 1:
                problems.append(
                    f"{r['id']}: {len(trs)} disposition-table rows, expected one"
                )
            elif trs[0]["Disposition"] != r["disposition"]:
                problems.append(
                    f"{r['id']}: table says {trs[0]['Disposition']!r}, wave-dispositions.json says {r['disposition']!r}"
                )
        for rid, pin in RETIRE_PINS.items():
            trs = by_id.get(rid, [])
            if len(trs) == 1 and pin not in column(trs[0], "Superseded"):
                problems.append(f"{rid}: disposition-table row does not cite {pin}")
        pd_rows = by_id.get("PD-004", [])
        if len(pd_rows) != 1:
            problems.append(
                f"PD-004: {len(pd_rows)} disposition-table rows, expected one"
            )
        elif not pd_rows[0]["Disposition"].startswith(
            "existence-only"
        ) or PD_004_FILE not in column(pd_rows[0], "Evidence anchor"):
            problems.append(
                f"PD-004: table row is not an existence-only marker at {PD_004_FILE}"
            )
    pd = next((x for x in existence if x["id"] == "PD-004"), None)
    if pd is None or pd["baselineAnchor"]["path"] != PD_004_FILE:
        problems.append(
            f"PD-004 existence check missing or not anchored at {PD_004_FILE}"
        )
    if not (ROOT / PD_004_FILE).is_file():
        problems.append(f"{PD_004_FILE} does not exist")
    if "PD-004" not in subsection("\n".join(section.lines), "#### Queued out"):
        problems.append("PD-004 is not named in the Queued out subsection")
    by_disp = Counter(r.get("disposition") for r in rows)
    detail = (
        f"{len(rows)} rows: {by_disp['keep']} keep / {by_disp['retire']} retire / {by_disp['sharpen']} sharpen; "
        f"{checked} anchors resolved through check-anchors.py at {baseline[:8]}; retire commits "
        + ", ".join(f"{rid} -> {sha[:8]}" for rid, sha in sorted(resolved.items()))
        + f"; live entries {len(entries)}, none headed by "
        + "/".join(DO_NOT_RERAISE)
    )
    return problems, detail


def check_lp(lp: dict[str, Any], tree: Sequence[str]) -> tuple[list[str], str]:
    problems: list[str] = []
    listed = [f["path"] for f in lp["files"]]
    for path, n in sorted(Counter(listed).items()):
        if n > 1:
            problems.append(f"listed {n} times: {path}")
    problems += [
        f"listed but absent from the tree: {p}" for p in sorted(set(listed) - set(tree))
    ]
    problems += [
        f"present in the tree but unlisted: {p}"
        for p in sorted(set(tree) - set(listed))
    ]
    totals = lp["totals"]
    if totals["file_count"] != len(listed) or totals["file_count"] != len(tree):
        problems.append(
            f"totals.file_count {totals['file_count']} vs {len(listed)} records and {len(tree)} .rs files under {LP}"
        )
    if totals["file_count"] != EXPECTED_LP_FILES:
        problems.append(
            f"totals.file_count {totals['file_count']}, the ticket pins {EXPECTED_LP_FILES} at the baseline"
        )
    for f in lp["files"]:
        syms = f.get("top_symbols")
        if not isinstance(syms, list) or not syms:
            problems.append(f"{f['path']}: empty top_symbols")
        if f["non_test_lines"] > f["total_lines"]:
            problems.append(
                f"{f['path']}: non_test_lines {f['non_test_lines']} > total_lines {f['total_lines']}"
            )
        target = ROOT / f["path"]
        if target.is_file():
            wc = target.read_bytes().count(b"\n")
            if wc != f["total_lines"]:
                problems.append(
                    f"{f['path']}: total_lines {f['total_lines']} != wc -l {wc}"
                )
    for key in ("total_lines", "non_test_lines"):
        if totals[key] != sum(f[key] for f in lp["files"]):
            problems.append(f"totals.{key} {totals[key]} != sum over records")
    detail = (
        f"{len(tree)} lp/ modules listed once each; totals {totals['total_lines']} lines / "
        f"{totals['non_test_lines']} non-test; every record carries top_symbols and non_test_lines <= total_lines == wc -l"
    )
    return problems, detail


def timing_hits(text: str) -> list[tuple[str, str]]:
    return [(kind, m.group(0)) for kind, rx in TIMING_RES for m in rx.finditer(text)]


def check_timing(
    section: bp.Section, entries: Sequence[bp.Entry], perf: dict[str, Any]
) -> tuple[list[str], str]:
    problems: list[str] = []
    for offset, raw in enumerate(section.lines):
        for kind, literal in timing_hits(raw):
            problems.append(
                f"{kind} literal {literal!r} at BACKLOG.md:{section.start + 2 + offset}"
            )
    claim_rows = [r for r in perf["queue"] if r.get("severity")]
    queued_ids = [r["id"] for r in claim_rows]
    for r in perf["queue"]:
        if r.get("status") != "UNMEASURED" or r.get("measured"):
            problems.append(f"perf-queue {r['id']}: not UNMEASURED")
    for r in claim_rows:
        if r.get("layout") not in LAYOUTS:
            problems.append(
                f"perf-queue {r['id']}: layout {r.get('layout')!r} not in 4t|2x2"
            )
    pd_entries = [e for e in entries if e.id.startswith("PD-")]
    if not pd_entries:
        problems.append("the section carries no PD entry")
    for e in pd_entries:
        m = e.fields.get("Measurement", "")
        if "UNMEASURED" not in m:
            problems.append(f"{e.id}: Measurement bullet lacks the UNMEASURED tag")
        lay = re.search(r"layout `([^`]+)`", m)
        if lay is None or lay.group(1) not in LAYOUTS:
            problems.append(
                f"{e.id}: Measurement bullet lacks a layout tag from 4t|2x2"
            )
        if e.id in queued_ids and "perf-queue.json" not in m:
            problems.append(
                f"{e.id}: queued, but its Measurement bullet carries no perf-queue.json marker"
            )
    live_pd = {e.id for e in pd_entries}
    problems += [
        f"perf-queue {q}: no PD entry in the section"
        for q in queued_ids
        if q not in live_pd
    ]
    queued = subsection("\n".join(section.lines), "#### Queued out")
    problems += [
        f"{q}: not named in the Queued out subsection"
        for q in queued_ids
        if q not in queued
    ]
    detail = (
        f"{len(section.lines)} section lines scanned by {len(TIMING_RES)} unit-anchored regexes, 0 figures; "
        f"{len(pd_entries)} PD entries UNMEASURED with layout in 4t|2x2; {len(queued_ids)} queued to perf-queue.json, "
        f"plus the PD-004 existence row"
    )
    return problems, detail


def check_read_only() -> tuple[list[str], str]:
    problems: list[str] = []
    porcelain = run(["git", "status", "--porcelain"]).stdout.splitlines()
    snapshot = (
        SNAPSHOT.read_text(encoding="utf-8").splitlines() if SNAPSHOT.exists() else []
    )
    for line in porcelain:
        if line in snapshot:
            continue
        path = line[3:].split(" -> ")[-1].strip()
        if path in CARRIED_IN or not SURFACE_RE.match(path):
            continue
        problems.append(
            f"NEW modified tracked path under an evaluated surface: {line.strip()}"
        )
    diff = run(["git", "diff", "--stat", "HEAD", "--", *DIFF_STAT_ROOTS]).stdout
    if diff.strip():
        problems.append(
            f"git diff --stat HEAD -- {' '.join(DIFF_STAT_ROOTS)} is not empty: {' / '.join(diff.strip().splitlines())}"
        )
    snap = (
        f"snapshot `{REL}/stations/{SLUG}/tree-baseline.porcelain` ({len(snapshot)} lines)"
        if SNAPSHOT.exists()
        else "no pre-station porcelain snapshot on disk (the tree was clean when the station opened)"
    )
    detail = (
        f"{snap}; carried-in exemption `.gitignore`; porcelain lines outside the snapshot are the station's own plans/ writes; "
        f"`git diff --stat HEAD -- {' '.join(DIFF_STAT_ROOTS)}` empty"
    )
    return problems, detail


def render(rows: Sequence[Row], failures: Sequence[str]) -> str:
    out = [
        MARKER,
        "",
        "| Check | Command | Exit | Result |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        exit_cell = "-" if r.code is None else str(r.code)
        result = "SKIP" if r.code is None else ("PASS" if r.ok else "FAIL")
        out.append(f"| {r.label} | {r.command} | {exit_cell} | {result} |")
    out.append("")
    out += [f"- {r.label}: {r.detail}" for r in rows if r.detail]
    out.append(
        "- the perf lens is queue-only at this station: every figure is owned by the performance sweep (`perf-queue.json`), none is asserted in the register."
    )
    if failures:
        out += ["", "Failures:", *(f"- {f}" for f in failures)]
    out.append("")
    return "\n".join(out)


def main(argv: Sequence[str]) -> int:
    flags = {a for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 1 or flags - {"--no-shared"}:
        print(USAGE, file=sys.stderr)
        return 2
    title = args[0]
    lines = bp.read_register(AUDIT / "BACKLOG.md")
    hits = matching_headings(lines, title)
    if len(hits) != 1:
        where = "; ".join(f"L{n}: {t}" for n, t in hits) or "none"
        print(
            f"FAIL section-title: {title!r} matches {len(hits)} headings in BACKLOG.md (expected exactly 1): {where}",
            file=sys.stderr,
        )
        return 2
    section = bp.find_section(lines, title)
    entries = bp.iter_entries(section)
    if not entries:
        print(
            f"FAIL section-title: {section.heading!r} carries no entries — an empty section does not pass vacuously",
            file=sys.stderr,
        )
        return 2

    inv = load("inventory.json")
    baseline = str(inv["baseline"])
    rows: list[Row] = []
    failures: list[str] = []
    shared_display = f"`bash {REL}/tools/verify-station.sh {SLUG} '{title}'`"
    run_the_shared = "--no-shared" not in flags
    if run_the_shared:
        row, p = run_shared(title)
        rows.append(row)
        failures += p
    else:
        rows.append(
            Row(
                "shared-verifier",
                shared_display,
                None,
                "skipped (--no-shared: the station test module invokes this script from inside the shared verifier)",
            )
        )

    p, d = check_partition(inv, find_rs(SRC))
    failures += p
    rows.append(
        Row(
            "partition",
            f"inventory.json sub-stations 5a-5d as a multiset vs `find {SRC} -name '*.rs'`",
            1 if p else 0,
            d,
        )
    )
    p, d = check_dispositions(
        load("wave-dispositions.json"), baseline, section, entries
    )
    failures += p
    rows.append(
        Row(
            "dispositions",
            f"wave-dispositions.json (22 rows) through `python3 {REL}/tools/check-anchors.py '{PROBE_TITLE}' --register <stub> --baseline {baseline[:8]} --json` and `git rev-parse --verify <sha>^{{commit}}`; polarity over the parsed section",
            1 if p else 0,
            d,
        )
    )
    p, d = check_lp(
        json.loads(
            (AUDIT / "measurements/lp-inventory.json").read_text(encoding="utf-8")
        ),
        find_rs(LP),
    )
    failures += p
    rows.append(
        Row(
            "lp-reconciliation",
            f"measurements/lp-inventory.json vs `find {LP} -name '*.rs'`",
            1 if p else 0,
            d,
        )
    )
    p, d = check_timing(section, entries, load("perf-queue.json"))
    failures += p
    rows.append(
        Row(
            "no-timing",
            "unit-anchored duration/throughput/speedup regexes over the section; PD entries' Measurement bullets; perf-queue.json layouts",
            1 if p else 0,
            d,
        )
    )
    p, d = check_read_only()
    failures += p
    rows.append(
        Row(
            "read-only-snapshot",
            f"`git status --porcelain` minus the pre-station snapshot, filtered to the evaluated surfaces; `git diff --stat HEAD -- {' '.join(DIFF_STAT_ROOTS)}`",
            1 if p else 0,
            d,
        )
    )

    block = render(rows, failures)
    if run_the_shared:
        target = HERE / "verification.md"
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
        head = existing.split(MARKER, 1)[0].rstrip() + "\n\n" if existing else ""
        target.write_text(head + block, encoding="utf-8")
    else:
        sys.stdout.write(block)
    for f in failures:
        print(f"FAIL {f}", file=sys.stderr)
    station_rows = [r for r in rows if r.label in STATION_CHECKS]
    shared = rows[0]
    shared_note = (
        "shared verifier skipped"
        if shared.code is None
        else f"shared verifier exit {shared.code}"
    )
    print(
        f"verify-sddp.py {SLUG}: {'PASS' if not failures else 'FAIL'} "
        f"({sum(r.ok for r in station_rows)}/{len(station_rows)} station-specific checks; {shared_note})"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
