#!/usr/bin/env python3
"""Roadmap DAG Check: verify the unified roadmap's ordering mechanically (checks 1-9).

Reads the 7-column roadmap table and the waved Milestones block through the
shared parser. Exit codes: 0 clean, 1 any check-2..9 violation (a cycle is
printed as an ordered closed path), 2 malformed table, 3 unknown milestone or
phase token. Read-only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from lib.backlog_parse import (  # noqa: E402
    FIELD_RE,
    Milestone,
    Section,
    SectionNotFound,
    find_section,
    iter_entries,
    all_evaluation_sections,
    parse_baseline,
    parse_milestones,
    parse_table,
    register_findings,
    repo_root,
)

COLUMNS = ["Wave", "Entry", "Findings", "Depends on", "Phase", "Effort", "Trigger/deadline"]
PHASES = ("serves 0a", "serves 0b", "serves 1", "neutral")
MILESTONES = ("0a", "0b", "1", "gnl-import")
PHASE_TO_MILESTONE = {"serves 0a": "0a", "serves 0b": "0b", "serves 1": "1"}
SCHEDULABLE = {"ratified", "kept", "sharpened", "deferred"}
RETIRED = {"cleared", "retracted", "refuted", "do-not-touch"}
EXIT_OK, EXIT_VIOLATION, EXIT_MALFORMED, EXIT_UNKNOWN_MILESTONE = 0, 1, 2, 3
DEFAULT_SECTION = "unified-roadmap"
DEFAULT_BACKLOG = repo_root() / "plans" / "architecture-debt-audit" / "BACKLOG.md"
OVERRIDE_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})\b")


class Malformed(Exception):
    def __init__(self, code: int, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.code, self.kind, self.detail = code, kind, detail


@dataclass(frozen=True)
class RoadmapRow:
    wave: int
    entry_id: str
    findings: tuple[str, ...]
    depends_on: tuple[str, ...]
    phase: str
    effort: str
    trigger: str
    line: int

    @classmethod
    def from_cells(cls, line: int, cells: list[str]) -> "RoadmapRow":
        wave, entry, findings, deps, phase, effort, trigger = (c.strip() for c in cells)
        wave_text = wave.strip("`* ")
        if not wave_text.isdigit():
            raise Malformed(EXIT_MALFORMED, "malformed-table", f"line {line}: wave {wave!r} is not an integer")
        return cls(int(wave_text), entry.strip("`"),
                   tuple(f.strip("` ") for f in findings.split(",") if f.strip("` ") and f.strip("` ") != "-"),
                   tuple(d.strip("` ") for d in deps.split(",") if d.strip("` ") and d.strip("` ") != "-"),
                   phase.strip("`"), effort.strip("`"), trigger, line)


def load_rows(lines: list[str], section: Section) -> list[RoadmapRow]:
    table = None
    idx = 0
    while (candidate := parse_table(section, idx)) is not None:
        idx += 1
        if candidate.header[:1] == ["Wave"] or candidate.header == COLUMNS:
            table = candidate
            break
    if table is None:
        raise Malformed(EXIT_MALFORMED, "malformed-table", f"section '{section.heading}' has no roadmap table")
    if table.header != COLUMNS:
        raise Malformed(EXIT_MALFORMED, "malformed-table",
                        f"line {table.header_line}: header {table.header} != {COLUMNS}")
    rows: list[RoadmapRow] = []
    for lineno, cells in table.body:
        if len(cells) != len(COLUMNS):
            raise Malformed(EXIT_MALFORMED, "malformed-table",
                            f"line {lineno}: {len(cells)} cells, expected {len(COLUMNS)}")
        row = RoadmapRow.from_cells(lineno, cells)
        if row.phase not in PHASES:
            raise Malformed(EXIT_UNKNOWN_MILESTONE, "unknown-milestone",
                            f"line {lineno}: phase {row.phase!r} outside {PHASES}")
        rows.append(row)
    if not rows:
        raise Malformed(EXIT_MALFORMED, "malformed-table", "roadmap table has zero body rows")
    return rows


def load_milestones(lines: list[str], section: Section) -> dict[str, Milestone]:
    milestones = parse_milestones(lines, section)
    if not milestones:
        raise Malformed(EXIT_MALFORMED, "malformed-table", "Milestones block with a Wave column is absent")
    for name in milestones:
        if name not in MILESTONES:
            raise Malformed(EXIT_UNKNOWN_MILESTONE, "unknown-milestone",
                            f"line {milestones[name].line}: milestone {name!r} outside {MILESTONES}")
    return milestones


def check_reference_integrity(rows, index, violations) -> None:
    for row in rows:
        for fid in row.findings:
            status = index.get(fid)
            if status is None:
                violations.append({"check": 2, "kind": "unknown-finding", "ids": [row.entry_id, fid]})
            elif status in RETIRED:
                violations.append({"check": 2, "kind": f"retired-{status}", "ids": [row.entry_id, fid]})
            elif status not in SCHEDULABLE:
                violations.append({"check": 2, "kind": f"not-schedulable-{status}", "ids": [row.entry_id, fid]})


def check_coverage(rows, index, violations) -> None:
    seen: dict[str, list[str]] = {}
    for row in rows:
        for fid in row.findings:
            seen.setdefault(fid, []).append(row.entry_id)
    for fid, status in sorted(index.items()):
        if status not in SCHEDULABLE:
            continue
        where = seen.get(fid, [])
        if not where:
            violations.append({"check": 9, "kind": "coverage-missing", "ids": [fid]})
        elif len(where) > 1:
            violations.append({"check": 9, "kind": "coverage-duplicate", "ids": [fid, *where]})


def extract_cycle(graph: dict[str, list[str]]) -> list[str]:
    stack: list[str] = []
    on_stack: set[str] = set()
    seen: set[str] = set()

    def walk(node: str) -> list[str] | None:
        stack.append(node)
        on_stack.add(node)
        seen.add(node)
        for dep in sorted(graph.get(node, ())):
            if dep in on_stack:
                return stack[stack.index(dep):] + [dep]
            if dep in graph and dep not in seen:
                found = walk(dep)
                if found:
                    return found
        stack.pop()
        on_stack.discard(node)
        return None

    for node in sorted(graph):
        if node not in seen:
            found = walk(node)
            if found:
                return found
    return sorted(graph)


def check_acyclic(rows, violations) -> list[str]:
    by_id = {r.entry_id: r for r in rows}
    indeg = {rid: 0 for rid in by_id}
    succ: dict[str, list[str]] = {rid: [] for rid in by_id}
    for r in rows:
        for dep in r.depends_on:
            if dep not in by_id:
                violations.append({"check": 2, "kind": "unknown-dependency", "ids": [r.entry_id, dep]})
                continue
            succ[dep].append(r.entry_id)
            indeg[r.entry_id] += 1
    ready = sorted(rid for rid, d in indeg.items() if d == 0)
    order: list[str] = []
    while ready:
        rid = ready.pop(0)
        order.append(rid)
        for nxt in sorted(succ[rid]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
        ready.sort()
    if len(order) != len(by_id):
        done = set(order)
        residual = {rid: [d for d in by_id[rid].depends_on if d in by_id and d not in done]
                    for rid in by_id if rid not in done}
        cycle = extract_cycle(residual)
        print("cycle: " + " -> ".join(cycle), file=sys.stderr)
        violations.append({"check": 3, "kind": "cycle", "ids": cycle})
    return order


def check_wave_order(rows, violations) -> None:
    wave_of = {r.entry_id: r.wave for r in rows}
    for r in rows:
        for dep in r.depends_on:
            if dep in wave_of and wave_of[dep] >= r.wave:
                violations.append({"check": 4, "kind": "wave-order", "ids": [r.entry_id, dep],
                                   "detail": f"wave({dep})={wave_of[dep]} >= wave({r.entry_id})={r.wave}"})


def check_milestone_order(milestones, violations) -> None:
    chain = [m for m in ("0a", "0b", "1") if m in milestones]
    if len(chain) != 3:
        violations.append({"check": 6, "kind": "milestone-missing",
                           "ids": [m for m in ("0a", "0b", "1") if m not in milestones]})
    for earlier, later in zip(chain, chain[1:], strict=False):
        if milestones[earlier].wave >= milestones[later].wave:
            violations.append({"check": 6, "kind": "milestone-order", "ids": [earlier, later],
                               "detail": f"wave({earlier})={milestones[earlier].wave} >= wave({later})={milestones[later].wave}"})
    if "gnl-import" not in milestones:
        violations.append({"check": 6, "kind": "gnl-import-missing", "ids": ["gnl-import"]})


def check_trigger_and_phase(rows, milestones, violations) -> None:
    for row in rows:
        if not row.trigger or row.trigger == "-":
            violations.append({"check": 5, "kind": "trigger-missing", "ids": [row.entry_id]})
        name = PHASE_TO_MILESTONE.get(row.phase)
        if name in milestones:
            bound = milestones[name].wave
            if row.wave >= bound:
                violations.append({"check": 7, "kind": "phase-after-milestone", "ids": [row.entry_id],
                                   "detail": f"{row.phase} at wave {row.wave} >= wave({name})={bound}"})
        if "gnl-import" in row.trigger and "gnl-import" in milestones:
            if row.wave >= milestones["gnl-import"].wave:
                violations.append({"check": 7, "kind": "trigger-after-gnl-import", "ids": [row.entry_id],
                                   "detail": f"trigger names gnl-import at wave {row.wave} >= wave(gnl-import)={milestones['gnl-import'].wave}"})


def alignment_and_overrides(lines: list[str]) -> tuple[dict[str, str], set[str]]:
    alignment: dict[str, str] = {}
    overrides: set[str] = set()
    for section in all_evaluation_sections(lines):
        for entry in iter_entries(section):
            value = entry.fields.get("Alignment", "").split("(", 1)[0].strip().strip("`")
            if value:
                alignment[entry.id] = value
            if OVERRIDE_RE.match(entry.fields.get("Owner-override", "")):
                overrides.add(entry.id)
    return alignment, overrides


def check_conflicts(rows, alignment, overrides, violations) -> None:
    for row in rows:
        for fid in row.findings:
            if alignment.get(fid) == "conflicts" and fid not in overrides:
                violations.append({"check": 8, "kind": "conflicts-unaccepted", "ids": [row.entry_id, fid],
                                   "detail": "Alignment: conflicts with no dated Owner-override line"})


def baseline_of(lines: list[str], section: Section) -> str:
    try:
        return parse_baseline(lines)
    except LookupError:
        m = re.search(r"baseline\s+([0-9a-fA-F]{7,40}|[A-Z]+)", section.heading)
        return m.group(1) if m else "unknown"


def evaluate_lines(lines: list[str], section_name: str) -> tuple[int, dict]:
    try:
        section = find_section(lines, section_name)
    except SectionNotFound:
        return EXIT_MALFORMED, {"violations": [], "error": f"section-not-found: {section_name}"}
    report: dict = {"baseline": baseline_of(lines, section), "section": section.heading,
                    "rows": 0, "order": [], "milestones": {}, "violations": []}
    try:
        rows = load_rows(lines, section)
        milestones = load_milestones(lines, section)
    except Malformed as exc:
        report["error"] = f"{exc.kind}: {exc.detail}"
        report["exit"] = exc.code
        return exc.code, report
    violations: list[dict] = []
    index = register_findings(lines)
    alignment, overrides = alignment_and_overrides(lines)
    check_reference_integrity(rows, index, violations)
    order = check_acyclic(rows, violations)
    check_wave_order(rows, violations)
    check_trigger_and_phase(rows, milestones, violations)
    check_milestone_order(milestones, violations)
    check_conflicts(rows, alignment, overrides, violations)
    check_coverage(rows, index, violations)
    code = EXIT_VIOLATION if violations else EXIT_OK
    report.update({"rows": len(rows), "order": order,
                   "milestones": {name: m.wave for name, m in sorted(milestones.items())},
                   "violations": violations, "exit": code})
    return code, report


def evaluate_text(text: str, section_name: str = DEFAULT_SECTION) -> tuple[int, dict]:
    return evaluate_lines(text.splitlines(), section_name)


def run(args: argparse.Namespace) -> int:
    lines = pathlib.Path(args.backlog).read_text(encoding="utf-8").splitlines()
    code, report = evaluate_lines(lines, args.section)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        if "error" in report:
            print(report["error"], file=sys.stderr)
        for v in report["violations"]:
            detail = f" — {v['detail']}" if v.get("detail") else ""
            print(f"check-{v['check']} {v['kind']}: {' -> '.join(v['ids']) if v['kind'] == 'cycle' else ', '.join(v['ids'])}{detail}")
        if code in (EXIT_OK, EXIT_VIOLATION):
            print(f"{report['rows']} rows, {len(report['violations'])} violation(s)")
    return code


def run_self_test() -> int:
    fixtures = pathlib.Path(__file__).resolve().parent / "fixtures"
    good = (fixtures / "good-roadmap.md").read_text(encoding="utf-8")
    cyclic = (fixtures / "cyclic-roadmap.md").read_text(encoding="utf-8")
    malformed = good.replace("| S | before the GNL import |", "| S |")
    unknown = good.replace("| serves 0a |", "| serves 0c |", 1)
    w3 = "| 3 | W3-boundary-frame | CD-042 | W2-output-owner | serves 0a | M | with the 0a seam work |"
    late_phase = good.replace(
        w3, w3.replace("| 3 |", "| 6 |") + "\n| 6 | W6-late-sibling | - | W3-boundary-frame | neutral | S | after the frame lands |")
    unaccepted = (good.replace(w3, w3.replace("CD-042", "CD-042, CD-045, PD-004"))
                  .replace("- **Status:** rejected", "- **Status:** ratified")
                  .replace("- **Status:** dismissed", "- **Status:** kept"))
    cases = [
        ("good-roadmap", good, EXIT_OK, set()),
        ("cyclic-roadmap", cyclic, EXIT_VIOLATION, {"cycle"}),
        ("malformed(derived)", malformed, EXIT_MALFORMED, set()),
        ("unknown-milestone(derived)", unknown, EXIT_UNKNOWN_MILESTONE, set()),
        ("phase-and-wave-order(derived)", late_phase, EXIT_VIOLATION, {"phase-after-milestone", "wave-order"}),
        ("conflicts-retired-coverage(derived)", unaccepted, EXIT_VIOLATION,
         {"conflicts-unaccepted", "retired-do-not-touch", "coverage-missing"}),
    ]
    failures = 0
    for name, text, want_code, want_kinds in cases:
        got_code, report = evaluate_text(text)
        kinds = {v["kind"] for v in report.get("violations", [])}
        ok = got_code == want_code and want_kinds <= kinds
        failures += 0 if ok else 1
        extra = f" error={report['error']}" if "error" in report else ""
        print(f"{'PASS' if ok else 'FAIL'} {name}: exit {got_code} (want {want_code}) kinds={sorted(kinds)}{extra}")
    print(f"self-test: {len(cases) - failures}/{len(cases)} cases passed")
    return EXIT_OK if failures == 0 else EXIT_VIOLATION


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="check-roadmap-dag.py", description=__doc__.splitlines()[0])
    ap.add_argument("--section", default=DEFAULT_SECTION, help="roadmap section title or station tail")
    ap.add_argument("--backlog", default=str(DEFAULT_BACKLOG), help="register (or fixture) file")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    return run_self_test() if args.self_test else run(args)


if __name__ == "__main__":
    sys.exit(main())
