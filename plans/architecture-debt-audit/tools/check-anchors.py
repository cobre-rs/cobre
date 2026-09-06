#!/usr/bin/env python3
"""Resolve every repo-rooted anchor of one register section at the pinned baseline.

Exit codes: 0 all anchors resolve, 1 anchor(s) missing, 2 baseline drift
(HEAD's evaluated surfaces differ from the pin; pass --allow-drift to proceed),
3 section not found. Read-only: never writes the register or a fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from lib.backlog_parse import (  # noqa: E402
    Anchor,
    BaselineDrift,
    SectionNotFound,
    ensure_baseline,
    find_section,
    git_show,
    iter_anchors,
    parse_baseline,
    read_register,
    repo_root,
)

EXIT_OK = 0
EXIT_ANCHOR_MISSING = 1
EXIT_BASELINE_DRIFT = 2
EXIT_SECTION_NOT_FOUND = 3

DEFAULT_REGISTER = repo_root() / "plans" / "architecture-debt-audit" / "BACKLOG.md"

DECL = (r"^\s*(pub(\([^)]*\))?\s+)?(async\s+)?"
        r"(fn|struct|enum|trait|type|const|static|mod|impl)\s+{sym}\b")


def check_anchor(baseline: str, anchor: Anchor) -> str | None:
    blob = git_show(baseline, anchor.path)
    if blob is None:
        return "path-missing"
    if anchor.symbol:
        pattern = re.compile(DECL.format(sym=re.escape(anchor.symbol)), re.MULTILINE)
        return None if pattern.search(blob) else "symbol-missing"
    if anchor.line is not None and anchor.line > blob.count("\n") + 1:
        return "line-out-of-range"
    return None


def _evaluate(register: pathlib.Path, section_name: str, baseline: str) -> tuple[int, dict]:
    lines = read_register(register)
    try:
        section = find_section(lines, section_name)
    except SectionNotFound:
        print(f"section-not-found: {section_name}", file=sys.stderr)
        return EXIT_SECTION_NOT_FOUND, {}
    anchors = iter_anchors(section)
    failures = [{"anchor": a.raw, "kind": k, "entry": a.entry_id, "line": a.lineno}
                for a in anchors if (k := check_anchor(baseline, a))]
    report = {"baseline": baseline, "section": section_name,
              "checked": len(anchors), "failures": failures}
    return (EXIT_ANCHOR_MISSING if failures else EXIT_OK), report


def run(section_name: str, register: pathlib.Path, baseline: str | None,
        allow_drift: bool, as_json: bool) -> int:
    baseline = baseline or parse_baseline(read_register(DEFAULT_REGISTER))
    try:
        ensure_baseline(baseline, allow_drift)
    except BaselineDrift as exc:
        print(f"baseline-drift: {exc}", file=sys.stderr)
        return EXIT_BASELINE_DRIFT
    code, report = _evaluate(register, section_name, baseline)
    if code == EXIT_SECTION_NOT_FOUND:
        return code
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for f in report["failures"]:
            print(f"anchor-missing[{f['kind']}] {f['anchor']} "
                  f"(entry {f['entry']}, {register.name}:{f['line']})")
        print(f"checked {report['checked']} anchors, {len(report['failures'])} failing")
    return code


CASES = [
    ("good-section.md", "good-section", EXIT_OK, 0),
    ("bad-anchor.md", "bad-anchor", EXIT_ANCHOR_MISSING, 2),
    ("good-section.md", "no-such-station", EXIT_SECTION_NOT_FOUND, 0),
]


def self_test(baseline: str) -> int:
    fixtures = pathlib.Path(__file__).resolve().parent / "fixtures"
    watched = [fixtures / name for name, _, _, _ in CASES] + [DEFAULT_REGISTER]
    digests = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    failed = 0
    for filename, section, want_code, want_failures in CASES:
        code, report = _evaluate(fixtures / filename, section, baseline)
        got = len(report.get("failures", []))
        ok = code == want_code and got == want_failures
        failed += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {filename}:{section} "
              f"exit={code} (want {want_code}) failures={got} (want {want_failures})")
    unchanged = all(hashlib.sha256(p.read_bytes()).hexdigest() == d for p, d in digests.items())
    print(f"{'ok  ' if unchanged else 'FAIL'} fixtures and register byte-identical after the run")
    failed += 0 if unchanged else 1
    print(f"self-test: {len(CASES) + 1 - failed}/{len(CASES) + 1} cases passed")
    return EXIT_OK if failed == 0 else EXIT_ANCHOR_MISSING


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("section", nargs="?", help="section title or station name (after the em dash)")
    parser.add_argument("--register", type=pathlib.Path, default=DEFAULT_REGISTER)
    parser.add_argument("--baseline", help="override the SHA pinned in the register header")
    parser.add_argument("--allow-drift", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        baseline = args.baseline or parse_baseline(read_register(DEFAULT_REGISTER))
        try:
            ensure_baseline(baseline, args.allow_drift)
        except BaselineDrift as exc:
            print(f"baseline-drift: {exc}", file=sys.stderr)
            return EXIT_BASELINE_DRIFT
        return self_test(baseline)
    if not args.section:
        parser.error("section is required unless --self-test is given")
    return run(args.section, args.register, args.baseline, args.allow_drift, args.json)


if __name__ == "__main__":
    sys.exit(main())
