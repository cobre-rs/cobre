#!/usr/bin/env python3
"""Validate the ratified finding-entry shape in one register section or all of them.

Header line: `**<ID> · Sev A|B|C · <category> · effort S|M|L · confidence high|med|low**`.
Mandatory bullets: Station, Baseline, Anchors, Evidence, Fix-shape, Alignment; the
Alignment value is one of advances-0a | advances-0b | advances-1 | neutral | conflicts
followed by a roadmap citation (`conflicts` is valid here; the owner gate holds it).
Exit codes: 0 ok, 1 field-missing / alignment-invalid, 3 section not found,
4 duplicate finding ID. Read-only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from lib.backlog_parse import (  # noqa: E402
    Entry,
    Section,
    SectionNotFound,
    all_evaluation_sections,
    find_section,
    iter_entries,
    read_register,
    repo_root,
)

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_SECTION_NOT_FOUND = 3
EXIT_DUPLICATE = 4

REGISTER = repo_root() / "plans" / "architecture-debt-audit" / "BACKLOG.md"
HEADER_FIELDS = ("id", "severity", "category", "effort", "confidence")
BODY_FIELDS = ("Station", "Baseline", "Anchors", "Evidence", "Fix-shape", "Alignment")
ALIGNMENT_VOCAB = frozenset({"advances-0a", "advances-0b", "advances-1", "neutral", "conflicts"})
HEADER_RE = re.compile(
    r"^(?P<id>(?:CD|PD|OD|TD)-\d{3})\s*·\s*(?P<severity>Sev\s+[ABC](?:\s*\([^)]*\))?)\s*·\s*"
    r"(?P<category>[^·]+?)\s*·\s*effort\s+(?P<effort>[SML](?:\s*[–-]\s*[SML])?)\s*·\s*"
    r"confidence\s+(?P<confidence>high|med|low)(?:\s*\([^)]*\))?\s*$"
)


def header_fields(entry: Entry) -> dict[str, str]:
    m = HEADER_RE.match(entry.heading)
    if not m:
        return {"id": entry.id}
    return {k: m.group(k) for k in HEADER_FIELDS}


def check_entry(entry: Entry, required: tuple[str, ...]) -> tuple[list[str], str | None]:
    head = header_fields(entry)
    missing = [f for f in required if f in HEADER_FIELDS and not head.get(f)]
    missing += [f for f in required if f not in HEADER_FIELDS and not (entry.fields.get(f) or "").strip()]
    bad_alignment = None
    if "Alignment" in required and "Alignment" not in missing:
        value = entry.fields["Alignment"].split("(", 1)[0].strip().strip("`")
        if value not in ALIGNMENT_VOCAB:
            bad_alignment = value
    return missing, bad_alignment


def select_sections(lines: list[str], section_name: str | None, sweep: bool) -> list[Section]:
    if sweep or section_name is None:
        return all_evaluation_sections(lines)
    return [find_section(lines, section_name)]


def evaluate(register: pathlib.Path, section_name: str | None, sweep: bool,
             required: tuple[str, ...]) -> tuple[int, dict]:
    lines = read_register(register)
    try:
        sections = select_sections(lines, section_name, sweep)
    except SectionNotFound as exc:
        print(f"section-not-found: {exc}", file=sys.stderr)
        return EXIT_SECTION_NOT_FOUND, {}
    seen: dict[str, int] = {}
    incomplete: list[dict] = []
    total = 0
    for section in sections:
        for entry in iter_entries(section):
            total += 1
            if entry.id in seen:
                print(f"duplicate-id {entry.id}: {register.name}:{seen[entry.id]} and "
                      f"{register.name}:{entry.lineno}", file=sys.stderr)
                return EXIT_DUPLICATE, {"entries": total, "duplicate": entry.id,
                                        "lines": [seen[entry.id], entry.lineno]}
            seen[entry.id] = entry.lineno
            missing, bad = check_entry(entry, required)
            if missing or bad:
                incomplete.append({"id": entry.id, "line": entry.lineno, "missing": missing,
                                   "badAlignment": bad})
    report = {"entries": total, "sections": len(sections), "required": list(required),
              "incomplete": incomplete}
    return (EXIT_INVALID if incomplete else EXIT_OK), report


def resolve_target(arg: str | None, register: pathlib.Path | None) -> tuple[pathlib.Path, str | None, bool]:
    if arg is None:
        return register or REGISTER, None, True
    candidate = pathlib.Path(arg)
    if candidate.is_file():
        return candidate, None, True
    return register or REGISTER, arg, False


def run(arg: str | None, register: pathlib.Path | None, sweep: bool,
        required: tuple[str, ...], as_json: bool) -> int:
    target, section_name, implied = resolve_target(arg, register)
    code, report = evaluate(target, section_name, sweep or implied, required)
    if code in (EXIT_SECTION_NOT_FOUND, EXIT_DUPLICATE):
        if as_json and report:
            print(json.dumps(report, indent=2, sort_keys=True))
        return code
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for item in report["incomplete"]:
            if item["missing"]:
                print(f"field-missing {item['id']} ({target.name}:{item['line']}): {', '.join(item['missing'])}")
            if item["badAlignment"] is not None:
                print(f"alignment-invalid {item['id']} ({target.name}:{item['line']}): '{item['badAlignment']}'")
        print(f"checked {report['entries']} entries in {report['sections']} section(s), "
              f"{len(report['incomplete'])} incomplete")
    return code


def self_test() -> int:
    fx = pathlib.Path(__file__).resolve().parent / "fixtures"
    failed = 0

    def expect(label: str, got: int, want: int, extra_ok: bool = True) -> None:
        nonlocal failed
        ok = got == want and extra_ok
        failed += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {label}: exit={got} (want {want})")

    code, _ = evaluate(fx / "reraise-seeded.md", None, True, HEADER_FIELDS + BODY_FIELDS)
    expect("reraise-seeded.md fully templated entries pass", code, EXIT_OK)
    code, report = evaluate(fx / "missing-alignment.md", None, True, HEADER_FIELDS + BODY_FIELDS)
    ids = {i["id"]: i for i in report.get("incomplete", [])}
    expect("missing-alignment.md field-missing + alignment-invalid", code, EXIT_INVALID,
           set(ids) == {"CD-902", "CD-903"} and ids["CD-902"]["missing"] == ["Alignment"]
           and ids["CD-903"]["badAlignment"] == "maybe")
    with tempfile.TemporaryDirectory() as tmp:
        dup = pathlib.Path(tmp) / "dup.md"
        dup.write_text((fx / "missing-alignment.md").read_text(encoding="utf-8").replace("CD-903", "CD-902", 1),
                       encoding="utf-8")
        code, report = evaluate(dup, None, True, HEADER_FIELDS + BODY_FIELDS)
        expect("duplicate-id on a tempfile copy", code, EXIT_DUPLICATE,
               report.get("duplicate") == "CD-902" and len(report.get("lines", [])) == 2)
    print(f"self-test: {'FAIL' if failed else 'ok'} ({failed} failing cases)")
    return EXIT_INVALID if failed else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("section", nargs="?", help="register section name, or a fixture file path")
    ap.add_argument("--register", type=pathlib.Path, help="register file when `section` is a name or omitted")
    ap.add_argument("--all", action="store_true", help="sweep every dated evaluation section")
    ap.add_argument("--require", action="append", default=[], metavar="FIELD",
                    help="check only the named field(s); implies --all without a section")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    required = tuple(args.require) if args.require else HEADER_FIELDS + BODY_FIELDS
    return run(args.section, args.register, args.all, required, args.json)


if __name__ == "__main__":
    sys.exit(main())
