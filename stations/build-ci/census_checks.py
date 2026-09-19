#!/usr/bin/env python3
"""Census, inventory and section-body assertions behind verify-census.sh (build-ci).

The driver re-measures every figure from the tree at the station baseline and hands the
measurements here; the functions are pure so test_station.py can feed tampered census,
inventory and register copies and prove each failure path. Read-only: nothing here writes.

  census_checks.py heading <register> "<title>"                   prints the 1-based heading line
  census_checks.py census  <station-dir> <register> <line> k=v…   census + inventory + section figures
  census_checks.py body    <register> <line> <calibration.json>   Alignment, two-site, reviewer rating

Exit 0 clean / 1 findings (one `FAIL …` line each) / 2 the heading resolves to zero or many lines.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any

WIRING = frozenset(
    {"blocking", "advisory-by-design", "unwired", "transitively-advisory"}
)
ALIGNMENT_VOCAB = frozenset(
    {"advances-0a", "advances-0b", "advances-1", "neutral", "conflicts"}
)
EXPECTED_JOBS = (
    "check",
    "test",
    "clippy",
    "clp",
    "fmt",
    "quality-scripts",
    "schemas",
    "docs-examples",
    "docs",
    "security",
    "deny",
    "license-notices",
    "coverage",
    "python",
)
INDIRECT = frozenset({"unwired", "transitively-advisory"})
INT_KEYS = ("mpichCi", "mpichTotal", "schemas", "buildRs", "workflows", "jobsRaw")
BOOL_KEYS = ("fbs", "rootFbs", "actionsDir")
LIST_KEYS = ("jobIds", "unwired", "transitive", "gateFiles")
ENTRY_HEAD_RE = re.compile(r"^\*\*((?:CD|PD|OD|TD)-\d{3}) · Sev [ABC] · ")
# An entry runs from its `**ID · Sev` heading to the next entry heading, lens heading or
# UNMEASURED informational row, so the performance rows never fold into the last entry.
BLOCK_SPLIT_RE = re.compile(
    r"(?m)^(?=\*\*(?:CD|PD|OD|TD)-\d{3} · Sev [ABC] · |#### |\*\*Informational · )"
)
CANDIDATE_RE = re.compile(r"^#+ .*(build[-/ ]?ci|station 7)", re.I)


def resolve_heading(lines: Sequence[str], title: str) -> list[int]:
    """1-based line numbers whose whole text is `## `, `### ` or `#### ` followed by `title`."""
    wanted = {f"{'#' * level} {title}" for level in (2, 3, 4)}
    return [n for n, raw in enumerate(lines, 1) if raw in wanted]


def heading_candidates(lines: Sequence[str]) -> list[str]:
    return [f"{n}: {raw}" for n, raw in enumerate(lines, 1) if CANDIDATE_RE.match(raw)]


def section_body(lines: Sequence[str], heading_line: int) -> str:
    """The text below the heading at `heading_line` up to the next heading of level 1-3."""
    body: list[str] = []
    for raw in lines[heading_line:]:
        if re.match(r"^#{1,3} ", raw):
            break
        body.append(raw)
    return "\n".join(body)


def entries_of(section: str) -> dict[str, str]:
    """Finding id -> its block, for every `**ID · Sev` entry in the section."""
    out: dict[str, str] = {}
    for block in BLOCK_SPLIT_RE.split(section):
        head = ENTRY_HEAD_RE.match(block)
        if head:
            out[head.group(1)] = block
    return out


def parse_measured(pairs: Sequence[str]) -> dict[str, Any]:
    """`key=value` arguments into the measured dict (ints, 0/1 flags, comma-joined lists)."""
    out: dict[str, Any] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if key in INT_KEYS:
            out[key] = int(value)
        elif key in BOOL_KEYS:
            out[key] = value not in ("", "0")
        elif key in LIST_KEYS:
            out[key] = [v for v in value.split(",") if v]
        else:
            raise KeyError(f"unknown measurement {key!r}")
    missing = [k for k in (*INT_KEYS, *BOOL_KEYS, *LIST_KEYS) if k not in out]
    if missing:
        raise KeyError(f"measurements missing: {missing}")
    return out


def tree_findings(m: Mapping[str, Any]) -> list[str]:
    """The figures the ticket asserts at the baseline, each re-derived by the driver."""
    bad: list[str] = []
    jobs = len(m["jobIds"])
    if jobs != len(EXPECTED_JOBS) or tuple(m["jobIds"]) != EXPECTED_JOBS:
        bad.append(
            f"ci.yml jobs scoped to the jobs: block = {jobs} {m['jobIds']}, "
            f"expected the {len(EXPECTED_JOBS)} named jobs"
        )
    if m["jobsRaw"] != jobs + 1:
        bad.append(
            f"raw job-id grep = {m['jobsRaw']}; expected the scoped count + the one "
            f"`push:` trigger hit ({jobs + 1})"
        )
    if set(m["unwired"]) != {"check-comment-bloat.sh"}:
        bad.append(
            f"gates without direct wiring = {sorted(m['unwired'])}, "
            "expected exactly check-comment-bloat.sh"
        )
    if list(m["transitive"]) != ["scripts/ci/quality-report.sh:132"]:
        bad.append(
            f"check-comment-bloat.sh invokers = {m['transitive']}, "
            "expected scripts/ci/quality-report.sh:132 only"
        )
    if not m["fbs"] or m["rootFbs"]:
        bad.append(
            f"policy.fbs: crates/cobre-io/schemas present={int(m['fbs'])}, "
            f"root schemas/policy.fbs present={int(m['rootFbs'])} (expected 1 / 0)"
        )
    if m["actionsDir"]:
        bad.append(
            ".github/actions exists at the baseline; the eight-copies MPICH premise is stale"
        )
    return bad


def census_completeness(census: Mapping[str, Any], m: Mapping[str, Any]) -> list[str]:
    """One classified row per gate file on disk, each class carrying its evidence."""
    bad: list[str] = []
    rows = census["rows"]
    gate_files = list(m["gateFiles"])
    unwired = set(m["unwired"])
    transitive = list(m["transitive"])
    by_name = {r["script"].rsplit("/", 1)[-1]: r for r in rows}
    if len(rows) != len(gate_files):
        bad.append(f"census rows {len(rows)} != measured gate files {len(gate_files)}")
    in_census = {r["script"] for r in rows}
    if in_census != set(gate_files):
        bad.append(
            "census scripts != the gate glob: only-in-census "
            f"{sorted(in_census - set(gate_files))}, only-on-disk "
            f"{sorted(set(gate_files) - in_census)}"
        )
    for name, r in sorted(by_name.items()):
        cls = r.get("class")
        if cls not in WIRING:
            bad.append(f"{name}: class={cls!r} outside the vocabulary")
            continue
        if cls in INDIRECT:
            if name not in unwired:
                bad.append(
                    f"{name}: recorded {cls} but a direct reference exists at the baseline"
                )
            ti = r.get("transitiveInvocation")
            if not ti or f"{ti['path']}:{ti['line']}" not in transitive:
                bad.append(
                    f"{name}: {cls} row without its invoking gate anchored "
                    f"(measured {transitive})"
                )
            continue
        if not all(r.get(k) for k in ("workflow", "workflowJob", "step")):
            bad.append(f"{name}: {cls} row without workflow/job/step")
        if cls == "advisory-by-design":
            ev = r.get("advisoryEvidence") or {}
            if not (
                ev.get("stepComment")
                and ev.get("stepCommentAnchor")
                and ev.get("terminalExit0Anchor")
            ):
                bad.append(
                    f"{name}: advisory row without the step comment + terminal exit 0 "
                    "proving exit 0 is intentional"
                )
    indirect = {n for n, r in by_name.items() if r.get("class") in INDIRECT}
    for s in sorted(unwired - indirect):
        bad.append(
            f"{s}: measured unwired but not classified unwired/transitively-advisory"
        )
    counts = census.get("counts", {})
    if sum(counts.values()) != len(rows):
        bad.append(f"census counts {counts} do not sum to {len(rows)} rows")
    return bad


def inventory_findings(inv: Mapping[str, Any], m: Mapping[str, Any]) -> list[str]:
    """inventory.json recorded == re-measured, recorded vs re-measured named on every miss."""
    ci, gates, ws, sch = inv["ci"], inv["gates"], inv["workspace"], inv["schemas"]
    recorded: dict[str, tuple[Any, Any]] = {
        "ci.ciJobCount": (ci["ciJobCount"], len(m["jobIds"])),
        "ci.ciJobs[]": ([j["id"] for j in ci["ciJobs"]], list(m["jobIds"])),
        "ci.workflowCount": (ci["workflowCount"], m["workflows"]),
        "ci.mpichFromSourceBlocks[ci.yml]": (
            ci["mpichFromSourceBlocks"][".github/workflows/ci.yml"],
            m["mpichCi"],
        ),
        "ci.mpichFromSourceTotal": (ci["mpichFromSourceTotal"], m["mpichTotal"]),
        "ci.compositeActionsDir": (
            ci["compositeActionsDir"] is not None,
            bool(m["actionsDir"]),
        ),
        "gates.totalExecutables": (gates["totalExecutables"], len(m["gateFiles"])),
        "gates.files[]": (
            sorted(f["path"] for f in gates["files"]),
            sorted(m["gateFiles"]),
        ),
        "gates.shCount+pyCount+libHelpers": (
            gates["shCount"] + gates["pyCount"] + len(gates["libHelpers"]),
            len(m["gateFiles"]),
        ),
        "schemas.jsonCount": (sch["jsonCount"], m["schemas"]),
        "schemas.rootPolicyFbsExists": (sch["rootPolicyFbsExists"], bool(m["rootFbs"])),
        "workspace.buildScripts[]": (len(ws["buildScripts"]), m["buildRs"]),
    }
    return [
        f"inventory.{key} recorded {rec!r} but re-measured {got!r}"
        for key, (rec, got) in recorded.items()
        if rec != got
    ]


def section_figure_findings(
    section: str, inv: Mapping[str, Any], m: Mapping[str, Any]
) -> list[str]:
    """Every census figure the section quotes equals the re-measured one."""
    bad: list[str] = []
    jobs = len(m["jobIds"])
    gatefiles = len(m["gateFiles"])

    def quoted(label: str, pattern: str, expect: tuple[int, ...]) -> None:
        found = re.search(pattern, section)
        if not found:
            bad.append(f"section no longer quotes {label} ({pattern})")
            return
        got = tuple(int(g) for g in found.groups())
        if got != expect:
            bad.append(f"section quotes {label} {got} but re-measured {expect}")

    quoted(
        "workflows + ci.yml jobs",
        r"(\d+) workflows, (\d+) `ci\.yml` jobs",
        (m["workflows"], jobs),
    )
    quoted(
        "gate corpus",
        r"(\d+) `\.sh` \+ (\d+) `\.py` gates and the shared `lib/comment_scan\.sh`",
        (inv["gates"]["shCount"], inv["gates"]["pyCount"]),
    )
    quoted("schema exports", r"\((\d+) JSON exports\)", (m["schemas"],))
    quoted("census heading", r"census \((\d+) gate files", (gatefiles,))
    for found in re.finditer(r"\b(\d+) (?:`ci\.yml` )?jobs\b", section):
        if int(found.group(1)) != jobs:
            bad.append(
                f"section quotes {found.group(0)!r}; ci.yml has {jobs} jobs "
                f"(the raw grep's {m['jobsRaw']} includes the push: trigger)"
            )
    if (
        not re.search(r"\b(?:eight|8)\b[^\n]{0,40}\b(?:copies|times)\b", section)
        or m["mpichCi"] != 8
    ):
        bad.append(
            f"section does not state eight MPICH copies in ci.yml / re-measured {m['mpichCi']}"
        )
    if (
        not re.search(r"\b(?:ten|10) (?:consumers|call sites)\b", section)
        or m["mpichTotal"] != 10
    ):
        bad.append(
            f"section does not state ten MPICH consumers / re-measured {m['mpichTotal']}"
        )
    table = [
        line
        for line in section.splitlines()
        if line.startswith("| `") and (".sh`" in line or ".py`" in line)
    ]
    if len(table) != gatefiles:
        bad.append(
            f"rendered census table has {len(table)} rows, measured {gatefiles} gate files"
        )
    bloat = [line for line in table if "check-comment-bloat.sh" in line]
    if (
        len(bloat) != 1
        or "transitively-advisory" not in bloat[0]
        or "quality-report.sh:132" not in bloat[0]
    ):
        bad.append(
            "rendered census: check-comment-bloat.sh row is not the single "
            "transitively-advisory row citing quality-report.sh:132"
        )
    indirect_rows = sum("transitively-advisory" in line for line in table)
    if indirect_rows != len(m["unwired"]):
        bad.append(
            f"rendered census: {indirect_rows} transitively-advisory rows, measured "
            f"{len(m['unwired'])} gates without direct wiring"
        )
    return bad


def census_findings(
    census: Mapping[str, Any],
    inv: Mapping[str, Any],
    m: Mapping[str, Any],
    section: str,
) -> list[str]:
    return (
        tree_findings(m)
        + census_completeness(census, m)
        + inventory_findings(inv, m)
        + section_figure_findings(section, inv, m)
    )


def body_findings(section: str, calibration: Mapping[str, Any]) -> list[str]:
    """One Alignment per entry, complete two-site fix-shapes, the reviewer's rating on every
    downgrade — over the rendered section and against calibration.json."""
    bad: list[str] = []
    entries = entries_of(section)
    if not entries:
        bad.append(
            "the section parses to zero entries — an empty section does not pass vacuously"
        )
    for fid, block in entries.items():
        aligns = re.findall(r"^- \*\*Alignment:\*\*\s*`?([a-z0-9-]+)", block, re.M)
        if len(aligns) != 1 or aligns[0] not in ALIGNMENT_VOCAB:
            bad.append(
                f"{fid}: Alignment values {aligns} (exactly one from the five-value "
                "vocabulary expected)"
            )
        fix = "".join(
            re.findall(r"^- \*\*(?:Fix-shape|Two-site rule[^*]*):\*\*.*$", block, re.M)
        )
        if "check-infra-genericity.sh" in fix and "CLAUDE.md" not in fix:
            bad.append(
                f"{fid}: fix-shape names check-infra-genericity.sh without the CLAUDE.md "
                "rule text (incomplete two-site edit)"
            )
        for two_site in re.findall(r"^- \*\*Two-site rule[^\n]*$", block, re.M):
            if not (
                "CLAUDE.md" in two_site
                and re.search(r"scripts/ci/|Cargo\.toml", two_site)
                and ".github/workflows/" in two_site
            ):
                bad.append(
                    f"{fid}: Two-site rule line does not name rule + script + ci_step"
                )
        if re.search(r"downgrad", block, re.I) and not re.search(
            r"^- \*\*Reviewer rating:\*\*\s*[ABC]\b", block, re.M
        ):
            bad.append(
                f"{fid}: a downgrade is described without the reviewer's original rating"
            )
    for row in calibration["assigned"]:
        if row.get("downgradeReason"):
            block = entries.get(row["id"], "")
            if not re.search(
                rf"^- \*\*Reviewer rating:\*\*\s*{row['reviewerRating']}\b", block, re.M
            ):
                bad.append(
                    f"{row['id']}: calibration downgrade from {row['reviewerRating']} "
                    "not recorded in the entry"
                )
        if row.get("hardRule"):
            kinds = {s["kind"] for s in row.get("sites", [])}
            if kinds != {"rule", "script", "ci_step"}:
                bad.append(
                    f"{row['id']}: hardRule {row['hardRule']} names sites {sorted(kinds)}"
                )
    return bad


def _read_lines(path: str) -> list[str]:
    return pathlib.Path(path).read_text(encoding="utf-8").splitlines()


def _load(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _report(bad: list[str], ok: str) -> int:
    print("\n".join(f"FAIL {b}" for b in bad) or ok)
    return 1 if bad else 0


def main(argv: Sequence[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 1
    sub, rest = argv[0], argv[1:]
    if sub == "heading":
        lines = _read_lines(rest[0])
        hits = resolve_heading(lines, rest[1])
        if len(hits) != 1:
            print(
                f"FAIL heading-resolution: {len(hits)} headings match {rest[1]!r} "
                "(expected exactly 1)",
                file=sys.stderr,
            )
            for candidate in heading_candidates(lines):
                print(f"  candidate: {candidate}", file=sys.stderr)
            return 2
        print(hits[0])
        return 0
    if sub == "census":
        station = pathlib.Path(rest[0])
        section = section_body(_read_lines(rest[1]), int(rest[2]))
        measured = parse_measured(rest[3:])
        bad = census_findings(
            _load(station / "gate-census.json"),
            _load(station / "inventory.json"),
            measured,
            section,
        )
        return _report(
            bad,
            f"OK {len(measured['gateFiles'])} gate files / {len(measured['unwired'])} without "
            f"direct wiring / {len(measured['jobIds'])} ci.yml jobs (raw grep "
            f"{measured['jobsRaw']}) / {measured['mpichCi']}x MPICH in ci.yml "
            f"({measured['mpichTotal']} total) / {measured['schemas']} schemas + policy.fbs / "
            f"{measured['buildRs']} build.rs / {measured['workflows']} workflows",
        )
    if sub == "body":
        section = section_body(_read_lines(rest[0]), int(rest[1]))
        calibration = _load(pathlib.Path(rest[2]))
        bad = body_findings(section, calibration)
        downgrades = sum(1 for r in calibration["assigned"] if r.get("downgradeReason"))
        return _report(
            bad,
            f"OK {len(entries_of(section))} entries: one Alignment each, two-site fix-shapes "
            f"complete, {downgrades} downgrades carry the reviewer's rating",
        )
    print(f"unknown subcommand: {sub!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
