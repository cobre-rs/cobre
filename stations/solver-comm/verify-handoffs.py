#!/usr/bin/env python3
"""Station-specific verification for cobre-solver + cobre-comm — what the shared verifier cannot know.

Run AFTER `tools/verify-station.sh solver-comm "<title>"` (which writes the generic table into
verification.md); this script appends — idempotently — the station-specific table:

1. sanctioned-seam polarity: the shared-memory hierarchy identifiers may appear ONLY under the
   Cleared subsection with a mirror citation, and the superseded cut-sync methods ONLY as the E5
   dup-of handoff whose anchors resolve in crates/cobre-sddp/src/cut/cut_sync.rs at the station
   baseline; either one attached to a fresh CD/PD/OD/TD id is a failure (check-reraise's zero exit
   proves absence of a token, not its polarity);
2. handoff shape: every queued perf claim carries layout in {4t, 2x2}, non-empty exercising call
   sites and no measured number; the E9 handoff covers all five StageTemplate multistage fields
   with a disposition and a resolving types.rs anchor; the E7 handoff names its gate script,
   evading identifier and evaded pattern;
3. the blind-spot evidence pair: the genericity gate exits 0 WHILE `cut_nz_per_col` is still
   present in freeze.rs above the `#[cfg(test)]` line the gate's awk pre-filter truncates at;
4. read-only workspace: `git status --porcelain` minus the carried-in paths names no NEW modified
   tracked path under an evaluated surface.

Exit 0 when every row passes, 1 otherwise; every command and exit status is recorded in the table.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
from collections.abc import Sequence

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUDIT = ROOT / "plans/architecture-debt-audit"
sys.path.insert(0, str(AUDIT / "tools"))

from lib import backlog_parse  # noqa: E402
from lib import station_checks as sc  # noqa: E402

SLUG = "solver-comm"
MIRROR = "docs/design/reserved-seams-and-deferred-debt.md"
SHARED_MEM = (
    "SharedMemoryProvider",
    "SharedRegion",
    "LocalCommunicator",
    "LocalCommKind",
    "HeapRegion",
    "split_local",
)
CUT_SYNC_FILE = "crates/cobre-sddp/src/cut/cut_sync.rs"
CUT_SYNC = {
    "sync_cuts": 243,
    "pack_local_records": 400,
    "sync_packed_records": 495,
    "sync_level_records": 581,
}
SUPERSEDED = ("sync_cuts", "pack_local_records", "sync_packed_records")
LAYOUTS = {"4t", "2x2"}
ITEM8_FIELDS = ("n_state", "n_transfer", "n_dual_relevant", "n_hydro", "max_par_order")
DISPOSITIONS = {"keep", "retire", "sharpen", "retire-with-commit"}
GATE = "scripts/ci/check-infra-genericity.sh"
FREEZE = "crates/cobre-solver/src/freeze.rs"
EVADER = "cut_nz_per_col"
EXPECTED_HITS = [22, 137, 138, 141, 188]
TIMING = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:x\b|×|%|(?:ms|µs|us|ns|s|sec|secs|seconds|minutes|min|speedup|faster|slower)\b)",
    re.I,
)
SURFACE_RE = re.compile(
    r"^.{2} (crates|scripts|schemas|\.github|Cargo\.(toml|lock)|docs|examples|tests)(/|$)"
)
CARRIED_IN = (".gitignore",)
MARKER = "## Station-specific checks — solver-comm"
# Subsections where the seam identifiers may legitimately appear: never inside an id'd entry.
READ_ONLY_HEADINGS = (
    "↩︎ Cleared",
    "Positives",
    "Prior-register",
    "Owner gate — decisions",
    "Informational (recorded, no severity, no id)",
)


class Row:
    def __init__(self, label: str, command: str, code: int, detail: str = "") -> None:
        self.label, self.command, self.code, self.detail = label, command, code, detail

    @property
    def ok(self) -> bool:
        return self.code == 0


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def section_and_entries() -> tuple[backlog_parse.Section, list[backlog_parse.Entry]]:
    lines = backlog_parse.read_register(sc.BACKLOG)
    section = backlog_parse.find_section(lines, SLUG)
    entries = backlog_parse.iter_entries(section)
    if not entries:
        raise SystemExit(
            f"FAIL: section resolved by slug {SLUG!r} carries no entries — an empty section does not pass vacuously"
        )
    return section, entries


def subsection(section_text: str, heading: str) -> str:
    if heading not in section_text:
        return ""
    return section_text.split(heading, 1)[1].split("\n#### ", 1)[0]


def check_polarity(
    baseline: str,
    section: backlog_parse.Section,
    entries: Sequence[backlog_parse.Entry],
) -> list[str]:
    problems: list[str] = []
    for entry in entries:
        # iter_entries lets the LAST entry's body run to the end of the section, through the
        # report subsections; an entry's own text stops at the next `####` heading.
        own = []
        for line in entry.body:
            if line.startswith("#### "):
                break
            own.append(line)
        body = "\n".join([entry.heading, *own])
        for ident in SHARED_MEM:
            if re.search(rf"\b{ident}\b", body):
                problems.append(
                    f"{entry.id}: shared-memory identifier {ident!r} inside an id'd entry (sanctioned seam carries a new id)"
                )
        for meth in SUPERSEDED:
            if re.search(rf"\b{meth}\b", body):
                problems.append(
                    f"{entry.id}: cut-sync method {meth!r} inside an id'd entry — expected only the E5 dup-of handoff"
                )
    text = "\n".join(section.lines)
    cleared = subsection(text, "#### ↩︎ Cleared")
    if (
        "Shared-memory communicator trait hierarchy" not in cleared
        or MIRROR not in cleared
    ):
        problems.append(
            "Cleared subsection does not name the shared-memory hierarchy with its mirror citation"
        )
    for ident in SHARED_MEM:
        outside = [
            h
            for h in re.findall(r"^#### (.+)$", text, re.M)
            if ident in subsection(text, f"#### {h}")
            and not h.startswith(READ_ONLY_HEADINGS)
        ]
        if outside:
            problems.append(
                f"shared-memory identifier {ident!r} appears outside the read-only subsections: {outside}"
            )
    if "Superseded cut-sync public methods" not in cleared:
        problems.append("Cleared subsection does not record the cut-sync dup-of")
    handoffs = load("handoffs.json")
    e5 = handoffs.get("E5", {})
    if e5.get("idAssigned") is not None or any(
        r.get("assignedId") is not None for r in handoffs.get("records", [])
    ):
        problems.append("E5 dup-of handoff carries an assigned id")
    if e5.get("dupOf") != "Superseded cut-sync public methods":
        problems.append(f"E5.dupOf is {e5.get('dupOf')!r}")
    src = backlog_parse.git_show(baseline, CUT_SYNC_FILE)
    if src is None:
        problems.append(f"{CUT_SYNC_FILE} missing at {baseline[:8]}")
    else:
        lines = src.splitlines()
        for meth, line in CUT_SYNC.items():
            if line > len(lines) or not re.match(
                rf"\s*pub fn {meth}\b", lines[line - 1]
            ):
                problems.append(
                    f"cut-sync anchor {meth} does not resolve at {CUT_SYNC_FILE}:{line} at {baseline[:8]}"
                )
    return problems


def check_handoff_shapes(baseline: str) -> list[str]:
    problems: list[str] = []
    handoffs, perf, part_i = (
        load("handoffs.json"),
        load("perf-queue.json"),
        load("partI-handoff.json"),
    )
    queue = perf.get("queue") or perf.get("claims") or []
    e10 = handoffs.get("E10", {}).get("block") or []
    if not queue:
        problems.append(
            "perf-queue.json carries no claims (an empty queue must be an explicit decision, not a default)"
        )
    if {q["id"] for q in queue} != {q["entryId"] for q in e10}:
        problems.append(
            "perf-queue.json and handoffs.json E10 disagree on the queued ids"
        )
    for q in [*queue, *e10]:
        ref = q.get("id") or q.get("entryId")
        if q.get("layout") not in LAYOUTS:
            problems.append(
                f"{ref}: layout={q.get('layout')!r} not in {sorted(LAYOUTS)}"
            )
        if not q.get("exercisingCallSites"):
            problems.append(f"{ref}: exercisingCallSites is empty")
        if q.get("measured") is not None or q.get("measuredSeverity") is not None:
            problems.append(
                f"{ref}: carries a measured number; the perf sweep measures, not this station"
            )
        blob = " ".join(str(q.get(k, "")) for k in ("claim", "byteNeutralFixShape"))
        hit = TIMING.search(blob)
        if hit:
            problems.append(
                f"{ref}: timing/ratio number in the queued claim: {hit.group(0)!r}"
            )
        if q.get("status") != "UNMEASURED":
            problems.append(f"{ref}: status={q.get('status')!r}, expected UNMEASURED")
    e9 = handoffs.get("E9", {}).get("block") or {}
    dispositions = e9.get("perFieldDisposition") or {}
    for field in ITEM8_FIELDS:
        if dispositions.get(field) not in DISPOSITIONS:
            problems.append(
                f"E9: StageTemplate.{field} disposition={dispositions.get(field)!r}"
            )
    types_rs = (
        backlog_parse.git_show(baseline, "crates/cobre-solver/src/types.rs") or ""
    )
    type_lines = types_rs.splitlines()
    anchors = {d["field"]: d["anchor"] for d in part_i.get("perFieldDisposition", [])}
    for field in ITEM8_FIELDS:
        anchor = anchors.get(field, "")
        m = re.fullmatch(r"crates/cobre-solver/src/types\.rs:(\d+)", anchor)
        line = int(m.group(1)) if m else 0
        if not (0 < line <= len(type_lines)) or not re.search(
            rf"\bpub {field}\b", type_lines[line - 1]
        ):
            problems.append(
                f"I.3-8: StageTemplate.{field} anchor {anchor!r} does not resolve to `pub {field}` at {baseline[:8]}"
            )
    if e9.get("proposedAlignment") != "advances-1":
        problems.append(f"E9.proposedAlignment={e9.get('proposedAlignment')!r}")
    if len(e9.get("propagationSites") or []) != 3:
        problems.append("E9 does not name the three propagation sites")
    e7 = handoffs.get("E7", {}).get("block") or {}
    for key in ("gateScript", "evadingIdentifier", "patternEvaded"):
        if not e7.get(key):
            problems.append(f"E7: missing {key}")
    if e7.get("changeProposed") is not False:
        problems.append(
            "E7: a gate edit is proposed here; the build-ci station owns the gate"
        )
    return problems


def run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd), cwd=ROOT, capture_output=True, text=True, check=False
    )


def check_blind_spot() -> tuple[int, str, list[str]]:
    problems: list[str] = []
    gate = run(["bash", GATE])
    freeze = (ROOT / FREEZE).read_text(encoding="utf-8").splitlines()
    cfg_test = next(
        (i + 1 for i, text in enumerate(freeze) if text.strip() == "#[cfg(test)]"),
        len(freeze) + 1,
    )
    hits = [
        i + 1 for i, text in enumerate(freeze) if EVADER in text and i + 1 < cfg_test
    ]
    if gate.returncode != 0:
        problems.append(
            f"the genericity gate now reports a violation (exit {gate.returncode}); the finding's premise broke"
        )
    if not hits:
        problems.append(
            f"{EVADER} no longer appears in production code above the #[cfg(test)] at L{cfg_test}; retire the finding"
        )
    elif hits != EXPECTED_HITS:
        problems.append(
            f"{EVADER} production hits moved: {hits} (recorded {EXPECTED_HITS})"
        )
    detail = f"gate exit {gate.returncode}; production hits above cfg(test) L{cfg_test}: {' '.join(map(str, hits))}"
    return gate.returncode, detail, problems


def check_read_only() -> tuple[str, list[str]]:
    porcelain = run(
        ["git", "status", "--porcelain", "--untracked-files=no"]
    ).stdout.splitlines()
    carried = [line for line in porcelain if line[3:] in CARRIED_IN]
    new = [line for line in porcelain if line not in carried and SURFACE_RE.match(line)]
    detail = "carried-in: " + (", ".join(f"`{c[3:]}`" for c in carried) or "none")
    return detail, [f"NEW modified tracked path: {line}" for line in new]


def render(rows: Sequence[Row], carried: str, failures: Sequence[str]) -> str:
    out = [
        MARKER,
        "",
        "| Check | Command | Exit | Result |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        out.append(
            f"| {r.label} | {r.command} | {r.code} | {'PASS' if r.ok else 'FAIL'} |"
        )
    out.append("")
    for r in rows:
        if r.detail:
            out.append(f"- {r.label}: {r.detail}")
    out.append(
        f"- read-only-workspace: {carried} (the shared verifier's row 7 already ties HEAD to the register pin; this row is the porcelain-vs-carried-in restatement)."
    )
    out.append(
        "- perf claims are queued to the perf sweep UNMEASURED, each with a layout and its exercising call sites: see `perf-queue.json`."
    )
    if failures:
        out.append("")
        out.append("Failures:")
        out.extend(f"- {f}" for f in failures)
    out.append("")
    return "\n".join(out)


def main(argv: Sequence[str]) -> int:
    append = "--no-append" not in argv
    baseline = str(load("inventory.json")["baseline"])
    section, entries = section_and_entries()
    failures: list[str] = []
    rows: list[Row] = []

    p = check_polarity(baseline, section, entries)
    failures += p
    rows.append(
        Row(
            "sanctioned-polarity",
            f"shared-memory only in Cleared w/ `{MIRROR}` citation; cut-sync only as the E5 dup-of, no id",
            1 if p else 0,
            f"{len(entries)} entries scanned for the {len(SHARED_MEM)} shared-memory identifiers and the {len(SUPERSEDED)} superseded methods",
        )
    )
    anchor_problems = [x for x in p if "cut-sync anchor" in x]
    rows.append(
        Row(
            "cut-sync-anchors",
            f"`git show {baseline[:8]}:{CUT_SYNC_FILE}` L{'/'.join(str(v) for v in CUT_SYNC.values())}",
            1 if anchor_problems else 0,
            ", ".join(f"{k} L{v}" for k, v in CUT_SYNC.items()),
        )
    )

    h = check_handoff_shapes(baseline)
    failures += h
    rows.append(
        Row(
            "handoff-shape",
            "`python3 plans/architecture-debt-audit/stations/solver-comm/verify-handoffs.py` (E5/E7/E9/E10, perf-queue.json)",
            1 if h else 0,
            "perf rows carry layout ∈ {4t, 2x2}, exercising call sites, no measured number; E9 covers the five StageTemplate fields with resolving types.rs anchors; E7 names gateScript/evadingIdentifier/patternEvaded",
        )
    )

    _gate_exit, detail, b = check_blind_spot()
    failures += b
    rows.append(
        Row(
            "blind-spot",
            f"`bash {GATE}` exit 0 AND `grep -n {EVADER} {FREEZE}` → {' '.join(map(str, EXPECTED_HITS))} (cfg(test) at 244)",
            1 if b else 0,
            detail,
        )
    )

    carried, ro = check_read_only()
    failures += ro
    rows.append(
        Row(
            "read-only-workspace",
            "`git status --porcelain --untracked-files=no` minus the carried-in paths, filtered to the evaluated surfaces",
            1 if ro else 0,
        )
    )

    block = render(rows, carried, failures)
    if append:
        target = HERE / "verification.md"
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
        head = existing.split(MARKER, 1)[0].rstrip() + "\n\n" if existing else ""
        target.write_text(head + block, encoding="utf-8")
    else:
        sys.stdout.write(block)
    for f in failures:
        print(f"FAIL {f}", file=sys.stderr)
    print(
        f"verify-handoffs.py {SLUG}: {'PASS' if not failures else 'FAIL'} ({sum(r.ok for r in rows)}/{len(rows)} station-specific checks)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
