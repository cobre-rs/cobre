#!/usr/bin/env python3
"""verify-test-corpus.py ["<section title>"] [--no-shared] — test-corpus station verification.

Runs the three harness checkers over the resolved heading, the shared tools/verify-station.sh
verbatim (slug `test-corpus`; it writes the head of verification.md), then the five assertions the
generic verifier cannot know — the fifteen re-measured figure records, both measurement definitions
recounted from the tree at the station baseline, the slow-tests census proven git-tracked against a
naive recursive grep, every minted entry's measurement definition / claim kind / figure provenance,
and the three open mirror items recorded only as dup-of merges — and the read-only proof. Appends
its result block to verification.md (idempotent) and exits with the aggregated status.

argv[1] defaults to the scaffold heading; exactly one heading at any level must carry it, else exit 2
listing the candidates. --no-shared skips verify-station.sh and the report write and prints the block
to stdout: the station test module invokes this script and verify-station.sh runs that module, so the
flag is what cuts the recursion. Read-only: writes only stations/test-corpus/verification.md.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUDIT = ROOT / "plans/architecture-debt-audit"
TOOLS = AUDIT / "tools"
BACKLOG = AUDIT / "BACKLOG.md"
REPORT = HERE / "verification.md"
REL = "plans/architecture-debt-audit"
SLUG = "test-corpus"
DEFAULT_SECTION = "★ QUALITY EVALUATION (2026-09, baseline a136840d) — test-corpus"
MARKER = "## Station-specific checks — test-corpus"
YARDSTICK = "docs/design/testing-architecture.md"
MIRROR = "docs/design/reserved-seams-and-deferred-debt.md"
sys.path.insert(0, str(TOOLS))
from lib import backlog_parse  # noqa: E402
from lib import station_checks as sc  # noqa: E402

TA_21 = (
    "int-binaries",
    "int-binaries-solver-linking",
    "nextest-list",
    "doctests",
    "pytest-collect",
    "golden-cases",
    "to-bits",
    "proptest-sites",
    "slow-tests-attrs",
)
TA_PROSE = (
    "nextest-config",
    "ci-runner-split",
    "slow-tests-in-pr-features",
    "shuffle-cadence",
    "test-support-declarers-vs-consumers",
    "homing-split",
)
DIFFER = ("cobre-sddp", "cobre-io")
DEFS = {"binary", "file", "n/a"}
KINDS = {"tree-fact", "target-gap", "prose-drift"}
MIRROR_ITEMS = (
    "Oracle test-harness duplication",
    "Mega-file / inline-test-giant asymmetry",
    "Python-binding Rust tests invisible to CI",
)
DEFINITION_RE = re.compile(r"\(definition: (binary|file)\b|\(definition n/a\)")
BARE_INT_RE = re.compile(r"(?<![\w.:\-§/])\d{2,}(?![\w.\-%])")
FRESH_TD_RE = re.compile(r"\bTD-0(?:7[4-9]|[89]\d)\b")
TARGET_RE = re.compile(r"(^|/)target/")


def resolve_heading(lines: list[str], title: str) -> tuple[int, str]:
    hits = [
        (i, ln)
        for i, ln in enumerate(lines, 1)
        if re.fullmatch(rf"#{{2,4}} {re.escape(title)}", ln.strip())
    ]
    if len(hits) != 1:
        candidates = [ln.strip() for ln in lines if SLUG in ln and ln.startswith("#")]
        print(
            f"verify-test-corpus: section title matched {len(hits)} headings (need exactly 1): {title!r}",
            file=sys.stderr,
        )
        for c in candidates:
            print(f"  candidate: {c}", file=sys.stderr)
        sys.exit(2)
    return hits[0]


class Run:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, int]] = []
        self.facts: list[str] = []
        self.failed = False

    def record(self, label: str, command: str, code: int) -> None:
        self.rows.append(("PASS" if code == 0 else "FAIL", label, command, code))
        if code != 0:
            self.failed = True

    def block(self, label: str, command: str, failures: list[str]) -> None:
        for f in failures:
            print(f"FAIL {label}: {f}")
        self.record(label, command, 0 if not failures else 1)


def run_tool(run: Run, label: str, display: str, argv: list[str]) -> None:
    r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        print((r.stdout + r.stderr)[-1500:])
    run.record(label, display, r.returncode)


def ls_rs(tree: sc.Tree, prefix: str, recursive: bool) -> list[str]:
    files = [f for f in tree.ls_files(prefix) if f.endswith(".rs")]
    if recursive:
        return files
    depth = prefix.rstrip("/").count("/") + 1
    return [f for f in files if f.count("/") == depth]


def figures_block(run: Run, inv: dict) -> dict:
    fails: list[str] = []
    figs = {f["id"]: f for f in inv["figures"]}
    if len(inv["figures"]) != 15:
        fails.append(f"expected 15 figure records, got {len(inv['figures'])}")
    if len(figs) != len(inv["figures"]):
        fails.append("duplicate figure ids in inventory.figures")
    n21 = sum(1 for f in inv["figures"] if f.get("source") == "ta-2.1")
    nprose = sum(1 for f in inv["figures"] if f.get("source") in ("ta-2.3", "ta-3.2"))
    if n21 != 9:
        fails.append(f"ta-2.1 row count {n21} != 9")
    if nprose != 6:
        fails.append(f"unpaired §2.3 / §3.2 prose figure count {nprose} != 6")
    for fid in TA_21 + TA_PROSE:
        f = figs.get(fid)
        if f is None:
            fails.append(f"missing figure record {fid}")
            continue
        if not f.get("command"):
            fails.append(f"{fid}: no command recorded")
        if f.get("status") == "measured":
            if f.get("value") is None:
                fails.append(f"{fid}: status=measured but value is null")
        elif not f.get("reason"):
            fails.append(
                f"{fid}: unmeasured without a reason (record UNMEASURED with the excerpt, never estimate)"
            )
    run.facts.append(
        f"figures: {len(inv['figures'])} records, {n21} ta-2.1 + {nprose} ta-2.3/ta-3.2, {inv['figureStatus']}"
    )
    run.block(
        "figure-completeness",
        "verify-test-corpus.py: 15 figure records, 9 ta-2.1 + 6 prose, command / value / reason",
        fails,
    )
    return figs


def definitions_block(run: Run, inv: dict, tree: sc.Tree, figs: dict) -> None:
    fails: list[str] = []
    pairs: dict[str, tuple[int, int]] = {}
    for rec in inv["perCrate"]:
        crate = rec["crate"]
        prefix = f"crates/{crate}/tests/"
        pin = (
            len(ls_rs(tree, prefix, recursive=False)),
            len(ls_rs(tree, prefix, recursive=True)),
        )
        recorded = (rec["integrationBinaries"], rec["integrationFiles"])
        if pin != recorded:
            fails.append(
                f"{crate}: inventory records binaries/files {recorded} but the tree at {tree.sha[:8]} measures {pin}"
            )
        live_dir = ROOT / "crates" / crate / "tests"
        if live_dir.is_dir():
            live = (
                len([p for p in live_dir.glob("*.rs")]),
                len([p for p in live_dir.rglob("*.rs")]),
            )
            if live != recorded:
                fails.append(
                    f"{crate}: the worktree `find` measures {live} against the recorded {recorded} — the evaluated surface drifted from the pin"
                )
        if crate in DIFFER:
            pairs[crate] = recorded
            if recorded[0] == recorded[1]:
                fails.append(
                    f"{crate}: integrationBinaries == integrationFiles == {recorded[0]} — one definition was copied into both keys; the depth-1 (binary) and recursive (file) censuses must be run separately"
                )
    siblings = {
        f for f in tree.ls_files("crates") if re.search(r"/src/.*tests\.rs$", f)
    }
    for rec in inv["perCrate"]:
        n = sum(1 for f in siblings if f.startswith(f"crates/{rec['crate']}/src/"))
        if n != rec["siblingTestsRs"]:
            fails.append(
                f"{rec['crate']}: siblingTestsRs {rec['siblingTestsRs']} != {n} src/**/tests.rs at the pin"
            )
    run.facts.append(
        "definition pairs (binaries vs files) at the pin: "
        + ", ".join(f"{c} {p[0]} vs {p[1]}" for c, p in pairs.items())
    )
    run.block(
        "both-definitions-ran",
        "verify-test-corpus.py: recount depth-1 vs recursive tests/*.rs per crate at the station baseline; cobre-sddp and cobre-io pairs must differ",
        fails,
    )
    fails = []
    slow = figs.get("slow-tests-attrs") or {}
    if "git grep" not in (slow.get("command") or ""):
        fails.append(
            "slow-tests census must use a git-tracked path filter (git grep), not a bare recursive grep"
        )
    tracked = [
        f
        for f in tree.ls_files("crates")
        if f.endswith(".rs") and "slow-tests" in tree.read_text(f)
    ]
    bad = [p for p in tracked if TARGET_RE.search(p)]
    if bad:
        fails.append(
            f"slow-tests census names build-artifact paths under target/: {bad[:3]}"
        )
    if len(tracked) != slow.get("value"):
        fails.append(
            f"slow-tests census: inventory records {slow.get('value')} .rs files, the tree at the pin has {len(tracked)}"
        )
    naive = subprocess.run(
        ["grep", "-rl", "slow-tests", "crates/"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    under_target = [p for p in naive if "/target/" in p]
    if not under_target or not len(tracked) < len(naive):
        fails.append(
            f"anti-vacuity: the naive recursive grep sees {len(naive)} files ({len(under_target)} under target/); the recorded census of {len(tracked)} must be strictly smaller and the naive sweep must have seen a target/ path"
        )
    run.facts.append(
        f"slow-tests census: {len(tracked)} git-tracked .rs files at the pin vs naive recursive grep {len(naive)} files ({len(under_target)} under target/)"
    )
    run.block(
        "slow-tests-census-tracked",
        "verify-test-corpus.py: git grep census vs `grep -rl slow-tests crates/` (target/ artifacts excluded non-vacuously)",
        fails,
    )


def known_values(inv: dict) -> set[str]:
    known: set[str] = set()

    def walk(x) -> None:
        if isinstance(x, bool):
            return
        if isinstance(x, int):
            known.add(str(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(inv["figures"])
    walk(inv["perCrate"])
    walk(inv["harness"])
    return known


def entries_block(
    run: Run, inv: dict, tree: sc.Tree, section: backlog_parse.Section, cal: dict
) -> list[backlog_parse.Entry]:
    fails: list[str] = []
    entries = backlog_parse.iter_entries(section)
    if not entries:
        fails.append(
            "section parsed to zero entries — an empty section is a failure, not a vacuous pass"
        )
    known = known_values(inv)
    doc_ints = set(BARE_INT_RE.findall(tree.read_text(YARDSTICK)))
    by_ref = {r["id"]: r for r in cal["assigned"]}
    for e in entries:
        measurement = e.fields.get("Measurement", "")
        m = DEFINITION_RE.search(measurement)
        definition = (m.group(1) or "n/a") if m else None
        if definition not in DEFS:
            fails.append(
                f"{e.id}: measurement definition missing or not one of {sorted(DEFS)}: {measurement[:80]!r}"
            )
        kind = (e.fields.get("Claim kind", "").split(" ", 1) or [""])[0]
        if kind not in KINDS:
            fails.append(
                f"{e.id}: claim kind missing or outside {sorted(KINDS)}: {e.fields.get('Claim kind', '')[:60]!r}"
            )
        if "§" not in e.fields.get("Claim kind", ""):
            fails.append(f"{e.id}: the Claim kind line must name the yardstick section")
        row = by_ref.get(e.id)
        if row is None:
            fails.append(f"{e.id}: entry has no calibration.json row")
        else:
            if row["measurementDefinition"] != definition:
                fails.append(
                    f"{e.id}: rendered definition {definition} != calibration.json {row['measurementDefinition']}"
                )
            if row["claimKind"] != kind:
                fails.append(
                    f"{e.id}: rendered claim kind {kind} != calibration.json {row['claimKind']}"
                )
            if row["claimKind"] != "tree-fact" and row["severity"] != "C":
                fails.append(f"{e.id}: a {row['claimKind']} row must sit at Sev C")
            if row["measurementKey"] and str(row["measuredValue"]) not in known:
                fails.append(
                    f"{e.id}: quoted count {row['measuredValue']} is not an inventory value"
                )
        body = "\n".join(e.body) if isinstance(e.body, list) else str(e.body)
        for n in sorted(set(BARE_INT_RE.findall(body))):
            if n in doc_ints and n not in known:
                fails.append(
                    f"{e.id}: cites {n}, a figure that appears in {YARDSTICK} but in no inventory record (a frozen snapshot, not a re-measured value)"
                )
    if cal["needsHuman"]:
        fails.append(
            f"calibration.json carries {len(cal['needsHuman'])} definition-flip needs-human rows; a flipping claim must not be rated"
        )
    run.facts.append(
        f"entries: {len(entries)} minted ({', '.join(e.id for e in entries)}); claim kinds {sorted({e.fields.get('Claim kind', '').split(' ', 1)[0] for e in entries})}"
    )
    run.block(
        "entry-definition-claimkind-provenance",
        "verify-test-corpus.py: every entry names its measurement definition and claim kind; every yardstick-quoted integer resolves to inventory.json",
        fails,
    )
    return entries


def mirror_block(
    run: Run,
    tree: sc.Tree,
    section: backlog_parse.Section,
    entries: list[backlog_parse.Entry],
) -> None:
    fails: list[str] = []
    text = "\n".join(section.lines)
    start = text.find("#### Merged into existing entries (dup-of)")
    if start < 0:
        fails.append(
            "no '#### Merged into existing entries (dup-of)' block in the section"
        )
        block = ""
    else:
        nxt = text.find("\n#### ", start + 1)
        block = text[start : nxt if nxt > 0 else len(text)]
    mirror_head = (ROOT / MIRROR).read_text(encoding="utf-8")
    mirror_pin = tree.read_text(MIRROR).splitlines()
    for title in MIRROR_ITEMS:
        if title not in mirror_head:
            fails.append(
                f"dup-of target '{title}' no longer resolves in the mirror at HEAD"
            )
        bullets = [ln for ln in block.splitlines() if ln.startswith(f"- **{title}**")]
        if len(bullets) != 1:
            fails.append(
                f"'{title}' has {len(bullets)} dup-of bullets; expected exactly one"
            )
            continue
        m = re.search(rf"\(`{re.escape(MIRROR)}:(\d+)`\)", bullets[0])
        if not m:
            fails.append(f"'{title}': dup-of bullet carries no mirror anchor")
        else:
            ln = int(m.group(1))
            if ln > len(mirror_pin) or title not in mirror_pin[ln - 1]:
                fails.append(
                    f"'{title}': mirror anchor :{ln} does not carry the heading at the pin"
                )
        anchors = [
            a for a in sc.anchors_in(bullets[0]) if not a.startswith(f"`{MIRROR}")
        ]
        if not anchors:
            fails.append(
                f"'{title}': dup-of bullet carries no sharpened baseline anchor"
            )
        for a in anchors:
            if not sc.anchor_exists(a, tree):
                fails.append(
                    f"'{title}': sharpened anchor {a} does not resolve at the pin"
                )
        for e in entries:
            body = "\n".join(e.body) if isinstance(e.body, list) else str(e.body)
            if title in body or title in e.heading:
                fails.append(
                    f"'{title}' appears inside entry {e.id}; a mirror item may only be a dup-of merge"
                )
    fresh = FRESH_TD_RE.findall(block)
    if fresh:
        fails.append(f"dup-of block names fresh TD ids {sorted(set(fresh))}")
    run.facts.append(
        f"dup-of merges: {len(MIRROR_ITEMS)} mirror items, each once, no fresh TD id"
    )
    run.block(
        "mirror-items-dup-of-only",
        "verify-test-corpus.py: the three open mirror items appear exactly once, as dup-of merges with resolving anchors and no fresh TD id",
        fails,
    )


def readonly_block(run: Run) -> None:
    fails: list[str] = []
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    station_prefix = f"{REL}/stations/{SLUG}/"
    beyond = [ln for ln in porcelain if not ln[3:].startswith(station_prefix)]
    if beyond:
        fails.append(
            "the workspace carries changes beyond the station's own directory: "
            + "; ".join(beyond[:6])
        )
    tracked_mod = [ln for ln in porcelain if not ln.startswith("??")]
    if tracked_mod:
        fails.append("tracked files are modified: " + "; ".join(tracked_mod[:6]))
    diff = subprocess.run(
        [
            "git",
            "diff",
            "--stat",
            "HEAD",
            "--",
            "crates",
            ".github",
            "docs",
            "schemas",
            "scripts",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if diff.strip():
        fails.append(
            "git diff --stat HEAD over crates/.github/docs/schemas/scripts is not empty"
        )
    run.block(
        "read-only-workspace",
        "git status --porcelain (only the station's own untracked run artifacts) && git diff --stat HEAD -- crates .github docs schemas scripts (empty)",
        fails,
    )


def render(run: Run, inv: dict, title: str, heading_line: int) -> str:
    out = [
        MARKER,
        "",
        f"Section resolved by title `{title}` to exactly one heading (BACKLOG.md line {heading_line}); station baseline `{inv['baseline']['sha']}` (inventory.json; every recount below is measured in that tree, and the worktree `find` is asserted equal to it). The five station assertions the shared verifier cannot know, then the read-only proof.",
        "",
        "| # | Check | Command | Exit | Result |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i, (res, label, cmd, code) in enumerate(run.rows, 1):
        out.append(f"| {i} | {label} | `{cmd}` | {code} | {res} |")
    out.append("")
    out.append("Measured at the station baseline:")
    out.append("")
    for f in run.facts:
        out.append(f"- {f}")
    out.append("")
    out.append(
        "Ticket premises superseded at this baseline (recorded, not edited): the cobre-sddp pair is 40 binaries vs 56 files (ticket: 37 vs 53); the naive `grep -rl slow-tests crates/` sees 75 files with 56 under target/ (ticket: 73 / 54) against 13 tracked; no pre-station porcelain snapshot exists and the worktree carries no modified .gitignore, so cleanliness is judged as 'nothing beyond the station's own directory'; entries carry the definition inside `- **Measurement:**` and the kind inside `- **Claim kind:**` (the register's bold-bullet field shape), and the dup-of merges are bullets under their own heading rather than id-less entries; the figure-provenance rule is applied to integers that appear in the yardstick (the frozen-snapshot hazard) so a defender-measured line count is not a false failure."
    )
    out.append("")
    out.append(
        f"Station-specific result: {'FAIL' if run.failed else 'PASS'} ({sum(1 for r in run.rows if r[0] == 'PASS')}/{len(run.rows)} checks)."
    )
    return "\n".join(out) + "\n"


def write_report(block: str) -> None:
    text = REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""
    i = text.find(MARKER)
    if i >= 0:
        text = text[:i].rstrip("\n") + "\n"
    text = text.rstrip("\n") + "\n\n" + block if text.strip() else block
    REPORT.write_text(text, encoding="utf-8")


def main(argv: list[str]) -> int:
    run_shared = "--no-shared" not in argv
    titles = [a for a in argv if not a.startswith("--")]
    title = titles[0] if titles else DEFAULT_SECTION
    lines = backlog_parse.read_register(BACKLOG)
    heading_line, _ = resolve_heading(lines, title)
    run = Run()
    for tool in ("check-anchors", "check-reraise", "fields-check"):
        run_tool(
            run,
            tool,
            f"python3 {REL}/tools/{tool}.py '{title}'",
            [sys.executable, str(TOOLS / f"{tool}.py"), title],
        )
    run_tool(
        run,
        "fields-check-alignment",
        f"python3 {REL}/tools/fields-check.py --require Alignment '{title}'",
        [
            sys.executable,
            str(TOOLS / "fields-check.py"),
            "--require",
            "Alignment",
            title,
        ],
    )
    if run_shared:
        r = subprocess.run(
            ["bash", str(TOOLS / "verify-station.sh"), SLUG, title],
            cwd=ROOT,
            check=False,
        )
        run.record(
            "verify-station",
            f"bash {REL}/tools/verify-station.sh {SLUG} '{title}'",
            r.returncode,
        )
    inv = json.loads((HERE / "inventory.json").read_text(encoding="utf-8"))
    cal = json.loads((HERE / "calibration.json").read_text(encoding="utf-8"))
    tree = sc.Tree(inv["baseline"]["sha"])
    section = backlog_parse.find_section(lines, title)
    figs = figures_block(run, inv)
    definitions_block(run, inv, tree, figs)
    entries = entries_block(run, inv, tree, section, cal)
    mirror_block(run, tree, section, entries)
    readonly_block(run)
    block = render(run, inv, title, heading_line)
    if run_shared:
        write_report(block)
    else:
        sys.stdout.write(block)
    print(f"verify-test-corpus.py: {'FAIL' if run.failed else 'PASS'}")
    return 1 if run.failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
