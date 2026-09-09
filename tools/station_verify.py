#!/usr/bin/env python3
"""The four bespoke per-station checks that go beyond the three harness checkers.

verify-station.sh orchestrates the three checkers (check-anchors, check-reraise,
fields-check) and this module's four subcommands; test_station.py imports the pure
functions below and exercises them on tampered inputs, so a regression in a check is
caught rather than passing silently. Read-only: nothing here writes a file.

Subcommands (each prints its findings and exits 0 clean / 1 violation / 3 no section):
  register   <audit-dir> <station-slug>     Alignment vocabulary + do-not-touch denylist + non-empty
  inventory  <inventory.json> <repo-root>   frozen module census == find over the crate src roots
  genericity <repo-root> <partI-handoff>    the genericity gate + EXCLUDED_FILES=() + I.3-6 disposition
  readonly   <repo-root> <baseline-sha>     no station write to any evaluated surface (worktree or commit)
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from lib import backlog_parse as bp  # noqa: E402

EXIT_OK = 0
EXIT_VIOLATION = 1
EXIT_SECTION_NOT_FOUND = 3

ALIGNMENT_VOCAB = frozenset(
    {"advances-0a", "advances-0b", "advances-1", "neutral", "conflicts"}
)
HARD_STATUSES = frozenset(
    {"retracted", "refuted", "wontfix", "deferred", "do-not-touch"}
)
FINDING_ID_RE = re.compile(r"\b(?:CD|PD|OD|TD)-\d{3}\b")
MIRROR = "docs/design/reserved-seams-and-deferred-debt.md"
SEAM_SECTION = "Reserved-seam register"
EVALUATED_ROOTS = (
    "crates",
    "docs",
    "scripts",
    ".github",
    "schemas",
    "examples",
    "tests",
)
EVALUATED_FILES = ("Cargo.toml", "Cargo.lock")


def alignment_violations(entries: list[bp.Entry]) -> list[tuple[str, str, str]]:
    """(id, 'alignment', offending-value) for every entry whose Alignment is off-vocabulary."""
    out: list[tuple[str, str, str]] = []
    for entry in entries:
        value = entry.fields.get("Alignment", "").split("(", 1)[0].strip().strip("`")
        if value not in ALIGNMENT_VOCAB:
            out.append((entry.id, "alignment", repr(entry.fields.get("Alignment"))))
    return out


def hard_reasons(lines: list[str]) -> dict[str, str]:
    """Finding id -> the do-not-touch reason (retracted / refuted / deferred / wontfix / do-not-touch).

    The register's do-not-touch list marks its ids `do-not-touch`; the finer reason is
    the upper-case marker in the id's own heading, so a re-raise names why the id is off-limits.
    """
    statuses = bp.register_findings(lines)
    deny = {fid: st for fid, st in statuses.items() if st in HARD_STATUSES}
    markers = (
        ("RETRACTED", "retracted"),
        ("REFUTED", "refuted"),
        ("DEFERRED", "deferred"),
        ("WONTFIX", "wontfix"),
    )
    for raw in lines:
        hit = bp.ENTRY_RE.match(raw)
        if not hit or hit.group("id") not in deny:
            continue
        for token, name in markers:
            if token in raw:
                deny[hit.group("id")] = name
                break
    return deny


def seam_anchor_keys(baseline: str) -> tuple[frozenset[str], frozenset[str]]:
    """Sanctioned (paths, symbols) drawn only from the mirror's Reserved-seam register."""
    blob = bp.git_show(baseline, MIRROR)
    if blob is None:
        return frozenset(), frozenset()
    try:
        section = bp.find_section(blob.splitlines(), SEAM_SECTION)
    except bp.SectionNotFound:
        return frozenset(), frozenset()
    paths: set[str] = set()
    syms: set[str] = set()
    for anchor in bp.parse_anchors(section.lines):
        paths.add(anchor.path)
        if anchor.symbol:
            syms.add(anchor.symbol)
    return frozenset(paths), frozenset(syms)


def denylist_violations(
    entries: list[bp.Entry],
    deny: dict[str, str],
    seam_paths: frozenset[str],
    seam_syms: frozenset[str],
) -> list[tuple[str, str, str]]:
    """(id, kind, detail) for entries that re-raise a hard-retired id or a ratified reserved seam.

    A hard-retired id mentioned in an entry is a re-raise unless the entry's Re-raise-of
    line names it. A seam anchor (same path, and same symbol when the seam is symbol-scoped)
    is a re-raise unless the entry body cites the seam as sanctioned/reserved.
    """
    out: list[tuple[str, str, str]] = []
    for entry in entries:
        justified = set(FINDING_ID_RE.findall(entry.fields.get("Re-raise-of", "")))
        body = "\n".join([entry.heading, *entry.body])
        for fid in set(FINDING_ID_RE.findall(body)):
            if fid != entry.id and fid in deny and fid not in justified:
                out.append((entry.id, "re-raise", f"{fid} is {deny[fid]}"))
        cited = "sanction" in body.lower() or "reserved seam" in body.lower()
        for anchor in bp.parse_anchors([entry.fields.get("Anchors", "")]):
            if (
                anchor.path in seam_paths
                and (not anchor.symbol or anchor.symbol in seam_syms)
                and not cited
            ):
                out.append(
                    (
                        entry.id,
                        "reserved-seam",
                        f"{anchor.raw} is a ratified reserved seam; cite it as sanctioned",
                    )
                )
    return out


def register_violations(
    entries: list[bp.Entry],
    deny: dict[str, str],
    seam_paths: frozenset[str],
    seam_syms: frozenset[str],
) -> list[tuple[str, str, str]]:
    """Every register-integrity violation in one section: empty, off-vocabulary Alignment, or denylist re-raise."""
    bad: list[tuple[str, str, str]] = []
    if not entries:
        bad.append(("<section>", "empty", "section parsed to zero entries"))
    bad += alignment_violations(entries)
    bad += denylist_violations(entries, deny, seam_paths, seam_syms)
    return bad


def reconstruct_listed(inventory: dict[str, Any]) -> list[str]:
    """Every .rs path the inventory's frozen census names, once each.

    Two census shapes are accepted so the verifier serves every station unchanged:
    the multi-crate `crates{}.modules[]` shape (a `file` module is one .rs path; a
    `directory` module contributes the .rs names in its `files[]`), and the single-crate
    `files[]` shape whose rows each carry a `path`.
    """
    if "crates" in inventory:
        listed: list[str] = []
        for meta in inventory["crates"].values():
            for module in meta["modules"]:
                if module["kind"] == "file":
                    listed.append(module["path"])
                else:
                    listed += [
                        f"{module['path']}/{name}"
                        for name in module.get("files", [])
                        if name.endswith(".rs")
                    ]
        return listed
    return [f["path"] for f in inventory["files"]]


def crate_src_roots(inventory: dict[str, Any]) -> list[str]:
    if "crates" in inventory:
        return [f"crates/{crate}/src" for crate in inventory["crates"]]
    return [f"crates/{inventory['crate']}/src"]


def inventory_diff(listed: list[str], tree: list[str]) -> tuple[list[str], list[str]]:
    """(listed-but-absent, present-but-unlisted) between the inventory census and the tree."""
    listed_set, tree_set = set(listed), set(tree)
    return sorted(listed_set - tree_set), sorted(tree_set - listed_set)


def genericity_premises(
    gate_returncode: int,
    gate_text: str,
    disposition: str | None,
    owns_item6: bool = True,
) -> list[str]:
    """Broken-premise messages behind Part-I item 6; empty when all hold.

    The genericity gate + EXCLUDED_FILES=() premises are checked for every station.
    The I.3-6 disposition premise is checked only for the station that owns item 6
    (`owns_item6`); a station whose Part-I refs do not include I.3-6 asserts the gate
    alone. `owns_item6` defaults True so the three-arg core-io call is unchanged.
    """
    broke: list[str] = []
    if gate_returncode != 0:
        broke.append(f"check-infra-genericity.sh exited {gate_returncode}")
    if "EXCLUDED_FILES=()" not in gate_text:
        broke.append(
            "EXCLUDED_FILES=() is no longer present: the output/policy exemption returned"
        )
    if owns_item6 and disposition not in {"retire", "sharpen"}:
        broke.append(f"I.3-6 disposition is {disposition!r}, not retire/sharpen")
    return broke


def readonly_offenders(
    worktree_porcelain: list[str], committed: list[str]
) -> tuple[list[str], list[str]]:
    """(worktree offenders, committed offenders) touching an evaluated surface.

    A porcelain line is `XY <path>`; the carried-in .gitignore and everything outside the
    evaluated surfaces (i.e. the station's own plans/ writes) are exempt.
    """

    def on_surface(path: str) -> bool:
        return path in EVALUATED_FILES or any(
            path.startswith(f"{root}/") for root in EVALUATED_ROOTS
        )

    worktree: list[str] = []
    for line in worktree_porcelain:
        if not line.strip():
            continue
        path = line[3:].split(" -> ")[-1].strip()
        if path != ".gitignore" and on_surface(path):
            worktree.append(line.strip())
    return worktree, [p for p in committed if p.strip()]


def _git(root: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )


def _cmd_register(audit: pathlib.Path, station: str) -> int:
    lines = bp.read_register(audit / "BACKLOG.md")
    try:
        section = bp.find_section(lines, station)
    except bp.SectionNotFound:
        print(f"FAIL section-not-found: {station!r} resolves to no register heading")
        return EXIT_SECTION_NOT_FOUND
    entries = bp.iter_entries(section)
    seam_paths, seam_syms = seam_anchor_keys(bp.parse_baseline(lines))
    bad = register_violations(entries, hard_reasons(lines), seam_paths, seam_syms)
    for entry_id, kind, detail in bad:
        print(f"FAIL {entry_id} {kind} {detail}")
    print(f"checked {len(entries)} entries, {len(bad)} violation(s)")
    return EXIT_VIOLATION if bad else EXIT_OK


def _cmd_inventory(inventory_path: pathlib.Path, root: pathlib.Path) -> int:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    listed = reconstruct_listed(inventory)
    if len(listed) != len(set(listed)):
        dupes = sorted({p for p in listed if listed.count(p) > 1})
        print(f"FAIL inventory self-consistency: duplicate module paths {dupes}")
        return EXIT_VIOLATION
    roots = crate_src_roots(inventory)
    tree = _git(root, "ls-files", "--", *(f"{r}/*.rs" for r in roots)).stdout.split()
    if not tree:
        tree = subprocess.run(
            ["find", *roots, "-name", "*.rs"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    absent, unlisted = inventory_diff(listed, tree)
    if absent or unlisted:
        print(
            f"FAIL inventory-set-equality: {len(absent)} listed-but-absent / "
            f"{len(unlisted)} present-but-unlisted"
        )
        for path in absent:
            print(f"  listed-but-absent  {path}")
        for path in unlisted:
            print(f"  present-but-unlisted {path}")
        return EXIT_VIOLATION
    print(
        f"inventory and tree agree on {len(set(tree))} .rs files under {', '.join(roots)}"
    )
    return EXIT_OK


def _cmd_genericity(root: pathlib.Path, handoff_path: pathlib.Path) -> int:
    gate = root / "scripts" / "ci" / "check-infra-genericity.sh"
    returncode = subprocess.run(
        ["bash", str(gate)], cwd=root, capture_output=True, text=True, check=False
    ).returncode
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    owns_item6 = any(d.get("partIRef") == "I.3-6" for d in handoff["dispositions"])
    disposition = next(
        (
            d.get("disposition")
            for d in handoff["dispositions"]
            if d.get("partIRef") == "I.3-6"
        ),
        None,
    )
    broke = genericity_premises(
        returncode, gate.read_text(encoding="utf-8"), disposition, owns_item6
    )
    for line in broke:
        print(f"FAIL item-6 premise: {line}")
    print(f"item-6 premises checked; {len(broke)} broken")
    return EXIT_VIOLATION if broke else EXIT_OK


def _cmd_readonly(root: pathlib.Path, baseline: str) -> int:
    worktree = _git(
        root, "status", "--porcelain", "--untracked-files=no"
    ).stdout.splitlines()
    committed = _git(
        root,
        "diff",
        "--name-only",
        baseline,
        "HEAD",
        "--",
        *EVALUATED_ROOTS,
        *EVALUATED_FILES,
    ).stdout.splitlines()
    wt_off, cm_off = readonly_offenders(worktree, committed)
    if wt_off:
        print("FAIL read-only-workspace: worktree modified tracked source:")
        for line in wt_off:
            print(f"  {line}")
    if cm_off:
        print(
            f"FAIL read-only-workspace: commits since {baseline} touch evaluated surfaces:"
        )
        for path in cm_off:
            print(f"  {path}")
    if not wt_off and not cm_off:
        print("read-only: no station write to any evaluated surface")
    return EXIT_VIOLATION if (wt_off or cm_off) else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(__doc__, file=sys.stderr)
        return EXIT_VIOLATION
    sub, rest = args[0], args[1:]
    if sub == "register":
        return _cmd_register(pathlib.Path(rest[0]), rest[1])
    if sub == "inventory":
        return _cmd_inventory(pathlib.Path(rest[0]), pathlib.Path(rest[1]))
    if sub == "genericity":
        return _cmd_genericity(pathlib.Path(rest[0]), pathlib.Path(rest[1]))
    if sub == "readonly":
        return _cmd_readonly(pathlib.Path(rest[0]), rest[1])
    print(f"unknown subcommand: {sub!r}", file=sys.stderr)
    return EXIT_VIOLATION


if __name__ == "__main__":
    sys.exit(main())
