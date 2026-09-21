#!/usr/bin/env python3
"""The four bespoke per-station checks that go beyond the three harness checkers.

verify-station.sh orchestrates the three checkers (check-anchors, check-reraise,
fields-check) and this module's four subcommands; test_station.py imports the pure
functions below and exercises them on tampered inputs, so a regression in a check is
caught rather than passing silently. Read-only: nothing here writes a file.

Subcommands (each prints its findings and exits 0 clean / 1 violation / 3 no section):
  register   <audit-dir> <station-slug>     Alignment vocabulary + do-not-touch denylist + non-empty
  inventory  <inventory.json> <repo-root>   frozen census == the .rs (gate census: .sh/.py) set at the census baseline
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
MIRROR = bp.MIRROR
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
MODULE_SUFFIXES = (".rs",)
GATE_SUFFIXES = (".sh", ".py")


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
    """Every path the inventory's frozen census names, once each.

    Five census shapes are accepted so the verifier serves every station unchanged: a
    top-level `files[]` whose rows each carry a `path` (single- or multi-crate; wins when
    present), the nested `src.files[]` shape of a census split into src/ and tests/ roots
    (the src rows are the module census; `src.root` names the tree root), the
    multi-crate `crates{}.modules[]` shape (a `file` module is one .rs path; a `directory`
    module contributes the .rs names in its `files[]`), and the crate-less `gates.files[]`
    shape of a station whose corpus is the scripts/ci gate set (each row carries a `path`;
    `gates.root` names the tree root and the census is its executable set, see
    `census_suffixes`).
    """
    if "files" in inventory:
        return [f["path"] for f in inventory["files"]]
    if "files" in inventory.get("src", {}):
        return [f["path"] for f in inventory["src"]["files"]]
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
    if "files" in inventory.get("gates", {}):
        return [f["path"] for f in inventory["gates"]["files"]]
    raise KeyError(
        "inventory carries neither `files`, `src.files`, `crates` nor `gates.files`"
    )


def crate_src_roots(inventory: dict[str, Any]) -> list[str]:
    """The tree roots the census covers: `crates{}` keys, `crates[]` rows (each carrying
    its `srcRoot`, or `name` when the root is the default `crates/<name>/src`), the
    nested `src.root`, the gate census's `gates.root`, or the single `crate`."""
    if "crates" in inventory:
        crates = inventory["crates"]
        if isinstance(crates, list):
            return [c.get("srcRoot") or f"crates/{c['name']}/src" for c in crates]
        return [f"crates/{crate}/src" for crate in crates]
    if "root" in inventory.get("src", {}):
        return [inventory["src"]["root"]]
    if "root" in inventory.get("gates", {}):
        return [inventory["gates"]["root"]]
    return [f"crates/{inventory['crate']}/src"]


def census_suffixes(inventory: dict[str, Any]) -> tuple[str, ...]:
    """The file suffixes the census enumerates under its roots: a gate census is the
    executable .sh/.py set (data files such as the allowlist are not gates); every
    module census is the .rs set."""
    return GATE_SUFFIXES if "gates" in inventory else MODULE_SUFFIXES


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

    A porcelain line is `XY <path>`; the carried-in .gitignore, the ID-free mirror the
    evaluation itself writes, and everything outside the evaluated surfaces (i.e. the
    station's own plans/ writes) are exempt.
    """

    def on_surface(path: str) -> bool:
        if path == MIRROR:
            return False
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
    return worktree, [p for p in committed if p.strip() and p.strip() != MIRROR]


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


def baseline_rs_files(
    root: pathlib.Path,
    baseline: str,
    roots: list[str],
    suffixes: tuple[str, ...] = MODULE_SUFFIXES,
) -> list[str]:
    """Every tracked file carrying one of `suffixes` under `roots` in the tree at
    `baseline` (the station's evaluated tree); the default is the .rs module set."""
    out: list[str] = []
    for src_root in roots:
        listing = _git(root, "ls-tree", "-r", "--name-only", baseline, "--", src_root)
        if listing.returncode != 0:
            raise RuntimeError(
                listing.stderr.strip() or f"{src_root} at {baseline[:8]}"
            )
        out += [p for p in listing.stdout.split() if p.endswith(suffixes)]
    return out


def count_census_rows(inventory: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The per-crate COUNT census of a station whose corpus is the test tree itself
    (`perCrate[]` rows carrying `integrationBinaries` / `integrationFiles` /
    `siblingTestsRs`, no path list) — the sixth census shape; None for the five
    path-listing shapes `reconstruct_listed` handles."""
    rows = inventory.get("perCrate")
    if not isinstance(rows, list) or any(
        k in inventory for k in ("files", "src", "crates", "gates")
    ):
        return None
    return rows


def census_baseline(inventory: dict[str, Any]) -> str:
    """The census tree: a bare sha, or the `{sha, describe, …}` record a re-measurement writes."""
    baseline = inventory["baseline"]
    return str(baseline["sha"] if isinstance(baseline, dict) else baseline)


def count_census_diff(
    root: pathlib.Path, baseline: str, rows: list[dict[str, Any]]
) -> list[str]:
    """One line per crate whose recorded test-file counts differ from the tree at `baseline`:
    depth-1 `crates/<c>/tests/*.rs` are the linked integration binaries, the recursive
    `tests/**/*.rs` set is the file reading, `src/**/tests.rs` the extracted siblings."""
    bad: list[str] = []
    for rec in rows:
        crate = rec["crate"]
        tests = f"crates/{crate}/tests"
        src = f"crates/{crate}/src"
        listing = _git(root, "ls-tree", "-r", "--name-only", baseline, "--", tests, src)
        if listing.returncode != 0:
            raise RuntimeError(listing.stderr.strip() or f"{crate} at {baseline[:8]}")
        paths = listing.stdout.split()
        rs_tests = [p for p in paths if p.startswith(tests + "/") and p.endswith(".rs")]
        measured = {
            "integrationBinaries": sum(1 for p in rs_tests if p.count("/") == 3),
            "integrationFiles": len(rs_tests),
            "siblingTestsRs": sum(
                1 for p in paths if p.startswith(src + "/") and p.endswith("tests.rs")
            ),
        }
        for key, value in measured.items():
            if key in rec and rec[key] != value:
                bad.append(
                    f"{crate}: {key} recorded {rec[key]}, the tree at {baseline[:8]} measures {value}"
                )
    return bad


def _cmd_inventory(inventory_path: pathlib.Path, root: pathlib.Path) -> int:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    rows = count_census_rows(inventory)
    if rows is not None:
        baseline = census_baseline(inventory)
        bad = count_census_diff(root, baseline, rows)
        for line in bad:
            print(f"FAIL inventory count-census at {baseline[:8]}: {line}")
        if bad:
            return EXIT_VIOLATION
        print(
            f"inventory and the tree at {baseline[:8]} agree on the test-file census of "
            f"{len(rows)} crates (integration binaries / files, sibling tests.rs)"
        )
        return EXIT_OK
    listed = reconstruct_listed(inventory)
    if len(listed) != len(set(listed)):
        dupes = sorted({p for p in listed if listed.count(p) > 1})
        print(f"FAIL inventory self-consistency: duplicate census paths {dupes}")
        return EXIT_VIOLATION
    roots = crate_src_roots(inventory)
    suffixes = census_suffixes(inventory)
    baseline = str(inventory["baseline"])
    tree = baseline_rs_files(root, baseline, roots, suffixes)
    absent, unlisted = inventory_diff(listed, tree)
    if absent or unlisted:
        print(
            f"FAIL inventory-set-equality at {baseline[:8]}: {len(absent)} listed-but-absent / "
            f"{len(unlisted)} present-but-unlisted"
        )
        for path in absent:
            print(f"  listed-but-absent  {path}")
        for path in unlisted:
            print(f"  present-but-unlisted {path}")
        return EXIT_VIOLATION
    print(
        f"inventory and the tree at {baseline[:8]} agree on {len(set(tree))} "
        f"{'/'.join(suffixes)} files under {', '.join(roots)}"
    )
    return EXIT_OK


def _cmd_genericity(root: pathlib.Path, handoff_path: pathlib.Path) -> int:
    gate = root / "scripts" / "ci" / "check-infra-genericity.sh"
    returncode = subprocess.run(
        ["bash", str(gate)], cwd=root, capture_output=True, text=True, check=False
    ).returncode
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    # A handoff lists its Part-I rows as `dispositions[]` or, for a single-item station, `entries[]`.
    rows = handoff.get("dispositions") or handoff.get("entries") or []
    owns_item6 = any(d.get("partIRef") == "I.3-6" for d in rows)
    disposition = next(
        (d.get("disposition") for d in rows if d.get("partIRef") == "I.3-6"),
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
