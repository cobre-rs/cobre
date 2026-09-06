"""Shared assertions for the per-station test modules under stations/<slug>/tests/.

Every station imports this instead of re-implementing the checker runner, the anchor
resolver, the clean-tree probe and the loc-stats.sh line classifier.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import unittest

from lib import backlog_parse

REPO = pathlib.Path(__file__).resolve().parents[4]
AUDIT = REPO / "plans" / "architecture-debt-audit"
TOOLS = AUDIT / "tools"
STATIONS = AUDIT / "stations"
BACKLOG = AUDIT / "BACKLOG.md"

# `path`, `path:line`, `path:line-line`, `path::symbol` — a superset of backlog_parse.ANCHOR_RE
# that also accepts a line range.
ANCHOR_LOC_RE = re.compile(
    r"`(?P<path>(?:crates|scripts|docs|plans|schemas|examples|tests|\.github|\.claude)/[\w./-]+?"
    r"\.(?:rs|toml|md|json|fbs|sh|py|yml|yaml|txt|csv|lock))"
    r"(?:::(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)|:(?P<line>\d+)(?:-(?P<line_to>\d+))?)?`"
)
_DECL_RE = r"^\s*(pub(\([^)]*\))?\s+)?(async\s+)?(unsafe\s+)?(fn|struct|enum|trait|type|const|static|mod|impl|macro_rules!)\s+{sym}\b"
_FIELD_RE = r"^\s*(pub(\([^)]*\))?\s+)?{sym}\s*:"
_SHELL_RE = r"^\s*(export\s+)?{sym}=|^\s*{sym}\s*\(\)\s*\{{"


def run_checker(tool: str, section_title: str, *extra: str) -> int:
    """Exit code of tools/<tool> run over one BACKLOG section from the repo root."""
    return subprocess.run(["python3", str(TOOLS / tool), section_title, *extra], cwd=REPO,
                          capture_output=True, text=True, check=False).returncode


def load_json(path: pathlib.Path | str):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def split_anchor(anchor: str) -> tuple[str, str | None, int | None, int | None]:
    hit = ANCHOR_LOC_RE.fullmatch(anchor.strip())
    if hit is None:
        raise ValueError(f"not an anchor: {anchor!r}")
    line = int(hit.group("line")) if hit.group("line") else None
    line_to = int(hit.group("line_to")) if hit.group("line_to") else None
    return hit.group("path"), hit.group("symbol"), line, line_to


def symbol_resolves(path: pathlib.Path, symbol: str) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    sym = re.escape(symbol)
    patterns = [_DECL_RE, _FIELD_RE] if path.suffix == ".rs" else [_SHELL_RE, _DECL_RE, _FIELD_RE]
    return any(re.search(p.format(sym=sym), text, re.M) for p in patterns)


def anchor_exists(anchor: str) -> bool:
    """True when the anchor's path exists in the tree and its symbol or line range resolves."""
    try:
        rel, symbol, line, line_to = split_anchor(anchor)
    except ValueError:
        return False
    path = REPO / rel
    if not path.is_file():
        return False
    if symbol is not None:
        return symbol_resolves(path, symbol)
    if line is not None:
        count = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
        if line_to is None:
            return 1 <= line <= count
        return 1 <= line <= line_to <= count
    return True


def anchors_in(text: str) -> list[str]:
    return [m.group(0) for m in ANCHOR_LOC_RE.finditer(text)]


def tracked_modifications() -> list[str]:
    """`git status --porcelain` over tracked files, minus the carried-in .gitignore line."""
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    return [l for l in out.splitlines() if l.strip() != "M .gitignore"]


def section_entries(section_title: str) -> list[backlog_parse.Entry]:
    lines = backlog_parse.read_register(BACKLOG)
    return backlog_parse.iter_entries(backlog_parse.find_section(lines, section_title))


def rs_files(path: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path.rglob("*.rs")) if path.is_dir() else [path]


def raw_lines(path: pathlib.Path | str) -> int:
    """`wc -l` over every .rs file under the path (the file itself when it is a file)."""
    return sum(f.read_bytes().count(b"\n") for f in rs_files(REPO / path))


def _is_test_file(rel: str) -> bool:
    base = rel.rsplit("/", 1)[-1]
    return ("/tests/" in f"/{rel}" or "/benches/" in f"/{rel}"
            or base in {"tests.rs", "test_support.rs"})


def classify_lines(rel: str, text: str) -> tuple[int, int]:
    """(non-test physical lines, non-test code lines) under the scripts/loc-stats.sh rule."""
    ftype = _is_test_file(rel)
    in_test = pend = False
    tind = -1
    phys = code = 0
    for raw in text.splitlines():
        t = raw.strip(" \t")
        indent = len(raw) - len(raw.lstrip(" "))
        tl = False
        if ftype:
            tl = True
        elif in_test:
            tl = True
            if t.startswith("}") and indent == tind:
                in_test = False
        elif pend:
            tl = True
            if "{" in raw:
                in_test, pend = True, False
            elif t.endswith(";"):
                pend = False
        elif t.startswith("#[cfg") and re.search(r"\(test[,)]", t) and not re.search(r"not\s*\(test", t):
            tl = True
            tind = indent
            if "{" in raw:
                in_test = True
            elif not t.endswith(";"):
                pend = True
        if not tl:
            phys += 1
            if t and not t.startswith("//"):
                code += 1
    return phys, code


def non_test_lines(path: pathlib.Path | str, code: bool = False) -> int:
    """Non-test physical (or code) lines under the loc-stats.sh rule, summed over the path."""
    total = 0
    for f in rs_files(REPO / path):
        rel = f.relative_to(REPO).as_posix()
        phys, cod = classify_lines(rel, f.read_text(encoding="utf-8", errors="replace"))
        total += cod if code else phys
    return total


def loc_stats(crate: str) -> dict[str, int]:
    """One crate's row of `scripts/loc-stats.sh --csv` (files, prod_code, prod_all, ...)."""
    out = subprocess.run(["bash", "scripts/loc-stats.sh", "--csv"], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout
    header = out.splitlines()[0].split(",")
    for row in out.splitlines()[1:]:
        cells = row.split(",")
        if cells[0] == crate:
            return {k: int(v) for k, v in zip(header[1:], cells[1:])}
    raise KeyError(crate)


class StationCase(unittest.TestCase):
    SLUG = ""
    SECTION_TITLE = ""
    STATION_DIR: pathlib.Path | None = None

    @classmethod
    def station_dir(cls) -> pathlib.Path:
        return cls.STATION_DIR or STATIONS / cls.SLUG

    def artifact(self, name: str) -> pathlib.Path:
        return self.station_dir() / name
