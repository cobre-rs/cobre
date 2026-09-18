"""Shared assertions for the per-station test modules under stations/<slug>/tests/.

Every station imports this instead of re-implementing the checker runner, the anchor
resolver, the clean-tree probe and the loc-stats.sh line classifier.

A station measures the tree it evaluated — its `inventory.json` `baseline` — never the
worktree: the register pin moves (`pin-baseline.sh --repin`) and develop is merged into
the evaluation branch, so HEAD's crates are not the station's crates. Every tree read
goes through a `Tree` (git-backed at a sha; the worktree only when `sha` is None).
"""

from __future__ import annotations

import contextlib
import fnmatch
import functools
import json
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest
from collections.abc import Iterator
from typing import Any

from lib import backlog_parse

REPO = pathlib.Path(__file__).resolve().parents[4]
AUDIT = REPO / "plans" / "architecture-debt-audit"
TOOLS = AUDIT / "tools"
STATIONS = AUDIT / "stations"
BACKLOG = AUDIT / "BACKLOG.md"
PREVIOUS_BASELINES_RE = re.compile(r"^Previous baselines:\s+(?P<body>.+)$")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, check=False, text=True
    )


def _git_bytes(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=False)


class Tree:
    """Read-only view of the repository at one commit, or of the worktree when `sha` is None."""

    def __init__(self, sha: str | None):
        self.sha = sha

    def __repr__(self) -> str:
        return f"Tree({self.sha[:8] if self.sha else 'worktree'})"

    @functools.cached_property
    def _files(self) -> frozenset[str]:
        if self.sha is None:
            out = _git("ls-files").stdout
        else:
            out = _git("ls-tree", "-r", "--name-only", self.sha).stdout
        return frozenset(out.splitlines())

    def files(self) -> frozenset[str]:
        """Every tracked path (repo-relative, posix) in the tree."""
        return self._files

    def is_file(self, rel: str | pathlib.Path) -> bool:
        return pathlib.PurePosixPath(rel).as_posix() in self._files

    def is_dir(self, rel: str | pathlib.Path) -> bool:
        prefix = pathlib.PurePosixPath(rel).as_posix().rstrip("/") + "/"
        return any(f.startswith(prefix) for f in self._files)

    def exists(self, rel: str | pathlib.Path) -> bool:
        return self.is_file(rel) or self.is_dir(rel)

    def read_bytes(self, rel: str | pathlib.Path) -> bytes:
        rel = pathlib.PurePosixPath(rel).as_posix()
        if self.sha is None:
            return (REPO / rel).read_bytes()
        proc = _git_bytes("show", f"{self.sha}:{rel}")
        if proc.returncode != 0:
            raise FileNotFoundError(f"{rel} at {self.sha[:8]}")
        return bytes(proc.stdout)

    def read_text(self, rel: str | pathlib.Path) -> str:
        return self.read_bytes(rel).decode("utf-8", errors="replace")

    def ls_files(self, *pathspecs: str) -> list[str]:
        """Tracked paths under the given pathspecs (`<root>/*.rs` globs and directories).

        `git ls-tree` does not expand wildcards, so at a sha the pathspecs are matched
        here with `git ls-files` semantics: `*` spans `/`, a bare directory is a prefix.
        """
        if self.sha is None:
            return list(_git("ls-files", "--", *pathspecs).stdout.splitlines())

        def matches(path: str) -> bool:
            for spec in pathspecs:
                spec = spec.rstrip("/")
                if any(ch in spec for ch in "*?["):
                    if fnmatch.fnmatchcase(path, spec):
                        return True
                elif path == spec or path.startswith(spec + "/"):
                    return True
            return not pathspecs

        return sorted(f for f in self._files if matches(f))

    def rs_files(self, rel: str | pathlib.Path) -> list[str]:
        """Every tracked .rs under `rel` (or `rel` itself when it is a file), sorted."""
        rel = pathlib.PurePosixPath(rel).as_posix()
        if self.is_file(rel):
            return [rel]
        prefix = rel.rstrip("/") + "/"
        return sorted(
            f for f in self._files if f.startswith(prefix) and f.endswith(".rs")
        )

    def children(self, rel: str | pathlib.Path) -> tuple[list[str], list[str]]:
        """(files, directories) directly under `rel`, as repo-relative paths, sorted."""
        prefix = pathlib.PurePosixPath(rel).as_posix().rstrip("/") + "/"
        files: set[str] = set()
        dirs: set[str] = set()
        for f in self._files:
            if not f.startswith(prefix):
                continue
            head, sep, _ = f[len(prefix) :].partition("/")
            (dirs if sep else files).add(prefix + head)
        return sorted(files), sorted(dirs)

    def dirs(self, rel: str | pathlib.Path) -> list[str]:
        """Every directory strictly under `rel` at any depth (a tracked file implies its parents)."""
        prefix = pathlib.PurePosixPath(rel).as_posix().rstrip("/") + "/"
        out: set[str] = set()
        for f in self._files:
            if not f.startswith(prefix):
                continue
            parts = f[len(prefix) :].split("/")[:-1]
            for depth in range(1, len(parts) + 1):
                out.add(prefix + "/".join(parts[:depth]))
        return sorted(out)

    @contextlib.contextmanager
    def checkout(self) -> Iterator[pathlib.Path]:
        """A temporary detached git worktree at this sha (scripts that need `git` run there)."""
        if self.sha is None:
            yield REPO
            return
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="station-tree-"))
        dest = tmp / "wt"
        proc = _git("worktree", "add", "--detach", str(dest), self.sha)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr)
        try:
            yield dest
        finally:
            _git("worktree", "remove", "--force", str(dest))
            shutil.rmtree(tmp, ignore_errors=True)


WORKTREE = Tree(None)


@functools.lru_cache(maxsize=8)
def _station_tree(sha: str) -> Tree:
    """One shared `Tree` per sha so its file listing is read from git once per test run."""
    return Tree(sha)


def pin_history() -> list[str]:
    """The register's current pin followed by every superseded pin (short or full shas)."""
    lines = backlog_parse.read_register(BACKLOG)
    pins = [backlog_parse.parse_baseline(lines)]
    for raw in lines[:60]:
        m = PREVIOUS_BASELINES_RE.match(raw.strip())
        if m:
            pins.extend(re.findall(r"\b[0-9a-f]{8,40}\b", m.group("body")))
    return pins


def is_register_pin(sha: str) -> bool:
    """True when `sha` is the current register pin or one it superseded."""
    return any(sha.startswith(p[:8]) or p.startswith(sha[:8]) for p in pin_history())


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
# `.py` files resolve def / class at any indent or a column-0 module constant — never the
# field form, which would accept a type-annotated parameter as a declaration.
_PY_DECL_RE = r"^\s*(async\s+)?(def|class)\s+{sym}\b|^{sym}\s*(:[^=\n]*)?="


def run_checker(tool: str, section_title: str, *extra: str) -> int:
    """Exit code of tools/<tool> run over one BACKLOG section from the repo root."""
    return subprocess.run(
        ["python3", str(TOOLS / tool), section_title, *extra],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).returncode


def load_json(path: pathlib.Path | str) -> Any:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def split_anchor(anchor: str) -> tuple[str, str | None, int | None, int | None]:
    hit = ANCHOR_LOC_RE.fullmatch(anchor.strip())
    if hit is None:
        raise ValueError(f"not an anchor: {anchor!r}")
    line = int(hit.group("line")) if hit.group("line") else None
    line_to = int(hit.group("line_to")) if hit.group("line_to") else None
    return hit.group("path"), hit.group("symbol"), line, line_to


def symbol_resolves(
    path: pathlib.Path | str, symbol: str, tree: Tree = WORKTREE
) -> bool:
    text = tree.read_text(path)
    sym = re.escape(symbol)
    suffix = pathlib.PurePosixPath(path).suffix
    if suffix == ".rs":
        patterns = [_DECL_RE, _FIELD_RE]
    elif suffix == ".py":
        patterns = [_PY_DECL_RE]
    else:
        patterns = [_SHELL_RE, _DECL_RE, _FIELD_RE]
    return any(re.search(p.format(sym=sym), text, re.M) for p in patterns)


def anchor_exists(anchor: str, tree: Tree = WORKTREE) -> bool:
    """True when the anchor's path exists in `tree` and its symbol or line range resolves."""
    try:
        rel, symbol, line, line_to = split_anchor(anchor)
    except ValueError:
        return False
    if not tree.is_file(rel):
        return False
    if symbol is not None:
        return symbol_resolves(rel, symbol, tree)
    if line is not None:
        count = len(tree.read_text(rel).splitlines())
        if line_to is None:
            return 1 <= line <= count
        return 1 <= line <= line_to <= count
    return True


def anchors_in(text: str) -> list[str]:
    return [m.group(0) for m in ANCHOR_LOC_RE.finditer(text)]


def tracked_modifications() -> list[str]:
    """`git status --porcelain` over tracked files, minus the carried-in .gitignore line."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in out.splitlines() if line.strip() != "M .gitignore"]


def section_entries(section_title: str) -> list[backlog_parse.Entry]:
    lines = backlog_parse.read_register(BACKLOG)
    return backlog_parse.iter_entries(backlog_parse.find_section(lines, section_title))


def rs_files(path: pathlib.Path | str, tree: Tree = WORKTREE) -> list[str]:
    """Repo-relative .rs paths under `path` in `tree` (the path itself when it is a file)."""
    rel = pathlib.PurePosixPath(path)
    if rel.is_absolute():
        rel = rel.relative_to(REPO.as_posix())
    return tree.rs_files(rel.as_posix())


def raw_lines(path: pathlib.Path | str, tree: Tree = WORKTREE) -> int:
    """`wc -l` over every .rs file under the path (the file itself when it is a file)."""
    return sum(tree.read_bytes(f).count(b"\n") for f in rs_files(path, tree))


def _is_test_file(rel: str) -> bool:
    base = rel.rsplit("/", 1)[-1]
    return (
        "/tests/" in f"/{rel}"
        or "/benches/" in f"/{rel}"
        or base in {"tests.rs", "test_support.rs"}
    )


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
        elif (
            t.startswith("#[cfg")
            and re.search(r"\(test[,)]", t)
            and not re.search(r"not\s*\(test", t)
        ):
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


def non_test_lines(
    path: pathlib.Path | str, code: bool = False, tree: Tree = WORKTREE
) -> int:
    """Non-test physical (or code) lines under the loc-stats.sh rule, summed over the path."""
    total = 0
    for rel in rs_files(path, tree):
        phys, cod = classify_lines(rel, tree.read_text(rel))
        total += cod if code else phys
    return total


@functools.lru_cache(maxsize=4)
def _loc_stats_csv(sha: str | None) -> str:
    """`scripts/loc-stats.sh --csv` output over the tree at `sha` (the worktree when None)."""
    with Tree(sha).checkout() as root:
        return subprocess.run(
            ["bash", "scripts/loc-stats.sh", "--csv"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout


def loc_stats(crate: str, tree: Tree = WORKTREE) -> dict[str, int]:
    """One crate's row of `scripts/loc-stats.sh --csv` (files, prod_code, prod_all, ...)."""
    out = _loc_stats_csv(tree.sha)
    header = out.splitlines()[0].split(",")
    for row in out.splitlines()[1:]:
        cells = row.split(",")
        if cells[0] == crate:
            return {k: int(v) for k, v in zip(header[1:], cells[1:])}
    raise KeyError(crate)


class StationCase(unittest.TestCase):
    """Base for a station's test module; `tree` is the tree the station evaluated.

    The baseline comes from the station's `inventory.json` (`baseline`), which must be
    the register's current pin or one it superseded (`is_register_pin`).
    """

    SLUG = ""
    SECTION_TITLE = ""
    STATION_DIR: pathlib.Path | None = None

    @classmethod
    def baseline(cls) -> str:
        return str(load_json(cls.station_dir() / "inventory.json")["baseline"])

    @classmethod
    def tree(cls) -> Tree:
        return _station_tree(cls.baseline())

    @classmethod
    def station_dir(cls) -> pathlib.Path:
        return cls.STATION_DIR or STATIONS / cls.SLUG

    def artifact(self, name: str) -> pathlib.Path:
        return self.station_dir() / name
