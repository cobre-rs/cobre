#!/usr/bin/env python3
"""Enumerate every #[test]-annotated function in the six in-scope crates.

Walks {src,tests} subtrees of the in-scope crates and emits a CSV row for
every ``#[test]`` attribute found.  The ``category``, ``guards``, and
``notes`` columns are emitted as empty strings; those are filled in by the
human tagger in ticket 003.

Known limitations (documented, not fixed):
  - ``body_loc`` may undercount when ``{`` or ``}`` appear inside format-macro
    string literals such as ``format!("{}", x)``; the state machine tracks
    only regular string delimiters and ``r#"..."#`` raw strings.
  - Braces inside ``/*``-style block comments that span a line boundary are
    only handled when the nesting depth of ``/* */`` is tracked.  Single-line
    occurrences are handled correctly.
  - ``test_module`` reflects the nearest enclosing ``mod <ident> {`` that has
    not yet been closed at the depth where ``fn`` appears.  Nested mods work;
    mods split across files (via ``mod foo;``) are not resolved.
  - ``#[tokio::test]`` and other proc-macro test attributes are intentionally
    NOT counted.  Only the bare ``#[test]`` attribute is detected.

Usage:
    python3 scripts/test_inventory.py
    python3 scripts/test_inventory.py --crates cobre-sddp,cobre-solver
    python3 scripts/test_inventory.py --output /tmp/inventory.csv
    python3 scripts/test_inventory.py --include-slow-marker
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CRATES: list[str] = [
    "cobre-sddp",
    "cobre-solver",
    "cobre-io",
    "cobre-cli",
    "cobre-python",
    "cobre-comm",
]

CSV_COLUMNS: list[str] = [
    "crate",
    "file",
    "line",
    "function",
    "body_loc",
    "test_module",
    "category",
    "guards",
    "notes",
]

CSV_COLUMNS_WITH_SLOW: list[str] = CSV_COLUMNS + ["slow_marker"]

# Regex: a line that is *exactly* #[test] (with optional surrounding whitespace)
_RE_TEST_ATTR = re.compile(r"^\s*#\[test\]\s*$")

# Regex: ``fn <ident>(`` — captures the function name
_RE_FN = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[(<]")

# Regex: ``mod <ident>`` — captures module name
_RE_MOD = re.compile(r"\bmod\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{")

# Regex: slow-marker attributes — #[cfg(feature = "...")] or #[ignore]
_RE_SLOW = re.compile(
    r"""#\[\s*(?:cfg\s*\(\s*feature\s*=\s*"[^"]*"\s*\)|ignore)\s*\]"""
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class TestEntry(NamedTuple):
    """One detected ``#[test]`` function."""

    crate: str
    file: str  # relative to repo root
    line: int  # 1-based line of the #[test] attribute
    function: str
    body_loc: int
    test_module: str
    slow_marker: str  # populated only when --include-slow-marker


class ParseError(Exception):
    """Raised when a .rs file cannot be fully parsed (e.g. unbalanced braces)."""


# ---------------------------------------------------------------------------
# Brace / string-literal state machine
# ---------------------------------------------------------------------------


def _count_body_loc(lines: list[str], fn_line_idx: int) -> int:
    """Count lines from opening ``{`` on *fn_line_idx* through the matching ``}``.

    The parser tracks the following lexical states to avoid counting braces
    inside non-code contexts:
    - ``in_line_comment``  — from ``//`` to end-of-line
    - ``in_block_comment`` — from ``/*`` to ``*/`` (nesting counted)
    - ``in_raw_string``    — from ``r"`` / ``r#"``/ ``r##"`` … to matching terminator
    - ``in_string``        — from ``"`` to ``"`` with ``\\`` escapes

    Returns the body LoC (number of lines from the opening ``{`` up to and
    including the line containing the matching ``}``).

    Raises ``ParseError`` if EOF is reached before braces balance.
    """
    # States
    in_line_comment = False
    in_block_comment = 0  # nesting depth
    in_string = False
    in_raw_string = False
    raw_string_hashes = 0  # number of # in r#"  (closing: " + this many #)

    depth = 0
    body_start_line: int | None = None

    i = fn_line_idx
    total = len(lines)

    while i < total:
        line = lines[i]
        j = 0
        line_len = len(line)
        in_line_comment = False  # reset at start of each line

        while j < line_len:
            ch = line[j]

            # ----------------------------------------------------------------
            # Priority 1: raw string mode
            # ----------------------------------------------------------------
            if in_raw_string:
                # Look for the closing terminator: " followed by raw_string_hashes #
                if ch == '"':
                    # Check if followed by the right number of #
                    suffix = line[j + 1 : j + 1 + raw_string_hashes]
                    if suffix == "#" * raw_string_hashes:
                        in_raw_string = False
                        j += 1 + raw_string_hashes
                        continue
                j += 1
                continue

            # ----------------------------------------------------------------
            # Priority 2: regular string mode
            # ----------------------------------------------------------------
            if in_string:
                if ch == "\\" and j + 1 < line_len:
                    j += 2  # skip escaped char
                    continue
                if ch == '"':
                    in_string = False
                j += 1
                continue

            # ----------------------------------------------------------------
            # Priority 3: line comment — rest of line is comment
            # ----------------------------------------------------------------
            if in_line_comment:
                break  # skip to next line

            # ----------------------------------------------------------------
            # Priority 4: block comment
            # ----------------------------------------------------------------
            if in_block_comment > 0:
                # Check for nested /* or closing */
                if ch == "/" and j + 1 < line_len and line[j + 1] == "*":
                    in_block_comment += 1
                    j += 2
                    continue
                if ch == "*" and j + 1 < line_len and line[j + 1] == "/":
                    in_block_comment -= 1
                    j += 2
                    continue
                j += 1
                continue

            # ----------------------------------------------------------------
            # Normal code: detect state transitions and count braces
            # ----------------------------------------------------------------

            # Line comment?
            if ch == "/" and j + 1 < line_len and line[j + 1] == "/":
                in_line_comment = True
                break

            # Block comment open?
            if ch == "/" and j + 1 < line_len and line[j + 1] == "*":
                in_block_comment += 1
                j += 2
                continue

            # Raw string: r" or r#" or r##" …
            if ch == "r" and j + 1 < line_len and line[j + 1] in ('"', "#"):
                hashes = 0
                k = j + 1
                while k < line_len and line[k] == "#":
                    hashes += 1
                    k += 1
                if k < line_len and line[k] == '"':
                    in_raw_string = True
                    raw_string_hashes = hashes
                    j = k + 1
                    continue

            # Regular string?
            if ch == '"':
                in_string = True
                j += 1
                continue

            # Char literal? (single quotes) — skip to closing '
            if ch == "'":
                j += 1
                if j < line_len and line[j] == "\\":
                    j += 1  # escaped char
                while j < line_len and line[j] != "'":
                    j += 1
                j += 1  # closing '
                continue

            # Brace counting
            if ch == "{":
                if depth == 0:
                    body_start_line = i
                depth += 1

            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0:
                        if body_start_line is None:
                            raise ParseError(
                                f"Unexpected closing brace at line {i + 1}"
                            )
                        return i - body_start_line + 1

            j += 1

        i += 1

    raise ParseError(
        f"Unbalanced braces: depth={depth} at EOF (fn started near line {fn_line_idx + 1})"
    )


# ---------------------------------------------------------------------------
# Module-context tracker
# ---------------------------------------------------------------------------


class _ModuleStack:
    """Track ``mod <name> { ... }`` nesting as we scan lines.

    This is a simplified tracker: it only records mod entries when the opening
    ``{`` is on the same line as ``mod``.  This covers all normal Rust style.
    """

    def __init__(self) -> None:
        # Each element: (name, brace_depth_at_open)
        self._stack: list[tuple[str, int]] = []
        self._brace_depth: int = 0

    def feed_line(self, line: str) -> None:
        """Update brace depth and module stack from a single line of code."""
        # Quick skip: track braces from the simplified (non-state-machine) view.
        # For module tracking we just need gross depth; this is good enough for
        # the common indentation-aligned style used in this codebase.
        open_count = line.count("{") - line.count(r"\{")
        close_count = line.count("}") - line.count(r"\}")

        # Detect mod push BEFORE updating depth (opening { on same line)
        m = _RE_MOD.search(line)
        if m and "{" in line:
            mod_name = m.group(1)
            self._stack.append((mod_name, self._brace_depth))

        self._brace_depth += open_count - close_count

        # Pop mods that are now closed
        while self._stack and self._brace_depth <= self._stack[-1][1]:
            self._stack.pop()

    def current_module(self) -> str:
        """Return the innermost module name, or '' if at top level."""
        return self._stack[-1][0] if self._stack else ""


# ---------------------------------------------------------------------------
# File parser
# ---------------------------------------------------------------------------


def _collect_slow_markers(
    lines: list[str], attr_line_idx: int, lookahead: int = 5
) -> str:
    """Scan a few lines *before* the ``#[test]`` line for slow-marker attributes.

    Returns a space-separated string of all detected markers, or ``""``.
    """
    markers: list[str] = []
    start = max(0, attr_line_idx - lookahead)
    for line in lines[start:attr_line_idx]:
        for m in _RE_SLOW.finditer(line):
            markers.append(m.group(0).strip())
    return " ".join(markers)


def parse_file(
    rs_file: Path,
    repo_root: Path,
    crate_name: str,
    include_slow_marker: bool,
) -> list[TestEntry]:
    """Parse a single ``.rs`` file and return a ``TestEntry`` for each ``#[test]``.

    Raises ``ParseError`` on unbalanced braces or UTF-8 decode failure.
    """
    try:
        text = rs_file.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ParseError(
            f"UTF-8 decode error in {rs_file}: {exc}"
        ) from exc

    lines = text.splitlines()
    rel_path = str(rs_file.relative_to(repo_root))

    entries: list[TestEntry] = []
    mod_stack = _ModuleStack()

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Update module context first
        mod_stack.feed_line(line)

        if not _RE_TEST_ATTR.match(line):
            i += 1
            continue

        # Found #[test] at line i (1-based: i+1)
        test_attr_line = i + 1  # 1-based
        test_module = mod_stack.current_module()

        slow_marker = ""
        if include_slow_marker:
            slow_marker = _collect_slow_markers(lines, i)

        # Scan forward (up to 20 lines) for the ``fn <name>`` declaration
        fn_name = ""
        fn_line_idx = -1
        for k in range(i + 1, min(i + 20, n)):
            m = _RE_FN.search(lines[k])
            if m:
                fn_name = m.group(1)
                fn_line_idx = k
                break

        if not fn_name or fn_line_idx < 0:
            # No fn found within lookahead — skip this #[test]
            i += 1
            continue

        # Feed the lines between the #[test] and the fn declaration to the
        # module stack so the brace depth stays consistent when we jump.
        for k in range(i + 1, fn_line_idx + 1):
            mod_stack.feed_line(lines[k])

        # Count body LoC
        body_loc = _count_body_loc(lines, fn_line_idx)

        entries.append(
            TestEntry(
                crate=crate_name,
                file=rel_path,
                line=test_attr_line,
                function=fn_name,
                body_loc=body_loc,
                test_module=test_module,
                slow_marker=slow_marker,
            )
        )

        # Advance past the fn line to avoid re-triggering on #[test] inside
        # the body (shouldn't happen but be safe).  The fn line itself was
        # already fed above; lines beyond fn_line_idx will be fed normally.
        i = fn_line_idx + 1

    return entries


# ---------------------------------------------------------------------------
# Walk crate subtrees
# ---------------------------------------------------------------------------


def walk_crate(
    crates_dir: Path,
    crate_name: str,
    repo_root: Path,
    include_slow_marker: bool,
) -> tuple[list[TestEntry], list[str]]:
    """Walk ``{src,tests}`` under ``crates_dir / crate_name``.

    Returns ``(entries, errors)`` where errors is a list of human-readable
    error strings for any file that failed to parse.
    """
    crate_root = crates_dir / crate_name
    entries: list[TestEntry] = []
    errors: list[str] = []

    for subtree_name in ("src", "tests"):
        subtree = crate_root / subtree_name
        if not subtree.exists():
            continue
        for rs_file in sorted(subtree.rglob("*.rs")):
            # Skip build artifacts
            if "target" in rs_file.parts:
                continue
            try:
                file_entries = parse_file(
                    rs_file, repo_root, crate_name, include_slow_marker
                )
                entries.extend(file_entries)
            except ParseError as exc:
                errors.append(str(exc))

    return entries, errors


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def write_csv(
    entries: list[TestEntry],
    dest: Path | None,
    include_slow_marker: bool,
) -> None:
    """Write the inventory CSV to *dest* (file) or stdout."""
    columns = CSV_COLUMNS_WITH_SLOW if include_slow_marker else CSV_COLUMNS

    if dest is None:
        # Write directly to sys.stdout (text mode).  Using sys.stdout directly
        # (rather than sys.stdout.buffer) keeps the function testable when
        # stdout is replaced with io.StringIO in unit tests.
        writer = csv.writer(sys.stdout)
        writer.writerow(columns)
        for e in entries:
            row = [
                e.crate,
                e.file,
                e.line,
                e.function,
                e.body_loc,
                e.test_module,
                "",  # category
                "",  # guards
                "",  # notes
            ]
            if include_slow_marker:
                row.append(e.slow_marker)
            writer.writerow(row)
        sys.stdout.flush()
    else:
        with dest.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            for e in entries:
                row = [
                    e.crate,
                    e.file,
                    e.line,
                    e.function,
                    e.body_loc,
                    e.test_module,
                    "",  # category
                    "",  # guards
                    "",  # notes
                ]
                if include_slow_marker:
                    row.append(e.slow_marker)
                writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate every #[test]-annotated function in the six in-scope crates "
            "and emit a CSV inventory.  The category, guards, and notes columns are "
            "left empty for the human tagger (ticket 003)."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Path to the repository root (default: current working directory). "
            "Must contain a Cargo.toml workspace file."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write CSV to PATH instead of stdout.",
    )
    parser.add_argument(
        "--crates",
        type=str,
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated list of crate names to include "
            "(default: all six in-scope crates). "
            "The 'cobre-' prefix is optional. "
            "Example: --crates sddp,solver  or  --crates cobre-sddp,cobre-io"
        ),
    )
    parser.add_argument(
        "--include-slow-marker",
        action="store_true",
        default=False,
        help=(
            "Append a slow_marker column listing any #[cfg(feature = \"...\")] "
            "or #[ignore] attribute found near the #[test] line."
        ),
    )
    return parser.parse_args(argv)


def _resolve_crates(raw: str | None) -> list[str]:
    """Normalise the --crates argument to full crate names with 'cobre-' prefix."""
    if raw is None:
        return list(DEFAULT_CRATES)
    names: list[str] = []
    for part in raw.split(","):
        name = part.strip()
        if not name:
            continue
        if not name.startswith("cobre-"):
            name = "cobre-" + name
        names.append(name)
    return names


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    repo_root: Path = args.repo_root if args.repo_root is not None else Path.cwd()
    repo_root = repo_root.resolve()

    # Validate repo root
    cargo_toml = repo_root / "Cargo.toml"
    if not cargo_toml.exists():
        print(
            f"test_inventory: error: {repo_root} does not contain a Cargo.toml "
            f"workspace file",
            file=sys.stderr,
        )
        sys.exit(1)

    crates_dir = repo_root / "crates"
    crate_names = _resolve_crates(args.crates)
    include_slow = args.include_slow_marker

    all_entries: list[TestEntry] = []
    all_errors: list[str] = []
    file_count = 0

    for crate_name in crate_names:
        crate_root = crates_dir / crate_name
        for subtree_name in ("src", "tests"):
            subtree = crate_root / subtree_name
            if subtree.exists():
                file_count += sum(
                    1
                    for f in subtree.rglob("*.rs")
                    if "target" not in f.parts
                )

        entries, errors = walk_crate(crates_dir, crate_name, repo_root, include_slow)
        all_entries.extend(entries)
        all_errors.extend(errors)

    # Report parse errors to stderr
    for err in all_errors:
        print(f"test_inventory: parse error: {err}", file=sys.stderr)

    # Summary to stderr
    print(
        f"test_inventory: inventoried {len(all_entries)} tests across {file_count} files",
        file=sys.stderr,
    )

    write_csv(all_entries, args.output, include_slow)

    if all_errors:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
