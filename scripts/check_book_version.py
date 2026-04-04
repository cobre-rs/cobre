#!/usr/bin/env python3
"""Check that all version references in book/src/**/*.md match the workspace Cargo.toml.

Scans all Markdown files under book/src/ for two patterns:
  - COBRE v<VERSION>      (uppercase banner, e.g. quickstart.md)
  - cobre   v<VERSION>    (lowercase CLI output with three spaces, e.g. installation.md)

Parses the workspace version from `Cargo.toml` (the first `version = "..."` line)
and reports any version reference that does not match.

Usage:
    python3 scripts/check_book_version.py
    python3 scripts/check_book_version.py --root /path/to/repo

Exit code 0 if all matched versions equal the Cargo.toml version, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_cargo_version(cargo_path: Path) -> str | None:
    """Extract the workspace version from Cargo.toml."""
    if not cargo_path.exists():
        return None
    for line in cargo_path.read_text().splitlines():
        m = re.match(r'^version\s*=\s*"([^"]+)"', line)
        if m:
            return m.group(1)
    return None


_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"COBRE v(\d+\.\d+\.\d+)"),
    re.compile(r"cobre\s{3}v(\d+\.\d+\.\d+)"),
]


def scan_book_versions(book_src: Path, expected: str) -> list[tuple[Path, int, str]]:
    """Scan all .md files under book_src for version mismatches.

    Returns a list of (file_path, line_number, found_version) tuples where the
    found version does not equal expected.  File read errors are printed to
    stderr and counted as a single mismatch entry with found_version="<unreadable>".
    """
    mismatches: list[tuple[Path, int, str]] = []

    md_files = sorted(book_src.rglob("*.md"))
    if not md_files:
        print(f"WARNING: No .md files found under {book_src}", file=sys.stderr)
        return mismatches

    for md_path in md_files:
        try:
            lines = md_path.read_text(encoding="utf-8").splitlines()
        except OSError as err:
            print(f"WARNING: Could not read {md_path}: {err}", file=sys.stderr)
            mismatches.append((md_path, 0, "<unreadable>"))
            continue

        for lineno, line in enumerate(lines, start=1):
            for pattern in _PATTERNS:
                for m in pattern.finditer(line):
                    found = m.group(1)
                    if found != expected:
                        mismatches.append((md_path, lineno, found))

    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check book/src/**/*.md version references match Cargo.toml."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root (default: current directory).",
    )
    args = parser.parse_args()

    cargo_path = args.root / "Cargo.toml"
    book_src = args.root / "book" / "src"

    cargo_version = parse_cargo_version(cargo_path)
    if cargo_version is None:
        print(f"ERROR: Could not parse version from {cargo_path}", file=sys.stderr)
        sys.exit(1)

    mismatches = scan_book_versions(book_src, cargo_version)

    if not mismatches:
        # Count total matched references to include in the OK message.
        total = 0
        if book_src.exists():
            for md_path in sorted(book_src.rglob("*.md")):
                try:
                    text = md_path.read_text(encoding="utf-8")
                    for pattern in _PATTERNS:
                        total += len(pattern.findall(text))
                except OSError:
                    pass
        print(
            f"OK: All {total} version references in book/ match Cargo.toml (v{cargo_version})."
        )
        sys.exit(0)
    else:
        for file_path, lineno, found in mismatches:
            print(
                f"MISMATCH: {file_path}:{lineno}: found v{found}, expected v{cargo_version}"
            )
        print(f"FAIL: {len(mismatches)} mismatches found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
