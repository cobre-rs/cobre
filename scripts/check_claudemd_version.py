#!/usr/bin/env python3
"""Check that the version in CLAUDE.md matches the workspace Cargo.toml.

Parses the workspace version from `Cargo.toml` (the first `version = "..."` line)
and the version from `CLAUDE.md` (the `Current State (v...)` heading).

Usage:
    python3 scripts/check_claudemd_version.py
    python3 scripts/check_claudemd_version.py --root /path/to/repo

Exit code 0 if versions match, 1 if they differ or cannot be parsed.
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


def parse_claudemd_version(claude_path: Path) -> str | None:
    """Extract the version from the 'Current State (v...)' heading in CLAUDE.md."""
    if not claude_path.exists():
        return None
    for line in claude_path.read_text().splitlines():
        m = re.search(r"Current State \(v([^)]+)\)", line)
        if m:
            return m.group(1)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check CLAUDE.md version matches Cargo.toml."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root (default: current directory).",
    )
    args = parser.parse_args()

    cargo_path = args.root / "Cargo.toml"
    claude_path = args.root / "CLAUDE.md"

    cargo_version = parse_cargo_version(cargo_path)
    claude_version = parse_claudemd_version(claude_path)

    if cargo_version is None:
        print(f"ERROR: Could not parse version from {cargo_path}", file=sys.stderr)
        sys.exit(1)

    if claude_version is None:
        print(f"ERROR: Could not parse version from {claude_path}", file=sys.stderr)
        sys.exit(1)

    if cargo_version == claude_version:
        print(
            f"OK: CLAUDE.md version (v{claude_version}) matches Cargo.toml (v{cargo_version})."
        )
        sys.exit(0)
    else:
        print(
            f"FAIL: Version mismatch — Cargo.toml has v{cargo_version}, "
            f"CLAUDE.md has v{claude_version}."
        )
        print("Fix: Update the 'Current State' heading in CLAUDE.md.")
        sys.exit(1)


if __name__ == "__main__":
    main()
