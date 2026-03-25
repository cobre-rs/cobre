#!/usr/bin/env python3
"""Check for clippy lint suppressions in production Rust code.

Production code = everything before the first #[cfg(test)] in each file.
Test code (after #[cfg(test)]) is excluded entirely.

Usage:
    python3 scripts/check_suppressions.py                    # default: --check too_many_arguments --max 0
    python3 scripts/check_suppressions.py --max 15           # allow up to 15 (current legacy count)
    python3 scripts/check_suppressions.py --check too_many_lines --max 0
    python3 scripts/check_suppressions.py --check too_many_arguments --check too_many_lines --max 0

Exit code 1 if violations exceed --max.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def find_production_suppressions(
    root: Path,
    lint: str,
) -> list[tuple[str, int, str]]:
    """Return (filepath, line_number, line_text) for each suppression in production code."""
    target = f"clippy::{lint}"
    results = []

    for rs_file in sorted(root.rglob("*.rs")):
        # Skip build artifacts
        if "target" in rs_file.parts:
            continue

        try:
            lines = rs_file.read_text(errors="replace").splitlines()
        except OSError:
            continue

        # Find where #[cfg(test)] starts — everything after is test code
        test_start = len(lines)
        for i, line in enumerate(lines):
            if "#[cfg(test)]" in line:
                test_start = i
                break

        # Search only production portion (lines 0..test_start)
        for i in range(test_start):
            line = lines[i]
            if "allow" in line and target in line:
                rel_path = rs_file.relative_to(root)
                results.append((str(rel_path), i + 1, line.strip()))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check for clippy lint suppressions in production Rust code."
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Maximum allowed suppressions (default: 0). Exit 1 if exceeded.",
    )
    parser.add_argument(
        "--check",
        type=str,
        action="append",
        default=None,
        help=(
            "Clippy lint to check (repeatable). "
            "Default: too_many_arguments. "
            "Example: --check too_many_arguments --check too_many_lines"
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("crates"),
        help="Root directory to search (default: crates)",
    )
    args = parser.parse_args()

    lints = args.check or ["too_many_arguments"]
    total_violations = 0
    all_results: list[tuple[str, str, int, str]] = []

    for lint in lints:
        results = find_production_suppressions(args.root, lint)
        total_violations += len(results)
        for path, line, text in results:
            all_results.append((lint, path, line, text))

    if total_violations > args.max:
        lint_names = ", ".join(lints)
        print(
            f"FAIL: {total_violations} production clippy suppression(s) "
            f"for [{lint_names}] (max allowed: {args.max})"
        )
        print()
        for lint, path, line, _text in all_results:
            print(f"  {path}:{line}  (clippy::{lint})")
        print()
        print(
            "Fix: absorb parameters into context structs instead of suppressing the lint."
        )
        print("See: .claude/architecture-rules.md")
        sys.exit(1)
    else:
        lint_names = ", ".join(lints)
        print(
            f"OK: {total_violations} production clippy suppression(s) "
            f"for [{lint_names}] (max allowed: {args.max})"
        )
        sys.exit(0)


if __name__ == "__main__":
    main()