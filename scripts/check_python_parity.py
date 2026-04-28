#!/usr/bin/env python3
"""Check that CLI and Python bindings write the same output files.

Parses both `crates/cobre-cli/src/commands/run.rs` and
`crates/cobre-python/src/run.rs` for calls to `cobre_io::write_*`,
`write_results`, `write_checkpoint` / `write_policy_checkpoint`,
`write_scaling_report`, and the stochastic export functions.

Usage:
    python3 scripts/check_python_parity.py              # default: --max 0
    python3 scripts/check_python_parity.py --max 0      # strict — zero mismatches allowed

Exit code 0 if parity holds (mismatches <= --max), 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Patterns that capture write function calls in Rust source.
# We look for: `cobre_io::write_<name>(`, `write_results(`, `write_checkpoint(`,
# `write_policy_checkpoint(`, `io_write_policy_checkpoint(`,
# and the stochastic output helpers imported from cobre_io::output.
WRITE_CALL_RE = re.compile(
    r"""
    (?:
        cobre_io::(?P<qual>write_\w+)           # qualified cobre_io::write_*
        | (?<!\w)(?P<bare>                       # bare (imported) calls
            write_results
            | write_checkpoint
            | write_policy_checkpoint
            | io_write_policy_checkpoint
            | write_scaling_report
            | write_noise_openings
            | write_inflow_annual_component
            | write_inflow_ar_coefficients
            | write_inflow_seasonal_stats
            | write_correlation_json
            | write_load_seasonal_stats
            | write_fitting_report
        )(?=\()                                  # followed by (
    )
    """,
    re.VERBOSE,
)

# Normalisation map: different names that map to the same logical write.
NORMALISE: dict[str, str] = {
    "write_checkpoint": "write_policy_checkpoint",
    "io_write_policy_checkpoint": "write_policy_checkpoint",
}


def extract_write_functions(path: Path) -> set[str]:
    """Extract the set of write function names from a Rust source file."""
    if not path.exists():
        print(f"WARNING: {path} does not exist", file=sys.stderr)
        return set()

    text = path.read_text(errors="replace")
    names: set[str] = set()

    for line in text.splitlines():
        stripped = line.strip()
        # Skip comments and use/import lines (we want actual calls, not imports).
        if stripped.startswith("//"):
            continue
        # Skip `use` imports — we only want call sites.
        if stripped.startswith("use "):
            continue

        for m in WRITE_CALL_RE.finditer(line):
            name = m.group("qual") or m.group("bare")
            if name:
                name = NORMALISE.get(name, name)
                names.add(name)

    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Python parity for output writes."
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Maximum allowed mismatches (default: 0). Exit 1 if exceeded.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root (default: current directory).",
    )
    args = parser.parse_args()

    cli_path = args.root / "crates" / "cobre-cli" / "src" / "commands" / "run.rs"
    python_path = args.root / "crates" / "cobre-python" / "src" / "run.rs"

    cli_writes = extract_write_functions(cli_path)
    python_writes = extract_write_functions(python_path)

    cli_only = sorted(cli_writes - python_writes)
    python_only = sorted(python_writes - cli_writes)
    mismatches = len(cli_only) + len(python_only)

    if mismatches > args.max:
        print(f"FAIL: {mismatches} parity mismatch(es) (max allowed: {args.max})")
        if cli_only:
            print()
            print("  In CLI but missing from Python:")
            for name in cli_only:
                print(f"    - {name}")
        if python_only:
            print()
            print("  In Python but missing from CLI:")
            for name in python_only:
                print(f"    - {name}")
        print()
        print("Fix: add the missing write call(s) to the other path.")
        print("CLI path:    crates/cobre-cli/src/commands/run.rs")
        print("Python path: crates/cobre-python/src/run.rs")
        sys.exit(1)
    else:
        shared = sorted(cli_writes & python_writes)
        print(
            f"OK: {mismatches} parity mismatch(es) (max allowed: {args.max}). "
            f"{len(shared)} write functions in both paths."
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
