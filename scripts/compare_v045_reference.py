#!/usr/bin/env python3
"""compare_v045_reference.py

Compare a freshly-captured sha256.txt against the v0.4.5 reference map
embedded in docs/assessments/v0_4_5_reference.md.

Exit codes:
  0 -- every mismatch is in the expected-drifts allowlist (prints confirmation)
  1 -- at least one unexpected mismatch found (prints diff-style report)

Usage:
  python3 scripts/compare_v045_reference.py \\
      --reference-sha256 docs/assessments/v0_4_5_reference.md \\
      --actual-sha256 target/v045-reference-post-epic03/sha256.txt
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Expected-drifts allowlist
#
# Each entry is a glob pattern (fnmatch / pathlib semantics) that matches a
# relative path inside the sha256 map.  Hashes for paths matching any of
# these patterns are allowed to differ between the reference and the freshly-
# captured output.
#
# Justification column:
#   - ticket-007  schema rename: `basis_rejections` / `basis_non_alien_rejections`
#                 merged into `basis_consistency_failures`; the parquet schema
#                 changes, therefore the file hash changes.
#   - ticket-001  timing-bearing files already excluded from the reference map;
#                 listed here defensively in case a capture tool includes them.
# ---------------------------------------------------------------------------

EXPECTED_DRIFTS: list[tuple[str, str]] = [
    (
        "**/training/solver/iterations.parquet",
        "ticket-007: basis_consistency_failures schema rename causes hash drift",
    ),
    (
        "**/simulation/solver/iterations.parquet",
        "ticket-007: basis_consistency_failures schema rename causes hash drift",
    ),
    # Defensive: timing-bearing files are already excluded from the reference
    # map by the capture script, but added here so a partial capture does not
    # produce a false-positive unexpected-drift failure.
    (
        "**/training/convergence.parquet",
        "ticket-001: timing columns (time_*_ms) are wall-clock unstable",
    ),
    (
        "**/training/timing/iterations.parquet",
        "ticket-001: pure wall-clock timing file, excluded from stable map",
    ),
    (
        "**/metadata.json",
        "embeds completed_at timestamp and hostname, changes on every run",
    ),
]


def _matches_any_drift(path: str) -> tuple[bool, str]:
    """Return (matched, justification) for the first allowlist hit."""
    for pattern, justification in EXPECTED_DRIFTS:
        # fnmatch does not natively support **.  Normalise by testing both
        # the full path and the path after stripping any leading component,
        # and also test with fnmatch's own ** expansion via pathlib.
        # pathlib.PurePosixPath.match() supports ** semantics.
        from pathlib import PurePosixPath

        if PurePosixPath(path).match(pattern):
            return True, justification
        # Also try plain fnmatch on the basename for simple patterns.
        if fnmatch.fnmatch(Path(path).name, Path(pattern).name):
            # Only accept if the parent glob part also matches.
            parent_pattern = str(Path(pattern).parent)
            if parent_pattern in ("**", "."):
                return True, justification
    return False, ""


# ---------------------------------------------------------------------------
# Reference markdown parser
#
# The SHA256 map is delimited by:
#   ```          ← opening fence (line starting with ```)
#   <entries>    ← one or more lines: "<sha256>  <path>"
#   ```          ← closing fence
#
# The reference doc may contain multiple fenced blocks.  We take the first
# block whose content lines all match the "<64-hex>  <path>" pattern.
# ---------------------------------------------------------------------------

_SHA256_LINE_RE = re.compile(r"^([0-9a-f]{64})\s{2}(.+)$")


def _parse_reference_markdown(path: Path) -> dict[str, str]:
    """Extract the SHA256 map from the reference markdown.

    Returns a dict mapping relative path -> sha256 hex string.
    Raises SystemExit(2) if the marker block cannot be found or parsed.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    in_fence = False
    candidate_lines: list[str] = []
    result: dict[str, str] = {}

    for line in lines:
        stripped = line.strip()
        if not in_fence:
            if stripped == "```":
                in_fence = True
                candidate_lines = []
        else:
            if stripped == "```":
                # End of a fenced block.  Check if all non-empty lines match.
                valid = True
                block_map: dict[str, str] = {}
                for cl in candidate_lines:
                    if not cl.strip():
                        continue
                    m = _SHA256_LINE_RE.match(cl)
                    if not m:
                        valid = False
                        break
                    sha, rel_path = m.group(1), m.group(2).strip()
                    block_map[rel_path] = sha
                if valid and block_map:
                    result.update(block_map)
                in_fence = False
            else:
                candidate_lines.append(line)

    if not result:
        print(
            f"ERROR: No SHA256 map found in reference file: {path}",
            file=sys.stderr,
        )
        sys.exit(2)

    return result


def _parse_sha256_file(path: Path) -> dict[str, str]:
    """Parse a plain sha256.txt file (sha256sum output format).

    Returns a dict mapping relative path -> sha256 hex string.
    """
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _SHA256_LINE_RE.match(line)
        if not m:
            print(
                f"WARNING: skipping malformed line in {path}: {raw_line!r}",
                file=sys.stderr,
            )
            continue
        sha, rel_path = m.group(1), m.group(2).strip()
        result[rel_path] = sha
    return result


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compare(
    reference: dict[str, str],
    actual: dict[str, str],
) -> tuple[list[tuple[str, str, str]], list[str], list[str]]:
    """Compare reference vs actual SHA256 maps.

    Returns:
        unexpected_diffs   -- list of (path, ref_sha, act_sha) for unallowlisted mismatches
        allowed_diffs      -- list of paths whose drift is allowlisted
        missing_in_actual  -- list of paths in reference but absent in actual
    """
    unexpected_diffs: list[tuple[str, str, str]] = []
    allowed_diffs: list[str] = []
    missing_in_actual: list[str] = []

    for path, ref_sha in sorted(reference.items()):
        act_sha = actual.get(path)
        if act_sha is None:
            matched, _ = _matches_any_drift(path)
            if matched:
                allowed_diffs.append(path)
            else:
                missing_in_actual.append(path)
            continue
        if act_sha != ref_sha:
            matched, _ = _matches_any_drift(path)
            if matched:
                allowed_diffs.append(path)
            else:
                unexpected_diffs.append((path, ref_sha, act_sha))

    return unexpected_diffs, allowed_diffs, missing_in_actual


def _format_report(
    unexpected_diffs: list[tuple[str, str, str]],
    missing_in_actual: list[str],
    allowed_diffs: list[str],
    reference_path: Path,
    actual_path: Path,
) -> str:
    lines: list[str] = [
        "=== v0.4.5 regression comparison report ===",
        f"  reference : {reference_path}",
        f"  actual    : {actual_path}",
        "",
    ]

    if allowed_diffs:
        lines.append(
            f"Allowed drifts ({len(allowed_diffs)} file(s) — in expected-drifts allowlist):"
        )
        for p in sorted(allowed_diffs):
            lines.append(f"  ~ {p}")
        lines.append("")

    if missing_in_actual:
        lines.append(
            f"MISSING in actual ({len(missing_in_actual)} file(s) — present in reference, absent in actual):"
        )
        for p in sorted(missing_in_actual):
            lines.append(f"  - {p}")
        lines.append("")

    if unexpected_diffs:
        lines.append(
            f"UNEXPECTED drifts ({len(unexpected_diffs)} file(s) — NOT in expected-drifts allowlist):"
        )
        for path, ref_sha, act_sha in sorted(unexpected_diffs):
            lines.append(f"  ! {path}")
            lines.append(f"      reference : {ref_sha}")
            lines.append(f"      actual    : {act_sha}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a freshly-captured sha256.txt against the v0.4.5 "
            "reference map embedded in docs/assessments/v0_4_5_reference.md."
        ),
    )
    parser.add_argument(
        "--reference-sha256",
        metavar="PATH",
        default="docs/assessments/v0_4_5_reference.md",
        help=(
            "Path to the reference SHA256 source.  "
            "If the file has a .md extension, the SHA256 map is extracted "
            "from the first fenced code block containing <sha256>  <path> lines.  "
            "If the file has a .txt extension, it is parsed directly as sha256sum "
            "output.  Default: docs/assessments/v0_4_5_reference.md"
        ),
    )
    parser.add_argument(
        "--actual-sha256",
        metavar="PATH",
        required=True,
        help="Path to the freshly-captured sha256.txt (sha256sum output format).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full report even when the comparison passes.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    reference_path = Path(args.reference_sha256)
    actual_path = Path(args.actual_sha256)

    if not reference_path.exists():
        print(f"ERROR: reference file not found: {reference_path}", file=sys.stderr)
        sys.exit(2)
    if not actual_path.exists():
        print(f"ERROR: actual sha256 file not found: {actual_path}", file=sys.stderr)
        sys.exit(2)

    # Load reference
    if reference_path.suffix.lower() == ".md":
        reference = _parse_reference_markdown(reference_path)
    else:
        reference = _parse_sha256_file(reference_path)

    # Load actual
    actual = _parse_sha256_file(actual_path)

    # Compare
    unexpected_diffs, allowed_diffs, missing_in_actual = compare(reference, actual)

    # Report
    if args.verbose or unexpected_diffs or missing_in_actual:
        print(
            _format_report(
                unexpected_diffs,
                missing_in_actual,
                allowed_diffs,
                reference_path,
                actual_path,
            )
        )

    if unexpected_diffs:
        print(
            f"FAIL: {len(unexpected_diffs)} unexpected drift(s) detected — "
            "investigation required (see Error Handling in ticket-009).",
            file=sys.stderr,
        )
        sys.exit(1)

    if missing_in_actual:
        print(
            f"FAIL: {len(missing_in_actual)} file(s) present in reference but "
            "absent in actual output — re-run the capture script.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Success path
    total = len(reference)
    n_allowed = len(allowed_diffs)
    n_ok = total - n_allowed
    print(
        f"all mismatches are in the expected-drifts allowlist "
        f"({n_ok}/{total} files byte-identical, "
        f"{n_allowed} allowlisted drift(s))"
    )


# ---------------------------------------------------------------------------
# Self-test (run with: python3 scripts/compare_v045_reference.py --selftest)
# ---------------------------------------------------------------------------


def _selftest() -> None:
    """Embedded self-tests; called when the module is executed directly with
    the --selftest flag (not via the normal argparse path)."""
    import tempfile

    # 64-char hex strings used throughout the self-test
    SHA_A = "a" * 64
    SHA_B = "b" * 64
    SHA_C = "c" * 64
    SHA_D = "d" * 64
    SHA_E = "e" * 64

    ref_md_content = f"""\
# Test reference

## SHA256 map

```
{SHA_A}  d01-thermal-dispatch/training/dictionaries/bounds.parquet
{SHA_B}  d02-single-hydro/training/solver/iterations.parquet
{SHA_C}  d03-two-hydro-cascade/simulation/solver/iterations.parquet
```
"""

    actual_content = f"""\
{SHA_A}  d01-thermal-dispatch/training/dictionaries/bounds.parquet
{SHA_D}  d02-single-hydro/training/solver/iterations.parquet
{SHA_E}  d03-two-hydro-cascade/simulation/solver/iterations.parquet
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path = Path(tmpdir) / "reference.md"
        act_path = Path(tmpdir) / "sha256.txt"
        ref_path.write_text(ref_md_content, encoding="utf-8")
        act_path.write_text(actual_content, encoding="utf-8")

        reference = _parse_reference_markdown(ref_path)
        assert len(reference) == 3, f"Expected 3 entries, got {len(reference)}"
        assert (
            reference["d01-thermal-dispatch/training/dictionaries/bounds.parquet"]
            == SHA_A
        )

        actual = _parse_sha256_file(act_path)
        assert len(actual) == 3

        unexpected, allowed, missing = compare(reference, actual)

        # d01 is byte-identical: no diff
        assert len(unexpected) == 0, f"Expected 0 unexpected diffs, got: {unexpected}"
        # d02 (training/solver/iterations.parquet) and d03 (simulation/solver)
        # are in the allowlist
        assert len(allowed) == 2, f"Expected 2 allowed diffs, got: {allowed}"
        assert len(missing) == 0

    # Test that an unexpected drift is detected correctly
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path = Path(tmpdir) / "reference.md"
        act_path = Path(tmpdir) / "sha256.txt"
        ref_md_content2 = f"""\
# Test

```
{SHA_A}  d01-thermal-dispatch/training/dictionaries/bounds.parquet
```
"""
        actual_content2 = f"""\
{SHA_D}  d01-thermal-dispatch/training/dictionaries/bounds.parquet
"""
        ref_path.write_text(ref_md_content2, encoding="utf-8")
        act_path.write_text(actual_content2, encoding="utf-8")

        reference2 = _parse_reference_markdown(ref_path)
        actual2 = _parse_sha256_file(act_path)
        unexpected2, allowed2, missing2 = compare(reference2, actual2)
        assert len(unexpected2) == 1, f"Expected 1 unexpected diff, got: {unexpected2}"
        assert len(allowed2) == 0
        assert len(missing2) == 0

    print("all self-tests passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
