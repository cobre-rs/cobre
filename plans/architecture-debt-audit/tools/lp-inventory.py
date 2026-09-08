#!/usr/bin/env python3
"""Emit lp-inventory.json: the kernel-boundary census of crates/cobre-sddp/src/lp.

The machine-readable handoff Epic 9 classifies engine-neutral emission against SDDP
geometry from, and the evidence base for the roadmap's "fifth to a quarter of lp/" claim.
Python 3 stdlib only (no third-party import). Read-only: writes two untracked artifacts
under plans/, touches no tracked file, runs no cargo command.

`non_test_lines` is NOT invented here: it mirrors scripts/loc-stats.sh's awk classifier
line-for-line so the numbers reconcile with the crate-level figures the station quotes.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import pathlib
import re
import subprocess
import sys

TEST_FILE_NAMES = {"tests.rs", "test_support.rs"}
CFG_TEST = re.compile(r"^#\[cfg.*\(test[,)]")
CFG_NOT_TEST = re.compile(r"not\s*\(\s*test")
MAX_SYMBOLS = 12
ITEM = re.compile(
    r"^(?:pub(?:\([^)]*\))?\s+)?"  # pub | pub(crate) | pub(super) | pub(in ...)
    r"(?:default\s+|const\s+|async\s+|unsafe\s+|extern\s+\"[^\"]*\"\s+)*"
    r"(fn|struct|enum|trait|type|impl|mod|use|const|static|macro_rules!)\b"
    r"\s*([^\s({<;=]*)"
)
NON_TEST_RULE = (
    "mirrors scripts/loc-stats.sh: sibling tests.rs and test_support.rs count zero; "
    'inline #[cfg(test)] items excluded; #[cfg(any(test, feature = "test-support"))] '
    "counts as test; #[cfg(not(test))] does not"
)


@dataclasses.dataclass(frozen=True)
class FileRecord:
    """One row of lp-inventory.json; the unit Epic 9 classifies engine-neutral vs SDDP geometry."""

    path: str
    total_lines: int
    non_test_lines: int
    top_symbols: list[str]

    def as_json(self) -> dict[str, object]:
        return {
            "path": self.path,
            "total_lines": self.total_lines,
            "non_test_lines": self.non_test_lines,
            "top_symbols": list(self.top_symbols),
        }


def is_whole_file_test(path: str) -> bool:
    return (
        path.rsplit("/", 1)[-1] in TEST_FILE_NAMES
        or "/tests/" in path
        or "/benches/" in path
    )


def _advance_test_state(
    raw: str, in_test: bool, pending: bool, tind: int
) -> tuple[bool, bool, int, bool]:
    """Advance the loc-stats.sh test-region state machine one line.

    Returns (in_test, pending, tind, skip); skip is True when the line is inside a test
    region or is the attribute/opening that introduces one, so it must not count as
    non-test and must not yield a symbol. Shared by non_test_line_count and top_symbols so
    the two can never diverge from the yardstick.
    """
    t = raw.strip()
    indent = len(raw) - len(raw.lstrip(" "))
    if in_test:
        if t.startswith("}") and indent == tind:
            return False, pending, tind, True
        return True, pending, tind, True
    if pending:  # multi-line #[allow(...)] between the attribute and its `mod X {`
        if "{" in raw:
            return True, False, tind, True
        if t.endswith(";"):
            return (
                False,
                False,
                tind,
                True,
            )  # `#[cfg(test)] mod tests;` — one line, no block
        return False, True, tind, True
    if CFG_TEST.match(t) and not CFG_NOT_TEST.search(t):
        if "{" in raw:
            return True, False, indent, True
        if not t.endswith(";"):
            return False, True, indent, True
        return False, False, indent, True
    return in_test, pending, tind, False


def non_test_line_count(path: str, lines: list[str]) -> int:
    if is_whole_file_test(path):
        return 0
    kept = 0
    in_test = pending = False
    tind = -1
    for raw in lines:
        in_test, pending, tind, skip = _advance_test_state(raw, in_test, pending, tind)
        if not skip:
            kept += 1
    return kept


def top_symbols(path: str, lines: list[str]) -> list[str]:
    """Column-0 item declarations in declaration order, capped at MAX_SYMBOLS.

    Whole-file test modules are scanned in full (their non-test region is empty by
    construction), so no record ever gets an empty list.
    """
    scan_all = is_whole_file_test(path)
    out: list[str] = []
    in_test = pending = False
    tind = -1
    for raw in lines:
        if not scan_all:
            in_test, pending, tind, skip = _advance_test_state(
                raw, in_test, pending, tind
            )
            if skip:
                continue
        if raw[:1] in (" ", "\t") or not raw.strip():
            continue
        m = ITEM.match(raw.rstrip())
        if not m:
            continue
        kind, name = m.group(1), m.group(2).rstrip(":,")
        out.append(f"{kind} {name}".strip() if name else kind)
        if len(out) == MAX_SYMBOLS:
            break
    return out


def discover(root: pathlib.Path) -> list[str]:
    """Every .rs file under lp/, repo-relative POSIX, byte-sorted so re-runs diff cleanly."""
    return sorted(p.as_posix() for p in root.rglob("*.rs") if p.is_file())


def build_record(path: str) -> FileRecord:
    lines = (
        pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    )
    return FileRecord(
        path=path,
        total_lines=len(lines),
        non_test_lines=non_test_line_count(path, lines),
        top_symbols=top_symbols(path, lines),
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="lp/ inventory for the kernel-boundary classification"
    )
    ap.add_argument(
        "--root", type=pathlib.Path, default=pathlib.Path("crates/cobre-sddp/src/lp")
    )
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(
            "plans/architecture-debt-audit/measurements/lp-inventory.json"
        ),
    )
    ap.add_argument(
        "--baseline", default=None, help="pinned SHA; defaults to `git rev-parse HEAD`"
    )
    ap.add_argument(
        "--expect-count", type=int, default=30, help="drift guard; 0 disables it"
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    baseline = (
        args.baseline
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )

    paths = discover(args.root)
    if args.expect_count and len(paths) != args.expect_count:
        print(
            f"drift: found {len(paths)} .rs under {args.root}, expected {args.expect_count}",
            file=sys.stderr,
        )
        return 2

    records = [build_record(p) for p in paths]
    totals = {
        "file_count": len(records),
        "total_lines": sum(r.total_lines for r in records),
        "non_test_lines": sum(r.non_test_lines for r in records),
    }
    envelope: dict[str, object] = {
        "baseline": baseline,
        "generatedAt": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "nonTestRule": NON_TEST_RULE,
        "files": [r.as_json() for r in records],
        "totals": totals,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
    print(
        f"lp-inventory: {totals['file_count']} files, {totals['total_lines']} lines, "
        f"{totals['non_test_lines']} non-test -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
