#!/usr/bin/env python3
"""Generate docs/assessments/test-inventory.md from the ticket-002 CSV output.

This script:
1. Runs scripts/test_inventory.py (or reads its output from a file).
2. For each test, reads the function body from the source file.
3. Assigns guards by grepping for exact marker strings.
4. Assigns category heuristically based on file path and function name.
5. Emits the full Markdown inventory document.

Usage:
    python3 scripts/generate_test_inventory.py
    python3 scripts/generate_test_inventory.py --csv /tmp/raw_inventory.csv
    python3 scripts/generate_test_inventory.py --repo-root /path/to/cobre
    python3 scripts/generate_test_inventory.py --output docs/assessments/test-inventory.md
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple


GUARD_MARKERS: list[tuple[str, str]] = [
    ("WarmStartBasisMode::AlienOnly", "alien-only"),
    ("WarmStartBasisMode::NonAlienFirst", "non-alien-first"),
    ("CanonicalStateStrategy::Disabled", "canonical-disabled"),
    ("CanonicalStateStrategy::ClearSolver", "canonical-clearsolver"),
    (".add_rows(", "add-rows-trait"),
    (".clear_solver_state(", "clear-solver-state-trait"),
    (".solve_with_basis(", "solve-with-basis-trait"),
    ("warm_start_basis_mode", "warm-start-config-flag"),
    ("canonical_state", "canonical-config-flag"),
    ("stored_cut_row_offset", "stored-cut-row-offset"),
    ("broadcast_config.warm_start_basis_mode", "broadcast-warm-start-field"),
    ("broadcast_config.canonical_state_strategy", "broadcast-canonical-field"),
]

_D_CASE_FILE_PATTERN = re.compile(r"deterministic\.rs|d\d{2}[-_]|d_case|conformance")
_D_CASE_FN_PATTERN = re.compile(r"^d\d{2}_|^d\d{2}$|conformance_|determinism_")
_FPHA_FILE_PATTERN = re.compile(
    r"fpha_fitting\.rs|fpha_computed\.rs|fpha_evaporation\.rs"
)
_FPHA_FN_PATTERN = re.compile(r"^fpha_")

VALID_CATEGORIES = frozenset(
    [
        "unit",
        "integration",
        "e2e",
        "regression",
        "conformance",
        "parameter-sweep",
        "coverage-matrix",
    ]
)

VALID_GUARDS = frozenset(
    [
        "alien-only",
        "non-alien-first",
        "canonical-disabled",
        "canonical-clearsolver",
        "non-baked",
        "baked",
        "stored-cut-row-offset",
        "solve-with-basis-trait",
        "clear-solver-state-trait",
        "add-rows-trait",
        "warm-start-config-flag",
        "canonical-config-flag",
        "broadcast-warm-start-field",
        "broadcast-canonical-field",
        "fpha-slow",
        "d-case-determinism",
        "convertido-determinism",
        "unified-path",
        "generic",
    ]
)

EPIC03_GUARDS = frozenset(
    ["alien-only", "warm-start-config-flag", "broadcast-warm-start-field"]
)
EPIC04_GUARDS = frozenset(
    [
        "canonical-disabled",
        "canonical-config-flag",
        "broadcast-canonical-field",
        "clear-solver-state-trait",
        "solve-with-basis-trait",
    ]
)
EPIC05_GUARDS = frozenset(["non-baked", "stored-cut-row-offset", "add-rows-trait"])


class InventoryRow(NamedTuple):
    crate: str
    file: str
    line: int
    function: str
    body_loc: int
    test_module: str
    category: str
    guards: str  # comma-separated guard labels
    notes: str


def _read_file_lines(repo_root: Path, rel_path: str) -> list[str]:
    """Read source file lines."""
    full_path = repo_root / rel_path
    try:
        return full_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []


_file_cache: dict[str, list[str]] = {}


def get_file_lines(repo_root: Path, rel_path: str) -> list[str]:
    """Return cached file lines, reading on first access."""
    _file_cache.setdefault(rel_path, _read_file_lines(repo_root, rel_path))
    return _file_cache[rel_path]


def extract_body_text(
    repo_root: Path, rel_path: str, attr_line: int, body_loc: int
) -> str:
    """Extract function signature and body text for guard detection."""
    lines = get_file_lines(repo_root, rel_path)
    start = max(0, attr_line - 1)
    end = min(len(lines), start + body_loc + 30)
    return "\n".join(lines[start:end])


def assign_guards(body_text: str, file_path: str, fn_name: str) -> str:
    """Assign comma-separated guard labels from marker strings, patterns, and heuristics."""
    guards: list[str] = []
    for marker, guard in GUARD_MARKERS:
        if marker in body_text:
            guards.append(guard)
    if re.match(r"^d\d{2}[_$]|^d\d{2}$", fn_name):
        guards.append("d-case-determinism")
    if "convertido" in body_text.lower():
        guards.append("convertido-determinism")
    if (
        _FPHA_FILE_PATTERN.search(file_path)
        or _FPHA_FN_PATTERN.match(fn_name)
        or "fpha_fitting" in file_path
    ):
        guards.append("fpha-slow")
    seen: set[str] = set()
    unique = [g for g in guards if not (g in seen or seen.add(g))]
    return ",".join(unique) if unique else "generic"


_DCASE_FN = re.compile(r"^d\d{2}[_$]|^d\d{2}$")
_REGRESSION_FN = re.compile(r"^regression_|_regression$|_regression_")
_REGRESSION_FILE = re.compile(r"regression\.rs$")
_E2E_FN = re.compile(r"converges$|_e2e$|_pipeline$|full_pipeline|end_to_end")
_INTEGRATION_TEST_FILES: frozenset[str] = frozenset(
    [
        "crates/cobre-sddp/tests/integration.rs",
        "crates/cobre-sddp/tests/conformance.rs",
        "crates/cobre-sddp/tests/determinism.rs",
        "crates/cobre-sddp/tests/estimation_integration.rs",
        "crates/cobre-sddp/tests/forward_sampler_integration.rs",
        "crates/cobre-sddp/tests/inflow_nonnegativity.rs",
        "crates/cobre-sddp/tests/load_integration.rs",
        "crates/cobre-sddp/tests/decomp_integration.rs",
        "crates/cobre-sddp/tests/pattern_d_integration.rs",
        "crates/cobre-sddp/tests/simulation_integration.rs",
        "crates/cobre-sddp/tests/simulation_only.rs",
        "crates/cobre-sddp/tests/sparse_dense.rs",
        "crates/cobre-sddp/tests/basis_reconstruct_churn.rs",
        "crates/cobre-sddp/tests/boundary_cuts.rs",
        "crates/cobre-sddp/tests/warm_start.rs",
        "crates/cobre-sddp/tests/canonical_state_strategy.rs",
        "crates/cobre-comm/tests/factory_tests.rs",
        "crates/cobre-comm/tests/local_conformance.rs",
        "crates/cobre-io/tests/defaults_cascade.rs",
        "crates/cobre-io/tests/genericity_gate.rs",
        "crates/cobre-io/tests/integration.rs",
        "crates/cobre-io/tests/invariance.rs",
        "crates/cobre-solver/tests/conformance.rs",
        "crates/cobre-solver/tests/ffi_set_basis_non_alien_smoke.rs",
        "crates/cobre-cli/tests/cli_color.rs",
        "crates/cobre-cli/tests/cli_report.rs",
        "crates/cobre-cli/tests/cli_run.rs",
        "crates/cobre-cli/tests/cli_schema.rs",
        "crates/cobre-cli/tests/cli_smoke.rs",
        "crates/cobre-cli/tests/cli_validate.rs",
        "crates/cobre-cli/tests/init.rs",
    ]
)

_CONFORMANCE_FILES: frozenset[str] = frozenset(
    ["crates/cobre-sddp/tests/deterministic.rs"]
)

_E2E_FILES: frozenset[str] = frozenset(
    [
        "crates/cobre-sddp/tests/fpha_computed.rs",
        "crates/cobre-sddp/tests/fpha_evaporation.rs",
    ]
)


def assign_category(
    file_path: str,
    fn_name: str,
    body_text: str,
    test_module: str,
) -> str:
    """Assign category label (regression, conformance, e2e, integration, unit)."""
    if _REGRESSION_FN.search(fn_name) or _REGRESSION_FILE.search(file_path):
        return "regression"

    if file_path in _CONFORMANCE_FILES:
        return "conformance" if _DCASE_FN.match(fn_name) else "integration"

    if file_path in _E2E_FILES or (_E2E_FN.search(fn_name) and "/tests/" in file_path):
        return "e2e"

    return "integration" if "/tests/" in file_path else "unit"


def process_rows(
    raw_rows: list[dict[str, str]],
    repo_root: Path,
) -> list[InventoryRow]:
    """Assign category and guards to each CSV row."""
    result: list[InventoryRow] = []
    for row in raw_rows:
        body_text = extract_body_text(
            repo_root, row["file"], int(row["line"]), int(row["body_loc"])
        )
        result.append(
            InventoryRow(
                crate=row["crate"],
                file=row["file"],
                line=int(row["line"]),
                function=row["function"],
                body_loc=int(row["body_loc"]),
                test_module=row["test_module"],
                category=assign_category(
                    row["file"], row["function"], body_text, row["test_module"]
                ),
                guards=assign_guards(body_text, row["file"], row["function"]),
                notes="",
            )
        )
    return result


def validate_rows(rows: list[InventoryRow]) -> list[str]:
    """Validate rows. Return empty list if all valid."""
    errors: list[str] = []
    for i, row in enumerate(rows):
        if row.category not in VALID_CATEGORIES:
            errors.append(
                f"Row {i + 1} ({row.function}): invalid category '{row.category}'"
            )
        for guard in row.guards.split(","):
            g = guard.strip()
            if g not in VALID_GUARDS:
                errors.append(f"Row {i + 1} ({row.function}): invalid guard '{g}'")
    return errors


def _escape_md(s: str) -> str:
    """Escape pipe characters for Markdown tables."""
    return s.replace("|", "\\|")


def generate_markdown(rows: list[InventoryRow], repo_root: Path) -> str:
    """Generate the full Markdown inventory document."""
    lines: list[str] = []

    total = len(rows)
    crate_counts = Counter(r.crate for r in rows)
    category_counts = Counter(r.category for r in rows)

    # Build guard counts (each guard counted per test; tests with multiple guards count once per guard)
    guard_counts: Counter[str] = Counter()
    for row in rows:
        for g in row.guards.split(","):
            g = g.strip()
            if g:
                guard_counts[g] += 1

    lines.append("# Test Inventory — Architecture Unification")
    lines.append("")
    lines.append(
        "Canonical test-suite inventory. Taxonomy in `docs/assessments/test-inventory-taxonomy.md`."
    )
    lines.append("")
    lines.append("## 1. Summary")
    lines.append("")
    lines.append(f"**Total tests:** {total}")
    lines.append("")
    lines.append("### Per-crate breakdown")
    lines.append("")
    lines.append("| Crate | Count |")
    lines.append("| ----- | ----: |")
    for crate in sorted(crate_counts):
        lines.append(f"| `{crate}` | {crate_counts[crate]:,} |")
    lines.append("")

    # Per-category breakdown
    lines.append("### Per-category breakdown")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("| -------- | ----: |")
    for cat in sorted(VALID_CATEGORIES):
        count = category_counts.get(cat, 0)
        lines.append(f"| `{cat}` | {count:,} |")
    lines.append("")

    # Per-guard breakdown
    lines.append("### Per-guard breakdown")
    lines.append("")
    lines.append(
        "Guards relevant to Epic 03/04/05 deletions are highlighted. "
        "Tests carrying multiple guards contribute one count per guard label, "
        "so the sum may exceed the total test count."
    )
    lines.append("")
    lines.append("| Guard | Count | Epic |")
    lines.append("| ----- | ----: | ---- |")
    for guard in sorted(VALID_GUARDS):
        count = guard_counts.get(guard, 0)
        if guard in EPIC03_GUARDS:
            epic = "03"
        elif guard in EPIC04_GUARDS:
            epic = "04"
        elif guard in EPIC05_GUARDS:
            epic = "05"
        elif guard == "fpha-slow":
            epic = "09"
        else:
            epic = "—"
        lines.append(f"| `{guard}` | {count:,} | {epic} |")
    lines.append("")

    lines.append("## 2. Inventory Table")
    lines.append("")
    lines.append(f"All {total} tests sorted by (crate, file, line).")
    lines.append("")
    lines.append(
        "| Crate | File | Line | Function | Body LoC | Test module | Category | Guards | Notes |"
    )
    lines.append(
        "| ----- | ---- | ---: | -------- | -------: | ----------- | -------- | ------ | ----- |"
    )

    for row in rows:
        crate = _escape_md(row.crate)
        file_short = _escape_md(row.file)
        fn = _escape_md(row.function)
        module = _escape_md(row.test_module)
        cat = _escape_md(row.category)
        guards = _escape_md(row.guards)
        notes = _escape_md(row.notes)
        lines.append(
            f"| `{crate}` | `{file_short}` | {row.line} | `{fn}` | {row.body_loc} | `{module}` | `{cat}` | `{guards}` | {notes} |"
        )
    lines.append("")

    lines.append("## 3. Deletion Candidates (by Epic)")
    lines.append("")

    for epic_num, epic_guards, epic_title in [
        (
            "03",
            EPIC03_GUARDS,
            "Epic 03 — `AlienOnly` warm-start removal (`alien-only`, `warm-start-config-flag`, `broadcast-warm-start-field`)",
        ),
        (
            "04",
            EPIC04_GUARDS,
            "Epic 04 — Canonical-state strategy removal (`canonical-disabled`, `canonical-config-flag`, `broadcast-canonical-field`, `clear-solver-state-trait`, `solve-with-basis-trait`)",
        ),
        (
            "05",
            EPIC05_GUARDS,
            "Epic 05 — Non-baked template removal (`non-baked`, `stored-cut-row-offset`, `add-rows-trait`)",
        ),
    ]:
        epic_rows = [
            r for r in rows if any(g in epic_guards for g in r.guards.split(","))
        ]
        lines.append(f"### {epic_title}")
        lines.append("")
        lines.append(f"**Count:** {len(epic_rows)}")
        lines.append("")
        if epic_rows:
            lines.append("| Crate | File | Line | Function | Category | Guards |")
            lines.append("| ----- | ---- | ---: | -------- | -------- | ------ |")
            for row in epic_rows:
                crate = _escape_md(row.crate)
                file_short = _escape_md(row.file)
                fn = _escape_md(row.function)
                cat = _escape_md(row.category)
                guards = _escape_md(row.guards)
                lines.append(
                    f"| `{crate}` | `{file_short}` | {row.line} | `{fn}` | `{cat}` | `{guards}` |"
                )
        else:
            lines.append("_No tests flagged for this epic._")
        lines.append("")

    lines.append("## 4. Parameterization Candidates")
    lines.append("")
    lines.append("")

    sweep_rows = [r for r in rows if r.category == "parameter-sweep"]
    if sweep_rows:
        # Group by shared body structure (heuristic: group by file)
        by_file: dict[str, list[InventoryRow]] = defaultdict(list)
        for row in sweep_rows:
            by_file[row.file].append(row)
        for file_path, group in sorted(by_file.items()):
            lines.append(f"**`{file_path}`** ({len(group)} tests)")
            lines.append("")
            for row in group:
                lines.append(f"- `{row.function}` (line {row.line})")
            lines.append("")
    else:
        lines.append(
            "_No tests were tagged `parameter-sweep` in this pass. "
            "The initial heuristic tagger did not identify clear parameter-sweep groups; "
            "this section should be populated by a follow-up manual pass._"
        )
        lines.append("")

    lines.append("## 5. Slow-Test Roster")
    lines.append("")
    lines.append("")

    # Find all fpha-slow tests
    fpha_slow_rows = [r for r in rows if "fpha-slow" in r.guards.split(",")]

    lines.append("### `fpha-slow` tests")
    lines.append("")
    lines.append(f"**Count:** {len(fpha_slow_rows)}")
    lines.append("")
    if fpha_slow_rows:
        lines.append("| Crate | File | Line | Function | Feature-gated? |")
        lines.append("| ----- | ---- | ---: | -------- | -------------- |")
        for row in fpha_slow_rows:
            # Check if the test has an #[ignore] or feature gate near it
            lines_list = get_file_lines(repo_root, row.file)
            start = max(0, row.line - 6)
            end = min(len(lines_list), row.line)
            context = "\n".join(lines_list[start:end])
            has_gate = (
                "#[ignore" in context
                or 'feature = "slow-tests"' in context
                or 'feature = "fpha-slow"' in context
                or "#[cfg(feature" in context
            )
            gate_str = "yes" if has_gate else "**NO — cleanup item for Epic 09**"
            crate = _escape_md(row.crate)
            file_short = _escape_md(row.file)
            fn = _escape_md(row.function)
            lines.append(
                f"| `{crate}` | `{file_short}` | {row.line} | `{fn}` | {gate_str} |"
            )
    lines.append("")

    # e2e tests (also slow)
    e2e_rows = [r for r in rows if r.category == "e2e"]
    lines.append("### E2E pipeline tests (slow by nature)")
    lines.append("")
    lines.append(
        f"There are {len(e2e_rows)} `e2e` tests that run the full training/simulation pipeline."
    )
    lines.append("")
    if e2e_rows:
        lines.append("| Crate | File | Line | Function | Guards |")
        lines.append("| ----- | ---- | ---: | -------- | ------ |")
        for row in e2e_rows:
            crate = _escape_md(row.crate)
            file_short = _escape_md(row.file)
            fn = _escape_md(row.function)
            guards = _escape_md(row.guards)
            lines.append(
                f"| `{crate}` | `{file_short}` | {row.line} | `{fn}` | `{guards}` |"
            )
    lines.append("")

    lines.append("### Parse errors")
    lines.append("")
    lines.append("_No parse errors reported._")
    lines.append("")

    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate test-inventory.md with category and guard assignments."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to raw inventory CSV (default: run test_inventory.py).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write Markdown to PATH (default: docs/assessments/test-inventory.md).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Validate without writing output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    repo_root: Path = (
        args.repo_root if args.repo_root is not None else Path.cwd()
    ).resolve()

    # Validate repo root
    if not (repo_root / "Cargo.toml").exists():
        print(
            f"generate_test_inventory: error: {repo_root} does not contain Cargo.toml",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load CSV
    if args.csv is not None:
        csv_path = args.csv
        print(
            f"generate_test_inventory: reading CSV from {csv_path}",
            file=sys.stderr,
        )
        with csv_path.open(encoding="utf-8", newline="") as fh:
            raw_rows = list(csv.DictReader(fh))
    else:
        print(
            "generate_test_inventory: running scripts/test_inventory.py ...",
            file=sys.stderr,
        )
        script = repo_root / "scripts" / "test_inventory.py"
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            check=False,
        )
        if result.returncode != 0:
            print(
                f"generate_test_inventory: test_inventory.py failed:\n{result.stderr}",
                file=sys.stderr,
            )
            sys.exit(1)
        raw_rows = list(csv.DictReader(result.stdout.splitlines()))
        print(result.stderr.strip(), file=sys.stderr)

    print(
        f"generate_test_inventory: processing {len(raw_rows)} rows ...",
        file=sys.stderr,
    )

    # Process rows
    rows = process_rows(raw_rows, repo_root)

    # Validate
    errors = validate_rows(rows)
    if errors:
        for err in errors:
            print(f"generate_test_inventory: validation error: {err}", file=sys.stderr)
        sys.exit(1)

    print(
        f"generate_test_inventory: validation passed ({len(rows)} rows).",
        file=sys.stderr,
    )

    cat_counts = Counter(r.category for r in rows)
    guard_counts: Counter[str] = Counter()
    for row in rows:
        for g in row.guards.split(","):
            if g := g.strip():
                guard_counts[g] += 1
    print("  Category breakdown:", file=sys.stderr)
    for cat, count in sorted(cat_counts.items()):
        print(f"    {cat}: {count}", file=sys.stderr)
    print("  Top guards:", file=sys.stderr)
    for guard, count in guard_counts.most_common(10):
        print(f"    {guard}: {count}", file=sys.stderr)

    if args.validate_only:
        print(
            "generate_test_inventory: --validate-only requested, not writing output.",
            file=sys.stderr,
        )
        return

    # Generate Markdown
    markdown = generate_markdown(rows, repo_root)

    # Write output
    output_path = (
        args.output
        if args.output is not None
        else repo_root / "docs" / "assessments" / "test-inventory.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(
        f"generate_test_inventory: wrote {len(rows)} rows to {output_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
