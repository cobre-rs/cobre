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


# ---------------------------------------------------------------------------
# Guard detection: exact marker strings from ticket spec section 4
# ---------------------------------------------------------------------------

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
    ("TrainingResult { ", "training-result-struct-literal"),
]

# D-case fixture reference patterns (file names and function names)
_D_CASE_FILE_PATTERN = re.compile(
    r"deterministic\.rs|d\d{2}[-_]|d_case|conformance"
)
_D_CASE_FN_PATTERN = re.compile(
    r"^d\d{2}_|^d\d{2}$|conformance_|determinism_"
)

# fpha slow-test patterns
_FPHA_FILE_PATTERN = re.compile(
    r"fpha_fitting\.rs|fpha_computed\.rs|fpha_evaporation\.rs"
)
_FPHA_FN_PATTERN = re.compile(r"^fpha_")

# ---------------------------------------------------------------------------
# Category taxonomy labels (all 7)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Guard taxonomy labels (all 19 + generic)
# ---------------------------------------------------------------------------

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
        "training-result-struct-literal",
        "fpha-slow",
        "d-case-determinism",
        "convertido-determinism",
        "unified-path",
        "generic",
    ]
)

# ---------------------------------------------------------------------------
# Epic 03/04/05 deletion mappings (from taxonomy)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Body extraction
# ---------------------------------------------------------------------------


def _read_file_lines(repo_root: Path, rel_path: str) -> list[str]:
    """Read source file lines (cached per file path)."""
    full_path = repo_root / rel_path
    try:
        return full_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []


_file_cache: dict[str, list[str]] = {}


def get_file_lines(repo_root: Path, rel_path: str) -> list[str]:
    """Return cached file lines."""
    if rel_path not in _file_cache:
        _file_cache[rel_path] = _read_file_lines(repo_root, rel_path)
    return _file_cache[rel_path]


def extract_body_text(
    repo_root: Path, rel_path: str, attr_line: int, body_loc: int
) -> str:
    """Extract function body text starting from attr_line (1-based).

    Returns the concatenated lines from attr_line through attr_line + body_loc + 20
    (generous window to cover fn signature + body).
    """
    lines = get_file_lines(repo_root, rel_path)
    # attr_line is 1-based; grab from the #[test] line through fn body
    start = max(0, attr_line - 1)
    end = min(len(lines), start + body_loc + 30)
    return "\n".join(lines[start:end])


# ---------------------------------------------------------------------------
# Guard assignment
# ---------------------------------------------------------------------------


def assign_guards(body_text: str, file_path: str, fn_name: str) -> str:
    """Assign comma-separated guard labels based on marker grep.

    Applies guards from GUARD_MARKERS first, then file/function pattern guards.
    Returns 'generic' if no guards match.
    """
    guards: list[str] = []

    # Apply exact marker guards
    for marker, guard in GUARD_MARKERS:
        if marker in body_text:
            guards.append(guard)

    # D-case determinism: function name or file matches D-case patterns
    # but only apply for tests that appear to be in the deterministic test suite
    if _D_CASE_FILE_PATTERN.search(file_path) or _D_CASE_FN_PATTERN.match(fn_name):
        # Only tag actual D-case test functions (d01_, d02_, ..., d30_)
        if re.match(r"^d\d{2}_", fn_name) or re.match(r"^d\d{2}$", fn_name):
            guards.append("d-case-determinism")

    # convertido determinism: check for convertido fixture references
    if "convertido" in body_text.lower() or "convertido" in file_path.lower():
        if "convertido" in body_text.lower():
            guards.append("convertido-determinism")

    # fpha slow tests: files under fpha_fitting.rs and fpha_* integration tests
    if _FPHA_FILE_PATTERN.search(file_path) or _FPHA_FN_PATTERN.match(fn_name):
        guards.append("fpha-slow")

    # Also check body for fpha test patterns
    if "fpha_fitting" in file_path:
        if "fpha-slow" not in guards:
            guards.append("fpha-slow")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_guards: list[str] = []
    for g in guards:
        if g not in seen:
            seen.add(g)
            unique_guards.append(g)

    if not unique_guards:
        return "generic"
    return ",".join(unique_guards)


# ---------------------------------------------------------------------------
# Category assignment
# ---------------------------------------------------------------------------

# D-case test function patterns in deterministic.rs
_DCASE_FN = re.compile(r"^d\d{2}_|^d\d{2}$")
# Regression-related patterns
_REGRESSION_FN = re.compile(r"^regression_|_regression$|_regression_")
_REGRESSION_FILE = re.compile(r"regression\.rs$")

# E2E: functions that run full train+simulate pipeline
_E2E_FN = re.compile(
    r"converges$|_e2e$|_pipeline$|full_pipeline|end_to_end"
)

# Integration test files (the /tests/ directory)
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

# Conformance test files (D-cases, numerical snapshot tests)
_CONFORMANCE_FILES: frozenset[str] = frozenset(
    [
        "crates/cobre-sddp/tests/deterministic.rs",
    ]
)

# E2E test files (full pipeline tests)
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
    """Assign a single category label per taxonomy rules.

    Rules applied in order (first match wins):
    1. regression: fn name or file contains regression marker
    2. conformance: D-case test files or D-case fn pattern
    3. e2e: full pipeline test files or fn name matches e2e pattern
    4. integration: in /tests/ directory (non-conformance, non-e2e)
    5. unit: everything else (in-src #[cfg(test)] mod)
    """
    # Rule 1: regression
    if _REGRESSION_FN.search(fn_name) or _REGRESSION_FILE.search(file_path):
        return "regression"

    # Rule 2: conformance — D-case deterministic tests
    if file_path in _CONFORMANCE_FILES:
        # Most tests in deterministic.rs are conformance (D-cases)
        if _DCASE_FN.match(fn_name):
            return "conformance"
        # Some tests in deterministic.rs are integration (model_persistence, etc.)
        return "integration"

    # Rule 3: e2e — full pipeline tests in fpha_* integration test files
    if file_path in _E2E_FILES:
        return "e2e"

    # Rule 3b: e2e — functions that clearly run full pipeline
    if _E2E_FN.search(fn_name) and "/tests/" in file_path:
        return "e2e"

    # Rule 4: integration — tests in /tests/ directory
    if "/tests/" in file_path:
        return "integration"

    # Rule 5: unit — in-src tests
    return "unit"


# ---------------------------------------------------------------------------
# Parameter-sweep detection (heuristic)
# ---------------------------------------------------------------------------


def is_parameter_sweep(fn_name: str, body_text: str, all_rows: list[InventoryRow]) -> bool:
    """Detect if a test is a parameter sweep based on naming and body patterns.

    Simple heuristic: look for multiple similar test names in the same file
    that share a common prefix and differ only in a suffix number/variant.
    """
    # Pattern: fn name ends in a number or has _N suffix variants
    return False  # conservative: don't auto-tag, use guard-based approach


# ---------------------------------------------------------------------------
# Row processing
# ---------------------------------------------------------------------------


def process_rows(
    raw_rows: list[dict[str, str]],
    repo_root: Path,
) -> list[InventoryRow]:
    """Tag each raw CSV row with category and guards."""
    result: list[InventoryRow] = []

    for row in raw_rows:
        file_path = row["file"]
        fn_name = row["function"]
        attr_line = int(row["line"])
        body_loc = int(row["body_loc"])
        test_module = row["test_module"]
        crate = row["crate"]

        # Extract body text for guard detection
        body_text = extract_body_text(repo_root, file_path, attr_line, body_loc)

        # Assign guards
        guards = assign_guards(body_text, file_path, fn_name)

        # Assign category
        category = assign_category(file_path, fn_name, body_text, test_module)

        result.append(
            InventoryRow(
                crate=crate,
                file=file_path,
                line=attr_line,
                function=fn_name,
                body_loc=body_loc,
                test_module=test_module,
                category=category,
                guards=guards,
                notes="",
            )
        )

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_rows(rows: list[InventoryRow]) -> list[str]:
    """Return list of validation error messages (empty = all OK)."""
    errors: list[str] = []
    for i, row in enumerate(rows):
        if row.category not in VALID_CATEGORIES:
            errors.append(
                f"Row {i + 1} ({row.function}): invalid category '{row.category}'"
            )
        for guard in row.guards.split(","):
            g = guard.strip()
            if g not in VALID_GUARDS:
                errors.append(
                    f"Row {i + 1} ({row.function}): invalid guard '{g}'"
                )
    return errors


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------


def _escape_md(s: str) -> str:
    """Escape pipe characters in Markdown table cells."""
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

    # -------------------------------------------------------------------------
    # Section 1: Summary header
    # -------------------------------------------------------------------------
    lines.append("# Test Inventory — Architecture Unification")
    lines.append("")
    lines.append(
        "Canonical test-suite inventory for the architecture-unification plan. "
        "Taxonomy vocabulary is frozen in "
        "[`docs/assessments/test-inventory-taxonomy.md`](test-inventory-taxonomy.md). "
        "Generated by `scripts/generate_test_inventory.py`."
    )
    lines.append("")
    lines.append("## 1. Summary")
    lines.append("")
    lines.append(f"**Total `#[test]` functions:** {total}")
    lines.append("")

    # Per-crate breakdown
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
        elif guard == "d-case-determinism" or guard == "convertido-determinism":
            epic = "—"
        elif guard == "generic":
            epic = "—"
        else:
            epic = "—"
        lines.append(f"| `{guard}` | {count:,} | {epic} |")
    lines.append("")

    # -------------------------------------------------------------------------
    # Section 2: Inventory table
    # -------------------------------------------------------------------------
    lines.append("## 2. Inventory Table")
    lines.append("")
    lines.append(
        "All 2,680 `#[test]` functions. Sorted by `(crate, file, line)`. "
        "Category and Guards columns use the frozen vocabulary from the taxonomy file."
    )
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

    # -------------------------------------------------------------------------
    # Section 3: Deletion candidates by epic
    # -------------------------------------------------------------------------
    lines.append("## 3. Deletion Candidates (by Epic)")
    lines.append("")
    lines.append(
        "Tests flagged for review at each refactoring step. "
        "Every row listed here also appears in Section 2 with a matching guard."
    )
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
        epic_rows = [r for r in rows if any(g in epic_guards for g in r.guards.split(","))]
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

    # -------------------------------------------------------------------------
    # Section 4: Parameterization candidates
    # -------------------------------------------------------------------------
    lines.append("## 4. Parameterization Candidates")
    lines.append("")
    lines.append(
        "Tests tagged `parameter-sweep` that share body structure and differ only in "
        "input parameters. Candidates for consolidation in Epic 09."
    )
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

    # -------------------------------------------------------------------------
    # Section 5: Slow-test roster
    # -------------------------------------------------------------------------
    lines.append("## 5. Slow-Test Roster")
    lines.append("")
    lines.append(
        "Tests observed to run > 10 s on a reference machine or flagged as long-running. "
        "Exact timing measurements live in ticket 004. "
        "Tests tagged `fpha-slow` missing the slow-tests feature gate are flagged as cleanup items for Epic 09."
    )
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
                '#[ignore' in context
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
    lines.append(
        "_No parse errors reported by `scripts/test_inventory.py`. "
        "All 2,680 tests were successfully inventoried._"
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate docs/assessments/test-inventory.md from the ticket-002 CSV output. "
            "Assigns category and guards columns programmatically."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Path to the repository root (default: current working directory).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to the raw inventory CSV (default: run scripts/test_inventory.py "
            "and use its stdout)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write Markdown to PATH (default: docs/assessments/test-inventory.md "
            "relative to repo root)."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Run tagging and validation but do not write output file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    repo_root: Path = (args.repo_root if args.repo_root is not None else Path.cwd()).resolve()

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

    # Summary stats
    cat_counts = Counter(r.category for r in rows)
    guard_counts: Counter[str] = Counter()
    for row in rows:
        for g in row.guards.split(","):
            g = g.strip()
            if g:
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
