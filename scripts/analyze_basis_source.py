#!/usr/bin/env python3
"""analyze_basis_source.py.

Analyzer for the `basis_source` column in
`training/solver/iterations.parquet`, introduced by Epic 02.

Produces two reports on stdout:

  Report A: Per-(iteration, stage) cache-hit rate table for backward-pass
            ω=0 rows at iteration >= 2. Columns: iteration, stage, n_rows,
            backward, forward, none, hit_rate.

  Report B: Overall summary line plus cold-start (iter=1) invariant line.

The analyzer is the tool used by Epic 03 A/B #3 to validate that the
backward-basis cache fires at scale in production runs and to surface the
three failure modes distinguished by the parquet:

  - R4 infeasibility (iter >= 2, basis_source is NULL on fresh iteration).
  - Broadcast failure (per-rank divergence in hit-rate summary).
  - Cold-start behaviour (iter 1 rows uniformly basis_source = 2 or NULL).

Usage:
    python3 scripts/analyze_basis_source.py <case-output-dir>
    python3 scripts/analyze_basis_source.py --parquet path/to/iterations.parquet

Exit codes:
  0 -- analysis produced successfully
  1 -- schema mismatch (missing `basis_source` column or wrong type)
  2 -- prerequisite missing (parquet file not found, polars unavailable)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import polars as pl
except ImportError:
    print(
        "ERROR: polars is not available. Run `uv pip install polars` in the active venv.",
        file=sys.stderr,
    )
    sys.exit(2)


RULE = "=" * 72
SUB_RULE = "-" * 72

# BasisSource discriminant values (must match cobre_sddp::solver_stats::BasisSource).
BASIS_BACKWARD = 1
BASIS_FORWARD = 2


def _i(x: object) -> int:
    """Coerce a polars scalar (possibly None) to int."""
    if x is None:
        return 0
    try:
        return int(float(x))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _resolve_parquet(args: argparse.Namespace) -> Path:
    """Return the path to iterations.parquet from either --parquet or case_dir."""
    if args.parquet is not None:
        path = Path(args.parquet).expanduser().resolve()
        if not path.exists():
            print(
                f"ERROR: parquet file not found: {path}",
                file=sys.stderr,
            )
            sys.exit(2)
        return path

    root = Path(args.case_dir).expanduser().resolve()
    candidates = [
        root / "training" / "solver" / "iterations.parquet",
        root / "output" / "training" / "solver" / "iterations.parquet",
        root / "iterations.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    print(
        f"ERROR: iterations.parquet not found under {root}. "
        "Tried training/solver/, output/training/solver/, and the root itself.",
        file=sys.stderr,
    )
    sys.exit(2)


def _check_schema(df: pl.DataFrame) -> None:
    """Exit with code 1 if the parquet is missing the `basis_source` column."""
    if "basis_source" not in df.columns:
        print(
            "ERROR: `basis_source` column not found in parquet schema.\n"
            "  Expected: Int32 nullable column added by Epic 02 ticket-003.\n"
            "  The parquet was likely produced by a pre-Epic-02 build of Cobre.\n"
            "  Re-run training with a version >= 0.5.x to populate the column.",
            file=sys.stderr,
        )
        sys.exit(1)


def _header(title: str) -> None:
    print(RULE)
    print(title)
    print(RULE)


def _report_per_iter_stage(df: pl.DataFrame) -> None:
    """Report A: per-(iteration, stage) cache-hit rate table."""
    _header(
        "Report A: backward ω=0 iter >= 2  —  per-(iteration, stage) cache-hit rate"
    )

    filtered = df.filter(
        (pl.col("phase") == "backward")
        & (pl.col("opening") == 0)
        & (pl.col("iteration") >= 2)
    )

    if len(filtered) == 0:
        print("(no backward ω=0 iter >= 2 rows found)\n")
        return

    grouped = (
        filtered.group_by(["iteration", "stage"])
        .agg(
            pl.len().alias("n_rows"),
            (pl.col("basis_source") == BASIS_BACKWARD).sum().alias("backward_count"),
            (pl.col("basis_source") == BASIS_FORWARD).sum().alias("forward_count"),
            pl.col("basis_source").is_null().sum().alias("none_count"),
        )
        .sort(["iteration", "stage"])
    )

    # Fixed-width column headers.
    print(
        f"{'iteration':>10}  {'stage':>6}  {'n_rows':>7}  {'backward':>9}  "
        f"{'forward':>8}  {'none':>6}  {'hit_rate':>9}"
    )
    print(SUB_RULE)

    for row in grouped.iter_rows(named=True):
        n = _i(row["n_rows"]) or 1
        bwd = _i(row["backward_count"])
        fwd = _i(row["forward_count"])
        none = _i(row["none_count"])
        rate = bwd / n
        print(
            f"{row['iteration']:>10}  {row['stage']:>6}  {n:>7}  "
            f"{bwd:>9}  {fwd:>8}  {none:>6}  {rate:>9.3f}"
        )

    print()


def _report_summary(df: pl.DataFrame) -> None:
    """Report B: overall summary line + cold-start invariant line."""
    _header("Report B: summary  (iter >= 2 ω=0 backward)  +  cold-start invariant")

    # --- Iter >= 2 summary ---
    warm = df.filter(
        (pl.col("phase") == "backward")
        & (pl.col("opening") == 0)
        & (pl.col("iteration") >= 2)
    )

    n_warm = len(warm)
    if n_warm == 0:
        print("SUMMARY (iter >= 2): no rows found\n")
    else:
        summary = warm.select(
            pl.len().alias("total_rows"),
            (pl.col("basis_source") == BASIS_BACKWARD).sum().alias("backward"),
            (pl.col("basis_source") == BASIS_FORWARD).sum().alias("forward"),
            pl.col("basis_source").is_null().sum().alias("none"),
        )
        row = summary.row(0, named=True)
        total = _i(row["total_rows"]) or 1
        bwd = _i(row["backward"])
        fwd = _i(row["forward"])
        none = _i(row["none"])
        hit_rate = bwd / total

        print(
            f"SUMMARY: total_rows={total}, "
            f"backward={bwd} ({bwd / total:.3f}), "
            f"forward={fwd} ({fwd / total:.3f}), "
            f"none={none} ({none / total:.3f}), "
            f"cache_hit_rate={hit_rate:.3f}"
        )

    # --- Cold-start (iter 1) ---
    cold = df.filter(
        (pl.col("phase") == "backward")
        & (pl.col("opening") == 0)
        & (pl.col("iteration") == 1)
    )

    n_cold = len(cold)
    if n_cold == 0:
        print("COLD_START (iter=1): no rows found")
    else:
        fwd_cold = _i(
            cold.select((pl.col("basis_source") == BASIS_FORWARD).sum()).item()
        )
        none_cold = _i(cold.select(pl.col("basis_source").is_null().sum()).item())
        bwd_cold = _i(
            cold.select((pl.col("basis_source") == BASIS_BACKWARD).sum()).item()
        )
        print(
            f"COLD_START (iter=1): n={n_cold}, forward={fwd_cold}, none={none_cold}",
            end="",
        )
        if bwd_cold > 0:
            print(
                f"  WARNING: {bwd_cold} row(s) have basis_source=Backward on iter 1 "
                "(unexpected — cache should be empty on cold start)",
                end="",
            )
        print()

    print()


def main() -> int:
    """Entry point.  Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the `basis_source` column in training/solver/iterations.parquet "
            "to report backward-basis cache hit rates per (iteration, stage)."
        ),
    )
    parser.add_argument(
        "case_dir",
        nargs="?",
        default=None,
        help="Path to a case output directory (auto-discovers training/solver/iterations.parquet).",
    )
    parser.add_argument(
        "--parquet",
        default=None,
        help="Direct path to an iterations.parquet file (overrides case_dir).",
    )
    args = parser.parse_args()

    if args.case_dir is None and args.parquet is None:
        parser.error("Provide either case_dir or --parquet")

    parquet_path = _resolve_parquet(args)
    print(f"Reading {parquet_path}\n")

    df = pl.read_parquet(parquet_path)
    if len(df) == 0:
        print("ERROR: iterations.parquet is empty.", file=sys.stderr)
        return 2

    _check_schema(df)

    _report_per_iter_stage(df)
    _report_summary(df)

    return 0


if __name__ == "__main__":
    sys.exit(main())
