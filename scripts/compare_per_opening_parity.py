#!/usr/bin/env python3
"""compare_per_opening_parity.py

Compare ``training/solver/iterations.parquet`` files from three cobre runs
(rank-1, rank-2, rank-4) to verify that per-opening counter columns are
invariant to MPI rank count.

Exit codes:
  0 -- all three runs agree on every non-timing counter column
  1 -- at least one counter column differs between runs
  2 -- a prerequisite is missing (file not found, pyarrow unavailable)

Usage:
  python3 scripts/compare_per_opening_parity.py \\
      --dir1 target/parity_1 \\
      --dir2 target/parity_2 \\
      --dir4 target/parity_4 \\
      [--parquet-path training/solver/iterations.parquet]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ImportError:
    print(
        "ERROR: pyarrow is not available. Install it or use python3.14 which has pyarrow.",
        file=sys.stderr,
    )
    sys.exit(2)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

# Sort key used to align rows across all three runs before comparison.
SORT_KEYS: list[str] = ["iteration", "phase", "stage", "opening"]

# Columns that must be identical across rank counts.  Timing columns are
# intentionally excluded because wall-clock time is rank-count-dependent.
COMPARE_COLS: list[str] = [
    "opening",
    "lp_solves",
    "lp_successes",
    "lp_retries",
    "lp_failures",
    "retry_attempts",
    "basis_offered",
    "basis_consistency_failures",
    "simplex_iterations",
    "basis_reconstructions",
]

# Timing columns excluded from parity assertion.
TIMING_COLS: list[str] = [
    "solve_time_ms",
    "basis_set_time_ms",
    "load_model_time_ms",
    "add_rows_time_ms",
    "set_bounds_time_ms",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_sorted(path: Path) -> pa.Table:
    """Load a parquet file, aggregate per-(rank, worker_id) rows,
    and sort by SORT_KEYS.

    Post-T005, the parquet contains one row per ``(iteration, phase, stage,
    opening, rank, worker_id)`` tuple. To compare across rank counts we must
    first aggregate via ``SUM(...) GROUP BY (iteration, phase, stage, opening)``
    so the row shape matches the pre-T005 reference (one row per
    ``(iteration, phase, stage, opening)``).

    Pre-T005 parquets (without ``rank`` / ``worker_id`` columns) are returned
    as-is for backward compatibility.
    """
    if not path.exists():
        print(f"ERROR: parquet file not found: {path}", file=sys.stderr)
        sys.exit(2)
    table = pq.read_table(str(path))

    has_per_worker = "rank" in table.schema.names and "worker_id" in table.schema.names
    if has_per_worker:
        table = _aggregate_per_worker(table)

    # pyarrow sort_indices returns indices; apply them for a sorted table.
    sort_indices = pc.sort_indices(
        table,
        sort_keys=[(k, "ascending") for k in SORT_KEYS],
    )
    return table.take(sort_indices)


def _aggregate_per_worker(table: pa.Table) -> pa.Table:
    """SUM all numeric counter columns across (rank, worker_id) per group.

    Group key: ``SORT_KEYS`` (iteration, phase, stage, opening). Backward rows
    are first asserted to carry one row per ``(rank, worker_id)`` distinct pair
    per group (no duplicates from MPI allgatherv layout bugs); forward and
    lower_bound rows have ``worker_id IS NULL`` and pass through unchanged
    aside from the SUM aggregation.
    """
    # Backward rows carry per-worker_id; forward / lower_bound rows are NULL on
    # worker_id. Verify the (rank, worker_id) tuple is unique per group on the
    # backward subset — duplicates would indicate an unpack-order bug in T005.
    backward_mask = pc.equal(table.column("phase"), "backward")
    backward = table.filter(backward_mask)
    if backward.num_rows > 0:
        # Count rows per (group_key, rank, worker_id). Any count > 1 is a bug.
        dup_check = backward.group_by(
            [*SORT_KEYS, "rank", "worker_id"],
        ).aggregate([("phase", "count")])
        max_count = pc.max(dup_check.column("phase_count")).as_py()
        if max_count is not None and max_count > 1:
            print(
                f"ERROR: per-(rank, worker_id) row appears {max_count} times in some "
                "backward (iter, phase, stage, opening) group — allgatherv layout bug",
                file=sys.stderr,
            )
            sys.exit(1)

    # All numeric columns except the group keys, rank, and worker_id are summed.
    numeric_cols = [
        name
        for name in table.schema.names
        if name not in SORT_KEYS and name not in ("rank", "worker_id", "phase")
    ]
    aggregations = [(col, "sum") for col in numeric_cols]
    grouped = table.group_by(SORT_KEYS).aggregate(aggregations)
    # Rename the *_sum columns back to the original names so downstream
    # comparisons see the same schema as pre-T005 parquets.
    rename_map = {f"{c}_sum": c for c in numeric_cols}
    new_names = [rename_map.get(name, name) for name in grouped.schema.names]
    return grouped.rename_columns(new_names)


def _compare_tables(
    t_ref: pa.Table,
    t_other: pa.Table,
    ref_label: str,
    other_label: str,
) -> list[str]:
    """Compare two sorted tables on COMPARE_COLS.

    Returns a list of human-readable divergence descriptions (empty = match).
    """
    divergences: list[str] = []

    if t_ref.num_rows != t_other.num_rows:
        divergences.append(
            f"row count mismatch: {ref_label} has {t_ref.num_rows} rows, "
            f"{other_label} has {t_other.num_rows} rows"
        )
        # Cannot continue column-by-column if row counts differ.
        return divergences

    for col in COMPARE_COLS:
        col_ref = t_ref.column(col)
        col_other = t_other.column(col)
        # pc.equal handles nulls: null == null → True (via pc.equal with null handling)
        # We want null-safe equality: treat two nulls as equal.
        not_equal_mask = pc.invert(
            pc.or_(
                pc.and_(pc.is_null(col_ref), pc.is_null(col_other)),
                pc.equal(col_ref, col_other),
            )
        )
        ne_indices = pc.list_flatten(
            pa.chunked_array(
                [pa.array([pc.indices_nonzero(not_equal_mask).to_pylist()])]
            )
        ).to_pylist()

        if ne_indices:
            first_idx = ne_indices[0]
            ref_val = col_ref[first_idx].as_py()
            other_val = col_other[first_idx].as_py()
            # Build a human-readable key for the first differing row.
            row_key = {k: t_ref.column(k)[first_idx].as_py() for k in SORT_KEYS}
            divergences.append(
                f"column '{col}': first difference at row index {first_idx} "
                f"(key={row_key}): {ref_label}={ref_val!r}, {other_label}={other_val!r}; "
                f"total differing rows: {len(ne_indices)}"
            )

    return divergences


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare training/solver/iterations.parquet across three cobre "
            "runs with 1, 2, and 4 MPI ranks."
        ),
    )
    parser.add_argument(
        "--dir1",
        metavar="PATH",
        required=True,
        help="Output directory from the rank-1 run.",
    )
    parser.add_argument(
        "--dir2",
        metavar="PATH",
        required=True,
        help="Output directory from the rank-2 run.",
    )
    parser.add_argument(
        "--dir4",
        metavar="PATH",
        required=True,
        help="Output directory from the rank-4 run.",
    )
    parser.add_argument(
        "--parquet-path",
        metavar="REL_PATH",
        default="training/solver/iterations.parquet",
        help=(
            "Relative path to the parquet file within each output directory. "
            "Default: training/solver/iterations.parquet"
        ),
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    rel: str = args.parquet_path
    path1 = Path(args.dir1) / rel
    path2 = Path(args.dir2) / rel
    path4 = Path(args.dir4) / rel

    print(f"Loading and sorting: {path1}")
    t1 = _load_sorted(path1)
    print(f"Loading and sorting: {path2}")
    t2 = _load_sorted(path2)
    print(f"Loading and sorting: {path4}")
    t4 = _load_sorted(path4)

    print(
        f"\nRow counts: rank-1={t1.num_rows}, rank-2={t2.num_rows}, rank-4={t4.num_rows}"
    )

    all_divergences: list[str] = []

    print("\nComparing rank-1 vs rank-2 ...")
    d12 = _compare_tables(t1, t2, "rank-1", "rank-2")
    all_divergences.extend(d12)

    print("Comparing rank-1 vs rank-4 ...")
    d14 = _compare_tables(t1, t4, "rank-1", "rank-4")
    all_divergences.extend(d14)

    print("Comparing rank-2 vs rank-4 ...")
    d24 = _compare_tables(t2, t4, "rank-2", "rank-4")
    all_divergences.extend(d24)

    print()
    if all_divergences:
        print("FAIL: parity divergence detected:", file=sys.stderr)
        for i, msg in enumerate(all_divergences, start=1):
            print(f"  [{i}] {msg}", file=sys.stderr)
        sys.exit(1)

    print(
        f"PASS: all 3 runs agree on {len(COMPARE_COLS)} counter columns "
        f"({t1.num_rows} rows each, sorted by {SORT_KEYS})"
    )


if __name__ == "__main__":
    main()
