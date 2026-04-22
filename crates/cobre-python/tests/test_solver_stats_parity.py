"""Python parity tests for `training/solver/iterations.parquet` (ticket-009).

Verifies that `cobre.run.run()` produces an identical per-opening solver-stats
Parquet file to the `cobre` CLI for the D01 deterministic case.

## What "parity" means here

The Parquet file contains timing columns (`solve_time_ms`, `load_model_time_ms`,
`add_rows_time_ms`, `set_bounds_time_ms`, `basis_set_time_ms`) that record
wall-clock durations.  These values differ between any two independent runs
even when the logical outputs are identical.  Therefore:

- **Byte-for-byte SHA256 identity is NOT guaranteed** and is not tested here.
  The solver-stats parquet cannot be bit-for-bit reproducible across runs
  because it contains real-time measurements.
- **Schema identity IS guaranteed**: same column names, same dtypes, same
  column order.
- **Non-timing column identity IS guaranteed**: all integer and string columns
  (`iteration`, `phase`, `stage`, `opening`, `lp_solves`, etc.) must be
  identical for the same deterministic case.

This is the correct definition of "Python parity" for outputs that record
wall-clock timing information.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_solver_stats_parity.py -v
"""

from __future__ import annotations

import pathlib
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


D01_CASE = "examples/deterministic/d01-thermal-dispatch"

# Columns that record wall-clock time and will legitimately differ between runs.
TIMING_COLS = frozenset(
    {
        "solve_time_ms",
        "load_model_time_ms",
        "set_bounds_time_ms",
        "basis_set_time_ms",
    }
)

# Expected schema for training/solver/iterations.parquet.
# rank and worker_id are present; add_rows_time_ms was removed;
# 4 basis_preserved/new_tight/new_slack/demotions columns were consolidated
# to a single basis_reconstructions column.
EXPECTED_COLUMNS = [
    "iteration",
    "phase",
    "stage",
    "opening",
    "rank",
    "worker_id",
    "lp_solves",
    "lp_successes",
    "lp_retries",
    "lp_failures",
    "retry_attempts",
    "basis_offered",
    "basis_consistency_failures",
    "simplex_iterations",
    "solve_time_ms",
    "load_model_time_ms",
    "set_bounds_time_ms",
    "basis_set_time_ms",
    "basis_reconstructions",
]


def _cli_binary() -> pathlib.Path:
    """Return the path to the compiled `cobre` CLI binary.

    Looks for a release build first, then a debug build.  Skips the test
    with a clear message if neither is found.
    """
    repo_root = pathlib.Path(__file__).parents[3]
    for profile in ("release", "debug"):
        candidate = repo_root / "target" / profile / "cobre"
        if candidate.is_file():
            return candidate
    pytest.skip(
        "No compiled `cobre` binary found in target/release or target/debug. "
        "Run `cargo build -p cobre-cli --release` first."
    )
    raise RuntimeError("unreachable: pytest.skip raises Skipped")


def _run_cli(case_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    """Run the cobre CLI for `case_dir` and write outputs to `output_dir`."""
    binary = _cli_binary()
    result = subprocess.run(
        [str(binary), "run", str(case_dir), "--output", str(output_dir)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            f"cobre CLI failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


@pytest.fixture(scope="module")
def d01_cli_output(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Run D01 via the CLI and return the output directory."""
    output_dir = tmp_path_factory.mktemp("d01_cli")
    case_dir = pathlib.Path(D01_CASE)
    _run_cli(case_dir, output_dir)
    return output_dir


@pytest.fixture(scope="module")
def d01_python_output(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Run D01 via `cobre.run.run()` and return the output directory."""
    cobre_run = pytest.importorskip("cobre.run")

    output_dir = tmp_path_factory.mktemp("d01_python")
    cobre_run.run(D01_CASE, output_dir=str(output_dir))
    return output_dir


# ---------------------------------------------------------------------------
# test_python_writes_opening_column
# ---------------------------------------------------------------------------


def test_python_writes_opening_column(d01_python_output: pathlib.Path) -> None:
    """Python run produces iterations.parquet with the `opening` column (Int32, nullable)."""
    parquet_path = d01_python_output / "training" / "solver" / "iterations.parquet"
    assert parquet_path.exists(), (
        "training/solver/iterations.parquet must exist after Python run"
    )

    schema = pq.read_schema(parquet_path)
    assert "opening" in schema.names, (
        f"iterations.parquet must contain 'opening' column; got: {schema.names}"
    )

    # opening must be nullable Int32.
    field = schema.field("opening")
    assert pa.types.is_int32(field.type), (
        f"opening column must be Int32, got {field.type}"
    )
    assert field.nullable, (
        "opening column must be nullable (NULL for non-backward phases)"
    )


def test_python_opening_column_values(d01_python_output: pathlib.Path) -> None:
    """Backward rows have opening=0, forward and lower_bound rows have opening=NULL."""
    table = pq.read_table(
        d01_python_output / "training" / "solver" / "iterations.parquet"
    )

    phases = table.column("phase").to_pylist()
    openings = table.column("opening").to_pylist()

    for phase, opening in zip(phases, openings):
        if phase == "backward":
            assert opening is not None, (
                f"backward row must have non-NULL opening; got None for phase={phase}"
            )
        else:
            assert opening is None, (
                f"non-backward phase '{phase}' must have NULL opening; got {opening}"
            )


# ---------------------------------------------------------------------------
# test_python_matches_cli_sha256 (logical parity — not byte-identical)
# ---------------------------------------------------------------------------


def test_python_matches_cli_schema(
    d01_cli_output: pathlib.Path, d01_python_output: pathlib.Path
) -> None:
    """CLI and Python runs produce iterations.parquet with identical schemas.

    Schema = column names, dtypes, and nullability.
    """
    cli_schema = pq.read_schema(
        d01_cli_output / "training" / "solver" / "iterations.parquet"
    )
    py_schema = pq.read_schema(
        d01_python_output / "training" / "solver" / "iterations.parquet"
    )
    assert cli_schema.equals(py_schema), (
        f"Schema mismatch between CLI and Python:\n"
        f"  CLI: {cli_schema}\n"
        f"  PY:  {py_schema}"
    )


def test_python_matches_cli_row_count(
    d01_cli_output: pathlib.Path, d01_python_output: pathlib.Path
) -> None:
    """CLI and Python runs produce the same number of rows."""
    cli_table = pq.read_table(
        d01_cli_output / "training" / "solver" / "iterations.parquet"
    )
    py_table = pq.read_table(
        d01_python_output / "training" / "solver" / "iterations.parquet"
    )
    assert cli_table.num_rows == py_table.num_rows, (
        f"Row count mismatch: CLI={cli_table.num_rows}, Python={py_table.num_rows}"
    )


def test_python_matches_cli_nontiming_columns(
    d01_cli_output: pathlib.Path, d01_python_output: pathlib.Path
) -> None:
    """All non-timing columns are identical between CLI and Python runs.

    Timing columns (solve_time_ms, load_model_time_ms, add_rows_time_ms,
    set_bounds_time_ms, basis_set_time_ms) record wall-clock durations and
    legitimately differ between runs.  All other columns must be identical
    for the same deterministic case.

    Note: byte-for-byte SHA256 identity is NOT expected because timing columns
    cause the raw Parquet bytes to differ.  This test checks logical parity.
    """
    cli_table = pq.read_table(
        d01_cli_output / "training" / "solver" / "iterations.parquet"
    )
    py_table = pq.read_table(
        d01_python_output / "training" / "solver" / "iterations.parquet"
    )

    mismatches: list[str] = []
    for col_name in cli_table.schema.names:
        if col_name in TIMING_COLS:
            continue
        cli_col = cli_table.column(col_name)
        py_col = py_table.column(col_name)
        if not cli_col.equals(py_col):
            mismatches.append(
                f"  column '{col_name}':\n"
                f"    CLI first 5: {cli_col[:5].to_pylist()}\n"
                f"    PY  first 5: {py_col[:5].to_pylist()}"
            )

    assert not mismatches, (
        "Non-timing column mismatch between CLI and Python:\n" + "\n".join(mismatches)
    )


def test_python_schema_matches_expected_columns(
    d01_python_output: pathlib.Path,
) -> None:
    """Python output has exactly the expected 19-column schema."""
    schema = pq.read_schema(
        d01_python_output / "training" / "solver" / "iterations.parquet"
    )
    assert schema.names == EXPECTED_COLUMNS, (
        f"Column names mismatch:\n"
        f"  expected: {EXPECTED_COLUMNS}\n"
        f"  got:      {schema.names}"
    )


def test_python_basis_reconstructions_column_shape(
    d01_python_output: pathlib.Path,
) -> None:
    """basis_reconstructions column is UInt64 non-nullable."""
    parquet_path = d01_python_output / "training" / "solver" / "iterations.parquet"
    schema = pq.read_schema(parquet_path)
    field = schema.field("basis_reconstructions")
    assert pa.types.is_uint64(field.type), (
        f"basis_reconstructions must be UInt64, got {field.type}"
    )
    assert not field.nullable, "basis_reconstructions must be non-nullable"

    # Verify the counter is non-zero somewhere in the forward phase: warm-start
    # solves after iteration 1 must invoke reconstruct_basis, so at least one row
    # should have basis_reconstructions > 0. (`>= 0` would be vacuously true for
    # UInt64.)
    table = pq.read_table(parquet_path)
    phases = table.column("phase").to_pylist()
    values = table.column("basis_reconstructions").to_pylist()
    forward_max = max(
        (v for p, v in zip(phases, values) if p == "forward"),
        default=0,
    )
    assert forward_max > 0, (
        f"forward phase must invoke basis reconstruction at least once after iteration 1, "
        f"got max basis_reconstructions={forward_max}"
    )
