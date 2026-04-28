"""CLI/Python byte-identity parity test for ``stochastic/inflow_annual_component.parquet``.

Verifies that ``cobre.run.run()`` and the ``cobre`` CLI produce byte-identical
``inflow_annual_component.parquet`` files for the same deterministic input.

## What "parity" means here

``inflow_annual_component.parquet`` contains only fitted PAR-A model statistics
(annual coefficient, annual mean, and annual standard deviation per hydro per
stage).  It carries no wall-clock timing data.  Both the CLI and Python paths
write the file through the same Rust writer (``write_inflow_annual_component``
in ``cobre-io``) using ``ParquetWriterConfig::default()`` and
``write_parquet_atomic``, which is byte-deterministic by construction.

Byte equality is therefore the correct and tightest correctness gate:

- **Byte-for-byte identity IS expected and tested here** because the file
  contains no runtime-dependent data.
- Any divergence (different data, encoding, or library version embedded in
  Parquet metadata) surfaces as a test failure with a clear diagnostic.

Run with (from the repo root)::

    pytest crates/cobre-python/tests/test_inflow_annual_component_parity.py -v
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

D12_CASE = "examples/deterministic/d12-par-annual"


def _cli_binary() -> pathlib.Path:
    """Return the path to the compiled ``cobre`` CLI binary.

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
    """Run the cobre CLI for ``case_dir`` and write outputs to ``output_dir``."""
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


def test_inflow_annual_component_byte_identical(tmp_path: pathlib.Path) -> None:
    """CLI and Python produce byte-identical ``inflow_annual_component.parquet``.

    Both the CLI and Python paths write through the same Rust writer with
    deterministic Parquet encoding.  Any byte-level divergence indicates a
    regression in the parity guarantee.
    """
    cobre_run = pytest.importorskip(
        "cobre.run",
        reason=(
            "`cobre` Python module is not installed. "
            "Run `maturin develop --uv -m crates/cobre-python/Cargo.toml --release` first."
        ),
    )

    repo_root = pathlib.Path(__file__).parents[3]
    case_dir = repo_root / D12_CASE

    cli_out = tmp_path / "cli_out"
    py_out = tmp_path / "py_out"
    cli_out.mkdir()
    py_out.mkdir()

    # Run CLI.
    _run_cli(case_dir, cli_out)

    # Run Python.
    cobre_run.run(str(case_dir), output_dir=str(py_out))

    cli_parquet = cli_out / "stochastic" / "inflow_annual_component.parquet"
    py_parquet = py_out / "stochastic" / "inflow_annual_component.parquet"

    cli_bytes = cli_parquet.read_bytes()
    py_bytes = py_parquet.read_bytes()

    if cli_bytes != py_bytes:
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(cli_bytes, py_bytes)) if a != b),
            -1,
        )
        pytest.fail(
            f"inflow_annual_component.parquet is not byte-identical between CLI and Python.\n"
            f"  CLI path:  {cli_parquet}  ({len(cli_bytes)} bytes)\n"
            f"  Python path: {py_parquet}  ({len(py_bytes)} bytes)\n"
            f"  First differing byte offset: {first_diff}"
        )
