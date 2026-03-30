"""Tests for output file existence and Parquet schema verification.

These tests verify that a completed run produces the expected directory
structure, and that Parquet files have correct schemas.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_outputs.py
"""

from __future__ import annotations

import json
import pathlib
import shutil

import pyarrow.parquet as pq
import pytest


VALID_CASE = "examples/1dtoy"
D20_CASE = "examples/deterministic/d20-operational-violations"


@pytest.fixture(scope="module")
def run_output(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Run 1dtoy once and return the output directory."""
    import cobre.run  # noqa: PLC0415

    output_dir = tmp_path_factory.mktemp("outputs_test")
    cobre.run.run(VALID_CASE, output_dir=str(output_dir))
    return output_dir


def test_training_output_files_exist(run_output: pathlib.Path) -> None:
    """A successful run produces all expected training output files."""
    # _SUCCESS is a zero-byte sentinel marker; the rest must be non-empty
    markers = {"training/_SUCCESS"}
    expected = [
        "training/_SUCCESS",
        "training/_manifest.json",
        "training/convergence.parquet",
        "training/metadata.json",
        "training/scaling_report.json",
        "training/solver/iterations.parquet",
        "training/timing/iterations.parquet",
    ]
    for rel in expected:
        path = run_output / rel
        assert path.exists(), f"missing training output: {rel}"
        if rel not in markers:
            assert path.stat().st_size > 0, f"empty training output: {rel}"


def test_simulation_output_files_exist(run_output: pathlib.Path) -> None:
    """A successful run produces simulation output directories."""
    expected_dirs = [
        "simulation/buses",
        "simulation/costs",
        "simulation/hydros",
        "simulation/thermals",
    ]
    for rel in expected_dirs:
        path = run_output / rel
        assert path.is_dir(), f"missing simulation output dir: {rel}"
        parquets = list(path.rglob("*.parquet"))
        assert len(parquets) > 0, f"no parquet files in {rel}"

    assert (run_output / "simulation/_SUCCESS").exists()
    assert (run_output / "simulation/_manifest.json").exists()


def test_convergence_parquet_schema(run_output: pathlib.Path) -> None:
    """convergence.parquet has the expected column names."""
    schema = pq.read_schema(run_output / "training" / "convergence.parquet")
    names = set(schema.names)
    required = {
        "iteration",
        "lower_bound",
        "upper_bound_mean",
        "upper_bound_std",
        "gap_percent",
        "cuts_added",
        "cuts_active",
        "time_forward_ms",
        "time_backward_ms",
        "time_total_ms",
    }
    missing = required - names
    assert not missing, f"convergence.parquet missing columns: {missing}"


def test_training_manifest_structure(run_output: pathlib.Path) -> None:
    """_manifest.json has expected top-level keys."""
    manifest = json.loads((run_output / "training" / "_manifest.json").read_text())
    assert isinstance(manifest, dict)
    assert "version" in manifest
    assert "status" in manifest
    assert "convergence" in manifest


def test_policy_output_exists(run_output: pathlib.Path) -> None:
    """A successful run produces policy cuts and metadata."""
    assert (run_output / "policy" / "metadata.json").exists()
    cuts = list((run_output / "policy" / "cuts").iterdir())
    assert len(cuts) > 0, "policy/cuts/ must contain stage files"


# ---------------------------------------------------------------------------
# D20 operational violation slack columns — Python parity verification
# ---------------------------------------------------------------------------

# The 4 operational violation slack columns that must appear in hydro output.
HYDRO_SLACK_COLUMNS = {
    "turbined_slack_m3s",
    "outflow_slack_below_m3s",
    "outflow_slack_above_m3s",
    "generation_slack_mw",
}


@pytest.fixture(scope="module")
def d20_output(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Run D20 with simulation enabled and return the output directory.

    D20 ships with ``simulation.enabled = false`` (the Rust deterministic
    test runs the simulation programmatically). For the Python parity test
    we need the full write path exercised, so we copy the case to a temp
    directory and flip the simulation flag on.
    """
    import cobre.run  # noqa: PLC0415

    src = pathlib.Path(D20_CASE)
    case_dir = tmp_path_factory.mktemp("d20_case")

    # Copy all case files except config.json (we'll overwrite it).
    for item in src.iterdir():
        dest = case_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    # Rewrite config with simulation enabled.
    config = json.loads((src / "config.json").read_text())
    config["simulation"]["enabled"] = True
    (case_dir / "config.json").write_text(json.dumps(config))

    output_dir = tmp_path_factory.mktemp("d20_output")
    cobre.run.run(str(case_dir), output_dir=str(output_dir))
    return output_dir


def test_hydro_parquet_has_slack_columns(d20_output: pathlib.Path) -> None:
    """Hydro simulation Parquet files include all 4 operational violation
    slack columns in their schema."""
    hydros_dir = d20_output / "simulation" / "hydros"
    assert hydros_dir.is_dir(), "simulation/hydros/ must exist after D20 run"

    parquets = list(hydros_dir.rglob("*.parquet"))
    assert len(parquets) > 0, "no parquet files in simulation/hydros/"

    schema = pq.read_schema(parquets[0])
    column_names = set(schema.names)
    missing = HYDRO_SLACK_COLUMNS - column_names
    assert not missing, f"hydro parquet missing slack columns: {missing}"


def test_hydro_slack_values_nonzero_on_violations(d20_output: pathlib.Path) -> None:
    """D20 forces operational violations. At least one scenario/stage must
    have non-zero ``outflow_slack_below_m3s`` and ``turbined_slack_m3s``."""
    import pyarrow as pa  # noqa: PLC0415

    hydros_dir = d20_output / "simulation" / "hydros"
    parquets = sorted(hydros_dir.rglob("*.parquet"))
    assert len(parquets) > 0

    # Concatenate all scenario files into a single table.
    tables = [pq.read_table(p) for p in parquets]
    table: pa.Table = pa.concat_tables(tables)

    outflow_below = table.column("outflow_slack_below_m3s").to_pylist()
    turbined_slack = table.column("turbined_slack_m3s").to_pylist()

    assert any(v > 1e-10 for v in outflow_below), (
        "D20: expected non-zero outflow_slack_below_m3s in at least one row. "
        "Stage 1 has inflow=10 m3/s but min_outflow=40 m3/s."
    )
    assert any(v > 1e-10 for v in turbined_slack), (
        "D20: expected non-zero turbined_slack_m3s in at least one row. "
        "Stage 1 has inflow=10 m3/s but min_turbined=30 m3/s."
    )
