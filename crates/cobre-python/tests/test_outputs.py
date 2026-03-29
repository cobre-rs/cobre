"""Tests for output file existence and Parquet schema verification.

These tests verify that a completed run produces the expected directory
structure, and that Parquet files have correct schemas.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_outputs.py
"""

import json
import pathlib

import pyarrow.parquet as pq
import pytest


VALID_CASE = "examples/1dtoy"


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
