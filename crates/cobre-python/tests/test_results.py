"""Integration tests for cobre.results — result loading and inspection.

These tests verify that, after a completed run, the result loading functions
return correctly-shaped Python objects.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_results.py

Note: tests that invoke run() write to a temporary directory created by
pytest's tmp_path fixture. The 1dtoy case is small enough that tests complete
in a few seconds.
"""

import pathlib

import pytest


VALID_CASE = "examples/1dtoy"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_output(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Run the 1dtoy case once and return the output directory.

    Module-scoped so the solver only runs once per test session.
    """
    import cobre.run  # noqa: PLC0415

    output_dir = tmp_path_factory.mktemp("results_output")
    cobre.run.run(VALID_CASE, output_dir=str(output_dir))
    return output_dir


# ---------------------------------------------------------------------------
# load_results tests
# ---------------------------------------------------------------------------


def test_load_results_after_run(run_output: pathlib.Path) -> None:
    """load_results() returns a dict with training.complete == True."""
    import cobre.results  # noqa: PLC0415

    result = cobre.results.load_results(str(run_output))

    assert isinstance(result, dict), "load_results must return a dict"
    assert "training" in result, "result must have 'training' key"
    assert result["training"]["complete"] is True, "training.complete must be True"


def test_load_results_manifest_keys(run_output: pathlib.Path) -> None:
    """result['training']['manifest'] contains required top-level keys."""
    import cobre.results  # noqa: PLC0415

    result = cobre.results.load_results(str(run_output))
    manifest = result["training"]["manifest"]

    assert isinstance(manifest, dict), "manifest must be a dict"
    assert "version" in manifest, "manifest must contain 'version'"
    assert "status" in manifest, "manifest must contain 'status'"
    assert "convergence" in manifest, "manifest must contain 'convergence'"


def test_load_results_metadata_present(run_output: pathlib.Path) -> None:
    """result['training']['metadata'] is a non-empty dict."""
    import cobre.results  # noqa: PLC0415

    result = cobre.results.load_results(str(run_output))
    metadata = result["training"]["metadata"]

    assert isinstance(metadata, dict), "metadata must be a dict"
    assert len(metadata) > 0, "metadata must not be empty"


def test_load_results_convergence_path_is_file(run_output: pathlib.Path) -> None:
    """result['training']['convergence_path'] points to an existing file."""
    import cobre.results  # noqa: PLC0415

    result = cobre.results.load_results(str(run_output))
    convergence_path = result["training"]["convergence_path"]

    assert isinstance(convergence_path, str), "convergence_path must be a str"
    assert pathlib.Path(convergence_path).is_file(), (
        f"convergence_path must point to an existing file: {convergence_path}"
    )


def test_load_results_timing_path_is_file(run_output: pathlib.Path) -> None:
    """result['training']['timing_path'] points to an existing file."""
    import cobre.results  # noqa: PLC0415

    result = cobre.results.load_results(str(run_output))
    timing_path = result["training"]["timing_path"]

    assert isinstance(timing_path, str), "timing_path must be a str"
    assert pathlib.Path(timing_path).is_file(), (
        f"timing_path must point to an existing file: {timing_path}"
    )


def test_load_results_simulation_section_present(run_output: pathlib.Path) -> None:
    """result['simulation'] is a dict with 'manifest' and 'complete' keys."""
    import cobre.results  # noqa: PLC0415

    result = cobre.results.load_results(str(run_output))

    assert "simulation" in result, "result must have 'simulation' key"
    sim = result["simulation"]
    assert isinstance(sim, dict), "simulation must be a dict"
    assert "manifest" in sim, "simulation must have 'manifest' key"
    assert "complete" in sim, "simulation must have 'complete' key"


def test_load_results_simulation_ran(
    run_output: pathlib.Path,
) -> None:
    """1dtoy has simulation.enabled=true, so simulation results should exist."""
    import cobre.results  # noqa: PLC0415

    result = cobre.results.load_results(str(run_output))
    sim = result["simulation"]
    assert sim["complete"] is True, "simulation must be complete after a successful run"
    assert isinstance(sim["manifest"], dict), "simulation manifest must be a dict"


def test_load_results_no_success_raises(tmp_path: pathlib.Path) -> None:
    """load_results() raises FileNotFoundError when training/_SUCCESS is absent."""
    import cobre.results  # noqa: PLC0415

    with pytest.raises(FileNotFoundError):
        cobre.results.load_results(str(tmp_path))


def test_load_results_nonexistent_dir_raises() -> None:
    """load_results() raises FileNotFoundError for a non-existent directory."""
    import cobre.results  # noqa: PLC0415

    with pytest.raises(FileNotFoundError):
        cobre.results.load_results("/tmp/nonexistent_cobre_output_xzy123")


# ---------------------------------------------------------------------------
# load_convergence tests
# ---------------------------------------------------------------------------


def test_load_convergence_returns_list(run_output: pathlib.Path) -> None:
    """load_convergence() returns a non-empty list of dicts."""
    import cobre.results  # noqa: PLC0415

    rows = cobre.results.load_convergence(str(run_output))

    assert isinstance(rows, list), "load_convergence must return a list"
    assert len(rows) > 0, "convergence list must be non-empty after a real run"


def test_load_convergence_dict_keys(run_output: pathlib.Path) -> None:
    """Each dict in the convergence list has the required keys."""
    import cobre.results  # noqa: PLC0415

    rows = cobre.results.load_convergence(str(run_output))
    required_keys = {
        "iteration",
        "lower_bound",
        "upper_bound_mean",
        "upper_bound_std",
        "gap_percent",
        "cuts_added",
        "cuts_removed",
        "cuts_active",
        "time_forward_ms",
        "time_backward_ms",
        "time_total_ms",
        "forward_passes",
        "lp_solves",
    }

    for i, row in enumerate(rows):
        assert isinstance(row, dict), f"row {i} must be a dict"
        missing = required_keys - row.keys()
        assert not missing, f"row {i} is missing keys: {missing}"


def test_load_convergence_value_types(run_output: pathlib.Path) -> None:
    """Convergence rows have correct Python types for key columns."""
    import cobre.results  # noqa: PLC0415

    rows = cobre.results.load_convergence(str(run_output))
    assert rows, "must have at least one row"

    row = rows[0]
    assert isinstance(row["iteration"], int), "iteration must be int"
    assert isinstance(row["lower_bound"], float), "lower_bound must be float"
    assert isinstance(row["upper_bound_mean"], float), "upper_bound_mean must be float"
    assert isinstance(row["upper_bound_std"], float), "upper_bound_std must be float"
    # gap_percent may be None or float
    assert row["gap_percent"] is None or isinstance(row["gap_percent"], float), (
        "gap_percent must be float or None"
    )
    assert isinstance(row["cuts_added"], int), "cuts_added must be int"
    assert isinstance(row["cuts_active"], int), "cuts_active must be int"
    assert isinstance(row["time_total_ms"], int), "time_total_ms must be int"


def test_load_convergence_iteration_is_one_based(run_output: pathlib.Path) -> None:
    """The first iteration row has iteration == 1."""
    import cobre.results  # noqa: PLC0415

    rows = cobre.results.load_convergence(str(run_output))
    assert rows, "must have at least one row"
    assert rows[0]["iteration"] == 1, "first iteration must be 1-based"


def test_load_convergence_empty_dir_raises(tmp_path: pathlib.Path) -> None:
    """load_convergence() raises FileNotFoundError for a directory without Parquet."""
    import cobre.results  # noqa: PLC0415

    with pytest.raises(FileNotFoundError):
        cobre.results.load_convergence(str(tmp_path))


def test_convergence_path_is_readable(run_output: pathlib.Path) -> None:
    """The convergence_path from load_results() is a valid, non-empty Parquet path."""
    import cobre.results  # noqa: PLC0415

    result = cobre.results.load_results(str(run_output))
    path = pathlib.Path(result["training"]["convergence_path"])

    assert path.exists(), "convergence_path must exist"
    assert path.stat().st_size > 0, "convergence.parquet must not be empty"
