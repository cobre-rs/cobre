"""Integration tests for cobre.run.run() Python wrapper.

These tests verify that the full solve lifecycle (load -> train -> simulate ->
write) can be invoked from Python, with the GIL released during computation.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_run.py

Note: each test that invokes run() writes to a temporary directory created by
pytest's tmp_path fixture. The 1dtoy case is small enough that tests complete
in a few seconds.
"""

import pathlib

import pytest


VALID_CASE = "examples/1dtoy"
D56_EXTERNAL_AUTHORITATIVE_CASE = "examples/deterministic/d56-external-authoritative"
MISSING_CASE = "/tmp/nonexistent_cobre_case_xzy123"


def test_run_1dtoy_succeeds(tmp_path: pathlib.Path) -> None:
    """run() returns a dict with converged, iterations, and lower_bound keys."""
    import cobre.run  # noqa: PLC0415

    result = cobre.run.run(VALID_CASE, output_dir=str(tmp_path))

    assert isinstance(result, dict), "run() must return a dict"
    assert isinstance(result["converged"], bool), "converged must be bool"
    assert isinstance(result["iterations"], int), "iterations must be int"
    assert result["iterations"] > 0, "iterations must be > 0"
    assert isinstance(result["lower_bound"], float), "lower_bound must be float"
    assert isinstance(result["output_dir"], str), "output_dir must be str"


def test_run_1dtoy_creates_output(tmp_path: pathlib.Path) -> None:
    """After run(), the output directory contains training/_SUCCESS."""
    import cobre.run  # noqa: PLC0415

    cobre.run.run(VALID_CASE, output_dir=str(tmp_path))

    success_marker = tmp_path / "training" / "_SUCCESS"
    assert success_marker.exists(), "training/_SUCCESS must exist after run()"

    convergence = tmp_path / "training" / "convergence.parquet"
    assert convergence.exists(), "training/convergence.parquet must exist after run()"


def test_run_d56_external_authoritative_converges(tmp_path: pathlib.Path) -> None:
    """run() accepts a sigma=0 External load/inflow deck with no seasonal-stats
    twins and reaches numeric convergence (lower_bound == upper_bound), mirroring
    the CLI's behavior on the same deck and the deck-level Rust regression
    (`d56_external_authoritative_loads_and_converges`).

    d56's config declares only an `iteration_limit` stopping rule, so the
    returned `converged` flag stays False even at a 0% gap (no `gap` or
    `bound_stalling` rule can trigger) — this asserts the numeric
    convergence the deck actually demonstrates instead.
    """
    import cobre.run  # noqa: PLC0415

    result = cobre.run.run(D56_EXTERNAL_AUTHORITATIVE_CASE, output_dir=str(tmp_path))

    assert result["iterations"] >= 1, "d56 training must run at least one iteration"

    lower_bound = result["lower_bound"]
    upper_bound = result["upper_bound"]
    assert lower_bound is not None and upper_bound is not None, (
        "d56 training must report both bounds"
    )
    assert abs(lower_bound - upper_bound) < 1e-6, (
        "d56 is a sigma=0 deck: lower_bound and upper_bound must match to "
        f"within 1e-6, got lower_bound={lower_bound} upper_bound={upper_bound}"
    )

    success_marker = tmp_path / "training" / "_SUCCESS"
    assert success_marker.exists(), "training/_SUCCESS must exist after run()"


def test_run_simulation_disabled_by_config(tmp_path: pathlib.Path) -> None:
    """run() with config_overrides disabling simulation returns result['simulation'] as None."""
    import cobre.run  # noqa: PLC0415

    result = cobre.run.run(
        VALID_CASE,
        output_dir=str(tmp_path),
        config_overrides={"simulation": {"enabled": False}},
    )

    assert result["simulation"] is None, (
        "simulation must be None when config.simulation.enabled is False"
    )


def test_run_rejects_skip_simulation_parameter(tmp_path: pathlib.Path) -> None:
    """run() raises TypeError when the removed skip_simulation parameter is passed."""
    import cobre.run  # noqa: PLC0415

    with pytest.raises(TypeError, match="unexpected keyword argument") as exc_info:
        cobre.run.run(VALID_CASE, output_dir=str(tmp_path), skip_simulation=True)

    assert "skip_simulation" in str(exc_info.value), (
        "TypeError message must mention 'skip_simulation'"
    )


def test_run_nonexistent_raises(tmp_path: pathlib.Path) -> None:
    """run() raises OSError when the case directory does not exist."""
    import cobre.run  # noqa: PLC0415

    with pytest.raises(OSError):
        cobre.run.run(MISSING_CASE, output_dir=str(tmp_path))


def test_run_threads_parameter(tmp_path: pathlib.Path) -> None:
    """run() with threads=2 succeeds — verifies rayon thread pool integration."""
    import cobre.run  # noqa: PLC0415

    result = cobre.run.run(VALID_CASE, output_dir=str(tmp_path), threads=2)

    assert isinstance(result["converged"], bool), "converged must be bool"
    assert result["iterations"] > 0, "iterations must be > 0"


def test_run_rejects_threads_zero(tmp_path: pathlib.Path) -> None:
    """run() raises ValueError when threads=0."""
    import cobre.run  # noqa: PLC0415

    with pytest.raises(ValueError, match="threads"):
        cobre.run.run(VALID_CASE, output_dir=str(tmp_path), threads=0)


def _read_metadata_seed(output_dir: pathlib.Path) -> object:
    """Read configuration.seed from training/metadata.json."""
    import json  # noqa: PLC0415

    meta_path = output_dir / "training" / "metadata.json"
    with meta_path.open() as f:
        meta = json.load(f)
    return meta["configuration"]["seed"]


def test_run_config_overrides_none_matches_default(tmp_path: pathlib.Path) -> None:
    """config_overrides=None reproduces today's behavior exactly (no-op)."""
    import cobre.run  # noqa: PLC0415

    result = cobre.run.run(VALID_CASE, output_dir=str(tmp_path), config_overrides=None)

    assert result["iterations"] > 0, "iterations must be > 0 with no overrides"


def test_run_config_overrides_seed_changes_metadata(tmp_path: pathlib.Path) -> None:
    """An override of training.tree_seed is persisted to metadata as the seed."""
    import cobre.run  # noqa: PLC0415

    cobre.run.run(
        VALID_CASE,
        output_dir=str(tmp_path),
        config_overrides={"training.tree_seed": 7},
    )

    seed = _read_metadata_seed(tmp_path)
    assert seed == 7, f"effective config seed must be 7, got {seed!r}"


def test_run_config_overrides_typo_raises_value_error(tmp_path: pathlib.Path) -> None:
    """A typo override key is rejected via deny_unknown_fields → ValueError."""
    import cobre.run  # noqa: PLC0415

    with pytest.raises(ValueError):
        cobre.run.run(
            VALID_CASE,
            output_dir=str(tmp_path),
            config_overrides={"trainning.tree_seed": 7},
        )


def test_run_config_overrides_unsupported_value_raises_value_error(
    tmp_path: pathlib.Path,
) -> None:
    """A value with no JSON representation is rejected before the merge."""
    import cobre.run  # noqa: PLC0415

    with pytest.raises(ValueError):
        cobre.run.run(
            VALID_CASE,
            output_dir=str(tmp_path),
            config_overrides={"training.tree_seed": {1, 2, 3}},
        )


def test_run_dict_keys_stable(tmp_path: pathlib.Path) -> None:
    """run() returns exactly the public 11-key dict with the full simulation block.

    Guards the single-execution-path reimplementation: the public surface of
    cobre.run.run (its return-dict key set and the simulation sub-dict) must be
    byte-for-byte the same as before the collapse onto the Study lifecycle.
    """
    import cobre.run  # noqa: PLC0415

    result = cobre.run.run(VALID_CASE, output_dir=str(tmp_path))

    expected_keys = {
        "converged",
        "iterations",
        "lower_bound",
        "upper_bound",
        "gap_percent",
        "total_time_ms",
        "output_dir",
        "simulation",
        "stochastic",
        "hydro_models",
        "provenance",
    }
    assert set(result.keys()) == expected_keys, (
        f"run() dict key set must be exactly {expected_keys}, got {set(result.keys())}"
    )

    assert result["simulation"] == {"n_scenarios": 100, "completed": 100}, (
        "simulation block must report all 100 scenarios completed"
    )
