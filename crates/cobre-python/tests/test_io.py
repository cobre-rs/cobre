"""Integration tests for cobre.io Python wrapper functions.

These tests verify that load_case and validate are correctly exposed and
behave as expected for both valid and invalid case directories.

Run with (from the repo root):
    pytest crates/cobre-python/tests/
"""

import pathlib

import pytest


VALID_CASE = "examples/1dtoy"
MISSING_CASE = "/tmp/nonexistent_cobre_case_xzy123"


def test_load_case_returns_system() -> None:
    """load_case returns a System with correct entity counts for the 1dtoy case."""
    import cobre.io  # noqa: PLC0415
    import cobre.model  # noqa: PLC0415

    system = cobre.io.load_case(VALID_CASE)
    assert isinstance(system, cobre.model.System)
    assert isinstance(system.n_buses, int)
    assert system.n_buses > 0


def test_load_case_nonexistent_raises_oserror() -> None:
    """load_case raises OSError when the case directory does not exist."""
    import cobre.io  # noqa: PLC0415

    with pytest.raises(OSError):
        cobre.io.load_case(MISSING_CASE)


def test_load_case_accepts_pathlib() -> None:
    """load_case accepts a pathlib.Path argument, not just a str."""
    import cobre.io  # noqa: PLC0415

    system = cobre.io.load_case(pathlib.Path(VALID_CASE))
    assert system.n_buses > 0


def test_validate_valid_case() -> None:
    """validate returns valid=True with no errors for the 1dtoy case."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(VALID_CASE)
    assert isinstance(result, dict)
    assert result["valid"] is True
    assert result["errors"] == []
    assert "warnings" in result


def test_validate_nonexistent_case() -> None:
    """validate returns valid=False with at least one error for a missing path."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(MISSING_CASE)
    assert isinstance(result, dict)
    assert result["valid"] is False
    assert len(result["errors"]) > 0
    error = result["errors"][0]
    assert "kind" in error
    assert "message" in error
