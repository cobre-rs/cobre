"""Tests for error propagation across the FFI boundary.

These tests verify that Rust errors are correctly translated into appropriate
Python exceptions with meaningful error messages.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_errors.py
"""

import pathlib
import tempfile

import pytest


MISSING_CASE = "/tmp/nonexistent_cobre_case_xzy123"


def test_run_nonexistent_dir_raises_oserror(tmp_path: pathlib.Path) -> None:
    """run() raises OSError with a descriptive message for a non-existent directory."""
    import cobre.run  # noqa: PLC0415

    with pytest.raises(OSError, match="does not exist"):
        cobre.run.run(MISSING_CASE, output_dir=str(tmp_path))


def test_run_empty_dir_raises_runtime_error(tmp_path: pathlib.Path) -> None:
    """run() raises RuntimeError for a directory missing required case files."""
    import cobre.run  # noqa: PLC0415

    empty_case = tmp_path / "empty_case"
    empty_case.mkdir()
    output = tmp_path / "output"

    with pytest.raises(RuntimeError, match="constraint violation"):
        cobre.run.run(str(empty_case), output_dir=str(output))


def test_run_empty_dir_error_mentions_missing_files(tmp_path: pathlib.Path) -> None:
    """The RuntimeError for an empty case lists specific missing files."""
    import cobre.run  # noqa: PLC0415

    empty_case = tmp_path / "empty_case"
    empty_case.mkdir()
    output = tmp_path / "output"

    with pytest.raises(RuntimeError) as exc_info:
        cobre.run.run(str(empty_case), output_dir=str(output))

    msg = str(exc_info.value)
    assert "config.json" in msg, "error must mention config.json"
    assert "FileNotFound" in msg, "error must include FileNotFound kind"


def test_load_case_nonexistent_raises_oserror() -> None:
    """load_case raises OSError for a non-existent path."""
    import cobre.io  # noqa: PLC0415

    with pytest.raises(OSError, match="does not exist"):
        cobre.io.load_case(MISSING_CASE)


def test_load_results_empty_dir_raises_file_not_found(tmp_path: pathlib.Path) -> None:
    """load_results raises FileNotFoundError for a directory without _SUCCESS."""
    import cobre.results  # noqa: PLC0415

    with pytest.raises(FileNotFoundError):
        cobre.results.load_results(str(tmp_path))
