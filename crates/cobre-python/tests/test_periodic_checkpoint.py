"""Periodic training checkpoints written by the CLI and the Python bindings.

A run killed during training must leave the checkpoint of its last scheduled
write, loadable by `cobre.results.load_policy`, and both front ends must write
the same files when periodic checkpointing is on.

Run with (from the repo root, after `cargo build --release -p cobre-cli`):
    pytest crates/cobre-python/tests/test_periodic_checkpoint.py --require-cli-binary
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import time

import pytest

from _cobre_cli import run_cli

_REPO_ROOT = pathlib.Path(__file__).parents[3]
_TOY_CASE = _REPO_ROOT / "examples" / "1dtoy"

_UNREACHABLE_ITERATION_LIMIT = 100_000
_CHECKPOINT_WAIT_SECONDS = 120.0

_PYTHON_RUN = "import sys, cobre.run; cobre.run.run(sys.argv[1], output_dir=sys.argv[2])"


def _periodic_case(root: pathlib.Path, iteration_limit: int) -> pathlib.Path:
    """Copy 1dtoy under `root` with a checkpoint every iteration and no simulation."""
    case = root / "case"
    shutil.copytree(_TOY_CASE, case, ignore=shutil.ignore_patterns("output"))
    config_path = case / "config.json"
    config = json.loads(config_path.read_text())
    config["training"]["stopping_rules"] = [
        {"type": "iteration_limit", "limit": iteration_limit}
    ]
    config["simulation"]["enabled"] = False
    config["policy"] = {"checkpointing": {"enabled": True, "interval_iterations": 1}}
    config_path.write_text(json.dumps(config, indent=2))
    return case


def _kill_after_first_checkpoint(
    command: list[str], output_dir: pathlib.Path, stderr_path: pathlib.Path
) -> None:
    """Start `command`, wait for `<output_dir>/policy/manifest.bin`, then SIGKILL it."""
    manifest = output_dir / "policy" / "manifest.bin"
    with stderr_path.open("w") as stderr:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr)
        try:
            deadline = time.monotonic() + _CHECKPOINT_WAIT_SECONDS
            while not manifest.is_file():
                if process.poll() is not None:
                    pytest.fail(
                        f"the run exited with {process.returncode} before writing "
                        f"{manifest}:\n{stderr_path.read_text()}"
                    )
                if time.monotonic() > deadline:
                    pytest.fail(
                        f"no checkpoint at {manifest} within "
                        f"{_CHECKPOINT_WAIT_SECONDS:.0f} s:\n{stderr_path.read_text()}"
                    )
                time.sleep(0.05)
        finally:
            process.kill()
            process.wait(timeout=30)


def _assert_loadable_periodic_checkpoint(output_dir: pathlib.Path) -> None:
    import cobre.results  # noqa: PLC0415

    producer = cobre.results.load_policy(output_dir)["metadata"]["producer"]
    assert 1 <= producer["completed_iterations"] < _UNREACHABLE_ITERATION_LIMIT, producer
    assert producer["max_iterations"] == _UNREACHABLE_ITERATION_LIMIT, producer


def _relative_files(root: pathlib.Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_killed_python_run_leaves_a_loadable_periodic_checkpoint(
    tmp_path: pathlib.Path,
) -> None:
    case = _periodic_case(tmp_path, _UNREACHABLE_ITERATION_LIMIT)
    output_dir = tmp_path / "out"
    _kill_after_first_checkpoint(
        [sys.executable, "-c", _PYTHON_RUN, str(case), str(output_dir)],
        output_dir,
        tmp_path / "stderr.txt",
    )
    _assert_loadable_periodic_checkpoint(output_dir)


def test_killed_cli_run_leaves_a_loadable_periodic_checkpoint(
    tmp_path: pathlib.Path, cli_binary: pathlib.Path
) -> None:
    case = _periodic_case(tmp_path, _UNREACHABLE_ITERATION_LIMIT)
    output_dir = tmp_path / "out"
    _kill_after_first_checkpoint(
        [str(cli_binary), "run", str(case), "--output", str(output_dir), "--quiet"],
        output_dir,
        tmp_path / "stderr.txt",
    )
    _assert_loadable_periodic_checkpoint(output_dir)


def test_periodic_checkpoint_runs_write_the_same_file_set_on_cli_and_python(
    tmp_path: pathlib.Path, cli_binary: pathlib.Path
) -> None:
    import cobre.run  # noqa: PLC0415

    case = _periodic_case(tmp_path, iteration_limit=4)
    cli_output = tmp_path / "cli_out"
    python_output = tmp_path / "python_out"
    run_cli(case, cli_output, cli_binary)
    cobre.run.run(str(case), output_dir=str(python_output))

    cli_files = _relative_files(cli_output)
    python_files = _relative_files(python_output)
    assert cli_files == python_files, (
        f"only CLI: {sorted(cli_files - python_files)}; "
        f"only Python: {sorted(python_files - cli_files)}"
    )
    for output_dir in (cli_output, python_output):
        for sibling in ("policy.staging", "policy.previous"):
            assert not (output_dir / sibling).exists(), output_dir / sibling
