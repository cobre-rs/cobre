"""Linux CLI/Python parity for the shared NUMA-aware worker mapping."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest


D01_CASE = "examples/deterministic/d01-thermal-dispatch"


def _cli_binary() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).parents[3]
    for profile in ("release", "debug"):
        candidate = repo_root / "target" / profile / "cobre"
        if candidate.is_file():
            return candidate
    pytest.skip("No compiled `cobre` CLI binary found")
    raise RuntimeError("unreachable: pytest.skip raises Skipped")


def _policy_files(root: pathlib.Path) -> dict[pathlib.Path, bytes]:
    policy = root / "policy"
    return {
        path.relative_to(policy): path.read_bytes()
        for path in sorted(policy.rglob("*"))
        if path.is_file()
    }


@pytest.mark.skipif(sys.platform != "linux", reason="native affinity is Linux-only")
def test_cli_python_core_affinity_and_policy_parity(tmp_path: pathlib.Path) -> None:
    """Both front ends resolve the same mapping and numerical policy."""
    cobre_run = pytest.importorskip("cobre.run")
    repo_root = pathlib.Path(__file__).parents[3]
    case_dir = repo_root / D01_CASE
    cli_out = tmp_path / "cli"
    py_out = tmp_path / "python"

    result = subprocess.run(
        [
            str(_cli_binary()),
            "run",
            str(case_dir),
            "--output",
            str(cli_out),
            "--threads",
            "2",
            "--cpu-bind",
            "core",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(f"cobre CLI failed: {result.stderr}")

    cobre_run.run(
        str(case_dir),
        output_dir=str(py_out),
        threads=2,
        cpu_bind="core",
        skip_simulation=True,
    )

    cli_metadata = json.loads((cli_out / "training/metadata.json").read_text())
    py_metadata = json.loads((py_out / "training/metadata.json").read_text())
    cli_affinity = cli_metadata["distribution"]["rank_affinity"]
    py_affinity = py_metadata["distribution"]["rank_affinity"]
    assert cli_affinity == py_affinity
    assert _policy_files(cli_out) == _policy_files(py_out)
