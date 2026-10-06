"""CLI-vs-Python output file-set equality gate.

Every output file the CLI write path produces must also be produced by the
Python bindings write path -- the Python-parity hard rule stated in this
project's `CLAUDE.md`: "every output file the CLI writes must also be written
by the Python bindings." The existing `test_*_parity.py` / `test_outputs.py`
suite checks per-file byte/column parity one file at a time; none of them
asserts that neither write path gains or loses a file the other does not.
This module closes that gap: it runs both write paths end-to-end on each case
and asserts the two output trees contain the exact same SET of relative file
paths. The equality gate compares file SETS ONLY -- never bytes, never
columns, since content parity is already covered by the `test_*_parity.py`
suite and duplicating it here would be dead coverage.

`cobre-python` is excluded from the cargo workspace (it needs a Python
interpreter to build), so `cargo test --workspace` never runs this gate; it
runs in cobre-python's own test job, e.g.:

    maturin develop --release --manifest-path crates/cobre-python/Cargo.toml
    pytest crates/cobre-python/tests/test_cli_python_file_set_parity.py -v

The fixture case (`examples/deterministic/d28-decomp-weekly-monthly`) ships
training and simulation both enabled with 5 simulated scenarios, so a single
run exercises the surface most prone to drift between the two hand-maintained
write paths: `scenario_summary.parquet`, the reshaped `paths.parquet`, the
per-entity simulation Parquet files (buses/costs/hydros/thermals/
hydro_bus_generation), and the Hive-partitioned `scenario_id=NNNN/` layout.
The recursive walk below reads that layout by listing the directory tree
directly (`Path.rglob`), never through a pyarrow hive-partition dataset
inference: `scenario_id` is both the Hive partition key AND an in-file Int32
column, so `pyarrow.dataset`'s schema-merging would raise on the collision. A
plain directory walk has no such ambiguity.

The second case (`examples/deterministic/d57-fpha-zero-turbine-capacity`) has
one plant that requests FPHA but has no turbine capacity. It is here to check
that both write paths list that plant under the `no_turbine_capacity` key of
`training/hydro_models.json`.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil

import pytest

from _cobre_cli import resolve_cli_binary, run_cli

_REPO_ROOT = pathlib.Path(__file__).parents[3]
D28_CASE = _REPO_ROOT / "examples" / "deterministic" / "d28-decomp-weekly-monthly"
D57_CASE = (
    _REPO_ROOT / "examples" / "deterministic" / "d57-fpha-zero-turbine-capacity"
)

_SCENARIO_PARTITION_RE = re.compile(r"scenario_id=(\d{4})")

# The per-entity simulation subdirectories D28 (training + simulation, 5
# scenarios) is known to populate -- part of the drift-prone surface R3
# targets, alongside scenario_summary.parquet, paths.parquet, and the Hive
# scenario_id=NNNN/ partitions asserted below.
_EXPECTED_SIMULATION_ENTITY_DIRS = (
    "buses",
    "costs",
    "hydros",
    "thermals",
    "hydro_bus_generation",
)


def _relative_files(root: pathlib.Path) -> set[str]:
    """Return the set of file paths under `root`, POSIX-relative to `root`.

    A plain recursive directory walk (`Path.rglob`), not a Parquet-dataset
    read -- so the Hive-partitioned `scenario_id=NNNN/` directories are
    captured as ordinary paths with no hive-vs-column ambiguity.
    """
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def _file_set_diff_message(
    cli_root: pathlib.Path,
    py_root: pathlib.Path,
    cli_files: set[str],
    py_files: set[str],
) -> str:
    """Build a diff-naming message for a CLI-vs-Python file-set mismatch."""
    only_cli = sorted(cli_files - py_files)
    only_py = sorted(py_files - cli_files)
    lines = [
        "CLI and Python output file sets diverge "
        "(the Python-parity hard rule requires them equal):"
    ]
    if only_cli:
        lines.append(f"  present in CLI output ({cli_root}) but missing from Python:")
        lines.extend(f"    {name}" for name in only_cli)
    if only_py:
        lines.append(f"  present in Python output ({py_root}) but missing from CLI:")
        lines.extend(f"    {name}" for name in only_py)
    return "\n".join(lines)


@pytest.fixture(scope="module")
def d28_cli_output(
    tmp_path_factory: pytest.TempPathFactory, cli_binary: pathlib.Path
) -> pathlib.Path:
    """Run D28 (training + simulation enabled) through the compiled CLI binary."""
    assert D28_CASE.is_dir(), f"the D28 fixture must exist at {D28_CASE}"
    output_dir = tmp_path_factory.mktemp("d28_cli_out")
    run_cli(D28_CASE, output_dir, cli_binary)
    return output_dir


@pytest.fixture(scope="module")
def d28_python_output(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Run D28 through the module-level Python bindings entry point."""
    import cobre.run  # noqa: PLC0415

    assert D28_CASE.is_dir(), f"the D28 fixture must exist at {D28_CASE}"
    output_dir = tmp_path_factory.mktemp("d28_python_out")
    cobre.run.run(str(D28_CASE), output_dir=str(output_dir))
    return output_dir


@pytest.fixture(scope="module")
def d57_cli_output(
    tmp_path_factory: pytest.TempPathFactory, cli_binary: pathlib.Path
) -> pathlib.Path:
    """Run D57 (one zero-turbine-capacity FPHA plant) through the compiled CLI binary."""
    assert D57_CASE.is_dir(), f"the D57 fixture must exist at {D57_CASE}"
    output_dir = tmp_path_factory.mktemp("d57_cli_out")
    run_cli(D57_CASE, output_dir, cli_binary)
    return output_dir


@pytest.fixture(scope="module")
def d57_python_output(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Run D57 through the module-level Python bindings entry point."""
    import cobre.run  # noqa: PLC0415

    assert D57_CASE.is_dir(), f"the D57 fixture must exist at {D57_CASE}"
    output_dir = tmp_path_factory.mktemp("d57_python_out")
    cobre.run.run(str(D57_CASE), output_dir=str(output_dir))
    return output_dir


@pytest.mark.parametrize("case", ["d28", "d57"])
def test_cli_python_output_file_sets_are_equal(
    case: str, request: pytest.FixtureRequest
) -> None:
    """The CLI and Python write paths produce the exact same set of output files.

    Compares only the SET of relative file paths -- never bytes or columns,
    which the existing `test_*_parity.py` suite already covers per-file. An
    output added to one write path but not the other (the failure mode this
    gate exists to catch -- see `test_missing_python_output_file_fails_the_gate`
    for a live demonstration) fails here with the exact missing/extra
    filename named in the assertion message.
    """
    cli_output: pathlib.Path = request.getfixturevalue(f"{case}_cli_output")
    py_output: pathlib.Path = request.getfixturevalue(f"{case}_python_output")
    cli_files = _relative_files(cli_output)
    py_files = _relative_files(py_output)
    assert cli_files == py_files, _file_set_diff_message(
        cli_output, py_output, cli_files, py_files
    )


def test_zero_turbine_case_lists_the_plant_on_both_paths(
    d57_cli_output: pathlib.Path, d57_python_output: pathlib.Path
) -> None:
    """Both write paths list D57's zero-turbine-capacity plant in hydro_models.json.

    Non-vacuity check for the D57 file-set case: the plant must reach the
    `no_turbine_capacity` key of `training/hydro_models.json` on the CLI and
    on the Python path alike.
    """
    expected = [{"hydro_id": 0, "name": "H0-Degenerate"}]
    for root in (d57_cli_output, d57_python_output):
        summary = json.loads((root / "training" / "hydro_models.json").read_text())
        assert summary["no_turbine_capacity"] == expected, (
            f"unexpected no_turbine_capacity in {root}"
        )


def test_case_exercises_the_drift_prone_output_surface(
    d28_python_output: pathlib.Path,
) -> None:
    """D28 genuinely produces the outputs most prone to CLI/Python drift.

    Non-vacuity check for the gate above: training + simulation with 5
    scenarios must produce the scenario cost summary, the reshaped per-
    scenario path table, at least one file per per-entity simulation
    subdirectory, and multiple `scenario_id=NNNN/` Hive partitions. Without
    this, a case that happened to skip the drift-prone surface would make the
    file-set equality test pass vacuously.
    """
    files = _relative_files(d28_python_output)

    assert "simulation/scenario_summary.parquet" in files
    assert "simulation/paths.parquet" in files

    for entity in _EXPECTED_SIMULATION_ENTITY_DIRS:
        entity_files = {f for f in files if f.startswith(f"simulation/{entity}/")}
        assert entity_files, f"expected at least one simulation/{entity}/ output file"

    partition_ids = {
        match.group(1) for f in files if (match := _SCENARIO_PARTITION_RE.search(f))
    }
    assert len(partition_ids) >= 2, (
        f"expected multiple scenario_id=NNNN/ partitions; got {sorted(partition_ids)}"
    )


def test_missing_python_output_file_fails_the_gate(
    d28_cli_output: pathlib.Path,
    d28_python_output: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """A file missing from the Python write path makes the gate fail, by name.

    Proves the failure signal this gate exists to produce: copies the Python
    output tree to a scratch directory (never mutating the shared fixture),
    deletes one arbitrary file -- standing in for an output added to one
    write path but never mirrored to the other -- and asserts the SAME
    equality check `test_cli_python_output_file_sets_are_equal` uses now
    raises, with the removed file's name in the message.
    """
    scratch = tmp_path / "python_output_missing_one_file"
    shutil.copytree(d28_python_output, scratch)

    cli_files = _relative_files(d28_cli_output)
    py_files_before = _relative_files(scratch)
    removed = sorted(py_files_before)[0]
    (scratch / removed).unlink()
    py_files_after = _relative_files(scratch)
    assert removed not in py_files_after

    with pytest.raises(AssertionError, match=re.escape(removed)):
        assert cli_files == py_files_after, _file_set_diff_message(
            d28_cli_output, scratch, cli_files, py_files_after
        )


def test_resolve_cli_binary_not_required_skips_when_absent(
    tmp_path: pathlib.Path,
) -> None:
    """When required=False, a missing binary raises pytest.Skipped."""
    with pytest.raises(pytest.skip.Exception):
        resolve_cli_binary(tmp_path, required=False)


def test_resolve_cli_binary_required_fails_when_absent(tmp_path: pathlib.Path) -> None:
    """When required=True, a missing binary raises pytest.Failed with build guidance."""
    with pytest.raises(
        pytest.fail.Exception, match=r"cargo build --release -p cobre-cli"
    ):
        resolve_cli_binary(tmp_path, required=True)
