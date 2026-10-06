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
from typing import Any

import pyarrow.parquet as pq
import pytest

VALID_CASE = "examples/1dtoy"
D20_CASE = "examples/deterministic/d20-operational-violations"
D28_CASE = "examples/deterministic/d28-decomp-weekly-monthly"


@pytest.fixture(scope="module")
def run_output(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Run 1dtoy once and return the output directory."""
    import cobre.run

    output_dir = tmp_path_factory.mktemp("outputs_test")
    cobre.run.run(VALID_CASE, output_dir=str(output_dir))
    return output_dir


def test_training_output_files_exist(run_output: pathlib.Path) -> None:
    """A successful run produces all expected training output files."""
    # _SUCCESS is a zero-byte sentinel marker; the rest must be non-empty
    markers = {"training/_SUCCESS"}
    expected = [
        "training/_SUCCESS",
        "training/metadata.json",
        "training/convergence.parquet",
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
    assert (run_output / "simulation/metadata.json").exists()


def test_convergence_parquet_schema(run_output: pathlib.Path) -> None:
    """convergence.parquet has the expected column names."""
    schema = pq.read_schema(run_output / "training" / "convergence.parquet")
    names = set(schema.names)
    required = {
        "iteration",
        "lower_bound",
        "upper_bound",
        "upper_bound_std",
        "upper_bound_kind",
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
    """metadata.json has expected top-level keys."""
    manifest = json.loads((run_output / "training" / "metadata.json").read_text())
    assert isinstance(manifest, dict)
    assert manifest["software"] == "cobre"
    assert "software_version" in manifest
    assert "status" in manifest
    assert "convergence" in manifest


def test_policy_output_exists(run_output: pathlib.Path) -> None:
    """A successful run produces policy cuts and a self-describing manifest."""
    assert (run_output / "policy" / "manifest.bin").exists()
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
    import cobre.run

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
    import pyarrow as pa

    hydros_dir = d20_output / "simulation" / "hydros"
    parquets = sorted(hydros_dir.rglob("*.parquet"))
    assert len(parquets) > 0

    # Read each file directly (ParquetFile, not read_table): scenario_id is both
    # the Hive partition key and an in-file column, so the dataset path would try
    # to merge the two same-named fields and error.
    tables = [pq.ParquetFile(p).read() for p in parquets]
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


# ---------------------------------------------------------------------------
# D28 weekly+monthly parity tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def d28_output(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Run D28 case via Python bindings and return the output directory."""
    import cobre.run

    src = pathlib.Path(D28_CASE)
    case_dir = tmp_path_factory.mktemp("d28_case")

    for item in src.iterdir():
        dest = case_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    output_dir = tmp_path_factory.mktemp("d28_output")
    cobre.run.run(str(case_dir), output_dir=str(output_dir))
    return output_dir


def test_d28_runs_and_produces_outputs(d28_output: pathlib.Path) -> None:
    """D28 case produces all standard training and simulation outputs."""
    # Training artifacts
    assert (d28_output / "training" / "_SUCCESS").is_file()
    assert (d28_output / "training" / "metadata.json").is_file()
    assert (d28_output / "training" / "convergence.parquet").is_file()
    assert (d28_output / "training" / "scaling_report.json").is_file()

    # Policy artifacts
    assert (d28_output / "policy" / "manifest.bin").is_file()
    cuts_dir = d28_output / "policy" / "cuts"
    assert cuts_dir.is_dir()
    assert len(list(cuts_dir.iterdir())) > 0, "policy/cuts/ must contain files"

    # Simulation artifacts
    sim = d28_output / "simulation"
    assert (sim / "_SUCCESS").is_file()
    assert (sim / "metadata.json").is_file()
    for subdir in ("buses", "hydros", "thermals", "costs"):
        assert (sim / subdir).is_dir(), f"simulation/{subdir}/ must exist"


def test_d28_convergence_has_iterations(d28_output: pathlib.Path) -> None:
    """D28 convergence metadata reports at least 1 training iteration."""
    meta_path = d28_output / "training" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    assert "status" in meta
    assert meta.get("iterations", {}).get("completed", 0) > 0

    conv_path = d28_output / "training" / "convergence.parquet"
    table = pq.read_table(conv_path)
    assert len(table) > 0, "convergence.parquet must have at least one row"


# ---------------------------------------------------------------------------
# Phase success markers
# ---------------------------------------------------------------------------


def _run_1dtoy_with_a_directory_at(output_dir: pathlib.Path, blocked: str) -> None:
    import cobre.errors
    import cobre.run

    (output_dir / blocked).mkdir(parents=True)
    with pytest.raises(cobre.errors.CaseIoError):
        cobre.run.run(VALID_CASE, output_dir=str(output_dir))


def test_run_writes_no_training_marker_when_the_last_training_write_fails(
    tmp_path: pathlib.Path,
) -> None:
    """A failed training write leaves no training/_SUCCESS beside the files written before it."""
    _run_1dtoy_with_a_directory_at(tmp_path, "training/solver/retry_histogram.parquet.tmp")

    assert (tmp_path / "training" / "metadata.json").is_file()
    assert not (tmp_path / "training" / "_SUCCESS").exists()


def test_run_writes_no_simulation_marker_when_the_last_simulation_write_fails(
    tmp_path: pathlib.Path,
) -> None:
    """A failed simulation metadata write leaves no simulation/_SUCCESS."""
    _run_1dtoy_with_a_directory_at(tmp_path, "simulation/metadata.json.tmp")

    assert not (tmp_path / "simulation" / "_SUCCESS").exists()
    assert not (tmp_path / "simulation" / "metadata.json").exists()
    assert (tmp_path / "simulation" / "scenario_summary.parquet").is_file()
    assert (tmp_path / "training" / "_SUCCESS").is_file()


def _seed_empty_markers(output_dir: pathlib.Path, *phases: str) -> None:
    for phase in phases:
        (output_dir / phase).mkdir(parents=True, exist_ok=True)
        (output_dir / phase / "_SUCCESS").touch()


def _marker_states(output_dir: pathlib.Path) -> tuple[bool, bool]:
    return (
        (output_dir / "training" / "_SUCCESS").exists(),
        (output_dir / "simulation" / "_SUCCESS").exists(),
    )


def test_run_clears_stale_markers_before_training(tmp_path: pathlib.Path) -> None:
    """A run into a reused directory shows neither stale marker while it trains."""
    import cobre.run

    _seed_empty_markers(tmp_path, "training", "simulation")
    observed: list[tuple[bool, bool]] = []

    def on_iteration(_event: dict[str, Any]) -> None:
        observed.append(_marker_states(tmp_path))

    cobre.run.run(VALID_CASE, output_dir=str(tmp_path), on_iteration=on_iteration)

    assert observed, "on_iteration was never called"
    assert set(observed) == {(False, False)}
    assert _marker_states(tmp_path) == (True, True)


def test_simulate_clears_stale_marker_before_its_first_write(
    tmp_path: pathlib.Path,
) -> None:
    """A simulate() whose first write fails leaves no stale simulation/_SUCCESS."""
    import cobre
    import cobre.errors

    study = cobre.Study(VALID_CASE, output_dir=str(tmp_path / "trained"))
    policy = study.train()
    target = tmp_path / "target"
    _seed_empty_markers(target, "simulation")
    (target / "simulation" / "costs").touch()

    with pytest.raises(cobre.errors.CobreError):
        study.simulate(policy, output_dir=str(target))

    assert not (target / "simulation" / "_SUCCESS").exists()


def test_study_train_clears_only_its_own_marker(tmp_path: pathlib.Path) -> None:
    """Study.train() hides the stale training marker and keeps the simulation one."""
    import cobre

    _seed_empty_markers(tmp_path, "training", "simulation")
    study = cobre.Study(VALID_CASE, output_dir=str(tmp_path))
    assert _marker_states(tmp_path) == (True, True)
    observed: list[tuple[bool, bool]] = []

    def on_iteration(_event: dict[str, Any]) -> None:
        observed.append(_marker_states(tmp_path))

    study.train(on_iteration=on_iteration)

    assert observed, "on_iteration was never called"
    assert set(observed) == {(False, True)}
    assert _marker_states(tmp_path) == (True, True)


def test_load_policy_then_simulate_elsewhere_keeps_the_trained_markers(
    tmp_path: pathlib.Path,
) -> None:
    """Loading a policy from X and simulating into Y leaves X's markers in place."""
    import cobre
    import cobre.results
    import cobre.run

    trained = tmp_path / "trained"
    elsewhere = tmp_path / "elsewhere"
    cobre.run.run(VALID_CASE, output_dir=str(trained))

    study = cobre.Study(VALID_CASE, output_dir=str(trained))
    policy = study.load_policy()
    study.simulate(policy, output_dir=str(elsewhere))

    assert _marker_states(trained) == (True, True)
    assert (elsewhere / "simulation" / "_SUCCESS").is_file()
    cobre.results.load_results(str(trained))


def _seed_file(path: pathlib.Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_run_clears_stale_simulation_outputs_before_training(
    tmp_path: pathlib.Path,
) -> None:
    """A run into a reused directory shows no earlier simulation output while it trains."""
    import cobre.run

    sim = tmp_path / "simulation"
    stale = {
        sim / "costs" / "scenario_id=9999" / "data.parquet": "",
        sim / "solver" / "iterations.parquet": "stale",
        sim / "paths.parquet": "stale",
        sim / "metadata.json": "{}",
    }
    for path, content in stale.items():
        _seed_file(path, content)
    _seed_file(sim / "solver" / "stale.txt")
    observed: list[tuple[bool, ...]] = []

    def on_iteration(_event: dict[str, Any]) -> None:
        observed.append(tuple(path.exists() for path in stale))

    cobre.run.run(VALID_CASE, output_dir=str(tmp_path), on_iteration=on_iteration)

    assert observed, "on_iteration was never called"
    assert set(observed) == {(False, False, False, False)}
    assert (sim / "costs" / "scenario_id=0000" / "data.parquet").is_file()
    assert (sim / "solver" / "stale.txt").is_file()
    assert (sim / "_SUCCESS").is_file()
    pq.read_table(sim / "paths.parquet")
    pq.read_table(sim / "solver" / "iterations.parquet")


def test_simulate_clears_stale_outputs_before_writing(tmp_path: pathlib.Path) -> None:
    """simulate() into a reused directory drops earlier partitions and keeps foreign files."""
    import cobre

    study = cobre.Study(VALID_CASE, output_dir=str(tmp_path / "trained"))
    policy = study.train()
    target = tmp_path / "target"
    sim = target / "simulation"
    _seed_file(sim / "costs" / "scenario_id=9999" / "data.parquet")
    _seed_file(sim / "pumping_stations" / "scenario_id=0000" / "data.parquet")
    _seed_file(sim / "solver" / "stale.txt")
    _seed_file(sim / "notes.txt")

    study.simulate(policy, output_dir=str(target))

    assert not (sim / "costs" / "scenario_id=9999").exists()
    assert not (sim / "pumping_stations").exists()
    for kept in (
        "costs/scenario_id=0000/data.parquet",
        "solver/stale.txt",
        "notes.txt",
        "_SUCCESS",
    ):
        assert (sim / kept).is_file(), f"simulation/{kept} must exist"


# ---------------------------------------------------------------------------
# Conditional training outputs
# ---------------------------------------------------------------------------

# The conditional training outputs 1dtoy never writes.
_STALE_CONDITIONAL_TRAINING_OUTPUTS = (
    "training/cut_selection/iterations.parquet",
    "hydro_models/fpha_hyperplanes.parquet",
    "hydro_models/evaporation_models.parquet",
    "hydro_models/fpha_deviation_points.parquet",
    "generic_constraints/resolved_echo.parquet",
    "anticipated/fixed_deliveries.parquet",
)


def _rewrite_config(case_dir: pathlib.Path, edit: Any) -> None:
    config_path = case_dir / "config.json"
    config = json.loads(config_path.read_text())
    edit(config)
    config_path.write_text(json.dumps(config))


def test_run_clears_stale_training_outputs_before_training(
    tmp_path: pathlib.Path,
) -> None:
    """A run into a reused directory shows no earlier conditional training output while it trains."""
    import cobre.run

    stale = [tmp_path / relative for relative in _STALE_CONDITIONAL_TRAINING_OUTPUTS]
    for path in stale:
        _seed_file(path, "stale")
    observed: list[tuple[bool, ...]] = []

    def on_iteration(_event: dict[str, Any]) -> None:
        observed.append(tuple(path.exists() for path in stale))

    cobre.run.run(VALID_CASE, output_dir=str(tmp_path), on_iteration=on_iteration)

    assert observed, "on_iteration was never called"
    assert set(observed) == {(False,) * len(stale)}
    assert (tmp_path / "training" / "_SUCCESS").is_file()


def test_study_train_clears_stale_training_outputs(tmp_path: pathlib.Path) -> None:
    """Study() keeps an earlier run's conditional training outputs; train() removes them."""
    import cobre

    stale = [
        tmp_path / "hydro_models" / "fpha_hyperplanes.parquet",
        tmp_path / "training" / "cut_selection" / "iterations.parquet",
    ]
    for path in stale:
        _seed_file(path, "stale")
    study = cobre.Study(VALID_CASE, output_dir=str(tmp_path))
    assert all(path.is_file() for path in stale)

    study.train()

    assert not any(path.exists() for path in stale)
    assert (tmp_path / "training" / "_SUCCESS").is_file()


def test_run_clears_cut_selection_output_after_cut_selection_is_disabled(
    tmp_path: pathlib.Path,
) -> None:
    """A rerun without cut selection leaves no training/cut_selection/ from the first run."""
    import cobre.run

    case_dir = tmp_path / "case"
    output_dir = tmp_path / "output"
    shutil.copytree(VALID_CASE, case_dir)

    def enable_cut_selection(config: dict[str, Any]) -> None:
        config["training"]["stopping_rules"] = [{"type": "iteration_limit", "limit": 6}]
        config["training"]["cut_selection"] = {
            "selection": {"method": "level1", "check_frequency": 2}
        }
        config["simulation"]["enabled"] = False

    _rewrite_config(case_dir, enable_cut_selection)
    cobre.run.run(str(case_dir), output_dir=str(output_dir))
    assert (output_dir / "training" / "cut_selection" / "iterations.parquet").is_file()

    _rewrite_config(case_dir, lambda config: config["training"].pop("cut_selection"))
    cobre.run.run(str(case_dir), output_dir=str(output_dir))

    assert (output_dir / "training" / "_SUCCESS").is_file()
    assert not (output_dir / "training" / "cut_selection").exists()


def test_warm_start_rerun_reads_the_policy_the_training_clear_keeps(
    tmp_path: pathlib.Path,
) -> None:
    """A warm-start rerun into the same directory reads the policy the first run wrote."""
    import cobre.run

    cobre.run.run(VALID_CASE, output_dir=str(tmp_path))
    cobre.run.run(
        VALID_CASE,
        output_dir=str(tmp_path),
        config_overrides={"policy.mode": "warm_start"},
    )

    assert (tmp_path / "training" / "_SUCCESS").is_file()


def test_simulation_only_run_keeps_training_outputs(tmp_path: pathlib.Path) -> None:
    """A run with training disabled removes no training output."""
    import cobre.run

    cobre.run.run(VALID_CASE, output_dir=str(tmp_path))
    seeded = tmp_path / "hydro_models" / "fpha_hyperplanes.parquet"
    _seed_file(seeded, "stale")

    cobre.run.run(
        VALID_CASE,
        output_dir=str(tmp_path),
        config_overrides={"training.enabled": False},
    )

    assert seeded.is_file()
    assert (tmp_path / "training" / "solver" / "iterations.parquet").is_file()
    assert (tmp_path / "training" / "_SUCCESS").is_file()
