"""Integration tests for cobre.results — result loading and inspection.

These tests verify that, after a completed run, the result loading functions
return correctly-shaped Python objects.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_results.py

Note: tests that invoke run() write to a temporary directory created by
pytest's tmp_path fixture. The 1dtoy case is small enough that tests complete
in a few seconds.
"""

import json
import pathlib
import shutil

import pytest


VALID_CASE = "examples/1dtoy"

_REPO_ROOT = pathlib.Path(__file__).parents[3]

# No single deck declares both travel-time arcs and post-study stages, so the
# two feature classes are exercised by their respective decks; together they
# make the four families a hardcoded reader list historically omitted
# (in_transit, transit_seed, anticipated_lanes, hydro_bus_generation) present.
TRAVEL_TIME_CASE = (
    _REPO_ROOT / "crates" / "cobre-sddp" / "tests" / "fixtures" / "travel_time_arc"
)
POST_STUDY_CASE = (
    _REPO_ROOT / "examples" / "deterministic" / "d55-post-study-anticipated-lanes"
)


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
    assert manifest["software"] == "cobre", "manifest must name the software"
    assert "software_version" in manifest, "manifest must contain 'software_version'"
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
        "upper_bound",
        "upper_bound_std",
        "upper_bound_kind",
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


def test_load_convergence_keys_equal_written_schema_fields(
    run_output: pathlib.Path,
) -> None:
    """Every column the convergence file declares reaches the caller.

    Reads the written field names from the Parquet schema itself (never a
    frozen list) and asserts the returned dict's keys equal them, in order —
    so no written column, `mean_rows_in_lp` included, is silently dropped.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    import cobre.results  # noqa: PLC0415

    convergence_path = run_output / "training" / "convergence.parquet"
    written_fields = pq.read_schema(convergence_path).names

    assert "mean_rows_in_lp" in written_fields, (
        "1dtoy after training must write mean_rows_in_lp, else this "
        "regression does not exercise the dropped-column fix"
    )

    rows = cobre.results.load_convergence(str(run_output))
    assert rows, "convergence list must be non-empty after a real run"
    for i, row in enumerate(rows):
        assert list(row.keys()) == written_fields, (
            f"row {i} keys must equal the written schema fields "
            f"(got {list(row.keys())}, want {written_fields})"
        )


def test_load_convergence_value_types(run_output: pathlib.Path) -> None:
    """Convergence rows have correct Python types for key columns."""
    import cobre.results  # noqa: PLC0415

    rows = cobre.results.load_convergence(str(run_output))
    assert rows, "must have at least one row"

    row = rows[0]
    assert isinstance(row["iteration"], int), "iteration must be int"
    assert isinstance(row["lower_bound"], float), "lower_bound must be float"
    assert isinstance(row["upper_bound"], float), "upper_bound must be float"
    # upper_bound_std is None under an exact bound, else float.
    assert row["upper_bound_std"] is None or isinstance(
        row["upper_bound_std"], float
    ), "upper_bound_std must be float or None"
    assert isinstance(row["upper_bound_kind"], str), "upper_bound_kind must be str"
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


# ---------------------------------------------------------------------------
# load_policy tests
# ---------------------------------------------------------------------------


def test_load_policy_reads_real_run_output(run_output: pathlib.Path) -> None:
    """load_policy() reads the policy checkpoint a standard run writes.

    A run writes the checkpoint to <output_dir>/policy (the default policy_path
    "./policy"), so the default policy_subdir="policy" must resolve it. The
    returned dict must carry per-stage cut pools with non-empty stage-0 cuts.
    """
    import cobre.results  # noqa: PLC0415

    policy = cobre.results.load_policy(str(run_output))

    assert isinstance(policy, dict), "load_policy must return a dict"
    assert "stage_cuts" in policy, "policy dict must have a 'stage_cuts' key"
    assert len(policy["stage_cuts"]) > 0, "a trained policy must have stage cuts"
    assert len(policy["stage_cuts"][0]["cuts"]) > 0, (
        "stage 0 must carry at least one cut after a real run"
    )


def test_load_policy_missing_dir_raises(tmp_path: pathlib.Path) -> None:
    """load_policy() raises FileNotFoundError when the policy dir is absent."""
    import cobre.results  # noqa: PLC0415

    with pytest.raises(FileNotFoundError):
        cobre.results.load_policy(str(tmp_path))


def _tree_snapshot(root: pathlib.Path) -> list[tuple[str, bytes | None]]:
    return sorted(
        (
            path.relative_to(root).as_posix(),
            path.read_bytes() if path.is_file() else None,
        )
        for path in root.rglob("*")
    )


def test_load_policy_reads_a_staged_copy_when_the_policy_dir_is_absent(
    run_output: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """load_policy() reads <policy>.staging when <policy> is absent, changing nothing."""
    import cobre.results  # noqa: PLC0415

    shutil.copytree(run_output / "policy", tmp_path / "policy.staging")
    before = _tree_snapshot(tmp_path)

    staged = cobre.results.load_policy(str(tmp_path))
    committed = cobre.results.load_policy(str(run_output))

    assert (
        staged["metadata"]["producer"]["completed_iterations"]
        == committed["metadata"]["producer"]["completed_iterations"]
    )
    assert len(staged["stage_cuts"]) == len(committed["stage_cuts"])
    assert _tree_snapshot(tmp_path) == before
    assert not (tmp_path / "policy").exists()


# ---------------------------------------------------------------------------
# load_stochastic tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stochastic_output(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Train the 1dtoy case with ``exports.stochastic`` enabled, once.

    The default 1dtoy config does NOT export stochastic artifacts, so the
    override is required to produce ``stochastic/inflow_ar_coefficients.parquet``
    and ``stochastic/noise_openings.parquet``. Module-scoped so the solver runs
    only once per session.
    """
    import cobre  # noqa: PLC0415

    output_dir = tmp_path_factory.mktemp("stochastic_output")
    cobre.Study(
        VALID_CASE,
        output_dir=str(output_dir),
        config_overrides={"exports.stochastic": True},
    ).train()
    return output_dir


def test_load_stochastic_par_coefficients_shape(
    stochastic_output: pathlib.Path,
) -> None:
    """par_coefficients() returns a 2-D (n_rows, 4) float64 array."""
    numpy = pytest.importorskip("numpy")
    import cobre.results  # noqa: PLC0415

    arr = cobre.results.load_stochastic(str(stochastic_output)).par_coefficients()

    assert arr.ndim == 2, "par_coefficients must be 2-D"
    assert arr.shape[1] == 4, "par_coefficients must have 4 columns"
    assert arr.dtype == numpy.float64, "par_coefficients must be float64"


def test_load_stochastic_opening_tree_shape(
    stochastic_output: pathlib.Path,
) -> None:
    """opening_tree(0) returns a 2-D float64 array; shape[1] == stage-0 noise dim."""
    numpy = pytest.importorskip("numpy")
    import cobre.results  # noqa: PLC0415

    stoch = cobre.results.load_stochastic(str(stochastic_output))
    arr = stoch.opening_tree(0)

    assert arr.ndim == 2, "opening_tree must be 2-D"
    assert arr.dtype == numpy.float64, "opening_tree must be float64"
    # shape[1] is the number of distinct entity_index values at stage 0, i.e.
    # the noise dimension (1 hydro for the 1dtoy single-reservoir case).
    assert arr.shape[1] >= 1, "noise dimension must be at least 1"
    assert arr.shape[0] >= 1, "stage 0 must have at least one opening"


def test_load_stochastic_missing_artifacts_raises(tmp_path: pathlib.Path) -> None:
    """load_stochastic() raises FileNotFoundError on a default (no-exports) run.

    The 1dtoy default config does not set ``exports.stochastic``, so the
    ``stochastic/`` artifacts are absent and the error message must point the
    caller at the required export flag.
    """
    import cobre  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415

    cobre.Study(VALID_CASE, output_dir=str(tmp_path)).train()

    with pytest.raises(FileNotFoundError, match="exports.stochastic"):
        cobre.results.load_stochastic(str(tmp_path))


def test_load_stochastic_opening_tree_bad_stage_raises(
    stochastic_output: pathlib.Path,
) -> None:
    """opening_tree(999) raises IndexError for an absent stage."""
    pytest.importorskip("numpy")
    import cobre.results  # noqa: PLC0415

    stoch = cobre.results.load_stochastic(str(stochastic_output))

    with pytest.raises(IndexError):
        stoch.opening_tree(999)


def test_load_stochastic_reexport_identity() -> None:
    """load_stochastic is present and is the compiled function (identity)."""
    import cobre._native.results  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415

    assert hasattr(cobre.results, "load_stochastic")
    assert cobre.results.load_stochastic is cobre._native.results.load_stochastic


# ---------------------------------------------------------------------------
# load_simulation family-coverage regression
# ---------------------------------------------------------------------------


def _run_with_simulation(src: pathlib.Path, work: pathlib.Path) -> pathlib.Path:
    """Run ``src`` with the simulation pass enabled and return the output dir.

    Deterministic decks that ship ``simulation.enabled = false`` are copied and
    flipped on so the simulation write path is exercised (mirrors the parity
    suite's ``_make_case_with_simulation``).
    """
    import cobre.run  # noqa: PLC0415

    case_dir = work / "case"
    case_dir.mkdir()
    for item in src.iterdir():
        dst = case_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    config = json.loads((case_dir / "config.json").read_text())
    config.setdefault("simulation", {})["enabled"] = True
    (case_dir / "config.json").write_text(json.dumps(config))

    output_dir = work / "output"
    cobre.run.run(str(case_dir), output_dir=str(output_dir))
    return output_dir


def _family_dirs_with_partitions(simulation_dir: pathlib.Path) -> set[str]:
    """Family subpaths under ``simulation/`` that hold a scenario partition.

    Discovered from the on-disk ``scenario_id=NNNN`` Hive layout, independent of
    the reader's own family list — so a reader that skips a present family fails.
    """
    families: set[str] = set()
    for scenario_dir in simulation_dir.rglob("scenario_id=*"):
        if scenario_dir.is_dir():
            rel = scenario_dir.parent.relative_to(simulation_dir)
            families.add(rel.as_posix())
    return families


@pytest.fixture(scope="module")
def travel_time_output(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Run the travel-time deck once; its output has in_transit/transit_seed."""
    assert TRAVEL_TIME_CASE.is_dir(), (
        f"the travel-time fixture must exist at {TRAVEL_TIME_CASE}"
    )
    return _run_with_simulation(
        TRAVEL_TIME_CASE, tmp_path_factory.mktemp("travel_time")
    )


@pytest.fixture(scope="module")
def post_study_output(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Run the post-study deck once; its output has anticipated_lanes."""
    assert POST_STUDY_CASE.is_dir(), (
        f"the post-study case must exist at {POST_STUDY_CASE}"
    )
    return _run_with_simulation(POST_STUDY_CASE, tmp_path_factory.mktemp("post_study"))


def test_load_simulation_keys_cover_every_present_family(
    travel_time_output: pathlib.Path,
    post_study_output: pathlib.Path,
) -> None:
    """The no-argument load returns a key for every present family directory.

    Exercises the two feature classes whose families the reader historically
    omitted: the travel-time deck must surface ``in_transit`` and the post-study
    deck ``anticipated_lanes``, and on each deck no present family directory may
    be missing from the returned mapping.
    """
    import cobre.results  # noqa: PLC0415

    for output_dir, must_include in (
        (travel_time_output, "in_transit"),
        (post_study_output, "anticipated_lanes"),
    ):
        simulation_dir = output_dir / "simulation"
        present = _family_dirs_with_partitions(simulation_dir)
        assert must_include in present, (
            f"the deck at {output_dir} must exercise '{must_include}'; "
            f"present families: {sorted(present)}"
        )

        data = cobre.results.load_simulation(str(output_dir))
        assert isinstance(data, dict), "no-argument load must return a dict"

        missing = present - set(data.keys())
        assert not missing, (
            f"load_simulation omitted present family directories {sorted(missing)} "
            f"under {simulation_dir}"
        )
