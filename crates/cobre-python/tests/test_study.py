"""Integration tests for the cobre.Study pyclass.

These tests verify that a case directory can be loaded once into a live,
reusable Study (front half of the solve lifecycle: load -> stochastic
preprocessing -> hydro models -> StudySetup construction -> sidecar writes),
that `validate` reports the captured warnings and checks the configured policy
load without a reload, and that a missing case directory is rejected before any
work runs.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_study.py

Note: each construction writes the front-half sidecars to a temporary directory
created by pytest's tmp_path fixture. The 1dtoy case is small enough that tests
complete in a few seconds.
"""

import json
import pathlib
import shutil

import pytest


VALID_CASE = "examples/1dtoy"
MISSING_CASE = "/tmp/nonexistent_cobre_case_xzy123"


def _read_training_metadata(output_dir: pathlib.Path) -> dict:
    """Read back the training metadata.json written by a train/run."""
    with open(output_dir / "training" / "metadata.json") as handle:
        return json.load(handle)


def _read_simulation_metadata(output_dir: pathlib.Path) -> dict:
    """Read back the simulation metadata.json written by a simulate/run."""
    with open(output_dir / "simulation" / "metadata.json") as handle:
        return json.load(handle)


def test_study_constructs_and_validates(tmp_path: pathlib.Path) -> None:
    """Study(case_dir) loads once and exposes a valid report + system view."""
    import cobre  # noqa: PLC0415

    study = cobre.Study(VALID_CASE, output_dir=str(tmp_path))

    assert study.output_dir == str(tmp_path), "output_dir must echo the resolved path"

    system = study.system
    assert isinstance(system, cobre.model.System), (
        "system getter must return a cobre.model.System"
    )
    assert system.n_stages > 0, "loaded system must report stages"

    # The system view is a cheap Arc bump, so repeated access returns equivalent
    # snapshots without a reload.
    assert study.system.n_stages == system.n_stages, (
        "repeated system access must be consistent"
    )

    report = study.validate()
    assert isinstance(report, dict), "validate() must return a dict"
    assert report["valid"] is True, "a successfully loaded study is valid"
    assert report["errors"] == [], "a valid study has no errors"
    assert isinstance(report["warnings"], list), "warnings must be a list"
    for warning in report["warnings"]:
        assert {"kind", "message", "file", "entity"} <= set(warning.keys()), (
            "each warning must carry kind/message/file/entity keys"
        )


def test_study_writes_front_half_sidecars(tmp_path: pathlib.Path) -> None:
    """Constructing a Study writes the three front-half training sidecars."""
    import cobre  # noqa: PLC0415

    cobre.Study(VALID_CASE, output_dir=str(tmp_path))

    for sidecar in (
        "training/scaling_report.json",
        "training/model_provenance.json",
        "training/hydro_models.json",
    ):
        path = tmp_path / sidecar
        assert path.exists(), f"{sidecar} must exist after construction"


def test_study_missing_case_raises(tmp_path: pathlib.Path) -> None:
    """Study raises OSError when the case directory does not exist."""
    import cobre  # noqa: PLC0415

    with pytest.raises(OSError):
        cobre.Study(MISSING_CASE, output_dir=str(tmp_path))


def test_study_rejects_threads_zero(tmp_path: pathlib.Path) -> None:
    """Study raises ValueError when threads=0."""
    import cobre  # noqa: PLC0415

    with pytest.raises(ValueError, match="threads"):
        cobre.Study(VALID_CASE, output_dir=str(tmp_path), threads=0)


def test_study_train_returns_policy(tmp_path: pathlib.Path) -> None:
    """Study.train() trains in-memory, writes _SUCCESS, and returns a Policy."""
    import cobre  # noqa: PLC0415

    study = cobre.Study(VALID_CASE, output_dir=str(tmp_path))
    policy = study.train()

    assert isinstance(policy, cobre.Policy), "train() must return a cobre.Policy"
    assert policy.iterations > 0, "a trained policy must report completed iterations"
    # The headline bounds are exposed on the handle.
    assert isinstance(policy.final_lower_bound, float)
    assert isinstance(policy.final_upper_bound, float)
    assert (tmp_path / "training" / "_SUCCESS").exists(), (
        "train() must write the training/_SUCCESS marker"
    )


def test_study_train_matches_run_lower_bound(tmp_path: pathlib.Path) -> None:
    """Study.train and cobre.run.run train identically (same final_lower_bound).

    Both paths train the same 1dtoy case into separate temp dirs; the persisted
    training metadata final_lower_bound must agree within 1e-6 relative, proving
    Study.train and run.run share the training path (no divergence).
    """
    import cobre  # noqa: PLC0415

    study_out = tmp_path / "study_out"
    run_out = tmp_path / "run_out"

    study = cobre.Study(VALID_CASE, output_dir=str(study_out))
    policy = study.train()

    cobre.run.run(VALID_CASE, output_dir=str(run_out))

    study_meta = _read_training_metadata(study_out)
    run_meta = _read_training_metadata(run_out)

    study_lb = study_meta["bounds"]["final_lower_bound"]
    run_lb = run_meta["bounds"]["final_lower_bound"]
    rel = abs(study_lb - run_lb) / abs(run_lb)
    assert rel < 1e-6, (
        f"Study.train final_lower_bound {study_lb} not within 1e-6 of "
        f"run.run {run_lb} (rel={rel})"
    )

    # The handle's bound must agree with the persisted metadata.
    assert abs(policy.final_lower_bound - study_lb) / abs(study_lb) < 1e-6, (
        "policy.final_lower_bound must match the persisted training metadata"
    )


def test_study_train_callback_cooperative_stop(tmp_path: pathlib.Path) -> None:
    """A truthy callback return stops training cooperatively (async).

    The cooperative-stop contract is asynchronous: a truthy return at iteration N
    sets a shared flag that the solver observes at a later boundary (N+k). The
    test therefore asserts that the callback fired, that training stopped well
    before the configured iteration limit, and that partial artifacts were
    written — NOT a literal stop iteration. The `len(calls) <= 6` upper bound is
    a 1dtoy-calibrated ceiling (k is small for this case), documented here as a
    test heuristic, not an API guarantee.
    """
    import cobre  # noqa: PLC0415

    calls = []

    def on_iteration(event):
        calls.append(event)
        return len(calls) >= 3

    study = cobre.Study(VALID_CASE, output_dir=str(tmp_path))
    policy = study.train(on_iteration=on_iteration)

    assert len(calls) >= 1, "the callback must fire at least once"
    assert len(calls) <= 6, (
        "cooperative stop must take effect within the 1dtoy-calibrated ceiling"
    )

    meta = _read_training_metadata(tmp_path)
    completed = meta["iterations"]["completed"]
    max_iterations = meta["configuration"]["max_iterations"]
    assert max_iterations is not None, "1dtoy must persist a max_iterations limit"
    assert completed < max_iterations, (
        f"cooperative stop must end training ({completed}) before the limit "
        f"({max_iterations})"
    )

    assert policy.iterations == completed, (
        "policy.iterations must equal the persisted completed-iteration count"
    )
    assert (tmp_path / "training" / "_SUCCESS").exists(), (
        "a cooperatively stopped run still writes partial artifacts"
    )


def test_study_train_raising_callback_propagates_after_artifacts(
    tmp_path: pathlib.Path,
) -> None:
    """A raising callback re-raises verbatim AFTER artifacts are written."""
    import cobre  # noqa: PLC0415

    def on_iteration(event):
        raise ValueError("boom")

    study = cobre.Study(VALID_CASE, output_dir=str(tmp_path))

    with pytest.raises(ValueError, match="boom"):
        study.train(on_iteration=on_iteration)

    # The exception propagates only after the partial artifacts are persisted.
    assert (tmp_path / "training" / "metadata.json").exists(), (
        "training metadata must be written before the callback exception propagates"
    )


def test_study_train_then_simulate(tmp_path: pathlib.Path) -> None:
    """Study.train().simulate() runs the simulation against the in-memory policy.

    The trained Policy carries the live baked basis cache, so simulate runs
    without any checkpoint reload. Asserts the returned dict and the _SUCCESS
    marker.
    """
    import cobre  # noqa: PLC0415

    study = cobre.Study(VALID_CASE, output_dir=str(tmp_path))
    policy = study.train()
    sim = study.simulate(policy)

    assert sim == {"n_scenarios": 100, "completed": 100}, (
        "simulate() must return the scenario counts as a dict"
    )
    assert (tmp_path / "simulation" / "_SUCCESS").exists(), (
        "simulate() must write the simulation/_SUCCESS marker"
    )


def test_in_memory_simulate_matches_run_metadata(tmp_path: pathlib.Path) -> None:
    """P3 (make-or-break): in-memory train().simulate() == monolithic run.run.

    Study.train().simulate() into tmp_a and cobre.run.run into tmp_b for the same
    case + seed must write simulation metadata whose cost.mean_cost agrees within
    1e-6 relative and whose solve_stats.total_lp_solves is EXACTLY equal. This is
    the load-bearing invariant: the in-memory simulate must be bit-identical to
    the monolithic on-disk simulate.
    """
    import cobre  # noqa: PLC0415

    tmp_a = tmp_path / "study_out"
    tmp_b = tmp_path / "run_out"

    study = cobre.Study(VALID_CASE, output_dir=str(tmp_a))
    policy = study.train()
    study.simulate(policy)

    cobre.run.run(VALID_CASE, output_dir=str(tmp_b))

    meta_a = _read_simulation_metadata(tmp_a)
    meta_b = _read_simulation_metadata(tmp_b)

    cost_a = meta_a["cost"]["mean_cost"]
    cost_b = meta_b["cost"]["mean_cost"]
    rel = abs(cost_a - cost_b) / abs(cost_b)
    assert rel < 1e-6, (
        f"in-memory simulate mean_cost {cost_a} not within 1e-6 of run.run "
        f"{cost_b} (rel={rel})"
    )

    assert (
        meta_a["solve_stats"]["total_lp_solves"]
        == meta_b["solve_stats"]["total_lp_solves"]
    ), "in-memory simulate total_lp_solves must EXACTLY equal run.run"


def test_repeated_simulate_one_policy(tmp_path: pathlib.Path) -> None:
    """A single trained Policy may be simulated repeatedly (no reload).

    Train once, then simulate twice into two different output_dirs. Both calls
    return the full scenario counts and write equal mean_cost (the study's
    simulate is read-only over the setup, so repeated calls are deterministic).
    """
    import cobre  # noqa: PLC0415

    study = cobre.Study(VALID_CASE, output_dir=str(tmp_path))
    policy = study.train()

    out_1 = tmp_path / "sim_1"
    out_2 = tmp_path / "sim_2"

    sim_1 = study.simulate(policy, output_dir=str(out_1))
    sim_2 = study.simulate(policy, output_dir=str(out_2))

    assert sim_1 == {"n_scenarios": 100, "completed": 100}
    assert sim_2 == {"n_scenarios": 100, "completed": 100}

    cost_1 = _read_simulation_metadata(out_1)["cost"]["mean_cost"]
    cost_2 = _read_simulation_metadata(out_2)["cost"]["mean_cost"]
    assert cost_1 == cost_2, (
        "repeated simulation against one in-memory policy must be deterministic"
    )


def test_load_policy_missing_dir_raises_validation_error(tmp_path: pathlib.Path) -> None:
    """load_policy() with no prior training raises ValidationError.

    The error message must mention the missing policy directory so callers can
    diagnose a simulation-only request against an untrained output dir.
    """
    import cobre  # noqa: PLC0415
    import cobre.errors  # noqa: PLC0415

    study = cobre.Study(VALID_CASE, output_dir=str(tmp_path))

    with pytest.raises(cobre.errors.ValidationError, match="Policy directory not found"):
        study.load_policy()


def test_stochastic_data_refusal_raises_validation_error(tmp_path: pathlib.Path) -> None:
    """A historical inflow scheme over a case with no inflow history is refused
    at setup with ValidationError, from both `Study(...)` and `run(...)`."""
    import cobre  # noqa: PLC0415
    import cobre.errors  # noqa: PLC0415
    import cobre.run  # noqa: PLC0415

    case = tmp_path / "case"
    shutil.copytree(VALID_CASE, case)
    config_path = case / "config.json"
    config = json.loads(config_path.read_text())
    config["training"]["scenario_source"]["inflow"] = {"scheme": "historical"}
    config_path.write_text(json.dumps(config))
    expected = "stochastic error: insufficient data: no valid historical windows found"

    with pytest.raises(cobre.errors.ValidationError, match=expected):
        cobre.Study(str(case), output_dir=str(tmp_path / "study_output"))

    with pytest.raises(cobre.errors.ValidationError, match=expected):
        cobre.run.run(str(case), output_dir=str(tmp_path / "run_output"))


def test_simulate_zero_cut_policy_raises(tmp_path: pathlib.Path) -> None:
    """simulate() rejects a cut-less policy instead of silently simulating wrong.

    With training disabled, Study.train() returns a synthetic zero-cut policy.
    Feeding that policy to simulate() would run with no Benders cuts and produce
    a wrong result, so simulate() must raise and direct the caller to
    load_policy() instead.
    """
    import cobre  # noqa: PLC0415

    study = cobre.Study(
        VALID_CASE,
        output_dir=str(tmp_path),
        config_overrides={"training.enabled": False},
    )
    policy = study.train()

    with pytest.raises(RuntimeError, match="no cuts to simulate"):
        study.simulate(policy)


def test_load_policy_then_simulate_matches_run(tmp_path: pathlib.Path) -> None:
    """A loaded policy feeds the IDENTICAL simulate path as a trained one.

    Produce a completed run with cobre.run.run, then load that policy from disk
    into a fresh Study and simulate. The loaded policy's iteration count must
    equal the run's completed iterations and the simulate must return the full
    scenario counts.
    """
    import cobre  # noqa: PLC0415

    run_dir = tmp_path / "run_dir"
    study_dir = tmp_path / "study_dir"

    cobre.run.run(VALID_CASE, output_dir=str(run_dir))
    run_completed = _read_training_metadata(run_dir)["iterations"]["completed"]

    study = cobre.Study(VALID_CASE, output_dir=str(study_dir))
    policy = study.load_policy(output_dir=str(run_dir))
    assert policy.iterations == run_completed, (
        "load_policy().iterations must equal the run's completed-iteration count"
    )

    sim = study.simulate(policy)
    assert sim == {"n_scenarios": 100, "completed": 100}


def test_policy_evaluate_matches_cut_matrix_max(tmp_path: pathlib.Path) -> None:
    """policy.evaluate(stage, state) == max(intercept + coeffs @ state) exactly.

    The FCF value at a state is the upper envelope of its active Benders cuts.
    Both evaluate() and cut_matrix() read the same stored cut data, so the
    equality is exact f64 (no tolerance).
    """
    np = pytest.importorskip("numpy")
    import cobre  # noqa: PLC0415

    study = cobre.Study(VALID_CASE, output_dir=str(tmp_path))
    policy = study.train()

    intercepts, coeffs = policy.cut_matrix(0)
    n_cuts, dim = coeffs.shape
    assert n_cuts > 0, "a trained 1dtoy policy must have active cuts at stage 0"

    # Derive a state of the policy's dimension from the coefficient column count.
    state = np.linspace(0.5, 1.5, dim)

    expected = float(np.max(intercepts + coeffs @ state))
    got = policy.evaluate(0, list(state))

    assert got == expected, (
        f"evaluate {got} must equal max(intercept + coeffs @ state) {expected} exactly"
    )


def test_policy_cut_matrix_shapes_and_dtype(tmp_path: pathlib.Path) -> None:
    """cut_matrix(0) returns (n,)/(n, dim) float64 arrays; n == active-cut count."""
    np = pytest.importorskip("numpy")
    import cobre  # noqa: PLC0415

    study = cobre.Study(VALID_CASE, output_dir=str(tmp_path))
    policy = study.train()

    intercepts, coeffs = policy.cut_matrix(0)
    n_cuts, dim = coeffs.shape

    assert intercepts.shape == (n_cuts,), "intercepts must have shape (n_cuts,)"
    assert coeffs.shape == (n_cuts, dim), "coeffs must have shape (n_cuts, dim)"
    assert intercepts.dtype == np.float64, "intercepts dtype must be float64"
    assert coeffs.dtype == np.float64, "coeffs dtype must be float64"

    # n_cuts must equal the active-cut count for stage 0 from the on-disk policy.
    loaded = cobre.results.load_policy(str(tmp_path))
    active = [c for c in loaded["stage_cuts"][0]["cuts"] if c["is_active"]]
    assert n_cuts == len(active), (
        f"cut_matrix active-cut count {n_cuts} must equal the load_policy "
        f"active-cut count {len(active)}"
    )


def test_policy_evaluate_stage_out_of_range_raises_indexerror(
    tmp_path: pathlib.Path,
) -> None:
    """evaluate() with a stage index past the horizon raises IndexError."""
    pytest.importorskip("numpy")
    import cobre  # noqa: PLC0415

    study = cobre.Study(VALID_CASE, output_dir=str(tmp_path))
    policy = study.train()

    _, coeffs = policy.cut_matrix(0)
    dim = coeffs.shape[1]

    with pytest.raises(IndexError, match="out of range"):
        policy.evaluate(99, [0.0] * dim)


def test_policy_evaluate_bad_state_length_raises_valueerror(
    tmp_path: pathlib.Path,
) -> None:
    """evaluate() with a state of the wrong length raises ValueError."""
    pytest.importorskip("numpy")
    import cobre  # noqa: PLC0415

    study = cobre.Study(VALID_CASE, output_dir=str(tmp_path))
    policy = study.train()

    with pytest.raises(ValueError, match="expected"):
        policy.evaluate(0, [])


def test_train_and_simulate_docstrings_name_written_paths(
    tmp_path: pathlib.Path,
) -> None:
    """Docstrings for train() and simulate() list only paths they actually write.

    Unconditional paths documented in the train/simulate docstrings must both
    exist on disk after a 1dtoy train+simulate AND appear verbatim in the
    corresponding __doc__. The four historically drifted paths must be absent.
    """
    import cobre  # noqa: PLC0415

    study = cobre.Study(VALID_CASE, output_dir=str(tmp_path))
    policy = study.train()
    study.simulate(policy)

    train_paths = [
        "policy/",  # Policy tree directory
        "training/metadata.json",
        "training/_SUCCESS",
        "training/convergence.parquet",
        "training/timing/iterations.parquet",
        "training/solver/iterations.parquet",
        "training/solver/retry_histogram.parquet",
    ]

    # Dictionary files are written individually but documented as a group
    train_dict_files = [
        "training/dictionaries/variables.csv",
        "training/dictionaries/entities.csv",
        "training/dictionaries/codes.json",
        "training/dictionaries/bounds.parquet",
    ]

    train_doc = cobre.Study.train.__doc__
    assert train_doc is not None, "train() must have a docstring"

    for path in train_paths:
        full_path = tmp_path / path
        if path.endswith("/"):
            assert full_path.exists() and full_path.is_dir(), (
                f"train() must write {path} directory"
            )
        else:
            assert full_path.exists() and full_path.is_file(), (
                f"train() must write {path} file"
            )
        assert path in train_doc, f"train() docstring must mention {path}"

    assert "training/dictionaries/" in train_doc, (
        "train() docstring must mention training/dictionaries/"
    )
    for dict_file in train_dict_files:
        full_path = tmp_path / dict_file
        assert full_path.exists() and full_path.is_file(), (
            f"train() must write {dict_file} file"
        )
        filename = dict_file.split("/")[-1]
        assert filename in train_doc, f"train() docstring must mention {filename}"

    simulate_paths = [
        "simulation/metadata.json",
        "simulation/_SUCCESS",
        "simulation/paths.parquet",
        "simulation/scenario_summary.parquet",
        "simulation/solver/iterations.parquet",
        "simulation/solver/retry_histogram.parquet",
    ]

    simulate_doc = cobre.Study.simulate.__doc__
    assert simulate_doc is not None, "simulate() must have a docstring"

    for path in simulate_paths:
        full_path = tmp_path / path
        assert full_path.exists() and full_path.is_file(), (
            f"simulate() must write {path} file"
        )
        assert path in simulate_doc, f"simulate() docstring must mention {path}"

    false_paths = [
        "training/solver_stats.parquet",
        "training/cut_selection.parquet",
        "training/policy/",
        "simulation/solver_stats.parquet",
    ]

    for path in false_paths:
        assert path not in train_doc, (
            f"train() docstring must NOT mention the false path {path}"
        )
        assert path not in simulate_doc, (
            f"simulate() docstring must NOT mention the false path {path}"
        )


def test_study_summary_properties_match_run_result(tmp_path: pathlib.Path) -> None:
    """Study summary properties equal the same-named keys from run.run().

    The three construction-time reports (stochastic, hydro_models, provenance)
    must produce identical dicts whether accessed via Study properties or the
    run.run() result dict. Two output directories differ deliberately -- proving
    the sections carry no output-path-dependent values, which makes the equality
    assertion a parity claim rather than a coincidence.
    """
    import cobre  # noqa: PLC0415

    run_dir = tmp_path / "run"
    study_dir = tmp_path / "study"

    result = cobre.run.run(VALID_CASE, output_dir=str(run_dir))
    study = cobre.Study(VALID_CASE, output_dir=str(study_dir))

    assert study.stochastic == result["stochastic"], (
        "Study.stochastic must equal run.run()['stochastic']"
    )
    assert study.hydro_models == result["hydro_models"], (
        "Study.hydro_models must equal run.run()['hydro_models']"
    )
    assert study.provenance == result["provenance"], (
        "Study.provenance must equal run.run()['provenance']"
    )

    # Repeated access returns equal dicts (the getter builds a fresh dict per call).
    assert study.provenance == study.provenance, (
        "repeated access to Study.provenance must return equal dicts"
    )


_ONE_ITERATION_NO_SIMULATION = {
    "training.stopping_rules": [{"type": "iteration_limit", "limit": 1}],
    "simulation.enabled": False,
}
_WARM_START = {"policy.mode": "warm_start"}
_SIMULATION_ONLY = {
    "training.enabled": False,
    "simulation.enabled": True,
    "simulation.selection": {"method": "sampled", "num_scenarios": 1},
}


def _restamp_policy_version(policy_dir: pathlib.Path) -> None:
    """Rewrite the cobre version in `policy_dir/manifest.bin` to another string of
    the same byte length, so the FlatBuffers layout is unchanged."""
    import cobre  # noqa: PLC0415

    manifest = policy_dir / "manifest.bin"
    data = manifest.read_bytes()
    running = cobre.__version__.encode()
    assert data.count(running) == 1, "the running version occurs once in the manifest"
    other = (b"8" if running.startswith(b"9") else b"9") + running[1:]
    manifest.write_bytes(data.replace(running, other))


def _copy_case_with_extra_thermal(dest: pathlib.Path) -> None:
    """Copy 1dtoy into `dest` with a third thermal on the same bus, costlier than
    the others: more LP columns, identical state."""
    shutil.copytree(VALID_CASE, dest)
    thermals_path = dest / "system" / "thermals.json"
    thermals = json.loads(thermals_path.read_text())
    existing = thermals["thermals"]
    existing.append(
        {
            "id": max(t["id"] for t in existing) + 1,
            "name": "UTE_EXTRA",
            "operational_start_date": "2020-01-01",
            "bus_id": existing[0]["bus_id"],
            "generation": {"min_mw": 0.0, "max_mw": 15.0},
            "cost_per_mwh": max(t["cost_per_mwh"] for t in existing) + 10.0,
        }
    )
    thermals_path.write_text(json.dumps(thermals))


def test_study_validate_refuses_a_policy_that_train_refuses(
    tmp_path: pathlib.Path,
) -> None:
    """A warm-start Study whose policy `train()` refuses validates as invalid,
    with the errors `cobre.io.validate` reports for the same output directory."""
    import cobre  # noqa: PLC0415
    import cobre.errors  # noqa: PLC0415
    import cobre.io  # noqa: PLC0415
    import cobre.run  # noqa: PLC0415

    out = tmp_path / "out"
    cobre.run.run(
        VALID_CASE, output_dir=str(out), config_overrides=_ONE_ITERATION_NO_SIMULATION
    )
    _restamp_policy_version(out / "policy")
    study = cobre.Study(VALID_CASE, output_dir=str(out), config_overrides=_WARM_START)
    from_validate = cobre.io.validate(VALID_CASE, _WARM_START, output_dir=str(out))

    report = study.validate()

    assert report["valid"] is False, report
    assert report["warnings"] == []
    assert report["errors"] == from_validate["errors"]
    message = report["errors"][0]["message"]
    with pytest.raises(cobre.errors.PolicyIncompatibleError) as exc_info:
        study.train()
    reported = message[message.index("policy was written by") :]
    assert reported in str(exc_info.value), (reported, str(exc_info.value))


def test_study_validate_reports_unused_stored_bases_as_a_warning(
    tmp_path: pathlib.Path,
) -> None:
    """A simulation-only Study over a policy trained with an extra thermal is
    valid and warns about the stored bases that no longer fit, with the warnings
    `cobre.io.validate` reports for the same output directory."""
    import cobre  # noqa: PLC0415
    import cobre.io  # noqa: PLC0415
    import cobre.run  # noqa: PLC0415

    variant = tmp_path / "variant"
    _copy_case_with_extra_thermal(variant)
    out = tmp_path / "out"
    cobre.run.run(
        str(variant), output_dir=str(out), config_overrides=_ONE_ITERATION_NO_SIMULATION
    )
    study = cobre.Study(
        VALID_CASE, output_dir=str(out), config_overrides=_SIMULATION_ONLY
    )
    from_validate = cobre.io.validate(VALID_CASE, _SIMULATION_ONLY, output_dir=str(out))

    report = study.validate()

    assert report["valid"] is True, report
    assert report["errors"] == []
    messages = [w["message"] for w in report["warnings"]]
    assert sum("stored bases not used: " in m for m in messages) == 1, messages
    assert report["warnings"] == from_validate["warnings"]
