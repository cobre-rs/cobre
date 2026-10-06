"""Tests for the cobre.io.validate full pre-solver pipeline.

Verifies that validate() exercises the full phase sequence (path check,
cobre-io validation layers, SDDP preparation phases, and boundary
reconciliation) and returns a correctly shaped result dict.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_validate.py -v
"""

from __future__ import annotations

import json
import pathlib
import shutil
import tempfile

import pytest

VALID_CASE_1DTOY = "examples/1dtoy"
VALID_CASE_4REE = "examples/4ree"


# ── helper ────────────────────────────────────────────────────────────────────


def copy_case_to_tempdir(src: str) -> pathlib.Path:
    """Copy a case directory to a fresh temp dir and return the new path."""
    tmp = pathlib.Path(tempfile.mkdtemp())
    dest = tmp / pathlib.Path(src).name
    shutil.copytree(src, dest)
    return dest


# ── result shape contract ─────────────────────────────────────────────────────


def test_validate_result_has_required_keys() -> None:
    """validate() always returns a dict with valid, errors, and warnings keys."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(VALID_CASE_1DTOY)
    assert isinstance(result, dict)
    assert "valid" in result
    assert "errors" in result
    assert "warnings" in result


def test_validate_warning_entries_have_required_fields() -> None:
    """Each warning entry has kind, message, file, and entity fields."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(VALID_CASE_1DTOY)
    for w in result["warnings"]:
        assert "kind" in w, f"missing 'kind' in warning: {w}"
        assert "message" in w, f"missing 'message' in warning: {w}"
        assert "file" in w, f"missing 'file' in warning: {w}"
        assert "entity" in w, f"missing 'entity' in warning: {w}"


# ── clean-case tests ──────────────────────────────────────────────────────────


def test_validate_clean_case_returns_valid_true() -> None:
    """validate(examples/1dtoy) returns valid=True with no errors."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(VALID_CASE_1DTOY)
    assert result["valid"] is True, (
        f"expected valid=True, got errors: {result['errors']}"
    )
    assert result["errors"] == []


def test_validate_emits_penalty_ordering_warning() -> None:
    """validate() surfaces a penalty-ordering warning end-to-end.

    Rather than relying on a shipped example happening to violate the penalty
    hierarchy (which is curated to be well-formed), this constructs a case that
    deliberately re-inverts the ordering: the bus deficit-segment cost is set
    BELOW ``generation_violation_below_cost`` (1000 in penalties.json). That
    trips the same-unit check of the deficit costs against
    ``generation_violation_below_cost`` (both $/MWh), which emits a
    ``ModelQuality`` penalty-ordering warning.

    This pins the warnings plumbing (semantic layer → ReportEntry → Python dict)
    end-to-end and is robust to future curation of the shipped example. A
    warning must NOT invalidate the case, so ``valid`` stays ``True``.
    """
    import cobre.io  # noqa: PLC0415

    case_dir = copy_case_to_tempdir(VALID_CASE_1DTOY)
    try:
        # Re-invert the penalty hierarchy: drive the bus deficit-segment cost
        # below generation_violation_below_cost (1000 in penalties.json), the
        # same-unit ($/MWh) comparand. 1dtoy carries an entity-level deficit
        # segment on its single bus, so system/buses.json is the resolved cost
        # source.
        buses_path = case_dir / "system" / "buses.json"
        with buses_path.open() as f:
            buses = json.load(f)
        for bus in buses["buses"]:
            for segment in bus["deficit_segments"]:
                segment["cost"] = 500.0
        with buses_path.open("w") as f:
            json.dump(buses, f, indent=2)

        result = cobre.io.validate(str(case_dir))

        # Warnings never invalidate a case.
        assert result["valid"] is True, (
            f"warnings must not invalidate; got errors: {result['errors']}"
        )
        assert isinstance(result["warnings"], list)
        assert len(result["warnings"]) >= 1, (
            "expected at least one warning after re-inverting the penalty "
            "hierarchy, got zero — warnings are being dropped"
        )

        penalty_warnings = [
            w
            for w in result["warnings"]
            if w["kind"] == "ModelQuality"
            and (
                "penalty" in w["message"].lower() or "ordering" in w["message"].lower()
            )
        ]
        assert penalty_warnings, (
            "expected a ModelQuality penalty-ordering warning, got: "
            f"{result['warnings']}"
        )
    finally:
        shutil.rmtree(case_dir.parent, ignore_errors=True)


def test_validate_clean_case_accepts_pathlib_path() -> None:
    """validate() accepts a pathlib.Path, not just a str."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(pathlib.Path(VALID_CASE_1DTOY))
    assert result["valid"] is True


def test_validate_4ree_warnings_populated() -> None:
    """validate(examples/4ree) also populates warnings from the pipeline."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(VALID_CASE_4REE)
    assert result["valid"] is True, f"4ree should be valid, errors: {result['errors']}"
    # 4ree may emit zero warnings — we only assert the key is present and a list.
    assert isinstance(result["warnings"], list)


# ── missing-directory test ────────────────────────────────────────────────────


def test_validate_missing_directory_returns_invalid() -> None:
    """validate() returns valid=False for a non-existent path without raising."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate("/tmp/nonexistent_cobre_case_abc999xyz")
    assert result["valid"] is False
    assert len(result["errors"]) >= 1
    err = result["errors"][0]
    assert "kind" in err
    assert "message" in err


# ── Phase 8: StudyParams::from_config (ConfigValidationError) ─────────────────


def test_validate_basis_activity_window_is_rejected() -> None:
    """basis_activity_window was removed in the cut_selection restructure.

    cut_selection (RowSelectionConfig) uses deny_unknown_fields, so a config
    that still carries the removed basis_activity_window field fails to load.
    validate() must surface this as valid=False and name the offending field,
    rather than silently ignoring it.
    """
    import cobre.io  # noqa: PLC0415

    case_dir = copy_case_to_tempdir(VALID_CASE_1DTOY)
    try:
        config_path = case_dir / "config.json"
        with config_path.open() as f:
            config = json.load(f)

        config.setdefault("training", {}).setdefault("cut_selection", {})[
            "basis_activity_window"
        ] = 100

        with config_path.open("w") as f:
            json.dump(config, f, indent=2)

        result = cobre.io.validate(str(case_dir))
        assert result["valid"] is False, (
            "expected valid=False for the removed basis_activity_window field, "
            f"got: {result!r}"
        )
        assert any(
            "basis_activity_window" in err["message"] for err in result["errors"]
        ), f"an error must name the removed field, got: {result['errors']!r}"
    finally:
        shutil.rmtree(case_dir.parent, ignore_errors=True)


# ── Phase 1-6: cobre-io pipeline (ConstraintError / IoError) ──────────────────


def test_validate_missing_config_returns_error() -> None:
    """An empty directory (no config.json) returns valid=False with an error."""
    import cobre.io  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        result = cobre.io.validate(tmp)
        assert result["valid"] is False
        assert len(result["errors"]) >= 1
        err = result["errors"][0]
        # The cobre-io structural layer reports missing required files as
        # ConstraintError (aggregated) or IoError (direct read failure).
        assert err["kind"] in (
            "ConstraintError",
            "IoError",
            "ParseError",
            "SchemaError",
        ), f"unexpected kind for missing-config case: {err['kind']!r}"


def test_validate_never_raises_for_missing_case() -> None:
    """validate() must not raise any exception, even for bad inputs."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate("/dev/null/this/cannot/exist")
    assert isinstance(result, dict)
    assert result["valid"] is False


def test_validate_never_raises_for_valid_case() -> None:
    """validate() must not raise any exception for a valid case."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(VALID_CASE_1DTOY)
    assert isinstance(result, dict)


# ── config_overrides ──────────────────────────────────────────────────────────


def test_validate_config_overrides_none_is_valid() -> None:
    """config_overrides=None reproduces the default valid result."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(VALID_CASE_1DTOY, config_overrides=None)
    assert result["valid"] is True, f"errors: {result['errors']}"


def test_validate_config_overrides_invalid_value_surfaces_schema_error() -> None:
    """A semantically-invalid override surfaces in the dict as a SchemaError, not a raise.

    Setting `historical_years` while no scenario class uses the `historical`
    scheme violates validate_config, which `Config::with_overrides` runs
    identically to an edited config.json. 1dtoy uses `in_sample` for every
    class, so the override is rejected with the offending field named.
    """
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(
        VALID_CASE_1DTOY,
        config_overrides={"training.scenario_source.historical_years": [1990, 1991]},
    )
    assert result["valid"] is False, "an invalid override must mark the case invalid"
    assert len(result["errors"]) >= 1
    err = result["errors"][0]
    assert err["kind"] == "SchemaError", f"expected SchemaError, got {err['kind']!r}"
    assert "historical_years" in err["message"], (
        f"error message must name the offending field, got: {err['message']!r}"
    )


def test_validate_config_overrides_typo_surfaces_schema_error() -> None:
    """A typo override key is rejected via deny_unknown_fields as a SchemaError."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(
        VALID_CASE_1DTOY,
        config_overrides={"trainning.tree_seed": 7},
    )
    assert result["valid"] is False
    assert len(result["errors"]) >= 1
    assert result["errors"][0]["kind"] == "SchemaError"


def test_validate_config_overrides_unsupported_value_raises_value_error() -> None:
    """A malformed override value (no JSON form) raises ValueError before merging.

    Unlike case-validation failures (returned as data), a malformed call payload
    is a programming error and is raised under the GIL before py.detach.
    """
    import cobre.io  # noqa: PLC0415

    with pytest.raises(ValueError):
        cobre.io.validate(
            VALID_CASE_1DTOY,
            config_overrides={"training.cut_selection.row_activity_tolerance": {1, 2}},
        )


# ── Phase 11: boundary reconciliation (BoundaryReconciliationError) ───────────


def test_validate_missing_boundary_checkpoint_returns_invalid() -> None:
    """A case pointing to a non-existent boundary checkpoint returns valid=False
    with a BoundaryReconciliationError naming manifest.bin.

    Pre-fix: this same call returned {"valid": True, "errors": []}, diverging from
    the CLI validate which exited 1 for the identical case.
    """
    import cobre.io  # noqa: PLC0415

    case_dir = copy_case_to_tempdir(VALID_CASE_1DTOY)
    try:
        config_path = case_dir / "config.json"
        with config_path.open() as f:
            config = json.load(f)

        policy = config.get("policy", {})
        policy["boundary"] = {"path": str(case_dir / "no_such_policy")}
        config["policy"] = policy

        with config_path.open("w") as f:
            json.dump(config, f)

        result = cobre.io.validate(str(case_dir))
        assert result["valid"] is False, (
            f"expected valid=False for missing boundary checkpoint, got: {result!r}"
        )
        assert len(result["errors"]) == 1
        err = result["errors"][0]
        assert err["kind"] == "BoundaryReconciliationError"
        assert err["message"].startswith("policy.boundary: ")
        assert "manifest.bin" in err["message"]
    finally:
        shutil.rmtree(case_dir.parent, ignore_errors=True)


def test_validate_without_boundary_policy_is_unchanged() -> None:
    """Validate with no boundary policy skips phase 11 and returns the same
    result as before the phase 11 addition.
    """
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(VALID_CASE_1DTOY)
    assert result["valid"] is True
    assert result["errors"] == []
    assert isinstance(result["warnings"], list)


# ── Phase 11: scalar-parameter table gap classes ──────────────────────────────
#
# `StudySetup::new_with_boundary_requirements` now receives the caller-loaded
# `constraints/generic_parameters.json` table (never an empty placeholder), so
# a boundary-configured study whose table has a genuine gap is rejected here
# exactly as the CLI's `cobre validate` rejects it.


def _build_case_with_boundary_and_scalar_parameter(
    tmp_path: pathlib.Path, entry: dict[str, object]
) -> pathlib.Path:
    """Copy `examples/1dtoy`, point its boundary at a freshly trained
    self-checkpoint, and add a single scalar-parameter `entry` to
    `constraints/generic_parameters.json`.
    """
    import cobre.run  # noqa: PLC0415

    source_output = tmp_path / "source"
    cobre.run.run(VALID_CASE_1DTOY, output_dir=str(source_output))
    source_policy_dir = source_output / "policy"

    target_case = tmp_path / "target"
    shutil.copytree(VALID_CASE_1DTOY, target_case)

    config_path = target_case / "config.json"
    config = json.loads(config_path.read_text())
    policy = config.get("policy", {})
    policy["boundary"] = {"path": str(source_policy_dir)}
    config["policy"] = policy
    config_path.write_text(json.dumps(config))

    constraints_dir = target_case / "constraints"
    constraints_dir.mkdir(exist_ok=True)
    (constraints_dir / "generic_parameters.json").write_text(
        json.dumps({"scalar_parameters": [entry]})
    )
    return target_case


def test_validate_rejects_missing_season_scalar_parameter_gap(
    tmp_path: pathlib.Path,
) -> None:
    """A `seasonal` scalar parameter with no entry for a stage's resolved
    season (1dtoy's stages carry no `season_id`, so every stage resolves to
    season 0) is rejected as `BoundaryReconciliationError`.
    """
    import cobre.io  # noqa: PLC0415

    target_case = _build_case_with_boundary_and_scalar_parameter(
        tmp_path,
        {"id": 1, "name": "p_season_gap", "kind": "seasonal", "values": [[5, 1.0]]},
    )

    result = cobre.io.validate(str(target_case))
    assert result["valid"] is False, f"expected a MissingSeason reject, got: {result!r}"
    assert any("season" in err["message"] for err in result["errors"]), (
        f"expected an error naming the missing season, got: {result['errors']!r}"
    )
    assert result["errors"][0]["kind"] == "BoundaryReconciliationError"


def test_validate_rejects_per_stage_block_coverage_gap(
    tmp_path: pathlib.Path,
) -> None:
    """A `per_stage_block` scalar parameter covering only stage 0's block
    leaves every later stage's `(stage, block)` cell uncovered and is
    rejected. (cobre-io's parser requires at least one entry, so the gap must
    be a partial rather than an empty `block_values`.)
    """
    import cobre.io  # noqa: PLC0415

    target_case = _build_case_with_boundary_and_scalar_parameter(
        tmp_path,
        {
            "id": 2,
            "name": "p_block_gap",
            "kind": "per_stage_block",
            "block_values": [[0, 0, 1.0]],
        },
    )

    result = cobre.io.validate(str(target_case))
    assert result["valid"] is False, (
        f"expected a PerStageBlockCoverage reject, got: {result!r}"
    )
    assert any("not covered" in err["message"] for err in result["errors"]), (
        f"expected an error naming the uncovered cell, got: {result['errors']!r}"
    )


def test_validate_rejects_missing_specific_productivity(
    tmp_path: pathlib.Path,
) -> None:
    """A `computed`/`specific_productivity` scalar parameter for a hydro with
    no productivity override and no entity-level value is rejected. 1dtoy's
    hydro 0 declares neither.
    """
    import cobre.io  # noqa: PLC0415

    target_case = _build_case_with_boundary_and_scalar_parameter(
        tmp_path,
        {
            "id": 3,
            "name": "p_rho_esp_gap",
            "kind": "computed",
            "computed_spec": {"tag": "specific_productivity", "hydro_id": 0},
        },
    )

    result = cobre.io.validate(str(target_case))
    assert result["valid"] is False, (
        f"expected a MissingSpecificProductivity reject, got: {result!r}"
    )
    assert any("specific productivity" in err["message"] for err in result["errors"]), (
        f"expected an error naming the missing productivity, got: {result['errors']!r}"
    )


# ── Non-boundary scalar-parameter presence guard ──────────────────────────────
#
# Validate runs the scalar-parameter guard before it builds the study, so a gap on
# a deck with no boundary policy keeps the error kind
# GenericConstraintValidationError and the same message as `cobre validate`.


def _write_scalar_parameters(
    case_dir: pathlib.Path, scalar_parameters: list[dict[str, object]]
) -> None:
    """Write `constraints/generic_parameters.json` into an existing case dir."""
    constraints_dir = case_dir / "constraints"
    constraints_dir.mkdir(exist_ok=True)
    (constraints_dir / "generic_parameters.json").write_text(
        json.dumps({"scalar_parameters": scalar_parameters})
    )


def test_validate_rejects_non_boundary_scalar_parameter_gap() -> None:
    """A non-boundary deck whose scalar-parameter table has a resolution gap (a
    `seasonal` param with no entry for the resolved season; 1dtoy resolves every
    stage to season 0) is rejected as GenericConstraintValidationError, closing
    the gap that previously let it pass validate while `cobre run` failed.
    """
    import cobre.io  # noqa: PLC0415

    case_dir = copy_case_to_tempdir(VALID_CASE_1DTOY)
    try:
        _write_scalar_parameters(
            case_dir,
            [{"id": 1, "name": "p_gap", "kind": "seasonal", "values": [[5, 1.0]]}],
        )
        result = cobre.io.validate(str(case_dir))
        assert result["valid"] is False, f"expected a reject, got: {result!r}"
        assert len(result["errors"]) == 1
        err = result["errors"][0]
        assert err["kind"] == "GenericConstraintValidationError", (
            f"kind must match `cobre validate`, got: {err['kind']!r}"
        )
        assert err["message"] == (
            "constraints/: configuration validation error: parameter 'p_gap': "
            "no seasonal value for season_id=0 (needed by stage 0)"
        ), f"message must match `cobre validate` byte-for-byte, got: {err['message']!r}"
    finally:
        shutil.rmtree(case_dir.parent, ignore_errors=True)


def test_validate_accepts_non_boundary_resolved_scalar_parameter() -> None:
    """A non-boundary deck whose scalar-parameter table resolves cleanly still
    validates with zero errors (the new guard raises no false rejection)."""
    import cobre.io  # noqa: PLC0415

    case_dir = copy_case_to_tempdir(VALID_CASE_1DTOY)
    try:
        _write_scalar_parameters(
            case_dir,
            [{"id": 1, "name": "p_ok", "kind": "constant", "value": 1.0}],
        )
        result = cobre.io.validate(str(case_dir))
        assert result["valid"] is True, f"expected valid, got: {result!r}"
        assert result["errors"] == []
    finally:
        shutil.rmtree(case_dir.parent, ignore_errors=True)


# ── output_dir: the directory a configured policy is read from ────────────────

WARM_START = {"policy.mode": "warm_start"}


def _train_one_iteration(case: pathlib.Path, out: pathlib.Path) -> None:
    """Train `case` for one iteration into `out`, leaving a policy in `out/policy`."""
    import cobre.run  # noqa: PLC0415

    cobre.run.run(
        str(case),
        output_dir=str(out),
        config_overrides={
            "training.stopping_rules": [{"type": "iteration_limit", "limit": 1}],
            "simulation.enabled": False,
        },
    )


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


def _case_with_restamped_policy(
    tmp_path: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    """A copy of 1dtoy and a sibling output directory holding its restamped policy."""
    case = tmp_path / "case"
    shutil.copytree(VALID_CASE_1DTOY, case)
    out = tmp_path / "out"
    _train_one_iteration(case, out)
    _restamp_policy_version(out / "policy")
    return case, out


def test_validate_output_dir_checks_the_policy_run_loads_from_that_directory(
    tmp_path: pathlib.Path,
) -> None:
    """validate(output_dir=out) refuses the warm-start policy in `out` that
    run(output_dir=out) refuses, with the same message."""
    import cobre.errors  # noqa: PLC0415
    import cobre.io  # noqa: PLC0415
    import cobre.run  # noqa: PLC0415

    case, out = _case_with_restamped_policy(tmp_path)

    result = cobre.io.validate(str(case), WARM_START, output_dir=str(out))

    assert result["valid"] is False, result
    assert [e["kind"] for e in result["errors"]] == ["WarmStartIncompatible"]
    message = result["errors"][0]["message"]
    assert "policy was written by" in message, message
    with pytest.raises(cobre.errors.PolicyIncompatibleError) as exc_info:
        cobre.run.run(str(case), output_dir=str(out), config_overrides=WARM_START)
    reported = message[message.index("policy was written by") :]
    assert reported in str(exc_info.value), (reported, str(exc_info.value))


def test_validate_without_output_dir_reads_the_case_output_subdirectory(
    tmp_path: pathlib.Path,
) -> None:
    """Without output_dir, validate reads `<case>/output`, where the policy
    trained into a sibling directory is absent."""
    import cobre.io  # noqa: PLC0415

    case, _ = _case_with_restamped_policy(tmp_path)

    result = cobre.io.validate(str(case), WARM_START)

    assert result["valid"] is False, result
    message = result["errors"][0]["message"]
    assert "Policy directory not found" in message, message
    assert str(case / "output") in message, message


def test_validate_output_dir_is_never_created(tmp_path: pathlib.Path) -> None:
    """validate leaves an absent output_dir absent."""
    import cobre.io  # noqa: PLC0415

    absent = tmp_path / "absent"

    result = cobre.io.validate(VALID_CASE_1DTOY, WARM_START, output_dir=str(absent))

    assert result["valid"] is False, result
    assert not absent.exists()


def test_validate_relative_output_dir_resolves_against_the_working_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative output_dir is read from the process working directory."""
    import cobre.io  # noqa: PLC0415

    case, _ = _case_with_restamped_policy(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = cobre.io.validate(str(case), WARM_START, output_dir="out")

    assert result["valid"] is False, result
    assert "policy was written by" in result["errors"][0]["message"], result


def test_validate_output_dir_is_keyword_only() -> None:
    """output_dir cannot be passed positionally."""
    import cobre.io  # noqa: PLC0415

    with pytest.raises(TypeError):
        cobre.io.validate(VALID_CASE_1DTOY, None, "out")  # type: ignore[call-arg]
