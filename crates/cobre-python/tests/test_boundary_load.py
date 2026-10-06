"""Python-parity tests for the date-driven boundary policy load.

`Study.train` and `cobre.run.run` both route a configured `policy.boundary`
through the same `apply_training_policy_mode` boundary block the CLI's
`cobre run` uses, but report it to `stderr` in the binding's own wording
rather than the CLI's aligned summary block. This module is the first
Python-level coverage of that path: a self-boundary load (a case pointed at
its own freshly trained policy) proves the report reaches a Python caller,
and a renamed-hydro load proves the reject surface does too.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_boundary_load.py -v
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil

import pytest

from test_policy_load_validation import (
    _copy_case_with_renamed_hydro,
    _restamp_policy_version,
)

_REPO_ROOT = pathlib.Path(__file__).parents[3]
VALID_CASE = str(_REPO_ROOT / "examples" / "1dtoy")

_UNDATED_PRICED_STATE_DATE = -(2**31)


def _set_boundary_policy(
    case_dir: pathlib.Path,
    source_policy_dir: pathlib.Path,
    *,
    strict: bool | None = None,
) -> None:
    """Point `case_dir`'s `config.json` at `source_policy_dir` as a boundary
    source, merging into whatever `policy` block the case already declares.
    """
    config_path = case_dir / "config.json"
    config = json.loads(config_path.read_text())
    policy = config.get("policy", {})
    boundary: dict[str, object] = {"path": str(source_policy_dir)}
    if strict is not None:
        boundary["strict"] = strict
    policy["boundary"] = boundary
    config["policy"] = policy
    config_path.write_text(json.dumps(config))


def test_boundary_load_reports_source_date_and_reconciliation(
    tmp_path: pathlib.Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """A compatible self-boundary load reports the source, priced date, and
    cut count, then the reconciliation summary, on stderr.
    """
    import cobre.run  # noqa: PLC0415

    source_output = tmp_path / "source"
    cobre.run.run(VALID_CASE, output_dir=str(source_output))
    source_policy_dir = source_output / "policy"
    assert source_policy_dir.exists(), (
        f"expected a policy checkpoint at {source_policy_dir}"
    )

    target_case = tmp_path / "target"
    shutil.copytree(VALID_CASE, target_case)
    _set_boundary_policy(target_case, source_policy_dir)

    capfd.readouterr()  # discard the source run's own stderr
    cobre.run.run(str(target_case), output_dir=str(tmp_path / "target_output"))

    err = capfd.readouterr().err
    assert str(source_policy_dir) in err, f"expected the source path in stderr: {err}"
    assert re.search(r"priced at \d{4}-\d{2}-\d{2}", err), (
        f"expected a 'priced at <date>' clause in stderr: {err}"
    )
    assert re.search(r"boundary cuts:\s*\d+\s*loaded from", err), (
        f"expected the loaded cut count in stderr: {err}"
    )
    assert "boundary reconciliation:" in err, (
        f"expected the unchanged reconciliation summary line in stderr: {err}"
    )
    assert err.index("boundary cuts:") < err.index("boundary reconciliation:"), (
        "the source/date/count line must precede the reconciliation summary"
    )


def test_boundary_load_strict_accepts_a_faithful_self_boundary(
    tmp_path: pathlib.Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """A self-boundary load under `policy.boundary.strict = True` is not a
    superset of itself, so it loads cleanly and still reports the
    reconciliation summary on stderr.
    """
    import cobre.run  # noqa: PLC0415

    source_output = tmp_path / "strict_source"
    cobre.run.run(VALID_CASE, output_dir=str(source_output))
    source_policy_dir = source_output / "policy"
    assert source_policy_dir.exists(), (
        f"expected a policy checkpoint at {source_policy_dir}"
    )

    target_case = tmp_path / "strict_target"
    shutil.copytree(VALID_CASE, target_case)
    _set_boundary_policy(target_case, source_policy_dir, strict=True)

    capfd.readouterr()  # discard the source run's own stderr
    cobre.run.run(str(target_case), output_dir=str(tmp_path / "strict_target_output"))

    err = capfd.readouterr().err
    assert "boundary reconciliation:" in err, (
        f"expected the reconciliation summary line in stderr: {err}"
    )


def test_boundary_load_rejects_a_mismatched_source(tmp_path: pathlib.Path) -> None:
    """A source checkpoint naming a different hydro raises `ValidationError`,
    naming the offending hydro in the message.
    """
    import cobre.errors  # noqa: PLC0415
    import cobre.run  # noqa: PLC0415

    source_output = tmp_path / "source"
    cobre.run.run(VALID_CASE, output_dir=str(source_output))
    source_policy_dir = source_output / "policy"

    mismatched_hydro_id = 77
    target_case = tmp_path / "mismatched_target"
    target_case.mkdir()
    _copy_case_with_renamed_hydro(
        pathlib.Path(VALID_CASE), target_case, new_hydro_id=mismatched_hydro_id
    )
    _set_boundary_policy(target_case, source_policy_dir)

    with pytest.raises(
        cobre.errors.ValidationError, match="boundary cut error"
    ) as exc_info:
        cobre.run.run(str(target_case), output_dir=str(tmp_path / "mismatched_output"))

    assert str(mismatched_hydro_id) in str(exc_info.value), (
        f"expected the reject message to name hydro {mismatched_hydro_id}: "
        f"{exc_info.value}"
    )


def test_boundary_load_rejects_an_undated_checkpoint_with_validation_error(
    tmp_path: pathlib.Path,
) -> None:
    """A boundary source whose every pool carries the undated sentinel is
    refused with `ValidationError` once the run applies its policy mode.
    """
    import cobre  # noqa: PLC0415
    import cobre.errors  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415
    import cobre.run  # noqa: PLC0415

    source_output = tmp_path / "source"
    cobre.run.run(VALID_CASE, output_dir=str(source_output))

    loaded = cobre.results.load_policy(str(source_output))
    for stage in loaded["stage_cuts"]:
        stage["priced_state_date"] = _UNDATED_PRICED_STATE_DATE
    undated_output = tmp_path / "undated"
    cobre.write_policy_checkpoint(
        str(undated_output / "policy"), loaded["stage_cuts"], loaded["metadata"]
    )

    target_case = tmp_path / "target"
    shutil.copytree(VALID_CASE, target_case)
    _set_boundary_policy(target_case, undated_output / "policy")

    with pytest.raises(
        cobre.errors.ValidationError,
        match=(
            "boundary cut error: configuration validation error: "
            "boundary policy checkpoint at .* "
            "carries no priced_state_date on any pool"
        ),
    ):
        cobre.run.run(str(target_case), output_dir=str(tmp_path / "target_output"))


def test_boundary_load_rejects_a_source_written_by_another_version(
    tmp_path: pathlib.Path,
) -> None:
    """A self-boundary source restamped to record another cobre version is
    refused with `PolicyIncompatibleError`, naming both versions (the
    same-version self-boundary is
    `test_boundary_load_reports_source_date_and_reconciliation`).
    """
    import cobre  # noqa: PLC0415
    import cobre.errors  # noqa: PLC0415
    import cobre.run  # noqa: PLC0415

    source_output = tmp_path / "source"
    cobre.run.run(VALID_CASE, output_dir=str(source_output))
    source_policy_dir = source_output / "policy"

    other_version = _restamp_policy_version(source_policy_dir)

    target_case = tmp_path / "target"
    shutil.copytree(VALID_CASE, target_case)
    _set_boundary_policy(target_case, source_policy_dir)

    with pytest.raises(
        cobre.errors.PolicyIncompatibleError,
        match=f"written by cobre {other_version}",
    ) as exc_info:
        cobre.run.run(str(target_case), output_dir=str(tmp_path / "target_output"))

    assert cobre.__version__ in str(exc_info.value), (
        f"expected the running version in the message: {exc_info.value}"
    )
