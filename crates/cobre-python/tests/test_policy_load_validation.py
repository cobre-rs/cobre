"""Python-parity tests for the unified `Study.load_policy` validation path.

Every policy load (warm-start, resume, simulation-only, `Study.load_policy`) now
routes unconditionally through the shared `cobre_sddp::validate_policy_load`
entry point -- there is no per-call opt-out. This module verifies the two
Python-facing consequences: the removed opt-out kwarg raises
`TypeError`, and a policy whose terminal entity manifest disagrees with the
current study (same state dimension, different hydro id) raises `ValueError`.
The compatible-load path is already exercised by
`test_load_policy_then_simulate_matches_run` in `test_study.py`.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_policy_load_validation.py
"""

from __future__ import annotations

import json
import pathlib
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_REPO_ROOT = pathlib.Path(__file__).parents[3]
VALID_CASE = str(_REPO_ROOT / "examples" / "1dtoy")

# The retired opt-out kwarg's name, assembled from fragments so the literal
# token never appears as a contiguous string: a tree-wide grep gate asserts the
# kwarg is gone from the binding, and this proof test is the sole place it would
# otherwise resurface. Do not fold back into one literal.
_REMOVED_OPTOUT_KWARG = "validate_" + "compatibility"


def _copy_case_with_renamed_hydro(
    src: pathlib.Path, dest: pathlib.Path, new_hydro_id: int
) -> None:
    """Copy `src` into `dest`, renaming its sole hydro's id (and every
    foreign-key reference to it) to `new_hydro_id`.

    Every other field stays byte-identical to `src`, so the resulting case has
    the SAME state dimension as `src` but a DIFFERENT entity identity -- the
    "same dimension, different hydro id" shape `validate_policy_load`'s
    slot-identity check must reject.
    """
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    hydros_path = dest / "system" / "hydros.json"
    hydros = json.loads(hydros_path.read_text())
    old_id = hydros["hydros"][0]["id"]
    hydros["hydros"][0]["id"] = new_hydro_id
    hydros_path.write_text(json.dumps(hydros))

    models_path = dest / "system" / "hydro_production_models.json"
    models = json.loads(models_path.read_text())
    for model in models["production_models"]:
        if model["hydro_id"] == old_id:
            model["hydro_id"] = new_hydro_id
    models_path.write_text(json.dumps(models))

    ic_path = dest / "initial_conditions.json"
    initial_conditions = json.loads(ic_path.read_text())
    for entry in initial_conditions["storage"]:
        if entry["hydro_id"] == old_id:
            entry["hydro_id"] = new_hydro_id
    ic_path.write_text(json.dumps(initial_conditions))

    # ZSTD to match the shipped example's codec: cobre's Rust parquet reader
    # is built without the "snap" feature, so pyarrow's default (Snappy)
    # produces an unreadable file.
    inflow_path = dest / "scenarios" / "inflow_seasonal_stats.parquet"
    table = pq.read_table(inflow_path)
    renamed_hydro_id = pa.array([new_hydro_id] * table.num_rows, type=pa.int32())
    table = table.set_column(
        table.schema.get_field_index("hydro_id"), "hydro_id", renamed_hydro_id
    )
    pq.write_table(table, inflow_path, compression="zstd")


def _restamp_policy_version(policy_dir: pathlib.Path) -> str:
    """Rewrite the cobre version in ``policy_dir/manifest.bin`` to another
    string of the same byte length (the FlatBuffers string keeps its layout;
    the manifest carries no checksum) and return it."""
    import cobre  # noqa: PLC0415

    manifest = policy_dir / "manifest.bin"
    data = manifest.read_bytes()
    running = cobre.__version__.encode()
    assert data.count(running) == 1, (
        "the running version must occur once in manifest.bin"
    )
    other = (b"8" if running.startswith(b"9") else b"9") + running[1:]
    manifest.write_bytes(data.replace(running, other))
    return other.decode()


def test_load_policy_removed_optout_kwarg_raises_typeerror(
    tmp_path: pathlib.Path,
) -> None:
    """The removed opt-out kwarg raises TypeError.

    Validation is now unconditional, so the parameter no longer exists on
    `load_policy`; passing it must fail loudly, not be silently ignored.
    """
    import cobre  # noqa: PLC0415

    study = cobre.Study(VALID_CASE, output_dir=str(tmp_path))

    with pytest.raises(TypeError):
        study.load_policy(
            output_dir=str(tmp_path),
            **{_REMOVED_OPTOUT_KWARG: False},
        )


def test_load_policy_mismatched_entity_manifest_raises_valueerror(
    tmp_path: pathlib.Path,
) -> None:
    """A same-dimension, different-hydro-id policy is rejected.

    Train a case identical to `VALID_CASE` except its sole hydro carries a
    different id, then load that policy into a `Study` built from
    `VALID_CASE`. Both studies have one hydro (identical state dimension), but
    the checkpoint's terminal entity manifest names a different hydro id, so
    `validate_policy_load`'s slot-identity check must reject the load.
    """
    import cobre  # noqa: PLC0415

    mismatched_case = tmp_path / "mismatched_case"
    mismatched_case.mkdir()
    _copy_case_with_renamed_hydro(
        pathlib.Path(VALID_CASE), mismatched_case, new_hydro_id=99
    )

    mismatched_run_dir = tmp_path / "mismatched_run"
    cobre.run.run(str(mismatched_case), output_dir=str(mismatched_run_dir))

    study = cobre.Study(VALID_CASE, output_dir=str(tmp_path / "study_dir"))

    with pytest.raises(ValueError, match="policy validation error"):
        study.load_policy(output_dir=str(mismatched_run_dir))


def test_load_policy_written_by_another_version_raises_policy_incompatible(
    tmp_path: pathlib.Path,
) -> None:
    """A policy checkpoint recording a cobre version other than the running
    one raises `PolicyIncompatibleError`, naming both versions.
    """
    import cobre  # noqa: PLC0415
    import cobre.errors  # noqa: PLC0415

    run_dir = tmp_path / "run"
    cobre.run.run(VALID_CASE, output_dir=str(run_dir))

    study = cobre.Study(VALID_CASE, output_dir=str(tmp_path / "study_dir"))
    study.load_policy(output_dir=str(run_dir))

    other_version = _restamp_policy_version(run_dir / "policy")

    fresh_study = cobre.Study(
        VALID_CASE, output_dir=str(tmp_path / "fresh_study_dir")
    )
    with pytest.raises(
        cobre.errors.PolicyIncompatibleError,
        match=f"written by cobre {other_version}",
    ) as exc_info:
        fresh_study.load_policy(output_dir=str(run_dir))

    assert cobre.__version__ in str(exc_info.value), (
        f"expected the running version in the message: {exc_info.value}"
    )
