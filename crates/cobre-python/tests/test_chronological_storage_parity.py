"""Python-parity guard for chronological per-block storage and evaporation.

Chronological block mode evolves reservoir storage block-by-block within a stage
(S0 -> S1 -> ... -> SK), so each block row of the ``hydros`` simulation output
carries that block's own storage boundaries and its own storage-dependent
evaporation, instead of the single stage pair repeated across every block. The
per-block values are populated during shared simulation extraction, not in the
output writer; both the CLI (`write_simulation_outputs`) and all Python surfaces
(`run_via_study`, `Study.simulate`) converge on the same
`cobre_io::write_simulation_results` path and neither re-reads storage from the
LP. The Python-parity hard rule (`CLAUDE.md`) requires the per-block values to
reach the Python-written parquet identically to the CLI; this module pins that
end-to-end.

The durable contracts asserted here:

- Chronological: within a `(hydro_id, stage_id)` group spanning multiple blocks,
  `storage_initial_hm3` / `storage_final_hm3` are NOT all-equal across block rows
  (per-block storage survived the writer), and the storage-driven
  `evaporation_m3s` is NOT all-equal across those block rows.
- Parallel (the symmetric negative): within each such group, storage AND
  evaporation are identical across block rows, so a regression that leaked
  per-block resolution into parallel mode fails loudly.
- The `hydros` parquet schema is unchanged: its column-name set equals exactly
  the known field set (owned by `hydros_schema()` in cobre-io's simulation
  writer), so per-block resolution added no column.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_chronological_storage_parity.py -v
"""

from __future__ import annotations

import math
import pathlib
import tempfile

import pyarrow.parquet as pq

from _cobre_cli import run_cli

# Fixtures live under the cobre-sddp test tree (the convention topology fixtures
# use), resolved against the repo root so the test is independent of pytest's
# working directory. Both are single-reservoir, single-hydro, three-block,
# two-stage studies with an evaporating hydro and asymmetric per-block load
# factors; they differ only in stages.json `block_mode`.
_REPO_ROOT = pathlib.Path(__file__).parents[3]
_FIXTURES = _REPO_ROOT / "crates" / "cobre-sddp" / "tests" / "fixtures"
CHRONOLOGICAL_CASE = _FIXTURES / "chronological_storage"
PARALLEL_CASE = _FIXTURES / "parallel_storage"

# The exact field set of the `hydros` output schema, owned by `hydros_schema()`
# in cobre-io's simulation writer. Per-block storage/evaporation resolution reuses
# the existing columns with block-resolved values; the node axis adds the
# `node_id` and `scenario_id` axis columns on every entity row. A mismatch here
# (added/removed/renamed column beyond these) is a regression, not expected.
HYDROS_SCHEMA_FIELDS = {
    "scenario_id",
    "node_id",
    "stage_id",
    "block_id",
    "hydro_id",
    "turbined_m3s",
    "spillage_m3s",
    "outflow_m3s",
    "evaporation_m3s",
    "diverted_inflow_m3s",
    "diverted_outflow_m3s",
    "incremental_inflow_m3s",
    "inflow_m3s",
    "storage_initial_hm3",
    "storage_final_hm3",
    "generation_mw",
    "generation_mwh",
    "equivalent_productivity_mw_per_m3s",
    "accumulated_productivity_mw_per_m3s",
    "incremental_inflow_energy_mw",
    "stored_energy_initial_mwh",
    "stored_energy_final_mwh",
    "spillage_cost",
    "water_value_per_hm3",
    "storage_binding_code",
    "operative_state_code",
    "turbined_slack_m3s",
    "outflow_slack_below_m3s",
    "outflow_slack_above_m3s",
    "generation_slack_mw",
    "storage_violation_below_hm3",
    "filling_target_violation_hm3",
    "evaporation_violation_pos_m3s",
    "evaporation_violation_neg_m3s",
    "inflow_nonnegativity_slack_m3s",
    "water_withdrawal_violation_pos_m3s",
    "water_withdrawal_violation_neg_m3s",
    "integrated_equivalent_productivity_mw_per_m3s",
    "integrated_accumulated_productivity_mw_per_m3s",
    "stored_energy_initial_mw",
    "stored_energy_final_mw",
}


def _hydros_parquets(output_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return the hydros data.parquet partitions under an output dir."""
    hydros_dir = output_dir / "simulation" / "hydros"
    if not hydros_dir.is_dir():
        return []
    return sorted(hydros_dir.rglob("*.parquet"))


def _read_hydro_rows(output_dir: pathlib.Path) -> list[dict[str, object]]:
    """Read every hydros partition into a flat list of row dicts."""
    rows: list[dict[str, object]] = []
    for parquet in _hydros_parquets(output_dir):
        # ParquetFile().read() reads the file's own columns only; pq.read_table()
        # would add a dictionary<int32> `scenario_id` partition column from the
        # `scenario_id=NNNN/` Hive path that collides with the in-file int32 column.
        rows.extend(pq.ParquetFile(parquet).read().to_pylist())
    return rows


def _group_by_hydro_stage(
    rows: list[dict[str, object]],
) -> dict[tuple[int, int], list[dict[str, object]]]:
    """Group hydro rows by ``(hydro_id, stage_id)``, preserving block order."""
    groups: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["hydro_id"]), int(row["stage_id"]))
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        group.sort(key=lambda r: (r["block_id"] is None, r["block_id"]))
    return groups


def _multi_block_groups(
    groups: dict[tuple[int, int], list[dict[str, object]]],
) -> dict[tuple[int, int], list[dict[str, object]]]:
    """Keep only groups that span more than one block row."""
    return {key: rows for key, rows in groups.items() if len(rows) > 1}


def _all_close(values: list[float]) -> bool:
    """True when every value equals the first within floating-point tolerance."""
    if not values:
        return True
    first = values[0]
    return all(math.isclose(v, first, rel_tol=1e-9, abs_tol=1e-9) for v in values)


def _run_chronological(output_dir: pathlib.Path) -> list[dict[str, object]]:
    """Train + simulate the chronological fixture via cobre.run.run; read hydros."""
    import cobre.run  # noqa: PLC0415

    cobre.run.run(str(CHRONOLOGICAL_CASE), output_dir=str(output_dir))
    return _read_hydro_rows(output_dir)


def test_chronological_per_block_storage_varies_across_blocks() -> None:
    """Chronological run: storage boundaries differ across a stage's block rows.

    For at least one `(hydro_id, stage_id)` group with multiple block rows, both
    `storage_initial_hm3` and `storage_final_hm3` take more than one distinct
    value across the block rows — the per-block boundaries `(Sb, Sb+1)` survived
    the shared writer rather than collapsing to a single repeated stage pair.
    """
    assert CHRONOLOGICAL_CASE.is_dir(), (
        f"the chronological fixture must exist at {CHRONOLOGICAL_CASE}"
    )

    with tempfile.TemporaryDirectory() as out_dir:
        rows = _run_chronological(pathlib.Path(out_dir))

    groups = _multi_block_groups(_group_by_hydro_stage(rows))
    assert groups, "the chronological fixture must emit multi-block hydro groups"

    varied_initial = [
        key
        for key, grp in groups.items()
        if not _all_close([float(r["storage_initial_hm3"]) for r in grp])
    ]
    varied_final = [
        key
        for key, grp in groups.items()
        if not _all_close([float(r["storage_final_hm3"]) for r in grp])
    ]
    assert varied_initial, (
        "chronological storage_initial_hm3 must vary across block rows for at "
        "least one (hydro_id, stage_id) group; all groups were block-constant, "
        "so per-block storage did not survive the writer"
    )
    assert varied_final, (
        "chronological storage_final_hm3 must vary across block rows for at "
        "least one (hydro_id, stage_id) group"
    )


def test_chronological_per_block_evaporation_varies_across_blocks() -> None:
    """Chronological run: evaporation differs across a stage's block rows.

    Evaporation is linear in each block's local average storage, so per-block
    storage drawdown/refill drives a per-block `evaporation_m3s`. For at least one
    multi-block `(hydro_id, stage_id)` group the value is not block-constant.
    """
    assert CHRONOLOGICAL_CASE.is_dir(), (
        f"the chronological fixture must exist at {CHRONOLOGICAL_CASE}"
    )

    with tempfile.TemporaryDirectory() as out_dir:
        rows = _run_chronological(pathlib.Path(out_dir))

    groups = _multi_block_groups(_group_by_hydro_stage(rows))
    assert groups, "the chronological fixture must emit multi-block hydro groups"

    varied_evap = [
        key
        for key, grp in groups.items()
        if all(r["evaporation_m3s"] is not None for r in grp)
        and not _all_close([float(r["evaporation_m3s"]) for r in grp])
    ]
    assert varied_evap, (
        "chronological evaporation_m3s must vary across block rows for at least "
        "one (hydro_id, stage_id) group; a block-constant trajectory means "
        "per-block evaporation did not survive the writer"
    )


def test_chronological_hydros_schema_is_unchanged() -> None:
    """Chronological run: the hydros parquet column set equals the known field set.

    Per-block resolution reuses the existing columns; it must add none, so the
    column-name set equals exactly `HYDROS_SCHEMA_FIELDS`.
    """
    assert CHRONOLOGICAL_CASE.is_dir(), (
        f"the chronological fixture must exist at {CHRONOLOGICAL_CASE}"
    )

    with tempfile.TemporaryDirectory() as out_dir:
        out = pathlib.Path(out_dir)
        _run_chronological(out)
        parquets = _hydros_parquets(out)
        assert parquets, "the chronological run must emit a hydros partition"
        schema_names = set(pq.read_schema(parquets[0]).names)

    assert schema_names == HYDROS_SCHEMA_FIELDS, (
        "hydros parquet schema must equal exactly the known field set; "
        f"unexpected columns {sorted(schema_names - HYDROS_SCHEMA_FIELDS)}, "
        f"missing columns {sorted(HYDROS_SCHEMA_FIELDS - schema_names)}"
    )


def test_chronological_study_surface_matches_run_module() -> None:
    """The cobre.Study surface emits the same per-block storage variation.

    Study.train().simulate() and cobre.run.run both converge on
    run_simulation_phase_py, so a multi-block group's storage must vary across
    block rows through the Study surface too.
    """
    import cobre  # noqa: PLC0415

    assert CHRONOLOGICAL_CASE.is_dir(), (
        f"the chronological fixture must exist at {CHRONOLOGICAL_CASE}"
    )

    with tempfile.TemporaryDirectory() as out_dir:
        out = pathlib.Path(out_dir)
        study = cobre.Study(str(CHRONOLOGICAL_CASE), output_dir=out_dir)
        policy = study.train()
        study.simulate(policy)
        rows = _read_hydro_rows(out)

    groups = _multi_block_groups(_group_by_hydro_stage(rows))
    assert groups, "the Study surface must emit multi-block hydro groups"
    varied = [
        key
        for key, grp in groups.items()
        if not _all_close([float(r["storage_initial_hm3"]) for r in grp])
    ]
    assert varied, (
        "the Study surface must produce per-block storage variation identical to "
        "cobre.run.run (both reach run_simulation_phase_py)"
    )


def test_chronological_hydros_values_match_cli(
    cli_binary: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """CLI and Python report identical per-block hydro values, `water_value_per_hm3`
    included.

    Both front ends converge on the same shared extraction and writer path, so
    every `HYDROS_SCHEMA_FIELDS` column must agree once rows are matched by
    `(scenario_id, hydro_id, stage_id, block_id)`. This asserts value equality
    across the two front ends only — not that block values differ or coincide
    within a single run.
    """
    assert CHRONOLOGICAL_CASE.is_dir(), (
        f"the chronological fixture must exist at {CHRONOLOGICAL_CASE}"
    )

    cli_out = tmp_path / "cli"
    py_out = tmp_path / "python"
    run_cli(CHRONOLOGICAL_CASE, cli_out, cli_binary)

    import cobre.run  # noqa: PLC0415

    cobre.run.run(str(CHRONOLOGICAL_CASE), output_dir=str(py_out))

    cli_rows = _read_hydro_rows(cli_out)
    py_rows = _read_hydro_rows(py_out)

    key_cols = ("scenario_id", "hydro_id", "stage_id", "block_id")
    cli_by_key = {tuple(row[c] for c in key_cols): row for row in cli_rows}
    py_by_key = {tuple(row[c] for c in key_cols): row for row in py_rows}

    assert cli_by_key, "the chronological fixture must emit hydro rows"
    assert cli_by_key.keys() == py_by_key.keys(), (
        "CLI and Python hydros row sets must share the same "
        "(scenario_id, hydro_id, stage_id, block_id) keys"
    )

    value_cols = sorted(HYDROS_SCHEMA_FIELDS - set(key_cols))
    mismatches: list[str] = []
    for key, cli_row in cli_by_key.items():
        py_row = py_by_key[key]
        for col in value_cols:
            if cli_row[col] != py_row[col]:
                mismatches.append(
                    f"  key {key} column '{col}': CLI={cli_row[col]!r} "
                    f"PY={py_row[col]!r}"
                )
    assert not mismatches, "CLI/Python hydros value mismatch:\n" + "\n".join(mismatches)


def test_parallel_per_block_storage_and_evaporation_are_constant() -> None:
    """Parallel run (symmetric negative): storage and evaporation are block-constant.

    A parallel multi-block run models storage once per stage, so every block row
    of a `(hydro_id, stage_id)` group repeats the single stage pair and the single
    stage evaporation. Guards against per-block resolution leaking into parallel
    output.
    """
    assert PARALLEL_CASE.is_dir(), f"the parallel fixture must exist at {PARALLEL_CASE}"

    import cobre.run  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as out_dir:
        out = pathlib.Path(out_dir)
        cobre.run.run(str(PARALLEL_CASE), output_dir=out_dir)
        rows = _read_hydro_rows(out)

    groups = _multi_block_groups(_group_by_hydro_stage(rows))
    assert groups, "the parallel fixture must emit multi-block hydro groups"

    for key, grp in groups.items():
        assert _all_close([float(r["storage_initial_hm3"]) for r in grp]), (
            f"parallel storage_initial_hm3 must be identical across block rows "
            f"for group {key}"
        )
        assert _all_close([float(r["storage_final_hm3"]) for r in grp]), (
            f"parallel storage_final_hm3 must be identical across block rows "
            f"for group {key}"
        )
        evap = [r["evaporation_m3s"] for r in grp]
        if all(v is not None for v in evap):
            assert _all_close([float(v) for v in evap]), (
                f"parallel evaporation_m3s must be identical across block rows "
                f"for group {key}"
            )
