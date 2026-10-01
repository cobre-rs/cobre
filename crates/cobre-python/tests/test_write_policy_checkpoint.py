"""Integration tests for cobre.write_policy_checkpoint.

Verifies that a policy checkpoint authored from plain Python dicts round-trips
through cobre.results.load_policy — the read path whose emitted dict shapes
write_policy_checkpoint's input mirrors — and that malformed input raises a
clear error naming the offending stage/cut.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_write_policy_checkpoint.py
"""

import pathlib
from typing import Any, Optional

import pytest


def _make_metadata(
    cost_scale_factor: Optional[float] = 2_500_000.0,
    season_manifest: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    producer: dict[str, Any] = {
        "completed_iterations": 5,
        "final_lower_bound": 123.45,
        "best_upper_bound": 130.0,
        "max_iterations": 10,
        "forward_passes": 4,
        "warm_start_cuts": 0,
        "warm_start_counts": [0],
        "rng_seed": 42,
        "total_visited_states": 0,
        "training_block_mode": "parallel",
        "training_block_mode_per_stage": [],
    }
    if cost_scale_factor is not None:
        producer["cost_scale_factor"] = cost_scale_factor
    metadata = {
        "created_at": "2026-07-30T00:00:00Z",
        "num_stages": 1,
        "producer": producer,
    }
    if season_manifest is not None:
        metadata["season_manifest"] = season_manifest
    return metadata


def _make_stage_cuts() -> list[dict[str, Any]]:
    return [
        {
            "stage_id": 0,
            "state_dimension": 3,
            "capacity": 10,
            "cuts": [
                {
                    "cut_id": 1,
                    "slot_index": 0,
                    "iteration": 1,
                    "forward_pass_index": 0,
                    "intercept": 42.0,
                    "coefficients": [1.0, 2.0, 3.0],
                    "is_active": True,
                },
                {
                    "cut_id": 2,
                    "slot_index": 1,
                    "iteration": 1,
                    "forward_pass_index": 1,
                    "intercept": 10.5,
                    "coefficients": [0.5, -1.5, 2.5],
                    "is_active": True,
                },
            ],
        }
    ]


def test_write_policy_checkpoint_round_trip(tmp_path: pathlib.Path) -> None:
    """A synthetic checkpoint written from dicts reads back with matching cuts
    and metadata, including the cost_scale_factor provenance marker.
    """
    import cobre  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415

    cobre.write_policy_checkpoint(
        str(tmp_path / "policy"), _make_stage_cuts(), _make_metadata()
    )

    loaded = cobre.results.load_policy(str(tmp_path))

    assert loaded["metadata"]["format_version"] == 2
    assert loaded["metadata"]["producer"]["cost_scale_factor"] == pytest.approx(
        2_500_000.0
    )
    assert loaded["metadata"]["producer"]["completed_iterations"] == 5

    assert len(loaded["stage_cuts"]) == 1
    stage = loaded["stage_cuts"][0]
    assert stage["stage_id"] == 0
    assert stage["state_dimension"] == 3
    assert stage["capacity"] == 10
    assert stage["populated_count"] == 2, "populated_count must default to len(cuts)"

    cuts = stage["cuts"]
    assert len(cuts) == 2
    assert cuts[0]["cut_id"] == 1
    assert cuts[0]["intercept"] == pytest.approx(42.0)
    assert cuts[0]["coefficients"] == pytest.approx([1.0, 2.0, 3.0])
    assert cuts[1]["cut_id"] == 2
    assert cuts[1]["coefficients"] == pytest.approx([0.5, -1.5, 2.5])


def test_write_policy_checkpoint_stamps_the_running_version(
    tmp_path: pathlib.Path,
) -> None:
    """A caller-supplied cobre_version is ignored; the checkpoint always
    records the running cobre version.
    """
    import cobre  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415

    metadata = _make_metadata()
    metadata["cobre_version"] = "0.13.0"

    cobre.write_policy_checkpoint(
        str(tmp_path / "policy"), _make_stage_cuts(), metadata
    )

    loaded = cobre.results.load_policy(str(tmp_path))
    assert loaded["metadata"]["cobre_version"] == cobre.__version__


def test_write_policy_checkpoint_cost_scale_factor_omitted_reads_as_none(
    tmp_path: pathlib.Path,
) -> None:
    """Omitting cost_scale_factor from the metadata dict reads back as None,
    matching a legacy (pre-marker) checkpoint.
    """
    import cobre  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415

    cobre.write_policy_checkpoint(
        str(tmp_path / "policy"),
        _make_stage_cuts(),
        _make_metadata(cost_scale_factor=None),
    )

    loaded = cobre.results.load_policy(str(tmp_path))
    assert loaded["metadata"]["producer"]["cost_scale_factor"] is None


def test_write_policy_checkpoint_coefficient_length_mismatch_raises(
    tmp_path: pathlib.Path,
) -> None:
    """A cut whose coefficients length disagrees with its stage's
    state_dimension raises ValueError naming the stage and cut.
    """
    import cobre  # noqa: PLC0415

    stage_cuts = _make_stage_cuts()
    stage_cuts[0]["cuts"][0]["coefficients"] = [1.0, 2.0]  # state_dimension is 3

    with pytest.raises(ValueError, match=r"stage 0 cut 1.*coefficients"):
        cobre.write_policy_checkpoint(
            str(tmp_path / "policy"), stage_cuts, _make_metadata()
        )


def test_write_policy_checkpoint_defaults_apply_when_keys_omitted(
    tmp_path: pathlib.Path,
) -> None:
    """warm_start_count, active_cut_indices, and entity_manifest all default
    when their keys are absent from the stage_cuts dict.
    """
    import cobre  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415

    stage_cuts = _make_stage_cuts()
    assert "warm_start_count" not in stage_cuts[0]
    assert "active_cut_indices" not in stage_cuts[0]
    assert "entity_manifest" not in stage_cuts[0]

    cobre.write_policy_checkpoint(
        str(tmp_path / "policy"), stage_cuts, _make_metadata()
    )

    loaded = cobre.results.load_policy(str(tmp_path))
    assert loaded["stage_cuts"][0]["warm_start_count"] == 0


# EntityType discriminants from schemas/policy.fbs (owned by cobre-sddp, mirrored
# here for the manifest a bridge-style caller supplies).
_ENTITY_TYPE_HYDRO_STORAGE = 0
_ENTITY_TYPE_HYDRO_INFLOW_LAG = 1


def _storage_only_stage_cuts(
    hydro_ids: list[int], lag_coefficients: dict[int, list[float]]
) -> list[dict[str, Any]]:
    """A single stage whose manifest is one HydroStorage slot per hydro (no lag
    slots — the DECOMP bootstrap shape), one cut carrying storage-aligned
    coefficients plus keyed inflow_lag_coefficients.
    """
    manifest = [
        {
            "entity_type": _ENTITY_TYPE_HYDRO_STORAGE,
            "entity_id": hid,
            "subindex": 0,
            "was_active": True,
        }
        for hid in hydro_ids
    ]
    return [
        {
            "stage_id": 0,
            "state_dimension": len(hydro_ids),
            "capacity": 4,
            "entity_manifest": manifest,
            "cuts": [
                {
                    "cut_id": 1,
                    "slot_index": 0,
                    "iteration": 1,
                    "forward_pass_index": 0,
                    "intercept": 7.0,
                    "coefficients": [float(hid) for hid in hydro_ids],
                    "inflow_lag_coefficients": lag_coefficients,
                    "is_active": True,
                }
            ],
        }
    ]


def test_write_policy_checkpoint_reserves_inflow_lag_slots(
    tmp_path: pathlib.Path,
) -> None:
    """inflow_lag_depth=N widens the written manifest with N canonical
    HydroInflowLag slots per storage hydro, self-describing depth N, and places
    each keyed pi_qafl coefficient at its (hydro, depth) position.
    """
    import cobre  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415

    hydro_ids = [1, 2]
    # hydro 1: depth1=1.1, depth2=1.2; hydro 2: depth1=2.1, depth2 defaults 0.0.
    lag_coefficients = {1: [1.1, 1.2], 2: [2.1]}
    stage_cuts = _storage_only_stage_cuts(hydro_ids, lag_coefficients)

    cobre.write_policy_checkpoint(
        str(tmp_path / "policy"),
        stage_cuts,
        _make_metadata(),
        inflow_lag_depth=2,
    )

    loaded = cobre.results.load_policy(str(tmp_path))
    stage = loaded["stage_cuts"][0]

    # 2 storage + 2 hydros × 2 depths = 6 slots.
    assert stage["state_dimension"] == 6
    manifest = stage["entity_manifest"]
    assert len(manifest) == 6

    lag_slots = [
        s for s in manifest if s["entity_type"] == _ENTITY_TYPE_HYDRO_INFLOW_LAG
    ]
    assert len(lag_slots) == 4
    # Self-describes depth 2: deepest 1-based lag subindex is 2.
    assert max(s["subindex"] for s in lag_slots) == 2

    # Coefficients: storage[1.0, 2.0] ++ lag-major (h1,d1),(h2,d1),(h1,d2),(h2,d2).
    coeffs = stage["cuts"][0]["coefficients"]
    assert coeffs == pytest.approx([1.0, 2.0, 1.1, 2.1, 1.2, 0.0])


def test_write_policy_checkpoint_inflow_lag_depth_absent_is_byte_identical(
    tmp_path: pathlib.Path,
) -> None:
    """Omitting inflow_lag_depth (or passing 0) writes the exact same bytes as
    not passing it — the reservation path is inert by default.
    """
    import cobre  # noqa: PLC0415

    def _digest(policy_dir: pathlib.Path) -> dict[str, bytes]:
        return {
            p.name: p.read_bytes() for p in sorted((policy_dir / "cuts").glob("*.bin"))
        }

    stage_cuts = _storage_only_stage_cuts([1, 2], {})

    default_dir = tmp_path / "default"
    cobre.write_policy_checkpoint(str(default_dir), stage_cuts, _make_metadata())

    zero_dir = tmp_path / "zero"
    cobre.write_policy_checkpoint(
        str(zero_dir), stage_cuts, _make_metadata(), inflow_lag_depth=0
    )

    assert _digest(default_dir) == _digest(zero_dir)


def test_write_policy_checkpoint_unplaceable_lag_coefficient_raises(
    tmp_path: pathlib.Path,
) -> None:
    """A keyed inflow-lag coefficient for a hydro with no storage slot is
    unplaceable and raises ValueError naming the hydro — never silently dropped.
    """
    import cobre  # noqa: PLC0415

    # hydro 99 is not among the storage hydros [1, 2].
    stage_cuts = _storage_only_stage_cuts([1, 2], {99: [0.5]})

    with pytest.raises(ValueError, match=r"hydro 99"):
        cobre.write_policy_checkpoint(
            str(tmp_path / "policy"),
            stage_cuts,
            _make_metadata(),
            inflow_lag_depth=2,
        )


def test_write_policy_checkpoint_accepts_self_describing_stage_fields(
    tmp_path: pathlib.Path,
) -> None:
    """The binding accepts the per-stage self-describing facts (cost_scale_factor,
    node_id, graph_stage_id) that the CLI writer emits — Python write-parity — and
    produces a checkpoint whose read surfaces those facts again.
    """
    import cobre  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415

    stage_cuts = _make_stage_cuts()
    stage_cuts[0]["cost_scale_factor"] = 2_500_000.0
    stage_cuts[0]["node_id"] = 3
    stage_cuts[0]["graph_stage_id"] = 7

    cobre.write_policy_checkpoint(
        str(tmp_path / "policy"), stage_cuts, _make_metadata()
    )

    loaded = cobre.results.load_policy(str(tmp_path))
    stage = loaded["stage_cuts"][0]
    assert stage["state_dimension"] == 3
    assert len(stage["cuts"]) == 2
    assert stage["node_id"] == 3
    assert stage["graph_stage_id"] == 7
    assert stage["cost_scale_factor"] == pytest.approx(2_500_000.0)
    assert loaded["metadata"]["producer"]["cost_scale_factor"] == pytest.approx(
        2_500_000.0
    )


def test_load_policy_self_describing_fields_survive_rewrite(
    tmp_path: pathlib.Path,
) -> None:
    """A non-sentinel node_id/graph_stage_id and a stage-level cost_scale_factor
    survive a load_policy -> write_policy_checkpoint -> reload cycle — the
    load -> edit -> write round-trip the binding advertises.

    Regression: load_policy previously omitted these per-stage fields, so the
    rewrite defaulted the missing keys back to the -1 node/graph sentinel (and
    cost_scale_factor to None), collapsing a genuine single-node pool and
    breaking boundary-cut load. This test fails unless load_policy surfaces them.
    """
    import cobre  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415

    stage_cuts = _make_stage_cuts()
    stage_cuts[0]["node_id"] = 3
    stage_cuts[0]["graph_stage_id"] = 7
    stage_cuts[0]["cost_scale_factor"] = 7_777.0

    # metadata carries no cost_scale_factor, so only the surfaced stage-level
    # value can keep it non-None through the rewrite — isolating it from the
    # producer-block fallback that would otherwise mask a dropped field.
    metadata = _make_metadata(cost_scale_factor=None)

    first_dir = tmp_path / "first"
    cobre.write_policy_checkpoint(str(first_dir / "policy"), stage_cuts, metadata)

    loaded = cobre.results.load_policy(str(first_dir))

    second_dir = tmp_path / "second"
    cobre.write_policy_checkpoint(
        str(second_dir / "policy"), loaded["stage_cuts"], metadata
    )
    reloaded = cobre.results.load_policy(str(second_dir))

    stage = reloaded["stage_cuts"][0]
    assert stage["node_id"] == 3
    assert stage["graph_stage_id"] == 7
    assert stage["cost_scale_factor"] == pytest.approx(7_777.0)


def test_write_policy_checkpoint_active_cut_indices_do_not_survive_the_round_trip(
    tmp_path: pathlib.Path,
) -> None:
    """active_cut_indices is written but not returned by load_policy, so a
    load -> write cycle resets it; cut activity round-trips through is_active.
    """
    import cobre  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415

    stage_cuts = _make_stage_cuts()
    stage_cuts[0]["active_cut_indices"] = [0]

    cobre.write_policy_checkpoint(
        str(tmp_path / "policy"), stage_cuts, _make_metadata()
    )

    loaded = cobre.results.load_policy(str(tmp_path))
    stage = loaded["stage_cuts"][0]

    assert "active_cut_indices" not in stage
    assert stage["cuts"][0]["is_active"] is True


def test_write_policy_checkpoint_lag_coefficients_without_depth_raises(
    tmp_path: pathlib.Path,
) -> None:
    """inflow_lag_coefficients supplied without inflow_lag_depth raises
    ValueError naming the stage and cut — never silently dropped.
    """
    import cobre  # noqa: PLC0415

    stage_cuts = _storage_only_stage_cuts([1, 2], {1: [0.5]})

    with pytest.raises(ValueError, match=r"stage 0 cut .*inflow_lag_depth"):
        cobre.write_policy_checkpoint(
            str(tmp_path / "policy"), stage_cuts, _make_metadata()
        )


def test_write_policy_checkpoint_season_manifest_round_trips(
    tmp_path: pathlib.Path,
) -> None:
    """A checkpoint authored with season_manifest reads back with the same
    descriptor, and writing the loaded metadata produces a byte-identical
    checkpoint file.
    """
    import cobre  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415

    season_manifest = {
        "cycle_code": 0,
        "n_seasons": 3,
        "hydro_orders": [
            {"hydro_id": 1, "orders": [1, 2, 1]},
            {"hydro_id": 6, "orders": [2, 2, 1]},
        ],
    }

    first_dir = tmp_path / "first"
    cobre.write_policy_checkpoint(
        str(first_dir / "policy"),
        _make_stage_cuts(),
        _make_metadata(season_manifest=season_manifest),
    )

    loaded = cobre.results.load_policy(str(first_dir))
    assert loaded["metadata"]["season_manifest"] == season_manifest

    second_dir = tmp_path / "second"
    cobre.write_policy_checkpoint(
        str(second_dir / "policy"),
        loaded["stage_cuts"],
        loaded["metadata"],
    )

    first_manifest = (first_dir / "policy" / "manifest.bin").read_bytes()
    second_manifest = (second_dir / "policy" / "manifest.bin").read_bytes()
    assert first_manifest == second_manifest


def test_write_policy_checkpoint_season_manifest_omitted_is_absent(
    tmp_path: pathlib.Path,
) -> None:
    """A checkpoint authored with today's _make_metadata() (no season_manifest
    key) reads back with the absent descriptor.
    """
    import cobre  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415

    cobre.write_policy_checkpoint(
        str(tmp_path / "policy"), _make_stage_cuts(), _make_metadata()
    )

    loaded = cobre.results.load_policy(str(tmp_path))
    assert loaded["metadata"]["season_manifest"] == {
        "cycle_code": 255,
        "n_seasons": 0,
        "hydro_orders": [],
    }


def test_write_policy_checkpoint_season_manifest_unsorted_hydros_rejected_on_load(
    tmp_path: pathlib.Path,
) -> None:
    """hydro_orders with non-ascending hydro_id values: the write succeeds, and
    load_policy raises OutputError with 'not ascending by hydro_id'.
    """
    import cobre  # noqa: PLC0415
    import cobre.errors  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415

    season_manifest = {
        "cycle_code": 0,
        "n_seasons": 2,
        "hydro_orders": [
            {"hydro_id": 6, "orders": [1, 2]},
            {"hydro_id": 1, "orders": [2, 1]},
        ],
    }

    cobre.write_policy_checkpoint(
        str(tmp_path / "policy"),
        _make_stage_cuts(),
        _make_metadata(season_manifest=season_manifest),
    )

    with pytest.raises(cobre.errors.OutputError, match=r"not ascending by hydro_id"):
        cobre.results.load_policy(str(tmp_path))


def test_write_policy_checkpoint_season_manifest_order_length_rejected_on_load(
    tmp_path: pathlib.Path,
) -> None:
    """n_seasons=3 but one hydro with two orders: the write succeeds, and
    load_policy raises OutputError with 'expected n_seasons=3'.
    """
    import cobre  # noqa: PLC0415
    import cobre.errors  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415

    season_manifest = {
        "cycle_code": 0,
        "n_seasons": 3,
        "hydro_orders": [
            {"hydro_id": 1, "orders": [1, 2]},
        ],
    }

    cobre.write_policy_checkpoint(
        str(tmp_path / "policy"),
        _make_stage_cuts(),
        _make_metadata(season_manifest=season_manifest),
    )

    with pytest.raises(cobre.errors.OutputError, match=r"expected n_seasons=3"):
        cobre.results.load_policy(str(tmp_path))
