"""Integration tests for the ``on_iteration`` streaming callback of cobre.run.run().

These tests exercise the GIL-reacquiring drain thread that forwards each training
iteration boundary to a user-supplied Python callable while the solver runs with
the GIL released. They verify three behaviours:

1. callback-vs-parquet parity — the callback observes exactly the iterations that
   are persisted to ``training/convergence.parquet``, with matching bounds/gap;
2. cooperative early stop — a truthy return halts training at the next boundary
   while still writing the run's (partial) artifacts and running the configured
   simulation;
3. raising-callback propagation — an exception raised by the callback surfaces as
   the run's exception, after the partial ``training/metadata.json`` is written.

Run with (from the repo root, after building the extension):
    .venv/bin/maturin develop --manifest-path crates/cobre-python/Cargo.toml
    .venv/bin/python -m pytest crates/cobre-python/tests/test_callback.py -q
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pyarrow.parquet as pq
import pytest

VALID_CASE = "examples/1dtoy"


def _read_convergence_rows(output_dir: pathlib.Path) -> list[dict[str, Any]]:
    """Read ``training/convergence.parquet`` as a list of per-iteration dicts."""
    table = pq.read_table(output_dir / "training" / "convergence.parquet")
    columns = table.to_pydict()
    n = table.num_rows
    return [{name: columns[name][i] for name in columns} for i in range(n)]


def test_callback_matches_convergence_parquet(tmp_path: pathlib.Path) -> None:
    """The callback observes exactly the iterations persisted to the parquet.

    For each invocation the callback records the dict it received. After the run
    the recorded list must have one entry per convergence-parquet row, and the
    ``iteration``/``lower_bound``/``gap`` values must match the corresponding
    parquet row (the parquet stores ``gap_percent`` = ``gap`` * 100).
    """
    import cobre.run  # noqa: PLC0415

    observed: list[dict[str, Any]] = []

    def on_iteration(event: dict[str, Any]) -> None:
        observed.append(event)

    cobre.run.run(
        VALID_CASE,
        output_dir=str(tmp_path),
        config_overrides={"simulation": {"enabled": False}},
        on_iteration=on_iteration,
    )

    rows = _read_convergence_rows(tmp_path)

    assert len(observed) == len(rows), (
        "callback must be invoked once per convergence-parquet row: "
        f"observed {len(observed)} vs parquet {len(rows)}"
    )
    assert len(observed) > 0, "1dtoy must run at least one iteration"

    # The callback fires in iteration order; align positionally and verify the
    # per-row scalar parity.
    for event, row in zip(observed, rows, strict=True):
        assert event["kind"] == "iteration"
        assert int(event["iteration"]) == int(row["iteration"]), (
            "callback iteration must match parquet iteration"
        )
        assert event["lower_bound"] == pytest.approx(row["lower_bound"], rel=1e-9), (
            "callback lower_bound must match parquet lower_bound"
        )
        # The event carries the raw relative gap; the parquet stores gap * 100.
        parquet_gap_percent = row["gap_percent"]
        if parquet_gap_percent is not None:
            assert event["gap"] * 100.0 == pytest.approx(
                parquet_gap_percent, rel=1e-9, abs=1e-12
            ), "callback gap*100 must match parquet gap_percent"


def test_callback_truthy_return_stops_early(tmp_path: pathlib.Path) -> None:
    """A truthy callback return cooperatively halts training near the trigger.

    This asserts the guarantees the cooperative-async design actually provides, not
    a literal stop iteration. The drain thread invokes the callback under the GIL at
    each iteration boundary, but the solver runs GIL-released and reads the shared
    shutdown flag once per iteration, just before its stop decision. Between the
    boundary at which the callback returns truthy and the boundary at which the
    solver next observes the flag, the solver may advance a small, bounded number of
    extra iterations — the GIL-released solver outruns the GIL-reacquiring callback,
    so the flag is observed at a later boundary. This is the same lag as the CLI's
    SIGINT handling, and is why the bound below is a small ceiling rather than an
    exact iteration. Forcing a synchronous stop was rejected (it would violate P2
    and the hot-path no-allocation rules), so the test pins the contract, not the
    timing.

    Guarantees asserted:
    1. the callback fired (it observed the trigger iteration);
    2. the truthy return cut the run far below the normal iteration limit;
    3. the partial ``training/metadata.json`` exists on disk;
    4. the callback was invoked only a small bounded number of extra times after
       the trigger.
    """
    import cobre.run  # noqa: PLC0415

    trigger_iteration = 3
    calls: list[int] = []

    def on_iteration(event: dict[str, Any]) -> bool:
        iteration = int(event["iteration"])
        calls.append(iteration)
        return iteration >= trigger_iteration

    result = cobre.run.run(
        VALID_CASE,
        output_dir=str(tmp_path),
        config_overrides={"simulation": {"enabled": False}},
        on_iteration=on_iteration,
    )

    # (1) The callback fired and observed the trigger iteration.
    assert calls, "the callback must be invoked at least once"
    assert max(calls) >= trigger_iteration, (
        "the callback must have observed the trigger iteration "
        f"(>= {trigger_iteration}); saw {calls}"
    )

    # (2) The truthy return cut the run far below 1dtoy's normal iteration limit
    #     (an unstopped 1dtoy run trains to its configured limit).
    assert result["iterations"] < 10, (
        f"a cooperatively-stopped run must report < 10 iterations, got "
        f"{result['iterations']} (calls={calls})"
    )

    # (3) The partial artifacts are still written on a cooperative stop, and the
    #     metadata records the shutdown as the reason, not as convergence.
    metadata_path = tmp_path / "training" / "metadata.json"
    assert metadata_path.exists(), (
        "a cooperatively-stopped run must still write training/metadata.json"
    )
    convergence = json.loads(metadata_path.read_text())["convergence"]
    assert convergence["termination_reason"] == "graceful_shutdown", convergence
    assert convergence["achieved"] is False, convergence

    # (4) The cooperative lag is bounded: only a small number of extra callback
    #     invocations may occur after the trigger before the solver observes the
    #     flag at a later boundary.
    assert len(calls) <= 6, (
        f"the callback must be invoked only a small bounded number of times, "
        f"got {len(calls)}: {calls}"
    )


def test_callback_truthy_return_writes_partial_training_status(
    tmp_path: pathlib.Path,
) -> None:
    """A training a callback stop alone ended reports ``status`` ``partial``."""
    import cobre.run  # noqa: PLC0415

    def on_iteration(event: dict[str, Any]) -> bool:
        return int(event["iteration"]) >= 3

    cobre.run.run(
        VALID_CASE,
        output_dir=str(tmp_path),
        config_overrides={"simulation": {"enabled": False}},
        on_iteration=on_iteration,
    )

    metadata = json.loads((tmp_path / "training" / "metadata.json").read_text())
    assert metadata["status"] == "partial", metadata
    assert metadata["convergence"]["termination_reason"] == "graceful_shutdown", (
        metadata["convergence"]
    )


def test_callback_truthy_return_keeps_configured_simulation(
    tmp_path: pathlib.Path,
) -> None:
    """A callback stop leaves training ``partial`` and still runs the simulation."""
    import cobre.run  # noqa: PLC0415

    def on_iteration(event: dict[str, Any]) -> bool:
        return int(event["iteration"]) >= 2

    result = cobre.run.run(
        VALID_CASE,
        output_dir=str(tmp_path),
        on_iteration=on_iteration,
    )

    training = json.loads((tmp_path / "training" / "metadata.json").read_text())
    assert training["status"] == "partial", training
    simulation = json.loads((tmp_path / "simulation" / "metadata.json").read_text())
    assert simulation["status"] == "complete", simulation
    assert simulation["scenarios"]["completed"] == 100, simulation["scenarios"]
    assert (tmp_path / "simulation" / "_SUCCESS").exists()
    assert (tmp_path / "simulation" / "costs").exists()
    assert result["simulation"] == {"n_scenarios": 100, "completed": 100}


def test_callback_raises_propagates_with_partial_metadata(
    tmp_path: pathlib.Path,
) -> None:
    """An exception raised in the callback surfaces as the run's exception.

    The partial ``training/metadata.json`` must exist on disk (artifacts are
    written before the captured exception is re-raised), and the propagated
    exception's message must contain the callback's message.
    """
    import cobre.run  # noqa: PLC0415

    def on_iteration(_event: dict[str, Any]) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        cobre.run.run(
            VALID_CASE,
            output_dir=str(tmp_path),
            config_overrides={"simulation": {"enabled": False}},
            on_iteration=on_iteration,
        )

    assert (tmp_path / "training" / "metadata.json").exists(), (
        "a raising-callback run must still write the partial training/metadata.json"
    )
