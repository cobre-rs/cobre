#!/usr/bin/env python3
"""Benchmark Cobre worker-count and CPU-binding configurations.

The harness runs a Cartesian matrix of thread counts and binding policies,
retains every output directory, and rejects numerical drift by hashing the
policy checkpoint and deterministic output payloads. It uses only the Python
standard library.

Example:
    python3 scripts/benchmark_numa.py CASE \
        --binary target/release/cobre \
        --results numa-results \
        --threads 32 48 64 96 \
        --policies none core numa \
        --repetitions 3

Run the whole command under ``numactl`` or a scheduler allocation to compare
external memory/rank placement while preserving the same inner matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, help="Cobre case directory")
    parser.add_argument(
        "--binary", type=Path, default=Path("target/release/cobre"), help="cobre binary"
    )
    parser.add_argument("--results", type=Path, required=True, help="new results directory")
    parser.add_argument("--threads", type=int, nargs="+", required=True)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=("none", "core", "numa"),
        default=("none", "core", "numa"),
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=86_400, help="seconds per run")
    parser.add_argument("--order-seed", type=int, default=0, help="matched-epoch arm order")
    return parser.parse_args()


def files_digest(
    root: Path, files: list[Path], normalized_json: set[str] | None = None
) -> str:
    digest = hashlib.sha256()
    if not files:
        raise RuntimeError(f"no files selected under {root}")
    for path in sorted(files):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        relative_text = path.relative_to(root).as_posix()
        if normalized_json and relative_text in normalized_json:
            value = json.loads(path.read_text())
            value.pop("generated_at", None)
            value.pop("created_at", None)
            data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        else:
            data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def policy_digest(output: Path) -> str:
    root = output / "policy"
    return files_digest(
        root,
        [path for path in root.rglob("*") if path.is_file()],
        normalized_json={"metadata.json"},
    )


def numerical_payload_digest(output: Path) -> str:
    excluded_files = {
        "training/metadata.json",
        "training/convergence.parquet",
        "simulation/metadata.json",
        "training/_SUCCESS",
        "simulation/_SUCCESS",
    }
    excluded_directories = ("training/timing/", "training/solver/")
    files = []
    for path in output.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(output).as_posix()
        if relative in excluded_files or relative.startswith(excluded_directories):
            continue
        files.append(path)
    return files_digest(
        output,
        files,
        normalized_json={
            "policy/metadata.json",
            "training/dictionaries/codes.json",
        },
    )


def optional_command(*command: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": str(error)}
    return {
        "available": True,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def process_rss_kib(pid: int) -> int | None:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    for field in ("VmHWM:", "VmRSS:"):
        for line in status.splitlines():
            if line.startswith(field):
                return int(line.split()[1])
    return None


def run_process(
    command: list[str], timeout: int
) -> tuple[subprocess.CompletedProcess[str], int | None]:
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    deadline = time.monotonic() + timeout
    peak_rss_kib: int | None = None
    while process.poll() is None:
        sample = process_rss_kib(process.pid)
        if sample is not None:
            peak_rss_kib = max(peak_rss_kib or 0, sample)
        if time.monotonic() >= deadline:
            process.kill()
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(command, timeout, stdout, stderr)
        time.sleep(0.1)
    stdout, stderr = process.communicate()
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr), peak_rss_kib


def run_arm(
    args: argparse.Namespace, threads: int, policy: str, repetition: int
) -> dict[str, Any]:
    output = args.results / f"threads-{threads}" / policy / f"run-{repetition:02d}"
    output.mkdir(parents=True, exist_ok=False)
    command = [
        str(args.binary),
        "run",
        str(args.case),
        "--output",
        str(output),
        "--threads",
        str(threads),
        "--cpu-bind",
        policy,
        "--quiet",
    ]
    started = time.monotonic()
    result, peak_rss_kib = run_process(command, args.timeout)
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise RuntimeError(
            f"arm threads={threads} policy={policy} repetition={repetition} failed "
            f"with exit {result.returncode}:\n{result.stderr}"
        )

    metadata = json.loads((output / "training" / "metadata.json").read_text())
    simulation_path = output / "simulation" / "metadata.json"
    simulation = json.loads(simulation_path.read_text()) if simulation_path.is_file() else None
    solve_stats = metadata.get("solve_stats", {})
    phase_wall_seconds = {
        "forward": solve_stats.get("forward_phase_wall_seconds"),
        "backward": solve_stats.get("backward_phase_wall_seconds"),
        "lower_bound": solve_stats.get("serial_lower_bound_seconds"),
        "cut_selection": solve_stats.get("serial_row_selection_seconds"),
        "cut_sync": solve_stats.get("serial_row_sync_seconds"),
        "allreduce": solve_stats.get("serial_allreduce_seconds"),
        "scheduling": solve_stats.get("serial_scheduling_seconds"),
        "simulation": simulation["duration_seconds"] if simulation else None,
    }
    required_phase_fields = ("forward", "backward", "cut_selection")
    missing_phase_fields = [
        field for field in required_phase_fields if phase_wall_seconds[field] is None
    ]
    if missing_phase_fields:
        raise RuntimeError(
            "binary did not persist required phase timing fields: "
            + ", ".join(missing_phase_fields)
        )
    return {
        "threads": threads,
        "policy": policy,
        "repetition": repetition,
        "output": str(output),
        "process_wall_seconds": elapsed,
        "training_wall_seconds": metadata["duration_seconds"],
        "simulation_wall_seconds": simulation["duration_seconds"] if simulation else None,
        "peak_rss_kib": peak_rss_kib,
        "phase_wall_seconds": phase_wall_seconds,
        "setup": metadata.get("setup"),
        "solve_stats": solve_stats,
        "simulation_solve_stats": simulation.get("solve_stats", {}) if simulation else None,
        "simulation_numerical_metadata": {
            "scenarios": simulation.get("scenarios", {}),
            "cost": simulation.get("cost"),
        }
        if simulation
        else None,
        "iterations": metadata.get("iterations", {}),
        "bounds": metadata.get("bounds", {}),
        "row_pool": metadata.get("row_pool", {}),
        "rank_affinity": metadata["distribution"].get("rank_affinity", []),
        "policy_sha256": policy_digest(output),
        "numerical_payload_sha256": numerical_payload_digest(output),
    }


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault((run["threads"], run["policy"]), []).append(run)

    summary = []
    for (threads, policy), group in sorted(groups.items()):
        walls = [run["training_wall_seconds"] for run in group]
        phase_names = tuple(group[0]["phase_wall_seconds"])
        median_phase_wall_seconds = {}
        for phase in phase_names:
            values = [run["phase_wall_seconds"][phase] for run in group]
            median_phase_wall_seconds[phase] = (
                statistics.median(values) if all(value is not None for value in values) else None
            )
        summary.append(
            {
                "threads": threads,
                "policy": policy,
                "repetitions": len(group),
                "median_training_wall_seconds": statistics.median(walls),
                "min_training_wall_seconds": min(walls),
                "max_training_wall_seconds": max(walls),
                "training_wall_seconds_stddev": statistics.pstdev(walls),
                "training_wall_seconds_mad": statistics.median(
                    abs(wall - statistics.median(walls)) for wall in walls
                ),
                "median_phase_wall_seconds": median_phase_wall_seconds,
            }
        )
    baselines = {
        entry["threads"]: entry["median_training_wall_seconds"]
        for entry in summary
        if entry["policy"] == "none"
    }
    for entry in summary:
        baseline = baselines.get(entry["threads"])
        if baseline and entry["policy"] != "none":
            entry["median_speedup_percent_vs_none"] = (
                (baseline - entry["median_training_wall_seconds"]) / baseline * 100
            )
    return summary


def work_signature(run: dict[str, Any]) -> str:
    fields = ("total_lp_solves", "first_try", "retried", "failed")
    signature = {
        "training": {field: run["solve_stats"].get(field) for field in fields},
        "simulation": {
            field: (run["simulation_solve_stats"] or {}).get(field) for field in fields
        },
    }
    return json.dumps(signature, sort_keys=True)


def numerical_metadata_signature(run: dict[str, Any]) -> str:
    signature = {
        "iterations": run["iterations"],
        "bounds": run["bounds"],
        "row_pool": run["row_pool"],
        "simulation": run["simulation_numerical_metadata"],
    }
    return json.dumps(signature, sort_keys=True)


def main() -> int:
    args = parse_args()
    if not args.case.is_dir():
        raise SystemExit(f"case directory does not exist: {args.case}")
    if not args.binary.is_file():
        raise SystemExit(f"cobre binary does not exist: {args.binary}")
    if args.results.exists():
        raise SystemExit(f"results directory already exists: {args.results}")
    if args.repetitions < 1 or any(threads < 1 for threads in args.threads):
        raise SystemExit("threads and repetitions must be positive")
    args.results.mkdir(parents=True)

    report: dict[str, Any] = {
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
            "process_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "lscpu": optional_command("lscpu", "--json"),
            "numactl": optional_command("numactl", "--hardware"),
            "numactl_policy": optional_command("numactl", "--show"),
        },
        "case": str(args.case.resolve()),
        "binary": str(args.binary.resolve()),
        "order_seed": args.order_seed,
        "runs": [],
    }
    report_path = args.results / "numa-benchmark.json"

    arms = [(threads, policy) for threads in args.threads for policy in args.policies]
    order_rng = random.Random(args.order_seed)
    for repetition in range(1, args.repetitions + 1):
        epoch = arms.copy()
        order_rng.shuffle(epoch)
        for threads, policy in epoch:
            run = run_arm(args, threads, policy, repetition)
            run["epoch"] = repetition
            report["runs"].append(run)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"threads={threads} policy={policy} run={repetition}: "
                f"{run['training_wall_seconds']:.3f}s",
                flush=True,
            )

    policy_hashes = {run["policy_sha256"] for run in report["runs"]}
    payload_hashes = {run["numerical_payload_sha256"] for run in report["runs"]}
    work_signatures = {work_signature(run) for run in report["runs"]}
    metadata_signatures = {
        numerical_metadata_signature(run) for run in report["runs"]
    }
    report["numerical_policy_hashes_match"] = len(policy_hashes) == 1
    report["numerical_payload_hashes_match"] = len(payload_hashes) == 1
    report["numerical_metadata_matches"] = len(metadata_signatures) == 1
    report["solver_work_counts_match"] = len(work_signatures) == 1
    report["summary"] = summarize(report["runs"])
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if len(policy_hashes) != 1:
        print("error: numerical policy hashes differ between benchmark arms", file=sys.stderr)
        return 2
    if len(payload_hashes) != 1 or len(metadata_signatures) != 1:
        print("error: numerical result payloads differ between benchmark arms", file=sys.stderr)
        return 3
    if len(work_signatures) != 1:
        print("error: solver work counts differ between benchmark arms", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
