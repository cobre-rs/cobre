#!/usr/bin/env bash
# SLURM MPI integration test for `cobre run`.
#
# Exercises up to 5 MPI execution modes against the 4ree example case and
# verifies rank-count invariance: every classified output parquet of each
# test run is compared against the T1 single-rank baseline, decoded-value
# bit-exact (wall-clock timing columns and per-rank/worker execution-topology
# tables are exempt by design — see the CLASSIFICATION_ROWS table in
# compare_outputs_bitwise below).
#
# T1: mpiexec -n 1         (single rank, baseline)
# T2: mpiexec -n 2         (multi-rank, single machine)
# T3: sbatch -N 1 -n 2     (SLURM single-node)
# T4: sbatch -N 2 -n 2     (SLURM multi-node — catches MPICH 4.3.0 deadlock)
# T5: sbatch -N 2 -n 4     (SLURM multi-node, multiple ranks per node)
#
# The S cases run simulation only, from T1's policy, on a copy of the case
# whose stages 4 to 7 have their single block split in two. The stored bases
# of those stages no longer fit their LP and are not used; the others are.
# Each S run is compared against S1 as above, and its per-scenario solver
# counters are compared too, so the same stages run warm and cold in every
# layout.
#
# S1: mpiexec -n 1 --threads 1   (simulation-only baseline)
# S2: mpiexec -n 2 --threads 2
# S3: sbatch -N 2 -n 2           (mpiexec -n 2)
# S4: sbatch -N 2 -n 4           (mpiexec -n 4 --threads 1)
#
# Usage:
#   /shared/run-tests.sh [--mode reduced|full]
#
#   reduced   T1, T2, S1 and S2 only (mpiexec, no sbatch / SLURM cluster required).
#   full      every T and S case (default — preserves prior behavior).
#
# Prerequisites (satisfied by the Docker image):
#   - /shared/cobre-mpi   (cobre binary built with --features mpi)
#   - /shared/4ree        (example case directory)
#   - /opt/mpich/bin      (MPICH installation)
#   - python3 with pyarrow
set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
readonly COBRE_BIN="/shared/cobre-mpi"
readonly CASE_DIR="/shared/4ree"
readonly TIMEOUT=120
readonly WORK_DIR="/shared/work"

# Set by parse_mode(); read by main() and print_summary().
MODE="full"

# Prepend MPICH to PATH so mpiexec and Hydra proxy are found.
export PATH="/opt/mpich/bin:${PATH}"

# ---------------------------------------------------------------------------
# Result tracking (associative array: test name -> PASS|FAIL|<message>)
# ---------------------------------------------------------------------------
declare -A RESULTS

# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------
usage() {
    cat <<'USAGE'
Usage: run-tests.sh [--mode reduced|full]

  reduced   Run T1, T2, S1 and S2 (mpiexec only); no sbatch.
  full      Run T1 through T5 and S1 through S4 (default).
USAGE
}

# ---------------------------------------------------------------------------
# parse_mode <args...>
#
# Sets the global MODE from a "--mode reduced|full" argument (default
# "full", preserving prior behavior). Any other argument, a missing --mode
# value, or an unrecognized --mode value prints usage and exits 1.
# ---------------------------------------------------------------------------
parse_mode() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --mode)
                if [[ $# -lt 2 ]]; then
                    echo "ERROR: --mode requires a value (reduced|full)" >&2
                    usage
                    exit 1
                fi
                MODE="$2"
                shift 2
                ;;
            *)
                echo "ERROR: unrecognized argument: $1" >&2
                usage
                exit 1
                ;;
        esac
    done

    case "${MODE}" in
        reduced | full) ;;
        *)
            echo "ERROR: unrecognized --mode value: ${MODE} (expected reduced|full)" >&2
            usage
            exit 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
preflight_check() {
    local ok=1

    if [[ ! -x "${COBRE_BIN}" ]]; then
        echo "ERROR: cobre binary not found or not executable: ${COBRE_BIN}"
        ok=0
    fi

    if [[ ! -d "${CASE_DIR}" ]]; then
        echo "ERROR: case directory not found: ${CASE_DIR}"
        ok=0
    fi

    if ! command -v mpiexec &>/dev/null; then
        echo "ERROR: mpiexec not found in PATH (checked /opt/mpich/bin)"
        ok=0
    fi

    if ! python3 -c "import pyarrow.parquet" 2>/dev/null; then
        echo "ERROR: python3 with pyarrow is required for output comparison"
        ok=0
    fi

    if [[ "${ok}" -eq 0 ]]; then
        echo "ABORT: pre-flight checks failed — see errors above"
        exit 1
    fi

    mkdir -p "${WORK_DIR}"

    echo "INFO: cobre binary: ${COBRE_BIN}"
    echo "INFO: case directory: ${CASE_DIR}"
    echo "INFO: work directory: ${WORK_DIR}"
    echo "INFO: mpiexec: $(command -v mpiexec)"
    echo "INFO: python3: $(command -v python3)"
    echo "INFO: mode: ${MODE}"
    echo "INFO: timeout per test: ${TIMEOUT}s"
}

# ---------------------------------------------------------------------------
# compare_outputs_bitwise <baseline_dir> <test_dir>
#
# Classifies every relative *.parquet path found under either directory
# against the CLASSIFICATION_ROWS table (first-match-wins; a path matching
# no rule FAILs as "unclassified output file"), asserts the two non-exempt
# relative-path sets are equal, then for each classified file compares
# every non-exempt column index-aligned (rows are NEVER sorted) and
# bit-exact for floats (raw IEEE-754 bytes, so -0.0/NaN/last-bit differences
# are caught) or by value for other types. Exits 1 naming the file, column,
# and row index on the first mismatch; container-level parquet metadata
# (row groups, codec, created_by) is never inspected — only decoded values.
# ---------------------------------------------------------------------------
compare_outputs_bitwise() {
    local baseline_dir="$1"
    local test_dir="$2"
    python3 - "${baseline_dir}" "${test_dir}" <<'PYEOF'
import glob
import os
import re
import struct
import sys
from collections import namedtuple

import pyarrow as pa
import pyarrow.parquet as pq

baseline_dir = sys.argv[1]
test_dir = sys.argv[2]

# Ordered (glob_pattern, class, exempt_columns) rules -- first match wins.
# class is one of "compare", "compare-except", "exempt". A *.parquet path
# found under either directory that matches no rule FAILs as unclassified:
# a new output added later forces a conscious classification decision here,
# never a silent skip.
CLASSIFICATION_ROWS = [
    # `solver_iterations_schema()` stamps every backward row with the
    # producing `rank`/`worker_id`/`opening`; both the row count (every
    # (rank, worker) pair also emits an opening=0 sentinel row, so the count
    # scales with the rank/worker grid) and the row order (rank-major,
    # worker-minor traversal) legitimately vary with rank count, so this
    # table -- and `training/solver/iterations.parquet` below, which shares
    # the schema -- is exempt outright, never `compare`/`compare-except`
    # (the same rationale as the already-exempt `training/timing/**`). Must
    # stay ahead of the `simulation/**/*.parquet` catch-all: exempt wins only
    # on a first match.
    ("simulation/solver/iterations.parquet", "exempt", []),
    ("simulation/**/*.parquet", "compare", []),
    (
        "training/convergence.parquet",
        "compare-except",
        ["time_forward_ms", "time_backward_ms", "time_total_ms"],
    ),
    ("training/dictionaries/*.parquet", "compare", []),
    ("training/cut_selection/iterations.parquet", "compare-except", ["selection_time_ms"]),
    # Shares `solver_iterations_schema()` with `simulation/solver/iterations.parquet`
    # above -- same exempt rationale.
    ("training/solver/iterations.parquet", "exempt", []),
    ("training/solver/retry_histogram.parquet", "compare", []),
    ("training/timing/**", "exempt", []),
]

Rule = namedtuple("Rule", ["cls", "exempt_columns"])


def _segment_to_regex(segment):
    return re.escape(segment).replace(r"\*", "[^/]*")


def glob_to_regex(pattern):
    """Translate the small glob subset used above into a regex.

    Supports a trailing "/**" (zero or more path segments) and a middle
    "/**/" (zero or more whole path segments), plus "*" matching within a
    single path segment. This is deliberately not a general glob engine --
    it covers exactly the patterns in CLASSIFICATION_ROWS.
    """
    if pattern.endswith("/**"):
        prefix = _segment_to_regex(pattern[: -len("/**")])
        return re.compile(f"^{prefix}(?:/.*)?$")
    if "/**/" in pattern:
        prefix, suffix = pattern.split("/**/", 1)
        prefix_re = _segment_to_regex(prefix)
        suffix_re = _segment_to_regex(suffix)
        return re.compile(f"^{prefix_re}(?:/.*)?/{suffix_re}$")
    return re.compile(f"^{_segment_to_regex(pattern)}$")


COMPILED_RULES = [
    (glob_to_regex(pattern), Rule(cls, exempt_columns))
    for pattern, cls, exempt_columns in CLASSIFICATION_ROWS
]


def classify(rel_path):
    for regex, rule in COMPILED_RULES:
        if regex.match(rel_path):
            return rule
    return None


def relative_parquet_paths(root):
    return [
        os.path.relpath(abs_path, root).replace(os.sep, "/")
        for abs_path in glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)
    ]


def bits_of(value, byte_width):
    if value is None:
        return None
    return struct.pack(">d" if byte_width == 8 else ">f", value)


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


baseline_paths = set(relative_parquet_paths(baseline_dir))
test_paths = set(relative_parquet_paths(test_dir))
all_paths = sorted(baseline_paths | test_paths)

classified = {}
for rel_path in all_paths:
    rule = classify(rel_path)
    if rule is None:
        fail(f"{rel_path}: unclassified output file")
    classified[rel_path] = rule

non_exempt = sorted(p for p in all_paths if classified[p].cls != "exempt")
missing_from_test = [p for p in non_exempt if p not in test_paths]
missing_from_baseline = [p for p in non_exempt if p not in baseline_paths]
if missing_from_test or missing_from_baseline:
    for p in missing_from_test:
        print(f"FAIL: {p}: present in baseline, missing from test output")
    for p in missing_from_baseline:
        print(f"FAIL: {p}: present in test, missing from baseline output")
    sys.exit(1)

for rel_path in non_exempt:
    rule = classified[rel_path]
    # ParquetFile(path).read() reads the file's own columns only; pq.read_table()
    # would add a dictionary<int32> scenario_id partition column from the
    # scenario_id=NNNN/ Hive path that collides with the in-file int32 column
    # (the same read contract cobre's Python tests and native loader follow).
    baseline_table = pq.ParquetFile(os.path.join(baseline_dir, rel_path)).read()
    test_table = pq.ParquetFile(os.path.join(test_dir, rel_path)).read()

    baseline_cols = set(baseline_table.column_names)
    test_cols = set(test_table.column_names)
    if baseline_cols != test_cols:
        fail(
            f"{rel_path}: column set mismatch "
            f"(missing={sorted(baseline_cols - test_cols)}, "
            f"extra={sorted(test_cols - baseline_cols)})"
        )

    if baseline_table.num_rows != test_table.num_rows:
        fail(
            f"{rel_path}: row count mismatch "
            f"(baseline={baseline_table.num_rows}, test={test_table.num_rows})"
        )

    exempt_columns = set(rule.exempt_columns)
    for col_name in sorted(baseline_cols):
        if col_name in exempt_columns:
            continue

        field_type = baseline_table.schema.field(col_name).type
        is_float = pa.types.is_floating(field_type)
        byte_width = 4 if pa.types.is_float32(field_type) else 8

        # Index-aligned, never sorted: a row-order difference in a
        # compare-class table is a real finding, not noise to canonicalize.
        baseline_values = baseline_table.column(col_name).to_pylist()
        test_values = test_table.column(col_name).to_pylist()

        for row_idx, (baseline_value, test_value) in enumerate(
            zip(baseline_values, test_values)
        ):
            if is_float:
                baseline_repr = bits_of(baseline_value, byte_width)
                test_repr = bits_of(test_value, byte_width)
            else:
                baseline_repr = baseline_value
                test_repr = test_value

            if baseline_repr != test_repr:
                fail(
                    f"{rel_path}: column '{col_name}' row {row_idx} differs "
                    f"(baseline={baseline_value!r}, test={test_value!r})"
                )

print(f"INFO: {len(non_exempt)} classified output file(s) bit-identical")
sys.exit(0)
PYEOF
}

# ---------------------------------------------------------------------------
# compare_solver_counters <baseline_dir> <test_dir>
#
# Compares simulation/solver/iterations.parquet with the `rank` and
# `worker_id` columns and every `*_ms` column dropped. The remaining rows are
# sorted before comparison because their order follows the rank and worker
# that wrote them. Exits 1 naming the first differing row.
# ---------------------------------------------------------------------------
compare_solver_counters() {
    local baseline_dir="$1"
    local test_dir="$2"
    python3 - "${baseline_dir}" "${test_dir}" <<'PYEOF'
import os
import sys

import pyarrow.parquet as pq

REL_PATH = "simulation/solver/iterations.parquet"
TOPOLOGY_COLUMNS = ("rank", "worker_id")


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


def counter_rows(root):
    path = os.path.join(root, REL_PATH)
    if not os.path.isfile(path):
        fail(f"{path}: missing")
    table = pq.ParquetFile(path).read()
    for column in TOPOLOGY_COLUMNS:
        if column not in table.column_names:
            fail(f"{path}: column '{column}' missing")
    kept = [
        c for c in table.column_names if c not in TOPOLOGY_COLUMNS and not c.endswith("_ms")
    ]
    columns = [table.column(c).to_pylist() for c in kept]
    # A null sorts after every value, so a nullable column never compares None with a value.
    rows = sorted(zip(*columns), key=lambda row: [(v is None, v) for v in row])
    return kept, rows


baseline_columns, baseline_rows = counter_rows(sys.argv[1])
test_columns, test_rows = counter_rows(sys.argv[2])

if baseline_columns != test_columns:
    fail(f"{REL_PATH}: column mismatch (baseline={baseline_columns}, test={test_columns})")
if len(baseline_rows) != len(test_rows):
    fail(
        f"{REL_PATH}: row count mismatch "
        f"(baseline={len(baseline_rows)}, test={len(test_rows)})"
    )
for index, (baseline_row, test_row) in enumerate(zip(baseline_rows, test_rows)):
    if baseline_row != test_row:
        fail(
            f"{REL_PATH}: sorted row {index} differs "
            f"(baseline={dict(zip(baseline_columns, baseline_row))}, "
            f"test={dict(zip(test_columns, test_row))})"
        )

print(f"INFO: {len(baseline_rows)} solver counter row(s) equal")
sys.exit(0)
PYEOF
}

# ---------------------------------------------------------------------------
# assert_compare_exit <label> <baseline_dir> <case_dir> <expected_exit> [comparator]
#
# Runs <comparator> (default compare_outputs_bitwise) and asserts its exit
# code matches expectation, counting the case in SELF_CHECK_CASES and
# recording any mismatch in SELF_CHECK_FAILURES (both used by
# run_self_check).
# ---------------------------------------------------------------------------
assert_compare_exit() {
    local label="$1"
    local baseline_dir="$2"
    local case_dir="$3"
    local expected_exit="$4"
    local comparator="${5:-compare_outputs_bitwise}"
    local actual_exit=0

    SELF_CHECK_CASES=$((SELF_CHECK_CASES + 1))
    echo "--- self-check case: ${label} ---"
    "${comparator}" "${baseline_dir}" "${case_dir}" || actual_exit=$?

    if [[ "${actual_exit}" -eq "${expected_exit}" ]]; then
        echo "[self-check:${label}] PASS (exit ${actual_exit}, expected ${expected_exit})"
    else
        echo "[self-check:${label}] FAIL: exit ${actual_exit}, expected ${expected_exit}"
        SELF_CHECK_FAILURES=1
    fi
}

# ---------------------------------------------------------------------------
# run_self_check
#
# Exercises compare_outputs_bitwise against tiny generated fixtures before
# trusting it on a live run: identical dirs PASS, a single differing f64 bit
# in a compare-class column FAILs, exempt-column/exempt-path-only
# differences PASS, an unclassified parquet path FAILs, and a
# training/solver/iterations.parquet differing in `rank` and a wall-clock
# column PASSes (the whole file is exempt). Then exercises
# compare_solver_counters: rows differing only in `rank`, `worker_id`, a
# `*_ms` value and row order PASS, and one differing `basis_offered` FAILs.
# Aborts (exit 1) if any case's exit code does not match its expectation.
# ---------------------------------------------------------------------------
run_self_check() {
    local root
    root="$(mktemp -d)"

    echo ""
    echo "=== Self-check: comparator fixtures ==="

    python3 - "${root}" <<'PYEOF'
import math
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

root = sys.argv[1]


def write(path, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.table(columns), path)


def write_common(base):
    write(
        os.path.join(base, "simulation", "costs", "data.parquet"),
        {"scenario_id": [0, 0], "value": [1.0, 2.5]},
    )
    write(
        os.path.join(base, "training", "convergence.parquet"),
        {"lower_bound": [10.0, 10.0], "time_forward_ms": [5, 6]},
    )
    write(
        os.path.join(base, "training", "timing", "iterations.parquet"),
        {"iteration": [1, 2], "forward_wall_ms": [3, 4]},
    )


def write_solver_iterations(base, rank, solve_time_ms):
    # Mirrors the columns of solver_iterations_schema() relevant to the
    # exempt reclassification: `rank` (execution topology) and
    # `solve_time_ms` (wall-clock) are exactly the two column kinds that
    # made this table bitwise-incomparable across rank counts.
    write(
        os.path.join(base, "training", "solver", "iterations.parquet"),
        {
            "iteration": [1, 1],
            "phase": ["backward", "backward"],
            "stage": [0, 0],
            "opening": [0, 1],
            "rank": [rank, rank],
            "worker_id": [0, 0],
            "lp_solves": [1, 1],
            "solve_time_ms": [solve_time_ms, solve_time_ms],
        },
    )


# baseline: the reference every other case is compared against.
write_common(os.path.join(root, "baseline"))
write_solver_iterations(os.path.join(root, "baseline"), rank=0, solve_time_ms=1.0)

# case_identical: decoded values identical -> expect PASS.
write_common(os.path.join(root, "case_identical"))

# case_bitdiff: one f64 bit differs in a compare-class column -> expect FAIL.
write(
    os.path.join(root, "case_bitdiff", "simulation", "costs", "data.parquet"),
    {"scenario_id": [0, 0], "value": [1.0, math.nextafter(2.5, math.inf)]},
)
write(
    os.path.join(root, "case_bitdiff", "training", "convergence.parquet"),
    {"lower_bound": [10.0, 10.0], "time_forward_ms": [5, 6]},
)
write(
    os.path.join(root, "case_bitdiff", "training", "timing", "iterations.parquet"),
    {"iteration": [1, 2], "forward_wall_ms": [3, 4]},
)

# case_exempt_diff: differs only in exempt columns/paths -> expect PASS.
write(
    os.path.join(root, "case_exempt_diff", "simulation", "costs", "data.parquet"),
    {"scenario_id": [0, 0], "value": [1.0, 2.5]},
)
write(
    os.path.join(root, "case_exempt_diff", "training", "convergence.parquet"),
    {"lower_bound": [10.0, 10.0], "time_forward_ms": [999, 999]},
)
write(
    os.path.join(root, "case_exempt_diff", "training", "timing", "iterations.parquet"),
    {"iteration": [1, 2, 3], "forward_wall_ms": [100, 200, 300]},
)

# case_unclassified: identical to baseline plus one path matching no rule.
write_common(os.path.join(root, "case_unclassified"))
write(os.path.join(root, "case_unclassified", "random", "unclassified.parquet"), {"x": [1]})

# case_exempt_solver_iterations: training/solver/iterations.parquet differs
# in `rank` and in the wall-clock `solve_time_ms` column -> expect PASS
# (the whole file is exempt, so neither difference is ever compared).
write_common(os.path.join(root, "case_exempt_solver_iterations"))
write_solver_iterations(
    os.path.join(root, "case_exempt_solver_iterations"), rank=1, solve_time_ms=999.0
)


def write_simulation_solver_iterations(base, rows):
    names = ["scenario_id", "phase", "rank", "worker_id", "basis_offered", "solve_time_ms"]
    write(
        os.path.join(base, "simulation", "solver", "iterations.parquet"),
        {name: [row[i] for row in rows] for i, name in enumerate(names)},
    )


write_simulation_solver_iterations(
    os.path.join(root, "counters_baseline"),
    [(0, "simulation", 0, 0, 2, 1.0), (1, "simulation", 0, 1, 2, 1.5)],
)

# counters_layout_diff: rows differ only in rank, worker_id, a *_ms value
# and row order -> expect PASS.
write_simulation_solver_iterations(
    os.path.join(root, "counters_layout_diff"),
    [(1, "simulation", 1, 0, 2, 1.5), (0, "simulation", 0, 0, 2, 999.0)],
)

# counters_basis_diff: one basis_offered value differs -> expect FAIL.
write_simulation_solver_iterations(
    os.path.join(root, "counters_basis_diff"),
    [(0, "simulation", 0, 0, 2, 1.0), (1, "simulation", 0, 1, 0, 1.5)],
)
PYEOF

    SELF_CHECK_CASES=0
    SELF_CHECK_FAILURES=0
    assert_compare_exit "identical" "${root}/baseline" "${root}/case_identical" 0
    assert_compare_exit "bitdiff" "${root}/baseline" "${root}/case_bitdiff" 1
    assert_compare_exit "exempt_diff" "${root}/baseline" "${root}/case_exempt_diff" 0
    assert_compare_exit "unclassified" "${root}/baseline" "${root}/case_unclassified" 1
    assert_compare_exit "exempt_solver_iterations" "${root}/baseline" \
        "${root}/case_exempt_solver_iterations" 0
    assert_compare_exit "counters_layout_diff" "${root}/counters_baseline" \
        "${root}/counters_layout_diff" 0 compare_solver_counters
    assert_compare_exit "counters_basis_diff" "${root}/counters_baseline" \
        "${root}/counters_basis_diff" 1 compare_solver_counters

    rm -rf "${root}"

    if [[ "${SELF_CHECK_FAILURES}" -ne 0 ]]; then
        echo "ABORT: comparator self-check failed — see above"
        exit 1
    fi

    echo "INFO: comparator self-check passed (${SELF_CHECK_CASES}/${SELF_CHECK_CASES} cases)"
}

# ---------------------------------------------------------------------------
# prepare_partial_misfit_case <src_case> <dst_case>
#
# Copies <src_case> to <dst_case>, disables training, and splits the single
# block of stages 4 to 7 into two blocks of half its hours. A policy trained
# on <src_case> then holds stored bases that fit every stage but 4 to 7.
# ---------------------------------------------------------------------------
prepare_partial_misfit_case() {
    local src_case="$1"
    local dst_case="$2"
    mkdir -p "${dst_case}"
    cp -R "${src_case}/." "${dst_case}/"
    python3 - "${dst_case}" <<'PYEOF'
import json
import os
import sys

SPLIT_STAGE_IDS = {4, 5, 6, 7}

case_dir = sys.argv[1]


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


def load(name):
    path = os.path.join(case_dir, name)
    if not os.path.isfile(path):
        fail(f"{path}: missing")
    with open(path) as f:
        return path, json.load(f)


def dump(path, document):
    with open(path, "w") as f:
        json.dump(document, f, indent=2)
        f.write("\n")


config_path, config = load("config.json")
config.setdefault("training", {})["enabled"] = False
dump(config_path, config)

stages_path, stages = load("stages.json")
split = set()
for stage in stages["stages"]:
    if stage["id"] not in SPLIT_STAGE_IDS:
        continue
    if len(stage["blocks"]) != 1:
        fail(f"{stages_path}: stage {stage['id']} has {len(stage['blocks'])} blocks, expected 1")
    hours = stage["blocks"][0]["hours"]
    stage["blocks"] = [
        {"id": 0, "name": "A", "hours": hours / 2},
        {"id": 1, "name": "B", "hours": hours / 2},
    ]
    split.add(stage["id"])
if split != SPLIT_STAGE_IDS:
    fail(f"{stages_path}: stages {sorted(SPLIT_STAGE_IDS - split)} not found")
dump(stages_path, stages)
PYEOF
}

# ---------------------------------------------------------------------------
# seed_policy <policy_dir> <output_dir>
#
# Copies <policy_dir> into <output_dir>/policy, where a simulation-only run
# reads its policy.
# ---------------------------------------------------------------------------
seed_policy() {
    local policy_dir="$1"
    local output_dir="$2"
    mkdir -p "${output_dir}/policy"
    cp -R "${policy_dir}/." "${output_dir}/policy/"
}

# ---------------------------------------------------------------------------
# check_outputs <test_name> <exit_code> <output_dir> <baseline_dir> <marker> <seeded_policy>
#
# Verifies:
#   1. Exit code is 0 (or reports timeout on code 124)
#   2. Output directory exists
#   3. policy/manifest.bin exists (the self-describing checkpoint commit signal)
#   4. <marker> exists (training/convergence.parquet for a training run,
#      simulation/_SUCCESS for a simulation-only run)
#   5. Every classified output is bit-identical to the baseline directory
#   6. When <seeded_policy> is non-empty, the per-scenario solver counters
#      equal the baseline's
#
# Returns 0 on full pass, 1 on any failure. Prints diagnostics on failure.
# ---------------------------------------------------------------------------
check_outputs() {
    local test_name="$1"
    local exit_code="$2"
    local output_dir="$3"
    local baseline_dir="$4"
    local marker="$5"
    local seeded_policy="$6"

    if [[ "${exit_code}" -eq 124 ]]; then
        echo "[${test_name}] FAIL: timed out after ${TIMEOUT}s (exit 124)"
        echo "[${test_name}] NOTE: exit code 124 is the primary indicator of the MPICH 4.3.0 PMI2 deadlock"
        return 1
    fi

    if [[ "${exit_code}" -ne 0 ]]; then
        echo "[${test_name}] FAIL: command exited with code ${exit_code}"
        return 1
    fi

    if [[ ! -d "${output_dir}" ]]; then
        echo "[${test_name}] FAIL: output directory does not exist: ${output_dir}"
        return 1
    fi

    local policy_manifest="${output_dir}/policy/manifest.bin"
    if [[ ! -f "${policy_manifest}" ]]; then
        echo "[${test_name}] FAIL: policy/manifest.bin missing"
        echo "[${test_name}] Contents of ${output_dir}:"
        ls -la "${output_dir}" 2>&1 || true
        return 1
    fi

    local marker_path="${output_dir}/${marker}"
    if [[ ! -f "${marker_path}" ]]; then
        local marker_dir
        marker_dir="$(dirname "${marker_path}")"
        echo "[${test_name}] FAIL: ${marker} missing"
        echo "[${test_name}] Contents of ${marker_dir}:"
        ls -la "${marker_dir}" 2>&1 || true
        return 1
    fi

    if [[ -n "${baseline_dir}" ]]; then
        echo "[${test_name}] Comparing classified outputs against ${baseline_dir} (bitwise) ..."
        if ! compare_outputs_bitwise "${baseline_dir}" "${output_dir}"; then
            echo "[${test_name}] FAIL: bitwise output comparison failed (see above)"
            return 1
        fi
        if [[ -n "${seeded_policy}" ]]; then
            echo "[${test_name}] Comparing per-scenario solver counters against ${baseline_dir} ..."
            if ! compare_solver_counters "${baseline_dir}" "${output_dir}"; then
                echo "[${test_name}] FAIL: solver counter comparison failed (see above)"
                return 1
            fi
        fi
    fi

    return 0
}

# ---------------------------------------------------------------------------
# run_mpiexec_test <test_name> <case_dir> <baseline_tag> <marker> <policy_src>
#                  <n_ranks> <extra_args...>
#
# Seeds $WORK_DIR/<test_name_lower> with <policy_src> when it is non-empty,
# then runs: timeout $TIMEOUT mpiexec -n <n_ranks> $COBRE_BIN run <case_dir>
#              --output $WORK_DIR/<test_name_lower> --quiet [extra_args]
# Calls check_outputs against $WORK_DIR/<baseline_tag> (none when the test
# is the baseline itself) and updates RESULTS.
# ---------------------------------------------------------------------------
run_mpiexec_test() {
    local test_name="$1"
    local case_dir="$2"
    local baseline_tag="$3"
    local marker="$4"
    local policy_src="$5"
    local n_ranks="$6"
    shift 6
    local extra_args=("$@")

    local tag
    tag="$(echo "${test_name}" | tr '[:upper:]' '[:lower:]')"
    local output_dir="${WORK_DIR}/${tag}"

    echo ""
    echo "=== ${test_name}: mpiexec -n ${n_ranks} ==="

    if [[ -n "${policy_src}" ]]; then
        seed_policy "${policy_src}" "${output_dir}"
    fi

    local exit_code=0
    timeout "${TIMEOUT}" mpiexec -n "${n_ranks}" \
        "${COBRE_BIN}" run "${case_dir}" --output "${output_dir}" --quiet \
        "${extra_args[@]+"${extra_args[@]}"}" || exit_code=$?

    local baseline=""
    if [[ "${tag}" != "${baseline_tag}" ]]; then
        baseline="${WORK_DIR}/${baseline_tag}"
    fi

    if check_outputs "${test_name}" "${exit_code}" "${output_dir}" "${baseline}" \
        "${marker}" "${policy_src}"; then
        echo "[${test_name}] PASS"
        RESULTS["${test_name}"]="PASS"
    else
        RESULTS["${test_name}"]="FAIL"
        print_summary
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# run_sbatch_test <test_name> <case_dir> <baseline_tag> <marker> <policy_src>
#                 <sbatch_N> <sbatch_n> <mpiexec_n> <extra_args...>
#
# Seeds the output directory with <policy_src> when it is non-empty,
# generates a batch script as a heredoc, submits via sbatch --wait, then
# calls check_outputs against $WORK_DIR/<baseline_tag>. On failure, prints
# SLURM .out/.err logs.
# ---------------------------------------------------------------------------
run_sbatch_test() {
    local test_name="$1"
    local case_dir="$2"
    local baseline_tag="$3"
    local marker="$4"
    local policy_src="$5"
    local sbatch_N="$6"
    local sbatch_n="$7"
    local mpiexec_n="$8"
    shift 8
    local extra_args=("$@")

    local tag
    tag="$(echo "${test_name}" | tr '[:upper:]' '[:lower:]')"
    local output_dir="${WORK_DIR}/${tag}"
    local slurm_out="${WORK_DIR}/${tag}.slurm.out"
    local slurm_err="${WORK_DIR}/${tag}.slurm.err"
    local batch_script="${WORK_DIR}/${tag}.sbatch"

    echo ""
    echo "=== ${test_name}: sbatch -N ${sbatch_N} -n ${sbatch_n} (mpiexec -n ${mpiexec_n}) ==="

    if [[ -n "${policy_src}" ]]; then
        seed_policy "${policy_src}" "${output_dir}"
    fi

    # Build the extra args string for insertion into the batch script.
    # Paths are absolute; this heredoc is unquoted so variables expand.
    local extra_str="${extra_args[*]+"${extra_args[*]}"}"

    # Generate the batch script with absolute paths.
    # The heredoc delimiter is unquoted so WORK_DIR, COBRE_BIN, case_dir,
    # output_dir, and mpiexec_n expand at generation time.
    cat > "${batch_script}" << SBEOF
#!/bin/bash
export PATH=/opt/mpich/bin:\$PATH
mpiexec -n ${mpiexec_n} ${COBRE_BIN} run ${case_dir} --output ${output_dir} --quiet ${extra_str}
SBEOF

    chmod +x "${batch_script}"

    local exit_code=0
    timeout "${TIMEOUT}" sbatch --wait \
        -N "${sbatch_N}" \
        -n "${sbatch_n}" \
        --output="${slurm_out}" \
        --error="${slurm_err}" \
        "${batch_script}" || exit_code=$?

    local baseline="${WORK_DIR}/${baseline_tag}"

    if check_outputs "${test_name}" "${exit_code}" "${output_dir}" "${baseline}" \
        "${marker}" "${policy_src}"; then
        echo "[${test_name}] PASS"
        RESULTS["${test_name}"]="PASS"
    else
        RESULTS["${test_name}"]="FAIL"
        # Print SLURM logs for debugging before exiting.
        if [[ -f "${slurm_out}" ]]; then
            echo ""
            echo "--- SLURM stdout (${slurm_out}) ---"
            cat "${slurm_out}"
        fi
        if [[ -f "${slurm_err}" ]]; then
            echo ""
            echo "--- SLURM stderr (${slurm_err}) ---"
            cat "${slurm_err}"
        fi
        # For T4 multi-node failures, also print any Hydra proxy logs that
        # may have been captured in the SLURM output or work directory.
        if [[ "${test_name}" == "T4" ]]; then
            echo ""
            echo "--- T4 diagnostic: listing ${output_dir} (if it exists) ---"
            ls -la "${output_dir}" 2>&1 || echo "(directory missing)"
            echo "--- T4 diagnostic: Hydra proxy logs (if any in ${WORK_DIR}) ---"
            find "${WORK_DIR}" -name "hydra_pmi_proxy*" -o -name "*.hydra" 2>/dev/null \
                | while read -r f; do
                    echo "  ${f}:"
                    cat "${f}" 2>/dev/null || true
                done || true
        fi
        print_summary
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# print_summary
#
# Prints a pass/fail table of the tests attempted for the active MODE
# (T1, T2, S1 and S2 for "reduced", every T and S case for "full") so a
# "reduced" run is not marked FAILED merely for never attempting the sbatch
# cases.
# ---------------------------------------------------------------------------
print_summary() {
    local tests_to_check=(T1 T2)
    if [[ "${MODE}" == "full" ]]; then
        tests_to_check+=(T3 T4 T5)
    fi
    tests_to_check+=(S1 S2)
    if [[ "${MODE}" == "full" ]]; then
        tests_to_check+=(S3 S4)
    fi

    echo ""
    echo "=============================="
    echo " SLURM MPI Test Suite Summary (mode: ${MODE})"
    echo "=============================="
    local all_pass=1
    for t in "${tests_to_check[@]}"; do
        local status="${RESULTS[${t}]:-NOT RUN}"
        printf "  %-4s  %s\n" "${t}" "${status}"
        if [[ "${status}" != "PASS" ]]; then
            all_pass=0
        fi
    done
    echo "=============================="
    if [[ "${all_pass}" -eq 1 ]]; then
        echo " Result: ALL PASS"
    else
        echo " Result: FAILED"
    fi
    echo "=============================="
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    parse_mode "$@"

    preflight_check
    run_self_check

    # Case directory, baseline tag, completion marker, policy to seed.
    local training_run=("${CASE_DIR}" t1 training/convergence.parquet "")
    local simulation_run=(
        "${WORK_DIR}/4ree-partial-misfit" s1 simulation/_SUCCESS "${WORK_DIR}/t1/policy"
    )

    # T1: mpiexec single rank (baseline — no extra args)
    run_mpiexec_test "T1" "${training_run[@]}" 1

    # T2: mpiexec multi-rank single machine
    run_mpiexec_test "T2" "${training_run[@]}" 2 --threads 2

    if [[ "${MODE}" == "full" ]]; then
        # T3: sbatch single-node, 2 ranks
        run_sbatch_test "T3" "${training_run[@]}" 1 2 2

        # T4: sbatch multi-node, 1 rank per node (critical PMI2 deadlock test)
        run_sbatch_test "T4" "${training_run[@]}" 2 2 2

        # T5: sbatch multi-node, 2 ranks per node
        run_sbatch_test "T5" "${training_run[@]}" 2 4 4 --threads 1
    fi

    prepare_partial_misfit_case "${CASE_DIR}" "${simulation_run[0]}"

    # S1: mpiexec single rank, single thread (simulation-only baseline)
    run_mpiexec_test "S1" "${simulation_run[@]}" 1 --threads 1

    # S2: mpiexec multi-rank single machine
    run_mpiexec_test "S2" "${simulation_run[@]}" 2 --threads 2

    if [[ "${MODE}" == "full" ]]; then
        # S3: sbatch multi-node, 1 rank per node
        run_sbatch_test "S3" "${simulation_run[@]}" 2 2 2

        # S4: sbatch multi-node, 2 ranks per node
        run_sbatch_test "S4" "${simulation_run[@]}" 2 4 4 --threads 1
    fi

    print_summary
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
