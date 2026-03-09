#!/usr/bin/env bash
# MPI smoke test for `cobre run`.
#
# Builds the cobre binary with --features mpi, runs both a single-process
# baseline and a 2-rank MPI execution against the 1dtoy example, then
# verifies:
#   1. Exit code is 0 for both runs.
#   2. Output files exist (policy checkpoint, training results).
#   3. File counts match (no rank-multiplied duplicates).
#   4. Final lower bound is numerically equivalent within 1e-6 relative
#      tolerance.
#
# The test is skipped gracefully when mpiexec is not available.
set -euo pipefail

# ---------------------------------------------------------------------------
# Locate the repository root relative to this script's location.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Skip if mpiexec is unavailable.
# ---------------------------------------------------------------------------
if ! command -v mpiexec &>/dev/null; then
    echo "SKIP: mpiexec not found in PATH"
    exit 0
fi

# ---------------------------------------------------------------------------
# Detect MPI implementation to decide whether --oversubscribe is needed.
# OpenMPI requires --oversubscribe when running more ranks than physical
# cores. MPICH and derivatives do not support (and reject) that flag.
# ---------------------------------------------------------------------------
MPI_EXTRA_FLAGS=""
if mpiexec --version 2>&1 | grep -qi "open.mpi\|openmpi"; then
    MPI_EXTRA_FLAGS="--oversubscribe"
fi

echo "INFO: MPI implementation detected: $(mpiexec --version 2>&1 | head -1)"
echo "INFO: Extra mpiexec flags: '${MPI_EXTRA_FLAGS}'"

# ---------------------------------------------------------------------------
# Build the cobre binary with the mpi feature enabled.
# Change to repo root so cargo picks up the workspace Cargo.toml.
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"
echo "INFO: Building cobre with --features mpi ..."
cargo build -p cobre-cli --features mpi 2>&1

COBRE_BIN="${REPO_ROOT}/target/debug/cobre"
CASE_DIR="${REPO_ROOT}/examples/1dtoy"

if [ ! -f "${COBRE_BIN}" ]; then
    echo "FAIL: cobre binary not found at ${COBRE_BIN}"
    exit 1
fi

if [ ! -d "${CASE_DIR}" ]; then
    echo "FAIL: 1dtoy case directory not found at ${CASE_DIR}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Create a temporary directory for output; clean up on exit.
# ---------------------------------------------------------------------------
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

SINGLE_OUT="${WORK_DIR}/single"
MPI_OUT="${WORK_DIR}/mpi"

# ---------------------------------------------------------------------------
# Run single-process baseline.
# ---------------------------------------------------------------------------
echo "INFO: Running single-process baseline ..."
"${COBRE_BIN}" run "${CASE_DIR}" --output "${SINGLE_OUT}" --quiet
SINGLE_EXIT=$?

if [ "${SINGLE_EXIT}" -ne 0 ]; then
    echo "FAIL: single-process run exited with code ${SINGLE_EXIT}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Run 2-rank MPI execution.
# Using word-splitting intentionally for MPI_EXTRA_FLAGS (may be empty).
# ---------------------------------------------------------------------------
echo "INFO: Running MPI 2-rank execution ..."
# shellcheck disable=SC2086
mpiexec -np 2 ${MPI_EXTRA_FLAGS} \
    "${COBRE_BIN}" run "${CASE_DIR}" --output "${MPI_OUT}" --quiet
MPI_EXIT=$?

if [ "${MPI_EXIT}" -ne 0 ]; then
    echo "FAIL: MPI 2-rank run exited with code ${MPI_EXIT}"
    exit 1
fi

# ---------------------------------------------------------------------------
# AC-1: MPI output directory exists.
# ---------------------------------------------------------------------------
if [ ! -d "${MPI_OUT}" ]; then
    echo "FAIL: MPI output directory does not exist: ${MPI_OUT}"
    exit 1
fi

# ---------------------------------------------------------------------------
# AC-2: Policy checkpoint exists in MPI output.
# ---------------------------------------------------------------------------
if [ ! -d "${MPI_OUT}/policy" ]; then
    echo "FAIL: policy checkpoint directory missing in MPI output"
    exit 1
fi

POLICY_META="${MPI_OUT}/policy/metadata.json"
if [ ! -f "${POLICY_META}" ]; then
    echo "FAIL: policy/metadata.json missing in MPI output"
    exit 1
fi

# ---------------------------------------------------------------------------
# AC-3: Training convergence parquet exists in MPI output.
# ---------------------------------------------------------------------------
MPI_CONVERGENCE="${MPI_OUT}/training/convergence.parquet"
if [ ! -f "${MPI_CONVERGENCE}" ]; then
    echo "FAIL: training/convergence.parquet missing in MPI output"
    exit 1
fi

# ---------------------------------------------------------------------------
# AC-4: File counts match (no rank-multiplied duplicates).
# ---------------------------------------------------------------------------
SINGLE_COUNT=$(find "${SINGLE_OUT}" -type f | wc -l)
MPI_COUNT=$(find "${MPI_OUT}" -type f | wc -l)

if [ "${SINGLE_COUNT}" -ne "${MPI_COUNT}" ]; then
    echo "FAIL: file count mismatch (single=${SINGLE_COUNT}, mpi=${MPI_COUNT})"
    echo "  Single output files:"
    find "${SINGLE_OUT}" -type f | sort
    echo "  MPI output files:"
    find "${MPI_OUT}" -type f | sort
    exit 1
fi

echo "INFO: File counts match: ${SINGLE_COUNT} files in each output directory"

# ---------------------------------------------------------------------------
# AC-5: Final lower bound matches within 1e-6 relative tolerance.
# Requires Python 3 with pyarrow. Uses a uv-managed venv if uv is available.
# ---------------------------------------------------------------------------
SINGLE_CONVERGENCE="${SINGLE_OUT}/training/convergence.parquet"

VENV_DIR="${REPO_ROOT}/.venv-mpi-smoke"
PYTHON=""

# Bootstrap or reuse a uv-managed venv with pyarrow.
if command -v uv &>/dev/null; then
    if [ ! -d "${VENV_DIR}" ]; then
        echo "INFO: Creating venv with uv at ${VENV_DIR} ..."
        uv venv "${VENV_DIR}"
        uv pip install --python "${VENV_DIR}/bin/python" pyarrow
    fi
    PYTHON="${VENV_DIR}/bin/python"
elif [ -d "${VENV_DIR}" ] && [ -x "${VENV_DIR}/bin/python" ]; then
    # Reuse an existing venv even if uv is gone.
    PYTHON="${VENV_DIR}/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
fi

compare_lower_bounds() {
    local single_parquet="$1"
    local mpi_parquet="$2"
    "${PYTHON}" - "${single_parquet}" "${mpi_parquet}" <<'PYEOF'
import sys
import pyarrow.parquet as pq

single_path = sys.argv[1]
mpi_path = sys.argv[2]

single_table = pq.read_table(single_path)
mpi_table = pq.read_table(mpi_path)

single_lbs = single_table.column("lower_bound").to_pylist()
mpi_lbs = mpi_table.column("lower_bound").to_pylist()

if not single_lbs:
    print("FAIL: single-process convergence.parquet has no lower_bound rows")
    sys.exit(1)

if not mpi_lbs:
    print("FAIL: MPI convergence.parquet has no lower_bound rows")
    sys.exit(1)

single_final = single_lbs[-1]
mpi_final = mpi_lbs[-1]

TOL = 1e-6
denom = max(abs(single_final), 1e-12)
rel_diff = abs(single_final - mpi_final) / denom

if rel_diff > TOL:
    print(
        f"FAIL: final lower bounds differ beyond tolerance "
        f"(single={single_final}, mpi={mpi_final}, rel_diff={rel_diff:.2e}, tol={TOL:.2e})"
    )
    sys.exit(1)

print(
    f"INFO: final lower bounds match "
    f"(single={single_final:.6f}, mpi={mpi_final:.6f}, rel_diff={rel_diff:.2e})"
)
sys.exit(0)
PYEOF
}

if [ -n "${PYTHON}" ] && "${PYTHON}" -c "import pyarrow.parquet" 2>/dev/null; then
    echo "INFO: Comparing final lower bounds via pyarrow (${PYTHON}) ..."
    if ! compare_lower_bounds "${SINGLE_CONVERGENCE}" "${MPI_CONVERGENCE}"; then
        echo "FAIL: lower bound comparison failed (see above)"
        exit 1
    fi
else
    echo "SKIP: python3/pyarrow not available -- skipping numeric lower-bound comparison"
fi

echo "PASS: MPI smoke test"
