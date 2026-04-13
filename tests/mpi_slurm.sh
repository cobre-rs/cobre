#!/usr/bin/env bash
# End-to-end SLURM MPI integration test for `cobre run`.
#
# Single-script entrypoint that:
#   1. Builds the cobre binary with --features mpi
#   2. Builds and starts a 2-node Docker SLURM cluster
#   3. Copies the binary, test case, and test script into the cluster
#   4. Runs the 5-test suite inside the cluster
#   5. Tears down the cluster (always, even on failure)
#
# Prerequisites:
#   - docker and docker compose
#   - cargo (Rust toolchain)
#   - MPICH (mpicc in PATH for the cargo build)
#
# Usage:
#   ./tests/mpi_slurm.sh              # full run
#   ./tests/mpi_slurm.sh --no-build   # skip cargo build and docker build
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/tests/slurm/docker-compose.yml"
STAGING_DIR="${REPO_ROOT}/tests/slurm/staging"

NO_BUILD=false
for arg in "$@"; do
    case "$arg" in
        --no-build) NO_BUILD=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Cleanup: always tear down the cluster on exit
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "INFO: Tearing down SLURM cluster..."
    docker compose -f "${COMPOSE_FILE}" down -v 2>/dev/null || true
}
trap cleanup EXIT

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Step 1: Build the cobre binary
# ---------------------------------------------------------------------------
if [[ "${NO_BUILD}" == "false" ]]; then
    echo "INFO: Building cobre with --features mpi ..."
    cargo build --release --features mpi -p cobre-cli
    echo ""
    echo "INFO: Building Docker SLURM cluster image..."
    docker compose -f "${COMPOSE_FILE}" build
else
    echo "INFO: Skipping builds (--no-build)"
fi

# ---------------------------------------------------------------------------
# Step 2: Find the binary
# ---------------------------------------------------------------------------
if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
    COBRE_BIN="${CARGO_TARGET_DIR}/release/cobre"
elif [[ -f "${REPO_ROOT}/target/release/cobre" ]]; then
    COBRE_BIN="${REPO_ROOT}/target/release/cobre"
else
    echo "FAIL: Cannot find cobre binary. Set CARGO_TARGET_DIR or build first."
    exit 1
fi
echo "INFO: Binary: ${COBRE_BIN}"

# ---------------------------------------------------------------------------
# Step 3: Start the SLURM cluster
# ---------------------------------------------------------------------------
echo ""
echo "INFO: Starting 2-node SLURM cluster..."
docker compose -f "${COMPOSE_FILE}" up -d --wait
echo "INFO: Cluster is healthy."

# ---------------------------------------------------------------------------
# Step 4: Copy artifacts into the cluster
# ---------------------------------------------------------------------------
echo "INFO: Copying artifacts into cluster..."
mkdir -p "${STAGING_DIR}"
cp "${COBRE_BIN}" "${STAGING_DIR}/cobre-mpi"
cp -r "${REPO_ROOT}/examples/4ree" "${STAGING_DIR}/4ree"

docker compose -f "${COMPOSE_FILE}" cp "${STAGING_DIR}/cobre-mpi" controller:/shared/cobre-mpi
docker compose -f "${COMPOSE_FILE}" cp "${STAGING_DIR}/4ree" controller:/shared/4ree
docker compose -f "${COMPOSE_FILE}" cp "${REPO_ROOT}/tests/slurm/run-tests.sh" controller:/shared/run-tests.sh
docker compose -f "${COMPOSE_FILE}" exec controller chmod +x /shared/cobre-mpi /shared/run-tests.sh

# ---------------------------------------------------------------------------
# Step 5: Run the test suite
# ---------------------------------------------------------------------------
echo ""
echo "INFO: Running SLURM MPI integration tests..."
echo "======================================"
docker compose -f "${COMPOSE_FILE}" exec controller bash /shared/run-tests.sh
RESULT=$?
echo "======================================"

# ---------------------------------------------------------------------------
# Step 6: Diagnostic output on failure
# ---------------------------------------------------------------------------
if [[ "${RESULT}" -ne 0 ]]; then
    echo ""
    echo "INFO: Dumping SLURM logs for diagnostics..."
    docker compose -f "${COMPOSE_FILE}" logs 2>/dev/null || true
    echo ""
    echo "INFO: SLURM node status:"
    docker compose -f "${COMPOSE_FILE}" exec controller sinfo -N 2>/dev/null || true
fi

exit "${RESULT}"
