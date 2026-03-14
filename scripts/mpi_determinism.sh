#!/usr/bin/env bash
# mpi_determinism.sh
#
# Runs the 4ree example case with 1 rank and 2 ranks, then compares the
# resulting convergence.parquet files byte-by-byte to verify that the
# upper-bound summation fix in ticket-009 produces bit-identical output
# regardless of MPI rank count.
#
# Usage:
#   cargo build --release --features mpi
#   bash scripts/mpi_determinism.sh
#
# Exit codes:
#   0  -- convergence.parquet files are byte-identical across 1 and 2 ranks
#   1  -- convergence.parquet files differ between 1 and 2 ranks
#   2  -- a prerequisite is missing (mpirun or cobre binary)
#
# This is a manual verification script intended for machines with MPI
# installed. It is NOT intended for CI execution.

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

COBRE_BIN="${REPO_ROOT}/target/release/cobre"
CASE_DIR="${REPO_ROOT}/examples/4ree"
OUT_DIR="${REPO_ROOT}/target/mpi_determinism"

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

timestamp() {
    date '+%Y-%m-%dT%H:%M:%S'
}

log() {
    echo "[$(timestamp)] $*"
}

error() {
    echo "[$(timestamp)] ERROR: $*" >&2
}

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

check_prerequisites() {
    if ! command -v mpirun >/dev/null 2>&1; then
        error "mpirun not found. Install OpenMPI or MPICH and ensure mpirun is on PATH."
        exit 2
    fi

    if [[ ! -x "${COBRE_BIN}" ]]; then
        error "target/release/cobre not found. Build with: cargo build --release --features mpi"
        exit 2
    fi
}

# ---------------------------------------------------------------------------
# Run a single configuration
# ---------------------------------------------------------------------------

run_rank1() {
    local out="${OUT_DIR}/rank1"
    log "--- Configuration: 1 MPI rank ---"
    log "Output directory: ${out}"
    mkdir -p "${out}"
    log "Starting cobre run (single process) ..."
    local rc=0
    "${COBRE_BIN}" run "${CASE_DIR}" --output "${out}" || rc=$?
    if [[ "${rc}" -ne 0 ]]; then
        error "cobre run failed for 1 rank (exit code ${rc})."
        exit "${rc}"
    fi
    log "Completed: 1 rank"
}

run_rank2() {
    local out="${OUT_DIR}/rank2"
    log "--- Configuration: 2 MPI ranks ---"
    log "Output directory: ${out}"
    mkdir -p "${out}"
    log "Starting cobre run with mpirun -np 2 ..."
    local rc=0
    mpirun -np 2 "${COBRE_BIN}" run "${CASE_DIR}" --output "${out}" || rc=$?
    if [[ "${rc}" -ne 0 ]]; then
        error "cobre run failed for 2 ranks (exit code ${rc})."
        exit "${rc}"
    fi
    log "Completed: 2 ranks"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    log "============================================================"
    log "Cobre MPI determinism smoke test"
    log "Repo root : ${REPO_ROOT}"
    log "Case      : ${CASE_DIR}"
    log "Output    : ${OUT_DIR}"
    log "============================================================"

    check_prerequisites

    # Remove output directories from any previous run to avoid stale data.
    rm -rf "${OUT_DIR}"

    run_rank1
    run_rank2

    log "============================================================"
    log "Comparing convergence.parquet: 1 rank vs 2 ranks"
    log "============================================================"

    # convergence.parquet is written under the training/ subdirectory of the
    # output root. The ticket spec listed a flat path, but the cobre binary
    # nests training outputs under training/.
    local file1="${OUT_DIR}/rank1/training/convergence.parquet"
    local file2="${OUT_DIR}/rank2/training/convergence.parquet"

    if [[ ! -f "${file1}" ]]; then
        error "Expected output file not found: ${file1}"
        exit 1
    fi
    if [[ ! -f "${file2}" ]]; then
        error "Expected output file not found: ${file2}"
        exit 1
    fi

    if cmp --silent "${file1}" "${file2}"; then
        log "PASS: convergence.parquet files are byte-identical across 1 and 2 ranks"
        exit 0
    else
        log "FAIL: convergence.parquet files differ between 1 and 2 ranks"
        exit 1
    fi
}

main "$@"
