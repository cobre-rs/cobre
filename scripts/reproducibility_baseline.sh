#!/usr/bin/env bash
# reproducibility_baseline.sh
#
# Runs the 4ree example case under three MPI configurations (1, 2, and 4 ranks)
# and compares the resulting convergence.parquet files byte-for-byte.
#
# Usage:
#   cargo build --release --features mpi
#   bash scripts/reproducibility_baseline.sh
#
# Exit codes:
#   0  -- all convergence.parquet files are byte-identical across configurations
#   1  -- at least one pair of convergence.parquet files differs
#   2  -- a prerequisite is missing (mpirun, cobre binary, or case directory)
#
# This is a manual verification script for empirically detecting floating-point
# non-associativity in the upper-bound allreduce across different MPI rank counts.
# It is NOT intended for CI execution.

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

COBRE_BIN="${REPO_ROOT}/target/release/cobre"
CASE_DIR="${REPO_ROOT}/examples/4ree"
OUTPUT_BASE="${REPO_ROOT}/target/reproducibility"

RANK_COUNTS=(1 2 4)

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
    local ok=true

    if ! command -v mpirun >/dev/null 2>&1; then
        error "mpirun not found. Install OpenMPI or MPICH and ensure mpirun is on PATH."
        ok=false
    fi

    if [[ ! -x "${COBRE_BIN}" ]]; then
        error "cobre binary not found at ${COBRE_BIN}."
        error "Build it with: cargo build --release --features mpi"
        ok=false
    fi

    if [[ ! -d "${CASE_DIR}" ]]; then
        error "4ree case directory not found at ${CASE_DIR}."
        ok=false
    fi

    # Verify the binary was compiled with MPI support by checking for the
    # ferrompi symbol. If MPI support is absent the binary ignores mpirun -np N
    # and always runs as 1 rank, which would make all runs identical for the
    # wrong reason and produce a false-pass result.
    if command -v nm >/dev/null 2>&1; then
        if ! nm "${COBRE_BIN}" 2>/dev/null | grep -qi "mpi\|ferrompi"; then
            error "cobre binary does not appear to include MPI support."
            error "Rebuild with: cargo build --release --features mpi"
            ok=false
        fi
    else
        log "WARNING: nm not available -- cannot verify MPI feature compilation."
        log "         Ensure the binary was built with --features mpi."
    fi

    if [[ "${ok}" == false ]]; then
        exit 2
    fi
}

# ---------------------------------------------------------------------------
# Run a single MPI configuration
# ---------------------------------------------------------------------------

run_configuration() {
    local np="$1"
    local out_dir="${OUTPUT_BASE}/rank${np}"

    log "--- Configuration: ${np} MPI rank(s) ---"
    log "Output directory: ${out_dir}"

    mkdir -p "${out_dir}"

    log "Starting cobre run with mpirun -np ${np} ..."
    local rc=0
    mpirun -np "${np}" "${COBRE_BIN}" run "${CASE_DIR}" --output "${out_dir}" || rc=$?
    if [[ "${rc}" -ne 0 ]]; then
        error "cobre run failed for np=${np} (exit code ${rc})."
        exit "${rc}"
    fi

    log "Completed: np=${np}"
}

# ---------------------------------------------------------------------------
# Compare convergence.parquet files
# ---------------------------------------------------------------------------

compare_files() {
    local file_a="$1"
    local label_a="$2"
    local file_b="$3"
    local label_b="$4"

    local hash_a hash_b

    if [[ ! -f "${file_a}" ]]; then
        error "Expected output file not found: ${file_a}"
        return 1
    fi
    if [[ ! -f "${file_b}" ]]; then
        error "Expected output file not found: ${file_b}"
        return 1
    fi

    hash_a="$(sha256sum "${file_a}" | awk '{print $1}')"
    hash_b="$(sha256sum "${file_b}" | awk '{print $1}')"

    if [[ "${hash_a}" == "${hash_b}" ]]; then
        echo "  ${label_a} vs ${label_b}: identical  (sha256=${hash_a})"
        return 0
    else
        echo "  ${label_a} vs ${label_b}: DIFFERENT"
        echo "    ${label_a}: ${hash_a}"
        echo "    ${label_b}: ${hash_b}"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    log "============================================================"
    log "Cobre reproducibility baseline -- empirical FP check"
    log "Repo root : ${REPO_ROOT}"
    log "Case      : ${CASE_DIR}"
    log "Output    : ${OUTPUT_BASE}"
    log "============================================================"

    check_prerequisites

    # Run each configuration sequentially.
    for np in "${RANK_COUNTS[@]}"; do
        run_configuration "${np}"
    done

    log "============================================================"
    log "Comparing convergence.parquet across configurations"
    log "============================================================"

    local convergence_1="${OUTPUT_BASE}/rank1/training/convergence.parquet"
    local convergence_2="${OUTPUT_BASE}/rank2/training/convergence.parquet"
    local convergence_4="${OUTPUT_BASE}/rank4/training/convergence.parquet"

    local all_identical=true

    compare_files "${convergence_1}" "rank1" "${convergence_2}" "rank2" || all_identical=false
    compare_files "${convergence_1}" "rank1" "${convergence_4}" "rank4" || all_identical=false
    compare_files "${convergence_2}" "rank2" "${convergence_4}" "rank4" || all_identical=false

    log "============================================================"
    if [[ "${all_identical}" == true ]]; then
        log "RESULT: All convergence.parquet files are byte-identical."
        log "        The upper-bound allreduce is bit-reproducible across rank counts."
        exit 0
    else
        log "RESULT: convergence.parquet files DIFFER across rank counts."
        log "        Floating-point non-associativity in allreduce is confirmed."
        log "        Proceed with ticket-009 (canonical upper-bound summation)."
        exit 1
    fi
}

main "$@"
