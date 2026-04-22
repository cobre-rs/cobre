#!/usr/bin/env bash
# test_per_opening_mpi_parity.sh
#
# Runs the same cobre case with 1, 2, and 4 MPI ranks, then compares the
# resulting training/solver/iterations.parquet files to verify that per-
# opening counter columns are invariant to MPI rank count.
#
# This test confirms the cross-MPI parity property:
#   "per-opening values are identical across rank counts."
#
# Usage:
#   cargo build --release --features mpi
#   bash scripts/test_per_opening_mpi_parity.sh [CASE_DIR]
#
# Arguments:
#   CASE_DIR  Path to the cobre case directory.
#             Default: examples/deterministic/d01-thermal-dispatch
#
# Exit codes:
#   0  -- all 3 runs agree on every non-timing counter column
#   1  -- at least one counter column differs between runs (parity failure)
#   2  -- a prerequisite is missing (mpirun, cobre binary, or Python)
#
# Notes:
#   - Timing columns (solve_time_ms, basis_set_time_ms, load_model_time_ms,
#     add_rows_time_ms, set_bounds_time_ms) are excluded from the comparison
#     because wall-clock time is inherently rank-count-dependent.
#   - Parquets carry per-(rank, worker_id) backward rows.
#     The Python comparator first SUMs all numeric counter columns over those
#     rows GROUP BY (iteration, phase, stage, opening), recovering a
#     rank-invariant row shape so cross-rank comparison is meaningful. It also
#     asserts each (rank, worker_id) tuple appears exactly once per backward group.
#   - Output directories target/parity_1, target/parity_2, and target/parity_4
#     are preserved on failure for post-mortem investigation.
#   - On success, output directories are cleaned up.
#   - This script is NOT intended for CI execution; it is a manual verification
#     tool for machines with MPI installed.

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

COBRE_BIN="${REPO_ROOT}/target/release/cobre"
COMPARATOR="${SCRIPT_DIR}/compare_per_opening_parity.py"

# Case directory: first positional argument or the D01 default.
CASE_DIR="${1:-${REPO_ROOT}/examples/deterministic/d01-thermal-dispatch}"

OUT_BASE="${REPO_ROOT}/target"
OUT_1="${OUT_BASE}/parity_1"
OUT_2="${OUT_BASE}/parity_2"
OUT_4="${OUT_BASE}/parity_4"

PARQUET_REL="training/solver/iterations.parquet"

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

    # mpirun must be on PATH.
    if ! command -v mpirun >/dev/null 2>&1; then
        error "mpirun not found. Install OpenMPI or MPICH and ensure mpirun is on PATH."
        ok=false
    fi

    # The release binary must exist.
    if [[ ! -x "${COBRE_BIN}" ]]; then
        error "target/release/cobre not found or not executable."
        error "Build with: cargo build --release --features mpi"
        ok=false
    fi

    # Verify MPI support compiled in.  An nm-based check avoids running the
    # binary, which could be slow or require a real case directory.
    #
    # NOTE: nm output is written to a temp file before grepping.  This avoids
    # a SIGPIPE race: grep -q closes the pipe after the first match, which
    # sends SIGPIPE (exit 141) to nm.  With set -o pipefail the pipeline's
    # exit status would be 141, making the if-condition evaluate as false even
    # though grep found the symbol.
    if command -v nm >/dev/null 2>&1; then
        local nm_tmp
        nm_tmp=$(mktemp)
        nm "${COBRE_BIN}" 2>/dev/null > "${nm_tmp}" || true
        local mpi_found=false
        if grep -qi "mpi" "${nm_tmp}"; then
            mpi_found=true
        fi
        rm -f "${nm_tmp}"
        if [[ "${mpi_found}" == false ]]; then
            error "cobre binary does not appear to include MPI support."
            error "Rebuild with: cargo build --release --features mpi"
            ok=false
        fi
    else
        log "WARNING: nm not available -- cannot verify MPI feature compilation."
        log "         Ensure the binary was built with --features mpi."
    fi

    # Case directory must exist.
    if [[ ! -d "${CASE_DIR}" ]]; then
        error "Case directory not found: ${CASE_DIR}"
        ok=false
    fi

    # Python with pyarrow is required for the comparator script.
    # Try python3 first; fall back to python3.14.
    PYTHON_BIN=""
    for candidate in python3 python3.14; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            if "${candidate}" -c "import pyarrow" 2>/dev/null; then
                PYTHON_BIN="${candidate}"
                break
            fi
        fi
    done
    if [[ -z "${PYTHON_BIN}" ]]; then
        error "python3 (or python3.14) with pyarrow not found."
        error "Install pyarrow: pip install pyarrow"
        ok=false
    fi

    # Comparator script must exist.
    if [[ ! -f "${COMPARATOR}" ]]; then
        error "Comparator script not found: ${COMPARATOR}"
        ok=false
    fi

    if [[ "${ok}" == false ]]; then
        exit 2
    fi

    log "Prerequisites satisfied (python: ${PYTHON_BIN})"
}

# ---------------------------------------------------------------------------
# Run a single MPI configuration
# ---------------------------------------------------------------------------

run_configuration() {
    local np="$1"
    local out_dir="$2"

    log "--- Configuration: ${np} MPI rank(s) ---"
    log "Output directory: ${out_dir}"

    mkdir -p "${out_dir}"

    local rc=0
    if [[ "${np}" -eq 1 ]]; then
        log "Starting cobre run (single process, no mpirun) ..."
        "${COBRE_BIN}" run "${CASE_DIR}" \
            --output "${out_dir}" \
            --threads 5 \
            --quiet || rc=$?
    else
        log "Starting cobre run with mpirun -np ${np} ..."
        mpirun -np "${np}" "${COBRE_BIN}" run "${CASE_DIR}" \
            --output "${out_dir}" \
            --threads 5 \
            --quiet || rc=$?
    fi

    if [[ "${rc}" -ne 0 ]]; then
        error "cobre run failed for np=${np} (exit code ${rc})."
        error "Output directories preserved for investigation: ${OUT_1} ${OUT_2} ${OUT_4}"
        exit "${rc}"
    fi

    # Verify expected parquet was produced.
    local parquet="${out_dir}/${PARQUET_REL}"
    if [[ ! -f "${parquet}" ]]; then
        error "Expected output file not found: ${parquet}"
        error "Output directories preserved for investigation: ${OUT_1} ${OUT_2} ${OUT_4}"
        exit 1
    fi

    log "Completed: np=${np} (parquet: ${parquet})"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    log "============================================================"
    log "Cobre per-opening MPI parity test"
    log "Repo root : ${REPO_ROOT}"
    log "Case      : ${CASE_DIR}"
    log "Parquet   : ${PARQUET_REL}"
    log "============================================================"

    check_prerequisites

    # Remove output directories from any previous run to avoid stale data.
    rm -rf "${OUT_1}" "${OUT_2}" "${OUT_4}"

    run_configuration 1 "${OUT_1}"
    run_configuration 2 "${OUT_2}"
    run_configuration 4 "${OUT_4}"

    log "============================================================"
    log "Comparing ${PARQUET_REL} across 1, 2, and 4 ranks"
    log "============================================================"

    local rc=0
    "${PYTHON_BIN}" "${COMPARATOR}" \
        --dir1 "${OUT_1}" \
        --dir2 "${OUT_2}" \
        --dir4 "${OUT_4}" \
        --parquet-path "${PARQUET_REL}" || rc=$?

    log "============================================================"
    if [[ "${rc}" -eq 0 ]]; then
        log "PASS: per-opening counter columns are MPI-rank-invariant."
        # Clean up output directories on success.
        rm -rf "${OUT_1}" "${OUT_2}" "${OUT_4}"
        exit 0
    else
        log "FAIL: parity divergence detected -- see comparator output above."
        log "Output directories preserved for investigation:"
        log "  ${OUT_1}"
        log "  ${OUT_2}"
        log "  ${OUT_4}"
        exit 1
    fi
}

main "$@"
