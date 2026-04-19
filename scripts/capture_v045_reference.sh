#!/usr/bin/env bash
# capture_v045_reference.sh
#
# Captures SHA256 hashes of all stable parquet outputs produced by the CLI
# for the 26 deterministic d-cases and (optionally) the convertido benchmark
# case.  The resulting sha256.txt is the byte-level reference artifact against
# which Epic 03/04/05 regression tickets compare.
#
# This script freezes the pre-epic-03 "v0.4.5 baseline" configuration:
#   - WarmStartBasisMode::NonAlienFirst  (default; no config override)
#   - CanonicalStateStrategy::Disabled  (default; no config override)
#   - Baked-template simulation arm      (default)
#
# Unstable files (excluded from sha256.txt):
#   */training/convergence.parquet          -- contains time_*_ms columns
#   */training/solver/iterations.parquet    -- contains solve_time_ms etc.
#   */training/timing/iterations.parquet    -- pure wall-clock timing file
#   */simulation/solver/iterations.parquet  -- contains solve_time_ms etc.
# These embed actual wall-clock durations and produce different hashes on
# every run even when the algorithmic output is identical.  They are noted
# explicitly in docs/assessments/v0_4_5_reference.md.
#
# Usage:
#   bash scripts/capture_v045_reference.sh
#   bash scripts/capture_v045_reference.sh --convertido /path/to/convertido
#
# Options:
#   --convertido DIR  Path to the convertido case directory.
#                     Default: ~/git/cobre-bridge/example/convertido
#
# Outputs:
#   target/v045-reference/<case>/     CLI output tree for each case
#   target/v045-reference/sha256.txt  SHA256 of each stable *.parquet file,
#                                     sorted by relative path
#
# Exit codes:
#   0  -- completed successfully (convertido absence emits WARNING but is not fatal)
#   1  -- a required prerequisite is missing or a d-case run failed
#
# Note: target/ is already gitignored; parquet binaries are never committed.

set -euo pipefail

# ---------------------------------------------------------------------------
# Script location → repo root
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

COBRE_BIN="${REPO_ROOT}/target/release/cobre"
DETERMINISTIC_DIR="${REPO_ROOT}/examples/deterministic"
OUTPUT_BASE="${REPO_ROOT}/target/v045-reference"
CONVERTIDO_DIR="${HOME}/git/cobre-bridge/example/convertido"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --convertido)
            CONVERTIDO_DIR="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '/#/,/^[^#]/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

timestamp() {
    date '+%Y-%m-%dT%H:%M:%S'
}

log() {
    echo "[$(timestamp)] $*"
}

warn() {
    echo "[$(timestamp)] WARNING: $*" >&2
}

error() {
    echo "[$(timestamp)] ERROR: $*" >&2
}

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

check_prerequisites() {
    local ok=true

    if ! command -v sha256sum >/dev/null 2>&1; then
        error "sha256sum not found. Install GNU coreutils (on macOS: brew install coreutils)."
        ok=false
    fi

    if [[ ! -x "${COBRE_BIN}" ]]; then
        error "cobre binary not found at ${COBRE_BIN}."
        error "Build it with: cargo build --release --workspace"
        ok=false
    fi

    if [[ ! -d "${DETERMINISTIC_DIR}" ]]; then
        error "Deterministic cases directory not found at ${DETERMINISTIC_DIR}."
        ok=false
    fi

    if [[ "${ok}" == false ]]; then
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Run a single case
# ---------------------------------------------------------------------------

run_case() {
    local case_dir="$1"
    local case_name
    case_name="$(basename "${case_dir}")"
    local out_dir="${OUTPUT_BASE}/${case_name}"

    log "Running case: ${case_name}"

    mkdir -p "${out_dir}"

    local rc=0
    "${COBRE_BIN}" run "${case_dir}" --output "${out_dir}" --quiet || rc=$?
    if [[ "${rc}" -ne 0 ]]; then
        error "cobre run failed for case ${case_name} (exit code ${rc})."
        exit 1
    fi

    log "  Done: ${case_name}"
}

# ---------------------------------------------------------------------------
# Stable parquet predicate
#
# Returns exit code 0 (keep) or 1 (exclude) for a given parquet path.
# Excluded patterns embed wall-clock durations and produce different hashes
# on every run even when algorithmic output is identical.
# ---------------------------------------------------------------------------

is_stable_parquet() {
    local path="$1"
    case "${path}" in
        */training/convergence.parquet)         return 1 ;;
        */training/solver/iterations.parquet)   return 1 ;;
        */training/timing/iterations.parquet)   return 1 ;;
        */simulation/solver/iterations.parquet) return 1 ;;
        *)                                       return 0 ;;
    esac
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    cd "${REPO_ROOT}"

    log "============================================================"
    log "Cobre v0.4.5 reference capture"
    log "Repo root     : ${REPO_ROOT}"
    log "Binary        : ${COBRE_BIN}"
    log "Deterministic : ${DETERMINISTIC_DIR}"
    log "Output base   : ${OUTPUT_BASE}"
    log "============================================================"

    check_prerequisites

    mkdir -p "${OUTPUT_BASE}"

    # ------------------------------------------------------------------
    # D-cases: d01-d30 excluding d12, d17, d18 (not present on disk)
    # ------------------------------------------------------------------

    local d_cases=()
    while IFS= read -r -d '' case_dir; do
        d_cases+=("${case_dir}")
    done < <(find "${DETERMINISTIC_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

    if [[ ${#d_cases[@]} -eq 0 ]]; then
        error "No case directories found under ${DETERMINISTIC_DIR}."
        exit 1
    fi

    log "Found ${#d_cases[@]} deterministic case(s)."

    for case_dir in "${d_cases[@]}"; do
        run_case "${case_dir}"
    done

    # ------------------------------------------------------------------
    # Convertido benchmark (optional)
    # ------------------------------------------------------------------

    if [[ -d "${CONVERTIDO_DIR}" ]]; then
        log "Convertido directory found. Running with --threads 5 ..."
        local out_dir="${OUTPUT_BASE}/convertido"
        mkdir -p "${out_dir}"
        local rc=0
        "${COBRE_BIN}" run "${CONVERTIDO_DIR}" --output "${out_dir}" --threads 5 --quiet || rc=$?
        if [[ "${rc}" -ne 0 ]]; then
            error "cobre run failed for convertido (exit code ${rc})."
            exit 1
        fi
        log "  Done: convertido"
    else
        warn "Convertido directory not found at: ${CONVERTIDO_DIR}"
        warn "Convertido hashes will be absent from sha256.txt."
        warn "Run this script on the reference machine to capture convertido hashes."
    fi

    # ------------------------------------------------------------------
    # Produce sha256.txt: one row per STABLE parquet file, sorted by path
    # ------------------------------------------------------------------

    local sha_file="${OUTPUT_BASE}/sha256.txt"
    log "Computing SHA256 hashes (excluding timing/convergence files) ..."

    # Collect stable parquet paths, sort them, then hash.
    local stable_paths=()
    while IFS= read -r -d '' path; do
        if is_stable_parquet "${path}"; then
            stable_paths+=("${path}")
        fi
    done < <(find "${OUTPUT_BASE}" -name "*.parquet" -print0 | sort -z)

    if [[ ${#stable_paths[@]} -eq 0 ]]; then
        error "No stable parquet files found. Something went wrong with the runs."
        exit 1
    fi

    sha256sum "${stable_paths[@]}" \
        | sed "s|${OUTPUT_BASE}/||g" \
        | sort \
        > "${sha_file}"

    local count
    count="$(wc -l < "${sha_file}")"
    log "SHA256 map written: ${sha_file} (${count} stable entries)"

    log "============================================================"
    log "Capture complete."
    log "To embed in docs/assessments/v0_4_5_reference.md:"
    log "  cat ${sha_file}"
    log "============================================================"
}

main "$@"
