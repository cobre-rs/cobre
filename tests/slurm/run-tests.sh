#!/usr/bin/env bash
# SLURM MPI integration test for `cobre run`.
#
# Exercises 5 MPI execution modes against the 4ree example case and verifies
# numerical correctness by comparing each test's final lower bound against the
# T1 single-rank baseline within 1e-6 relative tolerance.
#
# T1: mpiexec -n 1         (single rank, baseline)
# T2: mpiexec -n 2         (multi-rank, single machine)
# T3: sbatch -N 1 -n 2     (SLURM single-node)
# T4: sbatch -N 2 -n 2     (SLURM multi-node — catches MPICH 4.3.0 deadlock)
# T5: sbatch -N 2 -n 4     (SLURM multi-node, multiple ranks per node)
#
# Usage:
#   /shared/run-tests.sh
#
# Prerequisites (satisfied by the ticket-003 Docker image):
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

# Prepend MPICH to PATH so mpiexec and Hydra proxy are found.
export PATH="/opt/mpich/bin:${PATH}"

# ---------------------------------------------------------------------------
# Result tracking (associative array: test name -> PASS|FAIL|<message>)
# ---------------------------------------------------------------------------
declare -A RESULTS

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
        echo "ERROR: python3 with pyarrow is required for lower-bound comparison"
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
    echo "INFO: timeout per test: ${TIMEOUT}s"
}

# ---------------------------------------------------------------------------
# compare_lower_bounds <baseline_parquet> <test_parquet>
#
# Reads the final lower_bound value from each parquet file and asserts that
# the relative difference is within 1e-6. Exits 1 on mismatch.
# ---------------------------------------------------------------------------
compare_lower_bounds() {
    local baseline_parquet="$1"
    local test_parquet="$2"
    python3 - "${baseline_parquet}" "${test_parquet}" <<'PYEOF'
import sys
import pyarrow.parquet as pq

baseline_path = sys.argv[1]
test_path = sys.argv[2]

baseline_table = pq.read_table(baseline_path)
test_table = pq.read_table(test_path)

baseline_lbs = baseline_table.column("lower_bound").to_pylist()
test_lbs = test_table.column("lower_bound").to_pylist()

if not baseline_lbs:
    print("FAIL: baseline convergence.parquet has no lower_bound rows")
    sys.exit(1)

if not test_lbs:
    print("FAIL: test convergence.parquet has no lower_bound rows")
    sys.exit(1)

baseline_final = baseline_lbs[-1]
test_final = test_lbs[-1]

TOL = 1e-6
denom = max(abs(baseline_final), 1e-12)
rel_diff = abs(baseline_final - test_final) / denom

if rel_diff > TOL:
    print(
        f"FAIL: final lower bounds differ beyond tolerance "
        f"(baseline={baseline_final}, test={test_final}, "
        f"rel_diff={rel_diff:.2e}, tol={TOL:.2e})"
    )
    sys.exit(1)

print(
    f"INFO: lower bounds match "
    f"(baseline={baseline_final:.6f}, test={test_final:.6f}, rel_diff={rel_diff:.2e})"
)
sys.exit(0)
PYEOF
}

# ---------------------------------------------------------------------------
# check_outputs <test_name> <exit_code> <output_dir> <baseline_parquet>
#
# Verifies:
#   1. Exit code is 0 (or reports timeout on code 124)
#   2. Output directory exists
#   3. policy/metadata.json exists
#   4. training/convergence.parquet exists
#   5. Lower bound matches baseline within 1e-6
#
# Returns 0 on full pass, 1 on any failure. Prints diagnostics on failure.
# ---------------------------------------------------------------------------
check_outputs() {
    local test_name="$1"
    local exit_code="$2"
    local output_dir="$3"
    local baseline_parquet="$4"

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

    local policy_meta="${output_dir}/policy/metadata.json"
    if [[ ! -f "${policy_meta}" ]]; then
        echo "[${test_name}] FAIL: policy/metadata.json missing"
        echo "[${test_name}] Contents of ${output_dir}:"
        ls -la "${output_dir}" 2>&1 || true
        return 1
    fi

    local convergence="${output_dir}/training/convergence.parquet"
    if [[ ! -f "${convergence}" ]]; then
        echo "[${test_name}] FAIL: training/convergence.parquet missing"
        echo "[${test_name}] Contents of ${output_dir}/training:"
        ls -la "${output_dir}/training" 2>&1 || true
        return 1
    fi

    if [[ -n "${baseline_parquet}" ]]; then
        echo "[${test_name}] Comparing lower bound against T1 baseline ..."
        if ! compare_lower_bounds "${baseline_parquet}" "${convergence}"; then
            echo "[${test_name}] FAIL: lower bound comparison failed (see above)"
            return 1
        fi
    fi

    return 0
}

# ---------------------------------------------------------------------------
# run_mpiexec_test <test_name> <n_ranks> <extra_args...>
#
# Runs: timeout $TIMEOUT mpiexec -n <n_ranks> $COBRE_BIN run $CASE_DIR
#         --output $WORK_DIR/<test_name_lower> --quiet [extra_args]
# Calls check_outputs and updates RESULTS.
# ---------------------------------------------------------------------------
run_mpiexec_test() {
    local test_name="$1"
    local n_ranks="$2"
    shift 2
    local extra_args=("$@")

    local tag
    tag="$(echo "${test_name}" | tr '[:upper:]' '[:lower:]')"
    local output_dir="${WORK_DIR}/${tag}"

    echo ""
    echo "=== ${test_name}: mpiexec -n ${n_ranks} ==="

    local exit_code=0
    timeout "${TIMEOUT}" mpiexec -n "${n_ranks}" \
        "${COBRE_BIN}" run "${CASE_DIR}" --output "${output_dir}" --quiet \
        "${extra_args[@]+"${extra_args[@]}"}" || exit_code=$?

    # Determine baseline: T1 uses no baseline (it IS the baseline), others use T1.
    local baseline=""
    if [[ "${test_name}" != "T1" ]]; then
        baseline="${WORK_DIR}/t1/training/convergence.parquet"
    fi

    if check_outputs "${test_name}" "${exit_code}" "${output_dir}" "${baseline}"; then
        echo "[${test_name}] PASS"
        RESULTS["${test_name}"]="PASS"
    else
        RESULTS["${test_name}"]="FAIL"
        print_summary
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# run_sbatch_test <test_name> <sbatch_N> <sbatch_n> <mpiexec_n> <extra_args...>
#
# Generates a batch script as a heredoc, submits via sbatch --wait, then
# calls check_outputs. On failure, prints SLURM .out/.err logs.
# ---------------------------------------------------------------------------
run_sbatch_test() {
    local test_name="$1"
    local sbatch_N="$2"
    local sbatch_n="$3"
    local mpiexec_n="$4"
    shift 4
    local extra_args=("$@")

    local tag
    tag="$(echo "${test_name}" | tr '[:upper:]' '[:lower:]')"
    local output_dir="${WORK_DIR}/${tag}"
    local slurm_out="${WORK_DIR}/${tag}.slurm.out"
    local slurm_err="${WORK_DIR}/${tag}.slurm.err"
    local batch_script="${WORK_DIR}/${tag}.sbatch"

    echo ""
    echo "=== ${test_name}: sbatch -N ${sbatch_N} -n ${sbatch_n} (mpiexec -n ${mpiexec_n}) ==="

    # Build the extra args string for insertion into the batch script.
    # Paths are absolute; this heredoc is unquoted so variables expand.
    local extra_str="${extra_args[*]+"${extra_args[*]}"}"

    # Generate the batch script with absolute paths.
    # The heredoc delimiter is unquoted so WORK_DIR, COBRE_BIN, CASE_DIR,
    # output_dir, and mpiexec_n expand at generation time.
    cat > "${batch_script}" << SBEOF
#!/bin/bash
export PATH=/opt/mpich/bin:\$PATH
mpiexec -n ${mpiexec_n} ${COBRE_BIN} run ${CASE_DIR} --output ${output_dir} --quiet ${extra_str}
SBEOF

    chmod +x "${batch_script}"

    local exit_code=0
    timeout "${TIMEOUT}" sbatch --wait \
        -N "${sbatch_N}" \
        -n "${sbatch_n}" \
        --output="${slurm_out}" \
        --error="${slurm_err}" \
        "${batch_script}" || exit_code=$?

    local baseline="${WORK_DIR}/t1/training/convergence.parquet"

    if check_outputs "${test_name}" "${exit_code}" "${output_dir}" "${baseline}"; then
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
# Prints a pass/fail table of all tests attempted so far.
# ---------------------------------------------------------------------------
print_summary() {
    echo ""
    echo "=============================="
    echo " SLURM MPI Test Suite Summary"
    echo "=============================="
    local all_pass=1
    for t in T1 T2 T3 T4 T5; do
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
preflight_check

# T1: mpiexec single rank (baseline — no extra args)
run_mpiexec_test "T1" 1

# T2: mpiexec multi-rank single machine
run_mpiexec_test "T2" 2 --threads 2

# T3: sbatch single-node, 2 ranks
run_sbatch_test "T3" 1 2 2

# T4: sbatch multi-node, 1 rank per node (critical PMI2 deadlock test)
run_sbatch_test "T4" 2 2 2

# T5: sbatch multi-node, 2 ranks per node
run_sbatch_test "T5" 2 4 4 --threads 1

print_summary
