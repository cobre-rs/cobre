# 4t + 2t single-process sweep — UNMEASURED claims (E10-3, 2026-09-20, baseline 077dbe2c)

Every claim below keeps its station, anchor and fix-shape; UNMEASURED is a terminal state for this pass, not a re-deferral. No claim hit `timeout` (every timed run finished well under 3x its bound) or `run-failed`, and no `unexercised-path` arose: every measurable symbol's code path is reached on its deck, so a 0-sample symbol is not-material (below the -F 99 sampling floor), not unexercised. The only UNMEASURED reason is `case-infeasible`.

## case-infeasible on the CLI + HiGHS profiling binary

### PD-032 (solver-comm): UNMEASURED / case-infeasible
Symbol: `cobre_clp_chg_bounds` (anchor `crates/cobre-solver/src/backends/clp/interface.rs:352`).
Evidence: CLP backend not compiled into the profiling binary (default feature = highs; nm shows no cobre_clp_chg_bounds/ClpSolver symbols; `cobre version` reports HiGHS 1.13.1). Measuring the CLP FFI needs a --features clp rebuild, a different binary than the sweep's pinned `cargo build --profile profiling --features mpi`.
needs-human: case-infeasible on the CLI+HiGHS harness — commission a CLP-backend (--features clp) rebuild to measure this claim?
No cost figure asserted; the claim's fix-shape is preserved as recorded.

### PD-033 (solver-comm): UNMEASURED / case-infeasible
Symbol: `add_rows` (anchor `crates/cobre-solver/src/backends/clp/interface.rs:215`).
Evidence: CLP backend not compiled into the profiling binary (no ClpSolver::add_rows symbol; HiGHS is the built backend). Needs a --features clp rebuild outside the pinned binary.
needs-human: case-infeasible on the CLI+HiGHS harness — commission a CLP-backend (--features clp) rebuild to measure this claim?
No cost figure asserted; the claim's fix-shape is preserved as recorded.

### PD-034 (solver-comm): UNMEASURED / case-infeasible
Symbol: `get_basis` (anchor `crates/cobre-solver/src/backends/clp/interface.rs:606`).
Evidence: CLP backend not compiled into the profiling binary (no ClpSolver::get_basis symbol; HiGHS is the built backend). Needs a --features clp rebuild outside the pinned binary.
needs-human: case-infeasible on the CLI+HiGHS harness — commission a CLP-backend (--features clp) rebuild to measure this claim?
No cost figure asserted; the claim's fix-shape is preserved as recorded.

### PD-049 (cli-python): UNMEASURED / case-infeasible
Symbol: `load_convergence` (anchor `crates/cobre-python/src/results.rs::load_convergence`).
Evidence: cobre-python is a separate maturin cdylib excluded from the workspace; its symbols are not linked into the `cobre` CLI binary and `cobre run` never loads the Python bindings. Profiling load_policy needs a Python-driven workload the perf-run.sh harness cannot express.
needs-human: case-infeasible on the CLI+HiGHS harness — commission a Python-driven (cobre-python) profile to measure this claim?
No cost figure asserted; the claim's fix-shape is preserved as recorded.

### PD-050 (cli-python): UNMEASURED / case-infeasible
Symbol: `read_parquet_partition_into` (anchor `crates/cobre-python/src/results.rs::read_parquet_partition_into`).
Evidence: cobre-python symbol (read_parquet_partition_into) absent from the CLI binary; the `cobre run` harness never enters the Python parquet reader. Needs a Python-driven profile.
needs-human: case-infeasible on the CLI+HiGHS harness — commission a Python-driven (cobre-python) profile to measure this claim?
No cost figure asserted; the claim's fix-shape is preserved as recorded.

### PD-051 (cli-python): UNMEASURED / case-infeasible
Symbol: `reshape_f64` (anchor `crates/cobre-python/src/results.rs::reshape_f64`).
Evidence: cobre-python symbol (cut_matrix/reshape_f64) absent from the CLI binary; the CLI run never enters the Python cut-matrix projection. Needs a Python-driven profile.
needs-human: case-infeasible on the CLI+HiGHS harness — commission a Python-driven (cobre-python) profile to measure this claim?
No cost figure asserted; the claim's fix-shape is preserved as recorded.

