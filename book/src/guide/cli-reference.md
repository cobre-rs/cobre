# CLI Reference

## Synopsis

```
cobre [--color <WHEN>] <SUBCOMMAND> [OPTIONS]
```

## Global Options

| Option           | Type                          | Default | Description                                                                                                                                                   |
| ---------------- | ----------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--color <WHEN>` | `auto` \| `always` \| `never` | `auto`  | Control ANSI color output on stderr. `always` forces color on — useful under `mpiexec` which pipes stderr through a non-TTY. Also honoured via `COBRE_COLOR`. |

## Subcommands

| Subcommand | Synopsis                           | Description                                                    |
| ---------- | ---------------------------------- | -------------------------------------------------------------- |
| `init`     | `cobre init [OPTIONS] [DIRECTORY]` | Scaffold a new case directory from an embedded template        |
| `run`      | `cobre run <CASE_DIR> [OPTIONS]`   | Load, train, simulate, and write results                       |
| `validate` | `cobre validate <CASE_DIR>`        | Validate a case directory and print a diagnostic report        |
| `report`   | `cobre report <RESULTS_DIR>`       | Query results from a completed run and print JSON to stdout    |
| `summary`  | `cobre summary <OUTPUT_DIR>`       | Display the post-run summary from a completed output directory |
| `schema`   | `cobre schema <COMMAND>`           | Manage JSON Schema files for case directory input types        |
| `version`  | `cobre version`                    | Print version, solver backend, and build information           |

---

## `cobre init`

Scaffolds a new case directory from an embedded template. Creates all required
input files (`config.json`, `penalties.json`, `stages.json`, system files, etc.)
so a new user can start from a working example.

### Arguments

| Argument      | Type | Description                                           |
| ------------- | ---- | ----------------------------------------------------- |
| `[DIRECTORY]` | Path | Target directory where template files will be written |

### Options

| Option              | Type   | Default | Description                                      |
| ------------------- | ------ | ------- | ------------------------------------------------ |
| `--template <NAME>` | string | —       | Template name to scaffold (e.g., `1dtoy`)        |
| `--list`            | flag   | off     | List all available templates and exit            |
| `--force`           | flag   | off     | Overwrite existing files in the target directory |

### Examples

```bash
# List available templates
cobre init --list

# Scaffold the 1dtoy example in a new directory
cobre init --template 1dtoy my_study

# Overwrite files in an existing directory
cobre init --template 1dtoy --force my_study
```

---

## `cobre run`

Executes the full solve lifecycle for a case directory:

1. **Load** — reads all input files and runs the 5-layer validation pipeline
2. **Train** — trains an SDDP policy using the configured stopping rules
3. **Simulate** — (optional) evaluates the trained policy over simulation scenarios
4. **Write** — writes all output files to the results directory

Whether simulation runs is controlled by `simulation.enabled` in `config.json`.
Stochastic artifact export is controlled by `exports.stochastic` in `config.json`.

### Arguments

| Argument     | Type | Description                                                              |
| ------------ | ---- | ------------------------------------------------------------------------ |
| `<CASE_DIR>` | Path | Path to the case directory containing input data files and `config.json` |

### Options

| Option           | Type    | Default              | Description                                                                                                                                                            |
| ---------------- | ------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--output <DIR>` | Path    | `<CASE_DIR>/output/` | Output directory for results                                                                                                                                           |
| `--threads <N>`  | integer | `1`                  | Number of worker threads per MPI rank. Each thread solves its own LP instances; scenarios are distributed across threads. Resolves: `--threads` > `COBRE_THREADS` > 1. |
| `--quiet`        | flag    | off                  | Suppress the banner and progress bars. Errors still go to stderr                                                                                                       |

### Config-First Principle

The CLI follows a **config-first** design: `config.json` defines _what_ to compute,
CLI flags define _how_ to run it. A study is fully specified by its case directory —
the same case produces the same results regardless of which CLI flags are used.

| Concern                    | Controlled by                                     |
| -------------------------- | ------------------------------------------------- |
| Simulation on/off          | `simulation.enabled` in `config.json`             |
| Stochastic export on/off   | `exports.stochastic` in `config.json`             |
| Forward passes, iterations | `training.*` in `config.json`                     |
| Cut selection              | `training.cut_selection` in `config.json`         |
| Inflow method              | `modeling.inflow_non_negativity` in `config.json` |

### Examples

```bash
# Run a study with default output location
cobre run /data/cases/hydro_study

# Write results to a custom directory
cobre run /data/cases/hydro_study --output /data/results/run_001

# Use 4 worker threads per MPI rank
cobre run /data/cases/hydro_study --threads 4

# Run without any terminal decorations (useful in scripts)
cobre run /data/cases/hydro_study --quiet

# Force color output when running under mpiexec
cobre --color always run /data/cases/hydro_study

# Run with MPI across 4 ranks
mpiexec -np 4 cobre run /data/cases/hydro_study
```

### SLURM clusters

On SLURM-managed clusters, launch Cobre with `srun` instead of `mpiexec`.
SLURM handles process placement, CPU binding, and NUMA-aware memory
allocation automatically.

**Basic launch:**

```bash
srun --mpi=pmi2 -n 4 ./cobre-mpi run /data/cases/hydro_study
```

**Hybrid MPI + threads** (recommended for production):

Cobre uses MPI for inter-node communication and rayon threads for
intra-node parallel LP solves. Set `--cpus-per-task` to control the
thread count per rank:

```bash
#!/bin/bash
#SBATCH --job-name=cobre
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem-bind=local
#SBATCH --output=cobre_%j.log

# Pin each rank to its allocated cores; use NUMA-local memory.
srun --cpu-bind=cores --mpi=pmi2 ./cobre-mpi run /data/case \
    --threads "$SLURM_CPUS_PER_TASK"
```

**Key SLURM flags for Cobre:**

| Flag                         | Purpose                                                       |
| ---------------------------- | ------------------------------------------------------------- |
| `--mpi=pmi2`                 | Use PMI-2 process startup (recommended for MPICH)             |
| `--mpi=pmix`                 | Alternative: use PMIx (SLURM 22.05+, MPICH 4+)                |
| `--ntasks-per-node=N`        | MPI ranks per node                                            |
| `--cpus-per-task=T`          | Cores per rank (sets rayon thread pool size)                  |
| `--cpu-bind=cores`           | Pin each rank's threads to specific cores                     |
| `--mem-bind=local`           | Allocate memory from the NUMA node closest to the bound cores |
| `--distribution=block:block` | Pack ranks on nodes, cores on sockets                         |
| `--hint=compute_bound`       | Use all cores per socket                                      |

> **Tip:** On modern SLURM clusters (22.05+), `--mpi=pmix` is preferred
> over `--mpi=pmi2` for better scalability. Check your cluster's default
> with `srun --mpi=list`.

---

## `cobre validate`

Runs the 5-layer validation pipeline and prints a diagnostic report to stdout.

On success, prints entity counts:

```
Valid case: 3 buses, 12 hydros, 8 thermals, 4 lines
  buses: 3
  hydros: 12
  thermals: 8
  lines: 4
```

On failure, prints each error prefixed with `error:` and exits with code 1.

### Arguments

| Argument     | Type | Description                            |
| ------------ | ---- | -------------------------------------- |
| `<CASE_DIR>` | Path | Path to the case directory to validate |

### Options

None.

### Examples

```bash
# Validate a case directory before running
cobre validate /data/cases/hydro_study

# Use in a script: only proceed if validation passes
cobre validate /data/cases/hydro_study && cobre run /data/cases/hydro_study
```

---

## `cobre report`

Reads the JSON manifests written by `cobre run` and prints a JSON summary to stdout.

The output has the following top-level shape:

```json
{
  "output_directory": "/abs/path/to/results",
  "status": "complete",
  "training": { "iterations": {}, "convergence": {}, "cuts": {} },
  "simulation": { "scenarios": {} },
  "metadata": { "run_info": {}, "configuration_snapshot": {} }
}
```

`simulation` and `metadata` are `null` when the corresponding files are absent
(e.g., when simulation was disabled in `config.json`).

### Arguments

| Argument        | Type | Description                                           |
| --------------- | ---- | ----------------------------------------------------- |
| `<RESULTS_DIR>` | Path | Path to the results directory produced by `cobre run` |

### Options

None.

### Examples

```bash
# Print the full report to the terminal
cobre report /data/cases/hydro_study/output

# Extract the convergence gap using jq
cobre report /data/cases/hydro_study/output | jq '.training.convergence.final_gap_percent'

# Check the run status in a script
status=$(cobre report /data/cases/hydro_study/output | jq -r '.status')
if [ "$status" = "complete" ]; then
  echo "Training converged"
fi
```

---

## `cobre summary`

Reads the training manifest and convergence log from a completed run's output
directory and prints the same human-readable summary table that `cobre run`
displays at the end of a study. This lets users inspect a past run without
re-executing it.

All output goes to stderr, matching the `cobre run` convention. stdout is
reserved for machine-readable output (see `cobre report`).

### File resolution

| File                           | Required | Behaviour when absent                                            |
| ------------------------------ | -------- | ---------------------------------------------------------------- |
| `training/_manifest.json`      | Yes      | Exits with code 2 (I/O error)                                    |
| `training/convergence.parquet` | No       | Falls back to zero-valued timing fields; gap comes from manifest |
| `simulation/_manifest.json`    | No       | Simulation section is omitted from the output                    |

### Output format

```
Training complete in 3m 42s (42 iterations, converged at iter 38)
  Lower bound:  4.85000e4 $/stage
  Upper bound:  4.90000e4 +/- 2.50000e2 $/stage
  Gap:          1.0%
  Cuts:         980000 active / 1250000 generated
  LP solves:    84000

Simulation complete in 0.0s (200 scenarios)
  Completed: 198  Failed: 2
```

The simulation section is omitted when `simulation/_manifest.json` is absent
(e.g., when simulation was disabled in `config.json`).

### Arguments

| Argument       | Type | Description                                          |
| -------------- | ---- | ---------------------------------------------------- |
| `<OUTPUT_DIR>` | Path | Path to the output directory produced by `cobre run` |

### Options

None.

### Examples

```bash
# Print the summary for a completed run
cobre summary /data/cases/hydro_study/output

# Inspect a run that used a custom output directory
cobre summary /data/results/run_001
```

---

## `cobre schema`

Manages JSON Schema files for case directory input types. Currently supports
exporting schemas.

### Subcommands

| Subcommand | Synopsis                                   | Description                                  |
| ---------- | ------------------------------------------ | -------------------------------------------- |
| `export`   | `cobre schema export [--output-dir <DIR>]` | Export JSON Schema files for all input types |

| Option               | Type | Default | Description                                                                                |
| -------------------- | ---- | ------- | ------------------------------------------------------------------------------------------ |
| `--output-dir <DIR>` | Path | `.`     | Directory to write schema files into. Created if absent. Existing schemas are overwritten. |

### Examples

```bash
# Export schemas to the current directory
cobre schema export

# Export schemas to a specific directory
cobre schema export --output-dir /data/schemas
```

---

## `cobre version`

Prints the binary version, active solver and communication backends, compression
support, host architecture, and build profile.

### Output Format

```
cobre   v0.5.0
solver: HiGHS
comm:   local
zstd:   enabled
arch:   x86_64-linux
build:  release (lto=thin)
```

| Line                               | Description                                                             |
| ---------------------------------- | ----------------------------------------------------------------------- |
| `cobre v{version}`                 | Binary version from `Cargo.toml`                                        |
| `solver: HiGHS`                    | Active LP solver backend (HiGHS in all standard builds)                 |
| `comm: local` or `comm: mpi`       | Communication backend (`mpi` only when compiled with the `mpi` feature) |
| `zstd: enabled`                    | Output compression support                                              |
| `arch: {arch}-{os}`                | Host CPU architecture and operating system                              |
| `build: release` or `build: debug` | Cargo build profile                                                     |

### Arguments

None.

### Options

None.

---

## Exit Codes

All subcommands follow the same exit code convention.

| Code | Category   | Cause                                                                                                                                                     |
| ---- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0`  | Success    | The command completed without errors                                                                                                                      |
| `1`  | Validation | Case directory failed the validation pipeline — schema errors, cross-reference errors, semantic constraint violations, or policy compatibility mismatches |
| `2`  | I/O        | File not found, permission denied, disk full, or write failure during loading or output                                                                   |
| `3`  | Solver     | LP infeasible subproblem or numerical solver failure during training or simulation                                                                        |
| `4`  | Internal   | Communication failure, unexpected channel closure, or other software/environment problem                                                                  |

Codes 1–2 indicate user-correctable input problems; codes 3–4 indicate case/environment
problems. Error messages are printed to stderr with `error:` prefix and hint lines.
See [Error Codes](../reference/error-codes.md) for a detailed catalog.

---

## Environment Variables

| Variable             | Description                                                                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `COBRE_COMM_BACKEND` | Override the communication backend at runtime. Set to `local` to force the local backend even when the binary was compiled with `mpi` support.          |
| `COBRE_THREADS`      | Number of worker threads per MPI rank for `cobre run`. Overridden by the `--threads` flag. Must be a positive integer.                                  |
| `COBRE_COLOR`        | Override color output when `--color auto` is in effect. Set to `always` or `never`. Ignored if `--color always` or `--color never` is given explicitly. |
| `FORCE_COLOR`        | Force color output on (any non-empty value). Checked after `COBRE_COLOR`. See [force-color.org](https://force-color.org).                               |
| `NO_COLOR`           | Disable colored terminal output. Respected by the banner and error formatters. Set to any non-empty value. See [no-color.org](https://no-color.org).    |
| `COLUMNS`            | Terminal width hint. Used by progress bars under MPI (where stderr is a pipe) to compute correct cursor movement. Inherited from the launching shell.   |
