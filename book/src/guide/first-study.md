# Your First Study

This page walks through running a complete SDDP study with Cobre from a case
directory you already have. It covers the three-step workflow — validate, run,
report — explains what each output file contains at a high level, and points
you to the right reference pages for deeper analysis.

If you do not yet have a case directory, the [Quickstart tutorial](../tutorial/quickstart.md)
shows how to scaffold one from a built-in template using `cobre init --template 1dtoy`.

---

## Prepare Your Case Directory

A case directory is a self-contained folder that holds all input files for a
single power system study. The minimum required structure is:

```
my_study/
  config.json
  penalties.json
  stages.json
  initial_conditions.json
  system/
    buses.json
    hydros.json
    thermals.json
    lines.json
```

`config.json` controls the training algorithm (number of forward passes,
stopping rules) and simulation settings. `stages.json` defines the planning
horizon, policy graph type, and time blocks. The `system/` files define the
physical elements of the power system. `scenarios/` holds optional Parquet
files with PAR(p) statistics for stochastic inflow and load generation.

For the complete schema of every file, see
[Case Directory Format](../reference/case-format.md).

---

## Validate the Inputs

Before running the solver, validate the case directory to catch input errors
early:

```bash
cobre validate /path/to/my_study
```

The validation pipeline runs five layers in sequence: schema correctness,
cross-reference consistency (e.g., every plant references a valid bus),
physical feasibility (e.g., capacity bounds are non-negative), stochastic
consistency (e.g., PAR(p) statistics are well-defined), and solver
feasibility (e.g., the LP is bounded). Each layer must pass before the next
runs.

On success, Cobre prints entity counts and exits with code 0:

```
Valid case: 3 buses, 12 hydros, 8 thermals, 4 lines
  buses: 3
  hydros: 12
  thermals: 8
  lines: 4
```

On failure, each error is printed with an `error:` prefix and Cobre exits
with code 1. Fix all reported errors before proceeding — `cobre run` runs the
same validation pipeline and will exit with code 1 on any validation failure.

---

## Run the Study

```bash
cobre run /path/to/my_study
```

By default, results are written to `<CASE_DIR>/output/`. To specify a
different location:

```bash
cobre run /path/to/my_study --output /path/to/results
```

The run proceeds through four lifecycle stages:

1. **Load** — reads all input files and runs the validation pipeline.
2. **Train** — iterates the SDDP forward/backward pass loop until the
   configured stopping rules are satisfied (gap threshold, iteration limit,
   or bound stalling).
3. **Simulate** — evaluates the trained policy over independent out-of-sample
   scenarios. Skip this stage with `--skip-simulation`.
4. **Write** — writes all output files: Hive-partitioned Parquet for tabular
   results, JSON manifests, and a FlatBuffers policy checkpoint.

When stdout is a terminal, a progress bar tracks training iterations. A
post-run summary is printed to stderr when all stages complete:

```
Training complete in 12.4s (128 iterations, converged at iter 94)
  Lower bound:  3812.6 $/stage
  Upper bound:  3836.1 +/- 14.2 $/stage
  Gap:          0.6%
  Cuts:         94 active / 94 generated
  LP solves:    4992

Simulation complete (100 scenarios)
  Completed: 100  Failed: 0

Output written to /path/to/my_study/output/
```

Use `--quiet` to suppress the banner and progress output in batch scripts,
or `--verbose` to enable debug-level logging when diagnosing solver issues.

---

## Inspect the Results

Use `cobre report` to get a machine-readable summary of a completed run
without loading any Parquet files:

```bash
cobre report /path/to/my_study/output
```

`cobre report` reads the JSON manifest files written by `cobre run` and
prints a JSON summary to stdout:

```json
{
  "output_directory": "/path/to/my_study/output",
  "status": "complete",
  "training": { "iterations": {}, "convergence": {}, "cuts": {} },
  "simulation": { "scenarios": {} },
  "metadata": { "run_info": {}, "configuration_snapshot": {} }
}
```

This is suitable for piping to `jq` for scripted checks:

```bash
# Extract the final convergence gap
cobre report /path/to/my_study/output | jq '.training.convergence.final_gap_percent'

# Check whether convergence was achieved
cobre report /path/to/my_study/output | jq '.training.convergence.achieved'
```

### Output file layout

The results directory contains:

```
output/
  policy/
    cuts/
      stage_000.bin  ...  stage_NNN.bin   # FlatBuffers Benders cuts
    basis/
      stage_000.bin  ...  stage_NNN.bin   # LP warm-start bases
    metadata.json
  training/
    _manifest.json                        # Convergence summary
    convergence.parquet                   # Per-iteration bounds and gap
  simulation/
    _manifest.json                        # Simulation summary
    costs/                                # Stage costs per scenario
    hydros/                               # Hydro dispatch per scenario
    thermals/                             # Thermal dispatch per scenario
    buses/                                # Bus balance per scenario
```

The key metric to check first is the convergence gap in
`training/_manifest.json`. A gap below 1% is typically very good; 1-5% is
acceptable for long-horizon planning; above 5% warrants investigation
(consider increasing the iteration limit or forward pass count in
`config.json`).

For a detailed walkthrough of every output file and how to load and analyze
the Parquet data in Python or R, see
[Interpreting Results](./interpreting-results.md).

---

## Next Steps

- [Configuration](./configuration.md) — every `config.json` field: stopping
  rules, forward pass count, seed, and simulation settings
- [Stochastic Modeling](./stochastic-modeling.md) — how PAR(p) scenario
  statistics are structured and what they control
- [Interpreting Results](./interpreting-results.md) — practical analysis
  patterns for convergence diagnostics and simulation output
- [Case Directory Format](../reference/case-format.md) — complete schema
  for every input file
- [CLI Reference](./cli-reference.md) — all flags, subcommands, and exit
  codes
