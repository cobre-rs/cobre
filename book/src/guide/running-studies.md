# Running Studies

End-to-end workflow for running an SDDP study with `cobre run`, interpreting output,
and inspecting results.

---

## Preparing a Case Directory

A case directory is a folder containing all input data files required by Cobre.
The minimum required structure is:

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

All eight files are required. Before running, validate the input:

```bash
cobre validate /path/to/my_study
```

Successful validation prints entity counts and exits with code 0. Fix any reported
errors before proceeding. See [Case Directory Format](../reference/case-format.md)
for the full schema.

---

## Running `cobre run`

```bash
cobre run /path/to/my_study
```

By default, results are written to `<CASE_DIR>/output/`. To specify a different
location:

```bash
cobre run /path/to/my_study --output /path/to/results
```

### Lifecycle Stages

1. **Load** — reads input files, runs 5-layer validation (exits code 1 on validation failure, 2 on I/O error)
2. **Train** — builds the SDDP policy by iterating forward/backward passes; stops when stopping rules are met
3. **Simulate** — (optional) evaluates the policy over independent scenarios; requires `simulation.enabled = true`
4. **Write** — writes Hive-partitioned Parquet (tabular), JSON manifests/metadata, and FlatBuffers output

---

## Terminal Output

### Banner

When stdout is a terminal, a banner shows the version and solver backend.
Suppress with `--no-banner` (keeps progress bars) or `--quiet` (suppresses all
except errors).

### Progress Bars

During training, a progress bar shows current iteration count. In `--quiet` mode,
no progress bars are printed. Errors are always written to stderr.

### Summary

After all stages complete, a run summary is printed to stderr with:

- **Training**: iteration count, convergence status, bounds, gap, cuts, solves, time
- **Simulation** (when enabled): scenarios requested, completed, failed
- **Output directory**: absolute path to results

---

## Checking Results

Use `cobre report` to inspect the results:

```bash
cobre report /path/to/my_study/output
```

Reads manifest files and prints JSON to stdout (suitable for piping to `jq`):

```bash
cobre report /path/to/my_study/output | jq '.training.convergence.final_gap_percent'
```

Exits with code 0 on success or 2 if the results directory does not exist.

---

## Common Workflows

### Training Only

```bash
cobre run /path/to/my_study --skip-simulation
```

Trains the policy without simulation.

### Quiet Mode for Scripts

```bash
cobre run /path/to/my_study --quiet
exit_code=$?
if [ $exit_code -ne 0 ]; then
  echo "Study failed with exit code $exit_code" >&2
fi
```

Suppresses banner and progress output, suitable for batch scripts.

### Checking Exit Codes

| Exit Code | Meaning          | Action                                         |
| --------- | ---------------- | ---------------------------------------------- |
| `0`       | Success          | Results are available in the output directory  |
| `1`       | Validation error | Fix the input data and re-run `cobre validate` |
| `2`       | I/O error        | Check file paths and permissions               |
| `3`       | Solver error     | Check constraint bounds in the case data       |
| `4`       | Internal error   | Check environment; report at the issue tracker |

See [CLI Reference](./cli-reference.md#exit-codes) for the full exit code table.
