# Understanding Results

After `cobre run` completes, the output directory contains three categories of
artifacts: training convergence data, a saved policy checkpoint, and simulation
dispatch results. This page explains how to read each category and how to query
the results programmatically using `cobre report`.

If you have not yet run the quickstart, complete [Quickstart](./quickstart.md)
first — this page references the `my_first_study/results/` directory produced
by that walkthrough.

---

## The Post-Run Summary

When `cobre run` finishes, it prints a summary block to stderr. The 1dtoy run
from the quickstart produces output similar to:

```
Training complete in 3.2s (128 iterations, iteration_limit)
  Lower bound:  142.3 $/stage
  Upper bound:  143.1 +/- 1.2 $/stage
  Gap:          0.6%
  Cuts:         384 active / 387 generated
  LP solves:    512

Simulation complete (100 scenarios)
  Completed: 100  Failed: 0

Output written to my_first_study/results/
```

Exact numerical values vary across runs because scenario sampling is stochastic.
The values below are representative of the 1dtoy example; your run will differ
slightly.

| Line                                                          | What it means                                                                                                                                                                      |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Training complete in 3.2s (128 iterations, iteration_limit)` | Training ran for 128 iterations (the limit set in `config.json`) and stopped because the iteration limit was reached, not because a convergence criterion was met.                 |
| `Lower bound:  142.3 $/stage`                                 | The optimizer's best proven lower bound on the minimum expected cost per stage. As training progresses this value rises and stabilizes.                                            |
| `Upper bound:  143.1 +/- 1.2 $/stage`                         | A statistical estimate of the true expected cost, computed from the forward-pass scenarios in the final iteration. The `+/- 1.2` is the standard deviation across those scenarios. |
| `Gap:  0.6%`                                                  | The relative distance between the lower and upper bounds expressed as a percentage. A gap of 0.6% means the policy cost is within 0.6% of the best possible. Smaller is better.    |
| `Cuts:  384 active / 387 generated`                           | The total number of optimality cuts in the policy pool. 384 are currently active; 3 were deactivated by the cut selection strategy.                                                |
| `LP solves:  512`                                             | Total number of linear programs solved across all stages and iterations.                                                                                                           |
| `Simulation complete (100 scenarios)`                         | The post-training simulation evaluated the trained policy over 100 independently sampled scenarios.                                                                                |
| `Completed: 100  Failed: 0`                                   | All 100 scenarios completed without solver errors.                                                                                                                                 |
| `Output written to my_first_study/results/`                   | Root path of the output directory.                                                                                                                                                 |

**Lower bound vs. upper bound.** The lower bound is the optimizer's proven best
estimate of the minimum achievable cost. The upper bound is the average cost
observed when running the current policy over sampled scenarios. When the gap is
small, the policy is near-optimal. When the gap is large, running more iterations
will typically narrow it further.

**Termination reasons.** The parenthetical after the iteration count explains why
training stopped:

- `iteration_limit` — the maximum iteration count was reached (the 1dtoy default).
- `converged at iter N` — a convergence criterion was met at iteration N and training
  stopped early. This appears when you configure a `bound_stalling` or similar rule
  in `config.json`.

> **Theory reference**: For the mathematical definition of lower and upper bounds,
> optimality gap, and stopping criteria, see
> [Convergence](https://cobre-rs.github.io/cobre-docs/theory/convergence.html)
> in the methodology reference.

---

## Output Directory Structure

All artifacts are written under the results directory you specified with `--output`.
The 1dtoy run produces:

```
my_first_study/results/
  training/
    _manifest.json          Completion manifest: status, iteration count, convergence, cut stats
    metadata.json           Run metadata: configuration snapshot, problem dimensions
    convergence.parquet     Per-iteration convergence metrics (lower bound, upper bound, gap)
    dictionaries/
      codes.json            Integer-to-string code mappings for entity categories
      state_dictionary.json State variable definitions and units
      entities.csv          Entity registry (id, name, type)
      variables.csv         LP variable registry
      bounds.parquet        LP variable bound definitions
    timing/
      iterations.parquet    Per-iteration wall-clock timing broken down by phase
  policy/
    cuts/
      stage_000.bin         FlatBuffers-encoded optimality cuts for stage 0
      stage_001.bin         ... stage 1
      stage_002.bin         ... stage 2
      stage_003.bin         ... stage 3
    basis/
      stage_000.bin         LP basis checkpoints for warm-starting
      stage_001.bin
      stage_002.bin
      stage_003.bin
    metadata.json           Policy metadata: stage count, cut counts per stage
  simulation/
    _manifest.json          Completion manifest: scenario counts
    buses/
      scenario_id=0000/data.parquet
      scenario_id=0001/data.parquet
      ...                   One partition per scenario
    costs/
      scenario_id=0000/data.parquet
      ...
    hydros/
      scenario_id=0000/data.parquet
      ...
    thermals/
      scenario_id=0000/data.parquet
      ...
    inflow_lags/            Inflow lag state data used to initialize scenario chains
```

The three top-level subdirectories have distinct roles:

- `training/` — everything produced during the training loop: convergence history,
  timing, and the dictionaries needed to interpret LP variable indices.
- `policy/` — the trained policy checkpoint. These binary files encode the optimality
  cuts built during training. They can be used to resume or extend a study.
- `simulation/` — the dispatch results from evaluating the trained policy over
  100 simulation scenarios.

---

## Training Results

### Reading `training/_manifest.json`

The training manifest is the canonical summary of what happened during training.
The 1dtoy run produces:

```json
{
  "version": "2.0.0",
  "status": "complete",
  "started_at": null,
  "completed_at": null,
  "iterations": {
    "max_iterations": null,
    "completed": 128,
    "converged_at": null
  },
  "convergence": {
    "achieved": false,
    "final_gap_percent": 0.0,
    "termination_reason": "iteration_limit"
  },
  "cuts": {
    "total_generated": 387,
    "total_active": 384,
    "peak_active": 384
  },
  "checksum": null,
  "distribution": {
    "backend": "local",
    "world_size": 1,
    "ranks_participated": 1,
    "num_nodes": 1,
    "threads_per_rank": 1
  }
}
```

Field-by-field explanation:

| Field                            | Meaning                                                                                                                                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `status`                         | `"complete"` when the training run finished normally. `"failed"` if a solver error aborted it.                                                                                             |
| `iterations.completed`           | Number of training iterations that were executed.                                                                                                                                          |
| `iterations.converged_at`        | If training stopped early due to a convergence criterion, the iteration number where it stopped. `null` for an iteration-limit stop.                                                       |
| `convergence.achieved`           | `true` if a convergence stopping rule was satisfied, `false` if the iteration limit was reached first.                                                                                     |
| `convergence.final_gap_percent`  | The gap between lower and upper bounds at the end of training, as a percentage. A value of `0.0` here reflects that the 1dtoy case converged very tightly within its 128-iteration budget. |
| `convergence.termination_reason` | Machine-readable reason for stopping. Common values: `"iteration_limit"`, `"bound_stalling"`.                                                                                              |
| `cuts.total_generated`           | Total optimality cuts created across all stages over the entire training run.                                                                                                              |
| `cuts.total_active`              | Cuts still active in the pool at the end of training (not deactivated by the cut selection strategy).                                                                                      |
| `cuts.peak_active`               | Maximum number of active cuts at any point during training.                                                                                                                                |
| `distribution.backend`           | Communication backend: `"local"` for single-process, `"mpi"` for distributed runs.                                                                                                        |
| `distribution.world_size`        | Number of MPI ranks involved in the run. `1` for single-process runs.                                                                                                                      |
| `distribution.threads_per_rank`  | Number of rayon worker threads per process.                                                                                                                                                |

**What "converged" means in practice.** A converged run (`convergence.achieved:
true`) means a stopping rule determined that continuing would not meaningfully
improve the policy. For the 1dtoy case, the gap reaches near zero within the
128-iteration budget even without an explicit convergence rule, which is why
`final_gap_percent` is `0.0` despite `achieved` being `false` — the run hit
its iteration limit at a point where the policy was already very tight.

For larger studies, configure a `bound_stalling` or `gap_threshold` stopping
rule in `config.json` to stop automatically when the gap stabilizes, rather
than running a fixed number of iterations.

---

## Simulation Results

### Hive-Partitioned Layout

The simulation output uses Hive partitioning: results are split into one
`data.parquet` file per scenario, stored in a directory named
`scenario_id=NNNN/`. This layout is natively understood by Polars, Pandas
(via PyArrow), R's arrow package, and DuckDB — they can read the entire
`simulation/costs/` directory as a single table and filter by `scenario_id`
at the storage layer without loading all data into memory.

The four entity categories are:

| Directory   | Contents                                                                                                                  |
| ----------- | ------------------------------------------------------------------------------------------------------------------------- |
| `buses/`    | Power balance results: load, generation injections, deficit, and excess at each bus per stage and block.                  |
| `hydros/`   | Hydro dispatch: turbined flow, spillage, reservoir storage levels, inflows, and generation per plant per stage and block. |
| `thermals/` | Thermal dispatch: generation output per unit per cost segment per stage and block.                                        |
| `costs/`    | Objective cost breakdown: total cost, thermal cost, hydro cost, penalty cost, and discount factor per stage.              |

Results are in Parquet format. To read them, use any columnar data tool:

```python
# Polars — reads all 100 scenarios at once
import polars as pl
df = pl.read_parquet("my_first_study/results/simulation/costs/")
print(df.head())
```

```python
# Pandas + PyArrow
import pandas as pd
df = pd.read_parquet("my_first_study/results/simulation/costs/")
print(df.head())
```

```sql
-- DuckDB — filter to a single scenario
SELECT * FROM read_parquet('my_first_study/results/simulation/costs/**/*.parquet')
WHERE scenario_id = 0;
```

```r
# R with arrow
library(arrow)
ds <- open_dataset("my_first_study/results/simulation/costs/")
dplyr::collect(dplyr::filter(ds, scenario_id == 0))
```

---

## Querying Results with `cobre report`

`cobre report` reads the JSON manifests and prints a structured JSON summary to
stdout. Use it with `jq` to extract specific metrics in scripts or CI pipelines.

```bash
# Print the full report
cobre report my_first_study/results
```

The output has this top-level shape:

```json
{
  "output_directory": "/abs/path/to/results",
  "status": "complete",
  "training": { "iterations": {}, "convergence": {}, "cuts": {} },
  "simulation": { "scenarios": {} },
  "metadata": { "run_info": {}, "configuration_snapshot": {} }
}
```

### Practical `jq` queries

```bash
# Extract the final convergence gap
cobre report my_first_study/results | jq '.training.convergence.final_gap_percent'

# Check how many iterations ran
cobre report my_first_study/results | jq '.training.iterations.completed'

# Check simulation scenario counts
cobre report my_first_study/results | jq '.simulation.scenarios'

# Use the status in a CI script: exit non-zero if training failed
status=$(cobre report my_first_study/results | jq -r '.status')
if [ "$status" != "complete" ]; then
  echo "Run did not complete successfully: $status" >&2
  exit 1
fi

# Check convergence was achieved (returns true or false)
cobre report my_first_study/results | jq '.training.convergence.achieved'
```

For the complete `cobre report` documentation and all available JSON fields,
see [CLI Reference](../guide/cli-reference.md#cobre-report).

For a detailed description of every field in every output file,
see [Output Format Reference](../reference/output-format.md).

---

## See Also

- [Convergence & Diagnostics](../guide/interpreting-results.md) — advanced analysis patterns and convergence assessment
- [CLI Reference](../guide/cli-reference.md) — all flags, subcommands, and exit codes
- [Configuration](../guide/configuration.md) — every `config.json` field documented
