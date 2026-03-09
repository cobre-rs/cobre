# Interpreting Results

The [Understanding Results](../tutorial/understanding-results.md) tutorial
explains what each output file contains and how to read it. This page goes
one level deeper: it provides practical analysis patterns for answering
domain questions from the data. It assumes you have already completed the
tutorial and are comfortable loading Parquet files in your preferred tool.

The focus is on convergence diagnostics and simulation analysis. By the end
of this page you will know how to assess whether a run converged, how to
extract generation and cost statistics across scenarios, and how to identify
common problems from the output data.

---

## Convergence Diagnostics

### Reading the gap from `training/_manifest.json`

The manifest is the first place to check after any run. The key fields for
convergence assessment are:

```json
{
  "convergence": {
    "achieved": false,
    "final_gap_percent": 0.6,
    "termination_reason": "iteration_limit"
  },
  "iterations": {
    "completed": 128,
    "converged_at": null
  }
}
```

| Field                            | What to look for                                                                                         |
| -------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `convergence.achieved`           | `true` means a stopping rule declared convergence. `false` means the run exhausted its iteration budget. |
| `convergence.final_gap_percent`  | The gap between lower and upper bounds at termination. Smaller is better. See guidelines below.          |
| `convergence.termination_reason` | `"iteration_limit"` is the most common; `"bound_stalling"` means the gap stopped shrinking.              |
| `iterations.converged_at`        | Non-null only when `achieved` is `true`. Tells you how many iterations the run actually needed.          |

**Gap guidelines.** There is no universal threshold — acceptable gap depends
on the decision being made and the study's time horizon. As rough guidance:

- Below 1%: typically very good. The policy cost is within 1% of the
  theoretical optimum.
- 1% to 5%: acceptable for long-horizon planning studies where model
  uncertainty is already large.
- Above 5%: warrants investigation. The policy may be significantly
  suboptimal.

**What to do if the gap is large:**

1. Increase `limit` in the `iteration_limit` stopping rule.
2. Increase `forward_passes` in `config.json` to reduce noise in the upper
   bound estimate per iteration.
3. Check `training/convergence.parquet` (see next section) to see whether
   the gap is still decreasing or has plateaued.
4. Check for solver infeasibilities: if `simulation/_manifest.json` shows
   failed scenarios, the policy may be encountering numerically difficult
   stages.

---

## Reading Convergence History

`training/convergence.parquet` contains one row per training iteration with
the full convergence history. Its schema:

| Column             | Type    | Description                                                 |
| ------------------ | ------- | ----------------------------------------------------------- |
| `iteration`        | INT32   | Iteration number (0-indexed)                                |
| `lower_bound`      | FLOAT64 | Optimizer's proven lower bound on the expected cost         |
| `upper_bound_mean` | FLOAT64 | Statistical upper bound estimate (mean over forward passes) |
| `upper_bound_std`  | FLOAT64 | Standard deviation of the upper bound estimate              |
| `gap_percent`      | FLOAT64 | Relative gap as a percentage (null when lower_bound <= 0)   |
| `cuts_added`       | INT32   | Cuts added to the pool in this iteration                    |
| `cuts_removed`     | INT32   | Cuts removed by the cut selection strategy                  |
| `cuts_active`      | INT64   | Total active cuts across all stages after this iteration    |
| `lp_solves`        | INT64   | Cumulative LP solves up to this iteration                   |

### Python (Polars)

```python
import polars as pl
import matplotlib.pyplot as plt

df = pl.read_parquet("results/training/convergence.parquet")

# Plot convergence bounds over iterations
plt.figure(figsize=(10, 4))
plt.plot(df["iteration"], df["lower_bound"], label="Lower bound")
plt.plot(df["iteration"], df["upper_bound_mean"], label="Upper bound (mean)")
plt.fill_between(
    df["iteration"].to_list(),
    (df["upper_bound_mean"] - df["upper_bound_std"]).to_list(),
    (df["upper_bound_mean"] + df["upper_bound_std"]).to_list(),
    alpha=0.2,
    label="Upper bound ± 1 std",
)
plt.xlabel("Iteration")
plt.ylabel("Expected cost ($/stage)")
plt.legend()
plt.tight_layout()
plt.show()

# Check final gap
final = df.filter(pl.col("iteration") == df["iteration"].max())
print(final.select(["iteration", "lower_bound", "upper_bound_mean", "gap_percent"]))
```

### R

```r
library(arrow)
library(ggplot2)

df <- read_parquet("results/training/convergence.parquet")

# Plot convergence bounds
ggplot(df, aes(x = iteration)) +
  geom_line(aes(y = lower_bound, color = "Lower bound")) +
  geom_line(aes(y = upper_bound_mean, color = "Upper bound")) +
  geom_ribbon(
    aes(
      ymin = upper_bound_mean - upper_bound_std,
      ymax = upper_bound_mean + upper_bound_std
    ),
    alpha = 0.2
  ) +
  labs(
    x = "Iteration",
    y = "Expected cost ($/stage)",
    color = NULL
  ) +
  theme_minimal()

# Print final gap
tail(df[, c("iteration", "lower_bound", "upper_bound_mean", "gap_percent")], 1)
```

**What to look for in the convergence plot:**

- Both bounds should move toward each other over iterations. The lower bound
  rises; the upper bound mean falls and its standard deviation narrows.
- A lower bound that stays flat after the first few iterations suggests the
  backward pass cuts are not improving: check `cuts_added` to confirm cuts
  are being generated.
- An upper bound that oscillates widely without narrowing suggests the
  `forward_passes` count is too low to produce a stable estimate.

---

## Analyzing Simulation Results

The simulation output is Hive-partitioned: results are stored in one
`data.parquet` file per scenario under `simulation/<category>/scenario_id=NNNN/`.
Polars, Pandas, R arrow, and DuckDB all support reading the entire directory
as a single table and filtering by `scenario_id` at the storage layer.

### Aggregating across scenarios

The most common operation is computing statistics across all scenarios for
a given entity or stage.

**Python (Polars) — mean and percentiles:**

```python
import polars as pl

# Load all hydro results across all scenarios
hydros = pl.read_parquet("results/simulation/hydros/")

# Mean generation per hydro plant per stage, across all scenarios
mean_gen = (
    hydros
    .group_by(["hydro_id", "stage_id"])
    .agg(
        pl.col("generation_mwh").mean().alias("mean_generation_mwh"),
        pl.col("generation_mwh").quantile(0.10).alias("p10_generation_mwh"),
        pl.col("generation_mwh").quantile(0.90).alias("p90_generation_mwh"),
    )
    .sort(["hydro_id", "stage_id"])
)
print(mean_gen)
```

**R:**

```r
library(arrow)
library(dplyr)

# Load all hydro results
hydros <- open_dataset("results/simulation/hydros/") |> collect()

# Mean and P10/P90 generation per hydro plant per stage
mean_gen <- hydros |>
  group_by(hydro_id, stage_id) |>
  summarise(
    mean_generation_mwh = mean(generation_mwh),
    p10_generation_mwh  = quantile(generation_mwh, 0.10),
    p90_generation_mwh  = quantile(generation_mwh, 0.90),
    .groups = "drop"
  ) |>
  arrange(hydro_id, stage_id)

print(mean_gen)
```

### Filtering to a single scenario

```python
# Polars — read only scenario 0 (avoids loading all partitions)
costs_s0 = pl.read_parquet(
    "results/simulation/costs/",
    hive_partitioning=True,
).filter(pl.col("scenario_id") == 0)
```

```sql
-- DuckDB
SELECT * FROM read_parquet('results/simulation/costs/**/*.parquet')
WHERE scenario_id = 0
ORDER BY stage_id;
```

---

## Common Analysis Tasks

### (a) Expected generation by hydro plant

```python
import polars as pl

hydros = pl.read_parquet("results/simulation/hydros/")
expected = (
    hydros
    .group_by("hydro_id")
    .agg(pl.col("generation_mwh").mean().alias("mean_annual_generation_mwh"))
    .sort("hydro_id")
)
print(expected)
```

### (b) Expected thermal generation cost

```python
thermals = pl.read_parquet("results/simulation/thermals/")
thermal_cost = (
    thermals
    .group_by("thermal_id")
    .agg(pl.col("generation_cost").mean().alias("mean_total_cost"))
    .sort("thermal_id")
)
print(thermal_cost)
```

In R:

```r
library(arrow)
library(dplyr)

thermals <- open_dataset("results/simulation/thermals/") |> collect()

thermal_cost <- thermals |>
  group_by(thermal_id) |>
  summarise(mean_total_cost = mean(generation_cost), .groups = "drop") |>
  arrange(thermal_id)

print(thermal_cost)
```

### (c) Deficit probability per bus

A scenario has a deficit at a given stage if `deficit_mwh > 0` for any bus
in that stage. The deficit probability is the fraction of scenarios where
this occurs.

```python
buses = pl.read_parquet("results/simulation/buses/")
n_scenarios = buses["scenario_id"].n_unique()

deficit_prob = (
    buses
    .group_by(["bus_id", "stage_id"])
    .agg(
        (pl.col("deficit_mwh") > 0).mean().alias("deficit_probability")
    )
    .sort(["bus_id", "stage_id"])
)
print(deficit_prob)
```

### (d) Water value (shadow price) from hydro output

The `water_value_per_hm3` column in `simulation/hydros/` records the shadow
price of reservoir storage at each stage — the marginal value of having one
additional hm³ of stored water. This is the water value, a key output of
the SDDP policy.

```python
hydros = pl.read_parquet("results/simulation/hydros/")
water_value = (
    hydros
    .group_by(["hydro_id", "stage_id"])
    .agg(pl.col("water_value_per_hm3").mean().alias("mean_water_value"))
    .sort(["hydro_id", "stage_id"])
)
print(water_value)
```

A high water value at a given stage means the reservoir is scarce relative
to expected future demand — the solver is conserving water for later stages.
A water value near zero means the reservoir is abundant and water has little
marginal value at that point in time.

---

## Using `cobre report`

`cobre report` provides a quick machine-readable summary without loading any
Parquet files:

```bash
cobre report results/
```

Use it in scripts or CI pipelines to extract a specific metric without
writing a data loading script:

```bash
# Check the final gap in a CI pipeline
gap=$(cobre report results/ | jq '.training.convergence.final_gap_percent')
echo "Final gap: ${gap}%"
```

For all available `cobre report` fields and flags, see
[CLI Reference](./cli-reference.md#cobre-report).

---

## Troubleshooting

### Gap not converging

The gap stays large after many iterations, or the lower bound rises very
slowly.

Possible causes:

- **Too few iterations.** The most common cause. Increase the `iteration_limit`.
- **Too few forward passes.** A `forward_passes` count of 1 (as in the
  1dtoy tutorial) gives high variance in the upper bound estimate. Increase
  to 10 or more for a stable gap reading.
- **Numerically difficult stages.** Check `training/convergence.parquet` for
  iterations where `cuts_added` is zero — this can indicate stages where the
  backward pass is not generating improving cuts.
- **Policy horizon issues.** Verify `stages.json` has the correct stage
  ordering and that `policy_graph.type` is set correctly.

### Unexpected deficit

Simulation scenarios show non-zero `deficit_mwh` in `simulation/buses/` but
the system should have enough capacity.

Possible causes:

- **Insufficient thermal capacity.** Compare total load (`load_mw` summed
  across buses) against total thermal capacity. If load exceeds generation
  capacity in some scenarios, deficit is unavoidable.
- **Hydro reservoir ran dry.** Check `storage_final_hm3` in
  `simulation/hydros/`. If it hits zero in early stages, subsequent stages
  have no hydro generation and may resort to deficit.
- **Very low deficit penalty.** If `deficit_segments` in `penalties.json`
  are priced below thermal generation cost, the solver will prefer deficit
  over generation. Increase the deficit cost.

### Zero generation from a plant

A thermal or hydro plant shows zero generation in all scenarios.

Possible causes:

- **Plant is more expensive than deficit.** Check the plant's cost against
  the bus deficit penalty. If the cost exceeds the penalty, deficit is
  cheaper and the solver avoids dispatching the plant.
- **Bus connectivity.** Verify the plant's `bus_id` matches a bus that
  actually has load. A plant connected to a zero-load bus will never be
  dispatched.
- **Hydro: reservoir constraints too tight.** If `min_storage_hm3` is close
  to the initial storage level, the solver cannot turbine water without
  risking a storage violation. Review `initial_conditions.json` and storage
  bounds in `hydros.json`.

---

## Related Pages

- [Understanding Results](../tutorial/understanding-results.md) — file-by-file walkthrough of every output artifact
- [Output Format Reference](../reference/output-format.md) — complete field-by-field schema for all output files
- [Configuration](./configuration.md) — all `config.json` fields including stopping rules and seed
