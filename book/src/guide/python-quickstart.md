# Python Quickstart

Install Cobre and run your first study in a few steps.

## Installation

```bash
pip install cobre-python
```

Requires Python 3.12, 3.13, or 3.14.

## Run a Case

```python
import cobre

result = cobre.run.run("path/to/case")
```

The `cobre.run.run()` function loads the case, trains an SDDP policy, optionally
runs simulation, and writes output files. It returns a dictionary with the
following keys:

| Key             | Type              | Description                                     |
| --------------- | ----------------- | ----------------------------------------------- |
| `converged`     | `bool`            | Whether training converged                      |
| `iterations`    | `int`             | Number of training iterations completed         |
| `lower_bound`   | `float`           | Final lower bound                               |
| `upper_bound`   | `float` or `None` | Final upper bound (None if no simulation)       |
| `gap_percent`   | `float` or `None` | Optimality gap percentage (None if unavailable) |
| `total_time_ms` | `int`             | Total wall-clock time in milliseconds           |
| `output_dir`    | `str`             | Path to the output directory                    |
| `simulation`    | `dict` or `None`  | Simulation summary (if enabled)                 |
| `stochastic`    | `dict` or `None`  | Stochastic preprocessing summary                |
| `hydro_models`  | `dict` or `None`  | Hydro model summary                             |
| `provenance`    | `dict`            | Build version and environment metadata          |

```python
print(f"Converged: {result['converged']}")
print(f"Iterations: {result['iterations']}")
print(f"Lower bound: {result['lower_bound']:.2f}")
if result['gap_percent'] is not None:
    print(f"Gap: {result['gap_percent']:.2f}%")
print(f"Output dir: {result['output_dir']}")
```

## Optional Parameters

```python
result = cobre.run.run(
    "path/to/case",
    output_dir="path/to/output",   # default: case_dir/output
    threads=4,                      # default: 1
    skip_simulation=True,           # default: False
)
```

## Read Output with Polars

Cobre writes results as Parquet files, which can be loaded directly with
Polars or any Arrow-compatible library:

```python
import polars as pl

# Convergence trajectory
convergence = pl.read_parquet("output/training/convergence.parquet")
print(convergence.head())

# Simulation costs (if simulation was enabled) — Hive-partitioned
costs = pl.read_parquet("output/simulation/costs/")
print(costs.describe())
```

## Arrow Zero-Copy Loading

For larger datasets, use the built-in Arrow loaders that avoid serialization
overhead:

```python
# Returns a pyarrow.Table (zero-copy)
convergence_table = cobre.results.load_convergence_arrow("output/")
simulation_tables = cobre.results.load_simulation_arrow("output/")

# Convert to Polars without copying
import polars as pl
df = pl.from_arrow(convergence_table)
```

## Next Steps

- See the [case directory format](../reference/case-format.md) for input file
  specifications.
- Explore the [examples](../examples/1dtoy.md) for ready-to-run cases.
- Read the [Jupyter quickstart notebook](https://github.com/cobre-rs/cobre/blob/main/examples/notebooks/quickstart.ipynb)
  for a complete end-to-end workflow with visualization.
