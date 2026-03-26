# Python Quickstart

Install Cobre and run your first study in a few steps.

## Installation

```bash
pip install cobre-python
```

Requires Python 3.12, 3.13, or 3.14.

## Run a Case

```python
import cobre_python as cobre

result = cobre.run("path/to/case")
```

The `run()` function loads the case, trains an SDDP policy, optionally runs
simulation, and writes output files. It returns a dictionary with:

```python
print(f"Converged: {result['converged']}")
print(f"Iterations: {result['iterations']}")
print(f"Lower bound: {result['lower_bound']:.2f}")
print(f"Gap: {result['gap_percent']:.2f}%")
print(f"Output dir: {result['output_dir']}")
```

## Optional Parameters

```python
result = cobre.run(
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

# Simulation costs (if simulation was enabled)
costs = pl.read_parquet("output/simulation/costs.parquet")
print(costs.describe())
```

## Arrow Zero-Copy Loading

For larger datasets, use the built-in Arrow loaders that avoid serialization
overhead:

```python
# Returns an Arrow RecordBatch (zero-copy)
convergence_batch = cobre.load_convergence_arrow("output/")
simulation_batch = cobre.load_simulation_arrow("output/")

# Convert to Polars without copying
import polars as pl
df = pl.from_arrow(convergence_batch)
```

## Next Steps

- See the [case directory format](../reference/case-format.md) for input file
  specifications.
- Explore the [examples](../examples/1dtoy.md) for ready-to-run cases.
- Read the [Jupyter quickstart notebook](https://github.com/cobre-rs/cobre/blob/main/examples/notebooks/quickstart.ipynb)
  for a complete end-to-end workflow with visualization.
