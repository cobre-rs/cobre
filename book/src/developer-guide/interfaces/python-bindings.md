# Python Bindings

## Purpose

This spec defines the `cobre-python` crate: a PyO3-based `cdylib` that exposes Cobre's SDDP hydrothermal dispatch solver to Python as the `cobre` module. It covers the complete Python API surface with type-annotated signatures for all public classes and functions, the 6-point GIL management contract that governs the Python/Rust boundary, zero-copy data paths via NumPy and Arrow FFI, single-process and multi-process execution modes (the latter via TCP or shared-memory backends -- never MPI) with the rationale for prohibiting MPI from Python, the Python exception hierarchy mapped from the structured error kind registry ([Structured Output](./structured-output.md) SS2.3), optional async support via `asyncio`, the FlatBuffers policy access API, memory ownership rules at the boundary, and build/distribution via maturin.

## 1. Crate Architecture

### 1.1 Crate Type and Dependencies

`cobre-python` is a `cdylib` crate that compiles to a shared library (`.so` / `.dylib` / `.pyd`) loadable by the Python interpreter. It is a leaf crate in the Cobre dependency graph: no internal crate depends on it.

| Attribute             | Value                                                                                                                                                |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Crate type**        | `cdylib` (PyO3 shared library)                                                                                                                       |
| **Module name**       | `import cobre`                                                                                                                                       |
| **Execution mode**    | Single-process only. No MPI. Rayon thread pool for parallel LP solves within the process. GIL released during all Rust computation via `py.detach()` |
| **What it owns**      | PyO3 class/function definitions, Python-to-Rust type conversions, Arrow IPC bridge for zero-copy result loading                                      |
| **What it delegates** | All computation to `cobre-sddp`; all I/O to `cobre-io`; all data model types from `cobre-core`                                                       |
| **MPI relationship**  | MUST NOT depend on `ferrompi`. MUST NOT initialize MPI. For distributed execution, launch `mpiexec cobre` as a subprocess                            |

**Dependency graph**:

```
cobre-python [cdylib, PyO3]
  +-- cobre-sddp
  |     +-- cobre-core
  |     +-- cobre-stochastic
  |     +-- cobre-solver
  +-- cobre-io
  |     +-- cobre-core
  +-- cobre-core

ferrompi: NOT a dependency of cobre-python
```

### 1.2 Single-Process Execution

`cobre-python` invokes `cobre-sddp` in single-rank mode. The initialization sequence from [Hybrid Parallelism](../hpc/hybrid-parallelism.md) SS6 is modified:

| Standard Step (SS6)                       | Python Mode                                                                             |
| ----------------------------------------- | --------------------------------------------------------------------------------------- |
| Step 1 -- MPI initialization              | **Skipped**. No MPI in the process                                                      |
| Step 2 -- Topology detection              | **Skipped**. No scheduler or rank detection                                             |
| Step 3 -- Shared memory communicator      | **Skipped**. No MPI windows                                                             |
| Step 4 -- Rayon configuration             | **Active**. Reads `RAYON_NUM_THREADS` from environment; defaults to physical core count |
| Step 5 -- LP solver threading suppression | **Active**. Validates `HIGHS_PARALLEL=false`, `MKL_NUM_THREADS=1`                       |
| Step 6 -- NUMA allocation policy          | **Active** on Linux. Local allocation via `libnuma` if available                        |
| Step 7 -- Workspace allocation            | **Active**. Thread-local solver workspaces allocated with first-touch NUMA placement    |
| Step 8 -- Startup logging                 | **Modified**. Logs to Python's `logging` module instead of stdout; reports thread count |

### 1.2a Multi-Process Execution

When `num_workers > 1` is requested (or when a non-local backend is explicitly selected), `cobre-python` operates in multi-process mode. The parent Python process spawns `num_workers` child processes via `multiprocessing.Process` (with `start_method="spawn"`). Each worker process is a fully independent Python interpreter that executes the same initialization steps as SS1.2 above, with the following differences:

| Standard Step (SS6)                       | Multi-Process Worker Mode                                                                                                             |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Step 1 -- MPI initialization              | **Replaced**. Creates a `TcpBackend` or `ShmBackend` instead of `LocalBackend`, depending on the selected backend (see SS7.5)         |
| Step 2 -- Topology detection              | **Replaced**. Rank and size are assigned by the parent process via function arguments, not by a scheduler                             |
| Step 3 -- Shared memory communicator      | **Replaced**. For `ShmBackend`, the POSIX shared memory segment provides true shared memory. For `TcpBackend`, `HeapFallback` is used |
| Step 4 -- Rayon configuration             | **Active**. Each worker reads `RAYON_NUM_THREADS` independently; defaults to physical core count divided by `num_workers`             |
| Step 5 -- LP solver threading suppression | **Active**. Same as single-process mode                                                                                               |
| Step 6 -- NUMA allocation policy          | **Active** on Linux. Each worker process has its own NUMA allocation policy                                                           |
| Step 7 -- Workspace allocation            | **Active**. Each worker allocates its own thread-local solver workspaces                                                              |
| Step 8 -- Startup logging                 | **Modified**. Each worker logs to Python's `logging` module with its rank prefix; parent process aggregates status                    |

This maps to the single-process mode described in [Hybrid Parallelism](../hpc/hybrid-parallelism.md) SS1.0a, extended to multiple cooperating processes that communicate via a non-MPI backend. The parent process is the orchestrator: it spawns workers, waits for completion, and collects results. It does not participate in the SDDP computation.

### 1.3 Thread Control

Users control the thread count via the `threads` parameter to `cobre.run.run()`:

```python
import cobre.run

# Use 4 worker threads for parallel LP solves
result = cobre.run.run("path/to/case", threads=4)
```

When `threads` is not specified, the solver runs with 1 thread. The thread pool is initialized via `rayon::ThreadPoolBuilder` at the beginning of each `run()` call. There is no separate `set_threads()` / `get_threads()` API; thread count is a per-invocation parameter.

## 2. Python API Surface

> **Python version support**: Requires Python 3.12, 3.13, or 3.14.

The `cobre` module uses a submodule architecture with four submodules: `cobre.io`, `cobre.run`, `cobre.model`, and `cobre.results`. Each submodule groups related functionality.

```python
import cobre
print(cobre.__version__)   # e.g. "0.4.1"

# Submodule imports
import cobre.io
import cobre.run
import cobre.model
import cobre.results
```

### 2.1 `cobre.io` -- Case Loading and Validation

The `cobre.io` submodule provides entry points for loading and validating case directories.

#### `cobre.io.load_case()`

Loads a case directory and returns a validated `System` object.

```python
def load_case(path: str | os.PathLike) -> cobre.model.System:
    """Load a Cobre case directory and return a validated System.

    Executes the five-layer validation pipeline (structural, schema,
    referential integrity, dimensional consistency, and semantic).
    Returns a fully-validated System on success.

    Args:
        path: Path to the case directory, as a str or pathlib.Path.
            Relative paths are resolved from the process working directory.

    Returns:
        A validated cobre.model.System instance.

    Raises:
        OSError: A required file is missing or cannot be read.
        ValueError: The case data fails schema, referential integrity,
            dimensional consistency, or semantic validation.
    """
    ...
```

Example:

```python
import cobre.io
system = cobre.io.load_case("examples/1dtoy")
print(system.n_buses)
```

#### `cobre.io.validate()`

Validates a case directory without raising on errors.

```python
def validate(path: str | os.PathLike) -> dict:
    """Validate a Cobre case directory and return a structured report dict.

    Unlike load_case(), this function never raises -- all errors are
    returned as data in the result dict. This is intentional: Jupyter
    workflows need to see all validation problems at once rather than
    stopping at the first failure.

    Args:
        path: Path to the case directory, as a str or pathlib.Path.

    Returns:
        A dict with the following keys:

        - "valid" (bool) -- True when the case loaded without errors.
        - "errors" (list[dict]) -- list of error dicts, each with
          "kind" and "message" string fields. Empty when valid is True.
        - "warnings" (list[dict]) -- list of warning dicts in the same
          format. Warnings do not affect the valid flag.
    """
    ...
```

Error kind mapping from Rust `LoadError` variants to Python exceptions:

| Rust variant                     | `validate()` error kind | `load_case()` exception |
| -------------------------------- | ----------------------- | ----------------------- |
| `LoadError::IoError`             | `"IoError"`             | `OSError`               |
| `LoadError::ParseError`          | `"ParseError"`          | `ValueError`            |
| `LoadError::SchemaError`         | `"SchemaError"`         | `ValueError`            |
| `LoadError::CrossReferenceError` | `"CrossReferenceError"` | `ValueError`            |
| `LoadError::ConstraintError`     | `"ConstraintError"`     | `ValueError`            |
| `LoadError::PolicyIncompatible`  | `"PolicyIncompatible"`  | `ValueError`            |

### 2.2 `cobre.run` -- Solver Execution

The `cobre.run` submodule provides the high-level run entry point.

#### `cobre.run.run()`

Runs the full solve lifecycle (load, train, optionally simulate, write results).

```python
def run(
    case_dir: str | os.PathLike,
    output_dir: str | os.PathLike | None = None,
    threads: int | None = None,
    skip_simulation: bool | None = None,
) -> dict:
    """Load a case, train an SDDP policy, optionally simulate, and write results.

    The GIL is released for the entire Rust computation. This function
    replicates the lifecycle of `cobre run` but without MPI, progress
    bars, or a terminal banner.

    Args:
        case_dir: Path to the case directory containing input data files
            and config.json.
        output_dir: Output directory for results. Defaults to
            case_dir/output if not specified.
        threads: Number of worker threads for parallel scenario processing.
            Each thread solves its own LP instances. Defaults to 1.
        skip_simulation: When True, skip the simulation phase even if
            enabled in config.json. Defaults to False.

    Returns:
        A dict with the following keys:

        - "converged" (bool) -- whether training converged.
        - "iterations" (int) -- number of training iterations completed.
        - "lower_bound" (float) -- final lower bound value.
        - "upper_bound" (float | None) -- final upper bound value.
        - "gap_percent" (float | None) -- relative gap as percentage.
        - "total_time_ms" (int) -- total computation time in milliseconds.
        - "output_dir" (str) -- absolute path to the output directory.
        - "simulation" (dict | None) -- simulation summary dict with
          "n_scenarios" and "completed" keys, or None if skipped.
        - "stochastic" (dict | None) -- stochastic preprocessing summary.
        - "hydro_models" (dict | None) -- hydro model summary.
        - "provenance" (dict) -- run provenance metadata with keys
          "cobre_version" (str), "started_at" (str, ISO 8601),
          "finished_at" (str, ISO 8601), "hostname" (str),
          and "config_hash" (str, SHA-256 of the resolved config).

    Raises:
        OSError: If case_dir does not exist or output write fails.
        RuntimeError: If training or simulation encounters a solver
            error, config parse error, or other computation failure.
    """
    ...
```

Example:

```python
import cobre.run

result = cobre.run.run("path/to/case")
print(f"Converged: {result['converged']}")
print(f"Iterations: {result['iterations']}")
print(f"Lower bound: {result['lower_bound']:.2f}")
print(f"Gap: {result['gap_percent']:.2f}%")
print(f"Output dir: {result['output_dir']}")

# With optional parameters
result = cobre.run.run(
    "path/to/case",
    output_dir="path/to/output",   # default: case_dir/output
    threads=4,                      # default: 1
    skip_simulation=True,           # default: False
)
```

### 2.3 `cobre.results` -- Result Loading and Inspection

The `cobre.results` submodule provides functions for reading output artifacts written by `cobre.run.run()`. JSON manifest and metadata files are read in Rust and returned as Python dicts. Parquet files can be loaded as Python dicts or as zero-copy Arrow tables.

#### `cobre.results.load_results()`

```python
def load_results(output_dir: str | os.PathLike) -> dict:
    """Load and inspect the output artifacts produced by a completed solver run.

    Returns a nested dict with training and simulation sections.

    Returns:
        {
            "training": {
                "manifest": { ... },
                "metadata": { ... },
                "convergence_path": "/abs/.../convergence.parquet",
                "timing_path": "/abs/.../timing/iterations.parquet",
                "complete": True,
            },
            "simulation": {
                "manifest": { ... } | None,
                "complete": False,
            },
        }

    Raises:
        FileNotFoundError: If output_dir does not exist or training
            did not complete (no _SUCCESS marker).
        ValueError: If JSON manifest files are malformed.
        OSError: For other I/O errors.
    """
    ...
```

#### `cobre.results.load_convergence()`

```python
def load_convergence(output_dir: str | os.PathLike) -> list[dict]:
    """Read training/convergence.parquet and return rows as a list of dicts.

    Each dict contains: iteration, lower_bound, upper_bound_mean,
    upper_bound_std, gap_percent, cuts_added, cuts_removed, cuts_active,
    time_forward_ms, time_backward_ms, time_total_ms, forward_passes,
    lp_solves.

    Raises:
        FileNotFoundError: If convergence.parquet does not exist.
        OSError: For Parquet decoding failures.
    """
    ...
```

#### `cobre.results.load_convergence_arrow()`

```python
def load_convergence_arrow(output_dir: str | os.PathLike) -> "pyarrow.Table":
    """Read training/convergence.parquet and return as a pyarrow.Table.

    Returns the data in Arrow IPC format for zero-copy consumption by
    polars.from_arrow() or any Arrow-compatible library. Requires
    pyarrow to be installed.

    Raises:
        FileNotFoundError: If convergence.parquet does not exist.
        OSError: For Parquet decoding or IPC serialisation failures.
        ImportError: If pyarrow is not installed.
    """
    ...
```

#### `cobre.results.load_simulation()`

```python
def load_simulation(
    output_dir: str | os.PathLike,
    entity_type: str | None = None,
) -> list[dict] | dict[str, list[dict]]:
    """Load simulation results from Hive-partitioned Parquet files.

    Reads simulation/{entity_type}/scenario_id=NNNN/data.parquet files
    and returns the rows with a scenario_id integer column added.

    Args:
        output_dir: Root output directory.
        entity_type: Optional entity type name ("costs", "buses",
            "hydros", "thermals", "exchanges", "pumping_stations",
            "contracts", "non_controllables", "inflow_lags",
            "violations/generic"). When provided, returns a flat
            list of dicts for that type. When None, returns a dict
            of lists keyed by entity type.

    Raises:
        FileNotFoundError: If output_dir or entity directory is absent.
        OSError: For corrupt Parquet files.
    """
    ...
```

#### `cobre.results.load_simulation_arrow()`

```python
def load_simulation_arrow(
    output_dir: str | os.PathLike,
    entity_type: str | None = None,
) -> "pyarrow.Table | dict[str, pyarrow.Table]":
    """Load simulation results as pyarrow.Table(s).

    Same data as load_simulation() but returned as Arrow tables for
    zero-copy consumption. Requires pyarrow.

    Args:
        output_dir: Root output directory.
        entity_type: Optional entity type. When provided, returns a
            single pyarrow.Table. When None, returns a dict of Tables.

    Raises:
        FileNotFoundError: If output_dir or entity directory is absent.
        OSError: For corrupt Parquet files or IPC errors.
        ImportError: If pyarrow is not installed.
    """
    ...
```

#### `cobre.results.load_policy()`

```python
def load_policy(output_dir: str | os.PathLike) -> dict:
    """Load a policy checkpoint from training/policy/.

    Reads the FlatBuffers policy checkpoint and returns a nested dict
    with metadata, per-stage cut pools, and per-stage solver bases.

    Returns:
        {
            "metadata": { "version": ..., "completed_iterations": ..., ... },
            "stage_cuts": [ { "stage_id": ..., "cuts": [...], ... }, ... ],
            "stage_bases": [ { "stage_id": ..., "column_status": [...], ... }, ... ],
        }

    Raises:
        FileNotFoundError: If output_dir or training/policy/ is absent.
        OSError: For corrupt FlatBuffers files.
    """
    ...
```

### 2.4 `cobre.model` -- Data Model Types

The `cobre.model` submodule exposes read-only wrapper classes for Cobre's core entity types. All wrappers are immutable: Python code reads entity data but cannot mutate it. Construction happens through `cobre.io.load_case()`, not through Python constructors.

Entity IDs are `int` (i32), not strings.

#### `cobre.model.System`

```python
class System:
    """Top-level system representation wrapping a loaded Cobre case.

    Produced by cobre.io.load_case(). Immutable after construction.
    Provides read-only access to entity collections and counts.
    """

    # Entity collection properties (canonical ID order)
    buses: list[Bus]
    lines: list[Line]
    thermals: list[Thermal]
    hydros: list[Hydro]
    contracts: list[EnergyContract]
    pumping_stations: list[PumpingStation]
    non_controllable_sources: list[NonControllableSource]

    # Count properties
    n_buses: int
    n_lines: int
    n_hydros: int
    n_thermals: int
    n_stages: int
```

#### `cobre.model.Bus`

```python
class Bus:
    """Electrical network node where energy balance is maintained."""
    id: int                                      # i32
    name: str
    deficit_segments: list[dict]                  # [{"depth_mw": float|None, "cost_per_mwh": float}]
    excess_cost: float                            # $/MWh
```

#### `cobre.model.Line`

```python
class Line:
    """Transmission interconnection between two buses."""
    id: int                                      # i32
    name: str
    source_bus_id: int                           # i32
    target_bus_id: int                           # i32
    direct_capacity_mw: float                    # MW (source -> target)
    reverse_capacity_mw: float                   # MW (target -> source)
    losses_percent: float                        # e.g. 2.5 means 2.5%
    exchange_cost: float                         # $/MWh regularization
```

#### `cobre.model.Thermal`

```python
class Thermal:
    """Thermal power plant with piecewise-linear generation cost curve."""
    id: int                                      # i32
    name: str
    bus_id: int                                  # i32
    min_generation_mw: float                     # MW (minimum stable load)
    max_generation_mw: float                     # MW (installed capacity)
    cost_segments: list[dict]                    # [{"capacity_mw": float, "cost_per_mwh": float}]
```

#### `cobre.model.Hydro`

```python
class Hydro:
    """Hydroelectric power plant with reservoir storage and cascade topology."""
    id: int                                      # i32
    name: str
    bus_id: int                                  # i32
    downstream_id: int | None                    # i32 or None
    min_storage_hm3: float                       # hm3 (dead volume)
    max_storage_hm3: float                       # hm3 (flood control level)
    min_turbined_m3s: float                      # m3/s
    max_turbined_m3s: float                      # m3/s (installed turbine capacity)
    productivity_mw_per_m3s: float | None        # MW/(m3/s), None for FPHA model
```

#### `cobre.model.EnergyContract`

```python
class EnergyContract:
    """Bilateral energy contract with an external system (stub entity).

    In the minimal viable solver this entity is data-complete but
    contributes no LP variables or constraints.
    """
    id: int                                      # i32
    name: str
```

#### `cobre.model.PumpingStation`

```python
class PumpingStation:
    """Pumping station that transfers water between hydro reservoirs (stub entity).

    In the minimal viable solver this entity is data-complete but
    contributes no LP variables or constraints.
    """
    id: int                                      # i32
    name: str
```

#### `cobre.model.NonControllableSource`

```python
class NonControllableSource:
    """Intermittent generation source that cannot be dispatched (stub entity).

    In the minimal viable solver this entity is data-complete but
    contributes no LP variables or constraints.
    """
    id: int                                      # i32
    name: str
```

> **Status: PLANNED -- NOT YET IMPLEMENTED.** Sections 2.5 through 2.10 document a class-based Python API that is designed but not yet present in the codebase. The current implementation exposes a functional, dict-based API (SS2.1 through SS2.4). The classes below represent the planned future API surface and are retained here as a design reference. Do not write code against these classes until they appear in the `cobre` module.

### 2.5 PARModel

```python
class PARModel:
    """Periodic autoregressive model for inflow generation.

    Wraps the PAR(p) model from cobre-stochastic with NumPy interfaces
    for parameters and noise sampling.
    """

    @property
    def order(self) -> int:
        """Maximum AR order across all entities and seasons."""
        ...

    @property
    def n_entities(self) -> int:
        """Number of entities (hydro plants) in the model."""
        ...

    @property
    def n_seasons(self) -> int:
        """Number of seasons (typically 12 for monthly)."""
        ...

    def seasonal_means(self, entity_index: int) -> numpy.ndarray:
        """Return seasonal means for a given entity.

        Args:
            entity_index: Zero-based entity index.

        Returns:
            numpy.ndarray of shape (n_seasons,) with seasonal mean inflows.
            Zero-copy: backed by the Rust contiguous array.
        """
        ...

    def ar_coefficients(self, entity_index: int, season: int) -> numpy.ndarray:
        """Return AR coefficients for a given entity and season.

        Args:
            entity_index: Zero-based entity index.
            season: Zero-based season index.

        Returns:
            numpy.ndarray of shape (order,) with AR coefficients phi_1, ..., phi_p.
            Zero-copy: backed by the Rust contiguous array.
        """
        ...

    def correlation_matrix(self, season: int) -> numpy.ndarray:
        """Return the spatial correlation matrix for a season.

        Args:
            season: Zero-based season index.

        Returns:
            numpy.ndarray of shape (n_entities, n_entities).
            Zero-copy: backed by the Rust contiguous array.
        """
        ...
```

### 2.6 Stochastic Functions

```python
def sample_noise(
    n_samples: int,
    correlation_matrix: numpy.ndarray,
    rng_seed: int | None = None,
) -> numpy.ndarray:
    """Sample correlated noise vectors.

    Generates n_samples correlated noise vectors using spectral
    decomposition (eigendecomposition) of the correlation matrix.

    Args:
        n_samples: Number of samples to generate.
        correlation_matrix: numpy.ndarray of shape (n_entities, n_entities).
        rng_seed: Optional random seed for reproducibility.

    Returns:
        numpy.ndarray of shape (n_samples, n_entities).
    """
    ...

class OpeningTree:
    """Read-only opening tree for backward pass scenarios.

    The opening tree is a 3D array of noise realizations used in the
    backward pass to evaluate the expected future cost function.
    """

    @property
    def data(self) -> numpy.ndarray:
        """The opening tree as a NumPy array.

        Returns:
            numpy.ndarray of shape (n_openings, n_stages, n_entities).
            Zero-copy: backed by the Rust contiguous 3D array.
        """
        ...

    @property
    def n_openings(self) -> int:
        """Number of openings (backward scenarios)."""
        ...

    @property
    def n_stages(self) -> int:
        """Number of stages."""
        ...

    @property
    def n_entities(self) -> int:
        """Number of entities."""
        ...

def load_external_scenarios(path: str | os.PathLike) -> pyarrow.Table:
    """Load external scenario data from a Parquet file.

    Reads a Parquet file containing externally generated scenario data
    and returns it as an Arrow table (zero-copy via Arrow FFI).

    Args:
        path: Path to the Parquet file.

    Returns:
        pyarrow.Table with scenario data columns.

    Raises:
        cobre.IOError: If the file does not exist or is unreadable.
    """
    ...
```

### 2.7 TrainingResult

```python
class TrainingResult:
    """Result of an SDDP training run."""

    @property
    def policy(self) -> Policy:
        """The trained policy (cut collection)."""
        ...

    @property
    def convergence_history(self) -> pyarrow.Table:
        """Convergence history as an Arrow table.

        Columns match the per-iteration output record from
        convergence-monitoring.md SS2.4:
        iteration, lower_bound, upper_bound, upper_bound_std,
        ci_95, gap, wall_time_ms, iteration_time_ms.

        Zero-copy: transferred via Arrow FFI.
        """
        ...

    @property
    def iterations(self) -> int:
        """Total number of training iterations completed."""
        ...

    @property
    def final_lower_bound(self) -> float:
        """Final lower bound value."""
        ...

    @property
    def final_upper_bound(self) -> float:
        """Final upper bound value."""
        ...

    @property
    def final_gap(self) -> float:
        """Final relative convergence gap."""
        ...

    @property
    def termination_reason(self) -> str:
        """Stopping rule that triggered termination.

        One of: 'bound_stalling', 'simulation', 'iteration_limit',
        'time_limit', 'shutdown'.
        """
        ...

    @property
    def total_cuts(self) -> int:
        """Total Benders cuts generated across all stages."""
        ...

    @property
    def workers(self) -> list[WorkerInfo] | None:
        """Per-worker metadata for multi-process training runs.

        Returns a list of WorkerInfo instances (one per worker, ordered
        by rank) when the training run used num_workers > 1. Returns
        None when num_workers == 1 (single-process mode).
        """
        ...


class WorkerInfo:
    """Metadata for a single worker process in a multi-process run.

    Instances are created by the library during result collection
    (SS2.1b) and are read-only.
    """

    @property
    def rank(self) -> int:
        """Worker rank index (0-based), as assigned during spawning (SS2.1a step 4)."""
        ...

    @property
    def wall_time_ms(self) -> int:
        """Wall-clock time for this worker in milliseconds.

        Measured from the start of the worker's SDDP computation
        (after Communicator initialization) to the end of its
        training loop. Does not include process spawn overhead.
        """
        ...

    @property
    def backend(self) -> str:
        """Communication backend used by this worker.

        One of: 'shm', 'tcp', 'local'. Reflects the actual
        backend selected after auto-detection resolution.
        """
        ...
```

### 2.8 SimulationResult

```python
class SimulationResult:
    """Result of an SDDP simulation run."""

    @property
    def scenarios(self) -> int:
        """Number of simulation scenarios evaluated."""
        ...

    @property
    def output_directory(self) -> str:
        """Path to the output directory containing Parquet files.

        Users can read simulation results directly from the Parquet files
        using polars or pyarrow:
            import polars as pl
            costs = pl.read_parquet(f"{result.output_directory}/simulation/costs.parquet")
        """
        ...

    @property
    def output_files(self) -> list[str]:
        """List of output Parquet file paths (relative to output_directory)."""
        ...

    @property
    def mean_cost(self) -> float:
        """Mean total cost across all simulation scenarios."""
        ...

    @property
    def std_cost(self) -> float:
        """Standard deviation of total costs across scenarios."""
        ...

    @property
    def wall_time_ms(self) -> int:
        """Simulation wall-clock time in milliseconds."""
        ...
```

### 2.9 Progress Callbacks

```python
class ProgressEvent:
    """Progress event emitted during training or simulation.

    Training events carry convergence data matching the per-iteration
    record from convergence-monitoring.md SS2.4. Simulation events carry
    scenario completion counts.
    """

    @property
    def phase(self) -> str:
        """Phase: 'training' or 'simulation'."""
        ...

    # Training-phase fields (None during simulation)
    @property
    def iteration(self) -> int | None:
        """Current iteration (1-based). None during simulation."""
        ...

    @property
    def lower_bound(self) -> float | None:
        """Current lower bound. None during simulation."""
        ...

    @property
    def upper_bound(self) -> float | None:
        """Current upper bound. None during simulation."""
        ...

    @property
    def gap(self) -> float | None:
        """Current relative gap. None during simulation."""
        ...

    @property
    def wall_time_ms(self) -> int:
        """Cumulative wall-clock time in milliseconds."""
        ...

    @property
    def iteration_time_ms(self) -> int | None:
        """Time for this iteration in milliseconds. None during simulation."""
        ...

    # Simulation-phase fields (None during training)
    @property
    def scenarios_complete(self) -> int | None:
        """Scenarios completed so far. None during training."""
        ...

    @property
    def scenarios_total(self) -> int | None:
        """Total scenarios to simulate. None during training."""
        ...

    # Multi-process fields
    @property
    def worker_id(self) -> int | None:
        """Worker rank (0-based) that emitted this event.

        None in single-process mode. In multi-process mode, progress
        events from all workers are multiplexed into the single
        progress_callback stream; this field disambiguates the source.
        Events are delivered in arrival order across workers; events
        from the same worker are in iteration order.
        """
        ...
```

### 2.10 ValidationResult

```python
class ValidationResult:
    """Result of a case directory validation."""

    @property
    def valid(self) -> bool:
        """True if no errors were found (warnings are allowed)."""
        ...

    @property
    def errors(self) -> list[ValidationRecord]:
        """List of validation error records."""
        ...

    @property
    def warnings(self) -> list[ValidationRecord]:
        """List of validation warning records."""
        ...

    @property
    def layers_completed(self) -> int:
        """Number of validation layers fully completed (1-5)."""
        ...


class ValidationRecord:
    """A single validation error or warning.

    Fields match the structured error schema from structured-output.md SS2.1.
    """

    @property
    def kind(self) -> str:
        """Error kind identifier from the registry."""
        ...

    @property
    def message(self) -> str:
        """Human-readable error description."""
        ...

    @property
    def context(self) -> dict:
        """Structured context data (file, entity_id, field, etc.)."""
        ...

    @property
    def suggestion(self) -> str | None:
        """Actionable remediation hint, or None."""
        ...
```

### 2.11 API Surface Summary

| Source Crate     | Python Classes / Functions                                                                                                                        | Exposed? |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| cobre-core       | `System`, `Hydro`, `Thermal`, `Bus`, `Line`, `EnergyContract`, `PumpingStation`, `NonControllableSource` (read-only entities)                     | Yes      |
| cobre-io         | `cobre.io.load_case()`, `cobre.io.validate()`                                                                                                     | Yes      |
| cobre-sddp       | `cobre.run.run()` (full lifecycle entry point)                                                                                                    | Yes      |
| cobre-io/results | `cobre.results.load_results()`, `load_convergence()`, `load_convergence_arrow()`, `load_simulation()`, `load_simulation_arrow()`, `load_policy()` | Yes      |
| cobre-solver     | (none)                                                                                                                                            | **No**   |
| ferrompi         | (none)                                                                                                                                            | **No**   |

## 3. GIL Management Contract

> **Decision [DEC-012](../overview/decision-log.md#dec-012) (active):** 6-point GIL management contract governs the Python/Rust boundary; MPI is prohibited from Python bindings for 3 independent reasons (`MPI_Init_thread` timing conflict, GIL vs `MPI_THREAD_MULTIPLE` deadlock risk, dual-FFI-layer fragility).

The Global Interpreter Lock (GIL) is the central concurrency constraint at the Python/Rust boundary. The following 6-point contract governs all interactions between Python and Cobre's Rust computation.

### 3.1 The 6-Point GIL Contract

1. **GIL acquired to receive Python call and validate arguments.** When Python calls a PyO3-wrapped function (e.g., `train()`), the GIL is held. PyO3 validates and converts arguments from Python objects to Rust types.

2. **Thread state detached via `py.detach()` before entering Rust computation.** Before invoking any Rust computation (LP solves, cut generation, scenario sampling), the binding code calls `py.detach(|| { ... })` to detach from the Python runtime. On GIL-enabled builds, this releases the GIL, allowing other Python threads to execute. On free-threaded builds (see SS7.5a), this detaches the thread state, preventing the thread from blocking stop-the-world synchronization events (GC, tracing). The same code is correct on both build types.

3. **No Rust thread within a Rayon parallel region ever acquires the GIL.** Rayon threads spawned during the training or simulation loop are pure Rust threads. They never call back into Python, never acquire the GIL, and never touch `PyObject` references. This is guaranteed by construction: the Rayon parallel iterator pattern ([Hybrid Parallelism](../hpc/hybrid-parallelism.md) SS5.3) calls Rust closures that have no access to `Python<'_>` tokens.

4. **GIL reacquired to convert results to Python objects on return.** When Rust computation completes, `py.detach()` returns, reacquiring the GIL. The binding code converts Rust results to Python objects (NumPy arrays, dicts, PyO3 class instances).

5. **No Python callbacks into the hot loop.** All customization is via configuration (`config_overrides` dict), not runtime callbacks. The optional `progress_callback` is invoked only at iteration boundaries (outside the LP solve parallel regions) with the GIL briefly reacquired.

6. **In multi-process mode, each worker process has its own Python interpreter and GIL.** Every worker spawned via `multiprocessing.Process` with `start_method="spawn"` runs a fresh Python interpreter with an independent GIL. The GIL contract (points 1-5) applies independently within each worker process. There is no GIL contention between workers because they are separate OS processes -- each worker releases its own GIL before entering Rust computation, and no GIL state is shared across process boundaries.

### 3.2 GIL State Transitions During `train()`

The following timeline shows GIL state transitions during a complete `train()` call:

```
Python thread                     Rust / Rayon threads
=============                     ====================

train(case, config)
  |
  +-- [GIL HELD] --------+
  |   Validate arguments  |
  |   Convert Python args |
  |   to Rust types       |
  +-- py.detach() ---------+
  |                        \
  |   [GIL RELEASED]       +-- Initialize training loop
  |                        |   Allocate solver workspaces
  |                        |
  |                        +-- for iteration = 1..N:
  |                        |     |
  |                        |     +-- [Rayon] Forward pass
  |                        |     |   (parallel LP solves, no GIL)
  |                        |     |
  |                        |     +-- Forward sync (local, no MPI)
  |                        |     |
  |                        |     +-- [Rayon] Backward pass
  |                        |     |   (parallel LP solves, no GIL)
  |                        |     |
  |                        |     +-- Convergence check
  |                        |     |
  |                        |     +-- [if callback] ----+
  |   [GIL REACQUIRED]  <-|-----+   Reacquire GIL     |
  |   Invoke callback     -|-----+   Call Python fn    |
  |   [GIL RELEASED]    ->|-----+   Release GIL       |
  |                        |     +---------------------+
  |                        |     |
  |                        |     +-- [if converged] break
  |                        |
  |                        +-- Collect results
  +-- [GIL REACQUIRED] ---/
  |   Convert results to  |
  |   Python objects       |
  +------------------------+
  |
  return TrainingResult
```

### 3.2a GIL State Transitions During Multi-Process `train()`

When `num_workers > 1`, the parent process spawns child processes and waits for results. Each child process follows the single-process GIL flow from SS3.2 independently. The following timeline shows the parent and worker GIL state transitions:

```
Parent Process                    Worker Processes (spawned)
==============                    =========================

train(case, num_workers=N)
  |
  +-- [GIL HELD] -----------+
  |   Validate arguments     |
  |   Check start_method     |
  |   Resolve backend        |
  |   Generate backend config|
  +-- [GIL HELD] -----------+
  |   Spawn N workers via    |
  |   multiprocessing.Process|
  +--------------------------+
  |                           \
  |                            +---> Worker 0           Worker 1  ...  Worker N-1
  |                            |     ==========         ==========     ==========
  |                            |
  |                            |     [NEW INTERPRETER]  [NEW INTERPRETER]
  |                            |     [OWN GIL]          [OWN GIL]
  |                            |       |                  |
  |                            |     [GIL HELD]         [GIL HELD]
  |                            |     Import cobre       Import cobre
  |                            |     Create Communicator Create Communicator
  |                            |       |                  |
  |                            |     py.detach()        py.detach()
  |                            |       |                  |
  |                            |     [GIL RELEASED]     [GIL RELEASED]
  |                            |     SDDP loop          SDDP loop
  |                            |     (SS3.2 flow)       (SS3.2 flow)
  |                            |     allgatherv <------> allgatherv
  |                            |     allreduce  <------> allreduce
  |                            |       |                  |
  |  [GIL HELD]                |     [if callback]      [if callback]
  |  Process.join() blocks     |     Reacquire own GIL  Reacquire own GIL
  |  (waiting for workers)     |     Call callback      Call callback
  |                            |     (worker_id=0)      (worker_id=1)
  |                            |     Release own GIL    Release own GIL
  |                            |       |                  |
  |                            |     [if converged]     [if converged]
  |                            |     Put result on Q    (no result)
  |                            |     Process exits      Process exits
  |                            |       |                  |
  +-- join() returns ----------+-------+------------------+
  |   [GIL HELD]               |
  |   Read result from Queue   |
  |   Build TrainingResult     |
  |   Attach WorkerInfo list   |
  +----------------------------+
  |
  return TrainingResult
```

**Key observations:**

- Each worker has its own Python interpreter and its own GIL. Releasing one worker's GIL has no effect on other workers or the parent.
- Worker-to-worker communication (allgatherv, allreduce) occurs entirely in Rust within the GIL-released section of each worker. No Python objects cross process boundaries during SDDP computation.
- Progress callbacks are invoked by each worker independently; the parent multiplexes them via `multiprocessing.Queue` with a `worker_id` field (see SS2.9).

### 3.3 Progress Callback GIL Protocol

When a `progress_callback` is provided, the GIL is reacquired at each iteration boundary -- **outside** any Rayon parallel region -- to invoke the callback. The sequence per iteration is:

1. All Rayon threads complete the backward pass and join (implicit barrier).
2. The main Rust thread reacquires the GIL via `Python::with_gil(|py| { ... })`.
3. The `ProgressEvent` is constructed as a Python object.
4. The callback is invoked with the event.
5. The GIL is released again before the next iteration begins.

This ensures that callback execution is serialized and does not interfere with parallel computation. The overhead is one GIL acquire/release pair per iteration, which is negligible compared to the iteration's computation time.

> **Invariant**: The progress callback is the only code path that reacquires the GIL during a `train()` or `simulate()` call. If no callback is provided, the GIL remains released for the entire computation.

## 4. Zero-Copy Data Paths

Zero-copy transfer avoids duplicating large arrays when passing data between Rust and Python. Cobre uses two mechanisms: PyO3's `numpy` crate for contiguous `f64` arrays, and Arrow FFI for tabular data.

### 4.1 Zero-Copy Mechanisms

| Data                   | Rust Type                                   | Python Type                           | Mechanism              | Copy Required? |
| ---------------------- | ------------------------------------------- | ------------------------------------- | ---------------------- | -------------- |
| Scenario noise vectors | `Vec<f64>` (contiguous)                     | `numpy.ndarray`                       | PyO3 `numpy` crate     | **No**         |
| Opening tree           | 3D contiguous `f64` array                   | `numpy.ndarray`                       | PyO3 `numpy` crate     | **No**         |
| Cut coefficients       | `Vec<f64>` per stage                        | `numpy.ndarray`                       | PyO3 `numpy` crate     | **No**         |
| PAR parameters         | Contiguous `f64` arrays                     | `numpy.ndarray`                       | PyO3 `numpy` crate     | **No**         |
| Convergence history    | `Vec<ConvergenceRecord>` (struct-of-arrays) | `pyarrow.Table`                       | Arrow FFI (`arrow-rs`) | **No**         |
| Simulation results     | Parquet files on disk                       | `polars.DataFrame` or `pyarrow.Table` | Python reads directly  | **No** (disk)  |

### 4.2 NumPy Zero-Copy Details

For contiguous `f64` arrays, the PyO3 `numpy` crate creates a `numpy.ndarray` that directly references the Rust-owned memory. The Rust `Vec<f64>` is moved into a `PyArray` that takes ownership, ensuring the backing memory lives as long as the Python object.

```rust
// Rust side (illustrative)
fn cuts_coefficients(py: Python<'_>, stage: usize) -> PyResult<Py<PyArray2<f64>>> {
    let coeffs: Vec<f64> = self.inner.cut_coefficients(stage);
    let n_cuts = self.inner.cut_count(stage);
    let n_states = self.inner.state_dimension();
    let array = PyArray2::from_vec2(py, &coeffs, [n_cuts, n_states])?;
    Ok(array.into())
}
```

**Lifetime rule**: Once a NumPy array is created from Rust data, the Rust `Vec` is consumed. The Python garbage collector manages the array's lifetime. No Rust lifetimes cross the Python boundary.

### 4.3 Arrow FFI Details

For tabular data (convergence history), Cobre uses the Arrow FFI bridge between `arrow-rs` (Rust) and `pyarrow` (Python). This transfers Arrow `RecordBatch` objects without copying column data.

```rust
// Rust side (illustrative)
fn convergence_history(py: Python<'_>) -> PyResult<PyObject> {
    let batches: Vec<RecordBatch> = self.inner.convergence_batches();
    // Export via Arrow FFI (C Data Interface)
    let py_table = arrow_to_pyarrow(py, &batches)?;
    Ok(py_table)
}
```

**Column mapping**: The Arrow table columns correspond to the [Convergence Monitoring](../architecture/convergence-monitoring.md) SS2.4 per-iteration record fields:

| Arrow Column        | Data Type | Description                         |
| ------------------- | --------- | ----------------------------------- |
| `iteration`         | `Int32`   | Iteration index (1-based)           |
| `lower_bound`       | `Float64` | Lower bound value                   |
| `upper_bound`       | `Float64` | Upper bound value (mean forward)    |
| `upper_bound_std`   | `Float64` | Standard deviation of forward costs |
| `ci_95`             | `Float64` | 95% confidence interval half-width  |
| `gap`               | `Float64` | Relative gap                        |
| `wall_time_ms`      | `Int64`   | Cumulative wall-clock time (ms)     |
| `iteration_time_ms` | `Int64`   | Iteration wall-clock time (ms)      |

### 4.4 Simulation Results: Python Reads Directly

Simulation outputs are written as Hive-partitioned Parquet files to disk. Python users read these directly with their preferred Parquet library. No Rust-to-Python data transfer is needed.

```python
result = cobre.simulate(case, policy)

# Read with polars (recommended for performance)
import polars as pl
costs = pl.read_parquet(f"{result.output_directory}/simulation/costs.parquet")

# Read with pyarrow
import pyarrow.parquet as pq
hydros = pq.read_table(f"{result.output_directory}/simulation/hydros.parquet")
```

This approach avoids keeping large simulation datasets in Rust memory and leverages Python's mature Parquet ecosystem.

## 5. FlatBuffers Policy Access

The `Policy` class provides a high-level Python API for inspecting SDDP policies stored in FlatBuffers format. FlatBuffers internals are not exposed to Python users.

### 5.1 Policy Class

```python
class Policy:
    """Trained SDDP policy (cut collection).

    Provides read-only access to the piecewise-linear future cost function
    approximation stored as Benders cuts. Internally backed by FlatBuffers
    binary data, but the Python API exposes only Python-native types.
    """

    @staticmethod
    def load(path: str | os.PathLike) -> Policy:
        """Load a policy from a directory.

        Reads the FlatBuffers policy files from the given directory
        (the 'policy/' subdirectory of a training output).

        Args:
            path: Path to the policy directory.

        Returns:
            A Policy object.

        Raises:
            cobre.IOError: If the path does not exist or files are corrupted.
        """
        ...

    def summary(self) -> dict:
        """Return policy summary statistics.

        Returns:
            Dictionary with keys:
            - 'stages': int -- Number of stages.
            - 'state_dimension': int -- Number of state variables (hydro plants).
            - 'total_cuts': int -- Total cuts across all stages.
            - 'cuts_per_stage': list[int] -- Cut count per stage.
            - 'active_cuts': int -- Number of active (non-dominated) cuts.
        """
        ...

    def cuts(self, stage: int) -> dict:
        """Return cut data for a specific stage.

        Args:
            stage: Stage index (1-based, matching SDDP convention).

        Returns:
            Dictionary with keys:
            - 'intercepts': numpy.ndarray of shape (n_cuts,)
                  The alpha_k coefficients (RHS constants).
                  Zero-copy: backed by Rust memory.
            - 'coefficients': numpy.ndarray of shape (n_cuts, n_states)
                  The beta_k coefficient matrix.
                  Row k contains the subgradient for cut k.
                  Zero-copy: backed by Rust memory.
            - 'active': numpy.ndarray of shape (n_cuts,), dtype=bool
                  Whether each cut is currently active.

        Raises:
            IndexError: If stage is out of range.
        """
        ...

    def evaluate(self, state: numpy.ndarray) -> float:
        """Evaluate the future cost function at a given state.

        Computes max_k { alpha_k + beta_k^T * state } over all cuts.

        Args:
            state: numpy.ndarray of shape (n_states,) representing
                reservoir volumes (in hm3).

        Returns:
            The FCF value (maximum of all affine cut functions).

        Raises:
            ValueError: If state has incorrect dimension.
        """
        ...
```

### 5.2 Design Rationale

FlatBuffers are not exposed directly for several reasons:

1. **Usability**: Python users expect dicts, NumPy arrays, and simple types, not binary buffer protocols.
2. **Stability**: The FlatBuffers schema is an internal implementation detail subject to change. The `Policy` class provides a stable API.
3. **Safety**: Direct `memoryview` access to mmap'd FlatBuffers files could lead to use-after-free if the file is modified externally. The `Policy` class manages file lifetimes correctly.

For advanced users who need raw FlatBuffers access (e.g., for custom tooling), the underlying bytes can be exported:

```python
raw_bytes: bytes = policy.raw_bytes(stage=5)  # Copy of the FlatBuffers data
```

This returns a Python `bytes` object (a copy, not zero-copy) to avoid lifetime issues.

## 6. Error Handling

### 6.1 Exception Mapping

Cobre's Python bindings use **standard Python exceptions** rather than a custom exception hierarchy. The PyO3 layer translates Rust error types from the structured error kind registry ([Structured Output](./structured-output.md) SS2.3) into the three standard exception types below:

```
Exception (Python built-in)
  +-- OSError       -- file/IO errors (missing files, write failures, corrupt outputs)
  +-- ValueError    -- validation and configuration errors (schema, referential, semantic)
  +-- RuntimeError  -- solver failures, internal errors, computation errors
```

This mapping is intentional: Python users expect standard exceptions in `try/except` blocks, and tools like `pytest.raises` work without importing custom exception types. The error message string includes the structured error kind from the registry for programmatic disambiguation when needed.

### 6.1a Worker Error Handling

In multi-process execution (`num_workers > 1`), when a worker process fails, the parent process raises a `RuntimeError` with a message that includes the failed worker's rank, exit code, and (when available) the original error message from the worker.

**Error propagation protocol:**

1. Each worker process runs inside a `try/except` block that catches all exceptions.
2. If a worker catches an exception, it places the error message and kind onto a shared error `multiprocessing.Queue` along with its rank, then exits with a non-zero exit code.
3. The parent process monitors worker processes via `Process.join()`. When a worker exits with a non-zero exit code (or is terminated by a signal), the parent:
   a. Reads the error from the error queue (if available).
   b. Calls `Process.terminate()` on all remaining live workers.
   c. Calls `Process.join()` on all terminated workers (with a timeout) to clean up OS resources.
   d. Raises `RuntimeError` with a message containing the rank, exit code, and inner error details from the first detected failure.
4. If multiple workers fail concurrently, only the first detected failure is reported. The message notes that additional workers may have failed.

### 6.2 Error Kind to Exception Mapping

| Error Kind (structured-output.md SS2.3) | Python Exception | Notes                                   |
| --------------------------------------- | ---------------- | --------------------------------------- |
| `MissingFile`                           | `OSError`        | Missing input file                      |
| `ParseError`                            | `ValueError`     | Malformed JSON/Parquet                  |
| `SchemaViolation`                       | `ValueError`     | Schema mismatch                         |
| `TypeMismatch`                          | `ValueError`     | Wrong data type                         |
| `OutOfRange`                            | `ValueError`     | Numeric value out of bounds             |
| `InvalidEnum`                           | `ValueError`     | Invalid enum string                     |
| `DuplicateId`                           | `ValueError`     | Duplicate entity ID                     |
| `MissingReference`                      | `ValueError`     | Broken foreign key                      |
| `CoverageMismatch`                      | `ValueError`     | Incomplete dimensional coverage         |
| `StageMismatch`                         | `ValueError`     | Stage-related inconsistency             |
| `IncompatibleSettings`                  | `ValueError`     | Mutually incompatible settings          |
| `PhysicalConstraint`                    | `ValueError`     | Domain rule violation                   |
| `CapacityViolation`                     | `ValueError`     | Inconsistent capacity bounds            |
| `PenaltyConsistency`                    | `ValueError`     | Penalty ordering violation              |
| `SolverFailure`                         | `RuntimeError`   | LP solver returned unexpected status    |
| `MpiError`                              | `RuntimeError`   | Not applicable in Python (no MPI)       |
| `CheckpointFailed`                      | `RuntimeError`   | Checkpoint read/write failure           |
| `OutputCorrupted`                       | `OSError`        | Output file exists but is unreadable    |
| `OutputNotFound`                        | `OSError`        | Required output file missing            |
| `IncompatibleRuns`                      | `ValueError`     | Compared runs have incompatible configs |
| `WorkerFailed`                          | `RuntimeError`   | Worker process failed (see SS6.1a)      |

### 6.3 Rust Panic Handling

If a Rust panic occurs during GIL-released computation (e.g., a Rayon thread panics), the panic is caught at the Rust boundary by PyO3's panic-catching mechanism and translated to a `RuntimeError` with a message prefixed by `"InternalPanic: "` followed by the panic message string.

> **Invariant**: No Rust panic ever propagates across the FFI boundary as undefined behavior. PyO3 converts all panics to Python exceptions.

### 6.4 GIL Release Failure

A failure to release the GIL (i.e., `py.detach()` failing) should never happen with correct PyO3 usage. If it occurs, it indicates a programming error in the binding code. This case is not a runtime-recoverable error; it manifests as a `pyo3::PanicException` with a diagnostic message.

## 7. Threading Model

### 7.1 Execution Mode Table

| Execution Mode                              | Supported         | Thread Count                     | GIL State During Computation      | Use Case                                        |
| ------------------------------------------- | ----------------- | -------------------------------- | --------------------------------- | ----------------------------------------------- |
| Single-process, single-thread               | Yes               | `RAYON_NUM_THREADS=1`            | Released                          | Small problems, debugging                       |
| Single-process, Rayon threads               | **Yes (default)** | `RAYON_NUM_THREADS=N`            | Released                          | Production use from Python                      |
| Multi-process via `multiprocessing.Process` | Yes (optional)    | `RAYON_NUM_THREADS=N` per worker | Released (per-worker independent) | Multi-worker SDDP via `multiprocessing.Process` |
| Multi-process via MPI                       | **No**            | --                               | --                                | Use `cobre` CLI via subprocess                  |

**`start_method` requirement:** Multi-process execution MUST use `multiprocessing.set_start_method("spawn")`. The `"fork"` start method is prohibited because `fork()` in a process with active Rayon threads causes undefined behavior (POSIX fork-safety rules). The `"spawn"` method creates a fresh Python interpreter per worker, avoiding all fork-safety issues with Rayon, POSIX locks, and GPU driver state.

### 7.2 MPI Prohibition Rationale

Multi-process SDDP execution from Python is fully supported via the TCP and shared-memory backends (see SS7.3, SS7.4, SS7.5). These backends provide inter-process communication without any MPI dependency, using standard OS primitives (TCP sockets, POSIX shared memory) that are fully compatible with Python's process model. The prohibition below applies exclusively to MPI -- not to distributed execution in general.

Python bindings MUST NOT initialize MPI. The prohibition rests on three independent technical reasons:

1. **`MPI_Init_thread` timing conflict.** ferrompi requires `MPI_Init_thread(MPI_THREAD_MULTIPLE)` as the very first MPI call ([Hybrid Parallelism](../hpc/hybrid-parallelism.md) SS6, Step 1). In a Python process, the interpreter initializes first. If the Python process attempts to initialize MPI via ferrompi, the MPI runtime may conflict with Python's signal handlers, thread state, and memory allocator. Some MPI implementations (OpenMPI, MPICH) are incompatible with being initialized after a Python interpreter is already running with threads.

2. **GIL vs `MPI_THREAD_MULTIPLE` deadlock risk.** `MPI_THREAD_MULTIPLE` means any thread can call MPI at any time. Python's GIL means only one thread can execute Python code at a time. When PyO3 holds the GIL to return results to Python, all other threads that might need to call MPI collective operations (e.g., `MPI_Allreduce`, `MPI_Allgatherv`) are blocked if they need to interact with Python. This creates a deadlock risk: MPI collectives require all ranks to participate, but the GIL serializes execution through Python.

3. **Dual-FFI-layer fragility.** Combining mpi4py (Python MPI bindings) with ferrompi (Rust MPI bindings) in the same process requires both to share the same `MPI_Comm` handle. Having two independent MPI FFI layers addressing the same MPI runtime is fragile and untested. The risk of ABI conflicts, double-free of communicators, or inconsistent threading levels is high.

### 7.3 Multi-Process Execution via Python

The recommended approach for multi-process SDDP from Python uses `multiprocessing.Process` with the shm or TCP backend. Each worker process is a separate Python interpreter that calls `cobre.train()` with backend-specific parameters. The workers communicate via the selected backend (TCP sockets or POSIX shared memory) -- not via Python IPC mechanisms.

**Architecture overview:**

```
  Parent Process (orchestrator)
  |
  |  multiprocessing.Process(target=run_rank, args=(rank,))
  |  start_method="spawn"
  |
  +---> Worker 0 (rank 0)    --+
  +---> Worker 1 (rank 1)    --+-- communicate via shm or TCP
  +---> Worker 2 (rank 2)    --+   (not Python IPC)
  +---> Worker 3 (rank 3)    --+
  |
  |  p.join() for all workers
  v
  Results available on disk (Parquet/FlatBuffers)
```

The complete Python code example for multi-process execution with the shm backend is provided in [Shared Memory Backend](../hpc/backend-shm.md) §7.3. That example demonstrates the high-level `cobre.train(num_workers=N, backend="shm")` call and the internal worker lifecycle.

For the TCP backend, the calling convention is identical: `cobre.train(case, num_workers=N, backend="tcp")`. The library auto-generates the TCP coordinator address and port internally. See [TCP Backend](../hpc/backend-tcp.md) §8.1 for the environment variables used by the library when configuring worker processes.

### 7.3a Subprocess + CLI Workflow (Secondary Option)

For users who need MPI-based distributed execution (multi-node with InfiniBand), or who prefer process isolation from the Python interpreter, the CLI subprocess workflow remains available:

```python
import subprocess
import polars as pl

# Step 1: Prepare case directory from Python
case_dir = "/data/my_study"
# ... write config.json, entity registries, scenario data ...

# Step 2: Launch distributed training via CLI
result = subprocess.run(
    ["mpiexec", "-n", "8", "cobre", "run", case_dir,
     "--output-format", "json"],
    capture_output=True, text=True, check=True,
)
import json
output = json.loads(result.stdout)

# Step 3: Read results from Parquet files
output_dir = output["data"]["output_directory"]
convergence = pl.read_parquet(f"{output_dir}/training/convergence.parquet")
costs = pl.read_parquet(f"{output_dir}/simulation/costs.parquet")
```

This workflow provides full MPI parallelism without any GIL/MPI interaction, and leverages the structured output protocol ([Structured Output](./structured-output.md) SS1) for programmatic result parsing. It is the only option for multi-node execution with MPI and InfiniBand interconnects.

### 7.4 Multi-Process Architecture

Each worker process in the multi-process execution model (SS7.3) has the following properties:

1. **Independent Python interpreter and GIL.** Every worker spawned via `multiprocessing.Process` with `start_method="spawn"` runs its own Python interpreter with its own GIL. There is zero GIL contention between workers -- each worker releases its own GIL independently before entering Rust computation.

2. **Per-worker `Communicator` instance.** Each worker calls into the PyO3 layer, which creates a backend-specific `Communicator` (either `TcpBackend` or `ShmBackend`). The `Communicator` is owned by the Rust side within the worker process. No `Communicator` state crosses the Python/Rust boundary.

3. **Parent process is orchestrator only.** The parent Python process spawns workers, waits for them to complete via `p.join()`, and reads results from disk. It does not participate in the SDDP computation and does not hold a `Communicator` instance.

4. **Workers communicate via the selected backend, not via Python IPC.** All inter-worker communication (cut sharing via `allgatherv`, bound synchronization via `allreduce`, barrier synchronization) occurs within the Rust layer through the backend's transport (TCP sockets or POSIX shared memory). Python's `multiprocessing.Queue`, `multiprocessing.Pipe`, and `multiprocessing.Value` are not used for SDDP data exchange.

5. **Independent Rayon thread pools.** Each worker process has its own Rayon runtime with its own thread pool. If the user sets `RAYON_NUM_THREADS=N` before spawning, each worker creates `N` Rayon threads. For optimal CPU utilization, the total thread count across all workers should not exceed the physical core count: `num_workers * RAYON_NUM_THREADS <= num_physical_cores`.

### 7.5 Backend Selection from Python

The `backend` parameter in multi-process mode controls which communication backend each worker creates. The following values are accepted:

| `backend` Value | Behavior                                                                                                                                                                                           |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"auto"`        | **Default.** When `num_workers > 1`: selects `"shm"` for single-node execution; selects `"tcp"` if the `COBRE_TCP_COORDINATOR` environment variable is set                                         |
| `"shm"`         | Forces the shared-memory backend. Requires all workers on the same physical node. Each worker opens the same POSIX shared memory segment. See [Shared Memory Backend](../hpc/backend-shm.md) SS7.1 |
| `"tcp"`         | Forces the TCP backend. Requires a coordinator address (passed as a parameter or via `COBRE_TCP_COORDINATOR`). See [TCP Backend](../hpc/backend-tcp.md) SS8.1                                      |
| `"local"`       | Forces the local backend. Single-process mode; the `num_workers` parameter is ignored. All computation runs in the calling process                                                                 |

**Auto-detection logic** (when `backend="auto"` and `num_workers > 1`):

1. If `COBRE_TCP_COORDINATOR` is set in the environment, select `"tcp"`.
2. Otherwise, select `"shm"` (assumes single-node execution).

This is the Python-side equivalent of the priority chain defined in [Backend Registration and Selection](../hpc/backend-selection.md) SS2.2, simplified for the Python context where MPI is never available. The full auto-detection algorithm (including MPI detection) is documented in that section.

### 7.5a Future: Free-Threaded Python (PEP 703)

CPython 3.14 (October 2025) introduced officially supported free-threaded builds ([PEP 779](https://peps.python.org/pep-0779/)) where the GIL can be disabled, allowing true multi-threaded parallelism within a single Python process. This section documents the impact on Cobre's Python bindings and the conditions under which the design may evolve.

**Current status (Phase II):** Free-threaded Python is available as an optional, separate build (`python3.14t`). It is not the default. Importing a C extension not marked as free-thread-safe automatically re-enables the GIL for the process lifetime. The GIL-disabled default (Phase III) is estimated for Python ~3.17-3.18 (2028-2030) and requires a separate PEP.

**Impact on the 6-point GIL contract (SS3.1):** All six contract points remain correct on both GIL-enabled and free-threaded builds. On free-threaded builds, `py.detach()` detaches the calling thread from the Python runtime instead of releasing the GIL. This detachment is still necessary: free-threaded CPython triggers stop-the-world synchronization during garbage collection and tracing, and a thread that stays attached while performing pure Rust computation would block these events. The same code works correctly on both build types without conditional compilation.

**Impact on MPI prohibition (SS7.2):** Free-threaded Python resolves the GIL/`MPI_THREAD_MULTIPLE` deadlock risk (SS7.2 point 2) because threads can truly execute concurrently. However, the remaining two prohibition reasons -- `MPI_Init_thread` timing conflict (point 1) and dual-FFI-layer fragility (point 3) -- are independent of the GIL and remain valid. The MPI prohibition therefore stands regardless of GIL state. mpi4py 4.1.1 ships free-threaded wheels (`cp314t`) and requests `MPI_THREAD_MULTIPLE` by default, but this benefits direct mpi4py usage, not Cobre's PyO3 layer which avoids MPI entirely.

**Impact on multi-process design (SS7.3-7.4):** The current `multiprocessing.Process`-based multi-worker design remains the recommended approach. A future alternative -- spawning worker threads instead of processes within a single free-threaded Python interpreter -- becomes architecturally viable when: (1) free-threading is the CPython default, (2) all Cobre dependencies in the Python wheel are free-thread-safe, and (3) PyO3's `#[pyclass]` types satisfy the `Sync` requirement (already the case for Cobre's types, which do not hold Python objects across the Rust boundary). This threading-based mode would eliminate `multiprocessing` serialization overhead and shared-memory segment management, but requires careful evaluation of Rayon thread pool interaction within a single-process multi-worker model. This extension is deferred pending ecosystem maturity.

**PyO3 requirements for free-threading:** Since PyO3 0.23, `#[pyclass]` types must implement `Sync`. Since PyO3 0.28, modules default to thread-safe (`Py_MOD_GIL_NOT_USED`). At implementation time, the `cobre-python` crate should: (1) audit all `#[pyclass]` types for `Sync` compliance, (2) avoid `GILProtected<T>` (removed in current PyO3), (3) use `pyo3::sync::critical_section` for any shared mutable state, and (4) set `gil_used = true` as a temporary escape hatch only if thread-safety audit is incomplete.

## 8. Memory Model

### 8.1 Ownership Rules at the Python/Rust Boundary

Data crossing the Python/Rust boundary follows strict ownership rules to prevent use-after-free and data races.

| Data Direction           | Ownership Transfer                                                                                       | Lifetime                                  |
| ------------------------ | -------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Python -> Rust (args)    | PyO3 borrows or copies Python objects. Borrows are valid only during the GIL-held validation phase       | Duration of the function call             |
| Rust -> Python (returns) | Rust data is either consumed (moved into `PyArray`) or converted (serialized to Python dict/list)        | Python garbage collector manages lifetime |
| Zero-copy NumPy          | Rust `Vec<f64>` is moved into `PyArray`. Rust no longer owns the data. Python GC frees when refcount = 0 | Python object lifetime                    |
| Zero-copy Arrow          | Arrow `RecordBatch` exported via C Data Interface. Ownership transferred to `pyarrow`                    | Python object lifetime                    |
| Wrapped Rust objects     | `#[pyclass]` structs are owned by Python. The Rust struct lives inside the Python object                 | Python object lifetime                    |

### 8.2 No Rust Lifetimes Cross the Boundary

All PyO3 classes use owned data, not references. This means:

- `Case` owns a clone of the resolved internal structures (or an `Arc<InternalStructures>`)
- `Policy` owns the loaded FlatBuffers data (or an `Arc<PolicyData>`)
- `TrainingResult` owns the convergence history and policy reference

No Python object holds a Rust `&'a T` reference. This eliminates lifetime-related safety issues at the FFI boundary.

### 8.3 Memory Categories in Single-Process Mode

The [Memory Architecture](../hpc/memory-architecture.md) SS1.1 ownership categories apply with the following modifications for single-process (no MPI) mode:

| Category (SS1.1)         | Single-Process Behavior                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------------- |
| **Shared read-only**     | Regular per-process allocation (no `SharedWindow`). The Python process owns all read-only data directly |
| **Thread-local mutable** | Unchanged. Rayon threads own their solver workspaces with first-touch NUMA placement                    |
| **Rank-local growing**   | Single-rank: the cut pool grows in the one process. No MPI synchronization needed                       |
| **Temporary**            | Unchanged. Pre-allocated in workspace, reused per LP solve                                              |

## 9. Async Support

> **Status: OPTIONAL.** This section documents the recommended async approach per architecture-021 SS6.1 Q-4. Async wrappers are optional and may be deferred to a later release. This design is flagged for user review.

### 9.1 Motivation

Long-running operations (`train()`, `simulate()`) block the Python event loop in async applications (web servers, Jupyter notebooks with async cells, agent frameworks). Async wrappers allow these operations to run without blocking.

### 9.2 Recommended Approach: `run_in_executor`

The async wrappers delegate to a thread pool executor, which calls the synchronous functions. Since the GIL is released during Rust computation (SS3, point 2), the executor thread does not block other Python coroutines.

```python
import asyncio

async def train_async(
    case: Case,
    config_overrides: dict | None = None,
    progress_callback: Callable[[ProgressEvent], None] | None = None,
) -> TrainingResult:
    """Async wrapper for train().

    Runs train() in a thread pool executor so that the calling
    coroutine yields control to the event loop during computation.

    OPTIONAL: This function may not be available in all builds.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,  # Default executor
        lambda: train(case, config_overrides, progress_callback),
    )

async def simulate_async(
    case: Case,
    policy: Policy,
    config_overrides: dict | None = None,
    progress_callback: Callable[[ProgressEvent], None] | None = None,
) -> SimulationResult:
    """Async wrapper for simulate().

    Runs simulate() in a thread pool executor so that the calling
    coroutine yields control to the event loop during computation.

    OPTIONAL: This function may not be available in all builds.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: simulate(case, policy, config_overrides, progress_callback),
    )
```

### 9.3 Why `run_in_executor` (Not Native Async)

A native async implementation would require the Rust training loop to yield cooperatively at iteration boundaries. This adds complexity to the core algorithm with no performance benefit -- the GIL is already released, so the Python event loop is free to schedule other coroutines. The `run_in_executor` approach achieves the same concurrency without modifying the Rust core.

### 9.4 Progress Callbacks in Async Context

When using `train_async()` with a progress callback, the callback runs on the executor thread (not the event loop thread). If the callback needs to update async state, it should use `loop.call_soon_threadsafe()`:

```python
async def monitor_training():
    loop = asyncio.get_running_loop()
    progress_queue = asyncio.Queue()

    def on_progress(event: ProgressEvent):
        loop.call_soon_threadsafe(progress_queue.put_nowait, event)

    result = await train_async(case, progress_callback=on_progress)
```

## 10. Build and Distribution

### 10.1 Build System

`cobre-python` uses [maturin](https://github.com/PyO3/maturin) for building Python wheels from the Rust `cdylib` crate.

```toml
# Cargo.toml excerpt
[lib]
name = "cobre"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.28.2", features = ["extension-module"] }
arrow = { version = "58", features = ["ffi"] }   # Arrow FFI for pyarrow
```

### 10.2 Platform Support

| Platform       | Support Level | Notes                                                 |
| -------------- | ------------- | ----------------------------------------------------- |
| Linux x86_64   | Primary       | Full support. Rayon threading (no external runtime)   |
| Linux aarch64  | Primary       | Full support. Rayon threading (no external runtime)   |
| macOS x86_64   | Secondary     | Full support. Rayon threading (no external runtime)   |
| macOS aarch64  | Secondary     | Apple Silicon. Rayon threading (no external runtime)  |
| Windows x86_64 | Optional      | Rayon threading works natively; no extra dependencies |

### 10.3 Wheel Contents

The wheel contains:

- The compiled `cobre` shared library (`.so` / `.dylib` / `.pyd`)
- The Rayon thread pool is built into the Rust binary (no external runtime library needed)
- Type stub file (`cobre.pyi`) for IDE autocompletion and `mypy` support
- No MPI libraries (ferrompi is not a dependency)
- No FlatBuffers Python package (policy access is via the Rust layer)

### 10.4 Python Version Support

Minimum Python version: 3.12 (matching the version requirement in SS2). Wheels are built for Python 3.12, 3.13, and 3.14. Free-threaded builds (`cp313t`, `cp314t`) are supported when the `cobre-python` crate passes the PyO3 free-threading audit (see SS7.5a); until then, importing `cobre` on a free-threaded interpreter will re-enable the GIL via the `gil_used = true` module flag.

## Cross-References

- [Structured Output](./structured-output.md) -- Error schema (SS2) and error kind registry (SS2.3) that define the exception mapping (SS6)
- [Convergence Monitoring](../architecture/convergence-monitoring.md) -- Per-iteration output record (SS2.4) that defines progress event fields and Arrow table columns (SS2.7, SS4.3)
- [Hybrid Parallelism](../hpc/hybrid-parallelism.md) -- Rayon threading model (SS5), initialization sequence (SS6), Rayon parallel iterator pattern (SS5.3) that guarantees GIL contract point 3
- [Memory Architecture](../hpc/memory-architecture.md) -- Data ownership categories (SS1.1) adapted for single-process mode (SS8.3); NUMA allocation principles (SS3) that apply to Rayon workspaces
- [Design Principles](../overview/design-principles.md) -- Format selection criteria (SS1), agent-readability rules (SS6.2)
- [Validation Architecture](../architecture/validation-architecture.md) -- 5-layer validation pipeline (SS2) invoked by `validate()` and `CaseLoader.load()`
- [Input System Entities](../data-model/input-system-entities.md) -- Entity field definitions for `Hydro`, `Thermal`, `Bus`, `Line` Python classes
- [Binary Formats](../data-model/binary-formats.md) -- FlatBuffers schemas for policy data accessed by the `Policy` class
- [Output Schemas](../data-model/output-schemas.md) -- Parquet output column definitions for simulation results read by Python directly
- [TCP Backend](../hpc/backend-tcp.md) -- TCP-based multi-process communication backend (SS8.1 for environment variables, SS8.2 for invocation examples) used by Python multi-process mode (SS7.3, SS7.5)
- [Shared Memory Backend](../hpc/backend-shm.md) -- POSIX shared-memory multi-process backend (SS7.3 for the Python multiprocessing code example) used by Python multi-process mode (SS7.3, SS7.5)
- [Backend Registration and Selection](../hpc/backend-selection.md) -- Auto-detection algorithm (SS2.2) and feature flag matrix (SS1.2) governing backend availability in Python wheel builds (SS7.5)
- [Communicator Trait](../hpc/communicator-trait.md) -- `Communicator` trait definition (SS1) that each worker's backend implements; `SharedMemoryProvider` (SS4) for shm backend shared regions
- [Backend Testing](../hpc/backend-testing.md) -- Conformance test suite (§1) and determinism verification (§4) that validate backend interchangeability for multi-process execution modes (SS7.3-7.5)
- [PEP 703](https://peps.python.org/pep-0703/) -- Making the Global Interpreter Lock Optional in CPython (referenced in SS7.5a)
- [PEP 779](https://peps.python.org/pep-0779/) -- Criteria for Supported Status for Free-Threaded CPython (referenced in SS7.5a)
- Architecture-021 SS2.2 -- `cobre-python` crate responsibility boundaries and API surface table
- Architecture-021 SS6.1 Q-4 -- Async support assumption (optional, `run_in_executor`)
- Architecture-021 SS6.3 -- 5 hard constraints from GIL/MPI analysis
