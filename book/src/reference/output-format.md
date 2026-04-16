# Output Format Reference

This page is the exhaustive schema reference for every file produced by
`cobre run`. It documents column names, Arrow data types, nullability, JSON
field structures, and binary format layouts for all 10 Parquet schemas, the
two manifest types, the training metadata file, the five dictionary files,
and the policy checkpoint format.

If you are new to Cobre output, start with
[Understanding Results](../tutorial/understanding-results.md) first. That page
explains what each file means conceptually and shows how to read results
programmatically. This page is for readers who need the precise schema
definition — for writing parsers, building dashboards, or implementing
compatibility checks.

---

## Output Directory Tree

A complete `cobre run` produces the following directory structure. Not every
entity directory appears in every run: `cobre run` only writes directories for
entity types present in the case. For example, a case with no pumping stations
will not produce `simulation/pumping_stations/`.

```
<output_dir>/
  training/
    _manifest.json
    metadata.json
    convergence.parquet
    dictionaries/
      codes.json
      entities.csv
      variables.csv
      bounds.parquet
      state_dictionary.json
    timing/
      iterations.parquet
      mpi_ranks.parquet
    solver/
      iterations.parquet
      retry_histogram.parquet
    scaling_report.json
    cut_selection/
      iterations.parquet         (when cut_selection is enabled)
  policy/
    cuts/
      stage_000.bin
      stage_001.bin
      ...
      stage_NNN.bin
    basis/
      stage_000.bin
      stage_001.bin
      ...
      stage_NNN.bin
    metadata.json
    states/                         # when exports.states = true
      stage_000.bin
      stage_001.bin
      ...
      stage_NNN.bin
  simulation/
    _manifest.json
    costs/
      scenario_id=0000/
        data.parquet
      scenario_id=0001/
        data.parquet
      ...
    hydros/
      scenario_id=0000/data.parquet
      ...
    thermals/
      scenario_id=0000/data.parquet
      ...
    exchanges/
      scenario_id=0000/data.parquet
      ...
    buses/
      scenario_id=0000/data.parquet
      ...
    pumping_stations/
      scenario_id=0000/data.parquet
      ...
    contracts/
      scenario_id=0000/data.parquet
      ...
    non_controllables/
      scenario_id=0000/data.parquet
      ...
    inflow_lags/
      scenario_id=0000/data.parquet
      ...
    violations/
      generic/
        scenario_id=0000/data.parquet
        ...
    solver/
      iterations.parquet
      retry_histogram.parquet
  hydro_models/
    fpha_hyperplanes.parquet         (when any hydro uses source: "computed")
  stochastic/
    inflow_seasonal_stats.parquet    (when estimation was performed)
    inflow_ar_coefficients.parquet   (when estimation was performed)
    correlation.json                 (always)
    fitting_report.json              (when estimation was performed)
    noise_openings.parquet           (always)
    load_seasonal_stats.parquet      (when load buses exist)
```

---

## Training Output

### `training/_manifest.json`

The training manifest is written atomically at the end of the training run (and
updated on each checkpoint if checkpointing is enabled). Consumers should read
`status` before interpreting any other field.

**JSON structure:**

```json
{
  "version": "2.0.0",
  "status": "complete",
  "started_at": "2026-01-17T08:00:00Z",
  "completed_at": "2026-01-17T12:30:00Z",
  "iterations": {
    "max_iterations": 200,
    "completed": 128,
    "converged_at": null
  },
  "convergence": {
    "achieved": false,
    "final_gap_percent": 0.45,
    "termination_reason": "iteration_limit"
  },
  "cuts": {
    "total_generated": 1250000,
    "total_active": 980000,
    "peak_active": 1100000
  },
  "checksum": null,
  "distribution": {
    "backend": "local",
    "world_size": 1,
    "ranks_participated": 1,
    "num_nodes": 1,
    "threads_per_rank": 4,
    "mpi_library": null,
    "mpi_standard": null,
    "thread_level": null,
    "slurm_job_id": null
  }
}
```

**Field reference:**

| Field                             | Type    | Nullable | Description                                                                                                                                              |
| --------------------------------- | ------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `version`                         | string  | No       | Manifest schema version. Current value: `"2.0.0"`.                                                                                                       |
| `status`                          | string  | No       | Run status: `"running"`, `"complete"`, `"failed"`, or `"converged"`.                                                                                     |
| `started_at`                      | string  | Yes      | ISO 8601 timestamp when training started. `null` in minimal viable version.                                                                              |
| `completed_at`                    | string  | Yes      | ISO 8601 timestamp when training finished. `null` while running.                                                                                         |
| `iterations.max_iterations`       | integer | Yes      | Maximum iterations allowed by the iteration-limit stopping rule. `null` if no limit was configured.                                                      |
| `iterations.completed`            | integer | No       | Number of training iterations that finished.                                                                                                             |
| `iterations.converged_at`         | integer | Yes      | Iteration number at which a convergence stopping rule triggered termination. `null` if training was terminated by a safety limit (e.g. iteration limit). |
| `convergence.achieved`            | boolean | No       | `true` if a convergence-oriented stopping rule terminated the run.                                                                                       |
| `convergence.final_gap_percent`   | number  | Yes      | Optimality gap between lower and upper bounds at termination, expressed as a percentage. `null` when upper bound evaluation is disabled.                 |
| `convergence.termination_reason`  | string  | No       | Machine-readable termination label. Common values: `"iteration_limit"`, `"bound_stalling"`.                                                              |
| `cuts.total_generated`            | integer | No       | Total Benders cuts generated across all stages and iterations.                                                                                           |
| `cuts.total_active`               | integer | No       | Cuts still active in the pool at termination.                                                                                                            |
| `cuts.peak_active`                | integer | No       | Maximum number of simultaneously active cuts at any point during training.                                                                               |
| `checksum`                        | object  | Yes      | Integrity checksum over policy and convergence files. `null` in current release (deferred).                                                              |
| `distribution.backend`            | string  | No       | Communication backend: `"mpi"` or `"local"`.                                                                                                             |
| `distribution.world_size`         | integer | No       | Total number of MPI ranks. `1` for single-process runs.                                                                                                  |
| `distribution.ranks_participated` | integer | No       | Number of MPI ranks that wrote data.                                                                                                                     |
| `distribution.num_nodes`          | integer | No       | Number of distinct physical hosts.                                                                                                                       |
| `distribution.threads_per_rank`   | integer | No       | Rayon worker threads per process.                                                                                                                        |
| `distribution.mpi_library`        | string  | Yes      | MPI library version (e.g. `"MPICH 4.2.3"`). `null` for local backend.                                                                                    |
| `distribution.mpi_standard`       | string  | Yes      | MPI standard version (e.g. `"MPI 4.0"`). `null` for local backend.                                                                                       |
| `distribution.thread_level`       | string  | Yes      | MPI thread safety level (e.g. `"Funneled"`). `null` for local backend.                                                                                   |
| `distribution.slurm_job_id`       | string  | Yes      | SLURM job ID when running under a SLURM scheduler. `null` otherwise.                                                                                     |

---

### `training/metadata.json`

The metadata file captures the configuration snapshot, problem dimensions,
performance summary, data integrity hashes, and runtime environment for
reproducibility and audit purposes. Fields marked "deferred" are `null` in the
current release and will be populated in a future minor version.

**Top-level structure:**

```json
{
  "version": "2.0.0",
  "run_info": { ... },
  "configuration_snapshot": { ... },
  "problem_dimensions": { ... },
  "performance_summary": null,
  "data_integrity": null,
  "environment": { ... }
}
```

**`run_info` fields:**

| Field              | Type   | Nullable | Description                                                                       |
| ------------------ | ------ | -------- | --------------------------------------------------------------------------------- |
| `run_id`           | string | No       | Unique run identifier. Placeholder value in current release.                      |
| `started_at`       | string | Yes      | ISO 8601 start timestamp.                                                         |
| `completed_at`     | string | Yes      | ISO 8601 completion timestamp.                                                    |
| `duration_seconds` | number | Yes      | Total run duration in seconds.                                                    |
| `cobre_version`    | string | No       | Version of the cobre binary that produced this output (from `CARGO_PKG_VERSION`). |
| `solver`           | string | Yes      | LP solver backend identifier (e.g. `"highs"`).                                    |
| `solver_version`   | string | Yes      | LP solver library version string.                                                 |
| `hostname`         | string | Yes      | Primary compute node hostname. `null` in current release.                         |
| `user`             | string | Yes      | Username that initiated the run. `null` in current release.                       |

**`configuration_snapshot` fields:**

| Field            | Type    | Nullable | Description                                                 |
| ---------------- | ------- | -------- | ----------------------------------------------------------- |
| `seed`           | integer | Yes      | Random seed used for scenario generation.                   |
| `forward_passes` | integer | Yes      | Number of forward-pass scenario trajectories per iteration. |
| `stopping_mode`  | string  | No       | How multiple stopping rules combine: `"any"` or `"all"`.    |
| `policy_mode`    | string  | No       | Policy warm-start mode: `"fresh"` or `"resume"`.            |

**`problem_dimensions` fields:**

| Field          | Type    | Nullable | Description                               |
| -------------- | ------- | -------- | ----------------------------------------- |
| `num_stages`   | integer | No       | Number of stages in the planning horizon. |
| `num_hydros`   | integer | No       | Total number of hydro plants.             |
| `num_thermals` | integer | No       | Total number of thermal plants.           |
| `num_buses`    | integer | No       | Total number of buses.                    |
| `num_lines`    | integer | No       | Total number of transmission lines.       |

**`performance_summary`:** Deferred. Always `null` in the current release. Will
contain `total_lp_solves`, `avg_lp_time_us`, `median_lp_time_us`,
`p99_lp_time_us`, and `peak_memory_mb` when implemented.

**`data_integrity`:** Deferred. Always `null` in the current release. Will
contain SHA-256 hashes of input files, config, policy, and convergence data
when implemented.

**`environment` fields:**

The `environment` object contains a `distribution` sub-object with the same
fields as the manifest's `distribution` (see above). Additionally:

| Field                           | Type    | Nullable | Description                                        |
| ------------------------------- | ------- | -------- | -------------------------------------------------- |
| `distribution.backend`          | string  | No       | Communication backend: `"mpi"` or `"local"`.       |
| `distribution.world_size`       | integer | No       | Total number of MPI ranks.                         |
| `distribution.num_nodes`        | integer | No       | Number of distinct physical hosts.                 |
| `distribution.threads_per_rank` | integer | No       | Rayon worker threads per process.                  |
| `distribution.mpi_library`      | string  | Yes      | MPI library version. `null` for local backend.     |
| `distribution.mpi_standard`     | string  | Yes      | MPI standard version. `null` for local backend.    |
| `distribution.thread_level`     | string  | Yes      | MPI thread safety level. `null` for local backend. |
| `distribution.slurm_job_id`     | string  | Yes      | SLURM job ID. `null` when not running under SLURM. |

---

### `training/convergence.parquet`

Per-iteration convergence log. One row per training iteration. 13 columns.

| Column             | Type    | Nullable | Description                                                                                                   |
| ------------------ | ------- | -------- | ------------------------------------------------------------------------------------------------------------- |
| `iteration`        | Int32   | No       | Training iteration number (1-based).                                                                          |
| `lower_bound`      | Float64 | No       | Best proven lower bound on the minimum expected cost after this iteration.                                    |
| `upper_bound_mean` | Float64 | No       | Mean upper bound estimate from the forward-pass scenarios in this iteration.                                  |
| `upper_bound_std`  | Float64 | No       | Standard deviation of the upper bound estimate across forward-pass scenarios.                                 |
| `gap_percent`      | Float64 | Yes      | Relative gap between lower and upper bounds as a percentage. `null` when the lower bound is zero or negative. |
| `cuts_added`       | Int32   | No       | Number of new cuts added to the pool during this iteration's backward pass.                                   |
| `cuts_removed`     | Int32   | No       | Number of cuts deactivated by the cut selection strategy in this iteration.                                   |
| `cuts_active`      | Int64   | No       | Total number of active cuts across all stages at the end of this iteration.                                   |
| `time_forward_ms`  | Int64   | No       | Wall-clock time spent in the forward pass, in milliseconds.                                                   |
| `time_backward_ms` | Int64   | No       | Wall-clock time spent in the backward pass, in milliseconds.                                                  |
| `time_total_ms`    | Int64   | No       | Total wall-clock time for this iteration, in milliseconds.                                                    |
| `forward_passes`   | Int32   | No       | Number of forward-pass scenario trajectories evaluated in this iteration.                                     |
| `lp_solves`        | Int64   | No       | Total number of LP solves across all stages and forward passes in this iteration.                             |

---

### `training/timing/iterations.parquet`

Per-iteration wall-clock timing breakdown by phase. One row per training
iteration. 16 columns. All columns are non-nullable.

The top-level non-overlapping phases are: `forward_wall_ms`,
`backward_wall_ms`, `cut_selection_ms`, `mpi_allreduce_ms`, and
`lower_bound_ms`. The backward parallel overhead is decomposed into three
components: `bwd_setup_ms` (aggregate non-solve work summed across
workers), `bwd_load_imbalance_ms` (max-worker minus average-worker),
and `bwd_scheduling_overhead_ms` (parallel wall minus max-worker). The
forward pass carries the same three sub-components with `fwd_` prefix.
The backward phase also has the sub-components `cut_sync_ms`,
`state_exchange_ms`, and `cut_batch_build_ms`. The residual not
attributed to any phase is `overhead_ms`.

| Column                       | Type  | Nullable | Description                                                                                                                                                                                 |
| ---------------------------- | ----- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `iteration`                  | Int32 | No       | Training iteration number (1-based).                                                                                                                                                        |
| `forward_wall_ms`            | Int64 | No       | Wall-clock time for the forward pass (all stages and scenarios).                                                                                                                            |
| `backward_wall_ms`           | Int64 | No       | Wall-clock time for the backward pass (all stages and trial points).                                                                                                                        |
| `cut_selection_ms`           | Int64 | No       | Time spent running the cut selection pipeline (all three stages).                                                                                                                           |
| `mpi_allreduce_ms`           | Int64 | No       | Time spent in MPI allreduce (forward-pass bound synchronization).                                                                                                                           |
| `cut_sync_ms`                | Int64 | No       | Time spent in per-stage cut sync allgatherv (sub-component of backward).                                                                                                                    |
| `lower_bound_ms`             | Int64 | No       | Time spent evaluating the lower bound (stage-0 LP solves for all openings).                                                                                                                 |
| `state_exchange_ms`          | Int64 | No       | Time spent in state exchange allgatherv (sub-component of backward).                                                                                                                        |
| `cut_batch_build_ms`         | Int64 | No       | Time spent assembling cut row batches (sub-component of backward).                                                                                                                          |
| `bwd_setup_ms`               | Int64 | No       | Aggregate non-solve work (load_model + add_rows + set_bounds + basis_set) summed across backward workers, in ms. May exceed `backward_wall_ms`; it is a cost metric, not a wall-time slice. |
| `bwd_load_imbalance_ms`      | Int64 | No       | Backward load imbalance: `max_worker_total - avg_worker_total`, clamped to zero.                                                                                                            |
| `bwd_scheduling_overhead_ms` | Int64 | No       | Backward scheduling overhead: `parallel_wall - max_worker_total`, clamped to zero.                                                                                                          |
| `fwd_setup_ms`               | Int64 | No       | Aggregate non-solve work summed across forward workers, in ms. Same aggregate semantics as `bwd_setup_ms`.                                                                                  |
| `fwd_load_imbalance_ms`      | Int64 | No       | Forward load imbalance: `max_worker_total - avg_worker_total`, clamped to zero.                                                                                                             |
| `fwd_scheduling_overhead_ms` | Int64 | No       | Forward scheduling overhead: `parallel_wall - max_worker_total`, clamped to zero.                                                                                                           |
| `overhead_ms`                | Int64 | No       | Residual wall-clock time not attributed to any of the above phases.                                                                                                                         |

> **Schema migration note (v0.4.x):** The single columns `bwd_rayon_overhead_ms` and `fwd_rayon_overhead_ms` from earlier releases were replaced with three columns each (`_setup_ms`, `_load_imbalance_ms`, `_scheduling_overhead_ms`). Downstream scripts that read the parquet by column name must be updated. The invariant `load_imbalance + scheduling <= parallel_wall` holds; `setup_ms` is a separate aggregate-across-workers cost and is not bounded by wall time.

### `training/timing/mpi_ranks.parquet`

Per-iteration, per-rank timing statistics for distributed runs. One row per
(iteration, rank) pair. 8 columns. All columns are non-nullable.

| Column                  | Type  | Nullable | Description                                                   |
| ----------------------- | ----- | -------- | ------------------------------------------------------------- |
| `iteration`             | Int32 | No       | Training iteration number (1-based).                          |
| `rank`                  | Int32 | No       | MPI rank index (0-based).                                     |
| `forward_time_ms`       | Int64 | No       | Wall-clock time this rank spent in the forward pass.          |
| `backward_time_ms`      | Int64 | No       | Wall-clock time this rank spent in the backward pass.         |
| `communication_time_ms` | Int64 | No       | Wall-clock time this rank spent in MPI communication.         |
| `idle_time_ms`          | Int64 | No       | Wall-clock time this rank was idle (waiting for other ranks). |
| `lp_solves`             | Int64 | No       | Number of LP solves performed by this rank in this iteration. |
| `scenarios_processed`   | Int32 | No       | Number of scenario trajectories processed by this rank.       |

---

### `training/solver/iterations.parquet`

Per-iteration, per-phase, per-stage LP solver statistics for diagnosing
conditioning issues and retry behavior. One row per `(iteration, phase,
stage)` triple. 16 columns. All columns are non-nullable.

The `phase` column is a string: `"forward"`, `"backward"`, or
`"lower_bound"`.

| Column               | Type    | Nullable | Description                                                                          |
| -------------------- | ------- | -------- | ------------------------------------------------------------------------------------ |
| `iteration`          | UInt32  | No       | Training iteration number (1-based).                                                 |
| `phase`              | Utf8    | No       | Algorithm phase: `"forward"`, `"backward"`, or `"lower_bound"`.                      |
| `stage`              | Int32   | No       | Stage index (0-based).                                                               |
| `lp_solves`          | UInt32  | No       | Number of LP solves in this `(iteration, phase, stage)` triple.                      |
| `lp_successes`       | UInt32  | No       | Number of solves that returned optimal.                                              |
| `lp_retries`         | UInt32  | No       | Number of solves that required at least one retry.                                   |
| `lp_failures`        | UInt32  | No       | Number of solves that failed after exhausting all retry levels.                      |
| `retry_attempts`     | UInt32  | No       | Total retry attempts across all LP solves.                                           |
| `basis_offered`      | UInt32  | No       | Number of `solve_with_basis` calls (warm-start attempts).                            |
| `basis_rejections`   | UInt32  | No       | Number of times a basis was rejected and the solver fell back to cold-start.         |
| `simplex_iterations` | UInt64  | No       | Total simplex iterations (or IPM iterations) across all solves.                      |
| `solve_time_ms`      | Float64 | No       | Cumulative LP solve wall-clock time in milliseconds.                                 |
| `load_model_time_ms` | Float64 | No       | Cumulative time spent in `load_model` calls, in milliseconds.                        |
| `add_rows_time_ms`   | Float64 | No       | Cumulative time spent in `add_rows` calls, in milliseconds.                          |
| `set_bounds_time_ms` | Float64 | No       | Cumulative time spent in `set_row_bounds` / `set_col_bounds` calls, in milliseconds. |
| `basis_set_time_ms`  | Float64 | No       | Cumulative time spent installing bases for warm-start, in milliseconds.              |

### `training/solver/retry_histogram.parquet`

Per-level retry success counts, normalized from the solver iterations
table. One row per `(iteration, phase, stage, retry_level)` tuple where
the count is positive (sparse encoding). 5 columns. All non-nullable.

| Column        | Type   | Nullable | Description                                                                                                      |
| ------------- | ------ | -------- | ---------------------------------------------------------------------------------------------------------------- |
| `iteration`   | UInt32 | No       | Training iteration number (1-based).                                                                             |
| `phase`       | Utf8   | No       | Algorithm phase: `"forward"`, `"backward"`, or `"lower_bound"`.                                                  |
| `stage`       | Int32  | No       | Stage index (0-based).                                                                                           |
| `retry_level` | UInt32 | No       | Retry escalation level (0--11). See [Solver Safeguards](../guide/performance-accelerators.md#solver-safeguards). |
| `count`       | UInt64 | No       | Number of LP solves recovered at this retry level.                                                               |

### `training/scaling_report.json`

LP prescaling diagnostics written once after stage template construction.
Documents the coefficient range before and after column/row scaling for
each stage. Useful for diagnosing numerical conditioning issues.

The JSON is an array of per-stage objects, each containing:

| Field                    | Type    | Description                                                   |
| ------------------------ | ------- | ------------------------------------------------------------- |
| `stage`                  | integer | Stage index (0-based).                                        |
| `before.coefficient_min` | number  | Smallest absolute non-zero matrix coefficient before scaling. |
| `before.coefficient_max` | number  | Largest absolute matrix coefficient before scaling.           |
| `before.rhs_min`         | number  | Smallest absolute non-zero RHS value before scaling.          |
| `before.rhs_max`         | number  | Largest absolute RHS value before scaling.                    |
| `after.coefficient_min`  | number  | Smallest absolute non-zero coefficient after scaling.         |
| `after.coefficient_max`  | number  | Largest absolute coefficient after scaling.                   |
| `after.rhs_min`          | number  | Smallest absolute non-zero RHS value after scaling.           |
| `after.rhs_max`          | number  | Largest absolute RHS value after scaling.                     |

### `training/cut_selection/iterations.parquet`

Per-stage cut selection statistics. One row per `(iteration, stage)` pair,
written only at iterations where selection ran. 10 columns.

| Column                 | Type    | Nullable | Description                                                                 |
| ---------------------- | ------- | -------- | --------------------------------------------------------------------------- |
| `iteration`            | Int32   | No       | Training iteration number (1-based).                                        |
| `stage`                | Int32   | No       | Stage index (0-based).                                                      |
| `cuts_populated`       | Int32   | No       | Total cut slots containing cuts (active + inactive).                        |
| `cuts_active_before`   | Int32   | No       | Active cuts before this iteration's selection pipeline.                     |
| `cuts_deactivated`     | Int32   | No       | Cuts deactivated by the strategy-based selection (Stage 1).                 |
| `cuts_active_after`    | Int32   | No       | Active cuts after Stage 1 selection.                                        |
| `selection_time_ms`    | Float64 | No       | Wall-clock time for the full selection pipeline.                            |
| `active_after_angular` | Int32   | Yes      | Active cuts after angular pruning (Stage 2). `null` when S2 is disabled.    |
| `budget_evicted`       | Int32   | Yes      | Cuts evicted by budget enforcement (Stage 3). `null` when S3 is disabled.   |
| `active_after_budget`  | Int32   | Yes      | Active cuts after budget enforcement (Stage 3). `null` when S3 is disabled. |

---

### `training/dictionaries/`

Five self-documenting files that allow output Parquet files to be interpreted
without reference to the original input case. All files are written atomically.

#### `codes.json`

Static mapping from integer codes to human-readable labels for all categorical
fields used in Parquet output. The same mapping applies for the lifetime of a
release (the version field tracks breaking changes).

```json
{
  "version": "1.0",
  "generated_at": "2026-01-17T08:00:00Z",
  "operative_state": {
    "0": "deactivated",
    "1": "maintenance",
    "2": "operating",
    "3": "saturated"
  },
  "storage_binding": {
    "0": "none",
    "1": "below_minimum",
    "2": "above_maximum",
    "3": "both"
  },
  "contract_type": {
    "0": "import",
    "1": "export"
  },
  "entity_type": {
    "0": "hydro",
    "1": "thermal",
    "2": "bus",
    "3": "line",
    "4": "pumping_station",
    "5": "contract",
    "7": "non_controllable"
  },
  "bound_type": {
    "0": "storage_min",
    "1": "storage_max",
    "2": "turbined_min",
    "3": "turbined_max",
    "4": "outflow_min",
    "5": "outflow_max",
    "6": "generation_min",
    "7": "generation_max",
    "8": "flow_min",
    "9": "flow_max"
  }
}
```

#### `entities.csv`

One row per entity across all entity types. Columns:

| Column             | Description                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------ |
| `entity_type_code` | Integer entity type code (see `codes.json` `entity_type` mapping).                         |
| `entity_id`        | Integer entity ID matching the `*_id` column in the corresponding simulation Parquet file. |
| `name`             | Human-readable entity name from the case input files.                                      |
| `bus_id`           | Integer bus ID to which this entity is connected. For buses, equals `entity_id`.           |
| `system_id`        | System partition index. Always `0` in the current release (single-system cases).           |

Rows are ordered by `entity_type_code` ascending, then by `entity_id`
ascending within each type.

#### `variables.csv`

One row per output column across all Parquet schemas. Documents every column
name, its parent schema, and its unit of measure. Useful for building generic
result readers that do not hard-code column names.

| Column        | Description                                                                                                                               |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `schema`      | Name of the Parquet schema this column belongs to (e.g. `"hydros"`, `"costs"`).                                                           |
| `column_name` | Exact column name as it appears in the Parquet file.                                                                                      |
| `arrow_type`  | Arrow data type string (e.g. `"Int32"`, `"Float64"`, `"Boolean"`).                                                                        |
| `nullable`    | `"true"` or `"false"`.                                                                                                                    |
| `unit`        | Physical unit or `"code"` for categorical fields, `"boolean"` for flag fields, `"id"` for identifiers, `"dimensionless"` for pure ratios. |
| `description` | Short description of the column's meaning.                                                                                                |

#### `bounds.parquet`

Per-entity, per-stage resolved LP variable bounds. Documents the actual
numerical bounds used in each LP solve, after applying the three-tier penalty
resolution (global / entity / stage overrides).

| Column             | Type    | Nullable | Description                                              |
| ------------------ | ------- | -------- | -------------------------------------------------------- |
| `entity_type_code` | Int8    | No       | Entity type code (see `codes.json`).                     |
| `entity_id`        | Int32   | No       | Entity ID.                                               |
| `stage_id`         | Int32   | No       | Stage index (0-based).                                   |
| `bound_type_code`  | Int8    | No       | Bound type code (see `codes.json` `bound_type` mapping). |
| `lower_bound`      | Float64 | No       | Resolved lower bound value in the bound's natural unit.  |
| `upper_bound`      | Float64 | No       | Resolved upper bound value in the bound's natural unit.  |

#### `state_dictionary.json`

Describes the state space structure used by the algorithm: which entities have
state variables, how many state dimensions they contribute, and what units
apply. Useful for interpreting cut coefficient vectors in the policy checkpoint.

```json
{
  "version": "1.0",
  "state_dimension": 164,
  "storage_states": [
    { "hydro_id": 0, "dimension_index": 0, "unit": "hm3" },
    { "hydro_id": 1, "dimension_index": 1, "unit": "hm3" }
  ],
  "inflow_lag_states": [
    { "hydro_id": 0, "lag_index": 1, "dimension_index": 2, "unit": "m3s" }
  ]
}
```

| Field                                 | Description                                                                                                   |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `state_dimension`                     | Total number of state variables. Equals the length of each cut's coefficient vector in the policy checkpoint. |
| `storage_states`                      | One entry per hydro plant that contributes a reservoir storage state variable.                                |
| `storage_states[].hydro_id`           | Hydro plant ID.                                                                                               |
| `storage_states[].dimension_index`    | 0-based index of this state variable in the coefficient vector.                                               |
| `storage_states[].unit`               | Physical unit: always `"hm3"` (hectare-metres cubed).                                                         |
| `inflow_lag_states`                   | One entry per (hydro, lag) pair that contributes an inflow lag state variable.                                |
| `inflow_lag_states[].hydro_id`        | Hydro plant ID.                                                                                               |
| `inflow_lag_states[].lag_index`       | Autoregressive lag order (1-based).                                                                           |
| `inflow_lag_states[].dimension_index` | 0-based index in the coefficient vector.                                                                      |
| `inflow_lag_states[].unit`            | Physical unit: always `"m3s"` (cubic metres per second).                                                      |

---

## Policy Checkpoint

### `policy/cuts/stage_NNN.bin`

FlatBuffers binary file encoding all cuts for a single stage. One file per
stage; file names are zero-padded to three digits (e.g. `stage_000.bin`,
`stage_012.bin`).

The binary is not human-readable. The logical record structure for each cut
contained in the file is:

| Field                | Type      | Description                                                                                                                          |
| -------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `cut_id`             | uint64    | Unique identifier for this cut across all iterations. Assigned monotonically by the training loop.                                   |
| `slot_index`         | uint32    | LP row position. Required for checkpoint reproducibility and basis warm-starting.                                                    |
| `iteration`          | uint32    | Training iteration that generated this cut.                                                                                          |
| `forward_pass_index` | uint32    | Forward pass index within the generating iteration.                                                                                  |
| `intercept`          | float64   | Pre-computed cut intercept: `alpha - beta' * x_hat`, where `x_hat` is the state at the generating forward pass node.                 |
| `coefficients`       | float64[] | Gradient coefficient vector. Length equals `state_dimension` from `state_dictionary.json`.                                           |
| `is_active`          | bool      | Whether this cut is currently active in the LP. Inactive cuts are retained for potential reactivation by the cut selection strategy. |
| `domination_count`   | uint32    | Cut selection bookkeeping counter. Number of times this cut has been dominated without being selected.                               |

The encoding uses the FlatBuffers runtime builder API (little-endian, no
reflection, no generated code). Field order in the binary matches the
declaration order above.

### `policy/basis/stage_NNN.bin`

FlatBuffers binary file encoding the LP simplex basis checkpoint for a single
stage. One file per stage. Used to warm-start LP solves when resuming a study.

The logical record structure is:

| Field           | Type    | Description                                                                                                 |
| --------------- | ------- | ----------------------------------------------------------------------------------------------------------- |
| `stage_id`      | uint32  | Stage index (0-based).                                                                                      |
| `iteration`     | uint32  | Training iteration that produced this basis.                                                                |
| `column_status` | uint8[] | One status code per LP column (variable). Encoding is HiGHS-specific.                                       |
| `row_status`    | uint8[] | One status code per LP row (constraint). Encoding is HiGHS-specific.                                        |
| `num_cut_rows`  | uint32  | Number of trailing rows in `row_status` that correspond to cut rows (as opposed to structural constraints). |

### `policy/states/stage_NNN.bin`

FlatBuffers binary file encoding the visited forward-pass trial points for a
single stage. One file per stage. Present only when `exports.states` is `true`
(default is `false`). The `states/` directory is omitted entirely when disabled.

Trial points are the state vectors observed at each forward-pass scenario
during training. They are always collected in memory regardless of the cut
selection method, but persisted to disk only when this export flag is set.
Dominated cut selection uses these states at pruning time; for other methods
they serve as a diagnostic and analysis artifact.

| Field             | Type      | Description                                                                      |
| ----------------- | --------- | -------------------------------------------------------------------------------- |
| `stage_id`        | uint32    | Stage index (0-based).                                                           |
| `state_dimension` | uint32    | Length of each state vector. Must match `state_dictionary.json`.                 |
| `count`           | uint32    | Number of state vectors stored for this stage.                                   |
| `data`            | float64[] | Flat array of `count * state_dimension` elements, row-major (one state per row). |

### `policy/metadata.json`

Small JSON file describing the checkpoint at a high level. Human-readable and
intended for compatibility checking on study resume.

| Field                  | Type    | Nullable | Description                                                                                 |
| ---------------------- | ------- | -------- | ------------------------------------------------------------------------------------------- |
| `version`              | string  | No       | Checkpoint schema version.                                                                  |
| `cobre_version`        | string  | No       | Version of the cobre binary that wrote this checkpoint.                                     |
| `created_at`           | string  | No       | ISO 8601 timestamp when the checkpoint was written.                                         |
| `completed_iterations` | integer | No       | Number of training iterations completed at checkpoint time.                                 |
| `final_lower_bound`    | number  | No       | Lower bound value after the final completed iteration.                                      |
| `best_upper_bound`     | number  | Yes      | Best upper bound observed during training. `null` when upper bound evaluation was disabled. |
| `state_dimension`      | integer | No       | Length of each cut's coefficient vector. Must match `state_dictionary.json`.                |
| `num_stages`           | integer | No       | Number of stages. Must match the case configuration on resume.                              |
| `config_hash`          | string  | No       | Hash of the algorithm configuration. Checked against the current config on resume.          |
| `system_hash`          | string  | No       | Hash of the system data. Checked against the current system on resume.                      |
| `max_iterations`       | integer | No       | Maximum iterations configured for the run.                                                  |
| `forward_passes`       | integer | No       | Number of forward passes per iteration configured for the run.                              |
| `warm_start_cuts`      | integer | No       | Number of cuts loaded from a previous policy at run start. `0` for fresh runs.              |
| `rng_seed`             | integer | No       | RNG seed used by the scenario sampler. Required for reproducibility.                        |
| `total_visited_states` | integer | No       | Total number of visited state vectors across all stages. `0` when `exports.states` is off.  |

---

## Simulation Output

All simulation results use Hive partitioning: one `data.parquet` file per
scenario stored in a `scenario_id=NNNN/` subdirectory. See
[Hive Partitioning](#hive-partitioning) below for how to read these files.

### `simulation/costs/`

Stage and block-level cost breakdown. One row per (stage, block) pair. 26 columns.

| Column                         | Type    | Nullable | Description                                                                    |
| ------------------------------ | ------- | -------- | ------------------------------------------------------------------------------ |
| `stage_id`                     | Int32   | No       | Stage index (0-based).                                                         |
| `block_id`                     | Int32   | Yes      | Load block index within the stage. `null` for stage-level (non-block) records. |
| `total_cost`                   | Float64 | No       | Total discounted cost for this stage/block (monetary units).                   |
| `immediate_cost`               | Float64 | No       | Immediate (undiscounted) cost for this stage/block.                            |
| `future_cost`                  | Float64 | No       | Future cost estimate (Benders cut value) at the end of this stage.             |
| `discount_factor`              | Float64 | No       | Discount factor applied to this stage's costs.                                 |
| `thermal_cost`                 | Float64 | No       | Thermal generation cost component.                                             |
| `contract_cost`                | Float64 | No       | Energy contract cost component (positive for imports, negative for exports).   |
| `deficit_cost`                 | Float64 | No       | Cost of unserved load (deficit penalty).                                       |
| `excess_cost`                  | Float64 | No       | Cost of excess generation (excess penalty).                                    |
| `storage_violation_cost`       | Float64 | No       | Cost of reservoir storage bound violations.                                    |
| `filling_target_cost`          | Float64 | No       | Cost of missing reservoir filling targets.                                     |
| `hydro_violation_cost`         | Float64 | No       | Cost of hydro operational bound violations.                                    |
| `outflow_violation_below_cost` | Float64 | No       | Cost of total outflow below-minimum violations.                                |
| `outflow_violation_above_cost` | Float64 | No       | Cost of total outflow above-maximum violations.                                |
| `turbined_violation_cost`      | Float64 | No       | Cost of turbined flow bound violations.                                        |
| `generation_violation_cost`    | Float64 | No       | Cost of generation bound violations.                                           |
| `evaporation_violation_cost`   | Float64 | No       | Cost of evaporation violations.                                                |
| `withdrawal_violation_cost`    | Float64 | No       | Cost of water withdrawal violations.                                           |
| `inflow_penalty_cost`          | Float64 | No       | Cost of inflow non-negativity slack (numerical penalty).                       |
| `generic_violation_cost`       | Float64 | No       | Cost of generic constraint violations.                                         |
| `spillage_cost`                | Float64 | No       | Cost of reservoir spillage.                                                    |
| `fpha_turbined_cost`           | Float64 | No       | Turbined flow penalty from the future-production hydro approximation.          |
| `curtailment_cost`             | Float64 | No       | Cost of non-controllable source curtailment.                                   |
| `exchange_cost`                | Float64 | No       | Transmission exchange cost component.                                          |
| `pumping_cost`                 | Float64 | No       | Pumping station energy cost component.                                         |

---

### `simulation/hydros/`

Hydro plant dispatch results. One row per (stage, block, hydro) triplet. 31 columns.

| Column                               | Type    | Nullable | Description                                                                                                              |
| ------------------------------------ | ------- | -------- | ------------------------------------------------------------------------------------------------------------------------ |
| `stage_id`                           | Int32   | No       | Stage index (0-based).                                                                                                   |
| `block_id`                           | Int32   | Yes      | Load block index. `null` for stage-level records.                                                                        |
| `hydro_id`                           | Int32   | No       | Hydro plant ID.                                                                                                          |
| `turbined_m3s`                       | Float64 | No       | Turbined flow in cubic metres per second (m³/s).                                                                         |
| `spillage_m3s`                       | Float64 | No       | Spilled flow in m³/s.                                                                                                    |
| `outflow_m3s`                        | Float64 | No       | Total outflow (turbined + spilled) in m³/s.                                                                              |
| `evaporation_m3s`                    | Float64 | Yes      | Evaporation loss in m³/s. `null` if evaporation is not modelled for this plant.                                          |
| `diverted_inflow_m3s`                | Float64 | Yes      | Diverted inflow to this reservoir in m³/s. `null` if no diversion is configured.                                         |
| `diverted_outflow_m3s`               | Float64 | Yes      | Diverted outflow from this reservoir in m³/s. `null` if no diversion is configured.                                      |
| `incremental_inflow_m3s`             | Float64 | No       | Natural incremental inflow to this reservoir in m³/s (excluding upstream contributions).                                 |
| `inflow_m3s`                         | Float64 | No       | Total inflow to this reservoir in m³/s (including upstream contributions).                                               |
| `storage_initial_hm3`                | Float64 | No       | Reservoir storage at the start of the stage in hectare-metres cubed (hm³).                                               |
| `storage_final_hm3`                  | Float64 | No       | Reservoir storage at the end of the stage in hm³.                                                                        |
| `generation_mw`                      | Float64 | No       | Average power generation over the block in megawatts (MW).                                                               |
| `generation_mwh`                     | Float64 | No       | Total energy generated over the block in megawatt-hours (MWh).                                                           |
| `productivity_mw_per_m3s`            | Float64 | Yes      | Effective productivity factor in MW/(m³/s). `null` for fixed-productivity plants when productivity is not stage-varying. |
| `spillage_cost`                      | Float64 | No       | Monetary cost attributed to spillage.                                                                                    |
| `water_value_per_hm3`                | Float64 | No       | Shadow price of the reservoir water balance constraint (monetary units per hm³).                                         |
| `storage_binding_code`               | Int8    | No       | Whether the storage bounds were binding (see `codes.json` `storage_binding` mapping).                                    |
| `operative_state_code`               | Int8    | No       | Operative state code (see `codes.json` `operative_state` mapping).                                                       |
| `turbined_slack_m3s`                 | Float64 | No       | Turbined flow slack variable (non-negativity enforcement). Zero under normal operation.                                  |
| `outflow_slack_below_m3s`            | Float64 | No       | Outflow lower-bound slack in m³/s.                                                                                       |
| `outflow_slack_above_m3s`            | Float64 | No       | Outflow upper-bound slack in m³/s.                                                                                       |
| `generation_slack_mw`                | Float64 | No       | Generation bound slack in MW.                                                                                            |
| `storage_violation_below_hm3`        | Float64 | No       | Reservoir storage below-minimum violation in hm³. Zero under feasible operation.                                         |
| `filling_target_violation_hm3`       | Float64 | No       | Filling target miss in hm³. Zero when the target is met.                                                                 |
| `evaporation_violation_pos_m3s`      | Float64 | No       | Evaporation over-target violation in m³/s (actual > modelled). Zero under normal operation.                              |
| `evaporation_violation_neg_m3s`      | Float64 | No       | Evaporation under-target violation in m³/s (actual < modelled). Zero under normal operation.                             |
| `inflow_nonnegativity_slack_m3s`     | Float64 | No       | Inflow non-negativity slack in m³/s. Zero under normal operation.                                                        |
| `water_withdrawal_violation_pos_m3s` | Float64 | No       | Water withdrawal over-target violation in m³/s. Zero when withdrawal is at or below target.                              |
| `water_withdrawal_violation_neg_m3s` | Float64 | No       | Water withdrawal under-target violation in m³/s. Zero when withdrawal is at or above target.                             |

---

### `simulation/thermals/`

Thermal unit dispatch results. One row per (stage, block, thermal) triplet. 10 columns.

| Column                 | Type    | Nullable | Description                                                                   |
| ---------------------- | ------- | -------- | ----------------------------------------------------------------------------- |
| `stage_id`             | Int32   | No       | Stage index (0-based).                                                        |
| `block_id`             | Int32   | Yes      | Load block index. `null` for stage-level records.                             |
| `thermal_id`           | Int32   | No       | Thermal unit ID.                                                              |
| `generation_mw`        | Float64 | No       | Average power generation over the block in MW.                                |
| `generation_mwh`       | Float64 | No       | Total energy generated over the block in MWh.                                 |
| `generation_cost`      | Float64 | No       | Monetary generation cost for this block.                                      |
| `is_gnl`               | Boolean | No       | `true` if this unit operates under GNL (gas natural liquefied) pricing rules. |
| `gnl_committed_mw`     | Float64 | Yes      | Committed capacity under GNL mode in MW. `null` for non-GNL units.            |
| `gnl_decision_mw`      | Float64 | Yes      | Dispatch decision under GNL mode in MW. `null` for non-GNL units.             |
| `operative_state_code` | Int8    | No       | Operative state code (see `codes.json` `operative_state` mapping).            |

---

### `simulation/exchanges/`

Transmission line flow results. One row per (stage, block, line) triplet. 11 columns.

| Column                 | Type    | Nullable | Description                                                        |
| ---------------------- | ------- | -------- | ------------------------------------------------------------------ |
| `stage_id`             | Int32   | No       | Stage index (0-based).                                             |
| `block_id`             | Int32   | Yes      | Load block index. `null` for stage-level records.                  |
| `line_id`              | Int32   | No       | Transmission line ID.                                              |
| `direct_flow_mw`       | Float64 | No       | Flow in the forward (direct) direction in MW.                      |
| `reverse_flow_mw`      | Float64 | No       | Flow in the reverse direction in MW.                               |
| `net_flow_mw`          | Float64 | No       | Net flow (direct minus reverse) in MW.                             |
| `net_flow_mwh`         | Float64 | No       | Net energy flow over the block in MWh.                             |
| `losses_mw`            | Float64 | No       | Transmission losses in MW.                                         |
| `losses_mwh`           | Float64 | No       | Transmission losses in MWh over the block.                         |
| `exchange_cost`        | Float64 | No       | Monetary cost attributed to this line's exchange.                  |
| `operative_state_code` | Int8    | No       | Operative state code (see `codes.json` `operative_state` mapping). |

---

### `simulation/buses/`

Bus load balance results. One row per (stage, block, bus) triplet. 10 columns.

| Column        | Type    | Nullable | Description                                                                                         |
| ------------- | ------- | -------- | --------------------------------------------------------------------------------------------------- |
| `stage_id`    | Int32   | No       | Stage index (0-based).                                                                              |
| `block_id`    | Int32   | Yes      | Load block index. `null` for stage-level records.                                                   |
| `bus_id`      | Int32   | No       | Bus ID.                                                                                             |
| `load_mw`     | Float64 | No       | Total load demand at this bus in MW.                                                                |
| `load_mwh`    | Float64 | No       | Total load energy demand over the block in MWh.                                                     |
| `deficit_mw`  | Float64 | No       | Unserved load (deficit) at this bus in MW. Zero under feasible dispatch.                            |
| `deficit_mwh` | Float64 | No       | Unserved load energy over the block in MWh.                                                         |
| `excess_mw`   | Float64 | No       | Excess generation at this bus in MW. Zero under feasible dispatch.                                  |
| `excess_mwh`  | Float64 | No       | Excess generation energy over the block in MWh.                                                     |
| `spot_price`  | Float64 | No       | Locational marginal price (shadow price of the power balance constraint) in monetary units per MWh. |

---

### `simulation/pumping_stations/`

Pumping station results. One row per (stage, block, pumping station) triplet. 9 columns.

| Column                   | Type    | Nullable | Description                                                        |
| ------------------------ | ------- | -------- | ------------------------------------------------------------------ |
| `stage_id`               | Int32   | No       | Stage index (0-based).                                             |
| `block_id`               | Int32   | Yes      | Load block index. `null` for stage-level records.                  |
| `pumping_station_id`     | Int32   | No       | Pumping station ID.                                                |
| `pumped_flow_m3s`        | Float64 | No       | Pumped flow rate in m³/s.                                          |
| `pumped_volume_hm3`      | Float64 | No       | Total pumped volume over the stage in hm³.                         |
| `power_consumption_mw`   | Float64 | No       | Power consumed by the pumping station in MW.                       |
| `energy_consumption_mwh` | Float64 | No       | Energy consumed over the block in MWh.                             |
| `pumping_cost`           | Float64 | No       | Monetary cost of pumping energy.                                   |
| `operative_state_code`   | Int8    | No       | Operative state code (see `codes.json` `operative_state` mapping). |

---

### `simulation/contracts/`

Energy contract results. One row per (stage, block, contract) triplet. 8 columns.

| Column                 | Type    | Nullable | Description                                                         |
| ---------------------- | ------- | -------- | ------------------------------------------------------------------- |
| `stage_id`             | Int32   | No       | Stage index (0-based).                                              |
| `block_id`             | Int32   | Yes      | Load block index. `null` for stage-level records.                   |
| `contract_id`          | Int32   | No       | Contract ID.                                                        |
| `power_mw`             | Float64 | No       | Contracted power in MW. Positive for imports, negative for exports. |
| `energy_mwh`           | Float64 | No       | Contracted energy over the block in MWh.                            |
| `price_per_mwh`        | Float64 | No       | Contract price in monetary units per MWh.                           |
| `total_cost`           | Float64 | No       | Total contract cost for this block. Positive for imports.           |
| `operative_state_code` | Int8    | No       | Operative state code (see `codes.json` `operative_state` mapping).  |

---

### `simulation/non_controllables/`

Non-controllable source results (wind, solar, run-of-river hydro without
storage, etc.). One row per (stage, block, non-controllable) triplet. 10 columns.

| Column                 | Type    | Nullable | Description                                                                   |
| ---------------------- | ------- | -------- | ----------------------------------------------------------------------------- |
| `stage_id`             | Int32   | No       | Stage index (0-based).                                                        |
| `block_id`             | Int32   | Yes      | Load block index. `null` for stage-level records.                             |
| `non_controllable_id`  | Int32   | No       | Non-controllable source ID.                                                   |
| `generation_mw`        | Float64 | No       | Actual generation dispatched in MW.                                           |
| `generation_mwh`       | Float64 | No       | Actual energy generated over the block in MWh.                                |
| `available_mw`         | Float64 | No       | Maximum available generation in MW (before curtailment).                      |
| `curtailment_mw`       | Float64 | No       | Generation curtailed in MW. Zero when all available generation is dispatched. |
| `curtailment_mwh`      | Float64 | No       | Curtailed energy over the block in MWh.                                       |
| `curtailment_cost`     | Float64 | No       | Monetary cost attributed to curtailment.                                      |
| `operative_state_code` | Int8    | No       | Operative state code (see `codes.json` `operative_state` mapping).            |

---

### `simulation/inflow_lags/`

Autoregressive inflow lag state variables. One row per (stage, hydro, lag)
triplet. No block dimension — inflow lags are stage-level state variables.
4 columns. All columns are non-nullable.

| Column       | Type    | Nullable | Description                                                               |
| ------------ | ------- | -------- | ------------------------------------------------------------------------- |
| `stage_id`   | Int32   | No       | Stage index (0-based).                                                    |
| `hydro_id`   | Int32   | No       | Hydro plant ID.                                                           |
| `lag_index`  | Int32   | No       | Autoregressive lag order (1-based). Lag 1 is the previous stage's inflow. |
| `inflow_m3s` | Float64 | No       | Inflow value for this lag in m³/s.                                        |

---

### `simulation/violations/generic/`

Generic user-defined constraint violations. One row per (stage, block,
constraint) triplet where a violation occurred. 5 columns.

| Column          | Type    | Nullable | Description                                                                    |
| --------------- | ------- | -------- | ------------------------------------------------------------------------------ |
| `stage_id`      | Int32   | No       | Stage index (0-based).                                                         |
| `block_id`      | Int32   | Yes      | Load block index. `null` for stage-level constraints.                          |
| `constraint_id` | Int32   | No       | Constraint ID as defined in the case input files.                              |
| `slack_value`   | Float64 | No       | Violation magnitude in the constraint's natural unit. Zero means no violation. |
| `slack_cost`    | Float64 | No       | Monetary cost attributed to this violation.                                    |

---

## Hive Partitioning

All simulation Parquet output uses Hive partitioning: results for each scenario
are stored in a directory named `scenario_id=NNNN/` containing a single
`data.parquet` file. The `scenario_id` column is encoded in the directory name,
not as a column inside the Parquet file.

All major columnar data tools understand this layout and can read an entire
`simulation/<entity>/` directory as a single table with an automatically
inferred `scenario_id` column:

```python
# Polars — reads all scenarios at once, infers scenario_id from directory names
import polars as pl
df = pl.read_parquet("results/simulation/costs/")
print(df.head())
```

```python
# Pandas with PyArrow backend
import pandas as pd
df = pd.read_parquet("results/simulation/costs/")
```

```sql
-- DuckDB — filter to a specific scenario at the storage layer
SELECT * FROM read_parquet('results/simulation/costs/**/*.parquet')
WHERE scenario_id = 0;
```

```r
# R with the arrow package
library(arrow)
ds <- open_dataset("results/simulation/costs/")
dplyr::collect(dplyr::filter(ds, scenario_id == 0))
```

Scenario IDs are zero-based integers. The total number of scenarios is
documented in `simulation/_manifest.json` under `scenarios.total`.

---

## Manifest Files

Both `training/_manifest.json` and `simulation/_manifest.json` follow the same
write protocol:

1. Serialize JSON to a temporary `.json.tmp` sibling file.
2. Atomically rename the `.tmp` file to the target path.

This ensures consumers never observe a partial manifest. If a manifest file
exists, it contains a complete JSON document. If a run is interrupted before
the final manifest write, the `.tmp` file may remain but the manifest itself
will reflect the last successful checkpoint, not a partial write.

The `status` field is always the first indicator to check:

| Status        | Meaning                                                                                          |
| ------------- | ------------------------------------------------------------------------------------------------ |
| `"running"`   | The run is in progress or was interrupted without writing a final status.                        |
| `"complete"`  | The run finished normally. All output files are present.                                         |
| `"converged"` | Training terminated because a convergence stopping rule was satisfied. (Training manifest only.) |
| `"failed"`    | The run encountered a terminal error. Output files up to the failure point are present.          |
| `"partial"`   | Not all scenarios completed. (Simulation manifest only.)                                         |

`cobre report` reads both manifests and `training/metadata.json` and prints
a combined JSON summary to stdout. Use it in CI pipelines or shell scripts
to inspect outcomes without parsing JSON directly:

```bash
# Extract the termination reason
cobre report results/ | jq '.training.convergence.termination_reason'

# Fail a CI job if the run did not complete
status=$(cobre report results/ | jq -r '.status')
[ "$status" = "complete" ] || exit 1
```

---

## Hydro Model Artifacts

When any hydro plant is configured with `fpha_config.source: "computed"` in
`system/hydro_production_models.json`, Cobre fits the FPHA hyperplanes from
`system/hydro_geometry.parquet` before training begins and writes the results to
the `hydro_models/` directory. The directory is not written when no hydro uses
`source: "computed"`.

### `hydro_models/fpha_hyperplanes.parquet`

Fitted FPHA hyperplane coefficients for all hydros that used `source: "computed"`
in the current run. The schema is identical to the input file
`system/fpha_hyperplanes.parquet`: 11 columns, all with the same names, types,
and nullability.

| Column            | Type   | Nullable | Description                                               |
| ----------------- | ------ | -------- | --------------------------------------------------------- |
| `hydro_id`        | INT32  | No       | Hydro plant ID                                            |
| `stage_id`        | INT32  | Yes      | Stage the plane applies to. `null` = valid for all stages |
| `plane_id`        | INT32  | No       | Plane index within this hydro (and stage)                 |
| `gamma_0`         | DOUBLE | No       | Intercept coefficient (MW), unscaled                      |
| `gamma_v`         | DOUBLE | No       | Volume coefficient (MW/hm³)                               |
| `gamma_q`         | DOUBLE | No       | Turbined flow coefficient (MW per m³/s)                   |
| `gamma_s`         | DOUBLE | No       | Spillage coefficient (MW per m³/s)                        |
| `kappa`           | DOUBLE | Yes      | Correction factor. Defaults to `1.0` when absent or null. |
| `valid_v_min_hm3` | DOUBLE | Yes      | Volume range minimum where this plane is valid (hm³)      |
| `valid_v_max_hm3` | DOUBLE | Yes      | Volume range maximum where this plane is valid (hm³)      |
| `valid_q_max_m3s` | DOUBLE | Yes      | Maximum turbined flow where this plane is valid (m³/s)    |

The file is written atomically (via a `.tmp` rename) and uses the same
`(hydro_id, stage_id, plane_id)`-sorted row order as the input schema. It can
be used directly as a future `source: "precomputed"` input by copying it to
`system/fpha_hyperplanes.parquet`.

See [Case Format Reference — `system/fpha_hyperplanes.parquet`](./case-format.md#systemfpha_hyperplanesparquet)
for the full column definitions and validity constraints.

---

## Stochastic Artifacts

When `exports.stochastic: true` is set in `config.json`, Cobre writes the
stochastic preprocessing artifacts to `output/stochastic/` before training
begins.

The directory is not written when the config field is not set. Export is off
by default.

### Exported files

| File path                                   | Export condition         | Schema source                                            |
| ------------------------------------------- | ------------------------ | -------------------------------------------------------- |
| `stochastic/inflow_seasonal_stats.parquet`  | Estimation was performed | Same as input `scenarios/inflow_seasonal_stats.parquet`  |
| `stochastic/inflow_ar_coefficients.parquet` | Estimation was performed | Same as input `scenarios/inflow_ar_coefficients.parquet` |
| `stochastic/correlation.json`               | Always                   | Same as input `scenarios/correlation.json`               |
| `stochastic/fitting_report.json`            | Estimation was performed | JSON diagnostic report (see below)                       |
| `stochastic/noise_openings.parquet`         | Always                   | Same schema as `scenarios/noise_openings.parquet`        |
| `stochastic/load_seasonal_stats.parquet`    | Load buses exist         | Same as input `scenarios/load_seasonal_stats.parquet`    |

"Estimation was performed" means the user did not supply the corresponding
scenario file directly; Cobre derived it from `inflow_history.parquet`.

### `stochastic/noise_openings.parquet`

The opening tree used during the training run, written in the same schema as
the input file `scenarios/noise_openings.parquet`. See the
[Case Format Reference](./case-format.md#scenariosnoise_openingsparquet) for
the 4-column schema (`stage_id`, `opening_index`, `entity_index`, `value`).

### `stochastic/fitting_report.json`

A JSON diagnostic report for the PAR model fitting. This file is written only
when Cobre performed estimation from `inflow_history.parquet`.

**Structure:**

```json
{
  "hydros": {
    "<hydro_id>": {
      "selected_order": 3,
      "aic_scores": [12.4, 11.1, 10.8, 11.3],
      "coefficients": [[0.42, -0.11, 0.07]]
    }
  }
}
```

| Field            | Type         | Description                                                                      |
| ---------------- | ------------ | -------------------------------------------------------------------------------- |
| `selected_order` | integer      | AIC-selected AR order for this hydro plant                                       |
| `aic_scores`     | number array | AIC score for each candidate order; `aic_scores[i]` is the score for order `i+1` |
| `coefficients`   | nested array | One row per season; each row contains the AR coefficients for that season        |

This file is diagnostic only. It is not consumed as input on subsequent runs.

### Round-trip workflow

Every exported Parquet and JSON file uses the exact same column names, types,
and layout as the corresponding input file. To replay a run with identical
stochastic context:

```bash
# Run with exports.stochastic: true in config.json
cobre run my_case

# Copy exported artifacts to scenarios/
cp -r my_case/output/stochastic/* my_case/scenarios/

# Re-run: the loader finds the files already present and skips estimation
cobre run my_case
```

The re-run produces bit-for-bit identical stochastic artifacts because the
round-trip eliminates the estimation step. The opening tree is loaded directly
from `scenarios/noise_openings.parquet` instead of being regenerated.

See [Exporting Stochastic Artifacts](../guide/running-studies.md#exporting-stochastic-artifacts)
in the Running Studies guide for the end-to-end workflow.
