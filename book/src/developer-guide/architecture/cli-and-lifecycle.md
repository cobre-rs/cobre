# CLI and Lifecycle

## Purpose

This spec defines the Cobre program entrypoint, command-line interface, exit codes, execution phase lifecycle, conditional execution modes, configuration resolution hierarchy, and job scheduler integration. It covers everything from process invocation through phase orchestration to shutdown.

## 1. Design Philosophy

Cobre adopts a **single-entrypoint design** optimized for HPC batch execution. The program is always invoked via MPI launchers (`mpiexec`, `mpirun`, or SLURM's `srun`) and all runtime behavior is controlled through configuration files rather than command-line arguments.

**Rationale**:

- HPC job scripts benefit from stable command-line interfaces
- Configuration files provide auditability and reproducibility
- Complex nested options are better expressed in JSON than CLI flags
- Reduces parsing complexity in the hot initialization path

### 1.1 Agent Composability Principle

Cobre serves two audiences with complementary interaction models:

- **HPC batch execution** -- The primary mode. MPI-launched, config-driven, human-readable output. Optimized for production-scale runs on cluster schedulers.
- **Agent-composable interfaces** -- Secondary modes (MCP server, Python bindings, TUI) that expose the same solver through programmatic APIs. These operate in single-process mode without MPI, producing structured output with stable schemas.

The agent composability principle states that every Cobre operation must be usable by a programmatic agent -- an AI coding assistant, a CI/CD pipeline, or a Python orchestration script -- without requiring human interpretation of output. This is achieved through structured JSON output (`--output-format json`), progress streaming (`--output-format json-lines`), and library-mode execution (no MPI, no signal handlers, no scheduler detection). See [Design Principles SS6](../overview/design-principles.md) for the full agent-readability design rules and [Structured Output](../interfaces/structured-output.md) for the JSON schema definitions.

## 2. Invocation Pattern

```bash
# Standard invocation
mpiexec -n 8 cobre /path/to/case_directory

# SLURM batch execution
srun cobre /path/to/case_directory

# Validation-only mode
mpiexec -n 1 cobre /path/to/case_directory --validate-only
```

### 2.1 Subcommand Invocation Patterns

Cobre supports subcommand-style invocations. The available subcommands are:

```bash
cobre init [OPTIONS] [DIRECTORY]           # Scaffold a new case directory
cobre run <CASE_DIR> [OPTIONS]             # Load, train, simulate, and write results
cobre validate <CASE_DIR>                  # Validate a case directory
cobre report <RESULTS_DIR>                 # Query results from a completed run
cobre summary <OUTPUT_DIR>                 # Display the post-run summary
cobre schema export [--output-dir <DIR>]   # Export JSON Schema files for all input types
cobre version                              # Print version and build information
```

For distributed execution, the `run` subcommand can be launched under MPI:

```bash
mpiexec -np 4 cobre run /path/to/case_directory
```

**MPI requirements by subcommand**:

| Subcommand | MPI Required                                           | Rationale                                               |
| ---------- | ------------------------------------------------------ | ------------------------------------------------------- |
| `init`     | No                                                     | Template scaffolding; no computation                    |
| `run`      | Yes (for distributed execution) or No (single-process) | Training/simulation can run with or without MPI         |
| `validate` | No                                                     | Validation is rank-0-only; single-process is sufficient |
| `report`   | No                                                     | Reads output files; no computation                      |
| `summary`  | No                                                     | Reads output files; no computation                      |
| `schema`   | No                                                     | Schema export; no computation                           |
| `version`  | No                                                     | Information only                                        |

## 3. Command-Line Interface

| Argument          | Required | Description                                                   |
| ----------------- | -------- | ------------------------------------------------------------- |
| `CASE_DIR`        | Yes      | Path to case directory containing `config.json`               |
| `--validate-only` | No       | Run Startup and Validation phases only, then exit (see SS5.3) |
| `--version`       | No       | Print version and exit                                        |
| `--help`          | No       | Print usage and exit                                          |

### 3.1 Global CLI Flags

The following flag applies to all subcommands:

| Flag             | Values                    | Default | Description                                                                                                                                                               |
| ---------------- | ------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--color <WHEN>` | `auto`, `always`, `never` | `auto`  | Control ANSI color output on stderr. `always` forces color on (useful under `mpiexec` which pipes stderr through a non-TTY). Also honoured via the `COBRE_COLOR` env var. |

Color resolution order (highest to lowest priority):

1. `--color <WHEN>` CLI flag
2. `COBRE_COLOR` environment variable (`always` or `never`; invalid values ignored)
3. `FORCE_COLOR=1` environment variable (forces color on; see <https://force-color.org>)
4. Console auto-detection (whether stderr is a TTY)

### 3.2 Subcommand Arguments

Each subcommand accepts specific positional and keyword arguments:

| Subcommand | Positional    | Additional Flags                             | Description                        |
| ---------- | ------------- | -------------------------------------------- | ---------------------------------- |
| `init`     | `[DIRECTORY]` | `--template <NAME>`, `--list`, `--force`     | Scaffold a new case directory      |
| `run`      | `CASE_DIR`    | `--output <DIR>`, `--threads <N>`, `--quiet` | Execute training and/or simulation |
| `validate` | `CASE_DIR`    | (none)                                       | Validate input files only          |
| `report`   | `RESULTS_DIR` | (none)                                       | Query output data as JSON          |
| `summary`  | `OUTPUT_DIR`  | (none)                                       | Display post-run summary           |
| `schema`   | (subcommands) | `export [--output-dir <DIR>]`                | Manage JSON Schema files           |
| `version`  | (none)        | (none)                                       | Print version information          |

**Design Decision**: All execution options (skip training, skip simulation, warm-start mode, etc.) are specified in `config.json`, not via CLI flags. This ensures:

1. Job scripts remain stable across configuration changes
2. Configuration is self-documenting and version-controlled
3. No ambiguity between CLI and config file settings

## 4. Exit Codes

| Code | Category   | Cause                                                                                                                                                      |
| ---- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0`  | Success    | The command completed without errors                                                                                                                       |
| `1`  | Validation | Case directory failed the validation pipeline -- schema errors, cross-reference errors, semantic constraint violations, or policy compatibility mismatches |
| `2`  | I/O        | File not found, permission denied, disk full, or write failure during loading or output                                                                    |
| `3`  | Solver     | LP infeasible subproblem or numerical solver failure during training or simulation                                                                         |
| `4`  | Internal   | Communication failure, unexpected channel closure, or other software/environment problem                                                                   |

Codes 1--2 indicate user-correctable input problems; codes 3--4 indicate case/environment problems. Error messages are printed to stderr with `error:` prefix and hint lines.

## 5. Execution Phases Overview

### 5.1 Phase Diagram

```mermaid
flowchart TB
    S["Startup<br/><i>MPI init, CLI parse · all ranks</i>"]
    V["Validation<br/><i>load, schema, refs · rank 0</i>"]
    I["Initialization<br/><i>broadcast, alloc, solver · all ranks</i>"]
    G["Scenario Gen<br/><i>PAR fit, opening tree · all ranks</i>"]
    M{"run mode?"}
    T["Training"]
    LP["Load policy"]
    SIM["Simulation"]
    F(["Finalize"])

    S --> V --> I --> G --> M
    M -->|Full Run / Training Only| T
    M -->|Simulation Only| LP
    M -->|Validation Only| F
    T -->|simulation.enabled| SIM
    T -->|training only| F
    LP --> SIM
    SIM --> F
```

### 5.2 Phase Responsibilities

| Phase          | MPI Ranks      | Key Operations                                  |
| -------------- | -------------- | ----------------------------------------------- |
| Startup        | All            | MPI init, scheduler detection, CLI parsing      |
| Validation     | Rank 0 only    | Load files, schema validation, cross-references |
| Initialization | All            | Broadcast, memory allocation, solver setup      |
| Scenario Gen   | All (parallel) | PAR fitting, noise sampling, correlation        |
| Training       | All (parallel) | SDDP iterations                                 |
| Simulation     | All (parallel) | Policy evaluation                               |
| Finalize       | All            | Output writing, cleanup                         |

### 5.2a Phase-Training Loop Alignment

This subsection documents the correspondence between CLI execution phases (SS5.2) and the spec sections that define each phase's operations. The purpose is to ensure that every phase boundary assumed by the [Training Loop](./training-loop.md) is explicitly sequenced in the lifecycle, and that implementers can trace each phase to its authoritative specification.

| Phase              | Operation                                              | Authoritative Spec                                                  | Ordering Constraint                                                           |
| ------------------ | ------------------------------------------------------ | ------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **Startup**        | MPI backend initialization (`create_communicator`)     | [Hybrid Parallelism SS6, Step 1](../hpc/hybrid-parallelism.md)      | Must be the first operation; precedes all file I/O and thread creation        |
| **Startup**        | Topology detection (rank, size, intra-node split)      | [Hybrid Parallelism SS6, Steps 2--3](../hpc/hybrid-parallelism.md)  | Requires MPI backend initialized                                              |
| **Startup**        | Scheduler detection (SLURM, PBS, local)                | [CLI and Lifecycle SS6.3](#63-scheduler-detection)                  | Reads environment variables; no MPI dependency                                |
| **Startup**        | CLI argument parsing and subcommand routing            | [CLI and Lifecycle SS3](#3-command-line-interface)                  | Determines execution mode before Validation                                   |
| **Validation**     | Rank-0 file loading and validation (`load_case`)       | [Input Loading Pipeline SS8.1](./input-loading-pipeline.md)         | Rank 0 only; produces the `System` struct                                     |
| **Initialization** | postcard broadcast of `System` to worker ranks         | [Input Loading Pipeline SS6](./input-loading-pipeline.md)           | Requires `System` from Validation; all ranks receive identical validated data |
| **Initialization** | OpenMP configuration and NUMA allocation policy        | [Hybrid Parallelism SS6, Steps 4--6](../hpc/hybrid-parallelism.md)  | Must precede workspace allocation (first-touch policy)                        |
| **Initialization** | Solver workspace allocation (thread-local, NUMA-aware) | [Solver Workspaces SS1.3](./solver-workspaces.md)                   | Each thread creates its own workspace on its NUMA node                        |
| **Initialization** | Stage LP template construction                         | [Solver Abstraction SS11.1](./solver-abstraction.md)                | Built from resolved `System`; shared read-only across threads                 |
| **Initialization** | Parallel policy loading (warm-start only)              | [Input Loading Pipeline SS7](./input-loading-pipeline.md)           | All ranks load in parallel after `System` broadcast                           |
| **Initialization** | FPHA hyperplane fitting (computed source only)         | [Input Loading Pipeline SS8](./input-loading-pipeline.md)           | Requires geometry and topology from `System`                                  |
| **Scenario Gen**   | PAR model preprocessing                                | [Scenario Generation SS1](./scenario-generation.md)                 | Transforms raw PAR parameters into contiguous cache-friendly layout           |
| **Scenario Gen**   | Opening tree generation (fixed before training)        | [Scenario Generation SS2.3](./scenario-generation.md)               | Generated once; remains fixed throughout training                             |
| **Scenario Gen**   | Spectral decomposition of correlation matrices         | [Scenario Generation SS2.1](./scenario-generation.md)               | Pre-decomposed during preprocessing; used at runtime                          |
| **Training**       | SDDP iteration loop (forward/backward/convergence)     | [Training Loop SS2.1](./training-loop.md)                           | Requires all preceding phases complete                                        |
| **Simulation**     | Policy evaluation on large scenario sets               | [Simulation Architecture SS1](./simulation-architecture.md)         | Requires trained FCF from Training (or loaded policy from Initialization)     |
| **Finalize**       | Output writing (Parquet, policy FlatBuffers, manifest) | [Output Infrastructure SS1](../data-model/output-infrastructure.md) | Rank 0 writes manifest; all ranks may write partitioned output                |
| **Finalize**       | MPI finalize and process exit                          | [Hybrid Parallelism SS6](../hpc/hybrid-parallelism.md)              | Must be the last MPI operation                                                |

**Key invariants enforced by phase ordering:**

1. **MPI-first**: MPI initialization is the first operation in Startup, before any file I/O or thread creation. This is required by the MPI standard when using `MPI_THREAD_MULTIPLE` ([Hybrid Parallelism SS6, Step 1](../hpc/hybrid-parallelism.md)).
2. **Rank-0 validation before broadcast**: The `load_case` function ([Input Loading Pipeline SS8.1](./input-loading-pipeline.md)) executes on rank 0 only during the Validation phase. The resulting `System` struct is broadcast via postcard during Initialization. This ensures all ranks receive identical, validated data.
3. **Workspaces before training**: Solver workspace allocation ([Solver Workspaces SS1.3](./solver-workspaces.md)) and stage template construction ([Solver Abstraction SS11.1](./solver-abstraction.md)) complete during Initialization, before the Training phase begins. The Training Loop assumes these are ready at entry.
4. **Scenarios before training**: PAR preprocessing and opening tree generation complete during Scenario Gen. The backward pass requires the fixed opening tree ([Scenario Generation SS2.3](./scenario-generation.md)) from iteration 1 onward.

### 5.3 Conditional Execution

The execution flow supports several modes. Which phases execute depends on the mode:

| Phase          | Full Run | Training Only | Simulation Only | Validation Only |
| -------------- | :------: | :-----------: | :-------------: | :-------------: |
| Startup        |   Yes    |      Yes      |       Yes       |       Yes       |
| Validation     |   Yes    |      Yes      |       Yes       |       Yes       |
| Initialization |   Yes    |      Yes      |       Yes       |        —        |
| Scenario Gen   |   Yes    |      Yes      |       Yes       |        —        |
| Training       |   Yes    |      Yes      |        —        |        —        |
| Simulation     |   Yes    |       —       |       Yes       |        —        |
| Finalize       |   Yes    |      Yes      |       Yes       |        —        |

**Mode selection:**

- **Full Run** — Default. Both training and simulation execute sequentially.
- **Training Only** — Produces a policy (cuts) without evaluating it. Useful for convergence analysis or when simulation will be run separately.
- **Simulation Only** — Evaluates an existing policy. Requires a `policy/` directory from a prior training run. Scenario generation still executes because simulation forward passes need scenario realizations.
- **Validation Only** — Validates all input files and configuration, then exits immediately after the Validation phase. No memory allocation, no solver setup, no outputs. Triggered by `--validate-only` on the command line (overrides config settings) or by disabling both `training.enabled` and `simulation.enabled` in `config.json`.

Training Only and Simulation Only are controlled by the `training.enabled` and `simulation.enabled` fields in `config.json`. See [Configuration Reference](../configuration/configuration-reference.md).

### 5.4 Subcommand Phase Mapping

Each subcommand participates in a subset of the execution phases:

| Subcommand | Startup | Validation | Initialization | Scenario Gen | Training | Simulation | Finalize |
| ---------- | :-----: | :--------: | :------------: | :----------: | :------: | :--------: | :------: |
| `init`     |   --    |     --     |       --       |      --      |    --    |     --     |    --    |
| `run`      |   Yes   |    Yes     |      Yes       |     Yes      |   Yes    |    Yes     |   Yes    |
| `validate` |  Yes\*  |    Yes     |       --       |      --      |    --    |     --     |    --    |
| `report`   |   --    |     --     |       --       |      --      |    --    |     --     |    --    |
| `summary`  |   --    |     --     |       --       |      --      |    --    |     --     |    --    |
| `schema`   |   --    |     --     |       --       |      --      |    --    |     --     |    --    |
| `version`  |   --    |     --     |       --       |      --      |    --    |     --     |    --    |

\* Startup for `validate` skips MPI initialization (single-process mode). The `init`, `report`, `summary`, `schema`, and `version` subcommands have no lifecycle phases -- they perform their operation and exit immediately.

**Library mode** (used by `cobre-mcp` and `cobre-python`): When invoked as a library rather than via the CLI binary, the execution lifecycle skips MPI initialization, scheduler detection, and signal handler installation. The library caller provides the case path directly and receives structured results as Rust types. See [Hybrid Parallelism §1](../hpc/hybrid-parallelism.md) for single-process mode initialization and [Python Bindings](../interfaces/python-bindings.md) for the Python API surface.

## 6. Configuration Resolution

There are two distinct categories of runtime settings, and they follow different resolution rules:

### 6.1 Resource Allocations (read-only from environment)

Resource allocations are determined by the MPI launcher and job scheduler. The program reads them from the environment and must not override them. These values are never sourced from `config.json` or compiled defaults.

| Parameter       | Source                                                 | Description                   |
| --------------- | ------------------------------------------------------ | ----------------------------- |
| MPI rank count  | MPI launcher (`mpiexec -n`, `srun`)                    | Number of processes           |
| CPUs per task   | Scheduler (`SLURM_CPUS_PER_TASK`) or `OMP_NUM_THREADS` | Threads per rank              |
| Memory per node | Scheduler (`SLURM_MEM_PER_NODE`)                       | Memory budget for pool sizing |
| Job ID          | Scheduler (`SLURM_JOB_ID`)                             | Recorded in output metadata   |

**Rationale:** Allowing config.json to override resource allocations would create dangerous mismatches — e.g., the program spawning 8 threads on a node where SLURM allocated 2 CPUs, causing oversubscription. Resource allocations are a contract between the job scheduler and the process; the program observes them, it does not negotiate.

If `OMP_NUM_THREADS` is not set and no scheduler is detected, the program defaults to 1 thread per rank.

### 6.2 Algorithm Parameters (config hierarchy)

Algorithm parameters (tolerances, buffer sizes, stopping rules, etc.) are resolved in priority order:

1. **`config.json`** — Explicit user configuration. See [Configuration Reference](../configuration/configuration-reference.md)
2. **Compiled defaults** — Internal constants for any parameter not specified in `config.json`

The resolved configuration is recorded in the training metadata file for reproducibility (see [Output Infrastructure](../data-model/output-infrastructure.md) SS2).

### 6.3 Scheduler Detection

Cobre detects the job scheduler environment at startup to read resource allocations.

**Supported schedulers:**

| Scheduler  | Detection                      | Initial Support |
| ---------- | ------------------------------ | --------------- |
| SLURM      | `SLURM_JOB_ID` environment var | Yes             |
| PBS/Torque | `PBS_JOBID` environment var    | Future          |
| LSF        | `LSB_JOBID` environment var    | Future          |
| Local      | No scheduler env vars detected | Yes (fallback)  |

If no scheduler is detected, the program falls back to local defaults: 1 thread per rank, no memory budget constraint.

> **Scope note:** SLURM is the primary target scheduler. PBS and LSF support is planned for future releases and listed here for completeness. The detection mechanism is the same (environment variable probing), so adding new schedulers is straightforward.

## 7. Signal Handling and Graceful Shutdown

Cobre installs signal handlers to support graceful shutdown during long-running training and simulation phases.

| Signal    | Behavior                                                                              |
| --------- | ------------------------------------------------------------------------------------- |
| `SIGTERM` | Graceful shutdown: set shutdown flag, checkpoint last completed iteration, exit       |
| `SIGINT`  | Same as `SIGTERM` — checkpoint last completed iteration, exit with code 130           |
| `SIGKILL` | Immediate termination (cannot be caught). Recovery via crash protocol on next startup |

**Graceful shutdown protocol:**

1. The signal handler sets a global shutdown flag.
2. The program does **not** wait for the current iteration to finish — iterations at production scale can take minutes, and SIGTERM is expected to result in a fast exit.
3. A checkpoint is written from the **last fully completed iteration's** policy state. This state is always consistent and ready to serialize.
4. All MPI ranks coordinate shutdown via a barrier before finalization.
5. The training manifest is updated with `status: "partial"` and the last completed iteration number.

This ensures that a `SIGTERM` from SLURM (e.g., approaching wall-time limit) results in a prompt shutdown without corrupting policy or output files. The next invocation can detect the partial state via the manifest and resume from the checkpoint. See [Output Infrastructure](../data-model/output-infrastructure.md) SS1.2 for manifest status values.

## 8. Structured Output Protocol

The structured output protocol defines how Cobre CLI responses are formatted for programmatic consumption. This section provides an overview and cross-references; the complete protocol specification is in the [Structured Output](../interfaces/structured-output.md) spec.

### 8.1 Output Format Negotiation

The `--output-format` global flag (SS3.1) selects between three presentation modes:

| Mode       | Flag Value        | Transport                        | Use Case                                         |
| ---------- | ----------------- | -------------------------------- | ------------------------------------------------ |
| Human      | `human` (default) | Text to stdout                   | Interactive terminal, HPC batch log files        |
| JSON       | `json`            | Single JSON document to stdout   | Programmatic result consumption, CI/CD pipelines |
| JSON-lines | `json-lines`      | Newline-delimited JSON to stdout | Real-time progress monitoring by agents and TUI  |

The output format affects only presentation. It does not change computation, output files on disk, or exit codes.

### 8.2 Response Envelope

All JSON responses use a common envelope schema. See [Structured Output SS2](../interfaces/structured-output.md) for the complete JSON Schema definition:

```json
{
  "$schema": "urn:cobre:response:v1",
  "command": "<subcommand>",
  "success": true,
  "exit_code": 0,
  "cobre_version": "2.0.0",
  "errors": [],
  "warnings": [],
  "data": { ... },
  "summary": { ... }
}
```

### 8.3 JSON-Lines Streaming

For long-running operations (`run`), the JSON-lines format emits per-iteration progress events matching the fields defined in [Convergence Monitoring SS2.4](./convergence-monitoring.md). The streaming protocol uses four envelope types: `started`, `progress`, `terminated`, and `result`. See [Structured Output SS3](../interfaces/structured-output.md) for the complete streaming protocol and [Convergence Monitoring SS4.1](./convergence-monitoring.md) for the JSON-lines schema.

## Cross-References

- [Configuration Reference](../configuration/configuration-reference.md) — Complete `config.json` schema and parameter documentation
- [Input Loading Pipeline](./input-loading-pipeline.md) — How input files are loaded after CLI parsing and config resolution
- [Validation Architecture](./validation-architecture.md) — Multi-layer validation executed during the Validation phase
- [Design Principles](../overview/design-principles.md) — Format selection and declaration order invariance governing input processing
- [Production Scale Reference](../overview/production-scale-reference.md) — Typical phase durations and resource requirements at production scale
- [Output Infrastructure](../data-model/output-infrastructure.md) — Manifest files, metadata, crash recovery protocol
- [SLURM Deployment](../hpc/slurm-deployment.md) — Job scripts and multi-node deployment patterns
- [Structured Output](../interfaces/structured-output.md) — Full JSON schema definitions for CLI response envelope, error schema, and JSON-lines streaming protocol
- [MCP Server](../interfaces/mcp-server.md) — MCP tool, resource, and prompt definitions for agent interaction
- [Python Bindings](../interfaces/python-bindings.md) — PyO3 API surface, zero-copy data paths, GIL management
- [Terminal UI](../interfaces/terminal-ui.md) — TUI event consumption, convergence plot, interactive features
