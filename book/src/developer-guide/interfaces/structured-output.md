# Structured Output Protocol

> **Status: Not Implemented.** This spec describes a planned design that has not yet been implemented.

## Purpose

This spec defines the Cobre structured output protocol: the JSON response envelope schema, the structured error schema with the complete error kind registry, the JSON-lines streaming protocol for long-running operations, per-subcommand output specifications, output format negotiation via CLI flags, and the schema versioning strategy. Together, these define the machine-parseable interface that agents, CI/CD pipelines, and programmatic tools use to interact with the Cobre CLI. The protocol implements the four agent-readability design rules from [Design Principles](../overview/design-principles.md) SS6.2 and the four design decisions resolved in the structured CLI impact assessment (findings-019 SS2.1--SS2.4): human-first default, JSON-lines streaming via structured stdout with envelope, hybrid subcommand detection, and the generalized validation report as error schema standard.

## 1. Response Envelope Schema

Every CLI subcommand produces a response conforming to this envelope when invoked with `--output-format json`. The envelope generalizes the existing validation report format ([Validation Architecture](../architecture/validation-architecture.md) SS5) into a universal response schema.

### 1.1 Top-Level Fields

| Field           | Type      | Required | Description                                                                                                  |
| --------------- | --------- | -------- | ------------------------------------------------------------------------------------------------------------ |
| `$schema`       | `string`  | Yes      | Schema version URN. See SS6 for format and evolution rules                                                   |
| `command`       | `string`  | Yes      | Subcommand that produced this response (`init`, `run`, `validate`, `report`, `summary`, `schema`, `version`) |
| `success`       | `boolean` | Yes      | `true` if the operation completed without errors; `false` otherwise                                          |
| `exit_code`     | `integer` | Yes      | Numeric exit code matching [CLI and Lifecycle](../architecture/cli-and-lifecycle.md) SS4                     |
| `cobre_version` | `string`  | Yes      | Cobre version string (SemVer, e.g. `"2.0.0"`)                                                                |
| `errors`        | `array`   | Yes      | Array of structured error records (SS2). Empty array when `success` is `true`                                |
| `warnings`      | `array`   | Yes      | Array of structured warning records (same schema as errors). May be non-empty even on success                |
| `data`          | `object`  | Yes      | Subcommand-specific result payload (SS4). `null` when the subcommand produces no data on failure             |
| `summary`       | `object`  | Yes      | Subcommand-specific summary statistics. `null` when not applicable                                           |

### 1.2 Invariants

> **Invariant 1**: When `success` is `false`, `errors` is non-empty and `exit_code` is non-zero.

> **Invariant 2**: When `success` is `true`, `errors` is an empty array and `exit_code` is `0`.

> **Invariant 3**: `warnings` may be non-empty regardless of the value of `success`.

### 1.3 Complete Example

```json
{
  "$schema": "urn:cobre:response:v1",
  "command": "validate",
  "success": false,
  "exit_code": 3,
  "cobre_version": "2.0.0",
  "errors": [
    {
      "kind": "MissingReference",
      "message": "bus_id 'BUS_99' not found in buses.json",
      "context": {
        "file": "system/hydros.json",
        "entity_id": "ITAIPU",
        "field": "bus_id",
        "referenced_value": "BUS_99",
        "referenced_registry": "buses.json"
      },
      "suggestion": "Check that bus_id 'BUS_99' exists in system/buses.json. Available bus IDs can be listed from that file."
    }
  ],
  "warnings": [
    {
      "kind": "PhysicalConstraint",
      "message": "Thermal 'THERMAL_OLD' has max_generation=0 for all stages",
      "context": {
        "file": "system/thermals.json",
        "entity_id": "THERMAL_OLD"
      },
      "suggestion": "This thermal is effectively inactive. If this is intentional, no action is needed."
    }
  ],
  "data": null,
  "summary": {
    "files_checked": 24,
    "entities_validated": 456,
    "error_count": 1,
    "warning_count": 1
  }
}
```

## 2. Error Schema

### 2.1 Error Record Fields

Every error and warning record contains the following four fields, implementing [Design Principles](../overview/design-principles.md) SS6.2 Rule 2.

| Field        | Type               | Required | Description                                                                                                                           |
| ------------ | ------------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `kind`       | `string`           | Yes      | Stable error kind identifier from the registry (SS2.3). Used for programmatic dispatch                                                |
| `message`    | `string`           | Yes      | Human-readable description of the error                                                                                               |
| `context`    | `object`           | Yes      | Structured key-value data about the error location and cause. Fields vary by kind (see SS2.3)                                         |
| `suggestion` | `string` or `null` | Yes      | Actionable remediation hint, or `null` when no specific suggestion applies. Written for an agent audience (SS4.4 of architecture-021) |

### 2.2 Design Principles for Error Records

Error records follow the agent-friendly conventions from architecture-021 SS4.4:

- **Specific**: Identify the file, field, entity, stage, or iteration where the problem occurred
- **Actionable**: Tell the consumer what to check or inspect, not just what went wrong
- **Contextual**: Include identifiers (entity IDs, stage numbers, iteration counts) needed to locate the problem
- **Non-prescriptive**: Suggest what to investigate, not what the fix must be

### 2.3 Error Kind Registry

The error kind registry is the single authoritative list of all structured error kinds emitted by Cobre. Error kinds are stable identifiers governed by semantic versioning (SS6): additions are non-breaking; removals or renames are breaking changes.

The registry is organized into two categories: **validation kinds** (emitted during input validation, corresponding to [Validation Architecture](../architecture/validation-architecture.md) SS4) and **runtime kinds** (emitted during execution or post-hoc operations).

#### 2.3.1 Validation Error Kinds

These 14 kinds map to the validation layers defined in [Validation Architecture](../architecture/validation-architecture.md) SS2.

| Kind                   | Severity         | Validation Layer      | Description                                                          | Context Fields                                                          |
| ---------------------- | ---------------- | --------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `MissingFile`          | Error            | Layer 1 (Structural)  | A required input file is missing from the case directory             | `file`                                                                  |
| `ParseError`           | Error            | Layer 1 (Structural)  | A file exists but cannot be parsed (malformed JSON, invalid Parquet) | `file`, `line` (if applicable), `detail`                                |
| `SchemaViolation`      | Error            | Layer 2 (Schema)      | A file field does not conform to its expected schema                 | `file`, `entity_id`, `field`, `expected`, `actual`                      |
| `TypeMismatch`         | Error            | Layer 2 (Schema)      | A field value has an incorrect data type                             | `file`, `entity_id`, `field`, `expected_type`, `actual_type`            |
| `OutOfRange`           | Error            | Layer 2 (Schema)      | A numeric value is outside its valid range                           | `file`, `entity_id`, `field`, `value`, `min`, `max`                     |
| `InvalidEnum`          | Error            | Layer 2 (Schema)      | A string value is not a member of the expected enumeration           | `file`, `entity_id`, `field`, `value`, `allowed_values`                 |
| `DuplicateId`          | Error            | Layer 2 (Schema)      | An entity ID appears more than once in the same registry             | `file`, `entity_id`, `registry`                                         |
| `MissingReference`     | Error            | Layer 3 (Referential) | A foreign key references a non-existent entity in another registry   | `file`, `entity_id`, `field`, `referenced_value`, `referenced_registry` |
| `CoverageMismatch`     | Error            | Layer 4 (Dimensional) | Cross-file dimensional coverage is incomplete                        | `file`, `entity_id`, `missing_dimension`, `expected_coverage`           |
| `StageMismatch`        | Error            | Layer 4 (Dimensional) | Stage-related dimensional inconsistency                              | `file`, `entity_id`, `stage_id`, `detail`                               |
| `IncompatibleSettings` | Error            | Layer 5 (Semantic)    | Configuration settings are mutually incompatible                     | `file`, `settings`, `detail`                                            |
| `PhysicalConstraint`   | Error or Warning | Layer 5 (Semantic)    | A domain-specific physical rule is violated                          | `file`, `entity_id`, `rule`, `detail`                                   |
| `CapacityViolation`    | Error            | Layer 5 (Semantic)    | Capacity bounds are inconsistent (e.g. `min > max`)                  | `file`, `entity_id`, `field_min`, `value_min`, `field_max`, `value_max` |
| `PenaltyConsistency`   | Error            | Layer 5 (Semantic)    | Penalty values violate the required priority ordering                | `file`, `entity_id`, `penalty_field`, `value`, `violated_rule`          |

> **Note on severity**: `PhysicalConstraint` is the only validation kind that may appear as either an error or a warning. For example, an unused entity (thermal with `max_generation = 0` for all stages) is a `PhysicalConstraint` warning, while an acyclic cascade violation is a `PhysicalConstraint` error. All other validation kinds are always errors.

> **Mapping to existing catalog**: These 14 kinds refine and generalize the error kinds from [Validation Architecture](../architecture/validation-architecture.md) SS4. The mapping is: `FileNotFound` becomes `MissingFile`; `InvalidReference` becomes `MissingReference`; `InvalidValue` becomes `OutOfRange`; `CycleDetected`, `BusinessRuleViolation`, `UnusedEntity`, and `ModelQuality` are subsumed by `PhysicalConstraint`, `IncompatibleSettings`, `CapacityViolation`, or `PenaltyConsistency` as appropriate; `DimensionMismatch` splits into `CoverageMismatch` and `StageMismatch`; `WarmStartIncompatible`, `ResumeIncompatible`, and `NotImplemented` are expressed as `IncompatibleSettings`.

**Example -- validation error**:

```json
{
  "kind": "OutOfRange",
  "message": "Hydro 'SOBRADINHO' has storage_min (5000) greater than storage_max (4500)",
  "context": {
    "file": "system/hydros.json",
    "entity_id": "SOBRADINHO",
    "field": "storage_min",
    "value": 5000,
    "min": 0,
    "max": 4500
  },
  "suggestion": "Check that storage_min <= storage_max in the hydro definition. Current values: storage_min=5000, storage_max=4500."
}
```

**Example -- validation warning**:

```json
{
  "kind": "PhysicalConstraint",
  "message": "PAR seasonal polynomial for hydro 'TUCURUI' at season 3 has root inside the unit circle",
  "context": {
    "file": "scenarios/inflow_ar_coefficients.parquet",
    "entity_id": "TUCURUI",
    "rule": "par_seasonal_stability",
    "detail": "Characteristic polynomial root magnitude: 0.87"
  },
  "suggestion": "The PAR model for this hydro-season may be unstable. Review the AR coefficients for season 3. This is a warning and does not prevent execution."
}
```

#### 2.3.2 Runtime Error Kinds

These 6 kinds cover errors that occur during execution or post-hoc operations, outside the scope of input validation.

| Kind               | Severity | Applicable Subcommands | Description                                                           | Context Fields                                                            |
| ------------------ | -------- | ---------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `SolverFailure`    | Error    | `run`                  | LP solver returned an unexpected status (infeasible, unbounded, etc.) | `stage`, `iteration`, `opening`, `solver_status`, `active_penalty_slacks` |
| `MpiError`         | Error    | `run`                  | MPI communication failure                                             | `operation`, `rank`, `error_code`, `detail`                               |
| `CheckpointFailed` | Error    | `run`                  | Checkpoint write or read failed                                       | `checkpoint_path`, `operation` (`read` or `write`), `detail`              |
| `OutputCorrupted`  | Error    | `report`, `summary`    | An output file exists but is unreadable or has invalid structure      | `file`, `detail`                                                          |
| `OutputNotFound`   | Error    | `report`, `summary`    | A required output file or directory is missing                        | `path`, `expected_content`                                                |
| `IncompatibleRuns` | Error    | `summary`              | Two runs cannot be compared due to mismatched configuration           | `field`, `run_a_value`, `run_b_value`, `detail`                           |

**Example -- runtime error**:

```json
{
  "kind": "SolverFailure",
  "message": "LP infeasible at stage 5, iteration 23, opening 7",
  "context": {
    "stage": 5,
    "iteration": 23,
    "opening": 7,
    "solver_status": "infeasible",
    "active_penalty_slacks": ["deficit_SE", "deficit_S"]
  },
  "suggestion": "Check penalty costs in penalties.json; large deficits may indicate missing generation capacity or overly tight constraints."
}
```

**Example -- post-hoc error**:

```json
{
  "kind": "IncompatibleRuns",
  "message": "Cannot compare runs: stage count differs (120 vs 60)",
  "context": {
    "field": "stages",
    "run_a_value": 120,
    "run_b_value": 60,
    "detail": "Both runs must have the same number of stages for a meaningful comparison"
  },
  "suggestion": "Verify that both output directories correspond to the same case configuration or adjust comparison parameters."
}
```

## 3. JSON-Lines Streaming Protocol

Long-running operations (training, simulation) emit a stream of self-describing JSON objects to stdout when `--output-format json-lines` is specified. This implements [Design Principles](../overview/design-principles.md) SS6.2 Rule 3 and the streaming transport decision from findings-019 SS2.2.

### 3.1 Line Format Rules

1. **One JSON object per line**: Each line is a complete, independently valid JSON object terminated by a newline character (`\n`, U+000A).
2. **No pretty-printing**: JSON objects are serialized on a single line with no embedded newlines.
3. **UTF-8 encoding**: All output is UTF-8 encoded.
4. **Type-discriminated**: Every JSON object contains a `type` field as the first key, enabling consumers to dispatch by envelope type without parsing the full object.
5. **Ordered emission**: Lines are emitted in the order defined below (SS3.2). Consumers may rely on this ordering.

### 3.2 Envelope Types

The JSON-lines stream uses four envelope types, emitted in a fixed lifecycle order:

```
started  -->  progress (repeated)  -->  terminated  -->  result
```

| Envelope Type | Cardinality | Description                                                           |
| ------------- | ----------- | --------------------------------------------------------------------- |
| `started`     | Exactly 1   | Emitted at the beginning of the operation, before the first iteration |
| `progress`    | 0 or more   | Emitted once per training iteration or at periodic intervals          |
| `terminated`  | Exactly 1   | Emitted when the iterative loop ends, with termination reason         |
| `result`      | Exactly 1   | Emitted as the final line, containing the complete response envelope  |

#### 3.2.1 `started` Payload

Emitted once when training or simulation begins. Corresponds to the `TrainingStarted` lifecycle event from architecture-021 SS3.2.

```json
{
  "type": "started",
  "case": "/data/case_001",
  "stages": 120,
  "hydros": 164,
  "thermals": 82,
  "ranks": 8,
  "threads_per_rank": 16,
  "timestamp": "2026-02-25T10:00:00Z",
  "cobre_version": "2.0.0"
}
```

| Field              | Type      | Description                           |
| ------------------ | --------- | ------------------------------------- |
| `type`             | `string`  | Always `"started"`                    |
| `case`             | `string`  | Absolute path to the case directory   |
| `stages`           | `integer` | Number of stages in the policy graph  |
| `hydros`           | `integer` | Number of hydro plants                |
| `thermals`         | `integer` | Number of thermal plants              |
| `ranks`            | `integer` | Number of MPI ranks                   |
| `threads_per_rank` | `integer` | Number of Rayon threads per rank      |
| `timestamp`        | `string`  | ISO 8601 timestamp of operation start |
| `cobre_version`    | `string`  | Cobre version string                  |

#### 3.2.2 `progress` Payload

Emitted once per training iteration after convergence evaluation. The payload matches the `ConvergenceUpdate` event struct from architecture-021 SS3.3 and the per-iteration output record from [Convergence Monitoring](../architecture/convergence-monitoring.md) SS2.4.

```json
{
  "type": "progress",
  "iteration": 42,
  "lower_bound": 72105.4,
  "upper_bound": 73211.8,
  "upper_bound_std": 1842.3,
  "ci_95": 361.1,
  "gap": 0.0151,
  "wall_time_ms": 523400,
  "iteration_time_ms": 12400
}
```

| Field               | Type      | Description                                                         |
| ------------------- | --------- | ------------------------------------------------------------------- |
| `type`              | `string`  | Always `"progress"`                                                 |
| `iteration`         | `integer` | Iteration index (1-based)                                           |
| `lower_bound`       | `number`  | Lower bound value (stage-1 LP objective)                            |
| `upper_bound`       | `number`  | Upper bound value (mean forward cost)                               |
| `upper_bound_std`   | `number`  | Standard deviation of forward costs                                 |
| `ci_95`             | `number`  | 95% confidence interval half-width ($1.96 \cdot \sigma / \sqrt{N}$) |
| `gap`               | `number`  | Relative gap $(UB - LB) / \lvert UB \rvert$                         |
| `wall_time_ms`      | `integer` | Cumulative wall-clock time in milliseconds                          |
| `iteration_time_ms` | `integer` | Wall-clock time for this iteration in milliseconds                  |

> **Canonical source**: The 8 fields of the `progress` payload are exactly the 8 fields of the per-iteration output record defined in [Convergence Monitoring](../architecture/convergence-monitoring.md) SS2.4. Implementations must keep these in sync. The `ConvergenceUpdate` Rust struct (architecture-021 SS3.3) is the shared type definition.

For simulation operations, a simulation-specific progress event is emitted instead:

```json
{
  "type": "progress",
  "phase": "simulation",
  "scenarios_complete": 500,
  "scenarios_total": 2000,
  "elapsed_ms": 34200
}
```

| Field                | Type      | Description                                     |
| -------------------- | --------- | ----------------------------------------------- |
| `type`               | `string`  | Always `"progress"`                             |
| `phase`              | `string`  | `"simulation"` to distinguish from training     |
| `scenarios_complete` | `integer` | Number of simulation scenarios completed so far |
| `scenarios_total`    | `integer` | Total number of simulation scenarios            |
| `elapsed_ms`         | `integer` | Elapsed time in milliseconds                    |

#### 3.2.3 `terminated` Payload

Emitted once when the iterative loop ends. Corresponds to the `TrainingFinished` lifecycle event from architecture-021 SS3.2.

```json
{
  "type": "terminated",
  "reason": "bound_stalling",
  "iterations": 87,
  "final_lb": 72105.4,
  "final_ub": 73211.8,
  "total_time_ms": 1082400,
  "total_cuts": 10440
}
```

| Field           | Type      | Description                                                                                               |
| --------------- | --------- | --------------------------------------------------------------------------------------------------------- |
| `type`          | `string`  | Always `"terminated"`                                                                                     |
| `reason`        | `string`  | Termination reason: `"bound_stalling"`, `"simulation"`, `"iteration_limit"`, `"time_limit"`, `"shutdown"` |
| `iterations`    | `integer` | Total iterations completed                                                                                |
| `final_lb`      | `number`  | Final lower bound value                                                                                   |
| `final_ub`      | `number`  | Final upper bound value                                                                                   |
| `total_time_ms` | `integer` | Total wall-clock time in milliseconds                                                                     |
| `total_cuts`    | `integer` | Total cuts generated across all stages                                                                    |

#### 3.2.4 `result` Payload

Emitted as the final line of the stream. Contains the complete response envelope (SS1) with the `data` field populated by the subcommand-specific result (SS4).

```json
{
  "type": "result",
  "$schema": "urn:cobre:response:v1",
  "command": "run",
  "success": true,
  "exit_code": 0,
  "cobre_version": "2.0.0",
  "errors": [],
  "warnings": [],
  "data": {
    "output_directory": "/data/case_001/output",
    "training": {
      "iterations": 87,
      "final_lower_bound": 72105.4,
      "final_upper_bound": 73211.8,
      "final_gap": 0.0151,
      "termination_reason": "bound_stalling",
      "total_cuts": 10440,
      "wall_time_ms": 1082400
    },
    "simulation": {
      "scenarios": 2000,
      "output_files": ["simulation/costs.parquet", "simulation/hydros.parquet"],
      "wall_time_ms": 245000
    }
  },
  "summary": {
    "status": "completed",
    "total_time_ms": 1327400
  }
}
```

### 3.3 Partial Output Semantics

> **Crash recovery**: Each line in the JSON-lines stream is independently valid JSON. If the process crashes or is killed mid-stream, a consumer can parse all completed lines without error. The absence of a `result` line indicates abnormal termination. The absence of a `terminated` line indicates the iterative loop did not complete.

> **Consumer guidance**: Consumers should not assume the `result` line will be present. A robust consumer treats the last successfully parsed `progress` line as the best-known state if no `terminated` or `result` line appears.

### 3.4 Relationship to Event Stream Architecture

The JSON-lines streaming protocol is one consumer of the shared event stream defined in architecture-021 SS3. The JSON-lines writer subscribes to the `broadcast` channel alongside other consumers (text logger, TUI renderer, Parquet writer). The `progress` payload is the JSON serialization of the `ConvergenceUpdate` event type. The `started` and `terminated` payloads serialize the `TrainingStarted` and `TrainingFinished` lifecycle events, respectively. See architecture-021 SS3.4 for the consumer registration pattern.

## 4. Subcommand Output Specifications

Each subcommand defines the shape of the `data` field in the response envelope (SS1) and its behavior in both `json` and `json-lines` output modes. Subcommand definitions are derived from the inventory in findings-019 SS2.3 and the phase mapping in [CLI and Lifecycle](../architecture/cli-and-lifecycle.md) SS5.

### 4.1 `run`

Executes the full lifecycle (validation, training, simulation) or a subset thereof.

**`--output-format json`**: Emits a single response envelope after the operation completes. No streaming progress is emitted. This is the appropriate mode when the consumer only needs the final result (e.g., CI/CD pipelines, batch post-processing).

**`--output-format json-lines`**: Emits the full streaming protocol (SS3): `started`, `progress` (per iteration), `terminated`, and `result`. This is the appropriate mode for real-time monitoring.

**`data` shape**:

```json
{
  "output_directory": "/data/case_001/output",
  "training": {
    "iterations": 87,
    "final_lower_bound": 72105.4,
    "final_upper_bound": 73211.8,
    "final_gap": 0.0151,
    "termination_reason": "bound_stalling",
    "total_cuts": 10440,
    "wall_time_ms": 1082400
  },
  "simulation": {
    "scenarios": 2000,
    "output_files": [
      "simulation/costs.parquet",
      "simulation/hydros.parquet",
      "simulation/thermals.parquet",
      "simulation/exchanges.parquet"
    ],
    "wall_time_ms": 245000
  }
}
```

| Field                         | Type      | Description                                                       |
| ----------------------------- | --------- | ----------------------------------------------------------------- |
| `output_directory`            | `string`  | Absolute path to the output directory                             |
| `training.iterations`         | `integer` | Total training iterations completed                               |
| `training.final_lower_bound`  | `number`  | Final lower bound                                                 |
| `training.final_upper_bound`  | `number`  | Final upper bound                                                 |
| `training.final_gap`          | `number`  | Final relative gap                                                |
| `training.termination_reason` | `string`  | Stopping rule that triggered termination                          |
| `training.total_cuts`         | `integer` | Total Benders cuts generated                                      |
| `training.wall_time_ms`       | `integer` | Training wall-clock time in milliseconds                          |
| `simulation.scenarios`        | `integer` | Number of simulation scenarios evaluated                          |
| `simulation.output_files`     | `array`   | Relative paths to generated Parquet files within output directory |
| `simulation.wall_time_ms`     | `integer` | Simulation wall-clock time in milliseconds                        |

> **Note**: If training is disabled (`training.enabled = false`), the `training` field is `null`. If simulation is disabled, the `simulation` field is `null`. At least one must be non-null for a `run` to succeed.

### 4.2 `validate`

Validates all input files in a case directory without executing the solver.

**`--output-format json`**: Emits a single response envelope. This is the primary use case for `validate`.

**`--output-format json-lines`**: Behaves identically to `json` mode. Validation is not a streaming operation; there is no iterative progress to report. A single JSON object is emitted on one line.

**`data` shape**:

```json
{
  "case_directory": "/data/case_001",
  "timestamp": "2026-02-25T10:05:00Z",
  "layers_completed": 5,
  "files_checked": 24,
  "entities_validated": 456
}
```

| Field                | Type      | Description                                        |
| -------------------- | --------- | -------------------------------------------------- |
| `case_directory`     | `string`  | Absolute path to the validated case directory      |
| `timestamp`          | `string`  | ISO 8601 timestamp of validation completion        |
| `layers_completed`   | `integer` | Number of validation layers fully completed (1--5) |
| `files_checked`      | `integer` | Total files checked across all layers              |
| `entities_validated` | `integer` | Total entities validated across all registries     |

**`summary` shape** (same for both modes):

```json
{
  "files_checked": 24,
  "entities_validated": 456,
  "error_count": 2,
  "warning_count": 1
}
```

### 4.3 `report`

Summarizes a completed run's outputs by reading manifests and metadata from an output directory.

**`--output-format json`**: Emits a single response envelope with the run summary.

**`--output-format json-lines`**: Behaves identically to `json` mode. Report generation is not a streaming operation.

**`data` shape**:

```json
{
  "output_directory": "/data/case_001/output",
  "case_name": "case_001",
  "status": "completed",
  "cobre_version": "2.0.0",
  "training": {
    "iterations": 87,
    "final_lower_bound": 72105.4,
    "final_upper_bound": 73211.8,
    "final_gap": 0.0151,
    "termination_reason": "bound_stalling",
    "total_cuts": 10440,
    "wall_time_ms": 1082400,
    "ranks": 8,
    "threads_per_rank": 16
  },
  "simulation": {
    "scenarios": 2000,
    "wall_time_ms": 245000,
    "output_files": ["simulation/costs.parquet", "simulation/hydros.parquet"]
  },
  "metadata": {
    "config_hash": "sha256:a1b2c3...",
    "system_hash": "sha256:d4e5f6...",
    "started_at": "2026-02-25T10:00:00Z",
    "finished_at": "2026-02-25T10:22:07Z"
  }
}
```

| Field              | Type     | Description                                               |
| ------------------ | -------- | --------------------------------------------------------- |
| `output_directory` | `string` | Absolute path to the output directory                     |
| `case_name`        | `string` | Case name from metadata                                   |
| `status`           | `string` | Run status from manifest: `"completed"` or `"partial"`    |
| `cobre_version`    | `string` | Version that produced the output                          |
| `training`         | `object` | Training summary (null if training was not performed)     |
| `simulation`       | `object` | Simulation summary (null if simulation was not performed) |
| `metadata`         | `object` | Configuration and system hashes, timestamps               |

### 4.4 `compare`

Compares two completed run output directories.

**`--output-format json`**: Emits a single response envelope with comparison results.

**`--output-format json-lines`**: Behaves identically to `json` mode. Comparison is not a streaming operation.

**`data` shape**:

```json
{
  "run_a": "/data/case_001/output_v1",
  "run_b": "/data/case_001/output_v2",
  "compatible": true,
  "training": {
    "iterations_a": 87,
    "iterations_b": 92,
    "final_lb_a": 72105.4,
    "final_lb_b": 72201.1,
    "final_lb_delta": 95.7,
    "final_lb_delta_pct": 0.0013,
    "final_ub_a": 73211.8,
    "final_ub_b": 73150.2,
    "final_ub_delta": -61.6,
    "final_ub_delta_pct": -0.0008,
    "final_gap_a": 0.0151,
    "final_gap_b": 0.013
  },
  "simulation": {
    "mean_cost_a": 72580.3,
    "mean_cost_b": 72615.1,
    "mean_cost_delta": 34.8,
    "mean_cost_delta_pct": 0.0005,
    "std_a": 1842.3,
    "std_b": 1790.1
  }
}
```

| Field                            | Type      | Description                                                       |
| -------------------------------- | --------- | ----------------------------------------------------------------- |
| `run_a`                          | `string`  | Absolute path to first output directory                           |
| `run_b`                          | `string`  | Absolute path to second output directory                          |
| `compatible`                     | `boolean` | Whether the two runs are structurally compatible for comparison   |
| `training`                       | `object`  | Training metric deltas (null if either run lacks training output) |
| `training.*_delta`               | `number`  | Absolute delta (b minus a)                                        |
| `training.*_delta_pct`           | `number`  | Relative delta as a fraction                                      |
| `simulation`                     | `object`  | Simulation metric deltas (null if either run lacks simulation)    |
| `simulation.mean_cost_delta`     | `number`  | Absolute delta in mean simulation cost                            |
| `simulation.mean_cost_delta_pct` | `number`  | Relative delta as a fraction                                      |

### 4.5 `version`

Prints version and build information.

**`--output-format json`**: Emits a single response envelope with version data.

**`--output-format json-lines`**: Behaves identically to `json` mode.

**`data` shape**:

```json
{
  "version": "2.0.0",
  "git_commit": "a1b2c3d",
  "build_date": "2026-02-20",
  "rustc_version": "1.85.0",
  "solver_backend": "highs",
  "solver_version": "1.9.0",
  "mpi_available": true,
  "features": ["mpi", "highs"]
}
```

| Field            | Type      | Description                                           |
| ---------------- | --------- | ----------------------------------------------------- |
| `version`        | `string`  | Cobre SemVer version                                  |
| `git_commit`     | `string`  | Short git commit hash of the build                    |
| `build_date`     | `string`  | ISO 8601 date of the build                            |
| `rustc_version`  | `string`  | Rust compiler version used for the build              |
| `solver_backend` | `string`  | Compiled solver backend (`"highs"`, `"clp"`, etc.)    |
| `solver_version` | `string`  | Version of the solver library                         |
| `mpi_available`  | `boolean` | Whether MPI support is compiled in (ferrompi feature) |
| `features`       | `array`   | List of enabled Cargo feature flags                   |

### 4.6 Summary of Output Mode Behavior

| Subcommand | `--output-format json`           | `--output-format json-lines`                        |
| ---------- | -------------------------------- | --------------------------------------------------- |
| `init`     | Single envelope                  | Single line (identical to json)                     |
| `run`      | Single envelope after completion | Streaming: started + progress + terminated + result |
| `validate` | Single envelope                  | Single line (identical to json)                     |
| `report`   | Single envelope                  | Single line (identical to json)                     |
| `summary`  | Single envelope                  | Single line (identical to json)                     |
| `schema`   | Single envelope                  | Single line (identical to json)                     |
| `version`  | Single envelope                  | Single line (identical to json)                     |

## 5. Output Format Negotiation

### 5.1 The `--output-format` Flag

The global CLI flag `--output-format <format>` controls the presentation layer. It does not affect computation, solver behavior, or output files written to disk.

| Value        | Description                                                                                     |
| ------------ | ----------------------------------------------------------------------------------------------- |
| `human`      | Default. Human-readable text output: decorated headers, formatted tables, text training log     |
| `json`       | Single JSON response envelope on stdout. Machine-parseable. No decorative text                  |
| `json-lines` | JSON-lines streaming to stdout. One JSON object per line. Machine-parseable. No decorative text |

> **Design decision reference**: The human-first default follows findings-019 SS2.1. Agents must explicitly opt in to structured output via `--output-format json` or `--output-format json-lines`.

### 5.2 Interaction with `--quiet` and `--no-progress`

| Flag            | Effect in `human` mode                                    | Effect in `json` mode                 | Effect in `json-lines` mode                                                      |
| --------------- | --------------------------------------------------------- | ------------------------------------- | -------------------------------------------------------------------------------- |
| (no flags)      | Full text output: header, per-iteration lines, summary    | Full envelope                         | Full stream: started + progress + terminated + result                            |
| `--quiet`       | Suppresses all stdout; exit code only                     | Suppresses all stdout                 | Suppresses all stdout                                                            |
| `--no-progress` | Suppresses per-iteration lines; header and summary remain | No effect (no streaming in json mode) | Suppresses `progress` lines; `started`, `terminated`, and `result` still emitted |

> **Rationale for `--quiet`**: Some CI/CD pipelines only inspect the exit code and the output files on disk. `--quiet` eliminates all stdout, reducing log noise. This applies uniformly across all output modes.

> **Rationale for `--no-progress`**: In `json-lines` mode, suppressing `progress` lines reduces output volume for consumers that only need the lifecycle events and final result. The `started` and `terminated` envelopes are retained because they carry metadata not available in the `result` envelope.

### 5.3 TTY Detection

Cobre does not auto-detect whether stdout is a TTY to switch output formats. The output format is always determined by the explicit `--output-format` flag (defaulting to `human`).

> **Rationale**: Auto-detection creates unpredictable behavior for agents that redirect stdout through pipes, MPI launchers, or container runtimes where TTY status is ambiguous. Explicit is better than implicit (findings-019 SS2.1).

TTY status is used for one purpose only: when `--output-format human` is active and stdout is a TTY, the text logger may use ANSI color codes for emphasis. When stdout is not a TTY, color codes are suppressed. This is purely cosmetic and does not affect content.

### 5.4 MPI Rank-0 Rule

> **Rule**: In MPI execution, only rank 0 produces structured output to stdout. Non-rank-0 processes suppress all stdout and stderr output. This is consistent with existing behavior documented in [CLI and Lifecycle](../architecture/cli-and-lifecycle.md) SS5.2 and the MPI note in findings-019 SS2.3.

This rule applies to all output formats (`human`, `json`, `json-lines`). The structured output protocol is a rank-0-only concern. Output files written to disk (Parquet, manifests, metadata) follow the existing distributed I/O patterns documented in [Design Principles](../overview/design-principles.md) SS2 Goal 5.

## 6. Schema Versioning

### 6.1 `$schema` URN Format

Every JSON response envelope includes a `$schema` field with a URN that identifies the schema version:

```
urn:cobre:response:v1
```

**URN structure**:

| Component | Value      | Description                       |
| --------- | ---------- | --------------------------------- |
| Namespace | `cobre`    | Fixed namespace for Cobre schemas |
| Type      | `response` | Schema type (response envelope)   |
| Version   | `v1`       | Major version number              |

The version component is a major version only. Minor/patch updates to the schema that are additive-only do not increment the version number.

### 6.2 Additive-Only Evolution Rules

Schema evolution within a major version follows strict additive-only rules:

| Allowed (non-breaking)                                 | Forbidden (breaking)                              |
| ------------------------------------------------------ | ------------------------------------------------- |
| Add new fields to any object                           | Remove existing fields                            |
| Add new values to string enums (e.g., new error kinds) | Rename existing fields                            |
| Add new envelope types to JSON-lines protocol          | Change the type of an existing field              |
| Add new subcommand `data` shapes                       | Change the semantics of an existing field         |
| Add new context fields to error kinds                  | Remove error kinds from the registry              |
|                                                        | Remove envelope types from the streaming protocol |
|                                                        | Change the `$schema` URN format                   |

### 6.3 Breaking Change Policy

When a breaking change is required:

1. The `$schema` version is incremented (e.g., `urn:cobre:response:v2`).
2. The previous version continues to be supported for at least one major Cobre release cycle.
3. Consumers detect the version via the `$schema` field and can dispatch to version-appropriate parsing logic.
4. The structured output spec is updated with both the old and new schema documented during the overlap period.

### 6.4 Version Compatibility Check

Consumers should validate the `$schema` field before parsing:

```
if response["$schema"] starts with "urn:cobre:response:v1":
    parse with v1 schema
else:
    report unsupported schema version
```

The `cobre_version` field (SemVer) provides additional granularity for consumers that need to detect specific additive changes within a schema major version.

## Cross-References

- [Design Principles](../overview/design-principles.md) -- Agent-readability rules (SS6.2 Rules 1-4), format selection criteria (SS1), declaration order invariance (SS3)
- [Validation Architecture](../architecture/validation-architecture.md) -- Validation layers (SS2), error type catalog (SS4), validation report format (SS5) that this spec generalizes
- [Convergence Monitoring](../architecture/convergence-monitoring.md) -- Per-iteration output record (SS2.4) that defines the `progress` payload fields; training log format (SS4) that is the `human` mode equivalent
- [Training Loop](../architecture/training-loop.md) -- Iteration lifecycle (SS2.1) defining the 7 steps at which events are emitted; forward/backward pass structure
- [CLI and Lifecycle](../architecture/cli-and-lifecycle.md) -- Invocation pattern (SS2), command-line interface (SS3), exit codes (SS4), execution phases (SS5)
- [Output Infrastructure](../data-model/output-infrastructure.md) (planned update) -- Manifests and metadata accessed by the `report` subcommand
- [Configuration Reference](../configuration/configuration-reference.md) -- Stopping rules and mode selection that determine training outcomes reported in `data`
- [MCP Server](./mcp-server.md) -- MCP tool definitions that consume this protocol's error schema and event types
- [Python Bindings](./python-bindings.md) -- Python exception hierarchy mapped from this protocol's error kinds
- [Terminal UI](./terminal-ui.md) -- TUI rendering that consumes the same event stream as the JSON-lines writer
