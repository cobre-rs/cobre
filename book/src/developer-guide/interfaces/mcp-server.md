# MCP Server

> **Status: Not Implemented.** This specification describes planned behavior. The current codebase contains only a `todo!()` stub.

## Purpose

This spec defines the Cobre MCP (Model Context Protocol) server: the complete tool, resource, and prompt definitions that expose Cobre as a first-class tool for AI agents, the transport and security model for local and remote operation, the progress reporting mechanism for long-running SDDP training, and the capability negotiation handshake. The `cobre-mcp` crate implements this spec as a standalone server binary that invokes the Cobre Rust library API directly -- single-process, no MPI, Rayon threads for computation. All tool responses reuse the response envelope and error schema defined in [Structured Output](structured-output.md).

## 1. Crate Architecture

### 1.1 Crate Type and Execution Mode

| Attribute             | Value                                                                                                                        |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Crate type**        | Binary crate (`[[bin]]`)                                                                                                     |
| **Binary name**       | `cobre-mcp` (standalone) or `cobre serve` (subcommand hosted in `cobre-cli`)                                                 |
| **Execution mode**    | Single-process. No MPI. Rayon threads for computation                                                                        |
| **What it owns**      | MCP protocol handling (stdio/SSE/streamable-HTTP transport), tool dispatch, resource serving, prompt definitions, sandboxing |
| **What it delegates** | All computation to `cobre-sddp`; all I/O to `cobre-io`; all data model types from `cobre-core`                               |
| **MPI relationship**  | Never initializes MPI. The `cobre-sddp` library operates in single-rank mode                                                 |

### 1.1a Future: Multi-Process Capability

The MCP server currently operates in single-process mode, which is the recommended configuration for AI agent interaction.

Multi-process SDDP execution via TCP or shm backends is architecturally possible using the same mechanism specified for the Python bindings API ([Python Bindings](./python-bindings.md) SS7.4). Exposing multi-process capability in the MCP server would require adding `backend` and `num_workers` parameters to the `cobre/run` tool input schema (SS2.2). This extension is deferred to a future release pending demonstrated demand from MCP-based agent workflows.

See [Python Bindings](./python-bindings.md) SS2.1 for the multi-process API pattern that would be adapted for `cobre/run`.

### 1.2 Dependency Graph

```
cobre-mcp
  +-- cobre-sddp
  |     +-- cobre-core
  |     +-- cobre-stochastic
  |     +-- cobre-solver
  +-- cobre-io
  |     +-- cobre-core
  +-- cobre-core
```

The `cobre-mcp` crate does **not** depend on `ferrompi`. Multi-process execution is not currently exposed in the MCP tool interface (see SS1.1a).

### 1.3 Operation Categories

| Operation Category | Tools                                                                                                                  |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Validation         | `cobre/validate`                                                                                                       |
| Execution          | `cobre/run`                                                                                                            |
| Query (read-only)  | `cobre/query-results`, `cobre/query-convergence`, `cobre/inspect-policy`, `cobre/inspect-case`, `cobre/list-scenarios` |
| Comparison         | `cobre/compare-policies`                                                                                               |
| Export             | `cobre/export-results`                                                                                                 |
| Introspection      | `cobre/get-config-schema`                                                                                              |

## 2. Tool Definitions

All 10 tools are defined below with full input/output JSON schemas, error conditions, timeout behavior, and example request/response pairs. Tool outputs reuse the error kind registry from [Structured Output](structured-output.md) SS2.3. All tools return content blocks with `type: "text"` containing a JSON string; the JSON conforms to the schemas below.

### 2.1 `cobre/validate`

Runs the 5-layer validation pipeline ([Validation Architecture](../architecture/validation-architecture.md) SS2) on a case directory without executing the solver.

**Input Schema:**

```json
{
  "type": "object",
  "properties": {
    "case_dir": {
      "type": "string",
      "description": "Absolute path to the case directory"
    }
  },
  "required": ["case_dir"],
  "additionalProperties": false
}
```

**Output Schema (success):**

```json
{
  "type": "object",
  "properties": {
    "valid": { "type": "boolean" },
    "errors": {
      "type": "array",
      "items": { "$ref": "#/$defs/ErrorRecord" }
    },
    "warnings": {
      "type": "array",
      "items": { "$ref": "#/$defs/ErrorRecord" }
    },
    "summary": {
      "type": "object",
      "properties": {
        "files_checked": { "type": "integer" },
        "entities_validated": { "type": "integer" },
        "layers_completed": { "type": "integer" },
        "error_count": { "type": "integer" },
        "warning_count": { "type": "integer" }
      },
      "required": [
        "files_checked",
        "entities_validated",
        "layers_completed",
        "error_count",
        "warning_count"
      ]
    }
  },
  "required": ["valid", "errors", "warnings", "summary"]
}
```

> The `ErrorRecord` schema is defined in [Structured Output](structured-output.md) SS2.1: `{ kind, message, context, suggestion }`.

**Error Conditions:**

| Error Kind             | Condition                                                                                                                    |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `MissingFile`          | `case_dir` does not exist or is not a directory                                                                              |
| `ParseError`           | A required input file cannot be parsed                                                                                       |
| (all validation kinds) | Any of the 14 validation error kinds from [Structured Output](structured-output.md) SS2.3.1 may appear in the `errors` array |

**Timeout:** Default 60 seconds. Validation is typically fast (under 5 seconds for large cases). No progress notifications emitted.

**Example Request:**

```json
{
  "method": "tools/call",
  "params": {
    "name": "cobre/validate",
    "arguments": {
      "case_dir": "/data/case_001"
    }
  }
}
```

**Example Response (validation failure):**

```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"valid\":false,\"errors\":[{\"kind\":\"MissingReference\",\"message\":\"bus_id 'BUS_99' not found in buses.json\",\"context\":{\"file\":\"system/hydros.json\",\"entity_id\":\"ITAIPU\",\"field\":\"bus_id\",\"referenced_value\":\"BUS_99\",\"referenced_registry\":\"buses.json\"},\"suggestion\":\"Check that bus_id 'BUS_99' exists in system/buses.json.\"}],\"warnings\":[],\"summary\":{\"files_checked\":24,\"entities_validated\":456,\"layers_completed\":3,\"error_count\":1,\"warning_count\":0}}"
    }
  ],
  "isError": false
}
```

**Example Response (validation success):**

```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"valid\":true,\"errors\":[],\"warnings\":[],\"summary\":{\"files_checked\":24,\"entities_validated\":456,\"layers_completed\":5,\"error_count\":0,\"warning_count\":0}}"
    }
  ],
  "isError": false
}
```

### 2.2 `cobre/run`

Executes SDDP training and/or simulation on a case directory. This is the only tool that may run for minutes to hours. It emits MCP progress notifications during execution (see SS6).

**Input Schema:**

```json
{
  "type": "object",
  "properties": {
    "case_dir": {
      "type": "string",
      "description": "Absolute path to the case directory"
    },
    "phases": {
      "type": "string",
      "enum": ["training", "simulation", "both"],
      "description": "Execution phases to run. Defaults to \"both\"",
      "default": "both"
    }
  },
  "required": ["case_dir"],
  "additionalProperties": false
}
```

**Output Schema (success):**

```json
{
  "type": "object",
  "properties": {
    "status": {
      "type": "string",
      "enum": ["completed", "partial"]
    },
    "output_dir": {
      "type": "string",
      "description": "Absolute path to the output directory"
    },
    "training": {
      "type": ["object", "null"],
      "properties": {
        "iterations": { "type": "integer" },
        "final_lower_bound": { "type": "number" },
        "final_upper_bound": { "type": "number" },
        "final_gap": { "type": "number" },
        "termination_reason": { "type": "string" },
        "total_cuts": { "type": "integer" },
        "wall_time_ms": { "type": "integer" }
      },
      "required": [
        "iterations",
        "final_lower_bound",
        "final_upper_bound",
        "final_gap",
        "termination_reason",
        "total_cuts",
        "wall_time_ms"
      ]
    },
    "simulation": {
      "type": ["object", "null"],
      "properties": {
        "scenarios": { "type": "integer" },
        "output_files": {
          "type": "array",
          "items": { "type": "string" }
        },
        "wall_time_ms": { "type": "integer" }
      },
      "required": ["scenarios", "output_files", "wall_time_ms"]
    }
  },
  "required": ["status", "output_dir"]
}
```

**Error Conditions:**

| Error Kind         | Condition                                              |
| ------------------ | ------------------------------------------------------ |
| `MissingFile`      | `case_dir` does not exist or is not a directory        |
| (validation kinds) | Validation fails before execution starts               |
| `SolverFailure`    | LP solver returns infeasible/unbounded during training |
| `CheckpointFailed` | Checkpoint write fails during training                 |

**Timeout:** No default timeout. The tool runs until training completes or a stopping rule triggers. Agents should monitor progress notifications to detect stalling and may cancel the call if needed.

**Interruption Behavior:**

| Scenario                | Behavior                                                                                                         |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Training succeeds       | Returns full result with training and/or simulation summaries                                                    |
| Training fails (solver) | Returns `isError: true` with a `SolverFailure` error record. Partial output may exist on disk                    |
| Client disconnects      | Server cancels the training run at the next iteration boundary. Checkpoint of last completed iteration preserved |
| Signal (SIGTERM)        | Graceful shutdown: completes current iteration, writes checkpoint, returns partial result                        |

**Example Request:**

```json
{
  "method": "tools/call",
  "params": {
    "name": "cobre/run",
    "arguments": {
      "case_dir": "/data/case_001",
      "phases": "both"
    },
    "_meta": {
      "progressToken": "run-001"
    }
  }
}
```

**Example Response (success):**

```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"status\":\"completed\",\"output_dir\":\"/data/case_001/output\",\"training\":{\"iterations\":87,\"final_lower_bound\":72105.4,\"final_upper_bound\":73211.8,\"final_gap\":0.0151,\"termination_reason\":\"bound_stalling\",\"total_cuts\":10440,\"wall_time_ms\":1082400},\"simulation\":{\"scenarios\":2000,\"output_files\":[\"simulation/costs.parquet\",\"simulation/hydros.parquet\"],\"wall_time_ms\":245000}}"
    }
  ],
  "isError": false
}
```

### 2.3 `cobre/query-results`

Reads Hive-partitioned Parquet simulation output and returns filtered results as JSON. This tool provides agent-friendly access to the output schemas defined in [Output Schemas](../data-model/output-schemas.md).

**Input Schema:**

```json
{
  "type": "object",
  "properties": {
    "output_dir": {
      "type": "string",
      "description": "Absolute path to the output directory"
    },
    "entity_type": {
      "type": "string",
      "enum": ["hydros", "thermals", "buses", "lines", "costs"],
      "description": "Entity type to query"
    },
    "filters": {
      "type": "object",
      "description": "Optional column filters",
      "properties": {
        "entity_id": {
          "type": ["string", "array"],
          "description": "Single entity ID or array of entity IDs"
        },
        "stage_range": {
          "type": "array",
          "items": { "type": "integer" },
          "minItems": 2,
          "maxItems": 2,
          "description": "Inclusive [min_stage, max_stage] range"
        },
        "scenario_range": {
          "type": "array",
          "items": { "type": "integer" },
          "minItems": 2,
          "maxItems": 2,
          "description": "Inclusive [min_scenario, max_scenario] range"
        },
        "columns": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Columns to include in the result. Defaults to all columns"
        },
        "limit": {
          "type": "integer",
          "minimum": 1,
          "maximum": 10000,
          "default": 1000,
          "description": "Maximum number of rows to return"
        }
      },
      "additionalProperties": false
    }
  },
  "required": ["output_dir", "entity_type"],
  "additionalProperties": false
}
```

**Output Schema:**

```json
{
  "type": "object",
  "properties": {
    "columns": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Column names in the result"
    },
    "rows": {
      "type": "array",
      "items": {
        "type": "array",
        "description": "Row values in column order"
      }
    },
    "total_rows": {
      "type": "integer",
      "description": "Total rows matching the filter (before limit)"
    },
    "truncated": {
      "type": "boolean",
      "description": "True if total_rows exceeds the limit"
    }
  },
  "required": ["columns", "rows", "total_rows", "truncated"]
}
```

**Error Conditions:**

| Error Kind        | Condition                                              |
| ----------------- | ------------------------------------------------------ |
| `OutputNotFound`  | `output_dir` does not exist or lacks simulation output |
| `OutputCorrupted` | Parquet files exist but cannot be read                 |

**Timeout:** Default 30 seconds.

**Example Request:**

```json
{
  "method": "tools/call",
  "params": {
    "name": "cobre/query-results",
    "arguments": {
      "output_dir": "/data/case_001/output",
      "entity_type": "hydros",
      "filters": {
        "entity_id": "ITAIPU",
        "stage_range": [1, 12],
        "columns": ["stage_id", "scenario_id", "storage_hm3", "generation_mw"],
        "limit": 100
      }
    }
  }
}
```

**Example Response:**

```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"columns\":[\"stage_id\",\"scenario_id\",\"storage_hm3\",\"generation_mw\"],\"rows\":[[1,1,54230.5,14000.0],[1,2,54180.2,14000.0]],\"total_rows\":24000,\"truncated\":true}"
    }
  ],
  "isError": false
}
```

### 2.4 `cobre/query-convergence`

Reads the `convergence.parquet` file from a completed training run and returns the convergence history as JSON. The record schema matches [Convergence Monitoring](../architecture/convergence-monitoring.md) SS2.4.

**Input Schema:**

```json
{
  "type": "object",
  "properties": {
    "output_dir": {
      "type": "string",
      "description": "Absolute path to the output directory"
    },
    "iteration_range": {
      "type": "array",
      "items": { "type": "integer" },
      "minItems": 2,
      "maxItems": 2,
      "description": "Optional inclusive [min_iteration, max_iteration] range. Defaults to all iterations"
    }
  },
  "required": ["output_dir"],
  "additionalProperties": false
}
```

**Output Schema:**

```json
{
  "type": "object",
  "properties": {
    "iterations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "iteration": { "type": "integer" },
          "lower_bound": { "type": "number" },
          "upper_bound": { "type": "number" },
          "upper_bound_std": { "type": "number" },
          "ci_95": { "type": "number" },
          "gap": { "type": "number" },
          "wall_time_ms": { "type": "integer" },
          "iteration_time_ms": { "type": "integer" }
        },
        "required": [
          "iteration",
          "lower_bound",
          "upper_bound",
          "upper_bound_std",
          "ci_95",
          "gap",
          "wall_time_ms",
          "iteration_time_ms"
        ]
      }
    },
    "total_iterations": { "type": "integer" },
    "termination_reason": { "type": "string" }
  },
  "required": ["iterations", "total_iterations", "termination_reason"]
}
```

**Error Conditions:**

| Error Kind        | Condition                                                  |
| ----------------- | ---------------------------------------------------------- |
| `OutputNotFound`  | `output_dir` does not exist or lacks `convergence.parquet` |
| `OutputCorrupted` | `convergence.parquet` exists but cannot be read            |

**Timeout:** Default 10 seconds.

**Example Request:**

```json
{
  "method": "tools/call",
  "params": {
    "name": "cobre/query-convergence",
    "arguments": {
      "output_dir": "/data/case_001/output",
      "iteration_range": [80, 87]
    }
  }
}
```

**Example Response:**

```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"iterations\":[{\"iteration\":80,\"lower_bound\":72050.1,\"upper_bound\":73300.5,\"upper_bound_std\":1860.2,\"ci_95\":365.6,\"gap\":0.0171,\"wall_time_ms\":998000,\"iteration_time_ms\":12300},{\"iteration\":87,\"lower_bound\":72105.4,\"upper_bound\":73211.8,\"upper_bound_std\":1842.3,\"ci_95\":361.1,\"gap\":0.0151,\"wall_time_ms\":1082400,\"iteration_time_ms\":12100}],\"total_iterations\":87,\"termination_reason\":\"bound_stalling\"}"
    }
  ],
  "isError": false
}
```

### 2.5 `cobre/inspect-policy`

Reads FlatBuffers policy data from a policy directory and returns a JSON summary. Does not expose FlatBuffers internals.

**Input Schema:**

```json
{
  "type": "object",
  "properties": {
    "policy_dir": {
      "type": "string",
      "description": "Absolute path to the policy directory (typically output_dir/policy)"
    },
    "stage": {
      "type": "integer",
      "minimum": 1,
      "description": "Optional stage index to inspect. If omitted, returns summary for all stages"
    }
  },
  "required": ["policy_dir"],
  "additionalProperties": false
}
```

**Output Schema (summary mode -- no `stage` argument):**

```json
{
  "type": "object",
  "properties": {
    "stages": { "type": "integer" },
    "total_cuts": { "type": "integer" },
    "state_dimension": { "type": "integer" },
    "cuts_per_stage": {
      "type": "array",
      "items": { "type": "integer" },
      "description": "Number of cuts at each stage"
    }
  },
  "required": ["stages", "total_cuts", "state_dimension", "cuts_per_stage"]
}
```

**Output Schema (stage detail mode -- `stage` argument provided):**

```json
{
  "type": "object",
  "properties": {
    "stage": { "type": "integer" },
    "cuts": { "type": "integer" },
    "state_dimension": { "type": "integer" },
    "intercepts": {
      "type": "array",
      "items": { "type": "number" },
      "description": "Cut intercept values (alpha)"
    },
    "coefficients_sample": {
      "type": "array",
      "items": {
        "type": "array",
        "items": { "type": "number" }
      },
      "description": "First 10 cuts' coefficient vectors (beta), for inspection"
    },
    "coefficients_truncated": {
      "type": "boolean",
      "description": "True if there are more than 10 cuts"
    }
  },
  "required": ["stage", "cuts", "state_dimension", "intercepts"]
}
```

**Error Conditions:**

| Error Kind        | Condition                                                 |
| ----------------- | --------------------------------------------------------- |
| `OutputNotFound`  | `policy_dir` does not exist or contains no policy files   |
| `OutputCorrupted` | Policy FlatBuffers files exist but cannot be deserialized |

**Timeout:** Default 10 seconds.

**Example Request:**

```json
{
  "method": "tools/call",
  "params": {
    "name": "cobre/inspect-policy",
    "arguments": {
      "policy_dir": "/data/case_001/output/policy"
    }
  }
}
```

**Example Response:**

```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"stages\":120,\"total_cuts\":10440,\"state_dimension\":164,\"cuts_per_stage\":[87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87]}"
    }
  ],
  "isError": false
}
```

### 2.6 `cobre/compare-policies`

Compares two or more policy directories to assess quality differences. Returns convergence and structural metrics for cross-study comparison.

**Input Schema:**

```json
{
  "type": "object",
  "properties": {
    "policy_dirs": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 2,
      "description": "Absolute paths to policy directories to compare"
    },
    "metrics": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["convergence", "cut_counts", "state_dimension", "all"]
      },
      "default": ["all"],
      "description": "Metrics to compare. Defaults to all"
    }
  },
  "required": ["policy_dirs"],
  "additionalProperties": false
}
```

**Output Schema:**

```json
{
  "type": "object",
  "properties": {
    "policies": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "policy_dir": { "type": "string" },
          "stages": { "type": "integer" },
          "total_cuts": { "type": "integer" },
          "state_dimension": { "type": "integer" },
          "final_lower_bound": { "type": ["number", "null"] },
          "final_upper_bound": { "type": ["number", "null"] },
          "final_gap": { "type": ["number", "null"] },
          "iterations": { "type": ["integer", "null"] }
        },
        "required": ["policy_dir", "stages", "total_cuts", "state_dimension"]
      }
    },
    "compatible": {
      "type": "boolean",
      "description": "True if all policies have the same stage count and state dimension"
    },
    "deltas": {
      "type": ["object", "null"],
      "description": "Pairwise delta metrics (only present when exactly 2 policies are compared and compatible)",
      "properties": {
        "lb_delta": { "type": "number" },
        "lb_delta_pct": { "type": "number" },
        "ub_delta": { "type": "number" },
        "ub_delta_pct": { "type": "number" },
        "cut_count_delta": { "type": "integer" }
      }
    }
  },
  "required": ["policies", "compatible"]
}
```

**Error Conditions:**

| Error Kind         | Condition                                                                                       |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| `OutputNotFound`   | Any `policy_dir` does not exist                                                                 |
| `OutputCorrupted`  | Policy data in any directory cannot be read                                                     |
| `IncompatibleRuns` | Policies have different stage counts or state dimensions (reported in output, not as MCP error) |

**Timeout:** Default 30 seconds.

**Example Request:**

```json
{
  "method": "tools/call",
  "params": {
    "name": "cobre/compare-policies",
    "arguments": {
      "policy_dirs": [
        "/data/case_001/output_v1/policy",
        "/data/case_001/output_v2/policy"
      ],
      "metrics": ["all"]
    }
  }
}
```

**Example Response:**

```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"policies\":[{\"policy_dir\":\"/data/case_001/output_v1/policy\",\"stages\":120,\"total_cuts\":10440,\"state_dimension\":164,\"final_lower_bound\":72105.4,\"final_upper_bound\":73211.8,\"final_gap\":0.0151,\"iterations\":87},{\"policy_dir\":\"/data/case_001/output_v2/policy\",\"stages\":120,\"total_cuts\":11040,\"state_dimension\":164,\"final_lower_bound\":72201.1,\"final_upper_bound\":73150.2,\"final_gap\":0.013,\"iterations\":92}],\"compatible\":true,\"deltas\":{\"lb_delta\":95.7,\"lb_delta_pct\":0.0013,\"ub_delta\":-61.6,\"ub_delta_pct\":-0.0008,\"cut_count_delta\":600}}"
    }
  ],
  "isError": false
}
```

### 2.7 `cobre/inspect-case`

Read-only case introspection: loads and summarizes case metadata without running the full validation pipeline. Useful for agents to understand the case structure before deciding what operations to perform.

**Input Schema:**

```json
{
  "type": "object",
  "properties": {
    "case_dir": {
      "type": "string",
      "description": "Absolute path to the case directory"
    }
  },
  "required": ["case_dir"],
  "additionalProperties": false
}
```

**Output Schema:**

```json
{
  "type": "object",
  "properties": {
    "config": {
      "type": "object",
      "description": "Parsed config.json contents"
    },
    "entities": {
      "type": "object",
      "properties": {
        "hydros": { "type": "integer" },
        "thermals": { "type": "integer" },
        "buses": { "type": "integer" },
        "lines": { "type": "integer" }
      },
      "required": ["hydros", "thermals", "buses", "lines"]
    },
    "stages": { "type": "integer" },
    "scenario_source": {
      "type": "string",
      "enum": ["par", "external"],
      "description": "Whether scenarios are generated from PAR model or loaded from external file"
    },
    "files_present": {
      "type": "array",
      "items": { "type": "string" },
      "description": "List of input files found in the case directory"
    }
  },
  "required": ["config", "entities", "stages"]
}
```

**Error Conditions:**

| Error Kind    | Condition                                           |
| ------------- | --------------------------------------------------- |
| `MissingFile` | `case_dir` does not exist or lacks `config.json`    |
| `ParseError`  | `config.json` or entity registries cannot be parsed |

**Timeout:** Default 10 seconds.

**Example Request:**

```json
{
  "method": "tools/call",
  "params": {
    "name": "cobre/inspect-case",
    "arguments": {
      "case_dir": "/data/case_001"
    }
  }
}
```

**Example Response:**

```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"config\":{\"stages\":120,\"training\":{\"enabled\":true},\"simulation\":{\"enabled\":true,\"num_scenarios\":2000}},\"entities\":{\"hydros\":164,\"thermals\":82,\"buses\":5,\"lines\":8},\"stages\":120,\"scenario_source\":\"par\",\"files_present\":[\"config.json\",\"system/hydros.json\",\"system/thermals.json\",\"system/buses.json\",\"system/lines.json\",\"scenarios/inflow_seasonal_stats.parquet\",\"scenarios/inflow_ar_coefficients.parquet\"]}"
    }
  ],
  "isError": false
}
```

### 2.8 `cobre/list-scenarios`

Returns scenario metadata (source type, count, coverage) without loading the full scenario data into memory.

**Input Schema:**

```json
{
  "type": "object",
  "properties": {
    "case_dir": {
      "type": "string",
      "description": "Absolute path to the case directory"
    }
  },
  "required": ["case_dir"],
  "additionalProperties": false
}
```

**Output Schema:**

```json
{
  "type": "object",
  "properties": {
    "scenario_source": {
      "type": "string",
      "enum": ["par", "external"],
      "description": "How scenarios are generated"
    },
    "count": {
      "type": "integer",
      "description": "Number of scenarios (from config or external file)"
    },
    "stages": {
      "type": "integer",
      "description": "Number of stages"
    },
    "entities": {
      "type": "integer",
      "description": "Number of stochastic entities (hydros with inflow model)"
    },
    "par_model": {
      "type": ["object", "null"],
      "description": "PAR model summary (present when scenario_source is 'par')",
      "properties": {
        "max_ar_order": { "type": "integer" },
        "seasons": { "type": "integer" },
        "correlation_groups": { "type": "integer" }
      }
    }
  },
  "required": ["scenario_source", "count", "stages", "entities"]
}
```

**Error Conditions:**

| Error Kind    | Condition                                         |
| ------------- | ------------------------------------------------- |
| `MissingFile` | `case_dir` does not exist or lacks scenario files |
| `ParseError`  | Scenario files cannot be parsed                   |

**Timeout:** Default 10 seconds.

**Example Request:**

```json
{
  "method": "tools/call",
  "params": {
    "name": "cobre/list-scenarios",
    "arguments": {
      "case_dir": "/data/case_001"
    }
  }
}
```

**Example Response:**

```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"scenario_source\":\"par\",\"count\":200,\"stages\":120,\"entities\":164,\"par_model\":{\"max_ar_order\":6,\"seasons\":12,\"correlation_groups\":5}}"
    }
  ],
  "isError": false
}
```

### 2.9 `cobre/export-results`

Converts Parquet simulation results to an agent-friendly format (CSV, JSON, or Arrow IPC). This is a **write operation**: it creates a new file in the output directory.

**Input Schema:**

```json
{
  "type": "object",
  "properties": {
    "output_dir": {
      "type": "string",
      "description": "Absolute path to the output directory"
    },
    "format": {
      "type": "string",
      "enum": ["csv", "json", "arrow"],
      "description": "Target export format"
    },
    "entity_type": {
      "type": "string",
      "enum": ["hydros", "thermals", "buses", "lines", "costs", "convergence"],
      "description": "Entity type to export"
    },
    "filters": {
      "type": "object",
      "description": "Optional filters (same schema as cobre/query-results filters)",
      "properties": {
        "entity_id": { "type": ["string", "array"] },
        "stage_range": {
          "type": "array",
          "items": { "type": "integer" },
          "minItems": 2,
          "maxItems": 2
        },
        "scenario_range": {
          "type": "array",
          "items": { "type": "integer" },
          "minItems": 2,
          "maxItems": 2
        }
      },
      "additionalProperties": false
    }
  },
  "required": ["output_dir", "format", "entity_type"],
  "additionalProperties": false
}
```

**Output Schema:**

```json
{
  "type": "object",
  "properties": {
    "file_path": {
      "type": "string",
      "description": "Absolute path to the exported file"
    },
    "rows": {
      "type": "integer",
      "description": "Number of rows exported"
    },
    "size_bytes": {
      "type": "integer",
      "description": "File size in bytes"
    }
  },
  "required": ["file_path", "rows", "size_bytes"]
}
```

**Error Conditions:**

| Error Kind        | Condition                                               |
| ----------------- | ------------------------------------------------------- |
| `OutputNotFound`  | `output_dir` does not exist or lacks target entity data |
| `OutputCorrupted` | Source Parquet files cannot be read                     |

> **Security**: This tool is classified as **read-write** (SS8.1). It requires `--allow-write` or equivalent server configuration to be enabled.

**Timeout:** Default 60 seconds.

**Example Request:**

```json
{
  "method": "tools/call",
  "params": {
    "name": "cobre/export-results",
    "arguments": {
      "output_dir": "/data/case_001/output",
      "format": "csv",
      "entity_type": "hydros",
      "filters": {
        "entity_id": "ITAIPU"
      }
    }
  }
}
```

**Example Response:**

```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"file_path\":\"/data/case_001/output/exports/hydros_ITAIPU.csv\",\"rows\":240000,\"size_bytes\":18450000}"
    }
  ],
  "isError": false
}
```

### 2.10 `cobre/get-config-schema`

Returns the JSON Schema for `config.json`, enabling agents to generate or validate configuration files programmatically.

**Input Schema:**

```json
{
  "type": "object",
  "properties": {},
  "additionalProperties": false
}
```

**Output Schema:**

```json
{
  "type": "object",
  "properties": {
    "schema": {
      "type": "object",
      "description": "Complete JSON Schema for config.json"
    }
  },
  "required": ["schema"]
}
```

**Error Conditions:** None. This tool always succeeds.

**Timeout:** Default 5 seconds.

**Example Request:**

```json
{
  "method": "tools/call",
  "params": {
    "name": "cobre/get-config-schema",
    "arguments": {}
  }
}
```

**Example Response:**

```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"schema\":{\"$schema\":\"https://json-schema.org/draft/2020-12/schema\",\"type\":\"object\",\"properties\":{\"stages\":{\"type\":\"integer\",\"minimum\":1},\"training\":{\"type\":\"object\",\"properties\":{\"enabled\":{\"type\":\"boolean\"}}},\"simulation\":{\"type\":\"object\",\"properties\":{\"enabled\":{\"type\":\"boolean\"},\"num_scenarios\":{\"type\":\"integer\",\"minimum\":1}}}}}}"
    }
  ],
  "isError": false
}
```

## 3. Resource Definitions

MCP resources provide read-only access to Cobre data artifacts via URI templates. All resources return `application/json` MIME type. Resources that read from Parquet or FlatBuffers perform server-side conversion to JSON according to the rules in SS3.2.

### 3.1 Resource Inventory

| #   | Resource URI Template                               | Data Source              | MIME Type          | Access Mode |
| --- | --------------------------------------------------- | ------------------------ | ------------------ | ----------- |
| 1   | `cobre://case/{case_dir}/config`                    | `config.json`            | `application/json` | Read-only   |
| 2   | `cobre://case/{case_dir}/entities/{entity_type}`    | System JSON registries   | `application/json` | Read-only   |
| 3   | `cobre://case/{case_dir}/scenarios/{entity_id}`     | Scenario Parquet files   | `application/json` | Read-only   |
| 4   | `cobre://output/{output_dir}/convergence`           | `convergence.parquet`    | `application/json` | Read-only   |
| 5   | `cobre://output/{output_dir}/results/{entity_type}` | Hive-partitioned Parquet | `application/json` | Read-only   |
| 6   | `cobre://output/{output_dir}/policy/summary`        | FlatBuffers policy dir   | `application/json` | Read-only   |

**URI template variables:**

| Variable      | Type   | Description                                                                                                                 |
| ------------- | ------ | --------------------------------------------------------------------------------------------------------------------------- |
| `case_dir`    | string | URL-encoded absolute path to the case directory                                                                             |
| `output_dir`  | string | URL-encoded absolute path to the output directory                                                                           |
| `entity_type` | string | Entity type identifier: `hydros`, `thermals`, `buses`, `lines`, `contracts`, `pumping_stations`, `non_controllable_sources` |
| `entity_id`   | string | Entity identifier (e.g., `ITAIPU`)                                                                                          |

### 3.2 Parquet-to-JSON Conversion Rules

When a resource reads from Parquet files, column values are converted to JSON types according to the following mapping.

| Parquet Physical Type | Parquet Logical Type | JSON Type | Notes                                     |
| --------------------- | -------------------- | --------- | ----------------------------------------- |
| `INT32`               | (none or INT)        | `number`  | Integer values preserved as JSON integers |
| `INT64`               | (none or INT)        | `number`  | Integer values preserved as JSON integers |
| `FLOAT`               | (none)               | `number`  | Floating-point precision may reduce       |
| `DOUBLE`              | (none)               | `number`  | Full double precision preserved           |
| `BYTE_ARRAY`          | `UTF8`               | `string`  | UTF-8 string                              |
| `BOOLEAN`             | (none)               | `boolean` | `true` or `false`                         |
| `INT32`               | `DATE`               | `string`  | ISO 8601 date string                      |
| `INT64`               | `TIMESTAMP_MILLIS`   | `string`  | ISO 8601 datetime string                  |

**Null handling:** Parquet null values are represented as JSON `null`.

**Nested structures:** Parquet nested columns (groups, repeated fields) are flattened to top-level JSON keys with dot-separated paths. For example, a Parquet column `timing.forward_ms` becomes a JSON field `"timing.forward_ms"`.

**Row limit:** Resource responses for large Parquet files are limited to 10,000 rows. If the data exceeds this limit, the response includes a `"truncated": true` field and a `"total_rows"` count. Agents should use `cobre/query-results` with filters for targeted access to large datasets.

### 3.3 FlatBuffers-to-JSON Conversion Rules

The policy summary resource (`cobre://output/{output_dir}/policy/summary`) reads FlatBuffers policy data and returns a JSON summary. The conversion produces the same schema as `cobre/inspect-policy` (SS2.5) in summary mode:

```json
{
  "stages": 120,
  "total_cuts": 10440,
  "state_dimension": 164,
  "cuts_per_stage": [87, 87, ...]
}
```

FlatBuffers internals (byte offsets, vtable structure, raw binary data) are never exposed through resources.

### 3.4 Resource Error Handling

When a resource cannot be served, the MCP server returns an error response with the appropriate error kind:

| Error Kind        | Condition                                                 |
| ----------------- | --------------------------------------------------------- |
| `OutputNotFound`  | The path in the URI does not exist on disk                |
| `OutputCorrupted` | The data file exists but cannot be read or converted      |
| `MissingFile`     | A case directory resource references a missing input file |

## 4. Prompt Definitions

Prompts provide guided interaction patterns for AI agents. Each prompt generates a structured message sequence that helps the agent perform complex Cobre workflows.

### 4.1 Prompt Inventory

| #   | Prompt Name                  | Arguments                                                     | Purpose                                                                                    |
| --- | ---------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 1   | `cobre/setup-study`          | `{ description: string }`                                     | Guide agent through case directory creation: config.json, entity registries, scenario data |
| 2   | `cobre/diagnose-convergence` | `{ output_dir: string }`                                      | Analyze convergence.parquet, identify stalling, suggest parameter adjustments              |
| 3   | `cobre/compare-runs`         | `{ output_dirs: string[] }`                                   | Compare multiple training/simulation runs for policy quality assessment                    |
| 4   | `cobre/explain-results`      | `{ output_dir: string, entity_type: string, metric: string }` | Explain simulation results in domain terms (water value, marginal cost, deficit patterns)  |

### 4.2 `cobre/setup-study`

**Argument Schema:**

```json
{
  "type": "object",
  "properties": {
    "description": {
      "type": "string",
      "description": "Natural language description of the study the agent should set up"
    }
  },
  "required": ["description"]
}
```

**Purpose:** Generates a message sequence that walks the agent through:

1. Creating the case directory structure
2. Populating `config.json` using the schema from `cobre/get-config-schema`
3. Creating entity registries (`hydros.json`, `thermals.json`, `buses.json`, `lines.json`)
4. Providing scenario data (PAR model parameters or external scenarios)
5. Running `cobre/validate` to verify the case

**Expected Agent Workflow:** The agent receives the prompt response, then iteratively calls `cobre/get-config-schema`, creates files, and calls `cobre/validate` until the case is valid.

### 4.3 `cobre/diagnose-convergence`

**Argument Schema:**

```json
{
  "type": "object",
  "properties": {
    "output_dir": {
      "type": "string",
      "description": "Absolute path to the output directory of a completed training run"
    }
  },
  "required": ["output_dir"]
}
```

**Purpose:** Generates a message sequence that guides the agent through convergence analysis:

1. Load convergence history via `cobre/query-convergence`
2. Identify stalling patterns (lower bound plateau, upper bound volatility)
3. Examine the gap trajectory and termination reason
4. Suggest parameter adjustments (stopping rule tolerances, number of forward scenarios, risk measure parameters)

**Expected Agent Workflow:** The agent calls `cobre/query-convergence`, analyzes the iteration history, and recommends configuration changes to improve convergence behavior.

### 4.4 `cobre/compare-runs`

**Argument Schema:**

```json
{
  "type": "object",
  "properties": {
    "output_dirs": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 2,
      "description": "Absolute paths to output directories to compare"
    }
  },
  "required": ["output_dirs"]
}
```

**Purpose:** Generates a message sequence for multi-run comparison:

1. Inspect each policy via `cobre/inspect-policy`
2. Compare convergence histories via `cobre/query-convergence`
3. Compare simulation results via `cobre/query-results`
4. Summarize differences in policy quality, convergence speed, and cost metrics

**Expected Agent Workflow:** The agent calls the comparison and query tools, builds a comparative analysis, and recommends which policy is preferable and why.

### 4.5 `cobre/explain-results`

**Argument Schema:**

```json
{
  "type": "object",
  "properties": {
    "output_dir": {
      "type": "string",
      "description": "Absolute path to the output directory"
    },
    "entity_type": {
      "type": "string",
      "enum": ["hydros", "thermals", "buses", "costs"],
      "description": "Entity type to explain"
    },
    "metric": {
      "type": "string",
      "description": "Specific metric to focus on (e.g., 'storage', 'generation', 'marginal_cost', 'deficit')"
    }
  },
  "required": ["output_dir", "entity_type", "metric"]
}
```

**Purpose:** Generates a message sequence for domain-specific result interpretation:

1. Load relevant simulation results via `cobre/query-results`
2. Compute summary statistics (mean, std, percentiles across scenarios)
3. Interpret results in hydrothermal dispatch terms (water value, thermal dispatch order, deficit patterns, reservoir operation)
4. Identify anomalies or notable patterns

**Expected Agent Workflow:** The agent calls `cobre/query-results` with appropriate filters, computes statistics, and produces a domain-informed explanation of the results.

## 5. Transport

### 5.1 stdio Transport (Local)

The primary transport for local agent interaction (Claude Desktop, agent CLI tools).

**Connection lifecycle:**

1. Agent spawns `cobre-mcp` (or `cobre serve --transport stdio`) as a child process
2. Agent writes JSON-RPC messages to the server's stdin
3. Server writes JSON-RPC responses to stdout
4. Server writes diagnostic logs to stderr (not part of the MCP protocol)
5. Connection terminates when the agent closes stdin or the server process exits

**Configuration:** Command-line flags or environment variables configure the server:

| Flag / Environment Variable | Description                                           | Default           |
| --------------------------- | ----------------------------------------------------- | ----------------- |
| `--transport stdio`         | Use stdio transport                                   | (default)         |
| `--allow-write`             | Enable read-write tools (`run`, `export-results`)     | Disabled          |
| `--allowed-dirs <paths>`    | Comma-separated allowlist of case/output directories  | Current directory |
| `--threads <n>`             | Number of Rayon threads for computation               | Auto-detect       |
| `COBRE_MCP_ALLOWED_DIRS`    | Environment variable alternative for `--allowed-dirs` | (none)            |

### 5.2 Streamable HTTP Transport (Remote)

For remote agent access, the server supports HTTP-based transport with SSE (Server-Sent Events) for streaming.

**Connection lifecycle:**

1. Agent sends HTTP POST to the server's MCP endpoint (e.g., `http://host:port/mcp`)
2. Server responds with SSE stream for notifications (progress events) and final responses
3. Connection uses standard HTTP keep-alive for session continuity
4. Multiple concurrent tool calls are supported via distinct request IDs

**Configuration:**

| Flag / Environment Variable | Description                          | Default           |
| --------------------------- | ------------------------------------ | ----------------- |
| `--transport http`          | Use streamable HTTP transport        | (not default)     |
| `--host <addr>`             | Bind address                         | `127.0.0.1`       |
| `--port <port>`             | Listen port                          | `3000`            |
| `--allow-write`             | Enable read-write tools              | Disabled          |
| `--allowed-dirs <paths>`    | Allowlist of case/output directories | Current directory |

### 5.3 Transport Selection Rationale

| Deployment                          | Recommended Transport | Rationale                                                                             |
| ----------------------------------- | --------------------- | ------------------------------------------------------------------------------------- |
| Local (same machine as agent)       | stdio                 | Lowest latency, simplest setup, standard for Claude Desktop / agent CLI tools         |
| Remote (agent on different machine) | Streamable HTTP (SSE) | HTTP-based, firewall-friendly, supports progress streaming for long `cobre/run` calls |

## 6. Progress Reporting

### 6.1 Progress Token Lifecycle

The MCP progress reporting mechanism is used exclusively by `cobre/run` to report training and simulation progress. The lifecycle follows the MCP specification for progress notifications.

**Sequence:**

1. Agent includes `_meta.progressToken` in the `tools/call` request
2. Server begins training and emits `notifications/progress` after each iteration
3. Each notification contains the progress token, current iteration, and optional total
4. When training completes, the server returns the final `tools/call` result
5. The progress token becomes invalid after the result is returned

### 6.2 ConvergenceUpdate-to-Progress Mapping

The server subscribes to the `ConvergenceUpdate` event from the shared event stream (architecture-021 SS3.2). Each event is mapped to an MCP progress notification.

**Event source:** `ConvergenceUpdate` struct from `cobre-core` ([Convergence Monitoring](../architecture/convergence-monitoring.md) SS2.4):

| Event Field         | Type  | Description                        |
| ------------------- | ----- | ---------------------------------- |
| `iteration`         | `i32` | Iteration index (1-based)          |
| `lower_bound`       | `f64` | Current lower bound                |
| `upper_bound`       | `f64` | Current upper bound (mean)         |
| `upper_bound_std`   | `f64` | Standard deviation of upper bound  |
| `ci_95`             | `f64` | 95% confidence interval half-width |
| `gap`               | `f64` | Relative gap                       |
| `wall_time_ms`      | `i64` | Cumulative wall-clock time         |
| `iteration_time_ms` | `i64` | Time for this iteration            |

**MCP progress notification format:**

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/progress",
  "params": {
    "progressToken": "run-001",
    "progress": 42,
    "total": null,
    "metadata": {
      "iteration": 42,
      "lower_bound": 72050.1,
      "upper_bound": 73300.5,
      "gap": 0.0171,
      "wall_time_ms": 523400
    }
  }
}
```

**Field mapping:**

| MCP Progress Field      | Source                             | Notes                                                                                |
| ----------------------- | ---------------------------------- | ------------------------------------------------------------------------------------ |
| `progress`              | `ConvergenceUpdate.iteration`      | Current iteration number                                                             |
| `total`                 | `null` or config `iteration_limit` | Total is unknown in general (stopping may be by bound stalling, not iteration limit) |
| `metadata.iteration`    | `ConvergenceUpdate.iteration`      | Redundant with `progress` for agent convenience                                      |
| `metadata.lower_bound`  | `ConvergenceUpdate.lower_bound`    | Current lower bound value                                                            |
| `metadata.upper_bound`  | `ConvergenceUpdate.upper_bound`    | Current upper bound value                                                            |
| `metadata.gap`          | `ConvergenceUpdate.gap`            | Current relative gap                                                                 |
| `metadata.wall_time_ms` | `ConvergenceUpdate.wall_time_ms`   | Cumulative wall-clock time for agent to estimate remaining time                      |

### 6.3 Simulation Progress

During the simulation phase, the server maps `SimulationProgress` events to MCP progress notifications:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/progress",
  "params": {
    "progressToken": "run-001",
    "progress": 500,
    "total": 2000,
    "metadata": {
      "phase": "simulation",
      "scenarios_complete": 500,
      "scenarios_total": 2000,
      "elapsed_ms": 34200
    }
  }
}
```

### 6.4 Backpressure

The MCP server is one consumer of the shared `broadcast` event channel (architecture-021 SS3.4). If the MCP transport cannot keep up with event emission (e.g., slow network for HTTP transport), the broadcast channel drops old events rather than blocking the training loop. Progress notifications may skip iterations in this case. The agent must tolerate gaps in the iteration sequence.

## 7. Long-Running Operations

> **Design Note -- Flagged for User Review:** The long-running operation model for `cobre/run` is an open design decision (architecture-021 SS6.1 Q-3). This section presents the recommended approach and an alternative. The final decision should be confirmed before implementation.

### 7.1 Recommended Approach: Progress During Call

The `cobre/run` tool call blocks until training and/or simulation completes. During execution, the server emits MCP progress notifications (SS6) to keep the agent informed. The agent cannot issue other `cobre/run` calls while one is in progress (the server processes one training run at a time).

**Advantages:**

- Simplest implementation: single synchronous code path
- Standard MCP pattern: progress notifications are a first-class MCP feature
- No state management: no need to track background tasks or provide polling APIs
- Agent receives the result in the same call context

**Limitations:**

- Agent framework must support long-lived tool calls (minutes to hours)
- If the agent's MCP client has a request timeout, training may be interrupted
- Single concurrent training run per server instance

### 7.2 Alternative Approach: Background Task with Polling

An alternative design would split `cobre/run` into an asynchronous pattern:

1. `cobre/run` returns immediately with a `task_id`
2. A new tool `cobre/get-task-status` polls for progress and completion
3. A new tool `cobre/cancel-task` requests cancellation

**Advantages:**

- Decouples training lifetime from the MCP request/response cycle
- Supports agent frameworks with strict request timeouts
- Could support multiple concurrent training runs

**Disadvantages:**

- More complex implementation: task registry, lifecycle management, cleanup
- Requires two additional tools not in the current 10-tool inventory
- Agent must implement polling logic

### 7.3 Current Resolution

The recommended approach (progress during call) is adopted as the default. If the MCP protocol specification evolves to support background tasks natively, or if agent framework limitations require it, the alternative approach can be implemented as a non-breaking addition (new tools, existing tools unchanged).

## 8. Security Model

### 8.1 Operation Classification

| Category       | Tools                                                                                                                                                                                         | Policy                                                              |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **Read-only**  | `cobre/validate`, `cobre/query-results`, `cobre/query-convergence`, `cobre/inspect-policy`, `cobre/compare-policies`, `cobre/inspect-case`, `cobre/list-scenarios`, `cobre/get-config-schema` | Default mode. No filesystem writes. Safe for untrusted agent access |
| **Read-write** | `cobre/run`, `cobre/export-results`                                                                                                                                                           | Writes output files. Requires explicit opt-in via `--allow-write`   |

> **Default-safe principle:** A freshly started `cobre-mcp` server with no flags exposes only read-only operations. Write operations return an error unless `--allow-write` is explicitly provided.

### 8.2 Sandboxed File Access

All file access operations (both read and write) are sandboxed to a configured allowlist of directories. This prevents agents from accessing arbitrary filesystem paths.

**Allowlist configuration:**

- `--allowed-dirs /data/cases,/data/outputs` -- comma-separated list of allowed root directories
- `COBRE_MCP_ALLOWED_DIRS=/data/cases:/data/outputs` -- environment variable (colon-separated on Unix)
- Default: current working directory of the server process

**Path validation rules:**

1. All paths provided in tool arguments (`case_dir`, `output_dir`, `policy_dir`) are resolved to absolute canonical paths (symlinks resolved, `..` segments collapsed)
2. The canonical path must be a descendant of at least one directory in the allowlist
3. If validation fails, the tool returns an error with a message indicating the path is outside the allowed directories. The actual allowlist is **not** disclosed in error messages (information leak prevention)
4. Path traversal attempts (`../`, symlinks pointing outside the allowlist) are rejected after canonicalization

**Example rejection:**

```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"kind\":\"MissingFile\",\"message\":\"Path '/etc/passwd' is not within the allowed directories\",\"context\":{\"path\":\"/etc/passwd\"},\"suggestion\":\"Provide a path within the server's configured allowed directories.\"}"
    }
  ],
  "isError": true
}
```

### 8.3 Resource Sandboxing

Resources (SS3) follow the same sandboxing rules. URI template variables (`case_dir`, `output_dir`) are validated against the allowlist before any file I/O occurs.

### 8.4 Write Operation Opt-In

When `--allow-write` is not set:

- `cobre/run` returns an error: `"Write operations are disabled. Start the server with --allow-write to enable training and export."`
- `cobre/export-results` returns the same error
- All other tools function normally

When `--allow-write` is set:

- `cobre/run` writes output files to subdirectories of `case_dir` (the output directory is always within or adjacent to the case directory)
- `cobre/export-results` writes export files to subdirectories of `output_dir`
- Both tools still validate that the write target is within the allowlist

## 9. Capability Negotiation

### 9.1 MCP Initialize Handshake

The server responds to the MCP `initialize` request with its capabilities declaration.

**Server capabilities response:**

```json
{
  "protocolVersion": "2025-03-26",
  "capabilities": {
    "tools": {
      "listChanged": false
    },
    "resources": {
      "subscribe": false,
      "listChanged": false
    },
    "prompts": {
      "listChanged": false
    }
  },
  "serverInfo": {
    "name": "cobre-mcp",
    "version": "2.0.0"
  }
}
```

**Capability details:**

| Capability              | Value   | Rationale                                                                                   |
| ----------------------- | ------- | ------------------------------------------------------------------------------------------- |
| `tools.listChanged`     | `false` | Tool list is static at server startup; no dynamic tool registration                         |
| `resources.subscribe`   | `false` | Resources are read-on-demand; no push-based resource updates                                |
| `resources.listChanged` | `false` | Resource list is static; available resources depend on disk state, not server state changes |
| `prompts.listChanged`   | `false` | Prompt list is static                                                                       |

### 9.2 Protocol Version

The server declares compatibility with MCP protocol version `2025-03-26` (or the latest stable version at implementation time). If the client requests an incompatible protocol version, the server rejects the initialization with an error.

### 9.3 Tool Availability Based on Configuration

After initialization, the `tools/list` response reflects the server's security configuration:

- When `--allow-write` is **not** set: `tools/list` includes all 10 tools, but `cobre/run` and `cobre/export-results` include an `annotation` field noting that write operations are disabled
- When `--allow-write` is set: all 10 tools are fully available

This allows agents to discover the server's capabilities and adapt their workflow accordingly (e.g., skip training if write is disabled, focus on read-only analysis).

## Cross-References

- [Structured Output](structured-output.md) -- Response envelope schema (SS1), error record schema (SS2.1), error kind registry (SS2.3), JSON-lines streaming protocol (SS3) that defines the event types reused by MCP progress reporting
- [Convergence Monitoring](../architecture/convergence-monitoring.md) -- Per-iteration output record (SS2.4) that defines the `ConvergenceUpdate` event payload mapped to MCP progress notifications; training log format (SS4) that is the human-readable equivalent
- [Validation Architecture](../architecture/validation-architecture.md) -- 5-layer validation pipeline (SS2) invoked by `cobre/validate`; error type catalog (SS4) that maps to the error kind registry
- [Output Schemas](../data-model/output-schemas.md) (planned update) -- Parquet output schemas read by `cobre/query-results` and `cobre/export-results`; convergence.parquet schema read by `cobre/query-convergence`
- [Configuration Reference](../configuration/configuration-reference.md) -- `config.json` schema returned by `cobre/get-config-schema`; stopping rules that determine training termination reported in `cobre/run` output
- [Training Loop](../architecture/training-loop.md) -- Iteration lifecycle (SS2.1) that produces the events consumed by progress reporting; event emission points for the shared event stream
- Architecture Blueprint (architecture-021) -- Crate responsibility boundaries (SS2.1), shared event stream architecture (SS3), scope definition (SS5.2), long-running operation assumption (SS6.1 Q-3)
- [Python Bindings](./python-bindings.md) -- Shares the single-process execution path (SS1.2); multi-process architecture (SS7.4) and multi-process `train()` API pattern (SS2.1) may be adopted in a future release
- [Terminal UI](./terminal-ui.md) -- Shares the same event stream as another consumer of `ConvergenceUpdate` events
