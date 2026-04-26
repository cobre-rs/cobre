# Design Principles

## Purpose

This spec defines the foundational design principles governing the Cobre data model: format selection criteria, key design goals, the critical declaration order invariance requirement, and the reference to the LP subproblem formulation. All other data model and architecture specs build on these principles.

## 1. Format Selection Criteria

| Data Type                          | Recommended Format | Rationale                                                                         |
| ---------------------------------- | ------------------ | --------------------------------------------------------------------------------- |
| Configuration & Parameters         | JSON               | Human-readable, easily editable, small size                                       |
| Entity Registries                  | JSON               | Structured objects with relationships                                             |
| Time Series Data                   | Parquet            | Columnar, compressed, efficient for large data                                    |
| Policy Data (Cuts/States/Vertices) | FlatBuffers        | Zero-copy deserialization, cache-friendly dense arrays, in-memory during training |
| Simulation Results                 | Parquet            | High volume, per-entity indexing                                                  |
| Dictionaries/Metadata              | CSV                | Human-readable, small, universal                                                  |

## 2. Key Design Goals

1. **Separation of Concerns**: Static system data vs. dynamic algorithm data vs. stochastic data
2. **Scalability**: File formats that scale to production sizes without memory explosion
3. **Reproducibility**: All inputs deterministically produce same outputs
4. **Warm-start Support**: Efficient serialization/deserialization of algorithm state
5. **Distributed I/O**: Rank 0 loads, broadcasts to workers (or parallel loading where beneficial)
6. **Declaration Order Invariance**: Results must not depend on the order entities are declared in input files
7. **Agent-Readability**: Every operation produces structured, machine-parseable output with stable schemas, enabling AI agents and programmatic tools to compose, monitor, and verify Cobre workflows without human intervention

## 3. Declaration Order Invariance (Critical Requirement)

> **⚠️ CRITICAL**: The optimization results MUST be identical regardless of the order in which entities are declared in input files. This is a fundamental correctness requirement.

**Principle**: If a user declares hydros A and B, runs the program, then exchanges the declaration order (B before A), the numerical results must be **bit-for-bit identical** (given the same random seed and same IDs).

**What determines identity**: The **entity ID** is the sole identifier. Two runs are equivalent if:

- All entity IDs are the same
- All entity properties are the same
- All relationships (by ID) are the same
- The random seed is the same

**What must NOT affect results**:

- Order of entities in JSON arrays (`hydros`, `thermals`, `buses`, `lines`)
- Order of rows in Parquet tables
- Order of constraints in `generic_constraints.json`
- Order of stages in `stages.json` (sorted by ID internally)
- Order of blocks within a stage (sorted by ID internally)
- Order of correlation blocks or entities within correlation blocks

### 3.1 Implementation Requirements

1. **Canonical Ordering**: After loading, all entity collections must be sorted by ID before any processing
2. **Deterministic Iteration**: All iterations over entities must use the canonical (sorted by ID) order
3. **LP Variable Ordering**: LP variables must be created in canonical order (by entity ID, then by block ID)
4. **LP Constraint Ordering**: LP constraints must be added in canonical order
5. **Random Number Generation**: Scenario generation must iterate entities in canonical order
6. **Cut Coefficients**: Cut coefficient ordering must follow the canonical state variable order

> **Note**: During implementation, one can assume that all the inputs are mostly sorted, so use sorting algorithms that have grater performance if all the inputs are already sorted by ID, ideally returning in few cycles if the data is already sorted by ID.

### 3.2 Validation Requirements

The test suite must include order-invariance tests that:

1. Run the same case with entities in different declaration orders
2. Verify bit-for-bit identical results (costs, decisions, cuts)

### 3.3 Canonical Ordering

After loading, a canonicalization step sorts every entity collection (buses, lines, hydros, thermals, pumping stations, contracts, non-controllable sources, generic constraints) by ID. All subsequent processing — LP variable layout, constraint construction, scenario generation, cut coefficient ordering — iterates in this canonical order. See [Input Loading Pipeline §3](../architecture/input-loading-pipeline.md) for the loading and canonicalization sequence.

### 3.4 Why This Matters

- **Debugging**: Users can reorganize input files without worrying about result changes
- **Version Control**: Reordering entities for readability doesn't create spurious diffs in results
- **Correctness**: Non-deterministic behavior from ordering is a bug, not a feature
- **Parallelism**: MPI ranks must agree on ordering without communication

## 4. LP Subproblem Formulation Reference

The mathematical specifications are organized across the [Mathematical Formulations](../math/) spec category. The data model specs focus on data structures and file formats; the math specs define what the solver computes.

**Data Model → Math Spec Mapping**:

| Data Model Input                                                        | Math Spec                                                     | Description                     |
| ----------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------- |
| hydros.json → generation_model                                          | [Hydro Production Models](../math/hydro-production-models.md) | Constant productivity, FPHA     |
| hydros.json → fpha\_\*                                                  | [Hydro Production Models](../math/hydro-production-models.md) | FPHA hyperplane coefficients    |
| stages.json → stages[].block_mode                                       | [Block Formulations](../math/block-formulations.md)           | Per-stage block formulation     |
| scenarios/inflow_seasonal_stats.parquet, inflow_ar_coefficients.parquet | [PAR Inflow Model](../math/par-inflow-model.md)               | PAR(p) model parameters         |
| config.json → inflow_non_negativity                                     | [Inflow Non-Negativity](../math/inflow-nonnegativity.md)      | Inflow treatment method         |
| stages.json → policy_graph.annual_discount_rate                         | [Discount Rate](../math/discount-rate.md)                     | Discount rate, per-stage factor |
| policy/cuts/                                                            | [Cut Management](../math/cut-management.md)                   | Cut coefficients and selection  |
| stages.json → stages[].risk_measure                                     | [Risk Measures](../math/risk-measures.md)                     | CVaR, per-stage risk profiles   |
| Full LP structure                                                       | [LP Formulation](../math/lp-formulation.md)                   | Objective, constraints, duals   |
| Algorithm flow                                                          | [SDDP Algorithm](../math/sddp-algorithm.md)                   | Forward/backward passes, cuts   |

**Variable/Constraint Sizing**: See [Production Scale Reference](./production-scale-reference.md) for production-scale LP dimensions.

## 5. Implementation Language & FFI Strategy

### 5.1 Decision

Cobre is implemented in **Rust** with `unsafe` FFI boundaries for LP solver interaction, MPI communication (ferrompi), and OpenMP thread management.

This decision was made with full awareness of the trade-offs. The discussion below documents the arguments for and against, so that future contributors understand why Rust was chosen and where the friction points are.

### 5.2 The Core Tension

Cobre is a system whose performance-critical inner loop calls C/C++ LP solver libraries. The natural question is: should the host language be C++ (same ecosystem as the solvers) or Rust (stronger safety guarantees for the parallel orchestration around the solvers)?

The concern is not theoretical. LP solver C++ APIs offer capabilities that their C APIs do not:

| Capability              | C++ API                                                                     | C API                                          |
| ----------------------- | --------------------------------------------------------------------------- | ---------------------------------------------- |
| Model cloning           | Copy constructors (efficient deep copy of solver state)                     | Not available — must rebuild or extract/reload |
| Hot-start               | `markHotStart`/`solveFromHotStart` (CLP)                                    | Not available                                  |
| Change tracking         | `whatsChanged_` bitflags — solver skips unnecessary re-initialization (CLP) | Not available                                  |
| Mutable internal access | Direct struct access, placement new, custom allocators                      | Through accessor functions                     |
| Factorization control   | Fine-grained control over when to refactor                                  | Implicit only                                  |

See [Solver HiGHS Implementation](../architecture/solver-highs-impl.md) and [Solver CLP Implementation](../architecture/solver-clp-impl.md) for the solver-specific C API usage.

### 5.3 What Rust Can and Cannot Do

> **Decision [DEC-013](./decision-log.md#dec-013) (active):** C API only for solver integration; Cobre does not use C++ solver APIs to maintain Rust FFI compatibility and cross-solver portability.

**Rust `unsafe` provides the same raw memory primitives as C/C++.** There is no performance penalty from choosing Rust — `unsafe` blocks generate identical machine code:

- `std::ptr::copy_nonoverlapping` = `memcpy`
- `std::ptr::copy` = `memmove`
- `std::alloc::alloc` / `dealloc` = `malloc` / `free`
- Raw pointer arithmetic, casting, reinterpretation — all available
- `#[repr(C)]` structs have identical layout to C structs

The limitation is not about what Rust _can_ do, but about **which solver APIs are accessible**. Rust calls C functions naturally via `extern "C"`. Calling C++ APIs requires either:

1. A thin C wrapper that exposes the C++ functionality through C-compatible functions
2. The `cxx` crate (limited to what it supports)
3. Direct use of the C API only (losing C++-exclusive features)

Cobre uses approach (3) — C APIs only — for solver portability. This means C++-exclusive features like model cloning, hot-start, and change tracking are not available through the standard solver abstraction. See [Solver Abstraction §10](../architecture/solver-abstraction.md) for the compile-time solver selection design.

### 5.4 Why This Does Not Block Performance

The adopted LP construction strategy is **Strategy 2+3: StageLpCache** (see [Solver Abstraction SS11.4](../architecture/solver-abstraction.md)). Each thread owns one solver instance and loads a complete pre-assembled LP per stage via a single bulk API call. The per-solve workflow is:

1. Load `StageLpCache[t]` via `passModel`/`loadProblem` — a single bulk copy of the complete LP including structural constraints and active Benders cuts in CSC format
2. Patch ~2,240 scenario-dependent values (incoming storage RHS, AR lag fixing, current-stage noise fixing) via bound modification
3. Load basis from previous iteration for warm-start
4. Solve (simplex pivots dominate runtime)

Steps 1-3 are cheap compared to step 4. The StageLpCache is shared across all ranks on the same node via `SharedRegion<T>`, which reduces per-node memory from ~91.8 GB (the previous per-thread rebuild approach, which required per-thread CSR assembly buffers) to ~22.3 GB node-wide. The C++-exclusive model clone pattern is not needed: the StageLpCache already provides a complete solver-ready LP via a single `passModel` bulk read at ~8.6 ms per stage transition at production scale.

**What we store and restore is solver-agnostic.** The data persisted across iterations — basis status arrays, column values and bounds, row values and bounds, primal and dual solutions — are standard optimization framework concepts, not solver-internal structures. We deliberately avoid storing solver-internal factorization structures (LU decomposition, pivot sequences, working memory), which are solver-specific and non-portable. This keeps the warm-start mechanism clean across solver backends.

### 5.5 Enlarged Unsafe Boundary

> **Decision [DEC-014](./decision-log.md#dec-014) (active):** Enlarged `unsafe` boundary: all performance-critical memory operations interacting with solver data are `unsafe`, not just FFI call sites.

The `unsafe` boundary in Cobre is **not limited to FFI function calls**. Performance-critical memory operations that would benefit from bypassing Rust's borrow checker are explicitly permitted within `unsafe` blocks:

- **Bulk memory copies** of LP data (coefficient arrays, bound vectors, solution vectors) using raw pointer operations when the safe alternative would require unnecessary allocations or copies
- **Pre-allocated buffer reuse** with raw pointer writes into thread-local scratch space, avoiding repeated allocation/deallocation in the solve loop
- **Direct manipulation of solver-owned memory** when the C API returns mutable pointers (e.g., CLP's mutable `double*` for row/column bounds), writing values in-place without intermediate buffers
- **Zero-copy views** into contiguous data structures (cut pool, scenario buffers) where creating safe slices would require lifetime gymnastics that add no real safety value
- **NUMA-aware allocation** with explicit placement and first-touch initialization patterns that require raw pointer manipulation

The principle is: **the `unsafe` boundary encloses all performance-critical memory operations that interact with solver data, not just the FFI call sites.** This includes preparing data before solver calls and extracting results after them. The safe Rust boundary sits above this — at the level of the training loop, scenario pipeline orchestration, and checkpoint/resume logic, where the parallelism and state management complexity actually lives.

This approach gives Cobre the memory manipulation efficiency of C++ in the solver-adjacent code while retaining compile-time safety guarantees in the ~80% of code that orchestrates the algorithm.

### 5.6 Arguments For and Against

**Why Rust holds for this project:**

- **Parallel correctness.** SDDP with MPI + threads is one of the hardest patterns to get right. Data races in cut aggregation, shared FCF access across MPI shared-memory windows, and scenario buffer management are catastrophic bugs that are nearly impossible to reproduce. Rust catches these at compile time in non-FFI code. In C++, a data race in the cut pool can silently produce wrong cuts, and the only symptom is degraded convergence — weeks to diagnose.
- **The solver interaction is contained.** The `unsafe` FFI + memory manipulation code is concentrated in a single module (~1,000-2,000 lines). The rest of the system — scenario pipeline, training loop, checkpoint/resume, MPI orchestration, input loading, validation — benefits from safety.
- **`unsafe` gives the same power as C++.** Raw memory operations, pointer arithmetic, and direct buffer manipulation are all available. The difference is that these operations are explicitly marked and auditable, not silently mixed with safe code.
- **The C API gap is manageable.** The modify-in-place + warm-start pattern works well through C APIs. The missing C++ features (cloning, hot-start, change tracking) provide incremental improvements, not fundamental capabilities.

**Where C++ would have a genuine advantage:**

- **Solver ecosystem is C++ native.** HiGHS, CLP, CPLEX, Gurobi all have C++ as their primary API. Bug reports, performance tuning, and internal documentation assume C++. Staying in the solver's language eliminates an entire class of friction.
- **No FFI translation layer.** Every solver call in Rust goes through an `extern "C"` boundary. In C++, you call solver methods directly. While the overhead per call is negligible, the ergonomic cost of maintaining bindings, handling error codes instead of exceptions, and wrapping opaque pointers adds friction throughout development.
- **C++-exclusive solver features.** If a future solver backend would benefit significantly from copy constructors, hot-start, or change tracking, accessing these from Rust requires writing and maintaining a C shim — additional code that must track upstream solver API changes.

### 5.7 Revisiting This Decision

This decision should be revisited if:

1. **Profiling shows that LP construction time (steps 1-3) becomes a significant fraction of total runtime.** This would indicate that the StageLpCache stage transition cost is too high and an alternative approach would provide meaningful speedup.
2. **The `unsafe` boundary grows beyond ~15% of the codebase**, suggesting that the safety benefits of Rust are being eroded by the volume of unsafe code needed.
3. **A solver backend requires C++-exclusive features for correctness** (not just performance), making the C API insufficient.

None of these conditions are expected at current production scale, but they should be monitored as the system evolves.

## 6. Agent-Readability (Detailed)

> **Principle**: Every Cobre operation must be fully usable by a programmatic agent -- an AI coding assistant, a CI/CD pipeline, or a Python orchestration script -- without requiring human interpretation of output, error messages, or progress indicators. Agent-readability is a first-class design concern, not an afterthought.

Agent-readability does not replace human-readability. The default CLI output remains human-friendly text. Agent-readability means that a structured alternative exists for every output path, and that this alternative is governed by stable, documented schemas. An agent interacting with Cobre should never need to scrape text, guess at error causes, or poll files to detect completion.

### 6.1 Relationship to Existing Principles

Agent-readability complements and strengthens the existing design goals:

- **Reproducibility (Goal 3)** guarantees that given the same inputs, Cobre produces the same outputs. Agent-readability extends this to the _interface layer_: an agent that replays the same command with the same inputs receives the same structured JSON response, enabling automated verification and regression testing of solver behavior.

- **Scalability (Goal 2)** ensures data formats scale to production sizes. Agent-readability respects this by _not_ requiring JSON equivalents for large tabular outputs (simulation Parquet files remain Parquet -- agents consume them via Arrow/Polars). Structured output applies to control-plane data: progress events, error reports, metadata summaries, and convergence diagnostics.

- **Declaration Order Invariance (Goal 6)** ensures that agent-constructed inputs produce consistent results regardless of how the agent orders entities. This is critical because agents generate JSON programmatically and cannot be expected to match any particular human-preferred ordering.

- **Distributed I/O (Goal 5)** means that in MPI execution, only rank 0 produces structured output. Agents interact with a single output stream, never with per-rank fragments.

### 6.2 Design Rules

The following four rules govern agent-readability across all Cobre interfaces: CLI, MCP server, Python bindings, and TUI event stream.

#### Rule 1: Every CLI operation has a JSON schema for its output

Every CLI subcommand (`run`, `validate`, `report`, `compare`, `version`) produces a response that conforms to a documented JSON schema when invoked with `--output-format json` or `--output-format json-lines`.

**Rationale**: Agents parse command output programmatically. Without a schema, the agent must reverse-engineer the output format from examples, which is fragile and breaks on version updates. A documented schema enables code generation, static validation, and version compatibility checks.

**Cobre example**: The `validate` subcommand already produces structured JSON (see [Validation Architecture §5](../architecture/validation-architecture.md)). This existing pattern is generalized to all subcommands via a response envelope:

```json
{
  "$schema": "urn:cobre:response:v1",
  "command": "validate",
  "success": false,
  "exit_code": 3,
  "cobre_version": "2.0.0",
  "errors": [
    {
      "kind": "SchemaViolation",
      "file": "hydros.json",
      "message": "Unknown field 'turbin_capacity'",
      "context": { "entity_id": "ITAIPU", "field": "turbin_capacity" },
      "suggestion": "Did you mean 'turbine_capacity'?"
    }
  ],
  "warnings": [],
  "data": null
}
```

The envelope schema is the same shape for all subcommands; only the `data` and `errors` content varies. See [Structured Output](../interfaces/structured-output.md) for the complete schema definition.

#### Rule 2: Every error is structured

Every error Cobre produces -- whether from input validation, solver failure, MPI communication error, or signal interruption -- is emitted as a machine-parseable record with four required fields:

| Field        | Type   | Description                                                                    |
| ------------ | ------ | ------------------------------------------------------------------------------ |
| `kind`       | string | Stable error identifier (e.g., `SchemaViolation`, `SolverFailure`, `MpiError`) |
| `message`    | string | Human-readable description of the error                                        |
| `context`    | object | Structured data about the error location and cause                             |
| `suggestion` | string | Actionable remediation hint (may be null)                                      |

**Rationale**: Agents cannot interpret free-text error messages reliably. The `kind` field enables programmatic dispatch (retry logic, fallback strategies, error categorization). The `context` field provides the structured data an agent needs to construct a fix. The `suggestion` field gives the agent a starting point for remediation without requiring domain knowledge of SDDP.

**Cobre example**: A solver infeasibility during training produces:

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
  "suggestion": "Check penalty costs in penalty_system.json; large deficits may indicate missing generation capacity or overly tight constraints"
}
```

The 14 validation error kinds defined in [Validation Architecture §4](../architecture/validation-architecture.md) are the foundation. Runtime error kinds (`SolverFailure`, `MpiError`, `CheckpointFailed`, `OutputCorrupted`, `OutputNotFound`, `IncompatibleRuns`) extend this set. All error kinds are stable identifiers that follow semantic versioning -- removing or renaming a kind is a breaking change.

#### Rule 3: Every long-running operation supports progress streaming

Operations that take more than a few seconds (training, simulation) emit a stream of progress events as JSON-lines to stdout when `--output-format json-lines` is specified. Each line is a self-describing JSON object with a `type` field that identifies the event kind.

**Rationale**: Agents monitoring a training run that takes minutes to hours need real-time feedback to (a) confirm the process is alive, (b) track convergence, (c) detect stalling, and (d) decide whether to intervene (e.g., adjust stopping rules or abort). Without streaming, the agent must poll files or wait for the process to terminate.

**Cobre example**: A training run emits one progress event per iteration, plus lifecycle events:

```
{"type": "started", "case": "/data/case_001", "stages": 120, "hydros": 164, "timestamp": "2026-02-25T10:00:00Z"}
{"type": "progress", "iteration": 1, "lower_bound": 45231.7, "upper_bound": 89442.1, "gap": 0.494, "wall_time_ms": 12400}
{"type": "progress", "iteration": 2, "lower_bound": 51882.3, "upper_bound": 76553.8, "gap": 0.322, "wall_time_ms": 24100}
...
{"type": "terminated", "reason": "bound_stalling", "iterations": 87, "final_lb": 72105.4, "final_ub": 73211.8}
{"type": "result", "status": "success", "exit_code": 0, "output_directory": "/data/case_001/output"}
```

The `progress` event payload matches the per-iteration output record defined in [Convergence Monitoring §2.4](../architecture/convergence-monitoring.md). The same event stream feeds the TUI convergence plot, the MCP server progress notifications, and the CLI JSON-lines output -- all consumers receive identical event types from a single emitter. See Training Loop §2.1 (planned update) for the event emission points in the iteration lifecycle.

#### Rule 4: Every result is queryable without re-running computation

After a training or simulation run completes, all results must be accessible via read-only query operations (`cobre report`, `cobre-mcp` query tools, Python `cobre.load_results()`) without re-executing the solver. Query operations return structured JSON or Arrow tables.

**Rationale**: Agent workflows are often multi-step: run training, inspect convergence, compare with a previous run, extract specific entity results, generate a report. If any of these steps requires re-running the solver, the workflow becomes impractically expensive. Results persisted on disk (Parquet, FlatBuffers, JSON manifests) must be queryable through structured APIs.

**Cobre example**: After training completes, an agent queries the convergence history:

```bash
cobre report /data/case_001/output --output-format json --section convergence
```

This returns a JSON object with the full convergence history (all per-iteration records) read from `training/convergence.parquet`. The agent can then programmatically determine whether convergence was satisfactory or whether parameter adjustments are needed for a subsequent run.

Similarly, the MCP tool `cobre/query-results` reads Hive-partitioned Parquet simulation results and returns them as JSON, and the Python binding `cobre.load_results(path)` returns Arrow tables for zero-copy consumption via Polars or Pandas.

The existing output infrastructure -- manifests, metadata, convergence Parquet, Hive-partitioned simulation Parquet, FlatBuffers policy files -- already stores all necessary data. Agent-readability requires only that structured _access paths_ exist for this data, not that the storage format change. See [Output Infrastructure §1-2](../data-model/output-infrastructure.md) for the on-disk format and [MCP Server](../interfaces/mcp-server.md) for the query tool definitions.

### 6.3 Implementation Requirements

1. **Output format flag**: The CLI must accept `--output-format <human|json|json-lines>` as a global flag. Default is `human` for backward compatibility. The flag does not affect computation, only presentation.

2. **Envelope consistency**: All JSON responses from all subcommands must use the same top-level envelope schema (fields: `$schema`, `command`, `success`, `exit_code`, `cobre_version`, `errors`, `warnings`, `data`, `summary`). Subcommand-specific content lives within `data`.

3. **Event type definitions**: Progress event types must be defined as shared Rust structs accessible to all consumers (CLI JSON-lines writer, TUI renderer, MCP progress notifications, Parquet convergence writer). The event types are derived from the per-iteration output record in [Convergence Monitoring §2.4](../architecture/convergence-monitoring.md) and the iteration lifecycle steps in [Training Loop §2.1](../architecture/training-loop.md).

4. **Error kind registry**: All error kinds must be documented in a single authoritative list. New error kinds follow semantic versioning: additions are non-breaking, removals or renames are breaking changes.

5. **Schema versioning**: The `$schema` field in every JSON response enables agents to detect schema changes. Schema evolution follows additive-only rules: new fields may be added, existing fields must not be removed or change type.

### 6.4 What Agent-Readability Does NOT Require

- **JSON equivalents for large tabular data**: Simulation Parquet files (hundreds of MB) remain Parquet. Agents access them via Arrow libraries. Only control-plane data (progress, errors, metadata, summaries) requires JSON schemas.
- **Changes to computation**: Agent-readability is a presentation-layer concern. The `--output-format` flag does not alter solver behavior, random seeds, convergence criteria, or output files on disk.
- **Replacing human output**: The default `--output-format human` remains unchanged. Text training logs, decorative headers, and formatted tables continue to work for human operators and HPC batch workflows.

### 6.5 Validation Requirements

The test suite must include agent-readability tests that:

1. Verify every subcommand produces valid JSON when `--output-format json` is specified
2. Validate JSON responses against the documented envelope schema
3. Verify that every error kind in the registry produces a parseable structured error
4. Verify that `--output-format json-lines` for the `run` subcommand produces valid JSON-lines (one valid JSON object per line)
5. Verify that the response envelope `$schema` field is present and matches the expected version

### 6.6 Why This Matters

- **AI agent workflows**: Coding assistants and automation agents can compose multi-step Cobre workflows without parsing human-formatted text
- **CI/CD pipelines**: Continuous integration can validate solver results, check convergence, and compare runs using structured output -- no `grep` or `awk` required
- **Programmatic embedding**: Python bindings and MCP tools consume the same structured types, ensuring consistency across all access paths
- **Interoperability**: Stable JSON schemas enable third-party tools (dashboards, reporting, monitoring) to integrate with Cobre without coupling to display format details

## 7. Numeric Representation

All numeric quantities in the Cobre data model use `f64` with unit suffixes in field names. Newtypes for physical dimensions are not used. This decision affects ~40 `f64` fields across entity types (`Hydro`, `ThermalUnit`, `Line`, `Bus`, `Contract`, `PumpingStation`) spanning 7 distinct physical dimensions: volume (hm³), flow (m³/s), power (MW), energy (MWh), marginal cost ($/MWh), height (m), and dimensionless fractions.

### 7.1 Alternatives Considered

**Newtypes throughout cobre-core (rejected).** Wrapper types for each physical dimension (`struct Hm3(pub f64)`) would live in `cobre-core`, `cobre-io`, and `cobre-stochastic`, stripped to raw `f64` at the performance-adapted view boundary in `cobre-sddp`. Rejected because: (1) the protected surface is narrow — newtypes catch dimension-mismatch bugs only in entity definitions and validation code, while the bug-prone code (LP construction, cut coefficient computation, solver interaction) operates on raw `f64` arrays regardless; (2) seven unit types require ~15–20 operator implementations for physically meaningful combinations, or `.0` extraction that defeats the purpose; (3) cut construction is inherently untyped — a Benders cut coefficient has units derived from dual variables of constraints with mixed physical meanings; (4) trajectory recording and Parquet output extract values from flat `Vec<f64>` solver buffers; (5) field name suffixes already convey units at every use site, making newtypes redundant labeling.

**External dimensional analysis crate (rejected).** The `uom` crate encodes dimensions as generic type parameters, producing unreadable type signatures and error messages with significant compile-time overhead. For a system with 7 fixed unit types where conversion formulas are well-known (see [Notation Conventions](./notation-conventions.md) section 3.1), full dimensional analysis is disproportionate.

### 7.2 Rationale

**The data flow has a natural untyped boundary.** The [Internal Structures](../data-model/internal-structures.md) spec defines a clear lifecycle boundary: `cobre-io` (JSON loading) → `cobre-core` (system entities) → `cobre-sddp` (performance-adapted `f64` views). Performance-adapted view construction already flattens entity structs into `Vec<f64>` slices. Newtypes would protect only the code above this boundary — input loading, entity validation, stochastic model parameter handling — the least error-prone parts of the system. The code below the boundary (LP coefficient assembly, cut construction, basis management, checkpoint serialization) is where dimension errors would be most harmful and where newtypes offer no protection.

**The bugs newtypes catch vs. the bugs they don't.** Newtypes would catch: passing volume (hm³) where flow (m³/s) is expected, forgetting time conversion factors, confusing MW and MWh. Newtypes would not catch: wrong index into a `Vec<f64>` array (the dominant bug class in SDDP implementations), wrong sign on a constraint coefficient, off-by-one in stage numbering, incorrect cut coefficients from wrong dual extraction, swapping two values of the same dimension. The uncaught bugs are harder to detect and have higher impact.

**Validation strategy makes newtypes redundant for correctness.** The primary correctness strategy is comparison against SDDP.jl reference cases and reproduction of known optimal bounds. A dimension-mismatch bug in entity loading would produce marginal costs off by orders of magnitude — immediately detectable in the first reference case comparison. Property-based tests on entity validation and statistical tests on PAR model preservation provide additional coverage for the domain-model layer.

### 7.3 Conventions

1. **Field names carry the unit suffix.** Every `f64` field representing a dimensional quantity includes a unit suffix: `_hm3`, `_m3s`, `_mw`, `_mwh`, `_per_mwh`, `_m`, `_percent`, `_hours`. Dimensionless quantities use no suffix or `_fraction` when ambiguity is possible.

2. **Unit suffixes match notation conventions.** The suffixes correspond to the units defined in [Notation Conventions](./notation-conventions.md) section 3. When a new physical dimension is introduced, a corresponding suffix must be established and documented.

3. **JSON field names mirror Rust field names.** Serde serialization uses snake_case. JSON keys carry the same unit suffix as the Rust field: `"min_storage_hm3": 42.0`.

4. **Conversion formulas are centralized.** The flow-to-volume conversion factor ζ (see [Notation Conventions](./notation-conventions.md) section 3.1) and other unit conversion formulas are implemented as named functions, not inline arithmetic.

5. **Doc comments state units explicitly.** Every `f64` field has a `///` doc comment that includes the unit in brackets: `/// Minimum storage (dead volume) [hm³]`.

### 7.4 Reversibility

This decision is reversible. Introducing newtypes in `cobre-core` would be a mechanical refactor scoped to `cobre-core`, `cobre-io`, and `cobre-stochastic`, with no changes to `cobre-sddp` or any crate below the performance-adapted view boundary.

## Cross-References

- [Notation Conventions](./notation-conventions.md) — Mathematical notation and symbol definitions used across all specs
- [Production Scale Reference](./production-scale-reference.md) — Production-scale LP dimensions and performance targets
- [LP Formulation](../math/lp-formulation.md) — Complete LP subproblem formulation
- [Input Directory Structure](../data-model/input-directory-structure.md) — File layout implementing these format choices
- [Validation Architecture](../architecture/validation-architecture.md) — Validation of order invariance and other requirements; foundation for structured error schema (§4-5)
- [Convergence Monitoring](../architecture/convergence-monitoring.md) — Per-iteration output record (§2.4) that defines the progress event payload
- [Training Loop](../architecture/training-loop.md) — Iteration lifecycle (§2.1) that defines the event emission points
- [Structured Output](../interfaces/structured-output.md) — Full JSON schema definitions for CLI response envelope, error schema, and JSON-lines streaming protocol
- [MCP Server](../interfaces/mcp-server.md) — MCP tool, resource, and prompt definitions for agent interaction
- [Python Bindings](../interfaces/python-bindings.md) — PyO3 API surface, zero-copy data paths, GIL management
- [Terminal UI](../interfaces/terminal-ui.md) — TUI event consumption, convergence plot, interactive features
- [Internal Structures](../data-model/internal-structures.md) — Entity type definitions where the suffix convention applies (~40 f64 fields)
- [Solver Workspaces](../architecture/solver-workspaces.md) — Thread-local f64 buffers in the solver hot path
- [Binary Formats](../data-model/binary-formats.md) — FlatBuffers schema with [double] coefficient arrays
- [Input Loading Pipeline](../architecture/input-loading-pipeline.md) — LoadError enum and serde requirements on core types
- [Output Schemas](../data-model/output-schemas.md) — Parquet column naming conventions for simulation output
