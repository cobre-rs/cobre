# Serialization and Persistence Formats

## Purpose

This spec defines the format decisions (JSON, Parquet, FlatBuffers) used across Cobre for data persistence and serialization, and provides the authoritative format decision framework referenced by all other data model specs. It covers:

1. **Format decision framework** — criteria for choosing JSON vs Parquet vs FlatBuffers for each data category
2. **FlatBuffers for policy data** — schema for cuts, visited states, vertices, and checkpoint data
3. **Cut pool persistence** — what must be serialized for checkpoint/resume/warm-start and why
4. **Output format requirements** — Parquet configuration requirements for simulation and training outputs

For the logical in-memory data model (what the solver holds at runtime), see [Internal Structures](internal-structures.md). For output file schemas, see [Output Schemas](output-schemas.md) and [Output Infrastructure](output-infrastructure.md).

## 1. Format Decision Framework

> **Decision [DEC-004](../overview/decision-log.md#dec-004) (active):** Parquet for all tabular input data (entity registries, stage-varying overrides, time series, scenario parameters).

This framework is the authoritative reference for all format choices across the data model. Each data model spec references this when justifying per-file format choices.

| Data Nature               | Format         | Key Examples                                        | Rationale                                                                |
| ------------------------- | -------------- | --------------------------------------------------- | ------------------------------------------------------------------------ |
| Registry / catalog        | JSON           | buses.json, hydros.json, thermals.json, lines.json  | Structured objects with nested fields, human-editable                    |
| Entity-level tabular data | Parquet        | Geometry tables, FPHA hyperplanes, bounds overrides | Columnar lookup tables indexed by entity — efficient typed storage       |
| Default-with-overrides    | JSON + Parquet | penalties.json (base) + stage overrides             | Hierarchical defaults in JSON, sparse stage-varying overrides in Parquet |
| Complex nested object     | JSON           | config.json, stages.json, constraints/\*.json       | Deep nesting, optional sections, human-editable                          |
| Correlation / matrix      | JSON           | correlation.json                                    | Sparse, small, human-reviewable                                          |
| Policy / binary           | FlatBuffers    | Policy cuts, states, vertices, checkpoint data      | Zero-copy deserialization, SIMD-friendly dense arrays                    |
| High-volume output        | Parquet        | Simulation results, training outputs                | Columnar compression, partition pruning, analytics tooling               |
| Metadata / dictionary     | CSV / JSON     | variables.csv, entities.csv, codes.json             | Human-readable, small volume                                             |
| MPI broadcast             | postcard       | `System` struct rank-0-to-all broadcast             | Fast `serde`-based serialization for one-time internal transfer          |

### Format Selection Criteria

| Criterion           | JSON          | Parquet             | FlatBuffers           |
| ------------------- | ------------- | ------------------- | --------------------- |
| Human editable      | Yes           | No                  | No                    |
| Schema evolution    | Moderate      | Good                | Good                  |
| Compression ratio   | Low           | High (~4x)          | Moderate              |
| Random access       | No            | Column + row group  | Field-level           |
| Zero-copy load      | No            | No                  | Yes                   |
| Analytics tooling   | Limited       | Excellent           | Limited               |
| Dense array storage | Poor          | Poor (many columns) | Yes                   |
| Write frequency     | Config (once) | Output (streaming)  | Checkpoint (periodic) |

## 2. Format Summary by Category

> **Decision [DEC-002](../overview/decision-log.md#dec-002) (active):** `postcard` for MPI broadcast serialization of the `System` struct from rank 0 to all worker ranks; replaces earlier `rkyv` decision.

| Data Category     | Read/Write | Format         | Rationale                                                                                                                                             |
| ----------------- | ---------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Algorithm Config  | Read       | JSON           | Small, editable                                                                                                                                       |
| System Registry   | Read       | JSON           | Structured objects                                                                                                                                    |
| Stage/Block Def   | Read       | JSON           | Graph structure                                                                                                                                       |
| Scenario Pipeline | Read       | JSON + Parquet | Complex nested config + tabular data in Parquet                                                                                                       |
| Correlation       | Read       | JSON           | Sparse, small                                                                                                                                         |
| Stage Overrides   | Read       | Parquet        | Sparse per-entity/per-stage tabular overrides                                                                                                         |
| Policy Cuts       | Read/Write | FlatBuffers    | Zero-copy, in-memory training                                                                                                                         |
| Policy States     | Read/Write | FlatBuffers    | Zero-copy, in-memory training                                                                                                                         |
| Policy Vertices   | Read/Write | FlatBuffers    | Zero-copy, in-memory training                                                                                                                         |
| Training Results  | Write      | Parquet        | Analytics-ready                                                                                                                                       |
| Simulation Detail | Write      | Parquet        | Large volume                                                                                                                                          |
| Dictionaries      | Write      | CSV            | Human-readable                                                                                                                                        |
| MPI Broadcast     | Internal   | postcard       | Fast serialization for rank-0-to-all broadcast of the `System` struct. See [Input Loading Pipeline SS6.1](../architecture/input-loading-pipeline.md). |

## 3. FlatBuffers for Policy Data

> **Decision [DEC-003](../overview/decision-log.md#dec-003) (active):** FlatBuffers for policy data persistence (cuts, states, vertices, checkpoint data); zero-copy deserialization, SIMD-friendly layout.

### 3.0 Runtime Access Pattern and Memory Model

Understanding why FlatBuffers was chosen requires understanding how cuts are accessed at runtime. The cut pool is **not** read directly by the LP solver — the solver (HiGHS, CLP, etc.) has its own internal problem representation. Instead, cut data flows through several stages:

1. **Cut pool in memory**: One copy per MPI rank (or shared via MPI shared memory windows across ranks on the same node). Updated during backward pass FCF updates.
2. **LP construction**: Before each LP solve, active cut constraints must be loaded into the solver's internal matrix. The solver owns its copy of the constraint data.
3. **Checkpoint/persistence**: Periodically serialized to disk. This is where FlatBuffers is used.

The FlatBuffers format serves two purposes:

- **Checkpoint writes**: Fast serialization from in-memory cut pool to disk
- **Policy loading**: Efficient deserialization on resume/warm-start (zero-copy access to coefficient arrays)

The in-memory cut pool itself is a native data structure optimized for the LP construction step — it is not stored as FlatBuffers at runtime.

#### Parallel Forward Pass Memory Problem

During training, multiple forward passes execute in parallel across MPI ranks and threads within each rank. All forward passes visiting the same stage solve the **same LP structure** — they share all structural constraints and all cut constraints. The only differences are scenario-dependent values:

**Identical across all forward passes at a given stage** (from `lp_sizing.py` production estimates):

| Component                                                                 | Size              |
| ------------------------------------------------------------------------- | ----------------- |
| Structural constraints (water balance, load balance, FPHA, outflow, etc.) | 65,628 rows       |
| Cut constraints (Benders cuts from previous iterations)                   | up to 15,000 rows |
| Cut coefficients (15,000 cuts x 2,080 state dimension)                    | **238 MB**        |
| Objective function, most variable bounds                                  | small             |

**Different per forward pass** (the "scenario snapshot"):

| Component                               | Size                       |
| --------------------------------------- | -------------------------- |
| Incoming storage (RHS of water balance) | 160 values                 |
| AR lag state (lagged inflows)           | 1,920 values               |
| Current-stage inflow noise              | 160 values                 |
| **Total scenario-dependent data**       | **~2,240 values (~17 KB)** |

The ratio is extreme: ~238 MB of shared structure vs. ~17 KB of per-scenario data.

#### Why This Matters for Format Decisions

Each concurrent LP solve requires its own solver instance — solvers mutate internal state (basis, factorization, working arrays) during solve and cannot be shared across threads. This creates a tension:

The solver needs the full constraint matrix loaded, including all active cut rows. But the 238 MB of cut coefficients per stage are identical across all forward passes. The architectural question — whether each thread rebuilds the LP from scratch or clones a template — directly impacts how frequently cut data must be read from the cut pool and how efficiently it must be laid out for bulk loading into solver APIs.

Four architectural options were evaluated:

| Option | Strategy                                                 | Memory per rank (16 threads)        | Feasibility                                               |
| ------ | -------------------------------------------------------- | ----------------------------------- | --------------------------------------------------------- |
| **A**  | One solver per thread, full rebuild per stage transition | ~4 GB solvers + 14.3 GB shared cuts | Feasible, highest rebuild cost                            |
| **B**  | One solver per (thread, stage), persistent               | 16 x 60 x 255 MB ~ 245 GB           | **Infeasible**                                            |
| **C**  | Master LP per stage + per-thread clone/patch             | ~15.3 GB masters + 4 GB workers     | Feasible, depends on clone efficiency                     |
| **D**  | Per-thread solver, incremental modify across stages      | ~4 GB solvers + 14.3 GB shared cuts | Limited benefit due to inter-stage structural differences |

> **Note (LP construction strategy):** StageLpCache is the adopted baseline for parallel forward pass LP construction — see [DEC-001](../overview/decision-log.md#dec-001) and [Solver Abstraction SS11.4](../architecture/solver-abstraction.md). Cut coefficients are pre-assembled into a per-stage CSC via SharedRegion; the cut pool retains metadata only. The `addRows` CSR path is used only during StageLpCache assembly between iterations by the leader rank, not on the hot-path stage transition.
>
> **Historical note (2026-02-16):** Full rebuild per stage was originally adopted. Solver API analysis confirmed that neither HiGHS nor CLP expose efficient LP cloning through their C APIs, making the clone-based approach solver-specific and complex. Full rebuild is portable (works identically for both solvers), simpler (no solver-specific clone paths), and the LP solve time dominates construction time regardless. This analysis remains valid historical context for why the clone-based and persistent-per-thread approaches were rejected. Superseded by the StageLpCache strategy (2026-02-28).

Policy data (cuts, states, vertices) has a unique persistence profile:

| Characteristic                   | Description                                                                                                                                                                                        |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **In-memory during training**    | Entire cut pool lives in RAM, shared across threads                                                                                                                                                |
| **Loaded into solver per solve** | Active cuts must be added to solver's internal matrix                                                                                                                                              |
| **Checkpointed periodically**    | Serialized to disk only at checkpoint intervals                                                                                                                                                    |
| **High state dimension**         | 2,080 coefficients per cut at production scale                                                                                                                                                     |
| **Large volume**                 | Up to 15,000 cuts/stage x 60 stages ~ 14.3 GB per rank (coefficients absorbed into StageLpCache SharedRegion at ~22.3 GB node-wide; see [Memory Architecture §2.1](../hpc/memory-architecture.md)) |

**Why not Parquet?** Using 2,080 individual columns (`coefficient_0` through `coefficient_2079`) is inefficient for Parquet, which is optimized for columnar analytics, not dense fixed-size arrays.

**Why FlatBuffers?**

1. **Zero-copy deserialization**: Load cut pool from checkpoint without parsing overhead
2. **Dense array access**: Coefficient vectors stored as contiguous `[double]` arrays, directly usable for bulk loading into solver row-addition APIs
3. **Simple schema**: Flat structure maps directly to in-memory representation
4. **Fast checkpoint writes**: Serialize directly from in-memory structures

### 3.1 FlatBuffers Schema

```flatbuffers
// File: schemas/policy.fbs
namespace cobre.policy;

// Benders cut: theta >= intercept + sum_i coefficients[i] * state[i]
// where intercept = alpha - beta' * x_hat (pre-computed)
table BendersCut {
    cut_id: uint64;
    slot_index: uint32;          // LP row position (REQUIRED for reproducibility)
    iteration: uint32;
    forward_pass_idx: uint32;
    intercept: double;           // alpha - beta' * x_hat
    coefficients: [double];      // beta (length = state_dimension)
    is_active: bool = true;
    domination_count: uint32 = 0;
}

// All cuts for a single stage (active AND inactive for reproducibility).
// On deserialization, the runtime cut pool builder copies these into a
// memory layout optimized for CSR assembly (see §3.4). The contiguous
// [double] coefficient vectors in each BendersCut map directly to CSR
// row data — each cut becomes one row in the solver's addRows call.
table StageCuts {
    stage_id: uint32;
    state_dimension: uint32;
    capacity: uint32;            // Total preallocated slots
    warm_start_count: uint32;    // Slots [0..warm_start_count) from loaded policy
    cuts: [BendersCut];          // Length = populated_count
    active_cut_indices: [uint32]; // Which cuts to load into solver (CSR assembly source)
    populated_count: uint32;
}

// Visited states for cut selection and analysis export.
//
// The flat data layout mirrors the in-memory VisitedStatesArchive:
// a single contiguous [double] buffer containing all state vectors
// for one stage, with count and state_dimension to index into it.
// This avoids per-state-record overhead and enables zero-copy access
// to the coefficient data during policy loading.
table StageStatesPayload {
    stage_id: uint32;
    state_dimension: uint32;     // Length of each state vector
    count: uint32;               // Number of states stored
    data: [double];              // Flat buffer: count * state_dimension elements
}

// Vertex for inner approximation (upper bound / SIDP)
table Vertex {
    vertex_id: uint64;
    iteration: uint32;
    forward_pass_idx: uint32;
    scenario_idx: uint32;
    components: [double];        // Length = state_dimension
    upper_bound_value: double;
    lipschitz_constant: double;
}

table StageVertices {
    stage_id: uint32;
    state_dimension: uint32;
    vertices: [Vertex];
    stage_lipschitz: double;
}

// Cached solver basis for warm-start at each stage transition.
// Status codes are solver-specific integers but always have the same
// structure: one code per column, one code per row. On checkpoint, the
// codes are stored as-is; on resume with a different solver, a
// translation layer maps between code sets.
table StageBasis {
    stage_id: uint32;
    iteration: uint32;           // Iteration that produced this basis
    num_columns: uint32;         // Number of column status codes
    num_rows: uint32;            // Number of row status codes (structural + cuts)
    column_status: [ubyte];      // One status code per column (variable)
    row_status: [ubyte];         // One status code per row (constraint)
    num_cut_rows: uint32;        // How many of the final rows are cut rows
                                 // (new cuts added after this basis was saved
                                 // should be set to "basic" status on load)
}

// Policy metadata for resume/warm-start
table PolicyMetadata {
    cobre_version: string;
    created_at: string;          // ISO 8601
    completed_iterations: uint32;
    final_lower_bound: double;
    best_upper_bound: double;
    max_iterations: uint32;
    forward_passes: uint32;
    warm_start_cuts: uint32;
    // capacity = warm_start_cuts + max_iterations * forward_passes
    rng_seed: uint64;
    state_dimension: uint32;
    num_stages: uint32;
    total_visited_states: uint64 = 0; // Sum of state counts across all stages
                                      // (default 0 for backward compat with
                                      // checkpoints written before this field)
}

root_type StageCuts;
```

### 3.2 Policy Directory Structure

```
policy/
├── metadata.json               # Human-readable (JSON for editability)
│                               # Includes total_visited_states field (default 0
│                               # for backward compat with older checkpoints)
├── state_dictionary.json       # State variable mapping
├── cuts/
│   ├── stage_000.bin          # FlatBuffers StageCuts
│   ├── stage_001.bin
│   └── ...
├── states/
│   ├── stage_000.bin          # FlatBuffers StageStatesPayload (flat data layout)
│   ├── stage_001.bin
│   └── ...
├── vertices/                   # Only if inner approximation enabled
│   ├── stage_000.bin          # FlatBuffers StageVertices
│   └── ...
└── basis/                      # Cached solver basis for warm-start (§3.4)
    ├── stage_000.bin          # FlatBuffers StageBasis
    ├── stage_001.bin
    └── ...
```

### 3.3 Encoding Guidelines

| Field Type                        | Encoding          | Rationale                       |
| --------------------------------- | ----------------- | ------------------------------- |
| `cut_id`, `state_id`, `vertex_id` | uint64            | Unique across all iterations    |
| `iteration`, `stage_id`           | uint32            | Sufficient for practical limits |
| `coefficients`, `components`      | `[double]` dense  | SIMD-friendly, no dictionary    |
| `is_active`                       | bool              | Bit-packed by FlatBuffers       |
| Timestamps                        | string (ISO 8601) | Human-readable in metadata      |

**Compression**: `.bin` (uncompressed, fast load) or `.bin.zst` (Zstd-compressed, archival/transfer).

### 3.4 Cut Pool Memory Layout Requirements

> **Critical requirement**: The in-memory cut pool layout must enable efficient assembly of CSR (Compressed Sparse Row) data for the solver's batch `addRows` API call. This is the primary data path for StageLpCache assembly — between iterations, the leader rank rebuilds each stage's CSC by assembling active cuts via `addRows` in CSR format, then persisting the result as the updated StageLpCache entry.

The cut pool layout enables efficient StageLpCache CSC assembly by the leader rank between iterations. The assembly step works as follows:

1. Query the cut pool for the active cut indices at the current stage (from `active_cut_indices`)
2. Assemble CSR arrays: `row_starts`, `column_indices`, `coefficient_values`, `row_lower_bounds`, `row_upper_bounds`
3. Call the solver's batch row-addition API once (e.g., `Highs_addRows`, `Clp_addRows`) to extend the stage template into a complete StageLpCache entry

At production scale this means assembling up to 15,000 active cut rows with 2,080 non-zeros each (all state variables participate in every cut — the coefficient vectors are dense) into CSR format. The cut pool layout must minimize the cost of this assembly.

**Layout requirements**:

| Requirement                                                                                                 | Rationale                                                                                                                                                                                                                                                                                  |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| All coefficients for a single cut stored as a contiguous dense `[double]` array of length `state_dimension` | Each cut becomes one CSR row. Contiguous storage means the coefficient values can be memcpy'd directly into the CSR `values` array without gather operations.                                                                                                                              |
| All cuts for a single stage stored in a contiguous, indexable collection                                    | Iterating active cuts by index (from `active_cut_indices`) must be O(1) per cut — no pointer chasing or hash lookups.                                                                                                                                                                      |
| Intercepts stored separately from coefficient arrays                                                        | During CSR assembly, only coefficients and bounds are needed. Intercepts are used to compute row bounds (`lower = intercept`, `upper = +inf`) but are not part of the sparse matrix data itself. Separating them avoids pulling unneeded data into cache during the coefficient copy loop. |
| Collection aligned to 64-byte cache lines                                                                   | The coefficient copy loop processes 2,080 x 8 = 16,640 bytes per cut. Cache-line alignment avoids split-line loads.                                                                                                                                                                        |

**Why dense, not sparse**: Every Benders cut has a non-zero coefficient for every state variable (storage volumes and AR lags all appear in the cut equation). The coefficient vectors are fully dense — there is no sparsity to exploit. The CSR `column_indices` array for cuts is therefore a fixed, repeating pattern `[0, 1, 2, ..., state_dimension-1]` that can be precomputed once and reused for all rows.

**FlatBuffers alignment**: The FlatBuffers schema (§3.1) stores coefficients as `[double]` vectors, which are inherently contiguous. On deserialization (checkpoint resume / warm-start load), the cut pool builder should copy these vectors into the runtime layout described above, preserving contiguity and alignment. The FlatBuffers format is not used at runtime — it is the serialization format that feeds the runtime layout.

#### Basis Caching for Warm-Start

At each stage transition, the StageLpCache provides the complete LP (structural constraints + active Benders cuts, pre-assembled in CSC format). Without basis information, the solver starts from a logical (slack) basis and must perform a full solve — potentially thousands of simplex iterations. By caching the basis from the previous iteration's solve at each stage and applying it to the LP loaded from StageLpCache, the solver warm-starts from a near-optimal basis and converges in far fewer iterations.

**How basis reuse works with StageLpCache**:

1. Solve the LP at stage `t` during iteration `k`
2. Extract the solver's basis (one status code per column + one per row) and store it indexed by stage
3. On iteration `k+1`, load `StageLpCache[t]` into the solver via `passModel`/`loadProblem` (the cache entry already contains the updated active cuts from iteration `k`'s backward pass)
4. Apply the cached basis from iteration `k` to the loaded LP
5. For cut rows that were added since the cached basis was saved (new cuts from iteration `k`'s backward pass), set their status to **basic** — meaning the slack variable for that constraint is in the basis. The solver will price these rows in and pivot as needed.
6. Solve with warm-start from this patched basis

**Basis structure**: Regardless of the solver (HiGHS, CLP, or others), a simplex basis is always represented as an array of small integer status codes — one code per column (variable) and one code per row (constraint). The codes encode whether each variable/constraint is basic, at its lower bound, at its upper bound, free, or fixed. The specific integer values differ between solvers (HiGHS uses `kHighsBasisStatus*` constants, CLP uses 0-5), but the structure is universal: two flat integer arrays.

**Sizing**: At production scale, the LP has ~8,360 columns and ~80,628 rows. At one byte per status code, a basis snapshot is ~87 KB per stage — negligible compared to the 238 MB of cut coefficients. Across 60 stages, total basis storage is ~5 MB.

> **Requirement**: Basis data must be stored with the same efficiency and accessibility as cut data — one contiguous array of status codes per stage, directly loadable into the solver's `setBasis`/`copyinStatus` API without transformation. The basis is updated every iteration, so both read and write must be fast.

See [Internal Structures](internal-structures.md) for the logical data model that this layout implements.

### 3.5 Access Pattern Requirements

The FlatBuffers schema and the runtime cut pool layout must satisfy two access pattern requirements. These requirements constrain any future revision of the schema in SS3.1 and complement the memory layout specification in SS3.4.

**Batch extraction.** The FlatBuffers schema and the runtime cut pool layout must support extracting all active cuts for a given stage in a single sequential pass. The CSR assembly loop (SS3.4) iterates the `active_cut_indices` list, and for each index reads the corresponding cut's coefficient vector as a contiguous `[f64; state_dimension]` slice. No indirection beyond the index list is permitted on this path. The schema in SS3.1 satisfies this via the `StageCuts.active_cut_indices` field and per-`BendersCut` `coefficients: [double]` vectors.

**Per-cut cache locality.** At production scale, a single cut coefficient vector is $2{,}080 \times 8 = 16{,}640$ bytes ($\approx 16.3$ KB). This fits within the L1 data cache of modern CPUs (32-48 KB), ensuring that the CSR copy loop for one cut executes entirely from L1. The full active cut set (up to 15,000 cuts, $\approx 238$ MB of coefficients) vastly exceeds all cache levels. The performance model therefore assumes **streaming access**: the CPU prefetcher loads the next cut's coefficient vector while the current cut is being copied to the CSR output buffer. Cache-line alignment (64 bytes, per SS3.4) ensures that prefetch requests align with hardware prefetch boundaries.

> **Schema evolution note.** The `.fbs` schema in SS3.1 is the current specification based on the architectural analysis in SS3.0. It may be revised during implementation when actual data access patterns (cut generation rate, active cut ratio, warm-start coefficient reuse) are measured under production-scale workloads. Any revision must preserve the batch extraction and per-cut cache locality requirements stated above. Schema backward compatibility is maintained through FlatBuffers' field ID mechanism — new fields receive new IDs, deprecated fields are retained but ignored, and readers compiled against an older schema version can still deserialize buffers produced by a newer version.

## 4. Cut Pool Persistence

> **Cut Preallocation Strategy**: Full preallocation with dynamic capacity.

The cut pool uses full preallocation to achieve:

- **Bit-for-bit reproducibility**: Checkpoints restore exact LP state (same slot indices, same row structure)
- **Zero runtime allocation**: No thread-safety concerns during parallel solve
- **Warm-start support**: Capacity = existing loaded cuts + new training cuts

### 4.1 Checkpoint Reproducibility

> **Critical**: Checkpoint/resume must produce **bit-for-bit identical** results.

**Why complete state serialization matters:**

```
Run A: Fresh start, 50 iterations → generates cuts in slots [0..10000)
  → deactivates some via Level 1 selection → specific LP row structure
  → specific solver pivots → specific duals → specific new cuts

Run B: Resume from iteration 25 checkpoint
  → MUST reconstruct identical LP structure (same slots, coefficients, bounds)
  → identical pivots → identical duals → identical cuts → identical results
```

| Data                         | Must Serialize | Rationale                |
| ---------------------------- | -------------- | ------------------------ |
| All cuts (active + inactive) | Yes            | LP row structure         |
| Slot indices                 | Yes            | Row mapping              |
| is_active flags              | Yes            | Bound values             |
| Coefficients                 | Yes            | Cut geometry             |
| RNG state                    | Yes            | Scenario reproducibility |
| Solver basis                 | Recommended    | Exact warm-start         |

### 4.2 Execution Modes

| Mode         | Cut Loading       | RNG State              | Capacity              | Results                   |
| ------------ | ----------------- | ---------------------- | --------------------- | ------------------------- |
| `fresh`      | None              | From config seed       | max_iter x fwd_passes | Deterministic from seed   |
| `warm_start` | All from policy   | Fresh from config seed | loaded + new          | Different from original   |
| `resume`     | All + exact state | Restored               | Same as checkpoint    | **Bit-for-bit identical** |

### 4.3 Cut Pool Sizing

At production scale, the cut pool for a single stage can hold:

- **Capacity**: `warm_start_cuts + (max_iterations x forward_passes)` cuts
- **Per-cut memory**: ~17 KB at 2,080 state dimensions (2,080 x 8 bytes for coefficients + metadata)
- **Per-stage total**: Up to 15,000 cuts (~238 MB of coefficients alone)
- **All stages**: 60 stages x 238 MB ~ 14.3 GB per MPI rank (coefficients absorbed into StageLpCache SharedRegion at ~22.3 GB node-wide; see [Memory Architecture §2.1](../hpc/memory-architecture.md))

The preallocation strategy means all memory is allocated at initialization. The `populated_count` tracks how many slots are filled; an active bitmap tracks which populated cuts are active for LP construction.

## 5. Parquet Output Configuration

Simulation and training outputs are written as Parquet files. The Parquet writer should be configured for a balance between compression ratio and write speed:

- **Compression**: Zstd (level 3) — good compression ratio without excessive CPU cost
- **Row group size**: ~100,000 rows — large enough for efficient column encoding, small enough for reasonable memory during writes
- **Statistics**: Enabled — allows predicate pushdown for analytics queries
- **Dictionary encoding**: Enabled for categorical columns (entity IDs, scenario IDs, stage IDs) with a dictionary page size limit of ~1 MB

For output schema definitions, see [Output Schemas](output-schemas.md). For production-scale output volume estimates, see [Output Infrastructure](output-infrastructure.md).

## Cross-References

- [Internal Structures](internal-structures.md) — Logical in-memory data model for the SDDP solver
- [Output Schemas](output-schemas.md) — Parquet column definitions for output files
- [Output Infrastructure](output-infrastructure.md) — Output directory structure, manifest, hive partitioning
- [Penalty System](penalty-system.md) — Penalty categories and cascade resolution
- [Input Constraints](input-constraints.md) — Initial conditions and policy directory
- [SDDP Algorithm](../math/sddp-algorithm.md) — Algorithm that produces/consumes cuts
- [Cut Management](../math/cut-management.md) — Cut selection strategies using cut pool
- [Solver Abstraction](../architecture/solver-abstraction.md) — Solver interface design
- [Input Loading Pipeline](../architecture/input-loading-pipeline.md) — MPI broadcast serialization format using `postcard` (SS6.1)
