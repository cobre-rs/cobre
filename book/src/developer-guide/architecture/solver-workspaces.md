# Solver Workspaces & LP Scaling

## Purpose

This spec defines the thread-local solver workspace infrastructure and the LP scaling specification. These components bridge the [Solver Abstraction Layer](./solver-abstraction.md) with the HPC execution layer, ensuring each Rayon thread has an exclusive, NUMA-local solver instance with pre-allocated buffers for the SDDP hot path. For solver-specific implementations, see [HiGHS Implementation](./solver-highs-impl.md) and [CLP Implementation](./solver-clp-impl.md).

## 1. Thread-Local Solver Infrastructure

### 1.1 Design Rationale

LP solvers (HiGHS, CLP, CPLEX) are **not thread-safe**. Each Rayon thread requires its own solver instance. The workspace pattern provides:

1. **Exclusive ownership** — Each thread owns one solver instance for the entire SDDP run (no pool-based borrow/return). This eliminates synchronization overhead and enables optimal warm-starting.
2. **Per-stage basis cache** — The workspace stores one basis per stage. The basis from solving stage `t` in iteration `i` is reused to warm-start stage `t` in iteration `i+1`, without external basis save/restore. For checkpoint/restart, the cache is serialized to FlatBuffers (see SS1.2).
3. **Pre-allocated buffers** — All solution extraction, RHS patching, and basis storage buffers are allocated once at initialization and reused for every solve, eliminating hot-path allocations.
4. **NUMA-local memory** — Each workspace is allocated on the NUMA node local to its owning thread, maximizing memory bandwidth during LP solves.

### 1.2 Workspace Contents

Each thread-local workspace contains the following components:

| Component             | Description                                                                                                                                                                                                                                                                                                                                                                                      | Sizing (production: 8,360 cols, 80,628 rows)      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------- |
| Solver instance       | Opaque solver handle (`void*` for HiGHS, `Clp_Simplex*` for CLP). Persists for the entire run. Includes LP matrix storage, working arrays (pricing, steepest edge weights), and LU factorization workspace. Does **not** include the LP data itself — that is loaded per stage via `load_model` (the Rust trait method; wraps the underlying C API calls `Highs_passModel` / `Clp_loadProblem`). | ~15 MB (see breakdown below)                      |
| RHS patch buffer      | Pre-allocated arrays for batch bound modification during scenario patching (row indices + lower/upper values). Sized to the maximum scenario-dependent value count across stages (~2,240 values at production scale).                                                                                                                                                                            | ~54 KB                                            |
| Primal buffer         | Pre-allocated array for primal solution extraction. These are **single-solve scratch buffers** — the SDDP algorithm copies the values it needs (state variables, generation, etc.) after each solve. The buffer is overwritten on the next solve.                                                                                                                                                | 8,360 × 8 = ~65 KB                                |
| Dual buffer           | Pre-allocated array for dual solution extraction. Same single-solve semantics as primal buffer.                                                                                                                                                                                                                                                                                                  | 80,628 × 8 = ~630 KB                              |
| Reduced cost buffer   | Pre-allocated array for reduced cost extraction. Same single-solve semantics as primal buffer.                                                                                                                                                                                                                                                                                                   | 8,360 × 8 = ~65 KB                                |
| Per-stage basis cache | Array of `T` basis slots (one per stage). Each slot stores the basis from the last successful solve at that stage, used for warm-starting the same stage in the next iteration. See SS1.5.                                                                                                                                                                                                       | T × (num_cols + num_rows) × 1–4 bytes (see below) |
| Solve statistics      | Accumulated per-thread counters (see SS1.6).                                                                                                                                                                                                                                                                                                                                                     | ~64 bytes                                         |
| NUMA node ID          | Which NUMA node this workspace's memory is allocated on.                                                                                                                                                                                                                                                                                                                                         | scalar                                            |

**Cache line padding**: Workspaces should be padded to cache line boundaries (64 bytes) to prevent false sharing between adjacent workspaces in an array.

**Solver instance breakdown** (production scale, from [HiGHS SS5](./solver-highs-impl.md) and [CLP SS6](./solver-clp-impl.md)):

| Sub-component     |       Size | Notes                                               |
| ----------------- | ---------: | --------------------------------------------------- |
| LP matrix storage |      ~5 MB | nnz × 16 bytes (value + index, CSC format)          |
| Working arrays    |      ~2 MB | (rows + cols) × 3 × 8 bytes (pricing, edge weights) |
| LU factorization  |   ~5–10 MB | Varies with sparsity; dominates solver memory       |
| Internal metadata |      ~1 MB | Iteration buffers, scaling arrays, options          |
| **Total**         | **~15 MB** | Consistent across HiGHS and CLP                     |

**Per-stage basis sizing** (production: T = 60 stages, 88,988 status entries per basis):

| Solver | Status size | Per basis | 60 stages | Notes                           |
| ------ | ----------: | --------: | --------: | ------------------------------- |
| CLP    |      1 byte |    ~87 KB |   ~5.1 MB | `unsigned char` per status code |
| HiGHS  |     4 bytes |   ~348 KB |  ~20.4 MB | `HighsInt` per status code      |

**Total memory footprint per workspace**:

| Component               | Per Thread (CLP) | Per Thread (HiGHS) | 48 Threads (HiGHS) |
| ----------------------- | ---------------: | -----------------: | -----------------: |
| Solver instance         |           ~15 MB |             ~15 MB |            ~720 MB |
| Solution buffers        |          ~815 KB |            ~815 KB |             ~38 MB |
| Per-stage basis cache   |          ~5.1 MB |           ~20.4 MB |            ~979 MB |
| **Total per workspace** |       **~21 MB** |         **~36 MB** |      **~1,737 MB** |

At 48 threads per rank (production configuration), the total workspace memory is under 2 GB per rank — well within acceptable bounds for production HPC nodes (384 GB RAM). The per-stage basis cache is the dominant cost with HiGHS due to its 4-byte status codes.

> **Note**: All sizing estimates use production-scale dimensions from `lp_sizing.py` (160 hydros, 130 thermals, 60 stages, 15K cut capacity). See [Production Scale Reference SS3](../overview/production-scale-reference.md) for the full configuration.

**Basis lifecycle**: The per-stage basis cache is the in-memory warm-start mechanism. For checkpoint/restart with reproducibility, the basis is serialized to FlatBuffers (`StageBasis` table in [Binary Formats SS3.1](../data-model/binary-formats.md)) at checkpoint boundaries. On resume, the checkpoint basis is loaded into the per-stage cache, restoring warm-start capability. The in-memory cache and the on-disk checkpoint are two lifecycle stages of the same data — the cache is the hot-path runtime structure, the checkpoint is the durable persistence format.

### 1.3 NUMA-Aware Initialization

Workspace initialization follows the first-touch NUMA allocation policy:

1. Each workspace **must be created by the thread that will own it** — not by the main thread. This ensures all allocations (solver internals, buffers) land on the NUMA node local to the owning thread.
2. During initialization, each thread pins to its assigned NUMA node, creates its solver instance, and fills all buffers with zeros (first-touch).
3. After initialization, the workspace array is immutable in structure — threads access their workspace by thread ID with no synchronization.

**Initialization sequence** (within a Rayon parallel scope):

1. Each thread computes its NUMA node: `numa_node = thread_id / threads_per_numa`
2. Pin to NUMA node (NUMA bind guard)
3. Create solver instance via `Highs_create()` or `Clp_newModel()`
4. Configure solver with SDDP-tuned settings (see solver impl specs SS4)
5. Allocate and zero-fill all buffers — solution buffers, RHS patch buffer, and per-stage basis cache (T slots, all initially empty) — ensuring first-touch on the local NUMA node
6. Store workspace in shared array at position `thread_id`

### 1.3a Workspace Lifecycle Summary

The preceding subsections define workspace creation (SS1.3), reuse (SS1.4--1.5), thread safety (SS1.8), and management (SS1.7) in separate sections. This subsection consolidates the complete workspace lifecycle into a single reference for implementers.

| Phase                    | Frequency                                     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Relevant Sections   |
| ------------------------ | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| **Creation**             | Once, at Initialization phase start           | Within a parallel region, each thread creates its own workspace on its NUMA node via first-touch allocation (SS1.3). The solver instance is created (`Highs_create()` or `Clp_newModel()`), all buffers are zero-filled, and the per-stage basis cache is initialized with `T` empty slots. The workspace manager allocates the workspace array. After creation, the workspace array is structurally immutable -- the number of workspaces equals the number of threads and does not change. | SS1.2, SS1.3, SS1.7 |
| **Per-iteration reuse**  | Many times, across all training iterations    | Each thread reuses its workspace for every LP solve. The per-stage basis cache (SS1.5) carries warm-start state across iterations -- the basis from solving stage `t` in iteration `i` is reused to warm-start stage `t` in iteration `i+1`. Solution buffers, RHS patch buffers, and reduced cost buffers are single-solve scratch buffers overwritten on each solve -- no per-iteration allocation occurs on the hot path.                                                                 | SS1.2, SS1.4, SS1.5 |
| **Per-stage transition** | Many times, within each forward/backward pass | When transitioning between stages, the workspace executes the full 6-step solve sequence (SS1.4): load StageLpCache, set scenario bounds, set basis, solve, extract solution, update statistics. Within a stage (multiple backward pass scenarios or batched forward passes at the same stage), step 1 is skipped since the structural LP and cuts are identical. This is the hot path.                                                                                                      | SS1.4, SS1.5        |
| **Destruction**          | Once, at Finalize phase end                   | Solver instances are freed via their C API destructor: `Highs_destroy()` for HiGHS, `Clp_deleteModel()` for CLP. All buffers (solution, RHS patch, reduced cost, per-stage basis cache) are deallocated. The workspace manager drops the workspace array. Destruction runs sequentially on the main thread after the last parallel region completes.                                                                                                                                         | SS1.2, SS1.7        |

> **Threading model interaction**: The workspace design assumes a **fixed-size thread pool** that is created at program start and persists until program exit. Workspaces are indexed by `thread_id` from this fixed pool. If the underlying runtime's work-stealing scheduler (e.g., rayon in test harnesses) moves a task to a different thread, the task must still access the workspace indexed by the executing thread's `rayon::current_thread_index()`, not by any task-local state. This is the key correctness invariant -- each physical thread always uses its own workspace, regardless of which logical task it is executing. See [Hybrid Parallelism SS1](../hpc/hybrid-parallelism.md) for the threading model specification.

> **Simulation reuse**: Workspace memory is **not** freed between the Training phase and the Simulation phase. The same workspaces are reused for simulation forward passes. This avoids re-initialization overhead (solver instance creation, NUMA-aware buffer allocation) and preserves the per-stage basis caches, which remain useful for simulation solves at the same stages the training phase visited. Phase boundaries are defined in [CLI and Lifecycle SS5.2](./cli-and-lifecycle.md).

**Cross-references**: [Hybrid Parallelism SS1](../hpc/hybrid-parallelism.md) (threading model, thread pool lifetime), [Training Loop SS4.3](./training-loop.md) (thread-trajectory affinity, parallel distribution), [CLI and Lifecycle SS5.2](./cli-and-lifecycle.md) (Initialization and Finalize phase boundaries).

### 1.4 Stage Solve Workflow

Each workspace provides a unified solve entry point that handles the full stage solve lifecycle. This is the main interface used by the forward and backward pass:

**Solve sequence**:

1. **Load StageLpCache** (if stage changed) — Bulk-load `StageLpCache[t]` ([Solver Abstraction SS11.4](./solver-abstraction.md)) into the solver via `load_model` (the Rust trait method; wraps the underlying C API calls `Highs_passModel` / `Clp_loadProblem`). The StageLpCache contains the complete LP for the stage (structural template + active cuts) in CSC format — no `addRows` call is needed.
2. **Set scenario bounds** — Write scenario-dependent values (incoming storage, AR lag fixing, noise fixing) into the pre-allocated `indices`, `lower`, and `upper` arrays of the RHS patch buffer, then apply directly via `set_row_bounds(indices, lower, upper)` ([Solver Interface Trait SS2.3](./solver-interface-trait.md)). The separate arrays are passed without tuple conversion, matching the SoA layout that both the caller and the C API naturally use.
3. **Set basis** (if warm-starting) — Look up the cached basis for the target stage from the per-stage basis cache. If a basis exists for this stage, apply it to the solver.
4. **Solve** — Call the solver. Retry logic is encapsulated within the solver implementation ([Solver Abstraction SS7](./solver-abstraction.md)).
5. **Extract solution** — Copy primal values, dual values, and reduced costs into pre-allocated buffers. Extract the basis and store it in the per-stage cache at the solved stage's slot, overwriting any previous basis for that stage.
6. **Update statistics** — Increment solve count, iteration count, timing.

Step 1 is skipped for **within-stage solves** (multiple backward pass scenarios at the same stage, or multiple forward passes at the same stage within a batch). Only steps 2–6 are needed, since the structural LP and cuts are identical.

When LP scaling is active, steps 1–2 and 5 are augmented with scaling transformations. See SS2.5 for the mapping.

### 1.5 Warm-Start Eligibility

Each workspace maintains a per-stage basis cache — an array of `T` basis slots, one per stage. The basis produced by solving stage `t` in iteration `i` is stored in slot `t` and reused to warm-start stage `t` in iteration `i+1`. This provides high-quality warm-starts because only the scenario-dependent RHS values change between iterations at the same stage; the structural LP and cut set are identical or nearly identical (new cuts may have been added).

| Condition                                  | Strategy   | Rationale                                                                                |
| ------------------------------------------ | ---------- | ---------------------------------------------------------------------------------------- |
| Basis exists in cache for the target stage | Warm-start | Basis from the previous iteration at this stage is near-optimal — only RHS changed.      |
| No cached basis for the target stage       | Cold start | First iteration, or basis was invalidated by a solver error. Solver starts from scratch. |

On solver error, the basis for the affected stage is invalidated (slot set to empty). The next solve at that stage will use cold start. Other stages' cached bases are unaffected.

**Backward pass reuse**: During the backward pass, a thread solves multiple scenarios at the same stage sequentially. The basis from the first scenario solve at stage `t` is stored in slot `t` and reused for subsequent scenarios at the same stage within the same backward pass. Since only the RHS changes between scenarios, this provides excellent warm-starts within a single backward sweep.

### 1.6 Solve Statistics

Each workspace accumulates per-thread statistics that are aggregated after the parallel region:

For the full list of per-solver statistics fields, see [Solver Interface Trait SS4.3](./solver-interface-trait.md). Key counters include solve counts, simplex iterations, retry histogram, timing breakdowns, and basis reuse metrics.

Aggregation across threads uses simple summation for all counters except max solve time (which uses the maximum across threads).

### 1.7 Workspace Manager

A workspace manager owns the array of all thread-local workspaces within an MPI rank:

- Creates one workspace per Rayon thread at initialization (within a parallel scope for first-touch NUMA).
- Provides indexed access by thread index — `workspace[thread_index]`. No locking required since each thread accesses only its own workspace.
- Provides aggregated statistics across all workspaces (called after parallel regions for logging).
- The workspace array is structurally immutable after initialization — the number of workspaces equals the number of Rayon threads and does not change.

> **Note on SolverPool alternative**: A pool-based pattern (borrow/return from a shared pool of N < threads solver instances) was considered and rejected. The persistent ownership pattern is preferred for SDDP because: (1) warm-starting requires per-stage basis affinity between iterations, which pools break; (2) the memory overhead of one workspace per thread (~36 MB × 48 threads ≈ 1,737 MB per rank) is acceptable on HPC nodes; (3) pool synchronization would add overhead on the hot path.

### 1.8 Thread Safety Invariants

The workspace design relies on clear ownership boundaries between shared and thread-local data:

| Data                     | Ownership                       | Access During Forward Pass | Access During Backward Pass      |
| ------------------------ | ------------------------------- | -------------------------- | -------------------------------- |
| StageLpCache (CSC)       | SharedRegion (NUMA-interleaved) | Read-only                  | Read-only                        |
| Cut pool metadata        | SharedRegion                    | Read-only                  | Write by single thread per stage |
| Cut pool activity bitmap | SharedRegion                    | Read-only                  | Write by single thread per stage |
| Solver instance          | Thread-local                    | Exclusive write            | Exclusive write                  |
| RHS patch buffer         | Thread-local                    | Exclusive write            | Exclusive write                  |
| Solution buffers         | Thread-local                    | Exclusive write            | Exclusive write                  |
| Per-stage basis cache    | Thread-local                    | Exclusive read/write       | Exclusive read/write             |

The key invariant: **no locking is required on the hot path**. StageLpCache is read-only during passes and updated between iterations by the leader rank (with `fence()` + barrier). Cut pool metadata is only modified during the backward pass, which is sequential per stage with a synchronization barrier at each stage boundary ([Training Loop SS4](./training-loop.md)). Each solver instance is exclusively owned by one thread.

### 1.9 Open Point — Structural LP Homogeneity

> **Open point — structural LP homogeneity across stages**: A potential performance optimization is to make all stage LPs structurally identical — same number of variables and constraints — even when system elements are not yet operative or have been decommissioned. Elements that are inactive at a given stage would be represented with zero bounds on their generation, exchange, and flow variables, preserving the LP structure. This would enable:
>
> - **Cheaper stage transitions**: The same CSC template structure can be reused across stages, with only coefficient/bound patches instead of different structural templates.
> - **Better basis reuse**: A basis from one stage could potentially warm-start a structurally identical stage, since the basis dimensions match exactly.
> - **Deeper solver integration**: With a fixed LP structure, the solver's factorization workspace can be sized once and reused without reallocation.
>
> **Trade-off**: This approach could reduce modeling flexibility (adding/removing system elements requires restructuring all stages) and may increase LP size at stages where many elements are inactive. It should be evaluated only if profiling shows that stage-template switching is a significant cost. Deferred until performance data is available.

### 1.10 Resolved — Cut Loading Cost

> **Resolved**: The StageLpCache resolves the cut loading bottleneck identified below. Cut coefficients are pre-assembled into the StageLpCache CSC between iterations by the leader rank. Stage transition reduces to a single `load_model` call that bulk-loads the complete LP (structural template + active cuts) — no per-thread `addRows` assembly. See [Solver Abstraction SS11.4](./solver-abstraction.md).
>
> The original analysis (pre-StageLpCache) identified cut loading via per-thread `addRows` as the dominant stage transition cost: ~5 ms loading vs ~2 ms solving per stage at production scale. This motivated the StageLpCache design which eliminates per-thread cut assembly entirely.

## 2. LP Scaling Specification

### 2.1 Motivation

Production SDDP LPs often suffer from numerical ill-conditioning due to:

- Variables spanning wide magnitudes (e.g., storage in hm³ at 10⁴ vs generation slacks at 10⁻²)
- Constraints with mixed coefficient magnitudes (water balance coefficients vs penalty bounds)
- The future cost variable θ dominating other variables in magnitude

LP scaling transforms the problem to improve solver numerical stability by normalizing coefficient magnitudes.

> **Scaling strategy resolved**: The minimal viable solver delegates LP scaling to the solver backend (`SolverAuto`). Cobre does not apply its own scaling. See [Solver Abstraction SS3](./solver-abstraction.md) for the full stakeholder decision and rationale. The scaling mechanics defined in SS2.2--2.4 below are retained as reference material for a future Cobre-managed scaling optimization pass.

### 2.2 Scaling Transformation

Given original LP: min c'x s.t. Ax ≤ b, x ≥ 0

Apply diagonal scaling matrices D_c (column/variable scaling) and D_r (row/constraint scaling):

| Quantity  | Original | Scaled            |
| --------- | -------- | ----------------- |
| Variables | x        | x̃ = D_c⁻¹ x       |
| Objective | c        | c̃ = D_c × c       |
| Matrix    | A        | Ã = D_r × A × D_c |
| RHS       | b        | b̃ = D_r × b       |

**Solution unscaling** (convert solver output back to physical units):

| Quantity      | Scaled → Physical |
| ------------- | ----------------- |
| Primal values | x = D_c × x̃       |
| Row duals     | π = D_r × π̃       |
| Reduced costs | rc = D_c⁻¹ × rc̃   |

### 2.3 Scaling Data Requirements

The scaling factors are computed once per stage from the stage template and stored alongside the template as shared, read-only data. They are not duplicated per workspace — all threads within a rank reference the same scaling factors for a given stage. The factors persist for the entire run:

| Field     | Type      | Description                                                                                                                                        |
| --------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| stage_id  | integer   | Which stage these factors apply to                                                                                                                 |
| col_scale | float64[] | Column scaling factors (one per variable). Multiplier convention: x_physical = col_scale[j] × x̃_scaled.                                            |
| row_scale | float64[] | Row scaling factors (one per constraint). Multiplier convention: row_scale[i] is applied to both sides of constraint i (Ã_i = row_scale[i] × A_i). |
| method    | enum      | Scaling method used: None, GeometricMean, Equilibration, SolverAuto                                                                                |
| applied   | boolean   | Whether scaling was actually applied (false if problem was already well-conditioned)                                                               |

Scaling factors are chosen so that the scaled matrix Ã = D_r × A × D_c has coefficient magnitudes closer to unity. Both D_r and D_c are diagonal matrices with row_scale and col_scale on their diagonals respectively.

**Scaling methods**:

| Method        | Description                                                | When to Use                        |
| ------------- | ---------------------------------------------------------- | ---------------------------------- |
| None          | No scaling applied                                         | Well-conditioned problems          |
| GeometricMean | Iterative geometric mean normalization of rows and columns | Recommended default for SDDP       |
| Equilibration | Scale so max abs(a_ij) = 1 in each row and column          | Alternative to geometric mean      |
| SolverAuto    | Delegate to solver's internal scaling                      | When Cobre does not manage scaling |

**Diagnostic metrics** (optional, for monitoring scaling quality):

| Metric          | Description                                         |
| --------------- | --------------------------------------------------- |
| max_coef_before | Maximum absolute coefficient before scaling         |
| min_coef_before | Minimum absolute nonzero coefficient before scaling |
| ratio_before    | max/min ratio before scaling (condition indicator)  |
| max_coef_after  | Maximum absolute coefficient after scaling          |
| min_coef_after  | Minimum absolute nonzero coefficient after scaling  |
| ratio_after     | max/min ratio after scaling (improvement indicator) |

### 2.4 Scaling Impact on Cut Coefficients

Cut coefficients must be stored in **physical (unscaled) units** for solver-agnostic portability. Scaling transformations are applied at solve time and immediately reversed.

**Transformation for cut coefficients**:

If the solver operates in scaled space and produces scaled duals π̃, the physical cut coefficients are:

β_physical = W' × D_r × π̃

where W is the technology matrix linking state variables to constraints.

When adding a cut to a scaled LP, the physical cut coefficients must be transformed to scaled space:

β̃[j] = β_physical[j] × col_scale[j]

This transformation is applied during the `addRows` step (SS1.4 step 2) and reversed when extracting solution duals (SS1.4 step 6).

**Key invariant**: The cut pool always stores cuts in physical units. Scaling is a transient transformation applied only within the solver workspace.

### 2.5 Scaling Integration with Stage Solve Workflow

When scaling is active, the stage solve workflow (SS1.4) is augmented at specific steps. The following table maps each scaling operation to the SS1.4 step it modifies:

| SS1.4 Step             | Scaling Augmentation                                                                                                                                                                            |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Load StageLpCache   | After loading, apply pre-computed scaling factors (D_r, D_c) to transform the LP into scaled space. Note: under StageLpCache, cuts are already in the CSC — scaling applies to the complete LP. |
| 2. Set scenario bounds | RHS values are patched in scaled space: multiply each bound by the corresponding row_scale[i].                                                                                                  |
| 3–4. Set basis, Solve  | No change — the basis is scale-independent (status codes, not values), and the solver operates entirely in scaled space.                                                                        |
| 5. Extract solution    | After extracting raw solver output, unscale: primal ×= D_c, duals ×= D_r, reduced costs ×= D_c⁻¹. Cut coefficients use the unscaled duals.                                                      |
| 6. Update statistics   | No change.                                                                                                                                                                                      |

> **Note (`SolverAuto` baseline)**: Under the minimal viable solver's `SolverAuto` scaling strategy ([Solver Abstraction SS3](./solver-abstraction.md)), all Cobre-managed scaling augmentation steps in the table above are **skipped**. The only active rows are "3--4. Set basis, Solve" and "6. Update statistics," which require no scaling transforms. Steps 1 (load StageLpCache), 2 (patch scenario values), and 5 (extract solution) proceed without any `col_scale`/`row_scale` multiplication or division -- the solver's internal scaling handles numerical conditioning transparently. This means the stage solve workflow in SS1.4 executes exactly as written, with no scaling-related augmentation, until Cobre-managed scaling is activated in a future optimization pass.

**Interaction with solver-internal scaling**: If Cobre manages its own scaling (augmentations above), solver-internal scaling should be disabled to avoid double-scaling. If Cobre delegates scaling to the solver (`SolverAuto` method), all scaling augmentations are skipped -- the solver handles scaling internally. See the solver-specific scaling configuration in [HiGHS Implementation SS4.2](./solver-highs-impl.md) and [CLP Implementation SS4.2](./solver-clp-impl.md).

## Cross-References

- [Solver Abstraction](./solver-abstraction.md) — Interface contract (SS4), LP layout convention (SS2), cut pool design (SS5), stage LP templates in CSC form (SS11), scaling decision (SS3)
- [HiGHS Implementation](./solver-highs-impl.md) — HiGHS-specific solver configuration, batch bound operations, scaling options
- [CLP Implementation](./solver-clp-impl.md) — CLP-specific solver configuration, mutable pointer patching, scaling options
- [LP Formulation](../math/lp-formulation.md) — Constraint structure that defines LP dimensions and row/column layout
- [Cut Management](../math/cut-management.md) — Cut generation algorithms that produce coefficients stored in physical units
- [Training Loop](./training-loop.md) — Forward/backward pass orchestration driving the stage solve workflow (SS1.4)
- [Hybrid Parallelism](../hpc/hybrid-parallelism.md) — Rayon threading model requiring thread-local solver workspaces
- [Memory Architecture](../hpc/memory-architecture.md) — NUMA topology and first-touch allocation policy for workspace buffers
- [Binary Formats](../data-model/binary-formats.md) — Cut pool CSR layout (SS3.4)
- [Configuration Reference](../configuration/configuration-reference.md) — Solver configuration parameters (tolerances, scaling method selection)
