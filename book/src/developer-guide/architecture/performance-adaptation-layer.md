# Performance Adaptation Layer

## Purpose

This spec defines the transformation layer that converts `cobre-core`'s clarity-first data model into `cobre-sddp`'s performance-adapted runtime representations. The [Internal Structures 1.1](../data-model/internal-structures.md) dual-nature design principle establishes that `cobre-core` optimizes for correctness and readability while `cobre-sddp` optimizes for cache locality, SIMD alignment, and zero-allocation hot paths. This spec fills the gap between that principle and the concrete performance-adapted types scattered across the solver, training loop, scenario generation, and binary format specs.

Specifically, this spec:

1. Defines a taxonomy of transformation strategies used across the adaptation layer
2. Inventories all performance-adapted types, their owning crate, source data, and authoritative spec
3. Specifies the initialization build order as a dependency graph
4. Maps entity fields from `cobre-core` structs to their performance consumers
5. States the contracts that the adaptation layer requires from `cobre-core` and guarantees to `cobre-sddp`

The adaptation layer executes entirely during the Initialization and Scenario Gen phases ([CLI and Lifecycle SS5.2a](./cli-and-lifecycle.md)). After these phases complete, all performance-adapted types are immutable for the duration of the Training and Simulation phases — no structural modifications, no heap allocations, no recomputation. The training loop operates exclusively on the adapted representations.

## 1. Transformation Taxonomy

The adaptation layer uses five distinct strategies to convert domain-oriented data into solver-ready representations. Each strategy addresses a specific performance concern. A single performance-adapted type may employ multiple strategies simultaneously.

### 1.1 Array-of-Structs to Struct-of-Arrays (AoS→SoA)

**What it does.** Transposes entity-oriented collections (`Vec<Hydro>`) into field-oriented parallel arrays (`Vec<f64>` per field). Each array holds one scalar field across all entities in canonical order.

**Why.** The SDDP hot path iterates over entities to patch RHS values, extract state, and assemble constraint rows. AoS layout scatters each field across cache lines (one `Hydro` struct spans hundreds of bytes; adjacent hydros' `max_storage_hm3` values are hundreds of bytes apart). SoA layout places all `max_storage_hm3` values in a contiguous `f64` slice, enabling sequential memory access and hardware prefetching.

**Where it appears.**

- PAR preprocessing arrays — `base[T][N]`, `coefficients[T][N][max_order]`, `scales[T][N]` ([Scenario Generation SS1.3](./scenario-generation.md))
- Pre-resolved bounds — stage×entity penalty and bound lookups ([Internal Structures 10–11](../data-model/internal-structures.md))
- Entity bounds for LP construction — `max_storage[N]`, `min_outflow[N]`, `max_turbined[N]`, etc., extracted from `Vec<Hydro>` for building stage template column/row bounds

**Extraction pattern.**

```rust
// AoS → SoA extraction at initialization (once, O(N) per field)
let max_storage: Vec<f64> = system.hydros.iter()
    .map(|h| h.max_storage_hm3)
    .collect();
let min_outflow: Vec<f64> = system.hydros.iter()
    .map(|h| h.min_outflow_m3s)
    .collect();
```

After extraction, the source `Vec<Hydro>` is still held by the `System` struct (it lives for the entire execution), but the training loop accesses only the SoA arrays.

### 1.2 Algebraic Absorption (Precomputation)

**What it does.** Evaluates algebraic expressions that combine multiple domain parameters into a single runtime value, eliminating redundant arithmetic from the hot path. The absorbed expression is typically a constant per (stage, entity) pair.

**Why.** Some LP coefficients and RHS values require combining 3–5 domain parameters that individually have clear physical meaning but, once combined, form a single constant. Computing this combination on every LP solve (millions of times) wastes cycles on arithmetic whose inputs never change within a stage.

**Where it appears.**

- `PrecomputedParLp.deterministic_base` — absorbs $b_{h,m(t)} = \mu_{m(t)} - \sum_\ell \psi_{m(t),\ell} \cdot \mu_{m(t-\ell)}$ so the hot path does one multiply-add (`base + sigma * noise`) instead of a loop ([Internal Structures 14](../data-model/internal-structures.md), [PAR Inflow Model 7](../math/par-inflow-model.md))
- Time conversion factor $\zeta$ per stage — absorbs $0.0036 \times \sum_k \tau_k$ from block durations ([Notation Conventions 3.1](../overview/notation-conventions.md))
- FPHA hyperplane coefficients — pre-fitted from geometry data during initialization, stored as dense `f64` arrays per (hydro, plane), consumed as LP constraint coefficients ([Hydro Production Models 2](../math/hydro-production-models.md))

**Correctness requirement.** The precomputed value must produce bit-for-bit identical results to evaluating the full expression inline. The theory spec (e.g., [PAR Inflow Model 7](../math/par-inflow-model.md)) defines the algebraic derivation; the precomputed struct is the implementation contract. A mismatch between the theory derivation and the precomputed expression is a bug — the theory spec is the oracle.

### 1.3 Index Flattening

**What it does.** Maps entity IDs (`EntityId(i32)`) to contiguous 0-based indices (`usize`) via the canonical ordering established during input loading ([Design Principles 3](../overview/design-principles.md)). All performance-adapted arrays use the flattened index as their access key.

**Why.** Entity IDs are sparse, user-assigned, and potentially non-contiguous (e.g., hydro IDs 101, 205, 310). LP variable positions, SoA array slots, and cut coefficient positions require dense 0-based indexing. The canonical sort (by ascending `EntityId`) produces a bijection: the entity at sort position `i` occupies index `i` in every performance array.

**Where it appears.** Everywhere. Every SoA array, every LP column/row formula, every cut coefficient, every state vector position uses the flattened index. The `StageIndexer` ([Training Loop SS5.5](./training-loop.md)) is the canonical expression of this mapping — it translates semantic names (storage, lags, theta) to `Range<usize>` positions in the LP solution vector.

**Mapping lifecycle.** The mapping is established once during input loading (canonical sort in [Input Loading Pipeline SS3](./input-loading-pipeline.md)) and is implicit thereafter — position `i` in any SoA array always refers to the entity at canonical position `i`. No explicit lookup table is needed at runtime because the canonical ordering is a structural invariant, not a runtime query.

### 1.4 Layout Reshaping for Access Pattern

**What it does.** Arranges data so that the dominant access pattern touches contiguous memory. This is distinct from SoA (section 1.1), which transposes field orientation. Layout reshaping concerns the ordering of elements _within_ a single array to match how the algorithm iterates over them.

**Why.** The hardware prefetcher performs best on sequential or strided access. An array whose layout matches the iteration order gets free prefetching. An array whose layout opposes the iteration order suffers cache misses on every access.

**Where it appears.**

- **LP column layout** — State variables (storage, then lags) are placed at the column prefix so that state extraction is a single `memcpy` from `primal[0..n_state]` ([Solver Abstraction SS2.1](./solver-abstraction.md)). Decision variables follow. This layout was _designed_ to make the hot-path access pattern (extract state, extract duals for cut coefficients) contiguous.
- **LP row layout** — Fixing constraints (storage, then lag) placed at the row prefix so that dual extraction for cut coefficients is `dual[0..n_state]`, with row-column symmetry: row `r`'s dual is the cut coefficient for state variable at column `r` ([Training Loop SS5.5.1](./training-loop.md)).
- **Opening tree tensor** — `[stages × openings × dim]` with stage as the outer dimension. The backward pass iterates stages in reverse, accessing all openings at a given stage — contiguous memory ([Scenario Generation SS2.3](./scenario-generation.md)).
- **Noise cache** — Scenario-major layout `[scenario × stage × entity]` for forward pass sequential access within a trajectory ([Scenario Generation SS5.1](./scenario-generation.md)). Trade-off: backward pass accesses non-contiguously, but the cache fits in L3 and the LP solve dominates latency (see scenario-generation SS5.1 rationale).
- **Cut pool coefficients** — Intercepts separated from coefficient vectors to avoid polluting cache during coefficient copy loops; 64-byte alignment for prefetch boundary matching ([Binary Formats 3.4](../data-model/binary-formats.md)).

### 1.5 Temporal Flattening

**What it does.** Pre-resolves season-indexed or period-indexed parameters into stage-indexed arrays, eliminating season-lookup indirection at runtime. The domain model naturally uses seasons (12 months or 52 weeks) because physical parameters are seasonal. The solver needs stage-indexed values (60–120 stages) because it iterates stage-by-stage.

**Why.** A season lookup (`season = stage_to_season_map[t]; value = seasonal_values[season]`) is a dependent load — the CPU cannot prefetch the value until the season index is resolved. At worst-case scale (160 hydros × 120 stages × millions of iterations), the cumulative cost of dependent loads is significant. Pre-resolving to `value = stage_values[t][h]` eliminates the indirection.

**Where it appears.**

- `PrecomputedParLp` arrays — All three arrays (`psi`, `deterministic_base`, `sigma`) are indexed `[stage][hydro]`, not `[season][hydro]`, despite the underlying PAR model being defined per-season ([Internal Structures 14](../data-model/internal-structures.md))
- PAR preprocessing arrays in scenario generation — `base[T][N]`, `coefficients[T][N][max_order]`, `scales[T][N]` use stage as the outer dimension ([Scenario Generation SS1.3](./scenario-generation.md))
- Pre-resolved penalties — stage-varying penalty costs resolved from the three-tier cascade (global → entity → stage override) into a per-(entity, stage) lookup ([Internal Structures 10](../data-model/internal-structures.md))
- Pre-resolved bounds — stage-varying entity bounds resolved from base values with sparse stage overrides applied ([Internal Structures 11](../data-model/internal-structures.md))

**Memory cost.** Temporal flattening trades memory for speed. A seasonal PAR model stores `12 × N` values; the stage-indexed version stores `T × N` values (at worst-case scale: 120 × 160 = 19,200 vs. 12 × 160 = 1,920 — a 10× expansion; at the production baseline of 60 stages: 60 × 160 = 9,600 vs. 12 × 160 = 1,920 — a 5× expansion). At 8 bytes per `f64`, the absolute cost is negligible (< 1 MB for all temporally flattened arrays combined).

## 2. Performance-Adapted Type Inventory

Every performance-adapted type in the Cobre ecosystem is listed below, grouped by owning crate and lifecycle phase. The "Source Data" column identifies which `cobre-core` types feed each adapted type. The "Strategies" column references the taxonomy in section 1.

### 2.1 Types Built During Initialization Phase

| Type                                                                                                                                 | Owner                                  | Source Data                                                                                                    | Strategies                                                                                                                                                                                                               | Authoritative Spec                                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `StageTemplate` (CSC arrays: `col_starts`, `row_indices`, `values`, `col_lower`, `col_upper`, `row_lower`, `row_upper`, `objective`) | `cobre-solver` (built by `cobre-sddp`) | `System` entities, `Stage`/`Block` definitions, pre-resolved bounds and penalties, `PrecomputedParLp.psi`      | AoS→SoA (entity fields → CSC coefficient values), Index flattening (entity ID → column/row position), Layout reshaping (state prefix, fixing constraint prefix), Temporal flattening (stage-varying bounds pre-resolved) | [Solver Abstraction SS11.1](./solver-abstraction.md), [Solver Interface Trait SS4.4](./solver-interface-trait.md) |
| `StageIndexer` (column/row range map)                                                                                                | `cobre-solver`                         | System dimensions (hydro count, max PAR order)                                                                 | Index flattening (semantic names → `Range<usize>`)                                                                                                                                                                       | [Training Loop SS5.5](./training-loop.md)                                                                         |
| `SolverWorkspace` (solver instance, pre-allocated buffers, per-stage basis cache)                                                    | `cobre-solver`                         | LP dimensions from `StageTemplate`, stage count                                                                | — (allocation, not transformation)                                                                                                                                                                                       | [Solver Workspaces SS1](./solver-workspaces.md)                                                                   |
| `PrecomputedParLp` (`psi`, `deterministic_base`, `sigma`)                                                                            | `cobre-sddp`                           | `ParModel` (seasonal means, AR coefficients, residual std devs), season-to-stage mapping                       | Algebraic absorption (`deterministic_base`), Temporal flattening (season → stage indexing)                                                                                                                               | [Internal Structures 14](../data-model/internal-structures.md), [PAR Inflow Model 7](../math/par-inflow-model.md) |
| `StageLpCache` (complete pre-assembled LP per stage in CSC format)                                                                   | `cobre-solver` (built by `cobre-sddp`) | `StageTemplate` CSC + `CutPool` slot structure + column/row bounds + objective                                 | Layout reshaping (structural template + cut coefficients → unified CSC with 15K pre-allocated cut slots)                                                                                                                 | [Solver Abstraction SS11.4](./solver-abstraction.md)                                                              |
| `CutPool` (slot-based, intercept/activity bitmap/metadata only)                                                                      | `cobre-sddp`                           | Pre-allocated from config (`max_cuts_per_stage`, `n_state`); warm-start cuts from FlatBuffers policy on resume | Index flattening (slot indices). Coefficients absorbed into StageLpCache CSC; pool retains metadata (~12 MB across 60 stages at 15K capacity)                                                                            | [Cut Management Impl SS1](./cut-management-impl.md), [Binary Formats 3.4](../data-model/binary-formats.md)        |
| FPHA hyperplane coefficients                                                                                                         | `cobre-sddp`                           | Hydro geometry, pre-fitted planes from `hydro_extensions`                                                      | Algebraic absorption (geometry → hyperplane coefficients)                                                                                                                                                                | [Hydro Production Models 2](../math/hydro-production-models.md)                                                   |
| Cascade topology arrays (upstream indices, downstream indices per hydro)                                                             | `cobre-sddp`                           | `Hydro.downstream_id`, `CascadeTopology`                                                                       | Index flattening (entity IDs → positional indices), AoS→SoA                                                                                                                                                              | [Internal Structures 5](../data-model/internal-structures.md)                                                     |
| Bus-entity membership arrays (which hydros/thermals/lines connect to each bus)                                                       | `cobre-sddp`                           | `Hydro.bus_id`, `Thermal.bus_id`, `Line.source_bus_id`/`target_bus_id`                                         | Index flattening, AoS→SoA                                                                                                                                                                                                | [Internal Structures 1.2](../data-model/internal-structures.md)                                                   |
| Pre-resolved bounds (`[stage][entity]` lookup)                                                                                       | `cobre-core` (loaded by `cobre-io`)    | Base entity bounds + sparse stage overrides from `hydro_bounds`/`thermal_bounds`                               | Temporal flattening                                                                                                                                                                                                      | [Internal Structures 11](../data-model/internal-structures.md)                                                    |
| Pre-resolved penalties (`[stage][entity]` lookup)                                                                                    | `cobre-core` (loaded by `cobre-io`)    | Global defaults + entity overrides + stage overrides from `penalties.json`                                     | Temporal flattening                                                                                                                                                                                                      | [Internal Structures 10](../data-model/internal-structures.md)                                                    |

### 2.2 Types Built During Scenario Gen Phase

| Type                                                                  | Owner              | Source Data                                                              | Strategies                                                                       | Authoritative Spec                                    |
| --------------------------------------------------------------------- | ------------------ | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------- | ----------------------------------------------------- |
| PAR preprocessing arrays (`base`, `coefficients`, `scales`, `orders`) | `cobre-stochastic` | `ParModel` parameters, season definitions, stage definitions             | AoS→SoA, Temporal flattening (season → stage), Algebraic absorption              | [Scenario Generation SS1.3](./scenario-generation.md) |
| Spectrally-decomposed correlation matrices                            | `cobre-stochastic` | Correlation matrices from `correlation.json` or estimated from residuals | Algebraic absorption ($\Sigma = V\operatorname{diag}(\lambda)V^T$ computed once) | [Scenario Generation SS2.1](./scenario-generation.md) |
| Opening tree tensor (`[stages × openings × dim]`)                     | `cobre-stochastic` | Derived RNG seeds, spectral factors, PAR preprocessing arrays            | Layout reshaping (stage-outer for backward pass access)                          | [Scenario Generation SS2.3](./scenario-generation.md) |

### 2.3 Types That Are Not Transformations

For completeness, these types are sometimes discussed alongside performance-adapted views but are not transformations of `cobre-core` data — they are runtime artifacts produced by the algorithm:

| Type                             | Owner        | Nature                                                     |
| -------------------------------- | ------------ | ---------------------------------------------------------- |
| State vectors (`[f64; n_state]`) | `cobre-sddp` | Extracted from LP primal solution during forward pass      |
| Cut coefficients                 | `cobre-sddp` | Extracted from LP dual solution during backward pass       |
| Convergence history              | `cobre-sddp` | Accumulated per-iteration bound statistics                 |
| Simulation results               | `cobre-sddp` | Extracted from LP solutions during simulation forward pass |

## 3. Initialization Build Order

The performance-adapted types form a dependency graph. Each type requires certain inputs to be available before construction. The build order below is the topological sort of this graph, grouped by lifecycle phase.

### 3.1 Dependency Graph

```
System (from Validation phase, broadcast to all ranks)
  │
  ├──→ Pre-resolved bounds [Internal Structures §11]
  │      (base entity bounds + stage overrides → [stage][entity] lookup)
  │
  ├──→ Pre-resolved penalties [Internal Structures §10]
  │      (global → entity → stage cascade → [stage][entity] lookup)
  │
  ├──→ Cascade topology arrays
  │      (Hydro.downstream_id → flattened upstream/downstream index arrays)
  │
  ├──→ Bus-entity membership arrays
  │      (entity bus_id fields → per-bus entity index lists)
  │
  ├──→ PrecomputedParLp [Internal Structures §14]
  │      (ParModel + season-stage mapping → psi, deterministic_base, sigma)
  │
  ├──→ FPHA hyperplane coefficients
  │      (hydro geometry → fitted plane arrays)
  │
  └──→ StageIndexer [Training Loop §5.5]
         (system dimensions → Range<usize> index map)
         │
         └──→ StageTemplate [Solver Abstraction §11.1]
                (ALL of the above + LP formulation rules → CSC arrays)
                │
                ├──→ SolverWorkspace [Solver Workspaces §1]
                │      (LP dimensions from template → allocated buffers + solver instance)
                │
                ├──→ CutPool [Cut Management Impl §1]
                │      (n_state from indexer, max_cuts from config → metadata-only allocation)
                │      │
                │      └──→ (optional) Warm-start cut loading from FlatBuffers policy
                │
                └──→ StageLpCache [Solver Abstraction §11.4]
                       (StageTemplate CSC + empty cut slots → complete LP per stage)
                       (SharedRegion with NUMA-interleaved allocation; ~22.3 GB)

─── Initialization / Scenario Gen boundary ───

PrecomputedParLp + System
  │
  ├──→ PAR preprocessing arrays [Scenario Generation §1.3]
  │      (seasonal PAR params → stage-indexed SoA arrays)
  │
  ├──→ Spectral decomposition [Scenario Generation §2.1]
  │      (correlation matrices → symmetric spectral factors)
  │
  └──→ Opening tree [Scenario Generation §2.3]
         (RNG seeds + spectral factors + PAR arrays → dense tensor)
```

### 3.2 Parallelism During Build

Most build steps are sequential (single-threaded on each rank), with two exceptions:

1. **Solver workspace allocation** must happen inside a parallel region so that first-touch NUMA policy places each workspace's buffers on the owning thread's NUMA node ([Solver Workspaces SS1.3](./solver-workspaces.md)).
2. **Opening tree generation** is embarrassingly parallel across stages — each stage's openings are independent once the spectral factors exist.

All other construction is sequential because the cost is negligible relative to training (one-time $O(T \times N)$ arithmetic vs. millions of LP solves).

### 3.3 Build Order as Implementation Checklist

The following sequence is a linearization of the dependency graph. Steps at the same indent level are independent and may execute in any order.

```
INITIALIZATION PHASE:
  1. Receive broadcast System from rank 0
  2. Build pre-resolved bounds (§11) and penalties (§10)
  3. Build cascade topology arrays
  4. Build bus-entity membership arrays
  5. Build PrecomputedParLp (§14) — needs ParModel + stage definitions
  6. Fit FPHA hyperplanes (if computed source) — needs hydro geometry
  7. Build StageIndexer for each stage — needs system dimensions only
  8. Build StageTemplate for each stage — needs everything from steps 2–7
  8a. Assemble initial StageLpCache from StageTemplate + empty cut slots (SharedRegion, NUMA-interleaved)
  9. Allocate SolverWorkspaces (in parallel region) — needs LP dimensions from step 8
  10. Pre-allocate CutPool — metadata-only allocation (intercepts + activity bitmap); needs n_state from step 7
  11. (Warm-start only) Load cuts from FlatBuffers policy into CutPool and StageLpCache

SCENARIO GEN PHASE:
  12. Build PAR preprocessing SoA arrays — needs PrecomputedParLp from step 5
  13. Decompose correlation matrices — needs correlation data from System
  14. Generate opening tree — needs steps 12–13
```

Steps 2–6 depend only on `System` and are independent of each other. Steps 7–8 depend on 2–6. Step 8a depends on 8. Steps 9–11 depend on 8 (and 8a for StageLpCache-aware warm-start). Steps 12–14 depend on 5 and are independent of 8–11.

## 4. Entity Data Flow

This section maps individual entity fields from `cobre-core` structs to their ultimate consumers in the performance layer. The mapping answers the question: "when I change field X in the `Hydro` struct, which performance-adapted types need to be rebuilt?"

### 4.1 Hydro Fields

The `Hydro` entity ([Internal Structures 1.9.4](../data-model/internal-structures.md)) is the most complex, with ~20 fields feeding multiple LP elements.

**Reservoir and flow bounds → StageTemplate column/row bounds:**

| Field               | LP Element                               | Template Array                  | How It Enters                                                                                         |
| ------------------- | ---------------------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `min_storage_hm3`   | Column lower bound on $v_h$              | `col_lower[h]`                  | Direct copy (with stage override from pre-resolved bounds)                                            |
| `max_storage_hm3`   | Column upper bound on $v_h$              | `col_upper[h]`                  | Direct copy (with stage override)                                                                     |
| `min_outflow_m3s`   | Row lower bound on outflow constraint    | `row_lower[outflow_row(h)]`     | Direct copy (with stage override). Soft — slack variable created with `outflow_violation_below_cost`. |
| `max_outflow_m3s`   | Row upper bound on outflow constraint    | `row_upper[outflow_row(h)]`     | Direct copy when `Some`; `+∞` when `None`. Soft — slack variable.                                     |
| `min_turbined_m3s`  | Column lower bound on $q_{h,k}$          | `col_lower[turbined_col(h, k)]` | Direct copy. Soft — slack variable.                                                                   |
| `max_turbined_m3s`  | Column upper bound on $q_{h,k}$          | `col_upper[turbined_col(h, k)]` | Direct copy. Hard bound.                                                                              |
| `min_generation_mw` | Row lower bound on generation constraint | `row_lower[gen_row(h, k)]`      | Direct copy. Soft — slack variable.                                                                   |
| `max_generation_mw` | Row upper bound on generation constraint | `row_upper[gen_row(h, k)]`      | Direct copy. Hard bound.                                                                              |

**Generation model → StageTemplate constraint coefficients:**

| Field                                                                | LP Element                                                           | Template Array                                     | How It Enters                                                                                                                |
| -------------------------------------------------------------------- | -------------------------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `generation_model: ConstantProductivity { productivity_mw_per_m3s }` | Coefficient in generation constraint linking $g_{h,k}$ and $q_{h,k}$ | `values[...]` in CSC                               | Written as LP coefficient: $g_{h,k} = \rho_h \cdot q_{h,k}$ becomes row with coefficients $[1, -\rho_h]$ on $[g, q]$ columns |
| `generation_model: Fpha`                                             | FPHA hyperplane constraints                                          | Multiple rows in CSC per hydro, one per hyperplane | Each plane: $g_{h,k} \leq a_m \cdot q_{h,k} + b_m \cdot v_h + c_m$. Coefficients from FPHA fitting.                          |

**Topology → Cascade arrays and water balance constraint structure:**

| Field           | Performance Consumer                                        | How It Enters                                                                                                               |
| --------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `bus_id`        | Bus-entity membership arrays                                | Determines which bus's load balance constraint includes this hydro's generation                                             |
| `downstream_id` | Cascade topology arrays, water balance constraint structure | Determines outflow coupling: this hydro's outflow appears as inflow to the downstream hydro in the water balance constraint |

**Penalties → StageTemplate objective coefficients:**

| Field (via `HydroPenalties`)   | LP Element                                              | Template Array                    |
| ------------------------------ | ------------------------------------------------------- | --------------------------------- |
| `spillage_cost`                | Objective coefficient on spillage variable $s_{h,k}$    | `objective[spillage_col(h, k)]`   |
| `storage_violation_below_cost` | Objective coefficient on storage slack $\sigma^{v-}_h$  | `objective[storage_slack_col(h)]` |
| All other penalty fields       | Objective coefficients on corresponding slack variables | `objective[slack_col(...)]`       |

Pre-resolved penalties ([Internal Structures 10](../data-model/internal-structures.md)) are already temporally flattened — the stage template builder reads the penalty for the target stage directly without cascade resolution.

**PAR model → PrecomputedParLp and LP constraints:**

| Source                     | Performance Consumer                                                                                | Transformation                                                                                     |
| -------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `ParModel.seasonal_means`  | `PrecomputedParLp.deterministic_base`                                                               | Algebraic absorption: $b_{h,m(t)} = \mu_{m(t)} - \sum_\ell \psi_{m(t),\ell} \cdot \mu_{m(t-\ell)}$ |
| `ParModel.ar_coefficients` | `PrecomputedParLp.psi` → StageTemplate AR dynamics constraint coefficients                          | Temporal flattening (season → stage), then written as LP constraint coefficients                   |
| `ParModel.residual_std`    | `PrecomputedParLp.sigma` → hot-path RHS patching: $\text{RHS}_{h,t} = b + \sigma \cdot \varepsilon$ | Temporal flattening                                                                                |

**Fields consumed only at initialization (not in any performance array):**

| Field                                        | Usage                                                                                                                                                                            |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`                                         | Canonical sorting key. After sort, the positional index replaces the ID for all performance access.                                                                              |
| `name`                                       | Logging, error messages, output column headers. Not in any LP or performance array.                                                                                              |
| `entry_stage_id`, `exit_stage_id`            | Entity lifecycle filtering. Determines which stages include this hydro in the LP. Not a runtime value — it gates which per-stage template includes the hydro.                    |
| `tailrace`, `hydraulic_losses`, `efficiency` | Consumed during FPHA hyperplane fitting (initialization). Not stored in performance arrays — their effect is absorbed into the fitted hyperplane coefficients.                   |
| `evaporation_coefficients_mm`                | Consumed during stage template construction for the evaporation constraint. Combined with surface area to produce a per-stage evaporation bound. Absorbed into the LP row bound. |
| `filling`                                    | Consumed during stage template construction to adjust constraints for filling-period stages.                                                                                     |

### 4.2 General Pattern

The Hydro mapping above illustrates the general pattern that applies to all entity types:

1. **Scalar bounds** (min/max values) become column or row bounds in the `StageTemplate` CSC arrays, potentially with stage overrides from pre-resolved bounds.
2. **Costs and penalties** become objective coefficients in the `StageTemplate`.
3. **Topology references** (`bus_id`, `downstream_id`, `source_bus_id`/`target_bus_id`) become index-flattened membership arrays and determine constraint structure.
4. **Identity fields** (`id`, `name`) are consumed at initialization for sorting and logging; they do not appear in performance arrays.
5. **Lifecycle fields** (`entry_stage_id`, `exit_stage_id`) gate per-stage entity inclusion; they are not runtime values.
6. **Model parameters** (productivity, AR coefficients, thermal cost curve segments) become LP constraint coefficients or are absorbed via precomputation.

The same analysis applies to `Thermal` (cost curve segments → piecewise objective, capacity → column bounds, bus_id → membership), `Line` (capacity → column bounds, losses → constraint coefficient, bus IDs → load balance structure), `Bus` (deficit segments → multiple deficit variables with piecewise costs), and other entity types. The mapping for each follows the same six categories above.

### 4.3 What Changes Require Rebuilding What

| Change in `cobre-core`                                     | Affected Performance Types                                                                                                                         | Rebuild Scope                                                                         |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Entity field value (e.g., `max_storage_hm3` for one hydro) | `StageTemplate` for stages where the hydro is active                                                                                               | Per-stage template rebuild. Does not affect `StageIndexer`, `CutPool`, or PAR arrays. |
| Entity count (add/remove a hydro)                          | All types: `StageIndexer`, `StageTemplate`, `CutPool` (n_state changes), `PrecomputedParLp`, PAR arrays, cascade arrays, membership arrays         | Full rebuild. This changes LP dimensions.                                             |
| Stage count or block structure                             | `StageTemplate` (all stages), `PrecomputedParLp`, PAR arrays, opening tree, `SolverWorkspace` (basis cache size), `CutPool` (per-stage pool count) | Full rebuild.                                                                         |
| PAR model coefficients only                                | `PrecomputedParLp`, PAR preprocessing arrays, opening tree                                                                                         | Partial rebuild. `StageTemplate` unaffected if AR constraint structure unchanged.     |
| Penalty value only                                         | `StageTemplate` objective coefficients for affected stages                                                                                         | Per-stage template rebuild of objective array only.                                   |

In practice, the initialization is fast enough (< 1 second for production-scale systems) that partial rebuilds are an unnecessary optimization. The table above documents dependencies for reasoning about correctness, not for implementing incremental rebuild.

## 5. Contracts and Invariants

### 5.1 What the Adaptation Layer Requires from `cobre-core`

| Requirement                                                                                                  | Guaranteed By                                      | Spec Reference                                                                                                     |
| ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| All entity collections sorted by ascending `EntityId` (canonical ordering)                                   | Input loading pipeline, canonicalization step      | [Design Principles 3](../overview/design-principles.md), [Input Loading Pipeline SS3](./input-loading-pipeline.md) |
| All defaults resolved — no `None` where a default exists                                                     | Input loading pipeline, default resolution step    | [Internal Structures 1.9](../data-model/internal-structures.md) (Resolved annotation)                              |
| All cross-references valid — `downstream_id` points to an existing hydro, `bus_id` points to an existing bus | Input loading pipeline, cross-reference validation | [Validation Architecture SS2](./validation-architecture.md)                                                        |
| All stage overrides applied — pre-resolved bounds and penalties available as `[stage][entity]` lookups       | Input loading pipeline, resolution steps           | [Internal Structures 10–11](../data-model/internal-structures.md)                                                  |
| `System` struct immutable for the duration of Initialization + Scenario Gen + Training + Simulation          | Ownership model: `System` is shared via `&System`  | [Internal Structures 1.3](../data-model/internal-structures.md)                                                    |
| `ParModel` parameters pass validation (positive residual variance, AR polynomial stability)                  | Input loading pipeline, model validation           | [PAR Inflow Model 6](../math/par-inflow-model.md)                                                                  |

### 5.2 What the Adaptation Layer Guarantees to the Training Loop

| Guarantee                                                                             | Implementation                                                                                                                                                                                             | Spec Reference                                                                                                          |
| ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| All performance-adapted types are immutable after construction (except StageLpCache)  | Built during Initialization/Scenario Gen; no mutation API exposed. StageLpCache is updated between iterations by leader rank via SharedRegion (fence + barrier) — read-only during forward/backward passes | [CLI and Lifecycle SS5.2a](./cli-and-lifecycle.md), [Solver Abstraction SS11.4](./solver-abstraction.md)                |
| No heap allocation during Training/Simulation phases from adapted types               | All buffers pre-allocated; `Vec` capacities set at construction                                                                                                                                            | [Ecosystem Guidelines 6](../overview/ecosystem-guidelines.md), [Memory Architecture 3.3](../hpc/memory-architecture.md) |
| SoA arrays and LP column/row indices use identical canonical ordering                 | Both derived from the same canonical sort                                                                                                                                                                  | [Design Principles 3](../overview/design-principles.md)                                                                 |
| State extraction is a contiguous `memcpy` from `primal[0..n_state]`                   | LP column layout places state variables at prefix                                                                                                                                                          | [Solver Abstraction SS2.1](./solver-abstraction.md)                                                                     |
| Dual extraction for cut coefficients is a contiguous `memcpy` from `dual[0..n_state]` | LP row layout places fixing constraints at prefix with row-column symmetry                                                                                                                                 | [Solver Abstraction SS2.2](./solver-abstraction.md), [Training Loop SS5.5.1](./training-loop.md)                        |
| Cut coefficient dot products operate on 64-byte-aligned `f64` arrays                  | Allocation uses `Layout::from_size_align(..., 64)`                                                                                                                                                         | [Solver Abstraction SS2.5](./solver-abstraction.md), [Training Loop SS5.1.1](./training-loop.md)                        |
| `StageTemplate` CSC arrays are shared read-only across all threads within a rank      | Templates are `Send + Sync`; no interior mutability                                                                                                                                                        | [Solver Abstraction SS11.1](./solver-abstraction.md)                                                                    |
| `StageIndexer` produces identical results on all MPI ranks                            | LP structure depends only on `System` (identical on all ranks)                                                                                                                                             | [Training Loop SS5.5.1](./training-loop.md)                                                                             |
| Thread-local `SolverWorkspace` buffers are NUMA-local to the owning thread            | First-touch allocation inside parallel region                                                                                                                                                              | [Solver Workspaces SS1.3](./solver-workspaces.md), [Memory Architecture 2.1](../hpc/memory-architecture.md)             |
| Pre-resolved penalties and bounds are O(1) per (entity, stage) lookup                 | Materialized as flat `[stage][entity]` arrays during loading                                                                                                                                               | [Internal Structures 10–11](../data-model/internal-structures.md)                                                       |

### 5.3 Boundary Location

The adaptation boundary is the `cobre-sddp` initialization function. Its signature (defined in [Internal Structures 1.3](../data-model/internal-structures.md)):

```rust
cobre_sddp::train(system: &System, config: &TrainingConfig, comm: &C) -> Result<TrainingResult, TrainError>
```

Inside `train`, the first operation (before entering the iteration loop) is to build all performance-adapted views from `&System`. After this build phase, the iteration loop never accesses `System` entity fields directly — all hot-path data comes from the adapted types.

The boundary is one-way: data flows from `cobre-core` types into performance-adapted types during initialization, and never flows back. The adapted types are consumed and discarded when `train` returns. Runtime artifacts (cuts, convergence history, simulation results) are written to output files, not back into `System`.

## Cross-References

- [Internal Structures 1.1](../data-model/internal-structures.md) — Dual-nature design principle (the high-level contract this spec elaborates)
- [Internal Structures 10–11](../data-model/internal-structures.md) — Pre-resolved penalties and bounds (temporal flattening at the cobre-io/cobre-core boundary)
- [Internal Structures 14](../data-model/internal-structures.md) — `PrecomputedParLp` (algebraic absorption + temporal flattening)
- [Solver Abstraction SS2](./solver-abstraction.md) — LP column and row layout convention (layout reshaping)
- [Solver Abstraction SS11.1](./solver-abstraction.md) — `StageTemplate` construction and CSC representation
- [Solver Abstraction SS11.4](./solver-abstraction.md) — `StageLpCache` design, sizing, SharedRegion ownership, update/read contracts
- [Solver Workspaces SS1](./solver-workspaces.md) — Thread-local workspace allocation and lifecycle
- [Training Loop SS5.5](./training-loop.md) — `StageIndexer` definition and usage
- [Scenario Generation SS1.3](./scenario-generation.md) — PAR preprocessing SoA arrays
- [Scenario Generation SS2.3](./scenario-generation.md) — Opening tree tensor layout
- [Binary Formats 3.4](../data-model/binary-formats.md) — Cut pool memory layout requirements
- [Cut Management Impl SS1](./cut-management-impl.md) — Cut pool runtime structure
- [CLI and Lifecycle SS5.2a](./cli-and-lifecycle.md) — Phase ordering and initialization sequencing
- [Design Principles 3](../overview/design-principles.md) — Declaration order invariance (canonical ordering foundation)
- [Ecosystem Guidelines 6](../overview/ecosystem-guidelines.md) — Performance principles (cache locality, SIMD, zero allocation)
- [Memory Architecture 2–3](../hpc/memory-architecture.md) — NUMA placement, cache line alignment, zero-allocation enforcement
- [PAR Inflow Model 7](../math/par-inflow-model.md) — Algebraic derivation of precomputed PAR components
- [Notation Conventions 3.1](../overview/notation-conventions.md) — Time conversion factor derivation
- [Design Principles 7](../overview/design-principles.md) — Decision to use f64 with unit suffixes; the adaptation boundary is the cast point identified in that section
