# Internal Structures

## Purpose

This spec defines the logical in-memory data model that the SDDP solver requires at runtime. It describes _what_ data the solver must hold in memory and _why_, without prescribing implementation types or data structures. The actual implementation (struct layout, indexing strategy, memory layout) will emerge during development.

These structures are the result of loading, validating, and resolving all input files into a unified, ready-to-use representation. They differ from input schemas in several ways:

| Concern    | Input Schemas                                | Internal Structures                      |
| ---------- | -------------------------------------------- | ---------------------------------------- |
| Scope      | One file at a time                           | Unified, cross-referenced                |
| Defaults   | May be absent (use global/entity default)    | All defaults resolved to concrete values |
| Penalties  | Three-tier cascade (global → entity → stage) | Pre-resolved per (entity, stage)         |
| Bounds     | Base values with sparse stage overrides      | Pre-resolved per (entity, stage)         |
| Validation | Deferred or partial                          | Complete — all cross-references verified |
| Ordering   | User-defined (arbitrary)                     | Canonical (sorted by ID)                 |

For input file schemas, see [Input System Entities](input-system-entities.md), [Input Hydro Extensions](input-hydro-extensions.md), [Input Scenarios](input-scenarios.md), [Input Constraints](input-constraints.md). For binary format choices (LP subproblems, cuts), see [Binary Formats](binary-formats.md).

### Role in Program Lifecycle

These internal structures serve two distinct purposes in the program, each with different performance characteristics:

**1. LP Definition Construction (initialization — built once)**

The system entities (§1–§9), pre-resolved penalties (§10), pre-resolved bounds (§11), generic constraints (§14), block factors (§12), and AR models (§13) are all loaded and resolved during program initialization — whether for training or simulation. From these resolved structures plus algorithm configuration (number of iterations, forward passes for cut preallocation, etc.), the program constructs a set of LP definitions, one per (stage, block) pair. These LP definitions are built once and reused throughout the algorithm. Since this happens once at startup, **data clarity and correctness are more important than performance** in this phase.

**2. Scenario Generation Pipeline (runtime — performance-critical)**

The scenario pipeline (§13) is exercised repeatedly during both training and simulation. Each forward pass requires sampling from standard normal distributions, applying spectrally-decomposed correlation matrices, and transforming the results into correlated scenario realizations. This is parallelized across MPI ranks with seed preprocessing to ensure reproducibility. **Performance is critical** in this phase — the noise generation step (sampling, correlation transforms, and method-specific operations) must be efficient at scale.

> **Opening tree lifecycle**: The noise openings used in the backward pass are generated **once before training begins** (fixed opening tree), not per-iteration. The forward pass may use the same opening tree or an alternative noise source (e.g., external scenarios). See [Scenario Generation §2.3 and §3](../architecture/scenario-generation.md).

## 1. System Representation

The top-level system object holds all entity collections needed by the solver. After loading from input files, all collections are sorted by entity ID (canonical ordering) to ensure deterministic behavior regardless of declaration order in input files. See [Design Principles §3](../overview/design-principles.md).

### 1.1 Dual-Nature Design Principle

The Cobre data model follows a **dual-nature design**: `cobre-core` defines clarity-first data models optimized for correctness and readability, while `cobre-sddp` builds performance-adapted views at initialization from `&System`.

**`cobre-core` (clarity-first)**. The `System` struct uses `Vec<Hydro>`, `Vec<Thermal>`, and other rich entity structs organized as arrays-of-structs with `HashMap`-based O(1) lookup by entity ID. This representation prioritizes data clarity: field names match domain concepts, entity relationships are explicit, and the structure is straightforward to validate, debug, and serialize. The `System` struct is the canonical representation shared across all crates that need to read the case data.

**`cobre-sddp` (performance-adapted)**. Before the training loop begins, `cobre-sddp` constructs performance-adapted views from `&System`. These views reorganize the data into struct-of-arrays layouts with contiguous LP variable indices, SIMD-friendly `f64` slices, and pre-computed lookup tables that eliminate indirection on the hot path. The performance-adapted types are consumed exclusively within the SDDP training and simulation loops. Their lifetime is bounded by the algorithm execution, and they hold no ownership over the underlying case data.

This separation means that adding a new entity type or modifying a domain struct in `cobre-core` does not require changing the performance-critical inner loops of `cobre-sddp` -- only the initialization code that builds the performance-adapted views needs to be updated. Conversely, hot-path data layout optimizations in `cobre-sddp` do not propagate to the rest of the ecosystem.

> **Cross-reference**: The performance-adapted types in `cobre-sddp` (stage templates, LP variable indexers, contiguous coefficient arrays) are defined in Epic 3 specs. See [Solver Abstraction](../architecture/solver-abstraction.md) for the stage LP template design and [Solver Workspaces](../architecture/solver-workspaces.md) for thread-local workspace layout.

### 1.2 Entity Collections

| Collection               | Source                                 | Description                                                     |
| ------------------------ | -------------------------------------- | --------------------------------------------------------------- |
| Buses                    | `system/buses.json`                    | Electrical network nodes with pre-resolved deficit segments     |
| Lines                    | `system/lines.json`                    | Transmission interconnections between buses                     |
| Hydros                   | `system/hydros.json` + extensions      | Hydro plants with generation model, reservoir, cascade topology |
| Thermals                 | `system/thermals.json`                 | Thermal plants with piecewise cost curves                       |
| Pumping stations         | `system/pumping_stations.json`         | Water transfer stations consuming power                         |
| Energy contracts         | `system/energy_contracts.json`         | Import/export agreements with external systems                  |
| Non-controllable sources | `system/non_controllable_sources.json` | Intermittent generation (wind/solar) with curtailment option    |

### 1.3 System Struct Sketch

The `System` struct is the top-level in-memory representation of a fully loaded, validated, and resolved case. It is produced by `cobre_io::load_case()` and consumed by `cobre_sddp::train()`, `cobre_sddp::simulate()`, and `cobre_stochastic` scenario generation.

```rust
/// Top-level system representation.
/// Produced by cobre-io, consumed by cobre-sddp and cobre-stochastic.
/// Immutable after construction. Shared read-only across threads.
///
/// NOTE: This is a simplified sketch showing field groupings. The actual
/// struct has 27+ fields, all private, accessed via getter methods.
/// See `cobre-io/src/system.rs` for the definitive field list.
pub struct System {
    // --- Entity collections (canonical ordering by ID) ---
    buses: Vec<Bus>,
    lines: Vec<Line>,
    hydros: Vec<Hydro>,
    thermals: Vec<Thermal>,
    pumping_stations: Vec<PumpingStation>,
    contracts: Vec<EnergyContract>,
    non_controllable_sources: Vec<NonControllableSource>,

    // --- O(1) lookup indices (entity ID -> position in collection) ---
    bus_index: HashMap<EntityId, usize>,
    line_index: HashMap<EntityId, usize>,
    hydro_index: HashMap<EntityId, usize>,
    thermal_index: HashMap<EntityId, usize>,
    pumping_station_index: HashMap<EntityId, usize>,
    contract_index: HashMap<EntityId, usize>,
    non_controllable_source_index: HashMap<EntityId, usize>,

    // --- Topology (resolved during loading) ---
    cascade: CascadeTopology,
    network: NetworkTopology,

    // --- Stages and temporal structure ---
    stages: Vec<Stage>,
    policy_graph: PolicyGraph,

    // --- Pre-resolved penalties and bounds ---
    penalties: ResolvedPenalties,
    bounds: ResolvedBounds,

    // --- Scenario pipeline parameters ---
    inflow_models: Vec<InflowModel>,     // per (hydro, stage)
    correlation: CorrelationModel,

    // --- Initial conditions ---
    initial_conditions: InitialConditions,

    // --- Generic constraints ---
    generic_constraints: Vec<GenericConstraint>,

    // --- Additional fields (see code for full list) ---
    // discount_factors, block_durations, entity_metadata,
    // evaporation_coefficients, stage_definitions, ...
}
```

**Field descriptions**:

| Field Group         | Fields                                                                                                                                | Description                                                                                                                                                                                              |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Entity collections  | `buses`, `lines`, `hydros`, `thermals`, `pumping_stations`, `contracts`, `non_controllable_sources`                                   | One `Vec` per entity type from the entity collections table (1.2). Each vector is sorted by `EntityId` in canonical order. Public for read access.                                                       |
| Lookup indices      | `bus_index`, `line_index`, `hydro_index`, `thermal_index`, `pumping_station_index`, `contract_index`, `non_controllable_source_index` | One `HashMap<EntityId, usize>` per entity type, mapping entity IDs to their position in the corresponding collection vector. Private -- accessed only through the public API methods (1.4).              |
| Cascade topology    | `cascade`                                                                                                                             | Resolved cascade topology for hydro plants. Contains downstream and upstream adjacency lists, with references to non-existing plants already redirected to the next operating downstream plant. See 1.5. |
| Network topology    | `network`                                                                                                                             | Resolved transmission network graph for buses and lines. Contains bus-line incidence relationships for power balance and flow constraints. See 1.5b.                                                     |
| Temporal structure  | `stages`, `policy_graph`                                                                                                              | Stage definitions (section 12) and policy graph transitions for finite or cyclic horizon.                                                                                                                |
| Pre-resolved data   | `penalties`, `bounds`                                                                                                                 | Fully resolved penalty and bound values per (entity, stage). See sections 10 and 11.                                                                                                                     |
| Scenario pipeline   | `par_models`, `correlation`                                                                                                           | PAR model parameters per (hydro, stage) and spectrally-decomposed correlation matrices. See section 14.                                                                                                  |
| Initial conditions  | `initial_conditions`                                                                                                                  | Initial storage volumes, filling storage, and GNL pipeline state. See section 16.                                                                                                                        |
| Generic constraints | `generic_constraints`                                                                                                                 | User-defined linear constraints with parsed expressions and resolved entity references. See section 15.                                                                                                  |

### 1.4 Public API Surface

The `System` struct exposes the following public methods. All methods are read-only -- `System` is immutable after construction.

**Entity collection accessors** -- return shared slices over the canonical-ordered collections:

```rust
impl System {
    pub fn buses(&self) -> &[Bus];
    pub fn lines(&self) -> &[Line];
    pub fn hydros(&self) -> &[Hydro];
    pub fn thermals(&self) -> &[Thermal];
    pub fn pumping_stations(&self) -> &[PumpingStation];
    pub fn contracts(&self) -> &[EnergyContract];
    pub fn non_controllable_sources(&self) -> &[NonControllableSource];
}
```

**Entity count queries** -- return the number of entities of each type (used for LP dimensioning and preallocation):

```rust
impl System {
    pub fn n_buses(&self) -> usize;
    pub fn n_lines(&self) -> usize;
    pub fn n_hydros(&self) -> usize;
    pub fn n_thermals(&self) -> usize;
    pub fn n_pumping_stations(&self) -> usize;
    pub fn n_contracts(&self) -> usize;
    pub fn n_non_controllable_sources(&self) -> usize;
}
```

**Entity lookup by ID** -- O(1) lookup via the private `HashMap` indices. Returns `None` if the ID does not exist in the system:

```rust
impl System {
    pub fn bus(&self, id: EntityId) -> Option<&Bus>;
    pub fn line(&self, id: EntityId) -> Option<&Line>;
    pub fn hydro(&self, id: EntityId) -> Option<&Hydro>;
    pub fn thermal(&self, id: EntityId) -> Option<&Thermal>;
    pub fn pumping_station(&self, id: EntityId) -> Option<&PumpingStation>;
    pub fn contract(&self, id: EntityId) -> Option<&EnergyContract>;
    pub fn non_controllable_source(&self, id: EntityId) -> Option<&NonControllableSource>;
}
```

**Topology and structure accessors**:

```rust
impl System {
    pub fn cascade(&self) -> &CascadeTopology;
    pub fn network(&self) -> &NetworkTopology;
    pub fn stages(&self) -> &[Stage];
    pub fn policy_graph(&self) -> &PolicyGraph;
}
```

All accessor methods are infallible: they operate on fully validated, resolved data. The validation guarantees are established during `cobre_io::load_case()` -- see [Input Loading Pipeline](../architecture/input-loading-pipeline.md) section 2 for the file loading sequence and cross-reference validation.

### 1.5 Cascade Topology

The `CascadeTopology` struct holds the resolved hydro cascade graph. During input loading, raw downstream references are resolved into a directed graph with non-existing plants already redirected to the next operating downstream plant (see section 2 on cascade redirection). The topology provides:

- **Downstream adjacency**: for each hydro, the ID of its immediate downstream hydro (after redirection), or `None` if it is a terminal node
- **Upstream adjacency**: for each hydro, the list of hydro IDs that flow into it (derived from downstream references)
- **Travel times**: water travel delay from upstream to downstream, in stages
- **Topological order**: a pre-computed ordering of hydros such that every upstream plant appears before its downstream plant, enabling single-pass cascade traversal

The cascade topology is immutable after construction and shared read-only. LP construction uses it to build water balance constraints without needing to walk or re-resolve the cascade graph at build time.

### 1.5b Network Topology

The `NetworkTopology` struct holds the resolved transmission network graph for buses and lines. During input loading, bus-line connectivity is resolved into a directed graph suitable for generating power balance and flow constraints. The topology provides:

- **Bus-line incidence**: for each bus, the list of lines connected to it (both incoming and outgoing), enabling single-pass power balance constraint generation
- **Line endpoints**: for each line, the from-bus and to-bus IDs (resolved to collection indices for O(1) access)
- **Bus generation map**: for each bus, the list of generating entities (hydros, thermals, non-controllable sources) connected to it, with their collection indices
- **Bus load map**: for each bus, the demand and contract entities attached to it

The network topology is immutable after construction and shared read-only. LP construction uses it to build power flow and bus balance constraints without needing to re-resolve bus-line relationships at build time. This is the transmission-network counterpart to the hydro cascade topology in 1.5.

### 1.6 Thread Safety and Lifetime

**`Send + Sync` bounds.** The `System` struct implements `Send + Sync`. This is required because:

1. **MPI broadcast**: after rank 0 constructs `System` from input files, the serialized representation is broadcast to all worker ranks. Each rank deserializes into its own `System` instance. `Send` is required for the serialization buffer to cross thread boundaries during broadcast.
2. **OpenMP-style thread sharing**: within each MPI rank, multiple threads read from the same `System` during forward and backward pass LP construction. `Sync` is required so that `&System` can be shared across threads without synchronization.

All fields of `System` are either inherently `Send + Sync` (`Vec<T>` where `T: Send + Sync`, `HashMap<K, V>` where `K, V: Send + Sync`) or must be chosen to satisfy these bounds.

**Immutability after construction.** Once constructed by `cobre_io::load_case()`, the `System` is never mutated. All access is through shared references (`&System`). This eliminates the need for interior mutability patterns (`Mutex`, `RwLock`) and guarantees data-race freedom by construction.

**Lifetime and ownership.** The `System` is owned by the caller of `load_case()` (typically the top-level orchestrator in `cobre-cli` or `cobre-python`). It is passed by shared reference to `cobre_sddp::train()` and other consumers. Because MPI broadcast creates a full copy on each rank, no cross-rank reference sharing occurs -- each rank owns its `System` instance independently. The `System` lifetime is effectively `'static` within the scope of a single training or simulation run.

**No `Arc` needed.** Since each MPI rank owns its own `System` and shares it across threads via `&System` (not `Arc<System>`), there is no reference-counting overhead. The borrow checker ensures that the `System` outlives all references to it.

### 1.7 Crate Boundary Contract

The `System` struct is the primary data type that crosses the `cobre-io` / `cobre-core` / `cobre-sddp` crate boundaries:

- **`cobre_io::load_case(path: &Path) -> Result<System, LoadError>`** -- loads, validates, and resolves all input files from the case directory into a `System`. See [Input Loading Pipeline](../architecture/input-loading-pipeline.md) SS8.1 for the full function signature, `LoadError` enum, and responsibility boundary.
- **`cobre_sddp::train(system: &System, config: &TrainingConfig, comm: &C) -> Result<TrainingResult, TrainError>`** -- consumes the `System` by shared reference to build performance-adapted views, construct stage LP templates, and execute the SDDP training loop. See [Training Loop](../architecture/training-loop.md) section 1 for the algorithm overview.
- **`cobre_sddp::simulate(system: &System, policy: &Policy, config: &SimulationConfig, comm: &C) -> Result<SimulationResult, SimError>`** -- consumes the `System` by shared reference for simulation.

The `System` type is defined in `cobre-core` so that both `cobre-io` (producer) and `cobre-sddp` (consumer) depend on `cobre-core` without creating a circular dependency. This follows the bottom-up build sequence defined in [Implementation Ordering](../overview/implementation-ordering.md).

### 1.8 EntityId Type

All entity collections use `EntityId` as the key type for identification and lookup. The input JSON schemas define entity `id` fields as `i32` (see [Input System Entities](input-system-entities.md)), so the internal type wraps `i32` directly.

```rust
/// Strongly-typed entity identifier.
///
/// Wraps the i32 identifier from JSON input files. The newtype pattern prevents
/// accidental confusion between entity IDs and collection indices (usize), which
/// is a common source of bugs in systems with both ID-based lookup and index-based
/// access. EntityId is used as the key in HashMap<EntityId, usize> lookup tables
/// and as the value in cross-reference fields (e.g., Hydro::bus_id, Line::source_bus_id).
///
/// Why i32 and not String: All JSON entity schemas use integer IDs (i32). Integer
/// keys are cheaper to hash, compare, and copy than strings — important because
/// EntityId appears in every lookup table and cross-reference field. If a future
/// input format requires string IDs, the newtype boundary isolates the change to
/// EntityId's internal representation and its From/Into impls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityId(pub i32);
```

**Design rationale**:

| Consideration    | Choice                           | Reason                                                                                                                                                                                                    |
| ---------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Inner type       | `i32`                            | Matches JSON schema `id` type across all 7 entity registries                                                                                                                                              |
| Newtype vs alias | Newtype (`struct EntityId(i32)`) | Type safety: prevents mixing entity IDs with raw indices or other integers                                                                                                                                |
| `Copy`           | Yes                              | `i32` is `Copy`; avoids unnecessary cloning in lookup paths                                                                                                                                               |
| `Hash + Eq`      | Yes                              | Required for `HashMap<EntityId, usize>` keys in System lookup indices (1.3)                                                                                                                               |
| No `Ord`         | Intentional                      | Canonical ordering uses collection position (Vec index), not ID magnitude. Sorting by ID happens once during loading; after that, position-based ordering is used. If needed, `Ord` can be derived later. |

### 1.9 Entity Struct Definitions

The following structs define the clarity-first entity representations stored in the `System` struct's entity collections (1.3). Each struct corresponds to one entity type from the entity collections table (1.2) and maps field-by-field to the JSON schema tables in [Input System Entities](input-system-entities.md).

These are `cobre-core` types following the dual-nature design (1.1): they prioritize **data clarity and correctness** over hot-path performance. The performance-adapted views in `cobre-sddp` are built from these structs at initialization time.

**What is included**: All fields that are loaded from JSON input files or resolved during input loading (default resolution, cross-reference validation). Fields are annotated as "base" (loaded directly from JSON) or "resolved" (computed during loading from defaults, cascades, or cross-references).

**What is NOT included**: LP variable indices, contiguous `f64` slices, SIMD-aligned layouts, or any other performance-adapted data. Those belong to the `cobre-sddp` performance layer (see [Solver Abstraction](../architecture/solver-abstraction.md) and [Solver Workspaces](../architecture/solver-workspaces.md)).

#### 1.9.1 Supporting Enums

Several entity fields use a fixed set of variants. These are modeled as Rust enums to prevent invalid states at the type level.

```rust
/// Hydro generation model variant.
/// Selects the production function used to convert turbined flow into generation.
/// See Input System Entities §3 and Hydro Production Models for mathematical details.
#[derive(Debug, Clone, PartialEq)]
pub enum HydroGenerationModel {
    /// Fixed productivity factor [MW/(m3/s)]. Training + simulation.
    ConstantProductivity {
        productivity_mw_per_m3s: f64,
    },
    /// Head-dependent productivity adjusted by storage level. Simulation-only.
    /// Excluded from training because the bilinear term breaks SDDP convergence.
    /// See Hydro Production Models §3.
    LinearizedHead {
        productivity_mw_per_m3s: f64,
    },
    /// Full piecewise-linear approximation via hyperplanes. Training + simulation.
    /// Hyperplanes are either computed from geometry or loaded from precomputed data.
    /// Configuration is in Input Hydro Extensions §2.
    Fpha,
}

/// Tailrace model for downstream water level computation.
/// Used by linearized_head and fpha (computed) models to determine net head.
/// See Input System Entities §3.
#[derive(Debug, Clone, PartialEq)]
pub enum TailraceModel {
    /// Tailrace level as polynomial function of outflow:
    /// h_tail(Q_out) = c0 + c1*Q + c2*Q^2 + ...
    Polynomial {
        /// Coefficients in ascending order [c0, c1, c2, ...]
        coefficients: Vec<f64>,
    },
    /// Tailrace level as piecewise-linear function of outflow.
    Piecewise {
        /// (outflow_m3s, height_m) points, sorted by outflow, monotonically increasing.
        points: Vec<TailracePoint>,
    },
}

/// A single point on the piecewise tailrace curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TailracePoint {
    pub outflow_m3s: f64,
    pub height_m: f64,
}

/// Hydraulic losses model for head loss computation.
/// See Input System Entities §3.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HydraulicLossesModel {
    /// Losses as a fraction of gross head: h_loss = value * h_gross
    Factor { value: f64 },
    /// Fixed head loss in meters, independent of operating point.
    Constant { value_m: f64 },
}

/// Turbine-generator efficiency model.
/// See Input System Entities §3.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EfficiencyModel {
    /// Single efficiency value across all operating points.
    Constant { value: f64 },
    // Future: FlowDependent { ... }
}

/// Energy contract direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractType {
    /// External power imported into the system (adds to supply).
    Import,
    /// System power exported externally (adds to demand).
    Export,
}
```

#### 1.9.2 Bus

```rust
/// Electrical network node where energy balance is maintained.
///
/// Buses represent aggregation points in the transmission network — regional
/// subsystems, substations, or any user-defined grouping. Each bus has a
/// piecewise-linear deficit cost curve that ensures LP feasibility when demand
/// cannot be met.
///
/// Source: system/buses.json. See Input System Entities §1.
pub struct Bus {
    // -- Base fields (loaded from JSON) --

    /// Unique bus identifier.
    pub id: EntityId,
    /// Human-readable bus name.
    pub name: String,

    // -- Resolved fields (defaults applied during loading) --

    /// Pre-resolved piecewise-linear deficit cost segments.
    /// If not specified in buses.json, uses the global default from penalties.json.
    /// Segments are ordered by ascending cost. The final segment has depth_mw = None
    /// (unbounded) to ensure LP feasibility.
    /// See Penalty System §2, Category 1 (recourse slack).
    pub deficit_segments: Vec<DeficitSegment>,
    /// Cost per MWh for surplus generation absorption (recourse slack).
    /// Resolved from entity-level override or global default.
    pub excess_cost: f64,
}

/// A single segment of the piecewise-linear deficit cost curve.
///
/// Segments are cumulative: the first depth_mw MW of deficit costs cost_per_mwh,
/// the next segment's depth_mw MW cost that segment's cost_per_mwh, and so on.
/// The final segment has depth_mw = None (extends to infinity).
#[derive(Debug, Clone, PartialEq)]
pub struct DeficitSegment {
    /// MW of deficit covered by this segment. None for the final unbounded segment.
    pub depth_mw: Option<f64>,
    /// Cost per MWh of deficit in this segment.
    pub cost_per_mwh: f64,
}
```

**Field descriptions**:

| Field              | Source                                                                  | Base/Resolved | Description                                                                         |
| ------------------ | ----------------------------------------------------------------------- | ------------- | ----------------------------------------------------------------------------------- |
| `id`               | `buses.json` `id`                                                       | Base          | Unique bus identifier (i32 from JSON)                                               |
| `name`             | `buses.json` `name`                                                     | Base          | Human-readable bus name                                                             |
| `deficit_segments` | `buses.json` `deficit_segments` or `penalties.json` global default      | Resolved      | Piecewise deficit cost curve with defaults applied. Final segment always unbounded. |
| `excess_cost`      | `buses.json` `penalties.excess_cost` or `penalties.json` global default | Resolved      | Surplus generation absorption cost                                                  |

#### 1.9.3 Line

```rust
/// Transmission interconnection between two buses.
///
/// Lines allow bidirectional power transfer subject to capacity limits and
/// transmission losses. Line flow is a hard constraint (no slack variables) --
/// the exchange_cost is a regularization penalty, not a violation penalty.
///
/// Source: system/lines.json. See Input System Entities §2.
pub struct Line {
    // -- Base fields (loaded from JSON) --

    /// Unique line identifier.
    pub id: EntityId,
    /// Human-readable line name.
    pub name: String,
    /// Source bus for direct flow direction.
    pub source_bus_id: EntityId,
    /// Target bus for direct flow direction.
    pub target_bus_id: EntityId,
    /// Stage when line enters service. None = always exists.
    pub entry_stage_id: Option<i32>,
    /// Stage when line is decommissioned. None = never decommissioned.
    pub exit_stage_id: Option<i32>,
    /// Maximum flow from source to target (MW). Hard bound.
    pub direct_capacity_mw: f64,
    /// Maximum flow from target to source (MW). Hard bound.
    pub reverse_capacity_mw: f64,
    /// Transmission losses as percentage (e.g., 2.5 means 2.5%).
    pub losses_percent: f64,

    // -- Resolved fields (defaults applied during loading) --

    /// Regularization cost per MWh exchanged.
    /// Resolved from entity-level override or global default from penalties.json.
    /// See Penalty System §2, Category 3.
    pub exchange_cost: f64,
}
```

**Field descriptions**:

| Field                 | Source                                                          | Base/Resolved | Description                                |
| --------------------- | --------------------------------------------------------------- | ------------- | ------------------------------------------ |
| `id`                  | `lines.json` `id`                                               | Base          | Unique line identifier                     |
| `name`                | `lines.json` `name`                                             | Base          | Human-readable line name                   |
| `source_bus_id`       | `lines.json` `source_bus_id`                                    | Base          | Source bus for direct flow                 |
| `target_bus_id`       | `lines.json` `target_bus_id`                                    | Base          | Target bus for direct flow                 |
| `entry_stage_id`      | `lines.json` `entry_stage_id`                                   | Base          | Lifecycle start (None = always)            |
| `exit_stage_id`       | `lines.json` `exit_stage_id`                                    | Base          | Lifecycle end (None = never)               |
| `direct_capacity_mw`  | `lines.json` `capacity.direct_mw`                               | Base          | Maximum direct flow (MW)                   |
| `reverse_capacity_mw` | `lines.json` `capacity.reverse_mw`                              | Base          | Maximum reverse flow (MW)                  |
| `losses_percent`      | `lines.json` `losses_percent`                                   | Base          | Transmission loss percentage (default 0.0) |
| `exchange_cost`       | `lines.json` `exchange_cost` or `penalties.json` global default | Resolved      | Regularization cost per MWh exchanged      |

#### 1.9.4 Hydro

```rust
/// Hydro plant with reservoir, generation model, and cascade topology.
///
/// Hydro plants are the most complex entity in the system. The struct captures
/// core identity, reservoir bounds, outflow limits, generation model (as a tagged
/// union), optional tailrace/losses/efficiency/evaporation data, diversion channel,
/// and filling configuration. The generation model can vary by stage or season --
/// see Input Hydro Extensions §2 for selection modes.
///
/// Source: system/hydros.json + extension files. See Input System Entities §3
/// and Input Hydro Extensions.
pub struct Hydro {
    // -- Core identity and topology (base) --

    /// Unique hydro identifier.
    pub id: EntityId,
    /// Human-readable hydro name.
    pub name: String,
    /// Bus where generation is injected.
    pub bus_id: EntityId,
    /// Physical downstream hydro in the cascade. None if terminal node.
    /// Always refers to the physical downstream -- cascade redirection for
    /// non-existing plants is handled in CascadeTopology (1.5).
    pub downstream_id: Option<EntityId>,
    /// Stage when hydro enters operation. None = always exists.
    pub entry_stage_id: Option<i32>,
    /// Stage when hydro is decommissioned. None = never.
    pub exit_stage_id: Option<i32>,

    // -- Reservoir (base) --

    /// Minimum storage (dead volume) in hm3. Soft lower bound --
    /// violation uses storage_violation_below slack.
    pub min_storage_hm3: f64,
    /// Maximum storage in hm3. Hard upper bound (emergency spillage handles excess).
    pub max_storage_hm3: f64,

    // -- Outflow bounds (base) --

    /// Minimum total outflow (environmental flow requirement) in m3/s.
    /// Soft lower bound -- violation slack available.
    pub min_outflow_m3s: f64,
    /// Maximum total outflow (flood control) in m3/s.
    /// None = no upper bound constraint. Soft upper bound when present.
    pub max_outflow_m3s: Option<f64>,

    // -- Generation model (base, tagged union) --

    /// Active generation model variant. Determines the production function
    /// used for this hydro. Can be overridden per stage or season via the
    /// production model selection mechanism in Input Hydro Extensions §2.
    pub generation_model: HydroGenerationModel,
    /// Minimum turbined flow (machine limits) in m3/s.
    /// Soft lower bound -- violation slack available.
    pub min_turbined_m3s: f64,
    /// Maximum turbined flow (machine capacity) in m3/s. Hard bound.
    pub max_turbined_m3s: f64,
    /// Minimum generation bound (user-defined) in MW.
    /// Soft lower bound -- violation slack available.
    pub min_generation_mw: f64,
    /// Maximum generation bound (user-defined) in MW. Hard bound.
    pub max_generation_mw: f64,

    // -- Optional hydro data (base, None = not provided) --

    /// Tailrace model for downstream water level computation.
    /// None = no tailrace adjustment. Only relevant for linearized_head and
    /// computed fpha models.
    pub tailrace: Option<TailraceModel>,
    /// Hydraulic losses model. None = zero losses assumed.
    pub hydraulic_losses: Option<HydraulicLossesModel>,
    /// Turbine-generator efficiency model. None = efficiency embedded in
    /// productivity_mw_per_m3s.
    pub efficiency: Option<EfficiencyModel>,
    /// Monthly evaporation coefficients (mm), January through December.
    /// None = no evaporation modeled for this hydro.
    /// Combined with surface area from hydro_geometry to compute evaporated volume.
    /// See Input Hydro Extensions §1 for geometry data.
    pub evaporation_coefficients_mm: Option<[f64; 12]>,

    // -- Diversion channel (base, optional) --

    /// Diversion channel configuration. None = no diversion.
    pub diversion: Option<DiversionChannel>,

    // -- Filling configuration (base, optional) --

    /// Dead-volume filling configuration. None = no filling period.
    /// See Input System Entities §3 (Filling Model) and Penalty System §7.
    pub filling: Option<FillingConfig>,

    // -- Resolved penalty overrides --

    /// Entity-level penalty overrides resolved from hydros.json penalties block.
    /// Stage-varying penalties are resolved separately in ResolvedPenalties (section 10).
    /// Fields here represent the entity-level base values after applying the
    /// three-tier cascade: global defaults -> entity overrides.
    pub penalties: HydroPenalties,
}

/// Diversion channel configuration for a hydro plant.
/// Diversion creates an additional flow variable bounded by [0, max_flow_m3s].
/// Diverted water is subtracted from the plant's balance and added to the
/// destination hydro's inflow. It does NOT generate power.
#[derive(Debug, Clone, PartialEq)]
pub struct DiversionChannel {
    /// Hydro plant receiving diverted water.
    pub downstream_id: EntityId,
    /// Maximum diversion flow in m3/s. Hard bound.
    pub max_flow_m3s: f64,
}

/// Dead-volume filling configuration.
/// Represents the commissioning period of a new hydro plant, during which the
/// reservoir fills from an initial level up to the dead volume (min_storage_hm3).
/// Based on CEPEL's dead-volume filling model (enchimento de volume morto).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FillingConfig {
    /// First stage of the filling period. Must be strictly less than entry_stage_id.
    pub start_stage_id: i32,
    /// Default filling inflow retained per stage (m3/s).
    /// Can be overridden per stage in hydro_bounds.
    /// Default: 0.0 (passive filling from natural inflows only).
    pub filling_inflow_m3s: f64,
}

/// Entity-level penalty overrides for a hydro plant.
/// These are the resolved base values after applying the global -> entity cascade.
/// Stage-varying overrides are stored in ResolvedPenalties (section 10).
/// All fields are resolved (never None at runtime) -- if no entity override exists,
/// the global default is used.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HydroPenalties {
    pub spillage_cost: f64,
    pub diversion_cost: f64,
    pub fpha_turbined_cost: f64,
    pub storage_violation_below_cost: f64,
    pub filling_target_violation_cost: f64,
    pub turbined_violation_below_cost: f64,
    pub outflow_violation_below_cost: f64,
    pub outflow_violation_above_cost: f64,
    pub generation_violation_below_cost: f64,
    pub evaporation_violation_cost: f64,
    pub water_withdrawal_violation_cost: f64,
}
```

**Field descriptions (Hydro)**:

| Field                         | Source                                                | Base/Resolved | Description                                                |
| ----------------------------- | ----------------------------------------------------- | ------------- | ---------------------------------------------------------- |
| `id`                          | `hydros.json` `id`                                    | Base          | Unique hydro identifier                                    |
| `name`                        | `hydros.json` `name`                                  | Base          | Human-readable hydro name                                  |
| `bus_id`                      | `hydros.json` `bus_id`                                | Base          | Bus where generation is injected                           |
| `downstream_id`               | `hydros.json` `downstream_id`                         | Base          | Physical downstream hydro (None if terminal)               |
| `entry_stage_id`              | `hydros.json` `entry_stage_id`                        | Base          | Lifecycle start (None = always)                            |
| `exit_stage_id`               | `hydros.json` `exit_stage_id`                         | Base          | Lifecycle end (None = never)                               |
| `min_storage_hm3`             | `hydros.json` `reservoir.min_storage_hm3`             | Base          | Dead volume (soft lower bound)                             |
| `max_storage_hm3`             | `hydros.json` `reservoir.max_storage_hm3`             | Base          | Full reservoir capacity (hard bound)                       |
| `min_outflow_m3s`             | `hydros.json` `outflow.min_outflow_m3s`               | Base          | Environmental flow requirement                             |
| `max_outflow_m3s`             | `hydros.json` `outflow.max_outflow_m3s`               | Base          | Flood control limit (None = unbounded)                     |
| `generation_model`            | `hydros.json` `generation.model` + variant fields     | Base          | Tagged union selecting production function                 |
| `min_turbined_m3s`            | `hydros.json` `generation.min_turbined_m3s`           | Base          | Machine minimum flow (soft bound)                          |
| `max_turbined_m3s`            | `hydros.json` `generation.max_turbined_m3s`           | Base          | Machine capacity (hard bound)                              |
| `min_generation_mw`           | `hydros.json` `generation.min_generation_mw`          | Base          | User-defined min generation (soft bound)                   |
| `max_generation_mw`           | `hydros.json` `generation.max_generation_mw`          | Base          | User-defined max generation (hard bound)                   |
| `tailrace`                    | `hydros.json` `tailrace`                              | Base          | Downstream water level model (None = no adjustment)        |
| `hydraulic_losses`            | `hydros.json` `hydraulic_losses`                      | Base          | Head losses model (None = zero losses)                     |
| `efficiency`                  | `hydros.json` `efficiency`                            | Base          | Turbine efficiency model (None = embedded in productivity) |
| `evaporation_coefficients_mm` | `hydros.json` `evaporation.coefficients_mm`           | Base          | 12 monthly evaporation rates (None = no evaporation)       |
| `diversion`                   | `hydros.json` `diversion`                             | Base          | Diversion channel config (None = no diversion)             |
| `filling`                     | `hydros.json` `filling`                               | Base          | Dead-volume filling config (None = no filling)             |
| `penalties`                   | `hydros.json` `penalties` + `penalties.json` defaults | Resolved      | Entity-level penalty values after cascade resolution       |

> **Cross-reference**: The `HydroGenerationModel` enum defines the base model loaded from `hydros.json`. Stage-varying or season-varying model selection (the `stage_ranges` and `seasonal` modes) is configured via [Input Hydro Extensions §2](input-hydro-extensions.md) and resolved into a per-(hydro, stage) lookup during loading. That resolved lookup is stored alongside the system data, not inside the `Hydro` struct itself.

#### 1.9.5 Thermal

```rust
/// Thermal plant with piecewise-linear cost curve.
///
/// Thermal dispatch is directly controllable (unlike hydro which depends on
/// exogenous inflows), so generation bounds are hard constraints without
/// slack variables.
///
/// Source: system/thermals.json. See Input System Entities §4.
pub struct Thermal {
    // -- Base fields (loaded from JSON) --

    /// Unique thermal identifier.
    pub id: EntityId,
    /// Human-readable thermal name.
    pub name: String,
    /// Bus where generation is injected.
    pub bus_id: EntityId,
    /// Stage when thermal enters service. None = always exists.
    pub entry_stage_id: Option<i32>,
    /// Stage when thermal is decommissioned. None = never.
    pub exit_stage_id: Option<i32>,
    /// Piecewise-linear cost curve, ordered by ascending cost.
    /// Each segment defines a tranche of generation capacity at a given cost.
    pub cost_segments: Vec<ThermalCostSegment>,
    /// Minimum stable generation in MW. Hard constraint (no slack).
    /// 0 if no minimum.
    pub min_generation_mw: f64,
    /// Maximum generation capacity in MW. Hard constraint (no slack).
    pub max_generation_mw: f64,
    /// GNL dispatch anticipation configuration. None = standard thermal.
    /// When present, adds state variables for committed dispatch pipeline.
    /// Currently rejected by validation -- see Input System Entities §4.
    pub gnl_config: Option<GnlConfig>,
}

/// A single segment of the thermal piecewise-linear cost curve.
///
/// Segments are ordered by ascending cost. The first segment covers the cheapest
/// tranche, and subsequent segments cover progressively more expensive tranches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalCostSegment {
    /// Generation capacity of this segment in MW.
    pub capacity_mw: f64,
    /// Marginal cost in this segment ($/MWh).
    pub cost_per_mwh: f64,
}

/// GNL (Gas Natural Liquefeito) dispatch anticipation configuration.
/// Requires dispatch decisions N stages ahead due to fuel ordering lead times.
/// Adds lag_stages state variables to the SDDP state vector per GNL thermal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GnlConfig {
    /// Number of stages of dispatch anticipation (e.g., 2 means the dispatch
    /// decision must be made 2 stages before execution).
    pub lag_stages: i32,
}
```

**Field descriptions**:

| Field               | Source                              | Base/Resolved | Description                               |
| ------------------- | ----------------------------------- | ------------- | ----------------------------------------- |
| `id`                | `thermals.json` `id`                | Base          | Unique thermal identifier                 |
| `name`              | `thermals.json` `name`              | Base          | Human-readable thermal name               |
| `bus_id`            | `thermals.json` `bus_id`            | Base          | Bus where generation is injected          |
| `entry_stage_id`    | `thermals.json` `entry_stage_id`    | Base          | Lifecycle start (None = always)           |
| `exit_stage_id`     | `thermals.json` `exit_stage_id`     | Base          | Lifecycle end (None = never)              |
| `cost_segments`     | `thermals.json` `cost_segments`     | Base          | Piecewise cost curve segments             |
| `min_generation_mw` | `thermals.json` `generation.min_mw` | Base          | Minimum stable generation (hard bound)    |
| `max_generation_mw` | `thermals.json` `generation.max_mw` | Base          | Maximum capacity (hard bound)             |
| `gnl_config`        | `thermals.json` `gnl_config`        | Base          | GNL anticipation config (None = standard) |

#### 1.9.6 PumpingStation

```rust
/// Water transfer station consuming electrical power.
///
/// Pumping stations are independent system elements -- not a property of any
/// specific hydro plant. They transfer water from a source reservoir to a
/// destination reservoir, with power consumption at the connected bus. The cost
/// of pumping is implicit via energy consumption in the bus load balance.
///
/// Source: system/pumping_stations.json. See Input System Entities §5.
pub struct PumpingStation {
    // -- Base fields (loaded from JSON) --

    /// Unique station identifier.
    pub id: EntityId,
    /// Station name.
    pub name: String,
    /// Bus where power is consumed.
    pub bus_id: EntityId,
    /// Hydro from which water is withdrawn.
    pub source_hydro_id: EntityId,
    /// Hydro to which water is delivered.
    pub destination_hydro_id: EntityId,
    /// Stage when station enters service. None = always exists.
    pub entry_stage_id: Option<i32>,
    /// Stage when station is decommissioned. None = never.
    pub exit_stage_id: Option<i32>,
    /// Power consumption rate [MW/(m3/s)].
    pub consumption_mw_per_m3s: f64,
    /// Minimum pumped flow in m3/s.
    pub min_flow_m3s: f64,
    /// Maximum pumped flow in m3/s.
    pub max_flow_m3s: f64,
}
```

**Field descriptions**:

| Field                    | Source                                           | Base/Resolved | Description                     |
| ------------------------ | ------------------------------------------------ | ------------- | ------------------------------- |
| `id`                     | `pumping_stations.json` `id`                     | Base          | Unique station identifier       |
| `name`                   | `pumping_stations.json` `name`                   | Base          | Station name                    |
| `bus_id`                 | `pumping_stations.json` `bus_id`                 | Base          | Bus where power is consumed     |
| `source_hydro_id`        | `pumping_stations.json` `source_hydro_id`        | Base          | Water source hydro              |
| `destination_hydro_id`   | `pumping_stations.json` `destination_hydro_id`   | Base          | Water destination hydro         |
| `entry_stage_id`         | `pumping_stations.json` `entry_stage_id`         | Base          | Lifecycle start (None = always) |
| `exit_stage_id`          | `pumping_stations.json` `exit_stage_id`          | Base          | Lifecycle end (None = never)    |
| `consumption_mw_per_m3s` | `pumping_stations.json` `consumption_mw_per_m3s` | Base          | Power consumption rate          |
| `min_flow_m3s`           | `pumping_stations.json` `flow.min_m3s`           | Base          | Minimum pumped flow             |
| `max_flow_m3s`           | `pumping_stations.json` `flow.max_m3s`           | Base          | Maximum pumped flow             |

#### 1.9.7 EnergyContract

```rust
/// Import/export agreement with an external system.
///
/// Import contracts add to supply at the connected bus; export contracts add
/// to demand. Contract limits are hard constraints (no slack variables).
/// The price represents cost (import, positive) or revenue (export, typically negative).
///
/// Source: system/energy_contracts.json. See Input System Entities §6.
pub struct EnergyContract {
    // -- Base fields (loaded from JSON) --

    /// Unique contract identifier.
    pub id: EntityId,
    /// Contract name.
    pub name: String,
    /// Bus connected to this contract.
    pub bus_id: EntityId,
    /// Contract direction (import or export).
    pub contract_type: ContractType,
    /// Stage when contract becomes active. None = always exists.
    pub entry_stage_id: Option<i32>,
    /// Stage when contract expires. None = never.
    pub exit_stage_id: Option<i32>,
    /// Cost per MWh (import, positive) or revenue per MWh (export, typically negative).
    pub price_per_mwh: f64,
    /// Minimum contract usage in MW. Hard bound.
    pub min_mw: f64,
    /// Maximum contract usage in MW. Hard bound.
    pub max_mw: f64,
}
```

**Field descriptions**:

| Field            | Source                                   | Base/Resolved | Description                       |
| ---------------- | ---------------------------------------- | ------------- | --------------------------------- |
| `id`             | `energy_contracts.json` `id`             | Base          | Unique contract identifier        |
| `name`           | `energy_contracts.json` `name`           | Base          | Contract name                     |
| `bus_id`         | `energy_contracts.json` `bus_id`         | Base          | Bus connected to contract         |
| `contract_type`  | `energy_contracts.json` `type`           | Base          | Import or export direction        |
| `entry_stage_id` | `energy_contracts.json` `entry_stage_id` | Base          | Lifecycle start (None = always)   |
| `exit_stage_id`  | `energy_contracts.json` `exit_stage_id`  | Base          | Lifecycle end (None = never)      |
| `price_per_mwh`  | `energy_contracts.json` `price_per_mwh`  | Base          | Cost (import) or revenue (export) |
| `min_mw`         | `energy_contracts.json` `limits.min_mw`  | Base          | Minimum usage (hard bound)        |
| `max_mw`         | `energy_contracts.json` `limits.max_mw`  | Base          | Maximum usage (hard bound)        |

#### 1.9.8 NonControllableSource

```rust
/// Intermittent generation source (wind, solar, small run-of-river hydro).
///
/// Non-controllable sources have stochastic availability per (stage, scenario)
/// from the scenario pipeline. The solver can only curtail generation below the
/// available amount -- it cannot dispatch upward. Curtailment is penalized with
/// curtailment_cost (regularization) to prioritize using available generation.
///
/// Source: system/non_controllable_sources.json. See Input System Entities §7.
pub struct NonControllableSource {
    // -- Base fields (loaded from JSON) --

    /// Unique source identifier.
    pub id: EntityId,
    /// Human-readable source name.
    pub name: String,
    /// Bus where generation is injected.
    pub bus_id: EntityId,
    /// Stage when source enters service. None = always exists.
    pub entry_stage_id: Option<i32>,
    /// Stage when source is decommissioned. None = never.
    pub exit_stage_id: Option<i32>,
    /// Installed capacity (hard upper bound, physical limit) in MW.
    pub max_generation_mw: f64,

    // -- Resolved fields (defaults applied during loading) --

    /// Regularization cost per MWh curtailed.
    /// Resolved from entity-level override or global default from penalties.json.
    /// See Penalty System §2, Category 3.
    pub curtailment_cost: f64,
}
```

**Field descriptions**:

| Field               | Source                                                                                | Base/Resolved | Description                           |
| ------------------- | ------------------------------------------------------------------------------------- | ------------- | ------------------------------------- |
| `id`                | `non_controllable_sources.json` `id`                                                  | Base          | Unique source identifier              |
| `name`              | `non_controllable_sources.json` `name`                                                | Base          | Human-readable source name            |
| `bus_id`            | `non_controllable_sources.json` `bus_id`                                              | Base          | Bus where generation is injected      |
| `entry_stage_id`    | `non_controllable_sources.json` `entry_stage_id`                                      | Base          | Lifecycle start (None = always)       |
| `exit_stage_id`     | `non_controllable_sources.json` `exit_stage_id`                                       | Base          | Lifecycle end (None = never)          |
| `max_generation_mw` | `non_controllable_sources.json` `max_generation_mw`                                   | Base          | Installed capacity (hard bound)       |
| `curtailment_cost`  | `non_controllable_sources.json` `curtailment_cost` or `penalties.json` global default | Resolved      | Regularization cost per MWh curtailed |

## 2. Operative State

Every entity with lifecycle attributes (`entry_stage_id`, `exit_stage_id`) has a stage-dependent operative state that determines which LP variables and constraints are active. The operative state is computed per entity per stage.

| State          | Applies To       | LP Treatment                                                                                                                |
| -------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Non-existing   | All entity types | No LP variables or constraints                                                                                              |
| Filling        | Hydros only      | Storage, spillage, evaporation, violation slacks. No generation (turbined = 0). See [Penalty System §7](penalty-system.md). |
| Operating      | All entity types | Full LP variables and constraints per entity type                                                                           |
| Decommissioned | All entity types | No LP variables or constraints (identical to Non-existing)                                                                  |

**Stakeholder decision**: Decommissioned entities are treated identically to Non-existing for all entity types, including hydros. Reservoir drainage after decommissioning is not modeled. This simplifies the lifecycle to three meaningful LP states: Non-existing/Decommissioned (no LP presence), Filling (hydros only), and Operating (full LP formulation).

**Hydro filling timeline**: `[start_stage_id, entry_stage_id)` = filling period. At `entry_stage_id`, transitions to operating. Based on CEPEL's dead-volume filling model (`enchimento de volume morto`).

**Cascade redirection**: When a downstream hydro is non-existing, outflows are automatically redirected to the next operating downstream in the cascade. The internal representation should resolve this redirection so the LP builder does not need to walk the cascade at build time.

## 3. Hydro Plant

Hydro plants are the most complex entity in the system. The in-memory hydro representation must include:

### Core Identity and Topology

- Entity ID, name, and bus assignment
- Downstream hydro ID (physical cascade), upstream hydro IDs (derived from downstream references)
- Lifecycle: `entry_stage_id`, `exit_stage_id`

### Reservoir

- `min_storage_hm3` (dead volume) — soft lower bound, violation uses `storage_violation_below` slack
- `max_storage_hm3` — hard upper bound (emergency spillage handles excess)

### Outflow Bounds

- `min_outflow_m3s` — soft lower bound (environmental flow), violation slack available
- `max_outflow_m3s` — soft upper bound (flood control), violation slack available. Null = no upper bound.

### Generation Model (Tagged Union)

The generation model is a tagged union selected by a `model` field. Only fields relevant to the selected variant are meaningful. See [Input System Entities §3](input-system-entities.md) for input schema and [Hydro Production Functions](../math/hydro-production-models.md) for math.

| Variant                 | Key Data                                                            | Requires Geometry              | Phase                 |
| ----------------------- | ------------------------------------------------------------------- | ------------------------------ | --------------------- |
| `constant_productivity` | Productivity factor [MW/(m3/s)]                                     | No                             | Training + Simulation |
| `linearized_head`       | Base productivity, adjusted by storage-dependent head               | Yes                            | **Simulation-only**   |
| `fpha`                  | Hyperplane coefficients (gamma_0, gamma_v, gamma_q, gamma_s, kappa) | Computed: yes; Precomputed: no | Training + Simulation |

> **Simulation-only restriction**: The `linearized_head` model is excluded from training because the bilinear term ($q \times v^{avg}$) requires re-fixing $v^{avg}$ between iterations, changing the LP structure and breaking SDDP convergence guarantees. See [Hydro Production Models §3](../math/hydro-production-models.md).

All variants carry:

- `min_turbined_m3s`, `max_turbined_m3s` — machine flow limits (lower bound has violation slack)
- `min_generation_mw`, `max_generation_mw` — user-defined generation bounds (lower bound has violation slack, upper is hard)

**FPHA turbined flow penalty**: When using the `fpha` production model, the concave geometry of the hyperplane approximation means that every point inside the region bounded by the FPHA hyperplanes is a feasible LP solution — including solutions where the generation is less than what the physical production function would yield for the given turbined flow. Without correction, the solver can effectively "spill through the turbines" by choosing these interior points. To prevent this, a `fpha_turbined_cost` regularization penalty is applied to the turbined flow variable, and it must be strictly greater than the `spillage_cost` for the same plant. This ensures the solver prefers to turbine (and generate) rather than spill, and when it does turbine, the FPHA constraints push generation toward the physical envelope. This penalty is only active for hydros using the `fpha` production model.

### Production Model Selection

The active generation model can vary by stage or season. See [Input Hydro Extensions §2](input-hydro-extensions.md) for the selection modes (`stage_ranges`, `seasonal` with `default_model` fallback).

### Optional Hydro Data

| Data             | Purpose                                                            | Source             |
| ---------------- | ------------------------------------------------------------------ | ------------------ |
| Tailrace model   | Downstream water level vs. outflow (polynomial or piecewise)       | `hydros.json`      |
| Hydraulic losses | Head losses (factor or constant)                                   | `hydros.json`      |
| Efficiency       | Turbine-generator efficiency (constant; future: flow-dependent)    | `hydros.json`      |
| Evaporation      | 12 monthly coefficients (mm)                                       | `hydros.json`      |
| Geometry         | Volume-Height-Area table for forebay/area computation              | `hydro_geometry`   |
| FPHA hyperplanes | Pre-computed production function planes (optionally stage-varying) | `fpha_hyperplanes` |

### Diversion Channel (Optional)

- `downstream_id` — hydro receiving diverted water
- `max_flow_m3s` — maximum diversion flow
- Diversion variable bounded by `[0, max_flow_m3s]`; only regularization cost (no violation slacks)

### Filling Configuration (Optional)

Based on CEPEL's dead-volume filling model. See [Input System Entities §3](input-system-entities.md) and [Penalty System §7](penalty-system.md).

- `start_stage_id` — first filling stage
- `filling_inflow_m3s` — entity-level default for filling inflow (overridable per stage in `hydro_bounds`)
- Filling target is always `min_storage_hm3`
- Terminal constraint at `entry_stage_id - 1` with `filling_target_violation` slack (highest penalty in system)

## 4. Thermal Plant

In-memory thermal representation:

- Entity ID, name, bus assignment
- Lifecycle: `entry_stage_id`, `exit_stage_id`
- **Cost segments**: Piecewise-linear cost curve — ordered list of (capacity_mw, cost_per_mwh) segments
- **Generation bounds**: `min_mw`, `max_mw` — hard constraints (no slack variables; thermal dispatch is directly controllable)

### GNL Dispatch Anticipation

> **Implementation Status**: Data model defined. Validation currently rejects GNL thermal plants (thermals with `gnl_config` present). This restriction will be removed when the GNL algorithm is implemented.

GNL (Gas Natural Liquefeito) thermals require dispatch decisions N stages ahead due to fuel ordering lead times. This fundamentally changes the LP structure for these plants:

- **`lag_stages`** — number of stages of dispatch anticipation (e.g., 2 means the dispatch must be decided 2 stages before execution)
- **GNL state variables**: For each GNL thermal, the algorithm adds `lag_stages` state variables to the SDDP state vector: `gnl_committed[thermal_id, t+1]`, `gnl_committed[thermal_id, t+2]`, ..., up to `lag_stages` ahead. These represent the MW dispatch already committed for future stages.
- **LP coupling**: At stage `t`, the GNL thermal's generation is constrained by the commitment made at stage `t - lag_stages`. The LP at stage `t` also includes decision variables for committing dispatch at stage `t + lag_stages`.
- **State dimension impact**: Each GNL thermal with `lag_stages = L` adds `L` state variables to the system, increasing the cut coefficient count and the FCF dimensionality.

See [Input System Entities §4](input-system-entities.md) for the input schema.

## 5. Transmission Line

In-memory line representation:

- Entity ID, name, source/target bus IDs
- Lifecycle: `entry_stage_id`, `exit_stage_id`
- **Capacity**: `direct_mw`, `reverse_mw` — hard bounds (no slack variables)
- **Exchange cost**: Regularization cost per MWh (not a violation penalty)
- **Losses**: Transmission losses as percentage

## 6. Bus

In-memory bus representation:

- Entity ID, name
- **Deficit segments**: Pre-resolved piecewise-linear deficit cost curve. Each segment has `depth_mw` and `cost`. The final segment is unbounded (ensures LP feasibility). Deficit is a recourse slack — see [Penalty System §2](penalty-system.md).
- **Excess cost**: Cost for surplus generation absorption (recourse slack)

## 7. Pumping Station

In-memory pumping station representation. Independent system element — not a property of any hydro. See [Input System Entities §5](input-system-entities.md).

- Entity ID, name, bus assignment
- `source_hydro_id`, `destination_hydro_id` — water transfer endpoints
- Lifecycle: `entry_stage_id`, `exit_stage_id`
- `consumption_mw_per_m3s` — power consumption rate
- Flow bounds: `min_m3s`, `max_m3s`
- No explicit cost — cost is implicit via energy consumption at the connected bus

## 8. Energy Contract

In-memory contract representation. See [Input System Entities §6](input-system-entities.md).

- Entity ID, name, bus assignment
- Type: `import` or `export`
- Lifecycle: `entry_stage_id`, `exit_stage_id`
- `price_per_mwh` — cost (import) or revenue (export)
- Limits: `min_mw`, `max_mw`

## 9. Non-Controllable Source

In-memory non-controllable source representation. See [Input System Entities §7](input-system-entities.md).

Non-controllable sources represent intermittent generation (wind farms, solar plants, small hydros run-of-river, etc.) whose available output depends on external conditions rather than dispatch decisions. The solver receives a stochastic availability value per scenario and can only curtail generation below that availability — it cannot dispatch upward.

- Entity ID, name, bus assignment
- Lifecycle: `entry_stage_id`, `exit_stage_id`
- `max_generation_mw` — installed capacity (hard upper bound, physical limit)
- **Availability**: stochastic generation available per (stage, scenario), provided by the scenario pipeline. Bounded by `[0, max_generation_mw]`.
- **Generation variable**: bounded by `[0, available_generation]` — the solver dispatches between 0 and whatever is available
- **Curtailment**: the difference between available generation and dispatched generation. Penalized with `curtailment_cost` (regularization, Category 3) to prioritize using non-controllable generation over curtailing it. Analogous to spillage cost for hydros — curtailment discards available "free" energy.

## 10. Penalty Tables (Pre-Resolved)

At runtime, the solver needs a single resolved penalty value for each (entity, stage, penalty_type) tuple. The three-tier cascade (global defaults → entity overrides → stage overrides) is resolved during input loading. See [Penalty System](penalty-system.md) for the cascade semantics and penalty categories.

### Pre-Resolution

During input loading, the implementation should build a resolved penalty lookup that can answer: "what is the effective penalty for entity X, stage T, penalty type P?" in constant time. The resolution walks:

1. Start with global defaults from `penalties.json`
2. Apply entity-level overrides from entity registry files (`hydros.json`, `buses.json`, etc.)
3. Apply stage-level overrides from stage override files

### Penalty Types Per Entity

**Bus penalties:**

| Penalty            | Category | Description                                |
| ------------------ | -------- | ------------------------------------------ |
| `deficit_segments` | Recourse | Piecewise deficit cost (not stage-varying) |
| `excess_cost`      | Recourse | Surplus generation absorption              |

**Line penalties:**

| Penalty         | Category       | Description         |
| --------------- | -------------- | ------------------- |
| `exchange_cost` | Regularization | Flow discouragement |

**Hydro penalties:**

| Penalty                           | Category             | Description                                                                  |
| --------------------------------- | -------------------- | ---------------------------------------------------------------------------- |
| `spillage_cost`                   | Regularization       | Prefer turbining over spilling                                               |
| `diversion_cost`                  | Regularization       | Prefer main channel flow                                                     |
| `fpha_turbined_cost`              | Regularization       | Prevent interior FPHA solutions (FPHA model only, must be > `spillage_cost`) |
| `storage_violation_below_cost`    | Constraint violation | Storage below dead volume                                                    |
| `filling_target_violation_cost`   | Constraint violation | Filling target not reached                                                   |
| `turbined_violation_below_cost`   | Constraint violation | Turbined flow below minimum                                                  |
| `outflow_violation_below_cost`    | Constraint violation | Outflow below environmental minimum                                          |
| `outflow_violation_above_cost`    | Constraint violation | Outflow above flood control limit                                            |
| `generation_violation_below_cost` | Constraint violation | Generation below contractual minimum                                         |
| `evaporation_violation_cost`      | Constraint violation | Evaporation constraint (bidirectional)                                       |
| `water_withdrawal_violation_cost` | Constraint violation | Unmet water withdrawal                                                       |

**Non-controllable source penalties:**

| Penalty            | Category       | Description                                                                          |
| ------------------ | -------------- | ------------------------------------------------------------------------------------ |
| `curtailment_cost` | Regularization | Penalizes curtailing available generation; prioritizes using non-controllable energy |

### Penalty Priority Ordering

$$\text{Filling target} > \text{Storage violation} > \text{Deficit} > \text{Constraint violations} > \text{Resource costs} > \text{Regularization}$$

Regularization costs (spillage, diversion, FPHA turbined, curtailment, exchange) are the lowest priority — they guide solution quality when the LP has degrees of freedom, but never override economic or feasibility objectives. Within FPHA hydros, the constraint `fpha_turbined_cost > spillage_cost` must hold per plant.

## 11. Pre-Resolved Entity Bounds

Similar to penalties, entity bounds are resolved during loading from the base values in entity registries with sparse stage overrides applied on top. The internal representation should provide: "what are the effective bounds for entity X, stage T?" in constant time.

### Bound Resolution

For each entity type, the base bounds come from the entity registry file (e.g., `hydros.json`). Stage-specific overrides from the constraints directory (e.g., `hydro_bounds`) replace individual fields where provided (null = keep base). See [Input Constraints §2](input-constraints.md).

### Hydro Bounds Per Stage

The resolved set of hydro bounds for a given (hydro, stage):

| Bound                  | Source                          | Slack                        |
| ---------------------- | ------------------------------- | ---------------------------- |
| `min_storage_hm3`      | Reservoir config                | `storage_violation_below`    |
| `max_storage_hm3`      | Reservoir config                | Hard (emergency spill)       |
| `min_turbined_m3s`     | Generation model                | `turbined_violation_below`   |
| `max_turbined_m3s`     | Generation model                | Hard                         |
| `min_outflow_m3s`      | Outflow config                  | `outflow_violation_below`    |
| `max_outflow_m3s`      | Outflow config                  | `outflow_violation_above`    |
| `min_generation_mw`    | Generation model                | `generation_violation_below` |
| `max_generation_mw`    | Generation model                | Hard                         |
| `max_diversion_m3s`    | Diversion config                | Hard                         |
| `filling_inflow_m3s`   | Filling config → stage override | —                            |
| `water_withdrawal_m3s` | Stage override                  | `water_withdrawal_violation` |

## 12. Stage and Block Definitions

The temporal structure must be fully loaded in memory for the solver to dimension LPs and iterate. These are `cobre-core` clarity-first types following the dual-nature design (1.1): they use `Vec<T>`, `String`, and `HashMap` for readability and correctness. LP-related fields (variable indices, constraint counts, coefficient arrays) belong to the `cobre-sddp` performance layer -- see [Solver Abstraction](../architecture/solver-abstraction.md) for the stage LP template design.

### 12.1 Supporting Enums

```rust
/// Block formulation mode controlling how blocks within a stage relate
/// to each other in the LP.
///
/// See [Block Formulations](../math/block-formulations.md) for the
/// mathematical treatment of each mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockMode {
    /// Blocks are independent sub-periods solved simultaneously.
    /// Water balance is aggregated across all blocks in the stage.
    /// This is the default and most common mode.
    Parallel,

    /// Blocks are sequential within the stage, with inter-block
    /// state transitions (intra-stage storage dynamics).
    /// Enables modeling of daily cycling patterns within monthly stages.
    Chronological,
}

/// Season cycle type controlling how season IDs map to calendar periods.
///
/// See [Input Scenarios §1.1](input-scenarios.md) for the JSON schema
/// and calendar mapping rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeasonCycleType {
    /// Each season corresponds to one calendar month (12 seasons).
    Monthly,
    /// Each season corresponds to one ISO calendar week (52 seasons).
    Weekly,
    /// User-defined date ranges with explicit boundaries per season.
    Custom,
}

/// Opening tree noise generation algorithm for a stage.
///
/// Controls which algorithm is used to generate noise vectors for
/// the opening tree at this stage. This is orthogonal to
/// `SamplingScheme`, which selects the forward-pass noise *source*
/// (in_sample, external, historical). `NoiseMethod` governs *how*
/// the noise vectors are produced (SAA, LHS, QMC-Sobol, QMC-Halton,
/// Selective).
///
/// See [Input Scenarios §1.8](input-scenarios.md) for the
/// full method catalog and use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseMethod {
    /// Sample Average Approximation. Pure Monte Carlo random sampling.
    Saa,
    /// Latin Hypercube Sampling. Stratified sampling ensuring uniform coverage.
    Lhs,
    /// Quasi-Monte Carlo with Sobol sequences. Low-discrepancy.
    QmcSobol,
    /// Quasi-Monte Carlo with Halton sequences. Low-discrepancy.
    QmcHalton,
    /// Selective/Representative Sampling. Clustering on historical data.
    Selective,
}
```

### 12.2 Block

```rust
/// A load block within a stage, representing a sub-period with uniform
/// demand and generation characteristics.
///
/// Blocks partition the stage duration into sub-periods (e.g., peak,
/// off-peak, shoulder). Block IDs are contiguous within each stage,
/// starting at 0. The block weight (fraction of stage duration) is
/// derived from `duration_hours` and is not stored -- it is computed
/// on demand as `duration_hours / sum(all block hours in stage)`.
///
/// Source: `stages.json` `stages[].blocks[]`.
/// See [Input Scenarios §1.5](input-scenarios.md).
pub struct Block {
    /// 0-based index within the parent stage.
    /// Matches the `id` field from `stages.json`, validated to be
    /// contiguous (0, 1, 2, ..., n-1) during loading.
    pub index: usize,

    /// Human-readable block label (e.g., "LEVE", "MEDIA", "PESADA").
    pub name: String,

    /// Duration of this block in hours. Must be positive.
    /// Validation: the sum of all block hours within a stage must
    /// equal the total stage duration in hours.
    /// See [Input Scenarios §1.10](input-scenarios.md), rule 3.
    pub duration_hours: f64,
}
```

**Field descriptions**:

| Field            | Source                         | Description                                                                                                                                                                                            |
| ---------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `index`          | `stages.json` `blocks[].id`    | 0-based block index within the stage. Contiguous by validation rule 2 ([Input Scenarios §1.10](input-scenarios.md)).                                                                                   |
| `name`           | `stages.json` `blocks[].name`  | Human-readable label for reporting and identification.                                                                                                                                                 |
| `duration_hours` | `stages.json` `blocks[].hours` | Duration in hours. The block **weight** is derived as `duration_hours / total_stage_hours` and is not stored as a field. Computed on demand because it depends on the sibling blocks within the stage. |

### 12.3 StageStateConfig

```rust
/// State variable flags controlling which variables carry state
/// between stages for a given stage.
///
/// Source: `stages.json` `stages[].state_variables`.
/// See [Input Scenarios §1.6](input-scenarios.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageStateConfig {
    /// Whether reservoir storage volumes are state variables.
    /// Default: true. Mandatory in most applications but kept as an
    /// explicit flag for transparency.
    pub storage: bool,

    /// Whether past inflow realizations (AR model lags) are state
    /// variables. Default: false. Required when PAR model order p > 0
    /// and inflow lag cuts are enabled.
    pub inflow_lags: bool,
}
```

### 12.4 StageRiskConfig

```rust
/// Per-stage risk measure configuration, representing the parsed and
/// validated risk parameters for a single stage.
///
/// This is the clarity-first representation stored in the Stage struct.
/// The solver-level `RiskMeasure` enum in
/// [Risk Measure Trait](../architecture/risk-measure-trait.md) is the
/// dispatch type built FROM this configuration during the variant
/// selection pipeline.
///
/// Source: `stages.json` `stages[].risk_measure`.
/// See [Input Scenarios §1.7](input-scenarios.md).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StageRiskConfig {
    /// Risk-neutral expected value. No additional parameters.
    Expectation,

    /// Convex combination of expectation and CVaR.
    /// See [Risk Measures](../math/risk-measures.md) for the
    /// mathematical formulation.
    CVaR {
        /// Confidence level alpha in (0, 1].
        /// alpha = 0.95 means 5% worst-case scenarios.
        alpha: f64,

        /// Risk aversion weight lambda in [0, 1].
        /// lambda = 0 reduces to Expectation.
        /// lambda = 1 is pure CVaR.
        lambda: f64,
    },
}
```

### 12.5 ScenarioSourceConfig

```rust
/// Scenario source configuration for one stage.
///
/// Groups the scenario-related settings that were formerly separate
/// `num_scenarios` and `sampling_method` fields. Sourced from
/// `stages.json scenario_source` and per-stage overrides.
///
/// See [Input Scenarios §1.4, §1.8](input-scenarios.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScenarioSourceConfig {
    /// Number of noise realizations per stage for both the opening
    /// tree and forward pass. Formerly `num_scenarios`.
    /// Must be positive. Controls the per-stage branching factor.
    pub branching_factor: usize,

    /// Algorithm for generating noise vectors in the opening tree.
    /// Orthogonal to `SamplingScheme`, which selects the noise
    /// source (in_sample, external, historical).
    /// Can vary per stage, allowing adaptive strategies (e.g., LHS
    /// for near-term, SAA for distant stages).
    pub noise_method: NoiseMethod,
}
```

### 12.6 Stage

```rust
/// A single stage in the multi-stage stochastic optimization problem.
///
/// Stages partition the study horizon into decision periods. Each stage
/// has a temporal extent, block structure, scenario configuration, risk
/// parameters, and state variable flags. Stages are sorted by `id` in
/// canonical order after loading (see Design Principles §3).
///
/// Study stages have non-negative IDs; pre-study stages (used only for
/// PAR model lag initialization) have negative IDs. Pre-study stages
/// carry only `id`, `start_date`, `end_date`, and `season_id` -- their
/// blocks, risk, and sampling fields are unused.
///
/// This struct does NOT contain LP-related fields (variable indices,
/// constraint counts, coefficient arrays). Those belong to the
/// cobre-sddp performance layer -- see Solver Abstraction SS11.
///
/// Source: `stages.json` `stages[]` and `pre_study_stages[]`.
/// See [Input Scenarios §1.4](input-scenarios.md).
pub struct Stage {
    // -- Identity and temporal extent --

    /// 0-based index of this stage in the canonical-ordered stages
    /// vector. Used for array indexing into per-stage data structures
    /// (cuts, results, penalty arrays). Assigned during loading after
    /// sorting by `id`.
    pub index: usize,

    /// Unique stage identifier from `stages.json`.
    /// Non-negative for study stages, negative for pre-study stages.
    /// The `id` is the domain-level identifier; `index` is the
    /// internal array position.
    pub id: i32,

    /// Stage start date (inclusive). Parsed from ISO 8601 string.
    /// Uses `chrono::NaiveDate` -- timezone-free calendar date, which
    /// is appropriate because stage boundaries are calendar concepts,
    /// not instants in time.
    pub start_date: NaiveDate,

    /// Stage end date (exclusive). Parsed from ISO 8601 string.
    /// The stage duration is `end_date - start_date`.
    pub end_date: NaiveDate,

    /// Season index linking to `SeasonDefinition`. Maps this stage to
    /// a position in the seasonal cycle (e.g., month 0-11 for monthly).
    /// Required for PAR model coefficient lookup and inflow history
    /// aggregation. `None` for stages without seasonal structure.
    pub season_id: Option<usize>,

    // -- Block structure --

    /// Ordered list of load blocks within this stage.
    /// Sorted by block index (0, 1, ..., n-1). The sum of all block
    /// `duration_hours` must equal the total stage duration in hours.
    pub blocks: Vec<Block>,

    /// Block formulation mode for this stage.
    /// Can vary per stage (e.g., chronological for near-term,
    /// parallel for distant stages).
    /// See [Block Formulations](../math/block-formulations.md).
    pub block_mode: BlockMode,

    // -- State, risk, and sampling --

    /// State variable flags controlling which variables carry state
    /// from this stage to the next.
    pub state_config: StageStateConfig,

    /// Risk measure configuration for this stage.
    /// Can vary per stage (e.g., CVaR for near-term, Expectation
    /// for distant stages).
    pub risk_config: StageRiskConfig,

    /// Scenario source configuration (branching factor and noise method).
    pub scenario_config: ScenarioSourceConfig,
}
```

**Field descriptions**:

| Field             | Source                                                              | Description                                                                                                                                               |
| ----------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `index`           | Assigned during loading                                             | 0-based canonical index in the `System.stages` vector. Assigned after sorting by `id`.                                                                    |
| `id`              | `stages.json` `stages[].id`                                         | Domain-level identifier. Non-negative for study, negative for pre-study.                                                                                  |
| `start_date`      | `stages.json` `stages[].start_date`                                 | Stage start date (inclusive). Parsed from ISO 8601 string.                                                                                                |
| `end_date`        | `stages.json` `stages[].end_date`                                   | Stage end date (exclusive). Duration = `end_date - start_date`.                                                                                           |
| `season_id`       | `stages.json` `stages[].season_id`                                  | Season index for PAR model cycling. `None` when null in JSON. Validated against `season_definitions`.                                                     |
| `blocks`          | `stages.json` `stages[].blocks`                                     | Ordered block list. Resolves the `System` field `pub stages: Vec<Stage>` block structure.                                                                 |
| `block_mode`      | `stages.json` `stages[].block_mode`                                 | Parallel or Chronological. Default: Parallel. See [Block Formulations](../math/block-formulations.md).                                                    |
| `state_config`    | `stages.json` `stages[].state_variables`                            | State variable flags (storage, inflow_lags). Defaults resolved during loading.                                                                            |
| `risk_config`     | `stages.json` `stages[].risk_measure`                               | Risk measure parameters. Expectation or CVaR(alpha, lambda). Validated by rules R1-R4 in [Risk Measure Trait SS5](../architecture/risk-measure-trait.md). |
| `scenario_config` | `stages.json` `scenario_source` (top-level and per-stage overrides) | Branching factor and noise method. Both can vary per stage.                                                                                               |

### 12.7 SeasonDefinition

```rust
/// A single season entry mapping a season ID to a calendar period.
///
/// Season definitions are required when deriving AR parameters from
/// inflow history — the season determines how history values are
/// aggregated into seasonal means and standard deviations.
///
/// Source: `stages.json` `season_definitions.seasons[]`.
/// See [Input Scenarios §1.1](input-scenarios.md).
pub struct SeasonDefinition {
    /// Season index (0-based). For monthly cycles: 0 = January, ...,
    /// 11 = December. For weekly cycles: 0-51 (ISO week numbers).
    pub id: usize,

    /// Human-readable label (e.g., "January", "Q1", "Wet Season").
    pub label: String,

    /// Calendar month where this season starts (1-12).
    /// For monthly cycle_type, this uniquely identifies the month.
    pub month_start: u32,

    /// Calendar day where this season starts (1-31).
    /// Only used when cycle_type is Custom. Default: 1.
    pub day_start: Option<u32>,

    /// Calendar month where this season ends (1-12).
    /// Only used when cycle_type is Custom.
    pub month_end: Option<u32>,

    /// Calendar day where this season ends (1-31).
    /// Only used when cycle_type is Custom.
    pub day_end: Option<u32>,
}
```

**Field descriptions**:

| Field         | Source                                     | Description                                                                                      |
| ------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| `id`          | `season_definitions.seasons[].id`          | 0-based season index. Count depends on `cycle_type`: 12 (monthly), 52 (weekly), or user-defined. |
| `label`       | `season_definitions.seasons[].label`       | Human-readable name for reporting.                                                               |
| `month_start` | `season_definitions.seasons[].month_start` | Calendar month (1-12) where season begins.                                                       |
| `day_start`   | `season_definitions.seasons[].day_start`   | Calendar day (1-31) for Custom cycle type. `None` for Monthly and Weekly.                        |
| `month_end`   | `season_definitions.seasons[].month_end`   | Calendar month (1-12) where season ends. Only for Custom.                                        |
| `day_end`     | `season_definitions.seasons[].day_end`     | Calendar day (1-31) where season ends. Only for Custom.                                          |

### 12.8 SeasonMap

```rust
/// Complete season definitions including cycle type and all season entries.
///
/// The SeasonMap is the resolved representation of the `season_definitions`
/// section in `stages.json`. It provides the season-to-calendar mapping
/// consumed by the PAR model and inflow history aggregation.
///
/// Source: `stages.json` `season_definitions`.
/// See [Input Scenarios §1.1](input-scenarios.md).
pub struct SeasonMap {
    /// Cycle type controlling how season IDs map to calendar periods.
    pub cycle_type: SeasonCycleType,

    /// Season entries sorted by `id`. Length depends on cycle_type:
    /// 12 for Monthly, 52 for Weekly, user-defined for Custom.
    pub seasons: Vec<SeasonDefinition>,
}
```

**Validation**: All stages sharing the same `season_id` must have exactly the same duration. Each stage's `[start_date, end_date)` interval must fall entirely within the calendar period defined by its `season_id`. See [Input Scenarios §1.10](input-scenarios.md), rules 4 and 5.

### 12.9 Transition

```rust
/// A single transition in the policy graph, representing a directed
/// edge from one stage to another with an associated probability and
/// optional discount rate override.
///
/// Transitions define the stage traversal order for both the forward
/// and backward passes. In finite horizon mode, transitions form a
/// linear chain. In cyclic mode, at least one transition creates a
/// back-edge (source_id >= target_id).
///
/// Source: `stages.json` `policy_graph.transitions[]`.
/// See [Input Scenarios §1.2](input-scenarios.md).
pub struct Transition {
    /// Source stage ID. Must exist in the stage set.
    pub source_id: i32,

    /// Target stage ID. Must exist in the stage set.
    pub target_id: i32,

    /// Transition probability. Outgoing probabilities from each source
    /// must sum to 1.0 (within tolerance).
    pub probability: f64,

    /// Per-transition annual discount rate override.
    /// When `None`, the global `annual_discount_rate` from the
    /// PolicyGraph is used. When `Some(r)`, this rate is converted to
    /// a per-transition factor using the source stage duration:
    /// d = 1 / (1 + r)^dt.
    /// See [Discount Rate §3](../math/discount-rate.md).
    pub annual_discount_rate_override: Option<f64>,
}
```

**Field descriptions**:

| Field                           | Source                                            | Description                                                                                               |
| ------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `source_id`                     | `policy_graph.transitions[].source_id`            | Source stage ID. Validated to exist in the stage set (rule H4).                                           |
| `target_id`                     | `policy_graph.transitions[].target_id`            | Target stage ID. Validated to exist in the stage set (rule H4).                                           |
| `probability`                   | `policy_graph.transitions[].probability`          | Transition probability. Sum per source must be 1.0 ([Input Scenarios §1.10](input-scenarios.md), rule 6). |
| `annual_discount_rate_override` | `policy_graph.transitions[].annual_discount_rate` | Optional per-transition override. `None` means use the global rate.                                       |

### 12.10 PolicyGraph

The `PolicyGraph` struct is the parsed, validated, clarity-first representation of the stage transition graph in `cobre-core`. It stores the raw graph topology (transitions, discount rate, horizon type) as loaded from `stages.json`. The solver-level `HorizonMode` enum -- defined in [Horizon Mode Trait SS1](../architecture/horizon-mode-trait.md) -- is a precomputed dispatch type built FROM this `PolicyGraph` during the variant selection pipeline ([Extension Points SS6](../architecture/extension-points.md)). The `PolicyGraph` captures _what_ the user specified; `HorizonMode` captures _how_ the solver will traverse it.

```rust
/// Parsed and validated policy graph defining stage transitions,
/// horizon type, and global discount rate.
///
/// This is the cobre-core clarity-first representation loaded from
/// `stages.json`. It stores the graph topology as specified by the
/// user. The solver-level `HorizonMode` enum (see Horizon Mode Trait
/// SS1) is built from this struct during initialization — it
/// precomputes transition maps, cycle detection, and discount factors
/// for efficient runtime dispatch.
///
/// Cross-reference: [Horizon Mode Trait](../architecture/horizon-mode-trait.md)
/// defines the `HorizonMode` enum that interprets this graph structure.
/// `PolicyGraphConfig` in that spec is the serde-level deserialization
/// type; this `PolicyGraph` is the validated cobre-core representation.
///
/// Source: `stages.json` `policy_graph`.
/// See [Input Scenarios §1.2](input-scenarios.md).
pub struct PolicyGraph {
    /// Horizon type: finite (acyclic chain) or cyclic (infinite periodic).
    /// Determines which `HorizonMode` variant will be constructed at
    /// solver initialization.
    pub graph_type: PolicyGraphType,

    /// Global annual discount rate.
    /// Converted to per-transition factors using source stage durations:
    /// d = 1 / (1 + annual_discount_rate)^dt.
    /// A value of 0.0 means no discounting (d = 1.0 for all transitions).
    /// For cyclic graphs, must be > 0 for convergence (validation rule 7).
    /// See [Discount Rate §3](../math/discount-rate.md).
    pub annual_discount_rate: f64,

    /// Stage transitions with probabilities and optional per-transition
    /// discount rate overrides. For finite horizon, these form a linear
    /// chain or DAG. For cyclic horizon, at least one transition has
    /// source_id >= target_id (the back-edge).
    pub transitions: Vec<Transition>,

    /// Season definitions loaded from `season_definitions` in
    /// `stages.json`. Required when PAR models or inflow history
    /// aggregation are used. `None` when no season definitions are
    /// provided and none are required.
    pub season_map: Option<SeasonMap>,
}
```

**Field descriptions**:

| Field                  | Source                              | Description                                                                                                                                                   |
| ---------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `graph_type`           | `policy_graph.type`                 | Horizon type tag. Reuses the `PolicyGraphType` enum (`FiniteHorizon`, `Cyclic`) defined in [Horizon Mode Trait SS3.1](../architecture/horizon-mode-trait.md). |
| `annual_discount_rate` | `policy_graph.annual_discount_rate` | Global annual rate (e.g., 0.06 = 6%). Converted to per-transition factors at solver initialization.                                                           |
| `transitions`          | `policy_graph.transitions`          | Directed edges with probabilities. Validated by rules in [Input Scenarios §1.10](input-scenarios.md).                                                         |
| `season_map`           | `season_definitions`                | Season-to-calendar mapping. Required when `inflow_history` is provided; optional otherwise.                                                                   |

> **Relationship to `HorizonMode`**: The `PolicyGraph` is a data structure -- it stores _what_ the user configured. The `HorizonMode` enum in [Horizon Mode Trait](../architecture/horizon-mode-trait.md) is an algorithm dispatch type -- it stores _precomputed_ transition maps (`TransitionMap`), cycle metadata, and discount factors for efficient O(1) lookup during the training loop. The conversion from `PolicyGraph` to `HorizonMode` happens in `HorizonMode::validate()` (Horizon Mode Trait SS2.4), which enforces rules H1-H4 and precomputes all derived quantities. This separation keeps `cobre-core` free of solver-specific precomputation while ensuring `cobre-sddp` has the runtime-optimized data it needs.

> **`PolicyGraphType` reuse**: The `PolicyGraph` struct uses the `PolicyGraphType` enum (`FiniteHorizon`, `Cyclic`) already defined in [Horizon Mode Trait SS3.1](../architecture/horizon-mode-trait.md). This type is simple enough (two-variant tag with no data) to be shared between `cobre-core` and `cobre-sddp` without introducing coupling. It lives in `cobre-core` and is re-exported by both crates.

## 13. Block Factors

Block factors modify LP bounds and demand values per (entity, stage, block). They are loaded during initialization and used when constructing LP definitions.

- **Load factors**: Per (bus, stage, block) multipliers on stochastic load demand. Affect the actual demand value in the load balance constraint.
- **Exchange factors**: Per (line, stage, block) multipliers on line capacity (direct and reverse). Affect the line flow bounds in the LP — a different role from load factors. Values >1.0 represent higher block-level capacity.

Default: 1.0 when not specified.

## 14. Scenario Pipeline

The scenario pipeline state loaded in memory:

### Inflow Models (Per Hydro, Per Stage)

Pre-resolved seasonal statistics and AR coefficients:

- `mean_m3s` (μ) — seasonal mean inflow
- `std_m3s` (s_m) — seasonal sample standard deviation (NOT the residual std σ_m)
- AR coefficients `[ψ*_1, ..., ψ*_p]` standardized by seasonal std (variable length, can be 0). AR order p is derived from the count of coefficient rows per (hydro, stage).
- `residual_std_ratio` (σ_m / s_m) — fraction of seasonal variability not explained by AR lags, in (0, 1]

Source: user-provided `inflow_seasonal_stats.parquet` (μ, s_m) + optional `inflow_ar_coefficients.parquet` (ψ\* per lag, residual_std_ratio), OR derived from `inflow_history` via season aggregation and Yule-Walker fitting. See [Input Scenarios §2-3](input-scenarios.md).

### Load Seasonal Statistics (Per Bus, Per Stage)

- `mean_mw`, `std_mw` — load statistics (typically no AR structure)

### Correlation Model

- Named profiles, each containing correlation groups with entity lists and correlation matrices
- Stage-to-profile mapping (schedule) — pre-resolved so the solver can look up the active profile per stage
- Spectral factors ready for scenario generation

Source: user-provided `correlation.json` OR estimated from AR residuals of inflow history. See [Input Scenarios §5](input-scenarios.md).

### PrecomputedParLp

The `PrecomputedParLp` struct caches LP-ready values derived from the PAR(p) model parameters. It is built once during initialization and consumed at every forward-pass stage transition when patching the LP right-hand side for the inflow balance constraint. The derivation of the three cached components is in [PAR Inflow Model §7](../math/par-inflow-model.md).

The lagged inflows $a_{h,t-\ell}$ are LP variables whose coefficients $-\psi_{m(t),\ell}$ appear in the constraint matrix of the AR dynamics row. They are fixed to incoming state values by separate lag fixing constraints (see [LP Formulation §5a](../math/lp-formulation.md)). Because the lag contribution lives in the constraint matrix rather than the RHS, the AR dynamics constraint RHS reduces to:

$$
\text{RHS}_{h,t} = b_{h,m(t)} + \sigma_{m(t)} \cdot \varepsilon_t
$$

```rust
/// LP-ready precomputed values derived from PAR(p) model parameters.
///
/// Cached once after model fitting. Two uses:
/// - `psi`: written into the AR dynamics constraint matrix once at LP construction
///   (coefficients on the lagged inflow LP variables; see LP Formulation §5a).
/// - `deterministic_base` + `sigma`: used to patch the AR dynamics constraint RHS
///   at each forward-pass stage transition:
///     RHS_{h,t} = deterministic_base[stage][hydro] + sigma[stage][hydro] * noise_draw
struct PrecomputedParLp {
    /// ψ coefficients in original units, indexed [stage][hydro][lag].
    /// Shape: [T][H][p_max] where p_max is the maximum PAR order across all seasons.
    /// Written into the constraint matrix at LP construction time; not used per-scenario.
    psi: Vec<Vec<Vec<f64>>>,
    /// Deterministic base b_{h,m(t)} = μ_{m(t)} - Σ ψ_{m(t),ℓ} μ_{m(t-ℓ)},
    /// indexed [stage][hydro]. Added to the AR dynamics constraint RHS at each stage.
    deterministic_base: Vec<Vec<f64>>,
    /// Per-season σ_{m(t)} (original units), indexed [stage][hydro].
    /// Multiplied by the scenario noise draw to form the stochastic RHS component.
    sigma: Vec<Vec<f64>>,
}
```

**Field descriptions**:

| Field                | Type                 | Description                                                                                                                                                                                                                                                                                                                                           |
| -------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `psi`                | `Vec<Vec<Vec<f64>>>` | AR lag coefficients $\psi_{m(t),\ell}$ in original units, indexed `[stage][hydro][lag]`. Shape is `[T][H][p_max]` where `p_max` is the maximum PAR order across all seasons. Trailing entries are 0.0 for hydros with lower order. Written into the AR dynamics constraint matrix at LP construction time; not used during per-scenario RHS patching. |
| `deterministic_base` | `Vec<Vec<f64>>`      | Precomputed constant $b_{h,m(t)} = \mu_{m(t)} - \sum_{\ell} \psi_{m(t),\ell} \cdot \mu_{m(t-\ell)}$ per (stage, hydro). Absorbs the mean-adjustment arithmetic so the hot path avoids repeated subtraction and multiplication.                                                                                                                        |
| `sigma`              | `Vec<Vec<f64>>`      | Residual standard deviation $\sigma_{m(t)}$ per (stage, hydro) in original units (m3/s). Multiplied by the scenario noise draw $\varepsilon_t$ to produce the stochastic innovation component of the LP RHS.                                                                                                                                          |

Caching these three arrays eliminates repeated multiply-divide operations (coefficient conversion, mean subtraction, standard deviation scaling) on the hot forward-pass path. The struct belongs to the `cobre-sddp` performance layer, not `cobre-core`, following the dual-nature design in §1.1: `cobre-core` stores the raw PAR parameters (`ParModel` in §1.3) while `cobre-sddp` builds this precomputed view at initialization from `&System`. The conversion is a one-time $O(T \cdot H \cdot p_{\max})$ cost that pays for itself over thousands of forward-pass iterations.

## 15. Generic Constraints

User-defined linear constraints parsed and validated during loading. See [Input Constraints §3](input-constraints.md).

### Constraint Definition

Each generic constraint has:

- `id`, `name`, `description`
- **Parsed expression**: A list of linear terms, each being `coefficient x variable_reference`, plus a constant
- **Sense**: `>=`, `<=`, or `==`
- **Slack config**: whether a slack variable is created, and its penalty cost

### Variable References

The expression parser produces typed variable references. The full catalog:

| Variable                       | Entity                  | Block-specific                      |
| ------------------------------ | ----------------------- | ----------------------------------- |
| `hydro_storage`                | hydro                   | No (stage-level)                    |
| `hydro_turbined`               | hydro                   | Yes                                 |
| `hydro_spillage`               | hydro                   | Yes                                 |
| `hydro_diversion`              | hydro                   | Yes                                 |
| `hydro_outflow`                | hydro                   | Yes (alias for turbined + spillage) |
| `hydro_generation`             | hydro                   | Yes                                 |
| `hydro_evaporation`            | hydro                   | No                                  |
| `hydro_withdrawal`             | hydro                   | No                                  |
| `thermal_generation`           | thermal                 | Yes                                 |
| `line_direct`                  | line                    | Yes                                 |
| `line_reverse`                 | line                    | Yes                                 |
| `bus_deficit`                  | bus                     | Yes                                 |
| `bus_excess`                   | bus                     | Yes                                 |
| `pumping_flow`                 | pumping station         | Yes                                 |
| `pumping_power`                | pumping station         | Yes                                 |
| `contract_import`              | contract                | Yes                                 |
| `contract_export`              | contract                | Yes                                 |
| `non_controllable_generation`  | non-controllable source | Yes                                 |
| `non_controllable_curtailment` | non-controllable source | Yes                                 |

### Constraint Bounds

Pre-resolved per (constraint_id, stage_id, block_id). If no bound exists for a (constraint, stage), the constraint is not active for that stage.

### Validation (at load time)

All entity IDs referenced in constraint expressions must exist in the loaded system. Block IDs (if specified) must be valid for the stage. Expression parsing errors are reported at load time, not at LP build time.

## 16. Initial Conditions

Pre-resolved initial state:

- **Storage** (operating hydros): `value_hm3` per hydro, validated within `[min_storage_hm3, max_storage_hm3]`
- **Filling storage** (filling hydros): `value_hm3` per hydro, validated within `[0, min_storage_hm3]`
- **GNL pipeline**: For each GNL thermal, the committed dispatch (MW) for each future stage within the lag window. Represented as `(thermal_id, stage_offset, committed_mw)` tuples where `stage_offset` ranges from 1 to `lag_stages`. These values initialize the GNL state variables at the start of the study. See [Input Constraints §1](input-constraints.md).

> **Note**: GNL pipeline initial conditions are currently rejected by validation (GNL thermals are not accepted). When GNL support is implemented, the validation rule will be relaxed and these initial conditions will be required for any thermal with `gnl_config` defined.

See [Input Constraints §1](input-constraints.md).

## Cross-References

- [Input System Entities](input-system-entities.md) — JSON schemas that map to entity representations
- [Input Hydro Extensions](input-hydro-extensions.md) — Geometry, production models, FPHA hyperplanes
- [Input Constraints](input-constraints.md) — Initial conditions, time-varying bounds, generic constraints
- [Input Scenarios](input-scenarios.md) — Stage/block, inflow, and load data, correlation
- [Penalty System](penalty-system.md) — Penalty cascade resolution and categories
- [Binary Formats](binary-formats.md) — Serialization format decisions, FlatBuffers policy schema, cut pool persistence
- [Scenario Generation](../architecture/scenario-generation.md) — Opening tree lifecycle (§2.3), sampling scheme abstraction (§3)
- [LP Formulation](../math/lp-formulation.md) — Stage subproblem LP that consumes these internal structures at runtime
- [Design Principles](../overview/design-principles.md) — Order invariance requirement (§3)
- [Input Loading Pipeline](../architecture/input-loading-pipeline.md) — File loading sequence, broadcast strategy, transition to in-memory model (sections 2, 6, 8)
- [Training Loop](../architecture/training-loop.md) — SDDP algorithm that consumes System via shared reference (section 1)
- [Solver Abstraction](../architecture/solver-abstraction.md) — Stage LP template design built from System data (section 11)
- [Solver Workspaces](../architecture/solver-workspaces.md) — Thread-local workspace layout for performance-adapted views
- [Implementation Ordering](../overview/implementation-ordering.md) — Build sequence with cobre-core as foundation (section 4)
