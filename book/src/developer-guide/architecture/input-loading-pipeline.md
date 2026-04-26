# Input Loading Pipeline

## Purpose

This spec defines the Cobre input loading architecture: the rank-0 centric loading pattern, file loading sequence with dependency ordering, dependency resolution, conditional loading rules, sparse time-series expansion, data broadcasting strategy, parallel policy loading for warm-start, and the transition to the in-memory data model.

For the file inventory and directory layout, see [Input Directory Structure](../data-model/input-directory-structure.md).
For the in-memory data model after loading completes, see [Internal Structures](../data-model/internal-structures.md).

## 1. Loading Architecture

Input loading follows a **rank-0 centric** pattern: rank 0 loads and validates all input data, then broadcasts to worker ranks. This design:

- Minimizes filesystem contention on parallel filesystems
- Centralizes validation logic on a single rank
- Reduces complexity of error handling across ranks
- Ensures all ranks receive identical, validated data

The one exception is **policy loading for warm-start** (SS7), where all ranks load in parallel to avoid bottlenecking on large policy files.

```mermaid
flowchart TB
    subgraph R0 ["Rank 0 — load + validate"]
        direction TB
        C["config.json<br/><i>schema + parse</i>"]
        ST["stages.json<br/><i>branching, noise</i>"]
        SE["System entities <i>(dependency order)</i><br/>buses → lines → hydros → thermals → ncs<br/><i>each: schema validate → cross-ref → index</i>"]
        SD["Stochastic data<br/>seasonal_stats → ar_coefficients → correlation → external/<br/><i>PAR fit (if history) or load precomputed</i>"]
        CN["Constraints<br/><i>generic_constraints + coefficients</i>"]
        VS(["Validated System"])
        C --> ST --> SE --> SD --> CN --> VS
    end

    BC(["broadcast"])
    ALL["All other ranks:<br/>receive System, allocate, set up solver"]
    PW["Policy warm-start<br/><i>all ranks read in parallel (exception)</i>"]

    VS --> BC --> ALL
    PW -.-> ALL
```

Fail-fast: the first validation error aborts the pipeline. All ranks receive identical validated data.

## 2. File Loading Sequence

Files are loaded in dependency order so that each file can be validated against previously loaded data. Loading fails fast on the first error.

**Schema validation first:** Every JSON file undergoes full schema validation before any other checks. This includes required vs. optional fields, data types, value ranges, and structural constraints (e.g., array lengths, enum values). Similarly, every Parquet file is validated for expected columns, column types, and per-column value constraints. The "Validation" column in the tables below lists only the **additional** cross-reference and semantic checks performed after schema validation passes.

> **Note:** The validations listed in this section are illustrative, not exhaustive. Additional validations may be introduced during implementation or as a result of reviewing other specs. See [Validation Architecture](./validation-architecture.md) for the complete multi-layer validation design.

### 2.1 Root-Level Files

| Order | File                      | Dependencies | Validation                                              |
| ----- | ------------------------- | ------------ | ------------------------------------------------------- |
| 1     | `config.json`             | None         | JSON schema, execution mode, section structure          |
| 2     | `stages.json`             | config       | Stage count, season mapping, policy graph, block counts |
| 3     | `penalties.json`          | None         | All penalty values > 0, required categories present     |
| 4     | `initial_conditions.json` | None         | Array lengths deferred until entity registries loaded   |

### 2.2 System Entity Registries

| Order | File                                   | Dependencies  | Validation                                       |
| ----- | -------------------------------------- | ------------- | ------------------------------------------------ |
| 5     | `system/buses.json`                    | None          | Bus IDs unique                                   |
| 6     | `system/lines.json`                    | buses         | Source/target bus references valid               |
| 7     | `system/hydros.json`                   | buses         | Bus references valid, cascade references acyclic |
| 8     | `system/thermals.json`                 | buses         | Bus references valid                             |
| 9     | `system/non_controllable_sources.json` | buses         | Bus references valid (optional file)             |
| 10    | `system/pumping_stations.json`         | hydros, buses | Hydro and bus references valid (optional file)   |
| 11    | `system/energy_contracts.json`         | buses         | Bus references valid (optional file)             |

### 2.3 System Extension Data

| Order | File                                  | Dependencies   | Validation                                                        |
| ----- | ------------------------------------- | -------------- | ----------------------------------------------------------------- |
| 12    | `system/hydro_geometry.parquet`       | hydros         | Hydro ID coverage, monotonic volume-area-level curves             |
| 13    | `system/hydro_production_models.json` | hydros, stages | Hydro/stage references valid (optional file)                      |
| 14    | `system/fpha_hyperplanes.parquet`     | hydros         | Required only when FPHA source is `"precomputed"` (optional file) |

### 2.4 Scenario Data

| Order | File                                       | Dependencies     | Validation                                                    |
| ----- | ------------------------------------------ | ---------------- | ------------------------------------------------------------- |
| 15    | `scenarios/inflow_seasonal_stats.parquet`  | hydros, stages   | Hydro/stage coverage (optional file)                          |
| 16    | `scenarios/inflow_ar_coefficients.parquet` | hydros, stages   | Hydro/stage/lag coverage, consistent AR order (optional file) |
| 17    | `scenarios/inflow_history.parquet`         | hydros           | Hydro ID coverage (optional file)                             |
| 18    | `scenarios/load_seasonal_stats.parquet`    | buses, stages    | Bus/stage coverage (optional file)                            |
| 19    | `scenarios/load_factors.json`              | stages           | Block count consistency (optional file)                       |
| 20    | `scenarios/correlation.json`               | hydros           | Group membership covers all hydros (optional file)            |
| 21    | `scenarios/external_scenarios.parquet`     | entities, stages | Entity/stage/scenario coverage (optional file)                |

### 2.5 Constraints and Overrides

| Order | File                                            | Dependencies                | Validation                                        |
| ----- | ----------------------------------------------- | --------------------------- | ------------------------------------------------- |
| 22    | `constraints/thermal_bounds.parquet`            | thermals, stages            | Entity/stage references valid (optional file)     |
| 23    | `constraints/hydro_bounds.parquet`              | hydros, stages              | Entity/stage references valid (optional file)     |
| 24    | `constraints/line_bounds.parquet`               | lines, stages               | Entity/stage references valid (optional file)     |
| 25    | `constraints/pumping_bounds.parquet`            | pumping, stages             | Entity/stage references valid (optional file)     |
| 26    | `constraints/contract_bounds.parquet`           | contracts, stages           | Entity/stage references valid (optional file)     |
| 27    | `constraints/exchange_factors.json`             | lines, stages               | Line/stage references valid (optional file)       |
| 28    | `constraints/generic_constraints.json`          | entities                    | Entity references valid (optional file)           |
| 29    | `constraints/generic_constraint_bounds.parquet` | generic constraints, stages | Constraint/stage references valid (optional file) |
| 30    | `constraints/penalty_overrides_bus.parquet`     | buses, stages               | Entity/stage references valid (optional file)     |
| 31    | `constraints/penalty_overrides_line.parquet`    | lines, stages               | Entity/stage references valid (optional file)     |
| 32    | `constraints/penalty_overrides_hydro.parquet`   | hydros, stages              | Entity/stage references valid (optional file)     |
| 33    | `constraints/penalty_overrides_ncs.parquet`     | NCS, stages                 | Entity/stage references valid (optional file)     |

### 2.6 Cross-Reference Validation

After all files are loaded, a final cross-reference validation pass checks every inter-entity reference for existence, consistency, and structural soundness. This pass corresponds to Layers 3 and 4 of the [Validation Architecture](./validation-architecture.md) pipeline. Failures produce `CrossReferenceError` or `ConstraintError` variants of `LoadError` (SS8.1).

The following table enumerates **all 26 cross-reference validation rules**. Rules are grouped by source entity type, followed by structural and coverage checks. Per-file schema validations (field types, required fields, value ranges) are NOT repeated here — those are covered by the "Schema validation first" note at the top of SS2.

| #   | Source Entity           | Reference Field                           | Target Entity       | Validation Rule                                                                                                                                                   | Error on Failure                                                                                                                            |
| --- | ----------------------- | ----------------------------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `Line`                  | `source_bus_id`                           | `Bus`               | Must resolve to an existing bus ID in `buses.json`                                                                                                                | `CrossReferenceError`: line references non-existent source bus                                                                              |
| 2   | `Line`                  | `target_bus_id`                           | `Bus`               | Must resolve to an existing bus ID in `buses.json`                                                                                                                | `CrossReferenceError`: line references non-existent target bus                                                                              |
| 3   | `Line`                  | `source_bus_id`, `target_bus_id`          | `Bus`               | `source_bus_id` must differ from `target_bus_id` (no self-loops)                                                                                                  | `ConstraintError`: line has identical source and target bus                                                                                 |
| 4   | `Thermal`               | `bus_id`                                  | `Bus`               | Must resolve to an existing bus ID in `buses.json`                                                                                                                | `CrossReferenceError`: thermal references non-existent bus                                                                                  |
| 5   | `Hydro`                 | `bus_id`                                  | `Bus`               | Must resolve to an existing bus ID in `buses.json`                                                                                                                | `CrossReferenceError`: hydro references non-existent bus                                                                                    |
| 6   | `Hydro`                 | `downstream_id`                           | `Hydro`             | When non-null, must resolve to an existing hydro ID in `hydros.json`                                                                                              | `CrossReferenceError`: hydro references non-existent downstream hydro                                                                       |
| 7   | `Hydro`                 | `diversion.downstream_id`                 | `Hydro`             | When diversion is present, must resolve to an existing hydro ID in `hydros.json`                                                                                  | `CrossReferenceError`: hydro diversion references non-existent destination hydro                                                            |
| 8   | `PumpingStation`        | `bus_id`                                  | `Bus`               | Must resolve to an existing bus ID in `buses.json`                                                                                                                | `CrossReferenceError`: pumping station references non-existent bus                                                                          |
| 9   | `PumpingStation`        | `source_hydro_id`                         | `Hydro`             | Must resolve to an existing hydro ID in `hydros.json`                                                                                                             | `CrossReferenceError`: pumping station references non-existent source hydro                                                                 |
| 10  | `PumpingStation`        | `destination_hydro_id`                    | `Hydro`             | Must resolve to an existing hydro ID in `hydros.json`                                                                                                             | `CrossReferenceError`: pumping station references non-existent destination hydro                                                            |
| 11  | `PumpingStation`        | `source_hydro_id`, `destination_hydro_id` | `Hydro`             | `source_hydro_id` must differ from `destination_hydro_id` (no self-pumping)                                                                                       | `ConstraintError`: pumping station has identical source and destination hydro                                                               |
| 12  | `EnergyContract`        | `bus_id`                                  | `Bus`               | Must resolve to an existing bus ID in `buses.json`                                                                                                                | `CrossReferenceError`: energy contract references non-existent bus                                                                          |
| 13  | `NonControllableSource` | `bus_id`                                  | `Bus`               | Must resolve to an existing bus ID in `buses.json`                                                                                                                | `CrossReferenceError`: non-controllable source references non-existent bus                                                                  |
| 14  | `GenericConstraint`     | entity IDs in `expression`                | Multiple            | Every entity ID in the parsed expression must exist in the corresponding entity registry (hydro, thermal, line, bus, pumping station, or contract)                | `CrossReferenceError`: generic constraint expression references non-existent entity                                                         |
| 15  | `GenericConstraint`     | `constraint_id` (in bounds file)          | `GenericConstraint` | Every `constraint_id` in `generic_constraint_bounds.parquet` must reference an existing constraint definition in `generic_constraints.json`                       | `CrossReferenceError`: constraint bounds reference non-existent constraint definition                                                       |
| 16  | `Hydro` (cascade)       | `downstream_id` (all hydros)              | `Hydro`             | The directed graph formed by all `downstream_id` references must be acyclic (DAG). Cycle detection via topological sort or DFS                                    | `ConstraintError`: hydro cascade contains a cycle (list participating hydro IDs)                                                            |
| 17  | `InitialConditions`     | `hydro_id` (in `storage`)                 | `Hydro`             | Every `hydro_id` in the `storage` array must exist in the hydro registry                                                                                          | `CrossReferenceError`: initial conditions reference non-existent hydro                                                                      |
| 18  | `InitialConditions`     | `hydro_id` (in `filling_storage`)         | `Hydro`             | Every `hydro_id` in the `filling_storage` array must exist in the hydro registry and must have a `filling` configuration                                          | `CrossReferenceError`: filling initial conditions reference non-existent or non-filling hydro                                               |
| 19  | `InitialConditions`     | coverage check                            | `Hydro`             | Every operating hydro must appear in `storage`; every filling hydro must appear in `filling_storage`. No hydro appears in both arrays. No extra entries allowed   | `ConstraintError`: initial conditions coverage mismatch (missing hydro or duplicate entry)                                                  |
| 20  | `PolicyGraph`           | transition `source_id`, `target_id`       | `Stage`             | Every stage ID referenced in policy graph transitions must exist in the stage collection                                                                          | `CrossReferenceError`: policy graph transition references non-existent stage                                                                |
| 21  | Inflow model            | hydro coverage                            | `Hydro`             | Every operating hydro with PAR-based scenario generation must have corresponding entries in `inflow_seasonal_stats.parquet` and `inflow_ar_coefficients.parquet`  | `ConstraintError`: inflow model coverage incomplete (list missing hydro IDs)                                                                |
| 22  | FPHA hyperplanes        | `hydro_id` coverage                       | `Hydro`             | Every hydro with FPHA source `"precomputed"` must have hyperplane entries in `fpha_hyperplanes.parquet`. Hydros with source `"computed"` do not require this file | `ConstraintError`: missing precomputed FPHA hyperplanes for hydro (planes are generated during Initialization only for source `"computed"`) |
| 23  | All entities            | `entry_stage_id`                          | `Stage`             | When non-null, must resolve to an existing stage ID in the stage collection. The entry stage marks when the entity becomes active                                 | `CrossReferenceError`: entity references non-existent entry stage                                                                           |
| 24  | All entities            | `exit_stage_id`                           | `Stage`             | When non-null, must resolve to an existing stage ID in the stage collection. Must be >= `entry_stage_id` when both are present                                    | `CrossReferenceError`: entity references non-existent exit stage, or exit precedes entry                                                    |
| 25  | `Hydro`                 | `filling.start_stage_id`                  | `Stage`             | When filling configuration is present, must resolve to an existing stage ID. Must fall within [`entry_stage_id`, `exit_stage_id`] range when lifecycle is bounded | `CrossReferenceError`: hydro filling references non-existent start stage or stage outside lifecycle                                         |
| 26  | `Thermal`               | `gnl_config.lag_stages`                   | `Stage` (implicit)  | When GNL config is present, `lag_stages` must be positive and the thermal's lifecycle must span at least `lag_stages + 1` stages for dispatch anticipation        | `ConstraintError`: GNL lag exceeds thermal lifecycle span                                                                                   |

**Execution order within SS2.6.** Rules 1--15 and 23--26 (referential integrity, Layer 3) execute first. Rules 16--22 (structural and dimensional, Layers 4--5) execute after all foreign-key references are confirmed valid, because structural checks such as cascade acyclicity (rule 16) depend on all `downstream_id` references being resolved.

**Error collection.** All 26 checks run to completion (within their layer), collecting every violation. The user sees every cross-reference problem in a single validation report, consistent with the error collection strategy in [Validation Architecture](./validation-architecture.md) SS3.

### 2.7 Policy Files (Warm-Start Only)

| Order | File       | Dependencies | Validation                                                |
| ----- | ---------- | ------------ | --------------------------------------------------------- |
| 34    | `policy/*` | All above    | State dictionary matches current system, cut format valid |

Policy loading uses a different pattern — see SS7 Parallel Policy Loading.

## 3. Dependency Graph

Input files form a directed acyclic graph (DAG) of dependencies. The loading sequence in SS2 is a valid topological ordering of this DAG. Files at the same dependency level may be loaded in any order relative to each other.

> **Placeholder** — The dependency graph diagram (`../../diagrams/exports/svg/data/json-schema-dependencies.svg`) will be revised after the text review is complete.

## 4. Conditional Loading

Some files are loaded only when certain conditions are met. Missing optional files are not errors — the loader uses defaults (typically empty collections or identity values).

| Condition                                                 | Effect                                                                                          |
| --------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `training.enabled = false` in `config.json`               | Skip scenario noise generation (but still load models if simulation needs them)                 |
| `simulation.enabled = false` in `config.json`             | Skip simulation-specific scenario setup                                                         |
| `policy.mode = "warm_start"` in `config.json`             | Load `policy/*` files (SS7)                                                                     |
| Policy graph has cycle (`stages.json`)                    | Validate cycle structure per [Infinite Horizon](../math/infinite-horizon.md)                    |
| Hydros with FPHA production model, source `"precomputed"` | Require `fpha_hyperplanes.parquet`                                                              |
| Hydros with FPHA production model, source `"computed"`    | FPHA hyperplanes computed during Initialization phase from geometry and topology data (see SS8) |
| `pumping_stations.json` present                           | Load pumping bounds, validate hydro/bus references                                              |
| `energy_contracts.json` present                           | Load contract bounds                                                                            |
| `non_controllable_sources.json` present                   | Load NCS penalty overrides                                                                      |
| `external_scenarios.parquet` present                      | Use external scenarios instead of PAR-generated ones                                            |

> **Note on external scenarios scope**: The presence of `external_scenarios.parquet` does NOT imply simulation-only usage. External scenarios can also be used during training — see [Scenario Generation SS4.2](./scenario-generation.md). Loading and validation apply identically regardless of the target phase.

## 5. Sparse Time-Series Expansion

Time-series Parquet files (bounds, penalty overrides) use **sparse representation**: only rows with non-default values are stored. Stages and entities not present in the file receive default values.

**Expansion behavior:**

1. Load the sparse Parquet file (only rows with explicit values).
2. Build a dense (stage × entity) structure initialized with defaults.
3. Overlay the sparse values onto the dense structure.
4. Validate that all stage IDs and entity IDs in the sparse data are valid (reference loaded registries).

**Default values:** Each file type has its own default. For bounds files, the default is the static bound from the entity registry. For penalty overrides, the default is the global value from `penalties.json`. This is defined per file in the relevant data model spec — see [Input System Entities](../data-model/input-system-entities.md) and [Penalty System](../data-model/penalty-system.md).

## 6. Broadcast Strategy

After rank 0 loads and validates all data, it broadcasts to worker ranks. Data is serialized to contiguous byte buffers for MPI broadcast. The broadcast uses a two-step protocol: first the buffer size, then the buffer contents.

**Broadcast categories:**

| Category          | Data                                                       | Strategy                                         |
| ----------------- | ---------------------------------------------------------- | ------------------------------------------------ |
| Small objects     | Config, stages, penalties, initial conditions              | Single `MPI_Bcast`                               |
| Entity registries | Buses, lines, hydros, thermals, NCS, pumping, contracts    | Single `MPI_Bcast`                               |
| Scenario models   | PAR parameters, correlation matrices, load models          | Single `MPI_Bcast`                               |
| Bounds/overrides  | All constraint bounds, penalty overrides, exchange factors | Single `MPI_Bcast` (sparse form, expand locally) |
| Policy cuts       | FCF cuts for warm-start                                    | Parallel load (SS7)                              |

**Sparse broadcast optimization:** Bounds and penalty override files are broadcast in their sparse Parquet form. Each rank performs the sparse-to-dense expansion locally (SS5). This reduces broadcast volume — only non-default values are transmitted.

### 6.1 Serialization Format

The serialization format for MPI broadcast is **postcard** (via serde).

postcard is a compact binary serde format. Rank 0 serializes the `System` struct via `postcard::to_vec(&system)`, producing a `Vec<u8>` byte buffer. The buffer is broadcast via `MPI_Bcast` (two-step: size first, then contents). Receiving ranks deserialize via `postcard::from_bytes::<System>(&buffer)` into an owned `System` value. The format uses variable-length integer encoding (varint) for compact payloads and serde's trait-based dispatch for handling all Rust types including `Vec`, `String`, `Option`, enums with data, and nested structs.

**Why postcard over rkyv:** rkyv was previously specified for this use case based on its zero-copy deserialization capability. An evidence-based evaluation (see `plans/implementation-readiness-audit/epic-07-serialization-eval-and-output-api/report-027-rkyv-evaluation.md`) found that the zero-copy benefit is immaterial for a once-per-execution operation: the marginal deserialization saving is under 2 ms for a ~6 MB payload, occurring exactly once in a program that runs for minutes to hours. Meanwhile, rkyv imposes 129 additional derive annotations across 43 types (on top of the 86 serde derives already required for JSON loading), requires wrapper types for external types like `chrono::NaiveDate`, and is pre-1.0 with a history of breaking API changes. postcard reuses the serde derives that all cobre-core types must implement for JSON input loading, adding zero additional trait burden. It is post-1.0 stable, has a minimal dependency footprint, and produces slightly smaller payloads than rkyv due to varint encoding.

**Why not bincode:** bincode was considered but is unmaintained. Its last release predates the current Rust edition.

**Sparse broadcast and Parquet pass-through:** For bounds and penalty override data broadcast in sparse Parquet form (see broadcast categories table above), postcard serializes only the metadata envelope (entity IDs, stage ranges, file identity). The raw Parquet bytes are passed through as-is in a separate broadcast buffer — they are not re-serialized through postcard. Each rank deserializes the metadata envelope via postcard and then performs sparse-to-dense expansion locally from the Parquet bytes.

### 6.2 Required Trait Bounds

All types that participate in the `System` broadcast must derive `serde::Serialize` and `serde::Deserialize`. These are the same trait bounds already required for JSON input loading — no additional broadcast-specific derives are needed. The complete list of types requiring these trait bounds:

**Entity types:**

- `System`, `Bus`, `Line`, `Hydro`, `Thermal`, `PumpingStation`, `EnergyContract`, `NonControllableSource`

**Temporal and topological types:**

- `Stage`, `PolicyGraph`, `CascadeTopology`, `NetworkTopology`

**Pre-resolved data types:**

- `ResolvedPenalties`, `ResolvedBounds`

**Scenario pipeline types:**

- `ParModel`, `CorrelationModel`

**Other types:**

- `InitialConditions`, `GenericConstraint`

All nested types within these top-level types (e.g., tagged union variants within `Hydro`, individual bound entries within `ResolvedBounds`) must also satisfy the same trait bounds transitively. External types such as `chrono::NaiveDate` implement serde traits natively via the `chrono/serde` feature flag.

**Derive example:**

```rust
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Hydro { /* ... */ }
```

**HashMap lookup indices are NOT serialized.** The `System` struct contains `HashMap<EntityId, usize>` fields (`bus_index`, `hydro_index`, `thermal_index`, etc.) that serve as O(1) lookup indices from entity ID to position in the corresponding `Vec`. These indices are **excluded** from serialization via `#[serde(skip)]`. Each receiving rank rebuilds them locally from the deserialized entity collections. This is correct because the indices are derived data — they are deterministically reconstructable from the entity vectors — and excluding them reduces the broadcast payload size and avoids serializing HashMap internal layout, which is not stable across allocator states.

### 6.3 Buffer Allocation

postcard produces a standard `Vec<u8>` with no special alignment requirements. The MPI receive buffer on worker ranks is a standard `Vec<u8>` allocated to the exact required size.

Rank 0 serializes via `postcard::to_vec(&system)`, which returns a `Vec<u8>`. The two-step broadcast protocol (size first, then contents) allows worker ranks to allocate a receive buffer of the exact required size before the second `MPI_Bcast` call. No aligned allocation or special allocator is needed.

### 6.4 Versioning Scope

postcard does **not** provide built-in schema evolution. The binary format is derived from the serde `Serialize`/`Deserialize` implementations, which are tied to the Rust struct layout — any change to field order, field types, or field count produces an incompatible archive. There is no field tagging, no optional field mechanism, and no forward/backward compatibility guarantee.

This is acceptable because postcard is used **exclusively for in-memory MPI broadcast** within a single program execution. All ranks run the same binary, so the struct layout is identical on the serializing and deserializing sides. There is no cross-version compatibility requirement for broadcast data — the buffer exists only for the duration of the broadcast operation and is never persisted to disk.

**postcard is NOT suitable for long-term storage.** Any data that must survive across program versions (policy cuts, checkpoints, warm-start files) uses FlatBuffers, which provides schema evolution and cross-version compatibility. See [Binary Formats](../data-model/binary-formats.md) SS3 for the FlatBuffers policy persistence format.

## 7. Parallel Policy Loading (Warm-Start)

Policy files (cuts, states, vertices, basis) can be large. Loading them on rank 0 and broadcasting would create a bottleneck. Instead, all ranks load in parallel:

1. Rank 0 loads and broadcasts the `policy/metadata.json` and `policy/state_dictionary.json` (small files).
2. All ranks validate the state dictionary against the current system (entity count, state variable mapping).
3. Each rank loads a subset of policy stage files from `policy/cuts/` — stages are assigned round-robin by rank.
4. After local loading, ranks exchange cuts so that every rank has the complete policy. The exchange strategy (e.g., `MPI_Allgatherv` or shared memory window + inter-node broadcast) is an implementation choice.

For the policy file format (FlatBuffers `.bin` files), see [Binary Formats](../data-model/binary-formats.md) SS3.2.

### SS7.1 Warm-Start Compatibility Validation

Cuts encode state-variable coefficients $\beta_t^k \in \mathbb{R}^{n_x}$ whose dimension and semantic ordering are determined by the system that produced them. If the state dimension changes (different hydro count), or the state vector structure changes (different PAR orders, different production models), loaded cuts become dimensionally inconsistent with the current LP and produce silent numerical errors — wrong dual coefficients applied to wrong state variables. Therefore, warm-start loading verifies structural compatibility before accepting any cuts.

The following four checks run in order after `policy/metadata.json` and `policy/state_dictionary.json` are broadcast (SS7 step 1) and before any cut deserialization begins (SS7 step 3). All four must pass; the first failure aborts loading with `LoadError::PolicyIncompatible` (SS8.1).

| Check                          | Policy Source                                                                | System Source                                                   | Comparison                          | Failure                         |
| ------------------------------ | ---------------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------- | ------------------------------- |
| Hydro count                    | `PolicyMetadata.state_dimension` (derived from hydro count at training time) | `system.n_hydros()`                                             | Exact equality                      | `LoadError::PolicyIncompatible` |
| Maximum PAR order per hydro    | Stored per-hydro in policy metadata                                          | `max(system.par_models[h].order for each season)` per hydro `h` | Exact equality per hydro            | `LoadError::PolicyIncompatible` |
| Production method per hydro    | Stored per-hydro variant tag in policy metadata                              | `system.hydros[h].generation_model` variant tag                 | Exact equality per hydro            | `LoadError::PolicyIncompatible` |
| PAR model parameters per hydro | Stored per-hydro PAR coefficients and seasonal statistics in policy metadata | `system.par_models[h]` (all seasons)                            | Exact equality per hydro per season | `LoadError::PolicyIncompatible` |

**Safe modifications.** The following system properties may change between the training run that produced the policy and the current warm-start run without invalidating the loaded cuts:

- Exchange limits (line bounds)
- Load profiles (demand values)
- Inflow scenarios (opening realizations)
- Block durations
- Penalty values
- Thermal costs
- Demand

These properties affect the right-hand side or objective coefficients of the stage LP but do not alter the state variable dimension or the structural mapping between cut coefficients and state variables.

**Error behavior.** All four validation failures are hard errors — `LoadError::PolicyIncompatible` — and training cannot proceed with an incompatible policy. This check runs on every rank (since all ranks received the metadata broadcast in SS7 step 1) and runs before any cut deserialization begins (SS7 step 3). If validation fails, no cut files are read, avoiding wasted I/O on an incompatible policy.

## 8. Transition to In-Memory Model

After loading and broadcasting, each rank constructs its in-memory data model from the loaded data. The in-memory structures are defined in [Internal Structures](../data-model/internal-structures.md) and are not specified here.

**FPHA preprocessing:** For hydros with FPHA production model and source `"computed"`, the FPHA hyperplanes are fitted during the Initialization phase (after validation, before training). Fitting uses the hydro geometry (volume-area-level curves from `hydro_geometry.parquet`), topology data (productivity, tailrace, hydraulic losses, turbine efficiency from `hydros.json`), and fitting configuration (discretization points from `hydro_production_models.json`). See [Hydro Production Models](../math/hydro-production-models.md) for the mathematical formulation and [Input Hydro Extensions](../data-model/input-hydro-extensions.md) for the required input data.

> **Implementation note (v0.1.4):** The computed-source FPHA path is fully implemented in `cobre-sddp::hydro_models::prepare_hydro_models`. It runs on MPI rank 0 during Initialization, before the training loop or simulation pipeline begins. The fitting grid spans three dimensions: volume, turbined flow, and spillage (see `fpha_fitting.rs`). After fitting, computed hyperplanes are written to `output/hydro_models/fpha_hyperplanes.parquet` (see [Output Schemas §8.1](../data-model/output-schemas.md)) and held in memory for LP assembly. No rank broadcast of raw hyperplane rows is performed — ranks receive hydro model assignments through the `StudySetup` struct.

The key invariant is: **after the Initialization phase completes, all ranks hold identical copies of the validated case data** (except for per-rank scenario assignments, which are determined during Scenario Generation). See [CLI and Lifecycle](./cli-and-lifecycle.md) SS5.2 for the phase sequence.

### 8.1 `load_case` Public API

`load_case` is the primary entry point from `cobre-cli` (rank 0) into `cobre-io`. It performs the complete loading sequence (SS2.1--2.6), cross-reference validation (SS2.6), and returns a fully resolved `System` value ready for MPI broadcast via postcard serialization (SS6.1).

**Function signature:**

```rust
/// Load, validate, and resolve a Cobre case from the input directory.
///
/// This is the primary entry point from cobre-cli (rank 0) into cobre-io.
/// Returns the fully resolved System struct ready for MPI broadcast.
///
/// # Errors
/// Returns LoadError if any file cannot be read, parsed, or validated.
pub fn load_case(path: &Path) -> Result<System, LoadError>
```

**Parameters:**

- `path: &Path` -- the case directory containing `config.json` and all subdirectories (`system/`, `scenarios/`, `constraints/`, `policy/`). This is the root directory of the case, not a path to any individual file.

**Return type:**

- `System` -- an owned value (not `Arc<System>`). The caller (`cobre-cli` or `cobre-python`) owns the returned `System` and is responsible for broadcasting it to worker ranks via the postcard serialization protocol (SS6.1). After broadcast, each rank owns its own deserialized copy. Ownership transfer is the simplest pattern: no reference counting, no shared-memory coordination, no lifetime entanglement between the loading phase and the algorithm phase.

**Config loading:**

`load_case` does NOT accept a `Config` parameter. The `config.json` file is loaded as step 1 within `load_case` itself, per the loading sequence (SS2.1). The config governs conditional loading decisions (SS4) such as whether to require FPHA hyperplanes or scenario model files. Accepting a pre-loaded config would split the loading sequence across two call sites, creating a risk that conditional loading decisions are evaluated against a config that does not match the case directory contents.

**Responsibility boundary:**

`load_case` performs the following steps, in order:

1. Load and validate `config.json` (SS2.1).
2. Load root-level files: `stages.json`, `penalties.json`, `initial_conditions.json` (SS2.1).
3. Load system entity registries (SS2.2).
4. Load system extension data (SS2.3).
5. Load scenario data (SS2.4).
6. Load constraints and overrides (SS2.5).
7. Apply sparse time-series expansion (SS5).
8. Perform cross-reference validation (SS2.6).
9. Construct and return the `System` struct with all collections in canonical order.

`load_case` does NOT load policy files (SS2.7). Policy loading follows a different pattern -- all ranks load in parallel (SS7) -- and is a separate operation invoked after the `System` broadcast. See [CLI and Lifecycle](./cli-and-lifecycle.md) SS5.2 for the phase sequence that separates case loading from policy loading.

**`LoadError` enum:**

```rust
/// Errors that can occur during case loading.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// Filesystem read failure (file not found, permission denied, I/O error).
    #[error("I/O error reading {path}: {source}")]
    IoError {
        path: PathBuf,
        source: std::io::Error,
    },

    /// JSON or Parquet parsing failure (malformed content, encoding error).
    #[error("parse error in {path}: {message}")]
    ParseError {
        path: PathBuf,
        message: String,
    },

    /// Schema validation failure (missing required field, wrong type, value out of range).
    #[error("schema error in {path}, field {field}: {message}")]
    SchemaError {
        path: PathBuf,
        field: String,
        message: String,
    },

    /// Cross-reference validation failure (dangling entity ID, broken foreign key).
    #[error("cross-reference error: {source_entity} in {source_file} references \
             non-existent {target_entity} in {target_collection}")]
    CrossReferenceError {
        source_file: PathBuf,
        source_entity: String,
        target_collection: String,
        target_entity: String,
    },

    /// Semantic constraint violation (acyclic cascade, complete coverage, consistency).
    #[error("constraint violation: {description}")]
    ConstraintError {
        description: String,
    },

    /// Warm-start policy is structurally incompatible with the current system.
    /// See SS7.1 for the four compatibility checks.
    #[error("policy incompatible: {check} mismatch — policy has {policy_value}, \
             system has {system_value}")]
    PolicyIncompatible {
        check: String,
        policy_value: String,
        system_value: String,
    },
}
```

The variants are ordered by the phase in which they typically occur:

| Variant               | Typical Phase                 | Example                                                             |
| --------------------- | ----------------------------- | ------------------------------------------------------------------- |
| `IoError`             | File read (any step)          | `system/hydros.json` not found                                      |
| `ParseError`          | File parse (any step)         | Malformed JSON in `stages.json`                                     |
| `SchemaError`         | Schema validation             | `hydros.json` entry missing required field `bus_id`                 |
| `CrossReferenceError` | Cross-reference validation    | Hydro `bus_id = "BUS_99"` not found in bus registry                 |
| `ConstraintError`     | Semantic validation           | Hydro cascade contains a cycle; inflow model coverage is incomplete |
| `PolicyIncompatible`  | Warm-start validation (SS7.1) | Policy trained with 45 hydros, system now has 47                    |

All variants carry enough context for the caller to produce a diagnostic message without re-reading the input files.

**Cross-references:**

- [Internal Structures](../data-model/internal-structures.md) SS1 -- defines the `System` struct returned by `load_case`
- [CLI and Lifecycle](./cli-and-lifecycle.md) SS5.2 -- Validation phase where `load_case` is invoked on rank 0

## Cross-References

- [Input Directory Structure](../data-model/input-directory-structure.md) — File inventory, directory layout, `config.json` schema
- [Internal Structures](../data-model/internal-structures.md) — In-memory data model constructed from loaded data
- [CLI and Lifecycle](./cli-and-lifecycle.md) — Execution phases (Validation, Initialization) that orchestrate loading
- [Validation Architecture](./validation-architecture.md) — Multi-layer validation applied during and after loading
- [Design Principles](../overview/design-principles.md) — Declaration order invariance requiring canonicalization after loading
- [Configuration Reference](../configuration/configuration-reference.md) — Complete `config.json` schema loaded in step 1
- [Input System Entities](../data-model/input-system-entities.md) — Entity registries and default bounds
- [Input Scenarios](../data-model/input-scenarios.md) — Scenario model files and policy graph
- [Input Constraints](../data-model/input-constraints.md) — Generic constraints, penalty overrides, exchange factors
- [PAR Inflow Model](../math/par-inflow-model.md) — PAR(p) model parameters loaded from `inflow_seasonal_stats.parquet` and `inflow_ar_coefficients.parquet`
- [Binary Formats](../data-model/binary-formats.md) — Policy file format (FlatBuffers) for warm-start loading
- [Penalty System](../data-model/penalty-system.md) — Default penalty values and override cascade
