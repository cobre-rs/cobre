# Data Model

This section specifies every data structure that flows into, through, and out of the Cobre solver. On the input side, that means the JSON registries that define physical entities (hydro plants, thermal units, buses, transmission lines), the Parquet tables that carry stage-varying bounds and stochastic parameters, the penalty cascade that guarantees LP feasibility, and the `config.json` file that controls all solver behavior. On the output side, it covers the schemas for training convergence reports, simulation result tables, MPI-partitioned file manifests, and the binary persistence formats used for warm-starting and resuming interrupted runs.

The specs are organized around the data lifecycle. Input specs describe what the user provides and how the solver validates and loads it. Internal structure specs describe the in-memory model that the solver builds from those inputs. Output specs describe what the solver produces and how distributed results are assembled. Binary format specs describe the FlatBuffers schemas used for zero-copy serialization of policy data (cuts, states, basis vectors). Together, these 10 specs fully define the contract between the user, the I/O layer (`cobre-io`), the data model library (`cobre-core`), and the MPI communication layer (`ferrompi`).

Every file path, column name, JSON key, and Parquet schema documented here is normative. The solver rejects inputs that do not conform to these specs, and the output schemas are guaranteed stable within a major version. Where design choices exist (e.g., JSON vs. Parquet for a given file, sparse vs. dense override tables), the rationale is stated inline using the format selection criteria from the [Design Principles](./overview/design-principles.md) spec.

## Reading Order

The specs have extensive cross-references, so reading order matters. The following sequence builds concepts from the filesystem inward:

1. **[Input Directory Structure](./data-model/input-directory-structure.md)** -- Start here. Defines the case directory layout, the `config.json` schema, and the penalty summary that motivates the dedicated penalty spec.
2. **[Input System Entities](./data-model/input-system-entities.md)** -- The seven entity registries in `system/`: buses, lines, hydros, thermals, non-controllable sources, pumping stations, and energy contracts. Required context for reading any other input spec.
3. **[Input Hydro Extensions](./data-model/input-hydro-extensions.md)** -- Hydro-specific extension files: geometry curves, stage-varying production model configuration, and precomputed FPHA hyperplanes.
4. **[Input Scenarios](./data-model/input-scenarios.md)** -- The `stages.json` schema (seasons, policy graph, blocks, risk measures), inflow and load models, correlation profiles, and external scenario injection.
5. **[Input Constraints](./data-model/input-constraints.md)** -- Initial conditions, stage-varying entity bounds, exchange capacity factors, and generic linear constraints.
6. **[Penalty System](./data-model/penalty-system.md)** -- The three-tier penalty cascade (global defaults, entity overrides, stage overrides), the full penalty inventory, and resolution semantics that guarantee LP feasibility.
7. **[Internal Structures](./data-model/internal-structures.md)** -- The in-memory solver model built from input files: indexed entity tables, assembled LP matrices, and the runtime state that the forward and backward passes operate on.
8. **[Output Schemas](./data-model/output-schemas.md)** -- Training convergence tables, simulation result Parquet files, and per-iteration diagnostic exports.
9. **[Output Infrastructure](./data-model/output-infrastructure.md)** -- File manifests, MPI rank-partitioned output assembly, streaming vs. buffered write modes, and compression settings.
10. **[Binary Formats](./data-model/binary-formats.md)** -- FlatBuffers schemas for policy persistence: cut coefficients, visited states, inner approximation vertices, and solver basis snapshots.

## Spec Index

| Spec                                                                   | Description                                                                           | Math Reference                                                                                         |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [Input Directory Structure](./data-model/input-directory-structure.md) | Case directory layout, `config.json` schema, penalty summary                          | --                                                                                                     |
| [Input System Entities](./data-model/input-system-entities.md)         | Seven entity registries: buses, lines, hydros, thermals, NCS, pumping, contracts      | [System Elements](./math/system-elements.md)                                                           |
| [Input Hydro Extensions](./data-model/input-hydro-extensions.md)       | Hydro geometry curves, production model config, FPHA hyperplanes                      | [Hydro Production Models](./math/hydro-production-models.md)                                           |
| [Input Scenarios](./data-model/input-scenarios.md)                     | Stage definitions, policy graph, inflow/load models, correlations, external scenarios | [PAR Inflow Model](./math/par-inflow-model.md), [Risk Measures](./math/risk-measures.md)               |
| [Input Constraints](./data-model/input-constraints.md)                 | Initial conditions, stage-varying bounds, exchange factors, generic constraints       | [LP Formulation](./math/lp-formulation.md)                                                             |
| [Penalty System](./data-model/penalty-system.md)                       | Three-tier penalty cascade, full inventory, resolution semantics                      | [LP Formulation](./math/lp-formulation.md)                                                             |
| [Internal Structures](./data-model/internal-structures.md)             | In-memory solver model: indexed tables, LP matrices, runtime state                    | [LP Formulation](./math/lp-formulation.md), [SDDP Algorithm](./math/sddp-algorithm.md)                 |
| [Output Schemas](./data-model/output-schemas.md)                       | Training convergence, simulation results, diagnostic exports                          | [Upper Bound Evaluation](./math/upper-bound-evaluation.md), [Stopping Rules](./math/stopping-rules.md) |
| [Output Infrastructure](./data-model/output-infrastructure.md)         | File manifests, MPI-partitioned assembly, streaming writes, compression               | --                                                                                                     |
| [Binary Formats](./data-model/binary-formats.md)                       | FlatBuffers schemas for cuts, states, vertices, basis persistence                     | [Cut Management](./math/cut-management.md)                                                             |

## Conventions

All specs in this section follow the format selection criteria and declaration order invariance principle defined in [Design Principles](./overview/design-principles.md). In brief: **JSON** is used for human-editable structured objects with nested or optional fields (entity registries, configuration, correlation profiles); **Parquet** is used for typed columnar tabular data (stage-varying overrides, time series, scenario parameters); **FlatBuffers** is used for binary persistence where zero-copy deserialization matters (policy data). Declaration order within JSON arrays and Parquet rows never affects solver behavior -- entities are identified by their `id` field, not by position.

The [Notation Conventions](./overview/notation-conventions.md) spec establishes the index sets, symbols, and unit conventions used throughout. Column names in Parquet schemas and JSON keys are chosen to match the mathematical notation where practical (e.g., `volume` for \\(v*{i,t}\\), `turbined` for \\(u*{i,t}\\)).
