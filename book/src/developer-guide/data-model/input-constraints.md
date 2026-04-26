# Input Constraints, Initial Conditions, and Policy

## Purpose

This spec defines the initial system state (storage), time-varying operational bounds for all entities, the generic constraint system for custom linear constraints, and the policy directory used for warm-starting and resuming SDDP training.

For entity base schemas (bounds in `hydros.json`, `thermals.json`, `lines.json`), see [Input System Entities](input-system-entities.md). For contract bounds, see [Input System Entities §6](input-system-entities.md).

## 1. Initial Conditions (`initial_conditions.json`)

> **Format Rationale — initial_conditions.json**
>
> **Registry** — One-time snapshot of system state with cross-references to entities. JSON is natural for config-like data with nested structures.

Initial storage is the reservoir level at the start of the study. For hydros with `entry_stage_id`, this is the storage when they enter the system (not at stage 0). Filling hydros use a separate `filling_storage` array because their initial volume can be below `min_storage_hm3`.

```json
{
  "$schema": "https://cobre.dev/schemas/v2/initial_conditions.schema.json",
  "storage": [
    { "hydro_id": 0, "value_hm3": 15000.0 },
    { "hydro_id": 1, "value_hm3": 8500.0 }
  ],
  "filling_storage": [{ "hydro_id": 10, "value_hm3": 200.0 }]
}
```

> **Note**: `storage` is for operating hydros — values must be within `[min_storage_hm3, max_storage_hm3]`. `filling_storage` is for hydros with `filling` config — values can be below `min_storage_hm3` (the reservoir is not yet filled to dead volume).

> **Note**: When GNL support is implemented, this file will also accept a `gnl_pipeline` array. See [GNL Pipeline Initial Conditions](#gnl-pipeline-initial-conditions--deferred) below.

### Validation

| Rule                     | Description                                                                                           |
| ------------------------ | ----------------------------------------------------------------------------------------------------- |
| Operating hydro coverage | Every operating hydro (no `filling` config, or past `entry_stage_id`) must have an entry in `storage` |
| Storage bounds           | `storage` values must be within `[min_storage_hm3, max_storage_hm3]`                                  |
| Filling hydro coverage   | Every hydro with `filling` config must have an entry in `filling_storage`                             |
| Filling storage bounds   | `filling_storage` values must be within `[0, min_storage_hm3]`                                        |
| Mutual exclusion         | A hydro must appear in either `storage` or `filling_storage`, not both                                |
| Late-entry hydros        | For hydros entering later (without filling), this is their initial storage at entry                   |

### GNL Pipeline Initial Conditions — Deferred

> **Implementation Status**: GNL thermal dispatch anticipation is defined in [Input System Entities §4](input-system-entities.md) but implementation is deferred. When implemented, `gnl_pipeline` entries will specify the committed dispatch pipeline for GNL thermals at the start of the study.

When GNL support is implemented, `initial_conditions.json` will also include a `gnl_pipeline` array with the following logical schema:

| Field          | Type | Description                                            |
| -------------- | ---- | ------------------------------------------------------ |
| `thermal_id`   | i32  | ID of the GNL thermal (must have `gnl_config` defined) |
| `stage_offset` | i32  | Future stage offset (1 = stage 1, 2 = stage 2, etc.)   |
| `committed_mw` | f64  | Committed dispatch in MW for that future stage         |

### Pre-Study Inflow History

AR model lag initialization requires realized inflow values for pre-study stages. This data is defined in [Input Scenarios](input-scenarios.md) as part of the inflow history schema (`inflow_history.parquet`).

## 2. Time-Varying Entity Bounds (`constraints/`)

Time-varying bounds allow entities to have different operational limits per stage. If an entity is not present for a stage, base bounds from the entity registry file are used. Partial overrides are supported — only specify stages that differ from base. All bounds files use Parquet format for consistency with other tabular input data. See [Binary Formats §1](binary-formats.md) for the format framework.

### Thermal Bounds (`constraints/thermal_bounds.parquet`) — Optional

| Column              | Type | Description                          |
| ------------------- | ---- | ------------------------------------ |
| `thermal_id`        | i32  | Thermal unit identifier              |
| `stage_id`          | i32  | Stage index                          |
| `min_generation_mw` | f64  | Minimum generation (null = use base) |
| `max_generation_mw` | f64  | Maximum generation (null = use base) |

### Hydro Bounds (`constraints/hydro_bounds.parquet`) — Optional

Useful for maintenance outages, seasonal restrictions, environmental constraints, and dead-volume filling periods.

| Column                 | Type | Description                                                                                       |
| ---------------------- | ---- | ------------------------------------------------------------------------------------------------- |
| `hydro_id`             | i32  | Hydro plant identifier                                                                            |
| `stage_id`             | i32  | Stage index                                                                                       |
| `min_turbined_m3s`     | f64  | Minimum turbined flow (null = use base)                                                           |
| `max_turbined_m3s`     | f64  | Maximum turbined flow (null = use base)                                                           |
| `min_storage_hm3`      | f64  | Minimum storage (null = use base)                                                                 |
| `max_storage_hm3`      | f64  | Maximum storage (null = use base)                                                                 |
| `min_outflow_m3s`      | f64  | Minimum outflow (null = use base)                                                                 |
| `max_outflow_m3s`      | f64  | Maximum outflow (null = use base)                                                                 |
| `min_generation_mw`    | f64  | Minimum generation (null = use base)                                                              |
| `max_generation_mw`    | f64  | Maximum generation (null = use base)                                                              |
| `max_diversion_m3s`    | f64  | Maximum diversion flow (null = use base). Only for hydros with diversion.                         |
| `filling_inflow_m3s`   | f64  | Filling inflow override for this stage (null = use entity default from `hydros.json`). See below. |
| `water_withdrawal_m3s` | f64  | Water withdrawal (positive = remove, negative = add)                                              |

**Filling inflow**: Minimum inflow retained for reservoir filling during dead-volume filling periods (`[start_stage_id, entry_stage_id)`). This water is removed from the cascade before the water balance — it goes directly to storage. Only valid for hydros with `filling` config, during filling stages. The effective value follows the entity default → stage override cascade: the entity-level `filling_inflow_m3s` in `hydros.json` (see [Input System Entities §3](input-system-entities.md)) provides the default for all filling stages; a non-null value here overrides it for this specific stage. If neither is specified, filling inflow is 0.0 (passive filling). See [Penalty System §7](penalty-system.md) for how filling interacts with outflow requirements and the terminal filling constraint.

> **Filling inflow sufficiency warning**: At input validation time, the system should compute the cumulative volume from the effective `filling_inflow_m3s` across all filling stages (entity default with per-stage overrides, accounting for stage durations) and compare against the required volume (`min_storage_hm3 - initial_filling_storage`). If the scheduled inflows are provably insufficient (even ignoring evaporation and assuming zero spillage), a **warning** should be emitted. This is a warning, not an error, because natural stochastic inflows beyond the scheduled minimum can supplement the filling process.

**Water withdrawal (retirada de água)**: Water removed from the reservoir for consumption, irrigation, or industrial use. Positive values represent water leaving the system; negative values represent external additions (transpositions).

> **Note on evaporation**: The evaporation model is defined by the 12 monthly coefficients in `hydros.json` (see [Input System Entities §3](input-system-entities.md)) combined with the volume-area relationship from `hydro_geometry` (see [Input Hydro Extensions §1](input-hydro-extensions.md)). There is no per-stage evaporation override — the monthly coefficients already capture seasonal variation.

### Line Bounds (`constraints/line_bounds.parquet`) — Optional

| Column       | Type | Description                             |
| ------------ | ---- | --------------------------------------- |
| `line_id`    | i32  | Transmission line identifier            |
| `stage_id`   | i32  | Stage index                             |
| `direct_mw`  | f64  | Direct flow capacity (null = use base)  |
| `reverse_mw` | f64  | Reverse flow capacity (null = use base) |

### Pumping Station Bounds (`constraints/pumping_bounds.parquet`) — Optional {#pumping-bounds}

> **Implementation status**: Schema defined but constraint application is a NO-OP stub in the current code. The file is parsed and validated, but bounds are not applied to the LP.

| Column       | Type | Description                           |
| ------------ | ---- | ------------------------------------- |
| `station_id` | i32  | Pumping station identifier            |
| `stage_id`   | i32  | Stage index                           |
| `min_m3s`    | f64  | Minimum pumped flow (null = use base) |
| `max_m3s`    | f64  | Maximum pumped flow (null = use base) |

### Contract Bounds (`constraints/contract_bounds.parquet`) — Optional {#contract-bounds}

> **Implementation status**: Schema defined but constraint application is a NO-OP stub in the current code. The file is parsed and validated, but bounds are not applied to the LP.

| Column          | Type | Description                               |
| --------------- | ---- | ----------------------------------------- |
| `contract_id`   | i32  | Energy contract identifier                |
| `stage_id`      | i32  | Stage index                               |
| `min_mw`        | f64  | Minimum contract usage (null = use base)  |
| `max_mw`        | f64  | Maximum contract usage (null = use base)  |
| `price_per_mwh` | f64  | Contract price override (null = use base) |

### Non-Controllable Sources

Non-controllable sources (wind, solar) represent available generation that cannot be dispatched — only curtailed. Since the system cannot control their output upward, there are no meaningful bounds to override per stage. Their availability is determined by the stochastic scenario model, not by operational bounds. Stage-varying bounds are therefore not applicable for this entity type.

### Exchange Factors (`constraints/exchange_factors.json`) — Optional

> **Format Rationale — exchange_factors.json**
>
> **Default-with-overrides** — Small number of exchange factor definitions. JSON for readability.

Exchange (transmission) capacity may vary by block due to thermal limits, contractual constraints, or operational patterns. Block factors are **multipliers** applied to the stage-level line capacity. Factors greater than 1.0 are intentional — they allow block-level capacity to exceed the stage-level base value, reflecting periods of higher thermal or contractual allowances.

This file lives under `constraints/` rather than `scenarios/` because exchange factors affect line capacity bounds in the LP (a constraint concern), not the stochastic scenario pipeline.

If missing, all block factors default to 1.0.

```json
{
  "$schema": "https://cobre.dev/schemas/v2/exchange_factors.schema.json",
  "exchange_factors": [
    {
      "line_id": 0,
      "stage_id": 0,
      "block_factors": [
        { "block_id": 0, "direct_factor": 0.9, "reverse_factor": 0.9 },
        { "block_id": 1, "direct_factor": 1.0, "reverse_factor": 1.0 },
        { "block_id": 2, "direct_factor": 1.1, "reverse_factor": 1.1 }
      ]
    }
  ]
}
```

For example, if a line has 5000 MW direct capacity and block factors are [0.90, 1.00, 1.10], the blocks get [4500, 5000, 5500] MW.

## 3. Generic Constraints (`constraints/generic_constraints.json`)

> **Format Rationale — generic_constraints.json**
>
> **Complex nested object** — Constraint definitions with variable references, coefficients, and conditional logic. JSON handles polymorphic structures well.

> **Order Invariance**: The order of constraints does NOT affect results. After loading, constraints are sorted by `id`. See [Design Principles §3](../overview/design-principles.md).

Users can express custom linear constraints combining multiple LP variables:

| Constraint Type            | Example                                           |
| -------------------------- | ------------------------------------------------- |
| Regional generation limits | Minimum/maximum total hydro generation per region |
| Energy contracts           | Sum of generation from specific plants            |
| Irrigation agreements      | Sum of outflows                                   |
| Environmental corridors    | Combined outflow requirements                     |
| Fuel availability          | Sum of thermal generation                         |

### Example

```json
{
  "$schema": "https://cobre.dev/schemas/v2/generic_constraints.schema.json",
  "constraints": [
    {
      "id": 0,
      "name": "min_southeast_hydro",
      "description": "Minimum hydro generation in Southeast region",
      "expression": "hydro_generation(0) + hydro_generation(1) + hydro_generation(2)",
      "sense": ">=",
      "slack": { "enabled": true, "penalty": 5000.0 }
    },
    {
      "id": 1,
      "name": "itaipu_contract",
      "description": "Itaipu energy contract - must deliver at least base amount",
      "expression": "hydro_generation(50)",
      "sense": ">=",
      "slack": { "enabled": true, "penalty": 8000.0 }
    },
    {
      "id": 2,
      "name": "environmental_flow",
      "description": "Combined outflow requirement for river stretch",
      "expression": "hydro_outflow(10) + hydro_outflow(11)",
      "sense": ">=",
      "slack": { "enabled": true, "penalty": 3000.0 }
    },
    {
      "id": 3,
      "name": "gas_availability",
      "description": "Total gas thermal generation limited by pipeline",
      "expression": "thermal_generation(5) + thermal_generation(6) + thermal_generation(7)",
      "sense": "<=",
      "slack": { "enabled": false }
    }
  ]
}
```

### Constraint Fields

| Field           | Type   | Required   | Description                                    |
| --------------- | ------ | ---------- | ---------------------------------------------- |
| `id`            | i32    | Yes        | Unique constraint identifier                   |
| `name`          | string | Yes        | Short name for reports                         |
| `description`   | string | No         | Human-readable description                     |
| `expression`    | string | Yes        | Linear expression (parsed using grammar below) |
| `sense`         | string | Yes        | `">="`, `"<="`, or `"=="`                      |
| `slack.enabled` | bool   | Yes        | Whether to add slack variable for feasibility  |
| `slack.penalty` | f64    | If enabled | Penalty per unit of violation                  |

### Variable Reference Syntax

Variables use function-like syntax: `variable_type(entity_id)`. For block-specific variables, use `variable_type(entity_id, block_id)` or omit `block_id` to sum over all blocks.

| Variable Name        | Syntax                             | Units | Notes                                              |
| -------------------- | ---------------------------------- | ----- | -------------------------------------------------- |
| `hydro_storage`      | `hydro_storage(id)`                | hm³   |                                                    |
| `hydro_turbined`     | `hydro_turbined(id [, block])`     | m³/s  |                                                    |
| `hydro_spillage`     | `hydro_spillage(id [, block])`     | m³/s  |                                                    |
| `hydro_diversion`    | `hydro_diversion(id [, block])`    | m³/s  | Only for hydros with diversion                     |
| `hydro_outflow`      | `hydro_outflow(id [, block])`      | m³/s  | Currently alias for turbined + spillage (see note) |
| `hydro_generation`   | `hydro_generation(id [, block])`   | MW    |                                                    |
| `hydro_evaporation`  | `hydro_evaporation(id)`            | m³/s  |                                                    |
| `hydro_withdrawal`   | `hydro_withdrawal(id)`             | m³/s  |                                                    |
| `thermal_generation` | `thermal_generation(id [, block])` | MW    |                                                    |
| `line_direct`        | `line_direct(id [, block])`        | MW    |                                                    |
| `line_reverse`       | `line_reverse(id [, block])`       | MW    |                                                    |
| `bus_deficit`        | `bus_deficit(id [, block])`        | MW    |                                                    |
| `bus_excess`         | `bus_excess(id [, block])`         | MW    |                                                    |
| `pumping_flow`       | `pumping_flow(id [, block])`       | m³/s  |                                                    |
| `pumping_power`      | `pumping_power(id [, block])`      | MW    |                                                    |
| `contract_import`    | `contract_import(id [, block])`    | MW    |                                                    |
| `contract_export`    | `contract_export(id [, block])`    | MW    |                                                    |
| `line_exchange`      | `line_exchange(id [, block])`      | MW    | Net exchange on transmission line                  |
| `nc_generation`      | `nc_generation(id [, block])`      | MW    | Non-controllable generation (wind/solar)           |
| `nc_curtailment`     | `nc_curtailment(id [, block])`     | MW    | Non-controllable generation curtailment            |

> **Note on `hydro_outflow`**: Currently defined as an alias for `turbined + spillage`. Future modeling enhancements (CEPEL advanced downstream flow formulations with participation factors) may require `hydro_outflow` to become an independent variable with a more detailed definition. See [Input System Entities §3](input-system-entities.md) future extensions for details.

### Expression Grammar

```ebnf
expression    ::= term (('+' | '-') term)*
term          ::= coefficient? variable | number
coefficient   ::= number '*'
variable      ::= var_name '(' entity_id (',' block_id)? ')'
entity_id     ::= integer
block_id      ::= integer
number        ::= float | integer
```

**Examples**: `hydro_generation(10) + hydro_generation(11)` · `2.5 * thermal_generation(5) - hydro_generation(3)` · `hydro_turbined(5, 0) + hydro_turbined(5, 1)`

### Constraint Bounds (`constraints/generic_constraint_bounds.parquet`)

Bounds define the RHS value for each generic constraint. Bounds can vary by stage and optionally by block:

| Column          | Type | Description                                |
| --------------- | ---- | ------------------------------------------ |
| `constraint_id` | i32  | References constraint definition           |
| `stage_id`      | i32  | Stage index                                |
| `block_id`      | i32  | Block index (null = applies to all blocks) |
| `bound`         | f64  | RHS value for the constraint               |

If a constraint has no bound entry for a given stage, the constraint is not active for that stage.

### LP Integration

For how generic constraints and their slack variables enter the LP objective function, see [LP Formulation](../math/lp-formulation.md). Slack variables are only created if `slack.enabled = true` in the constraint definition.

### Validation Rules

1. All entity IDs referenced in expressions must exist in the system
2. Block IDs (if specified) must be valid for the stage
3. Expressions must parse correctly according to the grammar
4. Constraint IDs must be unique and contiguous (0, 1, 2, …)
5. If `slack.enabled = true`, `slack.penalty` must be provided and positive
6. `constraint_id` values in the bounds data must reference existing constraint definitions

## 4. Policy Directory (`policy/`)

> **Scope note**: The policy directory is documented here because it is user-facing: users provide policy files for warm-start and resume operations. For the internal binary schemas of cuts, states, vertices, and basis files, see [Binary Formats](binary-formats.md).

> **Format Rationale — policy/**
>
> **Policy / binary data** — Pre-computed cuts for warm-start. FlatBuffers for zero-copy deserialization during hot-path loading. See [Binary Formats](binary-formats.md) for detailed FlatBuffers schemas.

Cobre uses a single `policy/` directory that serves both as input (loading existing cuts/states) and output (writing updated policy data).

### Directory Structure

```
policy/
├── metadata.json               # Algorithm state, RNG, bounds
├── state_dictionary.json       # State variable mapping
├── cuts/                       # Outer approximation (standard SDDP cuts)
│   ├── stage_000.bin
│   └── ...
├── states/                     # Visited states for cut selection
│   └── ...
├── vertices/                   # Inner approximation (upper bounds, if enabled)
│   └── ...
└── basis/                      # Solver basis for exact reproducibility (optional)
    └── ...
```

### Policy Modes

| Mode         | Reads From                                  | Behavior                                                       |
| ------------ | ------------------------------------------- | -------------------------------------------------------------- |
| `fresh`      | Nothing                                     | Start from scratch. Existing files ignored (not deleted).      |
| `warm_start` | `cuts/`, `states/`, `state_dictionary.json` | Load existing cuts/states, reset iteration count and RNG seed. |
| `resume`     | Everything including `metadata.json`        | Load full algorithm state, continue exactly where interrupted. |

### State Dictionary (`state_dictionary.json`)

Defines the mapping between coefficient indices and actual state variables. Follows canonical ordering (see [Design Principles §3](../overview/design-principles.md)): state variables ordered by entity type, then by entity ID.

```json
{
  "$schema": "https://cobre.dev/schemas/v2/state_dictionary.schema.json",
  "version": "2.0.0",
  "state_dimension": 320,
  "state_variables": [
    {
      "index": 0,
      "type": "storage",
      "entity_type": "hydro",
      "entity_id": 0,
      "name": "FURNAS"
    },
    {
      "index": 1,
      "type": "storage",
      "entity_type": "hydro",
      "entity_id": 1,
      "name": "MARIMBONDO"
    },
    {
      "index": 160,
      "type": "inflow_lag_1",
      "entity_type": "hydro",
      "entity_id": 0,
      "name": "FURNAS"
    }
  ],
  "checksum": "sha256:abc123..."
}
```

| Field         | Type   | Description                                                                                                                                                                                                   |
| ------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `index`       | i32    | Column index in cuts/states (`coefficient_N`, `component_N`)                                                                                                                                                  |
| `type`        | string | Variable type: `"storage"`, `"inflow_lag_1"`, `"inflow_lag_2"`, … (expands the boolean flags from `state_variables` in [Input Scenarios §1.6](input-scenarios.md) into individual entries per entity and lag) |
| `entity_type` | string | `"hydro"` (currently; future extensions may add `"thermal"` for GNL pipeline state)                                                                                                                           |
| `entity_id`   | i32    | Entity ID (matches entity registries)                                                                                                                                                                         |
| `name`        | string | Human-readable entity name (for debugging)                                                                                                                                                                    |

### Validation

**Resume** (`policy.mode = "resume"`):

1. Version compatibility with current Cobre version
2. Config hash match (or explicit override flag)
3. System hash match (entities must be identical)
4. State dictionary checksum match
5. All partitioned files exist for all stages

**Warm-start** (`policy.mode = "warm_start"`):

1. `state_dictionary.json` exists
2. State dimension matches current system
3. All entity IDs in dictionary exist in current system
4. At least one stage has cuts

### Reproducibility Guarantees

| Scenario                           | Reproducibility       | Notes                                                   |
| ---------------------------------- | --------------------- | ------------------------------------------------------- |
| Straight run (no prior policy)     | Bit-for-bit identical | Same seed → same results                                |
| Resume with basis                  | Bit-for-bit identical | Solver basis restored → same pivots                     |
| Resume without basis               | Equivalent optimum    | Same optimum, possibly different duals → different cuts |
| Warm-start                         | Different run         | Fresh RNG, may converge differently                     |
| Resume with modified config/system | Not supported         | Hash mismatch → error                                   |

For detailed binary schemas of cuts, states, vertices, and basis files, see [Binary Formats](binary-formats.md).

## Cross-References

- [Input System Entities](input-system-entities.md) — Base entity schemas referenced by bounds and constraints; includes pumping stations (§5), energy contracts (§6)
- [Input Hydro Extensions](input-hydro-extensions.md) — Hydro geometry, production models, FPHA hyperplanes
- [Input Scenarios](input-scenarios.md) — Stage definitions, stochastic models, state variables (§1.6), inflow history
- [Input Directory Structure](input-directory-structure.md) — Overall case directory layout
- [Binary Formats](binary-formats.md) — Detailed FlatBuffers schemas for policy binary files
- [LP Formulation](../math/lp-formulation.md) — How constraints enter the LP
- [Cut Management](../math/cut-management.md) — Cut storage and selection algorithms
- [SDDP Algorithm](../math/sddp-algorithm.md) — Forward/backward pass and convergence
- [Penalty System](penalty-system.md) — Penalty hierarchy for slack variables
- [Design Principles §3](../overview/design-principles.md) — Order invariance and canonical ordering
