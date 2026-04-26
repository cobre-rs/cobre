# Input Directory Structure

## Purpose

This spec defines the layout of a Cobre input case directory and the schema of the central configuration file `config.json`. It serves as the entry point for understanding how input data is organized and what options control solver behavior.

## 1. Directory Tree

<!-- TODO: diagram -- directory-structure -->

```
case/
├── config.json                                # Execution configuration (§2)
├── initial_conditions.json                    # Initial storage (operating + filling hydros)
├── stages.json                                # Stage/season definitions, policy graph, blocks
├── penalties.json                             # Global penalty defaults
│
├── system/                                    # Entity registries and extensions
│   ├── buses.json                             # Bus registry with deficit segments
│   ├── lines.json                             # Transmission line registry
│   ├── hydros.json                            # Hydro plant registry (includes tailrace,
│   │                                          #   losses, efficiency, evaporation)
│   ├── thermals.json                          # Thermal plant registry
│   ├── non_controllable_sources.json          # Wind/solar sources (optional)
│   ├── pumping_stations.json                  # Pumping station registry (optional)
│   ├── energy_contracts.json                  # Energy contract definitions (optional)
│   ├── hydro_geometry.parquet                 # Volume-area-level curves (optional)
│   ├── hydro_production_models.json           # Stage-varying production model config (optional)
│   └── fpha_hyperplanes.parquet               # Precomputed FPHA planes (optional)
│
├── scenarios/                                 # Stochastic models and time series
│   ├── inflow_history.parquet                 # Historical inflow observations (optional)
│   ├── inflow_seasonal_stats.parquet          # Seasonal mean/std per hydro/stage (optional)
│   ├── inflow_ar_coefficients.parquet         # PAR(p) AR coefficients per hydro/stage/lag (optional)
│   ├── external_inflow_scenarios.parquet      # Pre-computed inflow scenario values (optional)
│   ├── external_load_scenarios.parquet        # Pre-computed load scenario values (optional)
│   ├── external_ncs_scenarios.parquet         # Pre-computed NCS scenario values (optional)
│   ├── load_seasonal_stats.parquet            # Load mean/std per bus/stage (optional)
│   ├── load_factors.json                      # Block-level load scaling factors (optional)
│   ├── non_controllable_stats.parquet         # NCS stochastic availability factors (optional)
│   ├── non_controllable_factors.json          # NCS block-level scaling factors (optional)
│   ├── correlation.json                       # Spatial correlation profiles + schedule (optional)
│   └── noise_openings.parquet                 # User-supplied pre-correlated noise openings (Optional, ADR-008)
│
├── constraints/                               # Time-varying bounds and generic constraints
│   ├── thermal_bounds.parquet                 # Stage-varying thermal limits (optional)
│   ├── hydro_bounds.parquet                   # Stage-varying hydro limits (optional)
│   ├── line_bounds.parquet                    # Stage-varying line limits (optional)
│   ├── pumping_bounds.parquet                 # Stage-varying pumping limits (optional)
│   ├── contract_bounds.parquet                # Stage-varying contract limits (optional)
│   ├── ncs_bounds.parquet                     # Stage-varying NCS available generation bounds (optional)
│   ├── exchange_factors.json                  # Block-level exchange capacity factors (optional)
│   ├── generic_constraints.json               # Custom linear constraints (optional)
│   ├── generic_constraint_bounds.parquet      # RHS bounds for generic constraints (optional)
│   ├── penalty_overrides_bus.parquet          # Stage-varying bus penalties (optional)
│   ├── penalty_overrides_line.parquet         # Stage-varying line penalties (optional)
│   ├── penalty_overrides_hydro.parquet        # Stage-varying hydro penalties (optional)
│   └── penalty_overrides_ncs.parquet          # Stage-varying NCS penalties (optional)
│
└── policy/                                    # Warm-start / resume data (optional)
    ├── metadata.json                          # Algorithm state, RNG, bounds
    ├── state_dictionary.json                  # State variable mapping
    ├── cuts/                                  # Outer approximation (SDDP cuts)
    ├── states/                                # Visited states for cut selection
    ├── vertices/                              # Inner approximation (if enabled)
    └── basis/                                 # Solver basis for warm-start (optional)
```

The input case directory is organized into four top-level groups plus root-level configuration files:

| Directory      | Purpose                                                      | Format                      |
| -------------- | ------------------------------------------------------------ | --------------------------- |
| Root           | Configuration, penalties, stages, initial conditions         | JSON                        |
| `system/`      | Entity registries and extension data (all 7 element types)   | JSON + Parquet              |
| `scenarios/`   | Stochastic models, history, block factors, correlation       | JSON + Parquet              |
| `constraints/` | Stage-varying bounds, penalty overrides, generic constraints | JSON + Parquet              |
| `policy/`      | Warm-start and resume data (cuts, states, basis)             | JSON + FlatBuffers (binary) |

> **Format Rationale — Directory Layout**
>
> The separation follows the [Design Principles](../overview/design-principles.md) format selection criteria: **JSON** for human-editable structured objects with nested/optional fields (registries, configuration, correlation profiles); **Parquet** for typed columnar tabular data (entity-level lookup tables, stage-varying overrides, time series, scenario parameters). Root-level files are read once at startup; `system/` files define the physical model; `scenarios/` files define stochastic processes; `constraints/` files provide stage-varying overrides and block-level capacity factors; `policy/` stores algorithm state for warm-starting or resuming. Binary policy files use FlatBuffers for zero-copy deserialization — see [Binary Formats](binary-formats.md).
>
> **Why Parquet for all tabular data**: Parquet provides self-describing schemas with typed columns, columnar compression, efficient filtering, and excellent tooling in Python/R/Arrow. Even for small files (~100s of rows), the consistency benefit outweighs the minor overhead — users need Parquet tooling for the larger files regardless, and a future frontend will handle visual editing. See [Binary Formats §1](binary-formats.md) for the complete format decision framework.

### Root-Level Files

| File                      | Required | Description                                                                                                                                                                                                                                       | Spec Reference                               |
| ------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| `config.json`             | Yes      | Central execution configuration: modeling options, training settings, scenario source, simulation settings, export controls. Controls all solver behavior.                                                                                        | §2 below                                     |
| `penalties.json`          | Yes      | Global default penalty values for the three-tier cascade: deficit segment costs, regularization costs, constraint violation penalties. Entity and stage overrides layer on top.                                                                   | [Penalty System](penalty-system.md)          |
| `stages.json`             | Yes      | Season definitions with calendar mapping, policy graph (transitions, horizon type, annual discount rate), stage definitions with per-stage block structure, block mode, state variables, risk measure (CVaR), sampling method, and num_scenarios. | [Input Scenarios §1](input-scenarios.md)     |
| `initial_conditions.json` | Yes      | Initial system state: operating hydro storage levels (`storage` array) and filling hydro storage levels (`filling_storage` array, can be below dead volume). GNL pipeline state deferred — see [Input Constraints §1](input-constraints.md).      | [Input Constraints §1](input-constraints.md) |

## 2. Configuration (`config.json`)

> **Decision [DEC-018](../overview/decision-log.md#dec-018) (active):** MPI/HPC parameters removed from config.json — all are auto-detected implementation details or contradicted by approved architecture.

> **Note**: Solver selection (HiGHS, CPLEX, Gurobi) is determined at compile time via Cargo features due to licensing constraints. LP solver retry parameters are configurable via the `training.solver` section -- see section 2.5 below.

> **Format Rationale — config.json**
>
> JSON was chosen for the central configuration because it is human-readable, easily editable, and small in size. Configuration is a **nested object** with logical groupings (training, simulation, exports) that map naturally to JSON's hierarchical structure. All sections have solid code defaults — the minimal valid config is very small.

**Minimal example** — only fields with no reasonable default:

```json
{
  "$schema": "https://cobre.dev/schemas/v2/config.schema.json",
  "version": "2.0.0",

  "training": {
    "tree_seed": 42,
    "forward_passes": 192,
    "stopping_rules": [{ "type": "iteration_limit", "limit": 50 }]
  }
}
```

All omitted sections (`modeling`, `upper_bound_evaluation`, `policy`, `simulation`, `exports`) use code defaults. See [Configuration Reference](../configuration/configuration-reference.md) for all defaults.

**Full example** — all sections with explicit overrides:

```json
{
  "$schema": "https://cobre.dev/schemas/v2/config.schema.json",
  "version": "2.0.0",

  "modeling": {
    "inflow_non_negativity": {
      "method": "penalty",
      "penalty_cost": 1000.0
    }
  },

  "training": {
    "enabled": true,
    "tree_seed": 42,
    "forward_passes": 192,
    "stopping_rules": [
      { "type": "iteration_limit", "limit": 50 },
      { "type": "bound_stalling", "iterations": 10, "tolerance": 0.0001 }
    ],
    "stopping_mode": "any",
    "cut_formulation": "single",
    "forward_pass": {
      "type": "default"
    },
    "cut_selection": {
      "enabled": true,
      "method": "domination",
      "threshold": 0
    },
    "solver": {
      "retry_max_attempts": 5,
      "retry_time_budget_seconds": 30.0
    },
    "scenario_source": {
      "seed": 42,
      "inflow": { "scheme": "in_sample" },
      "load": { "scheme": "in_sample" },
      "ncs": { "scheme": "in_sample" }
    }
  },

  "upper_bound_evaluation": {
    "enabled": true,
    "initial_iteration": 10,
    "interval_iterations": 5
  },

  "policy": {
    "path": "./policy",
    "mode": "fresh",
    "checkpointing": {
      "enabled": true,
      "initial_iteration": 10,
      "interval_iterations": 10,
      "store_basis": true,
      "compress": true
    },
    "validate_compatibility": true
  },

  "simulation": {
    "enabled": true,
    "num_scenarios": 2000,
    "policy_type": "outer",
    "output_path": "./simulation",
    "output_mode": "streaming",
    "io_channel_capacity": 64
  },

  "estimation": {
    "max_order": 6,
    "order_selection": "pacf",
    "min_observations_per_season": 30
  },

  "exports": {
    "training": true,
    "cuts": true,
    "states": false,
    "vertices": true,
    "simulation": true,
    "forward_detail": false,
    "backward_detail": false,
    "compression": "zstd"
  }
}
```

The subsections below describe each configuration group. For the complete field-by-field reference with defaults and validation rules, see [Configuration Reference](../configuration/configuration-reference.md).

### 2.1 Modeling Configuration

| Field                   | Type   | Default                                           | Description                                                                                                         |
| ----------------------- | ------ | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `inflow_non_negativity` | object | `{ "method": "penalty", "penalty_cost": 1000.0 }` | Strategy for ensuring non-negative generated inflows. See [Inflow Non-Negativity](../math/inflow-nonnegativity.md). |

> **Note**: Block mode (`parallel` or `chronological`) is configured **per stage** in `stages.json`, not globally. See [Input Scenarios §1.5](input-scenarios.md). Horizon mode (`finite_horizon` or `cyclic`) is configured in the `policy_graph` section of `stages.json`. See [Input Scenarios §1.2](input-scenarios.md).

### 2.2 Training Configuration

Key training parameters include the random seed, number of forward passes, stopping rules, cut formulation, and forward/backward pass modes. The `stopping_mode` controls how multiple rules combine (`"any"` = OR, `"all"` = AND).

> **Validation**: At least one `iteration_limit` rule must be present in the `stopping_rules` array.

For the complete stopping rule types and their parameters, see [Configuration Reference](../configuration/configuration-reference.md).

### 2.3 Policy Directory Configuration

| Field                    | Type   | Default      | Description                                                  |
| ------------------------ | ------ | ------------ | ------------------------------------------------------------ |
| `path`                   | string | `"./policy"` | Directory for policy data (cuts, states, vertices, basis)    |
| `mode`                   | string | `"fresh"`    | How to initialize: `"fresh"`, `"warm_start"`, or `"resume"`  |
| `validate_compatibility` | bool   | true         | Verify state dimension and entity compatibility when loading |

**Policy Modes:**

| Mode         | Behavior                                                                                      |
| ------------ | --------------------------------------------------------------------------------------------- |
| `fresh`      | Start from scratch. Ignore any existing data in `policy/`.                                    |
| `warm_start` | Load existing cuts/states to initialize, but reset iteration count and use fresh RNG seed.    |
| `resume`     | Load full algorithm state including RNG, iteration count. Continue exactly where interrupted. |

### 2.4 Simulation Configuration

| Field                 | Type   | Default   | Description                                                                                                                   |
| --------------------- | ------ | --------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `enabled`             | bool   | false     | Enable post-training simulation                                                                                               |
| `num_scenarios`       | i32    | 2000      | Number of simulation scenarios                                                                                                |
| `policy_type`         | string | `"outer"` | `"outer"` (cuts) or `"inner"` (vertices)                                                                                      |
| `scenario_source`     | object | (inherit) | Per-class scenario source (same format as `training.scenario_source`). When absent, falls back to `training.scenario_source`. |
| `io_channel_capacity` | u32    | 64        | Bounded channel capacity between simulation threads and the I/O writer thread                                                 |

### 2.5 Training Solver Configuration (`training.solver`)

LP solver retry parameters for the training phase. These control how aggressively the solver retries failed LP solves before propagating a hard error.

| Field                       | Type | Default | Description                                                      |
| --------------------------- | ---- | ------- | ---------------------------------------------------------------- |
| `retry_max_attempts`        | u32  | 5       | Maximum solver retry attempts before propagating a hard error    |
| `retry_time_budget_seconds` | f64  | 30.0    | Total time budget in seconds across all retry attempts per solve |

> **Note**: The `training.enabled` field (boolean, default `true`) controls whether the training phase runs. When `false`, training is skipped and the solver proceeds directly to simulation using a previously-computed policy.

### 2.6 Estimation Configuration (`estimation`)

Controls automatic parameter estimation when historical inflow data is provided without explicit model statistics or AR coefficients. All fields are optional and fall back to defaults.

| Field                         | Type          | Default  | Description                                                                                                                                                                  |
| ----------------------------- | ------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `max_order`                   | u32           | 6        | Maximum lag order considered during autoregressive model fitting                                                                                                             |
| `order_selection`             | string        | `"pacf"` | Order selection criterion: `"pacf"` (periodic partial autocorrelation). The value `"fixed"` is a deprecated alias for `"pacf"` and should not be used in new configurations. |
| `min_observations_per_season` | u32           | 30       | Minimum observations required per (entity, season) group to proceed with estimation                                                                                          |
| `max_coefficient_magnitude`   | f64 or `null` | `null`   | Maximum allowed absolute AR coefficient; pairs exceeding this are reduced to order 0                                                                                         |

## 3. Penalties and Costs (Summary)

The LP must always be feasible. Penalty costs on slack variables ensure this by allowing constraint violations at a high cost. Cobre uses a **three-tier cascade** for penalty resolution:

1. **Global defaults** in `penalties.json` (required)
2. **Entity overrides** inline in entity JSON files (optional)
3. **Stage overrides** in per-entity-type Parquet files (optional, sparse)

Penalties are divided into three categories:

| Category                           | Examples                                                                                     | Purpose                                                     | Typical Range      |
| ---------------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------ |
| **Recourse slacks**                | `deficit_*`, `excess_cost`                                                                   | Ensure LP feasibility when demand cannot be met             | 100–10,000 \$/unit |
| **Constraint violation penalties** | `*_violation_*_cost`, `generic_violation_cost`                                               | Allow soft constraint violations at a cost (policy shaping) | 50–5,000 \$/unit   |
| **Regularization costs**           | `spillage_cost`, `diversion_cost`, `exchange_cost`, `fpha_turbined_cost`, `curtailment_cost` | Discourage undesirable but feasible operations              | 0.001–10 \$/unit   |

For the complete penalty specification — including `penalties.json` schema, entity override format, stage-varying override schemas, resolution semantics, and the full penalty inventory — see [Penalty System](penalty-system.md).

## Cross-References

- [Design Principles](../overview/design-principles.md) — format selection criteria and declaration order invariance
- [Configuration Reference](../configuration/configuration-reference.md) — complete field-by-field config.json reference
- [Penalty System](penalty-system.md) — full penalty specification with cascade resolution
- [Input System Entities](input-system-entities.md) — all 7 element type registries (buses, lines, hydros, thermals, non-controllable sources, pumping stations, energy contracts)
- [Input Hydro Extensions](input-hydro-extensions.md) — hydro geometry, production models, FPHA hyperplanes
- [Input Scenarios](input-scenarios.md) — stages.json schema, inflow models, load factors, correlations
- [Input Constraints](input-constraints.md) — initial conditions, stage-varying bounds, exchange factors, generic constraints
- [Internal Structures](internal-structures.md) — in-memory data model built from these input files
- [Binary Formats](binary-formats.md) — policy directory FlatBuffers schemas (cuts, states, basis)
- [Scenario Generation](../architecture/scenario-generation.md) — Scenario pipeline architecture
- [ADR-008: User-Supplied Opening Tree](../../cobre/docs/adr/008-user-supplied-opening-tree.md) — Schema and validation rules for `scenarios/noise_openings.parquet` (cobre repository)
- [ADR-009: Stochastic Artifact Export](../../cobre/docs/adr/009-stochastic-artifact-export.md) — `output/stochastic/` directory layout and round-trip invariant (cobre repository)
- [Production Scale Reference](../overview/production-scale-reference.md) — LP sizing and performance targets
