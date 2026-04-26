# Output Schemas

## Purpose

This spec defines the Parquet schemas for all output files produced by Cobre during simulation (policy evaluation) and training (policy construction). It covers the output directory layout, column-level schemas for every entity type, and the dictionary/metadata files that make outputs self-documenting.

For output infrastructure (manifests, MPI partitioning, configuration, scale reference, validation), see [Output Infrastructure](output-infrastructure.md).

## 1. Directory Structure

```
output/
├── simulation/                              # Hive-partitioned simulation results
│   ├── _manifest.json                       # Checksums, row counts, partitions
│   ├── _SUCCESS                             # Marker on successful completion
│   ├── costs/
│   │   └── scenario_id=XXXX/data.parquet
│   ├── hydros/
│   │   └── scenario_id=XXXX/data.parquet
│   ├── thermals/
│   │   └── scenario_id=XXXX/data.parquet
│   ├── exchanges/
│   │   └── scenario_id=XXXX/data.parquet
│   ├── buses/
│   │   └── scenario_id=XXXX/data.parquet
│   ├── pumping_stations/                    # Optional: only if entity exists
│   │   └── scenario_id=XXXX/data.parquet
│   ├── contracts/                           # Optional: only if entity exists
│   │   └── scenario_id=XXXX/data.parquet
│   ├── non_controllables/                   # Optional: only if entity exists
│   │   └── scenario_id=XXXX/data.parquet
│   ├── batteries/                           # DEFERRED (forward-compatible placeholder)
│   │   └── scenario_id=XXXX/data.parquet
│   ├── inflow_lags/                         # Optional: only if AR order > 0
│   │   └── scenario_id=XXXX/data.parquet
│   └── violations/
│       └── generic/
│           └── scenario_id=XXXX/data.parquet
│
├── stochastic/                              # Stochastic model fitting artifacts
│   ├── noise_openings.parquet               # Noise term realizations per opening
│   ├── inflow_seasonal_stats.parquet        # Monthly inflow statistics
│   ├── inflow_ar_coefficients.parquet       # Autoregressive model coefficients
│   ├── correlation.json                     # Cross-correlation structure
│   ├── load_seasonal_stats.parquet          # Monthly load statistics
│   └── fitting_report.json                  # Stochastic model fitting diagnostics
│
├── hydro_models/                            # Computed FPHA hyperplanes (optional)
│   └── fpha_hyperplanes.parquet             # Written when any hydro uses computed FPHA
│
└── training/                                # Training phase outputs
    ├── _manifest.json
    ├── _SUCCESS
    ├── convergence.parquet                  # Iteration-level convergence
    ├── scaling_report.json                  # LP scaling diagnostics
    ├── model_provenance.json                # Model provenance metadata
    ├── timing/
    │   ├── iterations.parquet               # Per-iteration timing breakdown
    │   └── mpi_ranks.parquet                # Per-rank timing statistics
    ├── solver/
    │   ├── iterations.parquet               # Per-iteration solver statistics
    │   └── retry_histogram.parquet          # Per-iteration retry level histogram
    ├── cut_selection/
    │   └── iterations.parquet               # Per-iteration cut selection stats
    ├── dictionaries/
    │   ├── codes.json                       # Categorical code mappings
    │   ├── bounds.parquet                   # Entity bounds by stage/block
    │   ├── state_dictionary.json            # State space definition
    │   ├── variables.csv                    # Variable metadata
    │   └── entities.csv                     # Entity metadata
    └── metadata.json                        # Run configuration and system info
```

> **Optional files**: Simulation entity directories are only written if the corresponding entities exist in the input model. The `hydro_models/` directory is only written when at least one hydro uses the computed-source FPHA model.

## 2. Design Principles

### 2.1 Hive Partitioning by Scenario

Simulation outputs use Hive-style partitioning by `scenario_id`:

- **Parallel writes**: Each MPI rank writes exclusively to its assigned scenario partitions
- **Partition pruning**: Queries filtering by scenario read only relevant files
- **Incremental updates**: Individual scenarios can be recomputed without rewriting all data

**Partition naming**: `{entity_type}/scenario_id={scenario_id:04d}/data.parquet`

The `scenario_id` column is NOT stored in Parquet data — it is derived from the partition path.

### 2.2 Categorical Encoding

All categorical columns use integer codes with mappings in `dictionaries/codes.json`:

- Reduces storage (i8 vs variable-length strings)
- Enables efficient filtering and grouping
- **Convention**: categorical columns end with `_code` suffix (e.g., `operative_state_code`)

### 2.3 Constraint Violation Handling

Two mechanisms:

1. **Slack columns in entity files**: Physical bound violations (e.g., `turbined_slack_m3s`) appear as dedicated columns, value 0 when no violation
2. **Generic violations file**: User-defined generic constraint violations in `violations/generic/`

### 2.4 File Naming Conventions

| Convention                   | Example                         | Rationale                |
| ---------------------------- | ------------------------------- | ------------------------ |
| Plural entity names          | `hydros.parquet`                | Multiple records         |
| Lowercase with underscores   | `pumping_stations/`             | Filesystem-safe          |
| `data.parquet` in partitions | `scenario_id=0001/data.parquet` | Standard Hive convention |

## 3. Categorical Codes (`dictionaries/codes.json`)

```json
{
  "version": "1.0",
  "generated_at": "2026-01-18T12:00:00Z",
  "operative_state": {
    "0": "deactivated",
    "1": "maintenance",
    "2": "operating",
    "3": "saturated"
  },
  "storage_binding": {
    "0": "none",
    "1": "below_minimum",
    "2": "above_maximum",
    "3": "both"
  },
  "contract_type": {
    "0": "import",
    "1": "export"
  },
  "entity_type": {
    "0": "hydro",
    "1": "thermal",
    "2": "bus",
    "3": "line",
    "4": "pumping_station",
    "5": "contract",
    "7": "non_controllable"
  },
  "bound_type": {
    "0": "storage_min",
    "1": "storage_max",
    "2": "turbined_min",
    "3": "turbined_max",
    "4": "outflow_min",
    "5": "outflow_max",
    "6": "generation_min",
    "7": "generation_max",
    "8": "flow_min",
    "9": "flow_max"
  }
}
```

> **Forward-compatible codes**: `entity_type` code 6 (`battery`) is reserved for future use. See [Deferred Features](../deferred.md) for implementation timeline.

**Usage in analysis:**

```python
import json, polars as pl

with open("training/dictionaries/codes.json") as f:
    codes = json.load(f)

df = pl.read_parquet("simulation/hydros/")
df = df.with_columns(
    pl.col("operative_state_code")
      .map_dict({int(k): v for k, v in codes["operative_state"].items()})
      .alias("operative_state")
)
```

## 4. Dictionary Files

### 4.1 Bounds Dictionary (`dictionaries/bounds.parquet`)

Centralizes all entity bounds by stage and block, eliminating redundant bound columns from entity output files.

| Column             | Type | Nullable | Description                                |
| ------------------ | ---- | -------- | ------------------------------------------ |
| `entity_type_code` | i8   | No       | Entity type code (see `codes.json`)        |
| `entity_id`        | i32  | No       | Entity identifier                          |
| `stage_id`         | i32  | No       | Stage index (0-based)                      |
| `block_id`         | i32  | Yes      | Block index (null = applies to all blocks) |
| `bound_type_code`  | i8   | No       | Bound type code (see `codes.json`)         |
| `bound_value`      | f64  | No       | Bound value in native units                |

Bounds are stored only when they differ from default/infinite values.

### 4.2 State Dictionary (`dictionaries/state_dictionary.json`)

Documents the state space structure for the SDDP policy. See [Input Constraints §4](input-constraints.md) for full state schema.

### 4.3 Variables Metadata (`dictionaries/variables.csv`)

| Column        | Type   | Description                                   |
| ------------- | ------ | --------------------------------------------- |
| `file`        | string | Source file (e.g., `hydros`, `thermals`)      |
| `column`      | string | Column name                                   |
| `type`        | string | Data type (`i8`, `i32`, `i64`, `f64`, `bool`) |
| `unit`        | string | Physical unit or null                         |
| `description` | string | Human-readable description                    |
| `nullable`    | bool   | Whether null values are allowed               |

### 4.4 Entities Metadata (`dictionaries/entities.csv`)

| Column             | Type   | Description                   |
| ------------------ | ------ | ----------------------------- |
| `entity_type_code` | i8     | Entity type code              |
| `entity_id`        | i32    | Entity identifier             |
| `name`             | string | Entity name from input        |
| `bus_id`           | i32    | Connected bus (if applicable) |
| `system_id`        | i32    | System/subsystem identifier   |

## 5. Simulation Output Schemas

### 5.1 Costs (`simulation/costs/`)

Stage and block-level cost breakdown for economic analysis. Cost columns are organized by the three penalty categories defined in [Penalty System §2](penalty-system.md).

| Column                          | Type | Nullable | Description                                                                                                                     |
| ------------------------------- | ---- | -------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `stage_id`                      | i32  | No       | Stage index (0-based)                                                                                                           |
| `block_id`                      | i32  | Yes      | Block index (null for stage-level aggregates)                                                                                   |
| `total_cost`                    | f64  | No       | Total stage cost (all components)                                                                                               |
| `immediate_cost`                | f64  | No       | Stage immediate cost (excluding future cost)                                                                                    |
| `future_cost`                   | f64  | No       | Future cost function value ($\theta$)                                                                                           |
| `discount_factor`               | f64  | No       | Cumulative discount factor $\prod_{s=1}^{t-1} d_{s \to s+1}$ (see [Discount Rate §5](../math/discount-rate.md))                 |
| **Resource costs**              |      |          |                                                                                                                                 |
| `thermal_cost`                  | f64  | No       | Thermal generation cost                                                                                                         |
| `contract_cost`                 | f64  | No       | Import/export contract cost (net: imports positive, exports negative)                                                           |
| **Category 1 — Recourse**       |      |          |                                                                                                                                 |
| `deficit_cost`                  | f64  | No       | Deficit (unmet demand) penalty — piecewise segments                                                                             |
| `excess_cost`                   | f64  | No       | Excess generation penalty                                                                                                       |
| **Category 2 — Violations**     |      |          |                                                                                                                                 |
| `storage_violation_cost`        | f64  | No       | Storage below minimum violation cost                                                                                            |
| `filling_target_cost`           | f64  | No       | Filling target shortfall cost                                                                                                   |
| `hydro_violation_cost`          | f64  | No       | Sum of all hydro violation costs (aggregate of the 6 granular columns below)                                                    |
| `outflow_violation_below_cost`  | f64  | No       | Outflow below minimum violation cost (all hydros)                                                                               |
| `outflow_violation_above_cost`  | f64  | No       | Outflow above maximum violation cost (all hydros)                                                                               |
| `turbined_violation_cost`       | f64  | No       | Turbined flow below minimum violation cost (all hydros)                                                                         |
| `generation_violation_cost`     | f64  | No       | Generation below minimum violation cost (all hydros)                                                                            |
| `evaporation_violation_cost`    | f64  | No       | Evaporation constraint violation cost (all hydros)                                                                              |
| `withdrawal_violation_cost`     | f64  | No       | Water withdrawal violation cost (all hydros)                                                                                    |
| `inflow_penalty_cost`           | f64  | No       | Inflow non-negativity penalty cost (see [Inflow Non-Negativity](../math/inflow-nonnegativity.md))                               |
| `generic_violation_cost`        | f64  | No       | Generic constraint violation penalties                                                                                          |
| **Category 3 — Regularization** |      |          |                                                                                                                                 |
| `spillage_cost`                 | f64  | No       | Spillage regularization cost (all hydros)                                                                                       |
| `fpha_turbined_cost`            | f64  | No       | FPHA turbined flow regularization cost (FPHA hydros only)                                                                       |
| `curtailment_cost`              | f64  | No       | Non-controllable source curtailment cost                                                                                        |
| `exchange_cost`                 | f64  | No       | Exchange regularization cost (all lines)                                                                                        |
| `pumping_cost`                  | f64  | No       | Imputed pumping cost (marginal price × energy consumed, not a direct LP cost term — see [Penalty System §8](penalty-system.md)) |

**Rows per scenario**: `num_stages × (1 + num_blocks)` (stage-level + block-level rows)

**Cost relationships**:

```
total_cost = immediate_cost + discount_factor_applied_to_future * future_cost
immediate_cost = thermal_cost + contract_cost
               + deficit_cost + excess_cost
               + storage_violation_cost + filling_target_cost + hydro_violation_cost
               + inflow_penalty_cost + generic_violation_cost
               + spillage_cost + fpha_turbined_cost + curtailment_cost
               + exchange_cost + pumping_cost

hydro_violation_cost = outflow_violation_below_cost + outflow_violation_above_cost
                     + turbined_violation_cost + generation_violation_cost
                     + evaporation_violation_cost + withdrawal_violation_cost
```

> **Note on `pumping_cost`**: Pumping stations have no explicit cost parameter in the LP. The `pumping_cost` column reports the imputed cost: marginal price at the connected bus × energy consumed. This is derived from dual variables after the solve, not from a direct LP cost term.

> **Note on `storage_violation_cost` and `filling_target_cost`**: These penalties apply to end-of-stage storage (hm³) and appear outside the $\tau_k$ block summation in the LP objective. They are NOT per-block costs. In the costs output, they appear only in stage-level rows (where `block_id` is null).

### 5.2 Hydros (`simulation/hydros/`)

| Column                               | Type | Nullable | Description                                                            |
| ------------------------------------ | ---- | -------- | ---------------------------------------------------------------------- |
| `stage_id`                           | i32  | No       | Stage index (0-based)                                                  |
| `block_id`                           | i32  | Yes      | Block index (null for stage-level)                                     |
| `hydro_id`                           | i32  | No       | Hydro plant identifier                                                 |
| `turbined_m3s`                       | f64  | No       | Turbined outflow (m³/s)                                                |
| `spillage_m3s`                       | f64  | No       | Spillage (m³/s)                                                        |
| `outflow_m3s`                        | f64  | No       | Total outflow: turbined + spillage (m³/s)                              |
| `evaporation_m3s`                    | f64  | Yes      | Evaporation loss (m³/s), null if not modeled                           |
| `diverted_inflow_m3s`                | f64  | Yes      | Inflow diverted from upstream                                          |
| `diverted_outflow_m3s`               | f64  | Yes      | Outflow diverted to downstream                                         |
| `incremental_inflow_m3s`             | f64  | No       | Realized incremental inflow (m³/s)                                     |
| `inflow_m3s`                         | f64  | No       | Total inflow including upstream                                        |
| `storage_initial_hm3`                | f64  | No       | Storage at start (hm³)                                                 |
| `storage_final_hm3`                  | f64  | No       | Storage at end (hm³)                                                   |
| `generation_mw`                      | f64  | No       | Power generation (MW)                                                  |
| `generation_mwh`                     | f64  | No       | Energy generation (MWh)                                                |
| `productivity_mw_per_m3s`            | f64  | Yes      | Effective productivity                                                 |
| `spillage_cost`                      | f64  | No       | Spillage regularization cost (this plant)                              |
| `water_value_per_hm3`                | f64  | No       | Marginal value of stored water (\$/hm³)                                |
| `storage_binding_code`               | i8   | No       | Storage bound binding status                                           |
| `operative_state_code`               | i8   | No       | Operative state                                                        |
| **Violation slacks**                 |      |          |                                                                        |
| `turbined_slack_m3s`                 | f64  | No       | Min turbined violation (0 if none)                                     |
| `outflow_slack_below_m3s`            | f64  | No       | Min outflow violation (0 if none)                                      |
| `outflow_slack_above_m3s`            | f64  | No       | Max outflow violation (0 if none)                                      |
| `generation_slack_mw`                | f64  | No       | Min generation violation (0 if none)                                   |
| `storage_violation_below_hm3`        | f64  | No       | Storage below minimum violation (0 if none, stage-level only)          |
| `filling_target_violation_hm3`       | f64  | No       | Filling target shortfall (0 if none, filling hydros at terminal stage) |
| `evaporation_violation_pos_m3s`      | f64  | No       | Evaporation over-estimate violation slack (0 if none)                  |
| `evaporation_violation_neg_m3s`      | f64  | No       | Evaporation under-estimate violation slack (0 if none)                 |
| `inflow_nonnegativity_slack_m3s`     | f64  | No       | Inflow non-negativity slack (0 if none, stage-level)                   |
| `water_withdrawal_violation_pos_m3s` | f64  | No       | Water withdrawal over-withdrawal violation slack (0 if none)           |
| `water_withdrawal_violation_neg_m3s` | f64  | No       | Water withdrawal under-withdrawal violation slack (0 if none)          |

**Rows per scenario**: `num_stages × num_blocks × num_hydros`

**Water balance**: `storage_final = storage_initial + (inflow - outflow - evaporation + diverted_inflow - diverted_outflow) × duration`

**Slack interpretation**: value > 0 means the corresponding constraint was relaxed. Storage and filling target slacks are stage-level (not per-block). See [Penalty System §4](penalty-system.md) for the full violation catalogue.

### 5.3 Thermals (`simulation/thermals/`)

| Column                 | Type | Nullable | Description                                      |
| ---------------------- | ---- | -------- | ------------------------------------------------ |
| `stage_id`             | i32  | No       | Stage index (0-based)                            |
| `block_id`             | i32  | Yes      | Block index                                      |
| `thermal_id`           | i32  | No       | Thermal unit identifier                          |
| `generation_mw`        | f64  | No       | Power generation (MW)                            |
| `generation_mwh`       | f64  | No       | Energy generation (MWh)                          |
| `generation_cost`      | f64  | No       | Generation cost                                  |
| `is_gnl`               | bool | No       | Whether unit has GNL configuration               |
| `gnl_committed_mw`     | f64  | Yes      | GNL committed capacity (null if not GNL)         |
| `gnl_decision_mw`      | f64  | Yes      | GNL decision for future stages (null if not GNL) |
| `operative_state_code` | i8   | No       | Operative state                                  |

**Rows per scenario**: `num_stages × num_blocks × num_thermals`

**GNL notes**: `gnl_committed_mw` is capacity committed previously and available now; `gnl_decision_mw` is the decision made now for future availability. GNL decisions are state variables coupling stages.

### 5.4 Exchanges (`simulation/exchanges/`)

| Column                 | Type | Nullable | Description                                                             |
| ---------------------- | ---- | -------- | ----------------------------------------------------------------------- |
| `stage_id`             | i32  | No       | Stage index (0-based)                                                   |
| `block_id`             | i32  | Yes      | Block index                                                             |
| `line_id`              | i32  | No       | Transmission line identifier                                            |
| `direct_flow_mw`       | f64  | No       | Direct flow $f^+$ bus_from → bus_to (MW)                                |
| `reverse_flow_mw`      | f64  | No       | Reverse flow $f^-$ bus_to → bus_from (MW)                               |
| `net_flow_mw`          | f64  | No       | Net flow: $f^+ - f^-$ (MW), derived                                     |
| `net_flow_mwh`         | f64  | No       | Net energy flow (MWh)                                                   |
| `losses_mw`            | f64  | No       | Transmission losses: $(1 - \eta) \cdot f^+ + (1 - \eta) \cdot f^-$ (MW) |
| `losses_mwh`           | f64  | No       | Transmission losses energy (MWh)                                        |
| `exchange_cost`        | f64  | No       | Exchange regularization cost (this line)                                |
| `operative_state_code` | i8   | No       | Operative state                                                         |

**Rows per scenario**: `num_stages × num_blocks × num_lines`

**Sign convention**: `net_flow_mw` positive = bus_from → bus_to; negative = bus_to → bus_from.

> **Note**: The LP decision variables are `direct_flow_mw` ($f^+$) and `reverse_flow_mw` ($f^-$), both non-negative. `net_flow_mw` and losses are derived columns for analysis convenience. See [System Elements §4](../math/system-elements.md) for the exchange model.

### 5.5 Buses (`simulation/buses/`)

| Column        | Type | Nullable | Description                          |
| ------------- | ---- | -------- | ------------------------------------ |
| `stage_id`    | i32  | No       | Stage index (0-based)                |
| `block_id`    | i32  | Yes      | Block index                          |
| `bus_id`      | i32  | No       | Bus identifier                       |
| `load_mw`     | f64  | No       | Realized load after curtailment (MW) |
| `load_mwh`    | f64  | No       | Realized load energy (MWh)           |
| `deficit_mw`  | f64  | No       | Unmet demand (MW)                    |
| `deficit_mwh` | f64  | No       | Unmet demand energy (MWh)            |
| `excess_mw`   | f64  | No       | Excess generation (MW)               |
| `excess_mwh`  | f64  | No       | Excess generation energy (MWh)       |
| `spot_price`  | f64  | No       | Marginal cost of energy (\$/MWh)     |

**Rows per scenario**: `num_stages × num_blocks × num_buses`

**Load balance**: `generation_total + imports − exports + deficit − excess = load`

To compute generation by source, join with `hydros`, `thermals`, etc. using `bus_id` from `entities.csv`.

### 5.6 Pumping Stations (`simulation/pumping_stations/`) — Optional

| Column                   | Type | Nullable | Description                |
| ------------------------ | ---- | -------- | -------------------------- |
| `stage_id`               | i32  | No       | Stage index (0-based)      |
| `block_id`               | i32  | Yes      | Block index                |
| `pumping_station_id`     | i32  | No       | Pumping station identifier |
| `pumped_flow_m3s`        | f64  | No       | Pumped water flow (m³/s)   |
| `pumped_volume_hm3`      | f64  | No       | Pumped volume (hm³)        |
| `power_consumption_mw`   | f64  | No       | Power consumed (MW)        |
| `energy_consumption_mwh` | f64  | No       | Energy consumed (MWh)      |
| `pumping_cost`           | f64  | No       | Imputed pumping cost       |
| `operative_state_code`   | i8   | No       | Operative state            |

**Rows per scenario**: `num_stages × num_blocks × num_pumping_stations`

> **Note on `pumping_cost`**: Pumping stations have no explicit cost parameter in the LP objective. This column reports the imputed cost: marginal price at the connected bus (dual of load balance constraint, \$/MWh) × energy consumed (MWh). It is computed after the solve from dual variables, not from a direct LP cost term. See [Penalty System §8](penalty-system.md).

### 5.7 Contracts (`simulation/contracts/`) — Optional

| Column                 | Type | Nullable | Description             |
| ---------------------- | ---- | -------- | ----------------------- |
| `stage_id`             | i32  | No       | Stage index (0-based)   |
| `block_id`             | i32  | Yes      | Block index             |
| `contract_id`          | i32  | No       | Contract identifier     |
| `power_mw`             | f64  | No       | Contracted power (MW)   |
| `energy_mwh`           | f64  | No       | Contracted energy (MWh) |
| `price_per_mwh`        | f64  | No       | Contract price (\$/MWh) |
| `total_cost`           | f64  | No       | Total contract cost     |
| `operative_state_code` | i8   | No       | Operative state         |

**Rows per scenario**: `num_stages × num_blocks × num_contracts`

### 5.8 Non-Controllable Sources (`simulation/non_controllables/`) — Optional

| Column                 | Type | Nullable | Description                             |
| ---------------------- | ---- | -------- | --------------------------------------- |
| `stage_id`             | i32  | No       | Stage index (0-based)                   |
| `block_id`             | i32  | Yes      | Block index                             |
| `non_controllable_id`  | i32  | No       | Non-controllable source identifier      |
| `generation_mw`        | f64  | No       | Dispatched generation (MW)              |
| `generation_mwh`       | f64  | No       | Dispatched generation (MWh)             |
| `available_mw`         | f64  | No       | Available generation from scenario (MW) |
| `curtailment_mw`       | f64  | No       | Curtailed generation (MW)               |
| `curtailment_mwh`      | f64  | No       | Curtailed generation (MWh)              |
| `curtailment_cost`     | f64  | No       | Curtailment regularization cost         |
| `operative_state_code` | i8   | No       | Operative state                         |

**Rows per scenario**: `num_stages × num_blocks × num_non_controllable_sources`

Non-controllable sources are fully defined in [Input System Entities §7](input-system-entities.md). Curtailment cost is a Category 3 regularization penalty — see [Penalty System §2](penalty-system.md).

### 5.9 Batteries (`simulation/batteries/`) — DEFERRED

| Column                 | Type | Nullable | Description                    |
| ---------------------- | ---- | -------- | ------------------------------ |
| `stage_id`             | i32  | No       | Stage index (0-based)          |
| `block_id`             | i32  | Yes      | Block index                    |
| `battery_id`           | i32  | No       | Battery identifier             |
| `charge_mw`            | f64  | No       | Charging power (MW)            |
| `discharge_mw`         | f64  | No       | Discharging power (MW)         |
| `soc_initial_mwh`      | f64  | No       | State of charge at start (MWh) |
| `soc_final_mwh`        | f64  | No       | State of charge at end (MWh)   |
| `cycle_cost`           | f64  | No       | Cycling degradation cost       |
| `operative_state_code` | i8   | No       | Operative state                |

> **Forward-compatible placeholder**: The schema and `entity_type` code 6 are reserved. Exceptional validation rejects battery entities at input loading until the implementation is ready. See [Deferred Features](../deferred.md) for implementation timeline.

### 5.10 Inflow Lags (`simulation/inflow_lags/`) — Optional

| Column       | Type | Nullable | Description                      |
| ------------ | ---- | -------- | -------------------------------- |
| `stage_id`   | i32  | No       | Stage index (0-based)            |
| `hydro_id`   | i32  | No       | Hydro plant identifier           |
| `lag_index`  | i32  | No       | Lag index (1 = t−1, 2 = t−2, …)  |
| `inflow_m3s` | f64  | No       | Inflow value for this lag (m³/s) |

**Rows per scenario**: `num_stages × num_hydros × max_ar_order`

`lag_index` uses 1-based indexing. Maximum lag index equals the AR model order. These values are state variables affecting inflow sampling.

### 5.11 Generic Violations (`simulation/violations/generic/`)

| Column          | Type | Nullable | Description                         |
| --------------- | ---- | -------- | ----------------------------------- |
| `stage_id`      | i32  | No       | Stage index (0-based)               |
| `block_id`      | i32  | Yes      | Block index                         |
| `constraint_id` | i32  | No       | Generic constraint identifier       |
| `slack_value`   | f64  | No       | Violation amount (constraint units) |
| `slack_cost`    | f64  | No       | Penalty cost incurred               |

**Rows per scenario**: `num_stages × num_blocks × num_generic_constraints` (only non-zero violations may be stored).

## 6. Training Output Schemas

### 6.1 Convergence Log (`training/convergence.parquet`)

| Column             | Type | Nullable | Description                                    |
| ------------------ | ---- | -------- | ---------------------------------------------- |
| `iteration`        | i32  | No       | Iteration number (1-based)                     |
| `lower_bound`      | f64  | No       | Lower bound (expected cost-to-go from stage 0) |
| `upper_bound_mean` | f64  | No       | Upper bound mean                               |
| `upper_bound_std`  | f64  | No       | Upper bound standard deviation                 |
| `gap_percent`      | f64  | Yes      | Optimality gap (null when not computable)      |
| `cuts_added`       | i32  | No       | Cuts added this iteration                      |
| `cuts_removed`     | i32  | No       | Cuts removed by cut selection                  |
| `cuts_active`      | i64  | No       | Total active cuts across all stages            |
| `time_forward_ms`  | i64  | No       | Forward pass wall time (ms)                    |
| `time_backward_ms` | i64  | No       | Backward pass wall time (ms)                   |
| `time_total_ms`    | i64  | No       | Total iteration wall time (ms)                 |
| `forward_passes`   | i32  | No       | Forward scenarios this iteration               |
| `lp_solves`        | i64  | No       | Total LP solves this iteration                 |

**Rows**: `num_iterations`

**Upper bound mechanisms**: Two distinct mechanisms populate the upper bound columns:

1. **Simulation-based (Monte Carlo)**: Runs the SDDP policy on sampled scenarios and averages costs. Provides `upper_bound_mean` and `upper_bound_std`. Valid only for **risk-neutral** problems — see [Stopping Rules](../math/stopping-rules.md).
2. **SIDP deterministic (inner approximation)**: Vertex-based upper bound via Lipschitz interpolation. Provides a deterministic `upper_bound_mean` (no std). Valid for **both risk-neutral and risk-averse** problems — see [Upper Bound Evaluation](../math/upper-bound-evaluation.md).

Which mechanism is active depends on configuration. Both may run simultaneously (simulation-based for reporting, SIDP for convergence).

**Gap computation**: `gap_percent = (upper_bound_mean − lower_bound) / |lower_bound| × 100`. If UB evaluation is disabled, `gap_percent` is null. If only LB is available, the gap is computed from the LB change between iterations (see [Stopping Rules §3](../math/stopping-rules.md)).

> **Risk-averse interpretation**: Under risk-averse (CVaR) settings, `lower_bound` is a **convergence indicator**, not a valid lower bound on the true risk-averse cost. It represents the first-stage objective value using the current cut approximation. For risk-averse convergence verification, the SIDP deterministic upper bound is required. `gap_percent` from the simulation-based UB is not meaningful for risk-averse problems. See [Risk Measures §10](../math/risk-measures.md).

### 6.2 Iteration Timing (`training/timing/iterations.parquet`)

| Column               | Type | Nullable | Description                             |
| -------------------- | ---- | -------- | --------------------------------------- |
| `iteration`          | i32  | No       | Iteration number (1-based)              |
| `forward_solve_ms`   | i64  | No       | LP solve time in forward pass           |
| `forward_sample_ms`  | i64  | No       | Scenario sampling time                  |
| `backward_solve_ms`  | i64  | No       | LP solve time in backward pass          |
| `backward_cut_ms`    | i64  | No       | Cut computation and storage time        |
| `cut_selection_ms`   | i64  | No       | Cut selection/pruning time              |
| `mpi_allreduce_ms`   | i64  | No       | MPI AllReduce communication time        |
| `mpi_broadcast_ms`   | i64  | No       | MPI Broadcast communication time        |
| `io_write_ms`        | i64  | No       | Output writing time                     |
| `state_exchange_ms`  | i64  | No       | State exchange time between passes      |
| `cut_batch_build_ms` | i64  | No       | Cut batch construction time             |
| `rayon_overhead_ms`  | i64  | No       | Rayon thread pool coordination overhead |
| `overhead_ms`        | i64  | No       | Unaccounted overhead                    |

**Rows**: `num_iterations`

### 6.3 MPI Rank Timing (`training/timing/mpi_ranks.parquet`)

| Column                  | Type | Nullable | Description                      |
| ----------------------- | ---- | -------- | -------------------------------- |
| `iteration`             | i32  | No       | Iteration number (1-based)       |
| `rank`                  | i32  | No       | MPI rank (0-based)               |
| `forward_time_ms`       | i64  | No       | Forward pass time on this rank   |
| `backward_time_ms`      | i64  | No       | Backward pass time on this rank  |
| `communication_time_ms` | i64  | No       | MPI communication time           |
| `idle_time_ms`          | i64  | No       | Time waiting for other ranks     |
| `lp_solves`             | i64  | No       | LP solves on this rank           |
| `scenarios_processed`   | i32  | No       | Scenarios processed on this rank |

**Rows**: `num_iterations × num_mpi_ranks`

Use `idle_time_ms` to identify load imbalance. Sum of `scenarios_processed` per iteration equals `forward_passes`. Communication patterns reveal MPI bottlenecks.

### 6.4 Solver Statistics (`training/solver/iterations.parquet`)

Per-iteration, per-phase, per-stage solver statistics for diagnosing LP conditioning and retry behavior. One row per `(iteration, phase, stage)` triple.

| Column               | Type   | Nullable | Description                                                               |
| -------------------- | ------ | -------- | ------------------------------------------------------------------------- |
| `iteration`          | u32    | No       | Iteration number                                                          |
| `phase`              | string | No       | SDDP phase: `"forward"`, `"backward"`, `"lower_bound"`, or `"simulation"` |
| `stage`              | i32    | No       | Stage index                                                               |
| `lp_solves`          | u32    | No       | Total LP solves at this stage                                             |
| `lp_successes`       | u32    | No       | LP solves that succeeded on first attempt                                 |
| `lp_retries`         | u32    | No       | LP solves that succeeded after retry                                      |
| `lp_failures`        | u32    | No       | LP solves that failed after all retry attempts                            |
| `retry_attempts`     | u32    | No       | Total retry attempts across all LP solves                                 |
| `basis_offered`      | u32    | No       | LP solves where a warm-start basis was offered                            |
| `basis_rejections`   | u32    | No       | LP solves where the offered basis was rejected                            |
| `simplex_iterations` | u64    | No       | Total simplex iterations across all LP solves                             |
| `solve_time_ms`      | f64    | No       | Cumulative LP solve wall time (ms)                                        |
| `load_model_time_ms` | f64    | No       | Cumulative model loading time (ms)                                        |
| `add_rows_time_ms`   | f64    | No       | Cumulative row addition time (ms)                                         |
| `set_bounds_time_ms` | f64    | No       | Cumulative bound setting time (ms)                                        |
| `basis_set_time_ms`  | f64    | No       | Cumulative basis setting time (ms)                                        |

**Rows**: `num_iterations × num_phases × num_stages`

**Phase values**: The `phase` column identifies the SDDP algorithmic phase that produced the LP solves. `"forward"` and `"backward"` are the standard SDDP passes. `"lower_bound"` appears when a dedicated lower-bound evaluation solve is performed (separate from the forward pass). `"simulation"` appears when simulation-based upper bound evaluation is enabled (see [Upper Bound Evaluation](../math/upper-bound-evaluation.md)).

Retry-level histograms are stored separately in `retry_histogram.parquet` (SS6.6). See [Solver Interface](../architecture/solver-interface-trait.md) for retry strategy details.

### 6.5 Cut Selection Statistics (`training/cut_selection/iterations.parquet`)

Per-iteration, per-stage cut selection statistics. One row per `(iteration, stage)` pair.

| Column               | Type | Nullable | Description                                          |
| -------------------- | ---- | -------- | ---------------------------------------------------- |
| `iteration`          | i32  | No       | Iteration number                                     |
| `stage`              | i32  | No       | Stage index                                          |
| `cuts_populated`     | i32  | No       | Total cuts in the pool before selection              |
| `cuts_active_before` | i32  | No       | Active cuts before this iteration's selection        |
| `cuts_deactivated`   | i32  | No       | Cuts deactivated by this iteration's selection       |
| `cuts_active_after`  | i32  | No       | Active cuts after this iteration's selection         |
| `selection_time_ms`  | f64  | No       | Wall-clock time for cut selection at this stage (ms) |

**Rows**: `num_iterations × num_stages` (only iterations where cut selection runs)

### 6.6 Retry Histogram (`training/solver/retry_histogram.parquet`)

Per-iteration, per-phase, per-stage histogram of retry escalation levels. One row per `(iteration, phase, stage, retry_level)` tuple where the count is non-zero.

| Column        | Type   | Nullable | Description                                                               |
| ------------- | ------ | -------- | ------------------------------------------------------------------------- |
| `iteration`   | i32    | No       | Iteration number                                                          |
| `phase`       | string | No       | SDDP phase: `"forward"`, `"backward"`, `"lower_bound"`, or `"simulation"` |
| `stage`       | i32    | No       | Stage index                                                               |
| `retry_level` | u32    | No       | Retry escalation level (0-based)                                          |
| `count`       | u64    | No       | Number of LP solves that reached this retry level                         |

**Rows**: sparse — only `(iteration, phase, stage, retry_level)` tuples with `count > 0` are stored.

The retry ladder applies increasingly aggressive solver reconfiguration at each level (e.g., scaling, presolve toggling, algorithm switching). See [Solver Interface](../architecture/solver-interface-trait.md) for the full retry strategy. This file complements the aggregate retry statistics in `iterations.parquet` (SS6.4) by providing per-level granularity.

## 7. Structured Output vs Parquet Schemas

The Cobre output system produces two distinct categories of schemas with different purposes and access patterns:

| Category                   | Format            | Transport  | Purpose                                               | Access Method                 |
| -------------------------- | ----------------- | ---------- | ----------------------------------------------------- | ----------------------------- |
| **Parquet schemas** (§5-6) | Apache Parquet    | Disk files | Post-hoc analysis, archival, large tabular data       | Arrow/Polars/Pandas libraries |
| **Structured CLI output**  | JSON / JSON-lines | stdout     | Real-time monitoring, programmatic result consumption | Standard JSON parsing         |

**Key distinction**: Parquet schemas define the on-disk storage format for simulation results and training diagnostics. Structured CLI output defines the presentation format for CLI responses. They serve different audiences and use different data paths.

**No JSON equivalents for simulation Parquet schemas**: The 11 simulation entity schemas (§5.1-5.11) and 2 of the 3 training diagnostic schemas (§6.2-6.3) remain Parquet-only. Agents access them via Arrow libraries (available in Python, R, Julia, and Rust). Converting hundreds of MB of tabular data to JSON would be impractical and unnecessary.

**Convergence log has a JSON equivalent**: The convergence log schema (§6.1) has a JSON equivalent via the JSON-lines streaming protocol. The `progress` events emitted during training (see [Convergence Monitoring §4.1](../architecture/convergence-monitoring.md)) contain the same fields as the `convergence.parquet` columns. The Parquet file is the permanent record; JSON-lines is the ephemeral real-time stream.

**MCP resource serialization**: When the MCP server exposes Parquet data as resources (see [MCP Server](../interfaces/mcp-server.md)), it converts Parquet columns to JSON types using the following mapping:

| Parquet/Arrow Type | JSON Type | Notes                             |
| ------------------ | --------- | --------------------------------- |
| `Int32`, `Int64`   | number    |                                   |
| `Float64`          | number    | NaN/Infinity serialized as `null` |
| `Utf8`             | string    |                                   |
| `Boolean`          | boolean   |                                   |
| `Timestamp`        | string    | ISO 8601 format                   |
| Nullable columns   | `null`    | JSON null for missing values      |

This conversion is performed on demand by the MCP server's resource handlers, not stored as separate JSON files.

## 8. Preprocessing Output Schemas

The following files are written during the Initialization phase (before training), not during simulation or training proper. They document the result of preprocessing steps so that users can inspect and, if needed, re-use computed artifacts.

### 8.1 Computed FPHA Hyperplanes (`output/hydro_models/fpha_hyperplanes.parquet`)

Written automatically when one or more hydros use the `source: "computed"` FPHA fitting path. The file uses the same 11-column schema as the input file `system/fpha_hyperplanes.parquet` and is round-trip compatible with the parser in `cobre-io`.

> **Implementation note (v0.1.4):** This file is written by `prepare_hydro_models` in `cobre-sddp` on MPI rank 0 only, immediately after the fitting pipeline completes. It is not present when all FPHA hydros use `source: "precomputed"` or when no hydros use FPHA.

| Column            | Type        | Nullable | Description                                                     |
| ----------------- | ----------- | -------- | --------------------------------------------------------------- |
| `hydro_id`        | i32         | No       | Hydro plant identifier                                          |
| `stage_id`        | i32 \| null | Yes      | Stage this plane set applies to (`null` = all stages)           |
| `plane_id`        | i32         | No       | Plane index within hydro (0-based)                              |
| `gamma_0`         | f64         | No       | Intercept coefficient (MW)                                      |
| `gamma_v`         | f64         | No       | Volume coefficient (MW/hm³)                                     |
| `gamma_q`         | f64         | No       | Turbined flow coefficient (MW per m³/s)                         |
| `gamma_s`         | f64         | No       | Spillage coefficient (MW per m³/s, ≤ 0)                         |
| `kappa`           | f64         | No       | Correction factor κ (worst-case approach, in (0, 1])            |
| `valid_v_min_hm3` | f64         | Yes      | Volume validity minimum — `null` for all computed planes        |
| `valid_v_max_hm3` | f64         | Yes      | Volume validity maximum — `null` for all computed planes        |
| `valid_q_max_m3s` | f64         | Yes      | Turbined flow validity maximum — `null` for all computed planes |

The validity range columns (`valid_v_min_hm3`, `valid_v_max_hm3`, `valid_q_max_m3s`) are always `null` for computed planes. They are populated only when hyperplanes are provided externally via `system/fpha_hyperplanes.parquet` and the supplier chooses to include them. See [Input Hydro Extensions §3](input-hydro-extensions.md) for the complete schema description and validation rules.

**Round-trip use**: The exported file can be copied to `system/fpha_hyperplanes.parquet` in a subsequent run and referenced with `source: "precomputed"` to skip re-fitting. This is useful for large systems where fitting cost is non-trivial.

## Cross-References

- [Output Infrastructure](output-infrastructure.md) — Manifests, MPI partitioning, config, validation
- [Input System Entities](input-system-entities.md) — Entity registries defining entity IDs
- [Penalty System](penalty-system.md) — Three-category penalty taxonomy, cost columns alignment
- [Input Constraints](input-constraints.md) — Generic constraints whose violations appear in output
- [LP Formulation](../math/lp-formulation.md) — Mathematical definitions of output variables
- [System Elements](../math/system-elements.md) — Exchange model (direct/reverse flow), NCS, Variable Units Convention
- [Risk Measures](../math/risk-measures.md) — Lower bound validity under risk aversion (§10)
- [Upper Bound Evaluation](../math/upper-bound-evaluation.md) — SIDP deterministic upper bounds
- [Stopping Rules](../math/stopping-rules.md) — Simulation-based stopping and convergence criteria
- [Discount Rate](../math/discount-rate.md) — Cumulative discount factor formula
- [Inflow Non-Negativity](../math/inflow-nonnegativity.md) — Inflow penalty in costs
- [Deferred Features](../deferred.md) — Batteries (deferred)
- [Structured Output](../interfaces/structured-output.md) — CLI response envelope and JSON-lines streaming protocol
- [MCP Server](../interfaces/mcp-server.md) — MCP resource handlers serving Parquet data as JSON
- [Convergence Monitoring](../architecture/convergence-monitoring.md) — JSON-lines streaming schema (§4.1) as the JSON equivalent of convergence.parquet
