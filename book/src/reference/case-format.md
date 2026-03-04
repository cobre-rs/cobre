# Case Format Reference

A Cobre case directory is a self-contained folder that holds all input data
for a single power system study. `load_case` reads this directory and produces
a fully-validated `System` ready for the solver.

For a description of how these files are parsed and validated, see
[cobre-io](../crates/io.md).

## Directory layout

```
my_case/
├── config.json                              # Solver configuration (required)
├── penalties.json                           # Global penalty defaults (required)
├── stages.json                              # Stage sequence and policy graph (required)
├── initial_conditions.json                  # Reservoir storage at study start (required)
├── system/
│   ├── buses.json                           # Electrical buses (required)
│   ├── lines.json                           # Transmission lines (required)
│   ├── hydros.json                          # Hydro plants (required)
│   ├── thermals.json                        # Thermal plants (required)
│   ├── non_controllable_sources.json        # Intermittent sources (optional)
│   ├── pumping_stations.json                # Pumping stations (optional)
│   ├── energy_contracts.json                # Bilateral contracts (optional)
│   ├── hydro_geometry.parquet               # Reservoir geometry tables (optional)
│   ├── hydro_production_models.json         # FPHA production function configs (optional)
│   └── fpha_hyperplanes.parquet             # FPHA hyperplane coefficients (optional)
├── scenarios/
│   ├── inflow_history.parquet               # Historical inflow series (optional)
│   ├── inflow_seasonal_stats.parquet        # PAR model seasonal statistics (optional)
│   ├── inflow_ar_coefficients.parquet       # PAR autoregressive coefficients (optional)
│   ├── external_scenarios.parquet           # Pre-generated external scenarios (optional)
│   ├── load_seasonal_stats.parquet          # Load model seasonal statistics (optional)
│   ├── load_factors.json                    # Load scaling factors (optional)
│   └── correlation.json                     # Cross-series correlation model (optional)
└── constraints/
    ├── thermal_bounds.parquet               # Stage-varying thermal bounds (optional)
    ├── hydro_bounds.parquet                 # Stage-varying hydro bounds (optional)
    ├── line_bounds.parquet                  # Stage-varying line bounds (optional)
    ├── pumping_bounds.parquet               # Stage-varying pumping bounds (optional)
    ├── contract_bounds.parquet              # Stage-varying contract bounds (optional)
    ├── exchange_factors.json                # Block exchange factors (optional)
    ├── generic_constraints.json             # User-defined LP constraints (optional)
    ├── generic_constraint_bounds.parquet    # Bounds for generic constraints (optional)
    ├── penalty_overrides_bus.parquet        # Stage-varying bus penalty overrides (optional)
    ├── penalty_overrides_line.parquet       # Stage-varying line penalty overrides (optional)
    ├── penalty_overrides_hydro.parquet      # Stage-varying hydro penalty overrides (optional)
    └── penalty_overrides_ncs.parquet        # Stage-varying NCS penalty overrides (optional)
```

## File summary

| File                                            | Format  | Required | Description                             |
| ----------------------------------------------- | ------- | -------- | --------------------------------------- |
| `config.json`                                   | JSON    | Yes      | Solver configuration                    |
| `penalties.json`                                | JSON    | Yes      | Global penalty defaults                 |
| `stages.json`                                   | JSON    | Yes      | Stage sequence and policy graph         |
| `initial_conditions.json`                       | JSON    | Yes      | Initial reservoir storage               |
| `system/buses.json`                             | JSON    | Yes      | Electrical bus registry                 |
| `system/lines.json`                             | JSON    | Yes      | Transmission line registry              |
| `system/hydros.json`                            | JSON    | Yes      | Hydro plant registry                    |
| `system/thermals.json`                          | JSON    | Yes      | Thermal plant registry                  |
| `system/non_controllable_sources.json`          | JSON    | No       | Intermittent source registry            |
| `system/pumping_stations.json`                  | JSON    | No       | Pumping station registry                |
| `system/energy_contracts.json`                  | JSON    | No       | Bilateral energy contract registry      |
| `system/hydro_geometry.parquet`                 | Parquet | No       | Reservoir geometry elevation tables     |
| `system/hydro_production_models.json`           | JSON    | No       | FPHA production function configs        |
| `system/fpha_hyperplanes.parquet`               | Parquet | No       | FPHA hyperplane coefficients            |
| `scenarios/inflow_history.parquet`              | Parquet | No       | Historical inflow time series           |
| `scenarios/inflow_seasonal_stats.parquet`       | Parquet | No       | PAR model seasonal statistics           |
| `scenarios/inflow_ar_coefficients.parquet`      | Parquet | No       | PAR autoregressive coefficients         |
| `scenarios/external_scenarios.parquet`          | Parquet | No       | Pre-generated scenario inflows          |
| `scenarios/load_seasonal_stats.parquet`         | Parquet | No       | Load model seasonal statistics          |
| `scenarios/load_factors.json`                   | JSON    | No       | Load scaling factors per bus/stage      |
| `scenarios/correlation.json`                    | JSON    | No       | Cross-series correlation model          |
| `constraints/thermal_bounds.parquet`            | Parquet | No       | Stage-varying thermal generation bounds |
| `constraints/hydro_bounds.parquet`              | Parquet | No       | Stage-varying hydro operational bounds  |
| `constraints/line_bounds.parquet`               | Parquet | No       | Stage-varying line flow capacity        |
| `constraints/pumping_bounds.parquet`            | Parquet | No       | Stage-varying pumping flow bounds       |
| `constraints/contract_bounds.parquet`           | Parquet | No       | Stage-varying contract power bounds     |
| `constraints/exchange_factors.json`             | JSON    | No       | Block exchange factors                  |
| `constraints/generic_constraints.json`          | JSON    | No       | User-defined LP constraints             |
| `constraints/generic_constraint_bounds.parquet` | Parquet | No       | Generic constraint RHS bounds           |
| `constraints/penalty_overrides_bus.parquet`     | Parquet | No       | Stage-varying bus excess cost           |
| `constraints/penalty_overrides_line.parquet`    | Parquet | No       | Stage-varying line exchange cost        |
| `constraints/penalty_overrides_hydro.parquet`   | Parquet | No       | Stage-varying hydro penalty costs       |
| `constraints/penalty_overrides_ncs.parquet`     | Parquet | No       | Stage-varying NCS curtailment cost      |

---

## Root-level files

### `config.json`

Controls all solver parameters. The `training` section is required; all other
sections are optional and fall back to documented defaults when absent.

**Top-level sections:**

| Section                  | Type   | Default      | Purpose                                        |
| ------------------------ | ------ | ------------ | ---------------------------------------------- |
| `version`                | string | `null`       | Config format version (informational)          |
| `modeling`               | object | `{}`         | Inflow non-negativity treatment                |
| `training`               | object | **required** | Iteration count, stopping rules, cut selection |
| `upper_bound_evaluation` | object | `{}`         | Inner approximation upper-bound settings       |
| `policy`                 | object | fresh mode   | Policy directory path and warm-start mode      |
| `simulation`             | object | disabled     | Post-training simulation settings              |
| `exports`                | object | all enabled  | Output file selection flags                    |

**`modeling` section:**

| Field                                         | Type   | Default     | Description                                                                                                        |
| --------------------------------------------- | ------ | ----------- | ------------------------------------------------------------------------------------------------------------------ |
| `modeling.inflow_non_negativity.method`       | string | `"penalty"` | How to handle negative modelled inflows. One of `"none"`, `"penalty"`, `"truncation"`, `"truncation_with_penalty"` |
| `modeling.inflow_non_negativity.penalty_cost` | number | `1000.0`    | Penalty coefficient when method is `"penalty"` or `"truncation_with_penalty"`                                      |

**`training` section (mandatory fields):**

| Field                      | Type            | Default      | Description                                                                                     |
| -------------------------- | --------------- | ------------ | ----------------------------------------------------------------------------------------------- |
| `training.forward_passes`  | integer         | **required** | Number of scenario trajectories per iteration (>= 1)                                            |
| `training.stopping_rules`  | array           | **required** | At least one stopping rule entry; must include an `iteration_limit` rule                        |
| `training.stopping_mode`   | string          | `"any"`      | How multiple rules combine: `"any"` (stop when any triggers) or `"all"` (stop when all trigger) |
| `training.enabled`         | boolean         | `true`       | When `false`, skip training and proceed directly to simulation                                  |
| `training.seed`            | integer or null | `null`       | Random seed for reproducible scenario generation                                                |
| `training.cut_formulation` | string or null  | `null`       | Cut type: `"single"` or `"multi"`                                                               |

**`training.stopping_rules` entries:**

Each entry has a `"type"` discriminator. Valid types:

| Type              | Required fields                                                       | Stops when                                                               |
| ----------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `iteration_limit` | `limit: integer`                                                      | Iteration count reaches `limit`                                          |
| `time_limit`      | `seconds: number`                                                     | Wall-clock time exceeds `seconds`                                        |
| `bound_stalling`  | `iterations: integer`, `tolerance: number`                            | Lower bound improvement falls below `tolerance` over `iterations` window |
| `simulation`      | `replications`, `period`, `bound_window`, `distance_tol`, `bound_tol` | Both policy cost and bound have stabilized                               |

**`training.cut_selection` sub-section:**

| Field                    | Type    | Default | Description                                             |
| ------------------------ | ------- | ------- | ------------------------------------------------------- |
| `enabled`                | boolean | `null`  | Enable cut pruning                                      |
| `method`                 | string  | `null`  | Pruning method: `"level1"`, `"lml1"`, or `"domination"` |
| `threshold`              | integer | `null`  | Minimum iterations before first pruning pass            |
| `check_frequency`        | integer | `null`  | Iterations between pruning checks                       |
| `cut_activity_tolerance` | number  | `null`  | Minimum dual multiplier for a cut to count as binding   |

**`upper_bound_evaluation` section:**

| Field                      | Type    | Default | Description                                   |
| -------------------------- | ------- | ------- | --------------------------------------------- |
| `enabled`                  | boolean | `null`  | Enable vertex-based inner approximation       |
| `initial_iteration`        | integer | `null`  | First iteration to compute the upper bound    |
| `interval_iterations`      | integer | `null`  | Iterations between upper-bound evaluations    |
| `lipschitz.mode`           | string  | `null`  | Lipschitz constant computation mode: `"auto"` |
| `lipschitz.fallback_value` | number  | `null`  | Fallback when automatic computation fails     |
| `lipschitz.scale_factor`   | number  | `null`  | Multiplicative safety margin                  |

**`policy` section:**

| Field                               | Type    | Default      | Description                                                            |
| ----------------------------------- | ------- | ------------ | ---------------------------------------------------------------------- |
| `path`                              | string  | `"./policy"` | Directory for policy data (cuts, states, vertices, basis)              |
| `mode`                              | string  | `"fresh"`    | Initialization mode: `"fresh"`, `"warm_start"`, or `"resume"`          |
| `validate_compatibility`            | boolean | `true`       | Verify entity and dimension compatibility when loading a stored policy |
| `checkpointing.enabled`             | boolean | `null`       | Enable periodic checkpointing                                          |
| `checkpointing.initial_iteration`   | integer | `null`       | First iteration to write a checkpoint                                  |
| `checkpointing.interval_iterations` | integer | `null`       | Iterations between checkpoints                                         |
| `checkpointing.store_basis`         | boolean | `null`       | Include LP basis in checkpoints                                        |
| `checkpointing.compress`            | boolean | `null`       | Compress checkpoint files                                              |

**`simulation` section:**

| Field                  | Type           | Default       | Description                                                        |
| ---------------------- | -------------- | ------------- | ------------------------------------------------------------------ |
| `enabled`              | boolean        | `false`       | Enable post-training simulation                                    |
| `num_scenarios`        | integer        | `2000`        | Number of simulation scenarios                                     |
| `policy_type`          | string         | `"outer"`     | Policy representation: `"outer"` (cuts) or `"inner"` (vertices)    |
| `output_path`          | string or null | `null`        | Directory for simulation output files                              |
| `output_mode`          | string or null | `null`        | Output mode: `"streaming"` or `"batched"`                          |
| `io_channel_capacity`  | integer        | `64`          | Channel capacity between simulation and I/O writer threads         |
| `sampling_scheme.type` | string         | `"in_sample"` | Scenario scheme: `"in_sample"`, `"out_of_sample"`, or `"external"` |

**`exports` section:**

| Field             | Type           | Default | Description                                        |
| ----------------- | -------------- | ------- | -------------------------------------------------- |
| `training`        | boolean        | `true`  | Export training summary metrics                    |
| `cuts`            | boolean        | `true`  | Export cut pool (outer approximation)              |
| `states`          | boolean        | `true`  | Export visited states                              |
| `vertices`        | boolean        | `true`  | Export inner approximation vertices                |
| `simulation`      | boolean        | `true`  | Export simulation results                          |
| `forward_detail`  | boolean        | `false` | Export per-scenario forward-pass detail            |
| `backward_detail` | boolean        | `false` | Export per-scenario backward-pass detail           |
| `compression`     | string or null | `null`  | Output compression: `"zstd"`, `"lz4"`, or `"none"` |

**Minimal valid example:**

```json
{
  "training": {
    "forward_passes": 192,
    "stopping_rules": [{ "type": "iteration_limit", "limit": 200 }]
  }
}
```

---

### `penalties.json`

Global penalty cost defaults used when no entity-level override is present.
All four sections are required. Every scalar cost must be strictly positive (> 0.0).
Deficit segment costs must be monotonically increasing and the last segment must
have `depth_mw: null` (unbounded).

| Section                   | Field                             | Type           | Description                                                |
| ------------------------- | --------------------------------- | -------------- | ---------------------------------------------------------- |
| `bus`                     | `deficit_segments`                | array          | Piecewise-linear deficit cost tiers                        |
| `bus`                     | `deficit_segments[].depth_mw`     | number or null | Segment depth (MW); `null` for the final unbounded segment |
| `bus`                     | `deficit_segments[].cost`         | number         | Cost per MWh of deficit in this tier (USD/MWh)             |
| `bus`                     | `excess_cost`                     | number         | Cost per MWh of excess injection (USD/MWh)                 |
| `line`                    | `exchange_cost`                   | number         | Cost per MWh of inter-bus exchange flow (USD/MWh)          |
| `hydro`                   | `spillage_cost`                   | number         | Spillage penalty                                           |
| `hydro`                   | `fpha_turbined_cost`              | number         | FPHA turbined flow violation penalty                       |
| `hydro`                   | `diversion_cost`                  | number         | Diversion flow penalty                                     |
| `hydro`                   | `storage_violation_below_cost`    | number         | Storage below-minimum violation penalty                    |
| `hydro`                   | `filling_target_violation_cost`   | number         | Filling target violation penalty                           |
| `hydro`                   | `turbined_violation_below_cost`   | number         | Turbined flow below-minimum violation penalty              |
| `hydro`                   | `outflow_violation_below_cost`    | number         | Total outflow below-minimum violation penalty              |
| `hydro`                   | `outflow_violation_above_cost`    | number         | Total outflow above-maximum violation penalty              |
| `hydro`                   | `generation_violation_below_cost` | number         | Generation below-minimum violation penalty                 |
| `hydro`                   | `evaporation_violation_cost`      | number         | Evaporation violation penalty                              |
| `hydro`                   | `water_withdrawal_violation_cost` | number         | Water withdrawal violation penalty                         |
| `non_controllable_source` | `curtailment_cost`                | number         | Curtailment penalty (USD/MWh)                              |

**Example:**

```json
{
  "bus": {
    "deficit_segments": [
      { "depth_mw": 500.0, "cost": 1000.0 },
      { "depth_mw": null, "cost": 5000.0 }
    ],
    "excess_cost": 100.0
  },
  "line": { "exchange_cost": 2.0 },
  "hydro": {
    "spillage_cost": 0.01,
    "fpha_turbined_cost": 0.05,
    "diversion_cost": 0.1,
    "storage_violation_below_cost": 10000.0,
    "filling_target_violation_cost": 50000.0,
    "turbined_violation_below_cost": 500.0,
    "outflow_violation_below_cost": 500.0,
    "outflow_violation_above_cost": 500.0,
    "generation_violation_below_cost": 1000.0,
    "evaporation_violation_cost": 5000.0,
    "water_withdrawal_violation_cost": 1000.0
  },
  "non_controllable_source": { "curtailment_cost": 0.005 }
}
```

---

### `stages.json`

Defines the temporal structure of the study: stage sequence, block decomposition,
policy graph horizon type, and scenario source configuration.

**Top-level fields:**

| Field                | Required | Description                                                                    |
| -------------------- | -------- | ------------------------------------------------------------------------------ |
| `policy_graph`       | Yes      | Horizon type (`"finite_horizon"`), annual discount rate, and stage transitions |
| `stages`             | Yes      | Array of study stage definitions                                               |
| `scenario_source`    | No       | Top-level sampling scheme and seed                                             |
| `season_definitions` | No       | Season labeling for seasonal model alignment                                   |
| `pre_study_stages`   | No       | Pre-study stages for AR model warm-up (negative IDs)                           |

**`stages[]` entry fields:**

| Field             | Required | Description                                                    |
| ----------------- | -------- | -------------------------------------------------------------- |
| `id`              | Yes      | Stage identifier (non-negative integer, unique)                |
| `start_date`      | Yes      | ISO 8601 date (e.g., `"2024-01-01"`)                           |
| `end_date`        | Yes      | ISO 8601 date; must be after `start_date`                      |
| `blocks`          | Yes      | Array of load blocks (`id`, `name`, `hours`)                   |
| `num_scenarios`   | Yes      | Number of forward-pass scenarios for this stage (>= 1)         |
| `season_id`       | No       | Reference to a season in `season_definitions`                  |
| `block_mode`      | No       | Block execution mode: `"parallel"` (default) or `"sequential"` |
| `state_variables` | No       | Which state variables are active: `storage`, `inflow_lags`     |
| `risk_measure`    | No       | Per-stage risk measure: `"expectation"` or CVaR config         |
| `sampling_method` | No       | Noise method: `"saa"` or other variants                        |

---

### `initial_conditions.json`

Initial reservoir storage values at the start of the study.

| Field             | Required | Description                                                                          |
| ----------------- | -------- | ------------------------------------------------------------------------------------ |
| `storage`         | Yes      | Array of `{ "hydro_id": integer, "value_hm3": number }` entries for operating hydros |
| `filling_storage` | Yes      | Array of `{ "hydro_id": integer, "value_hm3": number }` entries for filling hydros   |

Each `hydro_id` must be unique within its array and must not appear in both arrays.
All `value_hm3` values must be non-negative.

---

## `system/` files

### `system/buses.json`

Electrical bus registry. Buses are the nodes of the transmission network.

| Field                                 | Required | Description                                                                               |
| ------------------------------------- | -------- | ----------------------------------------------------------------------------------------- |
| `buses[].id`                          | Yes      | Bus identifier (integer, unique)                                                          |
| `buses[].name`                        | Yes      | Human-readable bus name (string)                                                          |
| `buses[].deficit_segments`            | No       | Entity-level deficit cost tiers; when absent, global defaults from `penalties.json` apply |
| `buses[].deficit_segments[].depth_mw` | No       | Segment MW depth; `null` for the final unbounded segment                                  |
| `buses[].deficit_segments[].cost`     | No       | Cost per MWh of deficit in this tier (USD/MWh)                                            |

---

### `system/lines.json`

Transmission line registry. Lines connect buses and carry power flows.

| Field                   | Required | Description                                      |
| ----------------------- | -------- | ------------------------------------------------ |
| `lines[].id`            | Yes      | Line identifier (integer, unique)                |
| `lines[].name`          | Yes      | Human-readable line name (string)                |
| `lines[].source_bus_id` | Yes      | Sending-end bus ID                               |
| `lines[].target_bus_id` | Yes      | Receiving-end bus ID                             |
| `lines[].direct_mw`     | Yes      | Maximum power flow in the direct direction (MW)  |
| `lines[].reverse_mw`    | Yes      | Maximum power flow in the reverse direction (MW) |

---

### `system/hydros.json`

Hydro plant registry. Each entry defines a complete hydro plant with reservoir,
turbine, and optional cascade linkage.

Key fields:

| Field                                         | Required           | Description                                                          |
| --------------------------------------------- | ------------------ | -------------------------------------------------------------------- |
| `hydros[].id`                                 | Yes                | Plant identifier (integer, unique)                                   |
| `hydros[].name`                               | Yes                | Human-readable plant name                                            |
| `hydros[].bus_id`                             | Yes                | Bus where generation is injected                                     |
| `hydros[].downstream_id`                      | No                 | Downstream plant ID in the cascade; `null` = tailwater               |
| `hydros[].reservoir`                          | Yes                | `min_storage_hm3` and `max_storage_hm3` (both >= 0)                  |
| `hydros[].outflow`                            | Yes                | `min_outflow_m3s` and `max_outflow_m3s` total outflow bounds         |
| `hydros[].generation`                         | Yes                | Generation model: `model`, turbine flow bounds, generation MW bounds |
| `hydros[].generation.model`                   | Yes                | Currently: `"constant_productivity"`                                 |
| `hydros[].generation.productivity_mw_per_m3s` | Yes (for constant) | Turbine productivity factor                                          |
| `hydros[].penalties`                          | No                 | Entity-level hydro penalty overrides                                 |

---

### `system/thermals.json`

Thermal plant registry. Each entry defines a dispatchable generation unit.

| Field                          | Required | Description                        |
| ------------------------------ | -------- | ---------------------------------- |
| `thermals[].id`                | Yes      | Plant identifier (integer, unique) |
| `thermals[].name`              | Yes      | Human-readable plant name          |
| `thermals[].bus_id`            | Yes      | Bus where generation is injected   |
| `thermals[].min_generation_mw` | Yes      | Minimum dispatch level (MW)        |
| `thermals[].max_generation_mw` | Yes      | Maximum dispatch level (MW)        |
| `thermals[].cost_per_mwh`      | Yes      | Linear generation cost (USD/MWh)   |

---

## `scenarios/` files (Parquet)

### `scenarios/inflow_seasonal_stats.parquet`

PAR(p) model seasonal statistics for each (hydro plant, stage) pair.

| Column     | Type   | Required | Description                                                 |
| ---------- | ------ | -------- | ----------------------------------------------------------- |
| `hydro_id` | INT32  | Yes      | Hydro plant ID                                              |
| `stage_id` | INT32  | Yes      | Stage ID                                                    |
| `mean_m3s` | DOUBLE | Yes      | Seasonal mean inflow (m³/s); must be finite                 |
| `std_m3s`  | DOUBLE | Yes      | Seasonal standard deviation (m³/s); must be >= 0 and finite |
| `ar_order` | INT32  | Yes      | AR model order (number of lags); must be >= 0               |

---

### `scenarios/inflow_ar_coefficients.parquet`

Autoregressive coefficients for the PAR(p) inflow model.

| Column        | Type   | Required | Description                                 |
| ------------- | ------ | -------- | ------------------------------------------- |
| `hydro_id`    | INT32  | Yes      | Hydro plant ID                              |
| `stage_id`    | INT32  | Yes      | Stage ID                                    |
| `lag`         | INT32  | Yes      | Lag index (1-based)                         |
| `coefficient` | DOUBLE | Yes      | AR coefficient for this (hydro, stage, lag) |

---

## `constraints/` files (Parquet)

All bounds Parquet files use sparse storage: only `(entity_id, stage_id)` pairs
that differ from the base entity-level value need rows. Absent rows use the
entity-level value unchanged.

### `constraints/thermal_bounds.parquet`

Stage-varying generation bound overrides for thermal plants.

| Column              | Type   | Required | Description                      |
| ------------------- | ------ | -------- | -------------------------------- |
| `thermal_id`        | INT32  | Yes      | Thermal plant ID                 |
| `stage_id`          | INT32  | Yes      | Stage ID                         |
| `min_generation_mw` | DOUBLE | No       | Minimum generation override (MW) |
| `max_generation_mw` | DOUBLE | No       | Maximum generation override (MW) |

---

### `constraints/hydro_bounds.parquet`

Stage-varying operational bound overrides for hydro plants.

| Column                 | Type   | Required | Description                     |
| ---------------------- | ------ | -------- | ------------------------------- |
| `hydro_id`             | INT32  | Yes      | Hydro plant ID                  |
| `stage_id`             | INT32  | Yes      | Stage ID                        |
| `min_turbined_m3s`     | DOUBLE | No       | Minimum turbined flow (m³/s)    |
| `max_turbined_m3s`     | DOUBLE | No       | Maximum turbined flow (m³/s)    |
| `min_storage_hm3`      | DOUBLE | No       | Minimum reservoir storage (hm³) |
| `max_storage_hm3`      | DOUBLE | No       | Maximum reservoir storage (hm³) |
| `min_outflow_m3s`      | DOUBLE | No       | Minimum total outflow (m³/s)    |
| `max_outflow_m3s`      | DOUBLE | No       | Maximum total outflow (m³/s)    |
| `min_generation_mw`    | DOUBLE | No       | Minimum generation (MW)         |
| `max_generation_mw`    | DOUBLE | No       | Maximum generation (MW)         |
| `max_diversion_m3s`    | DOUBLE | No       | Maximum diversion flow (m³/s)   |
| `filling_inflow_m3s`   | DOUBLE | No       | Filling inflow override (m³/s)  |
| `water_withdrawal_m3s` | DOUBLE | No       | Water withdrawal (m³/s)         |

---

### `constraints/line_bounds.parquet`

Stage-varying flow capacity overrides for transmission lines.

| Column       | Type   | Required | Description                         |
| ------------ | ------ | -------- | ----------------------------------- |
| `line_id`    | INT32  | Yes      | Transmission line ID                |
| `stage_id`   | INT32  | Yes      | Stage ID                            |
| `direct_mw`  | DOUBLE | No       | Direct-flow capacity override (MW)  |
| `reverse_mw` | DOUBLE | No       | Reverse-flow capacity override (MW) |

---

### `constraints/pumping_bounds.parquet`

Stage-varying flow bounds for pumping stations.

| Column       | Type   | Required | Description                 |
| ------------ | ------ | -------- | --------------------------- |
| `station_id` | INT32  | Yes      | Pumping station ID          |
| `stage_id`   | INT32  | Yes      | Stage ID                    |
| `min_m3s`    | DOUBLE | No       | Minimum pumping flow (m³/s) |
| `max_m3s`    | DOUBLE | No       | Maximum pumping flow (m³/s) |

---

### `constraints/contract_bounds.parquet`

Stage-varying power and price overrides for energy contracts.

| Column          | Type   | Required | Description              |
| --------------- | ------ | -------- | ------------------------ |
| `contract_id`   | INT32  | Yes      | Energy contract ID       |
| `stage_id`      | INT32  | Yes      | Stage ID                 |
| `min_mw`        | DOUBLE | No       | Minimum power (MW)       |
| `max_mw`        | DOUBLE | No       | Maximum power (MW)       |
| `price_per_mwh` | DOUBLE | No       | Price override (USD/MWh) |

---

## Penalty override files

All penalty override files use sparse storage. Only rows for `(entity_id, stage_id)`
pairs where the penalty differs from the entity-level or global default are required.
All penalty values must be strictly positive (> 0.0) and finite.

### `constraints/penalty_overrides_bus.parquet`

| Column        | Type   | Required | Description                              |
| ------------- | ------ | -------- | ---------------------------------------- |
| `bus_id`      | INT32  | Yes      | Bus ID                                   |
| `stage_id`    | INT32  | Yes      | Stage ID                                 |
| `excess_cost` | DOUBLE | No       | Excess injection cost override (USD/MWh) |

Note: Bus deficit segments are not stage-varying. Only `excess_cost` can be
overridden per stage for buses.

---

### `constraints/penalty_overrides_line.parquet`

| Column          | Type   | Required | Description                           |
| --------------- | ------ | -------- | ------------------------------------- |
| `line_id`       | INT32  | Yes      | Transmission line ID                  |
| `stage_id`      | INT32  | Yes      | Stage ID                              |
| `exchange_cost` | DOUBLE | No       | Exchange flow cost override (USD/MWh) |

---

### `constraints/penalty_overrides_hydro.parquet`

| Column                            | Type   | Required | Description                                 |
| --------------------------------- | ------ | -------- | ------------------------------------------- |
| `hydro_id`                        | INT32  | Yes      | Hydro plant ID                              |
| `stage_id`                        | INT32  | Yes      | Stage ID                                    |
| `spillage_cost`                   | DOUBLE | No       | Spillage penalty override                   |
| `fpha_turbined_cost`              | DOUBLE | No       | FPHA turbined flow violation override       |
| `diversion_cost`                  | DOUBLE | No       | Diversion penalty override                  |
| `storage_violation_below_cost`    | DOUBLE | No       | Storage below-minimum violation override    |
| `filling_target_violation_cost`   | DOUBLE | No       | Filling target violation override           |
| `turbined_violation_below_cost`   | DOUBLE | No       | Turbined below-minimum violation override   |
| `outflow_violation_below_cost`    | DOUBLE | No       | Outflow below-minimum violation override    |
| `outflow_violation_above_cost`    | DOUBLE | No       | Outflow above-maximum violation override    |
| `generation_violation_below_cost` | DOUBLE | No       | Generation below-minimum violation override |
| `evaporation_violation_cost`      | DOUBLE | No       | Evaporation violation override              |
| `water_withdrawal_violation_cost` | DOUBLE | No       | Water withdrawal violation override         |

---

### `constraints/penalty_overrides_ncs.parquet`

| Column             | Type   | Required | Description                            |
| ------------------ | ------ | -------- | -------------------------------------- |
| `source_id`        | INT32  | Yes      | Non-controllable source ID             |
| `stage_id`         | INT32  | Yes      | Stage ID                               |
| `curtailment_cost` | DOUBLE | No       | Curtailment penalty override (USD/MWh) |
