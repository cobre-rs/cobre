# Case Format Reference

A Cobre case directory is a self-contained folder that holds all input data
for a single power system study. `load_case` reads this directory and produces
a fully-validated `System` ready for the solver.

For a description of how these files are parsed and validated, see
[cobre-io](../crates/io.md).

> JSON Schema files for all JSON input types are available on the
> [Schemas](./schemas.md) page. Download them for use with your editor's JSON
> Schema validation feature.

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
│   ├── non_controllable_factors.json        # NCS block scaling factors (optional)
│   ├── non_controllable_stats.parquet      # NCS stochastic availability (optional)
│   ├── correlation.json                     # Cross-series correlation model (optional)
│   └── noise_openings.parquet              # User-supplied backward-pass opening tree (optional)
└── constraints/
    ├── thermal_bounds.parquet               # Stage-varying thermal bounds (optional)
    ├── hydro_bounds.parquet                 # Stage-varying hydro bounds (optional)
    ├── line_bounds.parquet                  # Stage-varying line bounds (optional)
    ├── pumping_bounds.parquet               # Stage-varying pumping bounds (optional)
    ├── contract_bounds.parquet              # Stage-varying contract bounds (optional)
    ├── ncs_bounds.parquet                   # Stage-varying NCS available generation bounds (optional)
    ├── exchange_factors.json                # Block exchange factors (optional)
    ├── generic_constraints.json             # User-defined LP constraints (optional)
    ├── generic_constraint_bounds.parquet    # Bounds for generic constraints (optional)
    ├── penalty_overrides_bus.parquet        # Stage-varying bus penalty overrides (optional)
    ├── penalty_overrides_line.parquet       # Stage-varying line penalty overrides (optional)
    ├── penalty_overrides_hydro.parquet      # Stage-varying hydro penalty overrides (optional)
    └── penalty_overrides_ncs.parquet        # Stage-varying NCS penalty overrides (optional)
```

## File summary

| File                                            | Format  | Required | Description                                   |
| ----------------------------------------------- | ------- | -------- | --------------------------------------------- |
| `config.json`                                   | JSON    | Yes      | Solver configuration                          |
| `penalties.json`                                | JSON    | Yes      | Global penalty defaults                       |
| `stages.json`                                   | JSON    | Yes      | Stage sequence and policy graph               |
| `initial_conditions.json`                       | JSON    | Yes      | Initial reservoir storage                     |
| `system/buses.json`                             | JSON    | Yes      | Electrical bus registry                       |
| `system/lines.json`                             | JSON    | Yes      | Transmission line registry                    |
| `system/hydros.json`                            | JSON    | Yes      | Hydro plant registry                          |
| `system/thermals.json`                          | JSON    | Yes      | Thermal plant registry                        |
| `system/non_controllable_sources.json`          | JSON    | No       | Intermittent source registry                  |
| `system/pumping_stations.json`                  | JSON    | No       | Pumping station registry                      |
| `system/energy_contracts.json`                  | JSON    | No       | Bilateral energy contract registry            |
| `system/hydro_geometry.parquet`                 | Parquet | No       | Reservoir geometry elevation tables           |
| `system/hydro_production_models.json`           | JSON    | No       | FPHA production function configs              |
| `system/fpha_hyperplanes.parquet`               | Parquet | No       | FPHA hyperplane coefficients                  |
| `scenarios/inflow_history.parquet`              | Parquet | No       | Historical inflow time series                 |
| `scenarios/inflow_seasonal_stats.parquet`       | Parquet | No       | PAR model seasonal statistics                 |
| `scenarios/inflow_ar_coefficients.parquet`      | Parquet | No       | PAR autoregressive coefficients               |
| `scenarios/external_scenarios.parquet`          | Parquet | No       | Pre-generated scenario inflows                |
| `scenarios/load_seasonal_stats.parquet`         | Parquet | No       | Load model seasonal statistics                |
| `scenarios/load_factors.json`                   | JSON    | No       | Load scaling factors per bus/stage            |
| `scenarios/non_controllable_factors.json`       | JSON    | No       | NCS block scaling factors per source/stage    |
| `scenarios/non_controllable_stats.parquet`      | Parquet | No       | NCS stochastic availability factors           |
| `scenarios/correlation.json`                    | JSON    | No       | Cross-series correlation model                |
| `scenarios/noise_openings.parquet`              | Parquet | No       | User-supplied backward-pass opening tree      |
| `constraints/thermal_bounds.parquet`            | Parquet | No       | Stage-varying thermal generation bounds       |
| `constraints/hydro_bounds.parquet`              | Parquet | No       | Stage-varying hydro operational bounds        |
| `constraints/line_bounds.parquet`               | Parquet | No       | Stage-varying line flow capacity              |
| `constraints/pumping_bounds.parquet`            | Parquet | No       | Stage-varying pumping flow bounds             |
| `constraints/contract_bounds.parquet`           | Parquet | No       | Stage-varying contract power bounds           |
| `constraints/ncs_bounds.parquet`                | Parquet | No       | Stage-varying NCS available generation bounds |
| `constraints/exchange_factors.json`             | JSON    | No       | Block exchange factors                        |
| `constraints/generic_constraints.json`          | JSON    | No       | User-defined LP constraints                   |
| `constraints/generic_constraint_bounds.parquet` | Parquet | No       | Generic constraint RHS bounds                 |
| `constraints/penalty_overrides_bus.parquet`     | Parquet | No       | Stage-varying bus excess cost                 |
| `constraints/penalty_overrides_line.parquet`    | Parquet | No       | Stage-varying line exchange cost              |
| `constraints/penalty_overrides_hydro.parquet`   | Parquet | No       | Stage-varying hydro penalty costs             |
| `constraints/penalty_overrides_ncs.parquet`     | Parquet | No       | Stage-varying NCS curtailment cost            |

---

## Root-level files

### `config.json`

Controls all solver parameters. The `training` section is required; all other
sections are optional and fall back to documented defaults when absent.

**Top-level sections:**

| Section                  | Type   | Default      | Purpose                                                           |
| ------------------------ | ------ | ------------ | ----------------------------------------------------------------- |
| `$schema`                | string | `null`       | JSON Schema URI for editor validation (ignored during processing) |
| `modeling`               | object | `{}`         | Inflow non-negativity treatment                                   |
| `training`               | object | **required** | Iteration count, stopping rules, cut selection                    |
| `upper_bound_evaluation` | object | `{}`         | Inner approximation upper-bound settings                          |
| `policy`                 | object | fresh mode   | Policy directory path and warm-start mode                         |
| `simulation`             | object | disabled     | Post-training simulation settings                                 |
| `exports`                | object | all enabled  | Output file selection flags                                       |

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
| `training.seed`            | integer or null | `null`       | Random seed for reproducible noise generation (see [Seed resolution](#seed-resolution))         |
| `training.cut_formulation` | string or null  | `null`       | Cut type: `"single"` or `"multi"`                                                               |

#### Seed resolution

`training.seed` in `config.json` is the **only** seed that controls noise generation
at runtime. It governs both the training forward pass and the post-training simulation.

- When `training.seed` is a non-null integer, the CLI uses `|seed|` (unsigned absolute
  value) as the base seed for deterministic SipHash-1-3 noise generation. Results are
  bit-for-bit reproducible across runs with the same seed.
- When `training.seed` is absent or `null`, the CLI applies a **default seed of 42**
  and prints a warning to stderr:

  ```
  warning: no random seed specified in config.json (training.seed); using default seed 42. Set training.seed for reproducible results.
  ```

  Runs will be reproducible (same output every time) but the seed value is arbitrary.
  Set `training.seed` explicitly to make the choice intentional and visible to other
  users of the case directory.

`scenario_source.seed` in `stages.json` is a separate field that is loaded and stored
in the `System` but is **not used at runtime** for training or simulation noise.
It is reserved for future out-of-sample and external sampling schemes. Do not rely on
it to control reproducibility.

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

| Field                  | Type           | Default       | Description                                                     |
| ---------------------- | -------------- | ------------- | --------------------------------------------------------------- |
| `enabled`              | boolean        | `false`       | Enable post-training simulation                                 |
| `num_scenarios`        | integer        | `2000`        | Number of simulation scenarios                                  |
| `policy_type`          | string         | `"outer"`     | Policy representation: `"outer"` (cuts) or `"inner"` (vertices) |
| `output_path`          | string or null | `null`        | Directory for simulation output files                           |
| `output_mode`          | string or null | `null`        | Output mode: `"streaming"` or `"batched"`                       |
| `io_channel_capacity`  | integer        | `64`          | Channel capacity between simulation and I/O writer threads      |
| `sampling_scheme.type` | string         | `"in_sample"` | Scenario scheme: `"in_sample"`, `"external"`, or `"historical"` |

**`exports` section:**

| Field             | Type           | Default | Description                                                        |
| ----------------- | -------------- | ------- | ------------------------------------------------------------------ |
| `training`        | boolean        | `true`  | Export training summary metrics                                    |
| `cuts`            | boolean        | `true`  | Export cut pool (outer approximation)                              |
| `states`          | boolean        | `true`  | Export visited states                                              |
| `vertices`        | boolean        | `true`  | Export inner approximation vertices                                |
| `simulation`      | boolean        | `true`  | Export simulation results                                          |
| `forward_detail`  | boolean        | `false` | Export per-scenario forward-pass detail                            |
| `backward_detail` | boolean        | `false` | Export per-scenario backward-pass detail                           |
| `compression`     | string or null | `null`  | Output compression: `"zstd"`, `"lz4"`, or `"none"`                 |
| `stochastic`      | boolean        | `false` | Export stochastic preprocessing artifacts to `output/stochastic/`. |

**Minimal valid example:**

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/config.schema.json",
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
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/penalties.schema.json",
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

| Field             | Required | Description                                                       |
| ----------------- | -------- | ----------------------------------------------------------------- |
| `id`              | Yes      | Stage identifier (non-negative integer, unique)                   |
| `start_date`      | Yes      | ISO 8601 date (e.g., `"2024-01-01"`)                              |
| `end_date`        | Yes      | ISO 8601 date; must be after `start_date`                         |
| `blocks`          | Yes      | Array of load blocks (`id`, `name`, `hours`)                      |
| `num_scenarios`   | Yes      | Number of forward-pass scenarios for this stage (>= 1)            |
| `season_id`       | No       | Reference to a season in `season_definitions`                     |
| `block_mode`      | No       | Block execution mode: `"parallel"` (default) or `"chronological"` |
| `state_variables` | No       | Which state variables are active: `storage`, `inflow_lags`        |
| `risk_measure`    | No       | Per-stage risk measure: `"expectation"` or CVaR config            |
| `sampling_method` | No       | Noise method: `"saa"` or other variants                           |

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

| Field                         | Required | Description                                                          |
| ----------------------------- | -------- | -------------------------------------------------------------------- |
| `lines[].id`                  | Yes      | Line identifier (integer, unique)                                    |
| `lines[].name`                | Yes      | Human-readable line name (string)                                    |
| `lines[].source_bus_id`       | Yes      | Sending-end bus ID                                                   |
| `lines[].target_bus_id`       | Yes      | Receiving-end bus ID                                                 |
| `lines[].entry_stage_id`      | No       | Stage when line enters service; `null` = always exists               |
| `lines[].exit_stage_id`       | No       | Stage when line is decommissioned; `null` = never                    |
| `lines[].capacity.direct_mw`  | Yes      | Maximum power flow in the direct direction (MW)                      |
| `lines[].capacity.reverse_mw` | Yes      | Maximum power flow in the reverse direction (MW)                     |
| `lines[].exchange_cost`       | No       | Entity-level exchange cost override ($/MWh); absent = global default |
| `lines[].losses_percent`      | No       | Transmission losses as percentage (default: 0.0)                     |

---

### `system/hydros.json`

Hydro plant registry. Each entry defines a complete hydro plant with reservoir,
turbine, and optional cascade linkage.

Key fields:

| Field                                         | Required           | Description                                                                            |
| --------------------------------------------- | ------------------ | -------------------------------------------------------------------------------------- |
| `hydros[].id`                                 | Yes                | Plant identifier (integer, unique)                                                     |
| `hydros[].name`                               | Yes                | Human-readable plant name                                                              |
| `hydros[].bus_id`                             | Yes                | Bus where generation is injected                                                       |
| `hydros[].downstream_id`                      | No                 | Downstream plant ID in the cascade; `null` = tailwater                                 |
| `hydros[].entry_stage_id`                     | No                 | Stage when plant enters service; `null` = always exists                                |
| `hydros[].exit_stage_id`                      | No                 | Stage when plant is decommissioned; `null` = never                                     |
| `hydros[].reservoir`                          | Yes                | `min_storage_hm3` and `max_storage_hm3` (both >= 0)                                    |
| `hydros[].outflow`                            | Yes                | `min_outflow_m3s` and `max_outflow_m3s` total outflow bounds                           |
| `hydros[].generation`                         | Yes                | Generation model: `model`, turbine flow bounds, generation MW bounds                   |
| `hydros[].generation.model`                   | Yes                | `"constant_productivity"`, `"linearized_head"`, or `"fpha"`                            |
| `hydros[].generation.productivity_mw_per_m3s` | Yes (for non-fpha) | Turbine productivity factor [MW/(m³/s)]                                                |
| `hydros[].tailrace`                           | No                 | Tailrace model: `"polynomial"` or `"piecewise"`                                        |
| `hydros[].hydraulic_losses`                   | No                 | Head loss model: `"factor"` or `"constant"`                                            |
| `hydros[].efficiency`                         | No                 | Turbine efficiency model: `"constant"`                                                 |
| `hydros[].evaporation`                        | No                 | Evaporation config: `coefficients_mm` (12 values) and optional `reference_volumes_hm3` |
| `hydros[].diversion`                          | No                 | Diversion channel: `downstream_id` and `max_flow_m3s`                                  |
| `hydros[].filling`                            | No                 | Filling config: `start_stage_id` and `filling_inflow_m3s`                              |
| `hydros[].penalties`                          | No                 | Entity-level hydro penalty overrides (all 11 fields optional, fall back to global)     |

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

### `system/hydro_geometry.parquet`

Volume-Height-Area (VHA) curves for hydro reservoirs. Required when any hydro is
configured with a computed FPHA production model (`source: "computed"`) or with
evaporation linearization. When absent, FPHA computation and evaporation
linearization are unavailable for all plants.

4 columns, all non-nullable. Rows are sorted by `(hydro_id, volume_hm3)` ascending.
Multiple rows per `hydro_id` together constitute the VHA curve for that plant.

| Column       | Type   | Required | Description                                                              |
| ------------ | ------ | -------- | ------------------------------------------------------------------------ |
| `hydro_id`   | INT32  | Yes      | Hydro plant ID                                                           |
| `volume_hm3` | DOUBLE | Yes      | Total reservoir volume at this point (hm³). Non-negative and finite.     |
| `height_m`   | DOUBLE | Yes      | Reservoir surface elevation at this volume (m). Non-negative and finite. |
| `area_km2`   | DOUBLE | Yes      | Water surface area at this volume (km²). Non-negative and finite.        |

**Validation:** all four columns must be present with the correct types. `volume_hm3`,
`height_m`, and `area_km2` must be non-negative and finite. Monotonicity of
`volume_hm3` within each hydro is enforced during Layer 5 semantic validation.

---

### `system/hydro_production_models.json`

Per-hydro production function assignment. When absent, all hydros use
`constant_productivity` (the productivity factor from `hydros.json`) for
every stage.

The file contains a `"production_models"` array. Each entry configures one hydro
plant and is identified by a unique `hydro_id`. Results are loaded in
`hydro_id`-ascending order regardless of declaration order.

**Top-level structure:**

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/production_models.schema.json",
  "production_models": [ ... ]
}
```

**Per-hydro entry fields:**

| Field            | Required | Description                                                                 |
| ---------------- | -------- | --------------------------------------------------------------------------- |
| `hydro_id`       | Yes      | Hydro plant ID. Must be unique within the file.                             |
| `selection_mode` | Yes      | How the model variant is chosen per stage: `"stage_ranges"` or `"seasonal"` |

**`stage_ranges` mode.** The model for each stage is determined by the first
matching `[start_stage_id, end_stage_id]` range. `end_stage_id` may be `null`
to mean "until end of horizon".

| Field within each range | Required | Description                                                                                                           |
| ----------------------- | -------- | --------------------------------------------------------------------------------------------------------------------- |
| `start_stage_id`        | Yes      | First stage (inclusive) to which this entry applies                                                                   |
| `end_stage_id`          | Yes      | Last stage (inclusive); `null` means open-ended                                                                       |
| `model`                 | Yes      | Model name: `"constant_productivity"`, `"linearized_head"`, or `"fpha"`                                               |
| `fpha_config`           | No       | Required when `model` is `"fpha"`. See FPHA config fields below.                                                      |
| `productivity_override` | No       | Replaces base `productivity_mw_per_m3s` for this range. Must be > 0. Rejected on `"fpha"`. Default: entity base value |

**`seasonal` mode.** The model for a stage is determined by its `season_id`.
Stages whose season is not listed use `default_model`.

| Field           | Required | Description                                                                                      |
| --------------- | -------- | ------------------------------------------------------------------------------------------------ |
| `default_model` | Yes      | Fallback model name for unlisted seasons                                                         |
| `seasons`       | Yes      | Array of season overrides: `season_id`, `model`, optional `fpha_config`, `productivity_override` |

**`fpha_config` fields (required when `model` is `"fpha"`):**

| Field                            | Required | Default        | Description                                                   |
| -------------------------------- | -------- | -------------- | ------------------------------------------------------------- |
| `source`                         | Yes      | —              | `"precomputed"` or `"computed"`                               |
| `volume_discretization_points`   | No       | solver default | Number of volume grid points for hyperplane computation       |
| `turbine_discretization_points`  | No       | solver default | Number of turbine-flow grid points for hyperplane computation |
| `spillage_discretization_points` | No       | solver default | Number of spillage grid points for hyperplane computation     |
| `max_planes_per_hydro`           | No       | solver default | Maximum hyperplanes per plant after selection heuristic       |
| `fitting_window`                 | No       | full range     | Volume range restriction for hyperplane computation           |

`source: "precomputed"` means the hyperplanes are loaded from
`system/fpha_hyperplanes.parquet`. `source: "computed"` means Cobre derives
them from `system/hydro_geometry.parquet`; in this case `hydro_geometry.parquet`
must be present and the computed planes are automatically written to
`output/hydro_models/fpha_hyperplanes.parquet`.

**`fitting_window` fields.** Absolute bounds (`volume_min_hm3`, `volume_max_hm3`)
and percentile bounds (`volume_min_percentile`, `volume_max_percentile`) are
mutually exclusive — set one pair or the other, not both.

| Field                   | Type   | Description                                          |
| ----------------------- | ------ | ---------------------------------------------------- |
| `volume_min_hm3`        | number | Explicit minimum volume for fitting (hm³)            |
| `volume_max_hm3`        | number | Explicit maximum volume for fitting (hm³)            |
| `volume_min_percentile` | number | Minimum as a percentile of the operating range (0–1) |
| `volume_max_percentile` | number | Maximum as a percentile of the operating range (0–1) |

**Example — hydro 0 uses computed FPHA for stages 0–24, then constant productivity:**

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/production_models.schema.json",
  "production_models": [
    {
      "hydro_id": 0,
      "selection_mode": "stage_ranges",
      "stage_ranges": [
        {
          "start_stage_id": 0,
          "end_stage_id": 24,
          "model": "fpha",
          "fpha_config": {
            "source": "computed",
            "volume_discretization_points": 7,
            "turbine_discretization_points": 15
          }
        },
        {
          "start_stage_id": 25,
          "end_stage_id": null,
          "model": "constant_productivity",
          "productivity_override": 0.72
        }
      ]
    }
  ]
}
```

**Example — hydro 5 uses FPHA in season 0, linearized_head in all other seasons:**

```json
{
  "production_models": [
    {
      "hydro_id": 5,
      "selection_mode": "seasonal",
      "default_model": "linearized_head",
      "seasons": [
        {
          "season_id": 0,
          "model": "fpha",
          "fpha_config": { "source": "precomputed" }
        }
      ]
    }
  ]
}
```

---

### `system/fpha_hyperplanes.parquet`

Pre-computed FPHA hyperplane coefficients for hydros configured with
`fpha_config.source: "precomputed"`. When absent, only `"computed"` source is
available.

11 columns. Rows are sorted by `(hydro_id, stage_id, plane_id)` ascending.
Null `stage_id` sorts before any non-null stage and means the plane is valid for
all stages of that hydro. One row per hyperplane; at least 3 planes are required
per `(hydro_id, stage_id)` group.

| Column            | Type   | Nullable | Description                                                 |
| ----------------- | ------ | -------- | ----------------------------------------------------------- |
| `hydro_id`        | INT32  | No       | Hydro plant ID                                              |
| `stage_id`        | INT32  | Yes      | Stage the plane applies to. `null` = valid for all stages   |
| `plane_id`        | INT32  | No       | Plane index within this hydro (and stage)                   |
| `gamma_0`         | DOUBLE | No       | Intercept coefficient (MW)                                  |
| `gamma_v`         | DOUBLE | No       | Volume coefficient (MW/hm³). Positive.                      |
| `gamma_q`         | DOUBLE | No       | Turbined flow coefficient (MW per m³/s)                     |
| `gamma_s`         | DOUBLE | No       | Spillage coefficient (MW per m³/s). Typically non-positive. |
| `kappa`           | DOUBLE | Yes      | Correction factor. Defaults to `1.0` when absent or null.   |
| `valid_v_min_hm3` | DOUBLE | Yes      | Volume range minimum where this plane is valid (hm³)        |
| `valid_v_max_hm3` | DOUBLE | Yes      | Volume range maximum where this plane is valid (hm³)        |
| `valid_q_max_m3s` | DOUBLE | Yes      | Maximum turbined flow where this plane is valid (m³/s)      |

**Validation:** required columns (`hydro_id`, `plane_id`, `gamma_0`, `gamma_v`,
`gamma_q`, `gamma_s`) must be present with the correct types. Optional columns
that are present must also have the correct types. Minimum planes per
`(hydro_id, stage_id)` group and sign constraints on `gamma_v` and `gamma_s`
are enforced during Layer 5 semantic validation.

The file produced by `output/hydro_models/fpha_hyperplanes.parquet` (written when
`source: "computed"` is used) has this exact same 11-column schema and is
suitable for use as a future precomputed input.

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

### `scenarios/noise_openings.parquet`

User-supplied backward-pass opening tree. When present, Cobre loads the opening
tree directly from this file instead of generating it internally via
`generate_opening_tree()`. This enables cross-tool comparison, sensitivity
analysis, and round-trip replay of a previously exported opening tree.

| Column          | Type   | Required | Description                                                                   |
| --------------- | ------ | -------- | ----------------------------------------------------------------------------- |
| `stage_id`      | INT32  | Yes      | Zero-based stage index (0 to n_stages − 1)                                    |
| `opening_index` | UINT32 | Yes      | Zero-based opening index within the stage (0 to openings_per_stage − 1)       |
| `entity_index`  | UINT32 | Yes      | Zero-based entity index in system dimension order (see entity ordering below) |
| `value`         | DOUBLE | Yes      | Noise realization for this (stage, opening, entity) triple                    |

**Entity ordering.** The `entity_index` column follows the system dimension
convention: hydro entities first (sorted by canonical ID), then load buses
(sorted by canonical ID), matching the ordering used by the internal opening
tree generator. Violating this convention causes silent value misassignment
because the file stores indices only, not entity identifiers.

**Validation rules.** The loader checks three conditions and raises a hard error
on failure:

- **Dimension mismatch** — the number of distinct `entity_index` values must
  equal `n_hydros + n_load_buses`.
- **Stage count mismatch** — the number of distinct `stage_id` values must
  equal the configured number of study stages.
- **Missing opening indices** — for each stage, every opening index from 0 to
  `openings_per_stage − 1` must be present for every entity. Gaps are not
  permitted; partial-stage override is not supported in v0.1.x.

The total row count must equal `n_stages × openings_per_stage × (n_hydros +
n_load_buses)`.

See the `noise_openings.rs` module for the full schema and validation
rules, and [User-Supplied Opening Trees](../guide/stochastic-modeling.md#user-supplied-opening-trees)
in the Stochastic Modeling guide for usage instructions.

---

## `scenarios/` files (JSON)

### `scenarios/load_factors.json`

Per-bus, per-stage, per-block load scaling factors. When present, each factor
multiplies the stochastic load demand realization at the specified bus for the
specified block. This allows you to model time-of-day or seasonal patterns in
load shape without changing the underlying statistical model.

When this file is absent, all load factors default to 1.0. When a
`(bus_id, stage_id)` pair is absent from the file, its factors also default
to 1.0 for every block.

**JSON structure:**

```json
{
  "load_factors": [
    {
      "bus_id": 0,
      "stage_id": 0,
      "block_factors": [
        { "block_id": 0, "factor": 0.8 },
        { "block_id": 1, "factor": 1.2 }
      ]
    }
  ]
}
```

**Fields per entry:**

| Field           | Type    | Description                                                        |
| --------------- | ------- | ------------------------------------------------------------------ |
| `bus_id`        | integer | Bus entity ID. Must refer to a bus defined in `system/buses.json`. |
| `stage_id`      | integer | Study stage index. Must be a valid stage ID from `stages.json`.    |
| `block_factors` | array   | Array of `{ block_id, factor }` pairs for each load block.         |

**`block_factors` entry fields:**

| Field      | Type    | Constraints                     | Description                                                                       |
| ---------- | ------- | ------------------------------- | --------------------------------------------------------------------------------- |
| `block_id` | integer | Must be a valid block for stage | Zero-based block index within the stage.                                          |
| `factor`   | number  | > 0, finite                     | Multiplier applied to the stochastic load realization (MW) at this bus and block. |

**Effect:** `load_rhs = mean_mw * stochastic_noise_factor * block_factor`.
A factor of 1.0 leaves the load unchanged. Values less than 1.0 reduce load;
values greater than 1.0 increase it.

---

### `scenarios/non_controllable_factors.json`

Per-NCS, per-stage, per-block scaling factors for non-controllable source (NCS)
available generation. When present, each factor multiplies the available
generation bound from `constraints/ncs_bounds.parquet` for the specified block.
This allows modeling of intra-stage availability patterns such as diurnal solar
irradiance profiles or wind speed variations across load blocks.

When this file is absent, all NCS block factors default to 1.0. When a
`(ncs_id, stage_id)` pair is absent from the file, its factors default to 1.0
for every block.

**JSON structure:**

```json
{
  "non_controllable_factors": [
    {
      "ncs_id": 0,
      "stage_id": 0,
      "block_factors": [
        { "block_id": 0, "factor": 0.3 },
        { "block_id": 1, "factor": 0.8 }
      ]
    }
  ]
}
```

**Fields per entry:**

| Field           | Type    | Description                                                                      |
| --------------- | ------- | -------------------------------------------------------------------------------- |
| `ncs_id`        | integer | NCS entity ID. Must refer to a source in `system/non_controllable_sources.json`. |
| `stage_id`      | integer | Study stage index. Must be a valid stage ID from `stages.json`.                  |
| `block_factors` | array   | Array of `{ block_id, factor }` pairs for each load block.                       |

**`block_factors` entry fields:**

| Field      | Type    | Constraints                     | Description                                                                |
| ---------- | ------- | ------------------------------- | -------------------------------------------------------------------------- |
| `block_id` | integer | Must be a valid block for stage | Zero-based block index within the stage.                                   |
| `factor`   | number  | >= 0, finite                    | Multiplier applied to the stage available generation bound for this block. |

**Effect:** `available_mw_block = available_generation_mw * block_factor`.
A factor of 1.0 leaves the bound unchanged. A factor of 0.0 sets availability
to zero for that block (complete generation unavailability).

---

### `scenarios/non_controllable_stats.parquet`

Per-NCS, per-stage stochastic availability model. Each row provides the mean
and standard deviation of the availability factor for one NCS entity at one
stage. The noise transform produces: `A_r = max_gen × clamp(mean + std × η, 0, 1)`.

| Column     | Type   | Required | Description                                      |
| ---------- | ------ | -------- | ------------------------------------------------ |
| `ncs_id`   | INT32  | Yes      | Non-controllable source ID                       |
| `stage_id` | INT32  | Yes      | Stage ID (0-based)                               |
| `mean`     | DOUBLE | Yes      | Mean availability factor in [0, 1]               |
| `std`      | DOUBLE | Yes      | Standard deviation of availability factor (>= 0) |

When absent, NCS availability is deterministic from `constraints/ncs_bounds.parquet`
or the entity's `max_generation_mw`.

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

### `constraints/ncs_bounds.parquet`

Stage-varying available generation bounds for non-controllable sources. Uses
sparse storage: only `(ncs_id, stage_id)` pairs that differ from the base
entity-level value need rows. Absent rows keep the entity's declared
`available_generation_mw` unchanged.

| Column                    | Type   | Required | Description                                                     |
| ------------------------- | ------ | -------- | --------------------------------------------------------------- |
| `ncs_id`                  | INT32  | Yes      | Non-controllable source ID                                      |
| `stage_id`                | INT32  | Yes      | Stage ID                                                        |
| `available_generation_mw` | DOUBLE | Yes      | Maximum available generation for this stage (MW). Must be >= 0. |

The per-block available generation bound in the LP is:
`available_mw_block = available_generation_mw * block_factor`, where
`block_factor` comes from `scenarios/non_controllable_factors.json`
(default 1.0 when absent).

---

### `constraints/exchange_factors.json`

Per-line, per-stage, per-block scaling factors for transmission line capacity
bounds. When present, each factor multiplies the line's direct or reverse
capacity for the specified block. This allows modeling of planned outages,
seasonal de-rating, or time-of-day capacity constraints without replacing the
base entity bounds.

When this file is absent, all exchange factors default to (1.0, 1.0). When a
`(line_id, stage_id)` pair is absent, its factors default to (1.0, 1.0) for
every block.

**JSON structure:**

```json
{
  "exchange_factors": [
    {
      "line_id": 0,
      "stage_id": 0,
      "block_factors": [
        { "block_id": 0, "direct_factor": 0.9, "reverse_factor": 1.0 }
      ]
    }
  ]
}
```

**Fields per entry:**

| Field           | Type    | Description                                                          |
| --------------- | ------- | -------------------------------------------------------------------- |
| `line_id`       | integer | Line entity ID. Must refer to a line defined in `system/lines.json`. |
| `stage_id`      | integer | Study stage index. Must be a valid stage ID from `stages.json`.      |
| `block_factors` | array   | Array of `{ block_id, direct_factor, reverse_factor }` pairs.        |

**`block_factors` entry fields:**

| Field            | Type    | Constraints                     | Description                                                        |
| ---------------- | ------- | ------------------------------- | ------------------------------------------------------------------ |
| `block_id`       | integer | Must be a valid block for stage | Zero-based block index within the stage.                           |
| `direct_factor`  | number  | >= 0, finite                    | Multiplier for the direct-direction flow capacity (`direct_mw`).   |
| `reverse_factor` | number  | >= 0, finite                    | Multiplier for the reverse-direction flow capacity (`reverse_mw`). |

**Effect:** `col_upper_fwd = direct_mw * direct_factor`,
`col_upper_rev = reverse_mw * reverse_factor`. A factor of 1.0 leaves the
capacity unchanged. A factor of 0.0 fully blocks flow in that direction for
the block.

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
