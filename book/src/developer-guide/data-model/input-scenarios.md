# Input Scenarios and Time Series

## Purpose

This spec defines the temporal structure (stages, seasons, blocks), policy graph (transitions, discounting), stochastic scenario pipeline (inflow history, uncertainty models, scenario sources), block-level scaling factors, and spatial correlation inputs. These inputs control how Cobre decomposes time, generates or consumes scenarios, and correlates random variables across the system.

For initial conditions (which bootstrap the stochastic process), see [Input Constraints](input-constraints.md) §1. For the PAR inflow model mathematics, see [PAR Inflow Model](../math/par-inflow-model.md). For discount rate mathematics, see [Discount Rate Formulation](../math/discount-rate.md).

## 1. Stage Definitions (`stages.json`)

> **Format Rationale — stages.json**
>
> **Complex nested object** — Stage definitions with nested block structures, season definitions, policy graph, scenario configuration, and risk parameters. JSON handles hierarchical config naturally.

> **Order Invariance**: The order of stages and blocks in their arrays does NOT affect results. After loading, stages are sorted by `id`, and blocks within each stage are sorted by `id`. See [Design Principles §3](../overview/design-principles.md).

### 1.1 Season Definitions

The `season_definitions` section formally maps each `season_id` to a calendar period. This mapping is required whenever the system needs to aggregate inflow history into season-level values (see §2).

```json
{
  "season_definitions": {
    "cycle_type": "monthly",
    "seasons": [
      { "id": 0, "month_start": 1, "label": "January" },
      { "id": 1, "month_start": 2, "label": "February" },
      { "id": 2, "month_start": 3, "label": "March" },
      { "id": 3, "month_start": 4, "label": "April" },
      { "id": 4, "month_start": 5, "label": "May" },
      { "id": 5, "month_start": 6, "label": "June" },
      { "id": 6, "month_start": 7, "label": "July" },
      { "id": 7, "month_start": 8, "label": "August" },
      { "id": 8, "month_start": 9, "label": "September" },
      { "id": 9, "month_start": 10, "label": "October" },
      { "id": 10, "month_start": 11, "label": "November" },
      { "id": 11, "month_start": 12, "label": "December" }
    ]
  }
}
```

| `cycle_type` | Meaning                             | Season count | Calendar rule                                                               |
| ------------ | ----------------------------------- | ------------ | --------------------------------------------------------------------------- |
| `monthly`    | Each season = one calendar month    | 12           | `month_start` maps to the calendar month                                    |
| `weekly`     | Each season = one ISO calendar week | 52           | Season `id` maps to ISO week number                                         |
| `custom`     | User-defined date ranges            | any          | Each season has explicit `month_start`, `day_start`, `month_end`, `day_end` |

For `custom` cycle type, each season requires explicit date boundaries:

```json
{
  "cycle_type": "custom",
  "seasons": [
    {
      "id": 0,
      "month_start": 1,
      "day_start": 1,
      "month_end": 4,
      "day_end": 1,
      "label": "Q1"
    },
    {
      "id": 1,
      "month_start": 4,
      "day_start": 1,
      "month_end": 7,
      "day_end": 1,
      "label": "Q2"
    },
    {
      "id": 2,
      "month_start": 7,
      "day_start": 1,
      "month_end": 10,
      "day_end": 1,
      "label": "Q3"
    },
    {
      "id": 3,
      "month_start": 10,
      "day_start": 1,
      "month_end": 1,
      "day_end": 1,
      "label": "Q4"
    }
  ]
}
```

**Validation rules:**

- Each stage's `[start_date, end_date)` interval must fall entirely within the calendar period defined by its `season_id`.
- All stages sharing the same `season_id` must have exactly the same duration. This ensures PAR parameters are truly periodic and history aggregation produces comparable values across years.

**When required:** `season_definitions` is required whenever `inflow_history` is provided (see §2.3). Otherwise it is optional but recommended for validation and reporting.

### 1.2 Policy Graph and Transitions

The `policy_graph` section defines the graph structure of stage transitions, the horizon type, and the global discount rate.

```json
{
  "policy_graph": {
    "type": "finite_horizon",
    "annual_discount_rate": 0.06,
    "transitions": [
      { "source_id": 0, "target_id": 1, "probability": 1.0 },
      { "source_id": 1, "target_id": 2, "probability": 1.0 }
    ]
  }
}
```

#### Policy Graph Types

| Type             | Description                                                                                                                                                                                                                |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `finite_horizon` | Linear chain of stages with a terminal condition. The simplest and most common structure.                                                                                                                                  |
| `cyclic`         | Stages form a cycle (e.g., stage 59 transitions back to stage 48). Used for infinite periodic horizon. Requires `annual_discount_rate > 0` for convergence. See [Discount Rate Formulation §15](../math/discount-rate.md). |

#### Discount Rate

The `annual_discount_rate` is specified as a yearly rate (e.g., `0.06` = 6% per year). The system automatically converts this to a per-transition discount factor based on each stage's duration:

- Stage duration `Δt` is derived from `end_date - start_date`, expressed in years
- Transition discount factor: `β = 1 / (1 + annual_discount_rate) ^ Δt`
- The duration used is that of the **source** stage (the stage whose future cost is being discounted)

A value of `0.0` means no discounting (β = 1.0 for all transitions).

Individual transitions may override the global rate:

```json
{
  "source_id": 59,
  "target_id": 48,
  "probability": 1.0,
  "annual_discount_rate": 0.1
}
```

Per-transition overrides follow the same annual-rate-to-factor conversion.

#### Transition Fields

| Field                  | Type | Required | Default      | Description                                         |
| ---------------------- | ---- | -------- | ------------ | --------------------------------------------------- |
| `source_id`            | i32  | Yes      | —            | Source stage ID                                     |
| `target_id`            | i32  | Yes      | —            | Target stage ID                                     |
| `probability`          | f64  | Yes      | —            | Transition probability (must sum to 1.0 per source) |
| `annual_discount_rate` | f64  | No       | Global value | Override annual discount rate for this transition   |

### 1.3 Pre-Study Stages

Stages with negative IDs represent historical periods before the study horizon. Used only for PAR model initialization (providing lag values). Pre-study stages only need `id`, `start_date`, and `end_date`.

```json
{
  "pre_study_stages": [
    { "id": -6, "start_date": "2023-07-01", "end_date": "2023-08-01" },
    { "id": -5, "start_date": "2023-08-01", "end_date": "2023-09-01" },
    { "id": -4, "start_date": "2023-09-01", "end_date": "2023-10-01" },
    { "id": -3, "start_date": "2023-10-01", "end_date": "2023-11-01" },
    { "id": -2, "start_date": "2023-11-01", "end_date": "2023-12-01" },
    { "id": -1, "start_date": "2023-12-01", "end_date": "2024-01-01" }
  ]
}
```

### 1.4 Stage Fields

Each stage defines its temporal extent, block structure, scenario configuration, and risk parameters.

| Field             | Type          | Required | Default             | Description                                                                               |
| ----------------- | ------------- | -------- | ------------------- | ----------------------------------------------------------------------------------------- |
| `id`              | i32           | Yes      | —                   | Unique stage identifier (non-negative)                                                    |
| `start_date`      | string        | Yes      | —                   | Stage start date (ISO 8601)                                                               |
| `end_date`        | string        | Yes      | —                   | Stage end date (ISO 8601)                                                                 |
| `season_id`       | i32 \| null   | No       | null                | Season index linking to `season_definitions`. Null for stages without seasonal structure. |
| `blocks`          | array         | Yes      | —                   | Load blocks within stage (see §1.5)                                                       |
| `block_mode`      | string        | No       | `"parallel"`        | Block formulation: `"parallel"` or `"chronological"` (see §1.5)                           |
| `state_variables` | object        | No       | `{"storage": true}` | State variable configuration (see §1.6)                                                   |
| `risk_measure`    | string/object | No       | `"expectation"`     | Risk measure: `"expectation"` or `{"cvar": {...}}` (see §1.7)                             |
| `num_scenarios`   | i32           | Yes      | —                   | Number of scenarios for this stage                                                        |
| `sampling_method` | string        | No       | `"saa"`             | Sampling method (see §1.8)                                                                |

### 1.5 Blocks and Block Mode

Each stage contains one or more load blocks. Block IDs within each stage must be contiguous starting at 0 (validated: 0, 1, 2, …, n-1).

```json
{
  "blocks": [
    { "id": 0, "name": "LEVE", "hours": 168 },
    { "id": 1, "name": "MEDIA", "hours": 336 },
    { "id": 2, "name": "PESADA", "hours": 168 }
  ]
}
```

Block weights are computed internally from block hours.

**Validation rule:** The sum of all block hours within a stage must equal the total stage duration (derived from `end_date - start_date` converted to hours).

The `block_mode` field controls the block formulation for each stage:

| Mode              | Description                                                                          |
| ----------------- | ------------------------------------------------------------------------------------ |
| `"parallel"`      | Blocks are independent sub-periods solved simultaneously within the stage (default). |
| `"chronological"` | Blocks are sequential within the stage, with inter-block state transitions.          |

Block mode can vary by stage, allowing adaptive strategies (e.g., chronological for near-term stages, parallel for distant stages). For the mathematical formulation of each mode, see [Block Formulations](../math/block-formulations.md).

### 1.6 State Variables

The `state_variables` field is an object with boolean flags indicating which variables carry state between stages:

```json
{
  "state_variables": {
    "storage": true,
    "inflow_lags": true
  }
}
```

| Flag          | Description                                    | Default |
| ------------- | ---------------------------------------------- | ------- |
| `storage`     | Reservoir storage volumes                      | `true`  |
| `inflow_lags` | Past inflow realizations used as AR model lags | `false` |

Storage is mandatory in most applications but kept as an explicit flag for transparency. Future extensions may add additional flags (e.g., `gnl_pipeline` for gas network state).

### 1.7 Risk Measure (CVaR)

The `risk_measure` field can be:

| Option            | Description                                                           |
| ----------------- | --------------------------------------------------------------------- |
| `"expectation"`   | Risk-neutral expected value (default)                                 |
| `{"cvar": {...}}` | CVaR parameters with `alpha` (confidence level) and `lambda` (weight) |

CVaR details: `alpha` = confidence level (e.g., 0.95 means 5% worst scenarios); `lambda` = weight of CVaR vs expectation. Final risk measure: `(1 - lambda) × E[cost] + lambda × CVaR_alpha[cost]`. CVaR parameters can vary by stage. See [Risk Measures](../math/risk-measures.md) for mathematical formulation.

### 1.8 Scenario Sampling Methods

| Method       | Description                                                                   | Use Case                             |
| ------------ | ----------------------------------------------------------------------------- | ------------------------------------ |
| `saa`        | **Sample Average Approximation** (default). Pure Monte Carlo random sampling. | General purpose, baseline            |
| `lhs`        | **Latin Hypercube Sampling**. Stratified sampling ensuring uniform coverage.  | Medium sample sizes (20–100)         |
| `qmc_sobol`  | **Quasi-Monte Carlo (Sobol sequences)**. Low-discrepancy sequences.           | High-dimensional, deterministic-like |
| `qmc_halton` | **Quasi-Monte Carlo (Halton sequences)**. Alternative low-discrepancy.        | Similar to Sobol                     |
| `selective`  | **Selective/Representative Sampling**. Clustering on historical data.         | Historical pattern-guided            |

Sampling method can vary by stage, allowing adaptive strategies.

### 1.9 Example

```json
{
  "$schema": "https://cobre.dev/schemas/v2/stages.schema.json",
  "season_definitions": {
    "cycle_type": "monthly",
    "seasons": [
      { "id": 0, "month_start": 1, "label": "January" },
      { "id": 1, "month_start": 2, "label": "February" },
      { "id": 2, "month_start": 3, "label": "March" }
    ]
  },
  "policy_graph": {
    "type": "finite_horizon",
    "annual_discount_rate": 0.06,
    "transitions": [
      { "source_id": 0, "target_id": 1, "probability": 1.0 },
      { "source_id": 1, "target_id": 2, "probability": 1.0 }
    ]
  },
  "pre_study_stages": [
    { "id": -6, "start_date": "2023-07-01", "end_date": "2023-08-01" },
    { "id": -5, "start_date": "2023-08-01", "end_date": "2023-09-01" },
    { "id": -4, "start_date": "2023-09-01", "end_date": "2023-10-01" },
    { "id": -3, "start_date": "2023-10-01", "end_date": "2023-11-01" },
    { "id": -2, "start_date": "2023-11-01", "end_date": "2023-12-01" },
    { "id": -1, "start_date": "2023-12-01", "end_date": "2024-01-01" }
  ],
  "stages": [
    {
      "id": 0,
      "start_date": "2024-01-01",
      "end_date": "2024-02-01",
      "season_id": 0,
      "blocks": [
        { "id": 0, "name": "LEVE", "hours": 248 },
        { "id": 1, "name": "MEDIA", "hours": 248 },
        { "id": 2, "name": "PESADA", "hours": 248 }
      ],
      "block_mode": "chronological",
      "risk_measure": { "cvar": { "alpha": 0.95, "lambda": 0.5 } },
      "state_variables": { "storage": true, "inflow_lags": true },
      "num_scenarios": 20,
      "sampling_method": "lhs"
    },
    {
      "id": 1,
      "start_date": "2024-02-01",
      "end_date": "2024-03-01",
      "season_id": 1,
      "blocks": [
        { "id": 0, "name": "LEVE", "hours": 232 },
        { "id": 1, "name": "MEDIA", "hours": 232 },
        { "id": 2, "name": "PESADA", "hours": 232 }
      ],
      "block_mode": "parallel",
      "risk_measure": { "cvar": { "alpha": 0.95, "lambda": 0.25 } },
      "state_variables": { "storage": true, "inflow_lags": true },
      "num_scenarios": 20,
      "sampling_method": "lhs"
    },
    {
      "id": 2,
      "start_date": "2024-03-01",
      "end_date": "2024-04-01",
      "season_id": 2,
      "blocks": [
        { "id": 0, "name": "LEVE", "hours": 248 },
        { "id": 1, "name": "MEDIA", "hours": 248 },
        { "id": 2, "name": "PESADA", "hours": 248 }
      ],
      "risk_measure": "expectation",
      "state_variables": { "storage": true },
      "num_scenarios": 20,
      "sampling_method": "saa"
    }
  ]
}
```

> **Note:** The `$schema` field is a placeholder. No live schema URL exists yet. All JSON examples in this spec and other approved specs use placeholder `$schema` values for future JSON Schema validation support.

### 1.10 Validation Rules

1. Stage IDs must be unique and non-negative.
2. Block IDs within each stage must be contiguous starting at 0.
3. Block hours within each stage must sum to the total stage duration.
4. All stages sharing the same `season_id` must have exactly the same duration.
5. Each stage's `[start_date, end_date)` must fall within the calendar period defined by its `season_id` in `season_definitions`.
6. Transition probabilities must sum to 1.0 per source stage.
7. For `cyclic` policy graphs, the cumulative discount factor around each cycle must be strictly less than 1.0.

## 2. Scenario Pipeline

### 2.1 Scenario Source and Sampling Scheme

The `scenario_source` object in `config.json` (under `training.scenario_source` for training, `simulation.scenario_source` for simulation) configures how scenarios are selected during the SDDP forward pass. Each stochastic class (inflow, load, NCS) has its own sampling scheme, configured via per-class sub-objects. When `simulation.scenario_source` is absent, it falls back to `training.scenario_source`. The sampling scheme abstraction is one of three orthogonal SDDP concerns formalized in [Scenario Generation §3](../architecture/scenario-generation.md).

#### Per-Class Format

```json
{
  "training": {
    "scenario_source": {
      "seed": 42,
      "inflow": { "scheme": "in_sample" },
      "load": { "scheme": "out_of_sample" },
      "ncs": { "scheme": "in_sample" },
      "historical_years": [1940, 1953, 1971]
    }
  }
}
```

| Field              | Type         | Required | Default | Description                                                                            |
| ------------------ | ------------ | -------- | ------- | -------------------------------------------------------------------------------------- |
| `seed`             | i64          | No       | —       | Base seed for reproducible noise generation (required when any class uses `in_sample`) |
| `inflow.scheme`    | string       | Yes      | —       | `"in_sample"`, `"out_of_sample"`, `"external"`, or `"historical"`                      |
| `load.scheme`      | string       | Yes      | —       | `"in_sample"`, `"out_of_sample"`, `"external"`, or `"historical"`                      |
| `ncs.scheme`       | string       | Yes      | —       | `"in_sample"`, `"out_of_sample"`, `"external"`, or `"historical"`                      |
| `historical_years` | array of i32 | No       | null    | Specific historical years for `historical` scheme                                      |

**Required inputs:** Uncertainty models (§3) — user-provided or derived from inflow history. For `external`, per-class external scenario files (§2.5). For `historical`, `inflow_history.parquet` (§2.4) + `season_definitions` (§1.1).

#### Sampling Scheme Variants

| Variant         | Description                                                                                                                                |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `in_sample`     | Forward pass samples from the fixed opening tree (PAR-generated noise). Standard SDDP forward sampling.                                    |
| `out_of_sample` | Forward pass draws from independently generated Monte Carlo noise, different from the opening tree. Backward pass uses the same PAR model. |
| `external`      | Forward pass draws from user-provided per-class scenario data. Backward pass uses a PAR model fitted to the external data.                 |
| `historical`    | Forward pass replays historical sequences mapped to stages. Backward pass uses a PAR model fitted to the historical data.                  |

#### Summary

| Sampling Scheme | Forward Noise Source                      | Backward Noise Source                         | Use Case                                      |
| --------------- | ----------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| `in_sample`     | Opening tree (PAR-generated)              | Same opening tree                             | Standard SDDP training                        |
| `out_of_sample` | Independently generated Monte Carlo noise | Opening tree from same PAR model              | Out-of-sample forward evaluation              |
| `external`      | User-provided per-class scenario values   | Opening tree from PAR fitted to external data | Training/simulation with imported scenarios   |
| `historical`    | Historical records mapped to stages       | Opening tree from PAR fitted to history       | Policy validation against observed conditions |

> **Noise inversion**: For `external` and `historical` schemes, the system internally performs reverse noise calculation — back-computing the noise vector epsilon that would produce the given values through the AR model. This is necessary because SDDP cuts are constructed in terms of state variables and the AR noise structure. This is an internal solver computation, not a data input concern. See [Scenario Generation §4.3](../architecture/scenario-generation.md).

### 2.2 Pipeline Flexibility

The scenario pipeline is a cascade of components, each of which can independently be **user-provided** or **derived from inflow history**:

| Component                  | User-provided via                       | Derived from                             |
| -------------------------- | --------------------------------------- | ---------------------------------------- |
| Seasonal statistics (μ, s) | `inflow_seasonal_stats.parquet` (§3.1)  | Inflow history aggregated by season      |
| AR coefficients (ψ\*)      | `inflow_ar_coefficients.parquet` (§3.2) | Fitted from inflow history (Yule-Walker) |
| Correlation matrices       | `correlation.json` (§5)                 | Estimated from AR residuals of history   |

**Presence or absence of input files controls the pipeline.** No explicit mode flags are needed:

| `inflow_seasonal_stats` | `inflow_ar_coefficients` | `correlation.json` | `inflow_history` | System behavior                                                       |
| ----------------------- | ------------------------ | ------------------ | ---------------- | --------------------------------------------------------------------- |
| present                 | present                  | present            | —                | Use all directly. No history needed.                                  |
| present                 | present                  | absent             | present          | Use AR models directly. Estimate correlations from history.           |
| present                 | absent                   | present            | present          | Use seasonal stats directly. Fit AR from history. Use correlations.   |
| present                 | absent                   | absent             | present          | Use seasonal stats. Fit AR and correlations from history.             |
| absent                  | absent                   | present            | present          | Fit seasonal stats + AR from history. Use provided correlations.      |
| absent                  | absent                   | absent             | present          | Derive everything from history.                                       |
| present                 | absent                   | —                  | absent           | No AR structure (AR order = 0 for all). Only seasonal stats apply.    |
| absent                  | present                  | —                  | —                | **Error** — AR coefficients require seasonal stats for normalization. |
| absent                  | absent                   | —                  | absent           | **Error** — no stochastic model possible.                             |

All combinations of user-provided and derived components are valid. For example, a user may provide AR coefficients but override seasonal means (μ) to force conditioned inflow regimes, while letting the system estimate correlations from history.

**When any component is derived from history**, the system requires `inflow_history` and `season_definitions` to aggregate raw observations into season-level values.

### 2.3 History Aggregation

The user can provide inflow history at any time resolution (daily, weekly, monthly, or other). The system aggregates observations to match the season resolution defined in `season_definitions`:

- **Finer resolution → season**: The system averages all observations within each season's calendar period. For example, daily observations are averaged across each calendar month for `monthly` seasons. Weekly observations spanning a season boundary are included via weighted average based on overlap days.
- **Same resolution**: Direct mapping, no aggregation needed.
- **Coarser resolution → season**: Error. The system cannot disaggregate observations into finer seasons without additional assumptions.

### 2.4 Inflow History (`scenarios/inflow_history.parquet`)

The `inflow_history` file contains raw historical inflow observations at the user's chosen time resolution.

> **Format Rationale — inflow_history.parquet**
>
> **Entity-level time series** — Historical observations per hydro indexed by date. Parquet for efficient columnar access across potentially thousands of rows (hydros x dates).

| Column      | Type | Description                                     |
| ----------- | ---- | ----------------------------------------------- |
| `hydro_id`  | i32  | Hydro plant ID (must exist in system entities)  |
| `date`      | date | Start date of the observation period (ISO 8601) |
| `value_m3s` | f64  | Mean inflow for the period (m³/s)               |

The resolution of the history data must be declared explicitly via the `inflow_history` configuration:

```json
{
  "inflow_history": {
    "resolution": "daily",
    "path": "inflow_history.parquet"
  }
}
```

| `resolution` | Meaning                                      |
| ------------ | -------------------------------------------- |
| `daily`      | One observation per day per hydro            |
| `weekly`     | One observation per ISO week per hydro       |
| `monthly`    | One observation per calendar month per hydro |

Declaring the resolution explicitly (rather than inferring it from date intervals) ensures deterministic validation and aggregation. The system knows exactly what intervals to expect and can flag missing or duplicate records without guessing.

**Usage contexts:**

1. **Deriving seasonal statistics** — Compute μ, σ per hydro per season.
2. **Fitting AR models** — Estimate ψ coefficients via Yule-Walker equations. See [PAR Inflow Model](../math/par-inflow-model.md).
3. **Estimating correlations** — Compute cross-correlation from AR model residuals.
4. **Historical scenario replay** — When the inflow class uses `"historical"` scheme, forward passes use actual historical sequences mapped to stages via `season_definitions`.

### 2.5 External Scenarios (Per-Class Files)

When a stochastic class uses the `"external"` scheme, the user provides pre-computed scenario values indexed directly by `stage_id` in a per-class Parquet file. This eliminates any need for season-calendar mapping — the user is responsible for ensuring the values match the stage structure. Each class has its own file:

| File                                          | Class  | Entity ID Column | Description                   |
| --------------------------------------------- | ------ | ---------------- | ----------------------------- |
| `scenarios/external_inflow_scenarios.parquet` | Inflow | `hydro_id`       | Pre-computed inflow scenarios |
| `scenarios/external_load_scenarios.parquet`   | Load   | `bus_id`         | Pre-computed load scenarios   |
| `scenarios/external_ncs_scenarios.parquet`    | NCS    | `ncs_id`         | Pre-computed NCS scenarios    |

> **Usage scope**: External scenarios can be used in both simulation AND training. In simulation, the forward pass replays external values directly. In training, the forward pass samples from external data, while the backward pass generates branchings from a PAR model fitted to the external data. See [Scenario Generation §3.2 and §4.2](../architecture/scenario-generation.md) for full details.

> **Format Rationale — external\_\*\_scenarios.parquet**
>
> **Stage-indexed scenario table** — Pre-computed values per stage, scenario, and entity. Parquet for large scenario trees with efficient columnar access. Per-class files enable independent class-level scheme selection (e.g., external inflows with in-sample load).

**Inflow external scenario schema** (`external_inflow_scenarios.parquet`):

| Column        | Type | Description                                |
| ------------- | ---- | ------------------------------------------ |
| `stage_id`    | i32  | Stage ID (must exist in `stages.json`)     |
| `scenario_id` | i32  | Scenario index (0-based)                   |
| `hydro_id`    | i32  | Hydro plant ID                             |
| `value_m3s`   | f64  | Inflow value for this stage/scenario/hydro |

**Load external scenario schema** (`external_load_scenarios.parquet`):

| Column        | Type | Description                            |
| ------------- | ---- | -------------------------------------- |
| `stage_id`    | i32  | Stage ID (must exist in `stages.json`) |
| `scenario_id` | i32  | Scenario index (0-based)               |
| `bus_id`      | i32  | Bus ID                                 |
| `value_mw`    | f64  | Load value for this stage/scenario/bus |

**NCS external scenario schema** (`external_ncs_scenarios.parquet`):

| Column        | Type | Description                                  |
| ------------- | ---- | -------------------------------------------- |
| `stage_id`    | i32  | Stage ID (must exist in `stages.json`)       |
| `scenario_id` | i32  | Scenario index (0-based)                     |
| `ncs_id`      | i32  | Non-controllable source ID                   |
| `value_mw`    | f64  | Generation value for this stage/scenario/NCS |

**Validation:** For each per-class file, the number of distinct `scenario_id` values per stage must equal the stage's `num_scenarios`.

## 3. Uncertainty Models

### 3.1 Inflow Seasonal Statistics (`scenarios/inflow_seasonal_stats.parquet`)

> **Format Rationale — inflow_seasonal_stats.parquet**
>
> **Entity-stage parameter table** — Per-entity-per-stage seasonal statistics (mean and standard deviation). Parquet for typed columnar access across hydros and stages.

When provided, this table supplies pre-computed seasonal mean and standard deviation directly. When absent, the system derives these from `inflow_history` via season aggregation (see §2.2).

| Column     | Type | Description                                                                                                                                                                                                                                                                                                           |
| ---------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hydro_id` | i32  | Hydro plant ID (must exist in system entities)                                                                                                                                                                                                                                                                        |
| `stage_id` | i32  | Stage ID (must exist in `stages.json`)                                                                                                                                                                                                                                                                                |
| `mean_m3s` | f64  | Seasonal mean inflow (μ)                                                                                                                                                                                                                                                                                              |
| `std_m3s`  | f64  | Seasonal sample standard deviation ($s_m$). 0 = deterministic. Note: this is the sample std of historical observations, NOT the residual std ($\sigma_m$) — the residual std is derived at runtime as $\sigma_m = s_m \cdot \texttt{residual\_std\_ratio}_m$. See [PAR Inflow Model §3](../math/par-inflow-model.md). |

The AR order $p_m$ is **not stored in this file**. It is derived from the count of coefficient rows per (hydro_id, stage_id) group in `inflow_ar_coefficients.parquet`. Storing it here would create a redundant data source and a possible informational mismatch — the user would have to keep the same value consistent across two files.

### 3.2 Inflow AR Coefficients (`scenarios/inflow_ar_coefficients.parquet`) — Optional

> **Format Rationale — inflow_ar_coefficients.parquet**
>
> **Entity-stage-lag parameter table** — Long-form table of AR coefficients with one row per (hydro, stage, lag). Parquet for typed columnar access; long-form avoids null columns and imposes no maximum AR order.

When provided, this table supplies pre-computed AR coefficients. When absent, the system either fits AR coefficients from `inflow_history` (if present) or uses AR order 0 (independent noise).

The AR model for a given stage uses lags from previous stages. Coefficients are stored in **standardized form** ($\psi^*_{m,\ell}$) — the direct output of the Yule-Walker fitting procedure. The system derives original-unit coefficients ($\psi_{m,\ell} = \psi^*_{m,\ell} \cdot s_m / s_{m-\ell}$) at runtime from these stored values and the seasonal std in `inflow_seasonal_stats.parquet`. Innovation terms (ε) are standard normal, transformed into correlated samples via spectral decomposition of the correlation matrix (see SS5).

The `residual_std_ratio` column stores the ratio $\sigma_m / s_m$ — the fraction of seasonal variability not explained by the AR lags. This is a pure model property (fixed per PAR fit), stored separately from the seasonal std $s_m$ (a conditioning property, swappable for climate scenario studies). The residual std is recovered at runtime as $\sigma_m = s_m \cdot \texttt{residual\_std\_ratio}_m$. See [PAR Inflow Model §3](../math/par-inflow-model.md) for the rationale.

| Column               | Type | Description                                                                        |
| -------------------- | ---- | ---------------------------------------------------------------------------------- |
| `hydro_id`           | i32  | Hydro plant ID (must exist in system entities)                                     |
| `stage_id`           | i32  | Stage ID (must exist in `stages.json`)                                             |
| `lag`                | i32  | Lag index (1-based: 1 = first lag, 2 = second, etc.)                               |
| `coefficient`        | f64  | AR coefficient $\psi^*_{m,\ell}$, **standardized by seasonal std** (dimensionless) |
| `residual_std_ratio` | f64  | Residual std as fraction of seasonal std ($\sigma_m / s_m$), in $(0, 1]$           |

**Example rows** (hydro 0, stage 5, AR order 3):

| hydro_id | stage_id | lag | coefficient | residual_std_ratio |
| -------- | -------- | --- | ----------- | ------------------ |
| 0        | 5        | 1   | 0.45        | 0.872              |
| 0        | 5        | 2   | 0.22        | 0.872              |
| 0        | 5        | 3   | 0.08        | 0.872              |

The `residual_std_ratio` value is the same for all lag rows of a given (hydro_id, stage_id) group — it is a per-season property of the model, not a per-lag property. Parquet dictionary encoding handles this efficiently.

**Validation rules:**

- Lags must be contiguous starting at 1: `[1, 2, ..., p]` for AR order `p`. The AR order $p$ for a given (hydro_id, stage_id) is derived from the count of rows in this group.
- `residual_std_ratio` must be in $(0, 1]$ and must be identical across all lag rows of the same (hydro_id, stage_id) group.
- AR coefficients present without seasonal stats = **error** (AR needs $s_m$ for runtime conversion to original-unit form).

See [PAR Inflow Model](../math/par-inflow-model.md) for the mathematical formulation.

### 3.3 Load Seasonal Statistics (`scenarios/load_seasonal_stats.parquet`)

> **Format Rationale — load_seasonal_stats.parquet**
>
> **Entity-stage parameter table** — Per-bus-per-stage load statistics. Parquet for typed columnar data consistent with other tabular inputs.

| Column     | Type | Description                            |
| ---------- | ---- | -------------------------------------- |
| `bus_id`   | i32  | Bus ID (must exist in system entities) |
| `stage_id` | i32  | Stage ID (must exist in `stages.json`) |
| `mean_mw`  | f64  | Mean load for this stage               |
| `std_mw`   | f64  | Standard deviation (0 = deterministic) |

Load models are typically independent (no AR structure), so no AR columns are included.

## 4. Load Factors by Block — Optional

> **Format Rationale — load_factors.json**
>
> **Default-with-overrides** — Small number of load factor definitions that rarely change. JSON for readability and simplicity.

If missing, all block factors default to 1.0.

Load uncertainty generates a base load realization in **MW** per stage. Block factors are **multipliers** applied to the stochastic load value. For example, if stage load is 1000 MW and block factors are [0.85, 1.00, 1.15], the blocks get [850, 1000, 1150] MW.

```json
{
  "$schema": "https://cobre.dev/schemas/v2/load_factors.schema.json",
  "load_factors": [
    {
      "bus_id": 0,
      "stage_id": 0,
      "block_factors": [
        { "block_id": 0, "factor": 0.85 },
        { "block_id": 1, "factor": 1.0 },
        { "block_id": 2, "factor": 1.15 }
      ]
    }
  ]
}
```

## 5. Non-Controllable Source Scenarios

Non-controllable sources (wind, solar) support two optional scenario input files for stochastic availability modeling and block-level scaling.

### 5.1 NCS Stochastic Availability (`scenarios/non_controllable_stats.parquet`) -- Optional

Defines per-NCS-per-stage mean and standard deviation of the stochastic availability factor. When present, the scenario pipeline generates stochastic NCS availability using these parameters. When absent, NCS generation is deterministic (uses the available generation from `constraints/ncs_bounds.parquet` directly).

| Column     | Type | Nullable | Description                                                         |
| ---------- | ---- | -------- | ------------------------------------------------------------------- |
| `ncs_id`   | i32  | No       | Non-controllable source entity ID                                   |
| `stage_id` | i32  | No       | Stage ID                                                            |
| `mean`     | f64  | No       | Mean availability factor, in [0, 1]                                 |
| `std`      | f64  | No       | Standard deviation of availability factor (>= 0; 0 = deterministic) |

**Validation rules:**

- All four columns must be present with the correct types
- `mean` must be finite and in [0, 1] (NaN, +/-inf, and out-of-range are rejected)
- `std` must be non-negative and finite
- Rows are sorted by `(ncs_id, stage_id)` ascending after loading

**Deferred validations** (not performed at parse time):

- `ncs_id` existence in the NCS registry (Layer 3 referential validation)
- `stage_id` existence in the stages registry (Layer 3 referential validation)

### 5.2 NCS Block Scaling Factors (`scenarios/non_controllable_factors.json`) -- Optional

Defines per-NCS-per-stage block-level generation scaling factors. When present, NCS generation at each block is multiplied by the corresponding factor. When absent, all block factors default to 1.0.

```json
{
  "$schema": "https://cobre.dev/schemas/v2/non_controllable_factors.schema.json",
  "non_controllable_factors": [
    {
      "ncs_id": 0,
      "stage_id": 0,
      "block_factors": [
        { "block_id": 0, "factor": 0.6 },
        { "block_id": 1, "factor": 0.8 }
      ]
    }
  ]
}
```

| Field                                                 | Type | Description                                           |
| ----------------------------------------------------- | ---- | ----------------------------------------------------- |
| `non_controllable_factors[].ncs_id`                   | i32  | Non-controllable source entity ID                     |
| `non_controllable_factors[].stage_id`                 | i32  | Stage ID                                              |
| `non_controllable_factors[].block_factors[].block_id` | i32  | Block ID                                              |
| `non_controllable_factors[].block_factors[].factor`   | f64  | Scaling factor (must be strictly positive and finite) |

**Validation rules:**

- No two entries may share the same `(ncs_id, stage_id)` pair
- Every `factor` value must be strictly positive (> 0.0) and finite
- Entries are sorted by `(ncs_id, stage_id)` ascending; block factors within each entry are sorted by `block_id` ascending

### 5.3 NCS Available Generation Bounds (`constraints/ncs_bounds.parquet`) -- Optional

Defines per-NCS-per-stage available generation bounds. When present, these specify the maximum generation (MW) for each non-controllable source at each stage.

| Column                    | Type | Nullable | Description                       |
| ------------------------- | ---- | -------- | --------------------------------- |
| `ncs_id`                  | i32  | No       | Non-controllable source entity ID |
| `stage_id`                | i32  | No       | Stage ID                          |
| `available_generation_mw` | f64  | No       | Available generation (MW), >= 0.0 |

**Validation rules:**

- All three columns must be present with the correct types
- `available_generation_mw` must be finite and >= 0.0
- Rows are sorted by `(ncs_id, stage_id)` ascending after loading

## 6. Correlation (`scenarios/correlation.json`)

> **Format Rationale — correlation.json**
>
> **Correlation / matrix data** — Symmetric correlation matrices between entities. JSON because data is small and structure is not tabular. Profile-based design avoids element-wise storage.

Defines spatial correlation between stochastic processes (inflows, loads, non-controllable generation). The system uses spectral decomposition (eigendecomposition with negative-eigenvalue clipping) to transform independent standard normal samples into correlated samples.

When provided, correlation matrices are used directly. When absent and `inflow_history` is available, the system estimates correlations from AR model residuals (see §2.2).

### 5.1 Profile-Based Time-Varying Correlation

Instead of storing element-wise overrides (O(stages × entities²) rows), Cobre uses a **profile-based system**:

1. **Named profiles** — Define multiple correlation matrices (e.g., `"default"`, `"wet_season"`, `"dry_season"`)
2. **Schedule table** — A compact tabular file maps each stage to a profile name

This reduces storage from potentially millions of rows to ~T rows plus a few matrix definitions.

```json
{
  "$schema": "https://cobre.dev/schemas/v2/correlation.schema.json",
  "method": "spectral",
  "profiles": {
    "default": {
      "correlation_groups": [
        {
          "name": "southeast_cascade",
          "entities": [
            { "type": "inflow", "id": 0 },
            { "type": "inflow", "id": 1 },
            { "type": "inflow", "id": 2 }
          ],
          "matrix": [
            [1.0, 0.75, 0.6],
            [0.75, 1.0, 0.7],
            [0.6, 0.7, 1.0]
          ]
        }
      ]
    },
    "wet_season": {
      "correlation_groups": [
        {
          "name": "southeast_cascade",
          "entities": [
            { "type": "inflow", "id": 0 },
            { "type": "inflow", "id": 1 },
            { "type": "inflow", "id": 2 }
          ],
          "matrix": [
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.85],
            [0.8, 0.85, 1.0]
          ]
        }
      ]
    }
  }
}
```

### 5.2 Correlation Profile Fields

| Field                                           | Type   | Required | Description                                                                                                                                                                                  |
| ----------------------------------------------- | ------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `method`                                        | string | Yes      | Correlation method: `"spectral"` (default) or `"cholesky"` (backward compatibility). Spectral decomposition handles non-positive-definite matrices by clipping negative eigenvalues to zero. |
| `profiles`                                      | object | Yes      | Map of profile names to correlation group definitions                                                                                                                                        |
| `profiles.<name>.correlation_groups`            | array  | Yes      | Array of correlation groups for this profile                                                                                                                                                 |
| `profiles.<name>.correlation_groups[].name`     | string | Yes      | Unique name for correlation group                                                                                                                                                            |
| `profiles.<name>.correlation_groups[].entities` | array  | Yes      | Entities in this correlation group                                                                                                                                                           |
| `profiles.<name>.correlation_groups[].matrix`   | array  | Yes      | Correlation matrix (must be positive semi-definite; per DEC-020). With `"spectral"` method, matrices with small negative eigenvalues are accepted and clipped.                               |

The profile named `"default"` is required and used for any stage not explicitly mapped in the schedule.

### 5.3 Time-Varying Correlation Schedule — Optional

The correlation schedule is **embedded in `correlation.json`** as a `"schedule"` array. Each entry maps a stage to a named profile. If the schedule is absent or a stage is not listed, the `"default"` profile is used.

```json
{
  "$schema": "https://cobre.dev/schemas/v2/correlation.schema.json",
  "method": "spectral",
  "profiles": {
    "default": { "...": "..." },
    "wet_season": { "...": "..." },
    "dry_season": { "...": "..." }
  },
  "schedule": [
    { "stage_id": 0, "profile_name": "wet_season" },
    { "stage_id": 1, "profile_name": "wet_season" },
    { "stage_id": 4, "profile_name": "default" },
    { "stage_id": 5, "profile_name": "dry_season" }
  ]
}
```

| Field          | Type   | Description                                                  |
| -------------- | ------ | ------------------------------------------------------------ |
| `stage_id`     | i32    | Stage ID                                                     |
| `profile_name` | string | Profile name (must exist in `profiles` within the same file) |

Stages not listed in the schedule use the `"default"` profile. Only stages that deviate from the default need to be listed.

### 5.4 Validation

1. All profile names in the schedule must exist in `correlation.json`.
2. All correlation matrices must be positive semi-definite.
3. Entity IDs in correlation groups must exist in the system.

### 5.5 Correlation Input Options Summary

| Approach             | Files Required                                                          | Use Case                           |
| -------------------- | ----------------------------------------------------------------------- | ---------------------------------- |
| Static correlation   | `correlation.json` with only `"default"` profile                        | Same correlation for all stages    |
| Seasonal correlation | `correlation.json` with multiple profiles + embedded `"schedule"` array | Different profiles by season/stage |
| Derived from history | `inflow_history` (no `correlation.json`)                                | System estimates from AR residuals |

## 7. Seasonal Override Pattern (Cross-Cutting)

Several data model elements exhibit the same pattern: a value or configuration that varies by season or stage. This appears in production model selection, load factors, exchange factors, and correlation profiles.

Two approaches have been identified for this pattern:

### 7.1 Profile + Schedule

Define named profiles (complete configurations) and a separate schedule table that maps stages to profile names. For correlation, the schedule is embedded in the same JSON file (see §6.3).

**Strengths:** Clean separation of definitions and temporal assignment. Profiles are reusable. Schedule table is tiny. Good for complex objects (correlation matrices, production models).

**Weaknesses:** Requires two files per concept. Indirection may be confusing for simple cases.

**Used in:** Correlation (§6), production model selection (see [Input Hydro Extensions](input-hydro-extensions.md)).

### 7.2 Stage/Season Tagged Union

Include the varying parameter directly in each stage definition or in a per-stage table. The value is a tagged union selecting between variants.

**Strengths:** Self-contained — no external schedule file. Good for simple variant selection (e.g., `block_mode`, `risk_measure`).

**Weaknesses:** Repetitive for large stage counts. Doesn't scale for complex objects.

**Used in:** `block_mode` (§1.5), `risk_measure` (§1.7), `state_variables` (§1.6).

The final decision on which approach to use for each element will be made during implementation. Both are valid and may coexist.

## Cross-References

- [Input Constraints](input-constraints.md) — Initial conditions (§1) that bootstrap the stochastic process; time-varying entity bounds (§2)
- [Input System Entities](input-system-entities.md) — Buses and hydros referenced by uncertainty models
- [Input Directory Structure](input-directory-structure.md) — Overall case directory layout
- [PAR Inflow Model](../math/par-inflow-model.md) — Mathematical formulation of the AR inflow model
- [Risk Measures](../math/risk-measures.md) — CVaR mathematical formulation
- [Block Formulations](../math/block-formulations.md) — How blocks partition each stage and parallel vs chronological modes
- [Discount Rate Formulation](../math/discount-rate.md) — Discount factor mathematics and infinite periodic horizon
- [Scenario Generation](../architecture/scenario-generation.md) — Scenario pipeline architecture: sampling scheme abstraction (§3), opening tree (§2.3), external scenario integration (§4), complete tree mode (§7)
- [Deferred Features](../deferred.md) — User-supplied pre-correlated noise openings (C.11), complete tree solver integration (C.12)
- [Design Principles §3](../overview/design-principles.md) — Order invariance and canonical ordering
