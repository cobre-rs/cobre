# Input System Entities

## Purpose

This spec defines the JSON schemas for all system entity registries. These files live under `system/` in the input case directory and define the physical model that the optimizer operates on.

The following system elements are defined here:

| Section | Entity                   | File                                   | Status |
| ------- | ------------------------ | -------------------------------------- | ------ |
| §1      | Buses                    | `system/buses.json`                    | Active |
| §2      | Transmission Lines       | `system/lines.json`                    | Active |
| §3      | Hydro Plants             | `system/hydros.json`                   | Active |
| §4      | Thermal Plants           | `system/thermals.json`                 | Active |
| §5      | Pumping Stations         | `system/pumping_stations.json`         | Active |
| §6      | Import/Export Contracts  | `system/energy_contracts.json`         | Active |
| §7      | Non-Controllable Sources | `system/non_controllable_sources.json` | Active |

For the overall directory layout and configuration, see [Input Directory Structure](input-directory-structure.md).

## 1. Buses (`system/buses.json`)

> **Order Invariance**: The order of buses in this array does NOT affect results. After loading, all buses are sorted by `id`. See [Design Principles §3](../overview/design-principles.md).

### Concept

A **bus** represents a node in the electrical network where energy balance must be maintained. Buses are a general network modeling concept — they can represent regional subsystems (as in the Brazilian SIN with 4 regions), individual substations, or any aggregation level suitable for the study. The number of buses can range from a handful for regional studies to hundreds or thousands for more detailed network representations.

> **Format Rationale — buses.json**
>
> Buses are a set of entities with cross-references (lines, hydros, and thermals all reference a `bus_id`). JSON is a natural fit for registry data: each entity is a structured object with a unique ID.

### Deficit Modeling

Deficit is modeled as piecewise linear segments. Each segment specifies a depth (MW of unmet demand) and cost. Segments are cumulative: first `depth_mw` MW at first cost, next `depth_mw` MW at second cost, etc. The last segment with `depth_mw: null` extends to infinity (required for LP feasibility).

Deficit is a **recourse slack** in the [Penalty System](penalty-system.md) — it ensures LP feasibility when demand cannot be met. The deficit cost represents the value of lost load and must be high enough to make load shedding a last resort. See [Penalty System §2, Category 1](penalty-system.md) for the role of deficit in the penalty taxonomy and appropriate cost ranges.

Global deficit defaults are defined in `penalties.json`; entity-level overrides are defined inline here.

```json
{
  "$schema": "https://cobre.dev/schemas/v2/buses.schema.json",
  "buses": [
    {
      "id": 0,
      "name": "SUDESTE",
      "deficit_segments": [
        { "depth_mw": 1000, "cost": 2000.0 },
        { "depth_mw": 2000, "cost": 5000.0 },
        { "depth_mw": null, "cost": 10000.0 }
      ]
    },
    {
      "id": 1,
      "name": "SUL"
    }
  ]
}
```

> **Note on Bus 1 (SUL)**: No `deficit_segments` defined — uses global default from `penalties.json`.

### Bus Fields

| Field              | Type   | Required | Description                                                                                                                         |
| ------------------ | ------ | -------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `id`               | i32    | Yes      | Unique bus identifier                                                                                                               |
| `name`             | string | Yes      | Human-readable bus name                                                                                                             |
| `deficit_segments` | array  | No       | Piecewise deficit cost segments (uses global default from `penalties.json` if omitted). See [Penalty System](penalty-system.md) §3. |

> **Runtime-resolved field**: The resolved Bus entity also carries `excess_cost: f64`, sourced from the global penalty defaults in `penalties.json` (or overridden per-bus per-stage via `penalty_overrides_bus.parquet`). This is not a user-provided JSON field but a runtime-resolved value populated during input loading.

### Deficit Segment Fields

| Field      | Type        | Required | Description                                                       |
| ---------- | ----------- | -------- | ----------------------------------------------------------------- |
| `depth_mw` | f64 \| null | Yes      | MW of deficit at this segment (`null` for final infinite segment) |
| `cost`     | f64         | Yes      | Cost per MWh of deficit in this segment                           |

For the mathematical formulation of bus load balance constraints and deficit variables, see [System Element Modeling](../math/system-elements.md).

## 2. Lines (`system/lines.json`)

> **Order Invariance**: The order of lines in this array does NOT affect results. After loading, all lines are sorted by `id`. See [Design Principles §3](../overview/design-principles.md).

### Concept

A **transmission line** represents an interconnection between two buses, allowing bidirectional power transfer subject to capacity limits and transmission losses. The number of lines scales with the number of buses — detailed network representations may have hundreds of lines.

> **Format Rationale — lines.json**
>
> Lines are entity definitions referencing buses. Each line connects exactly two buses and has capacity, cost, and lifecycle attributes. JSON is a natural fit because each line is a self-contained structured object with cross-references to bus IDs.

### Exchange Cost

The `exchange_cost` field is a **regularization cost** as defined in [Penalty System §2, Category 3](penalty-system.md). It discourages unnecessary power flow between buses, preventing degenerate solutions with power circulation. It is NOT a violation penalty — line capacity bounds are hard constraints without slack variables (see [Penalty System §4](penalty-system.md)). Default is defined in `penalties.json`.

### Line Operative States

| State            | Condition                                    | LP Variables              |
| ---------------- | -------------------------------------------- | ------------------------- |
| `non_existing`   | Before `entry_stage_id`                      | None (buses isolated)     |
| `operating`      | Between `entry_stage_id` and `exit_stage_id` | direct_flow, reverse_flow |
| `decommissioned` | After `exit_stage_id`                        | None (buses isolated)     |

```json
{
  "$schema": "https://cobre.dev/schemas/v2/lines.schema.json",
  "lines": [
    {
      "id": 0,
      "name": "SE-NE",
      "source_bus_id": 0,
      "target_bus_id": 1,
      "entry_stage_id": null,
      "exit_stage_id": null,
      "capacity": {
        "direct_mw": 5000.0,
        "reverse_mw": 3000.0
      },
      "exchange_cost": 0.01,
      "losses_percent": 2.5
    }
  ]
}
```

### Line Fields

| Field                 | Type        | Required | Default | Description                                                                                                                               |
| --------------------- | ----------- | -------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `id`                  | i32         | Yes      | —       | Unique line identifier                                                                                                                    |
| `name`                | string      | Yes      | —       | Human-readable line name                                                                                                                  |
| `source_bus_id`       | i32         | Yes      | —       | Source bus for direct flow                                                                                                                |
| `target_bus_id`       | i32         | Yes      | —       | Target bus for direct flow                                                                                                                |
| `entry_stage_id`      | i32 \| null | No       | null    | Stage when line enters service (`null` = always exists)                                                                                   |
| `exit_stage_id`       | i32 \| null | No       | null    | Stage when line is decommissioned (`null` = never)                                                                                        |
| `capacity.direct_mw`  | f64         | Yes      | —       | Maximum flow from source to target (MW)                                                                                                   |
| `capacity.reverse_mw` | f64         | Yes      | —       | Maximum flow from target to source (MW)                                                                                                   |
| `exchange_cost`       | f64         | No       | global  | Regularization cost per MWh exchanged (uses `penalties.json` default if omitted). See [Penalty System §2, Category 3](penalty-system.md). |
| `losses_percent`      | f64         | No       | 0.0     | Transmission losses as percentage                                                                                                         |

For the mathematical formulation of exchange variables and capacity constraints, see [System Element Modeling](../math/system-elements.md).

## 3. Hydro Registry (`system/hydros.json`) — Core Schema

> **Order Invariance**: The order of hydros in this array does NOT affect results. After loading, hydros are sorted by `id`. See [Design Principles §3](../overview/design-principles.md).

> **Format Rationale — hydros.json**
>
> Hydro plants are complex entities with many optional nested fields: reservoir bounds, outflow limits, generation model selection, diversion channels, filling configuration, and optional penalty overrides. JSON handles optional/nested structures well — absent fields use defaults, and the hierarchical layout mirrors the physical structure of the plant.

### Hydro Operative States

| State            | Condition                                                 | LP Variables                                                                               |
| ---------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `non_existing`   | Before filling or entry (no filling defined)              | None (no LP variables or constraints)                                                      |
| `filling`        | Between `filling.start_stage_id` and `entry_stage_id - 1` | storage, outflow (=spillage), violation slacks, evaporation                                |
| `operating`      | Between `entry_stage_id` and `exit_stage_id`              | storage, turbined, spillage, diversion, outflow, generation, evaporation, violation slacks |
| `decommissioned` | After `exit_stage_id`                                     | None (identical to non_existing)                                                           |

> **Dead-volume filling**: During filling stages, the reservoir accumulates water. `turbined_flow = 0` (hard constraint), and all released water goes through spillways/bottom gates. Environmental flow must be met via spillage.
>
> **Cascade redirection**: The `downstream_id` always refers to the physical downstream plant. During stages when the downstream plant doesn't exist, outflows are automatically redirected to the next operating downstream in the cascade.
>
> **Penalties**: Penalty defaults are defined in `penalties.json`. Entity-level overrides can be specified in an optional `penalties` block in the hydro definition. Stage-varying overrides use the stage override mechanism (format TBD). See [Penalty System](penalty-system.md).

> **Resolved — LP behavior for non-existing and decommissioned hydros**: Both `non_existing` and `decommissioned` hydros have no LP variables or constraints. Decommissioned is treated identically to non-existing for all entity types, including hydros. Reservoir drainage after decommissioning is not modeled. Cascade redirection handles water routing around absent plants (see cascade redirection note above). This simplifies the lifecycle to three meaningful LP states: Non-existing/Decommissioned (no LP presence), Filling (hydros only), and Operating (full LP formulation). LP shape regularity is achieved by omitting variables/constraints for inactive entities rather than adding zero-bounded placeholders.

### Generation Model (Tagged Union)

The `generation` object uses a tagged union pattern: the `model` field selects the variant, and only fields relevant to that variant are required. This ensures strict validation per model type.

**Variant: `constant_productivity`**

```json
{
  "generation": {
    "model": "constant_productivity",
    "productivity_mw_per_m3s": 0.8765,
    "min_turbined_m3s": 0.0,
    "max_turbined_m3s": 1692.0,
    "min_generation_mw": 0.0,
    "max_generation_mw": 1312.0
  }
}
```

| Field                     | Type | Required | Description                              |
| ------------------------- | ---- | -------- | ---------------------------------------- |
| `model`                   | str  | Yes      | Must be `"constant_productivity"`        |
| `productivity_mw_per_m3s` | f64  | Yes      | Constant productivity factor [MW/(m³/s)] |
| `min_turbined_m3s`        | f64  | Yes      | Minimum turbined flow (machine limits)   |
| `max_turbined_m3s`        | f64  | Yes      | Maximum turbined flow (machine capacity) |
| `min_generation_mw`       | f64  | Yes      | Minimum generation bound (user-defined)  |
| `max_generation_mw`       | f64  | Yes      | Maximum generation bound (user-defined)  |

**Variant: `linearized_head`** (**simulation-only** — excluded from training; see [Hydro Production Models §3](../math/hydro-production-models.md))

```json
{
  "generation": {
    "model": "linearized_head",
    "productivity_mw_per_m3s": 0.8765,
    "min_turbined_m3s": 0.0,
    "max_turbined_m3s": 1692.0,
    "min_generation_mw": 0.0,
    "max_generation_mw": 1312.0
  }
}
```

| Field                     | Type | Required | Description                                        |
| ------------------------- | ---- | -------- | -------------------------------------------------- |
| `model`                   | str  | Yes      | Must be `"linearized_head"`                        |
| `productivity_mw_per_m3s` | f64  | Yes      | Base productivity (adjusted by head linearization) |
| `min_turbined_m3s`        | f64  | Yes      | Minimum turbined flow                              |
| `max_turbined_m3s`        | f64  | Yes      | Maximum turbined flow                              |
| `min_generation_mw`       | f64  | Yes      | Minimum generation bound (user-defined)            |
| `max_generation_mw`       | f64  | Yes      | Maximum generation bound (user-defined)            |

Requires `hydro_geometry` data (see [Input Hydro Extensions](input-hydro-extensions.md)).

**Variant: `fpha`**

```json
{
  "generation": {
    "model": "fpha",
    "min_turbined_m3s": 0.0,
    "max_turbined_m3s": 1692.0,
    "min_generation_mw": 0.0,
    "max_generation_mw": 1312.0
  }
}
```

| Field               | Type | Required | Description                             |
| ------------------- | ---- | -------- | --------------------------------------- |
| `model`             | str  | Yes      | Must be `"fpha"`                        |
| `min_turbined_m3s`  | f64  | Yes      | Minimum turbined flow                   |
| `max_turbined_m3s`  | f64  | Yes      | Maximum turbined flow                   |
| `min_generation_mw` | f64  | Yes      | Minimum generation bound (user-defined) |
| `max_generation_mw` | f64  | Yes      | Maximum generation bound (user-defined) |

Note: `productivity_mw_per_m3s` is NOT required for `fpha` — the production function is defined entirely by the hyperplane coefficients. FPHA configuration (computed vs. precomputed, discretization parameters) is specified in the production models extension file. See [Input Hydro Extensions](input-hydro-extensions.md).

> **Generation bounds**: Both `min_generation_mw` and `max_generation_mw` are mandatory user-defined values. They are NOT derived from turbined flow bounds because the production function varies by model and may not be a simple linear relationship. See [Penalty System §7](penalty-system.md) for how generation bounds interact with violation slacks.

### JSON Example

```json
{
  "$schema": "https://cobre.dev/schemas/v2/hydros.schema.json",
  "hydros": [
    {
      "id": 0,
      "name": "FURNAS",
      "bus_id": 0,
      "downstream_id": 2,
      "entry_stage_id": null,
      "exit_stage_id": null,
      "filling": null,
      "diversion": null,
      "reservoir": {
        "min_storage_hm3": 5733.0,
        "max_storage_hm3": 22950.0
      },
      "outflow": {
        "min_outflow_m3s": 0.0,
        "max_outflow_m3s": null
      },
      "generation": {
        "model": "constant_productivity",
        "productivity_mw_per_m3s": 0.8765,
        "min_turbined_m3s": 0.0,
        "max_turbined_m3s": 1692.0,
        "min_generation_mw": 0.0,
        "max_generation_mw": 1312.0
      },
      "tailrace": {
        "type": "polynomial",
        "coefficients": [326.0, 0.0032, -1.2e-7]
      },
      "hydraulic_losses": {
        "type": "factor",
        "value": 0.03
      },
      "efficiency": {
        "type": "constant",
        "value": 0.92
      },
      "evaporation": {
        "coefficients_mm": [
          150, 130, 120, 90, 60, 40, 30, 40, 70, 100, 130, 150
        ]
      }
    },
    {
      "id": 10,
      "name": "NEW_HYDRO",
      "bus_id": 0,
      "downstream_id": 0,
      "entry_stage_id": 60,
      "exit_stage_id": null,
      "filling": {
        "start_stage_id": 48,
        "filling_inflow_m3s": 100.0
      },
      "diversion": null,
      "reservoir": {
        "min_storage_hm3": 1000.0,
        "max_storage_hm3": 5000.0
      },
      "outflow": {
        "min_outflow_m3s": 50.0,
        "max_outflow_m3s": null
      },
      "generation": {
        "model": "constant_productivity",
        "productivity_mw_per_m3s": 0.95,
        "min_turbined_m3s": 0.0,
        "max_turbined_m3s": 500.0,
        "min_generation_mw": 0.0,
        "max_generation_mw": 475.0
      }
    }
  ]
}
```

> **Note on NEW_HYDRO**: No `tailrace`, `hydraulic_losses`, `efficiency`, or `evaporation` fields — uses fallback assumptions (no tailrace adjustment, zero losses, efficiency derived from productivity, no evaporation).

### Core Hydro Fields

| Field                        | Type           | Required | Description                                                                                                                                                                         |
| ---------------------------- | -------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`                         | i32            | Yes      | Unique hydro identifier                                                                                                                                                             |
| `name`                       | string         | Yes      | Human-readable hydro name                                                                                                                                                           |
| `bus_id`                     | i32            | Yes      | Bus where generation is injected                                                                                                                                                    |
| `downstream_id`              | i32 \| null    | Yes      | Physical downstream hydro (`null` if none)                                                                                                                                          |
| `entry_stage_id`             | i32 \| null    | No       | Stage when hydro enters operation (`null` = always)                                                                                                                                 |
| `exit_stage_id`              | i32 \| null    | No       | Stage when hydro is decommissioned (`null` = never)                                                                                                                                 |
| `filling`                    | object \| null | No       | Dead-volume filling configuration (`null` = no filling). See [Filling Model](#filling-model) below.                                                                                 |
| `filling.start_stage_id`     | i32            | Yes\*    | First stage of the filling period                                                                                                                                                   |
| `filling.filling_inflow_m3s` | f64            | No\*     | Default filling inflow retained per stage (m³/s). Can be overridden per stage in `hydro_bounds` (see [Input Constraints §2](input-constraints.md)). Default: 0.0 (passive filling). |
| `diversion`                  | object \| null | No       | Diversion channel configuration (`null` = no diversion)                                                                                                                             |
| `diversion.downstream_id`    | i32            | Yes\*\*  | Hydro plant receiving diverted water                                                                                                                                                |
| `diversion.max_flow_m3s`     | f64            | Yes\*\*  | Maximum diversion flow                                                                                                                                                              |
| `reservoir.min_storage_hm3`  | f64            | Yes      | Minimum storage (dead volume)                                                                                                                                                       |
| `reservoir.max_storage_hm3`  | f64            | Yes      | Maximum storage (full reservoir)                                                                                                                                                    |
| `outflow.min_outflow_m3s`    | f64            | Yes      | Minimum total outflow (environmental flow requirement)                                                                                                                              |
| `outflow.max_outflow_m3s`    | f64 \| null    | Yes      | Maximum outflow (`null` = no flood control constraint)                                                                                                                              |
| `generation`                 | object         | Yes      | Tagged union — see Generation Model section above                                                                                                                                   |
| `tailrace`                   | object \| null | No       | Tailrace model (tagged union — see below). Omit for no tailrace adjustment.                                                                                                         |
| `hydraulic_losses`           | object \| null | No       | Hydraulic losses model (tagged union — see below). Omit for zero losses.                                                                                                            |
| `efficiency`                 | object \| null | No       | Turbine efficiency model (tagged union — see below). Omit to derive from productivity.                                                                                              |
| `evaporation`                | object \| null | No       | Evaporation coefficients. Omit for no evaporation modeling.                                                                                                                         |
| `penalties`                  | object         | No       | Entity-level penalty overrides (see [Penalty System](penalty-system.md) §1)                                                                                                         |

> \* Required when `filling` is not null. `filling_inflow_m3s` is optional (defaults to 0.0 — passive filling from natural inflows only). \*\* Required when `diversion` is not null.

### Diversion Channel

Diversion creates an additional flow variable `diversion_flow` bounded by `[0, max_flow_m3s]`. Diverted water is subtracted from the plant's balance and added to `diversion.downstream_id`'s inflow. It does NOT generate power. The `diversion_cost` is a **regularization cost** (see [Penalty System §2, Category 3](penalty-system.md)) that incentivizes keeping water in the main cascade.

### Filling Model

Dead-volume filling represents the commissioning period of a new hydro plant, during which the reservoir is being filled from an initial level (potentially zero) up to the dead volume (`min_storage_hm3`). This design is based on CEPEL's dead-volume filling model (`enchimento de volume morto`).

**Timeline**:

```
[start_stage_id]──── filling period ────[entry_stage_id]──── operating ────[exit_stage_id]
     enchendo                                operando
```

- Before `start_stage_id`: hydro does not exist
- `[start_stage_id, entry_stage_id)`: filling — storage can be below `min_storage_hm3`, no generation
- `[entry_stage_id, exit_stage_id)`: operating — normal constraints, generation active
- After `exit_stage_id`: decommissioned

**Filling target**: Always `min_storage_hm3`. The purpose of filling is to reach the dead volume so the hydro can begin operating. There is no user-configurable target — it is always the reservoir's minimum storage.

**Filling behavior**: During the filling period, the filling inflow determines the water retained for filling per stage. The entity-level `filling_inflow_m3s` provides a default value for all filling stages; per-stage overrides in `hydro_bounds` (see [Input Constraints §2](input-constraints.md)) replace it for specific stages. If neither is specified, filling inflow is 0.0 (passive filling — the reservoir fills only from natural stochastic inflows). The hydro has no generation (`turbined_flow = 0`, `generation = 0`). Outflow during filling is limited to spillage (once water reaches the spillway crest). See [Penalty System §7](penalty-system.md) for how filling interacts with outflow requirements, terminal filling constraints, and the penalty hierarchy.

> **Deferred: Bottom discharge** (`descargas de fundo`). CEPEL models bottom discharge outlets that allow water release when the reservoir level is below the spillway crest. This creates a conditional constraint (outflow capacity depends on whether the reservoir level is above/below the spillway crest), which requires either nonlinear or binary constraints — incompatible with LP-based SDDP. Bottom discharge is deferred to the simulation step only.

**Initial conditions**: Filling hydros use a separate `filling_storage` array in `initial_conditions.json` (see [Input Constraints §1](input-constraints.md)) because their initial volume can be below `min_storage_hm3`.

**Validation**: `start_stage_id` must be strictly less than `entry_stage_id`. If `entry_stage_id` is null, filling config is invalid (a filling hydro must eventually begin operating).

### Tailrace Model (Tagged Union) — Optional

Defines the downstream water level as a function of total outflow (turbined + spillage). Used by `linearized_head` and `fpha` (computed) models to determine net head.

**Variant: `polynomial`** — Tailrace level as polynomial function of outflow: `h_tail(Q_out) = c₀ + c₁·Q + c₂·Q² + ...`

| Field          | Type  | Required | Description                                                   |
| -------------- | ----- | -------- | ------------------------------------------------------------- |
| `type`         | str   | Yes      | Must be `"polynomial"`                                        |
| `coefficients` | [f64] | Yes      | Polynomial coefficients `[c₀, c₁, c₂, ...]` (ascending order) |

**Variant: `piecewise`** — Tailrace level as piecewise-linear function defined by (outflow, height) points with linear interpolation.

| Field    | Type                                    | Required | Description                                                                     |
| -------- | --------------------------------------- | -------- | ------------------------------------------------------------------------------- |
| `type`   | str                                     | Yes      | Must be `"piecewise"`                                                           |
| `points` | [{`outflow_m3s`: f64, `height_m`: f64}] | Yes      | Points defining the curve (must be sorted by outflow, monotonically increasing) |

**Fallback**: If `tailrace` is omitted, no tailrace adjustment is applied (only relevant for `linearized_head` and computed `fpha`).

### Hydraulic Losses Model (Tagged Union) — Optional

Defines head losses in the hydraulic circuit (intake, penstock, draft tube).

**Variant: `factor`** — Losses as a fraction (p.u.) of gross head: `h_loss = value × h_gross`

| Field   | Type | Required | Description                                             |
| ------- | ---- | -------- | ------------------------------------------------------- |
| `type`  | str  | Yes      | Must be `"factor"`                                      |
| `value` | f64  | Yes      | Loss factor in per-unit (e.g., 0.03 = 3% of gross head) |

**Variant: `constant`** — Fixed head loss in meters, independent of operating point.

| Field     | Type | Required | Description                  |
| --------- | ---- | -------- | ---------------------------- |
| `type`    | str  | Yes      | Must be `"constant"`         |
| `value_m` | f64  | Yes      | Constant head loss in meters |

**Fallback**: If `hydraulic_losses` is omitted, zero losses are assumed.

### Efficiency Model (Tagged Union) — Optional

Defines turbine-generator efficiency for production function computation.

**Variant: `constant`** — Single efficiency value across all operating points.

| Field   | Type | Required | Description                                 |
| ------- | ---- | -------- | ------------------------------------------- |
| `type`  | str  | Yes      | Must be `"constant"`                        |
| `value` | f64  | Yes      | Efficiency as a fraction (e.g., 0.92 = 92%) |

> **Future Variant**: `flow_dependent` — efficiency as a function of turbined flow. Not yet implemented.

**Fallback**: If `efficiency` is omitted, efficiency is implicitly embedded in `productivity_mw_per_m3s` (i.e., the productivity factor already accounts for average efficiency).

### Evaporation Coefficients — Optional

Defines monthly evaporation rates for the reservoir. The solver interpolates these 12 values based on the stage's date/season mapping to determine the evaporation rate for each stage. Combined with surface area from `hydro_geometry` (see [Input Hydro Extensions §1](input-hydro-extensions.md)), this determines evaporated volume.

| Field                   | Type              | Required | Description                                                                                                                                                                                                                           |
| ----------------------- | ----------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coefficients_mm`       | [f64; 12]         | Yes      | Monthly evaporation depth (mm), January through December                                                                                                                                                                              |
| `reference_volumes_hm3` | [f64; 12] or null | No       | Optional per-month reference volumes (hm3) for evaporation linearization. When null, the midpoint between min and max storage is used. Each value must be finite and lie within the plant's [min_storage_hm3, max_storage_hm3] range. |

**Fallback**: If `evaporation` is omitted, no evaporation is modeled for this hydro.

For the mathematical formulation of evaporation linearization, see [System Element Modeling](../math/system-elements.md) and [Hydro Production Functions](../math/hydro-production-models.md).

### Hydro Extensions

The following extension files augment the core hydro registry with additional data. They are documented in [Input Hydro Extensions](input-hydro-extensions.md):

| Data              | Purpose                                         |
| ----------------- | ----------------------------------------------- |
| Hydro geometry    | Volume-Height-Area relationship for evaporation |
| Production models | Stage-dependent production model selection      |
| FPHA hyperplanes  | Pre-computed FPHA hyperplane coefficients       |

> **Note on inflow models**: Inflow models are defined per hydro per stage in the scenarios directory, linked by `hydro_id`. See [Input Scenarios](input-scenarios.md). The file format for inflow model data is subject to the broader format discussion (see [Input Directory Structure](input-directory-structure.md)).

> **Future Extension — Generating Units**: The current schema defines generation capacity at the plant level (aggregate min/max turbined flow and generation bounds). A future enhancement may introduce a `units` field containing individual generating unit specifications (nominal power, turbined flow at nominal power, unit count, efficiency curves per unit). This would enable more accurate production function modeling and unit-level dispatch constraints.

> **Future Extension — CEPEL Advanced Flow Modeling**: Analysis of CEPEL documentation identified three potential input schema extensions that may be needed for more detailed hydro modeling. These are flagged for resolution during P2 math spec reviews:
>
> 1. **Lateral flow sources** (`Q_lat`): Some plants (e.g., Belo Monte/Pimental, Itaipu) receive lateral flows from river gauging posts or other plants' outflows that affect the tailwater level and production function. This would require a new field to specify lateral flow sources (e.g., `lateral_flow_sources: [{type, id}]`) and potentially a new flow variable in the LP.
> 2. **Downstream flow participation factors** (`Q_jus`): The downstream flow can include participation factors (`k_jus^Q`, `k_jus^S`, `k_jus^qa`, `k_jus^qd`) for turbined flow, spillage, lateral post inflows, and other plants' outflows. Our current `o = q + s` is the simplest case (all factors = 1.0). This would require a new `downstream_flow_factors` object in the hydro schema.
> 3. **Water travel time propagation curves**: Beyond simple translation delay (`τ_ij` implicit in `downstream_id`), CEPEL supports propagation curves where outflow arrives fractionally over `[τ_min, τ_max]` with participation percentages. This would require a `travel_time` configuration object (e.g., `{type: "simple", delay_stages: N}` or `{type: "propagation_curve", segments: [{delay, fraction}]}`), plus special end-of-horizon handling for water in transit.
>
> See [Deferred Features](../deferred.md) "Future Modeling Observations" for full details and CEPEL references.

For the mathematical formulation of hydro water balance, production function, and constraint details, see [System Element Modeling](../math/system-elements.md) and [Equipment Formulations](../math/equipment-formulations.md).

## 4. Thermal Registry (`system/thermals.json`)

> **Order Invariance**: The order of thermals in this array does NOT affect results. After loading, thermals are sorted by `id`. See [Design Principles §3](../overview/design-principles.md).

> **Format Rationale — thermals.json**
>
> Thermal plants are entity definitions with cost curve nesting. Each thermal has a piecewise-linear cost curve represented as an array of `cost_segments`, plus generation bounds and lifecycle attributes. JSON is a natural fit because cost curves are naturally nested arrays within each plant object.

> **Note**: Use `entry_stage_id` and `exit_stage_id` to model plants entering or exiting the system. Alternatively, stage-varying bounds can set generation to 0 for stages where the plant is offline (see [Input Constraints](input-constraints.md) for stage-varying override mechanisms; file format TBD).

### Thermal Operative States

| State            | Condition                                    | LP Variables           |
| ---------------- | -------------------------------------------- | ---------------------- |
| `non_existing`   | Before `entry_stage_id`                      | None                   |
| `operating`      | Between `entry_stage_id` and `exit_stage_id` | generation per segment |
| `decommissioned` | After `exit_stage_id`                        | None                   |

```json
{
  "$schema": "https://cobre.dev/schemas/v2/thermals.schema.json",
  "thermals": [
    {
      "id": 0,
      "name": "ANGRA1",
      "bus_id": 0,
      "entry_stage_id": null,
      "exit_stage_id": null,
      "cost_segments": [{ "capacity_mw": 640.0, "cost_per_mwh": 15.0 }],
      "generation": {
        "min_mw": 500.0,
        "max_mw": 640.0
      }
    }
  ]
}
```

### Thermal Fields

| Field                          | Type           | Required | Description                                             |
| ------------------------------ | -------------- | -------- | ------------------------------------------------------- |
| `id`                           | i32            | Yes      | Unique thermal identifier                               |
| `name`                         | string         | Yes      | Human-readable thermal name                             |
| `bus_id`                       | i32            | Yes      | Bus where generation is injected                        |
| `entry_stage_id`               | i32 \| null    | No       | Stage when thermal enters service (`null` = always)     |
| `exit_stage_id`                | i32 \| null    | No       | Stage when thermal is decommissioned (`null` = never)   |
| `cost_segments`                | array          | Yes      | Piecewise-linear cost curve (ordered by ascending cost) |
| `cost_segments[].capacity_mw`  | f64            | Yes      | Capacity of this cost segment (MW)                      |
| `cost_segments[].cost_per_mwh` | f64            | Yes      | Marginal cost in this segment (\$/MWh)                  |
| `generation.min_mw`            | f64            | Yes      | Minimum stable generation (0 if no minimum)             |
| `generation.max_mw`            | f64            | Yes      | Maximum generation capacity                             |
| `gnl_config`                   | object \| null | No       | GNL dispatch anticipation (see below)                   |

> **Thermal bounds**: Generation bounds (`min_mw`, `max_mw`) are hard constraints — no slack variables. Thermal dispatch is directly controllable (unlike hydro which depends on exogenous inflows), so if bounds cannot be met, this indicates a data error. See [Penalty System §4](penalty-system.md).
>
> **Stage-varying bounds**: Stage-varying thermal generation bounds can override the base values from `thermals.json`. The logical schema requires fields for `thermal_id`, `stage_id`, `min_generation_mw`, and `max_generation_mw` (nullable for partial overrides). The file format for stage-varying bounds is subject to the broader format discussion — see [Input Directory Structure](input-directory-structure.md).

### GNL Thermal Plants (Deferred)

> **Implementation Status**: Data model is ready. Implementation is planned but not yet complete.

GNL (Gas Natural Liquefeito) plants require **dispatch anticipation**: the dispatch decision must be made N stages ahead due to fuel ordering lead times. This creates additional state variables.

```json
{
  "id": 10,
  "name": "GNL_PLANT",
  "bus_id": 2,
  "gnl_config": {
    "lag_stages": 2
  },
  "cost_segments": [{ "capacity_mw": 500.0, "cost_per_mwh": 200.0 }],
  "generation": {
    "min_mw": 0.0,
    "max_mw": 500.0
  }
}
```

| Field                   | Type           | Description                                                 |
| ----------------------- | -------------- | ----------------------------------------------------------- |
| `gnl_config`            | object \| null | GNL configuration. If null or omitted, thermal is standard. |
| `gnl_config.lag_stages` | i32            | Number of stages ahead for dispatch decision.               |

When configured, the algorithm adds state variables for the committed dispatch pipeline:

| State Variable                   | Description                      |
| -------------------------------- | -------------------------------- |
| `gnl_committed[thermal_id, t+1]` | Dispatch committed for stage t+1 |
| `gnl_committed[thermal_id, t+2]` | Dispatch committed for stage t+2 |
| ...                              | ... up to `lag_stages` ahead     |

Initial values for the GNL pipeline are specified in `initial_conditions.json`.

> **Exceptional Validation Rule**: GNL thermals (thermals with `gnl_config` present) are currently **rejected by input validation**. The data model is fully specified to ensure coherence, but the GNL algorithm is not yet implemented. This validation rule will be relaxed when GNL support is complete. See [Internal Structures §4](internal-structures.md) for the runtime data model.

For the mathematical formulation of thermal cost curves and generation constraints, see [System Element Modeling](../math/system-elements.md).

## 5. Pumping Stations (`system/pumping_stations.json`) — Optional

> **Order Invariance**: The order of pumping stations in this array does NOT affect results. After loading, all stations are sorted by `id`. See [Design Principles §3](../overview/design-principles.md).

> **Format Rationale — pumping_stations.json**
>
> Small set of pump-turbine coupling definitions with cross-references to hydros and buses. JSON is natural for structured entities with unique IDs.

Pumping stations transfer water from a source reservoir to a destination reservoir, consuming electrical power. They are independent system elements — not a property of any specific hydro plant.

| Application             | Description                                                                    |
| ----------------------- | ------------------------------------------------------------------------------ |
| Pumped hydro storage    | Store energy by pumping water to upper reservoir during low-demand periods     |
| Inter-basin transfers   | Move water between river basins for irrigation or energy optimization          |
| Reversible hydro plants | Plants that can both generate and pump (model as hydro + pumping station pair) |

### Pumping Station Operative States

| State            | Condition                                    | LP Variables                   |
| ---------------- | -------------------------------------------- | ------------------------------ |
| `non_existing`   | Before `entry_stage_id`                      | None                           |
| `operating`      | Between `entry_stage_id` and `exit_stage_id` | pumped_flow, power_consumption |
| `decommissioned` | After `exit_stage_id`                        | None                           |

```json
{
  "$schema": "https://cobre.dev/schemas/v2/pumping_stations.schema.json",
  "pumping_stations": [
    {
      "id": 0,
      "name": "SANTA_CECILIA",
      "bus_id": 0,
      "source_hydro_id": 5,
      "destination_hydro_id": 10,
      "entry_stage_id": null,
      "exit_stage_id": null,
      "consumption_mw_per_m3s": 0.85,
      "flow": {
        "min_m3s": 0.0,
        "max_m3s": 150.0
      }
    }
  ]
}
```

### Pumping Station Fields

| Field                    | Type        | Required | Description                             |
| ------------------------ | ----------- | -------- | --------------------------------------- |
| `id`                     | i32         | Yes      | Unique station identifier               |
| `name`                   | string      | Yes      | Station name                            |
| `bus_id`                 | i32         | Yes      | Bus where power is consumed             |
| `source_hydro_id`        | i32         | Yes      | Hydro from which water is withdrawn     |
| `destination_hydro_id`   | i32         | Yes      | Hydro to which water is delivered       |
| `entry_stage_id`         | i32 \| null | No       | First operating stage (`null` = always) |
| `exit_stage_id`          | i32 \| null | No       | Last operating stage (`null` = forever) |
| `consumption_mw_per_m3s` | f64         | Yes      | Power consumption rate [MW/(m³/s)]      |
| `flow.min_m3s`           | f64         | Yes      | Minimum pumped flow                     |
| `flow.max_m3s`           | f64         | Yes      | Maximum pumped flow                     |

> **Cost note**: Pumping stations do **not** have a direct cost term in the objective function. The cost of pumping is implicitly captured through the energy consumed at the connected bus — pumping power appears as additional load in the bus energy balance. See [Penalty System §8](penalty-system.md) for details.

For the mathematical formulation of pumping constraints and water balance impact, see [System Element Modeling §7](../math/system-elements.md).

## 6. Energy Contracts (`system/energy_contracts.json`) — Optional

> **Order Invariance**: The order of contracts in this array does NOT affect results. After loading, all contracts are sorted by `id`. See [Design Principles §3](../overview/design-principles.md).

> **Format Rationale — energy_contracts.json**
>
> Contract definitions with nested structure and cross-references to buses. JSON is natural for structured entities with unique IDs.

Energy contracts represent agreements to buy (import) or sell (export) electricity with external systems (neighboring countries, bilateral agreements). Import adds to supply; export adds to demand in the bus load balance.

### Contract Operative States

| State            | Condition                                    | LP Variables                                             |
| ---------------- | -------------------------------------------- | -------------------------------------------------------- |
| `non_existing`   | Before `entry_stage_id`                      | None                                                     |
| `operating`      | Between `entry_stage_id` and `exit_stage_id` | One of `import_mw` or `export_mw` based on contract type |
| `decommissioned` | After `exit_stage_id`                        | None                                                     |

```json
{
  "$schema": "https://cobre.dev/schemas/v2/contracts.schema.json",
  "contracts": [
    {
      "id": 0,
      "name": "ITAIPU_BR",
      "bus_id": 0,
      "type": "import",
      "entry_stage_id": null,
      "exit_stage_id": null,
      "price_per_mwh": 50.0,
      "limits": {
        "min_mw": 0.0,
        "max_mw": 6000.0
      }
    },
    {
      "id": 1,
      "name": "ARGENTINA_EXPORT",
      "bus_id": 0,
      "type": "export",
      "entry_stage_id": null,
      "exit_stage_id": null,
      "price_per_mwh": -30.0,
      "limits": {
        "min_mw": 0.0,
        "max_mw": 2000.0
      }
    }
  ]
}
```

### Contract Fields

| Field            | Type        | Required | Description                                                        |
| ---------------- | ----------- | -------- | ------------------------------------------------------------------ |
| `id`             | i32         | Yes      | Unique contract identifier                                         |
| `name`           | string      | Yes      | Contract name                                                      |
| `bus_id`         | i32         | Yes      | Bus connected to contract                                          |
| `type`           | string      | Yes      | `"import"` (external to system) or `"export"` (system to external) |
| `entry_stage_id` | i32 \| null | No       | First active stage (`null` = always)                               |
| `exit_stage_id`  | i32 \| null | No       | Last active stage (`null` = forever)                               |
| `price_per_mwh`  | f64         | Yes      | Cost (import, positive) or revenue (export, typically negative)    |
| `limits.min_mw`  | f64         | Yes      | Minimum contract usage                                             |
| `limits.max_mw`  | f64         | Yes      | Maximum contract usage                                             |

> **Stage-varying bounds**: Contract limits and prices can be overridden per stage. The logical schema requires fields for `contract_id`, `stage_id`, `min_mw`, `max_mw`, and `price_per_mwh` (all nullable for partial overrides). The file format is subject to the broader format discussion — see [Input Directory Structure](input-directory-structure.md).

For the mathematical formulation of contract variables and constraints, see [System Element Modeling §8](../math/system-elements.md).

## 7. Non-Controllable Generation Sources (`system/non_controllable_sources.json`) — Optional

> **Order Invariance**: The order of sources in this array does NOT affect results. After loading, all sources are sorted by `id`. See [Design Principles §3](../overview/design-principles.md).

> **Format Rationale — non_controllable_sources.json**
>
> Small set of entity definitions with bus cross-references and lifecycle attributes. JSON is natural for structured entities with unique IDs.

### Concept

A **non-controllable source** represents intermittent generation (wind farms, solar plants, small run-of-river hydros, etc.) whose available output depends on external conditions (weather, river flow) rather than dispatch decisions. The solver receives a stochastic availability value per scenario from the scenario pipeline, and can only curtail generation below that availability — it cannot dispatch upward beyond what nature provides.

Non-controllable sources have near-zero marginal cost. The `curtailment_cost` is a **regularization penalty** (see [Penalty System §2, Category 3](penalty-system.md)) that incentivizes the solver to use all available generation before curtailing. Analogous to `spillage_cost` for hydros — curtailment discards available "free" energy.

### Non-Controllable Source Operative States

| State            | Condition                                    | LP Variables                      |
| ---------------- | -------------------------------------------- | --------------------------------- |
| `non_existing`   | Before `entry_stage_id`                      | None                              |
| `operating`      | Between `entry_stage_id` and `exit_stage_id` | generation, curtailment (derived) |
| `decommissioned` | After `exit_stage_id`                        | None                              |

### JSON Example

```json
{
  "$schema": "https://cobre.dev/schemas/v2/non_controllable_sources.schema.json",
  "non_controllable_sources": [
    {
      "id": 0,
      "name": "WIND_FARM_NE",
      "bus_id": 1,
      "entry_stage_id": null,
      "exit_stage_id": null,
      "max_generation_mw": 500.0,
      "curtailment_cost": 0.005
    },
    {
      "id": 1,
      "name": "SOLAR_SE",
      "bus_id": 0,
      "entry_stage_id": 12,
      "exit_stage_id": null,
      "max_generation_mw": 300.0
    }
  ]
}
```

> **Note on SOLAR_SE**: No `curtailment_cost` defined — uses global default from `penalties.json`.

### Non-Controllable Source Fields

| Field               | Type        | Required | Default | Description                                                                                                                               |
| ------------------- | ----------- | -------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `id`                | i32         | Yes      | —       | Unique source identifier                                                                                                                  |
| `name`              | string      | Yes      | —       | Human-readable source name                                                                                                                |
| `bus_id`            | i32         | Yes      | —       | Bus where generation is injected                                                                                                          |
| `entry_stage_id`    | i32 \| null | No       | null    | Stage when source enters service (`null` = always exists)                                                                                 |
| `exit_stage_id`     | i32 \| null | No       | null    | Stage when source is decommissioned (`null` = never)                                                                                      |
| `max_generation_mw` | f64         | Yes      | —       | Installed capacity (hard upper bound, physical limit)                                                                                     |
| `curtailment_cost`  | f64         | No       | global  | Regularization cost per MWh curtailed (uses `penalties.json` default if omitted). See [Penalty System §2, Category 3](penalty-system.md). |

### Generation and Curtailment

The generation variable for a non-controllable source is bounded by `[0, available_generation]`, where `available_generation` is the stochastic value from the scenario pipeline for the current (stage, scenario), itself bounded by `[0, max_generation_mw]`.

**Curtailment** is the difference: `curtailment = available_generation - generation`. It is not a separate LP decision variable — it is derived from the generation variable. The `curtailment_cost` in the objective penalizes `(available - generation)`, which is equivalent to giving the generation variable a negative regularization cost (reward for dispatching). Either formulation is valid; the implementation chooses whichever is simpler.

Non-controllable sources have **no stage-varying bounds** beyond the stochastic availability. Since they represent available generation that can only be curtailed, there are no meaningful operational bounds (min/max generation) to override per stage. Their availability comes entirely from the scenario model.

For the runtime in-memory representation, see [Internal Structures §9](internal-structures.md). For the mathematical formulation, see [System Element Modeling](../math/system-elements.md).

## Cross-References

- [Input Directory Structure](input-directory-structure.md) — case directory layout and `config.json` schema
- [Input Hydro Extensions](input-hydro-extensions.md) — geometry, production models, FPHA hyperplanes
- [Input Constraints](input-constraints.md) — time-varying bounds, generic constraints, stage-varying overrides
- [Internal Structures](internal-structures.md) — runtime in-memory data model, pre-resolved penalties and bounds
- [Penalty System](penalty-system.md) — three-tier penalty cascade, penalty categories, and all penalty schemas
- [Design Principles](../overview/design-principles.md) — format selection criteria and order invariance
- [System Element Modeling](../math/system-elements.md) — mathematical formulation of all system elements
- [Equipment Formulations](../math/equipment-formulations.md) — detailed per-equipment LP constraints
- [Hydro Production Models](../math/hydro-production-models.md) — FPHA and linearized head formulations
- [Deferred Features](../deferred.md) — battery storage, GNL implementation
