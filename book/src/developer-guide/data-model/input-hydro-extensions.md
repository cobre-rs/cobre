# Input Hydro Extensions

## Purpose

This spec defines the optional extension files for the hydro subsystem: reservoir geometry, production function model selection, and pre-computed FPHA hyperplanes. These files augment the core hydro registry defined in [Input System Entities](input-system-entities.md) and live under `system/` in the input case directory.

For other system elements (pumping stations, energy contracts, non-controllable sources), see [Input System Entities](input-system-entities.md).
For deferred features (battery storage), see [Deferred Features](../deferred.md).

## 1. Hydro Geometry (`system/hydro_geometry.parquet`) — Optional

> **Format Rationale**
>
> **Entity-level lookup table** — Per-hydro tabular data (Volume-Height-Area curves) with multiple rows per hydro forming a static physical relationship. Parquet for typed columnar data with efficient per-hydro filtering.

Defines the Volume-Height-Area relationship for reservoirs. This data is used for:

- **Evaporation modeling**: Surface area as a function of storage determines evaporated volume (see evaporation coefficients in [Input System Entities §3](input-system-entities.md))
- **Forebay level**: Storage-to-height relationship for production function models that account for head variation (`linearized_head`, `fpha`)

Instead of polynomial fits, Cobre uses a tabular approach with linear interpolation — more transparent and easier to validate against surveyed data.

For the mathematical formulation of evaporation linearization and forebay level computation, see [Hydro Production Functions](../math/hydro-production-models.md) and [System Element Modeling](../math/system-elements.md).

### Schema

| Column       | Type | Description                                   |
| ------------ | ---- | --------------------------------------------- |
| `hydro_id`   | i32  | Hydro plant identifier                        |
| `volume_hm3` | f64  | Total volume (hm³) — must include dead volume |
| `height_m`   | f64  | Reservoir surface elevation (m)               |
| `area_km2`   | f64  | Water surface area (km²)                      |

**Example rows (Sobradinho):**

| hydro_id | volume_hm3 | height_m | area_km2 |
| -------- | ---------- | -------- | -------- |
| 42       | 5447.0     | 380.0    | 800.0    |
| 42       | 8000.0     | 385.0    | 1200.0   |
| 42       | 12500.0    | 390.0    | 2000.0   |
| 42       | 18000.0    | 395.0    | 3000.0   |
| 42       | 28000.0    | 400.0    | 4200.0   |

### Validation

| Rule                 | Description                                                                                                           |
| -------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Volumes              | Must be monotonically increasing per hydro                                                                            |
| Heights              | Must be monotonically increasing with volume                                                                          |
| Areas                | Must be monotonically increasing with height                                                                          |
| Minimum volume entry | Should be at or below `min_storage_hm3`                                                                               |
| Maximum volume entry | Should be at or above `max_storage_hm3`                                                                               |
| Dead volume geometry | Geometry below `min_storage_hm3` is optional; if not provided, filling evaporation uses geometry at `min_storage_hm3` |

## 2. Hydro Production Models (`system/hydro_production_models.json`) — Optional

> **Format Rationale**
>
> **Registry** — Small config selecting model type per hydro with nested optional params and stage/season configuration. JSON handles optional/nested structures well.

Configures the hydro production function (HPF) modeling approach per hydro. Different stages or seasons can use different accuracy levels. See [Hydro Production Functions](../math/hydro-production-models.md) for the mathematical formulation.

**Model Hierarchy** (increasing complexity and accuracy):

1. **`constant_productivity`**: Fixed productivity factor — single multiplication, fastest
2. **`linearized_head`**: Accounts for head variation with storage — requires geometry data. **Simulation-only** — excluded from training because the bilinear term changes the LP between iterations, breaking SDDP convergence (see [Hydro Production Models §3](../math/hydro-production-models.md))
3. **`fpha`**: Full piecewise-linear approximation via hyperplanes — most accurate

See [Hydro Production Functions §4](../math/hydro-production-models.md) for model selection guidelines and accuracy trade-offs.

**Default Behavior**: If this file is not provided or a hydro is not listed, the model uses the `generation.model` field from `hydros.json` for all stages.

### Selection Modes (Tagged Union)

The `selection_mode` field determines how the model is chosen per stage:

**Mode: `stage_ranges`** — Explicit stage ranges (structural changes across the horizon)

```json
{
  "$schema": "https://cobre.dev/schemas/v2/production_models.schema.json",
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
            "turbine_discretization_points": 15,
            "fitting_window": {
              "volume_min_hm3": null,
              "volume_max_hm3": null
            }
          }
        },
        {
          "start_stage_id": 25,
          "end_stage_id": null,
          "model": "constant_productivity"
        }
      ]
    }
  ]
}
```

**Mode: `seasonal`** — Model selection keyed by season index, cycling across the horizon

Each stage is mapped to its season via the stage-to-season mapping defined in `stages.json` (see [Input Scenarios](input-scenarios.md)). For a monthly study, seasons 0–11 correspond to January–December. This mode is useful for studies where accuracy requirements vary cyclically (e.g., wet season uses FPHA, dry season uses linearized head).

```json
{
  "$schema": "https://cobre.dev/schemas/v2/production_models.schema.json",
  "production_models": [
    {
      "hydro_id": 5,
      "selection_mode": "seasonal",
      "default_model": "linearized_head",
      "seasons": [
        {
          "season_id": 0,
          "model": "fpha",
          "fpha_config": {
            "source": "computed",
            "volume_discretization_points": 5,
            "turbine_discretization_points": 10
          }
        },
        {
          "season_id": 1,
          "model": "fpha",
          "fpha_config": {
            "source": "computed",
            "volume_discretization_points": 5,
            "turbine_discretization_points": 10
          }
        }
      ]
    }
  ]
}
```

In this example, seasons 0 and 1 (e.g., January and February — wet season) use FPHA; all other seasons fall back to `default_model` (linearized head).

> **Note**: This example uses `linearized_head` as the default model, which restricts this hydro's configuration to **simulation only**. For training, replace the `default_model` with `constant_productivity` or `fpha`. See [Hydro Production Models §3](../math/hydro-production-models.md).

**Combined usage**: Different hydros in the same file can use different selection modes. Stage ranges and seasonal modes can coexist across hydros but not within a single hydro.

### Selection Mode Fields

| Field            | Type   | Required     | Description                                              |
| ---------------- | ------ | ------------ | -------------------------------------------------------- |
| `hydro_id`       | i32    | Yes          | Hydro plant identifier                                   |
| `selection_mode` | string | Yes          | `"stage_ranges"` or `"seasonal"`                         |
| `stage_ranges`   | array  | If mode = SR | Array of stage range objects (see below)                 |
| `default_model`  | string | If mode = S  | Fallback model for seasons not listed in `seasons` array |
| `seasons`        | array  | If mode = S  | Array of season-specific model configs (see below)       |

### Stage Range Fields

| Field            | Type        | Required | Description                                                                   |
| ---------------- | ----------- | -------- | ----------------------------------------------------------------------------- |
| `start_stage_id` | i32         | Yes      | First stage in range (inclusive)                                              |
| `end_stage_id`   | i32 \| null | Yes      | Last stage in range (`null` = until end)                                      |
| `model`          | string      | Yes      | `"constant_productivity"`, `"linearized_head"` (simulation-only), or `"fpha"` |
| `fpha_config`    | object      | If fpha  | FPHA configuration (see below)                                                |

### Season Fields

| Field         | Type   | Required | Description                                                                   |
| ------------- | ------ | -------- | ----------------------------------------------------------------------------- |
| `season_id`   | i32    | Yes      | Season index (0-based, matching `stages.json` season map)                     |
| `model`       | string | Yes      | `"constant_productivity"`, `"linearized_head"` (simulation-only), or `"fpha"` |
| `fpha_config` | object | If fpha  | FPHA configuration (see below)                                                |

### FPHA Configuration Fields

| Field                                  | Type   | Description                                                                           |
| -------------------------------------- | ------ | ------------------------------------------------------------------------------------- |
| `source`                               | string | `"computed"` (fit from topology) or `"precomputed"` (from `fpha_hyperplanes.parquet`) |
| `volume_discretization_points`         | i32    | Number of volume points in grid (default: 5)                                          |
| `turbine_discretization_points`        | i32    | Number of turbine flow points (default: 5)                                            |
| `spillage_discretization_points`       | i32    | Number of spillage flow points in grid (default: 5)                                   |
| `max_planes_per_hydro`                 | i32    | Maximum hyperplanes retained after heuristic selection (default: 10)                  |
| `fitting_window.volume_min_hm3`        | f64?   | Explicit minimum volume for fitting (null = physical min)                             |
| `fitting_window.volume_max_hm3`        | f64?   | Explicit maximum volume for fitting (null = physical max)                             |
| `fitting_window.volume_min_percentile` | f64?   | Minimum as percentile of operating range                                              |
| `fitting_window.volume_max_percentile` | f64?   | Maximum as percentile of operating range                                              |

Use absolute bounds (`volume_min_hm3`, `volume_max_hm3`) OR percentiles, not both. Percentiles are relative to `[min_storage_hm3, max_storage_hm3]` from `hydros.json`.

> **Implementation note (v0.1.4):** `spillage_discretization_points` and `max_planes_per_hydro` were added in v0.1.4 when the computed-source FPHA pipeline was implemented. The default for `turbine_discretization_points` is 5 (not 10 as originally specced); both defaults produce grids sufficient for typical reservoir configurations. All discretization counts must be >= 2; `max_planes_per_hydro` must be >= 1. The computed source path is fully implemented — no preprocessing step or external tool is required. Hyperplanes are fitted during the Initialization phase on rank 0 and written to `output/hydro_models/fpha_hyperplanes.parquet`.

### Required Data by Model

| Model                   | `hydros.json`              | `hydro_geometry` | `fpha_hyperplanes` |
| ----------------------- | -------------------------- | ---------------- | ------------------ |
| `constant_productivity` | `productivity_mw_per_m3s`  | ✗                | ✗                  |
| `linearized_head`       | `productivity_mw_per_m3s`  | ✓                | ✗                  |
| `fpha` (computed)       | `productivity_mw_per_m3s`¹ | ✓                | ✗                  |
| `fpha` (precomputed)    | `productivity_mw_per_m3s`¹ | ✗                | ✓                  |

¹ Used as fallback for stages without FPHA configuration.

> **Implementation note (v0.1.4):** The `"computed"` source path is implemented. Fitting runs during Initialization on MPI rank 0 from `hydro_geometry.parquet` and the `tailrace`, `hydraulic_losses`, and `efficiency` fields in `hydros.json`. Fitted planes are exported to `output/hydro_models/fpha_hyperplanes.parquet` using the same 11-column schema as the input file, enabling round-trip use in subsequent runs as `"precomputed"` planes.

> **Note**: For computed FPHA, the solver also uses the optional `tailrace`, `hydraulic_losses`, and `efficiency` fields from the hydro object in `hydros.json` (see [Input System Entities §3](input-system-entities.md)). If these fields are omitted, fallback assumptions apply (no tailrace adjustment, zero losses, efficiency derived from productivity).

## 3. FPHA Hyperplanes (`system/fpha_hyperplanes.parquet`) — Optional

> **Format Rationale**
>
> **Pre-computed coefficient table** — Per-hydro tabular data with many rows of hyperplane coefficients, potentially varying by stage. Parquet for typed columnar data with efficient per-hydro filtering.

Pre-computed FPHA hyperplane coefficients for hydro production function modeling. Allows using externally-fitted planes instead of computing them at runtime. Different stages can have different plane sets per hydro (e.g., near-term fitting with more detail vs. far-term with fewer planes).

| Use Case                 | Description                                             |
| ------------------------ | ------------------------------------------------------- |
| Legacy system migration  | Import FPHA coefficients from DECOMP/DESSEM input files |
| External calibration     | Use coefficients fitted by specialized tools            |
| Performance optimization | Skip runtime fitting for large systems                  |

If not provided, Cobre computes hyperplanes from `hydro_geometry` data and the hydro object's tailrace/losses/efficiency fields during preprocessing.

### Schema

| Column            | Type        | Required | Description                                                                                                                           |
| ----------------- | ----------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `hydro_id`        | i32         | Yes      | Hydro plant identifier                                                                                                                |
| `stage_id`        | i32 \| null | No       | Stage this plane set applies to (`null` = valid for all stages)                                                                       |
| `plane_id`        | i32         | Yes      | Plane index within hydro (0 to M-1)                                                                                                   |
| `gamma_0`         | f64         | Yes      | Intercept coefficient (MW)                                                                                                            |
| `gamma_v`         | f64         | Yes      | Volume coefficient (MW/hm³)                                                                                                           |
| `gamma_q`         | f64         | Yes      | Turbined flow coefficient (MW per m³/s)                                                                                               |
| `gamma_s`         | f64         | Yes      | Spillage coefficient (MW per m³/s, typically ≤ 0)                                                                                     |
| `kappa`           | f64         | No       | Correction factor κ (default: 1.0). See [Hydro Production Functions](../math/hydro-production-models.md) for mathematical definition. |
| `valid_v_min_hm3` | f64         | No       | Volume range minimum where plane is valid                                                                                             |
| `valid_v_max_hm3` | f64         | No       | Volume range maximum where plane is valid                                                                                             |
| `valid_q_max_m3s` | f64         | No       | Maximum turbined flow where plane is valid                                                                                            |

**Example rows (Itaipu):**

| hydro_id | stage_id | plane_id | gamma_0 | gamma_v | gamma_q | gamma_s | kappa |
| -------- | -------- | -------- | ------- | ------- | ------- | ------- | ----- |
| 66       | null     | 0        | 1250.5  | 0.0023  | 0.892   | -0.015  | 0.985 |
| 66       | null     | 1        | 1180.2  | 0.0031  | 0.875   | -0.012  | 0.985 |
| 66       | null     | 2        | 1320.8  | 0.0018  | 0.901   | -0.018  | 0.985 |
| 66       | null     | 3        | 1095.4  | 0.0042  | 0.858   | -0.010  | 0.985 |
| 66       | null     | 4        | 1410.1  | 0.0012  | 0.915   | -0.022  | 0.985 |

For the mathematical formulation of how these hyperplanes define the production function constraint, see [Hydro Production Functions](../math/hydro-production-models.md).

### Validation

| Rule            | Description                                                                    |
| --------------- | ------------------------------------------------------------------------------ |
| Minimum planes  | Each `hydro_id` (per `stage_id`) should have at least 3 planes                 |
| Typical range   | 5–30 planes per hydro per stage                                                |
| `gamma_v`       | Should be positive (higher storage → higher generation)                        |
| `gamma_q`       | Should be positive (more flow → more generation)                               |
| `gamma_s`       | Should be negative or zero (spillage reduces effective head)                   |
| Validity ranges | If provided, planes are only activated when hydro is within range              |
| Stage coverage  | If `stage_id` is used, stages without specific planes fall back to `null` rows |

## Cross-References

- [Input System Entities](input-system-entities.md) — Core hydro registry (`hydros.json`) with optional tailrace, hydraulic losses, efficiency, and evaporation fields; pumping stations, energy contracts, and all other system elements
- [Input Directory Structure](input-directory-structure.md) — Overall case directory layout
- [Input Constraints](input-constraints.md) — Time-varying bounds for hydros, thermals, lines, and contracts
- [Hydro Production Functions](../math/hydro-production-models.md) — Mathematical formulation of HPF models, FPHA constraint form, correction factor κ
- [LP Formulation](../math/lp-formulation.md) — How hydro variables enter the LP
- [System Element Modeling](../math/system-elements.md) — Evaporation linearization, forebay level computation
- [Deferred Features](../deferred.md) — Battery storage
- [Design Principles §3](../overview/design-principles.md) — Order invariance and canonical ordering
