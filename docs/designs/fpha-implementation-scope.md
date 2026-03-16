# Hydro Modeling Implementation — Complete Scope Design

**Date:** 2026-03-15 (updated 2026-03-16)
**Author:** Rogerio + Claude
**Status:** Draft — pending user validation

---

## 1. Design Principles (Lessons from Stochastic Preprocessing)

The stochastic preprocessing pipeline taught us four lessons that apply directly here:

1. **Provenance from day 1.** The stochastic summary used heuristics to guess where the opening tree and correlation came from (`max_openings > 1`). This was wrong in multiple cases and required `StochasticProvenance` as a retroactive fix. FPHA and evaporation must record provenance at construction time: was each hydro's hyperplane set loaded from file or computed from geometry? Were evaporation coefficients linearized from geometry or absent?

2. **Detect → Validate → Process → Summarize as a single pipeline.** The stochastic preprocessing was initially scattered across CLI and Python with ad-hoc branches. `prepare_stochastic()` fixed this by packaging the entire flow. Hydro modeling needs `prepare_hydro_models()` from the start, covering both production models and evaporation in one pass.

3. **Summary via Term, not random prints.** The `[stochastic]` `eprintln!` pattern was replaced with `StochasticSummary` + `print_stochastic_summary()` using `Term::stderr()` and `console::style`. Hydro model preprocessing must follow the same pattern.

4. **Shared function in cobre-sddp, callable from CLI and Python.** The orchestration logic must live in `cobre-sddp`, not be duplicated across entry points.

---

## 2. Scope: Two Hydro Modeling Gaps

This design covers **two distinct but coupled hydro modeling features** that share infrastructure (geometry data, preprocessing pipeline, LP builder changes):

### 2.1 FPHA (Production Function Approximation)

Replaces the constant productivity model `g = ρ·q` with a piecewise-linear approximation via tangent hyperplanes: `g ≤ γ₀ + γ_v·v_avg + γ_q·q + γ_s·s`. Required for any hydro with significant head variation (all NEWAVE plants).

**Current status:** Data model exists. LP builder rejects FPHA with a validation error. Pre-fitted hyperplane loading (cobre-io) is complete. Production model configuration parsing is complete.

### 2.2 Evaporation (Linearized Reservoir Evaporation)

Models water loss from reservoir surface evaporation in the water balance:
`Q_ev = k_evap0 + k_evapV · v`. Required for accurate multi-month reservoir simulation where evaporation can be 5-15% of a reservoir's inflow.

**Current status:** Data model exists (`evaporation_coefficients_mm: Option<[f64; 12]>` on `Hydro`, `evaporation_violation_cost` on penalties). Geometry data (`area_km2` in `hydro_geometry.parquet`) exists. The LP builder **completely ignores evaporation** — when a user provides evaporation coefficients, the data is silently discarded. This is the same class of bug as the original B4 (FPHA silent zero).

### 2.3 Why They're Coupled

Both features share:

- **`hydro_geometry.parquet`** dependency — FPHA uses height(v) for net head; evaporation uses area(v) for surface area
- **LP builder changes** — both add constraints/variables to `build_stage_templates`
- **Preprocessing pipeline** — both follow detect → validate → linearize/load → provenance → summary
- **`v_in` patching** — both depend on incoming storage (FPHA via `v_avg`, evaporation via its linearization around a reference volume)

Implementing them together avoids touching the LP builder twice with similar changes.

---

## 3. Architecture Overview

### What Exists Today

| Component                                              | Location                                           | Status                          |
| ------------------------------------------------------ | -------------------------------------------------- | ------------------------------- |
| `HydroGenerationModel::Fpha` enum variant              | `cobre-core/src/entities/hydro.rs`                 | Exists                          |
| `TailraceModel` (polynomial, piecewise-linear)         | `cobre-core/src/entities/hydro.rs`                 | Exists                          |
| `HydraulicLossesModel` (factor, constant)              | `cobre-core/src/entities/hydro.rs`                 | Exists                          |
| `evaporation_coefficients_mm: Option<[f64; 12]>`       | `cobre-core/src/entities/hydro.rs`                 | Exists — **silently ignored**   |
| `evaporation_violation_cost`                           | `cobre-core/src/penalty.rs`                        | Exists                          |
| `hydro_geometry.parquet` loader (volume, height, area) | `cobre-io/src/extensions/`                         | Exists                          |
| `FphaHyperplaneRow` + `parse_fpha_hyperplanes()`       | `cobre-io/src/extensions/fpha_hyperplanes.rs`      | Exists, well-tested             |
| `ProductionModelConfig` + `parse_production_models()`  | `cobre-io/src/extensions/production_models.rs`     | Exists, well-tested             |
| `fpha_turbined_cost` penalty field                     | `cobre-core/src/penalty.rs`                        | Exists                          |
| FPHA validation error (B4 fix)                         | `cobre-sddp/src/lp_builder.rs:1275`                | Exists — rejects FPHA           |
| FPHA spec (math, LP integration, cuts)                 | `cobre-docs/specs/math/hydro-production-models.md` | Complete                        |
| LP spec water balance with evaporation term            | `cobre-docs/specs/math/lp-formulation.md` §4       | Specified — **not implemented** |

### What Needs to Be Built

```
Phase 1: Hydro Model Preprocessing (the "prepare_hydro_models" pipeline)
  ├── FPHA:
  │   ├── Resolve per-hydro per-stage model assignments
  │   ├── Load pre-fitted hyperplanes (source: "precomputed")
  │   ├── Detect & reject computed fitting (source: "computed") — error for now
  │   └── Validate hyperplanes against hydro physical constraints
  ├── Evaporation:
  │   ├── Detect which hydros have evaporation coefficients
  │   ├── Load geometry (area-volume curve) for those hydros
  │   ├── Compute linearized evaporation coefficients (k_evap0, k_evapV)
  │   └── Validate coefficients (physical range checks)
  ├── Track provenance for both concerns
  ├── Build HydroModelSet (production + evaporation, runtime-ready)
  └── Produce HydroModelSummary for display

Phase 2: LP Integration
  ├── FPHA:
  │   ├── Modify build_stage_templates to add FPHA constraint rows
  │   ├── Modify StageIndexer to track FPHA row ranges
  │   ├── Add fpha_turbined_cost to objective
  │   └── Handle v_avg = (v_in + v)/2 in FPHA constraints
  ├── Evaporation:
  │   ├── Add evaporation variable (Q_ev) per hydro per block
  │   ├── Add evaporation constraint: Q_ev = k_evap0 + k_evapV/2 · v_in + k_evapV/2 · v
  │   ├── Add evaporation violation slacks (bidirectional)
  │   ├── Include evaporation in water balance
  │   └── Add evaporation violation cost to objective
  └── Remove the FPHA validation rejection

Phase 3: Result Extraction & End-to-End Validation
  ├── Extract evaporation and FPHA results in extract_stage_result
  ├── Ensure simulation pipeline handles the new templates
  ├── Add test case with FPHA + evaporation
  └── End-to-end convergence verification

Phase 4 (DEFERRED): Hyperplane Fitting from Geometry
  ├── Evaluate exact production function on grid
  ├── Construct concave envelope
  ├── Select representative hyperplanes
  ├── Compute correction factor κ
  └── Export fitted hyperplanes to output/
```

---

## 4. Phase 1: Hydro Model Preprocessing

### 4.1 The `prepare_hydro_models` Function

**Location:** `crates/cobre-sddp/src/hydro_models.rs` (new module)

**Parallel to:** `prepare_stochastic()` in `setup.rs`

```rust
/// Result of the hydro model preprocessing pipeline.
pub struct PrepareHydroModelsResult {
    /// Per-hydro, per-stage resolved production model assignments (FPHA / constant).
    pub production: ProductionModelSet,
    /// Per-hydro linearized evaporation coefficients (None if hydro has no evaporation).
    pub evaporation: EvaporationModelSet,
    /// Provenance metadata for summary display.
    pub provenance: HydroModelProvenance,
}

/// Prepare the hydro model pipeline: resolve production models (FPHA / constant),
/// linearize evaporation from geometry, validate, and track provenance.
pub fn prepare_hydro_models(
    system: &System,
    case_dir: &Path,
    config: &Config,
) -> Result<PrepareHydroModelsResult, SddpError>;
```

### 4.2 Detection and Branching Logic

The function processes both concerns in a single pass over `system.hydros()`:

```
For each hydro in system.hydros():

  ── PRODUCTION MODEL ──
  1. Check if hydro_production_models.json has an entry for this hydro
     ├── YES, source: "precomputed" → load planes from fpha_hyperplanes.parquet
     ├── YES, source: "computed"    → ERROR (not implemented yet)
     └── NO  → use HydroGenerationModel from hydros.json
               ├── ConstantProductivity → use as-is
               ├── Fpha without config  → ERROR (need hyperplane source)
               └── LinearizedHead       → allowed only in simulation stages

  ── EVAPORATION ──
  2. Check if hydro.evaporation_coefficients_mm is Some
     ├── YES → load hydro_geometry.parquet for this hydro
     │         ├── geometry found → compute linearized coefficients
     │         └── geometry missing → ERROR: evaporation requires geometry
     └── NO  → no evaporation for this hydro (EvaporationModel::None)
```

### 4.3 Evaporation Linearization

#### The Nonlinear Evaporation Model

The exact evaporation flow for hydro $h$ at stage $t$ (month $j$) is:

$$Q_{ev,h}(v) = \zeta_{evap} \cdot c_{ev}(h, j) \cdot A(v)$$

where:

- $c_{ev}(h, j)$ = monthly evaporation coefficient (mm/month) from `evaporation_coefficients_mm[j]`
- $A(v)$ = reservoir surface area (km²) as function of storage volume, from `hydro_geometry.parquet`
- $\zeta_{evap} = \frac{1}{3.6 \times \text{stage\_hours}}$ = unit conversion factor (mm·km² → m³/s)

The surface area $A(v)$ is nonlinear in $v$ (the area-volume curve from geometry data). For LP formulation, we need a linear approximation.

#### First-Order Taylor Linearization

Linearize around a reference volume $v_{ref}$ (typically the midpoint of the operating range):

$$Q_{ev,h}(v) \approx k_{evap0,h,t} + k_{evapV,h,t} \cdot v$$

where:

$$k_{evapV,h,t} = \zeta_{evap} \cdot c_{ev}(h, j(t)) \cdot \left.\frac{\partial A}{\partial v}\right|_{v_{ref}}$$

$$k_{evap0,h,t} = \zeta_{evap} \cdot c_{ev}(h, j(t)) \cdot A(v_{ref}) - k_{evapV,h,t} \cdot v_{ref}$$

The derivative $\partial A / \partial v$ is computed from the geometry table by finite differences between adjacent points, or analytically if a polynomial fit is used.

#### Reference Volume Selection

The reference volume $v_{ref}$ for linearization:

- **Default:** midpoint of the operating range: $v_{ref} = (v_{min} + v_{max}) / 2$
- **Seasonal (future):** stage-specific reference from historical average storage, improving accuracy for reservoirs with strong seasonal patterns

#### Per-Stage Coefficient Variation

Evaporation coefficients vary by stage because:

1. The monthly evaporation rate $c_{ev}(h, j)$ changes with the calendar month mapped to each stage
2. The stage duration (hours) affects the $\zeta_{evap}$ conversion factor
3. (Future) the reference volume may vary seasonally

The result is a pair $(k_{evap0}, k_{evapV})$ per hydro per stage.

#### Handling Negative Evaporation Coefficients

From the CEPEL reference: monthly coefficients can be negative when precipitation exceeds gross evaporation. This is physically valid — a negative coefficient means the reservoir gains water from rainfall. The linearization handles this naturally: $k_{evapV}$ can be negative (net inflow from rainfall reduces with volume because area increases) and $k_{evap0}$ adjusts accordingly. No special casing needed.

### 4.4 Evaporation Data Structures

```rust
/// Linearized evaporation coefficients for one hydro at one stage.
#[derive(Debug, Clone, Copy)]
pub struct LinearizedEvaporation {
    /// Intercept coefficient (m³/s). The evaporation flow at zero storage.
    pub k_evap0: f64,
    /// Volume slope coefficient (m³/s per hm³). Change in evaporation per
    /// unit change in storage volume.
    pub k_evap_v: f64,
}

/// Per-hydro evaporation model for all stages.
pub enum EvaporationModel {
    /// No evaporation modeled for this hydro.
    None,
    /// Linearized evaporation with per-stage coefficients.
    Linearized {
        /// One pair of coefficients per stage (matching template count).
        coefficients: Vec<LinearizedEvaporation>,
        /// Reference volume used for linearization (hm³).
        reference_volume_hm3: f64,
    },
}

/// Runtime-ready evaporation data for all hydros.
pub struct EvaporationModelSet {
    /// Per-hydro evaporation model, indexed by hydro position in system.hydros().
    models: Vec<EvaporationModel>,
}
```

### 4.5 Provenance Tracking

```rust
/// How a hydro's production model was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductionModelSource {
    /// Constant productivity from hydros.json (default, no FPHA data).
    DefaultConstant,
    /// FPHA hyperplanes loaded from fpha_hyperplanes.parquet.
    PrecomputedHyperplanes,
    /// FPHA hyperplanes fitted from geometry data (future).
    ComputedFromGeometry,
}

/// How a hydro's evaporation model was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaporationSource {
    /// No evaporation modeled (coefficients absent in hydros.json).
    NotModeled,
    /// Linearized from monthly coefficients + geometry area-volume curve.
    LinearizedFromGeometry,
}

/// Combined provenance for the hydro model preprocessing pipeline.
#[derive(Debug, Clone)]
pub struct HydroModelProvenance {
    /// Per-hydro production model source.
    pub production_sources: Vec<(EntityId, ProductionModelSource)>,
    /// Per-hydro evaporation source.
    pub evaporation_sources: Vec<(EntityId, EvaporationSource)>,
}
```

**Key design decision:** Provenance lives in `cobre-sddp` (not `cobre-core` or `cobre-stochastic`) because production model resolution and evaporation linearization involve algorithm-level decisions. This is different from `StochasticProvenance` which is a generic concept in an infrastructure crate.

### 4.6 The `ProductionModelSet` Runtime Structure

```rust
/// Runtime-ready production model data for all hydros across all stages.
///
/// Built once during preprocessing, consumed by `build_stage_templates`.
pub struct ProductionModelSet {
    /// Per-hydro resolved model for each stage.
    /// Indexed as models[hydro_index][stage_index].
    /// Only the non-pre-study stages are included (matching template count).
    stage_models: Vec<Vec<ResolvedProductionModel>>,
}

/// A fully resolved production model for one hydro at one stage.
pub enum ResolvedProductionModel {
    /// g = ρ × q (single equality constraint).
    ConstantProductivity { productivity: f64 },
    /// g ≤ γ₀ + γ_v·v_avg + γ_q·q + γ_s·s (M inequality constraints).
    Fpha {
        planes: Vec<FphaPlane>,
        turbined_cost: f64,
    },
}

/// A single FPHA hyperplane with pre-scaled intercept.
pub struct FphaPlane {
    /// Pre-scaled intercept: κ × γ₀ (MW).
    pub gamma_0: f64,
    /// Volume coefficient (MW/hm³).
    pub gamma_v: f64,
    /// Turbined flow coefficient (MW per m³/s).
    pub gamma_q: f64,
    /// Spillage coefficient (MW per m³/s, ≤ 0).
    pub gamma_s: f64,
}
```

### 4.7 Validation

| Check                                                                                                                                                                                       | Where                  | When    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | ------- |
| **FPHA**                                                                                                                                                                                    |                        |         |
| `source: "computed"` → error (not implemented)                                                                                                                                              | `prepare_hydro_models` | Phase 1 |
| FPHA hydros have ≥ 1 hyperplane per stage                                                                                                                                                   | `prepare_hydro_models` | Phase 1 |
| `gamma_v > 0` (physical: higher storage → more generation)                                                                                                                                  | `prepare_hydro_models` | Phase 1 |
| `gamma_s ≤ 0` (physical: spillage reduces generation)                                                                                                                                       | `prepare_hydro_models` | Phase 1 |
| `gamma_q > 0` (physical: more flow → more generation)                                                                                                                                       | `prepare_hydro_models` | Phase 1 |
| `kappa ∈ (0, 1]` (correction factor range)                                                                                                                                                  | `prepare_hydro_models` | Phase 1 |
| Stage coverage: no gaps in stage range assignments                                                                                                                                          | `prepare_hydro_models` | Phase 1 |
| FPHA hydros have geometry data if `source: "computed"`                                                                                                                                      | Deferred to Phase 4    | Phase 4 |
| **Evaporation**                                                                                                                                                                             |                        |         |
| Hydros with `evaporation_coefficients_mm` have geometry data                                                                                                                                | `prepare_hydro_models` | Phase 1 |
| Geometry table has `area_km2` column (not just volume + height)                                                                                                                             | `prepare_hydro_models` | Phase 1 |
| Linearized `k_evapV` is finite (no NaN/Inf from degenerate geometry)                                                                                                                        | `prepare_hydro_models` | Phase 1 |
| Stage-to-month mapping is consistent (each stage maps to a valid month index)                                                                                                               | `prepare_hydro_models` | Phase 1 |
| Hydros with evaporation but `EvaporationModel::None` → warning (coefficients present but geometry missing would be an error, not a warning — the user explicitly provided evaporation data) | `prepare_hydro_models` | Phase 1 |

### 4.8 Summary Display

**Location:** `crates/cobre-sddp/src/hydro_models.rs` (data) + `crates/cobre-cli/src/summary.rs` (display)

```rust
/// Summary of hydro model preprocessing for display.
pub struct HydroModelSummary {
    // ── Production ──
    /// Number of hydros using constant productivity (all stages).
    pub n_constant: usize,
    /// Number of hydros using FPHA (at least one stage).
    pub n_fpha: usize,
    /// Total hyperplane count across all FPHA hydros.
    pub total_planes: usize,
    /// Per-FPHA-hydro detail.
    pub fpha_details: Vec<FphaHydroDetail>,

    // ── Evaporation ──
    /// Number of hydros with linearized evaporation.
    pub n_evaporation: usize,
    /// Number of hydros without evaporation.
    pub n_no_evaporation: usize,
}

pub struct FphaHydroDetail {
    pub hydro_id: EntityId,
    pub name: String,
    pub source: ProductionModelSource,
    pub n_planes: usize,
}
```

**CLI display (following the stochastic summary pattern):**

```
Stochastic preprocessing
  Seed:          42
  Inflow stats:  loaded (4 hydros, 12 seasons)
  AR orders:     fixed (4x order-0)
  Correlation:   none (4×4)
  Opening tree:  estimated (10 openings/stage, 12 stages)
  Load noise:    0 stochastic buses

Hydro models
  Production:    2 constant, 2 FPHA (10 planes, loaded)
  Evaporation:   3 hydros linearized (from geometry), 1 without
```

For large systems (165+ hydros):

```
Hydro models
  Production:    165 FPHA (825 planes, loaded from fpha_hyperplanes.parquet)
  Evaporation:   162 hydros linearized, 3 run-of-river (no evaporation)
```

For Python: the summary is returned in the result dict under a `"hydro_models"` key, analogous to `"stochastic"`.

---

## 5. Phase 2: LP Integration

### 5.1 FPHA Changes to `build_stage_templates`

For each FPHA hydro at stage t, add M inequality constraints:

```
g_{h,k} ≤ κ·γ₀ᵐ + γ_vᵐ·v_avg + γ_qᵐ·q_{h,k} + γ_sᵐ·s_{h,k}
```

**Key changes:**

1. **Generation variable becomes free**: For FPHA hydros, `g_{h,k}` is bounded by `[0, max_generation]` instead of being fixed by `g = ρ·q`.
2. **Remove the equality constraint** `g = ρ·q` for FPHA hydros (keep for constant productivity).
3. **Add M rows** per FPHA hydro per block (one per hyperplane).
4. **v_avg handling**: The constraint involves `(v_in + v)/2`. Since `v_in` is fixed by the storage-fixing constraint, the FPHA row uses `γ_v/2 · v_in + γ_v/2 · v`. The `v_in` part contributes to the row RHS (patched at solve time), and the `v` part is a column coefficient.
5. **Add `fpha_turbined_cost`** to the objective for FPHA hydros' turbined flow variables.

### 5.2 Evaporation Changes to `build_stage_templates`

For each hydro with evaporation at stage t, add:

#### New LP Variables

| Variable         | Type       | Bounds    | Description                          |
| ---------------- | ---------- | --------- | ------------------------------------ |
| `Q_ev_h`         | continuous | `[0, +∞)` | Evaporation flow (m³/s)              |
| `f_evap_plus_h`  | continuous | `[0, +∞)` | Positive evaporation violation slack |
| `f_evap_minus_h` | continuous | `[0, +∞)` | Negative evaporation violation slack |

#### Evaporation Constraint

One row per hydro with evaporation (stage-level, not per-block):

$$Q_{ev,h} - \frac{k_{evapV,h,t}}{2} \cdot v_h + f^+_{evap,h} - f^-_{evap,h} = k_{evap0,h,t} + \frac{k_{evapV,h,t}}{2} \cdot v^{in}_h$$

The RHS involves $v^{in}_h$ which is fixed by the storage-fixing constraint. The $k_{evapV}/2 \cdot v^{in}$ term is patched at solve time (same mechanism as FPHA's $v_{avg}$ patching and the existing water balance patching).

#### Water Balance Modification

The water balance currently has the form:

$$v_h = v^{in}_h + \zeta \Big[ a_h + \sum_k w_k (\text{inflows} - q_{h,k} - s_{h,k} - \ldots) \Big]$$

Add evaporation as a deduction. Per the spec (cobre-docs LP formulation §4), evaporation $e_{h,k}$ is already in the formula but not implemented:

$$v_h = v^{in}_h + \zeta \Big[ a_h + \sum_k w_k (\ldots - q_{h,k} - s_{h,k} - Q_{ev,h} - \ldots) \Big]$$

Note: evaporation is stage-level (not block-level) — the same `Q_ev_h` value is subtracted in every block, weighted by the block weight $w_k$. This simplifies to $-\zeta \cdot Q_{ev,h}$ in the water balance.

#### Objective Contribution

Evaporation violation slacks carry the `evaporation_violation_cost`:

$$\text{obj} += c^{ev}_h \cdot (f^+_{evap,h} + f^-_{evap,h}) \cdot \sum_k \tau_k$$

### 5.3 Changes to `StageIndexer`

```rust
/// FPHA row range for one hydro at one stage.
pub struct FphaRowRange {
    /// First row index of this hydro's FPHA constraints.
    pub start: usize,
    /// Number of hyperplanes (rows).
    pub count: usize,
}

/// Evaporation variable indices for one hydro.
pub struct EvaporationIndices {
    /// Column index of the evaporation flow variable Q_ev.
    pub q_ev_col: usize,
    /// Column index of the positive violation slack.
    pub f_evap_plus_col: usize,
    /// Column index of the negative violation slack.
    pub f_evap_minus_col: usize,
    /// Row index of the evaporation constraint.
    pub evap_row: usize,
}
```

### 5.4 Impact on Benders Cuts

**Per the spec (§2.10):** No manual dual combination is needed for either FPHA or evaporation. The storage-fixing constraint dual `π_fix` already captures all contributions from constraints that involve `v_in`:

- FPHA hyperplane constraints → contribute via `γ_v/2 · v_in`
- Evaporation constraint → contributes via `k_evapV/2 · v_in`
- Water balance → contributes via the existing `v_in` term

The cut coefficient is simply:

```
π_v = dual of storage-fixing row
```

The backward pass, cut wire, and cut management code require **zero changes**.

### 5.5 Impact on `n_state` and State Dimension

Neither FPHA nor evaporation add new state variables. The state dimension remains `n_hydros` (one storage per hydro). Both features introduce `v_in`-dependent RHS terms that are patched at solve time, but no new state variables.

---

## 6. Phase 3: Result Extraction & End-to-End Validation

### 6.1 Result Extraction

`extract_stage_result` changes:

- **FPHA hydros**: generation is read from the `g_{h,k}` LP variable (already exists as an LP variable)
- **Evaporation**: extract `Q_ev_h` value for the evaporation output field (`evaporation_m3s` in the output schema, currently always `None`)
- **Evaporation violation**: extract `f_evap_plus` and `f_evap_minus` for the violation field

### 6.2 Simulation

The simulation pipeline already handles arbitrary stage templates. No simulation-specific changes are needed — the templates built by `build_stage_templates` with FPHA and evaporation rows will work as-is.

### 6.3 Test Case

Create a modified 4ree example with:

- 2 hydros using FPHA (with pre-fitted hyperplanes)
- 2 hydros using constant productivity
- 3 hydros with evaporation coefficients (requires geometry data)
- Verify convergence with FPHA + evaporation active

---

## 7. Phase 4 (DEFERRED): Hyperplane Fitting from Geometry

**This phase is out of scope for the initial implementation.** The detection branch in `prepare_hydro_models` returns an error for `source: "computed"`:

```rust
FphaConfig { source: "computed", .. } => {
    return Err(SddpError::NotImplemented {
        feature: "FPHA hyperplane fitting from geometry".into(),
        suggestion: "Provide pre-computed hyperplanes in fpha_hyperplanes.parquet \
                     with source: \"precomputed\" in hydro_production_models.json".into(),
    });
}
```

When implemented, this phase will:

1. Read `hydro_geometry.parquet` (volume-height-area curve)
2. Read `hydros.json` tailrace model and hydraulic losses
3. Evaluate the exact production function on a grid over `[v_min, v_max] × [0, q_max]`
4. Construct the concave envelope of the generation surface
5. Select representative hyperplanes from the envelope
6. Compute correction factor $\kappa$
7. Return the same `Vec<FphaPlane>` as the precomputed path
8. Optionally export fitted hyperplanes to `output/hydro_models/fpha_hyperplanes.parquet`

The export follows the same round-trip invariant as stochastic export: exported hyperplanes use the same schema as input hyperplanes, so they can be copied to the input directory and loaded via `source: "precomputed"`.

---

## 8. Tailrace Model Adequacy Assessment

The CEPEL reference describes three tailrace modeling approaches:

| Model                  | Description                                                           | Cobre Support                               |
| ---------------------- | --------------------------------------------------------------------- | ------------------------------------------- |
| Single polynomial      | `h_jus = a₀ + a₁Q + a₂Q² + ...`                                       | **Supported** — `TailraceModel::Polynomial` |
| Piecewise-linear table | Breakpoints with linear interpolation                                 | **Supported** — `TailraceModel::Piecewise`  |
| Piecewise polynomial   | Multiple polynomial segments, one per flow range, each up to degree 4 | **Not supported**                           |

### Why Piecewise Polynomial Is Not Needed Now

1. **For pre-fitted FPHA (Phase 1):** The tailrace model is already baked into the hyperplane coefficients. The $\gamma_s$ coefficient captures the tailrace effect as evaluated during the fitting process. Users who provide pre-fitted hyperplanes from NEWAVE/DECOMP already have any piecewise polynomial effect included in their planes.

2. **For computed FPHA fitting (Phase 4, deferred):** When fitting is implemented, we evaluate the exact production function $\phi(v, q, s)$ at grid points. At each grid point, we compute $h_{tail}(q_{out})$ using whatever tailrace model is configured. The single polynomial and piecewise-linear models already cover this evaluation. A `TailraceModel::PiecewisePolynomial` variant would be a data model extension — straightforward to add when needed.

3. **The CEPEL docs note:** "Piecewise polynomials already incorporate downstream backwater influence (remanso) in their calibration." This means the calibrated polynomial coefficients already account for typical downstream conditions. The downstream backwater effect is a cross-plant coupling that makes the problem nonlinear across plants — it is not something the LP handles dynamically.

### Deferred Action

When Phase 4 (auto-fitting) is implemented:

- Add `TailraceModel::PiecewisePolynomial { segments: Vec<TailraceSegment> }` to cobre-core
- Each segment has: `flow_min`, `flow_max`, and `coefficients: [f64; 5]` (degree 4 polynomial)
- Add parsing in cobre-io for the segmented tailrace JSON
- Use piecewise polynomial evaluation during FPH grid sampling

This is a data model extension, not an architectural change. The LP formulation is unaffected — the tailrace model only matters during the preprocessing step that produces the hyperplane coefficients.

---

## 9. Integration with StudySetup

```rust
impl StudySetup {
    pub fn new(
        system: &System,
        config: &Config,
        stochastic: StochasticContext,
        hydro_models: PrepareHydroModelsResult,  // NEW parameter (covers both FPHA and evaporation)
    ) -> Result<Self, SddpError> {
        // hydro_models.production is used by build_stage_templates for FPHA rows
        // hydro_models.evaporation is used by build_stage_templates for evaporation rows
        // hydro_models.provenance is stored for summary access
    }
}
```

The CLI and Python entry points call `prepare_hydro_models()` alongside `prepare_stochastic()`, then pass both results to `StudySetup::new()`.

### CLI flow (after implementation):

```
Loading case: examples/newave-case/
Stochastic preprocessing
  Seed:          42
  Inflow stats:  estimated (165 hydros, 12 seasons, AIC (orders 1-6: ...))
  Correlation:   estimated from residuals (165×165)
  Opening tree:  estimated (20 openings/stage, 120 stages)
  Load noise:    5 stochastic buses

Hydro models
  Production:    165 FPHA (825 planes, loaded from fpha_hyperplanes.parquet)
  Evaporation:   162 hydros linearized (from geometry), 3 without

Training   ████████████████████████████████████████ ...
```

---

## 10. Epic / Ticket Breakdown (Estimated)

### Epic 1: Hydro Model Preprocessing (5-6 days)

| Ticket | Description                                                                                                                                                                | Points |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| T-01   | Create `hydro_models.rs` module with types: `ProductionModelSet`, `ResolvedProductionModel`, `FphaPlane`, `EvaporationModelSet`, `LinearizedEvaporation`, provenance types | 3      |
| T-02   | Implement `prepare_hydro_models()` — FPHA branch: detection, loading, validation, provenance                                                                               | 3      |
| T-03   | Implement `prepare_hydro_models()` — evaporation branch: geometry loading, linearization, coefficient computation, validation                                              | 3      |
| T-04   | Add `HydroModelSummary` + CLI/Python display via Term                                                                                                                      | 2      |
| T-05   | Wire `prepare_hydro_models()` into CLI and Python entry points                                                                                                             | 2      |

### Epic 2: LP Integration — FPHA (5-7 days)

| Ticket | Description                                                                                                                  | Points |
| ------ | ---------------------------------------------------------------------------------------------------------------------------- | ------ |
| T-06   | Add FPHA row construction to `build_stage_templates` (generation as free variable, M hyperplane constraints, v_avg handling) | 5      |
| T-07   | Update `StageIndexer` for FPHA row ranges                                                                                    | 3      |
| T-08   | Add `fpha_turbined_cost` to objective for FPHA hydros                                                                        | 2      |
| T-09   | Remove FPHA validation rejection, add FPHA-only integration tests                                                            | 2      |

### Epic 3: LP Integration — Evaporation (4-5 days)

| Ticket | Description                                                                   | Points |
| ------ | ----------------------------------------------------------------------------- | ------ |
| T-10   | Add evaporation variables (Q_ev, violation slacks) to `build_stage_templates` | 3      |
| T-11   | Add evaporation constraint row with v_in patching                             | 3      |
| T-12   | Modify water balance to include evaporation term                              | 2      |
| T-13   | Add evaporation violation cost to objective                                   | 1      |

### Epic 4: Extraction & End-to-End Validation (3-4 days)

| Ticket | Description                                                                   | Points |
| ------ | ----------------------------------------------------------------------------- | ------ |
| T-14   | Update `extract_stage_result` for FPHA and evaporation outputs                | 2      |
| T-15   | Create test case: 4ree with FPHA hyperplanes + evaporation                    | 3      |
| T-16   | End-to-end test: train + simulate with FPHA + evaporation, verify convergence | 3      |

**Total Phases 1-4: 17-22 days (3.5-4.5 weeks)**

### Epic 5 (DEFERRED): Hyperplane Fitting

| Ticket | Description                                          | Points |
| ------ | ---------------------------------------------------- | ------ |
| T-17   | Exact production function evaluation                 | 3      |
| T-18   | Grid sampling and concave envelope                   | 5      |
| T-19   | Hyperplane selection and κ computation               | 3      |
| T-20   | Export fitted hyperplanes to output/                 | 2      |
| T-21   | Fitting integration tests with reference hyperplanes | 3      |

**Deferred total: ~3-4 weeks additional**

---

## 11. Risk Analysis

| Risk                                                             | Likelihood | Impact | Mitigation                                                                                                                                                                            |
| ---------------------------------------------------------------- | ---------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LP builder changes for FPHA too complex                          | Medium     | High   | The spec says the storage-fixing dual handles everything — no manual dual combination. Verify with a 1-hydro FPHA test first.                                                         |
| `v_avg` and evaporation patching share the same `v_in` mechanism | Low        | Medium | Both use the same storage-fixing row. The patch buffer already supports multiple patches per state variable. Verify with a test that has both FPHA and evaporation on the same hydro. |
| Evaporation linearization accuracy                               | Low        | Low    | First-order Taylor is standard in DECOMP/NEWAVE. The reference volume at operating midpoint is the established approach.                                                              |
| Geometry data missing `area_km2` column                          | Medium     | Low    | Validate explicitly in `prepare_hydro_models`. The column exists in the schema but may be absent in user-supplied files that only have volume+height.                                 |
| Existing tests break when FPHA rejection is removed              | Low        | Low    | Only the `test_fpha_model_rejected` test needs updating — it becomes a positive test.                                                                                                 |
| Performance regression from FPHA + evaporation LP rows           | Medium     | Low    | FPHA: ~5 rows/hydro/block. Evaporation: 1 row/hydro + 3 variables/hydro. For 165 hydros: ~2500 FPHA rows + 165 evap rows. LP solve time scales sublinearly with constraint count.     |

---

## 12. What This Design Does NOT Include

1. **Variable turbine efficiency** η(Q, h) — constant efficiency only
2. **Per-unit constraints** (min/max per generating unit) — plant-level aggregation only
3. **Lateral flow effects** on tailrace — `q_out = q + s` only
4. **Linearized head model** implementation — simulation-only enhancement, separate scope
5. **Cross-stage FPHA parameter changes** — supported by the data model but not actively tested in initial implementation
6. **FPHA output in simulation result Parquet** — which hyperplane was binding (nice-to-have, not blocking)
7. **Piecewise polynomial tailrace** — deferred to Phase 5 / auto-fitting (see §8)
8. **Seasonal reference volume for evaporation** — uses fixed midpoint; seasonal adaptation is a future enhancement
9. **Evaporation export to output/** — unlike stochastic export, linearized evaporation coefficients are derived deterministically from geometry+config; no round-trip export needed

---

**End of Design.**
