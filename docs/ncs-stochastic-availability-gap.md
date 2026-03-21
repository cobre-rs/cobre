# NCS Stochastic Availability Gap

> **Status: RESOLVED** (2026-03-20). The stochastic NCS availability gap
> described in this document has been closed. All six divergences (D1--D6) are
> resolved. See the implementation on branch `feat/block-factors-and-ncs`.
> The noise model uses availability factors `α_r ∈ [0,1]` per the specification,
> with `A_r = max_gen × clamp(mean + std × η, 0, 1)`.

## 1. Summary

The specification defines non-controllable source (NCS) availability as a
**stochastic** quantity produced by the scenario pipeline, varying per
(stage, scenario) trajectory. The current implementation (v0.1.6) provides
only **deterministic** per-stage availability via `constraints/ncs_bounds.parquet`,
with optional per-block scaling via `scenarios/non_controllable_factors.json`.
NCS generation does not vary across scenarios.

This document catalogs the divergences between the spec and the implementation,
assesses their modeling impact, and outlines the work required to close the gap.

---

## 2. Spec Expectations

The following spec sections define the intended NCS behavior:

| Spec File                             | Section       | Key Statement                                                                                                                                                                                     |
| ------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `math/system-elements.md`             | 6             | "$A_r$ = available generation for current (stage, scenario) from scenario pipeline, bounded by $[0, \bar{G}_r]$"                                                                                  |
| `math/equipment-formulations.md`      | 6             | "Bounds: $0 \leq g^{nc}_{r,k} \leq A_r$ where $A_r$ is stochastic available generation for current (stage, scenario)"                                                                             |
| `data-model/input-system-entities.md` | 7             | "Generation variable bounded by $[0, \text{available\_generation}]$, where available_generation is stochastic value from scenario pipeline for current (stage, scenario)"                         |
| `data-model/internal-structures.md`   | 9             | "Availability: stochastic generation available per (stage, scenario), provided by the scenario pipeline"                                                                                          |
| `data-model/input-scenarios.md`       | (correlation) | "Defines spatial correlation between stochastic processes (inflows, loads, non-controllable generation)"                                                                                          |
| `deferred.md`                         | C.5           | "Generation models in `scenarios/non_controllable_models.parquet` provide mean and standard deviation per source per stage. Correlation with inflows is supported via `correlation.json` blocks." |

### 2.1. Expected Data Flow (per spec)

```
non_controllable_models.parquet     correlation.json
  (mean, std per source/stage)       (NCS ↔ inflow correlation)
           │                                  │
           ▼                                  ▼
   ┌─────────────────────────────────────────────────┐
   │           Scenario Pipeline (cobre-stochastic)  │
   │  PAR-like model or direct stochastic generation │
   │  produces A_r(stage, scenario) per trajectory   │
   └─────────────────────────────────────────────────┘
                        │
                        ▼
              LP column upper bound
              col_upper = A_r(stage, scenario)
              (varies per scenario in forward/backward pass)
```

### 2.2. Expected Stochastic Properties

- **Per-scenario variation**: each forward/backward pass trajectory sees a
  different NCS availability realization, just as it sees different inflow
  realizations.
- **Temporal correlation**: NCS availability can be correlated across stages
  via an autoregressive model (analogous to PAR for inflows).
- **Spatial correlation**: NCS availability can be correlated with inflow
  and load processes via `correlation.json` Cholesky blocks.
- **Block distribution**: The spec flagged per-block NCS factors as an
  OPEN DESIGN QUESTION in `equipment-formulations.md` line 205.

---

## 3. Current Implementation

### 3.1. What Exists

| Component                | File                                      | Behavior                                                                                                                    |
| ------------------------ | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Stage-level availability | `constraints/ncs_bounds.parquet`          | Fixed `available_generation_mw` per (ncs_id, stage_id). Defaults to entity's `max_generation_mw` when absent.               |
| Block scaling factors    | `scenarios/non_controllable_factors.json` | Per-(ncs_id, stage_id, block_id) multiplicative factor on availability. Defaults to 1.0.                                    |
| LP column upper bound    | `lp_builder.rs:1257`                      | `col_upper = available_gen * block_factor` — deterministic, identical across all scenarios.                                 |
| Resolution pipeline      | `cobre-io` resolution step                | `ResolvedNcsBounds` (dense 2D, ncs × stages) and `ResolvedNcsFactors` (dense 3D, ncs × stages × blocks) stored in `System`. |
| Validation               | `referential.rs` rules 36-41              | Referential integrity, stage validity, non-negative bounds and factors.                                                     |
| Simulation output        | `extraction.rs`                           | Extracts `generation_mw`, `available_mw`, `curtailment_mw`, `curtailment_cost` per NCS per block.                           |

### 3.2. What Does Not Exist

| Component                                      | Spec Reference                | Status                                                                                   |
| ---------------------------------------------- | ----------------------------- | ---------------------------------------------------------------------------------------- |
| `scenarios/non_controllable_models.parquet`    | `deferred.md` C.5             | Not implemented. No mean/std statistical model for NCS availability.                     |
| Stochastic NCS generation in scenario pipeline | `system-elements.md` §6       | Not implemented. `cobre-stochastic` does not generate NCS noise.                         |
| NCS dimension in opening tree                  | `scenario-generation.md` §2.3 | Not implemented. Opening tree covers hydro inflows and loads only.                       |
| NCS entries in `correlation.json`              | `input-scenarios.md`          | Not implemented. Correlation model supports inflow and load entities only.               |
| Per-scenario LP bound patching for NCS         | `training-loop.md` §4.2       | Not implemented. NCS bounds are baked into the stage template, not patched per scenario. |

### 3.3. Resolved Design Question

The spec's OPEN DESIGN QUESTION about NCS block factors was resolved by the
implementation: `scenarios/non_controllable_factors.json` provides per-block
scaling factors analogous to `load_factors.json`. This is a reasonable resolution
but was never formally recorded in the spec.

---

## 4. Impact Assessment

### 4.1. What Works Correctly Today

- **Deterministic NCS profiles**: cases with known, fixed NCS generation per
  stage (e.g., contracted wind/solar with guaranteed output, capacity firming
  scenarios) are fully supported.
- **Block-level shaping**: the block factor mechanism correctly models
  time-of-day patterns (e.g., solar concentrated in daylight blocks).
- **LP formulation**: the variable bounds, objective coefficients, and load
  balance contributions are exactly as specified. The gap is in the _values_
  fed to the LP, not the LP structure.
- **Curtailment and cost accounting**: correct per the spec.

### 4.2. What Cannot Be Modeled

- **Wind/solar uncertainty**: the key use case for NCS — stochastic renewable
  generation — cannot be modeled. Every scenario sees the same availability,
  so the SDDP policy does not learn to hedge against NCS variability.
- **Correlated renewable + hydro risk**: the spec envisions joint stochastic
  modeling of wind/solar availability and river inflows (e.g., drought +
  low wind compound events). This requires NCS in the correlation structure.
- **NCS-driven deficit risk**: without stochastic availability, the solver
  cannot assess the probability of deficit events caused by low NCS output.
  This matters for reliability studies.

### 4.3. Severity

**Medium-High.** The LP structure is correct and extensible. The gap is
entirely in the scenario pipeline and the per-scenario bound patching.
Closing it requires:

1. A stochastic generation model for NCS (analogous to PAR for inflows)
2. Integration into the opening tree dimensionality
3. Integration into the correlation framework
4. Per-scenario LP bound patching in the forward/backward pass

The LP builder, simulation extraction, validation, and output infrastructure
require no structural changes — only the values fed into `col_upper` need to
become scenario-dependent.

---

## 5. Divergence Inventory

| #   | Spec Expectation                                      | Implementation                                                  | Gap Type                    |
| --- | ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------- |
| D1  | `A_r` varies per (stage, scenario)                    | `A_r` is fixed per stage                                        | Missing stochastic model    |
| D2  | `non_controllable_models.parquet` provides mean/std   | File does not exist; `ncs_bounds.parquet` provides fixed values | Missing input file          |
| D3  | NCS is a dimension in the opening tree                | Opening tree has hydro + load dimensions only                   | Missing dimension           |
| D4  | `correlation.json` supports NCS entities              | Correlation supports inflow + load only                         | Missing correlation support |
| D5  | Forward/backward pass patches NCS bounds per scenario | NCS bounds baked into template (no per-scenario patching)       | Missing LP patching         |
| D6  | Block factor question is open                         | Resolved via `non_controllable_factors.json`                    | Resolved (undocumented)     |

---

## 6. Proposed Resolution Approach

The following is a high-level outline. Detailed planning should produce
implementation tickets with spec references and acceptance criteria.

### Phase A: Stochastic NCS Model

**Goal**: NCS availability varies per scenario, following a statistical model.

1. **Define NCS generation model input** — decide whether to reuse the PAR
   framework (mean, std, AR coefficients per source per stage) or use a
   simpler model (e.g., Beta distribution for capacity factors, log-normal).
   The spec references `non_controllable_models.parquet` with mean/std,
   suggesting a normal or log-normal approach.

2. **Extend `cobre-stochastic`** — add NCS availability generation alongside
   inflow and load noise. Each NCS entity produces one stochastic dimension.

3. **Extend the opening tree** — add NCS dimensions after load dimensions.
   Update `OpeningTree` size calculations, entity index conventions, and
   the user-supplied opening tree loader (`noise_openings.parquet`).

4. **Extend `correlation.json`** — allow NCS entities in correlation groups.
   Update the Cholesky decomposition to include NCS dimensions.

### Phase B: Per-Scenario LP Patching

**Goal**: each forward/backward pass scenario sees its own NCS availability.

5. **Add NCS to the state/noise patching path** — in the forward pass,
   after patching inflow and load noise, patch NCS column upper bounds
   using the scenario's NCS availability realization. This requires
   `patch_col_bounds` calls for NCS columns (already supported by the
   solver interface).

6. **Interaction with block factors** — the per-scenario availability
   should be multiplied by the block factor:
   `col_upper = A_r(stage, scenario) * block_factor(ncs, stage, block)`.
   The block factor remains deterministic; only the base availability
   becomes stochastic.

7. **Backward pass dual extraction** — verify that NCS column bounds
   do not participate in Benders cut coefficients (they should not,
   since NCS availability is not a state variable — it is exogenous noise).

### Phase C: Spec Reconciliation

8. **Update cobre-docs specs** — formally document the block factor
   resolution (divergence D6) and the chosen stochastic model.

9. **Update `ncs_bounds.parquet` semantics** — decide whether this file
   continues to exist as a deterministic override (useful for contracted
   output), coexists with the stochastic model, or is superseded.

10. **Add deterministic test cases** — extend D15 with a stochastic
    variant that verifies NCS availability varies across scenarios and
    produces different costs per trajectory.

---

## 7. Compatibility Considerations

- **No breaking changes to existing cases.** Cases without
  `non_controllable_models.parquet` should continue to work with
  deterministic availability from `ncs_bounds.parquet` (or entity defaults).
  The stochastic model activates only when the model file is present.

- **Opening tree format change.** Adding NCS dimensions to the opening tree
  changes the `entity_index` convention in `noise_openings.parquet`.
  User-supplied opening trees from pre-stochastic-NCS versions would need
  regeneration. This is a format versioning concern.

- **`ResolvedNcsBounds` may become per-scenario.** Currently stored in
  `System` (shared, immutable). With stochastic availability, the
  per-scenario values would be computed on the fly during the training loop,
  not pre-resolved. The `ResolvedNcsBounds` struct could remain as the
  deterministic fallback / base value.

---

## 8. References

### Cobre-docs spec files

- `src/specs/math/system-elements.md` — Section 6 (NCS physical model)
- `src/specs/math/equipment-formulations.md` — Section 6 (NCS LP formulation)
- `src/specs/data-model/input-system-entities.md` — Section 7 (NCS entity schema)
- `src/specs/data-model/internal-structures.md` — Section 1.9.8, Section 9
- `src/specs/data-model/input-scenarios.md` — Correlation definition
- `src/specs/deferred.md` — Section C.5 (NCS deferred feature)
- `src/specs/math/lp-formulation.md` — Penalty system Category 3

### Implementation files

- `crates/cobre-core/src/entities/non_controllable.rs` — Entity struct
- `crates/cobre-core/src/resolved.rs` — `ResolvedNcsBounds`, `ResolvedNcsFactors`
- `crates/cobre-io/src/constraints/ncs_bounds.rs` — Parquet parser
- `crates/cobre-io/src/scenarios/non_controllable_factors.rs` — JSON parser
- `crates/cobre-sddp/src/lp_builder.rs:1245-1261` — NCS column construction
- `crates/cobre-sddp/src/simulation/extraction.rs:820-868` — NCS output extraction
