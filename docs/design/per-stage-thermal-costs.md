# Per-Stage Thermal Costs

## Implementation Status

**IMPLEMENTED** — v0.5.0, branch `feat/tier1-tier2-correctness-and-performance`

| Component                                                                              | Status   | Notes                                      |
| -------------------------------------------------------------------------------------- | -------- | ------------------------------------------ |
| cobre-core: `cost_per_mwh: f64` on `ThermalStageBounds`                                | DONE     |                                            |
| cobre-io: `cost_per_mwh: Option<f64>` + `block_id: Option<i32>` on `ThermalBoundsRow`  | DONE     |                                            |
| cobre-sddp: LP builder reads `tb.cost_per_mwh`                                         | DONE     |                                            |
| D27 deterministic regression test                                                      | DONE     |                                            |
| Negative cost validation in parquet parser                                             | DONE     |                                            |
| `ThermalCostSegment` struct removed; `cost_segments` replaced by scalar `cost_per_mwh` | DONE     | Design was simplified vs original proposal |
| cobre-bridge: CLAST.DAT per-year cost extraction                                       | NOT DONE | Separate repo; tracked separately          |

---

## Problem

CLAST.DAT in newave defines thermal generation costs per `(plant, study_year)`.
The current converter only reads `indice_ano_estudo == 1` and writes a static
cost in `thermals.json`. For 46 of 104 thermals in the example case, costs vary
across study years — up to +30% by year 5 (e.g., SUAPE II: 959 → 1250 R$/MWh).

This causes incorrect thermal dispatch ordering in later study years.

## Proposed Fix

Add an optional `cost_per_mwh` column to `thermal_bounds.parquet` and propagate
it through the cobre pipeline.

### Changes Required

#### cobre-core (`resolved.rs`)

Add `cost_per_mwh: f64` to `ThermalStageBounds`:

```rust
pub struct ThermalStageBounds {
    pub min_generation_mw: f64,
    pub max_generation_mw: f64,
    pub cost_per_mwh: f64,  // NEW — falls back to thermals.json segment cost
}
```

Update `ResolvedBounds::build()` to populate this field from the base
`thermal.cost_segments[0].cost_per_mwh`, then override with parquet values
when present.

#### cobre-io (`constraints/bounds.rs`)

Add `cost_per_mwh: Option<f64>` to `ThermalBoundsRow`:

```rust
pub struct ThermalBoundsRow {
    pub thermal_id: EntityId,
    pub stage_id: i32,
    pub min_generation_mw: Option<f64>,
    pub max_generation_mw: Option<f64>,
    pub cost_per_mwh: Option<f64>,  // NEW
}
```

Update `parse_thermal_bounds()` to read the new optional column from parquet.
The column must be optional (nullable) for backward compatibility — existing
cases without per-stage costs should continue to work.

Update the parquet schema in `_THERMAL_BOUNDS_SCHEMA`.

#### cobre-sddp (`lp_builder/matrix.rs`)

Replace the static cost lookup:

```rust
// BEFORE:
let marginal_cost_per_mwh = thermal
    .cost_segments
    .first()
    .map_or(0.0, |seg| seg.cost_per_mwh);

// AFTER:
let marginal_cost_per_mwh = tb.cost_per_mwh;
```

Since `ThermalStageBounds` already carries the resolved cost (with fallback
to the base value), no conditional logic is needed in the LP builder.

#### cobre-bridge (`converters/thermal.py`)

1. Build a cost lookup `{(codigo_usina, indice_ano_estudo): cost}` from
   CLAST.DAT (all years, not just year 1).

2. Add `cost_per_mwh` to `_THERMAL_BOUNDS_SCHEMA`:

   ```python
   pa.field("cost_per_mwh", pa.float64()),
   ```

3. In `convert_thermal_bounds()`, for each `(thermal, stage)` row, look up
   the cost for the corresponding study year index:

   ```python
   study_year_index = (stage_idx // 12) + 1  # 1-based year index
   cost = clast_map.get((newave_code, study_year_index), base_cost)
   ```

4. Emit the cost in every row of `thermal_bounds.parquet`.

### Backward Compatibility

- The parquet column is nullable. Old cases without `cost_per_mwh` will
  parse as `None`, and `ResolvedBounds::build()` will fall back to the
  static `thermals.json` cost.
- The `thermals.json` schema is unchanged.
- Existing deterministic regression tests (D01–D25) don't use thermal
  bounds with costs, so they pass without modification.

### Testing

- Add a deterministic test case with a 2-stage, 2-thermal system where
  thermal costs differ between stages. Verify that the LP objective
  coefficients change per stage.
- Integration test: convert the example newave case and verify that
  `thermal_bounds.parquet` contains per-year cost values matching CLAST.DAT.
