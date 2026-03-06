# ticket-002 Implement PrecomputedParLp struct and builder

## Context

### Background

The PAR(p) inflow model requires three precomputed arrays for hot-path LP patching: deterministic base values, residual standard deviations (sigma), and AR lag coefficients (psi). These are derived from the raw `InflowModel` parameters stored in `cobre-core::System`. The derivation follows PAR Inflow Model spec sections 3 and 7 in `cobre-docs/src/specs/math/par-inflow-model.md` and Scenario Generation spec section 1 in `cobre-docs/src/specs/architecture/scenario-generation.md`.

### Relation to Epic

This is the core data type ticket of Epic 01. It defines `PrecomputedParLp` -- the primary output of the PAR preprocessing pipeline. Ticket-003 (validation) and all Epic 02 tickets depend on this struct being available.

### Current State

- `crates/cobre-stochastic/src/par/precompute.rs` is a placeholder file (from ticket-001)
- `cobre-core` provides `InflowModel` with fields: `hydro_id: EntityId`, `stage_id: i32`, `mean_m3s: f64`, `std_m3s: f64`, `ar_coefficients: Vec<f64>` (standardized by seasonal std, ψ*), `residual_std_ratio: f64`
- `InflowModel` no longer has an `ar_order` field — use `inflow_model.ar_order()` method instead (returns `ar_coefficients.len()`)
- `cobre-core` provides `Stage` with fields including `index: usize`, `id: i32`, `season_id: Option<usize>`

## Specification

### Requirements

1. Define `PrecomputedParLp` in `crates/cobre-stochastic/src/par/precompute.rs`:

```rust
/// Cache-friendly PAR(p) model data for LP RHS patching.
///
/// Built once during initialization from raw [`InflowModel`] parameters.
/// Consumed read-only during iterative optimization.
///
/// All arrays use stage-major layout: outer dimension is stage index,
/// inner dimension is hydro index (sorted by canonical entity ID order).
/// This layout is optimal for sequential stage iteration within a
/// scenario trajectory.
///
/// See [PAR Inflow Model SS7](par-inflow-model.md) for the derivation
/// of each cached component.
pub struct PrecomputedParLp {
    /// Deterministic base b_{h,m(t)} = mu_{m(t)} - sum_l psi_{m(t),l} * mu_{m(t-l)}.
    /// Flat array indexed as [stage * n_hydros + hydro].
    /// Length: n_stages * n_hydros.
    deterministic_base: Box<[f64]>,

    /// Residual standard deviation sigma_{m(t)} per (stage, hydro).
    /// Derived as sigma = s_m * residual_std_ratio.
    /// Flat array indexed as [stage * n_hydros + hydro].
    /// Length: n_stages * n_hydros.
    sigma: Box<[f64]>,

    /// AR lag coefficients psi_{m(t),l} in original units per (stage, hydro, lag).
    /// Flat array indexed as [stage * n_hydros * max_order + hydro * max_order + lag].
    /// Length: n_stages * n_hydros * max_order.
    /// Padded with 0.0 for hydros with ar_order < max_order.
    psi: Box<[f64]>,

    /// AR order per hydro. Length: n_hydros.
    /// orders[h] gives the number of meaningful lags in psi for hydro h.
    orders: Box<[usize]>,

    /// Number of study stages.
    n_stages: usize,

    /// Number of hydro plants.
    n_hydros: usize,

    /// Maximum AR order across all hydros and stages.
    max_order: usize,
}
```

2. Implement accessor methods on `PrecomputedParLp`:
   - `pub fn deterministic_base(&self, stage: usize, hydro: usize) -> f64` -- panics if out of bounds
   - `pub fn sigma(&self, stage: usize, hydro: usize) -> f64` -- panics if out of bounds
   - `pub fn psi_slice(&self, stage: usize, hydro: usize) -> &[f64]` -- returns slice of length `max_order` (caller uses `orders[hydro]` to know how many are meaningful)
   - `pub fn order(&self, hydro: usize) -> usize` -- AR order for given hydro
   - `pub fn n_stages(&self) -> usize`
   - `pub fn n_hydros(&self) -> usize`
   - `pub fn max_order(&self) -> usize`

3. Implement `PrecomputedParLp::build(inflow_models: &[InflowModel], stages: &[Stage], hydro_ids: &[EntityId]) -> Result<Self, StochasticError>`:
   - `hydro_ids` must be in canonical sorted order (caller guarantees from System)
   - The builder creates a hydro_id-to-index map for O(1) lookup
   - For each (stage, hydro) pair, look up the corresponding `InflowModel` by matching `stage_id` and `hydro_id`
   - If no `InflowModel` exists for a (stage, hydro) pair, use defaults: mean=0, std=0, ar_order=0 (deterministic zero inflow)
   - Convert standardized coefficients to original-unit at runtime per design doc `docs/design/PAR-COEFFICIENT-REDESIGN.md` section 3.5:
     `psi[l] = psi_star[l] * s_m / s_{m-l}` where `s_m` is `std_m3s` for the stage's season and `s_{m-l}` is `std_m3s` for the season `l` stages prior
   - Compute residual standard deviation using the trivial formula (no autocorrelation reconstruction needed):
     `sigma[h][t] = inflow_model.std_m3s * inflow_model.residual_std_ratio`
   - Compute deterministic base per PAR Inflow Model spec section 7.4:
     `base[h][t] = mu[t] - sum_l psi[t][l] * mu[t-l]`
     where `mu[t-l]` is the mean for the stage that is `l` stages before stage `t`

### Inputs/Props

- `inflow_models: &[InflowModel]` -- sorted by `(hydro_id, stage_id)` from `System`
- `stages: &[Stage]` -- sorted by `index` from `System`
- `hydro_ids: &[EntityId]` -- canonical sorted order from `System`

### Outputs/Behavior

- Returns `Ok(PrecomputedParLp)` with all arrays populated
- Returns `Err(StochasticError::NonPositiveResidualVariance {...})` if sigma^2 <= 0 for any (hydro, stage) with ar_order > 0
- Returns `Err(StochasticError::InvalidParParameters {...})` if ar_coefficients length does not match ar_order

### Error Handling

- Non-positive residual variance: `StochasticError::NonPositiveResidualVariance`
- AR order mismatch: `StochasticError::InvalidParParameters`
- Missing stage season_id when needed for lag lookup: `StochasticError::InvalidParParameters`

## Acceptance Criteria

- [ ] Given `InflowModel` with hydro_id=1, stage_id=0, mean=100.0, std=30.0, ar_coefficients=[0.3] (standardized ψ*), residual_std_ratio=0.954, and the previous stage having mean=100.0 and std=30.0, when `PrecomputedParLp::build` is called, then `deterministic_base(0, 0)` returns `100.0 - (0.3 * 30.0/30.0) * 100.0 = 70.0`
- [ ] Given the same InflowModel, when `PrecomputedParLp::build` is called, then `sigma(0, 0)` returns `30.0 * 0.954 = 28.62`
- [ ] Given `InflowModel` with ar_order=0 and std=0.0, when `PrecomputedParLp::build` is called, then `sigma(stage, hydro)` returns 0.0 and `deterministic_base(stage, hydro)` returns `mean_m3s`
- [ ] Given two hydros with IDs 5 and 3 (non-canonical order in raw data), when `hydro_ids` is `[EntityId(3), EntityId(5)]`, then hydro index 0 maps to EntityId(3) and index 1 maps to EntityId(5) (declaration-order invariance)
- [ ] Given `psi_slice(stage, hydro)` for a hydro with ar_order=2 and max_order=3, when the returned slice is inspected, then it has length 3 with positions 0-1 containing the coefficients and position 2 containing 0.0

## Implementation Guide

### Suggested Approach

1. Define the `PrecomputedParLp` struct with all fields as specified
2. Implement accessor methods with bounds checking via `assert!` (these are debug-mode checks; the hot path uses unchecked indexing in release)
3. Implement `build()`:
   a. Build `hydro_id_to_index: HashMap<EntityId, usize>` from `hydro_ids`
   b. Determine `max_order` as the maximum `ar_order()` across all `InflowModel` entries
   c. Group `inflow_models` into a `HashMap<(i32, i32), &InflowModel>` keyed by `(hydro_id.0, stage_id)`
   d. Allocate flat arrays: `deterministic_base` and `sigma` of length `n_stages * n_hydros`, `psi` of length `n_stages * n_hydros * max_order`, `orders` of length `n_hydros`
   e. For each stage (by index) and each hydro (by canonical index):
      - Look up the `InflowModel`; if missing, fill zeros
      - Convert standardized coefficients to original-unit: `psi[l] = psi_star[l] * s_m / s_{m-l}`
      - Compute sigma using `s_m * residual_std_ratio`
      - Compute deterministic_base
      - Copy psi coefficients (original units) into the flat array
4. Add unit tests with hand-computable examples

### Key Files to Modify

- `crates/cobre-stochastic/src/par/precompute.rs` (primary implementation)
- `crates/cobre-stochastic/src/par/mod.rs` (re-exports)
- `crates/cobre-stochastic/src/lib.rs` (public re-exports)
- `crates/cobre-stochastic/src/error.rs` (may need minor adjustments)

### Patterns to Follow

- Flat array indexing: `stage * n_hydros + hydro` for 2D, `stage * n_hydros * max_order + hydro * max_order + lag` for 3D
- `Box<[f64]>` via `Vec<f64>::into_boxed_slice()` after construction
- Follow the entity module pattern from cobre-core: struct definition, impl block with methods, `#[cfg(test)] mod tests` at bottom
- Use `debug_assert!` for bounds checks in accessors (follows cobre-solver pattern)

### Pitfalls to Avoid

- Do NOT use nested `Vec<Vec<f64>>` -- the spec requires flat contiguous arrays for cache efficiency
- Do NOT assume all hydros have the same AR order -- use `max_order` for array sizing and pad shorter orders with 0.0
- Do NOT assume stages are contiguous integers starting at 0 -- use `Stage.index` (0-based canonical index) for array indexing, not `Stage.id`
- The lag lookup `mu[t-l]` requires finding the stage that is `l` positions before the current stage; pre-study stages (negative IDs) may provide lag means
- Do NOT reference "SDDP" in any doc comment or type name
- Do NOT call `inflow_model.ar_order` as a field -- it was removed. Use `inflow_model.ar_order()` method instead

## Testing Requirements

### Unit Tests

- Test `build` with a single hydro, single stage, AR order 0 (deterministic): verify deterministic_base = mean, sigma = 0
- Test `build` with a single hydro, single stage, AR order 1: verify deterministic_base and sigma against hand computation
- Test `build` with two hydros, three stages, varying AR orders: verify all array values
- Test accessor bounds: `deterministic_base(0, 0)` succeeds, `sigma(n_stages, 0)` panics
- Test declaration-order invariance: build with hydro IDs in non-canonical order, verify same results as canonical

### Integration Tests

None for this ticket.

### E2E Tests

Not applicable.

## Dependencies

- **Blocked By**: ticket-001-scaffold-crate-structure.md
- **Blocks**: ticket-003-implement-par-validation.md, all Epic 02 tickets

## Effort Estimate

**Points**: 3
**Confidence**: High
