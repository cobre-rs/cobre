# ticket-003 Implement PAR parameter validation

## Context

### Background

After constructing `PrecomputedParLp`, the system must validate the derived parameters against the invariants defined in PAR Inflow Model spec section 6. Validation catches model quality issues (overfitting, instability) before they manifest as numerical problems during the optimization loop.

### Relation to Epic

This is the final ticket in Epic 01. It adds validation on top of the `PrecomputedParLp` struct built in ticket-002, completing the PAR preprocessing pipeline.

### Current State

- `PrecomputedParLp` is implemented (ticket-002) with `build()` that already rejects non-positive sigma^2
- `crates/cobre-stochastic/src/par/validation.rs` is a placeholder file
- `StochasticError` has variants for `InvalidParParameters` and `NonPositiveResidualVariance`

## Specification

### Requirements

1. Implement `validate_par_parameters` in `crates/cobre-stochastic/src/par/validation.rs`:

```rust
/// Result of PAR parameter validation.
///
/// Contains a list of warnings (non-fatal issues) and a pass/fail status.
/// Errors are returned via `Result`; warnings are accumulated in
/// `ValidationReport`.
pub struct ParValidationReport {
    /// Non-fatal warnings (e.g., near-unit-circle roots).
    pub warnings: Vec<ParWarning>,
}

/// A non-fatal PAR validation warning.
#[derive(Debug, Clone)]
pub enum ParWarning {
    /// AR polynomial has roots near the unit circle (potential instability).
    NearUnitCircleRoot {
        hydro_id: i32,
        stage_id: i32,
        min_root_magnitude: f64,
    },
    /// Residual variance is very small relative to the sample variance,
    /// suggesting potential overfitting.
    LowResidualVariance {
        hydro_id: i32,
        stage_id: i32,
        ratio: f64,
    },
}

/// Validate PAR parameters for consistency and model quality.
///
/// Checks performed:
/// 1. AR order consistency: coefficient count matches declared ar_order (via ar_order() method)
/// 2. Positive sample std: std_m3s > 0 for all models with ar_order > 0
/// 3. Stationarity check: deferred per design doc section 8 (see note below)
///
/// Returns `Ok(ParValidationReport)` with accumulated warnings on success.
/// Returns `Err(StochasticError)` on fatal validation failures.
pub fn validate_par_parameters(
    inflow_models: &[InflowModel],
) -> Result<ParValidationReport, StochasticError>
```

2. Validation checks:
   - **AR order consistency** (error): For each `InflowModel`, verify `ar_coefficients.len() == inflow_model.ar_order()`. Return `StochasticError::InvalidParParameters` if mismatch.
   - **Positive sample std** (error): For each `InflowModel` with `ar_order() > 0`, verify `std_m3s > 0.0`. Return `StochasticError::InvalidParParameters` if zero std with nonzero AR order (AR model requires nonzero variance to normalize).
   - **Positive sigma** (guaranteed): `sigma = std_m3s * residual_std_ratio`. Since `residual_std_ratio ∈ (0, 1]` is validated by the cobre-io parser (ticket-006) and `std_m3s > 0` is checked above, positive sigma is guaranteed for models with ar_order > 0. No additional check needed here.
   - **Stationarity check (deferred)**: Per `docs/design/PAR-COEFFICIENT-REDESIGN.md` section 8, stationarity checking is deferred for the minimal viable implementation. The stored coefficients are in standardized form (ψ*), so any future stationarity check operates directly on ψ* without reverse-standardization. No `NearUnitCircleRoot` warnings are emitted in this implementation -- the warning type is retained in the enum for future use.
   - **Low residual variance ratio** (warning): If `residual_std_ratio^2 < 0.01` (the AR model explains > 99% of variance), emit `ParWarning::LowResidualVariance`.

3. Note on `ar_order()`: `InflowModel` no longer has an `ar_order` field. Use `inflow_model.ar_order()` method (returns `ar_coefficients.len()`) wherever AR order is needed.

### Inputs/Props

- `inflow_models: &[InflowModel]` -- raw PAR parameters from System

### Outputs/Behavior

- Returns `Ok(ParValidationReport)` with warnings list (may be empty)
- Returns `Err(StochasticError::InvalidParParameters)` on fatal failures

### Error Handling

- Fatal: AR order mismatch, zero std with nonzero AR order
- Warning: Low residual variance ratio
- Warnings are accumulated in `ParValidationReport`, not returned as errors

## Acceptance Criteria

- [ ] Given an `InflowModel` with `ar_order()=2` and `ar_coefficients` of length 3, when `validate_par_parameters` is called, then it returns `Err(StochasticError::InvalidParParameters { ... })` with a message containing "coefficient count"
- [ ] Given an `InflowModel` with `ar_order()=1`, `std_m3s=0.0`, and `ar_coefficients=[0.3]`, when `validate_par_parameters` is called, then it returns `Err(StochasticError::InvalidParParameters { ... })` with a message containing "zero standard deviation"
- [ ] Given an `InflowModel` with `ar_order()=0`, when `validate_par_parameters` is called, then no warnings or errors are produced for that model
- [ ] Given valid `InflowModel` entries with `ar_order()=1` and `ar_coefficients=[0.3]` and `residual_std_ratio=0.954`, when `validate_par_parameters` is called, then it returns `Ok(report)` with an empty warnings list
- [ ] Given an `InflowModel` with `residual_std_ratio=0.05` (explains 99.75% of variance), when `validate_par_parameters` is called, then the report contains a `ParWarning::LowResidualVariance` warning

## Implementation Guide

### Suggested Approach

1. Define `ParValidationReport` and `ParWarning` in `validation.rs`
2. Implement `validate_par_parameters`:
   a. Iterate over all `InflowModel` entries
   b. Check AR order consistency via `ar_coefficients.len() == inflow_model.ar_order()` (fatal)
   c. Check `std_m3s > 0` when `ar_order() > 0` (fatal)
   d. Check `residual_std_ratio^2 < 0.01`, warn if so (low residual variance)
   e. Note: stationarity check is deferred per design doc section 8 -- do not implement
3. Collect all warnings in a Vec; return on first fatal error
4. Add unit tests for each check

### Key Files to Modify

- `crates/cobre-stochastic/src/par/validation.rs` (primary implementation)
- `crates/cobre-stochastic/src/par/mod.rs` (re-exports)
- `crates/cobre-stochastic/src/lib.rs` (public re-exports if needed)

### Patterns to Follow

- Validation pattern: accumulate warnings, fail-fast on errors (similar to cobre-io validation pipeline)
- Warning type as an enum with data (similar to cobre-io's validation warnings)
- Use `f64::abs()` for magnitude checks, not `f64::signum()`

### Pitfalls to Avoid

- Do NOT implement stationarity checks in this ticket -- they are deferred per design doc section 8. The `NearUnitCircleRoot` variant is retained in the enum for future use, but no code should emit it here.
- Do NOT use `unwrap()` in library code -- return `StochasticError` variants
- Do NOT call `inflow_model.ar_order` as a field -- it was removed. Use `inflow_model.ar_order()` method instead
- The `residual_std_ratio ∈ (0, 1]` range is already validated by the cobre-io parser -- no duplicate range check is needed here. The positive sigma invariant follows from `std_m3s > 0` (checked here) and `residual_std_ratio > 0` (guaranteed by cobre-io).

## Testing Requirements

### Unit Tests

- Test AR order mismatch detection (3 coefficients, order 2)
- Test zero std with nonzero AR order detection
- Test low residual variance warning (residual_std_ratio=0.05 triggers warning)
- Test AR(0) model produces no warnings
- Test empty input returns empty report

### Integration Tests

None for this ticket.

### E2E Tests

Not applicable.

## Dependencies

- **Blocked By**: ticket-001-scaffold-crate-structure.md, ticket-002-implement-precomputed-par-lp.md
- **Blocks**: Epic 02 tickets (validation is a prerequisite for opening tree generation)

## Effort Estimate

**Points**: 2
**Confidence**: High
