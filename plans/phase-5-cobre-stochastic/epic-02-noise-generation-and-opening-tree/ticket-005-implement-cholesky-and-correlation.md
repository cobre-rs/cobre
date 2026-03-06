# ticket-005 Implement Cholesky decomposition and correlation resolution

## Context

### Background

Hydros within the same correlation group share spatially correlated noise. The correlation structure is defined via named profiles in `CorrelationModel` (loaded from `correlation.json`). Before the optimization loop, each correlation matrix is Cholesky-decomposed into a lower-triangular factor L such that Sigma = L * L^T. At runtime, independent N(0,1) samples z are transformed into correlated samples via eta = L * z. The Cholesky factors are computed once per profile and cached; a stage-to-profile schedule resolves which factor to use at each stage.

### Relation to Epic

This ticket provides the correlation infrastructure used by the opening tree generator (ticket-007). The Cholesky factor is applied to independent noise samples to produce spatially correlated noise vectors.

### Current State

- `crates/cobre-stochastic/src/correlation/cholesky.rs` is a placeholder
- `crates/cobre-stochastic/src/correlation/resolve.rs` is a placeholder
- `cobre-core` provides `CorrelationModel`, `CorrelationProfile`, `CorrelationGroup`, `CorrelationScheduleEntry`
- Correlation matrices are stored as `Vec<Vec<f64>>` in `CorrelationGroup.matrix`

## Specification

### Requirements

1. Implement Cholesky decomposition in `crates/cobre-stochastic/src/correlation/cholesky.rs`:

```rust
/// Lower-triangular Cholesky factor of a correlation matrix.
///
/// Stores L such that Sigma = L * L^T. The factor is stored as a flat
/// array in row-major order, including only the lower triangle
/// (L[i][j] for j <= i).
///
/// Used to transform independent standard normal samples into
/// spatially correlated samples: eta = L * z where z ~ N(0, I).
#[derive(Debug, Clone)]
pub struct CholeskyFactor {
    /// Lower-triangular factor in row-major packed storage.
    /// Length: n * (n + 1) / 2 where n is the matrix dimension.
    /// Element (i, j) with j <= i is at index i*(i+1)/2 + j.
    data: Box<[f64]>,
    /// Matrix dimension (number of entities in the correlation group).
    dim: usize,
}
```

2. Implement `CholeskyFactor::decompose(matrix: &[Vec<f64>]) -> Result<Self, StochasticError>`:
   - Standard Cholesky-Banachiewicz algorithm (row-by-row)
   - Returns `Err(StochasticError::CholeskyDecompositionFailed)` if matrix is not positive definite (diagonal element <= 0 during decomposition)
   - Validates matrix is square and symmetric within tolerance (1e-10)

3. Implement `CholeskyFactor::transform(&self, independent: &[f64], correlated: &mut [f64])`:
   - Computes eta = L * z where z is `independent` and eta is written to `correlated`
   - Both slices must have length `self.dim`
   - Panics on length mismatch (debug assertion)

4. Implement correlation resolution in `crates/cobre-stochastic/src/correlation/resolve.rs`:

```rust
/// Pre-decomposed correlation data for all profiles, with stage-to-profile mapping.
///
/// Built once during initialization. At runtime, the noise generator looks up
/// the active profile for the current stage via `profile_for_stage()` and
/// applies the Cholesky transform using the cached factor.
pub struct DecomposedCorrelation {
    /// Cholesky factors keyed by profile name.
    /// BTreeMap preserves deterministic iteration order.
    factors: BTreeMap<String, Vec<GroupFactor>>,

    /// Stage-to-profile-name mapping. For stages not in this map,
    /// the "default" profile is used.
    schedule: HashMap<i32, String>,

    /// Name of the default profile.
    default_profile: String,
}

/// A single correlation group's Cholesky factor with entity ID mapping.
pub struct GroupFactor {
    /// The Cholesky factor for this group.
    pub factor: CholeskyFactor,
    /// Entity IDs in this group, in the order matching the factor rows/columns.
    pub entity_ids: Vec<EntityId>,
}
```

5. Implement `DecomposedCorrelation::build(model: &CorrelationModel) -> Result<Self, StochasticError>`:
   - Decompose each profile's correlation groups
   - Validate that a "default" profile exists (or that there is exactly one profile)
   - Build the stage-to-profile schedule from `model.schedule`

6. Implement `DecomposedCorrelation::apply_correlation(&self, stage_id: i32, independent_noise: &mut [f64], entity_order: &[EntityId])`:
   - Look up the active profile for the stage
   - For each correlation group in the profile, find the matching entities in `entity_order`
   - Apply the Cholesky transform to the corresponding subset of `independent_noise`
   - Entities not in any group keep their independent noise unchanged

### Inputs/Props

- `matrix: &[Vec<f64>]` -- symmetric PD correlation matrix from `CorrelationGroup`
- `model: &CorrelationModel` -- from `System.correlation`
- `independent_noise: &mut [f64]` -- N(0,1) samples, modified in-place
- `entity_order: &[EntityId]` -- canonical hydro ordering from System

### Outputs/Behavior

- `CholeskyFactor::decompose` returns lower-triangular factor L
- `DecomposedCorrelation::apply_correlation` transforms independent noise in-place to correlated noise
- Entities not in any correlation group retain independent noise

### Error Handling

- Non-PD matrix: `StochasticError::CholeskyDecompositionFailed`
- Non-square matrix: `StochasticError::InvalidCorrelation`
- Non-symmetric matrix: `StochasticError::InvalidCorrelation`
- Missing default profile: `StochasticError::InvalidCorrelation`

## Acceptance Criteria

- [ ] Given a 2x2 identity matrix `[[1,0],[0,1]]`, when `CholeskyFactor::decompose` is called, then the factor L is also the identity (L[0][0]=1, L[1][0]=0, L[1][1]=1)
- [ ] Given `[[1.0, 0.8], [0.8, 1.0]]`, when `CholeskyFactor::decompose` is called, then L[0][0]=1.0, L[1][0]=0.8, L[1][1]=0.6 (sqrt(1-0.64)=0.6)
- [ ] Given independent samples `z = [1.0, 0.0]` and the 2x2 factor from the previous test, when `transform` is called, then `correlated = [1.0, 0.8]`
- [ ] Given a non-PD matrix `[[1.0, 2.0], [2.0, 1.0]]`, when `CholeskyFactor::decompose` is called, then it returns `Err(StochasticError::CholeskyDecompositionFailed { ... })`
- [ ] Given a `CorrelationModel` with profiles "default" and "wet", schedule mapping stage 0 to "wet", when `DecomposedCorrelation::build` is called and `apply_correlation(0, ...)` is invoked, then the "wet" profile's Cholesky factor is used

## Implementation Guide

### Suggested Approach

1. Implement `CholeskyFactor` struct with packed lower-triangular storage
2. Implement Cholesky-Banachiewicz decomposition:
   ```
   for i in 0..n:
     for j in 0..=i:
       sum = matrix[i][j] - sum(L[i][k] * L[j][k] for k in 0..j)
       if i == j:
         if sum <= 0: return Err(non-PD)
         L[i][j] = sqrt(sum)
       else:
         L[i][j] = sum / L[j][j]
   ```
3. Implement `transform` as matrix-vector multiply with the lower-triangular factor
4. Implement `GroupFactor` and `DecomposedCorrelation`
5. Implement `apply_correlation` with entity ID matching
6. Add comprehensive unit tests with hand-computed examples

### Key Files to Modify

- `crates/cobre-stochastic/src/correlation/cholesky.rs` (Cholesky implementation)
- `crates/cobre-stochastic/src/correlation/resolve.rs` (profile resolution)
- `crates/cobre-stochastic/src/correlation/mod.rs` (re-exports)
- `crates/cobre-stochastic/src/lib.rs` (public re-exports)

### Patterns to Follow

- Packed lower-triangular storage: element (i,j) at index `i*(i+1)/2 + j` for `j <= i`
- `BTreeMap` for deterministic profile ordering (follows cobre-core CorrelationModel pattern)
- In-place mutation for `apply_correlation` (avoids allocation on hot path)

### Pitfalls to Avoid

- Do NOT use a full n*n array for the Cholesky factor -- use packed lower-triangular storage to save memory
- Per DEC-020 and the PAR coefficient storage redesign (`docs/design/PAR-COEFFICIENT-REDESIGN.md` section 8), positive definite matrices are required for Cholesky decomposition. PSD matrices with zero eigenvalues produce a `CholeskyDecompositionFailed` error. Input correlation matrices must be positive definite; the cobre-io parser is responsible for surface-level format validation, but rank deficiency is detected and rejected here.
- Do NOT allocate temporary vectors inside `transform` -- write directly to the output buffer
- The entity IDs in `CorrelationGroup.entities` may not be in the same order as the canonical hydro ordering in `entity_order` -- the implementation must map between the two orderings

## Testing Requirements

### Unit Tests

- Cholesky of identity matrix (any dimension 1-4)
- Cholesky of known 2x2 correlated matrix with hand-computed factor
- Cholesky of 3x3 matrix with known factor (from sampling-scheme-testing.md fixture)
- Cholesky failure on non-PD matrix
- Cholesky failure on non-square matrix
- `transform` with identity factor: output equals input
- `transform` with known 2x2 factor: hand-computed correlated output
- `DecomposedCorrelation::build` with single default profile
- `DecomposedCorrelation::build` with schedule mapping
- `apply_correlation` with entities matching a subset of the canonical order
- `apply_correlation` leaves unmatched entities unchanged

### Integration Tests

None for this ticket.

### E2E Tests

Not applicable.

## Dependencies

- **Blocked By**: ticket-001-scaffold-crate-structure.md
- **Blocks**: ticket-007-implement-opening-tree-generation.md

## Effort Estimate

**Points**: 3
**Confidence**: High
