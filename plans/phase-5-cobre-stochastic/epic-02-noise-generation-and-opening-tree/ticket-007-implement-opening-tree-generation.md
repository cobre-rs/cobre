# ticket-007 Implement opening tree generation

## Context

### Background

The opening tree is generated once before the optimization loop begins. For each (opening_index, stage) pair, a deterministic seed is derived via SipHash-1-3 (ticket-004), a Pcg64 RNG is initialized from that seed, independent N(0,1) samples are drawn (one per entity), and the Cholesky correlation transform (ticket-005) is applied to produce spatially correlated noise. The results are stored in the `OpeningTree` (ticket-006) in stage-major layout.

### Relation to Epic

This is the integration ticket of Epic 02, combining seed derivation (ticket-004), Cholesky correlation (ticket-005), and the OpeningTree type (ticket-006) into the complete tree generation pipeline. It is the capstone of the noise generation infrastructure.

### Current State

- `crates/cobre-stochastic/src/tree/generate.rs` is a placeholder
- Seed derivation (`derive_opening_seed`) is implemented (ticket-004)
- Cholesky decomposition and `DecomposedCorrelation` are implemented (ticket-005)
- `OpeningTree::from_parts` is implemented (ticket-006)

## Specification

### Requirements

1. Implement the generation function in `crates/cobre-stochastic/src/tree/generate.rs`:

```rust
/// Generate a fixed opening tree with correlated noise realisations.
///
/// For each (opening_index, stage) pair:
/// 1. Derive a deterministic seed via SipHash-1-3
/// 2. Initialize a Pcg64 RNG from the derived seed
/// 3. Draw `dim` independent N(0,1) samples
/// 4. Apply the Cholesky correlation transform for the active profile
///
/// The generation order is opening-major (outer loop: openings, inner
/// loop: stages) to align with the parallel generation model where
/// each rank generates a contiguous block of openings.
///
/// # Arguments
///
/// * `base_seed` -- Base seed from scenario source configuration
/// * `stages` -- Study stages (provides branching factors and stage IDs)
/// * `dim` -- Number of entities (random variables per noise vector)
/// * `correlation` -- Pre-decomposed correlation data (Cholesky factors)
/// * `entity_order` -- Canonical entity IDs for correlation mapping
///
/// # Returns
///
/// An `OpeningTree` with all noise values populated.
pub fn generate_opening_tree(
    base_seed: u64,
    stages: &[Stage],
    dim: usize,
    correlation: &DecomposedCorrelation,
    entity_order: &[EntityId],
) -> OpeningTree
```

2. The function must:
   - Extract `openings_per_stage` from `stages[t].scenario_config.branching_factor`
   - Compute total data length: `sum(openings_per_stage[t] * dim for t in 0..n_stages)`
   - Allocate a single `Vec<f64>` of the total length
   - For each opening_index in `0..max(openings_per_stage)`:
     - For each stage t in `0..n_stages`:
       - If `opening_index >= openings_per_stage[t]`, skip (variable branching)
       - Derive seed: `derive_opening_seed(base_seed, opening_index as u32, stages[t].id as u32)`
       - Initialize RNG: `rng_from_seed(seed)`
       - Draw `dim` independent N(0,1) samples into the correct position in the data array
       - Apply correlation: `correlation.apply_correlation(stages[t].id, &mut noise_slice, entity_order)`
   - Call `OpeningTree::from_parts(data, openings_per_stage, dim)`

3. N(0,1) sampling uses `rand_distr::StandardNormal` with the Pcg64 RNG:
   ```rust
   use rand::Rng;
   use rand_distr::StandardNormal;
   for i in 0..dim {
       noise[i] = rng.sample(StandardNormal);
   }
   ```

### Inputs/Props

- `base_seed: u64` -- from `ScenarioSource.seed` (converted from `Option<i64>`)
- `stages: &[Stage]` -- canonical stage list
- `dim: usize` -- number of hydros (or total stochastic entities)
- `correlation: &DecomposedCorrelation` -- pre-decomposed Cholesky factors
- `entity_order: &[EntityId]` -- canonical entity IDs for correlation mapping

### Outputs/Behavior

- Returns a fully populated `OpeningTree`
- Given the same `base_seed`, stages, dim, and correlation, the output is bit-for-bit identical across calls

### Error Handling

This function is infallible: correlation decomposition errors are caught during `DecomposedCorrelation::build()` (ticket-005). The generation itself only uses pre-validated data.

## Acceptance Criteria

- [ ] Given `base_seed=42`, 2 stages with branching_factor=3, dim=2, identity correlation, when `generate_opening_tree` is called twice, then both calls produce identical `OpeningTree` data (determinism)
- [ ] Given the same setup, when `opening(0, 0)` is called on the result, then the returned slice has length 2 and contains finite f64 values
- [ ] Given `base_seed=42` and `base_seed=99` with identical stage/dim/correlation, when the two trees are compared element-by-element, then they differ (seed sensitivity)
- [ ] Given 3 stages with branching_factors `[2, 3, 1]` and dim=2, when the tree is generated, then `tree.n_openings(0)==2`, `tree.n_openings(1)==3`, `tree.n_openings(2)==1`, and `tree.len()==(2+3+1)*2=12`

## Implementation Guide

### Suggested Approach

1. Extract `openings_per_stage` from stages' scenario configs
2. Compute `max_openings = openings_per_stage.iter().max()`
3. Compute total elements and allocate `Vec<f64>` with zeros
4. Compute stage offsets for direct indexing into the data array
5. Outer loop: `for opening_idx in 0..max_openings`
   - Inner loop: `for (stage_idx, stage) in stages.iter().enumerate()`
     - Skip if `opening_idx >= openings_per_stage[stage_idx]`
     - Compute data offset: `stage_offsets[stage_idx] + opening_idx * dim`
     - Derive seed, init RNG, sample N(0,1) into `data[offset..offset+dim]`
     - Apply correlation to `data[offset..offset+dim]`
6. Return `OpeningTree::from_parts(data, openings_per_stage, dim)`

### Key Files to Modify

- `crates/cobre-stochastic/src/tree/generate.rs` (primary implementation)
- `crates/cobre-stochastic/src/tree/mod.rs` (re-exports)
- `crates/cobre-stochastic/src/lib.rs` (public API -- `generate_opening_tree`)

### Patterns to Follow

- Seed derivation + RNG init + sampling as a tight sequence (no intermediate allocation)
- Stage-major data layout: compute `stage_offsets` once, index directly
- Use `rng.sample::<f64, _>(StandardNormal)` for each element

### Pitfalls to Avoid

- Do NOT use `stage.index` as the `stage` parameter to `derive_opening_seed` -- use `stage.id` cast to `u32` for the seed derivation (the seed must be stable across reorderings; `stage.id` is the domain identifier, `stage.index` is the array position)
- Do NOT generate noise in stage-major order (stage outer, opening inner) -- the spec says opening-major (opening outer, stage inner) for parallel generation alignment, though for single-threaded generation both produce identical results. Follow the spec for consistency.
- Do NOT forget to handle `Option<i64>` for the base seed -- if `ScenarioSource.seed` is `None`, use OS entropy or document that the caller must provide a seed
- The `stage.id` may be negative for pre-study stages -- do NOT include pre-study stages in the opening tree (they have no scenario configuration)

## Testing Requirements

### Unit Tests

- Determinism: same inputs produce same tree (compare all f64 values)
- Seed sensitivity: different base seeds produce different trees
- Correct dimensions: `n_stages`, `n_openings`, `dim`, `len` match expected values
- Variable branching: stages with different branching factors
- Identity correlation: independent noise (no correlation) -- verify each noise vector has reasonable N(0,1) statistics (mean near 0, std near 1) over many openings
- Non-identity correlation: apply a known 2x2 correlation, verify the sample correlation of generated noise matches the target within statistical tolerance

### Integration Tests

None for this ticket (end-to-end integration is Epic 04).

### E2E Tests

Not applicable.

## Dependencies

- **Blocked By**: ticket-004-implement-siphash-seed-derivation.md, ticket-005-implement-cholesky-and-correlation.md, ticket-006-implement-opening-tree-types.md
- **Blocks**: Epic 03 tickets (InSample sampling reads from OpeningTree)

## Effort Estimate

**Points**: 3
**Confidence**: High
