# ticket-006 Implement OpeningTree and OpeningTreeView types

## Context

### Background

The backward pass in iterative optimization evaluates all N_t noise realizations (openings) at each stage. These openings are generated once before the optimization loop and remain fixed throughout. The `OpeningTree` struct holds the pre-generated noise values in a flat contiguous array with stage-major layout for cache-friendly backward pass access. `OpeningTreeView` is a borrowed view for read-only consumption by downstream crates.

### Relation to Epic

This ticket defines the data types; ticket-007 fills them with actual generated noise values. The types are defined separately to enable testing of the access API independently from the generation logic.

### Current State

- `crates/cobre-stochastic/src/tree/opening_tree.rs` is a placeholder
- The spec in `scenario-generation.md` SS2.3a defines the exact struct layout and access API

## Specification

### Requirements

1. Define `OpeningTree` in `crates/cobre-stochastic/src/tree/opening_tree.rs`:

```rust
/// Fixed opening tree holding pre-generated noise realisations for
/// the backward pass of iterative optimisation algorithms.
///
/// All noise values are stored in a flat contiguous array with
/// stage-major ordering: all openings for stage 0, then all openings
/// for stage 1, etc. Within each stage, openings are contiguous blocks
/// of `dim` f64 values.
///
/// Access pattern: `data[stage_offsets[stage] + opening_idx * dim .. + dim]`
///
/// See [Scenario Generation SS2.3a](scenario-generation.md) for the
/// full type specification and memory layout rationale.
pub struct OpeningTree {
    data: Box<[f64]>,
    stage_offsets: Box<[usize]>,
    openings_per_stage: Box<[usize]>,
    n_stages: usize,
    dim: usize,
}
```

2. Implement the full public API per the spec:
   - `pub fn opening(&self, stage: usize, opening_idx: usize) -> &[f64]` -- returns slice of length `dim`; panics if out of bounds
   - `pub fn n_openings(&self, stage: usize) -> usize` -- branching factor at given stage
   - `pub fn n_stages(&self) -> usize`
   - `pub fn dim(&self) -> usize`
   - `pub fn len(&self) -> usize` -- total f64 elements
   - `pub fn is_empty(&self) -> bool` -- required by clippy when `len()` exists
   - `pub fn size_bytes(&self) -> usize` -- total bytes of backing array

3. Implement a constructor for testing and generation:
   - `pub(crate) fn from_parts(data: Vec<f64>, openings_per_stage: Vec<usize>, dim: usize) -> Self` -- computes `stage_offsets` from `openings_per_stage` and `dim`; validates `data.len() == sum(openings_per_stage[t] * dim)`

4. Define `OpeningTreeView<'a>`:

```rust
/// Borrowed read-only view over opening tree data.
///
/// Provides the same access API as [`OpeningTree`] but borrows
/// the underlying storage. Used by downstream crates that consume
/// the tree without owning it.
pub struct OpeningTreeView<'a> {
    data: &'a [f64],
    stage_offsets: &'a [usize],
    openings_per_stage: &'a [usize],
    n_stages: usize,
    dim: usize,
}
```

5. Implement `OpeningTree::view(&self) -> OpeningTreeView<'_>` to create a borrowed view.
6. Implement the same accessor methods on `OpeningTreeView<'a>`.

### Inputs/Props

- `data: Vec<f64>` -- flat noise values
- `openings_per_stage: Vec<usize>` -- branching factors per stage
- `dim: usize` -- number of entities per noise vector

### Outputs/Behavior

- `opening(stage, idx)` returns a contiguous `&[f64]` slice of exactly `dim` elements
- Uniform branching is a degenerate case where all `openings_per_stage` entries are equal

### Error Handling

- `from_parts` panics (via `assert!`) if `data.len()` does not match the expected total
- Accessor methods panic on out-of-bounds access (via `assert!`)

## Acceptance Criteria

- [ ] Given `OpeningTree::from_parts(vec![1.0,2.0,3.0,4.0,5.0,6.0], vec![1,2], 3)`, when `opening(0, 0)` is called, then it returns `&[1.0, 2.0, 3.0]`
- [ ] Given the same tree, when `opening(1, 0)` is called, then it returns `&[4.0, 5.0, 6.0]` -- NOT `&[1.0, 2.0, 3.0]`... wait, stage 1 has 2 openings of dim 3, so data needs 1*3 + 2*3 = 9 elements. Let me recalculate: `from_parts(vec![1.,2.,3., 4.,5.,6., 7.,8.,9.], vec![1,2], 3)`. Then `opening(0,0) = [1,2,3]`, `opening(1,0) = [4,5,6]`, `opening(1,1) = [7,8,9]`.
- [ ] Given `OpeningTree::from_parts(data, vec![1,2], 3)` with 9 elements, when `n_openings(0)` is called, then it returns 1; when `n_openings(1)` is called, then it returns 2
- [ ] Given an `OpeningTree`, when `view()` is called and `opening(0, 0)` is called on the view, then the returned slice is identical to calling `opening(0, 0)` on the owned tree
- [ ] Given `OpeningTree::from_parts(vec![1.0; 30], vec![5,5,5], 2)`, when `len()` is called, then it returns 30; when `size_bytes()` is called, then it returns 240

## Implementation Guide

### Suggested Approach

1. Define `OpeningTree` struct with the five fields
2. Implement `from_parts`:
   - Compute `n_stages = openings_per_stage.len()`
   - Compute `stage_offsets`: `[0, ops[0]*dim, ops[0]*dim + ops[1]*dim, ...]` with sentinel
   - Assert `data.len() == stage_offsets[n_stages]`
   - Convert Vec to Box<[T]> via `into_boxed_slice()`
3. Implement accessor methods with `assert!(stage < self.n_stages)` and `assert!(opening_idx < self.openings_per_stage[stage])`
4. Define `OpeningTreeView` with identical accessors but borrowing from slices
5. Implement `view()` on `OpeningTree`
6. Add unit tests

### Key Files to Modify

- `crates/cobre-stochastic/src/tree/opening_tree.rs` (primary implementation)
- `crates/cobre-stochastic/src/tree/mod.rs` (re-exports)
- `crates/cobre-stochastic/src/lib.rs` (public re-exports)

### Patterns to Follow

- `Box<[f64]>` for immutable arrays (spec requirement -- communicates no-resize invariant)
- Offset-based access: `&self.data[offset..offset + self.dim]`
- `pub(crate)` for `from_parts` (only the generator creates trees; external callers use the generation API)

### Pitfalls to Avoid

- Do NOT store `stage_offsets` with length `n_stages` -- it must be `n_stages + 1` to include the sentinel value `data.len()`
- Do NOT assume uniform branching in the accessor -- always use `openings_per_stage[stage]` for bounds checking
- Do NOT implement `Index` trait -- explicit `opening()` method is clearer and matches the spec
- Clippy requires `is_empty()` when `len()` is present -- add it

## Testing Requirements

### Unit Tests

- Uniform branching: 3 stages x 5 openings x 2 entities (30 elements)
- Variable branching: stages with different opening counts
- Single stage, single opening (degenerate case)
- `from_parts` panics on incorrect data length
- `opening()` panics on out-of-bounds stage
- `opening()` panics on out-of-bounds opening_idx
- `view()` returns identical data to owned accessors
- `len()` and `size_bytes()` correctness

### Integration Tests

None for this ticket.

### E2E Tests

Not applicable.

## Dependencies

- **Blocked By**: ticket-001-scaffold-crate-structure.md
- **Blocks**: ticket-007-implement-opening-tree-generation.md

## Effort Estimate

**Points**: 2
**Confidence**: High
