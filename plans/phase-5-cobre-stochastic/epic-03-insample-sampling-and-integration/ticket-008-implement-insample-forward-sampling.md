# ticket-008 Implement InSample forward sampling

> **[OUTLINE]** This ticket requires refinement before execution.
> It will be refined with learnings from earlier epics.

## Objective

Implement the InSample forward sampling logic in `crates/cobre-stochastic/src/sampling/insample.rs`. InSample sampling selects a random index j in {0, ..., N_openings-1} at each stage and returns the corresponding pre-generated noise vector from the OpeningTree. The RNG for index selection uses SipHash-1-3 seed derivation with the forward-pass wire format (base_seed, iteration, scenario, stage) to ensure deterministic, communication-free reproducibility across MPI ranks. This is the minimal-viable sampling scheme required for the first solver vertical.

## Anticipated Scope

- **Files likely to be modified**:
  - `crates/cobre-stochastic/src/sampling/insample.rs` (primary implementation)
  - `crates/cobre-stochastic/src/sampling/mod.rs` (re-exports)
  - `crates/cobre-stochastic/src/lib.rs` (public re-exports)
- **Key decisions needed**:
  - Whether `sample_forward` returns a borrowed `&[f64]` slice from the OpeningTree or copies into a `NoiseVector` owned type (depends on OpeningTree lifetime patterns established in Epic 02)
  - Whether `NoiseVector` is a newtype wrapper or a plain `&[f64]` (depends on what downstream consumers need, which becomes clear when Epic 02 patterns are established)
  - Whether the sampling function takes an `OpeningTreeView<'_>` or `&OpeningTree` (depends on view API usability discovered in ticket-006)
- **Open questions**:
  - What is the exact signature of the index selection RNG? Does it reuse `rng_from_seed` + `derive_forward_seed`, or does the calling algorithm provide an already-initialized RNG?
  - Should the function also return the selected index j (for diagnostics/logging)?

## Dependencies

- **Blocked By**: ticket-004-implement-siphash-seed-derivation.md, ticket-006-implement-opening-tree-types.md
- **Blocks**: ticket-010-organize-public-api.md

## Effort Estimate

**Points**: 2
**Confidence**: Low (will be re-estimated during refinement)
