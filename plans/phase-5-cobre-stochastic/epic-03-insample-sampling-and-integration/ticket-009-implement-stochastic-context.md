# ticket-009 Implement StochasticContext initialization

> **[OUTLINE]** This ticket requires refinement before execution.
> It will be refined with learnings from earlier epics.

## Objective

Implement a `StochasticContext` struct that bundles `PrecomputedParLp`, `DecomposedCorrelation`, and `OpeningTree` into a single initialization entry point. The `build_stochastic_context()` function takes `&System` and `&ScenarioData` from cobre-io, runs the full preprocessing pipeline (PAR preprocessing, correlation decomposition, opening tree generation), and returns the assembled context. This is the primary public API surface for downstream crates that need stochastic infrastructure.

## Anticipated Scope

- **Files likely to be modified**:
  - `crates/cobre-stochastic/src/lib.rs` (StochasticContext struct and build function, or a dedicated `context.rs` module)
  - `crates/cobre-stochastic/src/par/precompute.rs` (may need minor API adjustments for integration)
  - `crates/cobre-stochastic/src/correlation/resolve.rs` (may need minor API adjustments)
  - `crates/cobre-stochastic/src/tree/generate.rs` (may need minor API adjustments)
  - `crates/cobre-stochastic/Cargo.toml` (may need cobre-io dependency if ScenarioData is used directly)
- **Key decisions needed**:
  - Whether `StochasticContext` owns the OpeningTree or holds a reference (depends on SharedRegion allocation patterns that will be determined during cobre-sddp integration)
  - Whether `build_stochastic_context` takes `&System` + `&ScenarioData` directly or takes decomposed parameters (depends on how tightly coupled the API should be to cobre-io types)
  - Whether to add cobre-io as a dependency or keep the interface at the cobre-core type level (affects crate layering)
  - Whether PAR validation is called inside `build_stochastic_context` or left to the caller (depends on whether validation should be mandatory or optional)
- **Open questions**:
  - How does the calling algorithm extract the base seed from `ScenarioSource.seed: Option<i64>`? Should `build_stochastic_context` handle the None case (OS entropy) or require a seed?
  - Should the context be `Send + Sync` for multi-threaded use within a rank?

## Dependencies

- **Blocked By**: ticket-002-implement-precomputed-par-lp.md, ticket-005-implement-cholesky-and-correlation.md, ticket-007-implement-opening-tree-generation.md
- **Blocks**: ticket-010-organize-public-api.md

## Effort Estimate

**Points**: 3
**Confidence**: Low (will be re-estimated during refinement)
