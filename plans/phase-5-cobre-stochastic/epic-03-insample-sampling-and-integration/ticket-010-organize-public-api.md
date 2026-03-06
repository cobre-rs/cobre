# ticket-010 Organize public API and re-exports

> **[OUTLINE]** This ticket requires refinement before execution.
> It will be refined with learnings from earlier epics.

## Objective

Finalize the public API surface of `cobre-stochastic` by organizing re-exports in `lib.rs`, ensuring that downstream crates (primarily `cobre-sddp`) can access all necessary types through a clean, minimal public interface. This includes auditing all `pub` visibility modifiers, ensuring `pub(crate)` is used for internal-only items, and verifying that the API follows the same patterns established in `cobre-core`, `cobre-solver`, and `cobre-comm`.

## Anticipated Scope

- **Files likely to be modified**:
  - `crates/cobre-stochastic/src/lib.rs` (re-export organization)
  - Various `mod.rs` files across submodules (visibility adjustments)
- **Key decisions needed**:
  - Which types should be in the public API vs `pub(crate)` (depends on what downstream consumers actually need, which becomes clearer after Epic 02 integration patterns)
  - Whether to use a flat re-export style (`pub use crate::tree::OpeningTree`) or module-path access (`cobre_stochastic::tree::OpeningTree`)
- **Open questions**:
  - Should `CholeskyFactor` be public or `pub(crate)`? It is an implementation detail of correlation, but advanced users might want to inspect decomposition results.
  - Should `GroupFactor` be public? Same question as above.
  - Are there any types that should implement additional standard traits (Clone, PartialEq, etc.) for downstream ergonomics?

## Dependencies

- **Blocked By**: ticket-008-implement-insample-forward-sampling.md, ticket-009-implement-stochastic-context.md
- **Blocks**: Epic 04 tickets (conformance tests import from the public API)

## Effort Estimate

**Points**: 2
**Confidence**: Low (will be re-estimated during refinement)
