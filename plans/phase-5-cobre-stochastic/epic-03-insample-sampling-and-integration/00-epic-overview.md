# Epic 03: InSample Sampling and Integration

## Goal

Implement the InSample forward sampling scheme and integrate all cobre-stochastic components into a cohesive public API. The InSample scheme samples a random index into the opening tree at each stage, returning the pre-generated noise vector. This epic also provides the top-level initialization function that wires PAR preprocessing, correlation decomposition, and opening tree generation into a single entry point.

## Scope

- InSample forward sampling: random index selection into opening tree per stage
- `NoiseVector` type for returning noise realisations to the calling algorithm
- Top-level `StochasticContext` struct that bundles `PrecomputedParLp`, `DecomposedCorrelation`, and `OpeningTree`
- Public API surface: `build_stochastic_context()` initialization function
- Re-export organization in `lib.rs`

## Out of Scope

- External and Historical sampling schemes (deferred variants)
- LHS, QMC, Selective noise methods (deferred)
- SharedRegion allocation for opening tree (cobre-sddp integration concern)
- Forward pass noise generation (InSample reads from the opening tree, does not generate new noise)

## Tickets

| Ticket | Title | Points | Blocks |
|--------|-------|--------|--------|
| ticket-008 | Implement InSample forward sampling | 2 | ticket-010 |
| ticket-009 | Implement StochasticContext initialization | 3 | ticket-010 |
| ticket-010 | Organize public API and re-exports | 2 | Epic 04 |

## Dependencies

- **Blocked By**: Epic 01 (PrecomputedParLp), Epic 02 (OpeningTree, DecomposedCorrelation, seed derivation)
- **Blocks**: Epic 04 (conformance tests exercise the public API)

## Success Criteria

- InSample sampling returns a noise vector of length `dim` for any valid (stage, opening_index) pair
- `build_stochastic_context()` constructs all stochastic infrastructure from `System` + `ScenarioData`
- Public API is minimal and well-documented with zero algorithm-specific references
